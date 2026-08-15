# SPDX-License-Identifier: Apache-2.0
"""vLLM DiffKV attention backend with an HPC-Ops GQLA decode fast path.

The backend deliberately keeps the public name ``FLASH_ATTN_DIFFKV``.  vLLM
uses that name to select an internal enum, while the subclass gives us a
different implementation class.  Cache allocation, cache updates, metadata
construction, prefill, and mixed batches therefore stay on vLLM's tested
FlashAttentionDiffKV path.  Only uniform one- or two-token decode batches that
match the HPC-Ops GQLA kernel contract are intercepted.

Runtime controls:

``GQLA_HPC_TRACE=1``
    Print one ``GQLA_HPC_TRACE_FALLBACK reason=...`` line per *eligible decode
    failure*.  Expected profiling, prefill, and mixed-batch fallbacks are
    silent.  The first successful kernel invocation prints
    ``GQLA_HPC_TRACE_HIT``, so benchmark harnesses can prove that they measured
    HPC-Ops.

``GQLA_HPC_STRICT=1``
    For a uniform decode batch (query length one or two), fail if the HPC-Ops
    contract is not met, the extension cannot be imported, or the kernel call
    fails.  Profiling, prefill, and mixed batches still fall back normally.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)


_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_HPC_ATTENTION_FN: Callable[..., torch.Tensor] | None = None
_HPC_IMPORT_ERROR: Exception | None = None
_TRACE_HIT_PRINTED = False
_TRACE_FALLBACK_REASONS: set[str] = set()


def _env_enabled(name: str) -> bool:
    """Return a predictable bool for a user-facing environment switch."""

    return os.getenv(name, "").strip().lower() in _TRUE_ENV_VALUES


def _uniform_decode_shape(
    attn_metadata: FlashAttentionMetadata | Any | None,
) -> tuple[int, int] | None:
    """Return ``(batch, query_len)`` for a uniform qlen=1/2 decode batch.

    This intentionally uses tensor *shapes* and scalar metadata only; it does
    not copy ``query_start_loc`` to the host or introduce a CUDA sync.  If every
    sequence has a positive query length no greater than ``max_query_len``, the
    equality ``num_tokens == batch * max_query_len`` proves uniformity.
    """

    if attn_metadata is None:
        return None
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if query_start_loc is None or query_start_loc.ndim != 1:
        return None
    batch = int(query_start_loc.shape[0]) - 1
    query_len = int(getattr(attn_metadata, "max_query_len", 0))
    num_tokens = int(getattr(attn_metadata, "num_actual_tokens", -1))
    if batch <= 0 or query_len not in (1, 2):
        return None
    if num_tokens != batch * query_len:
        return None
    return batch, query_len


def _trace_fallback(reason: str, enabled: bool) -> None:
    """Emit bounded, grep-stable fallback diagnostics."""

    if not enabled or reason in _TRACE_FALLBACK_REASONS:
        return
    _TRACE_FALLBACK_REASONS.add(reason)
    print(f"GQLA_HPC_TRACE_FALLBACK reason={reason}", flush=True)


def _load_hpc_attention() -> Callable[..., torch.Tensor]:
    """Import HPC-Ops on the first eligible decode, never during model import."""

    global _HPC_ATTENTION_FN, _HPC_IMPORT_ERROR
    if _HPC_ATTENTION_FN is not None:
        return _HPC_ATTENTION_FN
    if _HPC_IMPORT_ERROR is not None:
        raise RuntimeError("cached HPC-Ops import failure") from _HPC_IMPORT_ERROR
    try:
        hpc = importlib.import_module("hpc")
        fn = getattr(hpc, "attention_decode_bf16")
    except Exception as exc:
        _HPC_IMPORT_ERROR = exc
        raise RuntimeError("cannot import hpc.attention_decode_bf16") from exc
    _HPC_ATTENTION_FN = fn
    return fn


class GQLAHPCFlashAttentionDiffKVBackend(FlashAttentionDiffKVBackend):
    """DiffKV backend with a fixed QK=192, V=128 GQLA cache contract."""

    head_size_v = 128

    @staticmethod
    def get_name() -> str:
        # Attention.__init__ maps this through AttentionBackendEnum.
        return "FLASH_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["GQLAHPCFlashAttentionDiffKVImpl"]:
        return GQLAHPCFlashAttentionDiffKVImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # HPC-Ops' GQLA decode kernel has a fixed page size.  The FA fallback
        # also supports 64, so one cache layout serves both paths.
        return [64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        del default_block_size
        return 64

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, block_size, num_kv_heads, head_size + 128)


class GQLAHPCFlashAttentionDiffKVImpl(FlashAttentionDiffKVImpl):
    """Use HPC-Ops for eligible decode and parent DiffKV for everything else."""

    def __init__(self, *args, **kwargs) -> None:
        # The vLLM 0.22 DiffKV implementation consults the base backend's
        # class-level V dimension while choosing its FlashAttention version.
        FlashAttentionDiffKVBackend.set_head_size_v(128)
        super().__init__(*args, **kwargs)
        self._gqla_hpc_strict = _env_enabled("GQLA_HPC_STRICT")
        self._gqla_hpc_trace = _env_enabled("GQLA_HPC_TRACE")

    def _contract_failure(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> str | None:
        """Return the first reason an otherwise-uniform decode cannot use HPC."""

        if self.attn_type != AttentionType.DECODER:
            return "attention_type"
        if output_scale is not None or output_block_scale is not None:
            return "fused_output_quantization"
        if attn_metadata.use_cascade:
            return "cascade"
        if self.dcp_world_size != 1 or self.pcp_world_size != 1:
            return "context_parallel"
        if not bool(getattr(attn_metadata, "causal", True)):
            return "non_causal"
        if self.alibi_slopes is not None:
            return "alibi"
        if self.sliding_window != (-1, -1):
            return "sliding_window"
        if self.logits_soft_cap not in (None, 0, 0.0):
            return "logits_soft_cap"
        if self.sinks is not None:
            return "attention_sinks"
        if self.head_size != 192:
            return f"qk_dim_{self.head_size}"
        if GQLAHPCFlashAttentionDiffKVBackend.head_size_v != 128:
            return "v_dim"
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads != 0:
            return "head_divisibility"
        head_ratio = self.num_heads // self.num_kv_heads
        if head_ratio not in (16, 32):
            return f"head_ratio_{head_ratio}"
        if query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            return "activation_dtype"
        if kv_cache.dtype != torch.bfloat16:
            return "kv_cache_dtype"
        if not query.is_cuda or not kv_cache.is_cuda or not output.is_cuda:
            return "device"
        capability = torch.cuda.get_device_capability(query.device)
        if capability != (9, 0):
            return f"sm_{capability[0]}{capability[1]}"
        if kv_cache.ndim != 4:
            return "kv_cache_rank"
        if tuple(kv_cache.shape[-2:]) != (self.num_kv_heads, 192 + 128):
            return "kv_cache_shape"
        if int(kv_cache.shape[1]) != 64:
            return f"page_size_{int(kv_cache.shape[1])}"
        if kv_cache.stride(-1) != 1:
            return "kv_cache_last_stride"

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]
        if not query.is_contiguous():
            return "query_layout"
        if not output.is_contiguous():
            return "output_layout"

        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        if block_table.dtype != torch.int32 or not block_table.is_cuda:
            return "block_table_dtype_or_device"
        if block_table.ndim != 2 or not block_table.is_contiguous():
            return "block_table_layout"
        if seq_lens.dtype != torch.int32 or not seq_lens.is_cuda:
            return "seq_lens_dtype_or_device"
        if seq_lens.ndim != 1 or not seq_lens.is_contiguous():
            return "seq_lens_layout"
        return None

    def _fallback_or_raise(
        self,
        reason: str,
        *,
        decode_candidate: bool,
        cause: Exception | None = None,
    ) -> None:
        _trace_fallback(reason, self._gqla_hpc_trace)
        if self._gqla_hpc_strict and decode_candidate:
            message = f"GQLA HPC strict decode failure: {reason}"
            if cause is None:
                raise RuntimeError(message)
            raise RuntimeError(message) from cause

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        decode_shape = _uniform_decode_shape(attn_metadata)
        if decode_shape is None:
            # Profiling, prefill, and mixed batches are expected to use the
            # parent implementation.  Keep them silent even with tracing on so
            # benchmark harnesses can interpret FALLBACK as a failed decode
            # fast-path attempt rather than ordinary control flow.
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        batch, query_len = decode_shape
        reason = self._contract_failure(
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        if reason is not None:
            self._fallback_or_raise(reason, decode_candidate=True)
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        try:
            hpc_attention = _load_hpc_attention()
        except Exception as exc:
            self._fallback_or_raise(
                "hpc_import", decode_candidate=True, cause=exc
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache = kv_cache[..., : self.head_size]
        value_cache = kv_cache[..., self.head_size :]
        output_view = output[:num_actual_tokens]
        try:
            hpc_attention(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                mtp=query_len - 1,
                new_kv_included=True,
                splitk=False,
                task_map=None,
                split_flag=None,
                output=output_view,
                softmax_scale=float(self.scale),
            )
        except Exception as exc:
            self._fallback_or_raise(
                "hpc_call", decode_candidate=True, cause=exc
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        global _TRACE_HIT_PRINTED
        if self._gqla_hpc_trace and not _TRACE_HIT_PRINTED:
            _TRACE_HIT_PRINTED = True
            print(
                "GQLA_HPC_TRACE_HIT "
                f"batch={batch} qlen={query_len} "
                f"hq={self.num_heads} hkv={self.num_kv_heads} "
                f"softmax_scale={float(self.scale):.9g}",
                flush=True,
            )
        return output


__all__ = [
    "GQLAHPCFlashAttentionDiffKVBackend",
    "GQLAHPCFlashAttentionDiffKVImpl",
    "_env_enabled",
    "_uniform_decode_shape",
]
