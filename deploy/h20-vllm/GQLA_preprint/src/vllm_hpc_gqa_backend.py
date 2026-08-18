# SPDX-License-Identifier: Apache-2.0
"""vLLM DiffKV attention backend with an HPC-Ops GQLA decode fast path.

The backend deliberately keeps the public name ``FLASH_ATTN_DIFFKV``.  vLLM
uses that name to select an internal enum, while the subclass gives us a
different implementation class. Cache allocation and cache updates stay on
vLLM's tested FlashAttentionDiffKV path. Pure prefill batches use DiffKV, pure
decode batches use HPC-Ops, and mixed batches are split into a decode prefix
and prefill suffix so every decode token still uses HPC-Ops.

Runtime controls:

``GQLA_HPC_TRACE=1``
    Print one ``GQLA_HPC_TRACE_FALLBACK reason=...`` line per decode failure.
    The first mixed-batch split prints ``GQLA_HPC_TRACE_MIXED_SPLIT`` and the
    first successful kernel invocation prints
    ``GQLA_HPC_TRACE_HIT``, so benchmark harnesses can prove that they measured
    HPC-Ops.

``GQLA_HPC_STRICT=1``
    Guarantee the all-decode contract. Any decode-layout, extension-import, or
    kernel-call failure aborts instead of falling back to DiffKV. Pure prefill
    remains on DiffKV by design.

Every eligible decode call enables HPC-Ops' adaptive static split-K policy.
The kernel chooses its concrete split count from the rank-local batch, KV-head
count, context capacity, and SM count.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)


_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_HPC_ATTENTION_FN: Callable[..., torch.Tensor] | None = None
_HPC_IMPORT_ERROR: Exception | None = None
_TRACE_HIT_PRINTED = False
_TRACE_MIXED_SPLIT_PRINTED = False
_TRACE_STRICT_ENABLED_PRINTED = False
_TRACE_FALLBACK_REASONS: set[str] = set()


@dataclass(frozen=True)
class _HPCBatchSplit:
    """Contiguous request/token boundary between decode and prefill."""

    num_decode_reqs: int
    num_decode_tokens: int
    decode_query_len: int
    num_prefill_reqs: int
    num_prefill_tokens: int


@dataclass
class GQLAHPCFlashAttentionMetadata(FlashAttentionMetadata):
    """Full-batch metadata plus decode/prefill views for mixed execution."""

    num_decode_reqs: int = 0
    num_decode_tokens: int = 0
    decode_query_len: int = 0
    decode_metadata: FlashAttentionMetadata | None = None
    prefill_metadata: FlashAttentionMetadata | None = None


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


def _analyze_hpc_batch(
    common_attn_metadata: CommonAttentionMetadata | Any,
) -> _HPCBatchSplit:
    """Find a uniform decode prefix without misclassifying short prefills.

    The custom metadata builder advertises ``reorder_batch_threshold=2``, so
    vLLM orders one-/two-token requests before long prefills. ``is_prefilling``
    then distinguishes a real decode from a short final prefill chunk. The HPC
    ABI accepts one uniform query length per call; mixed qlen=1/2 speculative
    decode must therefore fail rather than silently use DiffKV.
    """

    num_reqs = int(common_attn_metadata.num_reqs)
    num_tokens = int(common_attn_metadata.num_actual_tokens)
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
    query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    is_prefilling = getattr(common_attn_metadata, "is_prefilling", None)
    if is_prefilling is None:
        raise ValueError("missing_is_prefilling")
    if is_prefilling.device.type != "cpu":
        raise ValueError("is_prefilling_not_cpu")
    is_prefilling = is_prefilling[:num_reqs].to(dtype=torch.bool)

    # Ignore zero-length CUDA-graph padding rows. Non-prefill active rows are
    # actual decode requests.
    is_decode = (query_lens > 0) & ~is_prefilling
    decode_indices = torch.nonzero(is_decode, as_tuple=False).flatten()
    if decode_indices.numel() == 0:
        return _HPCBatchSplit(0, 0, 0, num_reqs, num_tokens)

    num_decode_reqs = int(decode_indices.numel())
    expected_prefix = torch.arange(
        num_decode_reqs,
        dtype=decode_indices.dtype,
        device=decode_indices.device,
    )
    if not torch.equal(decode_indices, expected_prefix):
        raise ValueError("decode_not_contiguous_prefix")

    decode_query_lens = query_lens[:num_decode_reqs]
    decode_query_len = int(decode_query_lens[0].item())
    if decode_query_len not in (1, 2):
        raise ValueError(f"unsupported_decode_query_len_{decode_query_len}")
    if not torch.all(decode_query_lens == decode_query_len):
        unique = ",".join(
            str(int(value)) for value in torch.unique(decode_query_lens).tolist()
        )
        raise ValueError(f"nonuniform_decode_query_lens_{unique}")

    num_decode_tokens = int(query_start_loc_cpu[num_decode_reqs].item())
    if num_decode_tokens != num_decode_reqs * decode_query_len:
        raise ValueError("decode_token_accounting")
    return _HPCBatchSplit(
        num_decode_reqs=num_decode_reqs,
        num_decode_tokens=num_decode_tokens,
        decode_query_len=decode_query_len,
        num_prefill_reqs=num_reqs - num_decode_reqs,
        num_prefill_tokens=num_tokens - num_decode_tokens,
    )


def _slice_common_metadata(
    metadata: CommonAttentionMetadata,
    *,
    req_start: int,
    req_end: int,
    token_start: int,
    token_end: int,
) -> CommonAttentionMetadata:
    """Return a request/token-contiguous CommonAttentionMetadata view."""

    query_start_loc_cpu = (
        metadata.query_start_loc_cpu[req_start : req_end + 1] - token_start
    )
    query_start_loc = metadata.query_start_loc[req_start : req_end + 1]
    if token_start:
        query_start_loc = query_start_loc - token_start
    query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    max_query_len = int(query_lens.max().item()) if query_lens.numel() else 0

    def req_slice(value):
        return value[req_start:req_end] if value is not None else None

    seq_lens_cpu_upper_bound = req_slice(metadata.seq_lens_cpu_upper_bound)
    if seq_lens_cpu_upper_bound is not None and seq_lens_cpu_upper_bound.numel():
        max_seq_len = int(seq_lens_cpu_upper_bound.max().item())
    else:
        max_seq_len = metadata.max_seq_len

    return metadata.replace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=metadata.seq_lens[req_start:req_end],
        num_reqs=req_end - req_start,
        num_actual_tokens=token_end - token_start,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=metadata.block_table_tensor[req_start:req_end],
        slot_mapping=metadata.slot_mapping[token_start:token_end],
        positions=(
            metadata.positions[token_start:token_end]
            if metadata.positions is not None
            else None
        ),
        is_prefilling=req_slice(metadata.is_prefilling),
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=req_slice(metadata.dcp_local_seq_lens),
        dcp_local_seq_lens_cpu=req_slice(metadata.dcp_local_seq_lens_cpu),
        encoder_seq_lens=req_slice(metadata.encoder_seq_lens),
        encoder_seq_lens_cpu=req_slice(metadata.encoder_seq_lens_cpu),
        _seq_lens_cpu=req_slice(metadata._seq_lens_cpu),
        _num_computed_tokens_cpu=req_slice(metadata._num_computed_tokens_cpu),
        _num_computed_tokens_cache=None,
    )


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
    def get_builder_cls() -> type["GQLAHPCFlashAttentionMetadataBuilder"]:
        return GQLAHPCFlashAttentionMetadataBuilder

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


class GQLAHPCFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Build split metadata so mixed-batch decode never falls back to FA."""

    # Pure uniform qlen=1/2 decode remains CUDA-graph compatible. Mixed batches
    # use piecewise/eager attention where their dynamic split is explicit.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        global _TRACE_STRICT_ENABLED_PRINTED
        if _env_enabled("GQLA_HPC_STRICT") and not _TRACE_STRICT_ENABLED_PRINTED:
            _TRACE_STRICT_ENABLED_PRINTED = True
            print(
                "GQLA_HPC_ALL_DECODE_STRICT_ENABLED "
                "mixed_batch_split=1 decode_query_lens=1,2",
                flush=True,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GQLAHPCFlashAttentionMetadata:
        try:
            split = _analyze_hpc_batch(common_attn_metadata)
        except ValueError as exc:
            reason = f"batch_layout_{exc}"
            _trace_fallback(reason, _env_enabled("GQLA_HPC_TRACE"))
            if _env_enabled("GQLA_HPC_STRICT"):
                raise RuntimeError(
                    f"GQLA HPC all-decode metadata failure: {exc}"
                ) from exc
            split = _HPCBatchSplit(
                0,
                0,
                0,
                int(common_attn_metadata.num_reqs),
                int(common_attn_metadata.num_actual_tokens),
            )

        decode_metadata = None
        prefill_metadata = None
        if split.num_decode_tokens and split.num_prefill_tokens:
            decode_common = _slice_common_metadata(
                common_attn_metadata,
                req_start=0,
                req_end=split.num_decode_reqs,
                token_start=0,
                token_end=split.num_decode_tokens,
            )
            prefill_common = _slice_common_metadata(
                common_attn_metadata,
                req_start=split.num_decode_reqs,
                req_end=int(common_attn_metadata.num_reqs),
                token_start=split.num_decode_tokens,
                token_end=int(common_attn_metadata.num_actual_tokens),
            )
            # Split subcalls do not use FA AOT scheduler metadata. The complete
            # metadata is built last so a non-strict full-batch fallback keeps
            # a valid parent scheduler buffer.
            decode_metadata = super().build(0, decode_common, fast_build=True)
            prefill_metadata = super().build(0, prefill_common, fast_build=True)

            global _TRACE_MIXED_SPLIT_PRINTED
            if _env_enabled("GQLA_HPC_TRACE") and not _TRACE_MIXED_SPLIT_PRINTED:
                _TRACE_MIXED_SPLIT_PRINTED = True
                print(
                    "GQLA_HPC_TRACE_MIXED_SPLIT "
                    f"decode_reqs={split.num_decode_reqs} "
                    f"decode_tokens={split.num_decode_tokens} "
                    f"decode_qlen={split.decode_query_len} "
                    f"prefill_reqs={split.num_prefill_reqs} "
                    f"prefill_tokens={split.num_prefill_tokens}",
                    flush=True,
                )

        full_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        return GQLAHPCFlashAttentionMetadata(
            **vars(full_metadata),
            num_decode_reqs=split.num_decode_reqs,
            num_decode_tokens=split.num_decode_tokens,
            decode_query_len=split.decode_query_len,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
        )

    def update_block_table(
        self,
        metadata: GQLAHPCFlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GQLAHPCFlashAttentionMetadata:
        updated = super().update_block_table(metadata, blk_table, slot_mapping)
        assert isinstance(updated, GQLAHPCFlashAttentionMetadata)
        if metadata.decode_metadata is not None:
            updated.decode_metadata = super().update_block_table(
                metadata.decode_metadata,
                blk_table[: metadata.num_decode_reqs],
                slot_mapping[: metadata.num_decode_tokens],
            )
        if metadata.prefill_metadata is not None:
            updated.prefill_metadata = super().update_block_table(
                metadata.prefill_metadata,
                blk_table[metadata.num_decode_reqs :],
                slot_mapping[metadata.num_decode_tokens :],
            )
        return updated


class GQLAHPCFlashAttentionDiffKVImpl(FlashAttentionDiffKVImpl):
    """Use HPC-Ops for every decode token and DiffKV only for prefill."""

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

    def _run_hpc_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        *,
        batch: int,
        query_len: int,
        mixed: bool,
    ) -> bool:
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
            return False

        try:
            hpc_attention = _load_hpc_attention()
        except Exception as exc:
            self._fallback_or_raise("hpc_import", decode_candidate=True, cause=exc)
            return False

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
                splitk=True,
                task_map=None,
                split_flag=None,
                output=output_view,
                softmax_scale=float(self.scale),
            )
        except Exception as exc:
            self._fallback_or_raise("hpc_call", decode_candidate=True, cause=exc)
            return False

        global _TRACE_HIT_PRINTED
        if self._gqla_hpc_trace and not _TRACE_HIT_PRINTED:
            _TRACE_HIT_PRINTED = True
            print(
                "GQLA_HPC_TRACE_HIT "
                f"mode={'mixed' if mixed else 'uniform'} "
                f"batch={batch} qlen={query_len} "
                f"hq={self.num_heads} hkv={self.num_kv_heads} "
                "splitk=adaptive_static "
                f"softmax_scale={float(self.scale):.9g}",
                flush=True,
            )
        return True

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
        if isinstance(attn_metadata, GQLAHPCFlashAttentionMetadata):
            num_decode_tokens = attn_metadata.num_decode_tokens
            if num_decode_tokens == 0:
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

            mixed = attn_metadata.prefill_metadata is not None
            decode_metadata = attn_metadata.decode_metadata or attn_metadata
            decode_output = output[:num_decode_tokens]
            used_hpc = self._run_hpc_decode(
                query[:num_decode_tokens],
                kv_cache,
                decode_metadata,
                decode_output,
                output_scale,
                output_block_scale,
                batch=attn_metadata.num_decode_reqs,
                query_len=attn_metadata.decode_query_len,
                mixed=mixed,
            )
            if not used_hpc:
                # Non-strict diagnostics retain a correctness fallback. Formal
                # runs set GQLA_HPC_STRICT=1 and can never reach this branch.
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

            if mixed:
                assert attn_metadata.prefill_metadata is not None
                prefill_end = (
                    num_decode_tokens
                    + attn_metadata.prefill_metadata.num_actual_tokens
                )
                super().forward(
                    layer,
                    query[num_decode_tokens:prefill_end],
                    key[num_decode_tokens:prefill_end],
                    value[num_decode_tokens:prefill_end],
                    kv_cache,
                    attn_metadata.prefill_metadata,
                    output[num_decode_tokens:prefill_end],
                    output_scale,
                    output_block_scale,
                )
            return output

        # Backward-compatible guard for metadata created before the custom
        # builder is installed. Pure decode is still accelerated; strict mode
        # rejects a mixed batch instead of hiding a decode fallback.
        decode_shape = _uniform_decode_shape(attn_metadata)
        if decode_shape is None:
            if self._gqla_hpc_strict and attn_metadata is not None:
                self._fallback_or_raise(
                    "missing_all_decode_metadata",
                    decode_candidate=True,
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

        batch, query_len = decode_shape
        used_hpc = self._run_hpc_decode(
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            batch=batch,
            query_len=query_len,
            mixed=False,
        )
        if not used_hpc:
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

        return output


__all__ = [
    "GQLAHPCFlashAttentionDiffKVBackend",
    "GQLAHPCFlashAttentionDiffKVImpl",
    "GQLAHPCFlashAttentionMetadata",
    "GQLAHPCFlashAttentionMetadataBuilder",
    "_analyze_hpc_batch",
    "_env_enabled",
    "_slice_common_metadata",
    "_uniform_decode_shape",
]
