# SPDX-License-Identifier: Apache-2.0
"""vLLM model classes for GQLA-converted DeepSeek-V3 / V3.1 checkpoints.

Two explicit serving paths intentionally coexist so they can be benchmarked
against the same converted checkpoint:

* ``DeepseekV3GQLAForCausalLM`` is the original MLA-absorb/MQA-cache path.  It
  expands the checkpoint's grouped ``kv_b_proj`` on load and otherwise uses
  upstream vLLM unchanged.
* ``DeepseekV3GQLAHPCForCausalLM`` is the benchmark-facing real grouped-KV-
  cache path (``DeepseekV3GQLAGQAForCausalLM`` remains an alias).  Locally it
  materializes Q ``[H/tp, 192]``, K ``[G/tp, 192]``, and V ``[G/tp, 128]``
  and uses the HPC-Ops decode backend with a FlashAttentionDiffKV fallback.
"""

from __future__ import annotations

from itertools import islice
import re
from typing import Iterable, Iterator

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
    DeepseekV2Model,
    DeepseekV3ForCausalLM,
    _get_llama_4_scaling,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .vllm_hpc_gqa_backend import GQLAHPCFlashAttentionDiffKVBackend


_GQLA_ATTN_PROJECTION_NAMES = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)
_GQLA_ATTN_PARAMETER_RE = re.compile(
    r"(?:^|\.)(?:model\.)?layers\.(?P<layer>\d+)\.self_attn\."
    rf"(?P<projection>{'|'.join(_GQLA_ATTN_PROJECTION_NAMES)})\."
    r"(?P<parameter>[^.]+)$"
)


def _main_gqla_attention_parameter(
    name: str,
    num_hidden_layers: int,
) -> tuple[int, str, str] | None:
    """Parse a main-model GQLA attention parameter name.

    Speculative/MTP layers start at ``num_hidden_layers`` and deliberately do
    not match: they remain ordinary upstream FP8 layers and need their scale
    sidecars.
    """

    match = _GQLA_ATTN_PARAMETER_RE.search(name)
    if match is None:
        return None
    layer_idx = int(match.group("layer"))
    if layer_idx >= num_hidden_layers:
        return None
    return layer_idx, match.group("projection"), match.group("parameter")


def filter_gqla_attention_overrides(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_hidden_layers: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield exactly one BF16 override for each main attention projection.

    Incremental GQLA checkpoints patch their safetensors index to point the
    five attention weights at ``gqla_attention_overrides.safetensors``.  vLLM's
    iterator nevertheless scans the physical symlinked source shards, whose
    contents still include the old FP8 weights and ``weight_scale_inv``
    sidecars.  The destination modules are now unquantized BF16 linears, so
    those stale physical tensors must never reach the upstream loader.

    Matching is restricted to decoder layers ``0..num_hidden_layers-1``.
    MTP/speculative layers at and above that boundary pass through unchanged.
    The accepted canonical ``(layer, projection)`` keys are counted so a
    missing or duplicate override fails explicitly instead of silently leaving
    a random/stale parameter loaded.
    """

    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    expected = {
        (layer_idx, projection)
        for layer_idx in range(num_hidden_layers)
        for projection in _GQLA_ATTN_PROJECTION_NAMES
    }
    accepted: set[tuple[int, str]] = set()

    for name, loaded_weight in weights:
        parsed = _main_gqla_attention_parameter(name, num_hidden_layers)
        if parsed is None:
            yield name, loaded_weight
            continue

        layer_idx, projection, parameter = parsed
        canonical_key = (layer_idx, projection)

        # Main attention linears are uniformly BF16 in the override file.  All
        # scale sidecars and non-BF16 weights here are stale source-shard data.
        if parameter != "weight" or loaded_weight.dtype != torch.bfloat16:
            continue

        if canonical_key in accepted:
            raise ValueError(
                "duplicate BF16 GQLA attention override for "
                f"layer={layer_idx} projection={projection}: {name}"
            )
        accepted.add(canonical_key)
        yield name, loaded_weight

    missing = expected - accepted
    if missing:
        preview = ", ".join(
            f"layers.{layer_idx}.self_attn.{projection}.weight"
            for layer_idx, projection in sorted(missing)[:8]
        )
        if len(missing) > 8:
            preview += f", ... (+{len(missing) - 8} more)"
        raise ValueError(
            "incomplete BF16 GQLA attention overrides: accepted "
            f"{len(accepted)}/{len(expected)}; missing {preview}"
        )


def _expand_gqla_kv_b_weight(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    per_head_kv: int,
) -> torch.Tensor:
    """Expand a GQLA-packed kv_b_proj weight to MLA layout.

    Source: ``(num_kv_heads * per_head_kv, kv_lora_rank)`` — one (W_UK|W_UV)
    block per KV group (rows laid out nope-then-v inside each block by
    :func:`src.compression.compress_and_absorb`).

    Target: ``(num_heads * per_head_kv, kv_lora_rank)`` — block repeated
    ``group_size`` times so the standard MLA absorb path treats each of
    the ``gs`` heads in a group as having its own (identical) W_UK / W_UV.
    KV cache still stores only the latent ``(kv_lora_rank + qk_rope_head_dim)``.
    """
    G = num_kv_heads
    H = num_heads
    if H % G != 0:
        raise ValueError(f"num_heads={H} not divisible by num_kv_heads={G}")
    gs = H // G
    if weight.shape[0] != G * per_head_kv:
        raise ValueError(
            f"GQLA kv_b_proj weight has rows={weight.shape[0]}, expected "
            f"{G}*{per_head_kv}={G * per_head_kv}"
        )
    return (
        weight.view(G, per_head_kv, weight.shape[-1])
        .repeat_interleave(gs, dim=0)
        .reshape(H * per_head_kv, weight.shape[-1])
        .contiguous()
    )


def _ensure_gqla_attention_bf16(
    vllm_config: VllmConfig,
    prefix: str = "",
) -> None:
    """Expand the converted checkpoint's BF16-attention quant exclusions.

    The converter records ``modules_to_not_convert=["self_attn"]``.  vLLM
    0.22's FP8 config imports that value into ``ignored_layers`` but matches
    ordinary linears by exact full prefix, not substring.  Expand the marker to
    the five checkpoint projection names for every layer before either model
    constructs its linears.  For upstream MLA, ``fused_qkv_a_proj`` is mapped
    back to q_a + kv_a by vLLM's packed mapping, so excluding both source names
    makes the fused destination unquantized as well.
    """

    quant_config = vllm_config.quant_config
    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if ignored_layers is None:
        return

    config = vllm_config.model_config.hf_config
    model_prefix = maybe_prefix(prefix, "model")
    expanded = list(ignored_layers)
    seen = set(expanded)
    for layer_idx in range(config.num_hidden_layers):
        attention_prefix = f"{model_prefix}.layers.{layer_idx}.self_attn"
        for projection_name in _GQLA_ATTN_PROJECTION_NAMES:
            name = f"{attention_prefix}.{projection_name}"
            if name not in seen:
                expanded.append(name)
                seen.add(name)
    quant_config.ignored_layers = expanded


class DeepseekV3GQLAForCausalLM(DeepseekV3ForCausalLM):
    """vLLM DeepSeek-V3 ForCausalLM serving the GQLA checkpoint via MLA absorb.

    Inherits the upstream model wholesale (same attention / MoE / TP / cache
    layout) and only patches ``load_weights`` to expand the GQLA per-group
    ``kv_b_proj`` to full ``num_heads`` rows at load time.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        _ensure_gqla_attention_bf16(vllm_config, prefix)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.config
        H = config.num_attention_heads
        G = config.num_key_value_heads
        per_head_kv = config.qk_nope_head_dim + config.v_head_dim
        filtered = filter_gqla_attention_overrides(
            weights,
            config.num_hidden_layers,
        )

        # Filtering first guarantees that expansion only sees the canonical
        # override rather than the stale FP8 kv_b tensor in a symlinked shard.
        def _wrapped(it):
            for name, w in it:
                parsed = _main_gqla_attention_parameter(
                    name,
                    config.num_hidden_layers,
                )
                if (
                    parsed is not None
                    and parsed[1:] == ("kv_b_proj", "weight")
                    and w.shape[0] == G * per_head_kv
                ):
                    w = _expand_gqla_kv_b_weight(w, H, G, per_head_kv)
                yield name, w

        return super().load_weights(_wrapped(filtered))


class DeepseekV2GQLAForCausalLM(DeepseekV2ForCausalLM):
    """Same expand-on-load wrapper for DeepSeek-V2-family checkpoints
    (e.g. V2-Lite). DeepseekV2 also uses the MLA-absorb path in vLLM."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        _ensure_gqla_attention_bf16(vllm_config, prefix)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.config
        H = config.num_attention_heads
        G = config.num_key_value_heads
        per_head_kv = config.qk_nope_head_dim + config.v_head_dim

        def _wrapped(it):
            for name, w in it:
                if name.endswith("kv_b_proj.weight") and w.shape[0] == G * per_head_kv:
                    w = _expand_gqla_kv_b_weight(w, H, G, per_head_kv)
                yield name, w

        return super().load_weights(_wrapped(weights))


def _gqa_tp_head_partition(
    num_heads: int,
    num_kv_heads: int,
    tp_size: int,
) -> tuple[int, int]:
    """Validate the initial GQA TP contract and return local Q/KV heads.

    This first implementation deliberately supports only partitioning KV heads,
    not replicating them: both H and G must be divisible by TP.  Because the
    converter packs groups contiguously, ColumnParallelLinear then assigns each
    rank exactly the K/V groups corresponding to its contiguous query-head
    shard.
    """

    if num_heads <= 0 or num_kv_heads <= 0 or tp_size <= 0:
        raise ValueError("num_heads, num_kv_heads, and tp_size must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}"
        )
    if num_heads % tp_size != 0:
        raise ValueError(f"num_heads={num_heads} not divisible by tp_size={tp_size}")
    if num_kv_heads % tp_size != 0:
        raise ValueError(
            "GQA TP replication is not implemented: "
            f"num_kv_heads={num_kv_heads} must be divisible by tp_size={tp_size}"
        )
    local_heads = num_heads // tp_size
    local_kv_heads = num_kv_heads // tp_size
    if local_heads % local_kv_heads != 0:
        raise ValueError(
            f"local Q heads={local_heads} not divisible by local KV heads="
            f"{local_kv_heads}"
        )
    return local_heads, local_kv_heads


class DeepseekV3GQLAGQAAttention(nn.Module):
    """DeepSeek-V3.1 attention with a real grouped K/V cache.

    ``kv_a_proj_with_mqa`` remains replicated, as in DeepSeek MLA, but the
    decompression projection has only ``G`` grouped blocks.  Its output is
    sharded on that group axis.  The shared RoPE key is broadcast to each local
    group, producing K=192 and V=128 without padding V to the QK dimension.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # The converted checkpoint deliberately stores every self-attention
        # projection in BF16 (config.quantization_config has
        # modules_to_not_convert=["self_attn"]).  vLLM 0.22's FP8 ignore list
        # uses exact-prefix matching, so the short HF marker is not sufficient
        # by itself for out-of-tree projection names.  Make that checkpoint
        # contract explicit here: attention linears and its BF16 KV cache use
        # unquantized methods; MoE/MLP modules still receive the model's FP8
        # quant_config in the decoder below.
        attention_quant_config = None

        if (self.qk_head_dim, self.v_head_dim) != (192, 128):
            raise ValueError(
                "DeepSeek GQLA HPC path requires QK=192 and V=128, got "
                f"QK={self.qk_head_dim}, V={self.v_head_dim}"
            )

        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads, self.num_local_kv_heads = _gqa_tp_head_partition(
            num_heads, num_kv_heads, tp_size
        )
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        if q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                hidden_size,
                q_lora_rank,
                bias=False,
                quant_config=attention_quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=attention_quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=attention_quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
            quant_config=attention_quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=attention_quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=attention_quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling *= mscale * mscale

        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            cache_config=cache_config,
            quant_config=attention_quant_config,
            prefix=f"{prefix}.attn",
            attn_backend=GQLAHPCFlashAttentionDiffKVBackend,
            head_size_v=self.v_head_dim,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0]
        else:
            q = self.q_proj(hidden_states)[0]
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        _, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0].view(
            -1,
            self.num_local_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = kv.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # DeepSeek's compressed RoPE key is shared by all grouped KV heads.
        k_pe = k_pe.unsqueeze(1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = k_nope.new_empty(
            (k_nope.shape[0], self.num_local_kv_heads, self.qk_head_dim)
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        # Runtime scale (including the DeepSeek YaRN mscale^2 correction) is
        # held by the backend and forwarded to both FlashAttention and HPC-Ops.
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV3GQLAGQADecoderLayer(DeepseekV2DecoderLayer):
    """DeepSeek decoder with only its attention module replaced by real GQA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = vllm_config.model_config.hf_config
        if topk_indices_buffer is not None or hasattr(config, "index_topk"):
            raise ValueError("DeepSeek-V3.2 sparse indexer is not supported by GQLA GQA")

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.use_mha = False

        self.self_attn = DeepseekV3GQLAGQAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class DeepseekV3GQLAGQAModel(nn.Module):
    """Standalone DeepSeek-V3.1 model wiring the explicit GQA decoder."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        if hasattr(config, "index_topk"):
            raise ValueError("DeepSeek-V3.2 sparse indexer is not supported by GQLA GQA")
        self.config = config
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_v32 = False

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV3GQLAGQADecoderLayer(
                vllm_config=vllm_config,
                config=config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()
        self.use_mha = False
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    # The upstream loader contains the DeepSeek FP8/MoE/expert mappings and its
    # optional fused-qkv mapping falls back to these separate q_a/kv_a modules.
    load_weights = DeepseekV2Model.load_weights


class DeepseekV3GQLAGQAForCausalLM(DeepseekV3ForCausalLM):
    """Explicit real-GQA DeepSeek-V3.1 vLLM architecture."""

    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        _ensure_gqla_attention_bf16(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.use_mha = False
        self.fuse_qkv_a_proj = False
        self.packed_modules_mapping = dict(type(self).packed_modules_mapping)

        self.model = DeepseekV3GQLAGQAModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.set_moe_parameters()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = filter_gqla_attention_overrides(
            weights,
            self.config.num_hidden_layers,
        )
        return super().load_weights(filtered)


class DeepseekV3GQLAHPCForCausalLM(DeepseekV3GQLAGQAForCausalLM):
    """Benchmark-facing name for the real-GQA path using the HPC backend."""

    pass


__all__ = [
    "DeepseekV3GQLAForCausalLM",
    "DeepseekV2GQLAForCausalLM",
    "DeepseekV3GQLAGQAAttention",
    "DeepseekV3GQLAGQADecoderLayer",
    "DeepseekV3GQLAGQAModel",
    "DeepseekV3GQLAGQAForCausalLM",
    "DeepseekV3GQLAHPCForCausalLM",
    "filter_gqla_attention_overrides",
    "_expand_gqla_kv_b_weight",
    "_ensure_gqla_attention_bf16",
    "_gqa_tp_head_partition",
]
