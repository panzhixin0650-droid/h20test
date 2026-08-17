# SPDX-License-Identifier: Apache-2.0
"""CPU-only structural tests for the DeepSeek-V3.1 vLLM GQA path."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

from src.vllm_hpc_gqa_backend import (
    GQLAHPCFlashAttentionDiffKVBackend,
    GQLAHPCFlashAttentionDiffKVImpl,
    GQLAHPCFlashAttentionMetadata,
    GQLAHPCFlashAttentionMetadataBuilder,
    _analyze_hpc_batch,
    _env_enabled,
    _uniform_decode_shape,
)
from src.vllm_model_dsv3 import (
    _expand_gqla_kv_b_weight,
    _ensure_gqla_attention_bf16,
    _gqa_tp_head_partition,
    filter_gqla_attention_overrides,
)


class _SyntheticProjection(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))


class _SyntheticDeepseekGQALoaderModel(nn.Module):
    """Small named-parameter twin for exercising the upstream loader itself."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        attention = nn.Module()
        self.layers[0].self_attn = attention
        shapes = {
            "q_a_proj": (2, 3),
            "q_b_proj": (4, 2),
            "kv_a_proj_with_mqa": (5, 3),
            "kv_b_proj": (6, 4),
            "o_proj": (3, 7),
        }
        for name, shape in shapes.items():
            setattr(attention, name, _SyntheticProjection(shape))

        self.config = SimpleNamespace(
            n_routed_experts=0,
            n_shared_experts=0,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
        )
        self.num_redundant_experts = 0
        self.use_mha = False


class DeepseekV3GQLAStructureTest(unittest.TestCase):
    def test_gqa_tp_partition_g8(self) -> None:
        cases = [(1, (128, 8)), (2, (64, 4)), (4, (32, 2)), (8, (16, 1))]
        for tp_size, expected in cases:
            with self.subTest(tp_size=tp_size):
                self.assertEqual(_gqa_tp_head_partition(128, 8, tp_size), expected)

    def test_gqa_tp_partition_does_not_silently_replicate_kv(self) -> None:
        with self.assertRaisesRegex(ValueError, "replication is not implemented"):
            _gqa_tp_head_partition(128, 8, 16)

    def test_absorb_expansion_path_is_preserved(self) -> None:
        # Two KV groups, two rows per group, two query heads per group.
        weight = torch.arange(8, dtype=torch.float32).view(4, 2)
        expanded = _expand_gqla_kv_b_weight(
            weight, num_heads=4, num_kv_heads=2, per_head_kv=2
        )
        expected = torch.stack(
            [
                weight[0],
                weight[1],
                weight[0],
                weight[1],
                weight[2],
                weight[3],
                weight[2],
                weight[3],
            ]
        )
        torch.testing.assert_close(expanded, expected)

    def test_uniform_decode_shape(self) -> None:
        for query_len in (1, 2):
            batch = 3
            metadata = SimpleNamespace(
                query_start_loc=torch.empty(batch + 1, dtype=torch.int32),
                max_query_len=query_len,
                num_actual_tokens=batch * query_len,
            )
            with self.subTest(query_len=query_len):
                self.assertEqual(_uniform_decode_shape(metadata), (batch, query_len))

    def test_legacy_uniform_detector_rejects_prefill_and_mixed_batch(self) -> None:
        prefill = SimpleNamespace(
            query_start_loc=torch.empty(3, dtype=torch.int32),
            max_query_len=3,
            num_actual_tokens=6,
        )
        mixed = SimpleNamespace(
            query_start_loc=torch.empty(4, dtype=torch.int32),
            max_query_len=2,
            num_actual_tokens=5,
        )
        self.assertIsNone(_uniform_decode_shape(prefill))
        self.assertIsNone(_uniform_decode_shape(mixed))

    def test_mixed_batch_analysis_extracts_real_decode_prefix(self) -> None:
        metadata = SimpleNamespace(
            num_reqs=4,
            num_actual_tokens=10,
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3, 10]),
            # Request 2 is a one-token final prefill chunk. It must remain in
            # the DiffKV suffix instead of being mistaken for a decode.
            is_prefilling=torch.tensor([False, False, True, True]),
        )
        split = _analyze_hpc_batch(metadata)
        self.assertEqual(split.num_decode_reqs, 2)
        self.assertEqual(split.num_decode_tokens, 2)
        self.assertEqual(split.decode_query_len, 1)
        self.assertEqual(split.num_prefill_reqs, 2)
        self.assertEqual(split.num_prefill_tokens, 8)

    def test_mixed_batch_analysis_requires_contiguous_uniform_decode(self) -> None:
        noncontiguous = SimpleNamespace(
            num_reqs=3,
            num_actual_tokens=3,
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3]),
            is_prefilling=torch.tensor([False, True, False]),
        )
        with self.assertRaisesRegex(ValueError, "decode_not_contiguous_prefix"):
            _analyze_hpc_batch(noncontiguous)

        nonuniform = SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=3,
            query_start_loc_cpu=torch.tensor([0, 1, 3]),
            is_prefilling=torch.tensor([False, False]),
        )
        with self.assertRaisesRegex(ValueError, "nonuniform_decode_query_lens"):
            _analyze_hpc_batch(nonuniform)

    def test_hpc_backend_fixes_diffkv_page_and_dimensions(self) -> None:
        self.assertEqual(
            GQLAHPCFlashAttentionDiffKVBackend.get_name(), "FLASH_ATTN_DIFFKV"
        )
        self.assertEqual(
            GQLAHPCFlashAttentionDiffKVBackend.get_supported_kernel_block_sizes(),
            [64],
        )
        self.assertEqual(
            GQLAHPCFlashAttentionDiffKVBackend.get_preferred_block_size(16), 64
        )
        self.assertIs(
            GQLAHPCFlashAttentionDiffKVBackend.get_builder_cls(),
            GQLAHPCFlashAttentionMetadataBuilder,
        )
        self.assertEqual(
            GQLAHPCFlashAttentionMetadataBuilder.reorder_batch_threshold,
            2,
        )
        self.assertEqual(
            GQLAHPCFlashAttentionDiffKVBackend.get_kv_cache_shape(
                num_blocks=7,
                block_size=64,
                num_kv_heads=2,
                head_size=192,
            ),
            (7, 64, 2, 320),
        )

    def test_mixed_forward_routes_decode_prefix_only_to_hpc(self) -> None:
        impl = object.__new__(GQLAHPCFlashAttentionDiffKVImpl)
        impl._gqla_hpc_strict = True
        impl._gqla_hpc_trace = False

        decode_metadata = SimpleNamespace(num_actual_tokens=2)
        prefill_metadata = SimpleNamespace(num_actual_tokens=4)
        # Bypass the verbose parent dataclass constructor; forward only reads
        # the split fields below, while isinstance() still exercises the real
        # production dispatch contract.
        metadata = object.__new__(GQLAHPCFlashAttentionMetadata)
        metadata.num_decode_tokens = 2
        metadata.num_decode_reqs = 2
        metadata.decode_query_len = 1
        metadata.decode_metadata = decode_metadata
        metadata.prefill_metadata = prefill_metadata

        query = torch.zeros(6, 2, 3)
        key = torch.zeros(6, 1, 3)
        value = torch.zeros(6, 1, 2)
        kv_cache = torch.zeros(1)
        output = torch.zeros(6, 4)
        parent_calls = []

        def fake_hpc(
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            *,
            batch,
            query_len,
            mixed,
        ):
            del kv_cache, output_scale, output_block_scale
            self.assertIs(attn_metadata, decode_metadata)
            self.assertEqual(query.shape[0], 2)
            self.assertEqual((batch, query_len, mixed), (2, 1, True))
            output.fill_(7)
            return True

        def fake_parent(
            _self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            del layer, key, value, kv_cache, output_scale, output_block_scale
            parent_calls.append((query.shape[0], attn_metadata))
            output.fill_(9)
            return output

        with (
            patch.object(impl, "_run_hpc_decode", side_effect=fake_hpc),
            patch(
                "vllm.v1.attention.backends.flash_attn_diffkv."
                "FlashAttentionDiffKVImpl.forward",
                new=fake_parent,
            ),
        ):
            result = impl.forward(
                None,
                query,
                key,
                value,
                kv_cache,
                metadata,
                output,
            )

        self.assertIs(result, output)
        torch.testing.assert_close(output[:2], torch.full_like(output[:2], 7))
        torch.testing.assert_close(output[2:], torch.full_like(output[2:], 9))
        self.assertEqual(parent_calls, [(4, prefill_metadata)])

    def test_env_switches_are_explicit(self) -> None:
        with patch.dict(os.environ, {"GQLA_HPC_TRACE": "YeS"}, clear=False):
            self.assertTrue(_env_enabled("GQLA_HPC_TRACE"))
        with patch.dict(os.environ, {"GQLA_HPC_TRACE": "0"}, clear=False):
            self.assertFalse(_env_enabled("GQLA_HPC_TRACE"))

    def test_upstream_loader_falls_back_from_missing_fused_qkv_a(self) -> None:
        model = _SyntheticDeepseekGQALoaderModel()
        expected = {
            name: torch.full_like(param, float(index))
            for index, (name, param) in enumerate(model.named_parameters(), start=1)
        }
        self.assertFalse(
            any("fused_qkv_a_proj" in name for name, _ in model.named_parameters())
        )

        loaded = DeepseekV2Model.load_weights(model, iter(expected.items()))

        self.assertEqual(loaded, set(expected))
        for name, param in model.named_parameters():
            with self.subTest(name=name):
                torch.testing.assert_close(param, expected[name])

    def test_unquantized_parallel_loaders_use_expected_tp_axes(self) -> None:
        # Emulate rank 1/2 without creating a process group.  These are the
        # exact unquantized loaders attached when GQLA attention passes
        # quant_config=None (the converted checkpoint stores attention in BF16).
        column = object.__new__(ColumnParallelLinear)
        column.tp_size = 2
        column.tp_rank = 1
        global_column = torch.arange(24, dtype=torch.float32).view(8, 3)
        local_column = nn.Parameter(torch.zeros(4, 3))
        local_column.output_dim = 0
        ColumnParallelLinear.weight_loader(column, local_column, global_column)
        torch.testing.assert_close(local_column, global_column[4:8])

        row = object.__new__(RowParallelLinear)
        row.tp_size = 2
        row.tp_rank = 1
        global_row = torch.arange(24, dtype=torch.float32).view(3, 8)
        local_row = nn.Parameter(torch.zeros(3, 4))
        local_row.input_dim = 1
        RowParallelLinear.weight_loader(row, local_row, global_row)
        torch.testing.assert_close(local_row, global_row[:, 4:8])

        replicated = object.__new__(ReplicatedLinear)
        global_replicated = torch.arange(6, dtype=torch.float32).view(2, 3)
        local_replicated = nn.Parameter(torch.zeros(2, 3))
        ReplicatedLinear.weight_loader(
            replicated, local_replicated, global_replicated
        )
        torch.testing.assert_close(local_replicated, global_replicated)

    def test_bf16_exclusions_cover_mqa_fused_and_separate_gqa_projections(self) -> None:
        quant_config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["self_attn"],
            }
        )
        quant_config.packed_modules_mapping = {
            "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
        }
        q_b_prefix = "model.layers.0.self_attn.q_b_proj"
        fused_prefix = "model.layers.0.self_attn.fused_qkv_a_proj"

        # This captures the vLLM 0.22 regression the helper is guarding: the
        # short HF marker is an exact-prefix non-match for real projection names.
        self.assertFalse(
            is_layer_skipped(
                q_b_prefix,
                quant_config.ignored_layers,
                quant_config.packed_modules_mapping,
            )
        )
        self.assertFalse(
            is_layer_skipped(
                fused_prefix,
                quant_config.ignored_layers,
                quant_config.packed_modules_mapping,
            )
        )

        synthetic_vllm_config = SimpleNamespace(
            quant_config=quant_config,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=2)
            ),
        )
        _ensure_gqla_attention_bf16(synthetic_vllm_config)
        _ensure_gqla_attention_bf16(synthetic_vllm_config)  # idempotent

        self.assertTrue(
            is_layer_skipped(
                q_b_prefix,
                quant_config.ignored_layers,
                quant_config.packed_modules_mapping,
            )
        )
        # The MQA/MLA destination is fused, but vLLM maps its two shards back to
        # the newly excluded q_a + kv_a source prefixes.
        self.assertTrue(
            is_layer_skipped(
                fused_prefix,
                quant_config.ignored_layers,
                quant_config.packed_modules_mapping,
            )
        )
        self.assertEqual(
            quant_config.ignored_layers.count(q_b_prefix),
            1,
        )

    def test_override_filter_drops_stale_fp8_but_preserves_mtp(self) -> None:
        projections = (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        )
        stream: list[tuple[str, torch.Tensor]] = [
            ("model.embed_tokens.weight", torch.zeros(1, dtype=torch.bfloat16))
        ]
        expected_main_names: set[str] = set()
        for layer_idx in range(2):
            for projection in projections:
                base = f"model.layers.{layer_idx}.self_attn.{projection}"
                name = f"{base}.weight"
                expected_main_names.add(name)
                # The override file sorts before model-* shards in the real
                # iterator, so exercise BF16 first followed by both stale keys.
                stream.extend(
                    [
                        (name, torch.zeros(1, dtype=torch.bfloat16)),
                        (name, torch.zeros(1, dtype=torch.float8_e4m3fn)),
                        (
                            f"{base}.weight_scale_inv",
                            torch.zeros(1, dtype=torch.float32),
                        ),
                    ]
                )

        # Layer 2 is the first MTP layer for this synthetic two-layer model.
        # Its upstream FP8 weight and scale must both remain in the stream.
        mtp_weight_name = "model.layers.2.self_attn.q_b_proj.weight"
        mtp_scale_name = "model.layers.2.self_attn.q_b_proj.weight_scale_inv"
        stream.extend(
            [
                (
                    mtp_weight_name,
                    torch.zeros(1, dtype=torch.float8_e4m3fn),
                ),
                (mtp_scale_name, torch.zeros(1, dtype=torch.float32)),
            ]
        )

        filtered = list(filter_gqla_attention_overrides(stream, 2))
        filtered_by_name = dict(filtered)
        main = {
            name: weight
            for name, weight in filtered
            if name in expected_main_names
        }
        self.assertEqual(set(main), expected_main_names)
        self.assertEqual(len(main), 2 * len(projections))
        self.assertTrue(all(weight.dtype == torch.bfloat16 for weight in main.values()))
        self.assertEqual(filtered_by_name[mtp_weight_name].dtype, torch.float8_e4m3fn)
        self.assertEqual(filtered_by_name[mtp_scale_name].dtype, torch.float32)

    def test_override_filter_rejects_missing_or_duplicate_bf16(self) -> None:
        projections = (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        )

        def override(projection: str) -> tuple[str, torch.Tensor]:
            return (
                f"model.layers.0.self_attn.{projection}.weight",
                torch.zeros(1, dtype=torch.bfloat16),
            )

        missing = [override(projection) for projection in projections[:-1]]
        # A stale FP8 copy cannot satisfy the missing BF16 override contract.
        missing.append(
            (
                "model.layers.0.self_attn.o_proj.weight",
                torch.zeros(1, dtype=torch.float8_e4m3fn),
            )
        )
        with self.assertRaisesRegex(ValueError, "accepted 4/5"):
            list(filter_gqla_attention_overrides(missing, 1))

        duplicate = [override(projection) for projection in projections]
        duplicate.append(override("q_a_proj"))
        with self.assertRaisesRegex(ValueError, "duplicate BF16"):
            list(filter_gqla_attention_overrides(duplicate, 1))


if __name__ == "__main__":
    unittest.main()
