#!/usr/bin/env python3
"""Build a weights-free DeepSeek-V3.1 G=8 fixture for vLLM dummy loads.

The fixture preserves the production attention geometry (H=128, G=8,
QK=192, V=128), MLA ranks, and DeepSeek YaRN configuration.  Only the
vocabulary, hidden width, MLP width, and layer count are reduced.  It
deliberately contains no tokenizer and no weights: consumers must use
``skip_tokenizer_init=True`` and ``load_format="dummy"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FIXTURE_FORMAT = "dsv3p1-g8-vllm-dummy-v2"
DEFAULT_ARCHITECTURE = "DeepseekV3GQLAForCausalLM"
PRODUCTION_MAX_POSITION_EMBEDDINGS = 163840
PRODUCTION_ROPE_THETA = 10000.0
PRODUCTION_ROPE_SCALING: dict[str, int | float | str] = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn",
}
EXPECTED_NORMALIZED_ROPE_TYPE = "deepseek_yarn"
EXPECTED_SOFTMAX_SCALE = 0.1352337788608801


def build_config(*, architecture: str = DEFAULT_ARCHITECTURE) -> dict[str, Any]:
    """Return the minimal config while retaining production attention shapes."""

    return {
        "architectures": [architecture],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "ep_size": 1,
        # Keep the only decoder layer dense.  Transformers 5.x uses a strict
        # dataclass and rejects n_routed_experts=None, hence the small, unused
        # MoE values below instead of disabling MoE with None.
        "first_k_dense_replace": 2,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        # Preserve the production attention front-end ranks and head geometry.
        "kv_lora_rank": 512,
        # Preserve production YaRN verbatim.  The smoke runner limits runtime
        # allocation with max_model_len=64 instead of changing RoPE semantics.
        "max_position_embeddings": PRODUCTION_MAX_POSITION_EMBEDDINGS,
        "model_type": "deepseek_v3",
        "moe_intermediate_size": 32,
        "moe_layer_freq": 1,
        # This n_group is the (unused) MoE router setting, not GQA's G=8.
        "n_group": 2,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 128,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 8,
        "num_nextn_predict_layers": 0,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_scaling": dict(PRODUCTION_ROPE_SCALING),
        "rope_theta": PRODUCTION_ROPE_THETA,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "transformers_version": "5.12.1",
        "use_cache": True,
        "v_head_dim": 128,
        "vocab_size": 256,
    }


def derive_attention_geometry(config: dict[str, Any]) -> dict[str, Any]:
    h = int(config["num_attention_heads"])
    g = int(config["num_key_value_heads"])
    qk_nope = int(config["qk_nope_head_dim"])
    qk_rope = int(config["qk_rope_head_dim"])
    v_dim = int(config["v_head_dim"])
    kv_rank = int(config["kv_lora_rank"])
    q_rank = int(config["q_lora_rank"])
    hidden = int(config["hidden_size"])
    qk_dim = qk_nope + qk_rope
    rope_scaling = config["rope_scaling"]
    scaling_factor = float(rope_scaling["factor"])
    mscale_all_dim = float(rope_scaling["mscale_all_dim"])
    yarn_mscale = (
        1.0
        if scaling_factor <= 1.0
        else 1.0 + 0.1 * mscale_all_dim * math.log(scaling_factor)
    )
    base_softmax_scale = qk_dim**-0.5
    return {
        "num_query_heads": h,
        "num_kv_heads": g,
        "group_size": h // g,
        "qk_head_dim": qk_dim,
        "v_head_dim": v_dim,
        "gqa_cache_elements_per_token": g * (qk_dim + v_dim),
        "mla_absorb_cache_elements_per_token": kv_rank + qk_rope,
        "rope_scaling": dict(rope_scaling),
        "expected_normalized_rope_type": EXPECTED_NORMALIZED_ROPE_TYPE,
        "base_softmax_scale": base_softmax_scale,
        "yarn_mscale": yarn_mscale,
        "expected_softmax_scale": base_softmax_scale * yarn_mscale * yarn_mscale,
        "expected_global_weight_shapes": {
            "q_a_proj.weight": [q_rank, hidden],
            "q_b_proj.weight": [h * qk_dim, q_rank],
            "kv_a_proj_with_mqa.weight": [kv_rank + qk_rope, hidden],
            "kv_b_proj.gqa.weight": [g * (qk_nope + v_dim), kv_rank],
            "kv_b_proj.mla_absorb.weight": [h * (qk_nope + v_dim), kv_rank],
            "o_proj.weight": [hidden, h * v_dim],
        },
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate exact kernel geometry and the fixture's small-allocation guard."""

    if config.get("rope_scaling") != PRODUCTION_ROPE_SCALING:
        raise ValueError(
            "fixture YaRN mismatch: expected production rope_scaling "
            f"{PRODUCTION_ROPE_SCALING}, got {config.get('rope_scaling')!r}"
        )
    if int(config.get("max_position_embeddings", -1)) != PRODUCTION_MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            "fixture YaRN mismatch: max_position_embeddings must remain "
            f"{PRODUCTION_MAX_POSITION_EMBEDDINGS}"
        )
    if float(config.get("rope_theta", -1.0)) != PRODUCTION_ROPE_THETA:
        raise ValueError(
            f"fixture YaRN mismatch: rope_theta must remain {PRODUCTION_ROPE_THETA}"
        )

    geometry = derive_attention_geometry(config)
    expected = {
        "num_query_heads": 128,
        "num_kv_heads": 8,
        "group_size": 16,
        "qk_head_dim": 192,
        "v_head_dim": 128,
        "gqa_cache_elements_per_token": 2560,
        "mla_absorb_cache_elements_per_token": 576,
    }
    mismatches = {
        key: (geometry.get(key), value)
        for key, value in expected.items()
        if geometry.get(key) != value
    }
    if mismatches:
        raise ValueError(f"fixture attention geometry mismatch: {mismatches}")
    if geometry["expected_normalized_rope_type"] != EXPECTED_NORMALIZED_ROPE_TYPE:
        raise ValueError("fixture YaRN must normalize to deepseek_yarn")
    if not math.isclose(
        float(geometry["expected_softmax_scale"]),
        EXPECTED_SOFTMAX_SCALE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "fixture YaRN softmax scale mismatch: expected "
            f"{EXPECTED_SOFTMAX_SCALE}, got {geometry['expected_softmax_scale']}"
        )

    h = int(config["num_attention_heads"])
    g = int(config["num_key_value_heads"])
    if h % g:
        raise ValueError(f"num_attention_heads={h} is not divisible by num_key_value_heads={g}")
    if int(config["num_hidden_layers"]) != 1:
        raise ValueError("dummy fixture must have exactly one decoder layer")
    if int(config["first_k_dense_replace"]) <= int(config["num_hidden_layers"]):
        raise ValueError("dummy fixture must keep every decoder layer dense")
    if int(config["hidden_size"]) > 256 or int(config["vocab_size"]) > 1024:
        raise ValueError("refusing a fixture that exceeds the tiny-allocation guard")
    if config.get("quantization_config") is not None:
        raise ValueError("dummy fixture must not enable checkpoint quantization")
    return geometry


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_checked(path: Path, content: bytes, *, force: bool) -> None:
    if path.exists():
        old = path.read_bytes()
        if old == content:
            return
        if not force:
            raise FileExistsError(f"refusing to replace non-matching file without --force: {path}")
    path.write_bytes(content)


def write_fixture(
    output_dir: Path,
    *,
    architecture: str = DEFAULT_ARCHITECTURE,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(architecture=architecture)
    geometry = validate_config(config)
    config_bytes = _json_bytes(config)
    metadata = {
        "fixture_format": FIXTURE_FORMAT,
        "weights": "none; vLLM load_format=dummy is mandatory",
        "tokenizer": "none; vLLM skip_tokenizer_init=True is mandatory",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "attention_geometry": geometry,
        "expected_runtime": {
            "normalized_rope_type": geometry["expected_normalized_rope_type"],
            "softmax_scale": geometry["expected_softmax_scale"],
        },
        "supported_tensor_parallel_sizes": [1, 2],
    }
    generation_config = {
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
    }

    _write_checked(output_dir / "config.json", config_bytes, force=force)
    _write_checked(
        output_dir / "gqla_smoke_fixture.json",
        _json_bytes(metadata),
        force=force,
    )
    _write_checked(
        output_dir / "generation_config.json",
        _json_bytes(generation_config),
        force=force,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        help="base architecture stored in config; the smoke runner overrides it per path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the three fixture metadata files if their content differs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = write_fixture(
        args.output_dir,
        architecture=args.architecture,
        force=args.force,
    )
    print(
        "DSV3P1_G8_TINY_FIXTURE_OK "
        + json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "attention_geometry": metadata["attention_geometry"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
