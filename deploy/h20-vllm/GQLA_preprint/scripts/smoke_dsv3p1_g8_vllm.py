#!/usr/bin/env python3
"""Run one weights-free vLLM prefill+decode smoke for DeepSeek-V3.1 G=8.

This process runs exactly one path and one tensor-parallel size so vLLM worker
state cannot leak between comparison cases.  Use the accompanying shell driver
to run a matrix and to verify the backend-emitted HPC trace marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from scripts.make_dsv3p1_g8_tiny import FIXTURE_FORMAT, validate_config
except ModuleNotFoundError:  # direct `python scripts/foo.py` without PYTHONPATH
    from make_dsv3p1_g8_tiny import FIXTURE_FORMAT, validate_config


GQA_HPC_ARCHITECTURE = "DeepseekV3GQLAHPCForCausalLM"
MQA_ABSORB_ARCHITECTURE = "DeepseekV3GQLAForCausalLM"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def validate_fixture(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir = model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    metadata_path = model_dir / "gqla_smoke_fixture.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"{model_dir} is not a generated tiny fixture; run make_dsv3p1_g8_tiny.py first"
        )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    metadata = _read_json(metadata_path)
    if metadata.get("fixture_format") != FIXTURE_FORMAT:
        raise ValueError(f"unsupported fixture metadata: {metadata.get('fixture_format')!r}")
    digest = hashlib.sha256(config_bytes).hexdigest()
    if metadata.get("config_sha256") != digest:
        raise ValueError("fixture config checksum does not match gqla_smoke_fixture.json")
    geometry = validate_config(config)
    if metadata.get("attention_geometry") != geometry:
        raise ValueError("fixture metadata geometry does not match config.json")
    expected_runtime = {
        "normalized_rope_type": geometry["expected_normalized_rope_type"],
        "softmax_scale": geometry["expected_softmax_scale"],
    }
    if metadata.get("expected_runtime") != expected_runtime:
        raise ValueError("fixture runtime metadata does not match config.json")
    # A dummy fixture must not accidentally grow real weight files.
    weight_files = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    if weight_files:
        raise ValueError(f"dummy fixture unexpectedly contains weights: {weight_files}")
    return config, geometry


def _set_hpc_contract(*, enabled: bool, strict: bool, trace: bool) -> None:
    if enabled:
        if strict:
            os.environ["GQLA_HPC_STRICT"] = "1"
        else:
            os.environ.pop("GQLA_HPC_STRICT", None)
        if trace:
            os.environ["GQLA_HPC_TRACE"] = "1"
        else:
            os.environ.pop("GQLA_HPC_TRACE", None)
    else:
        # The control case must not inherit HPC routing from the caller.
        os.environ.pop("GQLA_HPC_STRICT", None)
        os.environ.pop("GQLA_HPC_TRACE", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--path", choices=("gqa-hpc", "mqa-absorb"), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gqa-architecture", default=GQA_HPC_ARCHITECTURE)
    parser.add_argument("--mqa-architecture", default=MQA_ABSORB_ARCHITECTURE)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--strict-hpc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: enabled for gqa-hpc, disabled for mqa-absorb",
    )
    parser.add_argument(
        "--trace-hpc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: enabled for gqa-hpc, disabled for mqa-absorb",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mixed-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "submit short and long prompts with chunked prefill so a scheduler "
            "step contains both decode and prefill tokens"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    config, geometry = validate_fixture(model_dir)
    if args.prompt_tokens < 2 or args.decode_tokens < 1:
        raise ValueError("prompt-tokens must be >=2 and decode-tokens must be >=1")
    longest_prompt_tokens = args.prompt_tokens * (3 if args.mixed_batch else 1)
    if longest_prompt_tokens + args.decode_tokens > args.max_model_len:
        raise ValueError("prompt + decode tokens exceed max-model-len")
    if 128 % args.tp or 8 % args.tp:
        raise ValueError(f"TP={args.tp} must divide both H=128 and G=8")

    use_hpc = args.path == "gqa-hpc"
    strict_hpc = use_hpc if args.strict_hpc is None else args.strict_hpc
    trace_hpc = use_hpc if args.trace_hpc is None else args.trace_hpc
    if not use_hpc and (strict_hpc or trace_hpc):
        raise ValueError("HPC strict/trace flags are invalid for the mqa-absorb control")
    _set_hpc_contract(enabled=use_hpc, strict=strict_hpc, trace=trace_hpc)
    # Importing vLLM may initialize CUDA while discovering platform plugins.
    # Spawn is therefore required for the engine/TP workers; fork would later
    # fail with "Cannot re-initialize CUDA in forked subprocess".
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    architecture = args.gqa_architecture if use_hpc else args.mqa_architecture
    start_record = {
        "path": args.path,
        "architecture": architecture,
        "tp": args.tp,
        "load_format": "dummy",
        "strict_hpc": strict_hpc,
        "trace_hpc": trace_hpc,
        "mixed_batch": args.mixed_batch,
        "geometry": geometry,
    }
    print("DSV3P1_G8_VLLM_SMOKE_BEGIN " + json.dumps(start_record, sort_keys=True), flush=True)

    # Import only after arming the environment contract.  Every vLLM worker
    # inherits GQLA_HPC_STRICT/GQLA_HPC_TRACE from this process.
    import src.vllm_register_dsv  # noqa: F401
    from vllm import LLM, ModelRegistry, SamplingParams
    from vllm.inputs import TokensPrompt

    if architecture not in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            f"vLLM architecture {architecture!r} is not registered. "
            "Install this project editable so the vllm.general_plugins entry point "
            "is visible in every TP worker."
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    init_start = time.perf_counter()
    llm_kwargs: dict[str, Any] = {
        "model": str(model_dir),
        "skip_tokenizer_init": True,
        "trust_remote_code": False,
        "tensor_parallel_size": args.tp,
        "dtype": args.dtype,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.tp == 2,
        "hf_overrides": {"architectures": [architecture]},
        "load_format": "dummy",
        "max_model_len": args.max_model_len,
        # HPC-Ops GQA consumes 64-token cache pages.  Pin the control to the
        # same page size so cache allocator geometry is not a confounder.
        "block_size": 64,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": args.mixed_batch,
        "disable_log_stats": True,
        "max_num_seqs": 4,
    }
    if args.mixed_batch:
        # With default 16/48-token prompts and a 32-token scheduler budget,
        # the short request starts decode while the long request is still in
        # chunked prefill.  That is the regression case for all-decode HPC.
        llm_kwargs["max_num_batched_tokens"] = args.prompt_tokens * 2
    llm = LLM(
        **llm_kwargs,
    )
    init_seconds = time.perf_counter() - init_start

    vocab_size = int(config["vocab_size"])
    prompt_lengths = [args.prompt_tokens]
    if args.mixed_batch:
        prompt_lengths.append(longest_prompt_tokens)
    prompt_token_ids = [
        [2 + (index % (vocab_size - 2)) for index in range(prompt_length)]
        for prompt_length in prompt_lengths
    ]
    prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids]
    sampling = SamplingParams(
        temperature=0.0,
        min_tokens=args.decode_tokens,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
        seed=args.seed,
    )
    run_start = time.perf_counter()
    requests = llm.generate(prompts, sampling, use_tqdm=False)
    run_seconds = time.perf_counter() - run_start
    if len(requests) != len(prompts) or any(len(request.outputs) != 1 for request in requests):
        raise RuntimeError(f"vLLM did not return exactly {len(prompts)} completions")
    generated_token_ids = [list(request.outputs[0].token_ids) for request in requests]
    if any(len(token_ids) != args.decode_tokens for token_ids in generated_token_ids):
        raise RuntimeError(
            "decode length mismatch: "
            f"got {[len(token_ids) for token_ids in generated_token_ids]}, "
            f"expected {args.decode_tokens} each"
        )

    result = {
        **start_record,
        "model_dir": str(model_dir),
        "visible_devices": visible_devices,
        "prompt_tokens": [len(token_ids) for token_ids in prompt_token_ids],
        "decode_tokens": [len(token_ids) for token_ids in generated_token_ids],
        "generated_token_ids": generated_token_ids,
        "init_seconds": init_seconds,
        "prefill_decode_seconds": run_seconds,
        "hpc_proof": (
            "backend trace marker must be verified by run_dsv3p1_g8_vllm_smoke.sh"
            if use_hpc
            else "control architecture selected; HPC environment contract cleared"
        ),
    }
    if args.result_json is not None:
        result_path = args.result_json.expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DSV3P1_G8_VLLM_SMOKE_OK " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
