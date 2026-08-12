#!/usr/bin/env python3
"""A/B test FA3 GQA varlen metadata versus a true fixed-length call."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from flash_attn_3 import flash_attn_interface as fa3


CONTEXT_LEN = 8192
NUM_HEADS_Q = 128
HEAD_DIM_QK = 192
HEAD_DIM_V = 128
DTYPE = torch.bfloat16
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM_QK)
GIB = 1 << 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FA3 GQA varlen and fixed-length scheduler paths"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-device-substring", default="H20")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--measured-hbm-tb-s", type=float, default=3.572)
    parser.add_argument("--measured-bf16-tflops", type=float, default=141.48)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup <= 0 or args.samples <= 0:
        parser.error("batch-size, warmup, and samples must be positive")
    return args


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(samples_us: list[float]) -> dict[str, Any]:
    return {
        "count": len(samples_us),
        "min_us": min(samples_us),
        "p20_us": percentile(samples_us, 0.20),
        "median_us": statistics.median(samples_us),
        "p80_us": percentile(samples_us, 0.80),
        "max_us": max(samples_us),
        "samples_us": samples_us,
    }


def time_one(function: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1_000.0
    del output
    return elapsed_us


def benchmark_modes(
    modes: dict[str, Callable[[], torch.Tensor]], *, warmup: int, samples: int
) -> dict[str, dict[str, Any]]:
    names = list(modes)
    for _ in range(warmup):
        for name in names:
            modes[name]()
    torch.cuda.synchronize()

    raw: dict[str, list[float]] = {name: [] for name in names}
    for sample_index in range(samples):
        order = names if sample_index % 2 == 0 else list(reversed(names))
        for name in order:
            raw[name].append(time_one(modes[name]))
    return {name: timing_summary(raw[name]) for name in names}


def make_inputs(
    *, batch_size: int, g: int, s_q: int, page_size: int | None, seed: int
) -> dict[str, Any]:
    torch.manual_seed(seed)
    q = torch.empty(
        (batch_size, s_q, NUM_HEADS_Q, HEAD_DIM_QK),
        device="cuda",
        dtype=DTYPE,
    )
    q.normal_(mean=0.0, std=0.1)
    if page_size is None:
        k_cache = torch.empty(
            (batch_size, CONTEXT_LEN, g, HEAD_DIM_QK),
            device="cuda",
            dtype=DTYPE,
        )
        v_cache = torch.empty(
            (batch_size, CONTEXT_LEN, g, HEAD_DIM_V),
            device="cuda",
            dtype=DTYPE,
        )
        page_table = None
    else:
        blocks_per_sequence = CONTEXT_LEN // page_size
        k_cache = torch.empty(
            (batch_size * blocks_per_sequence, page_size, g, HEAD_DIM_QK),
            device="cuda",
            dtype=DTYPE,
        )
        v_cache = torch.empty(
            (batch_size * blocks_per_sequence, page_size, g, HEAD_DIM_V),
            device="cuda",
            dtype=DTYPE,
        )
        page_table = torch.arange(
            batch_size * blocks_per_sequence,
            device="cuda",
            dtype=torch.int32,
        ).view(batch_size, blocks_per_sequence)
    k_cache.normal_(mean=0.0, std=0.1)
    v_cache.normal_(mean=0.0, std=0.1)
    cache_seqlens = torch.full(
        (batch_size,), CONTEXT_LEN, device="cuda", dtype=torch.int32
    )
    torch.cuda.synchronize()
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
    }


def make_varlen_callable(
    inputs: dict[str, Any], *, batch_size: int, g: int, s_q: int, page_size: int | None
) -> Callable[[], torch.Tensor]:
    scheduler_metadata = fa3.get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=s_q,
        max_seqlen_k=CONTEXT_LEN,
        num_heads_q=NUM_HEADS_Q,
        num_heads_kv=g,
        headdim=HEAD_DIM_QK,
        cache_seqlens=inputs["cache_seqlens"],
        qkv_dtype=DTYPE,
        headdim_v=HEAD_DIM_V,
        page_size=page_size,
        causal=True,
        num_splits=1,
        pack_gqa=True,
    )

    def run() -> torch.Tensor:
        return fa3.flash_attn_with_kvcache(
            inputs["q"],
            inputs["k_cache"],
            inputs["v_cache"],
            cache_seqlens=inputs["cache_seqlens"],
            page_table=inputs["page_table"],
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
            scheduler_metadata=scheduler_metadata,
            num_splits=1,
            pack_gqa=True,
        )

    return run


def make_fixed_callable(
    inputs: dict[str, Any], *, num_splits: int, pack_gqa: bool | None
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        return fa3.flash_attn_with_kvcache(
            inputs["q"],
            inputs["k_cache"],
            inputs["v_cache"],
            cache_seqlens=None,
            page_table=inputs["page_table"],
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
            scheduler_metadata=None,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
        )

    return run


def compare_outputs(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    difference = (actual.float() - reference.float()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "allclose": bool(
            torch.allclose(
                actual.float(), reference.float(), rtol=2.01 / 128, atol=8.0e-4
            )
        ),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def derive_metrics(
    *, args: argparse.Namespace, g: int, s_q: int, full_call_us: float
) -> dict[str, float]:
    physical_bytes_per_sequence = CONTEXT_LEN * g * (HEAD_DIM_QK + HEAD_DIM_V) * 2
    flops_per_sequence = (
        2 * NUM_HEADS_Q * s_q * CONTEXT_LEN * (HEAD_DIM_QK + HEAD_DIM_V)
    )
    memory_lower_us = physical_bytes_per_sequence / args.measured_hbm_tb_s / 1.0e6
    compute_lower_us = flops_per_sequence / args.measured_bf16_tflops / 1.0e6
    roofline_us = max(memory_lower_us, compute_lower_us)
    per_sequence_us = full_call_us / args.batch_size
    return {
        "per_sequence_us": per_sequence_us,
        "tokens_per_second": args.batch_size * s_q / (full_call_us * 1.0e-6),
        "physical_kv_tb_s": (
            args.batch_size * physical_bytes_per_sequence / full_call_us / 1.0e6
        ),
        "model_tflops": args.batch_size * flops_per_sequence / full_call_us / 1.0e6,
        "measured_memory_lower_us": memory_lower_us,
        "measured_compute_lower_us": compute_lower_us,
        "measured_roofline_us": roofline_us,
        "measured_roofline_efficiency_percent": 100.0 * roofline_us / per_sequence_us,
    }


def run_point(
    *, args: argparse.Namespace, layout: str, g: int, s_q: int, page_size: int | None
) -> dict[str, Any]:
    print(
        f"ALLOC layout={layout} g={g} s_q={s_q} B={args.batch_size} "
        f"page={page_size}",
        flush=True,
    )
    inputs = make_inputs(
        batch_size=args.batch_size,
        g=g,
        s_q=s_q,
        page_size=page_size,
        seed=7200 + g * 10 + s_q + (0 if page_size is None else page_size),
    )
    modes = {
        "varlen_metadata_matched": make_varlen_callable(
            inputs,
            batch_size=args.batch_size,
            g=g,
            s_q=s_q,
            page_size=page_size,
        ),
        "fixed_matched": make_fixed_callable(inputs, num_splits=1, pack_gqa=True),
        "fixed_auto": make_fixed_callable(inputs, num_splits=0, pack_gqa=None),
    }

    outputs = {name: function() for name, function in modes.items()}
    torch.cuda.synchronize()
    correctness = {
        name: compare_outputs(output, outputs["varlen_metadata_matched"])
        for name, output in outputs.items()
    }
    if not all(item["allclose"] and item["finite"] for item in correctness.values()):
        raise RuntimeError(f"correctness failure: {correctness}")
    del outputs

    timings = benchmark_modes(modes, warmup=args.warmup, samples=args.samples)
    mode_results: dict[str, Any] = {}
    for name, timing in timings.items():
        metrics = derive_metrics(
            args=args, g=g, s_q=s_q, full_call_us=timing["median_us"]
        )
        mode_results[name] = {
            "timing": timing,
            "correctness_vs_varlen": correctness[name],
            "derived": metrics,
        }
        print(
            f"RESULT layout={layout} g={g} s_q={s_q} mode={name} "
            f"full_us={timing['median_us']:.3f} "
            f"per_seq_us={metrics['per_sequence_us']:.3f} "
            f"physical_tb_s={metrics['physical_kv_tb_s']:.3f}",
            flush=True,
        )

    varlen_us = timings["varlen_metadata_matched"]["median_us"]
    fixed_us = timings["fixed_matched"]["median_us"]
    fixed_auto_us = timings["fixed_auto"]["median_us"]
    result = {
        "layout": layout,
        "g": g,
        "s_q": s_q,
        "batch_size": args.batch_size,
        "context_len": CONTEXT_LEN,
        "page_size": page_size,
        "modes": mode_results,
        "fixed_matched_speedup_over_varlen": varlen_us / fixed_us,
        "fixed_auto_speedup_over_varlen": varlen_us / fixed_auto_us,
        "fixed_auto_speedup_over_fixed_matched": fixed_us / fixed_auto_us,
        "scheduler_paths_from_pinned_fa3_source": {
            "varlen_metadata_matched": "VarlenDynamicPersistentTileScheduler",
            "fixed_matched": (
                "StaticPersistentTileScheduler"
                if s_q == 1
                else "DynamicPersistentTileScheduler"
            ),
            "fixed_auto": "FA3 fixed-length runtime heuristic",
        },
    }
    del modes, inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    if args.expect_device_substring.lower() not in name.lower():
        raise RuntimeError(
            f"expected device containing {args.expect_device_substring!r}, got {name!r}"
        )
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 0):
        raise RuntimeError(f"expected SM90, got {capability}")

    result: dict[str, Any] = {
        "schema": "gqla-fa3-gqa-scheduler-ab-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": {
            "name": name,
            "compute_capability": list(capability),
            "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "fixed_constants": {
            "context_len": CONTEXT_LEN,
            "num_heads_q": NUM_HEADS_Q,
            "head_dim_qk": HEAD_DIM_QK,
            "head_dim_v": HEAD_DIM_V,
            "dtype": str(DTYPE),
            "softmax_scale": SOFTMAX_SCALE,
        },
        "points": [],
    }
    formal_pages = {(8, 1): 64, (8, 2): 128, (4, 1): 64, (4, 2): 128}
    for layout in ("paged", "dense"):
        for g in (8, 4):
            for s_q in (1, 2):
                page_size = formal_pages[(g, s_q)] if layout == "paged" else None
                result["points"].append(
                    run_point(
                        args=args,
                        layout=layout,
                        g=g,
                        s_q=s_q,
                        page_size=page_size,
                    )
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)

    print("\n=== FA3 GQA SCHEDULER A/B SUMMARY ===", flush=True)
    print(
        "layout g sq | varlen_us/seq fixed_us/seq auto_us/seq | "
        "fixed_speedup auto_speedup",
        flush=True,
    )
    for point in result["points"]:
        modes = point["modes"]
        varlen = modes["varlen_metadata_matched"]["derived"]["per_sequence_us"]
        fixed = modes["fixed_matched"]["derived"]["per_sequence_us"]
        auto = modes["fixed_auto"]["derived"]["per_sequence_us"]
        print(
            f"{point['layout']:5s} {point['g']:1d} {point['s_q']:2d} | "
            f"{varlen:14.3f} {fixed:12.3f} {auto:11.3f} | "
            f"{point['fixed_matched_speedup_over_varlen']:13.3f}x "
            f"{point['fixed_auto_speedup_over_varlen']:12.3f}x",
            flush=True,
        )
    print(f"SCHEDULER_AB_JSON={args.output.resolve()}", flush=True)
    print("FA3_GQA_SCHEDULER_AB_OK", flush=True)


if __name__ == "__main__":
    main()
