#!/usr/bin/env python3
"""Measure sustained HBM traffic and dense BF16 Tensor Core throughput.

This is an independent hardware baseline for the GQLA Table 2 kernel results.
It deliberately does not import FlashMLA, FlashAttention, or vLLM.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import triton
import triton.language as tl


GIB = 1 << 30


@triton.jit
def _read_only_kernel(
    source,
    sink,
    n_elements,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """Read every FP32 input once and emit one checksum per VEC inputs."""
    program = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    chunk_start = program * BLOCK * VEC
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for vector_index in tl.static_range(0, VEC):
        offsets = chunk_start + vector_index * BLOCK + lanes
        values = tl.load(source + offsets, mask=offsets < n_elements, other=0.0)
        accumulator += values
    tl.store(sink + program * BLOCK + lanes, accumulator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent H20 sustained-bandwidth and BF16-GEMM baseline"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-device-substring", default="H20")
    parser.add_argument("--read-gib", type=float, default=4.0)
    parser.add_argument("--copy-gib", type=float, default=2.0)
    parser.add_argument("--stream-gib", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--bandwidth-inner", type=int, default=5)
    parser.add_argument("--gemm-warmup", type=int, default=3)
    parser.add_argument("--gemm-samples", type=int, default=9)
    parser.add_argument(
        "--gemm-sizes",
        default="8192,12288,16384",
        help="Comma-separated square BF16 GEMM dimensions",
    )
    parser.add_argument("--expected-hbm-tb-s", type=float, default=4.0)
    parser.add_argument("--expected-bf16-tflops", type=float, default=148.0)
    parser.add_argument(
        "--fa3-gqa-equivalent-tb-s",
        type=float,
        default=1.03,
        help="Previously observed stock-FA3 physical-cache equivalent bandwidth",
    )
    args = parser.parse_args()
    for name in ("read_gib", "copy_gib", "stream_gib"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "warmup",
        "samples",
        "bandwidth_inner",
        "gemm_warmup",
        "gemm_samples",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        args.gemm_sizes = [int(item) for item in args.gemm_sizes.split(",")]
    except ValueError as error:
        parser.error(f"invalid --gemm-sizes: {error}")
    if not args.gemm_sizes or any(size <= 0 for size in args.gemm_sizes):
        parser.error("--gemm-sizes must contain positive integers")
    return args


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "count": len(samples_ms),
        "min_ms": min(samples_ms),
        "p20_ms": percentile(samples_ms, 0.20),
        "median_ms": statistics.median(samples_ms),
        "p80_ms": percentile(samples_ms, 0.80),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def benchmark_cuda(
    function: Callable[[], Any], *, warmup: int, samples: int, inner: int
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    timings_ms: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            function()
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end) / inner)
    return timing_summary(timings_ms)


def bytes_from_gib(value: float, alignment: int) -> int:
    byte_count = int(value * GIB)
    return byte_count - byte_count % alignment


def require_free_memory(required_bytes: int, label: str) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < int(required_bytes * 1.10):
        raise RuntimeError(
            f"{label} needs about {required_bytes / GIB:.2f} GiB, but only "
            f"{free_bytes / GIB:.2f}/{total_bytes / GIB:.2f} GiB is free"
        )


def release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": repr(error)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def nvidia_smi_snapshot() -> dict[str, Any]:
    return run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id,driver_version,pstate,"
            "clocks.current.sm,clocks.current.memory,clocks.max.sm,"
            "clocks.max.memory,power.draw,power.limit,temperature.gpu,"
            "memory.total,memory.used,ecc.mode.current,mig.mode.current",
            "--format=csv,noheader,nounits",
        ]
    )


def measure_read_only(args: argparse.Namespace) -> dict[str, Any]:
    byte_count = bytes_from_gib(args.read_gib, 4)
    n_elements = byte_count // 4
    variants = [
        {"block": 1024, "vec": 32, "num_warps": 8},
        {"block": 2048, "vec": 32, "num_warps": 8},
        {"block": 4096, "vec": 32, "num_warps": 8},
        {"block": 4096, "vec": 64, "num_warps": 8},
    ]
    largest_sink_elements = max(
        triton.cdiv(n_elements, item["block"] * item["vec"]) * item["block"]
        for item in variants
    )
    require_free_memory(byte_count + largest_sink_elements * 4, "read-only test")
    print(f"[HBM read] allocating {byte_count / GIB:.2f} GiB input", flush=True)
    source = torch.empty(n_elements, device="cuda", dtype=torch.float32)
    source.fill_(1.0)
    torch.cuda.synchronize()

    results: list[dict[str, Any]] = []
    for variant in variants:
        block = variant["block"]
        vec = variant["vec"]
        num_warps = variant["num_warps"]
        grid = triton.cdiv(n_elements, block * vec)
        sink_elements = grid * block
        sink = torch.empty(sink_elements, device="cuda", dtype=torch.float32)

        def launch() -> None:
            _read_only_kernel[(grid,)](
                source,
                sink,
                n_elements,
                BLOCK=block,
                VEC=vec,
                num_warps=num_warps,
                num_stages=1,
            )

        print(
            f"[HBM read] BLOCK={block} VEC={vec} warps={num_warps}",
            flush=True,
        )
        try:
            timing = benchmark_cuda(
                launch,
                warmup=args.warmup,
                samples=args.samples,
                inner=args.bandwidth_inner,
            )
            checksum = float(sink[0].item())
            if not math.isfinite(checksum) or abs(checksum - vec) > 0.01:
                raise RuntimeError(
                    f"read-only checksum mismatch: got {checksum}, expected {vec}"
                )
            median_seconds = timing["median_ms"] / 1_000.0
            input_tb_s = byte_count / median_seconds / 1.0e12
            accounted_bytes = byte_count + sink_elements * 4
            result = {
                **variant,
                "status": "complete",
                "grid": grid,
                "input_bytes": byte_count,
                "sink_bytes": sink_elements * 4,
                "timing": timing,
                "input_read_tb_s": input_tb_s,
                "input_plus_sink_tb_s": accounted_bytes / median_seconds / 1.0e12,
                "checksum": checksum,
            }
            print(
                f"[HBM read] median={timing['median_ms']:.4f} ms, "
                f"input={input_tb_s:.3f} TB/s",
                flush=True,
            )
        except Exception as error:  # Keep other variants usable.
            result = {**variant, "status": "failed", "error": repr(error)}
            print(f"[HBM read] variant failed: {error!r}", flush=True)
        results.append(result)
        del sink
        release_cuda_memory()

    del source
    release_cuda_memory()
    completed = [item for item in results if item["status"] == "complete"]
    if not completed:
        raise RuntimeError("all Triton read-only variants failed")
    best = max(completed, key=lambda item: item["input_read_tb_s"])
    return {
        "description": "SM-issued read-only scan; primary rate counts input bytes only",
        "variants": results,
        "best": best,
    }


def measure_copy(args: argparse.Namespace) -> dict[str, Any]:
    byte_count = bytes_from_gib(args.copy_gib, 256)
    require_free_memory(byte_count * 2, "device-to-device copy test")
    print(f"[D2D copy] allocating 2 x {byte_count / GIB:.2f} GiB", flush=True)
    source = torch.empty(byte_count, device="cuda", dtype=torch.uint8)
    destination = torch.empty_like(source)
    source.fill_(17)

    def launch() -> None:
        destination.copy_(source)

    timing = benchmark_cuda(
        launch,
        warmup=args.warmup,
        samples=args.samples,
        inner=args.bandwidth_inner,
    )
    checksum = int(destination[0].item())
    if checksum != 17:
        raise RuntimeError(f"copy checksum mismatch: {checksum}")
    median_seconds = timing["median_ms"] / 1_000.0
    result = {
        "description": "contiguous CUDA device-to-device copy",
        "payload_bytes": byte_count,
        "timing": timing,
        "payload_tb_s": byte_count / median_seconds / 1.0e12,
        "read_plus_write_tb_s": 2 * byte_count / median_seconds / 1.0e12,
        "checksum": checksum,
    }
    print(
        f"[D2D copy] median={timing['median_ms']:.4f} ms, "
        f"R+W={result['read_plus_write_tb_s']:.3f} TB/s",
        flush=True,
    )
    del source, destination
    release_cuda_memory()
    return result


def measure_stream_add(args: argparse.Namespace) -> dict[str, Any]:
    byte_count = bytes_from_gib(args.stream_gib, 4)
    n_elements = byte_count // 4
    require_free_memory(byte_count * 3, "STREAM add test")
    print(f"[STREAM add] allocating 3 x {byte_count / GIB:.2f} GiB", flush=True)
    left = torch.empty(n_elements, device="cuda", dtype=torch.float32)
    right = torch.empty_like(left)
    output = torch.empty_like(left)
    left.fill_(1.25)
    right.fill_(2.50)

    def launch() -> None:
        torch.add(left, right, out=output)

    timing = benchmark_cuda(
        launch,
        warmup=args.warmup,
        samples=args.samples,
        inner=args.bandwidth_inner,
    )
    checksum = float(output[0].item())
    if abs(checksum - 3.75) > 1.0e-6:
        raise RuntimeError(f"STREAM add checksum mismatch: {checksum}")
    median_seconds = timing["median_ms"] / 1_000.0
    result = {
        "description": "FP32 elementwise add; two reads plus one write",
        "bytes_per_tensor": byte_count,
        "accounted_bytes": 3 * byte_count,
        "timing": timing,
        "read_plus_write_tb_s": 3 * byte_count / median_seconds / 1.0e12,
        "checksum": checksum,
    }
    print(
        f"[STREAM add] median={timing['median_ms']:.4f} ms, "
        f"2R+W={result['read_plus_write_tb_s']:.3f} TB/s",
        flush=True,
    )
    del left, right, output
    release_cuda_memory()
    return result


def measure_bf16_gemm(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    results: list[dict[str, Any]] = []
    for size in args.gemm_sizes:
        matrix_bytes = size * size * 2
        require_free_memory(matrix_bytes * 3, f"BF16 GEMM n={size}")
        print(f"[BF16 GEMM] n={size}", flush=True)
        left = torch.empty((size, size), device="cuda", dtype=torch.bfloat16)
        right = torch.empty_like(left)
        output = torch.empty_like(left)
        left.fill_(0.25)
        right.fill_(0.50)

        def launch() -> None:
            torch.mm(left, right, out=output)

        timing = benchmark_cuda(
            launch,
            warmup=args.gemm_warmup,
            samples=args.gemm_samples,
            inner=1,
        )
        checksum = float(output[0, 0].item())
        expected = size * 0.125
        if not math.isfinite(checksum) or abs(checksum - expected) > max(1.0, expected * 0.02):
            raise RuntimeError(
                f"BF16 GEMM checksum mismatch for n={size}: got {checksum}, "
                f"expected approximately {expected}"
            )
        flops = 2 * size * size * size
        median_seconds = timing["median_ms"] / 1_000.0
        tflops = flops / median_seconds / 1.0e12
        result = {
            "m": size,
            "n": size,
            "k": size,
            "dtype": "torch.bfloat16",
            "flops": flops,
            "timing": timing,
            "tflops": tflops,
            "checksum": checksum,
        }
        results.append(result)
        print(
            f"[BF16 GEMM] median={timing['median_ms']:.4f} ms, "
            f"{tflops:.2f} TFLOP/s",
            flush=True,
        )
        del left, right, output
        release_cuda_memory()
    return {
        "description": "dense square torch.mm BF16 Tensor Core baseline",
        "sizes": results,
        "best": max(results, key=lambda item: item["tflops"]),
    }


def device_metadata() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    metadata: dict[str, Any] = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "multi_processor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "l2_cache_size_bytes": getattr(properties, "L2_cache_size", None),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        target = triton.runtime.driver.active.get_current_target()
        metadata["triton_target"] = str(target)
    except Exception as error:
        metadata["triton_target_error"] = repr(error)
    return metadata


def build_diagnosis(
    args: argparse.Namespace,
    read_tb_s: float,
    copy_tb_s: float,
    stream_tb_s: float,
    gemm_tflops: float,
) -> dict[str, Any]:
    read_fraction = read_tb_s / args.expected_hbm_tb_s
    copy_fraction = copy_tb_s / args.expected_hbm_tb_s
    stream_fraction = stream_tb_s / args.expected_hbm_tb_s
    compute_fraction = gemm_tflops / args.expected_bf16_tflops
    fa3_fraction_of_measured_read = args.fa3_gqa_equivalent_tb_s / read_tb_s

    if max(read_fraction, copy_fraction, stream_fraction) >= 0.70:
        bandwidth_assessment = (
            "At least one independent bandwidth path reaches >=70% of the expected HBM rate. "
            "If FA3 remains near 1.03 TB/s, the dominant loss is likely in the FA3 GQA "
            "specialization/scheduling rather than the node's global HBM capability."
        )
    elif max(read_tb_s, copy_tb_s, stream_tb_s) <= 1.50:
        bandwidth_assessment = (
            "All bandwidth paths are <=1.50 TB/s. Check GPU contention, clocks, power limit, "
            "ECC events, virtualization/partitioning, and node health before blaming FA3."
        )
    else:
        bandwidth_assessment = (
            "The independent bandwidth paths are mixed or materially below the expected rate. "
            "Inspect the raw samples and collect Nsight Compute DRAM counters on FA3 before "
            "assigning a single cause."
        )

    if compute_fraction >= 0.70:
        compute_assessment = "Dense BF16 Tensor Core throughput is broadly healthy."
    else:
        compute_assessment = (
            "Dense BF16 throughput is below 70% of the expected value; check clocks, power, "
            "contention, and the installed CUDA/PyTorch GEMM stack."
        )
    return {
        "expected_hbm_tb_s": args.expected_hbm_tb_s,
        "expected_bf16_tflops": args.expected_bf16_tflops,
        "read_percent_of_expected": 100.0 * read_fraction,
        "copy_aggregate_percent_of_expected": 100.0 * copy_fraction,
        "stream_aggregate_percent_of_expected": 100.0 * stream_fraction,
        "gemm_percent_of_expected": 100.0 * compute_fraction,
        "fa3_gqa_equivalent_tb_s": args.fa3_gqa_equivalent_tb_s,
        "fa3_percent_of_measured_sm_read": 100.0 * fa3_fraction_of_measured_read,
        "bandwidth_assessment": bandwidth_assessment,
        "compute_assessment": compute_assessment,
        "caveat": (
            "These are independent sustained microbenchmarks. They do not replace Nsight "
            "Compute dram__bytes_read.sum / dram__throughput measurements of the FA3 kernel."
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = device_metadata()
    if args.expect_device_substring.lower() not in device["name"].lower():
        raise RuntimeError(
            f"expected device containing {args.expect_device_substring!r}, got {device['name']!r}"
        )
    if tuple(device["compute_capability"]) != (9, 0):
        raise RuntimeError(f"expected Hopper SM90, got {device['compute_capability']}")

    print(json.dumps(device, ensure_ascii=False, indent=2), flush=True)
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    result: dict[str, Any] = {
        "schema": "gqla-hardware-roofline-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": device,
        "nvidia_smi_before": nvidia_smi_snapshot(),
    }

    wall_start = time.monotonic()
    result["hbm_read_only"] = measure_read_only(args)
    result["nvidia_smi_after_read"] = nvidia_smi_snapshot()
    result["device_to_device_copy"] = measure_copy(args)
    result["stream_add"] = measure_stream_add(args)
    result["bf16_gemm"] = measure_bf16_gemm(args)
    result["nvidia_smi_after"] = nvidia_smi_snapshot()
    result["elapsed_seconds"] = time.monotonic() - wall_start

    read_tb_s = result["hbm_read_only"]["best"]["input_read_tb_s"]
    copy_tb_s = result["device_to_device_copy"]["read_plus_write_tb_s"]
    stream_tb_s = result["stream_add"]["read_plus_write_tb_s"]
    gemm_tflops = result["bf16_gemm"]["best"]["tflops"]
    result["summary"] = {
        "sustained_sm_read_tb_s": read_tb_s,
        "d2d_copy_read_plus_write_tb_s": copy_tb_s,
        "stream_add_two_read_one_write_tb_s": stream_tb_s,
        "best_dense_bf16_tflops": gemm_tflops,
    }
    result["diagnosis"] = build_diagnosis(
        args, read_tb_s, copy_tb_s, stream_tb_s, gemm_tflops
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)

    print("\n=== HARDWARE ROOFLINE SUMMARY ===", flush=True)
    print(f"SM read-only:       {read_tb_s:.3f} TB/s", flush=True)
    print(f"D2D copy (R+W):     {copy_tb_s:.3f} TB/s", flush=True)
    print(f"STREAM add (2R+W):  {stream_tb_s:.3f} TB/s", flush=True)
    print(f"Dense BF16 GEMM:    {gemm_tflops:.2f} TFLOP/s", flush=True)
    print(
        "FA3 1.03 TB/s / measured SM read: "
        f"{result['diagnosis']['fa3_percent_of_measured_sm_read']:.2f}%",
        flush=True,
    )
    print(result["diagnosis"]["bandwidth_assessment"], flush=True)
    print(result["diagnosis"]["compute_assessment"], flush=True)
    print(f"ROOFLINE_JSON={args.output.resolve()}", flush=True)
    print("HARDWARE_ROOFLINE_OK", flush=True)


if __name__ == "__main__":
    main()
