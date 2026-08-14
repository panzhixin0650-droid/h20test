#!/usr/bin/env python3
"""Time-bounded CUDA GEMM stress test with a temperature safety guard."""

from __future__ import annotations

import argparse
import gc
import math
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


STOP_REQUESTED = False


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"received signal {signum}; stopping after the current CUDA operation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated CUDA matrix multiplications for a bounded duration."
    )
    parser.add_argument("--duration", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--gpu-id", required=True, help="Physical GPU index for nvidia-smi")
    parser.add_argument("--max-temp", type=float, default=83.0)
    parser.add_argument("--resume-temp", type=float, default=78.0)
    parser.add_argument("--check-interval", type=float, default=10.0)
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=0,
        help="Square matrix size; 0 chooses a size from free memory",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.45,
        help="Maximum fraction of currently free memory used by the three matrices",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.resume_temp >= args.max_temp:
        parser.error("--resume-temp must be lower than --max-temp")
    if args.check_interval <= 0:
        parser.error("--check-interval must be positive")
    if args.matrix_size < 0:
        parser.error("--matrix-size cannot be negative")
    if not 0.10 <= args.memory_fraction <= 0.80:
        parser.error("--memory-fraction must be between 0.10 and 0.80")
    return args


def optional_float(value: str) -> float | None:
    value = value.strip()
    if value in {"N/A", "[N/A]", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def query_gpu_metrics(gpu_id: str) -> dict[str, float | None] | None:
    command = [
        "nvidia-smi",
        "-i",
        gpu_id,
        "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    line = result.stdout.strip().splitlines()
    if not line:
        return None
    fields = [field.strip() for field in line[0].split(",")]
    if len(fields) != 4:
        return None
    return {
        "temperature": optional_float(fields[0]),
        "power": optional_float(fields[1]),
        "utilization": optional_float(fields[2]),
        "memory_used": optional_float(fields[3]),
    }


def choose_matrix_size(
    requested_size: int,
    free_bytes: int,
    element_size: int,
    memory_fraction: float,
) -> int:
    if requested_size:
        return requested_size

    budget = int(free_bytes * memory_fraction)
    # A, B, and output C are all resident. Round to a Tensor-Core-friendly size.
    estimated = int(math.sqrt(budget / (3 * element_size)))
    estimated = (estimated // 256) * 256
    return max(2048, min(24576, estimated))


def allocate_matrices(torch: object, size: int, dtype: object) -> tuple[object, object, object, int]:
    current_size = size
    while current_size >= 2048:
        a = b = c = None
        try:
            a = torch.randn((current_size, current_size), device="cuda:0", dtype=dtype)
            b = torch.randn((current_size, current_size), device="cuda:0", dtype=dtype)
            c = torch.empty((current_size, current_size), device="cuda:0", dtype=dtype)
            torch.cuda.synchronize()
            return a, b, c, current_size
        except torch.OutOfMemoryError:
            a = b = c = None
            gc.collect()
            torch.cuda.empty_cache()
            next_size = (int(current_size * 0.80) // 256) * 256
            log(f"allocation failed at N={current_size}; retrying with N={next_size}")
            current_size = next_size
    raise RuntimeError("could not allocate stress-test matrices even at N=2048")


def format_metrics(metrics: dict[str, float | None]) -> str:
    temp = metrics["temperature"]
    power = metrics["power"]
    util = metrics["utilization"]
    memory = metrics["memory_used"]
    return (
        f"temp={temp:.0f}C " if temp is not None else "temp=N/A "
    ) + (
        f"power={power:.1f}W " if power is not None else "power=N/A "
    ) + (
        f"util={util:.0f}% " if util is not None else "util=N/A "
    ) + (
        f"memory={memory:.0f}MiB" if memory is not None else "memory=N/A"
    )


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        import torch
    except ImportError:
        log("PyTorch is not installed in this Python environment")
        return 2

    if not torch.cuda.is_available():
        log("CUDA is not available to PyTorch; check the driver and CUDA_VISIBLE_DEVICES")
        return 2

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    properties = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dtype_name = str(dtype).removeprefix("torch.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    element_size = torch.empty((), dtype=dtype).element_size()
    requested_size = choose_matrix_size(
        args.matrix_size, free_bytes, element_size, args.memory_fraction
    )

    initial_metrics = query_gpu_metrics(args.gpu_id)
    if initial_metrics is None or initial_metrics["temperature"] is None:
        log(
            f"cannot read the temperature of physical GPU {args.gpu_id} with nvidia-smi; "
            "refusing to run without the safety guard"
        )
        return 2

    log(
        f"starting on physical GPU {args.gpu_id}: {properties.name}; "
        f"dtype={dtype_name}; free={free_bytes / 2**30:.1f}GiB/"
        f"{total_bytes / 2**30:.1f}GiB; duration={args.duration}s"
    )
    log(
        f"thermal guard: pause at {args.max_temp:.0f}C, resume at "
        f"{args.resume_temp:.0f}C; {format_metrics(initial_metrics)}"
    )

    if initial_metrics["temperature"] >= args.max_temp:
        log(
            f"GPU is already at {initial_metrics['temperature']:.0f}C; waiting for "
            f"it to cool to {args.resume_temp:.0f}C before allocating matrices"
        )
        while not STOP_REQUESTED:
            time.sleep(30.0)
            initial_metrics = query_gpu_metrics(args.gpu_id)
            if initial_metrics is None or initial_metrics["temperature"] is None:
                log("lost temperature monitoring while waiting for initial cooldown")
                return 2
            log(f"initial cooling: {format_metrics(initial_metrics)}")
            if initial_metrics["temperature"] <= args.resume_temp:
                break
        if STOP_REQUESTED:
            log("stopped during initial cooldown")
            return 0

    a = b = c = None
    try:
        a, b, c, matrix_size = allocate_matrices(torch, requested_size, dtype)
        gib = 3 * matrix_size * matrix_size * element_size / 2**30
        log(f"allocated three {matrix_size}x{matrix_size} matrices ({gib:.2f}GiB total)")

        # Warm-up initializes CUDA libraries before the timed interval.
        for _ in range(3):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()

        started = time.monotonic()
        deadline = started + args.duration
        last_check = started
        last_rate_time = started
        last_rate_iterations = 0
        iterations = 0
        monitor_failures = 0

        while not STOP_REQUESTED and time.monotonic() < deadline:
            torch.mm(a, b, out=c)
            iterations += 1
            # Keep the CUDA queue short so the thermal guard can react promptly.
            if iterations % 8 != 0:
                continue
            torch.cuda.synchronize()
            now = time.monotonic()
            if now - last_check < args.check_interval:
                continue

            metrics = query_gpu_metrics(args.gpu_id)
            if metrics is None or metrics["temperature"] is None:
                monitor_failures += 1
                log(f"temperature query failed ({monitor_failures}/3)")
                if monitor_failures >= 3:
                    raise RuntimeError("temperature monitoring failed three times; stopping safely")
                last_check = now
                continue

            monitor_failures = 0
            interval = max(now - last_rate_time, 1e-9)
            interval_iterations = iterations - last_rate_iterations
            tflops = (
                interval_iterations * 2.0 * matrix_size**3 / interval / 1e12
            )
            elapsed = now - started
            remaining = max(0.0, deadline - now)
            log(
                f"elapsed={elapsed / 3600:.2f}h remaining={remaining / 3600:.2f}h "
                f"iterations={iterations} approx={tflops:.1f}TFLOP/s "
                f"{format_metrics(metrics)}"
            )
            last_check = now
            last_rate_time = now
            last_rate_iterations = iterations

            if metrics["temperature"] >= args.max_temp:
                log(
                    f"temperature reached {metrics['temperature']:.0f}C; pausing compute "
                    f"until it falls to {args.resume_temp:.0f}C"
                )
                while not STOP_REQUESTED and time.monotonic() < deadline:
                    time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))
                    metrics = query_gpu_metrics(args.gpu_id)
                    if metrics is None or metrics["temperature"] is None:
                        raise RuntimeError("lost temperature monitoring while thermally paused")
                    log(f"cooling: {format_metrics(metrics)}")
                    if metrics["temperature"] <= args.resume_temp:
                        log("temperature recovered; resuming compute")
                        break
                now = time.monotonic()
                last_check = now
                last_rate_time = now
                last_rate_iterations = iterations

        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        reason = "signal requested" if STOP_REQUESTED else "duration reached"
        checksum = float(c[0, 0].item())
        log(
            f"finished: {reason}; elapsed={elapsed / 3600:.2f}h; "
            f"iterations={iterations}; checksum={checksum:.6g}"
        )
        return 0
    except Exception as error:
        log(f"fatal error: {type(error).__name__}: {error}")
        return 1
    finally:
        a = b = c = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
