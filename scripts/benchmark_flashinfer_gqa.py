#!/usr/bin/env python3
"""Benchmark FlashInfer on the short-query GQA shapes from GQLA Table 2.

The benchmark intentionally uses ``BatchPrefillWithPagedKVCacheWrapper`` for
both S_Q=1 and S_Q=2.  Unlike FlashInfer's public batch-decode ``plan`` API,
the batch-prefill API accepts different QK and VO head dimensions, which is
required by this workload (D_QK=192, D_V=128).

The canonical backend is explicitly ``fa2``.  On Hopper, ``backend="auto"``
can select FlashInfer's FA3 path and reintroduce a large short-query M tile.
FlashInfer FA2 instead selects M16 for 16 packed query rows and M64 for 32/64
packed rows in the pinned 0.6.11 implementation.

K and V are compact views of one interleaved ``[..., D_QK + D_V]`` backing
tensor.  This gives K and V equal non-last-dimension strides, as required by
FlashInfer for unequal head dimensions, without padding either cache.  The
physical cache payload is therefore still exactly
``L * H_KV * (D_QK + D_V) * sizeof(BF16)`` per sequence.

Allocation, random initialization, wrapper planning, and first-use JIT are
outside the timed callable.  Timing uses ``triton.testing.do_bench``, matching
the existing Table 2 harness: CUDA events measure the full operator call and
Triton evicts L2 before timed invocations.  The output tensor is preallocated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


NUM_HEADS_Q = 128
HEAD_DIM_QK = 192
HEAD_DIM_V = 128
DTYPE = torch.bfloat16
BYTES_PER_ELEMENT = 2
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM_QK)
MIB = 1 << 20


@dataclass
class PreparedPoint:
    backend_requested: str
    backend_resolved: str
    wrapper: Any
    fn: Callable[[], torch.Tensor]
    output: torch.Tensor
    workspace: torch.Tensor
    plan_ms: float
    plan_details: dict[str, Any]


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in csv_items(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected a non-empty list of positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", default="local-h100-role")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--expect-device-substring",
        action="append",
        default=None,
        help=(
            "allowed torch CUDA device-name substring; repeat to accept aliases "
            "such as H100 and L20Z"
        ),
    )
    parser.add_argument(
        "--backends",
        default="fa2",
        help="comma-separated FlashInfer backends (fa2, fa3, or auto); default: fa2",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--g-values", type=csv_ints, default=[8, 4])
    parser.add_argument("--sq-values", type=csv_ints, default=[1, 2])
    parser.add_argument(
        "--page-size-sq1",
        type=int,
        default=64,
        help="page size used when S_Q=1 (the Table 2 protocol uses 64)",
    )
    parser.add_argument(
        "--page-size-sq2",
        type=int,
        default=128,
        help="page size used when S_Q=2 (the Table 2 protocol uses 128)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="override the page size for every S_Q",
    )
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument(
        "--disable-split-kv",
        action="store_true",
        help="disable FlashInfer FA2 split-KV planning (auto planning is the default)",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="skip the request-0 float32 PyTorch reference check",
    )
    parser.add_argument("--rtol", type=float, default=2.01 / 128)
    parser.add_argument("--atol", type=float, default=8.0e-4)
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="run a separate steady-state Kineto pass and record CUDA kernel names",
    )
    parser.add_argument("--profile-calls", type=int, default=5)
    parser.add_argument(
        "--measured-hbm-tb-s",
        type=float,
        default=None,
        help="optional sustained HBM read bandwidth for a per-sequence Roofline estimate",
    )
    parser.add_argument(
        "--measured-bf16-tflops",
        type=float,
        default=None,
        help="optional sustained dense BF16 throughput for a tile-aware Roofline estimate",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    backends = csv_items(args.backends)
    if not backends:
        parser.error("--backends must not be empty")
    invalid_backends = sorted(set(backends) - {"fa2", "fa3", "auto"})
    if invalid_backends:
        parser.error(f"unsupported backends: {invalid_backends}")
    args.backends = backends

    positive_names = (
        "batch_size",
        "context_len",
        "page_size_sq1",
        "page_size_sq2",
        "workspace_mib",
        "warmup_ms",
        "rep_ms",
        "rounds",
        "profile_calls",
    )
    if any(getattr(args, name) <= 0 for name in positive_names):
        parser.error("batch/shape/workspace/timing/profile arguments must be positive")
    if args.page_size is not None and args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.rtol < 0 or args.atol < 0:
        parser.error("--rtol and --atol must be non-negative")
    for value, name in (
        (args.measured_hbm_tb_s, "--measured-hbm-tb-s"),
        (args.measured_bf16_tflops, "--measured-bf16-tflops"),
    ):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if NUM_HEADS_Q % min(args.g_values) != 0:
        parser.error("every G value must divide NUM_HEADS_Q=128")
    for g in args.g_values:
        if NUM_HEADS_Q % g != 0:
            parser.error(f"G={g} does not divide NUM_HEADS_Q=128")
    return args


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    samples_us = [value * 1_000.0 for value in samples_ms]
    return {
        "count": len(samples_us),
        "min_us": min(samples_us),
        "p20_us": quantile(samples_us, 0.20),
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "p80_us": quantile(samples_us, 0.80),
        "max_us": max(samples_us),
        "samples_us": samples_us,
    }


def page_size_for(args: argparse.Namespace, s_q: int) -> int:
    if args.page_size is not None:
        return args.page_size
    if s_q == 1:
        return args.page_size_sq1
    if s_q == 2:
        return args.page_size_sq2
    # Non-canonical diagnostic points use the S_Q=2 page size unless explicitly
    # overridden.  The selected value is always recorded in JSON.
    return args.page_size_sq2


def default_output_path(label: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-").lower()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return root / "outputs" / "official" / "flashinfer_gqa" / f"{safe_label}-{timestamp}.json"


def device_metadata(device: torch.device, label: str) -> dict[str, Any]:
    import flashinfer
    import triton

    props = torch.cuda.get_device_properties(device)
    flashinfer_git = None
    try:
        from flashinfer import _build_meta

        flashinfer_git = getattr(_build_meta, "__git_version__", None)
    except Exception:
        pass
    return {
        "label": label,
        "name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "multi_processor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "flashinfer_git": flashinfer_git,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def allocate_inputs(
    *, args: argparse.Namespace, device: torch.device, g: int, s_q: int, page_size: int
) -> dict[str, Any]:
    batch_size = args.batch_size
    context_len = args.context_len
    pages_per_sequence = math.ceil(context_len / page_size)
    num_pages = batch_size * pages_per_sequence
    allocated_tokens_per_sequence = pages_per_sequence * page_size
    allocated_cache_bytes = (
        batch_size
        * allocated_tokens_per_sequence
        * g
        * (HEAD_DIM_QK + HEAD_DIM_V)
        * BYTES_PER_ELEMENT
    )
    q_bytes = batch_size * s_q * NUM_HEADS_Q * HEAD_DIM_QK * BYTES_PER_ELEMENT
    workspace_bytes = args.workspace_mib * MIB
    free_bytes, _ = torch.cuda.mem_get_info(device)
    required_bytes = allocated_cache_bytes + q_bytes + workspace_bytes
    # Leave room for FlashInfer's internal 8 MiB workspace, outputs, JIT, the
    # sampled float32 reference, and Triton's L2 eviction buffer.
    reserve_bytes = 768 * MIB
    if required_bytes + reserve_bytes > free_bytes:
        raise RuntimeError(
            "insufficient free GPU memory: "
            f"need about {(required_bytes + reserve_bytes) / MIB:.1f} MiB, "
            f"have {free_bytes / MIB:.1f} MiB"
        )

    point_seed = args.seed + 1000 * g + 10 * s_q + page_size
    torch.manual_seed(point_seed)
    q = torch.empty(
        (batch_size * s_q, NUM_HEADS_Q, HEAD_DIM_QK),
        device=device,
        dtype=DTYPE,
    )
    q.normal_(mean=0.0, std=0.1)

    # Compact K/V interleaving.  Both views have identical leading strides,
    # while their last dimensions remain 192 and 128 respectively.
    kv_storage = torch.empty(
        (num_pages, page_size, g, HEAD_DIM_QK + HEAD_DIM_V),
        device=device,
        dtype=DTYPE,
    )
    kv_storage.normal_(mean=0.0, std=0.1)
    k_cache = kv_storage[..., :HEAD_DIM_QK]
    v_cache = kv_storage[..., HEAD_DIM_QK:]
    if k_cache.stride()[:-1] != v_cache.stride()[:-1]:
        raise RuntimeError(
            f"internal error: K/V leading strides differ: {k_cache.stride()} vs {v_cache.stride()}"
        )

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * s_q
    paged_kv_indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32) * pages_per_sequence
    )
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32)
    last_page_len_value = context_len - (pages_per_sequence - 1) * page_size
    paged_kv_last_page_len = torch.full(
        (batch_size,), last_page_len_value, dtype=torch.int32
    )
    torch.cuda.synchronize(device)

    return {
        "q": q,
        "kv_storage": kv_storage,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "paged_kv_indptr": paged_kv_indptr,
        "paged_kv_indices": paged_kv_indices,
        "paged_kv_last_page_len": paged_kv_last_page_len,
        "pages_per_sequence": pages_per_sequence,
        "num_pages": num_pages,
        "last_page_len": last_page_len_value,
        "allocated_tokens_per_sequence": allocated_tokens_per_sequence,
        "point_seed": point_seed,
    }


def decode_fa2_plan_details(plan_info: Any) -> dict[str, Any]:
    try:
        values = [int(value) for value in plan_info]
    except Exception:
        return {"raw_repr": repr(plan_info)}
    result: dict[str, Any] = {"raw": values}
    if len(values) == 15:
        result.update(
            {
                "padded_batch_size": values[0],
                "total_num_rows": values[1],
                "cta_tile_q": values[3],
                "cuda_graph_enabled": bool(values[13]),
                "split_kv": bool(values[14]),
            }
        )
    return result


def prepare_backend(
    *,
    args: argparse.Namespace,
    device: torch.device,
    inputs: dict[str, Any],
    backend: str,
    g: int,
    page_size: int,
) -> PreparedPoint:
    import flashinfer

    workspace = torch.zeros(
        args.workspace_mib * MIB, dtype=torch.uint8, device=device
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend=backend
    )

    torch.cuda.synchronize(device)
    plan_start = time.perf_counter()
    wrapper.plan(
        inputs["qo_indptr"],
        inputs["paged_kv_indptr"],
        inputs["paged_kv_indices"],
        inputs["paged_kv_last_page_len"],
        NUM_HEADS_Q,
        g,
        HEAD_DIM_QK,
        page_size,
        head_dim_vo=HEAD_DIM_V,
        causal=True,
        pos_encoding_mode="NONE",
        sm_scale=SOFTMAX_SCALE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        o_data_type=DTYPE,
        disable_split_kv=args.disable_split_kv,
    )
    torch.cuda.synchronize(device)
    plan_ms = (time.perf_counter() - plan_start) * 1_000.0

    output = torch.empty(
        (args.batch_size * int(inputs["qo_indptr"][1]), NUM_HEADS_Q, HEAD_DIM_V),
        device=device,
        dtype=DTYPE,
    )

    def fn() -> torch.Tensor:
        return wrapper.run(
            inputs["q"],
            (inputs["k_cache"], inputs["v_cache"]),
            out=output,
        )

    fn()
    torch.cuda.synchronize(device)
    if not torch.isfinite(output).all().item():
        raise RuntimeError("FlashInfer smoke output contains NaN or Inf")
    backend_resolved = str(getattr(wrapper, "_backend", backend))
    plan_details = decode_fa2_plan_details(getattr(wrapper, "_plan_info", None))
    return PreparedPoint(
        backend_requested=backend,
        backend_resolved=backend_resolved,
        wrapper=wrapper,
        fn=fn,
        output=output,
        workspace=workspace,
        plan_ms=plan_ms,
        plan_details=plan_details,
    )


def pytorch_reference_check(
    *,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    actual: torch.Tensor,
    g: int,
    s_q: int,
) -> dict[str, Any]:
    pages_per_sequence = inputs["pages_per_sequence"]
    context_len = args.context_len
    group_size = NUM_HEADS_Q // g

    q = inputs["q"][:s_q].float().view(s_q, g, group_size, HEAD_DIM_QK)
    compact_request_kv = inputs["kv_storage"][:pages_per_sequence].reshape(
        -1, g, HEAD_DIM_QK + HEAD_DIM_V
    )[:context_len]
    k = compact_request_kv[..., :HEAD_DIM_QK].float()
    v = compact_request_kv[..., HEAD_DIM_QK:].float()
    scores = torch.einsum("sgrd,lgd->sgrl", q, k) * SOFTMAX_SCALE

    # FlashInfer uses bottom-right causal alignment when Q is shorter than KV:
    # query row i corresponds to absolute position L-S_Q+i.
    q_positions = torch.arange(
        context_len - s_q,
        context_len,
        device=scores.device,
    )
    k_positions = torch.arange(context_len, device=scores.device)
    causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    scores.masked_fill_(~causal_mask[:, None, None, :], -math.inf)
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.einsum("sgrl,lgv->sgrv", probabilities, v).reshape(
        s_q, NUM_HEADS_Q, HEAD_DIM_V
    )
    actual_float = actual[:s_q].float()
    difference = (actual_float - reference).abs()
    tolerance = args.atol + args.rtol * reference.abs()
    within = difference <= tolerance
    result = {
        "scope": "request 0, every Q head and query row, float32 PyTorch reference",
        "causal_alignment": "bottom-right",
        "rtol": args.rtol,
        "atol": args.atol,
        "finite": bool(torch.isfinite(actual_float).all().item()),
        "allclose": bool(torch.allclose(actual_float, reference, rtol=args.rtol, atol=args.atol)),
        "within_tolerance_fraction": float(within.float().mean().item()),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "reference_abs_max": float(reference.abs().max().item()),
        "reference_abs_mean": float(reference.abs().mean().item()),
    }
    del q, compact_request_kv, k, v, scores, probabilities, reference
    return result


def benchmark_callable(
    fn: Callable[[], torch.Tensor], *, warmup_ms: int, rep_ms: int, rounds: int
) -> dict[str, Any]:
    from triton.testing import do_bench

    all_samples_ms: list[float] = []
    for _ in range(rounds):
        round_samples = do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="all",
        )
        all_samples_ms.extend(float(value) for value in round_samples)
    return timing_summary(all_samples_ms)


def profile_cuda_kernels(
    fn: Callable[[], torch.Tensor], *, calls: int
) -> dict[str, Any]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()

    cuda_type = torch.autograd.DeviceType.CUDA
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        if event.device_type != cuda_type or event.self_device_time_total <= 0:
            continue
        rows.append(
            {
                "name": str(event.key),
                "count": int(event.count),
                "launches_per_call": int(event.count) / calls,
                "us_per_call": float(event.self_device_time_total) / calls,
                "us_per_launch": float(event.self_device_time_total) / int(event.count),
            }
        )
    rows.sort(key=lambda item: item["us_per_call"], reverse=True)
    return {
        "calls": calls,
        "cache_state": (
            "steady-state without an explicit profiler-side L2 flush; the canonical "
            "full-call timing uses triton.testing.do_bench"
        ),
        "cuda_kernel_sum_us_per_call": sum(item["us_per_call"] for item in rows),
        "cuda_kernels": rows,
    }


def fa2_expected_cta_tile_q(packed_query_rows: int) -> int:
    if packed_query_rows > 64 and HEAD_DIM_V < 256:
        return 128
    if packed_query_rows > 16:
        return 64
    return 16


def derived_metrics(
    *,
    args: argparse.Namespace,
    backend_resolved: str,
    g: int,
    s_q: int,
    page_size: int,
    timing: dict[str, Any],
    plan_details: dict[str, Any],
) -> dict[str, Any]:
    full_call_us = timing["median_us"]
    physical_bytes_per_sequence = (
        args.context_len
        * g
        * (HEAD_DIM_QK + HEAD_DIM_V)
        * BYTES_PER_ELEMENT
    )
    allocated_tokens = math.ceil(args.context_len / page_size) * page_size
    allocated_bytes_per_sequence = (
        allocated_tokens * g * (HEAD_DIM_QK + HEAD_DIM_V) * BYTES_PER_ELEMENT
    )
    model_flops_per_sequence = (
        2
        * NUM_HEADS_Q
        * s_q
        * args.context_len
        * (HEAD_DIM_QK + HEAD_DIM_V)
    )
    packed_query_rows = s_q * (NUM_HEADS_Q // g)
    result: dict[str, Any] = {
        "full_call_us": full_call_us,
        "per_sequence_us": full_call_us / args.batch_size,
        "aggregate_query_tokens_per_second": (
            args.batch_size * s_q / (full_call_us * 1.0e-6)
        ),
        "physical_kv_payload_bytes_per_sequence": physical_bytes_per_sequence,
        "physical_kv_payload_mib_per_sequence": physical_bytes_per_sequence / MIB,
        "allocated_kv_bytes_per_sequence": allocated_bytes_per_sequence,
        "physical_kv_payload_tb_s": (
            args.batch_size * physical_bytes_per_sequence / full_call_us / 1.0e6
        ),
        "model_flops_per_sequence": model_flops_per_sequence,
        "model_tflops": args.batch_size * model_flops_per_sequence / full_call_us / 1.0e6,
        "gqa_group_size": NUM_HEADS_Q // g,
        "packed_query_rows": packed_query_rows,
    }

    if backend_resolved == "fa2":
        cta_tile_q = plan_details.get(
            "cta_tile_q", fa2_expected_cta_tile_q(packed_query_rows)
        )
        num_m_tiles = math.ceil(packed_query_rows / cta_tile_q)
        tile_flops_per_sequence = (
            2
            * g
            * num_m_tiles
            * cta_tile_q
            * args.context_len
            * (HEAD_DIM_QK + HEAD_DIM_V)
        )
        result["fa2_tile_model"] = {
            "cta_tile_q": cta_tile_q,
            "num_m_tiles": num_m_tiles,
            "valid_row_fraction": packed_query_rows / (num_m_tiles * cta_tile_q),
            "tile_compute_amplification": tile_flops_per_sequence / model_flops_per_sequence,
            "tile_flops_per_sequence": tile_flops_per_sequence,
            "tile_arithmetic_intensity_flops_per_byte": (
                tile_flops_per_sequence / physical_bytes_per_sequence
            ),
            "source_rule_for_pinned_version": "packed<=16: M16; packed<=64: M64; else M128",
        }
        if args.measured_bf16_tflops is not None:
            result["fa2_tile_model"]["compute_lower_us"] = (
                tile_flops_per_sequence / args.measured_bf16_tflops / 1.0e6
            )

    if args.measured_hbm_tb_s is not None:
        result["measured_memory_lower_us"] = (
            physical_bytes_per_sequence / args.measured_hbm_tb_s / 1.0e6
        )
    compute_lower_us = result.get("fa2_tile_model", {}).get("compute_lower_us")
    memory_lower_us = result.get("measured_memory_lower_us")
    if compute_lower_us is not None and memory_lower_us is not None:
        roofline_us = max(compute_lower_us, memory_lower_us)
        result["tile_aware_roofline_us"] = roofline_us
        result["tile_aware_roofline_efficiency_percent"] = (
            100.0 * roofline_us / result["per_sequence_us"]
        )
        result["tile_aware_predicted_bottleneck"] = (
            "compute" if compute_lower_us > memory_lower_us else "memory"
        )
    return result


def run_point(
    *,
    args: argparse.Namespace,
    device: torch.device,
    g: int,
    s_q: int,
) -> dict[str, Any]:
    page_size = page_size_for(args, s_q)
    print(
        f"\nALLOC g={g} s_q={s_q} B={args.batch_size} L={args.context_len} "
        f"page={page_size}",
        flush=True,
    )
    inputs = allocate_inputs(
        args=args, device=device, g=g, s_q=s_q, page_size=page_size
    )
    point: dict[str, Any] = {
        "g": g,
        "s_q": s_q,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "page_size": page_size,
        "pages_per_sequence": inputs["pages_per_sequence"],
        "last_page_len": inputs["last_page_len"],
        "seed": inputs["point_seed"],
        "cache_layout": {
            "name": "NHD compact interleaved K/V backing",
            "storage_shape": list(inputs["kv_storage"].shape),
            "k_shape": list(inputs["k_cache"].shape),
            "v_shape": list(inputs["v_cache"].shape),
            "k_stride": list(inputs["k_cache"].stride()),
            "v_stride": list(inputs["v_cache"].stride()),
            "padding_elements_per_token_per_kv_head": 0,
        },
        "backends": [],
    }

    for backend in args.backends:
        print(f"PREPARE backend={backend} g={g} s_q={s_q}", flush=True)
        prepared: PreparedPoint | None = None
        try:
            prepared = prepare_backend(
                args=args,
                device=device,
                inputs=inputs,
                backend=backend,
                g=g,
                page_size=page_size,
            )
            print(
                f"PLAN backend={backend}->{prepared.backend_resolved} "
                f"plan_ms={prepared.plan_ms:.3f} details={prepared.plan_details}",
                flush=True,
            )
            correctness = None
            if not args.skip_correctness:
                correctness = pytorch_reference_check(
                    args=args,
                    inputs=inputs,
                    actual=prepared.output,
                    g=g,
                    s_q=s_q,
                )
                print(
                    f"CHECK backend={prepared.backend_resolved} "
                    f"allclose={correctness['allclose']} "
                    f"max_abs={correctness['max_abs']:.6g} "
                    f"mean_abs={correctness['mean_abs']:.6g}",
                    flush=True,
                )
                if not correctness["allclose"] or not correctness["finite"]:
                    raise RuntimeError(f"correctness check failed: {correctness}")

            timing = benchmark_callable(
                prepared.fn,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                rounds=args.rounds,
            )
            metrics = derived_metrics(
                args=args,
                backend_resolved=prepared.backend_resolved,
                g=g,
                s_q=s_q,
                page_size=page_size,
                timing=timing,
                plan_details=prepared.plan_details,
            )
            profile = (
                profile_cuda_kernels(prepared.fn, calls=args.profile_calls)
                if args.profile_kernels
                else None
            )
            record = {
                "backend_requested": backend,
                "backend_resolved": prepared.backend_resolved,
                "plan_ms": prepared.plan_ms,
                "plan": prepared.plan_details,
                "correctness": correctness,
                "timing": timing,
                "derived": metrics,
                "kineto": profile,
            }
            point["backends"].append(record)
            print(
                f"RESULT backend={prepared.backend_resolved} g={g} s_q={s_q} "
                f"full_us={timing['median_us']:.3f} "
                f"per_seq_us={metrics['per_sequence_us']:.3f} "
                f"tok_s={metrics['aggregate_query_tokens_per_second']:.1f} "
                f"payload_tb_s={metrics['physical_kv_payload_tb_s']:.3f}",
                flush=True,
            )
        except Exception as error:
            error_record = {
                "backend_requested": backend,
                "backend_resolved": (
                    prepared.backend_resolved if prepared is not None else None
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            point["backends"].append(error_record)
            print(
                f"ERROR backend={backend} g={g} s_q={s_q}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            if args.fail_fast:
                raise
        finally:
            del prepared
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return point


def write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def print_summary(result: dict[str, Any]) -> None:
    print("\n=== FLASHINFER GQLA GQA SUMMARY ===", flush=True)
    print(
        "backend g sq tile split | full_us per_seq_us tok/s payload_TB/s check",
        flush=True,
    )
    for point in result["points"]:
        for backend in point["backends"]:
            if "error" in backend:
                print(
                    f"{backend['backend_requested']:7s} {point['g']:1d} {point['s_q']:2d} "
                    f"   -     - | ERROR {backend['error']}",
                    flush=True,
                )
                continue
            plan = backend["plan"]
            metrics = backend["derived"]
            correctness = backend["correctness"]
            check = "skip" if correctness is None else ("ok" if correctness["allclose"] else "FAIL")
            tile = str(plan.get("cta_tile_q", "?"))
            split = str(plan.get("split_kv", "?"))
            print(
                f"{backend['backend_resolved']:7s} {point['g']:1d} {point['s_q']:2d} "
                f"{tile:>4s} {split:>5s} | "
                f"{metrics['full_call_us']:7.3f} "
                f"{metrics['per_sequence_us']:10.3f} "
                f"{metrics['aggregate_query_tokens_per_second']:9.1f} "
                f"{metrics['physical_kv_payload_tb_s']:12.3f} {check}",
                flush=True,
            )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    device_name = torch.cuda.get_device_name(device)
    if args.expect_device_substring and not any(
        expected.lower() in device_name.lower()
        for expected in args.expect_device_substring
    ):
        raise RuntimeError(
            f"device {device_name!r} matches none of {args.expect_device_substring!r}"
        )
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        raise RuntimeError(f"FlashInfer FA2 benchmark requires SM80 or newer, got {capability}")

    output = args.output or default_output_path(args.label)
    result: dict[str, Any] = {
        "schema": "gqla-flashinfer-short-gqa-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": device_metadata(device, args.label),
        "fixed_constants": {
            "num_heads_q": NUM_HEADS_Q,
            "head_dim_qk": HEAD_DIM_QK,
            "head_dim_v": HEAD_DIM_V,
            "dtype": str(DTYPE),
            "softmax_scale": SOFTMAX_SCALE,
            "causal": True,
            "kv_append": False,
        },
        "measurement_protocol": {
            "timed_scope": "FlashInfer wrapper.run with a preallocated output",
            "excluded": [
                "input/cache allocation",
                "cache initialization",
                "wrapper construction",
                "wrapper.plan",
                "first-use JIT/module loading",
                "correctness reference",
            ],
            "timer": "triton.testing.do_bench CUDA events",
            "cache_state": "Triton L2 eviction before timed invocation",
        },
        "points": [],
    }

    for g in args.g_values:
        for s_q in args.sq_values:
            result["points"].append(
                run_point(args=args, device=device, g=g, s_q=s_q)
            )
            # Preserve partial results if a later allocation/kernel fails.
            write_json(output, result)

    successful = [
        backend
        for point in result["points"]
        for backend in point["backends"]
        if "error" not in backend
    ]
    failed = [
        backend
        for point in result["points"]
        for backend in point["backends"]
        if "error" in backend
    ]
    result["status"] = {
        "successful_backend_points": len(successful),
        "failed_backend_points": len(failed),
        "ok": bool(successful) and not failed,
    }
    write_json(output, result)
    print_summary(result)
    print(f"FLASHINFER_GQA_JSON={output.resolve()}", flush=True)
    if not successful:
        print("FLASHINFER_GQA_FAILED", flush=True)
        return 1
    if failed:
        print("FLASHINFER_GQA_PARTIAL", flush=True)
        return 2
    print("FLASHINFER_GQA_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
