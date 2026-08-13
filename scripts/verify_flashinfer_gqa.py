#!/usr/bin/env python3
"""Validate and summarize a canonical FlashInfer short-query GQA run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_POINTS = {
    (8, 1): {"page_size": 64, "cta_tile_q": 16},
    (8, 2): {"page_size": 128, "cta_tile_q": 64},
    (4, 1): {"page_size": 64, "cta_tile_q": 64},
    (4, 2): {"page_size": 128, "cta_tile_q": 64},
}
EXPECTED_FLASHINFER = "0.6.11.post2"
NUM_HEADS_Q = 128
HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BYTES_PER_ELEMENT = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--expect-device-substring",
        action="append",
        default=None,
        help="repeat to allow multiple device-name aliases",
    )
    parser.add_argument(
        "--expect-flashinfer-version",
        default=EXPECTED_FLASHINFER,
    )
    parser.add_argument("--require-kineto", action="store_true")
    return parser.parse_args()


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def equal(self, actual: Any, expected: Any, field: str) -> None:
        self.require(actual == expected, f"{field}: expected {expected!r}, got {actual!r}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read benchmark JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"benchmark JSON root must be an object: {path}")
    return value


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def validate_top_level(
    check: Validation, result: dict[str, Any], args: argparse.Namespace
) -> None:
    check.equal(result.get("schema"), "gqla-flashinfer-short-gqa-v1", "schema")
    status = result.get("status", {})
    check.equal(status.get("ok"), True, "status.ok")
    check.equal(status.get("successful_backend_points"), 4, "status.successful_backend_points")
    check.equal(status.get("failed_backend_points"), 0, "status.failed_backend_points")

    constants = result.get("fixed_constants", {})
    check.equal(constants.get("num_heads_q"), NUM_HEADS_Q, "fixed_constants.num_heads_q")
    check.equal(constants.get("head_dim_qk"), HEAD_DIM_QK, "fixed_constants.head_dim_qk")
    check.equal(constants.get("head_dim_v"), HEAD_DIM_V, "fixed_constants.head_dim_v")
    check.equal(constants.get("dtype"), "torch.bfloat16", "fixed_constants.dtype")
    check.equal(constants.get("causal"), True, "fixed_constants.causal")
    check.equal(constants.get("kv_append"), False, "fixed_constants.kv_append")

    arguments = result.get("arguments", {})
    check.equal(arguments.get("batch_size"), 128, "arguments.batch_size")
    check.equal(arguments.get("context_len"), 8192, "arguments.context_len")
    check.equal(arguments.get("g_values"), [8, 4], "arguments.g_values")
    check.equal(arguments.get("sq_values"), [1, 2], "arguments.sq_values")
    check.equal(arguments.get("backends"), ["fa2"], "arguments.backends")
    check.equal(arguments.get("skip_correctness"), False, "arguments.skip_correctness")

    device = result.get("device", {})
    check.equal(device.get("compute_capability"), [9, 0], "device.compute_capability")
    check.equal(
        device.get("flashinfer"),
        args.expect_flashinfer_version,
        "device.flashinfer",
    )
    if args.expect_device_substring:
        name = str(device.get("name", ""))
        check.require(
            any(value.lower() in name.lower() for value in args.expect_device_substring),
            f"device.name {name!r} matches none of {args.expect_device_substring!r}",
        )


def validate_point(
    check: Validation,
    point: dict[str, Any],
    *,
    require_kineto: bool,
) -> tuple[int, int, dict[str, Any]] | None:
    g = point.get("g")
    s_q = point.get("s_q")
    key = (g, s_q)
    prefix = f"point[g={g},sq={s_q}]"
    if key not in EXPECTED_POINTS:
        check.errors.append(f"{prefix}: unexpected point")
        return None
    expected = EXPECTED_POINTS[key]

    check.equal(point.get("batch_size"), 128, f"{prefix}.batch_size")
    check.equal(point.get("context_len"), 8192, f"{prefix}.context_len")
    check.equal(point.get("page_size"), expected["page_size"], f"{prefix}.page_size")
    layout = point.get("cache_layout", {})
    check.equal(
        layout.get("padding_elements_per_token_per_kv_head"),
        0,
        f"{prefix}.cache_layout.padding",
    )
    k_stride = layout.get("k_stride", [])
    v_stride = layout.get("v_stride", [])
    check.require(
        isinstance(k_stride, list)
        and isinstance(v_stride, list)
        and k_stride[:-1] == v_stride[:-1],
        f"{prefix}: K/V leading strides differ",
    )

    backends = point.get("backends")
    check.require(isinstance(backends, list), f"{prefix}.backends must be a list")
    if not isinstance(backends, list) or len(backends) != 1:
        check.errors.append(f"{prefix}: expected exactly one backend, got {backends!r}")
        return None
    backend = backends[0]
    check.require("error" not in backend, f"{prefix}: backend error: {backend.get('error')}")
    check.equal(backend.get("backend_requested"), "fa2", f"{prefix}.backend_requested")
    check.equal(backend.get("backend_resolved"), "fa2", f"{prefix}.backend_resolved")
    plan = backend.get("plan", {})
    check.equal(plan.get("cta_tile_q"), expected["cta_tile_q"], f"{prefix}.cta_tile_q")

    correctness = backend.get("correctness", {})
    check.equal(correctness.get("finite"), True, f"{prefix}.correctness.finite")
    check.equal(correctness.get("allclose"), True, f"{prefix}.correctness.allclose")
    check.equal(
        correctness.get("within_tolerance_fraction"),
        1.0,
        f"{prefix}.correctness.within_tolerance_fraction",
    )

    timing = backend.get("timing", {})
    check.require(finite_positive(timing.get("median_us")), f"{prefix}.timing.median_us")
    check.require(timing.get("count", 0) > 0, f"{prefix}.timing.count must be positive")
    derived = backend.get("derived", {})
    check.require(
        finite_positive(derived.get("per_sequence_us")),
        f"{prefix}.derived.per_sequence_us",
    )
    expected_payload = 8192 * g * (HEAD_DIM_QK + HEAD_DIM_V) * BYTES_PER_ELEMENT
    check.equal(
        derived.get("physical_kv_payload_bytes_per_sequence"),
        expected_payload,
        f"{prefix}.physical_kv_payload_bytes_per_sequence",
    )

    kineto = backend.get("kineto")
    if require_kineto:
        check.require(isinstance(kineto, dict), f"{prefix}.kineto must be present")
        kernels = kineto.get("cuda_kernels", []) if isinstance(kineto, dict) else []
        check.require(bool(kernels), f"{prefix}.kineto.cuda_kernels must be non-empty")
        check.require(
            any("BatchPrefillWithPagedKVCacheKernel" in item.get("name", "") for item in kernels),
            f"{prefix}: main FlashInfer FA2 kernel absent from Kineto",
        )
    return int(g), int(s_q), backend


def render_summary(result: dict[str, Any], rows: list[tuple[int, int, dict[str, Any]]]) -> str:
    device = result["device"]
    lines = [
        "FLASHINFER_GQA_VERIFIED",
        f"device={device['name']}",
        (
            f"versions=python:{device['python']} torch:{device['torch']} "
            f"cuda:{device['torch_cuda']} triton:{device['triton']} "
            f"flashinfer:{device['flashinfer']}"
        ),
        "backend g sq tile split full_us per_seq_us payload_TB/s max_abs",
    ]
    for g, s_q, backend in sorted(rows, key=lambda item: (-item[0], item[1])):
        plan = backend["plan"]
        derived = backend["derived"]
        correctness = backend["correctness"]
        lines.append(
            f"fa2 {g} {s_q} {plan['cta_tile_q']} {str(plan.get('split_kv')).lower()} "
            f"{derived['full_call_us']:.6f} {derived['per_sequence_us']:.6f} "
            f"{derived['physical_kv_payload_tb_s']:.6f} {correctness['max_abs']:.8g}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        result = load_json(args.input)
    except RuntimeError as error:
        print(f"FLASHINFER_GQA_VERIFY_FAILED\n- {error}", file=sys.stderr)
        return 1

    check = Validation()
    validate_top_level(check, result, args)
    points = result.get("points")
    check.require(isinstance(points, list), "points must be a list")
    rows: list[tuple[int, int, dict[str, Any]]] = []
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                check.errors.append(f"point must be an object, got {point!r}")
                continue
            row = validate_point(check, point, require_kineto=args.require_kineto)
            if row is not None:
                rows.append(row)
    keys = [(g, s_q) for g, s_q, _ in rows]
    check.equal(len(rows), 4, "validated point count")
    check.equal(set(keys), set(EXPECTED_POINTS), "validated point set")
    check.equal(len(keys), len(set(keys)), "unique point count")

    if check.errors:
        print("FLASHINFER_GQA_VERIFY_FAILED", file=sys.stderr)
        for error in check.errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    summary = render_summary(result, rows)
    print(summary, end="")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
