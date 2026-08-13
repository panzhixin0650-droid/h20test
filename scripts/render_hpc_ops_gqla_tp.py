#!/usr/bin/env python3
"""Validate and publish one device's HPC-Ops GQLA TP sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


GLOBAL_NUM_HEAD_Q = 128
GLOBAL_NUM_HEAD_KV = (8, 4)
NUM_SEQ_Q = (1, 2)
TP_SIZES = (1, 2, 4, 8)
QK_DIM = 192
V_DIM = 128
BLOCK_SIZE = 64
TABLE2_LOGICAL_CACHE_BYTES_PER_TOKEN = {8: 4224, 4: 2176}
EXPECTED_CASES = tuple(
    (g, sq, tp) for g in GLOBAL_NUM_HEAD_KV for sq in NUM_SEQ_Q for tp in TP_SIZES
)
NUMERIC_FIELDS = (
    "call_us",
    "us_per_sequence",
    "main_kernel_us",
    "main_us_per_sequence",
    "query_tokens_per_second",
    "physical_kv_payload_tb_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expect-device-substring", required=True)
    parser.add_argument("--device-role", required=True)
    parser.add_argument("--expect-batch", type=int, default=128)
    parser.add_argument("--expect-seqlen", type=int, default=8192)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--patched-commit", required=True)
    parser.add_argument("--patch-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--table2-hbm-tb-s", type=float, required=True)
    parser.add_argument("--table2-bf16-tflops", type=float, required=True)
    parser.add_argument("--measured-hbm-tb-s", type=float, required=True)
    parser.add_argument("--measured-bf16-tflops", type=float, required=True)
    return parser.parse_args()


def finite_positive(row: dict, field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive, got {value!r}")
    return value


def validate(report: dict, args: argparse.Namespace) -> list[dict]:
    expected_metadata = {
        "batch": args.expect_batch,
        "seqlen": args.expect_seqlen,
        "num_head_q": GLOBAL_NUM_HEAD_Q,
        "global_num_head_q": GLOBAL_NUM_HEAD_Q,
        "global_num_head_kv": list(GLOBAL_NUM_HEAD_KV),
        "tp_sizes": list(TP_SIZES),
        "qk_dim": QK_DIM,
        "v_dim": V_DIM,
        "block_size": BLOCK_SIZE,
    }
    for field, expected in expected_metadata.items():
        if report.get(field) != expected:
            raise ValueError(f"expected {field}={expected!r}, got {report.get(field)!r}")

    device = report.get("device")
    if not isinstance(device, str) or args.expect_device_substring.lower() not in device.lower():
        raise ValueError(
            f"expected device containing {args.expect_device_substring!r}, got {device!r}"
        )
    if report.get("compute_capability") != "9.0":
        raise ValueError(
            f"expected compute_capability='9.0', got {report.get('compute_capability')!r}"
        )

    raw_rows = report.get("results")
    if not isinstance(raw_rows, list):
        raise ValueError("results must be a list")
    rows_by_case: dict[tuple[int, int, int], dict] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or row.get("mode") != "static":
            raise ValueError(f"only static rows are publishable, got {row!r}")
        case = (
            row.get("global_num_head_kv"),
            row.get("num_seq_q"),
            row.get("tp_size"),
        )
        if case not in EXPECTED_CASES:
            raise ValueError(f"unexpected benchmark case {case!r}")
        if case in rows_by_case:
            raise ValueError(f"duplicate benchmark case {case!r}")
        for field in NUMERIC_FIELDS:
            finite_positive(row, field)

        g, _, tp = case
        expected_local_hq = GLOBAL_NUM_HEAD_Q // tp
        expected_local_hkv = max(g // tp, 1)
        expected_replication = max(tp // g, 1)
        expected_payload = (
            args.expect_batch
            * args.expect_seqlen
            * expected_local_hkv
            * (QK_DIM + V_DIM)
            * 2
        )
        expected_fields = {
            "local_num_head_q": expected_local_hq,
            "local_num_head_kv": expected_local_hkv,
            "kv_replication_factor": expected_replication,
            "physical_kv_payload_bytes": expected_payload,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"case {case}: expected {field}={expected}, got {row.get(field)!r}"
                )
        rows_by_case[case] = row

    if set(rows_by_case) != set(EXPECTED_CASES):
        missing = sorted(set(EXPECTED_CASES) - set(rows_by_case))
        raise ValueError(f"missing benchmark cases: {missing}")
    return [rows_by_case[case] for case in EXPECTED_CASES]


def enrich(report: dict, rows: list[dict], args: argparse.Namespace) -> None:
    limits = {
        "table2_hbm_tb_s": args.table2_hbm_tb_s,
        "table2_bf16_tflops": args.table2_bf16_tflops,
        "measured_hbm_tb_s": args.measured_hbm_tb_s,
        "measured_bf16_tflops": args.measured_bf16_tflops,
    }
    for name, value in limits.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    report["device_role"] = args.device_role
    report["hardware_calibration"] = {
        "measured_hbm_tb_s": args.measured_hbm_tb_s,
        "measured_bf16_tflops": args.measured_bf16_tflops,
        "table2_hbm_tb_s": args.table2_hbm_tb_s,
        "table2_bf16_tflops": args.table2_bf16_tflops,
    }

    for row in rows:
        g = row["global_num_head_kv"]
        sq = row["num_seq_q"]
        local_hq = row["local_num_head_q"]
        local_hkv = row["local_num_head_kv"]
        physical_bytes = (
            2 * report["seqlen"] * local_hkv * (report["qk_dim"] + report["v_dim"])
        )
        local_flops = (
            2
            * local_hq
            * sq
            * report["seqlen"]
            * (report["qk_dim"] + report["v_dim"])
        )
        physical_mem_us = physical_bytes / (args.measured_hbm_tb_s * 1e12) * 1e6
        physical_cmp_us = local_flops / (args.measured_bf16_tflops * 1e12) * 1e6
        physical_step_us = max(physical_mem_us, physical_cmp_us)

        logical_bytes = TABLE2_LOGICAL_CACHE_BYTES_PER_TOKEN[g] * report["seqlen"]
        global_flops = (
            2
            * GLOBAL_NUM_HEAD_Q
            * sq
            * report["seqlen"]
            * (report["qk_dim"] + report["v_dim"])
        )
        table2_mem_us = logical_bytes / (args.table2_hbm_tb_s * 1e12) * 1e6
        table2_cmp_us = global_flops / (args.table2_bf16_tflops * 1e12) * 1e6
        table2_step_us = max(table2_mem_us, table2_cmp_us)
        row.update(
            {
                "physical_kv_payload_bytes_per_sequence": physical_bytes,
                "local_model_flops_per_sequence": local_flops,
                "local_arithmetic_intensity_flops_per_byte": local_flops
                / physical_bytes,
                "physical_memory_floor_us": physical_mem_us,
                "physical_compute_floor_us": physical_cmp_us,
                "physical_roofline_us": physical_step_us,
                "physical_roofline_bottleneck": (
                    "memory" if physical_mem_us >= physical_cmp_us else "compute"
                ),
                "physical_theoretical_query_tokens_per_second": sq
                * 1e6
                / physical_step_us,
                "physical_roofline_efficiency": physical_step_us
                / row["us_per_sequence"],
                "effective_local_model_tflops": local_flops
                / row["us_per_sequence"]
                / 1e6,
                "table2_logical_cache_bytes_per_token": TABLE2_LOGICAL_CACHE_BYTES_PER_TOKEN[
                    g
                ],
                "table2_memory_floor_us": table2_mem_us,
                "table2_compute_floor_us": table2_cmp_us,
                "table2_roofline_us": table2_step_us,
                "table2_theoretical_query_tokens_per_second": sq
                * 1e6
                / table2_step_us,
            }
        )


def write_csv(path: Path, report: dict, rows: list[dict]) -> None:
    fields = (
        "device_role",
        "device",
        "global_num_head_kv",
        "num_seq_q",
        "tp_size",
        "local_num_head_q",
        "local_num_head_kv",
        "kv_replication_factor",
        "call_us",
        "main_kernel_us",
        "us_per_sequence",
        "main_us_per_sequence",
        "query_tokens_per_second",
        "physical_kv_payload_tb_s",
        "physical_memory_floor_us",
        "physical_compute_floor_us",
        "physical_roofline_us",
        "physical_theoretical_query_tokens_per_second",
        "physical_roofline_efficiency",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            output["device_role"] = report["device_role"]
            output["device"] = report["device"]
            writer.writerow(output)


def write_markdown(path: Path, report: dict, rows: list[dict], provenance: dict) -> None:
    calibration = report["hardware_calibration"]
    lines = [
        f"# HPC-Ops GQLA {report['device_role']} TP 单独测速表",
        "",
        f"- 实际设备：`{report['device']}`（SM {report['compute_capability']}）",
        f"- Shape：`B={report['batch']}, L={report['seqlen']}, global H_Q=128, "
        "D_QK=192, D_V=128, page=64, BF16`",
        "- TP：单 GPU rank-local 模拟，不含 NCCL；吞吐不乘 TP。",
        f"- 计时：full 为 {report['latency_statistic']}；main 为 "
        f"{report['main_latency_statistic']}。",
        f"- 实测硬件上限：`{calibration['measured_hbm_tb_s']:.6g} TB/s`、"
        f"`{calibration['measured_bf16_tflops']:.6g} TFLOP/s`。",
        f"- 补丁提交：`{provenance['patched_commit']}`；补丁 SHA256："
        f"`{provenance['patch_sha256']}`；run ID：`{provenance['run_id']}`。",
        "",
        "| g | `s_q` | TP | local H_Q | local H_KV | KV 复制 | full/序列 (µs) | main/序列 (µs) | 实际 tok/s | 物理理论 tok/s | 达成率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['global_num_head_kv']} | {row['num_seq_q']} | {row['tp_size']} | "
            f"{row['local_num_head_q']} | {row['local_num_head_kv']} | "
            f"{row['kv_replication_factor']}× | {row['us_per_sequence']:.3f} | "
            f"{row['main_us_per_sequence']:.3f} | "
            f"{row['query_tokens_per_second'] / 1000:.1f}K | "
            f"{row['physical_theoretical_query_tokens_per_second'] / 1000:.1f}K | "
            f"{row['physical_roofline_efficiency'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "物理理论使用本机独立实测带宽/算力与 HPC-Ops 实际的 "
            "`K[...,H_KV,192] + V[...,H_KV,128]` BF16 payload。",
            "`(g=4, s_q=2)` 明确保留优化空间：TP=1/2/4 的 local head ratio "
            "仍为 32，使用 M64 specialization；TP=8 因 KV 复制使 local ratio "
            "变为 16，转用 M32。当前实测值均不做修正。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("benchmark JSON root must be an object")
    rows = validate(report, args)
    enrich(report, rows, args)
    provenance = {
        "run_id": args.run_id,
        "source_url": args.source_url,
        "base_commit": args.base_commit,
        "patched_commit": args.patched_commit,
        "patch_sha256": args.patch_sha256,
        "published_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    report["provenance"] = provenance
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv, report, rows)
    write_markdown(args.markdown, report, rows, provenance)
    print(f"HPC_OPS_GQLA_TP_TABLE_OK rows={len(rows)}", flush=True)
    print(f"JSON: {args.output_json}", flush=True)
    print(f"CSV: {args.csv}", flush=True)
    print(f"Markdown: {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
