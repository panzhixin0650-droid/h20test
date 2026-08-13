#!/usr/bin/env python3
"""Validate HPC-Ops GQLA benchmark JSON and publish JSON/CSV/Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_CASES = ((8, 1), (8, 2), (4, 1), (4, 2))
NUMERIC_FIELDS = (
    "call_us",
    "us_per_sequence",
    "query_tokens_per_second",
    "physical_kv_payload_tb_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expect-device-substring", default="H20")
    parser.add_argument("--expect-batch", type=int, default=128)
    parser.add_argument("--expect-seqlen", type=int, default=8192)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--patched-commit", required=True)
    parser.add_argument("--patch-sha256", required=True)
    parser.add_argument("--run-id", required=True)
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
        "num_head_q": 128,
        "qk_dim": 192,
        "v_dim": 128,
        "block_size": 64,
    }
    for field, expected in expected_metadata.items():
        if report.get(field) != expected:
            raise ValueError(f"expected {field}={expected}, got {report.get(field)!r}")

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
    rows_by_case: dict[tuple[int, int], dict] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or row.get("mode") != "static":
            raise ValueError(f"only static rows are publishable, got {row!r}")
        case = (row.get("num_head_kv"), row.get("num_seq_q"))
        if case not in EXPECTED_CASES:
            raise ValueError(f"unexpected benchmark case {case!r}")
        if case in rows_by_case:
            raise ValueError(f"duplicate benchmark case {case!r}")
        for field in NUMERIC_FIELDS:
            finite_positive(row, field)
        payload_bytes = row.get("physical_kv_payload_bytes")
        if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes <= 0:
            raise ValueError(f"invalid physical_kv_payload_bytes={payload_bytes!r}")
        rows_by_case[case] = row

    if set(rows_by_case) != set(EXPECTED_CASES):
        raise ValueError(
            f"expected cases {EXPECTED_CASES}, got {tuple(rows_by_case)}"
        )
    return [rows_by_case[case] for case in EXPECTED_CASES]


def write_csv(path: Path, report: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "device",
        "batch",
        "seqlen",
        "num_head_q",
        "num_head_kv",
        "num_seq_q",
        "qk_dim",
        "v_dim",
        "block_size",
        "call_us",
        "us_per_sequence",
        "query_tokens_per_second",
        "physical_kv_payload_tb_s",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "device": report["device"],
                    "batch": report["batch"],
                    "seqlen": report["seqlen"],
                    "num_head_q": report["num_head_q"],
                    "num_head_kv": row["num_head_kv"],
                    "num_seq_q": row["num_seq_q"],
                    "qk_dim": report["qk_dim"],
                    "v_dim": report["v_dim"],
                    "block_size": report["block_size"],
                    "call_us": f"{row['call_us']:.6f}",
                    "us_per_sequence": f"{row['us_per_sequence']:.6f}",
                    "query_tokens_per_second": f"{row['query_tokens_per_second']:.3f}",
                    "physical_kv_payload_tb_s": f"{row['physical_kv_payload_tb_s']:.6f}",
                }
            )


def write_markdown(path: Path, report: dict, rows: list[dict], provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HPC-Ops GQLA 单独测速表",
        "",
        "> 本表只包含打过 QK=192/V=128 补丁的 HPC-Ops 结果，不混入 FA3 或 FlashInfer。",
        "",
        f"- 设备：`{report['device']}`（SM {report['compute_capability']}）",
        f"- Shape：`B={report['batch']}, L={report['seqlen']}, H_Q=128, "
        "D_QK=192, D_V=128, page=64, BF16`",
        f"- 计时：{report.get('latency_statistic', 'median CUDA-event time')}；"
        f"warmup={report.get('warmup_iterations', 'unknown')}，"
        f"实测迭代={report.get('measured_iterations', 'unknown')}，"
        f"L2 flush={report.get('flush_gib', 'unknown')} GiB",
        f"- HPC-Ops：`{report.get('hpc_ops', 'unknown')}`；PyTorch：`{report.get('torch', 'unknown')}`",
        f"- 上游基线：`{provenance['base_commit']}`",
        f"- 补丁 SHA256：`{provenance['patch_sha256']}`",
        f"- 生成时间：`{provenance['published_at_utc']}`",
        "",
        f"| H_KV | S_Q | 整个 B={report['batch']} 调用 (us) | 每序列 (us) | Query tok/s | 物理 KV payload (TB/s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['num_head_kv']} | {row['num_seq_q']} | {row['call_us']:.2f} | "
            f"{row['us_per_sequence']:.3f} | {row['query_tokens_per_second']:.1f} | "
            f"{row['physical_kv_payload_tb_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`整个调用` 是单次 operator 的 CUDA-event 中位延迟；`每序列` 等于该值除以 batch，"
            "用于与现有 GQLA Table 2 口径对齐。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("benchmark JSON root must be an object")
    rows = validate(report, args)

    provenance = {
        "run_id": args.run_id,
        "source_url": args.source_url,
        "base_commit": args.base_commit,
        "patched_commit": args.patched_commit,
        "patch_sha256": args.patch_sha256,
        "published_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    report["provenance"] = provenance
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv, report, rows)
    write_markdown(args.markdown, report, rows, provenance)

    print(f"HPC_OPS_GQLA_TABLE_OK rows={len(rows)}", flush=True)
    print(f"JSON: {args.output_json}", flush=True)
    print(f"CSV: {args.csv}", flush=True)
    print(f"Markdown: {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
