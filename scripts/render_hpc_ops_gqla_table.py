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
NUM_HEAD_Q = 128
TABLE2_LOGICAL_CACHE_BYTES_PER_TOKEN = {8: 4224, 4: 2176}
FA3_H20_US_PER_SEQUENCE = {
    (8, 1): 40.677,
    (8, 2): 40.609,
    (4, 1): 20.408,
    (4, 2): 20.450,
}
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
    parser.add_argument("--table2-hbm-tb-s", type=float, default=4.0)
    parser.add_argument("--table2-bf16-tflops", type=float, default=148.0)
    parser.add_argument("--h20-measured-hbm-tb-s", type=float, default=3.572)
    parser.add_argument("--h20-measured-bf16-tflops", type=float, default=141.48)
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
    if (
        not isinstance(device, str)
        or args.expect_device_substring.lower() not in device.lower()
    ):
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
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0
        ):
            raise ValueError(f"invalid physical_kv_payload_bytes={payload_bytes!r}")
        rows_by_case[case] = row

    if set(rows_by_case) != set(EXPECTED_CASES):
        raise ValueError(f"expected cases {EXPECTED_CASES}, got {tuple(rows_by_case)}")
    return [rows_by_case[case] for case in EXPECTED_CASES]


def add_roofline_metrics(
    report: dict, rows: list[dict], args: argparse.Namespace
) -> None:
    roofline_values = {
        "table2_hbm_tb_s": args.table2_hbm_tb_s,
        "table2_bf16_tflops": args.table2_bf16_tflops,
        "h20_measured_hbm_tb_s": args.h20_measured_hbm_tb_s,
        "h20_measured_bf16_tflops": args.h20_measured_bf16_tflops,
    }
    for name, value in roofline_values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")

    table2_bandwidth = args.table2_hbm_tb_s * 1e12
    table2_compute = args.table2_bf16_tflops * 1e12
    measured_bandwidth = args.h20_measured_hbm_tb_s * 1e12
    measured_compute = args.h20_measured_bf16_tflops * 1e12
    report["roofline_assumptions"] = {
        "formula": "step=max(memory_bytes/bandwidth, model_flops/compute); query_tok_s=num_seq_q/step",
        "table2_h20_logical": {
            "hbm_tb_s": args.table2_hbm_tb_s,
            "bf16_tflops": args.table2_bf16_tflops,
            "cache_layout": "Table 2 logical layout with one shared 64-D RoPE key",
            "logical_cache_bytes_per_token_by_num_head_kv": {
                str(key): value
                for key, value in TABLE2_LOGICAL_CACHE_BYTES_PER_TOKEN.items()
            },
        },
        "h20_measured_physical": {
            "hbm_tb_s": args.h20_measured_hbm_tb_s,
            "bf16_tflops": args.h20_measured_bf16_tflops,
            "cache_layout": "HPC-Ops physical K[...,Hkv,192] plus V[...,Hkv,128] BF16 layout",
            "calibration": "independent H20 sustained SM-read and dense-BF16 microbenchmarks",
        },
    }

    for row in rows:
        num_head_kv = row["num_head_kv"]
        num_seq_q = row["num_seq_q"]
        physical_bytes = row["physical_kv_payload_bytes"] / report["batch"]
        expected_physical_bytes = (
            2 * report["seqlen"] * num_head_kv * (report["qk_dim"] + report["v_dim"])
        )
        if physical_bytes != expected_physical_bytes:
            raise ValueError(
                f"physical bytes mismatch for Hkv={num_head_kv}, Sq={num_seq_q}: "
                f"expected {expected_physical_bytes}, got {physical_bytes}"
            )

        model_flops = (
            2
            * NUM_HEAD_Q
            * num_seq_q
            * report["seqlen"]
            * (report["qk_dim"] + report["v_dim"])
        )
        logical_bytes = (
            TABLE2_LOGICAL_CACHE_BYTES_PER_TOKEN[num_head_kv] * report["seqlen"]
        )

        table2_mem_us = logical_bytes / table2_bandwidth * 1e6
        table2_compute_us = model_flops / table2_compute * 1e6
        table2_roofline_us = max(table2_mem_us, table2_compute_us)
        measured_mem_us = physical_bytes / measured_bandwidth * 1e6
        measured_compute_us = model_flops / measured_compute * 1e6
        measured_roofline_us = max(measured_mem_us, measured_compute_us)
        measured_theoretical_tok_s = num_seq_q * 1e6 / measured_roofline_us

        row.update(
            {
                "model_flops_per_sequence": model_flops,
                "physical_kv_payload_bytes_per_sequence": int(physical_bytes),
                "arithmetic_intensity_flops_per_byte": model_flops / physical_bytes,
                "table2_logical_cache_bytes_per_token": TABLE2_LOGICAL_CACHE_BYTES_PER_TOKEN[
                    num_head_kv
                ],
                "table2_memory_floor_us": table2_mem_us,
                "table2_compute_floor_us": table2_compute_us,
                "table2_roofline_us": table2_roofline_us,
                "table2_theoretical_query_tokens_per_second": num_seq_q
                * 1e6
                / table2_roofline_us,
                "table2_roofline_efficiency": table2_roofline_us
                / row["us_per_sequence"],
                "h20_physical_memory_floor_us": measured_mem_us,
                "h20_physical_compute_floor_us": measured_compute_us,
                "h20_physical_roofline_us": measured_roofline_us,
                "h20_physical_roofline_bottleneck": (
                    "memory" if measured_mem_us >= measured_compute_us else "compute"
                ),
                "h20_physical_theoretical_query_tokens_per_second": measured_theoretical_tok_s,
                "h20_physical_roofline_efficiency": row["query_tokens_per_second"]
                / measured_theoretical_tok_s,
                "effective_model_tflops": model_flops / row["us_per_sequence"] / 1e6,
            }
        )


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
        "model_flops_per_sequence",
        "physical_kv_payload_bytes_per_sequence",
        "arithmetic_intensity_flops_per_byte",
        "table2_logical_cache_bytes_per_token",
        "table2_memory_floor_us",
        "table2_compute_floor_us",
        "table2_roofline_us",
        "table2_theoretical_query_tokens_per_second",
        "table2_roofline_efficiency",
        "h20_physical_memory_floor_us",
        "h20_physical_compute_floor_us",
        "h20_physical_roofline_us",
        "h20_physical_roofline_bottleneck",
        "h20_physical_theoretical_query_tokens_per_second",
        "h20_physical_roofline_efficiency",
        "effective_model_tflops",
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
                    "model_flops_per_sequence": row["model_flops_per_sequence"],
                    "physical_kv_payload_bytes_per_sequence": row[
                        "physical_kv_payload_bytes_per_sequence"
                    ],
                    "arithmetic_intensity_flops_per_byte": f"{row['arithmetic_intensity_flops_per_byte']:.6f}",
                    "table2_logical_cache_bytes_per_token": row[
                        "table2_logical_cache_bytes_per_token"
                    ],
                    "table2_memory_floor_us": f"{row['table2_memory_floor_us']:.6f}",
                    "table2_compute_floor_us": f"{row['table2_compute_floor_us']:.6f}",
                    "table2_roofline_us": f"{row['table2_roofline_us']:.6f}",
                    "table2_theoretical_query_tokens_per_second": f"{row['table2_theoretical_query_tokens_per_second']:.3f}",
                    "table2_roofline_efficiency": f"{row['table2_roofline_efficiency']:.6f}",
                    "h20_physical_memory_floor_us": f"{row['h20_physical_memory_floor_us']:.6f}",
                    "h20_physical_compute_floor_us": f"{row['h20_physical_compute_floor_us']:.6f}",
                    "h20_physical_roofline_us": f"{row['h20_physical_roofline_us']:.6f}",
                    "h20_physical_roofline_bottleneck": row[
                        "h20_physical_roofline_bottleneck"
                    ],
                    "h20_physical_theoretical_query_tokens_per_second": f"{row['h20_physical_theoretical_query_tokens_per_second']:.3f}",
                    "h20_physical_roofline_efficiency": f"{row['h20_physical_roofline_efficiency']:.6f}",
                    "effective_model_tflops": f"{row['effective_model_tflops']:.6f}",
                }
            )


def write_markdown(
    path: Path, report: dict, rows: list[dict], provenance: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HPC-Ops GQLA H20 单独测速表",
        "",
        "> 本表只包含打过 `QK=192/V=128` 补丁的 HPC-Ops 正式结果，不混入 FA3、",
        "> FlashInfer 或本地 L20Z 数据。",
        "",
        f"- 设备：`{report['device']}`（SM {report['compute_capability']}）",
        f"- Shape：`B={report['batch']}, L={report['seqlen']}, H_Q=128, "
        "D_QK=192, D_V=128, page=64, BF16`",
        f"- 计时：{report.get('latency_statistic', 'median CUDA-event time')}；"
        f"warmup={report.get('warmup_iterations', 'unknown')}，"
        f"实测迭代={report.get('measured_iterations', 'unknown')}，"
        f"L2 flush={report.get('flush_gib', 'unknown')} GiB",
        f"- HPC-Ops：`{report.get('hpc_ops', 'unknown')}`；PyTorch：`{report.get('torch', 'unknown')}`",
        "- 调度：`static`、单 split",
        f"- 上游基线：`{provenance['base_commit']}`",
        f"- 补丁后提交：`{provenance['patched_commit']}`",
        f"- 补丁 SHA256：`{provenance['patch_sha256']}`",
        f"- H20 run ID：`{provenance['run_id']}`",
        f"- 结果生成时间：`{provenance['published_at_utc']}`",
        "",
        "## HPC-Ops 正式结果",
        "",
        f"| H_KV | S_Q | 整个 B={report['batch']} 调用 (µs) | 每序列 (µs) | Query tok/s | 物理 KV payload (TB/s) |",
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
            "用于与现有 GQLA Table 2 的吞吐等效口径对齐，而不是单请求等待时延。",
            "",
            "## 与理论输出速度对比",
            "",
            "这里同时保留两套理论口径：`Table 2` 使用论文 H20 的 4.0 TB/s、148 TFLOP/s "
            "和共享 RoPE 的逻辑 33/17 MiB cache；`H20 物理` 使用本机独立实测的 3.572 "
            "TB/s、141.48 TFLOP/s，以及 HPC-Ops 实际读取的 40/20 MiB cache。",
            "",
            "| H_KV | S_Q | Table 2 理论 tok/s | 实测 / Table 2 | H20 物理理论 tok/s | HPC-Ops 实测 tok/s | 物理 Roofline 达成率 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['num_head_kv']} | {row['num_seq_q']} | "
            f"{row['table2_theoretical_query_tokens_per_second'] / 1000:.1f}K | "
            f"{row['table2_roofline_efficiency'] * 100:.1f}% | "
            f"{row['h20_physical_theoretical_query_tokens_per_second'] / 1000:.1f}K | "
            f"{row['query_tokens_per_second'] / 1000:.1f}K | "
            f"{row['h20_physical_roofline_efficiency'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "### H20 物理 Roofline 分解",
            "",
            "| H_KV | S_Q | I (FLOP/B) | mem (µs) | cmp (µs) | 理论 step (µs) | 主导项 | 有效模型 TFLOP/s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |",
        ]
    )
    for row in rows:
        bottleneck = (
            "访存" if row["h20_physical_roofline_bottleneck"] == "memory" else "计算"
        )
        lines.append(
            f"| {row['num_head_kv']} | {row['num_seq_q']} | "
            f"{row['arithmetic_intensity_flops_per_byte']:.0f} | "
            f"{row['h20_physical_memory_floor_us']:.3f} | "
            f"{row['h20_physical_compute_floor_us']:.3f} | "
            f"{row['h20_physical_roofline_us']:.3f} | {bottleneck} | "
            f"{row['effective_model_tflops']:.2f} |"
        )

    lines.extend(
        [
            "",
            "理论 tok/s 按 `S_Q × 1e6 / 理论 step(µs)` 计算；Roofline 达成率等价于"
            " `实测 tok/s / 理论 tok/s`。表中 payload TB/s、有效 TFLOP/s 和达成率都是按"
            "建模字节/FLOPs 折算的等效值，不是 Nsight Compute 的硬件计数器。",
            "",
            "## 相对原 H20 FA3 paged GQA",
            "",
            "FA3 基线取自 `docs/H20_FA3_GQA_anomaly_analysis.md` 中相同 "
            "`B=128/L=8192` shape 的 full/序列结果。",
            "",
            "| H_KV | S_Q | FA3 full/序列 (µs) | HPC-Ops 每序列 (µs) | HPC-Ops 加速 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        case = (row["num_head_kv"], row["num_seq_q"])
        fa3_us = FA3_H20_US_PER_SEQUENCE[case]
        lines.append(
            f"| {case[0]} | {case[1]} | {fa3_us:.3f} | "
            f"{row['us_per_sequence']:.3f} | {fa3_us / row['us_per_sequence']:.3f}× |"
        )

    m32 = next(row for row in rows if (row["num_head_kv"], row["num_seq_q"]) == (4, 1))
    m64 = next(row for row in rows if (row["num_head_kv"], row["num_seq_q"]) == (4, 2))
    balance = (
        report["roofline_assumptions"]["h20_measured_physical"]["bf16_tflops"]
        / report["roofline_assumptions"]["h20_measured_physical"]["hbm_tb_s"]
    )
    lines.extend(
        [
            "",
            "## M64 判读",
            "",
            f"- H20 的实测硬件平衡点是 `{balance:.2f} FLOP/B`。`H_KV=4, S_Q=1` "
            f"的 M32 算术强度为 `{m32['arithmetic_intensity_flops_per_byte']:.0f} FLOP/B`，"
            "仍由访存主导；`S_Q=2` 的 M64 上升到 "
            f"`{m64['arithmetic_intensity_flops_per_byte']:.0f} FLOP/B`，因此转为计算受限。",
            f"- M64 的计算下界为 `{m64['h20_physical_compute_floor_us']:.3f} µs/序列`，"
            f"理论输出速度 `{m64['h20_physical_theoretical_query_tokens_per_second'] / 1000:.1f}K tok/s`；"
            f"实测为 `{m64['us_per_sequence']:.3f} µs/序列`、"
            f"`{m64['query_tokens_per_second'] / 1000:.1f}K tok/s`，达到物理 Roofline 的 "
            f"`{m64['h20_physical_roofline_efficiency'] * 100:.1f}%`。",
            f"- 同组 M64/M32 的实测时间比为 `{m64['us_per_sequence'] / m32['us_per_sequence']:.3f}×`。"
            "这不能再用一次 20 MiB KV 扫描的带宽下界解释；需要结合 M64 的寄存器、"
            "shared-memory 驻留和 WGMMA pipeline 继续分析。原始实测值未做修正。",
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
    add_roofline_metrics(report, rows, args)

    existing_provenance = report.get("provenance", {})
    provenance = {
        "run_id": args.run_id,
        "source_url": args.source_url,
        "base_commit": args.base_commit,
        "patched_commit": args.patched_commit,
        "patch_sha256": args.patch_sha256,
        "published_at_utc": existing_provenance.get("published_at_utc")
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
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
