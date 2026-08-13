#!/usr/bin/env python3
"""Merge official MQA and HPC-Ops GQA TP results into the strict Table 2."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


TP_SIZES = (1, 2, 4, 8)
STRICT_CASES = (
    ("mqa_absorb", 1, 1),
    ("mqa_absorb", 1, 2),
    ("gqa", 8, 1),
    ("gqa", 8, 2),
    ("gqa", 4, 1),
    ("gqa", 4, 2),
)
ROLE_ORDER = ("H100-role（L20Z）", "H20")
MARKER_BEGIN = "<!-- BEGIN HPC_OPS_TP_TABLE2 -->"
MARKER_END = "<!-- END HPC_OPS_TP_TABLE2 -->"

# Keep these cells byte-for-byte consistent with the reference report's Table 2.
THEORY = {
    "H100-role（L20Z）": {
        ("mqa_absorb", 1, 1): (1152, 242, 2.82, 2.31, 2.82, "354K"),
        ("mqa_absorb", 1, 2): (1152, 484, 2.82, 4.61, 4.61, "434K"),
        ("gqa", 8, 1): (4224, 19, 10.33, 0.68, 10.33, "96.8K"),
        ("gqa", 8, 2): (4224, 39, 10.33, 1.36, 10.33, "193.6K"),
        ("gqa", 4, 1): (2176, 38, 5.32, 0.68, 5.32, "187.9K"),
        ("gqa", 4, 2): (2176, 75, 5.32, 1.36, 5.32, "375.9K"),
    },
    "H20": {
        ("mqa_absorb", 1, 1): (1152, 242, 2.36, 15.42, 15.42, "65K"),
        ("mqa_absorb", 1, 2): (1152, 484, 2.36, 30.84, 30.84, "65K"),
        ("gqa", 8, 1): (4224, 19, 8.65, 4.53, 8.65, "116K"),
        ("gqa", 8, 2): (4224, 39, 8.65, 9.06, 9.06, "221K"),
        ("gqa", 4, 1): (2176, 38, 4.45, 4.53, 4.53, "221K"),
        ("gqa", 4, 2): (2176, 75, 4.45, 9.06, 9.06, "221K"),
    },
}
THEORY_TOK_S = {
    role: {
        case: float(display.removesuffix("K")) * 1000
        for case, (*_, display) in cases.items()
    }
    for role, cases in THEORY.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100-official-json", type=Path, required=True)
    parser.add_argument("--h20-official-json", type=Path, required=True)
    parser.add_argument("--h100-hpc-json", type=Path, required=True)
    parser.add_argument("--h20-hpc-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def official_mqa_rows(report: dict, role: str) -> dict[tuple[str, int, int, int], dict]:
    if report.get("status") != "complete" or report.get("schema_version") != 6:
        raise ValueError(f"{role}: official TP JSON must be complete schema-v6")
    output = {}
    for sweep in report.get("tp_sweeps", []):
        if sweep.get("path") != "mqa_absorb" or sweep.get("implementation") != "FlashMLA":
            continue
        g = sweep.get("g")
        sq = sweep.get("s_q")
        for point in sweep.get("points", []):
            tp = point.get("tp_size")
            key = ("mqa_absorb", g, sq, tp)
            if key in output:
                raise ValueError(f"{role}: duplicate official MQA point {key}")
            if point.get("status") != "complete":
                raise ValueError(f"{role}: incomplete official MQA point {key}")
            output[key] = point
    expected = {
        ("mqa_absorb", 1, sq, tp) for sq in (1, 2) for tp in TP_SIZES
    }
    if set(output) != expected:
        raise ValueError(f"{role}: official MQA point set mismatch")
    return output


def hpc_gqa_rows(report: dict, role: str) -> dict[tuple[str, int, int, int], dict]:
    if report.get("device_role") != role:
        raise ValueError(
            f"{role}: HPC report role mismatch, got {report.get('device_role')!r}"
        )
    if report.get("batch") != 128 or report.get("seqlen") != 8192:
        raise ValueError(f"{role}: HPC report must use B=128/L=8192")
    if report.get("tp_sizes") != list(TP_SIZES):
        raise ValueError(f"{role}: HPC report must contain TP=1/2/4/8")
    output = {}
    for point in report.get("results", []):
        if point.get("mode") != "static":
            continue
        key = (
            "gqa",
            point.get("global_num_head_kv"),
            point.get("num_seq_q"),
            point.get("tp_size"),
        )
        if key in output:
            raise ValueError(f"{role}: duplicate HPC GQA point {key}")
        output[key] = point
    expected = {
        ("gqa", g, sq, tp) for g in (8, 4) for sq in (1, 2) for tp in TP_SIZES
    }
    if set(output) != expected:
        raise ValueError(f"{role}: HPC GQA point set mismatch")
    return output


def build_rows(
    official_by_role: dict[str, dict], hpc_by_role: dict[str, dict]
) -> list[dict]:
    rows = []
    for role in ROLE_ORDER:
        for path, g, sq in STRICT_CASES:
            cache, intensity, mem_us, cmp_us, step_us, theory_display = THEORY[role][
                (path, g, sq)
            ]
            points = official_by_role[role] if path == "mqa_absorb" else hpc_by_role[role]
            tp_points = {tp: points[(path, g, sq, tp)] for tp in TP_SIZES}
            tp1 = tp_points[1]
            if path == "mqa_absorb":
                full_us = tp1["full_cold_us"] / tp1["batch_size"]
                main_us = tp1["main_cold_us"] / tp1["batch_size"]
                tok_s = {
                    tp: tp_points[tp]["derived_from_cold_median"]["query_tok_per_s"]
                    for tp in TP_SIZES
                }
                implementation = "FlashMLA"
                source = "official schema-v6 TP sweep"
                physical_theory_tok_s = None
                physical_efficiency = None
            else:
                full_us = tp1["us_per_sequence"]
                main_us = tp1["main_us_per_sequence"]
                tok_s = {tp: tp_points[tp]["query_tokens_per_second"] for tp in TP_SIZES}
                implementation = "HPC-Ops"
                source = "patched HPC-Ops TP rerun"
                physical_theory_tok_s = tp1[
                    "physical_theoretical_query_tokens_per_second"
                ]
                physical_efficiency = tp1["physical_roofline_efficiency"]
            theory_tok_s = THEORY_TOK_S[role][(path, g, sq)]
            rows.append(
                {
                    "gpu": role,
                    "path": "MQA-absorb" if path == "mqa_absorb" else "GQA",
                    "implementation": implementation,
                    "g": g,
                    "s_q": sq,
                    "cache_bytes_per_token": cache,
                    "arithmetic_intensity": intensity,
                    "memory_floor_us": mem_us,
                    "compute_floor_us": cmp_us,
                    "theoretical_step_us": step_us,
                    "theoretical_tok_s_display": theory_display,
                    "theoretical_tok_s": theory_tok_s,
                    "full_us_per_sequence": full_us,
                    "main_us_per_sequence": main_us,
                    "actual_tok_s": {str(tp): tok_s[tp] for tp in TP_SIZES},
                    "tp1_vs_table2_theory": tok_s[1] / theory_tok_s,
                    "physical_theoretical_tok_s": physical_theory_tok_s,
                    "tp1_vs_physical_theory": physical_efficiency,
                    "optimization_headroom": path == "gqa" and g == 4 and sq == 2,
                    "source": source,
                }
            )
    return rows


def fmt_tok(value: float) -> str:
    return f"{value / 1000:.1f}K"


def markdown_section(rows: list[dict], hpc_reports: dict[str, dict]) -> str:
    lines = [
        MARKER_BEGIN,
        "## Table 2 严格对照：H100-role/H20 各六行（GQA 已替换为 HPC-Ops）",
        "",
        "表结构、理论列和 MQA 数据与原 `Table 2 严格对照` 完全一致。MQA-absorb "
        "继续使用同一份官方 FlashMLA schema-v6 TP 数据；四个 GQA 行的 FA3 数据已全部"
        "替换为本次 `QK=192/V=128` HPC-Ops TP 重测。",
        "",
        "| GPU | Path | g | `s_q` | cache (B/tok) | I | mem (µs) | cmp (µs) | 理论 step (µs) | 理论 tok/s | 实测 full/序列 (µs) | 实测 main/序列 (µs) | 实际 tok/s (TP=1) | 实际 tok/s (TP=2) | 实际 tok/s (TP=4) | 实际 tok/s (TP=8) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        path = row["path"]
        if path == "GQA":
            path = "GQA（HPC-Ops）" + ("†" if row["optimization_headroom"] else "")
        actual = row["actual_tok_s"]
        lines.append(
            f"| {row['gpu']} | {path} | {row['g']} | {row['s_q']} | "
            f"{row['cache_bytes_per_token']} | {row['arithmetic_intensity']} | "
            f"{row['memory_floor_us']:.2f} | {row['compute_floor_us']:.2f} | "
            f"{row['theoretical_step_us']:.2f} | {row['theoretical_tok_s_display']} | "
            f"**{row['full_us_per_sequence']:.3f}** | "
            f"**{row['main_us_per_sequence']:.3f}** | "
            f"**{fmt_tok(actual['1'])}** | **{fmt_tok(actual['2'])}** | "
            f"**{fmt_tok(actual['4'])}** | **{fmt_tok(actual['8'])}** |"
        )

    lines.extend(
        [
            "",
            "† `(g=4, s_q=2)` 明确保留后续优化空间：TP=1/2/4 的 local head "
            "ratio 为 32，走 M64 specialization；TP=8 因 KV 复制使 local ratio "
            "变为 16，转用 M32。本表使用本轮原始实测，不做估算或修正。",
            "",
            "### 实测硬件带宽与算力",
            "",
            "这里的 H100-role 实际设备仍是 NVIDIA L20Z，按原报告口径保留角色名，"
            "不将产品名改写为零售 H100。",
            "",
            "| GPU 角色 | 实际设备 | 持续读取带宽 | Dense BF16 算力 |",
            "|---|---|---:|---:|",
        ]
    )
    for role in ROLE_ORDER:
        report = hpc_reports[role]
        calibration = report["hardware_calibration"]
        lines.append(
            f"| {role} | {report['device']} | "
            f"**{calibration['measured_hbm_tb_s']:.6g} TB/s** | "
            f"**{calibration['measured_bf16_tflops']:.6g} TFLOP/s** |"
        )

    lines.extend(
        [
            "",
            "### GQA TP=1：理论输出速度对比",
            "",
            "`Table 2 理论` 保留论文的逻辑 cache 与设备上限；`物理理论` 使用各设备"
            "独立实测带宽/算力，以及 HPC-Ops 实际读取的 40/20 MiB BF16 KV layout。",
            "",
            "| GPU | g | `s_q` | Table 2 理论 tok/s | HPC-Ops 实测 tok/s | 实测/Table 2 | 物理理论 tok/s | 实测/物理理论 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["path"] != "GQA":
            continue
        dagger = "†" if row["optimization_headroom"] else ""
        lines.append(
            f"| {row['gpu']} | {row['g']} | {row['s_q']}{dagger} | "
            f"{row['theoretical_tok_s_display']} | "
            f"{fmt_tok(row['actual_tok_s']['1'])} | "
            f"{row['tp1_vs_table2_theory'] * 100:.1f}% | "
            f"{fmt_tok(row['physical_theoretical_tok_s'])} | "
            f"{row['tp1_vs_physical_theory'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "TP=1 的 full 是未启用 profiler 的冷缓存 CUDA-event 中位数；main 是独立"
            "冷缓存 Kineto profile 中主 CUDA kernel 的均值。两列来自分开的采样过程，"
            "不使用 `full-main` 作为逐调用开销。TP=2/4/8 仍是单 GPU rank-local 理想切分，"
            "不含 NCCL 或其他 collective，吞吐不乘 TP。",
            MARKER_END,
        ]
    )
    return "\n".join(lines)


def update_markdown(path: Path, section: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# HPC-Ops GQLA 测速表\n"
    if MARKER_BEGIN in existing or MARKER_END in existing:
        if existing.count(MARKER_BEGIN) != 1 or existing.count(MARKER_END) != 1:
            raise ValueError("markdown has malformed Table 2 markers")
        before, remainder = existing.split(MARKER_BEGIN, 1)
        _, after = remainder.split(MARKER_END, 1)
        output = before.rstrip() + "\n\n" + section + after
    else:
        output = existing.rstrip() + "\n\n" + section + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "gpu",
        "path",
        "implementation",
        "g",
        "s_q",
        "cache_bytes_per_token",
        "arithmetic_intensity",
        "memory_floor_us",
        "compute_floor_us",
        "theoretical_step_us",
        "theoretical_tok_s",
        "full_us_per_sequence",
        "main_us_per_sequence",
        "actual_tok_s_tp1",
        "actual_tok_s_tp2",
        "actual_tok_s_tp4",
        "actual_tok_s_tp8",
        "tp1_vs_table2_theory",
        "physical_theoretical_tok_s",
        "tp1_vs_physical_theory",
        "optimization_headroom",
        "source",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            for tp in TP_SIZES:
                output[f"actual_tok_s_tp{tp}"] = row["actual_tok_s"][str(tp)]
            writer.writerow(output)


def main() -> None:
    args = parse_args()
    official_reports = {
        "H100-role（L20Z）": load_json(args.h100_official_json),
        "H20": load_json(args.h20_official_json),
    }
    hpc_reports = {
        "H100-role（L20Z）": load_json(args.h100_hpc_json),
        "H20": load_json(args.h20_hpc_json),
    }
    official_by_role = {
        role: official_mqa_rows(report, role)
        for role, report in official_reports.items()
    }
    hpc_by_role = {
        role: hpc_gqa_rows(report, role) for role, report in hpc_reports.items()
    }
    rows = build_rows(official_by_role, hpc_by_role)
    section = markdown_section(rows, hpc_reports)
    update_markdown(args.markdown, section)

    output = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "protocol": {
            "batch": 128,
            "context_length": 8192,
            "global_num_head_q": 128,
            "qk_dim": 192,
            "value_dim": 128,
            "tp_sizes": list(TP_SIZES),
            "tp_semantics": hpc_reports["H20"].get("tp_simulation_semantics"),
            "mqa_implementation": "FlashMLA official schema-v6",
            "gqa_implementation": "patched HPC-Ops QK=192/V=128 rerun",
        },
        "hardware_calibration": {
            role: {
                "actual_device": report["device"],
                **report["hardware_calibration"],
            }
            for role, report in hpc_reports.items()
        },
        "rows": rows,
        "sources": {
            "h100_official_json": str(args.h100_official_json.resolve()),
            "h20_official_json": str(args.h20_official_json.resolve()),
            "h100_hpc_json": str(args.h100_hpc_json.resolve()),
            "h20_hpc_json": str(args.h20_hpc_json.resolve()),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv, rows)
    print(f"HPC_OPS_TABLE2_TP_MERGE_OK rows={len(rows)}", flush=True)
    print(f"Markdown: {args.markdown}", flush=True)
    print(f"JSON: {args.output_json}", flush=True)
    print(f"CSV: {args.csv}", flush=True)


if __name__ == "__main__":
    main()
