#!/usr/bin/env python3
"""Validate the two authoritative H20 Table 2 result files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLASHATTENTION_COMMIT = "a369df707e1980fb328abcc1733e3457ec10155f"
FLASHMLA_COMMIT = "15f13e5030374295491c5ce31b02d7e63a7772c6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def audit_nested(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "allclose" and child is not True:
                raise AssertionError(f"{child_location} is not true")
            if key == "finite" and child is not True:
                raise AssertionError(f"{child_location} is not true")
            if key == "pack_gqa" and child is False:
                raise AssertionError(f"unsafe explicit pack_gqa=False at {child_location}")
            audit_nested(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            audit_nested(child, f"{location}[{index}]")


def validate_common(result: dict[str, Any], schema_version: int) -> None:
    assert result["schema_version"] == schema_version
    assert result["status"] == "complete"
    assert result["error_case_count"] == 0
    assert result["error_batch_sweep_count"] == 0

    environment = result["environment"]
    assert environment["paper_device_role"] == "h20"
    assert "h20" in environment["actual_device_name"].lower()
    assert environment["compute_capability"] == [9, 0]
    assert environment["flashattention_commit"] == FLASHATTENTION_COMMIT
    assert environment["flashmla_commit"] == FLASHMLA_COMMIT
    assert "vllm" not in environment["flash_attn_interface_module"].lower()
    assert "vllm" not in environment["flash_mla_module"].lower()

    protocol = result["protocol"]
    assert protocol["fake_tensors_only"] is True
    assert protocol["causal"] is True
    assert protocol["context_length"] == 8192
    assert protocol["dtype"] == "torch.bfloat16"
    assert protocol["num_heads_q"] == 128
    assert protocol["fa3_pack_gqa"] == [True, None]
    assert protocol["fa3_pack_gqa_excluded"] == [False]
    assert math.isclose(
        protocol["softmax_scales"]["mla_absorbed_576d"],
        1.0 / math.sqrt(576),
    )
    assert math.isclose(
        protocol["softmax_scales"]["gqa_192d"],
        1.0 / math.sqrt(192),
    )


def validate_v5(result: dict[str, Any]) -> None:
    validate_common(result, 5)
    sweeps = result["batch_sweeps"]
    assert len(sweeps) == 12
    assert sum(len(entry["points"]) for entry in sweeps) == 24
    identities = set()
    for entry in sweeps:
        assert entry["status"] == "complete"
        assert entry["error_point_count"] == 0
        identities.add(
            (
                entry["implementation"],
                entry["path"],
                entry["layout"],
                entry["g"],
                entry["s_q"],
            )
        )
        assert {point["batch_size"] for point in entry["points"]} == {64, 128}
        assert all(point["status"] == "complete" for point in entry["points"])
    assert len(identities) == 12
    audit_nested(sweeps, "batch_sweeps")


def validate_v6(result: dict[str, Any]) -> None:
    validate_common(result, 6)
    assert result["error_tp_sweep_count"] == 0
    sweeps = result["tp_sweeps"]
    assert len(sweeps) == 12
    assert sum(len(entry["points"]) for entry in sweeps) == 48
    identities = set()
    for entry in sweeps:
        assert entry["status"] == "complete"
        assert entry["error_point_count"] == 0
        assert entry["tp_sizes"] == [1, 2, 4, 8]
        identities.add(
            (
                entry["implementation"],
                entry["path"],
                entry["layout"],
                entry["g"],
                entry["s_q"],
            )
        )
        assert {point["tp_size"] for point in entry["points"]} == {1, 2, 4, 8}
        assert {point["batch_size"] for point in entry["points"]} == {128}
        assert all(point["status"] == "complete" for point in entry["points"])
    assert len(identities) == 12
    audit_nested(sweeps, "tp_sweeps")


def main() -> None:
    args = parse_args()
    v5_path = args.output_dir / "h20_table2_all_batch64_128_safe_sq2.json"
    v6_path = args.output_dir / "h20_table2_tp1_2_4_8.json"
    v5 = load(v5_path)
    v6 = load(v6_path)
    validate_v5(v5)
    validate_v6(v6)
    print(
        "H20_RESULTS_OK",
        f"device={v6['environment']['actual_device_name']}",
        f"schema_v5_completed_at={v5['completed_at']}",
        f"schema_v6_completed_at={v6['completed_at']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
