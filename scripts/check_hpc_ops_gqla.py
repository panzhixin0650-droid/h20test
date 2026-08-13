#!/usr/bin/env python3
"""Reference-check the patched HPC-Ops GQLA BF16 decode kernels."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch


BLOCK_SIZE = 64
GLOBAL_NUM_HEAD_Q = 128
QK_DIM = 192
V_DIM = 128
TP_SIZES = (1, 2, 4, 8)
STATIC_CASES = tuple(
    (global_num_head_kv, num_seq_q, tp_size)
    for global_num_head_kv in (8, 4)
    for num_seq_q in (1, 2)
    for tp_size in TP_SIZES
)
SPLITK_HINT_CASES = ((8, 1, 1), (4, 2, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expect-device-substring", default="H20")
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    return parser.parse_args()


def import_hpc(source_dir: Path):
    candidates = sorted(
        path
        for path in (source_dir / "build").glob("lib.*")
        if (path / "hpc" / "_C.abi3.so").is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one built HPC-Ops package under {source_dir / 'build'}, "
            f"found {candidates}"
        )
    sys.path.insert(0, os.fspath(candidates[0].resolve()))
    import hpc

    return hpc, candidates[0]


def make_inputs(num_head_q: int, num_head_kv: int, num_seq_q: int, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_lens = torch.tensor([70, 129], dtype=torch.int32, device="cuda")
    num_batch = total_lens.numel()
    num_blocks = (total_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(num_blocks.sum().item())

    q = torch.randn(
        (num_batch * num_seq_q, num_head_q, QK_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    kcache = torch.randn(
        (total_blocks, BLOCK_SIZE, num_head_kv, QK_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    vcache = torch.randn(
        (total_blocks, BLOCK_SIZE, num_head_kv, V_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_ids = torch.zeros(
        (num_batch, int(num_blocks.max().item())), dtype=torch.int32, device="cuda"
    )

    block_offset = 0
    for batch_idx, batch_blocks in enumerate(num_blocks.tolist()):
        block_ids[batch_idx, :batch_blocks] = torch.arange(
            block_offset,
            block_offset + batch_blocks,
            dtype=torch.int32,
            device="cuda",
        )
        block_offset += batch_blocks

    return q, kcache, vcache, block_ids, total_lens


def reference(
    q, kcache, vcache, block_ids, total_lens, num_head_q, num_head_kv, num_seq_q
):
    num_batch = total_lens.numel()
    heads_per_group = num_head_q // num_head_kv
    q = q.reshape(num_batch, num_seq_q, num_head_q, QK_DIM)
    output = torch.empty(
        (num_batch, num_seq_q, num_head_q, V_DIM),
        dtype=torch.float32,
        device="cuda",
    )

    for batch_idx, total_len_tensor in enumerate(total_lens):
        total_len = int(total_len_tensor.item())
        batch_blocks = math.ceil(total_len / BLOCK_SIZE)
        pages = block_ids[batch_idx, :batch_blocks].long()
        k = kcache[pages].reshape(-1, num_head_kv, QK_DIM)[:total_len]
        v = vcache[pages].reshape(-1, num_head_kv, V_DIM)[:total_len]

        q_batch = q[batch_idx].transpose(0, 1).float()
        k = k.transpose(0, 1).repeat_interleave(heads_per_group, dim=0).float()
        v = v.transpose(0, 1).repeat_interleave(heads_per_group, dim=0).float()
        scores = torch.matmul(q_batch, k.transpose(-1, -2)) / math.sqrt(QK_DIM)
        key_positions = torch.arange(total_len, device="cuda")
        query_limits = total_len - num_seq_q + torch.arange(num_seq_q, device="cuda")
        causal_mask = key_positions.unsqueeze(0) <= query_limits.unsqueeze(1)
        probabilities = torch.softmax(scores.masked_fill(~causal_mask, -torch.inf), dim=-1)
        output[batch_idx] = torch.matmul(probabilities, v).transpose(0, 1)

    return output.reshape(num_batch * num_seq_q, num_head_q, V_DIM).to(torch.bfloat16)


def check_case(
    hpc,
    global_num_head_kv: int,
    num_seq_q: int,
    tp_size: int,
    splitk: bool,
    atol: float,
    rtol: float,
):
    local_num_head_q = GLOBAL_NUM_HEAD_Q // tp_size
    local_num_head_kv = max(global_num_head_kv // tp_size, 1)
    seed = 2026 + global_num_head_kv * 100 + num_seq_q * 10 + tp_size
    q, kcache, vcache, block_ids, total_lens = make_inputs(
        local_num_head_q, local_num_head_kv, num_seq_q, seed
    )
    expected = reference(
        q,
        kcache,
        vcache,
        block_ids,
        total_lens,
        local_num_head_q,
        local_num_head_kv,
        num_seq_q,
    )
    output = torch.empty(
        (q.size(0), local_num_head_q, V_DIM), dtype=torch.bfloat16, device="cuda"
    )
    actual = hpc.attention_decode_bf16(
        q,
        kcache,
        vcache,
        block_ids,
        total_lens,
        mtp=num_seq_q - 1,
        new_kv_included=True,
        splitk=splitk,
        output=output,
    )
    torch.cuda.synchronize()

    if actual.data_ptr() != output.data_ptr():
        raise AssertionError("HPC-Ops did not return the caller-provided output tensor")
    if not torch.isfinite(actual).all():
        raise AssertionError("HPC-Ops output contains NaN or Inf")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    print(
        f"CHECK_OK g={global_num_head_kv} Sq={num_seq_q} TP={tp_size} "
        f"local_Hq={local_num_head_q} local_Hkv={local_num_head_kv} "
        f"splitk_hint={splitk} "
        f"max_abs_error={max_abs_error:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 0):
        raise RuntimeError(f"HPC-Ops GQLA patch requires SM90, got {capability}")
    if args.expect_device_substring.lower() not in device_name.lower():
        raise RuntimeError(
            f"expected device containing {args.expect_device_substring!r}, got {device_name!r}"
        )

    hpc, build_lib = import_hpc(args.source_dir.resolve())
    print(
        f"device={device_name} torch={torch.__version__} hpc_ops={hpc.__version__} "
        f"build_lib={build_lib}",
        flush=True,
    )

    for global_num_head_kv, num_seq_q, tp_size in STATIC_CASES:
        check_case(
            hpc,
            global_num_head_kv,
            num_seq_q,
            tp_size,
            False,
            args.atol,
            args.rtol,
        )
    for global_num_head_kv, num_seq_q, tp_size in SPLITK_HINT_CASES:
        check_case(
            hpc,
            global_num_head_kv,
            num_seq_q,
            tp_size,
            True,
            args.atol,
            args.rtol,
        )

    print("HPC_OPS_GQLA_CORRECTNESS_OK cases=18", flush=True)


if __name__ == "__main__":
    main()
