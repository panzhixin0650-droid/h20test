#!/usr/bin/env python3
"""GPU and optional standalone extension verification for the official kernel build."""

from __future__ import annotations

import argparse
import importlib.metadata

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extensions", action="store_true")
    parser.add_argument("--expect-device-substring", default="H20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("torch import: OK", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    print(f"torch CUDA: {torch.version.cuda}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"visible GPUs: {torch.cuda.device_count()}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("This host does not expose a CUDA GPU")

    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device properties: {properties}", flush=True)
    device_name = torch.cuda.get_device_name(0)
    print(f"device name: {device_name}", flush=True)
    print(f"compute capability: {capability}", flush=True)

    if capability != (9, 0):
        raise RuntimeError(f"The pinned official build requires Hopper SM90; got {capability}")
    if args.expect_device_substring.lower() not in device_name.lower():
        raise RuntimeError(
            f"Expected a device containing {args.expect_device_substring!r}; "
            f"got {device_name!r}"
        )

    test_tensor = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    print(f"CUDA allocation: {test_tensor}", flush=True)
    print("GPU_OK", flush=True)

    if args.extensions:
        import flash_mla
        from flash_attn_3 import flash_attn_interface as fa3

        flash_mla_path = str(flash_mla.__file__)
        fa3_path = str(fa3.__file__)
        print(
            f"flash-mla: {importlib.metadata.version('flash-mla')} {flash_mla_path}",
            flush=True,
        )
        print(
            f"flash-attn-3: {importlib.metadata.version('flash-attn-3')} {fa3_path}",
            flush=True,
        )
        if "vllm" in flash_mla_path.lower() or "vllm" in fa3_path.lower():
            raise RuntimeError("A vLLM-vendored extension was imported")
        print("EXTENSIONS_OK", flush=True)


if __name__ == "__main__":
    main()
