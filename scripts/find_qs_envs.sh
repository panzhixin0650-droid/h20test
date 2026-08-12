#!/usr/bin/env bash

# Read-only scan for reusable Python environments on a QS mounted data disk.

set -uo pipefail

if [[ -n "${1:-}" ]]; then
    scan_root=$1
elif [[ -d /mnt/nj-1/dataset/data ]]; then
    scan_root=/mnt/nj-1/dataset/data
else
    scan_root=/mnt
fi
max_depth=${GQLA_TABLE2_SCAN_DEPTH:-14}

if [[ ! -d "$scan_root" ]]; then
    printf 'Scan root is absent: %s\n' "$scan_root" >&2
    exit 2
fi
if [[ ! "$max_depth" =~ ^[1-9][0-9]*$ ]]; then
    printf 'GQLA_TABLE2_SCAN_DEPTH must be a positive integer\n' >&2
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    printf 'The timeout command is unavailable\n' >&2
    exit 2
fi

printf 'scan_root=%s max_depth=%s mode=read-only\n' \
    "$scan_root" "$max_depth"

candidate_count=0
compatible_count=0
compatible_pythons=()

while IFS= read -r -d '' candidate_python; do
    candidate_count=$((candidate_count + 1))
    candidate_root=${candidate_python%/bin/python}

    if [[ -f "$candidate_root/conda-meta/history" ]]; then
        candidate_kind=conda
    elif [[ -f "$candidate_root/pyvenv.cfg" ]]; then
        candidate_kind=venv
    else
        candidate_kind=python-prefix
    fi

    printf '\n============================================================\n'
    printf 'candidate=%s\nroot=%s\nkind=%s\npython=%s\n' \
        "$candidate_count" "$candidate_root" "$candidate_kind" \
        "$candidate_python"

    if [[ ! -x "$candidate_python" ]]; then
        printf 'status=broken-python\n'
        continue
    fi

    python_info=$(timeout 15 env \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
        "$candidate_python" -B -c \
        'import platform,sys; print("{}\t{}".format(platform.python_version(),sys.prefix))' \
        2>/dev/null)
    if [[ $? -ne 0 ]]; then
        printf 'status=python-start-failed-or-timeout\n'
        continue
    fi
    IFS=$'\t' read -r python_version python_prefix <<<"$python_info"
    printf 'python_version=%s\npython_prefix=%s\n' \
        "$python_version" "$python_prefix"

    torch_info=$(timeout 30 env \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_CACHE_DISABLE=1 \
        "$candidate_python" -B -c \
        'import torch; print("{}\t{}\t{}\t{}\t{}".format(torch.__version__,torch.version.cuda,"yes" if torch.cuda.is_available() else "no",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",torch._C._GLIBCXX_USE_CXX11_ABI))' \
        2>/dev/null)
    if [[ $? -ne 0 ]]; then
        printf 'torch=missing-or-import-failed\n'
        printf 'verdict=not-compatible\n'
        continue
    fi
    IFS=$'\t' read -r torch_version torch_cuda cuda_available \
        gpu_name cxx11abi <<<"$torch_info"
    printf 'torch=%s\ntorch_cuda=%s\ncuda_available=%s\ngpu=%s\ncxx11abi=%s\n' \
        "$torch_version" "$torch_cuda" "$cuda_available" \
        "$gpu_name" "$cxx11abi"

    if triton_version=$(timeout 15 env \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
        "$candidate_python" -B -c \
        'import triton; print(triton.__version__)' 2>/dev/null); then
        printf 'triton=%s\n' "$triton_version"
    else
        triton_version=missing
        printf 'triton=missing-or-import-failed\n'
    fi

    if flashmla_path=$(timeout 20 env \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_CACHE_DISABLE=1 \
        "$candidate_python" -B -c \
        'import flash_mla; print(flash_mla.__file__)' 2>/dev/null); then
        printf 'flash_mla=%s\n' "$flashmla_path"
    else
        flashmla_path=missing
        printf 'flash_mla=missing-or-import-failed\n'
    fi

    if fa3_path=$(timeout 20 env \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_CACHE_DISABLE=1 \
        "$candidate_python" -B -c \
        'from flash_attn_3 import flash_attn_interface as fa3; print(fa3.__file__)' \
        2>/dev/null); then
        printf 'fa3=%s\n' "$fa3_path"
    else
        fa3_path=missing
        printf 'fa3=missing-or-import-failed\n'
    fi

    if [[ "$python_version" == 3.12.* && \
          "$torch_version" == 2.11.0+cu128 && \
          "$torch_cuda" == 12.8 && \
          "$cuda_available" == yes ]]; then
        compatible_count=$((compatible_count + 1))
        compatible_pythons+=("$candidate_python")
        printf 'verdict=compatible-build-base\n'
        if [[ "$triton_version" != 3.6.0 ]]; then
            printf 'note=install-triton-3.6.0\n'
        fi
        if [[ "$flashmla_path" != missing && "$fa3_path" != missing ]]; then
            printf 'kernel_imports=both-ok-rebuild-pinned-commits-recommended\n'
        else
            printf 'kernel_imports=incomplete-build-required\n'
        fi
        printf 'export GQLA_TABLE2_PYTHON=%q\n' "$candidate_python"
        if [[ -x "$candidate_root/bin/uv" ]]; then
            printf 'export GQLA_TABLE2_UV=%q\n' "$candidate_root/bin/uv"
        else
            printf 'uv=missing\n'
        fi
    else
        printf 'verdict=not-compatible-with-python312-torch211-cu128\n'
    fi
done < <(
    find "$scan_root" -maxdepth "$max_depth" \
        \( -type d \( \
            -name site-packages -o \
            -name node_modules -o \
            -name __pycache__ -o \
            -name .git -o \
            -name .cache -o \
            -name wandb \
        \) -prune \) -o \
        \( \( -type f -o -type l \) \
            -path '*/bin/python' -print0 \) \
        2>/dev/null | sort -zu
)

printf '\n============================================================\n'
printf 'scan_complete candidates=%s compatible=%s\n' \
    "$candidate_count" "$compatible_count"

if (( compatible_count > 0 )); then
    printf 'compatible_python_paths:\n'
    for candidate_python in "${compatible_pythons[@]}"; do
        printf '  %s\n' "$candidate_python"
    done
else
    printf 'No directly compatible cu128 environment was found.\n'
fi
