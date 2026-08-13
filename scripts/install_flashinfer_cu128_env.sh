#!/usr/bin/env bash

# Build an isolated H20 environment for the FlashInfer GQA benchmark.  The
# FlashInfer wheel deliberately uses --no-deps so its unpinned `torch`
# dependency cannot replace the cu128 stack installed by install_cu128_env.sh.

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
conda_env_name=${FLASHINFER_GQA_CONDA_ENV:-flashinfer-gqa-cu128}
pypi_index=${FLASHINFER_GQA_PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}
fallback_index=${FLASHINFER_GQA_PYPI_FALLBACK_INDEX:-${GQLA_TABLE2_PYPI_FALLBACK_INDEX:-https://pypi.org/simple}}
flashinfer_version=${FLASHINFER_GQA_VERSION:-0.6.11.post2}
pip_timeout=${FLASHINFER_GQA_PIP_TIMEOUT:-300}
pip_retries=${FLASHINFER_GQA_PIP_RETRIES:-20}

if ! command -v conda >/dev/null 2>&1; then
    printf '%s\n' 'conda is required to create the isolated FlashInfer environment.' >&2
    exit 2
fi

# Keep the original Table 2 environment and any concurrently running setup
# untouched by giving this workload its own conda prefix.
GQLA_TABLE2_CONDA_ENV="$conda_env_name" \
GQLA_TABLE2_PYPI_INDEX="$pypi_index" \
GQLA_TABLE2_PYPI_FALLBACK_INDEX="$fallback_index" \
    bash "$script_dir/install_cu128_env.sh"

conda_env_prefix=$(conda env list | \
    awk -v target="$conda_env_name" '$1 == target {print $NF; exit}')
if [[ -z "$conda_env_prefix" || ! -x "$conda_env_prefix/bin/python" ]]; then
    printf 'Cannot resolve isolated conda environment %q.\n' "$conda_env_name" >&2
    exit 2
fi
python_bin="$conda_env_prefix/bin/python"

export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PIP_DEFAULT_TIMEOUT="$pip_timeout"
mirror_args=(
    --index-url "$pypi_index"
    --timeout "$pip_timeout"
    --retries "$pip_retries"
    --prefer-binary
)
fallback_args=(
    --index-url "$fallback_index"
    --timeout "$pip_timeout"
    --retries "$pip_retries"
    --prefer-binary
)

pip_install_with_fallback() {
    local rc
    if "$python_bin" -m pip install "${mirror_args[@]}" "$@"; then
        return 0
    else
        rc=$?
    fi
    if [[ -z "$fallback_index" || "$fallback_index" == "$pypi_index" ]]; then
        return "$rc"
    fi
    printf 'Primary PyPI mirror failed (rc=%s); retrying with %s\n' \
        "$rc" "$fallback_index" >&2
    "$python_bin" -m pip install "${fallback_args[@]}" "$@"
}

# Install every non-Torch direct dependency first.  None of these requirements
# asks pip to resolve Torch.  The core package is installed separately below.
pip_install_with_fallback \
    apache-tvm-ffi==0.1.9 \
    click==8.4.1 \
    cuda-tile==1.4.0 \
    einops==0.8.2 \
    ninja==1.13.0 \
    numpy==2.3.5 \
    nvidia-cudnn-frontend==1.18.0 \
    nvidia-cutlass-dsl==4.5.2 \
    nvidia-ml-py==13.610.43 \
    packaging==26.2 \
    requests==2.34.2 \
    tabulate==0.10.0 \
    tqdm==4.68.3 \
    cuda-python==12.9.4 \
    cuda-bindings==12.9.4

pip_install_with_fallback \
    --no-deps \
    "flashinfer-python==$flashinfer_version"

"$python_bin" -m pip check
"$python_bin" - "$flashinfer_version" <<'PY'
import sys

import flashinfer
import torch
import triton

expected_flashinfer = sys.argv[1]
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert triton.__version__ == "3.6.0", triton.__version__
assert flashinfer.__version__ == expected_flashinfer, flashinfer.__version__
print(
    "FLASHINFER_CU128_ENV_OK "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"triton={triton.__version__} flashinfer={flashinfer.__version__}",
    flush=True,
)
PY

printf 'Isolated environment: %s (%s)\n' "$conda_env_name" "$conda_env_prefix"
printf 'Python executable: %s\n' "$python_bin"
