#!/usr/bin/env bash

set -Eeuo pipefail

script_path=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")
script_dir=$(dirname -- "$script_path")
conda_env_name=${GQLA_TABLE2_CONDA_ENV:-h20table2}
tuna_index=${GQLA_TABLE2_PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}
torch_index=${GQLA_TABLE2_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
pip_timeout=${GQLA_TABLE2_PIP_TIMEOUT:-300}
pip_retries=${GQLA_TABLE2_PIP_RETRIES:-20}
cuda_home=${CUDA_HOME:-/usr/local/cuda}
flashinfer_version=${FLASHINFER_GQA_VERSION:-0.6.11.post2}

if [[ "${1:-}" != --inside-conda && "${CONDA_DEFAULT_ENV:-}" != "$conda_env_name" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        printf 'conda is unavailable; activate a Python 3.12 environment first.\n' >&2
        exit 2
    fi
    conda_env_prefix=$(conda env list | \
        awk -v target="$conda_env_name" '$1 == target {print $NF; exit}')
    if [[ -z "$conda_env_prefix" ]]; then
        conda_base=$(conda info --base)
        conda_default_prefix="$conda_base/envs/$conda_env_name"
        if [[ -d "$conda_default_prefix" ]]; then
            conda_env_prefix=$conda_default_prefix
        fi
    fi
    if [[ -z "$conda_env_prefix" ]]; then
        printf 'Creating conda environment: %s\n' "$conda_env_name"
        conda create -n "$conda_env_name" python=3.12 -y
        conda_env_prefix=$(conda env list | \
            awk -v target="$conda_env_name" '$1 == target {print $NF; exit}')
    else
        printf 'Reusing conda environment: %s (%s)\n' \
            "$conda_env_name" "$conda_env_prefix"
    fi
    if [[ -z "$conda_env_prefix" ]]; then
        printf 'Unable to resolve conda environment prefix: %s\n' \
            "$conda_env_name" >&2
        exit 2
    fi
    if [[ ! -x "$conda_env_prefix/bin/python" ]]; then
        printf 'Repairing Python in existing conda prefix: %s\n' \
            "$conda_env_prefix"
        conda install -p "$conda_env_prefix" python=3.12 -y
    fi
    test -x "$conda_env_prefix/bin/python"
    env \
        PATH="$conda_env_prefix/bin:$PATH" \
        CONDA_PREFIX="$conda_env_prefix" \
        CONDA_DEFAULT_ENV="$conda_env_name" \
        bash "$script_path" --inside-conda
    exit 0
fi

python_bin=$(command -v python)
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    printf 'Python is unavailable in the selected environment.\n' >&2
    exit 2
fi
if ! "$python_bin" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    "$python_bin" --version >&2
    printf 'This benchmark environment requires Python 3.12.\n' >&2
    exit 2
fi
if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
    printf 'nvcc is absent: %s/bin/nvcc\n' "$cuda_home" >&2
    exit 2
fi

cuda_release=$("$cuda_home/bin/nvcc" --version | \
    sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)
if [[ "$cuda_release" != 12.8* ]]; then
    printf 'Expected CUDA Toolkit 12.8, got %s from %s\n' \
        "${cuda_release:-unknown}" "$cuda_home" >&2
    exit 2
fi

# Ignore cluster-wide pip settings such as an unreachable pypi.nvidia mirror.
# Every install command below supplies its only intended package index.
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PIP_DEFAULT_TIMEOUT="$pip_timeout"

mirror_args=(
    --index-url "$tuna_index"
    --timeout "$pip_timeout"
    --retries "$pip_retries"
    --prefer-binary
)

printf 'Python: %s\n' "$python_bin"
printf 'CUDA Toolkit: %s\n' "$cuda_release"
printf 'PyPI mirror: %s\n' "$tuna_index"

"$python_bin" -m pip install "${mirror_args[@]}" --upgrade \
    pip \
    wheel \
    'setuptools<82' \
    packaging \
    ninja \
    psutil \
    uv \
    numpy \
    filelock \
    'typing-extensions>=4.10.0' \
    'sympy>=1.13.3' \
    'networkx>=2.5.1' \
    jinja2 \
    'fsspec>=0.8.5'

"$python_bin" -m pip install "${mirror_args[@]}" \
    'cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1' \
    cuda-bindings==12.9.4

"$python_bin" -m pip install "${mirror_args[@]}" \
    nvidia-cudnn-cu12==9.19.0.56 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nccl-cu12==2.28.9 \
    nvidia-nvshmem-cu12==3.4.5 \
    triton==3.6.0

# Resolve CUDA dependencies exclusively from TUNA above. Installing torch with
# --no-deps prevents the PyTorch index from redirecting cuda-toolkit downloads
# to a cluster-inaccessible NVIDIA package host.
"$python_bin" -m pip install \
    --index-url "$torch_index" \
    --timeout "$pip_timeout" \
    --retries "$pip_retries" \
    --prefer-binary \
    --no-deps \
    --upgrade \
    'torch==2.11.0+cu128'

"$python_bin" -m pip install "${mirror_args[@]}" \
    "flashinfer-python==$flashinfer_version"

"$python_bin" -m pip check
"$python_bin" -c \
    'import torch; assert torch.__version__ == "2.11.0+cu128", torch.__version__; assert torch.version.cuda == "12.8", torch.version.cuda; print(f"TORCH_OK version={torch.__version__} cuda={torch.version.cuda}", flush=True)'
"$python_bin" - "$flashinfer_version" <<'PY'
import sys

import flashinfer
import triton

assert flashinfer.__version__ == sys.argv[1], flashinfer.__version__
assert triton.__version__ == "3.6.0", triton.__version__
print(
    f"FLASHINFER_OK version={flashinfer.__version__} triton={triton.__version__}",
    flush=True,
)
PY
"$python_bin" -u "$script_dir/gpu_preflight.py" \
    --expect-device-substring H20

uv_bin=$(dirname -- "$python_bin")/uv
test -x "$uv_bin"
printf 'H20_CU128_ENV_OK\n'
printf 'Activate it in the current shell with:\n'
printf '  conda activate %s\n' "$conda_env_name"
printf '  export GQLA_TABLE2_PYTHON=%q\n' "$python_bin"
printf '  export GQLA_TABLE2_UV=%q\n' "$uv_bin"
