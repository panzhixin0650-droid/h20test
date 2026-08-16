#!/usr/bin/env bash
set -Eeuo pipefail

R=${GQLA_ROOT:-/mnt/public03/task/236362/GQLA}
E=${CONDA_ENV_DIR:-$R/envs/h20-conda-py312}
G=$R/code/GQLA_preprint
H=$R/code/hpc-ops
M=https://mirrors.aliyun.com/pypi/simple
D=$R/envs/downloads/cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb

command -v conda >/dev/null || { echo "conda not found"; exit 2; }
BASE=${BASE_CONDA_ENV:-${CONDA_PREFIX:-$(conda info --base)}}
[[ -f $G/pyproject.toml ]] || { echo "missing $G"; exit 2; }
[[ -f $H/setup.py ]] || { echo "missing $H"; exit 2; }

if [[ -d $E ]]; then
    echo "removing old conda environment: $E"
    conda env remove -y -p "$E"
fi

echo "cloning local Conda base without downloading Conda packages"
conda create -y -p "$E" --clone "$BASE"
P=$E/bin/python
"$P" - <<'PY'
import sys

if not (sys.version_info[:2] >= (3, 10) and sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"Python 3.10-3.12 is required; cloned {sys.version.split()[0]}")
PY

echo "installing Torch/vLLM from Aliyun PyPI"
"$P" -m pip --isolated install -i "$M" -U pip setuptools wheel
"$P" -m pip --isolated install -i "$M" \
    torch==2.11.0 vllm==0.22.1 transformers==5.12.1 cmake ninja

echo "installing local GQLA plugin"
"$P" -m pip install --no-deps --no-build-isolation -e "$G"

if [[ -f $D ]]; then
    X=$R/envs/cuda-compat-13
    mkdir -p "$X"
    dpkg-deb -x "$D" "$X"
    L=$X/usr/local/cuda-13.0/compat
    export LD_LIBRARY_PATH=$L${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    mkdir -p "$E/etc/conda/activate.d"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$L" \
        >"$E/etc/conda/activate.d/cuda-compat.sh"
fi

export PATH=$E/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export CUDACXX=$CUDA_HOME/bin/nvcc
export CMAKE_PREFIX_PATH=$("$P" -c 'import torch; print(torch.utils.cmake_prefix_path)')
export CMAKE_GENERATOR=Ninja

echo "building patched HPC-Ops"
B=$E/hpc-build
W=$E/hpc-wheel
mkdir -p "$W"
(
    cd "$H"
    "$P" setup.py build --build-base "$B" \
        bdist_wheel --dist-dir "$W" --bdist-dir "$E/hpc-bdist"
)
shopt -s nullglob
wheels=("$W"/hpc_ops-*.whl)
[[ ${#wheels[@]} == 1 ]] || { echo "HPC wheel build failed"; exit 2; }
"$P" -m pip install --force-reinstall --no-deps "${wheels[0]}"

"$P" - <<'PY'
from importlib.metadata import version
import torch
import hpc
import vllm

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(8))
schema = str(torch.ops.hpc.attention_decode_bf16.default._schema)
assert "softmax_scale" in schema
print("H20_CONDA_SETUP_OK", version("torch"), version("vllm"), schema)
PY

echo "activate with: conda activate $E"
