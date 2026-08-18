#!/usr/bin/env bash
# Deploy adaptive split-K sources and rebuild HPC-Ops inside an already
# complete H20 serving environment. This path is deliberately no-index and
# never installs Torch, vLLM, Transformers, CUDA wheels, Python, or uv.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEPLOY_SCRIPT=$SCRIPT_DIR/deploy_h20_vllm.sh
MODEL_NAME=dsv3p1_g8_sim_hess_no_mean_subtract
EXPECTED_HPC_COMMIT=de202c9bda942fdfd499d09e51ea6ff9c89c5d50

infer_gqla_root() {
    if [[ -f "/mnt/public03/task/236362/GQLA/outputs/convert/$MODEL_NAME/config.json" ]]; then
        printf '%s\n' /mnt/public03/task/236362/GQLA
    else
        printf '%s\n' /mnt/tidalfs-alwl01/task/236362/GQLA
    fi
}

die() {
    echo "error: $*" >&2
    exit 2
}

version_ge() {
    local actual=$1 minimum=$2
    [[ "$(printf '%s\n' "$minimum" "$actual" | sort -V | head -n1)" == "$minimum" ]]
}

core_stack_is_ready() {
    local python=$1
    [[ -x "$python" ]] || return 1
    env BASH_ENV=/dev/null ENV=/dev/null "$python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
import torch

base = lambda value: value.split("+", 1)[0]
assert base(version("torch")) == "2.11.0"
assert torch.version.cuda and (
    torch.version.cuda.startswith("12.9") or torch.version.cuda.startswith("13.0")
)
assert base(version("vllm")) == "0.22.1"
assert base(version("transformers")) == "5.12.1"
PY
}

runtime_cuda_is_ready() {
    local python=$1 runtime_file=$2
    [[ -f "$runtime_file" ]] || return 1
    RUNTIME_CHECK_FILE="$runtime_file" RUNTIME_CHECK_PYTHON="$python" \
        BASH_ENV=/dev/null ENV=/dev/null \
        /bin/bash --noprofile --norc >/dev/null 2>&1 <<'BASH'
set -Eeuo pipefail
source "$RUNTIME_CHECK_FILE"
"$RUNTIME_CHECK_PYTHON" <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(8))
PY
BASH
}

report_stack() {
    local python=$1 runtime_file=$2
    env BASH_ENV=/dev/null ENV=/dev/null "$python" - "$runtime_file" <<'PY' 2>&1 || true
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

def package(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"

try:
    import torch
    torch_version = torch.__version__
    torch_cuda = torch.version.cuda
except Exception as exc:
    torch_version = f"import-error:{type(exc).__name__}"
    torch_cuda = "unknown"

print(
    "H20_ENV_CANDIDATE",
    f"torch={torch_version}",
    f"torch_cuda={torch_cuda}",
    f"vllm={package('vllm')}",
    f"transformers={package('transformers')}",
    f"runtime_env={Path(sys.argv[1]).is_file()}",
)
PY
}

find_cuda_toolkit() {
    local candidate release
    local -a candidates=()

    [[ -n "${CUDA_HOME:-}" ]] && candidates+=("$CUDA_HOME")
    if command -v nvcc >/dev/null 2>&1; then
        candidates+=("$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)")
    fi
    candidates+=(/usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda)
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] && candidates+=("$candidate")
    done < <("$PYTHON" - <<'PY'
import site
from pathlib import Path

seen = set()
for site_dir in site.getsitepackages():
    for root in sorted(Path(site_dir).glob("nvidia/cu*"), reverse=True):
        if (root / "bin" / "nvcc").is_file() and str(root) not in seen:
            print(root)
            seen.add(str(root))
PY
)

    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate/bin/nvcc" && -f "$candidate/include/cuda.h" ]] || continue
        release=$("$candidate/bin/nvcc" --version 2>/dev/null \
            | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)
        if [[ -n "$release" ]] \
            && version_ge "$release" 12.8 \
            && [[ "${release%%.*}" == "$TORCH_CUDA_MAJOR" ]]; then
            CUDA_TOOLKIT_ROOT=$(cd "$candidate" && pwd -P)
            CUDA_TOOLKIT_VERSION=$release
            return 0
        fi
    done
    return 1
}

hpc_source_hash() {
    find \
        "$HPC_OPS_DIR/CMakeLists.txt" \
        "$HPC_OPS_DIR/setup.py" \
        "$HPC_OPS_DIR/hpc" \
        "$HPC_OPS_DIR/src" \
        "$HPC_OPS_DIR/3rd/cutlass/include" \
        -type f ! -name '*.pyc' ! -path "$HPC_OPS_DIR/hpc/version.py" -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{print $1}'
}

GQLA_ROOT=${GQLA_ROOT:-$(infer_gqla_root)}
CODE_ROOT=${CODE_ROOT:-$GQLA_ROOT/code}
ENV_ROOT=${ENV_ROOT:-$GQLA_ROOT/envs}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/$MODEL_NAME}
GQLA_REPO=${GQLA_REPO:-$CODE_ROOT/GQLA_preprint}
HPC_OPS_DIR=${HPC_OPS_DIR:-$CODE_ROOT/hpc-ops-de202c9}
MAX_JOBS=${MAX_JOBS:-16}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
requested_venv=${VENV_DIR:-}
requested_runtime=${RUNTIME_ENV_FILE:-}

[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || die "MAX_JOBS must be a positive integer"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] \
    || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -f "$DEPLOY_SCRIPT" ]] || die "deployment helper is missing: $DEPLOY_SCRIPT"
[[ -f "$MODEL_DIR/config.json" ]] || die "converted model is missing: $MODEL_DIR/config.json"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] \
    || die "converted model index is missing: $MODEL_DIR/model.safetensors.index.json"

selected_venv=
if [[ -n "$requested_venv" ]]; then
    candidate_runtime=${requested_runtime:-$requested_venv/h20-runtime.env}
    if core_stack_is_ready "$requested_venv/bin/python" \
        && runtime_cuda_is_ready "$requested_venv/bin/python" "$candidate_runtime"; then
        selected_venv=$requested_venv
        BASE_RUNTIME_ENV_FILE=$candidate_runtime
    else
        echo "[hpc-only] rejected explicit VENV_DIR=$requested_venv" >&2
        [[ -x "$requested_venv/bin/python" ]] \
            && report_stack "$requested_venv/bin/python" "$candidate_runtime" >&2
        die "existing-env update requires torch 2.11 with CUDA 12.9/13.0, vLLM 0.22.1, Transformers 5.12.1, and h20-runtime.env"
    fi
else
    declare -A seen=()
    candidates=(
        "$ENV_ROOT/venv-py312"
        "$ENV_ROOT/h20-new"
        "$ENV_ROOT/h20-cu129-py312"
    )
    while IFS= read -r -d '' python; do
        candidates+=("${python%/bin/python}")
    done < <(find "$ENV_ROOT" -mindepth 2 -maxdepth 3 \
        -path '*/bin/python' -print0 2>/dev/null | sort -z)

    for candidate in "${candidates[@]}"; do
        [[ -n "${seen[$candidate]:-}" ]] && continue
        seen[$candidate]=1
        candidate_runtime=$candidate/h20-runtime.env
        if core_stack_is_ready "$candidate/bin/python" \
            && runtime_cuda_is_ready "$candidate/bin/python" "$candidate_runtime"; then
            selected_venv=$candidate
            BASE_RUNTIME_ENV_FILE=$candidate_runtime
            break
        fi
    done

    if [[ -z "$selected_venv" ]]; then
        echo "[hpc-only] no compatible serving environment with h20-runtime.env under $ENV_ROOT" >&2
        for candidate in "${candidates[@]}"; do
            [[ -x "$candidate/bin/python" ]] || continue
            candidate_runtime=$candidate/h20-runtime.env
            echo "[hpc-only] python=$candidate/bin/python" >&2
            report_stack "$candidate/bin/python" "$candidate_runtime" >&2
        done
        die "refusing all dependency downloads; provide the prior formal VENV_DIR/runtime or relay an environment through OSS"
    fi
fi

echo "[hpc-only] selected VENV_DIR=$selected_venv"
report_stack "$selected_venv/bin/python" "$BASE_RUNTIME_ENV_FILE"
echo "[hpc-only] dependency downloads are disabled; only local source deployment and HPC-Ops rebuild may run"

target_gqla_root=$GQLA_ROOT
target_code_root=$CODE_ROOT
target_env_root=$ENV_ROOT
target_model_dir=$MODEL_DIR
target_gqla_repo=$GQLA_REPO
target_hpc_ops_dir=$HPC_OPS_DIR

# The prior runtime carries the CUDA 13 forward-compatibility library and the
# exact toolkit path used by the successful H20 serving environment.
# shellcheck disable=SC1090
source "$BASE_RUNTIME_ENV_FILE"

GQLA_ROOT=$target_gqla_root
CODE_ROOT=$target_code_root
ENV_ROOT=$target_env_root
MODEL_DIR=$target_model_dir
GQLA_REPO=$target_gqla_repo
HPC_OPS_DIR=$target_hpc_ops_dir
VENV_DIR=$selected_venv
PYTHON=$VENV_DIR/bin/python
SPLITK_RUNTIME_ENV_FILE=$VENV_DIR/h20-splitk-runtime.env
BUILD_LOG=$GQLA_ROOT/outputs/logs/h20-splitk-build-$(date -u +%Y%m%dT%H%M%SZ)-$$.log

export GQLA_ROOT CODE_ROOT ENV_ROOT MODEL_DIR GQLA_REPO HPC_OPS_DIR VENV_DIR PYTHON
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

core_stack_is_ready "$PYTHON" || die "selected environment changed after sourcing its runtime"
TORCH_CUDA_VERSION=$("$PYTHON" -c 'import torch; print(torch.version.cuda)')
TORCH_CUDA_MAJOR=${TORCH_CUDA_VERSION%%.*}
command -v cmake >/dev/null 2>&1 || die "existing environment lacks cmake; refusing to download it"
command -v ninja >/dev/null 2>&1 || die "existing environment lacks ninja; refusing to download it"
"$PYTHON" - <<'PY' >/dev/null 2>&1 \
    || die "existing environment lacks setuptools/wheel; refusing to download them"
import setuptools
import wheel
PY
command -v g++ >/dev/null 2>&1 || die "g++ is required to compile HPC-Ops"

pip_is_ready=1
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    pip_is_ready=0
    "$PYTHON" -c 'import ensurepip' >/dev/null 2>&1 \
        || die "existing environment lacks pip and Python's offline ensurepip bootstrap"
    echo "[hpc-only] pip is absent; the actual update will bootstrap it offline with ensurepip"
fi

CUDA_TOOLKIT_ROOT=
CUDA_TOOLKIT_VERSION=
find_cuda_toolkit \
    || die "existing runtime does not expose nvcc >=12.8 matching torch CUDA $TORCH_CUDA_VERSION"
echo "[hpc-only] CUDA toolkit=$CUDA_TOOLKIT_ROOT version=$CUDA_TOOLKIT_VERSION"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    echo "H20_SPLITK_PREFLIGHT_OK venv=$VENV_DIR runtime_env=$BASE_RUNTIME_ENV_FILE torch_cuda=$TORCH_CUDA_VERSION cuda_toolkit=$CUDA_TOOLKIT_ROOT"
    exit 0
fi

if [[ "$pip_is_ready" == 0 ]]; then
    "$PYTHON" -m ensurepip --upgrade --default-pip \
        || die "offline pip bootstrap failed"
    "$PYTHON" -m pip --version >/dev/null 2>&1 \
        || die "pip is still unavailable after offline ensurepip bootstrap"
fi

echo "[hpc-only] materializing the verified GQLA and HPC-Ops sources"
GQLA_ROOT="$GQLA_ROOT" \
CODE_ROOT="$CODE_ROOT" \
ENV_ROOT="$ENV_ROOT" \
MODEL_DIR="$MODEL_DIR" \
GQLA_DST="$GQLA_REPO" \
HPC_DST="$HPC_OPS_DIR" \
RUN_INSTALL=0 \
bash "$DEPLOY_SCRIPT"

current_commit=$(git -C "$HPC_OPS_DIR" rev-parse --verify HEAD 2>/dev/null || true)
if [[ -z "$current_commit" && -f "$HPC_OPS_DIR/.gqla_hpc_source_commit" ]]; then
    current_commit=$(<"$HPC_OPS_DIR/.gqla_hpc_source_commit")
fi
[[ "$current_commit" == "$EXPECTED_HPC_COMMIT" ]] \
    || die "HPC-Ops source is $current_commit; expected $EXPECTED_HPC_COMMIT"

"$PYTHON" -m pip install --no-index --disable-pip-version-check \
    --no-deps --no-build-isolation --editable "$GQLA_REPO"

export CUDA_HOME=$CUDA_TOOLKIT_ROOT
export CUDAToolkit_ROOT=$CUDA_TOOLKIT_ROOT
export CUDACXX=$CUDA_TOOLKIT_ROOT/bin/nvcc
export PATH="$VENV_DIR/bin:$CUDA_TOOLKIT_ROOT/bin:$PATH"
torch_lib=$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
if [[ -d "$CUDA_TOOLKIT_ROOT/lib" ]]; then
    toolkit_lib=$CUDA_TOOLKIT_ROOT/lib
else
    toolkit_lib=$CUDA_TOOLKIT_ROOT/lib64
fi
export LD_LIBRARY_PATH="$toolkit_lib:$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$("$PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
export MAX_JOBS

source_hash=$(hpc_source_hash)
build_root=$(mktemp -d /tmp/gqla-hpc-splitk-build.XXXXXX)
wheelhouse=$build_root/wheelhouse
persistent_wheel_dir=$ENV_ROOT/wheels/hpc-splitk/$source_hash
mkdir -p "$wheelhouse" "$persistent_wheel_dir" "$(dirname "$BUILD_LOG")"

echo "[hpc-only] building adaptive split-K HPC-Ops for the existing torch/CUDA ABI"
echo "[hpc-only] source_hash=$source_hash build_root=$build_root log=$BUILD_LOG"
if ! (
    cd "$HPC_OPS_DIR"
    export HPC_GIT_HASH_OVERRIDE=${source_hash:0:7}
    "$PYTHON" setup.py \
        build --build-base "$build_root/build" \
        bdist_wheel --dist-dir "$wheelhouse" --bdist-dir "$build_root/bdist"
) 2>&1 | tee "$BUILD_LOG"; then
    die "HPC-Ops build failed; inspect $BUILD_LOG (temporary objects retained at $build_root)"
fi

mapfile -t hpc_wheels < <(find "$wheelhouse" -maxdepth 1 -type f -name 'hpc_ops-*.whl' -print)
(( ${#hpc_wheels[@]} == 1 )) \
    || die "expected one HPC-Ops wheel, found ${#hpc_wheels[@]} under $wheelhouse"
install -m 0644 "${hpc_wheels[0]}" "$persistent_wheel_dir/"
installed_wheel=$persistent_wheel_dir/$(basename "${hpc_wheels[0]}")
"$PYTHON" -m pip install --no-index --disable-pip-version-check \
    --no-deps --force-reinstall "$installed_wheel"
printf '%s\n' "$source_hash" >"$VENV_DIR/.hpc-source.sha256"

"$PYTHON" - "$MODEL_DIR" <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch

model_dir = Path(sys.argv[1])
with (model_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
    index = json.load(handle)
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (model_dir / name).is_file()]
assert not missing, missing[:8]

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
capabilities = [torch.cuda.get_device_capability(i) for i in range(8)]
assert all(cap == (9, 0) for cap in capabilities), capabilities
for device in range(8):
    torch.empty(1, device=f"cuda:{device}")
torch.cuda.synchronize()

import src.vllm_register_dsv  # noqa: E402,F401
from vllm import ModelRegistry  # noqa: E402

required = {"DeepseekV3GQLAForCausalLM", "DeepseekV3GQLAHPCForCausalLM"}
missing_arches = required - set(ModelRegistry.get_supported_archs())
assert not missing_arches, missing_arches

import hpc  # noqa: E402

schema = str(torch.ops.hpc.attention_decode_bf16.default._schema)
assert "softmax_scale" in schema, schema
assert "use_splitk" in schema, schema
print(
    "H20_SPLITK_RUNTIME_OK",
    f"torch={torch.__version__}",
    f"torch_cuda={torch.version.cuda}",
    f"vllm={version('vllm')}",
    f"hpc={hpc.__version__}",
    f"gpus={torch.cuda.device_count()}",
    f"capabilities={capabilities}",
    f"checkpoint_shards={len(shards)}",
    f"schema={schema}",
)
PY

{
    printf 'source %q\n' "$BASE_RUNTIME_ENV_FILE"
    printf 'export H20_GQLA_EXISTING_RUNTIME=1\n'
    printf 'export GQLA_ROOT=%q\n' "$GQLA_ROOT"
    printf 'export MODEL_DIR=%q\n' "$MODEL_DIR"
    printf 'export CODE_ROOT=%q\n' "$CODE_ROOT"
    printf 'export GQLA_REPO=%q\n' "$GQLA_REPO"
    printf 'export HPC_OPS_DIR=%q\n' "$HPC_OPS_DIR"
    printf 'export ENV_ROOT=%q\n' "$ENV_ROOT"
    printf 'export VENV_DIR=%q\n' "$VENV_DIR"
    printf 'export PYTHON=%q\n' "$PYTHON"
    printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$GQLA_REPO"
} >"$SPLITK_RUNTIME_ENV_FILE"

cat >"$VENV_DIR/h20-splitk-manifest.env" <<EOF
updated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
base_runtime_env=$BASE_RUNTIME_ENV_FILE
splitk_runtime_env=$SPLITK_RUNTIME_ENV_FILE
model_dir=$MODEL_DIR
gqla_repo=$GQLA_REPO
hpc_ops_dir=$HPC_OPS_DIR
venv_dir=$VENV_DIR
python=$PYTHON
torch_cuda=$($PYTHON -c 'import torch; print(torch.version.cuda)')
cuda_toolkit=$CUDA_TOOLKIT_ROOT
hpc_source_commit=$EXPECTED_HPC_COMMIT
hpc_source_hash=$source_hash
hpc_wheel=$installed_wheel
build_log=$BUILD_LOG
EOF

find "$build_root" -depth -delete
echo "H20_SPLITK_HPC_ONLY_OK runtime_env=$SPLITK_RUNTIME_ENV_FILE source_hash=$source_hash"
