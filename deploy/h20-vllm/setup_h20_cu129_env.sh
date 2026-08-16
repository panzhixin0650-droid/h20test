#!/usr/bin/env bash
# Create a clean CUDA 12.9 vLLM environment and build the patched HPC-Ops
# extension for the exact Torch ABI. Large release wheels are fetched with
# aria2; the converted model is only validated and is never downloaded.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEPLOY_SCRIPT=$SCRIPT_DIR/deploy_h20_vllm.sh
MODEL_NAME=dsv3p1_g8_sim_hess_no_mean_subtract

infer_gqla_root() {
    local probe
    probe=$SCRIPT_DIR
    while [[ "$probe" != / ]]; do
        if [[ -f "$probe/outputs/convert/$MODEL_NAME/config.json" ]]; then
            printf '%s\n' "$probe"
            return 0
        fi
        probe=$(dirname "$probe")
    done
    if [[ -f "/mnt/public03/task/236362/GQLA/outputs/convert/$MODEL_NAME/config.json" ]]; then
        printf '%s\n' /mnt/public03/task/236362/GQLA
    else
        printf '%s\n' /mnt/tidalfs-alwl01/task/236362/GQLA
    fi
}

GQLA_ROOT=${GQLA_ROOT:-$(infer_gqla_root)}
CODE_ROOT=${CODE_ROOT:-$GQLA_ROOT/code}
ENV_ROOT=${ENV_ROOT:-$GQLA_ROOT/envs}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/$MODEL_NAME}
GQLA_REPO=${GQLA_REPO:-$CODE_ROOT/GQLA_preprint}
HPC_OPS_DIR=${HPC_OPS_DIR:-$CODE_ROOT/hpc-ops}
VENV_DIR=${VENV_DIR:-$ENV_ROOT/h20-cu129-py312}
WHEEL_DIR=${WHEEL_DIR:-$ENV_ROOT/wheels/cu129-vllm-0.22.1}
UV_CACHE_DIR=${UV_CACHE_DIR:-$GQLA_ROOT/.cache/uv-cu129}
RUNTIME_ENV_FILE=${RUNTIME_ENV_FILE:-$VENV_DIR/h20-runtime.env}
SETUP_LOG=${SETUP_LOG:-$ENV_ROOT/logs/setup-h20-cu129.log}

PYTHON_VERSION=3.12.12
TORCH_VERSION=2.11.0
VLLM_VERSION=0.22.1
TRANSFORMERS_VERSION=5.12.1
TORCH_WHEEL_NAME=torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
VLLM_WHEEL_NAME=vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
TORCH_WHEEL_URL=${TORCH_WHEEL_URL:-https://download.pytorch.org/whl/cu129/torch-2.11.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl}
VLLM_WHEEL_URL=${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}
TORCH_WHEEL_SHA256=${TORCH_WHEEL_SHA256:-68b83cb7d7d43bc67c2833c8aebaea6a966f2017c3389885affa3361c258b7e3}
VLLM_WHEEL_SHA256=${VLLM_WHEEL_SHA256:-365ee929afd73bb5d146235b65053fa948788ec2ee00a2c3e957d3f43bf2b0cd}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}

ARIA2_CONNECTIONS=${ARIA2_CONNECTIONS:-16}
ARIA2_CONCURRENT_FILES=${ARIA2_CONCURRENT_FILES:-4}
ARIA2_ACCELERATE_DEPS=${ARIA2_ACCELERATE_DEPS:-1}
ARIA2_MIN_WHEEL_MIB=${ARIA2_MIN_WHEEL_MIB:-32}
UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-16}
UV_CONCURRENT_INSTALLS=${UV_CONCURRENT_INSTALLS:-16}
MAX_JOBS=${MAX_JOBS:-16}
FORCE_RECREATE=${FORCE_RECREATE:-0}
FORCE_HPC_REBUILD=${FORCE_HPC_REBUILD:-0}
DRY_RUN=${DRY_RUN:-0}

# Exact large dependencies selected by the torch 2.11.0/cu129 and vLLM 0.22.1
# release metadata. At setup time official PyPI JSON metadata supplies the
# actual wheel URL, byte size, and SHA256. Only wheels above the configurable
# size threshold are handed to aria2; uv resolves and installs everything.
PYPI_ACCELERATED_SPECS=(
    nvidia-cublas-cu12==12.9.1.4
    nvidia-cuda-cupti-cu12==12.9.79
    nvidia-cuda-nvrtc-cu12==12.9.86
    nvidia-cuda-runtime-cu12==12.9.79
    nvidia-cudnn-cu12==9.17.1.4
    nvidia-cufft-cu12==11.4.1.4
    nvidia-cufile-cu12==1.14.1.1
    nvidia-curand-cu12==10.3.10.19
    nvidia-cusolver-cu12==11.7.5.82
    nvidia-cusparse-cu12==12.5.10.65
    nvidia-cusparselt-cu12==0.7.1
    nvidia-nccl-cu12==2.28.9
    nvidia-nvjitlink-cu12==12.9.86
    nvidia-nvshmem-cu12==3.4.5
    triton==3.6.0
    flashinfer-cubin==0.6.11.post2
    nvidia-cutlass-dsl==4.5.2
)

die() {
    echo "error: $*" >&2
    exit 2
}

require_bool() {
    local name=$1 value=$2
    case "$value" in
        0|1) ;;
        *) die "$name must be 0 or 1; got $value" ;;
    esac
}

version_ge() {
    local actual=$1 minimum=$2
    [[ "$(printf '%s\n' "$minimum" "$actual" | sort -V | head -n1)" == "$minimum" ]]
}

sanitize_cuda13_compat() {
    local entry cleaned=
    local -a entries=()
    IFS=: read -r -a entries <<<"${LD_LIBRARY_PATH:-}"
    for entry in "${entries[@]}"; do
        [[ -n "$entry" ]] || continue
        case "$entry" in
            */cuda-compat-*/usr/local/cuda-13.0/compat) continue ;;
        esac
        cleaned=${cleaned:+$cleaned:}$entry
    done
    export LD_LIBRARY_PATH=$cleaned
}

ensure_aria2() {
    if command -v aria2c >/dev/null 2>&1; then
        return 0
    fi
    if [[ "$(id -u)" == 0 ]] && command -v apt-get >/dev/null 2>&1; then
        echo "[setup] aria2c is missing; installing the small aria2 host package"
        DEBIAN_FRONTEND=noninteractive apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends aria2
    fi
    command -v aria2c >/dev/null 2>&1 \
        || die "aria2c is required (install package 'aria2' or run setup as root once)"
}

wheel_archive_is_valid() {
    local wheel=$1
    [[ -s "$wheel" ]] || return 1
    "$BASE_PYTHON" - "$wheel" <<'PY' >/dev/null 2>&1
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
assert any(name.endswith(".dist-info/METADATA") for name in names)
PY
}

sha256_is_valid() {
    local path=$1 expected=$2 actual
    [[ -n "$expected" ]] || return 0
    [[ -s "$path" ]] || return 1
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]]
}

wheel_is_valid() {
    local wheel=$1 expected_sha256=${2:-}
    wheel_archive_is_valid "$wheel" && sha256_is_valid "$wheel" "$expected_sha256"
}

move_bad_download() {
    local path=$1 suffix=$2
    if [[ -e "$path" ]]; then
        mv "$path" "$path.bad.$suffix"
    fi
    if [[ -e "$path.aria2" ]]; then
        mv "$path.aria2" "$path.aria2.bad.$suffix"
    fi
}

append_aria2_entry() {
    local input_file=$1 url=$2 output_name=$3 checksum=${4:-}
    {
        printf '%s\n' "$url"
        printf '  dir=%s\n' "$WHEEL_DIR"
        printf '  out=%s\n' "$output_name"
        if [[ -n "$checksum" ]]; then
            printf '  checksum=sha-256=%s\n' "$checksum"
        fi
    } >>"$input_file"
}

write_pypi_accelerated_metadata() {
    local output_file=$1
    if [[ "$ARIA2_ACCELERATE_DEPS" != 1 ]]; then
        : >"$output_file"
        return 0
    fi
    "$BASE_PYTHON" - "$ARIA2_MIN_WHEEL_MIB" "${PYPI_ACCELERATED_SPECS[@]}" \
        >"$output_file" <<'PY'
import concurrent.futures
import json
import re
import sys
import urllib.parse
import urllib.request

minimum_bytes = int(sys.argv[1]) * 1024 * 1024
specs = sys.argv[2:]


def compatible(filename: str) -> bool:
    lower = filename.lower()
    if not lower.endswith(".whl"):
        return False
    if any(token in lower for token in ("aarch64", "ppc64", "s390x", "win_", "musllinux")):
        return False
    if not ("x86_64" in lower or lower.endswith("-none-any.whl")):
        return False
    match = re.search(r"-cp(\d{2,3})-([^-]+)-", lower)
    if match and match.group(1) != "312" and match.group(2) != "abi3":
        return False
    return True


def resolve(spec: str):
    name, version = spec.split("==", 1)
    quoted_name = urllib.parse.quote(name, safe="")
    quoted_version = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{quoted_name}/{quoted_version}/json",
        headers={"User-Agent": "gqla-h20-aria2-resolver/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"[setup] warning: could not resolve {spec} for aria2: {exc}", file=sys.stderr)
        return None
    candidates = [
        item
        for item in payload.get("urls", [])
        if item.get("packagetype") == "bdist_wheel"
        and compatible(item.get("filename", ""))
        and int(item.get("size") or 0) >= minimum_bytes
    ]
    if not candidates:
        return None
    # A release can expose several compatible tags. Prefer the CPython 3.12
    # wheel, then abi3, then a pure-Python wheel; size breaks any final tie.
    def rank(item):
        filename = item["filename"].lower()
        tag_rank = 3 if "-cp312-" in filename else 2 if "-abi3-" in filename else 1
        return tag_rank, int(item.get("size") or 0)

    item = max(candidates, key=rank)
    digest = (item.get("digests") or {}).get("sha256")
    if not digest:
        print(f"[setup] warning: PyPI did not publish SHA256 for {spec}", file=sys.stderr)
        return None
    return item["url"], item["filename"], digest, int(item["size"])


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    resolved = list(executor.map(resolve, specs))
for item in sorted((item for item in resolved if item), key=lambda row: row[1]):
    print(*item, sep="\t")
PY
}

prepare_large_wheels() {
    local input_file=$WHEEL_DIR/aria2-input.txt
    local metadata_file=$WHEEL_DIR/aria2-pypi-metadata.tsv
    local accelerated_list=$WHEEL_DIR/aria2-accelerated-wheel-paths.txt
    local torch_wheel=$WHEEL_DIR/$TORCH_WHEEL_NAME
    local vllm_wheel=$WHEEL_DIR/$VLLM_WHEEL_NAME
    local bad_suffix url filename checksum size wheel

    mkdir -p "$WHEEL_DIR"
    bad_suffix=$(date -u +%Y%m%dT%H%M%SZ)
    if [[ -e "$torch_wheel" ]] && ! wheel_is_valid "$torch_wheel" "$TORCH_WHEEL_SHA256"; then
        move_bad_download "$torch_wheel" "$bad_suffix"
    fi
    if [[ -e "$vllm_wheel" ]] && ! wheel_is_valid "$vllm_wheel" "$VLLM_WHEEL_SHA256"; then
        move_bad_download "$vllm_wheel" "$bad_suffix"
    fi

    : >"$input_file"
    : >"$accelerated_list"
    if ! wheel_is_valid "$torch_wheel" "$TORCH_WHEEL_SHA256"; then
        append_aria2_entry \
            "$input_file" "$TORCH_WHEEL_URL" "$TORCH_WHEEL_NAME" "$TORCH_WHEEL_SHA256"
    fi
    if ! wheel_is_valid "$vllm_wheel" "$VLLM_WHEEL_SHA256"; then
        append_aria2_entry \
            "$input_file" "$VLLM_WHEEL_URL" "$VLLM_WHEEL_NAME" "$VLLM_WHEEL_SHA256"
    fi

    echo "[setup] resolving fixed large dependency wheels from official PyPI metadata"
    if ! write_pypi_accelerated_metadata "$metadata_file"; then
        echo "[setup] warning: large dependency metadata resolution failed; uv will download those dependencies" >&2
        : >"$metadata_file"
    fi
    while IFS=$'\t' read -r url filename checksum size; do
        [[ -n "$filename" ]] || continue
        wheel=$WHEEL_DIR/$filename
        if [[ -e "$wheel" ]] && ! wheel_is_valid "$wheel" "$checksum"; then
            move_bad_download "$wheel" "$bad_suffix"
        fi
        if ! wheel_is_valid "$wheel" "$checksum"; then
            append_aria2_entry "$input_file" "$url" "$filename" "$checksum"
        fi
        printf '%s\n' "$wheel" >>"$accelerated_list"
        printf '[setup] aria2 dependency: %s (%.1f MiB)\n' \
            "$filename" "$((size / 1024 / 1024))"
    done <"$metadata_file"

    if [[ -s "$input_file" ]]; then
        echo "[setup] downloading all selected large wheels with aria2"
        aria2c \
            --input-file="$input_file" \
            --continue=true \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --file-allocation=none \
            --max-concurrent-downloads="$ARIA2_CONCURRENT_FILES" \
            --max-connection-per-server="$ARIA2_CONNECTIONS" \
            --split="$ARIA2_CONNECTIONS" \
            --min-split-size=4M \
            --connect-timeout=30 \
            --timeout=120 \
            --retry-wait=2 \
            --max-tries=10
    else
        echo "[setup] verified large wheels already exist under $WHEEL_DIR"
    fi

    wheel_is_valid "$torch_wheel" "$TORCH_WHEEL_SHA256" \
        || die "downloaded Torch wheel is incomplete: $torch_wheel"
    wheel_is_valid "$vllm_wheel" "$VLLM_WHEEL_SHA256" \
        || die "downloaded vLLM wheel is incomplete: $vllm_wheel"
    while IFS=$'\t' read -r url filename checksum size; do
        [[ -n "$filename" ]] || continue
        wheel=$WHEEL_DIR/$filename
        wheel_is_valid "$wheel" "$checksum" \
            || die "aria2 dependency wheel failed validation: $wheel"
    done <"$metadata_file"
}

core_stack_is_ready() {
    [[ -x "$VENV_DIR/bin/python" ]] || return 1
    "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
from packaging.version import Version
import torch

assert Version(version("torch")).base_version == "2.11.0"
assert torch.version.cuda and torch.version.cuda.startswith("12.9")
assert Version(version("vllm")).base_version == "0.22.1"
assert Version(version("transformers")).base_version == "5.12.1"
PY
}

find_cuda_toolkit() {
    local candidate release
    local -a candidates=()
    [[ -n "${CUDA_HOME:-}" ]] && candidates+=("$CUDA_HOME")
    candidates+=(/usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda)
    if command -v nvcc >/dev/null 2>&1; then
        candidates+=("$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)")
    fi
    for candidate in "${candidates[@]}"; do
        [[ -x "$candidate/bin/nvcc" && -f "$candidate/include/cuda.h" ]] || continue
        release=$("$candidate/bin/nvcc" --version 2>/dev/null \
            | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)
        if [[ -n "$release" ]] && version_ge "$release" 12.8; then
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

hpc_runtime_is_ready() {
    local expected_hash=$1
    [[ -f "$VENV_DIR/.hpc-source.sha256" ]] || return 1
    [[ "$(<"$VENV_DIR/.hpc-source.sha256")" == "$expected_hash" ]] || return 1
    "$PYTHON" - <<'PY' >/dev/null 2>&1
import hpc
import torch

schema = str(torch.ops.hpc.attention_decode_bf16.default._schema)
assert "softmax_scale" in schema
assert "Tensor? output" in schema
PY
}

write_runtime_env() {
    local torch_lib torch_cmake_prefix toolkit_lib
    torch_lib=$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
    torch_cmake_prefix=$("$PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')
    if [[ -d "$CUDA_TOOLKIT_ROOT/lib" ]]; then
        toolkit_lib=$CUDA_TOOLKIT_ROOT/lib
    else
        toolkit_lib=$CUDA_TOOLKIT_ROOT/lib64
    fi
    {
        printf 'export H20_CU129_RUNTIME=1\n'
        printf 'export GQLA_ROOT=%q\n' "$GQLA_ROOT"
        printf 'export MODEL_DIR=%q\n' "$MODEL_DIR"
        printf 'export CODE_ROOT=%q\n' "$CODE_ROOT"
        printf 'export GQLA_REPO=%q\n' "$GQLA_REPO"
        printf 'export HPC_OPS_DIR=%q\n' "$HPC_OPS_DIR"
        printf 'export ENV_ROOT=%q\n' "$ENV_ROOT"
        printf 'export VENV_DIR=%q\n' "$VENV_DIR"
        printf 'export PYTHON=%q\n' "$PYTHON"
        printf 'export CUDA_HOME=%q\n' "$CUDA_TOOLKIT_ROOT"
        printf 'export CUDAToolkit_ROOT=%q\n' "$CUDA_TOOLKIT_ROOT"
        printf 'export CUDACXX=%q\n' "$CUDA_TOOLKIT_ROOT/bin/nvcc"
        printf 'export PATH=%q:%q:$PATH\n' "$VENV_DIR/bin" "$CUDA_TOOLKIT_ROOT/bin"
        printf 'export LD_LIBRARY_PATH=%q:%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' \
            "$toolkit_lib" "$torch_lib"
        printf 'export CMAKE_PREFIX_PATH=%q${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}\n' \
            "$torch_cmake_prefix"
        printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$GQLA_REPO"
        printf 'export CMAKE_GENERATOR=Ninja\n'
    } >"$RUNTIME_ENV_FILE"
}

require_bool FORCE_RECREATE "$FORCE_RECREATE"
require_bool FORCE_HPC_REBUILD "$FORCE_HPC_REBUILD"
require_bool DRY_RUN "$DRY_RUN"
require_bool ARIA2_ACCELERATE_DEPS "$ARIA2_ACCELERATE_DEPS"
[[ "$ARIA2_MIN_WHEEL_MIB" =~ ^[1-9][0-9]*$ ]] \
    || die "ARIA2_MIN_WHEEL_MIB must be a positive integer"
[[ "$(uname -m)" == x86_64 ]] || die "the pinned wheel set currently supports x86_64 only"
[[ -f "$DEPLOY_SCRIPT" ]] || die "deployment helper is missing: $DEPLOY_SCRIPT"
[[ -f "$MODEL_DIR/config.json" ]] || die "converted model is missing: $MODEL_DIR/config.json"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] \
    || die "converted model index is missing: $MODEL_DIR/model.safetensors.index.json"

if [[ "$DRY_RUN" == 1 ]]; then
    cat <<EOF
[setup dry-run]
GQLA_ROOT=$GQLA_ROOT
MODEL_DIR=$MODEL_DIR
VENV_DIR=$VENV_DIR
GQLA_REPO=$GQLA_REPO
HPC_OPS_DIR=$HPC_OPS_DIR
TORCH_WHEEL_URL=$TORCH_WHEEL_URL
VLLM_WHEEL_URL=$VLLM_WHEEL_URL
ARIA2_ACCELERATE_DEPS=$ARIA2_ACCELERATE_DEPS
ARIA2_MIN_WHEEL_MIB=$ARIA2_MIN_WHEEL_MIB
RUNTIME_ENV_FILE=$RUNTIME_ENV_FILE
EOF
    exit 0
fi

mkdir -p "$ENV_ROOT/logs" "$UV_CACHE_DIR"
exec > >(tee -a "$SETUP_LOG") 2>&1
echo "[setup] log=$SETUP_LOG"
echo "[setup] model is read-only input; it will not be downloaded or modified: $MODEL_DIR"

if command -v flock >/dev/null 2>&1; then
    exec 9>"$ENV_ROOT/.setup-h20-cu129.lock"
    echo "[setup] waiting for lock: $ENV_ROOT/.setup-h20-cu129.lock"
    flock 9
fi

ensure_aria2
sanitize_cuda13_compat

echo "[setup] materializing GQLA and the patched HPC-Ops source from the GitHub relay bundle"
GQLA_ROOT="$GQLA_ROOT" \
CODE_ROOT="$CODE_ROOT" \
ENV_ROOT="$ENV_ROOT" \
MODEL_DIR="$MODEL_DIR" \
GQLA_DST="$GQLA_REPO" \
HPC_DST="$HPC_OPS_DIR" \
RUN_INSTALL=0 \
bash "$DEPLOY_SCRIPT"

UV_BIN=$GQLA_REPO/tools/uv
BASE_PYTHON=$ENV_ROOT/python/cpython-$PYTHON_VERSION-linux-x86_64-gnu/bin/python3.12
[[ -x "$UV_BIN" ]] || die "bundled uv was not deployed: $UV_BIN"
[[ -x "$BASE_PYTHON" ]] || die "bundled Python was not deployed: $BASE_PYTHON"

if [[ "$FORCE_RECREATE" == 1 && -e "$VENV_DIR" ]]; then
    backup=$VENV_DIR.previous.$(date -u +%Y%m%dT%H%M%SZ)
    mv "$VENV_DIR" "$backup"
    echo "[setup] moved previous environment to $backup"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[setup] creating fresh Python environment: $VENV_DIR"
    "$UV_BIN" venv --no-project --no-config --allow-existing \
        --python "$BASE_PYTHON" "$VENV_DIR"
fi
PYTHON=$VENV_DIR/bin/python

export UV_CACHE_DIR
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10}
export UV_CONCURRENT_DOWNLOADS UV_CONCURRENT_INSTALLS
export UV_TORCH_BACKEND=cu129

if ! core_stack_is_ready; then
    prepare_large_wheels
    mapfile -t accelerated_wheels \
        <"$WHEEL_DIR/aria2-accelerated-wheel-paths.txt"
    echo "[setup] installing cu129 vLLM and the remaining dependency closure with uv"
    "$UV_BIN" pip install --no-config --python "$PYTHON" --compile-bytecode \
        --extra-index-url "$PYTORCH_INDEX_URL" \
        "$WHEEL_DIR/$TORCH_WHEEL_NAME" \
        "$WHEEL_DIR/$VLLM_WHEEL_NAME" \
        "${accelerated_wheels[@]}" \
        "transformers==$TRANSFORMERS_VERSION" \
        'setuptools==80.10.2' 'wheel>=0.45.1' 'cmake>=3.26,<4' 'ninja>=1.12'
else
    echo "[setup] clean cu129 Torch/vLLM environment is already installed"
fi
core_stack_is_ready || die "installed core stack does not match torch 2.11/cu129 + vLLM 0.22.1"

"$UV_BIN" pip install --no-config --python "$PYTHON" --no-deps \
    --no-build-isolation --editable "$GQLA_REPO"

CUDA_TOOLKIT_ROOT=
CUDA_TOOLKIT_VERSION=
find_cuda_toolkit \
    || die "HPC-Ops requires a local CUDA toolkit with nvcc >=12.8 (expected /usr/local/cuda-12.8)"
echo "[setup] CUDA toolkit=$CUDA_TOOLKIT_ROOT version=$CUDA_TOOLKIT_VERSION"

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
if [[ "$FORCE_HPC_REBUILD" == 1 ]] || ! hpc_runtime_is_ready "$source_hash"; then
    build_root=$ENV_ROOT/build/hpc-cu129-$(date -u +%Y%m%dT%H%M%SZ)-$$
    wheelhouse=$build_root/wheelhouse
    mkdir -p "$wheelhouse"
    echo "[setup] building patched HPC-Ops for torch=$TORCH_VERSION/cu129"
    echo "[setup] source_hash=$source_hash build_root=$build_root"
    if ! (
        cd "$HPC_OPS_DIR"
        export HPC_GIT_HASH_OVERRIDE=${source_hash:0:7}
        "$PYTHON" setup.py \
            build --build-base "$build_root/build" \
            bdist_wheel --dist-dir "$wheelhouse" --bdist-dir "$build_root/bdist"
    ) 2>&1 | tee "$build_root/build.log"; then
        die "HPC-Ops build failed; inspect $build_root/build.log"
    fi
    mapfile -t hpc_wheels < <(find "$wheelhouse" -maxdepth 1 -type f -name 'hpc_ops-*.whl' -print)
    (( ${#hpc_wheels[@]} == 1 )) \
        || die "expected one HPC-Ops wheel, found ${#hpc_wheels[@]}"
    "$UV_BIN" pip install --no-config --python "$PYTHON" --no-deps \
        --reinstall "${hpc_wheels[0]}"
    printf '%s\n' "$source_hash" >"$VENV_DIR/.hpc-source.sha256"
else
    echo "[setup] matching patched HPC-Ops wheel is already installed"
fi
hpc_runtime_is_ready "$source_hash" \
    || die "HPC-Ops import/schema validation failed after installation"

write_runtime_env
# shellcheck disable=SC1090
source "$RUNTIME_ENV_FILE"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
"$PYTHON" - "$MODEL_DIR" <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from packaging.version import Version

model_dir = Path(sys.argv[1])
with (model_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
    index = json.load(handle)
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"converted checkpoint has missing shards: {missing[:8]}")

assert Version(version("torch")).base_version == "2.11.0"
assert torch.version.cuda and torch.version.cuda.startswith("12.9")
assert Version(version("vllm")).base_version == "0.22.1"
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
print(
    "H20_CU129_SETUP_OK",
    f"torch={torch.__version__}",
    f"torch_cuda={torch.version.cuda}",
    f"vllm={version('vllm')}",
    f"hpc={hpc.__version__}",
    f"gpus={torch.cuda.device_count()}",
    f"capabilities={capabilities}",
    f"checkpoint_shards={len(shards)}",
)
PY

cat >"$VENV_DIR/h20-cu129-manifest.env" <<EOF
setup_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
model_dir=$MODEL_DIR
gqla_repo=$GQLA_REPO
hpc_ops_dir=$HPC_OPS_DIR
venv_dir=$VENV_DIR
runtime_env=$RUNTIME_ENV_FILE
torch_version=$TORCH_VERSION+cu129
vllm_version=$VLLM_VERSION+cu129
cuda_toolkit=$CUDA_TOOLKIT_ROOT
hpc_source_hash=$source_hash
EOF

echo "H20_CU129_ENVIRONMENT_READY runtime_env=$RUNTIME_ENV_FILE"
