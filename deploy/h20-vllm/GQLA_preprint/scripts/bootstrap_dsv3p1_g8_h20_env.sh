#!/usr/bin/env bash
# Bootstrap and validate the self-contained H20 benchmark environment.
#
# The model is never downloaded. Only Python, binary wheels, and build tools
# are fetched. uv is used instead of pip so large CUDA wheels are downloaded
# concurrently and retained in a persistent cache beside the environment.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
DEFAULT_GQLA_ROOT=/mnt/tidalfs-alwl01/task/236362/GQLA
DEFAULT_MODEL_DIR=$DEFAULT_GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract

MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
if [[ -z "${GQLA_ROOT:-}" ]]; then
    case "$MODEL_DIR" in
        */outputs/convert/*) GQLA_ROOT=${MODEL_DIR%%/outputs/convert/*} ;;
        *) GQLA_ROOT=$DEFAULT_GQLA_ROOT ;;
    esac
fi
ENV_ROOT=${ENV_ROOT:-$GQLA_ROOT/envs}
VENV_DIR=${VENV_DIR:-$ENV_ROOT/venv-py312}
UV_CACHE_DIR=${UV_CACHE_DIR:-$GQLA_ROOT/.cache/uv}
UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$ENV_ROOT/python}
UV_VERSION=${UV_VERSION:-0.9.3}
PYTHON_VERSION=${PYTHON_VERSION:-3.12.12}
REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-$REPO_DIR/requirements-h20-vllm.txt}
if [[ -z "${HPC_OPS_DIR:-}" ]]; then
    if [[ -d "$REPO_DIR/../hpc-ops" ]]; then
        HPC_OPS_DIR=$(cd "$REPO_DIR/../hpc-ops" && pwd -P)
    else
        HPC_OPS_DIR=$GQLA_ROOT/code/hpc-ops
    fi
fi
RUNTIME_ENV_FILE=${RUNTIME_ENV_FILE:-$VENV_DIR/h20-runtime.env}
BOOTSTRAP_LOG=${BOOTSTRAP_LOG:-$ENV_ROOT/logs/bootstrap-h20.log}
NEEDS_HPC=${NEEDS_HPC:-1}
FORCE_CORE_REINSTALL=${FORCE_CORE_REINSTALL:-0}
FORCE_HPC_REBUILD=${FORCE_HPC_REBUILD:-0}
MIN_FREE_GIB_WARN=${MIN_FREE_GIB_WARN:-60}

die() {
    echo "error: $*" >&2
    exit 2
}

require_bool() {
    local name=$1
    local value=$2
    case "$value" in
        0|1) ;;
        *) die "$name must be 0 or 1; got $value" ;;
    esac
}

version_ge() {
    local actual=$1
    local minimum=$2
    [[ "$(printf '%s\n' "$minimum" "$actual" | sort -V | head -n1)" == "$minimum" ]]
}

download_uv() {
    local machine target archive partial extract_dir candidate url
    machine=$(uname -m)
    case "$machine" in
        x86_64) target=x86_64-unknown-linux-gnu ;;
        aarch64|arm64) target=aarch64-unknown-linux-gnu ;;
        *) die "unsupported CPU architecture for uv bootstrap: $machine" ;;
    esac

    mkdir -p "$ENV_ROOT/tools" "$ENV_ROOT/downloads"
    archive=$ENV_ROOT/downloads/uv-${UV_VERSION}-${target}.tar.gz
    partial=$archive.part
    url=https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${target}.tar.gz
    echo "[bootstrap] downloading uv $UV_VERSION from $url"
    curl --fail --location --retry 8 --retry-delay 2 --connect-timeout 30 \
        --continue-at - --output "$partial" "$url"
    mv -f -- "$partial" "$archive"

    extract_dir=$(mktemp -d "$ENV_ROOT/tools/.uv-extract.XXXXXX")
    tar -xzf "$archive" -C "$extract_dir"
    candidate=$(find "$extract_dir" -type f -name uv -print -quit)
    [[ -n "$candidate" ]] || die "uv archive did not contain an executable"
    install -m 0755 "$candidate" "$ENV_ROOT/tools/uv"
    rm -rf -- "$extract_dir"
    UV_BIN=$ENV_ROOT/tools/uv
}

core_stack_is_ready() {
    "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

expected = {
    "torch": "2.11.0",
    "vllm": "0.22.1",
    "transformers": "5.12.1",
}
assert all(version(name) == wanted for name, wanted in expected.items())
PY
}

build_tools_are_ready() {
    [[ -x "$VENV_DIR/bin/cmake" && -x "$VENV_DIR/bin/ninja" ]] || return 1
    "$VENV_DIR/bin/cmake" --version >/dev/null 2>&1 || return 1
    "$VENV_DIR/bin/ninja" --version >/dev/null 2>&1 || return 1
}

find_cuda_toolkit() {
    local candidate release
    local -a candidates=()

    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("$CUDA_HOME")
    fi
    if command -v nvcc >/dev/null 2>&1; then
        candidates+=("$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)")
    fi
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
        [[ -f "$candidate/bin/nvcc" && -f "$candidate/include/cuda.h" ]] || continue
        if [[ "$candidate" == "$VENV_DIR/"* ]]; then
            # Some network filesystems lose executable bits while unpacking
            # CUDA toolkit wheels. Restore only the known toolkit bin trees.
            find "$candidate/bin" "$candidate/nvvm/bin" -maxdepth 1 -type f \
                -exec chmod u+x {} + 2>/dev/null || true
            if [[ ! -e "$candidate/lib/libcudart.so" \
                && -f "$candidate/lib/libcudart.so.13" ]]; then
                ln -s libcudart.so.13 "$candidate/lib/libcudart.so"
            fi
        fi
        if [[ ! -x "$candidate/bin/nvcc" ]]; then
            chmod u+x "$candidate/bin/nvcc" 2>/dev/null || continue
        fi
        release=$("$candidate/bin/nvcc" --version 2>/dev/null \
            | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)
        if [[ -n "$release" ]] && version_ge "$release" 12.8; then
            CUDA_TOOLKIT_ROOT=$candidate
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
    local torch_lib torch_cmake_prefix
    torch_lib=$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
    torch_cmake_prefix=$("$PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')
    mkdir -p "$(dirname "$RUNTIME_ENV_FILE")"
    {
        printf 'export GQLA_ROOT=%q\n' "$GQLA_ROOT"
        printf 'export MODEL_DIR=%q\n' "$MODEL_DIR"
        printf 'export ENV_ROOT=%q\n' "$ENV_ROOT"
        printf 'export VENV_DIR=%q\n' "$VENV_DIR"
        printf 'export PYTHON=%q\n' "$PYTHON"
        printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
        printf 'export PATH=%q:$PATH\n' "$VENV_DIR/bin"
        if [[ -n "${CUDA_TOOLKIT_ROOT:-}" ]]; then
            printf 'export CUDA_HOME=%q\n' "$CUDA_TOOLKIT_ROOT"
            printf 'export CUDAToolkit_ROOT=%q\n' "$CUDA_TOOLKIT_ROOT"
            printf 'export CUDACXX=%q\n' "$CUDA_TOOLKIT_ROOT/bin/nvcc"
            printf 'export PATH=%q:$PATH\n' "$CUDA_TOOLKIT_ROOT/bin"
            printf 'export LD_LIBRARY_PATH=%q:%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' \
                "$CUDA_TOOLKIT_ROOT/lib" "$torch_lib"
        fi
        printf 'export CMAKE_PREFIX_PATH=%q${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}\n' \
            "$torch_cmake_prefix"
        printf 'export CMAKE_GENERATOR=Ninja\n'
    } >"$RUNTIME_ENV_FILE"
}

require_bool NEEDS_HPC "$NEEDS_HPC"
require_bool FORCE_CORE_REINSTALL "$FORCE_CORE_REINSTALL"
require_bool FORCE_HPC_REBUILD "$FORCE_HPC_REBUILD"

for command_name in bash curl tar sha256sum sort awk sed find xargs install; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "required host command is missing: $command_name"
done
[[ -f "$REQUIREMENTS_FILE" ]] || die "requirements file is missing: $REQUIREMENTS_FILE"
[[ -f "$REPO_DIR/pyproject.toml" ]] || die "GQLA source tree is incomplete: $REPO_DIR"

MODEL_DIR=$(readlink -f -- "$MODEL_DIR") \
    || die "converted model directory does not exist: $MODEL_DIR"
for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
    [[ -f "$MODEL_DIR/$required" ]] \
        || die "missing converted-model file: $MODEL_DIR/$required"
done

mkdir -p "$ENV_ROOT/logs" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
touch "$BOOTSTRAP_LOG" || die "environment root is not writable: $ENV_ROOT"
echo "[bootstrap] log=$BOOTSTRAP_LOG"
echo "[bootstrap] model already present; no model download will be attempted: $MODEL_DIR"

available_kib=$(df -Pk "$ENV_ROOT" | awk 'NR==2 {print $4}')
available_gib=$((available_kib / 1024 / 1024))
if (( available_gib < MIN_FREE_GIB_WARN )); then
    echo "warning: only ${available_gib} GiB free under $ENV_ROOT; " \
        "the vLLM/CUDA environment, uv cache, and HPC build can require tens of GiB" >&2
fi

if command -v flock >/dev/null 2>&1; then
    exec 9>"$ENV_ROOT/.bootstrap.lock"
    echo "[bootstrap] waiting for environment lock: $ENV_ROOT/.bootstrap.lock"
    flock 9
fi

if [[ -n "${UV_BIN:-}" ]]; then
    [[ -x "$UV_BIN" ]] || die "UV_BIN is not executable: $UV_BIN"
elif [[ -x "$ENV_ROOT/tools/uv" ]]; then
    UV_BIN=$ENV_ROOT/tools/uv
elif [[ -x "$REPO_DIR/tools/uv" ]]; then
    UV_BIN=$REPO_DIR/tools/uv
elif command -v uv >/dev/null 2>&1; then
    UV_BIN=$(command -v uv)
else
    download_uv
fi
uv_detected_version=$("$UV_BIN" --version | awk '{print $2}')
if [[ -z "$uv_detected_version" ]] || ! version_ge "$uv_detected_version" 0.9.0; then
    echo "[bootstrap] uv ${uv_detected_version:-unknown} is too old; installing pinned uv $UV_VERSION"
    download_uv
fi
echo "[bootstrap] uv=$UV_BIN ($("$UV_BIN" --version))"

export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-16}
export UV_CONCURRENT_INSTALLS=${UV_CONCURRENT_INSTALLS:-16}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[bootstrap] creating Python $PYTHON_VERSION environment at $VENV_DIR"
    "$UV_BIN" venv --no-project --no-config --managed-python \
        --allow-existing --python "$PYTHON_VERSION" "$VENV_DIR"
fi
PYTHON=$VENV_DIR/bin/python

if [[ "$FORCE_CORE_REINSTALL" == 1 ]] || ! core_stack_is_ready; then
    echo "[bootstrap] installing the pinned vLLM stack with uv; wheel cache=$UV_CACHE_DIR"
    install_args=(
        pip install --no-config --python "$PYTHON"
        --requirements "$REQUIREMENTS_FILE" --strict --compile-bytecode
    )
    if [[ "$FORCE_CORE_REINSTALL" == 1 ]]; then
        install_args+=(--reinstall)
    fi
    "$UV_BIN" "${install_args[@]}"
else
    echo "[bootstrap] pinned torch/vLLM/transformers stack already present"
fi

if ! build_tools_are_ready; then
    echo "[bootstrap] installing/repairing CMake, Ninja, setuptools, and wheel"
    "$UV_BIN" pip install --no-config --python "$PYTHON" --reinstall \
        'cmake>=3.26,<4' 'ninja>=1.12' 'setuptools==80.10.2' 'wheel>=0.45.1'
    build_tools_are_ready || die "CMake/Ninja installation did not create working executables"
fi

export PATH="$VENV_DIR/bin:$PATH"
"$UV_BIN" pip install --no-config --python "$PYTHON" --no-deps \
    --no-build-isolation --editable "$REPO_DIR"

CUDA_TOOLKIT_ROOT=
CUDA_TOOLKIT_VERSION=
HPC_BUILD_ROOT_CREATED=
if [[ "$NEEDS_HPC" == 1 ]]; then
    [[ -d "$HPC_OPS_DIR" ]] || die "HPC-Ops source directory is missing: $HPC_OPS_DIR"
    [[ -f "$HPC_OPS_DIR/setup.py" && -f "$HPC_OPS_DIR/CMakeLists.txt" ]] \
        || die "HPC-Ops source tree is incomplete: $HPC_OPS_DIR"
    command -v g++ >/dev/null 2>&1 || die "g++ is required to compile HPC-Ops"

    if ! find_cuda_toolkit; then
        die "HPC-Ops needs a CUDA toolkit with nvcc >= 12.8; neither the host nor installed CUDA wheels provide one"
    fi
    echo "[bootstrap] CUDA toolkit=$CUDA_TOOLKIT_ROOT version=$CUDA_TOOLKIT_VERSION"
    export CUDA_HOME=$CUDA_TOOLKIT_ROOT
    export CUDAToolkit_ROOT=$CUDA_TOOLKIT_ROOT
    export CUDACXX=$CUDA_TOOLKIT_ROOT/bin/nvcc
    export PATH="$CUDA_TOOLKIT_ROOT/bin:$VENV_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib:$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CMAKE_PREFIX_PATH="$("$PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export CMAKE_GENERATOR=Ninja

    source_hash=$(hpc_source_hash)
    if [[ "$FORCE_HPC_REBUILD" == 1 ]] || ! hpc_runtime_is_ready "$source_hash"; then
        build_root=$ENV_ROOT/build/hpc-$(date -u +%Y%m%dT%H%M%SZ)-$$
        wheelhouse=$build_root/wheelhouse
        mkdir -p "$wheelhouse"
        echo "[bootstrap] building HPC-Ops for this exact torch/CUDA ABI"
        echo "[bootstrap] source_hash=$source_hash build_root=$build_root"
        if ! (
            cd "$HPC_OPS_DIR"
            export HPC_GIT_HASH_OVERRIDE=${source_hash:0:7}
            "$PYTHON" setup.py \
                build --build-base "$build_root/build" \
                bdist_wheel --dist-dir "$wheelhouse" --bdist-dir "$build_root/bdist"
        ) 2>&1 | tee "$build_root/build.log"; then
            die "HPC-Ops build failed; retained log: $build_root/build.log"
        fi
        mapfile -t hpc_wheels < <(find "$wheelhouse" -maxdepth 1 -type f -name 'hpc_ops-*.whl' -print)
        (( ${#hpc_wheels[@]} == 1 )) \
            || die "expected exactly one HPC-Ops wheel in $wheelhouse, found ${#hpc_wheels[@]}"
        "$UV_BIN" pip install --no-config --python "$PYTHON" --no-deps \
            --reinstall "${hpc_wheels[0]}"
        printf '%s\n' "$source_hash" >"$VENV_DIR/.hpc-source.sha256"
        HPC_BUILD_ROOT_CREATED=$build_root
    else
        echo "[bootstrap] matching HPC-Ops build already installed"
    fi
fi

write_runtime_env
# shellcheck disable=SC1090
source "$RUNTIME_ENV_FILE"

"$PYTHON" - "$MODEL_DIR" "$NEEDS_HPC" <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch

model_dir = Path(sys.argv[1])
needs_hpc = bool(int(sys.argv[2]))
with (model_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
    index = json.load(handle)
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (model_dir / name).is_file()]
if missing:
    preview = ", ".join(missing[:8])
    raise SystemExit(
        f"converted checkpoint has {len(missing)} missing/broken shard(s): {preview}"
    )

if version("torch") != "2.11.0" or version("vllm") != "0.22.1":
    raise SystemExit("installed torch/vLLM versions do not match the benchmark lock")
if not torch.cuda.is_available():
    raise SystemExit("torch cannot initialize CUDA; check the H20 driver/container runtime")
if torch.cuda.device_count() < 8:
    raise SystemExit(f"TP8 requires 8 visible GPUs; torch sees {torch.cuda.device_count()}")
capabilities = [torch.cuda.get_device_capability(i) for i in range(8)]
if needs_hpc and any(cap != (9, 0) for cap in capabilities):
    raise SystemExit(f"HPC GQLA requires SM90 GPUs; capabilities={capabilities}")
for device in range(8):
    torch.empty(1, device=f"cuda:{device}")
torch.cuda.synchronize()

import src.vllm_register_dsv  # noqa: E402,F401
from vllm import ModelRegistry  # noqa: E402

required_arches = {
    "DeepseekV3GQLAForCausalLM",
    "DeepseekV3GQLAHPCForCausalLM",
}
missing_arches = required_arches - set(ModelRegistry.get_supported_archs())
if missing_arches:
    raise SystemExit(f"GQLA plugin registration is missing: {sorted(missing_arches)}")
if needs_hpc:
    import hpc  # noqa: E402

    schema = str(torch.ops.hpc.attention_decode_bf16.default._schema)
    if "softmax_scale" not in schema:
        raise SystemExit("HPC-Ops schema lacks runtime softmax_scale")
    print(f"HPC-Ops {hpc.__version__}: {schema}")

print(
    "H20_ENVIRONMENT_OK",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"vllm={version('vllm')}",
    f"gpus={torch.cuda.device_count()}",
    f"capabilities={capabilities}",
    f"checkpoint_shards={len(shards)}",
)
PY

if [[ -n "$HPC_BUILD_ROOT_CREATED" ]]; then
    case "$HPC_BUILD_ROOT_CREATED" in
        "$ENV_ROOT"/build/hpc-*)
            rm -rf -- "$HPC_BUILD_ROOT_CREATED/build" "$HPC_BUILD_ROOT_CREATED/bdist"
            echo "[bootstrap] removed successful temporary HPC objects; retained wheel and build.log under $HPC_BUILD_ROOT_CREATED"
            ;;
        *) die "refusing to clean unexpected HPC build path: $HPC_BUILD_ROOT_CREATED" ;;
    esac
fi

{
    printf 'validated_utc=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'model_dir=%q\n' "$MODEL_DIR"
    printf 'repo_dir=%q\n' "$REPO_DIR"
    printf 'hpc_ops_dir=%q\n' "$HPC_OPS_DIR"
    printf 'python=%q\n' "$PYTHON"
    printf 'uv=%q\n' "$UV_BIN"
    printf 'uv_cache_dir=%q\n' "$UV_CACHE_DIR"
    printf 'needs_hpc=%q\n' "$NEEDS_HPC"
    printf 'cuda_toolkit_root=%q\n' "${CUDA_TOOLKIT_ROOT:-not-required}"
    printf 'cuda_toolkit_version=%q\n' "${CUDA_TOOLKIT_VERSION:-not-required}"
} >"$VENV_DIR/h20-bootstrap-manifest.env"

echo "H20_BOOTSTRAP_OK python=$PYTHON runtime_env=$RUNTIME_ENV_FILE"
