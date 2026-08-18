#!/usr/bin/env bash
# Update the GQLA/vLLM plugin and rebuild adaptive split-K HPC-Ops inside an
# already complete H20 serving environment. This path never downloads core
# Torch, vLLM, CUDA, FlashInfer, Python, or uv artifacts.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SETUP_SCRIPT=$SCRIPT_DIR/setup_h20_cu129_env.sh
MODEL_NAME=dsv3p1_g8_sim_hess_no_mean_subtract

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

core_stack_is_ready() {
    local python=$1
    [[ -x "$python" ]] || return 1
    env BASH_ENV=/dev/null ENV=/dev/null "$python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
import torch

base = lambda value: value.split("+", 1)[0]
assert base(version("torch")) == "2.11.0"
assert torch.version.cuda and torch.version.cuda.startswith("12.9")
assert base(version("vllm")) == "0.22.1"
assert base(version("transformers")) == "5.12.1"
PY
}

report_stack() {
    local python=$1
    env BASH_ENV=/dev/null ENV=/dev/null "$python" - <<'PY' 2>&1 || true
from importlib.metadata import PackageNotFoundError, version

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
)
PY
}

GQLA_ROOT=${GQLA_ROOT:-$(infer_gqla_root)}
ENV_ROOT=${ENV_ROOT:-$GQLA_ROOT/envs}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/$MODEL_NAME}
HPC_OPS_DIR=${HPC_OPS_DIR:-$GQLA_ROOT/code/hpc-ops-de202c9}
requested_venv=${VENV_DIR:-}

[[ -f "$SETUP_SCRIPT" ]] || die "setup helper is missing: $SETUP_SCRIPT"
[[ -f "$MODEL_DIR/config.json" ]] || die "converted model is missing: $MODEL_DIR/config.json"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] \
    || die "converted model index is missing: $MODEL_DIR/model.safetensors.index.json"

if [[ -n "$requested_venv" ]]; then
    selected_venv=$requested_venv
    python=$selected_venv/bin/python
    if ! core_stack_is_ready "$python"; then
        echo "[hpc-only] rejected explicit VENV_DIR=$selected_venv" >&2
        [[ -x "$python" ]] && report_stack "$python" >&2
        die "HPC-only update requires torch 2.11/cu129, vLLM 0.22.1, and transformers 5.12.1"
    fi
else
    selected_venv=
    preferred=$ENV_ROOT/h20-cu129-py312
    if core_stack_is_ready "$preferred/bin/python"; then
        selected_venv=$preferred
    else
        while IFS= read -r -d '' python; do
            candidate=${python%/bin/python}
            if core_stack_is_ready "$python"; then
                selected_venv=$candidate
                break
            fi
        done < <(find "$ENV_ROOT" -mindepth 2 -maxdepth 3 \
            -path '*/bin/python' -print0 2>/dev/null | sort -z)
    fi

    if [[ -z "$selected_venv" ]]; then
        echo "[hpc-only] no compatible serving environment found under $ENV_ROOT" >&2
        while IFS= read -r -d '' python; do
            [[ -x "$python" ]] || continue
            echo "[hpc-only] python=$python" >&2
            report_stack "$python" >&2
        done < <(find "$ENV_ROOT" -mindepth 2 -maxdepth 3 \
            -path '*/bin/python' -print0 2>/dev/null | sort -z)
        die "refusing all core-wheel downloads; provide a matching VENV_DIR or relay the serving environment through OSS"
    fi
fi

echo "[hpc-only] selected VENV_DIR=$selected_venv"
report_stack "$selected_venv/bin/python"
echo "[hpc-only] core downloads are disabled; only local source deployment and HPC-Ops rebuild may run"

exec env BASH_ENV=/dev/null ENV=/dev/null \
    GQLA_ROOT="$GQLA_ROOT" \
    ENV_ROOT="$ENV_ROOT" \
    MODEL_DIR="$MODEL_DIR" \
    HPC_OPS_DIR="$HPC_OPS_DIR" \
    VENV_DIR="$selected_venv" \
    RUNTIME_ENV_FILE="$selected_venv/h20-runtime.env" \
    ALLOW_CORE_DOWNLOADS=0 \
    FORCE_RECREATE=0 \
    FORCE_HPC_REBUILD=1 \
    PIP_NO_INDEX=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    bash "$SETUP_SCRIPT"
