#!/usr/bin/env bash
# Run exactly one H20 TP=8 benchmark case from a verified split-K runtime.
# Usage: bash run_h20_tp8_benchmark.sh {mla|gqla} {2k|8k|16k|16k-b20}

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
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

die() {
    echo "error: $*" >&2
    exit 2
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

validate_gqla_splitk_runtime() {
    local expected_commit=de202c9bda942fdfd499d09e51ea6ff9c89c5d50
    local current_commit current_hash installed_hash backend

    current_commit=$(git -C "$HPC_OPS_DIR" rev-parse HEAD 2>/dev/null || true)
    if [[ -z "$current_commit" && -f "$HPC_OPS_DIR/.gqla_hpc_source_commit" ]]; then
        current_commit=$(<"$HPC_OPS_DIR/.gqla_hpc_source_commit")
    fi
    [[ "$current_commit" == "$expected_commit" ]] \
        || die "HPC-Ops source is $current_commit; expected adaptive split-K $expected_commit"

    current_hash=$(hpc_source_hash)
    [[ -f "$VENV_DIR/.hpc-source.sha256" ]] \
        || die "installed HPC-Ops source fingerprint is missing: $VENV_DIR/.hpc-source.sha256"
    installed_hash=$(<"$VENV_DIR/.hpc-source.sha256")
    [[ "$installed_hash" == "$current_hash" ]] \
        || die "installed HPC-Ops hash $installed_hash does not match source $current_hash; rerun setup with FORCE_HPC_REBUILD=1"

    backend=$GQLA_REPO/src/vllm_hpc_gqa_backend.py
    [[ -f "$backend" ]] && grep -Fq 'splitk=True' "$backend" \
        || die "vLLM GQLA backend does not enable adaptive split-K: $backend"
    "$PYTHON" - <<'PY'
import hpc
import torch

schema = str(torch.ops.hpc.attention_decode_bf16.default._schema)
assert "use_splitk" in schema, schema
assert "softmax_scale" in schema, schema
print(f"H20_GQLA_SPLITK_PREFLIGHT_OK hpc={hpc.__version__} schema={schema}")
PY
}

GQLA_ROOT=${GQLA_ROOT:-$(infer_gqla_root)}
if [[ -z "${VENV_DIR:-}" ]]; then
    for candidate in \
        "$GQLA_ROOT/envs/venv-py312" \
        "$GQLA_ROOT/envs/h20-new" \
        "$GQLA_ROOT/envs/h20-cu129-py312"; do
        if [[ -f "$candidate/h20-splitk-runtime.env" ]]; then
            VENV_DIR=$candidate
            break
        fi
    done
fi
VENV_DIR=${VENV_DIR:-$GQLA_ROOT/envs/h20-cu129-py312}
if [[ -z "${RUNTIME_ENV_FILE:-}" ]]; then
    if [[ -f "$VENV_DIR/h20-splitk-runtime.env" ]]; then
        RUNTIME_ENV_FILE=$VENV_DIR/h20-splitk-runtime.env
    else
        RUNTIME_ENV_FILE=$VENV_DIR/h20-runtime.env
    fi
fi
ROUTE=${1:-${ROUTE:-}}
PROFILE=${2:-${PROFILE:-}}
(( $# <= 2 )) || die "usage: $0 {mla|gqla} {2k|8k|16k|16k-b20}"
[[ -n "$ROUTE" && -n "$PROFILE" ]] \
    || die "usage: $0 {mla|gqla} {2k|8k|16k|16k-b20}"

sanitize_cuda13_compat
[[ -f "$RUNTIME_ENV_FILE" ]] \
    || die "verified H20 runtime is not ready; first run update_h20_splitk_only.sh"
# shellcheck disable=SC1090
source "$RUNTIME_ENV_FILE"
[[ "${H20_CU129_RUNTIME:-0}" == 1 || "${H20_GQLA_EXISTING_RUNTIME:-0}" == 1 ]] \
    || die "runtime manifest is not a verified H20 split-K environment: $RUNTIME_ENV_FILE"

BASE_SCRIPT=${BASE_SCRIPT:-$GQLA_REPO/scripts/benchmark_dsv3p1_g8_h20_tp8.sh}
[[ -f "$BASE_SCRIPT" ]] || die "benchmark backend is missing: $BASE_SCRIPT"
[[ -f "$MODEL_DIR/config.json" ]] || die "converted model is missing: $MODEL_DIR"

case "$ROUTE" in
    mla|mqa-mla)
        route_name=mla
        path_name=mqa-mla
        ;;
    gqla|gqa-hpc)
        route_name=gqla
        path_name=gqa-hpc
        ;;
    *) die "route must be mla or gqla; got $ROUTE" ;;
esac

if [[ "$path_name" == gqa-hpc ]]; then
    validate_gqla_splitk_runtime
fi

case "$PROFILE" in
    2k)
        input_len=2048
        max_model_len=4096
        max_num_batched_tokens=8192
        profile_prompts=${NUM_PROMPTS:-1024}
        profile_warmups=${NUM_WARMUPS:-64}
        profile_concurrency=${MAX_CONCURRENCY:-64}
        ;;
    8k)
        input_len=8192
        max_model_len=9216
        max_num_batched_tokens=16384
        profile_prompts=${NUM_PROMPTS:-1024}
        profile_warmups=${NUM_WARMUPS:-64}
        profile_concurrency=${MAX_CONCURRENCY:-64}
        ;;
    16k)
        input_len=16384
        max_model_len=17408
        max_num_batched_tokens=32768
        profile_prompts=${NUM_PROMPTS:-1024}
        profile_warmups=${NUM_WARMUPS:-64}
        profile_concurrency=${MAX_CONCURRENCY:-64}
        ;;
    16k-b20)
        input_len=16384
        max_model_len=17408
        max_num_batched_tokens=32768
        profile_prompts=${NUM_PROMPTS:-20}
        profile_warmups=${NUM_WARMUPS:-20}
        profile_concurrency=${MAX_CONCURRENCY:-20}
        ;;
    *) die "profile must be 2k, 8k, 16k, or 16k-b20; got $PROFILE" ;;
esac

OUTPUT_LEN=${OUTPUT_LEN:-128}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$profile_concurrency}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
OUTPUT_ROOT=${OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_h20_tp8_split/$PROFILE/$route_name}
RUN_ID_BASE=${RUN_ID_BASE:-h20-tp8-$route_name-$PROFILE-$(date -u +%Y%m%dT%H%M%SZ)}
ATTEMPT=${ATTEMPT:-auto}
DRY_RUN=${DRY_RUN:-0}

echo "[bench] route=$route_name profile=$PROFILE TP=8"
echo "[bench] input=$input_len output=$OUTPUT_LEN concurrency=$profile_concurrency prompts=$profile_prompts warmups=$profile_warmups"
echo "[bench] model=$MODEL_DIR"
echo "[bench] output_root=$OUTPUT_ROOT run_id_base=$RUN_ID_BASE"

exec env BASH_ENV=/dev/null ENV=/dev/null \
    GQLA_ROOT="$GQLA_ROOT" \
    MODEL_DIR="$MODEL_DIR" \
    ENV_ROOT="$ENV_ROOT" \
    VENV_DIR="$VENV_DIR" \
    PYTHON="$PYTHON" \
    HPC_OPS_DIR="$HPC_OPS_DIR" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    BOOTSTRAP_ENV=0 \
    INSTALL_ONLY=0 \
    PATHS="$path_name" \
    INPUT_LEN="$input_len" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    MAX_MODEL_LEN="$max_model_len" \
    MAX_NUM_BATCHED_TOKENS="$max_num_batched_tokens" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    NUM_PROMPTS="$profile_prompts" \
    NUM_WARMUPS="$profile_warmups" \
    MAX_CONCURRENCY="$profile_concurrency" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    CPU_OFFLOAD_GB=0 \
    ENFORCE_EAGER=0 \
    ENABLE_CHUNKED_PREFILL=1 \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RUN_ID_BASE="$RUN_ID_BASE" \
    ATTEMPT="$ATTEMPT" \
    DRY_RUN="$DRY_RUN" \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}" \
    bash "$BASE_SCRIPT"
