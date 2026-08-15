#!/usr/bin/env bash
# One-command, single-node H20 TP=8 benchmark matrix for the converted
# DeepSeek-V3.1 G=8 checkpoint. Each profile/route uses a fresh vLLM server.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GQLA_ROOT=${GQLA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract}
BASE_SCRIPT=$SCRIPT_DIR/benchmark_dsv3p1_g8_h20_tp8.sh

# Saturated service profiles. Add 16k-b20 to isolate equal-concurrency
# behavior: PROFILES=2k,8k,16k,16k-b20.
PROFILES=${PROFILES:-2k,8k,16k}
PATHS=${PATHS:-mqa-mla,gqa-hpc}
MATRIX_OUTPUT_ROOT=${MATRIX_OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_h20_tp8_all}
MATRIX_RUN_ID_BASE=${MATRIX_RUN_ID_BASE:-h20-tp8-all-$(date -u +%Y%m%dT%H%M%SZ)}
ATTEMPT=${ATTEMPT:-auto}

OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-1024}
NUM_WARMUPS=${NUM_WARMUPS:-64}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
BOOTSTRAP_ENV=${BOOTSTRAP_ENV:-1}
INSTALL_ONLY=${INSTALL_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

die() {
    echo "error: $*" >&2
    exit 2
}

select_session_dir() {
    local index

    [[ "$MATRIX_RUN_ID_BASE" =~ ^[A-Za-z0-9._-]+$ ]] \
        || die "MATRIX_RUN_ID_BASE contains unsupported characters"
    if [[ "$ATTEMPT" == auto ]]; then
        index=0
        while [[ -e "$MATRIX_OUTPUT_ROOT/${MATRIX_RUN_ID_BASE}-attempt${index}" ]]; do
            index=$((index + 1))
        done
    else
        [[ "$ATTEMPT" =~ ^[0-9]+$ ]] \
            || die "ATTEMPT must be auto or a non-negative integer"
        index=$ATTEMPT
        [[ ! -e "$MATRIX_OUTPUT_ROOT/${MATRIX_RUN_ID_BASE}-attempt${index}" ]] \
            || die "matrix session already exists: $MATRIX_OUTPUT_ROOT/${MATRIX_RUN_ID_BASE}-attempt${index}"
    fi
    MATRIX_RUN_ID=${MATRIX_RUN_ID_BASE}-attempt${index}
    SESSION_DIR=$MATRIX_OUTPUT_ROOT/$MATRIX_RUN_ID
}

profile_config() {
    local profile=$1

    case "$profile" in
        2k)
            INPUT_LEN=2048
            MAX_MODEL_LEN=4096
            MAX_NUM_BATCHED_TOKENS=8192
            PROFILE_NUM_PROMPTS=$NUM_PROMPTS
            PROFILE_NUM_WARMUPS=$NUM_WARMUPS
            PROFILE_MAX_CONCURRENCY=$MAX_CONCURRENCY
            PROFILE_MAX_NUM_SEQS=$MAX_NUM_SEQS
            ;;
        8k)
            INPUT_LEN=8192
            MAX_MODEL_LEN=9216
            MAX_NUM_BATCHED_TOKENS=16384
            PROFILE_NUM_PROMPTS=$NUM_PROMPTS
            PROFILE_NUM_WARMUPS=$NUM_WARMUPS
            PROFILE_MAX_CONCURRENCY=$MAX_CONCURRENCY
            PROFILE_MAX_NUM_SEQS=$MAX_NUM_SEQS
            ;;
        16k)
            INPUT_LEN=16384
            MAX_MODEL_LEN=17408
            MAX_NUM_BATCHED_TOKENS=32768
            PROFILE_NUM_PROMPTS=$NUM_PROMPTS
            PROFILE_NUM_WARMUPS=$NUM_WARMUPS
            PROFILE_MAX_CONCURRENCY=$MAX_CONCURRENCY
            PROFILE_MAX_NUM_SEQS=$MAX_NUM_SEQS
            ;;
        16k-b20)
            INPUT_LEN=16384
            MAX_MODEL_LEN=17408
            MAX_NUM_BATCHED_TOKENS=32768
            PROFILE_NUM_PROMPTS=20
            PROFILE_NUM_WARMUPS=20
            PROFILE_MAX_CONCURRENCY=20
            PROFILE_MAX_NUM_SEQS=20
            ;;
        *) die "unknown profile '$profile'; use 2k, 8k, 16k, or 16k-b20" ;;
    esac
}

case "$BOOTSTRAP_ENV:$INSTALL_ONLY:$DRY_RUN" in
    [01]:[01]:[01]) ;;
    *) die "BOOTSTRAP_ENV, INSTALL_ONLY, and DRY_RUN must each be 0 or 1" ;;
esac
[[ -f "$BASE_SCRIPT" ]] || die "missing H20 base launcher: $BASE_SCRIPT"
[[ -f "$MODEL_DIR/config.json" ]] || die "missing converted model: $MODEL_DIR/config.json"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] \
    || die "missing converted model index: $MODEL_DIR/model.safetensors.index.json"

profiles_clean=${PROFILES//[[:space:]]/}
paths_clean=${PATHS//[[:space:]]/}
IFS=',' read -r -a profile_list <<<"$profiles_clean"
IFS=',' read -r -a path_list <<<"$paths_clean"
(( ${#profile_list[@]} > 0 )) || die "PROFILES is empty"
(( ${#path_list[@]} > 0 )) || die "PATHS is empty"
for route in "${path_list[@]}"; do
    case "$route" in
        mqa-mla|gqa-hpc) ;;
        *) die "unknown route '$route'; use mqa-mla or gqa-hpc" ;;
    esac
done

select_session_dir
mkdir -p "$SESSION_DIR"
exec > >(tee -a "$SESSION_DIR/launcher.log") 2>&1

echo "[matrix] session=$SESSION_DIR"
echo "[matrix] root=$GQLA_ROOT model=$MODEL_DIR"
echo "[matrix] profiles=$profiles_clean paths=$paths_clean"

if [[ "$BOOTSTRAP_ENV" == 1 ]]; then
    echo "[matrix] checking/installing the shared H20 environment once"
    env BASH_ENV=/dev/null ENV=/dev/null \
        GQLA_ROOT="$GQLA_ROOT" MODEL_DIR="$MODEL_DIR" \
        PATHS="$paths_clean" BOOTSTRAP_ENV=1 INSTALL_ONLY=1 \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
        bash "$BASE_SCRIPT"
fi
if [[ "$INSTALL_ONLY" == 1 ]]; then
    echo "H20_ALL_INSTALL_ONLY_OK session=$SESSION_DIR"
    exit 0
fi

{
    printf 'profile\troute\tstatus\tresult_root\n'
} >"$SESSION_DIR/matrix.tsv"

for profile in "${profile_list[@]}"; do
    profile_config "$profile"
    for route in "${path_list[@]}"; do
        case_root=$SESSION_DIR/results/$profile/$route
        run_id_base=$MATRIX_RUN_ID-$profile-$route
        echo "[matrix] START profile=$profile route=$route input=$INPUT_LEN output=$OUTPUT_LEN concurrency=$PROFILE_MAX_CONCURRENCY"

        if env BASH_ENV=/dev/null ENV=/dev/null \
            GQLA_ROOT="$GQLA_ROOT" MODEL_DIR="$MODEL_DIR" \
            OUTPUT_ROOT="$case_root" RUN_ID_BASE="$run_id_base" ATTEMPT=auto \
            PATHS="$route" BOOTSTRAP_ENV=0 INSTALL_ONLY=0 DRY_RUN="$DRY_RUN" \
            INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" \
            MAX_MODEL_LEN="$MAX_MODEL_LEN" \
            MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
            MAX_NUM_SEQS="$PROFILE_MAX_NUM_SEQS" \
            NUM_PROMPTS="$PROFILE_NUM_PROMPTS" \
            NUM_WARMUPS="$PROFILE_NUM_WARMUPS" \
            MAX_CONCURRENCY="$PROFILE_MAX_CONCURRENCY" \
            GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
            VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}" \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
            bash "$BASE_SCRIPT"; then
            printf '%s\t%s\t%s\t%s\n' "$profile" "$route" verified "$case_root" \
                >>"$SESSION_DIR/matrix.tsv"
            echo "[matrix] PASS profile=$profile route=$route"
        else
            status=$?
            printf '%s\t%s\t%s\t%s\n' "$profile" "$route" "failed:$status" "$case_root" \
                >>"$SESSION_DIR/matrix.tsv"
            echo "[matrix] FAIL profile=$profile route=$route status=$status" >&2
            exit "$status"
        fi
    done
done

echo "H20_ALL_BENCHMARK_OK session=$SESSION_DIR summary=$SESSION_DIR/matrix.tsv"
