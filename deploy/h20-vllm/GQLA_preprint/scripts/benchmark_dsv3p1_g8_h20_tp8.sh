#!/usr/bin/env bash
# Single-node TP=8 vLLM benchmark for the converted DeepSeek-V3.1 G=8
# checkpoint. By default it runs two fresh-server cases with the same workload:
# MLA control first, followed by true GQLA decode through HPC-Ops.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
DEFAULT_GQLA_ROOT=$(cd "$REPO_DIR/../.." && pwd -P)
DEFAULT_MODEL_DIR=$DEFAULT_GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
if [[ -z "${GQLA_ROOT:-}" ]]; then
    case "$MODEL_DIR" in
        */outputs/convert/*) GQLA_ROOT=${MODEL_DIR%%/outputs/convert/*} ;;
        *) GQLA_ROOT=$DEFAULT_GQLA_ROOT ;;
    esac
fi
BASE_SCRIPT=$SCRIPT_DIR/benchmark_dsv3p1_g8_vllm.sh
BOOTSTRAP_SCRIPT=$SCRIPT_DIR/bootstrap_dsv3p1_g8_h20_env.sh
ENV_ROOT=${ENV_ROOT:-$GQLA_ROOT/envs}
VENV_DIR=${VENV_DIR:-$ENV_ROOT/venv-py312}
PYTHON=${PYTHON:-$VENV_DIR/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_h20_tp8}
if [[ -z "${HPC_OPS_DIR:-}" ]]; then
    if [[ -d "$REPO_DIR/../hpc-ops" ]]; then
        HPC_OPS_DIR=$(cd "$REPO_DIR/../hpc-ops" && pwd -P)
    else
        HPC_OPS_DIR=$GQLA_ROOT/code/hpc-ops
    fi
fi
BOOTSTRAP_ENV=${BOOTSTRAP_ENV:-1}
INSTALL_ONLY=${INSTALL_ONLY:-0}
BOOTSTRAP_LOG=${BOOTSTRAP_LOG:-$ENV_ROOT/logs/bootstrap-h20.log}

# RUN_ID is a stable experiment base. The first unused -attemptN suffix is
# selected before handing control to the generic matrix runner.
RUN_ID_BASE=${RUN_ID_BASE:-${RUN_ID:-h20-tp8-gqa-vs-mla-$(date -u +%Y%m%dT%H%M%SZ)}}
ATTEMPT=${ATTEMPT:-auto}
RUN_ID=

# PATHS remains run-scoped so either route can also be measured independently.
PATHS=${PATHS:-mqa-mla,gqa-hpc}
readonly TPS=8
readonly PIPELINE_PARALLEL_SIZE=1
readonly DISTRIBUTED_EXECUTOR_BACKEND=mp
readonly NNODES=1
readonly NODE_RANK=0
readonly MASTER_ADDR=127.0.0.1
readonly VLLM_HOST_IP=127.0.0.1
readonly GQA_ARCHITECTURE=DeepseekV3GQLAHPCForCausalLM
readonly MQA_ARCHITECTURE=DeepseekV3GQLAForCausalLM
readonly SERVED_MODEL_NAME=dsv3p1-g8-h20-tp8
readonly EXPECTED_VLLM_VERSION=0.22.1

MASTER_PORT=${MASTER_PORT:-29501}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8100}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}

# Identical exact-length online workload for both paths.
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-1024}
NUM_WARMUPS=${NUM_WARMUPS:-64}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
REQUEST_RATE=${REQUEST_RATE:-inf}
SEED=${SEED:-42}
METRIC_PERCENTILES=${METRIC_PERCENTILES:-50,90,99}
PERCENTILE_METRICS=${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}

# The 660+ GiB FUSE checkpoint can take a long time to scan/load. This timeout
# is per route and exits early if the server process itself fails.
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-10800}
SHUTDOWN_TIMEOUT_SECONDS=${SHUTDOWN_TIMEOUT_SECONDS:-180}
DRY_RUN=${DRY_RUN:-0}

die() {
    echo "error: $*" >&2
    exit 2
}

resolve_run_id() {
    local attempt_index

    [[ "$RUN_ID_BASE" =~ ^[A-Za-z0-9._-]+$ ]] \
        || die "RUN_ID may contain only letters, digits, dot, underscore, and dash"
    if [[ "$ATTEMPT" == auto ]]; then
        attempt_index=0
        if [[ -e "$OUTPUT_ROOT/$RUN_ID_BASE" ]]; then
            attempt_index=1
        fi
        while [[ -e "$OUTPUT_ROOT/${RUN_ID_BASE}-attempt${attempt_index}" ]]; do
            attempt_index=$((attempt_index + 1))
        done
    else
        [[ "$ATTEMPT" =~ ^[0-9]+$ ]] \
            || die "ATTEMPT must be auto or a non-negative integer; got $ATTEMPT"
        attempt_index=$ATTEMPT
    fi
    ATTEMPT=$attempt_index
    RUN_ID=${RUN_ID_BASE}-attempt${ATTEMPT}
    echo "[attempt] base=$RUN_ID_BASE attempt=$ATTEMPT run_id=$RUN_ID"
}

case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1; got $DRY_RUN" ;;
esac
case "$BOOTSTRAP_ENV" in
    0|1) ;;
    *) die "BOOTSTRAP_ENV must be 0 or 1; got $BOOTSTRAP_ENV" ;;
esac
case "$INSTALL_ONLY" in
    0|1) ;;
    *) die "INSTALL_ONLY must be 0 or 1; got $INSTALL_ONLY" ;;
esac
if [[ "$CPU_OFFLOAD_GB" != 0 && "$CPU_OFFLOAD_GB" != 0.0 ]]; then
    die "formal H20 benchmark fixes CPU_OFFLOAD_GB=0; got $CPU_OFFLOAD_GB"
fi
[[ -f "$BASE_SCRIPT" ]] || die "base benchmark is missing: $BASE_SCRIPT"
requested_model_dir=$MODEL_DIR
MODEL_DIR=$(readlink -f -- "$requested_model_dir") \
    || die "converted model directory does not exist: $requested_model_dir"
[[ -d "$MODEL_DIR" ]] \
    || die "converted model directory does not exist: $requested_model_dir"
for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
    [[ -f "$MODEL_DIR/$required" ]] \
        || die "missing converted-model file: $MODEL_DIR/$required"
done

needs_hpc=0
case ",${PATHS//[[:space:]]/}," in
    *,gqa-hpc,*) needs_hpc=1 ;;
esac
if [[ "$BOOTSTRAP_ENV" == 1 ]]; then
    [[ -f "$BOOTSTRAP_SCRIPT" ]] \
        || die "environment bootstrap script is missing: $BOOTSTRAP_SCRIPT"
    mkdir -p "$ENV_ROOT/logs"
    echo "[h20-single] bootstrapping/checking environment; log=$BOOTSTRAP_LOG"
    if ! env BASH_ENV=/dev/null ENV=/dev/null \
        GQLA_ROOT="$GQLA_ROOT" MODEL_DIR="$MODEL_DIR" ENV_ROOT="$ENV_ROOT" \
        VENV_DIR="$VENV_DIR" HPC_OPS_DIR="$HPC_OPS_DIR" NEEDS_HPC="$needs_hpc" \
        BOOTSTRAP_LOG="$BOOTSTRAP_LOG" \
        bash "$BOOTSTRAP_SCRIPT" 2>&1 | tee -a "$BOOTSTRAP_LOG"; then
        die "H20 environment bootstrap failed; inspect $BOOTSTRAP_LOG"
    fi
fi
if [[ -f "$VENV_DIR/h20-runtime.env" ]]; then
    # Generated by the bootstrapper; carries the matching CUDA/toolchain paths.
    # shellcheck disable=SC1090
    source "$VENV_DIR/h20-runtime.env"
fi
[[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON (set BOOTSTRAP_ENV=1 to install it)"
if [[ "$INSTALL_ONLY" == 1 ]]; then
    echo "H20_INSTALL_ONLY_OK python=$PYTHON model=$MODEL_DIR"
    exit 0
fi
resolve_run_id

export GQLA_ROOT ENV_ROOT VENV_DIR PYTHON MODEL_DIR OUTPUT_ROOT RUN_ID
export BENCHMARK_ENTRYPOINT=${BENCHMARK_ENTRYPOINT:-${BASH_SOURCE[0]}}
export TPS PATHS PIPELINE_PARALLEL_SIZE DISTRIBUTED_EXECUTOR_BACKEND
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT VLLM_HOST_IP
export GQA_ARCHITECTURE MQA_ARCHITECTURE SERVED_MODEL_NAME EXPECTED_VLLM_VERSION
export HOST PORT GPU_MEMORY_UTILIZATION CPU_OFFLOAD_GB
export MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export ENFORCE_EAGER ENABLE_CHUNKED_PREFILL
export INPUT_LEN OUTPUT_LEN NUM_PROMPTS NUM_WARMUPS MAX_CONCURRENCY
export REQUEST_RATE SEED METRIC_PERCENTILES PERCENTILE_METRICS
export STARTUP_TIMEOUT_SECONDS SHUTDOWN_TIMEOUT_SECONDS DRY_RUN
export PP_LAYER_PARTITION=

# A single-node DLC parent may inject scheduler-level rank variables. vLLM must
# construct its own TP=8 process ranks instead of inheriting those values.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
    2>/dev/null || true

echo "[h20-single] paths=$PATHS tp=$TPS pp=$PIPELINE_PARALLEL_SIZE model=$MODEL_DIR"
exec env BASH_ENV=/dev/null ENV=/dev/null bash "$BASE_SCRIPT"
