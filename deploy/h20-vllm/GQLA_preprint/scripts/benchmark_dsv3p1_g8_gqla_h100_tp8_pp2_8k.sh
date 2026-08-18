#!/usr/bin/env bash
# DLC single-entry H100 benchmark: true GQLA cache + HPC-Ops decode, TP8/PP2,
# exact 8K input and 128-token output. Submit this same entrypoint to a
# two-worker DLC job with 8 visible GPUs per worker.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GQLA_ROOT=${GQLA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}
RUNTIME_ENV_FILE=${RUNTIME_ENV_FILE:-$GQLA_ROOT/envs/venv-py312/h20-runtime.env}
EXPECTED_HPC_SOURCE_HASH=609ecd832da7883692c0a13f8a3c789cc7006833213f367e702d67ca430f0fb0

[[ -f "$RUNTIME_ENV_FILE" ]] || {
    echo "error: runtime environment is missing: $RUNTIME_ENV_FILE" >&2
    exit 2
}
# shellcheck disable=SC1090
source "$RUNTIME_ENV_FILE"
[[ -f "$VENV_DIR/.hpc-source.sha256" ]] && \
[[ "$(<"$VENV_DIR/.hpc-source.sha256")" == "$EXPECTED_HPC_SOURCE_HASH" ]] || {
    echo "error: installed HPC-Ops is not the verified adaptive split-K build" >&2
    exit 2
}

export BENCHMARK_ENTRYPOINT=${BASH_SOURCE[0]}
export BENCHMARK_PATH=gqa-hpc
export MAX_CONCURRENCY=64
export MAX_NUM_SEQS=64
export NUM_PROMPTS=1024
export NUM_WARMUPS=64
export REQUEST_RATE=inf
export CPU_OFFLOAD_GB=0
export INPUT_LEN=8192
export OUTPUT_LEN=128
export MAX_MODEL_LEN=9216
export MAX_NUM_BATCHED_TOKENS=16384
export SERVED_MODEL_NAME=dsv3p1-g8-gqla-h100-tp8-pp2-8k
export OUTPUT_ROOT=${OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_gqla_h100_tp8_pp2_8k}
export RUN_ID=${RUN_ID:-h100-gqla-tp8-pp2-8k-001}

exec bash "$SCRIPT_DIR/benchmark_dsv3p1_g8_mla_tp8_pp2.sh" "$@"
