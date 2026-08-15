#!/usr/bin/env bash
# Single-node H20 benchmark: standard MLA absorb route, TP8/PP1, exact 16K
# input and 128-token output. The shared H20 launcher enforces route/backend
# verification and selects an unused -attemptN result directory.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GQLA_ROOT=${GQLA_ROOT:-/mnt/tidalfs-alwl01/task/236362/GQLA}
export GQLA_ROOT
export MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract}

export BENCHMARK_ENTRYPOINT=${BASH_SOURCE[0]}
export PATHS=mqa-mla
export INPUT_LEN=16384
export OUTPUT_LEN=128
export MAX_MODEL_LEN=17408
export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_mla_h20_tp8_16k}
export RUN_ID=${RUN_ID:-h20-mla-tp8-16k-001}

exec env BASH_ENV=/dev/null ENV=/dev/null \
    bash "$SCRIPT_DIR/benchmark_dsv3p1_g8_h20_tp8.sh" "$@"
