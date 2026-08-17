#!/usr/bin/env bash
# DLC single-entry H100 benchmark: true GQLA cache + HPC-Ops decode, TP8/PP2,
# exact 2K input and 128-token output. Submit this same entrypoint to a
# two-worker DLC job with 8 visible GPUs per worker.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GQLA_ROOT=${GQLA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}

export BENCHMARK_ENTRYPOINT=${BASH_SOURCE[0]}
export BENCHMARK_PATH=gqa-hpc
export INPUT_LEN=2048
export OUTPUT_LEN=128
export MAX_MODEL_LEN=4096
export MAX_NUM_BATCHED_TOKENS=8192
export SERVED_MODEL_NAME=dsv3p1-g8-gqla-h100-tp8-pp2-2k
export OUTPUT_ROOT=${OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_gqla_h100_tp8_pp2_2k}
export RUN_ID=${RUN_ID:-h100-gqla-tp8-pp2-2k-001}

exec bash "$SCRIPT_DIR/benchmark_dsv3p1_g8_mla_tp8_pp2.sh" "$@"
