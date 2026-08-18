#!/usr/bin/env bash
# One DLC entrypoint for the 2K H100 control/treatment comparison. DLC runs
# this same script on both 8-GPU workers. Each child resolves its own node role,
# so the two nodes run MLA first and all-decode HPC-Ops GQLA second.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GQLA_ROOT=${GQLA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}
MLA_SCRIPT=$SCRIPT_DIR/benchmark_dsv3p1_g8_mla_tp8_pp2.sh
GQLA_SCRIPT=$SCRIPT_DIR/benchmark_dsv3p1_g8_gqla_h100_tp8_pp2_2k.sh

SERIES_ID=${SERIES_ID:-${RUN_ID:-h100-tp8-pp2-2k-all-decode-001}}
MLA_RUN_ID=${MLA_RUN_ID:-${SERIES_ID}-mla}
GQLA_RUN_ID=${GQLA_RUN_ID:-${SERIES_ID}-gqla}
MLA_OUTPUT_ROOT=${MLA_OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_mla_h100_tp8_pp2_2k}
GQLA_OUTPUT_ROOT=${GQLA_OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_gqla_h100_tp8_pp2_2k}
SERIAL_COOLDOWN_SECONDS=${SERIAL_COOLDOWN_SECONDS:-30}
RUN_MIXED_SMOKE=${RUN_MIXED_SMOKE:-1}
DRY_RUN=${DRY_RUN:-0}
CHILD_PID=

die() {
    echo "error: $*" >&2
    exit 2
}

cleanup_child() {
    local pid=${CHILD_PID:-}
    CHILD_PID=
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup_child
    exit "$rc"
}

on_signal() {
    trap - EXIT INT TERM
    cleanup_child
    exit 130
}

run_route() {
    local label=$1
    local script=$2
    local run_id=$3
    local output_root=$4
    local rc

    echo "[serial] starting route=$label run_id=$run_id output_root=$output_root"
    env -u RUN_ID_BASE \
        RUN_ID="$run_id" \
        OUTPUT_ROOT="$output_root" \
        DRY_RUN="$DRY_RUN" \
        bash "$script" &
    CHILD_PID=$!
    set +e
    wait "$CHILD_PID"
    rc=$?
    set -e
    CHILD_PID=
    if (( rc != 0 )); then
        echo "[serial] route=$label failed rc=$rc; later routes will not run" >&2
        return "$rc"
    fi
    echo "[serial] completed route=$label run_id=$run_id"
}

run_mixed_smoke() {
    local node_rank=${MACHINE_RANK:-${NODE_RANK:-${RANK:-}}}
    local visible_device=${SMOKE_VISIBLE_DEVICE:-${CUDA_VISIBLE_DEVICES:-}}
    local smoke_root=$GQLA_ROOT/outputs/smoke/h100-all-decode/$SERIES_ID

    [[ "$node_rank" =~ ^[01]$ ]] \
        || die "mixed smoke needs DLC node rank 0 or 1; got ${node_rank:-unset}"
    visible_device=${visible_device%%,*}
    visible_device=${visible_device:-0}

    echo "[serial] mixed smoke node=$node_rank device=$visible_device"
    CUDA_VISIBLE_DEVICES="$visible_device" \
    PYTHON=${PYTHON:-$GQLA_ROOT/envs/venv-py312/bin/python} \
    TPS=1 \
    SMOKE_PATHS=gqa-hpc \
    VERIFY_HPC_TRACE=1 \
    VERIFY_HPC_MIXED_SPLIT=1 \
    LOG_ROOT="$smoke_root/node-$node_rank" \
    bash "$SCRIPT_DIR/run_dsv3p1_g8_vllm_smoke.sh"
    echo "[serial] mixed smoke passed node=$node_rank"
}

for path in "$MLA_SCRIPT" "$GQLA_SCRIPT"; do
    [[ -f "$path" ]] || die "missing child benchmark entrypoint: $path"
done
[[ "$SERIES_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "SERIES_ID may contain only letters, digits, dot, underscore, and dash"
[[ "$MLA_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "MLA_RUN_ID contains unsupported characters"
[[ "$GQLA_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "GQLA_RUN_ID contains unsupported characters"
[[ "$SERIAL_COOLDOWN_SECONDS" =~ ^[0-9]+$ ]] \
    || die "SERIAL_COOLDOWN_SECONDS must be a non-negative integer"
case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1; got $DRY_RUN" ;;
esac
case "$RUN_MIXED_SMOKE" in
    0|1) ;;
    *) die "RUN_MIXED_SMOKE must be 0 or 1; got $RUN_MIXED_SMOKE" ;;
esac

trap on_exit EXIT
trap on_signal INT TERM

echo "[serial] series=$SERIES_ID order=MLA,GQLA cooldown=${SERIAL_COOLDOWN_SECONDS}s"
if [[ "$DRY_RUN" == 0 && "$RUN_MIXED_SMOKE" == 1 ]]; then
    run_mixed_smoke
fi
run_route MLA "$MLA_SCRIPT" "$MLA_RUN_ID" "$MLA_OUTPUT_ROOT"
if [[ "$DRY_RUN" == 0 ]] && (( SERIAL_COOLDOWN_SECONDS > 0 )); then
    echo "[serial] cooling down ${SERIAL_COOLDOWN_SECONDS}s before GQLA"
    sleep "$SERIAL_COOLDOWN_SECONDS"
fi
run_route GQLA "$GQLA_SCRIPT" "$GQLA_RUN_ID" "$GQLA_OUTPUT_ROOT"

echo "DSV3P1_G8_H100_TP8_PP2_2K_SERIAL_OK series=$SERIES_ID"
