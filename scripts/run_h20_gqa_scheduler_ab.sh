#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=common.sh
source "$script_dir/common.sh"

ab_out=${GQLA_GQA_AB_OUTPUT_DIR:-"$h20_output_dir/gqa_scheduler_ab"}
ab_json="$ab_out/h20_fa3_gqa_scheduler_ab.json"
ab_log="$ab_out/h20_fa3_gqa_scheduler_ab.log"

mkdir -p "$ab_out"
exec > >(tee "$ab_log") 2>&1

on_error() {
    ab_rc=$?
    printf 'FA3_GQA_SCHEDULER_AB_FAILED rc=%s line=%s\n' \
        "$ab_rc" "${BASH_LINENO[0]}" >&2
    printf 'Log: %s\n' "$ab_log" >&2
    exit "$ab_rc"
}
trap on_error ERR

test -x "$h20_python"

CUDA_VISIBLE_DEVICES="$h20_visible_gpu" PYTHONUNBUFFERED=1 \
    "$h20_python" -u "$script_dir/gpu_preflight.py" \
        --extensions \
        --expect-device-substring "$h20_expected_device_substring"

CUDA_VISIBLE_DEVICES="$h20_visible_gpu" PYTHONUNBUFFERED=1 \
    "$h20_python" -u "$script_dir/diagnose_fa3_gqa_scheduler.py" \
        --output "$ab_json" \
        --expect-device-substring "$h20_expected_device_substring" \
        --batch-size "${GQLA_GQA_AB_BATCH_SIZE:-128}" \
        --warmup "${GQLA_GQA_AB_WARMUP:-5}" \
        --samples "${GQLA_GQA_AB_SAMPLES:-21}" \
        --measured-hbm-tb-s "${GQLA_GQA_AB_HBM_TB_S:-3.572}" \
        --measured-bf16-tflops "${GQLA_GQA_AB_BF16_TFLOPS:-141.48}"

printf 'FA3_GQA_SCHEDULER_AB_OK\n'
printf 'JSON: %s\n' "$ab_json"
printf 'Log: %s\n' "$ab_log"
