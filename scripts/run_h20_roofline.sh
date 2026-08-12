#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=common.sh
source "$script_dir/common.sh"

roofline_out=${GQLA_ROOFLINE_OUTPUT_DIR:-"$h20_output_dir/roofline"}
roofline_json="$roofline_out/h20_hardware_roofline.json"
roofline_log="$roofline_out/h20_hardware_roofline.log"

mkdir -p "$roofline_out"
exec > >(tee "$roofline_log") 2>&1

on_error() {
    roofline_rc=$?
    printf 'HARDWARE_ROOFLINE_FAILED rc=%s line=%s\n' \
        "$roofline_rc" "${BASH_LINENO[0]}" >&2
    printf 'Log: %s\n' "$roofline_log" >&2
    exit "$roofline_rc"
}
trap on_error ERR

test -x "$h20_python"
command -v nvidia-smi

printf 'Check that no other process is using this GPU before trusting the result.\n'
nvidia-smi -L | tee "$roofline_out/nvidia_smi_L.txt"
nvidia-smi -q | tee "$roofline_out/nvidia_smi_q_before.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>&1 | tee "$roofline_out/compute_processes_before.txt" || true

CUDA_VISIBLE_DEVICES="$h20_visible_gpu" PYTHONUNBUFFERED=1 \
    "$h20_python" -u "$script_dir/gpu_preflight.py" \
        --expect-device-substring "$h20_expected_device_substring"

CUDA_VISIBLE_DEVICES="$h20_visible_gpu" PYTHONUNBUFFERED=1 \
    "$h20_python" -u "$script_dir/benchmark_hardware_roofline.py" \
        --output "$roofline_json" \
        --expect-device-substring "$h20_expected_device_substring" \
        --read-gib "${GQLA_ROOFLINE_READ_GIB:-4}" \
        --copy-gib "${GQLA_ROOFLINE_COPY_GIB:-2}" \
        --stream-gib "${GQLA_ROOFLINE_STREAM_GIB:-1}" \
        --warmup "${GQLA_ROOFLINE_WARMUP:-5}" \
        --samples "${GQLA_ROOFLINE_SAMPLES:-15}" \
        --bandwidth-inner "${GQLA_ROOFLINE_BANDWIDTH_INNER:-5}" \
        --gemm-warmup "${GQLA_ROOFLINE_GEMM_WARMUP:-3}" \
        --gemm-samples "${GQLA_ROOFLINE_GEMM_SAMPLES:-9}" \
        --gemm-sizes "${GQLA_ROOFLINE_GEMM_SIZES:-8192,12288,16384}" \
        --expected-hbm-tb-s "${GQLA_ROOFLINE_EXPECTED_HBM_TB_S:-4.0}" \
        --expected-bf16-tflops "${GQLA_ROOFLINE_EXPECTED_BF16_TFLOPS:-148.0}" \
        --fa3-gqa-equivalent-tb-s "${GQLA_ROOFLINE_FA3_GQA_TB_S:-1.03}"

nvidia-smi -q | tee "$roofline_out/nvidia_smi_q_after.txt"
printf 'HARDWARE_ROOFLINE_OK\n'
printf 'JSON: %s\n' "$roofline_json"
printf 'Log: %s\n' "$roofline_log"
