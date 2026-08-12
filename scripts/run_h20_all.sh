#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=common.sh
source "$script_dir/common.sh"

skip_build=${GQLA_TABLE2_SKIP_BUILD:-0}
if [[ "$skip_build" != 0 && "$skip_build" != 1 ]]; then
    printf 'GQLA_TABLE2_SKIP_BUILD must be 0 or 1\n' >&2
    exit 2
fi

mkdir -p "$h20_output_dir"

CUDA_VISIBLE_DEVICES="$h20_visible_gpu" PYTHONUNBUFFERED=1 \
    "$h20_python" -u "$script_dir/gpu_preflight.py" \
        --expect-device-substring "$h20_expected_device_substring"

nvidia-smi -L | tee "$h20_output_dir/h20_nvidia_smi_L.txt"
nvidia-smi -q | tee "$h20_output_dir/h20_nvidia_smi_q.txt"
nvidia-smi topo -m | tee "$h20_output_dir/h20_topology.txt"

if [[ "$skip_build" == 0 ]]; then
    GQLA_KERNEL_ENABLE_FA3_MLA=1 \
        bash "$script_dir/build_official_kernels.sh"
else
    printf 'Skipping extension compilation (GQLA_TABLE2_SKIP_BUILD=1)\n'
fi

GQLA_TABLE2_DEVICE_ROLE=h20 \
GQLA_TABLE2_SMALL_BATCH_RETRY=0 \
GQLA_TABLE2_BATCH_PROFILE=0 \
GQLA_TABLE2_RERUN_ALL=1 \
GQLA_TABLE2_TP_SWEEP=0 \
    bash "$script_dir/run_official_dense_batch.sh"

GQLA_TABLE2_DEVICE_ROLE=h20 \
GQLA_TABLE2_SMALL_BATCH_RETRY=0 \
GQLA_TABLE2_BATCH_PROFILE=0 \
GQLA_TABLE2_RERUN_ALL=0 \
GQLA_TABLE2_TP_SWEEP=1 \
    bash "$script_dir/run_official_dense_batch.sh"

"$h20_python" "$script_dir/verify_results.py" \
    --output-dir "$h20_output_dir"

printf 'H20_REPRODUCTION_OK\n'
printf 'Results: %s\n' "$h20_output_dir"
