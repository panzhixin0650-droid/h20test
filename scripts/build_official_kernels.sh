#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=common.sh
source "$script_dir/common.sh"

bench_python=$h20_python
bench_uv=$h20_uv
bench_out=$h20_output_dir
bench_preflight="$script_dir/gpu_preflight.py"
bench_source_record=$h20_source_record
bench_run_log="$bench_out/build_driver.log"
bench_enable_fa3_mla=${GQLA_KERNEL_ENABLE_FA3_MLA:-1}
bench_fa3_only=${GQLA_KERNEL_FA3_ONLY:-0}
bench_visible_gpu=$h20_visible_gpu
bench_cuda_home=$h20_cuda_home

if [[ "$bench_enable_fa3_mla" != 0 && "$bench_enable_fa3_mla" != 1 ]]; then
    printf 'GQLA_KERNEL_ENABLE_FA3_MLA must be 0 or 1\n' >&2
    exit 2
fi
if [[ "$bench_fa3_only" != 0 && "$bench_fa3_only" != 1 ]]; then
    printf 'GQLA_KERNEL_FA3_ONLY must be 0 or 1\n' >&2
    exit 2
fi

bench_disable_hdim64=TRUE
bench_disable_hdimdiff64=TRUE
bench_fa3_log="$bench_out/build_fa3.log"
if [[ "$bench_enable_fa3_mla" == 1 ]]; then
    bench_disable_hdim64=FALSE
    bench_disable_hdimdiff64=FALSE
    bench_fa3_log="$bench_out/build_fa3_with_mla_64x512.log"
fi

mkdir -p "$bench_out"
exec > >(tee -a "$bench_run_log") 2>&1

on_error() {
    bench_rc=$?
    printf 'BUILD_FAILED rc=%s line=%s\n' "$bench_rc" "${BASH_LINENO[0]}" >&2
    printf 'Driver log: %s\n' "$bench_run_log" >&2
    exit "$bench_rc"
}
trap on_error ERR

test -x "$bench_python"
if [[ -z "$bench_uv" || ! -x "$bench_uv" ]]; then
    printf 'uv is unavailable; set GQLA_TABLE2_UV=/absolute/path/to/uv\n' >&2
    exit 2
fi
test -x "$bench_cuda_home/bin/nvcc"
test -d "$bench_cuda_home/include"
test -f "$bench_preflight"
command -v git
command -v nvidia-smi
command -v sha256sum
(cd "$h20_repo_root" && sha256sum -c patches/SHA256SUMS)

printf 'Running GPU preflight\n'
CUDA_VISIBLE_DEVICES="$bench_visible_gpu" PYTHONUNBUFFERED=1 \
    "$bench_python" -u "$bench_preflight" \
        --expect-device-substring "$h20_expected_device_substring" \
    2>&1 | tee "$bench_out/gpu_preflight.log"

nvidia-smi | tee "$bench_out/nvidia_smi_prebuild.txt"
free -h | tee "$bench_out/memory_prebuild.txt"
"$bench_cuda_home/bin/nvcc" --version | tee "$bench_out/nvcc_version.txt"

bench_cpu_count=$(nproc)
bench_worker_budget=${GQLA_KERNEL_BUILD_CPU_BUDGET:-$bench_cpu_count}
if [[ ! "$bench_worker_budget" =~ ^[1-9][0-9]*$ ]]; then
    printf 'GQLA_KERNEL_BUILD_CPU_BUDGET must be a positive integer\n' >&2
    exit 2
fi
if (( bench_worker_budget > bench_cpu_count )); then
    bench_worker_budget=$bench_cpu_count
fi

bench_pids_max=max
for bench_pids_file in /sys/fs/cgroup/pids.max /sys/fs/cgroup/pids/pids.max; do
    if [[ -r "$bench_pids_file" ]]; then
        bench_pids_max=$(<"$bench_pids_file")
        break
    fi
done
if [[ "$bench_pids_max" =~ ^[1-9][0-9]*$ ]]; then
    bench_pid_budget=$((bench_pids_max * 3 / 4))
    if (( bench_worker_budget > bench_pid_budget )); then
        bench_worker_budget=$bench_pid_budget
    fi
fi

bench_mla_split=$((bench_worker_budget / 23))
bench_fa3_split=$((bench_worker_budget / 12))
if (( bench_mla_split < 1 )); then
    bench_mla_split=1
fi
if (( bench_fa3_split < 1 )); then
    bench_fa3_split=1
fi

printf 'nproc=%s pids.max=%s worker_budget=%s\n' \
    "$bench_cpu_count" "$bench_pids_max" "$bench_worker_budget"
printf 'FlashMLA MAX_JOBS=24 NVCC split-compile=%s\n' "$bench_mla_split"
printf 'FA3 MAX_JOBS=13 NVCC split-compile=%s\n' "$bench_fa3_split"
printf 'FA3 MLA 64x512 enabled=%s FA3-only rebuild=%s\n' \
    "$bench_enable_fa3_mla" "$bench_fa3_only"

bench_tmp=
if [[ -s "$bench_source_record" ]]; then
    IFS= read -r bench_tmp < "$bench_source_record"
fi
if [[ -z "$bench_tmp" || ! -d "$bench_tmp" ]]; then
    test -d "$h20_tmp_parent"
    bench_tmp=$(mktemp -d "$h20_tmp_parent/gqla-table2-official-XXXXXXXX")
    printf '%s\n' "$bench_tmp" | tee "$bench_source_record"
fi
if [[ -e "$bench_tmp/FlashMLA" && ! -d "$bench_tmp/FlashMLA/.git" ]]; then
    bench_tmp=$(mktemp -d "$h20_tmp_parent/gqla-table2-official-XXXXXXXX")
    printf '%s\n' "$bench_tmp" | tee "$bench_source_record"
fi
if [[ -e "$bench_tmp/flash-attention" && ! -d "$bench_tmp/flash-attention/.git" ]]; then
    bench_tmp=$(mktemp -d "$h20_tmp_parent/gqla-table2-official-XXXXXXXX")
    printf '%s\n' "$bench_tmp" | tee "$bench_source_record"
fi

printf 'Official source directory: %s\n' "$bench_tmp"

if [[ ! -d "$bench_tmp/FlashMLA/.git" ]]; then
    git clone https://github.com/deepseek-ai/FlashMLA.git "$bench_tmp/FlashMLA"
fi
git -C "$bench_tmp/FlashMLA" checkout \
    --detach 15f13e5030374295491c5ce31b02d7e63a7772c6
git -C "$bench_tmp/FlashMLA" submodule update --init csrc/cutlass

if [[ ! -d "$bench_tmp/flash-attention/.git" ]]; then
    git clone https://github.com/Dao-AILab/flash-attention.git \
        "$bench_tmp/flash-attention"
fi
git -C "$bench_tmp/flash-attention" checkout \
    --detach a369df707e1980fb328abcc1733e3457ec10155f
git -C "$bench_tmp/flash-attention" submodule update --init csrc/cutlass

{
    printf 'FlashMLA='
    git -C "$bench_tmp/FlashMLA" rev-parse HEAD
    printf 'FlashMLA-CUTLASS='
    git -C "$bench_tmp/FlashMLA/csrc/cutlass" rev-parse HEAD
    printf 'FlashAttention='
    git -C "$bench_tmp/flash-attention" rev-parse HEAD
    printf 'FlashAttention-CUTLASS='
    git -C "$bench_tmp/flash-attention/csrc/cutlass" rev-parse HEAD
} | tee "$bench_out/source_commits.txt"

if [[ "$bench_fa3_only" == 0 ]]; then
    printf 'Starting FlashMLA compilation\n'
    bench_mla_env=(
        "CUDA_HOME=$bench_cuda_home"
        FLASH_MLA_DISABLE_SM100=1
        FLASH_MLA_DISABLE_FP16=1
        MAX_JOBS=24
        NVCC_THREADS=2
        "NVCC_APPEND_FLAGS=--split-compile=$bench_mla_split"
        "UV_CACHE_DIR=$bench_tmp/uv-cache"
        UV_LINK_MODE=copy
        'NINJA_STATUS=[%f/%t, %o/sec] '
    )
    # CUDA 13 moved CCCL below include/cccl. CUDA 12.8, which the
    # official FlashMLA/FA3 repositories recommend for Hopper, uses the
    # standard CUDA include search path and must not require that directory.
    if [[ -d "$bench_cuda_home/include/cccl" ]]; then
        bench_mla_env+=(
            "CPLUS_INCLUDE_PATH=$bench_cuda_home/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
            "C_INCLUDE_PATH=$bench_cuda_home/include/cccl${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
        )
    fi
    env "${bench_mla_env[@]}" \
        "$bench_uv" pip install \
            --python "$bench_python" \
            --no-build-isolation \
            --no-deps \
            --reinstall \
            -v "$bench_tmp/FlashMLA" \
        2>&1 | tee "$bench_out/build_flashmla.log"
else
    printf 'Skipping FlashMLA compilation (GQLA_KERNEL_FA3_ONLY=1)\n'
fi

printf 'Starting FlashAttention-3 compilation\n'
env \
    CUDA_HOME="$bench_cuda_home" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_OFFLINE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
    FLASH_ATTENTION_DISABLE_SPLIT=FALSE \
    FLASH_ATTENTION_DISABLE_PAGEDKV=FALSE \
    FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
    FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
    FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
    FLASH_ATTENTION_DISABLE_PACKGQA=FALSE \
    FLASH_ATTENTION_DISABLE_FP16=TRUE \
    FLASH_ATTENTION_DISABLE_FP8=TRUE \
    FLASH_ATTENTION_DISABLE_VARLEN=FALSE \
    FLASH_ATTENTION_DISABLE_CLUSTER=FALSE \
    FLASH_ATTENTION_DISABLE_HDIM64="$bench_disable_hdim64" \
    FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM128=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM192=FALSE \
    FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=TRUE \
    FLASH_ATTENTION_DISABLE_HDIMDIFF64="$bench_disable_hdimdiff64" \
    FLASH_ATTENTION_DISABLE_HDIMDIFF192=FALSE \
    MAX_JOBS=13 \
    NVCC_THREADS=2 \
    NVCC_APPEND_FLAGS="--split-compile=$bench_fa3_split" \
    UV_CACHE_DIR="$bench_tmp/uv-cache" \
    UV_LINK_MODE=copy \
    NINJA_STATUS='[%f/%t, %o/sec] ' \
    "$bench_uv" pip install \
        --python "$bench_python" \
        --no-build-isolation \
        --no-deps \
        --reinstall \
        -v "$bench_tmp/flash-attention/hopper" \
    2>&1 | tee "$bench_fa3_log"

if [[ "$bench_enable_fa3_mla" == 1 ]]; then
    printf 'Verifying FA3 MLA build flags\n'
    "$bench_python" -c \
        'from flash_attn_config import CONFIG; flags = CONFIG["build_flags"]; assert not flags["FLASHATTENTION_DISABLE_HDIM64"], flags; assert not flags["FLASH_ATTENTION_DISABLE_HDIMDIFF64"], flags; print(flags, flush=True)'
fi

printf 'Verifying standalone official extensions\n'
CUDA_VISIBLE_DEVICES="$bench_visible_gpu" PYTHONUNBUFFERED=1 \
    "$bench_python" -u "$bench_preflight" \
        --extensions \
        --expect-device-substring "$h20_expected_device_substring" \
    2>&1 | tee "$bench_out/build_verify.log"

printf 'BUILD_OK\n'
printf 'Official source directory: %s\n' "$bench_tmp"
printf 'Logs: %s\n' "$bench_out"
