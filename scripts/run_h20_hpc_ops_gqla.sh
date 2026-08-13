#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
patch_path="$repo_root/patches/hpc_ops_gqla_qk192_v128.patch"
source_url=${GQLA_HPC_OPS_REPO_URL:-https://github.com/Tencent/hpc-ops.git}
base_commit=${GQLA_HPC_OPS_BASE_COMMIT:-1cd332980ed46bd0172091c1c35d55338fcae47a}
output_root=${GQLA_HPC_OPS_OUTPUT_DIR:-/prodcpfs/user/panzhixin/GQLA/outputs/kernel_table2}
work_parent=${GQLA_HPC_OPS_WORK_PARENT:-${TMPDIR:-/tmp}}
expected_device=${GQLA_HPC_OPS_EXPECT_DEVICE_SUBSTRING:-H20}
visible_gpu=${GQLA_HPC_OPS_GPU:-${GQLA_TABLE2_GPU:-0}}
batch=${GQLA_HPC_OPS_BATCH:-128}
seqlen=${GQLA_HPC_OPS_SEQLEN:-8192}
warmup=${GQLA_HPC_OPS_WARMUP:-5}
iterations=${GQLA_HPC_OPS_ITERS:-20}
flush_gib=${GQLA_HPC_OPS_FLUSH_GIB:-8}
auto_install=${GQLA_HPC_OPS_AUTO_INSTALL:-1}
keep_workdir=${GQLA_HPC_OPS_KEEP_WORKDIR:-0}
conda_env_name=${GQLA_TABLE2_CONDA_ENV:-h20table2}
cuda_home=${CUDA_HOME:-/usr/local/cuda}
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
run_dir="$output_root/runs/hpc_ops_gqla_h20_$run_id"
run_log="$run_dir/hpc_ops_gqla_h20.log"
work_dir=

case "$auto_install" in
    0|1) ;;
    *) printf 'GQLA_HPC_OPS_AUTO_INSTALL must be 0 or 1\n' >&2; exit 2 ;;
esac
case "$keep_workdir" in
    0|1) ;;
    *) printf 'GQLA_HPC_OPS_KEEP_WORKDIR must be 0 or 1\n' >&2; exit 2 ;;
esac

mkdir -p "$run_dir"
exec > >(tee "$run_log") 2>&1

on_error() {
    hpc_rc=$?
    trap - ERR
    printf 'HPC_OPS_GQLA_FAILED rc=%s line=%s\n' \
        "$hpc_rc" "${BASH_LINENO[0]:-unknown}" >&2
    printf 'Log: %s\n' "$run_log" >&2
    if [[ -n "$work_dir" ]]; then
        printf 'Retained work directory for diagnosis: %s\n' "$work_dir" >&2
    fi
    exit "$hpc_rc"
}
trap on_error ERR

python_has_torch() {
    local candidate=$1
    [[ -x "$candidate" ]] && "$candidate" -c 'import torch' >/dev/null 2>&1
}

find_conda_python() {
    local prefix
    command -v conda >/dev/null 2>&1 || return 1
    prefix=$(conda env list | awk -v target="$conda_env_name" \
        '$1 == target {print $NF; exit}')
    [[ -n "$prefix" && -x "$prefix/bin/python" ]] || return 1
    printf '%s\n' "$prefix/bin/python"
}

resolve_python() {
    local candidate
    if [[ -n "${GQLA_HPC_OPS_PYTHON:-}" ]]; then
        printf '%s\n' "$GQLA_HPC_OPS_PYTHON"
        return 0
    fi
    if [[ -n "${GQLA_TABLE2_PYTHON:-}" ]]; then
        printf '%s\n' "$GQLA_TABLE2_PYTHON"
        return 0
    fi
    if [[ -n "${VIRTUAL_ENV:-}" ]] && python_has_torch "$VIRTUAL_ENV/bin/python"; then
        printf '%s\n' "$VIRTUAL_ENV/bin/python"
        return 0
    fi
    if candidate=$(find_conda_python) && python_has_torch "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if python_has_torch "$repo_root/.venv/bin/python"; then
        printf '%s\n' "$repo_root/.venv/bin/python"
        return 0
    fi
    candidate=$(command -v python3 || true)
    if [[ -n "$candidate" ]] && python_has_torch "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi
    return 1
}

printf 'Run ID: %s\n' "$run_id"
printf 'Output directory: %s\n' "$run_dir"

if ! hpc_python=$(resolve_python); then
    if [[ "$auto_install" == 0 ]]; then
        printf '%s\n' \
            'No Python environment with PyTorch was found.' \
            'Set GQLA_HPC_OPS_PYTHON or rerun with GQLA_HPC_OPS_AUTO_INSTALL=1.' >&2
        exit 2
    fi
    printf 'No usable environment found; creating/reusing conda env %s.\n' "$conda_env_name"
    GQLA_TABLE2_CONDA_ENV="$conda_env_name" \
        bash "$script_dir/install_cu128_env.sh"
    hpc_python=$(find_conda_python)
fi

if ! python_has_torch "$hpc_python"; then
    printf 'Selected Python cannot import torch: %s\n' "$hpc_python" >&2
    exit 2
fi
export PATH="$(dirname -- "$hpc_python"):$cuda_home/bin:$PATH"
export CUDA_HOME="$cuda_home"
export CUDA_VISIBLE_DEVICES="$visible_gpu"
export PYTHONUNBUFFERED=1

test -x "$cuda_home/bin/nvcc"
command -v git >/dev/null
command -v cmake >/dev/null
command -v nvidia-smi >/dev/null
"$hpc_python" -c 'import setuptools, torch, wheel; print(f"Python/Torch: {torch.__version__} CUDA={torch.version.cuda}", flush=True)'
cmake --version
"$cuda_home/bin/nvcc" --version
nvidia-smi -L

"$hpc_python" -u "$script_dir/gpu_preflight.py" \
    --expect-device-substring "$expected_device"

(
    cd -- "$repo_root"
    sha256sum --check patches/SHA256SUMS
)
patch_sha256=$(sha256sum "$patch_path" | awk '{print $1}')

mkdir -p "$work_parent"
work_parent=$(cd -- "$work_parent" && pwd -P)
work_dir=$(mktemp -d "$work_parent/hpc-ops-gqla.XXXXXXXX")
source_dir="$work_dir/hpc-ops"
printf 'Work directory: %s\n' "$work_dir"
printf 'Cloning %s at %s\n' "$source_url" "$base_commit"
git clone --depth 1 --filter=blob:none --no-checkout "$source_url" "$source_dir"
git -C "$source_dir" fetch --depth 1 origin "$base_commit"
git -C "$source_dir" checkout --detach "$base_commit"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$base_commit"

git -C "$source_dir" \
    -c user.name='GQLA HPC-Ops benchmark' \
    -c user.email='gqla-hpc-ops@localhost' \
    am "$patch_path"
test -z "$(git -C "$source_dir" status --porcelain)"
patched_commit=$(git -C "$source_dir" rev-parse HEAD)
printf 'Patched commit: %s\n' "$patched_commit"
git -C "$source_dir" show --no-patch --format=fuller HEAD

printf 'Building HPC-Ops wheel...\n'
(
    cd -- "$source_dir"
    "$hpc_python" setup.py bdist_wheel
)

mapfile -t wheels < <(find "$source_dir/dist" -maxdepth 1 -type f -name '*.whl' -print | sort)
if [[ "${#wheels[@]}" != 1 ]]; then
    printf 'Expected one wheel, found %s\n' "${#wheels[@]}" >&2
    exit 1
fi
cp -- "${wheels[0]}" "$run_dir/"
printf 'Wheel: %s/%s\n' "$run_dir" "$(basename -- "${wheels[0]}")"

printf 'Running six reference-correctness cases...\n'
"$hpc_python" -u "$script_dir/check_hpc_ops_gqla.py" \
    --source-dir "$source_dir" \
    --expect-device-substring "$expected_device"

raw_json="$run_dir/hpc_ops_gqla_h20_raw.json"
published_json="$run_dir/hpc_ops_gqla_h20.json"
table_csv="$run_dir/hpc_ops_gqla_h20.csv"
table_markdown="$run_dir/hpc_ops_gqla_h20.md"

printf 'Running B=%s L=%s HPC-Ops GQLA benchmark...\n' "$batch" "$seqlen"
"$hpc_python" -u \
    "$source_dir/benchmark/attention_decode/bench_attention_decode_bf16_gqla.py" \
    --batch "$batch" \
    --seqlen "$seqlen" \
    --warmup "$warmup" \
    --iters "$iterations" \
    --flush-gib "$flush_gib" \
    --modes static \
    --json "$raw_json"

"$hpc_python" -u "$script_dir/render_hpc_ops_gqla_table.py" \
    --input "$raw_json" \
    --output-json "$published_json" \
    --csv "$table_csv" \
    --markdown "$table_markdown" \
    --expect-device-substring "$expected_device" \
    --expect-batch "$batch" \
    --expect-seqlen "$seqlen" \
    --source-url "$source_url" \
    --base-commit "$base_commit" \
    --patched-commit "$patched_commit" \
    --patch-sha256 "$patch_sha256" \
    --run-id "$run_id"

install -m 0644 "$published_json" "$output_root/hpc_ops_gqla_h20.json"
install -m 0644 "$table_csv" "$output_root/hpc_ops_gqla_h20.csv"
install -m 0644 "$table_markdown" "$output_root/hpc_ops_gqla_h20.md"

if [[ "$keep_workdir" == 0 ]]; then
    case "$work_dir" in
        "$work_parent"/hpc-ops-gqla.*)
            rm -rf -- "$work_dir"
            work_dir=
            ;;
        *)
            printf 'Refusing to clean unexpected work directory: %s\n' "$work_dir" >&2
            exit 1
            ;;
    esac
fi

printf 'HPC_OPS_GQLA_H20_OK\n'
printf 'Standalone table: %s\n' "$output_root/hpc_ops_gqla_h20.md"
printf 'CSV: %s\n' "$output_root/hpc_ops_gqla_h20.csv"
printf 'JSON: %s\n' "$output_root/hpc_ops_gqla_h20.json"
printf 'Full run archive: %s\n' "$run_dir"
