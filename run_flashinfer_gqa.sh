#!/usr/bin/env bash

# One-command, repository-portable FlashInfer benchmark for the GQLA H20 GQA
# anomaly.  It fixes the canonical workload, validates the environment and
# result, and emits a self-contained result bundle.

set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
benchmark_py="$repo_root/scripts/benchmark_flashinfer_gqa.py"
verify_py="$repo_root/scripts/verify_flashinfer_gqa.py"
flashinfer_version=0.6.11.post2
torch_version=2.11.0
triton_version=3.6.0

usage() {
    cat <<'EOF'
Usage:
  bash run_flashinfer_gqa.sh [options]

Canonical run (auto-detect H20/H100/L20Z):
  bash run_flashinfer_gqa.sh

Fresh H20 node, including the pinned CUDA 12.8 environment bootstrap:
  bash run_flashinfer_gqa.sh --profile h20 --bootstrap-cu128

Existing PyTorch environment with FlashInfer absent:
  bash run_flashinfer_gqa.sh --profile h20 --python "$(command -v python)" \
    --install-missing

Options:
  --profile NAME       auto, h20, h100, or l20z (default: auto)
  --python PATH        Python executable to use
  --gpu INDEX          Physical GPU index for CUDA_VISIBLE_DEVICES (default: 0)
  --output-root PATH   Parent directory for run directories
  --label TEXT         Result label; defaults to the resolved profile
  --quick              Short timing smoke run; shape/correctness stay canonical
  --no-kernel-profile  Skip the separate Kineto kernel-name capture
  --no-bundle          Do not create the final tar.gz result bundle
  --allow-busy-gpu     Run even if nvidia-smi reports another compute process
  --install-missing    Install pinned FlashInfer into the selected Python env
  --bootstrap-cu128    Build/reuse the repo's complete pinned H20 conda env
  --hbm-tb-s VALUE     Override the profile's sustained HBM reference
  --bf16-tflops VALUE  Override the profile's sustained BF16 reference
  -h, --help           Show this help

Environment equivalents:
  FLASHINFER_GQA_PYTHON, FLASHINFER_GQA_GPU, FLASHINFER_GQA_OUTPUT_ROOT,
  FLASHINFER_GQA_PYPI_INDEX, FLASHINFER_GQA_MAX_JOBS,
  FLASHINFER_GQA_NVCC_THREADS
EOF
}

profile=auto
python_override=${FLASHINFER_GQA_PYTHON:-}
gpu_index=${FLASHINFER_GQA_GPU:-0}
output_root=${FLASHINFER_GQA_OUTPUT_ROOT:-}
label=
quick=0
kernel_profile=1
make_bundle=1
allow_busy_gpu=0
install_missing=0
bootstrap_cu128=0
hbm_override=
bf16_override=

while (($#)); do
    case "$1" in
        --profile)
            profile=${2:?--profile requires a value}
            shift 2
            ;;
        --python)
            python_override=${2:?--python requires a value}
            shift 2
            ;;
        --gpu)
            gpu_index=${2:?--gpu requires a value}
            shift 2
            ;;
        --output-root)
            output_root=${2:?--output-root requires a value}
            shift 2
            ;;
        --label)
            label=${2:?--label requires a value}
            shift 2
            ;;
        --quick)
            quick=1
            shift
            ;;
        --no-kernel-profile)
            kernel_profile=0
            shift
            ;;
        --no-bundle)
            make_bundle=0
            shift
            ;;
        --allow-busy-gpu)
            allow_busy_gpu=1
            shift
            ;;
        --install-missing)
            install_missing=1
            shift
            ;;
        --bootstrap-cu128)
            bootstrap_cu128=1
            shift
            ;;
        --hbm-tb-s)
            hbm_override=${2:?--hbm-tb-s requires a value}
            shift 2
            ;;
        --bf16-tflops)
            bf16_override=${2:?--bf16-tflops requires a value}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$profile" in
    auto|h20|h100|l20z) ;;
    *)
        printf 'Invalid --profile %q; expected auto, h20, h100, or l20z\n' \
            "$profile" >&2
        exit 2
        ;;
esac
if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
    printf -- '--gpu must be a non-negative integer, got %q\n' "$gpu_index" >&2
    exit 2
fi
for value_name in hbm_override bf16_override; do
    value=${!value_name}
    if [[ -n "$value" ]] && ! awk -v value="$value" \
        'BEGIN {exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0)}'; then
        printf '%s must be a positive number, got %q\n' "$value_name" "$value" >&2
        exit 2
    fi
done

if [[ "$bootstrap_cu128" == 1 ]]; then
    if [[ "$profile" == h100 || "$profile" == l20z ]]; then
        printf '%s\n' '--bootstrap-cu128 is only valid for an H20/auto profile.' >&2
        exit 2
    fi
    if ! command -v conda >/dev/null 2>&1; then
        printf '%s\n' '--bootstrap-cu128 requires conda on the H20 node.' >&2
        exit 2
    fi
    printf '%s\n' 'Bootstrapping the pinned H20 CUDA 12.8 environment...'
    bash "$repo_root/scripts/install_cu128_env.sh"
    conda_env_name=${GQLA_TABLE2_CONDA_ENV:-h20table2}
    conda_env_prefix=$(conda env list | \
        awk -v target="$conda_env_name" '$1 == target {print $NF; exit}')
    if [[ -z "$conda_env_prefix" || ! -x "$conda_env_prefix/bin/python" ]]; then
        printf 'Cannot resolve conda environment %q after bootstrap.\n' \
            "$conda_env_name" >&2
        exit 2
    fi
    python_override="$conda_env_prefix/bin/python"
fi

base_environment_ready() {
    local candidate=$1
    "$candidate" - "$torch_version" "$triton_version" <<'PY' >/dev/null 2>&1
import sys
import torch
import triton

expected_torch, expected_triton = sys.argv[1:]
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+")[0] == expected_torch, torch.__version__
assert triton.__version__ == expected_triton, triton.__version__
PY
}

complete_environment_ready() {
    local candidate=$1
    base_environment_ready "$candidate" || return 1
    "$candidate" - "$flashinfer_version" <<'PY' >/dev/null 2>&1
import sys
import flashinfer

assert flashinfer.__version__ == sys.argv[1], flashinfer.__version__
PY
}

python_candidates=()
add_python_candidate() {
    local candidate=${1:-}
    [[ -n "$candidate" ]] || return 0
    if [[ "$candidate" != */* ]]; then
        candidate=$(command -v -- "$candidate" 2>/dev/null || true)
    fi
    [[ -n "$candidate" && -x "$candidate" ]] || return 0
    local existing
    for existing in "${python_candidates[@]:-}"; do
        [[ "$existing" == "$candidate" ]] && return 0
    done
    python_candidates+=("$candidate")
}

if [[ -n "$python_override" ]]; then
    add_python_candidate "$python_override"
else
    add_python_candidate "${GQLA_TABLE2_PYTHON:-}"
    add_python_candidate "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
    add_python_candidate "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
    add_python_candidate "$repo_root/.venv/bin/python"
    add_python_candidate "$repo_root/../../envs/venv-py312/bin/python"
    add_python_candidate python
    add_python_candidate python3
fi

if ((${#python_candidates[@]} == 0)); then
    printf '%s\n' 'No usable Python executable was found.' >&2
    exit 2
fi

python_bin=
for candidate in "${python_candidates[@]}"; do
    if complete_environment_ready "$candidate"; then
        python_bin=$candidate
        break
    fi
done

if [[ -z "$python_bin" && "$install_missing" == 1 ]]; then
    for candidate in "${python_candidates[@]}"; do
        if base_environment_ready "$candidate"; then
            python_bin=$candidate
            break
        fi
    done
    if [[ -z "$python_bin" ]]; then
        printf '%s\n' \
            'No Python 3.12 + PyTorch 2.11.0 + Triton 3.6.0 environment was found.' \
            'On H20, use --bootstrap-cu128 to create the complete pinned environment.' >&2
        exit 2
    fi
    pypi_index=${FLASHINFER_GQA_PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}
    printf 'Installing FlashInfer %s into %s\n' "$flashinfer_version" "$python_bin"
    installer=()
    if "$python_bin" -m pip --version >/dev/null 2>&1; then
        installer=("$python_bin" -m pip install)
    elif [[ -x "$(dirname -- "$python_bin")/uv" ]]; then
        installer=("$(dirname -- "$python_bin")/uv" pip install --python "$python_bin")
    elif command -v uv >/dev/null 2>&1; then
        installer=("$(command -v uv)" pip install --python "$python_bin")
    else
        printf '%s\n' 'Neither pip nor uv is available to install FlashInfer.' >&2
        exit 2
    fi
    env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
        "${installer[@]}" \
        --index-url "$pypi_index" \
        "flashinfer-python==$flashinfer_version"
fi

if [[ -z "$python_bin" ]] || ! complete_environment_ready "$python_bin"; then
    printf '%s\n' \
        'A pinned benchmark environment was not found.' \
        'Required: Python 3.12, PyTorch 2.11.0, Triton 3.6.0, and' \
        "FlashInfer $flashinfer_version." \
        'Activate the intended environment, pass --python, or use --install-missing.' \
        'For a fresh H20 conda node, use --profile h20 --bootstrap-cu128.' >&2
    exit 2
fi

device_record=$(CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print("\t".join((name, str(major), str(minor), torch.__version__, str(torch.version.cuda))))
PY
)
IFS=$'\t' read -r device_name capability_major capability_minor \
    detected_torch detected_torch_cuda <<<"$device_record"
if [[ "$capability_major" != 9 || "$capability_minor" != 0 ]]; then
    printf 'This one-click protocol requires Hopper SM90; got %s.%s on %s\n' \
        "$capability_major" "$capability_minor" "$device_name" >&2
    exit 2
fi

device_name_lower=${device_name,,}
if [[ "$profile" == auto ]]; then
    case "$device_name_lower" in
        *h20*) profile=h20 ;;
        *h100*) profile=h100 ;;
        *l20z*) profile=l20z ;;
        *)
            printf 'Cannot map Hopper device %q to h20/h100/l20z; pass --profile.\n' \
                "$device_name" >&2
            exit 2
            ;;
    esac
fi

case "$profile" in
    h20)
        expected_device=H20
        default_label=h20
        hbm_tb_s=3.572
        bf16_tflops=141.48
        ;;
    h100)
        expected_device=H100
        default_label=h100
        hbm_tb_s=
        bf16_tflops=
        ;;
    l20z)
        expected_device=L20Z
        default_label=l20z-h100-role
        hbm_tb_s=3.186
        bf16_tflops=818.519
        ;;
esac
if [[ "$device_name_lower" != *"${expected_device,,}"* ]]; then
    printf 'Profile %q expects a device containing %q, got %q\n' \
        "$profile" "$expected_device" "$device_name" >&2
    exit 2
fi
hbm_tb_s=${hbm_override:-$hbm_tb_s}
bf16_tflops=${bf16_override:-$bf16_tflops}
label=${label:-$default_label}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'nvidia-smi is required for device capture and busy-GPU checks.' >&2
    exit 2
fi
compute_apps=$(nvidia-smi -i "$gpu_index" \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)
if [[ -n "$compute_apps" && "$allow_busy_gpu" == 0 ]]; then
    printf 'GPU %s is busy; refusing to produce a formal result:\n%s\n' \
        "$gpu_index" "$compute_apps" >&2
    printf '%s\n' 'Stop those processes or pass --allow-busy-gpu for a diagnostic run.' >&2
    exit 2
fi

if [[ -z "$output_root" ]]; then
    output_root="$repo_root/outputs/official/flashinfer_gqa/runs"
fi
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
hostname_short=$(hostname -s 2>/dev/null || hostname)
safe_host=$(printf '%s' "$hostname_short" | tr -cs 'A-Za-z0-9_.-' '-')
safe_label=$(printf '%s' "$label" | tr -cs 'A-Za-z0-9_.-' '-')
mode=formal
warmup_ms=100
rep_ms=1000
rounds=3
if [[ "$quick" == 1 ]]; then
    mode=quick
    warmup_ms=20
    rep_ms=150
    rounds=1
fi
run_id="${safe_label}_${safe_host}_${mode}_${timestamp}"
run_dir="$output_root/$run_id"
if [[ -e "$run_dir" ]]; then
    run_dir="${run_dir}_$$"
    run_id=$(basename -- "$run_dir")
fi
mkdir -p "$run_dir"

result_json="$run_dir/result.json"
benchmark_log="$run_dir/benchmark.log"
environment_txt="$run_dir/environment.txt"
nvidia_smi_txt="$run_dir/nvidia_smi_q.txt"
topology_txt="$run_dir/nvidia_topology.txt"
summary_txt="$run_dir/summary.txt"
verify_log="$run_dir/verify.log"

printf 'Run directory: %s\n' "$run_dir"
printf 'Python: %s\n' "$python_bin"
printf 'Device: %s\n' "$device_name"
printf 'Profile/mode: %s/%s\n' "$profile" "$mode"

nvidia-smi -i "$gpu_index" -q >"$nvidia_smi_txt"
nvidia-smi topo -m >"$topology_txt"
{
    printf 'created_at_utc=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'run_id=%s\n' "$run_id"
    printf 'profile=%s\n' "$profile"
    printf 'mode=%s\n' "$mode"
    printf 'python=%s\n' "$python_bin"
    printf 'device=%s\n' "$device_name"
    printf 'compute_capability=%s.%s\n' "$capability_major" "$capability_minor"
    printf 'torch=%s\n' "$detected_torch"
    printf 'torch_cuda=%s\n' "$detected_torch_cuda"
    printf 'flashinfer=%s\n' "$flashinfer_version"
    printf 'triton=%s\n' "$triton_version"
    printf 'cuda_visible_devices=%s\n' "$gpu_index"
    printf 'hbm_reference_tb_s=%s\n' "${hbm_tb_s:-unset}"
    printf 'bf16_reference_tflops=%s\n' "${bf16_tflops:-unset}"
    printf 'max_jobs=%s\n' "${FLASHINFER_GQA_MAX_JOBS:-8}"
    printf 'flashinfer_nvcc_threads=%s\n' "${FLASHINFER_GQA_NVCC_THREADS:-4}"
    printf 'repo_commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
    printf 'repo_status_begin\n'
    git -C "$repo_root" status --short 2>/dev/null || true
    printf 'repo_status_end\n'
    printf 'source_sha256_begin\n'
    sha256sum "$benchmark_py" "$verify_py" "${BASH_SOURCE[0]}"
    printf 'source_sha256_end\n'
    printf 'nvcc_begin\n'
    if [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
        "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc --version
    else
        printf 'nvcc unavailable\n'
    fi
    printf 'nvcc_end\n'
    printf 'python_packages_begin\n'
    "$python_bin" - <<'PY'
import importlib.metadata

packages = sorted(
    (distribution.metadata["Name"], distribution.version)
    for distribution in importlib.metadata.distributions()
)
for name, version in packages:
    print(f"{name}=={version}")
PY
    printf 'python_packages_end\n'
} >"$environment_txt"

benchmark_command=(
    "$python_bin" -u "$benchmark_py"
    --output "$result_json"
    --label "$label"
    --device 0
    --expect-device-substring "$expected_device"
    --backends fa2
    --batch-size 128
    --context-len 8192
    --g-values 8,4
    --sq-values 1,2
    --page-size-sq1 64
    --page-size-sq2 128
    --warmup-ms "$warmup_ms"
    --rep-ms "$rep_ms"
    --rounds "$rounds"
    --fail-fast
)
if [[ "$kernel_profile" == 1 ]]; then
    benchmark_command+=(--profile-kernels --profile-calls 5)
fi
if [[ -n "$hbm_tb_s" ]]; then
    benchmark_command+=(--measured-hbm-tb-s "$hbm_tb_s")
fi
if [[ -n "$bf16_tflops" ]]; then
    benchmark_command+=(--measured-bf16-tflops "$bf16_tflops")
fi

export MAX_JOBS=${FLASHINFER_GQA_MAX_JOBS:-8}
export FLASHINFER_NVCC_THREADS=${FLASHINFER_GQA_NVCC_THREADS:-4}
printf 'Starting canonical benchmark; first-use (192,128) JIT may take a while.\n'
set +e
CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONUNBUFFERED=1 \
    "${benchmark_command[@]}" 2>&1 | tee "$benchmark_log"
benchmark_status=${PIPESTATUS[0]}
set -e
if [[ "$benchmark_status" != 0 ]]; then
    printf 'Benchmark failed with status %s. Partial artifacts: %s\n' \
        "$benchmark_status" "$run_dir" >&2
    exit "$benchmark_status"
fi

verify_command=(
    "$python_bin" "$verify_py" "$result_json"
    --summary-output "$summary_txt"
    --expect-device-substring "$expected_device"
    --expect-flashinfer-version "$flashinfer_version"
)
if [[ "$kernel_profile" == 1 ]]; then
    verify_command+=(--require-kineto)
fi
set +e
"${verify_command[@]}" 2>&1 | tee "$verify_log"
verify_status=${PIPESTATUS[0]}
set -e
if [[ "$verify_status" != 0 ]]; then
    printf 'Result verification failed. Partial artifacts: %s\n' "$run_dir" >&2
    exit "$verify_status"
fi

(
    cd -- "$run_dir"
    sha256sum \
        result.json \
        benchmark.log \
        environment.txt \
        nvidia_smi_q.txt \
        nvidia_topology.txt \
        summary.txt \
        verify.log >SHA256SUMS
)

bundle_path=
if [[ "$make_bundle" == 1 ]]; then
    bundle_path="$output_root/${run_id}.tar.gz"
    tar -C "$run_dir" -czf "$bundle_path" .
    sha256sum "$bundle_path" >"$bundle_path.sha256"
fi

printf '%s\n' 'FLASHINFER_GQA_ONECLICK_OK'
printf 'RESULT_DIR=%s\n' "$run_dir"
printf 'RESULT_JSON=%s\n' "$result_json"
if [[ -n "$bundle_path" ]]; then
    printf 'RESULT_BUNDLE=%s\n' "$bundle_path"
    printf 'RESULT_BUNDLE_SHA256=%s.sha256\n' "$bundle_path"
fi
