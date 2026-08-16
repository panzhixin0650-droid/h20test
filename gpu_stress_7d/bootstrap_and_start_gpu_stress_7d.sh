#!/usr/bin/env bash

# One command for a fresh H20 node: reuse a CUDA PyTorch environment when one
# exists, otherwise prepare a minimal persistent environment with aria2 + uv,
# then run the seven-day stress test in the same tmux session.

set -Eeuo pipefail

script_path=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")
script_dir=$(dirname -- "$script_path")
repo_root=$(cd -- "$script_dir/.." && pwd -P)

gpu_selection=${GPU_IDS:-all}
cache_root=${GPU_STRESS_CACHE_ROOT:-}
session_name=${SESSION_NAME:-gpu-stress-7d}
foreground=0
worker=0

torch_wheel_name='torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl'
torch_wheel_url=${GPU_STRESS_TORCH_WHEEL_URL:-https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl}
torch_wheel_min_bytes=800000000
pypi_index=${GPU_STRESS_PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}
pypi_fallback=${GPU_STRESS_PYPI_FALLBACK_INDEX:-https://pypi.org/simple}
uv_version=${GPU_STRESS_UV_VERSION:-0.9.3}
aria2_file_connections=${GPU_STRESS_ARIA2_FILE_CONNECTIONS:-8}
aria2_parallel_files=${GPU_STRESS_ARIA2_PARALLEL_FILES:-8}

usage() {
    cat <<'EOF'
Usage:
  ./bootstrap_and_start_gpu_stress_7d.sh [GPU_SELECTION] [options]

Examples:
  ./bootstrap_and_start_gpu_stress_7d.sh all
  ./bootstrap_and_start_gpu_stress_7d.sh 0,1,2,3
  ./bootstrap_and_start_gpu_stress_7d.sh all --cache-root /persistent/path

Options:
  --gpus LIST          all or comma-separated physical GPU indexes
  --cache-root PATH    Persistent environment/wheel/cache root
  --session NAME       tmux session name (default: gpu-stress-7d)
  --foreground         Install and run without creating tmux
  -h, --help           Show this help

The default duration is seven days. DURATION_SECONDS, MAX_TEMP, RESUME_TEMP,
CHECK_INTERVAL, MATRIX_SIZE, and MEMORY_FRACTION are passed to the stress test.
EOF
}

positional_gpu_seen=0
while (($#)); do
    case "$1" in
        --gpus)
            gpu_selection=${2:?--gpus requires a value}
            positional_gpu_seen=1
            shift 2
            ;;
        --cache-root)
            cache_root=${2:?--cache-root requires a value}
            shift 2
            ;;
        --session)
            session_name=${2:?--session requires a value}
            shift 2
            ;;
        --foreground)
            foreground=1
            shift
            ;;
        --worker)
            worker=1
            foreground=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if (( positional_gpu_seen == 1 )); then
                printf 'Unexpected positional argument: %s\n' "$1" >&2
                exit 2
            fi
            gpu_selection=$1
            positional_gpu_seen=1
            shift
            ;;
    esac
done

if [[ "$gpu_selection" != all && ! "$gpu_selection" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'GPU selection must be all or a list such as 0,1,2,3; got %q\n' \
        "$gpu_selection" >&2
    exit 2
fi
if [[ ! "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf 'Unsafe tmux session name: %q\n' "$session_name" >&2
    exit 2
fi
for positive_integer in "$aria2_file_connections" "$aria2_parallel_files"; do
    if [[ ! "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
        printf 'aria2 concurrency values must be positive integers, got %q\n' \
            "$positive_integer" >&2
        exit 2
    fi
done
if (( aria2_file_connections > 16 )); then
    printf 'GPU_STRESS_ARIA2_FILE_CONNECTIONS cannot exceed aria2 limit 16.\n' >&2
    exit 2
fi

detect_persistent_data_mount() {
    local host task_id record_id target fstype candidate
    command -v findmnt >/dev/null 2>&1 || return 1
    host=$(hostname)
    task_id=
    record_id=
    if [[ "$host" =~ ^qs-([0-9]+)-([0-9]+)- ]]; then
        task_id=${BASH_REMATCH[1]}
        record_id=${BASH_REMATCH[2]}
    fi

    if [[ -n "$task_id" ]]; then
        while read -r target fstype; do
            if [[ "$target" == */task/"$task_id"/record/"$record_id"/data && \
                  "$fstype" != overlay ]]; then
                printf '%s\n' "$target"
                return 0
            fi
        done < <(findmnt -rn -o TARGET,FSTYPE)

        while read -r target fstype; do
            if [[ "$target" == */task/"$task_id" && "$fstype" != overlay ]]; then
                candidate="$target/record/$record_id/data"
                if [[ -d "$candidate" ]]; then
                    printf '%s\n' "$candidate"
                    return 0
                fi
            fi
        done < <(findmnt -rn -o TARGET,FSTYPE)
    fi

    while read -r target fstype; do
        if [[ "$target" =~ /task/[0-9]+/record/[0-9]+/data$ && \
              "$fstype" != overlay ]]; then
            printf '%s\n' "$target"
            return 0
        fi
    done < <(findmnt -rn -o TARGET,FSTYPE)
    return 1
}

if (( worker == 0 && foreground == 0 )); then
    if ! command -v tmux >/dev/null 2>&1; then
        printf '%s\n' 'tmux is unavailable; use --foreground or install tmux.' >&2
        exit 2
    fi
    if tmux has-session -t "$session_name" 2>/dev/null; then
        printf 'tmux session %q already exists.\n' "$session_name"
        printf 'Attach: tmux attach -t %q\n' "$session_name"
        exit 0
    fi

    mkdir -p -- "$script_dir/logs"
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    bootstrap_log="$script_dir/logs/bootstrap_${timestamp}.log"
    worker_args=(
        --worker
        --gpus "$gpu_selection"
        --session "$session_name"
    )
    if [[ -n "$cache_root" ]]; then
        worker_args+=(--cache-root "$cache_root")
    fi
    printf -v worker_command '%q ' bash "$script_path" "${worker_args[@]}"
    printf -v pane_body \
        'set -o pipefail; %s 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' \
        "$worker_command" "$bootstrap_log"
    printf -v tmux_command 'exec bash -c %q' "$pane_body"

    tmux new-session -d \
        -s "$session_name" \
        -c "$script_dir" \
        -e "PATH=$PATH" \
        -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
        -e "CONDA_PREFIX=${CONDA_PREFIX:-}" \
        -e "DURATION_SECONDS=${DURATION_SECONDS:-604800}" \
        -e "MAX_TEMP=${MAX_TEMP:-83}" \
        -e "RESUME_TEMP=${RESUME_TEMP:-78}" \
        -e "CHECK_INTERVAL=${CHECK_INTERVAL:-10}" \
        -e "MATRIX_SIZE=${MATRIX_SIZE:-0}" \
        -e "MEMORY_FRACTION=${MEMORY_FRACTION:-0.45}" \
        "$tmux_command"

    sleep 3
    if ! tmux has-session -t "$session_name" 2>/dev/null; then
        printf 'Bootstrap failed during startup. Log: %s\n' "$bootstrap_log" >&2
        tail -n 100 -- "$bootstrap_log" >&2 || true
        exit 1
    fi
    printf 'GPU stress bootstrap started in tmux.\n'
    printf 'Session: %s\n' "$session_name"
    printf 'GPUs: %s\n' "$gpu_selection"
    printf 'Bootstrap log: %s\n' "$bootstrap_log"
    printf 'Attach: tmux attach -t %q\n' "$session_name"
    printf 'Stop: %s/stop_gpu_stress_7d.sh %q\n' "$script_dir" "$session_name"
    exit 0
fi

if [[ -z "$cache_root" ]]; then
    persistent_mount=$(detect_persistent_data_mount || true)
    if [[ -n "$persistent_mount" ]]; then
        # Share the large Torch wheel and package caches with the repository's
        # FlashInfer H20 one-click launcher.
        cache_root="$persistent_mount/panzhixin/GQLA"
    else
        cache_root="$script_dir/.runtime"
        printf 'Persistent task mount not detected; using repository-local cache: %s\n' \
            "$cache_root"
    fi
fi
mkdir -p -- "$cache_root"
cache_root=$(cd -- "$cache_root" && pwd -P)

env_prefix="$cache_root/conda-envs/gpu-stress-cu128"
wheel_dir="$cache_root/wheels/cu128"
wheel_path="$wheel_dir/$torch_wheel_name"
conda_packages="$cache_root/conda-pkgs"
pip_cache="$cache_root/pip-cache"
uv_cache="$cache_root/uv-cache"
wheelhouse="$cache_root/wheelhouse/cu128-py312"
manifest_dir="$cache_root/manifests"
dependency_report="$manifest_dir/torch-cu128-py312-report.json"
aria2_manifest="$manifest_dir/torch-cu128-py312-aria2.txt"
manifest_builder="$script_dir/build_aria2_wheel_manifest.py"
mkdir -p -- \
    "$wheel_dir" "$conda_packages" "$pip_cache" "$uv_cache" \
    "$wheelhouse" "$manifest_dir"

export CONDA_PKGS_DIRS="$conda_packages"
export PIP_CACHE_DIR="$pip_cache"
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL

printf '\n===== GPU stress one-click bootstrap: %s =====\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Repository: %s\n' "$repo_root"
printf 'Cache root: %s\n' "$cache_root"
printf 'GPU selection: %s\n' "$gpu_selection"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'nvidia-smi is required but unavailable.' >&2
    exit 2
fi
if [[ "$gpu_selection" == all ]]; then
    first_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | \
        sed -n '1{s/[[:space:]]//g;p;}')
else
    first_gpu=${gpu_selection%%,*}
fi
if [[ -z "$first_gpu" ]]; then
    printf '%s\n' 'No GPU was detected.' >&2
    exit 2
fi
nvidia-smi -L

probe_cuda_python() {
    local candidate_python=$1
    local probe_code
    probe_code='import torch; assert torch.cuda.is_available(); print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()} device={torch.cuda.get_device_name(0)}")'
    timeout 30 env \
        PYTHONDONTWRITEBYTECODE=1 \
        CUDA_VISIBLE_DEVICES="$first_gpu" \
        "$candidate_python" -c "$probe_code"
}

if [[ ! -f "$manifest_builder" ]]; then
    printf 'aria2 manifest builder is absent: %s\n' "$manifest_builder" >&2
    exit 2
fi

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

add_python_candidate "${PYTHON_BIN:-}"
add_python_candidate "$env_prefix/bin/python"
add_python_candidate "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
add_python_candidate python
add_python_candidate python3
if command -v conda >/dev/null 2>&1; then
    while read -r conda_prefix; do
        add_python_candidate "$conda_prefix/bin/python"
    done < <(conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $NF}')
fi

python_bin=
python_probe_output=
for candidate_python in "${python_candidates[@]:-}"; do
    if candidate_output=$(probe_cuda_python "$candidate_python" 2>/dev/null); then
        python_bin=$candidate_python
        python_probe_output=$candidate_output
        break
    fi
done

validator_python=$(command -v python3 || command -v python)
wheel_is_complete() {
    [[ -s "$wheel_path" && ! -e "$wheel_path.aria2" ]] || return 1
    [[ $(stat -c %s "$wheel_path") -ge "$torch_wheel_min_bytes" ]] || return 1
    "$validator_python" -c \
        'import sys,zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' \
        "$wheel_path"
}

install_aria2_if_possible() {
    command -v aria2c >/dev/null 2>&1 && return 0
    [[ $(id -u) == 0 ]] || return 1
    command -v apt-get >/dev/null 2>&1 || return 1
    command -v timeout >/dev/null 2>&1 || return 1
    local -a apt_options
    apt_options=(
        -o Acquire::Retries=2
        -o Acquire::http::Timeout=15
        -o Acquire::https::Timeout=15
        -o DPkg::Lock::Timeout=30
    )
    printf 'Installing aria2 for up to %s connections per wheel...\n' \
        "$aria2_file_connections"

    # Most cluster images already have usable apt indexes. Try this fast path
    # before an update, which can hang silently on an unreachable mirror.
    if timeout 90 env DEBIAN_FRONTEND=noninteractive \
        apt-get "${apt_options[@]}" install -y aria2; then
        command -v aria2c >/dev/null 2>&1
        return
    fi

    printf '%s\n' \
        'Direct apt install failed; refreshing package indexes with a 120s limit...'
    timeout 120 env DEBIAN_FRONTEND=noninteractive \
        apt-get "${apt_options[@]}" update || return 1
    timeout 90 env DEBIAN_FRONTEND=noninteractive \
        apt-get "${apt_options[@]}" install -y aria2 || return 1
    command -v aria2c >/dev/null 2>&1
}

download_torch_wheel() {
    if wheel_is_complete; then
        printf 'Reusing complete Torch wheel: %s\n' "$wheel_path"
        return 0
    fi
    if [[ -f "$wheel_path" && ! -e "$wheel_path.aria2" ]]; then
        prior_size=$(stat -c %s "$wheel_path")
        if [[ "$prior_size" -ge "$torch_wheel_min_bytes" ]]; then
            invalid_path="$wheel_path.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
            printf 'Preserving invalid prior wheel as: %s\n' "$invalid_path"
            mv -- "$wheel_path" "$invalid_path"
        else
            printf 'Resuming partial Torch wheel (%s bytes).\n' "$prior_size"
        fi
    fi

    printf 'Downloading Torch wheel: %s\n' "$wheel_path"
    if ! command -v aria2c >/dev/null 2>&1; then
        printf '%s\n' 'aria2 is required for the fast bootstrap but is unavailable.' >&2
        return 2
    fi
    aria2c \
        --continue=true \
        --max-connection-per-server="$aria2_file_connections" \
        --split="$aria2_file_connections" \
        --min-split-size=4M \
        --file-allocation=none \
        --auto-file-renaming=false \
        --connect-timeout=15 \
        --timeout=30 \
        --max-tries=0 \
        --retry-wait=2 \
        --summary-interval=5 \
        --dir="$wheel_dir" \
        --out="$torch_wheel_name" \
        "$torch_wheel_url"
    wheel_is_complete
    printf 'TORCH_WHEEL_OK path=%s size_bytes=%s\n' \
        "$wheel_path" "$(stat -c %s "$wheel_path")"
}

download_pid=
cleanup_download() {
    if [[ -n "$download_pid" ]] && kill -0 "$download_pid" 2>/dev/null; then
        kill -TERM "$download_pid" 2>/dev/null || true
        wait "$download_pid" 2>/dev/null || true
    fi
}
trap cleanup_download EXIT INT TERM

if [[ -z "$python_bin" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        printf '%s\n' 'No compatible PyTorch environment exists and conda is unavailable.' >&2
        exit 2
    fi
    if ! install_aria2_if_possible; then
        printf '%s\n' \
            'Cannot install aria2 automatically; install aria2c and rerun.' >&2
        exit 2
    fi
    download_torch_wheel &
    download_pid=$!

    if [[ ! -x "$env_prefix/bin/python" ]]; then
        printf 'Creating minimal Python 3.12 environment: %s\n' "$env_prefix"
        conda create -p "$env_prefix" python=3.12 pip -y
    else
        printf 'Reusing environment prefix: %s\n' "$env_prefix"
    fi
    python_bin="$env_prefix/bin/python"

    if [[ -x "$env_prefix/bin/uv" ]]; then
        uv_bin="$env_prefix/bin/uv"
    elif command -v uv >/dev/null 2>&1; then
        uv_bin=$(command -v uv)
    else
        printf 'Installing uv %s from the primary mirror...\n' "$uv_version"
        if ! "$python_bin" -m pip install \
            --index-url "$pypi_index" \
            --timeout 120 \
            --retries 10 \
            "uv==$uv_version"; then
            "$python_bin" -m pip install \
                --index-url "$pypi_fallback" \
                --timeout 120 \
                --retries 10 \
                "uv==$uv_version"
        fi
        uv_bin="$env_prefix/bin/uv"
    fi
    if [[ ! -x "$uv_bin" ]]; then
        printf 'uv executable is unavailable after installation: %s\n' "$uv_bin" >&2
        exit 2
    fi

    if ! wait "$download_pid"; then
        download_pid=
        printf '%s\n' 'Torch wheel download failed.' >&2
        exit 2
    fi
    download_pid=

    printf '%s\n' 'Resolving the complete Torch wheel set (metadata only)...'
    "$python_bin" -m pip install \
        --dry-run \
        --ignore-installed \
        --report "$dependency_report" \
        --index-url "$pypi_index" \
        --extra-index-url "$pypi_fallback" \
        --only-binary :all: \
        'setuptools<82' \
        "$wheel_path"
    "$python_bin" "$manifest_builder" \
        --report "$dependency_report" \
        --destination "$wheelhouse" \
        --output "$aria2_manifest"

    printf '%s\n' \
        'Downloading every resolved dependency wheel with aria2...'
    aria2c \
        --input-file="$aria2_manifest" \
        --continue=true \
        --max-concurrent-downloads="$aria2_parallel_files" \
        --max-connection-per-server="$aria2_file_connections" \
        --split="$aria2_file_connections" \
        --min-split-size=4M \
        --file-allocation=none \
        --auto-file-renaming=false \
        --allow-overwrite=false \
        --check-integrity=true \
        --connect-timeout=15 \
        --timeout=30 \
        --max-tries=0 \
        --retry-wait=2 \
        --summary-interval=5

    printf '%s\n' 'Installing Torch completely offline from the aria2 wheelhouse...'
    env \
        -u PIP_INDEX_URL \
        -u PIP_EXTRA_INDEX_URL \
        UV_CACHE_DIR="$uv_cache" \
        UV_LINK_MODE=copy \
        UV_CONCURRENT_DOWNLOADS=${GPU_STRESS_UV_CONCURRENT_DOWNLOADS:-16} \
        "$uv_bin" pip install \
            --python "$python_bin" \
            --no-index \
            --find-links "$wheelhouse" \
            --find-links "$wheel_dir" \
            --only-binary :all: \
            --link-mode copy \
            --strict \
            'setuptools<82' \
            "$wheel_path"
    "$python_bin" -m pip check
    python_probe_output=$(probe_cuda_python "$python_bin")
fi

trap - EXIT INT TERM
printf 'CUDA PyTorch ready: %s\n' "$python_bin"
printf 'Environment probe: %s\n' "$python_probe_output"
printf '%s\n' 'Starting the seven-day GPU stress workers in this tmux session...'

export PYTHON_BIN="$python_bin"
export GPU_IDS="$gpu_selection"
exec bash "$script_dir/run_gpu_stress_7d.sh"
