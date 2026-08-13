#!/usr/bin/env bash

# Fresh-node H20 launcher for the FlashInfer GQA benchmark. It discovers the
# current task's persistent data mount, keeps downloads and the conda prefix on
# that mount, resumes the large Torch wheel, and runs the pinned benchmark in a
# tmux session by default.

set -Eeuo pipefail

script_path=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")
repo_root=$(dirname -- "$script_path")

gpu_index=${FLASHINFER_GQA_GPU:-0}
persistent_root=${FLASHINFER_GQA_PERSIST_ROOT:-}
session_name=${FLASHINFER_GQA_ONECLICK_SESSION:-flashinfer-h20}
foreground=0
quick=0
worker=0

usage() {
    cat <<'EOF'
Usage:
  bash run_flashinfer_gqa_h20_oneclick.sh [options]

Options:
  --gpu INDEX               Physical GPU index (default: 0)
  --persistent-root PATH    Persistent GQLA root; normally auto-detected
  --session NAME            tmux session name (default: flashinfer-h20)
  --quick                   Short timing smoke run with canonical shapes
  --foreground              Do not create/attach a tmux session
  -h, --help                Show this help

The default fresh-node invocation needs no arguments. It auto-detects mounts
such as /mnt/.../task/<task>/record/<record>/data, persists the Torch wheel,
pip/conda caches, environment, logs, and result bundles, and resumes downloads
after interruption. Override detection with FLASHINFER_GQA_PERSIST_ROOT or
--persistent-root only when necessary.
EOF
}

while (($#)); do
    case "$1" in
        --gpu)
            gpu_index=${2:?--gpu requires a value}
            shift 2
            ;;
        --persistent-root)
            persistent_root=${2:?--persistent-root requires a value}
            shift 2
            ;;
        --session)
            session_name=${2:?--session requires a value}
            shift 2
            ;;
        --quick)
            quick=1
            shift
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
        *)
            printf 'Unknown argument: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
    printf 'GPU index must be a non-negative integer, got %q\n' \
        "$gpu_index" >&2
    exit 2
fi
if [[ ! "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf 'Unsafe tmux session name: %q\n' "$session_name" >&2
    exit 2
fi
if [[ ! -f "$repo_root/run_flashinfer_gqa.sh" ]]; then
    printf 'Supporting benchmark entry is absent: %s/run_flashinfer_gqa.sh\n' \
        "$repo_root" >&2
    printf '%s\n' 'Clone the complete h20test repository, then run this script.' >&2
    exit 2
fi

detect_persistent_data_mount() {
    local host task_id record_id target fstype candidate
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

        # Some images expose only the task root as a mount even though the
        # record data directory below it is the intended persistent location.
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

if [[ -z "$persistent_root" ]]; then
    persistent_data_mount=$(detect_persistent_data_mount || true)
    if [[ -z "$persistent_data_mount" ]]; then
        printf '%s\n' \
            'Cannot auto-detect a persistent .../task/.../record/.../data mount.' >&2
        printf '%s\n' \
            'Pass --persistent-root PATH after checking it with findmnt -T PATH.' >&2
        exit 2
    fi
    persistent_root="$persistent_data_mount/panzhixin/GQLA"
fi

mkdir -p "$persistent_root"
persistent_fstype=$(findmnt -rn -T "$persistent_root" -o FSTYPE | head -n 1 || true)
persistent_target=$(findmnt -rn -T "$persistent_root" -o TARGET | head -n 1 || true)
if [[ -z "$persistent_fstype" || "$persistent_fstype" == overlay ]]; then
    printf 'Refusing non-persistent root %s (target=%s fstype=%s).\n' \
        "$persistent_root" "${persistent_target:-unknown}" \
        "${persistent_fstype:-unknown}" >&2
    exit 2
fi

printf 'Persistent root: %s\n' "$persistent_root"
printf 'Persistent mount: target=%s fstype=%s\n' \
    "$persistent_target" "$persistent_fstype"

start_tmux_worker() {
    local -a worker_args
    local worker_command

    if tmux has-session -t "$session_name" 2>/dev/null; then
        printf 'Attaching existing tmux session: %s\n' "$session_name"
        exec tmux attach-session -t "$session_name"
    fi

    worker_args=(
        --worker
        --persistent-root "$persistent_root"
        --gpu "$gpu_index"
        --session "$session_name"
    )
    if [[ "$quick" == 1 ]]; then
        worker_args+=(--quick)
    fi

    printf -v worker_command '%q ' bash "$script_path" "${worker_args[@]}"
    tmux new-session -d -s "$session_name" -c "$repo_root"
    tmux send-keys -t "$session_name:0.0" -l "$worker_command"
    tmux send-keys -t "$session_name:0.0" Enter
    printf 'Started tmux session: %s\n' "$session_name"
    printf '%s\n' 'Detach with Ctrl-b then d; reattach with:'
    printf '  tmux attach -t %q\n' "$session_name"
    exec tmux attach-session -t "$session_name"
}

if [[ "$worker" == 0 && "$foreground" == 0 && -z "${TMUX:-}" ]]; then
    if command -v tmux >/dev/null 2>&1; then
        start_tmux_worker
    fi
    printf '%s\n' 'tmux is unavailable; continuing in the foreground.' >&2
fi

output_base="$persistent_root/outputs/flashinfer_gqa"
wheel_dir="$persistent_root/wheels/cu128"
wheel_name='torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl'
wheel_path="$wheel_dir/$wheel_name"
wheel_url=${FLASHINFER_GQA_TORCH_WHEEL_URL:-https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl}
wheel_min_bytes=800000000
log_file="$output_base/flashinfer_h20_oneclick.log"
status_file="$output_base/flashinfer_h20_oneclick.status"

mkdir -p \
    "$persistent_root/conda-envs" \
    "$persistent_root/conda-pkgs" \
    "$persistent_root/pip-cache" \
    "$output_base/runs" \
    "$wheel_dir"

exec > >(tee -a "$log_file") 2>&1

on_error() {
    local rc=$1 line=$2
    trap - ERR
    printf 'FLASHINFER_H20_ONECLICK_FAILED rc=%s line=%s\n' "$rc" "$line"
    printf 'FAILED rc=%s line=%s log=%s\n' \
        "$rc" "$line" "$log_file" >"$status_file"
    exit "$rc"
}
trap 'on_error "$?" "$LINENO"' ERR

printf '\n===== FlashInfer H20 one-click start: %s =====\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Repository: %s\n' "$repo_root"
printf 'Persistent root: %s\n' "$persistent_root"
printf 'GPU index: %s\n' "$gpu_index"
printf 'RUNNING log=%s\n' "$log_file" >"$status_file"

export CONDA_ENVS_PATH="$persistent_root/conda-envs"
export CONDA_PKGS_DIRS="$persistent_root/conda-pkgs"
export PIP_CACHE_DIR="$persistent_root/pip-cache"
export PIP_FIND_LINKS="$wheel_dir"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export FLASHINFER_GQA_CONDA_ENV=${FLASHINFER_GQA_CONDA_ENV:-flashinfer-gqa-cu128}
export GQLA_TABLE2_TORCH_WHEEL="$wheel_path"

if ! command -v conda >/dev/null 2>&1; then
    printf '%s\n' 'conda is required but unavailable.' >&2
    exit 2
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf 'nvcc is absent: %s/bin/nvcc\n' "$CUDA_HOME" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'nvidia-smi is required but unavailable.' >&2
    exit 2
fi

validator_python=$(command -v python3 || command -v python)
wheel_is_complete() {
    [[ -s "$wheel_path" && ! -e "$wheel_path.aria2" ]] || return 1
    [[ $(stat -c %s "$wheel_path") -ge "$wheel_min_bytes" ]] || return 1
    "$validator_python" -c \
        'import sys, zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' \
        "$wheel_path"
}

install_aria2_if_possible() {
    command -v aria2c >/dev/null 2>&1 && return 0
    [[ $(id -u) == 0 ]] || return 1
    command -v apt-get >/dev/null 2>&1 || return 1
    printf '%s\n' 'Installing aria2 for resumable multi-connection download...'
    DEBIAN_FRONTEND=noninteractive apt-get update -qq || return 1
    DEBIAN_FRONTEND=noninteractive apt-get install -y aria2 || return 1
    command -v aria2c >/dev/null 2>&1
}

if wheel_is_complete; then
    printf 'Reusing complete persistent Torch wheel: %s\n' "$wheel_path"
else
    if [[ -f "$wheel_path" && ! -e "$wheel_path.aria2" ]]; then
        prior_size=$(stat -c %s "$wheel_path")
        if [[ "$prior_size" -ge "$wheel_min_bytes" ]]; then
            corrupt_path="$wheel_path.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
            printf 'Moving invalid prior wheel to: %s\n' "$corrupt_path"
            mv -- "$wheel_path" "$corrupt_path"
        else
            printf 'Resuming partial Torch wheel: %s bytes already present.\n' \
                "$prior_size"
        fi
    fi

    printf 'Downloading Torch wheel to persistent storage: %s\n' "$wheel_path"
    if install_aria2_if_possible; then
        aria2c \
            --continue=true \
            --max-connection-per-server=16 \
            --split=16 \
            --min-split-size=4M \
            --file-allocation=none \
            --auto-file-renaming=false \
            --connect-timeout=15 \
            --timeout=30 \
            --max-tries=0 \
            --retry-wait=2 \
            --summary-interval=5 \
            --dir="$wheel_dir" \
            --out="$wheel_name" \
            "$wheel_url"
    else
        printf '%s\n' 'aria2 is unavailable; falling back to resumable IPv4 curl.'
        curl -4 \
            --fail \
            --location \
            --continue-at - \
            --connect-timeout 20 \
            --speed-limit 1024 \
            --speed-time 60 \
            --retry 30 \
            --retry-delay 2 \
            --output "$wheel_path" \
            "$wheel_url"
    fi
fi

if ! wheel_is_complete; then
    printf 'Torch wheel is incomplete or invalid: %s\n' "$wheel_path" >&2
    exit 2
fi
printf 'TORCH_WHEEL_OK path=%s size_bytes=%s\n' \
    "$wheel_path" "$(stat -c %s "$wheel_path")"

printf '%s\n' '===== GPU and CUDA preflight ====='
nvidia-smi -L
"$CUDA_HOME/bin/nvcc" --version

benchmark_args=(
    --profile h20
    --gpu "$gpu_index"
    --bootstrap-cu128
    --output-root "$output_base/runs"
)
if [[ "$quick" == 1 ]]; then
    benchmark_args+=(--quick)
fi

bash "$repo_root/run_flashinfer_gqa.sh" "${benchmark_args[@]}"

printf 'OK log=%s output_root=%s\n' \
    "$log_file" "$output_base/runs" >"$status_file"
printf 'FLASHINFER_H20_ONECLICK_OK log=%s output_root=%s\n' \
    "$log_file" "$output_base/runs"
printf '===== FlashInfer H20 one-click end: %s =====\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
