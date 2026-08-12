#!/usr/bin/env bash

set -Eeuo pipefail

script_path=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")
script_dir=$(dirname -- "$script_path")
repo_root=$(cd -- "$script_dir/.." && pwd -P)
session_name=${GQLA_TABLE2_ENV_TMUX_SESSION:-h20-cu128-env}
output_dir=${GQLA_TABLE2_OUTPUT_DIR:-"$repo_root/outputs/official"}
install_log="$output_dir/install_cu128_env.log"
status_file="$output_dir/install_cu128_env.status"

if [[ ! "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf 'Invalid tmux session name: %s\n' "$session_name" >&2
    exit 2
fi

if [[ "${1:-}" == --worker ]]; then
    mkdir -p "$output_dir"
    exec > >(tee -a "$install_log") 2>&1

    worker_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf 'status=running started_at=%s pid=%s\n' \
        "$worker_started_at" "$$" > "$status_file"
    printf '\n===== H20 cu128 environment install start =====\n'
    printf 'started_at=%s\nrepo=%s\nlog=%s\n' \
        "$worker_started_at" "$repo_root" "$install_log"

    set +e
    bash "$script_dir/install_cu128_env.sh"
    worker_rc=$?
    set -e

    worker_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if (( worker_rc == 0 )); then
        worker_status=complete
    else
        worker_status=failed
    fi
    printf 'status=%s rc=%s started_at=%s finished_at=%s\n' \
        "$worker_status" "$worker_rc" "$worker_started_at" \
        "$worker_finished_at" > "$status_file"
    printf 'finished_at=%s status=%s rc=%s\n' \
        "$worker_finished_at" "$worker_status" "$worker_rc"
    printf '===== H20 cu128 environment install end =====\n'
    exit "$worker_rc"
fi

if ! command -v tmux >/dev/null 2>&1; then
    printf 'tmux is unavailable on this node\n' >&2
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
    printf 'conda is unavailable in the current PATH\n' >&2
    exit 2
fi
if tmux has-session -t "=$session_name" 2>/dev/null; then
    printf 'The environment installer is already running.\n'
    printf 'session=%s\nlog=%s\n' "$session_name" "$install_log"
    printf 'Attach: tmux attach -t %q\n' "$session_name"
    exit 0
fi

mkdir -p "$output_dir"
printf -v worker_command \
    'env PATH=%q HOME=%q CUDA_HOME=%q GQLA_TABLE2_CONDA_ENV=%q GQLA_TABLE2_OUTPUT_DIR=%q bash %q --worker' \
    "$PATH" "$HOME" "${CUDA_HOME:-/usr/local/cuda}" \
    "${GQLA_TABLE2_CONDA_ENV:-h20table2}" "$output_dir" "$script_path"

tmux new-session -d \
    -s "$session_name" \
    -c "$repo_root" \
    "$worker_command"

printf 'TMUX_ENV_INSTALL_STARTED\n'
printf 'session=%s\nlog=%s\nstatus_file=%s\n' \
    "$session_name" "$install_log" "$status_file"
printf 'Attach: tmux attach -t %q\n' "$session_name"
printf 'Detach without stopping: Ctrl-b then d\n'
printf 'Follow log: tail -f %q\n' "$install_log"
