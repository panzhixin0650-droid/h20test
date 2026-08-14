#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
session_name="${SESSION_NAME:-gpu-stress-7d}"
gpu_selection="${1:-${GPU_IDS:-0}}"
duration_seconds="${DURATION_SECONDS:-604800}"
max_temp="${MAX_TEMP:-83}"
resume_temp="${RESUME_TEMP:-78}"
check_interval="${CHECK_INTERVAL:-10}"
matrix_size="${MATRIX_SIZE:-0}"
memory_fraction="${MEMORY_FRACTION:-0.45}"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed" >&2
    exit 2
fi
if [[ ! "$session_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "SESSION_NAME may contain only letters, digits, underscores, and hyphens" >&2
    exit 2
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session '$session_name' already exists" >&2
    echo "Attach with: tmux attach -t $session_name" >&2
    exit 2
fi

printf -v tmux_command 'exec bash %q' "$script_dir/run_gpu_stress_7d.sh"
tmux new-session -d \
    -s "$session_name" \
    -c "$script_dir" \
    -e "GPU_IDS=$gpu_selection" \
    -e "DURATION_SECONDS=$duration_seconds" \
    -e "MAX_TEMP=$max_temp" \
    -e "RESUME_TEMP=$resume_temp" \
    -e "CHECK_INTERVAL=$check_interval" \
    -e "MATRIX_SIZE=$matrix_size" \
    -e "MEMORY_FRACTION=$memory_fraction" \
    -e "PYTHON_BIN=$python_bin" \
    "$tmux_command"

echo "Started tmux session: $session_name"
echo "GPU selection: $gpu_selection"
echo "Duration: $duration_seconds seconds (7 days by default)"
echo "Attach: tmux attach -t $session_name"
echo "Stop:   $script_dir/stop_gpu_stress_7d.sh $session_name"
