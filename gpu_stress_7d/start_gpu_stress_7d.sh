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
if ! resolved_python_bin="$(command -v "$python_bin")"; then
    echo "Python interpreter not found: $python_bin" >&2
    echo "Set PYTHON_BIN to the Python environment containing CUDA PyTorch." >&2
    exit 2
fi
python_bin="$resolved_python_bin"
if [[ ! "$session_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "SESSION_NAME may contain only letters, digits, underscores, and hyphens" >&2
    exit 2
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session '$session_name' already exists" >&2
    echo "Attach with: tmux attach -t $session_name" >&2
    exit 2
fi

mkdir -p -- "$script_dir/logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
launcher_log="$script_dir/logs/launcher_${timestamp}.log"
printf -v runner_command \
    'set -o pipefail; bash %q 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' \
    "$script_dir/run_gpu_stress_7d.sh" "$launcher_log"
printf -v tmux_command 'exec bash -c %q' "$runner_command"
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
    -e "PATH=$PATH" \
    -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
    -e "CONDA_PREFIX=${CONDA_PREFIX:-}" \
    "$tmux_command"

sleep 3
if ! tmux has-session -t "$session_name" 2>/dev/null; then
    echo "GPU stress test failed during startup." >&2
    echo "Startup log: $launcher_log" >&2
    echo "----- startup error -----" >&2
    tail -n 80 -- "$launcher_log" >&2 || true
    echo "-------------------------" >&2
    exit 1
fi

echo "Started tmux session: $session_name"
echo "GPU selection: $gpu_selection"
echo "Python: $python_bin"
echo "Duration: $duration_seconds seconds (7 days by default)"
echo "Startup log: $launcher_log"
echo "Attach: tmux attach -t $session_name"
echo "Stop:   $script_dir/stop_gpu_stress_7d.sh $session_name"
