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
if [[ -v PYTHON_BIN ]]; then
    python_was_explicit=1
    python_bin="$PYTHON_BIN"
else
    python_was_explicit=0
    python_bin=python3
fi

probe_cuda_python() {
    local candidate_python=$1
    local probe_code
    probe_code='import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is false"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")'
    if command -v timeout >/dev/null 2>&1; then
        timeout 30 env PYTHONDONTWRITEBYTECODE=1 \
            "$candidate_python" -c "$probe_code"
    else
        env PYTHONDONTWRITEBYTECODE=1 "$candidate_python" -c "$probe_code"
    fi
}

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

if python_probe_output="$(probe_cuda_python "$python_bin" 2>&1)"; then
    :
elif (( python_was_explicit == 1 )); then
    echo "PYTHON_BIN does not provide CUDA-enabled PyTorch: $python_bin" >&2
    echo "$python_probe_output" >&2
    exit 2
else
    initial_python_error="$python_probe_output"
    selected_python=""
    if command -v conda >/dev/null 2>&1; then
        mapfile -t conda_prefixes < <(
            conda env list | awk 'NF && $1 !~ /^#/ {print $NF}'
        )
        for conda_prefix in "${conda_prefixes[@]}"; do
            candidate_python="$conda_prefix/bin/python"
            if [[ ! -x "$candidate_python" || "$candidate_python" == "$python_bin" ]]; then
                continue
            fi
            if candidate_probe_output="$(probe_cuda_python "$candidate_python" 2>/dev/null)"; then
                selected_python="$candidate_python"
                python_probe_output="$candidate_probe_output"
                break
            fi
        done
    fi

    if [[ -z "$selected_python" ]]; then
        echo "No CUDA-enabled PyTorch environment was found." >&2
        echo "Initial Python: $python_bin" >&2
        echo "$initial_python_error" >&2
        echo >&2
        echo "If the repository's h20table2 environment already exists, run:" >&2
        echo "  conda activate h20table2" >&2
        echo "  PYTHON_BIN=\"\$(command -v python)\" ./start_gpu_stress_7d.sh all" >&2
        if [[ -f "$script_dir/../scripts/start_cu128_env_tmux.sh" ]]; then
            echo >&2
            echo "If it does not exist, start the repository's environment installer:" >&2
            echo "  bash $script_dir/../scripts/start_cu128_env_tmux.sh" >&2
        fi
        exit 2
    fi

    python_bin="$selected_python"
    echo "Automatically selected CUDA PyTorch environment: $python_bin"
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
echo "Python probe: $python_probe_output"
echo "Duration: $duration_seconds seconds (7 days by default)"
echo "Startup log: $launcher_log"
echo "Attach: tmux attach -t $session_name"
echo "Stop:   $script_dir/stop_gpu_stress_7d.sh $session_name"
