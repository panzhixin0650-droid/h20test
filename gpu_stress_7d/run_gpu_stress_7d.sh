#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
gpu_selection="${GPU_IDS:-0}"
duration_seconds="${DURATION_SECONDS:-604800}"
max_temp="${MAX_TEMP:-83}"
resume_temp="${RESUME_TEMP:-78}"
check_interval="${CHECK_INTERVAL:-10}"
matrix_size="${MATRIX_SIZE:-0}"
memory_fraction="${MEMORY_FRACTION:-0.45}"

for integer_value in "$duration_seconds" "$matrix_size"; do
    if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "DURATION_SECONDS and MATRIX_SIZE must be non-negative integers" >&2
        exit 2
    fi
done
if (( duration_seconds == 0 )); then
    echo "DURATION_SECONDS must be positive" >&2
    exit 2
fi

for command_name in nvidia-smi "$python_bin"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "required command not found: $command_name" >&2
        exit 2
    fi
done

mapfile -t available_gpus < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed 's/[[:space:]]//g'
)
if (( ${#available_gpus[@]} == 0 )); then
    echo "nvidia-smi reported no GPUs" >&2
    exit 2
fi

if [[ "$gpu_selection" == "all" ]]; then
    gpu_ids=("${available_gpus[@]}")
elif [[ "$gpu_selection" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    IFS=',' read -r -a gpu_ids <<< "$gpu_selection"
else
    echo "GPU_IDS must be 'all' or a comma-separated list such as 0,1,2,3" >&2
    exit 2
fi

for gpu_id in "${gpu_ids[@]}"; do
    found=0
    for available_gpu in "${available_gpus[@]}"; do
        if [[ "$gpu_id" == "$available_gpu" ]]; then
            found=1
            break
        fi
    done
    if (( found == 0 )); then
        echo "GPU $gpu_id is unavailable; detected GPUs: ${available_gpus[*]}" >&2
        exit 2
    fi
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${LOG_DIR:-$script_dir/logs/$timestamp}"
mkdir -p -- "$log_dir"

echo "[$(date -Is)] seven-day GPU stress test"
echo "GPUs: ${gpu_ids[*]}"
echo "Duration: $duration_seconds seconds"
echo "Thermal guard: pause=${max_temp}C resume=${resume_temp}C"
echo "Logs: $log_dir"
echo "Press Ctrl-C in this tmux pane to stop all workers safely."

pids=()
cleanup() {
    if (( ${#pids[@]} > 0 )); then
        kill -TERM "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
on_signal() {
    echo "[$(date -Is)] stop requested; terminating GPU workers"
    cleanup
    exit 130
}
trap cleanup EXIT
trap on_signal INT TERM HUP

for gpu_id in "${gpu_ids[@]}"; do
    worker_log="$log_dir/gpu_${gpu_id}.log"
    echo "[$(date -Is)] starting GPU $gpu_id -> $worker_log"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u "$script_dir/gpu_stress.py" \
        --gpu-id "$gpu_id" \
        --duration "$duration_seconds" \
        --max-temp "$max_temp" \
        --resume-temp "$resume_temp" \
        --check-interval "$check_interval" \
        --matrix-size "$matrix_size" \
        --memory-fraction "$memory_fraction" \
        > >(tee -a "$worker_log") 2>&1 &
    pids+=("$!")
done

result=0
active_pids=("${pids[@]}")
while (( ${#active_pids[@]} > 0 )); do
    finished_pid=""
    if wait -n -p finished_pid "${active_pids[@]}"; then
        worker_status=0
    else
        worker_status=$?
    fi

    finished_gpu="unknown"
    for index in "${!pids[@]}"; do
        if [[ "${pids[$index]}" == "$finished_pid" ]]; then
            finished_gpu="${gpu_ids[$index]}"
            break
        fi
    done

    next_active_pids=()
    for active_pid in "${active_pids[@]}"; do
        if [[ "$active_pid" != "$finished_pid" ]]; then
            next_active_pids+=("$active_pid")
        fi
    done
    active_pids=("${next_active_pids[@]}")

    if (( worker_status == 0 )); then
        echo "[$(date -Is)] GPU $finished_gpu worker completed"
    else
        echo "[$(date -Is)] GPU $finished_gpu worker exited with status $worker_status" >&2
        echo "[$(date -Is)] stopping remaining workers because the test is incomplete" >&2
        result=1
        if (( ${#active_pids[@]} > 0 )); then
            kill -TERM "${active_pids[@]}" 2>/dev/null || true
            wait "${active_pids[@]}" 2>/dev/null || true
        fi
        break
    fi
done
pids=()

echo "[$(date -Is)] stress test finished; logs remain in $log_dir"
exit "$result"
