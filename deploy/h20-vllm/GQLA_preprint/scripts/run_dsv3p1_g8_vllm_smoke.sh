#!/usr/bin/env bash
# Weights-free DeepSeek-V3.1 H=128/G=8 vLLM smoke matrix.
#
# Examples:
#   bash scripts/run_dsv3p1_g8_vllm_smoke.sh
#   TPS=1,2 SMOKE_PATHS=gqa-hpc,mqa-absorb \
#       bash scripts/run_dsv3p1_g8_vllm_smoke.sh

set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-/prodcpfs/user/panzhixin/GQLA/envs/venv-py312/bin/python}
TPS=${TPS:-1}
SMOKE_PATHS=${SMOKE_PATHS:-gqa-hpc,mqa-absorb}
GQA_ARCHITECTURE=${GQA_ARCHITECTURE:-DeepseekV3GQLAHPCForCausalLM}
MQA_ARCHITECTURE=${MQA_ARCHITECTURE:-DeepseekV3GQLAForCausalLM}
LOG_ROOT=${LOG_ROOT:-$HERE/outputs/smoke/dsv3p1_g8_vllm}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
PROMPT_TOKENS=${PROMPT_TOKENS:-16}
DECODE_TOKENS=${DECODE_TOKENS:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-64}
VERIFY_HPC_TRACE=${VERIFY_HPC_TRACE:-1}
VERIFY_HPC_MIXED_SPLIT=${VERIFY_HPC_MIXED_SPLIT:-1}
HPC_HIT_PATTERN=${HPC_HIT_PATTERN:-GQLA_HPC_TRACE_HIT}
HPC_FALLBACK_PATTERN=${HPC_FALLBACK_PATTERN:-GQLA_HPC_TRACE_FALLBACK}
HPC_SCALE_PATTERN=${HPC_SCALE_PATTERN:-softmax_scale=0\\.135233}
HPC_ALL_DECODE_PATTERN=${HPC_ALL_DECODE_PATTERN:-GQLA_HPC_ALL_DECODE_STRICT_ENABLED}
HPC_MIXED_SPLIT_PATTERN=${HPC_MIXED_SPLIT_PATTERN:-GQLA_HPC_TRACE_MIXED_SPLIT}
HPC_ADAPTIVE_SPLITK_PATTERN=${HPC_ADAPTIVE_SPLITK_PATTERN:-splitk=adaptive_static}

for pair in \
    "VERIFY_HPC_TRACE:$VERIFY_HPC_TRACE" \
    "VERIFY_HPC_MIXED_SPLIT:$VERIFY_HPC_MIXED_SPLIT"; do
    case "${pair#*:}" in
        0|1) ;;
        *) echo "${pair%%:*} must be 0 or 1" >&2; exit 2 ;;
    esac
done

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
mkdir -p "$LOG_ROOT"

auto_fixture=0
if [[ -z "${TINY_MODEL_DIR:-}" ]]; then
    TINY_MODEL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/dsv3p1-g8-vllm-smoke.XXXXXX")
    auto_fixture=1
fi

cleanup() {
    if [[ "$auto_fixture" == 1 && -d "$TINY_MODEL_DIR" ]]; then
        case "$TINY_MODEL_DIR" in
            "${TMPDIR:-/tmp}"/dsv3p1-g8-vllm-smoke.*) rm -rf -- "$TINY_MODEL_DIR" ;;
            *) echo "refusing to clean unexpected temporary path: $TINY_MODEL_DIR" >&2 ;;
        esac
    fi
}
trap cleanup EXIT

if [[ ! -f "$TINY_MODEL_DIR/gqla_smoke_fixture.json" ]]; then
    "$PYTHON" "$HERE/scripts/make_dsv3p1_g8_tiny.py" \
        --output-dir "$TINY_MODEL_DIR"
fi

IFS=',' read -r -a tp_values <<< "$TPS"
IFS=',' read -r -a path_values <<< "$SMOKE_PATHS"
max_tp=1
for tp in "${tp_values[@]}"; do
    case "$tp" in
        1|2) ;;
        *) echo "TPS supports only comma-separated 1 and 2; got $tp" >&2; exit 2 ;;
    esac
    (( tp > max_tp )) && max_tp=$tp
done
for path in "${path_values[@]}"; do
    case "$path" in
        gqa-hpc|mqa-absorb) ;;
        *) echo "SMOKE_PATHS contains unsupported path: $path" >&2; exit 2 ;;
    esac
done

visible=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
if (( visible < max_tp )); then
    echo "need at least $max_tp visible CUDA devices, found $visible" >&2
    exit 1
fi

echo "[smoke] fixture=$TINY_MODEL_DIR paths=$SMOKE_PATHS tp=$TPS visible_gpus=$visible"
for path in "${path_values[@]}"; do
    for tp in "${tp_values[@]}"; do
        tag=${path}-tp${tp}
        log=$LOG_ROOT/$tag.log
        result=$LOG_ROOT/$tag.json
        args=(
            --model-dir "$TINY_MODEL_DIR"
            --path "$path"
            --tp "$tp"
            --gqa-architecture "$GQA_ARCHITECTURE"
            --mqa-architecture "$MQA_ARCHITECTURE"
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
            --prompt-tokens "$PROMPT_TOKENS"
            --decode-tokens "$DECODE_TOKENS"
            --max-model-len "$MAX_MODEL_LEN"
            --result-json "$result"
        )
        if [[ "$VERIFY_HPC_MIXED_SPLIT" == 1 ]]; then
            args+=(--mixed-batch)
        fi

        echo "[smoke] start path=$path tp=$tp log=$log"
        set +e
        "$PYTHON" -u "$HERE/scripts/smoke_dsv3p1_g8_vllm.py" "${args[@]}" \
            2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        set -e
        if (( rc != 0 )); then
            echo "DSV3P1_G8_VLLM_CASE_FAILED path=$path tp=$tp rc=$rc log=$log" >&2
            exit "$rc"
        fi

        if [[ "$path" == gqa-hpc && "$VERIFY_HPC_TRACE" == 1 ]]; then
            if grep -Eq -- "$HPC_FALLBACK_PATTERN" "$log"; then
                echo "HPC strict trace reported fallback for path=$path tp=$tp" >&2
                exit 1
            fi
            if ! grep -Eq -- "$HPC_HIT_PATTERN" "$log"; then
                echo "HPC hit proof missing for path=$path tp=$tp; log=$log" >&2
                exit 1
            fi
            if ! grep -Eq -- "$HPC_ALL_DECODE_PATTERN" "$log"; then
                echo "all-decode HPC strict proof missing for path=$path tp=$tp; log=$log" >&2
                exit 1
            fi
            if ! grep -Eq -- "$HPC_ADAPTIVE_SPLITK_PATTERN" "$log"; then
                echo "adaptive static split-K proof missing for path=$path tp=$tp; log=$log" >&2
                exit 1
            fi
            if [[ "$VERIFY_HPC_MIXED_SPLIT" == 1 ]] \
                && ! grep -Eq -- "$HPC_MIXED_SPLIT_PATTERN" "$log"; then
                echo "mixed-batch HPC split proof missing for path=$path tp=$tp; log=$log" >&2
                exit 1
            fi
            if ! grep -Eq -- "$HPC_SCALE_PATTERN" "$log"; then
                echo "HPC production YaRN scale proof missing for path=$path tp=$tp; log=$log" >&2
                exit 1
            fi
        elif [[ "$path" == mqa-absorb ]] && grep -Eq -- "$HPC_HIT_PATTERN" "$log"; then
            echo "MQA/MLA control unexpectedly emitted the HPC hit marker; log=$log" >&2
            exit 1
        fi
        echo "DSV3P1_G8_VLLM_CASE_OK path=$path tp=$tp result=$result log=$log"
    done
done

echo "DSV3P1_G8_VLLM_MATRIX_OK paths=$SMOKE_PATHS tp=$TPS log_root=$LOG_ROOT"
