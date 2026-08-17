#!/usr/bin/env bash
# End-to-end vLLM benchmark for one route of the converted DeepSeek-V3.1 G=8
# checkpoint. The topology is intentionally fixed to TP=8, PP=2 (16 ranks).
# BENCHMARK_PATH defaults to the historical MLA control and may be set to
# gqa-hpc by the dedicated H100 GQLA entrypoints.
#
# DLC two-worker job (submit this same entry command once; DLC executes it on
# both workers and injects node-level RANK/WORLD_SIZE/MASTER_ADDR):
#   bash scripts/benchmark_dsv3p1_g8_mla_tp8_pp2.sh
#
# Single host with 16 visible GPUs (manual override):
#   NNODES=1 ROLE=head bash scripts/benchmark_dsv3p1_g8_mla_tp8_pp2.sh
#
# Two hosts with 8 visible GPUs each (manual fallback):
#   # node 0
#   MASTER_ADDR=<node0-ip> ROLE=head NODE_RANK=0 \
#     bash scripts/benchmark_dsv3p1_g8_mla_tp8_pp2.sh
#   # node 1
#   MASTER_ADDR=<node0-ip> WORKER_ADDR=<node1-ip> ROLE=worker NODE_RANK=1 \
#     bash scripts/benchmark_dsv3p1_g8_mla_tp8_pp2.sh

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
GQLA_ROOT=${GQLA_ROOT:-$(cd "$REPO_DIR/../.." && pwd -P)}
BASE_SCRIPT=$SCRIPT_DIR/benchmark_dsv3p1_g8_vllm.sh
PYTHON=${PYTHON:-/prodcpfs/user/panzhixin/GQLA/envs/venv-py312/bin/python}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract}
BENCHMARK_PATH=${BENCHMARK_PATH:-mqa-mla}
case "$BENCHMARK_PATH" in
    mqa-mla)
        DEFAULT_OUTPUT_ROOT=$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_mla_tp8_pp2
        ;;
    gqa-hpc)
        DEFAULT_OUTPUT_ROOT=$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_gqla_h100_tp8_pp2
        ;;
    *)
        echo "error: BENCHMARK_PATH must be mqa-mla or gqa-hpc; got $BENCHMARK_PATH" >&2
        exit 2
        ;;
esac
OUTPUT_ROOT=${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}
# RUN_ID is the stable experiment base. The head appends the first unused
# -attemptN suffix so a DLC whole-job restart never overwrites an earlier run.
RUN_ID_BASE=${RUN_ID_BASE:-${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}}
ATTEMPT=${ATTEMPT:-auto}
RUN_ID=

# Capture DLC's node-level scheduler variables before this launcher clears
# RANK/WORLD_SIZE and lets vLLM recreate process-level values for 16 GPU ranks.
SCHEDULER_NODE_RANK=${MACHINE_RANK:-${NODE_RANK:-${RANK:-}}}
SCHEDULER_NODE_COUNT=${NUM_MACHINES:-${WORLD_SIZE:-}}
SCHEDULER_MASTER_ADDR=${MAIN_PROCESS_IP:-${MASTER_ADDR:-}}
SCHEDULER_MASTER_PORT=${MAIN_PROCESS_PORT:-${MASTER_PORT:-29501}}

ROLE=${ROLE:-auto}
NNODES=${NNODES:-}
NODE_RANK=${NODE_RANK:-}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-}
WORKER_ADDR=${WORKER_ADDR:-}
VLLM_HOST_IP=${VLLM_HOST_IP:-}
DRY_RUN=${DRY_RUN:-0}

readonly TP_SIZE=8
readonly PP_SIZE=2
readonly MODEL_WORLD_SIZE=16
readonly PP_PARTITION=31,30
readonly GQA_ARCHITECTURE=DeepseekV3GQLAHPCForCausalLM
readonly MQA_ARCHITECTURE=DeepseekV3GQLAForCausalLM
if [[ "$BENCHMARK_PATH" == gqa-hpc ]]; then
    readonly ARCHITECTURE=$GQA_ARCHITECTURE
    readonly ROUTE_DISPLAY=GQLA-HPC
    readonly ROUTE_MARKER=GQLA_HPC
    readonly DEFAULT_SERVED_MODEL_NAME=dsv3p1-g8-gqla-tp8-pp2
else
    readonly ARCHITECTURE=$MQA_ARCHITECTURE
    readonly ROUTE_DISPLAY=MLA
    readonly ROUTE_MARKER=MLA
    readonly DEFAULT_SERVED_MODEL_NAME=dsv3p1-g8-mla-tp8-pp2
fi
readonly SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$DEFAULT_SERVED_MODEL_NAME}
readonly EXPECTED_VLLM_VERSION=0.22.1

GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8100}
SEED=${SEED:-42}

# Fixed-length online workload. These values match the GQA-vs-MLA benchmark
# protocol so the MLA result can later serve as the control row.
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-1024}
NUM_WARMUPS=${NUM_WARMUPS:-64}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
REQUEST_RATE=${REQUEST_RATE:-inf}
METRIC_PERCENTILES=${METRIC_PERCENTILES:-50,90,99}
PERCENTILE_METRICS=${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-7200}
SHUTDOWN_TIMEOUT_SECONDS=${SHUTDOWN_TIMEOUT_SECONDS:-180}
MASTER_RESOLVE_TIMEOUT_SECONDS=${MASTER_RESOLVE_TIMEOUT_SECONDS:-300}
MASTER_RESOLVE_POLL_SECONDS=${MASTER_RESOLVE_POLL_SECONDS:-2}

die() {
    echo "error: $*" >&2
    exit 2
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

require_bool() {
    local name=$1
    local value=$2
    case "$value" in
        0|1) ;;
        *) die "$name must be 0 or 1; got $value" ;;
    esac
}

resolve_topology() {
    local scheduler_rank=$SCHEDULER_NODE_RANK
    local scheduler_nodes=$SCHEDULER_NODE_COUNT

    case "$ROLE" in
        auto)
            [[ -n "$scheduler_rank" ]] \
                || die "DLC auto mode needs node-level RANK (or MACHINE_RANK/NODE_RANK)"
            [[ -n "$scheduler_nodes" ]] \
                || die "DLC auto mode needs node-level WORLD_SIZE (or NUM_MACHINES)"
            [[ "$scheduler_rank" =~ ^[0-9]+$ ]] \
                || die "DLC node rank must be a non-negative integer; got $scheduler_rank"
            [[ "$scheduler_nodes" =~ ^[1-9][0-9]*$ ]] \
                || die "DLC node count must be a positive integer; got $scheduler_nodes"
            [[ "$scheduler_nodes" == 2 ]] \
                || die "this TP8 x PP2 DLC entry requires exactly 2 workers; scheduler reports $scheduler_nodes"
            (( scheduler_rank < scheduler_nodes )) \
                || die "DLC node rank $scheduler_rank is outside 0..$((scheduler_nodes - 1))"

            NNODES=$scheduler_nodes
            NODE_RANK=$scheduler_rank
            MASTER_ADDR=${SCHEDULER_MASTER_ADDR:-$MASTER_ADDR}
            MASTER_PORT=${SCHEDULER_MASTER_PORT:-29501}
            if [[ "$NODE_RANK" == 0 ]]; then
                ROLE=head
            else
                ROLE=worker
            fi
            echo "[dlc-auto] scheduler node_rank=$NODE_RANK/$NNODES resolved_role=$ROLE"
            ;;
        head|worker)
            NNODES=${NNODES:-2}
            if [[ -z "$NODE_RANK" ]]; then
                if [[ "$ROLE" == worker ]]; then
                    NODE_RANK=1
                else
                    NODE_RANK=0
                fi
            fi
            MASTER_PORT=${MASTER_PORT:-29501}
            ;;
        *)
            die "ROLE must be auto, head, or worker; got $ROLE"
            ;;
    esac
}

detect_route_source_ip() {
    "$PYTHON" - "$MASTER_ADDR" "$MASTER_PORT" <<'PY'
import socket
import sys

master_addr = sys.argv[1]
master_port = int(sys.argv[2])
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # UDP connect selects the local source address without requiring the
        # rendezvous server to be listening yet.
        sock.connect((master_addr, master_port))
        local_ip = sock.getsockname()[0]
except OSError as exc:
    raise SystemExit(f"route lookup for {master_addr!r} failed: {exc}") from None
if local_ip in {"0.0.0.0", "127.0.0.1"}:
    raise SystemExit(f"route to {master_addr!r} selected unusable {local_ip!r}")
print(local_ip)
PY
}

resolve_worker_host_ip() {
    local deadline=$((SECONDS + MASTER_RESOLVE_TIMEOUT_SECONDS))
    local probe_count=0
    local inferred_ip

    while true; do
        if inferred_ip=$(detect_route_source_ip 2>/dev/null); then
            VLLM_HOST_IP=$inferred_ip
            echo "[dlc-auto] inferred worker VLLM_HOST_IP=$VLLM_HOST_IP from route to $MASTER_ADDR"
            return
        fi

        probe_count=$((probe_count + 1))
        if [[ "$DRY_RUN" == 1 ]] || (( SECONDS >= deadline )); then
            die "cannot infer worker VLLM_HOST_IP from route to $MASTER_ADDR after $probe_count probe(s); set VLLM_HOST_IP explicitly"
        fi
        if (( probe_count == 1 || probe_count % 15 == 0 )); then
            echo "[dlc-auto] waiting for MASTER_ADDR=$MASTER_ADDR to resolve before inferring worker VLLM_HOST_IP (probe=$probe_count)" >&2
        fi
        sleep "$MASTER_RESOLVE_POLL_SECONDS"
    done
}

resolve_run_id() {
    local attempt_index

    [[ "$ROLE" == head ]] || return 0
    [[ "$RUN_ID_BASE" =~ ^[A-Za-z0-9._-]+$ ]] \
        || die "RUN_ID may contain only letters, digits, dot, underscore, and dash"

    if [[ "$ATTEMPT" == auto ]]; then
        attempt_index=0
        # Treat a pre-attempt-suffix directory as attempt 0. This preserves
        # failed results made by older versions of this launcher.
        if [[ -e "$OUTPUT_ROOT/$RUN_ID_BASE" ]]; then
            attempt_index=1
        fi
        while [[ -e "$OUTPUT_ROOT/${RUN_ID_BASE}-attempt${attempt_index}" ]]; do
            attempt_index=$((attempt_index + 1))
        done
    else
        [[ "$ATTEMPT" =~ ^[0-9]+$ ]] \
            || die "ATTEMPT must be auto or a non-negative integer; got $ATTEMPT"
        attempt_index=$ATTEMPT
    fi

    ATTEMPT=$attempt_index
    RUN_ID=${RUN_ID_BASE}-attempt${ATTEMPT}
    echo "[attempt] base=$RUN_ID_BASE attempt=$ATTEMPT run_id=$RUN_ID"
}

validate_common() {
    require_bool DRY_RUN "$DRY_RUN"
    require_bool ENFORCE_EAGER "$ENFORCE_EAGER"
    require_bool ENABLE_CHUNKED_PREFILL "$ENABLE_CHUNKED_PREFILL"
    [[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON"
    case "$ROLE" in
        head|worker) ;;
        *) die "internal error: unresolved ROLE=$ROLE" ;;
    esac
    case "$NNODES" in
        1|2) ;;
        *) die "this fixed 16-rank launcher supports NNODES=1 or 2; got $NNODES" ;;
    esac
    if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
        die "NODE_RANK=$NODE_RANK is invalid for NNODES=$NNODES"
    fi
    if [[ "$ROLE" == head && "$NODE_RANK" != 0 ]]; then
        die "ROLE=head requires NODE_RANK=0"
    fi
    if [[ "$ROLE" == worker && ( "$NNODES" != 2 || "$NODE_RANK" != 1 ) ]]; then
        die "ROLE=worker is only valid as NODE_RANK=1 with NNODES=2"
    fi
    [[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] \
        || die "MASTER_PORT must be a positive integer; got $MASTER_PORT"
    (( MASTER_PORT <= 65535 )) \
        || die "MASTER_PORT must be <= 65535; got $MASTER_PORT"
    [[ "$MASTER_RESOLVE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
        || die "MASTER_RESOLVE_TIMEOUT_SECONDS must be a positive integer; got $MASTER_RESOLVE_TIMEOUT_SECONDS"
    [[ "$MASTER_RESOLVE_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] \
        || die "MASTER_RESOLVE_POLL_SECONDS must be a positive integer; got $MASTER_RESOLVE_POLL_SECONDS"
    if [[ "$NNODES" == 2 ]]; then
        case "$MASTER_ADDR" in
            ""|127.0.0.1|localhost) \
                die "two-node launch requires reachable MASTER_ADDR=<node0-ip>" ;;
        esac
        if [[ "$ROLE" == head ]]; then
            VLLM_HOST_IP=${VLLM_HOST_IP:-$MASTER_ADDR}
        else
            VLLM_HOST_IP=${VLLM_HOST_IP:-$WORKER_ADDR}
            if [[ -z "$VLLM_HOST_IP" ]]; then
                resolve_worker_host_ip
            fi
            case "$VLLM_HOST_IP" in
                ""|0.0.0.0|127.0.0.1|localhost) \
                    die "worker requires reachable WORKER_ADDR=<node1-ip> or VLLM_HOST_IP" ;;
            esac
        fi
    else
        MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
        VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}
    fi
    if [[ "$CPU_OFFLOAD_GB" != 0 && "$CPU_OFFLOAD_GB" != 0.0 ]]; then
        die "formal H100 TP8/PP2 benchmark fixes CPU_OFFLOAD_GB=0; got $CPU_OFFLOAD_GB"
    fi
    [[ -f "$BASE_SCRIPT" ]] || die "base benchmark is missing: $BASE_SCRIPT"
    MODEL_DIR=$(readlink -f -- "$MODEL_DIR") \
        || die "converted model directory does not exist: $MODEL_DIR"
    for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
        [[ -f "$MODEL_DIR/$required" ]] \
            || die "missing converted-model file: $MODEL_DIR/$required"
    done
    if [[ "$DRY_RUN" == 0 ]]; then
        "$PYTHON" - "$VLLM_HOST_IP" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind((sys.argv[1], 0))
    except OSError as exc:
        raise SystemExit(f"VLLM_HOST_IP={sys.argv[1]!r} is not local/bindable: {exc}")
PY
    fi
}

run_worker() {
    local hf_overrides
    local visible_gpus
    local vllm_version
    local -a worker_command
    local -a route_env

    hf_overrides=$(printf '{"architectures":["%s"]}' "$ARCHITECTURE")
    worker_command=(
        "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_DIR"
        --headless
        --served-model-name "$SERVED_MODEL_NAME"
        --hf-overrides "$hf_overrides"
        --model-impl vllm
        --trust-remote-code
        --tensor-parallel-size "$TP_SIZE"
        --pipeline-parallel-size "$PP_SIZE"
        --distributed-executor-backend mp
        --nnodes "$NNODES"
        --node-rank "$NODE_RANK"
        --master-addr "$MASTER_ADDR"
        --master-port "$MASTER_PORT"
        --dtype bfloat16
        --kv-cache-dtype auto
        --block-size 64
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --cpu-offload-gb 0
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --no-enable-prefix-caching
        --disable-cascade-attn
        --generation-config vllm
        --disable-log-stats
        --seed "$SEED"
    )
    if [[ "$ENFORCE_EAGER" == 1 ]]; then
        worker_command+=(--enforce-eager)
    else
        worker_command+=(--no-enforce-eager)
    fi
    if [[ "$ENABLE_CHUNKED_PREFILL" == 1 ]]; then
        worker_command+=(--enable-chunked-prefill)
    else
        worker_command+=(--no-enable-chunked-prefill)
    fi

    route_env=(
        env
        -u GQLA_HPC_STRICT \
        -u GQLA_HPC_TRACE \
        "PYTHONPATH=$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        VLLM_PP_LAYER_PARTITION="$PP_PARTITION" \
        "VLLM_HOST_IP=$VLLM_HOST_IP" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false
    )
    if [[ "$BENCHMARK_PATH" == gqa-hpc ]]; then
        route_env+=(GQLA_HPC_STRICT=1 GQLA_HPC_TRACE=1)
    fi

    echo "[worker] $ROUTE_DISPLAY TP=$TP_SIZE PP=$PP_SIZE node_rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT host_ip=$VLLM_HOST_IP"
    print_command "${route_env[@]}" "${worker_command[@]}"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DSV3P1_G8_${ROUTE_MARKER}_TP8_PP2_WORKER_DRY_RUN_OK"
        return
    fi

    visible_gpus=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
    (( visible_gpus >= 8 )) \
        || die "worker node needs 8 visible GPUs, found $visible_gpus"
    vllm_version=$("$PYTHON" -c \
        'from importlib.metadata import version; print(version("vllm"))')
    [[ "$vllm_version" == "$EXPECTED_VLLM_VERSION" ]] \
        || die "expected vLLM $EXPECTED_VLLM_VERSION, found $vllm_version"

    exec "${route_env[@]}" "${worker_command[@]}"
}

run_head() {
    export GQLA_ROOT PYTHON MODEL_DIR OUTPUT_ROOT RUN_ID
    export BENCHMARK_ENTRYPOINT=${BENCHMARK_ENTRYPOINT:-${BASH_SOURCE[0]}}
    export TPS=$TP_SIZE
    export PATHS=$BENCHMARK_PATH
    export PIPELINE_PARALLEL_SIZE=$PP_SIZE
    export PP_LAYER_PARTITION=$PP_PARTITION
    export DISTRIBUTED_EXECUTOR_BACKEND=mp
    export NNODES NODE_RANK MASTER_ADDR MASTER_PORT VLLM_HOST_IP
    export GQA_ARCHITECTURE MQA_ARCHITECTURE
    export SERVED_MODEL_NAME EXPECTED_VLLM_VERSION
    export GPU_MEMORY_UTILIZATION CPU_OFFLOAD_GB
    export MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS ENFORCE_EAGER
    export ENABLE_CHUNKED_PREFILL
    export HOST PORT SEED
    export INPUT_LEN OUTPUT_LEN NUM_PROMPTS NUM_WARMUPS MAX_CONCURRENCY
    export REQUEST_RATE METRIC_PERCENTILES PERCENTILE_METRICS
    export STARTUP_TIMEOUT_SECONDS SHUTDOWN_TIMEOUT_SECONDS DRY_RUN

    echo "[head] $ROUTE_DISPLAY TP=$TP_SIZE PP=$PP_SIZE world=$MODEL_WORLD_SIZE nnodes=$NNODES master=$MASTER_ADDR:$MASTER_PORT host_ip=$VLLM_HOST_IP"
    exec bash "$BASE_SCRIPT"
}

resolve_topology
validate_common
resolve_run_id

# DLC's parent values describe nodes, not GPU processes. vLLM/torch distributed
# must recreate these for the 16 TP/PP ranks; leaking WORLD_SIZE=2 is unsafe.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
    2>/dev/null || true

if [[ "$ROLE" == worker ]]; then
    run_worker
else
    run_head
fi
