#!/usr/bin/env bash
# Reproducible online vLLM benchmark matrix for the converted DeepSeek-V3.1 G=8
# checkpoint. Each case starts a fresh OpenAI-compatible server, runs the same
# fixed random workload, verifies the selected attention route, and shuts down
# only the process group created by this script.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
GQLA_ROOT=${GQLA_ROOT:-$(cd "$REPO_DIR/../.." && pwd -P)}
PYTHON=${PYTHON:-/prodcpfs/user/panzhixin/GQLA/envs/venv-py312/bin/python}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract}
OUTPUT_ROOT=${OUTPUT_ROOT:-$GQLA_ROOT/outputs/benchmarks/dsv3p1_g8_vllm}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
BENCHMARK_ENTRYPOINT=${BENCHMARK_ENTRYPOINT:-${BASH_SOURCE[0]}}

TPS=${TPS:-1,2,4,8}
PATHS=${PATHS:-gqa-hpc,mqa-mla}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
PP_LAYER_PARTITION=${PP_LAYER_PARTITION:-}
DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND:-}
RAY_ADDRESS=${RAY_ADDRESS:-}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
VLLM_HOST_IP=${VLLM_HOST_IP:-}
GQA_ARCHITECTURE=${GQA_ARCHITECTURE:-DeepseekV3GQLAHPCForCausalLM}
MQA_ARCHITECTURE=${MQA_ARCHITECTURE:-DeepseekV3GQLAForCausalLM}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-dsv3p1-g8-converted}
EXPECTED_VLLM_VERSION=${EXPECTED_VLLM_VERSION:-0.22.1}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8100}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-0}

# These benchmark parameters are computed once and reused verbatim for every
# path/TP case. Override them only at run scope, never per case.
DATASET_NAME=random
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-256}
NUM_WARMUPS=${NUM_WARMUPS:-16}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
REQUEST_RATE=${REQUEST_RATE:-inf}
SEED=${SEED:-42}
METRIC_PERCENTILES=${METRIC_PERCENTILES:-50,90,99}
PERCENTILE_METRICS=${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}

STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-3600}
SHUTDOWN_TIMEOUT_SECONDS=${SHUTDOWN_TIMEOUT_SECONDS:-120}
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-300}
DRY_RUN=${DRY_RUN:-0}
readonly BLOCK_SIZE=64

HPC_HIT_PATTERN=${HPC_HIT_PATTERN:-GQLA_HPC_TRACE_HIT}
HPC_FALLBACK_PATTERN=${HPC_FALLBACK_PATTERN:-GQLA_HPC_TRACE_FALLBACK}
HPC_SCALE_PATTERN=${HPC_SCALE_PATTERN:-softmax_scale=0\.135233}

SERVER_PID=
SERVER_PGID=
SERVER_LOG=
# Avoid a fragile dependency on the container's `ps` binary. Some DLC
# CUDA-compat images make `ps` fail with "fatal library error, lookup self".
# Python's getpgrp() is also correct when the launcher runs in a PID namespace.
SELF_PGID=$("$PYTHON" -c 'import os; print(os.getpgrp())')

die() {
    echo "error: $*" >&2
    exit 2
}

require_bool() {
    local name=$1
    local value=$2
    case "$value" in
        0|1) ;;
        *) die "$name must be 0 or 1; got $value" ;;
    esac
}

require_positive_int() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        die "$name must be a positive integer; got $value"
    fi
}

require_nonnegative_int() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        die "$name must be a non-negative integer; got $value"
    fi
}

require_nonnegative_number() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        die "$name must be a non-negative number; got $value"
    fi
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

write_command() {
    local destination=$1
    shift
    print_command "$@" >"$destination"
}

group_is_alive() {
    local pgid=$1
    kill -0 -- "-$pgid" 2>/dev/null
}

stop_server() {
    local pgid=${SERVER_PGID:-}
    local pid=${SERVER_PID:-}
    local waited=0

    SERVER_PID=
    SERVER_PGID=
    if [[ -z "$pgid" ]]; then
        if [[ -n "$pid" && "$pid" =~ ^[1-9][0-9]*$ ]]; then
            kill -TERM -- "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        return
    fi
    if [[ ! "$pgid" =~ ^[1-9][0-9]*$ || "$pgid" == "$SELF_PGID" ]]; then
        echo "refusing to signal unsafe server process group: $pgid" >&2
        return 1
    fi

    if group_is_alive "$pgid"; then
        echo "[cleanup] TERM server process group $pgid"
        kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
    while group_is_alive "$pgid" && (( waited < SHUTDOWN_TIMEOUT_SECONDS )); do
        sleep 1
        ((waited += 1))
    done
    if group_is_alive "$pgid"; then
        echo "[cleanup] KILL server process group $pgid after ${waited}s" >&2
        kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
    if [[ -n "$pid" ]]; then
        wait "$pid" 2>/dev/null || true
    fi
}

on_exit() {
    local rc=$?
    trap - EXIT
    stop_server || true
    exit "$rc"
}

on_signal() {
    trap - EXIT INT TERM
    stop_server || true
    exit 130
}

trap on_exit EXIT
trap on_signal INT TERM

capture_server_group() {
    local attempt
    local pgid
    for ((attempt = 0; attempt < 100; attempt += 1)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return 1
        fi
        pgid=$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d '[:space:]')
        if [[ "$pgid" =~ ^[1-9][0-9]*$ && "$pgid" != "$SELF_PGID" ]]; then
            SERVER_PGID=$pgid
            return 0
        fi
        sleep 0.1
    done
    return 1
}

port_is_free_at() {
    local host=$1
    local port=$2
    "$PYTHON" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((host, port))
PY
}

port_is_free() {
    port_is_free_at "$HOST" "$PORT"
}

server_is_ready() {
    "$PYTHON" - "http://$HOST:$PORT/health" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
}

inspect_ray_cluster() {
    local ray_address=$1
    local required_world_size=$2
    local required_tp=$3
    local required_pp=$4
    "$PYTHON" - "$ray_address" "$required_world_size" "$required_tp" \
        "$required_pp" <<'PY'
import json
import sys

import ray

address = sys.argv[1]
world_size, tp, pp = map(int, sys.argv[2:])
ray.init(
    address=address,
    ignore_reinit_error=True,
    log_to_driver=False,
    logging_level="ERROR",
)
try:
    nodes = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        gpu_count = int(resources.get("GPU", 0))
        nodes.append(
            {
                "node_id": node.get("NodeID"),
                "node_ip": node.get("NodeManagerAddress"),
                "gpu_count": gpu_count,
                "resources": resources,
            }
        )
    total_gpus = sum(node["gpu_count"] for node in nodes)
    available_gpus = int(ray.available_resources().get("GPU", 0))
    tp_capable_nodes = sum(node["gpu_count"] >= tp for node in nodes)
    single_node_capable = any(
        node["gpu_count"] >= world_size for node in nodes
    )
    summary = {
        "required_world_size": world_size,
        "required_tp": tp,
        "required_pp": pp,
        "total_alive_gpus": total_gpus,
        "available_gpus": available_gpus,
        "tp_capable_nodes": tp_capable_nodes,
        "single_node_capable": single_node_capable,
        "nodes": nodes,
    }
    if total_gpus < world_size:
        raise SystemExit(
            f"Ray cluster has {total_gpus} alive GPUs; {world_size} required"
        )
    if available_gpus < world_size:
        raise SystemExit(
            f"Ray cluster has {available_gpus} available GPUs; {world_size} required"
        )
    if not single_node_capable and tp_capable_nodes < pp:
        raise SystemExit(
            "Ray cluster cannot keep every TP group on one node: "
            f"need {pp} node(s) with at least {tp} GPUs each, found "
            f"{tp_capable_nodes}"
        )
    print(json.dumps(summary, sort_keys=True))
finally:
    ray.shutdown()
PY
}

wait_for_server() {
    local deadline=$(( $(date +%s) + STARTUP_TIMEOUT_SECONDS ))
    while (( $(date +%s) < deadline )); do
        if server_is_ready; then
            return 0
        fi
        if ! group_is_alive "$SERVER_PGID"; then
            echo "server process group exited before /health became ready" >&2
            return 1
        fi
        sleep 2
    done
    echo "server did not become ready within ${STARTUP_TIMEOUT_SECONDS}s" >&2
    return 1
}

verify_trace() {
    local path=$1
    local tp=$2
    local architecture=$3
    local proof_file=$4
    local hits
    local fallbacks
    local scale_hits
    local architecture_hits
    local topology_hits
    local mla_backend_hits
    hits=$(grep -Ec -- "$HPC_HIT_PATTERN" "$SERVER_LOG" || true)
    fallbacks=$(grep -Ec -- "$HPC_FALLBACK_PATTERN" "$SERVER_LOG" || true)
    scale_hits=$(grep -Ec -- "$HPC_SCALE_PATTERN" "$SERVER_LOG" || true)
    architecture_hits=$(grep -Fic -- "Resolved architecture: $architecture" "$SERVER_LOG" || true)
    topology_hits=$(grep -Ec -- \
        "tensor_parallel_size=${tp}, pipeline_parallel_size=${PIPELINE_PARALLEL_SIZE}" \
        "$SERVER_LOG" || true)
    mla_backend_hits=$(grep -Ec -- "Using FLASH_ATTN_MLA attention backend" \
        "$SERVER_LOG" || true)

    if (( architecture_hits == 0 )); then
        die "server log does not prove architecture $architecture: $SERVER_LOG"
    fi
    if (( topology_hits == 0 )); then
        die "server log does not prove TP=$tp PP=$PIPELINE_PARALLEL_SIZE: $SERVER_LOG"
    fi

    if [[ "$path" == "gqa-hpc" ]]; then
        if (( fallbacks != 0 )); then
            die "GQA strict run emitted $fallbacks HPC fallback marker(s): $SERVER_LOG"
        fi
        if (( hits == 0 )); then
            die "GQA run completed without an HPC hit marker: $SERVER_LOG"
        fi
        if (( scale_hits == 0 )); then
            die "GQA run completed without the production YaRN scale marker: $SERVER_LOG"
        fi
        printf 'route=gqa-hpc\ntp=%s\npp=%s\narchitecture=%s\narchitecture_hits=%s\ntopology_hits=%s\nhpc_hits=%s\nhpc_fallbacks=%s\nsoftmax_scale_hits=%s\nstatus=verified\n' \
            "$tp" "$PIPELINE_PARALLEL_SIZE" "$architecture" \
            "$architecture_hits" "$topology_hits" "$hits" "$fallbacks" \
            "$scale_hits" >"$proof_file"
    else
        if (( hits != 0 )); then
            die "MQA/MLA control unexpectedly emitted $hits HPC hit marker(s): $SERVER_LOG"
        fi
        if (( mla_backend_hits == 0 )); then
            die "MQA/MLA run did not select FLASH_ATTN_MLA: $SERVER_LOG"
        fi
        printf 'route=mqa-mla\ntp=%s\npp=%s\narchitecture=%s\narchitecture_hits=%s\ntopology_hits=%s\nflash_attn_mla_hits=%s\nhpc_hits=%s\nstatus=verified_mla\n' \
            "$tp" "$PIPELINE_PARALLEL_SIZE" "$architecture" \
            "$architecture_hits" "$topology_hits" "$mla_backend_hits" "$hits" \
            >"$proof_file"
    fi
}

validate_benchmark_result() {
    local result_json=$1
    local expected_prompts=$2
    local expected_input_tokens=$((expected_prompts * INPUT_LEN))
    local expected_output_tokens=$((expected_prompts * OUTPUT_LEN))

    "$PYTHON" - "$result_json" "$expected_prompts" \
        "$expected_input_tokens" "$expected_output_tokens" <<'PY'
import json
import sys

result_path, expected_completed, expected_input, expected_output = sys.argv[1:]
expected = {
    "completed": int(expected_completed),
    "failed": 0,
    "total_input_tokens": int(expected_input),
    "total_output_tokens": int(expected_output),
}
with open(result_path, encoding="utf-8") as handle:
    result = json.load(handle)

errors = []
for key, expected_value in expected.items():
    actual_value = result.get(key)
    if actual_value != expected_value:
        errors.append(f"{key}={actual_value!r}, expected {expected_value!r}")
if errors:
    raise SystemExit(
        "invalid benchmark result: " + "; ".join(errors) + f" ({result_path})"
    )
print(
    "validated benchmark result: "
    f"completed={result['completed']} failed={result['failed']} "
    f"input_tokens={result['total_input_tokens']} "
    f"output_tokens={result['total_output_tokens']}"
)
PY
}

run_case() {
    local path=$1
    local tp=$2
    local pp=$PIPELINE_PARALLEL_SIZE
    local world_size=$((tp * pp))
    local architecture
    local case_name=${path}-tp${tp}
    if (( pp > 1 )); then
        case_name+=-pp${pp}
    fi
    local case_dir=$RUN_DIR/$case_name
    local hf_overrides
    local bench_log=$case_dir/bench.log
    local result_json=$case_dir/benchmark.json
    local bench_rc
    local launch_rc
    local -a serve_command
    local -a launch_command
    local -a bench_command

    case "$path" in
        gqa-hpc) architecture=$GQA_ARCHITECTURE ;;
        mqa-mla) architecture=$MQA_ARCHITECTURE ;;
        *) die "unsupported path: $path" ;;
    esac
    hf_overrides=$(printf '{"architectures":["%s"]}' "$architecture")

    serve_command=(
        "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_DIR"
        --host "$HOST"
        --port "$PORT"
        --served-model-name "$SERVED_MODEL_NAME"
        --hf-overrides "$hf_overrides"
        --model-impl vllm
        --trust-remote-code
        --tensor-parallel-size "$tp"
        --pipeline-parallel-size "$pp"
        --distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND"
        --dtype bfloat16
        --kv-cache-dtype auto
        --block-size "$BLOCK_SIZE"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --cpu-offload-gb "$CPU_OFFLOAD_GB"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --no-enable-prefix-caching
        --disable-cascade-attn
        --generation-config vllm
        --disable-log-stats
        --disable-uvicorn-access-log
        --seed "$SEED"
    )
    if [[ "$DISTRIBUTED_EXECUTOR_BACKEND" == mp ]]; then
        serve_command+=(
            --nnodes "$NNODES"
            --node-rank "$NODE_RANK"
            --master-addr "$MASTER_ADDR"
            --master-port "$MASTER_PORT"
        )
    fi
    if [[ "$ENABLE_CHUNKED_PREFILL" == 1 ]]; then
        serve_command+=(--enable-chunked-prefill)
    else
        serve_command+=(--no-enable-chunked-prefill)
    fi
    if [[ "$ENFORCE_EAGER" == 1 ]]; then
        serve_command+=(--enforce-eager)
    else
        serve_command+=(--no-enforce-eager)
    fi

    # `setsid` creates an isolated process group. `env -u` first clears any
    # inherited route controls; only the GQA case explicitly re-arms them.
    launch_command=(
        setsid env
        -u GQLA_HPC_STRICT
        -u GQLA_HPC_TRACE
        "PYTHONPATH=$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
        VLLM_WORKER_MULTIPROC_METHOD=spawn
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
        HF_HUB_OFFLINE=1
        TRANSFORMERS_OFFLINE=1
        TOKENIZERS_PARALLELISM=false
    )
    if [[ -n "$VLLM_HOST_IP" ]]; then
        launch_command+=("VLLM_HOST_IP=$VLLM_HOST_IP")
    fi
    if [[ -n "$PP_LAYER_PARTITION" ]]; then
        launch_command+=("VLLM_PP_LAYER_PARTITION=$PP_LAYER_PARTITION")
    fi
    if [[ "$path" == "gqa-hpc" ]]; then
        launch_command+=(GQLA_HPC_STRICT=1 GQLA_HPC_TRACE=1)
    fi
    launch_command+=("${serve_command[@]}")

    bench_command=(
        "$PYTHON" -m vllm.entrypoints.cli.main bench serve
        --backend openai
        --base-url "http://$HOST:$PORT"
        --endpoint /v1/completions
        --model "$SERVED_MODEL_NAME"
        --tokenizer "$MODEL_DIR"
        --served-model-name "$SERVED_MODEL_NAME"
        --dataset-name "$DATASET_NAME"
        --random-input-len "$INPUT_LEN"
        --random-output-len "$OUTPUT_LEN"
        --random-prefix-len 0
        --random-range-ratio 0
        --num-prompts "$NUM_PROMPTS"
        --num-warmups "$NUM_WARMUPS"
        --request-rate "$REQUEST_RATE"
        --burstiness 1.0
        --max-concurrency "$MAX_CONCURRENCY"
        --seed "$SEED"
        --temperature 0
        --ignore-eos
        --disable-shuffle
        --disable-tqdm
        --metric-percentiles "$METRIC_PERCENTILES"
        --percentile-metrics "$PERCENTILE_METRICS"
        --save-result
        --save-detailed
        --result-dir "$case_dir"
        --result-filename "$(basename "$result_json")"
        --label "$case_name"
        --metadata
        "run_id=$RUN_ID"
        "route=$path"
        "tp=$tp"
        "pp=$pp"
        "world_size=$world_size"
        "distributed_executor_backend=$DISTRIBUTED_EXECUTOR_BACKEND"
        "nnodes=$NNODES"
        "node_rank=$NODE_RANK"
        "vllm_host_ip=${VLLM_HOST_IP:-auto}"
        "pp_layer_partition=${PP_LAYER_PARTITION:-default}"
        "block_size=$BLOCK_SIZE"
        "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
        "cpu_offload_gb=$CPU_OFFLOAD_GB"
        "enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL"
        "input_len=$INPUT_LEN"
        "output_len=$OUTPUT_LEN"
        "num_prompts=$NUM_PROMPTS"
        "max_concurrency=$MAX_CONCURRENCY"
    )

    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[dry-run] case=$path tp=$tp pp=$pp world_size=$world_size"
        print_command "${launch_command[@]}"
        print_command "${bench_command[@]}"
        return
    fi

    mkdir -p "$case_dir"
    SERVER_LOG=$case_dir/server.log
    write_command "$case_dir/server.cmd" "${launch_command[@]}"
    write_command "$case_dir/bench.cmd" "${bench_command[@]}"
    port_is_free || die "$HOST:$PORT is already in use; refusing to stop an unrelated process"
    if [[ "$DISTRIBUTED_EXECUTOR_BACKEND" == mp && "$NNODES" -gt 1 ]]; then
        port_is_free_at "$MASTER_ADDR" "$MASTER_PORT" \
            || die "$MASTER_ADDR:$MASTER_PORT is unavailable for MP rendezvous"
    fi

    echo "[serve] route=$path tp=$tp pp=$pp world_size=$world_size log=$SERVER_LOG"
    "${launch_command[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    if ! capture_server_group; then
        launch_rc=0
        wait "$SERVER_PID" || launch_rc=$?
        SERVER_PID=
        echo "server failed before creating an isolated process group (rc=$launch_rc)" >&2
        tail -n 100 "$SERVER_LOG" >&2 || true
        return 1
    fi
    if ! wait_for_server; then
        tail -n 100 "$SERVER_LOG" >&2 || true
        stop_server || true
        return 1
    fi

    echo "[bench] route=$path tp=$tp pp=$pp result=$result_json"
    set +e
    "${bench_command[@]}" 2>&1 | tee "$bench_log"
    bench_rc=${PIPESTATUS[0]}
    set -e
    if (( bench_rc != 0 )); then
        echo "benchmark failed with rc=$bench_rc" >&2
        stop_server || true
        tail -n 100 "$SERVER_LOG" >&2 || true
        return "$bench_rc"
    fi
    [[ -s "$result_json" ]] || die "benchmark did not create $result_json"
    if ! validate_benchmark_result "$result_json" "$NUM_PROMPTS"; then
        echo "benchmark result failed completeness validation" >&2
        stop_server || true
        tail -n 100 "$SERVER_LOG" >&2 || true
        return 1
    fi
    if ! group_is_alive "$SERVER_PGID"; then
        echo "server exited unexpectedly after benchmark" >&2
        tail -n 100 "$SERVER_LOG" >&2 || true
        return 1
    fi

    stop_server
    verify_trace "$path" "$tp" "$architecture" "$case_dir/verification.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$path" "$tp" "$pp" verified "$result_json" "$SERVER_LOG" \
        >>"$RUN_DIR/matrix.tsv"
    echo "[ok] route=$path tp=$tp pp=$pp result=$result_json"
}

require_bool ENFORCE_EAGER "$ENFORCE_EAGER"
require_bool ENABLE_CHUNKED_PREFILL "$ENABLE_CHUNKED_PREFILL"
require_bool DRY_RUN "$DRY_RUN"
require_nonnegative_number CPU_OFFLOAD_GB "$CPU_OFFLOAD_GB"
for pair in \
    "PORT:$PORT" \
    "MASTER_PORT:$MASTER_PORT" \
    "PIPELINE_PARALLEL_SIZE:$PIPELINE_PARALLEL_SIZE" \
    "NNODES:$NNODES" \
    "MAX_MODEL_LEN:$MAX_MODEL_LEN" \
    "MAX_NUM_SEQS:$MAX_NUM_SEQS" \
    "MAX_NUM_BATCHED_TOKENS:$MAX_NUM_BATCHED_TOKENS" \
    "INPUT_LEN:$INPUT_LEN" \
    "OUTPUT_LEN:$OUTPUT_LEN" \
    "NUM_PROMPTS:$NUM_PROMPTS" \
    "NUM_WARMUPS:$NUM_WARMUPS" \
    "MAX_CONCURRENCY:$MAX_CONCURRENCY" \
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" \
    "STARTUP_TIMEOUT_SECONDS:$STARTUP_TIMEOUT_SECONDS" \
    "SHUTDOWN_TIMEOUT_SECONDS:$SHUTDOWN_TIMEOUT_SECONDS"; do
    require_positive_int "${pair%%:*}" "${pair#*:}"
done
require_nonnegative_int NODE_RANK "$NODE_RANK"
if (( NODE_RANK >= NNODES )); then
    die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"
fi
if (( INPUT_LEN + OUTPUT_LEN > MAX_MODEL_LEN )); then
    die "INPUT_LEN + OUTPUT_LEN exceeds MAX_MODEL_LEN"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "RUN_ID may contain only letters, digits, dot, underscore, and dash"
fi
[[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON"
command -v setsid >/dev/null || die "setsid is required for isolated process cleanup"

GQLA_ROOT=$(readlink -m -- "$GQLA_ROOT")
MODEL_DIR=$(readlink -f -- "$MODEL_DIR") || die "model directory does not exist"
OUTPUT_ROOT=$(readlink -m -- "$OUTPUT_ROOT")
BENCHMARK_ENTRYPOINT=$(readlink -f -- "$BENCHMARK_ENTRYPOINT") \
    || die "benchmark entrypoint does not exist: $BENCHMARK_ENTRYPOINT"
case "$OUTPUT_ROOT/" in
    "$GQLA_ROOT/outputs/"*) ;;
    *) die "OUTPUT_ROOT must remain under $GQLA_ROOT/outputs; got $OUTPUT_ROOT" ;;
esac
for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
    [[ -f "$MODEL_DIR/$required" ]] || die "missing converted-model file: $MODEL_DIR/$required"
done

IFS=',' read -r -a raw_tp_values <<<"$TPS"
IFS=',' read -r -a raw_path_values <<<"$PATHS"
tp_values=()
path_values=()
max_tp=1
for tp in "${raw_tp_values[@]}"; do
    tp=${tp//[[:space:]]/}
    case "$tp" in
        1|2|4|8) ;;
        *) die "TPS accepts only comma-separated 1,2,4,8; got $tp" ;;
    esac
    tp_values+=("$tp")
    (( tp > max_tp )) && max_tp=$tp
done
needs_gqa=0
for path in "${raw_path_values[@]}"; do
    path=${path//[[:space:]]/}
    case "$path" in
        gqa-hpc) needs_gqa=1 ;;
        mqa-mla) ;;
        *) die "PATHS accepts gqa-hpc and mqa-mla; got $path" ;;
    esac
    path_values+=("$path")
done
(( ${#tp_values[@]} > 0 )) || die "TPS is empty"
(( ${#path_values[@]} > 0 )) || die "PATHS is empty"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
vllm_version=$("$PYTHON" -c \
    'from importlib.metadata import version; print(version("vllm"))')
if [[ "$vllm_version" != "$EXPECTED_VLLM_VERSION" ]]; then
    die "expected vLLM $EXPECTED_VLLM_VERSION, found $vllm_version"
fi

if [[ "$DRY_RUN" == 1 && -n "$DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
    visible_gpus=0
else
    visible_gpus=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
fi
max_world_size=$((max_tp * PIPELINE_PARALLEL_SIZE))
max_local_world_size=0
for tp in "${tp_values[@]}"; do
    world_size=$((tp * PIPELINE_PARALLEL_SIZE))
    if (( world_size % NNODES != 0 )); then
        die "TP=$tp x PP=$PIPELINE_PARALLEL_SIZE gives world size $world_size, which is not divisible by NNODES=$NNODES"
    fi
    local_world_size=$((world_size / NNODES))
    (( local_world_size > max_local_world_size )) \
        && max_local_world_size=$local_world_size
done

if [[ -z "$DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
    if (( NNODES > 1 || visible_gpus >= max_world_size )); then
        DISTRIBUTED_EXECUTOR_BACKEND=mp
    else
        DISTRIBUTED_EXECUTOR_BACKEND=ray
    fi
fi
case "$DISTRIBUTED_EXECUTOR_BACKEND" in
    mp|ray) ;;
    *) die "DISTRIBUTED_EXECUTOR_BACKEND must be mp or ray; got $DISTRIBUTED_EXECUTOR_BACKEND" ;;
esac
if [[ "$DISTRIBUTED_EXECUTOR_BACKEND" == ray && "$NNODES" != 1 ]]; then
    die "Ray discovers nodes from its cluster; use NNODES=1 with backend=ray"
fi
if [[ "$DISTRIBUTED_EXECUTOR_BACKEND" == ray && -z "$RAY_ADDRESS" ]]; then
    die "backend=ray requires explicit RAY_ADDRESS"
fi
if [[ "$NODE_RANK" != 0 ]]; then
    die "benchmark driver must run with NODE_RANK=0; use the dedicated headless-worker launcher on other MP nodes"
fi

num_hidden_layers=$("$PYTHON" - "$MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["num_hidden_layers"])
PY
)
if [[ -n "$PP_LAYER_PARTITION" ]]; then
    "$PYTHON" - "$PP_LAYER_PARTITION" "$PIPELINE_PARALLEL_SIZE" \
        "$num_hidden_layers" <<'PY'
import sys

partition = sys.argv[1]
pp = int(sys.argv[2])
layers = int(sys.argv[3])
try:
    parts = [int(item) for item in partition.split(",")]
except ValueError as exc:
    raise SystemExit(f"invalid PP_LAYER_PARTITION={partition!r}: {exc}")
if len(parts) != pp:
    raise SystemExit(
        f"PP_LAYER_PARTITION has {len(parts)} entries; PP={pp} requires {pp}"
    )
if any(item <= 0 for item in parts):
    raise SystemExit("PP_LAYER_PARTITION entries must be positive")
if sum(parts) != layers:
    raise SystemExit(
        f"PP_LAYER_PARTITION sums to {sum(parts)}; model has {layers} layers"
    )
PY
fi

RUN_DIR=$OUTPUT_ROOT/$RUN_ID
ray_inventory=
if [[ "$DRY_RUN" == 0 ]]; then
    [[ ! -e "$RUN_DIR" ]] || die "run directory already exists: $RUN_DIR"
    if [[ "$DISTRIBUTED_EXECUTOR_BACKEND" == mp ]] \
        && (( visible_gpus < max_local_world_size )); then
        die "MP node needs $max_local_world_size visible GPUs, found $visible_gpus"
    fi
    if [[ "$DISTRIBUTED_EXECUTOR_BACKEND" == ray ]]; then
        if ! "$PYTHON" -c 'import ray' >/dev/null 2>&1; then
            die "backend=ray requires Ray in $PYTHON; use two-node MP launch or install Ray"
        fi
        if ! ray_inventory=$(inspect_ray_cluster \
            "$RAY_ADDRESS" "$max_world_size" "$max_tp" \
            "$PIPELINE_PARALLEL_SIZE"); then
            die "Ray cluster preflight failed"
        fi
    fi

    "$PYTHON" - "$GQA_ARCHITECTURE" "$MQA_ARCHITECTURE" "$needs_gqa" <<'PY'
import sys

import src.vllm_register_dsv  # noqa: F401
from vllm import ModelRegistry

gqa_arch, mqa_arch, needs_gqa = sys.argv[1], sys.argv[2], int(sys.argv[3])
registered = set(ModelRegistry.get_supported_archs())
missing = {gqa_arch, mqa_arch} - registered
if missing:
    raise SystemExit(f"unregistered vLLM architecture(s): {sorted(missing)}")
if needs_gqa:
    import hpc
    import torch

    schema = str(torch.ops.hpc.attention_decode_bf16.default._schema)
    if "softmax_scale" not in schema:
        raise SystemExit("installed HPC-Ops lacks attention_decode_bf16 softmax_scale")
    print(f"HPC-Ops {hpc.__version__}: runtime softmax_scale present")
PY

    mkdir -p "$RUN_DIR"
    if [[ -n "$ray_inventory" ]]; then
        printf '%s\n' "$ray_inventory" >"$RUN_DIR/ray_cluster.json"
    fi
    config_sha256=$(sha256sum "$MODEL_DIR/config.json" | awk '{print $1}')
    script_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')
    entrypoint_sha256=$(sha256sum "$BENCHMARK_ENTRYPOINT" | awk '{print $1}')
    {
        printf 'run_id=%q\n' "$RUN_ID"
        printf 'created_utc=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'script=%q\n' "${BASH_SOURCE[0]}"
        printf 'script_sha256=%q\n' "$script_sha256"
        printf 'entrypoint=%q\n' "$BENCHMARK_ENTRYPOINT"
        printf 'entrypoint_sha256=%q\n' "$entrypoint_sha256"
        printf 'python=%q\n' "$PYTHON"
        printf 'vllm_version=%q\n' "$vllm_version"
        printf 'model_dir=%q\n' "$MODEL_DIR"
        printf 'model_config_sha256=%q\n' "$config_sha256"
        printf 'cuda_visible_devices=%q\n' "${CUDA_VISIBLE_DEVICES:-all}"
        printf 'paths=%q\n' "$PATHS"
        printf 'tps=%q\n' "$TPS"
        printf 'pipeline_parallel_size=%q\n' "$PIPELINE_PARALLEL_SIZE"
        printf 'pp_layer_partition=%q\n' "${PP_LAYER_PARTITION:-default}"
        printf 'world_size_max=%q\n' "$max_world_size"
        printf 'distributed_executor_backend=%q\n' "$DISTRIBUTED_EXECUTOR_BACKEND"
        printf 'ray_address=%q\n' "${RAY_ADDRESS:-none}"
        printf 'nnodes=%q\n' "$NNODES"
        printf 'node_rank=%q\n' "$NODE_RANK"
        printf 'master_addr=%q\n' "$MASTER_ADDR"
        printf 'master_port=%q\n' "$MASTER_PORT"
        printf 'vllm_host_ip=%q\n' "${VLLM_HOST_IP:-auto}"
        printf 'block_size=%q\n' "$BLOCK_SIZE"
        printf 'input_len=%q\n' "$INPUT_LEN"
        printf 'output_len=%q\n' "$OUTPUT_LEN"
        printf 'num_prompts=%q\n' "$NUM_PROMPTS"
        printf 'num_warmups=%q\n' "$NUM_WARMUPS"
        printf 'max_concurrency=%q\n' "$MAX_CONCURRENCY"
        printf 'request_rate=%q\n' "$REQUEST_RATE"
        printf 'seed=%q\n' "$SEED"
        printf 'max_model_len=%q\n' "$MAX_MODEL_LEN"
        printf 'max_num_seqs=%q\n' "$MAX_NUM_SEQS"
        printf 'max_num_batched_tokens=%q\n' "$MAX_NUM_BATCHED_TOKENS"
        printf 'gpu_memory_utilization=%q\n' "$GPU_MEMORY_UTILIZATION"
        printf 'cpu_offload_gb=%q\n' "$CPU_OFFLOAD_GB"
        printf 'enforce_eager=%q\n' "$ENFORCE_EAGER"
        printf 'enable_chunked_prefill=%q\n' "$ENABLE_CHUNKED_PREFILL"
        printf 'vllm_execute_model_timeout_seconds=%q\n' \
            "$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    } >"$RUN_DIR/manifest.env"
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,memory.total \
            --format=csv,noheader >"$RUN_DIR/gpu_inventory.csv"
    fi
    printf 'route\ttp\tpp\tstatus\tresult_json\tserver_log\n' \
        >"$RUN_DIR/matrix.tsv"
else
    echo "[dry-run] no server will be started and no result files will be written"
fi

echo "[matrix] run_id=$RUN_ID model=$MODEL_DIR paths=$PATHS tp=$TPS pp=$PIPELINE_PARALLEL_SIZE backend=$DISTRIBUTED_EXECUTOR_BACKEND nnodes=$NNODES"
echo "[matrix] block=$BLOCK_SIZE input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS concurrency=$MAX_CONCURRENCY"
for path in "${path_values[@]}"; do
    for tp in "${tp_values[@]}"; do
        run_case "$path" "$tp"
    done
done

if [[ "$DRY_RUN" == 0 ]]; then
    echo "DSV3P1_G8_VLLM_BENCHMARK_MATRIX_OK run_dir=$RUN_DIR"
else
    echo "DSV3P1_G8_VLLM_BENCHMARK_DRY_RUN_OK"
fi
