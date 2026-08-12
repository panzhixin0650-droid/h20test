#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=common.sh
source "$script_dir/common.sh"

bench_python=$h20_python
bench_out=$h20_output_dir
bench_source_record=$h20_source_record
bench_preflight="$script_dir/gpu_preflight.py"
bench_old_patch="$h20_patch_dir/benchmark_mla_decode_table2.patch"
bench_dense_patch="$h20_patch_dir/benchmark_mla_decode_table2_dense_batch.patch"
bench_maincold_patch="$h20_patch_dir/benchmark_mla_decode_table2_batch_maincold.patch"
bench_fa3_mla_patch="$h20_patch_dir/benchmark_mla_decode_table2_fa3_mla_corrected_scale.patch"
bench_safe_sq2_patch="$h20_patch_dir/benchmark_mla_decode_table2_safe_sq2.patch"
bench_tp_patch="$h20_patch_dir/benchmark_mla_decode_table2_tp_sweep.patch"
bench_device_role=${GQLA_TABLE2_DEVICE_ROLE:-h20}
bench_visible_gpu=$h20_visible_gpu
bench_small_retry=${GQLA_TABLE2_SMALL_BATCH_RETRY:-0}
bench_batch_profile=${GQLA_TABLE2_BATCH_PROFILE:-0}
bench_rerun_all=${GQLA_TABLE2_RERUN_ALL:-0}
bench_tp_sweep=${GQLA_TABLE2_TP_SWEEP:-0}

if [[ "$bench_small_retry" != 0 && "$bench_small_retry" != 1 ]]; then
    printf 'GQLA_TABLE2_SMALL_BATCH_RETRY must be 0 or 1\n' >&2
    exit 2
fi
if [[ "$bench_batch_profile" != 0 && "$bench_batch_profile" != 1 ]]; then
    printf 'GQLA_TABLE2_BATCH_PROFILE must be 0 or 1\n' >&2
    exit 2
fi
if [[ "$bench_rerun_all" != 0 && "$bench_rerun_all" != 1 ]]; then
    printf 'GQLA_TABLE2_RERUN_ALL must be 0 or 1\n' >&2
    exit 2
fi
if [[ "$bench_tp_sweep" != 0 && "$bench_tp_sweep" != 1 ]]; then
    printf 'GQLA_TABLE2_TP_SWEEP must be 0 or 1\n' >&2
    exit 2
fi
bench_mode_count=$((bench_small_retry + bench_batch_profile + bench_rerun_all + bench_tp_sweep))
if (( bench_mode_count > 1 )); then
    printf 'Small-batch retry, batch profiling, corrected all-rerun, and TP-sweep modes are mutually exclusive\n' >&2
    exit 2
fi

case "$bench_device_role" in
    l20z-h100)
        bench_result="$bench_out/l20z_h100_dense_batch.json"
        bench_log="$bench_out/l20z_h100_dense_batch.log"
        bench_smoke_log="$bench_out/l20z_h100_dense_smoke.log"
        ;;
    h20)
        bench_result="$bench_out/h20_dense_batch.json"
        bench_log="$bench_out/h20_dense_batch.log"
        bench_smoke_log="$bench_out/h20_dense_smoke.log"
        ;;
    *)
        printf 'Unsupported GQLA_TABLE2_DEVICE_ROLE=%s\n' "$bench_device_role" >&2
        exit 2
        ;;
esac

if [[ "$bench_small_retry" == 1 ]]; then
    case "$bench_device_role" in
        l20z-h100)
            bench_result="$bench_out/l20z_h100_dense_smallbatch_retry.json"
            bench_log="$bench_out/l20z_h100_dense_smallbatch_retry.log"
            bench_smoke_log="$bench_out/l20z_h100_dense_smallbatch_retry_smoke.log"
            ;;
        h20)
            bench_result="$bench_out/h20_dense_smallbatch_retry.json"
            bench_log="$bench_out/h20_dense_smallbatch_retry.log"
            bench_smoke_log="$bench_out/h20_dense_smallbatch_retry_smoke.log"
            ;;
    esac
elif [[ "$bench_batch_profile" == 1 ]]; then
    case "$bench_device_role" in
        l20z-h100)
            bench_result="$bench_out/l20z_h100_batch64_128_maincold.json"
            bench_log="$bench_out/l20z_h100_batch64_128_maincold.log"
            bench_smoke_log="$bench_out/l20z_h100_batch64_128_maincold_smoke.log"
            ;;
        h20)
            bench_result="$bench_out/h20_batch64_128_maincold.json"
            bench_log="$bench_out/h20_batch64_128_maincold.log"
            bench_smoke_log="$bench_out/h20_batch64_128_maincold_smoke.log"
            ;;
    esac
elif [[ "$bench_rerun_all" == 1 ]]; then
    case "$bench_device_role" in
        l20z-h100)
            bench_result="$bench_out/l20z_h100_table2_all_batch64_128_safe_sq2.json"
            bench_log="$bench_out/l20z_h100_table2_all_batch64_128_safe_sq2.log"
            bench_smoke_log="$bench_out/l20z_h100_table2_all_batch64_128_safe_sq2_smoke.log"
            ;;
        h20)
            bench_result="$bench_out/h20_table2_all_batch64_128_safe_sq2.json"
            bench_log="$bench_out/h20_table2_all_batch64_128_safe_sq2.log"
            bench_smoke_log="$bench_out/h20_table2_all_batch64_128_safe_sq2_smoke.log"
            ;;
    esac
elif [[ "$bench_tp_sweep" == 1 ]]; then
    case "$bench_device_role" in
        l20z-h100)
            bench_result="$bench_out/l20z_h100_table2_tp1_2_4_8.json"
            bench_log="$bench_out/l20z_h100_table2_tp1_2_4_8.log"
            bench_smoke_log="$bench_out/l20z_h100_table2_tp1_2_4_8_smoke.log"
            ;;
        h20)
            bench_result="$bench_out/h20_table2_tp1_2_4_8.json"
            bench_log="$bench_out/h20_table2_tp1_2_4_8.log"
            bench_smoke_log="$bench_out/h20_table2_tp1_2_4_8_smoke.log"
            ;;
    esac
fi

on_error() {
    bench_rc=$?
    printf 'DENSE_BATCH_RUN_FAILED rc=%s line=%s\n' \
        "$bench_rc" "${BASH_LINENO[0]}" >&2
    printf 'Log: %s\n' "$bench_log" >&2
    exit "$bench_rc"
}
trap on_error ERR

mkdir -p "$bench_out"
test -x "$bench_python"
test -f "$bench_preflight"
test -s "$bench_source_record"
test -s "$bench_old_patch"
test -s "$bench_dense_patch"
test -s "$bench_maincold_patch"
test -s "$bench_fa3_mla_patch"
test -s "$bench_safe_sq2_patch"
test -s "$bench_tp_patch"
command -v sha256sum >/dev/null
(cd "$h20_repo_root" && sha256sum -c patches/SHA256SUMS)

CUDA_VISIBLE_DEVICES="$bench_visible_gpu" PYTHONUNBUFFERED=1 \
    "$bench_python" -u "$bench_preflight" \
        --extensions \
        --expect-device-substring "$h20_expected_device_substring"

IFS= read -r bench_tmp < "$bench_source_record"
bench_repo="$bench_tmp/flash-attention"
bench_hopper="$bench_repo/hopper"
bench_script="$bench_hopper/benchmark_mla_decode.py"

if [[ ! -d "$bench_repo/.git" || ! -f "$bench_script" ]]; then
    printf 'Node-local official source is absent: %s\n' "$bench_repo" >&2
    printf 'Run build_official_kernels.sh on this GPU node first.\n' >&2
    exit 4
fi

bench_commit=$(git -C "$bench_repo" rev-parse HEAD)
if [[ "$bench_commit" != a369df707e1980fb328abcc1733e3457ec10155f ]]; then
    printf 'Unexpected flash-attention commit: %s\n' "$bench_commit" >&2
    exit 3
fi

bench_tp_applied=0
if git -C "$bench_repo" apply --reverse --check "$bench_tp_patch" >/dev/null 2>&1; then
    bench_tp_applied=1
    printf 'TP-sweep official benchmark patch is already applied\n'
elif git -C "$bench_repo" apply --reverse --check "$bench_safe_sq2_patch" >/dev/null 2>&1; then
    printf 'Safe-sq2 official benchmark patch is already applied\n'
else
    if git -C "$bench_repo" apply --reverse --check "$bench_fa3_mla_patch" >/dev/null 2>&1; then
        printf 'FA3-MLA/corrected-scale base patch is already applied\n'
    elif git -C "$bench_repo" apply --reverse --check "$bench_maincold_patch" >/dev/null 2>&1; then
        git -C "$bench_repo" apply --reverse "$bench_maincold_patch"
        git -C "$bench_repo" apply --check "$bench_fa3_mla_patch"
        git -C "$bench_repo" apply "$bench_fa3_mla_patch"
        printf 'Upgraded official benchmark patch from main-cold to FA3-MLA/corrected-scale\n'
    elif git -C "$bench_repo" apply --reverse --check "$bench_dense_patch" >/dev/null 2>&1; then
        git -C "$bench_repo" apply --reverse "$bench_dense_patch"
        git -C "$bench_repo" apply --check "$bench_fa3_mla_patch"
        git -C "$bench_repo" apply "$bench_fa3_mla_patch"
        printf 'Upgraded official benchmark patch from dense/batch to FA3-MLA/corrected-scale\n'
    elif git -C "$bench_repo" apply --reverse --check "$bench_old_patch" >/dev/null 2>&1; then
        git -C "$bench_repo" apply --reverse "$bench_old_patch"
        git -C "$bench_repo" apply --check "$bench_fa3_mla_patch"
        git -C "$bench_repo" apply "$bench_fa3_mla_patch"
        printf 'Upgraded official benchmark patch from paged-only to FA3-MLA/corrected-scale\n'
    elif git -C "$bench_repo" apply --check "$bench_fa3_mla_patch" >/dev/null 2>&1; then
        git -C "$bench_repo" apply "$bench_fa3_mla_patch"
        printf 'Applied FA3-MLA/corrected-scale base patch to official benchmark_mla_decode.py\n'
    else
        printf 'Official benchmark has unexpected local changes; refusing to overwrite it.\n' >&2
        git -C "$bench_repo" status --short -- hopper/benchmark_mla_decode.py >&2
        exit 5
    fi
    git -C "$bench_repo" apply --check "$bench_safe_sq2_patch"
    git -C "$bench_repo" apply "$bench_safe_sq2_patch"
    printf 'Applied safe-sq2 scheduler/correctness patch\n'
fi

if [[ "$bench_rerun_all" == 1 && "$bench_tp_applied" == 1 ]]; then
    git -C "$bench_repo" apply --reverse --check "$bench_tp_patch"
    git -C "$bench_repo" apply --reverse "$bench_tp_patch"
    bench_tp_applied=0
    printf 'Reverted TP-sweep patch to restore the schema-v5 safe-sq2 benchmark\n'
fi

if [[ "$bench_tp_sweep" == 1 && "$bench_tp_applied" == 0 ]]; then
    git -C "$bench_repo" apply --check "$bench_tp_patch"
    git -C "$bench_repo" apply "$bench_tp_patch"
    printf 'Applied TP=1/2/4/8 rank-local sweep patch\n'
fi

git -C "$bench_repo" diff --check -- hopper/benchmark_mla_decode.py
"$bench_python" -m py_compile "$bench_script"

printf 'device_role=%s visible_gpu=%s\n' "$bench_device_role" "$bench_visible_gpu"
printf 'small_batch_retry=%s\n' "$bench_small_retry"
printf 'batch_profile=%s\n' "$bench_batch_profile"
printf 'rerun_all=%s\n' "$bench_rerun_all"
printf 'tp_sweep=%s\n' "$bench_tp_sweep"
printf 'official_script=%s\n' "$bench_script"
printf 'result_json=%s\n' "$bench_result"

bench_smoke_args=(--smoke-only)
if [[ "$bench_rerun_all" == 1 || "$bench_tp_sweep" == 1 ]]; then
    bench_smoke_args+=(--smoke-fa3-mla)
fi
if [[ "$bench_tp_sweep" == 1 ]]; then
    bench_smoke_args+=(--smoke-tp --tp-sizes 1 2 4 8)
fi
CUDA_VISIBLE_DEVICES="$bench_visible_gpu" \
PYTHONUNBUFFERED=1 \
    "$bench_python" -P -u "$bench_script" "${bench_smoke_args[@]}" \
    2>&1 | tee "$bench_smoke_log"

if [[ "$bench_tp_sweep" == 1 ]]; then
    bench_args=(
        --device-role "$bench_device_role"
        --output "$bench_result"
        --batch-only
        --gqa-layout both
        --batch-sweep
        --tp-sweep
        --tp-sizes 1 2 4 8
        --include-fa3-mla
        --batch-sizes 128
        --profile-batch
        --batch-profile-sizes 128
        --batch-profile-num-tests 10
        --batch-sweep-warmup-ms 2
        --batch-sweep-rep-ms 8
        --batch-final-warmup-ms 10
        --batch-final-rep-ms 50
        --batch-pre-final-sleep-s 1
    )
elif [[ "$bench_rerun_all" == 1 ]]; then
    bench_args=(
        --device-role "$bench_device_role"
        --output "$bench_result"
        --batch-only
        --gqa-layout both
        --batch-sweep
        --include-fa3-mla
        --batch-sizes 64 128
        --profile-batch
        --batch-profile-sizes 64 128
        --batch-profile-num-tests 10
        --batch-sweep-warmup-ms 2
        --batch-sweep-rep-ms 8
        --batch-final-warmup-ms 10
        --batch-final-rep-ms 50
        --batch-pre-final-sleep-s 1
    )
elif [[ "$bench_batch_profile" == 1 ]]; then
    bench_args=(
        --device-role "$bench_device_role"
        --output "$bench_result"
        --batch-only
        --gqa-layout both
        --batch-sweep
        --batch-sizes 64 128
        --profile-batch
        --batch-profile-sizes 64 128
        --batch-profile-num-tests 10
        --batch-sweep-warmup-ms 2
        --batch-sweep-rep-ms 8
        --batch-final-warmup-ms 10
        --batch-final-rep-ms 50
        --batch-pre-final-sleep-s 1
    )
elif [[ "$bench_small_retry" == 1 ]]; then
    bench_args=(
        --device-role "$bench_device_role"
        --output "$bench_result"
        --gqa-only
        --gqa-layout both
        --batch-sweep
        --batch-sizes 1 2 4
        --sweep-warmup-ms 5
        --sweep-rep-ms 20
        --final-warmup-ms 20
        --final-rep-ms 100
        --pre-final-sleep-s 2
        --batch-sweep-warmup-ms 2
        --batch-sweep-rep-ms 8
        --batch-final-warmup-ms 10
        --batch-final-rep-ms 50
        --batch-pre-final-sleep-s 1
    )
else
    bench_args=(
        --device-role "$bench_device_role"
        --output "$bench_result"
        --gqa-layout both
        --batch-sweep
        --batch-sizes 1 2 4 8 16 32 64 128
        --hot
        --profile
        --sweep-warmup-ms 2
        --sweep-rep-ms 8
        --final-warmup-ms 20
        --final-rep-ms 100
        --hot-rep-ms 5
        --profile-num-tests 5
        --pre-final-sleep-s 1
        --batch-sweep-warmup-ms 1
        --batch-sweep-rep-ms 4
        --batch-final-warmup-ms 5
        --batch-final-rep-ms 20
        --batch-pre-final-sleep-s 0.2
    )
fi

CUDA_VISIBLE_DEVICES="$bench_visible_gpu" \
PYTHONUNBUFFERED=1 \
    "$bench_python" -P -u "$bench_script" "${bench_args[@]}" \
    2>&1 | tee "$bench_log"

printf 'DENSE_BATCH_RUN_OK\n'
printf 'Result: %s\n' "$bench_result"
printf 'Log: %s\n' "$bench_log"
