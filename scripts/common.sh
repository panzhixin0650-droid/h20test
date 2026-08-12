#!/usr/bin/env bash

# Shared path and tool resolution for the portable H20 Table 2 harness.

h20_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
h20_patch_dir="$h20_repo_root/patches"
h20_output_dir=${GQLA_TABLE2_OUTPUT_DIR:-"$h20_repo_root/outputs/official"}
h20_source_record="$h20_output_dir/source_dir.txt"
h20_cuda_home=${CUDA_HOME:-/usr/local/cuda}
h20_visible_gpu=${GQLA_TABLE2_GPU:-0}
h20_expected_device_substring=${GQLA_TABLE2_EXPECT_DEVICE_SUBSTRING:-H20}

if [[ -n "${GQLA_TABLE2_PYTHON:-}" ]]; then
    h20_python=$GQLA_TABLE2_PYTHON
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    h20_python="$VIRTUAL_ENV/bin/python"
elif [[ -x "$h20_repo_root/.venv/bin/python" ]]; then
    h20_python="$h20_repo_root/.venv/bin/python"
else
    printf '%s\n' \
        'Set GQLA_TABLE2_PYTHON=/absolute/path/to/the/benchmark/venv/bin/python' \
        'or activate a virtual environment before running this script.' >&2
    exit 2
fi

if [[ -n "${GQLA_TABLE2_UV:-}" ]]; then
    h20_uv=$GQLA_TABLE2_UV
elif command -v uv >/dev/null 2>&1; then
    h20_uv=$(command -v uv)
else
    h20_uv=
fi

h20_tmp_parent=${GQLA_TABLE2_TMPDIR:-${TMPDIR:-/tmp}}
