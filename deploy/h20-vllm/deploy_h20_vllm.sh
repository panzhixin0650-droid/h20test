#!/usr/bin/env bash
set -Eeuo pipefail

# Materialize the exact GQLA/vLLM deployment sources used by the H20 TP=8
# benchmarks. The converted checkpoint is deliberately not copied or changed.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
BUNDLE_DIR="$SCRIPT_DIR/GQLA_preprint"
PATCH_FILE="$REPO_ROOT/patches/0001-Add-runtime-attention-scale-for-GQLA-serving.patch"
HPC_BUNDLE_ARCHIVE="$SCRIPT_DIR/hpc-ops-f85ce84.tar.gz"
HPC_BUNDLE_SHA256=dd5a668ee454683a81d8857988e4285e9c1558d39cec92664c981e7b8c562e7e
UV_BUNDLE_ARCHIVE="$SCRIPT_DIR/uv-0.9.3-x86_64-unknown-linux-gnu.xz"
UV_BUNDLE_SHA256=dd572805f351a07266ba464a15fe134dbb32e082f048bae13e4f8f991e2b7b69
UV_BINARY_SHA256=d7198eb91c8b6a7ead6eb5b2e7aca159124695c8783ff2df708f6864bc574bbf
PYTHON_BUNDLE_ARCHIVE="$SCRIPT_DIR/cpython-3.12.12-linux-x86_64-gnu.tar.xz"
PYTHON_BUNDLE_SHA256=33febdd58c2aa359635c4e0381f5c00c7a4a25860e57469e36a744c0eeabe70d
PYTHON_BINARY_SHA256=4304de9fdcd8465bbf4bf0814bb7abadd6d33971bab26ae38d1080c355fec983

GQLA_ROOT=${GQLA_ROOT:-/mnt/tidalfs-alwl01/task/236362/GQLA}
CODE_ROOT=${CODE_ROOT:-$GQLA_ROOT/code}
ENV_ROOT=${ENV_ROOT:-$GQLA_ROOT/envs}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract}
GQLA_DST=${GQLA_DST:-$CODE_ROOT/GQLA_preprint}
HPC_DST=${HPC_DST:-$CODE_ROOT/hpc-ops}
HPC_REMOTE=${HPC_REMOTE:-https://github.com/Tencent/hpc-ops.git}
HPC_BASE_COMMIT=83165c3f7d1f2a4aa0bd1f8c0f37fab771b5190b
HPC_SOURCE_COMMIT=f85ce8457ce6ef46c4c89736576792f14533cc48
HPC_ARCHIVE_URL=${HPC_ARCHIVE_URL:-https://codeload.github.com/Tencent/hpc-ops/tar.gz/$HPC_BASE_COMMIT}
RUN_INSTALL=${RUN_INSTALL:-0}

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -f "$MODEL_DIR/config.json" ]] || \
  die "converted model is missing: $MODEL_DIR/config.json"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || \
  die "converted model index is missing: $MODEL_DIR/model.safetensors.index.json"
[[ -f "$BUNDLE_DIR/pyproject.toml" ]] || die "incomplete GQLA source bundle"
[[ -f "$PATCH_FILE" ]] || die "HPC-Ops patch is missing: $PATCH_FILE"
command -v git >/dev/null 2>&1 || die "git is required"

mkdir -p "$CODE_ROOT" "$GQLA_DST"
cp -a "$BUNDLE_DIR/." "$GQLA_DST/"
chmod +x "$GQLA_DST"/scripts/*.sh
echo "[deploy] GQLA source -> $GQLA_DST"

if [[ "$(uname -m)" == x86_64 ]] && [[ -f "$UV_BUNDLE_ARCHIVE" ]]; then
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
  command -v xz >/dev/null 2>&1 || die "xz is required to unpack bundled uv"
  uv_archive_actual_sha256=$(sha256sum "$UV_BUNDLE_ARCHIVE" | awk '{print $1}')
  [[ "$uv_archive_actual_sha256" == "$UV_BUNDLE_SHA256" ]] || \
    die "bundled uv archive SHA256 mismatch"
  uv_bundle_tmp=$(mktemp "$CODE_ROOT/.uv-bundle.XXXXXX")
  xz -dc "$UV_BUNDLE_ARCHIVE" >"$uv_bundle_tmp"
  uv_binary_actual_sha256=$(sha256sum "$uv_bundle_tmp" | awk '{print $1}')
  [[ "$uv_binary_actual_sha256" == "$UV_BINARY_SHA256" ]] || \
    die "bundled uv binary SHA256 mismatch"
  mkdir -p "$GQLA_DST/tools"
  install -m 0755 "$uv_bundle_tmp" "$GQLA_DST/tools/uv"
  rm -f -- "$uv_bundle_tmp"
  [[ "$("$GQLA_DST/tools/uv" --version)" == "uv 0.9.3" ]] || \
    die "bundled uv failed its version check"
  echo "[deploy] installed verified bundled uv 0.9.3 -> $GQLA_DST/tools/uv"
fi

if [[ "$(uname -m)" == x86_64 ]] && [[ -f "$PYTHON_BUNDLE_ARCHIVE" ]]; then
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
  command -v tar >/dev/null 2>&1 || die "tar is required"
  python_archive_actual_sha256=$(sha256sum "$PYTHON_BUNDLE_ARCHIVE" | awk '{print $1}')
  [[ "$python_archive_actual_sha256" == "$PYTHON_BUNDLE_SHA256" ]] || \
    die "bundled CPython archive SHA256 mismatch"

  python_install_root=$ENV_ROOT/python
  python_install_dir=$python_install_root/cpython-3.12.12-linux-x86_64-gnu
  python_binary=$python_install_dir/bin/python3.12
  python_ready=0
  if [[ -x "$python_binary" ]]; then
    python_binary_actual_sha256=$(sha256sum "$python_binary" | awk '{print $1}')
    if [[ "$python_binary_actual_sha256" == "$PYTHON_BINARY_SHA256" ]]; then
      python_ready=1
    fi
  fi

  if [[ "$python_ready" == 0 ]]; then
    mkdir -p "$python_install_root"
    python_bundle_tmp=$(mktemp -d "$ENV_ROOT/.python-bundle.XXXXXX")
    tar -xJf "$PYTHON_BUNDLE_ARCHIVE" -C "$python_bundle_tmp"
    python_staged_dir=$python_bundle_tmp/cpython-3.12.12-linux-x86_64-gnu
    python_staged_binary=$python_staged_dir/bin/python3.12
    [[ -x "$python_staged_binary" ]] || die "bundled CPython executable is missing"
    python_binary_actual_sha256=$(sha256sum "$python_staged_binary" | awk '{print $1}')
    [[ "$python_binary_actual_sha256" == "$PYTHON_BINARY_SHA256" ]] || \
      die "bundled CPython binary SHA256 mismatch"
    if [[ -e "$python_install_dir" ]]; then
      mv "$python_install_dir" "$python_install_dir.previous.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    mv "$python_staged_dir" "$python_install_dir"
    rmdir "$python_bundle_tmp"
  fi
  [[ "$("$python_binary" --version)" == "Python 3.12.12" ]] || \
    die "bundled CPython failed its version check"
  echo "[deploy] installed verified bundled CPython 3.12.12 -> $python_install_dir"
fi

if [[ ! -d "$HPC_DST/.git" ]]; then
  [[ ! -e "$HPC_DST" || -z "$(find "$HPC_DST" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || \
    die "$HPC_DST exists and is not an empty Git repository"
  [[ ! -d "$HPC_DST" ]] || rmdir "$HPC_DST"

  hpc_fetch_tmp=$(mktemp -d "$CODE_ROOT/.hpc-ops-fetch.XXXXXX")
  hpc_fetch_src="$hpc_fetch_tmp/source"
  hpc_fetch_archive="$hpc_fetch_tmp/hpc-ops.tar.gz"

  if [[ -f "$HPC_BUNDLE_ARCHIVE" ]]; then
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
    command -v tar >/dev/null 2>&1 || die "tar is required"
    hpc_bundle_actual_sha256=$(sha256sum "$HPC_BUNDLE_ARCHIVE" | awk '{print $1}')
    [[ "$hpc_bundle_actual_sha256" == "$HPC_BUNDLE_SHA256" ]] || \
      die "bundled HPC-Ops archive SHA256 mismatch"
    mkdir -p "$hpc_fetch_src"
    tar -xzf "$HPC_BUNDLE_ARCHIVE" --strip-components=1 -C "$hpc_fetch_src"
    git -C "$hpc_fetch_src" init -q
    printf '%s\n' "$HPC_SOURCE_COMMIT" >"$hpc_fetch_src/.gqla_hpc_source_commit"
    echo "[deploy] unpacked verified bundled HPC-Ops source $HPC_SOURCE_COMMIT"
  elif (
    set -Eeuo pipefail
    mkdir -p "$hpc_fetch_src"
    git -C "$hpc_fetch_src" init -q
    git -C "$hpc_fetch_src" remote add origin "$HPC_REMOTE"
    git -C "$hpc_fetch_src" fetch --depth=1 origin "$HPC_BASE_COMMIT" || exit 1
    git -C "$hpc_fetch_src" checkout -q --detach FETCH_HEAD
  ); then
    echo "[deploy] fetched HPC-Ops with Git"
  else
    echo "[deploy] Git fetch failed; retrying HPC-Ops through codeload archive" >&2
    command -v curl >/dev/null 2>&1 || die "curl is required for the archive fallback"
    command -v tar >/dev/null 2>&1 || die "tar is required for the archive fallback"
    rm -rf -- "$hpc_fetch_src"
    mkdir -p "$hpc_fetch_src"
    curl -fL \
      --retry 10 --retry-delay 2 --connect-timeout 30 --max-time 1800 \
      -o "$hpc_fetch_archive" "$HPC_ARCHIVE_URL"
    tar -xzf "$hpc_fetch_archive" --strip-components=1 -C "$hpc_fetch_src"
    git -C "$hpc_fetch_src" init -q
    printf '%s\n' "$HPC_BASE_COMMIT" >"$hpc_fetch_src/.gqla_hpc_base_commit"
    echo "[deploy] fetched pinned HPC-Ops codeload archive"
  fi

  mv "$hpc_fetch_src" "$HPC_DST"
  rm -rf -- "$hpc_fetch_tmp"
  echo "[deploy] HPC-Ops baseline -> $HPC_DST"
fi

if git -C "$HPC_DST" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "[deploy] HPC-Ops GQLA runtime-scale patch is already applied"
else
  hpc_base_ok=0
  current_commit=$(git -C "$HPC_DST" rev-parse --verify HEAD 2>/dev/null || true)
  if [[ "$current_commit" == "$HPC_BASE_COMMIT" ]]; then
    hpc_base_ok=1
  elif [[ -f "$HPC_DST/.gqla_hpc_base_commit" ]] && \
       [[ "$(<"$HPC_DST/.gqla_hpc_base_commit")" == "$HPC_BASE_COMMIT" ]]; then
    hpc_base_ok=1
  fi
  [[ "$hpc_base_ok" == "1" ]] || \
    die "HPC-Ops is not the expected pinned source $HPC_BASE_COMMIT"
  [[ -z "$(git -C "$HPC_DST" status --porcelain --untracked-files=no)" ]] || \
    die "HPC-Ops has tracked local changes; refusing to overwrite them"
  git -C "$HPC_DST" apply --check "$PATCH_FILE"
  git -C "$HPC_DST" apply "$PATCH_FILE"
  echo "[deploy] applied HPC-Ops GQLA runtime-scale patch"
fi

cat >"$CODE_ROOT/H20_VLLM_DEPLOYMENT.txt" <<EOF
GQLA source: $GQLA_DST
HPC-Ops source: $HPC_DST
HPC-Ops base: $HPC_BASE_COMMIT
HPC-Ops source revision: $HPC_SOURCE_COMMIT
HPC-Ops patch: $(basename "$PATCH_FILE")
Converted model: $MODEL_DIR
EOF

echo "[deploy] source deployment complete"

if [[ "$RUN_INSTALL" == "1" ]]; then
  echo "[deploy] starting uv environment install and H20 preflight"
  cd "$GQLA_DST"
  exec env \
    GQLA_ROOT="$GQLA_ROOT" \
    MODEL_DIR="$MODEL_DIR" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    INSTALL_ONLY=1 \
    bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
fi

cat <<EOF

Next step:
  cd $GQLA_DST
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 INSTALL_ONLY=1 \\
    bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
EOF
