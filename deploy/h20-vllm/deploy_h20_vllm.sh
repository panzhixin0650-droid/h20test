#!/usr/bin/env bash
set -Eeuo pipefail

# Materialize the exact GQLA/vLLM deployment sources used by the H20 TP=8
# benchmarks. The converted checkpoint is deliberately not copied or changed.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
BUNDLE_DIR="$SCRIPT_DIR/GQLA_preprint"
PATCH_FILE="$REPO_ROOT/patches/0001-Add-runtime-attention-scale-for-GQLA-serving.patch"

GQLA_ROOT=${GQLA_ROOT:-/mnt/tidalfs-alwl01/task/236362/GQLA}
CODE_ROOT=${CODE_ROOT:-$GQLA_ROOT/code}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract}
GQLA_DST=${GQLA_DST:-$CODE_ROOT/GQLA_preprint}
HPC_DST=${HPC_DST:-$CODE_ROOT/hpc-ops}
HPC_REMOTE=${HPC_REMOTE:-https://github.com/Tencent/hpc-ops.git}
HPC_BASE_COMMIT=83165c3f7d1f2a4aa0bd1f8c0f37fab771b5190b
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

if [[ ! -d "$HPC_DST/.git" ]]; then
  [[ ! -e "$HPC_DST" || -z "$(find "$HPC_DST" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || \
    die "$HPC_DST exists and is not an empty Git repository"
  [[ ! -d "$HPC_DST" ]] || rmdir "$HPC_DST"

  hpc_fetch_tmp=$(mktemp -d "$CODE_ROOT/.hpc-ops-fetch.XXXXXX")
  hpc_fetch_src="$hpc_fetch_tmp/source"
  hpc_fetch_archive="$hpc_fetch_tmp/hpc-ops.tar.gz"

  if (
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
