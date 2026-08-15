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
  mkdir -p "$HPC_DST"
  git -C "$HPC_DST" init -q
  git -C "$HPC_DST" remote add origin "$HPC_REMOTE"
  git -C "$HPC_DST" fetch --depth=1 origin "$HPC_BASE_COMMIT"
  git -C "$HPC_DST" checkout -q --detach FETCH_HEAD
  echo "[deploy] HPC-Ops baseline -> $HPC_DST"
fi

if git -C "$HPC_DST" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "[deploy] HPC-Ops GQLA runtime-scale patch is already applied"
else
  current_commit=$(git -C "$HPC_DST" rev-parse HEAD)
  [[ "$current_commit" == "$HPC_BASE_COMMIT" ]] || \
    die "HPC-Ops HEAD is $current_commit; expected $HPC_BASE_COMMIT"
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
