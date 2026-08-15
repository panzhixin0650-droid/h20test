# H20 vLLM deployment bundle

This directory transfers the GQLA vLLM plugin, H20 environment bootstrap, and
benchmark launchers. It does **not** contain or modify the converted model.

The deployment script checks that the converted checkpoint is already at:

```text
/mnt/tidalfs-alwl01/task/236362/GQLA/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract
```

It copies the bundled plugin to `GQLA/code/GQLA_preprint` and unpacks a
SHA256-verified HPC-Ops source archive at local revision `f85ce84`. That source
contains both local commits required by this integration:

- `83165c3`: GQLA QK192/V128 BF16 decode support, based on Tencent `1cd3329`.
- `f85ce84`: runtime softmax scale and build/bootstrap fixes.

The archive contains source only—no wheel, build tree, or model data. Keeping
the exact 4 MB source archive in this transit branch avoids depending on a
local-only commit that Tencent GitHub cannot serve.

## Clone and deploy

Run on the H20 machine:

```bash
mkdir -p /mnt/tidalfs-alwl01/task/236362/GQLA/code
cd /mnt/tidalfs-alwl01/task/236362/GQLA/code

git clone --branch deploy/h20-vllm-20260815 --single-branch \
  https://github.com/panzhixin0650-droid/h20test.git h20test

bash h20test/deploy/h20-vllm/deploy_h20_vllm.sh
```

If the machine's Git/GnuTLS client cannot complete a GitHub TLS handshake,
download the same branch through GitHub codeload instead:

```bash
mkdir -p /mnt/tidalfs-alwl01/task/236362/GQLA/code/h20test
cd /mnt/tidalfs-alwl01/task/236362/GQLA/code

curl -fL --retry 10 --retry-delay 2 --connect-timeout 30 \
  -o /tmp/h20test-h20-vllm.tar.gz \
  'https://codeload.github.com/panzhixin0650-droid/h20test/tar.gz/refs/heads/deploy/h20-vllm-20260815'

tar -xzf /tmp/h20test-h20-vllm.tar.gz \
  --strip-components=1 -C h20test

bash h20test/deploy/h20-vllm/deploy_h20_vllm.sh
```

The H20 deployment no longer needs a second GitHub request for HPC-Ops: its
verified source archive is already inside this transit bundle.

It also contains a SHA256-verified `uv 0.9.3` x86-64 executable. The deployer
installs it under `GQLA_preprint/tools/uv`, so bootstrap does not depend on the
often-blocked GitHub Releases object store before Python is available.

A portable, SHA256-verified CPython 3.12.12 runtime is bundled as well and is
installed under `GQLA/envs/python`. Therefore repairing a copied venv also
requires no GitHub Releases download.

Deploy and immediately create/check the environment in one command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_INSTALL=1 \
  bash h20test/deploy/h20-vllm/deploy_h20_vllm.sh
```

The environment is installed with `uv` under
`/mnt/tidalfs-alwl01/task/236362/GQLA/envs/venv-py312`. Re-running the command
reuses the environment, wheel cache, source tree, and compiled HPC wheel.

## Benchmark after install

The paths above are defaults only. For a checkpoint rooted at
`/mnt/public03/task/236362/GQLA`, pass `GQLA_ROOT` (and optionally
`MODEL_DIR`) to both deploy and benchmark commands.

Run the full single-node matrix—fresh servers for `2K / 8K / 16K` and both
MLA/GQLA routes—in one command:

```bash
cd /mnt/public03/task/236362/GQLA/code/GQLA_preprint
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  GQLA_ROOT=/mnt/public03/task/236362/GQLA \
  MATRIX_RUN_ID_BASE=h20-tp8-all-001 \
  bash scripts/benchmark_dsv3p1_g8_h20_tp8_all.sh
```

Use `PROFILES=2k,8k`, `PATHS=gqa-hpc`, or
`PROFILES=2k,8k,16k,16k-b20` to select a smaller or extended matrix.

Both 2K paths, sequentially:

```bash
cd /mnt/tidalfs-alwl01/task/236362/GQLA/code/GQLA_preprint
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  RUN_ID=h20-tp8-2k-001 \
  bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

GQLA/HPC only:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PATHS=gqa-hpc RUN_ID=h20-gqla-tp8-2k-001 \
  bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

Use a new `RUN_ID` for a fresh result. The launcher will also append an
`attemptN` suffix when a base run ID already exists.
