# H20 vLLM deployment bundle

This directory transfers the GQLA vLLM plugin, H20 environment bootstrap, and
benchmark launchers. It does **not** contain or modify the converted model.

The deployment script checks that the converted checkpoint is already at:

```text
/mnt/tidalfs-alwl01/task/236362/GQLA/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract
```

It copies the bundled plugin to `GQLA/code/GQLA_preprint`, fetches Tencent
HPC-Ops at commit `83165c3f7d1f2a4aa0bd1f8c0f37fab771b5190b`, and applies the
runtime-softmax-scale patch stored in this repository.

## Clone and deploy

Run on the H20 machine:

```bash
mkdir -p /mnt/tidalfs-alwl01/task/236362/GQLA/code
cd /mnt/tidalfs-alwl01/task/236362/GQLA/code

git clone --branch deploy/h20-vllm-20260815 --single-branch \
  https://github.com/panzhixin0650-droid/h20test.git h20test

bash h20test/deploy/h20-vllm/deploy_h20_vllm.sh
```

Deploy and immediately create/check the environment in one command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_INSTALL=1 \
  bash h20test/deploy/h20-vllm/deploy_h20_vllm.sh
```

The environment is installed with `uv` under
`/mnt/tidalfs-alwl01/task/236362/GQLA/envs/venv-py312`. Re-running the command
reuses the environment, wheel cache, source tree, and compiled HPC wheel.

## Benchmark after install

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
