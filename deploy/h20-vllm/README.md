# H20 vLLM deployment bundle

## Aliyun-only clean environment

When the source tree and converted checkpoint are already present, run the
standalone installer below. It creates `GQLA/envs/h20-aliyun-py312`; it does
not reuse `venv-py312`. All 194 pinned Python wheels are resolved from
`https://mirrors.aliyun.com/pypi/simple/`, downloaded concurrently by aria2
with the mirror-published SHA256, audited for CPython 3.12/glibc x86-64, and
installed offline with `--no-index`. The model is never downloaded.

```bash
R=/mnt/public03/task/236362/GQLA
export GQLA_ROOT=$R
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash $R/code/GQLA_preprint/scripts/setup_h20_aliyun_env.sh
```

Success is reported only after CUDA, all eight H20s, both vLLM model
registrations, and the rebuilt patched HPC-Ops ABI have passed bootstrap:

```text
H20_ALIYUN_SETUP_OK
```

## Optional: clean cu129 environment plus one-case benchmark

The current H20 DLC host exposes a CUDA 12.x driver. The two scripts below do
not reuse a copied virtual environment and do not require the CUDA 13 forward
compatibility workaround:

1. `setup_h20_cu129_env.sh` creates `GQLA/envs/h20-cu129-py312`, downloads the
   official Torch 2.11 cu129 and vLLM 0.22.1 cu129 release wheels with aria2.
   It also resolves the pinned cuBLAS, cuDNN, NCCL, FlashInfer cubin, Triton,
   and other large wheels from official PyPI metadata and gives every wheel at
   least 32 MiB to aria2. uv installs the remaining dependency closure
   concurrently. The script then deploys the GQLA/HPC sources embedded in this
   GitHub relay, verifies the exact source containing runtime `softmax_scale`
   and adaptive split-K, and builds HPC-Ops against the exact Torch ABI.
2. `run_h20_tp8_benchmark.sh` runs exactly one fresh-server case. The route is
   `mla` or `gqla`; the profile is `2k`, `8k`, `16k`, or `16k-b20`.

```bash
GQLA_ROOT=/mnt/public03/task/236362/GQLA \
MODEL_DIR=/mnt/public03/task/236362/GQLA/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash deploy/h20-vllm/setup_h20_cu129_env.sh
```

Each benchmark is explicitly launched and recorded separately:

```bash
GQLA_ROOT=/mnt/public03/task/236362/GQLA \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash deploy/h20-vllm/run_h20_tp8_benchmark.sh mla 2k

GQLA_ROOT=/mnt/public03/task/236362/GQLA \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash deploy/h20-vllm/run_h20_tp8_benchmark.sh gqla 2k
```

The benchmark script never installs packages. MLA must resolve to
`FLASH_ATTN_MLA`; GQLA runs with strict tracing and is accepted only when the
HPC kernel records a hit with no eligible-decode fallback.

After a failed or interrupted matrix, run the read-only environment/log
diagnostic against its session directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash deploy/h20-vllm/diagnose_h20_env_and_run.sh \
  /path/to/dsv3p1_g8_h20_tp8_all/h20-tp8-all-...-attempt0
```

It discovers the exact Python recorded by `manifest.env` when possible, checks
Torch/CUDA on all eight GPUs, the vLLM architecture registry, the patched HPC
schema, and every checkpoint shard. It then classifies common failures and
prints focused tails from the launcher, server, and benchmark logs. It never
installs packages or changes the environment.

This directory transfers the GQLA vLLM plugin, H20 environment bootstrap, and
benchmark launchers. It does **not** contain or modify the converted model.

The deployment script checks that the converted checkpoint is already at:

```text
/mnt/public03/task/236362/GQLA/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract
```

It copies the bundled plugin to `GQLA/code/GQLA_preprint` and unpacks a
SHA256-verified HPC-Ops source archive at local revision `de202c9`. That source
contains all three local commits required by this integration:

- `83165c3`: GQLA QK192/V128 BF16 decode support, based on Tencent `1cd3329`.
- `f85ce84`: runtime softmax scale and build/bootstrap fixes.
- `de202c9`: adaptive static split-K for small-batch GQLA decode.

The archive contains source only—no wheel, build tree, or model data. Keeping
the exact source archive in this transit branch avoids depending on a
local-only commit that Tencent GitHub cannot serve.

## Clone and deploy

Run on the H20 machine. Use the lightweight root-history relay branch. It
contains only the GQLA/vLLM integration, benchmark scripts, the exact 2.8 MB
HPC-Ops source archive, and one small patch. CPython, uv, wheels, model data,
and benchmark outputs are deliberately excluded. Clone it beside the existing
checkout so an older or locally modified `/root/h20test` is not overwritten:

```bash
R='/mnt/public03/task/236362/GQLA'
RELAY='/root/h20test-splitk-lite-20260818'
BRANCH='deploy/h20-vllm-splitk-lite-20260818'

git clone --depth 1 --branch "$BRANCH" --single-branch \
  https://github.com/panzhixin0650-droid/h20test.git \
  "$RELAY"
```

Keep the previous HPC-Ops tree as a rollback source and materialize `de202c9`
at a new path. The generated runtime records this new path, so no old source is
silently reused:

```bash
R='/mnt/public03/task/236362/GQLA'
RELAY='/root/h20test-splitk-lite-20260818'
MODEL="$R/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract"
HPC_NEW="$R/code/hpc-ops-de202c9"

GQLA_ROOT="$R" \
MODEL_DIR="$MODEL" \
HPC_OPS_DIR="$HPC_NEW" \
FORCE_HPC_REBUILD=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$RELAY/deploy/h20-vllm/setup_h20_cu129_env.sh"
```

If the machine's Git/GnuTLS client cannot complete a GitHub TLS handshake,
download the same branch through GitHub codeload instead:

```bash
RELAY='/root/h20test-splitk-lite-20260818'
BRANCH='deploy/h20-vllm-splitk-lite-20260818'

mkdir -p "$RELAY"

curl -fL --retry 10 --retry-delay 2 --connect-timeout 30 \
  -o /tmp/h20test-splitk.tar.gz \
  "https://codeload.github.com/panzhixin0650-droid/h20test/tar.gz/refs/heads/$BRANCH"

tar -xzf /tmp/h20test-splitk.tar.gz \
  --strip-components=1 -C "$RELAY"
```

The H20 deployment needs no second GitHub request for HPC-Ops: its verified
source archive is already inside this small transit bundle. The lightweight
setup reuses an existing `h20-cu129-py312` environment and can install the
updated editable GQLA package and rebuilt HPC-Ops wheel with that environment's
pip. If the full serving environment is absent or has the wrong Torch/vLLM
stack, transfer the environment or wheel bundle through a verified OSS prefix;
do not put Python, uv, CUDA wheels, or a venv in this Git branch.

For a split-K source refresh on a machine that already ran vLLM, use the
HPC-only updater. Always pass the serving venv and the runtime wrapper that
actually makes CUDA initialize. The updater accepts the stack only when it has
Torch 2.11 with CUDA 12.9 or 13.0, vLLM 0.22.1, Transformers 5.12.1, and eight
SM90 GPUs. If only `RUNTIME_ENV_FILE` is supplied, its parent is accepted as
the venv only when `bin/python` exists there; otherwise the command stops
instead of silently selecting another environment.

The updater preserves the wrapper's driver forward-compatibility setup and has
no dependency-download path. In actual-update mode it can bootstrap a missing
pip through offline `ensurepip`, restore executable bits stripped from a
transferred pip CUDA compiler, and construct an unversioned overlay for a
versioned-only `libcudart.so.<major>`. A compiler/header minor mismatch such as
CUDA compiler 13.3 with CUDA 13.0 headers is allowed only within the same major;
the build then records and applies `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK`.
Different CUDA majors remain a hard error. Set `REPAIR_TOOL_EXEC_BITS=0` or
`ALLOW_CCCL_MINOR_MISMATCH=0` to disable either repair explicitly.

Run its read-only preflight first. Success is
`H20_SPLITK_PREFLIGHT_OK`; this command does not deploy, compile, or install:

```bash
R='/mnt/public03/task/236362/GQLA'
RELAY='/root/h20test-splitk-lite-20260818'
MODEL="$R/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract"
E="$R/envs/venv-py312"
W="$E/h20-runtime-splitk.env"

GQLA_ROOT="$R" MODEL_DIR="$MODEL" VENV_DIR="$E" \
RUNTIME_ENV_FILE="$W" PREFLIGHT_ONLY=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$RELAY/deploy/h20-vllm/update_h20_splitk_only.sh"
```

Preflight never changes permissions or bootstraps pip. If it reports that nvcc
is present but non-executable, run the actual update with the default repair
enabled after confirming the path is inside the selected venv; a `noexec`
mount still requires infrastructure repair.

Then deploy the plugin and rebuild only HPC-Ops:

```bash
R='/mnt/public03/task/236362/GQLA'
RELAY='/root/h20test-splitk-lite-20260818'
MODEL="$R/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract"
HPC_NEW="$R/code/hpc-ops-de202c9"
E="$R/envs/venv-py312"
W="$E/h20-runtime-splitk.env"

GQLA_ROOT="$R" MODEL_DIR="$MODEL" HPC_OPS_DIR="$HPC_NEW" \
VENV_DIR="$E" RUNTIME_ENV_FILE="$W" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$RELAY/deploy/h20-vllm/update_h20_splitk_only.sh"
```

Success is `H20_SPLITK_HPC_ONLY_OK`. The updater writes
`h20-splitk-runtime.env` and a build manifest inside the selected venv, keeps
the compiled wheel under `envs/wheels/hpc-splitk`, and does not modify the
converted checkpoint. Re-running it intentionally rebuilds the exact
`de202c9` adaptive split-K source for the selected Torch/CUDA ABI. The manifest
also records compiler/header versions, the selected cudart, any executable-bit
repair, and any CCCL compatibility flag. These install markers prove the
extension build and runtime/schema check; a real vLLM request must still emit
the expected kernel HIT/policy trace with zero eligible fallback before the
integration is considered complete.
The lower-level `setup_h20_cu129_env.sh` also defaults to
`ALLOW_CORE_DOWNLOADS=0`; a full environment installation must now opt in
explicitly with `ALLOW_CORE_DOWNLOADS=1` and should use OSS when direct wheel
downloads are slow.

Run the three GQLA C=64 cases separately after setup:

```bash
R='/mnt/public03/task/236362/GQLA'
RELAY='/root/h20test-splitk-lite-20260818'

GQLA_ROOT="$R" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$RELAY/deploy/h20-vllm/run_h20_tp8_benchmark.sh" gqla 2k

GQLA_ROOT="$R" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$RELAY/deploy/h20-vllm/run_h20_tp8_benchmark.sh" gqla 8k

GQLA_ROOT="$R" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash "$RELAY/deploy/h20-vllm/run_h20_tp8_benchmark.sh" gqla 16k
```

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
