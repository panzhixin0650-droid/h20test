# DeepSeek-V3.1 G=8：H20 单机 TP8 的 MLA / GQLA 成对测速

入口脚本是：

```text
scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

单机一次串行跑完 `2K / 8K / 16K × MLA / GQLA` 六个 fresh-server case：

```text
scripts/benchmark_dsv3p1_g8_h20_tp8_all.sh
```

总入口默认使用 1024 个正式请求、64 个 warmup、最大并发 64 和固定 128-token
输出。每个长度/路径都重新启动服务，因此会加载六次完整 checkpoint，但不同 case 之间
不会共享 KV cache、CUDA graph 或 scheduler 状态。可用 `PROFILES=2k,8k` 或
`PATHS=gqa-hpc` 只跑子集；加入 `16k-b20` 可额外测相同并发的 16K 配对。

固定 16K/128 且每次只跑一条路径的入口是：

```text
scripts/benchmark_dsv3p1_g8_mla_h20_tp8_16k.sh
scripts/benchmark_dsv3p1_g8_gqla_h20_tp8_16k.sh
```

它在单台 8 卡机器上使用 vLLM 0.22.1 和同一份转换 checkpoint，顺序运行两个相互
隔离的服务 case：

| case | architecture | attention / KV 路径 | 有效性条件 |
|---|---|---|---|
| `mqa-mla-tp8` | `DeepseekV3GQLAForCausalLM` | vLLM `FLASH_ATTN_MLA`、latent MQA cache | 日志必须证明选择 `FLASH_ATTN_MLA`，且没有 HPC HIT |
| `gqa-hpc-tp8` | `DeepseekV3GQLAHPCForCausalLM` | 真实 GQLA cache；decode 调用 HPC-Ops，prefill 使用 DiffKV | strict 模式下必须出现 HPC HIT 和 `softmax_scale=0.135233...`，且不得出现 eligible-decode fallback |

两个 case 都固定 `TP=8`、`PP=1`、KV block size 64，并加载：

```text
/mnt/tidalfs-alwl01/task/236362/GQLA/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract
```

每个 case 都重新启动一个干净服务；MLA 完成并停止后才启动 GQLA。两边复用完全相同
的 random exact-length workload、seed、服务参数和采样参数，默认开启 chunked
prefill 和 CUDA graph，不使用 CPU offload。

## 裸环境自动安装

H20 入口默认启用 `BOOTSTRAP_ENV=1`。第一次执行时会自动调用：

```text
scripts/bootstrap_dsv3p1_g8_h20_env.sh
```

它会完成以下工作：

1. 只校验现有模型的 `model.safetensors.index.json` 及其全部 shard；**不会下载、复制或改写模型权重**。
2. 把 Python 3.12 环境装到 `/mnt/tidalfs-alwl01/task/236362/GQLA/envs/venv-py312`。
3. 使用 Rust 实现的 `uv` 并发下载大 wheel，而不是串行使用 pip；下载缓存固定保存在 `/mnt/tidalfs-alwl01/task/236362/GQLA/.cache/uv`，任务重启后可复用。
4. 安装已验证的 `torch==2.11.0`、`vllm==0.22.1`、`transformers==5.12.1` 和 CMake/Ninja；不安装测速不需要的 `lm_eval`、`datasets`。若 CUDA 13 Torch 遇到 CUDA 12.x 宿主驱动，脚本会从 NVIDIA 官方仓库下载并校验约 62 MiB 的 `cuda-compat-13-0` 用户态兼容库，解压到 `envs`，不会改宿主机驱动。
5. 以 editable 方式安装当前 GQLA plugin，确保 vLLM worker 不会指向另一台机器上的旧源码路径。
6. 当 `PATHS` 包含 `gqa-hpc` 时，使用目标环境自己的 Torch/CUDA ABI 在 H20 上编译并安装 HPC-Ops；源码或 ABI 没变时后续运行直接跳过。
7. 检查 8 张可见 GPU、CUDA 初始化、SM90 capability、vLLM architecture 注册以及 HPC `softmax_scale` schema。

安装日志保存在：

```text
/mnt/tidalfs-alwl01/task/236362/GQLA/envs/logs/bootstrap-h20.log
```

宿主容器仍须提供 NVIDIA driver、`curl`、`tar` 和 `g++`。CUDA toolkit 优先使用宿主上版本不低于 12.8 的 `nvcc`；如果没有，脚本会尝试使用 Torch 依赖中安装的 CUDA 13 toolkit wheel。缺少不能在用户目录安全补齐的宿主组件时，脚本会在加载 660+ GiB checkpoint 之前明确退出。

只安装并检查环境、不启动模型：

```bash
cd /mnt/tidalfs-alwl01/task/236362/GQLA/code/GQLA_preprint
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
INSTALL_ONLY=1 \
bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

`uv` 默认并发下载 16 个文件并使用持久缓存。如果 DLC 提供更快的内部 PyPI mirror，可以在首次安装时设置 `UV_DEFAULT_INDEX`，例如：

```bash
UV_DEFAULT_INDEX=https://<your-fast-pypi-mirror>/simple \
INSTALL_ONLY=1 bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

环境已准备好后可用 `BOOTSTRAP_ENV=0` 跳过 bootstrap；正常情况下保持默认更安全，因为快速检查会自动复用已有安装。需要重编 HPC-Ops 时设置 `FORCE_HPC_REBUILD=1`，需要重新解析并安装核心 wheel 时设置 `FORCE_CORE_REINSTALL=1`。

如果日志出现 `driver ... too old (found version 12040)`，说明复制来的环境是 CUDA 13
Torch，而宿主机驱动仍属于 CUDA 12.x。更新脚本后重新执行安装检查即可；bootstrap 会
自动配置 NVIDIA forward-compat 库并复用已经装好的 Torch/vLLM 和已经编译成功的
HPC-Ops，不需要重下数 GiB wheel，也不需要升级宿主机内核驱动。

## 运行

环境安装完成后，可只展开两条服务命令，不启动模型、不创建结果目录：

```bash
cd /mnt/tidalfs-alwl01/task/236362/GQLA/code/GQLA_preprint && \
BOOTSTRAP_ENV=0 DRY_RUN=1 RUN_ID=h20-tp8-gqa-vs-mla-001 \
bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

裸环境第一次运行 `DRY_RUN=1` 时若保留默认 `BOOTSTRAP_ENV=1`，仍会先安装并检查环境，只是不加载模型、不创建 benchmark 结果目录。

正式运行；DLC 已经正确设置 8 张可见卡时，不需要手工覆盖
`CUDA_VISIBLE_DEVICES`：

```bash
cd /mnt/tidalfs-alwl01/task/236362/GQLA/code/GQLA_preprint && \
RUN_ID=h20-tp8-gqa-vs-mla-001 \
bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

普通 shell 中也可以显式指定 8 张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RUN_ID=h20-tp8-gqa-vs-mla-001 \
bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

只跑 MLA 或只跑 GQLA：

```bash
PATHS=mqa-mla RUN_ID=h20-tp8-mla-only-001 \
bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh

PATHS=gqa-hpc RUN_ID=h20-tp8-gqla-only-001 \
bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

16K MLA 或 16K GQLA 使用独立入口，避免两条超长上下文 case 在同一个 DLC 作业中
连续加载模型：

```bash
RUN_ID=h20-mla-tp8-16k-001 \
bash scripts/benchmark_dsv3p1_g8_mla_h20_tp8_16k.sh

RUN_ID=h20-gqla-tp8-16k-001 \
bash scripts/benchmark_dsv3p1_g8_gqla_h20_tp8_16k.sh
```

这两个入口固定 `input/output=16384/128`、`max-model-len=17408` 和
`max-num-batched-tokens=32768`，其余请求数、warmup、并发和采样参数与默认协议一致。

`RUN_ID` 是实验基名。脚本自动选择首个未占用的后缀，第一次是 `-attempt0`，任务
重跑后依次使用 `-attempt1`、`-attempt2`。也可用 `ATTEMPT=3` 显式指定；已有目录
不会被覆盖。

脚本不按 `nvidia-smi` 返回的 H20/H100 等产品名拦截，只要求单机至少有 8 张可见
GPU。不过 GQLA/HPC kernel 本身要求 SM90；在非兼容架构上 strict case 会失败，不能
把 fallback 数据记作 GQLA/HPC 结果。

## 默认测速协议

| 参数 | 默认值 |
|---|---:|
| input / output tokens | 2048 / 128 |
| measured / warmup requests | 1024 / 64 |
| max concurrency | 64 |
| request rate | inf |
| max model length | 4096 |
| max batched tokens | 8192 |
| GPU memory utilization | 0.95 |
| CPU offload | 0（脚本拒绝非零值） |
| eager | 关闭，即正式数据允许 CUDA graph |
| prefix cache / cascade attention | 关闭 |
| chunked prefill | 开启 |

如需更长上下文，应在整个 run 上统一覆盖，例如：

```bash
INPUT_LEN=4096 OUTPUT_LEN=256 MAX_MODEL_LEN=8192 \
MAX_NUM_BATCHED_TOKENS=16384 NUM_PROMPTS=1024 NUM_WARMUPS=64 \
RUN_ID=h20-tp8-long-001 \
bash scripts/benchmark_dsv3p1_g8_h20_tp8.sh
```

不要给两条路径设置不同 workload。若完整模型在实际 H20 容量下仍无法初始化，正式
对比不要偷偷增加 CPU offload；先保留失败日志，确认是权重、CUDA graph workspace
还是 KV cache budget，再统一调整两条路径的配置。

## 结果目录

```text
/mnt/tidalfs-alwl01/task/236362/GQLA/outputs/benchmarks/dsv3p1_g8_h20_tp8/
└── <RUN_ID>-attempt<N>/
    ├── manifest.env
    ├── gpu_inventory.csv
    ├── matrix.tsv
    ├── mqa-mla-tp8/
    │   ├── server.cmd
    │   ├── bench.cmd
    │   ├── server.log
    │   ├── bench.log
    │   ├── benchmark.json
    │   └── verification.txt
    └── gqa-hpc-tp8/
        └── ...
```

只有两个 `verification.txt` 分别通过 MLA backend 和 GQLA/HPC trace 校验，并且最终
打印 `DSV3P1_G8_VLLM_BENCHMARK_MATRIX_OK`，这一组结果才算完整有效。启动和 checkpoint
加载时间不计入 `bench serve` 的吞吐与请求延迟指标。
