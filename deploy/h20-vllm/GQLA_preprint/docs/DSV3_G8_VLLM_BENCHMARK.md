# DeepSeek-V3.1 G=8 vLLM 端到端测速

`scripts/benchmark_dsv3p1_g8_vllm.sh` 针对同一个转换后 checkpoint，逐项启动
vLLM OpenAI-compatible server，并使用 vLLM 自带的 `bench serve` 客户端完成成对
测速。它测量的是包含 tokenizer、HTTP、vLLM scheduler、模型投影、MoE、KV cache
写入、attention、TP 通信和采样在内的在线服务表现，不是单 attention kernel 延迟。

H100 80GB 上使用 16 卡、`TP=8 + PP=2` 单独测试 MLA 时，优先使用固定拓扑入口
`scripts/benchmark_dsv3p1_g8_mla_tp8_pp2.sh`；两节点原生 MP 启动方法见
`docs/DSV3_G8_MLA_TP8_PP2_BENCHMARK.md`。

H20 单机 8 卡上使用 `TP=8` 成对测试 MLA 和 GQLA/HPC 时，使用固定入口
`scripts/benchmark_dsv3p1_g8_h20_tp8.sh`；协议和结果说明见
`docs/DSV3_G8_H20_TP8_BENCHMARK.md`。

脚本按本地 vLLM `0.22.1` CLI 的实际 `--help=all` 参数编写，并在运行前严格检查
版本。整个运行设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，不会下载模型或
数据集。

## 对比路径

每个 case 都从相同的 `MODEL_DIR` 加载完全相同的转换权重，只通过
`--hf-overrides` 改变 architecture：

| route | architecture | KV/attention 路径 | 路由证明 |
|---|---|---|---|
| `gqa-hpc` | `DeepseekV3GQLAHPCForCausalLM` | 真实 GQA cache；所有 decode 使用 HPC-Ops adaptive static split-K；纯 prefill 使用 DiffKV，混合 batch 只把 prefill 后缀交给 DiffKV | server 继承 `GQLA_HPC_STRICT=1`、`GQLA_HPC_TRACE=1`；必须出现 all-decode strict、`splitk=adaptive_static`、HPC HIT 和生产 YaRN scale，且不允许 fallback；启用 chunked prefill 时还要求 mixed-split 证据 |
| `mqa-mla` | `DeepseekV3GQLAForCausalLM` | 上游 MLA absorb / latent MQA cache control | 清除两个 HPC 环境变量；server log 不允许出现 HPC HIT |

仅仅成功加载 GQA architecture 或只出现一次 HIT 都不足以证明所有 decode 使用了
HPC-Ops。backend 必须先打印 `GQLA_HPC_ALL_DECODE_STRICT_ENABLED`；首次真实 kernel
命中还必须带有 `splitk=adaptive_static`，证明 vLLM 没有关闭 split-K；混合调度时还会打印
`GQLA_HPC_TRACE_MIXED_SPLIT`。HIT 是 `hpc.attention_decode_bf16` 实际返回后打印的。
strict 模式保证任意 decode 布局不满足 HPC ABI、扩展导入失败或 kernel 调用失败时
立即终止 case，而不是把该 batch 静默交回 DiffKV。

## 固定协议

同一个 matrix run 中，以下参数只计算一次，并原样复用于所有 route 和 TP case：

| 参数 | 默认值 |
|---|---:|
| TP | `1,2,4,8` |
| KV block size | `64`，不可覆盖 |
| dataset | vLLM `random`，不需要外部数据 |
| input tokens / request | `2048` |
| output tokens / request | `128` |
| measured requests | `256` |
| warmup requests | `16` |
| max concurrency | `64` |
| request rate | `inf`，由 concurrency 限流 |
| random length range | `0`，即严格定长 |
| random prefix | `0` |
| temperature | `0` |
| ignore EOS | 开启，确保生成长度一致 |
| seed | `42` |
| metric percentiles | `p50,p90,p99` |
| max model length | `4096` |
| max batched tokens | `8192` |
| prefix cache / chunked prefill / cascade | 全部关闭 |

协议参数可以通过环境变量在整个 run 级别覆盖，例如设置 `INPUT_LEN=4096`。不要给
单独 case 使用不同参数；那样的结果不能用于 GQA 与 MQA/MLA 的成对比较。

## 前置条件

1. 同一个 Python 环境中安装 vLLM `0.22.1`、当前工程的 editable plugin，以及带
   runtime `softmax_scale` 的 HPC-Ops。
2. 先运行 `scripts/run_dsv3p1_g8_vllm_smoke.sh`，确认 tiny dummy model 的 prefill、
   decode 和 HPC trace 都通过。
3. `MODEL_DIR` 必须是完整的本地转换目录。默认值为：

   ```text
   /prodcpfs/user/panzhixin/GQLA/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract
   ```

4. 设置 `CUDA_VISIBLE_DEVICES`，并确保其设备数量不少于本次最大的 TP。

完整 DeepSeek-V3.1 checkpoint 很大。脚本“支持 TP1/2/4/8”表示 CLI、模型分片和结果
矩阵能够表达这些 case，不表示每种 TP 都能装入当前机器。默认全矩阵需要八张可见
GPU；如果显存不足，应先跑 `TPS=8`，或者换到容量足够的节点。脚本不会使用 dummy
weights，也不会替用户绕过显存约束。

## 运行

先仅检查将要执行的命令，不启动模型、不创建结果：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint
DRY_RUN=1 \
TPS=1,2,4,8 \
PATHS=gqa-hpc,mqa-mla \
bash scripts/benchmark_dsv3p1_g8_vllm.sh
```

在八张指定 GPU 上运行完整矩阵：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/benchmark_dsv3p1_g8_vllm.sh
```

先做最有可能装下完整模型的 TP8 成对实验：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TPS=8 \
PATHS=gqa-hpc,mqa-mla \
bash scripts/benchmark_dsv3p1_g8_vllm.sh
```

在整个 matrix 上修改协议：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TPS=8 \
INPUT_LEN=4096 \
OUTPUT_LEN=256 \
NUM_PROMPTS=512 \
NUM_WARMUPS=32 \
MAX_CONCURRENCY=64 \
MAX_MODEL_LEN=8192 \
MAX_NUM_BATCHED_TOKENS=16384 \
bash scripts/benchmark_dsv3p1_g8_vllm.sh
```

性能正式数据默认使用 CUDA graph；诊断时可设置 `ENFORCE_EAGER=1`。两条路径在同一
run 内始终使用同一个值。

## 结果与复现信息

所有结果被强制写入以下目录的子目录，不允许把 `OUTPUT_ROOT` 指向 GQLA
`outputs/` 之外：

```text
/prodcpfs/user/panzhixin/GQLA/outputs/benchmarks/dsv3p1_g8_vllm/<RUN_ID>/
├── manifest.env
├── gpu_inventory.csv
├── matrix.tsv
├── gqa-hpc-tp8/
│   ├── server.cmd
│   ├── bench.cmd
│   ├── server.log
│   ├── bench.log
│   ├── benchmark.json
│   └── verification.txt
└── mqa-mla-tp8/
    └── ...
```

`manifest.env` 记录模型 config 和脚本 SHA256、vLLM 版本、可见 GPU、服务参数和完整
benchmark 协议。每个 case 的两个 `.cmd` 文件使用 shell-escaped 形式保留实际命令；
`benchmark.json` 包含聚合指标和逐请求明细；`verification.txt` 保存路由判定。

默认 `RUN_ID` 是 UTC 时间加 shell PID。显式复现实验可指定唯一名称：

```bash
RUN_ID=h20-tp8-gqa-vs-mla-001 TPS=8 \
bash scripts/benchmark_dsv3p1_g8_vllm.sh
```

脚本拒绝覆盖已有 run 目录。

## 服务进程安全

每个 case 使用全新的 server，完成后才进入下一个 case。服务由 `setsid` 放入独立
process group；脚本只保存并操作这个精确的 PGID：

1. 正常结束或收到 `INT`/`TERM` 时，向自己的 server process group 发送 `TERM`；
2. 最多等待 `SHUTDOWN_TIMEOUT_SECONDS`；
3. 超时后只对同一个 PGID 发送 `KILL`；
4. 如果 PGID 等于脚本自身 process group，则拒绝发送信号。

脚本不使用 `pkill`、`killall` 或按进程名匹配。若 `HOST:PORT` 已被占用，它会直接
失败，绝不会清理占用端口的未知进程。server 启动失败时会打印其最后 100 行日志。

## 解读限制

- `benchmark.json` 是服务端到端数据，不能与单 kernel CUDA-event 数字直接比较。
- GQA 与 MLA 的 KV cache 大小不同；这正是所比较架构的一部分，但会导致可容纳的
  batch/context 上限不同。
- 各 TP case 都重启模型，因此不共享 CUDA graph、allocator 或 prefix cache 状态。
- 只有 GQLA 的 `verification.txt` 为 `verified_all_decode_hpc_splitk`、MLA control 为
  `verified_no_hpc_hit`，且 matrix 最终打印
  `DSV3P1_G8_VLLM_BENCHMARK_MATRIX_OK` 的 case 才应进入汇总表。
