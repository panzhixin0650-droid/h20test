# DeepSeek-V3.1 G=8 H100：TP8 × PP2 的 2K/8K/16K 路径矩阵

这组入口都使用同一份转换后完整权重：

```text
/prodcpfs/user/panzhixin/GQLA/outputs/convert/dsv3p1_g8_sim_hess_no_mean_subtract
```

拓扑固定为 TP=8、PP=2、world size=16、layer partition=`31,30`。DLC 任务配置为
2 个 worker、每个 worker 8 张可见 GPU；每次只提交下面一个脚本，DLC 会在两个
worker 上执行同一入口并自动分出 head 和 headless worker。启动器只检查每节点至少
8 张可见 GPU，不按 GPU 型号名称拒绝任务。

## 可比较的六个实验

| 路径 | 输入/输出 | 入口 |
|---|---:|---|
| MLA absorb | 2K/128 | `benchmark_dsv3p1_g8_mla_tp8_pp2.sh` |
| GQLA + HPC-Ops decode | 2K/128 | `benchmark_dsv3p1_g8_gqla_h100_tp8_pp2_2k.sh` |
| MLA absorb | 8K/128 | `benchmark_dsv3p1_g8_mla_h100_tp8_pp2_8k.sh` |
| GQLA + HPC-Ops decode | 8K/128 | `benchmark_dsv3p1_g8_gqla_h100_tp8_pp2_8k.sh` |
| MLA absorb | 16K/128 | `benchmark_dsv3p1_g8_mla_h100_tp8_pp2_16k.sh` |
| GQLA + HPC-Ops decode | 16K/128 | `benchmark_dsv3p1_g8_gqla_h100_tp8_pp2_16k.sh` |

六组默认都使用 1024 条正式请求、64 条 warmup、并发 64、固定长度、temperature=0、
ignore EOS，并报告 TTFT/TPOT/ITL/E2EL 的 p50/p90/p99。2K 使用
`max-model-len=4096` 和 `max-num-batched-tokens=8192`；8K 使用
`max-model-len=9216` 和 `max-num-batched-tokens=16384`；16K 使用
`max-model-len=17408` 和 `max-num-batched-tokens=32768`。

## DLC 提交命令

优先在同一个 16-GPU DLC 作业中重跑 2K 配对。两个 worker 会先运行 MLA control，
清理分布式 engine 并等待 30 秒，再运行 all-decode HPC-Ops GQLA：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint

SERIES_ID=h100-tp8-pp2-2k-all-decode-001 \
bash scripts/benchmark_dsv3p1_g8_h100_tp8_pp2_2k_serial.sh
```

两条结果基名分别为 `${SERIES_ID}-mla` 和 `${SERIES_ID}-gqla`。这一入口保持相同
16 卡拓扑和服务协议，适合替换此前只验证“至少一次 HPC HIT”的旧 H100 2K 数据。
加载 660+ GiB 正式权重前，两个 worker 会各自在本机 GPU 0 上运行一个 tiny dummy
mixed-batch smoke，分别确认 all-decode、mixed-split、HPC HIT 且零 fallback；随后较早
完成的节点会在分布式 rendezvous 中等待另一节点。诊断时可用 `RUN_MIXED_SMOKE=0`
跳过，但这种运行不应作为新路由首次验收。

当前 MLA 2K：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint && \
RUN_ID=h100-mla-tp8-pp2-2k-001 \
bash scripts/benchmark_dsv3p1_g8_mla_tp8_pp2.sh
```

GQLA 2K：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint && \
RUN_ID=h100-gqla-tp8-pp2-2k-001 \
bash scripts/benchmark_dsv3p1_g8_gqla_h100_tp8_pp2_2k.sh
```

MLA 8K：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint && \
RUN_ID=h100-mla-tp8-pp2-8k-001 \
bash scripts/benchmark_dsv3p1_g8_mla_h100_tp8_pp2_8k.sh
```

GQLA 8K：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint && \
RUN_ID=h100-gqla-tp8-pp2-8k-001 \
bash scripts/benchmark_dsv3p1_g8_gqla_h100_tp8_pp2_8k.sh
```

MLA 16K：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint && \
RUN_ID=h100-mla-tp8-pp2-16k-001 \
bash scripts/benchmark_dsv3p1_g8_mla_h100_tp8_pp2_16k.sh
```

GQLA 16K：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint && \
RUN_ID=h100-gqla-tp8-pp2-16k-001 \
bash scripts/benchmark_dsv3p1_g8_gqla_h100_tp8_pp2_16k.sh
```

16K 两条路径也可以只提交一次 DLC 作业。下面的组合入口由两个 worker 共同执行，
先完整运行并清理 MLA 服务，等待默认 30 秒，再启动全新的 GQLA 服务：

```bash
cd /prodcpfs/user/panzhixin/GQLA/code/GQLA_preprint && \
SERIES_ID=h100-tp8-pp2-16k-pair-001 \
bash scripts/benchmark_dsv3p1_g8_h100_tp8_pp2_16k_serial.sh
```

两条结果的实验基名分别是 `${SERIES_ID}-mla` 和 `${SERIES_ID}-gqla`，并各自再追加
首个未占用的 `-attemptN`。任一路径失败时组合入口立即失败，不会把后一条路径伪装成
完整配对结果。可用 `SERIAL_COOLDOWN_SECONDS` 调整两个分布式 engine 之间的等待时间。

同一个 `RUN_ID` 遇到 DLC failover 时会自动选择首个未占用的 `-attemptN`，不会覆盖
之前的日志和结果。各入口有独立的 `OUTPUT_ROOT`，即使使用相似 RUN_ID 也不会
互相冲突。

## 路由验真

MLA case 必须解析为 `DeepseekV3GQLAForCausalLM`，日志必须选择
`FLASH_ATTN_MLA`，且不允许出现 HPC hit。

GQLA case 在 head 和远端 worker 上都显式设置 `GQLA_HPC_STRICT=1` 和
`GQLA_HPC_TRACE=1`，架构必须解析为 `DeepseekV3GQLAHPCForCausalLM`。正式请求结束后，
脚本要求 `GQLA_HPC_ALL_DECODE_STRICT_ENABLED`、至少一个
带 `splitk=adaptive_static` 的 `GQLA_HPC_TRACE_HIT`、chunked-prefill 下的
`GQLA_HPC_TRACE_MIXED_SPLIT`、零 fallback，
并要求日志中的 YaRN `softmax_scale` 命中 `0.135233...`；否则 case 退出失败，不能写
`status=verified_all_decode_hpc_splitk`。

GQLA 的纯 prefill 走 FlashAttention DiffKV。混合 batch 被拆成 decode 前缀和 prefill
后缀：前者必须走 HPC-Ops，只有后者走 DiffKV。因此长输入比较时 TTFT 主要反映
prefill，而 TPOT/ITL 更适合观察 decode 路径。旧版本只加速纯 decode batch，混合
batch 会整体回退；在新 all-decode 验证下重跑前，旧 H100 数字应标记为部分路由结果。

8K、并发 64 的最坏在途长度是 `64 * (8192 + 128) = 532480` tokens。服务日志中的
KV cache capacity 若低于该值，调度器可能无法维持完整 C64；应对两条路径使用同一个
较低并发重新测量，不要只改其中一条。

16K、并发 64 的对应上限是 `64 * (16384 + 128) = 1056768` tokens，更可能受 KV
capacity 限制。客户端并发仍可保持 64，但报告中应同时记录服务端 cache capacity 和
实际 peak concurrency，解释两条路径因 KV 表示不同而产生的调度差异。

## 结果目录

默认分别写入：

```text
/prodcpfs/user/panzhixin/GQLA/outputs/benchmarks/dsv3p1_g8_gqla_h100_tp8_pp2_2k/
/prodcpfs/user/panzhixin/GQLA/outputs/benchmarks/dsv3p1_g8_mla_h100_tp8_pp2_8k/
/prodcpfs/user/panzhixin/GQLA/outputs/benchmarks/dsv3p1_g8_gqla_h100_tp8_pp2_8k/
/prodcpfs/user/panzhixin/GQLA/outputs/benchmarks/dsv3p1_g8_mla_h100_tp8_pp2_16k/
/prodcpfs/user/panzhixin/GQLA/outputs/benchmarks/dsv3p1_g8_gqla_h100_tp8_pp2_16k/
```

每个 attempt 包含 `manifest.env`、`gpu_inventory.csv`、`matrix.tsv`，以及对应 case
目录中的 `server.cmd`、`server.log`、`bench.cmd`、`bench.log`、`benchmark.json` 和
`verification.txt`。
