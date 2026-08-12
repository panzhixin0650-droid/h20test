# H20 复现协议

## 1. 测试目标

本仓库复现两个互不覆盖的结果集：

1. **schema-v5**：在 B=64/128 下测试 12 个配置族，得到 24 个正式点，用于比较 batch、FlashMLA/FA3-MLA 和 paged/dense GQA。
2. **schema-v6**：固定 B=128，在一张 H20 上构造 TP=1/2/4/8 的代表性 rank-local 张量，得到 12 个配置族、48 个正式点。

schema-v6 不是多卡测试，没有 NCCL。每个模拟 rank 的局部 Query head 数是

$$
H_Q^{\mathrm{local}}=\frac{128}{\mathrm{TP}}.
$$

当 GQA 的全局 KV head 数小于 TP 时，每个 rank 至少保留一个 KV head，并记录 KV 复制因子。吞吐按整个同步 TP group 处理的全局 query positions 计算，不能再乘 TP：

$$
\mathrm{tok/s}_{\mathrm{TP}}
=\frac{128s_q\times10^6}
       {T_{\mathrm{full,rank}}\,[\mu\mathrm{s}]}.
$$

## 2. 固定数学形状

- Context length：$L=8192$；
- dtype：BF16；
- Query heads：$H_Q=128$；
- Query length：$s_q\in\{1,2\}$；
- MQA-absorb：QK 物理维度 576，V 维度 512，scale 为 $1/\sqrt{576}$；
- GQA：$g\in\{8,4\}$，QK 维度 192，V 维度 128，scale 为 $1/\sqrt{192}$；
- FA3 的 `num_splits` 扫描 `{1,2,4,8,16,32,0}`；
- FA3 的 `pack_gqa` 只允许 `True` 和 `None`；
- paged GQA 的 page size 扫描 64/128；
- dense GQA 使用连续 `[B,L,g,D]` cache 和 `page_table=None`。

显式 `pack_gqa=False` 被禁止：固定的 SM90 dispatcher 在部分 split-K/PagedKV 分支会强制 PackGQA，而 scheduler metadata 仍可能保留用户传入的 False，造成配置歧义。

## 3. 软件与硬件前提

推荐完全复用已验证环境：

- NVIDIA H20，SM90；
- Python 3.12；
- PyTorch `2.11.0+cu128`；
- CUDA toolkit 12.8；
- Triton 3.6.0；
- `uv`、Git、Ninja、NVCC 和 `nvidia-smi` 可用；
- 虚拟环境中已有编译 FlashMLA/FA3 所需的 PyTorch C++ extension 依赖。

至少设置：

```bash
export GQLA_TABLE2_PYTHON=/absolute/path/to/venv/bin/python
export CUDA_HOME=/usr/local/cuda
export GQLA_TABLE2_GPU=0
```

如果尚未准备虚拟环境，可直接创建并安装固定的 cu128 依赖：

```bash
bash scripts/install_cu128_env.sh
conda activate h20table2
export GQLA_TABLE2_PYTHON="$(command -v python)"
export GQLA_TABLE2_UV="$(command -v uv)"
```

该脚本忽略集群级 pip 配置，普通包和 NVIDIA CUDA 依赖只走清华 PyPI 镜像，随后以 `--no-deps` 从 PyTorch 官方 cu128 索引安装 Torch，避免依赖解析再次访问不可达的 NVIDIA 镜像。

如果 `uv` 不在 `PATH`：

```bash
export GQLA_TABLE2_UV=/absolute/path/to/uv
```

可选路径配置：

```bash
export GQLA_TABLE2_OUTPUT_DIR=/persistent/path/h20-table2-results
export GQLA_TABLE2_TMPDIR=/node-local/tmp
```

官方源码和编译 cache 默认放在节点本地 `/tmp`。构建和测速必须在同一个 H20 节点连续完成；换节点后应重新构建。

## 4. 预检

```bash
CUDA_VISIBLE_DEVICES=0 \
  "$GQLA_TABLE2_PYTHON" -u scripts/gpu_preflight.py \
    --expect-device-substring H20
```

必须出现：

```text
CUDA available: True
visible GPUs: 1
device name: NVIDIA H20
compute capability: (9, 0)
GPU_OK
```

预检会拒绝非 H20 或非 SM90 设备。

## 5. 编译官方扩展

```bash
GQLA_KERNEL_ENABLE_FA3_MLA=1 \
GQLA_KERNEL_BUILD_CPU_BUDGET=1024 \
  bash scripts/build_official_kernels.sh
```

构建包含本次所需的 SM90 BF16 forward、PagedKV、split-K、PackGQA、varlen、head-dim 64/192 和不同 V 维度 kernel；排除 backward、FP16、FP8 和无关 head dimensions。

成功标志：

```text
EXTENSIONS_OK
BUILD_OK
```

脚本会把固定提交的源码目录写入：

```text
outputs/official/source_dir.txt
```

该路径通常位于节点本地 `/tmp`，不是可跨节点复用的持久依赖。

## 6. schema-v5：B=64/128 安全基线

```bash
GQLA_TABLE2_RERUN_ALL=1 \
GQLA_TABLE2_DEVICE_ROLE=h20 \
GQLA_TABLE2_GPU=0 \
  bash scripts/run_official_dense_batch.sh
```

smoke 应以 `ALL_SMOKE_TESTS_OK` 结束，正式运行应以 `DENSE_BATCH_RUN_OK` 结束。权威 JSON 是：

```text
outputs/official/h20_table2_all_batch64_128_safe_sq2.json
```

它必须满足：schema version 5、12 个配置族、24 个正式点、所有错误计数为 0。

## 7. schema-v6：TP=1/2/4/8 张量模拟

在 schema-v5 成功后运行：

```bash
GQLA_TABLE2_TP_SWEEP=1 \
GQLA_TABLE2_DEVICE_ROLE=h20 \
GQLA_TABLE2_GPU=0 \
  bash scripts/run_official_dense_batch.sh
```

权威 JSON 是：

```text
outputs/official/h20_table2_tp1_2_4_8.json
```

它必须满足：schema version 6、12 个配置族、48 个正式点、所有错误计数为 0。

如果先前已经执行过 schema-v6，runner 在重跑 schema-v5 时会安全撤销最后一层 TP patch；随后再运行 schema-v6 会重新应用它。

## 8. 最终验证

```bash
"$GQLA_TABLE2_PYTHON" scripts/verify_results.py \
  --output-dir "${GQLA_TABLE2_OUTPUT_DIR:-$PWD/outputs/official}"
```

验证器检查：

- schema、完成状态、配置族和点数；
- 实际设备名包含 H20，compute capability 为 SM90；
- FlashMLA/FlashAttention 固定提交；
- 导入路径不含 vLLM；
- scale 分别为 $1/\sqrt{576}$ 和 $1/\sqrt{192}$；
- 没有正式候选使用显式 `pack_gqa=False`；
- 输出 finite，正确性比较均为 `allclose=True`。

## 9. 时间口径

- `full_cold_us`：Triton `do_bench` 的完整 operator CUDA Event p50，每个样本前清空 L2；
- `main_cold_us`：FlashMLA 官方 `kernelkit.bench_kineto`/Kineto 提取的主 attention kernel 时间；
- `combine_cold_us`：与 main 相同 Kineto profile 中的 combine/reduction kernel；
- full 与 main 来自不同采样集，不能用 `full-main` 计算 combine；
- `full/B` 是吞吐等效的每序列时间，不是单请求真实等待时间；
- 实际 tok/s 是 attention operator 的 query-position 吞吐，不是整模型生成吞吐。

## 10. 常见错误

### `ModuleNotFoundError: flash_attn_3._C`

FA3 没有安装进当前 `GQLA_TABLE2_PYTHON` 指向的虚拟环境。重新执行构建，并确认构建和测速使用同一个 Python。

### `This flash attention build does not support varlen`

使用了另一份裁剪过的 FA3。当前构建脚本明确设置 `FLASH_ATTENTION_DISABLE_VARLEN=FALSE`；重新完整构建。

### `Node-local official source is absent`

当前节点没有 `source_dir.txt` 指向的 `/tmp` 源码。通常是编译后换了节点，需在当前节点重新执行 `build_official_kernels.sh`。

### Bash 或 Python 缩进错误

不要把多行 Python 逐行粘贴到 shell。使用仓库内的 `verify_results.py`；所有正式入口都是完整 shell 脚本。

### 结果中出现非 H20 设备

预检和最终验证都会失败。不要用 L20Z/H100-role 数值填充 H20 行。
