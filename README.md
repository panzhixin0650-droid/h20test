# GQLA Table 2：H20 官方 Kernel 复现

这个仓库用于在单张 NVIDIA H20 上复现 GQLA 论文 Table 2 的 kernel-only 测试。它只构造 BF16 假张量并直接调用官方 FlashMLA 和 FlashAttention-3 接口：

- 不加载模型或 tokenizer；
- 不经过 vLLM；
- 不测 NCCL 或真实多卡通信；
- 不包含 Q/K/V/O projection GEMM；
- 正式主路径为 FlashMLA MQA-absorb 与 FA3 paged GQA；
- 额外保留 FA3-MLA 和 dense GQA 对照。

完整协议与故障排查见 [H20_REPRODUCTION_ZH.md](docs/H20_REPRODUCTION_ZH.md)。

## 固定版本

- FlashMLA：`15f13e5030374295491c5ce31b02d7e63a7772c6`
- FlashAttention：`a369df707e1980fb328abcc1733e3457ec10155f`
- 推荐复现环境：Python 3.12、PyTorch `2.11.0+cu130`、CUDA 13.0、Triton 3.6.0

构建脚本会从官方仓库检出上述提交，并将独立的 `flash-mla` 和 `flash-attn-3` 扩展安装到指定虚拟环境。补丁的 SHA256 固定在 `patches/SHA256SUMS`。

## 最短运行方法

```bash
git clone https://github.com/panzhixin0650-droid/h20test.git
cd h20test

export GQLA_TABLE2_PYTHON=/absolute/path/to/venv/bin/python
export CUDA_HOME=/usr/local/cuda
export GQLA_TABLE2_GPU=0
export GQLA_KERNEL_BUILD_CPU_BUDGET=1024

bash scripts/run_h20_all.sh
```

CPU budget 会自动限制到当前节点的 `nproc` 和 cgroup `pids.max`，不会直接创建 1024 个 Ninja worker。

如果扩展已经在当前 H20 节点、同一个虚拟环境中正确编译，可以跳过构建：

```bash
GQLA_TABLE2_SKIP_BUILD=1 bash scripts/run_h20_all.sh
```

## 分步运行

```bash
bash scripts/build_official_kernels.sh
```

```bash
GQLA_TABLE2_RERUN_ALL=1 \
GQLA_TABLE2_DEVICE_ROLE=h20 \
  bash scripts/run_official_dense_batch.sh
```

```bash
GQLA_TABLE2_TP_SWEEP=1 \
GQLA_TABLE2_DEVICE_ROLE=h20 \
  bash scripts/run_official_dense_batch.sh
```

```bash
"$GQLA_TABLE2_PYTHON" scripts/verify_results.py \
  --output-dir outputs/official
```

最终验证通过会打印：

```text
H20_RESULTS_OK ...
H20_REPRODUCTION_OK
```

## 权威输出

- `outputs/official/h20_table2_all_batch64_128_safe_sq2.json`：schema-v5，B=64/128 安全基线；
- `outputs/official/h20_table2_tp1_2_4_8.json`：schema-v6，B=128 的 TP=1/2/4/8 单卡 rank-local 张量模拟；
- 同名 `.log` 与 `_smoke.log`：正式运行和正确性日志；
- `h20_nvidia_smi_*.txt`、`h20_topology.txt`：设备状态记录。

`outputs/` 默认不进入 Git。需要回传结果时，请单独传上述两份 JSON、两份正式日志、两份 smoke 日志和设备状态文件。
