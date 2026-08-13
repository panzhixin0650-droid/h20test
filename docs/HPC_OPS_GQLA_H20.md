# H20：HPC-Ops GQLA QK=192 / V=128 一键测速

这个流程通过 GitHub 中转一份固定的 HPC-Ops 源码补丁，在 H20 上从干净的上游
commit 构建，不依赖当前机器上的 `/prodcpfs/user/panzhixin/GQLA/code/hpc-ops`
工作区。

补丁新增的 BF16 paged-decode 范围是：

- `H_Q=128`、`H_KV in {8,4}`；
- `S_Q in {1,2}`；
- `D_QK=192`、`D_V=128`；
- `page_size=64`、Hopper SM90；
- static scheduling。当前 GQLA shape 即使收到 `splitk=True` 也会安全地固定为单 split，
  不接受 dynamic `task_map`。

固定上游基线：
`1cd332980ed46bd0172091c1c35d55338fcae47a`。补丁文件是
`patches/hpc_ops_gqla_qk192_v128.patch`，其哈希记录在
`patches/SHA256SUMS`。

## 一键运行

在 H20 节点上：

```bash
git clone https://github.com/panzhixin0650-droid/h20test.git
cd h20test
bash scripts/run_h20_hpc_ops_gqla.sh
```

脚本会依次完成：

1. 校验 H20/SM90、补丁 SHA256 和构建工具；
2. 克隆固定版本的 Tencent HPC-Ops，并应用补丁；
3. 构建 wheel，但不改写当前 Python 环境；
4. 用 float32 PyTorch reference 检查六组 static/split-K-hint case；
5. 正式测量 `B=128, L=8192` 下的四组 `H_KV/S_Q` 组合；
6. 发布单独的 Markdown、CSV 和 JSON 表。

环境解析顺序是 `GQLA_HPC_OPS_PYTHON`、conda 环境 `h20hpcops`、
`GQLA_TABLE2_PYTHON`、当前 venv、conda 环境 `h20table2`、仓库 `.venv`、系统
`python3`。脚本会比较 `torch.version.cuda` 和 `nvcc` 的 CUDA major，并要求两者
一致；还会验证 `cmake>=3.26`、Ninja、wheel 和 setuptools。

如果没有兼容环境，或共享的 `h20table2` 已被其他 benchmark 升级成 cu130、但
H20 节点的 `/usr/local/cuda` 仍为 12.8，脚本会调用
现有的 `install_cu128_env.sh`、但把目标环境名覆盖为隔离的 `h20hpcops`。该环境
固定为 PyTorch `2.11.0+cu128`，不安装 FlashInfer，也不会修改 `h20table2`。
缺少的 CMake 等构建工具也会自动安装到最终选中的环境中。设
`GQLA_HPC_OPS_AUTO_INSTALL=0` 可以关闭这些自动修复。

正式结果固定发布到：

```text
/prodcpfs/user/panzhixin/GQLA/outputs/kernel_table2/hpc_ops_gqla_h20.md
/prodcpfs/user/panzhixin/GQLA/outputs/kernel_table2/hpc_ops_gqla_h20.csv
/prodcpfs/user/panzhixin/GQLA/outputs/kernel_table2/hpc_ops_gqla_h20.json
```

每次运行的完整日志、raw JSON、表格和构建出的 wheel 另存于：

```text
/prodcpfs/user/panzhixin/GQLA/outputs/kernel_table2/runs/hpc_ops_gqla_h20_<UTC时间>_<PID>/
```

因此重复运行时，顶层文件表示最新结果，历史运行仍可从 `runs/` 恢复。

## 常用覆盖项

复用指定 Python 和 GPU：

```bash
GQLA_HPC_OPS_PYTHON=/absolute/path/to/python \
GQLA_HPC_OPS_GPU=0 \
CUDA_HOME=/usr/local/cuda \
  bash scripts/run_h20_hpc_ops_gqla.sh
```

保留临时源码和 build 目录以便调试：

```bash
GQLA_HPC_OPS_KEEP_WORKDIR=1 bash scripts/run_h20_hpc_ops_gqla.sh
```

`GQLA_HPC_OPS_BATCH`、`GQLA_HPC_OPS_SEQLEN`、`GQLA_HPC_OPS_WARMUP`、
`GQLA_HPC_OPS_ITERS` 和 `GQLA_HPC_OPS_FLUSH_GIB` 仅用于快速诊断；正式 Table 2
请保持默认的 `128 / 8192 / 5 / 20 / 8`。
