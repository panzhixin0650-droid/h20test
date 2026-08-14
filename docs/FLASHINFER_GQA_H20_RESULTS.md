# FlashInfer FA2 在 H20 short-query GQA 上的正式结果

更新日期：2026-08-14

## 结论

FlashInfer 能运行 GQLA Table 2 的 `D_QK=192, D_V=128` 异维 GQA，并通过
float32 PyTorch reference 正确性检查。它通过 `BatchPrefillWithPagedKVCacheWrapper`
的 FA2 backend 为短 query 选择 M16/M64，不再像当前固定版本 FA3 一律执行 M128。

不过，这个替换是**部分解决**，不是四组 shape 的完整解决：

- `H_KV=8, S_Q=1` 从 FA3 的 `40.677 µs/序列` 降到 `13.323 µs/序列`，
  加速 `3.05×`。该点选择 M16，已经消除 M128 的 8 倍 tile 计算放大，
  并回到 HBM 主导。
- 其余三点选择 M64，只得到 `1.24×` 左右的加速。特别是
  `H_KV=8, S_Q=2` 从 M16 跨到 M64，step 时间反而由 `13.323 µs` 增到
  `32.664 µs`，说明当前 planner 仍缺 M32 specialization。
- `H_KV=4, S_Q=2` 的 64 行已经完整填满 M64；它的模型算术强度本身就是
  `64 FLOP/B`，高于这台 H20 的 `39.61 FLOP/B` 平衡点，所以即使没有空行，
  也不会自然回到纯 HBM Roofline。

因此，准确结论是：**FlashInfer FA2 修复了最严重的 M16 shape，并把其余
M128 shape 降到 M64，但当前版本仍未覆盖 M32，M64 kernel 在 H20 上也还有明显
计算效率空间。**

## 测量口径与环境

- run ID：`h20_qs-236362-1835988-ai-1372874-default0-0_formal_20260814T023711Z`
- 设备：NVIDIA H20，SM90，78 SM，约 96 GiB HBM
- shape：`B=128, L=8192, H_Q=128, D_QK=192, D_V=128`，BF16，causal
- 四点：`H_KV in {8,4}`、`S_Q in {1,2}`
- cache：紧凑 NHD interleaved K/V backing，无 padding；每序列物理 KV 为
  `40/20 MiB`
- 软件：Python 3.12.13、PyTorch `2.11.0+cu128`、CUDA 12.8、
  Triton 3.6.0、FlashInfer `0.6.11.post2`
- backend：显式 `fa2`；四点均 `split_kv=false`
- 计时：预分配输出上的 `wrapper.run`，`triton.testing.do_bench` CUDA Event
  冷 L2 完整调用 p50；allocation、plan、JIT、reference 不计时
- 硬件标定：持续 HBM read `3.572 TB/s`，Dense BF16 `141.48 TFLOP/s`

第一次 `plan` 的 JIT 为 `34.753 s`，被单独记录且不进入稳态延迟；后续 plan
约 `0.87–1.03 ms`。

## FA3 与 FlashInfer 实测对比

以下都是 B=128 的 full-call 时间除以 128；这是吞吐等效每序列时间，不是单请求
独占执行时的排队延迟。

| H_KV | S_Q | FA3 tile | FlashInfer tile | 有效行比例 | FA3 (µs/序列) | FlashInfer (µs/序列) | 加速 | FlashInfer tok/s | FlashInfer payload TB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | M128 | M16 | 100% | 40.677 | 13.323 | **3.053×** | 75.1K | 3.148 |
| 8 | 2 | M128 | M64 | 50% | 40.609 | 32.664 | **1.243×** | 61.2K | 1.284 |
| 4 | 1 | M128 | M64 | 50% | 20.408 | 16.477 | **1.239×** | 60.7K | 1.273 |
| 4 | 2 | M128 | M64 | 100% | 20.450 | 16.481 | **1.241×** | 121.4K | 1.272 |

相对 FA3，四点的 full latency 分别下降 `67.25%`、`19.56%`、`19.26%`、
`19.41%`。由于 B 和 S_Q 的口径相同，表中的 tok/s 加速与 latency 加速相同。

## Tile 与 Roofline 分解

| H_KV | S_Q | packed rows | FlashInfer tile | tile 计算放大 | tile AI (FLOP/B) | HBM 下界 (µs) | tile 计算下界 (µs) | 实测 (µs) | 预测主导项 | Roofline 达成率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 8 | 1 | 16 | M16 | 1× | 16 | 11.742 | 4.743 | 13.323 | 访存 | 88.1% |
| 8 | 2 | 32 | M64 | 2× | 64 | 11.742 | 18.973 | 32.664 | 计算 | 58.1% |
| 4 | 1 | 32 | M64 | 2× | 64 | 5.871 | 9.487 | 16.477 | 计算 | 57.6% |
| 4 | 2 | 64 | M64 | 1× | 64 | 5.871 | 9.487 | 16.481 | 计算 | 57.6% |

三个 M64 点的 tile 实际吞吐都约为 `81.4–82.2 TFLOP/s`，彼此高度一致，
因此 `32.66/16.48 µs` 平台不是随机噪声。它既包含 32 行 shape 的 2 倍空行
计算，也反映出该 M64 attention kernel 只达到 Dense BF16 标定的约 58%。

## 正确性与 kernel 证据

四点全部满足：

- `finite=true`、`allclose=true`、容差内元素比例为 100%；
- 最大绝对误差位于 `1.02e-5` 到 `1.18e-5`；
- Kineto 每次调用只观察到一个主
  `BatchPrefillWithPagedKVCacheKernel`，没有 combine；
- kernel 模板名和 plan metadata 分别确认了 M16/M64；
- run 内 `SHA256SUMS` 对全部八个文件校验通过。

`H_KV=8, S_Q=1` 的样本分布比另外三点更宽：p20/p50/p80 分别约为
`13.080/13.323/14.460 µs/序列`。它不影响相对 FA3 的三倍级结论；若数字将用于
正式论文表格，建议在锁频且确认无其他 GPU 进程后再单点复跑一次，报告多次 run
的中位数。

## 工程建议

1. 在 H20 上可优先把 `packed_query_rows <= 16` 路由到 FlashInfer FA2；
   当前 `H_KV=8, S_Q=1` 已接近物理 HBM 下界。
2. `packed_query_rows=32` 仍需要真正的 M32 specialization。若 M32 能保持
   足够的 HBM 效率，`H_KV=8,S_Q=2` 和 `H_KV=4,S_Q=1` 才可能回到各自
   `11.742/5.871 µs` 附近。
3. `packed_query_rows=64` 不存在 tile 空行，但在 H20 上天然位于计算侧；
   优化目标应是提升 M64 kernel 的 Tensor Core/流水效率，理论计算下界为
   `9.487 µs/序列`，而不是宣称它应达到纯内存下界。
4. 不要把 FlashInfer 设成所有设备、所有 shape 的无条件替换。本地
   L20Z/H100-role 上 FlashInfer FA2 四点比 FA3 慢约 1.9%–2.3%；合理做法是
   按设备和 packed rows dispatch。
5. 该 benchmark 不含模型框架接入和 KV layout 转换。生产替换还需要验证
   vLLM/推理框架中的 cache 生命周期、CUDA Graph、动态 batch 和调度开销。

## 原始产物

正式成功 run 的完整 bundle 已保存在：

- [`artifacts/flashinfer_gqa/h20_formal_20260814T023711Z.tar.gz`](../artifacts/flashinfer_gqa/h20_formal_20260814T023711Z.tar.gz)
- SHA-256：`ed1876274b8b871e8b325c8ca100f4bd1112d56f1c2e50dd3fd50824f98c52a0`

bundle 内含 `result.json`、完整 timing samples、`benchmark.log`、`verify.log`、
`environment.txt`、`nvidia-smi -q`、拓扑和逐文件 `SHA256SUMS`。
