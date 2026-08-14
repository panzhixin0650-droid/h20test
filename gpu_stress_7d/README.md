# GPU Stress Test (7 days)

一个基于 PyTorch CUDA GEMM 的长时间 GPU 压力测试。默认运行 7 天，支持多卡、
`tmux` 后台运行、温度保护和独立日志。

## 依赖

- NVIDIA 驱动和 `nvidia-smi`
- 支持 CUDA 的 PyTorch
- `tmux`
- Bash 5.1 或更高版本

## 启动

单卡 GPU 0：

```bash
./start_gpu_stress_7d.sh 0
```

指定多卡：

```bash
./start_gpu_stress_7d.sh 0,1,2,3
```

全部可见 GPU：

```bash
./start_gpu_stress_7d.sh all
```

进入 tmux：

```bash
tmux attach -t gpu-stress-7d
```

按 `Ctrl-b`、再按 `d` 可退出 tmux 并保持测试运行。

## 停止

```bash
./stop_gpu_stress_7d.sh
```

## 参数

参数通过环境变量设置：

```bash
MAX_TEMP=80 \
RESUME_TEMP=75 \
DURATION_SECONDS=604800 \
PYTHON_BIN=/path/to/python \
./start_gpu_stress_7d.sh all
```

常用变量：

- `DURATION_SECONDS`：运行秒数，默认 `604800`（7 天）。
- `MAX_TEMP`：达到该温度时暂停计算，默认 `83` 摄氏度。
- `RESUME_TEMP`：降至该温度后恢复，默认 `78` 摄氏度。
- `CHECK_INTERVAL`：状态检查间隔，默认 10 秒。
- `MEMORY_FRACTION`：三个矩阵占当前空闲显存的目标比例，默认 `0.45`。
- `MATRIX_SIZE`：手动指定方阵维度；默认 `0` 表示自动选择。
- `PYTHON_BIN`：使用的 Python 解释器，默认 `python3`。
- `SESSION_NAME`：tmux 会话名，默认 `gpu-stress-7d`。

若默认 `python3` 没有 CUDA PyTorch，启动器会扫描 `conda env list`，自动选择第一套
能够访问 GPU 的 PyTorch 环境。显式设置了 `PYTHON_BIN` 时不会自动切换环境，而是
在该解释器不可用时直接报错。

日志保存在 `logs/<UTC 启动时间>/`。建议先观察 10–30 分钟温度、功耗与错误日志，
确认散热稳定后再进行长时间测试。

启动器还会写入 `logs/launcher_<UTC 启动时间>.log`。如果 tmux 会话在启动阶段
退出，启动器会自动打印该日志中的错误。启动器会把当前 `python3` 解析成绝对路径，
避免已有 tmux server 使用另一套 Conda/Python 环境。
