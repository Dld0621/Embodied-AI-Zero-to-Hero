# Isaac Lab Setup Contract / Isaac Lab 环境契约

> [Environment hub](README.md) · [中文指南](#中文指南)

## English guide

Isaac Lab is tightly coupled to Isaac Sim, a specific Python version, supported host platforms, GPU memory, and NVIDIA drivers. Those constraints change. The [current official installation page](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/) is the source of truth; this repository intentionally does not freeze a “latest” version in prose.

### Installation decision

1. Open the official page and record its update date.
2. Check the exact OS, CPU architecture, RAM, VRAM, driver, Isaac Sim, Isaac Lab, and Python requirements.
3. Use the upstream-recommended beginner route. At this review, it installs Isaac Sim through its documented package route and Isaac Lab from source, but the commands and pins must be copied from the current page.
4. Create a dedicated environment. Do not reuse the ROS system Python or a general VLA environment.
5. Run the upstream compatibility checker before downloading large assets when available.

### Evidence ladder

| Check | Evidence |
|---|---|
| Import | Isaac Sim and Isaac Lab modules load in the selected interpreter. |
| Application | The simulator launches and closes cleanly. |
| Minimal scene | An official empty-scene tutorial renders or runs headless. |
| Environment | One official task resets and steps with the recorded device. |
| Training smoke | A short run writes logs and a checkpoint; it is not a converged policy. |
| Benchmark | Fixed task, seeds, steps, hardware, and success definition are retained. |

### Common boundary errors

- An Omniverse-era tutorial may target retired package names or launch paths.
- `nvidia-smi` working does not prove the selected Python environment has a compatible PyTorch or Isaac stack.
- A viewport is application evidence, not task success.
- A short training command is execution evidence, not a reproduced paper result.

## 中文指南

Isaac Lab 与 Isaac Sim、指定 Python 版本、宿主平台、显存和 NVIDIA 驱动强耦合，而且这些要求会变化。因此必须把[当前官方安装页](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/)作为唯一安装真源，本仓库不复制一个很快会过期的“最新版”命令。

安装前记录官方页面更新时间与精确的 OS、架构、内存、显存、驱动、Isaac Sim、Isaac Lab、Python 组合；为它创建独立环境，不与 ROS 系统 Python 或 VLA 环境混用。优先使用官方当前推荐的新手路径，并在下载大体积资源前运行兼容性检查器。

验收必须分层：模块导入、应用启动、最小场景、环境 reset/step、短训练产物、固定协议 Benchmark。窗口能打开不代表任务成功，短训练能运行也不代表复现论文结果；WSL2 支持不能由 Windows 或 Ubuntu 原生支持自行推导。
