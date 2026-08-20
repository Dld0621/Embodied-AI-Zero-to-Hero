# Layered Troubleshooting / 分层故障排查

> [Environment hub](README.md) · [中文排查表](#中文排查表)

## English decision table

| Symptom | First read-only checks | Likely boundary | Safe next move |
|---|---|---|---|
| `python` imports the wrong package | `python -c "import sys; print(sys.executable)"`; `python -m pip --version` | Interpreter and installer differ | Re-enter the intended environment; install through `python -m pip`. |
| ROS package is missing | `printenv ROS_DISTRO`; inspect `AMENT_PREFIX_PATH`; `ros2 pkg prefix <name>` | Underlay/overlay sourcing | Source the matching underlay, then workspace overlay, in a clean shell. |
| `colcon build` fails | Read the first compiler error; run `rosdep check` from the workspace | Dependency or source error | Fix the first failing package; avoid deleting every cache initially. |
| Gazebo launches but ROS sees no topic | Inspect topics and types on both sides; check simulation clock and namespace | Bridge configuration | Compare the bridge config with the current `ros_gz` message-support table. |
| MuJoCo simulates but cannot render | Record OS, session type, and OpenGL backend | Rendering context | Select a documented GLX/EGL/OSMesa path; keep physics and rendering tests separate. |
| PyTorch cannot see CUDA | Record `nvidia-smi`, Python executable, `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` | Driver, wheel, or device exposure | Re-select the official wheel for the host; do not reinstall drivers blindly. |
| Isaac Lab fails during import or launch | Compare exact Python, Isaac Sim, Isaac Lab, OS, and driver versions with current docs | Coupled compatibility matrix | Recreate a dedicated supported environment from the current upstream route. |
| Genesis is unexpectedly slow | Record backend, first-run compilation, rendering, scene size, and environment count | Fallback or measurement protocol | Confirm selected backend; separate warm-up from steady-state measurement. |
| WSL GUI or GPU is unavailable | `wsl --status`; `wsl --version`; Windows driver status | WSL/WSLg/driver integration | Update through documented Windows paths; do not install a Linux display driver in WSL. |

## Stop conditions

Do not “fix” an environment by disabling TLS verification, adding an unreviewed `trusted-host`, piping an unknown network script into a shell, replacing system Python globally, or installing a random driver version from a static tutorial. Preserve the first error, package list, and environment receipt before making a change.

For a physical robot, stop at the software boundary. Driver discovery, message traffic, or simulator motion does not authorize actuators. Calibration, limits, watchdog, emergency stop, operator ownership, and a hardware-specific test plan are separate requirements.

## 中文排查表

| 现象 | 首轮只读检查 | 常见边界 | 安全处理 |
|---|---|---|---|
| Python 导入了错误包 | 检查 `sys.executable` 与 `python -m pip --version` | 解释器与安装器不一致 | 重新进入目标环境，并始终通过 `python -m pip` 安装。 |
| ROS 找不到包 | 检查 `ROS_DISTRO`、`AMENT_PREFIX_PATH` 与 `ros2 pkg prefix` | Underlay/Overlay 未按序加载 | 在干净终端先 source 系统层，再 source 工作空间。 |
| `colcon build` 失败 | 保留第一个编译错误，运行 `rosdep check` | 源码或依赖 | 先修第一个失败包，不要直接删除所有缓存。 |
| Gazebo 有窗口但 ROS 无 Topic | 核对两侧 Topic、类型、时钟与命名空间 | 桥接配置 | 对照当前 `ros_gz` 支持表修正配置。 |
| MuJoCo 有物理但无画面 | 记录 OS、会话类型与 OpenGL 后端 | 渲染上下文 | 按官方 GLX/EGL/OSMesa 路径处理，物理与渲染分开验收。 |
| PyTorch 看不到 CUDA | 同时记录驱动、解释器、Torch 版本、运行时与可用性 | 驱动/Wheel/设备暴露 | 重新从官方选择器选 Wheel，不盲目重装驱动。 |
| Isaac Lab 导入或启动失败 | 与当前官方矩阵逐项对照版本 | 强耦合兼容 | 重建独立且受支持的环境。 |
| WSL GUI/GPU 不可用 | 检查 WSL 状态、版本与 Windows 驱动 | WSLg/驱动集成 | 走 Windows 官方更新路径，不在 WSL 安装 Linux 显示驱动。 |

禁止通过关闭 TLS 校验、加入未审查 `trusted-host`、执行未知网络脚本、全局替换系统 Python 或照搬静态教程中的驱动版本来“修复”环境。任何修改前先保存第一条错误、包清单和环境回执。
