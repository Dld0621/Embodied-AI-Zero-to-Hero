# 机器人开发环境

> [English](README.md) · 简体中文 · [兼容矩阵](stack-matrix.md) · [迁移记录](MIGRATION.md)

本模块用于把一台新工作站配置成**可检查、可复现的具身智能开发环境**。内容覆盖版本选择、环境隔离、Smoke Test、仿真器选择、ROS 接入与实验记录；“安装命令成功”不等于训练、渲染、任务或真实机器人已经验证。

## 先选择宿主路径

| 目标 | 建议起点 | 原因 |
|---|---|---|
| ROS 2 与 Gazebo 入门 | Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic | 经审阅的 LTS 教学组合，`ros_gz` 由 ROS 仓库提供。 |
| 维护 Ubuntu 22.04 机器人栈 | ROS 2 Humble + Gazebo Fortress | 保持 Humble 的默认二进制组合。 |
| Windows 笔记本学习 ROS | WSL2 + Ubuntu 24.04 | 使用 Linux ROS 软件包和 WSLg，但不把 WSL 当成实时真机证据。 |
| 接触仿真与模型编辑 | 原生 Python + MuJoCo | 安装轻、MJCF/URDF 路径直接、物理与渲染边界清楚。 |
| GPU 并行机器人学习 | Isaac Lab 或 Genesis World | 安装前必须重新核对上游 OS、Python、驱动与加速器矩阵。 |

下面两个命令只读取信息，不修改系统：

```bash
python tools/robotdev/stack_resolver.py --host wsl2 --ubuntu 24.04
bash tools/robotdev/check_env.sh
```

## 环境搭建 Pipeline

```text
研究目标
    ↓
宿主机与加速器盘点
    ↓
从官方文档选择受支持版本组合
    ↓
隔离环境与依赖锁定
    ↓
导入 → 最小场景 → 渲染 → ROS 桥接
    ↓
任务 Smoke → 确定性回归 → Benchmark
    ↓
硬件专属安全审查（独立门禁）
```

| 门禁 | 通过条件 | 不能证明 |
|---|---|---|
| 导入 | 包可以导入并报告版本。 | 渲染、物理任务或 GPU 正确。 |
| 场景 | 最小世界可构建并推进。 | 机器人模型具有真实物理保真度。 |
| 渲染 | 可生成交互画面或离屏帧。 | 训练实际使用了预期加速器。 |
| 桥接 | 指定消息按时间戳跨越 ROS 与仿真器。 | 控制延迟稳定或真机安全。 |
| 任务 | 固定种子与指标通过限定任务。 | 泛化或 Sim-to-Real 迁移。 |

## 分项指南

| 指南 | 学习结果 |
|---|---|
| [兼容矩阵](stack-matrix.md) | 选择宿主路径，并为易变版本结论记录日期。 |
| [ROS 2 + Gazebo](ros2-gazebo.md#中文指南) | 构建工作空间、使用默认组合并检查桥接边界。 |
| [MuJoCo](mujoco.md#中文指南) | 安装官方 Python 绑定并运行正确的 MJCF Smoke。 |
| [Isaac Lab](isaac-lab.md#中文指南) | 遵循当前官方安装路线，区分启动证据与任务证据。 |
| [Genesis World](genesis.md#中文指南) | 安装当前软件包并执行官方最小场景模式。 |
| [Python、CUDA 与 WSL2](python-cuda-wsl.md#中文指南) | 隔离环境，避免混淆驱动、Toolkit 与框架运行时。 |
| [故障排查](troubleshooting.md#中文排查表) | 按层定位问题，不使用危险的全局绕过。 |

## 可复现环境回执

进入研究基线的每套环境都应记录：

- 宿主 OS，以及原生、WSL2、容器还是远程环境；
- Python、仿真器、ROS、Gazebo、PyTorch、驱动与锁文件版本；
- GPU、CPU 架构、内存与渲染后端；
- 安装来源、完整命令、Git 提交、随机种子与 Smoke 输出；
- 证据等级：导入、场景、渲染、桥接、任务、基准或真机。

机器可读兼容数据位于 [`tools/robotdev/stack_matrix.json`](../../tools/robotdev/stack_matrix.json)。其中的审阅日期是内容新鲜度边界，不代表外部依赖永远不变。
