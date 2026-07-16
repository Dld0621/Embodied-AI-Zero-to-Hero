# RobotDev-Setup-Guide

<div align="center">

**Robotics & Embodied AI Development Environment Setup Guide**

从零搭建机器人与具身智能开发环境 · 9 篇安装指南 · 4 个自动化脚本（+3 计划中） · 涵盖 ROS 2 / MuJoCo / Isaac Lab / CUDA 全工具链

[![Linux](https://img.shields.io/badge/Platform-Ubuntu%2022.04%20%7C%2024.04-orange)](https://ubuntu.com/download)
[![Windows](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20WSL2-blue)](https://www.microsoft.com/windows)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble%20%7C%20Jazzy-34b276)](https://docs.ros.org/en/jazzy/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-00a98f)](https://mujoco.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.0+-76B900)](https://isaac-sim.github.io/IsaacLab/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Genesis](https://img.shields.io/badge/Genesis-Physics-9cf)](https://genesis-robotics.github.io/)
[![Guides](https://img.shields.io/badge/Guides-9-blue)](docs/)
[![Scripts](https://img.shields.io/badge/Automation_Scripts-4%20%2B%203_planned-green)](scripts/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

[快速开始](#快速开始) · [安装指南](#方案-b按需查阅文档) · [兼容矩阵](#环境版本兼容矩阵) · [常见问题](docs/09-Troubleshooting.md)

</div>

---

## 这是什么？

刚进实验室，面对一台新电脑，不知道从哪装起？ROS 2、MuJoCo、CUDA 版本冲突装到崩溃？本项目帮你一次性解决。

**RobotDev-Setup-Guide** 为机器人学、具身智能、强化学习方向的研究者提供**一站式环境配置指南**，覆盖从裸机到完整开发环境的全工具链：

| 类别 | 软件栈 | 支持平台 |
|------|--------|----------|
| **机器人中间件** | ROS 2 (Humble / Jazzy), Gazebo Harmonic | Linux |
| **物理仿真** | MuJoCo 3.x, NVIDIA Isaac Lab, Genesis, PyBullet | Linux / Windows |
| **GPU 计算** | NVIDIA CUDA 12.x, cuDNN 9.x | Linux / Windows |
| **深度学习** | PyTorch, TensorFlow, JAX | Linux / Windows |
| **Python 环境** | Miniconda, pip, venv | Linux / Windows / macOS |
| **开发工具** | VS Code, Git, Docker, WSL2 | 跨平台 |

---

## 快速开始

### 方案 A：一键脚本安装（推荐 Ubuntu 用户）

```bash
# 克隆仓库
git clone https://github.com/Dld0621/RobotDev-Setup-Guide.git
cd RobotDev-Setup-Guide

# 运行环境检测
bash scripts/check_env.sh

# 一键安装基础工具链（Conda + CUDA + PyTorch）
bash scripts/install_base.sh   # [计划中]

# 安装 ROS 2（Ubuntu 24.04 自动选择 Jazzy，22.04 选择 Humble）
bash scripts/install_ros2.sh

# 安装 MuJoCo + 常用仿真环境
bash scripts/install_mujoco.sh
```

### 方案 B：按需查阅文档

直接跳转到对应安装指南：

- [ROS 2 安装指南](docs/01-ROS2-installation.md) — Ubuntu 22.04/24.04
- [MuJoCo 安装指南](docs/02-MuJoCo-installation.md) — Linux + Windows
- [Isaac Lab 安装指南](docs/03-IsaacLab-installation.md) — GPU 仿真平台
- [Gazebo Harmonic 安装指南](docs/04-Gazebo-installation.md) — ROS 2 集成仿真
- [Genesis 安装指南](docs/05-Genesis-installation.md) — 新一代物理引擎
- [CUDA & cuDNN 安装指南](docs/06-CUDA-installation.md) — GPU 驱动与计算工具包
- [Python 环境配置指南](docs/07-Python-environment.md) — Conda / venv / pip
- [Windows WSL2 配置指南](docs/08-WSL2-setup.md) — Windows 子系统 for Linux
- [常见问题排查](docs/09-Troubleshooting.md) — 错误诊断与解决方案

---

## 环境版本兼容矩阵

> **核心原则：Ubuntu 版本与 ROS 2 版本严格绑定！**

| Ubuntu | ROS 2 | Gazebo | CUDA 推荐 | MuJoCo | Python |
|--------|-------|--------|-----------|--------|--------|
| 22.04 LTS | Humble (LTS) | Fortress / Harmonic | 11.8 / 12.x | 3.x | 3.10 |
| 24.04 LTS | Jazzy (LTS) | Harmonic | 12.x | 3.x | 3.12 |

---

## 目录结构

```
RobotDev-Setup-Guide/
├── README.md                    # 本文件
├── docs/
│   ├── 01-ROS2-installation.md
│   ├── 02-MuJoCo-installation.md
│   ├── 03-IsaacLab-installation.md
│   ├── 04-Gazebo-installation.md
│   ├── 05-Genesis-installation.md
│   ├── 06-CUDA-installation.md
│   ├── 07-Python-environment.md
│   ├── 08-WSL2-setup.md
│   └── 09-Troubleshooting.md
├── scripts/
│   ├── check_env.sh             # 环境检测脚本
│   ├── install_ros2.sh          # ROS 2 自动安装
│   ├── install_mujoco.sh        # MuJoCo 自动安装
│   ├── install_cuda.sh          # CUDA 安装脚本
│   ├── install_base.sh          # [计划中] 基础工具链安装
│   ├── install_isaac_lab.sh     # [计划中] Isaac Lab 安装辅助
│   ├── setup_windows.ps1        # [计划中] Windows PowerShell 配置脚本
│   └── install_wsl2.ps1         # [计划中] WSL2 安装脚本
└── assets/
    └── (示意图与截图)
```

---

## 推荐安装顺序

对于从零开始的新手，建议按以下顺序配置：

1. **操作系统** — 安装 Ubuntu 24.04 双系统 或 Windows WSL2
2. **Python 环境** — 安装 Miniconda，创建虚拟环境
3. **GPU 驱动 & CUDA** — 安装 NVIDIA 驱动 + CUDA Toolkit + cuDNN
4. **深度学习框架** — PyTorch / TensorFlow (GPU 版)
5. **ROS 2** — 按系统版本安装对应 ROS 2 发行版
6. **仿真软件** — MuJoCo / Isaac Lab / Gazebo (按研究方向选)
7. **开发工具** — VS Code + ROS 插件 + Docker

---

## 贡献指南

欢迎提交 Issue 和 PR！如果你发现：

- 某个安装步骤有误或过时
- 某个常见问题缺少解决方案
- 想添加新的软件安装指南（如 DexGraspNet、LEAP Hand 等）

请遵循以下格式：

1. 文档放在 `docs/` 目录，命名为 `NN-软件名-installation.md`
2. 脚本放在 `scripts/` 目录，添加可执行权限
3. 在本 README 的目录结构中更新链接

---

## 相关项目

- [Embodied-AI-Paper-Analysis](https://github.com/Dld0621/Embodied-AI-Paper-Analysis) — 具身智能论文分析
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — MuJoCo 机器人模型库
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — NVIDIA 机器人仿真框架

---

## 许可证

MIT License - 自由使用、修改和分发。

---

**如果这个项目对你有帮助，请给一个 Star ⭐**
