# Isaac Lab 安装指南

> NVIDIA Isaac Lab — GPU 加速机器人仿真框架

## 简介

Isaac Lab 是 NVIDIA 基于 Omniverse 构建的机器人仿真平台，是 Isaac Gym 的继任版本。支持 GPU 并行物理仿真、逼真渲染、传感器仿真，广泛用于大规模强化学习训练。

**硬件要求：**
- NVIDIA GPU (RTX 30 系列以上推荐)
- Ubuntu 22.04 / 24.04
- 至少 32 GB RAM（推荐 64 GB）
- 200+ GB 磁盘空间

---

## 前置条件

1. **NVIDIA 驱动** >= 550.x
2. **CUDA** >= 12.0
3. **Python** 3.10 / 3.12

---

## 安装方式一：Isaac Sim App (推荐新手)

### Step 1: 安装 Omniverse Launcher

```bash
# 下载 Omniverse Launcher
wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.sh
chmod +x omniverse-launcher-linux.sh
./omniverse-launcher-linux.sh
```

### Step 2: 通过 Launcher 安装 Isaac Sim

1. 打开 Omniverse Launcher
2. 进入 Exchange 标签页
3. 搜索 "Isaac Sim" 并安装最新版本 (6.0+)

### Step 3: 克隆并安装 Isaac Lab

```bash
# 克隆 Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 创建 conda 环境
conda create -n isaaclab python=3.12 -y
conda activate isaaclab

# 安装依赖
pip install --upgrade pip
pip install -e .

# 安装 Isaac Sim 额外依赖
pip install isaaclab --extra-index-url https://pypi.nvidia.com
```

### Step 4: 验证安装

```bash
# 运行基础测试
python -m omni.isaac.lab.app -h

# 启动可视化
python -m omni.isaac.lab.app --headless
```

---

## 安装方式二：Isaac Lab Standalone (推荐进阶用户)

```bash
# 克隆仓库
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 运行安装脚本（自动下载 Isaac Sim）
./isaaclab.sh --install

# 激活环境
source isaaclab.sh --conda
```

---

## 常用环境变量

```bash
# 添加到 ~/.bashrc
export ISAACLAB_PATH="/path/to/IsaacLab"
export ISAACSIM_PATH="${ISAACLAB_PATH}/_isaac_sim"
```

---

## 运行示例

```bash
# 启动基础场景
python source/standalone/tutorials/00_sim/spawn_prims.py

# RL 训练示例（CartPole）
python source/standalone/tutorials/02_rl/run_rl.py --task Isaac-CartPole-v0 --num_envs 64

# 强化学习训练（ humanoid ）
python source/standalone/tutorials/02_rl/run_rl.py --task Isaac-Humanoid-v0 --num_envs 4096

# 导航任务
python source/standalone/tutorials/02_rl/run_rl.py --task Isaac-Velocity-Flat-Anymal-C-v0 --num_envs 1024
```

---

## 常见问题

### `omni.kit.usd` ModuleNotFoundError

```bash
# 通常是因为 Isaac Sim 没有正确链接
pip install --upgrade isaaclab --extra-index-url https://pypi.nvidia.com

# 确认 Isaac Sim 路径
echo $ISAACSIM_PATH
```

### GPU 显存不足

```bash
# 减少并行环境数
python run_rl.py --task Isaac-Humanoid-v0 --num_envs 512

# 使用 headless 模式
python run_rl.py --task Isaac-CartPole-v0 --headless
```

### NVIDIA 驱动版本过低

```bash
# 检查驱动版本
nvidia-smi

# 升级驱动（Ubuntu）
sudo apt install nvidia-driver-550
```

## 参考资料

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim 下载](https://developer.nvidia.com/isaac-sim)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac Lab 教程](https://isaac-sim.github.io/IsaacLab/source/tutorials/index.html)
