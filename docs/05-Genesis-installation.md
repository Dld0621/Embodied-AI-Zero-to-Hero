# Genesis 安装指南

> Genesis — 新一代生成式通用物理引擎

## 简介

Genesis 是一个面向通用机器人与具身智能学习的生成式物理仿真平台，支持：
- 多种物理求解器（刚体、软体、流体、气体）
- GPU 加速大规模并行仿真
- 可微仿真（支持梯度计算）
- 程序化场景生成

---

## 系统要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA RTX 3060 | NVIDIA RTX 4080+ |
| CUDA | 11.8+ | 12.0+ |
| RAM | 16 GB | 32 GB+ |
| 磁盘 | 10 GB | 50 GB+ |

---

## 安装

### Step 1: 克隆仓库

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
```

### Step 2: 安装依赖

```bash
# 创建 conda 环境
conda create -n genesis python=3.10 -y
conda activate genesis

# 安装 Genesis
pip install -e .

# 安装额外依赖
pip install torch torchvision  # 深度学习框架
pip install gymnasium           # 强化学习环境接口
```

### Step 3: 验证安装

```python
import genesis as gs

# 初始化引擎
gs.init(backend=gs.gpu)

# 创建场景
scene = gs.Scene(
    sim_options=gs.options.SimOptions(),
    viewer_options=gs.options.ViewerOptions(),
    show_viewer=True,
)

# 添加地面
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# 添加机器人
robot = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

# 运行仿真
for i in range(1000):
    scene.step()

print("Genesis 安装成功！")
```

---

## 快速训练示例

```python
import genesis as gs
import torch

gs.init(backend=gs.gpu)

# 创建并行环境
envs = gs.Scene(
    sim_options=gs.options.SimOptions(),
    envs=[gs.options.Env(num=1024)],
)

# 运行 PPO 训练
for epoch in range(100):
    obs = envs.reset()
    for step in range(500):
        actions = policy(obs)  # 你的策略网络
        obs, reward, done, info = envs.step(actions)
```

---

## 常见问题

### CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvcc --version

# 如需安装指定版本 CUDA，参考 06-CUDA-installation.md
```

### 编译错误

```bash
# 确保安装了 C++ 编译器
sudo apt install -y build-essential cmake

# 清除缓存重装
pip uninstall genesis
pip install -e . --no-cache-dir
```

## 参考资料

- [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis)
- [Genesis 文档](https://genesis-embodied-ai.github.io/Genesis/)
