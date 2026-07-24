# RL 零到一：强化学习训练灵巧手策略

> **目标**: 理解强化学习 (RL) 的核心概念，使用 Stable-Baselines3 + Gymnasium-Robotics 在 Shadow Hand 灵巧手上训练操作策略，从零到能抓取物体的策略。

---

## 目录

1. [什么是强化学习？](#1-什么是强化学习)
2. [为什么选 SB3 + Gymnasium-Robotics？](#2-为什么选-sb3--gymnasium-robotics)
3. [环境准备](#3-环境准备)
4. [第一次训练（10 分钟）](#4-第一次训练10-分钟)
5. [RL 核心概念速览](#5-rl-核心概念速览)
6. [Shadow Hand 环境详解](#6-shadow-hand-环境详解)
7. [SAC 算法解析](#7-sac-算法解析)
8. [HER 做了什么](#8-her-做了什么)
9. [训练可视化与分析](#9-训练可视化与分析)
10. [六大开源 RL 框架对比](#10-六大开源-rl-框架对比)
11. [进阶：自定义训练](#11-进阶自定义训练)
12. [常见问题排查](#12-常见问题排查)
13. [参考文献](#13-参考文献)

---

## 1. 什么是强化学习？

**强化学习 (Reinforcement Learning, RL)** 是一种让智能体通过与环境的交互来学习最优策略的机器学习方法。

```
     ┌─────────────────────────────────┐
     │          RL 循环                 │
     │                                  │
     │  Agent ──action──► Environment   │
     │    ▲                   │         │
     │    └──state, reward───┘         │
     └─────────────────────────────────┘
```

**核心三要素**:

| 要素 | 符号 | 含义 | 在灵巧手中的例子 |
|------|------|------|----------------|
| **状态 (State)** | s | 智能体观察到的环境信息 | 关节角、指尖位置、物体位姿 |
| **动作 (Action)** | a | 智能体做出的决策 | 24 个关节的目标角度 |
| **奖励 (Reward)** | r | 环境对动作的反馈 | 指尖离物体越近，奖励越高 |

**目标**: 学习策略 π(a|s)，使累积奖励最大化。

---

## 2. 为什么选 SB3 + Gymnasium-Robotics？

| 框架 | 难度 | 硬件 | 灵巧手任务 | 适合 |
|------|------|------|-----------|------|
| **SB3 + Gymnasium-Robotics** | **极低** | **CPU 可运行** | **Shadow Hand** | 入门学习 |
| Isaac Lab + rl_games | 中高 | NVIDIA GPU | Allegro, Shadow | 专业研究 |
| DexterousHands | 中 | NVIDIA GPU | 双灵巧手 | 双臂操作 |
| ManiSkill3 | 低 | 推荐 GPU | 灵巧操作 | 轻量入门 |

**选 SB3 + Gymnasium-Robotics 的理由**:
1. **一行安装**: `pip install stable-baselines3 gymnasium-robotics`
2. **CPU 可训练**: 不需要 GPU（训练时间会久一些，但能跑通）
3. **算法完备**: SAC + HER 是灵巧手 RL 的经典组合
4. **文档丰富**: SB3 是 GitHub 13k+ stars 的 RL 标准库
5. **Shadow Hand 环境**: 24-DoF 灵巧手 + 方块操作

---

## 3. 环境准备

### 3.1 安装

```bash
pip install stable-baselines3 gymnasium-robotics
```

> 这两个包会自动安装 numpy、gymnasium、mujoco 等依赖。

### 3.2 验证安装

```python
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

env = gym.make("HandManipulateBlock-v1")
print(f"状态维度: {env.observation_space.shape}")
print(f"动作维度: {env.action_space.shape}")
```

---

## 4. 第一次训练（10 分钟）

### 4.1 最简训练代码

```python
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import gymnasium_robotics

# 创建环境
env = gym.make("HandManipulateBlock-v1", render_mode="human")

# 创建 SAC + HER 模型
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log="./shadow_hand_tensorboard/",
)

# 训练
model.learn(total_timesteps=50_000)

# 保存模型
model.save("shadow_hand_block")
```

### 4.2 使用项目脚本

```bash
cd examples
python rl_demo.py --mode train --env HandManipulateBlock-v1 --timesteps 50000
```

### 4.3 测试训练好的策略

```bash
python rl_demo.py --mode enjoy --model shadow_hand_block --env HandManipulateBlock-v1
```

---

## 5. RL 核心概念速览

### 5.1 关键术语

| 术语 | 解释 | 代码对应 |
|------|------|---------|
| **Episode** | 一次完整的任务尝试 | 从初始状态到终止条件 |
| **Step** | 单次交互：观察 → 动作 → 奖励 | `env.step(action)` |
| **Policy** | 策略函数，状态 → 动作 | `model.predict(obs)` |
| **Replay Buffer** | 存储历史经验，从中采样训练 | `HerReplayBuffer` |
| **Value Function** | 估计状态/动作的好坏 | Critic 网络 |
| **Discount Factor γ** | 未来奖励的折现率 | 通常 0.95-0.99 |

### 5.2 RL 算法族谱

```
强化学习
├── Model-Free（无模型）
│   ├── Policy Gradient (策略梯度)
│   │   ├── PPO (Proximal Policy Optimization) ← 最稳定
│   │   └── TRPO
│   ├── Actor-Critic
│   │   ├── SAC (Soft Actor-Critic) ← 连续控制首选
│   │   ├── TD3
│   │   └── A2C/A3C
│   └── Value-Based
│       ├── DQN (Deep Q-Network) ← 离散动作
│       └── Rainbow
│
└── Model-Based（有模型）
    ├── DreamerV3 ← 学习世界模型
    ├── TD-MPC2 ← 模型预测控制
    └── MBRL
```

### 5.3 为什么灵巧手 RL 用 SAC + HER？

| 挑战 | 解决方案 |
|------|---------|
| **高维连续动作** (24-DoF) | SAC 原生支持连续动作空间 |
| **稀疏奖励** (只有抓取成功才给奖励) | HER 将失败经历重标记为"成功" |
| **探索困难** (24 个关节需要协调) | SAC 的熵正则化鼓励探索 |
| **训练不稳定** | SAC 的自动温度调节 |

---

## 6. Shadow Hand 环境详解

### 6.1 可用环境

| 环境 ID | 任务 | 难度 |
|---------|------|------|
| `HandReach-v1` | 手指到达目标位置 | ⭐ |
| `HandManipulateBlock-v1` | 旋转方块到目标位姿 | ⭐⭐⭐ |
| `HandManipulateEgg-v1` | 旋转鸡蛋到目标位姿 | ⭐⭐⭐ |
| `HandManipulatePen-v1` | 旋转笔到目标位姿 | ⭐⭐⭐ |

### 6.2 状态空间

```
观察 = {
    "observation":  [24 关节角 + 24 关节速度 + 24 ctrl + 物体位姿]  (约 88 维)
    "achieved_goal": [物体当前位姿 × 7]  (物体位置 + 四元数)
    "desired_goal":  [物体目标位姿 × 7]
}
```

### 6.3 动作空间

```
动作 = [24 个关节的绝对角度]  ∈ [-1, 1]  (归一化后)
```

### 6.4 奖励函数

```
reward = -distance(achieved_goal, desired_goal)
```

距离越近，奖励越高（负距离 → 鼓励靠近目标）。

---

## 7. SAC 算法解析

**SAC (Soft Actor-Critic)** 是当前连续控制最优秀的算法之一。

### 7.1 核心思想

```
SAC = Actor-Critic + 最大熵 + off-policy

Actor:    学习策略 π(a|s)，最大化 Q 值 + 熵
Critic:   学习 Q 值，评估动作好坏
Entropy:  鼓励策略保持随机性，促进探索
Off-Policy: 从 Replay Buffer 随机采样，样本效率高
```

### 7.2 损失函数

```python
# Actor 损失: 最小化 -(Q + α * entropy)
actor_loss = -mean(Q(s, π(s)) + α * H(π(s)))

# Critic 损失: 最小化 Bellman 误差
target = r + γ * (Q_next(s', π(s')) + α * H(π(s')))
critic_loss = mean((Q(s, a) - target)²)

# 温度 α 自动调节: 保持目标熵水平
alpha_loss = -α * (log_prob + target_entropy)
```

### 7.3 为什么 SAC 适合灵巧手？

| 特性 | 对灵巧手的意义 |
|------|--------------|
| **连续动作空间** | 24 个关节角度是连续值 |
| **熵正则化** | 避免过早收敛到次优手势 |
| **离线学习** | 重复利用历史数据，样本效率高 |
| **自动温度调节** | 降低调参负担 |

---

## 8. HER 做了什么

**HER (Hindsight Experience Replay)** 是灵巧手 RL 成功的关键。

### 8.1 问题：稀疏奖励

```
任务: 旋转方块到目标位姿
奖励: 只有距离 < 0.01 时给 +1，否则 0

→ 99.9% 的 episode 奖励为 0
→ 智能体学不到任何东西
```

### 8.2 HER 的解决方案

```python
# 原始 episode（失败）
episode = [(s, a, r=0, s', goal=目标A), ...]

# HER 重标记（变为"成功"）
# 把 episode 最后达到的位姿当作"目标"
for transition in episode:
    if random():
        new_goal = achieved_goal  # "我本来就想来这里"
        new_reward = compute_reward(achieved_goal, new_goal)  # = 0!
        # 现在这条轨迹变成了成功案例
```

**关键洞察**: 虽然智能体没有达到原定目标，但它确实达到了某个位姿。HER 把"失败"重标记为"达到了另一个目标"，从而学会如何从不同状态到达不同目标。

### 8.3 HER 参数

```python
HerReplayBuffer(
    n_sampled_goal=4,           # 每步采样 4 个虚拟目标
    goal_selection_strategy="future",  # 从同一 episode 的未来状态采样目标
    online_sampling=True,       # 在线采样（更快）
)
```

---

## 9. 训练可视化与分析

### 9.1 TensorBoard 监控

```bash
tensorboard --logdir ./shadow_hand_tensorboard/
```

**关键指标**:

| 指标 | 好的趋势 | 坏的趋势 |
|------|---------|---------|
| `rollout/ep_rew_mean` | 持续上升 | 震荡或下降 |
| `train/actor_loss` | 稳定在 0 附近 | 剧烈震荡 |
| `train/critic_loss` | 缓慢下降 | 发散 |
| `rollout/ep_len_mean` | 稳定在合理值 | 持续增长（不收敛） |

### 9.2 渲染测试

```python
model = SAC.load("shadow_hand_block")
obs, _ = env.reset()

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

### 9.3 成功率评估

```python
success_count = 0
for _ in range(100):
    obs, _ = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("is_success", False):
            success_count += 1
            break

print(f"成功率: {success_count}%")
```

---

## 10. 六大开源 RL 框架对比

| 框架 | Stars | 灵巧手 | GPU | 安装 | 适合 |
|------|-------|--------|-----|------|------|
| **SB3 + Gym-Robotics** | 13k | Shadow | 可选 | `pip install` | 入门 |
| **Isaac Lab** | 7.7k | Allegro, Shadow | 必须 NVIDIA | 复杂 | 专业 |
| **DexterousHands** | 867 | 双灵巧手 | 必须 NVIDIA | 中 | 双臂 |
| **ManiSkill3** | 2.2k | 灵巧操作 | 推荐 | `pip install` | 轻量 |
| **rl_games** | 1k | Shadow, Allegro | 必须 NVIDIA | 中 | 高性能 |
| **SKRL** | 500 | 可对接 | 可选 | `pip install` | JAX 加速 |

---

## 11. 进阶：自定义训练

### 11.1 调整超参

```python
model = SAC(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,          # 学习率
    buffer_size=1_000_000,       # 缓冲区大小
    batch_size=256,              # 批次大小
    gamma=0.95,                  # 折扣因子
    tau=0.005,                   # 目标网络软更新率
    ent_coef="auto",             # 自动熵调节
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    policy_kwargs=dict(
        net_arch=[256, 256, 256],  # 网络结构
    ),
    verbose=1,
)
```

### 11.2 并行训练（加速）

```python
from stable_baselines3.common.env_util import make_vec_env

# 创建 4 个并行环境
env = make_vec_env("HandManipulateBlock-v1", n_envs=4)

model = SAC("MultiInputPolicy", env, ...)
model.learn(total_timesteps=200_000)  # 4x 速度
```

### 11.3 加载预训练模型继续训练

```python
model = SAC.load("shadow_hand_block", env=env)
model.learn(total_timesteps=50_000, reset_num_timesteps=False)
model.save("shadow_hand_block_v2")
```

---

## 12. 常见问题排查

### Q1: 训练不收敛

```python
# 检查是否用了 HER
assert isinstance(model.replay_buffer, HerReplayBuffer)

# 检查奖励函数是否正确
obs, _ = env.reset()
for i in range(10):
    action = env.action_space.sample()
    obs, reward, _, _, info = env.step(action)
    print(f"Step {i}: reward={reward:.3f}")
```

### Q2: 内存不足

```python
# 减小 buffer 和 batch
model = SAC(..., buffer_size=100_000, batch_size=64)
```

### Q3: 训练太慢

```python
# 选项 1: 并行环境
env = make_vec_env("HandManipulateBlock-v1", n_envs=4)

# 选项 2: GPU 加速
model = SAC(..., device="cuda")

# 选项 3: 减少训练步数，先跑通
model.learn(total_timesteps=10_000)
```

### Q4: 策略总是做同样的动作

```python
# 减少确定性，增加探索
model = SAC(..., ent_coef=0.1)  # 增大熵系数

# 测试时使用随机策略
action, _ = model.predict(obs, deterministic=False)
```

---

## 13. 参考文献

1. **SAC**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", ICML 2018.
2. **HER**: Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017.
3. **Stable-Baselines3**: Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations", JMLR 2021. [GitHub](https://github.com/DLR-RM/stable-baselines3)
4. **Gymnasium-Robotics**: Farama Foundation, [GitHub](https://github.com/Farama-Foundation/Gymnasium-Robotics)
5. **Isaac Lab**: NVIDIA, [GitHub](https://github.com/isaac-sim/IsaacLab)
6. **DexterousHands**: PKU-MARL, [GitHub](https://github.com/PKU-MARL/DexterousHands)
7. **ManiSkill3**: Haosu Lab, RSS 2025. [GitHub](https://github.com/haosulab/ManiSkill)

---

## 附录：命令速查表

```bash
# === 安装 ===
pip install stable-baselines3 gymnasium-robotics

# === 训练 ===
cd examples
python rl_demo.py --mode train --timesteps 50000
python rl_demo.py --mode train --env HandManipulatePen-v1 --timesteps 100000

# === 测试 ===
python rl_demo.py --mode enjoy --model shadow_hand_block

# === 评估 ===
python rl_demo.py --mode eval --model shadow_hand_block --episodes 100

# === 监控 ===
tensorboard --logdir ./shadow_hand_tensorboard/
```

---

> **本文目标达成**: 你理解了 RL 的核心概念（状态、动作、奖励、SAC、HER），能够在 CPU 上训练 Shadow Hand 灵巧手操作策略，并掌握了 RL 框架的横向对比。这就是 RL 的 0→1。