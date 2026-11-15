# RL 零到一：强化学习训练机器人策略

> **逐点图解 / Concept close-ups：**[强化学习与后训练](knowledge-atlas/learning-reinforcement-learning/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> **目标**: 理解强化学习 (RL) 的核心概念，使用 Stable-Baselines3 + Gymnasium-Robotics 在 Fetch 机械臂上训练操作策略，从零到能抓取物体的策略。

---

## 目录

1. [什么是强化学习？](#1-什么是强化学习)
2. [为什么选 SB3 + Gymnasium-Robotics？](#2-为什么选-sb3--gymnasium-robotics)
3. [环境准备](#3-环境准备)
4. [第一次训练（10 分钟）](#4-第一次训练10-分钟)
5. [RL 核心概念速览](#5-rl-核心概念速览)
6. [Fetch 机械臂环境详解](#6-fetch-机械臂环境详解)
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

| 要素 | 符号 | 含义 | 在机器人中的例子 |
|------|------|------|----------------|
| **状态 (State)** | s | 智能体观察到的环境信息 | 关节角、末端位置、物体位姿 |
| **动作 (Action)** | a | 智能体做出的决策 | 7 个关节的目标增量 |
| **奖励 (Reward)** | r | 环境对动作的反馈 | 末端离物体越近，奖励越高 |

**目标**: 学习策略 π(a|s)，使累积奖励最大化。

---

## 2. 为什么选 SB3 + Gymnasium-Robotics？

| 框架 | 难度 | 硬件 | 机器人任务 | 适合 |
|------|------|------|-----------|------|
| **SB3 + Gymnasium-Robotics** | **极低** | **CPU 可运行** | **Fetch 机械臂** | 入门学习 |
| Isaac Lab + rl_games | 中高 | NVIDIA GPU | Franka, UR | 专业研究 |
| RoboSuite | 中 | NVIDIA GPU | 双臂操作 | 双臂操作 |
| ManiSkill3 | 低 | 推荐 GPU | 通用操作 | 轻量入门 |

**选 SB3 + Gymnasium-Robotics 的理由**:
1. **一行安装**: `pip install stable-baselines3 gymnasium-robotics`
2. **CPU 可训练**: 不需要 GPU（训练时间会久一些，但能跑通）
3. **算法完备**: SAC + HER 是机器人 RL 的经典组合
4. **文档丰富**: SB3 是 GitHub 13k+ stars 的 RL 标准库
5. **Fetch 环境**: 7-DoF 机械臂 + 方块操作

---

## 3. 环境准备

### 3.1 安装

```bash
pip install stable-baselines3 gymnasium-robotics
```

> 这两个包会自动安装 numpy、gymnasium、mujoco 等依赖。

> 本文保留仓库示例使用的 `-v2` 环境 ID，属于历史配置，不保证当前未锁版本安装仍可创建。首次运行先核对已安装环境注册表，并记录 Gymnasium / Gymnasium-Robotics / SB3 版本；不要在报告中混合不同版本的结果。

### 3.2 验证安装

```python
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPush-v2")
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
env = gym.make("FetchPush-v2", render_mode="human")

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
    tensorboard_log="./fetch_tensorboard/",
)

# 训练
model.learn(total_timesteps=100_000)

# 保存模型
model.save("fetch_push")
```

### 4.2 使用项目脚本

```bash
cd examples
python rl_demo.py --mode train --env FetchPush-v2 --timesteps 100000
```

### 4.3 测试训练好的策略

```bash
python rl_demo.py --mode enjoy --model fetch_push --env FetchPush-v2
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

### 5.3 为什么机器人 RL 用 SAC + HER？

| 挑战 | 解决方案 |
|------|---------|
| **连续动作** (7-DoF) | SAC 原生支持连续动作空间 |
| **稀疏奖励** (只有抓取成功才给奖励) | HER 将失败经历重标记为"成功" |
| **探索困难** (多个关节需要协调) | SAC 的熵正则化鼓励探索 |
| **训练不稳定** | SAC 的自动温度调节 |

---

## 6. Fetch 机械臂环境详解

### 6.1 可用环境

| 环境 ID | 任务 | 难度 |
|---------|------|------|
| `FetchReach-v2` | 末端到达目标位置 | ⭐ |
| `FetchPush-v2` | 推送方块到目标位置 | ⭐⭐ |
| `FetchPickAndPlace-v2` | 抓取并放置方块到目标位姿 | ⭐⭐⭐ |
| `FetchSlide-v2` | 滑动方块到目标位置 | ⭐⭐⭐ |

### 6.2 状态空间

```
观察 = {
    "observation":  [末端位置 + 物体位姿 + 物体速度 + 夹爪状态]  (约 25 维)
    "achieved_goal": [物体当前位置 × 3]
    "desired_goal":  [物体目标位置 × 3]
}
```

### 6.3 动作空间

```
动作 = [3 维末端增量 + 1 维夹爪]  ∈ [-1, 1]  (归一化后)
```

### 6.4 奖励函数

本文不带 `Dense` 后缀的 FetchPush 使用稀疏奖励，不能与距离奖励混读：

```
稀疏（默认）: 未到目标为 -1，到达阈值内为 0
稠密（Dense）: -distance(achieved_goal, desired_goal)
```

负距离只属于稠密版本。环境 ID 的版本号随安装版本变化；复现实验时记录完整 ID、`reward_type` 和时间上限，见 [FetchPush 官方奖励及回合定义](https://robotics.farama.org/envs/fetch/push/)。

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

### 7.3 为什么 SAC 适合机器人？

| 特性 | 对机器人的意义 |
|------|--------------|
| **连续动作空间** | 末端增量和夹爪开合是连续值 |
| **熵正则化** | 避免过早收敛到次优动作 |
| **Off-policy 学习** | 可复用回放数据；这不等于已适配固定数据集的 offline RL |
| **自动温度调节** | 降低调参负担 |

---

## 8. HER 做了什么

**HER (Hindsight Experience Replay)** 是机器人 RL 成功的关键。

### 8.1 问题：稀疏奖励

```
任务: 推送方块到目标位置
默认稀疏奖励: 到达约 0.05 m 阈值内为 0，否则 -1

尚未成功的不同动作可能得到相同奖励，缺少接近目标的距离信号。
这会增加探索难度，但不能推出“必然学不到”或固定失败比例。
```

### 8.2 HER 的解决方案

```text
# 原始 episode（失败）
episode = [(s, a, r=-1, s_next, goal=目标A), ...]

# HER 重标记（变为"成功"）
# 把 episode 最后达到的位姿当作"目标"
for transition in episode:
    if random():
        new_goal = 从同一回合当前转换或后续转换的 next_achieved_goal 中采样
        new_reward = env.compute_reward(transition.next_achieved_goal, new_goal, info)
        # 按新目标重新计算奖励；达到新目标的转换为 0，其余仍可为 -1
```

**关键洞察**: 虽然智能体没有达到原定目标，但它确实达到了某个位姿。HER 把"失败"重标记为"达到了另一个目标"，从而学会如何从不同状态到达不同目标。

### 8.3 HER 参数

```python
model = SAC(
    "MultiInputPolicy", env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,  # 用于决定 HER 重标记样本比例
        goal_selection_strategy="future",
    ),
)
```

这是承接前文已导入 `SAC` / `HerReplayBuffer` 和创建 `env` 的配置片段。[当前 SB3 接口](https://stable-baselines3.readthedocs.io/en/master/modules/her.html) 不接受旧的 `online_sampling` 参数；算法负责传入缓冲区容量、空间与环境等必需参数。

---

## 9. 训练可视化与分析

### 9.1 TensorBoard 监控

```bash
tensorboard --logdir ./fetch_tensorboard/
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
model = SAC.load("fetch_push")
obs, _ = env.reset()

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
```

### 9.3 成功率评估

```python
success_count = 0
for _ in range(100):
    obs, _ = env.reset()
    while True:  # env 必须带有限回合长度的 TimeLimit
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("is_success", False):
            success_count += 1
            break
        if terminated or truncated:
            break

print(f"成功率: {success_count}%")
```

这里定义的是 **100 个回合中，在各自时间上限内至少成功一次的比例**，不是回合结束时仍保持成功的比例。FetchPush 默认通常在 50 步截断；若要改为 100 步，应在创建环境时显式设置 `max_episode_steps=100` 并记录该协议，不能在截断后继续累计成功。

---

## 10. 六大开源 RL 框架对比

| 框架 | Stars | 机器人 | GPU | 安装 | 适合 |
|------|-------|--------|-----|------|------|
| **SB3 + Gym-Robotics** | 13k | Fetch | 可选 | `pip install` | 入门 |
| **Isaac Lab** | 7.7k | Franka, UR | 必须 NVIDIA | 复杂 | 专业 |
| **RoboSuite** | 867 | 双臂 | 必须 NVIDIA | 中 | 双臂 |
| **ManiSkill3** | 2.2k | 通用操作 | 推荐 | `pip install` | 轻量 |
| **rl_games** | 1k | Franka, UR | 必须 NVIDIA | 中 | 高性能 |
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

### 11.2 多环境采样与进程并行

```python
from stable_baselines3.common.env_util import make_vec_env

# 创建 4 个向量化环境；默认 DummyVecEnv 在同一进程依次执行
env = make_vec_env("FetchPush-v2", n_envs=4)

model = SAC("MultiInputPolicy", env)
model.learn(total_timesteps=200_000)
```

`n_envs=4` 不等于使用 4 个 CPU 核，也不保证 4 倍速度。需要进程并行时可显式选择 `SubprocVecEnv` 并在脚本入口保护下运行；是否更快要测量环境计算量和进程通信开销，见 [SB3 环境工具](https://stable-baselines3.readthedocs.io/en/master/common/env_util.html)。

### 11.3 加载预训练模型继续训练

```python
model = SAC.load("fetch_push", env=env)
model.learn(total_timesteps=100_000, reset_num_timesteps=False)
model.save("fetch_push_v2")
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
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: reward={reward:.3f}")
    if terminated or truncated:
        obs, _ = env.reset()
```

### Q2: 内存不足

```python
# 减小 buffer 和 batch
model = SAC(..., buffer_size=100_000, batch_size=64)
```

### Q3: 训练太慢

```python
# 选项 1: 并行环境
env = make_vec_env("FetchPush-v2", n_envs=4)

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
6. **RoboSuite**: ARISE Initiative, [GitHub](https://github.com/ARISE-Initiative/robosuite)
7. **ManiSkill3**: Haosu Lab, RSS 2025. [GitHub](https://github.com/haosulab/ManiSkill)

---

## 附录：命令速查表

```bash
# === 安装 ===
pip install stable-baselines3 gymnasium-robotics

# === 训练 ===
cd examples
python rl_demo.py --mode train --env FetchPush-v2 --timesteps 100000
python rl_demo.py --mode train --env FetchPickAndPlace-v2 --timesteps 100000

# === 测试 ===
python rl_demo.py --mode enjoy --model fetch_push --env FetchPush-v2

# === 评估 ===
python rl_demo.py --mode eval --model fetch_push --env FetchPush-v2 --episodes 100

# === 监控 ===
tensorboard --logdir ./fetch_tensorboard/
```

---

> **本文目标达成**: 你理解了 RL 的核心概念（状态、动作、奖励、SAC、HER），能够在 CPU 上训练 Fetch 机械臂操作策略，并掌握了 RL 框架的横向对比。这就是 RL 的 0→1。

## 毕业验收

完成以下所有项目即算 RL 0→1 毕业：

- [ ] 运行 `rl_demo.py --mode demo --task reach`（理解 Q-Learning 循环）
- [ ] 运行 `rl_demo.py --mode train`（在 FetchPush-v2 上训练 SAC+HER）
- [ ] 训练至少 3 个随机种子（--seed 0, 1, 2）
- [ ] 报告 success rate mean ± std（--mode eval --episodes 100）
- [ ] 保存模型、配置、reward 曲线和 evaluation log
- [ ] 能解释：MDP、SAC、HER、replay buffer、goal-conditioned RL
- [ ] 能解释 observation/achieved_goal/desired_goal 的区别
- [ ] （可选）在 FetchPickAndPlace-v2 上训练
