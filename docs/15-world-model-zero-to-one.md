# World Model 零到一：世界模型如何预测未来

> **逐点图解 / Concept close-ups：**[世界模型与预测 rollout](knowledge-atlas/planning-world-models/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> **目标**: 理解世界模型 (World Model) 的核心概念，掌握 DreamerV3 的架构原理，了解如何用世界模型在"想象"中训练策略，以及四大开源世界模型的对比。

---

## 目录

1. [什么是世界模型？](#1-什么是世界模型)
2. [为什么需要世界模型？](#2-为什么需要世界模型)
3. [DreamerV3 架构解析](#3-dreamerv3-架构解析)
4. [环境准备](#4-环境准备)
5. [第一次运行（10 分钟）](#5-第一次运行10-分钟)
6. [世界模型 Pipeline](#6-世界模型-pipeline)
7. [在想象中训练策略](#7-在想象中训练策略)
8. [世界模型 vs 无模型 RL](#8-世界模型-vs-无模型-rl)
9. [四大开源世界模型对比](#9-四大开源世界模型对比)
10. [在灵巧操作中的应用](#10-在灵巧操作中的应用)
11. [常见问题排查](#11-常见问题排查)
12. [参考文献](#12-参考文献)

---

## 1. 什么是世界模型？

**世界模型 (World Model)** 是一个学习环境动态的神经网络：给定当前状态和动作，预测下一状态和奖励。

```
传统 RL:                         世界模型 RL:
Agent ──action──► Real World      Agent ──action──► World Model (想象)
  ▲                   │              ▲                   │
  └──state, reward───┘              └──predicted state───┘
                                          reward
```

**核心思想**: 与其在真实世界中试错，不如在"梦境"中学习。世界模型学会了预测"如果我这样做，会发生什么"，然后策略在模型的想象中高效训练。

### 1.1 直观类比

| 人类 | 世界模型 |
|------|---------|
| 闭上眼睛想象"如果我把球推出去，它会滚到桌子边缘" | 世界模型预测 `next_state = f(state, action)` |
| 在脑中排练动作，不用真的做 | 策略在想象轨迹上训练，不消耗真实环境交互 |
| 经验越多，想象越准确 | 世界模型从交互数据中学习，预测越来越准 |

---

## 2. 为什么需要世界模型？

### 2.1 样本效率

| 方法 | 需要多少交互？ | 例子 |
|------|-------------|------|
| **无模型 RL** (SAC, PPO) | 百万级 | 500,000 步学会抓取 |
| **世界模型 RL** (DreamerV3) | 十万级 | 50,000 步学会抓取 |
| **人类** | 十次级 | 看几次演示就能模仿 |

在标准控制任务上，基于世界模型的 RL（如 DreamerV3）通常比无模型 RL 需要更少的真实交互步数。具体提升幅度取决于任务复杂度和模型容量，详见 [DreamerV3 论文](https://arxiv.org/abs/2301.04104) 的对比实验。

### 2.2 计划能力

```
世界模型可以在想象中"搜索"未来:
  当前状态 → 尝试动作 A → 预测结果不好
            → 尝试动作 B → 预测结果好！ → 执行 B
```

这就是 **Model Predictive Control (MPC)** 的核心——在模型内部做树搜索。

### 2.3 泛化性

同一个世界模型可以用于多个下游任务：
- 策略学习（RL）
- 运动规划（Planning）
- 探索（Curiosity-driven Exploration）
- 表示学习（Representation Learning）

---

## 3. DreamerV3 架构解析

**DreamerV3**（Nature 2025）是当前最流行的世界模型 RL 算法，由 Danijar Hafner 团队开发。它使用固定超参数，在 Atari、DMControl、Minecraft 等截然不同的领域都取得了 SOTA。

### 3.1 RSSM 世界模型

DreamerV3 的核心是 **RSSM (Recurrent State-Space Model)**：

```
                    ┌──────────────┐
  image ──► Encoder │  CNN/MLP     │──► latent z_t
                    └──────────────┘

                    ┌──────────────────────────────────┐
  z_t, a_t ──► RSSM │  h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})  │──► h_t
                    │  z_t ~ Categorical(MLP(h_t))             │──► z_t
                    │  r_t  = MLP(h_t, z_t)                    │──► reward
                    └──────────────────────────────────┘

                    ┌──────────────┐
  h_t, z_t ──► Decoder │  CNN/MLP     │──► reconstructed image
                    └──────────────┘
```

**三个组件**:

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **Encoder** | 原始图像 | 潜在表征 z | 压缩高维观测 |
| **RSSM** | z, a, h_prev | 下一状态 h, z, 奖励 r | 学习动态 |
| **Decoder** | h, z | 重建图像 | 监督信号 |

### 3.2 潜在状态设计

DreamerV3 的关键创新是**类别潜在变量**：

```python
# 不是高斯分布（连续），而是类别分布（离散）
h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))  # 确定性状态
z_t ~ Categorical(logits=MLP(h_t))             # 随机类别状态
```

**为什么用类别？**
- 更容易建模多模态分布（未来可能有多种可能）
- 避免 posterior collapse（后验坍缩）
- 通过 straight-through estimator 保持梯度可微

### 3.3 训练流程

```
World Model 训练:
  for step in range(world_model_steps):
      batch = sample_replay_buffer()
      z = encoder(batch.image)
      z_pred, reward_pred = rssm(z, batch.action)
      image_recon = decoder(z_pred)
      loss = kl_loss + reward_loss + image_loss
      optimizer.step()

Policy 训练（在想象中）:
  for step in range(imagination_steps):
      z = sample_start_state()
      for horizon in range(imagination_horizon):
          action = actor(z)
          z, reward = rssm(z, action)  # 在想象中前进
          critic_loss + actor_loss  # 更新策略
```

### 3.4 关键超参数

| 参数 | 值 | 作用 |
|------|-----|------|
| 潜在维度 | 32×32 (1024) | 类别 × 类别数 |
| 确定性状态维度 | 512 | GRU 隐藏状态 |
| 想象视野 | 15 | 在想象中前瞻多少步 |
| 世界模型训练频率 | 每步 | 与环境交互同步更新 |

---

## 4. 环境准备

### 4.1 安装 DreamerV3

```bash
git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3
pip install -r requirements.txt
```

### 4.2 硬件要求

| 需求 | 最低 | 推荐 |
|------|------|------|
| GPU | 4 GB 显存 | 8 GB+ |
| RAM | 16 GB | 32 GB+ |
| 训练时间 | 几小时 | 1-2 天 |

### 4.3 使用项目脚本

```bash
cd examples
python world_model_demo.py --mode concept   # 理解概念（numpy 模拟，无需 GPU）
python world_model_demo.py --mode dreamer-guide --task dmc_cartpole_swingup --steps 50000  # DreamerV3 安装与运行指南
```

> **注意**: `--mode dreamer-guide` 不会实际运行 DreamerV3 训练，而是输出详细的安装步骤、环境配置说明和官方训练命令，供你参考后手动执行。

---

## 5. 第一次运行（10 分钟）

### 5.1 概念演示（无需 GPU）

```bash
cd examples
python world_model_demo.py --mode concept
```

输出展示：
- 用一个简化的 2D 环境演示世界模型如何学习
- 编码 → 预测 → 想象 → 规划
- 可视化训练损失和想象轨迹

### 5.2 DreamerV3 安装与运行指南

```bash
cd examples
python world_model_demo.py --mode dreamer-guide --task dmc_cheetah_run --steps 100000
```

此命令会打印 DreamerV3 的完整安装与运行指引（包括 `git clone`、依赖安装、硬件要求和官方训练命令）。由于 DreamerV3 需要 GPU 和较长训练时间，本脚本不直接执行训练，而是输出步骤供你按需手动运行。

---

## 6. 世界模型 Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    World Model RL Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: 数据收集                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │  Random  │───│  Policy  │───│  Mixed   │  ──► Replay Buffer     │
│  │  Explore │    │  Explore │    │  Explore │                       │
│  └──────────┘    └──────────┘    └──────────┘                       │
│                                                                      │
│  Phase 2: 世界模型训练                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │ Encoder  │───│   RSSM   │───│ Decoder  │                       │
│  │ (image→z)│    │(z,a→z',r)│    │(z→image) │                       │
│  └──────────┘    └──────────┘    └──────────┘                       │
│                                                                      │
│  Phase 3: 想象训练                                                   │
│  ┌──────────┐    ┌──────────┐                                       │
│  │  Actor   │───│  Critic  │  在 RSSM 想象的轨迹上训练              │
│  │ (z→a)    │    │ (z→V)    │  不需要真实环境交互！                 │
│  └──────────┘    └──────────┘                                       │
│                                                                      │
│  Repeat: 收集新数据 → 改进世界模型 → 改进策略 → 收集更新数据         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.1 关键设计决策

| 决策 | DreamerV3 的做法 | 原因 |
|------|-----------------|------|
| 预测什么？ | 下一潜在状态 + 奖励 | 不预测像素，在紧凑空间学习 |
| 如何表示状态？ | 离散类别 × 32 | 多模态、稳定、可解释 |
| 如何训练策略？ | 在想象中训练 Actor-Critic | 样本效率高，可并行 |
| 如何探索？ | 策略熵 + 随机动作 | 平衡探索与利用 |

---

## 7. 在想象中训练策略

### 7.1 想象轨迹生成

```python
def imagine_trajectory(world_model, actor, horizon=15):
    """在世界模型中生成想象轨迹。"""
    z = sample_start_state()  # 从 replay buffer 随机采样起始状态
    trajectory = []

    for t in range(horizon):
        # 策略选择动作
        action = actor(z)

        # 世界模型预测下一状态
        z_next = world_model.rssm(z, action)

        # 记录
        trajectory.append((z, action, z_next))

        z = z_next

    return trajectory
```

### 7.2 Actor-Critic 训练

```python
# 在想象轨迹上训练策略
for trajectory in imagine_trajectories(world_model, actor, N=16):
    # Critic: 学习价值函数
    values = critic(trajectory.states)
    targets = trajectory.rewards + gamma * critic(trajectory.next_states)
    critic_loss = mse(values, targets)

    # Actor: 最大化 Q 值
    actions = actor(trajectory.states)
    q_values = critic(trajectory.states, actions)
    actor_loss = -mean(q_values) + alpha * entropy(actions)

    # 只在想象中更新，不消耗真实环境
    critic.optimizer.step()
    actor.optimizer.step()
```

---

## 8. 世界模型 vs 无模型 RL

| 维度 | 无模型 (SAC, PPO) | 世界模型 (DreamerV3) |
|------|-------------------|---------------------|
| **样本效率** | 低（百万级） | 高（十万级） |
| **计算开销** | 低（单 GPU） | 中（需要训练世界模型） |
| **泛化性** | 任务特定 | 跨任务 |
| **计划能力** | 无 | 可在想象中规划 |
| **实现复杂度** | 低 | 中 |
| **灵巧操作适用** | 高（SAC+HER） | 中（高维动作空间有挑战） |
| **经典论文** | SAC (ICML 2018) | DreamerV3 (Nature 2025) |

---

## 9. 四大开源世界模型对比

| 维度 | DreamerV3 | TD-MPC2 | LeWorldModel | Nano World Model |
|------|-----------|---------|-------------|-----------------|
| **Stars** | ~3,500 | ~723 | 新项目 | ~676 |
| **框架** | JAX | PyTorch | PyTorch | PyTorch |
| **参数量** | ~20M | ~5M-317M | **15M** | ~50M |
| **动作空间** | 离散+连续 | 连续 | 连续 | 连续 |
| **单 GPU 训练** | 是 | 是 | **是（几小时）** | 是 |
| **灵巧操作** | 部分 | **高度适用** | 部分 | 部分 |
| **推理机制** | 想象中训练 | **MPC 规划** | MPC 规划 | 视频预测 |
| **安装难度** | 中 | 中高 | **低** | **低** |
| **许可证** | MIT | MIT | MIT | MIT |

### 选型指南

| 场景 | 推荐 | 理由 |
|------|------|------|
| **学习/教学** | LeWorldModel | 15M 参数，几小时训练，PyTorch |
| **灵巧操作** | TD-MPC2 | 原生支持 ManiSkill，MPC 推理 |
| **通用研究** | DreamerV3 | Nature 论文，跨域 SOTA |
| **视频预测** | Nano World Model | Diffusion Forcing 框架 |

---

## 10. 在灵巧操作中的应用

### 10.1 当前进展

| 应用 | 模型 | 成果 |
|------|------|------|
| 空中操控 | DreamerV3 | 浮动基座物体操控 |
| 移动操控 | DreamerV3 | 移动机器人操控 |
| 人形机器人 | TD-MPC2 | 61-DoF HumanoidBench |
| 桌面操控 | LeWorldModel | PushT, Cube 任务 |

### 10.2 挑战

| 挑战 | 说明 |
|------|------|
| **高维动作空间** | 24-DoF 灵巧手使想象推演误差累积严重 |
| **长时程预测** | 精细操作需要数十步预测，误差指数增长 |
| **接触建模** | 手指-物体接触的物理难以在潜在空间建模 |
| **多模态** | 同一状态可能有多种有效操作方式 |

### 10.3 未来方向

1. **层次化世界模型**: 高层规划 + 低层执行
2. **结合 Retargeting**: 遥操作数据 → 世界模型预训练
3. **多模态输入**: 图像 + 触觉 + 语言
4. **Sim-to-Real**: 仿真中训练世界模型，迁移到真实机器人

---

## 11. 常见问题排查

### Q1: DreamerV3 训练不收敛

```bash
# 使用 debug 配置快速验证
python dreamerv3/train.py --configs debug --task dmc_cheetah_run

# 检查世界模型损失
# KL loss 应该稳定在 1-3 nat
# 图像重建应该能看到模糊的重建
```

### Q2: 显存不够

```python
# 减小模型
--rssm.units 256  # 默认 512
--batch_size 8    # 默认 16
```

### Q3: 训练太慢

```bash
# 使用更简单的环境
--task dmc_cartpole_swingup  # 比 cheetah 简单

# 减少想象步数
--imagination_horizon 8  # 默认 15
```

---

## 12. 参考文献

1. **DreamerV3**: Hafner et al., "Mastering Diverse Domains through World Models", Nature 2025. [GitHub](https://github.com/danijar/dreamerv3)
2. **TD-MPC2**: Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control", ICLR 2024. [GitHub](https://github.com/nicklashansen/tdmpc2)
3. **LeWorldModel**: Maes et al., "LeWorldModel: Stable End-to-End JEPA from Pixels", arXiv 2026. [GitHub](https://github.com/lucas-maes/le-wm)
4. **Nano World Models**: Simchowitz et al., "Nano World Models: A Minimalist Implementation of Future Video Prediction", 2026. [GitHub](https://github.com/SimchowitzLabPublic/nano-world-model)
5. **RSSM**: Hafner et al., "Learning Latent Dynamics for Planning from Pixels", ICML 2019.
6. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.

---

## 附录：命令速查表

```bash
# === 概念演示 ===
cd examples
python world_model_demo.py --mode concept
python world_model_demo.py --mode concept --visualize

# === DreamerV3 安装与运行指南 ===
python world_model_demo.py --mode dreamer-guide --task dmc_cartpole_swingup --steps 50000

# === 安装 DreamerV3 ===
git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3 && pip install -r requirements.txt
python dreamerv3/train.py --configs defaults dmc_vision --task dmc_cartpole_swingup

# === 监控训练 ===
tensorboard --logdir ~/dreamerv3/logdir
```

---

> **本文目标达成**: 你理解了世界模型的核心思想（编码→预测→想象→规划），掌握了 DreamerV3 的 RSSM 架构，了解了四大开源世界模型的对比，以及世界模型在灵巧操作中的挑战和未来方向。这就是世界模型的 0→1。

## 毕业验收

完成以下所有项目即算 World Model 0→1 毕业：

- [ ] 运行 `world_model_demo.py`（理解 one-step dynamics prediction）
- [ ] 运行 `world_model_vla_pipeline.py`（理解四种 WM+Policy 融合方式）
- [ ] 运行 `dreamer_rssm.py`（理解 RSSM prior/posterior + burn-in）
- [ ] 能解释：prior vs posterior、deterministic vs stochastic、burn-in 的作用
- [ ] 能解释四种融合方式（数据增强、评估器、规划器、Latent BC）的适用场景
- [ ] 能修改 horizon 或噪声参数，观察预测误差变化
- [ ] 报告 1/5/10/20 步 prior 预测误差
