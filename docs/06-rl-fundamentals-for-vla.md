# 06 - 强化学习基础（VLA 视角）

> 面向 VLA 学习者的强化学习速成指南。不需要你从零啃 Sutton & Barto，而是从 **VLA 工程师的实际需求** 出发，聚焦 RL 与 VLA 的交叉点。

---

## 1. 为什么 VLA 工程师需要懂 RL

### 1.1 Behavior Cloning 的天花板

VLA 的主流训练范式是 **Behavior Cloning（行为克隆）**——本质上就是监督学习：给模型看专家示范（状态-动作对），让它学会模仿。

但 BC 有几个根本性问题：

| 问题 | 描述 | 后果 |
|------|------|------|
| **复合误差（Compounding Error）** | 单步预测误差随时间步累积 | 长任务中轨迹越来越偏离 |
| **分布外泛化（OOD Generalization）** | 模型只见过训练分布内的状态 | 遇到新情况就"懵了" |
| **多模态行为** | 同一状态可能有多种合理动作 | BC 会取平均，导致次优行为 |
| **无反馈信号** | BC 只模仿，不知道自己做对做错 | 无法主动纠正错误 |

### 1.2 RL Fine-tuning 是突破天花板的关键

在 BC 预训练之后，用 **RL fine-tuning** 可以显著提升 VLA 性能。核心逻辑：

- **BC 学会了"大概怎么做"**（模仿人类示范）
- **RL 学会了"怎么做更好"**（通过 reward 信号优化）
- 两者结合，既保留了 BC 的数据效率，又获得了 RL 的优化能力

这和 NLP 领域的 **RLHF（Reinforcement Learning from Human Feedback）** 思路完全一致：先 SFT（对应 BC），再 RLHF（对应 RL fine-tuning）。

### 1.3 Reward Shaping 帮助设计更好的评估指标

理解 RL 中的 reward function 和 value function，能帮助你：

- **设计更合理的 VLA 评估指标**（不只是成功率，还有效率、安全性）
- **理解 VLA 在线部署时的优化目标**（多目标 trade-off）
- **调试 VLA 策略的行为偏差**（哪里 reward 设计有问题）

---

## 2. RL 核心概念速览

### 2.1 MDP（马尔可夫决策过程）

RL 的数学框架是 MDP，定义为元组 $(S, A, T, R, \gamma)$：

| 元素 | VLA 中的对应 | 说明 |
|------|-------------|------|
| **S（状态 State）** | 机器人观测（RGB 图像 + 本体感知） | VLA 的输入 |
| **A（动作 Action）** | VLA 的输出（末端位姿增量 / 关节角度 / 语言指令） | VLA 的输出 |
| **T（转移 Transition）** | 环境物理规律 | 执行动作后世界如何变化 |
| **R（奖励 Reward）** | 任务完成度（抓取成功 +1，碰撞 -0.1） | 告诉模型"做得好不好" |
| **$\gamma$（折扣因子）** | 0.95 ~ 0.99 | 衡量未来 reward 的重要程度 |

> **关键洞察**：VLA 的训练数据（人类演示轨迹）可以看作 MDP 中的 expert policy 采样。BC 就是从这些采样中学习一个近似 expert 的策略。

### 2.2 Value Function

Value Function 衡量的是"从某个状态（或状态-动作对）出发，当前策略预期能拿到多少 reward"：

- **$V(s)$**：状态价值函数 —— 在状态 $s$ 下，按当前策略行动，预期能获得的总 reward
- **$Q(s, a)$**：动作价值函数 —— 在状态 $s$ 下执行动作 $a$，之后按当前策略行动，预期能获得的总 reward

在 VLA 语境下，**Value Function 帮你回答**：

> "当前这个场景（状态），我的 VLA 模型接下来大概能得多少分？"

#### Bellman 方程

$$V(s) = \mathbb{E}_{a \sim \pi}[R(s,a) + \gamma V(s')]$$

这个方程说的是：当前状态的价值 = 立即获得的 reward + 下一个状态的期望价值（打折后）。

### 2.3 Policy Gradient 定理

Policy Gradient 是最直觉的 RL 优化方法——直接对策略参数 $\theta$ 求梯度，让期望 reward 最大化：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau) \right]$$

直觉解释：

1. **$\nabla_\theta \log \pi_\theta(a_t | s_t)$**：增大选中动作的概率梯度方向
2. **$R(\tau)$**：整条轨迹的 return（reward 总和）
3. 如果 return 高 → 增大该动作的概率；return 低 → 减小该动作的概率

> **VLA 视角**：BC 是固定目标（专家动作），Policy Gradient 是用 reward 信号做目标。在 VLA 微调阶段，可以用 reward 替代固定的专家标签。

### 2.4 On-Policy vs Off-Policy

| 类型 | 定义 | 代表算法 | VLA 中的对应 |
|------|------|---------|-------------|
| **On-Policy** | 用当前策略自己采集的数据训练 | PPO, TRPO | VLA 在线部署后收集交互数据 |
| **Off-Policy** | 用其他策略（或人类）采集的数据训练 | SAC, DDPG, DQN | 使用预收集的人类演示数据（BC） |

> **关键区别**：BC 本质上是 off-policy（数据是专家收集的），而 PPO fine-tuning 是 on-policy（需要 VLA 自己去环境里试错）。

---

## 3. VLA 中的 RL 应用

### 3.1 RLVF（RL from VLA Feedback）

**核心思想**：用一个强大的 VLA 模型（或 VLM）作为 "裁判"，为底层策略提供 reward 信号。

```
┌──────────────┐    动作     ┌──────────────┐
│  底层策略     │ ────────→  │  环境         │
│ (小模型)     │ ←────────  │              │
└──────────────┘    观测     └──────┬───────┘
                                   │ 观测 + 动作
                                   ↓
                          ┌──────────────┐
                          │  VLA 裁判     │ ──→ reward 信号
                          │ (大模型)     │
                          └──────────────┘
```

- **类比 NLP**：类似 RLHF 中用 LLM 作为 reward model
- **优势**：不需要手动设计 reward function
- **挑战**：VLA 裁判的 reward 质量决定了训练效果

### 3.2 PPO Fine-tuning

PPO（Proximal Policy Optimization）是目前最主流的 RL fine-tuning 方法，流程如下：

```
Stage 1: BC Pre-training（监督学习）
  人类演示数据 → 训练 VLA 基础模型

Stage 2: PPO Fine-tuning（强化学习）
  ┌─────────────────────────────────────────────┐
  │ 1. VLA 在环境中执行动作，收集 (s, a, r, s')  │
  │ 2. 用 GAE 计算优势函数 A(s, a)              │
  │ 3. 用 clipped objective 更新策略             │
  │ 4. 重复 1-3 直到收敛                        │
  └─────────────────────────────────────────────┘
```

PPO 的关键设计：

| 组件 | 作用 |
|------|------|
| **Clipped Objective** | 防止策略更新步子太大（保持稳定） |
| **GAE（Generalized Advantage Estimation）** | 估计每个动作比平均水平好多少 |
| **Value Network** | 辅助估计 $V(s)$，降低方差 |
| **Entropy Bonus** | 鼓励探索，防止策略过早坍缩 |

### 3.3 Reward Shaping for Manipulation

为操作任务设计 reward 是一门艺术。以下是常见策略：

#### 稀疏 vs 稠密 Reward

| 类型 | 例子 | 优缺点 |
|------|------|--------|
| **稀疏** | 成功 +1，失败 0 | 简单，但信号太少，学习困难 |
| **稠密** | 距离目标 $-d$，接触 $+0.1$ | 信号丰富，但可能引入 reward hacking |

#### 常用 Reward 组件

```python
# 常见的操作任务 reward 设计
reward = 0.0
reward += -distance_to_target          # 距离奖励：越近越好
reward += 1.0 if grasped else 0.0      # 抓取奖励：是否抓到
reward += 1.0 if task_complete else 0.0 # 完成奖励：任务是否完成
reward += -0.1 if collision else 0.0   # 碰撞惩罚：避免碰撞
reward += -0.01 * action_smoothness     # 平滑惩罚：鼓励平滑动作
```

#### Reward Shaping 原则

1. **对齐目标**：reward 应该引导 VLA 完成你真正想要的行为
2. **避免 hacking**：VLA 可能找到"偷懒"的方法获得高 reward
3. **多阶段设计**：先稀疏 reward 确保方向对，再加稠密 reward 加速学习
4. **安全约束**：始终加入碰撞惩罚等安全项

### 3.4 Hierarchical RL（分层强化学习）

高层（LLM）负责**规划**，低层（VLA）负责**执行**：

```
┌─────────────────────────────────────────────────┐
│                 高层策略（LLM）                    │
│  输入：任务指令 + 当前状态                          │
│  输出：子目标序列 [子目标1, 子目标2, 子目标3, ...]   │
│  频率：每 N 步决策一次                              │
└───────────────────────┬─────────────────────────┘
                        │ 子目标
                        ↓
┌─────────────────────────────────────────────────┐
│                 低层策略（VLA）                     │
│  输入：子目标 + 当前观测                            │
│  输出：末端执行器动作 (dx, dy, dz, droll, dgripper) │
│  频率：每控制步（如 3Hz / 10Hz）                    │
└─────────────────────────────────────────────────┘
```

- **优势**：高层处理长程规划，低层专注于短程操作
- **VLA 的角色**：低层执行器，接收 LLM 分解的子任务
- **代表工作**：SayCan, Inner Monologue, Code as Policies

---

## 4. 关键算法对比（VLA 视角）

| 算法 | 在 VLA 中的角色 | 核心思想 | 优点 | 缺点 |
|------|---------------|---------|------|------|
| **BC** (Behavior Cloning) | VLA 主训练范式 | 最小化与专家动作的距离（监督学习） | 简单、数据利用率高、无需在线交互 | 复合误差、OOD 泛化差、多模态问题 |
| **PPO** | VLA 微调 / 对齐 | Clipped policy gradient | 稳定、超参鲁棒、易于实现 | 样本效率低（需要大量在线交互） |
| **SAC** (Soft Actor-Critic) | 连续控制 fine-tuning | 最大熵 RL，平衡探索与利用 | 样本效率高、自动调节探索 | 需要精心设计 reward |
| **DQN** | 离散动作 VLA | Q-learning + 神经网络 + experience replay | 简单有效、off-policy | 只支持离散动作空间 |
| **DAgger** | BC 改进 | 主动查询专家纠正 | 解决分布偏移 | 需要专家在线参与 |
| **Diffusion Policy + RL** | 最新趋势 | 用 diffusion model 参数化策略 + RL 优化 | 能处理多模态动作分布 | 计算量大 |

### 何时用哪种算法？

```
                你有大量专家数据吗？
                   /          \
                 是             否
                /                \
        任务需要泛化吗？         必须用 RL
          /       \              (SAC / PPO)
        是         否
       /            \
   BC + RL fine-tune  纯 BC
   (BC → PPO)        足够用了
```

---

## 5. 实践指南

### 5.1 如何为 VLA 设计 Reward Function

**Step 1**：明确任务成功的判据
- 抓取任务：物体被抓起并移到目标位置
- 打开抽屉任务：抽屉被拉开一定距离

**Step 2**：分解为可计算的组件
- 目标距离（$\ell_2$ 距离或姿态误差）
- 关键里程碑（接触、抓取、放置）
- 安全约束（碰撞检测、关节限位）

**Step 3**：确定 reward 类型
- 任务简单 → 稀疏 reward（成功 / 失败）
- 任务复杂 → 稠密 reward（距离 + 里程碑 + 约束）
- 不确定 → 先稀疏，再逐步加稠密项

**Step 4**：验证 reward 设计
- 用固定策略随机采样，检查 reward 分布
- 确认 reward 没有意外的高分捷径（reward hacking）
- 在模拟器中快速迭代验证

### 5.2 BC + RL 两阶段训练流程

```python
# ============ Stage 1: BC Pre-training ============
# 使用离线的人类演示数据
vl_model = VLAWithTransformerBackbone()
bc_loss = CrossEntropyLoss(vl_model(actions), expert_actions)
optimizer.step()  # 标准监督学习

# ============ Stage 2: RL Fine-tuning (PPO) ============
# 在线与环境交互
for iteration in range(num_iterations):
    # 收集轨迹
    trajectories = collect_trajectories(vl_model, env, num_steps=2048)
    
    # 计算优势函数（GAE）
    advantages = compute_gae(trajectories, value_network)
    
    # PPO 更新（多个 epoch）
    for epoch in range(num_ppo_epochs):
        # Clipped surrogate objective
        ratio = new_policy / old_policy
        clipped = clip(ratio, 1 - eps, 1 + eps) * advantages
        ppo_loss = -min(ratio * advantages, clipped)
        
        # Value loss
        value_loss = mse(value_network(states), returns)
        
        # 总 loss
        total_loss = ppo_loss + c1 * value_loss - c2 * entropy
        optimizer.step()
```

### 5.3 什么时候用 BC，什么时候用 RL

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 有大量高质量演示数据 | 纯 BC | 数据足够，简单高效 |
| BC 性能已到瓶颈 | BC → PPO fine-tune | RL 可以突破 BC 天花板 |
| 任务需要超出演示分布的泛化 | BC + RL | RL 的探索能力处理新情况 |
| 安全关键任务（手术、驾驶） | BC + 保守 RL | 加安全约束的 RL，如 constrained PPO |
| 无演示数据 | 纯 RL（SAC / PPO） | 从零开始探索 |
| 只有稀疏 reward | SAC + curiosity / BC 引导 | 用内在动机或 BC 初始化辅助 |

### 5.4 常见陷阱与建议

| 陷阱 | 说明 | 建议 |
|------|------|------|
| **Reward Hacking** | VLA 找到获得高 reward 但不是你想要的行为 | 多测试，加约束项 |
| **探索不足** | PPO 微调时 VLA 过快收敛到局部最优 | 调大 entropy coefficient |
| **Catastrophic Forgetting** | RL 训练破坏了 BC 学到的能力 | 加 KL 散度约束（参考 DPO） |
| **Sample Efficiency** | 在线 RL 需要大量交互，真实机器人成本高 | 先在模拟器中训练，再 sim-to-real |
| **Reward Scale** | reward 数值范围影响训练稳定性 | 对 reward 做 running normalization |

---

## 交叉引用

- **[01-what-is-vla.md](./01-what-is-vla.md)** — VLA 的基本定义和训练范式
- **[02-key-papers.md](./02-key-papers.md)** — RT-2, Octo 等关键论文中的 RL 成分
- **[07-world-models-for-vla.md](./07-world-models-for-vla.md)** — 世界模型如何替代/辅助 RL 中的环境交互
- **[05-interview-prep.md](./05-interview-prep.md)** — RL 相关面试问题

---

## 推荐阅读

| 资源 | 类型 | 难度 | 说明 |
|------|------|------|------|
| Sutton & Barto 《Reinforcement Learning》 | 教材 | 入门 | RL 圣经，选读前 6 章即可 |
| Spinning Up in Deep RL (OpenAI) | 教程 | 入门 | 代码驱动的 RL 教程 |
| PPO 原始论文 (Schulman et al., 2017) | 论文 | 中等 | 理解 PPO 的设计动机 |
| OpenVLA 论文 (2024) | 论文 | 中等 | VLA + BC 训练的实践参考 |
| RT-2 论文 (2023) | 论文 | 中等 | VLA 模型在机器人上的应用 |
| Low-cost Policy Learning (Chi et al., 2023) | 论文 | 中等 | 在线 RL fine-tuning VLA |
