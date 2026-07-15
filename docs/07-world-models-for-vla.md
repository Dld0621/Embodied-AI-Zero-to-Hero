# 07 - 世界模型（VLA 视角）

> 面向 VLA 学习者的世界模型（World Model）指南。重点不是全面梳理世界模型领域的所有工作，而是回答一个核心问题：**世界模型如何服务于机器人控制和 VLA 系统？**

---

## 1. 什么是世界模型（机器人视角）

### 1.1 一句话定义

**世界模型学习的是"世界的运行规律"**：给定当前观测和执行的动作，预测下一个观测（或状态）。

$$p(o_{t+1} \mid o_t, a_t)$$

- $o_t$：当前时刻的观测（摄像头图像、深度、本体感知等）
- $a_t$：当前时刻执行的动作（VLA 的输出）
- $o_{t+1}$：下一个时刻的观测（世界变化后的样子）

### 1.2 世界模型 vs VLA

| 维度 | VLA（Vision-Language-Action） | 世界模型（World Model） |
|------|---------------------------|----------------------|
| **输入** | 观测 + 语言指令 | 观测 + 动作 |
| **输出** | 动作（"我该怎么做"） | 下一时刻的观测（"世界会怎么变"） |
| **本质** | 策略（Policy） | 转移模型（Transition Model / Dynamics Model） |
| **目标** | 让机器人执行正确的动作 | 理解和预测物理世界的因果关系 |
| **训练信号** | 专家动作（BC）或 reward（RL） | 预测误差（自监督） |

> **直觉类比**：
> - VLA 就像一个**驾驶员**——看到路况，输出方向盘和油门操作
> - 世界模型就像一个**副驾驶**——看到你打了方向盘，预测车会怎么动
> - 两者互补：驾驶员负责"做"，副驾驶负责"预判"

### 1.3 为什么 VLA 工程师需要关心世界模型？

1. **数据 augment**：真实世界数据采集昂贵，世界模型可以生成仿真数据
2. **安全验证**：在让 VLA 真实执行前，用世界模型"预演"一下是否安全
3. **提升 sample efficiency**：世界模型让 RL 在"想象空间"中训练，减少真实交互
4. **长程规划**：VLA 擅长短程反应，世界模型擅长长程推演
5. **2025 趋势**：World Action Model 等新范式正在融合两者

---

## 2. 世界模型的分类框架

理解世界模型的关键是搞清楚它在预测什么、怎么用。以下是一个实用的分类框架：

### 2.1 预测什么？

| 预测目标 | 描述 | 代表方法 | 难度 |
|---------|------|---------|------|
| **像素** | 直接预测下一帧图像 | Sora, UniSim, Genie | 最难，信息量最大 |
| **Latent（潜空间）** | 在压缩表征空间预测 | Dreamer V3, PlaNet | 较易，信息压缩 |
| **结构化状态** | 预测物体位姿、速度等 | MeshSDF, KPL | 中等，需要结构化标注 |
| **动作 / Reward / Value** | 不预测观测，只预测奖励和值函数 | MuZero, IRIS | 最易，信息最压缩 |

### 2.2 是否 Action-Conditioned？

| 类型 | 描述 | 用途 |
|------|------|------|
| **Action-Conditioned** | 输入包含动作 $a_t$，预测动作执行后的结果 | RL 训练、策略评估、安全验证 |
| **Unconditioned** | 只输入观测，预测下一帧（无动作信息） | 视频预测、数据生成、理解动力学 |

> 对 VLA 场景而言，**action-conditioned** 更有实用价值，因为我们需要知道"执行了 VLA 的动作后会发生什么"。

### 2.3 服务目标

| 目标 | 描述 | 场景 |
|------|------|------|
| **生成 / 仿真** | 生成逼真的视频或交互环境 | 数据 augment、sim-to-real |
| **评估策略** | 用世界模型替代真实环境评估策略好坏 | 安全验证、离线评估 |
| **训练策略** | 在世界模型里做"想象训练"（imagination） | 高采样效率 RL（Dreamer） |
| **理解物理** | 学习物理规律，辅助下游任务 | 抓取稳定性判断、工具使用推理 |

---

## 3. 五条主流技术路线

### 3.1 路线一：Latent Dynamics（潜空间动力学）

**代表工作**：Dreamer V3 (Hafner et al., 2023)

**核心思想**：在高维像素空间预测太难了，不如先学一个 latent space，在 latent space 里预测动力学。

Dreamer V3 的关键架构是 **RSSM（Recurrent State-Space Model）**：

```
观测 o_t          动作 a_t
   │                 │
   ↓                 ↓
┌─────────┐     ┌─────────┐
│ Encoder  │     │ Action  │
│ CNN      │     │ Input   │
└────┬────┘     └────┬────┘
     │               │
     ↓               ↓
┌─────────────────────────┐
│   RSSM Transition       │
│   ┌───────────────────┐ │
│   │ deterministic: h_t │ ← GRU 隐状态（确定性）
│   │ = f(h_{t-1}, z_{t-1}, a_t) │
│   │                   │ │
│   │ stochastic: z_t    │ ← 潜状态（随机性） │
│   │ ~ p(z_t | h_t, a_t)    │
│   └───────────────────┘ │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────┐
│   RSSM 产出的内容  │
│   ├─ 预测 o_{t+1} │ ← 重建观测（验证学习效果）
│   ├─ 预测 r_t      │ ← 预测 reward（用于 RL）
│   └─ 预测 continue │ ← 预测 episode 是否结束   │
└─────────────────┘
```

**为什么有效？**
- latent space 过滤了无关细节（光照、纹理），保留了关键动力学
- 可以在 latent space 里做"想象训练"，极大提升 RL 的 sample efficiency
- Dreamer V3 是**首个在 55 个 diverse 任务上统一成功的 model-based RL**

**适用场景**：高采样效率 RL、需要大量训练但环境交互成本高的任务

### 3.2 路线二：隐式动力学（Implicit Dynamics）

**代表工作**：MuZero (Schrittwieser et al., 2020)

**核心思想**：不预测未来的观测或状态，只预测 **reward、value、policy**。用这三者隐式地学到环境的动力学。

```
观测 o_t + 动作 a_t
       │
       ↓
┌──────────────────────────┐
│   Representation Network │  → 隐状态 s_t
│   (编码当前观测)          │
└───────────┬──────────────┘
            │
            ↓
┌──────────────────────────┐
│   Dynamics Network       │  → 下一时刻隐状态 s_{t+1}
│   (只输入动作 a_t + s_t)  │    不预测像素，只预测 reward
└───────────┬──────────────┘
            │
            ↓
┌──────────────────────────┐
│   Prediction Network     │  → value(s_t) + policy(s_t)
│   (预测价值和策略)         │
└──────────────────────────┘
```

**优势**：不需要重建像素，计算效率高
**局限**：无法生成可视化结果，不适合需要"看见"未来场景的任务
**适用场景**：棋类、Atari 等 reward 信号明确的环境

### 3.3 路线三：Transformer / Diffusion World Model

**代表工作**：IRIS (2023)、DIAMOND (2024)

**核心思想**：用大模型架构（Transformer 或 Diffusion）作为世界模型，进行像素级预测。

#### IRIS（Transformer World Model）

- 用 Transformer 做世界模型，接收历史 token 序列
- 在 latent space 做下一步预测
- 关键优势：Transformer 的 scaling 和上下文建模能力

#### DIAMOND（Diffusion World Model）

- 用 **Diffusion Model** 预测下一帧图像
- 核心洞察：下一帧的生成是一个**去噪过程**
- 优势：能生成高保真、多样化的预测结果

| 方法 | 架构 | 预测目标 | 生成质量 | 计算成本 |
|------|------|---------|---------|---------|
| IRIS | Transformer | Latent | 中等 | 中等 |
| DIAMOND | Diffusion | Pixel | 高 | 高 |

**适用场景**：需要像素级预测、生成逼真未来场景的任务

### 3.4 路线四：非生成式世界模型

**代表工作**：V-JEPA 2 (Meta, 2025)

**核心思想**：世界模型不一定需要生成像素！只学习 **预测表征（representation）**，不生成可见输出。

```
观测 o_t          动作 a_t
   │                 │
   ↓                 ↓
┌─────────┐     ┌─────────┐
│ Vision  │     │ Action  │
│ Encoder │     │ Encoder │
└────┬────┘     └────┬────┘
     │               │
     ↓               ↓
┌─────────────────────────┐
│   Latent Prediction     │
│   预测: z_{t+1}         │  ← 只在 latent space 预测
│   (不重建像素!)          │     不需要解码回图像
└─────────────────────────┘
```

**为什么不做生成？**
- 生成像素浪费算力：很多像素细节对决策无用
- 专注于学习物理理解，而非渲染能力
- 类比 V-JEPA 1 的思想："Feature Prediction is All You Need"

**适用场景**：需要物理理解但不需可视化的场景（抓取稳定性、碰撞预测）

### 3.5 路线五：Foundation World Model（基础世界模型）

**代表工作**：Genie (Google DeepMind, 2024)、UniSim (2024)

**核心思想**：训练一个**通用的**世界模型，可以处理多种环境、多种任务——类似 GPT 之于语言。

| 工作 | 训练数据 | 能力 | 特色 |
|------|---------|------|------|
| **Genie** | 无标注视频 | 从单张图片生成交互式环境 | 可玩的游戏环境生成 |
| **UniSim** | 多源视频（含动作标注） | 统一可控视频预测 | 跨场景泛化 |

```
┌─────────────────────────────────────────┐
│         Foundation World Model          │
│                                         │
│  输入: 图像 + 动作 (可选) + 语言 (可选)   │
│  输出: 未来帧序列 (可控)                 │
│                                         │
│  能力:                                   │
│  ├─ 从单帧生成可交互环境                   │
│  ├─ 跨场景、跨任务泛化                     │
│  └─ 支持语言条件控制                      │
└─────────────────────────────────────────┘
```

**适用场景**：数据生成、仿真环境创建、通用视频预测

### 五条路线对比总结

| 路线 | 代表工作 | 核心思想 | 预测目标 | 适用场景 |
|------|---------|---------|---------|---------|
| Latent Dynamics | Dreamer V3 | 在潜空间学环境动力学 | Latent + reward | 高采样效率 RL |
| 隐式动力学 | MuZero | 只预测 reward/value/policy | reward + value + policy | 棋类 / Atari |
| Transformer / Diffusion | IRIS, DIAMOND | 大模型做世界模型 | Pixel / Latent | 像素级预测 |
| 非生成式 | V-JEPA 2 | 预测表征不生成像素 | Latent only | 物理理解 |
| Foundation WM | Genie, UniSim | 可交互基础世界模型 | Pixel (通用) | 数据生成 / 仿真 |

---

## 4. 世界模型 + VLA 的融合方式

世界模型和 VLA 不是竞争关系，而是互补的。以下是四种主要融合方式：

### 4.1 世界模型作为数据生成器

```
┌──────────────────┐
│  世界模型 (WM)    │ ──→ 生成仿真数据 (o, a, o') 轨迹
└──────────────────┘              │
                                 ↓
                    ┌──────────────────┐
                    │  BC 训练 VLA       │
                    │  (用 WM 数据)      │
                    └──────────────────┘
```

**价值**：
- 真实机器人数据采集昂贵（人工操作、设备损耗）
- WM 可以无成本地生成大量训练数据
- 特别适合数据稀缺场景

**挑战**：
- WM 生成的数据可能不够逼真（domain gap）
- 需要确保 WM 学到的动力学与真实世界一致

**代表工作**：RoboGen, MimicGen 等

### 4.2 世界模型作为评估器

```
VLA 输出候选动作 a_1, a_2, ..., a_n
         │
         ↓
┌──────────────────────────────────┐
│  世界模型评估：                    │
│  WM(o_t, a_1) → 预测 o_{t+1}^1   │  评估安全性和效果
│  WM(o_t, a_2) → 预测 o_{t+1}^2   │
│  ...                              │
└──────────────────────────────────┘
         │
         ↓
  选择最优动作执行
```

**价值**：
- 在真实执行前"预演"，避免危险操作
- 可以评估多个候选动作，选最优的
- 类似 LLM 中的"自我评估"（self-evaluation）

**适用场景**：安全关键任务（手术机器人、自动驾驶）

### 4.3 世界模型作为规划器（Model-Based Planning）

```
┌─────────────────────────────────────────────────────────┐
│                    规划流程                                │
│                                                          │
│  当前状态 o_0                                            │
│       │                                                  │
│       ↓                                                  │
│  ┌─ WM 展开 N 步 ──────────────────────────────────┐     │
│  │ o_0 → a_0 → o_1 → a_1 → ... → o_N              │     │
│  └──────────────────────────────────────────────┬┘     │
│                                                 │       │
│  搜索 / 优化 → 选出最优 action sequence           │       │
│                                                 │       │
│       ↓                                          │       │
│  VLA 执行第一步动作 a_0                             │
│  然后重新观测 → 重新规划 → 执行 a_1 → ...             │
└─────────────────────────────────────────────────────────┘
```

**优势**：
- WM 帮 VLA "看到未来"，避免短视决策
- 规划（planning）和执行（execution）解耦
- 类似 AlphaGo 的 MCTS + 神经网络评估

**挑战**：
- 规划搜索空间随时间步指数增长（维度灾难）
- WM 的预测误差在多步展开中累积
- 实时性要求高（机器人控制频率通常 > 3Hz）

### 4.4 World Action Model (WAM)

**核心思想**：同时预测未来状态 **和** 动作——世界模型不再是纯粹的"观察者"，而是兼具策略功能。

```
普通世界模型:  p(o_{t+1} | o_t, a_t)         → 只预测状态
World Action Model: p(o_{t+1}, a_{t+1} | o_t, a_t)  → 同时预测状态和动作
```

**代表工作**：DreamZero (2025)

**价值**：
- 将世界模型和策略统一到一个模型中
- 天然支持长程任务（多步预测 + 多步动作生成）
- 减少模型数量，简化系统架构

**局限**：
- 训练难度大（同时学动力学和策略）
- 动作预测的质量受限于世界模型的能力

### 四种融合方式对比

| 融合方式 | 世界模型的角色 | VLA 的角色 | 优势 | 挑战 |
|---------|-------------|-----------|------|------|
| **数据生成器** | 生成训练数据 | 接收数据训练 | 解决数据稀缺 | 生成数据的逼真度 |
| **评估器** | 评估候选动作安全 | 生成候选动作 | 安全验证 | 评估准确性 |
| **规划器** | 多步前瞻搜索 | 执行选定动作 | 长程规划 | 计算成本高 |
| **WAM** | 统一预测状态和动作 | 融合进世界模型 | 架构简洁 | 训练困难 |

---

## 5. 关键论文导读（6 篇）

### 5.1 Dreamer V3 (2023)

- **全称**：Mastering Diverse Domains through World Models
- **作者**：Danijar Hafner et al.
- **核心贡献**：首个在 55 个不同连续/离散任务上统一成功的 model-based RL
- **关键技术**：RSSM（Recurrent State-Space Model）—— 用确定性 GRU + 随机 latent state 建模环境
- **VLA 关联**：RSSM 的设计思想（分离确定性和随机性）可以启发 VLA 的状态表征设计
- **阅读建议**：重点理解 RSSM 的结构，这是世界模型的基础架构

### 5.2 IRIS (2023)

- **全称**：World Model as a Policy Prior for Reinforcement Learning
- **作者**：Janner et al.
- **核心贡献**：用 Transformer 做世界模型，将其作为 policy prior 辅助 RL
- **关键技术**：将世界模型的输出作为 RL 策略的先验，减少探索需求
- **VLA 关联**：展示了 Transformer 架构在世界模型中的有效性（类似 VLA 用 Transformer 做策略）
- **阅读建议**：关注 Transformer 如何建模时序 dynamics

### 5.3 DIAMOND (2024)

- **全称**：DIffusion As a MOdel of eNvironment Dynamics
- **作者**：Micheal et al.
- **核心贡献**：首次用 Diffusion Model 做世界模型，实现了 Atari 游戏的像素级预测
- **关键技术**：下一帧预测 = 去噪过程；用 classifier-free guidance 控制预测多样性
- **VLA 关联**：Diffusion 架构在策略（Diffusion Policy）和世界模型中都有效
- **阅读建议**：理解为什么 diffusion 天然适合做 "下一步预测"

### 5.4 V-JEPA 2 (2025)

- **全称**：Video-JEPA 2（Meta）
- **作者**：Meta FAIR
- **核心贡献**：提出非生成式世界模型范式——只学表征预测，不生成像素
- **关键技术**：Invariance-based prediction；在 latent space 做条件预测
- **VLA 关联**：证明了"理解世界"不一定需要"画出世界"，对 VLA 的感知模块设计有启发
- **阅读建议**：重点理解 non-generative 范式的动机和优势

### 5.5 UniSim (2024)

- **全称**：Learning Unified Interactive Real-World Simulators
- **作者**：Google DeepMind
- **核心贡献**：统一可控视频预测——一个模型处理多种场景的动作条件生成
- **关键技术**：利用多源数据（含标注和未标注）训练；跨场景泛化
- **VLA 关联**：展示了大规模预训练世界模型的可能性——类似 foundation model 思路
- **阅读建议**：关注如何利用不同质量/类型的数据做统一训练

### 5.6 DreamZero (2025)

- **全称**：World Action Model
- **核心贡献**：提出 World Action Model（WAM）范式——世界模型同时预测状态和动作
- **关键技术**：将 policy 和 dynamics 统一到同一个模型中
- **VLA 关联**：直接展示了世界模型和 VLA 融合的新方向
- **阅读建议**：理解 WAM 与传统 world model + policy 分离架构的区别

### 论文阅读优先级

| 优先级 | 论文 | 理由 |
|--------|------|------|
| **P0** | Dreamer V3 | 理解世界模型的基础架构 RSSM |
| **P1** | DIAMOND | 理解 Diffusion 作为世界模型 |
| **P1** | V-JEPA 2 | 理解非生成式世界模型的思路 |
| **P2** | UniSim | 理解基础世界模型的思路 |
| **P2** | IRIS | 理解 Transformer 世界模型 |
| **P3** | DreamZero | 理解 WAM 范式（最新进展） |

---

## 6. 从 VLA 视角怎么入门世界模型

### 第一步：理解 Dreamer V3 的 RSSM

这是世界模型的基石。重点理解：

1. **确定性部分**（GRU）：捕捉世界的可预测规律
2. **随机性部分**（Latent z）：捕捉世界的不确定性
3. **为什么这种分离很重要**：机器人环境中，运动学是确定性的，但碰撞和物体交互有随机性

**实践建议**：
- 读 Dreamer V3 论文的 Section 3（World Model Architecture）
- 关注 Figure 1 的 RSSM 结构图
- 理解 `reconstruction loss` + `reward loss` + `continue loss` 的训练目标

### 第二步：跑 simple_world_model.py

在 VLA 项目 examples 目录下有简单的世界模型实现：

```bash
# VLA-Zero-to-Hero/examples/
python simple_world_model.py --env PandaReach --epochs 50
```

这个脚本实现了一个最简版的世界模型：
- 输入：当前观测 + 动作
- 输出：预测的下一个观测
- 训练目标：最小化预测观测和真实观测的差异（MSE）

**学习要点**：
- 理解"学习动力学"和"学习策略"的区别
- 观察 prediction error 随训练的变化
- 尝试多步展开预测，看看误差如何累积

### 第三步：读 IRIS 或 DIAMOND 源码

- **IRIS**：[官方实现](https://github.com/janner/iris) — Transformer-based，代码结构清晰
- **DIAMOND**：[官方实现](https://github.com/eliteraspberries/diamond) — Diffusion-based，可以对比与 Diffusion Policy 的架构异同

**阅读建议**：
- 先看 World Model 部分（不看 RL 部分）
- 理解 tokenization / latent encoding 策略
- 对比 VLA 的 encoder 架构

### 第四步：尝试用世界模型生成仿真数据训练 VLA

这是最终的实践目标：

```
1. 训练一个简单世界模型（用人类演示数据）
       ↓
2. 用世界模型生成虚拟轨迹 (o_0, a_0, o_1, a_1, ...)
       ↓
3. 用虚拟轨迹做 BC 训练 VLA
       ↓
4. 评估：虚拟数据训练的 VLA vs 真实数据训练的 VLA
```

**预期收获**：
- 量化世界模型质量对 VLA 性能的影响
- 理解 domain gap 的问题和解决方向
- 为后续研究打下基础

---

## 交叉引用

- **[01-what-is-vla.md](./01-what-is-vla.md)** — VLA 的定义、架构和训练范式
- **[06-rl-fundamentals-for-vla.md](./06-rl-fundamentals-for-vla.md)** — RL 基础，世界模型的核心应用场景是 model-based RL
- **[02-key-papers.md](./02-key-papers.md)** — VLA 关键论文中的世界模型相关内容
- **[03-learning-path.md](./03-learning-path.md)** — 推荐的学习路线中包含世界模型部分
- **[04-glossary.md](./04-glossary.md)** — 专业术语速查（RSSM、latent dynamics 等）
- **[05-interview-prep.md](./05-interview-prep.md)** — 世界模型相关面试问题

---

## 推荐阅读

| 资源 | 类型 | 难度 | 说明 |
|------|------|------|------|
| Dreamer V3 论文 | 论文 | 中等 | 世界模型入门首选 |
| Danijar Hafner 的博客 | 博客 | 入门 | Dreamer 系列的第一作者，写作清晰 |
| DIAMOND 论文 | 论文 | 中等 | 理解 Diffusion World Model |
| V-JEPA 2 论文 | 论文 | 中等 | 理解非生成式世界模型 |
| Model-Based RL 综述 (Moerland et al., 2023) | 综述 | 入门 | 系统梳理世界模型方法 |
| Spinning Up — MBRL 章节 | 教程 | 入门 | OpenAI 的 model-based RL 教程 |
