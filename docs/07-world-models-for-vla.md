# 07 - 世界模型（VLA 视角）

> 面向 VLA 学习者的世界模型（World Model）指南。重点不是全面梳理世界模型领域的所有工作，而是回答一个核心问题：**世界模型如何服务于机器人控制和 VLA 系统？**

> **专项入口：**本页用于建立世界模型基础。需要系统进入 World Action Model 研究时，请学习 [WAM 从零到一](specializations/wam-zero-to-one-cn.md)；英文版见 [WAM Zero to One](specializations/wam-zero-to-one.md)。

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
5. **近期研究方向**：WorldVLA（2025）和 DreamZero（2026）等工作开始探索世界表征与动作生成的联合建模

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

> **注意：Gaussian vs Categorical Latent 的区别**
>
> 本项目中 `examples/dreamer_rssm.py` 使用 **Gaussian latent（连续高斯分布）** 作为教学 baseline，便于理解和调试。但完整 DreamerV3 实际使用 **categorical latent（离散类别分布）**，原因如下：
> - **表达能力更强**：类别分布天然支持多模态（如"物体可能向左滚，也可能向右滚"），而单峰高斯只能表达一种可能性
> - **避免 posterior collapse**：类别分布通过 straight-through estimator 保持梯度可微，同时避免后验坍缩到先验
> - **训练更稳定**：离散潜变量在长程 rollout 中累积误差更小
>
> 在阅读论文或复现实验结果时，请务必注意这一关键差异。`dreamer_rssm.py` 是"RSSM 教育简化版"，不是 DreamerV3 的完整复现。

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

**狭义工作定义**：在一个联合或对齐框架中建模未来世界状态（常见为视频或视频潜变量）与机器人动作，使模型既包含未来演化信号，也能产生动作。WAM 是仍在形成中的研究标签，不是所有论文都采用同一种概率分解或模块结构。

```
动作条件世界模型:  p(o_{t+1} | o_≤t, a_t) + 独立规划器
联合 WAM（教学分解）: p(o_{t+1:t+H}, a_{t:t+H-1} | o_≤t, g)
```

其中 `g` 可以是语言指令或任务条件。这只是帮助理解的分解，不代表所有 WAM 实现都使用完全相同的 Loss、主干或推理过程。

**必须区分的基线**：

- Action-conditioned World Model + MPC 能预测候选动作的后果，但不会仅因此自动成为 WAM。
- WorldVLA 探索图像与动作的自回归联合理解和生成。
- DreamZero 是 2026 年发布的预印本，在视频扩散主干上联合预测未来视频与动作。

**研究价值与代价**：未来监督可能提供比纯动作 Loss 更密集的训练信号，但联合模型也引入视频—动作时间对齐、异构 Loss 权重、生成延迟和“视频合理但动作不可控”等新失败模式。因此必须和直接策略、世界模型 + 规划器做匹配预算比较。

### 四种融合方式对比

| 融合方式 | 世界模型的角色 | VLA 的角色 | 优势 | 挑战 |
|---------|-------------|-----------|------|------|
| **数据生成器** | 生成训练数据 | 接收数据训练 | 解决数据稀缺 | 生成数据的逼真度 |
| **评估器** | 评估候选动作安全 | 生成候选动作 | 安全验证 | 评估准确性 |
| **规划器** | 多步前瞻搜索 | 执行选定动作 | 长程规划 | 计算成本高 |
| **联合 WAM** | 对齐预测未来世界与动作 | 联合模型输出动作 | 可能共享未来表示 | 训练、对齐与实时性负担高 |

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

### 5.6 DreamZero (2026)

- **论文**：[DreamZero: World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- **发布状态**：2026 年 arXiv 预印本；仓库不把预印本描述为顶会录用
- **核心思路**：在预训练视频扩散主干上联合建模未来视频与机器人动作，并把这种联合系统称为 World Action Model
- **阅读重点**：视频与动作如何对齐、动作组件如何利用预测未来、闭环推理如何组织，以及哪些结果只属于该系统配置
- **证据边界**：论文报告的规模、迁移和机器人结果不能外推为整个 WAM 算法族的保证

### 5.7 LaDi-WM (CoRL 2025)

- **全称**：LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation
- **作者**：国防科大 / 北京大学 / 深圳大学
- **arXiv**: [2505.11528](https://arxiv.org/abs/2505.11528)
- **项目页**: [https://guhuangai.github.io/LaDiWM.github.io/](https://guhuangai.github.io/LaDiWM.github.io/)

**核心贡献**：首次将 **Latent Diffusion** 引入机器人世界模型，在隐空间而非像素空间建模动力学，显著提升操作任务的样本效率与跨场景泛化。

**关键技术**：
- **双塔隐空间表示**：DINOv2 提取几何特征 + SigLIP 提取语义特征，构建通用隐空间
- **交互扩散（Interactive Diffusion）**：在扩散过程中让几何与语义表征充分交互，学习二者依赖关系
- **迭代策略优化**：推理时利用世界模型的未来预测多次引导策略输出，逐步降低动作分布熵

**实验亮点**：
- LIBERO-LONG 仅用 **10 条轨迹** 训练即达 **68.7%** 成功率，超越之前所有方法
- 跨场景泛化：在 LIBERO 上训练的世界模型直接引导 CALVIN 策略学习，性能提升 0.61
- 真实场景（叠碗、开抽屉、抓取放入篮子）将模仿学习策略成功率提升 **20%**

**VLA 关联**：
- LaDi-WM 与 VLA 的融合方式对应本文档第 4.3 节"世界模型作为规划器"——用 WM 的未来预测迭代优化策略
- 其隐空间设计（DINOv2 + SigLIP）与 OpenVLA 的视觉编码器选择一致，天然兼容 VLA 感知模块
- 证明了"世界模型泛化能力优于策略模型"，为 VLA 的数据生成与 sim-to-real 提供新思路

**阅读建议**：重点理解为什么隐空间扩散比像素级预测更适合机器人操作，以及交互扩散如何让几何与语义协同建模动力学。

### 5.8 DreamDojo (ICML 2026)

- **全称**：DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos
- **作者**：UT Austin, NVIDIA
- **会议**：ICML 2026
- **arXiv**: [2602.06949](https://arxiv.org/abs/2602.06949)
- **项目页**: [https://dreamdojo-world.github.io/](https://dreamdojo-world.github.io/)
- **代码**: [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo) (Apache-2.0)

**核心贡献**：首个从 **4.4 万小时第一视角人类视频** 预训练的通用机器人世界模型，引入 latent action 将无动作标签的人类视频转化为可学习世界动力学的资源。

**关键技术**：
- **Latent Action Model**：将人类视频动作隐式编码为 latent action，无需显式机器人动作标签即可训练
- **蒸馏加速**：将生成速度提升到约 **10 FPS**，支持实时交互
- **Post-training**：预训练后仅需少量机器人数据微调即可用于策略评估和基于模型的规划

**实验亮点**：
- 在机器人数据 post-training 后，可用于实时交互、policy evaluation 和 model-based planning
- 提供 pretrained / post-trained checkpoints，复现完整度极高

**VLA 关联**：
- 直接对应 VLA 的数据瓶颈问题：如何将海量人类视频（而非昂贵的机器人演示）用于世界模型训练
- 与你当前研究的 **人类数据→机器人控制** 方向高度契合
- latent action 的概念为跨 embodiment 动作表示提供了新思路

**阅读建议**：重点理解 latent action 如何让无动作标签的人类视频也能训练世界模型，以及蒸馏方案如何兼顾生成质量和推理速度。

### 5.9 RISE (RSS 2026)

- **全称**：RISE: Self-Improving Robot Policy with Compositional World Model
- **作者**：OpenDriveLab
- **会议**：RSS 2026
- **arXiv**: [2602.11075](https://arxiv.org/abs/2602.11075)
- **项目页**: [https://opendrivelab.com/RISE/](https://opendrivelab.com/RISE/)
- **代码**: [OpenDriveLab/RISE](https://github.com/OpenDriveLab/RISE)

**核心贡献**：将世界模型拆分为 **可控 dynamics model** 和 **progress/value model**，让机器人策略在 imagination rollout 中进行强化学习，减少真机 RL 的硬件磨损和安全成本。

**关键技术**：
- **组合式世界模型**：Dynamics Model + Progress/Value Model 分离设计
- **Imagination RL**：在想象世界中训练策略，避免真实环境交互
- **PiPER 部署**：支持从想象训练到真实机器人部署的完整 pipeline

**实验亮点**：
- 在动态砖块分类、背包整理、关箱等真实操作任务上取得明显提升
- 提供完整的 offline policy → value model → online RL → real robot 代码链路

**VLA 关联**：
- 对应融合方式 4.2（评估器）和 4.3（规划器）的工程化实现
- 展示了从"高质量跨具身数据 → 世界模型 → 想象 rollout → 策略自提升"的完整闭环
- 与你当前的跨具身适应研究可以自然延伸结合

**阅读建议**：重点理解组合式世界模型的设计动机，以及 imagination rollout 如何替代真实环境的 RL 交互。

### 5.10 PointWorld (CVPR 2026 Highlight)

- **全称**：PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation
- **作者**：NVIDIA / Stanford
- **会议**：CVPR 2026 Highlight
- **arXiv**: [2601.03782](https://arxiv.org/abs/2601.03782)
- **项目页**: [https://point-world.github.io/](https://point-world.github.io/)
- **代码**: [NVlabs/PointWorld](https://github.com/NVlabs/PointWorld)

**核心贡献**：首次提出 **3D Point Flow** 作为跨 embodiment 的统一世界表示，用 3D 点流而非 RGB 视频预测动力学，天然适合不同形态机器人的通用操作。

**关键技术**：
- **3D Point Flow 表示**：统一表示 world state + robot action，不依赖特定机器人的关节空间
- **跨 embodiment 设计**：输入 RGB-D + 动作 → 预测场景未来 3D 点流变化
- **MPC 实时规划**：推理速度约 **0.1 秒**，可直接用于模型预测控制

**实验亮点**：
- 约 **200 万条轨迹、500 小时数据** 训练
- 在真实机器人操作任务上验证，支持 DROID 和 BEHAVIOR 数据集
- 提供完整的 training / evaluation pipeline 和 pretrained checkpoints

**VLA 关联**：
- 与 **跨具身适应**（cross-embodiment adaptation）高度相关
- PointWorld 的"embodiment-agnostic 3D representation"为跨机器人泛化提供了统一的 3D 表示
- 未来可以结合：3D point flow 作为中间表示 → 机器人适配器 → 具体机器人动作

**阅读建议**：重点理解为什么 3D point flow 比 RGB 视频更适合跨本体世界模型，以及它如何减少对特定机器人 action representation 的依赖。

### 三篇论文核心差异对比

| 论文 | 核心世界表示 | 主要解决问题 | 最适合研究 |
|------|-------------|-------------|----------|
| **DreamDojo** | 视频 / latent action | 从海量人类视频学习通用世界动力学 | Foundation World Model 预训练规模化 |
| **RISE** | Dynamics + Value Model | 在想象世界里提升机器人策略 | World Model + RL 策略自提升 |
| **PointWorld** | 3D Point Flow | 跨本体 3D 动力学预测与规划 | 3D World Model / Embodiment Generalization |

### 论文阅读优先级

| 优先级 | 论文 | 理由 |
|--------|------|------|
| **P0** | Dreamer V3 | 理解世界模型的基础架构 RSSM |
| **P1** | DIAMOND | 理解 Diffusion 作为世界模型 |
| **P1** | V-JEPA 2 | 理解非生成式世界模型的思路 |
| **P1** | LaDi-WM | 理解隐空间扩散世界模型 + 与 VLA 的最新融合方式 |
| **P1** | PointWorld | 理解 3D 跨本体世界模型，与你跨具身适应研究直接相关 |
| **P1** | DreamDojo | 理解从人类视频预训练 Foundation WM 的规模化路径 |
| **P1** | RISE | 理解 WM + RL 的完整闭环工程化实现 |
| **P2** | UniSim | 理解基础世界模型的思路 |
| **P2** | IRIS | 理解 Transformer 世界模型 |
| **P3** | DreamZero | 理解 2026 年视频—动作联合 WAM 的一种实现，而非通用模板 |

**主题式阅读顺序示例**：
```
PointWorld → DreamDojo → RISE
```
- **PointWorld 优先**：用于理解跨本体 3D 动力学问题
- **DreamDojo 其次**：对应"人类数据 → 机器人世界模型"的规模化方向
- **RISE 最后**：对应"高质量数据 → 世界模型 → RL 自提升"的完整闭环

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

### 第二步：理解世界模型的核心组件

在 VLA 项目 docs/07（即本文档）中有完整的世界模型理论介绍：

```bash
# 阅读世界模型文档
# 实际代码实现请参考 docs/07-world-models-for-vla.md 中引用的开源项目
```

本文档聚焦理论与论文导读，暂未附带完整的最简世界模型实现。建议读者作为练习自行实现一个最小化的 latent dynamics 模型：
- 输入：当前观测 + 动作
- 输出：预测的下一个观测
- 训练目标：最小化预测观测和真实观测的差异（MSE）

**学习要点**：
- 理解"学习动力学"和"学习策略"的区别
- 观察 prediction error 随训练的变化
- 尝试多步展开预测，看看误差如何累积

### 第三步：读 IRIS 或 DIAMOND 源码

- **IRIS**：[官方实现](https://github.com/janner/iris) — Transformer-based，代码结构清晰
- **DIAMOND**：[官方实现](https://github.com/eloialonso/diamond) — Diffusion-based，可以对比与 Diffusion Policy 的架构异同

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
| [Dreamer V3 论文](https://arxiv.org/abs/2301.04104) | 论文 | 中等 | 世界模型入门首选，RSSM 架构详解 |
| [Danijar Hafner 博客](https://danijar.com/blog/) | 博客 | 入门 | Dreamer 系列第一作者，RSSM 深度解读 |
| [DIAMOND 论文](https://arxiv.org/abs/2405.12399) | 论文 | 中等 | Diffusion 世界模型的代表性工作 |
| [V-JEPA 2 论文](https://arxiv.org/abs/2502.05055) | 论文 | 中等 | 非生成式世界模型，Meta AI |
| [Model-Based RL 综述 (Moerland et al., 2023)](https://arxiv.org/abs/2006.16712) | 综述 | 入门 | 世界模型方法系统分类与对比 |
| [Spinning Up — MBRL](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) | 教程 | 入门 | OpenAI 的 model-based RL 入门 |
| [Yannic Kilcher — Dreamer/DIAMOND](https://www.youtube.com/c/YannicKilcher) | 视频 | 中等 | 论文精讲频道，Dreamer V3 / DIAMOND 均有覆盖 |
| [NeurIPS 2020 MBRL Tutorial](https://sites.google.com/view/mbrl-tutorial) | 教程 | 进阶 | NeurIPS 官方 model-based RL 教程 |
| [Lilian Weng — World Models](https://lilianweng.github.io/posts/2023-06-23-llm-agent/) | 博客 | 中等 | 世界模型与模型预测控制的系统梳理 |
| [DIAMOND 官方实现](https://github.com/eloialonso/diamond) | 代码 | 中等 | Diffusion 世界模型参考实现 |
| [IRIS 官方实现](https://github.com/eloialonso/iris) | 代码 | 中等 | Transformer 世界模型，代码结构清晰 |
