# Stage 5: 世界模型 (World Models)

> **目标**：从零理解世界模型的核心思想、主流架构、与 VLA 的融合方式，并通过可运行代码掌握世界模型的训练和使用。

---

## 为什么学世界模型？

VLA 模型解决的是 **"看到什么 → 做什么"**（策略），而世界模型解决的是 **"做了什么 → 会发生什么"**（动力学）。

两者结合才能实现真正的机器人智能：
- VLA 负责 **执行**（短期反应）
- 世界模型负责 **规划**（长期推理）

```
VLA:     观测 + 指令 → 动作（策略）
世界模型: 观测 + 动作 → 下一观测（动力学）
```

## 前置要求

- 理解 PyTorch 基础（nn.Module, 训练循环）
- 了解 RL 基础概念（状态、动作、奖励、MDP）
- 完成前 4 个 Stage 的 VLA 学习

## 本阶段结构

| 阶段 | 主题 | 对应文件 | 说明 |
|------|------|---------|------|
| 5.1 | 世界模型是什么 | [`docs/07-world-models-for-vla.md`](../../docs/07-world-models-for-vla.md) | 理论基础、分类框架、论文导读 |
| 5.2 | 最小世界模型实现 | [`minimal_world_model.py`](../../examples/minimal_world_model.py) | 30 行核心代码，从零训练一个 WM |
| 5.3 | WM + Policy 融合管线 | [`world_model_vla_pipeline.py`](../../examples/world_model_vla_pipeline.py) | 四种融合方式的可运行对比 |
| 5.4 | RSSM 深度解析 | [`dreamer_rssm.py`](../../examples/dreamer_rssm.py) | Dreamer V3 核心架构简化实现 |

---

## 5.1 世界模型是什么

**详细理论请阅读**：[`docs/07-world-models-for-vla.md`](../../docs/07-world-models-for-vla.md)

### 一句话定义

> 世界模型学习环境的动力学规律（"如果做动作 A，环境会变成什么样"），让机器人能在"脑中"预演未来。

### 三大核心组件

```
┌─────────────────────────────────────────────────┐
│                  World Model                      │
│                                                   │
│  观测 o_t ──→ [编码器 Encoder] ──→ z_t (latent)   │
│                                                  │
│  z_t + a_t ──→ [转移模型 Transition] ──→ z_{t+1}  │
│                                                  │
│  z_t ──→ [奖励预测 Reward Head] ──→ r_t           │
└─────────────────────────────────────────────────┘
```

1. **编码器（Encoder）**：高维观测（图像/点云）→ 低维 latent
2. **转移模型（Transition）**：在 latent space 预测动力学
3. **奖励预测（Reward Head）**：MuZero 风格，从 latent 预测奖励

### 五大主流架构

| 架构 | 代表工作 | 核心思路 | 适用场景 |
|------|---------|---------|---------|
| **RNN/GRU** | Dreamer V3 (RSSM) | 确定性 + 随机性分离 | 通用 RL |
| **Transformer** | IRIS, UniSim | 注意力建模长程依赖 | 长时序任务 |
| **Diffusion** | DIAMOND, LaDi-WM | 去噪生成未来状态 | 多模态分布 |
| **非生成式** | V-JEPA 2 | 只学表征，不生成像素 | 高效学习 |
| **WAM** | DreamZero | 同时预测状态和动作 | 端到端 |

### 与 VLA 的四种融合方式

对应 [`docs/07-world-models-for-vla.md`](../../docs/07-world-models-for-vla.md) 第 4 节：

| 融合方式 | 说明 | 复杂度 | 代表 |
|---------|------|--------|------|
| **数据生成器** | WM 生成虚拟轨迹训练 VLA | 低 | MimicGen |
| **评估器** | WM 预演候选动作，选最优 | 中 | 安全验证 |
| **规划器** | WM 多步展开 + 搜索 | 高 | MBPO, LaDi-WM |
| **WAM** | WM 直接输出动作 | 最高 | DreamZero |

---

## 5.2 最小世界模型实现

**文件**：[`examples/minimal_world_model.py`](../../examples/minimal_world_model.py)

### 核心思想

用合成数据（2D 点质量环境）训练一个最简世界模型，理解三大组件的工作方式。

### 运行

```bash
cd examples
python minimal_world_model.py --epochs 30
```

### 你会学到

1. **编码器**：将 4D 状态 `[x, y, vx, vy]` 压缩到 16D latent space
2. **转移模型**：在 latent space 预测 `(z_t, a_t) → z_{t+1}`，用残差连接
3. **奖励预测**：从 latent 预测标量 reward（MuZero 风格）
4. **多步展开**：观察误差如何随步数累积（compounding error）

### 预期输出

```
Epoch  5/30 | Trans Loss: 0.0412 | Rew Loss: 0.3821
Epoch 10/30 | Trans Loss: 0.0156 | Rew Loss: 0.2945
...
  单步预测误差: 0.0234
  5步后误差:   0.1245
  10步后误差:  0.2987
  → 误差随步数累积，这是世界模型的核心挑战
```

### 关键代码片段

```python
# 核心只有 3 行
z_t = world_model.encode(state_t)                    # 编码
z_pred = world_model.predict_next(z_t, action_t)      # 转移预测
reward = world_model.predict_reward(z_pred)           # 奖励预测
```

---

## 5.3 WM + Policy 融合管线

**文件**：[`examples/world_model_vla_pipeline.py`](../../examples/world_model_vla_pipeline.py)

### 核心思想

在同一个 2D 导航任务中，对比世界模型与 VLA 的 **四种融合方式** 的效果差异。

### 运行

```bash
cd examples
python world_model_vla_pipeline.py
```

### 你会学到

1. **Baseline**：纯 BC 训练策略（无 WM）
2. **融合 1**：用 WM 生成虚拟数据训练新策略
3. **融合 2**：用 WM 预演 5 个候选动作，选 reward 最高的（最实用）
4. **融合 3**：用 WM 展开 5 步 + 随机搜索（最强但最慢）
5. **融合 4**：WAM — WM 直接输出动作（最简洁但最难训练）

### 预期输出

```
[融合方式 1] 世界模型作为数据生成器
  WM 数据生成训练的策略平均 reward: -1.23

[融合方式 2] 世界模型作为评估器
  WM 评估器引导的策略 reward: -0.87

[融合方式 3] 世界模型作为规划器
  WM 规划器引导的策略 reward: -0.72

[融合方式 4] World Action Model
  WAM 策略 reward: -0.95
```

### 关键结论

> **融合方式 2（评估器）和 3（规划器）最常用**。评估器简单高效，规划器适合长程任务。WAM 架构优雅但训练难度最大。

---

## 5.4 RSSM 深度解析

**文件**：[`examples/dreamer_rssm.py`](../../examples/dreamer_rssm.py)

### 核心思想

RSSM（Recurrent State-Space Model）是 Dreamer V3 的核心，也是世界模型最重要的架构之一。

### 运行

```bash
cd examples
python dreamer_rssm.py --epochs 30
```

### 你会学到

1. **确定性部分**（GRU）：捕捉世界的可预测规律（运动学）
2. **随机性部分**（Latent z）：捕捉不确定性（碰撞、摩擦）
3. **为什么分离很重要**：机器人环境确定性（关节运动）和随机性（物体交互）需要分别建模
4. **与 VLA 编码器的对比**：RSSM 的"确定性 + 随机"分离思想可以启发 VLA 的状态表征设计

### 关键代码片段

```python
# RSSM 前向传播
h_det = gru(h_prev, z_prev)       # 确定性：GRU 记忆
z_stoch = posterior(h_det, obs)    # 随机性：从观测推断
z_prior = prior(h_det)            # 先验：不依赖观测

# 训练目标
loss = KL(posterior || prior) + reconstruction(z, obs) + reward_loss(z)
```

---

## 学习路线总结

```
5.1 理论 → 读 docs/07-world-models-for-vla.md
  ↓
5.2 实践 → 跑 minimal_world_model.py（理解三大组件）
  ↓
5.3 融合 → 跑 world_model_vla_pipeline.py（对比四种融合方式）
  ↓
5.4 深入 → 跑 dreamer_rssm.py（掌握 RSSM 架构）
  ↓
进阶 → 读 Dreamer V3 / DIAMOND / LaDi-WM 源码
```

## 推荐阅读

- Dreamer V3 论文（P0，RSSM 基础）
- DIAMOND 论文（P1，Diffusion 世界模型）
- LaDi-WM 论文（P1，隐空间扩散 + 迭代策略优化）
- **PointWorld** 论文（P1，3D 跨本体世界模型，与你重定向研究直接相关）
- **DreamDojo** 论文（P1，人类视频预训练 Foundation WM 规模化）
- **RISE** 论文（P1，WM + RL 完整闭环工程化）
- V-JEPA 2 论文（P1，非生成式世界模型）
- 完整论文导读见 [`docs/07-world-models-for-vla.md`](../../docs/07-world-models-for-vla.md#5-关键论文导读10-篇)