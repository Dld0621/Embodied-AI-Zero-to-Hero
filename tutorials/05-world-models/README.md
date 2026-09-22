# Stage 5: 世界模型 (World Models)

> **目标**：从零理解世界模型的核心思想、主流架构、与 VLA 的融合方式，并通过可运行代码掌握世界模型的训练和使用。

> 验证边界：本页是概念与教学脚本导读。本次审查未训练这些模型，未复现论文成绩；下方数值只展示输出格式，不是保存的实验结果。合成 2D 环境不代表真实机器人、语言条件 VLA 或大规模 WAM 已被复现。

---

## 为什么学世界模型？

VLA 模型解决的是 **"看到什么 → 做什么"**（策略），而世界模型解决的是 **"做了什么 → 会发生什么"**（动力学）。

两者可以结合，但这不是机器人智能的必要条件，也不是严格的短期/长期分工：
- VLA 输出条件动作，能力与时间跨度取决于模型和训练
- 世界模型预测后果；还需规划/策略学习算法，才形成决策系统

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
| 5.3 | WM + Policy 融合管线 | [`world_model_vla_pipeline.py`](../../examples/world_model_vla_pipeline.py) | 合成任务中四种接口思路的教学对比 |
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

1. **编码器（Encoder）**：观测→ latent 表征，是否降维取决于输入与设计
2. **转移模型（Transition）**：在 latent space 预测动力学
3. **奖励预测（Reward Head）**：本例从 latent 预测奖励；并非所有世界模型都必须包含奖励头

### 五大主流架构

| 架构 | 代表工作 | 核心思路 | 适用场景 |
|------|---------|---------|---------|
| **RNN/GRU** | Dreamer V3 (RSSM) | 确定性 + 随机性分离 | 通用 RL |
| **Transformer** | IRIS | 注意力建模序列依赖 | 时序预测 |
| **Diffusion** | UniSim, DIAMOND, LaDi-WM | 去噪生成未来观测或表征 | 多模态分布 |
| **非生成式** | V-JEPA 2 | 只学表征，不生成像素 | 高效学习 |
| **WAM** | DreamZero | 同时预测状态和动作 | 端到端 |

这些不是互斥分类：Transformer 是网络结构，diffusion 是生成建模方式，WAM 描述联合预测目标。UniSim 明确采用条件视频扩散模型。[UniSim 原论文](https://arxiv.org/html/2310.06114v3)

### 与 VLA 的四种融合方式

对应 [`docs/07-world-models-for-vla.md`](../../docs/07-world-models-for-vla.md) 第 4 节：

| 融合方式 | 说明 | 复杂度 | 代表 |
|---------|------|--------|------|
| **数据生成器** | 学习模型生成合成数据供策略学习 | 依实现而定 | UniSim 的策略训练应用 |
| **评估器** | 模型比较候选动作的预测回报 | 依候选数而定 | 教学候选筛选，不是安全证明 |
| **规划器** | 模型多步展开并搜索动作 | 依搜索预算而定 | 本页随机射击规划演示 |
| **WAM** | 联合建模动作与未来世界 | 依模型而定 | DreamZero（本页不复现） |

MimicGen 是利用人类示范生成新示范的数据生成系统，不能仅因“产生数据”就称为学习世界模型；MBPO 用从真实数据分支的短模型 rollout 做策略优化，不等同每步 MPC 搜索。[MimicGen](https://mimicgen.github.io/)、[MBPO](https://arxiv.org/abs/1906.08253)

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

1. **编码器**：将 4D 状态 `[x, y, vx, vy]` 映射到 16D latent space，这是维度扩展，不是压缩
2. **转移模型**：在 latent space 预测 `(z_t, a_t) → z_{t+1}`，用残差连接
3. **奖励预测**：从 latent 预测标量 reward（MuZero 风格）
4. **多步展开**：观察误差如何随步数累积（compounding error）

### 输出格式示意（非运行记录）

```
Epoch  5/30 | Trans Loss: 0.0412 | Rew Loss: 0.3821
Epoch 10/30 | Trans Loss: 0.0156 | Rew Loss: 0.2945
...
  单步预测误差: 0.0234
  5步后误差:   0.1245
  10步后误差:  0.2987
  → 误差随步数累积，这是世界模型的核心挑战
```

### 关键接口示意

下面省略了模型构造、训练与输入准备。本仓库 `MinimalWorldModel.encode` 返回编码均值（其内部 encoder 才返回均值和 logvar）；替换模型时须重新核对签名，不能把示意变量名当完整可执行程序。

```text
# 核心只有 3 行
z_t = world_model.encode(state_t)                    # 编码
z_pred = world_model.predict_next(z_t, action_t)      # 转移预测
reward = world_model.predict_reward(z_pred)           # 奖励预测
```

---

## 5.3 WM + Policy 融合管线

**文件**：[`examples/world_model_vla_pipeline.py`](../../examples/world_model_vla_pipeline.py)

### 核心思想

在同一个合成 2D 导航任务中对比 WM + Policy 的四种接口；这里没有因此复现带图像/语言输入的大规模 VLA。

### 运行

```bash
cd examples
python world_model_vla_pipeline.py
```

### 你会学到

1. **Baseline**：纯 BC 训练策略（无 WM）
2. **融合 1**：用 WM 生成虚拟数据训练新策略
3. **融合 2**：比较候选动作的模型预测 reward
4. **融合 3**：用模型多步展开与随机搜索，检查计算成本和模型误差
5. **融合 4**：带动作输出头的教学结构，不代表 DreamZero 等 WAM 的完整训练方案

### 输出格式示意（非运行记录）

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

> 这些方法没有通用胜负顺序。比较时固定数据预算、种子、候选数与规划时域，分别报告真实环境回报、模型预测误差和计算耗时；不能以本页示意数字推出方法排名。

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

1. **确定性部分**（GRU）：按确定函数更新历史记忆，输入包括先前随机状态与动作
2. **随机性部分**（Latent z）：用分布表达剩余不确定性，观测可帮助更新后验
3. **为什么分离**：这是表征与推断设计，不是把运动学归给 GRU、碰撞/摩擦归给随机项；碰撞也可由确定动力学建模，不确定性取决于观测和模型
4. **与 VLA 编码器的对比**：RSSM 的"确定性 + 随机"分离思想可以启发 VLA 的状态表征设计

### 关键代码片段

```text
# RSSM 数据流伪代码，不是可以直接执行的 PyTorch 调用
h_det = recurrent_update(h_prev, z_prev, action_prev)
posterior_dist = posterior(h_det, encoded_obs)  # 条件于当前观测
prior_dist = prior(h_det)                      # 不用当前观测
z_stoch = sample(posterior_dist)

# 训练目标
loss = weighted_KL(posterior_dist, prior_dist) + reconstruction_loss + reward_loss
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
