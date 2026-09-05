# VLA 从零到一：算法、选型与证据

**逐点细解：**[神经网络](../knowledge-atlas/learning-neural-networks/index.md) → [Transformer 与多模态](../knowledge-atlas/learning-transformers-multimodal/index.md) → [动作表示](../knowledge-atlas/learning-action-representations/index.md) → [VLA](../knowledge-atlas/learning-vla/index.md)。每一步都有手算例子、图和自测。

**[English](vla-zero-to-one.md) · 简体中文 · [专项首页](README_CN.md)**

这是仓库的 VLA 规范主线。目标不是“成功调用一次大模型 checkpoint”，而是理解从同步机器人数据到闭环动作之间的全部合同，能够实现主要算法族，并知道什么时候不应该选择 VLA。

## 1. 毕业合同

只有能够完成以下工作，才算学完本路线：

1. 无歧义地定义观测、语言、状态、动作、时序和 episode 语义；
2. 在预训练 VLA 之前实现匹配预算的行为克隆基线；
3. 推导离散动作 token、直接回归、扩散和流匹配目标；
4. 根据任务多模态性、延迟、数据与算力选择动作算法；
5. 使用安全过滤进行滚动时域闭环评估；
6. 通过受控消融证明语言是否真正影响行为；
7. 分别报告任务成功、失败、延迟、不确定性和分布偏移。

## 2. 问题定义

时刻 $t$ 的上下文定义为：

$$
c_t = (I_{t-k:t}^{1:V},\; q_{t-k:t},\; \ell,\; m_t),
$$

其中 $I^{1:V}$ 是同步的多相机观测， $q$ 是机器人状态， $\ell$ 是任务指令或目标， $m_t$ 包含 mask 和本体元数据。VLA 预测动作或动作块：

$$
\pi_\theta(A_t \mid c_t), \qquad A_t=[a_t,\ldots,a_{t+H-1}].
$$

这个公式隐藏了最常见的错误。每个符号都必须有合同：

| 合同 | 必须回答的问题 |
|---|---|
| 图像 | 相机、曝光、裁剪、颜色顺序、时间戳和丢帧策略是什么？ |
| 状态 | 关节空间还是任务空间？位置、速度、力？坐标系、单位、顺序和频率是什么？ |
| 语言 | Episode 级还是 step 级？模板、改写、目标图像还是允许为空？ |
| 动作 | 绝对量还是增量？关节、末端、夹爪、速度还是力矩？坐标系和 horizon 是什么？ |
| 时序 | 观测到动作延迟、控制频率、预测频率和实际执行 horizon 是什么？ |
| 边界 | Reset、成功、失败、超时、人工干预和 padding mask 如何定义？ |

这些语义不明确时，训练 Loss 无法被正确解释。

> **时序交互实验：** 在[动作块时间线](../learning-lab-cn.md#timing)中分别改变预测长度 H、执行前缀 E 与推理耗时。预测 16 步不意味着必须执行 16 步；串行模型下，100 ms 推理加上 4 个 50 ms 动作构成 300 ms 周期，最后动作开始时依据的观测已有 250 ms 历史。先解释这一计算，再讨论异步推理如何改变假设。实验不代表任何 VLA 的实测速度。

<div class="dof-principle" role="group" aria-label="VLA 组件与控制闭环原理图">
  <p class="dof-principle__caption"><strong>原理 · 表示学习只完成了一半。</strong>策略融合视觉、语言和状态上下文；但只有机器人适配器、频率合同、命令边界和重新观测共同生效，输出才会成为有效闭环。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 270" role="img" aria-labelledby="vla-track-cn-title">
      <title id="vla-track-cn-title">VLA 编码、融合、动作生成、适配与反馈闭环</title>
      <text class="dof-diagram-title" x="26" y="35">上下文编码 → 多模态融合 → 动作分布 → 有界执行</text>
      <rect class="dof-diagram-fill-blue" x="26" y="66" width="150" height="39" rx="8"/><text class="dof-diagram-label" x="58" y="91">图像历史</text>
      <rect class="dof-diagram-fill-violet" x="26" y="115" width="150" height="39" rx="8"/><text class="dof-diagram-label" x="54" y="140">指令 / 目标</text>
      <rect class="dof-diagram-fill-good" x="26" y="164" width="150" height="39" rx="8"/><text class="dof-diagram-label" x="58" y="189">机器人状态</text>
      <path class="dof-diagram-accent" d="M190 135 H250"/><path class="dof-diagram-arrow" d="M250 135 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="266" y="79" width="157" height="112" rx="14"/><text class="dof-diagram-title" x="318" y="112">融合</text><text class="dof-diagram-note" x="289" y="139">Attention / FiLM</text><text class="dof-diagram-note" x="301" y="160">Mask + 元数据</text>
      <path class="dof-diagram-accent" d="M438 135 H496"/><path class="dof-diagram-arrow" d="M496 135 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="512" y="79" width="170" height="112" rx="14"/><text class="dof-diagram-title" x="551" y="111">动作模型</text><text class="dof-diagram-note" x="540" y="138">Token / 回归</text><text class="dof-diagram-note" x="536" y="159">Diffusion / Flow</text>
      <path class="dof-diagram-accent" d="M697 135 H747"/><path class="dof-diagram-arrow" d="M747 135 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="763" y="79" width="132" height="112" rx="14"/><text class="dof-diagram-label" x="797" y="110">适配器</text><text class="dof-diagram-note" x="786" y="138">单位 · 边界</text><text class="dof-diagram-note" x="781" y="159">频率 · Watchdog</text>
      <path class="dof-diagram-violet" d="M829 205 C829 252 104 252 104 215"/><path class="dof-diagram-arrow-violet" d="M104 215 l-7 11 h13z"/><text class="dof-diagram-note" x="347" y="252">只执行有界前缀，重新观测，再次预测</text>
    </svg>
  </div>
</div>

## 3. 数据 Pipeline

### 3.1 最小 Episode Schema

每一步至少保留：

```text
episode_id, step_id, timestamp
images[camera_name]
robot_state, state_names, state_units
action, action_names, action_units, action_frame
instruction_or_goal, task_id
success, terminal, timeout, intervention
embodiment_id, calibration_id, dataset_version
```

记录的观测必须是动作决策时真正可见的观测，而不是更晚到达的相机帧。数据应按 episode 划分；如果要测试物体、场景、任务或本体泛化，还必须按对应实体隔离，不能随机拆分单帧。

### 3.2 归一化

只用训练集拟合统计量，并保存正变换与逆变换。动作可以按训练集分位数或已知物理边界映射，但必须保留单位和饱和语义。执行往返检查：

$$
a \xrightarrow{N} \tilde a \xrightarrow{N^{-1}} \hat a,
\qquad \|a-\hat a\| < \varepsilon.
$$

两个机器人动作向量维度相同，不代表动作语义相同。

### 3.3 覆盖审计

分别报告任务、物体、位姿、背景、相机、轨迹长度、失败/干预和动作范围。大量同义语言描述不等于物理运动多样性；同一条 episode 的大量帧也不等于大量独立示范。

## 4. 各组成部分的算法

### 4.1 视觉表示

| 选择 | 适用情况 | 主要风险 | 必做消融 |
|---|---|---|---|
| 从头训练的小 CNN | 单一受控任务，匹配图像较充分 | 过拟合外观 | 背景/视角偏移 |
| 冻结预训练视觉编码器 | 机器人数据有限 | 特征可能忽略接触几何 | 冻结 vs 局部微调 |
| VLM 视觉塔 | 语言落地是核心 | 语义特征可能损失精细几何 | 语言与空间扰动 |
| 视频/历史编码器 | 遮挡、速度或接触阶段重要 | 时序泄漏与延迟 | 单帧 vs 历史 |

从能暴露任务信号的最小表示开始。更强的语义预训练不能替代标定、深度、本体感知或力觉。

### 4.2 语言条件

常见做法包括把语言 token 与视觉 token 拼接后做 Attention、Cross-Attention、FiLM 调制或目标 Embedding。关键不是使用了哪一种模块，而是进行因果检查：

- 正确指令；
- 同义改写；
- 与另一个任务交换指令；
- 空指令或 Mask；
- 视觉完全相同但目标不同。

如果正确指令和交换指令得到相同行为，不能声称模型完成了语言落地。

### 4.3 多模态融合

| 融合方式 | 优点 | 局限 |
|---|---|---|
| Pool 后拼接 + MLP | 简单、快速 | 细粒度 token 交互弱 |
| Encoder Transformer | 跨模态 Attention 灵活 | Token 成本随长度增长 |
| Decoder-only Token 流 | 可复用自回归语言建模 | 顺序、Mask 与解码延迟重要 |
| 独立 VLM + Action Expert | 保留语义主干并专门建模控制 | 接口和优化更复杂 |

### 4.4 动作表示

先选择控制语义，再选择神经网络动作头。

| 表示 | 适用情况 | 重点风险 |
|---|---|---|
| 关节位置/增量 | 固定本体、关节伺服可靠 | 难以跨机器人迁移 |
| 末端增量 + 夹爪 | 存在 IK/控制器的操作任务 | 坐标系和奇异位形错误 |
| 关节速度 | 平滑速率控制 | 积分漂移和边界 |
| 力矩 | 动力学丰富的底层控制 | 安全与模型要求高 |
| 跨本体规范动作 | 多机器人数据 | 适配器可能隐藏不可行动作 |

旋转必须声明具体表示和 Loss。不要直接对存在周期跳变的欧拉角使用普通 MSE。

### 4.5 直接回归与动作分块行为克隆

确定性连续动作块可以使用：

$$
\mathcal L_{\text{reg}} = \frac{1}{H}\sum_{h=0}^{H-1}
\|a_{t+h}-\hat a_{t+h}\|_1
$$

或 MSE。它是第一基线，因为成本低、易诊断、延迟小。其关键假设是一个中心预测能够代表条件动作分布；当数据包含互不兼容但都正确的动作时，回归可能输出二者均值。

[ACT](https://arxiv.org/abs/2304.13705) 是面向精细双臂模仿的动作分块生成策略。必须区分“预测动作块”和“完整 ACT/CVAE”：许多模型使用 Action Chunk，但不是 ACT。

### 4.6 离散动作 Token

把归一化动作分箱，或使用离散码表示动作，再优化交叉熵：

$$
\mathcal L_{\text{token}} = -\sum_n \log p_\theta(z_n\mid z_{<n},c_t).
$$

这种方式容易接入自回归 VLM，[RT-2](https://arxiv.org/abs/2307.15818)和原始 [OpenVLA](https://arxiv.org/abs/2406.09246)属于这条路线。代价是量化误差、Token 顺序选择和串行解码延迟。必须报告 Token 解码回动作的往返误差。

### 4.7 连续并行动作块

连续动作头并行预测整个 Action Chunk。[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)说明连续动作表示、并行解码和动作分块会显著改变自回归 VLA 的微调与推理权衡。但它报告的吞吐量或基准提升不能脱离原实验直接迁移到其他实现。

### 4.8 Diffusion 动作模型

简化的噪声预测形式为：

$$
x_\tau=\sqrt{\bar\alpha_\tau}A+\sqrt{1-\bar\alpha_\tau}\epsilon,
\qquad
\mathcal L_{\text{diff}}=\|\epsilon-\epsilon_\theta(x_\tau,\tau,c_t)\|^2.
$$

[Diffusion Policy](https://arxiv.org/abs/2303.04137)展示了这类方法处理多模态、高维动作分布和滚动时域控制的价值。代价是迭代推理以及额外的噪声调度和采样器选择。比较时必须固定观测、数据、Action Chunk 和执行预算。

### 4.9 Flow Matching Action Expert

在简化条件流匹配路径中，采样噪声 $x_0$、真实动作块 $x_1=A$ 和时间 $\tau$：

$$
x_\tau=(1-\tau)x_0+\tau x_1,\quad
u_\tau=x_1-x_0,\quad
\mathcal L_{\text{FM}}=\|v_\theta(x_\tau,\tau,c_t)-u_\tau\|^2.
$$

推理时对学习到的向量场进行数值积分。[π0](https://arxiv.org/abs/2410.24164)和 [SmolVLA](https://arxiv.org/abs/2506.01844)都使用流式连续动作生成，但二者架构和训练细节不可互换。必须实测端到端采样速度，不能把“Flow”当成统一的低延迟保证。

## 5. 如何选择算法

### 第一步：语言是否会改变正确动作？

- **不会：**从 BC、ACT 或 Diffusion Policy 开始。VLA 只会增加容量和虚假的语言叙事。
- **会：**在匹配基线中加入语言，然后执行交换/缺失语言消融。

### 第二步：条件动作分布是否多模态？

- **较低或未知：**先使用连续动作分块回归。
- **明确多模态：**对比 Diffusion 或 Flow Matching；同时保留多次采样和任务结果。

### 第三步：延迟合同是什么？

- **硬实时预算：**优先并行回归/动作分块；联合测量预处理、模型、后处理、网络和队列。
- **软预算：**可以尝试生成式动作头，但仍然只执行短前缀并重新观测。

### 第四步：数据和算力是什么规模？

| 情况 | 推荐第一个实验 |
|---|---|
| 单任务数据、算力有限 | 小型 Chunked BC/ACT；任务不变时不加入语言 |
| 多任务数据、单卡 | 微调紧凑或连续 Action Chunk VLA，同时保留从头训练基线 |
| 异构多机器人数据 | 显式本体适配器、分机器人归一化和分机器人指标 |
| 较大算力、语言多样 | 在相同数据上对比离散、连续和 Flow 动作头 |

### 第五步：不要只按参数量选模型

选型依据应是任务语义、数据兼容性、控制频率、动作多模态性、可复现性、许可和部署约束。可以运行：

```bash
python scripts/select_vla_wam_algorithm.py \
  --goal multimodal-action \
  --compute single-gpu \
  --data task-specific \
  --latency soft
```

## 6. 从零到一构建顺序

| 阶段 | 构建内容 | 晋级门槛 |
|---:|---|---|
| 0 | 任务与接口合同 | 坐标系、单位、频率、成功和停止条件通过审查 |
| 1 | State-only BC | 通过极小集过拟合和闭环状态基线 |
| 2 | Vision BC | 图像动作同步和未见初始状态评估通过 |
| 3 | 语言条件基线 | 正确语言优于交换/缺失语言 |
| 4 | Action Chunk | 滚动时域执行达到或超过单步策略 |
| 5 | 生成式动作头 | 多模态收益体现在任务结果，而不只是似然 |
| 6 | 预训练 VLA 微调 | 保留匹配从头训练基线和数据回执 |
| 7 | 分布偏移测试 | 单独报告物体、位姿、场景、语言和本体偏移 |
| 8 | 部署演练 | 延迟、边界、Watchdog、Shadow Mode 和回滚通过 |

仓库内建议路径：

```bash
python scripts/run_knowledge_map.py --path-to learning-vla
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run vla-policy
```

仓库中的 PushCube VLA 是教学基线，不等于复现 OpenVLA、π0 或 SmolVLA 预训练。

## 7. 训练与微调协议

1. 用少量样本检查 Schema、Mask 和归一化。
2. 过拟合极小子集；无法过拟合通常意味着数据、目标或实现错误。
3. 调参前冻结 train/validation/test episode 身份。
4. 记录主干初始化、冻结层、优化器参数组、精度、梯度累积和增强。
5. 只在相同数据和步数下比较全量、局部和参数高效微调。
6. 按预先声明的验证规则选 checkpoint，不能按测试集选择。
7. 使用固定 Reset、种子/试验分配和人工干预规则进行闭环评估。

[LeRobot 官方 SmolVLA 指南](https://huggingface.co/docs/lerobot/smolvla)建议针对具体任务微调，并给出一个数据和训练起点。其 Episode 或训练步数只适用于相应设置，不能当作普适样本复杂度规律。

## 8. 评估矩阵

| 维度 | 必报内容 |
|---|---|
| 离线 | 分动作维度/horizon Loss、解码误差、验证集身份 |
| 闭环 | 任务成功、进度、时间、干预和约束违规 |
| 语言 | 正确、同义改写、交换、缺失和矛盾指令 |
| 视觉 | 相机移除、遮挡、背景、视角和光照偏移 |
| 动作 | 边界、平滑、饱和、Chunk 重叠和执行延迟 |
| 泛化 | 分别报告任务、物体、场景、本体和运动偏移 |
| 系统 | 端到端延迟分布、内存、丢帧和队列年龄 |
| 统计 | Episode/试验数、种子策略、置信区间和失败数 |

离线动作误差下降时，任务成功可能反而下降；语言条件成功提高也可能只是因为任务和背景相关。必须报告这些失败路径，不能只给平均分。

## 9. 失败定位

| 现象 | 优先检查 |
|---|---|
| 训练 Loss 好但任务失败 | 时间同步、逆归一化、动作坐标系、Reset 分布 |
| 动作抖动或延迟 | 队列年龄、Chunk 重叠、频率不匹配、网络延迟 |
| 忽略语言 | 任务不平衡、视觉捷径、交换/缺失语言消融 |
| 两种策略被平均 | 多模态示范；对比 Diffusion/Flow 动作头 |
| 接触阶段失败 | 相机几何、本体/力反馈、动作频率和控制器，而不只是 VLA 规模 |
| 跨机器人崩溃 | 分本体适配、单位、Mask、动作可行性和被汇总隐藏的失败 |
| 离线优秀、偏移失败 | 数据泄漏、覆盖不足、Checkpoint 选择和未报告 OOD |

## 10. 值得研究的问题

- 语言是否提供了超越视觉捷径的因果任务消歧？
- 哪种动作表示可以跨本体迁移且不会隐藏不可执行命令？
- 在固定延迟预算下，生成式动作头是否改善闭环多模态行为？
- 小规模域内数据应该适配哪些层？
- 历史观测是缓解部分可观测，还是泄漏了动作时序？
- 不确定性能否足够提前预测失败并触发安全回退？

每个问题都需要匹配基线、只改变一个因素、冻结偏移测试集，并声明什么结果会否定假设。

## 11. 一手来源

- [RT-1](https://arxiv.org/abs/2212.06817)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [ACT](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Octo](https://arxiv.org/abs/2405.12213)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0](https://arxiv.org/abs/2410.24164)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [SmolVLA](https://arxiv.org/abs/2506.01844)

论文结果属于一手来源的报告。本路线提供算法与证据课程，不声称仓库复现了这些系统。
