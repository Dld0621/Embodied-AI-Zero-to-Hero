# WAM 从零到一：从世界模型到视频—动作联合策略

**[English](wam-zero-to-one.md) · 简体中文 · [专项首页](README_CN.md)**

World Action Model 是正在快速发展的研究方向，还不是已经收敛的标准配方。本路线按正确顺序建立能力：动力学、Rollout 评估、规划、逆动力学、未来—动作联合学习，最后才是大规模视频模型主干。

## 1. 术语与证据边界

请严格区分：

| 系统 | 学习对象 | 如何选择动作 | 是否属于 WAM |
|---|---|---|---|
| 行为克隆策略/VLA | $p(a_t\mid o_{\le t},\ell)$ | 策略直接输出 | 否 |
| Action-conditioned World Model | $p(o_{t+1}\mid o_{\le t},a_t)$ 或潜空间等价形式 | 独立规划器/策略 | 不会仅因此成为 WAM |
| 潜空间模型强化学习/MPC | 动力学 + Reward/Value/Policy | 搜索、优化或想象策略学习 | 必做基线，不属于狭义 WAM |
| 图像—动作联合模型 | 在同一框架中建模未来图像/潜变量和动作 Token | 联合模型输出动作 | WAM 算法族 |
| 视频—动作联合生成模型 | 对齐的未来视频和动作块 | 未来生成 + 逆动力学/动作组件 | WAM 算法族 |

[WorldVLA](https://arxiv.org/abs/2506.21539)提出统一图像和动作理解/生成的自回归 Action World Model。[DreamZero](https://arxiv.org/abs/2602.15922)是 2026 年发布的工作，用 WAM 描述在视频模型基础上联合预测未来视频和动作的系统。这个名称仍然较新，不同论文的概率分解和结构并不相同；不能把所有世界模型追溯性地改名为 WAM。

## 2. 毕业合同

只有能够完成以下工作，才算学完本路线：

1. 实现并审计单步动力学模型；
2. 按预测 Horizon 测量累积 Rollout 误差；
3. 使用 MPC/搜索，并验证是否改善闭环任务结果；
4. 区分正向动力学、逆动力学、策略学习和联合建模；
5. 在受控任务上实现小型未来—动作联合模型；
6. 分别测量视频—动作对齐和控制效用；
7. 在匹配预算下对比 WAM、策略基线和世界模型基线；
8. 从数据、算力、延迟与安全解释为什么选择或放弃 WAM。

## 3. 数学进阶

### 3.1 Markov 与部分可观测设置

在状态空间中，动力学模型近似：

$$
p_\theta(s_{t+1},r_t,d_t\mid s_t,a_t),
$$

其中 $d_t$ 表示终止。在像素控制中，真实状态通常不可见，因此先把历史编码为潜状态：

$$
z_t=e_\theta(o_{\le t}), \qquad
p_\theta(z_{t+1}\mid z_t,a_t).
$$

规划时模型会递归接收自己的预测，因此单步拟合正确不代表多步有效。

### 3.2 Action-conditioned World Model + MPC

对候选动作序列 $A^{(i)}=[a_t,\ldots,a_{t+H-1}]$ 进行模型 Rollout，计算 Cost 或 Value，只执行最优序列的第一个动作，然后重新观测和规划。候选可以由 Random Shooting、CEM、梯度优化或学习到的 Proposal Policy 产生。

这个模块化基线非常重要，因为它可以分别定位：

- 表示误差；
- 动力学误差；
- Reward/Value 误差；
- 规划/搜索误差；
- 控制器和系统误差。

### 3.3 联合 World Action Model

通用联合目标为：

$$
p_\theta(O^+_t,A_t\mid h_t,\ell),
$$

其中 $h_t$ 是历史观测/状态，$O^+_t$ 是未来视觉序列，$A_t$ 是与其对齐的动作块。便于学习的概念分解是：

$$
p_\theta(O^+_t,A_t\mid h_t,\ell)
=p_\theta(O^+_t\mid h_t,\ell)
\,p_\theta(A_t\mid h_t,O^+_t,\ell).
$$

第一项预测未来世界；第二项类似逆动力学，从当前和预测的视觉演化推断机器人动作。这是教学分解，不代表所有 WAM 都使用完全相同的模块或 Loss。

<div class="dof-principle" role="group" aria-label="世界模型与世界动作模型对比图">
  <p class="dof-principle__caption"><strong>原理 · 只有与动作对齐，预测才能成为策略。</strong>传统世界模型在候选动作下预测后果；WAM 算法族联合学习未来世界演化和与之对齐的动作。只有合理视频而没有可执行动作对齐，不等于控制。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 940 300" role="img" aria-labelledby="wam-track-cn-title">
      <title id="wam-track-cn-title">模块化世界模型规划与联合世界—动作建模对比</title>
      <text class="dof-diagram-title" x="28" y="34">模块化基线</text>
      <rect class="dof-diagram-surface" x="28" y="56" width="133" height="62" rx="12"/><text class="dof-diagram-label" x="58" y="83">当前上下文</text><text class="dof-diagram-note" x="55" y="103">历史 + 目标</text>
      <path class="dof-diagram-accent" d="M175 87 H222"/><path class="dof-diagram-arrow" d="M222 87 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-blue" x="238" y="56" width="146" height="62" rx="12"/><text class="dof-diagram-label" x="267" y="83">候选动作</text><text class="dof-diagram-note" x="271" y="103">规划 / 搜索</text>
      <path class="dof-diagram-accent" d="M398 87 H445"/><path class="dof-diagram-arrow" d="M445 87 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="461" y="56" width="151" height="62" rx="12"/><text class="dof-diagram-label" x="489" y="83">世界 Rollout</text><text class="dof-diagram-note" x="493" y="103">预测 + 评分</text>
      <path class="dof-diagram-accent" d="M626 87 H673"/><path class="dof-diagram-arrow" d="M673 87 l-10 -6 v12z"/>
      <rect class="dof-diagram-surface" x="689" y="56" width="144" height="62" rx="12"/><text class="dof-diagram-label" x="720" y="83">首个动作</text><text class="dof-diagram-note" x="711" y="103">执行 + 重规划</text>
      <path class="dof-diagram-dash" d="M28 148 H910"/>
      <text class="dof-diagram-title" x="28" y="181">联合世界—动作算法族</text>
      <rect class="dof-diagram-surface" x="28" y="203" width="166" height="66" rx="12"/><text class="dof-diagram-label" x="66" y="231">历史上下文</text><text class="dof-diagram-note" x="54" y="251">视频 · 状态 · 目标</text>
      <path class="dof-diagram-violet" d="M209 236 H272"/><path class="dof-diagram-arrow-violet" d="M272 236 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="288" y="196" width="265" height="80" rx="14"/><text class="dof-diagram-title" x="347" y="227">共享世界—动作模型</text><text class="dof-diagram-note" x="329" y="250">联合 Token 或耦合生成目标</text>
      <path class="dof-diagram-violet" d="M568 218 H631"/><path class="dof-diagram-arrow-violet" d="M631 218 l-10 -6 v12z"/>
      <path class="dof-diagram-violet" d="M568 255 H631"/><path class="dof-diagram-arrow-violet" d="M631 255 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-blue" x="647" y="190" width="174" height="57" rx="12"/><text class="dof-diagram-label" x="688" y="218">未来世界</text><text class="dof-diagram-note" x="681" y="237">视频 / 潜变量</text>
      <rect class="dof-diagram-fill-good" x="647" y="254" width="174" height="36" rx="10"/><text class="dof-diagram-label" x="686" y="278">对齐动作</text>
    </svg>
  </div>
</div>

## 4. 数据合同

WAM 需要 VLA 的全部字段，还需要密集时序对齐：

```text
历史视频/状态上下文
未来视频或未来潜变量目标
对齐的动作块与动作语义
指令/目标和本体元数据
相机标定、帧率、曝光和丢帧标记
Episode、Reset、干预、成功和终止边界
```

### 4.1 动作—视频对齐

令图像时间戳为 $t^I_k$，命令时间戳为 $t^a_j$，测量状态时间戳为 $t^q_m$。需要明确在相机、网络、控制器和执行器延迟后，哪条命令对应哪次视觉变化。错一帧可能让逆动力学学到前一个或后一个动作，而不是因果动作。

可以在仿真或安全记录环境中施加已知动作脉冲，测量状态和图像第一次响应，保留估计延迟与不确定性。

### 4.2 纯视频数据

纯视频数据可以预训练视觉动力学或运动先验，但没有目标机器人动作标签。要用于控制，必须有逆动力学桥接、动作标注、本体适配或其他被明确验证的假设。不能把视频规模描述成机器人动作监督规模。

### 4.3 数据划分

至少分开测试：

- 任务/动词偏移；
- 物体偏移；
- 场景/环境偏移；
- 相机/视角偏移；
- 运动/轨迹偏移；
- 本体偏移。

“Zero-shot”必须说明哪个轴未在训练中出现。未见文本但已见运动，不等于未见物理运动。

## 5. 各组成部分的算法

### 5.1 未来表示

| 表示 | 预测对象 | 优点 | 局限 |
|---|---|---|---|
| 物理状态 | 关节/物体状态 | 可解释、紧凑 | 依赖状态估计和任务 Schema |
| 重建像素 | 未来 RGB | 监督密集、可视化 | 昂贵，浪费容量在无关细节 |
| 压缩视频潜变量 | VAE/Token Latent | 适合扩展生成模型 | 解码伪影和潜变量语义不透明 |
| JEPA/预测特征 | 未来表征 | 避免像素重建 | 难检查，需要学习规划 Cost |
| 物体/关键点潜变量 | 结构化场景元素 | 几何数据效率高 | 检测/跟踪错误传入模型 |

选择能够保留可控任务变量的最小表示。未来图像更漂亮，不代表它是更好的规划状态。

### 5.2 动力学模型

| 算法族 | 核心更新 | 适用情况 | 主要风险 |
|---|---|---|---|
| 确定性 MLP/RNN | $\hat z_{t+1}=f(z_t,a_t)$ | 第一基线、低维状态 | 平均随机未来 |
| 随机状态空间模型 | Prior/Posterior 潜动力学 | 部分可观测和不确定性 | Posterior Collapse、校准 |
| 自回归 Token 模型 | 顺序预测未来 Token | 离散图像/动作 Token | 长序列误差和延迟 |
| Diffusion/Flow 未来模型 | 条件迭代生成 | 多模态视觉未来 | 算力、采样与可控性 |
| 无解码器潜模型 | Reward/Value/Latent 特征 | 规划而非可视化 | 可能隐藏物理错误 |

[TD-MPC2](https://arxiv.org/abs/2310.16828)是无解码器潜空间规划基线；[V-JEPA 2-AC](https://arxiv.org/abs/2506.09985)是动作条件潜空间预测规划示例。二者能够处理动作和预测，但不能仅因此称为 WAM。

### 5.3 模块化基线的规划器

- **Random Shooting：**最容易验证正确性的基线，但维度扩展差。
- **CEM：**迭代用低 Cost 动作序列重拟合分布，是很好的 MPC 教学/默认基线。
- **梯度规划：**模型和 Cost 平滑时高效，但可能利用梯度或模型漏洞。
- **MCTS/Tree Search：**适合离散或结构化决策，在连续高维动作中成本高。
- **学习 Proposal/Value：**减少搜索，但规划器质量依赖学习先验。

规划器必须与随机动作、已有 Expert/Behavior Policy，以及不使用 World Model Lookahead 的同一策略对比。

### 5.4 逆动力学

逆模型根据相邻状态或视觉未来推断动作：

$$
q_\phi(a_t\mid o_{\le t},o_{t+1:t+H},\ell).
$$

它连接“未来应该是什么样”和“哪条命令可能产生它”。逆动力学通常不是唯一的：多个动作可能产生相同视觉变化，隐藏的力和接触也可能不可见。此时可能需要生成式动作分布，但真正缺失的信息也可能是力觉或状态传感，而不是模型规模。

### 5.5 自回归图像—动作联合建模

[WorldVLA](https://arxiv.org/abs/2506.21539)在自回归框架中统一图像理解/生成与动作生成。该算法族把图像和动作序列 Tokenize 或 Embedding，通过 Attention Mask 管理依赖，并优化 Token 预测。关键选择包括：

- 图像/动作交错流还是独立流；
- Action Token Codebook 和往返误差；
- 因果 Attention Mask；
- Teacher Forcing 还是 Free Running；
- Chunk 长度与误差传播。

### 5.6 视频—动作联合 Diffusion/Flow

对于视频潜变量 $Y$ 和动作块 $A$，简化的耦合 Flow 目标为：

$$
\begin{aligned}
\mathcal L ={}& \lambda_v\|v^Y_\theta(Y_\tau,A_\tau,c)-u^Y_\tau\|^2 \\
&+\lambda_a\|v^A_\theta(Y_\tau,A_\tau,c)-u^A_\tau\|^2.
\end{aligned}
$$

两种模态可以共享主干，同时使用不同 Embedding、Head、Noise Schedule 或 Loss Weight。[DreamZero](https://arxiv.org/abs/2602.15922)是建立在预训练视频扩散主干上、联合预测未来视频和动作的前沿例子。其规模、优化、迁移和真机结果只属于该系统，不是 WAM 算法族的通用保证。

### 5.7 闭环推理

1. 缓存有界历史。
2. 生成未来/动作块。
3. 检查动作 Shape、单位、边界、新鲜度和本体 ID。
4. 只执行安全前缀。
5. 重新观测，并比较真实变化与预测未来。
6. 当偏差或不确定性超过已验证边界时，重新规划、回退或停止。

开环生成视觉连贯的视频不等于机器人任务成功。

## 6. 如何在 VLA、世界模型和 WAM 之间选择

| 需求 | 首选算法族 | 原因 |
|---|---|---|
| 语义任务选择，不需要生成未来 | VLA | 直接语言到动作更简单 |
| 单任务精细反馈 | 任务策略/ACT/Diffusion + 控制器 | 大语义/视频模型不一定解决精度 |
| 反事实动作搜索 | World Model + MPC | 可分别测量模型误差和规划效用 |
| 利用视频运动先验与对齐动作 | 小型联合 WAM 实验 | 检验密集未来监督是否有效 |
| 大规模异构任务/环境/本体 | 完成基线后研究 WAM | 可能利用广泛视频动力学，但成本极高 |

只有同时满足以下条件，才建议选择联合 WAM：

- 未来视觉演化是研究假设的一部分，而不是装饰输出；
- 视频与动作对齐足以学习逆/联合动力学；
- 数据覆盖目标运动和分布偏移轴；
- 算力支持视频训练与闭环推理实验；
- 已有策略基线和模块化世界模型基线；
- 能按任务成功评估，而不只是视频指标。

否则使用能够回答问题的最小策略或世界模型。

```bash
python scripts/select_vla_wam_algorithm.py \
  --goal future-video-and-action \
  --compute cluster \
  --data heterogeneous \
  --latency soft
```

## 7. 从零到一构建顺序

| 阶段 | 构建内容 | 晋级门槛 |
|---:|---|---|
| 0 | 状态 Transition 数据集 | Episode 边界、时间对齐、单位和划分通过 |
| 1 | 单步确定性动力学 | 极小集过拟合和保留集单步误差通过 |
| 2 | 多步 Rollout | 报告按 Horizon 分解的误差与稳定性 |
| 3 | CEM/Random Shooting MPC | 规划器比匹配基线改善任务结果 |
| 4 | 视觉/潜空间世界模型 | 潜变量保留任务变量，并能在动作条件下预测 |
| 5 | 逆动力学动作模型 | 测量动作—视频对齐和歧义 |
| 6 | 小型未来—动作联合模型 | 相比模块化基线有收益或明确失败原因 |
| 7 | 分布偏移评估 | 分开任务、场景、运动、相机和本体偏移 |
| 8 | 系统优化 | 端到端频率、内存、上下文过期和回退通过 |

仓库内起点：

```bash
python scripts/run_knowledge_map.py --path-to planning-world-models
python scripts/run_pipeline.py --run world-model-planning --dry-run
python scripts/run_pipeline.py --run world-model-planning
```

本地 Pipeline 是低维教学世界模型 + 规划器，是必须先完成的基线，不是 DreamZero 或 WorldVLA 复现。

## 8. 训练目标与消融清单

至少对比：

1. 仅策略 Action Loss；
2. 仅世界 Future Loss；
3. 匹配主干/数据的联合 Loss；
4. 无语言联合模型；
5. Mask 或打乱 Future Target；
6. 正确与错位的 Action-Video Alignment；
7. Teacher-forced 与 Free-running Rollout；
8. 相同推理时间预算。

如果联合模型只改善视频质量，没有改善动作对齐或任务行为，WAM 假设没有通过。

## 9. 评估矩阵

| 维度 | 必报内容 |
|---|---|
| 世界预测 | 单步和按 Horizon 分解的 Latent/Video Error |
| 动作 | 分维度/Chunk 误差、边界和多次采样 |
| 对齐 | 同一视觉变化对应的真实/推断动作；时间偏移扫描 |
| 规划效用 | 使用/不使用模型 Lookahead 的任务成功或进度 |
| 闭环 | 成功、干预、恢复、预测偏差和约束违规 |
| 泛化 | 分别报告任务、物体、环境、运动、相机和本体 |
| 视频 | 视觉指标 + 任务相关状态/接触一致性 |
| 系统 | 生成频率、端到端延迟、内存和 Cache/历史年龄 |
| 统计 | 试验数、种子分配、置信区间和负结果 |

重建或感知相似度等视频指标不能证明动作正确；Action MSE 不能证明预测未来物理一致；任务成功本身也不能证明未来模型组件产生了贡献。三个层次都需要报告。

## 10. 失败定位

| 现象 | 优先检查 |
|---|---|
| 单步清晰、多步漂移 | Exposure Bias、随机性缺失、动作覆盖、递归 Rollout |
| 未来合理、动作错误 | 时间对齐、逆动力学歧义、动作坐标系/归一化 |
| 规划器利用不可能未来 | 模型不确定性、OOD 候选动作、约束和更短 Horizon |
| 联合 Loss 改善、任务不变 | Loss 权重、捷径、未来目标相关性和执行接口 |
| 闭环过慢 | 视频分辨率/Token、采样步数、Cache、网络、动作前缀长度 |
| 跨本体失败 | 相机/形态条件、本体适配、动作可行性和分机器人指标 |
| 精细接触失败 | 力觉/触觉/状态缺失、视觉分辨率和控制器带宽 |

## 11. 值得研究的问题

- 联合未来预测能否在冻结的运动偏移下改善动作泛化？
- 哪种未来表示以最低算力保留接触相关状态？
- 逆动力学是否受益于多未来采样或显式不确定性？
- 在相同延迟下，模块化规划器何时优于 WAM 直接动作生成？
- 如何混合纯视频数据而不淹没动作落地数据？
- 预测未来与真实观测的差异能否在失败前触发回退？
- 迁移收益究竟来自运动先验，还是额外的模型和数据规模？

比较时必须控制数据、尽可能匹配主干规模、动作语义、训练预算、推理预算和试验分配。

## 12. 一手来源

- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [DreamZero](https://arxiv.org/abs/2602.15922)
- [Action Images](https://arxiv.org/abs/2604.06168)——2026 年出现的像素化动作表示预印本，应当视为研究方向，而不是已建立的默认方案。

本路线解释算法族并定义可复现比较，不声称仓库复现了这些论文的大规模训练或真机结果。
