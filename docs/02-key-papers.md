# 关键论文导读

> **阅读边界：**2026-09-05 修正了 RT-1 / SPOC 引用、动作 token、π0 推理以及若干无依据性能断言。下文是论文阅读入口，不是本仓库实验复现。模型版本、训练预算、任务分布和统计口径不同，不能直接横向排名；完整核查状态见 [审查交接](reviews/content-audit-handoff.md)。

> 10 篇策略、表示与机器人学习论文 + 4 篇世界模型补充。并非每篇都是语言条件 VLA；每篇附机制、阅读重点与适用边界。

> 📚 **想查看更完整的具身智能论文分类与整理？** 欢迎访问 [Embodied-AI-Paper-Analysis](https://github.com/Dld0621/Embodied-AI-Paper-Analysis) — 覆盖 VLA、强化学习、世界模型等 7 大方向的论文体系化梳理，按顶会分类、带 venue tier 标注。

---

## 里程碑论文

### 1. RT-1: Robotics Transformer 1

- **论文**: *RT-1: Robotics Transformer for Real-World Control at Scale* (Google DeepMind, 2022)
- **arXiv**: [2212.06817](https://arxiv.org/abs/2212.06817)
- **代码**: [google-research/robotics_transformer](https://github.com/google-research/robotics_transformer)

**为什么读**：研究语言条件、多任务机器人策略如何同时兼顾数据规模和推理预算；不是此前不存在 Transformer 机器人控制。

**核心架构**：
- 输入：6 张历史图像（FiLM 条件化语言指令）+ 自然语言指令
- 主干：EfficientNet-B3 视觉编码器 → TokenLearner 压缩 → Transformer Decoder
- 输出：各动作维度离散到 256 个 bin；最终 RT-1 不逐动作维度自回归生成，见原文附录 D.4
- 数据：130k 条演示，700+ 任务

**核心收获**：
- 历史帧（temporal context）对操作任务至关重要
- 每帧 EfficientNet 特征展平为 81 个 token，TokenLearner 压到 8 个；6 帧合计 48 个视觉 token，不是“6k → 81”
- 分别读数据规模、多样性和模型消融；不能把该实验归纳成所有任务的通用胜负公式。依据：[原文 §5.1 / 附录 D.4](https://arxiv.org/html/2212.06817v1)

---

### 2. RT-2: Vision-Language-Action Models

- **论文**: *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (Google DeepMind, 2023)
- **arXiv**: [2307.15818](https://arxiv.org/abs/2307.15818)
- **代码**: 官方未开源完整训练代码，推理参考社区实现

**为什么读**：理解如何让预训练 VLM（PaLI-X / PaLM-E）直接输出机器人动作，以及论文如何定义 VLA。

**核心思想**：
- 将机器人动作表示为文本 token（如 `"1 128 91 241 5 1"`）
- 直接微调预训练 VLM，使其输出这些"动作文本"
- 同时保留 VLM 的语义推理能力（可解释符号、推理物体关系）

**关键创新**：
- **Co-training**：联合使用机器人与视觉-语言数据，以保留语义能力；不是保证永不遗忘
- 论文探索先产生语义推理再输出动作；不要据此推出模型具有未验证的夹持力感知或安全力控能力

**核心收获**：
- VLM 的互联网知识可以通过微调迁移到物理控制
- 动作离散化是将连续控制问题转化为语言建模问题的桥梁
- 语义泛化需按原文任务和测试分布理解；RT-2 本身也利用示范监督，不能简单与“BC”视为互斥类别

---

### 3. OpenVLA: An Open-Source Vision-Language-Action Model

- **论文**: *OpenVLA: An Open-Source Vision-Language-Action Model* (Stanford / UC Berkeley / MPI, 2024)
- **arXiv**: [2406.09246](https://arxiv.org/abs/2406.09246)
- **代码**: [openvla/openvla](https://github.com/openvla/openvla) ⭐ 强烈推荐

**为什么读**：开放 7B VLA 的权重、训练与适配流程，适合把架构与代码相互对照；不使用没有同口径证据的“最活跃”、固定成本或性能百分比。

**核心架构**：
- 视觉编码器：**DINOv2** + **SigLIP**（双塔融合）
- 语言主干：**Llama 2**（7B）
- 动作输出：vanilla OpenVLA 将各维离散为 256 个 bin，以动作 token 的交叉熵训练；不是连续 MLP + MSE
- 训练数据：从 Open X-Embodiment 筛选的约 970k 条机器人轨迹；目标任务微调与预训练要分开记录

**关键设计**：
- DINOv2 提供空间几何理解，SigLIP 提供语言对齐
- 使用 **Llama 2** 语言主干；主干选择不自动证明动作决策正确
- 基础版使用单图输入；不能把后续多图、动作分块或连续动作变体的能力写回 vanilla 版本

**核心收获**：
- 用同一任务、同一机器人与评估次数对照论文中的基线，而不是引用一个通用“85%”
- 分清离散化、反归一化和控制器转换三步；模型名称不能决定动作坐标、单位或关节含义
- 微调所需步数取决于数据、任务和配置，没有“5k–10k 步必能适配”的保证。依据：[原文 §3.2–3.4 / 实验设置](https://arxiv.org/html/2406.09246v3)

**快速上手**：

先按 [官方安装与推理说明](https://github.com/openvla/openvla) 固定依赖和模型 revision；不要使用未经核对的 `pip install openvla` 或把 shell 命令与 Python 混在同一代码块。以下只创建源码安装环境，不会下载权重或完成推理；真实模型示例见 [入门教程](../tutorials/03-simple-vla/README.md)。

```bash
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
```

---

### 4. π0 (pi-zero): A Vision-Language-Action Flow Model

- **论文**: *π0: A Vision-Language-Action Flow Model for General Robot Control* (Physical Intelligence, 2024)
- **arXiv**: [2410.24164](https://arxiv.org/abs/2410.24164)
- **代码**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

**为什么读**：使用**流匹配（Flow Matching）**生成动作，在精细操作任务上表现出色（叠衣服、装袋等）。

**核心架构**：
- 基于 **Conditional Flow Matching** 生成连续动作块，不能与所有扩散采样器混作同一实现
- VLM 主干由 **PaliGemma** 初始化，随后在机器人数据上继续训练
- **Action Expert** 为状态和动作 token 提供另一组参数；“独立参数组”不等于“VLM 永久冻结”

**关键设计**：
- 原论文推理执行 10 步 flow 积分，不是单次前向；观察侧缓存与较小 action expert 降低重复计算
- 区分动作执行频率和重推理频率：50 Hz 的执行不表示每 20 ms 完整生成新动作块
- 训练混合包含多种机器人操作平台和移动操作数据，不能直接视为通用导航策略

**核心收获**：
- 流模型能够表示复杂动作分布，但轨迹平滑、可达与安全仍需单独检查
- 读懂共享注意力与两组权重的关系，再判断具体配置冻结哪些参数
- 高频执行需要合适的推理和调度预算。依据：[原文 §IV / 附录 A-D](https://arxiv.org/html/2410.24164v1)

---

### 5. Octo: An Open-Source Generalist Robot Policy

- **论文**: *Octo: An Open-Source Generalist Robot Policy* (Berkeley / Stanford / Google, 2024)
- **arXiv**: [2405.12213](https://arxiv.org/abs/2405.12213)
- **代码**: [octo-models/octo](https://github.com/octo-models/octo)

**为什么读**：研究一个可适配新观察与动作空间的通用策略初始化；“可以修改后微调”不等于“任意输入、任务和机器人直接可用”。

**核心架构**：
- 基于 **Transformer** 的模块化输入与输出
- **Readout token + 分块注意力结构**：
  - 输入 tokenizer 编码观察和任务信息
  - readout token 汇聚信息，交给 diffusion action head 生成动作块
- 支持 **Goal Conditioning**：目标图像 + 语言指令

**关键设计**：
- 以 token 接口组合模态，但实际 checkpoint 仍要求确定的字段、维数、历史 mask 与统计
- 新动作空间通常需更换/适配输出头并微调；基础评估不能解释为任意动作直接互换
- 论文发布 Octo-Small（27M）与 Octo-Base（93M）；这是所报策略模型规模，不可忽略外部编码组件

**核心收获**：
- 将架构、数据与参数量分开比较，不能仅凭 27M / 7B 推断哪个更好
- Goal Image Conditioning 对需要目标状态的任务非常有用
- 多机器人联合训练需要显式动作合同与数据映射。依据：[原文 §III / 附录 D](https://arxiv.org/html/2405.12213v2)

---

## 重要扩展论文

### 6. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

- **论文**: *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (Columbia / MIT, 2023)
- **arXiv**: [2303.04137](https://arxiv.org/abs/2303.04137)
- **代码**: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)

**为什么读**：理解视觉条件的动作分布建模；原始 Diffusion Policy 不是语言条件 VLA，π0 的 flow matching 也不是照搬它的 DDPM 实现。

**核心思想**：将动作生成建模为去噪过程：

下面是机制伪代码：调度器、条件网络、训练更新都未定义，不是可运行程序；动作块长度 H 与扩散迭代数 K 是两个不同量。

```text
# 训练：向真实动作加噪，训练去噪网络
noise = torch.randn_like(action)
noisy_action = sqrt(alpha) * action + sqrt(1-alpha) * noise
predicted_noise = denoiser(noisy_action, obs, timestep)
loss = MSE(predicted_noise, noise)

# 推理：从纯噪声逐步去噪
action = torch.randn(H, action_dim)
for t in reversed(range(K)):
    action = denoiser.step(action, obs, t)
```

**核心收获**：
- 扩散模型可以表示多峰动作分布（一个场景有多个可行解）
- 对照同协议实验比较 GMM、VAE 与扩散头；不存在不依赖数据和预算的通用最优选择
- 去噪迭代次数影响速度与质量的 trade-off

---

### 7. CLIP: Learning Transferable Visual Models From Natural Language Supervision

- **论文**: *Learning Transferable Visual Models From Natural Language Supervision* (OpenAI, 2021)
- **arXiv**: [2103.00020](https://arxiv.org/abs/2103.00020)
- **代码**: [openai/CLIP](https://github.com/openai/CLIP)

**为什么读**：学习图文对比对齐的基本范式；不是所有 VLA 都使用 CLIP，DINOv2 也不是 CLIP 的派生版本。

**核心思想**：对比学习，让匹配的图像-文本对在嵌入空间靠近：

以下是省略 L2 归一化、温度参数化与训练循环的机制伪代码：

```text
# 图像编码器 + 文本编码器
image_features = image_encoder(image)   # [N, D]
text_features = text_encoder(text)       # [N, D]

# 对比损失
logits = image_features @ text_features.T / temperature
labels = arange(N)
loss = cross_entropy(logits, labels) + cross_entropy(logits.T, labels)
```

**核心收获**：
- 视觉-语言对齐是 VLA 的基石
- zero-shot 通常指没有目标任务的监督微调，不证明目标对象从未出现在互联网预训练数据中
- SigLIP 属于图文对齐路线；DINOv2 是不使用文本配对监督的视觉自监督路线

---

### 8. ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

- **论文**: *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Stanford, 2023)
- **arXiv**: [2304.13705](https://arxiv.org/abs/2304.13705)
- **代码**: [tonyzhaozh/act](https://github.com/tonyzhaozh/act)

**为什么读**：结合 ALOHA 的示范采集与 ACT 学习动作分块；本页不把历史硬件成本当当前采购报价，也不声称整套系统仅 USD 2k。

**核心架构**：
- 基于 Transformer 的编码器-解码器
- **Action Chunking with Transformer (ACT)**：一次性预测未来 K 步动作
- **CVAE**：训练时以潜变量表达示范差异；原方法测试时将潜变量设为先验均值，不是每次随机采样

**核心收获**：
- Action chunking 与 temporal ensembling 是不同设计；分块并不强制把推理从 50 Hz 降到 5 Hz，需检查实际调度
- 低成本硬件 + 简单算法可以实现令人惊讶的操作精度
- 双手协调需要同时建模两个臂的动作相关性。依据：[ACT / ALOHA 官方项目](https://tonyzhaozh.github.io/aloha/)

---

<a id="9-spoc-semantic-policy-open-vocabulary-control"></a>

### 9. SPOC: Imitating Shortest Paths in Simulation

- **论文**: *Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World* (AI2 等，2023)
- **arXiv**: [2312.02976](https://arxiv.org/abs/2312.02976)
- **代码**: [allenai/spoc-robot-training](https://github.com/allenai/spoc-robot-training)

**为什么读**：理解大规模模仿仿真规划器的轨迹如何支持导航和操作；不是原先错误展开的“Semantic-Policy-Open-vocabulary-Control”。

**核心思想**：
- 在程序化仿真场景中采集启发式规划器轨迹
- 以长上下文 Transformer 模仿规划行为
- 主模型部署使用 RGB 观察，不要求深度图或显式地图；不能和带特权检测的消融版本混为一谈

**核心收获**：
- 区分训练阶段规划器可用信息和测试时策略可用输入
- 零样本迁移结果限定于论文测试的平台、场景与任务
- 阅读成功和失败案例，特别是感知错误对长时任务的影响。依据：[官方项目与方法说明](https://spoc-robot.github.io/)

---

### 10. DINOv2: Learning Robust Visual Features without Supervision

- **论文**: *DINOv2: Learning Robust Visual Features without Supervision* (Meta, 2023)
- **arXiv**: [2304.07193](https://arxiv.org/abs/2304.07193)
- **代码**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

**为什么读**：OpenVLA 使用的视觉编码器之一；研究没有图文配对监督时如何学习可迁移视觉特征。

**核心思想**：
- 自蒸馏（self-distillation）：学生网络预测教师网络的输出
- 使用 DINO 损失 + iBOT 掩码预测
- 在大规模图像数据集上预训练（142M 图像）

**核心收获**：
- 视觉自监督与图文对齐提供不同训练信号，应按具体任务评估而非宣布普遍优劣
- attention map 是权重可视化，不是模型因果决策过程的完整解释
- 与 SigLIP 互补：DINOv2 提供空间理解，SigLIP 提供语言对齐

---

## 世界模型补充（VLA 融合方向）

> 随着项目从纯 VLA 扩展为"VLA + RL + 世界模型"三大支柱，以下补充论文聚焦世界模型如何直接服务于机器人操作与 VLA 系统。

### 11. LaDi-WM: Latent Diffusion World Model for Predictive Manipulation

- **论文**: *LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation* (国防科大 / 北京大学 / 深圳大学, CoRL 2025)
- **arXiv**: [2505.11528](https://arxiv.org/abs/2505.11528)
- **项目页**: [LaDi-WM Project](https://guhuangai.github.io/LaDiWM.github.io/)

**为什么读**：研究在视觉特征空间预测未来，并让预测引导 diffusion policy；不主张它是所有机器人 latent diffusion 的“首次”。

**核心架构**：
- **特征空间**：结合预训练视觉模型的几何与语义特征，具体编码器配置查原文
- **隐空间扩散**：预测未来特征，而不是直接生成完整 RGB 图像
- **预测引导策略**：让未来特征参与动作的迭代细化；“细化”不等于已证明动作熵单调下降

**核心收获**：
- 对照该论文的像素/特征预测实验，不把单篇结果扩展成所有操作任务的普遍优劣
- 泛化必须写清训练域、测试域、冻结/微调部分及策略如何接入预测
- 不把“10 条轨迹 / 某个成功率”作为新任务保证；本页未复算其所有表格与试验分母

**VLA 关联**：
- 视觉表示相近是可研究的融合起点，不能推出模型可直接插接或权重兼容
- 对应思路是“预测作为策略条件”；是否另做显式规划需检查实现。依据：[论文摘要与方法入口](https://arxiv.org/abs/2505.11528)
- 完整解读见 [`docs/07-world-models-for-vla.md`](./07-world-models-for-vla.md#57-ladi-wm-cori-2025)

### 12. DreamDojo: Generalist Robot World Model from Large-Scale Human Videos

- **论文**: *DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos* (Gao 等，arXiv 2026)
- **arXiv**: [2602.06949](https://arxiv.org/abs/2602.06949)
- **代码**: [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo) (Apache-2.0)

**为什么读**：从海量人类视频预训练通用机器人世界模型的代表性工作，引入 latent action 解决无动作标签数据训练问题。

**核心架构**：
- **Latent Action Model**：隐式编码人类视频中的动作信息
- **蒸馏加速**：论文报告指定系统的 10.81 FPS 生成速度，不是本仓库实测或机器人控制端到端速率
- **Post-training**：用目标机器人数据适配；不能省略这一阶段声称无需机器人数据

**核心收获**：
- 人类视频可为预训练提供大量交互信息，但目标动作映射与适配仍需要机器人数据/协议
- latent action 是跨 embodiment 动作表示的重要思路
- 蒸馏需分别比较生成质量、速度及动作可控性。依据：[论文摘要](https://arxiv.org/abs/2602.06949)

**VLA 关联**：
- 直接对应 VLA 的数据瓶颈问题
- 与你当前研究的"人类数据→机器人控制"方向高度契合
- 完整解读见 [`docs/07-world-models-for-vla.md`](./07-world-models-for-vla.md#58-dreamdojo-icml-2026)

### 13. RISE: Self-Improving Robot Policy with Compositional World Model

- **论文**: *RISE: Self-Improving Robot Policy with Compositional World Model* (OpenDriveLab, RSS 2026)
- **arXiv**: [2602.11075](https://arxiv.org/abs/2602.11075)
- **代码**: [OpenDriveLab/RISE](https://github.com/OpenDriveLab/RISE)

**为什么读**：学习如何用组合式世界模型产生想象轨迹与策略更新信号；代码入口不等于本仓库已经复现完整闭环。

**核心架构**：
- **Dynamics Model**：可控环境动力学
- **Progress/Value Model**：评估任务进度
- **Imagination RL**：在想象世界中训练策略
- **PiPER 部署**：想象训练 → 真实机器人

**核心收获**：
- 分离 dynamics 和 progress/value 后，可以针对各自目标设计模型；稳定性仍需实际实验
- 想象 rollout 可减少策略更新中的物理交互，但不能取消模型数据、真实评估和安全检查
- 复现时分别记录 world model、策略更新与真机评估的结果。依据：[论文摘要](https://arxiv.org/abs/2602.11075)

**VLA 关联**：
- 对应"高质量重定向数据 → 世界模型 → RL 自提升"的完整路线
- 与你当前的重定向项目可以自然延伸结合
- 完整解读见 [`docs/07-world-models-for-vla.md`](./07-world-models-for-vla.md#59-rise-rss-2026)

### 14. PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation

- **论文**: *PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation* (Huang 等，arXiv 2026)
- **arXiv**: [2601.03782](https://arxiv.org/abs/2601.03782)
- **代码**: [NVlabs/PointWorld](https://github.com/NVlabs/PointWorld)

**为什么读**：研究将状态与动作映射到统一 3D point flow 的方法，以及它如何参与跨本体预测。

**核心架构**：
- **3D Point Flow**：统一表示 world state + action
- **跨 embodiment**：用机器人几何把低层动作转换成 3D 表示，不表示无需本体模型或动作适配
- **MPC 接入**：论文报告约 0.1 秒模型推理；完整 MPC 搜索和闭环端到端耗时需要另量测

**核心收获**：
- 比较 3D 与 RGB 表示时同时考虑深度输入、遮挡、几何标定和计算预算
- 统一表示减少接口异构，但没有消除本体几何差异
- 论文的数据集约 200 万条轨迹、500 小时；这是作者数据范围，不是本仓库已保存或复现的数据。依据：[论文摘要](https://arxiv.org/abs/2601.03782)

**VLA 关联**：
- 3D point flow 可作为跨本体世界模型的统一表示
- 未来可结合：3D point flow 作为中间表示 → 机器人适配器 → 具体机器人动作
- 完整解读见 [`docs/07-world-models-for-vla.md`](./07-world-models-for-vla.md#510-pointworld-cvpr-2026-highlight)

---

## 阅读路线图

```
入门路线：
CLIP → RT-1 → RT-2 → OpenVLA
        ↓
    Diffusion Policy → π0
        ↓
    ACT（动手实践）

进阶路线：
OpenVLA 源码精读 → Octo 架构设计 → π0 Flow Matching
```

---

## 论文资源汇总

| 论文 | arXiv | 代码 | 难度 |
|------|-------|------|------|
| CLIP | [2103.00020](https://arxiv.org/abs/2103.00020) | [GitHub](https://github.com/openai/CLIP) | ★☆☆ |
| DINOv2 | [2304.07193](https://arxiv.org/abs/2304.07193) | [GitHub](https://github.com/facebookresearch/dinov2) | ★★☆ |
| Diffusion Policy | [2303.04137](https://arxiv.org/abs/2303.04137) | [GitHub](https://github.com/real-stanford/diffusion_policy) | ★★☆ |
| RT-1 | [2212.06817](https://arxiv.org/abs/2212.06817) | [GitHub](https://github.com/google-research/robotics_transformer) | ★★☆ |
| ACT | [2304.13705](https://arxiv.org/abs/2304.13705) | [GitHub](https://github.com/tonyzhaozh/act) | ★★☆ |
| RT-2 | [2307.15818](https://arxiv.org/abs/2307.15818) | 社区实现 | ★★★ |
| SPOC | [2312.02976](https://arxiv.org/abs/2312.02976) | [GitHub](https://github.com/allenai/spoc-robot-training) | ★★★ |
| Octo | [2405.12213](https://arxiv.org/abs/2405.12213) | [GitHub](https://github.com/octo-models/octo) | ★★★ |
| OpenVLA | [2406.09246](https://arxiv.org/abs/2406.09246) | [GitHub](https://github.com/openvla/openvla) | ★★★ |
| π0 | [2410.24164](https://arxiv.org/abs/2410.24164) | [GitHub](https://github.com/Physical-Intelligence/openpi) | ★★★★ |
| LaDi-WM | [2505.11528](https://arxiv.org/abs/2505.11528) | [Project](https://guhuangai.github.io/LaDiWM.github.io/) | ★★★★ |
| DreamDojo | [2602.06949](https://arxiv.org/abs/2602.06949) | [GitHub](https://github.com/NVIDIA/DreamDojo) | ★★★★ |
| RISE | [2602.11075](https://arxiv.org/abs/2602.11075) | [GitHub](https://github.com/OpenDriveLab/RISE) | ★★★★ |
| PointWorld | [2601.03782](https://arxiv.org/abs/2601.03782) | [GitHub](https://github.com/NVlabs/PointWorld) | ★★★★ |
