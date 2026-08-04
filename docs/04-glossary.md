# 术语表

> VLA 领域常见术语速查，按字母顺序排列。

---

## A

**Action Chunking（动作分块）**
一次模型推理预测未来多步动作序列，而非单步。例如预测未来 8 步动作，执行前 4 步后重新推理。可减少推理频率，提高动作平滑性。

**Action Expert（动作专家）**
π0 中的设计：冻结预训练 VLM 的参数，只训练专门用于生成动作的小型模块。既保留语义理解能力，又适配特定机器人。

**Autoregressive（自回归）**
逐个 token 生成输出的方式。RT-2 将动作离散化为 token，用自回归方式逐个预测。类似 GPT 生成文本。

## B

**Backbone（主干网络）**
模型的主要特征提取部分，通常是预训练的大模型。VLA 中常用 LLaMA、T5、PaLI-X 等作为语言主干。

**Behavior Cloning（行为克隆，BC）**
监督学习方法：收集专家演示数据，训练模型直接模仿专家动作。是 VLA 最基础的训练范式。

**BC-Zero / BC-Z**
零样本行为克隆的变体，通过语言指令条件化策略，实现对未见过任务的泛化。

## C

**CLIP（Contrastive Language-Image Pre-training）**
OpenAI 提出的视觉-语言对齐模型。通过对比学习将图像和文本映射到同一嵌入空间。是几乎所有 VLA 模型的视觉编码器基础。

**Co-training（联合训练）**
同时在多种类型数据上训练，防止模型遗忘预训练知识。RT-2 在机器人数据 + 视觉-语言数据上联合训练。

**CVAE（Conditional Variational Autoencoder）**
条件变分自编码器。ACT 使用 CVAE 生成多样化的动作 chunk，增加策略的探索性。

## D

**Delta Action（增量动作）**
预测相对于当前状态的增量变化，而非绝对目标值。例如 `[dx, dy, dz]` 而非 `[x, y, z]`。更鲁棒，但误差会累积。

**DINOv2**
Meta 提出的自监督视觉模型。通过自蒸馏学习视觉特征，空间理解能力强于 CLIP。OpenVLA 的视觉编码器之一。

**Diffusion Policy（扩散策略）**
将动作生成建模为去噪扩散过程。可表示多峰分布，生成平滑动作。π0 和 Diffusion Policy 项目采用。

**Discretization（离散化）**
将连续动作值映射到有限个离散 bin 中。RT-2 将动作值离散化为 256 个 token，使其可用语言模型生成。

## E

**End-to-End（端到端）**
从原始输入（像素）直接到最终输出（动作）的单一模型，中间没有显式的人工设计模块。VLA 是端到端范式的代表。

**End-Effector（末端执行器）**
机械臂末端的工具，如夹爪、吸盘、手爪等。

## F

**FiLM（Feature-wise Linear Modulation）**
用语言嵌入对视觉特征做仿射变换，实现语言条件化。RT-1 使用 FiLM 将指令注入视觉编码器。

**Flow Matching（流匹配）**
一种生成模型技术，比传统扩散模型更高效。π0 使用流匹配生成动作，支持高频控制。

**FK（Forward Kinematics，正向运动学）**
已知关节角度，计算末端执行器位姿。是机器人学的基础。

## G

**Goal Conditioning（目标条件化）**
策略不仅接收当前观察，还接收目标状态（目标图像或目标描述）。Octo 支持目标图像条件化。

## H

**High-level Policy（高层策略）**
负责任务分解和规划的模块。与低层策略（负责具体动作生成）相对。在复杂任务中，通常需要两者结合。

## I

**IK（Inverse Kinematics，逆向运动学）**
已知末端执行器目标位姿，求解关节角度。可能有多个解或无可行解（奇异点）。

**In-Context Learning（上下文学习）**
模型在不更新参数的情况下，通过提示中的示例学习新任务。VLA 模型通常不具备此能力，需要微调。

## L

**LLM（Large Language Model）**
大语言模型，如 GPT、LLaMA、T5 等。VLA 使用 LLM 作为语言理解和推理的主干。

**LIBERO**
开源的单臂操作基准数据集，包含 130 个语言条件任务。适合快速实验 VLA 模型。

## M

**MLP（Multi-Layer Perceptron）**
多层感知机。VLA 中常用作策略头，将融合后的多模态特征映射为动作向量。

**MuJoCo**
物理仿真引擎，广泛用于机器人仿真。可模拟刚体动力学、接触、摩擦等。

## O

**Observation（观察）**
策略接收的环境信息，通常包括 RGB 图像、深度图、关节角度、力传感器读数等。

**Octo**
Berkeley / Stanford / Google 开源的通用机器人策略。27M-93M 参数，支持任意观察、任意任务、任意机器人。

**Open X-Embodiment（OXE）**
最大的开源机器人数据集，整合 22+ 机器人平台的 1M+ 条轨迹。OpenVLA、Octo 都在此上预训练。

**OpenVLA**
Stanford / UC Berkeley 开源的 7B 参数 VLA 模型。DINOv2 + SigLIP + Llama 2 架构，性能接近 RT-2-X。

## P

**PaLI-X / PaLM-E**
Google 的多语言多模态大模型。RT-2 使用 PaLI-X 作为主干，利用其强大的视觉-语言理解能力。

**Policy（策略）**
将观察映射到动作的函数。在 VLA 中，策略通常是深度神经网络。

**Policy Head（策略头）**
模型的最后一部分，负责将融合后的特征转换为动作输出。常见形式：MLP 回归、离散 token 生成、扩散模型。

**π0 (pi-zero)**
Physical Intelligence 提出的 VLA 模型，使用流匹配生成动作。在精细操作任务上表现出色。

## R

**Regression（回归）**
直接输出连续值。OpenVLA 使用 MLP 回归头输出连续动作向量，而非离散 token。

**RT-1 / RT-2（Robotics Transformer）**
Google DeepMind 的机器人 Transformer 系列。RT-1 是首个大规模真实机器人 Transformer；RT-2 首次将 VLM 转化为 VLA，是"VLA"术语的来源。

## S

**SigLIP**
Google 提出的视觉-语言模型，使用 sigmoid 损失替代 CLIP 的 softmax 损失。零样本能力强，是 OpenVLA 的视觉编码器之一。

**Sim-to-Real（仿真到真实）**
在仿真环境中训练策略，然后迁移到真实机器人。是降低数据收集成本的关键技术，但存在域差异挑战。

**State（状态）**
描述环境当前状况的完整信息。在部分可观察环境中，策略只能获取 observation 而非完整 state。

## T

**Temporal Context（时序上下文）**
历史观察序列，帮助模型理解动态变化。RT-1 使用 6 帧历史；操作任务中时序信息至关重要。

**Token**
模型处理的基本单位。在 VLA 中，可以是文本 token、图像 patch token，或离散化后的动作 token。

**TokenLearner**
Google 提出的模块，将大量视觉 token 压缩为少量关键 token。RT-1 使用它将 6k+ token 压缩到 81 个。

**Transformer**
基于自注意力机制的神经网络架构。VLA 模型的核心架构，用于融合多模态信息。

## U

**Unnormalization（反归一化）**
将模型输出的归一化动作转换回原始动作空间。不同数据集有不同的统计量（均值、标准差），需要对应的 `unnorm_key`。

## V

**VLA（Vision-Language-Action）**
视觉-语言-动作模型。接收视觉图像和自然语言指令，输出机器人动作。具身智能的核心范式。

**VLM（Vision-Language Model）**
视觉-语言模型。接收图像和文本，输出文本或图文相似度。分为两类：**生成式 VLM**（GPT-4V、LLaVA，输出文本）和**双编码器 VLM**（CLIP、SigLIP，输出 embedding/相似度，非生成式）。VLA 在 VLM 基础上增加了动作输出能力。

**ViT（Vision Transformer）**
将图像切分为 patch，用 Transformer 处理的视觉模型。CLIP、DINOv2 都基于 ViT。

## Z

**Zero-Shot（零样本）**
模型在训练时未见过某类任务，但能在测试时直接执行。VLA 通过语言指令和开放词汇视觉理解实现一定程度的零样本泛化。

---

## 缩写速查表

| 缩写 | 全称 | 中文 |
|------|------|------|
| BC | Behavior Cloning | 行为克隆 |
| CVAE | Conditional Variational Autoencoder | 条件变分自编码器 |
| DOF | Degrees of Freedom | 自由度 |
| EE | End-Effector | 末端执行器 |
| FiLM | Feature-wise Linear Modulation | 特征线性调制 |
| FK | Forward Kinematics | 正向运动学 |
| IK | Inverse Kinematics | 逆向运动学 |
| LLM | Large Language Model | 大语言模型 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| MSE | Mean Squared Error | 均方误差 |
| OXE | Open X-Embodiment | 开放跨平台数据集 |
| RL | Reinforcement Learning | 强化学习 |
| ViT | Vision Transformer | 视觉 Transformer |
| VLA | Vision-Language-Action | 视觉-语言-动作 |
| VLM | Vision-Language Model | 视觉-语言模型 |
