# VLA 核心概念详解

> 从 VLM 到 VLA：为什么机器人需要"看懂"并"听懂"才能行动？

---

## 1. 从 VLM 到 VLA：多了什么？

### VLM（Vision-Language Model）

输入：一张图 + 一句话
输出：一句话（描述、回答、推理）或图文相似度

典型例子：GPT-4V、LLaVA（生成式）；CLIP、SigLIP（双编码器，输出 embedding/similarity，非生成式）

```
[图像: 猫坐在沙发上] + [问题: 图里有什么？] → "一只橘猫坐在蓝色沙发上"
```

### VLA（Vision-Language-Action Model）

输入：一张图 + 一句话
输出：**机器人动作**（关节角度、末端位姿、速度指令等）

典型例子：RT-2、OpenVLA、π0

```
[图像: 桌面场景] + [指令: 把红杯子放到左边] → [动作序列: 夹爪张开→移动到杯子→闭合→提起→移动到左边→释放]
```

**关键区别**：VLM 输出的是**语义**（文字），VLA 输出的是**物理动作**（数字向量）。

---

## 2. VLA 的典型架构

几乎所有现代 VLA 模型都遵循以下三阶段架构：

```
视觉编码器        语言编码器         策略头
    │                │                │
    ▼                ▼                ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│  DINOv2 │     │  SigLIP │     │  MLP/   │
│  SigLIP │  +  │  T5/    │  →  │  Diffusion│  →  动作 token/向量
│  CLIP   │     │  LLaMA  │     │  Transformer│
└─────────┘     └─────────┘     └─────────┘
     ▲                ▲
     └──── 融合 ──────┘
```

### 2.1 视觉编码器（Vision Encoder）

将 RGB 图像编码为视觉特征向量。常用选择：

| 编码器 | 特点 | 代表模型 |
|--------|------|---------|
| **CLIP ViT** | 语言对齐强，通用性好 | RT-2, Octo |
| **DINOv2** | 自监督学习，空间理解好 | OpenVLA |
| **SigLIP** | 对比学习，零样本能力强 | OpenVLA |
| **SAM** | 分割级理解 | 部分工作 |

### 2.2 语言编码器 / 主干（Backbone）

理解自然语言指令，通常使用预训练 LLM：

| 主干 | 参数量 | 特点 | 代表模型 |
|------|--------|------|---------|
| **T5** | 多种 | 编码器-解码器，适合 seq2seq | 部分工作 |
| **LLaMA 2** | 7B-70B | 开源，推理强 | OpenVLA |
| **PaLI-X** | 5B/55B | Google 多语言多模态 | RT-2 |
| **Phi-3** | 3.8B | 小参数，端侧友好 | 部分工作 |

### 2.3 策略头（Policy Head）

将融合后的多模态特征映射为动作。常见形式：

**a) 离散动作 token（Autoregressive）**

将连续动作空间离散化为 token，用自回归方式逐个预测：

```python
# RT-2 / vanilla OpenVLA 的做法：将动作值映射到 256 个 bin
action_tokens = discretize(actions, bins=256)
# 然后像生成文本一样自回归预测
```

优点：可直接用预训练 LLM 的解码能力，泛化性强
缺点：量化误差，自回归生成慢

> **注意**：vanilla OpenVLA 使用 256-bin 离散 token（与 RT-2 相同）；而 OpenVLA-OFT 改用连续动作输出（L1 回归），支持 action chunking 和更高频率。两者是同一模型的不同微调方案。

**b) 连续动作回归（Regression）**

直接在输出层后接回归头输出连续动作（如 OpenVLA-OFT）：

```python
# OpenVLA-OFT 的做法
action = mlp(hidden_states)  # 输出 [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

优点：无量化误差，动作平滑
缺点：需要更多微调数据

**c) 扩散策略（Diffusion）**

用扩散模型从噪声中逐步去噪生成动作：

```python
# Diffusion Policy 的做法
noise = torch.randn(B, T, action_dim)
for t in reversed(range(T)):
    noise = denoiser(noise, t, observation_embedding)
action = noise
```

优点：可建模多峰分布，生成多样且平滑的动作
缺点：推理慢，需要多次去噪步

**d) Flow Matching（流匹配）**

Flow matching 是扩散的变体，通过学习向量场将噪声传输到动作分布。π0 和 SmolVLA 都使用此方法：

```python
# π0 / SmolVLA 的做法：flow matching action expert
# 学习一个向量场 v_θ(x_t, t) 将噪声传输到动作
x_t = (1 - t) * noise + t * action_target  # 线性插值路径
v_pred = flow_model(x_t, t, observation_embedding)
# 推理时从噪声出发，沿向量场积分得到动作
```

优点：比标准扩散推理更快（确定性 ODE 求解器），可建模多峰分布，支持 action chunking
缺点：训练和推理流程与标准扩散不同，需要专门的 action expert 模块

---

## 3. 动作表示方式

VLA 输出的动作可以有多种表示，选择取决于任务和机器人平台：

### 3.1 关节角度（Joint Angles）

直接预测每个关节的目标角度：

```python
action = [q1, q2, q3, q4, q5, q6, q7, gripper]  # 7-DOF 臂 + 夹爪
```

- 优点：直接可执行
- 缺点：不同机器人关节数不同，跨平台迁移难

### 3.2 末端位姿（End-Effector Pose）

预测末端执行器在任务空间中的 6D 位姿 + 夹爪：

```python
action = [x, y, z, roll, pitch, yaw, gripper_open]
```

- 优点：与机器人构型无关，迁移性好
- 缺点：需要 IK 解算，可能有奇异点

### 3.3 增量/ delta 动作

预测相对于当前状态的增量：

```python
action = [dx, dy, dz, droll, dpitch, dyaw, dgripper]
```

- 优点：对绝对坐标系不敏感，更鲁棒
- 缺点：误差会累积

### 3.4 Action Chunking

一次预测未来 T 步的动作序列，减少推理频率：

```python
action_chunk = model(obs, text)  # 形状: [T, action_dim]
# 执行前 K 步，然后重新推理
for t in range(K):
    robot.step(action_chunk[t])
```

- RT-1 使用 T=15, K=1（预测 15 步，每次只执行第 1 步）
- Octo 使用 T=4, K=4

---

## 4. 训练范式

### 4.1 行为克隆（Behavior Cloning, BC）

最直接的方式：收集人类演示数据，用监督学习训练模型模仿：

```
损失 = MSE(model(image, instruction), human_action)
```

- 优点：简单直接，数据利用率高
- 缺点：分布外泛化差，复合误差问题

### 4.2 预训练 + 微调

先在大型异构数据集（如 Open X-Embodiment）上预训练，再在小规模目标数据上微调：

```python
# Step 1: 在 OXE 上预训练
model.pretrain(dataset="open_x_embodiment")

# Step 2: 在目标数据上微调
model.finetune(dataset="my_robot_data")
```

这是当前最主流的范式（OpenVLA、Octo 都采用）。

### 4.3 数据格式

标准的训练数据样本：

```json
{
  "observation": {
    "image": "rgb_image.jpg",
    "instruction": "pick up the red cup"
  },
  "action": [0.1, -0.05, 0.02, 0.0, 0.0, 0.0, 1.0],
  "dataset_name": "bridge",
  "robot_type": "widowx"
}
```

---

## 5. VLA vs 传统机器人 pipeline

| | 传统 Pipeline | VLA |
|--|--------------|-----|
| 感知 | 手工设计特征 / 目标检测 | 端到端视觉编码器 |
| 理解 | 有限指令集 / 状态机 | 开放词汇自然语言 |
| 规划 | 显式运动规划 | 隐式在模型内部 |
| 控制 | PID / MPC | 模型直接输出动作 |
| 泛化 | 场景特定 | 跨场景、跨机器人 |
| 可解释性 | 高 | 低（黑盒） |

---

## 6. 当前挑战

1. **Sim-to-Real Gap**：仿真中训练的策略迁移到真实机器人有困难
2. **数据瓶颈**：高质量机器人演示数据收集昂贵
3. **推理速度**：大模型推理延迟高，难以满足实时控制需求
4. **安全性**：端到端模型的安全性难以保证
5. **长程任务**：复杂多步骤任务的规划能力仍有限

---

## 7. 推荐阅读顺序

1. 先理解 **CLIP**（视觉-语言对齐的基础）
2. 再看 **RT-1**（首个大规模机器人 Transformer）
3. 然后 **RT-2**（将 VLM 转化为 VLA）
4. 最后 **OpenVLA / π0**（最新的开源 SOTA）

详细论文导读见 [`02-key-papers.md`](02-key-papers.md)。
