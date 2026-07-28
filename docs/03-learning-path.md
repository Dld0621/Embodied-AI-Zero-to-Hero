# 完整学习路线

> 从 PyTorch 新手到能微调 VLA 模型的系统学习路径。每个阶段附具体任务和验证标准。

---

## Stage 0: 前置知识（可选复习）

如果以下概念还不熟悉，先花 1-2 天补充：

| 主题 | 资源 | 时间 |
|------|------|------|
| PyTorch 基础 | [PyTorch 官方 Tutorial](https://pytorch.org/tutorials/) | 半天 |
| Transformer 原理 | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 2 小时 |
| 注意力机制 | [李宏毅 Attention 讲解](https://www.youtube.com/watch?v=ugWDIIOHtPA) | 1 小时 |
| CLIP 原理 | [`tutorials/01-vlm-basics/`](../tutorials/01-vlm-basics/) | 2 小时 |

**验证标准**：能独立实现一个带注意力机制的 Seq2Seq 模型。

---

## Stage 1: VLM 基础（1-2 天）

**目标**：理解视觉-语言模型如何将图像和文本映射到同一嵌入空间。

### 任务 1.1: 实现 CLIP 风格的对比学习

使用预训练 CLIP 模型，实现图像-文本相似度计算：

```python
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 输入
image = ...  # PIL Image
texts = ["a photo of a cat", "a photo of a dog", "a robot arm"]

# 计算相似度
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits = outputs.logits_per_image  # [1, 3]
probs = logits.softmax(dim=1)
```

### 任务 1.2: 理解视觉 token 化

- 了解 ViT 如何将图像切分为 14x14 的 patch
- 可视化 CLIP 的 attention map，看模型"在看哪里"
- 对比 DINOv2 和 CLIP 的特征差异

### 任务 1.3: 运行一个 VLM 推理

使用 LLaVA 或类似模型，输入图像+问题，观察输出：

```python
# 示例代码见 tutorials/01-vlm-basics/README.md
from transformers import AutoProcessor, LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
```

**验证标准**：能解释 CLIP 的 contrastive loss 公式，并能计算给定图像-文本对的相似度分数。

---

## Stage 2: 动作表示（1 天）

**目标**：掌握机器人动作空间的各种数学表示。

### 任务 2.1: 理解动作空间

实现三种动作表示的相互转换：

```python
# 关节角度 → 末端位姿（正向运动学）
import numpy as np

def forward_kinematics(joint_angles):
    """简化的 2-DOF 正向运动学"""
    q1, q2 = joint_angles
    l1, l2 = 1.0, 1.0
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, y])

# 末端位姿 → 关节角度（逆向运动学，数值解）
def inverse_kinematics(target_xy, initial_guess):
    """使用 Jacobian 迭代求解"""
    ...
```

### 任务 2.2: Action Chunking 实现

实现一个简单的时间序列动作预测器：

```python
class ActionChunker(nn.Module):
    """预测未来 T 步的动作序列"""
    def __init__(self, obs_dim, action_dim, chunk_size=8):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, 128)
        self.decoder = nn.GRU(128, 128, batch_first=True)
        self.head = nn.Linear(128, action_dim)
        self.chunk_size = chunk_size

    def forward(self, obs):
        # obs: [B, obs_dim]
        z = self.encoder(obs).unsqueeze(1).repeat(1, self.chunk_size, 1)
        h, _ = self.decoder(z)
        actions = self.head(h)  # [B, T, action_dim]
        return actions
```

### 任务 2.3: 理解不同动作表示的优缺点

用表格总结：

| 表示方式 | 维度 | 优点 | 缺点 | 适用场景 |
|---------|------|------|------|---------|
| 关节角度 | N | 直接可执行 | 跨平台难 | 单一机器人 |
| 末端位姿 | 6/7 | 迁移性好 | 需 IK | 多平台 |
| 增量 delta | 6/7 | 鲁棒 | 累积误差 | 实时控制 |

**验证标准**：能实现 2-DOF 机械臂的 FK/IK，并实现一个预测未来 8 步动作的 Chunker。

---

## Stage 3: 简单 VLA（2-3 天）

**目标**：搭建一个最小的 VLA 推理 pipeline。

### 任务 3.1: 搭建最小 VLA 架构

组合预训练组件，构建端到端 pipeline：

```
图像 → CLIP ViT → 视觉特征 ──┐
                              ├──→ 融合层 → MLP → 动作
文本 → BERT/CLIP Text → 文本特征 ──┘
```

参考 `examples/minimal_vla.py`。

### 任务 3.2: 使用 OpenVLA 进行推理

在本地运行 OpenVLA 推理：

```bash
# 安装
pip install openvla

# 下载模型
from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

# 推理
from PIL import Image
image = Image.open("scene.jpg")
prompt = "In: What action should the robot take to pick up the red cup?\nOut:"
inputs = processor(prompt, image).to("cuda")
action = model.predict_action(inputs, unnorm_key="bridge")  # 反归一化到原始动作空间
```

### 任务 3.3: 理解模型输出

- OpenVLA 输出的是什么？（归一化后的 delta 位姿）
- `unnorm_key` 的作用？（不同数据集有不同的动作统计量）
- 如何将模型输出映射到真实机器人动作？

**验证标准**：能成功运行 OpenVLA 推理，输出合理的动作向量，并理解每个维度的含义。

---

## Stage 4: 微调实践（3-5 天）

**目标**：在自定义数据上微调 VLA 模型。

### 任务 4.1: 准备数据

创建符合 OpenVLA 格式的小型数据集：

```
data/my_robot/
├── episode_0000/
│   ├── 000.jpg
│   ├── 001.jpg
│   ├── ...
│   └── trajectory.jsonl
└── episode_0001/
    └── ...
```

每条轨迹包含：图像、语言指令、动作序列。

### 任务 4.2: 在 LIBERO 或仿真环境上微调

使用 LIBERO 基准（无需真实机器人）：

```bash
# 下载 LIBERO
pip install libero

# 使用本项目提供的微调脚本
python tutorials/04-fine-tuning/finetune_libero.py \
    --vla_path openvla/openvla-7b \
    --benchmark libero_spatial \
    --max_steps 10000
```

### 任务 4.3: 评估与迭代

- 在测试任务上评估成功率
- 分析失败案例（是感知问题还是动作问题？）
- 调整数据增强、学习率、chunk size

**验证标准**：能在 LIBERO 上微调 OpenVLA，并在至少一个任务上达到 50%+ 成功率。

---

## Stage 5: 世界模型（2-3 天）

**目标**：理解世界模型核心架构，掌握 WM + Policy 的融合方式。

**教程入口**：[`tutorials/05-world-models/README.md`](../tutorials/05-world-models/README.md)

### 任务 5.1: 理论基础

阅读 [`docs/07-world-models-for-vla.md`](./07-world-models-for-vla.md)，重点理解：
- 世界模型的三大组件（Encoder / Transition / Reward Head）
- 五大主流架构（RSSM / Transformer / Diffusion / 非生成式 / WAM）
- WM + Policy 的四种融合方式

### 任务 5.2: 最小世界模型

```bash
python examples/minimal_world_model.py --epochs 30
```

理解 Encoder + Transition Model + Reward Predictor 的训练方式，观察多步展开的误差累积。

### 任务 5.3: WM + Policy 融合对比

```bash
python examples/world_model_vla_pipeline.py
```

在同一个 2D 导航任务中，对比四种融合方式（数据生成器 / 评估器 / 规划器 / WAM）的效果差异。

### 任务 5.4: RSSM 深度实现

```bash
python examples/dreamer_rssm.py --epochs 25
```

理解 Dreamer V3 的核心 RSSM 架构：确定性 GRU + 随机性 Gaussian，Posterior vs Prior，想象展开。

**验证标准**：能解释 RSSM 中 prior 和 posterior 的区别，能说清四种 WM+Policy 融合方式的适用场景。

---

## 进阶方向（可选）

完成以上 5 个阶段后，可以选择以下方向深入：

| 方向 | 内容 | 推荐资源 |
|------|------|---------|
| **Diffusion VLA** | 用扩散模型替代回归头 | π0 论文 + Diffusion Policy 代码 |
| **多机器人** | 训练跨平台的通用策略 | Octo 论文 + 代码 |
| **Sim-to-Real** | 在仿真中训练，迁移到真实机器人 | Isaac Sim + MuJoCo |
| **长程任务** | 多步骤任务规划 | SayCan, Inner Monologue |
| **高效推理** | 模型量化、蒸馏，部署到边缘设备 | TensorRT, ONNX |

---

## 时间规划建议

| 周次 | 内容 | 产出 |
|------|------|------|
| 第 1 周 | Stage 0-1 | 理解 CLIP，能运行 VLM demo |
| 第 2 周 | Stage 2-3 | 实现最小 VLA，能运行 OpenVLA 推理 |
| 第 3 周 | Stage 4 | 在 LIBERO 上微调，达到目标成功率 |
| 第 4 周 | Stage 5 | 掌握世界模型，跑通 RSSM + WM+Policy 融合 demo |
| 第 5 周 | 进阶方向 | 选择一个方向深入，产出代码或实验报告 |

---

## 常见问题

**Q: 没有 GPU 怎么办？**
A: 可以使用 Google Colab（免费 T4 GPU），或使用较小的模型如 Octo-Base（27M 参数可在 CPU 上运行）。

**Q: 没有真实机器人怎么办？**
A: 使用 LIBERO、MetaWorld 或 PyBullet 仿真环境。本项目所有代码都支持仿真。

**Q: 数学基础不够？**
A: 重点理解矩阵乘法、梯度下降、注意力公式即可。本项目尽量避免复杂的数学推导。

**Q: 看不懂论文？**
A: 先看 [`02-key-papers.md`](02-key-papers.md) 的"核心收获"部分，再决定是否精读原文。

---

## 外部学习资源

> 除本项目文档外，以下外部资源按 VLA / RL / 世界模型三支柱分类，帮助你系统学习。

### VLA

| 资源 | 类型 | 难度 | 说明 |
|------|------|------|------|
| [OpenVLA 官方文档](https://github.com/openvla/openvla) | 文档 | 中等 | 开源 VLA 的推理、微调、部署全指南 |
| [Hugging Face VLA 博客](https://huggingface.co/blog/vlms) | 博客 | 入门 | 从 VLM 到 VLA 的演进与核心概念 |
| [Diffusion Policy 官方教程](https://diffusion-policy.cs.columbia.edu/) | 教程 | 中等 | 扩散策略从原理到代码的完整教程 |
| [Open X-Embodiment 数据集文档](https://robotics-transformer-x.github.io/) | 文档 | 入门 | 最大的开源机器人操作数据集 |
| [Stanford CS224R](https://cs224r.stanford.edu/) | 课程 | 进阶 | Stanford 机器人学习课程（含 VLA 前沿） |

### 强化学习

| 资源 | 类型 | 难度 | 说明 |
|------|------|------|------|
| [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) | 教程 | 入门 | RL 最经典的入门教程（PPO/SAC/TRPO） |
| [Stable Baselines3 文档](https://stable-baselines3.readthedocs.io/) | 文档 | 中等 | PyTorch RL 算法库，含完整 API 和示例 |
| [David Silver RL Course](https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) | 视频 | 入门 | DeepMind 经典 RL 课程（16 讲） |
| [UCB CS285 — Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) | 课程 | 进阶 | Berkeley 深度强化学习（Sergey Levine） |
| [Hugging Face RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) | 教程 | 入门 | 从 Q-Learning 到 PPO 的交互式课程 |

### 世界模型

| 资源 | 类型 | 难度 | 说明 |
|------|------|------|------|
| [Danijar Hafner 博客](https://danijar.com/blog/) | 博客 | 中等 | Dreamer 系列第一作者，RSSM 架构深度解读 |
| [Yannic Kilcher — Dreamer/DIAMOND](https://www.youtube.com/c/YannicKilcher) | 视频 | 中等 | 论文精讲频道（Dreamer V3、DIAMOND 等） |
| [Model-Based RL 综述](https://arxiv.org/abs/2006.16712) | 论文 | 入门 | 世界模型方法系统分类与对比 |
| [NeurIPS 2020 MBRL Tutorial](https://sites.google.com/view/mbrl-tutorial) | 教程 | 进阶 | NeurIPS 官方 model-based RL 教程 |
| [DIAMOND 官方实现](https://github.com/eloialonso/diamond) | 代码 | 中等 | Diffusion 世界模型参考实现 |
