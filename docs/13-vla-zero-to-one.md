# VLA 零到一：视觉-语言-动作模型实战

> **目标**: 理解 VLA (Vision-Language-Action) 模型是什么，并在本地 GPU 上运行 SmolVLA 推理，完成"语言指令 + 图像 → 机器人动作"的端到端 pipeline。

---

## 目录

1. [什么是 VLA？](#1-什么是-vla)
2. [为什么选 SmolVLA？](#2-为什么选-smolvla)
3. [环境准备](#3-环境准备)
4. [第一次推理（5 分钟）](#4-第一次推理5-分钟)
5. [理解 Pipeline](#5-理解-pipeline)
6. [SmolVLA 架构解析](#6-smolvla-架构解析)
7. [动作空间与控制](#7-动作空间与控制)
8. [与机器人控制系统的衔接](#8-与机器人控制系统的衔接)
9. [四大开源 VLA 对比](#9-四大开源-vla-对比)
10. [进阶：微调与部署](#10-进阶微调与部署)
11. [常见问题排查](#11-常见问题排查)
12. [参考文献](#12-参考文献)

---

## 1. 什么是 VLA？

**VLA (Vision-Language-Action)** 是一种将视觉感知、语言理解和动作生成统一在一个模型中的架构。输入图像 + 语言指令，直接输出机器人动作。

```
传统 Pipeline:                VLA Pipeline:
图像 → 感知模块 → 规划 → 控制    图像 + 语言 → VLA → 动作
(多个独立模块拼接)              (端到端单一模型)
```

**核心优势**:
- **端到端**: 不需要手工设计中间表示
- **多模态**: 同时理解视觉和语言
- **泛化性**: 预训练后可适应新任务、新环境
- **简洁性**: 一个模型替代一整条 pipeline

---

## 2. 为什么选 SmolVLA？

| 模型 | 参数量 | 最低显存 | pip 安装 | 灵巧操作 | 许可证 |
|------|--------|---------|---------|---------|--------|
| **SmolVLA** | **450M** | **~4 GB** | **是** | 单臂 | Apache-2.0 |
| Octo-Small | 27M | <2 GB | 是 | 单臂 | MIT |
| OpenVLA-7B | 7B | ~15 GB | 是 | 单臂 | Llama-2 |
| RDT-1B | 1B | ~8 GB | 是 | **双臂灵巧** | MIT |

**选择 SmolVLA 的理由**:
1. **最小可用 VLA**: 450M 参数是当前最小的实用级 VLA
2. **HuggingFace 生态**: `pip install lerobot` 一行安装，模型自动下载
3. **Jetson 可部署**: 在 NVIDIA Jetson Orin NX 上实测可运行（3.5GB 显存）
4. **Apache-2.0 许可**: 商用友好
5. **文档优秀**: LeRobot 框架提供完整教程和 API 文档

> **如果你需要双臂灵巧操作**，请参考 [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer)，本文最后有对比。

---

## 3. 环境准备

### 3.1 硬件要求

| 需求 | 最低 | 推荐 |
|------|------|------|
| GPU 显存 | 4 GB | 8 GB+ |
| GPU 型号 | RTX 3050 Laptop | RTX 4070+ |
| RAM | 8 GB | 16 GB+ |
| 磁盘 | 5 GB（模型权重） | 10 GB+ |

> **CPU 可以吗？** 技术上可以但推理极慢（>10s/帧），不推荐。VLA 模型本质是大型 Transformer，GPU 加速是必需的。

### 3.2 安装依赖

```bash
# 方式 1: 安装 PyTorch（先根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 方式 2: 安装 LeRobot（包含 SmolVLA）
pip install lerobot
```

**安装验证**:

```bash
python -c "import lerobot; print(lerobot.__version__)"
```

### 3.3 LeRobot 是什么？

[LeRobot](https://github.com/huggingface/lerobot) 是 HuggingFace 开源的机器人学习框架，包含：

| 组件 | 说明 |
|------|------|
| **SmolVLA** | 450M VLA 模型（本文重点） |
| **ACT** | Action Chunking with Transformers |
| **Diffusion Policy** | 扩散策略 |
| **TD-MPC2** | 模型预测控制 |
| **数据集工具** | RLDS 格式转换、数据加载 |
| **训练/评估** | 统一的训练和评估 pipeline |

---

## 4. 第一次推理（5 分钟）

### 4.1 最简代码（5 行）

```python
import torch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 加载预训练模型（首次运行会自动下载，约 2GB）
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
model.eval()

# 准备观测（语言 + 图像 + 机器人状态）
observation = {
    "observation.images.front": torch.randn(1, 3, 256, 256),  # 前方摄像头
    "observation.images.left_wrist": torch.randn(1, 3, 256, 256),  # 左腕摄像头
    "observation.images.right_wrist": torch.randn(1, 3, 256, 256),  # 右腕摄像头
    "observation.state": torch.randn(1, 14),  # 14-DoF 双臂关节角
    "task": "pick up the apple",  # 语言指令
}

# 推理
with torch.no_grad():
    action = model.select_action(observation)

print(f"预测动作形状: {action.shape}")  # (1, 14) — 双臂各 7-DoF
print(f"动作值: {action}")
```

### 4.2 使用项目演示脚本

```bash
cd examples
python vla_demo.py --mode synthetic --task "pick up the apple"
```

### 4.3 使用真实数据推理

```bash
cd examples
python vla_demo.py --mode aloha --episode 0
```

---

## 5. 理解 Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    VLA Inference Pipeline                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │ Front   │  │ Left    │  │ Right   │   ← 3 个摄像头图像     │
│  │ Camera  │  │ Wrist   │  │ Wrist   │     (256×256×3)       │
│  │ (256²)  │  │ Camera  │  │ Camera  │                      │
│  └────┬────┘  └────┬────┘  └────┬────┘                      │
│       │            │            │                            │
│  ┌────▼────────────▼────────────▼────┐                      │
│  │        SmolVLM2 Vision Encoder     │                      │
│  │   (ViT backbone, 500M params)      │                      │
│  └─────────────┬─────────────────────┘                      │
│                │                                              │
│  ┌─────────────▼─────────────────────┐                      │
│  │     Language Token (task text)     │   ← "pick up apple"  │
│  └─────────────┬─────────────────────┘                      │
│                │                                              │
│  ┌─────────────▼─────────────────────┐                      │
│  │       Transformer Backbone         │                      │
│  │   (Cross-attention fusion)         │                      │
│  └─────────────┬─────────────────────┘                      │
│                │                                              │
│  ┌─────────────▼─────────────────────┐                      │
│  │       Flow Matching Head           │   ← 动作生成         │
│  │  (去噪 → 连续 6-DoF 动作)          │                      │
│  └─────────────┬─────────────────────┘                      │
│                │                                              │
│                ▼                                              │
│         Action (14-DoF)                                      │
│     [left_arm_7dof, right_arm_7dof]                          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 5.1 输入

| 输入 | 形状 | 说明 |
|------|------|------|
| `observation.images.front` | `(B, 3, 256, 256)` | 前方固定视角 |
| `observation.images.left_wrist` | `(B, 3, 256, 256)` | 左腕摄像头 |
| `observation.images.right_wrist` | `(B, 3, 256, 256)` | 右腕摄像头 |
| `observation.state` | `(B, 14)` | 双臂末端位姿或关节角 |
| `task` | `str` | 语言指令 |

### 5.2 输出

| 输出 | 形状 | 说明 |
|------|------|------|
| `action` | `(B, 14)` | 双臂各 7-DoF 动作增量 |

> SmolVLA 输出的是 **动作增量 (delta action)**，不是绝对位置。需要累加到当前状态：
> `next_state = current_state + action * action_scale`

---

## 6. SmolVLA 架构解析

### 6.1 整体架构

SmolVLA 基于 [SmolVLM2](https://huggingface.co/huggingface/SmolVLM2-500M-Video-Instruct)，修改了以下部分：

```
SmolVLM2 (原始)                    SmolVLA (修改)
─────────────                     ──────────────
图像编码 → 语言解码                图像编码 → Flow Matching 动作头
(输出文本 token)                   (输出连续动作向量)
```

### 6.2 关键组件

| 组件 | 基础模型 | 参数量 | 作用 |
|------|---------|--------|------|
| **Vision Encoder** | SmolVLM2 ViT | ~350M | 编码 3 路图像为 patch embedding |
| **Language Tokenizer** | SmolVLM2 | 嵌入层 | 将 task 文本转为 token |
| **Transformer** | SmolVLM2 | ~100M | 融合视觉和语言特征 |
| **Flow Matching Head** | 新增 | ~几M | 将特征映射为连续动作分布 |

### 6.3 Flow Matching vs Diffusion

SmolVLA 使用 **Flow Matching** 而非传统 Diffusion 生成动作：

| 方法 | 原理 | 采样步数 | 速度 |
|------|------|---------|------|
| **Diffusion** | 学习去噪过程 | 20-100 步 | 较慢 |
| **Flow Matching** | 学习向量场（ODE 积分） | **1-10 步** | **快 5-10x** |

```python
# Flow Matching 采样（简化）
# 1. 从标准正态分布采样
action_noise = torch.randn(batch_size, action_dim)

# 2. 沿学到的向量场积分 (Euler method)
for t in reversed(timesteps):
    velocity = model.predict_velocity(observation, action_noise, t)
    action_noise = action_noise - velocity * dt

# 3. 得到动作
action = action_noise
```

---

## 7. 动作空间与控制

### 7.1 ALOHA 双臂动作空间

SmolVLA 在 ALOHA 数据集上预训练，输出 14 维动作：

```
action[0:7]  = 左臂 7-DoF  (dx, dy, dz, droll, dpitch, dyaw, gripper)
action[7:14] = 右臂 7-DoF  (dx, dy, dz, droll, dpitch, dyaw, gripper)
```

> **注意**: 这是 **末端执行器增量**（delta end-effector pose），不是关节角。

### 7.2 控制循环

```python
# 典型 VLA 控制循环
import time

model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
model.eval()

current_state = get_robot_state()  # 14-DoF
control_freq = 10  # Hz

while not task_done:
    # 1. 获取观测
    images = get_camera_images()  # 3 × (256, 256, 3)
    state = normalize(current_state)

    # 2. 构建 observation
    observation = {
        "observation.images.front": to_tensor(images["front"]),
        "observation.images.left_wrist": to_tensor(images["left_wrist"]),
        "observation.images.right_wrist": to_tensor(images["right_wrist"]),
        "observation.state": to_tensor(state),
        "task": task_instruction,
    }

    # 3. VLA 推理
    with torch.no_grad():
        action = model.select_action(observation)

    # 4. 执行动作
    delta_action = action.cpu().numpy().flatten()
    current_state = current_state + delta_action * action_scale
    send_to_robot(current_state)

    # 5. 控制频率
    time.sleep(1.0 / control_freq)
```

### 7.3 Action Chunking

VLA 通常使用 **Action Chunking** 来提升平滑性。SmolVLAPolicy 内部使用 action queue 实现：

**核心机制**：
1. 第一次调用 `select_action()`：执行一次完整模型推理，生成 T 步 action chunk，存入内部 queue，并弹出第 1 步返回
2. 后续 T-1 次调用：从 queue 弹出动作返回，**不重新执行模型**
3. 第 T 次调用：返回 queue 中最后一个动作，queue 变空（仍不重新执行模型）
4. 第 T+1 次调用：queue 为空，重新执行模型，生成新的 action chunk
5. 每个 episode 开始时：调用 `policy.reset()` 清空 queue

```python
# SmolVLAPolicy 的 action queue 机制
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.reset()  # 每个 episode 开始时清空 queue

# 第 1 次调用: queue 为空 → 执行完整 forward，生成 chunk_size 步动作，弹出第 1 步返回
action_1 = policy.select_action(observation)  # ~GPU 推理

# 第 2 ~ chunk_size-1 次调用: queue 非空 → 直接弹出，不消耗 GPU
action_2 = policy.select_action(next_observation)  # ~0 ms

# 第 chunk_size 次调用: queue 中仅剩 1 个动作 → 弹出后 queue 变空（不重新推理）
action_T = policy.select_action(next_observation)  # ~0 ms

# 第 chunk_size+1 次调用: queue 为空 → 重新执行模型，生成新 chunk
action_T1 = policy.select_action(next_observation)  # ~GPU 推理
```

> **注意**：上述代码示例中的 `select_action()` 连续调用正是 action queue 的消费过程，不是"重复单步推理"。真正的 action chunking 发生在 policy 内部，而非通过外部循环多次调用模型。

---

## 8. 与机器人控制系统的衔接

VLA 模型生成的动作需要通过 Robot Adapter 与底层控制器对接，才能驱动机器人执行：

```
语言指令 → VLA → Action Chunk → Robot Adapter → Controller → 机器人执行
                                                              ↓
                                                          收集数据
                                                              ↓
图像 + 语言 → VLA → 动作 → 机器人执行（自主）
```

### 8.1 VLA 输出作为机器人控制输入

```python
# VLA 输出的 Action Chunk 通过 Robot Adapter 转换为控制指令
action_chunk = policy.select_action(observation)         # (chunk_size, action_dim)

# Robot Adapter 将 VLA 动作映射为机器人可执行的控制指令
robot_adapter = RobotAdapter(robot_type="aloha")
control_cmd = robot_adapter.to_command(action_chunk[0])  # 转为关节/末端指令

# 通过 Controller 下发到机器人
controller.send_command(control_cmd)

# 同一格式也可用于记录训练数据
episode = {
    "observation.images.front": images,           # (n_frames, 3, 256, 256)
    "observation.images.left_wrist": left_imgs,   # (n_frames, 3, 256, 256)
    "observation.images.right_wrist": right_imgs, # (n_frames, 3, 256, 256)
    "observation.state": states,                  # (n_frames, n_dofs)
    "action": actions,                            # (n_frames, n_dofs)
    "task": "pick up the cup",
}

# 保存为 LeRobot 格式
save_to_lerobot_dataset(episode, "output_dir/")
```

### 8.2 完整 pipeline

```
阶段 1: 专家演示数据收集（遥操作/脚本策略）
  专家演示 → 记录 (图像, 状态, 动作) → VLA 训练数据

阶段 2: VLA 训练（离线）
  数据集 → SmolVLA 微调 → 策略模型

阶段 3: 自主执行（机器人自主）
  语言指令 + 摄像头 → SmolVLA → Action Chunk → Robot Adapter → Controller → 机器人
```

---

## 9. 四大开源 VLA 对比

### 9.1 详细对比

| 维度 | SmolVLA | Octo | OpenVLA | RDT-1B |
|------|---------|------|---------|--------|
| **机构** | HuggingFace | UC Berkeley | Stanford/UT Austin | 清华 TSAIL |
| **参数量** | 450M | 27M / 93M | 7B | 170M / 1B |
| **视觉编码** | SmolVLM2 ViT | ViT-B | DINOv2 + SigLIP | SigLIP |
| **语言编码** | 内置 Tokenizer | T5 | Llama-2 | T5-XXL |
| **动作生成** | Flow Matching | Diffusion | 自回归 | Diffusion |
| **预训练数据** | ALOHA | 800K episodes | Open X-Embodiment | 1M+ episodes |
| **最小显存** | ~4 GB | <2 GB | ~15 GB | ~8 GB |
| **推理速度** | ~1.2s (Jetson) | 17 it/s (4090) | ~0.5s (4090) | 10 it/s (4090) |
| **支持双臂** | 是 (ALOHA) | 否 | 否 (OFT 支持) | **是** |
| **灵巧手** | 二指夹爪 | 二指夹爪 | 二指夹爪 | **ALOHA 灵巧** |
| **微调方式** | LoRA | LoRA | LoRA | 全量/LoRA |
| **边缘部署** | **Jetson Orin** | 否 | 否 | 否 |
| **许可证** | Apache-2.0 | MIT | Llama-2 | MIT |

### 9.2 选型指南

| 你的场景 | 推荐 | 理由 |
|---------|------|------|
| **学习/教学** | SmolVLA | 生态最好，pip 安装，文档丰富 |
| **资源极有限** | Octo-Small | 27M 参数，任意 GPU 可跑 |
| **通用性最强** | OpenVLA | 970K 轨迹预训练，社区最大 |
| **双臂灵巧操作** | RDT-1B | 专为双臂设计，ICLR 2025 |
| **边缘部署** | SmolVLA | 唯一支持 Jetson 的 VLA |
| **学术研究** | RDT-1B | 清华出品，论文详实 |

---

## 10. 进阶：微调与部署

### 10.1 在自定义数据上微调

```bash
# 使用 LeRobot 框架微调 SmolVLA
python lerobot/scripts/train.py \
    --dataset.repo_id=your_username/your_dataset \
    --policy.type=smolvla \
    --policy.pretrained_model_name_or_path=lerobot/smolvla_base \
    --training.num_epochs=100 \
    --evaluation.eval_freq=10 \
    --device=cuda
```

### 10.2 数据集格式

LeRobot 使用统一的 `LeRobotDataset` 格式：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载 ALOHA 数据集（自动下载）
dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")

print(f"Episodes: {dataset.num_episodes}")
print(f"Frames:   {dataset.num_frames}")
print(f"Features: {dataset.features}")

# 获取一个样本
sample = dataset[0]
print(f"Image shape: {sample['observation.images.front'].shape}")  # (3, 256, 256)
print(f"State shape: {sample['observation.state'].shape}")         # (14,)
print(f"Action shape: {sample['action'].shape}")                    # (14,)
```

### 10.3 Jetson 部署

```bash
# 在 Jetson Orin NX 上
pip install lerobot
```

> 实测数据：模型加载 ~45s，推理 1.2-1.8s/帧，显存 ~3.5GB。

---

## 11. 常见问题排查

### Q1: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决**:
```python
# 1. 使用 float16
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", torch_dtype=torch.float16)

# 2. 减小 batch size
# 3. 关闭其他 GPU 程序
```

### Q2: 模型下载很慢

**解决**: 使用 HuggingFace 镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
python vla_demo.py --mode synthetic
```

### Q3: 没有 GPU 怎么办？

**选项**:
1. 使用 Google Colab（免费 T4 GPU）
2. 使用 Kaggle Notebooks（免费 P100 GPU）
3. 仅学习理论，跳过实际推理

### Q4: 输出动作值不合理

**可能原因**:
- 图像不是 256×256
- state 维度不是 14
- 动作未正确反归一化

**解决**:
```python
# 确保输入归一化
state = (raw_state - state_mean) / state_std

# 确保输出反归一化
raw_action = action * action_std + action_mean
```

### Q5: 如何用于自己的机器人？

你需要：
1. 收集 50-200 条演示数据（遥操作或脚本策略）
2. 转换为 LeRobot 数据集格式
3. 微调 SmolVLA
4. 部署到机器人

详见 [LeRobot 官方教程](https://github.com/huggingface/lerobot)。

### Q6: SmolVLA 支持单臂吗？

SmolVLA 预训练模型是 ALOHA 双臂格式。单臂使用时：
- 将 `observation.state` 维度改为 7（或填充 0）
- `action` 输出中只使用其中 7 维

---

## 12. 参考文献

1. **SmolVLA**: HuggingFace LeRobot Team, "SmolVLA: A Compact Vision-Language-Action Model", 2025. [GitHub](https://github.com/huggingface/lerobot)
2. **OpenVLA**: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", RSS 2024. [GitHub](https://github.com/openvla/openvla)
3. **RDT-1B**: Chen et al., "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation", ICLR 2025. [GitHub](https://github.com/thu-ml/RoboticsDiffusionTransformer)
4. **Octo**: Ghosh et al., "Octo: An Open-Source Generalist Robot Policy", 2024. [GitHub](https://github.com/octo-models/octo)
5. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
6. **ALOHA**: Zhao et al., "ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation", 2024.
7. **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023.

---

## 附录：命令速查表

```bash
# === 环境安装 ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lerobot

# === 合成数据推理 ===
cd examples
python vla_demo.py --mode synthetic --task "pick up the apple"

# === 真实 ALOHA 数据推理 ===
python vla_demo.py --mode aloha --episode 0

# === 微调 ===
python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --policy.type=smolvla

# === 可视化动作序列 ===
python vla_demo.py --mode synthetic --visualize
```

---

> **本文目标达成**: 你现在理解了 VLA 的核心概念，能够在本地 GPU 上运行 SmolVLA 推理，并了解了如何将 VLA 输出通过 Robot Adapter 与机器人控制系统衔接。这就是 VLA 的 0→1。

## 毕业验收

完成以下所有项目即算 VLA 0→1 毕业：

- [ ] 运行 `minimal_vla.py`（理解 VLA 输入输出结构）
- [ ] 运行 `vla_01_toy_training.py`（观察 loss 从随机下降到收敛）
- [ ] 运行 `vla_demo.py --mode synthetic`（理解真实 API 调用流程）
- [ ] 能解释：图像编码器、语言编码器、融合层、动作头的作用
- [ ] 能解释 Action Chunking 与单步推理的区别
- [ ] （可选）运行 `vla_demo.py --mode aloha`（需要 GPU）