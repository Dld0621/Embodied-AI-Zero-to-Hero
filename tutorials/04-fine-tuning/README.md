# Stage 4: 微调实践

> 在 LIBERO 仿真基准或自定义数据上微调 OpenVLA 模型，从训练到评估的完整流程。

---

## 概述

本目录包含完整的、可直接运行的微调脚本，用于在 LIBERO 仿真基准上微调 OpenVLA-7B 模型。所有脚本基于 OpenVLA 官方代码，但做了大量简化和教学化处理，每个步骤都有详细的中文注释。

### 文件说明

| 文件 | 用途 | 说明 |
|------|------|------|
| `dataset_utils.py` | 数据工具函数库 | 数据集类、动作归一化、图像预处理等通用工具 |
| `finetune_libero.py` | LIBERO 微调脚本 | 在 LIBERO benchmark 上 LoRA 微调 OpenVLA |
| `evaluate_libero.py` | LIBERO 评估脚本 | 在仿真环境中闭环评估微调后的模型 |
| `train_custom_data.py` | 自定义数据微调 | 使用自己的 JSONL 数据微调 VLA |
| `README.md` | 本文档 | 使用说明和常见问题 |

---

## 环境准备

### 1. 基础依赖

```bash
# 创建 conda 环境
conda create -n vla-ft python=3.10 -y
conda activate vla-ft

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 HuggingFace 和 PEFT
pip install transformers peft accelerate

# 安装图像处理
pip install pillow numpy imageio

# 安装量化支持（可选，省显存用）
pip install bitsandbytes
```

### 2. LIBERO 环境（如果使用 LIBERO benchmark）

```bash
pip install libero
```

### 3. RLDS 数据支持（如果使用 HuggingFace RLDS 数据）

```bash
pip install tensorflow tensorflow_datasets rlds
```

### 4. WandB 日志（可选）

```bash
pip install wandb
wandb login  # 首次使用需要登录
```

### 5. 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU 显存 | 24 GB（省显存模式） | 48 GB+ |
| GPU 型号 | RTX 3090 / 4090 | A6000 / A100 |
| 系统内存 | 32 GB | 64 GB |
| 磁盘空间 | 30 GB（模型 + 数据） | 100 GB |

---

## 快速开始：LIBERO 微调到评估

### 第一步：准备数据

有两种数据加载方式：

**方式 A：本地 LIBERO 安装**

```bash
# 安装 LIBERO
pip install libero

# 下载数据（首次运行时会自动下载）
python -c "from libero.libero import benchmark; b = benchmark.get_benchmark_dict()['libero_spatial'](); print(f'已加载 {b.n_tasks} 个任务')"
```

**方式 B：HuggingFace RLDS 数据**

从 [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds) 下载数据。

### 第二步：开始微调

```bash
python finetune_libero.py \
    --vla_path openvla/openvla-7b \
    --data_root ~/.cache/libero \
    --benchmark libero_spatial \
    --output_dir ./checkpoints/openvla-libero-spatial \
    --batch_size 4 \
    --max_steps 100000 \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --save_steps 10000 \
    --log_interval 50
```

**24GB 显存的省显存配置：**

```bash
python finetune_libero.py \
    --vla_path openvla/openvla-7b \
    --data_root ~/.cache/libero \
    --benchmark libero_spatial \
    --output_dir ./checkpoints/openvla-libero-spatial \
    --batch_size 2 \
    --grad_accumulation_steps 4 \
    --max_steps 100000 \
    --learning_rate 5e-4 \
    --use_gradient_checkpointing \
    --save_steps 10000
```

### 第三步：评估

```bash
python evaluate_libero.py \
    --checkpoint_path ./checkpoints/openvla-libero-spatial/checkpoint-final \
    --benchmark libero_spatial \
    --num_trials_per_task 20 \
    --save_videos \
    --video_dir ./rollouts
```

---

## 详细用法

### 1. finetune_libero.py -- LIBERO 微调脚本

完整的 LIBERO 微调脚本，基于 OpenVLA 官方 `vla-scripts/finetune.py` 流程。

#### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vla_path` | `openvla/openvla-7b` | 预训练模型路径或 HuggingFace ID |
| `--data_root` | (必填) | 数据根目录 |
| `--benchmark` | `libero_spatial` | LIBERO benchmark 名称 |
| `--output_dir` | `./runs` | checkpoint 保存目录 |
| `--batch_size` | 4 | 批次大小 |
| `--max_steps` | 200000 | 最大训练步数 |
| `--learning_rate` | 5e-4 | 学习率 |
| `--lora_rank` | 32 | LoRA 低秩维度 |
| `--lora_alpha` | 32 | LoRA 缩放因子 |
| `--lora_dropout` | 0.1 | LoRA dropout |
| `--chunk_size` | 1 | Action Chunking 大小 |
| `--image_aug` | True | 是否使用图像增强 |
| `--use_gradient_checkpointing` | False | 梯度检查点（省显存） |
| `--use_wandb` | False | 使用 WandB 日志 |

#### 支持的 Benchmark

| Benchmark | 任务数 | 难度 | 推荐起始 |
|-----------|--------|------|----------|
| `libero_spatial` | 10 | 简单 | 是 |
| `libero_object` | 10 | 中等 | |
| `libero_goal` | 10 | 中等 | |
| `libero_10` | 10 | 困难 | |

#### 训练流程解析

```
数据加载 → 动作归一化 → 图像预处理 → 模型加载 → LoRA 配置
    ↓
训练循环（前向 → 反向 → 优化）
    ↓
定期保存 checkpoint + 记录日志
    ↓
训练结束，保存最终模型 + 数据集统计量
```

**数据集统计量（dataset_statistics.json）的作用：**

微调后，模型的输出是归一化的动作值。推理时需要知道训练数据的均值和标准差，才能将输出还原到真实的动作空间。这个 JSON 文件就是记录这些统计量的，评估脚本会自动读取。

#### 使用 RLDS 数据

```bash
python finetune_libero.py \
    --vla_path openvla/openvla-7b \
    --data_root ./datasets/modified_libero_rlds \
    --benchmark libero_spatial \
    --use_rlds \
    --output_dir ./checkpoints/openvla-libero-spatial
```

---

### 2. evaluate_libero.py -- LIBERO 评估脚本

独立于训练的评估脚本，加载任意 checkpoint 在仿真环境中闭环评估。

#### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_path` | (必填) | checkpoint 目录路径 |
| `--task_suite_name` | `libero_spatial` | 评估的 benchmark |
| `--num_trials_per_task` | 20 | 每个 task 的 episode 数 |
| `--center_crop` | True | 是否中心裁剪（与训练时 image_aug 对应） |
| `--save_videos` | False | 是否保存回放视频 |
| `--load_in_4bit` | False | 4-bit 量化加载（省显存） |

#### 输出

- 控制台实时打印每个 episode 和每个 task 的成功率
- 保存 JSON 格式的完整评估结果
- 可选：保存每个 episode 的回放视频

#### 评估注意事项

1. **center_crop 参数**：如果训练时使用了 `--image_aug`（random_crop），评估时必须使用 `--center_crop`。如果不一致，模型看到的图像分布与训练时不同，成功率会显著下降。

2. **unnorm_key**：脚本会自动从 checkpoint 目录中的 `dataset_statistics.json` 读取 unnorm_key。确保微调时保存了该文件。

3. **num_trials_per_task**：20 个 episode 通常能得到可靠的统计结果。50 个更稳定，但耗时更长。

4. **初始等待步数**：默认前 10 步执行空操作，等待仿真中物体稳定。可以通过 `--num_steps_wait` 调整。

---

### 3. train_custom_data.py -- 自定义数据微调

使用自己收集的机器人数据进行微调。

#### 数据格式

每行一个 JSON 对象：

```json
{"image_path": "images/ep001_step005.jpg", "instruction": "pick up the red cup", "action": [0.01, -0.02, 0.005, 0.0, 0.0, 0.01, 1.0]}
{"image_path": "images/ep001_step006.jpg", "instruction": "pick up the red cup", "action": [0.02, -0.01, 0.003, 0.0, 0.0, 0.02, 1.0]}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_path` | string | 图像路径（相对或绝对） |
| `instruction` | string | 自然语言任务指令 |
| `action` | array[float] | 7 维动作 [dx, dy, dz, droll, dpitch, dyaw, gripper] |

#### 用法

```bash
# 基本用法
python train_custom_data.py \
    --vla_path openvla/openvla-7b \
    --jsonl_path ./data/my_data.jsonl \
    --image_root ./data \
    --output_dir ./checkpoints/my-vla

# 带验证集
python train_custom_data.py \
    --vla_path openvla/openvla-7b \
    --jsonl_path ./data/train.jsonl \
    --val_jsonl_path ./data/val.jsonl \
    --image_root ./data \
    --output_dir ./checkpoints/my-vla \
    --val_interval 500

# 使用 WandB
python train_custom_data.py \
    --vla_path openvla/openvla-7b \
    --jsonl_path ./data/train.jsonl \
    --image_root ./data \
    --output_dir ./checkpoints/my-vla \
    --use_wandb \
    --wandb_project my-vla-project
```

#### 准备数据的建议

1. **数据量**：至少 100-500 条样本。1000+ 条样本通常能得到更好的效果。
2. **多样性**：尽量覆盖不同的任务、场景、物体位置。
3. **图像质量**：确保图像清晰、光照正常。OpenVLA 预训练使用 224x224 的图像。
4. **动作范围**：确保动作值在合理的物理范围内。

---

### 4. dataset_utils.py -- 数据工具函数

被其他脚本 import 的通用工具库，一般不直接运行。

可以直接运行来验证工具函数是否正常工作：

```bash
python dataset_utils.py
```

---

## 常见问题

### Q1: CUDA Out of Memory

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：
1. 减小 batch_size：`--batch_size 2` 或 `--batch_size 1`
2. 启用梯度检查点：`--use_gradient_checkpointing`
3. 使用梯度累积弥补小 batch_size：`--grad_accumulation_steps 4`
4. 使用 4-bit 量化：`--load_in_4bit`（仅评估时推荐）
5. 关闭图像增强可能略微省显存：`--no_image_aug`（但可能降低效果）

### Q2: 模型下载失败

**症状**：`ConnectionError` 或 `OSError` 无法下载模型

**解决方案**：
1. 设置 HuggingFace 镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. 手动下载模型到本地，然后使用本地路径：
   ```bash
   python finetune_libero.py --vla_path /path/to/local/openvla-7b ...
   ```

### Q3: 找不到 prismatic 模块

**症状**：`ModuleNotFoundError: No module named 'prismatic'`

**解决方案**：
```bash
pip install prismatic-vla
# 或者从源码安装
pip install git+https://github.com/openvla/openvla.git
```

如果从 HuggingFace Hub 加载模型（路径为 `openvla/openvla-7b`），prismatic 不是必须的依赖。但如果从本地 checkpoint 加载，则需要安装。

### Q4: 训练 loss 不下降

**可能原因和解决方案**：
1. **数据归一化问题**：检查 dataset_statistics.json 是否正确生成
2. **学习率太大/太小**：尝试 1e-4 或 1e-3
3. **LoRA 配置问题**：运行时查看 `print_trainable_parameters()` 输出
4. **数据量不足**：增加训练样本
5. **prompt 格式错误**：确认 prompt 模板与预训练时一致

### Q5: 评估时成功率很低

**可能原因和解决方案**：
1. **center_crop 不匹配**：训练时用了 image_aug，评估时必须用 `--center_crop`
2. **unnorm_key 不正确**：检查 checkpoint 的 dataset_statistics.json
3. **训练不充分**：增加训练步数
4. **action chunking**：尝试增大 chunk_size（需要重新训练）
5. **图像翻转**：评估脚本已处理了 180 度翻转，确认你的数据也正确

### Q6: ImportError: cannot import name 'OffScreenRenderEnv'

**症状**：导入 libero 环境失败

**解决方案**：
```bash
pip install libero --upgrade
# 或者重新安装
pip uninstall libero
pip install libero
```

### Q7: 训练速度太慢

**优化方案**：
1. 确保 GPU 驱动和 CUDA 版本匹配
2. 安装 flash-attention-2：
   ```bash
   pip install flash-attn --no-build-isolation
   ```
3. 增加 batch_size（如果显存允许）
4. 减少数据增强的复杂度

### Q8: 如何从 LoRA checkpoint 恢复训练？

LoRA adapter 保存的是增量权重，不是完整模型。要恢复训练，需要同时有基础模型和 adapter：

```bash
# 方式 1：使用 PEFT 库加载（推荐）
# 修改 finetune_libero.py 中的模型加载逻辑，使用 PeftModel.from_pretrained()

# 方式 2：先合并 LoRA 权重到基础模型，再作为新 checkpoint 加载
from peft import PeftModel
base_model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", ...)
model = PeftModel.from_pretrained(base_model, "./checkpoint-dir")
model = model.merge_and_unload()
model.save_pretrained("./merged-checkpoint")
```

---

## 微调原理简述

### LoRA 是什么？

LoRA（Low-Rank Adaptation）是一种参数高效微调方法。核心思想：冻结预训练模型的原始权重，只训练少量新增的低秩矩阵。

```
原始：    y = W * x           （W 是 d x d 的矩阵）
LoRA：    y = W * x + B * A * x  （A 是 d x r, B 是 r x d，r << d）
```

对于 OpenVLA-7B（7B 参数），LoRA rank=32 时：
- 可训练参数：约 7M（0.1%）
- 显存占用：约 20-24 GB（单卡 RTX 3090/4090 可训练）

### 为什么用 cosine schedule + warmup？

```
学习率
  ^
  |    /‾‾‾‾\
  |   /      \
  |  /        \___
  | /warmup    cosine decay
  +---------------------------> 训练步数
```

- Warmup 阶段：从 0 线性增加到目标学习率，避免初始阶段的大梯度
- Cosine decay：平滑衰减到接近 0，训练后期精细调优

### Action Chunking

让模型一次预测多步连续动作（而非单步）：
- 减少推理时的累积误差
- 提高长期任务的连贯性
- 需要与训练数据中的 chunk_size 对应

---

## 延伸阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685) -- LoRA 原始论文
- [OpenVLA GitHub](https://github.com/openvla/openvla) -- 官方代码仓库
- [OpenVLA 微调文档](https://github.com/openvla/openvla?tab=readme-ov-file#fine-tuning) -- 官方微调说明
- [LIBERO 基准](https://github.com/Lifelong-Robot-Learning/LIBERO) -- LIBERO 仿真环境
- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft) -- PEFT 库使用指南
- [Action Chunking with Transformers](https://arxiv.org/abs/2304.13705) -- ACT 论文

---

## 验证检查点

- [ ] 能成功加载数据集（本地 LIBERO 或 RLDS）
- [ ] 能加载 OpenVLA 模型并配置 LoRA
- [ ] 训练 loss 能持续下降
- [ ] 能保存 checkpoint 和 dataset_statistics.json
- [ ] 评估脚本能在仿真环境中运行
- [ ] 评估成功率 > 50%（libero_spatial 基准）
