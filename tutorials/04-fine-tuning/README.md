# Stage 4: 微调实践

> 在 LIBERO 仿真基准或自定义数据上微调 OpenVLA 模型，从训练到评估的完整流程。

---

## 概述

本目录提供微调与评估的教学实现，不是已经完成端到端复现的发行包。本次内容审查只核对文档、参数与部分 API，未下载模型/数据、训练或运行 LIBERO 闭环；数据适配、动作编码、归一化统计、checkpoint 装载仍须逐项验证。需要复现论文结果时，以 [OpenVLA 官方安装与训练流程](https://github.com/openvla/openvla) 为基准，不要把教学简化等同于官方实现。

下面命令是配置模板：先核对版本、实际数据根目录及各脚本 `--help`，再在独立环境尝试单批前向检查。不能凭命令列在这里就认定训练、评估或硬件控制已经通过。

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

不要在现有工作环境直接升级所有包。先采用官方测试过的依赖组合并记录锁定版本；下列包清单用于说明依赖类别，不保证任意最新版相互兼容。

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

按 [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO#installation) 克隆源码并在其目录内安装，记录 commit；不要假设同名 PyPI 包等价。

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

完成官方要求的其他依赖后，回到本仓库 `tutorials/04-fine-tuning/` 再使用本页训练脚本。

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

以下仅是资源预算示例，非已测试最低要求；显存随分辨率、序列长度、目标层、优化器和 batch size 变化。

| 配置 | 预算示例 | 较宽裕的预算示例 |
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
# 在已安装的 LIBERO 源码根目录执行，数据下载是独立步骤
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
```

`benchmark.get_benchmark_dict()["libero_spatial"]()` 只创建任务集合，不会自动下载 demonstration 数据。记录真实下载目录，再确认本地教学加载器支持该 HDF5 布局；任务可枚举不代表训练数据已可读。[官方数据下载说明](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets)

**方式 B：HuggingFace RLDS 数据**

从 [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds) 下载数据。

RLDS 与原始 LIBERO HDF5 不是同一格式，不能只替换目录。训练前验证一批图像、指令与动作的对应关系；下面的 `--data_root ~/.cache/libero` 是占位配置，请改成实际支持的目录。

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
    --task_suite_name libero_spatial \
    --num_trials_per_task 20 \
    --save_videos \
    --video_dir ./rollouts
```

---

## 详细用法

### 1. finetune_libero.py -- LIBERO 微调脚本

教学版 LIBERO 微调入口；接口与官方 `vla-scripts/finetune.py` 不保证等价，须先做数据和损失合同检查。

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

统计量必须与实际训练的动作编码一一匹配。基础 OpenVLA 的 `predict_action` 使用 `q01` / `q99` 与可选 `mask`，并已在 API 内反归一化；它不是通用的 `action * std + mean`。本地教学数据工具的归一化方案须与推理适配共同核验，仅保存 JSON 或文件名相同不能证明兼容。[官方实现](https://huggingface.co/openvla/openvla-7b/blob/main/modeling_prismatic.py)

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

独立于训练的教学评估入口；只应加载结构、适配器和统计量均兼容的 checkpoint，不能加载任意权重就认为可评估。

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

1. **center_crop 参数**：采用与该检查点官方评估协议相符的图像处理，记录裁剪、翻转与缩放。分布偏移可能影响结果，不能笼统断言某个裁剪设置对所有检查点都正确。

2. **unnorm_key**：脚本会自动从 checkpoint 目录中的 `dataset_statistics.json` 读取 unnorm_key。确保微调时保存了该文件。

3. **num_trials_per_task**：20 个 episode 只意味着有限样本，不能自动称作可靠。报告成功数/总数、任务、初始状态、随机种子及置信区间；任务之间的相关性也应说明。

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

1. **数据量**：没有通用的“100–500 条即可”门槛。区分帧数、episode 数和任务覆盖，按 episode 划分训练/验证，测量增加数据后的学习曲线。
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
1. 检查官方 Hub 的连接、访问权限与本地缓存；不要为排错随意切换到未审查的第三方镜像或向其发送令牌。
2. 从核实的官方来源下载并固定 revision，再使用本地路径：
   ```bash
   python finetune_libero.py --vla_path /path/to/local/openvla-7b ...
   ```

### Q3: 找不到 prismatic 模块

**症状**：`ModuleNotFoundError: No module named 'prismatic'`

**解决方案**：
```bash
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
```

是否需要 `prismatic` 取决于脚本的导入和模型实现，不是“本地路径”本身决定的。Hub 的 `trust_remote_code` 实现与直接导入源码是不同加载路线；按官方安装说明选择其中一致的一条。

### Q4: 训练 loss 不下降

**可能原因和解决方案**：
1. **数据归一化问题**：检查 dataset_statistics.json 是否正确生成
2. **学习率太大/太小**：尝试 1e-4 或 1e-3
3. **LoRA 配置问题**：运行时查看 `print_trainable_parameters()` 输出
4. **数据量不足**：增加训练样本
5. **prompt 格式错误**：确认 prompt 模板与预训练时一致

### Q5: 评估时成功率很低

**可能原因和解决方案**：
1. **预处理不匹配**：核对检查点对应的裁剪、翻转、尺寸与 prompt 协议
2. **unnorm_key 不正确**：检查 checkpoint 的 dataset_statistics.json
3. **训练不充分**：增加训练步数
4. **action chunking**：基础单动作头不能仅靠改参数获得分块能力；先确认模型头、标签与评估都支持同一 chunk 合同
5. **图像翻转**：评估脚本已处理了 180 度翻转，确认你的数据也正确

### Q6: ImportError: cannot import name 'OffScreenRenderEnv'

**症状**：导入 libero 环境失败

**解决方案**：确认导入的是官方 LIBERO 源码安装，核对其 commit、依赖和 `OffScreenRenderEnv` 的导入路径；回到本页的源码安装步骤。不要盲目卸载/升级同名包来掩盖环境冲突。

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

LoRA adapter 保存的是增量权重，不是完整模型。装载基础模型与 adapter 可以从权重继续学习，但若要精确恢复中断训练，还需优化器、调度器、步数与随机状态；合并权重不等于恢复这些状态。

```python
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
LoRA：    y = W * x + (alpha/r) * B * A * x
          A 是 r x d，B 是 d x r，x 是 d 维列向量，r << d
```

一个 d×d 线性层新增 2dr 个低秩参数；矩形层则是 r(d_in+d_out)。总数取决于实际目标层与是否还有可训练头，不由 rank 单独决定。用 `print_trainable_parameters()` 和实测峰值显存记录本次配置，不把示例预算当保证。[LoRA 论文](https://arxiv.org/abs/2106.09685)

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
- 可以减少重规划次数，但较长开环执行也可能累积误差
- 连贯性和成功率需要在同协议下实验验证
- 需要动作头、训练标签与执行端一起支持 chunk_size；基础 OpenVLA 单动作输出不能直接当动作序列

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
- [ ] 报告指定任务、成功数/总数、种子、置信区间与失败案例；与同协议基线比较，不用任意 50% 门槛替代学习证据
