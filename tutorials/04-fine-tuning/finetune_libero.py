#!/usr/bin/env python3
"""
finetune_libero.py -- 在 LIBERO 仿真基准上 LoRA 微调 OpenVLA

================================================================================
  Embodied-AI-Zero-to-Hero 教学项目 · Stage 4: 微调实践
================================================================================

本脚本基于 OpenVLA 官方 vla-scripts/finetune.py 的流程，但做了大幅简化和教学化处理。
目标是让学习者理解微调的每个步骤，而不是简单封装。

功能：
  - 支持 LoRA 微调 OpenVLA-7B 在 LIBERO 数据集上
  - 使用 HuggingFace transformers + PEFT 库
  - 支持 LIBERO-Spatial / LIBERO-Object / LIBERO-Goal / LIBERO-10 四个 benchmark
  - 支持 Action Chunking（默认 chunk_size=1）
  - 完整的训练循环：数据加载 → 模型加载 → LoRA 配置 → 训练 → 保存 checkpoint
  - 支持 WandB 日志（可选）
  - 命令行参数控制所有超参数
  - 假设学习者可能只有单卡 GPU（24GB），提供省显存选项

依赖：
  pip install torch transformers peft accelerate pillow numpy
  pip install wandb          # 可选，用于训练日志
  pip install libero          # 使用本地 LIBERO 数据
  pip install tensorflow tensorflow_datasets  # 使用 RLDS 格式数据

用法示例：
  # 基本用法：在 LIBERO-Spatial 上微调
  python finetune_libero.py \\
      --vla_path openvla/openvla-7b \\
      --benchmark libero_spatial \\
      --data_root ./datasets/libero \\
      --output_dir ./checkpoints/openvla-libero-spatial

  # 省显存模式（24GB GPU）
  python finetune_libero.py \\
      --vla_path openvla/openvla-7b \\
      --benchmark libero_spatial \\
      --data_root ./datasets/libero \\
      --output_dir ./checkpoints/openvla-libero-spatial \\
      --batch_size 2 \\
      --use_gradient_checkpointing \\
      --max_steps 10000

  # 使用 RLDS 格式数据
  python finetune_libero.py \\
      --vla_path openvla/openvla-7b \\
      --benchmark libero_spatial \\
      --data_root ./datasets/modified_libero_rlds \\
      --use_rlds \\
      --output_dir ./checkpoints/openvla-libero-spatial

  # 使用 WandB 日志
  python finetune_libero.py \\
      --vla_path openvla/openvla-7b \\
      --benchmark libero_spatial \\
      --data_root ./datasets/libero \\
      --output_dir ./checkpoints/openvla-libero-spatial \\
      --use_wandb \\
      --wandb_project vla-finetuning

================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# 禁用 tokenizer 的并行化（避免 warning）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入同目录下的数据工具
from dataset_utils import (
    LIBERODataset,
    LIBEROCollator,
    ACTION_DIM,
    create_vla_dataloader,
    get_action_stats,
    get_libero_dataset,
    load_dataset_statistics,
    normalize_action,
    save_dataset_statistics,
)

# ============================================================
# 全局常量
# ============================================================

# 默认的超参数，参考 OpenVLA 官方配置
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_STEPS = 200_000
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32  # 通常设为与 rank 相同
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_SAVE_STEPS = 5000
DEFAULT_WARMUP_RATIO = 0.1  # warmup 占总训练步数的比例

# LoRA 的目标模块
# 这些是 transformer 注意力层中的投影矩阵。
# 只微调这些层，就能以极小的参数量（约 0.1%）获得显著的性能提升。
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    每个参数都附带了详细说明，帮助学习者理解每个选项的含义。
    """
    parser = argparse.ArgumentParser(
        description="在 LIBERO 仿真基准上 LoRA 微调 OpenVLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 模型相关 ----
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--vla_path", type=str, default="openvla/openvla-7b",
        help="预训练 VLA 模型的路径或 HuggingFace ID。"
             "可以是本地路径或 HuggingFace Hub 上的模型名称。",
    )
    model_group.add_argument(
        "--use_lora", action="store_true", default=True,
        help="是否使用 LoRA 微调（推荐，大幅减少显存占用）。",
    )
    model_group.add_argument(
        "--lora_rank", type=int, default=DEFAULT_LORA_RANK,
        help="LoRA 低秩矩阵的维度 r。"
             "r 越大，可学习的参数越多，表达能力越强，但显存占用也越大。"
             "推荐范围：16-64。",
    )
    model_group.add_argument(
        "--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA,
        help="LoRA 的缩放因子 alpha。实际缩放比例 = alpha / r。"
             "通常设为与 rank 相同或 2 倍。",
    )
    model_group.add_argument(
        "--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT,
        help="LoRA 层的 dropout 概率，用于防止过拟合。",
    )
    model_group.add_argument(
        "--use_4bit", action="store_true", default=False,
        help="是否使用 4-bit 量化加载模型。"
             "可以进一步减少显存，但会降低性能。"
             "24GB 显存不需要此选项。",
    )

    # ---- 数据相关 ----
    data_group = parser.add_argument_group("数据配置")
    data_group.add_argument(
        "--data_root", type=str, required=True,
        help="数据根目录路径。"
             "本地 LIBERO：指向 LIBERO 数据目录（如 ~/.cache/libero）。"
             "RLDS 格式：指向 RLDS 数据集根目录。",
    )
    data_group.add_argument(
        "--benchmark", type=str, default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO benchmark 名称。",
    )
    data_group.add_argument(
        "--use_rlds", action="store_true", default=False,
        help="使用 RLDS 格式数据（需要 tensorflow_datasets）。"
             "默认使用本地 LIBERO 的 HDF5 格式。",
    )
    data_group.add_argument(
        "--chunk_size", type=int, default=1,
        help="Action Chunking 大小。让模型一次预测多步连续动作。"
             "默认为 1（不 chunk）。"
             "Action Chunking 可以减少推理时的累积误差，提高长期任务的连贯性。",
    )
    data_group.add_argument(
        "--image_aug", action="store_true", default=True,
        help="训练时是否使用图像增强（随机裁剪、颜色抖动等）。"
             "推荐开启，可以有效防止过拟合。",
    )
    data_group.add_argument(
        "--max_transitions", type=int, default=None,
        help="最大加载的 transition 数量。用于快速调试和测试。"
             "正式训练时设为 None（加载全部数据）。",
    )

    # ---- 训练相关 ----
    train_group = parser.add_argument_group("训练配置")
    train_group.add_argument(
        "--output_dir", type=str, default="./runs",
        help="模型 checkpoint 和日志的保存目录。",
    )
    train_group.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="训练批次大小。24GB 显存建议 2-4，48GB 建议 8-12。",
    )
    train_group.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help="最大训练步数。OpenVLA 官方使用 200k 步。"
             "快速测试可以设为 1000。",
    )
    train_group.add_argument(
        "--save_steps", type=int, default=DEFAULT_SAVE_STEPS,
        help="每隔多少步保存一次 checkpoint。",
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="学习率。LoRA 微调可以使用较大的学习率（1e-4 ~ 5e-4）。",
    )
    train_group.add_argument(
        "--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO,
        help="Warmup 占总训练步数的比例。"
             "warmup 期间学习率从 0 线性增加到目标值，有助于训练稳定。",
    )
    train_group.add_argument(
        "--grad_accumulation_steps", type=int, default=1,
        help="梯度累积步数。相当于增大了有效的 batch_size。"
             "例如 batch_size=2, grad_accum=4 → 有效 batch_size=8。"
             "在显存有限时很有用。",
    )
    train_group.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="权重衰减（L2 正则化），防止过拟合。",
    )
    train_group.add_argument(
        "--use_gradient_checkpointing", action="store_true", default=False,
        help="启用梯度检查点。用计算时间换取显存空间。"
             "24GB 显存强烈建议开启。",
    )
    train_group.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，用于结果可复现。",
    )

    # ---- 日志相关 ----
    log_group = parser.add_argument_group("日志配置")
    log_group.add_argument(
        "--use_wandb", action="store_true", default=False,
        help="是否使用 Weights & Biases 记录训练日志。",
    )
    log_group.add_argument(
        "--wandb_project", type=str, default="vla-finetuning",
        help="WandB 项目名称。",
    )
    log_group.add_argument(
        "--wandb_entity", type=str, default=None,
        help="WandB 用户/团队名。",
    )
    log_group.add_argument(
        "--log_interval", type=int, default=10,
        help="每隔多少步打印一次日志。",
    )

    # ---- 分布式训练 ----
    dist_group = parser.add_argument_group("分布式训练")
    dist_group.add_argument(
        "--local_rank", type=int, default=-1,
        help="分布式训练的 local rank（由 torchrun 自动设置）。",
    )

    return parser.parse_args()


# ============================================================
# 模型加载和 LoRA 配置
# ============================================================

def register_openvla_classes() -> None:
    """
    注册 OpenVLA 的自定义类到 HuggingFace Auto Classes。

    为什么需要注册？
    OpenVLA 使用了自定义的 config、processor 和 model 类，
    但 HuggingFace 默认不知道这些类的存在。
    通过 register() 告诉 HuggingFace 如何自动加载这些类。

    如果模型已经在 HuggingFace Hub 上注册了，则不需要手动注册。
    但从本地 checkpoint 加载时必须先注册。
    """
    try:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ImportError as e:
        print(f"警告：无法导入 OpenVLA 自定义类。如果从 HuggingFace Hub 加载模型，"
              f"这不影响使用。错误详情：{e}")


def load_model_and_processor(
    vla_path: str,
    device: torch.device,
    use_4bit: bool = False,
    use_gradient_checkpointing: bool = False,
) -> Tuple[Any, Any]:
    """
    加载 OpenVLA 模型和 Processor。

    参数：
        vla_path: 模型路径或 HuggingFace ID
        device: 目标设备
        use_4bit: 是否使用 4-bit 量化
        use_gradient_checkpointing: 是否启用梯度检查点

    返回：
        (model, processor) 元组
    """
    # 注册自定义类
    register_openvla_classes()

    print(f"[模型加载] 正在从 '{vla_path}' 加载模型...")

    # 加载 Processor
    # Processor 负责：图像预处理、文本 tokenization、两者对齐
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    print(f"  Processor 加载完成")

    # 配置量化（可选）
    quantization_config = None
    if use_4bit:
        print(f"  使用 4-bit NF4 量化加载模型（省显存，但会降低精度）")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    # 加载模型
    # torch_dtype=torch.bfloat16: 使用半精度浮点数，减少显存占用
    # low_cpu_mem_usage=True: 加载时尽量减少 CPU 内存使用
    # trust_remote_code=True: 允许执行模型代码中的自定义 Python 代码
    model = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 4-bit 量化模型需要特殊处理
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device)

    # 梯度检查点：用时间换空间
    # 原理：前向传播时不保存中间激活值，反向传播时重新计算
    # 效果：可以减少约 30-50% 的显存占用，但训练速度降低约 20-30%
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"  梯度检查点已启用（省显存模式）")

    print(f"  模型加载完成，参数量：{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, processor


def setup_lora(model: Any, rank: int, alpha: int, dropout: float) -> Any:
    """
    为模型配置 LoRA（Low-Rank Adaptation）。

    LoRA 的核心思想：
    不直接修改原始权重矩阵 W，而是学习两个小矩阵 A 和 B，使得：
        W' = W + A * B
    其中 A 是 (d, r)，B 是 (r, d)，r << d。
    这样可训练参数从 O(d^2) 减少到 O(2*d*r)。

    对于 7B 模型，LoRA rank=32 时，可训练参数约 7M（占总参数的 0.1%）。

    参数：
        model: 基础模型
        rank: LoRA 低秩维度
        alpha: LoRA 缩放因子
        dropout: LoRA dropout

    返回：
        包装了 LoRA 的模型
    """
    print(f"[LoRA 配置] rank={rank}, alpha={alpha}, dropout={dropout}")
    print(f"  目标模块: {LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",          # 不训练偏置项
        init_lora_weights="gaussian",  # 使用高斯分布初始化 LoRA 权重
    )

    # 将 LoRA adapter 应用到模型上
    model = get_peft_model(model, lora_config)

    # 打印可训练参数信息
    model.print_trainable_parameters()

    return model


# ============================================================
# 学习率调度器
# ============================================================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Any:
    """
    创建学习率调度器。

    使用 cosine schedule with linear warmup：
    1. Warmup 阶段：学习率从 0 线性增加到 peak LR
    2. 训练阶段：学习率按余弦曲线衰减到接近 0

    为什么用 cosine schedule？
    - 前期大学习率快速收敛
    - 后期小学习率精细调整
    - 比 constant 或 step decay 更平滑

    参数：
        optimizer: 优化器
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数

    返回：
        学习率调度器
    """
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


# ============================================================
# 训练主循环
# ============================================================

def train(
    model: Any,
    processor: Any,
    dataloader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int = -1,
) -> None:
    """
    主训练循环。

    训练流程：
    1. 准备优化器和调度器
    2. 遍历数据，执行前向 → 反向 → 优化
    3. 定期保存 checkpoint
    4. 记录训练指标

    参数：
        model: 已配置 LoRA 的模型
        processor: HuggingFace Processor
        dataloader: 训练数据 DataLoader
        args: 命令行参数
        device: 训练设备
        local_rank: 分布式训练的 local rank
    """
    is_main_process = (local_rank == -1) or (local_rank == 0)
    world_size = 1 if local_rank == -1 else dist.get_world_size()

    # ---- 准备输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 保存配置 ----
    if is_main_process:
        config_path = output_dir / "finetune_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"[训练配置] 已保存到 {config_path}")

    # ---- 准备优化器 ----
    # 只优化 LoRA 参数（requires_grad=True 的参数）
    # AdamW 是目前最常用的优化器，比 Adam 多了 decoupled weight decay
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(f"[优化器] AdamW, lr={args.learning_rate}, weight_decay={args.weight_decay}")
    print(f"  可训练参数量: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    # ---- 学习率调度器 ----
    # 有效训练步数 = 总步数 / 梯度累积 / world_size
    steps_per_epoch = len(dataloader) // args.grad_accumulation_steps
    effective_total_steps = args.max_steps // world_size
    num_warmup_steps = int(effective_total_steps * args.warmup_ratio)
    scheduler = create_scheduler(optimizer, num_warmup_steps, effective_total_steps)

    # ---- WandB 初始化 ----
    if is_main_process and args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"ft-{args.benchmark}-lr{args.learning_rate}-r{args.lora_rank}",
                config=vars(args),
            )
            print(f"[WandB] 已初始化")
        except ImportError:
            print("警告：未安装 wandb。请运行 pip install wandb 或使用 --no_wandb。")
            args.use_wandb = False

    # ---- 训练统计 ----
    recent_losses = deque(maxlen=args.grad_accumulation_steps)

    # ---- 开始训练 ----
    print(f"\n{'='*60}")
    print(f"开始训练！")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  总步数: {args.max_steps}")
    print(f"  有效 batch_size: {args.batch_size * args.grad_accumulation_steps * world_size}")
    print(f"  Warmup 步数: {num_warmup_steps}")
    print(f"{'='*60}\n")

    model.train()
    optimizer.zero_grad()

    global_step = 0
    epoch = 0

    # 使用 epoch 循环确保所有数据都被使用
    while global_step < args.max_steps:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= args.max_steps:
                break

            # ---- 前向传播 ----
            # 将数据移动到 GPU
            input_ids = batch["input_ids"].to(device) if "input_ids" in batch else None
            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)

            # 使用自动混合精度（bfloat16）
            # 这可以减少显存占用并加速训练，同时保持数值精度
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if input_ids is not None and "labels" in batch:
                    # 模式 1：使用 token-based 训练（官方方式）
                    # 模型同时输出 input_ids 和 labels，loss 自动计算
                    labels = batch["labels"].to(device)
                    output: CausalLMOutputWithPast = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = output.loss
                else:
                    # 模式 2：直接使用连续动作的 MSE loss（简化版，教学用）
                    # 先让模型生成 token，再与真实动作比较
                    # 注意：这是简化版，实际应使用 Action Tokenizer
                    actions = batch["actions"].to(device, dtype=torch.bfloat16)
                    # 使用 processor 处理 inputs
                    processed = processor.tokenizer(
                        batch["prompt"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    output = model(
                        input_ids=processed["input_ids"].to(device),
                        attention_mask=processed["attention_mask"].to(device),
                        pixel_values=pixel_values,
                    )
                    # 简化的 MSE loss：取模型最后隐藏状态作为动作预测
                    # 注意：这不是 OpenVLA 的正确训练方式！
                    # 正确方式应使用 Action Tokenizer 将动作 tokenize 为 labels
                    loss = torch.nn.functional.mse_loss(
                        output.logits[:, -1, :ACTION_DIM].float(),
                        actions.float(),
                    )

            # 归一化 loss 以考虑梯度累积
            normalized_loss = loss / args.grad_accumulation_steps

            # ---- 反向传播 ----
            normalized_loss.backward()

            # 存储最近的 loss 值
            recent_losses.append(loss.item())

            # ---- 梯度更新 ----
            # 每累积 args.grad_accumulation_steps 步后，执行一次参数更新
            if (batch_idx + 1) % args.grad_accumulation_steps == 0:
                # 梯度裁剪：防止梯度爆炸
                # 将梯度的范数限制在 1.0 以内
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # 优化器 step：更新参数
                optimizer.step()

                # 学习率调度器 step：更新学习率
                scheduler.step()

                # 清空梯度
                optimizer.zero_grad()

                global_step += 1

                # ---- 打印日志 ----
                if is_main_process and global_step % args.log_interval == 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    current_lr = scheduler.get_last_lr()[0]
                    log_msg = (
                        f"[Step {global_step}/{args.max_steps}] "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Epoch: {epoch}"
                    )
                    print(log_msg)

                    if args.use_wandb:
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                            "epoch": epoch,
                        }, step=global_step)

            # ---- 保存 checkpoint ----
            if is_main_process and global_step > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, processor, output_dir, global_step, args)

        # 如果一个 epoch 内就达到了 max_steps
        if global_step >= args.max_steps:
            break

    # ---- 训练结束 ----
    print(f"\n训练完成！共 {global_step} 步。")

    # 保存最终 checkpoint
    if is_main_process:
        save_checkpoint(model, processor, output_dir, global_step, args, is_final=True)

    # 结束 WandB
    if is_main_process and args.use_wandb:
        wandb.finish()


# ============================================================
# Checkpoint 保存
# ============================================================

def save_checkpoint(
    model: Any,
    processor: Any,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    is_final: bool = False,
) -> None:
    """
    保存模型 checkpoint。

    保存内容：
    1. LoRA adapter 权重（如果使用 LoRA）
    2. Processor（图像预处理 + tokenizer）
    3. 训练配置

    参数：
        model: 训练中的模型
        processor: HuggingFace Processor
        output_dir: 输出目录
        step: 当前步数
        args: 训练参数
        is_final: 是否是最终 checkpoint
    """
    checkpoint_name = "checkpoint-final" if is_final else f"checkpoint-step-{step}"
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[保存 Checkpoint] 正在保存到 {checkpoint_dir}...")

    # 判断是否是 PeftModel（LoRA 模型）
    if isinstance(model, DDP):
        model_to_save = model.module
    else:
        model_to_save = model

    # 如果是 LoRA 模型，只保存 adapter 权重
    # LoRA adapter 权重通常只有几十 MB，远小于完整模型的 14GB
    if isinstance(model_to_save, PeftModel):
        model_to_save.save_pretrained(checkpoint_dir)
        print(f"  LoRA adapter 已保存")
    else:
        model_to_save.save_pretrained(checkpoint_dir)
        print(f"  完整模型已保存")

    # 保存 Processor
    processor.save_pretrained(checkpoint_dir)

    print(f"  Processor 已保存")
    print(f"  Checkpoint 保存完成: {checkpoint_dir}")


# ============================================================
# 数据集统计量计算和保存
# ============================================================

def compute_and_save_stats(
    dataset: LIBERODataset,
    output_dir: Path,
    benchmark_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算并保存数据集的动作统计量。

    统计量保存在 dataset_statistics.json 中。
    推理时 OpenVLA 会读取该文件来反归一化动作。

    参数：
        dataset: 训练数据集
        output_dir: 输出目录
        benchmark_name: benchmark 名称（用作 unnorm_key）

    返回：
        (action_mean, action_std)
    """
    print(f"\n[统计量计算] 正在计算 {benchmark_name} 的动作统计量...")

    all_actions = []
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        all_actions.append(sample["action"])

    all_actions = np.stack(all_actions, axis=0)
    action_mean, action_std = get_action_stats(all_actions)

    print(f"  动作均值: {action_mean}")
    print(f"  动作标准差: {action_std}")

    # 保存到输出目录
    save_dataset_statistics(action_mean, action_std, output_dir, unnorm_key=benchmark_name)

    return action_mean, action_std


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    """程序入口。"""
    args = parse_args()

    # ---- 设置随机种子 ----
    # 固定种子确保实验可复现
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # ---- 检查 GPU ----
    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 GPU。微调 7B 模型需要至少一块 GPU。\n"
            "如果只有 CPU，可以考虑使用更小的模型或云 GPU 服务。"
        )
    gpu_count = torch.cuda.device_count()
    print(f"[环境] 检测到 {gpu_count} 块 GPU")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_mem / (1024 ** 3)
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")

    # ---- 设置设备 ----
    if args.local_rank != -1:
        # 分布式训练模式
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
        print(f"[分布式] 初始化完成, local_rank={args.local_rank}")
    else:
        # 单卡模式
        device = torch.device("cuda:0")
    torch.cuda.empty_cache()  # 清空 GPU 缓存

    # ---- 加载数据集 ----
    print(f"\n[数据加载] 加载 {args.benchmark} 数据集...")
    dataset = get_libero_dataset(
        data_root=args.data_root,
        benchmark_name=args.benchmark,
        use_local_libero=not args.use_rlds,
        image_aug=args.image_aug,
        chunk_size=args.chunk_size,
        max_transitions=args.max_transitions,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"数据集为空！请检查数据目录 '{args.data_root}' 是否包含正确的数据。\n"
            f"提示：\n"
            f"  - 本地模式：确保 LIBERO 已安装且数据在正确路径\n"
            f"  - RLDS 模式：确保数据已下载到指定目录"
        )

    # ---- 计算并保存统计量 ----
    output_dir = Path(args.output_dir)
    compute_and_save_stats(dataset, output_dir, args.benchmark)

    # ---- 加载模型 ----
    print(f"\n[模型] 加载 OpenVLA...")
    model, processor = load_model_and_processor(
        vla_path=args.vla_path,
        device=device,
        use_4bit=args.use_4bit,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    # ---- 配置 LoRA ----
    if args.use_lora:
        model = setup_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        print("[警告] 未使用 LoRA，将进行全参数微调。"
              "这需要大量显存（建议 80GB+ GPU）。")

    # ---- 包装 DDP（多卡训练）----
    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    # ---- 创建 DataLoader ----
    print(f"\n[DataLoader] 创建训练 DataLoader...")
    dataloader = create_vla_dataloader(
        dataset=dataset,
        processor=processor,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # RLDS 数据需要设为 0
    )

    print(f"  批次数: {len(dataloader)}")
    print(f"  样本数: {len(dataset)}")

    # ---- 开始训练 ----
    train(
        model=model,
        processor=processor,
        dataloader=dataloader,
        args=args,
        device=device,
        local_rank=args.local_rank,
    )


if __name__ == "__main__":
    main()
