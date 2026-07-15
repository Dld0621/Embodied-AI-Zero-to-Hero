#!/usr/bin/env python3
"""
train_custom_data.py -- 使用自定义数据微调 OpenVLA

================================================================================
  VLA-Zero-to-Hero 教学项目 · Stage 4: 微调实践
================================================================================

本脚本用于学习者使用自己收集的机器人数据微调 VLA 模型。

数据格式（JSONL）：
  {"image_path": "path/to/image.jpg", "instruction": "pick up the cup",
   "action": [dx, dy, dz, droll, dpitch, dyaw, gripper]}

每行是一个 (图像, 指令, 动作) 训练样本。

与 finetune_libero.py 的区别：
  - 数据来源：自定义 JSONL 文件，而非 LIBERO 仿真环境
  - 数据加载：从本地图像文件读取，而非 HDF5 或 RLDS
  - 动作归一化：自动从数据中计算统计量
  - 图像增强：支持更多选项（翻转、旋转等）

依赖：
  pip install torch transformers peft accelerate pillow numpy
  pip install wandb          # 可选

用法示例：
  # 基本用法
  python train_custom_data.py \\
      --vla_path openvla/openvla-7b \\
      --jsonl_path ./data/my_robot_data.jsonl \\
      --image_root ./data/images \\
      --output_dir ./checkpoints/my-vla

  # 省显存模式
  python train_custom_data.py \\
      --vla_path openvla/openvla-7b \\
      --jsonl_path ./data/my_robot_data.jsonl \\
      --image_root ./data/images \\
      --output_dir ./checkpoints/my-vla \\
      --batch_size 2 \\
      --use_gradient_checkpointing \\
      --max_steps 5000

  # 使用 WandB 日志和图像增强
  python train_custom_data.py \\
      --vla_path openvla/openvla-7b \\
      --jsonl_path ./data/my_robot_data.jsonl \\
      --image_root ./data/images \\
      --output_dir ./checkpoints/my-vla \\
      --use_wandb \\
      --image_aug

================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset_utils import (
    ACTION_DIM,
    CustomJSONLDataset,
    create_vla_dataloader,
    get_action_stats,
    get_train_image_transforms,
    normalize_action,
    save_dataset_statistics,
)

# ============================================================
# 常量
# ============================================================

DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_STEPS = 50_000
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_SAVE_STEPS = 2000
DEFAULT_WARMUP_RATIO = 0.1

LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用自定义 JSONL 数据微调 OpenVLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 模型相关 ----
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--vla_path", type=str, default="openvla/openvla-7b",
        help="预训练 VLA 模型的路径或 HuggingFace ID。",
    )
    model_group.add_argument(
        "--use_lora", action="store_true", default=True,
        help="是否使用 LoRA 微调。",
    )
    model_group.add_argument(
        "--lora_rank", type=int, default=DEFAULT_LORA_RANK,
        help="LoRA 低秩维度。",
    )
    model_group.add_argument(
        "--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA,
        help="LoRA 缩放因子。",
    )
    model_group.add_argument(
        "--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT,
        help="LoRA dropout。",
    )

    # ---- 数据相关 ----
    data_group = parser.add_argument_group("数据配置")
    data_group.add_argument(
        "--jsonl_path", type=str, required=True,
        help="JSONL 训练数据文件路径。"
             "每行格式：{\"image_path\": \"path.jpg\", \"instruction\": \"...\", \"action\": [...]}",
    )
    data_group.add_argument(
        "--image_root", type=str, default="",
        help="图像根目录。如果 JSONL 中的 image_path 是相对路径，则拼接此目录。",
    )
    data_group.add_argument(
        "--action_mean", type=float, nargs="+", default=None,
        help="预计算的动作均值（7 个浮点数）。如不指定则从数据中自动计算。",
    )
    data_group.add_argument(
        "--action_std", type=float, nargs="+", default=None,
        help="预计算的动作标准差（7 个浮点数）。如不指定则从数据中自动计算。",
    )
    data_group.add_argument(
        "--image_aug", action="store_true", default=True,
        help="是否使用图像增强（随机裁剪、颜色抖动等）。",
    )
    data_group.add_argument(
        "--unnorm_key", type=str, default="custom_data",
        help="动作反归一化的 key 名称，保存到 dataset_statistics.json 中。",
    )
    data_group.add_argument(
        "--val_jsonl_path", type=str, default=None,
        help="验证集 JSONL 文件路径（可选）。如果提供，会定期在验证集上评估。",
    )
    data_group.add_argument(
        "--val_interval", type=int, default=500,
        help="每隔多少步在验证集上评估。",
    )

    # ---- 训练相关 ----
    train_group = parser.add_argument_group("训练配置")
    train_group.add_argument(
        "--output_dir", type=str, default="./runs-custom",
        help="checkpoint 和日志的保存目录。",
    )
    train_group.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="训练批次大小。",
    )
    train_group.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help="最大训练步数。",
    )
    train_group.add_argument(
        "--save_steps", type=int, default=DEFAULT_SAVE_STEPS,
        help="checkpoint 保存间隔。",
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="学习率。",
    )
    train_group.add_argument(
        "--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO,
        help="Warmup 占总训练步数的比例。",
    )
    train_group.add_argument(
        "--grad_accumulation_steps", type=int, default=1,
        help="梯度累积步数。",
    )
    train_group.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="权重衰减。",
    )
    train_group.add_argument(
        "--use_gradient_checkpointing", action="store_true", default=False,
        help="启用梯度检查点（省显存）。",
    )
    train_group.add_argument(
        "--seed", type=int, default=42,
        help="随机种子。",
    )

    # ---- 日志 ----
    log_group = parser.add_argument_group("日志配置")
    log_group.add_argument(
        "--use_wandb", action="store_true", default=False,
        help="是否使用 WandB。",
    )
    log_group.add_argument(
        "--wandb_project", type=str, default="vla-custom-finetuning",
        help="WandB 项目名。",
    )
    log_group.add_argument(
        "--wandb_entity", type=str, default=None,
        help="WandB 用户名。",
    )
    log_group.add_argument(
        "--log_interval", type=int, default=10,
        help="日志打印间隔。",
    )

    return parser.parse_args()


# ============================================================
# 数据验证和准备
# ============================================================

def validate_jsonl(jsonl_path: str) -> int:
    """
    验证 JSONL 文件格式并统计样本数。

    在开始训练之前先验证数据，可以避免训练中途因数据问题而中断。

    参数：
        jsonl_path: JSONL 文件路径

    返回：
        样本数量
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")

    num_samples = 0
    errors = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"第 {line_idx + 1} 行: JSON 解析错误 - {e}")
                continue

            # 检查必要字段
            for field in ["image_path", "instruction", "action"]:
                if field not in sample:
                    errors.append(f"第 {line_idx + 1} 行: 缺少 '{field}' 字段")

            # 检查动作维度
            if "action" in sample:
                action = sample["action"]
                if len(action) != ACTION_DIM:
                    errors.append(
                        f"第 {line_idx + 1} 行: 动作维度应为 {ACTION_DIM}，实际为 {len(action)}"
                    )
                elif not all(isinstance(a, (int, float)) for a in action):
                    errors.append(f"第 {line_idx + 1} 行: 动作值应全部为数字")

            num_samples += 1

    if errors:
        print(f"[数据验证] 发现 {len(errors)} 个问题：")
        for err in errors[:10]:  # 最多显示 10 个错误
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误未显示")

    if num_samples == 0:
        raise ValueError(f"JSONL 文件中没有有效的训练样本: {jsonl_path}")

    print(f"[数据验证] 共 {num_samples} 个样本")
    if errors:
        print(f"[数据验证] 警告：{len(errors)} 个样本存在问题，可能影响训练")

    return num_samples


# ============================================================
# 模型加载
# ============================================================

def register_openvla_classes() -> None:
    """注册 OpenVLA 自定义类到 HuggingFace Auto Classes。"""
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
    except ImportError:
        pass


def load_model_and_processor(
    vla_path: str,
    device: torch.device,
    use_gradient_checkpointing: bool = False,
) -> Tuple[Any, Any]:
    """加载 OpenVLA 模型和 Processor。"""
    register_openvla_classes()

    print(f"[模型加载] 正在从 '{vla_path}' 加载...")
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"  梯度检查点已启用")

    print(f"  模型加载完成")
    return model, processor


def setup_lora(model: Any, rank: int, alpha: int, dropout: float) -> Any:
    """配置 LoRA adapter。"""
    print(f"[LoRA] rank={rank}, alpha={alpha}, dropout={dropout}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        init_lora_weights="gaussian",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ============================================================
# 训练
# ============================================================

def train(
    model: Any,
    processor: Any,
    train_dataset: CustomJSONLDataset,
    val_dataset: Optional[CustomJSONLDataset],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    训练循环。

    与 finetune_libero.py 的主要区别：
    1. 数据来自自定义 JSONL，而非 LIBERO
    2. 没有分布式训练逻辑（简化）
    3. 支持验证集评估

    参数：
        model: LoRA 模型
        processor: HuggingFace Processor
        train_dataset: 训练数据集
        val_dataset: 验证数据集（可选）
        args: 命令行参数
        device: 训练设备
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 保存配置 ----
    config_path = output_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"[配置] 已保存到 {config_path}")

    # ---- 准备优化器 ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(f"[优化器] AdamW, lr={args.learning_rate}")
    print(f"  可训练参数: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    # ---- 学习率调度器 ----
    num_warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = create_scheduler(optimizer, num_warmup_steps, args.max_steps)

    # ---- WandB ----
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"custom-ft-lr{args.learning_rate}-r{args.lora_rank}",
                config=vars(args),
            )
        except ImportError:
            print("警告：未安装 wandb。")
            args.use_wandb = False

    # ---- DataLoader ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"开始训练自定义数据")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset) if val_dataset else '无'}")
    print(f"  总步数: {args.max_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  有效 batch: {args.batch_size * args.grad_accumulation_steps}")
    print(f"{'='*60}\n")

    model.train()
    recent_losses = deque(maxlen=args.grad_accumulation_steps)
    global_step = 0
    best_val_loss = float("inf")

    while global_step < args.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            # ---- 准备 batch ----
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            actions = batch["action"].to(device, dtype=torch.bfloat16)

            # Tokenize prompts
            prompts = batch["prompt"]
            tokenized = processor.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            # ---- 前向传播 ----
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 获取模型输出
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )

                # 计算 MSE loss
                # 注意：这里使用简化的连续动作 MSE loss
                # 对于更精确的训练，应使用 OpenVLA 的 Action Tokenizer
                # 将动作 tokenize 为 token ID，然后使用 cross-entropy loss
                pred_actions = outputs.logits[:, -1, :ACTION_DIM].float()
                loss = torch.nn.functional.mse_loss(pred_actions, actions.float())

            # 归一化 loss
            normalized_loss = loss / args.grad_accumulation_steps
            normalized_loss.backward()

            recent_losses.append(loss.item())

            # ---- 梯度更新 ----
            if (batch_idx + 1) % args.grad_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                # 优化器 step
                optimizer.step()
                # 学习率调度
                scheduler.step()
                # 清空梯度
                optimizer.zero_grad()

                global_step += 1

                # ---- 打印日志 ----
                if global_step % args.log_interval == 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"[Step {global_step}/{args.max_steps}] "
                        f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}"
                    )

                    if args.use_wandb:
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                        }, step=global_step)

                # ---- 验证 ----
                if (val_loader is not None and
                    global_step % args.val_interval == 0 and
                    global_step > 0):
                    val_loss = evaluate(model, processor, val_loader, device)
                    print(f"  [验证] Val Loss: {val_loss:.6f}")

                    if args.use_wandb:
                        wandb.log({"val_loss": val_loss}, step=global_step)

                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = output_dir / "checkpoint-best"
                        save_checkpoint(model, processor, best_dir, global_step, args)
                        print(f"  [最佳模型] Val Loss 改善: {val_loss:.6f}")

                # ---- 保存 checkpoint ----
                if global_step > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(
                        model, processor, output_dir, global_step, args
                    )

    # ---- 训练结束 ----
    print(f"\n训练完成！共 {global_step} 步。")
    save_checkpoint(model, processor, output_dir, global_step, args, is_final=True)

    if args.use_wandb:
        wandb.finish()


def evaluate(
    model: Any,
    processor: Any,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    在验证集上评估模型。

    参数：
        model: 模型
        processor: Processor
        val_loader: 验证集 DataLoader
        device: 设备

    返回：
        平均验证 loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            actions = batch["action"].to(device, dtype=torch.bfloat16)
            prompts = batch["prompt"]

            tokenized = processor.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096,
            )

            outputs = model(
                input_ids=tokenized["input_ids"].to(device),
                attention_mask=tokenized["attention_mask"].to(device),
                pixel_values=pixel_values,
            )

            pred_actions = outputs.logits[:, -1, :ACTION_DIM].float()
            loss = torch.nn.functional.mse_loss(pred_actions, actions.float())
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


# ============================================================
# 工具函数
# ============================================================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Any:
    """创建 cosine schedule with warmup 学习率调度器。"""
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def save_checkpoint(
    model: Any,
    processor: Any,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    is_final: bool = False,
) -> None:
    """保存 checkpoint。"""
    name = "checkpoint-final" if is_final else f"checkpoint-step-{step}"
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[保存] 正在保存到 {checkpoint_dir}...")

    if isinstance(model, PeftModel):
        model.save_pretrained(checkpoint_dir)
        print("  LoRA adapter 已保存")
    else:
        model.save_pretrained(checkpoint_dir)
        print("  完整模型已保存")

    processor.save_pretrained(checkpoint_dir)
    print(f"  Processor 已保存")


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    """程序入口。"""
    args = parse_args()

    # ---- 设置随机种子 ----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # ---- 检查 GPU ----
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU 来微调 7B 模型。")
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"[环境] GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    # ---- 验证数据 ----
    print(f"\n[数据] 验证 JSONL 文件...")
    num_samples = validate_jsonl(args.jsonl_path)

    # ---- 处理动作统计量 ----
    action_mean = None
    action_std = None
    if args.action_mean is not None and args.action_std is not None:
        if len(args.action_mean) != ACTION_DIM or len(args.action_std) != ACTION_DIM:
            raise ValueError(
                f"action_mean 和 action_std 应各有 {ACTION_DIM} 个值，"
                f"但得到 {len(args.action_mean)} 和 {len(args.action_std)}"
            )
        action_mean = np.array(args.action_mean, dtype=np.float32)
        action_std = np.array(args.action_std, dtype=np.float32)
        print(f"[数据] 使用外部提供的动作统计量")

    # ---- 加载训练数据集 ----
    print(f"\n[数据] 加载训练数据集...")
    train_dataset = CustomJSONLDataset(
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        image_aug=args.image_aug,
        action_mean=action_mean,
        action_std=action_std,
    )

    # ---- 保存数据集统计量 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_statistics(
        train_dataset.action_mean,
        train_dataset.action_std,
        output_dir,
        unnorm_key=args.unnorm_key,
    )

    # ---- 加载验证集（可选）----
    val_dataset = None
    if args.val_jsonl_path:
        print(f"\n[数据] 加载验证数据集...")
        val_dataset = CustomJSONLDataset(
            jsonl_path=args.val_jsonl_path,
            image_root=args.image_root,
            image_aug=False,  # 验证集不做增强
            action_mean=train_dataset.action_mean,
            action_std=train_dataset.action_std,
        )

    # ---- 加载模型 ----
    print(f"\n[模型] 加载 OpenVLA...")
    model, processor = load_model_and_processor(
        vla_path=args.vla_path,
        device=device,
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

    # ---- 开始训练 ----
    train(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
        device=device,
    )


if __name__ == "__main__":
    main()
