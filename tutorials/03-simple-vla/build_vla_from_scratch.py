#!/usr/bin/env python3
"""
build_vla_from_scratch.py -- 从零搭建 VLA（教学版）

================================================================================
  VLA-Zero-to-Hero 教学项目 · Stage 3: 简单 VLA
================================================================================

本脚本使用预训练 CLIP 视觉编码器 + 文本编码器，配合小型 MLP 融合层和
策略头，搭建一个端到端的 VLA (Vision-Language-Action) 模型。

核心思想：
  VLA = 预训练视觉编码器 (CLIP ViT)
      + 预训练语言编码器 (CLIP Text)
      + 融合层 (MLP)
      + 策略头 (MLP → 7维动作)

  预训练部分（CLIP）冻结，只训练融合层和策略头。
  这就是 VLA 的「迁移学习」本质。

功能：
  - Step 1: 加载预训练 CLIP 并冻结
  - Step 2: 定义融合层 + 策略头
  - Step 3: 生成模拟训练数据（随机图像 + 随机指令 + 随机动作）
  - Step 4: 训练融合层和策略头
  - Step 5: 推理并可视化输出动作分布和训练 loss 曲线

注意：
  本脚本的目的不是得到可用的策略，而是让学习者理解 VLA 的训练流程。
  模型使用随机数据训练，输出动作不对应任何真实物理行为。
  要获得可用的 VLA，需要真实的机器人操作数据（如 Bridge、LIBERO 等）。

依赖：
  pip install torch transformers pillow numpy matplotlib

用法示例：
  # 基本用法：训练 100 步
  python build_vla_from_scratch.py

  # 更多训练步数和更大 batch
  python build_vla_from_scratch.py --num_epochs 20 --batch_size 16

  # 调整模型参数
  python build_vla_from_scratch.py --hidden_dim 512 --action_dim 7 --lr 1e-3

  # 保存训练曲线
  python build_vla_from_scratch.py --save_loss_curve loss_curve.png
"""

import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

import matplotlib.pyplot as plt

# ── 中文字体设置 ──────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
#  Step 1: 加载预训练 CLIP（冻结）
# ==============================================================================

def load_clip_model(device: str = "cpu"):
    """
    加载预训练 CLIP 模型并冻结其所有参数。

    CLIP (Contrastive Language-Image Pre-training):
      - 视觉编码器: ViT-B/32，输出 512 维图像特征
      - 文本编码器: Transformer，输出 512 维文本特征
      - 在 4 亿图文对上预训练，具备强大的视觉-语言理解能力

    在 VLA 中，CLIP 的作用是提供高质量的视觉和语言特征，
    我们只需要训练一个小型 MLP 将这些特征映射到动作空间。

    Args:
        device: 设备 ('cpu' 或 'cuda')

    Returns:
        tuple: (clip_model, clip_processor, clip_dim)
    """
    print("[Step 1] 加载预训练 CLIP 模型...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 冻结所有 CLIP 参数（不参与训练）
    for param in clip_model.parameters():
        param.requires_grad = False

    clip_model.to(device)
    clip_model.eval()  # 设为评估模式（关闭 dropout 等）

    clip_dim = clip_model.config.projection_dim  # 512
    print(f"  CLIP 模型已加载并冻结")
    print(f"  视觉编码器: ViT-B/32")
    print(f"  输出维度:   {clip_dim}")
    print(f"  设备:       {device}")
    print(f"  可训练参数: {sum(p.numel() for p in clip_model.parameters() if p.requires_grad):,}")

    return clip_model, clip_processor, clip_dim


# ==============================================================================
#  Step 2: 定义融合层 + 策略头
# ==============================================================================

class VLAPolicyHead(nn.Module):
    """
    VLA 的融合层和策略头。

    结构：
      图像特征 (512) ─┐
                      ├→ 拼接 (1024) → 融合 MLP (256) → 策略 MLP (128) → Tanh → 动作 (action_dim)
      文本特征 (512) ─┘

    设计要点：
      - 融合层：将视觉和语言特征统一到同一语义空间
      - 策略头：从融合特征映射到动作空间
      - Tanh 激活：将输出限制在 [-1, 1]，适合归一化的动作
      - 真实 VLA（如 OpenVLA）会用 LLM decoder 做策略头，
        这里用简单 MLP 以便教学理解
    """

    def __init__(self, clip_dim: int = 512, hidden_dim: int = 256, action_dim: int = 7):
        super().__init__()
        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 融合层：将 [图像特征; 文本特征] 映射到统一表示
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 策略头：从融合表示映射到动作空间
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # 输出在 [-1, 1] 范围内
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, clip_dim] — CLIP 图像特征
            text_features:  [B, clip_dim] — CLIP 文本特征

        Returns:
            actions: [B, action_dim] — 预测的动作
        """
        # 拼接视觉和语言特征
        fused = torch.cat([image_features, text_features], dim=-1)  # [B, clip_dim*2]

        # 融合
        z = self.fusion(fused)  # [B, hidden_dim]

        # 策略输出
        actions = self.policy_head(z)  # [B, action_dim]

        return actions


# ==============================================================================
#  Step 3: 生成模拟训练数据
# ==============================================================================

class SyntheticVLADataset(Dataset):
    """
    生成模拟 VLA 训练数据。

    每个样本包含：
      - 随机图像（32x32 彩色噪声图，足够让 CLIP 提取特征）
      - 随机指令文本
      - 随机目标动作（7维，范围 [-1, 1]）

    模拟指令集合：模仿真实机器人任务中的常见指令。
    """

    # 模拟指令集合
    INSTRUCTIONS = [
        "pick up the red cup",
        "place the object on the table",
        "push the button",
        "open the drawer",
        "close the door",
        "pour water into the glass",
        "stack the blocks",
        "wipe the table",
        "reach for the apple",
        "move the robot arm left",
    ]

    def __init__(self, num_samples: int = 500, image_size: int = 32):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = 7

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # 随机生成 32x32 RGB 图像（归一化到 [0,1]）
        image = np.random.rand(self.image_size, self.image_size, 3).astype(np.float32)

        # 随机选择指令
        instruction = random.choice(self.INSTRUCTIONS)

        # 随机生成目标动作（均匀分布在 [-1, 1]）
        action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)

        return {
            "image": image,
            "instruction": instruction,
            "action": torch.from_numpy(action),
        }


# ==============================================================================
#  特征提取（使用冻结的 CLIP）
# ==============================================================================

def extract_clip_features(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    batch: dict,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用冻结的 CLIP 提取图像和文本特征。

    Args:
        clip_model: 冻结的 CLIP 模型
        clip_processor: CLIP 预处理器
        batch: 包含 'image' 和 'instruction' 的批次数据
        device: 计算设备

    Returns:
        tuple: (image_features, text_features, target_actions)
    """
    # 图像预处理：CLIP processor 需要 PIL Image
    from PIL import Image
    images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in batch["image"]]
    texts = batch["instruction"]

    # CLIP 预处理
    inputs = clip_processor(
        text=texts, images=images_pil,
        return_tensors="pt", padding=True,
    )

    # 将输入移到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 提取特征（不计算梯度，因为 CLIP 是冻结的）
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds   # [B, 512]
        text_features = outputs.text_embeds     # [B, 512]

    # 目标动作
    target_actions = torch.stack(batch["action"]).to(device)  # [B, 7]

    return image_features, text_features, target_actions


# ==============================================================================
#  Step 4: 训练
# ==============================================================================

def train(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    policy_head: VLAPolicyHead,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list:
    """
    训练融合层和策略头。

    训练策略：
      - 损失函数: MSE Loss（预测动作 vs 目标动作）
      - 优化器: Adam
      - CLIP 完全冻结，只更新 policy_head 的参数
      - 每个 epoch 打印 loss

    Args:
        clip_model: 冻结的 CLIP 模型
        clip_processor: CLIP 预处理器
        policy_head: 待训练的策略头
        train_loader: 训练数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 计算设备

    Returns:
        list: 每个 batch 的 loss 记录
    """
    print(f"\n[Step 4] 开始训练...")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率:   {lr}")
    print(f"  批次大小: {train_loader.batch_size}")
    print(f"  总批次数: {len(train_loader)}")

    # 只优化策略头参数（CLIP 已冻结）
    optimizer = optim.Adam(policy_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []

    policy_head.to(device)
    policy_head.train()

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in train_loader:
            # 提取 CLIP 特征
            image_features, text_features, target_actions = extract_clip_features(
                clip_model, clip_processor, batch, device
            )

            # 前向传播（仅通过策略头）
            pred_actions = policy_head(image_features, text_features)

            # 计算损失
            loss = criterion(pred_actions, target_actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            loss_history.append(batch_loss)

        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch [{epoch+1:>3d}/{num_epochs}]  Loss: {avg_loss:.6f}")

    print(f"\n  训练完成! 最终平均 Loss: {np.mean(loss_history[-len(train_loader):]):.6f}")

    return loss_history


# ==============================================================================
#  Step 5: 推理与可视化
# ==============================================================================

def inference_and_visualize(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    policy_head: VLAPolicyHead,
    loss_history: list,
    save_loss_path: str = None,
    save_action_path: str = None,
    device: str = "cpu",
):
    """
    推理演示和可视化。

    包含两部分：
      1. 训练 loss 曲线
      2. 推理输出动作分布（展示模型在测试样本上的输出）
    """
    print("\n[Step 5] 推理与可视化...")

    # ── 5a: 训练 loss 曲线 ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(loss_history, linewidth=1, alpha=0.7, color="#2196F3")
    # 平滑曲线
    if len(loss_history) > 20:
        window = min(20, len(loss_history) // 5)
        smoothed = np.convolve(loss_history, np.ones(window)/window, mode="valid")
        ax1.plot(np.arange(window-1, len(loss_history)), smoothed,
                 linewidth=2, color="#E91E63", label=f"平滑 (窗口={window})")

    ax1.set_title("训练 Loss 曲线", fontsize=14, fontweight="bold")
    ax1.set_xlabel("训练步数 (batch)", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── 5b: 推理输出动作分布 ──────────────────────────────────────────────
    policy_head.eval()

    # 生成测试样本
    test_instructions = [
        "pick up the red cup",
        "place the object on the table",
        "push the button",
        "open the drawer",
        "pour water into the glass",
    ]
    n_test = len(test_instructions)
    test_images = [np.random.rand(32, 32, 3) for _ in range(n_test)]

    from PIL import Image
    images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in test_images]

    with torch.no_grad():
        inputs = clip_processor(
            text=test_instructions, images=images_pil,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        pred_actions = policy_head(image_features, text_features).cpu().numpy()

    # 可视化预测动作（7个维度的柱状图）
    action_dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    colors = plt.cm.Set2(np.linspace(0, 1, n_test))
    x_pos = np.arange(len(action_dim_names))
    bar_width = 0.15

    for i in range(n_test):
        offset = (i - n_test / 2 + 0.5) * bar_width
        ax2.bar(x_pos + offset, pred_actions[i], bar_width,
                label=test_instructions[i][:15], color=colors[i], alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(action_dim_names, fontsize=10, rotation=30)
    ax2.set_title("推理输出动作分布（模拟数据）", fontsize=14, fontweight="bold")
    ax2.set_ylabel("动作值", fontsize=12)
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # 添加说明文字
    ax2.text(
        0.5, -0.22,
        "注意：使用随机数据训练，输出动作不对应真实物理行为。此图仅展示模型输出的数值分布。",
        transform=ax2.transAxes, fontsize=9, ha="center", style="italic",
        color="gray",
    )

    fig.suptitle("VLA 从零搭建 — 训练与推理可视化", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_loss_path or save_action_path:
        save_path = save_loss_path or save_action_path or "vla_training.png"
        print(f"  保存可视化到 {save_path}")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    # 打印推理结果
    print(f"\n  推理结果（{n_test} 个测试样本）:")
    for i, inst in enumerate(test_instructions):
        action = pred_actions[i]
        action_str = ", ".join([f"{v:+.4f}" for v in action])
        print(f"    指令: \"{inst}\"")
        print(f"    动作: [{action_str}]")
        print()

    # ── 训练前后对比 ──────────────────────────────────────────────────────
    print("  关键理解:")
    print("    1. Loss 应该从较高值逐渐下降（即使数据是随机的）")
    print("    2. 由于使用随机数据，模型学习的是噪声模式，不对应真实物理")
    print("    3. 要获得有意义的 VLA，需要用真实机器人操作数据替代模拟数据")
    print("    4. 真实训练数据格式通常为 RLDS 或自定义数据集（见 Stage 4）")


# ==============================================================================
#  主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从零搭建 VLA（教学版）— CLIP + MLP 端到端训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build_vla_from_scratch.py                        # 基本用法
  python build_vla_from_scratch.py --num_epochs 20       # 训练更多轮
  python build_vla_from_scratch.py --hidden_dim 512       # 更大的融合层
  python build_vla_from_scratch.py --save_loss_curve vla_loss.png
        """,
    )
    parser.add_argument("--num_samples", type=int, default=500, help="模拟数据样本数，默认: 500")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数，默认: 10")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小，默认: 8")
    parser.add_argument("--hidden_dim", type=int, default=256, help="融合层隐藏维度，默认: 256")
    parser.add_argument("--action_dim", type=int, default=7, help="动作维度，默认: 7")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率，默认: 1e-3")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda)，默认: auto")
    parser.add_argument("--save_loss_curve", type=str, default=None, help="保存训练曲线 PNG 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")

    args = parser.parse_args()

    # 设置随机种子（可复现）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("  VLA 从零搭建（教学版）")
    print("  CLIP + MLP → 端到端训练")
    print("=" * 60)
    print(f"  设备:       {device}")
    print(f"  数据样本:   {args.num_samples}")
    print(f"  训练轮数:   {args.num_epochs}")
    print(f"  批次大小:   {args.batch_size}")
    print(f"  隐藏维度:   {args.hidden_dim}")
    print(f"  动作维度:   {args.action_dim}")
    print(f"  学习率:     {args.lr}")

    # ── Step 1: 加载 CLIP ────────────────────────────────────────────────
    clip_model, clip_processor, clip_dim = load_clip_model(device)

    # ── Step 2: 定义策略头 ─────────────────────────────────────────────────
    print("\n[Step 2] 定义融合层 + 策略头...")
    policy_head = VLAPolicyHead(
        clip_dim=clip_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
    )
    policy_head.to(device)

    total_params = sum(p.numel() for p in policy_head.parameters())
    trainable_params = sum(p.numel() for p in policy_head.parameters() if p.requires_grad)
    print(f"  策略头总参数量:   {total_params:,}")
    print(f"  可训练参数量:     {trainable_params:,}")
    print(f"  模型结构:")
    print(f"    融合层: Linear({clip_dim*2} → {args.hidden_dim}) → ReLU → Dropout → "
          f"Linear({args.hidden_dim} → {args.hidden_dim}) → ReLU")
    print(f"    策略头: Linear({args.hidden_dim} → {args.hidden_dim//2}) → ReLU → "
          f"Linear({args.hidden_dim//2} → {args.action_dim}) → Tanh")

    # ── Step 3: 生成模拟数据 ───────────────────────────────────────────────
    print("\n[Step 3] 生成模拟训练数据...")
    dataset = SyntheticVLADataset(num_samples=args.num_samples)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次数:     {len(train_loader)}")

    # 展示一个样本
    sample = dataset[0]
    print(f"  样本示例:")
    print(f"    图像形状: {sample['image'].shape}")
    print(f"    指令:     \"{sample['instruction']}\"")
    print(f"    动作:     {sample['action'].numpy()}")

    # ── Step 4: 训练 ──────────────────────────────────────────────────────
    loss_history = train(
        clip_model, clip_processor, policy_head,
        train_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
    )

    # ── Step 5: 推理与可视化 ─────────────────────────────────────────────
    inference_and_visualize(
        clip_model, clip_processor, policy_head,
        loss_history,
        save_loss_path=args.save_loss_curve,
        device=device,
    )

    print("\n完成! 这就是 VLA 从零搭建的完整流程。")
    print("下一步: 学习 Stage 4 — 在真实数据上微调 OpenVLA。")


if __name__ == "__main__":
    main()
