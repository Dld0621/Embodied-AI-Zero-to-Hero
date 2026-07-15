#!/usr/bin/env python3
"""
visualize_vla.py -- VLA 可视化工具
====================================
独立的可视化工具，帮助理解 VLA 的内部状态和输出。

功能：
  1. 动作轨迹可视化：7-DOF 动作时间序列 + gripper 状态
  2. 注意力可视化：CLIP/OpenVLA 的 attention rollout 热力图
  3. 评估结果统计：从评估日志 JSON 绘制各任务成功率柱状图
  4. 图像-动作对比：并排显示输入图像和对应动作向量

用法：
    # 绘制动作轨迹
    python visualize_vla.py --mode action_trajectory --traj_path results/episode_0.json

    # 绘制评估结果
    python visualize_vla.py --mode eval_results --eval_path results/eval_results.json

    # CLIP 注意力可视化
    python visualize_vla.py --mode attention --image_path scene.jpg

    # 图像-动作对比
    python visualize_vla.py --mode image_action --traj_path results/episode_0.json

依赖：
    pip install matplotlib numpy torch transformers
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，确保在无 GUI 环境下也能工作
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors


# ===========================================================================
# 模式 1: 动作轨迹可视化
# ===========================================================================

def plot_action_trajectory(
    traj_path: str,
    output_dir: str = "results",
    highlight_steps: Optional[List[int]] = None,
):
    """
    绘制 7-DOF 动作的时间序列曲线。

    Args:
        traj_path: episode JSON 文件路径（由 sim_closed_loop_demo.py 生成）
        output_dir: 输出图片目录
        highlight_steps: 需要高亮标记的步骤索引列表
    """
    # 加载数据
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = np.array(data["actions"])
    num_steps = len(actions)
    action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    action_units = ["m", "m", "m", "rad", "rad", "rad", "open/close"]

    print(f"Loaded {num_steps} steps from {traj_path}")
    print(f"Action shape: {actions.shape}")

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"VLA Action Trajectory  |  {data.get('policy', 'unknown')}  |  "
        f"{num_steps} steps  |  Instruction: {data.get('instruction', 'N/A')}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.25,
                  left=0.06, right=0.97, top=0.93, bottom=0.05)

    # --- 上半部分：6-DOF 位姿动作 ---
    ax_pos = fig.add_subplot(gs[0, :])

    # 位置分量 (dx, dy, dz)
    pos_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i in range(3):
        line = ax_pos.plot(
            actions[:, i], label=f"{action_names[i]} ({action_units[i]})",
            color=pos_colors[i], linewidth=0.9, alpha=0.85,
        )
        # 添加范围阴影
        mean_val = np.mean(actions[:, i])
        std_val = np.std(actions[:, i])
        ax_pos.fill_between(
            range(num_steps),
            mean_val - std_val, mean_val + std_val,
            color=pos_colors[i], alpha=0.08,
        )

    ax_pos.set_xlabel("Step", fontsize=11)
    ax_pos.set_ylabel("Delta Position (m)", fontsize=11)
    ax_pos.set_title("Position Actions (dx, dy, dz)", fontsize=12, fontweight="bold")
    ax_pos.legend(loc="upper right", fontsize=9, ncol=3)
    ax_pos.grid(True, alpha=0.3, linestyle="--")
    ax_pos.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    if highlight_steps:
        for hs in highlight_steps:
            ax_pos.axvline(x=hs, color="orange", linestyle=":", linewidth=1.0, alpha=0.6)

    # --- 中间：旋转动作 ---
    ax_rot = fig.add_subplot(gs[1, 0])
    rot_colors = ["#9b59b6", "#e67e22", "#1abc9c"]
    for i in range(3, 6):
        ax_rot.plot(
            actions[:, i], label=f"{action_names[i]} ({action_units[i]})",
            color=rot_colors[i - 3], linewidth=0.9, alpha=0.85,
        )
    ax_rot.set_xlabel("Step", fontsize=11)
    ax_rot.set_ylabel("Delta Rotation (rad)", fontsize=11)
    ax_rot.set_title("Rotation Actions (droll, dpitch, dyaw)", fontsize=12, fontweight="bold")
    ax_rot.legend(loc="upper right", fontsize=9, ncol=3)
    ax_rot.grid(True, alpha=0.3, linestyle="--")
    ax_rot.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    if highlight_steps:
        for hs in highlight_steps:
            ax_rot.axvline(x=hs, color="orange", linestyle=":", linewidth=1.0, alpha=0.6)

    # --- 中间右侧：Gripper 状态 ---
    ax_grip = fig.add_subplot(gs[1, 1])
    grip_vals = actions[:, 6]
    ax_grip.fill_between(
        range(num_steps), grip_vals, alpha=0.3, color="steelblue",
    )
    ax_grip.plot(grip_vals, color="steelblue", linewidth=1.0)

    # 标记 gripper 状态变化点
    grip_diff = np.diff(grip_vals)
    close_points = np.where(grip_diff < -0.2)[0]
    open_points = np.where(grip_diff > 0.2)[0]
    if len(close_points) > 0:
        ax_grip.scatter(close_points, grip_vals[close_points],
                        c="red", s=30, zorder=5, label=f"Close events ({len(close_points)})")
    if len(open_points) > 0:
        ax_grip.scatter(open_points, grip_vals[open_points],
                        c="green", s=30, zorder=5, label=f"Open events ({len(open_points)})")

    ax_grip.set_xlabel("Step", fontsize=11)
    ax_grip.set_ylabel("Gripper Value", fontsize=11)
    ax_grip.set_title("Gripper State (0=close, 1=open)", fontsize=12, fontweight="bold")
    ax_grip.set_ylim(-0.1, 1.15)
    ax_grip.legend(loc="upper right", fontsize=9)
    ax_grip.grid(True, alpha=0.3, linestyle="--")

    if highlight_steps:
        for hs in highlight_steps:
            ax_grip.axvline(x=hs, color="orange", linestyle=":", linewidth=1.0, alpha=0.6)

    # --- 下半部分：动作统计信息 ---
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_stats.axis("off")

    # 统计信息表格
    stats_text = "Action Statistics\n" + "-" * 50 + "\n"
    stats_text += f"{'Name':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}\n"
    stats_text += "-" * 50 + "\n"
    for i, name in enumerate(action_names):
        vals = actions[:, i]
        stats_text += (
            f"{name:<12} {np.mean(vals):>+8.4f} {np.std(vals):>8.4f} "
            f"{np.min(vals):>+8.4f} {np.max(vals):>+8.4f}\n"
        )
    stats_text += "-" * 50

    ax_stats.text(
        0.05, 0.95, stats_text,
        transform=ax_stats.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    # --- 下半部分右侧：动作幅度分布 ---
    ax_hist = fig.add_subplot(gs[2, 1])
    action_magnitudes = np.linalg.norm(actions[:, :6], axis=1)
    ax_hist.hist(action_magnitudes, bins=40, color="coral", alpha=0.7, edgecolor="white")
    ax_hist.axvline(
        x=np.mean(action_magnitudes), color="red", linestyle="--",
        linewidth=1.5, label=f"Mean: {np.mean(action_magnitudes):.4f}",
    )
    ax_hist.set_xlabel("Action Magnitude (6-DOF L2 norm)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title("Action Magnitude Distribution", fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3, linestyle="--")

    # 如果有阶段信息，添加阶段背景色
    if "phase_names" in data and data["phase_names"]:
        phase_list = data["phase_names"]
        unique_phases = list(dict.fromkeys(phase_list))  # 去重保序
        phase_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_phases), 1)))
        phase_color_map = {p: c for p, c in zip(unique_phases, phase_colors)}

        # 在位置图上标注阶段
        current_phase = phase_list[0]
        start = 0
        for idx in range(1, len(phase_list) + 1):
            if idx == len(phase_list) or phase_list[idx] != current_phase:
                color = phase_color_map.get(current_phase, (0.8, 0.8, 0.8, 0.1))
                ax_pos.axvspan(start, idx, alpha=0.06, color=color[:3])
                ax_pos.text(
                    (start + idx) / 2, ax_pos.get_ylim()[1] * 0.95,
                    current_phase, fontsize=7, ha="center", va="top",
                    color=color[:3], fontweight="bold",
                )
                if idx < len(phase_list):
                    current_phase = phase_list[idx]
                start = idx

    # 保存
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "action_trajectory.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Action trajectory saved to: {save_path}")


# ===========================================================================
# 模式 2: 注意力可视化
# ===========================================================================

def plot_attention(
    image_path: str,
    output_dir: str = "results",
    model_name: str = "openai/clip-vit-base-patch32",
):
    """
    可视化 CLIP 视觉编码器的注意力图（attention rollout）。

    通过提取 CLIP Vision Transformer 各层的 attention weights，
    使用 attention rollout 方法生成热力图，叠加在原图上。

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        model_name: HuggingFace 模型名称（需支持 CLIP 架构）
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
    except ImportError as e:
        print(f"ERROR: Missing dependency for attention visualization: {e}")
        print("  Install with: pip install torch transformers pillow")
        return

    print(f"Loading CLIP model: {model_name}...")
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    except Exception as e:
        print(f"ERROR: Failed to load model '{model_name}': {e}")
        print("  Try: --model_name openai/clip-vit-base-patch16")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    print(f"Image size: {image.size}")
    print(f"Running attention extraction...")

    # 提取各层 attention
    attn_maps = []

    def hook_fn(module, input, output):
        """Hook to capture attention weights."""
        # output: (batch, num_heads, seq_len, seq_len)
        # 取 [CLS] token 对所有 patch tokens 的 attention
        attn = output[1]  # cross_attention or self_attention
        if attn is not None:
            attn_maps.append(attn.detach().cpu())

    # 注册 hooks
    hooks = []
    vision_model = model.vision_model
    for layer in vision_model.encoder.layers:
        h = layer.self_attn.register_forward_hook(hook_fn)
        hooks.append(h)

    # 前向传播
    with torch.no_grad():
        _ = model.get_image_features(**inputs)

    # 移除 hooks
    for h in hooks:
        h.remove()

    if not attn_maps:
        print("WARNING: No attention maps captured. Trying alternative extraction...")
        # 使用 Grad-CAM 风格的简化方法
        plot_attention_simple(image, image_path, output_dir)
        return

    print(f"Captured {len(attn_maps)} attention layers")

    # Attention Rollout: 累积所有层的 attention
    num_layers = len(attn_maps)
    num_patches = attn_maps[0].shape[-1] - 1  # 减去 [CLS] token

    # 获取 patch grid 尺寸（取决于模型配置）
    patch_size = model.vision_model.config.patch_size
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    img = np.array(image)
    h, w = img.shape[:2]
    grid_h = h // patch_size[0]
    grid_w = w // patch_size[1]

    # Attention rollout 算法
    rollout = torch.eye(attn_maps[0].shape[-1])
    for attn in attn_maps:
        # 取 [CLS] 行作为权重，然后平均多头的 attention
        attn_head_avg = attn.mean(dim=1)  # (batch, seq_len, seq_len)
        attn_cls = attn_head_avg[0, 0:1, :]  # (1, seq_len) -- CLS 对所有 token 的 attention

        # 归一化
        attn_cls = attn_cls / attn_cls.sum(dim=-1, keepdim=True)

        # 累积
        rollout = torch.matmul(attn_cls, rollout)

    # 去掉 [CLS] token，保留 patch tokens
    attn_rollout = rollout[0, 1:].reshape(grid_h, grid_w).numpy()

    # 归一化到 [0, 1]
    attn_rollout = (attn_rollout - attn_rollout.min()) / (
        attn_rollout.max() - attn_rollout.min() + 1e-8
    )

    # 上采样到原图尺寸
    from PIL import Image as PILImage
    attn_resized = PILImage.fromarray((attn_rollout * 255).astype(np.uint8)).resize(
        (w, h), PILImage.BILINEAR
    )
    attn_resized = np.array(attn_resized).astype(np.float32) / 255.0

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"CLIP Attention Visualization  |  Model: {model_name}",
        fontsize=14, fontweight="bold",
    )

    # 原图
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # 注意力热力图
    im = axes[1].imshow(attn_resized, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Attention Rollout Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 叠加
    axes[2].imshow(img)
    axes[2].imshow(
        attn_resized, cmap="jet", alpha=0.5, interpolation="bilinear"
    )
    axes[2].set_title("Attention Overlay", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "attention_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention visualization saved to: {save_path}")


def plot_attention_simple(image_pil, image_path: str, output_dir: str):
    """
    简化的注意力可视化：使用 Grad-CAM 风格的显著性图。

    当无法直接提取 CLIP attention 时的备选方案。
    """
    from PIL import Image as PILImage
    import numpy as np

    img = np.array(image_pil).astype(np.float32)

    # 简单的显著性图：基于图像梯度的边缘检测
    gray = np.mean(img, axis=2)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))

    # 上采样到相同尺寸
    saliency = np.zeros_like(gray)
    saliency[:, 1:] += grad_x
    saliency[1:, :] += grad_y
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img.astype(np.uint8))
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img.astype(np.uint8))
    axes[1].imshow(saliency, cmap="hot", alpha=0.6)
    axes[1].set_title("Saliency Map (fallback)", fontsize=12)
    axes[1].axis("off")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "attention_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saliency map (fallback) saved to: {save_path}")


# ===========================================================================
# 模式 3: 评估结果统计
# ===========================================================================

def plot_eval_results(
    eval_path: str,
    output_dir: str = "results",
    metric_key: str = "success_rate",
):
    """
    从评估日志 JSON 中读取结果，绘制各任务成功率柱状图。

    支持的 JSON 格式：
    {
        "tasks": [
            {"task_name": "pick_up_cup", "success_rate": 0.85, "num_episodes": 20, ...},
            ...
        ],
        "overall": {"success_rate": 0.72, ...}
    }

    也支持简化格式：
    {
        "pick_up_cup": {"success_rate": 0.85},
        "pour_water": {"success_rate": 0.60},
        ...
    }

    Args:
        eval_path: 评估结果 JSON 文件路径
        output_dir: 输出目录
        metric_key: 要绘制的指标键名
    """
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 解析数据格式
    if "tasks" in data:
        tasks = data["tasks"]
        task_names = [t.get("task_name", t.get("name", f"task_{i}")) for i, t in enumerate(tasks)]
        success_rates = [t.get(metric_key, t.get("success_rate", 0)) for t in tasks]
        num_episodes = [t.get("num_episodes", t.get("n_episodes", 0)) for t in tasks]
    else:
        # 简化格式
        task_names = list(data.keys())
        success_rates = []
        num_episodes = []
        for key in task_names:
            val = data[key]
            if isinstance(val, dict):
                success_rates.append(val.get(metric_key, val.get("success_rate", 0)))
                num_episodes.append(val.get("num_episodes", val.get("n_episodes", 0)))
            else:
                success_rates.append(float(val))
                num_episodes.append(0)

    print(f"Loaded {len(task_names)} tasks from {eval_path}")

    success_rates = np.array(success_rates)
    num_episodes = np.array(num_episodes)

    # 颜色编码：根据成功率
    bar_colors = []
    for sr in success_rates:
        if sr >= 0.8:
            bar_colors.append("#27ae60")   # 绿色 - 高成功率
        elif sr >= 0.5:
            bar_colors.append("#f39c12")   # 橙色 - 中等
        else:
            bar_colors.append("#e74c3c")   # 红色 - 低成功率

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "VLA Evaluation Results",
        fontsize=14, fontweight="bold",
    )

    # --- 左侧：成功率柱状图 ---
    x_pos = np.arange(len(task_names))
    bars = axes[0].bar(x_pos, success_rates * 100, color=bar_colors, alpha=0.85,
                       edgecolor="white", linewidth=1.2)

    # 在柱子上方标注数值
    for bar, sr, n_ep in zip(bars, success_rates, num_episodes):
        height = bar.get_height()
        label = f"{sr*100:.1f}%"
        if n_ep > 0:
            label += f"\n(n={n_ep})"
        axes[0].text(
            bar.get_x() + bar.get_width() / 2., height + 1,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # 平均线
    avg_sr = np.mean(success_rates)
    axes[0].axhline(
        y=avg_sr * 100, color="navy", linestyle="--", linewidth=1.5,
        label=f"Average: {avg_sr*100:.1f}%",
    )

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(task_names, rotation=35, ha="right", fontsize=9)
    axes[0].set_ylabel("Success Rate (%)", fontsize=12)
    axes[0].set_title("Per-Task Success Rate", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, min(110, max(success_rates * 100) + 20))
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y", linestyle="--")

    # --- 右侧：排名和分布 ---
    sorted_idx = np.argsort(success_rates)[::-1]
    sorted_names = [task_names[i] for i in sorted_idx]
    sorted_rates = success_rates[sorted_idx]

    y_pos = np.arange(len(sorted_names))
    axes[1].barh(
        y_pos, sorted_rates * 100,
        color=[bar_colors[i] for i in sorted_idx],
        alpha=0.85, edgecolor="white", linewidth=1.0,
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(sorted_names, fontsize=9)
    axes[1].set_xlabel("Success Rate (%)", fontsize=12)
    axes[1].set_title("Task Ranking", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, 110)
    axes[1].grid(True, alpha=0.3, axis="x", linestyle="--")

    # 标注数值
    for i, (name, rate) in enumerate(zip(sorted_names, sorted_rates)):
        axes[1].text(
            rate * 100 + 1, i, f"{rate*100:.1f}%",
            va="center", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "eval_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Evaluation results saved to: {save_path}")


# ===========================================================================
# 模式 4: 图像-动作对比
# ===========================================================================

def plot_image_action(
    traj_path: str,
    output_dir: str = "results",
    num_samples: int = 8,
    sample_indices: Optional[List[int]] = None,
):
    """
    并排显示输入图像和对应的动作向量。

    从 episode JSON 中选取若干关键帧，每帧显示：
    - 观测图像（如果有保存）
    - 对应的 7-DOF 动作向量（以条形图表示）

    Args:
        traj_path: episode JSON 文件路径
        output_dir: 输出目录
        num_samples: 采样数量
        sample_indices: 指定采样的步骤索引（覆盖 num_samples）
    """
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = np.array(data["actions"])
    num_steps = len(actions)
    action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    # 选择采样的步骤
    if sample_indices is not None:
        indices = [min(i, num_steps - 1) for i in sample_indices]
    else:
        indices = np.linspace(0, num_steps - 1, num_samples, dtype=int).tolist()

    print(f"Creating image-action comparison for {len(indices)} samples")

    # 计算布局
    n = len(indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols * 2  # 每个样本占 2 行（图像 + 动作条形图）

    fig = plt.figure(figsize=(5 * cols, 3 * rows))
    fig.suptitle(
        f"Image-Action Comparison  |  {n} samples from {num_steps} steps",
        fontsize=14, fontweight="bold",
    )

    for plot_idx, step_idx in enumerate(indices):
        row_block = plot_idx // cols
        col_idx = plot_idx % cols

        # 上方：图像占位（如果 JSON 中没有图像数据，则用状态信息代替）
        ax_img = fig.add_subplot(rows, cols, row_block * 2 * cols + col_idx + 1)
        ax_img.set_facecolor("#f0f0f0")

        # 尝试绘制 gripper/cube 位置图作为替代
        if "gripper_positions" in data and "cube_positions" in data:
            grip_pos = np.array(data["gripper_positions"][step_idx])
            cube_pos = np.array(data["cube_positions"][step_idx])

            # 绘制简化的俯视图
            ax_img.plot(grip_pos[0], grip_pos[1], "bo", markersize=12, label="Gripper")
            ax_img.plot(cube_pos[0], cube_pos[1], "rs", markersize=10, label="Cube")

            # 连线表示距离
            ax_img.plot(
                [grip_pos[0], cube_pos[0]], [grip_pos[1], cube_pos[1]],
                "g--", linewidth=1, alpha=0.5,
            )
            dist = np.linalg.norm(grip_pos[:2] - cube_pos[:2])
            ax_img.set_title(
                f"Step {step_idx} | XY View | Dist: {dist:.3f}m",
                fontsize=9,
            )
            ax_img.set_xlabel("X (m)", fontsize=8)
            ax_img.set_ylabel("Y (m)", fontsize=8)
            ax_img.legend(fontsize=7, loc="upper right")
            ax_img.set_aspect("equal")
            ax_img.grid(True, alpha=0.3)
        else:
            ax_img.text(
                0.5, 0.5, f"Step {step_idx}\n(No image data)",
                ha="center", va="center", fontsize=12,
                transform=ax_img.transAxes,
            )
            ax_img.set_title(f"Step {step_idx}", fontsize=10)
            ax_img.set_xticks([])
            ax_img.set_yticks([])

        # 下方：动作向量条形图
        ax_act = fig.add_subplot(rows, cols, (row_block * 2 + 1) * cols + col_idx + 1)
        action = actions[step_idx]
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 7))

        bars = ax_act.barh(
            range(7), action, color=colors, edgecolor="white", linewidth=0.8, height=0.7,
        )
        ax_act.set_yticks(range(7))
        ax_act.set_yticklabels(action_names, fontsize=8)
        ax_act.set_xlabel("Value", fontsize=8)
        ax_act.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
        ax_act.grid(True, alpha=0.2, axis="x", linestyle="--")

        # 标注数值
        for bar, val in zip(bars, action):
            x_pos = bar.get_width()
            offset = 0.02 if x_pos >= 0 else -0.02
            ha = "left" if x_pos >= 0 else "right"
            ax_act.text(
                x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", ha=ha, va="center", fontsize=7,
            )

    plt.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "image_action_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Image-action comparison saved to: {save_path}")


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLA Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制动作轨迹
  python visualize_vla.py --mode action_trajectory --traj_path results/episode_0.json

  # 绘制评估结果
  python visualize_vla.py --mode eval_results --eval_path results/eval_results.json

  # CLIP 注意力可视化
  python visualize_vla.py --mode attention --image_path scene.jpg

  # 图像-动作对比（指定步骤）
  python visualize_vla.py --mode image_action --traj_path results/episode_0.json --sample_indices 0 50 100 150
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["action_trajectory", "attention", "eval_results", "image_action"],
        help="可视化模式",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default=None,
        help="Episode JSON 轨迹文件路径（用于 action_trajectory 和 image_action 模式）",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        help="评估结果 JSON 文件路径（用于 eval_results 模式）",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="输入图像路径（用于 attention 模式）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="输出图片目录 (default: results/)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP 模型名称（用于 attention 模式）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="图像-动作对比采样数量 (default: 8)",
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="图像-动作对比指定步骤索引",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VLA Visualization Tool")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    if args.mode == "action_trajectory":
        if not args.traj_path:
            parser.error("action_trajectory mode requires --traj_path")
        if not os.path.exists(args.traj_path):
            print(f"ERROR: File not found: {args.traj_path}")
            sys.exit(1)
        plot_action_trajectory(
            traj_path=args.traj_path,
            output_dir=args.output_dir,
        )

    elif args.mode == "attention":
        if not args.image_path:
            parser.error("attention mode requires --image_path")
        if not os.path.exists(args.image_path):
            print(f"ERROR: File not found: {args.image_path}")
            sys.exit(1)
        plot_attention(
            image_path=args.image_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
        )

    elif args.mode == "eval_results":
        if not args.eval_path:
            parser.error("eval_results mode requires --eval_path")
        if not os.path.exists(args.eval_path):
            print(f"ERROR: File not found: {args.eval_path}")
            sys.exit(1)
        plot_eval_results(
            eval_path=args.eval_path,
            output_dir=args.output_dir,
        )

    elif args.mode == "image_action":
        if not args.traj_path:
            parser.error("image_action mode requires --traj_path")
        if not os.path.exists(args.traj_path):
            print(f"ERROR: File not found: {args.traj_path}")
            sys.exit(1)
        plot_image_action(
            traj_path=args.traj_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            sample_indices=args.sample_indices,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()