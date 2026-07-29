"""
VLA Zero-to-One: SmolVLA 推理演示
==================================
使用 HuggingFace LeRobot 的 SmolVLA 模型，完成：
  语言指令 + 图像 + 状态 → 机器人动作

Modes:
  synthetic  — 合成数据（无需 GPU，展示 API 用法）
  aloha      — 真实 ALOHA 数据集（需要 GPU + 网络下载）

Usage:
    # 合成数据演示（无需 GPU，展示 API）
    python vla_demo.py --mode synthetic --task "pick up the apple"

    # ALOHA 真实数据推理（需要 GPU）
    python vla_demo.py --mode aloha --episode 0

    # 可视化动作序列
    python vla_demo.py --mode synthetic --visualize

Learning Paths:
  A. CPU入门:  python vla_demo.py --mode synthetic --task "pick up the apple"
  B. 8GB GPU实践: python vla_demo.py --mode aloha --episode 0
  C. 24GB+ GPU研究: 参见 docs/13-vla-zero-to-one.md 微调章节

Note:
    - --mode aloha 需要 GPU (>=4GB) 和网络连接（下载模型 + 数据集）
    - --mode synthetic 可在 CPU 上运行，仅展示 API 调用流程
"""

import argparse
import sys
import time
from typing import List
import numpy as np


def check_lerobot():
    """检查 lerobot 是否安装。"""
    try:
        import lerobot
        return True, lerobot.__version__
    except ImportError:
        return False, None


def run_synthetic_demo(args):
    """
    合成数据演示：展示 SmolVLA 的完整 API 调用流程。
    使用随机数据，不需要 GPU，帮助理解 pipeline。
    """
    print("=" * 70)
    print(" VLA Synthetic Demo: SmolVLA API Walkthrough")
    print("=" * 70)

    has_lerobot, version = check_lerobot()

    if not has_lerobot:
        if getattr(args, 'strict', False):
            print("\n[Strict Mode Error] lerobot 未安装，严格模式下不允许 fallback。")
            print("  请先安装: pip install lerobot")
            sys.exit(1)
        print("\n[Info] lerobot 未安装，使用 numpy 模拟 VLA 输出。")
        print("  [Fallback] lerobot not installed — running numpy simulation")
        print("  Execution Mode:  NUMPY_SIMULATION")
        print("  Pretrained Model Executed: NO")
        print("  Result Validity: API FORMAT DEMO ONLY")
        print("  安装: pip install lerobot")
        print("  安装后可运行 --mode aloha 使用真实模型。\n")
        run_numpy_simulation(args)
        return

    print(f"\n[Info] lerobot v{version} 已安装")
    print("  注意: 合成模式使用随机数据，输出无实际意义")
    print("  运行 --mode aloha 使用真实模型推理\n")

    try:
        import torch
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as e:
        if getattr(args, 'strict', False):
            print(f"[Strict Mode Error] 无法导入 SmolVLAPolicy: {e}")
            print("  lerobot 版本可能不包含 SmolVLAPolicy，请更新: pip install -U lerobot")
            print("  严格模式下不允许 fallback，退出。")
            sys.exit(1)
        print(f"[Warning] 无法导入 SmolVLAPolicy: {e}")
        print("  lerobot 版本可能不包含 SmolVLAPolicy，请更新: pip install -U lerobot")
        run_numpy_simulation(args)
        return

    # --- Step 1: 加载模型 ---
    print("[Step 1/4] 加载 SmolVLA 模型")
    print("  模型: lerobot/smolvla_base")
    print("  参数量: 450M")
    print("  动作空间: 14-DoF (ALOHA 双臂)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    if device == "cpu":
        print("  [Warning] CPU 模式，推理会很慢。推荐使用 GPU。")

    try:
        model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        model.to(device)
        model.eval()
        print("  [OK] 模型加载成功")
    except Exception as e:
        if getattr(args, 'strict', False):
            print(f"  [Strict Mode Error] 模型加载失败: {e}")
            print("  可能原因: 网络问题、磁盘空间不足、lerobot 版本不兼容")
            print("  严格模式下不允许 fallback，退出。")
            sys.exit(1)
        print(f"  [Error] 模型加载失败: {e}")
        print("  可能原因: 网络问题、磁盘空间不足、lerobot 版本不兼容")
        print("  回退到 numpy 模拟...")
        run_numpy_simulation(args)
        return

    # --- Step 2: 准备观测 ---
    print(f"\n[Step 2/4] 准备观测 (合成随机数据)")
    print(f"  语言指令: '{args.task}'")

    observation = {
        "observation.images.front": torch.randn(1, 3, 256, 256, device=device),
        "observation.images.left_wrist": torch.randn(1, 3, 256, 256, device=device),
        "observation.images.right_wrist": torch.randn(1, 3, 256, 256, device=device),
        "observation.state": torch.randn(1, 14, device=device),
        "task": args.task,
    }

    print(f"  输入形状:")
    print(f"    front camera:  {list(observation['observation.images.front'].shape)}")
    print(f"    left wrist:    {list(observation['observation.images.left_wrist'].shape)}")
    print(f"    right wrist:   {list(observation['observation.images.right_wrist'].shape)}")
    print(f"    robot state:   {list(observation['observation.state'].shape)}")

    # --- Step 3: VLA 推理 ---
    # Note: This is repeated single-step inference, NOT true action chunking.
    # True action chunking generates [B, T, action_dim] in one forward pass.
    # See docs/13-vla-zero-to-one.md §7.3 for the distinction.
    print(f"\n[Step 3/4] VLA 推理 (repeated single-step, {args.chunk_size} iterations)")

    n_steps = args.chunk_size if args.chunk_size > 1 else 1
    start_time = time.time()

    actions = []
    with torch.no_grad():
        for step in range(n_steps):
            action = model.select_action(observation)
            actions.append(action.cpu().numpy())
            if step == 0 or (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{n_steps}: action range [{action.min():.4f}, {action.max():.4f}]")

    elapsed = time.time() - start_time
    actions = np.array(actions).squeeze()  # (n_steps, 14) or (14,)

    if actions.ndim == 1:
        actions = actions[np.newaxis, ...]

    print(f"\n  推理耗时: {elapsed:.3f}s ({elapsed/n_steps*1000:.1f} ms/步)")
    print(f"  输出形状: {actions.shape}")

    # --- Step 4: 解析动作 ---
    print(f"\n[Step 4/4] 解析动作")
    action_names = [
        "L_x", "L_y", "L_z", "L_roll", "L_pitch", "L_yaw", "L_gripper",
        "R_x", "R_y", "R_z", "R_roll", "R_pitch", "R_yaw", "R_gripper",
    ]

    print(f"  {'关节':12s} {'动作值':>10s}")
    print(f"  {'-'*12} {'-'*10}")
    for name, val in zip(action_names, actions[0]):
        print(f"  {name:12s} {val:10.4f}")

    # --- 可视化 ---
    if args.visualize and len(actions) > 1:
        visualize_actions(actions, action_names, args.task)

    # --- 总结 ---
    print(f"\n{'=' * 70}")
    print(f" Pipeline 完成！")
    print(f"{'=' * 70}")
    print(f" 模型:     SmolVLA 450M")
    print(f" 指令:     {args.task}")
    print(f" 动作维度: {actions.shape[1]} (双臂各 7-DoF)")
    print(f" 推理速度: {elapsed/n_steps*1000:.1f} ms/步")
    print(f"{'=' * 70}")


def run_numpy_simulation(args):
    """用 numpy 模拟 VLA 推理流程（无 GPU / 无 lerobot 时的 fallback）。"""

    print("\n[Step 1/4] 模拟 SmolVLA 模型")
    print("  模型: SmolVLA 450M (numpy 模拟)")

    np.random.seed(42)

    print(f"\n[Step 2/4] 准备观测 (合成数据)")
    print(f"  语言指令: '{args.task}'")

    # 模拟观测
    front_image = np.random.randn(3, 256, 256).astype(np.float32)
    state = np.random.randn(14).astype(np.float32)

    print(f"  输入形状:")
    print(f"    front camera:  (3, 256, 256)")
    print(f"    robot state:   (14,)")
    print(f"    task:          '{args.task}'")

    print(f"\n[Step 3/4] 模拟 VLA 推理")

    n_steps = args.chunk_size if args.chunk_size > 1 else 5
    start_time = time.time()

    # 模拟 Flow Matching 采样（简化版 ODE 积分）
    action_dim = 14
    actions = np.zeros((n_steps, action_dim))

    for step in range(n_steps):
        # 初始噪声
        x = np.random.randn(action_dim).astype(np.float32)

        # 模拟去噪（简化：直接投影到合理范围）
        for t in range(5):  # 5 步 ODE
            dt = 0.2
            # 模拟 velocity field: 推向目标动作
            target = np.array([
                0.01, 0.02, -0.01, 0.0, 0.0, 0.0, 0.5,   # 左臂
                0.0,  0.0,  0.0,   0.0, 0.0, 0.0, 0.8,    # 右臂
            ], dtype=np.float32)
            velocity = (target - x) * 0.5
            x = x + velocity * dt

        actions[step] = x

    elapsed = time.time() - start_time

    print(f"  Flow Matching 采样: {n_steps} 步 × 5 ODE iterations")
    print(f"  耗时: {elapsed*1000:.1f} ms (numpy 模拟，无 GPU)")

    print(f"\n[Step 4/4] 解析动作")
    action_names = [
        "L_x", "L_y", "L_z", "L_roll", "L_pitch", "L_yaw", "L_gripper",
        "R_x", "R_y", "R_z", "R_roll", "R_pitch", "R_yaw", "R_gripper",
    ]

    print(f"  {'关节':12s} {'动作值':>10s}")
    print(f"  {'-'*12} {'-'*10}")
    for name, val in zip(action_names, actions[0]):
        print(f"  {name:12s} {val:10.4f}")

    if args.visualize and len(actions) > 1:
        visualize_actions(actions, action_names, args.task)

    # 保存
    if args.output:
        np.save(args.output, actions)
        print(f"\n[Output] 动作序列已保存到 {args.output}，形状: {actions.shape}")

    print(f"\n{'=' * 70}")
    print(f"  Execution Mode:  NUMPY_SIMULATION")
    print(f"  Pretrained Model Executed: NO")
    print(f"  Result Validity: API FORMAT DEMO ONLY — not a real VLA inference result")
    print(f"{'=' * 70}")
    print(f" 模拟完成！（安装 lerobot 后可运行真实推理）")
    print(f"{'=' * 70}")


def run_aloha_demo(args):
    """使用真实 ALOHA 数据集运行 SmolVLA 推理。"""
    print("=" * 70)
    print(" VLA ALOHA Demo: Real Data Inference")
    print("=" * 70)

    if getattr(args, 'strict', False):
        print("\n  [Strict Mode] 任何步骤失败将直接退出，不使用通用 fallback。\n")

    has_lerobot, version = check_lerobot()
    if not has_lerobot:
        print("[Error] 此模式需要安装 lerobot: pip install lerobot")
        sys.exit(1)

    import torch
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # --- Step 1: 加载模型 ---
    print(f"\n[Step 1/5] 加载 SmolVLA 模型")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Warning] CPU 模式不推荐，推理会非常慢")

    try:
        model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        model.to(device)
        model.eval()
        print(f"  [OK] 模型加载到 {device}")
    except Exception as e:
        print(f"  [Error] 模型加载失败: {e}")
        print("  可能原因: 网络问题、磁盘空间不足、lerobot 版本不兼容")
        if getattr(args, 'strict', False):
            print("  严格模式下不允许继续，退出。")
            sys.exit(1)
        print("  非严格模式，但 aloha 模式必须使用真实模型，退出。")
        sys.exit(1)

    # --- Step 2: 加载数据集 ---
    print(f"\n[Step 2/5] 加载 ALOHA 数据集")
    print(f"  数据集: lerobot/aloha_sim_transfer_cube_human")
    print(f"  (首次运行会自动下载，约 500MB)")

    try:
        dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
    except Exception as e:
        print(f"  [Error] 数据集加载失败: {e}")
        print("  请检查网络连接")
        if getattr(args, 'strict', False):
            print("  [Strict Mode] 数据集加载失败，直接退出。")
        sys.exit(1)

    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Steps:    {dataset.num_samples}")
    print(f"  Task:     {dataset.task}")

    # --- Step 3: 获取观测 ---
    print(f"\n[Step 3/5] 获取 Episode {args.episode} 的观测")

    # 找到 episode 起始索引
    episode_indices = []
    for i in range(len(dataset)):
        if dataset.episode_data_index["from"][i].item() == args.episode:
            episode_indices.append(i)
            break
    for i in range(len(dataset)):
        idx = dataset.episode_data_index["from"][i].item()
        if idx == args.episode:
            episode_indices.append(i)

    if not episode_indices:
        print(f"  [Error] Episode {args.episode} 不存在")
        sys.exit(1)

    # 取第一帧
    sample = dataset[episode_indices[0]]
    print(f"  取第 {episode_indices[0]} 帧")

    # --- Step 4: VLA 推理 ---
    print(f"\n[Step 4/5] VLA 推理")

    observation = {}
    for key in ["observation.images.front", "observation.images.left_wrist",
                 "observation.images.right_wrist", "observation.state"]:
        if key in sample:
            observation[key] = sample[key].unsqueeze(0).to(device)

    observation["task"] = dataset.task

    start_time = time.time()
    with torch.no_grad():
        action = model.select_action(observation)
    elapsed = time.time() - start_time

    print(f"  推理耗时: {elapsed*1000:.1f} ms")

    # --- Step 5: 对比预测 vs 真实 ---
    print(f"\n[Step 5/5] 对比: 预测动作 vs 真实动作")
    true_action = sample["action"].numpy()
    pred_action = action.cpu().numpy().flatten()

    action_names = [
        "L_x", "L_y", "L_z", "L_roll", "L_pitch", "L_yaw", "L_gripper",
        "R_x", "R_y", "R_z", "R_roll", "R_pitch", "R_yaw", "R_gripper",
    ]

    print(f"\n  {'关节':12s} {'预测':>10s} {'真实':>10s} {'误差':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name, pred, true in zip(action_names, pred_action, true_action):
        err = abs(pred - true)
        print(f"  {name:12s} {pred:10.4f} {true:10.4f} {err:10.4f}")

    mae = np.mean(np.abs(pred_action - true_action))
    print(f"\n  平均绝对误差 (MAE): {mae:.4f}")

    print(f"\n{'=' * 70}")
    print(f" Pipeline 完成！")
    print(f"  MAE: {mae:.4f}")
    print(f"  推理速度: {elapsed*1000:.1f} ms")
    print(f"{'=' * 70}")


def visualize_actions(actions: np.ndarray, action_names: List[str], task: str):
    """可视化动作序列。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib 未安装，跳过可视化")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 左臂
    axes[0].plot(actions[:, :7])
    axes[0].set_title(f'Left Arm Actions — "{task}"')
    axes[0].set_ylabel('Action Value')
    axes[0].legend(action_names[:7], loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 右臂
    axes[1].plot(actions[:, 7:])
    axes[1].set_title(f'Right Arm Actions — "{task}"')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Action Value')
    axes[1].legend(action_names[7:], loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="VLA Zero-to-One: SmolVLA Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合成数据演示（无需 GPU）
  python vla_demo.py --mode synthetic --task "pick up the apple"
  python vla_demo.py --mode synthetic --task "pour water" --visualize

  # ALOHA 真实数据推理（需要 GPU）
  python vla_demo.py --mode aloha --episode 0

  # 保存输出
  python vla_demo.py --mode synthetic --output actions.npy
        """
    )

    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "aloha"],
                        help="synthetic: 合成数据 CPU 演示（理解 VLA API 结构）; "
                             "aloha: 真实 SmolVLA 推理（需要 GPU + 网络下载）")
    parser.add_argument("--task", type=str, default="pick up the apple",
                        help="语言指令")
    parser.add_argument("--episode", type=int, default=0,
                        help="ALOHA episode 编号（仅 aloha 模式）")
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="动作序列长度")
    parser.add_argument("--visualize", action="store_true",
                        help="可视化动作序列")
    parser.add_argument("--output", type=str, default=None,
                        help="保存动作序列到 .npy 文件")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式：模型加载失败时直接退出，不 fallback 到 numpy 模拟")

    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_demo(args)
    elif args.mode == "aloha":
        run_aloha_demo(args)


if __name__ == "__main__":
    main()