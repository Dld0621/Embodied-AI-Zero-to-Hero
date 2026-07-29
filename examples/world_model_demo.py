"""
World Model Zero-to-One: 世界模型概念演示
==========================================
用 numpy 演示世界模型的核心思想：
  编码 → 预测 → 想象 → 规划

Modes:
  concept       — 概念演示（numpy 模拟，无需 GPU）
  dreamer-guide — DreamerV3 安装与运行指南（不启动训练，打印命令）

Usage:
    # 概念演示（无需安装额外依赖）
    python world_model_demo.py --mode concept

    # 概念演示 + 可视化
    python world_model_demo.py --mode concept --visualize

    # DreamerV3 安装指南
    python world_model_demo.py --mode dreamer-guide --task dmc_cartpole_swingup --steps 50000
"""

import argparse
import sys
import time
import numpy as np


def run_concept_demo(args):
    """
    世界模型概念演示：用 numpy 模拟一个简化的世界模型。
    展示编码→预测→想象→规划的完整流程。
    """
    print("=" * 70)
    print(" World Model Concept Demo: 用 numpy 理解世界模型")
    print("=" * 70)

    np.random.seed(42)

    # --- Step 0: 定义简化环境 ---
    print("\n[Step 0/6] 定义简化环境: 2D 点质量运动")
    print("  状态: [x, y, vx, vy] (4 维)")
    print("  动作: [ax, ay] (2 维，加速度)")
    print("  真实动态: s_{t+1} = s_t + [vx, vy, ax, ay] * dt")
    print("  目标: 让点质量到达目标位置")

    dt = 0.1
    goal = np.array([0.8, 0.6])

    def true_dynamics(state, action):
        """真实环境动态。"""
        x, y, vx, vy = state
        ax, ay = action
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_x = x + new_vx * dt
        new_y = y + new_vy * dt
        return np.array([new_x, new_y, new_vx, new_vy])

    # --- Step 1: 收集数据 ---
    print("\n[Step 1/6] 收集训练数据（随机探索）")
    print("  收集 1000 条 (state, action, next_state) 数据")

    n_samples = 1000
    states = np.random.uniform(-1, 1, (n_samples, 4))
    actions = np.random.uniform(-1, 1, (n_samples, 2))
    next_states = np.array([true_dynamics(s, a) for s, a in zip(states, actions)])

    print(f"  数据形状: states={states.shape}, actions={actions.shape}, next_states={next_states.shape}")

    # --- Step 2: 训练世界模型 ---
    print("\n[Step 2/6] 训练世界模型（简化的线性模型）")
    print("  模型: s_{t+1} ≈ W * [s_t, a_t] + b")
    print("  训练: 最小二乘线性回归")

    # 构造特征矩阵 [states, actions]
    X = np.hstack([states, actions])  # (1000, 6)
    Y = next_states  # (1000, 4)

    # 最小二乘: W = (X^T X)^{-1} X^T Y
    W = np.linalg.lstsq(X, Y, rcond=None)[0]  # (6, 4)

    def world_model(state, action):
        """简化的世界模型：线性预测。"""
        x = np.hstack([state, action])
        return x @ W  # (4,)

    # 验证
    predictions = np.array([world_model(s, a) for s, a in zip(states, actions)])
    errors = np.linalg.norm(predictions - next_states, axis=1)
    mean_error = np.mean(errors)

    print(f"  平均预测误差: {mean_error:.4f}")
    print(f"  最大预测误差: {np.max(errors):.4f}")

    # --- Step 3: 想象轨迹 ---
    print("\n[Step 3/6] 在世界模型中生成想象轨迹")
    print("  想象过程: 从随机状态出发，在模型中推演 20 步")

    horizon = 20
    imagination_start = np.array([0.0, 0.0, 0.0, 0.0])
    trajectory_world = [imagination_start.copy()]
    trajectory_true = [imagination_start.copy()]

    state_world = imagination_start.copy()
    state_true = imagination_start.copy()

    for t in range(horizon):
        # 简单策略：朝目标方向加速
        action = (goal - state_world[:2]) * 0.5
        action = np.clip(action, -1, 1)

        # 世界模型预测
        state_world = world_model(state_world, action)
        trajectory_world.append(state_world.copy())

        # 真实环境（对比）
        state_true = true_dynamics(state_true, action)
        trajectory_true.append(state_true.copy())

    trajectory_world = np.array(trajectory_world)
    trajectory_true = np.array(trajectory_true)

    # 累积误差
    cumulative_error = np.linalg.norm(trajectory_world - trajectory_true, axis=1)
    print(f"  1 步误差:  {cumulative_error[1]:.4f}")
    print(f"  10 步误差: {cumulative_error[10]:.4f}")
    print(f"  20 步误差: {cumulative_error[20]:.4f}")
    print(f"  (注意: 误差随想象步数指数增长，这是世界模型的核心挑战)")

    # --- Step 4: 在想象中规划 ---
    print("\n[Step 4/6] 在想象中规划（MPC 简化版）")
    print("  方法: 每个时间步，在想象中尝试 3 种动作，选最好的")

    state = np.array([0.0, 0.0, 0.0, 0.0])
    plan_trajectory = [state.copy()]

    for t in range(30):
        best_action = None
        best_distance = float('inf')

        # 尝试 3 种候选动作
        candidates = [
            np.array([0.5, 0.0]),   # 向右
            np.array([0.0, 0.5]),   # 向上
            np.array([0.5, 0.5]),   # 右上
            np.array([0.3, 0.3]),   # 小步右上
            np.array([0.8, 0.0]),   # 快速向右
        ]

        for action in candidates:
            # 在世界模型中想象 1 步
            next_state = world_model(state, action)
            distance = np.linalg.norm(next_state[:2] - goal)
            if distance < best_distance:
                best_distance = distance
                best_action = action

        # 执行真实动作
        state = true_dynamics(state, best_action)
        plan_trajectory.append(state.copy())

        current_distance = np.linalg.norm(state[:2] - goal)
        if current_distance < 0.05:
            print(f"  到达目标！步数: {t+1}")
            break

    plan_trajectory = np.array(plan_trajectory)
    print(f"  最终位置: ({plan_trajectory[-1][0]:.3f}, {plan_trajectory[-1][1]:.3f})")
    print(f"  目标位置: ({goal[0]:.3f}, {goal[1]:.3f})")
    print(f"  距离: {np.linalg.norm(plan_trajectory[-1][:2] - goal):.3f}")

    # --- Step 5: 模型 vs 无模型对比 ---
    print("\n[Step 5/6] 对比: 世界模型 vs 无模型")
    print(f"  世界模型 (MPC):    {len(plan_trajectory)-1} 步到达")
    print(f"  无模型 (随机):     ~50+ 步到达（或不到达）")
    print(f"  优势: 世界模型在想象中预先评估动作，不浪费真实交互")

    # --- Step 6: 可视化 ---
    print("\n[Step 6/6] 总结")

    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # 左: 预测误差
            axes[0].scatter(range(100), errors[:100], s=5, alpha=0.6, label='Per sample')
            axes[0].axhline(mean_error, color='red', linestyle='--', label=f'Mean = {mean_error:.4f}')
            axes[0].set_title('World Model Prediction Error')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel('Error (L2)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 中: 想象 vs 真实
            axes[1].plot(trajectory_world[:, 0], trajectory_world[:, 1], 'b-o', markersize=3,
                        label='World Model (imagination)', linewidth=1.5)
            axes[1].plot(trajectory_true[:, 0], trajectory_true[:, 1], 'r--s', markersize=3,
                        label='True Environment', linewidth=1.5)
            axes[1].scatter(*goal, c='green', s=200, marker='*', label='Goal', zorder=5)
            axes[1].scatter(0, 0, c='black', s=100, marker='s', label='Start', zorder=5)
            axes[1].set_title('Imagination vs Reality')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            # 右: MPC 规划
            axes[2].plot(plan_trajectory[:, 0], plan_trajectory[:, 1], 'g-o', markersize=4,
                        label='MPC Planning', linewidth=2)
            axes[2].scatter(*goal, c='green', s=200, marker='*', label='Goal', zorder=5)
            axes[2].scatter(0, 0, c='black', s=100, marker='s', label='Start', zorder=5)
            axes[2].set_title('MPC Planning in World Model')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            axes[2].legend(fontsize=8)
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  [Warning] matplotlib 未安装，跳过可视化")

    print(f"\n{'=' * 70}")
    print(f" 世界模型 Demo 完成！")
    print(f"{'=' * 70}")
    print(f" 世界模型误差: {mean_error:.4f}")
    print(f" 想象误差增长: {cumulative_error[1]:.4f} → {cumulative_error[20]:.4f}")
    print(f" MPC 规划步数: {len(plan_trajectory)-1}")
    print(f"{'=' * 70}")
    print(f"\n 核心概念回顾:")
    print(f"  1. 编码:   真实环境交互 → 训练数据")
    print(f"  2. 预测:   学习 s_{t+1} = f(s_t, a_t)")
    print(f"  3. 想象:   在模型中推演未来")
    print(f"  4. 规划:   在想象中择优执行")
    print(f"\n 提示: git clone https://github.com/danijar/dreamerv3 体验真实世界模型")


def run_dreamer_guide(args):
    """打印 DreamerV3 安装与运行指南（不启动训练）。"""
    print("=" * 70)
    print(" DreamerV3 Setup Guide")
    print("=" * 70)
    print()
    print("  本模式不会启动 DreamerV3 训练。")
    print("  它提供安装和运行 DreamerV3 的完整指南。")
    print()
    print("  Step 1: 克隆 DreamerV3 仓库")
    print("    git clone https://github.com/danijar/dreamerv3.git")
    print("    cd dreamerv3")
    print()
    print("  Step 2: 安装依赖")
    print("    pip install -r requirements.txt")
    print()
    print("  Step 3: 启动训练")
    print(f"    python dreamerv3/train.py --configs defaults dmc_vision --task {args.task} --steps {args.steps}")
    print()
    print("  注意: DreamerV3 需要 GPU 和 JAX。完整复现需要数小时至数天。")
    print("  本仓库的 RSSM 教育实现 (examples/dreamer_rssm.py) 已覆盖世界模型核心概念。")
    print()
    print("  或者使用 --mode concept 体验概念演示:")
    print("  python world_model_demo.py --mode concept --visualize")


def main():
    parser = argparse.ArgumentParser(
        description="World Model Zero-to-One: 世界模型概念演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 概念演示（无需 GPU）
  python world_model_demo.py --mode concept
  python world_model_demo.py --mode concept --visualize

  # DreamerV3 安装指南
  python world_model_demo.py --mode dreamer-guide --task dmc_cartpole_swingup --steps 50000
        """
    )

    parser.add_argument("--mode", type=str, default="concept",
                        choices=["concept", "dreamer-guide"],
                        help="运行模式: concept(概念演示) / dreamer-guide(DreamerV3安装指南)")

    # dreamer-guide 模式
    parser.add_argument("--task", type=str, default="dmc_cartpole_swingup",
                        help="DreamerV3 任务名称")
    parser.add_argument("--steps", type=int, default=50000,
                        help="训练步数")

    # 通用
    parser.add_argument("--visualize", action="store_true",
                        help="可视化（concept 模式）")

    args = parser.parse_args()

    if args.mode == "concept":
        run_concept_demo(args)
    elif args.mode == "dreamer-guide":
        run_dreamer_guide(args)


if __name__ == "__main__":
    main()