#!/usr/bin/env python3
"""
action_representation_comparison.py -- 机器人动作表示方式对比

================================================================================
  VLA-Zero-to-Hero 教学项目 · Stage 2: 动作表示
================================================================================

本脚本用一条预定义的 7-DOF pick-and-place 演示轨迹，
对比 4 种常见的机器人动作表示方式：
  1. 关节角度 (Joint Angles) — 最直接的控制输入
  2. 末端位姿 (End-Effector Pose) — 与构型无关的表示
  3. 增量 Delta (Delta Actions) — 相对变化量
  4. 关节速度 (Joint Velocities) — 连续导数形式

每种表示都打印统计特征（范围、均值、方差），并用 4 子图可视化对比。

功能：
  - 生成模拟 7-DOF pick-and-place 轨迹（预定义，无需真实机器人）
  - 将同一条轨迹转换为 4 种表示
  - 打印每种表示的统计特征
  - 4 子图对比可视化
  - 讨论每种表示的适用场景

依赖：
  pip install numpy matplotlib

用法示例：
  python action_representation_comparison.py
  python action_representation_comparison.py --save action_comparison.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── 中文字体设置 ──────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 7-DOF 机械臂维度名称 ────────────────────────────────────────────────────
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
]
POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


# ==============================================================================
#  模拟轨迹生成
# ==============================================================================

def generate_pick_and_place_trajectory(n_steps: int = 200) -> np.ndarray:
    """
    生成一条模拟的 7-DOF pick-and-place 轨迹。

    轨迹分 4 个阶段：
      1. 接近 (approach): 机械臂从初始位置下降到物体上方
      2. 抓取 (grasp): 夹爪闭合
      3. 提升 (lift): 将物体提升到目标位置上方
      4. 放置 (place): 下降并松开夹爪

    Args:
        n_steps: 轨迹总步数

    Returns:
        np.ndarray: shape (n_steps, 7)，每行是 [q1, q2, q3, q4, q5, q6, q7]
    """
    t = np.linspace(0, 1, n_steps)
    trajectory = np.zeros((n_steps, 7))

    # 每个阶段的时间分配
    t_approach_end = 0.25
    t_grasp_end = 0.35
    t_lift_end = 0.75
    # t_place_end = 1.0

    for i, ti in enumerate(t):
        if ti < t_approach_end:
            # 阶段 1: 接近 — 关节平滑下降到预抓取位姿
            s = ti / t_approach_end  # 0→1
            # 使用 smoothstep 插值让运动更自然
            s = s * s * (3 - 2 * s)
            # 初始位姿和预抓取位姿
            q_init = np.array([0.0, -0.5, 0.3, 0.0, -1.2, 0.0, 0.0])
            q_pre_grasp = np.array([0.2, -0.8, 0.6, 0.0, -1.5, 0.0, 0.0])
            trajectory[i] = q_init + s * (q_pre_grasp - q_init)

        elif ti < t_grasp_end:
            # 阶段 2: 抓取 — 夹爪闭合（关节7 变化），其他关节微调
            s = (ti - t_approach_end) / (t_grasp_end - t_approach_end)
            s = s * s * (3 - 2 * s)
            q_pre_grasp = np.array([0.2, -0.8, 0.6, 0.0, -1.5, 0.0, 0.0])
            q_grasp = np.array([0.22, -0.82, 0.62, 0.0, -1.52, 0.0, -0.5])
            trajectory[i] = q_pre_grasp + s * (q_grasp - q_pre_grasp)

        elif ti < t_lift_end:
            # 阶段 3: 提升 — 关节运动到放置位姿上方
            s = (ti - t_grasp_end) / (t_lift_end - t_grasp_end)
            s = s * s * (3 - 2 * s)
            q_grasp = np.array([0.22, -0.82, 0.62, 0.0, -1.52, 0.0, -0.5])
            q_pre_place = np.array([-0.3, -0.3, 0.8, 0.5, -1.0, 0.3, -0.5])
            trajectory[i] = q_grasp + s * (q_pre_place - q_grasp)

        else:
            # 阶段 4: 放置 — 下降并松开夹爪
            s = (ti - t_lift_end) / (1.0 - t_lift_end)
            s = s * s * (3 - 2 * s)
            q_pre_place = np.array([-0.3, -0.3, 0.8, 0.5, -1.0, 0.3, -0.5])
            q_place = np.array([-0.1, -0.4, 0.5, 0.3, -1.2, 0.1, 0.0])
            trajectory[i] = q_pre_place + s * (q_place - q_pre_place)

    return trajectory


# ==============================================================================
#  动作表示转换函数
# ==============================================================================

def joint_angles_representation(trajectory: np.ndarray) -> np.ndarray:
    """
    表示方式 1: 关节角度（原始轨迹，直接使用）。

    这是最直接的控制输入：直接发送给关节控制器。
    优点：无信息损失，可直接执行。
    缺点：与具体构型耦合，无法跨平台迁移。
    """
    return trajectory.copy()


def end_effector_pose_representation(trajectory: np.ndarray) -> np.ndarray:
    """
    表示方式 2: 末端位姿（简化模型）。

    在真实场景中需要完整的 FK 计算，这里用简化的非线性映射模拟
    关节角度到末端位姿的转换，目的是展示末端位姿的特征。

    模拟映射：
      x, y, z — 受所有关节角度影响的非线性函数
      roll, pitch, yaw — 受后三个关节影响
      gripper — 直接取关节7（或夹爪值）

    注意：这里使用简化的映射函数仅用于教学演示。
    真实场景应使用 README.md 中的 fk_6dof() 或 robot kinematics 库。
    """
    n = trajectory.shape[0]
    ee_pose = np.zeros((n, 7))

    for i in range(n):
        q = trajectory[i]
        # 简化的 FK 映射（仅用于演示形状差异）
        ee_pose[i, 0] = 0.3 * np.cos(q[0]) + 0.5 * np.cos(q[1] + q[2]) + 0.2 * np.sin(q[3])  # x
        ee_pose[i, 1] = 0.3 * np.sin(q[0]) + 0.5 * np.sin(q[1] + q[2]) + 0.2 * np.cos(q[3])  # y
        ee_pose[i, 2] = 0.4 + 0.3 * np.cos(q[2]) + 0.1 * np.sin(q[4])  # z
        ee_pose[i, 3] = q[4] + 0.5 * q[5]  # roll
        ee_pose[i, 4] = q[5] + 0.3 * q[4]  # pitch
        ee_pose[i, 5] = q[6]  # yaw
        ee_pose[i, 6] = q[6]  # gripper（直接映射）

    return ee_pose


def delta_representation(trajectory: np.ndarray) -> np.ndarray:
    """
    表示方式 3: 增量动作（delta / relative actions）。

    每步的动作 = 当前步关节角度 - 上一步关节角度。
    优点：对坐标系不敏感，具有平移不变性，更鲁棒。
    缺点：误差会随时间累积，需要累积才能得到绝对位置。

    第一步的 delta 设为 0（因为没有前一步）。
    """
    deltas = np.diff(trajectory, axis=0)
    # 在开头补一个零行，保持长度一致
    delta_with_first = np.vstack([np.zeros((1, trajectory.shape[1])), deltas])
    return delta_with_first


def joint_velocity_representation(trajectory: np.ndarray, dt: float = 0.05) -> np.ndarray:
    """
    表示方式 4: 关节速度。

    velocity = dq / dt，是关节角度的时间导数。
    与 delta 类似但考虑了时间间隔，具有物理意义。
    优点：平滑连续，可以直接用作速度控制接口。
    缺点：需要积分才能得到位置，频率敏感。

    Args:
        trajectory: 关节角度轨迹
        dt: 时间步长（秒），假设控制频率 20Hz
    """
    velocities = np.diff(trajectory, axis=0) / dt
    velocity_with_first = np.vstack([np.zeros((1, trajectory.shape[1])), velocities])
    return velocity_with_first


# ==============================================================================
#  统计特征分析
# ==============================================================================

def compute_statistics(data: np.ndarray, name: str, dim_names: list) -> None:
    """
    打印一种表示的统计特征。

    Args:
        data: shape (n_steps, n_dims)
        name: 表示方式名称
        dim_names: 每个维度的名称列表
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  形状: {data.shape}")
    print(f"  {'维度':>12s}  {'最小值':>10s}  {'最大值':>10s}  {'均值':>10s}  {'标准差':>10s}  {'范围':>10s}")
    print(f"  {'-'*66}")

    for j in range(data.shape[1]):
        col = data[:, j]
        print(f"  {dim_names[j]:>12s}  {col.min():+10.4f}  {col.max():+10.4f}  "
              f"{col.mean():+10.4f}  {col.std():10.4f}  {col.max()-col.min():10.4f}")

    # 总体统计
    flat = data.flatten()
    print(f"  {'-'*66}")
    print(f"  {'总体':>12s}  {flat.min():+10.4f}  {flat.max():+10.4f}  "
          f"{flat.mean():+10.4f}  {flat.std():10.4f}  {flat.max()-flat.min():10.4f}")


# ==============================================================================
#  可视化
# ==============================================================================

def plot_comparison(
    trajectory: np.ndarray,
    ee_pose: np.ndarray,
    deltas: np.ndarray,
    velocities: np.ndarray,
    save_path: str = None,
):
    """
    4 子图对比展示 4 种动作表示。

    Args:
        trajectory: 关节角度轨迹
        ee_pose: 末端位姿
        deltas: 增量动作
        velocities: 关节速度
        save_path: 若指定，保存图像
    """
    n_steps = trajectory.shape[0]
    t_axis = np.arange(n_steps) * 0.05  # 假设 20Hz

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("机器人动作表示方式对比 — 7-DOF Pick-and-Place 轨迹", fontsize=16, fontweight="bold")

    configs = [
        (trajectory, JOINT_NAMES, axes[0, 0], "1. 关节角度 (Joint Angles)", "角度 (rad)"),
        (ee_pose, POSE_NAMES, axes[0, 1], "2. 末端位姿 (End-Effector Pose)", "值"),
        (deltas, JOINT_NAMES, axes[1, 0], "3. 增量动作 (Delta Actions)", "增量 (rad)"),
        (velocities, JOINT_NAMES, axes[1, 1], "4. 关节速度 (Joint Velocities)", "速度 (rad/s)"),
    ]

    # 为每个子图使用不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    for data, names, ax, title, ylabel in configs:
        for j in range(data.shape[1]):
            ax.plot(t_axis, data[:, j], label=names[j], linewidth=1.2, color=colors[j], alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("时间 (s)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # 标注阶段分界线（pick-and-place 的 4 个阶段）
        phase_boundaries = [0.25 * n_steps * 0.05, 0.35 * n_steps * 0.05,
                            0.75 * n_steps * 0.05]
        phase_labels = ["接近", "抓取", "提升", "放置"]
        ax.axvline(x=phase_boundaries[0], color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=phase_boundaries[1], color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=phase_boundaries[2], color="gray", linestyle="--", alpha=0.5)

    # 添加阶段标注（在第一个子图的顶部）
    axes[0, 0].text(
        0.125, 1.05, "接近", transform=axes[0, 0].get_xaxis_transform(),
        ha="center", fontsize=9, color="gray"
    )
    axes[0, 0].text(
        0.3, 1.05, "抓取", transform=axes[0, 0].get_xaxis_transform(),
        ha="center", fontsize=9, color="gray"
    )
    axes[0, 0].text(
        0.55, 1.05, "提升", transform=axes[0, 0].get_xaxis_transform(),
        ha="center", fontsize=9, color="gray"
    )
    axes[0, 0].text(
        0.875, 1.05, "放置", transform=axes[0, 0].get_xaxis_transform(),
        ha="center", fontsize=9, color="gray"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        print(f"\n[可视化] 保存对比图到 {save_path}")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ==============================================================================
#  适用场景讨论
# ==============================================================================

def print_usage_discussion():
    """
    打印每种动作表示方式的适用场景讨论。

    对应 README.md 中 2.3 节的表格。
    """
    print("\n" + "=" * 60)
    print("  动作表示方式适用场景讨论")
    print("=" * 60)
    print("""
  1. 关节角度 (Joint Angles)
     维度: N (等于机器人自由度，通常 7)
     优点: 直接可执行，无信息损失
     缺点: 与构型强耦合，无法跨平台迁移
     适用: 单一机器人平台，固定场景

  2. 末端位姿 (End-Effector Pose)
     维度: 7 (x, y, z, roll, pitch, yaw, gripper)
     优点: 与构型无关，迁移性好，符合人类直觉
     缺点: 需要 IK 解算，可能存在奇异点，IK 有多解
     适用: 跨平台通用策略（如 Octo, OpenVLA），目标-reaching

  3. 增量动作 (Delta Actions)
     维度: 7
     优点: 对坐标系不敏感，平移不变性，更鲁棒
     缺点: 误差会随时间累积，需要累积得绝对位置
     适用: 高频实时控制 (>20Hz)，需要快速响应的场景

  4. 关节速度 (Joint Velocities)
     维度: N
     优点: 平滑连续，具有物理意义，可直接用于速度控制
     缺点: 需积分得位置，对频率敏感
     适用: 直接速度控制接口，需要运动平滑的场景

  选择建议:
     - 单机器人、固定场景     --> 关节角度
     - 跨平台策略 (Octo/VLA)  --> 末端位姿或增量
     - 高频控制 (>20Hz)      --> 增量 delta
     - 需要目标-reaching      --> 末端位姿
     - 需要运动平滑          --> 关节速度
""")


# ==============================================================================
#  主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="机器人动作表示方式对比演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_steps", type=int, default=200, help="轨迹步数，默认: 200")
    parser.add_argument("--dt", type=float, default=0.05, help="时间步长(秒)，默认: 0.05 (20Hz)")
    parser.add_argument("--save", type=str, default=None, help="保存对比图为 PNG 文件路径")

    args = parser.parse_args()

    print("=" * 60)
    print("  机器人动作表示方式对比")
    print("  7-DOF Pick-and-Place 轨迹")
    print("=" * 60)
    print(f"  轨迹步数: {args.n_steps}")
    print(f"  时间步长: {args.dt}s ({1.0/args.dt:.0f}Hz)")
    print(f"  总时间: {args.n_steps * args.dt:.1f}s")
    print(f"  自由度: 7")

    # ── Step 1: 生成模拟轨迹 ──────────────────────────────────────────────
    print("\n--- Step 1: 生成模拟 Pick-and-Place 轨迹 ---")
    trajectory = generate_pick_and_place_trajectory(n_steps=args.n_steps)
    print(f"  轨迹形状: {trajectory.shape}")
    print(f"  首步: {trajectory[0]}")
    print(f"  末步: {trajectory[-1]}")

    # ── Step 2: 转换为 4 种表示 ───────────────────────────────────────────
    print("\n--- Step 2: 计算 4 种动作表示 ---")

    print("  [1/4] 关节角度...")
    joint_angles = joint_angles_representation(trajectory)

    print("  [2/4] 末端位姿...")
    ee_pose = end_effector_pose_representation(trajectory)

    print("  [3/4] 增量动作...")
    deltas = delta_representation(trajectory)

    print("  [4/4] 关节速度...")
    velocities = joint_velocity_representation(trajectory, dt=args.dt)

    # ── Step 3: 打印统计特征 ─────────────────────────────────────────────
    print("\n--- Step 3: 统计特征分析 ---")
    compute_statistics(joint_angles, "关节角度 (Joint Angles)", JOINT_NAMES)
    compute_statistics(ee_pose, "末端位姿 (End-Effector Pose)", POSE_NAMES)
    compute_statistics(deltas, "增量动作 (Delta Actions)", JOINT_NAMES)
    compute_statistics(velocities, "关节速度 (Joint Velocities)", JOINT_NAMES)

    # ── Step 4: 适用场景讨论 ─────────────────────────────────────────────
    print_usage_discussion()

    # ── Step 5: 可视化 ────────────────────────────────────────────────────
    print("\n--- Step 5: 可视化 ---")
    plot_comparison(trajectory, ee_pose, deltas, velocities, save_path=args.save)

    print("\n完成!")


if __name__ == "__main__":
    main()
