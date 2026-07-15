#!/usr/bin/env python3
"""
fk_ik_demo.py -- 正向/逆向运动学演示

================================================================================
  VLA-Zero-to-Hero 教学项目 · Stage 2: 动作表示
================================================================================

本脚本实现 2-DOF 平面机械臂的正向运动学 (FK) 和逆向运动学 (IK)，
并通过 matplotlib 动画可视化 IK 的迭代求解过程。

功能：
  - FK 解析解：给定关节角度，计算末端执行器位置 (x, y)
  - IK 数值解：Jacobian 伪逆迭代法，给定目标位置求解关节角度
  - 动画可视化：实时展示连杆随 IK 迭代逼近目标点的过程
  - 命令行接口：通过 --target 指定目标坐标

依赖：
  pip install numpy matplotlib

用法示例：
  # 默认目标点 (1.2, 0.8)
  python fk_ik_demo.py

  # 指定自定义目标坐标
  python fk_ik_demo.py --target 1.5,0.3

  # 指定目标并调整参数
  python fk_ik_demo.py --target 0.5,1.0 --max_iter 200 --step_size 0.3

扩展思路（注释中说明）：
  - 3-DOF：增加第三段连杆 l3 和关节角 q3，Jacobian 变为 2x3 矩阵
  - 6-DOF：使用 DH 参数法和 4x4 齐次变换矩阵，Jacobian 为 6x6
  - 可参考 README.md 中的 fk_6dof() 实现
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ── 中文字体设置 ──────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
#  正向运动学 (FK)
# ==============================================================================

def fk_2dof(q1: float, q2: float, l1: float = 1.0, l2: float = 1.0) -> np.ndarray:
    """
    2-DOF 平面机械臂的正向运动学（解析解）。

    原理：
      第一个关节在原点，第二个关节在第一段连杆末端。
      末端执行器位置 = 关节1位置 + 连杆1贡献 + 连杆2贡献。

    Args:
        q1: 第一关节角度（弧度）
        q2: 第二关节角度（弧度，相对于连杆1）
        l1: 第一段连杆长度
        l2: 第二段连杆长度

    Returns:
        np.ndarray: 末端执行器坐标 [x, y]
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, y])


def get_joint_positions(q1: float, q2: float, l1: float = 1.0, l2: float = 1.0):
    """
    获取所有关节和末端的坐标，用于绘图。

    Returns:
        tuple: (joints, end_effector)
            joints: np.ndarray, shape (3, 2) — [原点, 关节2, 末端]
    """
    j0 = np.array([0.0, 0.0])  # 基座
    j1 = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    end = fk_2dof(q1, q2, l1, l2)
    return np.array([j0, j1, end])


# ==============================================================================
#  逆向运动学 (IK) — Jacobian 伪逆迭代法
# ==============================================================================

def compute_jacobian_2dof(q1: float, q2: float, l1: float = 1.0, l2: float = 1.0) -> np.ndarray:
    """
    计算 2-DOF 机械臂的 Jacobian 矩阵。

    Jacobian 描述了关节速度与末端速度之间的关系：
        [dx/dt]     [dx/dq1  dx/dq2] [dq1/dt]
        [dy/dt]  =  [dy/dq1  dy/dq2] [dq2/dt]

    对于 2-DOF 平面臂，J 是 2x2 矩阵。

    Args:
        q1, q2: 当前关节角度（弧度）
        l1, l2: 连杆长度

    Returns:
        np.ndarray: 2x2 Jacobian 矩阵
    """
    J = np.array([
        [
            -l1 * np.sin(q1) - l2 * np.sin(q1 + q2),  # dx/dq1
            -l2 * np.sin(q1 + q2),                       # dx/dq2
        ],
        [
            l1 * np.cos(q1) + l2 * np.cos(q1 + q2),  # dy/dq1
            l2 * np.cos(q1 + q2),                       # dy/dq2
        ],
    ])
    return J


def ik_jacobian(
    target_xy: np.ndarray,
    initial_guess: np.ndarray = None,
    l1: float = 1.0,
    l2: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    step_size: float = 0.5,
) -> tuple:
    """
    使用 Jacobian 伪逆迭代法求解 2-DOF IK。

    算法步骤：
        1. 计算当前末端位置与目标的误差 e = target - current
        2. 如果误差小于阈值 tol，停止迭代
        3. 计算当前关节配置的 Jacobian 矩阵 J
        4. 通过伪逆计算关节增量：dq = J^+ @ e
        5. 更新关节角度：q += step_size * dq
        6. 重复 1-5

    Jacobian 伪逆法（也叫 Newton-Raphson 法在运动学中的应用）：
      - 优点：收敛快，实现简单
      - 缺点：可能陷入局部极小值，需要好的初始猜测
      - 注意：对于冗余自由度（N > 任务空间维度），伪逆自动处理多解

    Args:
        target_xy: 目标位置 [x, y]
        initial_guess: 初始关节角度 [q1, q2]，默认 [0.5, 0.5]
        l1, l2: 连杆长度
        max_iter: 最大迭代次数
        tol: 收敛阈值（末端位置误差）
        step_size: 迭代步长（阻尼因子），防止步子太大导致发散

    Returns:
        tuple: (q_final, history)
            q_final: np.ndarray, 最终关节角度 [q1, q2]
            history: list of dict, 每一步的记录（用于动画回放）
    """
    if initial_guess is None:
        initial_guess = np.array([0.5, 0.5])

    q = np.array(initial_guess, dtype=float)
    target = np.array(target_xy, dtype=float)

    history = []

    for i in range(max_iter):
        # 1. 当前末端位置
        current = fk_2dof(q[0], q[1], l1, l2)
        error = target - current
        error_norm = np.linalg.norm(error)

        # 2. 记录当前状态
        joints = get_joint_positions(q[0], q[1], l1, l2)
        history.append({
            "step": i,
            "q": q.copy(),
            "end_pos": current.copy(),
            "error_norm": error_norm,
            "joints": joints,
        })

        # 3. 检查收敛
        if error_norm < tol:
            print(f"[IK] 第 {i} 步收敛，误差 = {error_norm:.6f}")
            break

        # 4. 计算 Jacobian
        J = compute_jacobian_2dof(q[0], q[1], l1, l2)

        # 5. 伪逆更新
        #    伪逆 = J^T @ (J @ J^T)^{-1}  （当 J 非方阵时用此公式）
        #    对于 2x2 方阵，等价于矩阵求逆，但伪逆更通用
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ error

        # 6. 更新关节角度
        q += step_size * dq
    else:
        print(f"[IK] 达到最大迭代次数 {max_iter}，残余误差 = {error_norm:.6f}")

    return q, history


# ==============================================================================
#  可达性检查
# ==============================================================================

def check_reachability(target_xy: np.ndarray, l1: float, l2: float) -> bool:
    """
    检查目标点是否在机械臂的工作空间内。

    2-DOF 平面臂的工作空间是一个环形区域：
      - 内径 = |l1 - l2|（两连杆完全折叠）
      - 外径 = l1 + l2（两连杆完全伸展）
    """
    dist = np.linalg.norm(target_xy)
    r_min = abs(l1 - l2)
    r_max = l1 + l2
    return r_min - 1e-6 <= dist <= r_max + 1e-6


# ==============================================================================
#  动画可视化
# ==============================================================================

def animate_ik_solution(
    history: list,
    target: np.ndarray,
    l1: float = 1.0,
    l2: float = 1.0,
    save_path: str = None,
):
    """
    用 matplotlib 动画展示 IK 迭代过程中机械臂的运动。

    Args:
        history: IK 求解的历史记录
        target: 目标位置
        l1, l2: 连杆长度
        save_path: 若指定，保存动画为 GIF
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 绘制工作空间范围（内外圆）
    theta = np.linspace(0, 2 * np.pi, 200)
    r_max = l1 + l2
    r_min = abs(l1 - l2)
    ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), "g--", alpha=0.2, linewidth=1, label="工作空间外边界")
    ax.plot(r_min * np.cos(theta), r_min * np.sin(theta), "r--", alpha=0.2, linewidth=1, label="工作空间内边界")

    # 目标点（固定）
    target_circle = Circle(target, 0.04, color="red", zorder=5, label="目标位置")
    ax.add_patch(target_circle)
    ax.plot(target[0], target[1], "rx", markersize=12, markeredgewidth=2, zorder=6)

    # 初始化绘图元素
    (link_line,) = ax.plot([], [], "o-", color="#2196F3", linewidth=6, markersize=10,
                          markerfacecolor="#FFC107", markeredgecolor="black", markeredgewidth=1.5,
                          solid_capstyle="round", label="机械臂连杆")
    (trail_line,) = ax.plot([], [], "-", color="#E91E63", linewidth=1, alpha=0.4, label="末端轨迹")
    error_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11,
                         verticalalignment="top", fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # 收集轨迹点
    trail_x = [h["end_pos"][0] for h in history]
    trail_y = [h["end_pos"][1] for h in history]

    # 坐标轴设置
    margin = 0.5
    ax.set_xlim(-(r_max + margin), r_max + margin)
    ax.set_ylim(-(r_max + margin), r_max + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("2-DOF 平面机械臂 — IK 求解动画", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    def init():
        link_line.set_data([], [])
        trail_line.set_data([], [])
        error_text.set_text("")
        return link_line, trail_line, error_text

    def update(frame):
        h = history[frame]
        joints = h["joints"]

        # 更新连杆位置
        link_line.set_data(joints[:, 0], joints[:, 1])

        # 更新末端轨迹
        trail_line.set_data(trail_x[: frame + 1], trail_y[: frame + 1])

        # 更新文本
        converged = h["error_norm"] < 1e-4
        status = "已收敛" if converged else "迭代中..."
        error_text.set_text(
            f"步骤: {h['step']:>4d}/{len(history)-1}\n"
            f"q1 = {h['q'][0]:+.4f} rad ({np.degrees(h['q'][0]):+.1f}°)\n"
            f"q2 = {h['q'][1]:+.4f} rad ({np.degrees(h['q'][1]):+.1f}°)\n"
            f"末端: ({h['end_pos'][0]:.4f}, {h['end_pos'][1]:.4f})\n"
            f"误差: {h['error_norm']:.6f}\n"
            f"状态: {status}"
        )

        return link_line, trail_line, error_text

    n_frames = len(history)
    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=80, blit=True, repeat=True,
    )

    plt.tight_layout()

    if save_path:
        print(f"[可视化] 保存动画到 {save_path}")
        ani.save(save_path, writer="pillow", fps=15)
    else:
        plt.show()


# ==============================================================================
#  静态可视化（不使用动画，适合快速查看）
# ==============================================================================

def plot_static_comparison(
    initial_q: np.ndarray,
    solved_q: np.ndarray,
    target: np.ndarray,
    l1: float = 1.0,
    l2: float = 1.0,
):
    """
    在一张图上对比 IK 求解前后的机械臂配置。

    Args:
        initial_q: 初始关节角度
        solved_q: 求解后的关节角度
        target: 目标位置
        l1, l2: 连杆长度
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 工作空间
    theta = np.linspace(0, 2 * np.pi, 200)
    r_max = l1 + l2
    ax.fill(
        r_max * np.cos(theta), r_max * np.sin(theta),
        alpha=0.05, color="green", label="工作空间",
    )

    # 初始配置
    joints_init = get_joint_positions(initial_q[0], initial_q[1], l1, l2)
    ax.plot(joints_init[:, 0], joints_init[:, 1], "o--", color="gray", linewidth=3,
            markersize=8, alpha=0.5, label="初始配置")

    # 求解后配置
    joints_solved = get_joint_positions(solved_q[0], solved_q[1], l1, l2)
    ax.plot(joints_solved[:, 0], joints_solved[:, 1], "o-", color="#2196F3", linewidth=5,
            markersize=10, markerfacecolor="#FFC107", markeredgecolor="black",
            markeredgewidth=1.5, label="求解后配置")

    # 目标
    ax.plot(target[0], target[1], "rx", markersize=15, markeredgewidth=3, label="目标位置")
    ax.add_patch(Circle(target, 0.05, color="red", alpha=0.2, zorder=3))

    # 标注
    end_solved = fk_2dof(solved_q[0], solved_q[1], l1, l2)
    error = np.linalg.norm(end_solved - target)
    ax.annotate(
        f"误差 = {error:.6f}",
        xy=end_solved, xytext=(end_solved[0] + 0.2, end_solved[1] + 0.2),
        fontsize=10, arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_xlim(-(r_max + 0.5), r_max + 0.5)
    ax.set_ylim(-(r_max + 0.5), r_max + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("IK 求解结果对比", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


# ==============================================================================
#  主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2-DOF 平面机械臂 — 正向/逆向运动学演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python fk_ik_demo.py                          # 默认目标 (1.2, 0.8)
  python fk_ik_demo.py --target 1.5,0.3         # 自定义目标
  python fk_ik_demo.py --target 0.5,1.0 --max_iter 200 --step_size 0.3
  python fk_ik_demo.py --no_animation           # 仅显示静态对比图
  python fk_ik_demo.py --save_gif ik_demo.gif   # 保存动画为 GIF
        """,
    )
    parser.add_argument("--target", type=str, default="1.2,0.8",
                        help="目标坐标 x,y（逗号分隔），默认: 1.2,0.8")
    parser.add_argument("--l1", type=float, default=1.0, help="第一段连杆长度，默认: 1.0")
    parser.add_argument("--l2", type=float, default=1.0, help="第二段连杆长度，默认: 1.0")
    parser.add_argument("--max_iter", type=int, default=100, help="IK 最大迭代次数，默认: 100")
    parser.add_argument("--tol", type=float, default=1e-4, help="收敛阈值，默认: 1e-4")
    parser.add_argument("--step_size", type=float, default=0.5, help="IK 迭代步长，默认: 0.5")
    parser.add_argument("--initial_guess", type=str, default="0.5,0.5",
                        help="初始关节角度 q1,q2（弧度），默认: 0.5,0.5")
    parser.add_argument("--no_animation", action="store_true", help="不播放动画，仅显示静态图")
    parser.add_argument("--save_gif", type=str, default=None, help="保存动画为 GIF 文件路径")

    args = parser.parse_args()

    # 解析目标坐标
    try:
        target = np.array([float(v) for v in args.target.split(",")])
        assert len(target) == 2
    except (ValueError, AssertionError):
        print(f"[错误] 无效的目标坐标: {args.target}，格式应为 x,y")
        return

    # 解析初始猜测
    try:
        initial_guess = np.array([float(v) for v in args.initial_guess.split(",")])
        assert len(initial_guess) == 2
    except (ValueError, AssertionError):
        print(f"[错误] 无效的初始关节角度: {args.initial_guess}，格式应为 q1,q2")
        return

    l1, l2 = args.l1, args.l2

    print("=" * 60)
    print("  2-DOF 平面机械臂 — FK / IK 演示")
    print("=" * 60)
    print(f"  连杆长度: l1={l1}, l2={l2}")
    print(f"  工作空间: [{abs(l1-l2):.2f}, {l1+l2:.2f}]")
    print(f"  目标位置: ({target[0]}, {target[1]})")
    print(f"  初始关节: ({initial_guess[0]}, {initial_guess[1]})")
    print(f"  最大迭代: {args.max_iter}")
    print(f"  收敛阈值: {args.tol}")
    print(f"  步长:     {args.step_size}")
    print("=" * 60)

    # ── 可达性检查 ────────────────────────────────────────────────────────
    dist = np.linalg.norm(target)
    if not check_reachability(target, l1, l2):
        print(f"\n[警告] 目标 ({target[0]}, {target[1]}) 距离原点 {dist:.3f}")
        print(f"        超出工作空间 [{abs(l1-l2):.2f}, {l1+l2:.2f}]")
        print("        IK 可能无法收敛，但仍会尝试求解...\n")

    # ── 正向运动学验证 ────────────────────────────────────────────────────
    print("\n--- 正向运动学 (FK) ---")
    q_demo = initial_guess
    end_fk = fk_2dof(q_demo[0], q_demo[1], l1, l2)
    print(f"  输入关节角: q1={q_demo[0]:.4f} rad ({np.degrees(q_demo[0]):.1f}°), "
          f"q2={q_demo[1]:.4f} rad ({np.degrees(q_demo[1]):.1f}°)")
    print(f"  FK 输出位置: ({end_fk[0]:.6f}, {end_fk[1]:.6f})")

    # Jacobian 矩阵展示
    J = compute_jacobian_2dof(q_demo[0], q_demo[1], l1, l2)
    print(f"\n  Jacobian 矩阵:")
    print(f"    J = [[{J[0,0]:+.4f}, {J[0,1]:+.4f}],")
    print(f"         [{J[1,0]:+.4f}, {J[1,1]:+.4f}]]")

    # ── 逆向运动学求解 ────────────────────────────────────────────────────
    print("\n--- 逆向运动学 (IK) ---")
    print(f"  目标: ({target[0]}, {target[1]})")

    q_solved, history = ik_jacobian(
        target_xy=target,
        initial_guess=initial_guess,
        l1=l1, l2=l2,
        max_iter=args.max_iter,
        tol=args.tol,
        step_size=args.step_size,
    )

    end_solved = fk_2dof(q_solved[0], q_solved[1], l1, l2)
    final_error = np.linalg.norm(end_solved - target)
    print(f"\n  最终关节角: q1={q_solved[0]:.6f} rad ({np.degrees(q_solved[0]):.2f}°), "
          f"q2={q_solved[1]:.6f} rad ({np.degrees(q_solved[1]):.2f}°)")
    print(f"  验证 FK:   ({end_solved[0]:.6f}, {end_solved[1]:.6f})")
    print(f"  最终误差:  {final_error:.8f}")

    # ── 扩展思路提示 ──────────────────────────────────────────────────────
    print("\n--- 扩展思路 ---")
    print("  3-DOF: 增加第三段连杆 l3 和关节角 q3，Jacobian 变为 2x3（冗余自由度）")
    print("         伪逆法自动处理冗余，会找到最小范数解")
    print("  6-DOF: 使用 DH 参数法计算 4x4 齐次变换矩阵链")
    print("         Jacobian 为 6x6，任务空间包含位置(3)和姿态(3)")
    print("         逆运动学有多解（如 elbow-up / elbow-down），需选择策略")
    print("  参考 README.md 中的 fk_6dof() 实现了解 DH 参数法")

    # ── 可视化 ────────────────────────────────────────────────────────────
    print("\n--- 可视化 ---")
    if args.no_animation:
        plot_static_comparison(initial_guess, q_solved, target, l1, l2)
    else:
        if args.save_gif:
            animate_ik_solution(history, target, l1, l2, save_path=args.save_gif)
        else:
            animate_ik_solution(history, target, l1, l2)


if __name__ == "__main__":
    main()
