"""
RL Zero-to-One: 强化学习训练灵巧手策略
========================================
使用 Stable-Baselines3 (SAC + HER) + Gymnasium-Robotics (Shadow Hand)
训练灵巧手操作策略。

Modes:
  train   — 训练 SAC+HER 策略（CPU 可运行）
  enjoy   — 加载并渲染训练好的策略
  eval    — 评估成功率
  demo    — 演示 RL 概念（numpy 模拟，无需安装依赖）

Usage:
    # 训练（CPU 可运行，约 30 分钟）
    python rl_demo.py --mode train --timesteps 50000

    # 快速演示 RL 概念（无需额外安装）
    python rl_demo.py --mode demo --task reach

    # 测试训练好的策略
    python rl_demo.py --mode enjoy --model shadow_hand_block

    # 评估成功率
    python rl_demo.py --mode eval --model shadow_hand_block --episodes 100
"""

import argparse
import sys
import os
import time
import numpy as np

DEFAULT_MODEL_NAME = "shadow_hand_block"


def check_dependencies():
    """检查依赖。"""
    missing = []
    try:
        import gymnasium
    except ImportError:
        missing.append("gymnasium")
    try:
        import stable_baselines3
    except ImportError:
        missing.append("stable-baselines3")
    return missing


def run_demo(args):
    """
    RL 概念演示：用 numpy 模拟 RL 训练循环。
    展示 Q-learning 的核心思想，无需安装任何 RL 库。
    """
    print("=" * 70)
    print(" RL Concept Demo: 用 numpy 理解强化学习")
    print("=" * 70)

    np.random.seed(42)

    # --- Step 1: 定义环境 ---
    print("\n[Step 1/5] 定义环境: 简化的 2D 灵巧手抓取")
    print("  状态: [finger_x, finger_y, object_x, object_y] (4 维)")
    print("  动作: [dx, dy] (2 维，手指移动方向)")
    print("  奖励: -distance(finger, object)")

    # --- Step 2: 定义 Q-table ---
    print("\n[Step 2/5] 初始化 Q-table (Q-learning)")
    print("  Q-table 记录每个 (状态, 动作) 对的期望奖励")

    state_bins = 10
    n_actions = 4  # 上、下、左、右
    Q = np.zeros((state_bins, state_bins, state_bins, state_bins, n_actions))

    # --- Step 3: 训练循环 ---
    print("\n[Step 3/5] 开始训练 (Q-learning)")
    print("  算法: ϵ-greedy 探索 + Bellman 更新")

    alpha = 0.1      # 学习率
    gamma = 0.9      # 折扣因子
    epsilon = 0.3    # 探索率
    n_episodes = 500

    actions = np.array([[0, 0.05], [0, -0.05], [0.05, 0], [-0.05, 0]])

    def discretize(val, min_val=-1, max_val=1, bins=state_bins):
        return int(np.clip((val - min_val) / (max_val - min_val) * (bins - 1), 0, bins - 1))

    def get_state_idx(finger, obj):
        f = discretize(finger[0]), discretize(finger[1])
        o = discretize(obj[0]), discretize(obj[1])
        return f[0], f[1], o[0], o[1]

    reward_history = []
    for episode in range(n_episodes):
        obj = np.random.uniform(-0.5, 0.5, 2)  # 随机目标位置
        finger = np.array([0.0, 0.0])           # 手指初始位置

        total_reward = 0
        for step in range(50):
            state_idx = get_state_idx(finger, obj)

            # ϵ-greedy
            if np.random.random() < epsilon:
                action_idx = np.random.randint(n_actions)
            else:
                action_idx = np.argmax(Q[state_idx])

            # 执行动作
            finger = finger + actions[action_idx]
            finger = np.clip(finger, -1, 1)

            # 计算奖励
            distance = np.linalg.norm(finger - obj)
            reward = -distance

            # Bellman 更新
            next_state_idx = get_state_idx(finger, obj)
            Q[state_idx][action_idx] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx][action_idx])

            total_reward += reward

        reward_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"  Episode {episode+1}/{n_episodes}: avg_reward = {avg_reward:.3f}")

    # --- Step 4: 评估 ---
    print("\n[Step 4/5] 评估训练结果")
    test_rewards = []
    for _ in range(50):
        obj = np.random.uniform(-0.5, 0.5, 2)
        finger = np.array([0.0, 0.0])
        total_reward = 0
        for step in range(50):
            state_idx = get_state_idx(finger, obj)
            action_idx = np.argmax(Q[state_idx])  # 纯 exploitation
            finger = finger + actions[action_idx]
            finger = np.clip(finger, -1, 1)
            distance = np.linalg.norm(finger - obj)
            total_reward += -distance
        test_rewards.append(total_reward)

    avg_test_reward = np.mean(test_rewards)

    # --- Step 5: 可视化 ---
    print(f"\n[Step 5/5] 结果")
    print(f"  平均测试奖励: {avg_test_reward:.3f}")
    print(f"  训练前 vs 训练后:")
    print(f"    训练前: 随机动作，平均距离 ~0.5")
    print(f"    训练后: 学习策略，平均距离 ~{(-avg_test_reward/50):.3f}")

    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # 奖励曲线
            ax1.plot(reward_history, alpha=0.3, color='blue', label='Per episode')
            ax1.plot(np.convolve(reward_history, np.ones(50)/50, mode='valid'),
                     color='red', linewidth=2, label='Moving avg (50)')
            ax1.set_title('Q-Learning Training Curve')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 最终策略演示
            obj = np.array([0.3, 0.4])
            finger = np.array([0.0, 0.0])
            trajectory = [finger.copy()]
            for _ in range(50):
                state_idx = get_state_idx(finger, obj)
                action_idx = np.argmax(Q[state_idx])
                finger = finger + actions[action_idx]
                finger = np.clip(finger, -1, 1)
                trajectory.append(finger.copy())
                if np.linalg.norm(finger - obj) < 0.05:
                    break
            trajectory = np.array(trajectory)

            ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', markersize=3, linewidth=1, label='Trajectory')
            ax2.scatter(*obj, c='red', s=200, marker='*', label='Target', zorder=5)
            ax2.scatter(0, 0, c='green', s=100, marker='s', label='Start', zorder=5)
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(-1, 1)
            ax2.set_title('Learned Policy: Finger → Target')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  [Warning] matplotlib 未安装，跳过可视化")

    print(f"\n{'=' * 70}")
    print(f" RL Demo 完成！")
    print(f"{'=' * 70}")
    print(f" 算法:  Q-Learning (ϵ-greedy)")
    print(f" 状态:  4 维 (finger_xy + object_xy)")
    print(f" 动作:  4 种 (上下左右)")
    print(f" 奖励:  -distance")
    print(f" 结果:  学会了用手指接近目标")
    print(f"{'=' * 70}")
    print(f"\n 提示: pip install stable-baselines3 gymnasium-robotics 后可运行真实训练")
    print(f"   python rl_demo.py --mode train --timesteps 50000")


def run_train(args):
    """使用 SB3 + Gymnasium-Robotics 训练 Shadow Hand 策略。"""
    print("=" * 70)
    print(" RL Training: SAC + HER on Shadow Hand")
    print("=" * 70)

    missing = check_dependencies()
    if missing:
        print(f"\n[Error] 缺少依赖: {missing}")
        print(f"  pip install {' '.join(missing)}")
        print("  或先运行 --mode demo 体验 RL 概念")
        sys.exit(1)

    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    from stable_baselines3 import SAC, HerReplayBuffer

    env_id = args.env
    print(f"\n[Step 1/4] 创建环境: {env_id}")

    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except Exception as e:
        print(f"  [Error] 环境创建失败: {e}")
        print(f"  可用环境: HandReach-v1, HandManipulateBlock-v1, HandManipulateEgg-v1, HandManipulatePen-v1")
        sys.exit(1)

    print(f"  状态: {env.observation_space}")
    print(f"  动作: {env.action_space}")
    print(f"  最大步数: {env.spec.max_episode_steps if env.spec else 'N/A'}")

    print(f"\n[Step 2/4] 创建 SAC + HER 模型")
    print(f"  算法: SAC (Soft Actor-Critic)")
    print(f"  回放: HER (Hindsight Experience Replay)")
    print(f"  设备: {'cuda' if args.device == 'cuda' else 'cpu'}")

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
    )

    print(f"\n[Step 3/4] 开始训练 ({args.timesteps} steps)")

    start_time = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - start_time

    print(f"\n  训练耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  速度: {args.timesteps / elapsed:.0f} steps/s")

    print(f"\n[Step 4/4] 保存模型")

    model_name = args.model_name or DEFAULT_MODEL_NAME
    model.save(model_name)
    print(f"  模型已保存到: {model_name}.zip")

    print(f"\n{'=' * 70}")
    print(f" 训练完成！")
    print(f"{'=' * 70}")
    print(f" 环境:     {env_id}")
    print(f" 步数:     {args.timesteps}")
    print(f" 耗时:     {elapsed:.1f}s")
    print(f" 模型:     {model_name}.zip")
    print(f" 测试:     python rl_demo.py --mode enjoy --model {model_name} --env {env_id}")
    print(f"{'=' * 70}")


def run_enjoy(args):
    """加载并渲染训练好的策略。"""
    print("=" * 70)
    print(" RL Enjoy: 渲染训练好的策略")
    print("=" * 70)

    missing = check_dependencies()
    if missing:
        print(f"\n[Error] 缺少依赖: {missing}")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    from stable_baselines3 import SAC

    model_path = args.model or DEFAULT_MODEL_NAME
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"\n[Error] 找不到模型文件: {model_path}")
        print(f"  请先训练: python rl_demo.py --mode train --timesteps 50000")
        sys.exit(1)

    print(f"\n[Step 1/3] 加载模型: {model_path}")
    model = SAC.load(model_path)
    print(f"  [OK] 模型加载成功")

    env_id = args.env
    print(f"\n[Step 2/3] 创建环境: {env_id}")

    try:
        env = gym.make(env_id, render_mode="human")
    except Exception as e:
        print(f"  [Error] 环境创建失败: {e}")
        sys.exit(1)

    print(f"\n[Step 3/3] 运行策略 (50 episodes)")

    for episode in range(50):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(env.spec.max_episode_steps if env.spec else 100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        success = info.get("is_success", False)
        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"  Episode {episode+1}/50: reward={total_reward:.3f} {status}")

        if episode >= 5 and not args.render_all:
            break

    env.close()
    print(f"\n{'=' * 70}")
    print(f" 渲染完成！")
    print(f"{'=' * 70}")


def run_eval(args):
    """评估策略成功率。"""
    print("=" * 70)
    print(" RL Eval: 评估策略成功率")
    print("=" * 70)

    missing = check_dependencies()
    if missing:
        print(f"\n[Error] 缺少依赖: {missing}")
        sys.exit(1)

    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    from stable_baselines3 import SAC

    model = SAC.load(args.model or DEFAULT_MODEL_NAME)
    env = gym.make(args.env)

    success_count = 0
    total_reward = 0

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if info.get("is_success", False):
                success_count += 1
                break

            if terminated or truncated:
                break

        total_reward += episode_reward

        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1}/{args.episodes}: success_rate = {success_count/(episode+1)*100:.1f}%")

    env.close()

    success_rate = success_count / args.episodes * 100
    avg_reward = total_reward / args.episodes

    print(f"\n{'=' * 70}")
    print(f" 评估结果")
    print(f"{'=' * 70}")
    print(f" Episodes:     {args.episodes}")
    print(f" 成功率:       {success_rate:.1f}%")
    print(f" 平均奖励:     {avg_reward:.3f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="RL Zero-to-One: 强化学习训练灵巧手策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # RL 概念演示（无需额外安装）
  python rl_demo.py --mode demo

  # 训练 Shadow Hand 策略
  python rl_demo.py --mode train --timesteps 50000

  # 渲染训练好的策略
  python rl_demo.py --mode enjoy --model shadow_hand_block

  # 评估成功率
  python rl_demo.py --mode eval --model shadow_hand_block --episodes 100
        """
    )

    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "train", "enjoy", "eval"],
                        help="运行模式")

    # demo 模式
    parser.add_argument("--task", type=str, default="reach",
                        help="Demo 任务类型")

    # train 模式
    parser.add_argument("--env", type=str, default="HandManipulateBlock-v1",
                        help="Gymnasium 环境 ID")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="训练步数")
    parser.add_argument("--model-name", type=str, default=None,
                        help="保存模型的文件名")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="训练设备")
    parser.add_argument("--tensorboard-log", type=str, default="./rl_tensorboard/",
                        help="TensorBoard 日志目录")

    # enjoy / eval 模式
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径（默认: shadow_hand_block）")
    parser.add_argument("--episodes", type=int, default=100,
                        help="评估 episode 数")
    parser.add_argument("--render-all", action="store_true",
                        help="渲染所有 episode（enjoy 模式）")

    # 通用
    parser.add_argument("--visualize", action="store_true",
                        help="可视化（demo 模式）")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "enjoy":
        run_enjoy(args)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()