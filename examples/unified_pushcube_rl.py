"""
Unified PushCube — RL Track
============================
Train an RL policy on PushCube using pure NumPy:
  Input:  state (13-D) — [arm_x, arm_y, cube1_x, cube1_y, cube2_x, cube2_y,
                          target_x, target_y, cube1_r, cube1_g,
                          cube2_r, cube2_g, active_idx]
  Output: action (2-D)  — [dx, dy]

Uses REINFORCE (policy gradient) with a 2-layer MLP policy.
This is a lightweight teaching implementation (no SB3 dependency).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv, expert_action


def train_rl(args):
    """Train a simple policy network with REINFORCE (policy gradient)."""
    print("=" * 70)
    print(" Unified PushCube — RL Training (REINFORCE)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Get dimensions from environment (state_dim = 13)
    # ------------------------------------------------------------------
    _env_tmp = PushCubeEnv()
    state_dim = _env_tmp.state_dim       # 13
    action_dim = _env_tmp.action_dim     # 2
    hidden_dim = 32

    rng = np.random.RandomState(args.seed)

    # Initialize weights
    W1 = rng.randn(state_dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.randn(hidden_dim, action_dim).astype(np.float32) * 0.1
    b2 = np.zeros(action_dim, dtype=np.float32)
    W_logstd = np.zeros(action_dim, dtype=np.float32) - 1.0  # log(std)

    def relu(x):
        return np.maximum(0, x)

    def policy_forward(state):
        h = relu(state @ W1 + b1)
        mean = h @ W2 + b2
        _logstd = W_logstd
        std = np.exp(_logstd)
        return mean, std

    def sample_action(state):
        mean, std = policy_forward(state)
        action = mean + std * rng.randn(action_dim)
        # Fix: use _logstd alias to avoid closure ambiguity with W_logstd
        _logstd = W_logstd
        log_prob = -0.5 * np.sum(((action - mean) / std) ** 2) - np.sum(_logstd)
        return np.clip(action, -1.0, 1.0), log_prob

    # Training hyper-parameters
    gamma = 0.99
    lr = 1e-3

    print(f"\nState dim: {state_dim}, Action dim: {action_dim}")
    print(f"Training for {args.n_episodes} episodes (seed={args.seed})...")
    best_reward = -float("inf")
    reward_history = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in range(args.n_episodes):
        env = PushCubeEnv()
        obs = env.reset(seed=args.seed + ep)
        state = env.get_state_vector()

        states = []
        actions = []
        log_probs = []
        rewards = []

        for step in range(env.max_steps):
            action, log_prob = sample_action(state)
            next_obs, reward, done, truncated, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = env.get_state_vector()
            if done or truncated:
                break

        if len(states) == 0:
            continue

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        dW2 = np.zeros_like(W2)
        db2 = np.zeros_like(b2)
        dW1 = np.zeros_like(W1)
        db1 = np.zeros_like(b1)
        dW_logstd = np.zeros_like(W_logstd)

        # Fix: use enumerate instead of states.index(s) to correctly
        # handle duplicate states within an episode
        for t, (s, lp, ret) in enumerate(zip(states, log_probs, returns)):
            h = relu(s @ W1 + b1)
            mean, std = policy_forward(s)
            a = actions[t]

            # Gradient of log_prob w.r.t. mean and logstd
            dmean = (a - mean) / (std ** 2)
            dlogstd = ((a - mean) ** 2) / (std ** 2) - 1.0

            # Backprop
            dW2 += np.outer(h, dmean) * ret
            db2 += dmean * ret
            dh = dmean @ W2.T
            dh[h <= 0] = 0
            dW1 += np.outer(s, dh) * ret
            db1 += dh * ret
            dW_logstd += dlogstd * ret

        W2 += lr * dW2 / len(states)
        b2 += lr * db2 / len(states)
        W1 += lr * dW1 / len(states)
        b1 += lr * db1 / len(states)
        W_logstd += lr * dW_logstd / len(states)

        total_reward = sum(rewards)
        reward_history.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        # Print roughly 10 progress lines
        print_interval = max(1, args.n_episodes // 10)
        if (ep + 1) % print_interval == 0:
            window = min(100, len(reward_history))
            avg = np.mean(reward_history[-window:])
            print(f"  Episode {ep+1}/{args.n_episodes}: "
                  f"avg_reward={avg:.3f}, best={best_reward:.3f}")

    # ------------------------------------------------------------------
    # Evaluate trained policy (deterministic)
    # ------------------------------------------------------------------
    print(f"\nEvaluating trained policy (deterministic, {args.n_eval} episodes)...")
    success_count = 0
    eval_rewards = []

    for ep in range(args.n_eval):
        env = PushCubeEnv()
        obs = env.reset(seed=10000 + ep)
        state = env.get_state_vector()
        total_reward = 0

        for step in range(env.max_steps):
            mean, _ = policy_forward(state)
            action = np.clip(mean, -1.0, 1.0)
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = env.get_state_vector()
            if done:
                success_count += 1
                break
            if truncated:
                break

        eval_rewards.append(total_reward)

    success_rate = success_count / args.n_eval * 100
    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))

    print(f"Success rate: {success_count}/{args.n_eval} = {success_rate:.1f}%")
    print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")

    # ------------------------------------------------------------------
    # Expert baseline (for comparison)
    # ------------------------------------------------------------------
    print(f"\nExpert baseline ({args.n_eval} episodes)...")
    expert_success = 0
    for ep in range(args.n_eval):
        env = PushCubeEnv()
        obs = env.reset(seed=10000 + ep)
        for step in range(env.max_steps):
            action = expert_action(env)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                expert_success += 1
                break
            if truncated:
                break
    expert_success_rate = expert_success / args.n_eval * 100
    print(f"Expert success rate: {expert_success}/{args.n_eval} "
          f"= {expert_success_rate:.1f}%")

    # ------------------------------------------------------------------
    # Save policy and results
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / "pushcube_rl_policy.npz",
        W1=W1, b1=b1, W2=W2, b2=b2, W_logstd=W_logstd,
    )
    print(f"\nPolicy saved to {save_dir / 'pushcube_rl_policy.npz'}")

    results = {
        "task": "PushCube RL (REINFORCE)",
        "state_dim": state_dim,
        "n_episodes": args.n_episodes,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "best_reward": round(float(best_reward), 3),
        "success_rate": round(success_rate, 1),
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "expert_success_rate": round(expert_success_rate, 1),
    }
    results_path = save_dir / "rl_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified PushCube — RL Track")
    parser.add_argument("--n-episodes", type=int, default=1000,
                        help="Training episodes (default: 1000)")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Evaluation episodes (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str,
                        default="../results/unified_pushcube/rl",
                        help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test (10 train, 5 eval) for CI")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_episodes = 10
        args.n_eval = 5
        print("[SMOKE TEST MODE] 10 train episodes, 5 eval episodes")

    train_rl(args)


if __name__ == "__main__":
    main()
