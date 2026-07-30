"""
Unified PushCube — RL Track
============================
Train an RL policy on PushCube using pure NumPy:
  Input: state (8-D)
  Output: action (2-D)

This is a lightweight teaching implementation (no SB3 dependency).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv


def train_rl(args):
    """Train a simple policy network with REINFORCE (policy gradient)."""
    print("=" * 70)
    print(" Unified PushCube — RL Training (REINFORCE)")
    print("=" * 70)

    # Simple 2-layer MLP policy
    state_dim = 8
    action_dim = 2
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
        logstd = W_logstd
        std = np.exp(logstd)
        return mean, std

    def sample_action(state):
        mean, std = policy_forward(state)
        action = mean + std * rng.randn(action_dim)
        _logstd = W_logstd  # alias for closure
        log_prob = -0.5 * np.sum(((action - mean) / std) ** 2) - np.sum(_logstd)
        return np.clip(action, -1.0, 1.0), log_prob

    # Training
    gamma = 0.99
    lr = 1e-3

    print(f"\nTraining for {args.n_episodes} episodes (seed={args.seed})...")
    best_reward = -float("inf")
    reward_history = []

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

        # Compute returns
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

        if (ep + 1) % 100 == 0:
            avg = np.mean(reward_history[-100:])
            print(f"  Episode {ep+1}/{args.n_episodes}: avg_reward={avg:.3f}, best={best_reward:.3f}")

    # ------------------------------------------------------------------
    # Evaluate
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

    print(f"Success rate: {success_count}/{args.n_eval} = {success_count/args.n_eval*100:.1f}%")
    print(f"Mean reward: {np.mean(eval_rewards):.3f} ± {np.std(eval_rewards):.3f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / "pushcube_rl_policy.npz",
        W1=W1, b1=b1, W2=W2, b2=b2, W_logstd=W_logstd,
    )
    print(f"\nPolicy saved to {save_dir / 'pushcube_rl_policy.npz'}")

    import json
    with open(save_dir / "rl_config.json", "w") as f:
        json.dump({
            "task": "PushCube RL (REINFORCE)",
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "best_reward": round(float(best_reward), 3),
            "success_rate": round(success_count / args.n_eval * 100, 1),
            "mean_reward": round(float(np.mean(eval_rewards)), 3),
            "std_reward": round(float(np.std(eval_rewards)), 3),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unified PushCube — RL Track")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--n-eval", type=int, default=50, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="../results/unified_pushcube/rl", help="Output directory")
    args = parser.parse_args()
    train_rl(args)


if __name__ == "__main__":
    main()
