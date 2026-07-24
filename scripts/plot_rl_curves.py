"""Plot RL training curves from SB3 Monitor CSV logs.

Reads monitor.csv and eval_results.npz from the training log directory,
then generates reward curve and success rate plots.

Usage:
    python scripts/plot_rl_curves.py --log-dir ./rl_logs/
    python scripts/plot_rl_curves.py --log-dir ./rl_logs/ --output assets/demos/rl_training_curves.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_monitor_csv(csv_path):
    """Read SB3 Monitor CSV (skips the 2-line header)."""
    rewards = []
    lengths = []
    with open(csv_path, "r") as f:
        # Skip first two lines (metadata)
        next(f)
        next(f)
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["r"]))
            lengths.append(int(row["l"]))
    return np.array(rewards), np.array(lengths)


def read_eval_npz(npz_path):
    """Read SB3 EvalCallback results."""
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]
    # Mean over eval episodes
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    return timesteps, mean_rewards, std_rewards


def main():
    parser = argparse.ArgumentParser(description="Plot RL training curves")
    parser.add_argument("--log-dir", type=str, default="./rl_logs/",
                        help="Training log directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: log_dir/training_curves.png)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_path = Path(args.output) if args.output else log_dir / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Episode Reward (from monitor.csv) ---
    monitor_path = log_dir / "monitor.csv"
    if monitor_path.exists():
        rewards, lengths = read_monitor_csv(monitor_path)
        episodes = np.arange(1, len(rewards) + 1)

        # Raw reward
        axes[0].plot(episodes, rewards, alpha=0.2, color="#4C78A8", label="Per episode")

        # Moving average
        window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[0].plot(
                episodes[window - 1:], moving_avg,
                color="#1f3a5f", linewidth=2,
                label=f"Moving avg ({window})"
            )

        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Episode Reward")
        axes[0].set_title("Training Reward Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, f"monitor.csv not found\nin {log_dir}",
                     ha="center", va="center", fontsize=12, color="gray")
        axes[0].set_title("Training Reward Curve (N/A)")

    # --- Plot 2: Evaluation Reward (from eval_results) ---
    eval_npz = log_dir / "eval_results" / "evaluations.npz"
    if eval_npz.exists():
        timesteps, mean_rewards, std_rewards = read_eval_npz(eval_npz)

        axes[1].plot(timesteps, mean_rewards, color="#F58518", linewidth=2, label="Mean eval reward")
        axes[1].fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2, color="#F58518", label="± 1 std"
        )
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Evaluation Reward")
        axes[1].set_title("Periodic Evaluation Reward")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, f"evaluations.npz not found\nin {log_dir}/eval_results/",
                     ha="center", va="center", fontsize=12, color="gray")
        axes[1].set_title("Evaluation Reward (N/A)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {output_path}")


if __name__ == "__main__":
    main()
