"""Generate illustrative RL learning curves with honest label."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    rng = np.random.default_rng(seed=42)
    episodes = np.arange(1, 501)

    # Synthetic noisy exponential rise for episode reward
    reward_mean = -50 + 55 * (1 - np.exp(-episodes / 120))
    reward_noise = rng.normal(0, 3, size=len(episodes))
    rewards = reward_mean + reward_noise

    # Synthetic noisy exponential rise for success rate (0-1)
    success_mean = 0.85 * (1 - np.exp(-episodes / 100))
    success_noise = rng.normal(0, 0.03, size=len(episodes))
    success_rates = np.clip(success_mean + success_noise, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(episodes, rewards, color="#4C78A8", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Episode Reward over Training", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(episodes, success_rates, color="#F58518", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Success Rate over Training", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.02,
        "Illustrative — not from completed SAC+HER training",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "assets" / "demos" / "learning_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved RL curves to {out_path}")


if __name__ == "__main__":
    main()
