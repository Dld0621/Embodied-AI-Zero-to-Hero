"""生成 README 中文版使用的教学规模强化学习曲线。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rng = np.random.default_rng(seed=42)
    episodes = np.arange(1, 501)
    reward_mean = -50 + 55 * (1 - np.exp(-episodes / 120))
    rewards = reward_mean + rng.normal(0, 3, size=len(episodes))
    success_mean = 0.85 * (1 - np.exp(-episodes / 100))
    success_rates = np.clip(success_mean + rng.normal(0, 0.03, size=len(episodes)), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(episodes, rewards, color="#4C78A8", alpha=0.8, linewidth=1.2)
    axes[0].set_xlabel("训练回合")
    axes[0].set_ylabel("回合奖励")
    axes[0].set_title("训练过程中的回合奖励")
    axes[1].plot(episodes, success_rates, color="#F58518", alpha=0.8, linewidth=1.2)
    axes[1].set_xlabel("训练回合")
    axes[1].set_ylabel("成功率")
    axes[1].set_title("训练过程中的成功率")
    axes[1].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15)
    fig.text(0.5, 0.02, "示意曲线——并非已完成的 SAC+HER 基准结果",
             ha="center", fontsize=10, style="italic", color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = Path(__file__).resolve().parents[1] / "assets" / "demos" / "learning_curves-cn.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Chinese RL curves to {out_path}")


if __name__ == "__main__":
    main()
