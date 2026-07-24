"""Regenerate benchmark bar chart from JSON results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    repo_root = Path(__file__).resolve().parent.parent
    json_path = repo_root / "benchmarks" / "benchmark_results.json"
    out_path = repo_root / "assets" / "demos" / "benchmark_bar_chart.png"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    methods = list(data["results"].keys())
    mean_fpe = [data["results"][m]["mean_fpe_mm"] for m in methods]

    colors = ["#4C78A8", "#F58518", "#E45756"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, mean_fpe, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean FPE (mm)", fontsize=12)
    ax.set_title("Mean Fingertip Position Error by Method\n(n=1000, seed=42)", fontsize=13)
    ax.set_ylim(0, max(mean_fpe) * 1.2)

    for bar, val in zip(bars, mean_fpe):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved benchmark chart to {out_path}")


if __name__ == "__main__":
    main()
