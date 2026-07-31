"""
PushCube Dataset Collector for SmolVLA
========================================
Collect demonstration episodes from the dual-cube PushCube environment
using the expert heuristic policy, then save in canonical format.

The expert policy uses a two-phase approach:
1. Move behind the active cube (opposite from target).
2. Push toward the target.

Usage
-----
.. code-block:: bash

    # Collect 50 episodes (default)
    python collect_pushcube_dataset.py --n_episodes 50 --output datasets/pushcube_canonical/

    # Quick smoke test (5 episodes)
    python collect_pushcube_dataset.py --n_episodes 5 --output /tmp/pushcube_smoke/

    # Visualize one episode
    python collect_pushcube_dataset.py --viz --output datasets/pushcube_canonical/

Output Structure
----------------
::

    output_dir/
    ├── episode_0000.pkl
    ├── episode_0001.pkl
    ├── ...
    └── dataset_info.json

Each ``.pkl`` is a ``CanonicalEpisode`` with images (128x128 RGB),
14-D state, 2-D action, language instruction, reward, and success flag.

Connecting to SmolVLA
---------------------
After collection, convert to LeRobot format::

    python -c "
    from examples.robot_foundation_models.common.canonical_dataset import load_episodes_from_dir
    from examples.robot_foundation_models.common.to_lerobot import convert_to_lerobot
    episodes = load_episodes_from_dir('datasets/pushcube_canonical/')
    convert_to_lerobot(episodes, 'datasets/pushcube_lerobot/')
    "

Then fine-tune::

    python finetune.py --dataset_dir datasets/pushcube_lerobot/ --output_dir models/smolvlа_pushcube/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.unified_pushcube_env import PushCubeEnv, expert_action
from examples.robot_foundation_models.common.canonical_dataset import (
    CanonicalEpisode,
    EpisodeBuilder,
)


def collect_episode(env: PushCubeEnv, seed: int, render: bool = False) -> CanonicalEpisode:
    """Collect one episode using the expert policy.

    Returns
    -------
    CanonicalEpisode with all timesteps recorded.
    """
    obs = env.reset(seed=seed)
    lang = env.get_language_instruction()

    builder = EpisodeBuilder(
        task=lang,
        robot_type="pushcube_2d",
        control_frequency=20.0,
        action_type="ee_delta_2d",
        metadata={"seed": seed, "active_idx": int(env.active_idx), "active_color": int(env.active_color_idx)},
    )

    for step in range(env.max_steps):
        # Render image
        img = env.render(size=128)
        img_uint8 = (img * 255).astype(np.uint8)
        state = env.get_state_vector()

        # Expert action
        action = expert_action(env)
        reward = env._compute_reward()
        success = env._check_success()

        builder.add_step(
            observation={"images": {"front": img_uint8}, "state": state},
            action=action,
            language=lang,
            reward=reward,
            success=success,
            timestamp=step / 20.0,
        )

        if render and step % 5 == 0:
            print(f"  Step {step}: arm={env.arm_pos.round(3)}, reward={reward:.3f}")

        # Step environment
        env.step(action)

        if success:
            break

    return builder.to_episode()


def collect_dataset(
    n_episodes: int = 50,
    output_dir: str = "datasets/pushcube_canonical/",
    seed_start: int = 1000,
    render: bool = False,
) -> List[CanonicalEpisode]:
    """Collect multiple episodes and save to disk.

    Returns
    -------
    List of collected episodes (also saved to ``output_dir``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes: List[CanonicalEpisode] = []
    successes = 0

    print(f"Collecting {n_episodes} episodes to {output_dir}")
    print("-" * 50)

    for i in range(n_episodes):
        env = PushCubeEnv()
        seed = seed_start + i

        t0 = time.time()
        ep = collect_episode(env, seed=seed, render=render)
        dt = time.time() - t0

        # Determine final success
        final_success = any(ep.success)
        if final_success:
            successes += 1

        episodes.append(ep)
        ep.save(output_dir / f"episode_{i:04d}.pkl")

        print(
            f"Episode {i+1}/{n_episodes}: "
            f"length={ep.length}, success={final_success}, "
            f"time={dt:.2f}s"
        )

    success_rate = successes / max(1, n_episodes) * 100
    print("-" * 50)
    print(f"Collection complete: {successes}/{n_episodes} successes ({success_rate:.1f}%)")

    # Write dataset metadata
    meta = {
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "seed_start": seed_start,
        "avg_length": float(np.mean([ep.length for ep in episodes])),
        "action_dim": episodes[0].action_dim if episodes else 0,
        "state_dim": episodes[0].state_dim if episodes else 0,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    return episodes


def visualize_episode(episode: CanonicalEpisode, output_path: str = "episode_viz.png"):
    """Save a grid of frames from an episode."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    imgs = episode.observation["images"]["front"]
    n = min(len(imgs), 8)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        idx = i * (len(imgs) - 1) // max(1, n - 1)
        axes[i].imshow(imgs[idx])
        axes[i].set_title(f"t={idx}\n{episode.language[idx][:20]}...")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect PushCube expert demonstrations")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--output", default="datasets/pushcube_canonical/", help="Output directory")
    parser.add_argument("--seed_start", type=int, default=1000, help="Random seed offset")
    parser.add_argument("--render", action="store_true", help="Print per-step debug info")
    parser.add_argument("--viz", action="store_true", help="Visualize first episode")
    args = parser.parse_args()

    episodes = collect_dataset(
        n_episodes=args.n_episodes,
        output_dir=args.output,
        seed_start=args.seed_start,
        render=args.render,
    )

    if args.viz and episodes:
        viz_path = Path(args.output) / "episode_000_viz.png"
        visualize_episode(episodes[0], str(viz_path))


if __name__ == "__main__":
    main()
