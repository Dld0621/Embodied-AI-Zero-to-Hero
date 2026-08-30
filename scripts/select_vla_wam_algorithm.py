#!/usr/bin/env python3
"""Explainable VLA/WAM algorithm-family selector.

The selector narrows an experiment starting point. It deliberately returns
algorithm families rather than claiming that one named model is universally
best. Repository evidence and deployment authorization remain separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "learning_tracks" / "vla_wam_algorithms.json"


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def recommend(
    *, goal: str, compute: str, data: str, latency: str
) -> tuple[list[str], list[str]]:
    """Return an ordered shortlist and explicit boundary notes."""

    notes: list[str] = []

    if goal == "single-task-control":
        choices = ["direct-chunked-bc", "diffusion-action-policy"]
        notes.append("A VLA is not the default when language and broad task variation are not causal.")
    elif goal == "language-generalization":
        if latency == "hard" or compute == "limited":
            choices = ["continuous-chunk-vla", "direct-chunked-bc", "discrete-token-vla"]
        else:
            choices = ["continuous-chunk-vla", "flow-matching-vla", "discrete-token-vla"]
        notes.append("Use correct, swapped, paraphrased, and absent-language ablations.")
    elif goal == "multimodal-action":
        choices = ["diffusion-action-policy", "flow-matching-vla", "direct-chunked-bc"]
        notes.append("Confirm that the data actually contains multiple valid actions for similar context.")
    elif goal == "model-based-planning":
        choices = ["latent-world-model-mpc", "direct-chunked-bc"]
        notes.append("This is a world-model baseline, not a WAM under the narrow joint-modeling definition.")
    else:
        if compute == "cluster" and data == "heterogeneous":
            choices = ["joint-video-action-wam", "autoregressive-joint-wam", "latent-world-model-mpc"]
        else:
            choices = ["latent-world-model-mpc", "autoregressive-joint-wam", "joint-video-action-wam"]
            notes.append("Do not start with a large video WAM before a latent world-model baseline passes.")
        notes.append("Evaluate video-action alignment and closed-loop task behavior separately.")

    if latency == "hard":
        iterative = {"diffusion-action-policy", "flow-matching-vla", "joint-video-action-wam"}
        choices = [item for item in choices if item not in iterative] + [
            item for item in choices if item in iterative
        ]
        notes.append("Hard latency budgets require measured end-to-end timing; family names do not guarantee rate.")

    if data == "task-specific" and choices[0] in {
        "discrete-token-vla",
        "flow-matching-vla",
        "autoregressive-joint-wam",
        "joint-video-action-wam",
    }:
        choices.insert(0, "direct-chunked-bc")
        notes.append("Task-specific data requires a smaller matched-budget policy baseline.")

    return list(dict.fromkeys(choices)), list(dict.fromkeys(notes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal",
        required=True,
        choices=[
            "single-task-control",
            "language-generalization",
            "multimodal-action",
            "model-based-planning",
            "future-video-and-action",
        ],
    )
    parser.add_argument("--compute", required=True, choices=["limited", "single-gpu", "cluster"])
    parser.add_argument(
        "--data", required=True, choices=["task-specific", "multi-task", "heterogeneous"]
    )
    parser.add_argument("--latency", required=True, choices=["hard", "soft"])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    args = parser.parse_args()

    catalog = load_catalog()
    by_id = {item["id"]: item for item in catalog["families"]}
    choice_ids, notes = recommend(
        goal=args.goal, compute=args.compute, data=args.data, latency=args.latency
    )
    result = {
        "inputs": vars(args) | {"json": None},
        "recommendations": [by_id[item] for item in choice_ids],
        "notes": notes,
        "boundary": catalog["scope"],
    }
    result["inputs"].pop("json")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Recommended starting order:")
        for index, item in enumerate(result["recommendations"], 1):
            print(f"{index}. {item['label']} [{item['track']}; {item['maturity']}]")
            print(f"   predicts: {item['predicts']}")
            print(f"   main risk: {item['risks'][0]}")
        print("\nBoundary notes:")
        for note in result["notes"]:
            print(f"- {note}")
        print(f"- {result['boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
