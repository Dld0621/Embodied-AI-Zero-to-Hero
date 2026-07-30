"""
Rule-Based Task Planner
=======================
Simple deterministic planner that decomposes high-level language
instructions into sub-goals for the foundation model.

This is the simplest possible "embodied reasoner" — no LLM, no neural
network, just hand-coded rules.  It serves as:
1. A baseline for comparing against learned planners.
2. A fallback when VLM-based planners are unavailable.
3. A teaching tool for understanding task decomposition.

Example
-------
.. code-block:: python

    from examples.robot_foundation_models.planners.rule_based_planner import RuleBasedPlanner

    planner = RuleBasedPlanner()
    sub_goals = planner.plan("push the red cube to the target")
    # [
    #   {"action": "locate", "target": "red cube"},
    #   {"action": "approach", "target": "red cube"},
    #   {"action": "push", "target": "red cube", "destination": "target"},
    #   {"action": "verify", "condition": "red cube at target"},
    # ]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SubGoal:
    """A single step in a decomposed task plan."""
    action: str              # "locate", "approach", "push", "grasp", "place", "verify"
    target: str = ""         # object name
    destination: str = ""    # where to move it (for push/place)
    condition: str = ""      # success condition (for verify)
    language: str = ""       # refined language instruction for the VLA


@dataclass
class TaskPlan:
    """A complete decomposed plan for a task."""
    original_instruction: str
    sub_goals: List[SubGoal] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [f"TaskPlan({self.original_instruction!r})"]
        for i, sg in enumerate(self.sub_goals):
            lines.append(f"  [{i}] {sg.action}: {sg.target} → {sg.destination or sg.condition}")
        return "\n".join(lines)


class RuleBasedPlanner:
    """Decomposes natural-language instructions into sub-goals.

    Supported patterns:
    - "push X to Y" → locate → approach → push → verify
    - "pick up X" → locate → approach → grasp → lift
    - "place X on Y" → locate → approach → grasp → move → place → release
    - "move to X" → locate → move

    The planner is intentionally simple — it pattern-matches on keywords
    rather than using any NLP.  This makes it fully deterministic and
    easy to debug.
    """

    # Pattern → action mapping
    PATTERNS = [
        (r"push\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+)", "push"),
        (r"pick\s+up\s+(?:the\s+)?(.+)", "pick_up"),
        (r"grab\s+(?:the\s+)?(.+)", "pick_up"),
        (r"place\s+(?:the\s+)?(.+?)\s+on\s+(?:the\s+)?(.+)", "place"),
        (r"put\s+(?:the\s+)?(.+?)\s+on\s+(?:the\s+)?(.+)", "place"),
        (r"move\s+to\s+(?:the\s+)?(.+)", "move"),
        (r"go\s+to\s+(?:the\s+)?(.+)", "move"),
    ]

    def plan(self, instruction: str) -> TaskPlan:
        """Decompose an instruction into sub-goals."""
        instruction_lower = instruction.lower().strip()

        for pattern, action_type in self.PATTERNS:
            match = re.match(pattern, instruction_lower)
            if match:
                if action_type == "push":
                    return self._plan_push(instruction, match.group(1), match.group(2))
                elif action_type == "pick_up":
                    return self._plan_pick_up(instruction, match.group(1))
                elif action_type == "place":
                    return self._plan_place(instruction, match.group(1), match.group(2))
                elif action_type == "move":
                    return self._plan_move(instruction, match.group(1))

        # Fallback: single-step plan
        return TaskPlan(
            original_instruction=instruction,
            sub_goals=[SubGoal(
                action="execute",
                language=instruction,
            )],
        )

    # ------------------------------------------------------------------
    # Pattern-specific planners
    # ------------------------------------------------------------------
    def _plan_push(self, instruction: str, target: str, destination: str) -> TaskPlan:
        return TaskPlan(
            original_instruction=instruction,
            sub_goals=[
                SubGoal(action="locate", target=target,
                        language=f"find the {target}"),
                SubGoal(action="approach", target=target,
                        language=f"move behind the {target}"),
                SubGoal(action="push", target=target, destination=destination,
                        language=f"push the {target} to the {destination}"),
                SubGoal(action="verify", condition=f"{target} at {destination}",
                        language=f"check if the {target} is at the {destination}"),
            ],
        )

    def _plan_pick_up(self, instruction: str, target: str) -> TaskPlan:
        return TaskPlan(
            original_instruction=instruction,
            sub_goals=[
                SubGoal(action="locate", target=target,
                        language=f"find the {target}"),
                SubGoal(action="approach", target=target,
                        language=f"move to the {target}"),
                SubGoal(action="grasp", target=target,
                        language=f"grasp the {target}"),
                SubGoal(action="lift", target=target,
                        language=f"lift the {target}"),
            ],
        )

    def _plan_place(self, instruction: str, target: str, destination: str) -> TaskPlan:
        return TaskPlan(
            original_instruction=instruction,
            sub_goals=[
                SubGoal(action="locate", target=target,
                        language=f"find the {target}"),
                SubGoal(action="approach", target=target,
                        language=f"move to the {target}"),
                SubGoal(action="grasp", target=target,
                        language=f"grasp the {target}"),
                SubGoal(action="move", target=target, destination=destination,
                        language=f"move to the {destination}"),
                SubGoal(action="place", target=target, destination=destination,
                        language=f"place the {target} on the {destination}"),
                SubGoal(action="release", target=target,
                        language=f"release the {target}"),
            ],
        )

    def _plan_move(self, instruction: str, target: str) -> TaskPlan:
        return TaskPlan(
            original_instruction=instruction,
            sub_goals=[
                SubGoal(action="locate", target=target,
                        language=f"find the {target}"),
                SubGoal(action="move", target=target,
                        language=f"move to the {target}"),
            ],
        )


if __name__ == "__main__":
    planner = RuleBasedPlanner()

    tests = [
        "push the red cube to the target",
        "pick up the blue cup",
        "place the block on the table",
        "move to the charging station",
    ]

    for t in tests:
        plan = planner.plan(t)
        print(plan)
        print()
