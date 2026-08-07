#!/usr/bin/env python3
"""Deterministic grid-navigation smoke test with guarded replanning.

The demo validates planning, tracking metrics, obstacle interception, and one
recovery path. It intentionally excludes continuous dynamics, legged
locomotion, learned policies, and real-robot safety claims.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Cell = tuple[int, int]
DEFAULT_OUTPUT = Path("results/pipelines/navigation/smoke/metrics.json")


@dataclass(frozen=True)
class Scenario:
    name: str
    width: int
    height: int
    start: Cell
    goal: Cell
    obstacles: frozenset[Cell]
    inject_dynamic_obstacle: bool = False


def _neighbors(cell: Cell, width: int, height: int, obstacles: set[Cell]) -> list[Cell]:
    x, y = cell
    candidates = ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
    return [
        candidate
        for candidate in candidates
        if 0 <= candidate[0] < width
        and 0 <= candidate[1] < height
        and candidate not in obstacles
    ]


def astar(
    start: Cell,
    goal: Cell,
    width: int,
    height: int,
    obstacles: set[Cell],
) -> list[Cell] | None:
    if start in obstacles or goal in obstacles:
        return None
    frontier: list[tuple[int, int, Cell]] = [(0, 0, start)]
    came_from: dict[Cell, Cell] = {}
    cost = {start: 0}
    serial = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for neighbor in _neighbors(current, width, height, obstacles):
            new_cost = cost[current] + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                serial += 1
                heuristic = abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
                heapq.heappush(frontier, (new_cost + heuristic, serial, neighbor))
    return None


def _scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "wall-gap",
            8,
            8,
            (0, 0),
            (7, 7),
            frozenset((3, y) for y in range(7) if y != 5),
        ),
        Scenario(
            "offset-corridor",
            9,
            7,
            (0, 3),
            (8, 3),
            frozenset({(4, 0), (4, 1), (4, 3), (4, 4), (4, 5), (6, 2)}),
        ),
        Scenario(
            "dynamic-replan",
            8,
            8,
            (0, 0),
            (7, 7),
            frozenset({(2, 2), (2, 3), (3, 3), (5, 4)}),
            inject_dynamic_obstacle=True,
        ),
    )


def _run_scenario(scenario: Scenario, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    obstacles = set(scenario.obstacles)
    current = scenario.start
    path = astar(current, scenario.goal, scenario.width, scenario.height, obstacles)
    if path is None:
        return {"name": scenario.name, "goal_reached": False, "planning_failed": True}

    path_index = 1
    moves = 0
    collisions = 0
    interventions = 0
    recovery_attempts = 0
    recovery_successes = 0
    dynamic_injected = False
    localization_squared_errors: list[float] = []
    tracking_squared_errors: list[float] = []

    while current != scenario.goal and moves < scenario.width * scenario.height * 4:
        if path_index >= len(path):
            break
        next_cell = path[path_index]

        if scenario.inject_dynamic_obstacle and not dynamic_injected and moves == 2:
            obstacles.add(next_cell)
            dynamic_injected = True

        if next_cell in obstacles:
            interventions += 1
            recovery_attempts += 1
            replanned = astar(current, scenario.goal, scenario.width, scenario.height, obstacles)
            if replanned is None:
                break
            recovery_successes += 1
            path = replanned
            path_index = 1
            continue

        actual_x = next_cell[0] + rng.uniform(-0.045, 0.045)
        actual_y = next_cell[1] + rng.uniform(-0.045, 0.045)
        estimate_x = actual_x + rng.uniform(-0.035, 0.035)
        estimate_y = actual_y + rng.uniform(-0.035, 0.035)
        tracking_squared_errors.append(
            (actual_x - next_cell[0]) ** 2 + (actual_y - next_cell[1]) ** 2
        )
        localization_squared_errors.append(
            (estimate_x - actual_x) ** 2 + (estimate_y - actual_y) ** 2
        )

        occupied_cell = (round(actual_x), round(actual_y))
        if occupied_cell in obstacles:
            collisions += 1
            break
        current = next_cell
        moves += 1
        path_index += 1

    return {
        "name": scenario.name,
        "goal_reached": current == scenario.goal,
        "planning_failed": False,
        "moves": moves,
        "collisions": collisions,
        "safety_interventions": interventions,
        "recovery_attempts": recovery_attempts,
        "recovery_successes": recovery_successes,
        "localization_squared_errors": localization_squared_errors,
        "tracking_squared_errors": tracking_squared_errors,
    }


def run_demo(seed: int = 11) -> dict[str, Any]:
    scenario_results = [
        _run_scenario(scenario, seed + index * 101)
        for index, scenario in enumerate(_scenarios())
    ]
    total_moves = sum(result.get("moves", 0) for result in scenario_results)
    total_collisions = sum(result.get("collisions", 0) for result in scenario_results)
    total_interventions = sum(
        result.get("safety_interventions", 0) for result in scenario_results
    )
    recovery_attempts = sum(
        result.get("recovery_attempts", 0) for result in scenario_results
    )
    recovery_successes = sum(
        result.get("recovery_successes", 0) for result in scenario_results
    )
    localization_errors = [
        value
        for result in scenario_results
        for value in result.get("localization_squared_errors", [])
    ]
    tracking_errors = [
        value
        for result in scenario_results
        for value in result.get("tracking_squared_errors", [])
    ]
    scenario_count = len(scenario_results)
    metrics = {
        "localization_or_state_error": math.sqrt(sum(localization_errors) / len(localization_errors)),
        "path_or_velocity_tracking_error": math.sqrt(sum(tracking_errors) / len(tracking_errors)),
        "goal_success_rate": sum(result["goal_reached"] for result in scenario_results) / scenario_count,
        "collision_or_fall_rate": total_collisions / scenario_count,
        "recovery_success_rate": recovery_successes / max(recovery_attempts, 1),
        "safety_intervention_rate": total_interventions / max(total_moves, 1),
    }
    checks = {
        "all_scenarios_reach_goal": metrics["goal_success_rate"] == 1.0,
        "zero_collisions": metrics["collision_or_fall_rate"] == 0.0,
        "localization_rmse_below_0_10_cells": metrics["localization_or_state_error"] < 0.10,
        "tracking_rmse_below_0_10_cells": metrics["path_or_velocity_tracking_error"] < 0.10,
        "dynamic_recovery_succeeds": recovery_attempts > 0 and recovery_successes == recovery_attempts,
    }
    public_scenarios = [
        {key: value for key, value in result.items() if not key.endswith("_squared_errors")}
        for result in scenario_results
    ]
    return {
        "schema_version": 1,
        "pipeline": "navigation-locomotion",
        "evidence": {
            "level": "synthetic-smoke",
            "supports": "grid planning, tracking metrics, safety interception, and replanning wiring",
            "does_not_support": "continuous dynamics, legged locomotion, learned-policy, or real-robot safety claims",
        },
        "configuration": {"seed": seed, "scenario_count": scenario_count},
        "metrics": metrics,
        "metric_units": {
            "localization_or_state_error": "grid_cell_rmse",
            "path_or_velocity_tracking_error": "grid_cell_rmse",
            "goal_success_rate": "successful_scenarios_over_scenarios",
            "collision_or_fall_rate": "colliding_scenarios_over_scenarios",
            "recovery_success_rate": "successful_replans_over_replan_attempts",
            "safety_intervention_rate": "interventions_over_executed_moves",
        },
        "checks": checks,
        "scenarios": public_scenarios,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="return non-zero when a gate fails")
    args = parser.parse_args()

    report = run_demo(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 1 if args.check and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
