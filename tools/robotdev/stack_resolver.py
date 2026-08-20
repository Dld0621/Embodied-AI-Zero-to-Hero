#!/usr/bin/env python3
"""Resolve a reviewed teaching stack without installing or modifying the host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MATRIX = Path(__file__).with_name("stack_matrix.json")


def load_matrix(path: Path = MATRIX) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if data.get("schema_version") != 1 or not isinstance(profiles, list) or not profiles:
        raise ValueError("stack matrix must use schema_version 1 and contain profiles")
    return data


def resolve_profile(host: str, ubuntu: str | None = None) -> dict[str, Any]:
    if host == "windows":
        profile_id = "windows-11"
    elif host == "wsl2":
        profile_id = f"wsl2-ubuntu-{ubuntu}"
    else:
        profile_id = f"ubuntu-{ubuntu}"
    for profile in load_matrix()["profiles"]:
        if profile["id"] == profile_id:
            return profile
    raise ValueError(f"unsupported profile: {profile_id}")


def render_text(profile: dict[str, Any], reviewed_on: str) -> str:
    sources = "\n".join(f"  - {source}" for source in profile["sources"])
    return (
        f"Profile: {profile['id']}\n"
        f"OS: {profile['os']}\n"
        f"ROS: {profile['ros']}\n"
        f"Gazebo: {profile['gazebo']}\n"
        f"Notes: {profile['notes']}\n"
        f"Reviewed: {reviewed_on}\n"
        "Re-check:\n"
        f"{sources}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a reviewed robot-development teaching profile; never installs packages."
    )
    parser.add_argument("--host", choices=("ubuntu", "wsl2", "windows"), required=True)
    parser.add_argument("--ubuntu", choices=("22.04", "24.04"))
    parser.add_argument("--json", action="store_true", help="emit the selected profile as JSON")
    args = parser.parse_args()

    if args.host != "windows" and args.ubuntu is None:
        parser.error("--ubuntu is required for ubuntu and wsl2 hosts")
    if args.host == "windows" and args.ubuntu is not None:
        parser.error("--ubuntu does not apply to the native Windows profile")

    matrix = load_matrix()
    profile = resolve_profile(args.host, args.ubuntu)
    if args.json:
        print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(profile, matrix["reviewed_on"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
