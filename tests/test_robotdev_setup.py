from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "robotdev" / "stack_resolver.py"
MATRIX = ROOT / "tools" / "robotdev" / "stack_matrix.json"


def load_resolver():
    spec = importlib.util.spec_from_file_location("stack_resolver", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_has_unique_profiles_and_primary_sources() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    ids = [profile["id"] for profile in data["profiles"]]
    assert data["schema_version"] == 1
    assert len(ids) == len(set(ids)) == 5
    for profile in data["profiles"]:
        assert profile["sources"]
        assert all(source.startswith("https://") for source in profile["sources"])


@pytest.mark.parametrize(
    ("host", "ubuntu", "expected"),
    [
        ("ubuntu", "22.04", "ROS 2 Humble"),
        ("ubuntu", "24.04", "ROS 2 Jazzy"),
        ("wsl2", "24.04", "ROS 2 Jazzy"),
        ("windows", None, "Not selected by this apt-based guide"),
    ],
)
def test_resolver_selects_reviewed_profile(host: str, ubuntu: str | None, expected: str) -> None:
    profile = load_resolver().resolve_profile(host, ubuntu)
    assert profile["ros"] == expected


def test_cli_json_is_machine_readable() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--host", "wsl2", "--ubuntu", "22.04", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["id"] == "wsl2-ubuntu-22.04"
