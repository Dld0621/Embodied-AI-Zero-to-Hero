"""
Benchmark Provenance Utility
============================
Shared helper for all unified_pushcube_*.py benchmark scripts.

Each script calls ``build_provenance(command_str)`` and assigns the
result to ``results["provenance"]`` before writing JSON.  This ensures
provenance is always auto-generated at runtime — never hand-edited —
so timestamps, git commits, and library versions are accurate.

Usage
-----
.. code-block:: python

    from benchmark_provenance import build_provenance

    results["provenance"] = build_provenance(
        command="python unified_pushcube_rl.py --algo ppo --n_episodes 500",
    )
    with open(save_dir / "rl_results.json", "w") as f:
        json.dump(results, f, indent=2)
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path


def _get_git_commit() -> str:
    """Return short git commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_torch_version() -> str | None:
    """Return torch version string, or None if torch is not installed."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def _get_device() -> str:
    """Return the active compute device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    except ImportError:
        return "cpu"


def build_provenance(command: str, result_generated_by: str | None = None) -> dict:
    """Build a provenance dictionary for a benchmark result file.

    Parameters
    ----------
    command : str
        The exact command line used to run the benchmark.
    result_generated_by : str or None
        Path to the script that generated the result.  If None, uses
        this file's path (benchmark_provenance.py) — callers should
        pass their own ``__file__`` instead.

    Returns
    -------
    dict with keys: git_commit, command, python_version, torch_version,
                    device, timestamp, result_generated_by
    """
    return {
        "git_commit": _get_git_commit(),
        "command": command,
        "python_version": platform.python_version(),
        "torch_version": _get_torch_version(),
        "device": _get_device(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "result_generated_by": result_generated_by or __file__,
    }
