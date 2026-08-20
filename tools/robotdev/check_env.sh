#!/usr/bin/env bash
# Read-only environment inventory. This script never installs or changes packages.
set -uo pipefail

found=0
missing=0

probe() {
  local label="$1"
  shift
  if command -v "$1" >/dev/null 2>&1; then
    printf '[FOUND] %-12s ' "$label"
    "$@" 2>&1 | head -n 1 || true
    found=$((found + 1))
  else
    printf '[INFO ] %-12s not found (may be optional)\n' "$label"
    missing=$((missing + 1))
  fi
}

printf 'Robot development environment inventory\n'
printf 'Kernel: %s\n' "$(uname -srmo 2>/dev/null || uname -a)"
if grep -qi microsoft /proc/version 2>/dev/null; then
  printf 'Host mode: WSL\n'
else
  printf 'Host mode: native or containerized Unix\n'
fi

probe Python python3 --version
probe Pip python3 -m pip --version
probe Git git --version
probe CMake cmake --version
probe ROS2 ros2 --help
probe Gazebo gz --version
probe NVIDIA nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
probe Docker docker --version

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY' || true
from importlib.util import find_spec

for package in ("mujoco", "torch", "genesis"):
    state = "installed" if find_spec(package) else "not installed"
    print(f"[PYTHON] {package:<10} {state}")
PY
fi

printf 'Summary: %d command(s) found; %d optional command(s) absent.\n' "$found" "$missing"
printf 'No package or system setting was changed.\n'
