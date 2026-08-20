# Robot Development Environment

> English · [简体中文](README_CN.md) · [Compatibility matrix](stack-matrix.md) · [Migration record](MIGRATION.md)

This module turns a new workstation into a **measurable development environment** for embodied-AI work. It covers installation decisions, isolation, smoke checks, simulator selection, ROS integration, and reproducibility. It does not treat “the command completed” as proof that training, rendering, a task, or a real robot works.

## Choose one host path

| Goal | Recommended starting point | Why |
|---|---|---|
| ROS 2 and Gazebo coursework | Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic | Reviewed LTS teaching pair with ROS-distributed `ros_gz` packages. |
| Existing Ubuntu 22.04 robot stack | ROS 2 Humble + Gazebo Fortress | Keeps the default Humble binary pairing. |
| Windows laptop for ROS learning | WSL2 with Ubuntu 24.04 | Linux ROS packages plus WSLg, without pretending WSL is real-time hardware validation. |
| Fast contact simulation and model editing | Native Python + MuJoCo | Small install, direct MJCF/URDF workflow, CPU physics and optional rendering. |
| GPU-parallel robot learning | Isaac Lab or Genesis World | Select only after checking the current upstream OS, Python, driver, and accelerator matrix. |

Print the reviewed teaching profile without changing the machine:

```bash
python tools/robotdev/stack_resolver.py --host ubuntu --ubuntu 24.04
bash tools/robotdev/check_env.sh
```

## Setup pipeline

```text
research goal
    ↓
host + accelerator inventory
    ↓
supported version pair from upstream docs
    ↓
isolated environment and pinned dependencies
    ↓
import → minimal scene → renderer → ROS bridge
    ↓
task smoke → deterministic regression → benchmark
    ↓
hardware-specific safety review (separate gate)
```

| Gate | Pass condition | Does not prove |
|---|---|---|
| Import | Package imports and reports a version. | Renderer, physics task, or GPU is correct. |
| Scene | A minimal world builds and advances. | Your robot model is physically faithful. |
| Render | An interactive or headless frame is produced. | Training uses the intended accelerator. |
| Bridge | Named messages cross ROS and simulator with timestamps. | Stable control latency or hardware safety. |
| Task | A fixed seed and metric pass a scoped task. | Generalization or Sim-to-Real transfer. |

## Guides

| Guide | Outcome |
|---|---|
| [Compatibility matrix](stack-matrix.md) | Choose a host and keep changing version claims dated. |
| [ROS 2 + Gazebo](ros2-gazebo.md) | Build a workspace, install the default pairing, and verify the bridge boundary. |
| [MuJoCo](mujoco.md) | Install the official Python bindings and run a valid MJCF smoke. |
| [Isaac Lab](isaac-lab.md) | Follow the current supported install route and separate app launch from task evidence. |
| [Genesis World](genesis.md) | Install the current package and execute the official minimal scene pattern. |
| [Python, CUDA, and WSL2](python-cuda-wsl.md) | Isolate environments and avoid driver/toolkit confusion. |
| [Troubleshooting](troubleshooting.md) | Diagnose by layer instead of applying unsafe global fixes. |

## Reproducibility receipt

For every environment that becomes a research baseline, record:

- host OS and whether it is native, WSL2, containerized, or remote;
- Python, simulator, ROS, Gazebo, PyTorch, driver, and package-lock versions;
- GPU model, CPU architecture, RAM, and rendering backend;
- exact install source, command, Git commit, seed, and smoke-test output;
- evidence level: import, scene, render, bridge, task, benchmark, or hardware.

The compatibility data is machine-readable in [`tools/robotdev/stack_matrix.json`](../../tools/robotdev/stack_matrix.json). Its review date is a freshness boundary, not a promise that external packages never change.
