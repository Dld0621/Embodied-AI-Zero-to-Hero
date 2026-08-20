# Genesis World Setup / Genesis World 环境

> [Environment hub](README.md) · [中文指南](#中文指南)

## English guide

The maintained project and PyPI distribution are named **Genesis World** and `genesis-world`. Follow the [current installation guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html), not a command copied from the earlier repository naming.

### Install

1. Create a clean Python environment.
2. Install PyTorch with the [official platform selector](https://pytorch.org/get-started/locally/).
3. Install Genesis World:

```bash
python -m pip install genesis-world
```

Check the current Python and platform range on the official page before creating the environment. Rendering extras and source-development extras are separate choices; install them only when the selected workflow requires them.

### Minimal CPU smoke

```python
import genesis as gs

gs.init(backend=gs.cpu, seed=0)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.build()
for _ in range(10):
    scene.step()
print("Genesis scene stepped")
```

This follows the official sequence: initialize once, create a scene, add entities, build, and step. First execution can include compilation work; record warm-up separately from steady-state timing.

### GPU and parallel environments

Verify three facts independently:

- the selected PyTorch build can see the intended accelerator;
- Genesis reports the selected backend rather than silently falling back;
- the scene and tensor batch dimensions match the number of environments.

Do not convert a vendor speed headline into a local throughput claim. Report scene complexity, backend, warm-up, number of environments, control decimation, rendering state, and measured steps per second on the local machine.

## 中文指南

当前维护的项目与 PyPI 包名是 **Genesis World** 和 `genesis-world`。先根据 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)安装与平台匹配的 PyTorch，再执行 `python -m pip install genesis-world`；Python 与平台范围以[当前官方安装页](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)为准。

最小程序遵循 `gs.init` → `Scene` → `add_entity` → `build` → `step`。第一次执行可能包含编译开销，因此预热时间与稳态吞吐必须分开记录。GPU 路径还要分别检查 PyTorch 是否看到加速器、Genesis 是否选择了预期后端、并行环境批维是否正确。官方宣传数字不能替代本机测量；报告吞吐时需同时记录场景复杂度、后端、环境数、控制降频和渲染状态。

权威来源：[Installation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html) · [Initialization](https://genesis-world.readthedocs.io/en/latest/user_guide/configuration/initialization.html) · [Hello Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html)
