# Python, CUDA, and WSL2 / Python、CUDA 与 WSL2

> [Environment hub](README.md) · [中文指南](#中文指南)

## English guide

### Keep dependency domains separate

| Domain | Environment rule |
|---|---|
| ROS 2 binary packages | Begin with the Python provided by the matching Ubuntu/ROS distribution. Install `rclpy` through ROS packages, not generic PyPI. |
| MuJoCo and teaching examples | Use a project-local `venv` and a pinned requirements or lock file. |
| VLA / PyTorch | Use a dedicated environment selected from the current PyTorch wheel matrix. |
| Isaac Lab | Use the exact Python and package combination required by the current Isaac Sim release. |
| Genesis World | Install PyTorch first, then Genesis, in its own reproducible environment. |

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install --upgrade pip
python -m pip freeze > environment-observed.txt
```

The observed freeze is a receipt, not automatically a portable lock across operating systems and architectures. For stronger repeatability, use reviewed pins and hashes where the ecosystem permits them. See the official [Python `venv`](https://docs.python.org/3/library/venv.html) and [pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/) guidance.

### Separate the NVIDIA layers

```text
Windows/Linux NVIDIA driver
        ↓ exposes device capability
CUDA toolkit (nvcc, headers; needed for compilation workflows)
        ↓ optional for many prebuilt wheels
framework runtime bundled/selected by PyTorch or simulator
        ↓
application backend actually selected at runtime
```

- `nvidia-smi` reports driver-visible GPU information; it does not identify the CUDA runtime bundled with every Python wheel.
- `nvcc --version` reports an installed toolkit compiler; it does not prove PyTorch uses that toolkit.
- Use the [PyTorch selector](https://pytorch.org/get-started/locally/) for a supported wheel rather than constructing an index URL from an arbitrary local toolkit version.
- Use the current [NVIDIA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for toolkit work; prefer its distribution package route unless the target product documents another requirement.

### WSL2

From an elevated Windows terminal, follow [Microsoft's current WSL install page](https://learn.microsoft.com/en-us/windows/wsl/install/):

```powershell
wsl --install
wsl --update
wsl --status
```

For CUDA on WSL, install the supported NVIDIA driver on **Windows**. NVIDIA explicitly warns not to install a Linux display driver inside WSL; if a toolkit is needed in WSL, follow the constrained toolkit package names in the [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

Record WSL distribution, kernel, WSLg, Windows driver, filesystem location, and whether Docker runs through Desktop or inside the distribution. Performance and device access must be measured for the target workflow.

## 中文指南

ROS 2、MuJoCo、VLA/PyTorch、Isaac Lab 与 Genesis World 应使用不同的依赖域。ROS 二进制包先使用对应 Ubuntu/ROS 的系统 Python，`rclpy` 通过 ROS 包安装；机器学习与仿真项目使用项目级虚拟环境和经审阅的锁文件，不要在同一个环境里不断覆盖依赖。

NVIDIA 驱动、CUDA Toolkit、框架自带运行时和应用实际选择的后端是四层不同证据。`nvidia-smi` 成功不代表 Python 框架兼容，`nvcc --version` 也不代表 PyTorch 正在使用该 Toolkit。PyTorch 安装命令应来自[官方选择器](https://pytorch.org/get-started/locally/)。

WSL2 按[微软官方说明](https://learn.microsoft.com/en-us/windows/wsl/install/)安装和更新。CUDA on WSL 只在 Windows 安装受支持的 NVIDIA 驱动，不在 WSL 内安装 Linux 显示驱动；需要 Toolkit 时遵循 [NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)列出的受限包名。WSL 开发成功不等于 USB、实时性或真机控制已经验证。
