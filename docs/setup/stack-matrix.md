# Reviewed Stack Matrix / 经审阅的技术栈矩阵

> [Environment hub](README.md) · [中文入口](README_CN.md)

**Reviewed / 审阅日期: 2026-08-20.** This page records teaching defaults, not universal compatibility. Before installing, open the linked upstream page and confirm its current platform and version table. / 本页记录教学默认组合，不代表所有组合都受支持；安装前必须打开官方链接重新核对。

| Host / 宿主 | ROS teaching default | Gazebo default | MuJoCo | Isaac Lab | Genesis World |
|---|---|---|---|---|---|
| Ubuntu 22.04 | Humble | Fortress | Official Python package | Check the current Isaac Lab matrix | Check current package prerequisites |
| Ubuntu 24.04 | Jazzy | Harmonic | Official Python package | Do not infer support from Ubuntu alone | Check current package prerequisites |
| WSL2 Ubuntu 22.04 | Humble | Fortress | Python; verify WSLg/headless rendering | Not asserted by this guide | Verify backend and rendering separately |
| WSL2 Ubuntu 24.04 | Jazzy | Harmonic | Python; verify WSLg/headless rendering | Not asserted by this guide | Verify backend and rendering separately |
| Windows 11 native | Use an upstream Windows ROS route or WSL2 | Not selected here | Official Python package | Upstream-supported Windows route | Official package; verify backend |

## Decision rules / 决策规则

1. Keep the default ROS/Gazebo pairing unless a dependency forces a non-default combination. / 初学者保持 ROS 与 Gazebo 默认组合。
2. Treat ROS Python packages and ML Python packages as different dependency domains. / ROS Python 与机器学习 Python 依赖分开管理。
3. Do not select CUDA from a static blog table. Match the driver, framework wheel, simulator, and OS using their current official selectors. / 不使用静态博客矩阵选择 CUDA。
4. WSL2 is a useful development host, but it is not evidence for hard real-time behavior, USB reliability, or robot safety. / WSL2 开发通过不能推出实时性、USB 稳定性或真机安全。
5. Record the chosen row and source access date in the experiment receipt. / 在实验回执中记录所选组合与来源访问日期。

## Primary sources / 权威来源

- [ROS 2 Jazzy Ubuntu packages](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- [ROS 2 Humble Ubuntu packages](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Gazebo and ROS version pairing](https://gazebosim.org/docs/jetty/ros_installation/)
- [MuJoCo Python installation](https://mujoco.readthedocs.io/en/stable/python.html)
- [Isaac Lab local installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/)
- [Genesis World installation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)
- [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install)

The executable source of this table is [`stack_matrix.json`](../../tools/robotdev/stack_matrix.json), validated by repository tests. / 本表的机器可读来源由仓库测试校验。
