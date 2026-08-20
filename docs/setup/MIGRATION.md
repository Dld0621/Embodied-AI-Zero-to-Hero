# RobotDev Setup Guide Consolidation / 开发环境指南合并记录

The former first-party repository `Dld0621/RobotDev-Setup-Guide` was consolidated into this module from source commit `361f098f48a2b0d418c9f1db2f45a9316d4bac73` on 2026-08-20. Its Git ancestry is retained when the consolidation merge is published. / 原独立仓库在上述提交基础上合入本模块；发布时保留其 Git 提交祖先。

## Coverage map / 内容映射

| Former area | Integrated destination | Decision |
|---|---|---|
| ROS 2 | [`ros2-gazebo.md`](ros2-gazebo.md) | Re-authored around official repositories, valid workspace commands, and evidence gates. |
| Gazebo | [`ros2-gazebo.md`](ros2-gazebo.md) | Replaced Gazebo Classic defaults with the supported `ros_gz` pairing route. |
| MuJoCo | [`mujoco.md`](mujoco.md) and [scene tutorial](../tutorials/mujoco-scene-building.md) | Corrected XML loading, rendering, and GPU-boundary explanations. |
| Isaac Lab | [`isaac-lab.md`](isaac-lab.md) | Removed stale launcher/module commands and delegated volatile pins to current official docs. |
| Genesis | [`genesis.md`](genesis.md) | Updated project/package naming and replaced unverified pseudo-code with the official minimal sequence. |
| CUDA and WSL2 | [`python-cuda-wsl.md`](python-cuda-wsl.md) | Separated driver, toolkit, framework runtime, and application backend evidence. |
| Python | [`python-cuda-wsl.md`](python-cuda-wsl.md) | Removed generic `pip install rclpy`, certificate bypasses, and global-environment advice. |
| Troubleshooting | [`troubleshooting.md`](troubleshooting.md) | Converted ad-hoc fixes into a read-only, layered decision table. |
| Environment script | [`tools/robotdev/check_env.sh`](../../tools/robotdev/check_env.sh) | Rewritten as non-mutating inventory; optional GPU/tools no longer make the script fail. |
| Install scripts | No direct replacement | Intentionally retired because they hard-coded mirrors, versions, drivers, or privileged mutations. |

## Preserved license / 许可保留

The source guide was released under the MIT License, copyright 2025 Dld0621. The integrated and re-authored material remains covered by this repository's [MIT License](../../LICENSE). This record preserves origin and scope even after the standalone remote repository is removed. / 原指南采用 MIT License，版权归 2025 Dld0621；合并并重写后的内容继续遵循本仓库 MIT License。本记录在独立远端删除后仍保留来源与范围。

## Accuracy boundary / 准确性边界

The migration audit corrected known errors but is not a permanent certificate for external software. Version-sensitive statements carry a review date and primary-source link, and the automated checks verify structure and known regressions rather than every future upstream change. / 本次审校修复已知问题，但不能永久认证外部软件；易变结论必须带审阅日期和官方来源，自动检查只验证结构与已知回归。
