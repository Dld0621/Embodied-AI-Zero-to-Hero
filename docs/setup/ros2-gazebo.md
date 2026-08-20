# ROS 2 + Gazebo

> [Environment hub](README.md) · [Compatibility matrix](stack-matrix.md) · [中文指南](#中文指南)

## English guide

### 1. Freeze the supported pair

For this curriculum, use the default binary pair exposed by the ROS repository:

| Ubuntu | ROS 2 | Gazebo |
|---|---|---|
| 22.04 | Humble | Fortress |
| 24.04 | Jazzy | Harmonic |

The [Gazebo pairing guide](https://gazebosim.org/docs/jetty/ros_installation/) explains why beginners should prefer the default pair. A non-default pair is an advanced dependency decision, not an upgrade by definition.

### 2. Install from official repositories

Follow the matching ROS page first: [Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) or [Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html). Do not paste an old signing key or third-party mirror configuration from this repository.

After the official ROS repository is configured and `ROS_DISTRO` is set to the installed distribution:

```bash
sudo apt update
sudo apt install ros-${ROS_DISTRO}-ros-gz
```

This selects the Gazebo libraries recommended for that ROS distribution. It is not the same as the retired Gazebo Classic `gazebo_ros_pkgs` path.

### 3. Build an isolated workspace

```bash
mkdir -p ~/robot_ws/src
cd ~/robot_ws/src
ros2 pkg create --build-type ament_python --license Apache-2.0 robotdev_demo
cd ~/robot_ws
colcon build --symlink-install
source install/setup.bash
ros2 pkg prefix robotdev_demo
```

Keep the underlay (`/opt/ros/<distro>`) and overlay (`~/robot_ws/install`) order visible. When a package disappears, open a clean terminal, source the underlay, then the overlay, and inspect `AMENT_PREFIX_PATH` before rebuilding.

### 4. Verify each boundary

```bash
printenv ROS_DISTRO
ros2 doctor --report
gz sim --versions
ros2 pkg prefix ros_gz_bridge
```

Then run one official [`ros_gz` demo](https://gazebosim.org/docs/harmonic/ros2_integration/) and verify:

- the expected topic name and message type exist on both sides;
- timestamps advance and use the intended simulation clock;
- commands do not silently cross an unintended namespace;
- the simulator can pause, reset, and restart without stale publishers.

### Evidence boundary

Package discovery proves installation. A visible Gazebo world proves rendering. A bridged topic proves message transport. None proves controller stability, real-time latency, or safe hardware behavior.

## 中文指南

### 1. 固定受支持组合

本课程默认使用 ROS 仓库提供的组合：Ubuntu 22.04 对应 Humble + Fortress；Ubuntu 24.04 对应 Jazzy + Harmonic。初学者不要因为版本号更大就混装非默认组合，具体边界以 [Gazebo 官方配对说明](https://gazebosim.org/docs/jetty/ros_installation/) 为准。

### 2. 只跟随官方安装页

先按 [Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) 或 [Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 官方页面配置软件源。本仓库不保存可能过期的签名密钥或固定镜像源。完成 ROS 安装后，再安装默认桥接包：

```bash
sudo apt update
sudo apt install ros-${ROS_DISTRO}-ros-gz
```

这里使用现代 Gazebo 的 `ros_gz`，不要与 Gazebo Classic 的旧 `gazebo_ros_pkgs` 教程混用。

### 3. 建立 Overlay 工作空间

```bash
mkdir -p ~/robot_ws/src
cd ~/robot_ws/src
ros2 pkg create --build-type ament_python --license Apache-2.0 robotdev_demo
cd ~/robot_ws
colcon build --symlink-install
source install/setup.bash
ros2 pkg prefix robotdev_demo
```

排查“找不到包”时，先在干净终端依次 source 系统 Underlay 和工作空间 Overlay，再检查 `AMENT_PREFIX_PATH`，不要先删除整个工作空间。

### 4. 分层验收

依次检查 ROS 发行版、`ros2 doctor`、Gazebo 版本与 `ros_gz_bridge` 包；再运行 [官方桥接示例](https://gazebosim.org/docs/harmonic/ros2_integration/)，核对 topic、类型、时间戳、命名空间、暂停与重置行为。

软件包可发现只证明安装完成；出现 Gazebo 窗口只证明渲染；消息能跨桥只证明通信。三者都不能证明控制稳定、实时性或真机安全。
