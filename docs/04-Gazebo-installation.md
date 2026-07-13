# Gazebo Harmonic 安装指南

> Gazebo Sim — 机器人仿真平台，ROS 2 官方集成仿真器

## 版本说明

| Gazebo 版本 | 代号 | ROS 2 兼容 | 状态 |
|-------------|------|-----------|------|
| Gazebo Classic (11) | - | ROS 2 Humble | 已停止维护 (2025.01) |
| Gazebo Sim | Harmonic | ROS 2 Jazzy | **推荐** |
| Gazebo Sim | Fortress | ROS 2 Humble | LTS |

**注意：** Gazebo Classic 已于 2025 年 1 月终止生命周期，请迁移到 Gazebo Sim (Harmonic/Fortress)。

---

## Ubuntu 24.04 + ROS 2 Jazzy（推荐组合）

### 安装 Gazebo Harmonic

```bash
sudo apt update
sudo apt install -y gz-harmonic

# ROS 2 集成包
sudo apt install -y ros-jazzy-gazebo-ros-pkg

# 验证
gz sim --versions
```

### 运行 demo

```bash
# 启动空仿真世界
gz sim -v 4

# 启动 ROS 2 集成 demo
ros2 launch gazebo_ros gazebo.launch.py

# 运行示例机器人
ros2 launch gazebo_ros empty_world.launch.py
```

---

## Ubuntu 22.04 + ROS 2 Humble

### 安装 Gazebo Fortress

```bash
sudo apt update
sudo apt install -y gz-fortress
sudo apt install -y ros-humble-gazebo-ros-pkg

# 验证
gz sim --versions
```

---

## 创建自定义 URDF 机器人

### Step 1: 创建工作空间

```bash
mkdir -p ~/gazebo_ws/src
cd ~/gazebo_ws/src

# 创建 ROS 2 功能包
ros2 pkg create --build-type ament_python my_robot_description
```

### Step 2: 编写 URDF

```xml
<!-- my_robot.urdf -->
<?xml version="1.0"?>
<robot name="my_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### Step 3: 启动仿真

```bash
ros2 launch gazebo_ros gazebo.launch.py urdf_file:=my_robot.urdf
```

---

## 常见问题

### GUI 黑屏

```bash
# 更新显卡驱动
sudo apt install -y mesa-vulkan-drivers

# 或使用 NVIDIA 驱动
nvidia-smi
```

### `gz sim` 找不到命令

```bash
# 确认 Gazebo 已安装
dpkg -l | grep gz-harmonic

# 添加到 PATH
echo 'export PATH=/usr/bin/gz:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## 参考资料

- [Gazebo 官方文档](https://gazebosim.org/docs)
- [Gazebo + ROS 2 教程](https://gazebosim.org/docs/harmonic/ros2_integration.html)
