# ROS 2 安装指南

> Robot Operating System 2 | 适用于 Ubuntu 22.04 / 24.04

## 版本选择

| Ubuntu 版本 | ROS 2 版本 | 状态 | 支持到 |
|-------------|-----------|------|--------|
| 22.04 LTS | Humble Hawksbill | LTS | 2027 年 5 月 |
| 24.04 LTS | Jazzy Jalisco | LTS | 2029 年 5 月 |

**核心原则：Ubuntu 版本与 ROS 2 版本严格绑定，不可混装！**

---

## 1. 设置 locale

```bash
# 确保支持 UTF-8
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

## 2. 添加 ROS 2 apt 仓库

```bash
# 安装依赖
sudo apt install -y software-properties-common

# 添加 ROS 2 GPG 密钥
sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 添加仓库源（使用清华镜像加速）
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

## 3. 安装 ROS 2

```bash
sudo apt update

# 根据你的 Ubuntu 版本选择：
# Ubuntu 22.04 → ros-humble-desktop
# Ubuntu 24.04 → ros-jazzy-desktop

# 桌面完整版（推荐，包含 RViz2、Gazebo 集成等）
sudo apt install -y ros-jazzy-desktop

# 如果只需要基础功能（无 GUI）：
# sudo apt install -y ros-jazzy-ros-base
```

## 4. 环境配置

```bash
# 将 source 命令写入 bashrc，每次开终端自动加载
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 验证安装
ros2 --version
# 输出示例: ros2 cli version: 0.28.0
```

## 5. 安装 ROS 2 开发工具

```bash
# 编译构建工具
sudo apt install -y python3-colcon-common-extensions

# 常用 ROS 2 包
sudo apt install -y \
  ros-jazzy-tf2-tools \
  ros-jazzy-vision-msgs \
  ros-jazzy-geometric-msgs \
  ros-jazzy-sensor-msgs \
  ros-jazzy-cv-bridge \
  ros-jazzy-image-transport
```

## 6. 创建 ROS 2 工作空间（测试）

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# 创建测试功能包
cd src
ros2 pkg create --build-type ament_python my_test_package

# 编译
cd ~/ros2_ws
colcon build

# 加载环境
source install/setup.bash

# 运行测试节点
ros2 run my_test_package my_test_package
```

## 7. 常用命令速查

```bash
# 列出所有可用节点
ros2 node list

# 列出所有 topic
ros2 topic list

# 实时查看 topic 数据
ros2 topic echo /chatter

# 查看 topic 频率
ros2 topic hz /chatter

# 运行小海龟 demo
ros2 run turtlesim turtlesim_node
ros2 run turtlesim turtle_teleop_key
```

---

## 卸载 ROS 2

```bash
sudo apt remove -y ~nros-jazzy-*
sudo apt autoremove -y
```

## 常见问题

### GPG 密钥错误

```bash
# 如果 key 过期，重新获取
sudo rm /usr/share/keyrings/ros-archive-keyring.gpg
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
```

### 网络超时

```bash
# 换用中科大镜像
sudo sed -i 's/mirrors.tuna.tsinghua.edu.cn/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/ros2.list
```

## 参考资料

- [ROS 2 Jazzy 官方文档](https://docs.ros.org/en/jazzy/)
- [ROS 2 Humble 官方文档](https://docs.ros.org/en/humble/)
- [ROS 2 教程](https://docs.ros.org/en/jazzy/Tutorials.html)
