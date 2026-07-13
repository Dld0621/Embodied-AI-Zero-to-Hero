#!/bin/bash
# =============================================================
# RobotDev-Setup-Guide: ROS 2 自动安装脚本
# 自动检测 Ubuntu 版本并安装对应的 ROS 2
# =============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  ROS 2 自动安装脚本${NC}"
echo -e "${BLUE}============================================================${NC}"

# 检测 Ubuntu 版本
if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_CODENAME=$VERSION_CODENAME
else
    echo -e "${RED}无法检测系统版本${NC}"
    exit 1
fi

echo -e "${YELLOW}检测到 Ubuntu: $UBUNTU_CODENAME${NC}"

case "$UBUNTU_CODENAME" in
    jammy)
        ROS_DISTRO="humble"
        ROS_VERSION="ROS 2 Humble Hawksbill (LTS)"
        ;;
    noble)
        ROS_DISTRO="jazzy"
        ROS_VERSION="ROS 2 Jazzy Jalisco (LTS)"
        ;;
    *)
        echo -e "${RED}不支持的 Ubuntu 版本: $UBUNTU_CODENAME${NC}"
        echo -e "${YELLOW}ROS 2 支持的版本: Ubuntu 22.04 (jammy) 或 24.04 (noble)${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}将安装: $ROS_VERSION${NC}"
echo ""

# Step 1: 设置 locale
echo -e "${BLUE}[1/6] 设置 locale...${NC}"
sudo apt update -qq
sudo apt install -y locales software-properties-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Step 2: 添加 ROS 2 仓库
echo -e "${BLUE}[2/6] 添加 ROS 2 apt 仓库...${NC}"
sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 使用清华镜像
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu $UBUNTU_CODENAME main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Step 3: 安装 ROS 2
echo -e "${BLUE}[3/6] 安装 ROS 2 $ROS_DISTRO (desktop)...${NC}"
sudo apt update -qq
sudo apt install -y ros-$ROS_DISTRO-desktop

# Step 4: 配置环境
echo -e "${BLUE}[4/6] 配置环境变量...${NC}"
if ! grep -q "ros-$ROS_DISTRO/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
fi
source /opt/ros/$ROS_DISTRO/setup.bash

# Step 5: 安装开发工具
echo -e "${BLUE}[5/6] 安装开发工具...${NC}"
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-pip \
    ros-$ROS_DISTRO-tf2-tools \
    ros-$ROS_DISTRO-vision-msgs \
    ros-$ROS_DISTRO-sensor-msgs \
    ros-$ROS_DISTRO-cv-bridge

# Step 6: 安装 Gazebo
echo -e "${BLUE}[6/6] 安装 Gazebo...${NC}"
if [ "$ROS_DISTRO" = "jazzy" ]; then
    sudo apt install -y gz-harmonic ros-$ROS_DISTRO-gazebo-ros-pkg
else
    sudo apt install -y gz-fortress ros-$ROS_DISTRO-gazebo-ros-pkg
fi

# 验证
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}  安装完成！${NC}"
echo -e "${BLUE}============================================================${NC}"
ros2 --version
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo -e "  source ~/.bashrc"
echo -e "  ros2 run turtlesim turtlesim_node"
echo ""
