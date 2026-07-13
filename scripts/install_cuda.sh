#!/bin/bash
# =============================================================
# RobotDev-Setup-Guide: CUDA 安装脚本 (Ubuntu)
# =============================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

CUDA_VERSION="${1:-12.4}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  CUDA $CUDA_VERSION 安装脚本 (Ubuntu)${NC}"
echo -e "${BLUE}============================================================${NC}"

# 前置检查
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}未检测到 NVIDIA 驱动！请先安装 NVIDIA 驱动。${NC}"
    exit 1
fi

driver_cuda=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
echo -e "${GREEN}NVIDIA 驱动支持的 CUDA 版本: $driver_cuda${NC}"

# 检测 Ubuntu 版本
if [ -f /etc/os-release ]; then
    . /etc/os-release
else
    echo -e "${RED}无法检测系统版本${NC}"
    exit 1
fi

# Step 1: 安装 CUDA Toolkit
echo -e "${BLUE}[1/3] 安装 CUDA Toolkit $CUDA_VERSION...${NC}"
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MAJOR_MINOR=$(echo $CUDA_VERSION | tr -d '.')

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt update -qq
sudo apt install -y cuda-toolkit-${CUDA_MAJOR}-$(echo $CUDA_VERSION | tr -d '.')
rm -f /tmp/cuda-keyring.deb

# Step 2: 配置环境变量
echo -e "${BLUE}[2/3] 配置环境变量...${NC}"
CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
if [ ! -d "$CUDA_PATH" ]; then
    CUDA_PATH="/usr/local/cuda"
fi

if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA" >> ~/.bashrc
    echo "export CUDA_HOME=$CUDA_PATH" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Step 3: 验证
echo -e "${BLUE}[3/3] 验证安装...${NC}"
nvcc --version

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  CUDA $CUDA_VERSION 安装完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}下一步: 安装 PyTorch GPU 版本${NC}"
echo -e "  pip install torch --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d '.')"
echo ""
