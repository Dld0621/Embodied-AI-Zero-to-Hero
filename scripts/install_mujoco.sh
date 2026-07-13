#!/bin/bash
# =============================================================
# RobotDev-Setup-Guide: MuJoCo 自动安装脚本
# =============================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  MuJoCo 自动安装脚本${NC}"
echo -e "${BLUE}============================================================${NC}"

ENV_NAME="${1:-mujoco}"
echo -e "${YELLOW}Conda 环境名: $ENV_NAME${NC}"
echo ""

# Step 1: 创建 Conda 环境
echo -e "${BLUE}[1/4] 创建 Conda 环境...${NC}"
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}  环境 $ENV_NAME 已存在，跳过创建${NC}"
else
    conda create -n $ENV_NAME python=3.12 -y
fi

# Step 2: 激活环境并安装 MuJoCo
echo -e "${BLUE}[2/4] 安装 MuJoCo...${NC}"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
pip install --upgrade pip
pip install mujoco numpy imageio glfw

# Step 3: 安装常用依赖
echo -e "${BLUE}[3/4] 安装常用依赖...${NC}"
pip install gymnasium stable-baselines3

# Step 4: 验证
echo -e "${BLUE}[4/4] 验证安装...${NC}"
python3 -c "
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')

import gymnasium as gym
env = gym.make('Ant-v5', render_mode=None)
print('Gymnasium Ant-v5: OK')
env.close()
"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  MuJoCo 安装完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}使用方法:${NC}"
echo -e "  conda activate $ENV_NAME"
echo -e "  python -c 'import mujoco; print(mujoco.__version__)'"
echo ""
