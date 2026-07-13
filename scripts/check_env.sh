#!/bin/bash
# =============================================================
# RobotDev-Setup-Guide: 环境检测脚本
# 用途：检测当前系统已安装的机器人开发工具及其版本
# =============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

passed=0
warned=0
failed=0

check_pass() {
    echo -e "  ${GREEN}[OK]${NC} $1"
    ((passed++))
}

check_warn() {
    echo -e "  ${YELLOW}[WARN]${NC} $1"
    ((warned++))
}

check_fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
    ((failed++))
}

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  RobotDev 环境检测报告${NC}"
echo -e "${BLUE}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# --- 系统信息 ---
echo -e "${BLUE}[1] 系统信息${NC}"
if command -v lsb_release &> /dev/null; then
    distro=$(lsb_release -ds 2>/dev/null || echo "Unknown")
    codename=$(lsb_release -cs 2>/dev/null || echo "Unknown")
    echo -e "  发行版: $distro ($codename)"
else
    echo -e "  发行版: 非 Linux 系统"
fi
echo -e "  架构: $(uname -m)"
echo -e "  内核: $(uname -r)"
echo -e "  内存: $(free -h | awk '/Mem:/ {print $2}')"
echo -e "  磁盘: $(df -h / | awk 'NR==2 {print $4}') 可用"
echo ""

# --- GPU ---
echo -e "${BLUE}[2] GPU & CUDA${NC}"
if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    cuda_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    check_pass "GPU: $gpu_name"
    check_pass "驱动版本: $driver_ver"
    check_pass "CUDA 计算能力: $cuda_cap"
else
    check_fail "未检测到 NVIDIA GPU / nvidia-smi 不可用"
fi

if command -v nvcc &> /dev/null; then
    cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    check_pass "CUDA Toolkit: $cuda_ver"
else
    check_warn "CUDA Toolkit 未安装 (nvcc not found)"
fi

if command -v ldconfig &> /dev/null && ldconfig -p 2>/dev/null | grep -q libcudnn; then
    cudnn_ver=$(ldconfig -p 2>/dev/null | grep libcudnn | grep -oP '\d+\.\d+\.\d+' | head -1)
    check_pass "cuDNN: $cudnn_ver"
else
    check_warn "cuDNN 未检测到"
fi
echo ""

# --- Python ---
echo -e "${BLUE}[3] Python 环境${NC}"
if command -v python3 &> /dev/null; then
    py_ver=$(python3 --version 2>&1)
    check_pass "Python: $py_ver"
else
    check_fail "Python3 未安装"
fi

if command -v conda &> /dev/null; then
    conda_ver=$(conda --version 2>&1)
    check_pass "Conda: $conda_ver"
    echo -e "  环境: $(conda env list 2>/dev/null | grep -v '^#' | grep -v '^$' | tr '\n' ' ')"
else
    check_warn "Conda 未安装"
fi

if command -v pip3 &> /dev/null; then
    pip_ver=$(pip3 --version 2>&1 | awk '{print $1, $2}')
    check_pass "pip: $pip_ver"
fi
echo ""

# --- ROS 2 ---
echo -e "${BLUE}[4] ROS 2${NC}"
ros2_found=false
for distro in jazzy humble iron; do
    if [ -f "/opt/ros/$distro/setup.bash" ]; then
        ros2_found=true
        source /opt/ros/$distro/setup.bash 2>/dev/null
        ros2_ver=$(ros2 --version 2>/dev/null || echo "unknown")
        check_pass "ROS 2 $distro: $ros2_ver"

        # 检查常用包
        pkg_count=$(dpkg -l 2>/dev/null | grep "ros-$distro-" | wc -l)
        echo -e "  已安装 $pkg_count 个 ros-$distro 包"
        break
    fi
done
if [ "$ros2_found" = false ]; then
    check_fail "ROS 2 未安装"
fi
echo ""

# --- 仿真软件 ---
echo -e "${BLUE}[5] 仿真软件${NC}"

# MuJoCo
if python3 -c "import mujoco" &> /dev/null; then
    mj_ver=$(python3 -c "import mujoco; print(mujoco.__version__)" 2>/dev/null)
    check_pass "MuJoCo: $mj_ver"
else
    check_warn "MuJoCo 未安装 (python -c 'import mujoco' failed)"
fi

# Gazebo
if command -v gz &> /dev/null; then
    gz_ver=$(gz sim --versions 2>/dev/null | head -1 || gz --version 2>/dev/null)
    check_pass "Gazebo: $gz_ver"
else
    check_warn "Gazebo 未安装"
fi

# Isaac Sim / Lab
if [ -n "$ISAACSIM_PATH" ] && [ -d "$ISAACSIM_PATH" ]; then
    check_pass "Isaac Sim: $ISAACSIM_PATH"
elif python3 -c "import omni.isaac.lab" &> /dev/null; then
    check_pass "Isaac Lab: 已安装"
else
    check_warn "Isaac Sim / Lab 未安装"
fi

# PyBullet
if python3 -c "import pybullet" &> /dev/null; then
    check_pass "PyBullet: $(python3 -c 'import pybullet; print(pybullet.getNumpyVersion())' 2>/dev/null)"
else
    check_warn "PyBullet 未安装"
fi
echo ""

# --- 深度学习框架 ---
echo -e "${BLUE}[6] 深度学习框架${NC}"

if python3 -c "import torch" &> /dev/null; then
    torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    if python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        cuda_ver=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        check_pass "PyTorch: $torch_ver (CUDA $cuda_ver, GPU: $gpu_name)"
    else
        check_warn "PyTorch: $torch_ver (CPU only, GPU 不可用)"
    fi
else
    check_warn "PyTorch 未安装"
fi

if python3 -c "import tensorflow" &> /dev/null; then
    tf_ver=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
    check_pass "TensorFlow: $tf_ver"
else
    check_warn "TensorFlow 未安装"
fi

if python3 -c "import jax" &> /dev/null; then
    jax_ver=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null)
    check_pass "JAX: $jax_ver"
else
    check_warn "JAX 未安装"
fi
echo ""

# --- 开发工具 ---
echo -e "${BLUE}[7] 开发工具${NC}"

if command -v git &> /dev/null; then
    check_pass "Git: $(git --version | awk '{print $3}')"
else
    check_fail "Git 未安装"
fi

if command -v docker &> /dev/null; then
    check_pass "Docker: $(docker --version | awk '{print $3}')"
else
    check_warn "Docker 未安装"
fi

if command -v code &> /dev/null; then
    check_pass "VS Code: $(code --version | head -1)"
else
    check_warn "VS Code 未安装"
fi
echo ""

# --- 汇总 ---
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  检测结果汇总${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "  ${GREEN}通过: $passed${NC}  |  ${YELLOW}警告: $warned${NC}  |  ${RED}失败: $failed${NC}"
echo ""

if [ $failed -gt 0 ]; then
    echo -e "  ${YELLOW}建议: 查看对应安装指南解决失败项${NC}"
    echo -e "  ${YELLOW}https://github.com/Dld0621/RobotDev-Setup-Guide${NC}"
fi

exit 0
