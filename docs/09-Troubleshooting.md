# 常见问题排查 (Troubleshooting)

> 机器人开发环境配置中的常见错误与解决方案

---

## ROS 2 相关

### `ros2 command not found`

```bash
# 确认已 source 环境
source /opt/ros/jazzy/setup.bash

# 如果刚安装，检查是否真的安装成功
dpkg -l | grep ros-jazzy

# 永久解决：写入 bashrc
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```

### `rosdep init` 失败

```bash
# 网络问题导致 rosdep 更新失败
sudo apt install -y python3-rosdep
sudo rosdep init
rosdep update --rosdistro jazzy

# 使用国内源
export ROSDEP_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/github-rosdistro/rosdep
```

### Gazebo 启动黑屏

```bash
# 检查 OpenGL 支持
glxinfo | grep "OpenGL version"

# 安装 Mesa 驱动
sudo apt install -y mesa-utils libgl1-mesa-glx
```

---

## MuJoCo 相关

### `ImportError: No module named 'mujoco'`

```bash
# 确认在正确的虚拟环境中
which python
pip list | grep mujoco

# 重新安装
pip uninstall mujoco -y
pip install mujoco
```

### `freeglut` 错误 (Linux)

```bash
sudo apt install -y freeglut3-dev
```

### 渲染问题（无显示器服务器）

```bash
# 使用 osmesa 渲染后端
export MUJOCO_GL=osmesa
sudo apt install -y libosmesa6-dev
```

---

## CUDA / GPU 相关

### `CUDA out of memory`

```python
# 减少 batch size
# 或检查是否有内存泄漏
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
```

### `CUDA version mismatch`

```bash
# PyTorch 的 CUDA 版本必须 <= 驱动支持的 CUDA 版本
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # 查看 "CUDA Version" 列

# 重装匹配版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 多 GPU 使用

```python
import torch
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
print(f"GPU 1: {torch.cuda.get_device_name(1)}")

# 指定使用哪块 GPU
device = torch.device("cuda:1")
tensor = torch.randn(100).to(device)
```

---

## Python 环境相关

### `pip install` 权限错误

```bash
# 错误: ERROR: Could not install packages due to an EnvironmentError
# 解决: 不要用 sudo！激活虚拟环境后再安装
conda activate my_env
pip install package_name
```

### 包版本冲突

```bash
# 查看冲突
pip check

# 强制重装
pip install --force-reinstall package_name

# 使用 conda 管理复杂依赖
conda install package_name
```

### SSL 证书错误

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name

# 或更新证书
pip install --upgrade certifi
```

---

## WSL2 相关

### WSL2 无法识别 GPU

```powershell
# 确认 Windows NVIDIA 驱动 >= 550.x
# 重启 WSL2
wsl --shutdown
# 重新进入
wsl
nvidia-smi
```

### WSL2 内存占用过高

```powershell
# 创建 %USERPROFILE%\.wslconfig
[wsl2]
memory=16GB
swap=8GB
localhostForwarding=true
```

---

## Isaac Lab 相关

### `ModuleNotFoundError: omni.kit.usd`

```bash
# Isaac Sim 未正确链接
# 确认环境变量
echo $ISAACSIM_PATH

# 重新安装
pip install --upgrade isaaclab --extra-index-url https://pypi.nvidia.com
```

### Omniverse Launcher 无法连接

```
1. 检查网络代理
2. 清除 Launcher 缓存
3. 重新安装 Launcher
```

---

## 环境检测清单

运行以下命令逐一排查：

```bash
echo "=== 系统信息 ==="
lsb_release -a
uname -m

echo "=== GPU ==="
nvidia-smi

echo "=== CUDA ==="
nvcc --version

echo "=== Python ==="
python --version
which python

echo "=== Conda ==="
conda --version
conda env list

echo "=== ROS 2 ==="
ros2 --version 2>/dev/null || echo "ROS 2 not found"

echo "=== MuJoCo ==="
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')" 2>/dev/null || echo "MuJoCo not found"

echo "=== PyTorch GPU ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not found"

echo "=== 磁盘空间 ==="
df -h / /home
```

或者直接使用本项目的环境检测脚本：

```bash
bash scripts/check_env.sh
```

---

## 获取更多帮助

- [ROS 2 Answers](https://answers.ros.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [MuJoCo Discussions](https://github.com/google-deepmind/mujoco/discussions)
- [Isaac Lab Issues](https://github.com/isaac-sim/IsaacLab/issues)
