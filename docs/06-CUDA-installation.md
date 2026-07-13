# CUDA & cuDNN 安装指南

> NVIDIA CUDA Toolkit + cuDNN — GPU 并行计算基础

## 版本选择

| CUDA Toolkit | cuDNN | 驱动最低要求 |
|-------------|-------|------------|
| 12.8 | 9.x | 550.x |
| 12.6 | 9.x | 550.x |
| 12.4 | 9.x | 550.x |
| 12.1 | 8.x | 530.x |
| 11.8 | 8.x | 520.x |

**检查 GPU 与驱动支持的 CUDA 版本：**
```bash
nvidia-smi  # 查看右上角 "CUDA Version"
```

> `nvidia-smi` 显示的是驱动支持的最高 CUDA 版本，实际安装的 CUDA Toolkit 可以等于或低于此值。

---

## Linux 安装 (Ubuntu 22.04 / 24.04)

### Step 1: 安装 NVIDIA 驱动

```bash
# 检查是否已安装
nvidia-smi

# 如果未安装
sudo apt update
sudo apt install -y nvidia-driver-550
sudo reboot

# 验证
nvidia-smi
```

### Step 2: 安装 CUDA Toolkit

```bash
# 方法一：使用 NVIDIA 官方 runfile（推荐，可自定义路径）

# 下载 CUDA Toolkit（以 12.4 为例）
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# 方法二：使用 apt 安装（简单但路径固定）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

### Step 3: 配置环境变量

```bash
# 添加到 ~/.bashrc
echo '# CUDA' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
# 输出示例: nvcc: NVIDIA (R) Cuda compiler driver; Cuda compilation tools, release 12.4
```

### Step 4: 安装 cuDNN

```bash
# 下载 cuDNN（需要 NVIDIA 开发者账号）
# 访问: https://developer.nvidia.com/cudnn
# 下载对应 CUDA 版本的 cuDNN

# 或通过 apt 安装（Ubuntu 24.04）
sudo apt install -y cudnn-cuda-12
```

---

## Windows 安装

### Step 1: 安装 NVIDIA 驱动

1. 下载 [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx)
2. 运行安装程序，选择"自定义安装"

### Step 2: 安装 CUDA Toolkit

1. 下载 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. 选择 Windows → x86_64 → 11 → exe (local)
3. 运行安装程序（默认路径 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`）

### Step 3: 安装 cuDNN

1. 下载 [cuDNN](https://developer.nvidia.com/cudnn)（需要登录）
2. 解压并将文件复制到 CUDA 安装目录：
   ```
   cudnn-windows-x86_64-9.x.x\bin\*.dll   → %CUDA_PATH%\bin\
   cudnn-windows-x86_64-9.x.x\include\*.h   → %CUDA_PATH%\include\
   cudnn-windows-x86_64-9.x.x\lib\x64\*.lib → %CUDA_PATH%\lib\x64\
   ```

### Step 4: 验证

```powershell
# PowerShell
nvcc --version
nvidia-smi
```

---

## PyTorch GPU 验证

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# GPU 计算测试
x = torch.randn(10000, 10000).cuda()
y = torch.mm(x, x)
print(f"GPU matrix multiplication: {y.shape}")
```

---

## 多版本 CUDA 管理

```bash
# 如果需要同时安装多个 CUDA 版本
# 使用 update-alternatives 切换
sudo update-alternatives --config cuda

# 或手动切换环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## 常见问题

### `nvcc: command not found`

```bash
# 检查 CUDA 安装路径
ls /usr/local/cuda*

# 手动添加 PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

### PyTorch 无法识别 GPU

```bash
# 检查 CUDA 版本匹配
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # 驱动 CUDA 版本

# 如果版本不匹配，重装匹配版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 驱动与 CUDA 冲突

```bash
# 查看已安装的 NVIDIA 包
dpkg -l | grep nvidia

# 卸载旧版本
sudo apt remove --purge nvidia-*
sudo apt autoremove
sudo apt install nvidia-driver-550
```

## 参考资料

- [CUDA Toolkit 下载](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 下载](https://developer.nvidia.com/cudnn)
- [PyTorch 安装](https://pytorch.org/get-started/locally/)
