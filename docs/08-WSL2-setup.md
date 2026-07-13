# Windows WSL2 配置指南

> Windows Subsystem for Linux 2 — 在 Windows 上运行 Ubuntu

## 为什么要用 WSL2？

Windows WSL2 让你可以在 Windows 上原生运行完整的 Linux 环境，同时访问 GPU（CUDA），非常适合机器人开发：

- 运行 ROS 2（仅支持 Linux）
- 使用 Linux 版 MuJoCo / Isaac Lab
- 无需双系统切换

---

## 安装 WSL2

### 方法一：一键安装（推荐）

```powershell
# PowerShell (管理员)
wsl --install

# 默认安装 Ubuntu 24.04
# 安装完成后重启电脑
```

### 方法二：指定发行版

```powershell
# 查看可用发行版
wsl --list --online

# 安装 Ubuntu 24.04
wsl --install -d Ubuntu-24.04

# 或 Ubuntu 22.04（对应 ROS 2 Humble）
wsl --install -d Ubuntu-22.04
```

---

## 首次配置

```bash
# 启动 Ubuntu
# 设置用户名和密码（不要和 Windows 密码相同也可以）

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential cmake git curl wget vim
```

---

## WSL2 + GPU (CUDA)

### 确认 CUDA 支持

```bash
# 在 WSL2 内检查
nvidia-smi

# 如果能输出 GPU 信息，说明 NVIDIA 驱动已透传到 WSL2
# （不需要在 WSL2 内单独安装 NVIDIA 驱动！）
```

### 安装 CUDA Toolkit（WSL2 内）

```bash
# 下载 CUDA Toolkit for WSL-Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# 配置环境变量
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

---

## WSL2 文件系统说明

```
Windows 文件系统:  /mnt/c/  /mnt/d/
WSL 文件系统:      /home/username/

建议:
- 代码和模型放在 WSL 文件系统内（性能更好）
- 大型数据集可以放在 /mnt/ 下共享
```

---

## VS Code 集成

### 安装 WSL 扩展

1. 在 VS Code 中安装 "WSL" 扩展
2. 在 WSL2 终端中输入：

```bash
code .  # 用 VS Code 打开当前目录
```

### 推荐插件

- **Remote - WSL** — 远程连接 WSL
- **Python** — Python 语言支持
- **ROS** — ROS 2 开发支持
- **CMake Tools** — CMake 项目支持

---

## 常用操作

```powershell
# PowerShell 中管理 WSL

# 查看运行中的发行版
wsl --list --verbose

# 关闭 WSL
wsl --shutdown

# 进入指定发行版
wsl -d Ubuntu-24.04

# 设置默认发行版
wsl --set-default Ubuntu-24.04

# 导出/导入 WSL 镜像
wsl --export Ubuntu-24.04 D:\wsl-backup\ubuntu.tar
wsl --import Ubuntu-24.04 D:\wsl-backup D:\wsl-backup\ubuntu.tar
```

---

## 常见问题

### `nvidia-smi` 在 WSL2 内无法运行

```
解决方案:
1. 确认 Windows 上已安装 NVIDIA 驱动 (>= 550.x)
2. 更新到最新驱动
3. 重启 WSL: wsl --shutdown
```

### GUI 应用无法显示

```bash
# 安装 WSLg 支持的 Ubuntu 24.04 自带 GUI 支持
# 如果不支持，手动安装：
sudo apt install -y x11-apps
export DISPLAY=:0
```

### 内存占用过高

```powershell
# 在 Windows 用户目录创建 .wslconfig
notepad "$env:USERPROFILE\.wslconfig"

# 添加以下内容
[wsl2]
memory=16GB
processors=8
swap=8GB
```

---

## WSL2 vs 双系统 vs 虚拟机

| 特性 | WSL2 | 双系统 | 虚拟机 (VM) |
|------|------|--------|------------|
| GPU 直通 | 原生支持 | 原生 | 需配置 |
| ROS 2 支持 | 完整 | 完整 | 完整 |
| 文件互访 | 无缝 | 需挂载 | 需共享 |
| 性能 | ~95% 原生 | 100% 原生 | ~80% |
| 开机时间 | 秒级 | 需重启 | 分钟级 |

## 参考资料

- [WSL2 官方文档](https://docs.microsoft.com/windows/wsl/)
- [WSL2 CUDA 支持](https://docs.nvidia.com/cuda/wsl-user-guide/)
