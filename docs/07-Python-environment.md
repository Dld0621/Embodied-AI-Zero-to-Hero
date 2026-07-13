# Python 环境配置指南

> Conda / venv / pip — Python 环境管理最佳实践

## 为什么需要虚拟环境？

机器人开发项目通常依赖不同版本的库（如不同项目的 PyTorch、MuJoCo、ROS 2 版本）。**虚拟环境**确保各项目互不干扰。

---

## 方案一：Miniconda（推荐）

### 安装 Miniconda

```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows: 下载 exe 安装器
# https://docs.anaconda.com/free/miniconda/

# 安装完成后重启终端
```

### 基础命令

```bash
# 创建环境
conda create -n my_robot python=3.12 -y

# 激活环境
conda activate my_robot

# 退出环境
conda deactivate

# 列出所有环境
conda env list

# 删除环境
conda env remove -n my_robot

# 导出/导入环境
conda env export > environment.yml
conda env create -f environment.yml
```

### 常用镜像源配置

```bash
# 清华镜像（推荐国内用户）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --set show_channel_urls yes
```

---

## 方案二：venv（轻量）

```bash
# Python 3.10+ 内置
python3 -m venv my_robot_env

# 激活
source my_robot_env/bin/activate  # Linux
# my_robot_env\Scripts\activate   # Windows

# 退出
deactivate
```

---

## pip 配置

### 镜像加速

```bash
# 永久设置（推荐）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 或临时使用
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 常用命令

```bash
# 升级 pip
pip install --upgrade pip

# 安装指定版本
pip install mujoco==3.1.6

# 从 requirements.txt 安装
pip install -r requirements.txt

# 导出依赖
pip freeze > requirements.txt

# 查看已安装包
pip list

# 查看某个包的信息
pip show mujoco
```

---

## 机器人开发常用 packages

```bash
# 创建机器人开发环境
conda create -n robot_dev python=3.12 -y
conda activate robot_dev

# 核心科学计算
pip install numpy scipy matplotlib

# 仿真
pip install mujoco gymnasium

# 深度学习（GPU）
pip install torch torchvision  # 或 pip install tensorflow

# 3D 处理
pip install open3d trimesh

# 强化学习
pip install stable-baselines3[extras]

# ROS 2 Python 客户端
pip install rclpy  # 通常随 ROS 2 安装

# 机器人学工具
pip install roboticstoolbox-python pinocchio

# 计算机视觉
pip install opencv-python-headless

# 数据处理
pip install h5py pandas

# Jupyter Notebook
pip install jupyterlab ipywidgets
```

---

## requirements.txt 模板

```txt
# robot_project requirements.txt
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
torch>=2.0
mujoco>=3.0
gymnasium>=0.29
opencv-python-headless>=4.8
stable-baselines3>=2.0
```

---

## 常见问题

### Conda 初始化失败

```bash
# 手动初始化
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### pip 权限错误

```bash
# 不要使用 sudo pip install！
# 使用 --user 或激活虚拟环境
pip install --user package_name
```

### SSL 证书错误

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name
```

## 参考资料

- [Conda 文档](https://docs.conda.io/)
- [pip 文档](https://pip.pypa.io/)
- [Python 虚拟环境教程](https://docs.python.org/3/library/venv.html)
