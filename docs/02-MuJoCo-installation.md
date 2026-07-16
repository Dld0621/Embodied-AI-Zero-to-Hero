# MuJoCo 安装指南

> Multi-Joint dynamics with Contact | 物理仿真引擎

## 简介

MuJoCo (Multi-Joint dynamics with Contact) 是 DeepMind 开源的高效物理仿真引擎，广泛用于机器人学习、强化学习和运动控制研究。

---

## Linux 安装

### 方法一：pip 安装（推荐）

```bash
# 创建虚拟环境（推荐）
conda create -n mujoco python=3.12 -y
conda activate mujoco

# 安装 MuJoCo
pip install mujoco

# 安装常用依赖
pip install numpy imageio glfw

# 验证安装
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
```

### 方法二：conda 安装

```bash
conda create -n mujoco python=3.12 -y
conda activate mujoco
conda install -c conda-forge mujoco
```

### 方法三：从源码编译

```bash
# 安装编译依赖
sudo apt install -y cmake libgl-dev libosmesa6-dev

# 克隆源码
git clone https://github.com/google-deepmind/mujoco.git
cd mujoco
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
make -j$(nproc)
make install
```

---

## Windows 安装

### 方法一：pip 安装（推荐）

```powershell
# 创建虚拟环境
conda create -n mujoco python=3.12 -y
conda activate mujoco

# 安装 MuJoCo
pip install mujoco
pip install numpy imageio glfw

# 验证
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
```

### 方法二：手动安装（旧版 mujoco-py）

```powershell
# 如果需要 mujoco-py（旧版兼容）
pip install mujoco-py

# 设置环境变量
setx MUJOCO_PY_MUJOCO_PATH "%USERPROFILE%\.mujoco"
```

---

## 运行示例

```python
import mujoco
import mujoco.viewer

# 加载示例模型
model = mujoco.MjModel.from_xml_path('''<mujoco model="example">
  <worldbody>
    <light diffuse="0.5 0.5 0.5" pos="0 0 3"/>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="capsule" fromto="0 0 -0.5 0 0 0.5" size="0.1" rgba="1 0 0 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>''')

data = mujoco.MjData(model)

# 启动可视化
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

---

## 常用 ML 框架集成

### 与 PyTorch 集成

```bash
pip install torch torchvision
# GPU 版本参考 https://pytorch.org/get-started/locally/
```

### 与 Gymnasium 集成

```bash
pip install gymnasium
# MuJoCo 已内置于 Gymnasium 的 mujoco 环境中
python -c "import gymnasium as gym; env = gym.make('Ant-v5'); print('OK')"
```

### 与强化学习库集成

```bash
# Stable-Baselines3
pip install stable-baselines3[extras]

# 强化学习训练示例
python -c "
import gymnasium as gym
from stable_baselines3 import PPO
env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=10000)
print('Training done!')
"
```

---

## MuJoCo Menagerie（预置机器人模型）

```bash
# Google DeepMind 维护的机器人 URDF/MJCF 模型库
git clone https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie

# 浏览可用模型
ls
# franka_fr3/  iiwa/  ur5e/  allegro_hand/  shadow_hand/  ...

# 加载模型测试
python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('franka_fr3/franka_fr3.xml')
print(f'DOF: {model.nv}')
"
```

---

## 常见问题

### OpenGL 渲染错误

```bash
# Linux: 安装 mesa 驱动
sudo apt install -y libgl1-mesa-glx libosmesa6

# 无头服务器（无显示器）
export MUJOCO_GL=osmesa
```

### macOS 兼容

```bash
# macOS 使用 EGL 渲染
export MUJOCO_GL=egl
pip install mujoco
```

### 性能优化

```python
import mujoco
# 启用 GPU 加速（需要 CUDA）
mujoco.glfw.glfw.init()
```

## 参考资料

- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)
- [MuJoCo GitHub](https://github.com/google-deepmind/mujoco)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
