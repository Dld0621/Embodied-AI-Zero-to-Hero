# 优质开源项目：从复现到实战

> 精选可直接复现的开源项目，按“视觉捕捉 → Retargeting → 仿真/真机部署”链路组织。每个项目标注**复现难度**、**核心学习点**和**与 Retargeting 的关联**。

---

## 目录

| 项目 | 方向 | 难度 | 关键标签 |
|------|------|------|---------|
| [AnyTeleop](#anyteleop) | 遥操作框架 | ⭐⭐⭐ | Vision-based teleop, Dexterous manipulation |
| [HaMeR](#hamer) | 手部 mesh 恢复 | ⭐⭐ | Hand mesh, MANO, RGB-only |
| [LEAP Hand](#leap-hand) | 低成本灵巧手 | ⭐⭐ | 16-DOF, URDF, MuJoCo |
| [DexPilot](#dexpilot) | 视觉遥操作 | ⭐⭐⭐⭐ | Vision-based, 无穿戴设备 |
| [RDT-1B](#rdt-1b) | 双臂扩散策略 | ⭐⭐⭐ | Diffusion Transformer, Bimanual |
| [DexGraspNet](#dexgraspnet) | 灵巧抓取数据集 | ⭐⭐ | Grasp generation, Shadow Hand |
| [OpenTeleVision](#opentelevision) | VR 遥操作 | ⭐⭐ | Meta Quest, Stereoscopic vision |
| [MyoSuite / MyoHand](#myosuite) | 肌骨仿真+RL | ⭐⭐⭐ | RL, Hand biomechanics |

---

## AnyTeleop

| 属性 | 信息 |
|------|------|
| **论文** | AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System (RSS 2023) |
| **arXiv** | [2307.04577](https://arxiv.org/abs/2307.04577) |
| **项目主页** | [https://yzqin.github.io/anyteleop/](https://yzqin.github.io/anyteleop/) |
| **机构** | UC San Diego, NVIDIA |
| **链接** | [https://github.com/dexsuite/dex-teleop](https://github.com/dexsuite/dex-teleop) |
| **难度** | ⭐⭐⭐ |
| **硬件需求** | RGB / RGB-D 摄像头（单目/双目），NVIDIA GPU |

> **重要提示**：截至 2026 年 7 月，`dex-teleop` 仓库仍为**私有（Private）**，无法直接 clone。以下内容基于论文原文、补充材料和 FrankMocap 公开仓库整理，可作为论文复现的技术参考。建议关注 [dexsuite 组织](https://github.com/dexsuite) 或联系作者 Yuzhe Qin 获取访问权限。

### 为什么值得复现

AnyTeleop 是**无穿戴视觉遥操作**的代表性工作。它用单个 RGB 摄像头捕捉人手 21 点，通过 IK Retargeting 控制 Shadow Hand + Franka 机械臂。复现它能让你完整走通：

```
MediaPipe 21点 → 局部坐标系 → 手掌姿态估计 → 手指 IK → Shadow Hand 控制
```

### 全部模型文件

AnyTeleop 的核心优势：**Retargeting 和碰撞检测均无学习模型**，仅依赖 URDF 运动学 + 数值优化。但感知模块需要以下预训练模型：

#### 1. MediaPipe Hand Landmark 模型（自动下载）

| 属性 | 信息 |
|------|------|
| **用途** | 检测手腕坐标系中 21 个手指关节 3D 关键点 |
| **包含** | BlazePalm 检测器 + HandLandmark 模型 |
| **下载方式** | `pip install mediapipe` 后自动下载，无需手动操作 |
| **文件大小** | ~5-10MB（mediapipe pip 包自动管理） |
| **运行位置** | CPU 实时运行 |

#### 2. FrankMocap 手部模型（纯 RGB 模式的腕部姿态估计）

| 文件路径 | 文件名 | 用途 |
|---------|--------|------|
| `./extra_data/hand_module/` | `mean_mano_params.pkl` | MANO 均值参数（默认形状） |
| `./extra_data/hand_module/` | `SMPLX_HAND_INFO.pkl` | SMPL-X 手部模型元数据 |
| `./extra_data/hand_module/pretrained_weights/` | `pose_shape_best.pth` | 手部姿态回归网络权重（~200-300MB） |
| `./extra_data/hand_module/hand_detector/` | `faster_rcnn_1_8_132028.pth` | Faster R-CNN 手部检测 backbone（~200-400MB） |
| `./extra_data/hand_module/hand_detector/` | `model_0529999.pth` | Faster R-CNN 手部检测模型（~200-400MB） |
| `./extra_data/smpl/` | `SMPLX_NEUTRAL.pkl` | SMPL-X 中性体模型（~50MB，**需手动下载**） |

**下载方式**：
```bash
# 自动下载除 SMPLX_NEUTRAL.pkl 外的所有文件
git clone https://github.com/facebookresearch/frankmocap.git
cd frankmocap
bash scripts/install_frankmocap.sh

# SMPLX_NEUTRAL.pkl 需手动注册下载
# 注册地址：https://smpl-x.is.tue.mpg.de/
# 下载后放置到：./extra_data/smpl/SMPLX_NEUTRAL.pkl
```

#### 3. 机器人 URDF 模型文件（无需训练，仅需运动学描述）

| 模型 | 来源 | 说明 |
|------|------|------|
| Shadow Hand | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/master/shadow_hand) | 20-DOF 类人灵巧手 |
| Allegro Hand | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/master/allegro_hand) | 16-DOF 四指灵巧手 |
| Franka Panda | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/master/franka_fr3) | 7-DOF 机械臂 |

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| **操作系统** | Ubuntu 20.04 LTS | 论文实验基于 Linux |
| **GPU** | NVIDIA GPU（CUDA 兼容） | 碰撞检测使用 CUDA 加速几何查询 |
| **CUDA** | 11.x | FrankMocap 要求 10.1+，IsaacGym 推荐 11.x |
| **Python** | 3.7 - 3.8 | FrankMocap 推荐 3.7 |
| **PyTorch** | 1.6.0+ | FrankMocap 依赖 |
| **IsaacGym** | Preview 4 | GPU 并行仿真环境 |
| **SAPIEN** | v2.x | 可选仿真环境 |
| **MuJoCo** | 2.x | 部分任务使用 |
| **Detectron2** | 兼容 PyTorch 1.6+ | FrankMocap 手部检测器 |
| **MediaPipe** | 最新版 | pip 安装，自动管理模型 |

### 每一步流程详解

```
摄像头 RGB/RGB-D 图像流
    ↓
[Step 1] MediaPipe 手指关键点检测 (CPU, 实时)
    → 21 个关键点的 3D 坐标（腕部坐标系）+ 2D 像素坐标
    → 模型：MediaPipe 内置 hand_landmark（自动下载）
    ↓
[Step 2] 腕部姿态估计
    ├─ RGB-D 模式: 像素坐标 + 深度图 + 相机内参 → 3D 点 → PnP → 6D 手腕姿态
    └─ 纯 RGB 模式: FrankMocap 弱透视比例网络 → 近似 3D 手腕位置
    → 模型：FrankMocap pose_shape_best.pth + faster_rcnn_*.pth
    ↓
[Step 3] 检测融合（多摄像头时）
    ├─ 前 50 帧自动标定：计算相机间相对旋转 R ∈ SO(3)
    ├─ SMPL-X 手形参数置信度评分
    └─ 选择最高置信度摄像头的相对运动
    ↓
[Step 4] 手部姿态重定向 (IK 优化) ← 核心步骤
    → 优化目标函数（见下方）
    → 方法：scipy.optimize.minimize (L-BFGS-B)
    → 输入：人手 21 关键点 + 机器人 URDF
    → 输出：机器人手关节角度
    ↓
[Step 5] 运动生成 (手臂控制 + 碰撞避免)
    ├─ 手腕位姿 → 机械臂末端轨迹
    ├─ CUDA 加速碰撞检测
    └─ 关节限位约束（来自 URDF）
    ↓
关节控制命令 → 网络 → 客户端 → 机器人/模拟器
```

#### Step 4 详细：IK 优化目标函数

AnyTeleop 将 retargeting 建模为带约束的非线性最小二乘问题：

$$\min_{q_t} \sum_i \|\text{FK}_{\text{robot}}(q_t, i) - s \cdot v^{\text{human}}_{\text{keypoint},i}\|^2 + w_s \|q_t - q_{t-1}\|^2$$

$$\text{s.t.} \quad q^{\text{lower}} \leq q_t \leq q^{\text{upper}}$$

其中：
- $q_t$：时间步 $t$ 时机器人手的关节角度（**求解目标**）
- $\text{FK}_{\text{robot}}(q_t, i)$：机器人 FK，输入关节角 → 输出第 $i$ 个关键点 3D 位置
- $v^{\text{human}}_{\text{keypoint},i}$：人手第 $i$ 个关键点向量（来自 MediaPipe 21 点）
- $s$：**缩放因子**，补偿人手与机器人手尺寸差异
- $w_s$：**时间平滑权重**，惩罚相邻帧关节角跳变（本项目对应 EMA 平滑）
- $q^{\text{lower}}, q^{\text{upper}}$：URDF 定义的关节角度下限和上限（本项目对应 joint clamping [0, 1.2]）

**求解器**：`scipy.optimize.minimize(method='L-BFGS-B')` — 带边界约束的拟牛顿法

**与本项目对比**：
- AnyTeleop 用 **L-BFGS-B**（拟牛顿法），本项目用 **scipy.least_squares(method='lm')**（Levenberg-Marquardt）
- 两者都是任务空间 IK，核心思想一致：最小化人手关键点与机器人关键点的位置误差
- AnyTeleop 额外加了时间平滑项 $w_s \|q_t - q_{t-1}\|^2$，本项目用 PostProcessor 的 EMA 单独处理

### 关键源文件角色（基于论文架构推断）

由于仓库私有，以下基于论文模块描述给出预期文件结构：

| 模块 | 预期文件/功能 | 对应本项目 |
|------|-------------|-----------|
| **手指关键点检测** | 调用 MediaPipe Hands，输出 21 个 3D 关键点 | `tutorials/04-landmark-pipeline/` |
| **腕部姿态估计** | RGB-D: PnP 算法 / 纯 RGB: FrankMocap 弱透视 | `docs/03-human-hand-to-robot-hand.md` 手掌姿态估计 |
| **检测融合** | 多摄像头自动标定 + SMPL-X 置信度评分 | 本项目暂无（单摄像头场景） |
| **FK 模块** | 以关节角为输入，计算机器人关键点 3D 位置 | `examples/finger_chain_3d.py` |
| **IK 优化** | scipy L-BFGS-B，目标函数如上 | `examples/complete_retargeting_pipeline.py` VectorOptimizationRetargeter |
| **碰撞检测** | CUDA 加速几何查询（trimesh/cupy） | 本项目暂无 |
| **运动生成** | 手腕→机械臂 IK + 轨迹平滑 | 本项目对应 X1 机械臂 DLS IK |
| **Web 可视化** | meshcat 服务器 + Three.js 前端 | 本项目暂无 |

### 论文性能基准

- **整体运行频率**：~93 Hz（含检测 + 融合 + Retargeting + 碰撞检测）
- **支持机器人**：Shadow Hand, Allegro Hand, DexHand, DClaw（三指爪）
- **支持机械臂**：Franka Panda, xArm6

### 与 Retargeting 的关系

- 核心思路与本项目 **Stage 3 Vector Optimization** 完全一致：任务空间 IK 优化
- 时间平滑项 $w_s \|q_t - q_{t-1}\|^2$ 对应本项目的 **EMA 时域滤波**
- 缩放因子 $s$ 对应本项目的**尺度归一化**（手腕到中指 MCP 的距离）
- 碰撞检测是本项目暂未覆盖的进阶方向

---

## HaMeR

| 属性 | 信息 |
|------|------|
| **论文** | Reconstructing Hands in 3D with Transformers (CVPR 2024) |
| **机构** | MPI for Intelligent Systems |
| **链接** | [https://github.com/geopavlakos/hamer](https://github.com/geopavlakos/hamer) |
| **难度** | ⭐⭐ |
| **硬件需求** | NVIDIA GPU (>= 8GB VRAM) |

### 为什么值得复现

HaMeR 解决了 MediaPipe 的痛点：**深度估计不准、自遮挡严重**。它从单张 RGB 直接回归 MANO 参数（778 个顶点 + 16 个关节角），输出的是**完整的 3D 手模型**，而非稀疏 21 点。

### 全部模型文件

#### 1. HaMeR 主模型权重包

| 属性 | 信息 |
|------|------|
| **文件** | `hamer_demo_data.tar.gz` |
| **Google Drive** | `https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT` |
| **直接下载 (wget)** | `https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz` |
| **解压至** | 项目根目录（运行 `bash fetch_demo_data.sh` 自动解压到 `_DATA/`） |

解压后包含以下文件：

| 文件路径 | 用途 |
|---------|------|
| `_DATA/hamer_ckpts/checkpoints/hamer.ckpt` | HaMeR 主模型 checkpoint（PyTorch Lightning 格式） |
| `_DATA/hamer_ckpts/model_config.yaml` | 模型配置文件（MANO 路径、backbone 类型、head 参数） |
| `_DATA/data/mano_mean_params.npz` | MANO 均值参数（pose、shape、cam 的先验均值） |
| `_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth` | ViTPose-H 全身关键点检测权重 |

#### 2. 人体检测器权重（ViTDet，自动下载）

| 属性 | 信息 |
|------|------|
| **URL** | `https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl` |
| **说明** | Detectron2 的 ViTDet-H 模型，检测人体 bounding box |
| **下载方式** | 首次运行 `demo.py` 时由 Detectron2 自动下载 |

#### 3. MANO 手部模型文件（需手动注册下载）

| 属性 | 信息 |
|------|------|
| **下载地址** | https://mano.is.tue.mpg.de/ |
| **所需文件** | 仅需 `MANO_RIGHT.pkl`（右手模型） |
| **放置路径** | `_DATA/data/mano/MANO_RIGHT.pkl` |
| **版本** | v1.2（2019-01-16 发布） |
| **许可证** | **仅限非商业科学研究用途**。禁止商业、色情、军事、监控用途；不可再分发；引用需标注 Romero et al., SIGGRAPH Asia 2017 |

> **关于左右手**：HaMeR **只需要右手模型文件**。左手通过 x 轴翻转处理（`verts[:,0] = (2*is_right-1)*verts[:,0]`），不依赖左手 pkl。

#### 4. 训练用预训练 backbone（仅训练需要）

| 文件 | 用途 |
|------|------|
| `hamer_training_data/vitpose_backbone.pth` | ViTPose-H backbone 权重，用于初始化 HaMeR 的 ViT |
| **下载方式** | `bash fetch_training_data.sh` |

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.10 | README 明确指定 |
| **PyTorch** | 2.2.0 | Dockerfile 指定 |
| **torchvision** | 0.17.0 | Dockerfile 指定 |
| **CUDA** | 11.7 或 11.8 | README 推荐 11.7；Dockerfile 用 11.8 |
| **GPU** | NVIDIA, >= 8GB VRAM | ViT-Huge + Detectron2 需要较大显存 |
| **OS** | Linux (Ubuntu 22.04) | PyRender EGL 后端依赖 Linux |
| **smplx** | 0.1.28（固定版本） | `setup.py` 中硬编码，不可改 |
| **mmcv** | 1.3.9（固定版本） | ViTPose 依赖，不可改 |
| **detectron2** | main 分支最新 | 从 GitHub 安装 |
| **pyrender** | 最新版 | 离屏渲染 |

### 关键源文件及其角色

```
hamer/
├── demo.py                          # [入口] 演示推理脚本
├── eval.py                          # [入口] 评估脚本
├── train.py                         # [入口] 训练脚本
├── vitpose_model.py                 # [手部关键点] ViTPose 全身关键点检测封装
├── fetch_demo_data.sh               # 模型下载脚本
├── setup.py                         # 安装配置（固定 smplx==0.1.28, mmcv==1.3.9）
├── hamer/
│   ├── configs/
│   │   ├── __init__.py              # [配置] yacs 默认配置，定义 CACHE_DIR_HAMER = "./_DATA"
│   │   ├── cascade_mask_rcnn_vitdet_h_75ep.py  # [检测] ViTDet 配置
│   │   ├── datasets_eval.yaml       # [数据] 评估数据集路径
│   │   └── datasets_tar.yaml        # [数据] 训练数据集配置
│   ├── configs_hydra/
│   │   ├── experiment/
│   │   │   ├── default.yaml         # [配置] MANO 路径: _DATA/data/mano/MANO_RIGHT.pkl
│   │   │   └── hamer_vit_transformer.yaml  # [配置] ViT-H + Transformer Decoder 完整参数
│   │   └── train.yaml               # [配置] Hydra 主训练配置
│   ├── models/
│   │   ├── __init__.py              # [模型加载] download_models(), load_hamer()
│   │   ├── hamer.py                 # [核心模型] HAMER 类（PyTorch Lightning Module）
│   │   ├── mano_wrapper.py          # [MANO 封装] 扩展 smplx.MANOLayer，含 joint_map
│   │   ├── backbones/
│   │   │   └── vit.py               # [图像编码] ViT-H backbone（embed_dim=1280, depth=32）
│   │   └── heads/
│   │       └── mano_head.py         # [参数回归] Transformer Decoder（6层, 8头, mlp_dim=1024）
│   ├── datasets/
│   │   └── vitdet_dataset.py        # [数据预处理] 图像裁剪、归一化、左右手翻转
│   └── utils/
│       ├── renderer.py              # [可视化] PyRender 渲染器
│       └── geometry.py              # [几何] rot6d_to_rotmat, aa_to_rotmat, perspective_projection
├── third-party/ViTPose/             # [子模块] ViTPose 全身关键点检测
└── _DATA/                           # [运行时生成] 模型缓存目录
```

### 每一步流程详解

```
RGB 图像
    ↓
[Step 1] 人体检测
    → Detectron2 ViTDet-H 检测人体 bounding boxes
    → 过滤：pred_classes == 0（人）且 scores > 0.5
    → 权重：model_final_f05665.pkl（首次运行自动下载）
    ↓
[Step 2] 手部关键点检测 + 手部定位
    → ViTPose+-H 输出全身 133 关键点
    → 最后 42 个为手部关键点：
        keypoints[-42:-21] → 左手 21 个
        keypoints[-21:]   → 右手 21 个
    → 置信度 > 0.5 的关键点 > 3 个则认为检测到手
    → 生成 is_right 标记（0=左手, 1=右手）
    → 权重：_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth
    → 代码：vitpose_model.py
    ↓
[Step 3] 图像裁剪与预处理
    → 手部 bounding box → ViTDetDataset → 256×256 图像 patch
    → 左手图像水平翻转（flip = right == 0）
    → ImageNet 均值/标准差归一化
    → 代码：hamer/datasets/vitdet_dataset.py
    ↓
[Step 4] ViT-H Backbone 图像编码
    → 输入 B×3×256×256，左右各裁 32px → B×3×256×192
    → ViT-Huge：patch_size=16, embed_dim=1280, depth=32, heads=16
    → 输出 B×1280×H'×W' 特征图
    → 代码：hamer/models/backbones/vit.py
    ↓
[Step 5] Transformer Decoder 回归 MANO 参数
    → 特征展平为 B×(H'×W')×1280 token 序列
    → 6 层 Transformer Decoder（8 头, mlp_dim=1024），单 token query 做 cross-attention
    → 回归输出（6D 旋转表示 → 转 3×3 旋转矩阵）：
        global_orient: B×1×3×3 — 手腕全局朝向
        hand_pose:     B×15×3×3 — 15 个手指关节旋转
        betas:         B×10      — 手部形状参数（retargeting 时忽略）
        cam:           B×3       — 弱透视相机参数 (scale, tx, ty)
    → 代码：hamer/models/heads/mano_head.py
    ↓
[Step 6] MANO 前向 → 网格 + 关键点
    → 输入 global_orient, hand_pose（旋转矩阵, pose2rot=False）, betas
    → smplx.MANOLayer 输出：
        vertices: B×778×3 — 手部网格顶点
        joints:   B×21×3  — OpenPose 21 关键点（经 joint_map 重排序）
    → 代码：hamer/models/mano_wrapper.py
    ↓
[Step 7] 相机转换与后处理
    → pred_cam (scale, tx, ty) + box_center + box_size → 全图坐标系 3D 平移
    → 左手 x 方向乘以 -1: verts[:,0] = (2*is_right-1)*verts[:,0]
    → 代码：hamer/models/hamer.py forward_step() + hamer/utils/renderer.py cam_crop_to_full()
    ↓
输出: pred_vertices (778×3), pred_keypoints_3d (21×3), pred_cam_t
```

### 从 MANO 参数到 Retargeting 可用的关节角

HaMeR 输出两种可用于 retargeting 的数据：

#### 方式一：直接使用 21 个 3D 关键点（推荐）

```python
out = model(batch)
keypoints_3d = out['pred_keypoints_3d']  # shape: (B, 21, 3) — OpenPose 格式
```

这 21 个关键点顺序：`[手腕, 拇指CMC, 拇指MCP, 拇指IP, 拇指Tip, 食指MCP, 食指PIP, 食指DIP, 食指Tip, 中指MCP, 中指PIP, 中指DIP, 中指Tip, 无名指MCP, 无名指PIP, 无名指DIP, 无名指Tip, 小指MCP, 小指PIP, 小指DIP, 小指Tip]`

**直接送入本项目的 `VectorOptimizationRetargeter` 或 `RuleBasedRetargeter` 即可。**

#### 方式二：使用 MANO 关节旋转矩阵

```python
pred_mano_params = out['pred_mano_params']
global_orient = pred_mano_params['global_orient']  # (B, 1, 3, 3) — 手腕
hand_pose = pred_mano_params['hand_pose']          # (B, 15, 3, 3) — 15 个手指关节

# 旋转矩阵 → 轴角
from scipy.spatial.transform import Rotation as R
rotmat = hand_pose[0, joint_idx].cpu().numpy()  # (3, 3)
aa = R.from_matrix(rotmat).as_rotvec()           # (3,) — 轴角表示
```

15 个关节对应 MANO 骨骼：每根手指 3 个关节（MCP、PIP、DIP），5 根手指 × 3 = 15。顺序：拇指(3) → 食指(3) → 中指(3) → 无名指(3) → 小指(3)。

> **坐标系说明**：HaMeR 所有输出（无论左右手）都在**右手坐标系**中。左手需在最终输出时翻转 x 轴。MANO 坐标系单位为**毫米**，原点在手腕根部。

### 完整复现步骤

```bash
# 1. 克隆（含 ViTPose 子模块）
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer

# 2. 创建环境
conda create --name hamer python=3.10
conda activate hamer

# 3. 安装 PyTorch（CUDA 11.7）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 4. 安装 HaMeR 及依赖
pip install -e .[all]

# 5. 安装 ViTPose 子模块
pip install -v -e third-party/ViTPose

# 6. 下载演示模型
bash fetch_demo_data.sh

# 7. 手动下载 MANO（需注册 https://mano.is.tue.mpg.de/ ）
#    将 MANO_RIGHT.pkl 放到：_DATA/data/mano/MANO_RIGHT.pkl

# 8. 运行推理
python demo.py --img_folder example_data --out_folder output --batch_size 8
```

### 与 Retargeting 的关系

- 输出 21 个 3D 关键点，**可直接送入本项目 `complete_retargeting_pipeline.py`** 的 `HandPreprocessor` → `VectorOptimizationRetargeter`
- 比 MediaPipe 更精确的 3D 位置（尤其深度方向） → 任务空间 IK 精度更高
- 可替换本项目 Stage 1 的 21 点检测模块，作为更高质量的输入源
- MANO 的 15 个关节旋转矩阵提供了另一种 retargeting 路径：直接映射旋转参数，而非通过 21 点做 IK

---

## LEAP Hand

| 属性 | 信息 |
|------|------|
| **论文** | LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning (CoRL 2023) |
| **机构** | Columbia University |
| **链接** | [https://github.com/leap-hand/LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim) |
| **难度** | ⭐⭐ |
| **硬件需求** | GPU for Isaac Gym sim |

### 为什么值得复现

LEAP Hand 是**低成本开源灵巧手**的标杆（整机 < USD 2000）。它的仿真环境基于 Isaac Gym，提供：
- URDF/MJCF 模型
- RL 训练代码（抓取、旋转等任务）
- 与真实 LEAP Hand 的通信接口

### 核心学习点

1. **16-DOF 灵巧手建模**：理解 Allegro/Shadow 之外的另一种 DOF 分配方案
2. **Isaac Gym 仿真**：大规模并行 RL 训练（可同时跑 4096 个环境）
3. **Sim-to-Real**：从仿真策略直接迁移到真实 LEAP Hand

### 复现步骤

```bash
git clone https://github.com/leap-hand/LEAP_Hand_Sim.git
cd LEAP_Hand_Sim && pip install -e .

# 训练抓取策略（Isaac Gym）
python train.py --task leap_hand_grasp --num_envs 2048

# 部署到真实 LEAP Hand（需硬件）
python deploy_real.py --checkpoint checkpoints/grasp.pt
```

### 与 Retargeting 的关系

- LEAP Hand 的 URDF 可直接用于本项目的 `VectorOptimizationRetargeter`（替换 O10 模型）
- 可作为 GeoRT 部署的目标平台（`assets/` 下放 LEAP Hand URDF，`config/` 写 joint_order JSON）

---

## DexPilot

| 属性 | 信息 |
|------|------|
| **论文** | DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System (ICRA 2020) |
| **机构** | UC Berkeley |
| **链接** | [https://github.com/byte-dance/dexpilot](https://github.com/byte-dance/dexpilot)（社区镜像） |
| **难度** | ⭐⭐⭐⭐ |
| **硬件需求** | RGB-D 摄像头（RealSense D435 推荐） |

### 为什么值得复现

DexPilot 是**早期 vision-based teleoperation 的经典工作**。它用 RGB-D 点云做手部跟踪，通过**优化问题**直接求解机器人关节角，无需显式的 21 点检测。

### 核心学习点

1. **点云配准**：人手的深度点云与机器人手模型点云对齐
2. **直接优化**：以机器人关节角为变量，最小化点云距离 + 关节正则化
3. **无穿戴**：不需要数据手套或动捕设备

### 复现步骤

```bash
git clone https://github.com/byte-dance/dexpilot.git
cd dexpilot && pip install -r requirements.txt

# 启动 RealSense 并运行跟踪
python run_dexpilot.py --camera realsense --robot shadow
```

### 与 Retargeting 的关系

- DexPilot 本质上是一个**端到端优化 retargeting**：
  ```
  目标函数 = w1 * ||人点云 - 机器人点云|| + w2 * ||关节正则化|| + w3 * 碰撞惩罚
  ```
- 与本项目 `docs/04-optimization-methods.md` 中 SLSQP 约束优化的思想一致
- 适合理解“为什么直接用优化比 Rule-based 更鲁棒”

---

## RDT-1B

| 属性 | 信息 |
|------|------|
| **论文** | RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation (2024) |
| **机构** | 清华大学，上海 AI Lab |
| **链接** | [https://github.com/thu-ml/RDT-1B](https://github.com/thu-ml/RDT-1B) |
| **难度** | ⭐⭐⭐ |
| **硬件需求** | GPU (>= 24GB VRAM for 1B model) |

### 为什么值得复现

RDT-1B 是目前**最大的双臂机器人扩散模型**（1B 参数），支持视觉-语言-动作联合生成。它的 action space 包含双手关节角，retargeting 结果可直接作为训练数据输入。

### 核心学习点

1. **Diffusion Transformer (DiT)**：用扩散模型生成动作序列，而非直接回归
2. **Bimanual Action Space**：双臂 + 双手的联合动作表示
3. **Scaling Law**：1B 参数模型在小样本机器人任务上的涌现能力

### 复现步骤

```bash
git clone https://github.com/thu-ml/RDT-1B.git
cd RDT-1B && pip install -r requirements.txt

# 下载预训练权重（~4GB）
python scripts/download_weights.py --model rdt-1b

# 推理 demo（需提供初始图像和语言指令）
python inference.py --prompt "pick up the red cube with both hands" \
    --initial_image demo/start.png
```

### 与 Retargeting 的关系

- RDT 的输入动作可以是**retargeting 后的关节角**：人做一遍动作 → retargeting → 存入数据集 → 训练 RDT
- 本项目 `examples/complete_retargeting_pipeline.py` 的输出可直接作为 RDT 的 action chunk
- 理解 Diffusion Policy 对 retargeting 精度的要求（低精度动作会导致扩散模型收敛慢）

---

## DexGraspNet

| 属性 | 信息 |
|------|------|
| **论文** | DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects (NeurIPS 2023) |
| **机构** | PKU, BIGAI |
| **链接** | [https://github.com/PKU-EPIC/DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) |
| **难度** | ⭐⭐ |
| **硬件需求** | GPU for grasp generation |

### 为什么值得复现

DexGraspNet 提供了 **1.32 million 个 Shadow Hand 抓取姿态**，覆盖 5355 个物体。它是理解“灵巧手能怎么抓”的绝佳数据集。

### 核心学习点

1. **Grasp Generation**：从物体点云生成可行的 Shadow Hand 抓取姿态
2. **物理可解性筛选**：用 Isaac Gym 验证每个抓取是否稳定（不掉落）
3. **Dataset Structure**：理解大规模机器人数据集的组织方式

### 复现步骤

```bash
git clone https://github.com/PKU-EPIC/DexGraspNet.git
cd DexGraspNet && pip install -r requirements.txt

# 下载数据集（~50GB）
python scripts/download_dataset.py

# 可视化抓取姿态
python visualize.py --object mug --grasp_id 42

# 训练 grasp generator（可选）
python train.py --config configs/pointnet_grasp.yaml
```

### 与 Retargeting 的关系

- 可作为 Learning-based retargeting 的**先验数据**：先学 DexGraspNet 中的抓取姿态分布，再微调到人手上
- 理解 Shadow Hand 的关节限位和可达空间，有助于设置 retargeting 的 bounds
- `docs/02-retargeting-taxonomy.md` 中 Learning-based 方法的数据来源参考

---

## OpenTeleVision

| 属性 | 信息 |
|------|------|
| **论文** | Open-TeleVision: Teleoperation with Immersive Active Visual Feedback (2024) |
| **机构** | UC San Diego |
| **链接** | [https://github.com/OpenTeleVision/TeleVision](https://github.com/OpenTeleVision/TeleVision) |
| **难度** | ⭐⭐ |
| **硬件需求** | Meta Quest 3 / Apple Vision Pro |

### 为什么值得复现

OpenTeleVision 用**VR 头显做沉浸式遥操作**，操作者通过双目立体视觉感知机器人视角，用手柄控制机器人双臂。它展示了 retargeting 在**人机交互层面**的应用。

### 核心学习点

1. **Stereo Visual Feedback**：延迟 < 100ms 的双目视频流回传
2. **VR 手柄到机器人末端映射**：手柄位姿 → 机械臂末端位姿（6-DOF tracking）
3. **沉浸式操作**：操作者临场感（telepresence）对任务成功率的影响

### 复现步骤

```bash
git clone https://github.com/OpenTeleVision/TeleVision.git
cd TeleVision && pip install -r requirements.txt

# 启动 VR 服务端
python server/television_server.py --device quest3

# 启动机器人控制端（需配合真实/仿真机器人）
python robot_control/franka_control.py
```

### 与 Retargeting 的关系

- VR 手柄提供的是**末端位姿**（position + quaternion），需要用 IK 转换为关节角 → 与本项目 Stage 3 机械臂 IK 直接相关
- 双手操作时的**左右手镜像问题**与 Dexterous Retargeting 双手系统完全一致
- 可作为人机交互层面的 retargeting 应用场景参考

---

## MyoSuite / MyoHand

| 属性 | 信息 |
|------|------|
| **论文** | MyoSuite: A contact-rich simulation suite for musculoskeletal motor control (NeurIPS 2022) |
| **机构** | Meta AI |
| **链接** | [https://github.com/facebookresearch/myosuite](https://github.com/facebookresearch/myosuite) |
| **难度** | ⭐⭐⭐ |
| **硬件需求** | GPU 可选（MuJoCo CPU 即可） |

### 为什么值得复现

MyoSuite 是**肌骨级别的人手仿真环境**，包含 MyoHand（20 个肌腱驱动的手指关节）。它能让你理解“人手的运动学约束”是什么，从而设计更好的 retargeting 映射。

### 核心学习点

1. **Tendon-driven Actuation**：肌腱驱动 vs 电机直驱对关节控制的影响
2. **Contact-rich Tasks**：钥匙旋转、球抓取等需要精细接触的任务
3. **RL for Hand Control**：PPO/SAC 在精细操作任务上的表现

### 复现步骤

```bash
git clone https://github.com/facebookresearch/myosuite.git
cd myosuite && pip install -e .

# 运行 MyoHand 抓取任务
python -m myosuite.utils.examine_env --env_name myoHandGrabExp-v0

# 训练 RL 策略
python myosuite/agents/train_rl.py --env myoHandPenTwirl-v0 --algorithm PPO
```

### 与 Retargeting 的关系

- MyoHand 的关节结构更接近真实人手（20 DOF），可作为 retargeting 的**参考源模型**
- 理解人手运动的生物力学约束，有助于解释为什么某些 retargeting 映射会“不自然”
- `examples/complete_retargeting_pipeline.py` 中 `SyntheticHandGenerator` 的灵感来源

---

## 项目对比与选型建议

| 你的目标 | 推荐项目 | 原因 |
|---------|---------|------|
| **快速跑通视觉遥操作全流程** | AnyTeleop | 开箱即用，MediaPipe + Shadow Hand |
| **获得更精确的 3D 手模型** | HaMeR | 单 RGB → MANO mesh，替代 MediaPipe |
| **低成本真机实验** | LEAP Hand | < USD 2000，URDF/MuJoCo/Isaac Gym 齐全 |
| **理解优化式 retargeting** | DexPilot | 点云直接优化，无显式 landmark |
| **生成双臂操作训练数据** | RDT-1B | Diffusion 生成动作，retargeting 输出可作输入 |
| **研究灵巧抓取** | DexGraspNet | 百万级抓取数据，Shadow Hand 物理验证 |
| **沉浸式人机交互** | OpenTeleVision | VR 双目反馈，双臂协同控制 |
| **理解人手生物力学** | MyoSuite | 肌腱驱动，接触丰富任务 |

---

## 复现路线图建议

### 路径 A：快速上手（1-2 周）

```
Week 1: 本项目 Stage 1-3 + LEAP Hand Sim 跑通抓取
Week 2: AnyTeleop + HaMeR 替换 MediaPipe，对比精度
```

### 路径 B：深入研究（1 个月）

```
Week 1-2: DexPilot + 本项目 Vector Optimization 方法对比
Week 3: DexGraspNet 数据分析 → 设计 Learning-based retargeter
Week 4: RDT-1B 微调，用 retargeting 数据训练双臂策略
```

### 路径 C：真机部署（2 周 + 硬件）

```
准备：LEAP Hand / Shadow Hand + RealSense D435
Week 1: 本项目 complete pipeline → LEAP Hand URDF → MuJoCo 仿真
Week 2: GeoRT 部署（URDF → config JSON → ROS2 控制节点）
```

---

## 附录：Retargeting 相关工具库

| 库 | 用途 | 链接 |
|----|------|------|
| **scipy.optimize.least_squares** | 数值 IK / 向量优化 | [SciPy Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) |
| **PyTorch3D** | 可微分渲染 / 3D 运算 | [GitHub](https://github.com/facebookresearch/pytorch3d) |
| **trimesh** | 3D mesh 处理（碰撞检测） | [GitHub](https://github.com/mikedh/trimesh) |
| **pinocchio** | 刚体动力学 / 解析 Jacobian | [GitHub](https://github.com/stack-of-tasks/pinocchio) |
| **pybullet** | 物理仿真（比 MuJoCo 轻量） | [PyBullet Docs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA) |
| **manopth** | MANO 模型 PyTorch 实现 | [GitHub](https://github.com/hassony2/manopth) |
