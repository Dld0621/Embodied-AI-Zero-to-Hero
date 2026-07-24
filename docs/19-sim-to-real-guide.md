# Sim-to-Real 实战指南：人形机器人与灵巧手迁移全栈手册

> **版本**: v1.0 (2026.07)  
> **适用对象**: 从事人形机器人/灵巧手 Sim-to-Real 迁移的工程师与研究人员  
> **核心目标**: 总结将 Sim-to-Real gap 控制在较低水平的常见方法，并提供可用于实验设计的工程模板；实际性能取决于任务、硬件和标定质量

---

## 目录

- [1. 核心概念：Sim-to-Real Gap 的来源](#1-核心概念sim-to-real-gap-的来源)
- [2. 2026 前沿方法概览](#2-2026-前沿方法概览)
  - [2.1 DexSim2Real：基础模型引导的域随机化](#21-dexsim2real基础模型引导的域随机化)
  - [2.2 Phys2Real：真实→仿真→真实的 RL 流程](#22-phys2real真实仿真真实的-rl-流程)
- [3. Domain Randomization 完整实现](#3-domain-randomization-完整实现)
  - [3.1 MuJoCo 中系统性的域随机化（代码片段）](#31-mujoco-中系统性的域随机化代码片段)
  - [3.2 视觉策略的域随机化参数范围](#32-视觉策略的域随机化参数范围)
  - [3.3 动力学随机化参数表](#33-动力学随机化参数表)
  - [3.4 课程式域随机化（CDR）实现](#34-课程式域随机化cdr实现)
- [4. System Identification（系统辨识）](#4-system-identification系统辨识)
  - [4.1 摩擦系数测量](#41-摩擦系数测量)
  - [4.2 质心与惯量辨识](#42-质心与惯量辨识)
  - [4.3 关节动力学参数辨识](#43-关节动力学参数辨识)
- [5. 视觉域适应](#5-视觉域适应)
  - [5.1 渲染器对齐](#51-渲染器对齐)
  - [5.2 基于 VLM 的视觉 realism critic](#52-基于-vlm-的视觉-realism-critic)
- [6. 延迟与接触补偿](#6-延迟与接触补偿)
  - [6.1 延迟测量与补偿](#61-延迟测量与补偿)
  - [6.2 接触参数调优](#62-接触参数调优)
  - [6.3 摩擦建模进阶](#63-摩擦建模进阶)
- [7. 灵巧手特殊考量](#7-灵巧手特殊考量)
  - [7.1 触觉传感器的 Sim-to-Real Gap](#71-触觉传感器的-sim-to-real-gap)
  - [7.2 指尖力控的迁移](#72-指尖力控的迁移)
  - [7.3 Shadow / Allegro / LEAP Hand 迁移经验](#73-shadow--allegro--leap-hand-迁移经验)
- [8. 调试与诊断流程](#8-调试与诊断流程)
  - [8.1 常见失败模式与诊断方法](#81-常见失败模式与诊断方法)
  - [8.2 渐进式调试流程](#82-渐进式调试流程)
- [9. 从仿真到真实的检查清单（Checklist）](#9-从仿真到真实的检查清单checklist)
- [10. 参考文献](#10-参考文献)

---

## 1. 核心概念：Sim-to-Real Gap 的来源

Sim-to-Real Gap 并非单一问题，而是由多个耦合的误差源叠加而成。对于人形机器人和灵巧手操作，主要的 Gap 来源可归纳为以下五类：

| 误差来源 | 典型偏差幅度 | 对灵巧手的影响 | 优先处理等级 |
|---------|------------|--------------|------------|
| **动力学参数误差** | 摩擦 ±30%，质量 ±10%，惯量 ±20% | 抓取力不足/过度，物体滑落 | P0 |
| **接触模型误差** | 静摩擦系数偏差 0.1-0.3 | 指尖接触不稳定，虚假滑动 | P0 |
| **感知域差异** | 光照、纹理、相机位姿 | 视觉策略失效，定位错误 | P1 |
| **执行器动力学** | 延迟 20-100ms，带宽限制 | 高频力控不稳定，振荡 | P1 |
| **传感器噪声** | 触觉零漂、力矩分辨率有限 | 力估计错误，闭环控制发散 | P2 |

**关键洞察**：DexSim2Real (Zeng et al., 2026) 的实验表明，在六组复杂灵巧操作任务中，**动力学参数误差**和**接触模型误差**合计贡献了约 65% 的 Sim-to-Real 性能损失。因此，工程上应优先解决物理参数不匹配问题，再处理视觉域差异。

---

## 2. 2026 前沿方法概览

### 2.1 DexSim2Real：基础模型引导的域随机化

**论文**: *DexSim2Real: Foundation Model-Guided Sim-to-Real Transfer for Generalizable Dexterous Manipulation* (arXiv:2605.05241, Tsinghua/Alibaba, 2026)

**核心贡献**：
- **FM-DR (Foundation Model-Guided Domain Randomization)**：利用 VLM（GPT-4V）作为视觉 realism critic，通过闭环 CMA-ES 优化仿真参数分布，而非依赖人工设定的随机化范围。
- **TVCAP (Tactile-Visual Cross-Attention Policy)**：将视觉-触觉交叉注意力机制引入零样本 Sim-to-Real RL，无需真实世界示教。
- **PSC (Progressive Skill Curriculum)**：基于 LLM 的任务分解 + δ-based 难度调度器，专门针对接触密集型灵巧任务。

**性能数据**：在六项真实世界灵巧操作任务（含盲评）上，平均真实世界成功率 **78.2%**，Sim-to-Real gap 仅 **8.3%**，超越 DrEureka 和 DeXtreme。

**FM-DR 的工程化要点**：
```python
# FM-DR 核心逻辑伪代码
# 1. 参数化分布为高斯混合模型 p(ξ; θ)
# 2. VLM 评分函数：s(ξ) = VLM(I_sim(ξ), I_real_ref, prompt_realism)
# 3. CMA-ES 优化目标：θ* = argmax E[s(ξ)] + λ·H(p)
#    λ=0.3 为 diversity-realism 权衡系数（网格搜索最优值）

# 典型收敛设置：
# - 种群大小: 16
# - 初始 σ: 0.3
# - 迭代轮数: 10-15 轮
# - 每轮样本: 20 个
# - 总 VLM 查询: 200-300 次/任务
```

**VLM Prompt 模板**：
```
Compare this simulated robot image to the real reference. 
Rate visual realism 1-10 considering lighting, texture, geometry, and physical plausibility.
```

**优化后的典型参数范围**（DexSim2Real 输出示例）：
| 参数 | 优化后范围 | 说明 |
|-----|----------|------|
| 摩擦系数 | 0.3 - 1.2 | 基于视觉 realism + 物理合理性联合优化 |
| 质量缩放 | 0.8x - 1.5x | 包含物体与末端执行器 |
| 光照强度 | 3 维参数 | 环境光 + 方向光 + 点光源 |
| 纹理噪声幅度 | 0.0 - 0.15 | 叠加在 base texture 上 |
| 相机位姿噪声 | ±5cm, ±5° | 位置与姿态偏移 |

### 2.2 Phys2Real：真实→仿真→真实的 RL 流程

**论文**: *Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation* (ICRA 2026, Stanford)

**核心贡献**：
Phys2Real 提出三阶段 real-to-sim-to-real 流程，核心创新在于将 **VLM 先验** 与 **交互式在线适应** 通过不确定性感知融合：

1. **Real-to-Sim**：通过分割的 3D Gaussian Splats 重建仿真就绪的 mesh
2. **Policy Learning**：训练以物理参数（如质心 CoM）为条件的策略
3. **Sim-to-Real Transfer**：不确定性感知的 VLM 先验 + 交互估计融合

**在线适应机制**：
- VLM 提供物理参数的**先验分布**（均值 + 不确定性）
- RMA-style 估计器在交互过程中**在线精化**参数信念
- 基于 ensemble 的**认知不确定性**加权融合两者

**性能数据**（平面推送任务）：
- T-block（底部配重）：Phys2Real **100%** vs DR **79%**
- T-block（顶部配重）：Phys2Real **57%** vs DR **23%**
- Hammer 推送：Phys2Real 比 DR 快 **15%**

**工程启示**：
- **物理参数条件化策略**（physics-conditioned policy）比纯 DR 更样本高效
- **VLM 先验 + 交互融合** 缺一不可，单独使用任一方均无法达到 privileged 性能
- 对于质心偏移显著的物体（如锤子），Phys2Real 的优势尤为明显

---

## 3. Domain Randomization 完整实现

### 3.1 MuJoCo 中系统性的域随机化（代码片段）

以下代码提供了 MuJoCo 环境下完整的域随机化实现，涵盖动力学、几何、视觉三大类参数。基于 2026 年最佳实践，包含 DexSim2Real 推荐的参数范围。

```python
"""
MuJoCo Domain Randomization 完整实现
适用：人形机器人 / 灵巧手操作任务
基于：DexSim2Real (2026) + 社区最佳实践
"""

import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class RandomizationConfig:
    """域随机化配置"""
    # 动力学参数
    friction_range: Tuple[float, float] = (0.3, 1.2)        # 摩擦系数范围
    mass_scale_range: Tuple[float, float] = (0.8, 1.5)      # 质量缩放范围
    armature_scale_range: Tuple[float, float] = (0.7, 1.4)  # 电机惯量缩放
    damping_scale_range: Tuple[float, float] = (0.8, 1.3)   # 关节阻尼缩放
    frictionloss_range: Tuple[float, float] = (0.0, 0.05)   # 摩擦损失范围
    
    # 关节参数
    joint_pos_range: Tuple[float, float] = (-0.05, 0.05)    # 初始关节位置偏移 (rad)
    joint_vel_range: Tuple[float, float] = (-0.1, 0.1)      # 初始关节速度偏移
    
    # 接触参数
    solref_range: Tuple[float, float] = (0.001, 0.02)       # 接触求解器参考参数
    solimp_range: Tuple[float, float] = (0.8, 0.99)         # 接触求解器阻抗
    margin_range: Tuple[float, float] = (0.0, 0.005)        # 接触边界
    
    # 视觉参数
    light_pos_range: float = 0.3                            # 光源位置偏移 (m)
    light_diffuse_range: Tuple[float, float] = (0.7, 1.3)   # 漫反射强度缩放
    camera_pos_range: float = 0.05                          # 相机位置偏移 (m)
    camera_quat_range: float = 0.05                         # 相机姿态偏移
    
    # 纹理/渲染
    rgb_noise_sigma: float = 0.02                           # RGB 噪声标准差
    texture_scale_range: Tuple[float, float] = (0.9, 1.1)   # 纹理缩放


class MuJoCoDomainRandomizer:
    """MuJoCo 域随机化器"""
    
    def __init__(self, model: mujoco.MjModel, config: RandomizationConfig = None):
        self.model = model
        self.config = config or RandomizationConfig()
        self._defaults = self._capture_defaults()
    
    def _capture_defaults(self) -> Dict[str, np.ndarray]:
        """捕获默认参数，用于恢复和相对随机化"""
        defaults = {}
        defaults['body_mass'] = self.model.body_mass.copy()
        defaults['dof_frictionloss'] = self.model.dof_frictionloss.copy()
        defaults['dof_damping'] = self.model.dof_damping.copy()
        defaults['dof_armature'] = self.model.dof_armature.copy()
        defaults['geom_friction'] = self.model.geom_friction.copy()
        defaults['geom_solref'] = self.model.geom_solref.copy()
        defaults['geom_solimp'] = self.model.geom_solimp.copy()
        defaults['geom_margin'] = self.model.geom_margin.copy()
        defaults['light_pos'] = self.model.light_pos.copy()
        defaults['light_diffuse'] = self.model.light_diffuse.copy()
        defaults['cam_pos'] = self.model.cam_pos.copy()
        defaults['cam_quat'] = self.model.cam_quat.copy()
        return defaults
    
    def randomize_dynamics(self) -> None:
        """随机化动力学参数"""
        cfg = self.config
        defs = self._defaults
        
        # 1. 摩擦系数 (geom_friction: [slide, spin, roll])
        # 对操作任务，主要关注滑动摩擦 (第一维)
        friction_scale = np.random.uniform(*cfg.friction_range)
        self.model.geom_friction[:, 0] = defs['geom_friction'][:, 0] * friction_scale
        self.model.geom_friction[:, 1] = defs['geom_friction'][:, 1] * friction_scale * 0.1
        self.model.geom_friction[:, 2] = defs['geom_friction'][:, 2] * friction_scale * 0.01
        
        # 2. 质量缩放（排除固定基座）
        mass_scale = np.random.uniform(*cfg.mass_scale_range)
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] != 0:  # 非世界坐标系
                self.model.body_mass[i] = defs['body_mass'][i] * mass_scale
        
        # 3. 电机惯量 (armature)
        armature_scale = np.random.uniform(*cfg.armature_scale_range)
        self.model.dof_armature[:] = defs['dof_armature'] * armature_scale
        
        # 4. 关节阻尼
        damping_scale = np.random.uniform(*cfg.damping_scale_range)
        self.model.dof_damping[:] = defs['dof_damping'] * damping_scale
        
        # 5. 摩擦损失 (frictionloss)
        frictionloss = np.random.uniform(*cfg.frictionloss_range)
        self.model.dof_frictionloss[:] = defs['dof_frictionloss'] + frictionloss
    
    def randomize_contacts(self) -> None:
        """随机化接触参数"""
        cfg = self.config
        defs = self._defaults
        
        # 接触求解器参数 solref: [timeconst, dampratio]
        timeconst = np.random.uniform(0.001, 0.02)
        dampratio = np.random.uniform(0.5, 1.0)
        self.model.geom_solref[:, 0] = timeconst
        self.model.geom_solref[:, 1] = dampratio
        
        # 接触阻抗 solimp: [dmin, dmax, width, midpoint, power]
        dmin = np.random.uniform(0.8, 0.95)
        dmax = np.random.uniform(0.95, 0.999)
        self.model.geom_solimp[:, 0] = dmin
        self.model.geom_solimp[:, 1] = dmax
        
        # 接触边界 margin
        margin = np.random.uniform(*cfg.margin_range)
        self.model.geom_margin[:] = defs['geom_margin'] + margin
    
    def randomize_visual(self) -> None:
        """随机化视觉/渲染参数（视觉策略用）"""
        cfg = self.config
        defs = self._defaults
        
        # 光源位置
        light_offset = np.random.uniform(-cfg.light_pos_range, cfg.light_pos_range, size=3)
        self.model.light_pos[:] = defs['light_pos'] + light_offset
        
        # 光源漫反射强度
        diffuse_scale = np.random.uniform(*cfg.light_diffuse_range)
        self.model.light_diffuse[:] = np.clip(defs['light_diffuse'] * diffuse_scale, 0, 1)
        
        # 相机位姿
        cam_pos_offset = np.random.uniform(-cfg.camera_pos_range, cfg.camera_pos_range, size=3)
        self.model.cam_pos[:] = defs['cam_pos'] + cam_pos_offset
        
        # 相机姿态 (小角度扰动，用轴角表示后转四元数)
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        angle = np.random.uniform(-cfg.camera_quat_range, cfg.camera_quat_range)
        # 简化为直接扰动四元数 (小角度近似)
        self.model.cam_quat[:] = defs['cam_quat'] + np.random.randn(*defs['cam_quat'].shape) * 0.02
        self.model.cam_quat[:] /= (np.linalg.norm(self.model.cam_quat, axis=-1, keepdims=True) + 1e-8)
    
    def randomize_initial_state(self, data: mujoco.MjData) -> None:
        """随机化初始状态"""
        cfg = self.config
        
        # 关节初始位置偏移
        qpos_noise = np.random.uniform(*cfg.joint_pos_range, size=self.model.nq)
        data.qpos[:] += qpos_noise
        
        # 关节初始速度
        qvel_noise = np.random.uniform(*cfg.joint_vel_range, size=self.model.nv)
        data.qvel[:] = qvel_noise
    
    def apply_all(self, data: mujoco.MjData = None) -> None:
        """应用全部随机化"""
        self.randomize_dynamics()
        self.randomize_contacts()
        self.randomize_visual()
        if data is not None:
            self.randomize_initial_state(data)
    
    def reset(self) -> None:
        """恢复默认参数"""
        for key, val in self._defaults.items():
            attr = getattr(self.model, key)
            attr[:] = val


# ============ 使用示例 ============
"""
# 在训练循环中集成：
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
randomizer = MuJoCoDomainRandomizer(model)

for episode in range(N_EPISODES):
    # 每个 episode 前重新随机化
    randomizer.apply_all(data)
    
    # 运行 episode...
    
    # 如需恢复默认参数进行验证
    # randomizer.reset()
"""
```

### 3.2 视觉策略的域随机化参数范围

对于视觉策略（如端到端的 image-to-action 策略），除了物理参数外，**视觉域的随机化**至关重要。以下参数范围基于 DexSim2Real 优化结果与 2026 年社区最佳实践：

```python
# 视觉域随机化：基于图像增强的实现（适用于像素输入策略）
import cv2
import numpy as np

def randomize_visual_observation(image: np.ndarray, training: bool = True) -> np.ndarray:
    """
    对输入图像进行域随机化增强
    image: HWC, uint8, [0, 255]
    """
    if not training:
        return image
    
    img = image.astype(np.float32) / 255.0
    
    # 1. 亮度 / 对比度
    alpha = np.random.uniform(0.8, 1.2)  # 对比度
    beta = np.random.uniform(-0.1, 0.1)  # 亮度偏移
    img = np.clip(img * alpha + beta, 0, 1)
    
    # 2. 颜色抖动 (HSV 空间)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-0.02, 0.02)) % 1.0  # Hue
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 1)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 1)  # Value
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 3. 高斯噪声
    noise = np.random.normal(0, np.random.uniform(0.005, 0.02), img.shape)
    img = np.clip(img + noise, 0, 1)
    
    # 4. 随机阴影/遮挡 (Simulating lighting variation)
    if np.random.rand() < 0.3:
        h, w = img.shape[:2]
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(w // 4, w // 2)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = np.clip(1 - dist / radius, 0, 1)
        img = img * (0.7 + 0.3 * mask[:, :, None])
    
    # 5. 随机裁剪/缩放 (模拟相机位置变化)
    if np.random.rand() < 0.3:
        scale = np.random.uniform(0.95, 1.05)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        if scale > 1.0:
            # 中心裁剪回原尺寸
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img = img[start_h:start_h+h, start_w:start_w+w]
        else:
            # 边缘填充
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = np.pad(img, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='edge')
    
    return (img * 255).astype(np.uint8)
```

**视觉域随机化参数速查表**：

| 参数类型 | 随机化范围 | 分布 | 备注 |
|---------|----------|------|------|
| 相机位置偏移 | ±5cm (xyz) | Uniform | DexSim2Real 优化值；桌面操作可放宽至 ±10cm |
| 相机姿态偏移 | ±5° (各轴) | Uniform | 小角度扰动，避免策略依赖精确视角 |
| 亮度偏移 | ±10% | Uniform | 像素值缩放因子 0.9-1.1 |
| 对比度 | ±20% | Uniform | alpha 因子 0.8-1.2 |
| 饱和度 | ±20% | Uniform | HSV 空间调整 |
| 色相偏移 | ±2% | Uniform | H 通道 ±0.02 |
| 高斯噪声 σ | 0.5% - 2% | Uniform | 相对于 [0,1] 像素值 |
| 随机遮挡块 | 0-3 块，尺寸 10x10-30x30 | Bernoulli + Uniform | 模拟传感器坏点/遮挡 |
| 纹理噪声幅度 | 0-15% | Uniform | 叠加在 base texture 上 |
| 背景替换 | 概率 0.3 | Bernoulli | 使用随机真实场景背景 |

### 3.3 动力学随机化参数表

以下表格汇总了人形机器人/灵巧手操作中各类动力学参数的推荐随机化范围：

| 参数类别 | 具体参数 | 基础值获取方式 | 随机化范围 | 分布类型 | 关键程度 |
|---------|---------|-------------|----------|---------|---------|
| **摩擦** | geom 滑动摩擦 | 斜坡实验测量 | ±30% (或 0.3-1.2) | Uniform/Log-Uniform | ★★★ |
| **摩擦** | geom 滚动摩擦 | 经验值 | 基础值 × [0.5, 2.0] | Uniform | ★☆☆ |
| **质量** | 连杆质量 | CAD 模型 / 称重 | ±10% (或 0.8x-1.5x) | Uniform | ★★★ |
| **惯量** | 连杆惯量张量 | CAD 模型估算 | ±20% | Uniform | ★★☆ |
| **质心** | body 质心偏移 | CAD 模型 / 悬挂法 | ±1cm (xyz) | Uniform | ★★★ |
| **关节** | 阻尼 (dof_damping) | 系统辨识 | ±30% (或 0.8x-1.3x) | Uniform | ★★☆ |
| **关节** | 摩擦损失 (frictionloss) | 系统辨识 | 0.0 - 0.05 Nm | Uniform | ★★☆ |
| **关节** | 电机惯量 (armature) | 厂商规格 | ±40% (或 0.7x-1.4x) | Uniform | ★★☆ |
| **关节** | 关节范围 (jnt_range) | 实际测量 | ±2° | Uniform | ★☆☆ |
| **接触** | solref (timeconst) | 默认 0.02 | [0.001, 0.02] | Uniform | ★★☆ |
| **接触** | solimp (dmin/dmax) | 默认 0.9/0.95 | dmin∈[0.8,0.95], dmax∈[0.95,0.999] | Uniform | ★★☆ |
| **接触** | margin | 默认 0.0 | [0.0, 0.005] | Uniform | ★☆☆ |
| **执行器** | 增益缩放 (ctrl_range) | 标定值 | ±10% | Uniform | ★★☆ |
| **环境** | 重力加速度 | 9.81 | [9.5, 10.0] | Uniform | ★☆☆ |
| **环境** | 时间步长 (timestep) | 默认 | [0.0005, 0.002] | Uniform | ★★☆ |

### 3.4 课程式域随机化（CDR）实现

均匀随机化（UDR）在宽范围下会导致大量不现实的仿真配置，降低训练效率。课程式域随机化（Curriculum Domain Randomization）通过逐步扩展随机化范围解决此问题。

```python
class CurriculumDomainRandomizer:
    """课程式域随机化：逐步扩展随机化范围"""
    
    def __init__(self, randomizer: MuJoCoDomainRandomizer, total_steps: int):
        self.randomizer = randomizer
        self.total_steps = total_steps
        self.current_step = 0
        
        # 初始范围和最终范围
        self.friction_start = (0.8, 1.0)
        self.friction_end = (0.3, 1.2)
        self.mass_start = (0.95, 1.05)
        self.mass_end = (0.8, 1.5)
        
    def _schedule(self, start: Tuple, end: Tuple, progress: float) -> Tuple:
        """线性插值扩展范围"""
        return (
            start[0] + (end[0] - start[0]) * progress,
            start[1] + (end[1] - start[1]) * progress
        )
    
    def step(self) -> None:
        """每训练步调用，更新随机化范围"""
        progress = min(self.current_step / self.total_steps, 1.0)
        
        # 使用 ease-in-out 曲线：前期慢、后期快
        progress = progress ** 2 * (3 - 2 * progress)
        
        # 更新配置
        cfg = self.randomizer.config
        cfg.friction_range = self._schedule(self.friction_start, self.friction_end, progress)
        cfg.mass_scale_range = self._schedule(self.mass_start, self.mass_end, progress)
        
        self.current_step += 1
    
    def apply(self, data: mujoco.MjData = None) -> None:
        """应用当前阶段的随机化"""
        self.randomizer.apply_all(data)
```

---

## 4. System Identification（系统辨识）

系统辨识是 Sim-to-Real 的前提。**"在随机化之前，先测量；在测量之后，再随机化"**。

### 4.1 摩擦系数测量

**方法：斜坡法（最可靠）**

```python
def measure_friction_coefficient():
    """
    斜坡法测量静摩擦系数
    1. 将物体放置于可倾斜平面上
    2. 缓慢增加倾角 θ 直到物体开始滑动
    3. μ_static = tan(θ)
    
    关键：测量 5-10 次取平均，分别测量不同材质配对
    """
    # 示例数据（桌面操作常见材质）
    friction_data = {
        'finger_rubber__object_plastic': 0.65,   # 橡胶指尖 - 塑料物体
        'finger_rubber__object_wood': 0.55,      # 橡胶指尖 - 木质物体
        'finger_rubber__object_metal': 0.45,     # 橡胶指尖 - 金属物体
        'finger_silicone__object_plastic': 0.75, # 硅胶指尖 - 塑料物体
        'table_wood__object_plastic': 0.35,      # 桌面 - 物体底部
    }
    
    # 在仿真中设置基础摩擦，并在此基础上 ±30% 随机化
    base_friction = friction_data['finger_rubber__object_plastic']
    sim_friction_range = (base_friction * 0.7, base_friction * 1.3)
    
    return base_friction, sim_friction_range
```

**进阶：动摩擦测量**
- 使用力传感器以恒定速度拉动物体，测量稳定拉力 F
- μ_kinetic = F / N（N 为正压力）
- 通常 μ_kinetic ≈ 0.7-0.9 × μ_static

### 4.2 质心与惯量辨识

**质心测量（悬挂法）**：
1. 用细线悬挂物体，从悬挂点铅垂向下画线
2. 换一个悬挂点重复
3. 两线交点即为质心投影
4. 对于不规则物体，重复 3 次以上取平均

**惯量测量（扭摆法）**：
```python
def estimate_inertia_tensor(mass: float, dimensions: Tuple[float, float, float]) -> np.ndarray:
    """
    对于规则形状物体，使用理论公式估算惯量张量
    对于不规则物体，建议使用扭摆实验测量
    
    box: (w, h, d) -> I_xx = m*(h^2+d^2)/12
    cylinder: (r, h) -> I_xx = m*r^2/2, I_yy = I_zz = m*(3r^2+h^2)/12
    sphere: (r) -> I = 2*m*r^2/5
    """
    w, h, d = dimensions
    I_xx = mass * (h**2 + d**2) / 12
    I_yy = mass * (w**2 + d**2) / 12
    I_zz = mass * (w**2 + h**2) / 12
    return np.diag([I_xx, I_yy, I_zz])
```

### 4.3 关节动力学参数辨识

```python
import numpy as np
from scipy.optimize import minimize

def identify_joint_dynamics(joint_data: np.ndarray):
    """
    关节动力学参数辨识
    
    joint_data: N x 3 数组，列分别为 [位置(rad), 速度(rad/s), 力矩(Nm)]
    
    模型: τ = I·q̈ + b·q̇ + c·sign(q̇) + τ_frictionloss
    简化为稳态辨识: τ = b·q̇ + c·sign(q̇) + τ_f
    """
    q, qd, tau = joint_data[:, 0], joint_data[:, 1], joint_data[:, 2]
    
    def model(params, qd):
        b, c, tau_f = params
        return b * qd + c * np.sign(qd) + tau_f
    
    def loss(params):
        tau_pred = model(params, qd)
        return np.mean((tau - tau_pred) ** 2)
    
    # 初始猜测: [阻尼, 库仑摩擦, 摩擦损失]
    result = minimize(loss, x0=[0.01, 0.05, 0.0], method='L-BFGS-B',
                     bounds=[(0, 1), (0, 1), (0, 0.1)])
    
    damping, coulomb_friction, frictionloss = result.x
    return {
        'damping': damping,
        'coulomb_friction': coulomb_friction,
        'frictionloss': frictionloss,
        'rmse': np.sqrt(result.fun)
    }
```

---

## 5. 视觉域适应

### 5.1 渲染器对齐

视觉策略的 Sim-to-Real gap 主要来源于渲染器与真实相机之间的差异。2026 年最佳实践建议：

1. **使用真实相机标定参数**：在 MuJoCo/Isaac Sim 中精确设置相机内参 (fx, fy, cx, cy) 和畸变系数
2. **HDR 环境贴图**：使用真实场景捕获的 HDRi 作为环境光照，而非纯色背景
3. **基于物理的材质**：物体材质参数（roughness, metallic, specular）应参考真实测量

```python
# MuJoCo 相机参数设置示例（匹配真实相机）
camera_config = {
    'fovy': 45.0,           # 垂直视场角，与真实相机匹配
    'ipd': 0.0,             # 单目相机设为 0
    'resolution': (640, 480),
    # 内参矩阵（从相机标定获得）
    'intrinsics': {
        'fx': 554.25,       # 焦距 x (pixels)
        'fy': 554.25,       # 焦距 y (pixels)
        'cx': 320.0,        # 主点 x
        'cy': 240.0,        # 主点 y
    }
}

# 畸变系数（OpenCV 模型）
distortion_coeffs = {
    'k1': 0.05, 'k2': -0.12, 'p1': 0.001, 'p2': -0.001, 'k3': 0.02
}
```

### 5.2 基于 VLM 的视觉 Realism Critic

受 DexSim2Real 启发，可利用 VLM 自动化评估仿真渲染与真实图像的相似度：

```python
import openai
import base64
from io import BytesIO

def vlm_realism_score(sim_image: np.ndarray, real_images: List[np.ndarray]) -> float:
    """
    使用 VLM 评估仿真图像的真实度
    
    sim_image: 仿真渲染图像 (RGB, uint8)
    real_images: 真实参考图像列表
    
    返回: 1-10 的真实度评分
    """
    def encode_image(img):
        buffered = BytesIO()
        Image.fromarray(img).save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    sim_b64 = encode_image(sim_image)
    real_b64_list = [encode_image(img) for img in real_images]
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare this simulated robot image to the real reference. Rate visual realism 1-10 considering lighting, texture, geometry, and physical plausibility."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sim_b64}"}},
            {"type": "text", "text": "Reference real images:"},
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in real_b64_list]
    }]
    
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=50
    )
    
    # 解析评分（从文本中提取数字）
    text = response.choices[0].message.content
    score = float(re.search(r'\b(\d+(?:\.\d+)?)\b', text).group(1))
    return min(max(score, 1.0), 10.0)
```

---

## 6. 延迟与接触补偿

### 6.1 延迟测量与补偿

**延迟来源分析**：
| 延迟来源 | 典型值 | 补偿方法 |
|---------|--------|---------|
| 相机曝光 + 传输 | 5-33ms | 使用全局快门相机，ROS2 零拷贝传输 |
| 视觉推理 (CNN/ViT) | 10-50ms | TensorRT/ONNX 优化，模型量化 |
| 策略网络前向 | 2-10ms | 批量推理，GPU 预热 |
| 通信延迟 (USB/CAN/EtherCAT) | 1-5ms | EtherCAT 优先，控制频率 ≥ 1kHz |
| 执行器响应 | 5-20ms | 电流环前馈，低频指令滤波 |
| **总延迟** | **20-120ms** | **状态预测 / 延迟增强训练** |

**延迟测量方法**：
```python
def measure_system_delay(robot, camera):
    """
    测量端到端延迟
    方法：发送一个已知的控制指令（如关节速度阶跃），
          同时记录相机时间戳和关节反馈时间戳
    """
    # 1. 发送阶跃指令并记录发送时间 t_cmd
    t_cmd = time.time()
    robot.set_joint_velocity([0.1, 0, 0, 0, ...])
    
    # 2. 轮询检测关节速度响应超过阈值
    while True:
        state = robot.get_joint_state()
        t_feedback = state.timestamp
        if abs(state.velocity[0]) > 0.05:
            break
    
    # 3. 计算延迟
    delay = t_feedback - t_cmd
    return delay
```

**延迟补偿方案**：

```python
class DelayCompensator:
    """基于状态预测的延迟补偿器"""
    
    def __init__(self, delay_steps: int, dt: float):
        self.delay_steps = delay_steps  # 以控制周期为单位的延迟
        self.dt = dt
        self.action_buffer = deque(maxlen=delay_steps + 1)
        
    def predict_future_state(self, current_state: np.ndarray, 
                            dynamics_fn, steps: int) -> np.ndarray:
        """
        使用前向动力学预测未来状态
        current_state: [q, qd] 当前观测
        dynamics_fn: 状态转移函数
        """
        state = current_state.copy()
        for action in list(self.action_buffer)[-steps:]:
            state = dynamics_fn(state, action, self.dt)
        return state
    
    def compensate(self, current_state: np.ndarray, 
                   policy_fn, dynamics_fn) -> np.ndarray:
        """
        主补偿逻辑：
        1. 预测 delay_steps 后的状态
        2. 在该预测状态上计算策略输出
        """
        predicted_state = self.predict_future_state(
            current_state, dynamics_fn, self.delay_steps
        )
        action = policy_fn(predicted_state)
        self.action_buffer.append(action)
        return action

# 训练时的延迟增强（更简单的方案）
def add_delay_to_observations(obs_buffer: deque, delay_steps: int):
    """
    在训练时，向策略输入延迟的观测而非当前观测
    使策略对延迟天然鲁棒
    """
    if len(obs_buffer) < delay_steps + 1:
        return obs_buffer[0]
    return obs_buffer[-(delay_steps + 1)]
```

### 6.2 接触参数调优

MuJoCo 接触参数直接影响仿真的稳定性和真实性。以下是经过验证的参数调优指南：

```xml
<!-- MuJoCo XML 接触参数推荐配置 -->
<mujoco model="dexterous_hand">
  <option timestep="0.002" solver="Newton" iterations="50" tolerance="1e-10">
    <!-- timestep: 灵巧手操作建议 0.001-0.002s -->
    <!-- solver: Newton 对接触问题更稳定 -->
  </option>
  
  <default>
    <geom contype="1" conaffinity="1" 
          friction="0.8 0.005 0.0001"  <!-- [滑动, 扭转, 滚动] -->
          solref="0.01 1"               <!-- [timeconst, dampratio] -->
          solimp="0.9 0.95 0.001"       <!-- [dmin, dmax, width] -->
          margin="0.0"                  <!-- 接触边界，0 表示精确接触 -->
          gap="0.0"/>                   <!-- 间隙，0 表示无间隙 -->
    
    <joint armature="0.01"             <!-- 电机转子惯量 -->
            damping="0.5"               <!-- 关节阻尼 -->
            frictionloss="0.01"/>       <!-- 摩擦损失 -->
  </default>
  
  <!-- 灵巧手指尖特殊配置（更硬的接触） -->
  <default class="fingertip">
    <geom friction="1.2 0.01 0.0001"
          solref="0.005 1"
          solimp="0.95 0.99 0.001"
          condim="6"/>                  <!-- 6维接触用于精确力控 -->
  </default>
</mujoco>
```

**接触参数调优决策树**：

```
仿真出现问题？
├── 物体穿透 (penetration)
│   ├── 减小 timestep（0.002 → 0.001）
│   ├── 增大 solimp 的 dmax（0.95 → 0.99）
│   └── 减小 solref 的 timeconst（0.02 → 0.005）
│
├── 接触抖动 / 物体弹跳
│   ├── 增大 solref 的 timeconst（0.005 → 0.01-0.02）
│   ├── 调整 solimp 的 dmin/dmax 比例（保持 dmin < dmax）
│   └── 检查是否有过度约束（冗余接触）
│
├── 物体异常滑动
│   ├── 增大滑动摩擦系数（geom friction 第一维）
│   ├── 使用 elliptic friction cone（cone="elliptic"）
│   └── 检查法向力是否足够（增加物体质量或接触刚度）
│
└── 仿真发散 (NaN)
    ├── 减小 timestep
    ├── 检查质量/惯量是否为正
    ├── 检查是否有无限大约束
    └── 启用 flag: <flag warmstart="enable"/>
```

### 6.3 摩擦建模进阶

对于需要精确力控的灵巧操作，推荐使用 **elliptic friction cone** 和完整的 6 维接触：

```xml
<!-- 高精度摩擦建模（MuJoCo） -->
<geom name="fingertip" type="sphere" size="0.008"
      friction="1.0 0.1 0.001"   <!-- [μ_slide, μ_spin, μ_roll] -->
      condim="6"                 <!-- 6维接触：法向 + 2切向 + 扭转 + 2滚动 -->
      cone="elliptic"            <!-- 椭圆摩擦锥，更真实 -->
      solref="0.005 1"
      solimp="0.95 0.99 0.001"/>
```

**摩擦参数物理含义**：
- `μ_slide` (第一维): 滑动摩擦系数，决定切向力上限 = μ × 法向力
- `μ_spin` (第二维): 扭转摩擦系数，抵抗绕法向轴的旋转
- `μ_roll` (第三维): 滚动摩擦系数，抵抗滚动运动

**推荐比例**：对于橡胶/硅胶指尖接触，μ_spin ≈ 0.1 × μ_slide，μ_roll ≈ 0.001 × μ_slide。

---

## 7. 灵巧手特殊考量

### 7.1 触觉传感器的 Sim-to-Real Gap

灵巧手操作高度依赖触觉反馈，但触觉传感器的 Sim-to-Real Gap 是极具挑战性的：

| 传感器类型 | 仿真模型 | 真实 Gap 来源 | 缓解策略 |
|-----------|---------|-------------|---------|
| **XELA uSkin** (电容式) | 接触力→taxel 压力图 | 非线性响应、温度漂移、串扰 | 仿真中加高斯噪声 + 温度缩放 |
| **OptoForce/ BioTac** (光学) | 接触法向 + 切向力 | 校准误差、迟滞、频率限制 | 低通滤波 + 增益随机化 |
| **FSR** (力敏电阻) | 二元/压力接触 | 阈值波动、磨损 | 二元化输出 + 阈值随机化 |
| **GelSight** (视觉触觉) | 弹性体变形渲染 | 光照变化、凝胶磨损 | 渲染域随机化 + 纹理替换 |

**触觉仿真→真实迁移方案**：

```python
class TactileSim2RealAdapter:
    """
    触觉传感器 Sim-to-Real 适配器
    适用于 XELA / OptoForce 等阵列式触觉传感器
    """
    
    def __init__(self, n_taxels: int = 15, sensor_type: str = 'xela'):
        self.n_taxels = n_taxels
        self.sensor_type = sensor_type
        
        # 从真实传感器标定获得的参数
        self.calibration = {
            'sensitivity': np.ones(n_taxels) * 0.1,      # N / taxel unit
            'zero_offset': np.zeros(n_taxels),            # 零漂
            'crosstalk_matrix': np.eye(n_taxels) * 0.9,   # 串扰矩阵
            'noise_sigma': 0.02,                          # 噪声标准差 (N)
            'temp_coeff': 0.001,                          # 温度系数 (N/°C)
        }
    
    def sim_to_real(self, sim_contact_forces: np.ndarray, 
                    temperature: float = 25.0) -> np.ndarray:
        """
        将仿真接触力转换为模拟的真实传感器读数
        
        sim_contact_forces: [n_taxels] 仿真计算的各 taxel 接触力 (N)
        """
        # 1. 应用灵敏度缩放
        real_units = sim_contact_forces / self.calibration['sensitivity']
        
        # 2. 添加串扰
        real_units = self.calibration['crosstalk_matrix'] @ real_units
        
        # 3. 添加零漂和温度漂移
        temp_drift = (temperature - 25.0) * self.calibration['temp_coeff']
        real_units += self.calibration['zero_offset'] + temp_drift
        
        # 4. 添加传感器噪声
        noise = np.random.normal(0, self.calibration['noise_sigma'], self.n_taxels)
        real_units += noise / self.calibration['sensitivity']
        
        # 5. 量化和截断 (模拟 ADC)
        real_units = np.clip(real_units, 0, 4095)  # 12-bit ADC
        
        return real_units
    
    def train_time_randomization(self, sim_contact_forces: np.ndarray) -> np.ndarray:
        """
        训练时的域随机化：随机化触觉传感器参数
        """
        # 随机化灵敏度 ±20%
        sensitivity = self.calibration['sensitivity'] * np.random.uniform(0.8, 1.2, self.n_taxels)
        
        # 随机化零漂 ±0.05N
        zero_offset = np.random.uniform(-0.05, 0.05, self.n_taxels)
        
        # 随机化串扰 ±10%
        crosstalk = np.eye(self.n_taxels) + np.random.randn(self.n_taxels, self.n_taxels) * 0.1
        
        # 随机化噪声 ±50%
        noise_sigma = self.calibration['noise_sigma'] * np.random.uniform(0.5, 1.5)
        
        real_units = sim_contact_forces / sensitivity
        real_units = crosstalk @ real_units + zero_offset
        real_units += np.random.normal(0, noise_sigma, self.n_taxels) / sensitivity
        
        return np.clip(real_units, 0, 4095)
```

### 7.2 指尖力控的迁移

指尖力控是灵巧操作的核心。Sim-to-Real 迁移时的关键工程实践：

**1. 力控策略训练建议**
```python
# 在仿真中训练力控策略时，向力矩指令添加噪声
def add_force_control_noise(torque_cmd: np.ndarray, 
                            noise_ratio: float = 0.05) -> np.ndarray:
    """
    模拟真实执行器的力控噪声
    """
    noise = np.random.normal(0, noise_ratio * np.abs(torque_cmd))
    return torque_cmd + noise

# 同时随机化力传感器读数（如果策略使用力反馈）
def randomize_force_sensor_reading(force: np.ndarray,
                                   bias_range: float = 0.5,
                                   noise_sigma: float = 0.1) -> np.ndarray:
    bias = np.random.uniform(-bias_range, bias_range, size=force.shape)
    noise = np.random.normal(0, noise_sigma, size=force.shape)
    return force + bias + noise
```

**2. 力控带宽限制模拟**
```python
class ForceControlBandwidthLimiter:
    """模拟真实力控的低通特性"""
    
    def __init__(self, cutoff_freq: float = 20.0, dt: float = 0.001):
        # 一阶低通滤波器：tau = 1/(2*pi*fc)
        self.tau = 1.0 / (2 * np.pi * cutoff_freq)
        self.dt = dt
        self.prev_output = 0.0
        
    def filter(self, cmd: float) -> float:
        alpha = self.dt / (self.tau + self.dt)
        self.prev_output = alpha * cmd + (1 - alpha) * self.prev_output
        return self.prev_output
```

### 7.3 Shadow / Allegro / LEAP Hand 迁移经验

不同灵巧手的 Sim-to-Real 迁移有其独特的工程考量：

#### Shadow Hand
- **DOF**: 20 (4 × 5finger，含手腕)
- **驱动**: 气压肌腱 (Pneumatic tendons) + Smart Motor
- **关键挑战**:
  - 肌腱的**迟滞非线性**极强，仿真中难以精确建模
  - 建议：使用数据驱动的肌腱模型（lookup table + 神经网络补偿）
  - 每个关节的实际扭矩-位置关系需单独标定
- **传感器**: 每个 Smart Motor 内置 tendon 力传感器（约 30mN 分辨率）
- **Sim-to-Real 建议**: 
  - 随机化 tendon 刚度 ±30%
  - 随机化关节摩擦损失 0.01-0.1 Nm
  - 控制频率 ≥ 500Hz（ tendon 动力学快）

#### Allegro Hand
- **DOF**: 16 (4 fingers × 4 joints)
- **驱动**: 直流电机 + 谐波减速器
- **关键挑战**:
  - 指尖**六轴力矩传感器**（1kHz）提供了丰富的力反馈
  - 但传感器存在明显的**串扰**和**温度漂移**
- **Sim-to-Real 建议**:
  - 重点做触觉传感器的域随机化（见 7.1）
  - 指尖摩擦系数精确标定（Allegro 指尖材质变化大）
  - 关节阻尼通常比理论值高 20-40%
  - **DexSim2Real 实验平台选用 Allegro + XELA，可直接参考其参数**

#### LEAP Hand
- **DOF**: 16 (低成本的 3D 打印灵巧手)
- **驱动**: 舵机 / 小型直流电机
- **关键挑战**:
  - 机械间隙大，回程误差显著
  - 电机力矩控制精度有限
  - 低成本传感器的噪声水平高
- **Sim-to-Real 建议**:
  - 随机化关节间隙 (backlash) 0.5°-2°
  - 随机化执行器死区 (deadzone) ±5% 满量程
  - 力控策略需更保守（低增益）
  - 建议采用位置控制 + 力阈值监控的混合策略

**通用经验**：
```python
# 不同灵巧手的推荐随机化范围（基于社区经验）
HAND_RANDOMIZATION_RANGES = {
    'shadow_hand': {
        'tendon_stiffness': (0.7, 1.3),
        'joint_frictionloss': (0.01, 0.1),
        'motor_bandwidth': (50, 200),  # Hz
        'sensor_noise': 0.05,
    },
    'allegro_hand': {
        'fingertip_friction': (0.5, 1.5),
        'joint_damping': (0.5, 1.5),
        'tactile_noise': 0.1,
        'tactile_crosstalk': (0.0, 0.2),
    },
    'leap_hand': {
        'backlash': (0.5, 2.0),  # degrees
        'deadzone': (0.02, 0.08),  # fraction
        'gear_efficiency': (0.7, 0.95),
        'motor_noise': 0.1,
    }
}
```

---

## 8. 调试与诊断流程

### 8.1 常见失败模式与诊断方法

| 失败模式 | 真实表现 | 根因诊断 | 解决方案 |
|---------|---------|---------|---------|
| **物体滑落** | 抓取后物体掉落 | 仿真摩擦过高 / 接触法向力不足 | 降低仿真摩擦，增加力控增益，检查指尖包覆材质 |
| **抖动/振荡** | 手臂/手指高频振动 | 力控增益过高 / 延迟未补偿 / 接触刚度失配 | 降低增益，实施延迟补偿，调整 solref |
| **到达错误位置** | 无法对准物体 | 视觉域差距 / 相机标定误差 / 手眼标定误差 | 视觉域随机化，重新标定相机，检查 extrinsic |
| **过度抓取** | 物体变形/压碎 | 力控阈值过低 / 无触觉反馈 | 增加力上限，引入触觉阈值，训练力调节策略 |
| **策略保守** | 动作幅度小，任务超时 | 仿真随机化过度 / 奖励塑形不当 | 收紧随机化范围，检查奖励函数，使用 CDR |
| **接触爆炸** | 仿真中物体飞射 | 接触参数不当 / 时间步过大 / 惯量错误 | 减小 timestep，检查质量属性，调整 solimp |
| **真实中发散** | 策略导致危险动作 | 观测分布外 / 执行器饱和 / 未建模约束 | 添加观测 clipping，检查 ctrl_range，增加安全层 |

**诊断工具链**：
```python
def diagnose_sim2real_failure(sim_data: dict, real_data: dict):
    """
    Sim-to-Real 失败诊断脚本
    """
    issues = []
    
    # 1. 检查观测分布偏移
    sim_obs_mean = np.mean(sim_data['observations'], axis=0)
    real_obs_mean = np.mean(real_data['observations'], axis=0)
    obs_drift = np.abs(sim_obs_mean - real_obs_mean)
    if np.max(obs_drift) > 3.0:
        issues.append(f"观测分布严重偏移，最大偏差: {np.max(obs_drift):.2f}")
    
    # 2. 检查动作饱和度
    sim_action_std = np.std(sim_data['actions'])
    real_action_std = np.std(real_data['actions'])
    if real_action_std > 1.5 * sim_action_std:
        issues.append("真实动作方差显著大于仿真，可能随机化不足或存在未建模动态")
    
    # 3. 检查力/力矩异常
    if 'contact_forces' in real_data:
        max_force = np.max(real_data['contact_forces'])
        if max_force > 50:  # N
            issues.append(f"接触力过大 ({max_force:.1f}N)，检查摩擦和力控增益")
    
    # 4. 检查延迟
    if 'command_timestamps' in real_data and 'state_timestamps' in real_data:
        delay = np.mean(real_data['state_timestamps'] - real_data['command_timestamps'])
        if delay > 0.05:  # 50ms
            issues.append(f"系统延迟过高 ({delay*1000:.0f}ms)，需要延迟补偿")
    
    return issues
```

### 8.2 渐进式调试流程

推荐的 Sim-to-Real 调试采用**由内向外、由简到繁**的渐进策略：

```
Phase 1: 纯仿真验证
├── 策略在默认参数仿真中达到目标性能
├── 在极端随机化配置下仍能完成任务
└── 记录观测/动作/奖励的分布作为基准

Phase 2: 硬件在环 (HIL)
├── 将真实关节状态输入仿真策略（state-based policy）
├── 验证状态估计和通信链路
├── 在无接触任务中测试（如 reach/tracking）
└── 诊断纯控制层面的问题

Phase 3: 开环策略测试
├── 在仿真中记录一条成功轨迹
├── 在真实硬件上**开环复放**（不依赖传感器反馈）
├── 比较仿真与真实的轨迹差异
└── 识别动力学/接触模型的不匹配

Phase 4: 闭环策略 + 简单接触
├── 在真实环境中测试简单的抓取/放置
├── 使用力阈值作为安全保护
├── 逐步增加任务复杂度
└── 收集失败数据进行针对性随机化

Phase 5: 完整任务 + 安全监控
├── 部署完整策略
├── 启用多层安全监控（力阈值、关节限位、紧急停止）
├── 记录真实数据用于下一轮迭代
└── 基于真实数据优化仿真参数（如 Phys2Real 流程）
```

---

## 9. 从仿真到真实的检查清单（Checklist）

### 仿真环境准备

- [ ] MuJoCo/Isaac Sim 模型与 URDF 质量属性一致（质量、惯量、质心）
- [ ] 摩擦系数已通过斜坡实验测量，并设置为基础值
- [ ] 关节限制与真实硬件一致（包括软限位）
- [ ] 接触参数经过调优（无穿透、无抖动、无滑动异常）
- [ ] 控制频率 ≥ 真实硬件控制频率（建议 ≥ 500Hz）
- [ ] 时间步长足够小（建议 ≤ 0.002s，灵巧操作 ≤ 0.001s）
- [ ] 相机内参/外参与真实相机标定结果一致
- [ ] 光照配置至少包含方向光 + 环境光

### 域随机化配置

- [ ] 摩擦系数随机化范围包含真实测量值（建议 ±30% 或 0.3-1.2）
- [ ] 质量缩放范围合理（建议 0.8x-1.5x，避免极端不物理配置）
- [ ] 关节阻尼/摩擦损失已基于系统辨识设置基础值
- [ ] 视觉随机化包含：亮度、对比度、颜色、噪声、相机位姿
- [ ] 触觉传感器噪声/串扰/零漂已建模
- [ ] 执行器延迟已在训练中被考虑（延迟增强或状态预测）
- [ ] 如果使用 CDR，课程进度与总训练步数匹配

### 策略训练验证

- [ ] 策略在默认参数仿真中成功率 > 90%
- [ ] 策略在宽范围随机化下成功率 > 70%
- [ ] 策略对观测噪声鲁棒（加 5% 高斯噪声仍能工作）
- [ ] 策略动作输出平滑（无高频抖动）
- [ ] 策略未利用仿真 artifact（如穿透、求解器误差）
- [ ] 仿真→真实性能差距 < 20%（如差距过大，检查上述项）

### 真实硬件准备

- [ ] 所有关节已校准（零点、方向、限位）
- [ ] 力/力矩传感器已标定（零漂、增益、串扰矩阵）
- [ ] 相机已标定（内参 + 手眼外参）
- [ ] 系统延迟已测量并记录
- [ ] 紧急停止按钮和安全边界已配置
- [ ]  fingertip 材质/包覆与仿真中一致
- [ ] 工作台表面材质与仿真中一致

### 部署前最终检查

- [ ] 首次运行使用保守参数（降低速度/力上限）
- [ ] 有人员在旁监控，可随时触发急停
- [ ] 已准备开环测试轨迹用于基准对比
- [ ] 已配置数据记录（观测、动作、传感器、时间戳）
- [ ] 已准备诊断脚本分析失败原因
- [ ] 已设定明确的回退方案（如策略失败时的安全姿势）

---

## 10. 参考文献

1. **DexSim2Real**: Zeng, Z., et al. (2026). *Foundation Model-Guided Sim-to-Real Transfer for Generalizable Dexterous Manipulation*. arXiv:2605.05241.
2. **Phys2Real**: Wang, M., et al. (2026). *Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation*. ICRA 2026. arXiv:2510.11689.
3. **Video2Sim2Real**: *Full-Stack Autonomous Dexterous Skill Acquisition from a Single Human Video* (2026). arXiv:2606.08828.
4. **Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning** (2026). arXiv:2606.11767.
5. **Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation** (2026). arXiv:2601.02778.
6. **DrEureka**: Ma, Y.J., et al. (2024). *DrEureka: Language Model Guided Sim-To-Real Transfer*. RSS 2024.
7. **DeXtreme**: Handa, A., et al. (2023). *DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality*. ICRA 2023.
8. **RialTo**: Taylor, I., et al. (2024). *Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation*. arXiv:2403.03949.
9. **Domain Randomization Survey**: Tobin, J., et al. (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. IROS 2017.
10. **MuJoCo Contact Modeling**: Todorov, E. (2014). *Convex and Analytically-Invertible Dynamics with Contacts and Constraints: Theory and Implementation in MuJoCo*. ICML 2014.
11. **ASRoBALLET** (2026). *Closing the Sim2Real Gap via Friction-Aware Reinforcement Learning for Underactuated Spherical Dynamics*. arXiv:2604.24916.
12. **Sim-to-Real Practitioner's Guide 2026**: Robotics Center AI Blog. *Sim-to-Real Transfer: A Practitioner's Guide for 2026*.

---

> **免责声明**: 本文档中的参数范围基于 2026 年公开研究成果与社区最佳实践。实际应用中，参数需根据具体硬件平台和任务场景进行调优。物理机器人实验存在安全风险，请在有安全措施的环境下进行。
