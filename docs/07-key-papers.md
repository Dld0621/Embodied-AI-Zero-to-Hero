# 关键论文导读

> 37 篇 IK Retargeting 与灵巧手控制领域核心论文，按主题分类，每篇附"为什么读"、"核心收获"和在线链接。

---

## 人手追踪与姿态估计

### 1. MediaPipe Hands: On-device Real-time Hand Tracking

- **论文**: *MediaPipe Hands: On-device Real-time Hand Tracking* (Google, 2019)
- **代码**: [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)

**为什么读**：最广泛使用的实时手部 landmarks 检测工具，21 点 3D 坐标，CPU 实时运行。

**核心收获**：
- 21 点模型已成为人手追踪的事实标准
- 单目 RGB 即可估计 3D 姿态（虽然深度有误差）
- 在机器人遥操作中，MediaPipe 的 3D 估计足以驱动关节角映射

---

### 2. InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation

- **论文**: *InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image* (ECCV 2020)
- **arXiv**: [2008.09309](https://arxiv.org/abs/2008.09309)

**为什么读**：双手交互的 3D 姿态估计数据集和基准方法，解决双手遮挡问题。

**核心收获**：
- 双手同时追踪需要处理严重的自遮挡
- 单目 3D 估计的精度可通过多帧联合估计提升
- 可通过前 30 帧建立 source basis，实现相机坐标系到场景坐标系的映射

---

### 3. HaMeR: Reconstructing Hands in 3D with Transformers

- **论文**: *HaMeR: Reconstructing Hands in 3D with Transformers* (CVPR 2024)
- **代码**: [geopavlakos/hamer](https://github.com/geopavlakos/hamer)

**为什么读**：使用 Vision Transformer 从单张图像重建 3D 手部 mesh，精度远超 MediaPipe。

**核心收获**：
- 输出 MANO 参数化模型（pose + shape），可直接用于 retargeting
- 对遮挡和截断更鲁棒
- 可作为 MediaPipe 的高精度替代方案

---

## 运动重映射（Motion Retargeting）经典工作

### 4. Motion Retargeting to Characters

- **论文**: *Motion Retargeting to Characters* (Gleicher, SIGGRAPH 1998)

**为什么读**：运动重映射的奠基之作，定义了 retargeting 的核心问题。

**核心思想**：
- 保留"语义"而非复制"数值"
- 关键约束：关节限位、自碰撞、平衡
- 时空优化框架（spacetime constraints）

---

### 5. Unsupervised Motion Retargeting

- **论文**: *Unsupervised Motion Retargeting* (Aberman et al., CVPR 2020 Oral)
- **代码**: [arielai/unsup_moving_in_depth](https://github.com/arielai/unsup_moving_in_depth)

**为什么读**：首次提出无监督跨角色运动重映射，无需配对数据。

**核心收获**：
- 使用 CycleGAN 思想：人 A 的运动 → 人 B → 回人 A 应一致
- 证明了无监督学习在 retargeting 中的可行性
- 对机器人领域启示：可用不同人手的运动互相学习

---

### 6. Skeleton-Aware Networks for Deep Motion Retargeting

- **论文**: *Skeleton-Aware Networks for Deep Motion Retargeting* (Aberman et al., TOG/SIGGRAPH 2020)
- **代码**: [arielai/skeleton_aware](https://github.com/arielai/skeleton_aware)

**为什么读**：将图神经网络引入 motion retargeting，处理任意骨骼结构。

**核心收获**：
- 骨骼结构表示为图，节点是关节，边是骨骼
- 图卷积可泛化到任意拓扑结构
- 对机器人启示：同一网络可处理不同自由度的机器人手

---

### 7. Learning Character-Agnostic Motion for Motion Retargeting

- **论文**: *Learning Character-Agnostic Motion for Motion Retargeting* (2022)

**为什么读**：提出了 character-agnostic 的运动表示，分离运动内容与角色结构。

---

## 人手到机器人手重映射

### 8. AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System

- **论文**: *AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System* (RSS 2023)
- **项目**: [twitter.com/anyteleop](https://twitter.com/anyteleop)

**为什么读**：2023 年最具影响力的遥操作系统之一，展示了完整的视觉→机器人重映射 pipeline。

**核心架构**：
- 单手 RGB 即可驱动双臂 + 双手灵巧手
- Retargeting 模块：MediaPipe → landmarks → 机器人关节
- 支持多种机器人平台（Franka + Allegro / Shadow Hand）

**核心收获**：
- Rule-based 映射在工程上足够稳定
- 实时性是关键（延迟 < 50ms）
- 手指间耦合需要特殊处理

---

### 9. Open-TeleVision: Teleoperation with Immersive Active Visual Feedback

- **论文**: *Open-TeleVision: Teleoperation with Immersive Active Visual Feedback* (2024)
- **代码**: [OpenTeleVision/TeleVision](https://github.com/OpenTeleVision/TeleVision)

**为什么读**：现代遥操作系统的标杆，包含手部 retargeting 作为核心子模块。

**核心架构**：
- Apple Vision Pro / VR 头显提供沉浸式反馈
- 双手 + 双臂协同控制
- 基于 retargeting 的实时遥操作

---

### 10. Vision-Based Teleoperation of Shadow Dexterous Hand

- **论文**: *Vision-Based Teleoperation of Shadow Dexterous Hand* (ICRA 2020)

**为什么读**：将视觉手部追踪直接用于 20-DOF Shadow Hand 遥操作的早期工作。

**核心收获**：
- 直接角度映射是最简单的基线
- 需要处理人手 20-DOF 与 Shadow Hand 20-DOF 的自由度差异
- 时域滤波对减少抖动至关重要

---

### 11. LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning

- **论文**: *LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning* (CoRL 2023)
- **代码**: [leap-hand](https://github.com/leap-hand)

**为什么读**：低成本开源灵巧手的代表，其 retargeting 方案值得学习。

**核心收获**：
- 16-DOF 设计，成本和功能性平衡
- 提供了完整的 retargeting 和仿真 pipeline
- MuJoCo 模型开源，可直接复用

---

## 灵巧手操作与抓取

### 12. Learning Dexterous In-Hand Manipulation

- **论文**: *Learning Dexterous In-Hand Manipulation* (OpenAI, ICRA 2018)

**为什么读**：展示了高自由度灵巧手（20 DOF Shadow Hand）在精细操作上的潜力。

**核心收获**：
- 20 DOF 的 Shadow Hand 可以实现人手的绝大多数手势
- 强化学习（PPO）是训练灵巧操作策略的有效方法
- Sim-to-Real 转移是部署的关键挑战

---

### 13. GraspTTA: Test-Time Adaptation for Category-Level Object Manipulation

- **论文**: *GraspTTA: Test-Time Adaptation for Category-Level Object Manipulation* (CVPR 2021)

**为什么读**：将人手抓取姿态迁移到机器人抓取，与 retargeting 密切相关。

---

### 14. ContactGrasp: Functional Multi-finger Grasp Synthesis from Contact

- **论文**: *ContactGrasp: Functional Multi-finger Grasp Synthesis from Contact* (ICRA 2019)

**为什么读**：从接触点合成多指抓取，评估指标可直接用于 retargeting 质量衡量。

---

## 基于学习的重映射

### 15. GANerated Hands for Real-time 3D Hand Tracking

- **论文**: *GANerated Hands for Real-time 3D Hand Tracking* (CVPR 2018)

**为什么读**：使用生成模型改善手部追踪，以及从人手到参数化手模型的映射。

---

### 16. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

- **论文**: *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (Columbia / Stanford, 2023)
- **arXiv**: [2303.04137](https://arxiv.org/abs/2303.04137)
- **代码**: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)

**为什么读**：虽然主要用于机器人策略学习，但其动作生成思想可应用于 retargeting。

**核心收获**：
- 扩散模型可表示多峰动作分布
- 去噪过程可生成平滑的动作序列
- 可用于从人手姿态生成机器人动作分布

---

### 17. Neural Network-based Hand Retargeting for Teleoperation

- **相关论文**: *Real-time Hand Tracking under Occlusion from an Egocentric RGB Camera* (ICCV 2021)
- **相关工作**: 使用神经网络学习人手到机器人手的端到端映射

**为什么读**：探索了学习-based 方法在 retargeting 中的实际应用。

---

## 评估与基准

### 18. A Review of Robot Learning for Manipulation

- **论文**: *A Review of Robot Learning for Manipulation: Challenges, Representations, and Algorithms* (J. Kober et al., 2013)

**为什么读**：机器人操作学习的经典综述，涵盖运动表示和策略学习。

---

### 19. DexYCB: A Benchmark for Capturing Hand Grasping of Objects

- **论文**: *DexYCB: A Benchmark for Capturing Hand Grasping of Objects* (CVPR 2021)
- **项目**: [dex-ycb.github.io](https://dex-ycb.github.io/)

**为什么读**：包含人手抓取物体的 RGB-D 视频 + 3D 标注，可用于评估 retargeting 的抓取成功率。

---

### 20. FreiHAND: A Dataset for Markerless Hand Capture

- **论文**: *FreiHAND: A Dataset for Markerless Hand Capture* (ICCV 2019)
- **项目**: [lmb.informatik.uni-freiburg.de/projects/freihand](https://lmb.informatik.uni-freiburg.de/projects/freihand/)

**为什么读**：单目 RGB 手部姿态估计基准数据集。

---

## 推荐阅读路线图

```
入门路线：
MediaPipe Hands → Rule-based Retargeting → O10/Shadow Hand 控制
        ↓
InterHand2.6M → 双手追踪 → 双手遥操作
        ↓
AnyTeleop/Open-TeleVision → 完整系统架构

进阶路线：
Gleicher 1998 → Aberman CVPR 2020 → Skeleton-Aware GNN
        ↓
Diffusion Policy → 生成模型 retargeting
        ↓
Sim-to-Real → 真实机器人部署
```

---

## 2026 年最新进展（后几何时代）

2026 年灵巧手重定向已进入**"后几何时代"**——单纯追求 fingertip position error 最小化的方法正在被淘汰，取而代之的是综合考虑 human intent、object interaction、robot morphology 和 physics 的系统性框架。

### 21. TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation

- **论文**: *TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation* (2026)
- **arXiv**: [2606.16272](https://arxiv.org/abs/2606.16272)
- **项目主页**: [toporetarget2026.github.io](https://toporetarget2026.github.io/TopoRetarget/)
- **作者**: Jielin Wu, Shenzhe Yao, Guanqi He 等 (清华大学 IIIS, Hang Zhao 组)

**为什么读**：从"姿态复现"转向"交互保持"的代表作，解决重定向后手-物接触关系被破坏的问题。

**核心收获**：
- 关注的不是 `p_tip^r ≈ p_tip^h`，而是保持哪根手指接触物体、接触之间的相对关系
- 代表方向转变：pose-preserving → interaction-preserving
- 与 morphology calibration 可直接组合：先校准几何，再保持交互

---

### 22. AnyDexRT: Calibration-Free Dexterous Hand Retargeting with Few-Shot Human Guidance

- **论文**: *AnyDexRT: Calibration-Free Dexterous Hand Retargeting with Few-Shot Human Guidance* (2026)
- **arXiv**: [2607.08341](https://arxiv.org/abs/2607.08341)
- **作者**: Chenxi Wang, Ying Feng, Cewu Lu 等 (上海交大 + Noematrix)

**为什么读**：免标定灵巧重定向的标杆，解决新操作者/新机器人需重新标定的问题。

**核心收获**：
- `calibration-free + few-shot human guidance`
- 与显式 finger-wise scaling 方法问题层面高度相关，但方法不同（学习式 vs 参数化）
- 说明"标定"仍是核心痛点，但解决方案正在从手工规则转向数据驱动

---

### 23. SPIDER: Scalable Physics-Informed Dexterous Retargeting

- **论文**: *SPIDER: Scalable Physics-Informed Dexterous Retargeting* (2025)
- **arXiv**: [2511.09484](https://arxiv.org/abs/2511.09484)
- **项目主页**: [jc-bao.github.io/spider-project](https://jc-bao.github.io/spider-project/)
- **代码**: [github.com/facebookresearch/spider](https://github.com/facebookresearch/spider)
- **文档**: [facebookresearch.github.io/spider](https://facebookresearch.github.io/spider/)
- **作者**: Chaoyi Pan, Changhao Wang, Haozhi Qi, Jitendra Malik 等 (Meta FAIR)

**为什么读**：物理可行性 ≠ 运动学可行性的经典论证，解决手指穿物、接触不稳定问题。

**核心收获**：
- 大规模人手数据不能直接作为机器人参考轨迹
- 物理约束（碰撞、力、接触）必须在重定向阶段就纳入
- 与 DexMV 等纯几何优化方法形成直接对比

---

### 24. GraspGraphNet: Graph-Structured Multi-Embodiment Dexterous Grasp Generation

- **论文**: *GraspGraphNet: Graph-Structured Multi-Embodiment Dexterous Grasp Generation* (2026)
- **作者**: Yeonseo Lee, Taeyeop Lee, Hyosup Shin 等

**为什么读**：跨形态灵巧手统一表示的最前沿，图结构能处理不同拓扑（三指 vs 五指）。

**核心收获**：
- 将机器人关节/连杆/接触表示为图，共享抓取模型
- 比 finger-wise scale 更普适：能处理不同关节连接关系、actuator coupling
- 直接在可执行空间生成，绕过 IK 和 Retargeting

---

### 25. UniDexTok: A Unified Dexterous Hand Tokenizer from Real Data

- **论文**: *UniDexTok: A Unified Dexterous Hand Tokenizer from Real Data* (2026)
- **作者**: Dong Fang, Youjun Wu, Yuanxin Zhong 等

**为什么读**：统一灵巧手表示的尝试，提出 22-DoF 语义接口跨越不同机器人手。

**核心收获**：
- UDHM (Unified Dexterous Hand Model)：共享 22-DoF 语义空间
- 无 Retargeting 的 tokenizer：人类和机器人手状态映射到统一表示
- 未来愿景：`Human canonical → Shared latent → Robot-specific decoder`

---

### 26. DexGrasp-Zero: Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Grasping

- **论文**: *DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping* (2026)
- **作者**: Yuliang Wu, Yanhan Lin, Wei-Shi Zheng 等

**为什么读**：直接把 morphology alignment 写入标题，代表领域对形态差距的重视。

**核心收获**：
- `morphology alignment` 从"误差因素"升级为"策略输入或对齐变量"
- train on some hands → zero-shot transfer to unseen hands
- 解决的不是人手差异，而是 robot embodiment A → robot embodiment B

---

### 27. One-Policy-Fits-All: Geometry-Aware Action Latents for Cross-Embodiment Manipulation

- **论文**: *One-Policy-Fits-All: Geometry-Aware Action Latents for Cross-Embodiment Manipulation* (2026)
- **作者**: Juncheng Mu, Sizhe Yang, Huazhe Xu, Jiangmiao Pang 等

**为什么读**：跨形态机器人动作空间不一致问题的系统解决方案。

**核心收获**：
- 几何感知动作潜空间：不同机器人动作编码为共享潜变量
- 统一 latent retargeting 解码器，无需形态特定调优
- 方向转变：直接映射关节 → 共享几何/交互潜空间

---

### 28. VTAP Gripper: Synergizing Fingertip Sensing and Visuo-Tactile Active Palm

- **论文**: *VTAP Gripper: Synergizing Fingertip Sensing and a Visuo-Tactile Active Palm for Dexterous In-Hand Manipulation* (2026)
- **arXiv**: [2607.15448](https://arxiv.org/abs/2607.15448)
- **作者**: Yuhao Zhou, Sheeraz Athar, Yunzhu Li 等

**为什么读**：阶段化、手势条件化重定向的实例，跨越人手与异构三指夹爪的形态差距。

**核心收获**：
- staged, gesture-conditioned retargeting
- 不再假设"一个统一映射适用于所有动作"
- pinch / enveloping grasp / in-hand manipulation 各用不同映射

---

### 29. Smooth Operator: Real-Time Sampling-Based Kinematic Hand Retargeting

- **论文**: *Smooth Operator: A Real-Time Sampling-Based Algorithm for Kinematic Hand Retargeting* (2026)
- **arXiv**: [2607.07491](https://arxiv.org/abs/2607.07491)
- **作者**: Robert Jomar Malate, Erik Bauer, Benedek Forrai 等 (ETH Zurich)

**为什么读**：用采样替代梯度优化，解决实时性和稳定性问题。

**核心收获**：
- 梯度法易陷局部最优、初值敏感、快速动作下稳定性差
- 采样方法避免抖动，提升遥操作数据质量
- 可与 morphology-aware input 组合：你优化输入，它优化求解器

---

### 30. ObjRetarget: Object-Aware Motion Retargeting Framework

- **论文**: *ObjRetarget: An Object-Aware Motion Retargeting Framework with Anthropomorphic Arm Constraints and Polyhedral Hand Modeling* (2026)
- **arXiv**: [2607.03828](https://arxiv.org/abs/2607.03828)
- **作者**: Yuanchuan Lai, Qing Gao, Zhaojie Ju 等

**为什么读**：把物体几何纳入重定向的代表作，从仅匹配人手到同时考虑手-物关系。

**核心收获**：
- 目标函数：`L_human-robot + L_hand-object + L_arm_feasibility`
- 单独看手姿态无法保证任务有效
- 物体感知是 morphology calibration 的自然扩展

---

### 31. Kilohertz-Safe: A Scalable Framework for Constrained Dexterous Retargeting

- **论文**: *Kilohertz-Safe: A Scalable Framework for Constrained Dexterous Retargeting* (2026)
- **arXiv**: [2603.29213](https://arxiv.org/abs/2603.29213)
- **作者**: Wuji Hand 团队

**为什么读**：学习如何把运动学限制、碰撞避免和计算预算放进同一重定向框架，并分别检查运行时间与约束满足率。

**核心收获**：
- 将运动学限制和碰撞避免通过系统性线性化统一到计算框架中
- 使用控制屏障函数（CBF）表达所研究的安全条件，其适用性依赖建模、约束与执行假设
- 官方摘要报告平均延迟 9.05 ms，超过 95% 的重定向帧满足论文安全判据
- 阅读时分别核对关节限位、执行器约束、自碰撞、连续性和端到端时延的实验条件

以上运行数值于 2026-11-14 按[官方摘要](https://arxiv.org/abs/2603.29213)核对。

---

### 32. GeoRT: Geometric Retargeting — A Principled, Ultrafast Neural Hand Retargeting Algorithm

- **论文**: *Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm* (IROS 2025)
- **arXiv**: [2503.07541](https://arxiv.org/abs/2503.07541)
- **作者**: Zhao-Heng Yin, Changhao Wang, Luis Pineda, Pieter Abbeel, Mustafa Mukadam 等 (UC Berkeley + Meta)

**为什么读**：从"每帧在线迭代求解"向"学习后的快速前向推理"转变的代表作。

**核心收获**：
- 无监督训练，无需人工标注手部配对
- 1KHz 速度将人手关键点转换为机器人手关键点
- 新颖的几何目标函数：motion fidelity、C-space coverage、high flatness、pinch correspondence
- morphology-aware normalization 可作为 GeoRT 类网络的输入前处理或条件变量

---

### 33. DexTwist: Dexterous Hand Retargeting for Twist Motion via Mixed Reality-based Teleoperation

- **论文**: *DexTwist: Dexterous Hand Retargeting for Twist Motion via Mixed Reality-based Teleoperation* (2026)
- **arXiv**: [2605.12182](https://arxiv.org/abs/2605.12182)

**为什么读**：针对扭转操作（twist motion）的功能性重定向，填补了旋拧类操作的空白。

**核心收获**：
- 检测三指捏合（tripod pinch），估计操作者意图的螺旋轴和扭转量
- 实时残差关节空间优化：跟踪旋转进度同时正则化机器人三指几何
- 虚拟物体目标函数：turning angle + screw axis consistency + fingertip closure + tripod stability
- 功能性重定向的典型范例：不追求姿态复制，而追求功能等效

---

### 34. Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation

- **论文**: *Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation* (RA-P 2025)
- **arXiv**: [2506.09384](https://arxiv.org/abs/2506.09384)
- **作者**: Chendong Xin, Mingrui Yu, Yongpeng Jiang, Zhefeng Zhang, Xiang Li

**为什么读**：系统分析重定向到底应该优化什么，是目标函数设计的实验性指南。

**核心收获**：
- 指尖相对位置、指尖朝向、全局手姿态各有不同重要性
- 实验消融揭示各目标项对实际操作成功率的影响
- 后续实验不能只对比 joint error，还应比较 fingertip error、finger-vector error、pinch-distance error、hand-shape error、task success
- 使用 SLSQP（NLopt 库）实时优化完整目标函数

---

### 35. DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation

- **论文**: *DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation* (2025)
- **arXiv**: [2505.24853](https://arxiv.org/abs/2505.24853)
- **项目主页**: [project-dexmachina.github.io](https://project-dexmachina.github.io/)
- **代码**: [github.com/MandiZhao/dexmachina](https://github.com/MandiZhao/dexmachina)
- **文档**: [mandizhao.github.io/dexmachina-docs](https://mandizhao.github.io/dexmachina-docs/)
- **作者**: Zhao Mandi 等 (Stanford + NVIDIA)

**为什么读**："Functional retargeting" 关键词的代表作，关注功能等效而非关节形状复制。

**核心收获**：
- 课程学习算法：虚拟物体控制器强度逐渐衰减，策略逐步接管
- 从一条人类演示重定向到多种灵巧手 embodiment
- 双手铰接物体长时程操作
- 人用三根手指支撑物体，机器人可能用两根手指和掌面——只要功能一致即可

---

### 36. DexFlow: A Unified Approach for Dexterous Hand Pose Retargeting and Interaction

- **论文**: *DexFlow: A Unified Approach for Dexterous Hand Pose Retargeting and Interaction* (2025)
- **arXiv**: [2505.01083](https://arxiv.org/abs/2505.01083)
- **作者**: Xiaoyi Lin, Kunpeng Yao, Lixin Xu 等

**为什么读**：统一的灵巧手姿态重定向与交互方法，生成多模态抓取序列。

**核心收获**：
- 改进的优化 pipeline 将 MANO 手部动捕数据重定向到 ShadowHand/Allegro
- 为 50 个 YCB 物体生成 292k 帧轨迹（右手），覆盖稳定抓取等场景
- 多模态抓取序列：pose + joint angles series data
- 统一框架处理 retargeting 和 interaction 两个子问题

---

### 37. EquiDexFlow: Contact-Grounded SE(3)-Equivariant Dexterous Grasp Generative Flows

- **论文**: *EquiDexFlow: Contact-Grounded SE(3)-Equivariant Dexterous Grasp Generative Flows* (2026)
- **项目主页**: [equidexflow.github.io](https://equidexflow.github.io/)

**为什么读**：将接触、法向和力学量联合预测的生成式方法，接触不再只是后处理验证。

**核心收获**：
- 联合预测：wrist pose + joint angles + fingertip contacts + surface normals + contact-related physical quantities
- 在 8,100 个力闭合抓取、81 个物体、16-DoF Allegro Hand 上训练
- 零摩擦违反，最佳综合得分，最低力矩残差
- 接触解码后通过 per-finger IK 重定向到 16-DoF LEAP Hand
- 代表趋势：`z → (q, T_wrist, C, N, F)` — 接触成为生成目标的一部分

---

## 推荐阅读路线图（2026 更新版）

```
入门路线：
MediaPipe Hands → Rule-based Retargeting → O10/Shadow Hand 控制
        ↓
InterHand2.6M → 双手追踪 → 双手遥操作
        ↓
AnyTeleop/Open-TeleVision → 完整系统架构

进阶路线：
Gleicher 1998 → Aberman CVPR 2020 → Skeleton-Aware GNN
        ↓
Diffusion Policy → 生成模型 retargeting
        ↓
TopoRetarget / SPIDER → 交互保持 + 物理感知
        ↓
AnyDexRT / DexGrasp-Zero → 免标定 + 形态对齐
        ↓
GraspGraphNet / UniDexTok → 跨形态统一表示
        ↓
Sim-to-Real → 真实机器人部署
```

---

## 论文资源汇总

| 论文 | 会议 | 方向 | 代码 | 难度 |
|------|------|------|------|------|
| MediaPipe Hands | Google 2019 | 人手追踪 | [GitHub](https://github.com/google-ai-edge/mediapipe) | ★☆☆ |
| InterHand2.6M | ECCV 2020 | 双手 3D 姿态 | [GitHub](https://github.com/facebookresearch/InterHand2.6M) | ★★☆ |
| HaMeR | CVPR 2024 | 手部 mesh 重建 | [GitHub](https://github.com/geopavlakos/hamer) | ★★☆ |
| Gleicher 1998 | SIGGRAPH | 重映射奠基 | — | ★★☆ |
| Aberman CVPR 2020 | CVPR Oral | 无监督重映射 | [GitHub](https://github.com/arielai/unsup_moving_in_depth) | ★★★ |
| Skeleton-Aware | TOG 2020 | GNN 重映射 | [GitHub](https://github.com/arielai/skeleton_aware) | ★★★ |
| AnyTeleop | RSS 2023 | 遥操作系统 | — | ★★★ |
| Open-TeleVision | 2024 | 遥操作系统 | [GitHub](https://github.com/OpenTeleVision/TeleVision) | ★★★ |
| LEAP Hand | CoRL 2023 | 灵巧手设计 | [GitHub](https://github.com/leap-hand) | ★★☆ |
| Diffusion Policy | 2023 | 动作生成 | [GitHub](https://github.com/real-stanford/diffusion_policy) | ★★★ |
| DexYCB | CVPR 2021 | 抓取基准 | [项目页](https://dex-ycb.github.io/) | ★★☆ |
| **TopoRetarget** | 2026 | 交互保持重定向 | [项目页](https://toporetarget2026.github.io/TopoRetarget/) | ★★★ |
| **AnyDexRT** | 2026 | 免标定灵巧重定向 | — | ★★★ |
| **SPIDER** | 2025 | 物理感知重定向 | [GitHub](https://github.com/facebookresearch/spider) | ★★★ |
| **GraspGraphNet** | 2026 | 图结构跨形态抓取 | — | ★★★ |
| **UniDexTok** | 2026 | 统一灵巧手 Tokenizer | — | ★★★ |
| **DexGrasp-Zero** | 2026 | 零样本跨手抓取 | — | ★★★ |
| **One-Policy-Fits-All** | 2026 | 几何感知跨形态策略 | — | ★★★ |
| **VTAP Gripper** | 2026 | 手势条件化重定向 | — | ★★★ |
| **Smooth Operator** | 2026 | 采样式实时重定向 | — | ★★★ |
| **ObjRetarget** | 2026 | 物体感知重定向 | — | ★★★ |
| **Kilohertz-Safe** | 2026 | 高频安全重定向 | — | ★★★ |
| **GeoRT** | IROS 2025 | 超快神经重定向 | — | ★★★ |
| **DexTwist** | 2026 | 扭转功能性重定向 | — | ★★★ |
| **Key Objectives** | RA-P 2025 | 目标函数分析 | — | ★★★ |
| **DexMachina** | 2025 | 功能性重定向 | [GitHub](https://github.com/MandiZhao/dexmachina) | ★★★ |
| **DexFlow** | 2025 | 统一重定向+交互 | — | ★★★ |
| **EquiDexFlow** | 2026 | 接触感知生成流 | [项目页](https://equidexflow.github.io/) | ★★★ |
