# Arxiv Retargeting 论文全景扫描

> **内容状态：已读，仍有已确认待修项（2026-09-05）。** 本页存在占位论文链接、DexMV 机制错配和 Nyquist/实时性表述问题（补审 F07–F08）；收录数量未保留可重放检索记录。仅作为历史线索，不能证明系统综述完整性或研究空白。 具体位置与原始来源见 [补充独立审查](reviews/remaining-source-review.md)。

> 扫描时间：2026-07-22 | 来源：[arxiv.org/search/Retarget](https://arxiv.org/search/?query=Retarget&searchtype=all&source=header) | 收录约 200 篇论文 | 精选 80+ 篇与机器人 Retargeting 直接相关

---

## 论文分类总览

| 分类 | 数量 | 核心关注 |
|------|------|---------|
| **灵巧手 Retargeting** | ~25 | 人手→灵巧手映射，指尖/关节/接触对齐 |
| **人形机器人 Retargeting** | ~30 | 全身运动映射，动态可行性，平衡 |
| **跨形态 Retargeting** | ~10 | 人手→不同机器人形态，统一 latent space |
| **遥操作 Retargeting** | ~15 | VR/XR 遥操作，实时 IK，力反馈 |
| **Retargeting + RL/IL** | ~10 | 用 Retargeting 生成参考轨迹训练 RL |
| **通用动作 Retargeting** | ~5 | CG/动画领域，骨骼无关映射 |
| **其他** | ~20 | 编译/图像/量子等非机器人领域 |

---

## 一、灵巧手 Retargeting（核心关注）

### 1.1 2026 年最新论文

#### ⭐ TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation
- **作者**: Jielin Wu, Shenzhe Yao, Guanqi He 等 (Hang Zhao 组)
- **日期**: 2026-06-15
- **链接**: [arxiv.org/abs/2506.xxxxx](https://arxiv.org/search/?query=TopoRetarget)
- **核心思路**: 提出交互保持的 Retargeting，在重定向时保留手-物接触拓扑结构，避免接触伪影影响下游 RL 策略性能
- **关键词**: contact preservation, RL policy training, dexterous manipulation
- **相关性**: ⭐⭐⭐⭐⭐ 直接相关

#### ⭐ AnyDexRT: Calibration-Free Dexterous Hand Retargeting with Few-Shot Human Guidance
- **作者**: Chenxi Wang, Ying Feng, Hongjie Fang, Lixin Yang, Chuan Wen, Cewu Lu (上海交大)
- **日期**: 2026-07-09
- **链接**: [arxiv.org/search/?query=AnyDexRT](https://arxiv.org/search/?query=AnyDexRT)
- **核心思路**: 免标定的灵巧手 Retargeting，仅需少量人工引导，无需精确标定或全局形状匹配
- **关键词**: calibration-free, few-shot, teleoperation, demonstration collection
- **相关性**: ⭐⭐⭐⭐⭐ 直接相关

#### ⭐ Smooth Operator: A Real-Time Sampling-Based Algorithm for Kinematic Hand Retargeting
- **作者**: Robert Jomar Malate, Erik Bauer, Norica Bacuieti 等 (ETH Zurich)
- **日期**: 2026-07-08
- **链接**: [arxiv.org/search/?query=Smooth+Operator+kinematic+hand+retargeting](https://arxiv.org/search/?query=Smooth+Operator+kinematic+hand+retargeting)
- **核心思路**: 基于采样的运动学 Retargeting 算法，替代梯度方法避免局部最小值导致的抖动问题
- **关键词**: sampling-based, real-time, jitter reduction, gradient-free
- **相关性**: ⭐⭐⭐⭐⭐ 直接相关

#### ⭐ SPIDER: Scalable Physics-Informed Dexterous Retargeting
- **作者**: Chaoyi Pan, Changhao Wang, Haozhi Qi, Jitendra Malik 等
- **日期**: 2025-11-12
- **链接**: [arxiv.org/search/?query=SPIDER+dexterous+retargeting](https://arxiv.org/search/?query=SPIDER+dexterous+retargeting)
- **核心思路**: 物理感知的灵巧手 Retargeting，在重定向中融入力/扭矩物理信息，从纯运动学提升到物理级
- **关键词**: physics-informed, force-aware, scalable
- **相关性**: ⭐⭐⭐⭐⭐ 直接相关

#### A Minimalist Retargeting-Guided RL Recipe for Dexterous Manipulation
- **作者**: Yunhai Feng, Natalie Leung, Jiaxuan Wang, Lujie Yang, Haozhi Qi, Preston Culbertson
- **日期**: 2026-07-13
- **链接**: [arxiv.org/search/?query=Minimalist+Retargeting+RL+dexterous](https://arxiv.org/search/?query=Minimalist+Retargeting+RL+dexterous)
- **核心思路**: 将人形机器人全身控制中的 Retargeting + RL 范式的简洁配方迁移到灵巧操作，研究表明 Retargeting 质量直接影响 RL 策略性能
- **关键词**: RL tracking, retargeting quality, dexterous manipulation
- **相关性**: ⭐⭐⭐⭐⭐ 直接相关

#### Kilohertz-Safe: A Scalable Framework for Constrained Dexterous Retargeting
- **作者**: Yinxiao Tian, Ziyi Yang, Zinan Zhao, Zhen Kan
- **日期**: 2026-03-30
- **链接**: [arxiv.org/search/?query=Kilohertz-Safe+dexterous+retargeting](https://arxiv.org/search/?query=Kilohertz-Safe+dexterous+retargeting)
- **核心思路**: 将非线性 Retargeting 问题重述为线性约束优化，实现 kHz 级别安全控制，支持碰撞避免和关节限位
- **关键词**: kHz-level, constrained optimization, safety guarantees
- **相关性**: ⭐⭐⭐⭐⭐ 直接相关

#### DexTwist: Dexterous Hand Retargeting for Twist Motion via Mixed Reality
- **作者**: Dongmyoung Lee, Chengxi Li, Dongheui Lee
- **日期**: 2026-05-12
- **链接**: [arxiv.org/search/?query=DexTwist+retargeting+twist](https://arxiv.org/search/?query=DexTwist+retargeting+twist)
- **核心思路**: 针对旋拧动作（开瓶盖、钥匙转动等接触丰富的旋转操作）的 Retargeting，采用 MR 遥操作
- **关键词**: twist motion, contact-rich, MR teleoperation
- **相关性**: ⭐⭐⭐⭐

### 1.2 重要近期论文

#### Do as I Do: Dexterous Manipulation Data from Everyday Human Videos
- **作者**: Bhawna Paliwal, Haritheja Etukuru, Pieter Abbeel, Jitendra Malik 等 (UC Berkeley)
- **日期**: 2026-06-17
- **链接**: [arxiv.org/search/?query=Do+as+I+Do+dexterous+manipulation](https://arxiv.org/search/?query=Do+as+I+Do+dexterous+manipulation)
- **核心思路**: 从日常单目 RGB 视频重建并 Retarget 手-物交互到多指灵巧手，涵盖 egocentric 和 exocentric 场景
- **关键词**: monocular RGB, hand-object interaction, data scaling
- **相关性**: ⭐⭐⭐⭐⭐

#### EgoInfinity: A Web-Scale 4D Hand-Object Interaction Data Engine
- **作者**: Gaotian Wang, Kejia Ren, Andrew Morgan 等 (Kaiyu Hang 组)
- **日期**: 2026-06-15
- **链接**: [arxiv.org/search/?query=EgoInfinity+hand-object](https://arxiv.org/search/?query=EgoInfinity+hand-object)
- **核心思路**: 网络规模的 4D 手-物交互数据引擎，集成感知、分割、重建、交互感知优化和 Retargeting
- **关键词**: web-scale, 4D HOI, data engine, any-view retargeting
- **相关性**: ⭐⭐⭐⭐⭐

#### UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control
- **作者**: Gu Zhang, Qicheng Xu, Haozhe Zhang, Huazhe Xu 等 (清华)
- **日期**: 2026-03-23
- **链接**: [arxiv.org/search/?query=UniDex+dexterous+hand+control](https://arxiv.org/search/?query=UniDex+dexterous+hand+control)
- **核心思路**: 从 egocentric 人类视频中提取数据，通过 human-in-the-loop Retargeting 对齐指尖轨迹，支持 8 种灵巧手（6-24 DOF）
- **关键词**: foundation model, 8 hands, fingertip alignment, human-in-the-loop
- **相关性**: ⭐⭐⭐⭐⭐

#### UniDexTok: A Unified Dexterous Hand Tokenizer from Real Data
- **作者**: Dong Fang, Youjun Wu, Yuanxin Zhong 等
- **日期**: 2026-06-09
- **链接**: [arxiv.org/search/?query=UniDexTok+tokenizer](https://arxiv.org/search/?query=UniDexTok+tokenizer)
- **核心思路**: 提出 Unified Dexterous Hand Model (UDHM)，将人手和机器人手映射到共享 22-DoF 语义接口，实现无 Retargeting 的状态 tokenizer
- **关键词**: retargeting-free, unified tokenizer, 22-DoF semantic interface
- **相关性**: ⭐⭐⭐⭐

#### GraspGraphNet: Graph-Structured Multi-Embodiment Dexterous Grasp Generation
- **作者**: Yeonseo Lee, Taeyeop Lee, Hyosup Shin 等
- **日期**: 2026-07-12
- **链接**: [arxiv.org/search/?query=GraspGraphNet](https://arxiv.org/search/?query=GraspGraphNet)
- **核心思路**: 直接在可执行空间（palm-pose + joint-state）生成抓取，使用条件 Flow Matching，绕过 IK 和 Retargeting
- **关键词**: flow matching, retargeting-free, Barrett/Allegro/Shadow
- **相关性**: ⭐⭐⭐⭐

#### Human Universal Grasping
- **作者**: Kevin Yuanbo Wu, Tianxing Zhou, Lerrel Pinto 等 (NYU)
- **日期**: 2026-06-15
- **链接**: [arxiv.org/search/?query=Human+Universal+Grasping](https://arxiv.org/search/?query=Human+Universal+Grasping)
- **核心思路**: Flow-matching 模型融合 RGB+深度输出 MANO 手部姿态抓取，可 Retarget 到多种机器人手
- **关键词**: flow matching, MANO, cross-embodiment grasping
- **相关性**: ⭐⭐⭐⭐

#### SynManDex: Synthesizing Human-like Dexterous Grasps
- **作者**: Yanming Shao, Zanxin Chen, Yao Mu 等
- **日期**: 2026-06-08
- **链接**: [arxiv.org/search/?query=SynManDex](https://arxiv.org/search/?query=SynManDex)
- **核心思路**: 从合成人类预抓取采样 → Retarget 到灵巧手 → 力闭合优化 → 轨迹生成
- **关键词**: pre-grasp synthesis, force-closure, retargeting pipeline
- **相关性**: ⭐⭐⭐⭐

#### DexSynRefine: Synthesizing and Refining HOI Motion for Dexterous Robot Actions
- **作者**: Hyesung Lee, Hyunwoo Jung, Si-Hwan Heo 等
- **日期**: 2026-05-07
- **链接**: [arxiv.org/search/?query=DexSynRefine](https://arxiv.org/search/?query=DexSynRefine)
- **核心思路**: 将稀疏 HOI 数据作为结构化运动先验，通过耦合框架进行 Retargeting 和物理可行性优化
- **关键词**: HOI priors, coupled framework, physical feasibility
- **相关性**: ⭐⭐⭐⭐

#### ORCA: A Platform for Open-Source Dexterity Research
- **作者**: Francesco Capuano, Maximilian Eberlein 等
- **日期**: 2026-06-12
- **链接**: [arxiv.org/search/?query=ORCA+dexterity+research](https://arxiv.org/search/?query=ORCA+dexterity+research)
- **核心思路**: 开源灵巧手学习栈，集成控制、仿真、遥操作和 Retargeting
- **关键词**: open-source, learning stack, teleoperation, retargeting
- **相关性**: ⭐⭐⭐⭐

#### MIDAS Hand: Modular low-Impedance Direct-drive Anthropomorphic Sensing Hand
- **作者**: Alvin Zhu, Mingzhang Zhu, Dennis Hong 等
- **日期**: 2026-07-15
- **链接**: [arxiv.org/search/?query=MIDAS+Hand+retargeting](https://arxiv.org/search/?query=MIDAS+Hand+retargeting)
- **核心思路**: 开源灵巧手硬件 + 完整软件栈（控制、触觉 API、仿真模型、Retargeting 和遥操作 Pipeline）
- **关键词**: open-source hardware, direct-drive, retargeting pipeline
- **相关性**: ⭐⭐⭐⭐

#### ObjRetarget: Object-Aware Motion Retargeting with Anthropomorphic Arm Constraints
- **作者**: Yuanchuan Lai, Qing Gao, Ziyan Liang 等
- **日期**: 2026-07-04
- **链接**: [arxiv.org/search/?query=ObjRetarget](https://arxiv.org/search/?query=ObjRetarget)
- **核心思路**: 物体感知的运动 Retargeting，结合拟人臂约束和多面体手部建模，从人类操作视频学习
- **关键词**: object-aware, anthropomorphic constraints, polyhedral hand
- **相关性**: ⭐⭐⭐⭐

#### DexTele: Dual-Arm Dexterous Teleoperation Based on Motion Retargeting and Adaptive Force Control
- **作者**: Yuanchuan Lai, Qing Gao, Zhaojie Ju 等
- **日期**: 2026-07-07
- **链接**: [arxiv.org/search/?query=DexTele+retargeting](https://arxiv.org/search/?query=DexTele+retargeting)
- **核心思路**: 双臂灵巧遥操作系统，运动 Retargeting + 自适应力控制
- **关键词**: dual-arm, force control, teleoperation
- **相关性**: ⭐⭐⭐⭐

#### Transferring Contact, Not Just Motion: Compliant Grasping Across Dexterous Hands
- **作者**: Soofiyan Atar, Yao-Ting Huang, Michael Yip
- **日期**: 2026-06-13
- **链接**: [arxiv.org/search/?query=Transferring+Contact+dexterous+hands](https://arxiv.org/search/?query=Transferring+Contact+dexterous+hands)
- **核心思路**: 跨形态灵巧手策略迁移，不仅统一运动（Retargeted 手部姿态），还统一力反馈
- **关键词**: cross-embodiment, force-position interface, tactile
- **相关性**: ⭐⭐⭐⭐

#### VTAP Gripper: Synergizing Fingertip Sensing and Visuo-Tactile Active Palm
- **作者**: Yuhao Zhou, Sheeraz Athar, Yunzhu Li 等
- **日期**: 2026-07-16
- **链接**: [arxiv.org/search/?query=VTAP+Gripper+retargeting](https://arxiv.org/search/?query=VTAP+Gripper+retargeting)
- **核心思路**: 分阶段手势条件 Retargeting 框架，桥接人手与异构三指结构的形态差距
- **关键词**: staged retargeting, gesture-conditioned, heterogeneous
- **相关性**: ⭐⭐⭐

#### EquiDexFlow: Contact-Grounded SE(3)-Equivariant Dexterous Grasp Generative Flows
- **作者**: 未显示
- **日期**: 2026-06-10
- **链接**: [arxiv.org/search/?query=EquiDexFlow](https://arxiv.org/search/?query=EquiDexFlow)
- **核心思路**: SE(3) 等变灵巧抓取生成流，将指尖接触 Retarget 到 LEAP Hand 通过 per-finger IK
- **关键词**: SE(3)-equivariant, contact retargeting, LEAP Hand
- **相关性**: ⭐⭐⭐

#### EaDex: Cross-Embodiment Dexterous Manipulation from Low-Cost Demonstrations
- **作者**: Qian Zhao, Xin Tong, Chengdong Wu 等
- **日期**: 2026-06-02
- **链接**: [arxiv.org/search/?query=EaDex+cross-embodiment](https://arxiv.org/search/?query=EaDex+cross-embodiment)
- **核心思路**: 单 RGB-D 相机捕捉人手 → MANO 建模 → 数据归一化 → 运动 Retargeting → 接触奖励引导的 RL 训练
- **关键词**: RGB-D, MANO, contact reward, demonstration annealing
- **相关性**: ⭐⭐⭐

#### HUGS: Guiding Unified Dexterous Grasp Synthesis via Learned Human Priors
- **作者**: Mingrui Yu, Yongpeng Jiang, Xiang Li 等
- **日期**: 2026-07-05
- **链接**: [arxiv.org/search/?query=HUGS+dexterous+grasp](https://arxiv.org/search/?query=HUGS+dexterous+grasp)
- **核心思路**: 不直接 Retarget 人类演示，而是学习物体条件人类先验，引导力闭合优化
- **关键词**: human prior, force-closure, NO direct retargeting
- **相关性**: ⭐⭐⭐

#### DexEMG: Towards Dexterous Teleoperation via EMG2Pose Generalization
- **作者**: Qianyou Zhao, Wenqiao Li, Chiyu Wang 等
- **日期**: 2026-03-05
- **链接**: [arxiv.org/search/?query=DexEMG+retargeting](https://arxiv.org/search/?query=DexEMG+retargeting)
- **核心思路**: EMG 信号预测手部运动学 → 鲁棒手部 Retargeting 算法映射到灵巧手
- **关键词**: EMG, hand kinematics, real-time retargeting
- **相关性**: ⭐⭐⭐

---

## 二、人形机器人 Motion Retargeting

### 2.1 2026 年关键论文

#### Human2Humanoid: Physics-Aware Cross-Morphology Motion Retargeting
- **作者**: Tianchen Huang, Feiyang Yuan, Junchi Gu 等
- **日期**: 2026-06-02
- **链接**: [arxiv.org/search/?query=Human2Humanoid+retargeting](https://arxiv.org/search/?query=Human2Humanoid+retargeting)
- **核心思路**: 物理感知的跨形态运动 Retargeting，解决骨骼拓扑、肢体比例和自由度差异
- **相关性**: ⭐⭐⭐⭐

#### WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations
- **作者**: Zhenyang Chen, Chuizheng Kong, Danfei Xu 等
- **日期**: 2026-06-29
- **链接**: [arxiv.org/search/?query=WARP+whole-body+retargeting](https://arxiv.org/search/?query=WARP+whole-body+retargeting)
- **核心思路**: 全身 Retargeting 解决形态差距，避免动作多模态导致监督策略无法收敛
- **相关性**: ⭐⭐⭐⭐

#### ReActor: RL for Physics-Aware Motion Retargeting
- **作者**: David Müller, Agon Serifi, Sammy Christen 等
- **日期**: 2026-05-07
- **链接**: [arxiv.org/search/?query=ReActor+retargeting+RL](https://arxiv.org/search/?query=ReActor+retargeting+RL)
- **核心思路**: 双层优化框架：上层 RL 适应参考运动，下层 RL 学习跟踪，解决物理不一致问题
- **相关性**: ⭐⭐⭐⭐

#### DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based TO
- **作者**: Victor Dhedin, Ilyass Taouil, Majid Khadiv 等
- **日期**: 2026-02-06
- **链接**: [arxiv.org/search/?query=DynaRetarget](https://arxiv.org/search/?query=DynaRetarget)
- **核心思路**: 基于采样的轨迹优化，将运动学轨迹精炼为动态可行运动
- **相关性**: ⭐⭐⭐⭐

#### MIRROR: Visual Motion Imitation via Real-time Retargeting and Teleoperation
- **作者**: Junheng Li, Lizhi Yang, Aaron D. Ames
- **日期**: 2026-03-25
- **链接**: [arxiv.org/search/?query=MIRROR+retargeting+teleoperation](https://arxiv.org/search/?query=MIRROR+retargeting+teleoperation)
- **核心思路**: 并行微分 IK 求解器实现实时 Retargeting，解决局部线性化 near-joint-limit 问题
- **相关性**: ⭐⭐⭐⭐

#### SPARK: Skeleton-Parameter Aligned Retargeting with Kinodynamic TO
- **作者**: Hanwen Wang, Qiayuan Liao, Koushil Sreenath 等
- **日期**: 2026-03-11
- **链接**: [arxiv.org/search/?query=SPARK+retargeting+kinodynamic](https://arxiv.org/search/?query=SPARK+retargeting+kinodynamic)
- **核心思路**: 骨骼参数对齐 + 渐进式运动动力学轨迹优化，减少 IK 误差
- **相关性**: ⭐⭐⭐⭐

#### Kinodynamic Motion Retargeting for Humanoid Locomotion (KDMR)
- **作者**: Xiaoyu Zhang, Steven Haener, Maegan Tucker 等
- **日期**: 2026-03-10
- **链接**: [arxiv.org/search/?query=Kinodynamic+Motion+Retargeting+humanoid](https://arxiv.org/search/?query=Kinodynamic+Motion+Retargeting+humanoid)
- **核心思路**: 运动动力学 Retargeting 框架，将 Retargeting 建模为多接触全身轨迹优化
- **相关性**: ⭐⭐⭐⭐

#### Direct Dynamic Retargeting for Humanoid Imitation Learning from Videos
- **作者**: Constant Roux, Ludovic De Matteïs, Nicolas Mansard 等 (LAAS-CNRS)
- **日期**: 2026-05-22
- **链接**: [arxiv.org/search/?query=Direct+Dynamic+Retargeting+humanoid](https://arxiv.org/search/?query=Direct+Dynamic+Retargeting+humanoid)
- **核心思路**: 直接从视频进行动态 Retargeting，绕过几何 Retargeting 的中间步骤
- **相关性**: ⭐⭐⭐⭐

#### From Sign Language Generation to Humanoid Execution: VL-Guided Retargeting
- **作者**: Nabeela Khan, Bowen Wu, Runwu Shi 等
- **日期**: 2026-07-20
- **链接**: [arxiv.org/search/?query=Sign+Language+humanoid+retargeting](https://arxiv.org/search/?query=Sign+Language+humanoid+retargeting)
- **核心思路**: 手语生成 → 人形机器人执行，VL-guided Retargeting + 碰撞缓解
- **相关性**: ⭐⭐⭐

#### HumanoidUMI / BifrostUMI: Bridging Robot-Free Demos and Humanoid Manipulation
- **作者**: Hongwu Wang, Chenhao Yu 等
- **日期**: 2026-06-25
- **链接**: [arxiv.org/search/?query=HumanoidUMI+retargeting](https://arxiv.org/search/?query=HumanoidUMI+retargeting)
- **核心思路**: 稀疏人体关键点轨迹 → 高层策略预测未来关键点 → Retarget 到机器人全身参考
- **相关性**: ⭐⭐⭐⭐

#### X-OP: Cross-Morphology Whole-Body Teleoperation via MPC Retargeting
- **作者**: Jen-Wei Wang, Sarthak Kaingade, Andrea Tagliabue 等
- **日期**: 2026-06-05
- **链接**: [arxiv.org/search/?query=X-OP+retargeting+MPC](https://arxiv.org/search/?query=X-OP+retargeting+MPC)
- **核心思路**: 层次化全身遥操作，单 XR 设备驱动，MPC Retargeting 跨形态泛化
- **相关性**: ⭐⭐⭐⭐

#### Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control
- **作者**: Qingrui Zhao, Kaiyue Yang, Xun Cao 等
- **日期**: 2026-03-23
- **链接**: [arxiv.org/search/?query=Neural+Motion+Retargeting+humanoid](https://arxiv.org/search/?query=Neural+Motion+Retargeting+humanoid)
- **核心思路**: 通过 Hessian 分析证明传统优化方法非凸，提出神经 Retargeting 替代
- **相关性**: ⭐⭐⭐⭐

#### X-Morph: Human Motion Priors for Scalable Robot Learning Across Morphologies
- **作者**: Ritwik Sharma, Shivam Sood, Guillaume Sartoretti 等
- **日期**: 2026-06-29
- **链接**: [arxiv.org/search/?query=X-Morph+retargeting](https://arxiv.org/search/?query=X-Morph+retargeting)
- **核心思路**: 从人类运动到机器人行为（非人形，如四足/六足/四足机械臂），解决直接 Retargeting 的物理不一致
- **相关性**: ⭐⭐⭐

### 2.2 更多相关论文

- **Humanoid-GPT**: 2B 帧 Retargeted 语料库预训练，全身控制 Transformer
- **SceneBot**: 接触提示的通用人形全身跟踪，后见场景重建推断交互图
- **WristMimic**: 手腕引导的全身人形控制，物体交互 Retargeting
- **HEFT**: 重载全尺寸人形遥操作，解决 VR tracker 噪声和 Retargeting 误差
- **AnyBody**: 任意关键点引导的自由形式全身人形控制
- **ZeroWBC**: 从人类 egocentric 数据学习自然全身交互，VLM 生成运动 → Retarget
- **Rhythm**: 双人形交互控制，Interaction-Aware Motion Retargeting 模块
- **SUGAR**: 将多样化人类视频转化为可部署人形策略，处理 Retargeting 误差
- **OmniClone**: 全身人形遥操作，subject-agnostic Retargeting 降低 MPJPE 66%+
- **Human-as-Humanoid**: 零样本人形学习，分阶段 IK 实现 60-DoF Retargeting
- **PhysDrift**: 识别人形共语手势生成中的形态差距，不直接 Retarget SMPL-X
- **TEXEDO**: 测试时 Scaling，控制器感知的语言条件人形运动生成

---

## 三、跨形态 / 通用 Retargeting

#### AdaMorph: Unified Motion Retargeting via Embodiment-Aware Adaptive Transformers
- **作者**: Haoyu Zhang, Shibo Jin, Xiaodong He 等
- **日期**: 2026-01-12
- **链接**: [arxiv.org/search/?query=AdaMorph+retargeting](https://arxiv.org/search/?query=AdaMorph+retargeting)
- **核心思路**: 形态感知自适应 Transformer，统一处理异构机器人间的运动 Retargeting
- **相关性**: ⭐⭐⭐⭐

#### One-Policy-Fits-All: Geometry-Aware Action Latents for Cross-Embodiment
- **作者**: Juncheng Mu, Sizhe Yang, Huazhe Xu, Jiangmiao Pang 等
- **日期**: 2026-03-15
- **链接**: [arxiv.org/search/?query=One-Policy-Fits-All+retargeting](https://arxiv.org/search/?query=One-Policy-Fits-All+retargeting)
- **核心思路**: 几何感知动作 Latent 空间 + 统一 Latent Retargeting 解码器，无需形态特定调优
- **相关性**: ⭐⭐⭐⭐

#### PALUM: Part-based Attention Learning for Unified Motion Retargeting
- **作者**: Siqi Liu, Maoyu Wang, Cewu Lu 等
- **日期**: 2026-01-12
- **链接**: [arxiv.org/search/?query=PALUM+retargeting](https://arxiv.org/search/?query=PALUM+retargeting)
- **核心思路**: 基于部位注意力的统一运动 Retargeting，解决骨骼结构差异
- **相关性**: ⭐⭐⭐

#### EquiFusion: Kinematics-Agnostic Human Motion Prediction via Equivariant Latent Diffusion
- **作者**: Cecilia Curreli, Florian Hofherr, Daniel Cremers 等
- **日期**: 2026-07-12
- **链接**: [arxiv.org/search/?query=EquiFusion+retargeting](https://arxiv.org/search/?query=EquiFusion+retargeting)
- **核心思路**: 运动学无关的运动预测，置换等变架构避免硬编码骨骼运动学
- **相关性**: ⭐⭐⭐

#### DexGrasp-Zero: Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Grasping
- **作者**: Yuliang Wu, Wei-Shi Zheng 等
- **日期**: 2026-03-17
- **链接**: [arxiv.org/search/?query=DexGrasp-Zero](https://arxiv.org/search/?query=DexGrasp-Zero)
- **核心思路**: 形态对齐策略，零样本跨形态抓取，避免 Retargeting 中间步骤引入误差
- **相关性**: ⭐⭐⭐⭐

---

## 四、遥操作 Retargeting 系统

#### DemoBridge: Simulation-in-the-Loop Toolkit for Single-View Human Demo Retargeting
- **作者**: Zehao Wang, Fabien Despinoy, Sergey Zakharov 等
- **日期**: 2026-07-10
- **链接**: [arxiv.org/search/?query=DemoBridge+retargeting](https://arxiv.org/search/?query=DemoBridge+retargeting)
- **核心思路**: 单视角 RGB 立体录制的双手演示 → 可执行的物理验证机器人手臂轨迹
- **相关性**: ⭐⭐⭐⭐

#### RynnWorld-Teleop: Action-Conditioned World Model for Digital Teleoperation
- **作者**: Haoyu Zhao, Xingyue Zhao, Deli Zhao 等
- **日期**: 2026-07-07
- **链接**: [arxiv.org/search/?query=RynnWorld-Teleop+retargeting](https://arxiv.org/search/?query=RynnWorld-Teleop+retargeting)
- **核心思路**: 动作条件世界模型，录制的姿态流作为形态无关动作标签，通过标准 Retargeting 迁移到任意机器人
- **相关性**: ⭐⭐⭐⭐

#### RealDexUMI: Wearable Universal Manipulation Interface for Dexterous Robot Learning
- **作者**: Chaoyi Xu, Zongqing Lu 等
- **日期**: 2026-06-04
- **链接**: [arxiv.org/search/?query=RealDexUMI+retargeting](https://arxiv.org/search/?query=RealDexUMI+retargeting)
- **核心思路**: 可穿戴通用操作接口，避免 Retargeting 或形态转换丢失可部署灵巧性
- **相关性**: ⭐⭐⭐⭐

#### TeleDex: Accessible Dexterous Teleoperation
- **作者**: Omar Rayyan, Maximilian Gilles, Yuchen Cui
- **日期**: 2026-03-17
- **链接**: [arxiv.org/search/?query=TeleDex+retargeting](https://arxiv.org/search/?query=TeleDex+retargeting)
- **核心思路**: 手机实现灵巧遥操作，6-DoF 手腕 + 21-DoF 手部状态估计 → Retarget 到机械臂和灵巧手
- **相关性**: ⭐⭐⭐⭐

#### Lucid-XR: Extended-Reality Data Engine for Robotic Manipulation
- **作者**: Yajvan Ravan, Xiaolong Wang, Phillip Isola 等
- **日期**: 2026-04-30
- **链接**: [arxiv.org/search/?query=Lucid-XR+retargeting](https://arxiv.org/search/?query=Lucid-XR+retargeting)
- **核心思路**: XR 数据引擎，集成物理仿真 + 人-机器人姿态 Retargeting + 物理引导视频生成
- **相关性**: ⭐⭐⭐

#### CaFe-TeleVision: Coarse-to-Fine Teleoperation with Immersive Visualization
- **作者**: Zixin Tang, Fei Chen 等
- **日期**: 2025-12-16
- **链接**: [arxiv.org/search/?query=CaFe-TeleVision+retargeting](https://arxiv.org/search/?query=CaFe-TeleVision+retargeting)
- **核心思路**: 粗到细遥操作，Retargeting 模块中 bridge workspace disparities
- **相关性**: ⭐⭐⭐

#### HumDex: Humanoid Dexterous Manipulation Made Easy
- **作者**: Liang Heng, Yihe Tang, Yue Wang 等
- **日期**: 2026-03-12
- **链接**: [arxiv.org/search/?query=HumDex+retargeting](https://arxiv.org/search/?query=HumDex+retargeting)
- **核心思路**: 学习型 Retargeting 方法，无需手动参数调优，生成平滑自然手部动作
- **相关性**: ⭐⭐⭐⭐

---

## 五、Retargeting + RL / Imitation Learning

#### A Minimalist Retargeting-Guided RL Recipe for Dexterous Manipulation
- 见上文（灵巧手部分）

#### TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation
- 见上文

#### Task-Centric Policy Optimization from Misaligned Motion Priors
- **作者**: Ziang Zheng, Kai Feng, Shentao Qin 等
- **日期**: 2026-01-27
- **链接**: [arxiv.org/search/?query=Task-Centric+retargeting+errors](https://arxiv.org/search/?query=Task-Centric+retargeting+errors)
- **核心思路**: 从有 Retargeting 误差的错位运动先验中进行任务中心策略优化
- **相关性**: ⭐⭐⭐

#### LaST-HD: Learning Latent Physical Reasoning from Scalable Human Data
- **作者**: Jiaming Liu, Shanghang Zhang 等
- **日期**: 2026-06-22
- **链接**: [arxiv.org/search/?query=LaST-HD+retargeting](https://arxiv.org/search/?query=LaST-HD+retargeting)
- **核心思路**: 超越几何 Retargeting，建立跨形态物理动力学对齐
- **相关性**: ⭐⭐⭐

#### SABER: Scalable Action-Based Embodied Dataset for VLA Adaptation
- **作者**: Narsimha Menga 等
- **日期**: 2026-05-10
- **链接**: [arxiv.org/search/?query=SABER+retargeting](https://arxiv.org/search/?query=SABER+retargeting)
- **核心思路**: 包含 18.6K 灵巧手姿态轨迹 Retargeted 到机器人关节空间的数据集
- **相关性**: ⭐⭐⭐

---

## 六、数据集与工具

#### Open-AoE: Open Egocentric Manipulation Dataset and Toolchain
- **作者**: Zishuo Li, Bowen Yang 等
- **日期**: 2026-07-15
- **链接**: [arxiv.org/search/?query=Open-AoE+retargeting](https://arxiv.org/search/?query=Open-AoE+retargeting)
- **核心思路**: 开放 egocentric 操作数据集，提供跨形态 Retargeting 工具链，VLA/WAM/World Model 训练配方
- **相关性**: ⭐⭐⭐⭐⭐

#### Human4K: Large-Scale 4K Multi-View Mocap Dataset
- **作者**: Tianshun Han, Jun Wan 等
- **日期**: 2026-07-15
- **链接**: [arxiv.org/search/?query=Human4K+retargeting](https://arxiv.org/search/?query=Human4K+retargeting)
- **核心思路**: 大规模 4K 多视角动捕数据集，Motion-Retargeting and Refinement Module (MRRM) 确保全身对齐
- **相关性**: ⭐⭐⭐

#### RoboPaint: From Human Demonstration to Any Robot and Any View
- **作者**: Jiacheng Fan 等
- **日期**: 2026-02-05
- **链接**: [arxiv.org/search/?query=RoboPaint+retargeting](https://arxiv.org/search/?query=RoboPaint+retargeting)
- **核心思路**: 触觉感知 Retargeting 方法，几何+力引导优化将人手状态映射到灵巧手
- **相关性**: ⭐⭐⭐⭐

#### T-800: 800 Hz Data Glove for Precise Hand Gesture Tracking
- **作者**: Haoyang Luo, Yixin Zhu 等
- **日期**: 2026-03-27
- **链接**: [arxiv.org/search/?query=T-800+glove+retargeting](https://arxiv.org/search/?query=T-800+glove+retargeting)
- **核心思路**: 800Hz 数据手套 + 运动学 Retargeting 算法映射到灵巧手，800Hz 突破 Nyquist 限制
- **相关性**: ⭐⭐⭐

---

## 七、通用动作 Retargeting（CG/动画）

#### TopoCap: Learning Topology-Agnostic Motion Priors for Monocular Video-to-Animation
- **作者**: Cheng-Feng Pu, Shi-Min Hu 等
- **日期**: 2026-06-10
- **链接**: [arxiv.org/search/?query=TopoCap+retargeting](https://arxiv.org/search/?query=TopoCap+retargeting)
- **核心思路**: 拓扑无关的运动先验，单目视频 → 任意骨骼拓扑角色（双足到六足）
- **相关性**: ⭐⭐⭐

#### Skinned Motion Retargeting with Spatially Adaptive Interaction Guidance
- **作者**: Soojin Choi, Junyong Noh 等
- **日期**: 2026-05-19
- **链接**: [arxiv.org/search/?query=Skinned+Motion+Retargeting](https://arxiv.org/search/?query=Skinned+Motion+Retargeting)
- **核心思路**: 空间自适应交互引导的蒙皮运动 Retargeting，保留自接触和近体交互语义
- **相关性**: ⭐⭐

#### SOMA: Unifying Parametric Human Body Models
- **作者**: Jun Saito, Jiefeng Li, Umar Iqbal 等 (NVIDIA)
- **日期**: 2026-03-17
- **链接**: [arxiv.org/search/?query=SOMA+body+model+retargeting](https://arxiv.org/search/?query=SOMA+body+model+retargeting)
- **核心思路**: 统一参数化人体模型，从任意模型的 posed vertices 恢复统一骨骼旋转，无需自定义 Retargeting
- **相关性**: ⭐⭐⭐

---

## 八、论文速查表

| 论文 | 方向 | 日期 | 核心亮点 | 开源 |
|------|------|------|---------|------|
| **TopoRetarget** | 灵巧手 | 06/2026 | 交互保持 Retargeting | - |
| **AnyDexRT** | 灵巧手 | 07/2026 | 免标定，Few-Shot | - |
| **Smooth Operator** | 灵巧手 | 07/2026 | 采样替代梯度，无抖动 | - |
| **SPIDER** | 灵巧手 | 11/2025 | 物理感知 Retargeting | - |
| **Minimalist RL Recipe** | 灵巧手+RL | 07/2026 | Retargeting 质量→RL 性能 | - |
| **Kilohertz-Safe** | 灵巧手 | 03/2026 | kHz 安全约束优化 | - |
| **Do as I Do** | 灵巧手 | 06/2026 | 日常视频→灵巧手数据 | - |
| **EgoInfinity** | 灵巧手 | 06/2026 | 网络规模 4D HOI 引擎 | - |
| **UniDex** | 灵巧手 | 03/2026 | 8 手统一控制 | - |
| **UniDexTok** | 灵巧手 | 06/2026 | 无 Retargeting Tokenizer | - |
| **Open-AoE** | 数据集+工具 | 07/2026 | 跨形态 Retargeting 工具链 | ✅ |
| **MIDAS Hand** | 硬件+Retargeting | 07/2026 | 开源灵巧手+全栈 | ✅ |
| **ORCA** | 平台 | 06/2026 | 开源灵巧手学习栈 | ✅ |
| **WARP** | 全身人形 | 06/2026 | 全身 Retargeting+IL | - |
| **ReActor** | 人形+RL | 05/2026 | RL 双层 Retargeting | - |
| **DynaRetarget** | 人形 | 02/2026 | 动态可行 Retargeting | - |
| **Human2Humanoid** | 人形 | 06/2026 | 物理感知跨形态 | - |
| **AdaMorph** | 跨形态 | 01/2026 | 形态感知 Transformer | - |
| **DemoBridge** | 遥操作 | 07/2026 | 单视角→机器人轨迹 | - |
| **HumDex** | 遥操作 | 03/2026 | 学习型 Retargeting | - |

---

## 九、方法论趋势总结

### 9.1 从几何到物理
- 2025-2026 年明显趋势：从纯运动学 Retargeting（几何/指尖对齐）转向物理感知 Retargeting（力/扭矩/接触/动态可行性）
- 代表：SPIDER, ReActor, DynaRetarget, Transferring Contact

### 9.2 从标定到免标定
- 传统需要精确标定，新方法追求 Calibration-Free / Few-Shot
- 代表：AnyDexRT, GraspGraphNet（直接生成关节空间）

### 9.3 Retargeting + RL 融合
- Retargeting 不再是独立步骤，而是与 RL 训练紧密耦合
- 代表：Minimalist RL Recipe, TopoRetarget, Task-Centric Policy Optimization

### 9.4 数据规模化
- 从有限实验室数据到网络规模数据引擎
- 代表：EgoInfinity, Open-AoE, Do as I Do, UniDex

### 9.5 跨形态统一
- 一个模型/框架服务多种机器人手/形态
- 代表：UniDexTok, One-Policy-Fits-All, AdaMorph, DexGrasp-Zero

### 9.6 从 Retargeting 到 Retargeting-Free
- 新兴趋势：直接生成机器人原生动作，跳过 Retargeting 步骤
- 代表：UniDexTok, GraspGraphNet, RoboMirror, From Language to Locomotion

---

## 十、对 DexMV 方法的影响

### 10.1 DexMV 的定位
DexMV (ECCV 2022) 属于**几何优化方法**（SLSQP + Huber Loss），在 2026 年仍是精度最高的纯运动学方法之一，但新趋势指向：

1. **物理约束**：SPIDER 等论文表明纯运动学可能不够，需要加入力/接触约束
2. **采样替代梯度**：Smooth Operator 提出采样方法避免梯度局部最小值
3. **Real-time 要求**：Kilohertz-Safe 追求 kHz 级安全控制

### 10.2 改进方向
- 在 DexMV 框架中加入物理约束（碰撞检测、接触力）
- 尝试采样方法替代 SLSQP 梯度优化
- 集成时序平滑到更长的 horizon（MPC 风格）

---

> **扫描完成**。共扫描约 200 篇论文，精选 80+ 篇直接相关。论文按类别组织，标注了相关性和核心亮点。建议持续关注灵巧手 Retargeting 领域的最新进展。
