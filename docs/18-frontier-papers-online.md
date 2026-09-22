# 前沿论文在线资源目录

> **内容状态：已读，仍有已确认待修项（2026-09-05）。** Kilohertz-Safe 的报告时间被误写为微秒而非毫秒，且安全满足比例被扩大；DexMV 目标表述也有错误（补审 F07–F08）。相关段落尚未完成重写，引用前请打开报告中的原论文。 具体位置与原始来源见 [补充独立审查](reviews/remaining-source-review.md)。

> 2025-2026 灵巧手重定向领域 12+ 篇核心论文的完整在线链接汇总，含 arXiv、项目主页、代码仓库和文档。
>
> 配套阅读：[`07-key-papers.md`](07-key-papers.md)（论文导读）| [`17-research-trends-and-positioning.md`](17-research-trends-and-positioning.md)（研究趋势分析）

---

## 一、直接研究重定向算法的论文

### 1. AnyDexRT — 少量人工引导的免标定重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2607.08341](https://arxiv.org/abs/2607.08341) |
| **PDF** | [https://arxiv.org/pdf/2607.08341](https://arxiv.org/pdf/2607.08341) |
| **作者** | Chenxi Wang, Ying Feng, Cewu Lu 等 |
| **机构** | 上海交大 + Noematrix + 上海创新研究院 |
| **年份** | 2026 |

**核心方法**：`calibration-free + few-shot human guidance`

- 自监督指尖对应学习 + 少量人工引导锚定任务相关区域
- 接触分类器精细化捏合相关姿态
- 解决新操作者/新机器人需重新校准的问题

---

### 2. Smooth Operator — 实时采样式运动学重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2607.07491](https://arxiv.org/abs/2607.07491) |
| **PDF** | [https://arxiv.org/pdf/2607.07491](https://arxiv.org/pdf/2607.07491) |
| **作者** | Robert Jomar Malate, Erik Bauer, Benedek Forrai 等 |
| **机构** | ETH Zurich |
| **年份** | 2026 |

**核心方法**：`sampling-based search → real-time, smooth robot hand configuration`

- 用采样搜索替代梯度优化，避免局部最优和初值敏感
- 解决快速动作下稳定性差的问题
- 提升遥操作数据质量

---

### 3. Kilohertz-Safe — 高频带约束安全重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2603.29213](https://arxiv.org/abs/2603.29213) |
| **PDF** | [https://arxiv.org/pdf/2603.29213](https://arxiv.org/pdf/2603.29213) |
| **作者** | Wuji Hand 团队 |
| **年份** | 2026 |

**核心方法**：`high-frequency + constraint satisfaction + safety`

- 系统性线性化统一运动学限制和碰撞避免
- 控制屏障函数（CBF）提供形式化安全保证
- 平均延迟 9.0μs，超越 Dex-Retargeting 和 GeoRT
- 同时保证 joint limit、actuator constraint、self-collision、控制频率、连续性

---

### 4. GeoRT — 超快神经几何重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2503.07541](https://arxiv.org/abs/2503.07541) |
| **PDF** | [https://arxiv.org/pdf/2503.07541](https://arxiv.org/pdf/2503.07541) |
| **作者** | Zhao-Heng Yin, Changhao Wang, Luis Pineda, Pieter Abbeel, Mustafa Mukadam 等 |
| **机构** | UC Berkeley + Meta |
| **会议** | IROS 2025 |

**核心方法**：`geometric structure + neural inference → ultrafast retargeting`

- 无监督训练，无需人工标注手部配对
- 1KHz 速度转换人手关键点到机器人手关键点
- 几何目标函数：motion fidelity、C-space coverage、high flatness、pinch correspondence
- 从"每帧在线迭代求解"向"学习后快速前向推理"转变

---

### 5. Analyzing Key Objectives — 系统分析重定向应优化什么

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2506.09384](https://arxiv.org/abs/2506.09384) |
| **PDF** | [https://arxiv.org/pdf/2506.09384](https://arxiv.org/pdf/2506.09384) |
| **作者** | Chendong Xin, Mingrui Yu, Yongpeng Jiang, Zhefeng Zhang, Xiang Li |
| **发表** | IEEE Robotics and Automation Practice (RA-P) 2025 |

**核心贡献**：实验性目标函数设计指南

- 系统分析指尖位置、指尖相对位置、指尖朝向、全局手姿态等目标项的重要性
- 实验消融揭示各目标项对实际操作成功率的影响
- 使用 SLSQP（NLopt 库）实时优化完整目标函数

---

## 二、解决 morphology gap 的工作

### 6. VTAP Gripper — 阶段化手势条件化重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2607.15448](https://arxiv.org/abs/2607.15448) |
| **PDF** | [https://arxiv.org/pdf/2607.15448](https://arxiv.org/pdf/2607.15448) |
| **作者** | Yuhao Zhou, Sheeraz Athar, Yunzhu Li 等 |
| **年份** | 2026 |

**核心方法**：`staged, gesture-conditioned retargeting`

- 根据手势/操作阶段选择不同映射：pinch / enveloping grasp / in-hand manipulation / release
- 跨越人手与异构三指夹爪之间的 embodiment gap
- 不再假设"一个统一映射适用于所有动作"

---

## 三、从"姿态复现"转向"交互保持"

### 7. TopoRetarget — 保持手-物交互拓扑

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2606.16272](https://arxiv.org/abs/2606.16272) |
| **PDF** | [https://arxiv.org/pdf/2606.16272](https://arxiv.org/pdf/2606.16272) |
| **项目主页** | [https://toporetarget2026.github.io/TopoRetarget/](https://toporetarget2026.github.io/TopoRetarget/) |
| **作者** | Jielin Wu, Shenzhe Yao, Guanqi He 等 |
| **机构** | 清华大学 IIIS (Hang Zhao 组) |
| **年份** | 2026 |

**核心方法**：`pose-preserving → interaction-preserving`

- 距离加权拉普拉斯优化保持手-物交互拓扑
- 方向一致性、运动学约束、穿透处理
- 在 ContactPose Dataset 上取得最佳接触精度和对齐
- Pen-Spin 训练成功率比基线高 40.6 个百分点
- 零样本迁移到 Wuji Hand 硬件

---

### 8. ObjRetarget — 物体感知重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2607.03828](https://arxiv.org/abs/2607.03828) |
| **PDF** | [https://arxiv.org/pdf/2607.03828](https://arxiv.org/pdf/2607.03828) |
| **作者** | Yuanchuan Lai, Qing Gao, Zhaojie Ju 等 |
| **年份** | 2026 |

**核心方法**：`L_human-robot + L_hand-object + L_arm_feasibility`

- 多面体手建模 + 仿人臂约束
- 将物体几何纳入重定向目标函数
- 解耦臂级全局运动与接触敏感手部操作

---

### 9. SPIDER — 物理感知可扩展重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2511.09484](https://arxiv.org/abs/2511.09484) |
| **PDF** | [https://arxiv.org/pdf/2511.09484](https://arxiv.org/pdf/2511.09484) |
| **项目主页** | [https://jc-bao.github.io/spider-project/](https://jc-bao.github.io/spider-project/) |
| **代码** | [https://github.com/facebookresearch/spider](https://github.com/facebookresearch/spider) |
| **文档** | [https://facebookresearch.github.io/spider/](https://facebookresearch.github.io/spider/) |
| **作者** | Chaoyi Pan, Changhao Wang, Haozhi Qi, Jitendra Malik 等 |
| **机构** | Meta FAIR |
| **年份** | 2025 |

**核心方法**：`kinematic feasibility ≠ physical feasibility`

- 物理仿真框架将运动学人类演示转化为动力学可行机器人轨迹
- 课程式虚拟接触引导 + 大规模物理采样
- 支持灵巧手和人形机器人
- MuJoCo Warp (MJWP) 工作流：GPU 加速物理仿真

---

### 10. DexFlow — 统一重定向与交互

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2505.01083](https://arxiv.org/abs/2505.01083) |
| **PDF** | [https://arxiv.org/pdf/2505.01083](https://arxiv.org/pdf/2505.01083) |
| **HTML** | [https://arxiv.org/html/2505.01083v1](https://arxiv.org/html/2505.01083v1) |
| **作者** | Xiaoyi Lin, Kunpeng Yao, Lixin Xu 等 |
| **年份** | 2025 |

**核心方法**：统一框架处理 retargeting + interaction

- 改进优化 pipeline：MANO → ShadowHand/Allegro
- 50 个 YCB 物体，292k 帧轨迹
- 多模态抓取序列：pose + joint angles series data

---

### 11. EquiDexFlow — 接触感知 SE(3) 等变生成流

| 资源 | 链接 |
|------|------|
| **项目主页** | [https://equidexflow.github.io/](https://equidexflow.github.io/) |
| **年份** | 2026 |

**核心方法**：`z → (q, T_wrist, C, N, F)`

- 联合预测：wrist pose + joint angles + fingertip contacts + surface normals + 力学量
- 8,100 个力闭合抓取，81 个物体，16-DoF Allegro Hand
- 零摩擦违反，最佳综合得分
- 接触解码后通过 per-finger IK 重定向到 16-DoF LEAP Hand
- 接触成为生成目标的一部分，而非后处理验证

---

## 四、从手工重定向走向"学习功能和任务"

### 12. DexTwist — 扭转运动功能性重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2605.12182](https://arxiv.org/abs/2605.12182) |
| **PDF** | [https://arxiv.org/pdf/2605.12182](https://arxiv.org/pdf/2605.12182) |
| **年份** | 2026 |

**核心方法**：功能性 twist-retargeting

- 检测三指捏合，估计螺旋轴和扭转量
- 实时残差关节空间优化
- 虚拟物体目标函数：turning angle + screw axis consistency + fingertip closure + tripod stability

---

### 13. DexMachina — 功能性重定向

| 资源 | 链接 |
|------|------|
| **arXiv** | [https://arxiv.org/abs/2505.24853](https://arxiv.org/abs/2505.24853) |
| **PDF** | [https://arxiv.org/pdf/2505.24853](https://arxiv.org/pdf/2505.24853) |
| **项目主页** | [https://project-dexmachina.github.io/](https://project-dexmachina.github.io/) |
| **代码** | [https://github.com/MandiZhao/dexmachina](https://github.com/MandiZhao/dexmachina) |
| **文档** | [https://mandizhao.github.io/dexmachina-docs/](https://mandizhao.github.io/dexmachina-docs/) |
| **作者** | Zhao Mandi 等 |
| **机构** | Stanford + NVIDIA |
| **年份** | 2025 |

**核心方法**：`functional retargeting` — 功能等效而非关节形状复制

- 课程学习：虚拟物体控制器强度逐渐衰减
- 从一条人类演示重定向到多种灵巧手 embodiment
- 双手铰接物体长时程操作

---

## 五、按研究问题分类速查

### 重定向求解器

| 论文 | 求解方式 | 频率 | 约束 | 链接 |
|------|---------|------|------|------|
| DexMV | SLSQP + Huber Loss | ~400Hz | joint limit | [doc](11-dexmv-research-guide.md) |
| GeoRT | 神经网络前向推理 | 1KHz | self-collision | [arXiv](https://arxiv.org/abs/2503.07541) |
| Smooth Operator | 采样搜索 | real-time | smoothness | [arXiv](https://arxiv.org/abs/2607.07491) |
| Kilohertz-Safe | 线性化 + CBF | kHz | kinematic + safety | [arXiv](https://arxiv.org/abs/2603.29213) |

### 跨形态适配

| 论文 | 方法 | 跨什么 | 链接 |
|------|------|--------|------|
| AnyDexRT | few-shot guidance | 新操作者 + 新机器人 | [arXiv](https://arxiv.org/abs/2607.08341) |
| VTAP Gripper | gesture-conditioned | 人手 → 三指夹爪 | [arXiv](https://arxiv.org/abs/2607.15448) |
| GraspGraphNet | 图结构 | 不同拓扑灵巧手 | [07-key-papers.md](07-key-papers.md) |
| UniDexTok | 统一 tokenizer | 不同 DoF 机器人手 | [07-key-papers.md](07-key-papers.md) |

### 交互保持

| 论文 | 保持什么 | 链接 |
|------|---------|------|
| TopoRetarget | 手-物接触拓扑 | [arXiv](https://arxiv.org/abs/2606.16272) / [项目页](https://toporetarget2026.github.io/TopoRetarget/) |
| ObjRetarget | 手-物空间关系 | [arXiv](https://arxiv.org/abs/2607.03828) |
| SPIDER | 物理可行性 | [arXiv](https://arxiv.org/abs/2511.09484) / [GitHub](https://github.com/facebookresearch/spider) |
| EquiDexFlow | 接触 + 法向 + 力学 | [项目页](https://equidexflow.github.io/) |

### 功能性重定向

| 论文 | 功能类型 | 链接 |
|------|---------|------|
| DexMachina | 双手铰接物体操作 | [arXiv](https://arxiv.org/abs/2505.24853) / [GitHub](https://github.com/MandiZhao/dexmachina) |
| DexTwist | 扭转/旋拧操作 | [arXiv](https://arxiv.org/abs/2605.12182) |

### 目标函数分析

| 论文 | 分析维度 | 链接 |
|------|---------|------|
| Analyzing Key Objectives | fingertip / vector / pinch / shape / task | [arXiv](https://arxiv.org/abs/2506.09384) |

---

## 六、开源代码汇总

| 项目 | 代码 | 语言 | 说明 |
|------|------|------|------|
| **SPIDER** | [facebookresearch/spider](https://github.com/facebookresearch/spider) | Python | Meta FAIR 物理感知重定向框架 |
| **DexMachina** | [MandiZhao/dexmachina](https://github.com/MandiZhao/dexmachina) | Python | Stanford+NVIDIA 功能性重定向 |
| **DexMV** | [yzqin/dexmv-sim](https://github.com/yzqin/dexmv-sim) | Python | ECCV 2022 SLSQP+Huber Loss |
| **Dex-Retargeting** | [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | Python | DexSuite 通用重定向库 |

---

## 七、引用本目录

如果在研究中使用这些论文，请引用原始论文。本目录按研究问题组织，方便快速定位相关工作。

```bibtex
% 示例：TopoRetarget
@article{toporetarget2026,
  title={TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation},
  author={Wu, Jielin and Yao, Shenzhe and He, Guanqi and others},
  journal={arXiv preprint arXiv:2606.16272},
  year={2026}
}

% 示例：SPIDER
@article{spider2025,
  title={SPIDER: Scalable Physics-Informed Dexterous Retargeting},
  author={Pan, Chaoyi and Wang, Changhao and Qi, Haozhi and others},
  journal={arXiv preprint arXiv:2511.09484},
  year={2025}
}
```
