# 开源灵巧手对比分析

> **内容状态：已读，仍有已确认待修项（2026-09-05）。** LEAP Hand 的机构及 V1/V2 驱动、成本口径混淆尚待逐表修正（补审 F05）；其余设备价格、许可证与规格也未逐产品验证。本表不作为采购依据，先核对对应版本的官方资料。 具体位置与原始来源见 [补充独立审查](reviews/remaining-source-review.md)。

> 面向具身 AI / 灵巧操作研究的灵巧机器人手全面对比，包含每根手指的 DOF 结构分解、开源模型下载链接、驱动方式、价格和代表性论文。

## 人手结构基准（对比参考）

人手可作为所有灵巧手设计的生物学参考。人手骨骼由 **腕骨、掌骨、指骨** 组成，每根手指的关节与自由度如下：

| 手指 | 关节组成 | DOF | 运动类型 |
|------|---------|-----|---------|
| **拇指** | CMC（腕掌）+ MCP（掌指）+ IP（指间） | **5** | CMC 外展/内收(1) + CMC 屈伸(1) + CMC 旋前/旋后(1) + MCP 屈伸(1) + IP 屈伸(1) |
| **食指** | MCP + PIP + DIP | **4** | MCP 外展/内收(1) + MCP 屈伸(1) + PIP 屈伸(1) + DIP 屈伸(1) |
| **中指** | MCP + PIP + DIP | **4** | 同上 |
| **环指** | MCP + PIP + DIP | **4** | 同上 |
| **小指** | MCP + PIP + DIP + 掌骨外展 | **5** | MCP 外展/内收(1) + MCP 屈伸(1) + PIP 屈伸(1) + DIP 屈伸(1) + 掌骨外展(1) |
| **腕部** | — | **6** | 3 轴旋转 + 3 轴平移（通常不计入手指 DOF） |

**人手手指总计**：约 20-22 DOF（不含腕部）。所有灵巧手的设计都在这 20+ DOF 的框架上做减法或耦合，以平衡成本、复杂度和功能。

---

## 总览对比

| 手名称 | 手指数 | 总 DOF | 开源硬件 | URDF | MJCF | 驱动方式 | 价格 (USD) | 许可证 |
|--------|--------|--------|---------|------|------|---------|-----------|--------|
| **LEAP Hand** | 4 | 16 | **是** | 有 | 有 | 直驱电机 | USD 200–2,000 | MIT (代码) / CC BY-NC-SA (CAD) |
| **ORCA Hand** | 5 | 17 | **是** | 有 | 有 | 腱驱动 | USD 800–1,000 | MIT |
| **Shadow Hand** | 5 | 24 | 否 | 有 | 有 | 腱驱动 | ~USD 100,000+ | Apache 2.0 (ROS) |
| **Allegro Hand** | 4 | 16 | 否 | 有 | 无 | 直驱电机 | ~USD 19,000 | Apache 2.0 (ROS) |
| **O10 / OmniHand** | 5 | 10 | 否 | 有 | 有 | 直驱电机 | ~USD 5,000–8,000 | 商业 |
| **DEX-EE** | 3 | 12 | 否 | 有 | 有 | 直驱 + 触觉 | 未公开 | — |
| **Robotiq 3F** | 3 | 4 | 否 | 有 | 无 | 欠驱动 | USD 3,000–5,000 | BSD (ROS) |
| **Schunk SVH** | 5 | 9 | 否 | 有 | 无 | 直驱 | USD 50,000–70,000 | Apache 2.0 (ROS) |
| **AR10** | 5 | 10 | 否 | 有 | 无 | 舵机 | USD 5,000–8,000 | BSD |

---

## 1. LEAP Hand（CMU / RSS 2023）

**最成熟的完全开源灵巧手方案，社区活跃，仿真支持最完善。**

| 属性 | 信息 |
|------|------|
| **机构** | Columbia University |
| **论文** | LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning (RSS 2023) |
| **总 DOF** | 16（4 指 x 4 自由度） |
| **驱动方式** | 直驱电机（每关节独立电机） |
| **GitHub** | [leap-hand](https://github.com/leap-hand) |
| **价格** | V1 ~USD 2,000（BOM）；V2 ~USD 200–300（简化版） |
| **许可证** | MIT（代码），CC BY-NC-SA（CAD 文件） |

### 每指 DOF 分解

LEAP Hand **无小指**，共 4 指（拇指、食指、中指、环指），每指 4 DOF：

| 手指 | 关节名 | 类型 | 运动范围 (rad) | 说明 |
|------|--------|------|---------------|------|
| **拇指** | joint 12 (根部 yaw) | revolute | -0.349 ~ 2.094 | 拇指根部旋转，类似 CMC |
| | joint 13 (PIP) | revolute | -0.470 ~ 2.443 | 近端指节屈曲 |
| | joint 14 (DIP) | revolute | -1.200 ~ 1.900 | 远端指节屈曲 |
| | joint 15 (指尖) | revolute | -1.340 ~ 1.880 | 末端指节屈曲 |
| **食指** | joint 0 (MCP 外展) | revolute | -1.047 ~ 1.047 | MCP 外展/内收 |
| | joint 1 (MCP 屈曲) | revolute | -0.314 ~ 2.230 | MCP 屈曲 |
| | joint 2 (PIP) | revolute | -0.506 ~ 1.885 | 中间指节屈曲 |
| | joint 3 (DIP) | revolute | -0.366 ~ 2.042 | 末端指节屈曲 |
| **中指** | joint 4-7 | revolute | 同食指 | 同食指结构 |
| **环指** | joint 8-11 | revolute | 同食指 | 同食指结构 |

**关键特性**：
- 完全开源硬件（3D CAD、STL、组装指南），4 小时可组装
- V2 版本（2025）含腱驱动版本，成本降至 USD 200–300
- 创新的通用外展/内收机构
- 配套遥操作：BiDex 搭配 Manus 手套

**模型下载**（已放入 `pretrained/urdf/leap_hand_sim/assets/leap_hand/`）：
```bash
# 已下载到本项目
ls pretrained/urdf/leap_hand_sim/assets/leap_hand/
# robot.urdf, *.stl
```

---

## 2. ORCA Hand（ETH Zurich）

**完全开源，同时提供 URDF + MJCF，硬件成本极低，腱驱动。**

| 属性 | 信息 |
|------|------|
| **机构** | ETH Zurich |
| **总 DOF** | 17（5 指 + 腕部 1） |
| **驱动方式** | 腱驱动（电机置于前臂，通过腱绳传动） |
| **核心仓库** | [orca_core](https://github.com/orcahand/orca_core) |
| **模型文件** | [orcahand_description](https://github.com/orcahand/orcahand_description) |
| **价格** | ~USD 800–1,000（BOM 成本） |
| **许可证** | MIT |

### 每指 DOF 分解

ORCA Hand 共 5 指，含完整腕部关节：

| 手指 | 关节名 | 类型 | 运动范围 (rad) | 说明 |
|------|--------|------|---------------|------|
| **拇指 (T)** | right_t-cmc | revolute | -0.785 ~ 0.576 | CMC 旋转（类似人手腕掌关节） |
| | right_t-abd | revolute | -0.314 ~ 0.960 | 外展/内收 |
| | right_t-mcp | revolute | -0.436 ~ 1.745 | MCP 屈曲 |
| | right_t-pip | revolute | -0.262 ~ 1.868 | PIP 屈曲 |
| **食指 (I)** | right_i-abd | revolute | -0.436 ~ 0.524 | MCP 外展/内收 |
| | right_i-mcp | revolute | -0.436 ~ 1.745 | MCP 屈曲 |
| | right_i-pip | revolute | -0.262 ~ 1.868 | PIP 屈曲 |
| **中指 (M)** | right_m-abd | revolute | -0.471 ~ 0.471 | MCP 外展/内收 |
| | right_m-mcp | revolute | -0.436 ~ 1.745 | MCP 屈曲 |
| | right_m-pip | revolute | -0.262 ~ 1.868 | PIP 屈曲 |
| **环指 (R)** | right_r-abd | revolute | -0.471 ~ 0.471 | MCP 外展/内收 |
| | right_r-mcp | revolute | -0.436 ~ 1.745 | MCP 屈曲 |
| | right_r-pip | revolute | -0.262 ~ 1.868 | PIP 屈曲 |
| **小指 (P)** | right_p-abd | revolute | -0.524 ~ 0.524 | MCP 外展/内收 |
| | right_p-mcp | revolute | -0.436 ~ 1.745 | MCP 屈曲 |
| | right_p-pip | revolute | -0.262 ~ 1.868 | PIP 屈曲 |
| **腕部** | right_wrist | revolute | -1.134 ~ 0.611 | 腕部屈伸 |

**关键特性**：
- 拇指有 4 DOF（含 CMC 旋转 + 外展），是所有开源手中拇指最灵活的
- 腱驱动引入非线性映射，Retargeting 需考虑腱绳耦合
- 除拇指外，其余 4 指均无 DIP 独立关节（PIP 和 DIP 耦合或省略）
- 左右手模型均已提供（`orcahand_left.urdf` / `orcahand_right.urdf`）

**模型下载**（已放入 `pretrained/urdf/orcahand_description/`）：
```bash
# v1 和 v2 版本均已下载
ls pretrained/urdf/orcahand_description/v1/models/urdf/
ls pretrained/urdf/orcahand_description/v2/models/urdf/
```

---

## 3. Shadow Hand（Shadow Robot Company）

**最经典的仿人灵巧手，24-DOF，大量顶会论文使用，但硬件昂贵。**

| 属性 | 信息 |
|------|------|
| **机构** | Shadow Robot Company (UK) |
| **总 DOF** | 24（20 个驱动 + 4 个欠驱动，5 指 + 腕部） |
| **驱动方式** | 腱驱动（空气肌肉 / 直流电机） |
| **ROS 仓库** | [shadow-robot](https://github.com/shadow-robot) |
| **MuJoCo 模型** | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/master/shadow_hand) |
| **价格** | ~USD 100,000+ |
| **许可证** | 代码 Apache 2.0，硬件商业闭源 |

### 每指 DOF 分解

Shadow Hand 是**最接近人手**的商用灵巧手，每根手指结构如下：

| 手指 | 关节名 | 类型 | 运动范围 (rad) | 说明 |
|------|--------|------|---------------|------|
| **腕部** | rh_WRJ2 | revolute | -0.524 ~ 0.175 | 腕部 y 轴（屈伸） |
| | rh_WRJ1 | revolute | -0.698 ~ 0.489 | 腕部 x 轴（外展） |
| **拇指 (TH)** | rh_THJ5 | revolute | -1.047 ~ 1.047 | TM 外展/内收（z轴） |
| | rh_THJ4 | revolute | 0 ~ 1.222 | TM 屈曲（x轴） |
| | rh_THJ3 | revolute | -0.209 ~ 0.209 | TM 旋转（hub） |
| | rh_THJ2 | revolute | -0.698 ~ 0.698 | MCP 屈曲（y轴） |
| | rh_THJ1 | revolute | -0.262 ~ 1.571 | IP 屈曲 |
| **食指 (FF)** | rh_FFJ4 | revolute | -0.349 ~ 0.349 | MCP 外展/内收 |
| | rh_FFJ3 | revolute | -0.262 ~ 1.571 | MCP 屈曲 |
| | rh_FFJ2 | revolute | 0 ~ 1.571 | PIP 屈曲（欠驱动） |
| | rh_FFJ1 | revolute | 0 ~ 1.571 | DIP 屈曲（欠驱动） |
| **中指 (MF)** | rh_MFJ4 ~ MFJ1 | revolute | 同食指 | 同食指结构 |
| **环指 (RF)** | rh_RFJ4 ~ RFJ1 | revolute | 同食指 | 同食指结构 |
| **小指 (LF)** | rh_LFJ5 | revolute | 0 ~ 0.785 | 掌骨外展（metacarpal） |
| | rh_LFJ4 | revolute | -0.349 ~ 0.349 | MCP 外展/内收 |
| | rh_LFJ3 | revolute | -0.262 ~ 1.571 | MCP 屈曲 |
| | rh_LFJ2 | revolute | 0 ~ 1.571 | PIP 屈曲（欠驱动） |
| | rh_LFJ1 | revolute | 0 ~ 1.571 | DIP 屈曲（欠驱动） |

**欠驱动说明**：Shadow Hand 的食指/中指/环指/小指的 PIP 和 DIP 通过 **腱绳耦合**（tendon coupling），即 `FFJ0 = FFJ2 + FFJ1`，因此 4 根手指各只有 3 个独立驱动器（外展 + MCP + PIP/DIP 耦合）。拇指 5 个关节全部独立驱动。

**驱动器统计**：
- 腕部 2 + 拇指 5 + 食指 3 + 中指 3 + 环指 3 + 小指 4 = **20 个驱动器**
- 24 个关节中，4 个为欠驱动（PIP/DIP 耦合）

**关键特性**：
- 129 个传感器（位置、力、触觉）
- 力矩控制环 5kHz，位置控制环 1kHz
- OpenAI Dactyl（解魔方）使用的就是 Shadow Hand

**模型下载**（已放入 `pretrained/urdf/mujoco_menagerie/shadow_hand/`）：
```bash
ls pretrained/urdf/mujoco_menagerie/shadow_hand/
# right_hand.xml, left_hand.xml, scene_right.xml, scene_left.xml, assets/*.obj
```

---

## 4. Allegro Hand（Wonik Robotics / 韩国）

**商业产品中性价比最高的灵巧手，ROS 生态完善，4 指 16-DOF。**

| 属性 | 信息 |
|------|------|
| **总 DOF** | 16（4 指 x 4 自由度） |
| **驱动方式** | 直驱电机（每关节独立电流控制） |
| **ROS V4 仓库** | [allegro_hand_ros_v4](https://github.com/simlabrobotics/allegro_hand_ros_v4) |
| **价格** | ~USD 19,000（V4），V5 ~USD 20,000+（新增指尖触觉） |
| **许可证** | ROS 驱动 Apache 2.0 |

### 每指 DOF 分解

Allegro Hand **无小指**，共 4 指，每指 4 DOF：

| 手指 | 关节名 | 类型 | 运动范围 (rad) | 说明 |
|------|--------|------|---------------|------|
| **拇指** | joint_12.0 | revolute | — | 拇指根部外展/内收（TM） |
| | joint_13.0 | revolute | — | 拇指根部屈曲 |
| | joint_14.0 | revolute | — | MCP 屈曲 |
| | joint_15.0 | revolute | — | IP 屈曲 |
| **食指** | joint_0.0 | revolute | -0.47 ~ 0.47 | MCP 外展/内收 |
| | joint_1.0 | revolute | -0.196 ~ 1.61 | MCP 屈曲 |
| | joint_2.0 | revolute | — | PIP 屈曲 |
| | joint_3.0 | revolute | — | DIP 屈曲 |
| **中指** | joint_4.0 ~ 7.0 | revolute | 同食指 | 同食指结构 |
| **环指** | joint_8.0 ~ 11.0 | revolute | 同食指 | 同食指结构 |

**关键特性**：
- 与 LEAP Hand 同为 16-DOF 4 指结构，但 Allegro 为商业产品
- 直驱电机，控制精度高
- ROS 驱动成熟，Gazebo 仿真支持完善

**模型下载**（已放入 `pretrained/urdf/allegro_hand_right/` 和 `allegro_hand_left/`）：
```bash
ls pretrained/urdf/allegro_hand_right/
# allegro_hand_right.urdf, meshes/visual/*.obj, meshes/collision/*.obj
```

---

## 5. O10 / OmniHand（AgiBot）

**本项目实际使用的目标灵巧手，10 个主动关节，结构简洁。**

| 属性 | 信息 |
|------|------|
| **总 DOF** | 10（5 指 x 2 自由度） |
| **驱动方式** | 直驱电机 |
| **关节限位** | [0, 1.2] rad |
| **价格** | ~USD 5,000–8,000 |

### 每指 DOF 分解

O10 Hand 共 5 指，每指仅 **2 DOF**，是所有人形灵巧手中结构最精简的：

| 手指 | 关节名 | 类型 | 运动范围 | 说明 |
|------|--------|------|---------|------|
| **拇指** | thumb_abd | revolute | [0, 1.2] | 外展/内收 |
| | thumb_flex | revolute | [0, 1.2] | 屈曲（PIP/DIP 耦合） |
| **食指** | index_abd | revolute | [0, 1.2] | 外展/内收 |
| | index_flex | revolute | [0, 1.2] | 屈曲（PIP/DIP 耦合） |
| **中指** | middle_abd | revolute | [0, 1.2] | 外展/内收 |
| | middle_flex | revolute | [0, 1.2] | 屈曲（PIP/DIP 耦合） |
| **环指** | ring_abd | revolute | [0, 1.2] | 外展/内收 |
| | ring_flex | revolute | [0, 1.2] | 屈曲（PIP/DIP 耦合） |
| **小指** | pinky_abd | revolute | [0, 1.2] | 外展/内收 |
| | pinky_flex | revolute | [0, 1.2] | 屈曲（PIP/DIP 耦合） |

**关键特性**：
- 每指仅 2 DOF（外展 + 耦合屈曲），Retargeting 最简单
- 10 个主动关节，控制维度低，适合初学者
- PIP 和 DIP 通过机械结构耦合，无法独立控制
- 项目中通过 `finger_curl` 参数同时驱动屈曲关节

---

## 6. DEX-EE（Shadow Robot / Google DeepMind, 2024）

**面向研究的新型三指灵巧手，指尖集成光学触觉传感器。**

| 属性 | 信息 |
|------|------|
| **总 DOF** | 12（3 指 x 4 自由度） |
| **驱动方式** | 直驱电机 |
| **指尖传感器** | 光学触觉（数百个 taxel） |
| **仓库** | [dx_system](https://github.com/shadow-robot/dx_system) |

### 每指 DOF 分解

DEX-EE 为 **3 指**设计，每指 4 DOF：

| 手指 | 关节 | DOF | 说明 |
|------|------|-----|------|
| **指 1** | MCP 外展 + MCP 屈曲 + PIP + DIP | 4 | 类似 Allegro/LEAP 单指结构 |
| **指 2** | MCP 外展 + MCP 屈曲 + PIP + DIP | 4 | 同上 |
| **指 3** | MCP 外展 + MCP 屈曲 + PIP + DIP | 4 | 同上 |

**关键特性**：
- 3 指 12-DOF 设计，面向特定抓取任务优化
- 指尖光学触觉传感器提供高分辨率接触信息
- 与 Google DeepMind 合作开发

---

## 7. Robotiq 3-Finger Gripper

**自适应欠驱动夹爪，更适合工业抓取而非灵巧操作。**

| 属性 | 信息 |
|------|------|
| **总 DOF** | 4（3 指，自适应欠驱动） |
| **驱动方式** | 欠驱动（单电机驱动多关节） |
| **价格** | USD 3,000–5,000 |

### DOF 分解

| 手指 | 关节 | DOF | 说明 |
|------|------|-----|------|
| **指 1** | 根部旋转 + 近端屈曲 + 远端屈曲 | 1（驱动）+ 2（欠驱动） | 单电机驱动，phalanx 自适应 |
| **指 2** | 同上 | 1（驱动）+ 2（欠驱动） | 同上 |
| **指 3** | 根部旋转 + 近端屈曲 + 远端屈曲 | 1（驱动）+ 2（欠驱动） | 同上 |
| **手掌** | 手掌旋转 | 1 | 3 指围绕中心旋转 |

**说明**：Robotiq 3F 本质是 **自适应夹爪**，非真正灵巧手。每指无法独立控制各关节，依靠弹簧和机械结构自适应包裹物体。不适合精细灵巧操作研究。

---

## 8. Schunk SVH

**工业级五指灵巧手，可靠性高但自由度较少。**

| 属性 | 信息 |
|------|------|
| **总 DOF** | 9（5 指，9 个独立驱动关节） |
| **驱动方式** | 直驱电机 |
| **通信** | EtherCAT |
| **价格** | USD 50,000–70,000+ |

### 每指 DOF 分解

| 手指 | 关节 | DOF | 说明 |
|------|------|-----|------|
| **拇指** | CMC 旋转 + MCP 屈曲 | 2 | 拇指可对掌 |
| **食指** | MCP 屈曲 + PIP 屈曲 | 2 | 无外展独立关节 |
| **中指** | MCP 屈曲 | 1 | 仅根部屈曲 |
| **环指** | MCP 屈曲 + PIP 屈曲 | 2 | — |
| **小指** | MCP 屈曲 + PIP 屈曲 | 2 | — |

**说明**：Schunk SVH 设计偏向工业可靠性而非高灵活性。中指仅 1 DOF，且多指无外展关节，限制了手势表达能力。

---

## 9. AR10

**低成本舵机驱动灵巧手，结构与 O10 类似。**

| 属性 | 信息 |
|------|------|
| **总 DOF** | 10（5 指 x 2 自由度） |
| **驱动方式** | 舵机（伺服电机） |
| **价格** | USD 5,000–8,000 |

### 每指 DOF 分解

与 O10 相同，每指 2 DOF（外展 + 耦合屈曲）。使用舵机驱动，成本较低但控制精度不如直驱电机。

---

## 灵巧手 DOF 结构对比图

```
人手 (20 DOF)          Shadow (24 DOF)        LEAP/Allegro (16 DOF)
拇指: CMC2+MCP+IP      拇指: 5 DOF            拇指: 4 DOF
食指: MCP2+PIP+DIP     食指: 4 DOF            食指: 4 DOF
中指: MCP2+PIP+DIP     中指: 4 DOF            中指: 4 DOF
环指: MCP2+PIP+DIP     环指: 4 DOF            环指: 4 DOF
小指: MCP2+PIP+DIP+MC  小指: 5 DOF            (无小指)
腕部: —                腕部: 2 DOF            (无腕部)

ORCA (17 DOF)          O10/AR10 (10 DOF)      Robotiq 3F (4 DOF)
拇指: 4 DOF            拇指: 2 DOF            指1: 1(驱动)+2(欠)
食指: 3 DOF            食指: 2 DOF            指2: 1(驱动)+2(欠)
中指: 3 DOF            中指: 2 DOF            指3: 1(驱动)+2(欠)
环指: 3 DOF            环指: 2 DOF            手掌: 1 DOF
小指: 3 DOF            小指: 2 DOF
腕部: 1 DOF
```

---

## 与 Retargeting 的适配性分析

| 手名称 | 总 DOF | 每指平均 DOF | Retargeting 难度 | 适合本项目的方法 | 说明 |
|--------|--------|-------------|-----------------|----------------|------|
| **O10** | 10 | 2.0 | ⭐ 最低 | Rule-based + Vector Opt | 本项目默认目标，结构最简单 |
| **AR10** | 10 | 2.0 | ⭐ 低 | Rule-based + Vector Opt | 与 O10 类似结构 |
| **Robotiq 3F** | 4 | 1.3 | — | 不适用 | 欠驱动夹爪，不适合灵巧 retargeting |
| **Schunk SVH** | 9 | 1.8 | ⭐⭐ 中 | Rule-based | 自由度少但结构不规则 |
| **LEAP Hand** | 16 | 4.0 | ⭐⭐ 中 | Rule-based + Vector Opt | 每指 4 DOF，比 O10 多 1 个外展/指 |
| **Allegro Hand** | 16 | 4.0 | ⭐⭐ 中 | Rule-based + Vector Opt | 直驱，与 LEAP 类似结构 |
| **ORCA Hand** | 17 | 3.4 | ⭐⭐⭐ 中高 | Vector Opt | 腱驱动非线性，需考虑腱耦合 |
| **DEX-EE** | 12 | 4.0 | ⭐⭐ 中 | Vector Opt | 3 指，与 4 指手略有不同 |
| **Shadow Hand** | 24 | 4.8 | ⭐⭐⭐⭐ 高 | Vector Opt | 自由度最多，欠驱动耦合复杂 |

### Retargeting 难度说明

| 难度因素 | 影响 |
|---------|------|
| **DOF 数量** | DOF 越多，IK 求解越复杂，但表达力越强 |
| **外展/内收关节** | 无外展关节（如 O10）Retargeting 更简单；有外展关节（如 LEAP）需额外映射 |
| **腱驱动** | ORCA/Shadow 的腱驱动引入非线性传动比，需校准 |
| **欠驱动** | Shadow Hand 的 PIP/DIP 耦合减少了独立控制维度 |
| **拇指结构** | 拇指是最难映射的部分（人手拇指有 5 DOF，机器人拇指通常 3-5 DOF） |

### 推荐选择

- **入门学习**：O10 / AR10（10 DOF，结构最简单）
- **研究复现**：LEAP Hand（完全开源，仿真成熟，社区活跃）
- **商业应用**：Allegro Hand（可靠性高，ROS 生态完善）
- **极限灵巧**：Shadow Hand（24 DOF，但价格昂贵）
- **低成本硬件**：ORCA Hand（USD 800，完全开源，腱驱动有研究价值）
