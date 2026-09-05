# 灵巧手抓取与精细操作 / Dexterous Grasping and Fine Manipulation

> **逐点图解 / Concept close-ups：**[接触、摩擦与抓取稳定性](../knowledge-atlas/robot-contact-friction/index.md) · [操作与模仿学习系统](../knowledge-atlas/task-manipulation/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

## English contract

- **Objective:** turn perception, a grasp target, and hand state into contact-rich task execution: approach → contact → secure grasp → lift → hold → optional regrasp, reorientation, tool use, or placement.
- **Inputs:** object geometry/pose and uncertainty, task affordance, hand kinematics and limits, collision geometry, actuator and tactile state, contact/friction assumptions, and a task success predicate.
- **Stages:** object state → grasp proposal → pre-grasp IK → collision-aware approach → contact transition → force/impedance regulation → lift/hold → disturbance test → task evaluation → guarded deployment.
- **Acceptance:** geometry, contact, retention, task, robustness, and hardware evidence are reported separately. A valid joint trajectory or fingertip contact is not by itself a successful grasp.
- **Evidence:** the committed smoke path uses real MuJoCo contact dynamics with an abstract four-finger hand. It validates one deterministic approach-contact-lift-hold fixture, not a learned policy, in-hand reorientation, a production hand, or real hardware.

## 目标与边界

精细操作不是“把手指弯起来”，而是让机器人在不确定感知和接触动力学下，持续改变物体状态并满足任务条件。本 Pipeline 把已有的[灵巧手重定向](08-dexterous-retargeting.md)继续推进到**接触、保持与任务成功**层。

当前仓库已经提供一个可执行的 MuJoCo 接触动力学 smoke test：抽象四指手完成接近、闭合、抬升和抗扰保持。它证明测试夹具中的接触任务闭环能够运行，但不能推出 Shadow Hand、LEAP Hand、OrcaHand 或真实机器人上的抓取性能。

> [!NOTE]
> 本页仅整理公开、通用的机器人学知识，并使用从零构建的独立教学夹具。它不包含任何当前项目的算法、数据、实验协议、指标阈值、手型配置或研究结论。

## 重定向与精细操作的区别

| 层级 | 要回答的问题 | 典型指标 | 当前证据 |
|:---|:---|:---|:---|
| 几何重定向 | 机器人关节是否复现了目标手势？ | 指尖误差、关节违规、抖动、延迟 | 合成 smoke-tested |
| 碰撞检查 | 手与自身/物体是否发生不允许的穿透？ | 最大穿透、碰撞帧比例 | 必须单独报告 |
| 接触建立 | 正确手指是否在正确表面形成接触？ | 接触数、接触位置、法向与滑动 | MuJoCo smoke-tested |
| 抓取保持 | 物体能否被抬升并抵抗扰动？ | 抬升高度、滑移、保持时间、掉落率 | 抽象任务 smoke-tested |
| 任务成功 | 是否完成拿取、放置、旋转、装配等目标？ | 分任务成功率、耗时、恢复率 | 仅当前抓取夹具 |
| 真机能力 | 传感、控制、安全与物理参数是否在硬件上成立？ | 真机成功率、峰值力、干预率 | 未验证 |

## 需要掌握的知识点

### 1. 手—物体几何

- **坐标系与 SE(3)：** 相机、物体、手腕、掌心、指尖和机器人基座必须有明确变换链。
- **FK / Jacobian / IK：** 预抓取位姿由 IK 求解；接近阶段要处理关节限位、奇异位形和任务空间速度。
- **碰撞几何：** 视觉 mesh 与碰撞 mesh 应分开；凸分解、SDF 或简化几何用于快速距离和穿透检查。
- **抓取表示：** 可表示为手腕 6-DoF 位姿、关节姿态、接触点/法向，以及任务相关的功能接触区域。

前置章节：[SE(3)](../foundations/06-se3-and-rotation.md) · [FK/Jacobian/IK](../foundations/07-fk-jacobian-ik.md) · [概率与优化](../foundations/11-probability-and-optimization.md)

### 2. 接触力学与抓取稳定性

单个库仑接触的切向力需要满足摩擦锥约束：

$$
f_n \ge 0, \qquad \|f_t\|_2 \le \mu f_n
$$

所有接触力通过抓取矩阵 $G$ 转换为物体上的合力/合力矩（wrench）：

$$
w = Gf
$$

<div class="dof-principle" role="group" aria-label="抓取摩擦锥与合力矩原理图">
  <p class="dof-principle__caption"><strong>原理图 · Contact forces become an object wrench.</strong> 每个接触只能在本地摩擦锥允许的方向内施力；多接触力经抓取矩阵 <em>G</em> 组合，才形成作用在物体上的合力与合力矩。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 330" role="img" aria-labelledby="grasp-figure-title grasp-figure-desc">
      <title id="grasp-figure-title">Friction cones and grasp wrench</title>
      <desc id="grasp-figure-desc">Planar cross-section of two forces acting on the object. Each normal points inward from its contact; its tangential component is perpendicular. The resultant, not the tangential component alone, must lie in the local friction cone. The grasp matrix maps contact forces to an object wrench.</desc>
      <defs>
        <marker id="grasp-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow" d="M0,0 L7,3 L0,6 Z"/></marker>
        <marker id="grasp-arrow-good" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow-good" d="M0,0 L7,3 L0,6 Z"/></marker>
        <marker id="grasp-arrow-violet" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow-violet" d="M0,0 L7,3 L0,6 Z"/></marker>
      </defs>
      <rect class="dof-diagram-surface" x="48" y="44" width="512" height="244" rx="20"/>
      <text class="dof-diagram-title" x="74" y="78">Local contact constraints</text>
      <rect class="dof-diagram-fill-blue" x="254" y="112" width="110" height="110" rx="26"/>
      <text class="dof-diagram-label" x="283" y="214">object</text>
      <circle class="dof-diagram-fill-good" cx="254" cy="166" r="6"/><circle class="dof-diagram-fill-good" cx="364" cy="166" r="6"/>
      <path id="grasp-cone-left" class="dof-diagram-dash" d="M299 136 L254 166 L299 196 Z"/>
      <path id="grasp-cone-right" class="dof-diagram-dash" d="M319 136 L364 166 L319 196 Z"/>
      <path id="grasp-normal-left" class="dof-diagram-good" d="M254 166 H290" marker-end="url(#grasp-arrow-good)"/>
      <path id="grasp-normal-right" class="dof-diagram-good" d="M364 166 H328" marker-end="url(#grasp-arrow-good)"/>
      <text class="dof-diagram-math" x="267" y="186">fₙ¹</text><text class="dof-diagram-math" x="329" y="186">fₙ²</text>
      <text class="dof-diagram-note" x="86" y="108">forces ON object</text><text class="dof-diagram-note" x="376" y="108">cones point inward</text>
      <path id="grasp-tangent-left" class="dof-diagram-violet" d="M254 166 V148" marker-end="url(#grasp-arrow-violet)"/>
      <path id="grasp-tangent-right" class="dof-diagram-violet" d="M364 166 V148" marker-end="url(#grasp-arrow-violet)"/>
      <text class="dof-diagram-math" x="218" y="143">fₜ¹</text><text class="dof-diagram-math" x="380" y="143">fₜ²</text>
      <text class="dof-diagram-note" x="95" y="250">normal + tangent = resultant</text>
      <text class="dof-diagram-note" x="95" y="272">resultant in cone: ‖fₜ‖ ≤ μ fₙ</text>
      <path class="dof-diagram-accent" d="M574 166 H676" marker-end="url(#grasp-arrow)"/>
      <text class="dof-diagram-math" x="589" y="148">f = [f¹, f², …]</text>
      <rect class="dof-diagram-surface" x="694" y="91" width="176" height="146" rx="18"/>
      <text class="dof-diagram-title" x="722" y="124">Object wrench</text>
      <text class="dof-diagram-math" x="730" y="161">w = G f</text>
      <text class="dof-diagram-note" x="720" y="188">net force + torque</text>
      <path class="dof-diagram-violet" d="M782 210 A32 32 0 1 0 810 173" marker-end="url(#grasp-arrow)"/>
      <text class="dof-diagram-note" x="694" y="266">stable grasp requires more than a contact count</text>
    </svg>
  </div>
</div>

**读图约定：** 图为二维截面，所有箭头表示“手指对物体的力”，法向从接触点指向物体内侧，切向与法向垂直。绿色和紫色箭头是分量；受摩擦锥约束的是二者的合力，不是要求切向分量自身在锥内。图未画出重力，也不凭两个接触证明三维力闭合。参见 [Modern Robotics：摩擦锥](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-1-friction/)。

- **Form closure：** 仅依靠几何约束即可阻止物体任意微小运动。
- **Force closure：** 允许接触力在摩擦锥内变化，并能抵抗任意方向的小扰动 wrench。
- **Grasp quality：** 常见解析指标包括最小奇异值、wrench-space 余量与 Ferrari–Canny $\epsilon$；它们是代理指标，不能替代动态抬升与扰动测试。
- **滑移与滚动：** 观察切向相对速度、接触位置漂移和触觉变化，而不只统计 `ncon`。
- **软接触：** 指腹形变、扭转摩擦和接触面积会影响真实抓取；点接触模型需要明确近似边界。

MuJoCo 的接触包含法向、切向以及可选扭转/滚动摩擦；`contact.dist < 0` 表示穿透。详见 [MuJoCo contact computation](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)。

### 3. 感知与状态估计

| 信号 | 用途 | 失败时的表现 |
|:---|:---|:---|
| RGB-D / 点云 | 分割、6-DoF 位姿、形状与遮挡 | 抓取位姿偏离、碰撞 |
| 本体感知 | 关节位置/速度、电流或力矩 | 闭合状态不可判断 |
| 触觉 / 力觉 | 接触时刻、压力分布、滑移 | 过度挤压或掉落 |
| 物体状态 | 抬升、旋转、放置和任务判定 | “手在动”被误判为任务成功 |
| 时间戳与置信度 | 多传感器同步和安全退化 | 延迟反馈导致振荡 |

前置章节：[感知与传感器](../foundations/12-perception-and-sensors.md) · [感知与状态估计 Pipeline](09-perception-state-estimation.md)

### 4. 控制

- **位置控制：** 入门最容易，但高刚度闭合可能产生过大接触力。
- **力矩控制：** 可直接优化动力学行为，但依赖模型、频率和安全约束。
- **阻抗/导纳控制：** 把期望位姿与接触力联系起来，适合从自由空间切换到接触空间。
- **混合位置—力控制：** 沿切向跟踪运动，沿法向调节接触力，常用于插接、旋钮和擦拭。
- **触觉闭环：** 通过压力分布和滑移事件调整各指闭合量，实现抓稳且不过度挤压。
- **安全层：** 独立限制关节、速度、力矩、接触力、工作空间和通信超时。

前置章节：[控制基础](../foundations/08-control-basics.md) · [机器人系统与安全](../foundations/13-robot-systems-and-safety.md)

### 5. 规划与学习

| 方法 | 适用位置 | 优点 | 主要局限 |
|:---|:---|:---|:---|
| 解析抓取 + IK | 已知物体、结构化场景 | 可解释、样本需求低 | 难覆盖复杂接触与遮挡 |
| 轨迹优化 / MPC | 接近、接触和短时程调整 | 可显式加入碰撞与动力学 | 初值和模型敏感 |
| 模仿学习 | 从遥操作/示范学习精细动作 | 能保留人类策略先验 | 分布偏移和示范质量敏感 |
| 强化学习 | 抗扰、重抓、在手操作 | 能优化长时程任务奖励 | 样本量和 sim-to-real 成本高 |
| Diffusion / VLA | 多模态、多峰动作生成 | 能表达多种抓取策略 | 数据、延迟与低层安全仍关键 |

抓取姿态生成和抓取执行应分开评测。UniDexGrasp 也采用“抓取 proposal + goal-conditioned execution”的两阶段结构；见 [CVPR 2023 official page](https://cvpr2023.thecvf.com/virtual/2023/poster/21614)。

<div class="dof-concept" role="group" aria-label="从预抓取到保持的灵巧操作闭环">
  <span class="dof-concept__eyebrow">Contact-rich manipulation</span>
  <p class="dof-concept__title">关节轨迹、接触、保持和任务成功是连续阶段，但必须以独立证据分别报告。</p>
  <div class="dof-stage-flow">
    <div class="dof-stage dof-stage--input"><span>01 · PLAN</span><strong>物体状态与预抓取</strong><small>pose · affordance · IK · collision check</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage"><span>02 · CONTACT</span><strong>接近与力切换</strong><small>first contact · compliance · force limits</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage"><span>03 · RETAIN</span><strong>闭合、抬升与保持</strong><small>friction · slip · disturbance · recovery</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage dof-stage--gate"><span>04 · PROVE</span><strong>任务与安全评测</strong><small>geometry ≠ contact ≠ task ≠ hardware</small></div>
  </div>
</div>

## 任务级 Pipeline

| 阶段 | 关键动作 | 输出 | 晋级检查 |
|:---|:---|:---|:---|
| 1. Task contract | 定义物体、初始分布、动作和成功条件 | task spec | 成功条件可由状态自动判断 |
| 2. Perception | 分割、位姿/形状估计、置信度 | object state | 标定、同步和不确定性合格 |
| 3. Grasp proposal | 选择腕位姿、手型、接触区域 | grasp candidates | 碰撞、可达性和多样性 |
| 4. Pre-grasp IK | 求解手臂/手腕与张开手型 | feasible pre-grasp | 关节、速度与自碰约束 |
| 5. Approach | 沿安全方向接近并监控早接触 | approach trajectory | 无意外碰撞、可安全停止 |
| 6. Contact transition | 检测首接触并切换低刚度/力控制 | contact state | 正确接触、峰值力受限 |
| 7. Grasp stabilization | 多指闭合、力分配、滑移抑制 | stable grasp | 接触数、滑移和穿透合格 |
| 8. Task execution | 抬升、保持、移动、旋转或使用工具 | object trajectory | 任务状态变化达到阈值 |
| 9. Recovery | 检测掉落/滑移并重抓或回撤 | recovery event | 不隐藏失败，记录恢复结果 |
| 10. Evaluation | 固定种子、物体和扰动测试 | metrics + replay | 达到任务与安全门槛 |
| 11. Deployment | HIL → 低速影子 → 受控真机 | release record | 人工授权、急停与回滚 |

## 可扩展的精细动作

1. **抓取与抬升：** 包络抓、指尖捏取、薄物体边缘抓取、易碎物体柔顺抓取。
2. **在手操作：** 物体平移、滚动、旋转、手指步态（finger gaiting）与重抓。
3. **功能抓取：** 按工具用途选择接触区域，例如握笔、握锤或使用剪刀。
4. **装配与交互：** 插接、旋钮、开盖、按键、线缆整理，需要位置—力混合控制。
5. **动态交互：** 人机交接、移动物体抓取和扰动恢复，需要预测、低延迟感知与安全策略。

先用抓取与抬升建立最小任务闭环，再扩展到在手旋转和工具使用。不要同时更换物体、手型、控制器和奖励，否则失败原因不可定位。

## MuJoCo 仿真闭环

### 已提交 smoke test

```bash
pip install numpy mujoco
python scripts/run_pipeline.py --run dexterous-manipulation

# 或直接指定产物
python examples/dexterous_grasping_smoke.py \
  --check \
  --output results/pipelines/dexterous_grasping/smoke/metrics.json
```

- 模型：[`assets/simulation/dexterous_grasp_smoke.xml`](../../assets/simulation/dexterous_grasp_smoke.xml)
- 入口：[`examples/dexterous_grasping_smoke.py`](../../examples/dexterous_grasping_smoke.py)

夹具运行三种确定性条件：标称摩擦、较低摩擦、较强横向扰动。阶段状态机为：

```text
OPEN / ABOVE OBJECT
        ↓ approach
PRE-GRASP ALIGNED
        ↓ close until multi-finger contact
CONTACT ESTABLISHED
        ↓ lift while regulating closure
OBJECT LIFTED
        ↓ bounded disturbance
RETAINED or DROP / RECOVER
```

当前验收指标：

| 指标 | smoke 门槛 | 含义 |
|:---|---:|:---|
| `grasp_success_rate` | `1.0` across 3 fixtures | 所有已登记条件完成抓取保持 |
| `mean_lift_height_m` | `≥ 0.045` | 物体离开支撑面，而非只接触 |
| `max_lateral_slip_m` | `≤ 0.020` | 扰动期间物体未明显滑移 |
| `minimum_final_finger_contacts` | `≥ 2` | 最终仍存在多指保持 |
| `max_contact_penetration_m` | 记录，不作为成功替代项 | 监控接触数值质量 |

### 换成真实手模型

仓库包含 Shadow Hand、LEAP Hand、Allegro Hand 和 OrcaHand 资产，但接入某个模型时仍需逐项完成：

1. 固定 joint/actuator 顺序、限位、耦合关系与控制频率。
2. 检查 collision geom、质量、惯量、接触对、摩擦与 solver 参数。
3. 定义指尖、指腹、掌心 site/sensor，并验证接触语义。
4. 将抽象 `closure_target` 替换为该手的 synergy、关节目标或力矩动作。
5. 加入机械臂 6-DoF 预抓取与接近，避免只在“物体已位于掌心”条件下评测。
6. 在对象族上随机化质量、尺寸、摩擦、初始位姿、感知噪声和控制延迟。
7. 分别评估 proposal quality、execution success、retention、recovery 与 sim-to-real。

可参考仓库中的 [MuJoCo Shadow Hand model](../../pretrained/urdf/mujoco_menagerie/shadow_hand/README.md)；模型可加载不等于抓取任务已经验证。

## 训练课程建议

| Stage | 环境与目标 | 何时晋级 |
|:---|:---|:---|
| A. Kinematics | 空手目标姿态、reach、关节限位 | 几何与时序门槛通过 |
| B. Static contact | 已知物体固定在掌心附近 | 正确接触、低穿透、不过力 |
| C. Grasp and lift | 随机物体位姿、抬升与保持 | 成功率、滑移、掉落门槛通过 |
| D. Robust grasp | 摩擦/质量/尺寸/延迟随机化 | 固定 holdout 扰动下稳定 |
| E. In-hand manipulation | 旋转、平移、重抓 | 位置/旋转误差与掉落率合格 |
| F. Functional task | 工具、装配、交接 | 按任务条件评测，不只看姿态 |
| G. Hardware gate | HIL、低速、限空间、人工监督 | 安全报告与授权齐备 |

## 评测与失败分类

- **Geometry:** fingertip/object error, joint-limit and self-collision violations.
- **Contact:** intended-contact precision/recall, force distribution, peak force, penetration, slip events.
- **Task:** grasp/lift/hold success, pose error, completion time, drop and recovery rate.
- **Robustness:** object, pose, friction, mass, sensor noise, latency, disturbance, and morphology holdouts.
- **Safety:** torque/force limit violations, unsafe collision, stale-command rate, intervention and rollback events.
- **Hardware:** reported only after a separately authorized real-robot protocol.

失败应定位为：感知错误、proposal 不可达、接近碰撞、接触位置错误、摩擦不足、闭合过力、抬升后滑移、策略振荡、恢复失败或 sim-to-real 参数偏差。

## Primary references

- [MuJoCo computation and contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)
- [Gymnasium-Robotics Shadow Dexterous Hand tasks](https://robotics.farama.org/envs/shadow_dexterous_hand/)
- [Isaac Lab manipulation and dexterous-hand environments](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [DexGraspNet, ICRA 2023](https://arxiv.org/abs/2210.02697)
- [UniDexGrasp, CVPR 2023](https://cvpr2023.thecvf.com/virtual/2023/poster/21614)
- [Dexterous Grasp Transformer, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Dexterous_Grasp_Transformer_CVPR_2024_paper.html)
- [Dexterous Functional Grasping, CoRL 2023](https://proceedings.mlr.press/v229/agarwal23a.html)
