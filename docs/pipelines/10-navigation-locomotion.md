# 导航与运动控制 / Navigation and Locomotion

> **逐点图解 / Concept close-ups：**[导航与移动具身智能体](../knowledge-atlas/task-navigation/index.md) · [运动控制、平衡与人形系统](../knowledge-atlas/task-locomotion-humanoids/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

## English contract

- **Objective:** move a mobile or legged embodiment toward a task goal while preserving state-estimation, collision, stability, and recovery constraints.
- **Inputs:** robot model, frame tree, state estimate, map or terrain representation, goal, footprint/support constraints, controller limits, and recovery policy.
- **Stages:** state estimation → world/terrain representation → global task or route → local motion/gait command → low-level tracking → safety monitor → recovery → evaluation.
- **Acceptance:** report localization/state error, path or velocity tracking, goal success, collision/fall rate, intervention, latency, energy where relevant, and recovery success across held-out environments.
- **Evidence:** the repository includes a deterministic grid-navigation smoke test with A* planning, tracking metrics, obstacle interception, and replanning recovery. Continuous dynamics, base navigation, and legged locomotion still require separate protocols.

## 目标与边界

把任务目标转化为可执行的移动行为，同时显式处理定位漂移、动态障碍、接触稳定性、控制限幅和失败恢复。这里将移动底盘导航与足式运动放在同一系统接口下，但不把两者的物理指标混为一谈。

当前状态为 **smoke-tested（栅格导航）**。仓库提供固定地图与固定种子的 A* 教学脚本，检查目标到达、跟踪误差、动态障碍安全拦截和重规划恢复；它不等于 Nav2/SLAM 复现，也不覆盖连续动力学、足式运动或真机安全。

## Quick smoke / 快速验证

```bash
python scripts/run_pipeline.py --run navigation-locomotion
```

默认产物为 `results/pipelines/navigation/smoke/metrics.json`。三组确定性场景报告以栅格单元为单位的定位与跟踪 RMSE、目标成功率、碰撞率、安全介入率和恢复成功率；动态场景必须先拦截新障碍，再重规划到达目标。

| 本地 smoke 能证明 | 不能据此声称 |
|---|---|
| A*、轨迹执行、指标聚合、安全拦截、恢复记录和 JSON 产物连通 | 连续控制稳定性、动态避障性能、Nav2/SLAM 复现、足式策略质量或真机无碰撞 |

## 两类执行模式

| 模式 | 核心输出 | 关键风险 | 必报指标 |
|---|---|---|---|
| Mobile navigation | global path、local velocity command | 定位漂移、障碍、卡死、局部最优 | goal success、collision、path efficiency、recovery |
| Legged locomotion | base velocity / pose target、joint command | 滑移、失稳、跌倒、接触冲击 | tracking error、fall rate、energy、terrain generalization |

## 前置知识与输入

- [坐标变换](../foundations/05-coordinate-transform.md)、[控制基础](../foundations/08-control-basics.md)、[MuJoCo](../foundations/09-mujoco-basics.md)
- [感知与传感器](../foundations/12-perception-and-sensors.md)、[机器人系统与安全](../foundations/13-robot-systems-and-safety.md)、[评估与复现](../foundations/14-evaluation-and-reproducibility.md)
- [多模态感知与状态估计](09-perception-state-estimation.md)

<div class="dof-principle" role="group" aria-label="导航中坐标系全局路径局部控制和安全重规划之间的关系">
  <p class="dof-principle__caption"><strong>原理图 · Global intent needs local, safe execution</strong>：全局规划负责“往哪里去”，局部控制负责在当前 <code>map / odom / base</code> 状态、障碍和动力学限制下“现在怎么走”。新障碍或定位漂移必须触发安全拦截和重规划。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 248" role="img" aria-labelledby="nav-principle-title">
      <title id="nav-principle-title">导航的全局路径、局部控制和安全重规划</title><rect class="dof-diagram-surface" x="23" y="53" width="372" height="158" rx="18"/><text class="dof-diagram-title" x="47" y="83">World / map frame</text><path class="dof-diagram-line" d="M55 184 H361 M55 184 V100"/><path class="dof-diagram-accent" d="M85 164 C144 91 251 176 342 103"/><circle class="dof-diagram-fill-good" cx="342" cy="103" r="10"/><text class="dof-diagram-note" x="320" y="88">goal</text><rect class="dof-diagram-fill-warn" x="218" y="126" width="28" height="28" rx="4"/><text class="dof-diagram-note" x="207" y="174">global path</text><circle class="dof-diagram-fill-blue" cx="85" cy="164" r="9"/><text class="dof-diagram-note" x="62" y="198">base pose</text>
      <path class="dof-diagram-accent" d="M412 130 H475"/><path class="dof-diagram-arrow" d="M475 130 l-10 -6 v12z"/><rect class="dof-diagram-fill-violet" x="490" y="53" width="158" height="158" rx="18"/><text class="dof-diagram-label" x="522" y="88">local motion</text><text class="dof-diagram-note" x="519" y="114">track + avoid</text><text class="dof-diagram-math" x="528" y="146">v, ω / gait</text><text class="dof-diagram-note" x="512" y="177">limits + dynamics</text><path class="dof-diagram-accent" d="M663 130 H719"/><path class="dof-diagram-arrow" d="M719 130 l-10 -6 v12z"/><rect class="dof-diagram-fill-warn" x="734" y="53" width="101" height="158" rx="18"/><text class="dof-diagram-label" x="750" y="88">safety</text><text class="dof-diagram-note" x="748" y="116">collision</text><text class="dof-diagram-note" x="748" y="137">stability</text><text class="dof-diagram-note" x="748" y="158">replan</text><path class="dof-diagram-violet" d="M784 226 C784 243 208 243 208 218"/><path class="dof-diagram-arrow-violet" d="M208 218 l-7 11 h13z"/>
    </svg>
  </div>
</div>

## Pipeline 与门禁

| 阶段 | 关键动作 | 晋级条件 |
|---|---|---|
| 1. Frames & state | 建立 `map/odom/base` 或机身/足端 frame，估计 pose/velocity | frame 连续、状态延迟与漂移受控 |
| 2. World model | 构建地图、代价地图或地形/接触表示 | 障碍膨胀、未知区域和动态更新可解释 |
| 3. Global objective | 生成路线、覆盖计划或期望机身速度 | 目标可达、约束与终止条件明确 |
| 4. Local motion | 路径跟踪、避障或步态/策略输出 | 命令连续、限幅、满足局部安全约束 |
| 5. Low-level control | 速度、位置或力矩闭环 | 频率、饱和、watchdog 和停止状态通过 |
| 6. Recovery | 重规划、脱困、稳定恢复或安全停机 | 失败可检测，恢复次数和结果有记录 |
| 7. Evaluation | 多地图/地形、扰动、延迟和传感器退化 | 分场景报告成功、碰撞/跌倒和最坏情况 |

## 验收门槛

- 导航与足式运动必须分别给出任务定义、控制频率、动力学和安全边界。
- 路径成功不能掩盖碰撞；速度跟踪不能掩盖跌倒或高能耗。
- 地图、初始位姿、地形、随机种子和传感器配置必须保留。
- 仿真结果只支持仿真结论；HIL、影子模式和真机部署继续遵循 [Sim-to-Real 管线](07-sim-to-real.md)。

## 主要失败模式

`map/odom/base` 语义混乱、定位跳变导致控制尖峰、代价地图陈旧、全局路径可达但局部控制不可行、策略在训练地形过拟合、跌倒后继续输出命令、只报告平均回报而不报告碰撞/跌倒。

权威入口：[Nav2 Concepts](https://docs.nav2.org/concepts/) · [ROS REP 105](https://www.ros.org/reps/rep-0105.html) · [Isaac Lab Environments](https://isaac-sim.github.io/IsaacLab/develop/source/overview/environments.html)
