# 导航与运动控制 / Navigation and Locomotion

## English contract

- **Objective:** move a mobile or legged embodiment toward a task goal while preserving state-estimation, collision, stability, and recovery constraints.
- **Inputs:** robot model, frame tree, state estimate, map or terrain representation, goal, footprint/support constraints, controller limits, and recovery policy.
- **Stages:** state estimation → world/terrain representation → global task or route → local motion/gait command → low-level tracking → safety monitor → recovery → evaluation.
- **Acceptance:** report localization/state error, path or velocity tracking, goal success, collision/fall rate, intervention, latency, energy where relevant, and recovery success across held-out environments.
- **Evidence:** the repository records this contract but contains no reproduced navigation or legged-locomotion benchmark. Simulation, base navigation, and dynamic locomotion require different protocols.

## 目标与边界

把任务目标转化为可执行的移动行为，同时显式处理定位漂移、动态障碍、接触稳定性、控制限幅和失败恢复。这里将移动底盘导航与足式运动放在同一系统接口下，但不把两者的物理指标混为一谈。

当前状态为 **documented**。仓库尚未包含 Nav2、SLAM 或足式机器人训练环境的本地复现实验，因此此页定义工程门禁，不声明任务成功率。

## 两类执行模式

| 模式 | 核心输出 | 关键风险 | 必报指标 |
|---|---|---|---|
| Mobile navigation | global path、local velocity command | 定位漂移、障碍、卡死、局部最优 | goal success、collision、path efficiency、recovery |
| Legged locomotion | base velocity / pose target、joint command | 滑移、失稳、跌倒、接触冲击 | tracking error、fall rate、energy、terrain generalization |

## 前置知识与输入

- [坐标变换](../foundations/05-coordinate-transform.md)、[控制基础](../foundations/08-control-basics.md)、[MuJoCo](../foundations/09-mujoco-basics.md)
- [感知与传感器](../foundations/12-perception-and-sensors.md)、[机器人系统与安全](../foundations/13-robot-systems-and-safety.md)、[评估与复现](../foundations/14-evaluation-and-reproducibility.md)
- [多模态感知与状态估计](09-perception-state-estimation.md)

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
