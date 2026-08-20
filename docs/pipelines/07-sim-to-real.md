# Sim-to-Real 部署 / Simulation-to-Real Deployment

## English contract

- **Objective:** transfer a frozen policy through progressively higher-risk gates without treating simulation as authorization for physical motion.
- **Inputs:** checkpoint, simulation report, robot and sensor calibration, control interface, limits, emergency-stop procedure, operator, and rollback plan.
- **Stages:** robust simulation → offline replay → hardware-in-the-loop → shadow mode → guarded rollout → bounded expansion → rollback readiness.
- **Acceptance:** record distribution gaps, latency, jitter, dropped/stale observations, task success, intervention, safety-filter activation, emergency stops, configuration, and incidents.
- **Evidence:** this repository documents the gates but claims no locally reproduced hardware result. Physical execution requires separate site-specific approval and safety review.

## 目标与边界

把仿真中通过的策略按风险递增顺序迁移到真实系统。此路线故意没有“一键真机”命令：硬件、急停、场地、负载和操作者确认不能被本地 smoke test 代替。

## 前置知识与输入

- [MuJoCo 基础](../foundations/09-mujoco-basics.md)、[感知与传感器](../foundations/12-perception-and-sensors.md)
- [机器人系统与安全](../foundations/13-robot-systems-and-safety.md)、[评估与复现](../foundations/14-evaluation-and-reproducibility.md)
- [Sim-to-Real 指南](../19-sim-to-real-guide.md)
- 输入：冻结 checkpoint、仿真评估报告、机器人/传感器标定、控制接口、限制参数、急停与回滚步骤。

## Pipeline 与门禁

| 阶段 | 必做检查 | 晋级条件 |
|---|---|---|
| 1. Simulation | 随机化动力学、观测噪声、延迟和初态 | 多 seed 与最坏场景达标 |
| 2. Replay | 真实日志离线回放，不发命令 | schema、单位、时间戳一致 |
| 3. HIL | 控制器和传感器接入，执行器隔离或低风险 | watchdog/急停/限幅有效 |
| 4. Shadow mode | 策略只建议动作，与人工/基线对比 | 时延和偏差在阈值内 |
| 5. Guarded rollout | 空载、低速、单任务、单次运行 | 无越界和异常状态 |
| 6. Expansion | 逐步增加速度、负载和场景 | 每级都有回归报告 |
| 7. Rollback | 异常立即停止并恢复已知安全版本 | 事件可追踪、可复盘 |

<div class="dof-concept" role="group" aria-label="从仿真到现实的风险递增门禁">
  <span class="dof-concept__eyebrow">Risk increases · evidence must increase with it</span>
  <p class="dof-concept__title">每向真实世界前进一步，都要新增独立的安全证据；仿真通过不构成真机授权。</p>
  <div class="dof-evidence-rail">
    <div><span>01</span><strong>Robust simulation</strong><small>随机化与最坏场景</small></div>
    <div><span>02</span><strong>Offline replay</strong><small>只读真实日志</small></div>
    <div><span>03</span><strong>HIL</strong><small>硬件接口、限幅与急停</small></div>
    <div><span>04</span><strong>Shadow mode</strong><small>建议动作，不发命令</small></div>
    <div><span>05 · AUTHORIZED</span><strong>Guarded rollout</strong><small>低速、空载、单任务、可回滚</small></div>
  </div>
</div>

## 验收指标

- sim/real 观测与动作分布差距，按传感器和关节分别报告。
- 端到端控制延迟、抖动、丢帧率、陈旧观测率。
- 任务成功率、人工接管率、安全过滤触发率、急停次数。
- 版本、校准、环境、操作者和异常事件完整记录。

## 失败模式

坐标系或单位错配、控制频率不一致、通信断开后保持旧命令、仿真接触参数过拟合、策略输出越界、只比较平均值而忽略最坏情况。任何新硬件、持久化系统修改或真机动作都应单独获得明确授权并执行现场安全流程。
