# 13 · 机器人软件系统与安全

> **逐点图解 / Concept close-ups：**[接口、配置与测试](../knowledge-atlas/computing-software-contracts/index.md) · [频率、看门狗、状态机与安全](../knowledge-atlas/system-realtime-safety/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Robot systems and safety](../SOURCES.md#13-robot-systems-and-safety)

> 目标：理解模型如何进入真实控制栈，以及实时性、消息接口、状态机和安全约束为什么必须独立于策略模型。

## 1. 分层控制栈

```text
任务规划器（1–2 Hz）
  → VLA / 策略（5–30 Hz）
  → 机器人适配器（动作与关节映射）
  → 安全过滤器（限位、速度、碰撞、工作空间）
  → 低层控制器（100–1000 Hz）
  → 驱动器与机器人
```

大模型不应直接发送电机电流。高层策略输出目标，低层控制器负责稳定跟踪，安全层拥有最终否决权。

项目接口：[`model_interface.py`](../../examples/robot_foundation_models/common/model_interface.py)、[`embodiment_adapter.py`](../../examples/robot_foundation_models/common/embodiment_adapter.py)、[`safety_filter.py`](../../examples/robot_foundation_models/common/safety_filter.py)。

<div class="dof-principle" role="group" aria-label="机器人控制栈的多频率分层和安全否决权">
  <p class="dof-principle__caption"><strong>原理图 · The safety layer owns the final veto</strong>：上层模型生成“意图”，不能直接控制电机。动作必须先映射到机器人语义、通过独立安全过滤，再交给高频低层控制器。任一异常都能在安全层截断动作。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 260" role="img" aria-labelledby="safety-stack-title">
      <title id="safety-stack-title">从规划到驱动器的分层控制栈</title><text class="dof-diagram-title" x="31" y="38">Intent flows downward · feedback and veto flow upward</text>
      <rect class="dof-diagram-fill-violet" x="91" y="58" width="678" height="34" rx="10"/><text class="dof-diagram-label" x="117" y="81">Task planner / VLA policy · 1–30 Hz · proposes a target or action</text>
      <path class="dof-diagram-accent" d="M430 94 V110"/><path class="dof-diagram-arrow" d="M430 110 l-6 -10 h12z"/>
      <rect class="dof-diagram-fill-blue" x="91" y="116" width="678" height="38" rx="10"/><text class="dof-diagram-label" x="117" y="141">Embodiment adapter · units, joint order, action semantics, rate</text>
      <path class="dof-diagram-accent" d="M430 156 V172"/><path class="dof-diagram-arrow" d="M430 172 l-6 -10 h12z"/>
      <rect class="dof-diagram-fill-warn" x="91" y="178" width="678" height="42" rx="10"/><text class="dof-diagram-label" x="117" y="204">Safety filter · limits, collision, watchdog, stale command → HOLD / STOP</text>
      <path class="dof-diagram-good" d="M430 222 V236"/><path class="dof-diagram-arrow-good" d="M430 236 l-6 -10 h12z"/><text class="dof-diagram-note" x="569" y="245">low-level controller / driver · 100–1000 Hz</text>
      <path class="dof-diagram-violet" d="M75 223 C37 223 37 75 75 75"/><path class="dof-diagram-arrow-violet" d="M75 75 l-10 -6 v12z"/><text class="dof-diagram-note" x="16" y="151">feedback</text>
    </svg>
  </div>
</div>

## 2. ROS 2 / 中间件基本概念

| 机制 | 用途 | 典型例子 |
|:---|:---|:---|
| Topic | 连续数据流 | 图像、关节状态、动作命令 |
| Service | 短请求/响应 | 重置、查询状态 |
| Action | 可取消的长任务 | 移动到位、抓取任务 |
| TF | 坐标系树 | `base → camera → object` |
| QoS | 可靠性和历史策略 | 相机可丢旧帧，安全命令需可靠 |

接口必须写清：消息字段、单位、坐标系、频率、时间戳、超时和错误码。

## 3. 实时性与 watchdog

控制周期预算：

$$
T_{sense}+T_{sync}+T_{infer}+T_{filter}+T_{network}<T_{control}
$$

当新动作超时，应执行明确回退，而不是继续使用无限期陈旧动作：

```text
RUNNING → STALE_COMMAND → HOLD → SAFE_HOME / ESTOP
```

watchdog 至少监控：策略心跳、关节反馈、通信延迟、控制循环抖动和安全状态。

## 4. 安全约束层

最低限度：

- 关节位置、速度、加速度和力矩限制。
- 工作空间边界和自碰撞/环境碰撞检查。
- 单步动作变化限制与平滑。
- 传感器无效、通信超时和模型异常时停止。
- 独立急停和人工接管。

安全过滤器的输出应同时包含“修正后的动作”和“为什么修正/拒绝”的原因。

## 5. 安全状态机

| 状态 | 允许动作 | 进入条件 |
|:---|:---|:---|
| INIT | 无运动 | 启动、标定未完成 |
| READY | 保持/低速测试 | 所有检查通过 |
| RUNNING | 受约束策略动作 | 人工授权且心跳正常 |
| HOLD | 保持或阻尼模式 | 动作过期、短暂异常 |
| FAULT | 受控回零或停止 | 越界、持续通信故障 |
| ESTOP | 硬件急停 | 严重风险或人工触发 |

真实硬件验证必须从低速度、无负载、单步命令开始；仿真通过不能自动等价于真机安全。

## 6. 日志与可观测性

每次运行记录：commit SHA、配置、模型 checkpoint、设备、传感时间戳、原始动作、过滤后动作、安全事件和任务结果。若无法回答“失败前 2 秒发生了什么”，日志还不合格。

## 7. 检查理解

1. **架构题**：画出高层规划、策略、适配器、安全过滤和低层控制器的频率分层与数据方向。
2. **接口题**：为图像、关节反馈、复位请求和长时抓取任务选择 Topic、Service 或 Action，并说明 QoS。
3. **状态机题**：定义 stale command 从检测到 HOLD、SAFE_HOME 或 ESTOP 的转移条件与超时。
4. **安全题**：比较软件限位、独立急停和人工接管各自能覆盖与不能覆盖的故障。

下一课：[`14-evaluation-and-reproducibility.md`](14-evaluation-and-reproducibility.md)。
