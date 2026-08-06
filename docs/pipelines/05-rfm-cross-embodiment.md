# 机器人基础模型与跨本体 / Robot Foundation Models and Cross-embodiment

## 目标与边界

用统一观测与动作协议接入不同 VLA/RFM，再通过 embodiment adapter 处理相机、关节、动作语义和控制频率差异。本地 SmolVLA 路径默认是 mock 接口测试，不包含真实大模型权重。

## 前置知识与输入

- [感知与传感器](../foundations/12-perception-and-sensors.md)、[机器人系统与安全](../foundations/13-robot-systems-and-safety.md)
- [机器人基础模型](../23-robot-foundation-models.md)、[跨本体适配](../25-cross-embodiment-adaptation.md)
- 输入：相机字典、机器人状态、语言、时间戳、目标机器人 action schema 和校准文件。

## Pipeline

| 阶段 | 关键动作 | 输出/检查 |
|---|---|---|
| 1. Observation schema | 统一 image/state/language/timestamp | `RobotObservation` |
| 2. Preprocess | 相机映射、resize、归一化、状态编码 | 模型 batch |
| 3. Model adapter | 加载或 mock 推理，隔离框架差异 | canonical action chunk |
| 4. Embodiment adapter | 动作维度、语义、尺度、频率转换 | robot-native command |
| 5. Safety filter | 限幅、速度/加速度约束、watchdog | safe command/status |
| 6. Closed loop | 执行、观测、缓存、异常恢复 | rollout 与事件日志 |
| 7. Cross-robot eval | 相同任务协议比较不同本体 | 分机器人结果表 |

## 运行与产物

```bash
python scripts/run_pipeline.py --run rfm-cross-embodiment
```

入口：[SmolVLA adapter](../../examples/robot_foundation_models/smolvla/inference.py)，公共协议位于 [`common/`](../../examples/robot_foundation_models/common/)。真实权重部署参考 [SmolVLA GPU runbook](../28-smolvla-gpu-finetuning-runbook.md)，不能把 mock 输出当模型效果。

## 验收门槛

- 每个 adapter 都通过 shape、dtype、范围、频率和 reset 测试。
- 动作语义必须显式标注：位置/增量/速度/力矩不可混用。
- 分别报告同本体、跨场景和跨本体结果，不能只给混合平均值。
- 模型故障、延迟、陈旧观测或越界动作触发安全状态而非继续执行。

常见失败：相机名称错配、关节顺序错位、动作尺度重复归一化、chunk 缓存跨 episode 泄漏、把接口通过误报为权重验证通过。
