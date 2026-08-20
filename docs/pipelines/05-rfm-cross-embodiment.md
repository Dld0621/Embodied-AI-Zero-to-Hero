# 机器人基础模型与跨本体 / Robot Foundation Models and Cross-embodiment

## English contract

- **Objective:** connect VLA/RFM implementations to one canonical observation/action protocol, then adapt camera, joint, action-semantic, scale, and rate differences across robots.
- **Inputs:** camera map, timestamped state, language, target action schema, calibration, limits, and model-specific preprocessing.
- **Stages:** canonical observation → preprocessing → model adapter → embodiment adapter → safety filter → closed loop → per-robot evaluation.
- **Acceptance:** validate shape, dtype, range, joint order, action semantics, rate, reset behavior, stale-observation handling, and safe failure for every adapter.
- **Evidence:** local paths validate interfaces; mock inference is not weight-level or hardware evidence. Apply the [validation policy](../VALIDATION.md).

## 目标与边界

用统一观测与动作协议接入不同 VLA/RFM，再通过 embodiment adapter 处理相机、关节、动作语义和控制频率差异。本地 SmolVLA 路径默认是 mock 接口测试，不包含真实大模型权重。

## 前置知识与输入

- [感知与传感器](../foundations/12-perception-and-sensors.md)、[机器人系统与安全](../foundations/13-robot-systems-and-safety.md)
- [机器人基础模型](../23-robot-foundation-models.md)、[跨本体适配](../25-cross-embodiment-adaptation.md)
- 输入：相机字典、机器人状态、语言、时间戳、目标机器人 action schema 和校准文件。

<div class="dof-principle" role="group" aria-label="机器人基础模型在不同本体之间通过规范动作和适配器转换">
  <p class="dof-principle__caption"><strong>原理图 · A foundation model needs an embodiment contract</strong>：模型输出的不是任意机器人的电机命令，而是规范观测上的规范动作语义。每种机器人必须由独立 adapter 把它变成自己的关节、尺度和频率，并再次经过安全过滤。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 242" role="img" aria-labelledby="cross-body-title">
      <title id="cross-body-title">规范协议通过本体适配器映射到不同机器人</title><rect class="dof-diagram-fill-blue" x="24" y="79" width="172" height="82" rx="16"/><text class="dof-diagram-label" x="49" y="109">canonical observation</text><text class="dof-diagram-note" x="44" y="133">image · state · language</text><path class="dof-diagram-accent" d="M210 120 H276"/><path class="dof-diagram-arrow" d="M276 120 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="291" y="63" width="210" height="114" rx="18"/><text class="dof-diagram-title" x="347" y="99">RFM / VLA</text><text class="dof-diagram-math" x="329" y="129">canonical Δpose</text><text class="dof-diagram-note" x="340" y="151">or action chunk</text><path class="dof-diagram-accent" d="M515 120 H565"/><path class="dof-diagram-arrow" d="M565 120 l-10 -6 v12z"/>
      <path class="dof-diagram-line" d="M580 120 V69 H621 M580 120 V173 H621"/><rect class="dof-diagram-surface" x="635" y="43" width="194" height="55" rx="14"/><text class="dof-diagram-label" x="660" y="67">adapter A · arm</text><text class="dof-diagram-note" x="660" y="87">joint order + scale</text><rect class="dof-diagram-surface" x="635" y="145" width="194" height="55" rx="14"/><text class="dof-diagram-label" x="660" y="169">adapter B · hand</text><text class="dof-diagram-note" x="660" y="189">semantic + rate map</text><text class="dof-diagram-note" x="321" y="219">never assume “same vector” means “same physical action”</text>
    </svg>
  </div>
</div>

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
