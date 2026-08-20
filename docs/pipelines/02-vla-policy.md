# VLA 策略 / Vision-Language-Action Policy

## English contract

- **Objective:** learn image/language/state-to-action mappings and verify language use through closed-loop evaluation and controlled ablations.
- **Inputs:** synchronized images, instructions, robot state, actions, control rate, episode boundaries, and success labels.
- **Stages:** schema → dataset → representation → behavioral cloning/sequence training → closed loop → language ablation.
- **Acceptance:** tiny-set overfit first; then report closed-loop success, latency, episode count, confidence, and correct/swapped/absent-language gaps. Offline loss alone is insufficient.
- **Evidence:** the included PushCube model is a teaching baseline, not evidence of production-scale VLA performance. Apply the [validation policy](../VALIDATION.md).

## 目标与边界

学习从视觉、语言和机器人状态到动作或动作块的映射，并用闭环任务表现验证语言是否真正影响行为。本地示例是教学型 PushCube 基线，不代表大规模预训练 VLA 的性能。

## 前置知识与输入

- [深度学习基础](../foundations/03-deep-learning-basics.md)、[Transformer](../foundations/04-transformer-basics.md)
- [数据集与训练](../foundations/10-dataset-and-training.md)、[VLA Zero-to-One](../13-vla-zero-to-one.md)
- 输入：同步图像、语言指令、机器人状态、动作、控制频率、episode 边界和成功标签。

<div class="dof-concept" role="group" aria-label="VLA 从多模态观测到安全动作的闭环">
  <span class="dof-concept__eyebrow">Vision · Language · State → Action</span>
  <p class="dof-concept__title">VLA 的关键不是“拼接三种输入”，而是让动作、控制频率与闭环反馈保持同一契约。</p>
  <div class="dof-stage-flow">
    <div class="dof-stage dof-stage--input"><span>01 · CONTEXT</span><strong>图像 · 指令 · 状态</strong><small>对齐时间戳、shape、mask 与 episode 边界</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage"><span>02 · POLICY</span><strong>多模态表征</strong><small>encoder · fusion · action / action chunk</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage dof-stage--gate"><span>03 · EXECUTE</span><strong>限幅与闭环执行</strong><small>频率、延迟、安全过滤与下一帧观测</small></div>
  </div>
</div>

<div class="dof-principle" role="group" aria-label="VLA 将图像语言状态融合为闭环动作块">
  <p class="dof-principle__caption"><strong>原理图 · Fuse context, then act in a closed loop</strong>：VLA 的输出不只是一个标签；模型将图像、语言和本体状态编码为同一上下文，预测动作或动作块。执行端仍必须受频率、限幅和下一帧观测约束。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 246" role="img" aria-labelledby="vla-principle-title">
      <title id="vla-principle-title">VLA 多模态融合和动作块的闭环执行</title><text class="dof-diagram-title" x="30" y="39">Multimodal context → action chunk → re-observe</text>
      <rect class="dof-diagram-fill-blue" x="30" y="73" width="154" height="32" rx="9"/><text class="dof-diagram-label" x="56" y="95">image patches</text><rect class="dof-diagram-fill-violet" x="30" y="113" width="154" height="32" rx="9"/><text class="dof-diagram-label" x="56" y="135">instruction tokens</text><rect class="dof-diagram-fill-good" x="30" y="153" width="154" height="32" rx="9"/><text class="dof-diagram-label" x="56" y="175">robot state tokens</text>
      <path class="dof-diagram-accent" d="M198 129 H271"/><path class="dof-diagram-arrow" d="M271 129 l-10 -6 v12z"/><rect class="dof-diagram-surface" x="286" y="73" width="184" height="112" rx="16"/><text class="dof-diagram-title" x="332" y="108">VLA policy</text><text class="dof-diagram-note" x="318" y="134">attention / fusion</text><text class="dof-diagram-note" x="315" y="157">conditioned action model</text><path class="dof-diagram-accent" d="M485 129 H558"/><path class="dof-diagram-arrow" d="M558 129 l-10 -6 v12z"/>
      <rect class="dof-diagram-fill-violet" x="573" y="73" width="146" height="112" rx="16"/><text class="dof-diagram-label" x="598" y="105">action chunk</text><text class="dof-diagram-math" x="595" y="134">aₜ … aₜ₊ₖ</text><text class="dof-diagram-note" x="591" y="160">rate + bounds</text><path class="dof-diagram-accent" d="M733 129 H787"/><path class="dof-diagram-arrow" d="M787 129 l-10 -6 v12z"/><text class="dof-diagram-label" x="753" y="108">execute</text><text class="dof-diagram-note" x="753" y="130">safely</text>
      <path class="dof-diagram-violet" d="M798 196 C798 228 88 228 88 196"/><path class="dof-diagram-arrow-violet" d="M88 196 l-7 11 h13z"/><text class="dof-diagram-note" x="313" y="225">new observation changes the next prediction</text>
    </svg>
  </div>
</div>

## Pipeline

| 阶段 | 关键动作 | 输出/检查 |
|---|---|---|
| 1. Contract | 固定相机、状态、动作与语言 schema | batch 形状和 mask |
| 2. Dataset | 划分 episode、归一化、增强 | 数据统计和无泄漏 split |
| 3. Representation | 视觉/文本编码，动作连续化或 token 化 | 编解码往返误差 |
| 4. Training | BC/序列建模，记录 seed 与 checkpoint | loss、梯度、验证集曲线 |
| 5. Closed loop | 在未见初态运行策略 | 成功率、时延、安全事件 |
| 6. Ablation | 正确/交换/置零语言，视觉或状态基线 | 语言条件差距 |

## 运行与产物

```bash
python scripts/run_pipeline.py --run vla-policy
python scripts/run_pipeline.py --run vla-policy --full
```

入口：[unified_pushcube_vla.py](../../examples/unified_pushcube_vla.py)。产物位于 `results/pipelines/vla/`，包括策略权重与 `vla_results.json`。完整训练前先检查 [动作表示与 token 化](../24-action-representation-and-tokenization.md)。

## 验收门槛

- 先用极小数据过拟合，证明目标、mask 与动作对齐正确。
- 闭环成功率优先于离线 MSE；同时报告置信区间和 episode 数。
- 正确语言应优于交换/置零语言，否则不能声称策略使用了语言。
- 推理时延和动作频率满足部署预算，输出经限幅和安全过滤。

常见失败：图像与动作错一帧、训练/评估归一化不同、动作块执行重叠、只报告最好 seed、语言被模型忽略。
