# 12 · 机器人感知与传感器

> **逐点图解 / Concept close-ups：**[传感器模型与观测语义](../knowledge-atlas/system-sensor-models/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Perception and sensors](../SOURCES.md#12-perception-and-sensors)

> 目标：把“图像输入”扩展为真实机器人观测系统，理解相机、深度、状态、力觉和触觉数据如何标定、同步、验证并送入策略。

## 1. 观测不是一张图片

机器人策略的观测通常是：

$$
o_t=\{I_t^{front},I_t^{wrist},d_t,q_t,\dot q_t,g_t,f_t,\tau_t\}
$$

其中包括多视角 RGB、深度、关节位置/速度、夹爪状态、力和触觉。项目统一接口见 [`observation_schema.py`](../../examples/robot_foundation_models/common/observation_schema.py)。

## 2. 主要传感器

| 模态 | 提供的信息 | 关键风险 |
|:---|:---|:---|
| RGB 相机 | 纹理、颜色、目标类别 | 光照、遮挡、曝光 |
| RGB-D/深度 | 几何距离、点云 | 空洞、反光、外参漂移 |
| 关节编码器 | 关节位置和速度 | 零点偏移、延迟 |
| 力矩/六维力传感器 | 接触力与力矩 | 偏置、温漂、过载 |
| 触觉阵列 | 局部接触分布、滑移 | 采样频率和标定差异 |
| IMU | 姿态和加速度 | 积分漂移、振动 |

## 3. 相机几何

针孔模型：

$$
s\begin{bmatrix}u\\v\\1\end{bmatrix}
=K[R\mid t]\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
$$

- `K`：相机内参，描述焦距和主点。
- `[R|t]`：外参，把世界或机器人坐标变换到相机坐标。
- 手眼标定回答“相机与机器人基座/末端之间如何变换”。

<div class="dof-principle" role="group" aria-label="针孔相机投影原理图">
  <p class="dof-principle__caption"><strong>原理图 · Pinhole projection.</strong> 外参先把世界点带入相机坐标；同一条光线穿过针孔后，在像平面上形成像素。深度 Z 决定投影尺度。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 920 330" role="img" aria-labelledby="camera-figure-title camera-figure-desc">
      <title id="camera-figure-title">Pinhole camera projection</title>
      <desc id="camera-figure-desc">A three dimensional point in camera coordinates projects through a pinhole onto an image plane at pixel u v. Intrinsics map the ray to pixels and extrinsics establish the camera frame.</desc>
      <defs>
        <marker id="camera-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path class="dof-diagram-arrow" d="M0,0 L7,3 L0,6 Z"/></marker>
      </defs>
      <rect class="dof-diagram-surface" x="32" y="44" width="220" height="226" rx="18"/>
      <text class="dof-diagram-title" x="58" y="79">World / robot frame</text>
      <path class="dof-diagram-accent" d="M85 208 h70" marker-end="url(#camera-arrow)"/>
      <path class="dof-diagram-violet" d="M85 208 v-72" marker-end="url(#camera-arrow)"/>
      <text class="dof-diagram-note" x="164" y="212">xᵂ</text><text class="dof-diagram-note" x="78" y="129">yᵂ</text>
      <circle class="dof-diagram-fill-good" cx="180" cy="126" r="10"/>
      <text class="dof-diagram-label" x="119" y="111">3D point Pᵂ</text>
      <path class="dof-diagram-dash" d="M260 153 H334" marker-end="url(#camera-arrow)"/>
      <text class="dof-diagram-math" x="269" y="137">[R | t]</text>
      <line class="dof-diagram-violet" x1="392" y1="52" x2="392" y2="270"/>
      <text class="dof-diagram-title" x="345" y="38">image plane</text>
      <circle class="dof-diagram-fill-blue" cx="500" cy="162" r="9"/>
      <text class="dof-diagram-label" x="466" y="193">pinhole</text>
      <path class="dof-diagram-accent" d="M500 162 H715" marker-end="url(#camera-arrow)"/>
      <text class="dof-diagram-note" x="645" y="151">optical axis zᶜ</text>
      <circle class="dof-diagram-fill-good" cx="730" cy="80" r="11"/>
      <text class="dof-diagram-label" x="747" y="83">Pᶜ = (X, Y, Z)</text>
      <path class="dof-diagram-line" d="M730 80 L500 162 L392 202"/>
      <path class="dof-diagram-line" d="M730 80 L500 162 L392 126"/>
      <circle class="dof-diagram-fill-violet" cx="392" cy="202" r="8"/>
      <text class="dof-diagram-label" x="304" y="226">pixel (u, v)</text>
      <path class="dof-diagram-dash" d="M392 162 h-35 M392 162 v36"/>
      <text class="dof-diagram-note" x="333" y="157">principal point</text>
      <rect class="dof-diagram-surface" x="610" y="220" width="246" height="60" rx="13"/>
      <text class="dof-diagram-math" x="634" y="246">u = fₓ X / Z + cₓ</text>
      <text class="dof-diagram-note" x="634" y="267">K maps camera rays to pixels</text>
    </svg>
  </div>
</div>

若外参误差为 1 cm，视觉目标即使检测正确，末端也可能稳定地到达错误位置。

## 4. 感知 pipeline

<div class="dof-concept" role="group" aria-label="从传感器到机器人观测的感知闭环">
  <span class="dof-concept__eyebrow">Sensor streams → RobotObservation</span>
  <p class="dof-concept__title">每一次状态估计都必须同时保留时间、坐标、置信度与健康信息。</p>
  <div class="dof-stage-flow">
    <div class="dof-stage dof-stage--input"><span>01 · RAW</span><strong>传感器采集</strong><small>RGB-D · 编码器 · 力/触觉 · IMU</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage"><span>02 · ALIGN</span><strong>同步与标定</strong><small>timestamp · intrinsics · extrinsics</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage"><span>03 · ESTIMATE</span><strong>表示与融合</strong><small>去畸变 · 变换 · state / covariance</small></div>
    <i class="dof-flow-arrow" aria-hidden="true">→</i>
    <div class="dof-stage dof-stage--gate"><span>04 · USE OR DEGRADE</span><strong>任务级观测</strong><small>freshness · confidence · missing-data policy</small></div>
  </div>
</div>

原则：保留原始数据；派生特征带版本号；每一帧都能追溯到时间戳和标定文件。

## 5. 时间同步与延迟

不同模态不能按数组下标直接拼接。应以时间戳对齐，并记录最大允许时间差：

```python
def nearest_sample(samples, target_t, max_dt=0.02):
    best = min(samples, key=lambda item: abs(item[0] - target_t))
    if abs(best[0] - target_t) > max_dt:
        raise ValueError("sensor sample is stale")
    return best[1]
```

20 Hz 策略的一个周期只有 50 ms；30 ms 感知延迟加 30 ms 推理延迟已经超过周期预算。

## 6. 数据质量门禁

送入训练前至少检查：

- 图像尺寸、颜色空间、深度单位固定。
- 关节维度、顺序和单位与机器人描述一致。
- 时间戳单调，模态间时间差有统计。
- NaN、丢帧、过曝、深度空洞有明确标记。
- 相机内外参和机器人零点有版本与日期。
- 训练集和测试集不共享同一连续轨迹片段。

## 7. 选择表示

- 端到端 VLA：尽量保留 RGB 和本体状态，让模型学习表征。
- 小数据任务：检测/分割/关键点可降低样本复杂度。
- 跨机器人：优先使用任务空间、归一化状态和明确的 embodiment metadata。
- 接触任务：仅靠 RGB 往往不够，应考虑力/触觉或接触状态。

## 8. 检查理解

1. **概念题**：解释内参、外参和手眼标定分别解决什么问题，并写出坐标变换的方向。
2. **预算题**：为 20 Hz 控制回路分配采集、同步、预处理、推理、通信和执行延迟预算。
3. **设计题**：定义 RGB、关节状态和力传感器的时间同步规则，包括最大允许时间差和过期处理。
4. **排错题**：列出训练前至少五项数据质量检查，并说明其中一项失败会怎样污染策略。

下一课：[`13-robot-systems-and-safety.md`](13-robot-systems-and-safety.md)。
