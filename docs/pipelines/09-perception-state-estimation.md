# 多模态感知与状态估计 / Multimodal Perception and State Estimation

## English contract

- **Objective:** convert calibrated, timestamped sensor streams into a task-relevant state estimate with uncertainty and health signals.
- **Inputs:** camera, depth, proprioception, force/tactile, IMU or odometry streams; intrinsics, extrinsics, frame conventions, timestamps, and ground truth where available.
- **Stages:** sensor health → calibration → synchronization → preprocessing → representation → fusion/state estimation → uncertainty → task interface → validation.
- **Acceptance:** report calibration error, inter-sensor skew, missing/stale data, estimation error, uncertainty calibration, latency, and behavior under sensor dropout.
- **Evidence:** the repository includes a deterministic synthetic smoke test for reprojection, timestamp health, scalar sensor fusion, and uncertainty coverage. It does not claim a reproduced camera, SLAM, task-level, or real-robot benchmark.

## 目标与边界

将多源传感器数据变成带时间戳、坐标系、置信度和健康状态的任务级观测。目标不是“模型能输出一个框”，而是下游策略能够判断这份状态是否足够新、足够准、可以安全使用。

当前状态为 **smoke-tested（合成数据）**：固定种子的教学脚本检查小型针孔投影夹具、消息延迟/同步、标量 Kalman 融合与 `2σ` 覆盖率，并写出机器可读 JSON。它不代表真实相机、SLAM、任务级感知或真机性能。

## Quick smoke / 快速验证

```bash
python scripts/run_pipeline.py --run perception-state-estimation
```

默认产物为 `results/pipelines/perception_state/smoke/metrics.json`。固定夹具使用像素（重投影误差）、毫秒（跨传感器 skew）和米（合成一维位置 RMSE）；门禁还检查 stale rate 与 `2σ` 覆盖率。`--check` 失败会返回非零退出码，因此可直接进入 CI。

| 本地 smoke 能证明 | 不能据此声称 |
|---|---|
| 标定 sanity check、时间戳健康、融合更新、uncertainty 字段和 JSON 产物连通 | 真实标定质量、3D 检测/跟踪精度、SLAM 漂移、闭环任务成功或真机安全 |

## 前置知识与输入

- [坐标变换](../foundations/05-coordinate-transform.md)、[感知与传感器](../foundations/12-perception-and-sensors.md)
- [机器人系统与安全](../foundations/13-robot-systems-and-safety.md)、[评估与复现](../foundations/14-evaluation-and-reproducibility.md)
- 输入：传感器消息、时间戳、内外参、frame tree、单位约定、期望输出 schema、可选真值和故障注入配置。

## Pipeline

| 阶段 | 关键动作 | 产物 / 门禁 |
|---|---|---|
| 1. Sensor health | 检查频率、时间戳、范围、饱和与丢包 | 每传感器健康报告 |
| 2. Calibration | 求相机内参、传感器外参和坐标系链 | 标定文件与重投影/对齐误差 |
| 3. Synchronization | 按消息时间而非到达时间对齐 | skew、drop 与 stale 统计 |
| 4. Preprocessing | 去畸变、滤波、归一化和无效值处理 | 可复现预处理配置 |
| 5. Representation | 形成 image/depth/point/state/contact token | 显式 shape、dtype、frame、unit |
| 6. Fusion / estimation | 融合观测并估计 pose、velocity、object/contact state | 状态与 covariance / confidence |
| 7. Task interface | 转成策略或规划器需要的 canonical observation | schema 与 freshness 检查 |
| 8. Validation | 留出场景、遮挡、延迟和传感器失效测试 | 误差、延迟、置信度与失败报告 |

## 验收门槛

- 内参/外参和坐标系约定必须版本化；frame 与单位不能靠变量名猜测。
- 同步窗口、队列长度和丢弃策略必须记录；“大致同时到达”不能替代时间戳检查。
- 误差按传感器、场景和状态变量分别报告，同时保留均值、分位数和最坏情况。
- 置信度或 covariance 必须接受校准检查；低置信度要触发降级、保持或停止策略。
- 通过离线感知指标不等于闭环任务成功，二者必须分别验证。

## 主要失败模式

内外参漂移、frame 方向错误、消息时间与到达时间混用、训练/测试预处理不一致、遮挡后沿用陈旧状态、融合器过度自信、只报告检测精度而不报告闭环延迟。

权威入口：[OpenCV Camera Calibration](https://docs.opencv.org/5.0/py_tutorials/py_calib3d/py_calibration/py_calibration.html) · [ROS 2 message_filters](https://docs.ros.org/en/ros2_packages/rolling/api/message_filters/message_filters.html) · [robot_localization](https://docs.ros.org/en/noetic/api/robot_localization/html/index.html)
