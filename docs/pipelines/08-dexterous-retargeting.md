# 灵巧手重定向 / Dexterous Hand Retargeting

## English contract

- **Objective:** map human landmarks or pose parameters to robot-hand joints while respecting morphology, limits, temporal continuity, latency, and task-relevant geometry.
- **Inputs:** calibrated landmarks, coordinate frames, robot kinematics, joint limits, objective weights, timestamps, and optional contact/task labels.
- **Stages:** calibration → landmark/pose representation → geometric objective → constrained optimization → filtering → latency/quality evaluation.
- **Acceptance:** report retargeting error, limit violations, temporal jitter, solver failures, and latency across seeds and morphologies; synthetic IK success is not grasp success.
- **Evidence:** the included vector and position methods are synthetic smoke tests. Real hand tracking, contact quality, and robot execution require separate validation.

## 目标与边界

把人手关键点或姿态映射到机器人手关节，同时满足几何相似、关节限制、时序稳定和实时性。本仓库的统一 smoke test 使用合成关键点；真实相机、标定、接触与真机控制需要额外验证。

## 前置知识与输入

- [SE(3) 与旋转](../foundations/06-se3-and-rotation.md)、[FK/Jacobian/IK](../foundations/07-fk-jacobian-ik.md)
- [IK 与 Retargeting](../01-what-is-ik-retargeting.md)
- [完整教程](../../tutorials/05-complete-pipeline/README.md)
- 输入：带置信度和时间戳的手部关键点、左右手标记、相机/手腕变换、机器人 URDF/MJCF、关节顺序与限位。

## Pipeline

| 阶段 | 关键动作 | 输出/检查 |
|---|---|---|
| 1. Perception | 检测关键点并过滤低置信帧 | landmarks + confidence |
| 2. Canonicalization | 相机系转手腕/掌心局部系，尺度归一 | canonical hand pose |
| 3. Correspondence | 定义人手与机器人指尖/骨段映射 | task-space targets |
| 4. Retargeting | rule-based、IK 或向量优化 | robot joint targets |
| 5. Constraints | 关节限位、速度/加速度与自碰检查 | feasible command |
| 6. Temporal filter | EMA/低通/预测补偿，处理丢帧 | smooth command stream |
| 7. Evaluation | 几何误差、违规率、抖动和时延 | 分场景报告 |
| 8. Deployment gate | 仿真回放 → HIL → 受控真机 | 安全与回滚记录 |

## 运行与产物

```bash
python scripts/run_pipeline.py --run dexterous-retargeting
python scripts/run_pipeline.py --run dexterous-retargeting --full
```

入口：[complete_retargeting_pipeline.py](../../examples/complete_retargeting_pipeline.py)。它比较 rule-based 与 vector optimization 的合成输入路径；不要把其成功率解释为真实手部数据或真实机器人结果。

## 验收门槛

- 明确坐标系、单位、左右手镜像和关节顺序，并做已知姿态单元测试。
- 分别报告指尖/方向误差、关节限位违规、速度/加速度违规、抖动和端到端延迟。
- 对遮挡、低置信、突变、手离开视野和通信中断定义安全退化。
- 真机前先回放保存序列并在仿真/HIL 中检查自碰与极限姿态。

常见失败：相机系直接当机器人基座系、尺度未归一、只优化单帧几何、平滑造成过大相位延迟、丢帧时重复旧动作。

下一步：完成几何与时序门槛后，进入[灵巧手抓取与精细操作 Pipeline](11-dexterous-manipulation.md)，继续验证接触建立、物体抬升、保持、滑移、扰动恢复和任务成功。重定向成功不能替代这些证据。
