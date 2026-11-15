# Robot Foundation Models：模型、适配器与执行证据

**更新：2026-11-15。** 本页说明可复用模型如何接入完整机器人系统。模型家族与最新发布见[研究导读](30-vla-research-guide.md)，逐步课程见[第 09 课](learning-core/09-vla-systems.md)。

## 1. 为什么“加载模型”只完成一小步？

一个机器人基础模型可以在不同任务或机器人数据上预训练，再适配目标任务。它不必具备语言输入，也不必预测视频。VLA 强调视觉、语言到动作的条件策略；世界模型强调结果或状态预测；规划器、奖励模型和低层控制器承担不同职责。

以红杯任务为例：

```text
任务指令 + 带时间戳的图像/状态
  → checkpoint 专属预处理
  → 策略预测动作块
  → 反归一化与动作语义检查
  → 目标机器人坐标/运动学转换
  → 时效、限位、速度、碰撞与停止策略
  → 控制器执行 → 记录实际响应 → 新观测
```

这是系统职责图。每一条箭头都需要实现和测试。统一 Python 接口能够复用日志与调度代码，不能消除相机要求、动作空间和控制器差异。

## 2. 仓库中有哪些真实的接口？

| 层 | 仓库入口 | 阅读时核对 |
|---|---|---|
| 观测容器 | [observation_schema.py](../examples/robot_foundation_models/common/observation_schema.py) | 图像键、dtype、状态和时间戳 |
| 动作容器 | [action_schema.py](../examples/robot_foundation_models/common/action_schema.py) | horizon、维度、action_type、control_frequency |
| 模型协议 | [model_interface.py](../examples/robot_foundation_models/common/model_interface.py) | reset 与 predict_action 的调用边界 |
| 本体适配 | [embodiment_adapter.py](../examples/robot_foundation_models/common/embodiment_adapter.py) | 转换是否真的实现、坐标/单位是否匹配 |
| 动作检查 | [safety_filter.py](../examples/robot_foundation_models/common/safety_filter.py) | 所有检查是否串联、停止是否符合控制模式 |
| 教学环境 | [unified_pushcube_env.py](../examples/unified_pushcube_env.py) | 平面推块合同和成功判据 |

容器中相同维度的 `actions` 可能表示关节目标、末端位姿或增量。`action_type` 是需要验证的标签，不是转换算法。`control_frequency` 是元数据，只有实际调度循环遵守它才有执行意义。一个未经校准的 `confidence` 浮点数也不是安全保证。

## 3. 模型能力与本仓库集成状态分开看

| 对象 | 原始项目说明什么？ | 本仓库应如何使用？ |
|---|---|---|
| [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) | VLM 条件的流匹配动作专家，支持 LeRobot 微调 | 先读[版本固定手册](28-smolvla-gpu-finetuning-runbook.md)，区别本地 mock、小型模型和官方权重 |
| [OpenVLA](https://github.com/openvla/openvla) | 7B、离散动作 token 的预训练策略 | 本地适配器仍有动作标签与反归一化开放问题，不能按标签直接下发硬件 |
| [OpenVLA-OFT](https://openvla-oft.github.io/) | 针对 OpenVLA 的动作表示、并行解码和微调方案 | 不是给原 OpenVLA 调用改个名字；需要对应实现与 checkpoint |
| [Octo](https://octo-models.github.io/) | 语言/目标图像条件的通用扩散策略 | 作为跨数据与模块化适配阅读对象，按官方实现核对输入输出 |
| [openpi](https://github.com/Physical-Intelligence/openpi) | π0、π0-FAST、π0.5 等公开实现入口 | 官方开放范围与公司后续模型发布分开记录 |
| GR00T、Gemini Robotics 等 | 具身模型的其他系统设计与官方评测 | 访问条件、代码/权重和本地适配逐项核查，见[更新导读](30-vla-research-guide.md) |

模型规模不是选择顺序。先问：数据提供哪种动作、是否真正需要语言、目标平台允许多大观测年龄、能否建立匹配预算的基线。使用[算法选择器](../scripts/select_vla_wam_algorithm.py)整理条件，随后回到实际数据做验证。

## 4. “统一接口”应该验证的五个合同

### 观测合同

相机名、RGB/BGR、图像尺寸、裁剪/缩放、本体状态顺序、缺失输入 mask 和时间戳都要记录。多相机合并不是简单地拼字典；各帧必须满足定义的同步容差。

### 动作合同

逐通道记录物理量、单位、坐标系、绝对/增量、旋转表示和夹爪极性。反归一化必须使用训练所对应的统计、mask 和数据集 key。详见[动作表示](24-action-representation-and-tokenization.md)。

### 时间合同

模型输出 H 步不代表全部执行。写清 K、动作间隔、推理期间保持行为、过期块的处理、重置时缓存清除。`select_action()` 可能从内部动作队列取一步，调用次数未必等于完整模型前向次数。

### 执行合同

验证无穷/NaN、维度、限位、逐步速度、碰撞和停止处理。绝对位置命令全零可能让所有关节移动到零位；速度命令全零也不意味着所有系统都会安全制动。必须按控制器和驱动器定义验证。

### 评测合同

记录模型/数据版本、初态分布、种子、试验数、成功/失败/超时/干预和耗时。把“模型返回浮点数组”“仿真任务成功”“真机完成任务”作为不同证据层。

## 5. 从无 GPU 练习到模型微调

1. 运行 `python examples/vla_contract_lab.py --json`，检查单位映射、量化、padding mask 和语言对照。这是合成数据教学练习。
2. 按[数据指南](21-vla-dataset-organization.md)审查自己的 episode，再交给固定版本的真实 loader，核对抽帧和动作窗口。
3. 按[微调手册](28-smolvla-gpu-finetuning-runbook.md)记录真实配置、可训练参数和梯度；运行基线及语言/视觉消融。
4. 按[部署指南](20-vla-deployment-guide.md)测量观测到执行的时间分布；先在有明确合同的离线/仿真环境逐层检查。

已有 PushCube 汇总见[验证政策](VALIDATION.md)和[基准报告](benchmark_report.md)。本次文档与教学交互更新不构成新的官方模型 GPU 训练或真机成功记录。

## 6. 已知执行问题保留可见

[历史独立审查](reviews/content-audit-handoff.md)中的 F002/F003 涉及 OpenVLA 适配器动作语义与 SafetyFilter 检查/停止逻辑。本页重写不等于修复这些控制实现。接入工作的验收必须覆盖开放项的回归测试与目标控制模式，不能因接口测试通过就跳过它们。

## 7. 迁移到灵巧手时，多了什么？

夹爪开合标量不能表示整只手的多关节动作。人手视频或 GRAB 类数据提供的是人体/物体运动与交互信息，不能直接当目标机器人手的可执行关节或力标签。还需要形态映射、机器人运动学、接触对应、可达性和动作时序。

同一策略可以在高层指定目标或抓持阶段，低层由重定向、跟踪与触觉反馈实现；也可以训练直接输出手关节的策略。两条路线都要独立评估抓起、稳定载荷、搬运、放置与释放。结合[接触与残差课](learning-core/08-contact-retargeting.md)，说明修正到底加在目标、关节命令、力请求还是策略残差上。
