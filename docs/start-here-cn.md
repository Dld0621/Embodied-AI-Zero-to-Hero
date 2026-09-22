# 从这里开始：第一小时到专家路线

**[English](start-here.md) · 简体中文**

这不是一份“把所有文章从头读到尾”的清单。你要循环完成：理解 → 构建 → 破坏 → 测量 → 解释 → 评审。学习时间只是排期参考，只有产物和门禁决定是否晋级。

## 第一个 30 分钟

如果连“向量、误差、采样、策略”还讲不清，先打开[逐点图解](knowledge-atlas/index.md)，一次只学一个小点：读直觉 → 手算 → 对照图 → 独立回答自测。每页都注明前置知识；不要把展开答案当作通过验收。

还没有安装环境？先用 [交互实验室](learning-lab-cn.md)完成“坐标变换 → 反馈控制 → 成功率区间”三项短实验。每项先写预测、再改一个参数、最后解释结果并导出记录。它们建立直觉；下方可运行 Pipeline 与正式课程验收仍需完成。

```bash
git clone https://github.com/Dld0621/Embodied-AI-Zero-to-Hero.git
cd Embodied-AI-Zero-to-Hero
python -m pip install numpy
python scripts/run_curriculum.py --validate
python scripts/run_curriculum.py --diagnose --lang zh
python scripts/run_pipeline.py --run simulation-data
```

你应该得到一份课程合同验证结果、一组证据自测问题和一次合成 Pipeline 输出。它们只能证明入口可执行，不能证明你已经掌握课程或模型具有真实任务能力。

## 第一个 60 分钟挑战

1. 找到 Pipeline 的输入、阶段、指标和输出文件。
2. 改变一个参数并预测结果，再运行验证。
3. 故意制造一个失败，例如错误 Shape、非法范围或缺失字段。
4. 保存命令、环境、随机种子、结果与失败解释。
5. 用[实验卡模板](../learner/templates/experiment-card.md)写出第一张实验卡。

如果你只会复制命令，却不能解释输入输出、预测改变或定位失败，请从 M00 开始。

## 用证据定位起点

运行：

```bash
python scripts/run_curriculum.py --diagnose --lang zh
```

对每个模块只回答一个问题：**我能否展示要求的产物，并让另一个人按门禁复核？**

- “上过这门课”“看过论文”“代码跑过”都不是通过证据。
- 没有证据时标记 `not_started` 或 `in_progress`，不要凭背景自动跳级。
- 已有经验的人可以跳过学习材料，但不能跳过验收。

## 按背景选择第一重点

| 背景 | 先保留的能力 | 最容易漏掉的部分 | 建议起点 |
|---|---|---|---|
| 完全新手 | 能运行 Python，理解数组 Shape | 环境、单位、坐标系、实验记录 | M00 |
| 机械/控制 | 运动学、动力学、控制直觉 | Python、数据契约、学习评估 | M00 诊断后进入 M05–M07 |
| CS/ML | 编程、训练与模型实现 | 坐标系、接触、闭环控制、安全 | M02–M04 |
| 机器人研发 | 系统集成、仿真或真机经验 | 匹配基线、泄漏、消融和不确定性 | M05、M11 |

## 生成你的路线

查看目标：

```bash
python scripts/run_curriculum.py --list-goals --lang zh
```

以每周 8 小时生成全栈路线：

```bash
python scripts/run_curriculum.py \
  --plan full-stack-expert \
  --hours-per-week 8 \
  --lang zh
```

如果 M00、M01 已有通过证据，可只用于排期地标记：

```bash
python scripts/run_curriculum.py \
  --plan full-stack-expert \
  --completed M00,M01 \
  --hours-per-week 8 \
  --lang zh
```

`--completed` 不验证证据。正式记录应初始化为：

```bash
python scripts/run_curriculum.py \
  --init-progress learner/progress.json \
  --goal full-stack-expert \
  --learner your-name
```

模块标记为 `passed` 时，必须填写实际存在的证据、评审者和评审日期。检查记录：

```bash
python scripts/run_curriculum.py \
  --report-progress learner/progress.json \
  --lang zh
```

## 每个模块的工作循环

1. **诊断：**确认前置节点与已知缺口。
2. **学习：**阅读主文档并亲手完成推导或代码。
3. **构建：**产生模块要求的最小产物。
4. **破坏：**至少注入一个故障、扰动或反事实条件。
5. **测量：**保存原始指标、配置、日志与失败案例。
6. **解释：**区分模型、数据、接口、控制与任务定义问题。
7. **评审：**按[统一评估规范](assessment-cn.md)决定通过、返工或停止。

## 从“小白”到“专家”意味着什么

| 阶段 | 你必须证明的能力 |
|---|---|
| L0 | 别人能够复现你的实验，你能修复一个故意制造的错误 |
| L1 | 你能推导并数值验证公式，识别坐标系和数值陷阱 |
| L2 | 你能让感知—状态—控制—仿真形成有边界闭环并定位故障 |
| L3 | 你能构建数据、策略、RL、世界模型和规划基线，区分离线与闭环指标 |
| L4 | 你能端到端负责一类任务，处理扰动、恢复与分阶段失败 |
| L5 | 你能独立复现基线、提出可证伪假设、完成受控对比并通过独立评审 |

专家不是“读完全部文档”，也不是“训练 Loss 下降”。专家级毕业要求见[三级 Capstone](capstone-cn.md)。

## 卡住时怎么做

- 环境失败：查看[环境排错](setup/troubleshooting.md)，保留第一条错误而不是盲目重装。
- 不知道缺哪个概念：运行 `python scripts/run_knowledge_map.py --path-to <node> --lang zh`。
- 系统能跑但任务失败：按阶段记录误差，不要直接扩大模型或重新训练。
- 想上真机：先完成 Sim-to-Real 与安全门禁；仿真通过不等于获得真机授权。
