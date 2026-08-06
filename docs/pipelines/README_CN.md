# 具身智能 Pipeline 总览

这里把仓库中的知识章节整理为端到端工程闭环。每条路线都明确前置知识、输入、阶段、产物、指标与晋级门槛；命令统一登记在可机器校验的 [`pipelines/manifest.json`](../../pipelines/manifest.json) 中。

## 快速开始

```bash
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --show vla-policy
python scripts/run_pipeline.py --run vla-policy --dry-run
python scripts/run_pipeline.py --run vla-policy
```

先运行 smoke 模式确认接口和数据流，再在算力、数据和记录条件满足后使用 `--full`。`--dry-run` 只展示命令，不实际训练。

## 八条主线

| 方向 | 工程闭环 | 当前证据 | 文档 |
|---|---|---|---|
| 仿真与数据 | 任务定义 → 仿真器 → 专家策略 → 轨迹 → 质量检查 | 已有 smoke test | [进入](01-simulation-data.md) |
| VLA 策略 | 多模态数据 → 训练 → 闭环评估 → 消融 | 教学基线可 smoke test | [进入](02-vla-policy.md) |
| 世界模型 | 转移数据 → 动力学模型 → rollout → 规划 | 模型可 smoke test，规划独立验证 | [进入](03-world-model-planning.md) |
| RL 后训练 | MDP → 奖励 → PPO → 评估 → 回归检查 | 教学基线可 smoke test | [进入](04-rl-post-training.md) |
| 机器人基础模型 | 观测协议 → 模型适配 → 动作块 → 安全层 | Mock 接口已验证 | [进入](05-rfm-cross-embodiment.md) |
| 具身推理 | 指令 → 任务计划 → 技能 → 反馈 → 重规划 | 规则规划接口已验证 | [进入](06-embodied-reasoning.md) |
| Sim-to-Real | 仿真 → 鲁棒性 → HIL → 影子模式 → 受控部署 | 工程门禁已文档化 | [进入](07-sim-to-real.md) |
| 灵巧手重定向 | 关键点 → 几何 → IK/优化 → 平滑 → 评估 | 合成输入可 smoke test | [进入](08-dexterous-retargeting.md) |

English navigation: [Pipeline Catalog](README.md).

## 证据标签

- **smoke-tested**：仓库内有轻量可执行路径，可检查基本连通性。
- **interface-tested**：无需真实权重或硬件即可检查协议、形状和适配器。
- **documented**：门禁与步骤已明确，但真实系统不能由一条本地命令代替。
- **experimental**：探索路线，不应直接当作已验证基线。

Smoke test 通过只代表“管线接通”，不代表研究效果达标，更不代表可直接上真机。任何结果都应同时保存配置、随机种子、硬件、权重版本和结果文件。
