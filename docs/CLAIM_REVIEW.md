# Claim Review and Accuracy Gate · 论断审查与真实性门禁

This repository treats factual accuracy as a release gate. Passing code tests or rendering the site does not prove that every scientific statement is true. The automated claim guard prevents known regressions; primary-source review and reproducible artifacts remain mandatory.

本仓库把事实准确性作为发布门禁。代码测试通过或站点成功构建，不能证明每一条科学表述都正确。自动化论断检查只能阻止已知错误回归，不能替代一手来源核验和可复现实验证据。

## Evidence labels · 证据标签

| Label | Meaning | 中文解释 |
|:---|:---|:---|
| `source-backed` | A technical statement is supported by an adjacent primary source with the applicable scope stated. | 技术论断附近有一手来源，并明确适用条件。 |
| `reproduced` | The repository contains the command, environment, seed, raw artifact, and machine-readable metric needed to rerun or re-aggregate the result. | 仓库保留命令、环境、种子、原始产物和机器可读指标，可重新运行或聚合。 |
| `reported-aggregate` | A summary or configuration is committed, but one or more raw artifacts required for independent re-aggregation are absent. | 已提交汇总或配置，但独立重新聚合所需的部分原始产物缺失。 |
| `not-evaluated` | The implementation or interface exists, but the stated task metric has not been measured. | 实现或接口存在，但尚未测量对应任务指标。 |
| `hardware-validated` | A named hardware configuration has raw logs and a bounded acceptance result; it is never a universal safety certification. | 指定硬件配置有原始日志和有限验收结果，但不等于通用安全认证。 |

## Rules for scientific statements · 科学表述规则

1. Quantitative, causal, comparative, and superlative claims require an adjacent primary source or a committed artifact pointer.
2. A preprint must be labeled as a preprint unless an authoritative venue page confirms acceptance.
3. One run can report an observation; it cannot establish a universal data threshold or a single failure cause without controlled ablations.
4. Training loss, import success, smoke execution, simulation contact, benchmark success, and hardware success are separate evidence levels.
5. Missing raw data must remain visible. Do not reconstruct, infer, or imply an artifact that is not committed.
6. Chinese and English entry points must preserve the same evidence status and limitations.

1. 定量、因果、比较和“最强/最大”等结论必须就近链接一手来源或仓库内证据产物。
2. 未由权威会议页面确认录用的论文必须标注为预印本。
3. 单次运行只能报告观察结果；没有受控消融时，不能推出通用数据阈值或唯一失败原因。
4. 训练损失、成功导入、Smoke 执行、仿真接触、基准成功和真机成功是不同证据等级。
5. 原始数据缺失必须明确显示，不能补写、推断或暗示未提交的产物存在于仓库中。
6. 中文和英文入口必须保持相同的证据状态与限制条件。

## Review workflow · 审查流程

1. Locate the exact sentence, number, table, or diagram label.
2. Open the primary paper, official documentation, or committed raw artifact.
3. Record what the source supports and what it does not support.
4. Rewrite the statement with its conditions; attach the source beside it.
5. Run `python scripts/check_claims.py`, repository tests, strict documentation build, and local-link validation.
6. For high-impact claims, require an independent reviewer before release.

自动检查的边界与 [`VALIDATION.md`](VALIDATION.md) 一致：它能检查已知禁用表述、证据标签、路径混淆和结构契约，但不能自动证明外部论文中的每个语义解释都正确。

## Environment claims · 环境与安装论断

- Never call a simulator, driver, framework, or Python release “latest” without a dated upstream comparison.
- Keep host support, driver support, Python support, and application support as separate claims.
- An import, visible window, ROS bridge, task smoke, benchmark, and hardware run are different evidence levels.
- Installation commands for volatile stacks such as Isaac Lab must defer exact pins to the current official compatibility page.
- WSL2 development evidence cannot be promoted to hard real-time, USB reliability, or robot-safety evidence.

- 未经带日期的官方对比，不把仿真器、驱动、框架或 Python 版本描述为“最新”。
- 宿主支持、驱动支持、Python 支持与应用支持必须分别表述。
- 成功导入、显示窗口、ROS 桥接、任务 Smoke、Benchmark 与真机运行是不同证据等级。
- Isaac Lab 等易变技术栈的精确版本必须以当前官方兼容页为准。
- WSL2 开发证据不能提升为硬实时、USB 稳定性或机器人安全证据。

## Canonical sources used in the current correction · 本轮修订的一手来源

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) and the [authors' implementation](https://github.com/tonyzhaozh/act)
- [DexSim2Real arXiv record](https://arxiv.org/abs/2605.05241) — preprint status and reported experiment tables
- [Octo paper](https://octo-models.github.io/paper.pdf) and [Open X-Embodiment project](https://robotics-transformer-x.github.io/) — model interfaces, released configuration, and date-scoped dataset scale
- [SmolVLA official LeRobot documentation](https://huggingface.co/docs/lerobot/smolvla) — model and task-specific fine-tuning guidance
- [Shadow Dexterous Hand product page](https://shadowrobot.com/dexterous-hand-series/) and [Wonik Robotics Allegro resources](https://wonikrobotics.com/en/sub/support/data.php) — version-scoped hardware specifications
- [MuJoCo modeling documentation](https://mujoco.readthedocs.io/en/stable/modeling.html) — simulator and model-format behavior
