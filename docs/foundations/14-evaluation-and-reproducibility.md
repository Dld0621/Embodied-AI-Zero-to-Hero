# 14 · 评估与可复现性

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Evaluation and reproducibility](../SOURCES.md#14-evaluation-and-reproducibility)

> 目标：把“能运行”升级为“能比较、能复查、能复现”，并为每条 pipeline 建立统一证据等级。

## 1. 五层验证

| 等级 | 证明了什么 | 没有证明什么 |
|:---|:---|:---|
| L1 Import | 文件与依赖可导入 | 算法正确 |
| L2 Smoke | 最小路径能结束 | 指标有效 |
| L3 Deterministic | 固定 seed 可重复 | 泛化能力 |
| L4 Benchmark | 同一任务可公平比较 | 真机可用 |
| L5 Hardware | 指定硬件条件下验证 | 其他硬件也安全 |

状态必须写成“在哪个任务、多少回合、哪个 commit、什么硬件验证”，不能只写“已完成”。

## 2. 数据切分

- 按 episode 或场景切分，不能随机拆连续帧。
- 测试集不参与归一化统计、早停或超参数搜索。
- 跨物体、跨背景、跨相机和跨 embodiment 分别报告。
- 保留固定的 smoke fixture，避免每次 CI 下载大型数据。

## 3. 指标矩阵

| 层 | 核心指标 | 辅助指标 |
|:---|:---|:---|
| 感知 | 检测/姿态误差 | 延迟、丢帧率 |
| 策略 | 闭环成功率 | 动作误差、轨迹长度 |
| 世界模型 | 多步预测误差 | reward/continue 误差、校准 |
| RL | 样本效率、最终回报 | 方差、崩溃次数 |
| 控制 | 跟踪误差 | 抖动、饱和比例 |
| 安全 | 违规率 | 过滤次数、急停次数 |

成功定义必须在运行前固定。例如 PushCube 的成功条件是目标方块进入目标区域，而不是“看起来靠近”。

## 4. 实验清单

每个结果至少保存：

```yaml
experiment:
  git_sha: <commit>
  command: <exact command>
  seed: 42
  environment: pushcube-dual-v1
  dataset: <name + version + split>
  checkpoint: <path + hash>
  hardware: <CPU/GPU/robot>
  episodes: 100
  metrics: <machine-readable result file>
```

项目基准入口见 [`benchmarks/run_benchmark.py`](../../benchmarks/run_benchmark.py) 和 [`BENCHMARK.md`](../../BENCHMARK.md)。

## 5. 公平比较

- 使用相同环境版本、初始状态集合、时间预算和成功定义。
- 报告所有方法的训练数据量和预训练差异。
- 超参数搜索预算应一致或明确披露。
- 失败和 N/A 不能替换成 0；0 表示已经评估且未成功。
- 教学规模实验必须标注为 teaching-scale，不外推为 SOTA。

## 6. Ablation 与失败分析

VLA 至少比较正确语言、打乱语言和无语言；可从 [`examples/unified_pushcube_vla.py`](../../examples/unified_pushcube_vla.py) 的语言消融路径开始。世界模型比较 posterior 与 prior；RL 比较随机/专家/BC 初始化；安全层比较过滤前后违规率。

失败分析按类别统计，而不是只挑视频：感知失败、目标选择错误、接触失败、控制超调、超时、安全拒绝和环境异常。

## 7. 检查理解

1. **证据题**：区分 import、smoke test、deterministic test、benchmark 和 hardware validation 各自能证明什么。
2. **数据题**：为连续机器人轨迹写出不会发生帧级泄漏的 train/val/test 切分方案。
3. **复现题**：为一个结果写出最小实验清单，至少包含命令、seed、commit、数据、checkpoint、硬件和回合数。
4. **报告题**：解释 0、N/A、planned 和 external 的区别，并各写一个正确使用场景。

完成基础层后，进入统一方向入口：[`../pipelines/README.md`](../pipelines/README.md)。
