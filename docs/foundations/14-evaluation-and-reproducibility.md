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

<div class="dof-principle" role="group" aria-label="五层验证证据等级的递进关系">
  <p class="dof-principle__caption"><strong>原理图 · Evidence is cumulative, not interchangeable</strong>：验证等级从“能导入”逐层增加约束、环境真实性和可复查证据。高层证据包含更多条件，但低层通过绝不能推导出 benchmark 或真机成功。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 260" role="img" aria-labelledby="evidence-title">
      <title id="evidence-title">L1 到 L5 的验证层级</title><text class="dof-diagram-title" x="36" y="40">Validation ladder</text><text class="dof-diagram-note" x="36" y="62">each level adds a new claim that must be independently evidenced</text>
      <rect class="dof-diagram-fill-blue" x="38" y="190" width="148" height="35" rx="10"/><text class="dof-diagram-label" x="65" y="213">L1 · Import</text>
      <rect class="dof-diagram-fill-blue" x="160" y="154" width="148" height="35" rx="10"/><text class="dof-diagram-label" x="190" y="177">L2 · Smoke</text>
      <rect class="dof-diagram-fill-violet" x="282" y="118" width="148" height="35" rx="10"/><text class="dof-diagram-label" x="300" y="141">L3 · Deterministic</text>
      <rect class="dof-diagram-fill-violet" x="404" y="82" width="148" height="35" rx="10"/><text class="dof-diagram-label" x="424" y="105">L4 · Benchmark</text>
      <rect class="dof-diagram-fill-good" x="526" y="46" width="148" height="35" rx="10"/><text class="dof-diagram-label" x="549" y="69">L5 · Hardware</text>
      <path class="dof-diagram-accent" d="M186 207 H208 V171 H160 M308 171 H330 V135 H282 M430 135 H452 V99 H404 M552 99 H574 V63 H526"/><text class="dof-diagram-note" x="704" y="86">more realism</text><text class="dof-diagram-note" x="704" y="108">more risk</text><text class="dof-diagram-note" x="704" y="130">more provenance</text>
    </svg>
  </div>
</div>

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
