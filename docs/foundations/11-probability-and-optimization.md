# 11 · 概率、统计与优化

> English contract: [Foundations overview](README_EN.md#route) · Primary references: [Probability and optimization](../SOURCES.md#11-probability-and-optimization)

> 目标：理解机器人学习中“数据有噪声、模型有不确定性、训练是数值优化”这三件事，并能读懂 VLA、世界模型和 RL 中常见的目标函数。

## 1. 为什么机器人学习离不开概率

同一个动作在真实世界中可能得到不同结果：相机有噪声，摩擦系数未知，标注也不完全准确。因此模型通常学习条件分布，而不是唯一答案：

$$
p(a_t\mid o_t,l),\qquad p(s_{t+1}\mid s_t,a_t)
$$

- `p(a_t | o_t, l)`：给定观测和语言，动作的分布。
- `p(s_{t+1} | s_t, a_t)`：给定当前状态和动作，未来状态的分布。
- 均值描述最可能结果，方差描述不确定程度。

<div class="dof-principle" role="group" aria-label="贝叶斯规则以观测更新机器人状态信念">
  <p class="dof-principle__caption"><strong>原理图 · Observation updates a belief, not a certainty</strong>：机器人先有一个由运动模型得到的先验，再用带噪观测修正它。观测可靠时后验会更集中；观测不可靠时，系统应保留更大的不确定性而不是假装“已经知道”。</p>
  <div class="dof-principle__canvas">
    <svg viewBox="0 0 860 254" role="img" aria-labelledby="bayes-title">
      <title id="bayes-title">先验、观测似然和后验概率分布</title><text class="dof-diagram-title" x="31" y="41">Belief update</text><text class="dof-diagram-math" x="31" y="65">p(z | o) ∝ p(o | z) p(z)</text>
      <rect class="dof-diagram-surface" x="22" y="86" width="212" height="132" rx="16"/><text class="dof-diagram-label" x="45" y="116">prior · motion model</text><path class="dof-diagram-line" d="M43 190 H213"/><path class="dof-diagram-violet" d="M49 190 C79 188 82 129 128 116 C174 129 177 188 208 190"/><text class="dof-diagram-note" x="45" y="210">wide: uncertain state</text>
      <path class="dof-diagram-accent" d="M252 152 H326"/><path class="dof-diagram-arrow" d="M326 152 l-11 -6 v12z"/><rect class="dof-diagram-fill-blue" x="344" y="107" width="174" height="90" rx="14"/><text class="dof-diagram-label" x="382" y="139">sensor reading</text><text class="dof-diagram-math" x="392" y="164">oₜ ± noise</text><path class="dof-diagram-accent" d="M536 152 H610"/><path class="dof-diagram-arrow" d="M610 152 l-11 -6 v12z"/>
      <rect class="dof-diagram-surface" x="628" y="86" width="212" height="132" rx="16"/><text class="dof-diagram-label" x="652" y="116">posterior · fused belief</text><path class="dof-diagram-line" d="M649 190 H819"/><path class="dof-diagram-good" d="M687 190 C711 187 715 125 734 112 C754 125 758 187 782 190"/><text class="dof-diagram-note" x="652" y="210">narrower only if evidence supports it</text>
    </svg>
  </div>
</div>

## 2. 必须掌握的概率概念

| 概念 | 机器人学习中的用途 | 常见误区 |
|:---|:---|:---|
| 条件概率 | 观测条件下预测动作或未来 | 把相关性当因果性 |
| 期望与方差 | 回报、误差和风险统计 | 只报告均值，不报告波动 |
| 高斯分布 | 连续动作、传感噪声、潜变量 | 假设所有误差都对称 |
| 交叉熵 | 离散 token 或类别动作 | 忽略类别不平衡 |
| KL 散度 | VAE/RSSM 的先验与后验对齐 | 把 KL 当对称距离 |
| 贝叶斯规则 | 由观测更新状态信念 | 混淆先验与后验 |

RSSM 中常见的目标可写成：

$$
\mathcal{L}=\lambda_o\mathcal{L}_{recon}+\lambda_r\mathcal{L}_{reward}
+\lambda_c\mathcal{L}_{continue}+\beta D_{KL}(q(z_t)\Vert p(z_t))
$$

对应代码：[`examples/dreamer_rssm.py`](../../examples/dreamer_rssm.py)。

## 3. MLE、MAP 与经验风险

- **最大似然 MLE**：寻找最能解释训练数据的参数。
- **最大后验 MAP**：在 MLE 上加入参数先验；L2 正则可视作高斯先验。
- **经验风险最小化**：对有限数据上的损失取平均。

行为克隆常用：

$$
\theta^*=\arg\min_\theta\frac1N\sum_i\|\pi_\theta(o_i,l_i)-a_i\|_2^2
$$

但低训练误差不等于闭环成功。策略会改变未来观测分布，因此必须做闭环评估。

## 4. 梯度优化的工程直觉

| 方法 | 特点 | 何时使用 |
|:---|:---|:---|
| SGD | 更新稳定、需要调学习率 | 大批量基线 |
| Adam/AdamW | 自适应学习率、收敛快 | Transformer/VLA 默认选择 |
| 梯度裁剪 | 限制异常大梯度 | RNN、长序列、RL |
| 学习率预热 | 避免训练早期不稳定 | 大模型微调 |
| 余弦退火 | 后期逐步减小步长 | 中长程训练 |

调参顺序建议：先确认数据和损失量纲，再调学习率与 batch size，最后调正则和模型规模。

## 5. 不确定性与置信区间

单个 seed 的成功率不能代表方法水平。二项成功率至少报告：成功次数、总回合数、均值和置信区间。若 20 次中成功 14 次，报告 `14/20 = 70%`，而不是只写 `70%`。

常见不确定性：

- **数据不确定性**：传感器噪声、执行随机性。
- **模型不确定性**：训练数据没有覆盖当前场景。
- **分布外输入**：新物体、新相机、新机器人形态。

对应工程动作：多 seed、数据切分、校准曲线、OOD 检测和安全回退。

## 6. 最小 NumPy 练习

```python
import numpy as np

rng = np.random.default_rng(42)
success = rng.binomial(1, 0.7, size=100)
mean = success.mean()
stderr = np.sqrt(mean * (1 - mean) / len(success))
ci95 = (mean - 1.96 * stderr, mean + 1.96 * stderr)
print(f"success={mean:.3f}, 95% CI={ci95}")
```

## 7. 检查理解

1. **概念题**：用自己的话解释先验、后验和 KL 散度，并说明 KL 为什么不是对称距离。
2. **分析题**：为什么训练损失下降不能推出闭环任务成功率提升？列出至少两个反例。
3. **统计题**：某策略在 20 个回合中成功 14 次，计算成功率与近似 95% 置信区间，并说明样本量限制。
4. **设计题**：任选 VLA、世界模型或 RL，写出输入、预测量、目标函数和至少一个数值稳定措施。

下一课：[`12-perception-and-sensors.md`](12-perception-and-sensors.md)。
