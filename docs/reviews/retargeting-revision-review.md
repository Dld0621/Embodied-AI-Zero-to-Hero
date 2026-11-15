# Retargeting 旧教材 H02–H06 修订复核

日期：2026-09-05。范围严格限定为旧教材的 [方法分类](../02-retargeting-taxonomy.md)、[人手映射](../03-human-hand-to-robot-hand.md)、[优化方法](../04-optimization-methods.md)、[学习式方法](../05-learning-based-methods.md) 与 [评估指标](../06-evaluation-metrics.md)。本记录只说明已修的具体段落及离线证据，不代表五篇全文、整本教材或真实机器人系统已获正确性/安全认证。

## 复核结论

| 编号 | 已修段落 | 本轮证据 | 仍未验证 |
| --- | --- | --- | --- |
| H02 | 分段线性映射改为阈值连续、奇对称；有 bounds 的 `least_squares` 改用 `trf`，并检查可行初值与求解状态 | 从 Markdown 提取实际函数；阈值两侧连续性与奇对称通过；一维目标在上、下界之外时，解分别停在对应边界 | 特定机器人运动学、收敛率、实时耗时、真机 |
| H03 | 手腕中心化与手掌旋转变换分开；屈曲角伸直为 0；零长度骨段与退化手掌基显式报错；MuJoCo `jnt_qposadr` / `jnt_dofadr` 分开；O10 固定映射降级为待模型核对的教学特征 | 从 Markdown 提取实际函数；旋转+平移后的手掌局部坐标保持不变；0°/90°屈曲与退化输入通过；不同 qpos/qvel 地址的写入回归通过 | O10 具体关节/actuator 名称、顺序、方向、传动、限位、控制接口和硬件 |
| H04 | 纠正 Jacobian 转置与伪逆的奇异值效应；明确自适应阻尼在阈值处及阈值外返回 0；指尖间距改称代理约束；移除 CMA-ES 有限预算全局最优保证 | 从 Markdown 提取 `dls_ik` 与 `adaptive_damping`；近奇异方向的 DLS 增益及阈值两侧行为通过 | 全 link/geom 自碰撞、整条路径碰撞、特定模型收敛与性能 |
| H05 | 每指输入统一为 `5×3=15`；必需参数移到默认参数前；显式解包 `B,T,C,H,W`；时序 Transformer 前加入正弦时间位置编码；补齐 loss 的可选状态/限位参数 | 所有 Python fence 通过 AST 语法检查 | 环境无 PyTorch，MLP 前向形状与时间编码形状两项均为 **skip / 未执行**；未训练、未下载权重、未评估精度或泛化 |
| H06 | SO(3) 矩阵对数 Frobenius 范数与旋转向量二范数统一；jerk 三阶差分、`dt`、均值和单位统一；互相关输入改为信号并定义正 lag；空列表返回 `None`；retargeting 函数调用耗时与端到端 latency 分开 | 从 Markdown 提取实际函数；60° 旋转、三次轨迹 jerk、4-sample 正延迟及空数据集汇总通过 | 相机曝光到执行器/机器人响应的同步端到端时间戳、真实轨迹统计、真机任务成功率 |

## 回归边界

新增 `tests/test_legacy_teaching_corrections.py`，只执行从上述五篇 Markdown 提取的选定函数或代码块；没有复制另一套“同名正确公式”来自证。该测试不启动 viewer，不访问网络，不训练模型，不下载权重，不连接控制器或硬件。

定向结果：`15 passed, 2 skipped`。已执行的 15 项包括 AST 语法/接线检查与 NumPy/SciPy 数值回归；两项 skip 都由本地未安装 PyTorch 导致，分别是 `LandmarkToJointNet` 前向形状和 `SinusoidalTimeEncoding` 形状/位置差异，它们只是已编写的形状检查，不得记作已复现成功。

## 原始来源与推导依据

- SciPy `least_squares` 官方参考明确：`lm` 不处理 bounds，`trf` 适合有边界问题：[SciPy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。
- MuJoCo 的 `jnt_qposadr` 是 `qpos` 起始地址，`jnt_dofadr` 是 `qvel` 起始地址；`nq` 与 `nv` 也不必相同：[MuJoCo API types](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html)。
- Transformer 原论文在无循环/卷积的自注意力输入中显式加入位置编码以提供序列顺序：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。
- `numpy.correlate(a, v)` 的定义是 $c_k=\sum_n a_{n+k}\overline{v_n}$，且文档提醒互相关存在相反的另一种符号约定：[NumPy API](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html)。
- SO(3) 矩阵对数输出的是旋转指数坐标的反对称矩阵表示；向量值需通过 vee 映射取得：[Modern Robotics 3.2.3](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-3-exponential-coordinates-of-rotation-part-2-of-2/)。
- CMA-ES 官方材料把它描述为随机、无梯度优化器，并给出有限配置下到达目标的概率小于 1 的例子，不能据此承诺有限预算全局最优：[CMA-ES source and practical notes](https://cma-es.github.io/cmaes_sourcecode_page.html)。

其余连续性、屈曲角、奇异值增益、三阶有限差分和空集合处理，是对页面公式/代码的直接代数推导与离线数值回归，不冒充外部实验。

## 状态用语

五篇页首只标记“已修订的具体 H02–H06 段落”，同时保留未训练、未完整实验验证、未接硬件和模型合同待核实的边界。原 [内容审查交接](content-audit-handoff.md) 中的 F002/F003、H01、H07、H08 及其他未读/未复核内容均不在本轮范围，也没有被本记录关闭。
