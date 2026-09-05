# 内容正确性审查记录

状态：**范围清单全文阅读已完成，已确认问题仍有开放项**。审查日期：2026-09-05。基准提交：`58c1f546c8a27160000b076550eb65a6aa9c9d9d`。GPT-6 Astra 独立主审中断后，GPT-5.6 Sol 备用审查者补完剩余 19 篇；不是额度暂停状态，也不代表全书正确。最新覆盖、修订和未修项见 [审查交接表](content-audit-handoff.md)。

## 结论边界

本轮已确认多项内容错误，并对修正后的具体段落复核；这不是“全书无错”或安全认证。最重要的未修项是 OpenVLA 本地动作接口与 SafetyFilter 控制代码（F002/F003）。只增加文档警告不等于修复执行逻辑，不应连接真实硬件。

指定双语 README 和 docs 原始 Markdown 基线共 115 文件、31,615 行；另列 20 个支持文档。作者知识 JSON 为 3 文件、201 小点（65/68/68）。生成的 `docs/knowledge-atlas/` 是副本，不重复计算科学阅读量。另补 `knowledge/atlas/README.md` 的清单项。第三方源码、预训练文件和历史 changelog 不作逐行科学审阅声明。

## 当前覆盖与方法

逐文件及逐 atom 状态见 [机器可读清单](../../knowledge/content-audit.json)。135/135 份源 Markdown 均有全文阅读记录；3 份 JSON 的 201/201 小点教学字段全读。新增 19 篇的实际读取版本、行数与原始来源见 [补充独立审查](remaining-source-review.md)。这是主代理与多名子代理的合计，不是独立主审一人复核所有终稿；阅读不等于每条引用已核实或实验已复现。原基线 hash 与实际读取 hash 分开保存，之后的修订不自动计作再次独立审查。

| 审查层 | 实际所做 | 不支持的结论 |
|---|---|---|
| 机器结构 | 作者 JSON、ID/字段合同、生成关系和定向测试 | 科学内容全对 |
| 逐文阅读 | 实际打开的全文及 201 个小点教学字段，记录覆盖缺口 | 尚未打开的文件已读 |
| 数值复算 | 全部 atom 算例逐点推理，选定 NumPy/SciPy/MuJoCo 例与安全夹具执行 | 全部代码已执行 |
| 原始来源 | 对发现和高风险时效性论断打开官方文档/原论文；部分只核摘要或指定小节 | 全部引用、论文实验均复核 |
| 历史实验 | 对照已提交汇总、配置与实际文件存在性 | 本轮重新训练/重跑或真机成功 |

未使用浏览器、localhost、CUA、GitHub 批量 Markdown 渲染 API，也未部署网站。可用解释器有 NumPy/SciPy/MuJoCo，但无 PyTorch；没有安装训练依赖或接通硬件。

## 发现清单

编号和严重程度为本轮审查分类。位置主要按基线段落或行号，修订后行号会移动；精确历史位置和来源保留在 JSON。`段落复核` 仅指本条修正，不代表整文件所有事实均获外部验证。

### F001 · P2 · NumPy 与 Python list 的切片语义

- 位置：`knowledge/atlas/foundations.json`，numpy-copy-aliasing.worked_example。
- 问题：Python list 切片不会像 ndarray 基础切片那样共享元素存储。
- 修正/状态：已改成显式 np.array，并区分 list 浅复制。 当前记录：`fixed-rechecked`。
- 证据：[原始来源](https://numpy.org/doc/stable/user/basics.copies.html)。

### F002 · P1 · OpenVLA 动作语义与本地适配器默认标签

- 位置：`docs/24-action-representation-and-tokenization.md`，72,81-89,101,318-319,371-373; docs/23-robot-foundation-models.md:109-113; docs/25-cross-embodiment-adaptation.md:239; examples/robot_foundation_models/openvla/inference.py:60,71,138-147。
- 问题：模型名称、RLDS/LeRobot 格式不能决定绝对关节角或末端增量。vanilla OpenVLA 的末端控制输出被本地适配器贴上 joint_position 标签，未转换也未验证反归一化合同。
- 修正/状态：文档已澄清 checkpoint/数据/控制器合同；源码仍待单独修复。 当前记录：`documentation-fixed-code-open`。
- 证据：[原始来源](https://arxiv.org/html/2406.09246v3)；[原始来源](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json)。

### F003 · P1 · 安全过滤器提前返回和停止语义

- 位置：`examples/robot_foundation_models/common/safety_filter.py`，145-190,239-250; module docstring。
- 问题：关节裁剪分支跳过速度与碰撞；速度裁剪跳过碰撞。max_velocity 实际比较单步增量，无 dt；绝对位置全零不是停止；NaN 返回的 abort 没有 emergency_stop 标志。
- 修正/状态：只完成警告，未修改控制源码。见下方最小复现。 当前记录：`open`。
- 证据：`local-code-and-CPU-fixture`。

### F004 · P2 · FetchPush 默认稀疏奖励

- 位置：`docs/14-rl-zero-to-one.md`，216-222,270-292。
- 问题：默认稀疏奖励是阈值外 −1、内 0，而非负距离或 +1/0。
- 修正/状态：已区分 Sparse/Dense 与环境版本。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://robotics.farama.org/envs/fetch/push/)。

### F005 · P2 · 评估忽略回合终止/截断

- 位置：`docs/14-rl-zero-to-one.md`，328-351,429。
- 问题：TimeLimit 截断后继续推进并累计成功，改变了宣称的评估协议。
- 修正/状态：已在 terminated/truncated 处退出或重置，并标明首次成功指标。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://robotics.farama.org/envs/fetch/push/)；[原始来源](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)。

### F006 · P2 · HER 配置使用已不存在的参数

- 位置：`docs/14-rl-zero-to-one.md`，299-304。
- 问题：当前 SB3 HerReplayBuffer 不接受旧 online_sampling 参数，直接构造又缺必需参数。
- 修正/状态：已改为算法 replay_buffer_kwargs 配置。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)。

### F007 · P2 · 四个向量环境不等于四核四倍速

- 位置：`docs/14-rl-zero-to-one.md`，397-404。
- 问题：make_vec_env 默认 DummyVecEnv，同进程顺序执行，n_envs=4 不能推出四倍速。
- 修正/状态：已区分向量化与进程并行。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://stable-baselines3.readthedocs.io/en/master/common/env_util.html)。

### F008 · P2 · OpenVLA 24GB 显存保证没有依据

- 位置：`docs/26-rfm-finetuning-and-evaluation.md`，144,205,208,212-216,459; examples/robot_foundation_models/openvla/lora_config.yaml:5-6,67。
- 问题：冻结 bf16 权重之外的激活、梯度、优化器和临时显存不是固定常数；官方示例较小 batch 仍给出约 27GB 下限。
- 修正/状态：文档已取消 24GB 承诺；主代理随后也修正 YAML 注释，并验证解析后的参数与原版一致。没有实际训练或显存实测，不能据此保证某张显卡可运行。
- 证据：[原始来源](https://github.com/openvla/openvla)。

### F009 · P2 · IRIS 论文与算法归属错误

- 位置：`docs/07-world-models-for-vla.md`，57,385-391,606。
- 问题：IRIS 不是 Janner 的 policy-prior 方法，也不是不重建观测的 MuZero 同类。
- 修正/状态：已改为 Micheli/Alonso/Fleuret 的 Transformers are Sample-Efficient World Models 和官方仓库。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://arxiv.org/abs/2209.00588)；[原始来源](https://github.com/eloialonso/iris)。

### F010 · P2 · DIAMOND 作者、名称和技术声明错误

- 位置：`docs/07-world-models-for-vla.md`，395-399。
- 问题：DIAMOND 原标题、作者拼写、缩写与 CFG/首次声明不准确。
- 修正/状态：已改为 Diffusion for World Modeling: Visual Details Matter in Atari，并去掉无依据保证。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://diamond-wm.github.io/)。

### F011 · P2 · DreamerV2/V3 创新与任务数混淆

- 位置：`docs/07-world-models-for-vla.md`，122,380; docs/15-world-model-zero-to-one.md:116-127。
- 问题：离散潜变量与 55 Atari 的里程碑属于 V2；V3 正式结果是 150 多任务。Straight-through 是有偏估计，不保证避免后验坍缩。
- 修正/状态：已区分版本和机制。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://arxiv.org/abs/2010.02193)；[原始来源](https://danijar.com/project/dreamerv3/)。

### F012 · P2 · 最小化目标中熵项符号相反

- 位置：`docs/15-world-model-zero-to-one.md`，284-300。
- 问题：最小化 −Q+αH 会压低熵，不是鼓励探索；原伪代码混用 V/Q 且只调用 step。
- 修正/状态：已明确通用概念伪代码，使用负熵并解释梯度/回报边界。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://arxiv.org/abs/1812.05905)。

### F013 · P2 · V-JEPA 2 引用了无关论文

- 位置：`docs/07-world-models-for-vla.md`，653。
- 问题：2502.05055 是光度立体论文，不是 V-JEPA 2。
- 修正/状态：已修为 2506.09985。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://arxiv.org/abs/2502.05055)；[原始来源](https://arxiv.org/abs/2506.09985)；[原始来源](https://ai.meta.com/research/vjepa/)。

### F014 · P3 · 预抓取图的障碍中心线图注

- 位置：`knowledge/atlas/planning-evidence.json`，manipulation-pregrasp.visual.caption。
- 问题：曲线数值给出障碍中心 x=.4，图注却称边界；真实边界为 .39/.41。
- 修正/状态：主代理已将作者 JSON 改为中心线，并重新生成图和派生页面；派生页面不重复计审。此更新未再由中断的独立主审复核。
- 证据：`local-source-JSON`。

### F015 · P2 · Bellman 随机转移期望和策略梯度符号

- 位置：`docs/06-rl-fundamentals-for-vla.md`，73,81-87。
- 问题：Bellman 只写动作期望漏掉随机转移；较低但仍正的原始回报不会自动产生负更新。
- 修正/状态：已加入 T(s′|s,a) 期望与 advantage/基线符号解释。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)；[原始来源](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。

### F016 · P2 · 把 MPC 定义为树搜索

- 位置：`docs/15-world-model-zero-to-one.md`，70。
- 问题：MPC 是滚动时域优化，树搜索只是可能的求解器。
- 修正/状态：已改定义，并列采样、梯度和二次规划等可选求解方式。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://underactuated.mit.edu/trajopt.html)。

### F017 · P2 · BC/PPO 示例没有有效梯度更新

- 位置：`docs/06-rl-fundamentals-for-vla.md`，263-289。
- 问题：模型输入动作而非观测、CrossEntropyLoss 调用错误，且缺 zero_grad/backward。
- 修正/状态：改为明示概念伪代码并补足正确梯度流。 当前记录：`fixed-passages-rechecked`。
- 证据：`local-code-semantics`。

### F018 · P2 · 声称不存在的评估/检查点记录已保存，并把假设写成实验原因

- 位置：`docs/benchmark_report.md`，§3.4 artifact list; §2.2,4.3,4.4,5.2。
- 问题：磁盘缺两次逐episode评估和checkpoint_info，10K缺history；实际reward含-d距离+bonus，非纯稀疏；种子/预算不匹配不能作受控因果结论。
- 修正/状态：逐项区分available/missing；原因改待检验；保留45%/50%冲突。 当前记录：`documentation-fixed-raw-evidence-missing`。
- 证据：`results/benchmarks/benchmark_v2.json`；`examples/unified_pushcube_env.py`；`results/smolvla/`。

### F019 · P2 · loss下降且成功率0%被直接判定为过拟合、模型/数据不足或LoRA

- 位置：`docs/foundations/03-deep-learning-basics.md`，§7,8,Q6; foundations/10-dataset-and-training.md §8; docs/28-smolvla-gpu-finetuning-runbook.md。
- 问题：无法排除预处理、动作合同、闭环接口等因素；可训练子集不等于低秩适配器。
- 修正/状态：保留观察，明确缺少因果消融和原始评估；LoRA须有适配器结构证据。 当前记录：`fixed-passages-rechecked`。
- 证据：`BENCHMARK.md`；[原始来源](https://arxiv.org/abs/2106.09685)。

### F020 · P2 · 平均词嵌入必然无法区分红绿；舍入attention行和精确1

- 位置：`docs/foundations/04-transformer-basics.md`，§7,9,Q7。
- 问题：平均会丢词序，不必丢词身份；已构造红绿可分反例；显示行和有0.999和1.001。
- 修正/状态：改为词序不敏感，并用浮点容差验证未舍入权重；性能差异不作单因果归因。 当前记录：`fixed-passages-rechecked`。
- 证据：`local-counterexample`；[原始来源](https://aclanthology.org/P15-1162/)。

### F021 · P2 · apply_3d必然主动，求逆必然被动

- 位置：`docs/foundations/05-coordinate-transform.md`，§6 主动/被动变换。
- 问题：矩阵运算不决定语义；正反方向换坐标都可为被动。
- 修正/状态：明确帧下标、输入输出表达和物理运动约定。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-2-of-2/)。

### F022 · P2 · logR误当三维向量；调用内部API；万向锁例遗漏degrees

- 位置：`docs/foundations/06-se3-and-rotation.md`，§6,8,Q3。
- 问题：矩阵对数需vee；实际mjuu_quat2mat不存在；90默认弧度不能演示90度奇异姿态。
- 修正/状态：用omega_hat=logR与vee；mju_quat2Mat；显式degrees=True，限定ball/free四元数状态。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mju-quat2mat)；`local-SciPy-and-MuJoCo-fixture`。

### F023 · P2 · 伪逆简式漏满行秩条件；IK解与FK验算采用不同杆长

- 位置：`docs/foundations/07-fk-jacobian-ik.md`，§5.1,5.2。
- 问题：秩亏时JJT不可逆；solver l2=.8而验证默认1使目标[1.2,.5]变[1.259,.691]。
- 修正/状态：一般用SVD；统一几何参数，验证残差并断言。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html)；`local-NumPy-fixture`。

### F024 · P2 · 柔顺必须力矩接口；粘性摩擦导致PD恒定稳态误差；安全接口无缺陷

- 位置：`docs/foundations/08-control-basics.md`，§4,6–8。
- 问题：导纳可驱动位置/速度；静止粘性摩擦为0；源码存在F003早退等缺陷。
- 修正/状态：按接口限定；常值负载才产生Kp e=tau_load；停止不能一概归零；警告源码未修。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://control.ros.org/rolling/doc/ros2_controllers/admittance_controller/doc/userdoc.html)；`local-equilibrium-derivation`；`F003`。

### F025 · P2 · MJCF角度默认单位用错；目标撞限位；练习称地面接触但无地面；虚构SafetyFilter接线

- 位置：`docs/foundations/09-mujoco-basics.md`，§6,8,9。
- 问题：实际±1.57度仅±.0274rad，目标1rad不可行；无plane；实际retargeting使用优化器bounds。
- 修正/状态：显式radian、目标和力矩约束、加入plane并给可达触地几何；区分传感器和执行器、积分步长条件。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-angle)；`local-MuJoCo-fixture`；`examples/dexmv_style_retargeting/dexmv_retargeting.py`。

### F026 · P2 · 均值等于最可能结果

- 位置：`docs/foundations/11-probability-and-optimization.md`，§1。
- 问题：均值是期望，最可能值是众数；双峰均值可能根本不出现。
- 修正/状态：区分均值/众数/多峰，给离散反例。 当前记录：`fixed-passages-rechecked`。
- 证据：`local-discrete-counterexample`。

### F027 · P2 · 摩擦锥朝外，与作用物体法向相反；切向箭头不垂直

- 位置：`docs/pipelines/11-dexterous-manipulation.md`，SVG Local contact constraints。
- 问题：同一force-on-object约定下，可行合力锥轴须沿向内法向，切向与法向正交。
- 修正/状态：统一受力对象/锥轴/箭头，明确法切分解。 当前记录：`fixed-passages-rechecked`。
- 证据：[原始来源](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-1-friction/)。

### F028 · P2 · 同一点经同一针孔折出两条光线；像平面符号约定不明

- 位置：`docs/foundations/12-perception-and-sensors.md`，SVG Pinhole camera projection。
- 问题：一个物点经针孔直线唯一；第二条path折弯至另一像点；物理倒像与正焦距虚拟像面需区分。
- 修正/状态：删错误射线、统一物理/虚拟平面及公式正负。 当前记录：`fixed-passages-rechecked`。
- 证据：`local-SVG-coordinate-geometry`。

### F029 · P2 · LoRA单矩阵缩减128倍与OpenVLA只训练0.1%

- 位置：`docs/05-interview-prep.md`，LoRA/OpenVLA sections。
- 问题：4096²/(2×4096×32)=64；OpenVLA论文rank32为97.6M约1.4%，应用所有linear，非仅QKV。
- 修正/状态：重算参数量，并保留具体rank/模块/表格范围。 当前记录：`fix-reported-recheck-pending`。
- 证据：[原始来源](https://arxiv.org/html/2406.09246v3)；[原始来源](https://arxiv.org/abs/2106.09685)。

## 可复现的数值证据

### SafetyFilter 离线反例

以下仅处理数组，不导入模型权重、不连接硬件；从仓库根目录运行于现有 NumPy 环境：

```python
import numpy as np
from examples.robot_foundation_models.common.safety_filter import SafetyFilter
calls = []
f = SafetyFilter(np.array([-1.]), np.array([1.]), max_velocity=0.1,
                 collision_checker=lambda q: calls.append(q.copy()) or False)
a, status = f.check(np.array([2.]), current_state=np.array([0.]))
print(a, status.safe, len(calls))
# 当前源码：[1.] True 0；变化量 1 > 配置 0.1，碰撞检查未调用
```

另测输入 .5 时速度裁剪到约 .1，同样 `safe=True` 且碰撞回调 0 次。触发 emergency 后，当前绝对关节位置 .8 会得到 0，代表 −.8 位移目标而非保持。NaN 得到 `safe=False/action=abort/emergency_stop=False`。这些是源码缺陷反例，不是真机试验。

### 已修算例复算

- NumPy 切片修改能影响原 ndarray，copy 不影响；Python list 切片改元素不影响原 list。
- 2-DoF IK：统一 l1=1、l2=.8 后，目标 [1.2,.5] 的残差为 `1.176869779612264e-5`；原误用 l2=1 验证得到约 [1.259,.691]。
- MuJoCo 摆：原 XML 默认度制把限位编译成 ±.0274017 rad，却给 1 rad 目标。修后为 ±1.57 rad、力矩教学限幅 ±3 N·m；1000 步末角 `1.0078530644` rad，仍有重力未补偿的误差，未声称精确跟踪或真机参数。
- 万向锁题：显式 `degrees=True` 两组矩阵相同；按默认弧度并不相同。公开 API `mju_quat2Mat` 存在，内部名 `mjuu_quat2mat` 不存在。
- 原舍入 Attention 示例行和为 1/.999/1/1.001；不是“显示数值精确等于 1”。PyTorch 代码没有在本环境执行。

## 结构测试与实验限制

早期本代理执行的 atlas 测试为 18 通过、1 因旧页面合同失败；该失败发生在目录页拆分中，不能当最终状态。独立主审中断前的历史快照为全套 269 通过、14 个 PyTorch 相关跳过。主代理后续又增加了测试并修正文档；最终布局与测试记录见 [学习布局与验证记录](../LEARNING_LAYOUT_REVIEW.md)，不冒称为本代理独立重跑，也不用于论证科学内容全对。

SmolVLA 500/10K 的逐 episode eval 与 checkpoint_info 不在仓库，10K 完整 history 也缺失；0% 成功等为历史汇总。交换语言的 canonical 45% 与历史 50% 尚无法调和，平均 loss 的窗口也无法复算。没有重训 SmolVLA/OpenVLA/Dreamer、没有运行 GPU Benchmark、没有复现论文真机结果。

## 仍需完成

阅读缺口已补完，但问题关闭尚未完成。新增报告的 F01–F19 使用独立命名空间 `remaining-review/F01–F19`，不要与本报告 F001–F029 混同；已修 H01–H08 的具体段落见 [论文与示例修订](paper-and-example-revision.md) 和 [Retargeting 修订复核](retargeting-revision-review.md)。已知待修旧文章已加页首提示；Sim-to-real 中方向危险的建议已纠正，部署性能表已撤销“实测”身份，但这不等于其余代码已修或真机安全。

后续应逐项修复补审发现，并独立复核修订。全部官方链接的长期有效性、全部论文实验、所有示例在支持平台上的可运行性、所有图的浏览器视觉渲染均不在当前已完成证据内。代码级控制安全修复须独立授权与离线回归，不能被这份文档审查替代。
