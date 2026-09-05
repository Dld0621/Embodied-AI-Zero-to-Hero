# 内容审查交接：已读不等于全部正确

2026-09-05 · [完整发现记录](content-correctness-audit.md) · [布局与测试](../LEARNING_LAYOUT_REVIEW.md) · [逐文件台账](../../knowledge/content-audit.json)

## 当前结论

学习布局改造已实施；**全内容正确性审查尚未完成**。独立主审使用 GPT-6 Astra（xhigh），主审和文档修订子代理随后遇到账户额度限制而中断。没有自动兑换额度或把其他代理的阅读冒称为独立主审复核。

| 审查对象 | 当前覆盖 | 含义与缺口 |
| --- | --- | --- |
| 135 份源 Markdown | 116 份全文读；14 份待读；5 份只读指定段落 | 多名代理及主代理的基线阅读记录合计；并非所有修改后的段落均获独立复核 |
| 3 份作者知识 JSON | 201 个小知识点教学字段全读 | 逐点阅读和选定算例核查；不是全部引用或代码运行验证 |
| 自动生成图谱页 | 45 章、201 个独立小节、45 篇连续阅读页 | 是同源内容的呈现，不重复计作新科学审查 |
| 支持性教程 | 20 份支持文档有阅读记录，计入上述 135 份 | 10 篇 tutorial README 已修订，详见 [支持文档交接](supporting-review-handoff.md)；修订后尚未全面独立复核 |

仅通过构建、格式或单元测试，不能给整本教材“100 分”或保证初学者成为专家。没有训练模型、运行 GPU 评估或操作真机，也没有完成真实浏览器视觉验收。

## 优先处理：两个控制代码问题仍未修复

- **F002 / OpenVLA 适配器：**本地默认动作标签与模型、数据集、控制器的动作合同未得到验证；贴上关节角标签不会完成坐标或动作转换。
- **F003 / SafetyFilter：**提前返回跳过后续约束，速度/单步增量混用，停止语义和异常状态存在缺陷。

这些问题已经有文档警告和离线反例，但执行代码没有修复。**不要用这些脚手架接通真实机器人。** 控制实现与离线回归是单独的待授权工作，不由当前 UI 改造代替。

## 中断前新增、尚未完成修订的发现

以下保留来自审查代理的发现，防止交接丢失。页面顶部已提示相应风险；提示不等于修复。旧文章仍保留作历史学习材料，初学者优先使用新版 [基础课](../foundations/01-python-for-robotics.md) 与 [逐点图谱](../knowledge-atlas/index.md)。

| 编号 | 位置 | 待修内容与验收要求 |
| --- | --- | --- |
| H01 | [关键论文导读](../02-key-papers.md) | RT-1 链接误指其他论文；TokenLearner 数量混淆；vanilla OpenVLA 被写成连续 MSE 输出；π0 冻结/推理描述不准确。按原论文逐段修正并复核，其余性能/成本断言仍待核实。参考 [RT-1 原文](https://arxiv.org/abs/2212.06817)、[OpenVLA 原文](https://arxiv.org/abs/2406.09246)、[π0 原文](https://arxiv.org/abs/2410.24164)。 |
| H02 | [方法分类](../02-retargeting-taxonomy.md) | 分段映射在阈值 0.5 处从约 0.6 跳到 0.4；带 bounds 的 least_squares 使用不支持该组合的 LM。改为连续映射与合适约束求解器，并在阈值两侧和关节边界回归测试。 |
| H03 | [人手映射](../03-human-hand-to-robot-hand.md) | 减去手腕只去平移，不等于完成手掌旋转坐标变换；屈曲角与骨段内角混淆。关节地址、模型映射和硬件限位也需逐项核实，不能宣称通用完整控制流程。 |
| H04 | [优化方法](../04-optimization-methods.md) | Jacobian 转置与伪逆的奇异值效应混淆；自适应阻尼需明确阈值外行为；指尖距离不代表全连杆无碰撞；CMA-ES 没有有限预算的全局最优保证。 |
| H05 | [学习式 Retargeting](../05-learning-based-methods.md) | 5 个三维点与 12 维输入不一致；函数必需参数位于默认参数之后；H/W 未定义；时序 Transformer 缺少明确时间位置编码。需修正后运行形状与语法测试，不能照抄作为可运行实现。 |
| H06 | [评估指标](../06-evaluation-metrics.md) | SO(3) 矩阵对数范数与向量范数混用；jerk 公式、采样间隔与实现不一致；互相关延迟示例 dt 未定义且信号/方向约定不清；未填指标产生空均值；求解器耗时被混作端到端延迟。需逐指标加入单位、输入条件和合成轨迹回归。 |
| H07 | [DexMV-style 示例说明](../../examples/dexmv_style_retargeting/README.md) | MediaPipe 归一化坐标不自动等于米；精度/实时性表没有同协议的复测证据。需明确坐标转换与实测范围，不能直接视为真机或论文复现结果。 |
| H08 | 模型示例及数据集 README | 工作目录和数据格式说明仍需对照入口；mock Parquet 不应仅凭文件扩展名宣称为已验证的标准 LeRobotDataset。 |

F001–F029 的既有状态保留在原报告及 JSON。本轮主代理还修订了关节概念、百科中的 DH/坐标/传动/延迟定义、摩擦锥图和部分课程入口；这些改动不自动扩大独立审查覆盖。百科由主代理全文阅读，不冒称独立主审完成。

## 19 份尚未全文阅读的文件

以下“待读”是全文覆盖状态；其中一些文件已修过定向段落，不能因此标记全文完成。

### 待读：14 份

- [灵巧手分析](../09-dexterous-hands-analysis.md)
- [操作数据集](../10-manipulation-datasets.md)
- [DexMV 研究指南](../11-dexmv-research-guide.md)
- [新生零到一](../12-freshman-zero-to-one.md)
- [VLA 零到一旧课程](../13-vla-zero-to-one.md)
- [Retargeting 论文扫描](../16-arxiv-retargeting-scan.md)
- [研究趋势与定位](../17-research-trends-and-positioning.md)
- [前沿论文](../18-frontier-papers-online.md)
- [Sim-to-real 指南](../19-sim-to-real-guide.md)
- [VLA 部署](../20-vla-deployment-guide.md)
- [VLA 数据集组织](../21-vla-dataset-organization.md)
- [ACT 与 Diffusion Policy](../22-act-vs-diffusion-policy.md)
- [SmolVLA 微调运行手册](../28-smolvla-gpu-finetuning-runbook.md)
- [学习路线详解](../29-learning-tracks-detail.md)

### 仅定向段落：5 份

- [面试准备](../05-interview-prep.md)
- [机器人基础模型](../23-robot-foundation-models.md)
- [跨本体适配](../25-cross-embodiment-adaptation.md)
- [微调与评估](../26-rfm-finetuning-and-evaluation.md)
- [具身推理与规划](../27-embodied-reasoning-and-planning.md)

## 后续完成条件

继续审查时，以台账记录的基线和实际修订内容为起点；先处理控制风险与 H01–H08，再补全文覆盖。每条结论需区分原文核实、推导、离线数值实验和真机证据。修复作者 JSON 后重生成图谱；每次交付重新构建、跑回归，并更新发现状态，不把历史测试快照当本次结果。
