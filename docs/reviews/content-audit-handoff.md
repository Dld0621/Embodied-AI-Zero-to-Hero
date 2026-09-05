# 内容审查交接：已读不等于全部正确

2026-09-05 · [完整发现记录](content-correctness-audit.md) · [布局与测试](../LEARNING_LAYOUT_REVIEW.md) · [逐文件台账](../../knowledge/content-audit.json)

## 当前结论

学习布局改造已实施；**135 篇范围清单全文阅读完成，内容并非全部正确**。独立主审使用 GPT-6 Astra（xhigh），额度中断后由 GPT-5.6 Sol（xhigh）补完剩余 19 篇，并记录新发现。没有兑换额度；总覆盖是多位审查者与主代理的合计，不冒称一名代理独立验证所有终稿。

| 审查对象 | 当前覆盖 | 含义与缺口 |
| --- | --- | --- |
| 135 份源 Markdown | 135 份均有全文阅读记录 | 多名代理及主代理的实际版本阅读记录合计；并非所有修改后的段落均获独立复核 |
| 3 份作者知识 JSON | 201 个小知识点教学字段全读 | 逐点阅读和选定算例核查；不是全部引用或代码运行验证 |
| 自动生成图谱页 | 45 章、201 个独立小节、45 篇连续阅读页 | 是同源内容的呈现，不重复计作新科学审查 |
| 支持性教程 | 20 份支持文档有阅读记录，计入上述 135 份 | 10 篇 tutorial README 已修订，详见 [支持文档交接](supporting-review-handoff.md)；修订后尚未全面独立复核 |

仅通过构建、格式或单元测试，不能给整本教材“100 分”或保证初学者成为专家。没有训练模型、运行 GPU 评估或操作真机，也没有完成真实浏览器视觉验收。

## 优先处理：两个控制代码问题仍未修复

- **F002 / OpenVLA 适配器：**本地默认动作标签与模型、数据集、控制器的动作合同未得到验证；贴上关节角标签不会完成坐标或动作转换。
- **F003 / SafetyFilter：**提前返回跳过后续约束，速度/单步增量混用，停止语义和异常状态存在缺陷。

这些问题已经有文档警告和离线反例，但执行代码没有修复。**不要用这些脚手架接通真实机器人。** 控制实现与离线回归是单独的待授权工作，不由当前 UI 改造代替。

## 中断前新增发现的后续状态

以下保留来自审查代理的发现，防止交接丢失。H01–H08 的指定段落已于 2026-09-05 修订：H02–H06 见 [Retargeting 修订复核](retargeting-revision-review.md)，H01/H07/H08 见 [论文与示例修订](paper-and-example-revision.md)。这只关闭已列段落的具体错误，不等于全文、引用和真机系统均验证。补审又发现未修问题，见下一节。初学者优先使用新版 [基础课](../foundations/01-python-for-robotics.md) 与 [逐点图谱](../knowledge-atlas/index.md)；旧文的页首状态不可跳过。

| 编号 | 位置 | 待修内容与验收要求 |
| --- | --- | --- |
| H01 | [关键论文导读](../02-key-papers.md) | **已定向修订并检查：**修正 RT-1/SPOC 链接、TokenLearner、vanilla OpenVLA、π0、Octo/ACT 等机制及版本范围；撤回无依据性能/成本保证，保留 14 篇导读。没有复现论文全部实验。 |
| H02 | [方法分类](../02-retargeting-taxonomy.md) | **指定段落已修并离线回归：**连续分段映射与 `trf` bounds 求解已通过阈值两侧、上下关节边界测试。特定机器人收敛和耗时未验证。 |
| H03 | [人手映射](../03-human-hand-to-robot-hand.md) | **指定段落已修并离线回归：**已分开平移/旋转，修正屈曲角并拒绝退化输入，区分 MuJoCo qpos/qvel 地址。O10 模型映射、硬件限位与控制接口仍待目标模型逐项核实。 |
| H04 | [优化方法](../04-optimization-methods.md) | **指定段落已修并离线回归：**奇异值效应、阻尼阈值和 CMA-ES 保证已纠正；指尖距离明确仅为代理。完整 link/geom 与路径碰撞未验证。 |
| H05 | [学习式 Retargeting](../05-learning-based-methods.md) | **代码结构已修：**15 维输入、参数顺序、H/W 和时间位置编码已修；所有 Python fence 语法通过。环境无 PyTorch，两个网络形状测试为 skip，未训练或下载权重。 |
| H06 | [评估指标](../06-evaluation-metrics.md) | **指定段落已修并离线回归：**旋转、jerk、互相关符号和空指标已修；求解耗时单列。真实端到端时间戳、轨迹与任务实验未验证。 |
| H07 | [DexMV-style 示例说明](../../examples/dexmv_style_retargeting/README.md) | **README 已修：**明确标定后米制坐标、教学求解器和历史合成结果；撤回精度榜/端到端实时承诺，修复片段语法。相机转换、真机和原论文复现未实现；其他旧文的重复错误仍见补审 F07。 |
| H08 | 模型示例及数据集 README | **三个 README 中相关入口已修：**仓库根路径检查通过，mock metadata 与实际 writer 对照；明确不是已验证 LeRobotDataset。SmolVLA launcher 离线 test 四项通过，不代表训练。 |

F001–F029 的既有状态保留在原报告及 JSON。本轮主代理还修订了关节概念、百科中的 DH/坐标/传动/延迟定义、摩擦锥图和部分课程入口；这些改动不自动扩大独立审查覆盖。百科由主代理全文阅读，不冒称独立主审完成。

## 补充独立审查：19 份现已全文阅读

以下保留原缺口清单作为审计轨迹，现已由备用审查者从首行读至 EOF。每份实际读取 SHA256、行数、确认问题、原论文/官方来源及未核实边界见 [补审报告](remaining-source-review.md)。其 F01–F19 在 JSON 中记为 `remaining-review/F01–F19`，避免与早期 F001–F029 混淆。

主要开放项包括：旧面试题和 2026 论文机制、硬件/数据集分类、重复 DexMV 复现宣传、部署与数据格式代码、DDPM 采样器、跨本体/规划接口及无证据性能保证。已在有确认问题的 17 篇旧文添加针对性提示，初学者不会被默认告知其“已正确”。其中 Sim-to-real 的危险修正方向、部署中错误产品名与未验证性能表已明确纠正或撤销，其他技术问题仍开放；警告不是修复代码。

### 原待读的 14 份：已补读

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

### 原仅定向段落的 5 份：已补读

- [面试准备](../05-interview-prep.md)
- [机器人基础模型](../23-robot-foundation-models.md)
- [跨本体适配](../25-cross-embodiment-adaptation.md)
- [微调与评估](../26-rfm-finetuning-and-evaluation.md)
- [具身推理与规划](../27-embodied-reasoning-and-planning.md)

## 后续完成条件

继续修订时，以台账记录的读取版本、最新 diff 和补审 F01–F19 为起点；优先处理控制风险，并逐项确认修复和终稿复核。每条结论需区分原文核实、推导、离线数值实验和真机证据。修复作者 JSON 后重生成图谱；每次交付重新构建、跑回归，并更新发现状态，不把历史测试快照当本次结果。
