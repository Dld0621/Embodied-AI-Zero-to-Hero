# Remaining source review: pending / targeted-passages-only documents

审查日期：2026-09-05（Asia/Hong_Kong）  
审查性质：独立、有界、只读源审查；结论是 **baseline-reading-complete / findings-open**，不是“全书正确”。

## 1. 范围和方法

- 范围来自 `knowledge/content-audit.json` 与 `docs/reviews/content-audit-handoff.md`：19 篇 `read_status` 为 `pending` 或 `targeted-passages-only` 的 Markdown。
- 每篇都从实际文件首行读到 EOF；不是关键词检索替代全文阅读。下表的逻辑行数由 `awk 'END{print NR}'` 取得，SHA256 在全文阅读完成后重新计算。
- 检查内容：技术正误、内部矛盾、公式/单位/动作合同、代码明显不可执行处、没有证据边界的实测或论文断言。
- 只对会影响判断的论文机制、版本化模型资料和产品资料使用原论文或官方文档核对；没有逐条打开 19 篇中的所有外链。
- 没有读取或审查机器生成的知识图谱正文，也没有改动 `sources/`、现有文档或知识台账；本审查者只新增本文件。
- 没有下载模型、安装依赖、连接硬件、运行浏览器/localhost/CUA 或截图。离线验证只使用现有 `.venv` 的 Python/NumPy/SciPy/MuJoCo；没有运行 torch。

## 2. 全文覆盖归档

| 文件 | 实际读取范围 | SHA256 | 本轮确认点 | 本轮未核实边界 |
|---|---:|---|---|---|
| `docs/05-interview-prep.md` | 1–EOF（2609 行） | `67442e202dfa211c12dd54ffdebfc12f9d1debe15408097dec18313e0e705dd6` | F01–F04：OpenVLA、MHA、MuJoCo、代码题及 2026 论文问答有硬错误 | 招聘公司、薪资、面试频率和未列入 F01–F04 的具体数值没有逐项核实 |
| `docs/09-dexterous-hands-analysis.md` | 1–EOF（417 行） | `4c46cbf608cd2c97fee9e37eba29be441aca926efbc9da1969e0d94300b28124` | F05：LEAP V1/V2、机构和驱动描述混淆 | Shadow/Allegro/O10/DEX-EE/AR10/ORCA 的全部规格、价格、许可证未逐产品核实；价格均为时效性信息 |
| `docs/10-manipulation-datasets.md` | 1–EOF（225 行） | `8e4afe0d40d7e72069f314e217ef89b44dbaa0f355e0987549f7e9c426eedcfe` | F06：robomimic 被错误写成 Shadow Hand；DexCap 轨迹语义被扩大 | 其余数据集大小、许可证、机器人覆盖和“最大/唯一”没有做穷尽核实 |
| `docs/11-dexmv-research-guide.md` | 1–EOF（778 行） | `605a1a98ce9216349e20979126d7211250bc0cc1dafd5479dc84c0f29140a19b` | F07：本地 solver 被错误等同 DexMV；Jacobian/精度/实时断言越界；附录快照已陈旧 | 未接入真实手部输入；外部 `OmniHand_v19` 路径及所有第三方性能没有核实 |
| `docs/12-freshman-zero-to-one.md` | 1–EOF（641 行） | `6f49cea37b5bb0595fedb20578c3c1f3075f98b1c9a369ba12245a2bc9432207` | F07：重复 DexMV/Huber/<10 mm/100 Hz 错误边界；本地命令可运行但结果仅为合成演示 | MediaPipe/相机/外部控制路径没有运行；“真实数据 <10 mm”没有证据 |
| `docs/13-vla-zero-to-one.md` | 1–EOF（28 行） | `de9473b8b3f697a81a52c6743f5532d3f63576b04f9ac035573533bf100c0fe3` | 三条本地入口命令均离线成功；“教学闭环、非大模型复现”边界正确 | 没有因此核实任何真实 SmolVLA/OpenVLA checkpoint、GPU 或硬件结果 |
| `docs/16-arxiv-retargeting-scan.md` | 1–EOF（592 行） | `2ebd10118bfb4200210633be98f65dde00299a519ef8fbae6d671c2762b38b89` | F07–F08：DexMV 错配、占位 arXiv、Nyquist 表述和 kHz 性能边界错误 | 80+ 条论文不是系统综述；未逐篇核实，搜索链接与条目数也未复现 |
| `docs/17-research-trends-and-positioning.md` | 1–EOF（368 行） | `7c5cbcfbd6113a1942d2e1b34a48874a69a2ecd8f84910b527332f0788ebcdb0` | F03、F08：ZR-0/Pose-VLA/DexSim2Real 机制错配；有限扫描不足以证明 novelty gap | 其余 2026 工作、机构、SOTA 和趋势判断没有系统检索；个人研究对比不构成仓库证据 |
| `docs/18-frontier-papers-online.md` | 1–EOF（331 行） | `181f574d8e8b7eee8be544651298bc6b4dc796dcb33a2b5d5a11c979a30f736b` | F07–F08：Kilohertz-Safe 单位相差 1000 倍；DexMV Huber 继续被重复 | 除明确列出的原论文核对项外，其余论文机制、会议状态和指标未逐篇核实 |
| `docs/19-sim-to-real-guide.md` | 1–EOF（1181 行） | `47515f328b7cba0d222b71c6e32c92ff392c820d295282c4b66197b4a9e60c7f` | F09–F10：域随机化、VLM 示例、MJCF、触觉分类和故障建议有可确认错误 | 没有加载全文 XML、运行 VLM、连接触觉/真实机器人或验证推荐参数范围 |
| `docs/20-vla-deployment-guide.md` | 1–EOF（1397 行） | `f88c474e4c3c62d02221380d0165899ac3fb81297868e29dfe9d86a97b001967` | F11–F12：OpenVLA 动作合同、ONNX/KV/chunk/异步代码、推理显存技巧及“实测”表有硬错误 | 未运行 GPU、TensorRT、vLLM、Jetson 或量化；所有设备性能数字都未复测 |
| `docs/21-vla-dataset-organization.md` | 1–EOF（637 行） | `bb4657efb84b5047ef03cf6c0a32e2dffc6f07a3281c4d962e155fa0f814f0b9` | F13：统计量被压成标量；所谓 LeRobot 格式实际写 NPZ；特征/张量布局错误 | 没有安装 LeRobot 或执行版本特定 schema；真实多传感器时间同步未测试 |
| `docs/22-act-vs-diffusion-policy.md` | 1–EOF（479 行） | `c8026d9c8f52a63a4729d36857e8b302eaaf279509ae4d04802725ad147367df` | F14：采样器 NameError、后验方差错误；“排列不变”术语不准 | 按约束未运行 torch；训练结果与两条示例命令未复测 |
| `docs/23-robot-foundation-models.md` | 1–EOF（432 行） | `2c5a7b40b9a874a97c5aa69713603a126ed23157a4777d0efcef4f67bf297d78` | F15：控制循环不可直接运行，安全层顺序内部冲突 | SmolVLA/OpenVLA/Octo/GR00T/Gemini 的仓库状态及适配器未逐个执行 |
| `docs/25-cross-embodiment-adaptation.md` | 1–EOF（438 行） | `313d4304f8d1a7ce3bc64056a7786e2d167227a30ae1fbb16d004160f4e9c1ce` | F16：Octo 机制简化错误；Franka/UR 适配器积分虚拟状态并丢弃夹爪命令 | 典型控制频率未逐厂商核实；没有在仿真或硬件运行适配器 |
| `docs/26-rfm-finetuning-and-evaluation.md` | 1–EOF（459 行） | `90cffe06b15c718bfa53c89bbf573da0432350fa2e144d3b53c0ccd6b280b821` | F17：50 episode 建议被扩大；单一 selection 指标不能证明语言 grounding；示例百分比像未标注实测 | 没有 SmolVLA/OpenVLA 微调/评测回执，显存仅保留估算边界 |
| `docs/27-embodied-reasoning-and-planning.md` | 1–EOF（458 行） | `024a9ced8fea2ba58fc212aa95c8d17841566865c3ed59958889d905b5b05f0f` | F18：JSON fallback 不能“保证不崩溃”；“任意自然语言/空间推理”过度承诺 | 未调用 OpenAI/Gemini API；模型名、SDK 调用和延迟是时效性信息，未运行核实 |
| `docs/28-smolvla-gpu-finetuning-runbook.md` | 1–EOF（191 行） | `96eecefa06e38ef489b857386144dc3fb5a1b0c291ced56432fd813f740fc1e6` | F19：A100 训练时长与官方文档冲突；“full 450M”与约 100M trainable 内部冲突 | 没有 GPU/网络/LeRobot；历史 RTX aggregate 无权重、逐 episode 数据和完整历史，不能独立重算 |
| `docs/29-learning-tracks-detail.md` | 1–EOF（163 行） | `540d185f9fa000272474781b55874548b6daf832ef02b9c052f4942cac27050c` | 未发现新的独立硬错误；限制说明总体比状态表更谨慎 | 所有 `✅ Runnable`/benchmark 状态没有在本轮逐入口重跑；依赖被引用文档的 open findings |

### 2.1 可机器读取的覆盖清单

以下 JSON 仅记录本轮实际全文读取的文件版本；`read_line_count` 是逻辑行数，`read_sha256` 是读取完成后对整文件计算的 SHA256。

```json
[
  {"file":"docs/05-interview-prep.md","read_sha256":"67442e202dfa211c12dd54ffdebfc12f9d1debe15408097dec18313e0e705dd6","read_line_count":2609},
  {"file":"docs/09-dexterous-hands-analysis.md","read_sha256":"4c46cbf608cd2c97fee9e37eba29be441aca926efbc9da1969e0d94300b28124","read_line_count":417},
  {"file":"docs/10-manipulation-datasets.md","read_sha256":"8e4afe0d40d7e72069f314e217ef89b44dbaa0f355e0987549f7e9c426eedcfe","read_line_count":225},
  {"file":"docs/11-dexmv-research-guide.md","read_sha256":"605a1a98ce9216349e20979126d7211250bc0cc1dafd5479dc84c0f29140a19b","read_line_count":778},
  {"file":"docs/12-freshman-zero-to-one.md","read_sha256":"6f49cea37b5bb0595fedb20578c3c1f3075f98b1c9a369ba12245a2bc9432207","read_line_count":641},
  {"file":"docs/13-vla-zero-to-one.md","read_sha256":"de9473b8b3f697a81a52c6743f5532d3f63576b04f9ac035573533bf100c0fe3","read_line_count":28},
  {"file":"docs/16-arxiv-retargeting-scan.md","read_sha256":"2ebd10118bfb4200210633be98f65dde00299a519ef8fbae6d671c2762b38b89","read_line_count":592},
  {"file":"docs/17-research-trends-and-positioning.md","read_sha256":"7c5cbcfbd6113a1942d2e1b34a48874a69a2ecd8f84910b527332f0788ebcdb0","read_line_count":368},
  {"file":"docs/18-frontier-papers-online.md","read_sha256":"181f574d8e8b7eee8be544651298bc6b4dc796dcb33a2b5d5a11c979a30f736b","read_line_count":331},
  {"file":"docs/19-sim-to-real-guide.md","read_sha256":"47515f328b7cba0d222b71c6e32c92ff392c820d295282c4b66197b4a9e60c7f","read_line_count":1181},
  {"file":"docs/20-vla-deployment-guide.md","read_sha256":"f88c474e4c3c62d02221380d0165899ac3fb81297868e29dfe9d86a97b001967","read_line_count":1397},
  {"file":"docs/21-vla-dataset-organization.md","read_sha256":"bb4657efb84b5047ef03cf6c0a32e2dffc6f07a3281c4d962e155fa0f814f0b9","read_line_count":637},
  {"file":"docs/22-act-vs-diffusion-policy.md","read_sha256":"c8026d9c8f52a63a4729d36857e8b302eaaf279509ae4d04802725ad147367df","read_line_count":479},
  {"file":"docs/23-robot-foundation-models.md","read_sha256":"2c5a7b40b9a874a97c5aa69713603a126ed23157a4777d0efcef4f67bf297d78","read_line_count":432},
  {"file":"docs/25-cross-embodiment-adaptation.md","read_sha256":"313d4304f8d1a7ce3bc64056a7786e2d167227a30ae1fbb16d004160f4e9c1ce","read_line_count":438},
  {"file":"docs/26-rfm-finetuning-and-evaluation.md","read_sha256":"90cffe06b15c718bfa53c89bbf573da0432350fa2e144d3b53c0ccd6b280b821","read_line_count":459},
  {"file":"docs/27-embodied-reasoning-and-planning.md","read_sha256":"024a9ced8fea2ba58fc212aa95c8d17841566865c3ed59958889d905b5b05f0f","read_line_count":458},
  {"file":"docs/28-smolvla-gpu-finetuning-runbook.md","read_sha256":"96eecefa06e38ef489b857386144dc3fb5a1b0c291ced56432fd813f740fc1e6","read_line_count":191},
  {"file":"docs/29-learning-tracks-detail.md","read_sha256":"540d185f9fa000272474781b55874548b6daf832ef02b9c052f4942cac27050c","read_line_count":163}
]
```

## 3. 已确认问题与建议修正

### F01 — `05` 的 OpenVLA 输出与损失描述错误（高）

- 位置：`docs/05-interview-prep.md:312-314, 890-913`。
- 问题：vanilla OpenVLA 被写成连续 MSE 回归、新初始化 MLP 策略头；实际 vanilla OpenVLA 将每个连续动作维量化为 256 bins，映射为 token 并做 next-token prediction。OFT 是后续连续动作方案，不能回填成 vanilla 预训练机制。LoRA 官方实现默认作用于 `all-linear`，不是只对 QKV。
- 建议：把 vanilla 与 OFT 分成两栏；删除 MSE/MLP/固定 5k–10k 轨迹与步数保证，引用官方仓库与论文的动作 tokenizer、loss 和具体实验设置。
- 证据：[OpenVLA 官方仓库](https://github.com/openvla/openvla)、[OpenVLA 论文](https://arxiv.org/abs/2406.09246)。

### F02 — `05` 的 MHA 复杂度推导、RT-1 TokenLearner 与离散 bin 结论错误（高）

- 位置：`docs/05-interview-prep.md:289-295, 863-868, 1262-1268`。
- 问题：Q10 把每头成本写为 `O(N d_k²)` 并得出总成本比单头低 `1/h`，漏掉 attention 的 `O(N²d)` 及 QKV/out projection；标准相同 `d_model` 下，多头总阶数与单头相当。RT-1 是把每帧 81 个视觉 token 压成 8 个，6 帧共 48 个，不是“压缩到 81”。把 bins 从 256 增至 1024 增大词表/分类难度，但若每个动作维仍输出一个 token，并不会让序列变长。
- 建议：重写复杂度为 projection `O(Nd²)` + attention `O(N²d)`；改成 `81→8 tokens/frame`；把 bins 的代价写为输出词表/统计覆盖，而不是生成步数。
- 证据：[RT-1 原论文](https://arxiv.org/abs/2212.06817)、[Google RT-1 官方介绍](https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/)。

### F03 — `05`/`17` 的 ZR-0、Pose-VLA、DexSim2Real、MoDE-VLA 机制错配（高）

- 位置：`docs/05-interview-prep.md:2419-2609`，`docs/17-research-trends-and-positioning.md:13-19`。
- 问题：
  - ZR-0 的 ECoT 是 **Embodied Chain-of-Thought**，不是 Error-Correcting CoT；原论文在训练时用 dense ECoT 对齐，推理时可以完全跳过 ECoT，文档的错误注入/RL 三阶段/推理先输出 reasoning/“20%+”均没有原论文依据。
  - Pose-VLA 使用 discrete pose tokens、diverse 3D data 与 trajectory supervision；不是文档所写的 Ego4D/YouTube 无标签视频上的 Pose VAE/扩散潜变量。
  - DexSim2Real 的三部分是 FM-DR（VLM realism critic + CMA-ES）、TVCAP 和 PSC；不是把 GPT-4V 任务进度直接转标量奖励及自动失败修正的整套机制。
  - MoDE-VLA 原文强调异构力/触觉模态与 residual injection；文档画成按 pinch/push/twist 操作类别 top-k 路由的专家池，属于未经论文支持的架构重构。
- 建议：四段按原论文摘要/方法重写；没有被论文报告的训练阶段、数据源、专家语义和提升数值全部删掉或明确标“假设设计”。
- 证据：[ZR-0](https://arxiv.org/abs/2606.30552)、[Pose-VLA](https://arxiv.org/abs/2602.19710)、[DexSim2Real](https://arxiv.org/abs/2605.05241)、[MoDE-VLA](https://arxiv.org/abs/2603.08122)。

### F04 — `05` 的代码题与基础事实有硬错误（高）

- 位置：`docs/05-interview-prep.md:374-375, 1125-1131, 1203-1214, 1306-1316, 1427-1488, 2233-2256, 2273-2323, 2362-2365`。
- 问题：
  - 第 98 行说 sinusoidal 可外推，第 375 行又说“没有外推能力”，内部矛盾。
  - MuJoCo 源代码许可证是 Apache-2.0，不是 MIT。
  - “Diffusion Policy 的 1D=展平向量、2D=用 2D U-Net 按 `T×D` 处理”的分类不是原论文设计；原文比较 1D temporal CNN 与 time-series diffusion transformer。
  - Q85 固定创建 `(3,n)` Jacobian，但示例 `fk_2dof` 返回 `(2,)`；离线执行在 `J[:,j] = ...` 处得到 `ValueError: could not broadcast input array from shape (2,) into shape (3,)`。
  - SAC 示例注释称含 tanh squashing correction，实际既没有 `tanh` 也没有 Jacobian correction。
  - 100 Hz 是 10 ms 周期，不自动推出远程双边端到端/往返延迟必须小于 10 ms；推理超时继续返回上次动作也不是通用安全降级。
- 建议：按输出维度动态建立 Jacobian并返回 solver status；SAC 要么实现 squashed Gaussian，要么明确是 unsquashed 简化；把控制周期、sensor-to-action latency、round-trip latency 和 jitter 分开，并把 timeout 行为改为经机器人合同验证的 hold/controlled stop。
- 证据：[MuJoCo 官方仓库](https://github.com/google-deepmind/mujoco)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)。

### F05 — `09` 混合 LEAP Hand V1/V2 且机构写错（高）

- 位置：`docs/09-dexterous-hands-analysis.md:26, 38-72`。
- 问题：机构表写 Columbia University；LEAP Hand 为 Carnegie Mellon University 工作。段落又把 V1 的 16 actuator/较高成本与 V2 的约 USD 200–300、16 DOF/8 motors 欠驱动方案混在一起，并据此写成逐关节直驱。
- 建议：拆成 V1 与 V2 两行，分别记录 DOF、独立 actuator/motor 数、欠驱动关系、价格口径与日期；删除“本项目默认目标”除非有仓库配置/硬件证据。
- 证据：[LEAP Hand V2 官方页](https://v2.leaphand.com/)、[V2 parts](https://v2.leaphand.com/parts)。

### F06 — `10` 把 robomimic 当成 Shadow Hand 数据，并扩大 DexCap 轨迹含义（高）

- 位置：`docs/10-manipulation-datasets.md:11-15, 66-100, 100-124`。
- 问题：robomimic 的 Lift/Can/Square/Transport/Tool Hang 是 robosuite 操作任务数据，不是 Shadow Hand/“SHAPES”数据；下载命令不能产生 Shadow Hand 关节序列，因此相关 retargeting 用途推论失效。DexCap 官方链路包含人手 mocap、fingertip IK 到 LEAP Hand 与 DexIL，但“唯一真实 LEAP 双手数据”“真实 retargeting 数据”等表述必须区分 human mocap、retargeted robot action 与真实 robot rollout/correction。
- 建议：按 `human motion / retargeted command / robot observation-action rollout` 三层重写用途；删除未经系统检索支持的“最大/唯一”。
- 证据：[robomimic 数据集概览](https://robomimic.github.io/docs/datasets/overview.html)、[robomimic v0.1](https://robomimic.github.io/docs/v0.4/datasets/robomimic_v0.1.html)、[DexCap 官方页](https://dex-cap.github.io/)。

### F07 — `11`/`12`/`16`/`18` 将本地 Huber solver 错称为 DexMV 复现（高）

- 位置：`docs/11-dexmv-research-guide.md:34-107, 186-396, 637-778`；`docs/12-freshman-zero-to-one.md:29-30, 110-140, 504-516`；`docs/16-arxiv-retargeting-scan.md:579`；`docs/18-frontier-papers-online.md:262,306`。
- 问题：原 DexMV retargeting objective 使用 palm-to-fingertip 与 palm-to-middle-phalanx 的任务空间向量 L2，加低通与时间项；原论文使用 SLSQP 与 `α=8e-3`。这些文档把本地“绝对 fingertip position + Huber”实现当作 DexMV 原法/完整复现，并给出论文 `<10 mm`、最高精度等无依据断言。`mj_jacBody` 是解析运动学 Jacobian API，不是自动微分。solver-only 毫秒数不能证明 100 Hz 端到端控制。
- 离线观察：当前 `examples/dexmv_style_retargeting/run_pipeline.py --model shadow --n_frames 30` 成功，结果为 mean FPE `76.778 mm`、max `146.969 mm`、约 `0.4 ms/frame`；附录列出的 reference tip position 与当前输出不一致。`examples/freshman_zero_to_one.py --gesture open --model shadow` 成功，mean `61.02 mm`、max `114.62 mm`、约 `1.5 ms/frame`。这些只证明当前合成教学路径可运行。
- 建议：统一改称 “DexMV-style pedagogical solver”；单列与原论文 objective/滤波/数据/评测的差异；检查 `result.success`；去掉 `<10 mm`、最高精度和端到端实时结论，除非有可重算真实输入评测。
- 证据：[DexMV 原论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990562.pdf)、[DexMV 官方代码](https://github.com/yzqin/dexmv-sim)。

### F08 — `16`/`17`/`18` 的文献扫描和 Kilohertz-Safe 单位/保证越界（高）

- 位置：`docs/16-arxiv-retargeting-scan.md:22-31, 65-70, 483-493, 579-592`；`docs/17-research-trends-and-positioning.md:125-138, 271-277, 355-368`；`docs/18-frontier-papers-online.md:47-61`。
- 问题：`16` 含 `2506.xxxxx` 占位符，且“200/80+ 篇”没有可重放查询或导出；“800 Hz 突破 Nyquist 限制”没有物理意义，只是提高采样率/提高 Nyquist frequency。Kilohertz-Safe 摘要报告平均 `9.05 ms`，`18` 写成 `9.0 μs`，差 1000 倍；论文报告的是超过 95% frames 满足 safety conditions，不是所有帧/全部约束的无条件保证。有限 scan 中“没看到论文”不能证明研究空白或 novelty。
- 建议：保存检索式、日期、去重导出；将“研究空白”降为待系统 related-work 验证的 hypothesis；更正单位和论文保证范围。
- 证据：[Kilohertz-Safe](https://arxiv.org/abs/2603.29213)。抽查的 [AnyDexRT](https://arxiv.org/abs/2607.08341)、[Smooth Operator](https://arxiv.org/abs/2607.07491)、[TopoRetarget](https://arxiv.org/abs/2606.16272) 仅用于各自条目，不代表整表完成核实。

### F09 — `19` 的域随机化代码会系统性改变参数且破坏状态语义（高）

- 位置：`docs/19-sim-to-real-guide.md:208-294`。
- 问题：torsional/rolling friction 在默认值上额外乘 `0.1/0.01`，不是围绕默认值随机化；`body_parentid != 0` 排除的是世界直接子 body，不等价“固定基座”；相机 axis/angle 计算后未使用；对全部 `qpos` 加均匀噪声会直接扰动 free-joint quaternion，且未保证每 episode 从 baseline/reset 开始。
- 建议：每次 rollout 先 restore model defaults + `mj_resetData`；按 joint type 处理 qpos，四元数在 SO(3) 上组合小旋转并归一化；用显式 body/joint allowlist；每个摩擦分量以自己的 baseline 和物理范围采样。

### F10 — `19` 的 VLM/MJCF/触觉/故障修正示例不可直接使用（高）

- 位置：`docs/19-sim-to-real-guide.md:614-659, 748-822, 835-844, 1026-1036`。
- 问题：VLM 代码缺 `PIL.Image` 与 `re` import，并使用已过时的 `openai.ChatCompletion`/`gpt-4-vision-preview` 写法；XML 在 start-tag 属性之间插入注释，语法无效，且 `cone` 属于 `<option>` 而不是 `<geom>`。`condim=6` 只定义接触维度，不证明“精确力控”。BioTac 不是光学传感器，而是流体、阻抗电极、压力与温度的 biomimetic sensor。过度抓取/压碎的根因写“力控阈值过低”，解决却写“增加力上限”，方向会加剧风险；异常滑动时“增加物体质量”也不是通用修复。
- 建议：代码改成当前 SDK 的可运行示例或只保留伪代码；用 MuJoCo 编译器验证 XML；拆开传感器类型；把压碎修正改为降低并验证力/力矩/位置目标、增益和限位，所有真实硬件阶段先限速限力并有独立急停。
- 证据：[MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)、[BioTac 官方产品手册](https://www.syntouchllc.com/Products/_media/BioTac%20Product%20Manual.pdf)。

### F11 — `20` 把文本生成当成 OpenVLA 动作 API，多个导出/缓存示例明显不可运行（高）

- 位置：`docs/20-vla-deployment-guide.md:37-98, 119-205, 276-350, 464-525, 565-625`。
- 问题：官方推理合同是 PIL image + processor + `predict_action(..., unnorm_key=..., do_sample=False)`；文档用 dummy image + `generate(20 tokens)` 测“完整动作”，没有反归一化，也不能代表任务动作延迟。ONNX 代码假设不存在/未验证的 `vision_tower`/`mm_projector` 属性，并写 `729=(224/14)^2`，算术上应为 256。手动 KV cache 调 `model.language_model`、丢图像条件并把 attention mask 重置成长度 1。增加生成 token 再从文本解析浮点数组不会把 vanilla OpenVLA 自动变成 action chunk policy。
- 建议：基准必须调用 checkpoint 官方动作 API并记录 warmup、硬件、软件、输入、unnorm key 和动作维度；OFT/FAST/vanilla 分开；删除未经端到端导出验证的 wrapper，或用目标版本模型的真实模块图和数值对齐测试。
- 证据：[OpenVLA 官方仓库](https://github.com/openvla/openvla)。

### F12 — `20` 的异步安全语义和“实测”性能表不可信（高）

- 位置：`docs/20-vla-deployment-guide.md:629-711, 773-1042, 1064-1110, 1135-1234`。
- 问题：gradient checkpointing 是训练激活换算力技术，不会在无反向传播的推理中按文档所述节省显存；`torch.cuda.empty_cache()` 不是预分配。异步/chunk 路径在空解析时索引 `[0]`，没有 action age/sequence/contract/safety checks，并可能持续返回陈旧动作。`device_map="auto"` 是 layer placement/sharding，不是 tensor parallel；每 robot 进程都 `device_map=auto` 可能各自占满全部 GPU。性能表列出不存在于官方 SmolVLA 产品线的 `SmolVLA-4B/256M`；官方 SmolVLA 是 450M，表中 OpenVLA/Jetson 数值也没有 artifact。
- 建议：删除或改为“未验证示例”；异步结果加入 monotonic sequence、capture/action timestamp、maximum age、parse failure 和 robot-specific controlled-stop；性能表必须链接原始 benchmark artifact，否则标为假设预算。
- 证据：[SmolVLA 官方文档](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)、[SmolVLA 官方介绍](https://github.com/huggingface/blog/blob/main/smolvla.md)。

### F13 — `21` 的归一化与 LeRobot 代码合同错误（高）

- 位置：`docs/21-vla-dataset-organization.md:121-170, 421-431, 484-507, 531-604, 622-628`。
- 问题：逐帧 1D state/action 用 `np.concatenate` 得到扁平向量，再 `.min(axis=0)` 只得到标量；应 `np.stack` 成 `(N,D)`。函数 docstring 声称写 LeRobot parquet+mp4，实际写每 episode NPZ，不能称 compatible。`Value("float32")` 是标量 feature，不能代表任意 state/action 向量；image 张量由 HWC 直接 `unsqueeze(0)` 得 BHWC，却注释 BCHW。HDF5 支持 chunked/lazy slicing，“不支持流式读取”过于绝对。最近邻对齐也缺 maximum skew、gap/clock/causality 约束。
- 离线反例：两个二维向量经 `np.concatenate` 后 shape 为 `(4,)`，`min(axis=0)` 为标量；`np.stack` 后为 `(2,2)`，按 axis 0 才返回每维统计。
- 建议：改 `np.stack`；把 NPZ 明确叫本地简化格式；真正 LeRobot 转换复用仓库 `common/to_lerobot.py` 并锁定版本；图像显式 `permute(2,0,1)`；时间同步按 timestamp 和动作因果合同设最大允许偏差。

### F14 — `22` 的 DDPM sampler 有 NameError 且后验噪声尺度错（高）

- 位置：`docs/22-act-vs-diffusion-policy.md:145-173, 330-383, 451-465`。
- 问题：构造函数未保存 `self.n_steps`，`sample()` 使用未定义的 `n_steps`。代码把 `sqrt(beta_t)` 称“标准 DDPM”后验噪声；标准 posterior variance 是 `tilde_beta_t = beta_t(1-alpha_bar_{t-1})/(1-alpha_bar_t)`（或明确选择的 variance parameterization），不能直接混称。`deterministic=True` 只是不在每步加噪，但初始 `x` 仍随机；若要复现要重置 RNG/传固定噪声。self-attention 对 token permutation 是 equivariant，不是单层本身 invariant；但“不加位置无法编码顺序”的结论仍成立。
- 建议：保存/使用 `self.n_steps`；实现并命名所选 posterior variance/DDIM sampler；接口接受 generator/initial noise；改正 equivariant 术语。
- 证据：[DDPM 原论文](https://arxiv.org/abs/2006.11239)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)。

### F15 — `23` 的统一控制循环不可直接运行且安全层顺序冲突（高）

- 位置：`docs/23-robot-foundation-models.md:41-55, 77-87, 94-116, 129-156, 327-378`。
- 问题：`done` 与 `current_state` 未初始化，`RobotObservation.extras` 无默认值却没传；因此“代码不需随模型变化”不是可运行事实。顶部图把 Low-level Controller 放在 Safety Filter 之前，后文又把 Safety Filter 放在 controller 之前，内部矛盾。`ee_delta` 也缺单位、frame、左/右乘旋转与 gripper channel；已有文字提醒但 schema 本身没有承载这些字段。
- 建议：把片段标为 pseudocode 或补全 reset/step contract、extras 默认值、current state 和错误处理；统一 safety composition 顺序；action schema 加 frame/units/rotation convention/timestamp/max-age/checkpoint normalization identity。

### F16 — `25` 的 Octo 与机器人适配器示例不满足所宣称合同（高）

- 位置：`docs/25-cross-embodiment-adaptation.md:91-139, 241-327, 409-415`。
- 问题：Octo 不是“共享动作隐空间 + 各机器人 decoder，扩散天然支持变维输出”；官方是共享 transformer/readout tokens + 轻量 action head，适配新动作空间通常替换/初始化 action head。Franka/UR 适配器把 `_current_joints` 初始化为零，并用上次**命令**而非实测关节状态积分 delta，首步和长期都会漂移；它读取 gripper 标量后只转成字符串，最终命令返回 7/6 维关节，实际丢弃夹爪，和“7+1/6+1”表不一致。ZOH/插值也不能自动“保证连续平滑”。
- 建议：adapter 每个周期接收带 timestamp 的 measured state；完整输出 arm+gripper 合同并校验维度/单位/范围；Octo 描述改成 modular tokenizer/readout/action head，明确换 head 需要目标数据微调。
- 证据：[Octo 原论文](https://arxiv.org/abs/2405.12213)、[Octo 官方新动作空间示例](https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.py)。

### F17 — `26` 的微调/语言评测断言超出证据（中）

- 位置：`docs/26-rfm-finetuning-and-evaluation.md:89-138, 284-290, 336-383, 419-439`。
- 问题：官方 SmolVLA 文档把约 50 episodes 作为一个任务的 starting point，并强调每个 variation 要有足够重复；文档写成“每个任务变体至少 50”，随后又断言简单 PushCube 50 条“足够”，均扩大了官方边界。`selection_accuracy >90%` 单独不能证明语言 grounding，必须用同场景/同初态的 counterfactual language pair 和冻结的成功定义。图中的 80/60/70/50% 没标“假设例子”，容易被当成实测。
- 建议：改成 starting budget，不承诺 sufficiency；语言消融按 paired seed/scene、correct/swapped/null、置信区间和错误物体率报告；示例百分比明确标 hypothetical 或删除。
- 证据：[SmolVLA 官方文档](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)。

### F18 — `27` 的 VLM planner 能力与 fallback 安全性被过度承诺（中）

- 位置：`docs/27-embodied-reasoning-and-planning.md:176-260, 264-288, 431-454`。
- 问题：只清理 Markdown fence 后 `json.loads` 与字段索引仍可因 schema/type/missing fields 失败；回退规则规划器也可能不支持指令，所以不能“保证系统不会崩溃”。表格的“任意自然语言”“空间推理：有”“真实部署”没有模型/版本/评测限定。把 8-DOF 输出直接裁掉最后一维也只有在合同明确最后一维就是 gripper 时成立。
- 建议：使用 schema validation、allowlist、重试/拒绝、unknown plan 和安全停机；VLM 输出只作为未可信计划，执行前逐子目标验证可达性、前置条件与成功条件；禁止按索引猜 channel。

### F19 — `28` 的 SmolVLA GPU runbook 时间与训练范围矛盾（中）

- 位置：`docs/28-smolvla-gpu-finetuning-runbook.md:1-12, 83-100, 150-183`。
- 问题：官方文档给 20k steps 单 A100 约 4 小时，本文写 30–60 分钟。标题说 full 450M fine-tune，历史部分又只报告约 100M trainable 且承认机制无法核实；这两个实验身份不能混同。8 GB minimum 也没有对应 batch/冻结/optimizer/图像配置的可复现实测。
- 建议：官方 recipe 与历史私有 trainer 分开；把时间/显存写成指定硬件软件与配置下的 measurement，不设无证据 minimum；保留现有“aggregate only、权重与逐 episode 缺失”边界。
- 证据：[SmolVLA 官方文档](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)（20k steps 单 A100 约 4 小时）。

## 4. 离线检查回执

以下回执只证明命令或数值反例本身，不证明模型/真机能力：

1. `scripts/run_knowledge_map.py --path-to learning-vla --lang zh`：退出码 0。
2. `scripts/run_pipeline.py --show vla-policy`：退出码 0。
3. `scripts/select_vla_wam_algorithm.py ...`：退出码 0。
4. DexMV-style Shadow 30-frame pipeline：退出码 0；mean/max FPE 与 solver-only 时间见 F07。
5. Freshman Shadow/open pipeline：退出码 0；mean/max FPE 与 solver-only 时间见 F07。
6. `05` Q85 维度反例：按原片段的 `(3,n)` Jacobian + 2D FK 触发 NumPy broadcast `ValueError`。
7. `21` normalization 反例：`concatenate` 后变 `(N*D,)` 且按 axis 0 统计为标量；`stack` 才保留 `(N,D)`。

## 5. 已查原始/官方来源与未查范围

本轮实际用于判断的主要原始或官方来源：LEAP V2 官方页、robomimic 官方文档、DexCap 项目页、DexMV ECCV 论文/官方代码、OpenVLA 论文/官方仓库、SmolVLA 官方文档/官方介绍、Octo 论文/官方代码、Diffusion Policy/DDPM 原论文、MuJoCo 官方仓库/XML reference、BioTac 官方手册，以及 ZR-0、Pose-VLA、DexSim2Real、MoDE-VLA、Kilohertz-Safe、AnyDexRT、Smooth Operator、TopoRetarget 的原论文页。

明确没有完成的范围：

- 没有逐条验证 `16`/`18` 的全部 80+ 论文，也没有重做系统综述或会议录检索；因此不得据此宣称完整、最新或没有相关工作。
- 没有逐一验证 `09`/`10` 中所有硬件/数据集的价格、许可证、DOF、驱动方式、数据量与下载可用性。
- 没有验证 `05` 的招聘、薪资、公司团队、题目高频度及所有经验阈值。
- 没有 GPU/Jetson/TensorRT/量化/网络/硬件运行；`20`/`28` 的性能数字除官方明确值外仍是未核实。
- 没有真实相机、MediaPipe、触觉、机器人或闭环硬件回执；solver-only、保存的 aggregate 和 mock/合成成功不得提升为端到端或真机成功。
- 没有逐入口重跑 `29` 状态表中的所有 `✅`，也没有审查本范围外文档；本报告不能用于宣称“全书正确”。

## 6. 台账状态建议

这 19 篇可以把“阅读覆盖”更新为 `baseline-reading-complete`，但发现仍为 `findings-open`。建议在确认修文后逐项关闭 F01–F19，并分别记录：修复 commit、验证命令/原始来源、仍未核实的外部断言。不要把 `baseline-reading-complete` 改写成 `verified-correct`。
