# 支持文档审查交接记录

日期：2026-09-05。审查者：`/root/foundation_correctness_fixes`。

本记录补充主审计，不替代全书正确性认证。以下 **20 篇原始支持文档均从首行读到 EOF**，原始 `wc -l` 合计 3602；不是只读标题、关键词或自动扫描。授权修复仅限其中 10 篇 `tutorials/**/README.md`，另外 10 篇未由本审查者修改。主代理随后在 DexMV README 顶部新增 2 行证据警告；本审查者已阅读新增警告，技术正文未因此变成已修复。最终修订快照的 `wc -l` 合计 3701。

“全文阅读”不等于“所有论断已核实”或“全部代码可运行”。独立主审计因运行额度中断，下面 10 篇教程的最终改动尚未获得独立逐篇复核；当前只有作者自查、语法/链接检查和明确列出的离线数值测试。没有下载模型、安装依赖、训练、相机采集、网络控制、网页渲染测试或真机执行。本记录没有将仍有缺陷的执行源码修成安全控制器。

## 20 篇覆盖和最终快照

所有路径相对于仓库根目录。行数为 `wc -l`，SHA256 为本交接记录保存时的内容快照；后来任何修改都应更新审计状态，而非继续沿用旧 hash 作为证明。

| 文件 | 原始行数 → 当前行数 | 本次状态 | 当前 SHA256 |
|---|---:|---|---|
| `docs/tutorials/mujoco-scene-building.md` | 395 → 395 | 全文读；未改 | `97fe9517a0b293c67bf8fc552cd31c60eddbdbd99bbb30fd8296a5bd2e74d4a7` |
| `examples/dexmv_style_retargeting/README.md` | 244 → 246 | 全文读；主代理随后新增警告，技术问题未修 | `59572c367a5e29bb506f3b2ecb5d2ad553ebdf44e10c77d25c230d0bb20c3d61` |
| `examples/mujoco_scene_builder/README.md` | 39 → 39 | 全文读；未改 | `d98b0dcbbc29c5dfd9aafca8146eafe98fc6beaaa2ed9efc6408999716974c55` |
| `examples/robot_foundation_models/README.md` | 169 → 169 | 全文读；有未修问题 | `c61bbdcc74bfa74d9e5f9a2b3b554b54452535b83309ee11b05e79466d21c01e` |
| `examples/robot_foundation_models/smolvla/datasets/README.md` | 112 → 112 | 全文读；有未修问题 | `51c06af63bf90c5af511495ae582b80608df8d43bf18f5717dc5b2ead6314817` |
| `learner/README.md` | 19 → 19 | 全文读；未改 | `68d772763b588a36c923c5ce9a82b949a9fba3dbec0eda3db5d5b3305f54c0b5` |
| `learner/templates/capstone-review.md` | 45 → 45 | 全文读；未改 | `ce57d64b36a42ad9c69b66670623b79200861361e2ba59f9457d3aa273e07a33` |
| `learner/templates/experiment-card.md` | 57 → 57 | 全文读；未改 | `a2eaaf8f2eaaf3c8ceb1262bbe0db7d942cd1aaaa113fcb19e24429fb8fb5707` |
| `learner/templates/failure-report.md` | 44 → 44 | 全文读；未改 | `1f9cfae79d74bd47249bb227fd919063e42d753afd31d30ab625126ac1aa8741` |
| `tools/robotdev/README.md` | 17 → 17 | 全文读；未改 | `067663142e5412bd6b3672435147086517f13a9dcc81f7de104fa3e21648aceb` |
| `tutorials/01-fk-ik-basics/README.md` | 130 → 135 | 已修；待独立复核 | `b271b058f8714a9b20d0a368d440bf15db593f786d566ddb3f5af95847332e6b` |
| `tutorials/01-vlm-basics/README.md` | 177 → 192 | 已修；待独立复核 | `9094307ec5f0fa9fd65ecbe5f8baca8aedcdb4283f62949fbaa995f7df84f575` |
| `tutorials/02-action-representation/README.md` | 242 → 254 | 已修；待独立复核 | `17c4550af83214b9c67e86bbe2f79cabac8dd86869626feb0c6a72c46fa2ab1c` |
| `tutorials/02-rule-based-retargeting/README.md` | 72 → 78 | 已修；待独立复核 | `723dc8456d417efdee58ce24b7cae64267a7dd83f3cca60c4d5e63574ad64b80` |
| `tutorials/03-simple-vla/README.md` | 307 → 309 | 已修；待独立复核 | `28ec8902be0688b1ef4e5baef2642d94ccfbd26ac1d68c4352e091e06fbd99a0` |
| `tutorials/03-vector-optimization/README.md` | 83 → 91 | 已修；待独立复核 | `6856ab32d0cb2e1c98698ee7fb2435e50d92376f9a10982d9b584860464e1420` |
| `tutorials/04-fine-tuning/README.md` | 463 → 466 | 已修；待独立复核 | `cc739d102dc2fe28c01fe6996411e8cdfc2c816c58e1d778a0d52c42dd1eb61e` |
| `tutorials/04-landmark-pipeline/README.md` | 117 → 134 | 已修；待独立复核 | `80df60101bc079d74b2d863b39d9592df08bf46b181f549868564ccc1c91854b` |
| `tutorials/05-complete-pipeline/README.md` | 636 → 655 | 已修；待独立复核 | `4dd448546ea9207f1e3c4db9093265b27451b073e7b603c50aba6a76055a7e37` |
| `tutorials/05-world-models/README.md` | 234 → 244 | 已修；待独立复核 | `4a20f258636fed13c929b39078c2cc85a905a609177df55ede57afcd216ff414` |

## 逐篇结论与未核实边界

### 未修改的支持文件

1. **MuJoCo 场景搭建长教程**：全文检查了建模分工、primitive/mesh、MJCF/MJB、预览与例子入口。未发现本轮足够确定的新知识错误。未执行建模导出/GUI；MJZ、外部建模软件兼容性未做逐版本复现，不能称所有导出路线通过。
2. **DexMV 风格 README**：存在明确证据边界问题；主代理新增顶部警告，但以下技术问题尚未修。MediaPipe 接入片段把 `multi_hand_landmarks` 的归一化数直接交给位置目标处理，没有米制尺度/坐标标定。`huber_delta`“越小越精确”没有一般成立的保证；合成数据的 77 mm 与不同方法的精度区间不可未经同协议评估做排行榜；“真实数据通常 <10 mm”、求解 0.5–1 ms 即满足实时控制等结论没有相应端到端日志。第三方仓库可见性、InterHand 关键点顺序及“论文中精度最高”未完成独立原文核对，不应维持确定断言。
3. **MuJoCo scene builder 短 README**：全文阅读，未发现确定的新技术错误。未运行 preview/viewer；macOS 的 `mjpython` 特殊启动条件可补充，不能把普通 `python --viewer` 当跨平台保证。
4. **Robot foundation models README**：连续 quickstart 中的工作目录不一致。进入 `smolvla` 后，`from common...` 没有确保父目录在导入路径；`../../benchmarks/robot_foundation_models` 从该目录会落到 `examples/benchmarks/...`，不是仓库根下的 benchmarks。模型互换仍需动作合同/频率/归一化适配，不能称控制外循环无需核对就永不改变。已保存结果与当前 real-model 状态未逐条重跑。
5. **SmolVLA datasets README**：从 `examples/robot_foundation_models/smolvla` 执行 `cd ../../../..` 超过仓库根一级；后续又使用相对该目录的训练/数据路径，流程前后不一致。代码核对显示 mock PyArrow 路径保存 raw image bytes、单个 train parquet 与简化 meta，未证明可由指定版本 `LeRobotDataset` 直接加载；meta 实际没有文中承诺的 `action_type`。生成数据不等于已经随仓库提交的数据。`mock=True` 参数实际存在，`--test` 的 4 项配置检查存在，这两点不是错误；但配置测试不是模型训练。
6. **learner README**：全文阅读，未发现确定的新错误；学习证据目录与模板不等于已经取得成果。未替用户完成任何课程证据。
7. **capstone-review 模板**：全文阅读，未发现确定的新错误；标准是提交/评审约定，不是研究能力或真机成果证明。
8. **experiment-card 模板**：全文阅读，未发现确定的新错误；未验证任何用户填入的实验记录。
9. **failure-report 模板**：全文阅读，未发现确定的新错误；分类字段不证明故障原因已被实验隔离。
10. **robotdev README**：全文阅读，并读取其 `check_env.sh` / `stack_resolver.py` 辅助实现确认以环境检查/解析为主。未发现足够确定的新错误；本轮未在所有主机/ROS/CUDA 组合执行，安装兼容性仍需按目标栈核验。

### 已修改的 10 篇教程

11. **01 FK/IK**：补了解析 IK 的单支路边界；数值骨架改浮点数组、按任务维数建单位阵并用 solve，明确 FK/Jacobian 占位接口和残差检查；链接到实际存在的 `07-fk-jacobian-ik.md`。未把骨架补成完整机器人求解器。
12. **01 VLM**：CLIP 损失补批大小 N、输入检查和 labels 设备；patch32 的 224 输入为 49 patch + CLS，attention 网格从实际输入/config 推导；纠正 RT-2 并非直接采用 CLIP。未下载 CLIP/LLaVA 或验证完整 GPU 推理；LLaVA 各版本 AutoModel API 仍需独立核查。
13. **02 动作表示**：区分物理自由度与 Euler/四元数编码长度；delta 仍依赖坐标系与周期；DH 限定转动关节；数值 IK 添加实际残差断言。移除无出处的 RT-1/ACT/Octo 固定 chunk 调度表，改成明确假设 20 Hz 的调度例子并区分预测长度与执行长度。随机 GRU 动作头未训练。
14. **02 Rule-based**：明确伪代码与合成手势演示边界；撤去未定义机器人直接下发示意；不再把 <1 ms、最稳定和某组缩放值当保证；补归一化坐标不是米。真实校准、控制适配未实现。
15. **03 简单 VLA**：OpenVLA 改为 `**inputs`、匹配 bfloat16、`bridge_orig` 与确定性生成；说明内置 q01/q99 反归一化，不再手工 mean/std 二次缩放。单动作 7 分量不是 7 个未来动作；基础 API 不假设批量返回，逐样本调用与 chunk 模型区分。Octo 改为 observation 字典、history/batch 维、mask、rng 与数组返回，并取检查点统计。未执行 OpenVLA/Octo/量化，不能称推理已复现。
16. **03 向量优化**：区别位置残差与相对向量残差；坐标、单位和尺度应先对齐；把初值/边界改为显式参数，返回优化结果以检查成功与残差，标记依赖的 FK/Jacobian/关键点提取为骨架。没有实现完整手模型或验证拇指方法优劣。
17. **04 微调**：去掉“完整可直接运行/任意 checkpoint”的认证式措辞；按官方源码安装 LIBERO，明确任务构造不下载数据；纠正评估 CLI 为 `--task_suite_name`；区分 HDF5/RLDS、checkpoint 统计与基础 OpenVLA 分位数合同。LoRA A/B 形状及参数数公式修正，不再声称 rank32 固定 7M/24GB。20 次评估可靠、100–500 样本足够、成功率 >50% 通关改为证据协议；权重 warm start 与完整恢复训练区分。执行源码未修，数据加载、token 标签、归一化、checkpoint 装载仍需整合和端到端验证。
18. **04 Landmark**：补 NumPy、无检测结果保护与资源释放；强调 legacy/Tasks API 不可混用、图像归一化点与 world 米制点的差异、左右手镜像约定；UDP 打包不保证同步；`ctrl` 只有在正确角位置执行器合同时才能解释为角度。相机、网络、占位 XML、设备桥未测试。
19. **05 完整重定向**：纠正“减手腕即可消除距离/旋转/尺寸”的解释；先坐标合同和轴对齐再镜像，无量纲坐标不能直接当米。LM+bounds 改 TRF 并检查求解状态；真实型号限位与教学 `[0,1.2]` 分开。freejoint 分别使用 qpos/dof 地址、四元数归一化和 mj_forward；状态重置不是急停。EMA 更小 alpha 才更平滑、离线样条与实时滤波区分，插值方案均有可达/过冲风险。真实手 FK、GeoRT 驱动、相机标定、检测延迟/精度及硬件安全仍未实现/验证。
20. **05 世界模型**：UniSim 更正为视频扩散；MimicGen 数据生成不自动等于学习世界模型，MBPO 不等于 MPC 搜索。4D→16D 是扩维；RSSM 确定/随机分支不是运动学/碰撞的物理分类；RSSM 伪代码包含动作并显式标注非 Python 程序。示意成绩不作为运行证据，不给四种融合做通用胜负排名。DreamZero、DreamDojo、RISE 等推荐条目及全部论文论断未逐篇重新查证；没有训练或复现。

## 官方证据与核对层

以下仅列本轮实际用于核对的主源，不表示每篇论文或整个 API 文档逐行读完。

- [MediaPipe 官方输出定义](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)：核对图像归一化坐标、world 米制与原点；没有跑检测器。
- [OpenVLA 官方仓库](https://github.com/openvla/openvla)：核对安装约束、图像处理与单图推理示例。[官方 predict_action 实现](https://huggingface.co/openvla/openvla-7b/blob/main/modeling_prismatic.py)：核对第一条生成序列、动作维度及 q01/q99/mask 反归一化。
- [Octo 官方 notebook](https://github.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb)：核对 observations、mask、随机键、统计与数组返回；未装载检查点。
- [CLIP 官方模型配置](https://huggingface.co/openai/clip-vit-base-patch32/blob/main/config.json)：核对 image_size=224、patch_size=32；[RT-2 官方项目页](https://robotics-transformer2.github.io/)：核对 PaLI-X / PaLM-E 谱系。
- [LIBERO 官方 README](https://github.com/Lifelong-Robot-Learning/LIBERO)：核对源码安装、独立 demonstration 下载；本仓库评估脚本 CLI 从本地源核对。
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)：核对 LM 不支持 bounds 与 TRF 支持边界，并在本机复现差异。
- [MuJoCo 数据结构](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html)：核对 `jnt_qposadr` 与 `jnt_dofadr`。本机用两个 freejoint 验证两类地址确实不同。
- [LoRA 原论文](https://arxiv.org/abs/2106.09685)：低秩增量结构；本轮同时用矩阵维数检查 A/B 乘法，不用固定总参数数替代实际模型统计。
- [UniSim 原论文](https://arxiv.org/html/2310.06114v3)、[MimicGen 官方页](https://mimicgen.github.io/)、[MBPO 原论文](https://arxiv.org/abs/1906.08253)：核对已修的模型/算法类别，未复现论文实验。

## 实际执行与结果

1. 从 `tutorials/05-complete-pipeline/README.md` 提取真实 `vector_retarget` 函数，用恒等 FK 测试夹具执行带边界问题：残差 **4.96269692007445e-14**。同一环境复现旧 `method='lm', bounds=...` 抛 `ValueError`。这只测试求解器接口，不代表真实手模型精度。
2. 从同一文档提取真实 `reset_hand_position` 函数，用两个 freejoint 的最小 MJCF 执行：第二关节 qpos 地址 **7**、dof 地址 **6**；重置后第一个关节 qpos/速度保持不变，目标四元数归一化，指定速度清零。未跑 viewer、硬件或长时间稳定性测试。
3. 执行 `tutorials/02-action-representation/README.md` 实际 FK 与 IK 示例：残差 **7.161364213600686e-05**，小于文中 1e-4 门槛。
4. 所有这组 tutorials README 的 Python fenced code 经 AST 语法解析通过；这不检查未定义接口、依赖或运行时行为，伪代码仍是伪代码。
5. 全仓 `scripts/check_markdown_format.py`：空白、编码、数学格式通过。全仓 `scripts/check_markdown_links.py`：仓库内链接通过。没有以此声称 GitHub 渲染或公式知识正确。
6. 先前 foundations 文档回归文件 `tests/test_content_corrections.py` 本轮重跑：**7 passed / 1 skipped**，跳过项因为本机无 torch。该文件不覆盖这次所有教程修订。

## 后续接手顺序

先独立复核 10 篇教程的修改，再处理范围外的 3 篇 README 已确认问题。随后对真实训练流程做单批数据/标签/统计检查；具备单独资源与授权后才谈模型下载、训练与闭环评估。执行源码里的控制安全缺陷不能由本文新增警示代替修复。发布时应保持“已改、已测、未复核、未执行”四种状态，不宣称全书 100% 正确。

## 补充：学习界面静态只读检查

主代理另行委托对下列 3 个界面文件进行有界只读审查。本审查者均首行读到 EOF，未改文件，未运行浏览器或界面交互。

| 文件 | 行数 | 审查时 SHA256 |
|---|---:|---|
| `docs/javascripts/learning-shell.js` | 107 | `b5d4bd5c2de5ab6040d0f536fbf7f0575ae203590fe4beb6084fc4eca3258a20` |
| `docs/stylesheets/learning-shell.css` | 166 | `fbf2a542be53e899da40e1ea06653ccc74755b92b39a34011ee9a1ed5cb57c02` |
| `docs/overrides/main.html` | 17 | `2ee65c9c6076af009264410f47b604e7cca5ddcd6d9325fc2635420820c289be` |

未发现新的确定 P1/P2 阅读或导航阻断：书签 slug 经过格式检查，标题用 textContent；按首页 atlas base 拼接的链接与当前知识库目录结构相符。没有 JavaScript 时工具条/过滤器隐藏，正文和章节列表仍在。专注模式可通过原生按钮退出，字号 select 有标签，反馈区有 status/aria-live。

P3 韧性项（未修）：JS 68–74 行只校验书签格式，没有核对章节/小节是否仍存在；将来条目改名后，本地旧书签可能仍显示但指向失效链接。没有实际测试颜色对比、键盘焦点、读屏、窄屏或 instant-navigation 的运行时生命周期，不能称完整可访问性认证。
