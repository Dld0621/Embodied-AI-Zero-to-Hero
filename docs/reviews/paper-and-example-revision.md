# 论文导读与示例说明修订

2026-09-05 · 主代理定向修订；[总审查交接](content-audit-handoff.md)。

主代理全文读取 `docs/02-key-papers.md` 和三个示例 README 后，修正 H01、H07、H08 的已确认段落。它们的主审基线已读记录保留，修订后没有冒称完成全篇独立外部复核。

## 论文导读 H01

修正 RT-1 / SPOC 论文和代码入口、RT-1 的 81→8→48 token 数量及非动作自回归；区分 vanilla OpenVLA 的离散动作交叉熵与后续变体；修正 π0 的 PaliGemma / action expert / 10 步采样与执行频率；补充 Octo checkpoint/输入合同；区分 CLIP 与 DINOv2；修正 ACT 分块和测试潜变量说明。扩散/CLIP 机制片段明确标成伪代码，不再混入可执行代码块。

对无统一口径的“最活跃”“85% 性能”“固定微调步数”“单次前向”“任意机器人”“USD 2k 整套硬件”等断言撤回或限定。4 篇世界模型条目保留，区分论文报告、模型推理耗时与本仓库未复现结果。

实际核对范围为原文的相关节或作者项目页，不是全部论文、附录和实验逐行复现：

- [RT-1 §5.1 / 附录 D.4](https://arxiv.org/html/2212.06817v1)：token 数量、动作离散化与推理设计。
- [OpenVLA §3.2–3.4 / 实验设置](https://arxiv.org/html/2406.09246v3)：token、损失、数据、基础输入范围。
- [π0 §IV / 附录 A-D](https://arxiv.org/html/2410.24164v1) 与 [openpi](https://github.com/Physical-Intelligence/openpi)：架构、动作专家、采样和调度。
- [Octo §III / 附录 D](https://arxiv.org/html/2405.12213v2)：readout、diffusion head、checkpoint 与接口。
- [ACT 作者项目](https://tonyzhaozh.github.io/aloha/)、[SPOC 作者项目](https://spoc-robot.github.io/) 与其论文/代码链接：修正潜变量、标题和观察/训练设置。
- [LaDi-WM](https://arxiv.org/abs/2505.11528)、[DreamDojo](https://arxiv.org/abs/2602.06949)、[RISE](https://arxiv.org/abs/2602.11075)、[PointWorld](https://arxiv.org/abs/2601.03782) 的摘要/书目信息：限定阅读入口和所报结果范围；没有复算全部表格。

## 示例说明 H07 / H08

- DexMV-style README：取消无同协议证据的精度榜与实时/真实输入保证；区分 Huber 和按 delta 缩放的 Smooth L1；修复无效代码语法并标明骨架；图像归一化坐标不能当米，跨数据集骨架下标不可照搬。这里只改说明，不实现相机标定、完整骨架转换或控制器。
- Robot foundation models README：所有入口改为从仓库根目录运行，修正 converter 的导入路径；加入 F002/F003 未修控制缺陷警告；统一 protocol 不意味着换模型无需改调度或核对动作合同。
- SmolVLA datasets README：显式区分 `pushcube_mock_parquet` 与真实数据格式；对照本地 converter 确认 mock meta 不含 action_type，metadata 文件存在不等于格式有效。修正多层 cd 和训练/检查路径。提醒不存在的路径可能被当前 launcher 判成 Hub 标识，真实训练前必须验证路径和数据加载。

新增回归只检查已确认书目和版本措辞、真实脚本路径、当前 writer 的 metadata 键以及文档 Python 语法。不验证每条科学论断；没有下载模型、采集数据、训练、GUI 或真机执行。原 [支持文档审查快照](supporting-review-handoff.md) 的旧 hash 是历史记录，此后这三篇的修订以 git diff 为准。
