# Changelog

> 所有值得注意的变更都将记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Fixed
- 修正 DIAMOND 官方仓库链接：`ethz-rl/diamond` → `eloialonso/diamond`（作者 Vincent Micheli 的个人仓库）
- 修正 IRIS 官方仓库链接：`janner/iris` → `eloialonso/iris`
- 修复 `sim_closed_loop_demo.py` 中 `ScriptedPolicy._get_gripper_pos` 使用未定义 `self.gripper_id` 的隐患
- 修复 `freshman_zero_to_one.py` 中 `HumanHandVisualizer.__init__` 的 `self.ay` 笔误
- 修复 `world_model_vla_pipeline.py` 中 `fusion_1_data_generator` 注释与实现不符的问题，并增加 WM reward 预测作为增强信号
- 修复 `world_model_vla_pipeline.py` 中 `fusion_4_wam` 对 `_cached_data` 的隐式依赖，增加前置检查与友好错误提示

### Added
- 新增 `CONTRIBUTING.md`：完整的贡献指南，包含 Issue/PR 规范、内容质量标准、审查清单
- 新增 `CHANGELOG.md`：版本变更记录
- 新增 `requirements.txt`：标准 Pip 依赖文件（与 `setup/environment.yml` 同步）
- 新增 `tests/test_imports.py`：基础导入测试，确保核心模块无语法错误
- 新增 `docs/19-sim-to-real-guide.md`：Sim-to-Real 完整实战指南（含域随机化、系统辨识、延迟补偿、触觉传感器适配）
- 新增 `docs/20-vla-deployment-guide.md`：VLA 部署优化与边缘计算指南（量化、TensorRT、Jetson 部署、异步流水线）
- 新增 `examples/train_diffusion_policy.py`：完整的 Diffusion Policy 训练脚本，支持合成数据和 ALOHA 数据集
- 新增 `docs/README.md`：文档索引，包含完整项目结构树、核心概念速查、代码速查、外部资源
- 面试题升级：`docs/05-interview-prep.md` 新增 12 题（Q89-Q100），覆盖手写代码题、系统设计题、2026 前沿论文面试题
- 研究趋势升级：`docs/17-research-trends-and-positioning.md` 新增 2026 年中 9 项前沿工作速览（ZR-0、Pose-VLA、Xiaomi-Robotics-1、Hy-Embodied、ACE-Ego、DexSim2Real、Phys2Real、MoDE-VLA、CMU Touch Dreaming）
- 交叉引用：教育仓库与工程项目（OmniHand MuJoCo、OmniHand v19、GeoRT）之间建立双向链接

### Changed
- README.md 全面重构：新增视觉框架图、四大支柱详解、30 秒快速开始、适合人群推荐表
- 合并 Dexterous-Retargeting-Guide 仓库内容，统一为 Embodied AI Zero to Zero
- **README 骨架重写（2026.07.24）**：
  - 重新定义项目定位：Dexterous Retargeting 为核心研究主线，VLA/World Models/RL 为策略/预测/优化层
  - 增加端到端系统链路 Mermaid 图，四个模块回答不同问题
  - 新增 Project Status 表（✅/🟡/⏳/🔒 状态体系）
  - 新增 Choose Your Path 用户入口表
  - Quick Start 精简为单一入口（freshman_zero_to_one.py），明确 Input/Method/Output/Evaluation
  - 四个 Track 采用统一模板：Definition → Pipeline → Input/Method/Output/Evaluation → Learning Levels → Known Limitations
  - 新增 Benchmark 区域（TBD 占位，诚实声明当前 CI 覆盖范围）
  - 新增 Supported Robots 分级表（Model Loaded / IK Verified / Benchmark Verified / Hardware Verified）
  - 新增 Reproducibility 区域（测试环境 + L1-L5 复现等级）
  - 新增 Research Roadmap（Phase 1-5 时间线）
  - 删除所有 `../../` 本地路径引用
  - 长内容（完整文档树、代码速查、核心概念）迁移到 `docs/README.md`

---

## [2026.07.24]

### Added
- 完成 Dexterous-Retargeting-Guide 与 VLA-Zero-to-Hero 仓库融合
- 新增 27 篇核心文档，覆盖重定向、VLA、RL、世界模型四大支柱
- 新增 18+ 个可运行示例代码
- 新增 10 阶段教程（双轨道：重定向轨道 + VLA 轨道）
- 新增 `docs/18-frontier-papers-online.md`：20+ 篇前沿论文在线链接
- 新增 `docs/17-research-trends-and-positioning.md`：2026 研究趋势分析（六大研究转向）
- 新增 `docs/16-arxiv-retargeting-scan.md`：80+ 篇 Arxiv 重定向论文扫描

### Changed
- 统一环境依赖为 `embodied-ai` Conda 环境
- 重写 README.md 以反映四大支柱内容结构
- 更新 LICENSE 版权持有者为 Embodied AI Zero to Zero Contributors

---

## [2026.07.20]

### Added
- 新增 `examples/freshman_zero_to_one.py`：大一新生零外部依赖的完整重定向 pipeline
- 新增 `examples/dexmv_style_retargeting/`：DexMV SLSQP + Huber Loss 高精度实现
- 新增 `docs/11-dexmv-research-guide.md`：DexMV 论文深度解读
- 新增 `docs/12-freshman-zero-to-one.md`：从零开始的重定向实战指南

---

## [2026.07.15]

### Added
- 新增 VLA 内容模块：`docs/01-what-is-vla.md`、`docs/02-key-papers.md`、`docs/03-learning-path.md`
- 新增 `examples/minimal_vla.py`：最小可运行 VLA 架构演示
- 新增 `examples/vla_demo.py`：OpenVLA / SmolVLA 推理演示
- 新增 `tutorials/03-simple-vla/`：从零搭建 VLA 教程
- 新增 `tutorials/04-fine-tuning/`：LIBERO 微调完整代码

---

## [2026.07.10]

### Added
- 新增 RL 与世界模型模块
- 新增 `examples/rl_demo.py`：Q-Learning / SAC + HER 演示
- 新增 `examples/world_model_demo.py`：线性世界模型 + MPC 规划
- 新增 `examples/dreamer_rssm.py`：DreamerV3 RSSM 完整实现
- 新增 `docs/06-rl-fundamentals-for-vla.md`：面向 VLA 学习者的 RL 基础
- 新增 `docs/07-world-models-for-vla.md`：面向 VLA 学习者的世界模型指南

---

## [2026.07.01]

### Added
- 项目初始化：Embodied AI Zero to Zero
- 基础文档：关节概念、重定向概念、方法分类、评估指标
- 核心示例：`fk_ik_demo.py`、`landmark_to_joint.py`、`minimal_retargeting.py`
- 资源索引：`resources/README.md`、`setup/environment.yml`
