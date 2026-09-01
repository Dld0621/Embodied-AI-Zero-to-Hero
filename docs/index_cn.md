---
hide:
  - navigation
  - toc
---

<section class="dof-intro">
  <div class="dof-kicker">具身智能 · 学习与工程</div>
  <h1>理解闭环。<br>构建系统。<br>证明结果。</h1>
  <p class="dof-lead">一套连接第一性原理、可运行 Pipeline、科研路线与明确证据边界的双语课程体系。</p>
  <div class="dof-actions">
    <a class="dof-button dof-button--primary" href="../start-here-cn/">第一次来？从这里开始</a>
    <a class="dof-button" href="../curriculum_cn/">打开课程合同</a>
    <a class="dof-button" href="../knowledge-system/README_CN/">解析前置依赖</a>
    <a class="dof-button" href="../pipelines/README_CN/">探索 Pipeline</a>
    <a class="dof-button" href="../">English</a>
  </div>
</section>

<div class="dof-metrics" aria-label="仓库范围">
  <div class="dof-metric"><strong>45</strong><span>知识节点</span></div>
  <div class="dof-metric"><strong>14</strong><span>基础课程</span></div>
  <div class="dof-metric"><strong>12</strong><span>能力模块</span></div>
  <div class="dof-metric"><strong>11</strong><span>工程 Pipeline</span></div>
  <div class="dof-metric"><strong>7</strong><span>科研路线</span></div>
  <div class="dof-metric"><strong>3</strong><span>毕业项目</span></div>
</div>

## 按目标选择入口

<div class="dof-grid">
  <a class="dof-card" href="../start-here-cn/">
    <span class="dof-card__index">起步</span>
    <h3>完成第一小时挑战</h3>
    <p>用证据自测定位起点，生成个人路线，并保存第一张实验卡。</p>
  </a>
  <a class="dof-card" href="../curriculum_cn/">
    <span class="dof-card__index">学习</span>
    <h3>建立前置依赖链</h3>
    <p>沿六个阶段从实验纪律和数学基础进入任务系统与部署证据。</p>
  </a>
  <a class="dof-card" href="../setup/README_CN/">
    <span class="dof-card__index">准备</span>
    <h3>配置可复现工作站</h3>
    <p>选择经过审查的技术栈、隔离依赖、运行分层检查，并保留环境回执。</p>
  </a>
  <a class="dof-card" href="../pipelines/README_CN/">
    <span class="dof-card__index">构建</span>
    <h3>执行一个完整系统</h3>
    <p>从输入开始，保留中间产物，测量结果，并按阶段定位失败。</p>
  </a>
  <a class="dof-card" href="../learning-paths/README_CN/">
    <span class="dof-card__index">科研</span>
    <h3>把问题转化成证据</h3>
    <p>选择基线、冻结协议、设计消融，并声明晋级门禁。</p>
  </a>
  <a class="dof-card" href="../specializations/README_CN/">
    <span class="dof-card__index">专项</span>
    <h3>从零进入 VLA 与 WAM</h3>
    <p>理解各算法族，按约束选型，建立匹配基线，再进入科研级实验。</p>
  </a>
</div>

## 一个完整闭环

<div class="dof-loop" aria-label="具身智能闭环">
  <div><span>01</span><strong>观测</strong><small>传感器、坐标系、状态与不确定性</small></div>
  <i>→</i>
  <div><span>02</span><strong>决策</strong><small>目标、策略、推理与预测</small></div>
  <i>→</i>
  <div><span>03</span><strong>行动</strong><small>控制、限位、Watchdog 与停止</small></div>
  <i>→</i>
  <div><span>04</span><strong>评估</strong><small>任务指标、失败与学习</small></div>
</div>

每种方法最终都必须把观测连接到可评估行动。看起来合理的命令、开环 Loss 或已完成脚本本身都不等于任务成功。

## 七条科研路线

<div class="dof-route-grid">
  <a class="dof-route" href="../learning-paths/README_CN/#foundation-models-vla"><span>01</span><strong>基础模型与 VLA</strong><small>策略 · 适配器 · 消融</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#manipulation-imitation"><span>02</span><strong>操作与模仿学习</strong><small>基线 · 失败 · 闭环</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#dexterity-teleoperation"><span>03</span><strong>灵巧操作与遥操作</strong><small>重定向 · 接触 · 保持</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#navigation-embodied-agents"><span>04</span><strong>导航与具身智能体</strong><small>状态 · 规划 · 恢复</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#humanoids-locomotion"><span>05</span><strong>人形与运动控制</strong><small>运动 · 鲁棒性 · 安全</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#perception-world-models"><span>06</span><strong>感知与世界模型</strong><small>不确定性 · 预测 Rollout</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#simulation-data-evaluation"><span>07</span><strong>仿真、数据与评估</strong><small>数据说明 · 基准 · 门禁</small></a>
</div>

## Pipeline 覆盖

<div class="dof-legend" aria-label="Pipeline 证据图例">
  <span class="dof-legend__smoke">Smoke-tested · 8</span>
  <span class="dof-legend__interface">Interface-tested · 2</span>
  <span class="dof-legend__documented">硬件依赖 · 1</span>
</div>

<div class="dof-coverage">
  <a href="../pipelines/01-simulation-data/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>仿真与数据</strong><small>任务 → 专家 → 轨迹 → QA</small></a>
  <a href="../pipelines/02-vla-policy/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>VLA 策略</strong><small>多模态数据 → 策略 → 评估</small></a>
  <a href="../pipelines/03-world-model-planning/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>世界模型</strong><small>转移 → Rollout → 规划</small></a>
  <a href="../pipelines/04-rl-post-training/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>RL 后训练</strong><small>MDP → PPO → 评估 → 回归</small></a>
  <a href="../pipelines/05-rfm-cross-embodiment/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>RFM 与跨本体</strong><small>统一 Schema → 适配器 → 安全层</small></a>
  <a href="../pipelines/06-embodied-reasoning/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>具身推理</strong><small>指令 → 技能 → 反馈 → 重规划</small></a>
  <a href="../pipelines/07-sim-to-real/"><span class="dof-status dof-status--documented">DOC</span><strong>Sim-to-Real</strong><small>HIL → 影子模式 → 受控部署</small></a>
  <a href="../pipelines/08-dexterous-retargeting/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>灵巧手重定向</strong><small>关键点 → 几何 → 优化</small></a>
  <a href="../pipelines/09-perception-state-estimation/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>感知与状态</strong><small>标定 → 同步 → 融合</small></a>
  <a href="../pipelines/10-navigation-locomotion/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>导航与运动</strong><small>状态 → 规划 → 控制 → 恢复</small></a>
  <a href="../pipelines/11-dexterous-manipulation/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>灵巧操作</strong><small>预抓取 → 接触 → 抬升 → 保持</small></a>
</div>

## 证据先于结论

<div class="dof-proof">
  <strong>能够执行不等于有性能。仿真不等于真机验证。</strong>
  <p>仓库明确区分导入、Smoke、确定性测试、Benchmark 证据和有边界真机验证；较低等级不能推出较高等级。</p>
  <p><a href="../VALIDATION/">验证规范 →</a> · <a href="../CLAIM_REVIEW/">真实性门禁 →</a> · <a href="../benchmark_report/">基准报告 →</a></p>
</div>

## 本地验证

```bash
python scripts/run_knowledge_map.py --validate
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/run_curriculum.py --validate
python scripts/audit_repository.py
python -m pytest tests/ -q
```

根目录的 [English README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README.md) 和 [中文 README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README_CN.md)提供仓库总览；本站承载更细致的学习与工程内容。
