---
hide:
  - navigation
  - toc
---

<section class="dof-landing">
  <div class="dof-landing__copy">
    <div class="dof-kicker">具身智能 · 开放研究栈</div>
    <h1>从第一性原理<br><span>到物理行动。</span></h1>
    <p class="dof-lead">
      补齐基础知识，执行完整管线，再用证据判断结果。一个面向学习、复现与研究的双语具身智能系统。
    </p>
    <div class="dof-actions">
      <a class="dof-button dof-button--primary" href="../foundations/00-roadmap/">开始学习</a>
      <a class="dof-button" href="../setup/README_CN/">配置环境</a>
      <a class="dof-button" href="../learning-paths/README_CN/">选择路线</a>
      <a class="dof-button" href="../pipelines/README_CN/">探索管线</a>
      <a class="dof-button" href="../">English</a>
    </div>
  </div>
  <aside class="dof-signal" aria-label="仓库证据状态">
    <div class="dof-signal__top"><span>证据状态</span><span>实时</span></div>
    <strong>8 / 11</strong>
    <p>条管线包含可运行 smoke 路径</p>
    <div class="dof-signal__rail" aria-hidden="true">
      <span class="dof-signal__smoke"></span><span class="dof-signal__interface"></span><span class="dof-signal__documented"></span>
    </div>
    <dl>
      <div><dt>Smoke-tested</dt><dd>8</dd></div>
      <div><dt>Interface-tested</dt><dd>2</dd></div>
      <div><dt>硬件依赖</dt><dd>1</dd></div>
    </dl>
    <a href="../VALIDATION/">阅读证据规范 <span aria-hidden="true">↗</span></a>
  </aside>
</section>

<div class="dof-section-label">具身智能闭环</div>

<div class="dof-loop" aria-label="具身智能闭环">
  <div><span>01</span><strong>感知</strong><small>观测与状态</small></div>
  <i>→</i>
  <div><span>02</span><strong>理解</strong><small>目标与世界模型</small></div>
  <i>→</i>
  <div><span>03</span><strong>行动</strong><small>策略、控制与安全</small></div>
  <i>→</i>
  <div><span>04</span><strong>学习</strong><small>反馈与证据</small></div>
</div>

<div class="dof-metrics">
  <div class="dof-metric"><strong>14</strong><span>基础课程</span></div>
  <div class="dof-metric"><strong>11</strong><span>工程管线</span></div>
  <div class="dof-metric"><strong>7</strong><span>科研路线</span></div>
  <div class="dof-metric"><strong>中 · EN</strong><span>双语入口</span></div>
</div>

## 选择你的入口

<div class="dof-grid">
  <a class="dof-card" href="../foundations/00-roadmap/">
    <span class="dof-card__index">01 · 学习</span>
    <h3>建立完整知识模型</h3>
    <p>数学、机器学习、坐标系、运动学、感知、控制、系统、安全与评估。</p>
  </a>
  <a class="dof-card" href="../learning-paths/README_CN/">
    <span class="dof-card__index">02 · 定位</span>
    <h3>选择研究方向</h3>
    <p>从问题出发，依次进入前置课程、Pipeline、交付物、指标与晋级门槛。</p>
  </a>
  <a class="dof-card" href="../pipelines/README_CN/">
    <span class="dof-card__index">03 · 构建</span>
    <h3>执行一个完整系统</h3>
    <p>从输入开始运行每个阶段，保留产物，并检查明确的晋级门禁。</p>
  </a>
  <a class="dof-card" href="../benchmark_report/">
    <span class="dof-card__index">04 · 测量</span>
    <h3>带着上下文比较</h3>
    <p>把协议、数据预算、episode 数、负结果和原始产物边界一起阅读。</p>
  </a>
  <a class="dof-card dof-card--wide" href="../setup/README_CN/">
    <span class="dof-card__index">05 · 准备</span>
    <h3>构建可复现工作站</h3>
    <p>选择受支持技术栈、隔离依赖、分层验收，并保留完整环境回执。</p>
  </a>
</div>

<div class="dof-section-head">
  <div><span>目标驱动课程</span><h2>七条科研路线</h2></div>
  <a class="dof-section-link" href="../learning-paths/README_CN/">打开完整路线图 →</a>
</div>

<div class="dof-route-grid">
  <a class="dof-route" href="../learning-paths/README_CN/#foundation-models-vla"><span>01</span><strong>基础模型与 VLA</strong><small>策略 · 适配器 · 消融</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#manipulation-imitation"><span>02</span><strong>操作与模仿学习</strong><small>基线 · 失败 · 闭环</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#dexterity-teleoperation"><span>03</span><strong>灵巧操作与遥操作</strong><small>重定向 · 抓取 · 分层证据</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#navigation-embodied-agents"><span>04</span><strong>导航与具身智能体</strong><small>状态 · 规划 · 恢复</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#humanoids-locomotion"><span>05</span><strong>人形与运动控制</strong><small>运动 · 安全 · 迁移</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#perception-world-models"><span>06</span><strong>感知与世界模型</strong><small>不确定性 · 预测 rollout</small></a>
  <a class="dof-route" href="../learning-paths/README_CN/#simulation-data-evaluation"><span>07</span><strong>仿真、数据与评测</strong><small>数据说明 · 基准 · 门槛</small></a>
</div>

<div class="dof-section-head">
  <div><span>管线状态</span><h2>当前系统覆盖</h2></div>
  <div class="dof-legend" aria-label="管线证据图例">
    <span class="dof-legend__smoke">Smoke 8</span>
    <span class="dof-legend__interface">Interface 2</span>
    <span class="dof-legend__documented">Documented 1</span>
  </div>
</div>

<div class="dof-coverage">
  <a href="../pipelines/01-simulation-data/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>仿真与数据</strong><small>任务 → 专家 → 轨迹 → QA</small></a>
  <a href="../pipelines/02-vla-policy/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>VLA 策略</strong><small>多模态数据 → 策略 → 闭环评估</small></a>
  <a href="../pipelines/03-world-model-planning/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>世界模型</strong><small>转移 → 预测 → rollout → 规划</small></a>
  <a href="../pipelines/04-rl-post-training/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>RL 后训练</strong><small>MDP → PPO → 评估 → 回归</small></a>
  <a href="../pipelines/05-rfm-cross-embodiment/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>RFM / 跨本体</strong><small>统一协议 → 适配 → 安全层</small></a>
  <a href="../pipelines/06-embodied-reasoning/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>具身推理</strong><small>指令 → 子目标 → 技能 → 重规划</small></a>
  <a href="../pipelines/08-dexterous-retargeting/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>灵巧手重定向</strong><small>关键点 → 几何 → 优化 → 时序</small></a>
  <a href="../pipelines/07-sim-to-real/"><span class="dof-status dof-status--documented">DOC</span><strong>Sim-to-Real</strong><small>Replay → HIL → 影子 → 受控部署</small></a>
  <a href="../pipelines/09-perception-state-estimation/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>感知与状态估计</strong><small>标定 → 同步 → 融合 → 置信度</small></a>
  <a href="../pipelines/10-navigation-locomotion/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>导航与运动</strong><small>状态 → 规划 → 控制 → 恢复</small></a>
  <a href="../pipelines/11-dexterous-manipulation/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>灵巧抓取与精细操作</strong><small>预抓取 → 接触 → 抬升 → 保持</small></a>
</div>

## 证据先于结论

<div class="dof-proof">
  <strong>8 条可运行 smoke · 2 条接口路径 · 1 条硬件依赖契约。</strong>
  <p>能执行不等于有性能，合成仿真也不等于真机验证。仓库明确区分 import、smoke、确定性测试、benchmark 与硬件验证；较低等级不能推出较高等级。</p>
  <p><a href="../VALIDATION/">验证政策 →</a> · <a href="../CLAIM_REVIEW/">真实性门禁 →</a></p>
</div>

## 验证仓库

```bash
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/run_pipeline.py --list
python scripts/audit_repository.py
python -m pytest tests/ -q
```

根目录的 [中文 README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README_CN.md) 是精简产品首页；文档站提供更完整的学习与工程层。
