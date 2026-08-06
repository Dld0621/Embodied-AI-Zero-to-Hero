<div class="dof-landing" markdown="1">
  <div class="dof-kicker">具身智能 · 开放研究栈</div>

# 从第一性原理到物理行动。

  <p class="dof-lead">
    补齐基础知识，执行完整管线，再用证据判断结果。一个面向学习、复现与研究的双语具身智能系统。
  </p>

  <div class="dof-actions">
    <a class="dof-button dof-button--primary" href="../foundations/00-roadmap/">开始学习</a>
    <a class="dof-button" href="../field-map-cn/">查看领域地图</a>
    <a class="dof-button" href="../pipelines/README_CN/">探索管线</a>
    <a class="dof-button" href="../">English</a>
  </div>
</div>

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
  <div class="dof-metric"><strong>10</strong><span>工程管线</span></div>
  <div class="dof-metric"><strong>5</strong><span>证据等级</span></div>
  <div class="dof-metric"><strong>中 · EN</strong><span>双语入口</span></div>
</div>

## 选择你的入口

<div class="dof-grid">
  <a class="dof-card" href="../foundations/00-roadmap/">
    <span class="dof-card__index">01 · 学习</span>
    <h3>建立完整知识模型</h3>
    <p>数学、机器学习、坐标系、运动学、感知、控制、系统、安全与评估。</p>
  </a>
  <a class="dof-card" href="../field-map-cn/">
    <span class="dof-card__index">02 · 定位</span>
    <h3>选择研究方向</h3>
    <p>把方向、前置知识、工程管线和当前证据放在一张可检查的地图中。</p>
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
</div>

## 当前系统覆盖

<div class="dof-coverage">
  <a href="../pipelines/01-simulation-data/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>仿真与数据</strong><small>任务 → 专家 → 轨迹 → QA</small></a>
  <a href="../pipelines/02-vla-policy/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>VLA 策略</strong><small>多模态数据 → 策略 → 闭环评估</small></a>
  <a href="../pipelines/03-world-model-planning/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>世界模型</strong><small>转移 → 预测 → rollout → 规划</small></a>
  <a href="../pipelines/04-rl-post-training/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>RL 后训练</strong><small>MDP → PPO → 评估 → 回归</small></a>
  <a href="../pipelines/05-rfm-cross-embodiment/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>RFM / 跨本体</strong><small>统一协议 → 适配 → 安全层</small></a>
  <a href="../pipelines/06-embodied-reasoning/"><span class="dof-status dof-status--interface">INTERFACE</span><strong>具身推理</strong><small>指令 → 子目标 → 技能 → 重规划</small></a>
  <a href="../pipelines/08-dexterous-retargeting/"><span class="dof-status dof-status--smoke">SMOKE</span><strong>灵巧手重定向</strong><small>关键点 → 几何 → 优化 → 时序</small></a>
  <a href="../pipelines/07-sim-to-real/"><span class="dof-status dof-status--documented">DOC</span><strong>Sim-to-Real</strong><small>Replay → HIL → 影子 → 受控部署</small></a>
  <a href="../pipelines/09-perception-state-estimation/"><span class="dof-status dof-status--documented">DOC</span><strong>感知与状态估计</strong><small>标定 → 同步 → 融合 → 置信度</small></a>
  <a href="../pipelines/10-navigation-locomotion/"><span class="dof-status dof-status--documented">DOC</span><strong>导航与运动</strong><small>状态 → 规划 → 控制 → 恢复</small></a>
</div>

## 证据先于结论

<div class="dof-proof">
  <strong>能执行不等于有性能，仿真通过也不等于真机验证。</strong>
  <p>仓库明确区分 import、smoke、确定性测试、benchmark 与硬件验证。较低等级不能推出较高等级。</p>
  <p><a href="../VALIDATION/">阅读验证政策 →</a></p>
</div>

## 验证仓库

```bash
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --list
python scripts/audit_repository.py
python -m pytest tests/ -q
```

根目录的 [中文 README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README_CN.md) 是精简产品首页；文档站提供更完整的学习与工程层。
