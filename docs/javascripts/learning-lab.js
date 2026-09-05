/* Dependency-free teaching UI. No telemetry, remote computation, or persistent storage. */
(() => {
  "use strict";
  const ids = ["frames", "kinematics", "control", "timing", "evaluation"];
  const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
  let current = null;

  function mount(root) {
    if (root.dataset.initialized) return;
    const M = window.EmbodiedLabModels;
    if (!M) return; // The static worked examples below are the fallback.
    const f = M.formatNumber;
    root.dataset.initialized = "true";
    const zh = root.dataset.labLang === "zh";
    const t = (cn, en) => zh ? cn : en;
    const definitions = {
      frames: {
        tab: t("坐标变换", "Frames"), stage: t("01 · 观测", "01 · Observe"),
        title: t("同一个点，为什么有两组坐标？", "One point. Why two coordinates?"),
        lead: t("先预测：传感器转过 90°，局部的“前方”会指向世界的哪里？", "Predict first: after a 90° sensor rotation, where does local ‘forward’ point in the world?"),
        fields: [
          ["frame-theta", t("传感器朝向 θ", "Sensor angle θ"), -180, 180, 90, 1, "°"],
          ["frame-tx", t("平移 tₓ", "Translation tₓ"), -1.5, 1.5, .5, .1, "m"],
          ["frame-ty", t("平移 tᵧ", "Translation tᵧ"), -1.5, 1.5, .5, .1, "m"],
          ["frame-x", t("局部坐标 xₛ", "Local coordinate xₛ"), -1, 1, 1, .1, "m"],
          ["frame-y", t("局部坐标 yₛ", "Local coordinate yₛ"), -1, 1, 0, .1, "m"]
        ],
        legend: [t("传感器轴与世界点 pW", "Sensor axes & world point pW"), t("传感器原点 S", "Sensor origin S")],
        note: t("二维刚体变换 pW = R(θ) pS + t。θ 是传感器坐标轴在世界中的逆时针朝向；不是在旋转物体。图中长度单位均为 m。", "2D rigid transform pW = R(θ) pS + t. θ is the counterclockwise orientation of the sensor frame in world coordinates, not an active object rotation. All plot lengths are in m."),
        insight: t("试一试：只改变 tₓ。世界点会移动，但局部坐标不变。逆变换必须先减平移，再乘逆旋转。", "Try changing only tₓ. The world point moves while its local coordinates stay fixed. The inverse must subtract translation before applying inverse rotation.")
      },
      kinematics: {
        tab: t("机械臂", "Kinematics"), stage: t("02 · 几何", "02 · Geometry"),
        title: t("关节转一点，末端会往哪里走？", "A small joint turn. Where does the tip go?"),
        lead: t("改变两个关节角，再把 θ₂ 设为 0°。观察机械臂伸直时的局部运动能力。", "Change both joint angles, then set θ₂ to 0°. Observe the arm’s local motion capability when it is straight."),
        fields: [
          ["arm-q1", t("关节角 θ₁", "Joint angle θ₁"), -180, 180, 0, .1, "°"],
          ["arm-q2", t("相对关节角 θ₂", "Relative angle θ₂"), -180, 180, 90, .1, "°"],
          ["arm-target-x", t("目标 x", "Target x"), -2, 2, 1, .1, "m", "number"],
          ["arm-target-y", t("目标 y", "Target y"), -2, 2, .7, .1, "m", "number"]
        ],
        legend: [t("连杆与末端", "Links & end effector"), t("目标点 +", "Target +")],
        note: t("平面 2R 刚性机械臂，连杆长 1 m 与 0.7 m；无碰撞、动力学或关节限位模型。IK 按 θ₂ ≥ 0 的分支求解，角度取整到 0.1°，因此可能留下毫米级目标误差；det J 的关节单位采用 rad。", "Planar rigid 2R arm with 1 m and 0.7 m links; no collision, dynamics or joint-limit model. IK chooses the θ₂ ≥ 0 branch and rounds angles to 0.1°, which may leave millimetre-scale target error; joint units in det J are radians."),
        insight: t("伸直或完全折叠时 det J = 0：一个瞬时平移方向丢失。这不表示所有方向都无法运动，也不等于目标一定不可达。", "At full extension or folding, det J = 0: one instantaneous translation direction is lost. This does not mean all motion is impossible or every target is unreachable.")
      },
      control: {
        tab: t("反馈控制", "Control"), stage: t("03 · 闭环", "03 · Feedback"),
        title: t("更用力纠正，为什么反而会振荡？", "Why can stronger correction cause oscillation?"),
        lead: t("先减小 Kd，再增加观测延迟；比较整条轨迹，而不只看最后一个点。", "Lower Kd, then increase observation delay. Compare the whole trajectory, not just its final point."),
        fields: [
          ["control-kp", t("比例增益 Kp", "Proportional gain Kp"), 0, 80, 16, 1, "N/m"],
          ["control-kd", t("微分增益 Kd", "Derivative gain Kd"), 0, 20, 7.8, .2, "N·s/m"],
          ["control-delay", t("观测延迟", "Observation delay"), 0, 300, 0, 5, "ms"]
        ],
        legend: [t("实际位置 x", "Actual position x"), t("目标 1 m", "Target 1 m")],
        note: t("教学模型：1 kg 质点、阻尼 0.2 N·s/m、力限幅 ±10 N；位置和速度都延迟。步长 5 ms，半隐式 Euler 积分，共 6 s。稳定时间要求末段至少 0.5 s 保持在 ±2% 位置带内；不是长期稳定性证明。", "Teaching model: 1 kg point mass, damping 0.2 N·s/m, force clipped to ±10 N; both position and velocity are delayed. Semi-implicit Euler, 5 ms steps, 6 s horizon. Settling requires a final ≥0.5 s tail within a ±2% position band; this is not a long-term stability proof."),
        insight: t("Kp 把位置误差转成力，Kd 抑制速度。延迟让控制器对旧状态纠错；限幅使线性系统的临界阻尼公式不再能精确预测整条轨迹。", "Kp converts position error into force; Kd opposes velocity. Delay makes the controller correct an old state. Saturation prevents a linear critical-damping formula from predicting the entire trajectory exactly.")
      },
      timing: {
        tab: t("动作分块", "Action chunks"), stage: t("04 · 决策", "04 · Decide"),
        title: t("预测 16 步，为什么只执行 4 步？", "Why predict 16 steps but execute only 4?"),
        lead: t("分别改变预测长度 H、执行步数 K 和推理延迟。哪个参数真的改变重新观测的间隔？", "Change prediction horizon H, executed steps K and inference latency separately. Which actually changes the interval between observations?"),
        fields: [
          ["timing-horizon", t("预测长度 H", "Predicted horizon H"), 1, 24, 16, 1, t("步", "steps")],
          ["timing-execute", t("执行步数 K", "Executed steps K"), 1, 24, 4, 1, t("步", "steps")],
          ["timing-latency", t("推理耗时 L", "Inference latency L"), 0, 500, 100, 10, "ms"]
        ],
        legend: [t("执行动作", "Execute actions"), t("等待推理", "Wait for inference")],
        note: t("同步示意：观测 → 推理 → 执行 K 步 → 再观测。每步 50 ms，忽略观测开销；必须 K ≤ H。动作占空比不是处理器利用率。最后一步起始时的观测年龄为 L + (K−1) × 50 ms。", "Synchronous illustration: observe → infer → execute K steps → observe again. Each step lasts 50 ms; observation overhead is ignored. K ≤ H is required. Action duty is not processor utilization. Observation age at the last action’s start is L + (K−1) × 50 ms."),
        insight: t("增加 K 会摊薄推理等待占比，却让最后几步依赖更旧的观测。固定 K 和推理耗时时，仅增加 H 不会改变这个调度；真实模型的推理耗时可能随 H 变化。", "Increasing K amortizes inference waiting but makes later actions depend on older observations. With K and latency fixed, H alone does not change this schedule; real inference latency may depend on H.")
      },
      evaluation: {
        tab: t("评估证据", "Evaluation"), stage: t("05 · 证据", "05 · Evidence"),
        title: t("同样 80%，证据强度一样吗？", "Same 80%. Same strength of evidence?"),
        lead: t("对比 8/10 与 80/100。点估计相同，区间宽度会怎样变化？", "Compare 8/10 with 80/100. The point estimate is the same. What happens to interval width?"),
        fields: [
          ["eval-trials", t("总回合 n", "Total trials n"), 1, 100000, 100, 1, t("回合", "trials"), "number"],
          ["eval-successes", t("成功回合 k", "Successful trials k"), 0, 100000, 80, 1, t("回合", "trials"), "number"]
        ],
        legend: [t("点估计与 Wilson 区间", "Estimate & Wilson interval"), t("80% 参考线", "80% reference")],
        note: t("双侧约 95% Wilson 区间，z = 1.96；假设独立、同成功概率的 Bernoulli 试验。重复相邻帧、挑选最好种子或改变任务分布都不能靠增大 n 修复。0 回合没有成功率；0 次成功不等于真实概率为 0。", "Two-sided approximately 95% Wilson interval, z = 1.96; assumes independent Bernoulli trials with a common success probability. More n cannot fix correlated frames, cherry-picked seeds or a shifted task distribution. Zero trials has no success rate; zero successes does not establish a true probability of zero."),
        insight: t("这个区间描述抽样不确定性，不包含数据泄漏、仿真偏差或真机风险。比较方法还需要相同协议；两条区间是否重叠不是通用显著性检验。", "This interval addresses sampling uncertainty, not leakage, simulation bias or hardware risk. Comparing methods also needs a matched protocol; interval overlap is not a universal significance test.")
      }
    };

    function fieldMarkup(field) {
      const [name, label, min, max, value, step, unit, type = "range"] = field;
      return `<label class="eai-field"><span class="eai-field-head"><span>${label}</span><output data-value-for="${name}">${value} ${unit}</output></span><input name="${name}" aria-label="${label}" type="${type}" min="${min}" max="${max}" step="${step}" value="${value}" required></label>`;
    }
    root.innerHTML = `<div class="eai-lab-nav" role="tablist" aria-label="${t("选择实验", "Choose an experiment")}">${ids.map((id) => `<button type="button" role="tab" id="tab-${id}" data-lab-tab="${id}" aria-controls="${id}" aria-selected="false" tabindex="-1"><span>${definitions[id].stage}</span>${definitions[id].tab}</button>`).join("")}</div>` + ids.map((id) => {
      const d = definitions[id];
      return `<section id="${id}" role="tabpanel" data-lab-panel="${id}" aria-labelledby="tab-${id}" hidden>
        <header class="eai-lab-head"><span class="eai-eyebrow">${d.stage}</span><h2>${d.title}</h2><p>${d.lead}</p></header>
        <div class="eai-workbench"><div class="eai-figure"><figure><svg class="eai-chart" role="img" aria-labelledby="chart-title-${id} chart-desc-${id}"></svg><figcaption class="eai-legend"><span><i></i>${d.legend[0]}</span><span><i class="secondary"></i>${d.legend[1]}</span></figcaption></figure></div>
        <div class="eai-controls">${d.fields.map(fieldMarkup).join("")}
          <div class="eai-actions">${id === "kinematics" ? `<button type="button" data-action="inverse">${t("求解目标 IK", "Solve target IK")}</button>` : ""}
          ${id === "control" ? `<button type="button" data-preset="oscillate">${t("对比：低阻尼", "Compare: low damping")}</button>` : ""}
          ${id === "evaluation" ? `<button type="button" data-preset="small">8 / 10</button><button type="button" data-preset="medium">80 / 100</button><button type="button" data-preset="large">800 / 1000</button>` : ""}</div>
        </div></div>
        <div class="eai-result" role="status" aria-live="polite" aria-atomic="true"></div>
        <p class="eai-insight">${d.insight}</p><p class="eai-model-note">${d.note}</p>
        <details class="eai-record"><summary>${t("记录我的预测与解释（仅保留在当前页面）", "Record my prediction & explanation (this page only)")}</summary>
          <label>${t("操作前：我预测什么？", "Before: what do I predict?")}<textarea name="prediction" maxlength="4000"></textarea></label>
          <label>${t("操作后：什么变了，为什么？", "After: what changed, and why?")}<textarea name="reflection" maxlength="4000"></textarea></label>
        </details>
        <div class="eai-lab-footer"><button type="button" data-action="reset">${t("重置参数", "Reset parameters")}</button><button type="button" data-action="export">${t("导出实验记录", "Export experiment")}</button><a href="#${id}-guide">${t("推导与迁移练习 ↓", "Derivation & transfer exercise ↓")}</a></div>
        <p class="eai-export-status" aria-live="polite"></p>
      </section>`;
    }).join("");
    let selected = "frames";
    const panels = Object.fromEntries(ids.map((id) => [id, root.querySelector(`[data-lab-panel="${id}"]`)]));
    const values = (id) => {
      const result = {};
      for (const [name] of definitions[id].fields) {
        const input = panels[id].querySelector(`[name="${name}"]`);
        const valid = input.value.trim() !== "" && input.checkValidity() && Number.isFinite(Number(input.value));
        input.setAttribute("aria-invalid", String(!valid));
        if (!valid) throw new RangeError(t("请在标注范围内输入有效数值。", "Enter a valid value within the stated range."));
        result[name] = Number(input.value);
      }
      return result;
    };
    const set = (id, name, value) => { panels[id].querySelector(`[name="${name}"]`).value = value; };
    const get = (v, prefix, key) => v[`${prefix}-${key}`];
    function calculate(id, v) {
      switch (id) {
        case "frames": return M.frameTransform({x: get(v, "frame", "x"), y: get(v, "frame", "y"), theta: get(v, "frame", "theta"), tx: get(v, "frame", "tx"), ty: get(v, "frame", "ty")});
        case "kinematics": {
          const pose = M.armState({q1: get(v, "arm", "q1"), q2: get(v, "arm", "q2")});
          const targetX = v["arm-target-x"], targetY = v["arm-target-y"];
          const target = M.inverseArm({x: targetX, y: targetY});
          // Reachability and residual are part of every result, including exports
          // and redraws. They are not a transient message attached to the IK click.
          return {...pose, targetReachable: target.reachable, targetError: Math.hypot(pose.tip.x - targetX, pose.tip.y - targetY)};
        }
        case "control": return M.simulatePD({kp: get(v, "control", "kp"), kd: get(v, "control", "kd"), delayMs: get(v, "control", "delay")});
        case "timing":
          if (v["timing-execute"] > v["timing-horizon"]) throw new RangeError(t("执行步数 K 不能超过预测长度 H。请减少 K 或增大 H。", "Executed steps K cannot exceed predicted horizon H. Reduce K or increase H."));
          return M.actionSchedule({horizon: v["timing-horizon"], execute: v["timing-execute"], latencyMs: v["timing-latency"]});
        case "evaluation":
          if (v["eval-successes"] > v["eval-trials"]) throw new RangeError(t("成功回合不能超过总回合。", "Successful trials cannot exceed total trials."));
          return M.wilsonInterval({successes: v["eval-successes"], trials: v["eval-trials"]});
      }
    }
    function summary(id, r) {
      if (id === "frames") return `${t("世界点 pW", "World point pW")} = (${f(r.world.x)}, ${f(r.world.y)}) m · ${t("逆变换回 pS", "Inverse back to pS")} = (${f(r.inverse.x)}, ${f(r.inverse.y)}) m`;
      if (id === "kinematics") return `${t("末端", "Tip")} = (${f(r.tip.x)}, ${f(r.tip.y)}) m · ${t("目标误差", "Target error")} ${f(r.targetError * 1000, 3)} mm · ${r.targetReachable ? t("目标在几何工作空间内", "Target inside geometric workspace") : t("目标不可达：半径须在 0.3–1.7 m 内；IK 保持当前姿态", "Target unreachable: radius must be 0.3–1.7 m; IK leaves the current pose unchanged")} · det J = ${f(r.determinant, 3)} m²/rad² · ${Math.abs(r.determinant) < .02 ? t("接近奇异构型：局部运动退化", "Near a singular configuration: local motion degenerates") : t("当前 Jacobian 满秩", "Current Jacobian is full rank")}`;
      if (id === "control") return `${t("超调", "Overshoot")} ${f(r.overshootPct, 1)}% · RMSE ${f(r.rmse, 3)} m · ${t("观测稳定时间", "Observed settling")} ${r.settlingTime === null ? t("未达到", "not reached") : `${f(r.settlingTime)} s`} · ${t("力饱和占比", "Force saturation")} ${f(r.saturationPct, 1)}%`;
      if (id === "timing") return `${t("重观测周期", "Re-observation cycle")} ${f(r.cycleMs, 0)} ms · ${t("动作占空比", "Action duty")} ${f(r.dutyPct, 1)}% · ${t("末步起始观测年龄", "Observation age at last action start")} ${f(r.worstAgeMs, 0)} ms`;
      return `${t("成功率", "Success rate")} ${f(r.rate * 100, 1)}% · ${t("约 95% Wilson 区间", "Approx. 95% Wilson interval")} [${f(r.lower * 100, 1)}%, ${f(r.upper * 100, 1)}%] · ${t("宽度", "Width")} ${f((r.upper - r.lower) * 100, 1)} ${t("个百分点", "percentage points")}`;
    }
    const text = (x, y, label, extra = "") => `<text x="${x}" y="${y}" ${extra}>${escape(label)}</text>`;
    const line = (x1, y1, x2, y2, cls = "lab-axis") => `<line class="${cls}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
    const circle = (x, y, radius, cls) => `<circle class="${cls}" cx="${x}" cy="${y}" r="${radius}"/>`;
    function axes(width, height, xmin, xmax, ymin, ymax, xlabel, ylabel, square = false) {
      const left = 48, right = width - 16, top = 32;
      const bottom = square ? top + right - left : height - 45;
      const x = (v) => left + (v - xmin) / (xmax - xmin) * (right - left);
      const y = (v) => bottom - (v - ymin) / (ymax - ymin) * (bottom - top);
      let markup = "";
      for (let i = 0; i <= 4; i++) {
        const xv = xmin + (xmax - xmin) * i / 4, yv = ymin + (ymax - ymin) * i / 4;
        markup += line(x(xv), top, x(xv), bottom, "lab-grid") + line(left, y(yv), right, y(yv), "lab-grid");
        markup += text(x(xv), bottom + 20, f(xv, Math.abs(xmax - xmin) > 10 ? 0 : 1), 'text-anchor="middle"');
        markup += text(left - 8, y(yv) + 4, f(yv, Math.abs(ymax - ymin) > 10 ? 0 : 1), 'text-anchor="end"');
      }
      markup += text(left, 16, ylabel) + text((left + right) / 2, bottom + 41, xlabel, 'text-anchor="middle"');
      return {x, y, markup, height: bottom + 48};
    }
    function renderChart(id, v, r) {
      const svg = panels[id].querySelector("svg");
      const width = Math.max(230, Math.round(svg.parentElement.clientWidth));
      let height = 300, body = "";
      if (id === "frames" || id === "kinematics") {
        const extent = id === "frames" ? 3.2 : 2;
        const a = axes(width, height, -extent, extent, -extent, extent, "x (m)", "y (m)", true);
        const {x, y} = a; height = a.height;
        if (id === "kinematics") body += circle(x(0), y(0), x(1.7) - x(0), "lab-workspace") + circle(x(0), y(0), x(.3) - x(0), "lab-joint");
        body += a.markup + line(x(-extent), y(0), x(extent), y(0)) + line(x(0), y(-extent), x(0), y(extent));
        if (id === "frames") {
          const tx = v["frame-tx"], ty = v["frame-ty"], theta = v["frame-theta"] * Math.PI / 180;
          const endx = {x: tx + .65 * Math.cos(theta), y: ty + .65 * Math.sin(theta)};
          const endy = {x: tx - .65 * Math.sin(theta), y: ty + .65 * Math.cos(theta)};
          body += line(x(tx), y(ty), x(endx.x), y(endx.y), "lab-a") + line(x(tx), y(ty), x(endy.x), y(endy.y), "lab-a");
          body += text(x(endx.x), y(endx.y) - 8, "xS", 'text-anchor="middle"') + text(x(endy.x), y(endy.y) - 8, "yS", 'text-anchor="middle"');
          body += line(x(tx), y(ty), x(r.world.x), y(r.world.y), "lab-b lab-dashed");
          body += circle(x(tx), y(ty), 4, "lab-target") + text(x(tx) - 8, y(ty) + 18, "S", 'text-anchor="end"');
          const labelRight = x(r.world.x) < width - 45;
          body += circle(x(r.world.x), y(r.world.y), 6, "lab-point") + text(x(r.world.x) + (labelRight ? 9 : -9), y(r.world.y) - 10, "pW", `text-anchor="${labelRight ? "start" : "end"}"`);
          body += text(x(0) - 7, y(0) + 17, "W", 'text-anchor="end"');
        } else {
          const tx = x(v["arm-target-x"]), ty = y(v["arm-target-y"]);
          body += line(x(0), y(0), x(r.elbow.x), y(r.elbow.y), "lab-a lab-link") + line(x(r.elbow.x), y(r.elbow.y), x(r.tip.x), y(r.tip.y), "lab-b lab-link");
          body += circle(x(0), y(0), 7, "lab-joint") + circle(x(r.elbow.x), y(r.elbow.y), 7, "lab-joint") + circle(x(r.tip.x), y(r.tip.y), 6, "lab-point");
          body += line(tx - 7, ty, tx + 7, ty, "lab-target") + line(tx, ty - 7, tx, ty + 7, "lab-target");
        }
      } else if (id === "control") {
        const positions = r.samples.map((s) => s.x);
        const lo = Math.min(0, ...positions), hi = Math.max(1, ...positions), pad = (hi - lo) * .12;
        const a = axes(width, height, 0, 6, lo - pad, hi + pad, t("时间 (s)", "Time (s)"), t("位置 (m)", "Position (m)"));
        body = a.markup + line(a.x(0), a.y(1), a.x(6), a.y(1), "lab-b lab-dashed");
        const points = r.samples.map((s, i) => `${i === 0 ? "M" : "L"}${f(a.x(s.t))},${f(a.y(s.x))}`).join(" ");
        body += `<path class="lab-a" d="${points}"/>`;
      } else if (id === "timing") {
        const left = 20, right = width - 20, scale = (ms) => left + ms / r.cycleMs * (right - left);
        body = text(left, 22, width < 360 ? t("同步周期", "Synchronous cycle") : t("一次同步周期", "One synchronous cycle"));
        if (r.idleMs > 0) body += `<rect class="lab-fill-b" x="${left}" y="45" width="${scale(r.idleMs) - left}" height="35" rx="3"/>`;
        for (let i = 0; i < v["timing-execute"]; i++) body += `<rect class="lab-fill-a" x="${scale(r.idleMs + i * 50) + .5}" y="45" width="${Math.max(.5, (right - left) * 50 / r.cycleMs - 1)}" height="35"/>`;
        for (let i = 0; i <= 2; i++) body += text(scale(r.cycleMs * i / 2), 101, `${f(r.cycleMs * i / 2, 0)}`, `text-anchor="${i === 0 ? "start" : i === 2 ? "end" : "middle"}"`);
        body += text(right, 123, t("时间 (ms)", "Time (ms)"), 'text-anchor="end"');
        body += text(left, 173, width < 360 ? t("预测序列（1…H）", "Predicted slots (1…H)") : t("预测序列：亮色 = 执行，灰色 = 丢弃", "Prediction: colored = used; gray = discarded"));
        const count = v["timing-horizon"], cell = (right - left) / count;
        for (let i = 0; i < count; i++) body += `<rect class="${i < v["timing-execute"] ? "lab-fill-a" : "lab-fill-muted"}" x="${left + cell * i + .5}" y="193" width="${cell - 1}" height="28"/>`;
        body += text(left, 244, "1") + text(right, 244, `H = ${count}`, 'text-anchor="end"');
        body += text(left, 274, width < 360 ? t("灰色丢弃；下排不是时间轴", "Gray discarded; not a time axis") : t("下排按动作索引排列，不是时间轴", "Lower row is action index, not a time axis"));
      } else {
        const left = 25, right = width - 25, x = (value) => left + value * (right - left);
        body = text(left, 25, t("约 95% Wilson 区间", "Approx. 95% Wilson interval"));
        body += line(x(.8), 45, x(.8), 181, "lab-b lab-dashed");
        body += line(x(r.lower), 109, x(r.upper), 109, "lab-a") + line(x(r.lower), 99, x(r.lower), 119, "lab-a") + line(x(r.upper), 99, x(r.upper), 119, "lab-a");
        body += circle(x(r.rate), 109, 6, "lab-point");
        body += text(Math.min(right - 18, Math.max(left + 18, x(r.rate))), 84, `${f(r.rate * 100, 1)}%`, 'text-anchor="middle"');
        body += line(left, 181, right, 181);
        for (let i = 0; i <= 4; i++) body += text(x(i / 4), 202, `${i * 25}%`, `text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}"`);
        body += text((left + right) / 2, 228, t("成功概率", "Success probability"), 'text-anchor="middle"');
        body += text(left, 267, `n = ${v["eval-trials"]} · k = ${v["eval-successes"]}`);
      }
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.innerHTML = `<title id="chart-title-${id}">${escape(definitions[id].title)}</title><desc id="chart-desc-${id}">${escape(summary(id, r))}</desc>${body}`;
    }
    function update(id) {
      const panel = panels[id], output = panel.querySelector(".eai-result");
      for (const [name, , , , , , unit] of definitions[id].fields) {
        const input = panel.querySelector(`[name="${name}"]`);
        panel.querySelector(`[data-value-for="${name}"]`).textContent = `${input.value || "—"} ${unit}`;
      }
      try {
        const v = values(id), r = calculate(id, v);
        output.textContent = summary(id, r); output.dataset.error = String(id === "kinematics" && !r.targetReachable);
        renderChart(id, v, r);
        return {parameters: v, result: r};
      } catch (error) {
        output.dataset.error = "true";
        output.textContent = error instanceof RangeError ? error.message : t("实验暂不可用，请阅读下方推导。", "Experiment unavailable; use the worked example below.");
        // Never show an old plot beside a validation error as if it represented new inputs.
        panel.querySelector("svg").innerHTML = "";
        return null;
      }
    }
    function select(id, focus = false) {
      if (!ids.includes(id)) return;
      selected = id;
      ids.forEach((key) => {
        const tab = root.querySelector(`[data-lab-tab="${key}"]`);
        tab.setAttribute("aria-selected", String(key === id));
        tab.tabIndex = key === id ? 0 : -1;
        panels[key].hidden = key !== id;
      });
      update(id);
      if (focus) root.querySelector(`[data-lab-tab="${id}"]`).focus();
    }
    const selectHash = () => {
      // Only known fixed IDs are handled. Other anchors continue to reach static lessons.
      const hash = window.location.hash.slice(1);
      if (ids.includes(hash)) select(hash);
    };
    root.addEventListener("input", (event) => { if (event.target.matches("input")) update(selected); });
    root.addEventListener("keydown", (event) => {
      if (!event.target.matches("[data-lab-tab]")) return;
      const i = ids.indexOf(selected);
      const next = ({ArrowRight: (i + 1) % 5, ArrowLeft: (i + 4) % 5, Home: 0, End: 4})[event.key];
      if (next !== undefined) {
        event.preventDefault(); select(ids[next], true);
        history.replaceState(null, "", `#${ids[next]}`);
      }
    });
    root.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button || !root.contains(button)) return;
      if (button.dataset.labTab) {
        select(button.dataset.labTab); history.replaceState(null, "", `#${selected}`); return;
      }
      const panel = panels[selected];
      if (button.dataset.preset) {
        const preset = button.dataset.preset;
        if (preset === "oscillate") { set("control", "control-kp", 16); set("control", "control-kd", .4); set("control", "control-delay", 0); }
        else { const n = {small: 10, medium: 100, large: 1000}[preset]; set("evaluation", "eval-trials", n); set("evaluation", "eval-successes", n * .8); }
        update(selected);
      }
      if (button.dataset.action === "reset") {
        for (const [name, , , , value] of definitions[selected].fields) set(selected, name, value);
        update(selected);
      }
      if (button.dataset.action === "inverse") {
        const state = update(selected);
        if (!state) return;
        const v = state.parameters, solution = M.inverseArm({x: v["arm-target-x"], y: v["arm-target-y"]});
        if (!solution.reachable) return; // update() already reports persistent reachability.
        const wrap = (angle) => ((angle + 180) % 360 + 360) % 360 - 180;
        set(selected, "arm-q1", f(wrap(solution.q1), 1));
        set(selected, "arm-q2", f(solution.q2, 1));
        update(selected);
      }
      if (button.dataset.action === "export") {
        const state = update(selected);
        if (!state) return;
        const record = {
          schema: "embodied-ai-learning-lab/v1", modelVersion: 1,
          evidenceLevel: "interactive-teaching-model", assessed: false,
          lab: selected, language: zh ? "zh" : "en", createdAt: new Date().toISOString(),
          ...state, assumptions: definitions[selected].note,
          prediction: panel.querySelector('[name="prediction"]').value,
          reflection: panel.querySelector('[name="reflection"]').value
        };
        const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], {type: "application/json"}));
        const a = document.createElement("a"); a.href = url; a.download = `learning-lab-${selected}.json`;
        document.body.append(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        panel.querySelector(".eai-export-status").textContent = t("已生成下载：参数、结果和你的解释。记录不会上传，也不会自动计入课程通过。", "Download generated: parameters, results and your explanation. Nothing is uploaded or automatically counted as curriculum completion.");
      }
    });
    select(ids.includes(location.hash.slice(1)) ? location.hash.slice(1) : "frames");
    let observedWidth = -1;
    const resize = new ResizeObserver(([entry]) => {
      // A status message or opened notebook changes height, not chart geometry.
      // Ignore that change to avoid repeatedly announcing or overwriting results.
      const width = entry.contentRect.width;
      if (root.isConnected && Math.abs(width - observedWidth) > .5) {
        observedWidth = width;
        update(selected);
      }
    });
    resize.observe(root);
    return {root, selectHash, dispose: () => resize.disconnect()};
  }
  function init() {
    if (current && !current.root.isConnected) { current.dispose(); current = null; }
    const root = document.querySelector(".eai-labs");
    if (root && !root.dataset.initialized) current = mount(root);
  }
  window.addEventListener("hashchange", () => { if (current) current.selectHash(); });
  if (typeof document$ !== "undefined") document$.subscribe(init);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})();
