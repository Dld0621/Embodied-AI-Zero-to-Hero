/* Three-token, single-head numerical teaching model. No network or pretrained checkpoint. */
(function (scope) {
  "use strict";
  function attention(input, causal = false) {
    if (!Array.isArray(input) || input.length !== 3 || Array.from(input).some(row =>
      !Array.isArray(row) || row.length !== 2 || Array.from(row).some(x => !Number.isFinite(x) || Math.abs(x) > 1000))) {
      throw new RangeError("Expected a finite 3 × 2 teaching input, with |value| ≤ 1000.");
    }
    if (typeof causal !== "boolean") throw new TypeError("causal must be boolean");
    // Fixed W_Q=W_K=I_2 and W_V=diag(1,2), without biases or position encoding.
    const q = input.map(row => row.slice()), k = input.map(row => row.slice());
    const v = input.map(([a, b]) => [a, 2 * b]);
    const scores = q.map(row => k.map(key => (row[0] * key[0] + row[1] * key[1]) / Math.sqrt(2)));
    const allowed = scores.map((row, i) => row.map((_, j) => !causal || j <= i));
    const weights = scores.map((row, i) => {
      const largest = Math.max(...row.filter((_, j) => allowed[i][j]));
      const exp = row.map((s, j) => allowed[i][j] ? Math.exp(s - largest) : 0);
      const total = exp.reduce((a, b) => a + b, 0);
      return exp.map(x => x / total);
    });
    const output = weights.map(row => [0, 1].map(d => row.reduce((sum, w, j) => sum + w * v[j][d], 0)));
    return {q, k, v, scores, allowed, weights, output};
  }
  const api = {attention};
  if (typeof module === "object" && module.exports) module.exports = api;
  scope.VlaAttention = api;
  if (!scope.document) return;
  const document = scope.document;
  const fmt = n => String(Number(n.toFixed(3)));
  function initialize() {
    document.querySelectorAll("[data-vla-attention]").forEach(root => {
      if (root.dataset.ready) return;
      root.dataset.ready = "true";
      root.innerHTML = `<div class="va-heading"><p>动手算 · 三个位置，一个注意力头</p><h3>输入改变时，哪些数会改变？</h3></div>
        <p>参数固定：W<sub>Q</sub> = W<sub>K</sub> = I，W<sub>V</sub> = diag(1, 2)。A、B、C 是序列位置，不代表三个物体。</p>
        <div class="va-controls"><label>A 的第一个输入特征 <output data-va-value>1</output><input data-va-input aria-label="A 的第一个输入特征" type="range" min="0" max="2" step="0.25" value="1"></label>
        <label class="va-mask"><input type="checkbox" data-va-causal>启用因果掩码：只能读自己及之前的位置</label>
        <div class="va-queries" role="group" aria-label="选择要解释的查询位置"><span>解释哪一行？</span><button type="button" data-va-query="0" aria-pressed="true">Query A</button><button type="button" data-va-query="1" aria-pressed="false">Query B</button><button type="button" data-va-query="2" aria-pressed="false">Query C</button></div></div>
        <div class="va-features" data-va-features></div>
        <div class="va-matrix" data-va-matrix></div>
        <div class="va-explanation" role="status" aria-live="polite" data-va-explanation></div>
        <p class="va-boundary">这里没有位置编码、残差连接、归一化层或完整 Transformer。因果掩码仅演示可见性；不是所有 VLA 动作头都按 A→B→C 自回归。数值是原始精度计算后显示三位小数，显示值相加可能有舍入差异。</p>
        <button type="button" class="va-reset" data-va-reset>恢复默认算例</button>`;
      let selected = 0;
      const input = root.querySelector("[data-va-input]"), mask = root.querySelector("[data-va-causal]");
      const names = ["A", "B", "C"];
      function update() {
        const x = [[Number(input.value), 0], [0, 1], [1, 1]], result = attention(x, mask.checked);
        root.querySelector("[data-va-value]").textContent = fmt(x[0][0]);
        root.querySelectorAll("[data-va-query]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.vaQuery) === selected)));
        const vector = row => `[${row.map(fmt).join(", ")}]`;
        root.querySelector("[data-va-features]").innerHTML = `<table><caption>固定参数下计算的特征</caption><thead><tr><th>位置</th><th>X</th><th>Q = K</th><th>V</th></tr></thead><tbody>${names.map((name, i) => `<tr><th>${name}</th><td>${vector(x[i])}</td><td>${vector(result.q[i])}</td><td>${vector(result.v[i])}</td></tr>`).join("")}</tbody></table>`;
        root.querySelector("[data-va-matrix]").innerHTML = `<table><caption>注意力权重：每行沿 Key 轴归一化</caption><thead><tr><th>Query / Key</th>${names.map(n => `<th>${n}</th>`).join("")}</tr></thead><tbody>${names.map((name, i) => `<tr${i === selected ? ' class="va-selected-row"' : ""}><th>${name}</th>${result.weights[i].map((w, j) => `<td data-va-cell="${i}-${j}">${result.allowed[i][j] ? `<span>${fmt(w)}</span><i aria-hidden="true" style="width:${100 * w}%"></i>` : '<span class="va-blocked">禁读</span>'}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
        const row = result.weights[selected], out = result.output[selected];
        const terms = row.map((w, j) => result.allowed[selected][j] ? `${fmt(w)} × V${names[j]}` : null).filter(Boolean).join(" + ");
        root.querySelector("[data-va-explanation]").innerHTML = `<strong>Query ${names[selected]} 的输出</strong><p>${terms}</p><p data-va-output>${vector(out)}</p><p>W 参数没有改变；X、Q/K/V、打分、注意力权重和输出特征按当前输入重新计算。注意力权重不是物体概率或因果解释。</p>`;
      }
      input.addEventListener("input", update); mask.addEventListener("change", update);
      root.querySelectorAll("[data-va-query]").forEach(button => button.addEventListener("click", () => { selected = Number(button.dataset.vaQuery); update(); }));
      root.querySelector("[data-va-reset]").addEventListener("click", () => { input.value = "1"; mask.checked = false; selected = 0; update(); });
      update();
    });
  }
  if (typeof document$ !== "undefined") document$.subscribe(initialize);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, {once: true});
  else initialize();
})(typeof window !== "undefined" ? window : globalThis);
