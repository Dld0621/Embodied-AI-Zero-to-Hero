const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { branches, layout, svgFor, fitView, zoomAt, textLines } = require('../../docs/javascripts/vla-framework.js');

const modes = ['overview', 'full', 'relations', ...branches.map(branch => branch.id)];

test('overview is one connected ten-axis tree with five branches on each side', () => {
  const graph = layout();
  const axes = graph.nodes.filter(node => node.kind === 'branch');
  assert.equal(axes.length, 10);
  assert.equal(axes.filter(node => node.side === 'left').length, 5);
  assert.equal(axes.filter(node => node.side === 'right').length, 5);
  assert.equal(graph.nodes.filter(node => node.kind === 'group').length, 44);
  assert.ok(graph.nodes.filter(node => node.kind === 'group').every(node => node.sub.length > 3));
  const reached = new Set(['root']);
  for (let pass = 0; pass < 3; pass++) for (const edge of graph.edges) if (reached.has(edge.from)) reached.add(edge.to);
  assert.equal(reached.size, graph.nodes.length);
  assert.equal(graph.edges.length, graph.nodes.length - 1);
});

test('all views have unique nodes, valid edges, and bounded finite positions', () => {
  for (const mode of modes) {
    const graph = layout(mode);
    const ids = new Set(graph.nodes.map(node => node.id));
    assert.equal(ids.size, graph.nodes.length, mode);
    for (const node of graph.nodes) {
      assert.ok(Number.isFinite(node.x) && Number.isFinite(node.y), node.id);
      assert.ok(node.x >= 0 && node.x <= graph.width && node.y >= 0 && node.y <= graph.height, node.id);
    }
    for (const edge of graph.edges) assert.ok(ids.has(edge.from) && ids.has(edge.to) && edge.from !== edge.to, mode);
  }
  assert.throws(() => layout('not-a-branch'), RangeError);
});

test('full export retains 128 concrete term nodes and local composition semantics', () => {
  const full = layout('full');
  assert.equal(full.nodes.filter(node => node.kind === 'leaf').length, 128);
  assert.equal(full.nodes.length, 183);
  const pali = full.nodes.find(node => node.id === 'g-components-pali');
  assert.equal(pali.sub, '内含');
  assert.deepEqual(full.edges.filter(edge => edge.from === pali.id).map(edge => edge.relation), ['内含', '内含']);
  assert.equal(full.nodes.find(node => node.id === 'g-internals-block').sub, '典型组成');
  for (const branch of branches) {
    const focus = layout(branch.id);
    const focusedTerms = focus.nodes.filter(node => node.kind === 'leaf').map(node => node.label);
    const exportedTerms = full.nodes.filter(node => node.kind === 'leaf' && node.branch === branch.id).map(node => node.label);
    assert.deepEqual(focusedTerms, exportedTerms);
  }
  assert.ok(full.nodes.some(node => /教师预测/.test(node.label)));
  assert.ok(full.nodes.some(node => /电机伺服频率/.test(node.label)));
});

test('SigLIP associations name roles rather than drawing a false inheritance edge', () => {
  const graph = layout('relations');
  assert.deepEqual(graph.edges.map(edge => edge.relation), ['担任', '采用', '用作初始化']);
  assert.ok(graph.edges.every(edge => edge.dashed));
  assert.deepEqual(graph.nodes.slice(1).map(node => node.branch), ['modules', 'architecture', 'training']);
  assert.ok(!graph.nodes.some(node => node.label === 'VLA'));
});

test('compact branches preserve the knowledge tree in a narrow vertical reading layout', () => {
  for (const branch of branches) {
    const compact = layout(branch.id, { compact: true });
    const wide = layout(branch.id);
    assert.equal(compact.width, 560);
    assert.equal(compact.compact, true);
    assert.equal(compact.nodes.length, wide.nodes.length);
    assert.equal(compact.edges.length, compact.nodes.length - 1);
    assert.deepEqual(compact.nodes.filter(node => node.kind === 'leaf').map(node => node.label), wide.nodes.filter(node => node.kind === 'leaf').map(node => node.label));
    assert.ok(compact.nodes.every(node => node.x >= 0 && node.x <= compact.width && node.y < compact.height));
    const terms = compact.nodes.filter(node => node.kind === 'leaf');
    for (let i = 1; i < terms.length; i++) assert.ok(terms[i].y - terms[i - 1].y >= 43);
  }
  assert.match(svgFor({ mode: 'internals', compact: true }), /data-vf-compact="true"/);
});

test('fit contains full bounds and preserves viewport aspect', () => {
  for (const aspect of [.5, 1, 1.8, 3]) {
    const view = fitView(2000, 1320, aspect, 30);
    assert.ok(Math.abs(view.w / view.h - aspect) < 1e-12);
    assert.ok(view.x <= -30 && view.y <= -30);
    assert.ok(view.x + view.w >= 2030 && view.y + view.h >= 1350);
  }
  for (const args of [[0, 3], [1, -1], [2, 3, 0], [2, 3, 1, -1], [NaN, 3]]) assert.throws(() => fitView(...args), RangeError);
});

test('zoom leaves pointer anchor invariant and is reversible', () => {
  const original = { x: -15, y: 7, w: 1200, h: 800 };
  for (const [px, py] of [[0, 0], [1, 1], [.22, .76]]) {
    const zoomed = zoomAt(original, 1.7, px, py);
    assert.ok(Math.abs(original.x + px * original.w - zoomed.x - px * zoomed.w) < 1e-10);
    assert.ok(Math.abs(original.y + py * original.h - zoomed.y - py * zoomed.h) < 1e-10);
    const restored = zoomAt(zoomed, 1 / 1.7, px, py);
    for (const key of ['x', 'y', 'w', 'h']) assert.ok(Math.abs(restored[key] - original[key]) < 1e-10);
  }
  assert.throws(() => zoomAt(original, 0), RangeError);
  assert.throws(() => zoomAt(original, 1, 1.1), RangeError);
});

test('SVG is standalone, accessible, local-only, and preserves concrete leaf nodes', () => {
  const svg = svgFor({ mode: 'full' });
  assert.match(svg, /^<svg xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  assert.match(svg, /aria-labelledby="vf-title vf-desc"/);
  assert.match(svg, /<desc id="vf-desc">/);
  assert.match(svg, /data-vf-mode="full"/);
  assert.equal((svg.match(/data-vf-kind="leaf"/g) || []).length, 128);
  assert.match(svg, /SigLIP 图像编码器/);
  assert.match(svg, /LoRA/);
  assert.match(svg, /Full \/ Freeze \/ Head-only/);
  assert.doesNotMatch(svg, /<script|<foreignObject|<image|href=|url\(/i);
  assert.ok(!svg.includes('undefined'));
  assert.deepEqual(textLines('Attention 与视觉表征', 7).join(''), 'Attention 与视觉表征');
});

test('committed fallback diagrams match the same map used by the interactive view', () => {
  const assets = path.join(__dirname, '../../docs/assets');
  for (const [file, mode] of [['vla-knowledge-framework.svg', 'overview'], ['vla-knowledge-framework-full.svg', 'full']]) {
    assert.equal(fs.readFileSync(path.join(assets, file), 'utf8').trimEnd(), svgFor({ mode }));
  }
});
