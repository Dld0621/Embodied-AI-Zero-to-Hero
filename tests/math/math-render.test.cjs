'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const { equationKey, renderExpression, renderCache } = require('../../scripts/render_math.cjs');

test('subscripts, fractions, matrices, aligned equations, arrows and bold symbols render', () => {
  for (const tex of [
    'x_1^2', '\\frac{1}{2}', '\\begin{bmatrix}a&b\\\\c&d\\end{bmatrix}',
    '\\begin{aligned}x&=1\\\\y&=2\\end{aligned}',
    'x\\xrightarrow{f}y', '\\boldsymbol{\\theta}',
  ]) {
    const result = renderExpression(tex, true);
    assert.match(result.svg, /<path /);
    assert.match(result.mathml, /<math /);
    assert.doesNotMatch(result.svg, /<use\b|<image\b|href=|url\(/);
  }
});

test('CJK text is preserved with system-font SVG text and semantic MathML', () => {
  const result = renderExpression('\\text{机器人的观测}', false);
  assert.match(result.svg, /<text /);
  assert.match(result.svg, /机/);
  assert.match(result.mathml, /<mtext>/);
  assert.match(result.mathml, /&#x673A;/);
});

test('invalid commands and forbidden remote/HTML extensions fail instead of showing TeX', () => {
  for (const tex of ['\\garbagecmd', '\\frac{1}{', '\\href{https://example.invalid}{x}', '\\require{html}']) {
    assert.throws(() => renderExpression(tex, false));
  }
});

test('display mode changes metrics and is part of each equation hash', () => {
  const tex = '\\frac{1}{2}';
  assert.notEqual(equationKey(tex, false), equationKey(tex, true));
  assert.notEqual(renderExpression(tex, false).svg, renderExpression(tex, true).svg);
});

test('regeneration is deterministic and exactly matches every committed SVG and MathML', () => {
  const cached = JSON.parse(fs.readFileSync(path.join(__dirname, '../../generated/math-cache.json'), 'utf8'));
  const collection = { expressions: Object.fromEntries(Object.entries(cached.expressions)
    .map(([key, value]) => [key, { tex: value.tex, display: value.display }])) };
  assert.deepEqual(renderCache(collection), cached);
});

test('input source keys cannot be silently mismatched', () => {
  assert.throws(() => renderCache({ expressions: { wrong: { tex: 'x', display: false } } }),
    /formula key is incorrect/);
});
