'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {attention} = require('../../docs/javascripts/vla-attention.js');
const x = [[1, 0], [0, 1], [1, 1]];
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} differs from ${b}`);

test('single-head example matches a hand-computed normalized weighted sum', () => {
  const original = JSON.stringify(x), r = attention(x);
  const exp = Math.exp(1 / Math.sqrt(2)), denominator = 2 * exp + 1;
  close(r.weights[0][0], exp / denominator);
  close(r.weights[0][1], 1 / denominator);
  close(r.output[0][0], 2 * exp / denominator);
  close(r.output[0][1], 2 * (1 + exp) / denominator);
  assert.equal(JSON.stringify(x), original);
});
test('causal attention blocks future values and normalizes each permitted row', () => {
  const r = attention(x, true);
  assert.deepEqual(r.weights[0], [1, 0, 0]);
  assert.deepEqual(r.output[0], [1, 0]);
  close(r.weights[1][2], 0);
  r.weights.forEach(row => close(row.reduce((a, b) => a + b, 0), 1));
});
test('zero query is uniform over allowed keys and extreme finite scores remain stable', () => {
  assert.deepEqual(attention([[0, 0], [0, 1], [1, 1]]).weights[0], [1/3, 1/3, 1/3]);
  const r = attention([[1000, 1000], [-1000, 1000], [1000, -1000]]);
  r.weights.flat().forEach(v => assert.ok(Number.isFinite(v) && v >= 0 && v <= 1));
  r.weights.forEach(row => close(row.reduce((a, b) => a + b, 0), 1));
});
test('output is permutation-equivariant without positions or causal masking', () => {
  const order = [2, 0, 1], original = attention(x), changed = attention(order.map(i => x[i]));
  changed.output.forEach((row, i) => row.forEach((v, d) => close(v, original.output[order[i]][d])));
});
test('invalid tensor shapes, values and mask flags fail explicitly', () => {
  for (const bad of [[], new Array(3), [[1,0],[0,1],new Array(2)], [[1]], [[1,0],[0,1],[NaN,0]], [[1,0],[0,1],[Infinity,0]], [[1001,0],[0,1],[1,1]]]) assert.throws(() => attention(bad), RangeError);
  assert.throws(() => attention(x, 1), TypeError);
});
