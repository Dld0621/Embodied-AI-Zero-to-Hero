"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { formatNumber, armState } = require("../../docs/javascripts/lab-models.js");

test("formatNumber normalizes rounded negative zero at every UI precision", () => {
  for (const digits of [0, 1, 2, 3]) {
    const expected = (0).toFixed(digits);
    for (const value of [-0, -Number.EPSILON, -0.00001]) {
      assert.equal(formatNumber(value, digits), expected);
    }
  }
  assert.equal(formatNumber(-Number.EPSILON), "0.00");
});

test("formatNumber retains meaningful negative values and requested precision", () => {
  for (const digits of [1, 2, 3]) {
    assert.equal(formatNumber(-1.234, digits), (-1.234).toFixed(digits));
    assert.equal(formatNumber(1.234, digits), (1.234).toFixed(digits));
    assert.equal(formatNumber(0, digits), (0).toFixed(digits));
  }
  assert.equal(formatNumber(-0.1, 1), "-0.1");
  assert.equal(formatNumber(-0.01, 2), "-0.01");
  assert.equal(formatNumber(-0.001, 3), "-0.001");
});

test("formatNumber preserves existing coercion and non-finite formatting", () => {
  assert.equal(formatNumber("-1.25"), "-1.25");
  for (const digits of [1, 2, 3]) {
    assert.equal(formatNumber(NaN, digits), "NaN");
    assert.equal(formatNumber(Infinity, digits), "Infinity");
    assert.equal(formatNumber(-Infinity, digits), "-Infinity");
  }
  assert.throws(() => formatNumber(1, -1), RangeError);
  assert.throws(() => formatNumber(1, 101), RangeError);
});

test("a folded arm displays zero determinant without erasing a negative determinant", () => {
  const folded = armState({ q1: 0, q2: -180 });
  assert.ok(folded.determinant < 0 && folded.determinant > -1e-12);
  assert.equal(formatNumber(folded.determinant, 3), "0.000");
  assert.equal(formatNumber(armState({ q1: 0, q2: -90 }).determinant, 3), "-0.700");
});
