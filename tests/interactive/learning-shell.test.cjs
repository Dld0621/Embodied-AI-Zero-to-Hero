"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {matches, preferences, bookmark} = require("../../docs/javascripts/learning-shell.js");

test("catalog search handles Chinese, full-width Latin and AND terms", () => {
  assert.equal(matches("  ＰＤ   控制 ", "反馈控制 PD damping"), true);
  assert.equal(matches("pd 坐标", "反馈控制 PD damping"), false);
  assert.equal(matches("", "任意章节"), true);
  assert.equal(matches("关节", undefined), false);
});

test("reading settings accept only explicit supported choices", () => {
  assert.deepEqual(preferences(null), {font: "standard", focus: false});
  assert.deepEqual(preferences({font: "larger", focus: true}), {font: "larger", focus: true});
  assert.deepEqual(preferences({font: "giant", focus: "true"}), {font: "standard", focus: false});
});

test("bookmarks reject missing fields, path injection and malformed stored data", () => {
  const valid = {id: "numpy-axis-semantics", chapter: "computing-python-numpy", title: "轴语义"};
  assert.deepEqual(bookmark(valid), valid);
  for (const value of [null, {}, {...valid, id: "../escape"}, {...valid, chapter: "https://evil.test"},
    {...valid, title: ""}, {...valid, title: "x".repeat(201)}, {...valid, id: ["lesson"]},
    {...valid, id: "x".repeat(101)}, {...valid, chapter: "x?redirect=1"}]) {
    assert.equal(bookmark(value), null);
  }
});
