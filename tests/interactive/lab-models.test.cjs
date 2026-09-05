"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const models = require("../../docs/javascripts/lab-models.js");
const { frameTransform, armState, inverseArm, simulatePD, actionSchedule, wilsonInterval } = models;

function near(actual, expected, tolerance = 1e-10) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
}

test("models export in both a browser global and CommonJS without dependencies", () => {
  const context = vm.createContext({});
  const source = fs.readFileSync(path.join(__dirname, "../../docs/javascripts/lab-models.js"), "utf8");
  vm.runInContext(source, context);
  assert.deepEqual(Object.keys(context.EmbodiedLabModels), Object.keys(models));
  assert.ok(Object.isFrozen(models));
});

test("frame transform rotates before translating and recovers the local point", () => {
  const result = frameTransform({ x: 2, y: 1, theta: 90, tx: 3, ty: -2 });
  near(result.rotated.x, -1); near(result.rotated.y, 2);
  near(result.world.x, 2); near(result.world.y, 0);
  near(result.inverse.x, 2); near(result.inverse.y, 1);
});

test("frame rotation convention, periodicity, and inverse hold across quadrants", () => {
  for (const theta of [-450, -180, -35, 0, 35, 180, 450]) {
    const p = { x: -1.2, y: 3.4, theta, tx: 2.1, ty: -0.3 };
    const r = frameTransform(p), periodic = frameTransform({ ...p, theta: theta + 360 });
    near(r.inverse.x, p.x); near(r.inverse.y, p.y);
    near(Math.hypot(r.rotated.x, r.rotated.y), Math.hypot(p.x, p.y));
    near(r.world.x, periodic.world.x); near(r.world.y, periodic.world.y);
  }
});

test("arm forward kinematics and Jacobian have hand-computed values", () => {
  const r = armState({ q1: 0, q2: 90, l1: 1, l2: 0.7 });
  near(r.elbow.x, 1); near(r.elbow.y, 0);
  near(r.tip.x, 1); near(r.tip.y, 0.7);
  near(r.determinant, 0.7); near(r.manipulability, 0.7);
  near(r.jacobian[0][0], -0.7); near(r.jacobian[0][1], -0.7);
  near(r.jacobian[1][0], 1); near(r.jacobian[1][1], 0);
});

test("arm singularity, determinant sign, and Jacobian derivative use radians", () => {
  for (const q2 of [0, 180, -180, 360]) near(armState({ q1: 24, q2 }).manipulability, 0);
  near(armState({ q1: 45, q2: -90 }).determinant, -0.7);
  const state = armState({ q1: 31, q2: -47 });
  const epsilon = 1e-6;
  for (let joint = 0; joint < 2; joint += 1) {
    const values = { q1: 31, q2: -47 };
    values[joint === 0 ? "q1" : "q2"] += epsilon * 180 / Math.PI;
    const shifted = armState(values);
    near((shifted.tip.x - state.tip.x) / epsilon, state.jacobian[0][joint], 2e-6);
    near((shifted.tip.y - state.tip.y) / epsilon, state.jacobian[1][joint], 2e-6);
  }
});

test("inverse kinematics returns both valid branches and round-trips positions", () => {
  for (const point of [{ x: 1, y: 0.7 }, { x: -1, y: 0.5 }, { x: 0.2, y: -1.3 }]) {
    for (const elbow of [-1, 1]) {
      const solution = inverseArm({ ...point, elbow });
      assert.equal(solution.reachable, true);
      assert.equal(Math.sign(solution.q2), elbow);
      const pose = armState(solution);
      near(pose.tip.x, point.x); near(pose.tip.y, point.y);
    }
  }
});

test("inverse kinematics handles workspace boundaries and rejects both unreachable regions", () => {
  for (const x of [1.7, 0.3, -1.7, -0.3]) assert.equal(inverseArm({ x, y: 0 }).reachable, true);
  for (const x of [1.70001, 0.29999, 0]) {
    assert.deepEqual(inverseArm({ x, y: 0 }), { reachable: false, q1: null, q2: null });
  }
  for (const elbow of [-1, 1]) {
    const folded = inverseArm({ x: 0, y: 0, l1: 1, l2: 1, elbow });
    assert.equal(folded.reachable, true);
    const pose = armState({ ...folded, l1: 1, l2: 1 });
    near(pose.tip.x, 0); near(pose.tip.y, 0);
  }
});

test("PD zero target stays exactly at equilibrium", () => {
  const result = simulatePD({ kp: 20, kd: 8, delayMs: 100, target: 0 });
  assert.equal(result.samples.length, 1201);
  assert.equal(result.samples[0].t, 0);
  assert.equal(result.samples.at(-1).t, 6);
  assert.ok(result.samples.every(({ x, v, u }) => x === 0 && v === 0 && u === 0));
  assert.equal(result.overshootPct, 0); assert.equal(result.rmse, 0);
  assert.equal(result.settlingTime, 0); assert.equal(result.saturationPct, 0);
});

test("PD update is semi-implicit Euler with saturated actuator and delayed state", () => {
  const result = simulatePD({ kp: 2, kd: 3, delayMs: 1000, dt: 0.05, duration: 0.15, damping: 0, limit: 1 });
  near(result.samples[1].u, 1);
  near(result.samples[1].v, 0.05); near(result.samples[1].x, 0.0025);
  near(result.samples[2].v, 0.1); near(result.samples[2].x, 0.0075);
  near(result.samples.at(-1).v, 0.15); near(result.samples.at(-1).x, 0.015);
  near(result.saturationPct, 100); assert.equal(result.settlingTime, null);
});

test("fractional delay interpolates both measured position and velocity", () => {
  const result = simulatePD({ kp: 1, kd: 2, delayMs: 25, dt: 0.05, duration: 0.1, damping: 0 });
  // At t=.05, delayed measurement interpolates x=.00125 and v=.025.
  near(result.samples[1].u, 1 - 0.00125 - 2 * 0.025);
});

test("PD includes the exact final time for a non-integral duration/dt", () => {
  const result = simulatePD({ kp: 10, kd: 3, delayMs: 0, duration: 0.113, dt: 0.01 });
  assert.equal(result.samples.length, 13);
  near(result.samples.at(-1).t, 0.113);
  assert.ok(result.samples.every(({ t }, i, array) => i === 0 || t > array[i - 1].t));
  const decimal = simulatePD({ kp: 10, kd: 3, delayMs: 0, duration: 0.07, dt: 0.01 });
  assert.equal(decimal.samples.length, 8);
  assert.equal(decimal.samples.at(-1).t, 0.07);
});

test("derivative feedback reduces default-step overshoot; delayed feedback worsens error", () => {
  const unDamped = simulatePD({ kp: 20, kd: 0, delayMs: 0 });
  const damped = simulatePD({ kp: 20, kd: 8, delayMs: 0 });
  const delayed = simulatePD({ kp: 20, kd: 8, delayMs: 250 });
  assert.ok(damped.overshootPct < unDamped.overshootPct);
  assert.ok(damped.rmse < delayed.rmse);
  assert.ok(damped.settlingTime !== null);
  assert.ok(damped.samples.every(({ u }) => Math.abs(u) <= 10));
  assert.ok(damped.saturationPct > 0 && damped.saturationPct < 100);
});

test("PD settling requires a final contiguous half-second position band", () => {
  const result = simulatePD({ kp: 20, kd: 8, delayMs: 0 });
  const index = result.samples.findIndex(({ t }) => t === result.settlingTime);
  assert.ok(index > 0);
  assert.ok(Math.abs(result.samples[index - 1].x - 1) > 0.02);
  assert.ok(result.samples.slice(index).every(({ x }) => Math.abs(x - 1) <= 0.02));
  assert.ok(result.samples.at(-1).t - result.settlingTime >= 0.5);
  assert.equal(simulatePD({ kp: 0, kd: 0, delayMs: 0, target: 0, duration: 0.49 }).settlingTime, null);
  assert.equal(simulatePD({ kp: 0, kd: 0, delayMs: 0 }).settlingTime, null);
});

test("negative PD targets mirror positive targets including overshoot", () => {
  const positive = simulatePD({ kp: 16, kd: 3, delayMs: 50 });
  const negative = simulatePD({ kp: 16, kd: 3, delayMs: 50, target: -1 });
  near(positive.rmse, negative.rmse); near(positive.overshootPct, negative.overshootPct);
  positive.samples.forEach((sample, i) => near(sample.x, -negative.samples[i].x));
});

test("action scheduling distinguishes predicted horizon H from executed prefix K", () => {
  const result = actionSchedule({ horizon: 16, execute: 4, latencyMs: 100, stepMs: 50 });
  assert.equal(result.cycleMs, 300); assert.equal(result.actionMs, 200);
  assert.equal(result.idleMs, 100); near(result.dutyPct, 200 / 3);
  assert.equal(result.worstAgeMs, 250);
  assert.deepEqual(result, actionSchedule({ horizon: 32, execute: 4, latencyMs: 100, stepMs: 50 }));
});

test("schedule zero latency and single-action boundary use action START time", () => {
  const result = actionSchedule({ horizon: 1, execute: 1, latencyMs: 0 });
  assert.equal(result.cycleMs, 50); assert.equal(result.dutyPct, 100);
  assert.equal(result.worstAgeMs, 0);
  const slower = actionSchedule({ horizon: 4, execute: 4, latencyMs: 200 });
  assert.equal(slower.worstAgeMs, 350);
});

test("Wilson interval matches a hand-calculated 5/10 example", () => {
  const result = wilsonInterval({ successes: 5, trials: 10 });
  assert.equal(result.rate, 0.5);
  near(result.lower, 0.23658959361548731); near(result.upper, 0.7634104063845127);
});

test("Wilson endpoints never assert certainty with finite data", () => {
  const none = wilsonInterval({ successes: 0, trials: 10 });
  const all = wilsonInterval({ successes: 10, trials: 10 });
  assert.equal(none.lower, 0); assert.equal(all.upper, 1);
  near(none.upper, 3.8416 / 13.8416);
  near(all.lower, 10 / 13.8416);
  assert.ok(none.upper > 0); assert.ok(all.lower < 1);
});

test("Wilson intervals are symmetric and narrow with larger independent sample sizes", () => {
  const a = wilsonInterval({ successes: 7, trials: 10 });
  const b = wilsonInterval({ successes: 3, trials: 10 });
  const bigger = wilsonInterval({ successes: 70, trials: 100 });
  near(a.lower, 1 - b.upper); near(a.upper, 1 - b.lower);
  assert.ok(bigger.upper - bigger.lower < a.upper - a.lower);
});

test("public inputs reject invalid numbers, domains, and excessive simulation workloads", () => {
  const bad = [NaN, Infinity, -Infinity, "1", undefined];
  for (const value of bad) {
    assert.throws(() => frameTransform({ x: value, y: 0, theta: 0, tx: 0, ty: 0 }), RangeError);
    assert.throws(() => armState({ q1: value, q2: 0 }), RangeError);
    assert.throws(() => simulatePD({ kp: value, kd: 1, delayMs: 0 }), RangeError);
    assert.throws(() => wilsonInterval({ successes: 1, trials: value }), RangeError);
  }
  const invalidCalls = [
    () => frameTransform(null),
    () => armState({ q1: 0, q2: 0, l1: 0 }),
    () => inverseArm({ x: 1, y: 0, elbow: 0 }),
    () => inverseArm({ x: 1, y: 0, l2: -1 }),
    () => simulatePD({ kp: -1, kd: 1, delayMs: 0 }),
    () => simulatePD({ kp: 1, kd: -1, delayMs: 0 }),
    () => simulatePD({ kp: 1, kd: 1, delayMs: -1 }),
    () => simulatePD({ kp: 1, kd: 1, delayMs: 0, limit: 0 }),
    () => simulatePD({ kp: 1, kd: 1, delayMs: 0, mass: 0 }),
    () => simulatePD({ kp: 1, kd: 1, delayMs: 0, dt: 0 }),
    () => simulatePD({ kp: 1, kd: 1, delayMs: 0, duration: 0 }),
    () => simulatePD({ kp: 1, kd: 1, delayMs: 0, dt: 0.0001, duration: 60 }),
    () => simulatePD({ kp: 1, kd: 1, delayMs: 0, mass: 0.01, damping: 100 }),
    () => actionSchedule({ horizon: 2, execute: 3, latencyMs: 100 }),
    () => actionSchedule({ horizon: 2.5, execute: 1, latencyMs: 100 }),
    () => actionSchedule({ horizon: 2, execute: 1.5, latencyMs: 100 }),
    () => actionSchedule({ horizon: 2, execute: 1, latencyMs: -1 }),
    () => actionSchedule({ horizon: 2, execute: 1, latencyMs: 1, stepMs: 0 }),
    () => wilsonInterval({ successes: 0, trials: 0 }),
    () => wilsonInterval({ successes: -1, trials: 10 }),
    () => wilsonInterval({ successes: 11, trials: 10 }),
    () => wilsonInterval({ successes: 1.5, trials: 10 }),
    () => wilsonInterval({ successes: 1, trials: 10, z: 0 }),
  ];
  for (const call of invalidCalls) assert.throws(call, RangeError);
});
