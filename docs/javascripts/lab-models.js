/* Dependency-free teaching models: browser global and Node-testable CommonJS. */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.EmbodiedLabModels = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /** Fixed-precision UI text. Remove a negative sign only when the rounded
   * result is zero; preserve nonzero negatives and Number.toFixed semantics.
   */
  function formatNumber(value, digits = 2) {
    const formatted = Number(value).toFixed(digits);
    return /^-0(?:\.0+)?$/.test(formatted) ? formatted.slice(1) : formatted;
  }

  function options(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new RangeError("Model parameters must be an object.");
    }
    return value;
  }

  function number(name, value, min, max, integer) {
    if (typeof value !== "number" || !Number.isFinite(value) ||
        value < min || value > max || (integer && !Number.isSafeInteger(value))) {
      throw new RangeError(name + " must be " + (integer ? "an integer" : "finite") +
        " in [" + min + ", " + max + "].");
    }
    return value;
  }

  function coordinate(name, value) { return number(name, value, -1e6, 1e6); }
  function length(name, value) { return number(name, value, 1e-6, 1e6); }
  function radians(name, value) {
    number(name, value, -Number.MAX_VALUE, Number.MAX_VALUE);
    return (value % 360) * Math.PI / 180;
  }
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

  /** Local-to-world planar coordinate map using the frame's counterclockwise
   * orientation, then world-frame translation. All positions are metres and
   * theta is degrees. inverse recovers the local coordinates of the same point.
   */
  function frameTransform(params) {
    const { x, y, theta, tx, ty } = options(params);
    coordinate("x", x); coordinate("y", y);
    coordinate("tx", tx); coordinate("ty", ty);
    const angle = radians("theta", theta);
    const c = Math.cos(angle), s = Math.sin(angle);
    const rotated = { x: c * x - s * y, y: s * x + c * y };
    const world = { x: rotated.x + tx, y: rotated.y + ty };
    const dx = world.x - tx, dy = world.y - ty;
    return { world, rotated, inverse: { x: c * dx + s * dy, y: -s * dx + c * dy } };
  }

  /** Two-link planar arm, relative joint angles in degrees, lengths in metres.
   * The returned Jacobian differentiates position with respect to RADIANS.
   * determinant and manipulability have units m²/rad²; the latter is |det(J)|.
   */
  function armState(params) {
    const { q1, q2, l1 = 1, l2 = 0.7 } = options(params);
    length("l1", l1); length("l2", l2);
    const a = radians("q1", q1), b = radians("q2", q2);
    const elbow = { x: l1 * Math.cos(a), y: l1 * Math.sin(a) };
    const tip = { x: elbow.x + l2 * Math.cos(a + b), y: elbow.y + l2 * Math.sin(a + b) };
    const jacobian = [
      [-tip.y, -l2 * Math.sin(a + b)],
      [tip.x, l2 * Math.cos(a + b)],
    ];
    const determinant = l1 * l2 * Math.sin(b);
    return { elbow, tip, determinant, manipulability: Math.abs(determinant), jacobian };
  }

  /** Geometric inverse kinematics, no joint limits or collision constraints.
   * elbow=+1 selects nonnegative q2; elbow=-1 selects nonpositive q2.
   * Unreachable targets return null angles, never invented/clamped positions.
   */
  function inverseArm(params) {
    const { x, y, l1 = 1, l2 = 0.7, elbow = 1 } = options(params);
    coordinate("x", x); coordinate("y", y);
    length("l1", l1); length("l2", l2);
    if (elbow !== 1 && elbow !== -1) throw new RangeError("elbow must be +1 or -1.");
    const distance = Math.hypot(x, y);
    const tolerance = 32 * Number.EPSILON * Math.max(l1, l2);
    if (distance > l1 + l2 + tolerance || distance < Math.abs(l1 - l2) - tolerance) {
      return { reachable: false, q1: null, q2: null };
    }
    if (distance === 0 && l1 === l2) {
      return { reachable: true, q1: 0, q2: elbow * 180 };
    }
    const cosine = clamp((distance * distance - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1);
    const b = elbow * Math.acos(cosine);
    const a = Math.atan2(y, x) - Math.atan2(l2 * Math.sin(b), l1 + l2 * Math.cos(b));
    return { reachable: true, q1: a * 180 / Math.PI, q2: b * 180 / Math.PI };
  }

  /** Saturated PD position controller for m*x'' + damping*x' = u.
   * Zero initial state and zero measurement prehistory; target is a step at t=0.
   * Both measured position AND velocity have delayMs delay. A fractional delay
   * linearly interpolates stored measurements. No prediction or integral term.
   * Semi-implicit Euler: v += a*dt; x += v*dt. u at sample t applies AFTER t.
   * RMSE and saturation percentage are time-weighted over integration intervals.
   * Settling is the start of the final contiguous 2% position band only if the
   * band lasts at least 0.5 s. It is observed over duration, not a stability proof.
   * At target=0, tolerance is 0.0002 m: 2% of max(|target|, 0.01 m).
   */
  function simulatePD(params) {
    const {
      kp, kd, delayMs, limit = 10, duration = 6, dt = 0.005,
      target = 1, mass = 1, damping = 0.2,
    } = options(params);
    number("kp", kp, 0, 1000); number("kd", kd, 0, 200);
    number("delayMs", delayMs, 0, 60000);
    number("limit", limit, 1e-6, 10000);
    number("duration", duration, 0.001, 60);
    number("dt", dt, 0.0001, 0.05);
    number("target", target, -100, 100);
    number("mass", mass, 0.01, 1000);
    number("damping", damping, 0, 100);
    const steps = Math.ceil(duration / dt);
    if (steps > 120000) throw new RangeError("Simulation is limited to 120000 integration steps.");
    if (dt * damping / mass > 1) {
      throw new RangeError("dt is too large for the selected mass and viscous damping.");
    }

    const samples = [];
    let x = 0, v = 0, squaredErrorIntegral = 0, saturatedTime = 0, peak = 0;
    const delay = delayMs / 1000;
    for (let i = 0; i <= steps; i += 1) {
      const t = Math.min(i * dt, duration);
      const measuredAt = t - delay;
      let measuredX = 0, measuredV = 0;
      if (delay === 0) {
        measuredX = x; measuredV = v;
      } else if (measuredAt > 0) {
        const index = Math.min(Math.floor(measuredAt / dt), samples.length - 1);
        const left = samples[index];
        const right = samples[index + 1] || { t, x, v };
        const fraction = right.t === left.t ? 0 : clamp((measuredAt - left.t) / (right.t - left.t), 0, 1);
        measuredX = left.x + fraction * (right.x - left.x);
        measuredV = left.v + fraction * (right.v - left.v);
      }
      const requested = kp * (target - measuredX) - kd * measuredV;
      const u = clamp(requested, -limit, limit);
      samples.push({ t, x, v, u });
      peak = Math.max(peak, target < 0 ? -x : x);
      // A decimal duration/dt can round just above an integer. Stop at the
      // actual final time so a ceil() artefact never produces a duplicate row.
      if (t >= duration) break;
      const h = Math.min(dt, duration - t);
      squaredErrorIntegral += (target - x) ** 2 * h;
      if (Math.abs(requested) > limit) saturatedTime += h;
      v += ((u - damping * v) / mass) * h;
      x += v * h;
    }

    const tolerance = 0.02 * Math.max(Math.abs(target), 0.01);
    let bandStart = samples.length;
    for (let i = samples.length - 1; i >= 0; i -= 1) {
      if (Math.abs(samples[i].x - target) > tolerance) break;
      bandStart = i;
    }
    const firstSettled = bandStart < samples.length ? samples[bandStart].t : null;
    const settlingTime = firstSettled !== null && duration - firstSettled >= 0.5 - 1e-12 ? firstSettled : null;
    return {
      samples,
      overshootPct: target === 0 ? 0 : Math.max(0, (peak - Math.abs(target)) / Math.abs(target) * 100),
      settlingTime,
      rmse: Math.sqrt(squaredErrorIntegral / duration),
      saturationPct: clamp(saturatedTime / duration * 100, 0, 100),
    };
  }

  /** Synchronous policy: observe -> infer -> execute K actions -> observe.
   * horizon H is predicted length, execute K<=H is actually executed length.
   * No overlap, buffering, communication time, or asynchronous replanning.
   * Last-action age is measured at its START from the observation timestamp.
   */
  function actionSchedule(params) {
    const { horizon, execute, latencyMs, stepMs = 50 } = options(params);
    number("horizon", horizon, 1, 1000, true);
    number("execute", execute, 1, horizon, true);
    number("latencyMs", latencyMs, 0, 60000);
    number("stepMs", stepMs, 0.001, 10000);
    const actionMs = execute * stepMs;
    const cycleMs = latencyMs + actionMs;
    return {
      cycleMs, actionMs, idleMs: latencyMs,
      dutyPct: 100 * actionMs / cycleMs,
      worstAgeMs: latencyMs + (execute - 1) * stepMs,
    };
  }

  /** Two-sided Wilson score interval for an independent Bernoulli success rate.
   * z=1.96 approximates 95% confidence; trials must be nonzero. No claim of
   * independence or distribution-shift coverage follows from this computation.
   */
  function wilsonInterval(params) {
    const { successes, trials, z = 1.96 } = options(params);
    number("trials", trials, 1, Number.MAX_SAFE_INTEGER, true);
    number("successes", successes, 0, trials, true);
    number("z", z, 0.000001, 10);
    const rate = successes / trials;
    const zSquared = z * z;
    const denominator = 1 + zSquared / trials;
    const center = (rate + zSquared / (2 * trials)) / denominator;
    const halfWidth = z * Math.sqrt(rate * (1 - rate) / trials + zSquared / (4 * trials * trials)) / denominator;
    return {
      rate,
      lower: successes === 0 ? 0 : clamp(center - halfWidth, 0, 1),
      upper: successes === trials ? 1 : clamp(center + halfWidth, 0, 1),
    };
  }

  return Object.freeze({ formatNumber, frameTransform, armState, inverseArm, simulatePD, actionSchedule, wilsonInterval });
});
