"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const api = require("../../docs/javascripts/vla-concepts.js");

test("concept catalog has ten overlapping views, unique IDs and complete explanations", () => {
  assert.equal(api.dimensions.length, 10);
  assert.ok(api.terms.length >= 50);
  assert.equal(new Set(api.terms.map(t => t.id)).size, api.terms.length);
  const views = new Set(api.dimensions.map(d => d.id));
  for (const term of api.terms) {
    assert.ok(term.views.length && term.views.every(v => views.has(v)), term.id);
    for (const field of ["name", "aliases", "meaning", "location", "contrast", "example"]) {
      assert.ok(typeof term[field] === "string" && term[field].trim(), `${term.id}/${field}`);
    }
  }
});

test("term lookup accepts Chinese, full-width names and pi notation", () => {
  const ids = query => api.searchTerms(query).map(t => t.id);
  assert.equal(ids("ＬｏＲＡ")[0], "lora");
  assert.ok(ids("注意力").includes("attention"));
  assert.ok(ids("RoPE").includes("rope"));
  assert.ok(ids("π₀.₅").includes("pi05"));
  assert.equal(api.normalize(" π₀.₅ "), "pi0.5");
  assert.deepEqual(ids("unregistered_term_98765"), []);
  assert.equal(api.searchTerms("").length, api.terms.length);
});

test("search uses all query words and preserves the selected classification view", () => {
  const found = api.searchTerms("SigLIP ViT", "module").map(t => t.id);
  assert.ok(found.includes("siglip"));
  assert.deepEqual(api.searchTerms("SigLIP 不存在98765"), []);
  assert.ok(api.searchTerms("SigLIP", "instance").some(t => t.id === "siglip"));
  assert.ok(api.searchTerms("SigLIP", "architecture").some(t => t.id === "siglip"));
  assert.ok(!api.searchTerms("SigLIP", "category").some(t => t.id === "siglip"));
});

test("one concrete component can belong to role, architecture and component views", () => {
  const siglip = api.terms.find(t => t.id === "siglip");
  assert.ok(["module", "architecture", "instance"].every(view => siglip.views.includes(view)));
  assert.ok(api.terms.find(t => t.id === "mlp").views.includes("inside"));
  assert.ok(api.terms.find(t => t.id === "mlp").views.includes("architecture"));
});

test("SLA teacher path exists only during training and never becomes a deployment input", () => {
  const train = api.traceFor("sla", "train"), deploy = api.traceFor("sla", "deploy");
  assert.ok(train.spatial.some(n => n.id === "teacher"));
  assert.deepEqual(deploy.spatial, []);
  assert.deepEqual(deploy.supervision, []);
  assert.ok(!deploy.main.some(n => n.id === "teacher"));
  assert.ok(train.main.some(n => n.id === "pali"));
  for (const caseId of Object.keys(api.cases)) {
    assert.ok(!api.traceFor(caseId, "train").main.some(n => n.id === "control"));
    assert.ok(api.traceFor(caseId, "deploy").main.some(n => n.id === "control"));
  }
  assert.ok(![...train.main, ...train.supervision, ...train.spatial].some(n => n.terms.includes("lora")));
  assert.ok(deploy.main.some(n => n.id === "expert"));
});

test("model traces distinguish original token actions from pi05 continuous expert", () => {
  const original = api.traceFor("openvla"), pi05 = api.traceFor("pi05");
  assert.ok(original.main.some(n => n.id === "ar"));
  assert.ok(!original.main.some(n => n.id === "expert"));
  assert.ok(pi05.main.some(n => n.id === "expert"));
  assert.ok(!pi05.main.some(n => n.id === "ar"));
  assert.ok(api.traceFor("pi05", "train").supervision.some(n => n.id === "hybrid"));
});

test("every selectable path node points to real concepts and a primary-source case", () => {
  const ids = new Set(api.terms.map(t => t.id));
  for (const caseId of Object.keys(api.cases)) {
    assert.ok(api.cases[caseId].source.startsWith("https://"));
    for (const stage of ["train", "deploy"]) {
      const trace = api.traceFor(caseId, stage);
      for (const node of [...trace.main, ...trace.supervision, ...trace.spatial]) {
        assert.ok(node.terms.length && node.terms.every(id => ids.has(id)), `${caseId}/${node.id}`);
      }
    }
  }
});

test("unknown views, model cases and stages fail instead of returning an unrelated graph", () => {
  assert.throws(() => api.searchTerms("LoRA", "not-a-view"), RangeError);
  assert.throws(() => api.traceFor("future-model"), RangeError);
  assert.throws(() => api.traceFor("__proto__"), RangeError);
  assert.throws(() => api.searchTerms("LoRA", "constructor"), RangeError);
  assert.throws(() => api.traceFor("sla", "unknown-stage"), RangeError);
});
