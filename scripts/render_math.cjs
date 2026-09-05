#!/usr/bin/env node
/* Maintainer-only deterministic TeX -> standalone SVG and assistive MathML. */
'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { SerializedMmlVisitor } = require('mathjax-full/js/core/MmlTree/SerializedMmlVisitor.js');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');
require('mathjax-full/js/input/tex/boldsymbol/BoldsymbolConfiguration.js');

const RENDERER = Object.freeze({
  name: 'mathjax-full', version: '3.2.2', output: 'svg', fontCache: 'none',
});
assert.equal(require('mathjax-full/package.json').version, RENDERER.version,
  'Use npm ci: the formula renderer must match its pinned version.');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const input = new TeX({
  // No autoload, HTML, noerrors, or noundefined extension: invalid/unsupported
  // commands must fail instead of turning into red text or raw source.
  packages: ['base', 'ams', 'boldsymbol'],
  formatError: (_jax, error) => { throw error; },
});
const output = new SVG({ fontCache: 'none', internalSpeechTitles: false });
const document = mathjax.document('', {
  InputJax: input,
  OutputJax: output,
  compileError: (_document, _math, error) => { throw error; },
  typesetError: (_document, _math, error) => { throw error; },
});
const visitor = new SerializedMmlVisitor();

function equationKey(tex, display) {
  return crypto.createHash('sha256').update(`${Number(display)}\0${tex}`, 'utf8').digest('hex');
}

function renderExpression(tex, display) {
  assert.equal(typeof tex, 'string');
  assert.ok(tex.trim(), 'Empty formula');
  assert.equal(typeof display, 'boolean');
  input.reset();
  // This follows MathJax's AbstractMathDocument.convert, while retaining the
  // same MathItem's semantic tree for the screen-reader MathML representation.
  const item = new document.options.MathItem(tex, input, display);
  item.start.node = adaptor.body(document.document);
  item.setMetrics(16, 8, 80 * 8, 1000000, 1);
  item.convert(document);
  const svg = adaptor.outerHTML(adaptor.firstChild(item.typesetRoot));
  const mathml = visitor.visitTree(item.root);
  assert.ok(svg.startsWith('<svg '), 'MathJax did not return an SVG');
  assert.ok(mathml.startsWith('<math '), 'MathJax did not return MathML');
  assert.ok(!/<merror\b|data-mjx-error|<script\b|<foreignObject\b|<image\b|<use\b/i.test(svg + mathml),
    'Renderer returned an error, active element or dependent resource');
  assert.ok(!/\b(?:href|src)=|\bon[a-z]+=|url\(/i.test(svg + mathml),
    'Rendered formula must not depend on links, event handlers or external fonts');
  return { tex, display, svg, mathml };
}

function renderCache(collection) {
  assert.ok(collection && typeof collection.expressions === 'object', 'Missing collected expressions');
  const expressions = {};
  for (const key of Object.keys(collection.expressions).sort()) {
    const { tex, display } = collection.expressions[key];
    assert.equal(equationKey(tex, display), key, 'Collected formula key is incorrect');
    try {
      expressions[key] = renderExpression(tex, display);
    } catch (error) {
      throw new Error(`Cannot render ${JSON.stringify(tex)}: ${error.message}`, { cause: error });
    }
  }
  assert.ok(Object.keys(expressions).length > 0, 'No formulas collected');
  return { schema: 1, renderer: RENDERER, expressions };
}

if (require.main === module) {
  try {
    const collection = JSON.parse(fs.readFileSync(0, 'utf8'));
    process.stdout.write(`${JSON.stringify(renderCache(collection), null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { equationKey, renderExpression, renderCache };
