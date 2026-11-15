#!/usr/bin/env node
/* Generate static reading/download diagrams from the interactive framework data. */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const graph = require('../docs/javascripts/vla-framework.js');
const root = path.resolve(__dirname, '..');
const check = process.argv.includes('--check');
let stale = false;
for (const [name, mode] of [
  ['vla-knowledge-framework.svg', 'overview'],
  ['vla-knowledge-framework-full.svg', 'full'],
]) {
  const target = path.join(root, 'docs', 'assets', name);
  const svg = graph.svgFor({mode}).trim() + '\n';
  if (!svg.includes('<svg') || !svg.includes('<title') || !svg.includes('<desc')) throw new Error(`${name}: missing accessible SVG`);
  if (check) {
    if (!fs.existsSync(target) || fs.readFileSync(target, 'utf8') !== svg) {
      process.stderr.write(`${name} is out of date; run node scripts/generate_vla_framework.cjs\n`);
      stale = true;
    }
  } else {
    fs.writeFileSync(target, svg);
    process.stdout.write(`Generated ${name}: ${Buffer.byteLength(svg)} bytes\n`);
  }
}
if (stale) process.exitCode = 1;
else if (check) process.stdout.write('Static VLA framework diagrams match their source data.\n');
