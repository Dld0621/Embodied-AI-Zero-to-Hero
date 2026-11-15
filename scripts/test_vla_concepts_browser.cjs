#!/usr/bin/env node
/* Built-site verification. Requires local Playwright; no remote model or browser service. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require('playwright');
const catalog = require('../docs/javascripts/vla-concepts.js');
const base = (process.argv[2] || 'http://127.0.0.1:8765').replace(/\/$/, '');
const artifactDir = process.env.VLA_QA_ARTIFACTS || '';
let checks = 0;
function check(condition, label) { checks++; assert.ok(condition, label); }
async function snapshot(page, name) {
  if (!artifactDir) return;
  await fs.mkdir(artifactDir, {recursive:true});
  await page.screenshot({path:path.join(artifactDir, `${name}.png`), fullPage:true});
}
async function layout(page, label) {
  const state = await page.locator('.md-content').evaluate(node => ({
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    replacement: node.textContent.includes('\ufffd'),
    badMath: [...node.querySelectorAll('.arithmatex')].filter(m =>
      m.dataset.mathRendered !== 'static-svg' || !m.querySelector('.math-assistive math') || m.scrollWidth > m.clientWidth + 2).length,
  }));
  check(!state.overflow, `${label}: no whole-page horizontal overflow`);
  check(!state.replacement && !state.badMath, `${label}: readable text and static formulas fit`);
}
async function setRange(locator, value) {
  await locator.evaluate((input, next) => { input.value = String(next); input.dispatchEvent(new Event('input', {bubbles:true})); }, value);
}
(async () => {
  const browser = await chromium.launch({headless:true});
  try {
    for (const width of [320,360,1440]) for (const theme of ['default','slate']) {
      const context = await browser.newContext({viewport:{width,height:1000}, reducedMotion:'reduce'});
      await context.route('https://**/*', route => route.abort());
      const page = await context.newPage(), errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/31-vla-concept-map/`, {waitUntil:'networkidle'});
      await page.evaluate(value => {document.body.dataset.mdColorScheme=value;document.documentElement.style.scrollBehavior='auto';},theme);
      await page.locator('#vla-term-reference > summary').click();
      const root = page.locator('[data-vla-concepts]');
      await root.locator('.vc-detail h3').waitFor();
      check(await root.locator('[data-view]').count() === 11, 'ten views plus all');
      check(await root.locator('.vc-results [data-term]').count() === catalog.terms.length, 'all term cards rendered');
      check(/SigLIP/.test(await root.locator('.vc-detail').innerText()), 'default selected term');
      await root.locator('input[name="vla-concept-search"]').fill('ＬｏＲＡ');
      await root.locator('.vc-results [data-term="lora"]').click();
      check(/低秩/.test(await root.locator('.vc-detail').innerText()), 'full-width term lookup and explanation');
      check(await root.locator('[data-term="lora"]').evaluate(n=>n===document.activeElement), 'term selection retains keyboard focus');
      await root.locator('input').fill('no_match_812739');
      check(await root.locator('.vc-results [data-term]').count() === 0 && await root.locator('.vc-empty').count() === 1, 'zero-result state');
      await root.locator('input').fill('SigLIP');
      for (const view of ['module','instance','architecture']) {
        await root.locator(`[data-view="${view}"]`).click();
        check(await root.locator('.vc-results [data-term="siglip"]').count() === 1, `SigLIP overlaps ${view}`);
        check(await root.locator(`[data-view="${view}"]`).evaluate(n=>n===document.activeElement), 'view selection retains focus');
      }
      await root.locator('input').fill('');
      await root.locator('[data-view="all"]').click();
      for (const caseId of ['openvla','pi05','sla']) {
        await root.locator('select').selectOption(caseId);
        await root.locator('[data-stage="deploy"]').click();
        check(await root.locator('[data-node="control"]').count() === 1 && await root.locator('.vc-training').count() === 0, `${caseId}: deployment controls, no labels/teacher`);
        if (caseId==='openvla') check(/文字不经过/.test(await root.locator('.vc-branch-note').innerText()), 'text bypasses visual encoders');
        await root.locator('[data-stage="train"]').click();
        check(await root.locator('[data-node="control"]').count() === 0 && await root.locator('.vc-training').count() >= 1, `${caseId}: offline training without command execution`);
        check(await root.locator('[data-node="teacher"]').count() === (caseId==='sla'?1:0), `${caseId}: teacher scope`);
      }
      await root.locator('[data-node="teacher"]').click();
      check(await root.locator('[data-node="teacher"]').evaluate(n=>n===document.activeElement), 'path selection retains focus');
      check(await root.locator('.vc-linked').count() >= 1, 'path selection highlights related concept');
      check(/学生|VLA/.test(await root.locator('[data-node="align"]').innerText()), 'alignment compares student feature to teacher target');
      for (const axis of ['role','modules','architecture','internals','components','data','actions','training','execution','evaluation']) check(await page.locator(`#axis-${axis}`).count()===1, `unique axis ${axis}`);
      await layout(page, `concepts ${width}/${theme}`);
      if(width===320&&theme==='slate') await snapshot(page,'concepts-sla-320-dark');

      await page.goto(`${base}/32-vla-attention-lab/`, {waitUntil:'networkidle'});
      await page.evaluate(value=>document.body.dataset.mdColorScheme=value, theme);
      const lab = page.locator('[data-vla-attention]');
      await lab.locator('[data-va-output]').waitFor();
      check(await lab.locator('[data-va-output]').innerText()==='[0.802, 1.198]', 'default numerical attention output');
      await lab.locator('[data-va-causal]').check();
      check(await lab.locator('[data-va-output]').innerText()==='[1, 0]', 'causal A reads its own value');
      check(await lab.locator('[data-va-cell="0-1"]').innerText()==='禁读', 'future B visibly blocked');
      await lab.locator('[data-va-query="2"]').click();
      check(await lab.locator('[data-va-query="2"]').getAttribute('aria-pressed')==='true', 'query selection exposed');
      await lab.locator('[data-va-causal]').uncheck();
      await lab.locator('[data-va-query="0"]').click();
      await setRange(lab.locator('[data-va-input]'),0);
      check(await lab.locator('[data-va-cell="0-0"]').innerText()==='0.333', 'zero query gives uniform attention');
      await lab.locator('[data-va-reset]').click();
      check(await lab.locator('[data-va-output]').innerText()==='[0.802, 1.198]' && !await lab.locator('[data-va-causal]').isChecked(), 'reset restores the hand example');
      await lab.locator('[data-va-query="1"]').focus();
      await page.keyboard.press('Enter');
      check(await lab.locator('[data-va-query="1"]').getAttribute('aria-pressed')==='true', 'query button works by keyboard');
      await layout(page, `attention ${width}/${theme}`);
      if(width===320&&theme==='default') await snapshot(page,'attention-320-light');
      check(errors.length===0, `no uncaught errors: ${errors.join('; ')}`);
      await context.close();
    }
    // SVG text itself must stay inside the declared canvas.
    const page = await browser.newPage();
    for (const asset of ['vla-concepts-overview','vla-token-attention','vla-training-axes']) {
      await page.goto(`${base}/assets/${asset}.svg`);
      const clipped=await page.locator('text').evaluateAll(nodes=>nodes.filter(n=>{const b=n.getBBox(),v=n.ownerSVGElement.viewBox.baseVal;return b.x<0||b.y<0||b.x+b.width>v.width+1||b.y+b.height>v.height+1;}).map(n=>n.textContent));
      check(clipped.length===0, `${asset}: no clipped SVG text (${clipped})`);
    }
    await page.close();
    const nojs = await browser.newContext({javaScriptEnabled:false,viewport:{width:320,height:1000}});
    const reading = await nojs.newPage();
    for (const slug of ['31-vla-concept-map','32-vla-attention-lab']) {
      await reading.goto(`${base}/${slug}/`);
      check(await reading.locator('.md-content h2').count()>=7, `${slug}: static reading works without JS`);
      check(await reading.locator('.arithmatex[data-math-rendered="static-svg"]').count()===3, `${slug}: all three formulas exist without JS`);
    }
    await nojs.close();
    const context = await browser.newContext();
    const instant = await context.newPage();
    // Make Material's production sitemap resolve through the local preview.
    await instant.route('**/sitemap.xml', async route=>{
      const response=await route.fetch();
      await route.fulfill({response,body:(await response.text()).replaceAll('https://dld0621.github.io/Embodied-AI-Zero-to-Hero',base)});
    });
    await instant.goto(`${base}/31-vla-concept-map/`,{waitUntil:'networkidle'});
    await instant.evaluate(()=>window.__vlaInstantMarker=321);
    await instant.locator('.md-content a[href*="32-vla-attention-lab"]').first().click();
    await instant.locator('[data-vla-attention] [data-va-output]').waitFor();
    check(await instant.evaluate(()=>window.__vlaInstantMarker===321), 'attention page uses real instant navigation');
    await instant.locator('.md-content a[href*="31-vla-concept-map"]').first().click();
    await instant.locator('#vla-term-reference > summary').click();
    await instant.locator('[data-vla-concepts] .vc-detail h3').waitFor();
    check(await instant.locator('[data-vla-concepts] input[name="vla-concept-search"]').count()===1, 'instant return initializes one concept explorer');
    await instant.goto(`${base}/learning-lab-cn/`);
    await instant.locator('[data-lab-tab="vla"]').waitFor();
    check(await instant.locator('.eai-labs [role="tab"]').count()===7, 'existing seven-module lab remains initialized');
    await context.close();
    console.log(`VLA concept browser QA passed: ${checks} checks; 67 terms, ten views, model/stage boundaries, numeric attention, 320/360/1440px, both themes, blocked remote assets, no-JS and instant navigation.`);
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
