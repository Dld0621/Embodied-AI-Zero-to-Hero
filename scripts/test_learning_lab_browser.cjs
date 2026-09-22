#!/usr/bin/env node
/* End-to-end checks for the built MkDocs laboratory. No backend or GPU needed. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

const baseURL = (process.argv[2] || 'http://127.0.0.1:8765').replace(/\/$/, '');
const artifactDir = process.env.LAB_QA_ARTIFACTS || '';
const modules = ['frames', 'kinematics', 'control', 'timing', 'evaluation'];
const changes = {
  frames: ['frame-theta', '65'],
  kinematics: ['arm-q1', '65'],
  control: ['control-kp', '15'],
  timing: ['timing-latency', '180'],
  evaluation: ['eval-trials', '250'],
};
let assertions = 0;

function check(condition, message) {
  assertions += 1;
  assert.ok(condition, message);
}

async function waitForLab(page) {
  await page.locator('.eai-labs [data-lab-tab="frames"]').waitFor();
  await page.waitForFunction(() => {
    const roots = document.querySelectorAll('.eai-labs');
    return roots.length === 1 && roots[0].querySelectorAll('.eai-result').length >= 5;
  });
}

async function checkStaticMath(page, label, expected = 62) {
  const formulas = page.locator('.md-content .arithmatex');
  check(await formulas.count() === expected, `${label}: all ${expected} source formulas are present`);
  const invalid = await formulas.evaluateAll((nodes) => nodes.flatMap((node, index) => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll('.math-visual, .math-assistive').forEach((child) => child.remove());
    return node.dataset.mathRendered !== 'static-svg' ||
      !node.querySelector('.math-visual svg path') || !node.querySelector('.math-assistive math') ||
      node.querySelector('[data-mjx-error], merror') || copy.textContent.trim()
      ? [index] : [];
  }));
  check(invalid.length === 0, `${label}: SVG + semantic MathML, no visible raw TeX or renderer errors (${invalid})`);
  check(await page.locator('script[src*="mathjax"], script[src*="unpkg"]').count() === 0,
    `${label}: no runtime math script or CDN dependency`);
}

async function checkMathLayout(page, label) {
  const clipped = await page.locator('.arithmatex').evaluateAll((nodes) => nodes.flatMap((node, index) => {
    if (!node.getClientRects().length) return [];
    const style = getComputedStyle(node);
    const svg = node.querySelector('.math-visual svg');
    if (!svg || !svg.getClientRects().length) return [];
    const inlineClipped = node.tagName === 'SPAN' && !node.classList.contains('math-wide') && style.overflowY !== 'visible';
    const scrollable = node.tagName === 'DIV' || (node.classList.contains('math-wide') && style.overflowX === 'auto');
    const blockClipped = scrollable && node.clientHeight + 1 < node.scrollHeight;
    const ink = svg.getBoundingClientRect();
    const viewportClipped = !scrollable && (ink.left < -1 || ink.right > document.documentElement.clientWidth + 1);
    const unreachableStart = scrollable && node.scrollLeft === 0 && ink.left < node.getBoundingClientRect().left - 1;
    return inlineClipped || blockClipped || viewportClipped || unreachableStart ? [index] : [];
  }));
  check(clipped.length === 0, `${label}: inline glyphs and display fractions are not clipped vertically or beyond the viewport (${clipped})`);
}

async function openModule(page, name) {
  await page.locator(`[data-lab-tab="${name}"]`).click();
  const panel = page.locator(`[data-lab-panel="${name}"]`);
  await panel.waitFor({ state: 'visible' });
  check(await page.locator(`[data-lab-tab="${name}"]`).getAttribute('aria-selected') === 'true', `${name}: active tab exposed to assistive technology`);
  return panel;
}

async function setInput(panel, name, value) {
  const input = panel.locator(`[name="${name}"]`);
  check(await input.count() === 1, `one input named ${name}`);
  await input.evaluate((node, next) => {
    node.value = next;
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function parameterSnapshot(root) {
  return root.locator('input[name], select[name]').evaluateAll((nodes) =>
    Object.fromEntries(nodes.map((node) => [node.name, node.value])));
}

async function clippedChartLabels(panel) {
  return panel.locator('.eai-chart').evaluate((svg) => {
    const view = svg.viewBox.baseVal;
    return [...svg.querySelectorAll('text')].filter((label) => {
      const box = label.getBBox();
      return box.x < view.x - 1 || box.y < view.y - 1 ||
        box.x + box.width > view.x + view.width + 1 || box.y + box.height > view.y + view.height + 1;
    }).map((label) => label.textContent);
  });
}

async function testLanguage(browser, lang, slug) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  // Block all external resources, including any accidentally reintroduced math CDN.
  await context.route('https://**/*', (route) =>
    new URL(route.request().url()).origin === new URL(baseURL).origin ? route.continue() : route.abort());
  // Material only intercepts links listed in the sitemap. A production build
  // served on localhost has a different port and project prefix. Normalize the
  // preview sitemap, not the page/app, so the actual instant router is exercised.
  await context.route('**/sitemap.xml', async (route) => {
    const response = await route.fetch();
    const xml = await response.text();
    const rootMatch = xml.match(/<loc>\s*([^<]+?)\s*<\/loc>/);
    await route.fulfill({ response, body: rootMatch ? xml.replaceAll(rootMatch[1], `${baseURL}/`) : xml });
  });
  await page.goto(`${baseURL}/${slug}/#frames`, { waitUntil: 'domcontentloaded' });
  await waitForLab(page);
  await checkStaticMath(page, lang);
  const root = page.locator('.eai-labs');
  check(await root.getAttribute('data-lab-lang') === lang, `${lang}: correct laboratory language`);
  check(await root.locator('[role="tab"]').count() === modules.length, `${lang}: exactly five tabs`);
  check(await root.locator('.eai-chart').count() >= modules.length, `${lang}: each module has a chart`);
  check(await root.locator('.eai-result[aria-live]').count() === modules.length, `${lang}: result announcements exist`);
  const unlabelled = await root.locator('input').evaluateAll((inputs) => inputs.filter((input) =>
    !(input.labels && input.labels.length) && !input.getAttribute('aria-label') && !input.getAttribute('aria-labelledby'))
    .map((input) => input.name));
  check(unlabelled.length === 0, `${lang}: every input has an accessible label (${unlabelled.join(', ')})`);

  for (const name of modules) {
    const panel = await openModule(page, name);
    const original = await parameterSnapshot(panel);
    const originalResult = await panel.locator('.eai-result').innerText();
    let [inputName, value] = changes[name];
    if (original[inputName] === value) value = String(Number(value) + 5);
    await setInput(panel, inputName, value);
    const changedResult = await panel.locator('.eai-result').innerText();
    check(changedResult !== originalResult, `${lang}/${name}: changing a parameter changes the explanation`);
    check(!/NaN|undefined|Infinity/.test(changedResult), `${lang}/${name}: finite readable result`);
    const reset = panel.locator('[data-action="reset"]');
    await (await reset.count() ? reset : root.locator('[data-action="reset"]').first()).click();
    check(JSON.stringify(await parameterSnapshot(panel)) === JSON.stringify(original), `${lang}/${name}: reset restores module defaults`);
    check(await panel.locator('.eai-result').innerText() === originalResult, `${lang}/${name}: reset restores calculated result`);
  }

  const arm = await openModule(page, 'kinematics');
  await setInput(arm, 'arm-target-x', '1.9');
  await setInput(arm, 'arm-target-y', '1.9');
  await arm.locator('[data-action="inverse"]').click();
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  check(/unreachable|outside|不可达|工作空间|无法到达/i.test(await arm.innerText()), `${lang}: unreachable target explained`);
  check(!/NaN|undefined|Infinity/.test(await arm.locator('.eai-result').innerText()), `${lang}: unreachable target does not corrupt arm`);

  const timing = await openModule(page, 'timing');
  const timingBefore = await timing.locator('.eai-result').innerText();
  await setInput(timing, 'timing-horizon', '20');
  check(await timing.locator('.eai-result').innerText() === timingBefore, `${lang}: predicted horizon alone does not alter fixed-K scheduling`);
  await setInput(timing, 'timing-horizon', '1');
  check(await timing.locator('.eai-result').getAttribute('data-error') === 'true', `${lang}: K greater than H is an explicit validation error`);
  check(await timing.locator('.eai-chart *').count() === 0, `${lang}: invalid input never leaves a misleading stale plot`);
  await timing.locator('[data-action="reset"]').click();

  const evaluation = await openModule(page, 'evaluation');
  await setInput(evaluation, 'eval-trials', '0');
  check(/^0\s/.test(await evaluation.locator('[data-value-for="eval-trials"]').innerText()), `${lang}: invalid input still updates its displayed value`);
  await setInput(evaluation, 'eval-successes', '99999');
  check(!/NaN|undefined|Infinity/.test(await evaluation.locator('.eai-result').innerText()), `${lang}: invalid sample sizes never produce non-finite output`);
  await setInput(evaluation, 'eval-trials', '');
  check(!/NaN|undefined|Infinity/.test(await evaluation.locator('.eai-result').innerText()), `${lang}: blank sample size handled`);
  await evaluation.locator('[data-action="reset"]').click();

  const downloadPromise = page.waitForEvent('download');
  const exportButton = evaluation.locator('[data-action="export"]');
  await (await exportButton.count() ? exportButton : root.locator('[data-action="export"]').first()).click();
  const download = await downloadPromise;
  check(download.suggestedFilename().endsWith('.json'), `${lang}: export is a JSON learning record`);
  const downloaded = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
  const serialized = JSON.stringify(downloaded);
  check(/learning|学习|educational/i.test(serialized), `${lang}: export identifies itself as a learning record`);
  check(/param/i.test(serialized) && /result/i.test(serialized), `${lang}: export includes parameters and calculated results`);
  check(downloaded.assessed === false, `${lang}: export is explicitly not assessed curriculum progress`);
  check(downloaded.evidenceLevel === 'interactive-teaching-model', `${lang}: export preserves teaching-model evidence boundary`);
  check(downloaded.parameters['eval-trials'] === 100 && downloaded.parameters['eval-successes'] === 80, `${lang}: exported parameters match displayed defaults`);
  check(downloaded.result.rate === 0.8 && downloaded.result.lower < 0.8 && downloaded.result.upper > 0.8, `${lang}: exported interval contains displayed point estimate`);

  for (const name of modules) {
    await page.goto(`${baseURL}/${slug}/#${name}`, { waitUntil: 'domcontentloaded' });
    await waitForLab(page);
    await page.locator(`[data-lab-panel="${name}"]`).waitFor({ state: 'visible' });
    check(await page.locator(`[data-lab-tab="${name}"]`).getAttribute('aria-selected') === 'true', `${lang}/${name}: deep link selects matching tab`);
  }

  const framesTab = page.locator('[data-lab-tab="frames"]');
  await framesTab.click();
  await framesTab.focus();
  await page.keyboard.press('ArrowRight');
  check(await page.locator('[data-lab-tab="kinematics"]').getAttribute('aria-selected') === 'true', `${lang}: tabs support ArrowRight navigation`);
  await page.keyboard.press('Home');
  check(await framesTab.getAttribute('aria-selected') === 'true', `${lang}: tabs support Home navigation`);

  // Check all diagrams at narrow and wide widths in both Material schemes.
  for (const width of [320, 360, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const scheme of ['default', 'slate']) {
      await page.locator('body').evaluate((body, value) => body.setAttribute('data-md-color-scheme', value), scheme);
      await checkMathLayout(page, `${lang}/${width}/${scheme}`);
      for (const name of modules) {
        await openModule(page, name);
        const dimensions = await page.evaluate(() => ({
          viewport: document.documentElement.clientWidth,
          content: document.documentElement.scrollWidth,
          lab: document.querySelector('.eai-labs').getBoundingClientRect().width,
        }));
        check(dimensions.content <= dimensions.viewport + 1, `${lang}/${name}: no horizontal overflow at ${width}px in ${scheme} (${JSON.stringify(dimensions)})`);
        check(dimensions.lab > 0, `${lang}/${name}: laboratory visible at ${width}px in ${scheme}`);
        const clippedLabels = await clippedChartLabels(page.locator(`[data-lab-panel="${name}"]`));
        check(clippedLabels.length === 0, `${lang}/${name}: no clipped SVG text at ${width}px in ${scheme}: ${clippedLabels.join('; ')}`);
        if (artifactDir) await root.screenshot({
          path: path.join(artifactDir, `${lang}-${name}-${width}-${scheme}.png`),
          // Element screenshots scroll the lab under the sticky site header.
          // Hide only that unrelated overlay in the saved teaching-panel image.
          style: '.md-header { visibility: hidden !important; }',
        });
      }
    }
  }

  await page.setViewportSize({ width: 320, height: 1000 });
  const extremeFrame = await openModule(page, 'frames');
  for (const [name, value] of Object.entries({
    'frame-theta': '45', 'frame-tx': '1.5', 'frame-ty': '1.5', 'frame-x': '1', 'frame-y': '-1',
  })) await setInput(extremeFrame, name, value);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const extremeLabels = await clippedChartLabels(extremeFrame);
  check(extremeLabels.length === 0, `${lang}: extreme rightmost frame point label remains inside 320px plot (${extremeLabels.join('; ')})`);
  check(/\(2\.91, 1\.50\)/.test(await extremeFrame.locator('.eai-result').innerText()), `${lang}: extreme frame point agrees with hand calculation`);
  if (artifactDir) await root.screenshot({
    path: path.join(artifactDir, `${lang}-frames-extreme-320-slate.png`),
    style: '.md-header { visibility: hidden !important; }',
  });

  // A repeated DOM-ready event must not duplicate the fallback initialization.
  // The real Material navigation check below also catches stale/duplicate roots.
  await page.evaluate(() => { document.dispatchEvent(new Event('DOMContentLoaded')); });
  check(await page.locator('.eai-labs [role="tab"]').count() === modules.length, `${lang}: initialization remains idempotent`);
  const otherSlug = lang === 'zh' ? 'learning-lab' : 'learning-lab-cn';
  const otherLanguage = lang === 'zh' ? 'en' : 'zh';
  const switchLink = page.locator(`.md-content a[href*="${otherSlug}/"]`).first();
  check(await switchLink.count() > 0, `${lang}: visible link to other language`);
  await page.evaluate(() => { window.__labQAInstantMarker = 'same-document'; });
  await switchLink.click();
  await page.waitForURL(new RegExp(`/${otherSlug}/`));
  await page.locator(`.eai-labs[data-lab-lang="${otherLanguage}"]`).waitFor();
  await waitForLab(page);
  await checkStaticMath(page, `${lang}: after instant navigation`);
  check(await page.evaluate(() => window.__labQAInstantMarker) === 'same-document', `${lang}: language switch exercised Material instant navigation, not a full reload`);
  check(await page.locator('.eai-labs [role="tab"]').count() === modules.length, `${lang}: instant language navigation produces one initialized lab`);
  const switchedPanel = await openModule(page, 'frames');
  const before = await switchedPanel.locator('.eai-result').innerText();
  await setInput(switchedPanel, 'frame-theta', '47');
  check(await switchedPanel.locator('.eai-result').innerText() !== before, `${lang}: controls work after instant navigation`);
  // Repeat actual same-document language switches; formulas must never need a
  // second asynchronous typesetting pass or accumulate duplicate output.
  for (let turn = 0; turn < 3; turn += 1) {
    const destination = turn % 2 === 0 ? slug : otherSlug;
    await page.locator(`.md-content a[href*="${destination}/"]`).first().click();
    await page.waitForURL(new RegExp(`/${destination}/`));
    await waitForLab(page);
    await checkStaticMath(page, `${lang}: repeat navigation ${turn + 1}`);
  }
  check(errors.length === 0, `${lang}: no uncaught browser errors (${errors.join('; ')})`);
  await context.close();

  const staticContext = await browser.newContext({ javaScriptEnabled: false });
  await staticContext.route('https://**/*', (route) =>
    new URL(route.request().url()).origin === new URL(baseURL).origin ? route.continue() : route.abort());
  const staticPage = await staticContext.newPage();
  await staticPage.goto(`${baseURL}/${slug}/`, { waitUntil: 'domcontentloaded' });
  const prose = await staticPage.locator('.md-content').innerText();
  check(/worked example|手算|算例|例题/i.test(prose), `${lang}: worked examples remain readable without JavaScript`);
  check(prose.length > 2000, `${lang}: substantive static lesson remains without JavaScript`);
  await checkStaticMath(staticPage, `${lang}: JavaScript disabled`);
  for (const width of [320, 1440]) {
    await staticPage.setViewportSize({ width, height: 1000 });
    // Include formulas inside the worked-answer disclosure sections.
    await staticPage.locator('details').evaluateAll((nodes) => nodes.forEach((node) => { node.open = true; }));
    await checkMathLayout(staticPage, `${lang}: no-JS/${width}`);
  }
  await staticContext.close();
}

(async () => {
  if (artifactDir) await fs.mkdir(artifactDir, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    await testLanguage(browser, 'zh', 'learning-lab-cn');
    await testLanguage(browser, 'en', 'learning-lab');
    console.log(`Learning laboratory browser QA passed: ${assertions} checks; both languages, five modules, 320/360/1440px, light/dark, offline resources, no-JS fallback.`);
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
