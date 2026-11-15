#!/usr/bin/env node
/* Real-browser acceptance of the built VLA knowledge framework.
 * Requires Playwright; FRAMEWORK_QA_CHROME may select an installed browser.
 * Pass --layout-only after the URL for focused desktop/resize verification.
 * This validates diagram navigation/layout, not the scientific truth of the graph.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require('playwright');
const base = (process.argv[2] || 'http://127.0.0.1:8765').replace(/\/$/, '');
const baseOrigin = new URL(base).origin;
const artifacts = process.env.FRAMEWORK_QA_ARTIFACTS || '';
let checks = 0;
const boundaries = [];
function check(value, label) { checks++; assert.ok(value, label); }
async function offline(context) {
  await context.route('**/*', route => {
    const target = new URL(route.request().url());
    return target.protocol === 'https:' && target.origin !== baseOrigin ? route.abort() : route.continue();
  });
}
async function ready(page) {
  const root = page.locator('[data-vla-framework]');
  await root.locator('.vf-canvas svg').waitFor({state:'visible'});
  return root;
}
async function viewState(root) {
  return root.locator('.vf-canvas svg').evaluate(svg => ({
    viewBox:svg.getAttribute('viewBox'),
    transforms:[...svg.querySelectorAll('g[transform]')].map(n=>n.getAttribute('transform')).join('|'),
    mode:svg.dataset.vfMode || svg.closest('[data-vla-framework]').dataset.vfMode || '',
  }));
}
async function graphLayout(page, root, label) {
  const state = await root.evaluate(node => ({
    pageOverflow:document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    frameWidth:node.querySelector('.vf-canvas').getBoundingClientRect().width,
    replacement:node.textContent.includes('\ufffd'),
    svg:!!node.querySelector('svg title') && !!node.querySelector('svg desc'),
    cardCount:node.querySelectorAll('.vc-results,.vc-detail').length,
  }));
  check(!state.pageOverflow, `${label}: canvas does not overflow the whole page`);
  check(state.frameWidth > 200 && !state.replacement, `${label}: visible frame and readable encoding`);
  check(state.svg && state.cardCount === 0, `${label}: accessible SVG is the main framework`);
  const badMath = await page.locator('.arithmatex').evaluateAll(nodes => nodes.filter(n =>
    n.dataset.mathRendered !== 'static-svg' || !n.querySelector('.math-assistive math')).length);
  check(badMath === 0, `${label}: static equations remain rendered`);
}
async function textGeometry(root, label) {
  const issues=await root.locator('.vf-canvas svg').evaluate(svg=>{
    const bounds=[...svg.querySelectorAll('[data-vf-kind="group"]')].map(n=>{
      const rects=[...n.querySelectorAll('text')].map(t=>t.getBoundingClientRect());
      return {id:n.dataset.vfNode,branch:n.dataset.vfBranch,left:Math.min(...rects.map(r=>r.left)),right:Math.max(...rects.map(r=>r.right)),top:Math.min(...rects.map(r=>r.top)),bottom:Math.max(...rects.map(r=>r.bottom))};
    });
    const intersections=[];
    for(let i=0;i<bounds.length;i++)for(let j=i+1;j<bounds.length;j++){
      const a=bounds[i],b=bounds[j];
      if(a.branch===b.branch&&Math.min(a.right,b.right)-Math.max(a.left,b.left)>.5&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>.5)intersections.push([a.id,b.id]);
    }
    return intersections;
  });
  check(issues.length===0,`${label}: group titles and examples do not overlap (${JSON.stringify(issues)})`);
}
async function mobileReadability(root,label) {
  const data=await root.evaluate(node=>{
    const frame=node.querySelector('.vf-canvas').getBoundingClientRect();
    return [...node.querySelectorAll('[data-vf-kind="leaf"] .vf-label')].map(text=>{
      const r=text.getBoundingClientRect(),m=text.getScreenCTM();
      return {text:text.textContent,pixels:parseFloat(getComputedStyle(text).fontSize)*Math.hypot(m.a,m.b),full:r.left>=frame.left-1&&r.right<=frame.right+1&&r.top>=frame.top-1&&r.bottom<=frame.bottom+1};
    }).filter(item=>item.full);
  });
  check(data.length>0,`${label}: at least one complete term is initially visible in the canvas`);
  check(data.length>0&&data.every(item=>item.pixels>=11.5),`${label}: visible terms use a readable rendered font size (${data.map(n=>n.pixels.toFixed(1)).join(',')}px)`);
}
async function capture(page, root, name) {
  if (!artifacts) return;
  await fs.mkdir(artifacts, {recursive:true});
  const style=await page.addStyleTag({content:'.md-header{visibility:hidden!important}'});
  try { await root.screenshot({path:path.join(artifacts,`${name}.png`),animations:'disabled'}); }
  finally { await style.evaluate(node=>node.remove()); }
}
async function downloadedSvg(page, root) {
  const waiting=page.waitForEvent('download');
  await root.locator('[data-vf-action="download"]').click();
  const download=await waiting;
  check(download.suggestedFilename().endsWith('.svg'), 'framework export uses SVG');
  const contents=await fs.readFile(await download.path(),'utf8');
  const data=await page.evaluate(source=>{
    const doc=new DOMParser().parseFromString(source,'image/svg+xml');
    return {
      parseError:!!doc.querySelector('parsererror'),
      root:doc.documentElement.localName,
      title:doc.querySelector('title')?.textContent,
      desc:doc.querySelector('desc')?.textContent,
      branches:doc.querySelectorAll('[data-vf-kind="branch"]').length,
      leaves:doc.querySelectorAll('[data-vf-kind="leaf"]').length,
      unsafe:!!doc.querySelector('script,foreignObject') || [...doc.querySelectorAll('*')].some(n=>[...n.attributes].some(a=>/^on/i.test(a.name)||/javascript:/i.test(a.value))),
      text:doc.documentElement.textContent,
    };
  },contents);
  check(!data.parseError && data.root==='svg' && !!data.title && !!data.desc, 'download is accessible, parseable standalone SVG');
  check(!data.unsafe, 'SVG has no scripts, HTML foreign objects or event handlers');
  check(data.branches===10 && data.leaves>50, 'full export contains ten branches and detailed terms');
  check(/LoRA/.test(data.text) && /Transformer/.test(data.text) && /SigLIP/.test(data.text), 'full export contains the requested deeper knowledge');
  const geometry=await page.evaluate(source=>{
    const host=document.createElement('div');host.style.cssText='position:absolute;left:-20000px;top:0;pointer-events:none;visibility:hidden';
    host.innerHTML=source;document.body.append(host);
    try{
      const svg=host.querySelector('svg'),width=Number(svg.getAttribute('width')),height=Number(svg.getAttribute('height'));
      const texts=[...svg.querySelectorAll('text')].map(text=>{
        const b=text.getBBox();return {id:text.closest('[data-vf-node]')?.dataset.vfNode||'footer',text:text.textContent,left:b.x,right:b.x+b.width,top:b.y,bottom:b.y+b.height};
      });
      const outside=texts.filter(t=>t.left<-.5||t.top<-.5||t.right>width+.5||t.bottom>height+.5).map(t=>({id:t.id,text:t.text}));
      const collisions=[];
      for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
        const a=texts[i],b=texts[j];
        if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>.5&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>.5)collisions.push([a.id,b.id]);
      }
      return {outside,collisions};
    }finally{host.remove();}
  },contents);
  check(geometry.outside.length===0,`full SVG keeps every text label inside the exported viewBox (${JSON.stringify(geometry.outside.slice(0,8))})`);
  check(geometry.collisions.length===0,`full SVG text labels do not overlap across hierarchy levels (${JSON.stringify(geometry.collisions.slice(0,12))})`);
}
async function fullscreen(page,root) {
  const enabled=await page.evaluate(()=>document.fullscreenEnabled);
  const button=root.locator('[data-vf-action="fullscreen"]');
  await button.click();
  try { await page.waitForFunction(()=>!!document.fullscreenElement,null,{timeout:2500}); } catch {}
  if(await page.evaluate(()=>!!document.fullscreenElement)) {
    check(true,'fullscreen enters using the visible control');
    await button.click();
    await page.waitForFunction(()=>!document.fullscreenElement);
    check(true,'fullscreen exits using the visible control');
  } else {
    const text=await root.innerText();
    check(!enabled || /不支持|无法|不可用|失败|未允许|unsupported|unavailable/i.test(text), 'unavailable fullscreen has an explicit UI boundary');
    boundaries.push('Browser did not enter fullscreen; the interface availability/error path was checked.');
  }
}
async function responsiveFollowup(browser) {
  for(const theme of ['default','slate']) {
    const context=await browser.newContext({viewport:{width:1440,height:1100},reducedMotion:'reduce'});await offline(context);
    const page=await context.newPage();await page.goto(`${base}/31-vla-concept-map/`,{waitUntil:'networkidle'});
    await page.evaluate(value=>{document.body.dataset.mdColorScheme=value;document.documentElement.style.scrollBehavior='auto';},theme);
    const root=await ready(page);
    await page.waitForFunction(()=>document.querySelector('[data-vla-framework] .vf-canvas').getBoundingClientRect().height>=1000);
    await graphLayout(page,root,`desktop followup/${theme}`);await textGeometry(root,`desktop followup/${theme}`);
    const fonts=await root.locator('.vf-canvas svg').evaluate(svg=>{
      const pixels=selector=>[...svg.querySelectorAll(selector)].map(text=>{const m=text.getScreenCTM();return parseFloat(getComputedStyle(text).fontSize)*Math.hypot(m.a,m.b);});
      return {groups:Math.min(...pixels('[data-vf-kind="group"] .vf-label')),examples:Math.min(...pixels('[data-vf-kind="group"] .vf-sub'))};
    });
    check(fonts.groups>=16&&fonts.examples>=12.5,`desktop/${theme}: overview groups ${fonts.groups.toFixed(1)}px and examples ${fonts.examples.toFixed(1)}px are readable`);
    await capture(page,root,`framework-1440-${theme}`);
    await root.locator('select[name="vla-framework-branch"]').selectOption('internals');
    await page.setViewportSize({width:320,height:1100});
    await root.locator('.vf-canvas svg[data-vf-mode="internals"][data-vf-compact="true"]').waitFor();
    check(true,`resize/${theme}: an existing focused branch changes to the compact mobile tree`);
    await graphLayout(page,root,`mobile resize/${theme}`);await textGeometry(root,`mobile resize/${theme}`);await mobileReadability(root,`mobile resize/${theme}`);
    await capture(page,root,`framework-320-${theme}-focused-internals-resized`);
    await page.setViewportSize({width:1440,height:1100});
    await root.locator('.vf-canvas svg[data-vf-mode="internals"][data-vf-compact="false"]').waitFor();
    check(await root.locator('.vf-canvas svg').count()===1,`resize/${theme}: returning to desktop preserves the branch without duplicate SVGs`);
    await context.close();
  }
  console.log(`VLA framework layout followup passed: ${checks} checks on Chromium ${browser.version()}; readable desktop overview, both themes and bidirectional compact resize.`);
}
(async()=>{
  const browser=await chromium.launch({headless:true,...(process.env.FRAMEWORK_QA_CHROME?{executablePath:process.env.FRAMEWORK_QA_CHROME}:{})});
  try {
    if(process.argv.includes('--layout-only')) {await responsiveFollowup(browser);return;}
    for(const width of [320,360,1440]) for(const theme of ['default','slate']) {
      const context=await browser.newContext({viewport:{width,height:1100},reducedMotion:'reduce'});
      await offline(context);
      const page=await context.newPage(), errors=[];
      page.on('pageerror',e=>errors.push(e.message));
      await page.goto(`${base}/31-vla-concept-map/`,{waitUntil:'networkidle'});
      await page.evaluate(value=>{document.body.dataset.mdColorScheme=value;document.documentElement.style.scrollBehavior='auto';},theme);
      const root=await ready(page), svg=root.locator('.vf-canvas svg');
      if(width<=360) {
        check(await svg.locator('[data-vf-kind="leaf"]').count()>5,'small screens initially open a detailed branch');
        await mobileReadability(root,`${width}/${theme} initial focus`);
        await capture(page,root,`framework-${width}-${theme}-initial-focus`);
      }
      await root.locator('[data-vf-action="reset"]').click();
      check(await svg.locator('[data-vf-kind="root"]').count()===1,'one central topic');
      check(await svg.locator('[data-vf-kind="branch"]').count()===10,'ten connected main branches in overview');
      check(await svg.locator('[data-vf-kind="group"]').count()>=30,'overview expands into actual knowledge groups');
      check(await svg.locator('path').count()>=30,'diagram draws connected hierarchy paths');
      check(await page.locator('#vla-term-reference').getAttribute('open')===null,'term cards are secondary and initially collapsed');
      await graphLayout(page,root,`${width}/${theme}`);
      await textGeometry(root,`${width}/${theme} overview`);
      const initial=JSON.stringify(await viewState(root));
      await root.locator('[data-vf-action="in"]').click();
      check(JSON.stringify(await viewState(root))!==initial,'zoom-in changes the graph viewport');
      await root.locator('.vf-canvas').scrollIntoViewIfNeeded();
      const beforeDrag=JSON.stringify(await viewState(root)), box=await root.locator('.vf-canvas').boundingBox();
      await page.mouse.move(box.x+box.width*.80,box.y+box.height*.94);
      await page.mouse.down();await page.mouse.move(box.x+box.width*.80+35,box.y+box.height*.94-20,{steps:6});await page.mouse.up();
      check(JSON.stringify(await viewState(root))!==beforeDrag,'drag pans the large graph inside its frame');
      await root.locator('[data-vf-action="fit"]').focus();await page.keyboard.press('Enter');
      check(JSON.stringify(await viewState(root))===initial,'fit works by keyboard and restores the overview');
      await root.locator('.vf-canvas').focus();await page.keyboard.press('ArrowRight');
      check(JSON.stringify(await viewState(root))!==initial,'arrow keys pan the focused graph');
      await page.keyboard.press('0');
      check(JSON.stringify(await viewState(root))===initial,'keyboard zero fits the graph');
      await page.keyboard.press('+');
      check(JSON.stringify(await viewState(root))!==initial,'keyboard plus zooms the focused graph');
      await page.keyboard.press('0');
      const internals=svg.locator('[data-vf-kind="branch"][data-vf-branch="internals"]');
      await internals.focus();await page.keyboard.press('Enter');
      check(await svg.locator('[data-vf-kind="leaf"]').count()>5,'keyboard activation expands an actual diagram branch');
      await root.locator('[data-vf-action="reset"]').click();
      await internals.click();
      check(await svg.locator('[data-vf-kind="leaf"]').count()>5,'clicking an actual diagram branch expands its knowledge');
      await root.locator('[data-vf-action="reset"]').click();
      const options=await root.locator('select[name="vla-framework-branch"] option').evaluateAll(nodes=>nodes.map(n=>({value:n.value,text:n.textContent})));
      const branch=options.find(n=>/网络内部/.test(n.text));
      check(!!branch,'network-internals branch is selectable');
      await root.locator('select[name="vla-framework-branch"]').selectOption(branch.value);
      check(await svg.locator('[data-vf-kind="leaf"]').count()>5,'branch focus reveals detailed terminology');
      check(/Attention|注意力/.test(await svg.textContent()),'focused internals contain attention knowledge');
      await textGeometry(root,`${width}/${theme} focused`);
      if(width<=360) {await mobileReadability(root,`${width}/${theme} focused internals`);await capture(page,root,`framework-${width}-${theme}-focused-internals`);}
      await downloadedSvg(page,root);
      await root.locator('[data-vf-action="relations"]').click();
      const related=await svg.textContent();
      check(/SigLIP/.test(related)&&/ViT/.test(related),'relations view links a concrete component to other views');
      check(/担任|负责/.test(related)&&/采用|实现/.test(related)&&/预训练/.test(related),'cross-view links explain their relation');
      check(await svg.locator('[stroke-dasharray],.vf-relation').count()>=3,'relation edges are visually distinct from grouping edges');
      const named=await svg.locator('[data-relation]').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('data-relation')));
      check(['担任','采用','用作初始化'].every(name=>named.includes(name)),'SigLIP links have three precise, machine-readable relation names');
      await root.locator('[data-vf-action="reset"]').click();
      check(await svg.locator('[data-vf-kind="branch"]').count()===10,'reset returns to the complete framework');
      await root.locator('[data-vf-action="out"]').click();
      check(JSON.stringify(await viewState(root))!==initial,'zoom-out changes the graph viewport');
      await root.locator('[data-vf-action="fit"]').click();
      if(width===1440&&theme==='default') await fullscreen(page,root);
      await graphLayout(page,root,`${width}/${theme} after interaction`);
      if(width===1440||width===320) await capture(page,root,`framework-${width}-${theme}`);
      await page.locator('#vla-term-reference > summary').click();
      await page.locator('[data-vla-concepts] .vc-detail h3').waitFor();
      check(await page.locator('[data-vla-concepts] [name="vla-concept-search"]').count()===1,'optional term tool still opens');
      check(errors.length===0,`no uncaught script errors (${errors})`);
      await context.close();
    }
    const nojs=await browser.newContext({javaScriptEnabled:false,viewport:{width:320,height:1100}});
    await offline(nojs);
    const page=await nojs.newPage();await page.goto(`${base}/31-vla-concept-map/`);
    const fallback=page.locator('img[src*="vla-knowledge-framework"]');
    check(await fallback.count()>=1,'no-JS page contains the static framework diagram');
    check(await fallback.first().evaluate(n=>n.complete&&n.naturalWidth>0),'static framework image actually loads');
    check(await page.locator('#axis-training').count()===1&&await page.locator('#axis-actions').count()===1,'complete learning text remains available');
    await nojs.close();
    const context=await browser.newContext();await offline(context);const instant=await context.newPage();
    await instant.route('**/sitemap.xml',async route=>{const response=await route.fetch();await route.fulfill({response,body:(await response.text()).replaceAll('https://dld0621.github.io/Embodied-AI-Zero-to-Hero',base)});});
    await instant.goto(`${base}/31-vla-concept-map/`,{waitUntil:'networkidle'});await ready(instant);
    await instant.evaluate(()=>window.__frameworkInstantMarker=761);
    await instant.locator('.md-content a[href*="32-vla-attention-lab"]').first().click();
    await instant.locator('[data-vla-attention] [data-va-output]').waitFor();
    check(await instant.evaluate(()=>window.__frameworkInstantMarker===761),'outbound navigation uses the real Material instant router');
    await instant.locator('.md-content a[href*="31-vla-concept-map"]').first().click();
    const root=await ready(instant);
    check(await root.locator('.vf-canvas svg').count()===1&&await instant.locator('[data-vla-framework]').count()===1,'instant return mounts one framework, without duplicate canvases');
    check(await root.locator('[data-vf-action="fit"]').count()===1,'instant return does not duplicate controls');
    await context.close();
    console.log(`VLA framework browser QA passed: ${checks} checks on Chromium ${browser.version()}; connected hierarchy, focus, pan/zoom/fit, relation edges, SVG export, text geometry, responsive themes, keyboard, no-JS and instant navigation.`);
    if(boundaries.length) console.log(`Explicit boundaries: ${[...new Set(boundaries)].join(' ')}`);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
