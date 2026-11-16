#!/usr/bin/env node
/* Built-site checks for the analytic teaching lab, not robot/model evaluation. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {chromium}=require('playwright');
const base=(process.argv[2]||'http://127.0.0.1:8766').replace(/\/$/,'');
const artifacts=process.env.IL_QA_ARTIFACTS||'';
let checks=0;
const check=(v,msg)=>{checks++;assert.ok(v,msg);};
async function range(locator,value){await locator.evaluate((el,v)=>{el.value=String(v);el.dispatchEvent(new Event('input',{bubbles:true}));},value);}
async function ready(page){await page.locator('[data-il-action-lab][data-il-ready="true"]').waitFor();return page.locator('[data-il-action-lab]');}
async function screenshot(root,name){if(artifacts){await fs.mkdir(artifacts,{recursive:true});await root.screenshot({path:path.join(artifacts,name+'.png'),animations:'disabled'});}}
async function layout(page,root,label){
 const state=await root.evaluate(node=>({
  overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,
  unreadable:[...node.querySelectorAll('svg text')].filter(n=>getComputedStyle(n).display!=='none').map(n=>{const m=n.getScreenCTM();return parseFloat(getComputedStyle(n).fontSize)*Math.hypot(m.a,m.b);}).filter(n=>n<11.5),
  replacement:node.textContent.includes('\ufffd'),
  chartTitles:[...node.querySelectorAll('svg')].every(n=>n.querySelector('title')&&n.querySelector('desc')),
  horizontal:[...node.querySelectorAll('svg')].map(n=>n.getBoundingClientRect()).every(r=>r.left>=0&&r.right<=document.documentElement.clientWidth+1),
 }));
 check(!state.overflow&&state.horizontal,label+': no horizontal overflow or clipped SVG frame');
 check(!state.unreadable.length,label+': SVG labels render at least 11.5px');
 check(!state.replacement&&state.chartTitles,label+': readable text and accessible SVG alternatives');
 const bad=await page.locator('.arithmatex').evaluateAll(nodes=>nodes.filter(n=>n.dataset.mathRendered!=='static-svg'||!n.querySelector('.math-assistive math')).length);
 check(bad===0,label+': formulas use self-contained rendered mathematics');
}
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.IL_CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/33-il-action-generation/',{waitUntil:'networkidle'});
  let root=await ready(page);await root.scrollIntoViewIfNeeded();
  check(await root.getAttribute('data-method')==='fm'&&await root.getAttribute('data-mode')==='generate','default separates FM generation from training');
  check(await root.locator('[data-ia-execute]').isDisabled(),'execution begins locked');
  check(await root.locator('.ia-internal svg').count()===2,'FM has action chunk plot and data-space vector field');
  check((await root.locator('[data-ia-numbers]').innerText()).includes('当前场预测 v'),'FM shows current model field output');
  const initialRobot=await root.locator('.ia-gripper').getAttribute('cx'),initialData=await root.locator('[data-ia-numbers]').innerText();
  await range(root.locator('[data-ia-progress]'),4);
  check((await root.locator('[data-ia-numbers]').innerText())!==initialData,'scrubbing actually updates the internal numerical state');
  check(await root.locator('.ia-gripper').getAttribute('cx')===initialRobot&&await root.getAttribute('data-executed')==='0','internal iteration leaves execution unmoved');
  check(await root.locator('[data-ia-execute]').isDisabled(),'partial generation cannot execute');
  await range(root.locator('[data-ia-progress]'),16);
  check(await root.getAttribute('data-generation-complete')==='true'&&!(await root.locator('[data-ia-execute]').isDisabled()),'completed chunk unlocks separate execution');
  const finishedData=await root.locator('[data-ia-numbers]').innerText();
  await root.locator('[data-ia-execute]').click();
  check(await root.getAttribute('data-executed')==='1'&&await root.locator('.ia-gripper').getAttribute('cx')!==initialRobot,'execution moves only after explicit execute action');
  check(await root.locator('[data-ia-numbers]').innerText()===finishedData,'execution does not perform another generation step');
  await range(root.locator('[data-ia-execution]'),3);
  check((await root.locator('[data-ia-execution-status]').innerText()).includes('0.15 秒'),'execution timeline uses 20Hz physical-time convention');
  check(await root.locator('[data-ia-execute]').isDisabled()&&(await root.locator('[data-ia-execute]').innerText()).includes('前缀已执行完'),'completed prefix disables further execution and names the completed state');
  await root.locator('[data-ia-prefix]').selectOption('2');
  check(await root.getAttribute('data-executed')==='0'&&await root.locator('[data-ia-execution]').getAttribute('max')==='2','changing K resets only the execution prefix');
  await root.locator('[data-ia-reset]').click();
  check(await root.getAttribute('data-position')==='0'&&await root.locator('[data-ia-execute]').isDisabled(),'reset locks the execution gate again');
  const seedBefore=await root.locator('[data-ia-numbers]').innerText();
  await root.locator('[data-ia-seed]').fill('42');await root.locator('[data-ia-seed]').dispatchEvent('change');
  check(await root.locator('[data-ia-numbers]').innerText()!==seedBefore,'seed changes the stochastic initial state');
  await range(root.locator('[data-ia-progress]'),8);const left=await root.locator('[data-ia-numbers]').innerText();
  await root.locator('[data-ia-condition]').selectOption('right');await range(root.locator('[data-ia-progress]'),8);
  check(await root.locator('[data-ia-numbers]').innerText()!==left,'observation condition changes the numerical generation field');
  await root.locator('[data-ia-steps]').selectOption('4');await range(root.locator('[data-ia-progress]'),4);
  const coarse=await root.locator('[data-ia-numeric-note]').innerText();
  await root.locator('[data-ia-steps]').selectOption('64');await range(root.locator('[data-ia-progress]'),64);
  check(await root.locator('[data-ia-numeric-note]').innerText()!==coarse,'solver step count changes the actual Euler numerical error');
  await root.locator('[data-ia-mode="train"]').click();
  check(await root.locator('[data-ia-execute]').isDisabled()&&(await root.locator('[data-ia-numbers]').innerText()).includes('监督目标'),'training labels appear only in the training illustration');
  await range(root.locator('[data-ia-progress]'),32);
  check((await root.locator('[data-ia-operation]').innerText()).includes('A*−ε'),'FM training identifies pair velocity supervision');
  check((await root.locator('[data-ia-field]').innerText()).includes('训练目标'),'training vector field distinguishes the sample target');
  await root.locator('[data-ia-method="dp"]').click();
  check((await root.locator('[data-ia-numbers]').innerText()).includes('当前噪声预测 ε'),'DP exposes epsilon prediction rather than calling it velocity');
  await range(root.locator('[data-ia-progress]'),32);
  check((await root.locator('[data-ia-operation]').innerText()).includes('加噪'),'DP training shows forward noising');
  await root.locator('[data-ia-mode="generate"]').click();
  check(!(await root.locator('[data-ia-numbers]').innerText()).includes('监督目标'),'DP generation does not expose an expert action label as input');
  await range(root.locator('[data-ia-progress]'),64);
  check(await root.getAttribute('data-generation-complete')==='true'&&(await root.locator('[data-ia-operation]').innerText()).includes('t=0'),'DDPM generation finishes at noise index zero');
  await root.locator('[data-ia-method="act"]').click();
  check(await root.locator('[data-ia-steps]').isDisabled()&&await root.locator('[data-ia-seed]').isDisabled(),'ACT inference is one deterministic prior-mean illustration');
  check(await root.locator('[data-ia-progress]').getAttribute('max')==='1','ACT does not invent an iterative noise sampler');
  await root.locator('[data-ia-step]').click();
  check(await root.getAttribute('data-generation-complete')==='true','one ACT forward prediction produces the action chunk');
  check((await root.locator('[data-ia-overlap]').innerText()).includes('同一个执行时刻'),'ACT temporal ensemble combines overlapping predictions for the same action time');
  await root.locator('[data-ia-mode="train"]').click();
  check((await root.locator('[data-ia-operation]').innerText()).includes('后验潜变量')&&await root.locator('[data-ia-execute]').isDisabled(),'ACT training explains posterior and keeps execution locked');
  await root.locator('[data-ia-method="fm"]').click();await root.locator('[data-ia-mode="generate"]').click();await root.locator('[data-ia-steps]').selectOption('16');
  await root.locator('[data-ia-play]').click();await page.waitForFunction(()=>Number(document.querySelector('[data-il-action-lab]').dataset.position)>0);
  await root.locator('[data-ia-play]').click();const paused=await root.getAttribute('data-position');await page.waitForTimeout(600);
  check(await root.getAttribute('data-position')===paused,'pause stops numerical progression');
  await root.locator('[data-ia-reset]').click();
  await page.evaluate(()=>{window.__oldIlRoot=document.querySelector('[data-il-action-lab]');});
  await root.locator('[data-ia-play]').click();
  const lessonLink=page.locator('.md-content a[href*="learning-core/06-generative-policies"]').first();
  await lessonLink.click();await page.waitForURL('**/learning-core/06-generative-policies/');
  const detached=await page.evaluate(()=>({connected:window.__oldIlRoot?.isConnected,position:window.__oldIlRoot?.dataset.position}));
  await page.waitForTimeout(650);
  check(!detached.connected&&await page.evaluate(()=>window.__oldIlRoot?.dataset.position)===detached.position,'instant navigation removes the old lab and stops its animation');
  await page.goBack();root=await ready(page);
  check(await root.locator('[data-ia-method]').count()===3,'instant navigation back mounts exactly one control set');
  await root.locator('[data-ia-step]').click();
  check(await root.getAttribute('data-position')==='1','returning page has one listener per action');
  check(errors.length===0,'interactive navigation and all controls have no page errors');
  await context.close();
  for(const width of [320,768,1440])for(const theme of ['default','slate']) {
    const c=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'}),p=await c.newPage();
    await p.goto(base+'/33-il-action-generation/',{waitUntil:'networkidle'});
    const r=await ready(p);await p.evaluate(value=>{document.body.dataset.mdColorScheme=value;document.documentElement.style.scrollBehavior='auto';},theme);
    await r.scrollIntoViewIfNeeded();await range(r.locator('[data-ia-progress]'),8);
    await layout(p,r,`${width}/${theme}/FM`);
    await screenshot(r,`il-${width}-${theme}-fm`);
    check(await r.locator('[data-ia-play]').isDisabled(),'reduced-motion preference leaves explicit step and scrub controls');
    await r.locator('[data-ia-method="dp"]').click();await r.locator('[data-ia-mode="train"]').click();await range(r.locator('[data-ia-progress]'),8);
    await layout(p,r,`${width}/${theme}/DP-training`);
    await screenshot(r,`il-${width}-${theme}-dp-training`);
    await c.close();
  }
  const fallback=await browser.newContext({javaScriptEnabled:false,viewport:{width:320,height:1000}}),f=await fallback.newPage();
  await f.goto(base+'/33-il-action-generation/',{waitUntil:'networkidle'});
  check(await f.locator('[data-il-action-lab]').count()===1&&!(await f.locator('[data-il-action-lab]').getAttribute('data-il-ready')),'no-JS fallback remains present');
  check((await f.locator('.md-content').innerText()).includes('Flow Matching'),'lesson text remains available without scripts');
  const formulas=await f.locator('.arithmatex').count();
  check(formulas>5&&await f.locator('.arithmatex[data-math-rendered="static-svg"] svg').count()===formulas,'all equations remain readable with JavaScript disabled');
  check(await f.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+1),'static fallback is readable at 320px');
  await fallback.close();
  console.log(JSON.stringify({checks,base,result:'passed',artifacts},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
