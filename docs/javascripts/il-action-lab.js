/* Analytic action-generation teaching lab. No trained checkpoint or robot connection. */
(function (scope) {
  'use strict';
  const H = 6, SIGMA = .18, TRAIN_STEPS = 64, END_ALPHA = 1e-4;
  const METHODS = ['act', 'dp', 'fm'];
  function validChunk(value) {
    if (!Array.isArray(value) || value.length !== H || value.some(row => !Array.isArray(row) || row.length !== 2 || row.some(v => !Number.isFinite(v)))) throw new RangeError('Expected a finite 6 × 2 action chunk');
    return value;
  }
  function random(seed) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 999999) throw new RangeError('Seed must be an integer from 0 to 999999');
    let s = seed >>> 0;
    return () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return (s + .5) / 4294967296; };
  }
  function normal(rng) { return Math.sqrt(-2 * Math.log(rng())) * Math.cos(2 * Math.PI * rng()); }
  function noise(seed) { const rng = random(seed); return Array.from({length:H}, () => [normal(rng), normal(rng)]); }
  function mean(condition) {
    if (!['left', 'right'].includes(condition)) throw new RangeError('Unknown observation condition');
    const side = condition === 'left' ? -1 : 1;
    return [[.1,-.7],[.3,-.4],[.55,-.1],[.62,.2],[.62,.45],[.62,.7]].map(([x,y]) => [side*x,y]);
  }
  function combine(a,b,fa,fb) { return a.map((row,i) => row.map((v,j) => fa*v + fb*b[i][j])); }
  function demonstration(condition,seed) { return combine(mean(condition),noise((seed+7919)%1000000),1,SIGMA); }
  function alphaBar(index) {
    if (!Number.isFinite(index) || index < 0 || index > TRAIN_STEPS) throw new RangeError('Noise index outside [0,64]');
    return Math.exp(Math.log(END_ALPHA) * index / TRAIN_STEPS);
  }
  function flowVelocity(chunk,tau,condition) {
    validChunk(chunk);
    if (!Number.isFinite(tau) || tau < 0 || tau > 1) throw new RangeError('Flow progress outside [0,1]');
    const mu=mean(condition), variance=(1-tau)**2 + tau*tau*SIGMA*SIGMA;
    const coefficient=(tau*SIGMA*SIGMA-(1-tau))/variance;
    return chunk.map((row,i)=>row.map((x,j)=>mu[i][j]+coefficient*(x-tau*mu[i][j])));
  }
  function exactFlow(epsilon,tau,condition) {
    validChunk(epsilon);
    if (!Number.isFinite(tau) || tau < 0 || tau > 1) throw new RangeError('Flow progress outside [0,1]');
    return combine(mean(condition),epsilon,tau,Math.sqrt((1-tau)**2+tau*tau*SIGMA*SIGMA));
  }
  function epsilonPrediction(chunk,index,condition) {
    validChunk(chunk);
    const ab=alphaBar(index), variance=ab*SIGMA*SIGMA+1-ab, mu=mean(condition);
    return chunk.map((row,i)=>row.map((x,j)=>Math.sqrt(1-ab)/variance*(x-Math.sqrt(ab)*mu[i][j])));
  }
  function ddpmStep(chunk,index,previous,condition,rng) {
    if (!(previous >= 0 && previous < index && index <= TRAIN_STEPS)) throw new RangeError('Expected decreasing DDPM noise indices');
    validChunk(chunk);
    const ab=alphaBar(index), ap=alphaBar(previous), a=ab/ap, beta=1-a;
    const epsilon=epsilonPrediction(chunk,index,condition), variance=beta*(1-ap)/(1-ab);
    const next=chunk.map((row,i)=>row.map((x,j)=>(x-beta/Math.sqrt(1-ab)*epsilon[i][j])/Math.sqrt(a)+(previous ? Math.sqrt(variance)*normal(rng) : 0)));
    return {chunk:next,epsilon,variance,alpha:a};
  }
  function distance(a,b) { validChunk(a);validChunk(b);return Math.sqrt(a.flatMap((r,i)=>r.map((v,j)=>(v-b[i][j])**2)).reduce((s,x)=>s+x,0)/(H*2)); }
  function trace(method,condition,seed,steps) {
    if (!METHODS.includes(method)) throw new RangeError('Unknown action generator');
    if (![4,8,16,32,64].includes(steps)) throw new RangeError('Expected 4, 8, 16, 32 or 64 sampling steps');
    const mu=mean(condition), epsilon=noise(seed);
    if (method==='act') return [{chunk:null,progress:0},{chunk:mu,progress:1}];
    const frames=[{chunk:epsilon,progress:0,tau:0,noiseIndex:TRAIN_STEPS}];
    let chunk=epsilon;
    const rng=random((seed+104729)%1000000);
    for(let i=0;i<steps;i++) {
      if(method==='fm') {
        const tau=i/steps,velocity=flowVelocity(chunk,tau,condition);
        chunk=combine(chunk,velocity,1,1/steps);
        frames.push({chunk,progress:(i+1)/steps,tau:(i+1)/steps,velocity});
      } else {
        const index=TRAIN_STEPS*(steps-i)/steps,previous=TRAIN_STEPS*(steps-i-1)/steps;
        const result=ddpmStep(chunk,index,previous,condition,rng);
        chunk=result.chunk;
        frames.push({...result,progress:(i+1)/steps,noiseIndex:previous});
      }
    }
    return frames;
  }
  function training(method,condition,seed,progress) {
    if(!METHODS.includes(method) || !Number.isFinite(progress) || progress<0 || progress>1) throw new RangeError('Invalid training illustration');
    const expert=demonstration(condition,seed),epsilon=noise(seed),mu=mean(condition);
    if(method==='act') return {chunk:mu,expert,target:expert,progress};
    if(method==='fm') return {chunk:combine(epsilon,expert,1-progress,progress),expert,epsilon,target:combine(expert,epsilon,1,-1),progress};
    const index=progress*TRAIN_STEPS,ab=alphaBar(index);
    return {chunk:combine(expert,epsilon,Math.sqrt(ab),Math.sqrt(1-ab)),expert,epsilon,target:epsilon,noiseIndex:index,progress};
  }
  function ensemble(values,decay=.4) {
    if(!Array.isArray(values) || !values.length || values.some(v=>!Number.isFinite(v)) || !Number.isFinite(decay) || decay<0) throw new RangeError('Invalid overlap estimates');
    const weights=values.map((_,i)=>Math.exp(-decay*i)),total=weights.reduce((a,b)=>a+b,0);
    return {weights:weights.map(w=>w/total),value:values.reduce((s,v,i)=>s+v*weights[i],0)/total};
  }
  const api={H,SIGMA,TRAIN_STEPS,END_ALPHA,noise,mean,demonstration,alphaBar,flowVelocity,exactFlow,epsilonPrediction,ddpmStep,distance,trace,training,ensemble};
  if(typeof module==='object' && module.exports) module.exports=api;
  scope.IlActionLab=api;
  if(!scope.document) return;
  const document=scope.document, controllers=new Set();
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=x=>Number(x).toFixed(3), names={act:'ACT',dp:'Diffusion Policy',fm:'Flow Matching'};
  function mount(root) {
    if(root.dataset.ilReady) return;
    root.dataset.ilReady='true';
    root.innerHTML=`<div class="ia-intro"><p>动作生成动态实验 · 解析教学替身</p><h3>先在电脑里生成，再交给执行端</h3><p>同一个 6 × 2 动作块：6 个未来时刻，每个时刻包含归一化 x、y 位置。本实验没有训练模型，也没有连接机器人。</p></div>
      <div class="ia-hierarchy" aria-label="模仿学习动作生成知识框架"><span>模仿学习 IL<br><small>从示范学行为</small></span><b aria-hidden="true">→</b><span>示范监督／BC<br><small>怎样提供训练目标</small></span><b aria-hidden="true">→</b><span data-ia-category>动作生成方法<br><small>具体算法／生成机制</small></span></div>
      <div class="ia-methods" role="group" aria-label="动作生成方法">${METHODS.map(m=>`<button type="button" data-ia-method="${m}" aria-pressed="${m==='fm'}">${names[m]}</button>`).join('')}</div>
      <div class="ia-modes" role="group" aria-label="训练或生成"><button type="button" data-ia-mode="train" aria-pressed="false">训练：示范怎样变成监督？</button><button type="button" data-ia-mode="generate" aria-pressed="true">生成：没有未来示范怎么办？</button></div>
      <div class="ia-inputs"><label>当前观察条件<select data-ia-condition aria-label="当前观察条件"><option value="left">抓左侧杯子</option><option value="right">抓右侧杯子</option></select></label><label>随机种子<input data-ia-seed aria-label="随机种子" type="number" min="0" max="999999" step="1" value="7"></label><label>生成内部步数 S<select data-ia-steps aria-label="生成内部步数"><option>4</option><option>8</option><option selected>16</option><option>32</option><option>64</option></select></label><label>执行前缀 K<select data-ia-prefix aria-label="执行前缀"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option><option>6</option></select></label></div>
      <p class="ia-observation" data-ia-observation></p><p class="ia-boundary" data-ia-boundary></p>
      <section class="ia-internal" aria-label="模型内部生成"><h4>① 计算机内部：动作数值空间</h4><p data-ia-operation></p><div data-ia-data-plot></div><div class="ia-legend"><span><i class="ia-x"></i>x 分量</span><span><i class="ia-y"></i>y 分量</span><span><i class="ia-reference"></i><span data-ia-reference-label>条件分布中心</span></span></div>
      <div data-ia-field hidden></div><div class="ia-playback"><button type="button" data-ia-play>播放生成</button><button type="button" data-ia-step>单步</button><button type="button" data-ia-reset>重置</button></div><label class="ia-range-label"><span data-ia-progress-label>生成进度</span><output data-ia-progress-value></output><input data-ia-progress aria-label="内部过程进度" type="range" min="0" max="16" step="1" value="0"></label>
      <p role="status" aria-live="polite" data-ia-status></p><div class="ia-numbers" data-ia-numbers></div><div data-ia-overlap hidden></div><p class="ia-boundary" data-ia-numeric-note></p></section>
      <section class="ia-execution" aria-label="生成之后的执行"><h4>② 执行端：机器人执行时间</h4><div data-ia-robot></div><p data-ia-execution-status></p><div class="ia-playback"><button type="button" data-ia-execute disabled>执行下一步（最多 K 步）</button><button type="button" data-ia-clear-execution disabled>执行归零</button></div><label class="ia-range-label">执行到第几个动作<output data-ia-execution-value>0</output><input data-ia-execution aria-label="机器人执行进度" type="range" min="0" max="3" step="1" value="0" disabled></label><p class="ia-boundary">教学执行频率设为 20 Hz；动画刻意慢放。生成进度 τ／噪声索引 t 不是真实执行秒数。中间噪声只存在于上方数值空间，不会被发送给执行端。这里只展示一次动作块，未模拟真实相机反馈或安全控制。</p></section>
      <details class="ia-math"><summary>查看本实验的数学替身与限制</summary><p>观察条件确定高斯动作分布的中心 μ(c)，标准差固定 σ=0.18；随机初值 ε 与单条训练示范 A* 分开生成。这里的 μ 是解析分布参数，不是推理时偷看的未来示范。</p><p>FM：训练桥 Aτ=(1−τ)ε+τA*，目标 A*−ε。生成使用条件平均场 v=μ+[τσ²−(1−τ)]/D·(A−τμ)，D=(1−τ)²+τ²σ²。Euler 逐步积分，独立精确解 τμ+√D ε 仅用于核对数值误差。</p><p>DP：固定训练噪声表 ᾱ(t)=exp(log(0.0001)·t/64)，解析噪声预测器 E[ε|Aₜ,c]=√(1−ᾱ)/(ᾱσ²+1−ᾱ)·(Aₜ−√ᾱ μ)。按所选节点合并前向步，采用匹配的 DDPM 式后验均值与方差反向采样。有限终点使 N(0,I) 只是近似先验；固定后验方差也不保证末态精确服从目标高斯。少步结果不用于真实 DP 性能排名。</p><p>ACT：本例直接给出条件均值动作块，只演示一次输出与时间集成。没有训练 CVAE、Transformer 或真实策略；真实 ACT 还包括训练后验、KL 与推理 z=0 等设计。</p></details>`;
    let method='fm',mode='generate',condition='left',seed=7,steps=16,prefix=3,position=0,executed=0,frames=trace(method,condition,seed,steps),playing=false,raf=0,last=0;
    const reduced=scope.matchMedia && scope.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const el=s=>root.querySelector(s);
    function pause() {playing=false; if(raf)scope.cancelAnimationFrame(raf);raf=0;el('[data-ia-play]').textContent=mode==='train'?'播放训练路径':'播放生成';}
    const controller={root,pause};controllers.add(controller);
    function current() {return mode==='train'?training(method,condition,seed,position/(method==='act'?1:steps)):frames[position];}
    function ready() {return mode==='generate' && position===frames.length-1;}
    function changed() {pause();position=0;executed=0;frames=trace(method,condition,seed,steps);render();}
    function plot(state) {
      const reference=mode==='train'?state.expert:mean(condition), chunk=state.chunk;
      const all=[...(chunk||[]).flat(),...reference.flat(),...noise(seed).flat()], range=Math.max(1.3,...all.map(Math.abs))+.25;
      const px=i=>45+i*58, py=v=>112-v/range*77;
      const line=(values,d)=>values.map((r,i)=>`${px(i)},${py(r[d])}`).join(' ');
      const text=chunk?`当前6个动作的x和y分量。第1个动作为${fmt(chunk[0][0])},${fmt(chunk[0][1])}。`:'等待一次前向预测，当前没有可执行动作块。';
      let svg=`<svg viewBox="0 0 380 225" role="img" aria-label="动作数值随动作位置的分布"><title>动作块的两个位置分量</title><desc>${escape(text)}</desc><line x1="40" y1="112" x2="345" y2="112" class="ia-axis"/><text x="5" y="24" class="ia-svg-secondary">数值</text><text x="286" y="219" class="ia-svg-secondary">动作序号 h</text>`;
      for(const v of [-range,0,range]) svg+=`<text x="2" y="${py(v)+4}" class="ia-tick">${v.toFixed(1)}</text>`;
      for(let i=0;i<H;i++)svg+=`<text x="${px(i)}" y="204" text-anchor="middle">${i+1}</text>`;
      for(let d=0;d<2;d++) {
        svg+=`<polyline points="${line(reference,d)}" class="ia-reference-line"/>`;
        if(chunk)svg+=`<polyline points="${line(chunk,d)}" class="ia-line ia-line-${d}"/>${chunk.map((r,i)=>`<circle cx="${px(i)}" cy="${py(r[d])}" r="4" class="ia-dot-${d}"/>`).join('')}`;
      }
      svg+='</svg>';
      el('[data-ia-data-plot]').innerHTML=svg;
      el('[data-ia-reference-label]').textContent=mode==='train'?'训练示范（此时可见）':'条件分布中心（非未来示范）';
      const field=el('[data-ia-field]');field.hidden=method!=='fm';
      if(method==='fm') {
        const tau=mode==='train'?position/steps:state.progress,point=chunk[0],radius=Math.max(1.5,...point.map(Math.abs))+.2;
        const sx=x=>190+x/radius*120,sy=y=>120-y/radius*73;
        const arrow=(x,y,vx,vy,color)=>{const magnitude=Math.hypot(vx,vy),factor=.22/Math.max(.35,magnitude),dx=vx*factor,dy=vy*factor;
          const x1=sx(x),y1=sy(y),x2=sx(x+dx),y2=sy(y+dy),angle=Math.atan2(y2-y1,x2-x1),size=4;
          return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${color}"/><polygon points="${x2},${y2} ${x2-size*Math.cos(angle-.55)},${y2-size*Math.sin(angle-.55)} ${x2-size*Math.cos(angle+.55)},${y2-size*Math.sin(angle+.55)}" class="${color}"/>`;};
        let arrows='';
        for(const x of [-1,-.5,0,.5,1])for(const y of [-1,-.5,0,.5,1]){const probe=mean(condition);probe[0]=[x,y];const v=flowVelocity(probe,tau,condition)[0];arrows+=arrow(x,y,v[0],v[1],'ia-field-arrow');}
        const v=mode==='train'?state.target[0]:flowVelocity(chunk,tau,condition)[0];
        field.innerHTML=`<p class="ia-field-caption">动作一的二维数值空间 · τ=${tau.toFixed(2)}</p><svg viewBox="0 0 380 230" role="img" aria-label="第一个动作的二维数据变化场，不是机器人速度"><title>动作一的二维数值空间</title><desc>进度${tau.toFixed(3)}。当前变化量${fmt(v[0])},${fmt(v[1])}。</desc><line x1="50" y1="120" x2="330" y2="120" class="ia-axis"/><line x1="190" y1="35" x2="190" y2="205" class="ia-axis"/>${arrows}${arrow(point[0],point[1],v[0],v[1],'ia-target-arrow')}<circle cx="${sx(point[0])}" cy="${sy(point[1])}" r="5" class="ia-dot-1"/><text x="291" y="217" class="ia-svg-secondary">x 数值</text><text x="199" y="39" class="ia-svg-secondary">y 数值</text></svg><p class="ia-boundary">蓝箭头：解析条件平均场。橙点：当前动作一；橙箭头：${mode==='train'?'这一对示范／噪声的训练目标':'当前场预测的变化方向'} [${v.map(fmt).join(', ')}]。箭头长度经过缩放以便阅读，数值见此处；不是机械臂运动速度。</p>`;
      }
    }
    function drawRobot() {
      const final=ready()?frames.at(-1).chunk:null, point=executed&&final?final[executed-1]:[0,-.9], cup=condition==='left'?-.62:.62;
      const px=x=>190+x*110,py=y=>164-y*85;
      const path=final?[[0,-.9],...final.slice(0,prefix)].map(([x,y])=>`${px(x)},${py(y)}`).join(' '):'';
      el('[data-ia-robot]').innerHTML=`<svg viewBox="0 0 380 270" role="img" aria-label="教学执行端，只有完整生成后的动作才可播放"><title>执行端与数值生成分开</title><desc>已执行${executed}个动作。生成中执行端保持在初始位置。</desc><line x1="30" y1="245" x2="350" y2="245" class="ia-axis"/><rect x="${px(cup)-17}" y="${py(.2)-20}" width="34" height="40" rx="4" class="ia-cup"/><text x="${px(cup)}" y="${py(.2)-28}" text-anchor="middle">目标杯子</text>${path?`<polyline points="${path}" class="ia-execution-path"/>`:''}<polyline points="190,245 190,210 ${px(point[0])},${py(point[1])}" class="ia-arm"/><circle cx="${px(point[0])}" cy="${py(point[1])}" r="7" class="ia-gripper"/></svg>`;
      el('[data-ia-execution]').max=String(prefix);el('[data-ia-execution]').value=String(executed);
      el('[data-ia-execution-value]').textContent=String(executed);
      el('[data-ia-execution-status]').textContent=ready()?`生成已完成。H=${H}、K=${prefix}，执行前缀覆盖 ${(prefix/20).toFixed(2)} 秒。当前已执行 ${(executed/20).toFixed(2)} 秒；未执行其余动作。`:'执行锁定：训练路径或未完成的内部数值不能作为机器人命令。';
      for(const selector of ['[data-ia-execute]','[data-ia-clear-execution]','[data-ia-execution]'])el(selector).disabled=!ready();
      el('[data-ia-execute]').disabled=!ready()||executed>=prefix;
      el('[data-ia-execute]').textContent=ready()&&executed>=prefix?'前缀已执行完':'执行下一步（最多 K 步）';
      root.dataset.executed=String(executed);root.dataset.generationComplete=String(ready());
    }
    function render() {
      const state=current(),count=method==='act'?1:steps,at=mode==='train'?position/count:state.progress;
      root.dataset.method=method;root.dataset.mode=mode;root.dataset.position=String(position);
      root.querySelectorAll('[data-ia-method]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.iaMethod===method)));
      root.querySelectorAll('[data-ia-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.iaMode===mode)));
      el('[data-ia-category]').innerHTML=method==='act'?'ACT<br><small>具体 CVAE 动作分块算法</small>':method==='dp'?'Diffusion Policy<br><small>扩散动作策略</small>':'Flow Matching<br><small>可用于动作头的训练方法</small>';
      el('[data-ia-steps]').disabled=method==='act';
      el('[data-ia-seed]').disabled=method==='act'&&mode==='generate';
      el('[data-ia-observation]').textContent=`观察 c：${condition==='left'?'左':'右'}侧杯子；机器人初始归一化位置 [0, −0.9]。本例观察只改变解析分布中心，不加载相机或模型。`;
      el('[data-ia-boundary]').textContent=method==='act'?'ACT 只进行一次动作块输出；下方是确定性解析替身。分块也可用于 DP、FM 等方法。':method==='dp'?'扩散用解析噪声预测器和匹配的 DDPM 式更新，不是每次直接减去预测噪声。S 是内部采样步数。':'FM 学习动作数据空间的变化场。这里用解析条件平均场代替网络；箭头不代表真实机械臂速度。';
      el('[data-ia-progress]').max=String(count);el('[data-ia-progress]').value=String(position);
      el('[data-ia-progress-value]').textContent=method==='act'?`${position}/1 次预测`:`${position}/${count} · ${(at*100).toFixed(0)}%`;
      el('[data-ia-progress-label]').textContent=mode==='train'?(method==='dp'?'训练加噪进度 t：干净 → 噪声':'训练路径 τ：噪声 → 示范'):'生成内部进度：开始 → 完成';
      el('[data-ia-operation]').textContent=mode==='train'?(method==='act'?'观察＋训练示范 → 后验潜变量（真实ACT）→ 解码动作块 → 重建与KL监督；本例只展示监督动作。':method==='dp'?'真实示范 A* 加噪 → 条件噪声预测 → 与已知噪声 ε 比较。这里只查看训练样本，不更新参数。':'构造 ε 与 A* 的中间状态 → 回归 A*−ε。单条路径目标与条件平均场不必相同。'):(method==='act'?'当前观察 c → 一次动作块预测（真实原版推理 z=0）→ 完整6步动作。没有输入未来专家动作。':method==='dp'?`随机初值 → 噪声预测 → 合并噪声步后的 DDPM 式采样。当前噪声索引 t=${state.noiseIndex}。`:`随机初值 → 查询条件变化场 → Euler 更新。当前生成进度 τ=${at.toFixed(3)}，步长 Δτ=${(1/steps).toFixed(3)}。`);
      plot(state);
      const target=mode==='train'?state.target:null,chunk=state.chunk;
      const prediction=method==='fm'?flowVelocity(chunk,at,condition):method==='dp'?epsilonPrediction(chunk,state.noiseIndex,condition):null;
      const predictionName=method==='fm'?'当前场预测 v':'当前噪声预测 ε';
      el('[data-ia-numbers]').innerHTML=`<table><caption>${mode==='train'?(method==='fm'?'输入中间状态与回归目标（前3个动作）':method==='dp'?'带噪输入与噪声标签（前3个动作）':'动作预测与示范监督（前3个动作）'):'当前动作块数值（前3个动作；完整块共6步）'}</caption><thead><tr><th>动作</th><th>当前 x / y</th>${target?'<th>监督目标 x / y</th>':''}${prediction?`<th>${predictionName} x / y</th>`:''}</tr></thead><tbody>${[0,1,2].map(i=>`<tr><th>${i+1}</th><td>${chunk?chunk[i].map(fmt).join(' / '):'尚未预测'}</td>${target?`<td>${target[i].map(fmt).join(' / ')}</td>`:''}${prediction?`<td>${prediction[i].map(fmt).join(' / ')}</td>`:''}</tr>`).join('')}</tbody></table>`;
      let note='';
      if(mode==='generate' && method==='fm')note=`相同初始噪声下，与解析流解的逐元素 RMSE：${fmt(distance(chunk,exactFlow(noise(seed),at,condition)))}。这是 Euler 数值误差，不是机器人任务误差。`;
      else if(method==='dp')note='有限噪声表末端 ᾱ=0.0001，因此标准高斯起点是近似。合并步数改变采样近似；本例不声明输出分布精确或少步一定更好。';
      else if(mode==='train' && method==='fm')note='训练可见 A*，生成时不可见。推理使用的条件平均场由分布参数确定，不会读取这里这条示范。';
      else note='真实 ACT 的 CVAE 训练、视觉编码和参数更新没有在本解析示例中复现。';
      el('[data-ia-numeric-note]').textContent=note;
      const overlap=el('[data-ia-overlap]');overlap.hidden=method!=='act';
      if(method==='act') {const values=[.52,.58,.64],result=ensemble(values);overlap.innerHTML=`<p><strong>时间集成：同一个执行时刻</strong>，三次重叠预测给出 ${values.join('、')}；最旧→最新的权重为 ${result.weights.map(fmt).join('、')}，融合值 ${fmt(result.value)}。这不是把相邻三个执行时刻平均。原版 ACT 使用指数权重；此处是独立标量算例。</p>`;}
      el('[data-ia-status]').textContent=mode==='train'?`训练样本可见；内部进度${position}/${count}。执行端不会移动。`:ready()?`动作块生成完成，共 ${H} 个动作。现在可以单独执行前 ${prefix} 个。`:`内部计算 ${position}/${count}；执行端保持不动。`;
      el('[data-ia-step]').disabled=position===count;
      el('[data-ia-play]').disabled=position===count || reduced;
      if(reduced) el('[data-ia-play]').textContent='已减少动画：请单步或拖动';
      drawRobot();
      fitLabels();
    }
    function fitLabels() {
      root.querySelectorAll('svg').forEach(svg=>{
        const width=svg.getBoundingClientRect().width;
        if(width>0)svg.style.setProperty('--ia-svg-font',`${Math.max(12,12*380/width)}px`);
      });
    }
    if(scope.ResizeObserver) {const observer=new scope.ResizeObserver(fitLabels);observer.observe(root);controller.observer=observer;}
    function tick(now) {
      if(!playing || !root.isConnected || document.hidden) {pause();return;}
      if(now-last>440) {last=now;position++;executed=0;render();if(position===(method==='act'?1:steps)){pause();return;}}
      raf=scope.requestAnimationFrame(tick);
    }
    root.querySelectorAll('[data-ia-method]').forEach(b=>b.addEventListener('click',()=>{method=b.dataset.iaMethod;changed();}));
    root.querySelectorAll('[data-ia-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.iaMode;changed();}));
    el('[data-ia-condition]').addEventListener('change',e=>{condition=e.target.value;changed();});
    el('[data-ia-seed]').addEventListener('change',e=>{const v=Number(e.target.value);if(!Number.isInteger(v)||v<0||v>999999){e.target.value=String(seed);return;}seed=v;changed();});
    el('[data-ia-steps]').addEventListener('change',e=>{steps=Number(e.target.value);changed();});
    el('[data-ia-prefix]').addEventListener('change',e=>{prefix=Number(e.target.value);executed=0;render();});
    el('[data-ia-reset]').addEventListener('click',changed);
    el('[data-ia-step]').addEventListener('click',()=>{pause();position=Math.min(position+1,method==='act'?1:steps);executed=0;render();});
    el('[data-ia-progress]').addEventListener('input',e=>{pause();position=Number(e.target.value);executed=0;render();});
    el('[data-ia-play]').addEventListener('click',()=>{if(playing){pause();return;}playing=true;last=0;el('[data-ia-play]').textContent='暂停';raf=scope.requestAnimationFrame(tick);});
    el('[data-ia-execute]').addEventListener('click',()=>{if(ready()){executed=Math.min(executed+1,prefix);render();}});
    el('[data-ia-clear-execution]').addEventListener('click',()=>{executed=0;render();});
    el('[data-ia-execution]').addEventListener('input',e=>{if(ready()){executed=Number(e.target.value);render();}});
    render();
  }
  function initialize() {for(const c of controllers)if(!c.root.isConnected){c.pause();if(c.observer)c.observer.disconnect();controllers.delete(c);}document.querySelectorAll('[data-il-action-lab]').forEach(mount);}
  document.addEventListener('visibilitychange',()=>{if(document.hidden)controllers.forEach(c=>c.pause());});
  if(typeof document$!=='undefined') document$.subscribe(initialize);
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initialize,{once:true});else initialize();
})(typeof window!=='undefined'?window:globalThis);
