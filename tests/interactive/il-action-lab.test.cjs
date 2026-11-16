'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const lab=require('../../docs/javascripts/il-action-lab.js');
const close=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} differs from ${b}`);
const each=(a,f)=>a.forEach((r,i)=>r.forEach((v,j)=>f(v,i,j)));

test('six future xy actions preserve observation semantics and independent seeded noise',()=>{
  const left=lab.mean('left'),right=lab.mean('right');
  assert.equal(left.length,6);
  left.forEach(([x,y],i)=>{close(x,-right[i][0]);close(y,right[i][1]);});
  assert.deepEqual(lab.noise(7),lab.noise(7));
  assert.notDeepEqual(lab.noise(7),lab.noise(8));
  assert.notDeepEqual(lab.noise(7),lab.demonstration('left',7));
});
test('flow training bridge has correct endpoints and one constant pair target',()=>{
  const a=lab.training('fm','left',7,0),b=lab.training('fm','left',7,.5),c=lab.training('fm','left',7,1);
  assert.deepEqual(a.chunk,a.epsilon);assert.deepEqual(c.chunk,c.expert);assert.deepEqual(a.target,c.target);
  each(b.chunk,(v,i,j)=>{close(v,(a.epsilon[i][j]+c.expert[i][j])/2);close(a.target[i][j],c.expert[i][j]-a.epsilon[i][j]);});
});
test('conditional flow field agrees with the derivative of the independently known marginal flow',()=>{
  const eps=lab.noise(11),dt=1e-6;
  for(const tau of [.1,.5,.9]) {
    const x=lab.exactFlow(eps,tau,'right'),v=lab.flowVelocity(x,tau,'right');
    const plus=lab.exactFlow(eps,tau+dt,'right'),minus=lab.exactFlow(eps,tau-dt,'right');
    each(v,(value,i,j)=>close(value,(plus[i][j]-minus[i][j])/(2*dt),1e-8));
  }
});
test('flow generation integrates a field rather than jumping to a demonstration endpoint',()=>{
  const eps=lab.noise(7),original=JSON.stringify(eps),coarse=lab.trace('fm','left',7,4),fine=lab.trace('fm','left',7,64);
  assert.equal(coarse.length,5);assert.equal(fine.length,65);
  const exact=lab.exactFlow(eps,1,'left');
  assert.ok(lab.distance(fine.at(-1).chunk,exact)<lab.distance(coarse.at(-1).chunk,exact));
  assert.ok(lab.distance(coarse.at(-1).chunk,exact)>.001);
  assert.notDeepEqual(fine.at(-1).chunk,lab.demonstration('left',7));
  assert.equal(JSON.stringify(eps),original);
});
test('diffusion training starts at clean data and uses a fixed finite noise schedule',()=>{
  const clean=lab.training('dp','right',9,0),noisy=lab.training('dp','right',9,1);
  assert.deepEqual(clean.chunk,clean.expert);close(lab.alphaBar(0),1);close(lab.alphaBar(64),1e-4);
  each(noisy.chunk,(v,i,j)=>close(v,.01*noisy.expert[i][j]+Math.sqrt(1-1e-4)*noisy.epsilon[i][j]));
  assert.deepEqual(clean.target,noisy.target);
});
test('conditional epsilon prediction is zero at the noised conditional mean',()=>{
  for(const t of [1,16,64]) {
    const chunk=lab.mean('left').map(r=>r.map(x=>Math.sqrt(lab.alphaBar(t))*x));
    each(lab.epsilonPrediction(chunk,t,'left'),v=>close(v,0));
  }
});
test('last DDPM update equals Gaussian posterior mean and adds no final noise',()=>{
  const t=32,chunk=lab.noise(18),ab=lab.alphaBar(t),variance=ab*lab.SIGMA**2+1-ab,mu=lab.mean('right');
  const result=lab.ddpmStep(chunk,t,0,'right',()=>{throw Error('last update must not request fresh noise');});
  close(result.variance,0);
  each(result.chunk,(v,i,j)=>close(v,mu[i][j]+Math.sqrt(ab)*lab.SIGMA**2/variance*(chunk[i][j]-Math.sqrt(ab)*mu[i][j])));
});
test('diffusion samples all decreasing merged nodes reproducibly without claiming exact target variance',()=>{
  for(const steps of [4,8,16,32,64]) {
    const frames=lab.trace('dp','left',7,steps);
    assert.equal(frames.length,steps+1);assert.equal(frames[0].noiseIndex,64);assert.equal(frames.at(-1).noiseIndex,0);
    frames.forEach((f,i)=>{each(f.chunk,v=>assert.ok(Number.isFinite(v)));if(i)assert.ok(f.noiseIndex<frames[i-1].noiseIndex);});
    assert.deepEqual(frames,lab.trace('dp','left',7,steps));
  }
  assert.notDeepEqual(lab.trace('dp','left',7,16).at(-1).chunk,lab.trace('dp','left',8,16).at(-1).chunk);
});
test('ACT illustration is one deterministic conditional chunk with no sampled inference latent',()=>{
  const a=lab.trace('act','left',7,16),b=lab.trace('act','left',123,4);
  assert.equal(a.length,2);assert.equal(a[0].chunk,null);assert.deepEqual(a[1].chunk,lab.mean('left'));assert.deepEqual(a,b);
});
test('temporal ensemble aggregates estimates for one time and preserves convex-hull bounds',()=>{
  const r=lab.ensemble([.52,.58,.64]);
  close(r.weights.reduce((a,b)=>a+b,0),1);assert.ok(r.weights[0]>r.weights[1]&&r.weights[1]>r.weights[2]);
  assert.ok(r.value>.52&&r.value<.64);close(lab.ensemble([.5,.5,.5]).value,.5);close(lab.ensemble([.52,.58,.64],0).value,.58);
});
test('invalid shapes, progress, seeds and schedules fail explicitly',()=>{
  for(const value of [[],[[1,2]],Array.from({length:6},()=>[NaN,0])])assert.throws(()=>lab.flowVelocity(value,.5,'left'),RangeError);
  for(const seed of [-1,1.5,1000000,NaN])assert.throws(()=>lab.noise(seed),RangeError);
  assert.throws(()=>lab.mean('missing'),RangeError);assert.throws(()=>lab.trace('fm','left',7,5),RangeError);
  assert.throws(()=>lab.training('fm','left',7,2),RangeError);assert.throws(()=>lab.flowVelocity(lab.mean('left'),-.1,'left'),RangeError);
  assert.throws(()=>lab.ddpmStep(lab.mean('left'),8,16,'left',Math.random),RangeError);
});
