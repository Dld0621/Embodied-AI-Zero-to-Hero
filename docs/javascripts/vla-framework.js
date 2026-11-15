/* A local, connected knowledge framework. Lines group knowledge, not a universal model architecture. */
(function (scope) {
  "use strict";
  const group = (id, label, examples, leaves, relation = "学习分组") => ({id, label, examples, relation, leaves});
  const branches = [
    {id:"role",title:"模型类别与职责",anchor:"axis-role",color:"#2563eb",groups:[
      group("scope","技术范围","AI ⊃ ML ⊃ DL",["AI：人工智能（包含 ML）","ML：机器学习（包含 DL）","DL：深度学习"]),
      group("models","模型类别","LLM · VLM · VLA",["LLM：语言模型","VLM：图文模型","VLA：视觉语言动作模型"]),
      group("roles","系统职责","Policy · Agent · World Model",["Policy：动作规则或分布","Agent：观察、决策与执行系统","World Model：预测环境变化"]),
      group("foundation","迁移用途","Foundation Model",["基础模型：广泛数据预训练","下游任务适配与迁移"])]},
    {id:"modules",title:"系统功能模块",anchor:"axis-modules",color:"#7c3aed",groups:[
      group("encoding","输入编码","Encoder · Tokenizer",["视觉编码器 Vision Encoder","文本分词器 Tokenizer","状态编码 / Embedding"]),
      group("conditioning","条件处理","Projector · Fusion · Backbone",["Projector：特征投影","Fusion：多模态融合","Backbone：主要特征计算"]),
      group("heads","动作输出","Head · Action Expert",["Head：任务输出头","Action Expert：动作生成网络"]),
      group("controller","物理执行","Controller",["控制器：跟踪动作参考","反馈与执行器接口"])]},
    {id:"architecture",title:"神经网络结构",anchor:"axis-architecture",color:"#0f766e",groups:[
      group("basic","基础网络","MLP · CNN",["MLP：线性层与非线性","CNN：共享局部卷积"]),
      group("recurrent","循环时序","RNN → LSTM / GRU",["RNN：递推隐藏状态","LSTM / GRU：门控变体"],"结构家族"),
      group("transformer","Transformer 组织","Encoder · Decoder",["Encoder","Decoder","Encoder–Decoder"]),
      group("vision","视觉结构","ViT：图像 Patch 序列",["ViT：视觉 Transformer","原始 ViT 使用 Encoder","与组织方式不是互斥分类"])]},
    {id:"internals",title:"网络内部计算",anchor:"axis-internals",color:"#c2410c",groups:[
      group("representations","数据表示","Tensor · Token · Embedding",["Tensor / Shape：张量与形状","Patch / Token：序列单元","Embedding / Feature：向量表示"]),
      group("parameters","参数与激活","W、b ≠ X、Q/K/V、P",["W、b：模型参数","X、Q/K/V：当前特征","P：随输入计算的注意力权重"]),
      group("reading","信息读取","Attention · Self / Cross",["Query / Key / Value","Self-attention / Cross-attention","Multi-head：多组并行读取"]),
      group("block","典型 Transformer Block","Attention + FFN + Norm",["Attention","FFN / MLP","Norm 与 Residual 路径","常见组成；顺序依具体实现"],"典型组成"),
      group("constraints","约束与位置","Mask · RoPE",["Attention / Padding / Loss Mask","LayerNorm / RMSNorm","Residual ≠ 控制残差","Position Encoding / RoPE"])]},
    {id:"components",title:"具体模型与组件",anchor:"axis-components",color:"#be185d",groups:[
      group("vision","视觉组件","ResNet · DINOv2",["ResNet / EfficientNet","DINOv2：自监督 ViT"]),
      group("vl","图文组件","CLIP · SigLIP",["CLIP / SigLIP：图文表征","VLA 可使用其图像编码分支"]),
      group("pali","PaliGemma 组成","SigLIP + Gemma",["SigLIP 图像编码器","Gemma 语言主干"],"内含"),
      group("policies","策略与增强","OpenVLA · π₀.₅ · SLA",["RT-2 / OpenVLA","π₀ / π₀.₅","SLA：空间增强训练方案"]),
      group("geometry","三维感知","VGGT · OmniVGGT",["VGGT / OmniVGGT","教师输入 ≠ 学生部署输入"])]},
    {id:"data",title:"输入与训练数据",anchor:"axis-data",color:"#0369a1",groups:[
      group("modalities","输入模态","RGB · Language · State",["RGB / Depth / RGB-D","Point Cloud / Language","Proprioception / History","Force / Tactile"]),
      group("records","记录组织","Step → Episode → Dataset",["Step / Trajectory","Episode / Demonstration","Dataset 与版本元数据"]),
      group("processing","采集与处理","Teleop · Augmentation",["Teleoperation：遥操作示范","Augmentation：数据增强","Normalization：训练统计变换"]),
      group("provenance","时间与隔离","Timestamp · Split",["采集 / 接收 / 动作时间戳","Episode / 物体 / 场景划分","统计量只用训练集拟合"])]},
    {id:"actions",title:"动作表示与生成",anchor:"axis-actions",color:"#6d28d9",groups:[
      group("space","物理空间","Joint · End-effector",["关节位置 / 速度 / 力矩","末端位置 / 姿态","Gripper：夹爪通道"]),
      group("semantics","动作语义","Absolute · Delta · Units",["绝对目标 / 相对增量","参考坐标系 / 单位","旋转表示与夹爪极性"]),
      group("timing","动作时序","Chunk · H · K",["Action Chunk：未来动作块","H：预测长度","K：实际执行前缀"]),
      group("generation","生成方法","回归 · AR · Diffusion · Flow",["Regression：连续回归","Autoregressive：依次解码","Diffusion：条件去噪","Flow Matching：学习场；积分采样"])]},
    {id:"training",title:"训练与监督",anchor:"axis-training",color:"#a16207",groups:[
      group("stages","训练阶段","Pre / Continual / Post",["Pre-training：预训练","Continual pre-training","Post-training / Fine-tuning"]),
      group("signals","学习信号","BC · Self-supervision · RL",["Supervised / IL / BC","Self-supervised Learning","RL / Distillation"]),
      group("updates","参数更新范围","Full · Freeze · PEFT → LoRA",["Full / Freeze / Head-only","PEFT：参数高效适配","LoRA：选定权重的低秩更新"]),
      group("optimization","优化过程","Loss → Gradient → Optimizer",["Loss / 有效目标 Mask","Backprop / Gradient","Optimizer / Learning Rate","Batch / Microbatch / Step / Epoch"]),
      group("source","监督来源","Teacher 角色 ≠ 测量 GT",["测量 / 人工标注 / 仿真状态","教师预测 / 特征 / 伪标签","记录产生者与可用时刻"])]},
    {id:"execution",title:"软件与机器人执行",anchor:"axis-execution",color:"#047857",groups:[
      group("software","实现工具","NumPy · PyTorch · JAX",["NumPy：数组计算","PyTorch / JAX：模型与自动微分","openpi / LeRobot：项目工具"]),
      group("artifacts","配置与产物","Config · Checkpoint",["Config / Hyperparameter","Checkpoint：权重与可选训练状态"]),
      group("geometry","坐标与运动学","Frame · FK · IK",["Coordinate Frame / Calibration","FK：关节 → 末端","IK：末端 → 关节参考"]),
      group("control","控制与调度","SDK · 闭环 · Latency",["SDK / API / Controller","Closed Loop / Latency / Queue","策略生成频率 / 动作命令频率","电机伺服频率另行记录"])]},
    {id:"evaluation",title:"评测与泛化",anchor:"axis-evaluation",color:"#b91c1c",groups:[
      group("objects","评测对象","Dataset · Simulator · Benchmark",["Dataset：记录","Simulator：物理与渲染工具","Benchmark：任务和评分协议"]),
      group("protocol","实验协议","Split · Seed · Budget",["Train / Validation / Test","Seed / Budget / Intervention","初态、控制与终止协议"]),
      group("metrics","指标与统计","Success · Progress · 区间",["完整任务成功率与分母","阶段进度 / Score","不确定性与重复试验"]),
      group("generalization","泛化维度","ID · OOD · Zero / Few-shot",["物体 / 场景 / 任务 / 本体","ID / OOD / Zero-shot / Few-shot","Sim2Real：仿真到实物验证"]),
      group("comparison","贡献比较","Ablation · Baseline · SOTA",["Ablation：受控消融","Baseline：匹配条件的基线","SOTA：特定协议下的比较"])]}
  ];
  const byId = Object.fromEntries(branches.map(b => [b.id,b]));
  const esc = value => String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function textLines(value, limit = 24) {
    const lines=[];let line="",units=0;
    for(const char of String(value)){const cost=/[\x00-\x7f]/.test(char)?.55:1;if(units+cost>limit&&line){lines.push(line.trim());line="";units=0;}line+=char;units+=cost;}
    if(line)lines.push(line.trim());return lines;
  }
  function layout(mode="overview",options={}) {
    if(mode!=="overview"&&mode!=="full"&&mode!=="relations"&&!Object.hasOwn(byId,mode))throw new RangeError("Unknown framework view");
    const nodes=[],edges=[];
    function add(id,label,x,y,side,kind,color,branch,extra={}) {const n={id,label,x,y,side,kind,color,branch,...extra};nodes.push(n);return n;}
    function connect(a,b,relation="学习分组",extra={}) {edges.push({from:a.id,to:b.id,relation,color:b.color,...extra});}
    if(mode==="relations"){
      const root=add("relation-siglip","SigLIP 图像分支",900,500,"center","root","#2563eb","components",{sub:"同一组件的三种知识标签"});
      const specs=[["modules","视觉编码器",380,230,"担任","按系统功能"],["architecture","ViT",1420,230,"采用","按内部结构"],["training","预训练组件",900,790,"用作初始化","按权重来源"]];
      for(const [id,label,x,y,relation,sub]of specs){const n=add(`relation-${id}`,label,x,y,"center","relation","#2563eb",id,{sub});connect(root,n,relation,{dashed:true});}
      return {mode,width:1800,height:1030,nodes,edges};
    }
    if(Object.hasOwn(byId,mode)){
      if(options.compact){
        const b=byId[mode],root=add(`b-${b.id}`,b.title,280,80,"center","root",b.color,b.id,{sub:"当前分支 · 向下拖动继续阅读",compact:true});let cursor=210;
        for(const g of b.groups){
          const gn=add(`g-${b.id}-${g.id}`,g.label,30,cursor,"right","group",b.color,b.id,{sub:g.relation==="学习分组"?g.examples:g.relation,compact:true});connect(root,gn,"学习分组",{compactRoot:true});cursor+=64;
          for(let i=0;i<g.leaves.length;i++){const label=g.leaves[i],lines=textLines(label,18),leaf=add(`l-${b.id}-${g.id}-${i}`,label,60,cursor+(lines.length-1)*15,"right","leaf",b.color,b.id,{relation:g.relation,compact:true});connect(gn,leaf,g.relation,{compactChild:true});cursor+=Math.max(46,lines.length*31+12);}
          cursor+=55;
        }
        return {mode,compact:true,width:560,height:cursor+30,nodes,edges};
      }
      const b=byId[mode],count=b.groups.reduce((s,g)=>s+g.leaves.length,0),height=Math.max(800,count*46+b.groups.length*32+120);
      const root=add(`b-${b.id}`,b.title,185,height/2,"center","root",b.color,b.id,{sub:"当前分支 · 总图按钮返回"});let cursor=78;
      for(const g of b.groups){const start=cursor,ys=g.leaves.map(()=>{const y=cursor;cursor+=46;return y;}),gy=(start+ys[ys.length-1])/2;
        const gn=add(`g-${b.id}-${g.id}`,g.label,800,gy,"left","group",b.color,b.id,{sub:g.relation==="学习分组"?g.examples:g.relation,relation:g.relation});connect(root,gn);
        g.leaves.forEach((label,i)=>{const leaf=add(`l-${b.id}-${g.id}-${i}`,label,880,ys[i],"right","leaf",b.color,b.id,{relation:g.relation});connect(gn,leaf,g.relation);});cursor+=32;
      }
      return {mode,width:1600,height,nodes,edges};
    }
    if(mode==="overview"){
      const root=add("root","VLA 知识框架",1000,770,"center","root","#183449","",{sub:"十个观察视角"});
      branches.forEach((b,i)=>{const left=i<5,y=165+(i%5)*300,side=left?"left":"right",n=add(`b-${b.id}`,`${String(i+1).padStart(2,"0")}  ${b.title}`,left?520:1480,y,side,"branch",b.color,b.id);
        connect(root,n,"观察视角",{overviewRoot:true});
        b.groups.forEach((g,j)=>{const gy=y+(j-(b.groups.length-1)/2)*60,gn=add(`g-${b.id}-${g.id}`,g.label,left?420:1580,gy,side,"group",b.color,b.id,{sub:g.examples});connect(n,gn);});
      });
      return {mode,width:2000,height:1540,nodes,edges};
    }
    const sideHeights=[0,1].map(side=>branches.slice(side*5,side*5+5).reduce((sum,b)=>sum+b.groups.reduce((n,g)=>n+g.leaves.length*43+35,0)+90,0));
    const height=Math.max(...sideHeights)+160,root=add("root","VLA 知识框架",1500,height/2,"center","root","#183449","",{sub:"十个视角 · 完整知识树"});
    for(let sideIndex=0;sideIndex<2;sideIndex++){const left=sideIndex===0,side=left?"left":"right";let cursor=80+(height-160-sideHeights[sideIndex])/2;
      branches.slice(sideIndex*5,sideIndex*5+5).forEach((b,offset)=>{const start=cursor,groupNodes=[];
        for(const g of b.groups){const ys=g.leaves.map(()=>{const y=cursor;cursor+=43;return y;}),gy=(ys[0]+ys.at(-1))/2;
          const gn=add(`g-${b.id}-${g.id}`,g.label,left?900:2100,gy,side,"group",b.color,b.id,{sub:g.relation});groupNodes.push(gn);
          g.leaves.forEach((label,i)=>{const leaf=add(`l-${b.id}-${g.id}-${i}`,label,left?470:2530,ys[i],side,"leaf",b.color,b.id);connect(gn,leaf,g.relation);});cursor+=35;
        }
        const y=(start+cursor-35)/2,n=add(`b-${b.id}`,`${String(offset+sideIndex*5+1).padStart(2,"0")} ${b.title}`,left?1220:1780,y,side,"branch",b.color,b.id);connect(root,n,"观察视角");groupNodes.forEach(g=>connect(n,g));cursor+=90;
      });
    }
    // Keep a real margin around the longest English leaf in the downloadable tree.
    for(const node of nodes)node.x+=200;
    return {mode,width:3400,height,nodes,edges};
  }
  function fitView(width,height,aspect=1,padding=55){if(![width,height,aspect,padding].every(Number.isFinite)||width<=0||height<=0||aspect<=0||padding<0)throw new RangeError("Invalid canvas bounds");let w=width+2*padding,h=height+2*padding;if(w/h<aspect)w=h*aspect;else h=w/aspect;return{x:(width-w)/2,y:(height-h)/2,w,h};}
  function zoomAt(view,factor,px=.5,py=.5){if(!view||![view.x,view.y,view.w,view.h,factor,px,py].every(Number.isFinite)||view.w<=0||view.h<=0||factor<=0||px<0||px>1||py<0||py>1)throw new RangeError("Invalid zoom input");const w=view.w/factor,h=view.h/factor;return{x:view.x+px*(view.w-w),y:view.y+py*(view.h-h),w,h};}
  function labelSVG(n,mode){
    let x=n.x,anchor="middle",font=mode==="overview"?25:28;
    if(n.kind==="root")return `<g class="vf-node vf-root" data-vf-node="${n.id}" data-vf-kind="root" data-vf-branch="${n.branch}"><rect x="${x-150}" y="${n.y-58}" width="300" height="116" rx="40" fill="${n.color}"/><text x="${x}" y="${n.y-5}" text-anchor="middle" font-size="32" font-weight="700" fill="#fff">${esc(n.label)}</text><text x="${x}" y="${n.y+31}" text-anchor="middle" font-size="${n.compact?18:21}" fill="#e1ebf6">${esc(n.sub||"")}</text></g>`;
    if(n.side==="left"){anchor="end";x=n.x-20;if(n.kind==="branch"&&mode==="overview")x=n.x+300;}else if(n.side==="right"){anchor="start";x=n.x+20;if(n.kind==="branch"&&mode==="overview")x=n.x-300;}
    if(n.kind==="branch")font=32;
    if(n.kind==="leaf")font=n.compact?25:30;
    const lines=textLines(n.label,n.kind==="leaf"?(n.compact?18:22):26),lineH=font+5,startY=n.y-(n.sub?15:0)-(lines.length-1)*lineH/2;
    const text=lines.map((line,i)=>`<tspan x="${x}" y="${startY+i*lineH}">${esc(line)}</tspan>`).join("");
    const sub=n.sub?`<text class="vf-sub" x="${x}" y="${startY+lines.length*lineH-6}" text-anchor="${anchor}" font-size="${mode==="overview"?20:21}">${esc(n.sub)}</text>`:"";
    return `<g class="vf-node vf-${n.kind}" data-vf-node="${n.id}" data-vf-kind="${n.kind}" data-vf-branch="${n.branch}" tabindex="0" role="button" aria-label="${esc(n.label)}${n.kind==="branch"||n.kind==="group"?"，聚焦此知识分支":""}"><title>${esc(n.label)}${n.relation?"；"+esc(n.relation):""}</title><circle cx="${n.x}" cy="${n.y}" r="${n.kind==="leaf"?3.5:5}" fill="${n.color}"/><text class="vf-label" x="${x}" y="${startY}" text-anchor="${anchor}" font-size="${font}" font-weight="${n.kind==="branch"?700:500}" fill="${n.kind==="leaf"?"var(--vf-ink,#183449)":n.color}">${text}</text>${sub}</g>`;
  }
  function svgFor(options={}){
    const mode=options.mode||"overview",graph=layout(mode,options),index=Object.fromEntries(graph.nodes.map(n=>[n.id,n]));
    let edgeMarkup=graph.edges.map(e=>{const a=index[e.from],b=index[e.to];let d;
      if(e.compactRoot){d=`M${a.x} ${a.y+58} Q${a.x} 170 30 170 V${b.y}`;}
      else if(e.compactChild){d=`M${a.x} ${a.y} V${b.y} H${b.x}`;}
      else if(e.dashed){d=`M${a.x} ${a.y} Q${(a.x+b.x)/2} ${Math.min(a.y,b.y)-100} ${b.x} ${b.y}`;}
      else if(e.overviewRoot){const left=b.side==="left",start=left?850:1150,elbow=left?820:1180;d=`M${start} ${a.y} C${elbow} ${a.y} ${elbow} ${b.y} ${elbow} ${b.y} L${b.x} ${b.y}`;}
      else if(Object.hasOwn(byId,mode)&&a.kind==="root"){d=`M335 ${a.y} C620 ${a.y} 620 ${b.y} 620 ${b.y} L${b.x} ${b.y}`;}
      else {const bend=(a.x+b.x)/2;d=`M${a.x} ${a.y} C${bend} ${a.y} ${bend} ${b.y} ${b.x} ${b.y}`;}
      const label=e.dashed?`<text class="vf-relation-label" x="${(a.x+b.x)/2}" y="${(a.y+b.y)/2-45}" text-anchor="middle" font-size="25">${esc(e.relation)}</text>`:"";
      return `<g class="vf-edge${e.dashed?" vf-association":""}" data-from="${e.from}" data-to="${e.to}" data-relation="${esc(e.relation)}"><path d="${d}" stroke="${e.color}" stroke-width="${e.dashed?3:2.5}" fill="none"${e.dashed?' stroke-dasharray="10 8"':""}/>${label}</g>`;
    }).join("");
    const style=`.vf-label,.vf-sub,.vf-relation-label{paint-order:stroke;stroke:var(--vf-paper,#fff);stroke-width:7px;stroke-linejoin:round}.vf-sub{fill:var(--vf-muted,#5b6b80)}.vf-relation-label{fill:var(--vf-ink,#183449)}.vf-node{cursor:pointer}.vf-node:focus{outline:none}.vf-node:focus .vf-label{font-weight:800;text-decoration:underline}.vf-node:hover .vf-label{text-decoration:underline}`;
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${graph.width}" height="${graph.height}" viewBox="0 0 ${graph.width} ${graph.height}" data-vf-mode="${mode}" data-vf-compact="${Boolean(graph.compact)}" role="img" aria-labelledby="vf-title vf-desc"><title id="vf-title">VLA 知识框架${mode==="overview"?"总览":mode==="full"?"完整图":mode==="relations"?"：SigLIP 跨视角关联":"："+esc(byId[mode].title)}</title><desc id="vf-desc">中心主题向十个观察视角展开，再分为知识分组与具体术语。实线表示学习分组，不能全部理解为继承或物理组成。虚线表示标注了名称的跨视角关联。</desc><style>${style}</style><rect class="vf-background" width="100%" height="100%" fill="var(--vf-paper,#fff)"/><g font-family="system-ui,-apple-system,BlinkMacSystemFont,sans-serif">${edgeMarkup}${graph.nodes.map(n=>labelSVG(n,mode)).join("")}<text x="${graph.width/2}" y="${graph.height-22}" text-anchor="middle" font-size="20" fill="var(--vf-muted,#5b6b80)">${graph.compact?"实线：学习分组 · 向下拖动继续阅读":"实线：知识分组（非统一继承树）　虚线：具名关联"}</text></g></svg>`;
  }
  const api={branches,layout,svgFor,fitView,zoomAt,textLines};
  if(typeof module==="object"&&module.exports)module.exports=api;
  scope.EmbodiedVLAFramework=api;
  if(!scope.document)return;
  const doc=scope.document;let current=null;
  function mount(root){
    root.dataset.ready="true";
    root.innerHTML=`<div class="vf-toolbar"><label>知识分支<select name="vla-framework-branch"><option value="overview">总框架 · 十个视角</option>${branches.map((b,i)=>`<option value="${b.id}">${String(i+1).padStart(2,"0")} ${esc(b.title)}</option>`).join("")}</select></label><div class="vf-tools" role="group" aria-label="知识框架工具"><button type="button" data-vf-action="reset">总图</button><button type="button" data-vf-action="out" aria-label="缩小框架图">−</button><output data-vf-zoom aria-label="当前缩放">100%</output><button type="button" data-vf-action="in" aria-label="放大框架图">＋</button><button type="button" data-vf-action="fit">适应画布</button><button type="button" data-vf-action="relations" aria-pressed="false">看跨视角关系</button><button type="button" data-vf-action="fullscreen">全屏</button><button type="button" data-vf-action="download">下载完整 SVG</button></div></div><p class="vf-help">从中心沿连线向外读。点击分支展开具体名词；拖动画布移动，Ctrl/⌘＋滚轮或双指缩放。键盘：方向键移动，＋/−缩放，0适应。</p><div class="vf-canvas" tabindex="0" role="region" aria-label="可缩放拖动的 VLA 知识框架图"></div><div class="vf-selection" role="status" aria-live="polite">总览显示十个视角和二级知识分组。具体模型可以同时属于多个视角。</div>`;
    const canvas=root.querySelector(".vf-canvas"),select=root.querySelector("select"),status=root.querySelector(".vf-selection");
    let mode=scope.matchMedia?.("(max-width:600px)").matches?"role":"overview",previous=mode,graph,svg,view,baseView,drag=null,moved=false,pinch=null;const pointers=new Map();
    function apply(){svg.setAttribute("viewBox",`${view.x} ${view.y} ${view.w} ${view.h}`);root.querySelector("[data-vf-zoom]").textContent=`${Math.round(baseView.w/view.w*100)}%`;}
    function fit(initial=true){if(!graph)return;const r=canvas.getBoundingClientRect(),aspect=Math.max(.1,r.width/Math.max(r.height,1));baseView=graph.compact&&initial?{x:-20,y:0,w:600,h:600/aspect}:fitView(graph.width,graph.height,aspect,30);view={...baseView};apply();}
    function render(){root.dataset.vfMode=mode;const compact=canvas.getBoundingClientRect().width<=600;graph=layout(mode,{compact});canvas.innerHTML=svgFor({mode,compact});svg=canvas.querySelector("svg");svg.removeAttribute("width");svg.removeAttribute("height");svg.setAttribute("role","group");select.value=Object.hasOwn(byId,mode)?mode:"overview";root.querySelector('[data-vf-action="relations"]').setAttribute("aria-pressed",String(mode==="relations"));status.textContent=mode==="overview"?"总览显示十个视角与二级知识分组。点击分支，查看其具体术语。":mode==="relations"?"同一 SigLIP 图像分支，按功能担任视觉编码器、按结构采用 ViT、按权重来源属于预训练组件。":"当前展开："+byId[mode].title+(graph.compact?"；向下拖动画布继续阅读。":"。")+"实线首先表示学习分组，具名组成关系见分组说明。";fit();}
    function zoom(factor,px=.5,py=.5){const now=baseView.w/view.w,target=Math.max(.35,Math.min(6,now*factor));view=zoomAt(view,target/now,px,py);apply();}
    function focusBranch(id){if(!Object.hasOwn(byId,id))return;mode=id;previous=id;render();canvas.focus({preventScroll:true});}
    function action(name){if(name==="in")zoom(1.3);else if(name==="out")zoom(1/1.3);else if(name==="fit")fit(false);else if(name==="reset"){mode="overview";previous=mode;render();}else if(name==="relations"){if(mode==="relations")mode=previous;else{previous=mode;mode="relations";}render();}else if(name==="fullscreen"){const promise=doc.fullscreenElement===root?doc.exitFullscreen?.():root.requestFullscreen?.();if(promise&&promise.catch)promise.catch(()=>{status.textContent="当前浏览器未允许全屏；仍可拖动与缩放阅读。";});}else if(name==="download"){const url=scope.URL.createObjectURL(new scope.Blob([svgFor({mode:"full"})],{type:"image/svg+xml;charset=utf-8"}));const a=doc.createElement("a");a.href=url;a.download="VLA-知识框架-完整.svg";a.click();scope.setTimeout(()=>scope.URL.revokeObjectURL(url),1000);status.textContent="已导出完整知识框架 SVG；可在浏览器或矢量工具中缩放阅读。";}}
    root.querySelector(".vf-tools").addEventListener("click",e=>{const b=e.target.closest("[data-vf-action]");if(b)action(b.dataset.vfAction);});
    select.addEventListener("change",()=>{mode=select.value;previous=mode;render();});
    canvas.addEventListener("click",e=>{if(moved)return;const n=e.target.closest("[data-vf-node]");if(!n)return;if(mode==="overview")focusBranch(n.dataset.vfBranch);else if(mode==="relations"){const id=n.dataset.vfBranch;if(Object.hasOwn(byId,id))focusBranch(id);}else{const item=graph.nodes.find(x=>x.id===n.dataset.vfNode);if(item)status.innerHTML=`<strong>${esc(item.label)}</strong> · ${esc(item.relation||"学习分组")}。<a href="#${byId[mode].anchor}">阅读这个分支的详细讲解 ↓</a>`;}});
    canvas.addEventListener("keydown",e=>{const n=e.target.closest("[data-vf-node]");if(n&&(e.key==="Enter"||e.key===" ")){e.preventDefault();n.dispatchEvent(new MouseEvent("click",{bubbles:true}));return;}const amount=view.w*.06;if(["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key)){e.preventDefault();if(e.key==="ArrowLeft")view.x-=amount;if(e.key==="ArrowRight")view.x+=amount;if(e.key==="ArrowUp")view.y-=amount;if(e.key==="ArrowDown")view.y+=amount;apply();}else if(e.key==="+"||e.key==="="){e.preventDefault();zoom(1.3);}else if(e.key==="-"){e.preventDefault();zoom(1/1.3);}else if(e.key==="0"){e.preventDefault();fit(false);}});
    canvas.addEventListener("wheel",e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();const r=canvas.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.003),(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height);},{passive:false});
    canvas.addEventListener("pointerdown",e=>{if(e.button!==0)return;pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(!e.target.closest("[data-vf-node]"))canvas.setPointerCapture(e.pointerId);moved=false;if(pointers.size===1)drag={x:e.clientX,y:e.clientY,view:{...view}};if(pointers.size===2){const [a,b]=[...pointers.values()],r=canvas.getBoundingClientRect();pinch={distance:Math.hypot(a.x-b.x,a.y-b.y),view:{...view},px:((a.x+b.x)/2-r.left)/r.width,py:((a.y+b.y)/2-r.top)/r.height};drag=null;}});
    canvas.addEventListener("pointermove",e=>{if(!pointers.has(e.pointerId))return;pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(pinch&&pointers.size===2){const [a,b]=[...pointers.values()],ratio=Math.hypot(a.x-b.x,a.y-b.y)/Math.max(pinch.distance,1),old=baseView.w/pinch.view.w,target=Math.max(.35,Math.min(6,old*ratio));view=zoomAt(pinch.view,target/old,pinch.px,pinch.py);moved=true;apply();}else if(drag){const r=canvas.getBoundingClientRect(),dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)>4){moved=true;if(!canvas.hasPointerCapture(e.pointerId))canvas.setPointerCapture(e.pointerId);}if(moved){view={...drag.view,x:drag.view.x-dx/r.width*drag.view.w,y:drag.view.y-dy/r.height*drag.view.h};apply();}}});
    const end=e=>{pointers.delete(e.pointerId);drag=null;pinch=null;scope.setTimeout(()=>{moved=false;},0);};canvas.addEventListener("pointerup",end);canvas.addEventListener("pointercancel",end);
    const resized=()=>{const compact=canvas.getBoundingClientRect().width<=600;if(graph&&Object.hasOwn(byId,mode)&&Boolean(graph.compact)!==compact)render();else fit();};const observer=typeof scope.ResizeObserver==="function"?new scope.ResizeObserver(resized):null;observer?.observe(canvas);doc.addEventListener("fullscreenchange",resized);render();
    return {root,destroy(){observer?.disconnect();doc.removeEventListener("fullscreenchange",resized);}};
  }
  function init(){if(current&&!current.root.isConnected){current.destroy();current=null;}const root=doc.querySelector("[data-vla-framework]");if(root&&!root.dataset.ready)current=mount(root);}
  if(typeof document$!=="undefined")document$.subscribe(init);
  if(doc.readyState==="loading")doc.addEventListener("DOMContentLoaded",init,{once:true});else init();
})(typeof window!=="undefined"?window:globalThis);
