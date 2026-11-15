/* Local concept navigation. Different classification views may overlap. No telemetry. */
(function (scope) {
  "use strict";
  const dimensions = [
    ["category", "模型类别", "它解决什么问题？"],
    ["module", "功能模块", "它在系统中负责什么？"],
    ["architecture", "网络结构", "这个模块怎样实现？"],
    ["inside", "网络内部", "向量与计算怎样流动？"],
    ["instance", "具体组件", "这个名字对应哪套模型？"],
    ["data", "输入与数据", "观测与监督来自哪里？"],
    ["action", "动作与生成", "数字意味着什么、怎样产生？"],
    ["training", "训练与监督", "什么目标更新哪些参数？"],
    ["execution", "软件与执行", "代码怎样接上机器人？"],
    ["evaluation", "评测与泛化", "证据支持什么结论？"]
  ].map(([id, title, question], i) => ({id, title, question, number: i + 1}));
  const D = Object.fromEntries(dimensions.map(d => [d.id, d]));
  function term(id, name, aliases, views, meaning, location, contrast, example) {
    return {id, name, aliases, views: views.split(" "), meaning, location, contrast, example};
  }
  const terms = [
    term("vla", "VLA", "Vision Language Action 视觉语言动作", "category", "以视觉、语言等条件生成机器人动作的策略模型。", "观测与任务条件到动作参考之间。", "它是模型类别，不是一种网络层；也不代表自动具备世界模型。", "同一张桌面图配红杯/蓝杯指令，应得到不同的目标动作。"),
    term("vlm", "VLM", "Vision Language Model 视觉语言", "category", "联合处理视觉与语言信息的模型。", "可作为 VLA 的多模态表征主干。", "能描述抓取不代表能提供经过动作合同解码的控制命令。", "PaliGemma 提供图文处理能力，动作专家另负责动作生成。"),
    term("llm", "LLM", "Large Language Model 大语言模型", "category", "主要围绕语言序列训练和使用的模型。", "可作为多模态系统的语言主干。", "Llama 是具体模型家族，LLM 是类别；不是所有策略都有 LLM。", "原版 OpenVLA 使用 Llama 2 处理融合后的序列。"),
    term("policy", "Policy / 策略", "策略 观测动作映射", "category", "根据当前条件或历史决定动作的规则或概率分布。", "控制决策层。", "Policy 比 VLA 更宽，不依赖语言的 BC 策略也属于 policy。", "只看图像的推块策略仍然是策略。"),
    term("world-model", "World Model", "世界模型 环境预测 动力学", "category", "预测环境状态、观测或结果怎样随动作变化的模型。", "可用于训练监督、规划或模型学习。", "预测未来图像与直接输出动作是不同目标；两者可以组合。", "预测抓起后杯子的运动，再用预测评估候选动作。"),
    term("agent", "Agent / 智能体", "智能体 工具 规划", "category", "围绕观察、决策、工具调用、执行与反馈组织起来的系统。", "可包裹一个或多个模型与控制接口。", "Agent 是系统角色，VLA 是可被调用的一类动作模型。", "高层系统分解整理桌面的子任务，再调用低层策略。"),
    term("foundation", "Foundation Model", "基础模型 迁移 预训练", "category training", "在广泛数据上训练、可适配多种下游用途的模型。", "描述训练与迁移用途，不限定内部结构。", "参数多不自动证明泛化，基础模型也不等于 Transformer。", "比较目标任务微调前后表现，记录预训练来源。"),
    term("encoder", "Encoder / 编码器", "视觉编码器 Vision Encoder", "module", "把输入转成后续网络使用的表示。", "图像、状态或其他模态的输入端。", "编码器是职能；CNN 和 ViT 是可能的实现。", "RGB 图像经过视觉编码器，得到一组 patch 特征。"),
    term("tokenizer", "Tokenizer", "分词器 离散编码器", "module inside", "按一套规则把文本或动作转换为离散序列。", "原始数据与 token ID 之间。", "Tokenizer 不等于 Embedding；离散编号还不是向量。", "文字被切成多个编号，再查嵌入表。"),
    term("projector", "Projector / 投影器", "投影 特征对齐", "module", "将一种特征变为另一模块所需的维度或表示。", "视觉特征和语言主干之间，也可用于辅助监督。", "改变维度不自动保证语义或空间正确。", "原版 OpenVLA 把融合视觉特征投影到语言主干输入空间。"),
    term("backbone", "Backbone / 主干", "主干 骨干", "module", "承担主要特征变换的网络部分。", "输入编码与任务输出之间。", "主干是职责名称；Llama、Gemma 是具体实现来源。", "同一主干可以接动作头，也可以接其他任务头。"),
    term("fusion", "多模态融合", "Fusion 拼接 Cross Attention", "module", "让图像、语言、状态等信息发生条件联系。", "多模态主干或独立条件模块中。", "把数组放在同一个字典里还不是网络中的信息交互。", "图像和指令 token 经注意力交换信息。"),
    term("action-expert", "Action Expert", "动作专家 Action Head 动作头", "module", "专门生成动作的网络模块；可以远比一层输出层复杂。", "条件表征到动作样本之间。", "专家是网络模块，示教者是数据来源，二者不是一回事。", "π₀.₅ 的连续动作路径用流匹配动作专家。"),
    term("controller", "Controller / 控制器", "底层控制器 PID servo 伺服", "module execution", "根据目标和反馈，让真实运动跟随参考。", "模型动作参考与执行器之间。", "模型输出末端位移不等于直接输出电机电流。", "控制器把经过检查的末端目标交给运动学和关节伺服。"),
    term("mlp", "MLP / FFN", "多层感知机 Feed Forward Network 前馈网络", "architecture inside", "线性层与非线性组合而成的网络或子模块。", "可出现在投影器、状态编码、Transformer 内部和动作头。", "MLP 与 Transformer 可以嵌套，不是必须二选一。", "一个 Transformer block 通常在注意力之外还有 FFN。"),
    term("cnn", "CNN", "卷积神经网络 ResNet EfficientNet", "architecture", "利用共享局部卷积核处理空间结构的网络。", "常用于视觉编码，叠层可以扩大感受范围。", "视觉编码器不一定是 CNN，也可以用 ViT。", "小型图像策略可用 CNN 编码桌面图像。"),
    term("rnn", "RNN / LSTM / GRU", "循环神经网络 隐状态 历史", "architecture", "按序列递推并维护隐藏状态的一类网络。", "可以用于历史观测或时序建模。", "需要历史不等于必须使用 RNN，Transformer 也可处理历史。", "隐藏状态总结先前看到的运动信息。"),
    term("transformer", "Transformer", "变换器 Encoder Decoder", "architecture", "以注意力、前馈变换等模块处理序列的网络结构家族。", "可用于视觉、语言、多模态主干或动作专家。", "Transformer 是结构，VLA 是用途类别，Flow 是生成/学习方法。", "用 Transformer 实现流匹配动作专家并不矛盾。"),
    term("vit", "ViT", "Vision Transformer 图像Transformer 视觉Transformer", "architecture instance", "将图像 patch 转成序列并用 Transformer 处理的视觉设计。", "例如 SigLIP 图像塔和 DINOv2 的视觉主干。", "ViT 与 encoder/decoder 不是互斥的同级类别。", "同一视觉组件可同时标注视觉编码器、ViT、预训练。"),
    term("tensor", "Tensor / Shape", "张量 形状 batch 维度", "inside", "张量是多维数组；shape 说明每个轴有多少元素。", "输入、特征、梯度与动作都可用张量表示。", "维度数与每轴大小不同，动作维度也不是参数量。", "[B,H,D] 表示批大小、动作步数、每步通道数。"),
    term("patch", "Patch / 图像块", "图像块 patch embedding", "inside", "将图像按空间区域划分后得到的块。", "ViT 等视觉输入处理中。", "图像块不是物体标签，一个杯子可能覆盖多个 patch。", "224×224 图像按 16×16 分块得到 196 块；这是教学设定。"),
    term("token", "Token", "词元 序列单元 token id 编号", "inside", "网络序列中的处理单元，可来自文字、图像、状态或动作。", "输入序列、隐藏序列和输出编码中。", "文字 token ID、视觉向量和动作编码都可能叫 token，要看上下文。", "一个视觉 token 通常不能直接命名为一个独立物体。"),
    term("embedding", "Embedding", "嵌入 嵌入向量 嵌入层", "inside module", "把输入映射到向量空间的表示；也指完成映射的层。", "token ID 查表、图像 patch 投影等位置。", "嵌入向量随输入变化，嵌入表的参数通常在测试时固定。", "同一嵌入表给红杯与蓝杯的 token 返回不同向量。"),
    term("feature", "Hidden Feature", "特征 hidden state 中间特征 隐藏状态", "inside", "输入经过网络计算后得到的中间表示。", "各个网络层之间。", "特征随当前样本变化，不是训练后保存的权重矩阵。", "固定网络权重、换一张图像，中间特征通常会变化。"),
    term("parameter", "Parameter / 参数", "权重 Weight bias W b 可学习参数", "inside training", "训练中优化、在常规推理时固定的数值。", "线性矩阵、偏置、嵌入表和其他可训练模块。", "固定 W 不等于固定 XW；注意力分数也会随输入变化。", "Y=XW+b 中 W、b 是参数，X、Y 是本次计算的张量。"),
    term("attention", "Attention / QKV", "注意力 Query Key Value 查询 键 值", "inside", "从 Q/K 匹配得到权重，再对 V 所携带的信息加权汇总。", "Transformer 的信息交换运算。", "WQ/WK/WV 是参数；Q/K/V 和注意力权重是当前输入算出的量。", "文本查询可以读取与目标有关的视觉表示。"),
    term("self-attention", "Self-attention", "自注意力 self attention", "inside", "Q、K、V 来自同一组序列表示的注意力。", "文本内部、视觉内部或合并后的多模态序列中。", "Cross-attention 的 Q 与 K/V 来自不同表示流。", "合并图文序列可在 mask 允许的位置内部交换信息。"),
    term("cross-attention", "Cross-attention", "交叉注意力 cross attention", "inside", "一组表示提供 Q，另一组表示提供 K/V。", "动作专家读取语义上下文等跨表示接口。", "不同模态之间的信息联系不必都用 cross-attention。", "动作表示用自身查询读取图文条件。"),
    term("mask", "Mask / 掩码", "attention mask padding mask loss mask causal 因果", "inside data training", "指示哪些位置可读、哪些输入存在或哪些目标参与损失。", "注意力、数据补齐与训练目标中。", "attention/padding/loss mask 和图像分割 mask 不可混用。", "episode 尾部补齐的动作可被 loss mask 排除。"),
    term("residual", "Residual / 残差连接", "残差 skip connection 残差策略", "inside architecture", "把模块输入和相容形状的变换输出相加，例如 x+F(x)。", "神经网络内部的特征路径。", "残差控制策略是另一层接口，修正动作/力而非这里的隐藏特征。", "加法两侧要有兼容 shape；力矩残差不能直接加到角度。"),
    term("norm", "LayerNorm / RMSNorm", "归一化 normalization Norm 层归一化", "inside", "按约定轴对隐藏特征作尺度调整的网络运算。", "Transformer 等网络块内部。", "网络 Norm 与训练动作统计的反归一化是不同操作。", "改变动作单位需要训练统计，不能靠 LayerNorm 自动还原。"),
    term("rope", "位置编码 / RoPE", "Rotary Position Embedding 旋转位置编码 positional encoding", "inside", "将位置或顺序信息引入序列计算；RoPE 是一种旋转式方法。", "注意力输入的位置信息处理。", "网络的位置编码不是机器人世界坐标或相机外参。", "区分两个相同 token 出现在不同序列位置。"),
    term("siglip", "SigLIP", "sig lip 图文预训练 视觉塔", "instance module architecture", "图文预训练模型；VLA 可使用其中的图像编码分支。", "原版 OpenVLA 的融合视觉路径之一。", "图像分支按职责是视觉编码器、按结构可为 ViT；不是语言主干。", "它和 DINOv2 的特征先融合，再投影到 Llama 输入空间。"),
    term("dinov2", "DINOv2", "dino v2 自监督视觉 ViT", "instance module architecture", "以自监督预训练学习视觉特征的一组模型。", "原版 OpenVLA 的另一条视觉特征路径。", "DINOv2 提供视觉表示，不是完整 VLA。", "使用现成视觉权重，不意味着自动完成机器人动作学习。"),
    term("llama", "Llama / Gemma", "llama2 llama 2 gemma 语言主干", "instance", "具体语言模型家族，可被更大系统复用为主干。", "Llama 2 在原版 OpenVLA；Gemma 在 PaliGemma 中。", "模型家族是具体组件，LLM/主干/Transformer 是其他视角的标签。", "更换主干通常还要匹配 tokenizer、隐藏宽度和预训练权重。"),
    term("paligemma", "PaliGemma", "pali gemma SigLIP Gemma", "instance module", "由 SigLIP 图像编码与 Gemma 语言主干等构成的 VLM。", "π₀.₅ 图文主干的来源。", "不能把 PaliGemma、SigLIP、Gemma画成三套并列的完整网络。", "点击 π₀.₅ 路径，查看组件内嵌关系。"),
    term("openvla", "OpenVLA", "open vla 原版 vanilla", "instance category", "复用预训练图文表征并生成动作 token 的具体 VLA。", "图像/指令到离散动作编码的策略。", "原版 OpenVLA 与连续动作的 OFT 配方不能混为一谈。", "本图只描绘原版；动作语义还需匹配数据和控制器。"),
    term("pi05", "π₀.₅", "pi0.5 pi05 pi 0.5 π0.5", "instance category", "结合异构训练与高层语义、低层动作生成的具体策略。", "本图着重原论文的低层连续动作路径。", "原论文含离散与连续训练；openpi 当前公开 pi05 仅支持 flow 头训练/推理。", "动作专家与 PaliGemma 条件交互，生成连续动作块。"),
    term("sla", "SLA", "Spatial Language Action 空间对齐 持续预训练", "instance training", "从 π₀.₅ 出发，在训练时增加空间特征对齐等设计的工作。", "持续预训练与后训练期间。", "冻结教师和辅助门控/投影属于训练设计；部署没有这些辅助组件。", "切换 SLA 训练/部署，查看教师支路出现与移除。"),
    term("rgb", "RGB / Depth / 点云", "深度图 point cloud RGBD RGB-D", "data", "彩色像素、深度或三维点是不同的传感器/几何数据表示。", "输入模态或训练时辅助信息。", "不是每个 VLA 都需要深度，也不能将训练用深度默认当部署输入。", "SLA 教师可利用部分可用深度和标定，部署沿用原策略输入。"),
    term("state", "Proprioception", "本体状态 关节 状态 state", "data", "描述机器人自身情况的关节、夹爪或其他传感状态。", "与图像和语言一起作为某些策略的条件。", "状态不是动作目标；相同图像也可能对应不同机器人构型。", "7 个关节角输入与 7 维末端动作输出并非同一个空间。"),
    term("episode", "Episode / Trajectory", "回合 轨迹 Demonstration 示范 Dataset 数据集", "data", "Episode 描述一段交互边界，trajectory 是时序记录，示范是用于学习的行为数据。", "训练样本的来源与划分单位。", "相邻帧不是独立示范，重叠片段不能随意跨训练/测试集。", "从同一 episode 取动作窗口，结尾使用明确 padding mask。"),
    term("teleop", "Teleoperation", "遥操作 OXE DROID 演示采集", "data", "人通过设备控制机器人并记录行为的示范采集方式。", "数据生成阶段。", "人手视频不直接包含目标机器人电机命令。", "示教时保存图像、指令、状态、动作及时间戳。"),
    term("action-repr", "动作表示", "joint 关节 end effector 末端 absolute delta 增量 旋转 夹爪", "action", "说明每个动作通道对应的物理量、单位、坐标和绝对/增量含义。", "数据标签、输出解码和控制器接口之间。", "动作表示与生成方法独立；flow 可生成关节目标或末端动作。", "归一化 0.5 在 ±20mm 训练界下对应 +10mm，而不是 0.5rad。"),
    term("chunk", "Action Chunk", "动作块 预测时域 horizon 执行前缀 H K", "action execution", "一次预测未来 H 个采样位置，实际可能只执行前 K 步。", "策略输出与执行调度之间。", "H、K、动作采样频率、推理频率与电机伺服频率都不同。", "20Hz、H16/K4、推理100ms的串行周期为300ms。"),
    term("regression", "Regression", "回归 L1 L2 MSE 连续回归", "action training", "通过预测数值与目标间的误差学习连续输出。", "可用于直接动作头或其他数值目标。", "MSE 最优点是条件均值；逐分量 L1 对应中位数，不能全叫平均。", "左右绕障都可行时，单一中间动作可能撞上障碍。"),
    term("autoregressive", "Autoregressive", "自回归 action token Next Token FAST", "action", "按顺序生成编码，每个位置依赖已生成内容。", "原版 OpenVLA 的动作 token 路径。", "它是生成顺序，不是一种固定物理动作或独立网络结构。", "生成动作编码后仍要解码和反归一化。"),
    term("diffusion", "Diffusion", "扩散 去噪 Diffusion Policy", "action training", "学习噪声条件下的目标，再按匹配采样器生成动作样本。", "动作生成器可以用 Transformer 或其他网络实现。", "不是简单反复减噪声；更新式由训练参数化和采样器决定。", "从噪声动作块开始，多步得到条件动作样本。"),
    term("flow", "Flow Matching", "流匹配 向量场 ODE 噪声", "action training", "在指定概率路径上学习速度场，再通过积分生成动作。", "π₀.₅ 的低层连续动作专家等位置。", "生成时间不是机器人秒数；不同实现的时间方向和符号必须一致。", "教学路径从噪声到动作，向量目标可为动作减噪声。"),
    term("normalization", "动作归一化", "反归一化 训练统计 normalization unnorm", "action data", "利用训练统计或边界改变数值尺度，并按同一规则逆变换。", "数据预处理与动作解码。", "它不是网络 LayerNorm；格式或模型名不决定动作单位。", "训练界 ±20mm：归一化 0.5 还原成 +10mm。"),
    term("pretraining", "训练阶段", "预训练 持续预训练 后训练 微调 pre-training post-training", "training", "描述从初始化到基础能力学习、继续训练和用途适配的阶段。", "模型权重的整个学习过程。", "阶段名称不规定监督方式或必须更换结构；微调也不等于 LoRA。", "SLA 从已有权重继续训练，而不是从随机网络重新开始。"),
    term("bc", "Behavior Cloning", "BC 行为克隆 模仿学习 imitation learning", "training", "根据示范观测与任务条件学习示范动作的训练方法。", "一种模仿学习路线。", "训练动作 loss 低不自动等于闭环成功或语言理解。", "用同场景交换指令测试策略是否依赖任务语言。"),
    term("rl", "Reinforcement Learning", "RL 强化学习 奖励 回报", "training", "利用交互反馈和奖励优化行为的学习路线。", "可用于策略改进或后训练。", "奖励、环境交互预算和停止规则需要明确，不能只用成功视频代表。", "在同预算下对比初始策略与更新策略的完整任务结果。"),
    term("lora", "LoRA / PEFT", "低秩适配 参数高效微调 冻结 Freeze", "training", "LoRA 冻结原有矩阵并训练低秩更新，是 PEFT 的一种。", "指定网络权重的更新规则。", "冻结视觉塔不等于 LoRA；减少可训练参数也不等于所有显存同比减少。", "记录 LoRA 插入位置、rank、冻结项与实际训练参数。"),
    term("teacher", "Teacher / GT", "教师 真值 Ground Truth 伪标签 蒸馏 distillation", "training data", "Teacher 是监督提供者；GT 是被用作参考的数据，二者应标来源。", "标注、仿真、教师特征和训练目标链中。", "冻结教师的预测不自动等于真实测量；学生学习谁需明确。", "SLA 冻结几何教师提供空间特征，不将它称作传感器真值。"),
    term("loss", "Loss / Gradient", "损失 辅助损失 梯度 反向传播 optimizer AdamW 学习率", "training", "损失定义目标，反传计算梯度，优化器依据梯度更新选定参数。", "一次训练更新中。", "Loss、梯度和优化器是不同对象；训练中计算了梯度也不代表每个模块被更新。", "冻结教师与可训练学生可以在同一个计算图中。"),
    term("batch", "Batch / Step / Epoch", "批量 更新步 数据遍历 梯度累计", "training", "Batch 是一组样本，optimizer step 是一次更新，epoch 是一次遍历数据。", "训练配置与日志。", "梯度累计时多个小批次可对应一次更新；流式数据的 epoch 定义需声明。", "比较预算时同时记录样本数、更新次数和环境交互数。"),
    term("framework", "PyTorch / JAX", "框架 自动微分 NumPy", "execution", "PyTorch/JAX 支持自动微分与模型训练；NumPy 主要提供数组数值计算。", "代码实现层。", "PyTorch 不是一种 Transformer，NumPy 练习也不代表加载预训练模型。", "同一个算法可以有 PyTorch 和 JAX 的不同实现。"),
    term("openpi", "openpi / LeRobot", "OpenPI 软件项目 代码库 checkpoint config", "execution", "机器人模型与训练/推理工具项目；具体支持随版本变化。", "安装、配置、权重加载和执行工具链中。", "openpi 是代码项目，不是与 π₀.₅ 并列的单一模型名称。", "固定代码 commit、权重 revision、processor 和数据统计。"),
    term("checkpoint", "Checkpoint / Config", "检查点 配置 超参数 权重", "execution training", "检查点保存权重及可能的训练状态；配置说明模块与运行设置。", "训练保存、重载与复现。", "目录存在不证明可完整重载，也不证明任务成功。", "用新进程重载并对固定输入比较输出和队列 reset。"),
    term("ik", "FK / IK / SDK", "正运动学 逆运动学 接口 坐标 标定 API", "execution", "FK 由关节求末端；IK 由末端目标求构型；SDK 提供程序调用接口。", "动作转换与机器人执行层。", "改变 action_type 标签不会把末端动作转换成关节角。", "基座系末端目标经经过验证的 IK 与控制器执行。"),
    term("closed-loop", "闭环 / 延迟", "Closed Loop Latency 推理频率 控制频率 观测年龄", "execution", "执行后获取反馈再决策；延迟和观测年龄影响能否及时修正。", "观测、推理、调度与执行循环。", "推理吞吐、动作发送和低层伺服频率不是同一个数。", "增大执行前缀能减少等待占比，也使后续动作依据更旧的图像。"),
    term("benchmark", "Benchmark", "基准 Dataset 数据集 Simulator 仿真器", "evaluation", "规定任务、初始化、执行与评分的一整套评测协议。", "模型比较与结果解释。", "数据集、仿真器和 benchmark 不可互相替代。", "同一个 MuJoCo 仿真器可以承载多个不同评分协议。"),
    term("success", "Success / Progress", "成功率 进度 score 分母 置信区间", "evaluation", "成功率统计满足完整成功判据的回合；进度描述部分完成程度。", "逐 episode 结果聚合。", "抓起、放置和完整释放不是相同成功事件。", "40 次中24次完整任务成功，应报告24/40而非36次抓起。"),
    term("generalization", "Generalization", "泛化 ID OOD zero-shot few-shot 分布外", "evaluation", "在定义好的新物体、场景、任务或本体变化上保持能力。", "数据划分与留出测试。", "零样本的含义取决于允许的数据/适配；某一泛化轴不能替代所有轴。", "未见背景测试不自动证明换机器人仍能工作。"),
    term("ablation", "Ablation / 公平比较", "消融 基线 SOTA 随机种子 seed", "evaluation", "在匹配条件下移除或替换设计，检验它贡献了什么。", "实验设计和结论形成。", "更高成功率若数据、预算或干预不同，不能直接归因于结构改进。", "固定场景与预算，对比有无空间教师或正确/交换指令。"),
    term("sim2real", "Sim2Real", "仿真到现实 真机 迁移 MuJoCo Isaac SAPIEN", "evaluation execution", "将在仿真中获得的策略或能力迁移并验证到真实系统。", "跨域部署和评测。", "仿真视频、接口返回值和真实任务完成是不同证据层。", "分别记录模拟与真机的物理参数、传感噪声和控制延迟。")
  ];
  const byId = Object.fromEntries(terms.map(t => [t.id, t]));
  const normalize = value => String(value ?? "").normalize("NFKC").toLowerCase().replace(/π/g, "pi").replace(/[‐‑–—]/g, "-").trim();
  function searchTerms(query = "", view = "all") {
    if (view !== "all" && !Object.hasOwn(D, view)) throw new RangeError("Unknown classification view");
    const words = normalize(query).split(/\s+/).filter(Boolean);
    return terms.filter(t => (view === "all" || t.views.includes(view)) && words.every(word =>
      normalize([t.name, t.aliases, t.meaning, t.location, t.contrast, t.example, ...t.views.map(v => D[v].title)].join(" ")).includes(word)))
      .sort((a, b) => {
        const direct = t => words.every(word => normalize([t.id, t.name, t.aliases].join(" ")).includes(word));
        return Number(direct(b)) - Number(direct(a));
      });
  }
  const nodes = {
    vision: {label: "相机图像", detail: "文字由独立旁路汇入主干", terms: ["rgb", "vla"]},
    fused: {label: "SigLIP + DINOv2", detail: "融合两路视觉特征", terms: ["siglip", "dinov2", "vit", "encoder", "feature"]},
    projector: {label: "Projector", detail: "对接语言输入空间", terms: ["projector", "mlp", "embedding"]},
    llama: {label: "Llama 2 主干", detail: "读取视觉与文字条件", terms: ["tokenizer", "token", "embedding", "llama", "backbone", "transformer", "attention", "self-attention", "mask", "norm", "rope", "parameter", "feature", "fusion"]},
    ar: {label: "动作 token 生成", detail: "原版自回归路线", terms: ["autoregressive", "token", "openvla", "action-expert"]},
    decode: {label: "解码 / 反归一化", detail: "核对动作物理含义", terms: ["normalization", "action-repr"]},
    piinput: {label: "图像 / 指令 / 状态", detail: "依具体输入合同处理", terms: ["rgb", "state", "tokenizer", "embedding", "vla"]},
    pali: {label: "PaliGemma 主干", detail: "内含 SigLIP 与 Gemma", terms: ["paligemma", "siglip", "llama", "vit", "encoder", "backbone", "transformer", "attention", "self-attention", "mask", "norm", "rope", "feature", "fusion", "parameter"]},
    expert: {label: "动作专家", detail: "条件化连续动作生成", terms: ["action-expert", "transformer", "attention", "cross-attention", "flow", "pi05", "chunk"]},
    continuous: {label: "连续动作块", detail: "不是文本或电机电流", terms: ["chunk", "action-repr", "normalization", "flow"]},
    control: {label: "检查 / 控制 / 反馈", detail: "控制器与接口仍必需", terms: ["controller", "ik", "closed-loop", "action-repr", "policy"]},
    demo: {label: "示范 / 训练目标", detail: "未来动作只作标签", terms: ["episode", "teleop", "mask", "bc", "loss", "batch", "pretraining", "teacher"]},
    hybrid: {label: "离散与连续目标", detail: "原论文混合训练概览", terms: ["autoregressive", "flow", "loss", "pretraining", "pi05"]},
    update: {label: "梯度 / 参数更新", detail: "更新允许训练的权重", terms: ["parameter", "loss", "batch", "pretraining", "checkpoint"]},
    spatialinput: {label: "RGB + 可用几何信息", detail: "深度/标定可部分可用", terms: ["rgb", "teacher"]},
    teacher: {label: "冻结 OmniVGGT 教师", detail: "空间特征不是测量真值", terms: ["teacher", "feature", "pretraining", "sla"]},
    align: {label: "空间对齐损失", detail: "对比教师目标与 VLA 层经投影/门控的特征", terms: ["projector", "feature", "loss", "teacher", "sla"]}
  };
  const cases = {
    openvla: {name: "原版 OpenVLA", source: "https://openvla.github.io/#the-openvla-model", intro: "具体 VLA；SigLIP/DINOv2 是视觉组件，Llama 2 是语言主干，自回归是生成方法。图是功能位置示意，未展开每层计算。", textBranch: "文字旁路：指令 → 分词 / 嵌入 → 在 Llama 2 处与视觉条件汇合。文字不经过 SigLIP、DINOv2 图像分支。", route: ["vision", "fused", "projector", "llama", "ar", "decode", "control"], train: ["demo", "update"], note: "训练用示范动作监督 token 预测，不要求将预测下发机器人；部署时没有未来示范动作。不要把 OpenVLA-OFT 的连续动作配方套在这里。"},
    pi05: {name: "π₀.₅ 原论文", source: "https://arxiv.org/html/2504.16054v1", intro: "重点显示原论文低层连续动作路径。PaliGemma 内含 SigLIP + Gemma；高层子任务预测未在主线上展开。", route: ["piinput", "pali", "expert", "continuous", "control"], train: ["demo", "hybrid", "update"], note: "原论文包含离散与连续训练、高层语义等设计；当前 openpi 的 π₀.₅ 实现只支持流匹配头训练/推理。原论文与公开代码范围分开看。"},
    sla: {name: "SLA 空间对齐", source: "https://arxiv.org/html/2611.08373v1", intro: "从已有 π₀.₅ 权重出发，训练中增强空间表征。切换阶段，观察辅助支路的出现和移除。", route: ["piinput", "pali", "expert", "continuous", "control"], train: ["demo", "update"], spatial: ["spatialinput", "teacher", "align"], note: "教师、空间投影与自适应层门控只在训练时使用。部署移除辅助组件，保留进一步训练后的策略；不意味着部署必须额外提供教师深度。"}
  };
  function traceFor(caseId, stage = "deploy") {
    if (!Object.hasOwn(cases, caseId)) throw new RangeError("Unknown model case");
    if (!["train", "deploy"].includes(stage)) throw new RangeError("Unknown stage");
    const c = cases[caseId];
    return {caseId, stage, main: c.route.filter(id => stage === "deploy" || !["control", "decode"].includes(id)).map(id => ({id, ...nodes[id]})),
      supervision: stage === "train" ? c.train.map(id => ({id, ...nodes[id]})) : [],
      spatial: stage === "train" && c.spatial ? c.spatial.map(id => ({id, ...nodes[id]})) : []};
  }
  const api = {dimensions, terms, cases, normalize, searchTerms, traceFor};
  if (typeof module === "object" && module.exports) module.exports = api;
  scope.EmbodiedVLAConcepts = api;
  if (!scope.document) return;
  const doc = scope.document;
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  let current;
  function mount(root) {
    root.dataset.initialized = "true";
    let view = "all", query = "", selected = "siglip", caseId = "openvla", stage = "deploy";
    root.innerHTML = `<header class="vc-intro"><p class="vc-kicker">一张地图，十个观察角度</p><h2>新名词应该放在哪里？</h2><p>先搜索一个词，再沿具体模型看它的位置。同一术语可以属于多个视角：这些标签不是互斥的树枝。</p></header>
      <label class="vc-search-label">查术语、中文或英文别名<input type="search" name="vla-concept-search" placeholder="试试 SigLIP、注意力、LoRA、π₀.₅…" autocomplete="off" aria-controls="vc-results"></label>
      <nav class="vc-views" aria-label="十个分类视角"></nav>
      <div class="vc-search-summary" role="status" aria-live="polite"></div>
      <div class="vc-explorer"><section class="vc-results" id="vc-results" aria-label="术语搜索结果"></section><section class="vc-detail" aria-label="选中术语解释" aria-live="polite"></section></div>
      <section class="vc-case"><p class="vc-kicker">把术语放回一条路径</p><h3>换一个模型，再切换训练与部署</h3><div class="vc-case-controls"><label>具体案例<select name="vla-concept-case"><option value="openvla">原版 OpenVLA</option><option value="pi05">π₀.₅ 原论文</option><option value="sla">SLA 空间对齐</option></select></label><div class="vc-stage" role="group" aria-label="查看阶段"><button type="button" data-stage="deploy" aria-pressed="true">部署时</button><button type="button" data-stage="train" aria-pressed="false">训练时</button></div></div><div class="vc-trace"></div></section>
      <p class="vc-boundary">本图用于定位概念，未运行模型或机器人。节点仅表示所示案例的关键职能；未单独画出某个词，不表示网络内完全没有该计算。<a href="#axis-role">继续读十维详解 ↓</a></p>`;
    const filter = root.querySelector(".vc-views"), results = root.querySelector(".vc-results"), detail = root.querySelector(".vc-detail"), trace = root.querySelector(".vc-trace");
    function renderFilters() {
      filter.innerHTML = `<button type="button" data-view="all" aria-pressed="${view === "all"}"><strong>全部视角</strong><span>${terms.length} 个术语</span></button>` + dimensions.map(d => `<button type="button" data-view="${d.id}" aria-pressed="${view === d.id}"><strong>${String(d.number).padStart(2, "0")} · ${d.title}</strong><span>${d.question}</span></button>`).join("");
    }
    function renderList() {
      const matched = searchTerms(query, view);
      results.innerHTML = matched.length ? matched.map(t => `<button type="button" data-term="${t.id}" aria-pressed="${selected === t.id}">${escape(t.name)}</button>`).join("") : `<p class="vc-empty">没有匹配的术语。试试中文、英文别名，或切回全部视角。</p>`;
      root.querySelector(".vc-search-summary").textContent = `${view === "all" ? "全部视角" : D[view].title} · 找到 ${matched.length} 个相关术语。视角可以重叠。`;
    }
    function renderDetail() {
      const t = byId[selected];
      detail.innerHTML = `<p class="vc-kicker">${t.views.map(v => D[v].title).join(" · ")}</p><h3>${escape(t.name)}</h3><p class="vc-meaning">${escape(t.meaning)}</p><dl><dt>在系统哪里？</dt><dd>${escape(t.location)}</dd><dt>与什么不同？</dt><dd>${escape(t.contrast)}</dd><dt>用一个例子记住</dt><dd>${escape(t.example)}</dd></dl>`;
    }
    function renderTrace() {
      const c = cases[caseId], data = traceFor(caseId, stage);
      function lane(items, label, cls = "") {
        return `<div class="vc-lane ${cls}"><p class="vc-lane-label">${label}</p><ol>${items.map(n => `<li><button type="button" data-node="${n.id}" data-node-term="${n.terms.find(id => byId[id])}" class="${n.terms.includes(selected) ? "vc-linked" : ""}"><strong>${escape(n.label)}</strong><span>${escape(n.detail)}</span></button></li>`).join("")}</ol></div>`;
      }
      const linked = [...data.main, ...data.supervision, ...data.spatial].some(n => n.terms.includes(selected));
      trace.innerHTML = `<p>${escape(c.intro)}</p>${c.textBranch ? `<p class="vc-branch-note">${escape(c.textBranch)}</p>` : ""}<p class="vc-match">${linked ? `蓝色边框标出「${escape(byId[selected].name)}」关联的位置。点击路径节点可查看相应术语。` : `「${escape(byId[selected].name)}」未在此简图中单独列出；可切换案例或继续查看术语解释。`}</p>` + lane(data.main, stage === "train" ? "策略计算路径（与监督支路相连）" : "部署动作路径") +
        (data.supervision.length ? lane(data.supervision, "训练监督与更新：标签不是部署输入", "vc-training") : "") +
        (data.spatial.length ? lane(data.spatial, "SLA 额外空间监督支路：只在训练时存在", "vc-training vc-spatial") : "") +
        `<p class="vc-case-note">${escape(c.note)}</p><a class="vc-source" href="${c.source}">核对该案例的一手来源 ↗</a>`;
      root.querySelectorAll("[data-stage]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.stage === stage)));
    }
    function select(id, focusSelector) {
      if (!byId[id]) return;
      selected = id; renderList(); renderDetail(); renderTrace();
      if (focusSelector) root.querySelector(focusSelector)?.focus({preventScroll: true});
    }
    root.querySelector("input").addEventListener("input", e => { query = e.target.value; renderList(); });
    filter.addEventListener("click", e => { const b = e.target.closest("[data-view]"); if (b) {
      view = b.dataset.view;
      const matched = searchTerms(query, view);
      if (matched.length && !matched.some(t => t.id === selected)) selected = matched[0].id;
      renderFilters(); renderList(); renderDetail(); renderTrace();
      filter.querySelector(`[data-view="${view}"]`).focus({preventScroll: true});
    } });
    results.addEventListener("click", e => { const b = e.target.closest("[data-term]"); if (b) select(b.dataset.term, `[data-term="${b.dataset.term}"]`); });
    root.querySelector("select").addEventListener("change", e => { caseId = e.target.value; renderTrace(); });
    root.querySelector(".vc-stage").addEventListener("click", e => { const b = e.target.closest("[data-stage]"); if (b) {stage = b.dataset.stage; renderTrace();} });
    trace.addEventListener("click", e => { const b = e.target.closest("[data-node-term]"); if (b) select(b.dataset.nodeTerm, `[data-node="${b.dataset.node}"]`); });
    renderFilters(); renderList(); renderDetail(); renderTrace();
    return root;
  }
  function init() {
    if (current && !current.isConnected) current = null;
    const root = doc.querySelector("[data-vla-concepts]");
    if (root && !root.dataset.initialized) current = mount(root);
  }
  if (typeof document$ !== "undefined") document$.subscribe(init);
  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})(typeof globalThis !== "undefined" ? globalThis : this);
