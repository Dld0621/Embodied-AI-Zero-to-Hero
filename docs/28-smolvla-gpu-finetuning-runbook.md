# SmolVLA 微调：版本、数据、训练与重载验收

**更新与来源核查：2026-11-15。** 本页核对官方配置与训练接口，整理一条可逐关检查的实验路线。此次没有安装训练栈、下载完整权重、运行 GPU 微调或重跑历史评估。命令是待在目标环境验证的步骤，不能当作训练成功回执。

先读 [VLA 系统课](learning-core/09-vla-systems.md)、[数据组织](21-vla-dataset-organization.md)与 [部署测量](20-vla-deployment-guide.md)。所有本仓库路径命令从仓库根目录执行；安装 LeRobot 的命令在独立环境执行。

## 1. 先分清三个不同的模型/实验

| 对象 | 是什么 | 能支持的结论 |
| --- | --- | --- |
| 官方 `lerobot/smolvla_base` | 450M 级 VLA，视觉语言主干条件化动作专家 | 可研究与微调的预训练起点 |
| 仓库 LightweightVLA | 小型 CNN 与语言条件教学策略 | 小模型训练/评估流程的示例 |
| `finetune.py --mock` | 随机初始化的本地替代网络 | 训练管线冒烟检查，不能证明官方 SmolVLA 被加载 |

“模型总参数 450M”“可训练参数 100M”“全部参数微调”“LoRA”是不同合同。冻结部分权重不会自动变成 LoRA；下载 450M 模型也不代表 450M 参数都参与优化。

官方 [v0.4.1 SmolVLA 配置](https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/policies/smolvla/configuration_smolvla.py)默认冻结视觉编码器，并开启 `train_expert_only` 与 `train_state_proj`。最终以本次实例中 `requires_grad` 的参数数目、名称清单与优化器分组为准。[官方权重的固定版本配置](https://huggingface.co/lerobot/smolvla_base/blob/d560bdd24ed1230588eac43aeec23fe5beb73089/config.json)也保留这些开关。

## 2. 选择一份软件合同，避免跨版本拼教程

本仓库锁文件仍选择 LeRobot **0.4.1**；本页用该 tag 的官方源码解释路径与参数，不把它称为最新版本。2026-11-15 核查时，官方 `main` 文档的稳定版入口标为 **v0.7.0**；迁移应在独立实验环境比较并重新验证，不能只更新包号。参见 [当前官方文档](https://huggingface.co/docs/lerobot/main/smolvla)。

| 固定项目 | 需要保存的证据 |
| --- | --- |
| Python、PyTorch、CUDA 与驱动 | 版本、GPU 型号及可用内存 |
| LeRobot | 发布版本或源码 commit |
| 基础权重 | Hub revision、本地快照与文件校验和 |
| 数据 | repo/revision、split 清单、校验和 |
| 处理器 | 图像/语言/状态预处理和动作逆变换配置 |
| 训练 | 完整解析配置、可训练参数与随机种子 |

独立环境中可按仓库选定版本安装：

```bash
python -m pip install 'lerobot[smolvla]==0.4.1'
python -m pip check
lerobot-train --help
```

这是版本选择步骤，不能保证任意操作系统/GPU 都能解析或运行。视频解码依赖也要按目标环境检查。v0.4.1 的包声明见 [官方 pyproject](https://github.com/huggingface/lerobot/blob/v0.4.1/pyproject.toml)。

该版本的策略路径是：

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
```

不能沿用旧文档中的 `lerobot.common.policies...` 路径。检查导入路径只证明模块可用，尚未证明权重加载或数据匹配。[v0.4.1 策略实现](https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/policies/smolvla/modeling_smolvla.py)。

显存取决于精度、优化器状态、冻结设置、图像数量/分辨率、动作块、batch、梯度累计和缓存。先测一个 batch 的峰值和吞吐量，再扩大训练预算；删除旧版“8 GB 必够”和“A100 30–60 分钟”的无依据保证。

## 3. 数据关：格式正确之后还要验证语义

```text
原始演示 → 划分 episode / 物体 / 场景
        → 只在训练集拟合统计量
        → 转换并用所选版本的官方 loader 读取
        → 检查真实 batch 和动作逆变换
        → 才开始微调
```

对一个 batch 至少打印并保存：相机 key 与图像 shape/dtype/range；语言 token 与 mask；状态与动作通道；动作块时间索引及 padding mask。检查动作维度来自真实数据特征，不把 checkpoint 的补齐维度当作机器人实际关节数。

LeRobot 包版本与数据格式版本是两个编号。v0.4.1 的 [数据实现](https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/datasets/lerobot_dataset.py)声明 `CODEBASE_VERSION = "v3.0"`，不能根据包号猜成 v2.1。先检查自己数据的 `codebase_version` 与实际 loader 行为。

**PushCube 路线：** 参见 [仓库数据说明](../examples/robot_foundation_models/smolvla/datasets/README.md)。canonical 演示、mock Parquet 和真实 LeRobot 格式要放不同目录。转换器已取消隐式 mock 回退，并拒绝覆盖已有目录；缺少官方依赖会明确报错。mock 产物标记为教学格式，不能交给官方训练。官方路径的写入器关闭已有离线控制流程测试，真实 v3 loader 往返还需验证。

官方 SO100/SO101 演示与本地二维 PushCube 不是同一任务。把官方数据替换成二维 `[dx, dy]` 数据，需要重新验证状态、相机、动作空间、归一化和评估器，不能只替换目录。

## 4. 固定权重快照与训练数据

Hub 的 `main` 会变化。正式实验用完整 revision 下载基础权重到本地，并保存实际解析出的配置；同样固定其依赖的视觉语言模型与 processor 资源。可使用官方 [snapshot_download 的 revision 参数](https://huggingface.co/docs/huggingface_hub/guides/download)。

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/smolvla_base",
    revision="d560bdd24ed1230588eac43aeec23fe5beb73089",
    local_dir="artifacts/smolvla_base_d560bdd",
)
```

该 revision 是本页核对配置使用的可定位快照，不是“最新”保证，也未在本轮验证完整权重与目标训练栈联调。若改用其他 revision，保存新配置并重新做单 batch 验证。

数据应先拆分为独立训练/验证/测试版本；直接引用整个公共数据集不能证明其中已完成留出划分。若使用 episode 子集，须确认所用版本如何聚合统计量，不能让未选中的验证/测试轨迹进入训练统计。

## 5. 用官方 CLI 开始一轮小训练

下面模板使用 v0.4.1 官方训练入口与配置字段。路径与训练 repo ID 必须替换成已实际验证的对象；`200` 步与 batch `2` 只是小规模管线验收预算，没有任务成功含义。

```bash
lerobot-train \
  --policy.path=artifacts/smolvla_base_d560bdd \
  --dataset.repo_id=your-account/your-train-only-dataset \
  --output_dir=outputs/train/smolvla-contract-check \
  --job_name=smolvla_contract_check \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=200 \
  --wandb.enable=false
```

训练入口与基础参数核对 [v0.4.1 官方 recipe](https://github.com/huggingface/lerobot/blob/v0.4.1/docs/source/smolvla.mdx)。本模板缩小预算并关闭上传，不声称已经执行。

关键参数要看解析后的配置，而不是仅看命令行文本：

| 要改什么 | v0.4.1 中应核对什么 |
| --- | --- |
| 学习率 | 使用策略 preset 时看 `policy.optimizer_lr`；自定义 optimizer 时看完整优化器/调度器合同 |
| 本地数据目录 | `dataset.root` 指向数据根目录本身，不是其父目录 |
| 动作块/执行前缀 | `chunk_size` 与 `n_action_steps`，不能令后者大于前者 |
| 可训练子模块 | 冻结开关、实际梯度与优化器分组 |
| 混合精度 | 当前版本支持的 `use_amp` 等配置及硬件能力，不照抄其他版本的 dtype 参数 |
| 日志与保存 | save/log 频率、保存的解析配置、权重与优化器状态 |

参见 [训练配置源码](https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/configs/train.py)、[数据配置源码](https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/configs/default.py)和 [策略基础配置](https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/configs/policies.py)。不要用无版本依据的顶层 `--lr`。本地 `finetune.py` 包装器的合同测试不能替代官方 CLI 实际解析与训练。

### 本仓库包装器的修正范围

`examples/robot_foundation_models/smolvla/finetune.py` 已按上述版本合同修正学习率、本地 root 和输出目录处理，默认关闭模型上传。Hub 数据集需显式传 `--dataset_repo_id owner/name`；本地路径用 `--dataset_dir`，缺失目录会提前报错。新增的离线测试只验证参数和路径处理，不代表实际 GPU 训练或官方 CLI 完整解析已执行。

## 6. 训练关：下降的 loss 说明了什么？

先保存有限 loss、梯度、实际学习率、显存峰值和每步时间。做小样本拟合可以检查管线是否传递了学习信号；它不能证明泛化，也不能证明语言起作用。

随后固定训练预算，在留出 episode 上看动作误差、不同阶段误差和各通道饱和比例。flow 训练目标、MSE、L1 或 token 交叉熵的数值尺度不同，不能跨模型直接按 loss 大小排名。

至少做两种消融：正确/交换/空指令，以及同样数据上的简单动作分块 BC。比较时保持观测、动作合同、训练/测试划分和评价预算一致。

## 7. 重载关：保存目录不等于可用 checkpoint

结束训练后，在全新进程中：

1. 读取权重、配置、processor 与训练统计量。
2. 对固定观测做动作预测，保存归一化与物理单位的输出。
3. 对比保存前/重载后的输出；随机采样策略固定噪声或比较规定的统计量。
4. 重置模型内部历史/动作队列，再检查跨 episode 不残留旧动作。
5. 验证续训恢复的是同一优化器、调度器、步数与随机状态。

SmolVLA 的 `select_action` 可能使用内部动作队列；调用一次不一定代表重新采样完整动作块。测延迟和重载一致性时要说明调用的是哪个接口，并按 [官方策略实现](https://github.com/huggingface/lerobot/blob/v0.4.1/src/lerobot/policies/smolvla/modeling_smolvla.py)处理 reset。

## 8. 闭环关：先定义成功，再运行测试

每条留出 episode 保存初始条件、完整轨迹、正确指令、动作与执行时序、终止原因、干预和成功判据。语言实验用相同初始场景与随机条件对照，分开报告“选中目标”和“完成任务”。

灵巧操作还需报告接触建立、抓起、搬运、放置、释放与最终稳定；拿起一次不能替代完整任务完成。模型的离线误差与闭环结果分别列出。

本页不提供未经验证的硬件命令。官方 robot type、端口、相机与控制模式只有在对应机器人配置和停止流程通过验证后才有意义。部署测量参见 [部署指南](20-vla-deployment-guide.md)。

## 9. 仓库现有结果可以怎么读？

| 已有材料 | 可以确认什么 | 缺少什么 |
| --- | --- | --- |
| [LightweightVLA 汇总](../results/benchmarks/lightweight_vla_closed_loop.json) | 提交文件记录 195,266 参数、20 次评估、0% 正确指令任务成功、65% 目标选择 | 本轮未重跑；汇总的 commit 为 unknown，不能当作当前结果 |
| [GPU 训练历史文件](../examples/robot_foundation_models/smolvla/models/smolvla_pushcube_gpu/training_history_gpu.json) | 存在历史记录 | 不等于完整权重、逐 episode 评估与独立复现链 |
| [历史训练配置](../examples/robot_foundation_models/smolvla/models/smolvla_pushcube_20260804_100517/checkpoints/000100/pretrained_model/train_config.json) | 某次保存的特征、参数与配置可阅读 | 目录名与配置不证明本轮训练、重载或任务成功 |

不沿用旧页缺少原始证据的 A100 时长与“full 450M”表述。历史 500/10K 步报告的完整权重与逐 episode 数据不在本页验收范围，详见 [既有审查](reviews/remaining-source-review.md)。更新文档日期不会把历史实验升级成新的复现实验。

## 10. 实践验收与下一步

交付环境清单、权重 revision、数据与 split、真实 batch 检查、训练配置、可训练参数、完整日志、重载对照和逐 episode 评估。缺哪一项，就明确该项尚未完成。

先排查单位、预处理、时间索引、mask、语言捷径与评估器，再做数据规模曲线或更复杂的 RL 后训练。收集更多数据、延长训练或换更大模型都是可检验的假设，不能提前写成必然改善。
