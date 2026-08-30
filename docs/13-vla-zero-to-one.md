# VLA 零到一：兼容入口

本路径为历史链接保留，权威课程已经升级为双语专项：

- [VLA Zero to One](specializations/vla-zero-to-one.md)
- [VLA 从零到一](specializations/vla-zero-to-one-cn.md)
- [VLA 与 WAM 专项总览](specializations/README_CN.md)

新课程不绑定单一模型或固定硬件配置，而是从数据、动作和控制合同出发，依次讲解：

1. 图像、语言、本体状态与动作的时间对齐；
2. 视觉编码、语言条件和多模态融合；
3. 单步回归、动作块、离散 Action Token、Diffusion 与 Flow Matching；
4. 从头训练、参数高效微调与全量微调的适用条件；
5. 数据量、算力、时延、动作多模态性和语言泛化要求如何影响算法选择；
6. Tiny-set Overfit、语言/视觉消融、闭环任务评估与安全边界；
7. 进入科研时需要的匹配基线、消融矩阵和否证条件。

## 可执行入口

```bash
python scripts/run_knowledge_map.py --path-to learning-vla --lang zh
python scripts/run_pipeline.py --show vla-policy
python scripts/run_pipeline.py --run vla-policy
python scripts/select_vla_wam_algorithm.py --goal multimodal-action --compute single-gpu --data multi-task --latency soft
```

本地 PushCube 路径只是教学型闭环基线，不是对 OpenVLA、π0、SmolVLA 或其他大规模 VLA 的复现，也不支持 SOTA 或真机性能结论。请按[验证规范](VALIDATION.md)区分接口、合成 Smoke Test、教学基准与真机证据。
