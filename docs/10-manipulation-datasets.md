# 灵巧操作开源数据集

> **内容状态：已读，仍有已确认待修项（2026-09-05）。** robomimic 被误列为 Shadow Hand 数据；DexCap 的人手动作捕捉、重定向命令和真实机器人轨迹需要分开（补审 F06）。本页分类与用途推论尚待重写，不能按表直接选择训练数据。 具体位置与原始来源见 [补充独立审查](reviews/remaining-source-review.md)。

> 精选可直接下载的开源数据集，按**是否包含灵巧手数据**分类。重点标注数据格式、下载方式和与 Retargeting/训练的关系。

## 数据集总览

### 包含灵巧手（多指）数据的数据集

| 数据集 | 灵巧手 | 真实/仿真 | 大小 | 格式 | 许可证 |
|--------|--------|----------|------|------|--------|
| **DexGraspNet** | Shadow Hand | 仿真 | ~数GB | npz (抓取姿态) | CC-BY-NC 4.0 |
| **robomimic (SHAPES等)** | Shadow Hand | 仿真 | 数GB | HDF5 | MIT |
| **ManiSkill2/3** | Shadow Hand | 仿真 | 数GB (环境生成) | HDF5 / RLDS | Apache 2.0 |
| **DEXCap** | LEAP Hand (双臂) | **真实** | 数十GB | HDF5 | MIT |
| **DexMimicGen** | LEAP Hand (双手) | 仿真+真实 | 数十GB | HDF5 | — |
| **UniDexGrasp** | Shadow/LEAP/Allegro | 仿真 | 数十GB | npz | CC-BY-NC 4.0 |

### 仅含夹爪数据的通用操作数据集

| 数据集 | 夹爪类型 | 大小 | 格式 | 许可证 |
|--------|---------|------|------|--------|
| **Open X-Embodiment** | 多种夹爪 | 数TB | RLDS (TFRecord) | CC-BY 4.0 |
| **BridgeData V2** | WidowX 二指 | 数百GB | pkl / RLDS | MIT |
| **LIBERO** | Franka 二指 | 数GB | HDF5 | CC-BY 4.0 |
| **RLBench** | Franka 二指 | 按需生成 | numpy / h5py | — |
| **MimicGen** | Franka 二指 | 数十GB | HDF5 | CC-BY 4.0 |
| **DROID** | 多种二指 | 数百GB | 自定义 | — |

### 人手姿态数据集（用于手部建模/检测训练）

| 数据集 | 内容 | 大小 | 许可证 |
|--------|------|------|--------|
| **InterHand2.6M** | 260万帧 3D 手部关键点 | 数十GB | CC-BY-NC 4.0 |

---

## 1. DexGraspNet（北京大学 / NeurIPS 2023）

**最大的 Shadow Hand 仿真抓取数据集，适合学习灵巧抓取先验。**

| 属性 | 信息 |
|------|------|
| **论文** | DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset (NeurIPS 2023) |
| **GitHub** | [https://github.com/PKU-EPIC/DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) |
| **项目页** | [https://pku-epic.github.io/DexGraspNet/](https://pku-epic.github.io/DexGraspNet/) |
| **大小** | 132 万+ 抓取姿态，5355 个物体，~数 GB |
| **内容** | 抓取姿态（关节角度）、物体点云/网格、物理稳定性验证 |
| **灵巧手** | Shadow Hand（也支持 Allegro 和 MANO） |
| **格式** | npz |
| **许可证** | CC-BY-NC 4.0 |

**下载方式**：
```bash
git clone https://github.com/PKU-EPIC/DexGraspNet.git
cd DexGraspNet
python scripts/download_dataset.py
```

**与 Retargeting 的关系**：
- 可作为 Learning-based retargeting 的先验数据
- 分析 Shadow Hand 关节限位和可达空间
- `docs/02-retargeting-taxonomy.md` 中 Learning-based 方法的数据来源参考

---

## 2. robomimic 多任务数据集（Stanford / CoRL 2021）

**获取 Shadow Hand 仿真操作数据最方便的途径。**

| 属性 | 信息 |
|------|------|
| **论文** | What Matters in Learning from Offline Human Demonstrations for Robot Manipulation (CoRL 2021) |
| **GitHub** | [https://github.com/ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) |
| **包含任务** | lift, can, square, tool_hang, transport |
| **灵巧手** | Shadow Hand + Franka Panda（部分任务仅有 Franka 二指夹爪） |
| **格式** | HDF5（robosuite 兼容） |
| **许可证** | MIT |

**下载方式**：
```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic

# 下载 SHAPES + lift 任务（Shadow Hand）
python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph

# 下载全部任务
python robomimic/scripts/download_datasets.py --tasks all --dataset_types ph
```

**与 Retargeting 的关系**：
- Shadow Hand 关节角序列可直接分析 retargeting 精度
- 可用于训练 retargeting 的学习-based 优化器
- HDF5 格式含 `robot0_eef_pos` 和 `robot0_joint_pos`，可直接提取

---

## 3. DEXCap（Stanford/UCSD / 2024）

**唯一包含真实 LEAP Hand 双手操作的开源数据集。**

| 属性 | 信息 |
|------|------|
| **论文** | DexCap: Scalable and Portable Mocap Data Collection (arXiv:2403.07788) |
| **数据集** | [HuggingFace: DexCap-Data](https://huggingface.co/datasets/chenwangj/DexCap-Data) |
| **大小** | 数十 GB |
| **内容** | RGB 图像、深度图、手部关节位置(3D)、6-DoF 位姿、点云 |
| **灵巧手** | **LEAP Hand**（双手）+ WidowX 250 双臂 |
| **格式** | HDF5（robomimic 兼容） |
| **许可证** | MIT |

**下载方式**：
```python
# HuggingFace 下载
from huggingface_hub import snapshot_download
snapshot_download(repo_id="chenwangj/DexCap-Data", local_dir="dexcap_data")
```

**与 Retargeting 的关系**：
- 包含 **人手 → LEAP Hand 的真实 retargeting 数据**
- 可直接用于训练/评估 retargeting 算法
- 双手数据，含左右手镜像

---

## 4. ManiSkill2/3（UCSD/清华 / ICLR 2023, RSS 2025）

**GPU 并行仿真基准，支持 Shadow Hand 任务。**

| 属性 | 信息 |
|------|------|
| **论文** | ManiSkill2 (ICLR 2023) / ManiSkill3 (RSS 2025) |
| **GitHub** | [https://github.com/haosulab/ManiSkill](https://github.com/haosulab/ManiSkill) |
| **安装** | `pip install mani_skill` |
| **灵巧手** | Shadow Hand（部分任务），Franka 等多种机器人 |
| **格式** | HDF5 / RLDS / 自定义 |
| **许可证** | Apache 2.0 (代码)，CC-BY-NC 4.0 (资产) |

**与 Retargeting 的关系**：
- Shadow Hand 任务可用于验证 retargeting 后的控制策略
- GPU 并行生成大规模训练数据

---

## 5. Open X-Embodiment（Google DeepMind / ICRA 2024）

**最大的机器人操作数据集合集（100万+轨迹），但不含灵巧手数据。**

| 属性 | 信息 |
|------|------|
| **论文** | Open X-Embodiment: Robotic Learning Datasets and RT-X Models (ICRA 2024) |
| **GitHub** | [https://github.com/google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment) |
| **大小** | 数 TB（60+ 子数据集，22 种机器人） |
| **内容** | RGB 图像、关节角/末端位姿、语言指令 |
| **格式** | RLDS (TFRecord) |
| **许可证** | CC-BY 4.0 |

**下载方式**（需要 `gsutil`）：
```bash
pip install gsutil
# 列出可用数据集
gsutil ls gs://gdm-robotics-open-x-embodiment/

# 下载指定数据集（示例：fractal20220817_data）
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/fractal20220817_data ~/tensorflow_datasets/
```

**说明**：主要使用二指夹爪，**不含多指灵巧手数据**。但 RLDS 格式和 VLA 训练 pipeline 可参考。

---

## 6. BridgeData V2（UC Berkeley / CoRL 2023）

| 属性 | 信息 |
|------|------|
| **GitHub** | [https://github.com/rail-berkeley/bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2) |
| **下载** | [https://rail.eecs.berkeley.edu/datasets/bridge_release/data/](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/) |
| **大小** | 数百 GB |
| **格式** | JPEG/PNG + pkl / RLDS |
| **许可证** | MIT |

---

## 7. LIBERO（UT Austin / NeurIPS 2023）

| 属性 | 信息 |
|------|------|
| **GitHub** | [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| **HuggingFace** | [LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) |
| **大小** | 130 个任务，数 GB |
| **格式** | HDF5（128x128 图像 + 低维状态） |
| **许可证** | CC-BY 4.0 |

---

## 8. InterHand2.6M（Meta / ECCV 2020）

**最大的人手 3D 姿态数据集，可用于训练手部检测/建模。**

| 属性 | 信息 |
|------|------|
| **GitHub** | [https://github.com/facebookresearch/InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M) |
| **项目页** | [https://mks0601.github.io/InterHand2.6M/](https://mks0601.github.io/InterHand2.6M/) |
| **大小** | 260 万帧，数十 GB |
| **内容** | RGB 图像 + 3D 手部关键点标注 + MANO 参数 |
| **格式** | 图片 + JSON/npz 标注 |
| **许可证** | CC-BY-NC 4.0（需申请） |

**与 Retargeting 的关系**：
- 3D 关键点标注可用于训练/评估手部检测器
- MANO 参数可作为人手运动的先验分布
- 双手交互数据可用于双手 retargeting 研究

---

## 下载脚本

所有可直接下载的数据集已整理到 `datasets/download_datasets.py`：

```bash
cd datasets/
python download_datasets.py
```

> **注意**：部分数据集（Open X-Embodiment, DEXCap）需要 `gsutil` 或 HuggingFace CLI，脚本中会自动检测并提示安装。
