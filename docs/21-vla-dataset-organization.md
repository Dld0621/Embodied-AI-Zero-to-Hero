# VLA 数据组织系统：从原始采集到训练就绪

> **内容状态：已读，仍有已确认待修项（2026-09-05）。** 逐维归一化、NPZ/LeRobot 格式和图像 HWC/CHW 合同存在错误（补审 F13）；时间对齐亦缺最大偏差约束。不要照抄这些片段开始训练；先对照 [格式与 mock 边界](../examples/robot_foundation_models/smolvla/datasets/README.md)，再用锁定版本的实际 loader 验证。 具体位置与原始来源见 [补充独立审查](reviews/remaining-source-review.md)。

> **目标**：理解机器人模仿学习数据集的完整组织流程——episode 切分、多模态同步、归一化与 feature mapping——并能在统一 PushCube 环境中执行完整的数据采集 pipeline。

---

## 目录

1. [为什么数据组织是 VLA 的第一道门槛](#1-为什么数据组织是-vla-的第一道门槛)
2. [Episode：数据的原子单元](#2-episode数据的原子单元)
3. [多模态时间同步](#3-多模态时间同步)
4. [归一化策略](#4-归一化策略)
5. [Feature Mapping：原始数据 → 模型输入](#5-feature-mapping原始数据--模型输入)
6. [完整代码：PushCube 数据采集器](#6-完整代码pushcube-数据采集器)
7. [与 LeRobot / HuggingFace Datasets 对接](#7-与-lerobot--huggingface-datasets-对接)
8. [常见问题](#8-常见问题)

---

## 1. 为什么数据组织是 VLA 的第一道门槛

VLA 模型的输入通常包含：**RGB 图像** + **语言指令** + **本体感知状态** + **历史动作**，输出是**目标动作**。这些模态的来源、频率、格式各不相同：

| 模态 | 典型来源 | 频率 | 格式 | 关键难点 |
|------|---------|------|------|---------|
| 图像 | 摄像头 | 10–30 Hz | HWC uint8 | 与动作的时间对齐 |
| 语言 | 人工标注 / LLM 生成 | 每 episode 一条 | string | 与 episode 绑定 |
| 动作 | 机器人关节 / 末端位姿 | 10–100 Hz | float32 向量 | 单位与量纲差异大 |
| 状态 | 关节角 / 末端位置 | 10–100 Hz | float32 向量 | 传感器噪声与缺失 |

**核心挑战**：
- **时间对齐**：摄像头 30 Hz，机器人控制 100 Hz，如何配对？
- **Episode 切分**：一次长录制如何切成独立任务？
- **归一化**：不同关节角度范围差异巨大，直接输入网络会导致训练不稳定。
- **Feature mapping**：模型期望的 tensor shape 与原始数据格式往往不一致。

> 本章节以 **PushCube** 统一任务为例，演示从环境交互到训练就绪数据的完整流程。

---

## 2. Episode：数据的原子单元

### 2.1 什么是一个 Episode？

一个 episode = **一次任务尝试**，包含从初始状态到终止条件（成功 / 超时 / 失败）的完整轨迹：

```
Episode
├── task_index: int           # 任务类型编号
├── language_instruction: str  # "push the red cube to the right"
├── frames: List[Frame]        # 时序帧序列
│   ├── timestamp: float
│   ├── observation
│   │   ├── image: (H, W, 3) uint8
│   │   ├── state: (state_dim,) float32
│   │   └── language: str      # 与 episode 共享
│   └── action: (action_dim,) float32
└── metadata
    ├── success: bool
    ├── total_reward: float
    └── duration_sec: float
```

### 2.2 Episode 切分策略

**策略 A：基于环境重置（推荐用于仿真）**

利用环境自身的 `done` / `truncated` 信号自动切分：

```python
while True:
    obs = env.reset()
    episode = {"observations": [], "actions": []}
    for step in range(max_steps):
        action = policy(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        episode["observations"].append(obs)
        episode["actions"].append(action)
        if done or truncated:
            break
    save_episode(episode)
```

**策略 B：基于人工标注（真实机器人录制）**

录制一段长视频后，人工标注每段任务的起止帧：

```python
annotations = [
    {"start": 120, "end": 380, "task": "pick_red_cube"},
    {"start": 420, "end": 690, "task": "place_in_basket"},
]
```

**策略 C：基于阈值自动切分（无显式 done 信号）**

当静止时间超过阈值或物体状态发生突变时切分：

```python
if np.linalg.norm(action) < 0.01 and duration > 2.0:
    end_episode()
```

---

## 3. 多模态时间同步

### 3.1 频率不对齐问题

| 传感器 | 频率 | 每 100ms 帧数 |
|--------|------|--------------|
| 动作指令 | 100 Hz | 10 |
| 状态反馈 | 100 Hz | 10 |
| RGB 摄像头 | 30 Hz | ~3 |
| 语言指令 | 0.1 Hz | 0.01（每 episode 1 条）|

### 3.2 同步策略

**主时钟对齐（Master-clock alignment）**

以一个最高频信号（通常是动作/状态，100 Hz）为主时钟，其他模态通过最近邻或插值对齐：

```python
def align_to_master(master_timestamps, slave_data, slave_timestamps):
    """将 slave 数据对齐到 master 时间轴。"""
    aligned = []
    for t in master_timestamps:
        idx = np.argmin(np.abs(slave_timestamps - t))
        aligned.append(slave_data[idx])
    return aligned
```

**降采样到最低频（Subsampling）**

如果图像是最低频（30 Hz），可将动作和状态降采样到 30 Hz，与图像一一对应：

```python
camera_dt = 1.0 / 30  # 33.3 ms
aligned_actions = actions[::int(camera_dt / control_dt)]
```

**插值（Interpolation）**

对于连续信号（如末端位姿），可用线性插值获取任意时刻的值：

```python
from scipy.interpolate import interp1d
interp = interp1d(state_timestamps, states, axis=0, kind="linear")
state_at_t = interp(camera_timestamp)
```

### 3.3 PushCube 中的同步实践

在 lightweight 仿真环境中，所有模态都在同一 `step()` 调用中生成，天然同步：

```python
obs = env.reset()
for step in range(max_steps):
    img = env.render(size=128)           # 当前状态的图像
    lang = env.get_language_instruction()  # 当前任务的语言
    state = env.get_state_vector()         # 当前状态向量
    action = policy(img, lang)             # 策略输出

    # 所有数据共享同一个 step 索引，天然对齐
    save_frame(step, img, lang, state, action)

    next_obs, reward, done, truncated, info = env.step(action)
```

> **真实机器人注意事项**：真实场景中图像通常有 30–100 ms 的采集延迟，动作有执行延迟。严格的对齐需要记录每帧的**采集时间戳**而非 step 索引。

---

## 4. 归一化策略

### 4.1 为什么必须归一化？

机器人关节角度范围差异巨大：

| 关节 | 典型范围 | 量纲 |
|------|---------|------|
| 手臂 X 位置 | [-0.5, 0.5] m | 米 |
| 手臂 Y 位置 | [-0.5, 0.5] m | 米 |
| 关节角 1 | [-3.14, 3.14] rad | 弧度 |
| 关节角 2 | [0, 1.57] rad | 弧度 |

如果直接输入网络，大尺度的特征会压制小尺度的特征，导致梯度失衡。

### 4.2 常用归一化方法

**Min-Max 归一化（最常用）**

```python
def minmax_normalize(data, min_val, max_val):
    return 2.0 * (data - min_val) / (max_val - min_val) - 1.0  # 映射到 [-1, 1]

def minmax_denormalize(normalized, min_val, max_val):
    return (normalized + 1.0) / 2.0 * (max_val - min_val) + min_val
```

**Standard 标准化（Z-score）**

```python
def standard_normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)
```

**Quantile 归一化（对异常值鲁棒）**

```python
from sklearn.preprocessing import QuantileTransformer
transformer = QuantileTransformer(output_distribution="normal")
normalized = transformer.fit_transform(data)
```

### 4.3 统计量计算：训练集 vs 全集

**关键原则**：归一化参数（min/max 或 mean/std）必须仅在**训练集**上计算，然后固定应用于验证集和测试集。

```python
# 计算训练集统计量
train_actions = np.concatenate([ep["actions"] for ep in train_episodes])
action_min = train_actions.min(axis=0)
action_max = train_actions.max(axis=0)
action_mean = train_actions.mean(axis=0)
action_std = train_actions.std(axis=0)

# 保存统计量（用于推理时复用）
np.savez("action_stats.npz", min=action_min, max=action_max, mean=action_mean, std=action_std)

# 应用归一化
for ep in all_episodes:
    ep["actions_normalized"] = minmax_normalize(ep["actions"], action_min, action_max)
```

### 4.4 图像归一化

```python
# 原始图像: uint8 [0, 255] -> float32 [0, 1]
img_float = img.astype(np.float32) / 255.0

# ImageNet 预训练模型的标准化
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_normalized = (img_float - mean) / std
```

### 4.5 PushCube 归一化示例

```python
class PushCubeNormalizer:
    def __init__(self):
        self.state_min = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0], dtype=np.float32)
        self.state_max = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0], dtype=np.float32)
        self.action_min = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_max = np.array([1.0, 1.0], dtype=np.float32)

    def normalize_state(self, state):
        return 2.0 * (state - self.state_min) / (self.state_max - self.state_min) - 1.0

    def normalize_action(self, action):
        return 2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0

    def denormalize_action(self, normalized):
        return (normalized + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min
```

---

## 5. Feature Mapping：原始数据 → 模型输入

### 5.1 模型期望的输入格式

不同的 VLA 模型对输入格式有不同要求：

| 模型 | 图像格式 | 语言格式 | 动作格式 |
|------|---------|---------|---------|
| SmolVLA | `(B, C, H, W)` float32 | token IDs `(B, L)` int64 | 连续向量 `(B, action_dim)` |
| ACT | `(B, T, C, H, W)` float32 | token IDs `(B, L)` int64 | 动作块 `(B, T, action_dim)` |
| Diffusion Policy | `(B, T, C, H, W)` float32 | 可选嵌入 | 动作块 `(B, T, action_dim)` |

### 5.2 图像 Feature Mapping

```python
def prepare_image(img_np, size=128):
    """
    输入: (H, W, 3) uint8 numpy array (PushCube render 输出)
    输出: (1, 3, H, W) float32 tensor
    """
    # 1. 转置到 CHW
    img_chw = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
    # 2. 归一化到 [0, 1]
    img_float = img_chw.astype(np.float32) / 255.0
    # 3. 添加 batch 维度
    img_batch = np.expand_dims(img_float, axis=0)  # (1, 3, H, W)
    return torch.tensor(img_batch)
```

### 5.3 语言 Feature Mapping

```python
VOCAB = {
    "<pad>": 0, "push": 1, "the": 2, "red": 3, "green": 4, "yellow": 5,
    "cube": 6, "to": 7, "right": 8, "left": 9, "top": 10, "bottom": 11,
    "and": 12, "center": 13,
}

def prepare_language(text, max_len=10):
    """
    输入: "push the red cube to the right"
    输出: (1, max_len) int64 tensor，padding 到固定长度
    """
    words = text.lower().replace(".", "").split()
    tokens = [VOCAB.get(w, 0) for w in words]
    # Padding
    tokens += [VOCAB["<pad>"]] * (max_len - len(tokens))
    tokens = tokens[:max_len]
    return torch.tensor([tokens], dtype=torch.long)
```

### 5.4 动作 Feature Mapping

```python
def prepare_action(action_np, normalizer):
    """
    输入: (action_dim,) float32 numpy array
    输出: (1, action_dim) float32 tensor，已归一化
    """
    normalized = normalizer.normalize_action(action_np)
    return torch.tensor(np.expand_dims(normalized, axis=0), dtype=torch.float32)
```

### 5.5 状态 Feature Mapping（可选，用于策略融合）

```python
def prepare_state(state_np, normalizer):
    """
    输入: (state_dim,) float32 numpy array
    输出: (1, state_dim) float32 tensor，已归一化
    """
    normalized = normalizer.normalize_state(state_np)
    return torch.tensor(np.expand_dims(normalized, axis=0), dtype=torch.float32)
```

---

## 6. 完整代码：PushCube 数据采集器

以下代码演示从 PushCube 环境采集数据、组织成 episode、归一化并保存为训练就绪格式的完整流程：

```python
"""
PushCube 数据采集器
===================
从统一 PushCube 环境采集 demonstration 数据，组织为标准 episode 格式，
并保存为可直接用于训练的 HDF5 / npz 文件。
"""

import json
import h5py
import numpy as np
from pathlib import Path
from unified_pushcube_env import PushCubeEnv


class PushCubeDatasetCollector:
    def __init__(self, n_episodes=100, seed=42):
        self.n_episodes = n_episodes
        self.rng = np.random.RandomState(seed)

    def collect_heuristic_demo(self, env, seed):
        """使用启发式策略采集一条 episode。"""
        obs = env.reset(seed=seed)
        lang = env.get_language_instruction()

        frames = []
        for step in range(env.max_steps):
            img = env.render(size=128)  # (128, 128, 3) uint8
            state = env.get_state_vector()  # (8,) float32

            # 启发式策略：先靠近 cube，再推向 target
            arm = obs["arm_pos"]
            cube = obs["cube_pos"]
            target = obs["target_pos"]
            dist_to_cube = np.linalg.norm(cube - arm)

            if dist_to_cube > 0.08:
                dir_to_cube = cube - arm
                dir_to_cube /= np.linalg.norm(dir_to_cube) + 1e-6
                action = dir_to_cube * 0.8
            else:
                dir_to_target = target - cube
                dir_to_target /= np.linalg.norm(dir_to_target) + 1e-6
                action = dir_to_target * 0.8
            action = np.clip(action, -1.0, 1.0)

            frames.append({
                "image": img,
                "state": state,
                "language": lang,
                "action": action,
            })

            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        return {
            "language_instruction": lang,
            "frames": frames,
            "success": info["is_success"],
            "n_steps": len(frames),
        }

    def collect_all(self):
        episodes = []
        for ep_idx in range(self.n_episodes):
            env = PushCubeEnv()
            ep = self.collect_heuristic_demo(env, seed=ep_idx)
            episodes.append(ep)
            if (ep_idx + 1) % 20 == 0:
                print(f"  Collected {ep_idx + 1}/{self.n_episodes} episodes")
        return episodes

    def compute_normalization_stats(self, episodes, train_ratio=0.8):
        """仅在训练集上计算归一化统计量。"""
        n_train = int(len(episodes) * train_ratio)
        train_episodes = episodes[:n_train]

        all_states = np.concatenate([ep["frames"][i]["state"] for ep in train_episodes for i in range(len(ep["frames"]))])
        all_actions = np.concatenate([ep["frames"][i]["action"] for ep in train_episodes for i in range(len(ep["frames"]))])

        stats = {
            "state": {
                "min": all_states.min(axis=0).tolist(),
                "max": all_states.max(axis=0).tolist(),
                "mean": all_states.mean(axis=0).tolist(),
                "std": all_states.std(axis=0).tolist(),
            },
            "action": {
                "min": all_actions.min(axis=0).tolist(),
                "max": all_actions.max(axis=0).tolist(),
                "mean": all_actions.mean(axis=0).tolist(),
                "std": all_actions.std(axis=0).tolist(),
            },
        }
        return stats

    def save_hdf5(self, episodes, stats, output_path="pushcube_dataset.hdf5"):
        """保存为 HDF5 格式（适合大规模数据集）。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            # 保存统计量
            stats_group = f.create_group("stats")
            for key, val in stats.items():
                grp = stats_group.create_group(key)
                for stat_name, stat_val in val.items():
                    grp.create_dataset(stat_name, data=stat_val)

            # 保存 episodes
            episodes_group = f.create_group("episodes")
            for ep_idx, ep in enumerate(episodes):
                ep_grp = episodes_group.create_group(f"episode_{ep_idx:04d}")
                ep_grp.attrs["language_instruction"] = ep["language_instruction"]
                ep_grp.attrs["success"] = ep["success"]
                ep_grp.attrs["n_steps"] = ep["n_steps"]

                n_frames = len(ep["frames"])
                images = np.stack([f["image"] for f in ep["frames"]])  # (T, H, W, 3)
                states = np.stack([f["state"] for f in ep["frames"]])   # (T, state_dim)
                actions = np.stack([f["action"] for f in ep["frames"]]) # (T, action_dim)

                ep_grp.create_dataset("images", data=images, compression="gzip")
                ep_grp.create_dataset("states", data=states, compression="gzip")
                ep_grp.create_dataset("actions", data=actions, compression="gzip")

        print(f"Dataset saved to {output_path}")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Total frames: {sum(ep['n_steps'] for ep in episodes)}")
        print(f"  Success rate: {sum(ep['success'] for ep in episodes) / len(episodes) * 100:.1f}%")

    def save_lerobot_format(self, episodes, stats, output_dir="pushcube_lerobot"):
        """保存为 LeRobot 兼容格式（parquet + mp4）。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存统计量
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # 保存为简单 npz（LeRobot 的简化替代）
        for ep_idx, ep in enumerate(episodes):
            images = np.stack([f["image"] for f in ep["frames"]])
            states = np.stack([f["state"] for f in ep["frames"]])
            actions = np.stack([f["action"] for f in ep["frames"]])
            np.savez(
                output_dir / f"episode_{ep_idx:04d}.npz",
                images=images,
                states=states,
                actions=actions,
                language=ep["language_instruction"],
                success=ep["success"],
            )

        print(f"LeRobot-style dataset saved to {output_dir}")


# ------------------------------------------------------------------
# 使用示例
# ------------------------------------------------------------------
if __name__ == "__main__":
    collector = PushCubeDatasetCollector(n_episodes=100, seed=42)

    print("Collecting episodes...")
    episodes = collector.collect_all()

    print("\nComputing normalization stats...")
    stats = collector.compute_normalization_stats(episodes)

    print("\nSaving to HDF5...")
    collector.save_hdf5(episodes, stats, output_path="../datasets/pushcube_100eps.hdf5")

    print("\nSaving to LeRobot-style format...")
    collector.save_lerobot_format(episodes, stats, output_dir="../datasets/pushcube_lerobot")
```

---

## 7. 与 LeRobot / HuggingFace Datasets 对接

### 7.1 LeRobot Dataset 格式

LeRobot 使用 **HuggingFace `datasets`** 库管理数据，核心结构：

```python
from datasets import Dataset, Features, Image, Value

features = Features({
    "observation.image": Image(),           # PIL Image
    "observation.state": Value("float32"),   # 状态向量（序列化）
    "action": Value("float32"),              # 动作向量（序列化）
    "task": Value("string"),                 # 语言指令
    "episode_index": Value("int64"),         # episode 编号
    "frame_index": Value("int64"),           # 帧编号
    "timestamp": Value("float32"),           # 时间戳
    "next.done": Value("bool"),              # 是否终止
})
```

### 7.2 从 HDF5 加载到 LeRobot

```python
from datasets import Dataset
import h5py

def hdf5_to_lerobot(hdf5_path):
    data = {
        "observation.image": [],
        "observation.state": [],
        "action": [],
        "task": [],
        "episode_index": [],
        "frame_index": [],
    }

    with h5py.File(hdf5_path, "r") as f:
        for ep_name in f["episodes"]:
            ep = f["episodes"][ep_name]
            lang = ep.attrs["language_instruction"]
            n_steps = ep.attrs["n_steps"]

            for i in range(n_steps):
                data["observation.image"].append(ep["images"][i])
                data["observation.state"].append(ep["states"][i])
                data["action"].append(ep["actions"][i])
                data["task"].append(lang)
                data["episode_index"].append(int(ep_name.split("_")[1]))
                data["frame_index"].append(i)

    return Dataset.from_dict(data)
```

### 7.3 Feature Mapping 到 LeRobot Policy

LeRobot 的 `SmolVLAPolicy` 期望输入格式：

```python
# 从 dataset 采样
sample = dataset[0]

# 构造模型输入
inputs = {
    "observation.images": {
        "front": torch.tensor(sample["observation.image"]).unsqueeze(0),  # (1, C, H, W)
    },
    "observation.state": torch.tensor(sample["observation.state"]).unsqueeze(0),  # (1, state_dim)
    "task": [sample["task"]],  # list of strings
}

# 前向传播
action = policy.select_action(inputs)
```

---

## 8. 常见问题

**Q: 训练集和测试集的 episode 是否需要来自不同的任务？**

A: 理想情况下，测试集应包含**未见过的任务变体**（如不同颜色、不同目标位置），以评估泛化能力。如果任务单一，至少应使用不同的随机种子生成测试集。

**Q: 动作归一化后，模型输出的动作如何映射回真实动作？**

A: 使用与训练时相同的统计量进行反归一化：`action_real = denormalize(model_output)`。统计量文件（`stats.npz` 或 `stats.json`）必须与模型权重一起保存和加载。

**Q: 语言指令每 episode 只有一条，如何在帧级别重复？**

A: 语言指令在 episode 内是常量，可以在 dataloader 中通过 `repeat` 或 `expand` 操作在 batch 维度广播。无需在存储层面重复保存。

**Q: HDF5 vs TFRecord vs Parquet，选哪个？**

| 格式 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| HDF5 | 简单、压缩好、NumPy 原生 | 不支持流式读取 | 中小规模 (<100GB) |
| TFRecord | TensorFlow 生态原生、流式 | 需要 protobuf 定义 | 大规模、TF 训练 |
| Parquet | 列式存储、查询快、生态广 | 图像需要额外处理 | LeRobot、数据分析 |

---

## 参考文献

1. LeRobot Documentation: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
2. HuggingFace Datasets: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
3. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
4. Zhan et al., "ACT: Action Chunking with Transformers", arXiv 2022
