#!/usr/bin/env python3
"""
dataset_utils.py -- VLA 微调数据工具函数库

本文件包含微调 VLA 模型所需的所有数据处理工具，包括：
- LIBERO 数据集加载（支持本地 HDF5 和 HuggingFace RLDS 两种方式）
- 动作归一化 / 反归一化
- 图像预处理
- DataLoader 创建

依赖：
    pip install torch torchvision pillow numpy
    # 如果使用本地 LIBERO 数据，还需要：
    pip install libero
    # 如果使用 RLDS 格式数据，还需要：
    pip install tensorflow rlds

用法：
    from dataset_utils import get_libero_dataset, normalize_action, image_transforms
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# 常量定义
# ============================================================

# LIBERO 动作维度：7 维 (dx, dy, dz, droll, dpitch, dyaw, gripper)
ACTION_DIM = 7

# OpenVLA 默认图像尺寸
IMAGE_SIZE = 224

# LIBERO 各 benchmark 对应的 unnorm_key
# 用于在推理时将模型输出的归一化动作还原到真实动作空间
LIBERO_UNNORM_KEYS: Dict[str, str] = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
    "libero_90": "libero_90",
}

# LIBERO 各 benchmark 中每个 task 的最大步数（用于评估时的截断）
LIBERO_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# OpenVLA 的 prompt 模板
# 为什么用 "In: ... \nOut:"？这是 OpenVLA 预训练时使用的 prompt 格式，
# 微调时必须保持一致，否则模型无法正确理解输入。
OPENVLA_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


# ============================================================
# 动作归一化 / 反归一化
# ============================================================

def normalize_action(
    action: np.ndarray | torch.Tensor,
    action_mean: np.ndarray | torch.Tensor,
    action_std: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    对动作进行 Z-score 归一化：normalized = (action - mean) / std

    为什么需要归一化？
    - 不同动作维度的物理单位不同（位置 vs 角度 vs 夹爪），数值范围差异很大
    - 归一化后所有维度都在相似的尺度上，有利于模型学习
    - 这是 OpenVLA 预训练时使用的归一化方式，微调时必须保持一致

    参数：
        action: 原始动作，形状为 (7,) 或 (batch, 7)
        action_mean: 动作均值，形状为 (7,)
        action_std: 动作标准差，形状为 (7,)

    返回：
        归一化后的动作，形状与输入相同
    """
    if isinstance(action, np.ndarray):
        return (action - action_mean) / (action_std + 1e-6)
    else:
        return (action - torch.as_tensor(action_mean, device=action.device)) / (
            torch.as_tensor(action_std, device=action.device) + 1e-6
        )


def unnormalize_action(
    normalized_action: np.ndarray | torch.Tensor,
    action_mean: np.ndarray | torch.Tensor,
    action_std: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    将归一化的动作还原到真实动作空间：action = normalized * std + mean

    参数：
        normalized_action: 归一化后的动作
        action_mean: 动作均值
        action_std: 动作标准差

    返回：
        真实尺度的动作
    """
    if isinstance(normalized_action, np.ndarray):
        return normalized_action * (action_std + 1e-6) + action_mean
    else:
        return normalized_action * (torch.as_tensor(action_std, device=normalized_action.device) + 1e-6) + torch.as_tensor(
            action_mean, device=normalized_action.device
        )


def get_action_stats(
    actions: List[np.ndarray] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算动作的统计量（均值和标准差）。

    在开始训练之前，需要先遍历整个数据集计算所有动作的均值和标准差。
    这些统计量会在训练时用于归一化，在推理时用于反归一化。

    参数：
        actions: 动作列表，每个元素形状为 (7,)，或形状为 (N, 7) 的数组

    返回：
        (mean, std): 形状均为 (7,)
    """
    if isinstance(actions, list):
        actions = np.stack(actions, axis=0)
    assert actions.ndim == 2 and actions.shape[1] == ACTION_DIM, (
        f"动作形状应为 (N, {ACTION_DIM})，但得到 {actions.shape}"
    )
    mean = actions.mean(axis=0)
    std = actions.std(axis=0)
    # 防止某个维度的标准差为 0（该维度可能没有变化）
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


# ============================================================
# 图像预处理
# ============================================================

def get_train_image_transforms(
    image_size: int = IMAGE_SIZE,
    random_crop: bool = True,
    color_jitter: bool = True,
) -> transforms.Compose:
    """
    获取训练时的图像预处理 Pipeline。

    训练时的数据增强很重要，因为：
    1. 防止过拟合 — 仿真环境中每个场景的视角固定，模型容易记住特定像素位置
    2. 提高泛化能力 — 轻微的裁剪和颜色变化让模型关注物体的形状和位置，而非纹理细节

    注意：OpenVLA 官方使用 random_crop + resize 的方式做增强。
    如果训练时使用了 crop，推理时也需要 center_crop 来保持一致性。

    参数：
        image_size: 输出图像尺寸
        random_crop: 是否使用随机裁剪
        color_jitter: 是否使用颜色抖动

    返回：
        torchvision transforms pipeline
    """
    transform_list = []

    if random_crop:
        # 先 resize 到稍大的尺寸，再随机裁剪到目标尺寸
        # crop_scale=0.9 表示裁剪区域面积是原图面积的 90%
        # 这与 OpenVLA 官方的 crop_scale 一致
        transform_list.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.9, 1.0),  # 裁剪比例范围
                interpolation=transforms.InterpolationMode.LANCZOS,
            )
        )
    else:
        # 不做增强时，直接 resize
        transform_list.append(
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.LANCZOS,
            )
        )

    if color_jitter:
        # 轻微的颜色抖动，不会改变物体外观的语义信息
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            )
        )

    transform_list.append(transforms.ToTensor())
    # ImageNet 的归一化参数，因为 OpenVLA 的视觉编码器使用 ImageNet 预训练权重
    transform_list.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    return transforms.Compose(transform_list)


def get_eval_image_transforms(
    image_size: int = IMAGE_SIZE,
    center_crop: bool = True,
) -> transforms.Compose:
    """
    获取评估时的图像预处理 Pipeline。

    如果训练时使用了 random_crop，评估时需要 center_crop 来保持分布一致。
    这就是 OpenVLA 官方评估脚本中 center_crop 参数的含义。

    参数：
        image_size: 输出图像尺寸
        center_crop: 是否中心裁剪（应与训练时是否使用 random_crop 对应）

    返回：
        torchvision transforms pipeline
    """
    transform_list = []

    if center_crop:
        # center_crop: 先放大再裁剪中间部分
        # 这与训练时 RandomResizedCrop 的 scale=(0.9, 1.0) 对应
        transform_list.append(
            transforms.Resize(
                int(image_size / 0.9),
                interpolation=transforms.InterpolationMode.LANCZOS,
            )
        )
        transform_list.append(transforms.CenterCrop(image_size))
    else:
        transform_list.append(
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.LANCZOS,
            )
        )

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    return transforms.Compose(transform_list)


def image_transforms() -> transforms.Compose:
    """
    默认的图像预处理函数（兼容性包装）。
    直接返回训练时的 transform。
    """
    return get_train_image_transforms()


# ============================================================
# LIBERO 数据集类
# ============================================================

class LIBERODataset(Dataset):
    """
    LIBERO 仿真基准的数据集类。

    支持两种数据加载方式：
    - 方式 A（本地 HDF5）：从本地安装的 LIBERO 包中直接读取 HDF5 演示数据
    - 方式 B（HuggingFace RLDS）：从 OpenVLA 提供的 modified RLDS 数据集加载

    每个数据样本包含：
    - image: 机器人第三人称视角的 RGB 图像
    - instruction: 自然语言任务描述
    - action: 7 维连续动作 (dx, dy, dz, droll, dpitch, dyaw, gripper)
    """

    def __init__(
        self,
        data_root: str | Path,
        benchmark_name: str = "libero_spatial",
        use_local_libero: bool = True,
        image_aug: bool = True,
        chunk_size: int = 1,
        max_transitions: Optional[int] = None,
    ) -> None:
        """
        参数：
            data_root: 数据根目录
                - 本地模式：LIBERO 数据集的安装路径（通常是 ~/.cache/libero）
                - RLDS 模式：RLDS 数据集的根目录
            benchmark_name: benchmark 名称，支持 libero_spatial/object/goal/10
            use_local_libero: 是否使用本地 LIBERO 安装（True）还是 RLDS 格式（False）
            image_aug: 训练时是否使用图像增强
            chunk_size: Action Chunking 大小（默认 1 表示不 chunk）
                Action Chunking 是一种技巧：让模型一次预测多步动作，而不是只预测一步。
                好处：减少累积误差，提高长期任务的连贯性。
            max_transitions: 最多加载多少条 transition（用于调试和快速测试）
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.benchmark_name = benchmark_name
        self.use_local_libero = use_local_libero
        self.chunk_size = chunk_size
        self.image_size = IMAGE_SIZE

        # 图像预处理
        if image_aug:
            self.transform = get_train_image_transforms(image_size=IMAGE_SIZE)
        else:
            self.transform = get_eval_image_transforms(image_size=IMAGE_SIZE, center_crop=False)

        # 加载数据
        self.samples: List[Dict[str, Any]] = []
        if use_local_libero:
            self._load_from_local_libero(max_transitions)
        else:
            self._load_from_rlds(max_transitions)

        print(f"[LIBERODataset] 加载了 {len(self.samples)} 个训练样本 "
              f"(benchmark={benchmark_name}, chunk_size={chunk_size})")

    def _load_from_local_libero(self, max_transitions: Optional[int] = None) -> None:
        """
        方式 A：从本地 LIBERO 安装加载数据。

        LIBERO 使用 HDF5 格式存储演示数据。每个 benchmark 包含多个 task，
        每个 task 有多组人类演示轨迹，每组轨迹是一系列 (observation, action) 对。

        数据存储结构：
            data_root/
              libero_spatial/
                demo.hdf5    # 包含所有 task 的演示数据
              libero_object/
                demo.hdf5
              ...
        """
        try:
            from libero.libero import get_libero_path, benchmark as libero_benchmark
        except ImportError:
            raise ImportError(
                "无法导入 libero 包。请先安装：\n"
                "  pip install libero\n"
                "或者使用 --use_rlds 参数从 HuggingFace 加载 RLDS 格式数据。"
            )

        # 获取 benchmark
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        if self.benchmark_name not in benchmark_dict:
            raise ValueError(
                f"未知的 benchmark '{self.benchmark_name}'。"
                f"可选值：{list(benchmark_dict.keys())}"
            )

        task_suite = benchmark_dict[self.benchmark_name]()
        num_tasks = task_suite.n_tasks

        print(f"[LIBERODataset] 正在从本地加载 {self.benchmark_name} ({num_tasks} 个任务)...")

        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            instruction = task.language
            # 获取该任务的所有演示数据（初始状态列表）
            init_states = task_suite.get_task_init_states(task_id)

            for demo_idx, init_state in enumerate(init_states):
                try:
                    # 从 HDF5 文件加载轨迹
                    # 注意：这里需要通过 libero 的 API 访问演示数据
                    demo_path = get_libero_path(self.benchmark_name)
                    hdf5_file = os.path.join(demo_path, f"{self.benchmark_name}_demo.hdf5")

                    if not os.path.exists(hdf5_file):
                        # 尝试其他可能的文件名
                        hdf5_file = os.path.join(demo_path, "demo.hdf5")

                    if not os.path.exists(hdf5_file):
                        print(f"  警告：未找到 HDF5 文件 {hdf5_file}，跳过任务 {task_id}")
                        continue

                    import h5py
                    with h5py.File(hdf5_file, "r") as f:
                        episode_key = f"task_{task_id}/demo_{demo_idx}"
                        if episode_key not in f:
                            continue

                        num_steps = len(f[f"{episode_key}/actions"])
                        for step_t in range(num_steps - self.chunk_size + 1):
                            # 读取观测
                            obs_key = f"{episode_key}/obs"
                            # LIBERO 存储图像为 uint8 数组
                            img_data = f[f"{obs_key}/agentview_rgb"][step_t]
                            if isinstance(img_data, h5py.Dataset):
                                img_data = img_data[()]

                            # 读取动作
                            action = f[f"{episode_key}/actions"][step_t]

                            self.samples.append({
                                "image": img_data,       # (H, W, 3) uint8
                                "instruction": instruction,
                                "action": np.array(action, dtype=np.float32),
                                "task_id": task_id,
                                "demo_idx": demo_idx,
                                "step_t": step_t,
                            })

                            if max_transitions and len(self.samples) >= max_transitions:
                                return

                except Exception as e:
                    print(f"  警告：加载任务 {task_id} 的演示 {demo_idx} 时出错: {e}")
                    continue

    def _load_from_rlds(self, max_transitions: Optional[int] = None) -> None:
        """
        方式 B：从 OpenVLA 提供的 modified RLDS 数据集加载。

        OpenVLA 官方提供了一个预处理好的 RLDS 版本的 LIBERO 数据，
        在 HuggingFace 上可以下载：openvla/modified_libero_rlds

        RLDS 是一种结构化的数据格式，专为机器人学习设计。

        加载流程：
        1. 使用 tensorflow_datasets 加载 RLDS 数据
        2. 遍历每个 episode，提取 image、instruction、action
        3. 存储到 self.samples 列表
        """
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "无法导入 tensorflow_datasets。请先安装：\n"
                "  pip install tensorflow tensorflow_datasets\n"
                "或者使用本地 LIBERO 安装（默认方式）。"
            )

        dataset_name = f"modified_{self.benchmark_name}"
        data_dir = str(self.data_root)

        print(f"[LIBERODataset] 正在从 RLDS 加载 {dataset_name}...")

        try:
            ds = tfds.load(
                dataset_name,
                data_dir=data_dir,
                split="train",
                shuffle_files=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"无法加载 RLDS 数据集 '{dataset_name}'。\n"
                f"请确保数据已下载到 {data_dir}。\n"
                f"错误详情：{e}\n"
                f"提示：可以从 HuggingFace 下载 openvla/modified_libero_rlds"
            )

        for episode in tfds.as_numpy(ds):
            steps = episode["steps"]
            instruction = episode["episode_metadata"]["language"].decode("utf-8")
            num_steps = len(steps)

            for step_t in range(num_steps - self.chunk_size + 1):
                obs = steps[step_t]["observation"]
                # RLDS 中的图像可能存储在不同的 key 下
                if "image_primary" in obs:
                    img_data = obs["image_primary"]
                elif "agentview_rgb" in obs:
                    img_data = obs["agentview_rgb"]
                else:
                    # 尝试常见的 key
                    img_key = [k for k in obs.keys() if "image" in k.lower()][0]
                    img_data = obs[img_key]

                action = steps[step_t]["action"]

                self.samples.append({
                    "image": img_data,       # (H, W, 3) uint8
                    "instruction": instruction,
                    "action": np.array(action, dtype=np.float32).flatten()[:ACTION_DIM],
                    "task_id": 0,
                    "demo_idx": 0,
                    "step_t": step_t,
                })

                if max_transitions and len(self.samples) >= max_transitions:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个训练样本，包含：
        - pixel_values: 预处理后的图像张量 (3, H, W)
        - input_ids: tokenized prompt
        - attention_mask: attention mask
        - labels: tokenized action labels（用于计算 loss）
        - action: 原始动作值（用于统计和 debug）
        """
        sample = self.samples[idx]

        # ---- 图像处理 ----
        # 从 numpy array 转为 PIL Image
        if isinstance(sample["image"], np.ndarray):
            image = Image.fromarray(sample["image"].astype(np.uint8))
        else:
            image = sample["image"]
        image = image.convert("RGB")

        # 应用预处理（增强 + normalize）
        pixel_values = self.transform(image)

        # ---- 构造 prompt ----
        instruction = sample["instruction"].strip().lower()
        prompt = OPENVLA_PROMPT_TEMPLATE.format(instruction=instruction)

        # ---- 动作处理 ----
        # 如果使用 action chunking，拼接 chunk_size 个连续动作
        action = sample["action"].copy()
        if self.chunk_size > 1 and idx + self.chunk_size - 1 < len(self.samples):
            # 检查这些样本是否来自同一条轨迹
            base = self.samples[idx]
            actions_chunk = []
            valid_chunk = True
            for c in range(self.chunk_size):
                s = self.samples[idx + c]
                if (s["task_id"] != base["task_id"] or
                    s["demo_idx"] != base["demo_idx"] or
                    s["step_t"] != base["step_t"] + c):
                    valid_chunk = False
                    break
                actions_chunk.append(s["action"])
            if valid_chunk:
                action = np.concatenate(actions_chunk, axis=0)
            # 如果 chunk 不合法（跨越了不同轨迹），只使用单个动作

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "action": torch.from_numpy(action).float(),
            "task_id": sample["task_id"],
        }


class LIBEROCollator:
    """
    自定义的 Collator，用于将 LIBERO 样本批次化。

    主要功能：
    1. 将图像堆叠为 batch tensor
    2. 对 prompt 进行 tokenization
    3. 对 action 进行 tokenization（使用 Action Tokenizer）
    4. 构造 attention mask 和 labels
    """

    def __init__(
        self,
        processor: Any,
        action_tokenizer: Any = None,
        max_length: int = 4096,
        pad_token_id: int = 0,
    ) -> None:
        """
        参数：
            processor: HuggingFace AutoProcessor，用于 tokenization
            action_tokenizer: 动作 tokenizer，用于将连续动作离散化为 token ID
                如果为 None，则直接使用连续动作的 MSE loss（简化版）
            max_length: token 序列最大长度
            pad_token_id: padding token ID
        """
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将一个 batch 的样本整理为模型可用的格式。

        注意这里有两种模式：
        1. 有 action_tokenizer：使用 OpenVLA 官方的 token-based 训练方式
           - 动作被量化为 token ID
           - 模型输出 token logits，通过 cross-entropy loss 训练

        2. 无 action_tokenizer（简化版）：直接使用连续动作的 MSE loss
           - 更容易理解和调试
           - 适合教学目的
        """
        pixel_values = torch.stack([s["pixel_values"] for s in batch], dim=0)
        prompts = [s["prompt"] for s in batch]
        actions = torch.stack([s["action"] for s in batch], dim=0)

        if self.action_tokenizer is not None:
            # 模式 1：使用 action tokenizer（官方方式）
            # Tokenize prompt
            input_ids_list = []
            for prompt in prompts:
                tokens = self.processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                input_ids_list.append(tokens["input_ids"][0])

            # Tokenize actions
            action_labels_list = []
            for action in actions:
                action_ids = self.action_tokenizer.tokenize(action.numpy())
                action_labels_list.append(torch.tensor(action_ids))

            # 构造完整的 input_ids 和 labels
            batch_input_ids = []
            batch_labels = []
            for input_ids, action_ids in zip(input_ids_list, action_labels_list):
                full_input_ids = torch.cat([input_ids, action_ids[:-1]], dim=0)
                full_labels = torch.cat([
                    torch.full_like(input_ids, -100),  # prompt 部分不计算 loss
                    action_ids,
                ], dim=0)

                # 截断到最大长度
                if full_input_ids.shape[0] > self.max_length:
                    full_input_ids = full_input_ids[:self.max_length]
                    full_labels = full_labels[:self.max_length]

                batch_input_ids.append(full_input_ids)
                batch_labels.append(full_labels)

            # Padding
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            padded_labels = torch.nn.utils.rnn.pad_sequence(
                batch_labels,
                batch_first=True,
                padding_value=-100,
            )
            attention_mask = padded_input_ids.ne(self.pad_token_id).long()

            return {
                "input_ids": padded_input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "labels": padded_labels,
                "actions": actions,  # 保留原始动作用于统计
            }
        else:
            # 模式 2：简化版，直接使用连续动作（教学用）
            # Tokenize prompt（不含 action）
            tokenized = self.processor.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "pixel_values": pixel_values,
                "actions": actions,
            }


# ============================================================
# 自定义 JSONL 数据集类
# ============================================================

class CustomJSONLDataset(Dataset):
    """
    自定义 JSONL 格式的 VLA 数据集。

    每行 JSON 格式：
    {"image_path": "path/to/image.jpg", "instruction": "pick up the cup",
     "action": [dx, dy, dz, droll, dpitch, dyaw, gripper]}

    适用于学习者使用自己收集的机器人数据进行微调。
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = "",
        image_aug: bool = True,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        chunk_size: int = 1,
    ) -> None:
        """
        参数：
            jsonl_path: JSONL 文件路径
            image_root: 图像根目录（如果 image_path 是相对路径）
            image_aug: 是否使用图像增强
            action_mean: 预计算的动作均值（如果 None 则在 __init__ 时自动计算）
            action_std: 预计算的动作标准差
            chunk_size: Action Chunking 大小
        """
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.chunk_size = chunk_size

        # 图像预处理
        if image_aug:
            self.transform = get_train_image_transforms(image_size=IMAGE_SIZE)
        else:
            self.transform = get_eval_image_transforms(image_size=IMAGE_SIZE, center_crop=False)

        # 加载 JSONL 数据
        self.samples: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                # 验证必要字段
                if "image_path" not in sample:
                    raise ValueError(f"第 {line_idx + 1} 行缺少 'image_path' 字段")
                if "instruction" not in sample:
                    raise ValueError(f"第 {line_idx + 1} 行缺少 'instruction' 字段")
                if "action" not in sample:
                    raise ValueError(f"第 {line_idx + 1} 行缺少 'action' 字段")

                action = np.array(sample["action"], dtype=np.float32)
                if action.shape != (ACTION_DIM,):
                    raise ValueError(
                        f"第 {line_idx + 1} 行的动作维度应为 {ACTION_DIM}，但得到 {action.shape}"
                    )

                self.samples.append(sample)

        # 计算或使用提供的动作统计量
        if action_mean is not None and action_std is not None:
            self.action_mean = action_mean
            self.action_std = action_std
        else:
            print("[CustomJSONLDataset] 正在计算动作统计量...")
            all_actions = np.stack([s["action"] for s in self.samples], axis=0)
            self.action_mean, self.action_std = get_action_stats(all_actions)
            print(f"  动作均值: {self.action_mean}")
            print(f"  动作标准差: {self.action_std}")

        print(f"[CustomJSONLDataset] 加载了 {len(self.samples)} 个样本 "
              f"(从 {self.jsonl_path})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 加载图像
        image_path = self.image_root / sample["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        # 构造 prompt
        instruction = sample["instruction"].strip().lower()
        prompt = OPENVLA_PROMPT_TEMPLATE.format(instruction=instruction)

        # 归一化动作
        action = normalize_action(
            np.array(sample["action"], dtype=np.float32),
            self.action_mean,
            self.action_std,
        )

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "action": torch.from_numpy(action).float(),
        }


# ============================================================
# DataLoader 创建
# ============================================================

def create_vla_dataloader(
    dataset: Dataset,
    processor: Optional[Any] = None,
    action_tokenizer: Optional[Any] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    max_length: int = 4096,
    pad_token_id: int = 0,
) -> DataLoader:
    """
    创建 VLA 训练的 DataLoader。

    参数：
        dataset: 数据集实例
        processor: HuggingFace AutoProcessor（用于 LIBERO 数据的 collation）
        action_tokenizer: 动作 tokenizer（可选）
        batch_size: 批次大小
        shuffle: 是否打乱顺序
        num_workers: 数据加载线程数（设为 0 适用于 RLDS 数据，因为 TFDS 自行管理并行）
        max_length: token 序列最大长度
        pad_token_id: padding token ID

    返回：
        PyTorch DataLoader
    """
    if processor is not None:
        collator = LIBEROCollator(
            processor=processor,
            action_tokenizer=action_tokenizer,
            max_length=max_length,
            pad_token_id=pad_token_id,
        )
    else:
        # 简单的 default collate
        collator = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,  # 加速 CPU 到 GPU 的数据传输
        drop_last=True,    # 丢弃最后不完整的 batch，保证训练稳定
    )

    return dataloader


def get_libero_dataset(
    data_root: str | Path,
    benchmark_name: str = "libero_spatial",
    use_local_libero: bool = True,
    image_aug: bool = True,
    chunk_size: int = 1,
    max_transitions: Optional[int] = None,
) -> LIBERODataset:
    """
    获取 LIBERO 数据集的便捷函数。

    参数：
        data_root: 数据根目录
        benchmark_name: benchmark 名称
        use_local_libero: 使用本地 LIBERO（True）或 RLDS 格式（False）
        image_aug: 是否使用图像增强
        chunk_size: Action Chunking 大小
        max_transitions: 最大加载样本数

    返回：
        LIBERODataset 实例
    """
    return LIBERODataset(
        data_root=data_root,
        benchmark_name=benchmark_name,
        use_local_libero=use_local_libero,
        image_aug=image_aug,
        chunk_size=chunk_size,
        max_transitions=max_transitions,
    )


def save_dataset_statistics(
    action_mean: np.ndarray,
    action_std: np.ndarray,
    output_dir: str | Path,
    unnorm_key: str = "libero_spatial",
) -> Path:
    """
    保存数据集统计量到 JSON 文件。

    这些统计量在推理时被 OpenVLA 用于将模型输出反归一化到真实动作空间。
    文件名固定为 dataset_statistics.json，这是 OpenVLA 评估代码约定的名称。

    参数：
        action_mean: 动作均值
        action_std: 动作标准差
        output_dir: 输出目录
        unnorm_key: 归一化 key（通常是 benchmark 名称）

    返回：
        保存的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        unnorm_key: {
            "action": {
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "max": (action_mean + 3 * action_std).tolist(),
                "min": (action_mean - 3 * action_std).tolist(),
            }
        }
    }

    output_path = output_dir / "dataset_statistics.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[save_dataset_statistics] 统计量已保存到 {output_path}")
    return output_path


def load_dataset_statistics(checkpoint_dir: str | Path) -> Dict[str, Any]:
    """
    从 checkpoint 目录加载数据集统计量。

    参数：
        checkpoint_dir: checkpoint 目录路径

    返回：
        统计量字典
    """
    stats_path = Path(checkpoint_dir) / "dataset_statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"未找到数据集统计量文件: {stats_path}\n"
            f"该文件在微调时应自动保存到 checkpoint 目录。"
        )

    with open(stats_path, "r") as f:
        stats = json.load(f)

    return stats


# ============================================================
# 辅助函数
# ============================================================

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    将四元数转换为轴角表示。

    LIBERO 环境中的机器人末端执行器姿态使用四元数表示，
    但动作空间使用轴角表示，所以需要转换。

    参数：
        quat: 四元数 (x, y, z, w)，形状为 (4,)

    返回：
        轴角表示 (ax, ay, az)，形状为 (3,)
    """
    # 限制四元数 w 分量范围
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)

    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        # 零旋转，返回零向量
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    将夹爪动作从 [0, 1] 归一化到 [-1, +1]。

    OpenVLA 输出的夹爪动作在 [0, 1] 范围（0=关闭, 1=打开），
    但 LIBERO 仿真环境期望 [-1, +1]（-1=打开, +1=关闭）。

    参数：
        action: 动作向量，最后一个维度是夹爪
        binarize: 是否将夹爪动作二值化为 -1 或 +1

    返回：
        归一化后的动作
    """
    action = action.copy()
    action[-1] = 2.0 * action[-1] - 1.0
    if binarize:
        action[-1] = np.sign(action[-1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    翻转夹爪动作的符号。

    LIBERO 的 RLDS 数据加载器会将夹爪动作取反以统一格式
    （0=关闭, 1=打开），所以在执行时需要翻回来。

    参数：
        action: 动作向量

    返回：
        夹爪符号翻转后的动作
    """
    action = action.copy()
    action[-1] = action[-1] * -1.0
    return action


if __name__ == "__main__":
    # 测试代码：验证工具函数是否正常工作
    print("=" * 60)
    print("dataset_utils.py 自检")
    print("=" * 60)

    # 1. 测试动作归一化
    print("\n[测试] 动作归一化...")
    actions = np.random.randn(100, ACTION_DIM).astype(np.float32)
    mean, std = get_action_stats(actions)
    print(f"  均值: {mean}")
    print(f"  标准差: {std}")

    normalized = normalize_action(actions[:5], mean, std)
    unnormalized = unnormalize_action(normalized, mean, std)
    error = np.abs(actions[:5] - unnormalized).max()
    print(f"  归一化 → 反归一化 最大误差: {error:.6f}")
    assert error < 1e-5, "归一化/反归一化不可逆！"

    # 2. 测试图像预处理
    print("\n[测试] 图像预处理...")
    transform = get_train_image_transforms()
    dummy_image = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
    result = transform(dummy_image)
    print(f"  输入图像尺寸: {dummy_image.size}")
    print(f"  输出张量形状: {result.shape}")
    assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"图像尺寸不正确: {result.shape}"

    # 3. 测试统计量保存/加载
    print("\n[测试] 统计量保存/加载...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset_statistics(mean, std, tmpdir, unnorm_key="test_benchmark")
        loaded = load_dataset_statistics(tmpdir)
        loaded_mean = np.array(loaded["test_benchmark"]["action"]["mean"])
        assert np.allclose(mean, loaded_mean), "统计量保存/加载不一致！"
        print(f"  保存和加载的均值一致: {np.allclose(mean, loaded_mean)}")

    # 4. 测试 gripper 工具函数
    print("\n[测试] Gripper 动作处理...")
    action = np.array([0.01, -0.02, 0.005, 0.0, 0.0, 0.01, 0.8])
    normalized_g = normalize_gripper_action(action, binarize=True)
    print(f"  原始夹爪: {action[-1]:.2f} → 归一化后: {normalized_g[-1]:.2f}")
    inverted = invert_gripper_action(normalized_g)
    print(f"  翻转后: {inverted[-1]:.2f}")

    print("\n" + "=" * 60)
    print("所有自检通过！")
    print("=" * 60)
