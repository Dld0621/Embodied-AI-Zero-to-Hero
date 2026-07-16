#!/usr/bin/env python3
"""
evaluate_libero.py -- 在 LIBERO 仿真环境上评估微调后的 OpenVLA

================================================================================
  Embodied-AI-Zero-to-Hero 教学项目 · Stage 4: 微调实践
================================================================================

本脚本独立于训练过程，可以加载任意 checkpoint 在 LIBERO benchmark 上进行
闭环（closed-loop）评估。

评估流程：
  1. 加载微调后的 OpenVLA checkpoint
  2. 遍历 benchmark 中的每个 task
  3. 每个 task 运行 N 个 episode
  4. 在每个 episode 中，模型实时观测环境图像并预测动作
  5. 统计并打印每个 task 和总体的成功率

依赖：
  pip install torch transformers pillow numpy imageio
  pip install libero          # LIBERO 仿真环境
  pip install tensorflow       # 图像预处理（resize_image 函数需要）

用法示例：
  # 基本评估
  python evaluate_libero.py \\
      --checkpoint_path ./checkpoints/openvla-libero-spatial/checkpoint-final \\
      --benchmark libero_spatial \\
      --num_trials_per_task 20

  # 评估并保存 rollout 视频
  python evaluate_libero.py \\
      --checkpoint_path ./checkpoints/openvla-libero-spatial/checkpoint-final \\
      --benchmark libero_spatial \\
      --num_trials_per_task 20 \\
      --save_videos \\
      --video_dir ./rollouts

  # 使用 4-bit 量化加载（省显存）
  python evaluate_libero.py \\
      --checkpoint_path ./checkpoints/openvla-libero-spatial/checkpoint-final \\
      --benchmark libero_spatial \\
      --load_in_4bit

  # 使用 WandB 记录结果
  python evaluate_libero.py \\
      --checkpoint_path ./checkpoints/openvla-libero-spatial/checkpoint-final \\
      --benchmark libero_spatial \\
      --use_wandb \\
      --wandb_project vla-eval

================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from PIL import Image

# ============================================================
# 常量
# ============================================================

ACTION_DIM = 7
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# OpenVLA v0.1 的 system prompt
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# LIBERO benchmark 各 task 的最大步数
LIBERO_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ============================================================
# 辅助函数（来自 OpenVLA 官方 libero_utils.py）
# ============================================================

def set_seed_everywhere(seed: int) -> None:
    """设置全局随机种子，确保实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    将四元数转换为轴角表示。

    LIBERO 中机器人的末端执行器姿态用四元数表示 (x, y, z, w)，
    而动作空间使用轴角表示。需要转换以构造 observation。

    参数：
        quat: 四元数 (x, y, z, w)

    返回：
        轴角 (ax, ay, az)
    """
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_image(img: np.ndarray, resize_size: Tuple[int, int]) -> np.ndarray:
    """
    调整图像尺寸。

    使用与 OpenVLA 训练数据一致的 resize 方式（先 JPEG 编码再解码再 resize），
    以保持输入分布的一致性。

    参数：
        img: numpy array, 形状 (H, W, C)
        resize_size: 目标尺寸 (H, W)

    返回：
        resize 后的图像
    """
    import tensorflow as tf
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def get_libero_image(obs: Dict, resize_size: int) -> np.ndarray:
    """
    从 LIBERO 环境的观测中提取并预处理图像。

    重要：需要旋转 180 度以匹配训练数据的预处理方式。
    这是因为 LIBERO 仿真环境的相机视角与训练数据中的存储方式不同。

    参数：
        obs: 环境观测字典
        resize_size: 目标尺寸（正方形边长）

    返回：
        预处理后的 RGB 图像
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 旋转 180 度
    img = resize_image(img, (resize_size, resize_size))
    return img


def get_libero_env(task, model_family: str, resolution: int = 256):
    """
    初始化 LIBERO 仿真环境。

    参数：
        task: LIBERO task 对象
        model_family: 模型族名称（这里只支持 "openvla"）
        resolution: 环境渲染分辨率

    返回：
        (env, task_description) 元组
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # 固定环境种子
    return env, task_description


def get_libero_dummy_action(model_family: str) -> List[float]:
    """
    获取空操作（no-op）动作。

    在仿真开始的前几步，物体仍在下落，需要执行空操作等待物体稳定。
    """
    return [0, 0, 0, 0, 0, 0, -1]


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    将夹爪动作从 [0, 1] 归一化到 [-1, +1]。

    OpenVLA 输出 gripper ∈ [0,1]（0=关闭, 1=打开），
    LIBERO 期望 gripper ∈ [-1,+1]（-1=打开, +1=关闭）。
    """
    action = action.copy()
    action[-1] = 2.0 * action[-1] - 1.0
    if binarize:
        action[-1] = np.sign(action[-1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    翻转夹爪动作符号。

    RLDS 数据加载器统一了 gripper 格式（0=close, 1=open），
    但 LIBERO 环境使用相反的约定（-1=open, +1=close），
    所以执行前需要翻转回来。
    """
    action = action.copy()
    action[-1] = action[-1] * -1.0
    return action


def save_rollout_video(
    rollout_images: List[np.ndarray],
    episode_idx: int,
    success: bool,
    task_description: str,
    video_dir: str = "./rollouts",
) -> str:
    """
    保存 episode 回放视频。

    参数：
        rollout_images: 图像列表
        episode_idx: episode 编号
        success: 是否成功
        task_description: 任务描述
        video_dir: 视频保存目录

    返回：
        视频文件路径
    """
    os.makedirs(video_dir, exist_ok=True)
    # 文件名：安全化处理任务描述
    safe_name = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = os.path.join(
        video_dir,
        f"{DATE_TIME}--ep={episode_idx}--success={success}--task={safe_name}.mp4"
    )
    writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        writer.append_data(img)
    writer.close()
    print(f"  回放视频已保存: {mp4_path}")
    return mp4_path


# ============================================================
# 图像预处理（center_crop）
# ============================================================

def center_crop_image(
    image: np.ndarray,
    crop_scale: float = 0.9,
    target_size: int = 224,
) -> np.ndarray:
    """
    中心裁剪图像并 resize 回原始尺寸。

    如果训练时使用了 random_crop（image_aug=True），
    评估时需要 center_crop 来保持输入分布一致。

    crop_scale=0.9 表示裁剪面积是原图面积的 90%。
    具体来说，裁剪后每边的长度 = 原长度 * sqrt(0.9) ≈ 0.949。

    参数：
        image: numpy array (H, W, 3), uint8
        crop_scale: 裁剪比例
        target_size: 最终输出尺寸

    返回：
        处理后的图像
    """
    import tensorflow as tf

    image = tf.convert_to_tensor(image)
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)

    # 计算裁剪区域
    new_size = tf.clip_by_value(tf.sqrt(crop_scale), 0, 1)
    height_offset = (1 - new_size) / 2
    width_offset = (1 - new_size) / 2
    bbox = tf.stack([height_offset, width_offset, height_offset + new_size, width_offset + new_size])
    bbox = tf.expand_dims(bbox, 0)

    image = tf.image.crop_and_resize(
        tf.expand_dims(image, 0), bbox, tf.range(1), (target_size, target_size)
    )
    image = tf.clip_by_value(image[0], 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return image.numpy()


# ============================================================
# 模型加载
# ============================================================

def load_vla_model(
    checkpoint_path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[Any, Any]:
    """
    加载微调后的 OpenVLA 模型。

    参数：
        checkpoint_path: checkpoint 目录路径
        load_in_8bit: 是否 8-bit 量化加载
        load_in_4bit: 是否 4-bit 量化加载

    返回：
        (model, processor) 元组
    """
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    # 注册 OpenVLA 自定义类
    try:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ImportError:
        pass

    print(f"[模型加载] 从 '{checkpoint_path}' 加载...")
    print(f"  数据类型: bfloat16")

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    # 量化配置
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 将模型移到设备
    if not load_in_8bit and not load_in_4bit:
        vla = vla.to(DEVICE)

    # 加载数据集统计量（用于动作反归一化）
    stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            vla.norm_stats = json.load(f)
        print(f"  数据集统计量已加载: {list(vla.norm_stats.keys())}")
    else:
        print(f"  警告：未找到 dataset_statistics.json，模型可能无法正确反归一化动作")
        vla.norm_stats = {}

    print(f"  模型加载完成")
    return vla, processor


def get_vla_action(
    vla: Any,
    processor: Any,
    base_vla_name: str,
    image: np.ndarray,
    task_label: str,
    unnorm_key: str,
    center_crop: bool = False,
) -> np.ndarray:
    """
    使用 VLA 模型预测动作。

    参数：
        vla: VLA 模型
        processor: HuggingFace Processor
        base_vla_name: 基础模型名称（用于判断 prompt 格式）
        image: 预处理后的 RGB 图像 numpy array
        task_label: 任务描述文本
        unnorm_key: 动作反归一化 key
        center_crop: 是否中心裁剪

    返回：
        7 维动作向量
    """
    # numpy array → PIL Image
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert("RGB")

    # Center crop（如果训练时使用了 image augmentation）
    if center_crop:
        pil_image_np = np.array(pil_image)
        pil_image_np = center_crop_image(pil_image_np, crop_scale=0.9, target_size=224)
        pil_image = Image.fromarray(pil_image_np)
        pil_image = pil_image.convert("RGB")

    # 构造 prompt
    if "openvla-v01" in base_vla_name:
        # OpenVLA v0.1 使用对话格式
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} "
            f"USER: What action should the robot take to {task_label.lower()}? "
            f"ASSISTANT:"
        )
    else:
        # OpenVLA 使用 In/Out 格式
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # 处理输入
    inputs = processor(prompt, pil_image).to(DEVICE, dtype=torch.bfloat16)

    # 预测动作
    # predict_action() 是 OpenVLA 模型的方法：
    #   1. 模型前向推理，生成动作 token
    #   2. 使用 unnorm_key 将归一化的动作还原到真实空间
    #   3. 返回 7 维连续动作
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    assert action.shape == (ACTION_DIM,), (
        f"动作维度不正确: 预期 ({ACTION_DIM},), 实际 {action.shape}"
    )

    return action.cpu().numpy()


# ============================================================
# 评估主函数
# ============================================================

def eval_libero(args: argparse.Namespace) -> None:
    """
    在 LIBERO 仿真环境上评估 VLA 模型。

    评估流程详解：
    1. 加载模型和 processor
    2. 初始化 LIBERO task suite
    3. 对每个 task：
       a. 获取所有预定义的初始场景状态
       b. 对每个 episode：
          - 设置初始状态
          - 前几步执行空操作（等待物体稳定）
          - 之后每步：获取图像 → 模型预测动作 → 执行 → 判断成功/失败
       c. 统计该 task 的成功率
    4. 统计总体成功率
    """
    assert args.checkpoint_path, "必须指定 --checkpoint_path"

    # 设置随机种子
    set_seed_everywhere(args.seed)

    # ---- 加载模型 ----
    print(f"\n{'='*60}")
    print(f"LIBERO 评估")
    print(f"  Benchmark: {args.task_suite_name}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Trials per task: {args.num_trials_per_task}")
    print(f"{'='*60}\n")

    model, processor = load_vla_model(
        args.checkpoint_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    # ---- 设置 unnorm_key ----
    # unnorm_key 用于在推理时将模型输出的归一化动作还原到真实动作空间
    # 通常就是 benchmark 的名称
    unnorm_key = args.task_suite_name

    # 检查 unnorm_key 是否存在于模型的 norm_stats 中
    if hasattr(model, "norm_stats") and model.norm_stats:
        if unnorm_key not in model.norm_stats:
            # 尝试带 _no_noops 后缀的 key
            alt_key = f"{unnorm_key}_no_noops"
            if alt_key in model.norm_stats:
                unnorm_key = alt_key
                print(f"  使用备选 unnorm_key: {unnorm_key}")
            else:
                available_keys = list(model.norm_stats.keys())
                raise ValueError(
                    f"unnorm_key '{args.task_suite_name}' 不在模型的 norm_stats 中。\n"
                    f"可用的 key: {available_keys}\n"
                    f"请检查 checkpoint 是否与正确的 benchmark 对应。"
                )
    else:
        print("  警告：模型没有 norm_stats，跳过 unnorm_key 检查")

    # ---- 初始化 WandB ----
    if args.use_wandb:
        try:
            import wandb
            run_id = f"EVAL-{args.task_suite_name}-{DATE_TIME}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_id,
            )
        except ImportError:
            print("警告：未安装 wandb，跳过日志记录。")
            args.use_wandb = False

    # ---- 初始化 LIBERO task suite ----
    from libero.libero import benchmark as libero_benchmark

    benchmark_dict = libero_benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(
            f"未知的 benchmark '{args.task_suite_name}'。"
            f"可选: {list(benchmark_dict.keys())}"
        )

    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = LIBERO_MAX_STEPS.get(args.task_suite_name, 300)

    # 图像 resize 尺寸（OpenVLA 使用 224x224）
    resize_size = 224

    print(f"  任务数量: {num_tasks}")
    print(f"  每个任务最大步数: {max_steps}")
    print(f"  图像尺寸: {resize_size}x{resize_size}")
    print()

    # ---- 开始评估 ----
    total_episodes = 0
    total_successes = 0
    task_results: Dict[str, Dict[str, int]] = {}

    for task_id in range(num_tasks):
        # 获取任务
        task = task_suite.get_task(task_id)
        task_description = task.language

        # 获取该任务的所有初始状态
        initial_states = task_suite.get_task_init_states(task_id)

        # 初始化环境
        env, env_task_desc = get_libero_env(task, "openvla", resolution=256)

        task_episodes = 0
        task_successes = 0

        print(f"\n--- Task {task_id + 1}/{num_tasks}: {task_description} ---")

        for episode_idx in range(args.num_trials_per_task):
            # 重置环境
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            success = False

            while t < max_steps + args.num_steps_wait:
                try:
                    # 前 num_steps_wait 步执行空操作
                    # 原因：仿真环境中物体从高处落下，需要时间稳定
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action("openvla")
                        )
                        t += 1
                        continue

                    # 获取并预处理图像
                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    # 构造 observation（包含图像和机器人状态）
                    # 注意：OpenVLA 不使用 proprioception（关节角度等状态），
                    # 只使用图像 + 语言指令。
                    # 但有些评估脚本会提供 state 以便兼容其他模型。
                    observation = {
                        "full_image": img,
                        "state": np.concatenate([
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                    }

                    # 模型预测动作
                    action = get_vla_action(
                        model=model,
                        processor=processor,
                        base_vla_name=args.checkpoint_path,
                        image=img,
                        task_label=task_description,
                        unnorm_key=unnorm_key,
                        center_crop=args.center_crop,
                    )

                    # 归一化夹爪动作：[0,1] → [-1,+1]
                    action = normalize_gripper_action(action, binarize=True)

                    # 翻转夹爪符号（OpenVLA 的约定 vs LIBERO 环境的约定）
                    action = invert_gripper_action(action)

                    # 在环境中执行动作
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        success = True
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    print(f"    Episode {episode_idx + 1} 出错: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存回放视频（可选）
            if args.save_videos and len(replay_images) > 0:
                video_path = save_rollout_video(
                    replay_images, total_episodes, success, task_description,
                    video_dir=args.video_dir,
                )

            # 打印当前 episode 结果
            status = "成功" if success else "失败"
            print(f"  Episode {episode_idx + 1}/{args.num_trials_per_task}: {status}")

        # 记录该 task 的结果
        task_success_rate = task_successes / max(task_episodes, 1)
        task_results[task_description] = {
            "successes": task_successes,
            "episodes": task_episodes,
            "success_rate": task_success_rate,
        }

        print(f"  Task 成功率: {task_success_rate:.1%} ({task_successes}/{task_episodes})")
        print(f"  总体成功率: {total_successes / max(total_episodes, 1):.1%}")

        # WandB 记录
        if args.use_wandb:
            wandb.log({
                f"success_rate/{task_description}": task_success_rate,
            })

    # ---- 评估完成 ----
    total_success_rate = total_successes / max(total_episodes, 1)

    print(f"\n{'='*60}")
    print(f"评估结果汇总")
    print(f"{'='*60}")
    print(f"Benchmark: {args.task_suite_name}")
    print(f"总 Episodes: {total_episodes}")
    print(f"总成功: {total_successes}")
    print(f"总体成功率: {total_success_rate:.1%}")
    print(f"\n各任务详情:")
    for desc, result in task_results.items():
        print(f"  {desc[:60]:60s}: {result['success_rate']:.1%} ({result['successes']}/{result['episodes']})")
    print(f"{'='*60}")

    # WandB 记录总体结果
    if args.use_wandb:
        wandb.log({"success_rate/total": total_success_rate})
        wandb.finish()

    # 保存结果到 JSON 文件
    results = {
        "benchmark": args.task_suite_name,
        "checkpoint": args.checkpoint_path,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "task_results": task_results,
        "timestamp": DATE_TIME,
    }
    results_path = Path(args.output_dir) / f"eval_results_{args.task_suite_name}_{DATE}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 LIBERO 仿真环境上评估微调后的 OpenVLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 模型相关 ----
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="微调后的 checkpoint 目录路径。",
    )
    model_group.add_argument(
        "--load_in_8bit", action="store_true", default=False,
        help="使用 8-bit 量化加载模型（省显存）。",
    )
    model_group.add_argument(
        "--load_in_4bit", action="store_true", default=False,
        help="使用 4-bit 量化加载模型（更省显存，但精度更低）。",
    )
    model_group.add_argument(
        "--center_crop", action="store_true", default=True,
        help="是否中心裁剪图像。如果训练时使用了 image_aug（random_crop），"
             "评估时需要 center_crop 以保持一致。",
    )

    # ---- 评估相关 ----
    eval_group = parser.add_argument_group("评估配置")
    eval_group.add_argument(
        "--task_suite_name", type=str, default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="要评估的 LIBERO benchmark 名称。",
    )
    eval_group.add_argument(
        "--num_trials_per_task", type=int, default=20,
        help="每个 task 运行的 episode 数量。越多结果越可靠，但越慢。",
    )
    eval_group.add_argument(
        "--num_steps_wait", type=int, default=10,
        help="每个 episode 开始时空操作的步数（等待物体稳定）。",
    )

    # ---- 视频保存 ----
    video_group = parser.add_argument_group("视频配置")
    video_group.add_argument(
        "--save_videos", action="store_true", default=False,
        help="是否保存 episode 回放视频。",
    )
    video_group.add_argument(
        "--video_dir", type=str, default="./rollouts",
        help="视频保存目录。",
    )

    # ---- 输出 ----
    output_group = parser.add_argument_group("输出配置")
    output_group.add_argument(
        "--output_dir", type=str, default="./eval_results",
        help="评估结果 JSON 文件的保存目录。",
    )

    # ---- 日志 ----
    log_group = parser.add_argument_group("日志配置")
    log_group.add_argument(
        "--use_wandb", action="store_true", default=False,
        help="是否使用 WandB 记录评估结果。",
    )
    log_group.add_argument(
        "--wandb_project", type=str, default="vla-eval",
        help="WandB 项目名称。",
    )
    log_group.add_argument(
        "--wandb_entity", type=str, default=None,
        help="WandB 用户/团队名。",
    )

    # ---- 其他 ----
    misc_group = parser.add_argument_group("其他")
    misc_group.add_argument(
        "--seed", type=int, default=7,
        help="随机种子。",
    )

    return parser.parse_args()


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    """程序入口。"""
    args = parse_args()

    # 检查依赖
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        print("错误：评估需要 tensorflow。请运行 pip install tensorflow")
        sys.exit(1)

    try:
        from libero.libero import benchmark  # noqa: F401
    except ImportError:
        print("错误：评估需要 libero 包。请运行 pip install libero")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("警告：未检测到 GPU。评估将在 CPU 上运行，速度很慢。")

    eval_libero(args)


if __name__ == "__main__":
    main()
