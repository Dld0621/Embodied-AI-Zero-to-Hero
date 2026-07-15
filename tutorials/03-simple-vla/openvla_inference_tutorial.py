#!/usr/bin/env python3
"""
openvla_inference_tutorial.py -- OpenVLA 推理完整教程

================================================================================
  VLA-Zero-to-Hero 教学项目 · Stage 3: 简单 VLA
================================================================================

本脚本逐步引导学习者完成 OpenVLA 模型的推理全流程：
  1. 模型加载（含显存不足自动降级）
  2. 图像预处理
  3. 指令 tokenize
  4. 模型前向传播
  5. 动作反归一化
  6. 可视化预测动作

同时展示同一张图像在不同指令下的动作差异，帮助理解语言条件控制。

功能：
  - 逐步引导加载 OpenVLA 模型
  - 在示例图像（自动生成）上推理
  - 详细解析每个步骤
  - 多指令对比：同一图像，不同指令 → 不同动作
  - 可视化：输入图像 + 7 维预测动作的柱状图
  - 错误处理：显存不足时自动降级到 CPU 或量化模式

依赖：
  pip install torch transformers>=4.40.0 accelerate pillow numpy matplotlib

用法示例：
  # 默认推理（自动检测 GPU，自动降级）
  python openvla_inference_tutorial.py

  # 指定模型和图像
  python openvla_inference_tutorial.py --model_path openvla/openvla-7b --image_path scene.jpg

  # 强制使用 CPU
  python openvla_inference_tutorial.py --device cpu

  # 使用 8-bit 量化（省显存）
  python openvla_inference_tutorial.py --use_8bit

  # 保存结果图
  python openvla_inference_tutorial.py --save_output result.png

注意：
  - 首次运行会从 HuggingFace 下载模型（~15GB），需要稳定网络
  - GPU 推理建议 >= 16GB 显存（bfloat16 模式）
  - 如果没有真实图像，脚本会自动生成一张示例图像
  - 所有注释使用中文，教学风格
"""

import argparse
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ── 中文字体设置（延迟导入 matplotlib，避免在 CPU-only 模式下过早加载）─────────
def setup_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ==============================================================================
#  生成示例图像（当用户没有提供真实图像时）
# ==============================================================================

def generate_sample_image(size: int = 224) -> Image.Image:
    """
    生成一张简单的示例图像，模拟桌面场景。

    图像内容：一个灰色背景 + 一个红色方块（模拟杯子）+ 一个蓝色方块（模拟目标位置）

    Args:
        size: 图像尺寸（正方形）

    Returns:
        PIL.Image: 生成的示例图像
    """
    img = Image.new("RGB", (size, size), color=(200, 200, 180))  # 米色背景
    draw = ImageDraw.Draw(img)

    # 模拟桌面
    draw.rectangle([20, 100, size - 20, size - 20], fill=(139, 119, 101), outline=(100, 80, 60))

    # 模拟红色杯子（左侧）
    draw.rectangle([50, 50, 90, 100], fill=(220, 50, 50), outline=(180, 30, 30))

    # 模拟蓝色目标区域（右侧）
    draw.rectangle([140, 60, 180, 100], fill=(50, 100, 220), outline=(30, 70, 180))

    # 模拟绿色障碍物
    draw.rectangle([100, 80, 130, 110], fill=(50, 180, 50), outline=(30, 140, 30))

    return img


# ==============================================================================
#  Step 1: 模型加载（含自动降级）
# ==============================================================================

def load_model(
    model_path: str = "openvla/openvla-7b",
    device: str = "auto",
    use_8bit: bool = False,
):
    """
    加载 OpenVLA 模型，包含显存不足的自动降级逻辑。

    降级策略：
      1. 尝试 bfloat16 + GPU（最快，需要 ~15GB 显存）
      2. 如果 OOM → 尝试 8-bit 量化 + GPU（~8GB 显存）
      3. 如果仍 OOM → 降级到 CPU（最慢但一定能跑）

    Args:
        model_path: HuggingFace 模型路径
        device: 设备选择 ('auto' / 'cuda' / 'cpu')
        use_8bit: 是否强制使用 8-bit 量化

    Returns:
        tuple: (model, processor, actual_device)
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("\n[Step 1] 加载 OpenVLA 模型...")
    print(f"  模型路径: {model_path}")
    print(f"  请求设备: {device}")

    # 确定设备
    if device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        actual_device = device

    print(f"  实际设备: {actual_device}")

    # 加载 processor（不需要 GPU）
    print("  加载 Processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [错误] 加载 Processor 失败: {e}")
        print("  提示: 请确保已安装 transformers>=4.40.0 且网络连接正常")
        sys.exit(1)

    # 尝试加载模型
    model = None
    load_attempts = []

    if actual_device == "cuda" and not use_8bit:
        load_attempts.append(("bfloat16 + GPU", {"torch_dtype": torch.bfloat16}))
    if actual_device == "cuda":
        load_attempts.append(("8-bit 量化 + GPU", {"quantization_config": _get_8bit_config()}))
    load_attempts.append(("float32 + CPU", {"torch_dtype": torch.float32}))

    for attempt_name, kwargs in load_attempts:
        print(f"\n  尝试: {attempt_name}...")
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs,
            )

            if actual_device == "cuda" and "8-bit" not in attempt_name:
                model = model.to("cuda")

            print(f"  成功! 使用 {attempt_name}")
            actual_device = "cuda" if model.device.type == "cuda" else "cpu"
            break

        except torch.cuda.OutOfMemoryError:
            print(f"  显存不足 (OOM)，尝试降级...")
            # 清理
            if model is not None:
                del model
                model = None
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  加载失败: {e}")
            if model is not None:
                del model
                model = None

    if model is None:
        print("\n  [错误] 所有加载方式均失败。请检查:")
        print("    1. transformers 版本 >= 4.40.0")
        print("    2. accelerate 是否已安装")
        print("    3. 网络是否可以访问 HuggingFace")
        sys.exit(1)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params / 1e9:.2f}B")
    print(f"  设备: {model.device}")

    return model, processor, actual_device


def _get_8bit_config():
    """获取 8-bit 量化配置。"""
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        print("  [警告] bitsandbytes 未安装，无法使用 8-bit 量化")
        print("  安装: pip install bitsandbytes")
        return None


# ==============================================================================
#  Step 2-4: 预处理 → 推理 → 后处理
# ==============================================================================

def predict_action(
    model,
    processor,
    image: Image.Image,
    instruction: str,
    unnorm_key: str = "bridge",
    device: str = "cpu",
) -> np.ndarray:
    """
    完整的推理流程：预处理 → tokenize → 前向 → 反归一化。

    这是对 README.md 中 3.3 节 pipeline 的逐行实现。

    Args:
        model: OpenVLA 模型
        processor: OpenVLA Processor
        image: PIL Image（RGB）
        instruction: 语言指令（英文）
        unnorm_key: 反归一化统计量键名
        device: 计算设备

    Returns:
        np.ndarray: 7 维动作（已反归一化到物理单位）
    """
    # ── Step 2: 预处理 ────────────────────────────────────────────────────
    # OpenVLA 使用特定 prompt 格式: "In: <instruction>\nOut:"
    prompt = f"In: {instruction}\nOut:"

    # Processor 将图像和文本转换为模型输入 tensor
    # 内部操作:
    #   - 图像: resize → normalize (ImageNet mean/std) → pixel_values tensor
    #   - 文本: tokenize → input_ids tensor
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    # ── Step 3: 模型前向传播 ─────────────────────────────────────────────
    # 模型内部流程:
    #   1. 视觉编码器处理图像 → vision tokens
    #   2. 语言模型接收 text tokens + vision tokens
    #   3. 自回归生成动作 token
    #   4. 解码为 7 维浮点数（归一化到 [-1, 1]）
    with torch.no_grad():
        action = model.predict_action(inputs, unnorm_key=unnorm_key)

    # ── Step 5: 动作反归一化 ──────────────────────────────────────────────
    # predict_action 已经内置了反归一化，但让我们理解其原理：
    #
    # 归一化: action_norm = (action_phys - mean) / std
    # 反归一化: action_phys = action_norm * std + mean
    #
    # bridge 数据集的统计量:
    #   位置 (x,y,z):     std=0.02m,  mean=0.0
    #   旋转 (r,p,y):     std=0.05rad, mean=0.0
    #   夹爪:              std=1.0,     mean=0.0
    #
    # 模型输出范围 [-1,1] 反归一化后:
    #   dx,dy,dz:  [-0.02, 0.02] m
    #   dr,dp,dy:  [-0.05, 0.05] rad
    #   gripper:   [-1.0,  1.0]

    action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)
    if action_np.ndim > 1:
        action_np = action_np.squeeze()

    return action_np


# ==============================================================================
#  多指令对比推理
# ==============================================================================

MULTI_INSTRUCTION_PROMPTS = [
    "pick up the red cup",
    "push the green block to the right",
    "place the object on the blue target",
    "reach for the top of the red cup",
    "move the robot arm to the center of the table",
]


def multi_instruction_comparison(
    model, processor, image: Image.Image,
    device: str = "cpu",
    unnorm_key: str = "bridge",
) -> dict:
    """
    同一张图像，不同指令 → 对比预测动作。

    展示 VLA 的核心能力：语言条件控制。
    同一张图像，不同的指令会产生完全不同的动作。

    Args:
        model, processor: OpenVLA 模型和处理器
        image: 输入图像
        device: 计算设备
        unnorm_key: 反归一化键

    Returns:
        dict: {instruction: action_array}
    """
    print("\n[多指令对比] 同一张图像，5 种不同指令:")
    print(f"  {'指令':<50s}  {'动作摘要'}")
    print(f"  {'-'*50}  {'-'*60}")

    results = {}
    for instruction in MULTI_INSTRUCTION_PROMPTS:
        start_time = time.time()
        action = predict_action(model, processor, image, instruction, unnorm_key, device)
        elapsed = time.time() - start_time

        results[instruction] = action

        action_str = ", ".join([f"{v:+.4f}" for v in action])
        print(f"  {instruction:<50s}  [{action_str}]")
        print(f"  {'':>50s}  推理耗时: {elapsed:.2f}s")

    return results


# ==============================================================================
#  Step 6: 可视化
# ==============================================================================

def visualize_results(
    image: Image.Image,
    results: dict,
    save_path: str = None,
):
    """
    可视化推理结果：输入图像 + 多指令动作对比柱状图。

    Args:
        image: 输入图像
        results: {instruction: action_array} 字典
        save_path: 保存路径
    """
    plt = setup_matplotlib()

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("OpenVLA 推理结果 — 多指令动作对比", fontsize=16, fontweight="bold")

    # ── 左侧：输入图像 ───────────────────────────────────────────────────
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(image)
    ax_img.set_title("输入图像", fontsize=14, fontweight="bold")
    ax_img.axis("off")

    # ── 右侧：动作对比柱状图 ─────────────────────────────────────────────
    ax_bar = fig.add_subplot(1, 2, 2)

    action_dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    dim_labels = [
        "dx\n(位置x)",
        "dy\n(位置y)",
        "dz\n(位置z)",
        "droll\n(滚转)",
        "dpitch\n(俯仰)",
        "dyaw\n(偏航)",
        "gripper\n(夹爪)",
    ]

    n_instructions = len(results)
    colors = plt.cm.Set2(np.linspace(0, 1, n_instructions))
    x_pos = np.arange(len(action_dim_names))
    bar_width = 0.8 / n_instructions

    for i, (instruction, action) in enumerate(results.items()):
        offset = (i - n_instructions / 2 + 0.5) * bar_width
        label = instruction[:20] + "..." if len(instruction) > 20 else instruction
        ax_bar.bar(x_pos + offset, action, bar_width,
                  label=label, color=colors[i], alpha=0.85)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(dim_labels, fontsize=9)
    ax_bar.set_ylabel("动作值（反归一化后）", fontsize=12)
    ax_bar.set_title("预测动作的 7 个维度", fontsize=14, fontweight="bold")
    ax_bar.legend(fontsize=8, loc="upper left", ncol=2)
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.axhline(y=0, color="black", linewidth=0.5)

    # 添加单位说明
    unit_text = (
        "单位说明:\n"
        "  dx, dy, dz: 米 (m)\n"
        "  droll, dpitch, dyaw: 弧度 (rad)\n"
        "  gripper: 连续值 [-1, 1]"
    )
    ax_bar.text(
        0.98, 0.98, unit_text,
        transform=ax_bar.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        fontfamily="monospace",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        print(f"\n[可视化] 保存结果到 {save_path}")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ==============================================================================
#  动作输出详解
# ==============================================================================

def explain_action(action: np.ndarray, instruction: str):
    """
    详细解释预测动作的每个维度的含义。

    Args:
        action: 7 维动作数组
        instruction: 对应的语言指令
    """
    dim_info = [
        ("dx",     "末端 X 方向位移",     "m",   0.02),
        ("dy",     "末端 Y 方向位移",     "m",   0.02),
        ("dz",     "末端 Z 方向位移",     "m",   0.02),
        ("droll",  "末端绕 X 轴旋转",     "rad", 0.05),
        ("dpitch", "末端绕 Y 轴旋转",     "rad", 0.05),
        ("dyaw",   "末端绕 Z 轴旋转",     "rad", 0.05),
        ("gripper","夹爪开合",           "-",   1.00),
    ]

    print(f"\n  指令: \"{instruction}\"")
    print(f"  {'维度':<10s} {'含义':<20s} {'值':>10s} {'单位':<8s} {'解释'}")
    print(f"  {'-'*70}")

    for i, (name, meaning, unit, _) in enumerate(dim_info):
        val = action[i]
        abs_val = abs(val)

        # 生成人类可读的解释
        if name == "gripper":
            if val > 0.3:
                explanation = "张开夹爪"
            elif val < -0.3:
                explanation = "闭合夹爪"
            else:
                explanation = "保持当前夹爪状态"
        elif unit == "m":
            direction = "正方向（向前/向右）" if val > 0 else "负方向（向后/向左）"
            explanation = f"向{direction}移动 {abs_val*100:.1f}cm"
        else:
            direction = "逆时针" if val > 0 else "顺时针"
            explanation = f"绕轴{direction}旋转 {abs_val:.2f}rad ({np.degrees(abs_val):.1f}°)"

        print(f"  {name:<10s} {meaning:<20s} {val:>+10.4f} {unit:<8s} {explanation}")


# ==============================================================================
#  Pipeline 总览打印
# ==============================================================================

def print_pipeline_overview():
    """打印 VLA 推理 pipeline 的完整流程说明。"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              OpenVLA 推理 Pipeline 完整流程                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. 图像采集                                                          ║
║     └── 从相机获取 RGB 图像 (H, W, 3)                                ║
║                                                                      ║
║  2. 语言指令                                                          ║
║     └── 用户输入: "pick up the red cup"                              ║
║                                                                      ║
║  3. 预处理 (Processor)                                                ║
║     └── 图像: resize(224x224) → normalize(ImageNet) → pixel_values  ║
║     └── 文本: "In: <指令>\\nOut:" → tokenize → input_ids            ║
║                                                                      ║
║  4. 模型前向 (OpenVLA-7B)                                             ║
║     └── 视觉编码器: ViT → vision tokens                              ║
║     └── 语言模型: LLM(image_tokens, text_tokens) → action tokens    ║
║     └── 解码: tokens → 7维浮点数 (归一化到 [-1, 1])                  ║
║                                                                      ║
║  5. 后处理                                                            ║
║     └── 反归一化: action * std + mean                                ║
║     └── 裁剪到安全范围                                                ║
║     └── 可选: 平滑滤波 (EMA / 低通滤波)                              ║
║                                                                      ║
║  6. 执行                                                              ║
║     └── 发送动作到机器人控制器                                        ║
║     └── 或: 发送到仿真环境                                            ║
║                                                                      ║
║  7. 循环                                                              ║
║     └── 获取新图像 → 重复步骤 1-6                                     ║
║     └── 使用 Action Chunking 可降低推理频率                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ==============================================================================
#  主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA 推理完整教程 — 逐步引导",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python openvla_inference_tutorial.py                           # 默认（自动生成示例图）
  python openvla_inference_tutorial.py --image_path scene.jpg   # 使用真实图像
  python openvla_inference_tutorial.py --device cpu             # 强制 CPU
  python openvla_inference_tutorial.py --use_8bit              # 8-bit 量化
  python openvla_inference_tutorial.py --save_output result.png # 保存结果
        """,
    )
    parser.add_argument(
        "--model_path", type=str, default="openvla/openvla-7b",
        help="HuggingFace 模型路径，默认: openvla/openvla-7b"
    )
    parser.add_argument(
        "--image_path", type=str, default=None,
        help="输入图像路径（不指定则自动生成示例图像）"
    )
    parser.add_argument(
        "--unnorm_key", type=str, default="bridge",
        help="反归一化统计量键名，默认: bridge"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="计算设备，默认: auto（自动检测）"
    )
    parser.add_argument(
        "--use_8bit", action="store_true",
        help="强制使用 8-bit 量化（节省显存）"
    )
    parser.add_argument(
        "--save_output", type=str, default=None,
        help="保存可视化结果为 PNG"
    )

    args = parser.parse_args()

    print("=" * 64)
    print("  OpenVLA 推理完整教程")
    print("  逐步引导: 加载 → 预处理 → 推理 → 反归一化 → 可视化")
    print("=" * 64)

    # ── Pipeline 总览 ───────────────────────────────────────────────────
    print_pipeline_overview()

    # ── Step 1: 加载模型 ──────────────────────────────────────────────────
    model, processor, actual_device = load_model(
        model_path=args.model_path,
        device=args.device,
        use_8bit=args.use_8bit,
    )

    # ── 加载图像 ─────────────────────────────────────────────────────────
    if args.image_path:
        try:
            image = Image.open(args.image_path).convert("RGB")
            print(f"\n  加载图像: {args.image_path} ({image.size[0]}x{image.size[1]})")
        except Exception as e:
            print(f"\n  [警告] 无法加载图像 {args.image_path}: {e}")
            print("  使用自动生成的示例图像")
            image = generate_sample_image()
    else:
        image = generate_sample_image()
        print(f"\n  使用自动生成的示例图像 ({image.size[0]}x{image.size[1]})")
        print("  提示: 使用 --image_path 指定真实图像可获得更有意义的输出")

    # ── Step 2-5: 单指令推理演示 ─────────────────────────────────────────
    demo_instruction = "pick up the red cup"
    print(f"\n{'='*64}")
    print(f"  单指令推理演示")
    print(f"  指令: \"{demo_instruction}\"")
    print(f"{'='*64}")

    print("\n[Step 2] 预处理...")
    print(f"  Prompt 格式: \"In: {demo_instruction}\\nOut:\"")
    print(f"  图像将被 resize 到 224x224，并用 ImageNet 均值/方差归一化")
    print(f"  文本将被 tokenize 为 input_ids")

    print("\n[Step 3] 模型前向传播...")
    print(f"  设备: {actual_device}")
    action = predict_action(model, processor, image, demo_instruction, args.unnorm_key, actual_device)

    print("\n[Step 4-5] 输出与反归一化...")
    print(f"  原始输出 (归一化): 范围 [-1, 1]")
    print(f"  反归一化 (unnorm_key={args.unnorm_key}):")

    # 显示反归一化后的值
    action_str = ", ".join([f"{v:+.6f}" for v in action])
    print(f"  最终动作: [{action_str}]")

    # 详细解释
    explain_action(action, demo_instruction)

    # ── 多指令对比 ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  多指令对比（同一图像，不同指令）")
    print(f"  展示 VLA 的语言条件控制能力")
    print(f"{'='*64}")

    results = multi_instruction_comparison(
        model, processor, image,
        device=actual_device,
        unnorm_key=args.unnorm_key,
    )

    # ── Step 6: 可视化 ──────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  可视化")
    print(f"{'='*64}")

    visualize_results(image, results, save_path=args.save_output)

    # ── 总结 ────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  教程总结")
    print(f"{'='*64}")
    print("""
  你已经完成了 OpenVLA 推理的完整流程：

  1. 模型加载: 了解了显存不足时的自动降级策略
  2. 图像预处理: resize + normalize → tensor
  3. 文本 tokenize: "In: <指令>\\nOut:" 格式
  4. 模型前向: ViT + LLM → 7维动作（归一化）
  5. 反归一化: action * std + mean → 物理单位
  6. 可视化: 同一图像不同指令 → 不同动作

  关键理解:
  - VLA 的输出是 7 维 delta 位姿（增量），不是绝对位置
  - 需要累积 delta 才能得到实际轨迹
  - unnorm_key 决定了反归一化使用的统计量
  - Action Chunking 可以降低推理频率（每 K 步推理一次）

  下一步:
  - Stage 4: 在真实数据上微调 OpenVLA
  - 参考 README.md 了解更多推理优化技巧
""")

    print("完成!")


if __name__ == "__main__":
    main()
