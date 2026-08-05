"""
生成 PushCube 任务演示 GIF
==========================

为 PushCube 双方块推动任务生成 3 段对比演示 GIF：
  1. Expert  —— 调用 expert_action 的三阶段启发式（flank → behind → push），高成功率
  2. BC      —— 模拟行为克隆策略：先靠近主动方块、再朝目标推，带高斯噪声（有缺陷的朴素策略）
  3. Random  —— 随机动作基线

三个策略使用同一初始种子（SEED=36），便于横向对比同一开局下的不同表现。

每个 GIF：
  - 分辨率 128x128（与环境 render 尺寸一致）
  - 60~80 帧（一整局 episode，不足 60 帧则补齐最后一帧）
  - 每帧叠加语言指令文本（如 "push the red cube to the left and top"）
  - 输出到 assets/gifs/，文件名 expert_demo.gif / bc_demo.gif / random_demo.gif

依赖：优先使用 Pillow（Windows 上最稳定）；Pillow 不可用时回退到 matplotlib.animation。

Usage:
    python scripts/generate_demo_gifs.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

# ---------------------------------------------------------------------
# 导入项目环境（与仓库其它脚本一致的 sys.path 注入方式）
# ---------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.unified_pushcube_env import PushCubeEnv, expert_action  # noqa: E402

# ---------------------------------------------------------------------
# 依赖：Pillow 优先，matplotlib 回退
# ---------------------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:  # pragma: no cover - 仅在无 Pillow 时触发
    HAS_PIL = False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401

# ---------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------
RENDER_SIZE = 128          # 与环境 render(size=128) 一致
MAX_FRAMES = 80            # 每段 GIF 最多 80 帧
MIN_FRAMES = 60            # 不足 60 帧则用最后一帧补齐
FRAME_DURATION_MS = 90     # 每帧 90ms（约 11fps，8 秒左右一局）

# 三个策略使用同一初始条件，便于对比。
# SEED=36：expert 在 ~70 步成功（完整 flank→behind→push），BC 失败（形成对比）。
DEMO_SEED = 36

GIF_DIR = _PROJECT_ROOT / "assets" / "gifs"


# ---------------------------------------------------------------------
# 工具：图像 / 字体
# ---------------------------------------------------------------------
def to_uint8(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] RGB -> uint8 [0,255] RGB。"""
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_font(size: int):
    """加载 TrueType 字体，失败则回退到 PIL 默认位图字体。"""
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# 预加载字体（PIL 路径用）
if HAS_PIL:
    FONT_LABEL = load_font(9)   # 策略标签
    FONT_INS = load_font(7)     # 语言指令


def wrap_text(text: str, font, max_width: int) -> List[str]:
    """按词换行，使每行像素宽度不超过 max_width。"""
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        try:
            width = font.getlength(trial)
        except AttributeError:  # 位图字体无 getlength
            width = len(trial) * 4
        if width <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


# ---------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------
def expert_policy(env: PushCubeEnv, rng: np.random.RandomState) -> np.ndarray:
    """专家策略：直接调用环境自带的 expert_action（三阶段启发式）。"""
    return expert_action(env)


def bc_policy(env: PushCubeEnv, rng: np.random.RandomState) -> np.ndarray:
    """
    模拟行为克隆（BC）策略：用专家演示训练得到的“朴素”策略。

    与专家的关键区别：BC 不绕到方块背后，而是直接朝方块移动、靠近后
    朝目标推。由于常常从错误一侧接触方块，会把方块推离目标；再叠加
    高斯噪声模拟模仿学习的不完美，整体表现为“努力但常失败”。
    """
    active = env.cube_positions[env.active_idx]
    target = env.target_pos
    arm = env.arm_pos

    if np.linalg.norm(active - arm) > 0.10:
        # 阶段 1：靠近主动方块（不分侧别 —— 这正是 BC 的缺陷）
        direction = active - arm
    else:
        # 阶段 2：朝目标方向推
        direction = target - arm

    direction = direction / (np.linalg.norm(direction) + 1e-6)
    noise = rng.normal(0.0, 0.22, size=2).astype(np.float32)
    action = direction * 0.8 + noise
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def random_policy(env: PushCubeEnv, rng: np.random.RandomState) -> np.ndarray:
    """随机基线：均匀采样的动作。"""
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


# ---------------------------------------------------------------------
# 收集一局 episode 的原始帧
# ---------------------------------------------------------------------
PolicyFn = Callable[[PushCubeEnv, np.random.RandomState], np.ndarray]


def collect_episode(
    policy_fn: PolicyFn, seed: int
) -> Tuple[List[np.ndarray], str, bool, int]:
    """
    运行一局 episode，返回 (原始 uint8 帧列表, 语言指令, 是否成功, 实际步数)。
    帧数被限制在 [MIN_FRAMES, MAX_FRAMES]。
    """
    env = PushCubeEnv(seed=seed)
    env.reset(seed=seed)
    instruction = env.get_language_instruction()

    # 每个策略的随机源（BC/Random 用），与开局种子解耦但可复现
    rng = np.random.RandomState(seed + 1000)

    raw_frames: List[np.ndarray] = [to_uint8(env.render(RENDER_SIZE))]
    steps = 0
    success = False

    while len(raw_frames) < MAX_FRAMES:
        action = policy_fn(env, rng)
        _, _, terminated, truncated, _ = env.step(action)
        raw_frames.append(to_uint8(env.render(RENDER_SIZE)))
        steps += 1
        if terminated:
            success = True
            break
        if truncated:  # 达到 max_steps
            break

    # 不足 MIN_FRAMES 则用最后一帧补齐（保证 60~80 帧）
    while len(raw_frames) < MIN_FRAMES:
        raw_frames.append(raw_frames[-1])

    return raw_frames, instruction, success, steps


# ---------------------------------------------------------------------
# 文字叠加（PIL 路径）
# ---------------------------------------------------------------------
def overlay_text_pil(
    raw_frames: List[np.ndarray],
    instruction: str,
    label: str,
    label_color: Tuple[int, int, int, int],
) -> List:
    """在每帧顶部叠加半透明条带：策略标签 + 语言指令，返回 PIL RGB 帧列表。"""
    ins_lines = wrap_text(instruction, FONT_INS, RENDER_SIZE - 6)[:2]
    # 条带高度：标签行 + 指令行 + 内边距
    band_h = 3 + 10 + len(ins_lines) * 9 + 3

    out: List = []
    for f in raw_frames:
        base = Image.fromarray(f).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 半透明黑色条带
        draw.rectangle([0, 0, RENDER_SIZE, band_h], fill=(0, 0, 0, 170))
        # 策略标签（彩色）
        draw.text((4, 2), label, font=FONT_LABEL, fill=label_color)
        # 语言指令（白色，最多两行）
        y = 13
        for line in ins_lines:
            draw.text((4, y), line, font=FONT_INS, fill=(255, 255, 255, 255))
            y += 9

        out.append(Image.alpha_composite(base, overlay).convert("RGB"))
    return out


def save_gif_pil(frames_pil: List, path: str) -> None:
    """用 Pillow 保存 GIF（loop=0 无限循环，disposal=2 避免残影）。"""
    frames_pil[0].save(
        path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        disposal=2,
        optimize=False,
    )


# ---------------------------------------------------------------------
# 回退路径：matplotlib.animation（仅在 Pillow 不可用时）
# ---------------------------------------------------------------------
def save_gif_mpl(
    raw_frames: List[np.ndarray],
    instruction: str,
    label: str,
    label_color: Tuple[int, int, int, int],
    path: str,
) -> None:  # pragma: no cover - 仅在无 Pillow 时触发
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure(figsize=(1.28, 1.46), dpi=100)
    ax = fig.add_axes([0.0, 0.14, 1.0, 0.80])
    im = ax.imshow(raw_frames[0])
    ax.axis("off")
    hex_color = "#%02x%02x%02x" % label_color[:3]
    ax.text(
        0.5, 1.03, label, transform=ax.transAxes, ha="center", va="bottom",
        color=hex_color, fontsize=7, fontweight="bold",
    )
    ax.text(
        0.5, -0.05, instruction, transform=ax.transAxes, ha="center", va="top",
        color="white", fontsize=5.5,
    )
    fps = max(1, int(round(1000.0 / FRAME_DURATION_MS)))

    def update(i):
        im.set_array(raw_frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(raw_frames), interval=FRAME_DURATION_MS, blit=True
    )

    last_err = None
    for writer in ("pillow", "imagemagick"):
        try:
            ani.save(path, writer=writer, fps=fps)
            plt.close(fig)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    plt.close(fig)
    raise RuntimeError(
        "matplotlib GIF 保存失败（需要 pillow 或 imagemagick writer）: %s" % last_err
    )


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main() -> int:
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    # (名称, 标签, 标签颜色, 策略函数, 输出文件名)
    policies = [
        ("Expert", "EXPERT", (120, 255, 120, 255), expert_policy, "expert_demo.gif"),
        ("BC", "BC", (255, 190, 70, 255), bc_policy, "bc_demo.gif"),
        ("Random", "RANDOM", (210, 210, 210, 255), random_policy, "random_demo.gif"),
    ]

    print(f"项目根目录: {_PROJECT_ROOT}")
    print(f"GIF 输出目录: {GIF_DIR}")
    print(f"使用后端: {'Pillow' if HAS_PIL else 'matplotlib.animation (fallback)'}")
    print(f"演示种子: {DEMO_SEED}（三个策略共用同一初始条件）")
    print("-" * 70)

    for name, label, color, fn, fname in policies:
        raw_frames, instruction, success, steps = collect_episode(fn, DEMO_SEED)
        out_path = GIF_DIR / fname

        if HAS_PIL:
            frames = overlay_text_pil(raw_frames, instruction, label, color)
            save_gif_pil(frames, str(out_path))
        else:
            save_gif_mpl(raw_frames, instruction, label, color, str(out_path))

        status = "SUCCESS" if success else "fail"
        print(
            f"[{name:>6}] 指令: {instruction!r}\n"
            f"         帧数={len(raw_frames)} 步数={steps} 结果={status}\n"
            f"         -> {out_path}"
        )

    print("-" * 70)
    print("完成。3 个 GIF 已生成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
