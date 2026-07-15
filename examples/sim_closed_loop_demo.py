#!/usr/bin/env python3
"""
sim_closed_loop_demo.py -- PyBullet 仿真闭环 VLA Demo
=====================================================
展示：图像采集 -> VLA 推理 -> 动作执行 -> 环境反馈 -> 循环

使用 PyBullet 搭建桌面抓取场景，配合随机初始化的小型 VLA 策略，
演示从观察到执行的完整闭环控制流程。

用法：
    # 随机策略模式（纯演示闭环流程，无需训练）
    python sim_closed_loop_demo.py --mode random

    # 预编程策略模式（简单抓取脚本作为 baseline）
    python sim_closed_loop_demo.py --mode scripted

    # 保存视频
    python sim_closed_loop_demo.py --mode scripted --save_video --output_dir results/

依赖：
    pip install pybullet matplotlib numpy torch imageio
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# PyBullet 场景搭建
# ---------------------------------------------------------------------------

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not installed. Run: pip install pybullet")
    sys.exit(1)


# --- 场景参数 ---
TABLE_HEIGHT = 0.4
TABLE_SIZE = 0.6
CUBE_SIZE = 0.04
CUBE_COLOR = [0.9, 0.1, 0.1, 1.0]  # 红色方块

# 相机参数
CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224
CAMERA_FOV = 60.0
CAMERA_NEAR = 0.02
CAMERA_FAR = 5.0
CAMERA_TARGET_POS = [0.0, 0.0, TABLE_HEIGHT]       # 相机看向桌面上方
CAMERA_EYE_POS = [0.4, -0.4, TABLE_HEIGHT + 0.5]   # 相机位置

# 机器人 / 动作参数
ACTION_DIM = 7          # [dx, dy, dz, droll, dpitch, dyaw, gripper]
MAX_EPISODE_STEPS = 200
CONTROL_FREQ = 10       # Hz
DT = 1.0 / 240.0       # PyBullet 物理步长
STEPS_PER_ACTION = int(1.0 / CONTROL_FREQ / DT)  # 每个 action 之间的 physics steps

# 简易夹爪参数
GRIPPER_MAX_OPEN = 0.08
GRIPPER_FORCE = 50.0


def create_table_collision_shape(client_id: int) -> int:
    """创建桌子的碰撞形状（使用纯几何体，无需外部 URDF）。"""
    col_shape = client_id.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_SIZE / 2, TABLE_SIZE / 2, TABLE_HEIGHT / 2],
    )
    return col_shape


def create_table(client_id: int) -> int:
    """在场景中生成桌子。"""
    col_shape = create_table_collision_shape(client_id)
    visual_shape = client_id.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_SIZE / 2, TABLE_SIZE / 2, TABLE_HEIGHT / 2],
        rgbaColor=[0.6, 0.4, 0.2, 1.0],
    )
    table_id = client_id.createMultiBody(
        baseMass=0,  # static
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, TABLE_HEIGHT / 2],
    )
    return table_id


def create_cube(client_id: int, position: Optional[List[float]] = None) -> int:
    """在桌面上生成红色方块。"""
    if position is None:
        # 随机放置在桌面上
        x = np.random.uniform(-0.15, 0.15)
        y = np.random.uniform(-0.15, 0.15)
        position = [x, y, TABLE_HEIGHT + CUBE_SIZE / 2]

    col_shape = client_id.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[CUBE_SIZE / 2, CUBE_SIZE / 2, CUBE_SIZE / 2],
    )
    visual_shape = client_id.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[CUBE_SIZE / 2, CUBE_SIZE / 2, CUBE_SIZE / 2],
        rgbaColor=CUBE_COLOR,
    )
    cube_id = client_id.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
    )
    return cube_id


def create_simple_gripper_marker(client_id: int, position: Optional[List[float]] = None) -> int:
    """
    创建一个简化夹爪的视觉标记。
    真实场景中应使用 URDF 机器人模型，这里用球体代替。
    """
    if position is None:
        position = [0.0, 0.0, TABLE_HEIGHT + 0.15]

    col_shape = client_id.createCollisionShape(p.GEOM_SPHERE, radius=0.015)
    visual_shape = client_id.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[0.2, 0.6, 0.9, 1.0]
    )
    marker_id = client_id.createMultiBody(
        baseMass=0.0,  # kinematic
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
    )
    return marker_id


def setup_scene(headless: bool = False) -> Tuple[int, int, int, int]:
    """
    初始化 PyBullet 场景并创建桌面抓取环境。

    Returns:
        (client_id, table_id, cube_id, gripper_id)
    """
    physics_client = p.connect(p.DIRECT if headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(DT, physicsClientId=physics_client)

    # 设置地面
    plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client)

    # 创建场景物体
    table_id = create_table(physics_client)
    cube_id = create_cube(physics_client)
    gripper_id = create_simple_gripper_marker(physics_client)

    # 环境光照
    p.setLightPosition([0.5, -0.5, 1.5], physicsClientId=physics_client)
    light_id = p.createLight(
        position=[0.5, -0.5, 1.5],
        color=[1.0, 1.0, 1.0],
        intensity=1.0,
        physicsClientId=physics_client,
    )

    return physics_client, table_id, cube_id, gripper_id


# ---------------------------------------------------------------------------
# 相机接口
# ---------------------------------------------------------------------------

def get_camera_image(client_id: int) -> np.ndarray:
    """
    使用 PyBullet getCameraImage API 获取 224x224 RGB 图像。

    Returns:
        rgb: np.ndarray, shape (224, 224, 3), dtype uint8
    """
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAMERA_TARGET_POS,
        distance=0.8,
        yaw=45,
        pitch=-35,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client_id,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV,
        aspect=CAMERA_WIDTH / CAMERA_HEIGHT,
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR,
        physicsClientId=client_id,
    )

    _, _, rgb_pixels, _, _ = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )

    # rgb_pixels shape: (H, W, 4), 需要 RGB 三通道
    rgb = rgb_pixels[:, :, :3].astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# VLA 策略
# ---------------------------------------------------------------------------

class RandomPolicy:
    """随机策略：每步输出随机动作，纯演示闭环流程。"""

    def __init__(self, action_dim: int = 7, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.action_dim = action_dim

    def reset(self):
        """重置随机种子（可选）。"""
        pass

    def get_action(self, image: np.ndarray, instruction: str = "") -> np.ndarray:
        """
        根据当前图像和指令输出随机动作。

        Args:
            image: (H, W, 3) uint8 RGB 图像
            instruction: 语言指令（随机策略忽略此输入）

        Returns:
            action: (action_dim,) float32, 范围 [-1, 1]
        """
        action = self.rng.uniform(-0.3, 0.3, size=self.action_dim).astype(np.float32)
        # gripper 值限制在 [0, 1] 范围
        action[6] = self.rng.uniform(0, 1)
        return action

    @property
    def name(self) -> str:
        return "random"


class ScriptedPolicy:
    """
    预编程简单策略：先移动到物体上方 -> 下降 -> 夹紧 -> 提起。
    作为 baseline 展示一个合理的抓取序列。
    """

    # 策略阶段
    PHASE_MOVE_ABOVE = 0
    PHASE_DESCEND = 1
    PHASE_GRIP = 2
    PHASE_LIFT = 3
    PHASE_DONE = 4

    PHASE_NAMES = {
        0: "move_above",
        1: "descend",
        2: "grip",
        3: "lift",
        4: "done",
    }

    def __init__(self, action_dim: int = 7, client_id=None, cube_id=None):
        self.action_dim = action_dim
        self.client_id = client_id
        self.cube_id = cube_id
        self.phase = self.PHASE_MOVE_ABOVE
        self.phase_timer = 0
        self.gripper_pos = 1.0  # 初始张开

    def reset(self):
        """重置策略状态。"""
        self.phase = self.PHASE_MOVE_ABOVE
        self.phase_timer = 0
        self.gripper_pos = 1.0

    def set_environment(self, client_id: int, cube_id: int):
        """设置环境引用，用于获取物体位置。"""
        self.client_id = client_id
        self.cube_id = cube_id

    def _get_cube_pos(self) -> np.ndarray:
        """获取方块当前位置。"""
        pos, _ = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client_id)
        return np.array(pos)

    def _get_gripper_pos(self) -> np.ndarray:
        """获取当前夹爪位置（这里使用第一个 link 的状态）。"""
        # 简化：使用 marker 的位置
        pos, _ = p.getBasePositionAndOrientation(0, physicsClientId=self.client_id)
        return np.array(pos)

    def get_action(self, image: np.ndarray, instruction: str = "") -> np.ndarray:
        """
        根据当前阶段输出预编程动作。

        Args:
            image: 当前观测图像（脚本策略不直接使用，但保持接口一致）
            instruction: 语言指令

        Returns:
            action: (action_dim,) float32
        """
        action = np.zeros(self.action_dim, dtype=np.float32)
        self.phase_timer += 1

        cube_pos = self._get_cube_pos()

        if self.phase == self.PHASE_MOVE_ABOVE:
            # 移动到方块正上方
            target = np.array([cube_pos[0], cube_pos[1], TABLE_HEIGHT + 0.12])
            # 计算简单的比例控制（简化版，因为没有真正的末端执行器状态）
            # 这里直接输出朝向目标的归一化 delta
            delta = target - cube_pos
            action[0] = np.clip(delta[0] * 5, -0.5, 0.5)
            action[1] = np.clip(delta[1] * 5, -0.5, 0.5)
            action[2] = 0.02  # 向上移动
            action[6] = 1.0   # 夹爪张开

            if self.phase_timer > 30:
                self.phase = self.PHASE_DESCEND
                self.phase_timer = 0

        elif self.phase == self.PHASE_DESCEND:
            # 下降到方块附近
            action[2] = -0.015  # 缓慢下降
            action[6] = 1.0     # 保持张开

            if self.phase_timer > 25:
                self.phase = self.PHASE_GRIP
                self.phase_timer = 0

        elif self.phase == self.PHASE_GRIP:
            # 夹紧
            action[6] = max(0.0, 1.0 - self.phase_timer * 0.1)  # 逐渐关闭

            if self.phase_timer > 10:
                self.phase = self.PHASE_LIFT
                self.phase_timer = 0
                self.gripper_pos = 0.0

        elif self.phase == self.PHASE_LIFT:
            # 提起
            action[2] = 0.02  # 向上
            action[6] = 0.0   # 夹爪关闭

            if self.phase_timer > 40:
                self.phase = self.PHASE_DONE
                self.phase_timer = 0

        elif self.phase == self.PHASE_DONE:
            # 保持不动
            action[:] = 0.0
            action[6] = 0.0

        return action

    @property
    def name(self) -> str:
        return f"scripted_{self.PHASE_NAMES.get(self.phase, 'unknown')}"


class MinimalVLAPolicy:
    """
    使用 MinimalVLA 模型（参考 examples/minimal_vla.py）作为策略。
    注意：这是随机初始化的模型，输出无意义，仅演示接口。
    """

    def __init__(self, action_dim: int = 7):
        try:
            import torch
            # import from same directory
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from minimal_vla import MinimalVLA, simple_tokenize

            self.model = MinimalVLA(action_dim=action_dim)
            self.model.eval()
            self.tokenize = simple_tokenize
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self._torch = torch
            self._available = True
        except ImportError as e:
            print(f"WARNING: Cannot import MinimalVLA: {e}")
            print("  Falling back to random policy.")
            self._available = False
            self._fallback = RandomPolicy(action_dim=action_dim)

    def reset(self):
        if not self._available:
            self._fallback.reset()

    def get_action(self, image: np.ndarray, instruction: str = "pick up the object") -> np.ndarray:
        if not self._available:
            return self._fallback.get_action(image, instruction)

        # 预处理图像
        img_float = image.astype(np.float32) / 255.0
        img_norm = (img_float - 0.5) / 0.5
        img_tensor = self._torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # tokenize instruction
        text_tokens = self.tokenize(instruction)

        with self._torch.no_grad():
            action = self.model(img_tensor, text_tokens)

        return action.squeeze(0).numpy().astype(np.float32)

    @property
    def name(self) -> str:
        return "minimal_vla" if self._available else "random_fallback"


# ---------------------------------------------------------------------------
# 动作执行
# ---------------------------------------------------------------------------

def apply_action(
    client_id: int,
    gripper_id: int,
    cube_id: int,
    action: np.ndarray,
):
    """
    将动作应用到仿真环境中。

    这里使用简化的运动学模型：
    - delta 位置和旋转直接偏移夹爪标记的位置
    - gripper 值控制方块是否被"抓住"（简化物理）

    Args:
        client_id: PyBullet 客户端 ID
        gripper_id: 夹爪标记 ID
        cube_id: 方块 ID
        action: 7-DOF 动作 [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    # 获取当前夹爪位置
    pos, orn = p.getBasePositionAndOrientation(gripper_id, physicsClientId=client_id)
    pos = np.array(pos)
    orn = np.array(orn)

    # 应用 delta 位置
    scale_pos = 0.02  # 缩放因子，控制每步最大位移
    pos[:3] += action[:3] * scale_pos

    # 应用 delta 旋转（简化：使用欧拉角增量）
    scale_rot = 0.05
    euler_delta = action[3:6] * scale_rot
    # 简单地将欧拉角增量加到四元数上（非严格正确，但足够演示）
    if np.linalg.norm(euler_delta) > 1e-6:
        quat_delta = p.getQuaternionFromEuler(euler_delta.tolist())
        orn = p.multiplyTransforms([0, 0, 0], quat_delta, [0, 0, 0], orn.tolist())[1]

    # 限制位置范围（防止飞出场景）
    pos[0] = np.clip(pos[0], -0.3, 0.3)
    pos[1] = np.clip(pos[1], -0.3, 0.3)
    pos[2] = np.clip(pos[2], TABLE_HEIGHT - 0.05, TABLE_HEIGHT + 0.4)

    # 更新夹爪位置
    p.resetBasePositionAndOrientation(
        gripper_id, pos.tolist(), orn.tolist(), physicsClientId=client_id
    )

    # 简化的抓取物理：当 gripper < 0.3 且夹爪靠近方块时，方块跟随夹爪
    gripper_val = action[6]
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id, physicsClientId=client_id)
    cube_pos = np.array(cube_pos)
    dist = np.linalg.norm(pos - cube_pos)

    if gripper_val < 0.3 and dist < 0.06:
        # 方块跟随夹爪移动（简化抓取）
        offset = cube_pos - pos
        new_cube_pos = pos + offset
        # 保持方块在桌面上方
        new_cube_pos[2] = max(new_cube_pos[2], TABLE_HEIGHT + CUBE_SIZE / 2)
        p.resetBasePositionAndOrientation(
            cube_id, new_cube_pos.tolist(), cube_orn.tolist(), physicsClientId=client_id
        )


# ---------------------------------------------------------------------------
# 闭环执行循环
# ---------------------------------------------------------------------------

def run_episode(
    client_id: int,
    cube_id: int,
    gripper_id: int,
    policy,
    instruction: str = "pick up the red block",
    max_steps: int = MAX_EPISODE_STEPS,
    save_images: bool = True,
) -> Dict:
    """
    运行一个完整的闭环 episode。

    流程：获取图像 -> 策略推理 -> 执行动作 -> 物理步进 -> 循环

    Args:
        client_id: PyBullet 客户端 ID
        cube_id: 方块 ID
        gripper_id: 夹爪标记 ID
        policy: 策略对象（需实现 get_action 方法）
        instruction: 语言指令
        max_steps: 最大步数
        save_images: 是否记录图像序列

    Returns:
        episode_data: 包含图像序列、动作序列、元信息的字典
    """
    policy.reset()
    if hasattr(policy, 'set_environment'):
        policy.set_environment(client_id, cube_id)

    images = []
    actions = []
    gripper_positions = []
    cube_positions = []
    rewards = []
    phase_names = []

    print(f"\n{'='*60}")
    print(f"Starting episode: '{instruction}'")
    print(f"Policy: {policy.name}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for step in range(max_steps):
        # 1. 获取当前图像
        rgb_image = get_camera_image(client_id)

        # 2. 策略推理
        action = policy.get_action(rgb_image, instruction)

        # 3. 执行动作
        apply_action(client_id, gripper_id, cube_id, action)

        # 4. 物理步进
        for _ in range(STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=client_id)

        # 5. 记录数据
        if save_images:
            images.append(rgb_image.copy())
        actions.append(action.copy())

        # 记录物体位置
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id, physicsClientId=client_id)
        grip_pos, _ = p.getBasePositionAndOrientation(gripper_id, physicsClientId=client_id)
        cube_positions.append(list(cube_pos))
        gripper_positions.append(list(grip_pos))

        # 简单奖励：方块被提起的高度
        lift_height = cube_pos[2] - TABLE_HEIGHT - CUBE_SIZE / 2
        rewards.append(float(max(0, lift_height)))

        # 记录阶段名称
        if hasattr(policy, 'PHASE_NAMES') and hasattr(policy, 'phase'):
            phase_names.append(policy.PHASE_NAMES.get(policy.phase, 'unknown'))
        else:
            phase_names.append("n/a")

        # 6. 打印进度
        if step % 20 == 0 or step == max_steps - 1:
            elapsed = time.time() - start_time
            fps = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"Step {step:4d}/{max_steps} | "
                f"Action: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, "
                f"g={action[6]:.2f}] | "
                f"Cube Z: {cube_pos[2]:.3f} | "
                f"FPS: {fps:.1f}"
            )

    elapsed = time.time() - start_time
    print(f"\nEpisode finished in {elapsed:.2f}s ({max_steps/elapsed:.1f} Hz)")

    episode_data = {
        "instruction": instruction,
        "policy": policy.name,
        "num_steps": max_steps,
        "elapsed_time": elapsed,
        "actions": actions,
        "cube_positions": cube_positions,
        "gripper_positions": gripper_positions,
        "rewards": rewards,
        "phase_names": phase_names,
        "total_reward": sum(rewards),
    }

    return episode_data


# ---------------------------------------------------------------------------
# 可视化和记录
# ---------------------------------------------------------------------------

def visualize_realtime(
    client_id: int,
    episode_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    使用 matplotlib 实时显示当前图像、预测动作和轨迹。

    Args:
        client_id: PyBullet 客户端 ID
        episode_data: episode 数据
        save_path: 图片保存路径（可选）
        show: 是否显示窗口
    """
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    actions = np.array(episode_data["actions"])
    rewards = np.array(episode_data["rewards"])
    cube_pos = np.array(episode_data["cube_positions"])
    grip_pos = np.array(episode_data["gripper_positions"])

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"VLA Closed-Loop Demo | Policy: {episode_data['policy']} | "
        f"Steps: {episode_data['num_steps']}",
        fontsize=14,
        fontweight="bold",
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- 子图 1: 最后一帧图像 ---
    last_image = get_camera_image(client_id)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(last_image)
    ax_img.set_title("Last Observation (224x224)")
    ax_img.axis("off")

    # --- 子图 2: 7-DOF 动作时间序列 ---
    action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    ax_act = fig.add_subplot(gs[0, 1:])
    for i, (name, color) in enumerate(zip(action_names, colors)):
        ax_act.plot(actions[:, i], label=name, color=color, linewidth=0.8, alpha=0.8)
    ax_act.set_xlabel("Step")
    ax_act.set_ylabel("Action Value")
    ax_act.set_title("7-DOF Action Trajectory")
    ax_act.legend(loc="upper right", fontsize=8, ncol=4)
    ax_act.grid(True, alpha=0.3)
    ax_act.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # --- 子图 3: Gripper 状态 ---
    ax_grip = fig.add_subplot(gs[1, 0])
    ax_grip.fill_between(range(len(actions)), actions[:, 6], alpha=0.3, color="steelblue")
    ax_grip.plot(actions[:, 6], color="steelblue", linewidth=0.8)
    ax_grip.set_xlabel("Step")
    ax_grip.set_ylabel("Gripper (0=close, 1=open)")
    ax_grip.set_title("Gripper State")
    ax_grip.set_ylim(-0.1, 1.1)
    ax_grip.grid(True, alpha=0.3)

    # --- 子图 4: 方块 Z 轴轨迹（提升高度）---
    ax_z = fig.add_subplot(gs[1, 1])
    ax_z.plot(cube_pos[:, 2], color="red", linewidth=1.0, label="Cube Z")
    ax_z.axhline(y=TABLE_HEIGHT + CUBE_SIZE / 2, color="gray",
                  linestyle="--", linewidth=0.5, label="Table surface")
    ax_z.set_xlabel("Step")
    ax_z.set_ylabel("Z Position (m)")
    ax_z.set_title("Object Height")
    ax_z.legend(fontsize=8)
    ax_z.grid(True, alpha=0.3)

    # --- 子图 5: 累积奖励 ---
    ax_rew = fig.add_subplot(gs[1, 2])
    cum_reward = np.cumsum(rewards)
    ax_rew.plot(cum_reward, color="green", linewidth=1.0)
    ax_rew.fill_between(range(len(cum_reward)), cum_reward, alpha=0.2, color="green")
    ax_rew.set_xlabel("Step")
    ax_rew.set_ylabel("Cumulative Reward")
    ax_rew.set_title("Reward Curve")
    ax_rew.grid(True, alpha=0.3)

    # --- 子图 6: XY 平面轨迹 ---
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xy.plot(cube_pos[:, 0], cube_pos[:, 1], "r-", linewidth=1.0, label="Cube", alpha=0.7)
    ax_xy.plot(grip_pos[:, 0], grip_pos[:, 1], "b-", linewidth=1.0, label="Gripper", alpha=0.7)
    ax_xy.plot(cube_pos[0, 0], cube_pos[0, 1], "ro", markersize=8, label="Start")
    ax_xy.plot(cube_pos[-1, 0], cube_pos[-1, 1], "r*", markersize=12, label="End")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectory (Top View)")
    ax_xy.legend(fontsize=8)
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    # --- 子图 7: 3D 轨迹 ---
    ax_3d = fig.add_subplot(gs[2, 1:], projection="3d")
    ax_3d.plot(cube_pos[:, 0], cube_pos[:, 1], cube_pos[:, 2],
               "r-", linewidth=1.0, label="Cube", alpha=0.7)
    ax_3d.plot(grip_pos[:, 0], grip_pos[:, 1], grip_pos[:, 2],
               "b-", linewidth=1.0, label="Gripper", alpha=0.7)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D Trajectory")
    ax_3d.legend(fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def save_episode_data(episode_data: Dict, output_dir: str, episode_idx: int = 0):
    """
    保存 episode 数据（图像序列 + 动作序列）为 JSON。

    Args:
        episode_data: episode 数据字典
        output_dir: 输出目录
        episode_idx: episode 编号
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 保存元数据（不含图像）
    meta = {k: v for k, v in episode_data.items() if k != "images"}
    meta_path = out_path / f"episode_{episode_idx}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Episode data saved to: {meta_path}")

    return meta_path


def save_video(images: List[np.ndarray], output_path: str, fps: float = 10.0):
    """
    使用 imageio 将图像序列保存为 MP4 视频。

    Args:
        images: RGB 图像列表
        output_path: 输出视频路径
        fps: 帧率
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("WARNING: imageio not installed. Cannot save video.")
            print("  Install with: pip install imageio imageio-ffmpeg")
            return

    if not images:
        print("WARNING: No images to save.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

    for img in images:
        writer.append_data(img)

    writer.close()
    print(f"Video saved to: {output_path} ({len(images)} frames, {fps} fps)")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PyBullet Closed-Loop VLA Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python sim_closed_loop_demo.py --mode random
  python sim_closed_loop_demo.py --mode scripted --save_video
  python sim_closed_loop_demo.py --mode scripted --headless --output_dir results/
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "scripted", "minimal_vla"],
        default="random",
        help="策略模式: random(随机), scripted(预编程), minimal_vla(小型VLA模型)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_EPISODE_STEPS,
        help=f"Episode 最大步数 (default: {MAX_EPISODE_STEPS})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式（不打开 GUI，使用 DIRECT 物理引擎）",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="保存 episode 为 MP4 视频",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="输出目录 (default: results/)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the red block",
        help="语言指令",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    print("=" * 60)
    print("PyBullet Closed-Loop VLA Demo")
    print("=" * 60)
    print(f"Mode:       {args.mode}")
    print(f"Instruction: {args.instruction}")
    print(f"Max steps:  {args.max_steps}")
    print(f"Headless:   {args.headless}")
    print(f"Save video: {args.save_video}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # 1. 初始化场景
    print("\n[1/5] Setting up PyBullet scene...")
    client_id, table_id, cube_id, gripper_id = setup_scene(headless=args.headless)
    print(f"  Table ID:   {table_id}")
    print(f"  Cube ID:    {cube_id}")
    print(f"  Gripper ID: {gripper_id}")

    # 2. 创建策略
    print(f"\n[2/5] Creating policy: {args.mode}...")
    if args.mode == "random":
        policy = RandomPolicy(action_dim=ACTION_DIM, seed=args.seed)
    elif args.mode == "scripted":
        policy = ScriptedPolicy(action_dim=ACTION_DIM, client_id=client_id, cube_id=cube_id)
    elif args.mode == "minimal_vla":
        policy = MinimalVLAPolicy(action_dim=ACTION_DIM)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"  Policy name: {policy.name}")

    # 3. 运行闭环 episode
    print(f"\n[3/5] Running closed-loop episode ({args.max_steps} steps)...")
    episode_data = run_episode(
        client_id=client_id,
        cube_id=cube_id,
        gripper_id=gripper_id,
        policy=policy,
        instruction=args.instruction,
        max_steps=args.max_steps,
        save_images=args.save_video,
    )

    # 4. 保存数据
    print(f"\n[4/5] Saving results to {args.output_dir}/...")
    json_path = save_episode_data(episode_data, args.output_dir, episode_idx=0)

    if args.save_video and "images" in episode_data and episode_data["images"]:
        video_path = os.path.join(args.output_dir, "episode_0.mp4")
        save_video(episode_data["images"], video_path, fps=CONTROL_FREQ)

    # 5. 可视化
    print(f"\n[5/5] Generating visualization...")
    viz_path = os.path.join(args.output_dir, "episode_0_viz.png")
    visualize_realtime(
        client_id=client_id,
        episode_data=episode_data,
        save_path=viz_path,
        show=not args.headless,
    )

    # 打印总结
    print(f"\n{'='*60}")
    print("Episode Summary")
    print(f"{'='*60}")
    print(f"  Policy:       {episode_data['policy']}")
    print(f"  Steps:        {episode_data['num_steps']}")
    print(f"  Duration:     {episode_data['elapsed_time']:.2f}s")
    print(f"  Total reward: {episode_data['total_reward']:.4f}")
    print(f"  Avg reward:   {episode_data['total_reward']/max(episode_data['num_steps'],1):.4f}")
    print(f"  Max lift:     {max(episode_data['rewards']):.4f}m")
    print(f"  Results dir:  {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}")

    # 清理
    p.disconnect(physicsClientId=client_id)


if __name__ == "__main__":
    main()