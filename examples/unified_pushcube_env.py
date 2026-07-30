"""
Unified PushCube Environment — Shared Task for VLA / WM / RL
==============================================================
A lightweight 2D pushing task that all three tracks can share:

  VLA:   image + language("push the red cube right") -> action
  WM:    predict next cube position given current state + action
  RL:    learn pushing policy through environment interaction

State space (8-D): [arm_x, arm_y, cube_x, cube_y, target_x, target_y,
                     cube_color_r, cube_color_g]
Action space (2-D): [dx, dy] (arm movement)
Observation (for VLA): 128x128 RGB render + language instruction
"""

import numpy as np
from typing import Tuple, Dict, Optional


class PushCubeEnv:
    """
    Lightweight 2D pushing environment.

    The agent controls a pusher arm. A colored cube sits on a table.
    Goal: push the cube into a target zone.
    """

    def __init__(
        self,
        table_size: float = 1.0,
        arm_speed: float = 0.05,
        cube_size: float = 0.08,
        goal_threshold: float = 0.05,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ):
        self.table_size = table_size
        self.arm_speed = arm_speed
        self.cube_size = cube_size
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.arm_pos = None
        self.cube_pos = None
        self.target_pos = None
        self.cube_color = None
        self.step_count = 0
        self._lang_instruction = None

    # ------------------------------------------------------------------
    # Language instruction generation
    # ------------------------------------------------------------------
    def _color_name(self, rgb) -> str:
        r, g = rgb
        if r > 0.7 and g < 0.4:
            return "red"
        if r < 0.4 and g > 0.7:
            return "green"
        if r > 0.7 and g > 0.7:
            return "yellow"
        return "blue"

    def _direction_word(self, target) -> str:
        tx, ty = target
        words = []
        if tx > 0.15:
            words.append("right")
        elif tx < -0.15:
            words.append("left")
        if ty > 0.15:
            words.append("top")
        elif ty < -0.15:
            words.append("bottom")
        return " and ".join(words) if words else "center"

    def get_language_instruction(self) -> str:
        """Generate a natural-language instruction for current task."""
        if self._lang_instruction is not None:
            return self._lang_instruction
        color = self._color_name(self.cube_color)
        direction = self._direction_word(self.target_pos)
        self._lang_instruction = f"push the {color} cube to the {direction}"
        return self._lang_instruction

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.step_count = 0
        self._lang_instruction = None

        # Arm starts at center-left
        self.arm_pos = np.array([-0.3, 0.0], dtype=np.float32)

        # Cube at random position (avoiding center)
        self.cube_pos = self.rng.uniform(-0.4, 0.4, size=2).astype(np.float32)
        while np.linalg.norm(self.cube_pos) < 0.15:
            self.cube_pos = self.rng.uniform(-0.4, 0.4, size=2).astype(np.float32)

        # Target at random position on opposite side
        angle = self.rng.uniform(0, 2 * np.pi)
        dist = self.rng.uniform(0.3, 0.45)
        self.target_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)

        # Cube color (RG, B is always 0.2)
        colors = [
            np.array([0.85, 0.15], dtype=np.float32),  # red
            np.array([0.15, 0.85], dtype=np.float32),  # green
            np.array([0.85, 0.85], dtype=np.float32),  # yellow
        ]
        self.cube_color = colors[self.rng.randint(len(colors))]

        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        movement = action * self.arm_speed
        new_arm = self.arm_pos + movement

        # Clip to table
        limit = self.table_size / 2
        new_arm = np.clip(new_arm, -limit, limit)

        # Collision check: if arm touches cube, push cube
        dist = np.linalg.norm(new_arm - self.cube_pos)
        if dist < (self.cube_size / 2 + 0.03):
            # Push cube in same direction, with some damping
            push_dir = new_arm - self.arm_pos
            if np.linalg.norm(push_dir) > 1e-6:
                push_dir = push_dir / np.linalg.norm(push_dir)
                self.cube_pos = self.cube_pos + push_dir * 0.04
                self.cube_pos = np.clip(self.cube_pos, -limit, limit)

        self.arm_pos = new_arm
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        success = self._check_success()
        terminated = bool(success)
        truncated = self.step_count >= self.max_steps
        info = {"is_success": success, "steps": self.step_count}

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> Dict:
        return {
            "arm_pos": self.arm_pos.copy(),
            "cube_pos": self.cube_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "cube_color": self.cube_color.copy(),
        }

    def _compute_reward(self) -> float:
        dist = np.linalg.norm(self.cube_pos - self.target_pos)
        # Reward: negative distance, plus bonus for success
        reward = -dist
        if dist < self.goal_threshold:
            reward += 1.0
        return reward

    def _check_success(self) -> bool:
        return np.linalg.norm(self.cube_pos - self.target_pos) < self.goal_threshold

    # ------------------------------------------------------------------
    # Rendering (for VLA image observation)
    # ------------------------------------------------------------------
    def render(self, size: int = 128) -> np.ndarray:
        """Render a size x size RGB image."""
        img = np.ones((size, size, 3), dtype=np.float32) * 0.9  # light table

        # Coordinate transform: [-0.5, 0.5] -> [0, size-1]
        def to_pix(pos):
            return int((pos + 0.5) / 1.0 * (size - 1))

        # Draw target zone
        tx, ty = to_pix(self.target_pos[0]), to_pix(self.target_pos[1])
        r = int(self.goal_threshold / 0.5 * size)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - tx) ** 2 + (yy - ty) ** 2 <= r ** 2
        img[mask] = [0.7, 1.0, 0.7]  # light green target

        # Draw cube
        cx, cy = to_pix(self.cube_pos[0]), to_pix(self.cube_pos[1])
        cr = int(self.cube_size / 2 / 0.5 * size)
        cr = max(cr, 3)
        yy, xx = np.ogrid[:size, :size]
        mask = (np.abs(xx - cx) <= cr) & (np.abs(yy - cy) <= cr)
        img[mask] = [self.cube_color[0], self.cube_color[1], 0.2]

        # Draw arm
        ax, ay = to_pix(self.arm_pos[0]), to_pix(self.arm_pos[1])
        ar = int(0.03 / 0.5 * size)
        ar = max(ar, 2)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - ax) ** 2 + (yy - ay) ** 2 <= ar ** 2
        img[mask] = [0.2, 0.2, 0.8]  # blue arm

        return np.clip(img, 0, 1)

    # ------------------------------------------------------------------
    # Helpers for dataset collection
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return 8

    @property
    def action_dim(self) -> int:
        return 2

    def get_state_vector(self) -> np.ndarray:
        """Flatten state for RL / WM."""
        return np.concatenate([
            self.arm_pos,
            self.cube_pos,
            self.target_pos,
            self.cube_color,
        ]).astype(np.float32)


# ----------------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = PushCubeEnv(seed=42)
    obs = env.reset(seed=42)
    print("Language:", env.get_language_instruction())
    print("Initial state:", obs)

    for i in range(5):
        action = np.random.randn(2).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}, success={info['is_success']}")
        if done:
            break

    img = env.render(size=128)
    print("Render shape:", img.shape)
