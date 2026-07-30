"""
Unified PushCube Environment — Shared Task for VLA / WM / RL
==============================================================
Dual-cube 2D pushing task where language is *necessary* to identify
the correct target cube.

  VLA:   image + language("push the red cube to the target") -> action
  WM:    predict next state given current state + action
  RL:    learn pushing policy through environment interaction

Two cubes of distinct colors are placed on the table. Only ONE cube
(identified by the language instruction) must be pushed into the
target zone. A vision-only policy cannot disambiguate which cube to
push without the language signal.

State space (13-D): [arm_x, arm_y,
                     cube1_x, cube1_y, cube2_x, cube2_y,
                     target_x, target_y,
                     cube1_r, cube1_g, cube2_r, cube2_g,
                     active_idx]
Action space (2-D): [dx, dy] (arm movement)
Observation (for VLA): 128x128 RGB render + language instruction
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


# Distinct color pairs (R, G) — B is always 0.2
CUBE_COLORS = [
    (0.85, 0.15),  # red
    (0.15, 0.85),  # green
]
COLOR_NAMES = ["red", "green"]


class PushCubeEnv:
    """
    Dual-cube 2D pushing environment.

    The agent controls a pusher arm. Two colored cubes sit on a table.
    Language instruction specifies which cube to push into the target zone.
    """

    def __init__(
        self,
        table_size: float = 1.0,
        arm_speed: float = 0.06,
        cube_size: float = 0.08,
        goal_threshold: float = 0.05,
        max_steps: int = 80,
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
        self.cube_positions: List[np.ndarray] = [None, None]
        self.cube_colors: List[np.ndarray] = [None, None]
        self.target_pos = None
        self.active_idx = 0       # which cube position the language refers to
        self.active_color_idx = 0  # 0=red, 1=green — color of the active cube
        self.step_count = 0
        self._lang_instructions = [None, None]  # per-cube cached instructions

    # ------------------------------------------------------------------
    # Language instruction generation
    # ------------------------------------------------------------------
    def _color_name(self, color_idx: int) -> str:
        return COLOR_NAMES[color_idx]

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
        """Language instruction for the *active* cube."""
        if self._lang_instructions[self.active_idx] is not None:
            return self._lang_instructions[self.active_idx]
        color = COLOR_NAMES[self.active_color_idx]
        direction = self._direction_word(self.target_pos)
        self._lang_instructions[self.active_idx] = (
            f"push the {color} cube to the {direction}"
        )
        return self._lang_instructions[self.active_idx]

    def get_distractor_instruction(self) -> str:
        """Language instruction for the *other* cube (for ablation)."""
        other_color = 1 - self.active_color_idx
        color = COLOR_NAMES[other_color]
        direction = self._direction_word(self.target_pos)
        return f"push the {color} cube to the {direction}"

    def get_shuffled_language(self) -> str:
        """Return the distractor's instruction (for language-shuffled ablation)."""
        return self.get_distractor_instruction()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.step_count = 0
        self._lang_instructions = [None, None]

        limit = self.table_size / 2

        # Arm starts at a random edge position
        self.arm_pos = np.array(
            [self.rng.uniform(-0.4, -0.2), self.rng.uniform(-0.2, 0.2)],
            dtype=np.float32,
        )

        # Two cubes at well-separated random positions
        self.cube_positions = []
        for i in range(2):
            pos = self.rng.uniform(-0.4, 0.4, size=2).astype(np.float32)
            # Ensure cubes are far apart from each other and from arm
            while (
                np.linalg.norm(pos - self.arm_pos) < 0.2
                or (i == 1 and np.linalg.norm(pos - self.cube_positions[0]) < 0.25)
            ):
                pos = self.rng.uniform(-0.4, 0.4, size=2).astype(np.float32)
            self.cube_positions.append(pos)

        # Assign colors — randomize which color the active cube gets
        self.active_idx = self.rng.randint(2)
        self.active_color_idx = self.rng.randint(2)
        self.cube_colors = [None, None]
        self.cube_colors[self.active_idx] = np.array(
            CUBE_COLORS[self.active_color_idx], dtype=np.float32
        )
        self.cube_colors[1 - self.active_idx] = np.array(
            CUBE_COLORS[1 - self.active_color_idx], dtype=np.float32
        )

        # Target at random position, far from both cubes
        for _ in range(100):
            self.target_pos = self.rng.uniform(-0.45, 0.45, size=2).astype(np.float32)
            if all(
                np.linalg.norm(self.target_pos - c) > 0.2
                for c in self.cube_positions
            ):
                break

        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        movement = action * self.arm_speed
        new_arm = self.arm_pos + movement

        limit = self.table_size / 2
        new_arm = np.clip(new_arm, -limit, limit)

        # Collision check: arm can push either cube
        for i in range(2):
            dist = np.linalg.norm(new_arm - self.cube_positions[i])
            if dist < (self.cube_size / 2 + 0.03):
                push_dir = new_arm - self.arm_pos
                if np.linalg.norm(push_dir) > 1e-6:
                    push_dir = push_dir / np.linalg.norm(push_dir)
                    self.cube_positions[i] = self.cube_positions[i] + push_dir * 0.04
                    self.cube_positions[i] = np.clip(
                        self.cube_positions[i], -limit, limit
                    )

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
            "cube_positions": [c.copy() for c in self.cube_positions],
            "cube_colors": [c.copy() for c in self.cube_colors],
            "target_pos": self.target_pos.copy(),
            "active_idx": self.active_idx,
        }

    def _compute_reward(self) -> float:
        active_cube = self.cube_positions[self.active_idx]
        dist = np.linalg.norm(active_cube - self.target_pos)
        reward = -dist
        if dist < self.goal_threshold:
            reward += 1.0
        return reward

    def _check_success(self) -> bool:
        active_cube = self.cube_positions[self.active_idx]
        return np.linalg.norm(active_cube - self.target_pos) < self.goal_threshold

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, size: int = 128) -> np.ndarray:
        img = np.ones((size, size, 3), dtype=np.float32) * 0.9

        def to_pix(pos):
            return int((pos + 0.5) / 1.0 * (size - 1))

        # Target zone
        tx, ty = to_pix(self.target_pos[0]), to_pix(self.target_pos[1])
        r = max(int(self.goal_threshold / 0.5 * size), 3)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - tx) ** 2 + (yy - ty) ** 2 <= r ** 2
        img[mask] = [0.7, 1.0, 0.7]

        # Both cubes
        for i in range(2):
            cx, cy = to_pix(self.cube_positions[i][0]), to_pix(self.cube_positions[i][1])
            cr = max(int(self.cube_size / 2 / 0.5 * size), 3)
            yy, xx = np.ogrid[:size, :size]
            mask = (np.abs(xx - cx) <= cr) & (np.abs(yy - cy) <= cr)
            img[mask] = [self.cube_colors[i][0], self.cube_colors[i][1], 0.2]

        # Arm
        ax, ay = to_pix(self.arm_pos[0]), to_pix(self.arm_pos[1])
        ar = max(int(0.03 / 0.5 * size), 2)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - ax) ** 2 + (yy - ay) ** 2 <= ar ** 2
        img[mask] = [0.2, 0.2, 0.8]

        return np.clip(img, 0, 1)

    # ------------------------------------------------------------------
    # State vector helpers
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return 13

    @property
    def action_dim(self) -> int:
        return 2

    def get_state_vector(self) -> np.ndarray:
        return np.concatenate([
            self.arm_pos,
            self.cube_positions[0],
            self.cube_positions[1],
            self.target_pos,
            self.cube_colors[0],
            self.cube_colors[1],
            np.array([self.active_idx], dtype=np.float32),
        ]).astype(np.float32)


# ------------------------------------------------------------------
# Expert policy: "go behind cube → contact → push toward target"
# ------------------------------------------------------------------
def expert_action(env: PushCubeEnv) -> np.ndarray:
    """
    High-success-rate heuristic:
    1. Compute the approach point behind the active cube (opposite from target).
    2. If not yet at approach point, move toward it.
    3. Once behind the cube, push toward target.
    """
    active_cube = env.cube_positions[env.active_idx]
    target = env.target_pos
    arm = env.arm_pos

    # Direction from target to cube — we need to approach from this side
    dir_target_to_cube = active_cube - target
    dist_t2c = np.linalg.norm(dir_target_to_cube)
    if dist_t2c < 1e-6:
        dir_target_to_cube = np.array([1.0, 0.0])
    else:
        dir_target_to_cube = dir_target_to_cube / dist_t2c

    # Approach point: behind the cube, offset by cube_size
    approach_point = active_cube + dir_target_to_cube * (env.cube_size / 2 + 0.04)

    dist_arm_to_approach = np.linalg.norm(approach_point - arm)

    if dist_arm_to_approach > 0.03:
        # Phase 1: move to approach point (behind the cube)
        direction = approach_point - arm
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        action = direction * 0.9
    else:
        # Phase 2: push cube toward target
        direction = target - active_cube
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        action = direction * 0.9

    return np.clip(action, -1.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = PushCubeEnv(seed=42)
    obs = env.reset(seed=42)
    print("Language:", env.get_language_instruction())
    print("Distractor:", env.get_distractor_instruction())
    print("Active cube index:", env.active_idx)
    print("State dim:", env.state_dim)

    success_count = 0
    n_test = 20
    for ep in range(n_test):
        env = PushCubeEnv()
        obs = env.reset(seed=ep)
        for _ in range(env.max_steps):
            action = expert_action(env)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                success_count += 1
                break
            if truncated:
                break

    print(f"\nExpert success rate: {success_count}/{n_test} = {success_count/n_test*100:.1f}%")
    img = env.render(size=128)
    print("Render shape:", img.shape)
