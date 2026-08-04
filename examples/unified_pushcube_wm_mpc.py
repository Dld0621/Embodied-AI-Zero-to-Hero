"""
Unified PushCube — World Model + MPC Control Loop
=================================================
Connect the trained World Model to a Model Predictive Controller (MPC)
to close the loop: WM prediction → action optimization → environment execution.

This implements the Dreamer/MuZero idea:
    World Model → Planner → Action → Environment → Observation → World Model

Two planners are implemented:
  1. Random Shooting — sample N action sequences, pick the best by predicted reward
  2. CEM (Cross-Entropy Method) — iterative refinement of the action distribution

The World Model (from unified_pushcube_wm.py) predicts:
    (state, action) → (next_state, reward)

Usage:
    # First train the WM:
    python unified_pushcube_wm.py --n-episodes 200 --epochs 50

    # Then run MPC:
    python unified_pushcube_wm_mpc.py --planner random_shooting --n_eval 20
    python unified_pushcube_wm_mpc.py --planner cem --n_eval 20

This is a teaching implementation showing how a world model becomes a
controller, not just a predictor.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv, expert_action


# ----------------------------------------------------------------------
# World Model (same architecture as unified_pushcube_wm.py)
# ----------------------------------------------------------------------
def build_world_model(state_dim=14, action_dim=2):
    """Build the same TinyWorldModel architecture used in training."""
    import torch
    import torch.nn as nn

    class TinyWorldModel(nn.Module):
        def __init__(self, state_dim=14, action_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.next_state_head = nn.Linear(64, state_dim)
            self.reward_head = nn.Linear(64, 1)

        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            h = self.encoder(x)
            next_state = self.next_state_head(h)
            reward = self.reward_head(h)
            return next_state, reward

    return TinyWorldModel(state_dim=state_dim, action_dim=action_dim)


# ----------------------------------------------------------------------
# Reward computation from state (ground truth, for comparison)
# ----------------------------------------------------------------------
def compute_reward_from_state(state, goal_threshold=0.05):
    """Compute the true reward from a state vector.

    State (14-D): [arm_x, arm_y, cube1_x, cube1_y, cube2_x, cube2_y,
                    target_x, target_y, cube1_r, cube1_g, cube2_r, cube2_g,
                    goal_red, goal_green]
    """
    # Determine active cube from goal color one-hot
    goal_red = state[12]
    goal_green = state[13]

    if goal_red > 0.5:
        # Active cube is red → cube_positions[0] (cube1_x, cube1_y)
        active_cube = state[2:4]
    else:
        # Active cube is green → cube_positions[1] (cube2_x, cube2_y)
        active_cube = state[4:6]

    target = state[6:8]
    dist = np.linalg.norm(active_cube - target)
    reward = -dist
    if dist < goal_threshold:
        reward += 1.0
    return reward


# ----------------------------------------------------------------------
# Planners
# ----------------------------------------------------------------------
class RandomShootingPlanner:
    """Random Shooting MPC: sample N action sequences, pick the best.

    Simplest model-based planner. For each candidate sequence, use the
    world model to roll out the trajectory and accumulate predicted reward.
    """

    def __init__(self, wm, device, horizon=10, n_samples=500, action_dim=2):
        self.wm = wm
        self.device = device
        self.horizon = horizon
        self.n_samples = n_samples
        self.action_dim = action_dim

    def plan(self, state):
        """Return the best first action given the current state."""
        import torch

        # Sample N random action sequences: (N, horizon, action_dim)
        actions = np.random.uniform(-1, 1, size=(self.n_samples, self.horizon, self.action_dim)).astype(np.float32)

        # Convert to tensor
        actions_t = torch.tensor(actions, device=self.device)  # (N, H, A)

        # Roll out each sequence through the world model
        # Start state: broadcast to (N, state_dim)
        state_np = np.tile(state, (self.n_samples, 1)).astype(np.float32)
        current_state = torch.tensor(state_np, device=self.device)

        cumulative_reward = torch.zeros(self.n_samples, device=self.device)

        for t in range(self.horizon):
            action_t = actions_t[:, t, :]  # (N, A)
            with torch.no_grad():
                pred_next_state, pred_reward = self.wm(current_state, action_t)
                pred_reward = pred_reward.squeeze(-1)  # (N,)
            cumulative_reward += pred_reward
            current_state = pred_next_state

        # Pick the best sequence
        best_idx = torch.argmax(cumulative_reward).item()
        best_action = actions[best_idx, 0, :]  # first action of best sequence

        return best_action


class CEMPlanner:
    """Cross-Entropy Method MPC.

    Iteratively refine a Gaussian distribution over action sequences:
    1. Sample N sequences from current distribution
    2. Roll out each through the world model
    3. Select top-K (elite) sequences
    4. Update distribution mean/std from elites
    5. Repeat for n_iterations

    More sample-efficient than random shooting.
    """

    def __init__(self, wm, device, horizon=10, n_samples=500, n_elite=50,
                 n_iterations=3, action_dim=2, init_std=0.5):
        self.wm = wm
        self.device = device
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.action_dim = action_dim
        self.init_std = init_std

    def plan(self, state):
        """Return the best first action given the current state."""
        import torch

        # Initialize distribution
        mean = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        std = np.ones((self.horizon, self.action_dim), dtype=np.float32) * self.init_std

        for iteration in range(self.n_iterations):
            # Sample N action sequences from current distribution
            actions = mean + std * np.random.randn(self.n_samples, self.horizon, self.action_dim).astype(np.float32)
            actions = np.clip(actions, -1.0, 1.0)

            actions_t = torch.tensor(actions, device=self.device)

            # Roll out through world model
            state_np = np.tile(state, (self.n_samples, 1)).astype(np.float32)
            current_state = torch.tensor(state_np, device=self.device)

            cumulative_reward = torch.zeros(self.n_samples, device=self.device)

            for t in range(self.horizon):
                action_t = actions_t[:, t, :]
                with torch.no_grad():
                    pred_next_state, pred_reward = self.wm(current_state, action_t)
                    pred_reward = pred_reward.squeeze(-1)
                cumulative_reward += pred_reward
                current_state = pred_next_state

            # Select elite samples
            rewards_np = cumulative_reward.cpu().numpy()
            elite_indices = np.argsort(rewards_np)[-self.n_elite:]
            elite_actions = actions[elite_indices]

            # Update distribution
            mean = elite_actions.mean(axis=0)
            std = elite_actions.std(axis=0) + 1e-3  # prevent collapse

        # Return mean first action (best estimate)
        return mean[0]


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_mpc(wm, device, planner_name, n_eval=20, horizon=10,
                 n_samples=500, n_elite=50, n_iterations=3, seed=42):
    """Evaluate WM-MPC on PushCube."""
    np.random.seed(seed)

    # Build planner
    if planner_name == "random_shooting":
        planner = RandomShootingPlanner(
            wm, device, horizon=horizon, n_samples=n_samples
        )
    elif planner_name == "cem":
        planner = CEMPlanner(
            wm, device, horizon=horizon, n_samples=n_samples,
            n_elite=n_elite, n_iterations=n_iterations
        )
    else:
        raise ValueError(f"Unknown planner: {planner_name}")

    print(f"\nEvaluating WM-MPC ({planner_name}) on {n_eval} episodes...")
    print(f"  Horizon: {horizon}, Samples: {n_samples}")
    if planner_name == "cem":
        print(f"  Elite: {n_elite}, Iterations: {n_iterations}")

    success_count = 0
    step_total = 0
    rewards_all = []

    for ep in range(n_eval):
        env = PushCubeEnv()
        obs = env.reset(seed=8000 + ep)
        state = env.get_state_vector()

        ep_reward = 0.0
        done = False

        for step in range(env.max_steps):
            # Plan action using WM-MPC
            action = planner.plan(state)

            # Execute in real environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_total += 1

            # Update state
            state = env.get_state_vector()

            if terminated:
                success_count += 1
                done = True
                break
            if truncated:
                break

        rewards_all.append(ep_reward)
        status = "SUCCESS" if (done and info.get("is_success")) else "fail"
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Eval {ep+1}/{n_eval}: {status} (reward={ep_reward:.2f}, steps={step+1})")

    success_rate = success_count / n_eval * 100 if n_eval > 0 else 0.0
    avg_reward = float(np.mean(rewards_all)) if rewards_all else 0.0
    avg_steps = step_total / n_eval if n_eval > 0 else 0.0

    print(f"\nResults ({planner_name}):")
    print(f"  Success rate: {success_count}/{n_eval} = {success_rate:.1f}%")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg steps: {avg_steps:.1f}")

    return {
        "success_rate": round(success_rate, 1),
        "success_count": success_count,
        "n_eval": n_eval,
        "avg_reward": round(avg_reward, 2),
        "avg_steps": round(avg_steps, 1),
        "planner": planner_name,
        "horizon": horizon,
        "n_samples": n_samples,
        "n_elite": n_elite if planner_name == "cem" else None,
        "n_iterations": n_iterations if planner_name == "cem" else None,
    }


# ----------------------------------------------------------------------
# Expert baseline (for comparison)
# ----------------------------------------------------------------------
def evaluate_expert(n_eval=20, seed=42):
    """Evaluate the expert policy for comparison."""
    np.random.seed(seed)
    success_count = 0

    for ep in range(n_eval):
        env = PushCubeEnv()
        env.reset(seed=8000 + ep)

        for step in range(env.max_steps):
            action = expert_action(env)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                success_count += 1
                break
            if truncated:
                break

    success_rate = success_count / n_eval * 100
    print(f"\nExpert baseline: {success_count}/{n_eval} = {success_rate:.1f}%")
    return {"success_rate": round(success_rate, 1), "success_count": success_count, "n_eval": n_eval}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import torch

    parser = argparse.ArgumentParser(
        description="Unified PushCube — World Model + MPC Control Loop"
    )
    parser.add_argument("--planner", type=str, default="cem",
                        choices=["random_shooting", "cem"],
                        help="MPC planner type (default: cem)")
    parser.add_argument("--wm-checkpoint", type=str,
                        default="../results/unified_pushcube/wm/pushcube_wm.pt",
                        help="Path to trained world model checkpoint")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Planning horizon (default: 10)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of action sequences to sample (default: 500)")
    parser.add_argument("--n-elite", type=int, default=50,
                        help="Number of elite samples for CEM (default: 50)")
    parser.add_argument("--n-iterations", type=int, default=3,
                        help="CEM iterations (default: 3)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str,
                        default="../results/unified_pushcube/wm_mpc",
                        help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test: 3 episodes, small samples")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_eval = 3
        args.n_samples = 50
        args.horizon = 5
        print("[SMOKE TEST] n_eval=3, n_samples=50, horizon=5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" Unified PushCube — World Model + MPC Control Loop")
    print("=" * 70)
    print(f"Device: {device}")

    # Load trained world model
    wm = build_world_model(state_dim=14, action_dim=2).to(device)
    ckpt_path = Path(args.wm_checkpoint)

    if ckpt_path.exists():
        wm.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded WM checkpoint: {ckpt_path}")
    else:
        print(f"[Warning] WM checkpoint not found: {ckpt_path}")
        print("  Using randomly initialized WM (for testing only)")
        print("  Train first: python unified_pushcube_wm.py --n-episodes 200 --epochs 50")

    wm.eval()

    # Evaluate WM-MPC
    results = evaluate_mpc(
        wm, device, args.planner, n_eval=args.n_eval,
        horizon=args.horizon, n_samples=args.n_samples,
        n_elite=args.n_elite, n_iterations=args.n_iterations,
        seed=args.seed,
    )

    # Also evaluate expert for comparison
    expert_results = evaluate_expert(n_eval=args.n_eval, seed=args.seed)

    # Compare WM-predicted reward vs ground-truth reward
    print("\n--- WM Prediction Quality Check ---")
    test_env = PushCubeEnv()
    test_env.reset(seed=9999)
    test_state = test_env.get_state_vector()
    test_action = np.array([0.5, -0.3], dtype=np.float32)

    with torch.no_grad():
        s_t = torch.tensor(test_state, dtype=torch.float32).unsqueeze(0).to(device)
        a_t = torch.tensor(test_action, dtype=torch.float32).unsqueeze(0).to(device)
        pred_ns, pred_r = wm(s_t, a_t)
        pred_ns = pred_ns.cpu().numpy()[0]
        pred_r = pred_r.cpu().numpy()[0, 0]

    # Ground truth
    obs, true_reward, _, _, _ = test_env.step(test_action)
    true_state = test_env.get_state_vector()
    true_r = compute_reward_from_state(true_state)

    print(f"  Predicted reward: {pred_r:.4f}")
    print(f"  True reward (WM): {true_reward:.4f}")
    print(f"  True reward (state): {true_r:.4f}")
    print(f"  State prediction error (L2): {np.linalg.norm(pred_ns - true_state):.4f}")

    # Save results
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "task": "PushCube World Model + MPC Control Loop",
        "description": "World Model predicts (state, action) → (next_state, reward). "
                       "MPC planner uses WM to optimize action sequences. "
                       "Closes the loop: WM → Planner → Action → Environment.",
        "wm_checkpoint": str(ckpt_path),
        "planner": args.planner,
        "horizon": args.horizon,
        "n_samples": args.n_samples,
        "n_elite": args.n_elite if args.planner == "cem" else None,
        "n_iterations": args.n_iterations if args.planner == "cem" else None,
        "mpc_results": results,
        "expert_baseline": expert_results,
        "wm_prediction_check": {
            "predicted_reward": round(float(pred_r), 4),
            "true_reward_wm": round(float(true_reward), 4),
            "true_reward_state": round(float(true_r), 4),
            "state_prediction_error_l2": round(float(np.linalg.norm(pred_ns - true_state)), 4),
        },
        "smoke_test": args.smoke_test,
    }

    from benchmark_provenance import build_provenance
    full_results["provenance"] = build_provenance(
        command=f"python {__file__} --planner {args.planner} --n_eval {args.n_eval}",
        result_generated_by=__file__,
    )

    results_path = save_dir / f"wm_mpc_{args.planner}_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print(" WM-MPC Control Loop Summary")
    print("=" * 70)
    print(f"  Planner:          {args.planner}")
    print(f"  WM-MPC Success:   {results['success_rate']}% ({results['success_count']}/{results['n_eval']})")
    print(f"  Expert Success:   {expert_results['success_rate']}% ({expert_results['success_count']}/{expert_results['n_eval']})")
    print(f"  Avg Reward (MPC): {results['avg_reward']}")
    print(f"  Avg Steps (MPC):  {results['avg_steps']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
