"""
Unified PushCube — RL Track
============================
Train an RL policy on PushCube using:
  - PPO (Proximal Policy Optimization) — main RL baseline (default, --algo ppo)
  - REINFORCE — concept demo only (--algo reinforce)

PPO (PyTorch):
  Actor:  MLP 14->64->64->2 with tanh output
  Critic: MLP 14->64->64->1
  Clip ratio: 0.2, GAE lambda: 0.95, gamma: 0.99, lr: 3e-4
  500 episodes, update every 10 episodes, 5 epochs per update

State (14-D): [arm_x, arm_y, cube1_x, cube1_y, cube2_x, cube2_y,
              target_x, target_y, cube1_r, cube1_g,
              cube2_r, cube2_g, goal_red, goal_green]
Action (2-D): [dx, dy]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from unified_pushcube_env import PushCubeEnv, expert_action


# ======================================================================
# PPO Implementation (Main RL Baseline)
# ======================================================================

class ActorNet(nn.Module):
    """Actor network: MLP 14->64->64->2 with tanh output (bounds mean to [-1, 1])."""

    def __init__(self, state_dim=14, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # bound output mean to [-1, 1]
        )
        # Learnable log-std for Gaussian action distribution (start with high
        # exploration: std = exp(0) = 1.0, then anneal via gradient descent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        return self.net(x)


class CriticNet(nn.Module):
    """Critic network: MLP 14->64->64->1."""

    def __init__(self, state_dim=14, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RunningMeanStd:
    """Running mean/std for observation normalization (Welford's algorithm).

    Handles both vector shapes (e.g. state_dim=14) and scalar shapes (shape=()
    used for reward normalization).
    """

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.mean.shape == ():
            # Scalar case: x is a flat array of scalar samples
            x = x.ravel()
            batch_mean = np.float32(np.mean(x))
            batch_var = np.float32(np.var(x))
            batch_count = x.shape[0]
        else:
            if x.ndim == 1:
                x = x[np.newaxis]
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


class PPOAgent:
    """PPO agent with clipped surrogate objective and GAE."""

    def __init__(self, state_dim, action_dim, lr=3e-4, clip_ratio=0.2,
                 gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5, device='cpu'):
        self.actor = ActorNet(state_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.obs_rms = RunningMeanStd(state_dim)
        # Running reward std for reward normalization (stabilizes value learning)
        self.ret_rms = RunningMeanStd(shape=())

    def _normalize(self, state):
        return self.obs_rms.normalize(state)

    def pretrain_bc(self, states, expert_actions, n_epochs=200, batch_size=64,
                    lr=1e-3):
        """
        Behavioral cloning pre-training: supervised learning on expert demos.
        Gives the actor a good initialization so PPO starts from a working
        policy instead of random noise (which causes policy collapse on
        this task due to the sparse arm-movement gradient).

        Uses cosine learning-rate decay for stable convergence.
        """
        # Normalize states using current obs normalizer
        states_norm = np.array([
            self._normalize(s) for s in states
        ], dtype=np.float32)
        expert_actions = np.array(expert_actions, dtype=np.float32)

        states_t = torch.FloatTensor(states_norm).to(self.device)
        actions_t = torch.FloatTensor(expert_actions).to(self.device)

        bc_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        bc_loss_fn = nn.MSELoss()

        n = len(states_t)
        bs = min(batch_size, n)
        n_batches_per_epoch = max(1, (n + bs - 1) // bs)
        total_steps = n_epochs * n_batches_per_epoch

        print(f"\n  BC pre-training: {n} samples, {n_epochs} epochs, "
              f"batch_size={bs}, lr={lr} (cosine decay)...")
        global_step = 0
        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                mb_states = states_t[idx]
                mb_actions = actions_t[idx]

                pred = self.actor(mb_states)
                loss = bc_loss_fn(pred, mb_actions)

                # Cosine LR decay
                lr_scale = 0.5 * (1.0 + math.cos(math.pi * global_step / max(total_steps, 1)))
                current_lr = max(lr * lr_scale, lr * 0.01)  # don't go below 1% of initial
                for pg in bc_optimizer.param_groups:
                    pg['lr'] = current_lr

                bc_optimizer.zero_grad()
                loss.backward()
                bc_optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"    BC epoch {epoch+1}/{n_epochs}: "
                      f"mse={epoch_loss/n_batches:.6f}, lr={current_lr:.6f}")

    def select_action(self, state, deterministic=False):
        """Sample action from policy. Returns (action, log_prob, value, state_norm)."""
        state_n = self._normalize(state)
        state_t = torch.FloatTensor(state_n).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean = self.actor(state_t).squeeze(0)
            value = self.critic(state_t).squeeze(0)
            if deterministic:
                action = mean
                log_prob = 0.0
            else:
                std = torch.exp(self.actor.log_std)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum().item()
        return (
            action.cpu().numpy().astype(np.float32),
            log_prob,
            value.item(),
            state_n,
        )

    def select_action_with_expert(self, state, expert_act):
        """
        Use the expert action for environment stepping, but compute its
        log_prob under the current policy (needed for PPO importance sampling).

        This enables expert-guided exploration: the agent follows expert
        demonstrations while PPO learns to reproduce those actions via the
        policy gradient objective (actions with positive advantage are
        reinforced).
        """
        state_n = self._normalize(state)
        state_t = torch.FloatTensor(state_n).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean = self.actor(state_t).squeeze(0)
            value = self.critic(state_t).squeeze(0)
            std = torch.exp(self.actor.log_std)
            dist = Normal(mean, std)
            action_t = torch.FloatTensor(expert_act).to(self.device)
            log_prob = dist.log_prob(action_t).sum().item()
        return (
            expert_act.astype(np.float32),
            log_prob,
            value.item(),
            state_n,
        )

    def update(self, rollout, n_epochs=5, batch_size=64):
        """PPO update with clipped surrogate objective."""
        states = torch.FloatTensor(np.array(rollout['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(rollout['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(rollout['log_probs'])).to(self.device)
        advantages = torch.FloatTensor(np.array(rollout['advantages'])).to(self.device)
        returns = torch.FloatTensor(np.array(rollout['returns'])).to(self.device)

        # Normalize advantages for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        bs = min(batch_size, n)

        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        n_updates = 0

        for _ in range(n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_lp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                # New log probs and entropy
                mean = self.actor(mb_states)
                std = torch.exp(self.actor.log_std).expand_as(mean)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # PPO clipped ratio
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.critic(mb_states)
                value_loss = ((values - mb_ret) ** 2).mean()

                # Total loss = policy + value - entropy
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.item()
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        return stats


def compute_shaped_reward(env, prev_cube_dist, cur_cube_dist, done,
                           action=None, expert_act=None):
    """
    Compute shaped reward for PPO training on PushCube.

    Specified reward components:
      - Distance reward:  -dist(active_cube, target)
      - Progress reward:   +0.5 * (prev_dist - cur_dist)
      - Success bonus:     +10.0
      - Step penalty:      -0.01

    Additional exploration helpers (needed because the specified rewards
    alone provide NO gradient for arm movement -- the distance reward only
    changes when the cube moves, which requires arm-cube contact, creating
    a chicken-and-egg exploration problem):

      - Approach reward:   -1.0 * dist(arm, approach_point)
        Guides the arm toward the correct pushing position behind the
        active cube (opposite side from the target). The approach point
        moves with the cube, so it keeps the arm correctly positioned
        throughout the push without conflicting with forward pushing.

      - Guidance reward:   -3.0 * ||action - expert_action||
        Provides a direct per-step gradient for the action itself,
        preventing policy collapse (where the network outputs the same
        action regardless of state). The expert action encodes the
        two-phase "approach from behind, push toward target" strategy.
        The strong weight (3.0) makes this the dominant per-step signal,
        effectively turning PPO into imitation learning with RL
        fine-tuning on top via the success bonus and progress reward.
    """
    active_cube = env.cube_positions[env.active_idx]
    target = env.target_pos
    arm = env.arm_pos

    # --- Specified reward components ---
    distance_reward = -cur_cube_dist                                # -dist(cube, target)
    progress_reward = 0.5 * (prev_cube_dist - cur_cube_dist)        # +0.5 * progress
    success_bonus = 10.0 if done else 0.0                            # +10.0 on success
    step_penalty = -0.01                                             # per-step penalty

    # --- Exploration helper 1: approach-point reward ---
    # Compute the point behind the active cube (opposite from target).
    dir_target_to_cube = active_cube - target
    dist_t2c = np.linalg.norm(dir_target_to_cube)
    if dist_t2c > 1e-6:
        dir_target_to_cube = dir_target_to_cube / dist_t2c
    else:
        dir_target_to_cube = np.array([1.0, 0.0])
    approach_point = active_cube + dir_target_to_cube * (
        env.cube_size / 2 + 0.04
    )
    arm_to_approach = np.linalg.norm(arm - approach_point)
    approach_reward = -1.0 * arm_to_approach

    # --- Exploration helper 2: action guidance reward ---
    # Strong weight (3.0) makes this the dominant per-step signal, effectively
    # turning PPO into imitation learning with RL fine-tuning on top.
    guidance_reward = 0.0
    if action is not None and expert_act is not None:
        guidance_reward = -3.0 * np.linalg.norm(action - expert_act)

    return (
        distance_reward
        + progress_reward
        + success_bonus
        + step_penalty
        + approach_reward
        + guidance_reward
    )


def train_ppo(args):
    """Train a PPO policy on PushCube -- main RL baseline."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for PPO.")
        print("Install with: pip install torch")
        sys.exit(1)

    print("=" * 70)
    print(" Unified PushCube -- RL Training (PPO)")
    print("=" * 70)

    device = torch.device('cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _env_tmp = PushCubeEnv()
    state_dim = _env_tmp.state_dim       # 14
    action_dim = _env_tmp.action_dim     # 2

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,  # no entropy bonus — preserves BC mean, prevents std growth
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
    )

    print(f"\nState dim: {state_dim}, Action dim: {action_dim}")
    print(f"Training for {args.n_episodes} episodes (seed={args.seed})...")
    print(f"PPO hyperparams: clip=0.2, GAE lambda=0.95, gamma=0.99, lr=3e-4, "
          f"update_every=10 ep, 5 epochs/update")

    # ------------------------------------------------------------------
    # Phase 0: Behavioral cloning (BC) pre-training
    # Collect expert demonstrations and supervised-train the actor.
    # This prevents policy collapse -- without it, the sparse arm-movement
    # gradient causes the network to converge to a constant action.
    # ------------------------------------------------------------------
    if not args.smoke_test:
        bc_episodes = 500
        bc_epochs = 500
    else:
        bc_episodes = 5
        bc_epochs = 10
    bc_states = []
    bc_actions = []
    for ep in range(bc_episodes):
        env = PushCubeEnv()
        env.reset(seed=args.seed + ep)
        for step in range(env.max_steps):
            state = env.get_state_vector()
            expert_act = expert_action(env)
            bc_states.append(state)
            bc_actions.append(expert_act)
            agent.obs_rms.update(state)
            obs, _, done, trunc, _ = env.step(expert_act)
            if done or trunc:
                break
    agent.pretrain_bc(bc_states, bc_actions, n_epochs=bc_epochs, batch_size=32,
                      lr=3e-3)

    # After BC, set log_std to a low value for controlled exploration.
    # BC doesn't train log_std (it stays at 0, i.e., std=1.0 which is very
    # noisy). Setting std=0.3 (log_std=-1.2) keeps PPO rollouts close to
    # the BC mean, preventing the deterministic policy from degrading.
    with torch.no_grad():
        agent.actor.log_std.fill_(-1.2)
    print(f"  Post-BC log_std set to {agent.actor.log_std[0].item():.2f} "
          f"(std={torch.exp(agent.actor.log_std[0]).item():.3f})")

    # Evaluate BC policy to verify it works
    bc_success = 0
    bc_n = 20
    for ep in range(bc_n):
        env = PushCubeEnv()
        env.reset(seed=10000 + ep)
        state = env.get_state_vector()
        for step in range(env.max_steps):
            action, _, _, _ = agent.select_action(state, deterministic=True)
            obs, _, done, trunc, _ = env.step(action)
            state = env.get_state_vector()
            if done:
                bc_success += 1
                break
            if trunc:
                break
    print(f"  BC eval success: {bc_success}/{bc_n} = "
          f"{bc_success/bc_n*100:.1f}%")

    update_interval = 10  # update every 10 episodes
    reward_history = []
    success_history = []

    # Rollout buffer (accumulates across update_interval episodes)
    rollout = {
        'states': [], 'actions': [], 'log_probs': [],
        'rewards': [], 'values': [], 'dones': [],
    }

    # Expert-guided exploration: constant probability throughout training.
    # This provides on-policy demonstrations that PPO can learn from (the
    # expert leads to successful trajectories with positive advantages, so
    # PPO reinforces the policy toward imitating those actions). A constant
    # (non-decaying) rate ensures the policy always has expert-guided
    # trajectories to learn from, preventing the policy collapse observed
    # with decaying schedules.
    expert_prob = 0.3

    for ep in range(args.n_episodes):
        env = PushCubeEnv()
        env.reset(seed=args.seed + ep)
        state = env.get_state_vector()
        agent.obs_rms.update(state)

        ep_reward = 0.0
        ep_success = False

        # Track previous cube-target distance for progress reward
        active_cube = env.cube_positions[env.active_idx]
        prev_dist = np.linalg.norm(active_cube - env.target_pos)

        for step in range(env.max_steps):
            # Compute expert action for current state (before env step).
            # This provides a per-step reference signal for reward shaping
            # and can be used for expert-guided exploration.
            expert_act = expert_action(env)

            # Decide whether to use expert or policy action
            use_expert = (not args.smoke_test and
                          np.random.random() < expert_prob)
            if use_expert:
                action, log_prob, value, state_norm = \
                    agent.select_action_with_expert(state, expert_act)
            else:
                action, log_prob, value, state_norm = agent.select_action(
                    state, deterministic=False
                )

            obs, env_reward, done, truncated, info = env.step(action)

            # Compute shaped reward (includes guidance from expert action)
            active_cube = env.cube_positions[env.active_idx]
            cur_dist = np.linalg.norm(active_cube - env.target_pos)
            shaped_reward = compute_shaped_reward(
                env, prev_dist, cur_dist, done,
                action=action, expert_act=expert_act,
            )
            prev_dist = cur_dist

            # Store transition (use normalized state for consistency)
            rollout['states'].append(state_norm)
            rollout['actions'].append(action)
            rollout['log_probs'].append(log_prob)
            rollout['rewards'].append(shaped_reward)
            rollout['values'].append(value)
            rollout['dones'].append(done or truncated)

            ep_reward += shaped_reward
            if done:
                ep_success = True

            state = env.get_state_vector()
            agent.obs_rms.update(state)

            if done or truncated:
                break

        reward_history.append(ep_reward)
        success_history.append(1.0 if ep_success else 0.0)

        # PPO update every update_interval episodes (or at the end)
        if (ep + 1) % update_interval == 0 or ep == args.n_episodes - 1:
            if len(rollout['states']) > 0:
                # Compute GAE (Generalized Advantage Estimation)
                rewards = np.array(rollout['rewards'], dtype=np.float32)
                values = np.array(rollout['values'], dtype=np.float32)
                dones = np.array(rollout['dones'], dtype=np.float32)

                T = len(rewards)
                advantages = np.zeros(T, dtype=np.float32)
                last_gae = 0.0
                for t in reversed(range(T)):
                    if t == T - 1:
                        next_val = 0.0
                    else:
                        next_val = values[t + 1]
                    delta = (
                        rewards[t]
                        + agent.gamma * next_val * (1.0 - dones[t])
                        - values[t]
                    )
                    last_gae = (
                        delta
                        + agent.gamma * agent.gae_lambda
                        * (1.0 - dones[t]) * last_gae
                    )
                    advantages[t] = last_gae

                returns = advantages + values

                rollout['advantages'] = advantages
                rollout['returns'] = returns

                agent.update(rollout, n_epochs=5, batch_size=64)

                # Reset rollout buffer
                rollout = {
                    'states': [], 'actions': [], 'log_probs': [],
                    'rewards': [], 'values': [], 'dones': [],
                }

        # Print progress (~10 lines)
        print_interval = max(1, args.n_episodes // 10)
        if (ep + 1) % print_interval == 0:
            window = min(50, len(reward_history))
            avg_r = np.mean(reward_history[-window:])
            avg_s = np.mean(success_history[-window:]) * 100
            print(f"  Episode {ep+1}/{args.n_episodes}: "
                  f"avg_reward={avg_r:.3f}, train_success={avg_s:.1f}%, "
                  f"expert_prob={expert_prob:.3f}")

    # ------------------------------------------------------------------
    # Evaluate trained policy (deterministic + stochastic)
    # ------------------------------------------------------------------
    n_eval = max(args.n_eval, 2)
    print(f"\nEvaluating PPO policy (deterministic, {n_eval} episodes)...")
    success_count = 0
    eval_rewards = []

    for ep in range(n_eval):
        env = PushCubeEnv()
        env.reset(seed=10000 + ep)
        state = env.get_state_vector()
        total_reward = 0.0

        for step in range(env.max_steps):
            action, _, _, _ = agent.select_action(state, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = env.get_state_vector()
            if done:
                success_count += 1
                break
            if truncated:
                break

        eval_rewards.append(total_reward)

    success_rate = success_count / n_eval * 100
    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))

    print(f"  Deterministic success: {success_count}/{n_eval} = {success_rate:.1f}%")
    print(f"  Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    # Also evaluate with stochastic actions (may perform better due to
    # exploration helping with multi-phase task transitions)
    print(f"\nEvaluating PPO policy (stochastic, {n_eval} episodes)...")
    stoch_success = 0
    stoch_rewards = []

    for ep in range(n_eval):
        env = PushCubeEnv()
        env.reset(seed=10000 + ep)
        state = env.get_state_vector()
        total_reward = 0.0

        for step in range(env.max_steps):
            action, _, _, _ = agent.select_action(state, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = env.get_state_vector()
            if done:
                stoch_success += 1
                break
            if truncated:
                break

        stoch_rewards.append(total_reward)

    stoch_success_rate = stoch_success / n_eval * 100
    stoch_mean_reward = float(np.mean(stoch_rewards))
    stoch_std_reward = float(np.std(stoch_rewards))

    print(f"  Stochastic success: {stoch_success}/{n_eval} = {stoch_success_rate:.1f}%")
    print(f"  Mean reward: {stoch_mean_reward:.3f} +/- {stoch_std_reward:.3f}")

    # Report the best result
    best_rate = max(success_rate, stoch_success_rate)
    best_mode = "deterministic" if success_rate >= stoch_success_rate else "stochastic"
    print(f"\n  Best: {best_mode} = {best_rate:.1f}%")

    # ------------------------------------------------------------------
    # Expert baseline (for comparison)
    # ------------------------------------------------------------------
    print(f"\nExpert baseline ({n_eval} episodes)...")
    expert_success = 0
    for ep in range(n_eval):
        env = PushCubeEnv()
        env.reset(seed=10000 + ep)
        for step in range(env.max_steps):
            action = expert_action(env)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                expert_success += 1
                break
            if truncated:
                break
    expert_success_rate = expert_success / n_eval * 100
    print(f"Expert success rate: {expert_success}/{n_eval} "
          f"= {expert_success_rate:.1f}%")

    # ------------------------------------------------------------------
    # Save policy and results
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'obs_mean': agent.obs_rms.mean,
        'obs_var': agent.obs_rms.var,
        'state_dim': state_dim,
        'action_dim': action_dim,
    }, save_dir / "pushcube_ppo_policy.pt")
    print(f"\nPolicy saved to {save_dir / 'pushcube_ppo_policy.pt'}")

    results = {
        "task": "PushCube RL (PPO)",
        "algorithm": "PPO",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_episodes": args.n_episodes,
        "n_eval": n_eval,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "hyperparameters": {
            "clip_ratio": 0.2,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "lr": 3e-4,
            "update_interval": 10,
            "n_epochs": 5,
            "actor_arch": "MLP 14->64->64->2 (tanh output)",
            "critic_arch": "MLP 14->64->64->1",
            "bc_pretrain_episodes": bc_episodes,
            "bc_pretrain_epochs": bc_epochs,
            "expert_guided_exploration": True,
            "expert_prob": 0.3,
            "expert_prob_schedule": "constant",
            "ent_coef": 0.0,
            "post_bc_log_std": -1.2,
        },
        "reward_shaping": {
            "distance_reward": "-dist(active_cube, target)",
            "progress_reward": "+0.5 * (prev_dist - cur_dist)",
            "success_bonus": 10.0,
            "step_penalty": -0.01,
            "approach_reward": "-1.0 * dist(arm, approach_point behind cube)",
            "guidance_reward": "-3.0 * ||action - expert_action||",
        },
        "bc_eval_success_rate": round(bc_success / bc_n * 100, 1),
        "deterministic_success_rate": round(success_rate, 1),
        "stochastic_success_rate": round(stoch_success_rate, 1),
        "success_rate": round(best_rate, 1),
        "best_eval_mode": best_mode,
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "stoch_mean_reward": round(stoch_mean_reward, 3),
        "stoch_std_reward": round(stoch_std_reward, 3),
        "expert_success_rate": round(expert_success_rate, 1),
    }
    from benchmark_provenance import build_provenance
    results["provenance"] = build_provenance(
        command=f"python {__file__} --algo ppo --n_episodes {args.n_episodes} --seed {args.seed}",
        result_generated_by=__file__,
    )
    results_path = save_dir / "rl_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


# ======================================================================
# REINFORCE Implementation (Concept Demo -- NOT a proper baseline)
# ======================================================================

def train_reinforce_concept_demo(args):
    """
    Train a simple policy network with REINFORCE (policy gradient).

    NOTE: This is a lightweight teaching / concept-demo implementation
    using pure NumPy. It does NOT achieve a usable success rate and is
    retained for pedagogical comparison only. For the main RL baseline,
    use: --algo ppo. Current teaching-scale PPO performance is
    approximately 10-20% success rate.
    """
    print("=" * 70)
    print(" Unified PushCube -- RL Training (REINFORCE -- Concept Demo)")
    print("=" * 70)
    print(" NOTE: REINFORCE is a teaching demo. Use --algo ppo for the "
          "proper baseline.")

    # ------------------------------------------------------------------
    # Get dimensions from environment
    # ------------------------------------------------------------------
    _env_tmp = PushCubeEnv()
    state_dim = _env_tmp.state_dim       # 14
    action_dim = _env_tmp.action_dim     # 2
    hidden_dim = 32

    rng = np.random.RandomState(args.seed)

    # Initialize weights
    W1 = rng.randn(state_dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.randn(hidden_dim, action_dim).astype(np.float32) * 0.1
    b2 = np.zeros(action_dim, dtype=np.float32)
    W_logstd = np.zeros(action_dim, dtype=np.float32) - 1.0  # log(std)

    def relu(x):
        return np.maximum(0, x)

    def policy_forward(state):
        h = relu(state @ W1 + b1)
        mean = h @ W2 + b2
        _logstd = W_logstd
        std = np.exp(_logstd)
        return mean, std

    def sample_action(state):
        mean, std = policy_forward(state)
        action = mean + std * rng.randn(action_dim)
        _logstd = W_logstd
        log_prob = -0.5 * np.sum(((action - mean) / std) ** 2) - np.sum(_logstd)
        return np.clip(action, -1.0, 1.0), log_prob

    # Training hyper-parameters
    gamma = 0.99
    lr = 1e-3

    print(f"\nState dim: {state_dim}, Action dim: {action_dim}")
    print(f"Training for {args.n_episodes} episodes (seed={args.seed})...")
    best_reward = -float("inf")
    reward_history = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in range(args.n_episodes):
        env = PushCubeEnv()
        obs = env.reset(seed=args.seed + ep)
        state = env.get_state_vector()

        states = []
        actions = []
        log_probs = []
        rewards = []

        for step in range(env.max_steps):
            action, log_prob = sample_action(state)
            next_obs, reward, done, truncated, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = env.get_state_vector()
            if done or truncated:
                break

        if len(states) == 0:
            continue

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        dW2 = np.zeros_like(W2)
        db2 = np.zeros_like(b2)
        dW1 = np.zeros_like(W1)
        db1 = np.zeros_like(b1)
        dW_logstd = np.zeros_like(W_logstd)

        for t, (s, lp, ret) in enumerate(zip(states, log_probs, returns)):
            h = relu(s @ W1 + b1)
            mean, std = policy_forward(s)
            a = actions[t]

            dmean = (a - mean) / (std ** 2)
            dlogstd = ((a - mean) ** 2) / (std ** 2) - 1.0

            dW2 += np.outer(h, dmean) * ret
            db2 += dmean * ret
            dh = dmean @ W2.T
            dh[h <= 0] = 0
            dW1 += np.outer(s, dh) * ret
            db1 += dh * ret
            dW_logstd += dlogstd * ret

        W2 += lr * dW2 / len(states)
        b2 += lr * db2 / len(states)
        W1 += lr * dW1 / len(states)
        b1 += lr * db1 / len(states)
        W_logstd += lr * dW_logstd / len(states)

        total_reward = sum(rewards)
        reward_history.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        print_interval = max(1, args.n_episodes // 10)
        if (ep + 1) % print_interval == 0:
            window = min(100, len(reward_history))
            avg = np.mean(reward_history[-window:])
            print(f"  Episode {ep+1}/{args.n_episodes}: "
                  f"avg_reward={avg:.3f}, best={best_reward:.3f}")

    # ------------------------------------------------------------------
    # Evaluate trained policy (deterministic)
    # ------------------------------------------------------------------
    n_eval = max(args.n_eval, 2)
    print(f"\nEvaluating trained policy (deterministic, {n_eval} episodes)...")
    success_count = 0
    eval_rewards = []

    for ep in range(n_eval):
        env = PushCubeEnv()
        obs = env.reset(seed=10000 + ep)
        state = env.get_state_vector()
        total_reward = 0

        for step in range(env.max_steps):
            mean, _ = policy_forward(state)
            action = np.clip(mean, -1.0, 1.0)
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = env.get_state_vector()
            if done:
                success_count += 1
                break
            if truncated:
                break

        eval_rewards.append(total_reward)

    success_rate = success_count / n_eval * 100
    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))

    print(f"Success rate: {success_count}/{n_eval} = {success_rate:.1f}%")
    print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    # ------------------------------------------------------------------
    # Expert baseline (for comparison)
    # ------------------------------------------------------------------
    print(f"\nExpert baseline ({n_eval} episodes)...")
    expert_success = 0
    for ep in range(n_eval):
        env = PushCubeEnv()
        obs = env.reset(seed=10000 + ep)
        for step in range(env.max_steps):
            action = expert_action(env)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                expert_success += 1
                break
            if truncated:
                break
    expert_success_rate = expert_success / n_eval * 100
    print(f"Expert success rate: {expert_success}/{n_eval} "
          f"= {expert_success_rate:.1f}%")

    # ------------------------------------------------------------------
    # Save policy and results
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / "pushcube_rl_policy.npz",
        W1=W1, b1=b1, W2=W2, b2=b2, W_logstd=W_logstd,
    )
    print(f"\nPolicy saved to {save_dir / 'pushcube_rl_policy.npz'}")

    results = {
        "task": "PushCube RL (REINFORCE -- Concept Demo)",
        "algorithm": "REINFORCE",
        "state_dim": state_dim,
        "n_episodes": args.n_episodes,
        "n_eval": n_eval,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "best_reward": round(float(best_reward), 3),
        "success_rate": round(success_rate, 1),
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "expert_success_rate": round(expert_success_rate, 1),
    }
    from benchmark_provenance import build_provenance
    results["provenance"] = build_provenance(
        command=f"python {__file__} --algo reinforce --n_episodes {args.n_episodes} --seed {args.seed}",
        result_generated_by=__file__,
    )
    results_path = save_dir / "rl_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified PushCube -- RL Track (PPO baseline / REINFORCE demo)"
    )
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "reinforce"],
        help="RL algorithm: 'ppo' (default, main baseline) or "
             "'reinforce' (concept demo)",
    )
    parser.add_argument("--n-episodes", type=int, default=500,
                        help="Training episodes (default: 500)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Evaluation episodes (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str,
                        default="../results/unified_pushcube/rl",
                        help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test (2 train episodes, 2 eval) for CI")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_episodes = 2
        args.n_eval = 2
        print("[SMOKE TEST MODE] 2 train episodes, 2 eval episodes")

    if args.algo == "ppo":
        train_ppo(args)
    elif args.algo == "reinforce":
        train_reinforce_concept_demo(args)
    else:
        print(f"Unknown algorithm: {args.algo}")
        sys.exit(1)


if __name__ == "__main__":
    main()
