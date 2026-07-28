#!/usr/bin/env python3
"""
dreamer_rssm.py
================
Dreamer V3 核心架构 RSSM（Recurrent State-Space Model）的简化实现。

RSSM 是世界模型最重要的架构之一，核心思想是 **分离确定性和随机性**：
  - 确定性部分（GRU）：记忆历史，捕捉可预测规律
  - 随机性部分（Gaussian latent）：捕捉不可预测的不确定性

本文档用合成数据演示 RSSM 的训练和推理，帮助理解 Dreamer V3 的核心。

对应理论文档：docs/07-world-models-for-vla.md 第 5.1 节
对应论文：Mastering Diverse Domains through World Models (Hafner et al., 2023)

依赖：pip install torch matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# 1. 合成数据：带噪声的 2D 导航轨迹（状态相关 termination）
# ============================================================

class NoisyTrajectoryDataset(Dataset):
    """
    合成数据集：模拟在 2D 平面上导航至随机目标的 Agent。

    确定性部分：恒定速度 + 动作控制
    随机性部分：高斯噪声（模拟摩擦、碰撞等不确定因素）

    Termination 来源（状态相关，非仅序列末尾）：
      1. 成功到达目标（goal_threshold）
      2. 碰撞边界（|x| 或 |y| 超过 boundary_limit）
      3. 序列长度上限（truncation）
    """

    def __init__(self, num_samples=3000, seq_len=20, dt=0.1, noise_std=0.05,
                 goal_threshold=0.3, boundary_limit=4.0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt

        self.observations = []  # [N, T, obs_dim]
        self.actions = []       # [N, T, act_dim]
        self.rewards = []       # [N, T]
        self.continues = []     # [N, T]

        for _ in range(num_samples):
            obs_seq = []
            act_seq = []
            rew_seq = []
            cont_seq = []

            # 随机初始状态
            x, y = np.random.randn(2) * 2.0
            vx, vy = np.random.randn(2) * 0.3

            # 随机目标位置（每条序列不同）
            goal_x = np.random.uniform(-1.5, 1.5)
            goal_y = np.random.uniform(-1.5, 1.5)

            for t in range(seq_len):
                # 观测 = [位置, 速度, 到目标的相对偏移]  (obs_dim=6)
                goal_dx = goal_x - x
                goal_dy = goal_y - y
                obs = np.array([x, y, vx, vy, goal_dx, goal_dy], dtype=np.float32)
                obs_seq.append(obs)

                # 动作 = 朝目标方向的速度调整 + 探索噪声
                noise_action = np.random.randn(2).astype(np.float32) * 0.2
                action = noise_action
                act_seq.append(action)

                # 计算 reward: 接近目标为正
                dist_to_goal = np.sqrt(goal_dx**2 + goal_dy**2)
                reward = -dist_to_goal / 5.0  # 归一化的距离惩罚

                # --- 状态相关 termination 检测 ---
                terminated = False   # 环境自然终止（成功到达/碰撞）
                truncated = False     # 人为截断（序列长度上限）

                # 1) 成功到达目标
                if dist_to_goal < goal_threshold:
                    terminated = True
                    reward += 2.0  # 到达目标的奖励

                # 2) 碰撞边界
                if abs(x) > boundary_limit or abs(y) > boundary_limit:
                    terminated = True
                    reward -= 1.0  # 碰撞惩罚

                # 3) 序列末尾截断
                if t == seq_len - 1:
                    truncated = True

                done = 1.0 if (terminated or truncated) else 0.0
                cont_seq.append(1.0 - done)  # continue = 1.0 - float(terminated or truncated)
                rew_seq.append(reward)

                if done > 0.5:
                    # episode 结束后用零填充剩余步（保持固定 seq_len）
                    remaining = seq_len - t - 1
                    for _ in range(remaining):
                        obs_seq.append(np.zeros(6, dtype=np.float32))
                        act_seq.append(np.zeros(2, dtype=np.float32))
                        rew_seq.append(0.0)
                        cont_seq.append(0.0)
                    break

                # 确定性转移
                vx = vx + action[0] * dt
                vy = vy + action[1] * dt
                x = x + vx * dt
                y = y + vy * dt

                # 随机性（模拟不确定的环境因素）
                x += np.random.randn() * noise_std
                y += np.random.randn() * noise_std

            self.observations.append(np.array(obs_seq, dtype=np.float32))
            self.actions.append(np.array(act_seq, dtype=np.float32))
            self.rewards.append(np.array(rew_seq, dtype=np.float32))
            self.continues.append(np.array(cont_seq, dtype=np.float32))

        self.observations = np.array(self.observations)  # [N, T, 6]
        self.actions = np.array(self.actions)             # [N, T, 2]
        self.rewards = np.array(self.rewards)             # [N, T]
        self.continues = np.array(self.continues)         # [N, T]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.observations[idx]),  # [T, 6]
            torch.FloatTensor(self.actions[idx]),        # [T, 2]
            torch.FloatTensor(self.rewards[idx]),         # [T]
            torch.FloatTensor(self.continues[idx]),       # [T]
        )


# ============================================================
# 2. RSSM 架构
# ============================================================

class RSSM(nn.Module):
    """
    Recurrent State-Space Model（简化版）。

    核心变量：
      h_t: 确定性隐状态（GRU 输出），记忆历史信息
      z_t: 随机隐状态（Gaussian），捕捉当前不确定性
      s_t = (h_t, z_t): 完整的 RSSM 状态

    两个核心函数：
      prior:    p(z_t | h_t)           — 不依赖观测，只看历史
      posterior: q(z_t | h_t, o_t)     — 依赖观测，更准确
    """

    def __init__(self, obs_dim=4, act_dim=2, stoch_dim=16, deter_dim=64):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # --- 确定性部分：GRU ---
        self.gru = nn.GRUCell(deter_dim, deter_dim)
        # 输入到 GRU 的投影
        self.act_proj = nn.Linear(act_dim, deter_dim)
        self.z_proj = nn.Linear(stoch_dim, deter_dim)

        # --- 随机性部分：Gaussian ---
        # Prior: h_t → (mu, logstd)
        self.prior_net = nn.Linear(deter_dim, stoch_dim * 2)
        # Posterior: h_t + o_t → (mu, logstd)
        self.posterior_net = nn.Linear(deter_dim + obs_dim, stoch_dim * 2)

        # --- 观测重建 ---
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

        # --- Reward predictor ---
        self.reward_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # --- Continue predictor (predicts probability of episode continuing) ---
        self.continue_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_stoch_state(self, mean, logstd):
        """从高斯分布采样。训练时加噪声，推理时用均值。"""
        std = torch.exp(logstd.clamp(-5, 2))  # 限制范围防止数值问题
        if self.training:
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        return z

    def prior(self, h):
        """Prior: p(z_t | h_t)，不依赖观测。"""
        out = self.prior_net(h)
        mu = out[:, :self.stoch_dim]
        logstd = out[:, self.stoch_dim:]
        z = self.get_stoch_state(mu, logstd)
        return z, mu, logstd

    def posterior(self, h, obs):
        """Posterior: q(z_t | h_t, o_t)，依赖观测（更准确）。"""
        x = torch.cat([h, obs], dim=-1)
        out = self.posterior_net(x)
        mu = out[:, :self.stoch_dim]
        logstd = out[:, self.stoch_dim:]
        z = self.get_stoch_state(mu, logstd)
        return z, mu, logstd

    def imagine_step(self, h, z, action):
        """
        想象一步（推理/规划时用 prior，不依赖真实观测）。
        对应 Dreamer V3 中的 "imagination rollout"。
        """
        gru_input = self.act_proj(action) + self.z_proj(z)
        h_next = self.gru(gru_input, h)
        z_next, _, _ = self.prior(h_next)
        return h_next, z_next

    def reconstruct(self, h, z):
        """从 RSSM 状态重建观测。"""
        x = torch.cat([h, z], dim=-1)
        return self.obs_decoder(x)

    def predict_reward(self, h, z):
        """从 RSSM 状态预测 reward。"""
        return self.reward_head(torch.cat([h, z], dim=-1)).squeeze(-1)

    def predict_continue(self, h, z):
        """从 RSSM 状态预测 continue 概率（logit）。"""
        return self.continue_head(torch.cat([h, z], dim=-1)).squeeze(-1)


# ============================================================
# 3. 训练
# ============================================================

def train_rssm(model, train_loader, val_loader=None, epochs=25, lr=3e-4, device="cpu", kl_balance=0.5,
               reward_balance=1.0, continue_balance=0.1, early_stop_patience=None):
    """
    RSSM 训练循环。

    损失 = 观测重建 + KL(posterior || prior) + reward预测 + continue预测

    Args:
        model: RSSM 模型
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader (可选, 用于 early stopping)
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备
        kl_balance: KL 损失权重
        reward_balance: reward 损失权重
        continue_balance: continue 损失权重
        early_stop_patience: 早停耐心值 (None 表示不早停)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"recon_loss": [], "kl_loss": [], "reward_loss": [], "continue_loss": [], "total_loss": [],
               "val_recon_loss": [], "val_reward_mae": []}

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_recon = 0.0
        total_kl = 0.0
        total_reward = 0.0
        total_continue = 0.0
        n_batches = 0

        for obs_seq, act_seq, rew_seq, cont_seq in train_loader:
            # obs_seq: [B, T, obs_dim], act_seq: [B, T, act_dim]
            # rew_seq: [B, T], cont_seq: [B, T]
            B, T, obs_dim = obs_seq.shape
            obs_seq = obs_seq.to(device)
            act_seq = act_seq.to(device)
            rew_seq = rew_seq.to(device)
            cont_seq = cont_seq.to(device)

            recon_loss = 0.0
            kl_loss = 0.0
            reward_loss = 0.0
            continue_loss = 0.0

            # 初始化 RSSM 状态
            h = torch.zeros(B, model.deter_dim, device=device)
            z = torch.zeros(B, model.stoch_dim, device=device)

            for t in range(T):
                obs_t = obs_seq[:, t, :]
                act_t = act_seq[:, t, :]
                rew_t = rew_seq[:, t]
                cont_t = cont_seq[:, t]

                # 1. Prior
                z_prior, mu_prior, logstd_prior = model.prior(h)

                # 2. Posterior
                z_post, mu_post, logstd_post = model.posterior(h, obs_t)

                # 3. GRU 更新确定性状态
                gru_input = model.act_proj(act_t) + model.z_proj(z_post)
                h = model.gru(gru_input, h)

                # 4. 观测重建
                obs_recon = model.reconstruct(h, z_post)

                # 5. Reward prediction
                reward_pred = model.predict_reward(h, z_post)

                # 6. Continue prediction
                continue_pred = model.predict_continue(h, z_post)

                # 7. 损失
                recon_loss = recon_loss + F.mse_loss(obs_recon, obs_t)
                reward_loss = reward_loss + F.mse_loss(reward_pred, rew_t)
                continue_loss = continue_loss + F.binary_cross_entropy_with_logits(continue_pred, cont_t)

                # KL 散度: KL(q(z|h,o) || p(z|h))
                # = log(p/q) = logstd_post - logstd_prior + (var_post + (mu_post - mu_prior)^2) / (2*var_prior) - 0.5
                var_post = torch.exp(2 * logstd_post)
                var_prior = torch.exp(2 * logstd_prior)
                kl = 0.5 * (
                    (var_post + (mu_post - mu_prior) ** 2) / (var_prior + 1e-8)
                    - 1
                    + 2 * (logstd_prior - logstd_post)
                )
                kl_loss = kl_loss + kl.mean()

                z = z_post  # 后续步骤用 posterior

            # 平均到序列长度
            recon_loss = recon_loss / T
            kl_loss = kl_loss / T
            reward_loss = reward_loss / T
            continue_loss = continue_loss / T

            # 自由比特（free nats）：KL 低于阈值时停止更新，防止 posterior 坍缩到 prior
            free_nats = 1.0
            kl_loss = torch.clamp(kl_loss, min=free_nats)

            loss = (recon_loss + kl_balance * kl_loss
                    + reward_balance * reward_loss + continue_balance * continue_loss)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（Dreamer 系列的标准做法）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_reward += reward_loss.item()
            total_continue += continue_loss.item()
            n_batches += 1

        avg_recon = total_recon / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)
        avg_reward = total_reward / max(n_batches, 1)
        avg_continue = total_continue / max(n_batches, 1)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["reward_loss"].append(avg_reward)
        history["continue_loss"].append(avg_continue)
        history["total_loss"].append(avg_recon + kl_balance * avg_kl
                                      + reward_balance * avg_reward
                                      + continue_balance * avg_continue)

        # --- Validation (held-out data) ---
        val_recon, val_rew_mae = None, None
        if val_loader is not None:
            model.eval()
            v_recon_total, v_rew_total, v_count = 0.0, 0.0, 0
            with torch.no_grad():
                for v_obs, v_act, v_rew, v_cont in val_loader:
                    v_obs = v_obs.to(device)
                    v_rew = v_rew.to(device)
                    vB, vT, _ = v_obs.shape
                    h = torch.zeros(vB, model.deter_dim, device=device)
                    z = torch.zeros(vB, model.stoch_dim, device=device)
                    for t in range(vT):
                        z_post, _, _ = model.posterior(h, v_obs[:, t, :])
                        gru_input = model.act_proj(v_act[:, t, :].to(device)) + model.z_proj(z_post)
                        h = model.gru(gru_input, h)
                        recon = model.reconstruct(h, z_post)
                        rew_pred = model.predict_reward(h, z_post)
                        v_recon_total += F.mse_loss(recon, v_obs[:, t, :]).item()
                        v_rew_total += F.l1_loss(rew_pred, v_rew[:, t]).item()
                        v_count += 1
                        z = z_post
            val_recon = v_recon_total / max(v_count, 1)
            val_rew_mae = v_rew_total / max(v_count, 1)
            history["val_recon_loss"].append(val_recon)
            history["val_reward_mae"].append(val_rew_mae)

        if (epoch + 1) % 5 == 0:
            total_val = avg_recon + kl_balance * avg_kl + reward_balance * avg_reward + continue_balance * avg_continue
            line = f"Epoch {epoch+1:3d}/{epochs} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | " \
                   f"Reward: {avg_reward:.4f} | Continue: {avg_continue:.4f} | Total: {total_val:.4f}"
            if val_recon is not None:
                line += f" | Val Recon: {val_recon:.4f} | Val RewMAE: {val_rew_mae:.4f}"
            print(line)

        # Early stopping
        if val_loader is not None and early_stop_patience is not None:
            val_loss = val_recon + val_rew_mae
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"  Early stopping at epoch {epoch+1} (val_loss={val_loss:.4f})")
                    break

    return history


# ============================================================
# 4. 想象展开（Imagination Rollout）
# ============================================================

def imagine_rollout(model, obs_seq, act_seq, rew_seq, cont_seq, device="cpu",
                    burn_in_steps=5, eval_horizons=(1, 5, 10, 20)):
    """
    演示 RSSM 的核心能力：想象展开（不依赖真实观测，用 prior 预测未来）。

    对比 posterior 轨迹（用真实观测）vs prior 轨迹（纯想象）的差异，
    同时对比 reward/continue 预测与真实值。

    修改 2 新增：
      - burn-in: 先用 burn_in_steps 步真实观测建立 latent state（posterior warmup），
        然后从 warmed-up 状态开始 prior rollout。
      - 评估 horizon 1/5/10/20 的累积观测误差，回答：
        "给定当前真实观测，世界模型预测未来 H 步有多准确"。

    Args:
        burn_in_steps: posterior warmup 步数（默认 5）
        eval_horizons: 评估的 horizon 列表（默认 1, 5, 10, 20）
    Returns:
        dict 包含:
          - post_err, pri_err (per-step L2 误差)
          - horizon_errors: {horizon: mean_cumulative_error}
          - reward/continue 预测 vs 真实
          - post_recons, pri_recons (逐步重建)
    """
    model.eval()
    with torch.no_grad():
        obs_seq = obs_seq.unsqueeze(0).to(device)  # [1, T, obs_dim]
        act_seq = act_seq.unsqueeze(0).to(device)  # [1, T, act_dim]
        rew_seq = rew_seq.unsqueeze(0).to(device)  # [1, T]
        cont_seq = cont_seq.unsqueeze(0).to(device)  # [1, T]
        B, T, _ = obs_seq.shape

        # --- Posterior 轨迹（依赖真实观测，更准确） ---
        h_post = torch.zeros(B, model.deter_dim, device=device)
        z_post = torch.zeros(B, model.stoch_dim, device=device)
        post_recons = []
        post_rewards = []
        post_continues = []

        for t in range(T):
            z_post, _, _ = model.posterior(h_post, obs_seq[:, t, :])
            gru_input = model.act_proj(act_seq[:, t, :]) + model.z_proj(z_post)
            h_post = model.gru(gru_input, h_post)
            recon = model.reconstruct(h_post, z_post)
            reward_pred = model.predict_reward(h_post, z_post)
            continue_pred = model.predict_continue(h_post, z_post)
            post_recons.append(recon.cpu().numpy())
            post_rewards.append(reward_pred.cpu().numpy())
            post_continues.append(torch.sigmoid(continue_pred).cpu().numpy())

        # --- Prior 轨迹（带 burn-in posterior warmup） ---
        # Phase 1: Burn-in — 用真实观测建立 latent state
        h_pri = torch.zeros(B, model.deter_dim, device=device)
        z_pri = torch.zeros(B, model.stoch_dim, device=device)
        burn_in_steps = min(burn_in_steps, T)
        for t in range(burn_in_steps):
            z_pri, _, _ = model.posterior(h_pri, obs_seq[:, t, :])
            gru_input = model.act_proj(act_seq[:, t, :]) + model.z_proj(z_pri)
            h_pri = model.gru(gru_input, h_pri)

        # Phase 2: Prior rollout — 从 warmed-up 状态开始纯想象
        pri_recons = []
        pri_rewards = []
        pri_continues = []

        # 先用全零占位补齐 burn-in 步的重建结果（burn-in 期间不做 prior 预测）
        for t in range(burn_in_steps):
            pri_recons.append(np.zeros((B, obs_seq.shape[-1]), dtype=np.float32))
            pri_rewards.append(np.zeros((B,), dtype=np.float32))
            pri_continues.append(np.zeros((B,), dtype=np.float32))

        # 从 burn_in_steps 开始用 prior 想象
        for t in range(burn_in_steps, T):
            h_pri, z_pri = model.imagine_step(h_pri, z_pri, act_seq[:, t, :])
            recon = model.reconstruct(h_pri, z_pri)
            reward_pred = model.predict_reward(h_pri, z_pri)
            continue_pred = model.predict_continue(h_pri, z_pri)
            pri_recons.append(recon.cpu().numpy())
            pri_rewards.append(reward_pred.cpu().numpy())
            pri_continues.append(torch.sigmoid(continue_pred).cpu().numpy())

        # 计算逐步误差
        post_recons = np.array(post_recons).squeeze(1)  # [T, obs_dim]
        pri_recons = np.array(pri_recons).squeeze(1)
        ground_truth = obs_seq.squeeze(0).cpu().numpy()  # [T, obs_dim]

        post_err = np.linalg.norm(post_recons - ground_truth, axis=-1)
        pri_err = np.linalg.norm(pri_recons - ground_truth, axis=-1)

        # --- Horizon 累积误差评估（仅基于 prior rollout 部分） ---
        rollout_len = T - burn_in_steps
        horizon_errors = {}
        for horizon in eval_horizons:
            if horizon <= rollout_len:
                # 计算从 burn-in 结束后连续 horizon 步的累积 L2 误差
                err_slice = pri_err[burn_in_steps:burn_in_steps + horizon]
                horizon_errors[horizon] = float(np.mean(err_slice))
            else:
                horizon_errors[horizon] = float("nan")

        # Reward 和 continue 预测结果
        gt_rewards = rew_seq.squeeze(0).cpu().numpy()          # [T]
        gt_continues = cont_seq.squeeze(0).cpu().numpy()       # [T]
        post_rewards = np.array(post_rewards).squeeze(1)       # [T]
        post_continues = np.array(post_continues).squeeze(1)   # [T]
        pri_rewards = np.array(pri_rewards).squeeze(1)        # [T]
        pri_continues = np.array(pri_continues).squeeze(1)     # [T]

        return {
            "post_err": post_err,
            "pri_err": pri_err,
            "horizon_errors": horizon_errors,
            "post_rewards": post_rewards,
            "pri_rewards": pri_rewards,
            "gt_rewards": gt_rewards,
            "post_continues": post_continues,
            "pri_continues": pri_continues,
            "gt_continues": gt_continues,
            "burn_in_steps": burn_in_steps,
        }


def visualize_rssm(history, post_err, pri_err,
                   post_rewards, pri_rewards, gt_rewards,
                   post_continues, pri_continues, gt_continues):
    """可视化训练过程和 Posterior vs Prior 对比（2x2 布局）。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [0,0]: 损失曲线（含 reward_loss 和 continue_loss）
    ax = axes[0, 0]
    ax.plot(history["recon_loss"], label="Reconstruction")
    ax.plot(history["kl_loss"], label="KL (posterior || prior)")
    ax.plot(history["reward_loss"], label="Reward")
    ax.plot(history["continue_loss"], label="Continue")
    ax.plot(history["total_loss"], label="Total", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("RSSM Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1]: Posterior vs Prior 重建误差
    ax = axes[0, 1]
    steps = range(len(post_err))
    ax.plot(steps, post_err, label="Posterior (用真实观测)", linewidth=2)
    ax.plot(steps, pri_err, label="Prior (burn-in + 想象)", linewidth=2)
    # 标注 burn-in 与 prior rollout 的分界
    burn_in = 5
    if burn_in < len(post_err):
        ax.axvline(x=burn_in, color="red", linestyle="--", alpha=0.5, label=f"Burn-in end (t={burn_in})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reconstruction Error (L2)")
    ax.set_title("Posterior vs Prior: 重建误差 (burn-in 后为纯 prior)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,0]: Reward prediction vs ground truth
    ax = axes[1, 0]
    ax.plot(steps, gt_rewards, label="Ground Truth", linewidth=2, color="black")
    ax.plot(steps, post_rewards, label="Posterior Predict", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.plot(steps, pri_rewards, label="Prior Predict", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Prediction vs Ground Truth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,1]: Continue prediction vs actual
    ax = axes[1, 1]
    ax.plot(steps, gt_continues, label="Actual", linewidth=2, color="black")
    ax.plot(steps, post_continues, label="Posterior Predict", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.plot(steps, pri_continues, label="Prior Predict", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Continue Probability")
    ax.set_title("Continue Prediction vs Actual")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = Path(__file__).parent.parent / "results" / "world_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "rssm_training_analysis.png", dpi=150)
    print(f"\n[Saved] {out_dir / 'rssm_training_analysis.png'}")


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dreamer V3 RSSM Demo")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stoch_dim", type=int, default=16)
    parser.add_argument("--deter_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("Dreamer V3 RSSM (Recurrent State-Space Model) Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # --- 1. 数据 ---
    total_samples = 3000
    train_ratio, val_ratio = 0.7, 0.15  # test = 0.15
    print(f"\n[Data] 生成 {total_samples} 条 2D 导航轨迹 (seq_len={args.seq_len})...")
    dataset = NoisyTrajectoryDataset(num_samples=total_samples, seq_len=args.seq_len)
    print(f"  观测维度: 6 (x, y, vx, vy, goal_dx, goal_dy)")
    print(f"  动作维度: 2")
    print(f"  噪声 std: 0.05")
    print(f"  Termination: 目标到达(threshold=0.3) / 边界碰撞(limit=4.0) / 序列截断")

    # Train / Val / Test split
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"  数据划分: train={n_train}, val={n_val}, test={n_test}")

    # --- 2. 模型 ---
    model = RSSM(
        obs_dim=6,  # [x, y, vx, vy, goal_dx, goal_dy]
        act_dim=2,
        stoch_dim=args.stoch_dim,
        deter_dim=args.deter_dim,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] RSSM")
    print(f"  确定性维度 h: {args.deter_dim}")
    print(f"  随机性维度 z: {args.stoch_dim}")
    print(f"  总参数量: {total_params:,}")

    # --- 3. 训练 ---
    print(f"\n[Train] 开始训练 ({args.epochs} epochs, train={n_train}, val={n_val})...")
    print("-" * 60)
    history = train_rssm(model, train_loader, val_loader=val_loader, epochs=args.epochs, device=device)
    print("-" * 60)

    # --- 4. 想象展开 (held-out test set, 多样本平均) ---
    print(f"\n[Test] 在 {n_test} 条 held-out test 轨迹上评估...")
    n_test_samples = min(20, n_test)  # 取 20 条 test 轨迹求平均
    test_indices = list(range(n_test_samples))

    all_post_err, all_pri_err = [], []
    all_reward_mae, all_continue_f1 = [], []
    all_horizon_errors = {}  # {horizon: [errors across samples]}

    for idx in test_indices:
        test_obs, test_act, test_rew, test_cont = test_set[idx]
        result = imagine_rollout(
            model, test_obs, test_act, test_rew, test_cont, device=device)

        post_err = result["post_err"]
        pri_err = result["pri_err"]
        post_rewards = result["post_rewards"]
        gt_rewards = result["gt_rewards"]
        post_continues = result["post_continues"]
        gt_continues = result["gt_continues"]
        horizon_errors = result["horizon_errors"]

        all_post_err.append(post_err)
        all_pri_err.append(pri_err)
        all_reward_mae.append(np.mean(np.abs(post_rewards - gt_rewards)))
        # Continue F1: 预测 >0.5 为正类, gt >0.5 为正类
        pred_binary = (post_continues > 0.5).astype(float)
        gt_binary = (gt_continues > 0.5).astype(float)
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        all_continue_f1.append(f1)

        # 累积 horizon 误差
        for horizon, err_val in horizon_errors.items():
            if horizon not in all_horizon_errors:
                all_horizon_errors[horizon] = []
            all_horizon_errors[horizon].append(err_val)

    # 统计 prior rollout 的 mean err（排除 burn-in 步）
    burn_in = 5  # 默认 burn_in_steps
    mean_post_err = np.mean([e.mean() for e in all_post_err])
    mean_pri_err = np.mean([e[burn_in:].mean() for e in all_pri_err])  # 仅 rollout 部分
    mean_rew_mae = np.mean(all_reward_mae)
    mean_cont_f1 = np.mean(all_continue_f1)
    # Majority-class baseline: 随机预测 continue=1 的比例
    # 由于 termination 是状态相关的，baseline 不再是简单的 (seq_len-1)/seq_len
    majority_baseline = np.mean([gt.mean() for gt in
        [test_set[i][3].numpy() for i in range(n_test_samples)]])

    print(f"  [Test set, n={n_test_samples}, burn-in={burn_in} steps]")
    print(f"  Posterior 平均误差: {mean_post_err:.4f} (用真实观测)")
    print(f"  Prior 平均误差:    {mean_pri_err:.4f} (burn-in 后纯想象)")
    print(f"  差距:              {mean_pri_err - mean_post_err:.4f}")
    print(f"  Reward MAE:        {mean_rew_mae:.4f} (held-out)")
    print(f"  Continue F1:       {mean_cont_f1:.4f} (majority baseline={majority_baseline:.4f})")
    print()
    print("  --- Horizon 累积误差 (prior rollout, 越小越好) ---")
    for horizon in sorted(all_horizon_errors.keys()):
        vals = [v for v in all_horizon_errors[horizon] if not np.isnan(v)]
        if vals:
            print(f"    H={horizon:2d}: {np.mean(vals):.4f}")
    print()
    print("  -> Posterior 比 Prior 准确（因为它能'看到'真实观测）")
    print("  -> Prior 的误差随 horizon 增大而累积（只能靠历史'猜'未来）")
    print("  -> Continue F1 需超过 majority baseline 才说明模型学会了状态相关 termination")

    # --- 5. 可视化 (用第一条 test 轨迹) ---
    vis_obs, vis_act, vis_rew, vis_cont = test_set[0]
    vis_result = imagine_rollout(
        model, vis_obs, vis_act, vis_rew, vis_cont, device=device)

    visualize_rssm(history, vis_result["post_err"], vis_result["pri_err"],
                    vis_result["post_rewards"], vis_result["pri_rewards"], vis_result["gt_rewards"],
                    vis_result["post_continues"], vis_result["pri_continues"], vis_result["gt_continues"])

    # --- 6. 总结 ---
    print("\n" + "=" * 60)
    print("RSSM 核心概念回顾：")
    print("=" * 60)
    print("1. h_t (确定性 GRU): 记忆历史 -> 捕捉可预测的运动学规律")
    print("2. z_t (随机性 Gaussian): 捕捉不确定的碰撞/摩擦/滑动")
    print("3. Prior:     p(z_t | h_t)       -- 规划/想象时用（不看观测）")
    print("4. Posterior: q(z_t | h_t, o_t)   -- 训练/更新时用（看观测）")
    print("5. KL(posterior || prior): 让 prior 学会预测 posterior")
    print("6. Reward predictor: 从状态预测 reward -> 用于价值估计")
    print("7. Continue predictor: 从状态预测 episode 是否继续 -> 用于折扣")
    print("8. 想象展开: 用 prior 自回归预测未来 -> 在'脑中'模拟环境")
    print()
    print("与 VLA 的关系：")
    print("  VLA 用 Transformer 编码历史 → RSSM 用 GRU 编码历史")
    print("  两者都可以用于策略学习，但 RSSM 更高效（隐藏维度更小）")
    print("=" * 60)


if __name__ == "__main__":
    main()