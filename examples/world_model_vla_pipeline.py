#!/usr/bin/env python3
"""
world_model_vla_pipeline.py
============================
World Model + Policy Pipeline Demo。

演示世界模型的四种融合方式（对应 docs/07-world-models-for-vla.md 第 4 节）：
  1. World Model-assisted Reward Augmentation — 用 WM 增强 reward 信号训练策略
  2. 世界模型作为评估器 — 用 WM 预演多个候选动作，选最优
  3. 世界模型作为规划器 — WM 多步展开 + 搜索最优动作序列
  4. Latent-space Behavior Cloning — WM 在 latent space 同时预测状态和动作

环境：2D 导航（目标点趋近任务）。
策略：简单 MLP policy。
世界模型：Latent Dynamics Model。

依赖：pip install torch matplotlib
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 环境：2D 导航
# ============================================================

class Nav2DEnv:
    """简单的 2D 导航环境：Agent 从起点移动到目标点。"""

    def __init__(self, map_size=10.0, max_steps=50):
        self.map_size = map_size
        self.max_steps = max_steps

    def reset(self, start=None, goal=None):
        self.pos = np.array(start if start else np.random.rand(2) * self.map_size, dtype=np.float32)
        self.goal = np.array(goal if goal else np.random.rand(2) * self.map_size, dtype=np.float32)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.pos, self.goal, self.pos - self.goal])  # [6]

    def step(self, action, obs_before=None):
        """action: [2] 速度指令。返回 (obs_before, reward, done, next_obs)。"""
        if obs_before is None:
            obs_before = self._get_obs()
        self.pos = self.pos + action * 0.1
        self.pos = np.clip(self.pos, 0, self.map_size)
        self.step_count += 1

        next_obs = self._get_obs()
        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist  # 距离目标越近 reward 越高
        done = dist < 0.3 or self.step_count >= self.max_steps
        return obs_before, reward, done, next_obs

    def generate_demonstration(self, max_steps=30):
        """生成一条专家演示轨迹（简单 PD 控制器）。返回 (obs, next_obs, act, rew) 四元组。"""
        obs_list, next_obs_list, act_list, rew_list = [], [], [], []
        obs = self.reset()
        for _ in range(max_steps):
            # PD 控制器：朝目标移动
            diff = self.goal - self.pos
            dist = np.linalg.norm(diff)
            action = diff / (dist + 1e-6) * min(dist, 1.0)
            obs_list.append(obs)
            act_list.append(action.astype(np.float32))
            obs, reward, done, next_obs = self.step(action)
            next_obs_list.append(next_obs)
            rew_list.append(reward)
            if done:
                break
        return (
            np.array(obs_list, dtype=np.float32),
            np.array(next_obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32),
            np.array(rew_list, dtype=np.float32),
        )


# ============================================================
# 2. 策略网络 & 世界模型
# ============================================================

class PolicyNet(nn.Module):
    """简单 MLP 策略网络。"""
    def __init__(self, obs_dim=6, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),  # 限制动作范围 [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)


class LatentWorldModel(nn.Module):
    """Latent space 世界模型。"""
    def __init__(self, obs_dim=6, action_dim=2, latent_dim=32, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.latent_dim = latent_dim

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next(self, z, action):
        return z + self.transition(torch.cat([z, action], dim=-1))

    def predict_reward(self, z):
        return self.reward_head(z).squeeze(-1)


# ============================================================
# 3. 四种融合方式实现
# ============================================================

class WM_VLA_Pipeline:
    """世界模型 + VLA 的四种融合方式 Demo。"""

    def __init__(self, obs_dim=6, action_dim=2, latent_dim=32, device="cpu"):
        self.device = device
        self.wm = LatentWorldModel(obs_dim, action_dim, latent_dim).to(device)
        self.policy = PolicyNet(obs_dim, action_dim).to(device)

    def train_world_model(self, data, epochs=20, lr=1e-3, batch_size=64):
        """训练世界模型。"""
        print("\n[1/4] 训练世界模型...")
        obs_data, next_obs_data, act_data, rew_data = data

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(obs_data),
            torch.FloatTensor(next_obs_data),
            torch.FloatTensor(act_data),
            torch.FloatTensor(rew_data),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.wm.parameters(), lr=lr)
        mse = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for obs, next_obs, act, rew in loader:
                obs, next_obs, act, rew = obs.to(self.device), next_obs.to(self.device), act.to(self.device), rew.to(self.device)

                z = self.wm.encode(obs)
                z_next = self.wm.encode(next_obs).detach()  # FIXED: target is next_obs
                z_pred = self.wm.predict_next(z, act)
                rew_pred = self.wm.predict_reward(z_next)

                loss = mse(z_pred, z_next) * 0.5 + mse(rew_pred, rew) * 0.5
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.4f}")

    def train_policy_bc(self, data, epochs=20, lr=1e-3, batch_size=64):
        """行为克隆训练策略（baseline）。"""
        print("\n[2/4] BC 训练策略（baseline）...")
        obs_data, _, act_data, _ = data

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(obs_data), torch.FloatTensor(act_data)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        mse = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for obs, act in loader:
                obs, act = obs.to(self.device), act.to(self.device)
                pred = self.policy(obs)
                loss = mse(pred, act)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.4f}")

    def fusion_1_data_generator(self, env, num_gen=100, seq_len=20):
        """融合方式 1：World Model-assisted Reward Augmentation。"""
        print("\n[融合方式 1] World Model-assisted Reward Augmentation")
        print("  用策略在环境中采样，结合 WM 的 reward 预测生成增强数据，再训练新策略")
        print("  注：本 Demo 的 WM 无 decoder，observation 来自真实环境 step，reward 来自 WM 预测增强")

        self.wm.eval()
        policy_gen = PolicyNet(6, 2).to(self.device)

        # 用 WM 生成虚拟数据
        synthetic_obs, synthetic_act, synthetic_rew = [], [], []

        with torch.no_grad():
            for _ in range(num_gen):
                obs = env.reset()
                z = self.wm.encode(torch.FloatTensor(obs).unsqueeze(0).to(self.device))

                for t in range(seq_len):
                    # 用当前策略（加噪声探索）生成动作
                    act = self.policy(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                    act = act.squeeze(0).cpu().numpy() + np.random.randn(2) * 0.1

                    synthetic_obs.append(obs)
                    synthetic_act.append(act.astype(np.float32))

                    # 真实环境 step 获取下一观测（WM 无 decoder，无法从 latent 重建 observation）
                    _, env_rew, done, next_obs = env.step(act)
                    # 同时用 WM 预测 reward 作为对比/增强信号
                    with torch.no_grad():
                        z_next = self.wm.encode(torch.FloatTensor(next_obs).unsqueeze(0).to(self.device))
                        wm_rew = self.wm.predict_reward(z_next).item()
                    # 混合 reward：真实环境 reward + WM 预测 reward（加权平均）
                    synthetic_rew.append(0.5 * env_rew + 0.5 * wm_rew)
                    obs = next_obs
                    if done:
                        break

        synth_obs = np.array(synthetic_obs, dtype=np.float32)
        synth_act = np.array(synthetic_act, dtype=np.float32)
        synth_rew = np.array(synthetic_rew, dtype=np.float32)

        print(f"  生成 {len(synth_obs)} 条虚拟转移")

        # 用虚拟数据训练新策略
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(synth_obs), torch.FloatTensor(synth_act)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(policy_gen.parameters(), lr=1e-3)
        mse = nn.MSELoss()

        for epoch in range(10):
            for obs_b, act_b in loader:
                obs_b, act_b = obs_b.to(self.device), act_b.to(self.device)
                loss = mse(policy_gen(obs_b), act_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 评估
        avg_rew = self._evaluate_policy(policy_gen, env, num_episodes=20)
        print(f"  WM 数据生成训练的策略平均 reward: {avg_rew:.2f}")
        return avg_rew

    def fusion_2_evaluator(self, env, num_candidates=5):
        """融合方式 2：世界模型作为评估器。"""
        print("\n[融合方式 2] 世界模型作为评估器")
        print("  生成多个候选动作，用 WM 预演并选 reward 最高的")

        self.wm.eval()
        obs = env.reset()
        total_rew, steps = 0.0, 0

        for _ in range(env.max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # 生成 K 个候选动作
            with torch.no_grad():
                base_action = self.policy(obs_t).squeeze(0).cpu().numpy()
                candidates = [base_action + np.random.randn(2) * 0.3 for _ in range(num_candidates)]

                # 用 WM 评估每个候选动作
                z = self.wm.encode(obs_t)
                best_rew, best_act = -float("inf"), base_action
                for cand in candidates:
                    cand_t = torch.FloatTensor(cand).unsqueeze(0).to(self.device)
                    z_next = self.wm.predict_next(z, cand_t)
                    r_pred = self.wm.predict_reward(z_next).item()
                    if r_pred > best_rew:
                        best_rew = r_pred
                        best_act = cand

            obs, reward, done, _ = env.step(best_act)
            total_rew += reward
            steps += 1
            if done:
                break

        avg_rew = total_rew / max(steps, 1)
        print(f"  WM 评估器引导的策略 reward: {avg_rew:.2f}")
        return avg_rew

    def fusion_3_planner(self, env, horizon=5, num_rollouts=10):
        """融合方式 3：世界模型作为规划器。"""
        print("\n[融合方式 3] 世界模型作为规划器（Model-Based Planning）")
        print(f"  用 WM 展开 {horizon} 步，搜索最优动作序列，仅执行第 1 步")

        self.wm.eval()
        obs = env.reset()
        total_rew, steps = 0.0, 0

        for _ in range(env.max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                z = self.wm.encode(obs_t)
                best_action = self.policy(obs_t).squeeze(0).cpu().numpy()
                best_cum_rew = -float("inf")

                # 随机 rollout 搜索
                for _ in range(num_rollouts):
                    z_roll = z.clone()
                    cum_rew = 0.0
                    first_action = None

                    for h in range(horizon):
                        # 随机动作（加少量策略引导）
                        if h == 0:
                            action = self.policy(obs_t).squeeze(0).cpu() + torch.randn(2) * 0.3
                            first_action = action.numpy()
                        else:
                            action = torch.randn(2).to(self.device) * 0.5

                        z_roll = self.wm.predict_next(z_roll, action.unsqueeze(0))
                        cum_rew += self.wm.predict_reward(z_roll).item()

                    if cum_rew > best_cum_rew:
                        best_cum_rew = cum_rew
                        best_action = first_action

            obs, reward, done, _ = env.step(best_action)
            total_rew += reward
            steps += 1
            if done:
                break

        avg_rew = total_rew / max(steps, 1)
        print(f"  WM 规划器引导的策略 reward: {avg_rew:.2f}")
        return avg_rew

    def fusion_4_wam(self, env, steps=30):
        """融合方式 4：Latent-space Behavior Cloning（WM 直接输出动作）。"""
        print("\n[融合方式 4] Latent-space Behavior Cloning")
        print("  在 latent space 同时预测状态和动作，WM 本身就是策略")

        if not hasattr(self, '_cached_data') or self._cached_data is None:
            raise RuntimeError(
                "fusion_4_wam 需要先调用 run() 方法生成训练数据。"
                "示例：pipeline = WM_VLA_Pipeline(); pipeline.run()  # 先收集数据\n"
                "      pipeline.fusion_4_wam(env)  # 再调用本方法"
            )

        self.wm.eval()

        # 为 WM 添加一个动作预测头
        action_head = nn.Sequential(
            nn.Linear(self.wm.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh(),
        ).to(self.device)

        # 用演示数据快速训练动作头（模拟 DreamZero 思路）
        obs_data, _, act_data, _ = self._cached_data
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(obs_data), torch.FloatTensor(act_data)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(action_head.parameters(), lr=1e-3)
        mse = nn.MSELoss()

        for epoch in range(15):
            for obs_b, act_b in loader:
                obs_b, act_b = obs_b.to(self.device), act_b.to(self.device)
                z = self.wm.encode(obs_b)
                act_pred = action_head(z)
                loss = mse(act_pred, act_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 评估 WAM
        obs = env.reset()
        total_rew, step_count = 0.0, 0
        for _ in range(steps):
            with torch.no_grad():
                z = self.wm.encode(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                action = action_head(z).squeeze(0).cpu().numpy()
            obs, reward, done, _ = env.step(action)
            total_rew += reward
            step_count += 1
            if done:
                break

        avg_rew = total_rew / max(step_count, 1)
        print(f"  WAM 策略 reward: {avg_rew:.2f}")
        return avg_rew

    def _evaluate_policy(self, policy, env, num_episodes=20):
        """评估策略在真实环境中的表现。"""
        policy.eval()
        total_rewards = []
        for _ in range(num_episodes):
            obs = env.reset()
            ep_rew, steps = 0.0, 0
            for _ in range(env.max_steps):
                with torch.no_grad():
                    action = policy(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                    action = action.squeeze(0).cpu().numpy()
                obs, reward, done, _ = env.step(action)
                ep_rew += reward
                steps += 1
                if done:
                    break
            total_rewards.append(ep_rew / max(steps, 1))
        return np.mean(total_rewards)

    def run(self, num_demos=200):
        """运行完整管线。"""
        print("=" * 60)
        print("World Model + VLA Pipeline Demo")
        print("=" * 60)

        env = Nav2DEnv()

        # 生成专家演示
        print(f"\n[0/4] 生成 {num_demos} 条专家演示...")
        all_obs, all_next_obs, all_act, all_rew = [], [], [], []
        for _ in range(num_demos):
            obs, next_obs, act, rew = env.generate_demonstration()
            all_obs.append(obs)
            all_next_obs.append(next_obs)
            all_act.append(act)
            all_rew.append(rew)
        data = (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_next_obs, axis=0),
            np.concatenate(all_act, axis=0),
            np.concatenate(all_rew, axis=0),
        )
        self._cached_data = data
        print(f"  收集 {len(data[0])} 条转移四元组 (obs, next_obs, act, rew)")

        # 训练世界模型和策略
        self.train_world_model(data)
        self.train_policy_bc(data)

        # 评估 baseline
        baseline_rew = self._evaluate_policy(self.policy, env, num_episodes=20)
        print(f"\n  BC baseline 平均 reward: {baseline_rew:.2f}")

        # 四种融合方式
        results = {"BC Baseline": baseline_rew}
        results["WM 数据生成器"] = self.fusion_1_data_generator(env)
        results["WM 评估器"] = self.fusion_2_evaluator(env)
        results["WM 规划器"] = self.fusion_3_planner(env)
        results["WAM"] = self.fusion_4_wam(env)

        # 对比可视化
        self._plot_results(results)
        return results

    def _plot_results(self, results):
        """绘制四种融合方式的 reward 对比。"""
        fig, ax = plt.subplots(figsize=(10, 5))

        names = list(results.keys())
        values = list(results.values())
        colors = ["#6c757d", "#0d6efd", "#198754", "#ffc107", "#dc3545"]

        bars = ax.bar(names, values, color=colors[:len(names)], edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Average Reward (higher is better)")
        ax.set_title("World Model + Policy: Four Fusion Strategies")
        ax.set_xticklabels(names, rotation=15, ha="right")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("wm_vla_fusion_comparison.png", dpi=150)
        print(f"\n[Saved] wm_vla_fusion_comparison.png")


# ============================================================
# 主函数
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    pipeline = WM_VLA_Pipeline(device=device)
    results = pipeline.run(num_demos=200)

    print("\n" + "=" * 60)
    print("结果对比总结：")
    print("=" * 60)
    for name, rew in results.items():
        marker = " ★" if name != "BC Baseline" and rew > results["BC Baseline"] else ""
        print(f"  {name:20s}: {rew:.2f}{marker}")
    print()
    print("核心结论：")
    print("  1. WM 作为数据生成器：解决数据稀缺，但受 WM 质量影响")
    print("  2. WM 作为评估器：最简单有效的融合方式，安全验证场景首选")
    print("  3. WM 作为规划器：长程任务最强，但计算成本高")
    print("  4. WAM：架构最简洁，但训练难度最大")
    print("  → 实际项目中，融合方式 2（评估器）和 3（规划器）最常用")
    print("=" * 60)


if __name__ == "__main__":
    main()