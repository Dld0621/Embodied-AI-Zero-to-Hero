"""
Diffusion Policy 训练示例
=========================
State-conditioned Diffusion Policy Baseline.
完整的 Diffusion Policy 训练脚本，支持：
  - 合成数据（无需真实机器人数据，快速验证）
  - ALOHA 数据集（需下载 lerobot 数据集）
  - 推理与可视化

架构：
  - 观测编码器：MLP 或 ResNet（视觉）
  - 噪声预测网络：1D Temporal Convolution
  - 扩散调度：DDPM 平方余弦

注意：本脚本是 State-conditioned Diffusion Policy Baseline，不是 VLA 训练。
VLA 训练请参考 examples/vla_demo.py 和 examples/minimal_vla.py。

Usage:
    # 合成数据训练（CPU/GPU 均可，5 分钟）
    python train_diffusion_policy.py --mode train --data synthetic --epochs 50

    # ALOHA 数据训练（需 GPU + 数据集）
    python train_diffusion_policy.py --mode train --data aloha --dataset_dir ./data/aloha

    # 推理
    python train_diffusion_policy.py --mode infer --checkpoint ./checkpoints/dp_best.pt

    # 可视化动作分布
    python train_diffusion_policy.py --mode visualize --checkpoint ./checkpoints/dp_best.pt

Dependencies:
    pip install torch numpy matplotlib tqdm
"""

import argparse
import os
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. 扩散调度器 (DDPM)
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """DDPM 噪声调度器：平方余弦（cosine）调度。"""

    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        # 平方余弦 alpha_bar 调度
        timesteps = torch.arange(num_steps + 1, dtype=torch.float32)
        s = 0.008  # 偏移量，防止 beta 在 t=0 时过小
        f = torch.cos((timesteps / num_steps + s) / (1 + s) * np.pi / 2) ** 2
        self.alpha_bar = f / f[0]
        self.beta = torch.clamp(1 - self.alpha_bar[1:] / self.alpha_bar[:-1], 0.0001, 0.9999)
        self.alpha = 1.0 - self.beta
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar[1:])
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[1:])

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """前向扩散：q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def sample_step(self, model: nn.Module, xt: torch.Tensor, t: int, obs_cond: torch.Tensor) -> torch.Tensor:
        """单步去噪（DDPM）"""
        with torch.no_grad():
            t_batch = torch.full((xt.shape[0],), t, dtype=torch.long, device=xt.device)
            noise_pred = model(xt, t_batch, obs_cond)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t + 1]
            beta_t = self.beta[t]
            # 计算 x_{t-1}
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / (torch.sqrt(alpha_t) * torch.sqrt(1 - alpha_bar_t))
            mean = coef1 * (xt - coef2 * noise_pred)
            if t > 0:
                noise = torch.randn_like(xt)
                variance = torch.sqrt(beta_t) * noise
                return mean + variance
            return mean


# ---------------------------------------------------------------------------
# 2. 噪声预测网络 (1D Temporal Conv)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """时间步的正弦位置编码。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Conv1dBlock(nn.Module):
    """带 GroupNorm 和 Mish 激活的 1D Conv。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConditionalResidualBlock1D(nn.Module):
    """条件残差块：动作特征 + 时间条件 + 观测条件。"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = Conv1dBlock(in_ch, out_ch, kernel_size)
        self.conv2 = Conv1dBlock(out_ch, out_ch, kernel_size)
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch),
        )
        self.obs_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch),
        )
        self.residual_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, obs_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # t_emb, obs_emb: (B, cond_dim)
        out = self.conv1(x)
        # 加入时间条件（全局 bias）
        out = out + self.time_mlp(t_emb)[:, :, None]
        # 加入观测条件
        out = out + self.obs_mlp(obs_emb)[:, :, None]
        out = self.conv2(out)
        return out + self.residual_conv(x)


class NoisePredictionNet(nn.Module):
    """
    噪声预测网络：基于 1D Temporal Convolution U-Net。
    输入：带噪动作序列 (B, action_dim, T_pred)
    输出：预测的噪声 (B, action_dim, T_pred)
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        T_pred: int = 16,
        hidden_dim: int = 128,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.T_pred = T_pred

        # 观测编码器（简单的 MLP）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 时间编码
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # 输入投影
        self.input_conv = nn.Conv1d(action_dim, hidden_dim, 1)

        # 下采样 + 中间 + 上采样
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        ch = hidden_dim
        for i in range(num_blocks):
            self.down_blocks.append(ConditionalResidualBlock1D(ch, ch, hidden_dim))
            self.up_blocks.insert(0, ConditionalResidualBlock1D(ch, ch, hidden_dim))

        # 输出
        self.output_conv = nn.Conv1d(hidden_dim, action_dim, 1)

    def forward(self, noisy_action: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # noisy_action: (B, action_dim, T_pred)
        # t: (B,)
        # obs: (B, obs_dim)
        t_emb = self.time_encoder(t)
        obs_emb = self.obs_encoder(obs)

        x = self.input_conv(noisy_action)
        skips = []
        for block in self.down_blocks:
            x = block(x, t_emb, obs_emb)
            skips.append(x)
        for block in self.up_blocks:
            x = x + skips.pop()
            x = block(x, t_emb, obs_emb)
        return self.output_conv(x)


# ---------------------------------------------------------------------------
# 3. Diffusion Policy 主类
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy：将噪声预测网络包装为策略。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        T_pred: int = 16,
        num_diffusion_steps: int = 100,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.T_pred = T_pred
        self.noise_net = NoisePredictionNet(action_dim, obs_dim, T_pred, hidden_dim)
        self.scheduler = DDPMScheduler(num_diffusion_steps)

    def forward(self, noisy_action: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.noise_net(noisy_action, t, obs)

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        """推理：从纯噪声开始去噪，得到动作序列。"""
        B = obs.shape[0]
        device = obs.device
        # 初始化为标准高斯噪声
        xt = torch.randn(B, self.action_dim, self.T_pred, device=device)
        # 反向扩散
        for t in reversed(range(self.scheduler.num_steps)):
            xt = self.scheduler.sample_step(self.noise_net, xt, t, obs)
        return xt  # (B, action_dim, T_pred)


# ---------------------------------------------------------------------------
# 4. 数据集
# ---------------------------------------------------------------------------

class SyntheticPushDataset(Dataset):
    """
    合成数据集：模拟平面推送任务。
    观测：物体位置 (x, y) + 目标位置 (gx, gy) = 4-dim
    动作：末端执行器速度 (vx, vy) = 2-dim，预测未来 T_pred 步
    """

    def __init__(self, num_episodes: int = 1000, T_pred: int = 16, episode_len: int = 50):
        self.T_pred = T_pred
        self.data = []
        for _ in range(num_episodes):
            # 随机物体位置和目标
            obj_pos = np.random.randn(2) * 0.3
            goal_pos = np.random.randn(2) * 0.3
            # 生成轨迹（简单 PD 控制朝向目标）
            traj = np.zeros((episode_len, 2))
            pos = obj_pos.copy()
            for t in range(episode_len):
                diff = goal_pos - pos
                vel = np.clip(diff * 2.0, -0.1, 0.1)
                traj[t] = vel
                pos += vel * 0.05
            # 切分为 (obs, action_seq) 对
            for t in range(episode_len - T_pred):
                obs = np.concatenate([obj_pos + np.random.randn(2) * 0.02, goal_pos])
                action_seq = traj[t : t + T_pred].T  # (2, T_pred)
                self.data.append((obs.astype(np.float32), action_seq.astype(np.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class AlohaStyleDataset(Dataset):
    """
    ALOHA 风格数据集加载器。
    期望数据格式：HDF5 或 numpy 文件，包含 'observations' 和 'actions'。
    """

    def __init__(self, data_dir: str, T_pred: int = 16):
        self.T_pred = T_pred
        self.data = []
        import glob

        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"在 {data_dir} 中未找到 .npy 文件")
        for f in npy_files:
            d = np.load(f, allow_pickle=True).item()
            obs = d["observations"]  # (T, obs_dim)
            act = d["actions"]  # (T, action_dim)
            T = len(obs)
            for t in range(T - T_pred):
                self.data.append((obs[t].astype(np.float32), act[t : t + T_pred].T.astype(np.float32)))
        print(f"[Dataset] 加载 {len(self.data)} 条样本，来自 {len(npy_files)} 个文件")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


# ---------------------------------------------------------------------------
# 5. 训练
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 设备: {device}")

    # 数据集
    if args.data == "synthetic":
        dataset = SyntheticPushDataset(num_episodes=args.num_episodes, T_pred=args.T_pred)
        obs_dim, action_dim = 4, 2
    else:
        dataset = AlohaStyleDataset(args.dataset_dir, T_pred=args.T_pred)
        # 自动推断维度
        obs_dim = dataset.data[0][0].shape[0]
        action_dim = dataset.data[0][1].shape[0]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[Train] 数据集大小: {len(dataset)}, 批次大小: {args.batch_size}")

    # 模型
    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        T_pred=args.T_pred,
        num_diffusion_steps=args.diffusion_steps,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    # 训练循环
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        policy.train()
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for obs, action_seq in pbar:
            obs = obs.to(device)
            action_seq = action_seq.to(device)  # (B, action_dim, T_pred)

            # 随机采样时间步
            B = action_seq.shape[0]
            t = torch.randint(0, policy.scheduler.num_steps, (B,), device=device)

            # 前向扩散
            noisy_action, noise = policy.scheduler.add_noise(action_seq, t)

            # 预测噪声
            noise_pred = policy(noisy_action, t, obs)

            # MSE Loss
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = np.mean(epoch_losses)
        print(f"[Epoch {epoch+1}] 平均 Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            path = os.path.join(args.checkpoint_dir, "dp_best.pt")
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "T_pred": args.T_pred,
                    "args": vars(args),
                },
                path,
            )
            print(f"  -> 保存最佳模型到 {path}")

    print("[Train] 训练完成！")


# ---------------------------------------------------------------------------
# 6. 推理
# ---------------------------------------------------------------------------

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    ckpt = torch.load(args.checkpoint, map_location=device)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    T_pred = ckpt["T_pred"]

    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        T_pred=T_pred,
        num_diffusion_steps=ckpt["args"]["diffusion_steps"],
        hidden_dim=ckpt["args"]["hidden_dim"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # 合成测试观测
    num_samples = 10
    obs = torch.randn(num_samples, obs_dim, device=device) * 0.3
    if obs_dim == 4:
        # 如果是推送任务，构造合理的观测
        obs[:, 2:] = torch.randn(num_samples, 2, device=device) * 0.3  # 目标位置

    print(f"[Infer] 观测形状: {obs.shape}")
    print(f"[Infer] 推理中 (扩散步数={policy.scheduler.num_steps})...")

    start = time.time()
    with torch.no_grad():
        action_seq = policy.infer(obs)  # (B, action_dim, T_pred)
    elapsed = time.time() - start

    print(f"[Infer] 完成！耗时: {elapsed:.3f}s ({elapsed / num_samples * 1000:.1f}ms/样本)")
    print(f"[Infer] 动作序列形状: {action_seq.shape}")
    print(f"[Infer] 动作均值: {action_seq.mean().item():.4f}, 标准差: {action_seq.std().item():.4f}")

    # 保存
    out_path = args.checkpoint.replace(".pt", "_infer.npy")
    np.save(out_path, action_seq.cpu().numpy())
    print(f"[Infer] 结果保存到 {out_path}")


# ---------------------------------------------------------------------------
# 7. 可视化
# ---------------------------------------------------------------------------

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    T_pred = ckpt["T_pred"]

    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        T_pred=T_pred,
        num_diffusion_steps=ckpt["args"]["diffusion_steps"],
        hidden_dim=ckpt["args"]["hidden_dim"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # 采样多组动作
    num_samples = 100
    obs = torch.randn(1, obs_dim, device=device) * 0.3
    obs_batch = obs.repeat(num_samples, 1)

    with torch.no_grad():
        actions = policy.infer(obs_batch)  # (100, action_dim, T_pred)
    actions = actions.cpu().numpy()

    # 绘图
    fig, axes = plt.subplots(action_dim, 1, figsize=(8, 2 * action_dim))
    if action_dim == 1:
        axes = [axes]
    for d in range(action_dim):
        ax = axes[d]
        for i in range(min(20, num_samples)):
            ax.plot(actions[i, d, :], alpha=0.3, color="blue")
        ax.plot(actions[:, d, :].mean(axis=0), color="red", linewidth=2, label="Mean")
        ax.set_title(f"Action Dim {d}: 20 samples + mean")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
    plt.tight_layout()
    out_path = args.checkpoint.replace(".pt", "_viz.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Visualize] 图像保存到 {out_path}")
    plt.close()

    # 动作分布直方图（第一个时间步）
    fig, axes = plt.subplots(1, action_dim, figsize=(4 * action_dim, 3))
    if action_dim == 1:
        axes = [axes]
    for d in range(action_dim):
        axes[d].hist(actions[:, d, 0], bins=30, alpha=0.7)
        axes[d].set_title(f"Action Dim {d} @ t=0")
        axes[d].set_xlabel("Value")
        axes[d].set_ylabel("Count")
    plt.tight_layout()
    out_path2 = args.checkpoint.replace(".pt", "_hist.png")
    plt.savefig(out_path2, dpi=150)
    print(f"[Visualize] 直方图保存到 {out_path2}")


# ---------------------------------------------------------------------------
# 8. 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy 训练与推理")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer", "visualize"])
    parser.add_argument("--data", type=str, default="synthetic", choices=["synthetic", "aloha"])
    parser.add_argument("--dataset_dir", type=str, default="./data/aloha")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/dp_best.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T_pred", type=int, default=16, help="预测动作序列长度")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="扩散步数")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_episodes", type=int, default=1000, help="合成数据 episode 数")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "visualize":
        visualize(args)


if __name__ == "__main__":
    main()
