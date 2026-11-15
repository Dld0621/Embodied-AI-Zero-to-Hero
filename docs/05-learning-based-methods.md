# 基于学习的 Retargeting 方法

> **H05 已修订：**本页已统一每指 `5×3=15` 维输入、修正函数参数顺序和图像形状解包，并在时序 Transformer 前加入显式正弦时间位置编码。所有 Python fence 已做语法检查；PyTorch 缺失时，网络前向与位置编码形状测试会明确跳过，不能报告为已执行。未训练模型、下载权重或验证泛化性能。复核范围见 [修订记录](reviews/retargeting-revision-review.md#复核结论)。

> 从神经网络映射到扩散策略，数据驱动的 retargeting 如何在精度和泛化性之间取得平衡。涵盖监督学习、生成模型、端到端策略学习三大路线。

---

## 1. 问题重新 formulation

传统方法（Rule-based / Optimization）的瓶颈：

| 瓶颈 | 说明 |
|------|------|
| 尺寸差异 | 人手和机器人手指长度比不同，直接映射失效 |
| 结构差异 | 自由度不匹配（人手 20+ DOF vs O10 10 DOF） |
| 噪声敏感 | 视觉 landmarks 噪声导致输出抖动 |
| 泛化性差 | 换操作者/机器人需重新调参或建模型 |

**学习-based 的核心思想**：用大量配对数据训练一个从人手特征到机器人关节的映射函数 $f_\theta$，使得：

$$\boldsymbol{\theta}_{\text{robot}} = f_\theta(\mathbf{x}_{\text{human}})$$

其中 $\mathbf{x}_{\text{human}}$ 可以是人手关节角、landmarks、或图像。

---

## 2. 监督学习路线

### 2.1 网络架构设计

输入输出选择：

| 输入 | 输出 | 适用场景 |
|------|------|---------|
| 21×3 landmarks | 关节角 | 已知 3D 追踪 |
| 关节角 (20-D) | 关节角 (10-D) | 有动作捕捉手套 |
| RGB 图像 | 关节角 | 端到端 |

**推荐架构**（以 landmarks → 关节角为例）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkToJointNet(nn.Module):
    """
    21点 landmarks → 机器人关节角

    架构设计原则：
    1. 先对每个手指独立编码（局部特征）
    2. 再全局融合（手指间耦合）
    3. 输出带关节限位的回归值
    """

    def __init__(self, n_robot_joints=10):
        super().__init__()
        if n_robot_joints != 10:
            raise ValueError("this teaching architecture assumes two outputs for each of five fingers")
        self.n_joints = n_robot_joints

        # 每个手指的局部编码器
        self.finger_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(15, 32),  # 5 points × 3 = 15 (wrist + 4 finger landmarks)
                nn.ReLU(),
                nn.Linear(32, 16),
            ) for _ in range(5)
        ])

        # 全局融合
        self.global_fusion = nn.Sequential(
            nn.Linear(5 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 输出头（每根手指 2 个关节）
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
                nn.Sigmoid(),  # 输出 [0, 1]
            ) for _ in range(5)
        ])

        # 关节限位（可学习或固定）
        self.register_buffer('joint_mins', torch.zeros(n_robot_joints))
        self.register_buffer('joint_maxs', torch.ones(n_robot_joints))

    def forward(self, landmarks):
        """
        Args:
            landmarks: [B, 21, 3]
        Returns:
            joints: [B, 10] 映射到 joint_mins / joint_maxs 指定范围
        """
        batch_size = landmarks.shape[0]

        # 提取每根手指的 landmarks（包含 wrist 作为参考）
        finger_groups = [
            [0, 1, 2, 3, 4],    # thumb (wrist + 4)
            [0, 5, 6, 7, 8],    # index
            [0, 9, 10, 11, 12], # middle
            [0, 13, 14, 15, 16],# ring
            [0, 17, 18, 19, 20],# pinky
        ]

        finger_features = []
        for i, indices in enumerate(finger_groups):
            finger_lm = landmarks[:, indices, :].reshape(batch_size, -1)  # [B, 15]
            feat = self.finger_encoders[i](finger_lm)
            finger_features.append(feat)

        # 全局融合
        global_feat = torch.cat(finger_features, dim=-1)  # [B, 80]
        fused = self.global_fusion(global_feat)  # [B, 64]

        # 输出每根手指的关节角
        joints_list = []
        for head in self.output_heads:
            joints_list.append(head(fused))

        joints = torch.cat(joints_list, dim=-1)  # [B, 10]

        # 映射到实际关节范围
        joints = self.joint_mins + joints * (self.joint_maxs - self.joint_mins)

        return joints
```

### 2.2 损失函数设计

**多目标损失**：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{joint}} + \lambda_2 \mathcal{L}_{\text{task}} + \lambda_3 \mathcal{L}_{\text{smooth}} + \lambda_4 \mathcal{L}_{\text{limit}}$$

```python
def multi_objective_loss(
    pred_joints,
    target_joints,
    robot_model,
    prev_joints=None,
    joint_min=None,
    joint_max=None,
):
    """
    多目标损失函数
    """
    # 1. 关节空间 L1 损失
    loss_joint = F.l1_loss(pred_joints, target_joints)

    # 2. 任务空间损失（fingertip 位置对齐）
    pred_tips = robot_model.batch_forward_kinematics(pred_joints)
    target_tips = robot_model.batch_forward_kinematics(target_joints)
    loss_task = F.mse_loss(pred_tips, target_tips)

    # 3. 时域平滑损失（防止抖动）
    if prev_joints is not None:
        loss_smooth = F.mse_loss(pred_joints, prev_joints)
    else:
        loss_smooth = pred_joints.new_zeros(())

    # 4. 关节限位惩罚；调用方必须同时提供上下界
    if (joint_min is None) != (joint_max is None):
        raise ValueError("joint_min and joint_max must be provided together")
    if joint_min is None:
        loss_limit = pred_joints.new_zeros(())
    else:
        loss_limit = torch.relu(pred_joints - joint_max).sum() + \
                     torch.relu(joint_min - pred_joints).sum()

    total_loss = loss_joint + 0.5 * loss_task + 0.1 * loss_smooth + 10.0 * loss_limit
    return total_loss
```

### 2.3 训练数据生成

**伪标签策略**（无需真实遥操作数据）：

```python
def generate_pseudo_labels(robot_model, ik_solver, n_samples=10000):
    """
    使用 Vector Optimization 生成训练数据
    """
    dataset = []

    for _ in range(n_samples):
        # 随机生成人手姿态（在合理范围内）
        human_joints = sample_human_hand_pose()

        # 通过 FK 得到 fingertip 位置
        human_tips = human_hand_fk(human_joints)

        # 用 Vector Optimization 求解最优机器人关节
        robot_joints = ik_solver.solve(human_tips)

        # 保存配对数据
        dataset.append({
            'landmarks': human_joints_to_landmarks(human_joints),
            'robot_joints': robot_joints,
        })

    return dataset
```

---

## 3. 生成模型路线

### 3.1 CVAE：条件变分自编码器

**动机**：一个手势可能对应多个合理的机器人姿态（多模态）。VAE 可以建模这种分布。

```python
class HandRetargetingCVAE(nn.Module):
    """
    条件 VAE：输入人手 landmarks，输出机器人关节分布
    """

    def __init__(self, input_dim=63, latent_dim=16, output_dim=10):
        super().__init__()

        # 编码器：p(z | x, y)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # 解码器：p(y | x, z)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def encode(self, x, y):
        """编码到隐空间"""
        xy = torch.cat([x, y], dim=-1)
        h = self.encoder(xy)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        """从隐空间解码"""
        xz = torch.cat([x, z], dim=-1)
        return self.decoder(xz)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        y_recon = self.decode(x, z)
        return y_recon, mu, logvar
```

**损失函数**：

$$\mathcal{L} = \|y - \hat{y}\|^2 + \beta \cdot D_{KL}(q(z|x,y) \| p(z))$$

### 3.2 扩散策略（Diffusion Policy）

**核心思想**：将 retargeting 建模为条件去噪扩散过程。

```python
class DiffusionRetargeting(nn.Module):
    """
    扩散模型用于 retargeting

    前向：逐步加噪机器人关节
    反向：以人手 landmarks 为条件，逐步去噪得到机器人关节
    """

    def __init__(self, n_joints=10, n_timesteps=100):
        super().__init__()
        self.n_joints = n_joints
        self.n_timesteps = n_timesteps

        # 条件编码器（人手 landmarks）
        self.condition_encoder = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # U-Net 去噪网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(n_joints + 64 + 1, 128),  # joints + condition + timestep
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_joints),
        )

    def forward_diffusion(self, x_0, t):
        """前向扩散：加噪"""
        alpha_bar = self.alpha_bars[t]
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def reverse_diffusion(self, x_t, condition, t):
        """反向去噪"""
        cond_feat = self.condition_encoder(condition)
        t_emb = self.timestep_embedding(t)

        input_feat = torch.cat([x_t, cond_feat, t_emb], dim=-1)
        predicted_noise = self.noise_predictor(input_feat)

        return predicted_noise
```

**为什么可以考虑 Diffusion**：
- 建模多模态分布（一个手势 → 多种合理机器人姿态）
- 若把一段动作序列作为联合输出并加入时间条件，可显式学习时序相关性；本页的单步关节向量示例本身不保证轨迹平滑
- 可以融合多种条件（landmarks + 语义标签）

上面的 `DiffusionRetargeting` 仍是结构骨架：噪声日程 `alpha_bars`、`timestep_embedding`、训练目标和完整反向采样更新都需在具体实现中定义，不能把一次 `predicted_noise` 调用当成可运行的扩散策略。

---

## 4. 端到端策略学习

### 4.1 从图像直接到动作

跳过 landmarks 提取，直接从 RGB 图像预测机器人关节角：

```python
import math

from torchvision.models import resnet18


class SinusoidalTimeEncoding(nn.Module):
    """给 [B, T, D] 特征加入确定性的时间位置信息。"""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for paired sin/cos encoding")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.pe.shape[-1]:
            raise ValueError("x must have shape [B, T, d_model]")
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError("sequence is longer than max_len")
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype, device=x.device)


class ImageToHandPolicy(nn.Module):
    """
    端到端策略：RGB 图像 → 机器人关节角

    使用 ResNet 提取视觉特征 + Transformer 融合时序信息
    """

    def __init__(self, n_joints=10, max_seq_len=128):
        super().__init__()

        # 视觉编码器
        # weights=None 不会下载预训练权重；正式训练时应显式记录权重来源。
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        # 时序融合（处理视频序列）
        self.time_encoding = SinusoidalTimeEncoding(512, max_len=max_seq_len)
        self.temporal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=2
        )

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_joints),
            nn.Sigmoid(),
        )

    def forward(self, image_sequence):
        """
        Args:
            image_sequence: [B, T, 3, H, W]
        """
        if image_sequence.ndim != 5:
            raise ValueError("image_sequence must have shape [B, T, 3, H, W]")
        B, T, C, H, W = image_sequence.shape
        if C != 3:
            raise ValueError("expected three RGB channels")

        # 提取每帧特征
        images = image_sequence.reshape(B * T, C, H, W)
        features = self.backbone(images)  # [B*T, 512]
        features = features.reshape(B, T, 512)

        # TransformerEncoder 本身不自动知道帧序；先加入显式时间位置编码。
        features = self.time_encoding(features)
        fused = self.temporal_fusion(features)  # [B, T, 512]

        # 取最后一帧预测动作
        action = self.policy_head(fused[:, -1])  # [B, n_joints]

        return action
```

### 4.2 模仿学习（Behavioral Cloning）

```python
def train_bc_policy(dataset, robot_model, epochs=100):
    """
    行为克隆训练
    """
    policy = ImageToHandPolicy(n_joints=10)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for images, target_joints in dataloader:
            pred_joints = policy(images)

            # 关节空间损失
            loss_joint = F.l1_loss(pred_joints, target_joints)

            # 任务空间损失（通过 FK）
            pred_tips = robot_model.batch_fk(pred_joints)
            target_tips = robot_model.batch_fk(target_joints)
            loss_task = F.mse_loss(pred_tips, target_tips)

            loss = loss_joint + 0.5 * loss_task

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 5. 数据增强与域适应

### 5.1 训练时数据增强

```python
def augment_landmarks(landmarks):
    """
    对 landmarks 进行数据增强
    """
    # 随机旋转（绕手腕）
    angle = np.random.uniform(-0.1, 0.1)
    R = rotation_matrix_z(angle)
    landmarks_rotated = (R @ landmarks.T).T

    # 随机缩放
    scale = np.random.uniform(0.9, 1.1)
    landmarks_scaled = landmarks_rotated * scale

    # 加高斯噪声
    noise = np.random.normal(0, 0.01, landmarks.shape)
    landmarks_noisy = landmarks_scaled + noise

    return landmarks_noisy
```

### 5.2 Sim-to-Real 域适应

```python
class DomainAdversarialNetwork(nn.Module):
    """
    DANN：在仿真数据上训练，泛化到真实数据
    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = ...
        self.retargeting_head = ...
        self.domain_classifier = ...  # 区分仿真/真实

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)

        # 梯度反转层
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)

        joints = self.retargeting_head(features)
        return joints, domain_pred
```

---

## 6. 方法对比与选择

| 方法 | 数据需求 | 推理速度 | 多模态 | 泛化性 | 可解释性 |
|------|---------|---------|--------|--------|---------|
| MLP 监督 | 中等 | 极快 | 否 | 中 | 低 |
| CVAE | 中等 | 快 | 是 | 中 | 低 |
| Diffusion Policy | 大量 | 慢 | 是 | 好 | 低 |
| 端到端图像 | 大量 | 中等 | 否 | 差 | 极低 |
| BC + 数据增强 | 大量 | 中等 | 否 | 好 | 低 |

**工程建议**：
1. **快速原型**：MLP 监督学习 + 伪标签
2. **高精度**：CVAE + 多模态采样
3. **平滑动作**：Diffusion Policy + Action Chunking
4. **无 landmarks**：端到端图像策略（需要大量数据）
