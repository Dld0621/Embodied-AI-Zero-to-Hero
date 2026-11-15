# Stage 2: 动作表示

> 掌握机器人动作空间的各种数学表示，理解 Action Chunking。

---

## 学习目标

完成本阶段后，你应该能够：

1. 实现正向运动学（FK）和逆向运动学（IK）
2. 理解关节角度、末端位姿、增量动作的区别
3. 实现 Action Chunking
4. 理解不同表示方式的优缺点

---

## 2.1 正向运动学（FK）

已知关节角度，计算末端执行器位姿。

### 2-DOF 平面机械臂示例

```python
import numpy as np

def fk_2dof(q1, q2, l1=1.0, l2=1.0):
    """
    2-DOF 平面机械臂的正向运动学。

    Args:
        q1, q2: 关节角度（弧度）
        l1, l2: 连杆长度

    Returns:
        (x, y): 末端执行器位置
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, y])

# 示例
q1, q2 = np.pi/4, np.pi/4
end_pos = fk_2dof(q1, q2)
print(f"Joint angles: q1={q1:.3f}, q2={q2:.3f}")
print(f"End-effector position: ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
```

### 6-DOF 机械臂（使用变换矩阵）

```python
def rotation_matrix(axis, angle):
    """绕指定轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis == 'y':
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    elif axis == 'z':
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def fk_6dof(dh_params, joint_angles):
    """
    使用 DH 参数法计算 6-DOF 机械臂 FK。
    dh_params: list of [a, alpha, d, theta_offset]
    限定：标准 DH、全部为转动关节；移动关节应更新 d，而不是 theta。
    """
    if len(dh_params) != 6 or len(joint_angles) != 6:
        raise ValueError("本示例需要恰好 6 组 DH 参数和 6 个关节角")
    T = np.eye(4)
    for (a, alpha, d, theta_off), q in zip(dh_params, joint_angles):
        theta = theta_off + q
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        A_i = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
        T = T @ A_i

    return T  # 4x4 齐次变换矩阵
```

---

## 2.2 逆向运动学（IK）

已知末端位姿，求解关节角度。

### 数值解法（Jacobian 迭代）

```python
def ik_jacobian(target_xy, initial_guess, l1=1.0, l2=1.0, max_iter=100, tol=1e-4):
    """
    使用 Jacobian 迭代法求解 2-DOF IK。

    Args:
        target_xy: 目标位置 [x, y]
        initial_guess: 初始关节角度 [q1, q2]

    Returns:
        q: 求解的关节角度
    """
    q = np.array(initial_guess, dtype=float)
    target = np.array(target_xy, dtype=float)

    for i in range(max_iter):
        # 当前末端位置
        current = fk_2dof(q[0], q[1], l1, l2)
        error = target - current

        if np.linalg.norm(error) < tol:
            break

        # 计算 Jacobian
        q1, q2 = q[0], q[1]
        J = np.array([
            [-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
            [ l1*np.cos(q1) + l2*np.cos(q1+q2),  l2*np.cos(q1+q2)]
        ])

        # 伪逆更新
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ error
        q += 0.5 * dq  # 步长 0.5

    return q

# 示例
target = np.array([1.2, 0.8])
q_sol = ik_jacobian(target, [0.5, 0.5])
print(f"Target: {target}")
print(f"IK solution: q1={q_sol[0]:.3f}, q2={q_sol[1]:.3f}")
print(f"Verification: {fk_2dof(q_sol[0], q_sol[1])}")
assert np.linalg.norm(fk_2dof(*q_sol) - target) < 1e-4
```

这个教学求解器没有关节限位、碰撞约束或失败状态；达到最大迭代次数后返回的角度未必是解。使用时必须像上面一样验算残差；不可达目标、奇异位形或错误初值都可能导致不收敛。

---

## 2.3 动作表示对比

| 表示方式 | 维度 | 优点 | 缺点 | 适用场景 |
|---------|------|------|------|---------|
| **关节位置** | N（独立关节坐标数） | 适配位置接口较直接 | 需核对关节顺序、单位、限位 | 单一机器人平台 |
| **末端位姿** | Euler 位姿 6；四元数位姿 7；夹爪另加 1 | 可表达任务空间目标 | 仍依赖坐标系、IK 与工作空间 | 末端目标控制 |
| **增量 delta** | 例如平移 3 + 旋转增量 3 + 夹爪 1 | 表达相对当前状态的小步变化 | 仍依赖参考系、组合顺序与控制周期 | 经接口验证的闭环控制 |
| **关节速度** | N | 指定变化率 | 平滑性仍需加速度约束；仅位置接口时需积分 | 速度控制接口 |

位姿有 6 个物理自由度，但四元数用 4 个受单位范数约束的数编码旋转，所以“自由度数”不等于“数组长度”。增量也不是坐标系不变的：换基后平移向量需旋转，旋转增量还必须区分左乘与右乘。

### 选择建议

- **单机器人、固定场景** → 关节角度
- **跨平台策略（Octo, OpenVLA）** → 末端位姿或增量
- **选择控制频率** → 先匹配控制器合同；频率本身不能决定使用增量还是绝对目标
- **需要目标-reaching** → 末端位姿

---

## 2.4 Action Chunking

一次预测未来多步动作；如果执行多步后才重规划，可以减少推理次数。下面只有形状演示，随机初始化的输出不是有效控制策略。

```python
import torch
import torch.nn as nn

class ActionChunker(nn.Module):
    """
    简单的 Action Chunking 模型。
    输入当前观察，输出未来 T 步的动作序列。
    """
    def __init__(self, obs_dim, action_dim, chunk_size=8, hidden_dim=128):
        super().__init__()
        self.chunk_size = chunk_size

        # 观察编码
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # 序列生成（GRU）
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 动作头
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        """
        Args:
            obs: [B, obs_dim] 当前观察
        Returns:
            actions: [B, T, action_dim] 未来 T 步动作
        """
        B = obs.shape[0]

        # 编码观察
        z = self.obs_encoder(obs)  # [B, hidden_dim]

        # 重复 T 次作为 decoder 输入
        z = z.unsqueeze(1).repeat(1, self.chunk_size, 1)  # [B, T, hidden_dim]

        # 生成序列
        h, _ = self.decoder(z)  # [B, T, hidden_dim]

        # 映射到动作
        actions = self.action_head(h)  # [B, T, action_dim]

        return actions

# 示例
chunker = ActionChunker(obs_dim=512, action_dim=7, chunk_size=8)
obs = torch.randn(4, 512)  # batch_size=4
actions = chunker(obs)
print(f"Output shape: {actions.shape}")  # [4, 8, 7]
```

### Chunking 策略

下面是**假设控制频率为 20 Hz** 的调度例子，不是各论文的固定配置；推理耗时暂忽略。

| 调度方式 | 预测长度 T | 每次执行 K | 重新推理频率 |
|------|---------------|--------------|---------|
| 每步重规划 | 4 | 1 | 20 Hz |
| 分块执行 | 4 | 4 | 5 Hz |
| 长预测、短执行 | 16 | 2 | 10 Hz |

T、K、控制频率与是否进行时间集成是不同参数。ACT 的重叠预测可做 temporal ensembling，不能简单描述成“预测 100 步就总是执行全部”。[ACT 论文](https://arxiv.org/abs/2304.13705)

**关键权衡**：
- T 越大 → 一次推理覆盖时间越长 → 但预测误差随时间累积
- K 越小 → 重新推理越频繁 → 响应越快但计算开销大

---

## 验证检查点

- [ ] 能实现 2-DOF 机械臂的 FK 和数值 IK
- [ ] 能解释关节角度 vs 末端位姿 vs 增量动作的优缺点
- [ ] 能实现一个预测未来 8 步动作的 Action Chunker
- [ ] 理解 Action Chunking 中 T 和 K 的权衡

---

## 延伸阅读

- [Modern Robotics: Mechanics, Planning, and Control](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — 机器人学经典教材
- [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137) — 连续动作生成
