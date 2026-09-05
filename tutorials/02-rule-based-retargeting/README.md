# Stage 2: Rule-based Retargeting

> 从人手 landmarks 到机器人关节角的直接映射方法，最简单实用的 retargeting 基线。

---

## 核心思想

将人手的关键点坐标直接转换为机器人关节角，通过：

1. **坐标合同检查**：先区分图像归一化坐标与米制 3D，再做手腕平移与掌面轴对齐
2. **角度计算**：从 landmarks 计算弯曲角和外展角
3. **比例映射**：将人手角度按比例映射到机器人角度
4. **关节限位**：裁剪到机器人关节范围

---

## Pipeline 结构示意

这是伪代码，不提供检测器、预处理、角度计算与设备实现，也未经过端到端验证。MediaPipe 的图像归一化点不能直接当米制点参与机器人 FK 误差；详见[坐标合同](../04-landmark-pipeline/README.md)。限位裁剪不是完整安全系统。

```python
# 1. 获取 landmarks（来自 MediaPipe）
landmarks = mediapipe_hands.detect(image)  # [21, 3]

# 2. 预处理（局部坐标 + 归一化 + 镜像）
landmarks_local = preprocess(landmarks, is_left=True)

# 3. 计算弯曲角
angles = compute_flexion_angles(landmarks_local)

# 4. 映射到机器人关节
robot_joints = angles * scale_factor

# 5. 限位裁剪
robot_joints = np.clip(robot_joints, joint_min, joint_max)

# 6. 先检查仿真适配所需的顺序、单位、执行器类型；此处不发送硬件命令
print(robot_joints)
```

---

## 运行示例

从仓库根目录运行下面的合成手势演示；它不是相机到真机的完整流程。

```bash
cd examples
python landmark_to_joint.py --hand right --gesture open
python landmark_to_joint.py --hand left --gesture fist
```

---

## 关键参数调优

以下数值仅为待校准的教学例子，不是任何型号的安全默认值；增加映射增益可能使关节更快饱和。

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `scale_factor` | 补偿 landmark→curl 衰减 | 1.60 |
| `normalization_denominator` | 归一化分母 | 0.95 |
| `ema_alpha` | 时域滤波系数 | 0.3 |

---

## 优缺点

**优点**：
- 实现简单，无需训练数据
- 计算结构简单；实际耗时需在目标机器上测量
- 容易检查映射规则，但稳定性还依赖检测噪声、校准和控制器

**缺点**：
- 泛化性差（换操作者/机器人需重新调参）
- 单纯角度映射不能保证不同尺寸手具有相同指尖位置；可校准缩放，但仍需测量误差
- 拇指校准一般
