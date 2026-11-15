# Stage 4: 21 点 Landmark 完整 Pipeline

> 从视觉捕捉到机器人控制的完整数据流，包括 MediaPipe 集成、坐标转换、Retargeting 和 MuJoCo 仿真。

> 边界：这是集成结构教程，片段之间缺少标定、模型文件、完整错误处理与控制适配器，不是已验证的端到端程序。本次审查未开启相机、网络控制或真机。下面 `mp.solutions.hands` 是 legacy API，需使用仍提供该接口的明确版本；新项目可按官方 Tasks API 改写，二者返回类型不能混用。

---

## 系统架构

```
摄像头 → MediaPipe → 21点坐标 → 预处理 → Retargeting → MuJoCo/真实机器人
```

---

## MediaPipe 集成

必须先区分两种输出：图像 landmarks 的 x/y 按图像宽/高归一化，z 以手腕深度为参考且尺度近似 x；world landmarks 才是米制、原点在手部几何中心的估计。后者也不是已标定的机器人/相机世界位姿，仍需坐标与尺度校准。不能把前者直接送入单位为米的 FK 误差函数。[官方输出定义](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results)

```python
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 提取图像归一化点；这里不是米制 3D 坐标
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)

            # 后续处理...

cap.release()
hands.close()
```

---

## 双手系统

以下骨架依赖未在此定义的 `extract_landmarks`。legacy Hands 的左右手判断假设自拍镜像输入，须结合实际采集/显示翻转方式做左右手标定；不能只看显示屏上的方向。

```python
def process_both_hands(results):
    """处理左右手"""
    left_landmarks = None
    right_landmarks = None

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return left_landmarks, right_landmarks

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # 判断左右手
        handedness = results.multi_handedness[idx].classification[0].label
        is_left = (handedness == "Left")

        landmarks = extract_landmarks(hand_landmarks)

        if is_left:
            left_landmarks = landmarks
        else:
            right_landmarks = landmarks

    return left_landmarks, right_landmarks
```

---

## UDP 数据传输

这里只演示本机消息格式，不是可靠控制协议。部署前还需帧时间戳、序列号、坐标系/单位、丢包与陈旧数据拒绝；把两只手放进同一个包并不能校正采集时间差。

```python
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_hand_data(left_landmarks, right_landmarks, addr=("127.0.0.1", 9000)):
    """发送双手 landmarks 到仿真/控制端"""
    packet = {
        "left_landmarks": left_landmarks.tolist() if left_landmarks is not None else None,
        "right_landmarks": right_landmarks.tolist() if right_landmarks is not None else None,
    }
    sock.sendto(json.dumps(packet).encode(), addr)
```

---

## MuJoCo 仿真集成

以下是**待接入模型的接口骨架**，`o10_hand.xml` 是占位文件。只有确认每个 `ctrl[i]` 对应正确顺序的角位置执行器、传动比例和限值时，才能把角度写入 `ctrl`；力矩/速度执行器不能用同样解释。可先看[执行器原理](../../docs/foundations/08-control-basics.md)。

```python
import mujoco

# 加载模型
model = mujoco.MjModel.from_xml_path("o10_hand.xml")
data = mujoco.MjData(model)

def set_hand_position(joint_angles):
    """仅适用于已核对合同的角位置执行器；本片段不能直接接真机。"""
    for i, angle in enumerate(joint_angles):
        data.ctrl[i] = angle

    mujoco.mj_step(model, data)
```

---

## 完整 Pipeline 代码

可从 [`examples/complete_retargeting_pipeline.py`](../../examples/complete_retargeting_pipeline.py) 阅读教学实现；合成手势、相机接入、物理闭环和真机部署是不同验证层级，不能因脚本存在就一并标为完成。
