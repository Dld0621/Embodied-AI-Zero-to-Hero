# Stage 4: 21 点 Landmark 完整 Pipeline

> 从视觉捕捉到机器人控制的完整数据流，包括 MediaPipe 集成、坐标转换、Retargeting 和 MuJoCo 仿真。

---

## 系统架构

```
摄像头 → MediaPipe → 21点坐标 → 预处理 → Retargeting → MuJoCo/真实机器人
```

---

## MediaPipe 集成

```python
import mediapipe as mp
import cv2

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
            # 提取 21 点坐标
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)

            # 后续处理...
```

---

## 双手系统

```python
def process_both_hands(results):
    """处理左右手"""
    left_landmarks = None
    right_landmarks = None

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

```python
import mujoco

# 加载模型
model = mujoco.MjModel.from_xml_path("o10_hand.xml")
data = mujoco.MjData(model)

def set_hand_position(joint_angles):
    """设置灵巧手关节角"""
    for i, angle in enumerate(joint_angles):
        data.ctrl[i] = angle

    mujoco.mj_step(model, data)
```

---

## 完整 Pipeline 代码

详见 `examples/` 目录下的完整示例。
