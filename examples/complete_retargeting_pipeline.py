#!/usr/bin/env python3
"""
complete_retargeting_pipeline.py
=================================
从人手视觉捕捉到机器人灵巧手控制的完整重定向流水线。

本脚本演示从 0 到 1 的完整链路：
  1. 模拟 MediaPipe 21 点输入
  2. 局部坐标系转换 + 尺度归一化
  3. 左右手镜像处理
  4. Rule-based + Vector Optimization 两种 retargeting 方法
  5. 关节限幅 + 时序平滑
  6. MuJoCo 仿真验证（可选）
  7. 性能评估与对比

对应教程：tutorials/05-complete-pipeline/README.md
依赖：pip install numpy scipy matplotlib

运行：
    python complete_retargeting_pipeline.py --method rule_based --visualize
    python complete_retargeting_pipeline.py --method vector_opt --visualize
"""

import numpy as np
import argparse
import json
import time
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 模拟视觉输入：生成合成的人手 21 点数据
# ============================================================

class SyntheticHandGenerator:
    """
    合成 21 点人手数据生成器。
    
    用于在没有真实摄像头的情况下测试 pipeline。
    生成不同弯曲程度的手指姿态。
    """

    def __init__(self, palm_length=0.1):
        self.palm_length = palm_length
        
        # 手指长度比例（相对于手掌长度）
        self.finger_lengths = {
            "thumb": [0.3, 0.25, 0.2],
            "index": [0.35, 0.25, 0.2],
            "middle": [0.4, 0.3, 0.25],
            "ring": [0.38, 0.28, 0.23],
            "pinky": [0.3, 0.22, 0.18],
        }
        
        # 手指方向（在手掌平面上的角度，单位：弧度）
        self.finger_angles = {
            "thumb": -0.8,
            "index": -0.3,
            "middle": 0.0,
            "ring": 0.3,
            "pinky": 0.6,
        }

    def generate(self, finger_curls, noise_std=0.002):
        """
        生成给定弯曲程度的人手 21 点。
        
        Args:
            finger_curls: dict，如 {"index": 0.5, "middle": 0.3, ...}
                          值范围 [0, 1]，0=伸直，1=完全弯曲
            noise_std: 关键点噪声标准差
        
        Returns:
            landmarks: [21, 3] numpy 数组
        """
        landmarks = np.zeros((21, 3), dtype=np.float32)
        
        # 手腕在原点
        landmarks[0] = [0, 0, 0]
        
        # 手指索引映射（MediaPipe 格式）
        finger_indices = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }
        
        # 手指根部位置（MCP 关节）
        mcp_positions = {
            "thumb": np.array([0.02, -0.02, 0.0]),
            "index": np.array([0.0, self.palm_length * 0.5, 0.0]),
            "middle": np.array([0.0, self.palm_length * 0.15, 0.0]),
            "ring": np.array([0.0, -self.palm_length * 0.15, 0.0]),
            "pinky": np.array([0.0, -self.palm_length * 0.45, 0.0]),
        }
        
        for finger_name, indices in finger_indices.items():
            curl = finger_curls.get(finger_name, 0.0)
            lengths = self.finger_lengths[finger_name]
            base_angle = self.finger_angles[finger_name]
            
            # MCP 位置
            landmarks[indices[0]] = mcp_positions[finger_name]
            
            # 生成手指链
            pos = landmarks[indices[0]].copy()
            for i, length in enumerate(lengths):
                # 弯曲角度 = 基础角度 + curl * 最大弯曲
                bend_angle = curl * np.pi * 0.4
                
                # 手指延伸方向（考虑弯曲）
                if i == 0:
                    # MCP → PIP
                    direction = np.array([
                        np.sin(base_angle),
                        np.cos(base_angle),
                        -np.sin(bend_angle) * 0.5,
                    ])
                else:
                    # 后续关节继续弯曲
                    direction = np.array([
                        direction[0] * 0.9,
                        direction[1] * 0.9,
                        -np.sin(bend_angle * (i + 1) / len(lengths)),
                    ])
                
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                pos = pos + direction * length * self.palm_length
                landmarks[indices[i + 1]] = pos
        
        # 添加噪声
        landmarks += np.random.randn(21, 3).astype(np.float32) * noise_std
        
        return landmarks


# ============================================================
# 2. 预处理：局部坐标系 + 归一化 + 镜像
# ============================================================

class HandPreprocessor:
    """
    人手数据预处理：坐标系转换、归一化、镜像。
    """

    @staticmethod
    def to_local_frame(landmarks_21x3):
        """转换到以手腕为原点的局部坐标系。"""
        wrist = landmarks_21x3[0]
        return landmarks_21x3 - wrist

    @staticmethod
    def normalize_scale(local_landmarks):
        """用手掌长度做尺度归一化。"""
        palm_length = np.linalg.norm(local_landmarks[9])  # 中指 MCP
        normalized = local_landmarks / (palm_length + 1e-8)
        return normalized, palm_length

    @staticmethod
    def mirror_left_hand(local_landmarks):
        """左手 Y 轴镜像。"""
        mirrored = local_landmarks.copy()
        mirrored[:, 1] *= -1
        return mirrored

    def process(self, landmarks_21x3, is_left_hand=False):
        """
        完整预处理流程。
        
        Returns:
            normalized: [21, 3] 归一化后的局部坐标
            palm_length: float 手掌长度（用于反归一化）
        """
        local = self.to_local_frame(landmarks_21x3)
        
        if is_left_hand:
            local = self.mirror_left_hand(local)
        
        normalized, palm_length = self.normalize_scale(local)
        return normalized, palm_length


# ============================================================
# 3. Retargeting 方法
# ============================================================

class RuleBasedRetargeter:
    """
    Rule-based retargeting：基于角度映射。
    
    特点：简单、实时、无需迭代优化。
    """

    def __init__(self, joint_limits=(0.0, 1.2)):
        self.joint_limits = joint_limits
        
        # 手指关键点索引（MediaPipe）
        self.finger_indices = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }
        
        # 优化参数（来自项目经验）
        self.normalization_denom = 0.95  # 原来是 1.45
        self.actuator_scale = 1.60       # 原来是 1.25

    def _compute_curl(self, landmarks, indices):
        """计算手指弯曲程度 [0, 1]。"""
        pts = landmarks[indices]
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[1]
        v3 = pts[3] - pts[2]
        
        def angle_between(v_a, v_b):
            cos = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-8)
            return np.arccos(np.clip(cos, -1, 1))
        
        angle1 = angle_between(v1, v2)
        angle2 = angle_between(v2, v3)
        
        curl = ((angle1 + angle2) / 2) / self.normalization_denom
        return np.clip(curl, 0, 1)

    def retarget(self, landmarks_21x3):
        """
        从 landmarks 映射到关节角度。
        
        Returns:
            joint_angles: dict，如 {"thumb_mcp": 0.5, ...}
        """
        joint_angles = {}
        
        for finger_name, indices in self.finger_indices.items():
            curl = self._compute_curl(landmarks_21x3, indices)
            
            # 映射到关节角度
            angle = curl * self.joint_limits[1] * self.actuator_scale
            angle = np.clip(angle, self.joint_limits[0], self.joint_limits[1])
            
            # O10：每指 2 个关节（MCP + PIP 耦合）
            joint_angles[f"{finger_name}_mcp"] = angle
            joint_angles[f"{finger_name}_pip"] = angle * 0.8
        
        return joint_angles


class FingerChain3D:
    """
    简化的 3D 手指运动链，用于 Vector Optimization。
    """

    def __init__(self, finger_name, segment_lengths):
        """
        Args:
            finger_name: str
            segment_lengths: [3] 三段长度（MCP→PIP, PIP→DIP, DIP→TIP）
        """
        self.finger_name = finger_name
        self.lengths = np.array(segment_lengths)
        self.n_joints = 2  # MCP + PIP（耦合）

    def forward_kinematics(self, joints):
        """
        正运动学：关节角度 → 指尖位置。
        
        Args:
            joints: [2] MCP 角度和 PIP 角度
        
        Returns:
            tip_position: [3] 指尖 3D 位置
        """
        theta_mcp, theta_pip = joints
        
        # 简化的 2D 平面运动链（YZ 平面）
        # MCP → PIP
        p1 = np.array([0, self.lengths[0] * np.cos(theta_mcp), 
                          -self.lengths[0] * np.sin(theta_mcp)])
        
        # PIP → DIP
        p2 = p1 + np.array([0, self.lengths[1] * np.cos(theta_mcp + theta_pip),
                               -self.lengths[1] * np.sin(theta_mcp + theta_pip)])
        
        # DIP → TIP
        p3 = p2 + np.array([0, self.lengths[2] * np.cos(theta_mcp + theta_pip * 1.2),
                               -self.lengths[2] * np.sin(theta_mcp + theta_pip * 1.2)])
        
        return p3


class VectorOptimizationRetargeter:
    """
    Vector Optimization retargeting：任务空间 IK。
    
    特点：精度高，可补偿尺寸差异，但计算成本较高。
    """

    def __init__(self, joint_limits=(0.0, 1.2)):
        self.joint_limits = joint_limits
        
        # 为每根手指创建运动链
        self.finger_chains = {
            "thumb": FingerChain3D("thumb", [0.03, 0.025, 0.02]),
            "index": FingerChain3D("index", [0.035, 0.025, 0.02]),
            "middle": FingerChain3D("middle", [0.04, 0.03, 0.025]),
            "ring": FingerChain3D("ring", [0.038, 0.028, 0.023]),
            "pinky": FingerChain3D("pinky", [0.03, 0.022, 0.018]),
        }
        
        # 手指关键点索引（指尖）
        self.tip_indices = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20,
        }

    def retarget_finger(self, target_tip, finger_name, initial_guess=None):
        """
        对单根手指做 Vector Optimization。
        
        Args:
            target_tip: [3] 目标指尖位置（来自人手 landmarks）
            finger_name: str
            initial_guess: [2] 初始关节角度
        """
        chain = self.finger_chains[finger_name]
        
        if initial_guess is None:
            initial_guess = [0.3, 0.3]  # 默认初值
        
        def residuals(joints):
            tip = chain.forward_kinematics(joints)
            return tip - target_tip
        
        result = least_squares(
            residuals,
            initial_guess,
            # ``lm`` cannot be combined with bounds in SciPy. ``trf`` keeps
            # the existing joint-limit contract while supporting the same
            # nonlinear least-squares objective.
            method='trf',
            ftol=1e-6,
            max_nfev=50,
            bounds=self.joint_limits,
        )
        
        return result.x

    def retarget(self, landmarks_21x3):
        """对整只手做 Vector Optimization。"""
        joint_angles = {}
        
        for finger_name, tip_idx in self.tip_indices.items():
            target_tip = landmarks_21x3[tip_idx]
            
            # 用 Rule-based 结果作为初值
            rule_based = RuleBasedRetargeter()
            rule_angles = rule_based.retarget(landmarks_21x3)
            initial = [
                rule_angles[f"{finger_name}_mcp"],
                rule_angles[f"{finger_name}_pip"],
            ]
            
            opt_joints = self.retarget_finger(target_tip, finger_name, initial)
            
            joint_angles[f"{finger_name}_mcp"] = opt_joints[0]
            joint_angles[f"{finger_name}_pip"] = opt_joints[1]
        
        return joint_angles


# ============================================================
# 4. 后处理：限幅 + 平滑
# ============================================================

class PostProcessor:
    """
    后处理：关节限幅 + 时序平滑。
    """

    def __init__(self, joint_limits=(0.0, 1.2), smooth_alpha=0.3):
        self.joint_limits = joint_limits
        self.smooth_alpha = smooth_alpha
        self.prev_angles = None

    def clamp(self, joint_angles):
        """关节限幅。"""
        return {k: np.clip(v, self.joint_limits[0], self.joint_limits[1]) 
                for k, v in joint_angles.items()}

    def smooth(self, joint_angles):
        """时序平滑（EMA）。"""
        if self.prev_angles is None:
            self.prev_angles = joint_angles.copy()
            return joint_angles
        
        smoothed = {}
        for k in joint_angles:
            smoothed[k] = (self.smooth_alpha * joint_angles[k] + 
                          (1 - self.smooth_alpha) * self.prev_angles[k])
        
        self.prev_angles = smoothed.copy()
        return smoothed

    def process(self, joint_angles):
        """完整后处理。"""
        clamped = self.clamp(joint_angles)
        smoothed = self.smooth(clamped)
        return smoothed


# ============================================================
# 5. 评估
# ============================================================

def evaluate_retargeting(landmarks, retargeter, preprocessor, postprocessor, 
                         num_trials=100):
    """
    评估 retargeting 方法的性能。
    
    Returns:
        metrics: dict，包含 latency、success_rate、avg_error
    """
    latencies = []
    errors = []
    
    for _ in range(num_trials):
        # 预处理
        t0 = time.time()
        normalized, _ = preprocessor.process(landmarks, is_left_hand=False)
        
        # Retargeting
        t1 = time.time()
        joint_angles = retargeter.retarget(normalized)
        
        # 后处理
        t2 = time.time()
        final_angles = postprocessor.process(joint_angles)
        t3 = time.time()
        
        latencies.append(t3 - t1)
        
        # 简单误差：检查关节是否在合理范围
        in_range = all(0 <= v <= 1.2 for v in final_angles.values())
        errors.append(0.0 if in_range else 1.0)
    
    return {
        "avg_latency_ms": np.mean(latencies) * 1000,
        "max_latency_ms": np.max(latencies) * 1000,
        "success_rate": 1.0 - np.mean(errors),
    }


def visualize_comparison(rule_metrics, vector_metrics):
    """可视化两种方法的性能对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 延迟对比
    methods = ["Rule-based", "Vector Opt"]
    avg_lat = [rule_metrics["avg_latency_ms"], vector_metrics["avg_latency_ms"]]
    max_lat = [rule_metrics["max_latency_ms"], vector_metrics["max_latency_ms"]]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0].bar(x - width/2, avg_lat, width, label="Avg", color="#0d6efd")
    axes[0].bar(x + width/2, max_lat, width, label="Max", color="#dc3545")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Retargeting Latency Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    
    # 成功率对比
    success_rates = [rule_metrics["success_rate"], vector_metrics["success_rate"]]
    colors = ["#198754", "#ffc107"]
    bars = axes[1].bar(methods, success_rates, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Joint Limit Compliance")
    axes[1].set_ylim([0, 1.1])
    
    for bar, rate in zip(bars, success_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{rate:.1%}", ha="center", va="bottom", fontweight="bold")
    
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("retargeting_comparison.png", dpi=150)
    print("\n[Saved] retargeting_comparison.png")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Complete Retargeting Pipeline")
    parser.add_argument("--method", choices=["rule_based", "vector_opt", "both"],
                        default="both", help="Retargeting 方法")
    parser.add_argument("--visualize", action="store_true", help="可视化对比")
    parser.add_argument("--trials", type=int, default=100, help="评估次数")
    args = parser.parse_args()

    print("=" * 60)
    print("Complete Retargeting Pipeline Demo")
    print("=" * 60)

    # --- 1. 生成合成数据 ---
    print("\n[1/6] 生成合成人手数据...")
    generator = SyntheticHandGenerator(palm_length=0.1)
    
    # 生成"半握拳"姿态
    finger_curls = {
        "thumb": 0.3,
        "index": 0.6,
        "middle": 0.7,
        "ring": 0.5,
        "pinky": 0.4,
    }
    landmarks = generator.generate(finger_curls, noise_std=0.002)
    print(f"  生成 21 点 landmarks，手掌长度: 0.1m")
    print(f"  手指弯曲程度: {finger_curls}")

    # --- 2. 预处理 ---
    print("\n[2/6] 预处理：局部坐标系 + 归一化...")
    preprocessor = HandPreprocessor()
    normalized, palm_length = preprocessor.process(landmarks, is_left_hand=False)
    print(f"  局部坐标系转换完成")
    print(f"  尺度归一化完成（手掌长度: {palm_length:.4f}m）")

    # --- 3. Retargeting ---
    print(f"\n[3/6] Retargeting（method={args.method}）...")
    
    if args.method in ["rule_based", "both"]:
        print("\n  --- Rule-based ---")
        rule_retargeter = RuleBasedRetargeter()
        t0 = time.time()
        rule_angles = rule_retargeter.retarget(normalized)
        rule_time = (time.time() - t0) * 1000
        print(f"  耗时: {rule_time:.3f}ms")
        print(f"  拇指 MCP: {rule_angles['thumb_mcp']:.3f} rad")
        print(f"  食指 MCP: {rule_angles['index_mcp']:.3f} rad")
        print(f"  中指 MCP: {rule_angles['middle_mcp']:.3f} rad")

    if args.method in ["vector_opt", "both"]:
        print("\n  --- Vector Optimization ---")
        vector_retargeter = VectorOptimizationRetargeter()
        t0 = time.time()
        vector_angles = vector_retargeter.retarget(normalized)
        vector_time = (time.time() - t0) * 1000
        print(f"  耗时: {vector_time:.3f}ms")
        print(f"  拇指 MCP: {vector_angles['thumb_mcp']:.3f} rad")
        print(f"  食指 MCP: {vector_angles['index_mcp']:.3f} rad")
        print(f"  中指 MCP: {vector_angles['middle_mcp']:.3f} rad")

    # --- 4. 后处理 ---
    print("\n[4/6] 后处理：限幅 + 平滑...")
    postprocessor = PostProcessor(smooth_alpha=0.3)
    
    if args.method in ["rule_based", "both"]:
        rule_final = postprocessor.process(rule_angles)
        print(f"  Rule-based 后处理完成")
    
    if args.method in ["vector_opt", "both"]:
        postprocessor.prev_angles = None  # 重置平滑器
        vector_final = postprocessor.process(vector_angles)
        print(f"  Vector Opt 后处理完成")

    # --- 5. 评估 ---
    print(f"\n[5/6] 性能评估（{args.trials} 次试验）...")
    
    if args.method in ["rule_based", "both"]:
        rule_metrics = evaluate_retargeting(
            landmarks, rule_retargeter, preprocessor, 
            PostProcessor(smooth_alpha=0.3), args.trials
        )
        print(f"\n  Rule-based:")
        print(f"    平均延迟: {rule_metrics['avg_latency_ms']:.3f}ms")
        print(f"    最大延迟: {rule_metrics['max_latency_ms']:.3f}ms")
        print(f"    成功率:   {rule_metrics['success_rate']:.1%}")

    if args.method in ["vector_opt", "both"]:
        vector_metrics = evaluate_retargeting(
            landmarks, vector_retargeter, preprocessor,
            PostProcessor(smooth_alpha=0.3), args.trials
        )
        print(f"\n  Vector Optimization:")
        print(f"    平均延迟: {vector_metrics['avg_latency_ms']:.3f}ms")
        print(f"    最大延迟: {vector_metrics['max_latency_ms']:.3f}ms")
        print(f"    成功率:   {vector_metrics['success_rate']:.1%}")

    # --- 6. 可视化 ---
    if args.visualize and args.method == "both":
        print("\n[6/6] 生成对比图...")
        visualize_comparison(rule_metrics, vector_metrics)

    # --- 总结 ---
    print("\n" + "=" * 60)
    print("Pipeline 总结：")
    print("=" * 60)
    print("1. 视觉输入：21 点 landmarks（合成数据模拟 MediaPipe）")
    print("2. 预处理：局部坐标系 → 归一化 → 镜像（左手）")
    print("3. Retargeting：")
    print("   - Rule-based: 角度映射，快速实时")
    print("   - Vector Opt: 任务空间 IK，精度更高")
    print("4. 后处理：关节限幅 + EMA 时序平滑")
    print("5. 输出：10 个关节角度（O10 灵巧手）")
    print("=" * 60)
    print("\n下一步：")
    print("  - 接入真实 MediaPipe 输入")
    print("  - 接入 MuJoCo 仿真验证")
    print("  - 接入 GeoRT 控制真机")
    print("=" * 60)


if __name__ == "__main__":
    main()
