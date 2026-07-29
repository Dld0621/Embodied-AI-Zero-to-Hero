"""
RL Benchmark Runner: 三 Seed SAC+HER 实验编排
===============================================
顺序运行 3 个随机种子的训练 + 评估 + 绘图，最后汇总结果。

Usage:
    # 完整 benchmark（train + eval + plot + aggregate）
    python scripts/run_rl_benchmark.py

    # 仅评估已有模型（跳过训练）
    python scripts/run_rl_benchmark.py --skip-train

    # 自定义环境
    python scripts/run_rl_benchmark.py --env HandManipulateBlock-v1 --timesteps 200000

    # 指定结果目录
    python scripts/run_rl_benchmark.py --results-dir D:\rl_benchmark_results
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# 默认配置
DEFAULT_ENV = "HandReach-v1"
DEFAULT_TIMESTEPS = 100_000
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_RESULTS_DIR = Path("results/rl")


def run_command(cmd, cwd=None, desc=""):
    """运行子命令并实时打印输出。"""
    if desc:
        print(f"\n{'='*70}")
        print(f" {desc}")
        print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        print(line)

    process.wait()
    if process.returncode != 0:
        print(f"[Error] Command failed with exit code {process.returncode}")
        return False, "\n".join(output_lines)
    return True, "\n".join(output_lines)


def get_auto_paths(seed, env_id, project_root):
    """根据 rl_demo.py 的自动命名规则生成路径。"""
    env_short = env_id.replace("-v1", "").replace("-v0", "").lower()
    # 与 rl_demo.py 保持一致：HandReach/HandManipulate 等 goal-conditioned 环境使用 sac_her
    is_goal_env = "hand" in env_short and env_short != "pendulum"
    algo_suffix = "sac_her" if is_goal_env else "sac"
    model_name = f"{env_short}_{algo_suffix}_seed{seed}"
    log_dir = project_root / "results" / "rl" / f"{env_short}_{algo_suffix}" / f"seed_{seed}"
    model_file = project_root / "examples" / f"{model_name}.zip"
    return model_name, log_dir, model_file


def get_legacy_paths(seed, results_dir):
    """旧版命名（兼容已训练的 seed）。"""
    model_name = f"shadow_hand_reach_seed{seed}"
    log_dir = Path(results_dir) / f"seed_{seed}"
    return model_name, log_dir


def check_train_complete(log_dir, model_file):
    """检查该 seed 的训练是否已完成。"""
    log_dir = Path(log_dir)
    monitor_file = log_dir / "monitor.csv"
    config_file = log_dir / "train_config.json"
    return model_file.exists() and monitor_file.exists() and config_file.exists()


def train_seed(seed, env, timesteps, results_dir, project_root):
    """训练单个 seed。"""
    model_name, log_dir, model_file = get_auto_paths(seed, env, project_root)

    if check_train_complete(log_dir, model_file):
        print(f"\n[Skip] seed={seed} 训练已完成（检测到模型文件）")
        return True

    # rl_demo.py 在 --model-name 和 --log-dir 不传时会自动按 env+seed 生成
    # 这里显式传入以保持脚本间一致
    cmd = [
        sys.executable,
        "rl_demo.py",
        "--mode", "train",
        "--env", env,
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--log-dir", str(log_dir),
        "--model-name", model_name,
    ]

    success, _ = run_command(cmd, cwd=project_root / "examples", desc=f"Training seed={seed}")
    return success


def eval_seed(seed, env, results_dir, project_root):
    """评估单个 seed。优先使用新版自动命名，回退到旧版。"""
    model_name, log_dir, model_file = get_auto_paths(seed, env, project_root)
    legacy_model_name, legacy_log_dir = get_legacy_paths(seed, results_dir)

    # 优先检查新版路径，其次旧版
    if check_train_complete(log_dir, model_file):
        output_path = log_dir / "eval_detail"
        best_model = log_dir / "best_model" / "best_model.zip"
        model_path = str(best_model) if best_model.exists() else str(model_file)
    elif check_train_complete(legacy_log_dir, project_root / "examples" / f"{legacy_model_name}.zip"):
        log_dir = legacy_log_dir
        output_path = log_dir / "eval_detail"
        best_model = log_dir / "best_model" / "best_model.zip"
        model_path = str(best_model) if best_model.exists() else legacy_model_name
    else:
        print(f"[Error] seed={seed} 未找到训练完成的模型（新版或旧版路径）")
        return False

    cmd = [
        sys.executable,
        "rl_demo.py",
        "--mode", "eval",
        "--env", env,
        "--model", model_path,
        "--episodes", "100",
        "--output", str(output_path),
    ]

    success, _ = run_command(cmd, cwd=project_root / "examples", desc=f"Evaluating seed={seed}")
    return success


def plot_seed(seed, env, results_dir, project_root):
    """绘制单个 seed 的训练曲线。"""
    model_name, log_dir, model_file = get_auto_paths(seed, env, project_root)
    legacy_model_name, legacy_log_dir = get_legacy_paths(seed, results_dir)

    if check_train_complete(log_dir, model_file):
        target_log_dir = log_dir
    elif check_train_complete(legacy_log_dir, project_root / "examples" / f"{legacy_model_name}.zip"):
        target_log_dir = legacy_log_dir
    else:
        print(f"[Warning] seed={seed} 无日志，跳过绘图")
        return False

    output_path = target_log_dir / "training_curves.png"

    cmd = [
        sys.executable,
        "scripts/plot_rl_curves.py",
        "--log-dir", str(target_log_dir),
        "--output", str(output_path),
    ]

    success, _ = run_command(cmd, cwd=project_root, desc=f"Plotting seed={seed}")
    return success


def aggregate_results(seeds, env, results_dir, project_root):
    """汇总所有 seed 的评估结果。兼容新版自动路径和旧版路径。"""
    print(f"\n{'='*70}")
    print(" Aggregate Results")
    print(f"{'='*70}")

    summaries = []
    for seed in seeds:
        # 优先检查新版自动路径，其次旧版
        _, auto_log_dir, _ = get_auto_paths(seed, env, project_root)
        _, legacy_log_dir = get_legacy_paths(seed, results_dir)

        json_path = auto_log_dir / "eval_detail.json"
        if not json_path.exists():
            json_path = legacy_log_dir / "eval_detail.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append({
                    "seed": seed,
                    **data["summary"],
                })
        else:
            print(f"[Warning] seed={seed} 评估结果未找到: {json_path}")

    if not summaries:
        print("[Error] 没有可用的评估结果")
        return

    # 打印表格
    print(f"\n{'Seed':>6} | {'Success%':>8} | {'AvgReward':>10} | {'StdReward':>10} | {'Median':>10}")
    print("-" * 60)
    for s in summaries:
        print(f"{s['seed']:>6} | {s['success_rate_percent']:>8.1f} | {s['avg_reward']:>10.3f} | {s['std_reward']:>10.3f} | {s['median_reward']:>10.3f}")

    # 计算平均和标准差
    success_rates = [s["success_rate_percent"] for s in summaries]
    avg_rewards = [s["avg_reward"] for s in summaries]

    avg_success = np.mean(success_rates)
    std_success = np.std(success_rates, ddof=1)
    avg_reward = np.mean(avg_rewards)
    std_reward = np.std(avg_rewards, ddof=1)

    print("-" * 60)
    print(f"{'Mean ± Std':>6} | {avg_success:>8.1f}±{std_success:>5.1f} | {avg_reward:>10.3f}±{std_reward:>5.3f} | {'—':>10} | {'—':>10}")

    # 保存汇总 JSON
    aggregate = {
        "seeds": summaries,
        "mean_success_rate_percent": round(float(avg_success), 1),
        "std_success_rate_percent": round(float(std_success), 1),
        "mean_avg_reward": round(float(avg_reward), 3),
        "std_avg_reward": round(float(std_reward), 3),
        "n_seeds": len(summaries),
    }
    aggregate_path = results_dir / "aggregate_results.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    print(f"\n  汇总结果: {aggregate_path}")

    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="RL Benchmark: 3-seed SAC+HER experiment orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整 benchmark
  python scripts/run_rl_benchmark.py

  # 跳过训练，仅评估 + 汇总已有模型
  python scripts/run_rl_benchmark.py --skip-train

  # 自定义参数
  python scripts/run_rl_benchmark.py --env HandReach-v1 --timesteps 100000 --seeds 0 1 2
        """,
    )
    parser.add_argument("--env", type=str, default=DEFAULT_ENV, help="环境 ID")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS, help="训练步数")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="随机种子列表")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="结果保存目录")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，仅评估已有模型")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--skip-plot", action="store_true", help="跳过绘图")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" RL Benchmark Runner")
    print("=" * 70)
    print(f" 环境:       {args.env}")
    print(f" 训练步数:   {args.timesteps}")
    print(f" 种子:       {args.seeds}")
    print(f" 结果目录:   {results_dir}")
    print(f" 跳过训练:   {args.skip_train}")
    print(f" 跳过评估:   {args.skip_eval}")
    print(f" 跳过绘图:   {args.skip_plot}")
    print("=" * 70)

    start_time = time.time()
    completed_seeds = []
    any_failed = False

    # --- Stage 1: Training ---
    if not args.skip_train:
        for seed in args.seeds:
            success = train_seed(seed, args.env, args.timesteps, results_dir, project_root)
            if success:
                completed_seeds.append(seed)
            else:
                print(f"[Error] seed={seed} 训练失败")
                any_failed = True
    else:
        print("\n[Skip] 跳过训练阶段")
        completed_seeds = list(args.seeds)

    # --- Stage 2: Evaluation ---
    if not args.skip_eval:
        for seed in completed_seeds:
            success = eval_seed(seed, args.env, results_dir, project_root)
            if not success:
                any_failed = True
    else:
        print("\n[Skip] 跳过评估阶段")

    # --- Stage 3: Plotting ---
    if not args.skip_plot:
        for seed in completed_seeds:
            success = plot_seed(seed, args.env, results_dir, project_root)
            if not success:
                any_failed = True
    else:
        print("\n[Skip] 跳过绘图阶段")

    # --- Stage 4: Aggregation ---
    aggregate_results(completed_seeds, args.env, results_dir, project_root)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" Benchmark 完成！总耗时: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    if any_failed or len(completed_seeds) < len(args.seeds):
        print("\n[Warning] 部分 seed 失败或未完成")
        sys.exit(1)


if __name__ == "__main__":
    main()
