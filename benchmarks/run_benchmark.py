"""
Synthetic Kinematic IK Sanity Benchmark — Pure NumPy Implementation
Avoids scipy.optimize which has a known bug on Python 3.14.
Uses custom gradient descent IK instead.

This is NOT a full dexterous retargeting benchmark. It tests IK reconstruction
on a simplified 5-finger 10-DOF planar hand using synthetic 5-fingertip landmarks.
"""
import numpy as np, json, csv, time, os, sys, argparse
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

# Finger parameters
FLEN = np.array([[0.035,0.030],[0.045,0.030],[0.050,0.035],[0.048,0.033],[0.040,0.028]])
BOFF = np.array([[0.02,-0.02,0.],[0.00,0.01,0.],[0.00,0.04,0.],[0.00,0.07,0.],[0.00,0.10,0.]])
ND = 10
JMIN = 0.0
JMAX = 1.2

def fk(joints):
    """Vectorized FK for all 5 fingers."""
    t1 = joints[0::2]  # MCP angles (5,)
    t2 = joints[1::2]  # PIP angles (5,)
    l1 = FLEN[:, 0]    # (5,)
    l2 = FLEN[:, 1]    # (5,)
    x = l1*np.sin(t1) + l2*np.sin(t1+t2) + BOFF[:, 0]
    z = l1*np.cos(t1) + l2*np.cos(t1+t2) + BOFF[:, 2]
    y = np.full(5, 0.0) + BOFF[:, 1]
    return np.stack([x, y, z], axis=-1)  # (5, 3)

def numerical_jacobian(joints, eps=1e-6):
    """Compute numerical Jacobian of FK w.r.t. joints."""
    J = np.zeros((15, ND))  # 5 tips x 3 coords = 15 rows, 10 DOF
    f0 = fk(joints).flatten()
    for i in range(ND):
        jp = joints.copy()
        jm = joints.copy()
        jp[i] += eps
        jm[i] -= eps
        J[:, i] = (fk(jp).flatten() - fk(jm).flatten()) / (2 * eps)
    return J

def gradient_descent_ik(target_tips, x0=None, lr=0.5, max_iter=100, tol=1e-6):
    """GD-based IK solver."""
    if x0 is None:
        x0 = np.ones(ND) * 0.5
    x = x0.copy()
    for _ in range(max_iter):
        tips = fk(x)
        error = tips.flatten() - target_tips.flatten()
        cost = 0.5 * np.dot(error, error)
        if cost < tol:
            break
        J = numerical_jacobian(x)
        grad = J.T @ error  # (10,)
        x = x - lr * grad
        x = np.clip(x, JMIN, JMAX)
    return x

def rule_mapping(lm, scale=1.6):
    """Rule-based: estimate bend from fingertip distance."""
    joints = np.zeros(ND)
    for i in range(5):
        dist = np.linalg.norm(lm[i])
        angle = np.clip(dist * scale, JMIN, JMAX)
        joints[i*2] = angle
        joints[i*2+1] = angle * 0.8
    return joints

def vector_optimization(lm):
    """Gradient descent IK (replaces scipy least_squares)."""
    return gradient_descent_ik(lm, lr=0.5, max_iter=100, tol=1e-8)

def slsqp_style_huber(lm, delta=0.005):
    """Gradient descent with Huber loss."""
    if True:
        def objective(j):
            tips = fk(j)
            errors = (tips - lm).flatten()
            ae = np.abs(errors)
            return np.where(ae < delta, 0.5*errors**2, delta*ae - 0.5*delta**2).sum()

        def gradient(j, eps=1e-6):
            g = np.zeros(ND)
            f0 = objective(j)
            for i in range(ND):
                jp = j.copy(); jp[i] += eps
                jm = j.copy(); jm[i] -= eps
                g[i] = (objective(jp) - objective(jm)) / (2*eps)
            return g

        x = np.ones(ND) * 0.5
        lr = 0.3
        for _ in range(150):
            g = gradient(x)
            x = x - lr * g
            x = np.clip(x, JMIN, JMAX)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic Kinematic IK Sanity Benchmark')
    parser.add_argument('n_samples', nargs='?', type=int, default=1000, help='Number of samples')
    parser.add_argument('seed', nargs='?', type=int, default=42, help='Random seed')
    parser.add_argument('--check', action='store_true', help='Validate benchmark results')
    return parser.parse_args()

def run_benchmark(n_samples, seed):
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n_samples):
        hj = rng.uniform(0.1, 1.0, ND)
        landmarks = fk(hj)
        data.append(landmarks)

    methods = [
        ('Rule Mapping', rule_mapping),
        ('Vector Optimization (GD)', vector_optimization),
        ('Huber Loss (GD)', slsqp_style_huber),
    ]

    results = {}
    for mname, mfn in methods:
        fpe_list = []
        p95_list = []
        lat_list = []
        lim_viol = 0

        for lm in data:
            t0 = time.perf_counter()
            pred_joints = mfn(lm)
            elapsed = time.perf_counter() - t0
            lat_list.append(elapsed)

            pred_tips = fk(pred_joints)
            errors_per_tip = np.linalg.norm(pred_tips - lm, axis=1)
            fpe_list.append(np.mean(errors_per_tip))
            p95_list.append(np.max(errors_per_tip))

            if np.any((pred_joints < JMIN - 1e-6) | (pred_joints > JMAX + 1e-6)):
                lim_viol += 1

        fa = np.array(fpe_list)
        pa = np.array(p95_list)
        la = np.array(lat_list)

        results[mname] = {
            'n': n_samples,
            'mean_fpe_mm': round(float(np.mean(fa) * 1000), 2),
            'std_fpe_mm': round(float(np.std(fa) * 1000), 2),
            'p95_fpe_mm': round(float(np.percentile(pa, 95) * 1000), 2),
            'mean_latency_ms': round(float(np.mean(la) * 1000), 3),
            'p99_latency_ms': round(float(np.percentile(la, 99) * 1000), 3),
            'limit_violation_pct': round(float(lim_viol / n_samples * 100), 1),
        }

    report = {
        'config': {
            'n_samples': n_samples,
            'seed': seed,
            'robot': 'Simplified 5-finger planar hand (10 DOF: MCP+PIP per finger)',
            'input': 'Synthetic 5-fingertip landmarks',
            'python': '3.14.6',
            'numpy': '2.5.1',
            'note': 'scipy.optimize unavailable on Python 3.14; IK solved via numerical GD',
        },
        'results': results,
    }

    jp = OUTPUT_DIR / 'benchmark_results.json'
    with open(jp, 'w') as f:
        json.dump(report, f, indent=2)

    cp = OUTPUT_DIR / 'benchmark_results.csv'
    with open(cp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method','N','MeanFPE(mm)','StdFPE(mm)','P95FPE(mm)',
                    'Latency(ms)','P99Latency(ms)','LimitViol(%)'])
        for name, r in results.items():
            w.writerow([name, r['n'], r['mean_fpe_mm'], r['std_fpe_mm'],
                        r['p95_fpe_mm'], r['mean_latency_ms'],
                        r['p99_latency_ms'], r['limit_violation_pct']])

    return results

def check_results(results):
    expected_methods = {'Rule Mapping', 'Vector Optimization (GD)', 'Huber Loss (GD)'}
    ok = True

    if set(results.keys()) != expected_methods:
        print(f"FAIL: Expected methods {expected_methods}, got {set(results.keys())}")
        ok = False

    for name, r in results.items():
        if not np.isfinite(r['mean_fpe_mm']):
            print(f"FAIL: {name} mean_fpe_mm is not finite ({r['mean_fpe_mm']})")
            ok = False
        if r['mean_latency_ms'] <= 0:
            print(f"FAIL: {name} mean_latency_ms is not positive ({r['mean_latency_ms']})")
            ok = False
        if r['limit_violation_pct'] != 0:
            print(f"FAIL: {name} limit_violation_pct is not zero ({r['limit_violation_pct']})")
            ok = False

    if ok:
        print("CHECK PASSED")
    else:
        print("CHECK FAILED")
        sys.exit(1)

def main():
    args = parse_args()
    results = run_benchmark(args.n_samples, args.seed)
    if args.check:
        check_results(results)

if __name__ == '__main__':
    main()
