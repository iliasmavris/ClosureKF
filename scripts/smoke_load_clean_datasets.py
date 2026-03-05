"""
Smoke Test: Load Clean Datasets Through Evaluation Pipeline
===========================================================
NO TRAINING.  Compatibility check only.

Loads the clean v1 10 Hz datasets through the existing eval codepath,
runs a single KF forward pass, prints shapes and sanity metrics.
Does NOT attempt to match paper metrics (data is from different time periods).
"""

import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_closure import CLOSURE_PARAM_NAMES

DT = 0.1

# ============================================================
#  Model loading (copied from utils_no_train.py)
# ============================================================

S1_CKPT = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
           / "stage1_physics_only.pth")
V2_CKPT = ROOT / "final_lockbox_v2" / "checkpoints"
SEEDS = [42, 43, 44]


def load_s1_params():
    ck = torch.load(S1_CKPT, map_location='cpu', weights_only=False)
    return ck['params']


def load_averaged_closure():
    cl = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl['q_scale'] = 1.0
    for key in ['b2', 'd2', 'q_scale']:
        vals = []
        for seed in SEEDS:
            ck = torch.load(V2_CKPT / f"closure_2t_s{seed}.pth",
                            map_location='cpu', weights_only=False)
            vals.append(ck['closure'][key])
        cl[key] = float(np.mean(vals))
    return cl


# ============================================================
#  KF filter (simplified, from utils_no_train.py pattern)
# ============================================================

def kf_forward_pass(params, cl_params, t_arr, x_obs, v_arr):
    """Run Kalman filter forward pass (matches utils_no_train.py pattern).
    Returns innovations, x_pred, x_post, u_post arrays."""
    import math
    N = len(x_obs)
    alpha = params['alpha']
    c_val = params['c']
    vc = params['vc']
    kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    R_val = params['R']
    q_sc = cl_params.get('q_scale', 1.0)
    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    innovations = np.full(N, np.nan)
    x_pred = np.full(N, np.nan)
    x_post = np.full(N, np.nan)
    u_post = np.full(N, np.nan)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    x_post[0] = s[0]; u_post[0] = s[1]

    for k in range(1, N):
        dt = t_arr[k] - t_arr[k - 1]
        if dt <= 0:
            dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v_arr[k - 1]**2 - vc**2, 0.0)

        dv_w = v_arr[k - 1] - v_arr[k - 2] if k >= 2 else 0.0
        cl = b2_v * dv_w - d2 * s[1] * abs(v_arr[k - 1])
        cl_dt = cl * dt

        xp = s[0] + s[1] * dt
        up = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt + cl_dt
        s_pred = np.array([xp, up])
        x_pred[k] = xp

        F = np.array([[1, dt], [-kap * dt, rho_u]])
        Q = np.diag([q_sc * qx * dt, q_sc * qu * dt])
        P_pred = F @ P @ F.T + Q

        innov = x_obs[k] - xp
        S_val = P_pred[0, 0] + R_val
        innovations[k] = innov

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(2) - np.outer(K, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R_val * np.outer(K, K)

        x_post[k] = s[0]; u_post[k] = s[1]

    return innovations, x_pred, x_post, u_post


# ============================================================
#  Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("SMOKE TEST: Load Clean Datasets Through Eval Pipeline")
    print("NO TRAINING -- compatibility check only")
    print("=" * 70)

    # Load model
    print("\n--- Loading model ---")
    params = load_s1_params()
    cl_params = load_averaged_closure()
    print(f"  S1 params loaded from: {S1_CKPT}")
    print(f"  Closure: b2={cl_params['b2']:.3f}, d2={cl_params['d2']:.3f}, "
          f"q_scale={cl_params['q_scale']:.3f}")

    # Load clean 10 Hz data
    clean_dir = ROOT / "processed_data_10hz_clean_v1"
    print(f"\n--- Loading clean 10 Hz data from {clean_dir} ---")

    checks_pass = True
    for split in ['train', 'val', 'test']:
        fpath = clean_dir / f"{split}_10hz_ready.csv"
        if not fpath.exists():
            print(f"  [FAIL] {fpath} not found!")
            checks_pass = False
            continue

        df = pd.read_csv(fpath)
        print(f"\n  {split}:")
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
        print(f"    t range: [{df.timestamp.iloc[0]:.2f}, "
              f"{df.timestamp.iloc[-1]:.2f}]")
        print(f"    v range: [{df.velocity.min():.4f}, "
              f"{df.velocity.max():.4f}]")
        print(f"    x range: [{df.displacement.min():.6f}, "
              f"{df.displacement.max():.6f}]")

        # Verify schema matches expected
        expected_cols = ['timestamp', 'time_delta', 'velocity', 'displacement']
        if list(df.columns) != expected_cols:
            print(f"    [FAIL] Column mismatch! Expected {expected_cols}")
            checks_pass = False
        else:
            print(f"    Schema: OK")

        # Verify no NaN
        n_nan = df.isna().sum().sum()
        if n_nan > 0:
            print(f"    [FAIL] {n_nan} NaN values!")
            checks_pass = False
        else:
            print(f"    NaN check: OK")

    if not checks_pass:
        print("\n[FAIL] Schema/loading checks failed!")
        return

    # Run KF forward pass on test split (with warmup from val)
    print(f"\n--- KF Forward Pass on Clean Test ---")
    df_val = pd.read_csv(clean_dir / "val_10hz_ready.csv")
    df_test = pd.read_csv(clean_dir / "test_10hz_ready.csv")

    # Use last 50s of val as warmup
    warmup_start = df_val.timestamp.iloc[-1] - 50.0
    df_warmup = df_val[df_val.timestamp >= warmup_start].copy()
    df_eval = pd.concat([df_warmup, df_test], ignore_index=True)

    test_start = df_test.timestamp.iloc[0]
    test_mask = df_eval.timestamp.values >= test_start

    t_arr = df_eval['timestamp'].values
    x_obs = df_eval['displacement'].values
    v_arr = df_eval['velocity'].values

    print(f"  Warmup: {len(df_warmup)} pts from val "
          f"(t={df_warmup.timestamp.iloc[0]:.1f} to "
          f"{df_warmup.timestamp.iloc[-1]:.1f})")
    print(f"  Test: {len(df_test)} pts "
          f"(t={df_test.timestamp.iloc[0]:.1f} to "
          f"{df_test.timestamp.iloc[-1]:.1f})")

    # Run closure filter
    print("\n  Running closure filter...")
    innov_cl, xp_cl, xpost_cl, upost_cl = kf_forward_pass(
        params, cl_params, t_arr, x_obs, v_arr)

    # Run baseline filter (no closure)
    zero_cl = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    zero_cl['q_scale'] = 1.0
    print("  Running baseline filter...")
    innov_bl, xp_bl, xpost_bl, upost_bl = kf_forward_pass(
        params, zero_cl, t_arr, x_obs, v_arr)

    # Test-only metrics
    e_cl = innov_cl[test_mask]
    e_bl = innov_bl[test_mask]
    N_test = test_mask.sum()

    print(f"\n  Test-only results (N={N_test}):")
    print(f"    Closure  innovations: mean={np.nanmean(e_cl):.6f}, "
          f"std={np.nanstd(e_cl):.6f}")
    print(f"    Baseline innovations: mean={np.nanmean(e_bl):.6f}, "
          f"std={np.nanstd(e_bl):.6f}")

    # ACF(1)
    def acf1(e):
        e = e[~np.isnan(e)]
        e_c = e - np.mean(e)
        return np.sum(e_c[:-1] * e_c[1:]) / np.sum(e_c**2)

    acf1_cl = acf1(e_cl)
    acf1_bl = acf1(e_bl)
    print(f"    Closure  ACF(1): {acf1_cl:.4f}")
    print(f"    Baseline ACF(1): {acf1_bl:.4f}")

    # Simple 1-step prediction error
    x_test = x_obs[test_mask]
    xp_test_cl = xp_cl[test_mask]
    xp_test_bl = xp_bl[test_mask]
    rmse_cl = np.sqrt(np.nanmean((x_test - xp_test_cl)**2))
    rmse_bl = np.sqrt(np.nanmean((x_test - xp_test_bl)**2))
    print(f"    Closure  1-step RMSE: {rmse_cl:.6f}")
    print(f"    Baseline 1-step RMSE: {rmse_bl:.6f}")

    # Check for NaN/Inf
    n_nan_cl = np.sum(np.isnan(innov_cl) | np.isinf(innov_cl))
    n_nan_bl = np.sum(np.isnan(innov_bl) | np.isinf(innov_bl))
    filter_stable = (n_nan_cl <= 1 and n_nan_bl <= 1)  # first step is NaN
    print(f"\n  Filter stability:")
    print(f"    Closure  NaN/Inf: {n_nan_cl}")
    print(f"    Baseline NaN/Inf: {n_nan_bl}")
    print(f"    Stable: {'OK' if filter_stable else 'FAIL'}")

    elapsed = time.time() - t0
    overall = 'PASS' if checks_pass and filter_stable else 'FAIL'
    print(f"\n{'='*70}")
    print(f"SMOKE TEST: {overall} ({elapsed:.1f}s)")
    print(f"{'='*70}")

    if overall == 'PASS':
        print("\nClean datasets are compatible with the evaluation pipeline.")
        print("Note: Metrics differ from paper values because test data")
        print("is from a different time period (1527-1799s vs 1135-1261s).")


if __name__ == '__main__':
    main()
