"""
Step 5A: Transfer Evaluation (10 Hz -> 50 Hz, no retrain).

Evaluates 10 Hz-trained d2-only checkpoints (3 seeds) on both 10 Hz and 50 Hz
data using the SAME codepath, computing transfer ratios and diagnostics.

Usage:
  python -u ems_v1/eval/rate_transfer_step5/run_transfer_step5a.py

Inputs:
  - Checkpoints: ems_v1/runs/lockbox_ems_v1_d2only_10hz_3seed/seed{s}/checkpoints/
  - 10 Hz data: processed_data_10hz_clean_v1/{val,test}_10hz_ready.csv
  - 50 Hz data: processed_data_50hz_clean_v1/{val,test}_50hz_ready.csv

Outputs -> ems_v1/eval/rate_transfer_step5/
"""

import os, sys, math, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from ems_v1.eval.metrics_pack import (
    compute_deltax_metrics, compute_acf, ljung_box,
    compute_nis, compute_cov90,
)

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
TAU_PHYS = [0.1, 0.2, 0.5, 1.0, 2.0]
WARMUP_SEC = 50.0
SEEDS = [1, 2, 3]

# Paths
CKPT_DIR = ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_10hz_3seed"
DATA_10HZ = ROOT / "processed_data_10hz_clean_v1"
DATA_50HZ = ROOT / "processed_data_50hz_clean_v1"
OUT_DIR = ROOT / "ems_v1" / "eval" / "rate_transfer_step5"

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})


# ==============================================================================
#  HELPERS
# ==============================================================================

def infer_dt(df, expected, tol=1e-4):
    """Infer dt from CSV and assert it matches expected value."""
    dt = np.median(df['time_delta'].values[1:])  # skip row 0 (may be 0)
    assert abs(dt - expected) < tol, f"dt={dt}, expected {expected}"
    return float(dt)


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ==============================================================================
#  NUMPY 2-STATE FILTER (local helper, mirrors Step 4 exactly)
# ==============================================================================

def kf_filter_2state(params, cl_params, t, x_obs, v):
    """2-state KF with full tracking."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_x = np.zeros(N)
    states_u = np.zeros(N)
    cl_dt_arr = np.zeros(N)
    phys_arr = np.zeros(N)

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    a1 = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1 = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0)
    d3 = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0:
            dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)
        physics_drift = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt
        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = (-a1*u_st + b1_v*v_w + b2_v*dv_w
              - d1*u_st**2 - d2_v*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_d = cl * dt
        x_p = s[0] + s[1] * dt
        u_p = physics_drift + cl_d
        s_pred = np.array([x_p, u_p])
        cl_dt_arr[k] = cl_d
        phys_arr[k] = physics_drift

        dcl_du = (-a1 - 2*d1*u_st - d2_v*abs(v_w) - 2*d3*abs(u_st))
        F_mat = np.array([[1, dt], [-kap*dt, rho_u + dcl_du*dt]])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        innovations[k] = innov
        S_values[k] = S_val

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(2) - np.outer(K, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)
        states_x[k] = s[0]; states_u[k] = s[1]

    return {
        'innovations': innovations, 'S_values': S_values,
        'states_x': states_x, 'states_u': states_u,
        'cl_dt': cl_dt_arr, 'physics': phys_arr,
    }


# ==============================================================================
#  DxR2 MULTI-HORIZON (local helper, mirrors Step 4 exactly)
# ==============================================================================

def compute_dxr2_multihorizon(params, cl_params, states_x, states_u,
                               t, x_obs, v, max_h, eval_start, indices=None):
    """Returns per-horizon R2_dx, skill_dx, MAE_dx, RMSE_dx arrays."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    a1 = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1 = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0); d3 = cl_params.get('d3', 0.0)

    dx_pred = [[] for _ in range(max_h)]
    dx_true = [[] for _ in range(max_h)]

    if indices is None:
        indices = range(max(eval_start, 1), N - 1)

    for i in indices:
        if i < 1 or i >= N - 1:
            continue
        sx, su = states_x[i], states_u[i]
        max_steps = min(max_h, N - 1 - i)
        for step in range(max_steps):
            k_s = i + 1 + step
            dt_s = t[k_s] - t[k_s - 1] if k_s > 0 else 0.1
            if dt_s <= 0:
                dt_s = 0.1
            v_w = v[k_s - 1] if k_s >= 1 else 0.0
            dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
            rho = math.exp(-alpha * dt_s)
            g = max(v_w**2 - vc**2, 0.0)
            cl = (-a1*su + b1_v*v_w + b2_v*dv_w
                  - d1*su**2 - d2_v*su*abs(v_w) - d3*su*abs(su))
            sx_new = sx + su * dt_s
            su_new = rho*su - kap*sx*dt_s + c_val*g*dt_s + cl*dt_s
            sx, su = sx_new, su_new
            h = step + 1
            dx_pred[h-1].append(sx - x_obs[i])
            dx_true[h-1].append(x_obs[i + h] - x_obs[i])

    r2_arr = np.full(max_h, np.nan)
    skill_arr = np.full(max_h, np.nan)
    mae_arr = np.full(max_h, np.nan)
    rmse_arr = np.full(max_h, np.nan)
    n_arr = np.zeros(max_h, dtype=int)

    for h in range(max_h):
        if len(dx_pred[h]) < 10:
            continue
        m = compute_deltax_metrics(dx_true[h], dx_pred[h])
        r2_arr[h] = m['r2_dx']
        skill_arr[h] = m['skill_dx']
        mae_arr[h] = m['mae_dx']
        rmse_arr[h] = m['rmse_dx']
        n_arr[h] = m['n']

    return r2_arr, skill_arr, mae_arr, rmse_arr, n_arr


# ==============================================================================
#  EVALUATE ONE RATE CONFIG
# ==============================================================================

def evaluate_rate(label, params, cl_params, t, x_obs, v, eval_start,
                  max_h, headline_steps, headline_taus, dt_val):
    """Full evaluation: filter + innovations + multihorizon DxR2."""
    t0 = time.time()
    filt = kf_filter_2state(params, cl_params, t, x_obs, v)
    innov = filt['innovations']; S_vals = filt['S_values']
    sx = filt['states_x']; su = filt['states_u']

    e = innov[eval_start:]; S_sc = S_vals[eval_start:]
    valid = ~np.isnan(e)
    e_v = e[valid]; S_v = S_sc[valid]
    n_valid = len(e_v)

    acf_vals = compute_acf(e_v, max_lag=50)
    nis = compute_nis(e_v, S_v)
    cov90 = compute_cov90(e_v, S_v)
    lb = ljung_box(acf_vals, n_valid)

    cl_sc = filt['cl_dt'][eval_start:]
    ph_sc = filt['physics'][eval_start:]
    tot_sc = cl_sc + ph_sc
    var_cl = np.var(cl_sc)
    var_tot = np.var(tot_sc) if np.var(tot_sc) > 1e-15 else 1.0
    grey_frac = float(var_cl / var_tot) if var_tot > 1e-15 else 0.0

    r2, skill, mae, rmse, ns = compute_dxr2_multihorizon(
        params, cl_params, sx, su, t, x_obs, v, max_h, eval_start)

    elapsed = time.time() - t0
    for hi, hs in zip(headline_steps, headline_taus):
        if hi <= max_h:
            r2v = r2[hi-1] if not np.isnan(r2[hi-1]) else float('nan')
            print(f"    [{label}] h={hi} ({hs}s): DxR2={r2v:+.4f} "
                  f"MAE={mae[hi-1]:.6f} n={ns[hi-1]}")
    print(f"    [{label}] ACF1={acf_vals[1]:.4f} NIS={nis:.4f} "
          f"cov90={cov90:.3f} [{elapsed:.0f}s]")

    return {
        'acf1': float(acf_vals[1]), 'acf5': float(acf_vals[5]),
        'nis_mean': nis, 'cov90': cov90, 'grey_frac': grey_frac,
        'dxr2': r2, 'skill': skill, 'mae': mae, 'rmse': rmse,
        'n_per_h': ns, 'n_valid': n_valid,
    }


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    t0_global = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 5A: TRANSFER EVALUATION (10 Hz -> 50 Hz)")
    print("=" * 70)

    # ------------------------------------------------------------------
    #  Load data
    # ------------------------------------------------------------------
    print_section("LOAD DATA")

    df_val_10 = pd.read_csv(DATA_10HZ / "val_10hz_ready.csv")
    df_test_10 = pd.read_csv(DATA_10HZ / "test_10hz_ready.csv")
    df_val_50 = pd.read_csv(DATA_50HZ / "val_50hz_ready.csv")
    df_test_50 = pd.read_csv(DATA_50HZ / "test_50hz_ready.csv")

    dt_10hz = infer_dt(df_test_10, expected=0.1, tol=0.005)
    dt_50hz = infer_dt(df_test_50, expected=0.02, tol=0.005)
    print(f"  Inferred dt: 10Hz={dt_10hz:.4f}s, 50Hz={dt_50hz:.4f}s")

    # Headline horizon steps for each rate
    H_10HZ = [round(tau / dt_10hz) for tau in TAU_PHYS]
    H_50HZ = [round(tau / dt_50hz) for tau in TAU_PHYS]
    MAX_H_10 = round(10.0 / dt_10hz)   # 100
    MAX_H_50 = round(10.0 / dt_50hz)   # 500
    print(f"  10Hz headline steps: {H_10HZ}, max_h={MAX_H_10}")
    print(f"  50Hz headline steps: {H_50HZ}, max_h={MAX_H_50}")

    # Prepare warm-start arrays (val warmup + test)
    def prepare_warm(df_val, df_test, warmup_sec):
        test_start_time = df_test['timestamp'].iloc[0]
        warmup_start = df_val['timestamp'].iloc[-1] - warmup_sec
        df_warmup = df_val[df_val['timestamp'] >= warmup_start].copy()
        df_warm = pd.concat([df_warmup, df_test], ignore_index=True)
        t = df_warm['timestamp'].values.astype(np.float64)
        x = df_warm['displacement'].values.astype(np.float64)
        v = df_warm['velocity'].values.astype(np.float64)
        score_mask = t >= test_start_time
        eval_start = int(np.argmax(score_mask))
        return t, x, v, eval_start

    t_10, x_10, v_10, es_10 = prepare_warm(df_val_10, df_test_10, WARMUP_SEC)
    t_50, x_50, v_50, es_50 = prepare_warm(df_val_50, df_test_50, WARMUP_SEC)
    print(f"  10Hz: {len(t_10)} pts, warmup={es_10}, scored={len(t_10)-es_10}")
    print(f"  50Hz: {len(t_50)} pts, warmup={es_50}, scored={len(t_50)-es_50}")

    # ------------------------------------------------------------------
    #  Per-seed evaluation
    # ------------------------------------------------------------------
    transfer_rows = []
    diag_rows = []
    dense_curves = {s: {} for s in SEEDS}

    for seed in SEEDS:
        print_section(f"SEED {seed}")

        # Load checkpoint
        ckpt_path = (CKPT_DIR / f"seed{seed}" / "checkpoints"
                     / f"closure_d2only_seed{seed}.pth")
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        s2_params = ckpt['params']
        cl_sum = ckpt['closure']

        # Also load S1 physics for physics-only baseline
        s1_path = (CKPT_DIR / f"seed{seed}" / "checkpoints"
                   / f"stage1_physics_seed{seed}.pth")
        s1_ckpt = torch.load(s1_path, map_location='cpu', weights_only=False)
        s1_params = s1_ckpt['params']

        print(f"  Loaded seed {seed}: d2={cl_sum['d2']:.4f}, "
              f"q_scale={cl_sum['q_scale']:.4f}")
        print(f"  Physics: alpha={s1_params['alpha']:.4f}, "
              f"kappa={s1_params['kappa']:.4f}")

        # Build param dicts (closure model)
        clos_pp = {
            'alpha': s2_params['alpha'], 'c': s2_params['c'],
            'vc': s2_params['vc'], 'kappa': s2_params['kappa'],
            'qx': s2_params['qx'], 'qu': s2_params['qu'],
            'R': s2_params['R'],
            'P0_xx': s2_params['P0_xx'], 'P0_uu': s2_params['P0_uu'],
        }
        clos_cl = {
            'a1': cl_sum['a1'], 'b1': cl_sum['b1'], 'b2': cl_sum['b2'],
            'd1': cl_sum['d1'], 'd2': cl_sum['d2'], 'd3': cl_sum['d3'],
            'q_scale': cl_sum['q_scale'],
        }

        # Physics-only (for reference)
        phys_pp = {
            'alpha': s1_params['alpha'], 'c': s1_params['c'],
            'vc': s1_params['vc'], 'kappa': s1_params['kappa'],
            'qx': s1_params['qx'], 'qu': s1_params['qu'],
            'R': s1_params['R'],
            'P0_xx': s1_params['P0_xx'], 'P0_uu': s1_params['P0_uu'],
        }
        phys_cl = {k: 0.0 for k in ['a1', 'b1', 'b2', 'd1', 'd2', 'd3']}
        phys_cl['q_scale'] = 1.0

        # --- 10 Hz reference ---
        print(f"\n  --- 10 Hz reference (closure) ---")
        res_10_clos = evaluate_rate(
            f"10Hz-clos-s{seed}", clos_pp, clos_cl,
            t_10, x_10, v_10, es_10,
            MAX_H_10, H_10HZ, TAU_PHYS, dt_10hz)

        print(f"\n  --- 10 Hz reference (physics) ---")
        res_10_phys = evaluate_rate(
            f"10Hz-phys-s{seed}", phys_pp, phys_cl,
            t_10, x_10, v_10, es_10,
            MAX_H_10, H_10HZ, TAU_PHYS, dt_10hz)

        # --- 50 Hz transfer ---
        print(f"\n  --- 50 Hz transfer (closure) ---")
        res_50_clos = evaluate_rate(
            f"50Hz-clos-s{seed}", clos_pp, clos_cl,
            t_50, x_50, v_50, es_50,
            MAX_H_50, H_50HZ, TAU_PHYS, dt_50hz)

        print(f"\n  --- 50 Hz transfer (physics) ---")
        res_50_phys = evaluate_rate(
            f"50Hz-phys-s{seed}", phys_pp, phys_cl,
            t_50, x_50, v_50, es_50,
            MAX_H_50, H_50HZ, TAU_PHYS, dt_50hz)

        # --- Transfer table rows ---
        for ti, tau in enumerate(TAU_PHYS):
            h10 = H_10HZ[ti]
            h50 = H_50HZ[ti]
            dxr2_10 = res_10_clos['dxr2'][h10-1]
            dxr2_50 = res_50_clos['dxr2'][h50-1]
            skill_10 = res_10_clos['skill'][h10-1]
            skill_50 = res_50_clos['skill'][h50-1]
            mae_10 = res_10_clos['mae'][h10-1]
            mae_50 = res_50_clos['mae'][h50-1]
            ratio = dxr2_50 / dxr2_10 if abs(dxr2_10) > 1e-10 else float('nan')
            transfer_rows.append({
                'seed': seed, 'tau_s': tau,
                'h_10hz': h10, 'h_50hz': h50,
                'dxr2_10hz': dxr2_10, 'dxr2_50hz': dxr2_50,
                'ratio': ratio,
                'skill_10hz': skill_10, 'skill_50hz': skill_50,
                'mae_10hz': mae_10, 'mae_50hz': mae_50,
            })

        # --- Diagnostic rows ---
        for rate, res_c, res_p, dt_v in [
            ('10hz', res_10_clos, res_10_phys, dt_10hz),
            ('50hz', res_50_clos, res_50_phys, dt_50hz),
        ]:
            diag_rows.append({
                'seed': seed, 'rate': rate,
                'acf1_clos': res_c['acf1'], 'acf5_clos': res_c['acf5'],
                'nis_mean_clos': res_c['nis_mean'],
                'cov90_clos': res_c['cov90'],
                'grey_frac_clos': res_c['grey_frac'],
                'acf1_phys': res_p['acf1'], 'acf5_phys': res_p['acf5'],
                'nis_mean_phys': res_p['nis_mean'],
                'cov90_phys': res_p['cov90'],
            })

        # --- Dense horizon curves ---
        dense_10_phys = []
        dense_10_clos = []
        for h in range(MAX_H_10):
            dense_10_phys.append({
                'h': h + 1, 'tau_s': (h + 1) * dt_10hz,
                'dxr2_phys': res_10_phys['dxr2'][h],
                'dxr2_clos': res_10_clos['dxr2'][h],
            })
        dense_50_phys = []
        dense_50_clos = []
        for h in range(MAX_H_50):
            dense_50_phys.append({
                'h': h + 1, 'tau_s': (h + 1) * dt_50hz,
                'dxr2_phys': res_50_phys['dxr2'][h],
                'dxr2_clos': res_50_clos['dxr2'][h],
            })

        # Save per-seed dense curves
        df_dense_10 = pd.DataFrame(dense_10_phys)
        df_dense_10['rate'] = '10hz'
        df_dense_50 = pd.DataFrame(dense_50_phys)
        df_dense_50['rate'] = '50hz'
        df_dense = pd.concat([df_dense_10, df_dense_50], ignore_index=True)
        df_dense.to_csv(
            OUT_DIR / f"transfer_horizon_dense_seed{seed}.csv", index=False)

        dense_curves[seed] = {
            '10hz': df_dense_10, '50hz': df_dense_50,
        }

    # ------------------------------------------------------------------
    #  Save transfer table
    # ------------------------------------------------------------------
    print_section("SAVE RESULTS")

    df_transfer = pd.DataFrame(transfer_rows)
    df_transfer.to_csv(OUT_DIR / "transfer_10hz_to_50hz.csv", index=False)
    print(f"  Wrote transfer_10hz_to_50hz.csv ({len(df_transfer)} rows)")

    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(OUT_DIR / "transfer_diag_10hz_to_50hz.csv", index=False)
    print(f"  Wrote transfer_diag_10hz_to_50hz.csv ({len(df_diag)} rows)")

    # ------------------------------------------------------------------
    #  Summary statistics
    # ------------------------------------------------------------------
    print_section("TRANSFER SUMMARY")

    # Per-tau mean across seeds
    print(f"\n  {'tau_s':>6} {'DxR2_10':>10} {'DxR2_50':>10} "
          f"{'Ratio':>8} {'MAE_10':>10} {'MAE_50':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    for tau in TAU_PHYS:
        sub = df_transfer[df_transfer['tau_s'] == tau]
        m10 = sub['dxr2_10hz'].mean()
        m50 = sub['dxr2_50hz'].mean()
        mr = sub['ratio'].mean()
        mae10 = sub['mae_10hz'].mean()
        mae50 = sub['mae_50hz'].mean()
        print(f"  {tau:6.1f} {m10:+10.4f} {m50:+10.4f} "
              f"{mr:8.3f} {mae10:10.6f} {mae50:10.6f}")

    # Transfer ratio at 1.0s
    tau1_rows = df_transfer[df_transfer['tau_s'] == 1.0]
    mean_ratio_1s = tau1_rows['ratio'].mean()
    print(f"\n  Transfer ratio at tau=1.0s (mean): {mean_ratio_1s:.4f}")

    # Innovation diagnostics comparison
    print(f"\n  {'Rate':>6} {'ACF1_clos':>10} {'NIS_clos':>10} "
          f"{'cov90':>8} {'grey_frac':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for rate in ['10hz', '50hz']:
        sub = df_diag[df_diag['rate'] == rate]
        print(f"  {rate:>6} {sub['acf1_clos'].mean():10.4f} "
              f"{sub['nis_mean_clos'].mean():10.4f} "
              f"{sub['cov90_clos'].mean():8.3f} "
              f"{sub['grey_frac_clos'].mean():10.4f}")

    # ------------------------------------------------------------------
    #  VERIFICATION
    # ------------------------------------------------------------------
    print_section("VERIFICATION")

    pass_ratio = mean_ratio_1s >= 0.80
    print(f"  Transfer ratio at tau=1.0s >= 0.80: "
          f"{'PASS' if pass_ratio else 'FAIL'} ({mean_ratio_1s:.4f})")

    expected_files = [
        "transfer_10hz_to_50hz.csv",
        "transfer_diag_10hz_to_50hz.csv",
    ] + [f"transfer_horizon_dense_seed{s}.csv" for s in SEEDS]
    all_files = all((OUT_DIR / f).exists() for f in expected_files)
    print(f"  All CSV files generated: {'PASS' if all_files else 'FAIL'}")

    # Record NIS/ACF shifts (not enforced)
    diag_10 = df_diag[df_diag['rate'] == '10hz']
    diag_50 = df_diag[df_diag['rate'] == '50hz']
    nis_shift = diag_50['nis_mean_clos'].mean() - diag_10['nis_mean_clos'].mean()
    acf_shift = diag_50['acf1_clos'].mean() - diag_10['acf1_clos'].mean()
    print(f"  NIS shift (50Hz - 10Hz): {nis_shift:+.4f} (RECORD only)")
    print(f"  ACF1 shift (50Hz - 10Hz): {acf_shift:+.4f} (RECORD only)")
    print(f"  Grey-box frac: 10Hz={diag_10['grey_frac_clos'].mean():.4f}, "
          f"50Hz={diag_50['grey_frac_clos'].mean():.4f} (RECORD only)")

    # ------------------------------------------------------------------
    #  FIGURES
    # ------------------------------------------------------------------
    print_section("FIGURES")

    # --- Fig: DxR2 vs tau ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect per-seed curves for 10Hz and 50Hz
    for rate, color, ls, label_base in [
        ('10hz', 'steelblue', '-', '10 Hz'),
        ('50hz', 'indianred', '--', '50 Hz transfer'),
    ]:
        all_taus = None
        all_dxr2 = []
        for seed in SEEDS:
            dc = dense_curves[seed][rate]
            taus = dc['tau_s'].values
            dxr2 = dc['dxr2_clos'].values
            if all_taus is None:
                all_taus = taus
            all_dxr2.append(dxr2)
            ax.plot(taus, dxr2, color=color, ls=ls, alpha=0.2, lw=0.8)

        # Mean + band
        arr = np.array(all_dxr2)
        mean_curve = np.nanmean(arr, axis=0)
        std_curve = np.nanstd(arr, axis=0)
        ax.plot(all_taus, mean_curve, color=color, ls=ls, lw=2,
                label=f'{label_base} (mean)')
        ax.fill_between(all_taus, mean_curve - std_curve,
                        mean_curve + std_curve, color=color, alpha=0.15)

    for tau in TAU_PHYS:
        ax.axvline(tau, color='gray', ls=':', alpha=0.4)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Horizon (s)')
    ax.set_ylabel('DxR2(tau)')
    ax.set_title('Step 5A: Transfer DxR2 (10 Hz -> 50 Hz, no retrain)')
    ax.set_xlim(0, 2.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_transfer_dxr2_vs_tau.png")
    plt.close(fig)
    print(f"  Wrote fig_transfer_dxr2_vs_tau.png")

    # --- Fig: Calibration (NIS + cov90 bars) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: NIS bars
    ax = axes[0]
    rates = ['10 Hz', '50 Hz transfer']
    nis_10 = [diag_10['nis_mean_clos'].values, diag_10['nis_mean_phys'].values]
    nis_50 = [diag_50['nis_mean_clos'].values, diag_50['nis_mean_phys'].values]

    x_pos = np.arange(2)
    w = 0.35
    nis_clos_means = [diag_10['nis_mean_clos'].mean(),
                      diag_50['nis_mean_clos'].mean()]
    nis_clos_stds = [diag_10['nis_mean_clos'].std(),
                     diag_50['nis_mean_clos'].std()]
    nis_phys_means = [diag_10['nis_mean_phys'].mean(),
                      diag_50['nis_mean_phys'].mean()]
    nis_phys_stds = [diag_10['nis_mean_phys'].std(),
                     diag_50['nis_mean_phys'].std()]
    ax.bar(x_pos - w/2, nis_clos_means, w, yerr=nis_clos_stds,
           label='Closure', color='indianred', alpha=0.8, capsize=4)
    ax.bar(x_pos + w/2, nis_phys_means, w, yerr=nis_phys_stds,
           label='Physics', color='steelblue', alpha=0.8, capsize=4)
    ax.axhline(1.0, color='black', ls='--', alpha=0.5, label='Ideal NIS=1')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rates)
    ax.set_ylabel('NIS')
    ax.set_title('(A) Normalized Innovation Squared')
    ax.legend(fontsize=8)

    # Panel B: cov90 bars
    ax = axes[1]
    cov_clos_means = [diag_10['cov90_clos'].mean(),
                      diag_50['cov90_clos'].mean()]
    cov_clos_stds = [diag_10['cov90_clos'].std(),
                     diag_50['cov90_clos'].std()]
    cov_phys_means = [diag_10['cov90_phys'].mean(),
                      diag_50['cov90_phys'].mean()]
    cov_phys_stds = [diag_10['cov90_phys'].std(),
                     diag_50['cov90_phys'].std()]
    ax.bar(x_pos - w/2, cov_clos_means, w, yerr=cov_clos_stds,
           label='Closure', color='indianred', alpha=0.8, capsize=4)
    ax.bar(x_pos + w/2, cov_phys_means, w, yerr=cov_phys_stds,
           label='Physics', color='steelblue', alpha=0.8, capsize=4)
    ax.axhline(0.90, color='black', ls='--', alpha=0.5, label='Nominal 90%')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rates)
    ax.set_ylabel('Coverage (90%)')
    ax.set_title('(B) Empirical 90% Coverage')
    ax.legend(fontsize=8)

    fig.suptitle('Step 5A: Transfer Calibration Diagnostics', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig_transfer_calib.png")
    plt.close(fig)
    print(f"  Wrote fig_transfer_calib.png")

    # ------------------------------------------------------------------
    #  README
    # ------------------------------------------------------------------
    readme_lines = [
        "# Step 5A: Transfer Evaluation (10 Hz -> 50 Hz)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Protocol",
        "- 10 Hz-trained d2-only checkpoints (3 seeds) evaluated on both",
        "  10 Hz and 50 Hz test sets using identical codepath",
        "- Warm start: last 50s of validation for filter spinup",
        "- Oracle open-loop rollout for multihorizon DxR2",
        "- dt inferred from CSV at runtime, headline horizons in physical time",
        "",
        f"## Key Results",
        f"- Transfer ratio at tau=1.0s: {mean_ratio_1s:.4f} "
        f"({'PASS' if pass_ratio else 'FAIL'}, threshold >= 0.80)",
        f"- NIS shift: {nis_shift:+.4f}",
        f"- ACF1 shift: {acf_shift:+.4f}",
        "",
        "## Files",
        "- transfer_10hz_to_50hz.csv: Per-seed, per-tau transfer ratios",
        "- transfer_diag_10hz_to_50hz.csv: Innovation diagnostics",
        "- transfer_horizon_dense_seed{1,2,3}.csv: Dense DxR2 curves",
        "- fig_transfer_dxr2_vs_tau.png: DxR2 horizon curves",
        "- fig_transfer_calib.png: NIS + coverage bars",
        "",
        "## Checkpoints Used",
    ]
    for seed in SEEDS:
        ckpt_path = (CKPT_DIR / f"seed{seed}" / "checkpoints"
                     / f"closure_d2only_seed{seed}.pth")
        readme_lines.append(f"- seed {seed}: {ckpt_path.relative_to(ROOT)}")

    with open(OUT_DIR / "README.md", 'w') as f:
        f.write('\n'.join(readme_lines))
    print(f"  Wrote README.md")

    # ------------------------------------------------------------------
    #  FINAL
    # ------------------------------------------------------------------
    elapsed = time.time() - t0_global
    n_files = sum(1 for _ in OUT_DIR.iterdir() if _.is_file())
    print_section("DONE")
    print(f"  Total files: {n_files}")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Transfer ratio@1s: {mean_ratio_1s:.4f} "
          f"({'PASS' if pass_ratio else 'FAIL'})")


if __name__ == '__main__':
    main()
