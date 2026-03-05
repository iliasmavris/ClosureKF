"""
Extended Horizon Scan (h=1..500) for Closure 2t model.

No training. Uses frozen lockbox v2 checkpoints + stage-1 physics.
Computes open-loop predictions under oracle / persistence / no-forcing modes.

Outputs -> final_lockbox_vX_no_train_diagnostics/horizon_extended/
  - dxr2_extended.csv, level_r2_extended.csv, mae_extended.csv, rmse_extended.csv
  - dxr2_extended.png, level_r2_extended.png, mae_extended.png, rmse_extended.png
  - README.md
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

import torch
torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_closure import CLOSURE_PARAM_NAMES

# ===== Config =====
DATA_DIR = ROOT / "processed_data_10hz"
S1_CKPT = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
           / "stage1_physics_only.pth")
V2_CKPT_DIR = ROOT / "final_lockbox_v2" / "checkpoints"

OUT_DIR = ROOT / "final_lockbox_vX_no_train_diagnostics" / "horizon_extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_MAX = 500          # maximum forecast horizon (50 s at 10 Hz)
DT = 0.1             # constant timestep
WARMUP_SEC = 50.0    # warmup from dev tail
SEEDS = [42, 43, 44]

# Bootstrap config
BOOT_REPS = 200
BLOCK_LEN = 75       # 7.5 s blocks (respects autocorrelation)
BOOT_ALPHA = 0.05    # 95% CI

plt.rcParams.update({
    'figure.dpi': 200, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    'font.family': 'serif', 'lines.linewidth': 1.8,
})


# ===================================================================
# Numpy KF filter + open-loop rollout
# ===================================================================

def kf_filter_save_states(params, cl_params, t, x_obs, v):
    """Run KF filter, return state (s) and covariance (P) at each step."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']

    a1 = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0)
    d3 = cl_params.get('d3', 0.0)

    states = np.zeros((N, 2))
    covs = np.zeros((N, 2, 2))

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states[0] = s
    covs[0] = P

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
              - d1*u_st**2 - d2*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_dt = cl * dt

        x_p = s[0] + s[1] * dt
        u_p = physics_drift + cl_dt
        s_pred = np.array([x_p, u_p])

        F_mat = np.array([[1, dt], [-kap*dt, rho_u]])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        H_vec = np.array([1.0, 0.0])
        IKH = np.eye(2) - np.outer(K, H_vec)
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)

        states[k] = s
        covs[k] = P

    return states, covs


def open_loop_rollout(s0, params, cl_params, v_future, dt=0.1):
    """
    Open-loop rollout from state s0 for len(v_future) steps.

    v_future: array of future water velocity values [H]
    Returns: x_pred array [H]
    """
    H = len(v_future)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']

    a1 = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0)
    d3 = cl_params.get('d3', 0.0)

    x_pred = np.empty(H)
    s = s0.copy()

    for k in range(H):
        rho_u = math.exp(-alpha * dt)
        v_curr = v_future[k]
        v_prev = v_future[k - 1] if k > 0 else 0.0  # will be set by caller
        dv = v_curr - v_prev
        g = max(v_curr**2 - vc**2, 0.0)

        x_new = s[0] + s[1] * dt
        physics_drift = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt
        u_st = s[1]
        cl = (-a1*u_st + b1_v*v_curr + b2_v*dv
              - d1*u_st**2 - d2*u_st*abs(v_curr) - d3*u_st*abs(u_st))
        cl_dt = cl * dt
        u_new = physics_drift + cl_dt

        s = np.array([x_new, u_new])
        x_pred[k] = x_new

    return x_pred


def run_extended_predictions(params, cl_params, t_arr, x_arr, v_arr,
                             test_start_idx, H_max):
    """
    Run KF filter then open-loop rollouts for all valid test origins.

    Returns dict with keys 'oracle', 'persistence', 'no_forcing', each containing:
      x_pred: [N_valid, H_max]  predicted displacement
      x_true: [N_valid, H_max]  true displacement
      x_cur:  [N_valid]          current displacement (at origin)
    Also returns n_valid_per_h: [H_max] count of valid points per horizon.
    """
    N = len(x_arr)

    # Phase 1: Run full KF filter to get posterior states at each time step
    print("  Running KF filter through warmup+test...")
    states, _ = kf_filter_save_states(params, cl_params, t_arr, x_arr, v_arr)

    # Phase 2: Open-loop rollouts from each test origin
    n_test = N - test_start_idx
    # Maximum valid origins: need H_max future steps available
    max_origin = N - H_max  # last valid origin index in full array

    # Collect valid test origins
    valid_origins = []
    for i in range(test_start_idx, max_origin):
        valid_origins.append(i)

    n_valid = len(valid_origins)
    print(f"  Valid test origins for h={H_max}: {n_valid}")
    print(f"  Test points total: {n_test}, "
          f"origins used: {n_valid} ({100*n_valid/n_test:.1f}%)")

    # Pre-allocate prediction arrays
    results = {}
    for mode in ['oracle', 'persistence', 'no_forcing']:
        results[mode] = {
            'x_pred': np.full((n_test, H_max), np.nan),  # all test points
            'x_true': np.full((n_test, H_max), np.nan),
            'x_cur': np.full(n_test, np.nan),
        }

    # Process each test origin
    for count, orig_idx in enumerate(range(test_start_idx, N)):
        test_i = orig_idx - test_start_idx  # index within test array
        x_cur = x_arr[orig_idx]

        # How many future steps available from this origin?
        h_avail = min(H_max, N - orig_idx - 1)
        if h_avail <= 0:
            continue

        s0 = states[orig_idx].copy()

        # True future displacement
        x_true_fut = x_arr[orig_idx + 1: orig_idx + 1 + h_avail]

        # --- Oracle forcing: true future velocity ---
        v_oracle = v_arr[orig_idx + 1: orig_idx + 1 + h_avail].copy()
        # Prepend current v for dv computation at step 0
        v_oracle_ext = np.empty(h_avail + 1)
        v_oracle_ext[0] = v_arr[orig_idx]  # v at origin (for dv at step 0)
        v_oracle_ext[1:] = v_oracle
        x_pred_oracle = _rollout_with_vprev(s0, params, cl_params,
                                             v_oracle, v_oracle_ext)

        # --- Persistence forcing: last known v repeated ---
        v_last = v_arr[orig_idx]
        v_persist = np.full(h_avail, v_last)
        v_persist_ext = np.full(h_avail + 1, v_last)
        x_pred_persist = _rollout_with_vprev(s0, params, cl_params,
                                              v_persist, v_persist_ext)

        # --- No-forcing: v = 0 ---
        v_zero = np.zeros(h_avail)
        v_zero_ext = np.zeros(h_avail + 1)
        v_zero_ext[0] = v_arr[orig_idx]  # keep dv realistic at step 0
        x_pred_noforce = _rollout_with_vprev(s0, params, cl_params,
                                              v_zero, v_zero_ext)

        # Store
        for mode, xp in [('oracle', x_pred_oracle),
                          ('persistence', x_pred_persist),
                          ('no_forcing', x_pred_noforce)]:
            results[mode]['x_pred'][test_i, :h_avail] = xp
            results[mode]['x_true'][test_i, :h_avail] = x_true_fut
            results[mode]['x_cur'][test_i] = x_cur

        if (count + 1) % 200 == 0:
            print(f"    Processed {count + 1}/{n_test} test origins")

    print(f"  Done. Processed {n_test} test origins.")

    # Count valid points per horizon
    n_valid_per_h = np.zeros(H_max, dtype=int)
    for h in range(H_max):
        n_valid_per_h[h] = np.sum(~np.isnan(results['oracle']['x_pred'][:, h]))

    return results, n_valid_per_h


def _rollout_with_vprev(s0, params, cl_params, v_future, v_ext):
    """
    Open-loop rollout with proper dv = v_curr - v_prev handling.
    v_ext has length len(v_future)+1, where v_ext[0] = v at origin.
    """
    H = len(v_future)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']

    a1 = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0)
    d3 = cl_params.get('d3', 0.0)

    dt = DT
    x_pred = np.empty(H)
    s = s0.copy()

    for k in range(H):
        rho_u = math.exp(-alpha * dt)
        v_curr = v_future[k]
        v_prev = v_ext[k]  # v_ext[k] = v at step k-1 (ext[0]=origin v)
        dv = v_curr - v_prev
        g = max(v_curr**2 - vc**2, 0.0)

        x_new = s[0] + s[1] * dt
        physics_drift = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt
        u_st = s[1]
        cl = (-a1*u_st + b1_v*v_curr + b2_v*dv
              - d1*u_st**2 - d2*u_st*abs(v_curr) - d3*u_st*abs(u_st))
        cl_dt = cl * dt
        u_new = physics_drift + cl_dt

        s = np.array([x_new, u_new])
        x_pred[k] = x_new

    return x_pred


# ===================================================================
# Metric computation
# ===================================================================

def compute_metrics_per_horizon(x_pred, x_true, x_cur, H_max):
    """
    Compute DxR2, MAE_increment, R2_level, RMSE_level for each horizon.

    x_pred: [N, H_max]
    x_true: [N, H_max]
    x_cur:  [N]
    """
    dxr2 = np.full(H_max, np.nan)
    mae_inc = np.full(H_max, np.nan)
    r2_level = np.full(H_max, np.nan)
    rmse_level = np.full(H_max, np.nan)

    for h in range(H_max):
        valid = ~np.isnan(x_pred[:, h]) & ~np.isnan(x_true[:, h])
        if valid.sum() < 10:
            continue

        xp = x_pred[valid, h]
        xt = x_true[valid, h]
        xc = x_cur[valid]

        # Displacement increments
        dx_pred = xp - xc
        dx_true = xt - xc

        # DxR2: R^2 of increments (baseline = constant-mean increment)
        ss_res = np.sum((dx_true - dx_pred)**2)
        ss_tot = np.sum((dx_true - np.mean(dx_true))**2)
        dxr2[h] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # MAE on increment
        mae_inc[h] = np.mean(np.abs(dx_true - dx_pred))

        # R^2 on level x_{t+h}
        ss_res_lev = np.sum((xt - xp)**2)
        ss_tot_lev = np.sum((xt - np.mean(xt))**2)
        r2_level[h] = 1.0 - ss_res_lev / ss_tot_lev if ss_tot_lev > 0 else 0.0

        # RMSE on level
        rmse_level[h] = np.sqrt(np.mean((xt - xp)**2))

    return dxr2, mae_inc, r2_level, rmse_level


# ===================================================================
# Moving block bootstrap
# ===================================================================

def block_bootstrap_metrics(x_pred, x_true, x_cur, H_max,
                            n_boot=BOOT_REPS, block_len=BLOCK_LEN,
                            rng_seed=42):
    """
    Moving block bootstrap for confidence intervals on all metrics.
    Returns (lo, hi) arrays for each metric at each horizon.
    """
    rng = np.random.RandomState(rng_seed)

    # For each horizon, we only have a subset of valid points.
    # We'll bootstrap on the full index set (using only valid points per h).
    N = x_pred.shape[0]

    # Pre-compute number of blocks needed
    n_blocks = int(np.ceil(N / block_len))

    boot_dxr2 = np.full((n_boot, H_max), np.nan)
    boot_mae = np.full((n_boot, H_max), np.nan)
    boot_r2lev = np.full((n_boot, H_max), np.nan)
    boot_rmse = np.full((n_boot, H_max), np.nan)

    for b in range(n_boot):
        # Draw block starts
        block_starts = rng.randint(0, N - block_len + 1, size=n_blocks)
        # Build resampled index array
        idx = np.concatenate([np.arange(s, s + block_len) for s in block_starts])
        idx = idx[:N]  # trim to original length

        # Compute metrics on resampled data
        for h in range(H_max):
            valid = ~np.isnan(x_pred[idx, h]) & ~np.isnan(x_true[idx, h])
            if valid.sum() < 10:
                continue

            ii = idx[valid]
            xp = x_pred[ii, h]
            xt = x_true[ii, h]
            xc = x_cur[ii]

            dx_p = xp - xc
            dx_t = xt - xc

            ss_res = np.sum((dx_t - dx_p)**2)
            ss_tot = np.sum((dx_t - np.mean(dx_t))**2)
            boot_dxr2[b, h] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            boot_mae[b, h] = np.mean(np.abs(dx_t - dx_p))

            ss_res_l = np.sum((xt - xp)**2)
            ss_tot_l = np.sum((xt - np.mean(xt))**2)
            boot_r2lev[b, h] = 1.0 - ss_res_l / ss_tot_l if ss_tot_l > 0 else 0.0

            boot_rmse[b, h] = np.sqrt(np.mean((xt - xp)**2))

    # Compute percentile CIs
    lo_pct = 100 * BOOT_ALPHA / 2
    hi_pct = 100 * (1 - BOOT_ALPHA / 2)

    def _ci(arr):
        lo = np.nanpercentile(arr, lo_pct, axis=0)
        hi = np.nanpercentile(arr, hi_pct, axis=0)
        return lo, hi

    return {
        'dxr2': _ci(boot_dxr2),
        'mae': _ci(boot_mae),
        'r2_level': _ci(boot_r2lev),
        'rmse': _ci(boot_rmse),
    }


# ===================================================================
# Plotting
# ===================================================================

def plot_dxr2(h_arr, metrics_dict, ci_dict, n_valid, out_path):
    """DxR2(h) curve with CI for oracle, persistence, no_forcing."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {'oracle': '#2166ac', 'persistence': '#b2182b',
              'no_forcing': '#636363'}
    labels = {'oracle': 'Oracle v', 'persistence': 'Persistence v',
              'no_forcing': 'No forcing'}

    for mode in ['oracle', 'persistence', 'no_forcing']:
        y = metrics_dict[mode]['dxr2']
        lo, hi = ci_dict[mode]['dxr2']
        ax.plot(h_arr, y, color=colors[mode], label=labels[mode], linewidth=1.5)
        ax.fill_between(h_arr, np.clip(lo, -2, None), np.clip(hi, None, 1.5),
                         color=colors[mode], alpha=0.15)

    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Forecast horizon h (steps, dt=0.1 s)')
    ax.set_ylabel('DxR2(h)')
    ax.set_title('Displacement Increment Skill (DxR2) vs Horizon')
    ax.set_ylim(-1.5, 0.8)
    ax.legend(loc='best')

    # Add secondary x-axis for seconds
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * DT, ax.get_xlim()[1] * DT)
    ax2.set_xlabel('Horizon (seconds)')

    # Add n_valid annotation
    ax.text(0.98, 0.02, f'N valid: {n_valid[0]} (h=1) to {n_valid[-1]} (h={H_MAX})',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            color='gray')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_r2_level(h_arr, metrics_dict, ci_dict, n_valid, out_path):
    """R2_level(h) curve with CI."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {'oracle': '#2166ac', 'persistence': '#b2182b',
              'no_forcing': '#636363'}
    labels = {'oracle': 'Oracle v', 'persistence': 'Persistence v',
              'no_forcing': 'No forcing'}

    for mode in ['oracle', 'persistence', 'no_forcing']:
        y = metrics_dict[mode]['r2_level']
        lo, hi = ci_dict[mode]['r2_level']
        ax.plot(h_arr, y, color=colors[mode], label=labels[mode], linewidth=1.5)
        ax.fill_between(h_arr, np.clip(lo, -3, None), np.clip(hi, None, 1.5),
                         color=colors[mode], alpha=0.15)

    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Forecast horizon h (steps, dt=0.1 s)')
    ax.set_ylabel('R2 on level x(t+h)')
    ax.set_title('Level Skill (R2) vs Horizon')
    ax.set_ylim(-2.0, 0.8)
    ax.legend(loc='best')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * DT, ax.get_xlim()[1] * DT)
    ax2.set_xlabel('Horizon (seconds)')

    ax.text(0.98, 0.02, f'N valid: {n_valid[0]} (h=1) to {n_valid[-1]} (h={H_MAX})',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            color='gray')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_mae(h_arr, metrics_dict, ci_dict, n_valid, out_path):
    """MAE_increment(h) curve with CI."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {'oracle': '#2166ac', 'persistence': '#b2182b',
              'no_forcing': '#636363'}
    labels = {'oracle': 'Oracle v', 'persistence': 'Persistence v',
              'no_forcing': 'No forcing'}

    for mode in ['oracle', 'persistence', 'no_forcing']:
        y = metrics_dict[mode]['mae']
        lo, hi = ci_dict[mode]['mae']
        ax.plot(h_arr, y, color=colors[mode], label=labels[mode], linewidth=1.5)
        ax.fill_between(h_arr, lo, hi, color=colors[mode], alpha=0.15)

    ax.set_xlabel('Forecast horizon h (steps, dt=0.1 s)')
    ax.set_ylabel('MAE on increment dx (m)')
    ax.set_title('Mean Absolute Error on Displacement Increment vs Horizon')
    ax.legend(loc='best')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * DT, ax.get_xlim()[1] * DT)
    ax2.set_xlabel('Horizon (seconds)')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_rmse(h_arr, metrics_dict, ci_dict, n_valid, out_path):
    """RMSE_level(h) curve with CI."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {'oracle': '#2166ac', 'persistence': '#b2182b',
              'no_forcing': '#636363'}
    labels = {'oracle': 'Oracle v', 'persistence': 'Persistence v',
              'no_forcing': 'No forcing'}

    for mode in ['oracle', 'persistence', 'no_forcing']:
        y = metrics_dict[mode]['rmse']
        lo, hi = ci_dict[mode]['rmse']
        ax.plot(h_arr, y, color=colors[mode], label=labels[mode], linewidth=1.5)
        ax.fill_between(h_arr, lo, hi, color=colors[mode], alpha=0.15)

    ax.set_xlabel('Forecast horizon h (steps, dt=0.1 s)')
    ax.set_ylabel('RMSE on level x(t+h) (m)')
    ax.set_title('Root Mean Square Error on Displacement Level vs Horizon')
    ax.legend(loc='best')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * DT, ax.get_xlim()[1] * DT)
    ax2.set_xlabel('Horizon (seconds)')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("EXTENDED HORIZON SCAN (h=1..500)")
    print("=" * 70)
    print(f"Output -> {OUT_DIR}")

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n--- Step 1: Load data ---")
    df_val = pd.read_csv(DATA_DIR / "val_10hz_ready.csv")
    df_test = pd.read_csv(DATA_DIR / "test_10hz_ready.csv")
    TEST_START = df_test['timestamp'].iloc[0]

    # Dev = val rows before test start
    df_dev = df_val[df_val['timestamp'] < TEST_START].copy()

    # Warmup = last 50s of dev
    warmup_start = df_dev.timestamp.max() - WARMUP_SEC
    test_warmup = df_dev[df_dev['timestamp'] >= warmup_start].copy()

    # Concatenate warmup + test
    df_filter = pd.concat([test_warmup, df_test], ignore_index=True)
    t_arr = df_filter['timestamp'].values
    x_arr = df_filter['displacement'].values.astype(np.float64)
    v_arr = df_filter['velocity'].values.astype(np.float64)

    test_mask = t_arr >= TEST_START
    test_start_idx = int(np.argmax(test_mask))
    n_test = int(test_mask.sum())

    print(f"  Warmup: {len(test_warmup)} pts "
          f"[{test_warmup.timestamp.min():.1f}, {test_warmup.timestamp.max():.1f}]")
    print(f"  Test: {n_test} pts "
          f"[{df_test.timestamp.min():.1f}, {df_test.timestamp.max():.1f}]")
    print(f"  Filter array: {len(df_filter)} pts, test_start_idx={test_start_idx}")

    # ------------------------------------------------------------------
    # Step 2: Load model params
    # ------------------------------------------------------------------
    print("\n--- Step 2: Load model params ---")
    device = torch.device('cpu')
    ckpt_s1 = torch.load(S1_CKPT, map_location=device, weights_only=False)
    s1_params = ckpt_s1['params']
    print(f"  S1: alpha={s1_params['alpha']:.4f} c={s1_params['c']:.4f} "
          f"kappa={s1_params['kappa']:.4f} vc={s1_params['vc']:.4f}")

    # Load closure params (mean across 3 seeds)
    all_cl = []
    for seed in SEEDS:
        ck = torch.load(V2_CKPT_DIR / f"closure_2t_s{seed}.pth",
                        map_location=device, weights_only=False)
        all_cl.append(ck['closure'])
        print(f"  Seed {seed}: b2={ck['closure']['b2']:.4f} "
              f"d2={ck['closure']['d2']:.4f} q_scale={ck['closure']['q_scale']:.4f}")

    # Mean closure params
    cl_mean = {}
    for key in all_cl[0].keys():
        cl_mean[key] = float(np.mean([c[key] for c in all_cl]))
    print(f"  Mean closure: b2={cl_mean['b2']:.4f} d2={cl_mean['d2']:.4f} "
          f"q_scale={cl_mean['q_scale']:.4f}")

    # Also prepare baseline (zero closure)
    cl_zero = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl_zero['q_scale'] = 1.0

    # ------------------------------------------------------------------
    # Step 3: Run extended predictions (closure model)
    # ------------------------------------------------------------------
    print("\n--- Step 3: Extended predictions (closure 2t, mean params) ---")
    results_cl, n_valid = run_extended_predictions(
        s1_params, cl_mean, t_arr, x_arr, v_arr, test_start_idx, H_MAX)

    # ------------------------------------------------------------------
    # Step 4: Run extended predictions (baseline / physics-only)
    # ------------------------------------------------------------------
    print("\n--- Step 4: Extended predictions (baseline / physics-only) ---")
    results_base, _ = run_extended_predictions(
        s1_params, cl_zero, t_arr, x_arr, v_arr, test_start_idx, H_MAX)

    # ------------------------------------------------------------------
    # Step 5: Compute metrics
    # ------------------------------------------------------------------
    print("\n--- Step 5: Compute metrics ---")
    h_arr = np.arange(1, H_MAX + 1)

    metrics = {}
    for model_label, res in [('closure', results_cl), ('baseline', results_base)]:
        metrics[model_label] = {}
        for mode in ['oracle', 'persistence', 'no_forcing']:
            dxr2, mae, r2l, rmse = compute_metrics_per_horizon(
                res[mode]['x_pred'], res[mode]['x_true'],
                res[mode]['x_cur'], H_MAX)
            metrics[model_label][mode] = {
                'dxr2': dxr2, 'mae': mae, 'r2_level': r2l, 'rmse': rmse
            }
            # Quick sanity check at h=10
            if not np.isnan(dxr2[9]):
                print(f"  {model_label}/{mode}: DxR2@10={dxr2[9]:.4f}, "
                      f"R2_level@10={r2l[9]:.4f}, MAE@10={mae[9]:.6f}, "
                      f"RMSE@10={rmse[9]:.6f}")

    # ------------------------------------------------------------------
    # Step 6: Bootstrap CIs (closure model only, oracle + persistence)
    # ------------------------------------------------------------------
    print("\n--- Step 6: Bootstrap CIs (200 replicates, block=75) ---")
    ci = {}
    for mode in ['oracle', 'persistence', 'no_forcing']:
        print(f"  Bootstrap for closure/{mode}...")
        ci[mode] = block_bootstrap_metrics(
            results_cl[mode]['x_pred'], results_cl[mode]['x_true'],
            results_cl[mode]['x_cur'], H_MAX,
            n_boot=BOOT_REPS, block_len=BLOCK_LEN, rng_seed=42)

    # ------------------------------------------------------------------
    # Step 7: Save CSVs
    # ------------------------------------------------------------------
    print("\n--- Step 7: Save CSVs ---")

    for metric_name in ['dxr2', 'mae', 'r2_level', 'rmse']:
        fname = {
            'dxr2': 'dxr2_extended.csv',
            'mae': 'mae_extended.csv',
            'r2_level': 'level_r2_extended.csv',
            'rmse': 'rmse_extended.csv',
        }[metric_name]

        rows = []
        for h in range(H_MAX):
            row = {'h': h + 1, 'h_sec': (h + 1) * DT, 'n_valid': int(n_valid[h])}
            for model_label in ['closure', 'baseline']:
                for mode in ['oracle', 'persistence', 'no_forcing']:
                    col = f'{model_label}_{mode}'
                    row[col] = metrics[model_label][mode][metric_name][h]
            # CI columns (closure only)
            for mode in ['oracle', 'persistence', 'no_forcing']:
                lo, hi_ = ci[mode][metric_name]
                row[f'closure_{mode}_ci_lo'] = lo[h]
                row[f'closure_{mode}_ci_hi'] = hi_[h]
            rows.append(row)

        df_out = pd.DataFrame(rows)
        df_out.to_csv(OUT_DIR / fname, index=False)
        print(f"  Saved {fname} ({len(df_out)} rows)")

    # ------------------------------------------------------------------
    # Step 8: Plots (closure model, oracle + persistence + no_forcing)
    # ------------------------------------------------------------------
    print("\n--- Step 8: Plots ---")

    # Use closure metrics for main plots
    plot_metrics = {mode: metrics['closure'][mode] for mode in
                    ['oracle', 'persistence', 'no_forcing']}

    plot_dxr2(h_arr, plot_metrics, ci, n_valid,
              OUT_DIR / "dxr2_extended.png")
    plot_r2_level(h_arr, plot_metrics, ci, n_valid,
                  OUT_DIR / "level_r2_extended.png")
    plot_mae(h_arr, plot_metrics, ci, n_valid,
             OUT_DIR / "mae_extended.png")
    plot_rmse(h_arr, plot_metrics, ci, n_valid,
              OUT_DIR / "rmse_extended.png")

    # ------------------------------------------------------------------
    # Step 9: README.md summary
    # ------------------------------------------------------------------
    print("\n--- Step 9: README.md ---")

    # Analyze DxR2 curves
    readme_lines = ["# Extended Horizon Scan Results\n"]
    readme_lines.append(f"- Horizons: h = 1..{H_MAX} (dt={DT}s, "
                        f"max {H_MAX*DT:.0f}s)\n")
    readme_lines.append(f"- Bootstrap: {BOOT_REPS} replicates, "
                        f"block length = {BLOCK_LEN} steps ({BLOCK_LEN*DT:.1f}s)\n")
    readme_lines.append(f"- Model: Closure 2t (mean of seeds {SEEDS})\n")
    readme_lines.append(f"- N valid at h=1: {n_valid[0]}, "
                        f"at h={H_MAX}: {n_valid[H_MAX-1]}\n")

    readme_lines.append("\n## DxR2 Analysis (Closure model)\n")
    for mode in ['oracle', 'persistence', 'no_forcing']:
        dxr2 = metrics['closure'][mode]['dxr2']
        valid_mask = ~np.isnan(dxr2)
        if not valid_mask.any():
            readme_lines.append(f"### {mode}: no valid data\n")
            continue

        dxr2_valid = dxr2[valid_mask]
        h_valid = h_arr[valid_mask]

        # Peak skill
        peak_idx = np.argmax(dxr2_valid)
        peak_h = h_valid[peak_idx]
        peak_val = dxr2_valid[peak_idx]

        # First h after peak where DxR2 < 0
        after_peak = dxr2_valid[peak_idx:]
        h_after = h_valid[peak_idx:]
        cross_zero = np.where(after_peak < 0)[0]
        if len(cross_zero) > 0:
            zero_h = h_after[cross_zero[0]]
            zero_info = f"h={zero_h} ({zero_h*DT:.1f}s)"
        else:
            zero_info = "never (stays >= 0 through h={})".format(h_valid[-1])

        readme_lines.append(f"### {mode}\n")
        readme_lines.append(f"- Peak DxR2: **{peak_val:.4f}** at "
                            f"h={peak_h} ({peak_h*DT:.1f}s)\n")
        readme_lines.append(f"- Falls below 0: {zero_info}\n")
        readme_lines.append(f"- DxR2 at h=10: {dxr2[9]:.4f}\n")
        if not np.isnan(dxr2[49]):
            readme_lines.append(f"- DxR2 at h=50 (5s): {dxr2[49]:.4f}\n")
        if not np.isnan(dxr2[99]):
            readme_lines.append(f"- DxR2 at h=100 (10s): {dxr2[99]:.4f}\n")
        if not np.isnan(dxr2[349]):
            readme_lines.append(f"- DxR2 at h=350 (35s): {dxr2[349]:.4f}\n")

    # Oracle-persistence gap
    readme_lines.append("\n## Oracle vs Persistence Gap\n")
    for h_check in [10, 50, 100, 200, 350, 500]:
        idx = h_check - 1
        if idx >= H_MAX:
            continue
        orc = metrics['closure']['oracle']['dxr2'][idx]
        per = metrics['closure']['persistence']['dxr2'][idx]
        if not np.isnan(orc) and not np.isnan(per):
            gap = orc - per
            readme_lines.append(f"- h={h_check} ({h_check*DT:.1f}s): "
                                f"oracle={orc:.4f}, persistence={per:.4f}, "
                                f"gap={gap:.4f}\n")

    # Level R2 summary
    readme_lines.append("\n## Level R2 Summary (Oracle, Closure)\n")
    for h_check in [10, 50, 100, 200, 350]:
        idx = h_check - 1
        if idx >= H_MAX:
            continue
        r2l = metrics['closure']['oracle']['r2_level'][idx]
        if not np.isnan(r2l):
            readme_lines.append(f"- h={h_check} ({h_check*DT:.1f}s): R2_level={r2l:.4f}\n")

    readme_lines.append(f"\n---\nGenerated in {time.time()-t_start:.1f}s\n")

    with open(OUT_DIR / "README.md", 'w') as f:
        f.write("".join(readme_lines))
    print("  Saved README.md")

    # Summary
    print("\n" + "=" * 70)
    print(f"DONE in {time.time()-t_start:.1f}s")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
