"""
V6 Controls & Ablations: Fair baselines, causal knockouts, disturbance model.

Answers: Are v6 gains due to retraining or the memory state?

Models evaluated:
  1. old_2state       : Original 2-state (v2 lockbox, old S1 physics)
  2. fair_2state      : 2-state retrained with v6 pipeline
  3. v6_3state        : Existing v6 3-state (beta~0)
  4. v6_3state_beta0  : v6 3-state with beta hard-set to 0
  5. v6_3state_rhom0  : v6 3-state with rho_m hard-set to 0
  6. disturbance_fix1 : 3-state with beta=1 fixed, trained from scratch

For each model: raw+normalized ACF, Ljung-Box, DxR2(h) oracle+no_forcing, VoF(h).

Usage:  python -u scripts/v6_controls_and_ablations.py
Output: final_lockbox_v6_controls/
"""

import os, sys, math, json, hashlib, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_closure import KalmanForecasterClosure, CLOSURE_PARAM_NAMES

# ===== Paths =====
DATA_DIR   = ROOT / "processed_data_10hz"
S1_CKPT_2S = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
              / "stage1_physics_only.pth")
V2_CKPT    = ROOT / "final_lockbox_v2" / "checkpoints"
V6_DIR     = ROOT / "final_lockbox_v6_mem"

OUT = ROOT / "final_lockbox_v6_controls"
for d in ['fair_2state', 'ablations', 'disturbance_fix1',
          'diagnostics', 'figures', 'comparison']:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ===== Constants =====
DT = 0.1; FORCE_CPU = True; MAX_HORIZON = 10
L = 64; H = 20; BATCH = 128; VAR_FLOOR = 1e-6
SEEDS = [42, 43, 44]

S1_EPOCHS = 120; S1_LR = 1e-2; S1_PATIENCE = 20
S2_EPOCHS = 200; S2_LR = 1e-2; S2_PATIENCE = 30

BLOCK_LEN_S = 3.0; R_BOOT = 2000; RNG_SEED = 54321

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})


# ============================================================
#  HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_acf(e, max_lag=50):
    e_c = e - np.mean(e)
    var = np.var(e)
    n = len(e)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(e_c[:n-l] * e_c[l:]) / (n * var) if l > 0
                     else 1.0 for l in range(max_lag + 1)])


def ljung_box(acf_vals, n, lags=[5, 10, 20, 50]):
    results = []
    for m in lags:
        if m >= n or m >= len(acf_vals):
            continue
        Q = n * (n + 2) * np.sum(
            acf_vals[1:m+1]**2 / (n - np.arange(1, m+1)))
        p = 1.0 - sp_stats.chi2.cdf(Q, df=m)
        results.append({'lag': m, 'Q': float(Q), 'p': float(p)})
    return results


def crps_gaussian(y, mu, sigma):
    z = (y - mu) / (sigma + 1e-15)
    return sigma * (z * (2 * sp_stats.norm.cdf(z) - 1)
                   + 2 * sp_stats.norm.pdf(z)
                   - 1.0 / math.sqrt(math.pi))


def gaussian_nll(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ============================================================
#  NUMPY 2-STATE FILTER (full, with return_states)
# ============================================================

def kf_filter_2state_full(params, cl_params, t, x_obs, v):
    """2-state KF returning innovations, S_values, states, P_post."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values    = np.full(N, np.nan)
    states_x = np.zeros(N); states_u = np.zeros(N)
    P_post_list = [None] * N

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    a1   = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1   = cl_params.get('d1', 0.0)
    d2   = cl_params.get('d2', 0.0)
    d3   = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]
    P_post_list[0] = P.copy()

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
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
        innovations[k] = innov; S_values[k] = S_val
        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(2) - np.outer(K, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)
        states_x[k] = s[0]; states_u[k] = s[1]
        P_post_list[k] = P.copy()

    return innovations, S_values, states_x, states_u, P_post_list


# ============================================================
#  NUMPY 3-STATE FILTER (full, with return_states)
# ============================================================

def kf_filter_3state_full(params, cl_params, mem_params, t, x_obs, v):
    """3-state KF returning innovations, S_values, states, P_post."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values    = np.full(N, np.nan)
    states_x = np.zeros(N); states_u = np.zeros(N); states_m = np.zeros(N)
    P_post_list = [None] * N

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    a1   = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1   = cl_params.get('d1', 0.0)
    d2   = cl_params.get('d2', 0.0)
    d3   = cl_params.get('d3', 0.0)

    beta  = mem_params.get('beta', 0.0)
    rho_m = mem_params.get('rho_m', 0.0)
    qm    = mem_params.get('qm', 0.01)
    P0_mm = mem_params.get('P0_mm', 0.01)

    s = np.array([x_obs[0], 0.0, 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu'], P0_mm])
    states_x[0] = s[0]; states_u[0] = s[1]; states_m[0] = s[2]
    P_post_list[0] = P.copy()

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)
        physics_drift = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt
        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = (-a1*u_st + b1_v*v_w + b2_v*dv_w
              - d1*u_st**2 - d2*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_dt = cl * dt
        x_p = s[0] + s[1] * dt
        u_p = physics_drift + cl_dt + beta * s[2]
        m_p = rho_m * s[2]
        s_pred = np.array([x_p, u_p, m_p])

        F_mat = np.array([
            [1.0,      dt,   0.0  ],
            [-kap*dt,  rho_u, beta ],
            [0.0,      0.0,   rho_m],
        ])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt, qm])
        P_pred = F_mat @ P @ F_mat.T + Q
        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        innovations[k] = innov; S_values[k] = S_val
        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(3) - np.outer(K, np.array([1.0, 0.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)
        states_x[k] = s[0]; states_u[k] = s[1]; states_m[k] = s[2]
        P_post_list[k] = P.copy()

    return innovations, S_values, states_x, states_u, states_m, P_post_list


# ============================================================
#  DxR2 FUNCTIONS (both 2-state and 3-state, with mode)
# ============================================================

def _rollout_2state(sx, su, steps, t, v, start_k, params, cl_params, mode):
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    a1 = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0); d3 = cl_params.get('d3', 0.0)
    N = len(t)
    for step in range(steps):
        k_s = start_k + step
        if k_s >= N: break
        dt_s = t[k_s] - t[k_s - 1] if k_s > 0 else 0.1
        if dt_s <= 0: dt_s = 0.1
        if mode == 'oracle':
            v_w = v[k_s - 1] if k_s >= 1 else 0.0
            dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
        else:
            v_w = 0.0; dv_w = 0.0
        rho = math.exp(-alpha * dt_s)
        g = max(v_w**2 - vc**2, 0.0)
        cl = (-a1*su + b1_v*v_w + b2_v*dv_w
              - d1*su**2 - d2*su*abs(v_w) - d3*su*abs(su))
        sx_new = sx + su * dt_s
        su_new = rho*su - kap*sx*dt_s + c_val*g*dt_s + cl*dt_s
        sx, su = sx_new, su_new
    return sx, su


def _rollout_3state(sx, su, sm, steps, t, v, start_k, params, cl_params, mem_params, mode):
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    a1 = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0); d3 = cl_params.get('d3', 0.0)
    beta = mem_params.get('beta', 0.0); rho_m = mem_params.get('rho_m', 0.0)
    N = len(t)
    for step in range(steps):
        k_s = start_k + step
        if k_s >= N: break
        dt_s = t[k_s] - t[k_s - 1] if k_s > 0 else 0.1
        if dt_s <= 0: dt_s = 0.1
        if mode == 'oracle':
            v_w = v[k_s - 1] if k_s >= 1 else 0.0
            dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
        else:
            v_w = 0.0; dv_w = 0.0
        rho = math.exp(-alpha * dt_s)
        g = max(v_w**2 - vc**2, 0.0)
        cl = (-a1*su + b1_v*v_w + b2_v*dv_w
              - d1*su**2 - d2*su*abs(v_w) - d3*su*abs(su))
        sx_new = sx + su * dt_s
        su_new = rho*su - kap*sx*dt_s + c_val*g*dt_s + cl*dt_s + beta*sm
        sm_new = rho_m * sm
        sx, su, sm = sx_new, su_new, sm_new
    return sx, su, sm


def compute_dxr2(n_states, params, cl_params, mem_params,
                 states_x, states_u, states_m,
                 t, x_obs, v, max_h=10, eval_start=1, mode='oracle'):
    """DxR2(h) for either 2-state or 3-state."""
    N = len(x_obs)
    r2_arr = np.zeros(max_h)
    for h in range(1, max_h + 1):
        dx_pred = []; dx_obs = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            sm = states_m[i] if n_states == 3 else 0.0
            start_k = i + 1
            if n_states == 2:
                sx_end, _ = _rollout_2state(sx, su, h, t, v, start_k, params, cl_params, mode)
            else:
                sx_end, _, _ = _rollout_3state(sx, su, sm, h, t, v, start_k, params, cl_params, mem_params, mode)
            dx_pred.append(sx_end - x_obs[i])
            dx_obs.append(x_obs[i + h] - x_obs[i])
        dp = np.array(dx_pred); do = np.array(dx_obs)
        ss_res = np.sum((do - dp)**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h - 1] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ============================================================
#  UQ FUNCTIONS (h-step covariance propagation)
# ============================================================

def compute_hstep_uq(n_states, params, cl_params, mem_params,
                     states_x, states_u, states_m, P_post_list,
                     t, x_obs, v, max_h=10, eval_start=1, mode='oracle'):
    """Per-horizon predictive distributions."""
    N = len(x_obs)
    alpha = params['alpha']; kap = params['kappa']
    q_sc = cl_params.get('q_scale', 1.0); R = params['R']
    beta = mem_params.get('beta', 0.0) if mem_params else 0.0
    rho_m_val = mem_params.get('rho_m', 0.0) if mem_params else 0.0
    qm = mem_params.get('qm', 0.01) if mem_params else 0.0

    results = {}
    for h in range(1, max_h + 1):
        obs_list = []; mean_list = []; var_list = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            sm = states_m[i] if n_states == 3 else 0.0
            P_h = P_post_list[i].copy()
            start_k = i + 1
            for step in range(h):
                k_s = start_k + step
                if k_s >= N: break
                dt_s = t[k_s] - t[k_s - 1] if k_s > 0 else 0.1
                if dt_s <= 0: dt_s = 0.1
                if mode == 'oracle':
                    v_w = v[k_s - 1] if k_s >= 1 else 0.0
                    dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
                else:
                    v_w = 0.0; dv_w = 0.0
                rho = math.exp(-alpha * dt_s)
                # Rollout mean
                if n_states == 2:
                    sx, su = _rollout_2state(sx, su, 1, t, v, k_s, params, cl_params, mode)
                    F_mat = np.array([[1, dt_s], [-kap*dt_s, rho]])
                    Q_mat = np.diag([q_sc*params['qx']*dt_s, q_sc*params['qu']*dt_s])
                else:
                    sx, su, sm = _rollout_3state(sx, su, sm, 1, t, v, k_s, params, cl_params, mem_params, mode)
                    F_mat = np.array([
                        [1.0, dt_s, 0.0],
                        [-kap*dt_s, rho, beta],
                        [0.0, 0.0, rho_m_val],
                    ])
                    Q_mat = np.diag([q_sc*params['qx']*dt_s,
                                     q_sc*params['qu']*dt_s, qm])
                P_h = F_mat @ P_h @ F_mat.T + Q_mat
            obs_list.append(x_obs[i + h])
            mean_list.append(sx)
            var_list.append(P_h[0, 0] + R)
        results[f'h{h}'] = {
            'obs': np.array(obs_list),
            'mean': np.array(mean_list),
            'var': np.array(var_list),
        }
    return results


# ============================================================
#  VoF BOOTSTRAP
# ============================================================

def compute_vof_bootstrap(uq_orc, uq_nof, max_h=10,
                          block_len_idx=30, R_BOOT=2000, rng_seed=54321):
    """Compute VoF%(h) with block-bootstrap CIs."""
    crps_orc_h = np.zeros(max_h); crps_nof_h = np.zeros(max_h)
    for h in range(1, max_h + 1):
        hk = f'h{h}'
        sig_o = np.sqrt(uq_orc[hk]['var'])
        sig_n = np.sqrt(uq_nof[hk]['var'])
        crps_orc_h[h-1] = float(np.mean(crps_gaussian(
            uq_orc[hk]['obs'], uq_orc[hk]['mean'], sig_o)))
        crps_nof_h[h-1] = float(np.mean(crps_gaussian(
            uq_nof[hk]['obs'], uq_nof[hk]['mean'], sig_n)))

    vof_raw = crps_nof_h - crps_orc_h
    vof_pct = 100.0 * vof_raw / np.maximum(crps_nof_h, 1e-12)

    N_win = len(uq_orc['h1']['obs'])
    n_blocks = max(1, N_win // block_len_idx)
    block_windows = []
    nonempty = []
    for b in range(n_blocks):
        lo = b * block_len_idx; hi = min(N_win, (b + 1) * block_len_idx)
        if hi > lo:
            nonempty.append(b)
            block_windows.append(np.arange(lo, hi))
        else:
            block_windows.append(np.array([], dtype=int))

    rng = np.random.RandomState(rng_seed)
    boot_vof = np.zeros((R_BOOT, max_h))
    block_indices = np.array(nonempty)
    for r in range(R_BOOT):
        sampled = rng.choice(block_indices, size=len(nonempty), replace=True)
        win_idx = np.concatenate([block_windows[b] for b in sampled])
        if len(win_idx) == 0:
            boot_vof[r] = np.nan; continue
        for h in range(1, max_h + 1):
            hk = f'h{h}'
            idx = win_idx[win_idx < len(uq_orc[hk]['obs'])]
            if len(idx) == 0:
                boot_vof[r, h-1] = np.nan; continue
            sig_o = np.sqrt(uq_orc[hk]['var'][idx])
            sig_n = np.sqrt(uq_nof[hk]['var'][idx])
            c_o = float(np.mean(crps_gaussian(
                uq_orc[hk]['obs'][idx], uq_orc[hk]['mean'][idx], sig_o)))
            c_n = float(np.mean(crps_gaussian(
                uq_nof[hk]['obs'][idx], uq_nof[hk]['mean'][idx], sig_n)))
            boot_vof[r, h-1] = 100.0 * (c_n - c_o) / max(c_n, 1e-12)

    ci_lo = np.nanpercentile(boot_vof, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_vof, 97.5, axis=0)
    return vof_pct, ci_lo, ci_hi


# ============================================================
#  UNIFIED EVALUATION FUNCTION
# ============================================================

def evaluate_variant(name, n_states, phys_params, cl_params, mem_params,
                     t, x_obs, v, test_mask, eval_start, max_h=10):
    """Full evaluation for one model variant. Returns metrics dict."""
    t0 = time.time()
    print(f"\n  --- Evaluating: {name} ({n_states}-state) ---")

    # Step 1: Filter
    if n_states == 2:
        innov, S_vals, sx, su, P_list = kf_filter_2state_full(
            phys_params, cl_params, t, x_obs, v)
        sm = np.zeros(len(x_obs))
    else:
        innov, S_vals, sx, su, sm, P_list = kf_filter_3state_full(
            phys_params, cl_params, mem_params, t, x_obs, v)

    # Step 2: Extract test-set innovations
    e_raw = innov[test_mask]; S_test = S_vals[test_mask]
    valid = ~np.isnan(e_raw)
    e_raw_v = e_raw[valid]; S_v = S_test[valid]

    # Step 3: Normalized innovations
    e_norm_v = e_raw_v / np.sqrt(np.maximum(S_v, 1e-15))

    # Step 4: ACF raw + normalized
    acf_raw  = compute_acf(e_raw_v, max_lag=50)
    acf_norm = compute_acf(e_norm_v, max_lag=50)
    n_valid = int(valid.sum())

    # Step 5: Ljung-Box raw + normalized
    lb_raw  = ljung_box(acf_raw, n_valid)
    lb_norm = ljung_box(acf_norm, n_valid)

    print(f"    ACF(1) raw={acf_raw[1]:.4f}  norm={acf_norm[1]:.4f}")

    # Step 6: DxR2 oracle + no_forcing
    dxr2_orc = compute_dxr2(n_states, phys_params, cl_params, mem_params,
                             sx, su, sm, t, x_obs, v, max_h, eval_start, 'oracle')
    dxr2_nof = compute_dxr2(n_states, phys_params, cl_params, mem_params,
                             sx, su, sm, t, x_obs, v, max_h, eval_start, 'no_forcing')
    dr2 = dxr2_orc - dxr2_nof
    print(f"    DxR2@10 orc={dxr2_orc[9]:+.4f}  nof={dxr2_nof[9]:+.4f}  "
          f"mean(5-10) orc={np.mean(dxr2_orc[4:10]):+.4f}")

    # Step 7: VoF with bootstrap
    block_len_idx = max(1, round(BLOCK_LEN_S / DT))
    uq_orc = compute_hstep_uq(n_states, phys_params, cl_params, mem_params,
                                sx, su, sm, P_list, t, x_obs, v,
                                max_h, eval_start, 'oracle')
    uq_nof = compute_hstep_uq(n_states, phys_params, cl_params, mem_params,
                                sx, su, sm, P_list, t, x_obs, v,
                                max_h, eval_start, 'no_forcing')
    vof_pct, ci_lo, ci_hi = compute_vof_bootstrap(
        uq_orc, uq_nof, max_h, block_len_idx, R_BOOT, RNG_SEED)
    vof_avg = float(np.mean(vof_pct[4:10]))
    print(f"    VoF% avg(5-10)={vof_avg:+.3f}%")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s")

    return {
        'name': name,
        'n_states': n_states,
        'acf1_raw': float(acf_raw[1]),
        'acf1_norm': float(acf_norm[1]),
        'acf_raw': [float(x) for x in acf_raw],
        'acf_norm': [float(x) for x in acf_norm],
        'ljung_box_raw': lb_raw,
        'ljung_box_norm': lb_norm,
        'dxr2_oracle': [float(x) for x in dxr2_orc],
        'dxr2_no_forcing': [float(x) for x in dxr2_nof],
        'dxr2_10_oracle': float(dxr2_orc[9]),
        'dxr2_10_no_forcing': float(dxr2_nof[9]),
        'mean_dxr2_5_10_oracle': float(np.mean(dxr2_orc[4:10])),
        'mean_dxr2_5_10_no_forcing': float(np.mean(dxr2_nof[4:10])),
        'vof_pct': [float(x) for x in vof_pct],
        'vof_avg_5_10': vof_avg,
        'vof_ci_lo': [float(x) for x in ci_lo],
        'vof_ci_hi': [float(x) for x in ci_hi],
        'dr2': [float(x) for x in dr2],
        'phys_params': {k: float(v) for k, v in phys_params.items()},
        'cl_params': {k: float(v) for k, v in cl_params.items()},
        'mem_params': ({k: float(v) for k, v in mem_params.items()}
                       if mem_params else None),
    }


# ============================================================
#  3-STATE MODEL CLASS (copied from v6 for disturbance training)
# ============================================================

class KalmanForecasterClosure3State(nn.Module):
    """3-state KF: [x, u, m] with AR(1) memory."""

    def __init__(
        self,
        alpha_init=0.5, c_init=1.0, vc_init=0.15, kappa_init=0.1,
        log_qx_init=-6.0, log_qu_init=-6.0, log_r_init=-5.0,
        log_p0_xx_init=-6.0, log_p0_uu_init=-4.0,
        vc_min=0.0,
        a1_init=0.1, b1_init=0.0, b2_init=0.0,
        d1_init=0.05, d2_init=0.5, d3_init=0.5,
        beta_init=0.1, rho_m_init=0.9,
        log_qm_init=-4.0, log_p0_mm_init=-4.0,
    ):
        super().__init__()
        self.vc_min = vc_min
        a = max(min(alpha_init, 0.999), 0.001)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(a / (1.0 - a))))
        c_safe = max(c_init, 0.01)
        self.c_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(c_safe) - 1.0 + 1e-6)))
        k_safe = max(kappa_init, 0.001)
        self.kappa_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(k_safe) - 1.0 + 1e-6)))
        vc_eff = max(vc_init - vc_min, 0.01)
        self.vc_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(vc_eff) - 1.0 + 1e-6)))
        self.log_qx = nn.Parameter(torch.tensor(log_qx_init))
        self.log_qu = nn.Parameter(torch.tensor(log_qu_init))
        self.log_r = nn.Parameter(torch.tensor(log_r_init))
        self.log_q_scale = nn.Parameter(torch.tensor(0.0))
        self.log_p0_xx = nn.Parameter(torch.tensor(log_p0_xx_init))
        self.log_p0_uu = nn.Parameter(torch.tensor(log_p0_uu_init))

        def _sp_inv(x):
            return math.log(math.exp(max(x, 1e-4)) - 1.0 + 1e-6)

        self.a1_raw = nn.Parameter(torch.tensor(_sp_inv(a1_init)))
        self.d1_raw = nn.Parameter(torch.tensor(_sp_inv(d1_init)))
        self.d2_raw = nn.Parameter(torch.tensor(_sp_inv(d2_init)))
        self.d3_raw = nn.Parameter(torch.tensor(_sp_inv(d3_init)))
        self.b1 = nn.Parameter(torch.tensor(float(b1_init)))
        self.b2 = nn.Parameter(torch.tensor(float(b2_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.rho_m_raw = nn.Parameter(torch.tensor(
            math.atanh(max(min(rho_m_init, 0.999), -0.999))))
        self.log_qm = nn.Parameter(torch.tensor(log_qm_init))
        self.log_p0_mm = nn.Parameter(torch.tensor(log_p0_mm_init))

    @property
    def alpha(self): return torch.sigmoid(self.alpha_raw)
    @property
    def c(self): return F_nn.softplus(self.c_raw)
    @property
    def kappa(self): return F_nn.softplus(self.kappa_raw)
    @property
    def vc(self): return F_nn.softplus(self.vc_raw) + self.vc_min
    @property
    def qx(self): return torch.exp(self.log_qx)
    @property
    def qu(self): return torch.exp(self.log_qu)
    @property
    def q_scale(self): return torch.exp(self.log_q_scale)
    @property
    def R(self): return torch.exp(self.log_r)
    @property
    def rho_m(self): return torch.tanh(self.rho_m_raw)
    @property
    def qm(self): return torch.exp(self.log_qm)
    @property
    def a1(self): return F_nn.softplus(self.a1_raw)
    @property
    def d1(self): return F_nn.softplus(self.d1_raw)
    @property
    def d2(self): return F_nn.softplus(self.d2_raw)
    @property
    def d3(self): return F_nn.softplus(self.d3_raw)

    @property
    def P0(self):
        P = torch.zeros(3, 3, device=self.log_p0_xx.device,
                         dtype=self.log_p0_xx.dtype)
        P[0, 0] = torch.exp(self.log_p0_xx)
        P[1, 1] = torch.exp(self.log_p0_uu)
        P[2, 2] = torch.exp(self.log_p0_mm)
        return P

    def forcing(self, v):
        return F_nn.relu(v * v - self.vc * self.vc)

    def closure(self, u_state, v_water, dv_water):
        return (-self.a1 * u_state
                + self.b1 * v_water + self.b2 * dv_water
                - self.d1 * u_state ** 2
                - self.d2 * u_state * torch.abs(v_water)
                - self.d3 * u_state * torch.abs(u_state))

    def kf_predict(self, s, P, v_water, dv_water, dt,
                   collect_residuals=False):
        B = s.shape[0]
        rho = torch.exp(-self.alpha * dt)
        x_old, u_old, m_old = s[:, 0], s[:, 1], s[:, 2]
        x_pred = x_old + u_old * dt
        physics_drift = (rho * u_old - self.kappa * x_old * dt
                         + self.c * self.forcing(v_water) * dt)
        cl = self.closure(u_old, v_water, dv_water)
        cl_dt = cl * dt
        u_pred = physics_drift + cl_dt + self.beta * m_old
        m_pred = self.rho_m * m_old
        s_pred = torch.stack([x_pred, u_pred, m_pred], dim=1)
        F_mat = torch.zeros(B, 3, 3, device=s.device, dtype=s.dtype)
        F_mat[:, 0, 0] = 1.0; F_mat[:, 0, 1] = dt
        F_mat[:, 1, 0] = -self.kappa * dt; F_mat[:, 1, 1] = rho
        F_mat[:, 1, 2] = self.beta; F_mat[:, 2, 2] = self.rho_m
        qs = self.q_scale
        Q = torch.zeros(B, 3, 3, device=s.device, dtype=s.dtype)
        Q[:, 0, 0] = qs * self.qx * dt
        Q[:, 1, 1] = qs * self.qu * dt
        Q[:, 2, 2] = self.qm
        P_pred = torch.bmm(torch.bmm(F_mat, P), F_mat.transpose(1, 2)) + Q
        if collect_residuals:
            return s_pred, P_pred, cl_dt, physics_drift
        return s_pred, P_pred

    def kf_update(self, s_pred, P_pred, y_obs):
        R = self.R
        innov = y_obs - s_pred[:, 0]
        S = P_pred[:, 0, 0] + R
        K = P_pred[:, :, 0] / S.unsqueeze(1)
        s_upd = s_pred + K * innov.unsqueeze(1)
        eye = torch.eye(3, device=s_pred.device, dtype=s_pred.dtype).unsqueeze(0)
        H_vec = torch.tensor([1.0, 0.0, 0.0], device=s_pred.device, dtype=s_pred.dtype)
        KH = K.unsqueeze(2) * H_vec.unsqueeze(0).unsqueeze(0)
        IKH = eye - KH
        P_upd = (torch.bmm(torch.bmm(IKH, P_pred), IKH.transpose(1, 2))
                 + R * K.unsqueeze(2) * K.unsqueeze(1))
        return s_upd, P_upd

    def forward(self, v_hist, dt_hist, x_obs_hist, v_fut, dt_fut,
                collect_residuals=False):
        B, Lh = v_hist.shape
        Hf = v_fut.shape[1]
        dev = v_hist.device
        s = torch.zeros(B, 3, device=dev, dtype=v_hist.dtype)
        s[:, 0] = x_obs_hist[:, 0]
        P = self.P0.unsqueeze(0).expand(B, -1, -1).clone()
        all_cl = [] if collect_residuals else None
        all_ph = [] if collect_residuals else None
        for k in range(1, Lh):
            dt_k = dt_hist[:, k].clamp(min=1e-6)
            v_curr = v_hist[:, k - 1]
            v_prev = v_hist[:, k - 2] if k >= 2 else v_hist[:, 0]
            dv = v_curr - v_prev if k >= 2 else torch.zeros_like(v_curr)
            if collect_residuals:
                s, P, cl, phys = self.kf_predict(s, P, v_curr, dv, dt_k, True)
                all_cl.append(cl); all_ph.append(phys)
            else:
                s, P = self.kf_predict(s, P, v_curr, dv, dt_k)
            s, P = self.kf_update(s, P, x_obs_hist[:, k])
        x_preds, x_vars, u_ests = [], [], []
        for k in range(Hf):
            dt_k = dt_fut[:, k].clamp(min=1e-6)
            v_prev = v_hist[:, -1] if k == 0 else v_fut[:, k - 1]
            v_curr = v_fut[:, k]; dv = v_curr - v_prev
            if collect_residuals:
                s, P, cl, phys = self.kf_predict(s, P, v_curr, dv, dt_k, True)
                all_cl.append(cl); all_ph.append(phys)
            else:
                s, P = self.kf_predict(s, P, v_curr, dv, dt_k)
            x_preds.append(s[:, 0]); x_vars.append(P[:, 0, 0])
            u_ests.append(s[:, 1])
        result = (torch.stack(x_preds, dim=1),
                  torch.stack(x_vars, dim=1),
                  torch.stack(u_ests, dim=1))
        if collect_residuals:
            return result + (torch.stack(all_cl, dim=1),
                             torch.stack(all_ph, dim=1))
        return result

    def freeze_physics(self):
        for name in ['alpha_raw', 'c_raw', 'kappa_raw', 'vc_raw',
                      'log_r', 'log_qx', 'log_qu',
                      'log_p0_xx', 'log_p0_uu']:
            getattr(self, name).requires_grad_(False)

    def closure_and_memory_params_list(self):
        return [self.a1_raw, self.b1, self.b2,
                self.d1_raw, self.d2_raw, self.d3_raw,
                self.log_q_scale,
                self.beta, self.rho_m_raw, self.log_qm, self.log_p0_mm]

    def memory_summary(self):
        with torch.no_grad():
            return {
                'beta': self.beta.item(),
                'rho_m': self.rho_m.item(),
                'qm': self.qm.item(),
                'P0_mm': torch.exp(self.log_p0_mm).item(),
            }

    def closure_summary(self):
        with torch.no_grad():
            d = {
                'a1': self.a1.item(), 'b1': self.b1.item(),
                'b2': self.b2.item(), 'd1': self.d1.item(),
                'd2': self.d2.item(), 'd3': self.d3.item(),
                'q_scale': self.q_scale.item(),
            }
            d.update(self.memory_summary())
            return d

    def param_summary(self):
        with torch.no_grad():
            a = self.alpha.item()
            d = {
                'alpha': a,
                'tau': 1.0 / a if a > 1e-8 else float('inf'),
                'c': self.c.item(), 'vc': self.vc.item(),
                'kappa': self.kappa.item(),
                'qx': self.qx.item(), 'qu': self.qu.item(),
                'q_scale': self.q_scale.item(),
                'R': self.R.item(),
                'P0_xx': torch.exp(self.log_p0_xx).item(),
                'P0_uu': torch.exp(self.log_p0_uu).item(),
            }
            d.update(self.closure_summary())
            return d


# ============================================================
#  GENERIC TRAINING FUNCTION
# ============================================================

def train_stage(model, train_loader, val_loader, device,
                max_epochs, patience, lr, sched_patience=8,
                param_getter=None, tag=""):
    """Generic training loop with early stopping."""
    if param_getter:
        params = [p for p in param_getter() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        print(f"    [{tag}] WARNING: no trainable params!")
        return 0.0

    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=sched_patience)

    best_loss, best_state, wait = float('inf'), None, 0
    t0 = time.time()

    for ep in range(max_epochs):
        model.train()
        tot_nll, n = 0.0, 0
        for batch in train_loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            v_h = v_h.to(device); dt_h = dt_h.to(device)
            x_h = x_h.to(device); v_f = v_f.to(device)
            dt_f = dt_f.to(device); x_true = x_true.to(device)
            optimizer.zero_grad()
            xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
            loss = gaussian_nll(xp, xv, x_true, VAR_FLOOR)
            loss.backward(); optimizer.step()
            tot_nll += loss.item(); n += 1
        tr_nll = tot_nll / n

        model.eval()
        with torch.no_grad():
            vl_tot, vl_n = 0.0, 0
            for batch in val_loader:
                v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
                v_h = v_h.to(device); dt_h = dt_h.to(device)
                x_h = x_h.to(device); v_f = v_f.to(device)
                dt_f = dt_f.to(device); x_true = x_true.to(device)
                xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
                vl = gaussian_nll(xp, xv, x_true, VAR_FLOOR)
                vl_tot += vl.item(); vl_n += 1
            val_nll = vl_tot / vl_n

        scheduler.step(val_nll)
        if val_nll < best_loss:
            best_loss = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep+1) % 20 == 0 or ep == 0:
            print(f"    [{tag}] ep {ep+1:3d}  tr={tr_nll:.4f}  val={val_nll:.4f}")
        if wait >= patience:
            print(f"    [{tag}] Early stop at ep {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done {elapsed:.0f}s, best val={best_loss:.4f}")
    return best_loss


# ============================================================
#  MODEL BUILDERS
# ============================================================

def _freeze_closure_terms(model):
    """Freeze all closure terms to near-zero."""
    with torch.no_grad():
        model.a1_raw.fill_(-10.0); model.a1_raw.requires_grad_(False)
        model.b1.fill_(0.0); model.b1.requires_grad_(False)
        model.b2.fill_(0.0); model.b2.requires_grad_(False)
        model.d1_raw.fill_(-10.0); model.d1_raw.requires_grad_(False)
        model.d2_raw.fill_(-10.0); model.d2_raw.requires_grad_(False)
        model.d3_raw.fill_(-10.0); model.d3_raw.requires_grad_(False)
        model.log_q_scale.fill_(0.0); model.log_q_scale.requires_grad_(False)


def _freeze_closure_except_b2_d2(model):
    """For 2-state closure stage: freeze a1, b1, d1, d3."""
    with torch.no_grad():
        model.a1_raw.fill_(-10.0); model.a1_raw.requires_grad_(False)
        model.b1.fill_(0.0); model.b1.requires_grad_(False)
        model.d1_raw.fill_(-10.0); model.d1_raw.requires_grad_(False)
        model.d3_raw.fill_(-10.0); model.d3_raw.requires_grad_(False)


def build_fair_2state_physics_only(s1_params):
    """2-state physics-only model for fair Stage 1."""
    p = s1_params
    model = KalmanForecasterClosure(
        alpha_init=p['alpha'], c_init=p['c'],
        vc_init=p['vc'], kappa_init=p['kappa'],
        log_qx_init=math.log(p['qx']),
        log_qu_init=math.log(p['qu']),
        log_r_init=math.log(p['R']),
        log_p0_xx_init=math.log(p['P0_xx']),
        log_p0_uu_init=math.log(p['P0_uu']),
        a1_init=0.001, b1_init=0.0, b2_init=0.0,
        d1_init=0.001, d2_init=0.001, d3_init=0.001,
    )
    _freeze_closure_terms(model)
    return model


def build_fair_2state_closure_2t(fair_s1_params):
    """2-state closure model from fair S1 physics."""
    p = fair_s1_params
    model = KalmanForecasterClosure(
        alpha_init=p['alpha'], c_init=p['c'],
        vc_init=p['vc'], kappa_init=p['kappa'],
        log_qx_init=math.log(p['qx']),
        log_qu_init=math.log(p['qu']),
        log_r_init=math.log(p['R']),
        log_p0_xx_init=math.log(p['P0_xx']),
        log_p0_uu_init=math.log(p['P0_uu']),
        a1_init=0.01, b1_init=0.0, b2_init=6.34,
        d1_init=0.01, d2_init=10.46, d3_init=0.01,
    )
    model.freeze_physics()
    _freeze_closure_except_b2_d2(model)
    return model


def build_disturbance_physics_only(s1_params):
    """3-state with beta=1 fixed, for Stage 1."""
    p = s1_params
    model = KalmanForecasterClosure3State(
        alpha_init=p['alpha'], c_init=p['c'],
        vc_init=p['vc'], kappa_init=p['kappa'],
        log_qx_init=math.log(p['qx']),
        log_qu_init=math.log(p['qu']),
        log_r_init=math.log(p['R']),
        log_p0_xx_init=math.log(p['P0_xx']),
        log_p0_uu_init=math.log(p['P0_uu']),
        a1_init=0.001, b1_init=0.0, b2_init=0.0,
        d1_init=0.001, d2_init=0.001, d3_init=0.001,
        beta_init=1.0,
        rho_m_init=0.9,
        log_qm_init=-4.0,
        log_p0_mm_init=-4.0,
    )
    # Fix beta=1 non-trainable
    with torch.no_grad():
        model.beta.fill_(1.0)
    model.beta.requires_grad_(False)
    _freeze_closure_terms(model)
    return model


def build_disturbance_closure_2t(s1_dist_params, mem_init):
    """3-state closure with beta=1 fixed, from disturbance S1 params."""
    p = s1_dist_params
    model = KalmanForecasterClosure3State(
        alpha_init=p['alpha'], c_init=p['c'],
        vc_init=p['vc'], kappa_init=p['kappa'],
        log_qx_init=math.log(p['qx']),
        log_qu_init=math.log(p['qu']),
        log_r_init=math.log(p['R']),
        log_p0_xx_init=math.log(p['P0_xx']),
        log_p0_uu_init=math.log(p['P0_uu']),
        a1_init=0.01, b1_init=0.0, b2_init=6.34,
        d1_init=0.01, d2_init=10.46, d3_init=0.01,
        beta_init=1.0,
        rho_m_init=mem_init.get('rho_m', 0.9),
        log_qm_init=math.log(max(mem_init.get('qm', 0.01), 1e-10)),
        log_p0_mm_init=math.log(max(mem_init.get('P0_mm', 0.01), 1e-10)),
    )
    model.freeze_physics()
    with torch.no_grad():
        model.beta.fill_(1.0)
    model.beta.requires_grad_(False)
    _freeze_closure_except_b2_d2(model)
    # b2, d2_raw, log_q_scale, rho_m_raw, log_qm, log_p0_mm trainable
    return model


# ============================================================
#  MAIN
# ============================================================

def main():
    t0_global = time.time()
    device = torch.device('cpu')
    print("="*70)
    print("V6 CONTROLS & ABLATIONS")
    print("="*70)
    print(f"Output -> {OUT}")

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------
    print_section("DATA LOADING")

    # Load original 2-state params
    ck_2s = torch.load(S1_CKPT_2S, map_location=device, weights_only=False)
    s1_params_2s = ck_2s['params']
    print(f"  Original 2-state S1: alpha={s1_params_2s['alpha']:.4f} "
          f"c={s1_params_2s['c']:.4f} kappa={s1_params_2s['kappa']:.4f}")

    # Load original 2-state closure (averaged from v2)
    cl_2s_avg = {}
    for key in ['b2', 'd2', 'q_scale']:
        vals = []
        for s in SEEDS:
            ck = torch.load(V2_CKPT / f"closure_2t_s{s}.pth",
                            map_location='cpu', weights_only=False)
            vals.append(ck['closure'][key])
        cl_2s_avg[key] = float(np.mean(vals))
    cl_old_2s = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl_old_2s['q_scale'] = cl_2s_avg['q_scale']
    cl_old_2s['b2'] = cl_2s_avg['b2']
    cl_old_2s['d2'] = cl_2s_avg['d2']
    print(f"  Original closure: b2={cl_old_2s['b2']:.3f} d2={cl_old_2s['d2']:.3f}")

    # Load v6 3-state params
    ck_v6_s1 = torch.load(V6_DIR / "checkpoints" / "stage1_3state.pth",
                          map_location='cpu', weights_only=False)
    s1_3state = ck_v6_s1['params']
    mem_v6_s1 = ck_v6_s1['memory']
    print(f"  V6 3-state S1: alpha={s1_3state['alpha']:.4f} "
          f"c={s1_3state['c']:.4f} kappa={s1_3state['kappa']:.4f}")

    # Average v6 closure across seeds
    cl_v6_avg = {}
    mem_v6_avg = {}
    for seed in SEEDS:
        ck = torch.load(V6_DIR / "checkpoints" / f"closure_3state_s{seed}.pth",
                        map_location='cpu', weights_only=False)
        for key in ['b2', 'd2', 'q_scale']:
            cl_v6_avg.setdefault(key, []).append(ck['closure'][key])
        for key in ['beta', 'rho_m', 'qm', 'P0_mm']:
            mem_v6_avg.setdefault(key, []).append(ck['memory'][key])
    cl_ref_3s = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    for key in cl_v6_avg:
        cl_ref_3s[key] = float(np.mean(cl_v6_avg[key]))
    mem_ref_3s = {key: float(np.mean(vals)) for key, vals in mem_v6_avg.items()}
    print(f"  V6 closure: b2={cl_ref_3s['b2']:.3f} d2={cl_ref_3s['d2']:.3f} "
          f"beta={mem_ref_3s['beta']:.4f} rho_m={mem_ref_3s['rho_m']:.4f}")

    # Data splits
    df_train = pd.read_csv(DATA_DIR / "train_10hz_ready.csv")
    df_val   = pd.read_csv(DATA_DIR / "val_10hz_ready.csv")
    df_test  = pd.read_csv(DATA_DIR / "test_10hz_ready.csv")
    TEST_START = df_test['timestamp'].iloc[0]

    warmup_sec = 50.0
    warmup_start_test = df_val['timestamp'].max() - warmup_sec
    test_warmup = df_val[df_val['timestamp'] >= warmup_start_test].copy()
    df_test_eval = pd.concat([test_warmup, df_test], ignore_index=True)
    test_mask = df_test_eval['timestamp'].values >= TEST_START
    test_eval_start = int(np.argmax(test_mask))

    t_arr = df_test_eval['timestamp'].values
    x_arr = df_test_eval['displacement'].values
    v_arr = df_test_eval['velocity'].values

    # Dev split for training validation
    dev_path = OUT / "dev_10hz_ready.csv"
    df_dev = df_val[df_val['timestamp'] < TEST_START].copy()
    df_dev.to_csv(dev_path, index=False)

    # DataLoaders
    train_ds = StateSpaceDataset(
        [str(DATA_DIR / "train_10hz_ready.csv")], L=L, m=L, H=H,
        predict_deltas=False, normalize=False)
    val_ds = StateSpaceDataset(
        [str(dev_path)], L=L, m=L, H=H,
        predict_deltas=False, normalize=False,
        run_id_to_idx=train_ds.run_id_to_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=0)
    print(f"  Train: {len(train_ds)}, Val(dev): {len(val_ds)}")

    # ============================================================
    # PART A: FAIR 2-STATE BASELINE (retrain with v6 pipeline)
    # ============================================================
    print_section("PART A: FAIR 2-STATE BASELINE")

    # A1: Stage 1 - retrain physics
    print("  A1: Stage 1 - physics only")
    torch.manual_seed(42); np.random.seed(42)
    model_fair_s1 = build_fair_2state_physics_only(s1_params_2s)
    model_fair_s1 = model_fair_s1.to(device)
    train_stage(model_fair_s1, train_loader, val_loader, device,
                S1_EPOCHS, S1_PATIENCE, S1_LR, tag="fair_2s_S1")

    fair_s1_params = model_fair_s1.param_summary()
    print(f"  Fair 2-state S1: alpha={fair_s1_params['alpha']:.4f} "
          f"c={fair_s1_params['c']:.4f} kappa={fair_s1_params['kappa']:.4f}")
    torch.save({
        'state_dict': model_fair_s1.state_dict(),
        'params': fair_s1_params,
    }, OUT / "fair_2state" / "stage1_fair.pth")

    # A2: Stage 2 - closure (3 seeds)
    print("\n  A2: Stage 2 - closure (b2+d2+q_scale)")
    fair_cl_results = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_fair_2state_closure_2t(fair_s1_params)
        model = model.to(device)
        tag = f"fair_2s_s{seed}"
        train_stage(model, train_loader, val_loader, device,
                    S2_EPOCHS, S2_PATIENCE, S2_LR, sched_patience=10,
                    param_getter=model.closure_params_list, tag=tag)
        cs = model.closure_summary()
        torch.save({
            'state_dict': model.state_dict(),
            'params': fair_s1_params,
            'closure': cs,
            'seed': seed,
        }, OUT / "fair_2state" / f"closure_fair_s{seed}.pth")
        fair_cl_results.append(cs)
        print(f"    Seed {seed}: b2={cs['b2']:.4f} d2={cs['d2']:.4f} "
              f"q_scale={cs['q_scale']:.3f}")

    # Average fair closure
    cl_fair_2s = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    for key in ['b2', 'd2', 'q_scale']:
        cl_fair_2s[key] = float(np.mean([r[key] for r in fair_cl_results]))
    print(f"  Fair averaged: b2={cl_fair_2s['b2']:.3f} d2={cl_fair_2s['d2']:.3f} "
          f"q_scale={cl_fair_2s['q_scale']:.3f}")

    # ============================================================
    # PART D: FIXED-COUPLING DISTURBANCE (beta=1 fixed)
    # ============================================================
    print_section("PART D: DISTURBANCE MODEL (beta=1 fixed)")

    # D1: Stage 1 - physics + memory (beta=1 fixed)
    print("  D1: Stage 1 - physics + memory (beta=1 fixed)")
    torch.manual_seed(42); np.random.seed(42)
    model_dist_s1 = build_disturbance_physics_only(s1_params_2s)
    model_dist_s1 = model_dist_s1.to(device)
    train_stage(model_dist_s1, train_loader, val_loader, device,
                S1_EPOCHS, S1_PATIENCE, S1_LR, tag="dist_S1")

    dist_s1_params = model_dist_s1.param_summary()
    dist_mem_s1 = model_dist_s1.memory_summary()
    print(f"  Disturbance S1: alpha={dist_s1_params['alpha']:.4f} "
          f"c={dist_s1_params['c']:.4f} kappa={dist_s1_params['kappa']:.4f}")
    print(f"    beta={dist_mem_s1['beta']:.4f} rho_m={dist_mem_s1['rho_m']:.4f} "
          f"qm={dist_mem_s1['qm']:.6f}")
    torch.save({
        'state_dict': model_dist_s1.state_dict(),
        'params': dist_s1_params,
        'memory': dist_mem_s1,
    }, OUT / "disturbance_fix1" / "stage1_disturbance.pth")

    # D2: Stage 2 - closure + memory (beta=1 fixed, 3 seeds)
    print("\n  D2: Stage 2 - closure + memory (beta=1 fixed)")
    dist_cl_results = []
    dist_mem_results = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_disturbance_closure_2t(dist_s1_params, dist_mem_s1)
        model = model.to(device)
        tag = f"dist_s{seed}"
        train_stage(model, train_loader, val_loader, device,
                    S2_EPOCHS, S2_PATIENCE, S2_LR, sched_patience=10,
                    param_getter=model.closure_and_memory_params_list,
                    tag=tag)
        cs = model.closure_summary()
        ms = model.memory_summary()
        torch.save({
            'state_dict': model.state_dict(),
            'params': dist_s1_params,
            'closure': cs,
            'memory': ms,
            'seed': seed,
        }, OUT / "disturbance_fix1" / f"closure_dist_s{seed}.pth")
        dist_cl_results.append(cs)
        dist_mem_results.append(ms)
        print(f"    Seed {seed}: b2={cs['b2']:.4f} d2={cs['d2']:.4f} "
              f"q_scale={cs['q_scale']:.3f} rho_m={ms['rho_m']:.4f} "
              f"qm={ms['qm']:.6f}")

    # Average disturbance closure + memory
    cl_dist = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    for key in ['b2', 'd2', 'q_scale']:
        cl_dist[key] = float(np.mean([r[key] for r in dist_cl_results]))
    mem_dist = {}
    for key in ['beta', 'rho_m', 'qm', 'P0_mm']:
        mem_dist[key] = float(np.mean([r[key] for r in dist_mem_results]))
    print(f"  Dist averaged: b2={cl_dist['b2']:.3f} d2={cl_dist['d2']:.3f} "
          f"q_scale={cl_dist['q_scale']:.3f}")
    print(f"    beta={mem_dist['beta']:.4f} rho_m={mem_dist['rho_m']:.4f} "
          f"qm={mem_dist['qm']:.6f}")

    # ============================================================
    # EVALUATE ALL 6 VARIANTS
    # ============================================================
    print_section("EVALUATING ALL 6 VARIANTS")

    all_results = {}

    # 1. old_2state
    r = evaluate_variant('old_2state', 2, s1_params_2s, cl_old_2s, None,
                         t_arr, x_arr, v_arr, test_mask, test_eval_start)
    all_results['old_2state'] = r
    with open(OUT / "comparison" / "summary_old_2state.json", 'w') as f:
        json.dump(r, f, indent=2)

    # 2. fair_2state
    r = evaluate_variant('fair_2state', 2, fair_s1_params, cl_fair_2s, None,
                         t_arr, x_arr, v_arr, test_mask, test_eval_start)
    all_results['fair_2state'] = r
    with open(OUT / "fair_2state" / "summary_fair_2state.json", 'w') as f:
        json.dump(r, f, indent=2)

    # 3. v6_3state
    r = evaluate_variant('v6_3state', 3, s1_3state, cl_ref_3s, mem_ref_3s,
                         t_arr, x_arr, v_arr, test_mask, test_eval_start)
    all_results['v6_3state'] = r
    with open(OUT / "comparison" / "summary_v6_3state.json", 'w') as f:
        json.dump(r, f, indent=2)

    # 4. v6_3state_beta0
    mem_beta0 = dict(mem_ref_3s); mem_beta0['beta'] = 0.0
    r = evaluate_variant('v6_3state_beta0', 3, s1_3state, cl_ref_3s, mem_beta0,
                         t_arr, x_arr, v_arr, test_mask, test_eval_start)
    all_results['v6_3state_beta0'] = r
    with open(OUT / "ablations" / "summary_3state_beta0.json", 'w') as f:
        json.dump(r, f, indent=2)

    # 5. v6_3state_rhom0
    mem_rhom0 = dict(mem_ref_3s); mem_rhom0['rho_m'] = 0.0
    r = evaluate_variant('v6_3state_rhom0', 3, s1_3state, cl_ref_3s, mem_rhom0,
                         t_arr, x_arr, v_arr, test_mask, test_eval_start)
    all_results['v6_3state_rhom0'] = r
    with open(OUT / "ablations" / "summary_3state_rhom0.json", 'w') as f:
        json.dump(r, f, indent=2)

    # 6. disturbance_fix1
    r = evaluate_variant('disturbance_fix1', 3, dist_s1_params, cl_dist, mem_dist,
                         t_arr, x_arr, v_arr, test_mask, test_eval_start)
    all_results['disturbance_fix1'] = r
    with open(OUT / "disturbance_fix1" / "summary_3state_disturbance_fixed1.json", 'w') as f:
        json.dump(r, f, indent=2)

    # ============================================================
    # DIAGNOSTICS TABLE (all models, raw + normalized)
    # ============================================================
    print_section("DIAGNOSTICS SUMMARY")

    diag_rows = []
    for name, res in all_results.items():
        row = {
            'model': name,
            'acf1_raw': res['acf1_raw'],
            'acf1_norm': res['acf1_norm'],
            'dxr2_10_oracle': res['dxr2_10_oracle'],
            'dxr2_10_no_forcing': res['dxr2_10_no_forcing'],
            'mean_dxr2_5_10': res['mean_dxr2_5_10_oracle'],
            'vof_avg_5_10': res['vof_avg_5_10'],
        }
        # Add Ljung-Box Q at key lags (raw + norm)
        for lb_entry in res['ljung_box_raw']:
            row[f"LB_Q_raw_lag{lb_entry['lag']}"] = lb_entry['Q']
        for lb_entry in res['ljung_box_norm']:
            row[f"LB_Q_norm_lag{lb_entry['lag']}"] = lb_entry['Q']
        diag_rows.append(row)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT / "diagnostics" / "all_models_diagnostics.csv",
                   index=False, float_format='%.4f')

    # Print comparison table
    print(f"\n  {'Model':<22} {'ACF1_raw':>9} {'ACF1_norm':>10} "
          f"{'DxR2@10':>8} {'DxR2@10nf':>10} {'mean5-10':>9} {'VoF%5-10':>9}")
    print(f"  {'-'*85}")
    for name in ['old_2state', 'fair_2state', 'v6_3state',
                  'v6_3state_beta0', 'v6_3state_rhom0', 'disturbance_fix1']:
        r = all_results[name]
        print(f"  {name:<22} {r['acf1_raw']:>9.4f} {r['acf1_norm']:>10.4f} "
              f"{r['dxr2_10_oracle']:>+8.4f} {r['dxr2_10_no_forcing']:>+10.4f} "
              f"{r['mean_dxr2_5_10_oracle']:>+9.4f} {r['vof_avg_5_10']:>+9.3f}")

    # ============================================================
    # PART F: DECISION TABLE
    # ============================================================
    print_section("PART F: DECISION TABLE")

    old = all_results['old_2state']
    fair = all_results['fair_2state']
    v6 = all_results['v6_3state']
    b0 = all_results['v6_3state_beta0']
    rm0 = all_results['v6_3state_rhom0']
    dist = all_results['disturbance_fix1']

    # Decision 1: Pipeline vs memory
    fair_matches_v6 = abs(fair['acf1_raw'] - v6['acf1_raw']) < 0.05
    fair_acf_improved = fair['acf1_raw'] < old['acf1_raw'] - 0.02
    fair_dxr2_improved = fair['dxr2_10_oracle'] > old['dxr2_10_oracle'] + 0.02

    # Decision 2: Beta knockout
    beta0_delta_acf = abs(v6['acf1_raw'] - b0['acf1_raw'])
    beta0_delta_dxr2 = abs(v6['dxr2_10_oracle'] - b0['dxr2_10_oracle'])
    beta0_no_effect = beta0_delta_acf < 0.01 and beta0_delta_dxr2 < 0.01

    # Decision 3: Rho_m knockout
    rhom0_delta_acf = abs(v6['acf1_raw'] - rm0['acf1_raw'])
    rhom0_no_effect = rhom0_delta_acf < 0.01

    # Decision 4: Disturbance model
    dist_acf_better = dist['acf1_raw'] < v6['acf1_raw'] - 0.05
    dist_acf_below_04 = dist['acf1_raw'] < 0.4
    dist_skill_ok = dist['dxr2_10_oracle'] >= old['dxr2_10_oracle'] - 0.05
    dist_vof_ok = abs(dist['vof_avg_5_10']) < 3.0

    decision = {
        'pipeline_vs_memory': {
            'fair_matches_v6_acf': fair_matches_v6,
            'fair_improved_acf': fair_acf_improved,
            'fair_improved_dxr2': fair_dxr2_improved,
            'fair_acf1': fair['acf1_raw'],
            'v6_acf1': v6['acf1_raw'],
            'old_acf1': old['acf1_raw'],
            'conclusion': ('Pipeline accounts for most gains'
                           if fair_matches_v6
                           else 'Memory adds value beyond pipeline'),
        },
        'beta_knockout': {
            'delta_acf': beta0_delta_acf,
            'delta_dxr2': beta0_delta_dxr2,
            'no_effect': beta0_no_effect,
            'conclusion': ('Memory coupling inactive (beta~0 confirmed)'
                           if beta0_no_effect
                           else 'Memory coupling has measurable effect'),
        },
        'rhom_knockout': {
            'delta_acf': rhom0_delta_acf,
            'no_effect': rhom0_no_effect,
            'conclusion': ('Memory persistence irrelevant'
                           if rhom0_no_effect
                           else 'Memory persistence matters'),
        },
        'disturbance_model': {
            'acf1': dist['acf1_raw'],
            'acf_better_than_v6': dist_acf_better,
            'acf_below_0.4': dist_acf_below_04,
            'skill_ok': dist_skill_ok,
            'vof_ok': dist_vof_ok,
            'dxr2_10': dist['dxr2_10_oracle'],
            'vof_avg': dist['vof_avg_5_10'],
            'conclusion': ('Fixed-coupling disturbance is real and defensible'
                           if (dist_acf_better or dist_acf_below_04) and dist_skill_ok
                           else 'Fixed-coupling disturbance does not add material value'),
        },
        'overall': '',
        'physics_comparison': {
            'old_alpha': s1_params_2s['alpha'],
            'fair_alpha': fair_s1_params['alpha'],
            'v6_alpha': s1_3state['alpha'],
            'dist_alpha': dist_s1_params['alpha'],
            'old_kappa': s1_params_2s['kappa'],
            'fair_kappa': fair_s1_params['kappa'],
            'v6_kappa': s1_3state['kappa'],
            'dist_kappa': dist_s1_params['kappa'],
            'old_c': s1_params_2s['c'],
            'fair_c': fair_s1_params['c'],
            'v6_c': s1_3state['c'],
            'dist_c': dist_s1_params['c'],
        },
    }

    # Overall verdict
    if fair_matches_v6:
        decision['overall'] = ('PIPELINE EFFECT: Fair 2-state matches v6 3-state. '
                               'Improvements are from retraining, not memory.')
    elif dist_acf_better and dist_skill_ok:
        decision['overall'] = ('MEMORY IS REAL: Fixed-coupling disturbance materially '
                               'reduces ACF while maintaining skill. Option 2 defensible.')
    else:
        decision['overall'] = ('INCONCLUSIVE: Some gains from pipeline, disturbance model '
                               'does not clearly outperform. Consider discarding Option 2.')

    with open(OUT / "comparison" / "decision_table.json", 'w') as f:
        json.dump(decision, f, indent=2)

    print("\n  DECISION RULES:")
    print(f"    1. Pipeline vs Memory: {decision['pipeline_vs_memory']['conclusion']}")
    print(f"       fair ACF1={fair['acf1_raw']:.4f}  v6 ACF1={v6['acf1_raw']:.4f}  "
          f"old ACF1={old['acf1_raw']:.4f}")
    print(f"    2. Beta knockout: {decision['beta_knockout']['conclusion']}")
    print(f"       delta ACF={beta0_delta_acf:.6f}  delta DxR2={beta0_delta_dxr2:.6f}")
    print(f"    3. Rho_m knockout: {decision['rhom_knockout']['conclusion']}")
    print(f"       delta ACF={rhom0_delta_acf:.6f}")
    print(f"    4. Disturbance model: {decision['disturbance_model']['conclusion']}")
    print(f"       ACF1={dist['acf1_raw']:.4f}  DxR2@10={dist['dxr2_10_oracle']:+.4f}  "
          f"VoF%={dist['vof_avg_5_10']:+.3f}")

    print(f"\n  PHYSICS PARAMETER COMPARISON:")
    print(f"    {'Param':<10} {'Old_2S':>8} {'Fair_2S':>8} {'V6_3S':>8} {'Dist_3S':>8}")
    print(f"    {'-'*44}")
    for pname in ['alpha', 'kappa', 'c']:
        pc = decision['physics_comparison']
        print(f"    {pname:<10} {pc[f'old_{pname}']:>8.4f} {pc[f'fair_{pname}']:>8.4f} "
              f"{pc[f'v6_{pname}']:>8.4f} {pc[f'dist_{pname}']:>8.4f}")

    print(f"\n  OVERALL VERDICT: {decision['overall']}")

    # ============================================================
    # FIGURES
    # ============================================================
    print_section("GENERATING FIGURES")

    # Figure 1: ACF comparison (6 models, raw)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    model_names = ['old_2state', 'fair_2state', 'v6_3state',
                   'v6_3state_beta0', 'v6_3state_rhom0', 'disturbance_fix1']
    titles = ['Old 2-State', 'Fair 2-State', 'V6 3-State',
              '3-State beta=0', '3-State rho_m=0', 'Disturbance (beta=1)']
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
    n_valid_ref = int(test_mask.sum())
    sig_band = 1.96 / math.sqrt(n_valid_ref)

    for idx, (name, title, color) in enumerate(zip(model_names, titles, colors)):
        ax = axes[idx // 3, idx % 3]
        acf_vals = all_results[name]['acf_raw']
        ax.bar(np.arange(1, 51), acf_vals[1:], color=color, alpha=0.7,
               edgecolor='black', linewidth=0.3)
        ax.axhline(sig_band, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(-sig_band, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
        ax.set_title(f'{title}\nACF(1)={acf_vals[1]:.3f}')
        ax.set_xlim(0, 51)

    fig.suptitle('Innovation ACF: All Models (Raw)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_acf_all_models_raw.png", bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_acf_all_models_raw.png")

    # Figure 2: ACF comparison (normalized)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for idx, (name, title, color) in enumerate(zip(model_names, titles, colors)):
        ax = axes[idx // 3, idx % 3]
        acf_vals = all_results[name]['acf_norm']
        ax.bar(np.arange(1, 51), acf_vals[1:], color=color, alpha=0.7,
               edgecolor='black', linewidth=0.3)
        ax.axhline(sig_band, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(-sig_band, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
        ax.set_title(f'{title}\nACF(1)={acf_vals[1]:.3f}')
        ax.set_xlim(0, 51)

    fig.suptitle('Innovation ACF: All Models (Normalized)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_acf_all_models_norm.png", bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_acf_all_models_norm.png")

    # Figure 3: DxR2 curves comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    h_vals = np.arange(1, MAX_HORIZON + 1) * DT
    for name, title, color, ls in [
        ('old_2state', 'Old 2-State', '#d62728', '--'),
        ('fair_2state', 'Fair 2-State', '#ff7f0e', '-'),
        ('v6_3state', 'V6 3-State', '#1f77b4', '-'),
        ('disturbance_fix1', 'Disturbance (beta=1)', '#8c564b', '-'),
    ]:
        ax.plot(h_vals, all_results[name]['dxr2_oracle'],
                color=color, ls=ls, lw=2, marker='o', ms=4, label=f'{title} (oracle)')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Lead time (s)'); ax.set_ylabel('DxR2(h)')
    ax.set_title('DxR2(h) Oracle: Key Models')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_dxr2_comparison.png", bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_dxr2_comparison.png")

    # Figure 4: Physics parameter comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ['Old 2S', 'Fair 2S', 'V6 3S', 'Dist 3S']
    pc = decision['physics_comparison']
    for ax, pname, ylabel in [
        (axes[0], 'alpha', 'alpha (damping rate)'),
        (axes[1], 'kappa', 'kappa (restoring force)'),
        (axes[2], 'c', 'c (forcing coefficient)'),
    ]:
        vals = [pc[f'old_{pname}'], pc[f'fair_{pname}'],
                pc[f'v6_{pname}'], pc[f'dist_{pname}']]
        ax.bar(labels, vals, color=['#d62728', '#ff7f0e', '#1f77b4', '#8c564b'],
               alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel(ylabel); ax.set_title(pname)
    fig.suptitle('Physics Parameters: All Pipelines', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_physics_comparison.png", bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_physics_comparison.png")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print_section("FINAL SUMMARY")

    elapsed_total = time.time() - t0_global
    print(f"  Total runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"  Output: {OUT}")

    # List files
    print(f"\n  Output files:")
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            print(f"    {p.relative_to(OUT)}")

    # Save master results
    master = {
        'all_results': {k: {kk: vv for kk, vv in v.items()
                            if kk not in ['acf_raw', 'acf_norm', 'vof_ci_lo',
                                          'vof_ci_hi', 'phys_params', 'cl_params',
                                          'mem_params', 'ljung_box_raw',
                                          'ljung_box_norm', 'dxr2_oracle',
                                          'dxr2_no_forcing', 'vof_pct', 'dr2']}
                        for k, v in all_results.items()},
        'decision': decision,
        'runtime_s': elapsed_total,
    }
    with open(OUT / "comparison" / "master_results.json", 'w') as f:
        json.dump(master, f, indent=2)
    print("\n  Saved master_results.json")


if __name__ == '__main__':
    main()
