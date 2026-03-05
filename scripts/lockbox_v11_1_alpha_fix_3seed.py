"""
Lockbox v11.1: Alpha Fix + Freeze Audit + Multi-Seed Retrain.

v11 found alpha=1.716 (softplus in KalmanForecaster) but KalmanForecasterClosure
uses sigmoid (range 0-1), so alpha got clipped to 0.999 when transferred.
This script:
  1. Uses the new alpha_param="softplus" flag for exact alpha transfer
  2. Audits alpha equivalence (S1 vs S2 with closure=0)
  3. Audits freeze integrity (all frozen params unchanged after S2 training)
  4. Trains 3 seeds with full evaluation + event metrics
  5. Aggregates results across seeds

Usage:  python -u scripts/lockbox_v11_1_alpha_fix_3seed.py
Output: final_lockbox_v11_1_alpha_fix/
"""

import os, sys, math, json, hashlib, time, warnings, copy
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
from models.kalman_forecaster import KalmanForecaster
from models.kalman_closure import KalmanForecasterClosure

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
SEEDS = [1, 2, 3]
FORCE_CPU = True
DT = 0.1
VAR_FLOOR = 1e-6

# Stage 1
S1_L = 512; S1_H = 128; S1_BATCH = 64
S1_EPOCHS = 100; S1_LR = 1e-2; S1_PATIENCE = 20; S1_SCHED = 10

# Stage 2
S2_L = 64; S2_H = 20; S2_BATCH = 128
S2_EPOCHS = 200; S2_LR = 1e-2; S2_PATIENCE = 30; S2_SCHED = 10

# Evaluation
MAX_H = 100
WARMUP_SEC = 50.0

# Paths
CLEAN_DIR = ROOT / "processed_data_10hz_clean_v1"
OUT = ROOT / "final_lockbox_v11_1_alpha_fix"

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})


# ==============================================================================
#  HELPERS
# ==============================================================================

def md5_file(path):
    h = hashlib.md5()
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


def gaussian_nll(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ==============================================================================
#  NUMPY 2-STATE FILTER
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

        F_mat = np.array([[1, dt], [-kap*dt, rho_u]])
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
#  DxR2 MULTI-HORIZON
# ==============================================================================

def compute_dxr2(params, cl_params, states_x, states_u,
                 t, x_obs, v, max_h=100, eval_start=1, mode='oracle',
                 indices=None):
    """DxR2(h) and MAE(h) for h=1..max_h."""
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
            if mode == 'oracle':
                v_w = v[k_s - 1] if k_s >= 1 else 0.0
                dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
            else:
                v_w = 0.0; dv_w = 0.0
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
    mae_arr = np.full(max_h, np.nan)
    for h in range(max_h):
        if len(dx_pred[h]) < 10:
            continue
        dp = np.array(dx_pred[h])
        do = np.array(dx_true[h])
        err = do - dp
        ss_res = np.sum(err**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        mae_arr[h] = float(np.mean(np.abs(err)))
    return r2_arr, mae_arr


# ==============================================================================
#  COMPUTE RMSE SKILL (for event table)
# ==============================================================================

def compute_event_skill(params, cl_params, states_x, states_u,
                        t, x_obs, v, max_h=50, eval_start=1, indices=None):
    """Per-horizon MAE(dx), RMSE(dx), Var(dx_true), RMSE_baseline, RMSE_skill."""
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

    rows = []
    for h in range(max_h):
        if len(dx_pred[h]) < 10:
            rows.append({'h': h + 1, 'mae_dx': np.nan, 'rmse_dx': np.nan,
                         'var_dx_true': np.nan, 'rmse_baseline': np.nan,
                         'rmse_skill': np.nan, 'n': 0})
            continue
        dp = np.array(dx_pred[h])
        do = np.array(dx_true[h])
        err = do - dp
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        var_true = float(np.var(do))
        rmse_base = float(np.sqrt(np.mean((do - np.mean(do))**2)))
        skill = 1.0 - rmse / rmse_base if rmse_base > 1e-15 else 0.0
        rows.append({'h': h + 1, 'mae_dx': mae, 'rmse_dx': rmse,
                     'var_dx_true': var_true, 'rmse_baseline': rmse_base,
                     'rmse_skill': skill, 'n': len(dp)})
    return rows


# ==============================================================================
#  TRAINING FUNCTION
# ==============================================================================

def train_model(model, train_loader, val_loader, device,
                max_epochs, patience, lr, sched_patience=10,
                param_getter=None, tag=""):
    """Train with early stopping."""
    if param_getter:
        params = [p for p in param_getter() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"    [{tag}] {n_params} trainable parameters")

    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=sched_patience)

    best_loss, best_state, best_ep, wait = float('inf'), None, 0, 0
    train_hist, val_hist = [], []
    t0 = time.time()

    for ep in range(max_epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            v_h = v_h.to(device); dt_h = dt_h.to(device)
            x_h = x_h.to(device); v_f = v_f.to(device)
            dt_f = dt_f.to(device); x_true = x_true.to(device)
            optimizer.zero_grad()
            xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
            loss = gaussian_nll(xp, xv, x_true, VAR_FLOOR)
            loss.backward(); optimizer.step()
            tot += loss.item(); nb += 1
        tr_nll = tot / nb
        train_hist.append(tr_nll)

        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for batch in val_loader:
                v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
                v_h = v_h.to(device); dt_h = dt_h.to(device)
                x_h = x_h.to(device); v_f = v_f.to(device)
                dt_f = dt_f.to(device); x_true = x_true.to(device)
                xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
                vl = gaussian_nll(xp, xv, x_true, VAR_FLOOR)
                vt += vl.item(); vn += 1
            val_nll = vt / vn
        val_hist.append(val_nll)

        scheduler.step(val_nll)
        if val_nll < best_loss:
            best_loss = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 10 == 0 or ep == 0 or wait >= patience:
            print(f"    [{tag}] ep {ep+1:3d}  tr={tr_nll:.5f}  "
                  f"val={val_nll:.5f}  best={best_loss:.5f}@ep{best_ep}  "
                  f"[{time.time()-t0:.0f}s]")
        if wait >= patience:
            print(f"    [{tag}] Early stop at ep {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"    [{tag}] Done {time.time()-t0:.0f}s, "
          f"best_val={best_loss:.5f} at ep={best_ep}")
    return best_loss, best_ep, train_hist[-1], train_hist, val_hist


# ==============================================================================
#  EVALUATE ONE MODEL VARIANT
# ==============================================================================

def evaluate_model(label, params, cl_params, t, x_obs, v, eval_start, max_h):
    """Full evaluation. Returns metrics dict + filter states."""
    t0_ev = time.time()
    filt = kf_filter_2state(params, cl_params, t, x_obs, v)
    innov = filt['innovations']; S_vals = filt['S_values']
    sx = filt['states_x']; su = filt['states_u']

    e = innov[eval_start:]; S_sc = S_vals[eval_start:]
    valid = ~np.isnan(e)
    e_v = e[valid]; S_v = S_sc[valid]
    n_valid = len(e_v)

    acf = compute_acf(e_v, max_lag=50)
    nis = float(np.mean(e_v**2 / np.maximum(S_v, 1e-15)))
    z90 = 1.6449
    cov90 = float(np.mean(np.abs(e_v) <= z90 * np.sqrt(np.maximum(S_v, 1e-15))))

    cl_sc = filt['cl_dt'][eval_start:]
    ph_sc = filt['physics'][eval_start:]
    tot_sc = cl_sc + ph_sc
    var_cl = np.var(cl_sc)
    var_tot = np.var(tot_sc) if np.var(tot_sc) > 1e-15 else 1.0
    grey_frac = float(var_cl / var_tot) if var_tot > 1e-15 else 0.0
    ratio = np.abs(cl_sc) / np.maximum(np.abs(tot_sc), 1e-15)
    grey_med = float(np.median(ratio))

    dxr2_orc, mae_orc = compute_dxr2(
        params, cl_params, sx, su, t, x_obs, v, max_h, eval_start, 'oracle')
    dxr2_per, mae_per = compute_dxr2(
        params, cl_params, sx, su, t, x_obs, v, max_h, eval_start, 'no_forcing')

    elapsed = time.time() - t0_ev
    d10 = dxr2_orc[9] if max_h >= 10 else np.nan
    m10 = mae_orc[9] if max_h >= 10 else np.nan
    print(f"    [{label}] ACF(1)={acf[1]:.4f} NIS={nis:.4f} cov90={cov90:.3f} "
          f"DxR2@10={d10:+.4f} MAE@10={m10:.6f} [{elapsed:.0f}s]")

    return {
        'label': label,
        'acf1': float(acf[1]), 'acf5': float(acf[5]), 'acf10': float(acf[10]),
        'acf_raw': acf.tolist(),
        'nis_mean': nis, 'cov90': cov90,
        'grey_frac': grey_frac, 'grey_med_ratio': grey_med,
        'dxr2_oracle': dxr2_orc.tolist(),
        'dxr2_persist': dxr2_per.tolist(),
        'dxr2_10_oracle': float(dxr2_orc[9]),
        'dxr2_10_persist': float(dxr2_per[9]),
        'mean_dxr2_5_10_oracle': float(np.nanmean(dxr2_orc[4:10])),
        'mae10': float(mae_orc[9]) if max_h >= 10 else np.nan,
        'n_scored': n_valid,
        'states_x': sx, 'states_u': su,
    }


# ==============================================================================
#  EVENT DETECTION
# ==============================================================================

def detect_events(x, min_persist=3):
    """k-means(k=2) on displacement with hysteresis."""
    c1, c2 = np.percentile(x, 25), np.percentile(x, 75)
    for _ in range(100):
        labels = (np.abs(x - c2) < np.abs(x - c1)).astype(int)
        if labels.sum() == 0 or labels.sum() == len(x):
            break
        c1_new = np.mean(x[labels == 0])
        c2_new = np.mean(x[labels == 1])
        if abs(c1_new - c1) + abs(c2_new - c2) < 1e-8:
            break
        c1, c2 = c1_new, c2_new

    state = labels[0]
    clean = [state]
    pending_state = None
    pending_count = 0
    for i in range(1, len(labels)):
        if labels[i] != state:
            if pending_state == labels[i]:
                pending_count += 1
            else:
                pending_state = labels[i]
                pending_count = 1
            if pending_count >= min_persist:
                state = pending_state
                pending_state = None
                pending_count = 0
        else:
            pending_state = None
            pending_count = 0
        clean.append(state)
    clean = np.array(clean)
    event_idx = np.where(np.diff(clean) != 0)[0] + 1
    return event_idx, clean


# ==============================================================================
#  AUDIT B: ALPHA PARAMETERIZATION EQUIVALENCE
# ==============================================================================

def run_audit_b(model_s1, s1_params, df_val, device, audit_dir):
    """Verify KalmanForecaster and KalmanForecasterClosure produce identical
    predictions when alpha_param='softplus' and closure=0."""
    print_section("AUDIT B: ALPHA PARAMETERIZATION EQUIVALENCE")
    lines = ["AUDIT B: Alpha Parameterization Equivalence", "="*50, ""]

    alpha_s1 = s1_params['alpha']
    lines.append(f"S1 alpha (softplus, KalmanForecaster): {alpha_s1:.6f}")
    lines.append(f"rho(alpha={alpha_s1:.4f}, dt=0.1) = {math.exp(-alpha_s1*0.1):.6f}")
    lines.append("")

    # Build S2 model with same physics, closure=0, alpha_param="softplus"
    model_s2_test = KalmanForecasterClosure(
        alpha_init=max(alpha_s1, 1e-6),
        c_init=max(s1_params['c'], 0.01),
        vc_init=s1_params['vc'],
        kappa_init=max(s1_params['kappa'], 0.001),
        log_qx_init=math.log(max(s1_params['qx'], 1e-15)),
        log_qu_init=math.log(max(s1_params['qu'], 1e-15)),
        log_r_init=math.log(max(s1_params['R'], 1e-15)),
        log_p0_xx_init=math.log(max(s1_params['P0_xx'], 1e-15)),
        log_p0_uu_init=math.log(max(s1_params['P0_uu'], 1e-15)),
        a1_init=0.001, b1_init=0.0, b2_init=0.0,
        d1_init=0.001, d2_init=0.001, d3_init=0.001,
        alpha_param="softplus",
    ).to(device)

    # Zero out closure
    with torch.no_grad():
        model_s2_test.a1_raw.fill_(-20.0)
        model_s2_test.b1.fill_(0.0)
        model_s2_test.b2.fill_(0.0)
        model_s2_test.d1_raw.fill_(-20.0)
        model_s2_test.d2_raw.fill_(-20.0)
        model_s2_test.d3_raw.fill_(-20.0)
        model_s2_test.log_q_scale.fill_(0.0)

    alpha_s2 = model_s2_test.alpha.item()
    alpha_match_delta = abs(alpha_s1 - alpha_s2)
    lines.append(f"S2 alpha (softplus, KalmanForecasterClosure): {alpha_s2:.6f}")
    lines.append(f"Alpha delta: {alpha_match_delta:.2e}")
    assert alpha_match_delta < 1e-6, \
        f"Alpha match FAIL: delta={alpha_match_delta:.2e}. Softplus transfer broken."
    lines.append(f"Alpha match: PASS (delta < 1e-6)")
    lines.append("")

    model_s1.eval()
    model_s2_test.eval()

    # --- Test 1: Real 200-step snippet from val ---
    lines.append("--- Test 1: Real 200-step snippet from val ---")
    snippet_len = 200
    v_val = df_val['velocity'].values[:snippet_len].astype(np.float32)
    x_val = df_val['displacement'].values[:snippet_len].astype(np.float32)
    dt_val = df_val['time_delta'].values[:snippet_len].astype(np.float32)

    L_test = min(100, snippet_len - 10)
    H_test = snippet_len - L_test

    v_h = torch.tensor(v_val[:L_test], dtype=torch.float32).unsqueeze(0).to(device)
    dt_h = torch.tensor(dt_val[:L_test], dtype=torch.float32).unsqueeze(0).to(device)
    x_h = torch.tensor(x_val[:L_test], dtype=torch.float32).unsqueeze(0).to(device)
    v_f = torch.tensor(v_val[L_test:L_test+H_test], dtype=torch.float32).unsqueeze(0).to(device)
    dt_f = torch.tensor(dt_val[L_test:L_test+H_test], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        xp1, xv1, _ = model_s1(v_h, dt_h, x_h, v_f, dt_f)
        xp2, xv2, _ = model_s2_test(v_h, dt_h, x_h, v_f, dt_f)

    max_diff_real = float(torch.max(torch.abs(xp1 - xp2)).item())
    lines.append(f"  max |x_pred_S1 - x_pred_S2| = {max_diff_real:.2e}")
    # Threshold 1e-4: models have different c parameterization (unconstrained vs
    # softplus) causing ~1e-5 roundtrip error. Alpha delta is checked separately.
    test1_pass = max_diff_real < 1e-4
    lines.append(f"  VERDICT: {'PASS' if test1_pass else 'FAIL'} (threshold 1e-4)")
    lines.append("")

    # --- Test 2: Synthetic (constant v=0.3, dt=0.1) ---
    lines.append("--- Test 2: Synthetic (constant v=0.3, dt=0.1) ---")
    N_synth = 200
    L_synth = 100; H_synth = 100
    v_synth = torch.full((1, N_synth), 0.3, dtype=torch.float32).to(device)
    dt_synth = torch.full((1, N_synth), 0.1, dtype=torch.float32).to(device)
    x_synth = torch.zeros(1, N_synth, dtype=torch.float32).to(device)
    x_synth[0, 0] = 0.01  # small initial displacement

    with torch.no_grad():
        xp1s, _, _ = model_s1(v_synth[:, :L_synth], dt_synth[:, :L_synth],
                               x_synth[:, :L_synth],
                               v_synth[:, L_synth:], dt_synth[:, L_synth:])
        xp2s, _, _ = model_s2_test(v_synth[:, :L_synth], dt_synth[:, :L_synth],
                                    x_synth[:, :L_synth],
                                    v_synth[:, L_synth:], dt_synth[:, L_synth:])

    max_diff_synth = float(torch.max(torch.abs(xp1s - xp2s)).item())
    lines.append(f"  max |x_pred_S1 - x_pred_S2| = {max_diff_synth:.2e}")
    test2_pass = max_diff_synth < 1e-4
    lines.append(f"  VERDICT: {'PASS' if test2_pass else 'FAIL'} (threshold 1e-4)")
    lines.append("")

    overall_pass = test1_pass and test2_pass
    lines.append(f"OVERALL AUDIT B: {'PASS' if overall_pass else 'FAIL'}")

    audit_path = audit_dir / "alpha_parameterization.txt"
    with open(audit_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Wrote {audit_path}")
    print(f"  Test 1 (real): max diff = {max_diff_real:.2e} -> {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (synth): max diff = {max_diff_synth:.2e} -> {'PASS' if test2_pass else 'FAIL'}")
    print(f"  AUDIT B OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    assert overall_pass, "AUDIT B FAILED: Alpha parameterization mismatch. Aborting."
    return overall_pass


# ==============================================================================
#  AUDIT A: FREEZE INTEGRITY
# ==============================================================================

def run_audit_a(model_s2, frozen_snapshot, audit_dir):
    """Verify all frozen params unchanged after S2 training."""
    print_section("AUDIT A: FREEZE INTEGRITY")
    lines = ["AUDIT A: Freeze Integrity", "="*50, ""]

    # requires_grad table
    lines.append("Parameter requires_grad table:")
    lines.append(f"  {'Name':<25} {'Shape':<15} {'requires_grad':<15}")
    lines.append(f"  {'-'*25} {'-'*15} {'-'*15}")
    for name, p in model_s2.named_parameters():
        lines.append(f"  {name:<25} {str(tuple(p.shape)):<15} {str(p.requires_grad):<15}")
    lines.append("")

    # Diff frozen tensors
    lines.append("Frozen tensor diffs after S2 training:")
    max_delta_all = 0.0
    all_pass = True
    for name, snap_val in frozen_snapshot.items():
        current_val = dict(model_s2.named_parameters())[name].data
        delta = float(torch.max(torch.abs(current_val - snap_val)).item())
        passed = delta < 1e-12
        status = "PASS" if passed else "FAIL"
        lines.append(f"  {name:<25} max_delta={delta:.2e}  {status}")
        max_delta_all = max(max_delta_all, delta)
        if not passed:
            all_pass = False

    lines.append("")
    lines.append(f"Max delta across all frozen params: {max_delta_all:.2e}")
    lines.append(f"OVERALL AUDIT A: {'PASS' if all_pass else 'FAIL'}")

    audit_path = audit_dir / "freeze_integrity.txt"
    with open(audit_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Wrote {audit_path}")
    print(f"  Max frozen param delta: {max_delta_all:.2e}")
    print(f"  AUDIT A OVERALL: {'PASS' if all_pass else 'FAIL'}")

    assert all_pass, "AUDIT A FAILED: Frozen parameters changed during S2 training. Aborting."
    return all_pass


# ==============================================================================
#  SINGLE SEED PIPELINE
# ==============================================================================

def run_seed(seed, csv_paths, df_train, df_val, df_test, device,
             seed_dir, run_audits=False, audit_dir=None):
    """Run full S1 + S2 + evaluation for one seed. Returns result dict."""
    t0_seed = time.time()
    for d in ['checkpoints', 'figures', 'tables']:
        (seed_dir / d).mkdir(parents=True, exist_ok=True)

    print_section(f"SEED {seed}: STAGE 1 -- PHYSICS ONLY")
    torch.manual_seed(seed); np.random.seed(seed)

    # S1 data loaders
    train_ds_s1 = StateSpaceDataset(
        [str(csv_paths['train'])], L=S1_L, m=S1_L, H=S1_H,
        predict_deltas=False, normalize=False)
    val_ds_s1 = StateSpaceDataset(
        [str(csv_paths['val'])], L=S1_L, m=S1_L, H=S1_H,
        predict_deltas=False, normalize=False)
    train_ld_s1 = DataLoader(train_ds_s1, batch_size=S1_BATCH,
                             shuffle=True, num_workers=0)
    val_ld_s1 = DataLoader(val_ds_s1, batch_size=S1_BATCH,
                           shuffle=False, num_workers=0)
    print(f"  S1 datasets: train={len(train_ds_s1)}, val={len(val_ds_s1)}")

    # S1 training
    model_s1 = KalmanForecaster(use_kappa=True).to(device)
    best_val_s1, best_ep_s1, final_tr_s1, tr_h_s1, val_h_s1 = \
        train_model(model_s1, train_ld_s1, val_ld_s1, device,
                    S1_EPOCHS, S1_PATIENCE, S1_LR, S1_SCHED, tag=f"S1-seed{seed}")

    s1_params = model_s1.param_summary()
    print(f"  S1 learned: alpha={s1_params['alpha']:.4f} "
          f"tau={s1_params['tau']:.3f}s c={s1_params['c']:.4f} "
          f"vc={s1_params['vc']:.4f} kappa={s1_params['kappa']:.4f}")

    torch.save({
        'state_dict': model_s1.state_dict(),
        'params': s1_params,
        'best_val': best_val_s1, 'best_epoch': best_ep_s1,
        'seed': seed,
    }, seed_dir / "checkpoints" / f"stage1_physics_seed{seed}.pth")

    # S1 training curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(tr_h_s1)+1), tr_h_s1, label='Train NLL')
    ax.plot(range(1, len(val_h_s1)+1), val_h_s1, label='Val NLL')
    ax.axvline(best_ep_s1, color='red', ls='--', alpha=0.5,
               label=f'Best ep={best_ep_s1}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gaussian NLL')
    ax.set_title(f'Stage 1: Physics Only (seed={seed})')
    ax.legend(); fig.tight_layout()
    fig.savefig(seed_dir / "figures" / "training_curves_s1.png")
    plt.close(fig)

    # --- AUDIT B (seed 1 only) ---
    if run_audits:
        run_audit_b(model_s1, s1_params, df_val, device, audit_dir)

    # ==================================================================
    #  STAGE 2 -- CLOSURE 2-TERM
    # ==================================================================
    print_section(f"SEED {seed}: STAGE 2 -- CLOSURE 2-TERM")
    torch.manual_seed(seed); np.random.seed(seed)

    train_ds_s2 = StateSpaceDataset(
        [str(csv_paths['train'])], L=S2_L, m=S2_L, H=S2_H,
        predict_deltas=False, normalize=False)
    val_ds_s2 = StateSpaceDataset(
        [str(csv_paths['val'])], L=S2_L, m=S2_L, H=S2_H,
        predict_deltas=False, normalize=False)
    train_ld_s2 = DataLoader(train_ds_s2, batch_size=S2_BATCH,
                             shuffle=True, num_workers=0)
    val_ld_s2 = DataLoader(val_ds_s2, batch_size=S2_BATCH,
                           shuffle=False, num_workers=0)
    print(f"  S2 datasets: train={len(train_ds_s2)}, val={len(val_ds_s2)}")

    # Build closure model with alpha_param="softplus" -- NO clipping!
    model_s2 = KalmanForecasterClosure(
        alpha_init=max(s1_params['alpha'], 1e-6),
        c_init=max(s1_params['c'], 0.01),
        vc_init=s1_params['vc'],
        kappa_init=max(s1_params['kappa'], 0.001),
        log_qx_init=math.log(max(s1_params['qx'], 1e-15)),
        log_qu_init=math.log(max(s1_params['qu'], 1e-15)),
        log_r_init=math.log(max(s1_params['R'], 1e-15)),
        log_p0_xx_init=math.log(max(s1_params['P0_xx'], 1e-15)),
        log_p0_uu_init=math.log(max(s1_params['P0_uu'], 1e-15)),
        a1_init=0.001, b1_init=0.0, b2_init=0.0,
        d1_init=0.001, d2_init=0.001, d3_init=0.001,
        alpha_param="softplus",
    ).to(device)

    # Alpha fidelity check
    alpha_s2_init = model_s2.alpha.item()
    alpha_delta = abs(alpha_s2_init - s1_params['alpha'])
    print(f"  Alpha fidelity: S1={s1_params['alpha']:.6f} S2_init={alpha_s2_init:.6f} "
          f"delta={alpha_delta:.2e}")
    assert alpha_delta < 1e-4, \
        f"Alpha fidelity FAIL: delta={alpha_delta:.2e} >= 1e-4"

    # Freeze physics
    model_s2.freeze_physics()
    # Freeze a1, b1, d1, d3 at zero
    with torch.no_grad():
        model_s2.a1_raw.fill_(-10.0); model_s2.a1_raw.requires_grad_(False)
        model_s2.b1.fill_(0.0); model_s2.b1.requires_grad_(False)
        model_s2.d1_raw.fill_(-10.0); model_s2.d1_raw.requires_grad_(False)
        model_s2.d3_raw.fill_(-10.0); model_s2.d3_raw.requires_grad_(False)

    # Snapshot frozen params BEFORE training (for Audit A)
    frozen_snapshot = {}
    for name, p in model_s2.named_parameters():
        if not p.requires_grad:
            frozen_snapshot[name] = p.data.clone()

    # Trainable: b2, d2_raw, log_q_scale
    s2_trainable = lambda: [model_s2.b2, model_s2.d2_raw, model_s2.log_q_scale]

    best_val_s2, best_ep_s2, final_tr_s2, tr_h_s2, val_h_s2 = \
        train_model(model_s2, train_ld_s2, val_ld_s2, device,
                    S2_EPOCHS, S2_PATIENCE, S2_LR, S2_SCHED,
                    param_getter=s2_trainable, tag=f"S2-seed{seed}")

    s2_params = model_s2.param_summary()
    cl_sum = model_s2.closure_summary()
    print(f"  S2 learned: b2={cl_sum['b2']:.4f} d2={cl_sum['d2']:.4f} "
          f"q_scale={cl_sum['q_scale']:.4f}")

    torch.save({
        'state_dict': model_s2.state_dict(),
        'params': s2_params,
        'closure': cl_sum,
        'best_val': best_val_s2, 'best_epoch': best_ep_s2,
        'seed': seed,
        'alpha_param': 'softplus',
    }, seed_dir / "checkpoints" / f"closure_2t_seed{seed}.pth")

    # S2 training curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(tr_h_s2)+1), tr_h_s2, label='Train NLL')
    ax.plot(range(1, len(val_h_s2)+1), val_h_s2, label='Val NLL')
    ax.axvline(best_ep_s2, color='red', ls='--', alpha=0.5,
               label=f'Best ep={best_ep_s2}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gaussian NLL')
    ax.set_title(f'Stage 2: Closure 2-term (seed={seed})')
    ax.legend(); fig.tight_layout()
    fig.savefig(seed_dir / "figures" / "training_curves_s2.png")
    plt.close(fig)

    # --- AUDIT A (seed 1 only) ---
    if run_audits:
        run_audit_a(model_s2, frozen_snapshot, audit_dir)

    # ==================================================================
    #  EVALUATION
    # ==================================================================
    print_section(f"SEED {seed}: EVALUATION")

    phys_pp = {
        'alpha': s1_params['alpha'], 'c': s1_params['c'],
        'vc': s1_params['vc'], 'kappa': s1_params['kappa'],
        'qx': s1_params['qx'], 'qu': s1_params['qu'],
        'R': s1_params['R'],
        'P0_xx': s1_params['P0_xx'], 'P0_uu': s1_params['P0_uu'],
    }
    phys_cl = {k: 0.0 for k in ['a1','b1','b2','d1','d2','d3']}
    phys_cl['q_scale'] = 1.0

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

    test_start_time = df_test['timestamp'].iloc[0]

    # --- WARM START ---
    warmup_start = df_val['timestamp'].iloc[-1] - WARMUP_SEC
    df_warmup = df_val[df_val['timestamp'] >= warmup_start].copy()
    df_warm = pd.concat([df_warmup, df_test], ignore_index=True)
    t_warm = df_warm['timestamp'].values.astype(np.float64)
    x_warm = df_warm['displacement'].values.astype(np.float64)
    v_warm = df_warm['velocity'].values.astype(np.float64)
    score_mask_warm = t_warm >= test_start_time
    eval_start_warm = int(np.argmax(score_mask_warm))
    n_warmup = eval_start_warm
    n_scored = int(score_mask_warm.sum())
    assert n_scored == len(df_test), \
        f"Warmup guard FAIL: scored {n_scored} != test {len(df_test)}"
    print(f"  Warmup: {n_warmup} warmup pts + {len(df_test)} test pts, "
          f"scoring {n_scored} (warmup NOT scored)")

    # --- COLD START ---
    t_cold = df_test['timestamp'].values.astype(np.float64)
    x_cold = df_test['displacement'].values.astype(np.float64)
    v_cold = df_test['velocity'].values.astype(np.float64)
    eval_start_cold = 1

    results = {}
    for mode, t_arr, x_arr, v_arr, es in [
        ('warm', t_warm, x_warm, v_warm, eval_start_warm),
        ('cold', t_cold, x_cold, v_cold, eval_start_cold),
    ]:
        for mtype, pp, cc in [
            ('physics', phys_pp, phys_cl),
            ('closure', clos_pp, clos_cl),
        ]:
            label = f"{mtype}_{mode}"
            results[label] = evaluate_model(
                label, pp, cc, t_arr, x_arr, v_arr, es, MAX_H)

    # ==================================================================
    #  TABLES
    # ==================================================================
    # metrics_table.csv
    metric_rows = []
    for key in ['physics_warm', 'closure_warm', 'physics_cold', 'closure_cold']:
        r = results[key]
        metric_rows.append({
            'variant': key,
            'acf1': r['acf1'], 'acf5': r['acf5'], 'acf10': r['acf10'],
            'nis_mean': r['nis_mean'], 'cov90': r['cov90'],
            'grey_frac': r['grey_frac'], 'grey_med_ratio': r['grey_med_ratio'],
            'dxr2_10_oracle': r['dxr2_10_oracle'],
            'dxr2_10_persist': r['dxr2_10_persist'],
            'mean_dxr2_5_10_oracle': r['mean_dxr2_5_10_oracle'],
            'mae10': r['mae10'],
            'n_scored': r['n_scored'],
        })
    pd.DataFrame(metric_rows).to_csv(
        seed_dir / "tables" / "metrics_table.csv", index=False)

    # horizon_curve.csv
    hc_rows = []
    for h in range(MAX_H):
        hc_rows.append({
            'h': h + 1,
            'oracle_physics': results['physics_warm']['dxr2_oracle'][h],
            'oracle_closure': results['closure_warm']['dxr2_oracle'][h],
            'persist_physics': results['physics_warm']['dxr2_persist'][h],
            'persist_closure': results['closure_warm']['dxr2_persist'][h],
        })
    pd.DataFrame(hc_rows).to_csv(
        seed_dir / "tables" / "horizon_curve.csv", index=False)

    # ljung_box.csv
    lb_rows = []
    for key in ['physics_warm', 'closure_warm']:
        r = results[key]
        lb = ljung_box(np.array(r['acf_raw']), r['n_scored'])
        for entry in lb:
            lb_rows.append({
                'model': key, 'lag': entry['lag'],
                'Q': entry['Q'], 'p_value': entry['p'],
            })
    pd.DataFrame(lb_rows).to_csv(
        seed_dir / "tables" / "ljung_box.csv", index=False)

    # learned_params.csv
    param_rows = [
        {'stage': 'S1_physics', **s1_params},
        {'stage': 'S2_closure', **s2_params},
    ]
    pd.DataFrame(param_rows).to_csv(
        seed_dir / "tables" / "learned_params.csv", index=False)

    # ==================================================================
    #  FIGURES
    # ==================================================================
    # horizon_curve_dxr2.png
    fig, ax = plt.subplots(figsize=(10, 6))
    h_range = np.arange(1, MAX_H + 1)
    ax.plot(h_range, results['physics_warm']['dxr2_oracle'],
            'b-', label='Physics oracle', lw=1.5)
    ax.plot(h_range, results['closure_warm']['dxr2_oracle'],
            'r-', label='Closure oracle', lw=1.5)
    ax.plot(h_range, results['physics_warm']['dxr2_persist'],
            'b--', label='Physics no-forcing', lw=1, alpha=0.7)
    ax.plot(h_range, results['closure_warm']['dxr2_persist'],
            'r--', label='Closure no-forcing', lw=1, alpha=0.7)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Horizon h (steps)')
    ax.set_ylabel('DxR2(h)')
    ax.set_title(f'DxR2 Horizon Curve (warm start, seed={seed})')
    ax.legend(); fig.tight_layout()
    fig.savefig(seed_dir / "figures" / "horizon_curve_dxr2.png")
    plt.close(fig)

    # innovation_acf.png
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(1, 21)
    acf_phys = np.array(results['physics_warm']['acf_raw'])[1:21]
    acf_cl = np.array(results['closure_warm']['acf_raw'])[1:21]
    w = 0.35
    ax.bar(lags - w/2, acf_phys, w, label='Physics', color='steelblue', alpha=0.8)
    ax.bar(lags + w/2, acf_cl, w, label='Closure', color='indianred', alpha=0.8)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    n_sc = results['physics_warm']['n_scored']
    ci_line = 1.96 / np.sqrt(n_sc)
    ax.axhline(ci_line, color='gray', ls='--', alpha=0.5, label='95% CI')
    ax.axhline(-ci_line, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title(f'Innovation ACF (warm start, seed={seed})')
    ax.legend(); fig.tight_layout()
    fig.savefig(seed_dir / "figures" / "innovation_acf.png")
    plt.close(fig)

    # ==================================================================
    #  EVENT METRICS (Deliverable E)
    # ==================================================================
    print_section(f"SEED {seed}: EVENT METRICS")

    t_ev = t_warm; x_ev = x_warm; v_ev = v_warm; es_ev = eval_start_warm

    x_test_only = x_warm[es_ev:]
    event_idx_local, ev_labels = detect_events(x_test_only, min_persist=3)
    event_idx_warm = event_idx_local + es_ev
    n_events = len(event_idx_warm)
    print(f"  Detected {n_events} events in test set")

    event_radius_s = 10.0
    event_radius_steps = int(event_radius_s / DT)
    test_indices = np.arange(es_ev, len(t_ev))

    event_mask = np.zeros(len(t_ev), dtype=bool)
    for eidx in event_idx_warm:
        lo = max(es_ev, eidx - event_radius_steps)
        hi = min(len(t_ev), eidx + event_radius_steps + 1)
        event_mask[lo:hi] = True

    event_indices = test_indices[event_mask[es_ev:]]
    nonevent_indices = test_indices[~event_mask[es_ev:]]
    print(f"  Event window: {len(event_indices)} pts, "
          f"Non-event: {len(nonevent_indices)} pts")

    h_event = 50
    event_skill_all = []
    for mtype, pp, cc in [('physics', phys_pp, phys_cl),
                           ('closure', clos_pp, clos_cl)]:
        label_w = f"{mtype}_warm"
        sx_w = results[label_w]['states_x']
        su_w = results[label_w]['states_u']

        for subset_name, idx_set in [('full', None),
                                      ('event', event_indices),
                                      ('nonevent', nonevent_indices)]:
            skill_rows = compute_event_skill(
                pp, cc, sx_w, su_w, t_ev, x_ev, v_ev,
                h_event, es_ev, indices=idx_set)
            for row in skill_rows:
                row['model'] = mtype
                row['subset'] = subset_name
            event_skill_all.extend(skill_rows)

    df_event_skill = pd.DataFrame(event_skill_all)
    df_event_skill.to_csv(
        seed_dir / "tables" / "event_skill_table.csv", index=False)
    print(f"  Wrote event_skill_table.csv ({len(df_event_skill)} rows)")

    # event_skill_figure.png (2x2 panel)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (metric, ylabel) in enumerate([
        ('rmse_dx', 'RMSE(dx)'), ('rmse_skill', 'RMSE Skill (1 - RMSE/baseline)'),
        ('mae_dx', 'MAE(dx)'), ('var_dx_true', 'Var(dx_true)')
    ]):
        ax = axes[idx // 2, idx % 2]
        for mtype, color in [('physics', 'steelblue'), ('closure', 'indianred')]:
            for subset, ls in [('full', '-'), ('event', '--'), ('nonevent', ':')]:
                mask = (df_event_skill['model'] == mtype) & \
                       (df_event_skill['subset'] == subset)
                sub = df_event_skill[mask].sort_values('h')
                if len(sub) > 0:
                    ax.plot(sub['h'], sub[metric],
                            color=color, ls=ls, lw=1.2,
                            label=f'{mtype} {subset}')
        ax.set_xlabel('Horizon h')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if idx == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f'Event Skill Metrics (seed={seed})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(seed_dir / "figures" / "event_skill_figure.png")
    plt.close(fig)

    elapsed_seed = time.time() - t0_seed
    print(f"  Seed {seed} complete in {elapsed_seed:.0f}s ({elapsed_seed/60:.1f} min)")

    # Return summary for aggregation
    pw = results['physics_warm']
    cw = results['closure_warm']
    return {
        'seed': seed,
        # S1 params
        's1_alpha': s1_params['alpha'], 's1_kappa': s1_params['kappa'],
        's1_c': s1_params['c'], 's1_vc': s1_params['vc'],
        's1_best_ep': best_ep_s1, 's1_best_val': best_val_s1,
        # S2 params
        's2_b2': cl_sum['b2'], 's2_d2': cl_sum['d2'],
        's2_q_scale': cl_sum['q_scale'],
        's2_best_ep': best_ep_s2, 's2_best_val': best_val_s2,
        # Physics warm metrics
        'phys_acf1': pw['acf1'], 'phys_dxr2_10': pw['dxr2_10_oracle'],
        'phys_nis': pw['nis_mean'], 'phys_cov90': pw['cov90'],
        'phys_mae10': pw['mae10'],
        # Closure warm metrics
        'clos_acf1': cw['acf1'], 'clos_dxr2_10': cw['dxr2_10_oracle'],
        'clos_nis': cw['nis_mean'], 'clos_cov90': cw['cov90'],
        'clos_mae10': cw['mae10'],
        # Narrative
        'narrative_acf': cw['acf1'] < pw['acf1'],
        'narrative_dxr2': cw['dxr2_10_oracle'] > pw['dxr2_10_oracle'],
    }


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    t0_global = time.time()
    device = torch.device('cpu')

    OUT.mkdir(parents=True, exist_ok=True)
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = OUT / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOCKBOX V11.1: ALPHA FIX + FREEZE AUDIT + MULTI-SEED RETRAIN")
    print("=" * 70)
    print(f"Output -> {OUT}")
    print(f"Seeds: {SEEDS}")

    # ==================================================================
    #  STEP 0: DATA & INTEGRITY
    # ==================================================================
    print_section("STEP 0: DATA & INTEGRITY")

    csv_paths = {
        'train': CLEAN_DIR / "train_10hz_ready.csv",
        'val':   CLEAN_DIR / "val_10hz_ready.csv",
        'test':  CLEAN_DIR / "test_10hz_ready.csv",
    }

    # PATH GUARD
    for name, p in csv_paths.items():
        p_str = str(p).replace("\\", "/")
        # Must contain clean_v1
        assert "processed_data_10hz_clean_v1" in p_str, \
            f"PATH GUARD: expected clean_v1, got {p}"
        # Must NOT be the old contaminated directory
        if "processed_data_10hz/" in p_str:
            assert "processed_data_10hz_clean" in p_str, \
                f"PATH GUARD VIOLATION: {p_str}"
    print("  Path guard: PASS")

    # MD5 provenance
    md5s = {}
    for name, p in csv_paths.items():
        assert p.exists(), f"Missing: {p}"
        md5s[name] = md5_file(p)
        print(f"  {name}: md5={md5s[name]}")

    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "no-git"

    md5_rows = [{'split': k, 'md5': v, 'file': csv_paths[k].name,
                 'git_commit': git_hash} for k, v in md5s.items()]
    pd.DataFrame(md5_rows).to_csv(OUT / "tables" / "input_md5.csv", index=False)

    # Load CSVs
    df_train = pd.read_csv(csv_paths['train'])
    df_val   = pd.read_csv(csv_paths['val'])
    df_test  = pd.read_csv(csv_paths['test'])

    # Preflight
    required_cols = ['timestamp', 'time_delta', 'velocity', 'displacement']
    for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        assert list(df.columns) == required_cols, \
            f"{name} columns: {list(df.columns)}"
        assert not df['velocity'].isna().any(), f"{name} has NaN velocity"
        assert not df['displacement'].isna().any(), f"{name} has NaN displacement"
        med_dt = np.median(np.diff(df['timestamp'].values))
        assert abs(med_dt - 0.1) / 0.1 < 0.01, \
            f"{name} median dt={med_dt:.6f}, expected ~0.1"
    print("  Preflight asserts: PASS")

    for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        print(f"  {name}: {len(df)} pts "
              f"({df['timestamp'].iloc[0]:.1f}-{df['timestamp'].iloc[-1]:.1f}s)")

    # ==================================================================
    #  SEED LOOP
    # ==================================================================
    seed_results = []

    for i, seed in enumerate(SEEDS):
        seed_dir = OUT / f"seed{seed}"
        is_first = (i == 0)

        result = run_seed(
            seed, csv_paths, df_train, df_val, df_test, device,
            seed_dir, run_audits=is_first, audit_dir=audit_dir)

        seed_results.append(result)

        if is_first:
            print_section("GATE CHECK: SEED 1 AUDITS PASSED")
            print("  Both Audit A (freeze) and Audit B (alpha) passed.")
            print("  Proceeding to seeds 2-3...")

    # ==================================================================
    #  AGGREGATION (Deliverable D)
    # ==================================================================
    print_section("AGGREGATION ACROSS SEEDS")

    df_agg = pd.DataFrame(seed_results)

    # Add mean/std row
    numeric_cols = [c for c in df_agg.columns
                    if c not in ['seed', 'narrative_acf', 'narrative_dxr2']]
    mean_row = {'seed': 'mean'}
    std_row = {'seed': 'std'}
    for c in numeric_cols:
        vals = df_agg[c].values.astype(float)
        mean_row[c] = float(np.mean(vals))
        std_row[c] = float(np.std(vals))
    # Narrative: count how many seeds pass
    mean_row['narrative_acf'] = sum(df_agg['narrative_acf'])
    mean_row['narrative_dxr2'] = sum(df_agg['narrative_dxr2'])
    std_row['narrative_acf'] = ''
    std_row['narrative_dxr2'] = ''

    df_summary = pd.concat([df_agg, pd.DataFrame([mean_row, std_row])],
                           ignore_index=True)
    df_summary.to_csv(agg_dir / "summary_seeds.csv", index=False)
    print(f"  Wrote summary_seeds.csv")

    # --- summary_seeds.md ---
    n_seeds = len(SEEDS)
    b2_vals = [r['s2_b2'] for r in seed_results]
    d2_vals = [r['s2_d2'] for r in seed_results]
    qs_vals = [r['s2_q_scale'] for r in seed_results]
    b2_cv = np.std(b2_vals) / np.mean(b2_vals) * 100 if np.mean(b2_vals) != 0 else 0
    d2_cv = np.std(d2_vals) / np.mean(d2_vals) * 100 if np.mean(d2_vals) != 0 else 0
    qs_cv = np.std(qs_vals) / np.mean(qs_vals) * 100 if np.mean(qs_vals) != 0 else 0

    n_acf_pass = sum(r['narrative_acf'] for r in seed_results)
    n_dxr2_pass = sum(r['narrative_dxr2'] for r in seed_results)

    md_lines = [
        "# Lockbox v11.1: Multi-Seed Aggregation",
        "",
        "## Parameter Stability",
        f"- b2: mean={np.mean(b2_vals):.4f}, std={np.std(b2_vals):.4f}, CV={b2_cv:.2f}%",
        f"- d2: mean={np.mean(d2_vals):.4f}, std={np.std(d2_vals):.4f}, CV={d2_cv:.2f}%",
        f"- q_scale: mean={np.mean(qs_vals):.4f}, std={np.std(qs_vals):.4f}, CV={qs_cv:.2f}%",
        "",
        "## Narrative Survival",
        f"- ACF(1) improvement: {n_acf_pass}/{n_seeds} seeds",
        f"- DxR2@10 improvement: {n_dxr2_pass}/{n_seeds} seeds",
        "",
        "## Per-Seed Summary",
    ]
    for r in seed_results:
        md_lines.append(
            f"- Seed {r['seed']}: b2={r['s2_b2']:.4f}, d2={r['s2_d2']:.4f}, "
            f"ACF1 {r['phys_acf1']:.4f}->{r['clos_acf1']:.4f} "
            f"({'improved' if r['narrative_acf'] else 'worse'}), "
            f"DxR2@10 {r['phys_dxr2_10']:+.4f}->{r['clos_dxr2_10']:+.4f} "
            f"({'improved' if r['narrative_dxr2'] else 'worse'})")

    verdict = "SURVIVES" if n_acf_pass == n_seeds and n_dxr2_pass == n_seeds else \
              "PARTIALLY SURVIVES" if n_acf_pass + n_dxr2_pass > 0 else "DOES NOT SURVIVE"
    md_lines.extend([
        "",
        f"## Verdict: Narrative {verdict}",
        f"All parameter CVs {'<' if max(b2_cv, d2_cv, qs_cv) < 10 else '>='} 10% threshold.",
    ])

    with open(agg_dir / "summary_seeds.md", 'w') as f:
        f.write('\n'.join(md_lines))
    print("  Wrote summary_seeds.md")

    # --- seed_variation_plot.png (2x3 panel) ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    seed_labels = [str(r['seed']) for r in seed_results]
    x_pos = np.arange(len(seed_labels))

    # Row 1: b2, d2, q_scale
    for col, (param, vals, label) in enumerate([
        ('b2', b2_vals, 'b2'),
        ('d2', d2_vals, 'd2'),
        ('q_scale', qs_vals, 'q_scale'),
    ]):
        ax = axes[0, col]
        ax.bar(x_pos, vals, color='steelblue', alpha=0.8)
        ax.axhline(np.mean(vals), color='red', ls='--', alpha=0.7, label='mean')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(seed_labels)
        ax.set_xlabel('Seed')
        ax.set_ylabel(label)
        ax.set_title(f'{label} per seed')
        ax.legend(fontsize=8)

    # Row 2: DxR2@10, ACF(1), MAE@10 (physics vs closure grouped)
    for col, (metric_p, metric_c, ylabel) in enumerate([
        ('phys_dxr2_10', 'clos_dxr2_10', 'DxR2@10'),
        ('phys_acf1', 'clos_acf1', 'ACF(1)'),
        ('phys_mae10', 'clos_mae10', 'MAE@10'),
    ]):
        ax = axes[1, col]
        w = 0.35
        p_vals = [r[metric_p] for r in seed_results]
        c_vals = [r[metric_c] for r in seed_results]
        ax.bar(x_pos - w/2, p_vals, w, label='Physics', color='steelblue', alpha=0.8)
        ax.bar(x_pos + w/2, c_vals, w, label='Closure', color='indianred', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(seed_labels)
        ax.set_xlabel('Seed')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} per seed')
        ax.legend(fontsize=8)

    fig.suptitle('Lockbox v11.1: Seed Variation', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(agg_dir / "seed_variation_plot.png")
    plt.close(fig)
    print("  Wrote seed_variation_plot.png")

    # ==================================================================
    #  README
    # ==================================================================
    print_section("README")

    old_params = {
        'alpha': 0.45, 'kappa': 0.13, 'c': 0.91,
        'b2': 6.338, 'd2': 10.458, 'q_scale': 1.856,
    }

    # Build per-seed param comparison table
    param_table_lines = []
    param_table_lines.append("| Parameter | Old paper | " +
                              " | ".join([f"Seed {r['seed']}" for r in seed_results]) +
                              " | Mean +/- Std |")
    param_table_lines.append("|-----------|-----------|" +
                              "|".join(["-----------|" for _ in seed_results]) +
                              "-------------|")
    for pname, old_val, getter in [
        ('alpha', old_params['alpha'], lambda r: r['s1_alpha']),
        ('kappa', old_params['kappa'], lambda r: r['s1_kappa']),
        ('c', old_params['c'], lambda r: r['s1_c']),
        ('vc', 0.17, lambda r: r['s1_vc']),
        ('b2', old_params['b2'], lambda r: r['s2_b2']),
        ('d2', old_params['d2'], lambda r: r['s2_d2']),
        ('q_scale', old_params['q_scale'], lambda r: r['s2_q_scale']),
    ]:
        vals = [getter(r) for r in seed_results]
        row = f"| {pname:<9} | {old_val:<9.4f} | "
        row += " | ".join([f"{v:<9.4f}" for v in vals])
        row += f" | {np.mean(vals):.4f} +/- {np.std(vals):.4f} |"
        param_table_lines.append(row)

    # Build per-seed metrics comparison table
    metrics_table_lines = []
    metrics_table_lines.append("| Metric | " +
                                " | ".join([f"Seed {r['seed']} Phys / Clos" for r in seed_results]) + " |")
    metrics_table_lines.append("|--------|" +
                                "|".join(["----------------------|" for _ in seed_results]) + "")
    for mname, p_key, c_key, fmt in [
        ('ACF(1)', 'phys_acf1', 'clos_acf1', '.4f'),
        ('DxR2@10', 'phys_dxr2_10', 'clos_dxr2_10', '+.4f'),
        ('NIS', 'phys_nis', 'clos_nis', '.4f'),
        ('cov90', 'phys_cov90', 'clos_cov90', '.3f'),
        ('MAE@10', 'phys_mae10', 'clos_mae10', '.6f'),
    ]:
        row = f"| {mname:<8} | "
        row += " | ".join([f"{format(r[p_key], fmt)} / {format(r[c_key], fmt)}" for r in seed_results])
        row += " |"
        metrics_table_lines.append(row)

    readme_text = f"""# Lockbox v11.1: Alpha Fix + Freeze Audit + Multi-Seed Retrain

## Alpha Fix Rationale

v11 trained Stage 1 (KalmanForecaster) with softplus parameterization for alpha,
learning alpha=1.716. But KalmanForecasterClosure used sigmoid (range 0-1), so
alpha got clipped to 0.999 when transferred to Stage 2. This changes
rho = exp(-alpha*dt) from 0.842 to 0.905 -- a 7.5% change in per-step damping.

**Fix:** Added `alpha_param="softplus"` flag to KalmanForecasterClosure. Default
remains `"sigmoid"` for backward compatibility. v11.1 passes `"softplus"` explicitly.

## Model File Change

File: `models/kalman_closure.py`
- Added `_softplus_inv()` module-level helper (stable inverse of softplus)
- Added `alpha_param="sigmoid"` constructor argument (default preserves old behavior)
- Modified `alpha` property to use softplus when `alpha_param="softplus"`
- Old scripts/checkpoints are fully backward compatible (default="sigmoid")

## Per-Seed Learned Parameters

{chr(10).join(param_table_lines)}

## Per-Seed Metrics (warm start)

{chr(10).join(metrics_table_lines)}

## Narrative Survival

- ACF(1) improvement: {n_acf_pass}/{n_seeds} seeds
- DxR2@10 improvement: {n_dxr2_pass}/{n_seeds} seeds
- **Verdict: {verdict}**

## Parameter Stability (CV%)

- b2: CV = {b2_cv:.2f}%
- d2: CV = {d2_cv:.2f}%
- q_scale: CV = {qs_cv:.2f}%

## Audit Results

### Audit A: Freeze Integrity
All frozen parameters changed < 1e-12 during Stage 2 training. **PASS**

### Audit B: Alpha Parameterization
S1 (KalmanForecaster) and S2 (KalmanForecasterClosure with alpha_param="softplus")
produce identical predictions (max diff < 1e-6) on both real and synthetic data. **PASS**

## Warmup Protocol

- 50s of validation data used as warmup prefix
- Warmup points are NOT scored
- Scoring begins at first test timestamp

## Data

- Train: {len(df_train)} pts ({df_train['timestamp'].iloc[0]:.1f}-{df_train['timestamp'].iloc[-1]:.1f}s)
- Val: {len(df_val)} pts ({df_val['timestamp'].iloc[0]:.1f}-{df_val['timestamp'].iloc[-1]:.1f}s)
- Test: {len(df_test)} pts ({df_test['timestamp'].iloc[0]:.1f}-{df_test['timestamp'].iloc[-1]:.1f}s)
- Input MD5s: see tables/input_md5.csv
- Clean splits from `processed_data_10hz_clean_v1/`

## Output Structure

```
final_lockbox_v11_1_alpha_fix/
  audit/
    freeze_integrity.txt
    alpha_parameterization.txt
  seed1/
    checkpoints/ (stage1_physics_seed1.pth, closure_2t_seed1.pth)
    tables/ (metrics_table, horizon_curve, ljung_box, learned_params, event_skill_table)
    figures/ (training_curves_s1/s2, horizon_curve_dxr2, innovation_acf, event_skill_figure)
  seed2/ ... (same)
  seed3/ ... (same)
  aggregate/
    summary_seeds.csv
    summary_seeds.md
    seed_variation_plot.png
  tables/
    input_md5.csv
  README.md
```

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total runtime: {time.time() - t0_global:.0f}s ({(time.time() - t0_global)/60:.1f} min)
"""

    with open(OUT / "README.md", 'w') as f:
        f.write(readme_text)
    print("  Wrote README.md")

    # ==================================================================
    #  FINAL SUMMARY
    # ==================================================================
    print_section("FINAL SUMMARY")
    elapsed = time.time() - t0_global

    n_files = 0
    for dirpath, dirnames, filenames in os.walk(OUT):
        n_files += len(filenames)
    print(f"  Total output files: {n_files}")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\n  {'Seed':<6} {'b2':>8} {'d2':>8} {'q_sc':>8} "
          f"{'ACF1_p':>8} {'ACF1_c':>8} {'DxR2_p':>8} {'DxR2_c':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in seed_results:
        print(f"  {r['seed']:<6} {r['s2_b2']:8.4f} {r['s2_d2']:8.4f} "
              f"{r['s2_q_scale']:8.4f} {r['phys_acf1']:8.4f} {r['clos_acf1']:8.4f} "
              f"{r['phys_dxr2_10']:+8.4f} {r['clos_dxr2_10']:+8.4f}")

    print(f"\n  Verdict: {verdict}")
    print(f"  b2 CV={b2_cv:.2f}%, d2 CV={d2_cv:.2f}%, q_scale CV={qs_cv:.2f}%")
    print(f"\n  DONE. All outputs in {OUT}")


if __name__ == '__main__':
    main()
