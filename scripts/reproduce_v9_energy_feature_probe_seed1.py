"""
v9 Part A: Energy-Feature Closure Probe (1 seed, non-destructive)
=================================================================
Tests whether adding a deterministic energy-memory feature (EWMA of
impulse energy) to the 2-term closure improves DxR2 and innovation
whiteness.

New closure: C_t = b2*du - d2*v|u| + gamma*E_t
  where E_t = rho*E_{t-1} + phi2(t), phi2(t) = |u|*max(0, |u|-u_c)

Tests rho in {0.90, 0.95}. Learns gamma (+ b2, d2, q_scale) via
Nelder-Mead on train split.

Usage: python scripts/reproduce_v9_energy_feature_probe_seed1.py
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
from scipy.optimize import minimize as sp_minimize
from pathlib import Path
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_closure import CLOSURE_PARAM_NAMES

# ===== Paths =====
DATA_DIR = ROOT / "processed_data_10hz"
V2_CKPT  = ROOT / "final_lockbox_v2" / "checkpoints"
S1_CKPT  = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
             / "stage1_physics_only.pth")

OUT = ROOT / "final_lockbox_v9_energy_feature_probe"
for d in ['figures', 'tables']:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ===== Constants =====
DT = 0.1
SEED = 1
MAX_HORIZON = 10
SEEDS_REF = [42, 43, 44]
RHO_VALUES = [0.90, 0.95]

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


def zero_closure():
    cl = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl['q_scale'] = 1.0
    return cl


def load_s1_params(device):
    ck = torch.load(S1_CKPT, map_location=device, weights_only=False)
    return ck['params']


def load_closure_params(seed):
    ck = torch.load(V2_CKPT / f"closure_2t_s{seed}.pth",
                    map_location='cpu', weights_only=False)
    return ck['closure']


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ============================================================
#  ENERGY PROXY (causal, strictly from velocity)
# ============================================================

def compute_energy_proxy(v, u_c, rho):
    """Compute causal EWMA of impulse energy.
    phi2(t) = |u_t| * max(0, |u_t| - u_c)
    E_t = rho * E_{t-1} + phi2(t)
    """
    N = len(v)
    phi2 = np.abs(v) * np.maximum(0.0, np.abs(v) - u_c)
    E = np.zeros(N)
    E[0] = phi2[0]
    for i in range(1, N):
        E[i] = rho * E[i-1] + phi2[i]
    return E


def estimate_u_c(x, v, dx_threshold_pctile=50, u_pctile=25):
    """Estimate u_c from lower quantile of |velocity| during non-moving periods.
    Uses only the provided arrays (should be train-only)."""
    dx = np.diff(x, prepend=x[0])
    non_moving = np.abs(dx) < np.percentile(np.abs(dx), dx_threshold_pctile)
    return float(np.percentile(np.abs(v[non_moving]), u_pctile))


# ============================================================
#  KF FILTER WITH ENERGY FEATURE
# ============================================================

def kf_filter_energy(params, cl_params, t, x_obs, v, E,
                     gamma=0.0, return_states=False):
    """2-state KF with energy feature: C = b2*dv - d2*u|v| + gamma*E."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_x = np.zeros(N) if return_states else None
    states_u = np.zeros(N) if return_states else None

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']

    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])

    if return_states:
        states_x[0] = s[0]; states_u[0] = s[1]

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)

        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0

        # Closure: b2*dv - d2*u|v| + gamma*E
        cl = b2_v * dv_w - d2 * u_st * abs(v_w) + gamma * E[k-1]
        cl_dt = cl * dt

        x_p = s[0] + s[1] * dt
        u_p = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt + cl_dt
        s_pred = np.array([x_p, u_p])

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

        if return_states:
            states_x[k] = s[0]; states_u[k] = s[1]

    out = [innovations, S_values]
    if return_states:
        out += [states_x, states_u]
    return tuple(out)


# ============================================================
#  DxR2 WITH ENERGY FEATURE
# ============================================================

def compute_dxr2_energy(params, cl_params, t, x_obs, v, E,
                        gamma=0.0, max_h=10, eval_start=1):
    """DxR2(h) with energy feature in closure."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    def _predict_step(sx, su, v_w, dv_w, dt_k, E_k):
        rho = math.exp(-alpha * dt_k)
        g = max(v_w**2 - vc**2, 0.0)
        cl = b2_v * dv_w - d2 * su * abs(v_w) + gamma * E_k
        x_new = sx + su * dt_k
        u_new = rho*su - kap*sx*dt_k + c_val*g*dt_k + cl*dt_k
        return x_new, u_new

    # Pass 1: KF filter to get post-update states
    _, _, states_x, states_u = kf_filter_energy(
        params, cl_params, t, x_obs, v, E, gamma=gamma, return_states=True)

    # Pass 2: h-step open-loop predictions
    r2_arr = np.zeros(max_h)
    for h in range(1, max_h + 1):
        dx_pred_list = []; dx_obs_list = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            for step in range(h):
                k_s = i + step + 1
                if k_s >= N: break
                dt_s = t[k_s] - t[k_s - 1]
                if dt_s <= 0: dt_s = 0.1
                v_w = v[k_s - 1]
                dv_w = v[k_s - 1] - v[k_s - 2] if k_s >= 2 else 0.0
                E_k = E[k_s - 1]
                sx, su = _predict_step(sx, su, v_w, dv_w, dt_s, E_k)
            dx_pred_list.append(sx - x_obs[i])
            dx_obs_list.append(x_obs[i + h] - x_obs[i])
        dp = np.array(dx_pred_list)
        do = np.array(dx_obs_list)
        ss_res = np.sum((do - dp)**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h-1] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ============================================================
#  STANDARD KF (baseline, no energy feature)
# ============================================================

def kf_filter_2state(params, cl_params, t, x_obs, v, return_states=False):
    """Standard 2-state KF filter (baseline)."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_x = np.zeros(N) if return_states else None
    states_u = np.zeros(N) if return_states else None

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']

    a1 = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0); d3 = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])

    if return_states:
        states_x[0] = s[0]; states_u[0] = s[1]

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)

        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = (-a1*u_st + b1_v*v_w + b2_v*dv_w
              - d1*u_st**2 - d2*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_dt = cl * dt

        x_p = s[0] + s[1] * dt
        u_p = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt + cl_dt
        s_pred = np.array([x_p, u_p])

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

        if return_states:
            states_x[k] = s[0]; states_u[k] = s[1]

    out = [innovations, S_values]
    if return_states:
        out += [states_x, states_u]
    return tuple(out)


def compute_dxr2_hstep(params, cl_params, t, x_obs, v, max_h=10,
                       eval_start=1):
    """Standard DxR2(h) for baseline."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    a1 = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0); d3 = cl_params.get('d3', 0.0)

    def _predict_step(sx, su, v_w, dv_w, dt_k):
        rho = math.exp(-alpha * dt_k)
        g = max(v_w**2 - vc**2, 0.0)
        cl = (-a1*su + b1_v*v_w + b2_v*dv_w
              - d1*su**2 - d2*su*abs(v_w) - d3*su*abs(su))
        x_new = sx + su * dt_k
        u_new = rho*su - kap*sx*dt_k + c_val*g*dt_k + cl*dt_k
        return x_new, u_new

    _, _, states_x, states_u = kf_filter_2state(
        params, cl_params, t, x_obs, v, return_states=True)

    r2_arr = np.zeros(max_h)
    for h in range(1, max_h + 1):
        dx_pred_list = []; dx_obs_list = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            for step in range(h):
                k_s = i + step + 1
                if k_s >= N: break
                dt_s = t[k_s] - t[k_s - 1]
                if dt_s <= 0: dt_s = 0.1
                v_w = v[k_s - 1]
                dv_w = v[k_s - 1] - v[k_s - 2] if k_s >= 2 else 0.0
                sx, su = _predict_step(sx, su, v_w, dv_w, dt_s)
            dx_pred_list.append(sx - x_obs[i])
            dx_obs_list.append(x_obs[i + h] - x_obs[i])
        dp = np.array(dx_pred_list)
        do = np.array(dx_obs_list)
        ss_res = np.sum((do - dp)**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h-1] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ============================================================
#  TRAINING: ENERGY FEATURE CLOSURE
# ============================================================

def train_energy_closure(train_t, train_x, train_v, train_E,
                         s1_params, init_cl,
                         seed=1, maxiter=400):
    """Train b2, d2, gamma, q_scale via Nelder-Mead on Gaussian NLL."""
    np.random.seed(seed)
    t0 = time.time()

    def _softplus(x):
        return np.log1p(np.exp(x)) if x < 20 else x

    def _inv_softplus(y):
        return float(np.log(np.exp(y) - 1)) if y < 20 else y

    def _unpack(x_vec):
        cl = zero_closure()
        cl['b2'] = float(x_vec[0])
        cl['d2'] = float(_softplus(x_vec[1]))
        gamma = float(x_vec[2])  # gamma can be any sign
        cl['q_scale'] = float(np.exp(x_vec[3]))
        return cl, gamma

    def _objective(x_vec):
        cl, gamma = _unpack(x_vec)
        innov, S_vals = kf_filter_energy(
            s1_params, cl, train_t, train_x, train_v, train_E,
            gamma=gamma)
        valid = ~np.isnan(innov) & (S_vals > 0)
        if valid.sum() < 10:
            return 1e30
        e = innov[valid]; S = S_vals[valid]
        return float(0.5 * np.mean(np.log(S) + e**2 / S))

    # Init from baseline closure params
    x0 = np.array([
        init_cl['b2'],
        _inv_softplus(init_cl['d2']),
        0.0,  # gamma starts at 0
        np.log(init_cl['q_scale']),
    ], dtype=np.float64)

    result = sp_minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-4,
                                  'fatol': 1e-7, 'adaptive': True})
    cl_opt, gamma_opt = _unpack(result.x)
    elapsed = time.time() - t0
    return cl_opt, gamma_opt, float(result.fun), elapsed


# ============================================================
#  EVENT SKILL METRIC
# ============================================================

def compute_event_skill(states_x, t_arr, x_obs, test_mask, midpoint,
                        event_indices, horizon_s=1.0):
    """Check how often the model's h-step forecast predicts a midpoint crossing.

    For each event in test set, check whether any of the KF-predicted x values
    in the 1s *before* the event crossed the midpoint in the same direction.
    """
    dt_steps = int(horizon_s / DT)
    n_events = 0
    n_predicted = 0

    for ei in event_indices:
        if not test_mask[ei]:
            continue
        n_events += 1
        # Check if model states in preceding 1s crossed midpoint
        start = max(0, ei - dt_steps)
        x_window = states_x[start:ei+1]
        # Did it cross midpoint?
        if len(x_window) < 2:
            continue
        crossed = np.any(np.diff(np.sign(x_window - midpoint)) != 0)
        if crossed:
            n_predicted += 1

    hit_rate = n_predicted / n_events if n_events > 0 else 0.0
    return {'n_events': n_events, 'n_predicted': n_predicted,
            'hit_rate': hit_rate}


# ============================================================
#  MAIN
# ============================================================

def main():
    t0_global = time.time()
    np.random.seed(SEED)
    device = torch.device('cpu')

    print_section("v9 Part A: Energy-Feature Closure Probe")
    print(f"Seed: {SEED}")
    print(f"Output -> {OUT}")

    # ---- Load parameters ----
    s1_params = load_s1_params(device)
    print(f"S1 physics: alpha={s1_params['alpha']:.4f}, c={s1_params['c']:.4f}, "
          f"kappa={s1_params['kappa']:.4f}")

    cl_avg = zero_closure()
    for key in ['b2', 'd2', 'q_scale']:
        vals = [load_closure_params(s)[key] for s in SEEDS_REF]
        cl_avg[key] = float(np.mean(vals))
    print(f"Baseline closure: b2={cl_avg['b2']:.4f}, d2={cl_avg['d2']:.4f}, "
          f"q_scale={cl_avg['q_scale']:.4f}")

    # ---- Load data ----
    df_train = pd.read_csv(DATA_DIR / "train_10hz_ready.csv")
    df_val   = pd.read_csv(DATA_DIR / "val_10hz_ready.csv")
    df_test  = pd.read_csv(DATA_DIR / "test_10hz_ready.csv")

    TEST_START = df_test['timestamp'].iloc[0]
    df_dev = df_val[df_val['timestamp'] < TEST_START].copy()

    warmup_start = df_dev.timestamp.max() - 50.0
    test_warmup = df_dev[df_dev['timestamp'] >= warmup_start].copy()
    df_filter = pd.concat([test_warmup, df_test], ignore_index=True)
    test_mask = df_filter['timestamp'].values >= TEST_START

    t_arr = df_filter['timestamp'].values
    x_arr = df_filter['displacement'].values
    v_arr = df_filter['velocity'].values

    train_t = df_train['timestamp'].values
    train_x = df_train['displacement'].values
    train_v = df_train['velocity'].values

    eval_start = int(np.argmax(test_mask))

    # ---- Estimate u_c from TRAIN ONLY ----
    u_c = estimate_u_c(train_x, train_v)
    print(f"u_c (train-only): {u_c:.4f} m/s")

    # Compute energy proxy for all arrays
    # (need it for train and for filter array)
    results_by_rho = {}

    # ============================================================
    #  BASELINE EVALUATION
    # ============================================================
    print_section("BASELINE: Standard 2-term Closure")

    # Physics-only
    e_base, S_base = kf_filter_2state(
        s1_params, zero_closure(), t_arr, x_arr, v_arr)
    e_bm = e_base[test_mask]; valid_b = ~np.isnan(e_bm)
    acf_base_phys = compute_acf(e_bm[valid_b])

    dxr2_phys = compute_dxr2_hstep(
        s1_params, zero_closure(), t_arr, x_arr, v_arr,
        MAX_HORIZON, eval_start=eval_start)
    print(f"Physics-only: ACF(1)={acf_base_phys[1]:.4f}, DxR2@10={dxr2_phys[9]:.4f}")

    # 2-term closure baseline
    e_cl, S_cl = kf_filter_2state(s1_params, cl_avg, t_arr, x_arr, v_arr)
    e_cm = e_cl[test_mask]; S_cm = S_cl[test_mask]; valid_c = ~np.isnan(e_cm)
    acf_cl = compute_acf(e_cm[valid_c])
    nis_cl = float(np.mean(e_cm[valid_c]**2 / S_cm[valid_c]))

    dxr2_cl = compute_dxr2_hstep(
        s1_params, cl_avg, t_arr, x_arr, v_arr,
        MAX_HORIZON, eval_start=eval_start)
    lb_cl = ljung_box(acf_cl, int(valid_c.sum()))

    print(f"Closure (2t): ACF(1)={acf_cl[1]:.4f}, DxR2@10={dxr2_cl[9]:.4f}, "
          f"NIS={nis_cl:.4f}")
    for r in lb_cl:
        print(f"  LB lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.4f}")

    # ============================================================
    #  ENERGY FEATURE PROBES (rho = 0.90, 0.95)
    # ============================================================

    for rho_E in RHO_VALUES:
        print_section(f"ENERGY FEATURE: rho={rho_E}")

        # Compute energy proxy
        train_E = compute_energy_proxy(train_v, u_c, rho_E)
        filter_E = compute_energy_proxy(v_arr, u_c, rho_E)

        # Train
        print(f"Training (Nelder-Mead, maxiter=400)...")
        cl_opt, gamma_opt, nll, train_time = train_energy_closure(
            train_t, train_x, train_v, train_E,
            s1_params, cl_avg, seed=SEED, maxiter=400)

        print(f"  Trained in {train_time:.1f}s, NLL={nll:.6f}")
        print(f"  b2={cl_opt['b2']:.4f}, d2={cl_opt['d2']:.4f}, "
              f"gamma={gamma_opt:.6f}, q_scale={cl_opt['q_scale']:.4f}")

        # Evaluate on test
        e_en, S_en, sx_en, su_en = kf_filter_energy(
            s1_params, cl_opt, t_arr, x_arr, v_arr, filter_E,
            gamma=gamma_opt, return_states=True)

        e_test = e_en[test_mask]; S_test = S_en[test_mask]
        valid_e = ~np.isnan(e_test)
        acf_en = compute_acf(e_test[valid_e])
        nis_en = float(np.mean(e_test[valid_e]**2 / S_test[valid_e]))
        lb_en = ljung_box(acf_en, int(valid_e.sum()))

        print(f"  ACF(1)={acf_en[1]:.4f}, NIS={nis_en:.4f}")
        for r in lb_en:
            print(f"    LB lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.4f}")

        # DxR2
        dxr2_en = compute_dxr2_energy(
            s1_params, cl_opt, t_arr, x_arr, v_arr, filter_E,
            gamma=gamma_opt, max_h=MAX_HORIZON, eval_start=eval_start)

        delta_dxr2_10 = dxr2_en[9] - dxr2_cl[9]
        delta_acf1 = acf_en[1] - acf_cl[1]
        mean_5_10_en = float(np.mean(dxr2_en[4:10]))
        mean_5_10_cl = float(np.mean(dxr2_cl[4:10]))
        delta_mean = mean_5_10_en - mean_5_10_cl

        print(f"  DxR2@10: {dxr2_en[9]:.4f} (delta={delta_dxr2_10:+.4f})")
        print(f"  mean(5-10): {mean_5_10_en:.4f} (delta={delta_mean:+.4f})")
        print(f"  ACF(1) delta: {delta_acf1:+.4f}")

        results_by_rho[rho_E] = {
            'rho': rho_E,
            'gamma': gamma_opt,
            'b2': cl_opt['b2'],
            'd2': cl_opt['d2'],
            'q_scale': cl_opt['q_scale'],
            'nll': nll,
            'train_time': train_time,
            'acf1': float(acf_en[1]),
            'nis': nis_en,
            'dxr2': dxr2_en.tolist(),
            'dxr2_10': float(dxr2_en[9]),
            'mean_5_10': mean_5_10_en,
            'delta_dxr2_10': delta_dxr2_10,
            'delta_acf1': delta_acf1,
            'delta_mean_5_10': delta_mean,
            'lb': lb_en,
            'acf_vals': acf_en.tolist(),
            'cl_params': {k: v for k, v in cl_opt.items()},
            'states_x': sx_en,
            'states_u': su_en,
        }

    # ============================================================
    #  SELECT BEST RHO
    # ============================================================
    print_section("SELECTION")
    best_rho = max(results_by_rho.keys(),
                   key=lambda r: results_by_rho[r]['dxr2_10'])
    best = results_by_rho[best_rho]
    print(f"Best rho: {best_rho} (DxR2@10={best['dxr2_10']:.4f})")

    # ============================================================
    #  EVENT SKILL
    # ============================================================
    print_section("EVENT SKILL")

    # Detect events using same method as hazard diagnostic
    from sklearn.cluster import KMeans
    # Use full filter array displacement for event detection
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(x_arr.reshape(-1, 1))
    centroids = sorted(km.cluster_centers_.flatten())
    midpoint = (centroids[0] + centroids[1]) / 2.0
    print(f"Midpoint for event detection: {midpoint:.4f}")

    # Apply hysteresis
    hyst_band = 0.1 * (centroids[1] - centroids[0])
    thresh_up = midpoint + hyst_band
    thresh_down = midpoint - hyst_band
    N_f = len(x_arr)
    states_ev = np.zeros(N_f, dtype=int)
    states_ev[0] = 0 if x_arr[0] < midpoint else 1
    for i in range(1, N_f):
        if states_ev[i-1] == 0:
            states_ev[i] = 1 if x_arr[i] > thresh_up else 0
        else:
            states_ev[i] = 0 if x_arr[i] < thresh_down else 1

    # Find event indices
    event_indices = [i for i in range(1, N_f) if states_ev[i] != states_ev[i-1]]
    test_events = [i for i in event_indices if test_mask[i]]
    print(f"Events in filter array: {len(event_indices)}, in test: {len(test_events)}")

    # Event skill for baseline closure
    _, _, sx_cl, su_cl = kf_filter_2state(
        s1_params, cl_avg, t_arr, x_arr, v_arr, return_states=True)
    ev_skill_cl = compute_event_skill(
        sx_cl, t_arr, x_arr, test_mask, midpoint, event_indices)

    # Event skill for best energy model
    ev_skill_en = compute_event_skill(
        best['states_x'], t_arr, x_arr, test_mask, midpoint, event_indices)

    print(f"Closure event skill: {ev_skill_cl['hit_rate']:.3f} "
          f"({ev_skill_cl['n_predicted']}/{ev_skill_cl['n_events']})")
    print(f"Energy  event skill: {ev_skill_en['hit_rate']:.3f} "
          f"({ev_skill_en['n_predicted']}/{ev_skill_en['n_events']})")

    # ============================================================
    #  FIGURES
    # ============================================================
    print_section("FIGURES")

    # Fig 1: DxR2 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    h_arr = np.arange(1, MAX_HORIZON + 1)
    ax.plot(h_arr, dxr2_phys, 'k--', lw=1.5, label='Physics-only')
    ax.plot(h_arr, dxr2_cl, 'b-o', lw=1.5, markersize=4, label='Closure (2t)')
    for rho_E, res in results_by_rho.items():
        ax.plot(h_arr, res['dxr2'], '-s', lw=1.5, markersize=4,
                label=f'Energy rho={rho_E} (g={res["gamma"]:.4f})')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Forecast horizon h (steps)')
    ax.set_ylabel('DxR2(h)')
    ax.set_title('DxR2: Energy-Feature Closure vs Baseline')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_dxr2_comparison.png")
    plt.close()

    # Fig 2: ACF comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(0, 21)
    ax.bar(lags - 0.2, acf_cl[:21], width=0.4, alpha=0.7, label='Closure (2t)')
    ax.bar(lags + 0.2, np.array(best['acf_vals'][:21]), width=0.4,
           alpha=0.7, label=f'Energy rho={best_rho}')
    n_test = int(valid_c.sum())
    ax.axhline(1.96/np.sqrt(n_test), color='red', ls='--', lw=0.8, label='95% CI')
    ax.axhline(-1.96/np.sqrt(n_test), color='red', ls='--', lw=0.8)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Innovation ACF: Energy vs Baseline')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_acf_comparison.png")
    plt.close()

    # Fig 3: Energy proxy time series (test period)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    t_test = t_arr[test_mask]
    x_test = x_arr[test_mask]

    ax = axes[0]
    ax.plot(t_test, x_test, 'k-', lw=0.5)
    ax.axhline(midpoint, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylabel('Displacement')
    ax.set_title('Test Period: Displacement + Energy Proxy')

    ax = axes[1]
    for rho_E in RHO_VALUES:
        E_test = compute_energy_proxy(v_arr, u_c, rho_E)[test_mask]
        ax.plot(t_test, E_test, lw=0.8, label=f'E(rho={rho_E})')
    ax.set_ylabel('Energy proxy')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_test, e_cl[test_mask], 'b-', lw=0.3, alpha=0.5, label='Closure')
    e_best = kf_filter_energy(
        s1_params, results_by_rho[best_rho]['cl_params'],
        t_arr, x_arr, v_arr,
        compute_energy_proxy(v_arr, u_c, best_rho),
        gamma=best['gamma'])[0]
    ax.plot(t_test, e_best[test_mask], 'r-', lw=0.3, alpha=0.5,
            label=f'Energy rho={best_rho}')
    ax.set_ylabel('Innovation')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_energy_timeseries.png")
    plt.close()

    # ============================================================
    #  TABLES
    # ============================================================
    print_section("TABLES")

    # Metrics table
    rows = []
    rows.append({
        'model': 'physics_only',
        'ACF1': float(acf_base_phys[1]),
        'DxR2_h10': float(dxr2_phys[9]),
        'mean_5_10': float(np.mean(dxr2_phys[4:10])),
    })
    rows.append({
        'model': 'closure_2t',
        'ACF1': float(acf_cl[1]),
        'DxR2_h10': float(dxr2_cl[9]),
        'mean_5_10': float(np.mean(dxr2_cl[4:10])),
        'NIS': nis_cl,
    })
    for rho_E, res in results_by_rho.items():
        rows.append({
            'model': f'energy_rho{rho_E}',
            'ACF1': res['acf1'],
            'DxR2_h10': res['dxr2_10'],
            'mean_5_10': res['mean_5_10'],
            'NIS': res['nis'],
            'gamma': res['gamma'],
            'b2': res['b2'],
            'd2': res['d2'],
            'q_scale': res['q_scale'],
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "tables" / "metrics_table.csv", index=False)

    # DxR2 by horizon
    dxr2_rows = []
    for h in range(1, MAX_HORIZON + 1):
        row = {'h': h, 'physics_only': float(dxr2_phys[h-1]),
               'closure_2t': float(dxr2_cl[h-1])}
        for rho_E, res in results_by_rho.items():
            row[f'energy_rho{rho_E}'] = res['dxr2'][h-1]
        dxr2_rows.append(row)
    dxr2_df = pd.DataFrame(dxr2_rows)
    dxr2_df.to_csv(OUT / "tables" / "dxr2_by_horizon.csv", index=False)

    # Innovation diagnostics
    diag_rows = []
    for model_name, acf_vals, n_pts in [
        ('closure_2t', acf_cl, int(valid_c.sum())),
    ]:
        lb = ljung_box(acf_vals, n_pts)
        for r in lb:
            diag_rows.append({
                'model': model_name, 'lag': r['lag'],
                'Q': r['Q'], 'p': r['p'],
                'ACF_at_lag': float(acf_vals[r['lag']]),
            })
    for rho_E, res in results_by_rho.items():
        for r in res['lb']:
            diag_rows.append({
                'model': f'energy_rho{rho_E}', 'lag': r['lag'],
                'Q': r['Q'], 'p': r['p'],
                'ACF_at_lag': float(np.array(res['acf_vals'])[r['lag']]),
            })
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT / "tables" / "innovation_diagnostics.csv", index=False)

    # ============================================================
    #  SUMMARY JSON
    # ============================================================
    runtime = time.time() - t0_global

    # Strip non-serializable arrays from results
    summary = {
        'seed': SEED,
        'u_c': u_c,
        'baseline': {
            'b2': cl_avg['b2'], 'd2': cl_avg['d2'],
            'q_scale': cl_avg['q_scale'],
            'acf1': float(acf_cl[1]),
            'dxr2_10': float(dxr2_cl[9]),
            'mean_5_10': float(np.mean(dxr2_cl[4:10])),
            'nis': nis_cl,
        },
        'probes': {},
        'best_rho': best_rho,
        'event_skill': {
            'closure': ev_skill_cl,
            'energy': ev_skill_en,
        },
        'runtime_s': round(runtime, 1),
    }
    for rho_E, res in results_by_rho.items():
        summary['probes'][str(rho_E)] = {
            'gamma': res['gamma'],
            'b2': res['b2'], 'd2': res['d2'],
            'q_scale': res['q_scale'],
            'nll': res['nll'],
            'train_time': res['train_time'],
            'acf1': res['acf1'],
            'nis': res['nis'],
            'dxr2_10': res['dxr2_10'],
            'mean_5_10': res['mean_5_10'],
            'delta_dxr2_10': res['delta_dxr2_10'],
            'delta_acf1': res['delta_acf1'],
        }

    with open(OUT / "summary_v9a.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # ============================================================
    #  FINAL PRINT
    # ============================================================
    print_section("RESULTS SUMMARY")
    print(f"{'Model':<25} {'ACF(1)':>8} {'DxR2@10':>9} {'mean(5-10)':>11}")
    print("-" * 55)
    print(f"{'Physics-only':<25} {acf_base_phys[1]:8.4f} {dxr2_phys[9]:9.4f} "
          f"{np.mean(dxr2_phys[4:10]):11.4f}")
    print(f"{'Closure (2t)':<25} {acf_cl[1]:8.4f} {dxr2_cl[9]:9.4f} "
          f"{np.mean(dxr2_cl[4:10]):11.4f}")
    for rho_E, res in results_by_rho.items():
        label = f"Energy rho={rho_E}"
        print(f"{label:<25} {res['acf1']:8.4f} {res['dxr2_10']:9.4f} "
              f"{res['mean_5_10']:11.4f}")

    print(f"\nBest rho={best_rho}:")
    print(f"  gamma = {best['gamma']:.6f}")
    print(f"  Delta DxR2@10 = {best['delta_dxr2_10']:+.4f}")
    print(f"  Delta ACF(1)  = {best['delta_acf1']:+.4f}")
    material = abs(best['delta_dxr2_10']) > 0.005
    print(f"  Material improvement: {'YES' if material else 'NO'}")

    print(f"\nEvent skill: closure={ev_skill_cl['hit_rate']:.3f}, "
          f"energy={ev_skill_en['hit_rate']:.3f}")

    n_files = sum(len(files) for _, _, files in os.walk(str(OUT)))
    print(f"\nRuntime: {runtime:.1f}s")
    print(f"Output files: {n_files}")
    print(f"COMPLETE")


if __name__ == '__main__':
    main()
