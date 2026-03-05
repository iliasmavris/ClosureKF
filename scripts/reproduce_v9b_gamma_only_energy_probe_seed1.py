"""
v9b: Constrained gamma-only Energy Probe (1 seed, non-destructive)
==================================================================
Freezes ALL baseline closure/physics parameters and fits ONLY gamma
(the energy-memory coefficient) to isolate whether the causal energy
accumulation signal provides independent forecasting benefit.

Variant 1: learn gamma only (q_scale fixed)
Variant 2: learn gamma + q_scale

Usage: python scripts/reproduce_v9b_gamma_only_energy_probe_seed1.py
"""

import os, sys, math, json, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar, minimize as sp_minimize
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

OUT = ROOT / "final_lockbox_v9b_gamma_only_energy_probe"
for d in ['figures', 'tables']:
    (OUT / d).mkdir(parents=True, exist_ok=True)

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
#  ENERGY PROXY
# ============================================================

def compute_energy_proxy(v, u_c, rho):
    N = len(v)
    phi2 = np.abs(v) * np.maximum(0.0, np.abs(v) - u_c)
    E = np.zeros(N)
    E[0] = phi2[0]
    for i in range(1, N):
        E[i] = rho * E[i-1] + phi2[i]
    return E


def estimate_u_c(x, v):
    dx = np.diff(x, prepend=x[0])
    non_moving = np.abs(dx) < np.percentile(np.abs(dx), 50)
    return float(np.percentile(np.abs(v[non_moving]), 25))


# ============================================================
#  KF FILTER WITH ENERGY FEATURE (frozen closure)
# ============================================================

def kf_filter_energy(params, cl_params, t, x_obs, v, E,
                     gamma=0.0, return_states=False):
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

    _, _, states_x, states_u = kf_filter_energy(
        params, cl_params, t, x_obs, v, E, gamma=gamma, return_states=True)

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
#  STANDARD 2-STATE KF (baseline)
# ============================================================

def kf_filter_2state(params, cl_params, t, x_obs, v, return_states=False):
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
        innovations[k] = innov; S_values[k] = S_val
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
#  TRAINING: GAMMA ONLY (1-D Brent)
# ============================================================

def train_gamma_only(train_t, train_x, train_v, train_E,
                     s1_params, fixed_cl):
    """Optimize gamma alone via Brent's method on [-2, 2]."""
    t0 = time.time()

    def _objective(gamma):
        innov, S_vals = kf_filter_energy(
            s1_params, fixed_cl, train_t, train_x, train_v, train_E,
            gamma=gamma)
        valid = ~np.isnan(innov) & (S_vals > 0)
        if valid.sum() < 10:
            return 1e30
        e = innov[valid]; S = S_vals[valid]
        return float(0.5 * np.mean(np.log(S) + e**2 / S))

    result = minimize_scalar(_objective, bounds=(-2.0, 2.0), method='bounded',
                             options={'xatol': 1e-6, 'maxiter': 200})
    elapsed = time.time() - t0
    return float(result.x), float(result.fun), elapsed


# ============================================================
#  TRAINING: GAMMA + Q_SCALE (2-D Nelder-Mead)
# ============================================================

def train_gamma_qscale(train_t, train_x, train_v, train_E,
                       s1_params, fixed_cl):
    """Optimize gamma + q_scale with all other params frozen."""
    t0 = time.time()
    base_qsc = fixed_cl['q_scale']

    def _unpack(x_vec):
        gamma = float(x_vec[0])
        q_scale = float(np.exp(x_vec[1]))
        cl = dict(fixed_cl)
        cl['q_scale'] = q_scale
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

    x0 = np.array([0.0, np.log(base_qsc)], dtype=np.float64)
    result = sp_minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-6,
                                  'fatol': 1e-8, 'adaptive': True})
    cl_opt, gamma_opt = _unpack(result.x)
    elapsed = time.time() - t0
    return cl_opt, gamma_opt, float(result.fun), elapsed


# ============================================================
#  MAIN
# ============================================================

def main():
    t0_global = time.time()
    np.random.seed(SEED)
    device = torch.device('cpu')

    print_section("v9b: Constrained gamma-only Energy Probe")
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
    print(f"FROZEN closure: b2={cl_avg['b2']:.4f}, d2={cl_avg['d2']:.4f}, "
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

    # ---- u_c from TRAIN ONLY ----
    u_c = estimate_u_c(train_x, train_v)
    print(f"u_c (train-only): {u_c:.4f} m/s")

    # ============================================================
    #  BASELINE
    # ============================================================
    print_section("BASELINE: Closure (2t), gamma=0")

    e_cl, S_cl = kf_filter_2state(s1_params, cl_avg, t_arr, x_arr, v_arr)
    e_cm = e_cl[test_mask]; S_cm = S_cl[test_mask]; valid_c = ~np.isnan(e_cm)
    acf_cl = compute_acf(e_cm[valid_c])
    nis_cl = float(np.mean(e_cm[valid_c]**2 / S_cm[valid_c]))
    lb_cl = ljung_box(acf_cl, int(valid_c.sum()))
    nll_base = float(0.5 * np.mean(np.log(S_cm[valid_c]) + e_cm[valid_c]**2 / S_cm[valid_c]))

    dxr2_cl = compute_dxr2_hstep(
        s1_params, cl_avg, t_arr, x_arr, v_arr,
        MAX_HORIZON, eval_start=eval_start)

    print(f"ACF(1)={acf_cl[1]:.4f}, DxR2@10={dxr2_cl[9]:.4f}, NIS={nis_cl:.4f}")
    print(f"Test NLL={nll_base:.6f}")
    for r in lb_cl:
        print(f"  LB lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.4f}")

    # ============================================================
    #  PROBES
    # ============================================================
    all_results = {}

    for rho_E in RHO_VALUES:
        train_E = compute_energy_proxy(train_v, u_c, rho_E)
        filter_E = compute_energy_proxy(v_arr, u_c, rho_E)

        # ---- Variant 1: gamma only ----
        label_v1 = f"gamma_only_rho{rho_E}"
        print_section(f"VARIANT 1: gamma-only, rho={rho_E}")

        gamma_v1, nll_v1, time_v1 = train_gamma_only(
            train_t, train_x, train_v, train_E,
            s1_params, cl_avg)

        print(f"  gamma={gamma_v1:.6f}, train NLL={nll_v1:.6f}, time={time_v1:.1f}s")

        e_v1, S_v1 = kf_filter_energy(
            s1_params, cl_avg, t_arr, x_arr, v_arr, filter_E,
            gamma=gamma_v1)
        e_v1t = e_v1[test_mask]; S_v1t = S_v1[test_mask]
        valid_v1 = ~np.isnan(e_v1t)
        acf_v1 = compute_acf(e_v1t[valid_v1])
        nis_v1 = float(np.mean(e_v1t[valid_v1]**2 / S_v1t[valid_v1]))
        lb_v1 = ljung_box(acf_v1, int(valid_v1.sum()))
        nll_v1_test = float(0.5 * np.mean(np.log(S_v1t[valid_v1]) + e_v1t[valid_v1]**2 / S_v1t[valid_v1]))

        dxr2_v1 = compute_dxr2_energy(
            s1_params, cl_avg, t_arr, x_arr, v_arr, filter_E,
            gamma=gamma_v1, max_h=MAX_HORIZON, eval_start=eval_start)

        d_dxr2 = dxr2_v1[9] - dxr2_cl[9]
        d_acf = acf_v1[1] - acf_cl[1]
        print(f"  ACF(1)={acf_v1[1]:.4f} (d={d_acf:+.4f}), "
              f"DxR2@10={dxr2_v1[9]:.4f} (d={d_dxr2:+.4f}), "
              f"NIS={nis_v1:.4f}")
        print(f"  Test NLL={nll_v1_test:.6f} (d={nll_v1_test-nll_base:+.6f})")

        all_results[label_v1] = {
            'variant': 'gamma_only', 'rho': rho_E,
            'gamma': gamma_v1, 'q_scale': cl_avg['q_scale'],
            'nll_train': nll_v1, 'nll_test': nll_v1_test,
            'train_time': time_v1,
            'acf1': float(acf_v1[1]), 'nis': nis_v1,
            'dxr2': dxr2_v1.tolist(),
            'dxr2_10': float(dxr2_v1[9]),
            'mean_5_10': float(np.mean(dxr2_v1[4:10])),
            'delta_dxr2_10': d_dxr2,
            'delta_acf1': d_acf,
            'lb': lb_v1,
            'acf_vals': acf_v1[:21].tolist(),
        }

        # ---- Variant 2: gamma + q_scale ----
        label_v2 = f"gamma_qscale_rho{rho_E}"
        print_section(f"VARIANT 2: gamma+q_scale, rho={rho_E}")

        cl_v2, gamma_v2, nll_v2, time_v2 = train_gamma_qscale(
            train_t, train_x, train_v, train_E,
            s1_params, cl_avg)

        print(f"  gamma={gamma_v2:.6f}, q_scale={cl_v2['q_scale']:.4f}, "
              f"train NLL={nll_v2:.6f}, time={time_v2:.1f}s")

        e_v2, S_v2 = kf_filter_energy(
            s1_params, cl_v2, t_arr, x_arr, v_arr, filter_E,
            gamma=gamma_v2)
        e_v2t = e_v2[test_mask]; S_v2t = S_v2[test_mask]
        valid_v2 = ~np.isnan(e_v2t)
        acf_v2 = compute_acf(e_v2t[valid_v2])
        nis_v2 = float(np.mean(e_v2t[valid_v2]**2 / S_v2t[valid_v2]))
        lb_v2 = ljung_box(acf_v2, int(valid_v2.sum()))
        nll_v2_test = float(0.5 * np.mean(np.log(S_v2t[valid_v2]) + e_v2t[valid_v2]**2 / S_v2t[valid_v2]))

        dxr2_v2 = compute_dxr2_energy(
            s1_params, cl_v2, t_arr, x_arr, v_arr, filter_E,
            gamma=gamma_v2, max_h=MAX_HORIZON, eval_start=eval_start)

        d_dxr2_v2 = dxr2_v2[9] - dxr2_cl[9]
        d_acf_v2 = acf_v2[1] - acf_cl[1]
        print(f"  ACF(1)={acf_v2[1]:.4f} (d={d_acf_v2:+.4f}), "
              f"DxR2@10={dxr2_v2[9]:.4f} (d={d_dxr2_v2:+.4f}), "
              f"NIS={nis_v2:.4f}")
        print(f"  Test NLL={nll_v2_test:.6f} (d={nll_v2_test-nll_base:+.6f})")

        all_results[label_v2] = {
            'variant': 'gamma_qscale', 'rho': rho_E,
            'gamma': gamma_v2, 'q_scale': cl_v2['q_scale'],
            'nll_train': nll_v2, 'nll_test': nll_v2_test,
            'train_time': time_v2,
            'acf1': float(acf_v2[1]), 'nis': nis_v2,
            'dxr2': dxr2_v2.tolist(),
            'dxr2_10': float(dxr2_v2[9]),
            'mean_5_10': float(np.mean(dxr2_v2[4:10])),
            'delta_dxr2_10': d_dxr2_v2,
            'delta_acf1': d_acf_v2,
            'lb': lb_v2,
            'acf_vals': acf_v2[:21].tolist(),
        }

    # ============================================================
    #  FIGURES
    # ============================================================
    print_section("FIGURES")

    # Fig 1: DxR2 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    h_arr = np.arange(1, MAX_HORIZON + 1)
    ax.plot(h_arr, dxr2_cl, 'b-o', lw=2, markersize=5, label='Closure (2t) baseline')
    colors = {'0.9': 'red', '0.95': 'green'}
    for key, res in all_results.items():
        rho_s = str(res['rho'])
        ls = '-' if 'gamma_only' in key else '--'
        mk = 's' if 'gamma_only' in key else '^'
        short = 'g-only' if 'gamma_only' in key else 'g+qsc'
        ax.plot(h_arr, res['dxr2'], ls=ls, marker=mk, lw=1.5, markersize=4,
                color=colors[rho_s],
                label=f'{short} rho={rho_s} (g={res["gamma"]:.4f})')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Forecast horizon h (steps)')
    ax.set_ylabel('DxR2(h)')
    ax.set_title('v9b: Constrained gamma-only Energy Probe')
    ax.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_dxr2_comparison.png")
    plt.close()

    # Fig 2: ACF comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(0, 21)
    width = 0.15
    offsets = [-2, -1, 0, 1]
    ax.bar(lags + offsets[0]*width, acf_cl[:21], width=width, alpha=0.7,
           label='Baseline')
    idx = 1
    for key, res in all_results.items():
        short = key.replace('gamma_only_', 'g-').replace('gamma_qscale_', 'gq-')
        ax.bar(lags + offsets[idx]*width, res['acf_vals'][:21],
               width=width, alpha=0.7, label=short)
        idx += 1
        if idx >= len(offsets):
            break
    n_test = int(valid_c.sum())
    ax.axhline(1.96/np.sqrt(n_test), color='red', ls='--', lw=0.8, label='95% CI')
    ax.axhline(-1.96/np.sqrt(n_test), color='red', ls='--', lw=0.8)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Innovation ACF: gamma-only vs Baseline')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_acf_comparison.png")
    plt.close()

    # ============================================================
    #  TABLES
    # ============================================================
    print_section("TABLES")

    # Metrics summary
    rows = [{'model': 'closure_2t_baseline',
             'ACF1': float(acf_cl[1]),
             'DxR2_h10': float(dxr2_cl[9]),
             'mean_5_10': float(np.mean(dxr2_cl[4:10])),
             'NIS': nis_cl, 'gamma': 0.0,
             'q_scale': cl_avg['q_scale'],
             'NLL_test': nll_base}]
    for key, res in all_results.items():
        rows.append({
            'model': key,
            'ACF1': res['acf1'],
            'DxR2_h10': res['dxr2_10'],
            'mean_5_10': res['mean_5_10'],
            'NIS': res['nis'],
            'gamma': res['gamma'],
            'q_scale': res['q_scale'],
            'NLL_test': res['nll_test'],
            'delta_DxR2_h10': res['delta_dxr2_10'],
            'delta_ACF1': res['delta_acf1'],
        })
    pd.DataFrame(rows).to_csv(OUT / "tables" / "metrics_table.csv", index=False)

    # DxR2 by horizon
    dxr2_rows = []
    for h in range(1, MAX_HORIZON + 1):
        row = {'h': h, 'baseline': float(dxr2_cl[h-1])}
        for key, res in all_results.items():
            row[key] = res['dxr2'][h-1]
        dxr2_rows.append(row)
    pd.DataFrame(dxr2_rows).to_csv(OUT / "tables" / "dxr2_by_horizon.csv", index=False)

    # Innovation diagnostics
    diag_rows = []
    for model_name, acf_vals, n_pts in [('baseline', acf_cl, int(valid_c.sum()))]:
        lb = ljung_box(acf_vals, n_pts)
        for r in lb:
            diag_rows.append({'model': model_name, 'lag': r['lag'],
                              'Q': r['Q'], 'p': r['p'],
                              'ACF_at_lag': float(acf_vals[r['lag']])})
    for key, res in all_results.items():
        for r in res['lb']:
            diag_rows.append({'model': key, 'lag': r['lag'],
                              'Q': r['Q'], 'p': r['p'],
                              'ACF_at_lag': float(np.array(res['acf_vals'])[min(r['lag'], len(res['acf_vals'])-1)])})
    pd.DataFrame(diag_rows).to_csv(
        OUT / "tables" / "innovation_diagnostics.csv", index=False)

    # ============================================================
    #  DECISION
    # ============================================================
    print_section("DECISION")

    # Check if any variant improved DxR2@10 by > 0.005
    any_improved = False
    gamma_near_zero = True
    for key, res in all_results.items():
        if res['delta_dxr2_10'] > 0.005:
            any_improved = True
        if abs(res['gamma']) > 0.01:
            gamma_near_zero = False

    if any_improved:
        verdict = ("MATERIAL IMPROVEMENT: At least one constrained gamma variant "
                   "improved DxR2@10 by > 0.005. Consider for SI.")
    elif gamma_near_zero:
        verdict = ("GAMMA SHRINKS TO ZERO: Energy-memory feature provides no material "
                   "forecasting benefit under strict parsimony; gamma -> 0 confirms the "
                   "causal accumulation term carries no independent predictive information "
                   "beyond what b2*du - d2*v|u| already captures.")
    else:
        verdict = ("NO IMPROVEMENT: Energy-memory feature provides no material forecasting "
                   "benefit under strict parsimony; residual structure is not correctable "
                   "by this causal accumulation term.")

    print(f"Any DxR2@10 improvement > 0.005? {'YES' if any_improved else 'NO'}")
    print(f"All gamma near zero (|g|<0.01)? {'YES' if gamma_near_zero else 'NO'}")
    print(f"\nVERDICT: {verdict}")

    # ============================================================
    #  SUMMARY JSON
    # ============================================================
    runtime = time.time() - t0_global

    summary = {
        'seed': SEED, 'u_c': u_c,
        'frozen_params': {
            'b2': cl_avg['b2'], 'd2': cl_avg['d2'],
            'q_scale': cl_avg['q_scale'],
        },
        'baseline': {
            'acf1': float(acf_cl[1]),
            'dxr2_10': float(dxr2_cl[9]),
            'mean_5_10': float(np.mean(dxr2_cl[4:10])),
            'nis': nis_cl, 'nll_test': nll_base,
        },
        'probes': {},
        'verdict': verdict,
        'any_improved': any_improved,
        'gamma_near_zero': gamma_near_zero,
        'runtime_s': round(runtime, 1),
    }
    for key, res in all_results.items():
        summary['probes'][key] = {
            k: v for k, v in res.items()
            if k not in ('acf_vals', 'lb', 'dxr2')
        }
        summary['probes'][key]['lb'] = res['lb']

    with open(OUT / "summary_v9b.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # ============================================================
    #  README
    # ============================================================
    readme = [
        "# v9b: Constrained gamma-only Energy Probe",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime:** {runtime:.0f}s",
        f"**Seed:** {SEED}",
        "",
        "## Frozen Parameters",
        "",
        f"- b2 = {cl_avg['b2']:.4f} (fixed)",
        f"- d2 = {cl_avg['d2']:.4f} (fixed)",
        f"- q_scale = {cl_avg['q_scale']:.4f} (fixed in Variant 1, free in Variant 2)",
        f"- All physics params fixed at stage-1 values",
        f"- u_c = {u_c:.4f} m/s (train-only, 25th pctile rule)",
        "",
        "## Results",
        "",
        "| Model | gamma | q_scale | ACF(1) | DxR2@10 | mean(5-10) | NIS |",
        "|-------|-------|---------|--------|---------|------------|-----|",
        f"| Baseline (g=0) | 0.0000 | {cl_avg['q_scale']:.4f} | "
        f"{acf_cl[1]:.4f} | {dxr2_cl[9]:.4f} | {np.mean(dxr2_cl[4:10]):.4f} | "
        f"{nis_cl:.4f} |",
    ]
    for key, res in all_results.items():
        readme.append(
            f"| {key} | {res['gamma']:.4f} | {res['q_scale']:.4f} | "
            f"{res['acf1']:.4f} | {res['dxr2_10']:.4f} | {res['mean_5_10']:.4f} | "
            f"{res['nis']:.4f} |")
    readme.extend([
        "",
        "## Delta vs Baseline",
        "",
        "| Model | Delta ACF(1) | Delta DxR2@10 | Delta mean(5-10) |",
        "|-------|-------------|---------------|-------------------|",
    ])
    for key, res in all_results.items():
        readme.append(
            f"| {key} | {res['delta_acf1']:+.4f} | {res['delta_dxr2_10']:+.4f} | "
            f"{res['mean_5_10']-np.mean(dxr2_cl[4:10]):+.4f} |")
    readme.extend([
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Output Files",
        "",
        "- `figures/fig_dxr2_comparison.png`",
        "- `figures/fig_acf_comparison.png`",
        "- `tables/metrics_table.csv`",
        "- `tables/dxr2_by_horizon.csv`",
        "- `tables/innovation_diagnostics.csv`",
        "- `summary_v9b.json`",
        "- `README.md`",
    ])

    with open(OUT / "README.md", 'w') as f:
        f.write('\n'.join(readme))

    # ============================================================
    #  FINAL PRINT
    # ============================================================
    print_section("FINAL SUMMARY")
    print(f"{'Model':<30} {'gamma':>8} {'ACF(1)':>8} {'DxR2@10':>9} {'d_DxR2':>8}")
    print("-" * 66)
    print(f"{'Baseline (gamma=0)':<30} {'0.0000':>8} {acf_cl[1]:8.4f} "
          f"{dxr2_cl[9]:9.4f} {'--':>8}")
    for key, res in all_results.items():
        short = key[:28]
        print(f"{short:<30} {res['gamma']:8.4f} {res['acf1']:8.4f} "
              f"{res['dxr2_10']:9.4f} {res['delta_dxr2_10']:+8.4f}")

    n_files = sum(len(files) for _, _, files in os.walk(str(OUT)))
    print(f"\nRuntime: {runtime:.1f}s")
    print(f"Output files: {n_files}")
    print(f"\nVERDICT: {verdict}")


if __name__ == '__main__':
    main()
