"""
v9 Part B: Latent AR(1) Forcing State Probe (1 seed, non-destructive)
=====================================================================
Tests whether a 3-state KF with a latent AR(1) forcing variable g_t
improves DxR2 and innovation whiteness over the 2-term closure baseline.

Model:
  State = [x, u, g]
  x_{t+1} = x + u*dt
  u_{t+1} = rho_u*u - kappa*x*dt + c*g(v)*dt + [closure]*dt + g*dt
  g_{t+1} = rho_g*g + eps,  eps ~ N(0, q_g)

Learns rho_g (bounded [0.85, 0.99]) and q_g, plus b2/d2/q_scale.

Usage: python scripts/reproduce_v9_latent_force_probe_seed1.py
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

OUT = ROOT / "final_lockbox_v9_latent_force_probe"
for d in ['figures', 'tables']:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ===== Constants =====
DT = 0.1
SEED = 1
MAX_HORIZON = 10
SEEDS_REF = [42, 43, 44]

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
#  3-STATE KF FILTER: [x, u, g]
# ============================================================

def kf_filter_3state(params, cl_params, t, x_obs, v,
                     rho_g=0.95, q_g=0.001, return_states=False):
    """3-state KF: state = [x, u, g], g is AR(1) latent forcing.
    g enters the velocity update additively: u_{t+1} includes + g*dt.
    """
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_x = np.zeros(N) if return_states else None
    states_u = np.zeros(N) if return_states else None
    states_g = np.zeros(N) if return_states else None

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']

    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    # State: [x, u, g]
    s = np.array([x_obs[0], 0.0, 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu'], q_g / (1 - rho_g**2 + 1e-12)])

    if return_states:
        states_x[0] = s[0]; states_u[0] = s[1]; states_g[0] = s[2]

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g_phys = max(v[k-1]**2 - vc**2, 0.0)

        u_st, v_w, g_st = s[1], v[k-1], s[2]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0

        # Closure: b2*dv - d2*u|v|
        cl = b2_v * dv_w - d2 * u_st * abs(v_w)
        cl_dt = cl * dt

        # Predict
        x_p = s[0] + s[1] * dt
        u_p = rho_u * s[1] - kap * s[0] * dt + c_val * g_phys * dt + cl_dt + g_st * dt
        g_p = rho_g * g_st

        s_pred = np.array([x_p, u_p, g_p])

        # Jacobian F (3x3)
        F = np.array([
            [1.0,    dt,    0.0],
            [-kap*dt, rho_u, dt],
            [0.0,    0.0,   rho_g],
        ])

        # Process noise
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt, q_g])
        P_pred = F @ P @ F.T + Q

        # Observation: H = [1, 0, 0]
        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        innovations[k] = innov
        S_values[k] = S_val

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        H_vec = np.array([1.0, 0.0, 0.0])
        IKH = np.eye(3) - np.outer(K, H_vec)
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)

        if return_states:
            states_x[k] = s[0]; states_u[k] = s[1]; states_g[k] = s[2]

    out = [innovations, S_values]
    if return_states:
        out += [states_x, states_u, states_g]
    return tuple(out)


# ============================================================
#  DxR2 FOR 3-STATE MODEL
# ============================================================

def compute_dxr2_3state(params, cl_params, t, x_obs, v,
                        rho_g=0.95, q_g=0.001,
                        max_h=10, eval_start=1):
    """DxR2(h) for 3-state model with latent AR(1) g."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    def _predict_step_3(sx, su, sg, v_w, dv_w, dt_k):
        rho_u = math.exp(-alpha * dt_k)
        g_phys = max(v_w**2 - vc**2, 0.0)
        cl = b2_v * dv_w - d2 * su * abs(v_w)
        x_new = sx + su * dt_k
        u_new = rho_u*su - kap*sx*dt_k + c_val*g_phys*dt_k + cl*dt_k + sg*dt_k
        g_new = rho_g * sg  # deterministic rollout (mean)
        return x_new, u_new, g_new

    # Pass 1: KF filter to get post-update states
    result = kf_filter_3state(
        params, cl_params, t, x_obs, v,
        rho_g=rho_g, q_g=q_g, return_states=True)
    _, _, states_x, states_u, states_g = result

    # Pass 2: h-step open-loop predictions
    r2_arr = np.zeros(max_h)
    for h in range(1, max_h + 1):
        dx_pred_list = []; dx_obs_list = []
        for i in range(max(eval_start, 1), N - h):
            sx, su, sg = states_x[i], states_u[i], states_g[i]
            for step in range(h):
                k_s = i + step + 1
                if k_s >= N: break
                dt_s = t[k_s] - t[k_s - 1]
                if dt_s <= 0: dt_s = 0.1
                v_w = v[k_s - 1]
                dv_w = v[k_s - 1] - v[k_s - 2] if k_s >= 2 else 0.0
                sx, su, sg = _predict_step_3(sx, su, sg, v_w, dv_w, dt_s)
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
    """Standard 2-state KF for baseline."""
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
#  TRAINING: 3-STATE MODEL
# ============================================================

def train_3state(train_t, train_x, train_v, s1_params, init_cl,
                 seed=1, maxiter=500):
    """Train b2, d2, q_scale, rho_g, q_g via Nelder-Mead on 3-state NLL.
    rho_g bounded to [0.85, 0.99] via sigmoid transform.
    q_g via log transform (positive).
    """
    np.random.seed(seed)
    t0 = time.time()

    def _softplus(x):
        return np.log1p(np.exp(x)) if x < 20 else x

    def _inv_softplus(y):
        return float(np.log(np.exp(y) - 1)) if y < 20 else y

    def _bounded_sigmoid(x, lo=0.85, hi=0.99):
        """Map real line to [lo, hi] via sigmoid."""
        s = 1.0 / (1.0 + np.exp(-x))
        return lo + (hi - lo) * s

    def _inv_bounded_sigmoid(y, lo=0.85, hi=0.99):
        s = (y - lo) / (hi - lo)
        s = np.clip(s, 1e-6, 1-1e-6)
        return float(np.log(s / (1 - s)))

    def _unpack(x_vec):
        cl = zero_closure()
        cl['b2'] = float(x_vec[0])
        cl['d2'] = float(_softplus(x_vec[1]))
        cl['q_scale'] = float(np.exp(x_vec[2]))
        rho_g = float(_bounded_sigmoid(x_vec[3]))
        q_g = float(np.exp(x_vec[4]))
        return cl, rho_g, q_g

    def _objective(x_vec):
        cl, rho_g, q_g = _unpack(x_vec)
        innov, S_vals = kf_filter_3state(
            s1_params, cl, train_t, train_x, train_v,
            rho_g=rho_g, q_g=q_g)
        valid = ~np.isnan(innov) & (S_vals > 0)
        if valid.sum() < 10:
            return 1e30
        e = innov[valid]; S = S_vals[valid]
        return float(0.5 * np.mean(np.log(S) + e**2 / S))

    x0 = np.array([
        init_cl['b2'],
        _inv_softplus(init_cl['d2']),
        np.log(init_cl['q_scale']),
        _inv_bounded_sigmoid(0.95),  # init rho_g at 0.95 (tau ~ 2s)
        np.log(0.001),              # init q_g at 0.001
    ], dtype=np.float64)

    result = sp_minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-4,
                                  'fatol': 1e-7, 'adaptive': True})
    cl_opt, rho_g_opt, q_g_opt = _unpack(result.x)
    elapsed = time.time() - t0
    return cl_opt, rho_g_opt, q_g_opt, float(result.fun), elapsed


def train_3state_fixed_closure(train_t, train_x, train_v, s1_params,
                               fixed_cl, seed=1, maxiter=300):
    """Train only rho_g and q_g with b2/d2/q_scale fixed at baseline values."""
    np.random.seed(seed)
    t0 = time.time()

    def _bounded_sigmoid(x, lo=0.85, hi=0.99):
        s = 1.0 / (1.0 + np.exp(-x))
        return lo + (hi - lo) * s

    def _inv_bounded_sigmoid(y, lo=0.85, hi=0.99):
        s = (y - lo) / (hi - lo)
        s = np.clip(s, 1e-6, 1-1e-6)
        return float(np.log(s / (1 - s)))

    def _unpack(x_vec):
        rho_g = float(_bounded_sigmoid(x_vec[0]))
        q_g = float(np.exp(x_vec[1]))
        return rho_g, q_g

    def _objective(x_vec):
        rho_g, q_g = _unpack(x_vec)
        innov, S_vals = kf_filter_3state(
            s1_params, fixed_cl, train_t, train_x, train_v,
            rho_g=rho_g, q_g=q_g)
        valid = ~np.isnan(innov) & (S_vals > 0)
        if valid.sum() < 10:
            return 1e30
        e = innov[valid]; S = S_vals[valid]
        return float(0.5 * np.mean(np.log(S) + e**2 / S))

    x0 = np.array([
        _inv_bounded_sigmoid(0.95),
        np.log(0.001),
    ], dtype=np.float64)

    result = sp_minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-4,
                                  'fatol': 1e-7, 'adaptive': True})
    rho_g_opt, q_g_opt = _unpack(result.x)
    elapsed = time.time() - t0
    return rho_g_opt, q_g_opt, float(result.fun), elapsed


# ============================================================
#  MAIN
# ============================================================

def main():
    t0_global = time.time()
    np.random.seed(SEED)
    device = torch.device('cpu')

    print_section("v9 Part B: Latent AR(1) Forcing State Probe")
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

    # ============================================================
    #  BASELINE EVALUATION
    # ============================================================
    print_section("BASELINE: Standard 2-state KF + 2-term Closure")

    # Physics-only
    e_base, S_base = kf_filter_2state(
        s1_params, zero_closure(), t_arr, x_arr, v_arr)
    e_bm = e_base[test_mask]; valid_b = ~np.isnan(e_bm)
    acf_base_phys = compute_acf(e_bm[valid_b])
    dxr2_phys = compute_dxr2_hstep(
        s1_params, zero_closure(), t_arr, x_arr, v_arr,
        MAX_HORIZON, eval_start=eval_start)

    # Closure
    e_cl, S_cl = kf_filter_2state(s1_params, cl_avg, t_arr, x_arr, v_arr)
    e_cm = e_cl[test_mask]; S_cm = S_cl[test_mask]; valid_c = ~np.isnan(e_cm)
    acf_cl = compute_acf(e_cm[valid_c])
    nis_cl = float(np.mean(e_cm[valid_c]**2 / S_cm[valid_c]))
    lb_cl = ljung_box(acf_cl, int(valid_c.sum()))
    dxr2_cl = compute_dxr2_hstep(
        s1_params, cl_avg, t_arr, x_arr, v_arr,
        MAX_HORIZON, eval_start=eval_start)

    print(f"Physics-only: ACF(1)={acf_base_phys[1]:.4f}, DxR2@10={dxr2_phys[9]:.4f}")
    print(f"Closure (2t): ACF(1)={acf_cl[1]:.4f}, DxR2@10={dxr2_cl[9]:.4f}, "
          f"NIS={nis_cl:.4f}")
    for r in lb_cl:
        print(f"  LB lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.4f}")

    # ============================================================
    #  VARIANT 1: JOINT (learn b2, d2, q_scale, rho_g, q_g)
    # ============================================================
    print_section("VARIANT 1: Joint (b2, d2, q_scale, rho_g, q_g)")

    cl_joint, rho_g_joint, q_g_joint, nll_joint, time_joint = train_3state(
        train_t, train_x, train_v, s1_params, cl_avg,
        seed=SEED, maxiter=500)

    tau_g_joint = -DT / np.log(rho_g_joint) if rho_g_joint > 0 else float('inf')
    print(f"  Trained in {time_joint:.1f}s, NLL={nll_joint:.6f}")
    print(f"  rho_g={rho_g_joint:.4f} (tau_g={tau_g_joint:.2f}s), q_g={q_g_joint:.6f}")
    print(f"  b2={cl_joint['b2']:.4f}, d2={cl_joint['d2']:.4f}, "
          f"q_scale={cl_joint['q_scale']:.4f}")

    # Evaluate
    e_j, S_j, sx_j, su_j, sg_j = kf_filter_3state(
        s1_params, cl_joint, t_arr, x_arr, v_arr,
        rho_g=rho_g_joint, q_g=q_g_joint, return_states=True)
    e_jt = e_j[test_mask]; S_jt = S_j[test_mask]; valid_j = ~np.isnan(e_jt)
    acf_joint = compute_acf(e_jt[valid_j])
    nis_joint = float(np.mean(e_jt[valid_j]**2 / S_jt[valid_j]))
    lb_joint = ljung_box(acf_joint, int(valid_j.sum()))

    dxr2_joint = compute_dxr2_3state(
        s1_params, cl_joint, t_arr, x_arr, v_arr,
        rho_g=rho_g_joint, q_g=q_g_joint,
        max_h=MAX_HORIZON, eval_start=eval_start)

    print(f"  ACF(1)={acf_joint[1]:.4f}, DxR2@10={dxr2_joint[9]:.4f}, "
          f"NIS={nis_joint:.4f}")
    print(f"  Delta DxR2@10={dxr2_joint[9]-dxr2_cl[9]:+.4f}, "
          f"Delta ACF(1)={acf_joint[1]-acf_cl[1]:+.4f}")

    # ============================================================
    #  VARIANT 2: FIXED CLOSURE (learn only rho_g, q_g)
    # ============================================================
    print_section("VARIANT 2: Fixed closure (learn rho_g, q_g only)")

    rho_g_fix, q_g_fix, nll_fix, time_fix = train_3state_fixed_closure(
        train_t, train_x, train_v, s1_params, cl_avg,
        seed=SEED, maxiter=300)

    tau_g_fix = -DT / np.log(rho_g_fix) if rho_g_fix > 0 else float('inf')
    print(f"  Trained in {time_fix:.1f}s, NLL={nll_fix:.6f}")
    print(f"  rho_g={rho_g_fix:.4f} (tau_g={tau_g_fix:.2f}s), q_g={q_g_fix:.6f}")

    # Evaluate
    e_f, S_f, sx_f, su_f, sg_f = kf_filter_3state(
        s1_params, cl_avg, t_arr, x_arr, v_arr,
        rho_g=rho_g_fix, q_g=q_g_fix, return_states=True)
    e_ft = e_f[test_mask]; S_ft = S_f[test_mask]; valid_f = ~np.isnan(e_ft)
    acf_fix = compute_acf(e_ft[valid_f])
    nis_fix = float(np.mean(e_ft[valid_f]**2 / S_ft[valid_f]))
    lb_fix = ljung_box(acf_fix, int(valid_f.sum()))

    dxr2_fix = compute_dxr2_3state(
        s1_params, cl_avg, t_arr, x_arr, v_arr,
        rho_g=rho_g_fix, q_g=q_g_fix,
        max_h=MAX_HORIZON, eval_start=eval_start)

    print(f"  ACF(1)={acf_fix[1]:.4f}, DxR2@10={dxr2_fix[9]:.4f}, "
          f"NIS={nis_fix:.4f}")
    print(f"  Delta DxR2@10={dxr2_fix[9]-dxr2_cl[9]:+.4f}, "
          f"Delta ACF(1)={acf_fix[1]-acf_cl[1]:+.4f}")

    # ============================================================
    #  SELECT BEST VARIANT
    # ============================================================
    print_section("SELECTION")
    # Prefer fixed closure variant if DxR2@10 is within 0.005
    if dxr2_fix[9] >= dxr2_joint[9] - 0.005:
        best_label = "fixed_closure"
        best_rho_g = rho_g_fix; best_q_g = q_g_fix
        best_cl = cl_avg; best_dxr2 = dxr2_fix
        best_acf = acf_fix; best_lb = lb_fix
        best_nis = nis_fix; best_nll = nll_fix
        best_sg = sg_f; best_tau = tau_g_fix
    else:
        best_label = "joint"
        best_rho_g = rho_g_joint; best_q_g = q_g_joint
        best_cl = cl_joint; best_dxr2 = dxr2_joint
        best_acf = acf_joint; best_lb = lb_joint
        best_nis = nis_joint; best_nll = nll_joint
        best_sg = sg_j; best_tau = tau_g_joint
    print(f"Selected: {best_label}")

    # ============================================================
    #  FIGURES
    # ============================================================
    print_section("FIGURES")

    # Fig 1: DxR2 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    h_arr = np.arange(1, MAX_HORIZON + 1)
    ax.plot(h_arr, dxr2_phys, 'k--', lw=1.5, label='Physics-only')
    ax.plot(h_arr, dxr2_cl, 'b-o', lw=1.5, markersize=4, label='Closure (2t)')
    ax.plot(h_arr, dxr2_joint, 'r-s', lw=1.5, markersize=4,
            label=f'3-state joint (rho_g={rho_g_joint:.3f})')
    ax.plot(h_arr, dxr2_fix, 'g-^', lw=1.5, markersize=4,
            label=f'3-state fixed (rho_g={rho_g_fix:.3f})')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Forecast horizon h (steps)')
    ax.set_ylabel('DxR2(h)')
    ax.set_title('DxR2: 3-State Latent Force vs Baseline')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_dxr2_comparison.png")
    plt.close()

    # Fig 2: ACF comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(0, 21)
    ax.bar(lags - 0.3, acf_cl[:21], width=0.3, alpha=0.7, label='Closure (2t)')
    ax.bar(lags, acf_joint[:21], width=0.3, alpha=0.7, label='3-state joint')
    ax.bar(lags + 0.3, acf_fix[:21], width=0.3, alpha=0.7, label='3-state fixed')
    n_test = int(valid_c.sum())
    ax.axhline(1.96/np.sqrt(n_test), color='red', ls='--', lw=0.8)
    ax.axhline(-1.96/np.sqrt(n_test), color='red', ls='--', lw=0.8)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Innovation ACF: 3-State vs Baseline')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_acf_comparison.png")
    plt.close()

    # Fig 3: Latent g state over time (test period)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    t_test = t_arr[test_mask]
    x_test = x_arr[test_mask]

    ax = axes[0]
    ax.plot(t_test, x_test, 'k-', lw=0.5)
    ax.set_ylabel('Displacement')
    ax.set_title('Test Period: Displacement + Latent Force g(t)')

    ax = axes[1]
    ax.plot(t_test, sg_j[test_mask], 'r-', lw=0.8, alpha=0.7, label='g (joint)')
    ax.plot(t_test, sg_f[test_mask], 'g-', lw=0.8, alpha=0.7, label='g (fixed)')
    ax.set_ylabel('Latent force g')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t_test, v_arr[test_mask], 'b-', lw=0.3, alpha=0.5, label='Velocity')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_latent_force_timeseries.png")
    plt.close()

    # Fig 4: g vs displacement scatter (test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sg, label in [(axes[0], sg_j, 'Joint'), (axes[1], sg_f, 'Fixed')]:
        ax.scatter(x_test, sg[test_mask], s=1, alpha=0.3)
        ax.set_xlabel('Displacement')
        ax.set_ylabel('Latent force g')
        ax.set_title(f'{label}: g vs x')
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_g_vs_displacement.png")
    plt.close()

    # ============================================================
    #  TABLES
    # ============================================================
    print_section("TABLES")

    # Metrics table
    rows = [
        {'model': 'physics_only', 'ACF1': float(acf_base_phys[1]),
         'DxR2_h10': float(dxr2_phys[9]),
         'mean_5_10': float(np.mean(dxr2_phys[4:10]))},
        {'model': 'closure_2t', 'ACF1': float(acf_cl[1]),
         'DxR2_h10': float(dxr2_cl[9]),
         'mean_5_10': float(np.mean(dxr2_cl[4:10])),
         'NIS': nis_cl},
        {'model': '3state_joint', 'ACF1': float(acf_joint[1]),
         'DxR2_h10': float(dxr2_joint[9]),
         'mean_5_10': float(np.mean(dxr2_joint[4:10])),
         'NIS': nis_joint, 'rho_g': rho_g_joint, 'q_g': q_g_joint,
         'tau_g': tau_g_joint,
         'b2': cl_joint['b2'], 'd2': cl_joint['d2'],
         'q_scale': cl_joint['q_scale']},
        {'model': '3state_fixed', 'ACF1': float(acf_fix[1]),
         'DxR2_h10': float(dxr2_fix[9]),
         'mean_5_10': float(np.mean(dxr2_fix[4:10])),
         'NIS': nis_fix, 'rho_g': rho_g_fix, 'q_g': q_g_fix,
         'tau_g': tau_g_fix},
    ]
    pd.DataFrame(rows).to_csv(OUT / "tables" / "metrics_table.csv", index=False)

    # DxR2 by horizon
    dxr2_rows = []
    for h in range(1, MAX_HORIZON + 1):
        dxr2_rows.append({
            'h': h,
            'physics_only': float(dxr2_phys[h-1]),
            'closure_2t': float(dxr2_cl[h-1]),
            '3state_joint': float(dxr2_joint[h-1]),
            '3state_fixed': float(dxr2_fix[h-1]),
        })
    pd.DataFrame(dxr2_rows).to_csv(OUT / "tables" / "dxr2_by_horizon.csv", index=False)

    # Innovation diagnostics
    diag_rows = []
    for model_name, acf_vals, n_pts in [
        ('closure_2t', acf_cl, int(valid_c.sum())),
        ('3state_joint', acf_joint, int(valid_j.sum())),
        ('3state_fixed', acf_fix, int(valid_f.sum())),
    ]:
        lb = ljung_box(acf_vals, n_pts)
        for r in lb:
            diag_rows.append({
                'model': model_name, 'lag': r['lag'],
                'Q': r['Q'], 'p': r['p'],
                'ACF_at_lag': float(acf_vals[r['lag']]),
            })
    pd.DataFrame(diag_rows).to_csv(
        OUT / "tables" / "innovation_diagnostics.csv", index=False)

    # ============================================================
    #  SUMMARY JSON
    # ============================================================
    runtime = time.time() - t0_global

    summary = {
        'seed': SEED,
        'baseline': {
            'b2': cl_avg['b2'], 'd2': cl_avg['d2'],
            'q_scale': cl_avg['q_scale'],
            'acf1': float(acf_cl[1]),
            'dxr2_10': float(dxr2_cl[9]),
            'mean_5_10': float(np.mean(dxr2_cl[4:10])),
            'nis': nis_cl,
        },
        'joint': {
            'b2': cl_joint['b2'], 'd2': cl_joint['d2'],
            'q_scale': cl_joint['q_scale'],
            'rho_g': rho_g_joint, 'q_g': q_g_joint,
            'tau_g_s': tau_g_joint,
            'nll': nll_joint, 'train_time': time_joint,
            'acf1': float(acf_joint[1]),
            'dxr2_10': float(dxr2_joint[9]),
            'mean_5_10': float(np.mean(dxr2_joint[4:10])),
            'nis': nis_joint,
            'delta_dxr2_10': float(dxr2_joint[9] - dxr2_cl[9]),
            'delta_acf1': float(acf_joint[1] - acf_cl[1]),
            'g_stats': {
                'test_mean': float(np.mean(sg_j[test_mask])),
                'test_std': float(np.std(sg_j[test_mask])),
                'test_min': float(np.min(sg_j[test_mask])),
                'test_max': float(np.max(sg_j[test_mask])),
            },
        },
        'fixed_closure': {
            'rho_g': rho_g_fix, 'q_g': q_g_fix,
            'tau_g_s': tau_g_fix,
            'nll': nll_fix, 'train_time': time_fix,
            'acf1': float(acf_fix[1]),
            'dxr2_10': float(dxr2_fix[9]),
            'mean_5_10': float(np.mean(dxr2_fix[4:10])),
            'nis': nis_fix,
            'delta_dxr2_10': float(dxr2_fix[9] - dxr2_cl[9]),
            'delta_acf1': float(acf_fix[1] - acf_cl[1]),
            'g_stats': {
                'test_mean': float(np.mean(sg_f[test_mask])),
                'test_std': float(np.std(sg_f[test_mask])),
                'test_min': float(np.min(sg_f[test_mask])),
                'test_max': float(np.max(sg_f[test_mask])),
            },
        },
        'selected': best_label,
        'runtime_s': round(runtime, 1),
    }

    with open(OUT / "summary_v9b.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # ============================================================
    #  FINAL PRINT
    # ============================================================
    print_section("RESULTS SUMMARY")
    print(f"{'Model':<25} {'ACF(1)':>8} {'DxR2@10':>9} {'mean(5-10)':>11} {'rho_g':>7} {'tau_g':>7}")
    print("-" * 70)
    print(f"{'Physics-only':<25} {acf_base_phys[1]:8.4f} {dxr2_phys[9]:9.4f} "
          f"{np.mean(dxr2_phys[4:10]):11.4f} {'--':>7} {'--':>7}")
    print(f"{'Closure (2t)':<25} {acf_cl[1]:8.4f} {dxr2_cl[9]:9.4f} "
          f"{np.mean(dxr2_cl[4:10]):11.4f} {'--':>7} {'--':>7}")
    print(f"{'3-state joint':<25} {acf_joint[1]:8.4f} {dxr2_joint[9]:9.4f} "
          f"{np.mean(dxr2_joint[4:10]):11.4f} {rho_g_joint:7.4f} {tau_g_joint:7.2f}")
    print(f"{'3-state fixed':<25} {acf_fix[1]:8.4f} {dxr2_fix[9]:9.4f} "
          f"{np.mean(dxr2_fix[4:10]):11.4f} {rho_g_fix:7.4f} {tau_g_fix:7.2f}")

    print(f"\nSelected: {best_label}")
    print(f"  Delta DxR2@10: {best_dxr2[9]-dxr2_cl[9]:+.4f}")
    print(f"  Delta ACF(1):  {best_acf[1]-acf_cl[1]:+.4f}")
    material = abs(best_dxr2[9] - dxr2_cl[9]) > 0.005
    print(f"  Material improvement: {'YES' if material else 'NO'}")

    n_files = sum(len(files) for _, _, files in os.walk(str(OUT)))
    print(f"\nRuntime: {runtime:.1f}s")
    print(f"Output files: {n_files}")
    print(f"COMPLETE")


if __name__ == '__main__':
    main()
