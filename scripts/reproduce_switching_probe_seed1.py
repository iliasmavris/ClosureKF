"""
Switching Probe: 2-regime Markov-switching memory extension to 2-state SSM.

Tests whether a latent bed-contact state (stuck vs moving) improves
forecast skill and innovation whiteness.

Option A: regime scales drag only
  C_t = b2*dv - d2*eta(s_t)*u*|v|,  eta(0)=1, eta(1)=eta1
  Parameters added: eta1 + transition probs p01, p10 (3 new params)

Inference: IMM (Interacting Multiple Models) filter.
Training: Nelder-Mead on IMM NLL.

Usage: python scripts/reproduce_switching_probe_seed1.py
"""

import os, sys, math, json, hashlib, time, platform, warnings
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

torch.set_num_threads(1)  # avoid thread-safety crash with numpy in Nelder-Mead

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_closure import CLOSURE_PARAM_NAMES

# ===== Paths =====
DATA_DIR = ROOT / "processed_data_10hz"
V2_CKPT  = ROOT / "final_lockbox_v2" / "checkpoints"
S1_CKPT  = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
             / "stage1_physics_only.pth")
V3_DIR   = ROOT / "final_lockbox_v3"

OUT = ROOT / "final_lockbox_v8_switching_probe"
for d in ['figures', 'tables']:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ===== Constants =====
DT = 0.1
SEED = 1
FORCE_CPU = True
MAX_HORIZON = 10
SEEDS_REF = [42, 43, 44]

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})


# ============================================================
#  HELPERS (copied from v4 Apack, non-destructive)
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
#  STANDARD KF FILTER (baseline)
# ============================================================

def kf_filter_2state(params, cl_params, t, x_obs, v, return_states=False):
    """Numpy KF filter. Returns innovations, S_values, optionally states."""
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
    """Standard DxR2(h) computation."""
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

    # Pass 1: KF filter to get post-update states
    _, _, states_x, states_u = kf_filter_2state(
        params, cl_params, t, x_obs, v, return_states=True)

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
                sx, su = _predict_step(sx, su, v_w, dv_w, dt_s)
            dx_pred_list.append(sx - x_obs[i])
            dx_obs_list.append(x_obs[i + h] - x_obs[i])
        dp = np.array(dx_pred_list)
        do = np.array(dx_obs_list)
        ss_res = np.sum((do - dp)**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h - 1] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ============================================================
#  IMM FILTER (2-regime switching)
# ============================================================

def imm_filter(params, cl_base, sw_params, t, x_obs, v,
               return_states=False, return_nll=False):
    """
    IMM (Interacting Multiple Models) filter with 2 regimes.

    Regime 0: nominal closure (d2_eff = d2)
    Regime 1: scaled drag   (d2_eff = d2 * eta1)

    Returns:
        innovations, S_values, regime_prob, [states_x, states_u, mu_posts], [nll]
    """
    N = len(x_obs)

    p01 = sw_params['p01']
    p10 = sw_params['p10']
    eta = [1.0, sw_params['eta1']]

    Pi = np.array([[1-p01, p01],
                   [p10, 1-p10]])

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_base.get('q_scale', 1.0)
    R = params['R']

    b2_v = cl_base.get('b2', 0.0)
    d2_base = cl_base.get('d2', 0.0)

    # Per-model state and covariance
    s = [np.array([x_obs[0], 0.0]) for _ in range(2)]
    P = [np.diag([params['P0_xx'], params['P0_uu']]) for _ in range(2)]
    mu = np.array([0.5, 0.5])

    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    regime_prob = np.zeros((N, 2))
    regime_prob[0] = mu
    nll_sum = 0.0
    nll_count = 0

    states_x = np.zeros(N) if return_states else None
    states_u = np.zeros(N) if return_states else None
    mu_posts = np.zeros((N, 2)) if return_states else None

    if return_states:
        states_x[0] = x_obs[0]
        mu_posts[0] = mu.copy()

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        v_w = v[k-1]

        # --- Step 1: Mixing ---
        c_bar = Pi.T @ mu  # predicted mode probs [2]

        mu_mix = np.zeros((2, 2))
        for j in range(2):
            denom = c_bar[j] + 1e-30
            for i in range(2):
                mu_mix[i, j] = Pi[i, j] * mu[i] / denom

        s_mixed = [np.zeros(2) for _ in range(2)]
        P_mixed = [np.zeros((2, 2)) for _ in range(2)]
        for j in range(2):
            for i in range(2):
                s_mixed[j] += mu_mix[i, j] * s[i]
            for i in range(2):
                diff = s[i] - s_mixed[j]
                P_mixed[j] += mu_mix[i, j] * (P[i] + np.outer(diff, diff))

        # --- Step 2: Model-conditional predict + update ---
        log_likelihoods = np.zeros(2)
        x_preds_j = np.zeros(2)

        for j in range(2):
            u_st = s_mixed[j][1]
            x_st = s_mixed[j][0]

            physics_drift = rho_u * u_st - kap * x_st * dt + c_val * g * dt
            d2_eff = d2_base * eta[j]
            cl = b2_v * dv_w - d2_eff * u_st * abs(v_w)
            cl_dt = cl * dt

            x_p = x_st + u_st * dt
            u_p = physics_drift + cl_dt
            s_pred_j = np.array([x_p, u_p])
            x_preds_j[j] = x_p

            F_mat = np.array([[1, dt], [-kap*dt, rho_u]])
            Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
            P_pred_j = F_mat @ P_mixed[j] @ F_mat.T + Q

            innov_j = x_obs[k] - s_pred_j[0]
            S_j = P_pred_j[0, 0] + R

            # Log-likelihood (numerically stable)
            log_likelihoods[j] = (-0.5 * math.log(2 * math.pi * max(S_j, 1e-30))
                                  - 0.5 * innov_j**2 / max(S_j, 1e-30))

            # KF update
            K_j = P_pred_j[:, 0] / S_j
            s[j] = s_pred_j + K_j * innov_j
            IKH = np.eye(2) - np.outer(K_j, np.array([1.0, 0.0]))
            P[j] = IKH @ P_pred_j @ IKH.T + R * np.outer(K_j, K_j)

        # --- Step 3: Mode probability update (log-sum-exp) ---
        log_joint = np.log(c_bar + 1e-30) + log_likelihoods
        max_lj = np.max(log_joint)
        log_evidence = max_lj + math.log(np.sum(np.exp(log_joint - max_lj)))

        mu = np.exp(log_joint - log_evidence)
        mu = np.clip(mu, 1e-10, 1.0)
        mu /= mu.sum()

        nll_sum += -log_evidence
        nll_count += 1

        # Combined innovation (for ACF diagnostics)
        x_pred_combined = c_bar[0] * x_preds_j[0] + c_bar[1] * x_preds_j[1]
        innov_combined = x_obs[k] - x_pred_combined

        # Combined S (approximate from combined covariance)
        s_combined = mu[0] * s[0] + mu[1] * s[1]
        P_comb = np.zeros((2, 2))
        for j in range(2):
            diff = s[j] - s_combined
            P_comb += mu[j] * (P[j] + np.outer(diff, diff))
        S_combined = P_comb[0, 0] + R

        innovations[k] = innov_combined
        S_values[k] = S_combined
        regime_prob[k] = mu

        if return_states:
            states_x[k] = s_combined[0]
            states_u[k] = s_combined[1]
            mu_posts[k] = mu.copy()

    out = [innovations, S_values, regime_prob]
    if return_states:
        out += [states_x, states_u, mu_posts]
    if return_nll:
        out += [nll_sum / max(nll_count, 1)]
    return tuple(out)


def compute_dxr2_hstep_switching(params, cl_base, sw_params, t, x_obs, v,
                                  max_h=10, eval_start=1):
    """DxR2(h) for the switching model using regime-averaged dynamics."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    b2_v = cl_base.get('b2', 0.0)
    d2_base = cl_base.get('d2', 0.0)
    eta1 = sw_params['eta1']
    p01 = sw_params['p01']
    p10 = sw_params['p10']
    Pi = np.array([[1-p01, p01], [p10, 1-p10]])

    # Pass 1: IMM filter to get post-update combined states + regime probs
    result = imm_filter(params, cl_base, sw_params, t, x_obs, v,
                        return_states=True)
    _, _, _, states_x, states_u, mu_posts = result

    def _predict_step_averaged(sx, su, v_w, dv_w, dt_k, mu_k):
        rho = math.exp(-alpha * dt_k)
        g = max(v_w**2 - vc**2, 0.0)
        d2_eff = d2_base * (mu_k[0] * 1.0 + mu_k[1] * eta1)
        cl = b2_v * dv_w - d2_eff * su * abs(v_w)
        x_new = sx + su * dt_k
        u_new = rho*su - kap*sx*dt_k + c_val*g*dt_k + cl*dt_k
        return x_new, u_new

    # Pass 2: h-step predictions with evolving regime probs
    r2_arr = np.zeros(max_h)
    for h in range(1, max_h + 1):
        dx_pred_list = []; dx_obs_list = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            mu_k = mu_posts[i].copy()
            for step in range(h):
                k_s = i + step + 1
                if k_s >= N: break
                dt_s = t[k_s] - t[k_s - 1]
                if dt_s <= 0: dt_s = 0.1
                v_w = v[k_s - 1]
                dv_w = v[k_s - 1] - v[k_s - 2] if k_s >= 2 else 0.0
                mu_k = Pi.T @ mu_k  # evolve regime probs
                sx, su = _predict_step_averaged(sx, su, v_w, dv_w, dt_s, mu_k)
            dx_pred_list.append(sx - x_obs[i])
            dx_obs_list.append(x_obs[i + h] - x_obs[i])
        dp = np.array(dx_pred_list)
        do = np.array(dx_obs_list)
        ss_res = np.sum((do - dp)**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h - 1] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ============================================================
#  TRAINING
# ============================================================

def train_switching(train_t, train_x, train_v, s1_params,
                    init_cl, seed=1, maxiter=500):
    """Train switching model parameters using Nelder-Mead on IMM NLL."""
    np.random.seed(seed)

    def _softplus(x):
        return np.log1p(np.exp(x)) if x < 20 else float(x)

    def _inv_softplus(y):
        y = max(y, 1e-4)
        return float(np.log(np.exp(y) - 1)) if y < 20 else float(y)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _logit(p):
        p = np.clip(p, 0.01, 0.99)
        return float(np.log(p / (1.0 - p)))

    def _unpack(x_vec):
        b2 = float(x_vec[0])
        d2 = float(_softplus(x_vec[1]))
        eta1 = float(_softplus(x_vec[2]))
        p01 = float(_sigmoid(x_vec[3]))
        p10 = float(_sigmoid(x_vec[4]))
        q_scale = float(np.exp(x_vec[5]))

        cl = zero_closure()
        cl['b2'] = b2
        cl['d2'] = d2
        cl['q_scale'] = q_scale

        sw = {'eta1': eta1, 'p01': p01, 'p10': p10}
        return cl, sw

    def _objective(x_vec):
        cl, sw = _unpack(x_vec)
        try:
            result = imm_filter(s1_params, cl, sw, train_t, train_x, train_v,
                               return_nll=True)
            nll = result[-1]
        except Exception:
            return 1e30

        # Regularization: discourage extreme transition probs
        p01 = sw['p01']; p10 = sw['p10']
        reg = -0.01 * (math.log(max(p01*(1-p01), 1e-10)) +
                       math.log(max(p10*(1-p10), 1e-10)))
        return float(nll + reg)

    # Initial point from existing closure
    x0 = np.array([
        init_cl.get('b2', 5.0),
        _inv_softplus(init_cl.get('d2', 10.0)),
        _inv_softplus(1.5),        # eta1 init: 1.5 (50% more drag in regime 1)
        _logit(0.05),              # p01 init: low switching rate
        _logit(0.10),              # p10 init: higher return rate
        math.log(max(init_cl.get('q_scale', 1.0), 1e-4)),
    ], dtype=np.float64)

    print(f"    Initial: b2={init_cl.get('b2',5):.3f} d2={init_cl.get('d2',10):.3f} "
          f"eta1=1.5 p01=0.05 p10=0.10 q_sc={init_cl.get('q_scale',1):.3f}")
    print(f"    Training (seed={seed}, maxiter={maxiter})...")
    t_start = time.time()

    result = sp_minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-4,
                                  'fatol': 1e-7, 'adaptive': True})

    cl_opt, sw_opt = _unpack(result.x)
    elapsed = time.time() - t_start

    print(f"    Converged: nll={result.fun:.6f}, nfev={result.nfev}, "
          f"time={elapsed:.1f}s")
    print(f"    b2={cl_opt['b2']:.4f}, d2={cl_opt['d2']:.4f}, "
          f"eta1={sw_opt['eta1']:.4f}")
    print(f"    p01={sw_opt['p01']:.6f}, p10={sw_opt['p10']:.6f}, "
          f"q_scale={cl_opt['q_scale']:.4f}")

    return cl_opt, sw_opt, result.fun, elapsed


def train_switching_constrained(train_t, train_x, train_v, s1_params,
                                 fixed_cl, seed=1, maxiter=300):
    """Train ONLY switching params (eta1, p01, p10) with b2/d2/q_scale fixed."""
    np.random.seed(seed)

    def _softplus(x):
        return np.log1p(np.exp(x)) if x < 20 else float(x)

    def _inv_softplus(y):
        y = max(y, 1e-4)
        return float(np.log(np.exp(y) - 1)) if y < 20 else float(y)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _logit(p):
        p = np.clip(p, 0.01, 0.99)
        return float(np.log(p / (1.0 - p)))

    def _unpack(x_vec):
        eta1 = float(max(_softplus(x_vec[0]), 0.05))  # floor at 0.05
        p01 = float(_sigmoid(x_vec[1]))
        p10 = float(_sigmoid(x_vec[2]))
        sw = {'eta1': eta1, 'p01': p01, 'p10': p10}
        return sw

    def _objective(x_vec):
        sw = _unpack(x_vec)
        try:
            result = imm_filter(s1_params, fixed_cl, sw,
                               train_t, train_x, train_v, return_nll=True)
            nll = result[-1]
        except Exception:
            return 1e30

        # Regularization
        p01 = sw['p01']; p10 = sw['p10']
        reg = -0.01 * (math.log(max(p01*(1-p01), 1e-10)) +
                       math.log(max(p10*(1-p10), 1e-10)))
        # Discourage eta1 too far from 1
        reg += 0.001 * (sw['eta1'] - 1.0)**2
        return float(nll + reg)

    x0 = np.array([
        _inv_softplus(1.5),   # eta1 init
        _logit(0.05),         # p01
        _logit(0.10),         # p10
    ], dtype=np.float64)

    print(f"    Constrained training: b2={fixed_cl['b2']:.3f} d2={fixed_cl['d2']:.3f} "
          f"q_sc={fixed_cl['q_scale']:.3f} FIXED")
    print(f"    Optimizing only: eta1, p01, p10 (seed={seed}, maxiter={maxiter})")
    t_start = time.time()

    result = sp_minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-4,
                                  'fatol': 1e-7, 'adaptive': True})

    sw_opt = _unpack(result.x)
    elapsed = time.time() - t_start

    print(f"    Converged: nll={result.fun:.6f}, nfev={result.nfev}, "
          f"time={elapsed:.1f}s")
    print(f"    eta1={sw_opt['eta1']:.4f}, p01={sw_opt['p01']:.6f}, "
          f"p10={sw_opt['p10']:.6f}")

    return fixed_cl.copy(), sw_opt, result.fun, elapsed


# ============================================================
#  MAIN
# ============================================================

def main():
    t0_global = time.time()
    np.random.seed(SEED)
    device = torch.device('cpu')

    print_section("Switching Probe: 2-Regime Markov-Switching Memory Extension")
    print(f"Seed: {SEED}")
    print(f"Output -> {OUT}")

    # ---- Load parameters ----
    s1_params = load_s1_params(device)
    print(f"S1 physics: alpha={s1_params['alpha']:.4f}, c={s1_params['c']:.4f}, "
          f"kappa={s1_params['kappa']:.4f}")

    # Load closure reference (average of 3 seeds)
    cl_avg = zero_closure()
    for key in ['b2', 'd2', 'q_scale']:
        vals = [load_closure_params(s)[key] for s in SEEDS_REF]
        cl_avg[key] = float(np.mean(vals))
    print(f"Closure ref: b2={cl_avg['b2']:.4f}, d2={cl_avg['d2']:.4f}, "
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
    print_section("BASELINE: Standard 2-state KF + Closure")

    # Physics-only
    e_base, S_base = kf_filter_2state(
        s1_params, zero_closure(), t_arr, x_arr, v_arr)
    e_bm = e_base[test_mask]; valid = ~np.isnan(e_bm)
    acf_base_physics = compute_acf(e_bm[valid])

    # Closure
    e_cl, S_cl = kf_filter_2state(s1_params, cl_avg, t_arr, x_arr, v_arr)
    e_cm = e_cl[test_mask]; S_cm = S_cl[test_mask]; valid_c = ~np.isnan(e_cm)
    acf_base_cl = compute_acf(e_cm[valid_c])
    nis_cl = float(np.mean(e_cm[valid_c]**2 / S_cm[valid_c]))

    dxr2_base = compute_dxr2_hstep(
        s1_params, zero_closure(), t_arr, x_arr, v_arr,
        MAX_HORIZON, eval_start=eval_start)
    dxr2_cl = compute_dxr2_hstep(
        s1_params, cl_avg, t_arr, x_arr, v_arr,
        MAX_HORIZON, eval_start=eval_start)

    lb_base_cl = ljung_box(acf_base_cl, int(valid_c.sum()))

    print(f"Physics-only: ACF(1)={acf_base_physics[1]:.4f}, "
          f"DxR2@10={dxr2_base[9]:.4f}")
    print(f"Closure (2t): ACF(1)={acf_base_cl[1]:.4f}, "
          f"DxR2@10={dxr2_cl[9]:.4f}, NIS={nis_cl:.4f}")
    for r in lb_base_cl:
        print(f"  LB lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.4f}")

    # ============================================================
    #  VARIANT A: UNCONSTRAINED (all 6 params)
    # ============================================================
    print_section("VARIANT A: Unconstrained (b2, d2, eta1, p01, p10, q_scale)")

    cl_sw_A, sw_A, nll_A, time_A = train_switching(
        train_t, train_x, train_v, s1_params, cl_avg,
        seed=SEED, maxiter=500)

    # ============================================================
    #  VARIANT B: CONSTRAINED (only eta1, p01, p10; b2/d2/q_scale fixed)
    # ============================================================
    print_section("VARIANT B: Constrained (eta1, p01, p10 only)")

    cl_sw_B, sw_B, nll_B, time_B = train_switching_constrained(
        train_t, train_x, train_v, s1_params, cl_avg,
        seed=SEED, maxiter=300)

    # ============================================================
    #  EVALUATE BOTH VARIANTS
    # ============================================================
    def evaluate_switching(cl_sw, sw_params, label):
        """Evaluate a switching variant, return metrics dict."""
        print(f"\n  --- {label} ---")
        result = imm_filter(
            s1_params, cl_sw, sw_params, t_arr, x_arr, v_arr,
            return_states=True)
        e_sw, S_sw, rp, sx, su, mp = result

        e_test = e_sw[test_mask]; S_test = S_sw[test_mask]
        vld = ~np.isnan(e_test)
        acf_vals = compute_acf(e_test[vld])
        nis = float(np.mean(e_test[vld]**2 / S_test[vld]))
        lb = ljung_box(acf_vals, int(vld.sum()))

        print(f"  ACF(1)={acf_vals[1]:.4f}, NIS={nis:.4f}")
        for r in lb:
            print(f"    LB lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.4f}")

        print(f"  Computing DxR2(h)...")
        dxr2 = compute_dxr2_hstep_switching(
            s1_params, cl_sw, sw_params, t_arr, x_arr, v_arr,
            MAX_HORIZON, eval_start=eval_start)
        print(f"  DxR2@10={dxr2[9]:.4f}")

        rp_test = rp[test_mask]
        mean_p1 = float(np.mean(rp_test[:, 1]))
        std_p1 = float(np.std(rp_test[:, 1]))
        frac_r1 = float(np.mean(rp_test[:, 1] > 0.5))

        return {
            'acf': acf_vals, 'nis': nis, 'lb': lb, 'dxr2': dxr2,
            'regime_prob': rp, 'mean_p1': mean_p1, 'std_p1': std_p1,
            'frac_r1': frac_r1, 'e_sw': e_sw, 'S_sw': S_sw,
            'mu_posts': mp, 'states_x': sx, 'states_u': su,
        }

    print_section("EVALUATION")
    eval_A = evaluate_switching(cl_sw_A, sw_A, "Variant A (unconstrained)")
    eval_B = evaluate_switching(cl_sw_B, sw_B, "Variant B (constrained)")

    # Pick the better variant for main reporting
    # Prefer B if its DxR2@10 >= A's (constrained is more interpretable)
    if eval_B['dxr2'][9] >= eval_A['dxr2'][9] - 0.005:
        cl_sw, sw_params = cl_sw_B, sw_B
        nll_sw, train_time = nll_B, time_B
        eval_best = eval_B
        best_label = "B (constrained)"
    else:
        cl_sw, sw_params = cl_sw_A, sw_A
        nll_sw, train_time = nll_A, time_A
        eval_best = eval_A
        best_label = "A (unconstrained)"
    print(f"\n  Selected variant: {best_label}")

    # Unpack best variant results
    acf_sw = eval_best['acf']
    nis_sw = eval_best['nis']
    lb_sw = eval_best['lb']
    dxr2_sw = eval_best['dxr2']
    regime_prob = eval_best['regime_prob']
    mean_p1 = eval_best['mean_p1']
    std_p1 = eval_best['std_p1']
    frac_r1_dominant = eval_best['frac_r1']

    # Also save variant comparison
    variant_rows = []
    for vlabel, vcl, vsw, vnll, vtime, veval in [
        ('A_unconstrained', cl_sw_A, sw_A, nll_A, time_A, eval_A),
        ('B_constrained', cl_sw_B, sw_B, nll_B, time_B, eval_B),
    ]:
        variant_rows.append({
            'variant': vlabel,
            'b2': vcl['b2'], 'd2': vcl['d2'], 'q_scale': vcl['q_scale'],
            'eta1': vsw['eta1'], 'p01': vsw['p01'], 'p10': vsw['p10'],
            'train_nll': vnll, 'train_time_s': vtime,
            'ACF1': float(veval['acf'][1]),
            'DxR2_h10': float(veval['dxr2'][9]),
            'mean_DxR2_5_10': float(np.mean(veval['dxr2'][4:10])),
            'NIS': veval['nis'],
            'mean_p1': veval['mean_p1'], 'std_p1': veval['std_p1'],
        })
    pd.DataFrame(variant_rows).to_csv(
        OUT / "tables" / "variant_comparison.csv", index=False)

    # ============================================================
    #  SANITY CHECKS
    # ============================================================
    print_section("SANITY CHECKS")

    print(f"Selected variant: {best_label}")
    print(f"P(regime 1) on test: mean={mean_p1:.4f}, std={std_p1:.4f}")
    print(f"Fraction time in regime 1: {frac_r1_dominant:.4f}")

    degenerate = (mean_p1 < 0.01 or mean_p1 > 0.99)
    if degenerate:
        print("WARNING: Regime is degenerate (collapsed to single regime)")
    else:
        print("Regime usage: non-degenerate")

    print(f"eta1={sw_params['eta1']:.4f} (1.0 = identical to regime 0)")
    if abs(sw_params['eta1'] - 1.0) < 0.01:
        print("WARNING: eta1 ~ 1.0, no effective regime difference")
    elif sw_params['eta1'] > 10.0:
        print("WARNING: eta1 > 10, potentially explosive")
    else:
        print("eta1 is in reasonable range")

    print(f"p01={sw_params['p01']:.6f}, p10={sw_params['p10']:.6f}")
    if sw_params['p01'] < 0.001 and sw_params['p10'] < 0.001:
        print("WARNING: Both transition probs near zero (no switching)")
    elif sw_params['p01'] > 0.5 and sw_params['p10'] > 0.5:
        print("WARNING: Both transition probs high (rapid switching)")
    else:
        dur_0 = 1.0 / max(sw_params['p01'], 1e-6) * DT
        dur_1 = 1.0 / max(sw_params['p10'], 1e-6) * DT
        print(f"Expected duration: regime 0 = {dur_0:.1f}s, "
              f"regime 1 = {dur_1:.1f}s")

    # ============================================================
    #  RESULTS TABLE
    # ============================================================
    print_section("RESULTS TABLE")

    metrics_rows = []
    for h in range(1, MAX_HORIZON + 1):
        metrics_rows.append({
            'horizon': h,
            'base_dxr2': float(dxr2_base[h-1]),
            'closure_dxr2': float(dxr2_cl[h-1]),
            'switching_dxr2': float(dxr2_sw[h-1]),
            'gain_sw_vs_cl': float(dxr2_sw[h-1] - dxr2_cl[h-1]),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT / "tables" / "metrics_table.csv", index=False)

    print(f"{'h':>3}  {'Base':>8}  {'Closure':>8}  {'Switch':>8}  {'Sw-Cl':>8}")
    for _, row in metrics_df.iterrows():
        print(f"{int(row['horizon']):3d}  {row['base_dxr2']:8.4f}  "
              f"{row['closure_dxr2']:8.4f}  {row['switching_dxr2']:8.4f}  "
              f"{row['gain_sw_vs_cl']:+8.4f}")

    # Summary
    summary_df = pd.DataFrame({
        'model': ['Physics-only', 'Closure (2t)', 'Switching (2-regime)'],
        'ACF1': [float(acf_base_physics[1]), float(acf_base_cl[1]),
                 float(acf_sw[1])],
        'DxR2_h10': [float(dxr2_base[9]), float(dxr2_cl[9]),
                     float(dxr2_sw[9])],
        'mean_DxR2_5_10': [
            float(np.mean(dxr2_base[4:10])),
            float(np.mean(dxr2_cl[4:10])),
            float(np.mean(dxr2_sw[4:10])),
        ],
    })
    summary_df.to_csv(OUT / "tables" / "summary_table.csv", index=False)

    print(f"\nSummary:")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']:25s}: ACF1={row['ACF1']:.4f}, "
              f"DxR2@10={row['DxR2_h10']:.4f}, "
              f"mean(5-10)={row['mean_DxR2_5_10']:.4f}")

    # Innovation diagnostics
    e_sw_test = eval_best['e_sw'][test_mask]
    valid_sw = ~np.isnan(e_sw_test)

    diag_rows = []
    for name, acf_vals, n_valid in [
        ('Physics-only', acf_base_physics, int(valid.sum())),
        ('Closure (2t)', acf_base_cl, int(valid_c.sum())),
        ('Switching', acf_sw, int(valid_sw.sum())),
    ]:
        lb = ljung_box(acf_vals, n_valid)
        row = {'model': name, 'ACF1': float(acf_vals[1]),
               'ACF2': float(acf_vals[2]), 'ACF5': float(acf_vals[5]),
               'ACF10': float(acf_vals[10])}
        for r in lb:
            row[f'LB_p_{r["lag"]}'] = r['p']
            row[f'LB_Q_{r["lag"]}'] = r['Q']
        diag_rows.append(row)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT / "tables" / "innovation_diagnostics.csv", index=False)

    # ============================================================
    #  FIGURES
    # ============================================================
    print_section("FIGURES")

    # Figure 1: DxR2(h) comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = np.arange(1, MAX_HORIZON + 1)
    ax.plot(horizons, dxr2_base, 's-', color='#d62728',
            label='Physics-only', markersize=6)
    ax.plot(horizons, dxr2_cl, 'o-', color='#1f77b4',
            label='Closure (2t)', markersize=6)
    ax.plot(horizons, dxr2_sw, 'D-', color='#2ca02c',
            label='Switching (2-regime)', markersize=6)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xlabel('Forecast horizon h (steps)')
    ax.set_ylabel('DxR2(h)')
    ax.set_title('Displacement-Increment R2: Baseline vs Switching')
    ax.legend(fontsize=9)
    ax.set_xticks(horizons)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_dxr2_comparison.png",
               bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_dxr2_comparison.png")

    # Figure 2: Innovation ACF comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    max_lag_plot = 30
    lags_plot = np.arange(max_lag_plot + 1)
    n_test = int(valid_c.sum())
    sig_band = 1.96 / math.sqrt(n_test)
    width = 0.35
    ax.bar(lags_plot - width/2, acf_base_cl[:max_lag_plot+1], width=width,
           alpha=0.6, color='#1f77b4', label='Closure (2t)')
    ax.bar(lags_plot + width/2, acf_sw[:max_lag_plot+1], width=width,
           alpha=0.6, color='#2ca02c', label='Switching')
    ax.axhline(sig_band, color='gray', ls='--', lw=0.8,
               label='95% significance')
    ax.axhline(-sig_band, color='gray', ls='--', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Innovation Autocorrelation: Closure vs Switching')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_acf_comparison.png",
               bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_acf_comparison.png")

    # Figure 3: Regime probability time series
    test_times = t_arr[test_mask]
    regime_test_p1 = regime_prob[test_mask, 1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(test_times, x_arr[test_mask], color='k', lw=0.5)
    ax.set_ylabel('Displacement (m)')
    ax.set_title('Regime Interpretability: Regime Probability vs Movement')

    ax = axes[1]
    dx_test = np.abs(np.diff(x_arr[test_mask],
                             prepend=x_arr[test_mask][0]))
    ax.plot(test_times, dx_test, color='gray', lw=0.5, alpha=0.5,
            label='|dx|')
    ax.plot(test_times, np.abs(v_arr[test_mask]), color='steelblue',
            lw=0.5, alpha=0.7, label='|v_water|')
    ax.set_ylabel('Magnitude')
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[2]
    ax.fill_between(test_times, 0, regime_test_p1, alpha=0.5,
                    color='#2ca02c', label='P(regime 1)')
    ax.plot(test_times, regime_test_p1, color='#2ca02c', lw=0.5)
    ax.set_ylabel('P(regime 1)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_regime_probability.png",
               bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_regime_probability.png")

    # Figure 4: Regime probability vs |dx| scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dx_test, regime_test_p1, alpha=0.3, s=5, color='#2ca02c')
    ax.set_xlabel('|dx| (movement magnitude)')
    ax.set_ylabel('P(regime 1)')
    ax.set_title('Regime 1 Probability vs Movement Magnitude')
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_regime_vs_movement.png",
               bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_regime_vs_movement.png")

    # ============================================================
    #  SUMMARY JSON
    # ============================================================
    delta_acf = float(acf_sw[1] - acf_base_cl[1])
    delta_dxr2 = float(dxr2_sw[9] - dxr2_cl[9])

    improvement = delta_dxr2 > 0.01
    whitening = delta_acf < -0.01
    if improvement and whitening:
        verdict = "PROMISING: improves both skill and whiteness"
    elif improvement:
        verdict = "MIXED: skill improvement but no whitening"
    elif whitening:
        verdict = "MIXED: whitening improvement but no skill gain"
    elif abs(delta_dxr2) < 0.01 and abs(delta_acf) < 0.01:
        verdict = "NEUTRAL: no material change"
    else:
        verdict = "NEGATIVE: switching model degrades performance"

    summary_json = {
        'probe_type': '2-regime Markov-switching (Option A: drag scaling)',
        'seed': SEED,
        'selected_variant': best_label,
        'runtime_s': time.time() - t0_global,
        'training_time_s': train_time,
        'training_nll': nll_sw,
        'variant_A': {
            'b2': cl_sw_A['b2'], 'd2': cl_sw_A['d2'],
            'q_scale': cl_sw_A['q_scale'],
            'eta1': sw_A['eta1'], 'p01': sw_A['p01'], 'p10': sw_A['p10'],
            'ACF1': float(eval_A['acf'][1]),
            'DxR2_h10': float(eval_A['dxr2'][9]),
            'note': 'unconstrained: all 6 params optimized',
        },
        'variant_B': {
            'b2': cl_sw_B['b2'], 'd2': cl_sw_B['d2'],
            'q_scale': cl_sw_B['q_scale'],
            'eta1': sw_B['eta1'], 'p01': sw_B['p01'], 'p10': sw_B['p10'],
            'ACF1': float(eval_B['acf'][1]),
            'DxR2_h10': float(eval_B['dxr2'][9]),
            'note': 'constrained: only eta1/p01/p10 optimized, b2/d2/q_scale fixed',
        },
        'parameters': {
            'physics_frozen': {
                'alpha': s1_params['alpha'],
                'kappa': s1_params['kappa'],
                'c': s1_params['c'],
                'vc': s1_params['vc'],
            },
            'closure_learned': {
                'b2': cl_sw['b2'],
                'd2': cl_sw['d2'],
                'q_scale': cl_sw['q_scale'],
            },
            'switching_learned': sw_params,
            'total_params': 6,
            'new_params_vs_closure': 3,
        },
        'baseline_closure_ref': {
            'b2': cl_avg['b2'],
            'd2': cl_avg['d2'],
            'q_scale': cl_avg['q_scale'],
        },
        'metrics': {
            'physics_only': {
                'ACF1': float(acf_base_physics[1]),
                'DxR2_h10': float(dxr2_base[9]),
                'mean_DxR2_5_10': float(np.mean(dxr2_base[4:10])),
            },
            'closure_2t': {
                'ACF1': float(acf_base_cl[1]),
                'DxR2_h10': float(dxr2_cl[9]),
                'mean_DxR2_5_10': float(np.mean(dxr2_cl[4:10])),
                'NIS': nis_cl,
            },
            'switching': {
                'ACF1': float(acf_sw[1]),
                'DxR2_h10': float(dxr2_sw[9]),
                'mean_DxR2_5_10': float(np.mean(dxr2_sw[4:10])),
                'NIS': nis_sw,
            },
        },
        'deltas': {
            'ACF1_sw_minus_cl': delta_acf,
            'DxR2_h10_sw_minus_cl': delta_dxr2,
        },
        'verdict': verdict,
        'dxr2_by_horizon': {
            'base': {f'h{h}': float(dxr2_base[h-1]) for h in range(1, 11)},
            'closure': {f'h{h}': float(dxr2_cl[h-1]) for h in range(1, 11)},
            'switching': {f'h{h}': float(dxr2_sw[h-1]) for h in range(1, 11)},
        },
        'innovation_diagnostics': {
            'closure': {
                'ACF': [float(acf_base_cl[i]) for i in range(11)],
                'ljung_box': lb_base_cl,
            },
            'switching': {
                'ACF': [float(acf_sw[i]) for i in range(11)],
                'ljung_box': lb_sw,
            },
        },
        'regime_analysis': {
            'mean_p_regime1': mean_p1,
            'std_p_regime1': std_p1,
            'frac_time_regime1': frac_r1_dominant,
            'degenerate': degenerate,
        },
        'sanity_checks': {
            'eta1_reasonable': 0.01 < sw_params['eta1'] < 10.0,
            'p01_not_extreme': 0.001 < sw_params['p01'] < 0.999,
            'p10_not_extreme': 0.001 < sw_params['p10'] < 0.999,
            'regime_non_degenerate': not degenerate,
        },
    }

    with open(OUT / "summary_switching_probe.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved summary_switching_probe.json")

    # ============================================================
    #  README_probe.md
    # ============================================================
    readme_lines = []
    readme_lines.append("# Switching Probe: 2-Regime Markov-Switching Memory Extension\n")
    readme_lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    readme_lines.append(f"**Seed:** {SEED}")
    readme_lines.append(f"**Runtime:** {time.time() - t0_global:.0f}s "
                        f"(training: {train_time:.0f}s)\n")

    readme_lines.append("## Model\n")
    readme_lines.append("**Option A: regime scales drag only**")
    readme_lines.append("```")
    readme_lines.append("C_t = b2*dv - d2*eta(s_t)*u*|v|")
    readme_lines.append("  eta(0) = 1.0  (nominal)")
    readme_lines.append(f"  eta(1) = {sw_params['eta1']:.4f}  (regime 1)")
    readme_lines.append("```")
    readme_lines.append(f"- Transition: p01={sw_params['p01']:.6f}, "
                        f"p10={sw_params['p10']:.6f}")
    dur_0 = 1.0 / max(sw_params['p01'], 1e-6) * DT
    dur_1 = 1.0 / max(sw_params['p10'], 1e-6) * DT
    readme_lines.append(f"- Expected regime durations: regime 0 = {dur_0:.1f}s, "
                        f"regime 1 = {dur_1:.1f}s\n")

    readme_lines.append("## Learned Parameters\n")
    readme_lines.append("| Parameter | Value |")
    readme_lines.append("|-----------|-------|")
    readme_lines.append(f"| b2 | {cl_sw['b2']:.4f} |")
    readme_lines.append(f"| d2 | {cl_sw['d2']:.4f} |")
    readme_lines.append(f"| eta1 | {sw_params['eta1']:.4f} |")
    readme_lines.append(f"| p01 | {sw_params['p01']:.6f} |")
    readme_lines.append(f"| p10 | {sw_params['p10']:.6f} |")
    readme_lines.append(f"| q_scale | {cl_sw['q_scale']:.4f} |")
    readme_lines.append("")

    readme_lines.append("## Results\n")
    readme_lines.append("| Model | ACF(1) | DxR2@10 | mean(5-10) |")
    readme_lines.append("|-------|--------|---------|------------|")
    readme_lines.append(f"| Physics-only | {acf_base_physics[1]:.4f} | "
                        f"{dxr2_base[9]:.4f} | {np.mean(dxr2_base[4:10]):.4f} |")
    readme_lines.append(f"| Closure (2t) | {acf_base_cl[1]:.4f} | "
                        f"{dxr2_cl[9]:.4f} | {np.mean(dxr2_cl[4:10]):.4f} |")
    readme_lines.append(f"| Switching | {acf_sw[1]:.4f} | "
                        f"{dxr2_sw[9]:.4f} | {np.mean(dxr2_sw[4:10]):.4f} |")
    readme_lines.append("")

    readme_lines.append(f"**Delta vs Closure:** ACF(1) change = {delta_acf:+.4f}, "
                        f"DxR2@10 change = {delta_dxr2:+.4f}\n")

    readme_lines.append("## Verdict\n")
    readme_lines.append(f"**{verdict}**\n")

    readme_lines.append("## Sanity Checks\n")
    for check_name, check_val in summary_json['sanity_checks'].items():
        status = "PASS" if check_val else "WARN"
        readme_lines.append(f"- [{status}] {check_name}")
    readme_lines.append("")

    readme_lines.append("## Regime Interpretation\n")
    if degenerate:
        readme_lines.append("Regime has collapsed to a single mode. The switching "
                            "mechanism is not being utilized.")
    else:
        if sw_params['eta1'] > 1.0:
            readme_lines.append(
                f"Regime 1 has eta1={sw_params['eta1']:.2f} (enhanced drag). "
                f"This could correspond to a 'stuck/contact' state with "
                f"increased friction.")
        elif sw_params['eta1'] < 1.0:
            readme_lines.append(
                f"Regime 1 has eta1={sw_params['eta1']:.2f} (reduced drag). "
                f"This could correspond to a 'free/moving' state with "
                f"decreased friction.")
        readme_lines.append(
            f"\nRegime 1 is active {frac_r1_dominant:.1%} of the test period.")
        readme_lines.append(
            f"Mean P(regime 1) = {mean_p1:.3f} +/- {std_p1:.3f}")
    readme_lines.append("")

    readme_lines.append("## Output Files\n")
    for f_name in sorted(OUT.rglob('*')):
        if f_name.is_file():
            readme_lines.append(f"- `{f_name.relative_to(OUT)}`")

    with open(OUT / "README_probe.md", 'w') as f:
        f.write('\n'.join(readme_lines))
    print("Saved README_probe.md")

    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    elapsed = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f"SWITCHING PROBE COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"Output: {OUT}")
    n_files = sum(1 for _ in OUT.rglob('*') if _.is_file())
    print(f"Total files: {n_files}")

    print(f"\n{'='*40}")
    print(f"DECISION SUMMARY")
    print(f"{'='*40}")
    print(f"  Closure:   DxR2@10={dxr2_cl[9]:+.4f}, ACF(1)={acf_base_cl[1]:.4f}")
    print(f"  Switching: DxR2@10={dxr2_sw[9]:+.4f}, ACF(1)={acf_sw[1]:.4f}")
    print(f"  Delta DxR2@10: {delta_dxr2:+.4f}")
    print(f"  Delta ACF(1):  {delta_acf:+.4f}")
    print(f"  Verdict: {verdict}")


if __name__ == '__main__':
    main()
