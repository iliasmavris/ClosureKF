"""
V7: Innovation Diagnostics Verification + Colored Measurement Noise Experiment.

Part A: Verify diagnostics (S_k variability, raw vs normalized ACF, NIS).
Part B: AR(1) measurement noise augmentation [x, u, n] with frozen physics/closure.

Uses the fair 2-state model from v6 controls as the reference.

Usage:  python -u scripts/v7_measnoise_diagnostics.py
Output: final_lockbox_v7_measnoise/
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
from scipy.optimize import minimize
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_closure import KalmanForecasterClosure, CLOSURE_PARAM_NAMES

# ===== Paths =====
DATA_DIR   = ROOT / "processed_data_10hz"
FAIR_DIR   = ROOT / "final_lockbox_v6_controls" / "fair_2state"
V2_CKPT    = ROOT / "final_lockbox_v2" / "checkpoints"

OUT = ROOT / "final_lockbox_v7_measnoise"
for d in ['diagnostics', 'figures', 'measnoise']:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ===== Constants =====
DT = 0.1; FORCE_CPU = True; MAX_HORIZON = 10
SEEDS = [42, 43, 44]
BLOCK_LEN_S = 3.0; R_BOOT = 2000; RNG_SEED = 54321
VAR_FLOOR = 1e-6

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


def compute_acf(e, max_lag=100):
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


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ============================================================
#  NUMPY 2-STATE FILTER (returns full diagnostics)
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
#  NUMPY 3-STATE FILTER (augmented: [x, u, n] with H=[1,0,1])
# ============================================================

def kf_filter_measnoise(params, cl_params, mn_params, t, x_obs, v):
    """3-state KF [x, u, n] with AR(1) measurement noise.

    Observation: y_k = x_k + n_k  =>  H = [1, 0, 1]
    Noise dynamics: n_{k+1} = phi_n * n_k + xi_k, xi ~ N(0, q_n)

    Returns: innovations, S_values, states_x, states_u, states_n, P_post_list
    """
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values    = np.full(N, np.nan)
    states_x = np.zeros(N); states_u = np.zeros(N); states_n = np.zeros(N)
    P_post_list = [None] * N

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R_white = mn_params.get('R_white', params['R'])  # white remainder
    phi_n = mn_params['phi_n']
    q_n   = mn_params['q_n']
    P0_nn = mn_params.get('P0_nn', q_n / max(1.0 - phi_n**2, 1e-6))

    a1   = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1   = cl_params.get('d1', 0.0)
    d2   = cl_params.get('d2', 0.0)
    d3   = cl_params.get('d3', 0.0)

    # H = [1, 0, 1]
    H = np.array([1.0, 0.0, 1.0])

    s = np.array([x_obs[0], 0.0, 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu'], P0_nn])
    states_x[0] = s[0]; states_u[0] = s[1]; states_n[0] = s[2]
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
        n_p = phi_n * s[2]
        s_pred = np.array([x_p, u_p, n_p])

        F_mat = np.array([
            [1.0,      dt,    0.0  ],
            [-kap*dt,  rho_u, 0.0  ],
            [0.0,      0.0,   phi_n],
        ])
        Q_mat = np.diag([q_sc*qx*dt, q_sc*qu*dt, q_n])
        P_pred = F_mat @ P @ F_mat.T + Q_mat

        # Innovation: y_k - H @ s_pred = x_obs[k] - (x_p + n_p)
        innov = x_obs[k] - H @ s_pred
        S_val = H @ P_pred @ H + R_white
        innovations[k] = innov; S_values[k] = S_val

        K = P_pred @ H / S_val
        s = s_pred + K * innov
        IKH = np.eye(3) - np.outer(K, H)
        P = IKH @ P_pred @ IKH.T + R_white * np.outer(K, K)
        states_x[k] = s[0]; states_u[k] = s[1]; states_n[k] = s[2]
        P_post_list[k] = P.copy()

    return innovations, S_values, states_x, states_u, states_n, P_post_list


# ============================================================
#  DxR2 FUNCTIONS
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


def compute_dxr2_2state(params, cl_params, states_x, states_u,
                        t, x_obs, v, max_h=10, eval_start=1, mode='oracle'):
    N = len(x_obs)
    r2_arr = np.zeros(max_h)
    for h in range(1, max_h + 1):
        dx_pred = []; dx_obs = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            sx_end, _ = _rollout_2state(sx, su, h, t, v, i+1, params, cl_params, mode)
            dx_pred.append(sx_end - x_obs[i])
            dx_obs.append(x_obs[i + h] - x_obs[i])
        dp = np.array(dx_pred); do = np.array(dx_obs)
        ss_res = np.sum((do - dp)**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h - 1] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


def compute_dxr2_measnoise(params, cl_params, mn_params,
                           states_x, states_u, states_n,
                           t, x_obs, v, max_h=10, eval_start=1, mode='oracle'):
    """DxR2 for measnoise model: predicted observation = x_pred + n_pred."""
    N = len(x_obs)
    phi_n = mn_params['phi_n']
    r2_arr = np.zeros(max_h)
    for h in range(1, max_h + 1):
        dx_pred = []; dx_obs = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            sn = states_n[i]
            # Roll out x,u (same physics) and n (AR(1) decay)
            sx_end, _ = _rollout_2state(sx, su, h, t, v, i+1, params, cl_params, mode)
            n_end = sn * (phi_n ** h)
            # Predicted observation = x + n
            y_pred_end = sx_end + n_end
            y_pred_start = x_obs[i]  # y_i is observed
            dx_pred.append(y_pred_end - y_pred_start)
            dx_obs.append(x_obs[i + h] - x_obs[i])
        dp = np.array(dx_pred); do = np.array(dx_obs)
        ss_res = np.sum((do - dp)**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h - 1] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ============================================================
#  UQ FUNCTIONS (h-step covariance propagation)
# ============================================================

def compute_hstep_uq_2state(params, cl_params, states_x, states_u,
                            P_post_list, t, x_obs, v,
                            max_h=10, eval_start=1, mode='oracle'):
    N = len(x_obs)
    alpha = params['alpha']; kap = params['kappa']
    q_sc = cl_params.get('q_scale', 1.0); R = params['R']

    results = {}
    for h in range(1, max_h + 1):
        obs_list = []; mean_list = []; var_list = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            P_h = P_post_list[i][:2, :2].copy() if P_post_list[i].shape[0] > 2 else P_post_list[i].copy()
            for step in range(h):
                k_s = i + 1 + step
                if k_s >= N: break
                dt_s = t[k_s] - t[k_s-1] if k_s > 0 else 0.1
                if dt_s <= 0: dt_s = 0.1
                rho = math.exp(-alpha * dt_s)
                sx, su = _rollout_2state(sx, su, 1, t, v, k_s, params, cl_params, mode)
                F_mat = np.array([[1, dt_s], [-kap*dt_s, rho]])
                Q_mat = np.diag([q_sc*params['qx']*dt_s, q_sc*params['qu']*dt_s])
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


def compute_hstep_uq_measnoise(params, cl_params, mn_params,
                                states_x, states_u, states_n,
                                P_post_list, t, x_obs, v,
                                max_h=10, eval_start=1, mode='oracle'):
    """UQ for measnoise model. Observation = H @ state = x + n."""
    N = len(x_obs)
    alpha = params['alpha']; kap = params['kappa']
    q_sc = cl_params.get('q_scale', 1.0)
    R_white = mn_params.get('R_white', params['R'])
    phi_n = mn_params['phi_n']; q_n = mn_params['q_n']
    H = np.array([1.0, 0.0, 1.0])

    results = {}
    for h in range(1, max_h + 1):
        obs_list = []; mean_list = []; var_list = []
        for i in range(max(eval_start, 1), N - h):
            sx, su = states_x[i], states_u[i]
            sn = states_n[i]
            P_h = P_post_list[i].copy()
            for step in range(h):
                k_s = i + 1 + step
                if k_s >= N: break
                dt_s = t[k_s] - t[k_s-1] if k_s > 0 else 0.1
                if dt_s <= 0: dt_s = 0.1
                rho = math.exp(-alpha * dt_s)
                sx, su = _rollout_2state(sx, su, 1, t, v, k_s, params, cl_params, mode)
                sn = phi_n * sn
                F_mat = np.array([
                    [1.0, dt_s, 0.0],
                    [-kap*dt_s, rho, 0.0],
                    [0.0, 0.0, phi_n],
                ])
                Q_mat = np.diag([q_sc*params['qx']*dt_s, q_sc*params['qu']*dt_s, q_n])
                P_h = F_mat @ P_h @ F_mat.T + Q_mat
            y_pred = sx + sn  # H @ s
            obs_list.append(x_obs[i + h])
            mean_list.append(y_pred)
            var_list.append(H @ P_h @ H + R_white)
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
    block_windows = []; nonempty = []
    for b in range(n_blocks):
        lo = b * block_len_idx; hi = min(N_win, (b+1)*block_len_idx)
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
        if len(win_idx) == 0: boot_vof[r] = np.nan; continue
        for h in range(1, max_h + 1):
            hk = f'h{h}'
            idx = win_idx[win_idx < len(uq_orc[hk]['obs'])]
            if len(idx) == 0: boot_vof[r, h-1] = np.nan; continue
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
#  TORCH 3-STATE MEASNOISE MODEL (for gradient-based fitting)
# ============================================================

class KalmanMeasNoise(nn.Module):
    """3-state [x, u, n]: physics+closure frozen, learn tau_n + q_n only.

    n_{k+1} = phi_n * n_k + xi_k,  phi_n = exp(-dt/tau_n)
    y_k = x_k + n_k + eps_k,  eps_k ~ N(0, R_white)
    """

    def __init__(self, base_model, tau_n_init=0.5, q_n_init=1e-4, R_white_init=1e-6):
        super().__init__()
        self.base = base_model  # frozen KalmanForecasterClosure
        # Learnable: log_tau_n, log_q_n, log_R_white
        self.log_tau_n = nn.Parameter(torch.tensor(math.log(max(tau_n_init, 0.01))))
        self.log_q_n = nn.Parameter(torch.tensor(math.log(max(q_n_init, 1e-10))))
        self.log_R_white = nn.Parameter(torch.tensor(math.log(max(R_white_init, 1e-10))))
        self.log_P0_nn = nn.Parameter(torch.tensor(math.log(max(q_n_init, 1e-10))))

    @property
    def tau_n(self): return torch.exp(self.log_tau_n)
    @property
    def q_n(self): return torch.exp(self.log_q_n)
    @property
    def R_white(self): return torch.exp(self.log_R_white)
    @property
    def P0_nn(self): return torch.exp(self.log_P0_nn)

    def forward(self, v_hist, dt_hist, x_obs_hist, v_fut, dt_fut):
        """Full filter + forecast with augmented measurement noise state."""
        B, L = v_hist.shape
        Hf = v_fut.shape[1]
        dev = v_hist.device

        # Retrieve base model properties
        base = self.base
        alpha = base.alpha; c_val = base.c; kap = base.kappa; vc = base.vc
        qx = base.qx; qu = base.qu; R_base = base.R
        q_sc = base.q_scale

        tau_n = self.tau_n; q_n = self.q_n; R_w = self.R_white

        # H = [1, 0, 1] for observation y = x + n
        H_vec = torch.tensor([1.0, 0.0, 1.0], device=dev, dtype=v_hist.dtype)

        # Initial state [x, u, n]
        s = torch.zeros(B, 3, device=dev, dtype=v_hist.dtype)
        s[:, 0] = x_obs_hist[:, 0]

        P = torch.zeros(B, 3, 3, device=dev, dtype=v_hist.dtype)
        P[:, 0, 0] = torch.exp(base.log_p0_xx)
        P[:, 1, 1] = torch.exp(base.log_p0_uu)
        P[:, 2, 2] = self.P0_nn

        # Filter through history
        for k in range(1, L):
            dt_k = dt_hist[:, k].clamp(min=1e-6)
            v_curr = v_hist[:, k - 1]
            v_prev = v_hist[:, k - 2] if k >= 2 else v_hist[:, 0]
            dv = v_curr - v_prev if k >= 2 else torch.zeros_like(v_curr)

            rho = torch.exp(-alpha * dt_k)
            phi_n_k = torch.exp(-dt_k / tau_n)
            g = torch.relu(v_curr * v_curr - vc * vc)
            cl = base.closure(s[:, 1], v_curr, dv)

            x_p = s[:, 0] + s[:, 1] * dt_k
            u_p = (rho * s[:, 1] - kap * s[:, 0] * dt_k
                   + c_val * g * dt_k + cl * dt_k)
            n_p = phi_n_k * s[:, 2]
            s_pred = torch.stack([x_p, u_p, n_p], dim=1)

            # F matrix
            F_mat = torch.zeros(B, 3, 3, device=dev, dtype=v_hist.dtype)
            F_mat[:, 0, 0] = 1.0; F_mat[:, 0, 1] = dt_k
            F_mat[:, 1, 0] = -kap * dt_k; F_mat[:, 1, 1] = rho
            F_mat[:, 2, 2] = phi_n_k

            Q_mat = torch.zeros(B, 3, 3, device=dev, dtype=v_hist.dtype)
            Q_mat[:, 0, 0] = q_sc * qx * dt_k
            Q_mat[:, 1, 1] = q_sc * qu * dt_k
            Q_mat[:, 2, 2] = q_n

            P_pred = torch.bmm(torch.bmm(F_mat, P), F_mat.transpose(1, 2)) + Q_mat

            # Update: y = H @ s + eps
            innov = x_obs_hist[:, k] - (s_pred[:, 0] + s_pred[:, 2])
            S = (P_pred[:, 0, 0] + 2*P_pred[:, 0, 2] + P_pred[:, 2, 2]
                 + R_w)
            # K = P_pred @ H / S
            K_col = (P_pred[:, :, 0] + P_pred[:, :, 2]) / S.unsqueeze(1)
            s = s_pred + K_col * innov.unsqueeze(1)

            eye3 = torch.eye(3, device=dev, dtype=v_hist.dtype).unsqueeze(0)
            KH = K_col.unsqueeze(2) * H_vec.unsqueeze(0).unsqueeze(0)
            IKH = eye3 - KH
            P = (torch.bmm(torch.bmm(IKH, P_pred), IKH.transpose(1, 2))
                 + R_w * K_col.unsqueeze(2) * K_col.unsqueeze(1))

        # Forecast
        x_preds, x_vars = [], []
        for k in range(Hf):
            dt_k = dt_fut[:, k].clamp(min=1e-6)
            v_prev = v_hist[:, -1] if k == 0 else v_fut[:, k - 1]
            v_curr = v_fut[:, k]; dv = v_curr - v_prev

            rho = torch.exp(-alpha * dt_k)
            phi_n_k = torch.exp(-dt_k / tau_n)
            g = torch.relu(v_curr * v_curr - vc * vc)
            cl = base.closure(s[:, 1], v_curr, dv)

            x_p = s[:, 0] + s[:, 1] * dt_k
            u_p = (rho * s[:, 1] - kap * s[:, 0] * dt_k
                   + c_val * g * dt_k + cl * dt_k)
            n_p = phi_n_k * s[:, 2]
            s_pred = torch.stack([x_p, u_p, n_p], dim=1)

            F_mat = torch.zeros(B, 3, 3, device=dev, dtype=v_hist.dtype)
            F_mat[:, 0, 0] = 1.0; F_mat[:, 0, 1] = dt_k
            F_mat[:, 1, 0] = -kap * dt_k; F_mat[:, 1, 1] = rho
            F_mat[:, 2, 2] = phi_n_k

            Q_mat = torch.zeros(B, 3, 3, device=dev, dtype=v_hist.dtype)
            Q_mat[:, 0, 0] = q_sc * qx * dt_k
            Q_mat[:, 1, 1] = q_sc * qu * dt_k
            Q_mat[:, 2, 2] = q_n

            P_pred = torch.bmm(torch.bmm(F_mat, P), F_mat.transpose(1, 2)) + Q_mat
            s = s_pred; P = P_pred

            # Predicted observation = x + n, var = H @ P @ H + R_white
            y_pred = s[:, 0] + s[:, 2]
            y_var = (P[:, 0, 0] + 2*P[:, 0, 2] + P[:, 2, 2] + R_w)
            x_preds.append(y_pred)
            x_vars.append(y_var)

        return (torch.stack(x_preds, dim=1),
                torch.stack(x_vars, dim=1),
                s[:, 1].unsqueeze(1))  # dummy u_est


# ============================================================
#  TRAINING LOOP (lightweight, for fitting tau_n + q_n only)
# ============================================================

def train_measnoise(model, train_loader, val_loader, device,
                    max_epochs=150, patience=25, lr=1e-2, tag="mn"):
    params = [model.log_tau_n, model.log_q_n, model.log_R_white, model.log_P0_nn]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=8)

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
            v_clamped = torch.clamp(xv, min=VAR_FLOOR)
            loss = (0.5 * torch.log(2*math.pi*v_clamped)
                    + 0.5*(x_true - xp)**2 / v_clamped).mean()
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
                v_clamped = torch.clamp(xv, min=VAR_FLOOR)
                vl = (0.5*torch.log(2*math.pi*v_clamped)
                      + 0.5*(x_true-xp)**2/v_clamped).mean()
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
            tau_n = model.tau_n.item()
            q_n = model.q_n.item()
            r_w = model.R_white.item()
            print(f"    [{tag}] ep {ep+1:3d}  tr={tr_nll:.4f}  val={val_nll:.4f}  "
                  f"tau_n={tau_n:.4f}  q_n={q_n:.6f}  R_w={r_w:.2e}")
        if wait >= patience:
            print(f"    [{tag}] Early stop at ep {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done {elapsed:.0f}s, best val={best_loss:.4f}")
    return best_loss


# ============================================================
#  MAIN
# ============================================================

def main():
    t0_global = time.time()
    device = torch.device('cpu')
    print("="*70)
    print("V7: INNOVATION DIAGNOSTICS + MEASUREMENT NOISE")
    print("="*70)
    print(f"Output -> {OUT}")

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------
    print_section("DATA LOADING")

    # Load fair 2-state stage-1 physics
    ck_s1 = torch.load(FAIR_DIR / "stage1_fair.pth",
                        map_location='cpu', weights_only=False)
    fair_phys = ck_s1['params']
    print(f"  Fair S1: alpha={fair_phys['alpha']:.4f} "
          f"c={fair_phys['c']:.4f} kappa={fair_phys['kappa']:.4f}")

    # Average fair closure across seeds
    fair_cl_list = []
    for s in SEEDS:
        ck = torch.load(FAIR_DIR / f"closure_fair_s{s}.pth",
                        map_location='cpu', weights_only=False)
        fair_cl_list.append(ck['closure'])
    cl_fair = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    for key in ['b2', 'd2', 'q_scale']:
        cl_fair[key] = float(np.mean([r[key] for r in fair_cl_list]))
    print(f"  Fair closure: b2={cl_fair['b2']:.3f} d2={cl_fair['d2']:.3f} "
          f"q_scale={cl_fair['q_scale']:.3f}")

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

    # Dev split
    dev_path = OUT / "dev_10hz_ready.csv"
    df_dev = df_val[df_val['timestamp'] < TEST_START].copy()
    df_dev.to_csv(dev_path, index=False)

    L = 64; H = 20; BATCH = 128
    train_ds = StateSpaceDataset(
        [str(DATA_DIR / "train_10hz_ready.csv")], L=L, m=L, H=H,
        predict_deltas=False, normalize=False)
    val_ds = StateSpaceDataset(
        [str(dev_path)], L=L, m=L, H=H,
        predict_deltas=False, normalize=False,
        run_id_to_idx=train_ds.run_id_to_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)}, Val(dev): {len(val_ds)}")

    # ============================================================
    # PART A: VERIFY DIAGNOSTICS
    # ============================================================
    print_section("PART A: VERIFY INNOVATION DIAGNOSTICS")

    # A1: Filter and get S_k
    print("  A1: S_k variability")
    innov, S_vals, sx, su, P_list = kf_filter_2state_full(
        fair_phys, cl_fair, t_arr, x_arr, v_arr)

    e_raw = innov[test_mask]; S_test = S_vals[test_mask]
    valid = ~np.isnan(e_raw)
    e_raw_v = e_raw[valid]; S_v = S_test[valid]
    t_test = t_arr[test_mask][valid]

    S_stats = {
        'min':    float(np.min(S_v)),
        'p5':     float(np.percentile(S_v, 5)),
        'median': float(np.median(S_v)),
        'p95':    float(np.percentile(S_v, 95)),
        'max':    float(np.max(S_v)),
        'mean':   float(np.mean(S_v)),
        'std':    float(np.std(S_v)),
        'cv':     float(np.std(S_v) / np.mean(S_v)),
    }
    print(f"    S_k: min={S_stats['min']:.6e}  p5={S_stats['p5']:.6e}  "
          f"median={S_stats['median']:.6e}")
    print(f"         p95={S_stats['p95']:.6e}  max={S_stats['max']:.6e}  "
          f"CV={S_stats['cv']:.4f}")

    # Save S_k time series
    df_sk = pd.DataFrame({'timestamp': t_test, 'S_k': S_v, 'innovation': e_raw_v})
    df_sk.to_csv(OUT / "diagnostics" / "innovations_raw.csv", index=False)

    # S_k plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_test, S_v, lw=0.5, alpha=0.8)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('S_k (innovation variance)')
    ax.set_title(f'Innovation Covariance S_k (CV={S_stats["cv"]:.4f})')
    ax.axhline(S_stats['median'], color='r', ls='--', lw=1, label=f'median={S_stats["median"]:.2e}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_Sk_timeseries.png")
    plt.close(fig)
    print("    Saved fig_Sk_timeseries.png")

    # A2: Raw vs normalized ACF
    print("\n  A2: Raw vs normalized innovations ACF")
    e_norm_v = e_raw_v / np.sqrt(np.maximum(S_v, 1e-15))
    n_valid = len(e_raw_v)

    # Verify normalization is actually applied
    raw_std = np.std(e_raw_v)
    norm_std = np.std(e_norm_v)
    print(f"    Raw std:  {raw_std:.6f}")
    print(f"    Norm std: {norm_std:.6f}  (should be ~1 if consistent)")
    print(f"    Ratio:    {norm_std/raw_std:.6f}")

    acf_raw  = compute_acf(e_raw_v, max_lag=100)
    acf_norm = compute_acf(e_norm_v, max_lag=100)

    print(f"    ACF(1)  raw={acf_raw[1]:.6f}  norm={acf_norm[1]:.6f}  "
          f"delta={abs(acf_raw[1]-acf_norm[1]):.2e}")
    print(f"    ACF(5)  raw={acf_raw[5]:.6f}  norm={acf_norm[5]:.6f}")
    print(f"    ACF(10) raw={acf_raw[10]:.6f}  norm={acf_norm[10]:.6f}")

    # Check: if delta is tiny, it means S_k is nearly constant
    acf_delta_1 = abs(acf_raw[1] - acf_norm[1])
    sk_is_constant = acf_delta_1 < 0.01
    print(f"    S_k near-constant? {sk_is_constant} (ACF1 delta={acf_delta_1:.6f})")

    lb_raw  = ljung_box(acf_raw, n_valid, lags=[5, 10, 20, 50])
    lb_norm = ljung_box(acf_norm, n_valid, lags=[5, 10, 20, 50])

    for lb in lb_raw:
        print(f"    LB raw  lag={lb['lag']:2d}  Q={lb['Q']:.1f}  p={lb['p']:.4f}")
    for lb in lb_norm:
        print(f"    LB norm lag={lb['lag']:2d}  Q={lb['Q']:.1f}  p={lb['p']:.4f}")

    # Save ACF CSVs
    df_acf_raw = pd.DataFrame({'lag': np.arange(101), 'acf': acf_raw})
    df_acf_raw.to_csv(OUT / "diagnostics" / "acf_raw.csv", index=False)
    df_acf_norm = pd.DataFrame({'lag': np.arange(101), 'acf': acf_norm})
    df_acf_norm.to_csv(OUT / "diagnostics" / "acf_norm.csv", index=False)

    # ACF comparison plot
    sig_band = 1.96 / math.sqrt(n_valid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, acf, title in [(axes[0], acf_raw, 'Raw innovations'),
                           (axes[1], acf_norm, 'Normalized innovations')]:
        lags = np.arange(1, 51)
        ax.bar(lags, acf[1:51], width=0.7, color='steelblue', alpha=0.7)
        ax.axhline(sig_band, color='r', ls='--', lw=1)
        ax.axhline(-sig_band, color='r', ls='--', lw=1)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
        ax.set_title(f'{title}\nACF(1)={acf[1]:.4f}')
    fig.suptitle('Fair 2-State: Raw vs Normalized Innovation ACF', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_acf_raw_vs_norm.png")
    plt.close(fig)
    print("    Saved fig_acf_raw_vs_norm.png")

    # A3: NIS consistency
    print("\n  A3: NIS consistency")
    nis = e_raw_v**2 / np.maximum(S_v, 1e-15)
    nis_mean = float(np.mean(nis))
    chi2_lo = sp_stats.chi2.ppf(0.025, df=1)
    chi2_hi = sp_stats.chi2.ppf(0.975, df=1)
    frac_in_bounds = float(np.mean((nis >= chi2_lo) & (nis <= chi2_hi)))

    print(f"    NIS mean = {nis_mean:.4f} (should be ~1.0)")
    print(f"    95% chi2(1) bounds: [{chi2_lo:.4f}, {chi2_hi:.4f}]")
    print(f"    Fraction in bounds: {frac_in_bounds:.4f}")

    nis_summary = {
        'nis_mean': nis_mean,
        'chi2_lo_95': float(chi2_lo),
        'chi2_hi_95': float(chi2_hi),
        'frac_in_95_bounds': frac_in_bounds,
        'S_k_stats': S_stats,
        'acf1_raw': float(acf_raw[1]),
        'acf1_norm': float(acf_norm[1]),
        'acf1_delta': float(acf_delta_1),
        'sk_is_constant': bool(sk_is_constant),
        'raw_std': float(raw_std),
        'norm_std': float(norm_std),
        'ljung_box_raw': lb_raw,
        'ljung_box_norm': lb_norm,
        'n_valid': n_valid,
    }
    with open(OUT / "diagnostics" / "nis_summary.json", 'w') as f:
        json.dump(nis_summary, f, indent=2)

    # NIS time series plot
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t_test, nis, lw=0.3, alpha=0.5, color='steelblue', label='NIS')
    # Running mean for visibility
    win = 50
    if len(nis) > win:
        nis_smooth = np.convolve(nis, np.ones(win)/win, mode='valid')
        t_smooth = t_test[win//2:win//2+len(nis_smooth)]
        ax.plot(t_smooth, nis_smooth, lw=1.5, color='darkblue',
                label=f'Running mean (w={win})')
    ax.axhline(1.0, color='green', ls='-', lw=1.5, label='Expected (df=1)')
    ax.axhline(chi2_hi, color='r', ls='--', lw=1, label=f'95% upper={chi2_hi:.2f}')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('NIS = e_k^2 / S_k')
    ax.set_title(f'Normalized Innovation Squared (mean={nis_mean:.3f})')
    ax.set_ylim(0, min(np.percentile(nis, 99.5), 20))
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_nis_timeseries.png")
    plt.close(fig)
    print("    Saved fig_nis_timeseries.png")

    # ============================================================
    # PART B: MEASUREMENT NOISE AUGMENTATION
    # ============================================================
    print_section("PART B: AR(1) MEASUREMENT NOISE MODEL")

    # Build frozen base model
    print("  B1: Building frozen base model + fitting tau_n, q_n")
    p = fair_phys
    base_model = KalmanForecasterClosure(
        alpha_init=p['alpha'], c_init=p['c'],
        vc_init=p['vc'], kappa_init=p['kappa'],
        log_qx_init=math.log(p['qx']),
        log_qu_init=math.log(p['qu']),
        log_r_init=math.log(p['R']),
        log_p0_xx_init=math.log(p['P0_xx']),
        log_p0_uu_init=math.log(p['P0_uu']),
        a1_init=0.001, b1_init=0.0,
        b2_init=cl_fair['b2'],
        d1_init=0.001, d2_init=cl_fair['d2'], d3_init=0.001,
    )
    # Freeze everything in base model
    for param in base_model.parameters():
        param.requires_grad_(False)
    # Set closure params properly via state_dict
    with torch.no_grad():
        base_model.b2.fill_(cl_fair['b2'])
        # d2 uses softplus: need inverse
        d2_val = cl_fair['d2']
        base_model.d2_raw.fill_(math.log(math.exp(d2_val) - 1.0 + 1e-6))
        base_model.log_q_scale.fill_(math.log(cl_fair['q_scale']))

    base_model = base_model.to(device)

    # Train measurement noise model (3 seeds for robustness)
    mn_results = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = KalmanMeasNoise(
            base_model,
            tau_n_init=0.5,       # initial guess: 0.5s
            q_n_init=1e-4,        # small initial
            R_white_init=1e-6,    # very small white remainder
        ).to(device)
        tag = f"mn_s{seed}"
        train_measnoise(model, train_loader, val_loader, device,
                       max_epochs=150, patience=25, lr=1e-2, tag=tag)
        with torch.no_grad():
            res = {
                'seed': seed,
                'tau_n': model.tau_n.item(),
                'phi_n': math.exp(-DT / model.tau_n.item()),
                'q_n': model.q_n.item(),
                'R_white': model.R_white.item(),
                'P0_nn': model.P0_nn.item(),
            }
        mn_results.append(res)
        print(f"    Seed {seed}: tau_n={res['tau_n']:.4f}s  "
              f"phi_n={res['phi_n']:.4f}  q_n={res['q_n']:.6f}  "
              f"R_white={res['R_white']:.2e}")
        torch.save({
            'state_dict': model.state_dict(),
            'mn_params': res,
            'seed': seed,
        }, OUT / "measnoise" / f"measnoise_s{seed}.pth")

    # Average
    mn_avg = {}
    for key in ['tau_n', 'q_n', 'R_white', 'P0_nn']:
        mn_avg[key] = float(np.mean([r[key] for r in mn_results]))
    mn_avg['phi_n'] = math.exp(-DT / mn_avg['tau_n'])
    print(f"\n  Averaged: tau_n={mn_avg['tau_n']:.4f}s  "
          f"phi_n={mn_avg['phi_n']:.4f}  q_n={mn_avg['q_n']:.6f}  "
          f"R_white={mn_avg['R_white']:.2e}")

    # ============================================================
    # PART B2: EVALUATE MEASUREMENT NOISE MODEL
    # ============================================================
    print_section("PART B2: EVALUATE MEASUREMENT NOISE MODEL")

    # Filter with measnoise model on test data
    mn_eval_params = {
        'phi_n': mn_avg['phi_n'],
        'q_n': mn_avg['q_n'],
        'R_white': mn_avg['R_white'],
        'P0_nn': mn_avg.get('P0_nn', mn_avg['q_n'] / max(1.0 - mn_avg['phi_n']**2, 1e-6)),
    }
    innov_mn, S_mn, sx_mn, su_mn, sn_mn, P_mn_list = kf_filter_measnoise(
        fair_phys, cl_fair, mn_eval_params, t_arr, x_arr, v_arr)

    e_mn_raw = innov_mn[test_mask]
    S_mn_test = S_mn[test_mask]
    valid_mn = ~np.isnan(e_mn_raw)
    e_mn_raw_v = e_mn_raw[valid_mn]
    S_mn_v = S_mn_test[valid_mn]

    # S_k stats for measnoise
    S_mn_stats = {
        'min':    float(np.min(S_mn_v)),
        'p5':     float(np.percentile(S_mn_v, 5)),
        'median': float(np.median(S_mn_v)),
        'p95':    float(np.percentile(S_mn_v, 95)),
        'max':    float(np.max(S_mn_v)),
        'cv':     float(np.std(S_mn_v) / np.mean(S_mn_v)),
    }
    print(f"  S_k (measnoise): CV={S_mn_stats['cv']:.4f}  "
          f"median={S_mn_stats['median']:.6e}")

    # ACF
    e_mn_norm_v = e_mn_raw_v / np.sqrt(np.maximum(S_mn_v, 1e-15))
    acf_mn_raw  = compute_acf(e_mn_raw_v, max_lag=100)
    acf_mn_norm = compute_acf(e_mn_norm_v, max_lag=100)
    n_mn = len(e_mn_raw_v)

    print(f"  ACF(1) raw={acf_mn_raw[1]:.6f}  norm={acf_mn_norm[1]:.6f}")

    lb_mn_raw  = ljung_box(acf_mn_raw, n_mn, lags=[5, 10, 20, 50])
    lb_mn_norm = ljung_box(acf_mn_norm, n_mn, lags=[5, 10, 20, 50])

    # NIS
    nis_mn = e_mn_raw_v**2 / np.maximum(S_mn_v, 1e-15)
    nis_mn_mean = float(np.mean(nis_mn))
    frac_mn = float(np.mean((nis_mn >= chi2_lo) & (nis_mn <= chi2_hi)))
    print(f"  NIS mean = {nis_mn_mean:.4f}  frac_in_95% = {frac_mn:.4f}")

    # DxR2
    print("  Computing DxR2...")
    dxr2_mn_orc = compute_dxr2_measnoise(
        fair_phys, cl_fair, mn_eval_params, sx_mn, su_mn, sn_mn,
        t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'oracle')
    dxr2_mn_nof = compute_dxr2_measnoise(
        fair_phys, cl_fair, mn_eval_params, sx_mn, su_mn, sn_mn,
        t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'no_forcing')
    print(f"  DxR2@10 oracle={dxr2_mn_orc[9]:+.4f}  no_forcing={dxr2_mn_nof[9]:+.4f}")
    print(f"  mean DxR2(5-10) oracle={np.mean(dxr2_mn_orc[4:10]):+.4f}")

    # VoF
    print("  Computing VoF...")
    block_len_idx = max(1, round(BLOCK_LEN_S / DT))
    uq_mn_orc = compute_hstep_uq_measnoise(
        fair_phys, cl_fair, mn_eval_params, sx_mn, su_mn, sn_mn,
        P_mn_list, t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'oracle')
    uq_mn_nof = compute_hstep_uq_measnoise(
        fair_phys, cl_fair, mn_eval_params, sx_mn, su_mn, sn_mn,
        P_mn_list, t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'no_forcing')
    vof_mn_pct, vof_mn_lo, vof_mn_hi = compute_vof_bootstrap(
        uq_mn_orc, uq_mn_nof, MAX_HORIZON, block_len_idx, R_BOOT, RNG_SEED)
    vof_mn_avg = float(np.mean(vof_mn_pct[4:10]))
    print(f"  VoF% avg(5-10) = {vof_mn_avg:+.3f}%")

    # Also compute fair 2-state DxR2 + VoF for comparison table
    print("\n  Computing fair 2-state DxR2 + VoF for comparison...")
    dxr2_fair_orc = compute_dxr2_2state(
        fair_phys, cl_fair, sx, su,
        t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'oracle')
    dxr2_fair_nof = compute_dxr2_2state(
        fair_phys, cl_fair, sx, su,
        t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'no_forcing')
    print(f"  Fair DxR2@10 oracle={dxr2_fair_orc[9]:+.4f}  no_forcing={dxr2_fair_nof[9]:+.4f}")

    uq_fair_orc = compute_hstep_uq_2state(
        fair_phys, cl_fair, sx, su, P_list,
        t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'oracle')
    uq_fair_nof = compute_hstep_uq_2state(
        fair_phys, cl_fair, sx, su, P_list,
        t_arr, x_arr, v_arr, MAX_HORIZON, test_eval_start, 'no_forcing')
    vof_fair_pct, vof_fair_lo, vof_fair_hi = compute_vof_bootstrap(
        uq_fair_orc, uq_fair_nof, MAX_HORIZON, block_len_idx, R_BOOT, RNG_SEED)
    vof_fair_avg = float(np.mean(vof_fair_pct[4:10]))
    print(f"  Fair VoF% avg(5-10) = {vof_fair_avg:+.3f}%")

    # ============================================================
    # COMPARISON TABLE + FIGURES
    # ============================================================
    print_section("COMPARISON TABLE")

    print(f"  {'Metric':<25s}  {'Fair 2-State':>14s}  {'MeasNoise':>14s}  {'Delta':>14s}")
    print(f"  {'-'*70}")
    metrics = [
        ('ACF(1) raw',    acf_raw[1],       acf_mn_raw[1]),
        ('ACF(1) norm',   acf_norm[1],      acf_mn_norm[1]),
        ('NIS mean',      nis_mean,          nis_mn_mean),
        ('NIS 95% frac',  frac_in_bounds,    frac_mn),
        ('DxR2@10 orc',   dxr2_fair_orc[9],  dxr2_mn_orc[9]),
        ('DxR2@10 nof',   dxr2_fair_nof[9],  dxr2_mn_nof[9]),
        ('mean DxR2(5-10)', np.mean(dxr2_fair_orc[4:10]), np.mean(dxr2_mn_orc[4:10])),
        ('VoF% avg(5-10)', vof_fair_avg,      vof_mn_avg),
    ]
    for name, fair_val, mn_val in metrics:
        delta = mn_val - fair_val
        print(f"  {name:<25s}  {fair_val:>14.6f}  {mn_val:>14.6f}  {delta:>+14.6f}")

    acf1_drop = acf_raw[1] - acf_mn_raw[1]
    material = acf1_drop >= 0.20
    dxr2_degradation = dxr2_fair_orc[9] - dxr2_mn_orc[9]
    skill_ok = dxr2_degradation < 0.05

    print(f"\n  ACF(1) drop = {acf1_drop:.4f}  (material if >= 0.20: {material})")
    print(f"  DxR2@10 degradation = {dxr2_degradation:+.4f}  (skill ok if < 0.05: {skill_ok})")

    # Decision
    if material and skill_ok:
        verdict = "MATERIAL IMPROVEMENT: measurement noise model reduces innovation correlation without skill loss"
    elif acf1_drop >= 0.10:
        verdict = "MODERATE IMPROVEMENT: some ACF reduction, worth considering"
    else:
        verdict = "DIMINISHING RETURNS: measurement noise does not materially reduce innovation correlation. Stop whiteness work."

    print(f"\n  VERDICT: {verdict}")

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    print_section("SAVING OUTPUTS")

    summary = {
        'fair_2state': {
            'acf1_raw': float(acf_raw[1]),
            'acf1_norm': float(acf_norm[1]),
            'nis_mean': float(nis_mean),
            'nis_frac_95': float(frac_in_bounds),
            'dxr2_10_oracle': float(dxr2_fair_orc[9]),
            'dxr2_10_no_forcing': float(dxr2_fair_nof[9]),
            'mean_dxr2_5_10_oracle': float(np.mean(dxr2_fair_orc[4:10])),
            'vof_avg_5_10': float(vof_fair_avg),
            'S_k_stats': S_stats,
            'ljung_box_raw': lb_raw,
            'ljung_box_norm': lb_norm,
        },
        'measnoise': {
            'acf1_raw': float(acf_mn_raw[1]),
            'acf1_norm': float(acf_mn_norm[1]),
            'nis_mean': float(nis_mn_mean),
            'nis_frac_95': float(frac_mn),
            'dxr2_10_oracle': float(dxr2_mn_orc[9]),
            'dxr2_10_no_forcing': float(dxr2_mn_nof[9]),
            'mean_dxr2_5_10_oracle': float(np.mean(dxr2_mn_orc[4:10])),
            'vof_avg_5_10': float(vof_mn_avg),
            'S_k_stats': S_mn_stats,
            'ljung_box_raw': lb_mn_raw,
            'ljung_box_norm': lb_mn_norm,
            'tau_n': mn_avg['tau_n'],
            'phi_n': mn_avg['phi_n'],
            'q_n': mn_avg['q_n'],
            'R_white': mn_avg['R_white'],
        },
        'comparison': {
            'acf1_drop': float(acf1_drop),
            'material_threshold': 0.20,
            'is_material': bool(material),
            'dxr2_degradation': float(dxr2_degradation),
            'skill_ok': bool(skill_ok),
            'verdict': verdict,
        },
        'per_seed': mn_results,
    }
    with open(OUT / "summary_v7_measnoise.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary_v7_measnoise.json")

    # Combined ACF comparison figure (key deliverable)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    sig_band = 1.96 / math.sqrt(n_valid)
    lags = np.arange(1, 51)

    ax = axes[0]
    ax.bar(lags - 0.2, acf_raw[1:51], width=0.4, color='steelblue',
           alpha=0.7, label=f'Fair 2-state (ACF1={acf_raw[1]:.3f})')
    ax.bar(lags + 0.2, acf_mn_raw[1:51], width=0.4, color='coral',
           alpha=0.7, label=f'MeasNoise (ACF1={acf_mn_raw[1]:.3f})')
    ax.axhline(sig_band, color='grey', ls='--', lw=1)
    ax.axhline(-sig_band, color='grey', ls='--', lw=1)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title('Raw Innovations ACF')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(lags - 0.2, acf_norm[1:51], width=0.4, color='steelblue',
           alpha=0.7, label=f'Fair 2-state (ACF1={acf_norm[1]:.3f})')
    ax.bar(lags + 0.2, acf_mn_norm[1:51], width=0.4, color='coral',
           alpha=0.7, label=f'MeasNoise (ACF1={acf_mn_norm[1]:.3f})')
    ax.axhline(sig_band, color='grey', ls='--', lw=1)
    ax.axhline(-sig_band, color='grey', ls='--', lw=1)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_title('Normalized Innovations ACF')
    ax.legend(fontsize=8)

    fig.suptitle('Innovation ACF: Fair 2-State vs Measurement Noise Model', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_acf_comparison.png")
    plt.close(fig)
    print(f"  Saved fig_acf_comparison.png")

    # NIS comparison figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, nis_arr, title, mean_val in [
        (axes[0], nis, f'Fair 2-State (mean NIS={nis_mean:.3f})', nis_mean),
        (axes[1], nis_mn, f'MeasNoise (mean NIS={nis_mn_mean:.3f})', nis_mn_mean)
    ]:
        t_plot = t_test[:len(nis_arr)]
        ax.plot(t_plot, nis_arr, lw=0.3, alpha=0.4, color='steelblue')
        if len(nis_arr) > win:
            nis_sm = np.convolve(nis_arr, np.ones(win)/win, mode='valid')
            t_sm = t_plot[win//2:win//2+len(nis_sm)]
            ax.plot(t_sm, nis_sm, lw=1.5, color='darkblue', label='Running mean')
        ax.axhline(1.0, color='green', ls='-', lw=1.5, label='Expected')
        ax.axhline(chi2_hi, color='r', ls='--', lw=1)
        ax.set_ylabel('NIS'); ax.set_title(title)
        ax.set_ylim(0, min(np.percentile(nis_arr, 99.5), 20))
        ax.legend(loc='upper right', fontsize=8)
    axes[1].set_xlabel('Time (s)')
    fig.suptitle('NIS Comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_nis_comparison.png")
    plt.close(fig)
    print(f"  Saved fig_nis_comparison.png")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    elapsed = time.time() - t0_global
    print_section("FINAL SUMMARY")
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {OUT}")
    print(f"\n  VERDICT: {verdict}")

    # List output files
    all_files = sorted(OUT.rglob('*'))
    files = [str(f.relative_to(OUT)) for f in all_files if f.is_file()]
    print(f"\n  Output files ({len(files)}):")
    for f in files:
        print(f"    {f}")


if __name__ == '__main__':
    main()
