"""
Shared utilities for No-Training Diagnostics.
All functions are evaluation-only -- NO TRAINING.
"""

import sys, math, warnings, json, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_closure import CLOSURE_PARAM_NAMES

# ===== Paths =====
DATA_DIR = ROOT / "processed_data_10hz"
V2_CKPT  = ROOT / "final_lockbox_v2" / "checkpoints"
S1_CKPT  = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
             / "stage1_physics_only.pth")
OUT_ROOT = ROOT / "final_lockbox_vX_no_train_diagnostics"

DT = 0.1
SEEDS_REF = [42, 43, 44]
MODES = ['oracle', 'persistence', 'no_forcing']


# ============================================================
#  Model loading
# ============================================================

def zero_closure():
    cl = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl['q_scale'] = 1.0
    return cl


def load_s1_params():
    ck = torch.load(S1_CKPT, map_location='cpu', weights_only=False)
    return ck['params']


def load_closure_params(seed):
    ck = torch.load(V2_CKPT / f"closure_2t_s{seed}.pth",
                    map_location='cpu', weights_only=False)
    return ck['closure']


def load_averaged_closure():
    cl_avg = zero_closure()
    for key in ['b2', 'd2', 'q_scale']:
        vals = [load_closure_params(s)[key] for s in SEEDS_REF]
        cl_avg[key] = float(np.mean(vals))
    return cl_avg


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
#  Data loading + filter
# ============================================================

def load_final_model_and_data():
    """Load model params, data, run closure + baseline filters.
    Returns a dict with everything needed by diagnostics."""
    s1_params = load_s1_params()
    cl_params = load_averaged_closure()

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

    test_idx_full = np.where(test_mask)[0]
    N_test = len(test_idx_full)

    print("Running closure filter...")
    filt = run_filter_on_split(s1_params, cl_params, t_arr, x_arr, v_arr)
    print("Running baseline filter...")
    base = run_filter_on_split(s1_params, zero_closure(), t_arr, x_arr, v_arr)

    return {
        'params': s1_params,
        'cl_params': cl_params,
        'df_train': df_train,
        't_arr': t_arr, 'x_arr': x_arr, 'v_arr': v_arr,
        'test_mask': test_mask,
        'test_idx_full': test_idx_full,
        'N_test': N_test,
        't_test': t_arr[test_mask],
        'x_test': x_arr[test_mask],
        'v_test': v_arr[test_mask],
        'filter': filt,
        'baseline_filter': base,
        'TEST_START': TEST_START,
    }


def run_filter_on_split(params, cl_params, t, x_obs, v):
    """Full KF forward pass. Returns dict with all arrays + P_post list."""
    N = len(x_obs)
    x_pred     = np.full(N, np.nan)
    innovations = np.full(N, np.nan)
    S_values   = np.full(N, np.nan)
    states_x   = np.zeros(N)
    states_u   = np.zeros(N)
    P_post     = []

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]
    P_post.append(P.copy())

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)

        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = b2_v * dv_w - d2 * u_st * abs(v_w)
        cl_dt = cl * dt

        xp = s[0] + s[1] * dt
        up = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt + cl_dt
        s_pred = np.array([xp, up])
        x_pred[k] = xp

        F_mat = np.array([[1, dt], [-kap*dt, rho_u]])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - xp
        S_val = P_pred[0, 0] + R
        innovations[k] = innov
        S_values[k] = S_val

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(2) - np.outer(K, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)

        states_x[k] = s[0]; states_u[k] = s[1]
        P_post.append(P.copy())

    return {
        'x_pred': x_pred,
        'innovations': innovations,
        'S_values': S_values,
        'states_x': states_x,
        'states_u': states_u,
        'P_post': P_post,
    }


# ============================================================
#  Open-loop rollout
# ============================================================

def predict_step(sx, su, v_w, dv_w, dt_k, params, cl_params):
    """Single open-loop predict step."""
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    rho = math.exp(-alpha * dt_k)
    g = max(v_w**2 - vc**2, 0.0)
    cl = b2_v * dv_w - d2 * su * abs(v_w)

    x_new = sx + su * dt_k
    u_new = rho * su - kap * sx * dt_k + c_val * g * dt_k + cl * dt_k
    return x_new, u_new


def rollout_open_loop(sx, su, i_start, h, t, v, params, cl_params,
                      mode='oracle', P=None):
    """h-step open-loop rollout. Returns dict with path arrays.

    mode: 'oracle', 'persistence', 'no_forcing'
    P: optional 2x2 covariance for propagation
    """
    v_persist = v[i_start] if i_start < len(v) else 0.0
    path_x = np.full(h, np.nan)
    path_u = np.full(h, np.nan)
    path_Pxx = np.full(h, np.nan) if P is not None else None

    cur_x, cur_u = sx, su
    cur_P = P.copy() if P is not None else None

    alpha = params['alpha']; kap = params['kappa']
    q_sc = cl_params.get('q_scale', 1.0)

    for step in range(h):
        k = i_start + step + 1
        if k >= len(t):
            break
        dt_k = t[k] - t[k-1]
        if dt_k <= 0: dt_k = 0.1

        if mode == 'oracle':
            v_w = v[k-1]
            dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        elif mode == 'persistence':
            v_w = v_persist
            dv_w = 0.0
        else:  # no_forcing
            v_w = 0.0
            dv_w = 0.0

        cur_x, cur_u = predict_step(
            cur_x, cur_u, v_w, dv_w, dt_k, params, cl_params)

        if cur_P is not None:
            rho = math.exp(-alpha * dt_k)
            F = np.array([[1, dt_k], [-kap*dt_k, rho]])
            Q = np.diag([q_sc*params['qx']*dt_k, q_sc*params['qu']*dt_k])
            cur_P = F @ cur_P @ F.T + Q
            path_Pxx[step] = cur_P[0, 0]

        path_x[step] = cur_x
        path_u[step] = cur_u

    result = {'path_x': path_x, 'path_u': path_u}
    if path_Pxx is not None:
        result['path_Pxx'] = path_Pxx
    return result


def full_rollout_paths_3mode(i_start, max_h, states_x, states_u,
                              t, v, params, cl_params):
    """Rollout up to max_h steps for all 3 modes simultaneously.
    Returns (oracle, persistence, no_forcing) each of shape (max_h,)."""
    sx = states_x[i_start]
    su = states_u[i_start]
    sx_o, su_o = sx, su
    sx_p, su_p = sx, su
    sx_n, su_n = sx, su
    v_persist = v[i_start] if i_start < len(v) else 0.0

    po = np.full(max_h, np.nan)
    pp = np.full(max_h, np.nan)
    pn = np.full(max_h, np.nan)

    for step in range(max_h):
        k = i_start + step + 1
        if k >= len(t):
            break
        dt_k = t[k] - t[k-1]
        if dt_k <= 0: dt_k = 0.1

        v_w = v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        sx_o, su_o = predict_step(sx_o, su_o, v_w, dv_w, dt_k, params, cl_params)
        po[step] = sx_o

        sx_p, su_p = predict_step(sx_p, su_p, v_persist, 0.0, dt_k, params, cl_params)
        pp[step] = sx_p

        sx_n, su_n = predict_step(sx_n, su_n, 0.0, 0.0, dt_k, params, cl_params)
        pn[step] = sx_n

    return po, pp, pn


def compute_all_rollout_paths(test_idx_full, max_h, filter_out,
                               t, v, params, cl_params, verbose=True):
    """Compute full rollout paths for all test points, all 3 modes.
    Returns dict with 'oracle', 'persistence', 'no_forcing' each (N_test, max_h)."""
    N_test = len(test_idx_full)
    N_full = len(t)
    states_x = filter_out['states_x']
    states_u = filter_out['states_u']
    paths = {
        'oracle': np.full((N_test, max_h), np.nan),
        'persistence': np.full((N_test, max_h), np.nan),
        'no_forcing': np.full((N_test, max_h), np.nan),
    }

    for j in range(N_test):
        i_full = test_idx_full[j]
        if i_full + 1 >= N_full:
            continue
        po, pp, pn = full_rollout_paths_3mode(
            i_full, max_h, states_x, states_u, t, v, params, cl_params)
        paths['oracle'][j] = po
        paths['persistence'][j] = pp
        paths['no_forcing'][j] = pn

        if verbose and (j + 1) % 300 == 0:
            print(f"    ... {j+1}/{N_test}")

    return paths


# ============================================================
#  Metrics
# ============================================================

def dxr2_at_horizon(paths_mode, x_launch, x_full, idx_full, h, N_full,
                    mask=None):
    """Compute DxR2 at horizon h from precomputed paths."""
    N = len(idx_full)
    if mask is None:
        mask = np.ones(N, dtype=bool)
    dx_pred = []
    dx_obs = []
    for j in range(N):
        if not mask[j]:
            continue
        i_full = idx_full[j]
        if i_full + h >= N_full:
            continue
        if h - 1 >= paths_mode.shape[1] or np.isnan(paths_mode[j, h - 1]):
            continue
        dx_pred.append(paths_mode[j, h - 1] - x_launch[j])
        dx_obs.append(x_full[i_full + h] - x_launch[j])

    if len(dx_pred) < 10:
        return np.nan, len(dx_pred)
    dp = np.array(dx_pred)
    do = np.array(dx_obs)
    ss_res = np.sum((do - dp)**2)
    ss_tot = np.sum((do - np.mean(do))**2)
    return float(1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0), len(dp)


def mae_at_horizon(paths_mode, x_full, idx_full, h, N_full, mask=None):
    """Compute MAE at horizon h from precomputed paths."""
    N = len(idx_full)
    if mask is None:
        mask = np.ones(N, dtype=bool)
    errs = []
    for j in range(N):
        if not mask[j]:
            continue
        i_full = idx_full[j]
        if i_full + h >= N_full:
            continue
        if h - 1 >= paths_mode.shape[1] or np.isnan(paths_mode[j, h - 1]):
            continue
        errs.append(abs(x_full[i_full + h] - paths_mode[j, h - 1]))
    return (float(np.mean(errs)), len(errs)) if errs else (np.nan, 0)


def dxr2_by_horizon(paths_mode, x_launch, x_full, idx_full, horizons,
                    N_full, mask=None):
    """DxR2 for a list of horizons. Returns list of (r2, n) tuples."""
    return [dxr2_at_horizon(paths_mode, x_launch, x_full, idx_full, h,
                            N_full, mask) for h in horizons]


def mae_by_horizon(paths_mode, x_full, idx_full, horizons, N_full,
                   mask=None):
    """MAE for a list of horizons. Returns list of (mae, n) tuples."""
    return [mae_at_horizon(paths_mode, x_full, idx_full, h, N_full, mask)
            for h in horizons]


# ============================================================
#  Event detection
# ============================================================

def detect_events_from_x(x, t=None):
    """Detect A/B transitions via KMeans + hysteresis.
    Returns dict with states, event_indices, thresholds, etc."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(x.reshape(-1, 1))
    centroids = sorted(km.cluster_centers_.flatten())
    midpoint = (centroids[0] + centroids[1]) / 2.0
    hyst_band = 0.1 * (centroids[1] - centroids[0])
    thresh_up = midpoint + hyst_band
    thresh_down = midpoint - hyst_band

    N = len(x)
    states = np.zeros(N, dtype=int)
    states[0] = 0 if x[0] < midpoint else 1
    for i in range(1, N):
        if states[i-1] == 0:
            states[i] = 1 if x[i] > thresh_up else 0
        else:
            states[i] = 0 if x[i] < thresh_down else 1

    event_indices = [i for i in range(1, N) if states[i] != states[i-1]]
    event_times = [float(t[i]) for i in event_indices] if t is not None else None

    return {
        'states': states,
        'event_indices': event_indices,
        'event_times': event_times,
        'centroids': centroids,
        'midpoint': midpoint,
        'thresh_up': thresh_up,
        'thresh_down': thresh_down,
    }


# ============================================================
#  Statistics
# ============================================================

def compute_acf(e, max_lag=50):
    e_c = e - np.mean(e)
    var = np.var(e)
    n = len(e)
    max_lag = min(max_lag, n - 1)  # can't compute lag >= n
    if var < 1e-15 or n < 2:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(e_c[:n-l] * e_c[l:]) / (n * var) if l > 0
                     else 1.0 for l in range(max_lag + 1)])


def ljung_box(acf_vals, n, lags=None):
    from scipy import stats as sp_stats
    if lags is None:
        lags = [5, 10, 20, 50]
    results = []
    for m in lags:
        if m >= n or m >= len(acf_vals):
            continue
        Q = n * (n + 2) * np.sum(
            acf_vals[1:m+1]**2 / (n - np.arange(1, m+1)))
        p = 1.0 - sp_stats.chi2.cdf(Q, df=m)
        results.append({'lag': m, 'Q': float(Q), 'p': float(p)})
    return results


def compute_ccf(x, y, max_lag=50):
    """Cross-correlation of x and y for lags in [-max_lag, +max_lag].
    CCF[lag] = corr(x_t, y_{t-lag}).
    Positive lag = y leads x. Negative lag = x leads y."""
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-15 or sy < 1e-15:
        return np.zeros(2 * max_lag + 1), np.arange(-max_lag, max_lag + 1)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            ccf[i] = np.sum(x_c[lag:] * y_c[:n-lag]) / (n * sx * sy)
        else:
            ccf[i] = np.sum(x_c[:n+lag] * y_c[-lag:]) / (n * sx * sy)
    return ccf, lags


# ============================================================
#  Covariance propagation (for UQ diagnostics)
# ============================================================

def propagate_cov_trajectory(P0, max_h, params, cl_params, dt=0.1):
    """Propagate open-loop prediction covariance for h=1..max_h.
    Returns array of P_xx[h] (observation prediction variance at each h).
    Since F,Q are constant (dt fixed), this is independent of state/mode."""
    alpha = params['alpha']; kap = params['kappa']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    rho = math.exp(-alpha * dt)
    F = np.array([[1, dt], [-kap*dt, rho]])
    Q = np.diag([q_sc*params['qx']*dt, q_sc*params['qu']*dt])

    P = P0.copy()
    Pxx = np.zeros(max_h)
    for h in range(max_h):
        P = F @ P @ F.T + Q
        Pxx[h] = P[0, 0] + R  # observation prediction variance
    return Pxx


# ============================================================
#  Plotting + output helpers
# ============================================================

def setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 9, 'font.family': 'serif',
        'axes.grid': True, 'grid.alpha': 0.3,
    })
    return plt


def ensure_output_dir(subdir=None):
    d = OUT_ROOT
    if subdir:
        d = d / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


MODE_COLORS = {'oracle': 'red', 'persistence': 'green', 'no_forcing': 'blue'}
MODE_LABELS = {'oracle': 'Oracle v', 'persistence': 'Persistence v',
               'no_forcing': 'No forcing'}
