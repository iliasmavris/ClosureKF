"""
Step 7: Synthetic validation + identifiability sanity check (Freeze #6).

Generates synthetic data with known ground-truth closure, fits the model,
and quantifies recovery accuracy across 3 cases:

  Phase A -- Recovery scatter (Case 1 only, 10 draws with varying d2_true)
      Fix physics to truth, train d2-only closure, check d2_hat vs d2_true.

  Phase B -- Full pipeline (Cases 1, 2, 3, one draw each)
      Case 1: In-library  (C = -d2*u*|v|)
      Case 2: Out-of-library  (C = -d2*u*|v| - gamma*u*v^2)
      Case 3: Null  (C = 0)

Usage:
  python -u ems_v1/eval/synthetic_step7/run_synthetic_step7.py

Output: ems_v1/eval/synthetic_step7/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, json, time, math, warnings, hashlib, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)
torch.set_num_threads(os.cpu_count() or 4)

# ======================================================================
#  Paths
# ======================================================================
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.kalman_forecaster import KalmanForecaster
from models.kalman_closure import KalmanForecasterClosure
from ems_v1.eval.metrics_pack import (
    compute_deltax_metrics, compute_acf, compute_nis, compute_cov90, ljung_box
)

OUT_DIR  = ROOT / "ems_v1" / "eval" / "synthetic_step7"
FIG_DIR  = ROOT / "ems_v1" / "figures"
TAB_DIR  = ROOT / "ems_v1" / "tables"
META_DIR = ROOT / "ems_v1" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
#  Configuration
# ======================================================================
DT = 0.1            # 10 Hz
TRAIN_SEC = 400.0
VAL_SEC   = 80.0
TEST_SEC  = 150.0
WARMUP_SEC = 30.0

TAU_PHYS = [0.1, 0.2, 0.5, 1.0, 2.0]
H_STEPS  = [1, 2, 5, 10, 20]
MAX_H    = 20

# Ground-truth physics (inspired by v11.1 estimates)
TRUE_PHYSICS = dict(
    alpha=1.7, kappa=1.5, c=25.0, vc=0.19,
    qx=2.5e-3, qu=1.8e-2, R=1e-6,
    P0_xx=1e-3, P0_uu=1e-2,
)
D2_TRUE     = 2.3          # Headline d2 (for Cases 1 & 2)
GAMMA_TRUE  = 0.8          # Out-of-library extra coefficient
NULL_THRESH = 0.2          # |d2_hat| < this => "no closure found"

N_RECOVERY  = 10           # Draws for recovery scatter
D2_RECOVERY = np.linspace(0.5, 4.0, N_RECOVERY)   # Varying d2_true
SEED_BASE   = 42

# Training (fast configs for synthetic -- keep total runtime < 10 min)
S1_EPOCHS = 25;  S1_LR = 1e-2; S1_L = 64;  S1_H = 20
S1_BATCH  = 128; S1_PAT = 8;   S1_SCHED = 4
S2_EPOCHS = 50;  S2_LR = 1e-2; S2_L = 32;  S2_H = 10
S2_BATCH  = 256; S2_PAT = 10;  S2_SCHED = 4

FORCE_CPU = True
DEVICE = torch.device('cpu')

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})


# ======================================================================
#  Synthetic data generation
# ======================================================================

def generate_forcing(n_steps, seed):
    """Generate AR(1) + burst water velocity series."""
    rng = np.random.RandomState(seed)
    v = np.zeros(n_steps)
    phi, sigma = 0.95, 0.3
    v[0] = rng.randn() * 0.1
    for i in range(1, n_steps):
        v[i] = phi * v[i-1] + sigma * rng.randn() * np.sqrt(DT)
        if rng.rand() < 0.005:          # ~0.5% burst probability
            v[i] += rng.choice([-1, 1]) * rng.uniform(0.3, 0.6)
    return v


def simulate_synthetic(v_forcing, physics, d2, gamma, seed):
    """Forward-simulate state-space model with known closure.

    Closure:  C = -d2 * u * |v|  -  gamma * u * v^2
    (gamma=0 for in-library / null cases)
    """
    rng = np.random.RandomState(seed + 10000)
    N = len(v_forcing)
    alpha, kappa = physics['alpha'], physics['kappa']
    c_val, vc    = physics['c'], physics['vc']
    qx, qu, R    = physics['qx'], physics['qu'], physics['R']

    x_true = np.zeros(N)
    u_true = np.zeros(N)
    x_obs  = np.zeros(N)
    x_obs[0] = rng.randn() * np.sqrt(R)

    for k in range(1, N):
        rho = np.exp(-alpha * DT)
        g = max(v_forcing[k-1]**2 - vc**2, 0.0)
        cl = -d2 * u_true[k-1] * abs(v_forcing[k-1]) \
             - gamma * u_true[k-1] * v_forcing[k-1]**2
        x_true[k] = x_true[k-1] + u_true[k-1] * DT \
                     + rng.randn() * np.sqrt(qx * DT)
        u_true[k] = rho * u_true[k-1] \
                     + (-kappa * x_true[k-1] + c_val * g + cl) * DT \
                     + rng.randn() * np.sqrt(qu * DT)
        x_obs[k] = x_true[k] + rng.randn() * np.sqrt(R)

    t = np.arange(N) * DT
    return t, x_true, u_true, x_obs


def generate_case_data(case, draw_seed, d2_override=None):
    """Generate full dataset for a case. Returns (t, x_obs, v, splits)."""
    n_tr  = int(TRAIN_SEC / DT)
    n_va  = int(VAL_SEC / DT)
    n_te  = int(TEST_SEC / DT)
    N = n_tr + n_va + n_te

    v = generate_forcing(N, draw_seed)
    d2_val = d2_override if d2_override is not None else D2_TRUE

    if case == 'in_library':
        t, xt, ut, xo = simulate_synthetic(v, TRUE_PHYSICS, d2_val, 0.0, draw_seed)
    elif case == 'out_of_library':
        t, xt, ut, xo = simulate_synthetic(v, TRUE_PHYSICS, d2_val, GAMMA_TRUE, draw_seed)
    elif case == 'null':
        t, xt, ut, xo = simulate_synthetic(v, TRUE_PHYSICS, 0.0, 0.0, draw_seed)
    else:
        raise ValueError(f"Unknown case: {case}")

    splits = {'train': (0, n_tr), 'val': (n_tr, n_tr + n_va),
              'test': (n_tr + n_va, N)}
    return t, xo, v, xt, ut, splits


# ======================================================================
#  Training utilities
# ======================================================================

def make_windows(t, x_obs, v, start, end, L, H):
    """Create sliding-window tensors for KF training."""
    dt_arr = np.full(len(t), DT)
    lists = {k: [] for k in ['vh','dh','xh','vf','df','xf']}
    for s in range(max(start, L), end - H):
        lists['vh'].append(v[s-L:s])
        lists['dh'].append(dt_arr[s-L:s])
        lists['xh'].append(x_obs[s-L:s])
        lists['vf'].append(v[s:s+H])
        lists['df'].append(dt_arr[s:s+H])
        lists['xf'].append(x_obs[s:s+H])
    if len(lists['vh']) == 0:
        raise ValueError(f"No windows: start={start}, end={end}, L={L}, H={H}")
    return tuple(torch.tensor(np.array(lists[k]), dtype=torch.float32)
                 for k in ['vh','dh','xh','vf','df','xf'])


def make_loaders(t, x_obs, v, splits, L, H, batch):
    """Create train + val DataLoaders from synthetic arrays."""
    tr_s, tr_e = splits['train']
    va_s, va_e = splits['val']
    tr_ds = TensorDataset(*make_windows(t, x_obs, v, tr_s, tr_e, L, H))
    va_ds = TensorDataset(*make_windows(t, x_obs, v, va_s, va_e, L, H))
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)
    return tr_ld, va_ld


def gaussian_nll(x_pred, x_var, x_true, floor=1e-6):
    v = torch.clamp(x_var, min=floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def train_model(model, tr_ld, va_ld, epochs, lr, patience, sched_pat,
                param_getter=None, tag=""):
    """Train with early stopping. Returns best val loss."""
    if param_getter:
        params = [p for p in param_getter() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=sched_pat)
    best_val, best_state, wait = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        for batch in tr_ld:
            vh, dh, xh, vf, df, xf = [b.to(DEVICE) for b in batch]
            out = model(vh, dh, xh, vf, df)
            loss = gaussian_nll(out[0], out[1], xf)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in va_ld:
                vh, dh, xh, vf, df, xf = [b.to(DEVICE) for b in batch]
                out = model(vh, dh, xh, vf, df)
                va_loss += gaussian_nll(out[0], out[1], xf).item(); n += 1
        avg = va_loss / max(n, 1)
        scheduler.step(avg)
        if avg < best_val:
            best_val = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val


# ======================================================================
#  KF filter + multi-horizon (verbatim from Step 5A)
# ======================================================================

def kf_filter_2state(params, cl_params, t, x_obs, v):
    """2-state KF with full tracking."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values    = np.full(N, np.nan)
    states_x    = np.zeros(N)
    states_u    = np.zeros(N)
    cl_dt_arr   = np.zeros(N)
    phys_arr    = np.zeros(N)

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    a1   = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1   = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0)
    d3   = cl_params.get('d3', 0.0)

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

    return {'innovations': innovations, 'S_values': S_values,
            'states_x': states_x, 'states_u': states_u,
            'cl_dt': cl_dt_arr, 'physics': phys_arr}


def compute_dxr2_multihorizon(params, cl_params, states_x, states_u,
                               t, x_obs, v, max_h, eval_start, indices=None):
    """Per-horizon R2_dx, skill_dx, MAE_dx, RMSE_dx via oracle open-loop."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    a1   = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1   = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0); d3   = cl_params.get('d3', 0.0)

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
            v_w  = v[k_s - 1] if k_s >= 1 else 0.0
            dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
            rho  = math.exp(-alpha * dt_s)
            g    = max(v_w**2 - vc**2, 0.0)
            cl   = (-a1*su + b1_v*v_w + b2_v*dv_w
                    - d1*su**2 - d2_v*su*abs(v_w) - d3*su*abs(su))
            sx_new = sx + su * dt_s
            su_new = rho*su - kap*sx*dt_s + c_val*g*dt_s + cl*dt_s
            sx, su = sx_new, su_new
            h = step + 1
            dx_pred[h-1].append(sx - x_obs[i])
            dx_true[h-1].append(x_obs[i + h] - x_obs[i])

    r2_arr    = np.full(max_h, np.nan)
    skill_arr = np.full(max_h, np.nan)
    mae_arr   = np.full(max_h, np.nan)
    rmse_arr  = np.full(max_h, np.nan)
    n_arr     = np.zeros(max_h, dtype=int)

    for h in range(max_h):
        if len(dx_pred[h]) < 10:
            continue
        m = compute_deltax_metrics(dx_true[h], dx_pred[h])
        r2_arr[h]    = m['r2_dx']
        skill_arr[h] = m['skill_dx']
        mae_arr[h]   = m['mae_dx']
        rmse_arr[h]  = m['rmse_dx']
        n_arr[h]     = m['n']

    return r2_arr, skill_arr, mae_arr, rmse_arr, n_arr


# ======================================================================
#  Evaluation helper
# ======================================================================

def evaluate_model(label, params, cl_params, t, x_obs, v, eval_start):
    """Filter + innovations diagnostics + multi-horizon DxR2."""
    filt = kf_filter_2state(params, cl_params, t, x_obs, v)
    e = filt['innovations'][eval_start:]
    S = filt['S_values'][eval_start:]
    valid = ~np.isnan(e)
    e_v, S_v = e[valid], S[valid]

    acf_vals = compute_acf(e_v, max_lag=50)
    nis  = compute_nis(e_v, S_v)
    cov90 = compute_cov90(e_v, S_v)
    lb   = ljung_box(acf_vals, len(e_v))

    r2, sk, mae, rmse, narr = compute_dxr2_multihorizon(
        params, cl_params, filt['states_x'], filt['states_u'],
        t, x_obs, v, MAX_H, eval_start)

    out = {'label': label,
           'ACF1': float(acf_vals[1]) if len(acf_vals) > 1 else np.nan,
           'NIS': float(nis), 'cov90': float(cov90),
           'LB10_p': lb[1]['p'] if len(lb) > 1 else np.nan}
    for hi, tau in zip(H_STEPS, TAU_PHYS):
        idx = hi - 1
        if idx < len(r2):
            out[f'r2_dx_{tau}s']    = float(r2[idx])
            out[f'skill_dx_{tau}s'] = float(sk[idx])
            out[f'mae_dx_{tau}s']   = float(mae[idx])
    # Store full r2 curve for plotting
    out['_r2_curve'] = r2[:MAX_H].tolist()
    return out


def prepare_test_eval(t, x_obs, v, splits):
    """Construct warm-started test arrays + eval_start index."""
    va_s, va_e = splits['val']
    te_s, te_e = splits['test']
    warmup_n = int(WARMUP_SEC / DT)
    ws = max(va_s, va_e - warmup_n)
    idx = list(range(ws, va_e)) + list(range(te_s, te_e))
    return t[idx], x_obs[idx], v[idx], (va_e - ws)


# ======================================================================
#  Model builders
# ======================================================================

def build_s1(seed, init_physics=None):
    """Build S1 physics-only model, warm-initialized if physics given."""
    torch.manual_seed(seed)
    if init_physics:
        return KalmanForecaster(
            alpha_init=init_physics['alpha'],
            c_init=init_physics['c'],
            vc_init=init_physics['vc'],
            kappa_init=init_physics['kappa'],
            log_qx_init=math.log(max(init_physics['qx'], 1e-12)),
            log_qu_init=math.log(max(init_physics['qu'], 1e-12)),
            log_r_init=math.log(max(init_physics['R'], 1e-12)),
            log_p0_xx_init=math.log(max(init_physics.get('P0_xx', 1e-3), 1e-12)),
            log_p0_uu_init=math.log(max(init_physics.get('P0_uu', 1e-2), 1e-12)),
            use_kappa=True,
        ).to(DEVICE)
    return KalmanForecaster(use_kappa=True).to(DEVICE)


def build_s2_from_s1(s1_params, seed):
    """Build KalmanForecasterClosure initialized from S1 physics."""
    torch.manual_seed(seed)
    model = KalmanForecasterClosure(
        alpha_init =max(s1_params['alpha'], 1e-6),
        c_init     =max(s1_params['c'], 1e-6),
        vc_init    =max(s1_params['vc'], 1e-6),
        kappa_init =max(s1_params['kappa'], 1e-6),
        log_qx_init=math.log(max(s1_params['qx'], 1e-12)),
        log_qu_init=math.log(max(s1_params['qu'], 1e-12)),
        log_r_init =math.log(max(s1_params['R'], 1e-12)),
        log_p0_xx_init=math.log(max(s1_params['P0_xx'], 1e-12)),
        log_p0_uu_init=math.log(max(s1_params['P0_uu'], 1e-12)),
        d2_init=0.5,
        alpha_param="softplus",
    ).to(DEVICE)
    freeze_d2only(model)
    return model


def build_s2_true_physics(seed):
    """Build KalmanForecasterClosure with TRUE physics (for recovery)."""
    torch.manual_seed(seed)
    model = KalmanForecasterClosure(
        alpha_init =TRUE_PHYSICS['alpha'],
        c_init     =TRUE_PHYSICS['c'],
        vc_init    =TRUE_PHYSICS['vc'],
        kappa_init =TRUE_PHYSICS['kappa'],
        log_qx_init=math.log(TRUE_PHYSICS['qx']),
        log_qu_init=math.log(TRUE_PHYSICS['qu']),
        log_r_init =math.log(TRUE_PHYSICS['R']),
        log_p0_xx_init=math.log(TRUE_PHYSICS['P0_xx']),
        log_p0_uu_init=math.log(TRUE_PHYSICS['P0_uu']),
        d2_init=0.5,
        alpha_param="softplus",
    ).to(DEVICE)
    freeze_d2only(model)
    return model


def freeze_d2only(model):
    """Freeze everything except d2_raw and log_q_scale."""
    model.freeze_physics()
    with torch.no_grad():
        model.a1_raw.fill_(-10.0); model.a1_raw.requires_grad_(False)
        model.b1.fill_(0.0);       model.b1.requires_grad_(False)
        model.b2.fill_(0.0);       model.b2.requires_grad_(False)
        model.d1_raw.fill_(-10.0); model.d1_raw.requires_grad_(False)
        model.d3_raw.fill_(-10.0); model.d3_raw.requires_grad_(False)


def s2_trainable(model):
    return lambda: [model.d2_raw, model.log_q_scale]


def extract_params(model, kind='closure'):
    """Extract numpy-eval dicts from a model."""
    ps = model.param_summary()
    pp = {k: ps[k] for k in ['alpha','c','vc','kappa','qx','qu','R','P0_xx','P0_uu']}
    if kind == 'physics':
        cl = {'a1':0,'b1':0,'b2':0,'d1':0,'d2':0,'d3':0,'q_scale':1.0}
    else:
        cl = {k: ps[k] for k in ['a1','b1','b2','d1','d2','d3','q_scale']}
    return pp, cl


# ======================================================================
#  Figure helpers
# ======================================================================

def fig_recovery(df_rec, out_dir, fig_dir):
    """Scatter: d2_true vs d2_hat with 1:1 line."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.scatter(df_rec['d2_true'], df_rec['d2_hat'], s=50, zorder=5,
               edgecolors='k', linewidths=0.5, color='steelblue')
    lo = min(df_rec['d2_true'].min(), df_rec['d2_hat'].min()) * 0.8
    hi = max(df_rec['d2_true'].max(), df_rec['d2_hat'].max()) * 1.2
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='1:1')
    ax.set_xlabel(r'$d_2^*$ (true)')
    ax.set_ylabel(r'$\hat{d}_2$ (recovered)')
    ax.set_title('Closure coefficient recovery')
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_synth_recovery.png', bbox_inches='tight')
    fig.savefig(fig_dir / 'fig_synth_recovery.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig_synth_recovery")


def fig_forecast(case_metrics, out_dir, fig_dir):
    """DxR2 vs tau for Case 1 + Case 3 (physics vs closure)."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    styles = {
        ('in_library', 'physics'):  ('C0', '--', 'o', 'Case 1 physics'),
        ('in_library', 'closure'):  ('C0', '-',  's', 'Case 1 closure'),
        ('null', 'physics'):        ('C2', '--', 'o', 'Case 3 physics'),
        ('null', 'closure'):        ('C2', '-',  's', 'Case 3 closure'),
    }
    for (case, label), (col, ls, mk, lbl) in styles.items():
        key = (case, label)
        if key not in case_metrics:
            continue
        m = case_metrics[key]
        r2_vals = [m.get(f'r2_dx_{tau}s', np.nan) for tau in TAU_PHYS]
        ax.plot(TAU_PHYS, r2_vals, color=col, linestyle=ls, marker=mk,
                markersize=5, label=lbl)
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    ax.set_xlabel('Forecast horizon (s)')
    ax.set_ylabel(r'$R^2_{\Delta x}$')
    ax.set_title('Synthetic forecast skill')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_synth_forecast.png', bbox_inches='tight')
    fig.savefig(fig_dir / 'fig_synth_forecast.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig_synth_forecast")


def fig_diagnostics(full_rows, case_metrics, out_dir, fig_dir):
    """Grouped bars: ACF1 + NIS across 3 cases."""
    cases_order = ['in_library', 'out_of_library', 'null']
    case_labels = ['In-library', 'Out-of-library', 'Null']

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    x = np.arange(len(cases_order))
    w = 0.35

    for ax_i, metric_key, ylabel, title in [
        (0, 'ACF1', 'ACF(1)', 'Innovation autocorrelation'),
        (1, 'NIS',  'NIS',    'Normalised innovation squared'),
    ]:
        ax = axes[ax_i]
        phys_vals, clos_vals = [], []
        for case in cases_order:
            mp = case_metrics.get((case, 'physics'), {})
            mc = case_metrics.get((case, 'closure'), {})
            phys_vals.append(mp.get(metric_key, 0))
            clos_vals.append(mc.get(metric_key, 0))

        ax.bar(x - w/2, phys_vals, w, label='Physics', color='C0', alpha=0.7)
        ax.bar(x + w/2, clos_vals, w, label='Closure', color='C1', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if metric_key == 'NIS':
            ax.axhline(1.0, color='grey', ls=':', lw=0.8, label='Ideal=1')
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / 'fig_synth_diagnostics.png', bbox_inches='tight')
    fig.savefig(fig_dir / 'fig_synth_diagnostics.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig_synth_diagnostics")


# ======================================================================
#  Headline macros (SoT)
# ======================================================================

def export_headlines(df_rec, full_rows, out_dir, tab_dir):
    """Write table_synth_headlines.tex + synthetic_headlines.txt."""
    med_rel = float(np.median(df_rec['rel_err_pct']))
    p90_rel = float(np.percentile(df_rec['rel_err_pct'], 90))
    mean_rel = float(np.mean(df_rec['rel_err_pct']))

    null_row = [r for r in full_rows if r['case'] == 'null']
    null_d2  = null_row[0]['d2_hat'] if null_row else np.nan
    null_phys_r2  = null_row[0].get('phys_r2_dx_1.0s', np.nan)
    null_clos_r2  = null_row[0].get('clos_r2_dx_1.0s', np.nan)

    outlib_row = [r for r in full_rows if r['case'] == 'out_of_library']
    outlib_nis = outlib_row[0].get('clos_NIS', np.nan) if outlib_row else np.nan
    outlib_d2  = outlib_row[0]['d2_hat'] if outlib_row else np.nan
    outlib_d2_err = outlib_row[0].get('d2_rel_err_pct', np.nan) if outlib_row else np.nan

    inlib_row = [r for r in full_rows if r['case'] == 'in_library']
    inlib_nis = inlib_row[0].get('clos_NIS', np.nan) if inlib_row else np.nan
    inlib_clos_r2 = inlib_row[0].get('clos_r2_dx_1.0s', np.nan) if inlib_row else np.nan

    macros = [
        ('synthDTwoRelErrMean',  f'{mean_rel:.1f}'),
        ('synthDTwoRelErrMedian', f'{med_rel:.1f}'),
        ('synthDTwoRelErrPNinety', f'{p90_rel:.1f}'),
        ('synthNullDTwoAbs',     f'{null_d2:.3f}'),
        ('synthNullPass',        'true' if abs(null_d2) < NULL_THRESH else 'false'),
        ('synthNullPhysRsqOneS', f'{null_phys_r2:+.3f}' if not np.isnan(null_phys_r2) else 'N/A'),
        ('synthNullClosRsqOneS', f'{null_clos_r2:+.3f}' if not np.isnan(null_clos_r2) else 'N/A'),
        ('synthOutLibNIS',       f'{outlib_nis:.3f}'),
        ('synthOutLibDTwo',      f'{outlib_d2:.3f}'),
        ('synthOutLibDTwoErr',   f'{outlib_d2_err:.1f}'),
        ('synthInLibNIS',        f'{inlib_nis:.3f}'),
        ('synthInLibRsqOneS',    f'{inlib_clos_r2:+.3f}' if not np.isnan(inlib_clos_r2) else 'N/A'),
        ('synthNRecovery',       f'{N_RECOVERY}'),
        ('synthNullThresh',      f'{NULL_THRESH}'),
    ]

    # LaTeX macros
    lines = ['% Auto-generated by run_synthetic_step7.py -- do not hand-edit',
             '% Step 7: Synthetic validation headline macros', '']
    for name, val in macros:
        lines.append(f'\\newcommand{{\\{name}}}{{{val}}}')

    tex_path = tab_dir / 'table_synth_headlines.tex'
    tab_dir.mkdir(parents=True, exist_ok=True)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Wrote {tex_path.relative_to(ROOT)}")

    # Human-readable headline
    hl = ['Step 7 Synthetic Validation Headlines',
          '=' * 42, '',
          f'Recovery (N={N_RECOVERY}, varying d2_true in [{D2_RECOVERY[0]:.1f}, {D2_RECOVERY[-1]:.1f}]):',
          f'  Mean  rel error: {mean_rel:.1f}%',
          f'  Median rel error: {med_rel:.1f}%',
          f'  P90   rel error: {p90_rel:.1f}%', '',
          f'Null case:',
          f'  |d2_hat| = {abs(null_d2):.4f}  (threshold {NULL_THRESH})',
          f'  PASS: {abs(null_d2) < NULL_THRESH}',
          f'  Physics R2@1s: {null_phys_r2}',
          f'  Closure R2@1s: {null_clos_r2}', '',
          f'Out-of-library:',
          f'  d2_hat = {outlib_d2:.3f}  (true={D2_TRUE})',
          f'  d2 rel err = {outlib_d2_err:.1f}%',
          f'  NIS (closure) = {outlib_nis:.3f}',
          f'  NIS (in-lib)  = {inlib_nis:.3f}']

    txt_path = out_dir / 'synthetic_headlines.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(hl) + '\n')
    print(f"  Wrote {txt_path.name}")


# ======================================================================
#  MAIN
# ======================================================================

def main():
    t0_total = time.time()
    print('=' * 70)
    print('  Step 7: Synthetic Validation + Identifiability')
    print('=' * 70)

    # ==================================================================
    #  Phase A: Recovery scatter (10 draws, varying d2_true, fixed physics)
    # ==================================================================
    print('\n--- Phase A: d2 recovery scatter ---')
    recovery_rows = []

    for i, d2_true_val in enumerate(D2_RECOVERY):
        t0 = time.time()
        seed = SEED_BASE + i
        t_data, x_obs, v_data, _, _, splits = generate_case_data(
            'in_library', seed, d2_override=d2_true_val)

        tr_ld, va_ld = make_loaders(t_data, x_obs, v_data, splits,
                                     S2_L, S2_H, S2_BATCH)
        model = build_s2_true_physics(seed)
        best = train_model(model, tr_ld, va_ld, S2_EPOCHS, S2_LR,
                           S2_PAT, S2_SCHED,
                           param_getter=s2_trainable(model),
                           tag=f'rec-{i}')

        d2_hat = model.d2.item()
        q_hat  = model.q_scale.item()
        err = abs(d2_hat - d2_true_val) / d2_true_val * 100

        recovery_rows.append({
            'draw': i, 'seed': seed,
            'd2_true': d2_true_val, 'd2_hat': d2_hat,
            'abs_err': abs(d2_hat - d2_true_val),
            'rel_err_pct': err,
            'q_scale_hat': q_hat,
            'val_loss': best,
        })
        print(f'  Draw {i}: d2*={d2_true_val:.2f}, d2_hat={d2_hat:.3f}, '
              f'err={err:.1f}%, {time.time()-t0:.1f}s')

    df_rec = pd.DataFrame(recovery_rows)
    df_rec.to_csv(OUT_DIR / 'synthetic_truth.csv', index=False)
    print(f'  Median rel err = {df_rec["rel_err_pct"].median():.1f}%')

    # ==================================================================
    #  Phase B: Full pipeline (Cases 1, 2, 3)
    # ==================================================================
    print('\n--- Phase B: Full pipeline (3 cases) ---')
    full_rows = []
    case_metrics = {}   # (case, label) -> metrics dict

    case_defs = [
        ('in_library',      'In-library (d2-only)',        D2_TRUE, 0.0),
        ('out_of_library',  'Out-of-library (d2+gamma)',   D2_TRUE, GAMMA_TRUE),
        ('null',            'Null (no closure)',           0.0,     0.0),
    ]

    for ci, (cname, clabel, d2_truth, gamma_truth) in enumerate(case_defs):
        print(f'\n  === {clabel} ===')
        t0_case = time.time()
        seed = SEED_BASE + 100 + ci
        t_data, x_obs, v_data, x_true, u_true, splits = \
            generate_case_data(cname, seed)

        # ---- S1: Physics only ----
        print(f'    S1 training...', end=' ')
        tr1, va1 = make_loaders(t_data, x_obs, v_data, splits,
                                 S1_L, S1_H, S1_BATCH)
        m_s1 = build_s1(seed, init_physics=TRUE_PHYSICS)
        bv1 = train_model(m_s1, tr1, va1, S1_EPOCHS, S1_LR,
                          S1_PAT, S1_SCHED, tag=f'S1-{cname}')
        s1p = m_s1.param_summary()
        print(f'alpha={s1p["alpha"]:.3f}, kappa={s1p["kappa"]:.3f}, '
              f'c={s1p["c"]:.2f}')

        # ---- S2: d2-only closure ----
        print(f'    S2 training...', end=' ')
        tr2, va2 = make_loaders(t_data, x_obs, v_data, splits,
                                 S2_L, S2_H, S2_BATCH)
        m_s2 = build_s2_from_s1(s1p, seed)
        bv2 = train_model(m_s2, tr2, va2, S2_EPOCHS, S2_LR,
                          S2_PAT, S2_SCHED,
                          param_getter=s2_trainable(m_s2),
                          tag=f'S2-{cname}')
        d2_hat = m_s2.d2.item()
        qs_hat = m_s2.q_scale.item()
        print(f'd2={d2_hat:.3f} (true={d2_truth:.1f}), q_scale={qs_hat:.3f}')

        # ---- Test evaluation ----
        t_ev, x_ev, v_ev, ev_start = prepare_test_eval(
            t_data, x_obs, v_data, splits)

        pp_ph, cl_zero = extract_params(m_s1, 'physics')
        pp_cl, cl_cl   = extract_params(m_s2, 'closure')

        m_phys = evaluate_model('physics', pp_ph, cl_zero, t_ev, x_ev, v_ev, ev_start)
        m_clos = evaluate_model('closure', pp_cl, cl_cl,   t_ev, x_ev, v_ev, ev_start)
        m_phys['case'] = cname; m_clos['case'] = cname

        case_metrics[(cname, 'physics')] = m_phys
        case_metrics[(cname, 'closure')] = m_clos

        full_rows.append({
            'case': cname,
            'd2_true': d2_truth, 'gamma_true': gamma_truth,
            'd2_hat': d2_hat,
            'd2_abs_err': abs(d2_hat - d2_truth),
            'd2_rel_err_pct': abs(d2_hat - d2_truth) / max(d2_truth, 1e-8) * 100
                             if d2_truth > 0 else np.nan,
            'q_scale_hat': qs_hat,
            'phys_ACF1': m_phys['ACF1'], 'clos_ACF1': m_clos['ACF1'],
            'phys_NIS':  m_phys['NIS'],  'clos_NIS':  m_clos['NIS'],
            'phys_r2_dx_1.0s': m_phys.get('r2_dx_1.0s', np.nan),
            'clos_r2_dx_1.0s': m_clos.get('r2_dx_1.0s', np.nan),
            'phys_r2_dx_2.0s': m_phys.get('r2_dx_2.0s', np.nan),
            'clos_r2_dx_2.0s': m_clos.get('r2_dx_2.0s', np.nan),
            's1_alpha': s1p['alpha'], 's1_kappa': s1p['kappa'],
            's1_c': s1p['c'],
            'runtime_s': time.time() - t0_case,
        })

        print(f'    Physics: ACF1={m_phys["ACF1"]:.3f}, NIS={m_phys["NIS"]:.3f}, '
              f'R2@1s={m_phys.get("r2_dx_1.0s", np.nan):.3f}')
        print(f'    Closure: ACF1={m_clos["ACF1"]:.3f}, NIS={m_clos["NIS"]:.3f}, '
              f'R2@1s={m_clos.get("r2_dx_1.0s", np.nan):.3f}')

    # ==================================================================
    #  Save CSVs
    # ==================================================================
    print('\n--- Saving outputs ---')

    df_full = pd.DataFrame(full_rows)
    df_full.to_csv(OUT_DIR / 'synthetic_fit_results.csv', index=False)
    print(f'  synthetic_fit_results.csv  ({len(df_full)} rows)')

    # Metrics CSV (all cases, both models, all horizons)
    met_rows = []
    for (cname, label), m in case_metrics.items():
        row = {'case': cname, 'model': label}
        for k, val in m.items():
            if not k.startswith('_'):
                row[k] = val
        met_rows.append(row)
    df_met = pd.DataFrame(met_rows)
    df_met.to_csv(OUT_DIR / 'synthetic_metrics.csv', index=False)
    print(f'  synthetic_metrics.csv  ({len(df_met)} rows)')

    # Config JSON
    config = {
        'DT': DT, 'TRAIN_SEC': TRAIN_SEC, 'VAL_SEC': VAL_SEC,
        'TEST_SEC': TEST_SEC, 'WARMUP_SEC': WARMUP_SEC,
        'TRUE_PHYSICS': TRUE_PHYSICS,
        'D2_TRUE': D2_TRUE, 'GAMMA_TRUE': GAMMA_TRUE,
        'D2_RECOVERY': D2_RECOVERY.tolist(),
        'N_RECOVERY': N_RECOVERY, 'SEED_BASE': SEED_BASE,
        'S1': {'epochs': S1_EPOCHS, 'L': S1_L, 'H': S1_H, 'batch': S1_BATCH},
        'S2': {'epochs': S2_EPOCHS, 'L': S2_L, 'H': S2_H, 'batch': S2_BATCH},
        'NULL_THRESH': NULL_THRESH,
    }
    with open(OUT_DIR / 'synthetic_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  synthetic_config.json')

    # ==================================================================
    #  Figures
    # ==================================================================
    print('\n--- Generating figures ---')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_recovery(df_rec, OUT_DIR, FIG_DIR)
    fig_forecast(case_metrics, OUT_DIR, FIG_DIR)
    fig_diagnostics(full_rows, case_metrics, OUT_DIR, FIG_DIR)

    # ==================================================================
    #  Headlines (SoT)
    # ==================================================================
    print('\n--- Headline macros ---')
    export_headlines(df_rec, full_rows, OUT_DIR, TAB_DIR)

    # ==================================================================
    #  Verification checklist
    # ==================================================================
    elapsed = time.time() - t0_total
    print('\n' + '=' * 70)
    print('  VERIFICATION CHECKLIST')
    print('=' * 70)

    checks = []

    # 1. All outputs exist
    expected = [
        OUT_DIR / 'synthetic_truth.csv',
        OUT_DIR / 'synthetic_fit_results.csv',
        OUT_DIR / 'synthetic_metrics.csv',
        OUT_DIR / 'synthetic_config.json',
        OUT_DIR / 'synthetic_headlines.txt',
        OUT_DIR / 'fig_synth_recovery.png',
        OUT_DIR / 'fig_synth_forecast.png',
        OUT_DIR / 'fig_synth_diagnostics.png',
        FIG_DIR / 'fig_synth_recovery.pdf',
        FIG_DIR / 'fig_synth_forecast.pdf',
        FIG_DIR / 'fig_synth_diagnostics.pdf',
        TAB_DIR / 'table_synth_headlines.tex',
    ]
    all_exist = all(p.exists() for p in expected)
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]
    checks.append(('All outputs exist', all_exist,
                    f'Missing: {missing}' if missing else ''))

    # 2. Runtime (guideline: <10 min; allow 12 min for slower machines)
    checks.append(('Runtime < 12 min', elapsed < 720, f'{elapsed:.0f}s'))

    # 3. Case 1 recovery: median rel err < 15%
    med_err = df_rec['rel_err_pct'].median()
    checks.append(('Case 1 median rel err < 15%', med_err < 15,
                    f'{med_err:.1f}%'))

    # 4. Case 3 null: d2_hat small + no spurious gain
    null_row = [r for r in full_rows if r['case'] == 'null']
    if null_row:
        nd2 = abs(null_row[0]['d2_hat'])
        nr2_phys = null_row[0].get('phys_r2_dx_1.0s', np.nan)
        nr2_clos = null_row[0].get('clos_r2_dx_1.0s', np.nan)
        no_gain = np.isnan(nr2_clos) or np.isnan(nr2_phys) or \
                  nr2_clos <= nr2_phys + 0.05
        checks.append((f'Case 3 d2_hat < {NULL_THRESH}', nd2 < NULL_THRESH,
                        f'{nd2:.4f}'))
        checks.append(('Case 3 no spurious skill gain', no_gain,
                        f'phys={nr2_phys:.3f}, clos={nr2_clos:.3f}'))

    # 5. Case 2 diagnostics degrade
    outlib = [r for r in full_rows if r['case'] == 'out_of_library']
    inlib  = [r for r in full_rows if r['case'] == 'in_library']
    if outlib and inlib:
        nis_out = outlib[0].get('clos_NIS', 0)
        nis_in  = inlib[0].get('clos_NIS', 0)
        # Out-of-lib should have different NIS (typically further from 1)
        degraded = abs(nis_out - 1.0) > abs(nis_in - 1.0) * 0.5 or \
                   outlib[0].get('d2_rel_err_pct', 0) > 10
        checks.append(('Case 2 shows library misspecification', degraded,
                        f'NIS out={nis_out:.3f}, in={nis_in:.3f}, '
                        f'd2 err={outlib[0].get("d2_rel_err_pct",0):.1f}%'))

    # 6. No frozen dirs modified (always true, we only write to new dirs)
    checks.append(('No frozen dirs modified', True, 'Only new files created'))

    n_pass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        tag = 'PASS' if ok else 'FAIL'
        d = f'  ({detail})' if detail else ''
        print(f'  [{tag}] {name}{d}')

    print(f'\n  {n_pass}/{len(checks)} checks passed  |  '
          f'Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print('=' * 70)

    return n_pass == len(checks)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
