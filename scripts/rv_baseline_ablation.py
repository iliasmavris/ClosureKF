"""
Relative-Velocity Baseline Ablation (Task A).

Tests whether the discovered -d2*v*|u| closure term is simply compensating
for the baseline's non-relative coupling, by adding a quadratic
relative-velocity drag delta*(v-u)*|v-u| to the physics baseline and
re-running the discovery protocol.

Pipeline:
  1. Train original physics-only baseline (S1-orig, no RV drag)  [reuse v11.1 if avail]
  2. Train RV physics baseline (S1-rv, with use_rv_drag=True)
  3. Run 6-term closure discovery on top of RV baseline (S2-scipy)
  4. Apply term selection (variance + delta-NLL gate)
  5. Compute DxR2 table for all three rows:
     - Original physics-only
     - RV physics-only
     - RV + discovered closure (if any)
  6. Save outputs to final_lockbox_rv_ablation/

This ablation specifically tests whether the selected -d2*v*|u| term is
structurally substituting for the cross-term implied by quadratic
relative-velocity interactions under predominantly unidirectional flow.

Usage:  python -u scripts/rv_baseline_ablation.py
Output: final_lockbox_rv_ablation/
"""

import os, sys, math, json, time, warnings, hashlib, copy
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_forecaster import KalmanForecaster

# ==============================================================================
#  CONFIG
# ==============================================================================
SEED = 1
FORCE_CPU = True
DT = 0.1
VAR_FLOOR = 1e-6
WARMUP_SEC = 50.0

# Stage 1
S1_L = 512; S1_H = 128; S1_BATCH = 64
S1_EPOCHS = 100; S1_LR = 1e-2; S1_PATIENCE = 20; S1_SCHED = 10

# Stage 2 (scipy on numpy KF) -- reuse exact same protocol as discovery pipeline
S2_MAXITER = 300
TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']
SELECTION_THRESHOLD = 0.05
NLL_DELTA_MIN = 0.001

# Evaluation
MAX_H = 100  # 10s at 10 Hz

# Paths
CLEAN_DIR = ROOT / "processed_data_10hz_clean_v1"
OUT = ROOT / "final_lockbox_rv_ablation"

# Try to load cached v11.1 seed-1 checkpoint for the original baseline
V111_CKPT = ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "checkpoints" / "stage1_physics_seed1.pth"

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})

from scipy.optimize import minimize


# ==============================================================================
#  HELPERS
# ==============================================================================

def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def gaussian_nll(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def compute_acf(e, max_lag=50):
    e_c = e - np.mean(e)
    var = np.var(e)
    n = len(e)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(e_c[:n-l] * e_c[l:]) / (n * var) if l > 0
                     else 1.0 for l in range(max_lag + 1)])


# ==============================================================================
#  NUMPY 2-STATE KF FILTER (with optional RV drag)
# ==============================================================================

def kf_filter_2state(params, cl_params, t, x_obs, v):
    """2-state KF with full tracking. Supports optional RV drag via params['delta']."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_x = np.zeros(N)
    states_u = np.zeros(N)

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    delta = params.get('delta', 0.0)  # RV drag coeff
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
        dt_k = t[k] - t[k-1]
        if dt_k <= 0: dt_k = 0.1
        rho = math.exp(-alpha * dt_k)
        g = max(v[k-1]**2 - vc**2, 0.0)
        v_w = v[k-1]; u_st = s[1]
        rel_v = v_w - u_st
        physics_drift = (rho * u_st - kap * s[0] * dt_k + c_val * g * dt_k
                         + delta * rel_v * abs(rel_v) * dt_k)

        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = (-a1*u_st + b1_v*v_w + b2_v*dv_w
              - d1*u_st**2 - d2_v*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_d = cl * dt_k

        x_p = s[0] + s[1] * dt_k
        u_p = physics_drift + cl_d
        s_pred = np.array([x_p, u_p])

        # Jacobian includes RV drag: d/du [delta*(v-u)*|v-u|] = -2*delta*|v-u|
        F11 = rho - 2.0 * delta * abs(rel_v) * dt_k
        F_mat = np.array([[1, dt_k], [-kap*dt_k, F11]])
        Q = np.diag([q_sc*qx*dt_k, q_sc*qu*dt_k])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        innovations[k] = innov; S_values[k] = S_val

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(2) - np.outer(K, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)
        states_x[k] = s[0]; states_u[k] = s[1]

    return {'innovations': innovations, 'S_values': S_values,
            'states_x': states_x, 'states_u': states_u}


def kf_nll_numpy(innov, S_vals):
    valid = ~np.isnan(innov) & (S_vals > 0)
    e = innov[valid]
    S = np.maximum(S_vals[valid], 1e-12)
    return float(0.5 * np.mean(np.log(2 * math.pi * S) + e**2 / S))


# ==============================================================================
#  DxR2 MULTI-HORIZON (with optional RV drag)
# ==============================================================================

def compute_dxr2(params, cl_params, states_x, states_u,
                 t, x_obs, v, max_h=100, eval_start=1):
    """DxR2(h) and MAE(h) for h=1..max_h, oracle future v."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    delta = params.get('delta', 0.0)
    a1 = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1 = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0); d3 = cl_params.get('d3', 0.0)

    dx_pred = [[] for _ in range(max_h)]
    dx_true = [[] for _ in range(max_h)]

    for i in range(max(eval_start, 1), N - 1):
        sx, su = states_x[i], states_u[i]
        max_steps = min(max_h, N - 1 - i)
        for step in range(max_steps):
            k_s = i + 1 + step
            dt_s = t[k_s] - t[k_s - 1] if k_s > 0 else 0.1
            if dt_s <= 0: dt_s = 0.1
            v_w = v[k_s - 1] if k_s >= 1 else 0.0
            dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
            rho = math.exp(-alpha * dt_s)
            g = max(v_w**2 - vc**2, 0.0)
            rel_v = v_w - su
            cl = (-a1*su + b1_v*v_w + b2_v*dv_w
                  - d1*su**2 - d2_v*su*abs(v_w) - d3*su*abs(su))
            sx_new = sx + su * dt_s
            su_new = (rho*su - kap*sx*dt_s + c_val*g*dt_s
                      + delta * rel_v * abs(rel_v) * dt_s + cl*dt_s)
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
#  TRAINING
# ==============================================================================

def train_model(model, train_loader, val_loader, device,
                max_epochs, patience, lr, sched_patience=10, tag=""):
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"    [{tag}] {n_params} trainable parameters")

    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=sched_patience)

    best_loss, best_state, best_ep, wait = float('inf'), None, 0, 0
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

        scheduler.step(val_nll)
        if val_nll < best_loss:
            best_loss = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1; wait = 0
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
    return best_loss, best_ep


# ==============================================================================
#  S2: SCIPY CLOSURE ON NUMPY KF (with RV drag support)
# ==============================================================================

def train_s2_scipy(s1_pp, t_train, x_train, v_train,
                   t_val, x_val, v_val, tag="S2"):
    """Optimize 7 closure params on top of the (possibly RV) physics baseline."""
    n_eval = [0]

    def objective(cl_vec):
        n_eval[0] += 1
        a1, b1, b2, d1, d2, d3, log_qs = cl_vec
        cl = {'a1': a1, 'b1': b1, 'b2': b2,
              'd1': d1, 'd2': d2, 'd3': d3,
              'q_scale': math.exp(np.clip(log_qs, -10, 10))}
        filt = kf_filter_2state(s1_pp, cl, t_train, x_train, v_train)
        nll = kf_nll_numpy(filt['innovations'], filt['S_values'])
        if not np.isfinite(nll):
            return 1e10
        return nll

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(0, None), (None, None), (None, None),
              (0, None), (0, None), (0, None),
              (-5, 5)]

    t0 = time.time()
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': S2_MAXITER, 'maxfun': 2000,
                            'ftol': 1e-10, 'gtol': 1e-6})

    a1, b1, b2, d1, d2, d3, log_qs = res.x
    cl_best = {'a1': float(a1), 'b1': float(b1), 'b2': float(b2),
               'd1': float(d1), 'd2': float(d2), 'd3': float(d3),
               'q_scale': float(math.exp(np.clip(log_qs, -10, 10)))}

    train_nll = res.fun
    filt_val = kf_filter_2state(s1_pp, cl_best, t_val, x_val, v_val)
    val_nll = kf_nll_numpy(filt_val['innovations'], filt_val['S_values'])

    elapsed = time.time() - t0
    print(f"    [{tag}] L-BFGS-B done: {n_eval[0]} fevals, "
          f"train_nll={train_nll:.5f}, val_nll={val_nll:.5f}, "
          f"converged={res.success} [{elapsed:.1f}s]")

    return cl_best, train_nll, val_nll, n_eval[0]


# ==============================================================================
#  TERM SELECTION (with delta-NLL gate)
# ==============================================================================

def select_terms(cl_params, s1_pp, t, x_obs, v):
    """Variance + delta-NLL gate selection, identical to discovery pipeline."""
    N = len(x_obs)
    filt = kf_filter_2state(s1_pp, cl_params, t, x_obs, v)
    su = filt['states_u']

    start = N // 2
    contribs = {tn: np.zeros(N - start) for tn in TERM_NAMES}

    for i, k in enumerate(range(start, N)):
        u_st = su[k-1]; v_w = v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        contribs['a1'][i] = -cl_params['a1'] * u_st
        contribs['d1'][i] = -cl_params['d1'] * u_st**2
        contribs['d2'][i] = -cl_params['d2'] * u_st * abs(v_w)
        contribs['d3'][i] = -cl_params['d3'] * u_st * abs(u_st)
        contribs['b1'][i] =  cl_params['b1'] * v_w
        contribs['b2'][i] =  cl_params['b2'] * dv_w

    variances = {tn: float(np.var(contribs[tn])) for tn in TERM_NAMES}
    total_var = sum(variances.values())
    if total_var < 1e-15:
        return [], variances, {}

    rel_var = {tn: variances[tn] / total_var for tn in TERM_NAMES}
    candidates = [tn for tn in TERM_NAMES if rel_var[tn] >= SELECTION_THRESHOLD]

    # Delta-NLL gate
    base_nll = kf_nll_numpy(filt['innovations'], filt['S_values'])
    selected = []
    for tn in candidates:
        cl_without = dict(cl_params)
        cl_without[tn] = 0.0
        filt_without = kf_filter_2state(s1_pp, cl_without, t, x_obs, v)
        nll_without = kf_nll_numpy(filt_without['innovations'],
                                   filt_without['S_values'])
        delta_nll = nll_without - base_nll
        if delta_nll >= NLL_DELTA_MIN:
            selected.append(tn)

    return selected, variances, rel_var


# ==============================================================================
#  EVALUATE MODEL (filter + DxR2 + ACF + NIS)
# ==============================================================================

def evaluate_model(label, params, cl_params, t, x_obs, v, eval_start, max_h):
    t0_ev = time.time()
    filt = kf_filter_2state(params, cl_params, t, x_obs, v)
    sx = filt['states_x']; su = filt['states_u']

    e = filt['innovations'][eval_start:]
    S_sc = filt['S_values'][eval_start:]
    valid = ~np.isnan(e)
    e_v = e[valid]; S_v = S_sc[valid]
    n_valid = len(e_v)

    acf = compute_acf(e_v, max_lag=50)
    nis = float(np.mean(e_v**2 / np.maximum(S_v, 1e-15)))
    z90 = 1.6449
    cov90 = float(np.mean(np.abs(e_v) <= z90 * np.sqrt(np.maximum(S_v, 1e-15))))

    dxr2, mae = compute_dxr2(params, cl_params, sx, su, t, x_obs, v, max_h, eval_start)

    elapsed = time.time() - t0_ev
    d10 = dxr2[9] if max_h >= 10 else np.nan
    d20 = dxr2[19] if max_h >= 20 else np.nan
    print(f"    [{label}] ACF(1)={acf[1]:.4f} NIS={nis:.4f} cov90={cov90:.3f} "
          f"DxR2@1s={d10:+.4f} DxR2@2s={d20:+.4f} [{elapsed:.0f}s]")

    return {
        'label': label,
        'acf1': float(acf[1]),
        'nis_mean': nis, 'cov90': cov90,
        'dxr2': dxr2.tolist(),
        'mae': mae.tolist(),
        'n_scored': n_valid,
        'states_x': sx, 'states_u': su,
    }


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    t0_all = time.time()
    device = torch.device('cpu')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RELATIVE-VELOCITY BASELINE ABLATION")
    print("=" * 70)
    print(f"Output -> {OUT}")

    # ==================================================================
    #  STEP 0: DATA
    # ==================================================================
    print_section("STEP 0: DATA")

    csv_paths = {
        'train': CLEAN_DIR / "train_10hz_ready.csv",
        'val':   CLEAN_DIR / "val_10hz_ready.csv",
        'test':  CLEAN_DIR / "test_10hz_ready.csv",
    }

    for name, p in csv_paths.items():
        assert p.exists(), f"Missing: {p}"
        print(f"  {name}: {md5_file(p)}")

    df_train = pd.read_csv(csv_paths['train'])
    df_val   = pd.read_csv(csv_paths['val'])
    df_test  = pd.read_csv(csv_paths['test'])

    for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        print(f"  {name}: {len(df)} pts "
              f"({df['timestamp'].iloc[0]:.1f}-{df['timestamp'].iloc[-1]:.1f}s)")

    # Numpy arrays for evaluation
    t_train = df_train['timestamp'].values.astype(np.float64)
    x_train = df_train['displacement'].values.astype(np.float64)
    v_train = df_train['velocity'].values.astype(np.float64)
    t_val = df_val['timestamp'].values.astype(np.float64)
    x_val = df_val['displacement'].values.astype(np.float64)
    v_val = df_val['velocity'].values.astype(np.float64)

    # Warm evaluation array (warmup from val + test)
    warmup_start = df_val['timestamp'].iloc[-1] - WARMUP_SEC
    df_warmup = df_val[df_val['timestamp'] >= warmup_start].copy()
    df_warm = pd.concat([df_warmup, df_test], ignore_index=True)
    t_warm = df_warm['timestamp'].values.astype(np.float64)
    x_warm = df_warm['displacement'].values.astype(np.float64)
    v_warm = df_warm['velocity'].values.astype(np.float64)
    test_start_time = df_test['timestamp'].iloc[0]
    score_mask = t_warm >= test_start_time
    eval_start_warm = int(np.argmax(score_mask))
    print(f"  Warmup: {eval_start_warm} warmup pts + {len(df_test)} test pts")

    # ==================================================================
    #  STEP 1: ORIGINAL PHYSICS BASELINE (reuse v11.1 checkpoint if avail)
    # ==================================================================
    print_section("STEP 1: ORIGINAL PHYSICS-ONLY BASELINE")

    if V111_CKPT.exists():
        print(f"  Loading cached v11.1 seed-1 checkpoint: {V111_CKPT}")
        ckpt = torch.load(V111_CKPT, map_location='cpu', weights_only=False)
        s1_orig_params = ckpt['params']
        print(f"  Original S1: alpha={s1_orig_params['alpha']:.4f} "
              f"kappa={s1_orig_params['kappa']:.4f} c={s1_orig_params['c']:.4f} "
              f"vc={s1_orig_params['vc']:.4f}")
    else:
        print("  No cached checkpoint, training from scratch...")
        torch.manual_seed(SEED); np.random.seed(SEED)

        train_ds = StateSpaceDataset(
            [str(csv_paths['train'])], L=S1_L, m=S1_L, H=S1_H,
            predict_deltas=False, normalize=False)
        val_ds = StateSpaceDataset(
            [str(csv_paths['val'])], L=S1_L, m=S1_L, H=S1_H,
            predict_deltas=False, normalize=False)
        train_ld = DataLoader(train_ds, batch_size=S1_BATCH, shuffle=True, num_workers=0)
        val_ld = DataLoader(val_ds, batch_size=S1_BATCH, shuffle=False, num_workers=0)

        model_orig = KalmanForecaster(use_kappa=True).to(device)
        train_model(model_orig, train_ld, val_ld, device,
                    S1_EPOCHS, S1_PATIENCE, S1_LR, S1_SCHED, tag="S1-orig")
        s1_orig_params = model_orig.param_summary()

    orig_pp = {
        'alpha': s1_orig_params['alpha'], 'c': s1_orig_params['c'],
        'vc': s1_orig_params['vc'], 'kappa': s1_orig_params['kappa'],
        'qx': s1_orig_params['qx'], 'qu': s1_orig_params['qu'],
        'R': s1_orig_params['R'],
        'P0_xx': s1_orig_params['P0_xx'], 'P0_uu': s1_orig_params['P0_uu'],
        'delta': 0.0,  # original has no RV drag
    }
    null_cl = {k: 0.0 for k in TERM_NAMES}
    null_cl['q_scale'] = 1.0

    # ==================================================================
    #  STEP 2: RV PHYSICS BASELINE
    # ==================================================================
    print_section("STEP 2: RV PHYSICS BASELINE (delta*(v-u)*|v-u|)")

    torch.manual_seed(SEED); np.random.seed(SEED)

    train_ds = StateSpaceDataset(
        [str(csv_paths['train'])], L=S1_L, m=S1_L, H=S1_H,
        predict_deltas=False, normalize=False)
    val_ds = StateSpaceDataset(
        [str(csv_paths['val'])], L=S1_L, m=S1_L, H=S1_H,
        predict_deltas=False, normalize=False)
    train_ld = DataLoader(train_ds, batch_size=S1_BATCH, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=S1_BATCH, shuffle=False, num_workers=0)

    model_rv = KalmanForecaster(use_kappa=True, use_rv_drag=True, delta_init=0.1).to(device)
    best_val_rv, best_ep_rv = train_model(
        model_rv, train_ld, val_ld, device,
        S1_EPOCHS, S1_PATIENCE, S1_LR, S1_SCHED, tag="S1-rv")

    s1_rv_params = model_rv.param_summary()
    print(f"  RV S1: alpha={s1_rv_params['alpha']:.4f} "
          f"kappa={s1_rv_params['kappa']:.4f} c={s1_rv_params['c']:.4f} "
          f"vc={s1_rv_params['vc']:.4f} delta={s1_rv_params['delta']:.4f}")

    torch.save({
        'state_dict': model_rv.state_dict(),
        'params': s1_rv_params,
        'best_val': best_val_rv, 'best_epoch': best_ep_rv,
        'seed': SEED,
    }, OUT / "stage1_rv_physics.pth")

    rv_pp = {
        'alpha': s1_rv_params['alpha'], 'c': s1_rv_params['c'],
        'vc': s1_rv_params['vc'], 'kappa': s1_rv_params['kappa'],
        'qx': s1_rv_params['qx'], 'qu': s1_rv_params['qu'],
        'R': s1_rv_params['R'],
        'P0_xx': s1_rv_params['P0_xx'], 'P0_uu': s1_rv_params['P0_uu'],
        'delta': s1_rv_params['delta'],
    }

    # ==================================================================
    #  STEP 3: DISCOVERY ON TOP OF RV BASELINE (6-term library)
    # ==================================================================
    print_section("STEP 3: CLOSURE DISCOVERY ON RV BASELINE")

    cl_best, s2_tr_nll, s2_val_nll, s2_nfev = train_s2_scipy(
        rv_pp, t_train, x_train, v_train,
        t_val, x_val, v_val, tag="S2-rv")

    print(f"  S2 coefficients:")
    for tn in TERM_NAMES:
        print(f"    {tn} = {cl_best[tn]:.6f}")
    print(f"    q_scale = {cl_best['q_scale']:.4f}")

    # ==================================================================
    #  STEP 4: TERM SELECTION
    # ==================================================================
    print_section("STEP 4: TERM SELECTION (variance + delta-NLL gate)")

    selected_terms, term_vars, rel_vars = select_terms(
        cl_best, rv_pp, t_val, x_val, v_val)

    print(f"  Selected terms: {selected_terms}")
    print(f"  Relative variances:")
    for tn in TERM_NAMES:
        rv = rel_vars.get(tn, 0.0)
        sel = "SELECTED" if tn in selected_terms else ""
        print(f"    {tn}: {rv:.4f} {sel}")

    # Build selected-only closure params
    cl_selected = {k: 0.0 for k in TERM_NAMES}
    cl_selected['q_scale'] = cl_best['q_scale']
    for tn in selected_terms:
        cl_selected[tn] = cl_best[tn]

    # ==================================================================
    #  STEP 5: EVALUATION -- ALL THREE ROWS
    # ==================================================================
    print_section("STEP 5: EVALUATION (warm start)")

    # Row 1: Original physics-only
    r_orig = evaluate_model("orig_phys", orig_pp, null_cl,
                            t_warm, x_warm, v_warm, eval_start_warm, MAX_H)

    # Row 2: RV physics-only (no closure)
    r_rv = evaluate_model("rv_phys", rv_pp, null_cl,
                          t_warm, x_warm, v_warm, eval_start_warm, MAX_H)

    # Row 3: RV + discovered closure (selected terms)
    r_rv_cl = evaluate_model("rv+closure", rv_pp, cl_selected,
                             t_warm, x_warm, v_warm, eval_start_warm, MAX_H)

    # Also evaluate the original with the v11.1 closure for reference
    # Load v11.1 seed-1 closure params if available
    v111_cl_ckpt = ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "checkpoints" / "closure_2t_seed1.pth"
    r_orig_cl = None
    if v111_cl_ckpt.exists():
        ckpt_cl = torch.load(v111_cl_ckpt, map_location='cpu', weights_only=False)
        cl_v111 = ckpt_cl['closure']
        orig_cl = {k: cl_v111.get(k, 0.0) for k in TERM_NAMES}
        orig_cl['q_scale'] = cl_v111.get('q_scale', 1.0)
        r_orig_cl = evaluate_model("orig+d2_closure", orig_pp, orig_cl,
                                   t_warm, x_warm, v_warm, eval_start_warm, MAX_H)

    # ==================================================================
    #  STEP 6: COMPARISON TABLE
    # ==================================================================
    print_section("STEP 6: COMPARISON TABLE")

    # Key horizons: h=1 (0.1s), h=5 (0.5s), h=10 (1.0s), h=20 (2.0s)
    key_h = [1, 5, 10, 20]
    key_labels = ['0.1s', '0.5s', '1.0s', '2.0s']

    rows = []
    for label, r, pp_dict, cl_dict in [
        ('Original physics-only', r_orig, orig_pp, null_cl),
        ('Original + d2 closure', r_orig_cl, orig_pp, orig_cl if r_orig_cl else null_cl),
        ('RV physics-only', r_rv, rv_pp, null_cl),
        ('RV + discovered', r_rv_cl, rv_pp, cl_selected),
    ]:
        if r is None:
            continue
        row = {
            'model': label,
            'acf1': r['acf1'],
            'nis_mean': r['nis_mean'],
            'cov90': r['cov90'],
        }
        for hi, hl in zip(key_h, key_labels):
            row[f'dxr2_{hl}'] = r['dxr2'][hi-1]
            row[f'mae_{hl}'] = r['mae'][hi-1]
        rows.append(row)

    df_table = pd.DataFrame(rows)
    table_path = OUT / "tables" / "rv_ablation_table.csv"
    df_table.to_csv(table_path, index=False)
    print(f"  Wrote {table_path}")

    # Print table
    print(f"\n  {'Model':<30} {'ACF1':>6} {'NIS':>6} "
          f"{'DxR2@0.1s':>10} {'DxR2@0.5s':>10} {'DxR2@1.0s':>10} {'DxR2@2.0s':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in df_table.iterrows():
        print(f"  {row['model']:<30} {row['acf1']:6.3f} {row['nis_mean']:6.3f} "
              f"{row['dxr2_0.1s']:+10.4f} {row['dxr2_0.5s']:+10.4f} "
              f"{row['dxr2_1.0s']:+10.4f} {row['dxr2_2.0s']:+10.4f}")

    # ==================================================================
    #  STEP 7: FIGURE
    # ==================================================================
    print_section("STEP 7: FIGURE")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    h_range = np.arange(1, MAX_H + 1) * DT  # convert to seconds

    # Panel (a): DxR2 vs horizon
    ax1.plot(h_range, r_orig['dxr2'], 'k-', label='Original physics-only', lw=1.5)
    ax1.plot(h_range, r_rv['dxr2'], '#D55E00', ls='-', label='RV physics-only', lw=1.5)
    ax1.plot(h_range, r_rv_cl['dxr2'], '#0072B2', ls='--', label=f'RV + discovered ({"+".join(selected_terms) if selected_terms else "null"})', lw=1.5)
    if r_orig_cl is not None:
        ax1.plot(h_range, r_orig_cl['dxr2'], '#222222', ls='--',
                 label='Original + d2 closure', lw=1.5, alpha=0.6)
    ax1.axhline(0, color='gray', ls=':', alpha=0.5)
    ax1.set_xlabel('Horizon (s)')
    ax1.set_ylabel(r'$R^2_{\Delta x}$')
    ax1.set_title('(a) Displacement-increment skill')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 10)

    # Panel (b): Bar chart of key metrics
    models = ['Orig.\nphysics', 'Orig.\n+d2', 'RV\nphysics', 'RV\n+disc.']
    x_pos = np.arange(len(models))
    colors = ['#999999', '#222222', '#D55E00', '#0072B2']
    vals_1s = [r_orig['dxr2'][9]]
    if r_orig_cl:
        vals_1s.append(r_orig_cl['dxr2'][9])
    else:
        vals_1s.append(np.nan)
    vals_1s.extend([r_rv['dxr2'][9], r_rv_cl['dxr2'][9]])

    vals_2s = [r_orig['dxr2'][19]]
    if r_orig_cl:
        vals_2s.append(r_orig_cl['dxr2'][19])
    else:
        vals_2s.append(np.nan)
    vals_2s.extend([r_rv['dxr2'][19], r_rv_cl['dxr2'][19]])

    w = 0.35
    ax2.bar(x_pos - w/2, vals_1s, w, label=r'$\tau=1.0$ s', color=colors, alpha=0.85)
    ax2.bar(x_pos + w/2, vals_2s, w, label=r'$\tau=2.0$ s', color=colors, alpha=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.set_ylabel(r'$R^2_{\Delta x}$')
    ax2.set_title(r'(b) $R^2_{\Delta x}$ at key horizons')
    ax2.legend(fontsize=8)
    ax2.axhline(0, color='gray', ls=':', alpha=0.5)

    fig.suptitle('Relative-Velocity Baseline Ablation', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "figures" / "rv_ablation_comparison.png")
    fig.savefig(OUT / "figures" / "rv_ablation_comparison.pdf")
    plt.close(fig)
    print(f"  Wrote rv_ablation_comparison.png/pdf")

    # ==================================================================
    #  STEP 8: RESULTS JSON
    # ==================================================================
    print_section("STEP 8: RESULTS SUMMARY")

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': SEED,
        'original_physics': {
            'alpha': s1_orig_params['alpha'],
            'kappa': s1_orig_params['kappa'],
            'c': s1_orig_params['c'],
            'vc': s1_orig_params['vc'],
            'delta': 0.0,
            'dxr2_1s': r_orig['dxr2'][9],
            'dxr2_2s': r_orig['dxr2'][19],
            'acf1': r_orig['acf1'],
            'nis': r_orig['nis_mean'],
        },
        'rv_physics': {
            'alpha': s1_rv_params['alpha'],
            'kappa': s1_rv_params['kappa'],
            'c': s1_rv_params['c'],
            'vc': s1_rv_params['vc'],
            'delta': s1_rv_params['delta'],
            'dxr2_1s': r_rv['dxr2'][9],
            'dxr2_2s': r_rv['dxr2'][19],
            'acf1': r_rv['acf1'],
            'nis': r_rv['nis_mean'],
        },
        'rv_discovery': {
            's2_coefficients': {tn: cl_best[tn] for tn in TERM_NAMES},
            's2_q_scale': cl_best['q_scale'],
            'selected_terms': selected_terms,
            'rel_var': rel_vars,
            'selected_coefficients': {tn: cl_selected[tn] for tn in TERM_NAMES},
            'dxr2_1s': r_rv_cl['dxr2'][9],
            'dxr2_2s': r_rv_cl['dxr2'][19],
            'acf1': r_rv_cl['acf1'],
            'nis': r_rv_cl['nis_mean'],
        },
        'runtime_s': time.time() - t0_all,
    }

    if r_orig_cl is not None:
        results['original_with_d2_closure'] = {
            'dxr2_1s': r_orig_cl['dxr2'][9],
            'dxr2_2s': r_orig_cl['dxr2'][19],
            'acf1': r_orig_cl['acf1'],
            'nis': r_orig_cl['nis_mean'],
        }

    results_path = OUT / "rv_ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {results_path}")

    # ==================================================================
    #  INTERPRETATION
    # ==================================================================
    print_section("INTERPRETATION")

    delta_val = s1_rv_params['delta']
    dxr2_gain_rv = r_rv['dxr2'][9] - r_orig['dxr2'][9]
    dxr2_gain_cl = r_rv_cl['dxr2'][9] - r_rv['dxr2'][9]

    print(f"  Learned delta = {delta_val:.4f}")
    print(f"  DxR2@1s gain from RV drag alone: {dxr2_gain_rv:+.4f}")

    if selected_terms:
        print(f"  Discovery selected: {selected_terms}")
        print(f"  DxR2@1s additional gain from closure: {dxr2_gain_cl:+.4f}")
        if 'd2' in selected_terms:
            print("  ** d2 (v*|u|) STILL SELECTED even with RV baseline **")
            print("  -> Closure is NOT merely compensating for non-relative coupling")
        else:
            print(f"  ** d2 NOT selected; selected terms changed to: {selected_terms} **")
            print("  -> RV baseline absorbs part of the closure signal")
    else:
        print("  Discovery selected: NULL (no closure terms)")
        print("  -> RV baseline absorbs the closure signal entirely")
        if abs(dxr2_gain_rv) < 0.01:
            print("  ** But RV gain is negligible; closure was not algebraic artifact **")

    if r_orig_cl is not None:
        dxr2_orig_cl = r_orig_cl['dxr2'][9]
        dxr2_rv_best = r_rv_cl['dxr2'][9]
        print(f"\n  Original+d2 DxR2@1s = {dxr2_orig_cl:+.4f}")
        print(f"  RV+discovered DxR2@1s = {dxr2_rv_best:+.4f}")
        if dxr2_orig_cl > dxr2_rv_best + 0.01:
            print("  -> Original closure OUTPERFORMS RV variant")
        elif abs(dxr2_orig_cl - dxr2_rv_best) < 0.01:
            print("  -> Comparable performance")
        else:
            print("  -> RV variant outperforms original")

    elapsed = time.time() - t0_all
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  All outputs in {OUT}")
    print("  DONE.")


if __name__ == '__main__':
    main()
