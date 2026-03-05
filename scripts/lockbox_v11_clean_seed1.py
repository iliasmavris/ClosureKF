"""
Lockbox v11: Clean 10 Hz Retrain (Seed 1) + Full Evaluation.

Uses clean splits from processed_data_10hz_clean_v1/ (from v10 audit).
Trains physics-only (Stage 1) and 2-term closure (Stage 2) on clean data.
Evaluates with both warm-start (50s val prefix) and cold-start.

Usage:  python -u scripts/lockbox_v11_clean_seed1.py
Output: final_lockbox_v11_clean_10hz_seed1/
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
from models.kalman_forecaster import KalmanForecaster
from models.kalman_closure import KalmanForecasterClosure

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
SEED = 1
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
OUT = ROOT / "final_lockbox_v11_clean_10hz_seed1"

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
#  DxR2 MULTI-HORIZON (efficient single-pass rollout)
# ==============================================================================

def compute_dxr2(params, cl_params, states_x, states_u,
                 t, x_obs, v, max_h=100, eval_start=1, mode='oracle',
                 indices=None):
    """DxR2(h) and MAE(h) for h=1..max_h. Returns (r2_arr, mae_arr)."""
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
#  TRAINING FUNCTION
# ==============================================================================

def train_model(model, train_loader, val_loader, device,
                max_epochs, patience, lr, sched_patience=10,
                param_getter=None, tag=""):
    """Train with early stopping.
    Returns (best_val, best_epoch, final_train, train_hist, val_hist).
    """
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

    # Filter
    filt = kf_filter_2state(params, cl_params, t, x_obs, v)
    innov = filt['innovations']; S_vals = filt['S_values']
    sx = filt['states_x']; su = filt['states_u']

    # Scored segment
    e = innov[eval_start:]; S_sc = S_vals[eval_start:]
    valid = ~np.isnan(e)
    e_v = e[valid]; S_v = S_sc[valid]
    n_valid = len(e_v)

    # ACF
    acf = compute_acf(e_v, max_lag=50)

    # NIS
    nis = float(np.mean(e_v**2 / np.maximum(S_v, 1e-15)))

    # Coverage 90%
    z90 = 1.6449
    cov90 = float(np.mean(np.abs(e_v) <= z90 * np.sqrt(np.maximum(S_v, 1e-15))))

    # Grey-box
    cl_sc = filt['cl_dt'][eval_start:]
    ph_sc = filt['physics'][eval_start:]
    tot_sc = cl_sc + ph_sc
    var_cl = np.var(cl_sc)
    var_tot = np.var(tot_sc) if np.var(tot_sc) > 1e-15 else 1.0
    grey_frac = float(var_cl / var_tot) if var_tot > 1e-15 else 0.0
    ratio = np.abs(cl_sc) / np.maximum(np.abs(tot_sc), 1e-15)
    grey_med = float(np.median(ratio))

    # DxR2 oracle + persist
    dxr2_orc, mae_orc = compute_dxr2(
        params, cl_params, sx, su, t, x_obs, v, max_h, eval_start, 'oracle')
    dxr2_per, mae_per = compute_dxr2(
        params, cl_params, sx, su, t, x_obs, v, max_h, eval_start, 'no_forcing')

    elapsed = time.time() - t0_ev
    d10 = dxr2_orc[9] if max_h >= 10 else np.nan
    m10 = mae_orc[9] if max_h >= 10 else np.nan
    print(f"    [{label}] ACF(1)={acf[1]:.4f} NIS={nis:.4f} cov90={cov90:.3f} "
          f"DxR2@10={d10:+.4f} MAE@10={m10:.6f} grey={grey_frac:.3f} [{elapsed:.0f}s]")

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
#  EVENT DETECTION (simple 2-means + hysteresis)
# ==============================================================================

def detect_events(x, min_persist=3):
    """k-means(k=2) on displacement with hysteresis. Returns event indices."""
    # Simple 2-means
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

    # Apply hysteresis
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

    # Transitions
    event_idx = np.where(np.diff(clean) != 0)[0] + 1
    return event_idx, clean


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    t0_global = time.time()
    device = torch.device('cpu')

    for d in ['checkpoints', 'figures', 'tables']:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOCKBOX V11: CLEAN 10 Hz RETRAIN (SEED 1)")
    print("=" * 70)
    print(f"Output -> {OUT}")

    # ==================================================================
    #  STEP 0: DATA & INTEGRITY
    # ==================================================================
    print_section("STEP 0: DATA & INTEGRITY")

    csv_paths = {
        'train': CLEAN_DIR / "train_10hz_ready.csv",
        'val':   CLEAN_DIR / "val_10hz_ready.csv",
        'test':  CLEAN_DIR / "test_10hz_ready.csv",
    }

    # PATH GUARD: crash if any path references old contaminated directory
    for name, p in csv_paths.items():
        p_str = str(p)
        assert "processed_data_10hz/" not in p_str.replace("\\", "/") or \
               "processed_data_10hz_clean" in p_str.replace("\\", "/"), \
            f"PATH GUARD VIOLATION: {p_str}"
        assert "processed_data_10hz\\" not in p_str or \
               "processed_data_10hz_clean" in p_str, \
            f"PATH GUARD VIOLATION: {p_str}"
    # Stronger check: the directory name must be exactly clean_v1
    for name, p in csv_paths.items():
        assert "processed_data_10hz_clean_v1" in str(p), \
            f"PATH GUARD: expected clean_v1 dir, got {p}"
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

    # Preflight asserts
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
    #  STEP 1: STAGE 1 -- PHYSICS ONLY
    # ==================================================================
    print_section("STEP 1: STAGE 1 -- PHYSICS ONLY (seed=1)")
    torch.manual_seed(SEED); np.random.seed(SEED)

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

    model_s1 = KalmanForecaster(use_kappa=True).to(device)
    best_val_s1, best_ep_s1, final_tr_s1, tr_h_s1, val_h_s1 = \
        train_model(model_s1, train_ld_s1, val_ld_s1, device,
                    S1_EPOCHS, S1_PATIENCE, S1_LR, S1_SCHED, tag="S1")

    s1_params = model_s1.param_summary()
    print(f"  S1 learned: alpha={s1_params['alpha']:.4f} "
          f"tau={s1_params['tau']:.3f}s c={s1_params['c']:.4f} "
          f"vc={s1_params['vc']:.4f} kappa={s1_params['kappa']:.4f}")

    torch.save({
        'state_dict': model_s1.state_dict(),
        'params': s1_params,
        'best_val': best_val_s1, 'best_epoch': best_ep_s1,
    }, OUT / "checkpoints" / "stage1_physics_seed1.pth")

    # Training curves figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(tr_h_s1)+1), tr_h_s1, label='Train NLL')
    ax.plot(range(1, len(val_h_s1)+1), val_h_s1, label='Val NLL')
    ax.axvline(best_ep_s1, color='red', ls='--', alpha=0.5,
               label=f'Best ep={best_ep_s1}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gaussian NLL')
    ax.set_title('Stage 1: Physics Only Training')
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "figures" / "training_curves_s1.png")
    plt.close(fig)

    # ==================================================================
    #  STEP 2: STAGE 2 -- CLOSURE 2-TERM
    # ==================================================================
    print_section("STEP 2: STAGE 2 -- CLOSURE 2-TERM (seed=1)")
    torch.manual_seed(SEED); np.random.seed(SEED)

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

    # Build closure model from S1 params
    model_s2 = KalmanForecasterClosure(
        alpha_init=min(max(s1_params['alpha'], 0.001), 0.999),
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
    ).to(device)

    # Freeze physics
    model_s2.freeze_physics()
    # Freeze a1, b1, d1, d3 at zero
    with torch.no_grad():
        model_s2.a1_raw.fill_(-10.0); model_s2.a1_raw.requires_grad_(False)
        model_s2.b1.fill_(0.0); model_s2.b1.requires_grad_(False)
        model_s2.d1_raw.fill_(-10.0); model_s2.d1_raw.requires_grad_(False)
        model_s2.d3_raw.fill_(-10.0); model_s2.d3_raw.requires_grad_(False)

    # Trainable: b2, d2_raw, log_q_scale
    s2_trainable = lambda: [model_s2.b2, model_s2.d2_raw, model_s2.log_q_scale]

    best_val_s2, best_ep_s2, final_tr_s2, tr_h_s2, val_h_s2 = \
        train_model(model_s2, train_ld_s2, val_ld_s2, device,
                    S2_EPOCHS, S2_PATIENCE, S2_LR, S2_SCHED,
                    param_getter=s2_trainable, tag="S2")

    s2_params = model_s2.param_summary()
    cl_sum = model_s2.closure_summary()
    print(f"  S2 learned: b2={cl_sum['b2']:.4f} d2={cl_sum['d2']:.4f} "
          f"q_scale={cl_sum['q_scale']:.4f}")

    torch.save({
        'state_dict': model_s2.state_dict(),
        'params': s2_params,
        'closure': cl_sum,
        'best_val': best_val_s2, 'best_epoch': best_ep_s2,
    }, OUT / "checkpoints" / "closure_2t_seed1.pth")

    # Training curves figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(tr_h_s2)+1), tr_h_s2, label='Train NLL')
    ax.plot(range(1, len(val_h_s2)+1), val_h_s2, label='Val NLL')
    ax.axvline(best_ep_s2, color='red', ls='--', alpha=0.5,
               label=f'Best ep={best_ep_s2}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gaussian NLL')
    ax.set_title('Stage 2: Closure 2-term Training')
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "figures" / "training_curves_s2.png")
    plt.close(fig)

    # ==================================================================
    #  STEP 3: EVALUATION
    # ==================================================================
    print_section("STEP 3: EVALUATION")

    # Build param dicts for numpy filter
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
    print(f"  Warmup guard: {n_warmup} warmup + {len(df_test)} test, "
          f"scoring {n_scored}")

    # --- COLD START ---
    t_cold = df_test['timestamp'].values.astype(np.float64)
    x_cold = df_test['displacement'].values.astype(np.float64)
    v_cold = df_test['velocity'].values.astype(np.float64)
    eval_start_cold = 1

    # Evaluate 4 variants
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
    #  STEP 4: TABLES & FIGURES
    # ==================================================================
    print_section("STEP 4: TABLES & FIGURES")

    # --- metrics_table.csv ---
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
        OUT / "tables" / "metrics_table.csv", index=False)
    print("  Wrote metrics_table.csv")

    # --- horizon_curve.csv ---
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
        OUT / "tables" / "horizon_curve.csv", index=False)
    print("  Wrote horizon_curve.csv")

    # --- ljung_box.csv ---
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
        OUT / "tables" / "ljung_box.csv", index=False)
    print("  Wrote ljung_box.csv")

    # --- learned_params.csv ---
    param_rows = [
        {'stage': 'S1_physics', **s1_params},
        {'stage': 'S2_closure', **s2_params},
    ]
    pd.DataFrame(param_rows).to_csv(
        OUT / "tables" / "learned_params.csv", index=False)
    print("  Wrote learned_params.csv")

    # --- horizon_curve_dxr2.png ---
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
    ax.set_title('DxR2 Horizon Curve (warm start, clean 10 Hz)')
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "figures" / "horizon_curve_dxr2.png")
    plt.close(fig)
    print("  Wrote horizon_curve_dxr2.png")

    # --- innovation_acf.png ---
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
    ax.set_title('Innovation ACF (warm start)')
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "figures" / "innovation_acf.png")
    plt.close(fig)
    print("  Wrote innovation_acf.png")

    # ==================================================================
    #  STEP 5: EVENT ROBUSTNESS
    # ==================================================================
    print_section("STEP 5: EVENT ROBUSTNESS")

    # Use warm-start results
    t_ev = t_warm
    x_ev = x_warm
    v_ev = v_warm
    es_ev = eval_start_warm

    # Event detection on test displacement
    x_test_only = x_warm[es_ev:]
    event_idx_local, labels = detect_events(x_test_only, min_persist=3)
    # Convert to warm-array indices
    event_idx_warm = event_idx_local + es_ev
    n_events = len(event_idx_warm)
    print(f"  Detected {n_events} events in test set")

    # Event/non-event masks (in warm-array index space)
    event_radius_s = 10.0  # +/- 10 seconds
    event_radius_steps = int(event_radius_s / DT)
    test_indices = np.arange(es_ev, len(t_ev))

    # Mark indices within +/- event_radius of any event
    event_mask = np.zeros(len(t_ev), dtype=bool)
    for eidx in event_idx_warm:
        lo = max(es_ev, eidx - event_radius_steps)
        hi = min(len(t_ev), eidx + event_radius_steps + 1)
        event_mask[lo:hi] = True

    event_indices = test_indices[event_mask[es_ev:]]
    nonevent_indices = test_indices[~event_mask[es_ev:]]
    print(f"  Event window: {len(event_indices)} pts, "
          f"Non-event: {len(nonevent_indices)} pts")

    # DxR2@50 for subsets
    h_event = 50
    event_tbl = []
    for mtype, pp, cc in [('physics', phys_pp, phys_cl),
                           ('closure', clos_pp, clos_cl)]:
        label_w = f"{mtype}_warm"
        sx_w = results[label_w]['states_x']
        su_w = results[label_w]['states_u']

        r2_full, _ = compute_dxr2(pp, cc, sx_w, su_w, t_ev, x_ev, v_ev,
                                  h_event, es_ev, 'oracle')
        r2_event, _ = compute_dxr2(pp, cc, sx_w, su_w, t_ev, x_ev, v_ev,
                                   h_event, es_ev, 'oracle',
                                   indices=event_indices)
        r2_nonevent, _ = compute_dxr2(pp, cc, sx_w, su_w, t_ev, x_ev, v_ev,
                                      h_event, es_ev, 'oracle',
                                      indices=nonevent_indices)
        d50_full = float(r2_full[h_event - 1]) if h_event <= len(r2_full) else np.nan
        d50_event = float(r2_event[h_event - 1]) if h_event <= len(r2_event) else np.nan
        d50_nonevent = float(r2_nonevent[h_event - 1]) if h_event <= len(r2_nonevent) else np.nan

        event_tbl.append({
            'model': mtype,
            'dxr2_50_full': d50_full,
            'dxr2_50_event': d50_event,
            'dxr2_50_nonevent': d50_nonevent,
            'n_events': n_events,
            'n_event_pts': len(event_indices),
            'n_nonevent_pts': len(nonevent_indices),
        })
        print(f"  [{mtype}] DxR2@50: full={d50_full:+.4f} "
              f"event={d50_event:+.4f} nonevent={d50_nonevent:+.4f}")

    pd.DataFrame(event_tbl).to_csv(
        OUT / "tables" / "event_ablation.csv", index=False)
    print("  Wrote event_ablation.csv")

    # --- event_skill_bars.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.array([0, 1, 2])
    w = 0.35
    phys_vals = [event_tbl[0]['dxr2_50_full'],
                 event_tbl[0]['dxr2_50_event'],
                 event_tbl[0]['dxr2_50_nonevent']]
    clos_vals = [event_tbl[1]['dxr2_50_full'],
                 event_tbl[1]['dxr2_50_event'],
                 event_tbl[1]['dxr2_50_nonevent']]
    ax.bar(x_pos - w/2, phys_vals, w, label='Physics', color='steelblue', alpha=0.8)
    ax.bar(x_pos + w/2, clos_vals, w, label='Closure', color='indianred', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Full test', 'Event (+/-10s)', 'Non-event'])
    ax.set_ylabel('DxR2@50')
    ax.set_title('Event Robustness: DxR2@50')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "figures" / "event_skill_bars.png")
    plt.close(fig)
    print("  Wrote event_skill_bars.png")

    # ==================================================================
    #  STEP 6: README
    # ==================================================================
    print_section("STEP 6: README")

    # Gather comparison data
    pw = results['physics_warm']
    cw = results['closure_warm']
    pc = results['physics_cold']
    cc_r = results['closure_cold']

    # Old paper params for comparison
    old_params = {
        'alpha': 0.45, 'kappa': 0.13, 'c': 0.91,
        'b2': 6.338, 'd2': 10.458, 'q_scale': 1.856,
    }

    # Narrative survival check
    closure_improves_acf = cw['acf1'] < pw['acf1']
    closure_improves_dxr2 = cw['dxr2_10_oracle'] > pw['dxr2_10_oracle']

    # Warm vs cold comparison
    warm_cold_acf_delta = abs(cw['acf1'] - cc_r['acf1'])
    warm_cold_dxr2_delta = abs(cw['dxr2_10_oracle'] - cc_r['dxr2_10_oracle'])

    readme_text = f"""# Lockbox v11: Clean 10 Hz Retrain (Seed 1)

## Summary
Retrained the 2-state Kalman filter (physics-only Stage 1, 2-term closure Stage 2)
on clean 10 Hz splits from `processed_data_10hz_clean_v1/`.
Evaluated with warm-start (50s val prefix) and cold-start on the test set.

## Data
- Train: {len(df_train)} pts ({df_train['timestamp'].iloc[0]:.1f}-{df_train['timestamp'].iloc[-1]:.1f}s)
- Val: {len(df_val)} pts ({df_val['timestamp'].iloc[0]:.1f}-{df_val['timestamp'].iloc[-1]:.1f}s)
- Test: {len(df_test)} pts ({df_test['timestamp'].iloc[0]:.1f}-{df_test['timestamp'].iloc[-1]:.1f}s)
- Input MD5s: see tables/input_md5.csv

## 1. Learned Parameters vs Old Paper

| Parameter | Old paper | v11 clean |
|-----------|-----------|-----------|
| alpha     | {old_params['alpha']:.4f}    | {s1_params['alpha']:.4f}    |
| kappa     | {old_params['kappa']:.4f}    | {s1_params['kappa']:.4f}    |
| c         | {old_params['c']:.4f}     | {s1_params['c']:.4f}     |
| b2        | {old_params['b2']:.3f}     | {cl_sum['b2']:.4f}    |
| d2        | {old_params['d2']:.3f}    | {cl_sum['d2']:.4f}    |
| q_scale   | {old_params['q_scale']:.3f}     | {cl_sum['q_scale']:.4f}    |

## 2. Qualitative Narrative

Closure {'IMPROVES' if closure_improves_acf else 'does NOT improve'} ACF(1): \
{pw['acf1']:.4f} -> {cw['acf1']:.4f}
Closure {'IMPROVES' if closure_improves_dxr2 else 'does NOT improve'} DxR2@10: \
{pw['dxr2_10_oracle']:+.4f} -> {cw['dxr2_10_oracle']:+.4f}

**Narrative {'SURVIVES' if closure_improves_acf and closure_improves_dxr2 else 'DOES NOT SURVIVE'} on clean splits.**

## 3. Warm Start vs Cold Start

| Metric         | Warm physics | Warm closure | Cold physics | Cold closure |
|----------------|-------------|-------------|-------------|-------------|
| ACF(1)         | {pw['acf1']:.4f}      | {cw['acf1']:.4f}      | {pc['acf1']:.4f}      | {cc_r['acf1']:.4f}      |
| DxR2@10 oracle | {pw['dxr2_10_oracle']:+.4f}    | {cw['dxr2_10_oracle']:+.4f}    | {pc['dxr2_10_oracle']:+.4f}    | {cc_r['dxr2_10_oracle']:+.4f}    |
| NIS            | {pw['nis_mean']:.4f}      | {cw['nis_mean']:.4f}      | {pc['nis_mean']:.4f}      | {cc_r['nis_mean']:.4f}      |
| cov90          | {pw['cov90']:.3f}       | {cw['cov90']:.3f}       | {pc['cov90']:.3f}       | {cc_r['cov90']:.3f}       |
| MAE@10         | {pw['mae10']:.6f}  | {cw['mae10']:.6f}  | {pc['mae10']:.6f}  | {cc_r['mae10']:.6f}  |

Warm-cold delta ACF(1): {warm_cold_acf_delta:.4f}
Warm-cold delta DxR2@10: {warm_cold_dxr2_delta:.4f}
Conclusions {'DO NOT' if warm_cold_dxr2_delta < 0.05 else 'DO'} depend on warm start \
(delta DxR2@10 {'<' if warm_cold_dxr2_delta < 0.05 else '>='} 0.05).

## 4. Training Details

### Stage 1: Physics Only
- Selected epoch: {best_ep_s1}
- Best val NLL: {best_val_s1:.5f}
- Final train NLL: {final_tr_s1:.5f}

### Stage 2: Closure 2-term
- Selected epoch: {best_ep_s2}
- Best val NLL: {best_val_s2:.5f}
- Final train NLL: {final_tr_s2:.5f}

## Event Robustness
- {n_events} events detected (k-means k=2, hysteresis 0.3s)
- Physics DxR2@50: full={event_tbl[0]['dxr2_50_full']:+.4f}, event={event_tbl[0]['dxr2_50_event']:+.4f}, non-event={event_tbl[0]['dxr2_50_nonevent']:+.4f}
- Closure DxR2@50: full={event_tbl[1]['dxr2_50_full']:+.4f}, event={event_tbl[1]['dxr2_50_event']:+.4f}, non-event={event_tbl[1]['dxr2_50_nonevent']:+.4f}

## Output Files
- checkpoints/stage1_physics_seed1.pth
- checkpoints/closure_2t_seed1.pth
- figures/training_curves_s1.png
- figures/training_curves_s2.png
- figures/horizon_curve_dxr2.png
- figures/innovation_acf.png
- figures/event_skill_bars.png
- tables/input_md5.csv
- tables/metrics_table.csv
- tables/horizon_curve.csv
- tables/ljung_box.csv
- tables/event_ablation.csv
- tables/learned_params.csv

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total runtime: {time.time() - t0_global:.0f}s
"""
    with open(OUT / "README.md", 'w') as f:
        f.write(readme_text)
    print("  Wrote README.md")

    # ==================================================================
    #  FINAL SUMMARY
    # ==================================================================
    print_section("FINAL SUMMARY")
    elapsed = time.time() - t0_global

    # Count output files
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(OUT):
        n_files += len(filenames)
    print(f"  Total output files: {n_files}")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Quick summary table
    print(f"\n  {'Metric':<25} {'Physics warm':>14} {'Closure warm':>14}")
    print(f"  {'-'*25} {'-'*14} {'-'*14}")
    for key, fmt in [('acf1', '14.4f'), ('dxr2_10_oracle', '+14.4f'),
                     ('nis_mean', '14.4f'), ('cov90', '14.3f'), ('mae10', '14.6f')]:
        print(f"  {key:<25} {format(pw[key], fmt)} {format(cw[key], fmt)}")

    print(f"\n  DONE. All outputs in {OUT}")


if __name__ == '__main__':
    main()
