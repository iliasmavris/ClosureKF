"""
97_ab_alignment_impact.py -- A/B micro-test for forcing-alignment fix.

Trains S1 (physics) + S2 (closure 2-term) under two conventions:
  A = OLD: v_fut = v[t+1:t+H+1]  (1-step lookahead, the old bug)
  B = NEW: v_fut = v[t:t+H]      (start-of-interval, the fix)

Both arms use the SAME numpy KF evaluation (always correct start-of-interval).
Compares learned parameters + headline metrics to quantify impact.

Usage:  python -u scripts/97_ab_alignment_impact.py
Output: outputs/alignment_ab/
Runtime: ~20-40 min (2 seeds x 2 arms, shortened configs)
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
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_forecaster import KalmanForecaster
from models.kalman_closure import KalmanForecasterClosure

# ==============================================================================
#  CONFIGURATION (shorter than full v11.1 for speed)
# ==============================================================================
SEEDS = [1, 2]
FORCE_CPU = True
DT = 0.1
VAR_FLOOR = 1e-6

# Shortened training configs (enough for parameter convergence comparison)
S1_L = 64; S1_H = 20; S1_BATCH = 128
S1_EPOCHS = 60; S1_LR = 1e-2; S1_PATIENCE = 15; S1_SCHED = 5

S2_L = 64; S2_H = 20; S2_BATCH = 128
S2_EPOCHS = 100; S2_LR = 1e-2; S2_PATIENCE = 20; S2_SCHED = 5

MAX_H = 20  # Only need h up to 2.0s = 20 steps at 10 Hz
WARMUP_SEC = 50.0
HEADLINE_H = [5, 10, 20]  # 0.5s, 1.0s, 2.0s

CLEAN_DIR = ROOT / "processed_data_10hz_clean_v1"
OUT = ROOT / "outputs" / "alignment_ab"
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device('cpu') if FORCE_CPU else torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
#  DATASET WITH SWITCHABLE ALIGNMENT
# ==============================================================================
class AlignmentTestDataset(Dataset):
    """Minimal dataset with switchable v_fut alignment convention."""

    def __init__(self, csv_path, L, H, convention='new'):
        """
        convention: 'new' = v_fut = v[t:t+H] (start-of-interval, correct)
                    'old' = v_fut = v[t+1:t+H+1] (end-of-interval, old bug)
        """
        df = pd.read_csv(csv_path).sort_values('timestamp').reset_index(drop=True)
        self.v = df['velocity'].values.astype(np.float32)
        self.x = df['displacement'].values.astype(np.float32)
        self.dt = df['time_delta'].values.astype(np.float32)
        self.ts = df['timestamp'].values.astype(np.float32)
        self.L = L
        self.H = H
        self.convention = convention

        T = len(self.v)
        min_t = L - 1
        # Need H+1 future points for OLD convention (goes to t+H+1)
        max_t = T - H - 2 if convention == 'old' else T - H - 1
        self.indices = list(range(min_t, max_t + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        v_hist = self.v[t - self.L + 1:t + 1].copy()
        dt_hist = self.dt[t - self.L + 1:t + 1].copy()
        x_hist = self.x[t - self.L + 1:t + 1].copy()

        if self.convention == 'new':
            v_fut = self.v[t:t + self.H].copy()         # START-OF-INTERVAL
        else:
            v_fut = self.v[t + 1:t + self.H + 1].copy()  # OLD BUG

        dt_fut = self.dt[t + 1:t + self.H + 1].copy()
        x_fut_true = self.x[t + 1:t + self.H + 1].copy()
        x_current = self.x[t]

        return (
            torch.tensor(v_hist, dtype=torch.float32),
            torch.tensor(dt_hist, dtype=torch.float32),
            torch.tensor(x_hist, dtype=torch.float32),
            torch.tensor(v_fut, dtype=torch.float32),
            torch.tensor(dt_fut, dtype=torch.float32),
            torch.tensor(x_fut_true, dtype=torch.float32),
            torch.tensor(x_current, dtype=torch.float32),
            {'t_index': t},
        )


# ==============================================================================
#  HELPERS
# ==============================================================================
def gaussian_nll(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def compute_acf(e, max_lag=50):
    e_c = e - np.mean(e)
    var_e = np.var(e)
    n = len(e)
    if var_e < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(e_c[:n-l] * e_c[l:]) / (n * var_e) if l > 0
                     else 1.0 for l in range(max_lag + 1)])


# ==============================================================================
#  NUMPY 2-STATE FILTER (always correct START-OF-INTERVAL)
# ==============================================================================
def kf_filter_2state(params, cl_params, t, x_obs, v):
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    states_x = np.zeros(N)
    states_u = np.zeros(N)

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    b2_v = cl_params.get('b2', 0.0)
    d2_v = cl_params.get('d2', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]

    for k in range(1, N):
        dt_k = t[k] - t[k-1]
        if dt_k <= 0:
            dt_k = 0.1
        rho_u = math.exp(-alpha * dt_k)
        g = max(v[k-1]**2 - vc**2, 0.0)
        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = b2_v * dv_w - d2_v * u_st * abs(v_w)
        cl_d = cl * dt_k
        x_p = s[0] + s[1] * dt_k
        u_p = rho_u * s[1] - kap * s[0] * dt_k + c_val * g * dt_k + cl_d
        s_pred = np.array([x_p, u_p])

        F_mat = np.array([[1, dt_k], [-kap*dt_k, rho_u]])
        Q = np.diag([q_sc*qx*dt_k, q_sc*qu*dt_k])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        innovations[k] = innov

        K_vec = P_pred[:, 0] / S_val
        s = s_pred + K_vec * innov
        IKH = np.eye(2) - np.outer(K_vec, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K_vec, K_vec)
        states_x[k] = s[0]; states_u[k] = s[1]

    return {'innovations': innovations, 'states_x': states_x, 'states_u': states_u}


def compute_dxr2(params, cl_params, states_x, states_u,
                 t, x_obs, v, max_h=20, eval_start=1):
    """DxR2(h) for h=1..max_h. Always start-of-interval."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    b2_v = cl_params.get('b2', 0.0)
    d2_v = cl_params.get('d2', 0.0)

    dx_pred = [[] for _ in range(max_h)]
    dx_true = [[] for _ in range(max_h)]

    for i in range(max(eval_start, 1), N - 1):
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
            cl = b2_v * dv_w - d2_v * su * abs(v_w)
            sx_new = sx + su * dt_s
            su_new = rho * su - kap * sx * dt_s + c_val * g * dt_s + cl * dt_s
            sx, su = sx_new, su_new
            h = step + 1
            dx_pred[h-1].append(sx - x_obs[i])
            dx_true[h-1].append(x_obs[i + h] - x_obs[i])

    r2_arr = np.full(max_h, np.nan)
    for h in range(max_h):
        if len(dx_pred[h]) < 10:
            continue
        dp = np.array(dx_pred[h])
        do = np.array(dx_true[h])
        err = do - dp
        ss_res = np.sum(err**2)
        ss_tot = np.sum((do - np.mean(do))**2)
        r2_arr[h] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ==============================================================================
#  TRAINING
# ==============================================================================
def train_model(model, train_loader, val_loader,
                max_epochs, patience, lr, sched_patience,
                param_getter=None, tag=""):
    if param_getter:
        params = [p for p in param_getter() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

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
                xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
                vl = gaussian_nll(xp, xv, x_true, VAR_FLOOR)
                vt += vl.item(); vn += 1
            val_nll = vt / vn

        scheduler.step(val_nll)
        if val_nll < best_loss:
            best_loss = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 20 == 0 or ep == 0 or wait >= patience:
            print(f"    [{tag}] ep {ep+1:3d}  tr={tr_nll:.5f}  "
                  f"val={val_nll:.5f}  best={best_loss:.5f}@ep{best_ep}  "
                  f"[{time.time()-t0:.0f}s]")
        if wait >= patience:
            print(f"    [{tag}] Early stop at ep {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"    [{tag}] Done {time.time()-t0:.0f}s, best_val={best_loss:.5f} at ep={best_ep}")
    return best_loss, best_ep


# ==============================================================================
#  SINGLE ARM
# ==============================================================================
def run_arm(arm_name, convention, seed, csv_train, csv_val,
            df_val, df_test):
    """Run S1 + S2 + eval for one (convention, seed) arm."""
    print(f"\n{'='*60}")
    print(f"  ARM={arm_name}  CONVENTION={convention}  SEED={seed}")
    print(f"{'='*60}")
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- S1 --
    train_ds = AlignmentTestDataset(csv_train, S1_L, S1_H, convention)
    val_ds   = AlignmentTestDataset(csv_val,   S1_L, S1_H, convention)
    train_ld = DataLoader(train_ds, batch_size=S1_BATCH, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=S1_BATCH, shuffle=False, num_workers=0)
    print(f"  S1 datasets: train={len(train_ds)}, val={len(val_ds)}")

    model_s1 = KalmanForecaster(use_kappa=True).to(device)
    train_model(model_s1, train_ld, val_ld,
                S1_EPOCHS, S1_PATIENCE, S1_LR, S1_SCHED,
                tag=f"S1-{arm_name}")
    s1_p = model_s1.param_summary()
    print(f"  S1: alpha={s1_p['alpha']:.4f} kappa={s1_p['kappa']:.4f} "
          f"c={s1_p['c']:.4f} vc={s1_p['vc']:.4f}")

    # -- S2 --
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds2 = AlignmentTestDataset(csv_train, S2_L, S2_H, convention)
    val_ds2   = AlignmentTestDataset(csv_val,   S2_L, S2_H, convention)
    train_ld2 = DataLoader(train_ds2, batch_size=S2_BATCH, shuffle=True, num_workers=0)
    val_ld2   = DataLoader(val_ds2,   batch_size=S2_BATCH, shuffle=False, num_workers=0)

    model_s2 = KalmanForecasterClosure(
        alpha_init=max(s1_p['alpha'], 1e-6),
        c_init=max(s1_p['c'], 0.01),
        vc_init=s1_p['vc'],
        kappa_init=max(s1_p['kappa'], 0.001),
        log_qx_init=math.log(max(s1_p['qx'], 1e-15)),
        log_qu_init=math.log(max(s1_p['qu'], 1e-15)),
        log_r_init=math.log(max(s1_p['R'], 1e-15)),
        log_p0_xx_init=math.log(max(s1_p['P0_xx'], 1e-15)),
        log_p0_uu_init=math.log(max(s1_p['P0_uu'], 1e-15)),
        a1_init=0.001, b1_init=0.0, b2_init=0.0,
        d1_init=0.001, d2_init=0.001, d3_init=0.001,
        alpha_param="softplus",
    ).to(device)
    model_s2.freeze_physics()
    with torch.no_grad():
        model_s2.a1_raw.fill_(-10.0); model_s2.a1_raw.requires_grad_(False)
        model_s2.b1.fill_(0.0); model_s2.b1.requires_grad_(False)
        model_s2.d1_raw.fill_(-10.0); model_s2.d1_raw.requires_grad_(False)
        model_s2.d3_raw.fill_(-10.0); model_s2.d3_raw.requires_grad_(False)

    s2_trainable = lambda: [model_s2.b2, model_s2.d2_raw, model_s2.log_q_scale]
    train_model(model_s2, train_ld2, val_ld2,
                S2_EPOCHS, S2_PATIENCE, S2_LR, S2_SCHED,
                param_getter=s2_trainable, tag=f"S2-{arm_name}")

    s2_p = model_s2.param_summary()
    cl = model_s2.closure_summary()
    print(f"  S2: b2={cl['b2']:.4f} d2={cl['d2']:.4f} q_scale={cl['q_scale']:.4f}")

    # -- EVAL (numpy KF, always correct start-of-interval) --
    warmup_start = df_val['timestamp'].iloc[-1] - WARMUP_SEC
    df_warmup = df_val[df_val['timestamp'] >= warmup_start].copy()
    df_warm = pd.concat([df_warmup, df_test], ignore_index=True)
    t_arr = df_warm['timestamp'].values.astype(np.float64)
    x_arr = df_warm['displacement'].values.astype(np.float64)
    v_arr = df_warm['velocity'].values.astype(np.float64)

    test_start_time = df_test['timestamp'].iloc[0]
    eval_start = int(np.searchsorted(t_arr, test_start_time))

    # Physics-only params
    phys_pp = {
        'alpha': s1_p['alpha'], 'c': s1_p['c'], 'vc': s1_p['vc'],
        'kappa': s1_p['kappa'], 'qx': s1_p['qx'], 'qu': s1_p['qu'],
        'R': s1_p['R'], 'P0_xx': s1_p['P0_xx'], 'P0_uu': s1_p['P0_uu'],
    }
    phys_cl = {'b2': 0.0, 'd2': 0.0, 'q_scale': 1.0}

    # Closure params
    clos_pp = {
        'alpha': s2_p['alpha'], 'c': s2_p['c'], 'vc': s2_p['vc'],
        'kappa': s2_p['kappa'], 'qx': s2_p['qx'], 'qu': s2_p['qu'],
        'R': s2_p['R'], 'P0_xx': s2_p['P0_xx'], 'P0_uu': s2_p['P0_uu'],
    }
    clos_cl = {'b2': cl['b2'], 'd2': cl['d2'], 'q_scale': cl['q_scale']}

    # Filter
    filt_phys = kf_filter_2state(phys_pp, phys_cl, t_arr, x_arr, v_arr)
    filt_clos = kf_filter_2state(clos_pp, clos_cl, t_arr, x_arr, v_arr)

    # ACF
    innov_phys = filt_phys['innovations'][eval_start:]
    innov_clos = filt_clos['innovations'][eval_start:]
    innov_phys = innov_phys[~np.isnan(innov_phys)]
    innov_clos = innov_clos[~np.isnan(innov_clos)]
    acf_phys = compute_acf(innov_phys, max_lag=20)
    acf_clos = compute_acf(innov_clos, max_lag=20)

    # DxR2
    r2_phys = compute_dxr2(phys_pp, phys_cl,
                           filt_phys['states_x'], filt_phys['states_u'],
                           t_arr, x_arr, v_arr, MAX_H, eval_start)
    r2_clos = compute_dxr2(clos_pp, clos_cl,
                           filt_clos['states_x'], filt_clos['states_u'],
                           t_arr, x_arr, v_arr, MAX_H, eval_start)

    # NLL (quick approx from innovations)
    valid = ~np.isnan(filt_clos['innovations'])
    valid[0:eval_start] = False

    elapsed = time.time() - t0

    result = {
        'arm': arm_name,
        'convention': convention,
        'seed': seed,
        's1_alpha': float(s1_p['alpha']),
        's1_kappa': float(s1_p['kappa']),
        's1_c': float(s1_p['c']),
        's1_vc': float(s1_p['vc']),
        's2_b2': float(cl['b2']),
        's2_d2': float(cl['d2']),
        's2_q_scale': float(cl['q_scale']),
        'acf1_phys': float(acf_phys[1]),
        'acf1_clos': float(acf_clos[1]),
    }
    for h_idx in HEADLINE_H:
        h_label = f"{h_idx * DT:.1f}s"
        result[f'dxr2_phys_{h_label}'] = float(r2_phys[h_idx - 1]) if h_idx <= MAX_H else np.nan
        result[f'dxr2_clos_{h_label}'] = float(r2_clos[h_idx - 1]) if h_idx <= MAX_H else np.nan
        p_val = float(r2_phys[h_idx - 1]) if h_idx <= MAX_H else np.nan
        c_val_r2 = float(r2_clos[h_idx - 1]) if h_idx <= MAX_H else np.nan
        result[f'delta_dxr2_{h_label}'] = c_val_r2 - p_val

    result['runtime_s'] = elapsed
    return result


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("  97_ab_alignment_impact.py  --  A/B forcing-alignment test")
    print("=" * 70)

    # Load data
    csv_train = str(CLEAN_DIR / "train_10hz_ready.csv")
    csv_val   = str(CLEAN_DIR / "val_10hz_ready.csv")
    csv_test  = str(CLEAN_DIR / "test_10hz_ready.csv")

    for p in [csv_train, csv_val, csv_test]:
        assert os.path.exists(p), f"Missing: {p}"

    df_val  = pd.read_csv(csv_val).sort_values('timestamp').reset_index(drop=True)
    df_test = pd.read_csv(csv_test).sort_values('timestamp').reset_index(drop=True)

    all_results = []
    for seed in SEEDS:
        for conv, arm_label in [('old', f'OLD_s{seed}'), ('new', f'NEW_s{seed}')]:
            r = run_arm(arm_label, conv, seed, csv_train, csv_val, df_val, df_test)
            all_results.append(r)
            print(f"\n  >> {arm_label}: ACF1_phys={r['acf1_phys']:.4f} "
                  f"ACF1_clos={r['acf1_clos']:.4f}")
            for h_label in ['0.5s', '1.0s', '2.0s']:
                print(f"     DxR2@{h_label}: phys={r[f'dxr2_phys_{h_label}']:.4f} "
                      f"clos={r[f'dxr2_clos_{h_label}']:.4f} "
                      f"delta={r[f'delta_dxr2_{h_label}']:.4f}")

    # Save CSV
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(OUT / "ab_summary.csv", index=False)
    print(f"\n  Saved: {OUT / 'ab_summary.csv'}")

    # ================================================================
    #  COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("  A/B COMPARISON")
    print("=" * 70)

    lines = []
    lines.append("# A/B Alignment Impact Report\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Seeds:** {SEEDS}")
    lines.append(f"**Training:** S1={S1_EPOCHS}ep L={S1_L}, S2={S2_EPOCHS}ep L={S2_L}\n")

    lines.append("## Parameter Comparison\n")
    lines.append("| Seed | Arm | alpha | kappa | c | vc | b2 | d2 | q_scale |")
    lines.append("|------|-----|-------|-------|---|----|----|----|---------| ")

    for r in all_results:
        lines.append(
            f"| {r['seed']} | {r['convention'].upper()} | "
            f"{r['s1_alpha']:.4f} | {r['s1_kappa']:.4f} | "
            f"{r['s1_c']:.2f} | {r['s1_vc']:.4f} | "
            f"{r['s2_b2']:.3f} | {r['s2_d2']:.3f} | {r['s2_q_scale']:.3f} |")

    lines.append("\n## Metric Comparison (numpy KF evaluation, always start-of-interval)\n")
    lines.append("| Seed | Arm | ACF1_phys | ACF1_clos | DxR2_phys@1s | DxR2_clos@1s | Delta@1s |")
    lines.append("|------|-----|-----------|-----------|--------------|--------------|----------|")

    for r in all_results:
        lines.append(
            f"| {r['seed']} | {r['convention'].upper()} | "
            f"{r['acf1_phys']:.4f} | {r['acf1_clos']:.4f} | "
            f"{r['dxr2_phys_1.0s']:.4f} | {r['dxr2_clos_1.0s']:.4f} | "
            f"{r['delta_dxr2_1.0s']:.4f} |")

    # Compute deltas between OLD and NEW for each seed
    lines.append("\n## Impact Deltas (NEW - OLD)\n")
    lines.append("| Seed | d(alpha) | d(kappa) | d(b2) | d(d2) | d(ACF1_clos) | d(DxR2@1s) | d(Delta@1s) |")
    lines.append("|------|----------|----------|-------|-------|--------------|------------|-------------|")

    max_delta_metric = 0.0
    for seed in SEEDS:
        old = [r for r in all_results if r['seed'] == seed and r['convention'] == 'old'][0]
        new = [r for r in all_results if r['seed'] == seed and r['convention'] == 'new'][0]

        d_alpha = new['s1_alpha'] - old['s1_alpha']
        d_kappa = new['s1_kappa'] - old['s1_kappa']
        d_b2 = new['s2_b2'] - old['s2_b2']
        d_d2 = new['s2_d2'] - old['s2_d2']
        d_acf1 = new['acf1_clos'] - old['acf1_clos']
        d_r2_1s = new['dxr2_clos_1.0s'] - old['dxr2_clos_1.0s']
        d_delta_1s = new['delta_dxr2_1.0s'] - old['delta_dxr2_1.0s']

        max_delta_metric = max(max_delta_metric, abs(d_delta_1s))

        lines.append(
            f"| {seed} | {d_alpha:+.4f} | {d_kappa:+.4f} | "
            f"{d_b2:+.3f} | {d_d2:+.3f} | "
            f"{d_acf1:+.4f} | {d_r2_1s:+.4f} | {d_delta_1s:+.4f} |")

    # Verdict
    threshold = 0.02
    if max_delta_metric < threshold:
        verdict = "SMALL"
        verdict_detail = (
            f"Max |d(Delta DxR2@1s)| = {max_delta_metric:.4f} < {threshold}. "
            f"Impact is small. Rerun for rigor but no panic.")
    else:
        verdict = "MATERIAL"
        verdict_detail = (
            f"Max |d(Delta DxR2@1s)| = {max_delta_metric:.4f} >= {threshold}. "
            f"Impact is material. Full rerun required.")

    lines.append(f"\n## Verdict: **{verdict}**\n")
    lines.append(verdict_detail)
    lines.append(f"\n**Threshold:** |d(Delta DxR2@1s)| < {threshold}")
    lines.append(f"**Max observed:** {max_delta_metric:.4f}")

    lines.append("\n## Interpretation\n")
    lines.append("- The alignment fix changes `v_fut[0]` from `v[t+1]` to `v[t]` in the "
                 "PyTorch training dataset.")
    lines.append("- All evaluation metrics are computed by standalone numpy KF loops "
                 "that always used the correct `v[k-1]` convention.")
    lines.append("- Therefore the question is: does the 1-step lookahead in v_fut "
                 "during training materially change the learned parameters?")
    lines.append("- If Delta DxR2 (closure improvement over physics) is stable across "
                 "OLD vs NEW, the relative story holds and the fix is a correctness "
                 "improvement without changing conclusions.")

    report = "\n".join(lines)
    (OUT / "ab_report.md").write_text(report, encoding='utf-8')
    print(report)
    print(f"\n  Saved: {OUT / 'ab_report.md'}")

    # Summary JSON
    summary = {
        'verdict': verdict,
        'max_delta_metric': max_delta_metric,
        'threshold': threshold,
        'seeds': SEEDS,
        'n_arms': len(all_results),
        'runtime_total_s': time.time() - t_start,
    }
    with open(OUT / "ab_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total runtime: {time.time() - t_start:.0f}s")
    print(f"  VERDICT: {verdict}")


if __name__ == '__main__':
    main()
