"""
Phase 4: Run full discovery pipeline (S1 physics + S2 6-term closure + automated
term selection) on each Phase-3 virtual-lab condition.  Then compute oracle gap
ratio on the truth acceleration (FREE test indices).

S1 uses PyTorch NLL training (KalmanForecaster).
S2 uses scipy L-BFGS-B on the numpy KF filter NLL (much faster than PyTorch
autograd for the closure model).

Usage:  python -u virtual_lab_v1/scripts/23_run_discovery_v2.py
Output: virtual_lab_v1/outputs/phase4_discovery/
"""

import os, sys, math, json, time, warnings, tempfile, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent.parent
VL = ROOT / "virtual_lab_v1"
sys.path.insert(0, str(ROOT))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_forecaster import KalmanForecaster

# ==========================================================================
#  CONFIG
# ==========================================================================
FORCE_CPU = True
DT = 0.1
VAR_FLOOR = 1e-6
SEED = 42

# Stage 1 -- physics only (PyTorch)
S1_L = 32;  S1_H = 10;  S1_BATCH = 128
S1_EPOCHS = 60;  S1_LR = 1e-2;  S1_PATIENCE = 12;  S1_SCHED = 6

# Stage 2 -- full closure (scipy + numpy KF)
S2_MAXITER = 300

# Term selection
TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']
SELECTION_THRESHOLD = 0.05   # 5% relative variance contribution
NLL_DELTA_MIN = 0.001        # min NLL increase when term removed (null-test gate)

# Oracle gap
ORACLE_TRAIN_FRAC = 0.60
ORACLE_VAL_FRAC   = 0.20
N_MIN_FREE = 5000
GAP_EPS = 1e-12

# Paths
DATASETS = VL / "datasets_v2"
ORACLE_CSV = VL / "outputs" / "sweep_v2" / "oracle_summary_v2.csv"
OUT = VL / "outputs" / "phase4_discovery"


# ==========================================================================
#  HELPERS
# ==========================================================================

def gaussian_nll_torch(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def compute_a_true_numdiff(v_p, dt, smooth_hz=10.0):
    """Smoothed numerical dv_p/dt (same as oracle eval fallback)."""
    fs = 1.0 / dt
    if fs <= 2 * smooth_hz:
        return np.gradient(v_p, dt)
    sos = butter(4, smooth_hz, fs=fs, output='sos')
    v_sm = sosfiltfilt(sos, v_p)
    return np.gradient(v_sm, dt)


# ==========================================================================
#  NUMPY 2-STATE KF FILTER
# ==========================================================================

def kf_filter_2state(params, cl_params, t, x_obs, v):
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values    = np.full(N, np.nan)
    states_x    = np.zeros(N)
    states_u    = np.zeros(N)

    alpha = params['alpha'];  c_val = params['c']
    vc    = params['vc'];     kap   = params['kappa']
    qx    = params['qx'];    qu    = params['qu']
    q_sc  = cl_params.get('q_scale', 1.0)
    R     = params['R']
    a1_v  = cl_params.get('a1', 0.0)
    b1_v  = cl_params.get('b1', 0.0)
    b2_v  = cl_params.get('b2', 0.0)
    d1_v  = cl_params.get('d1', 0.0)
    d2_v  = cl_params.get('d2', 0.0)
    d3_v  = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]

    for k in range(1, N):
        dt_k = t[k] - t[k-1]
        if dt_k <= 0: dt_k = 0.1
        rho = math.exp(-alpha * dt_k)
        g   = max(v[k-1]**2 - vc**2, 0.0)
        physics_drift = rho * s[1] - kap * s[0] * dt_k + c_val * g * dt_k

        u_st = s[1]; v_w = v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = (-a1_v*u_st + b1_v*v_w + b2_v*dv_w
              - d1_v*u_st**2 - d2_v*u_st*abs(v_w) - d3_v*u_st*abs(u_st))
        cl_d = cl * dt_k

        x_p = s[0] + s[1] * dt_k
        u_p = physics_drift + cl_d
        s_pred = np.array([x_p, u_p])

        F_mat = np.array([[1, dt_k], [-kap*dt_k, rho]])
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
    """Compute Gaussian NLL from innovations and S values."""
    valid = ~np.isnan(innov) & (S_vals > 0)
    e = innov[valid]
    S = np.maximum(S_vals[valid], 1e-12)
    return float(0.5 * np.mean(np.log(2 * math.pi * S) + e**2 / S))


# ==========================================================================
#  S1: PYTORCH TRAINING
# ==========================================================================

def train_s1(model, train_loader, val_loader, device,
             max_epochs, patience, lr, sched_patience, tag=""):
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"    [{tag}] {n_params} trainable params")

    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=sched_patience)

    best_loss, best_state, best_ep, wait = float('inf'), None, 0, 0

    for ep in range(max_epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            v_h = v_h.to(device); dt_h = dt_h.to(device)
            x_h = x_h.to(device); v_f  = v_f.to(device)
            dt_f = dt_f.to(device); x_true = x_true.to(device)
            optimizer.zero_grad()
            xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
            loss = gaussian_nll_torch(xp, xv, x_true, VAR_FLOOR)
            loss.backward(); optimizer.step()
            tot += loss.item(); nb += 1
        tr_nll = tot / nb

        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for batch in val_loader:
                v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
                v_h = v_h.to(device); dt_h = dt_h.to(device)
                x_h = x_h.to(device); v_f  = v_f.to(device)
                dt_f = dt_f.to(device); x_true = x_true.to(device)
                xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
                vl = gaussian_nll_torch(xp, xv, x_true, VAR_FLOOR)
                vt += vl.item(); vn += 1
            val_nll = vt / vn

        scheduler.step(val_nll)
        if val_nll < best_loss:
            best_loss = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1; wait = 0
        else:
            wait += 1

        if (ep + 1) % 20 == 0 or ep == 0 or wait >= patience:
            print(f"    [{tag}] ep {ep+1:3d}  tr={tr_nll:.5f}  "
                  f"val={val_nll:.5f}  best={best_loss:.5f}@ep{best_ep}  "
                  f"wait={wait}")
        if wait >= patience:
            print(f"    [{tag}] Early stop at ep {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return best_loss, best_ep


# ==========================================================================
#  S2: SCIPY OPTIMIZATION OF CLOSURE PARAMS (NUMPY KF)
# ==========================================================================

def train_s2_scipy(s1_pp, t_train, x_train, v_train, t_val, x_val, v_val,
                   tag="S2"):
    """
    Optimize 7 closure params (a1, b1, b2, d1, d2, d3, log_q_scale) by
    minimizing the Gaussian NLL of the numpy KF filter on training data.

    Uses L-BFGS-B with bounds: a1,d1,d2,d3 >= 0; b1,b2,log_q_scale free.
    Returns (best_cl_params_dict, train_nll, val_nll, n_feval).
    """
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

    # Initial: all closure terms zero, q_scale=1 (log=0)
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Bounds: a1,d1,d2,d3 >= 0
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

    # Evaluate on val
    filt_val = kf_filter_2state(s1_pp, cl_best, t_val, x_val, v_val)
    val_nll = kf_nll_numpy(filt_val['innovations'], filt_val['S_values'])

    elapsed = time.time() - t0
    print(f"    [{tag}] L-BFGS-B done: {n_eval[0]} fevals, "
          f"train_nll={train_nll:.5f}, val_nll={val_nll:.5f}, "
          f"converged={res.success} [{elapsed:.1f}s]")

    return cl_best, train_nll, val_nll, n_eval[0]


# ==========================================================================
#  TERM SELECTION: variance-based
# ==========================================================================

def select_terms(cl_params, s1_pp, t, x_obs, v):
    """
    Run numpy KF filter, compute per-term contribution variance on second
    half, select terms with relative variance > SELECTION_THRESHOLD.
    Then apply delta-NLL gate: only keep terms whose removal worsens
    the validation NLL by at least NLL_DELTA_MIN.
    """
    N = len(x_obs)
    filt = kf_filter_2state(s1_pp, cl_params, t, x_obs, v)
    su = filt['states_u']

    start = N // 2
    contribs = {tn: np.zeros(N - start) for tn in TERM_NAMES}

    for i, k in enumerate(range(start, N)):
        # Match kf_filter_2state timing: closure at step k uses k-1 values
        u_st = su[k-1]
        v_w  = v[k-1]
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

    # Delta-NLL gate: keep only terms whose removal materially worsens NLL
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


# ==========================================================================
#  ACF(1) HELPER
# ==========================================================================

def acf1(innov):
    e = innov[~np.isnan(innov)]
    if len(e) < 10: return np.nan
    ec = e - np.mean(e)
    var = np.var(e)
    if var < 1e-15: return 0.0
    return float(np.sum(ec[:-1]*ec[1:]) / (len(e)*var))


# ==========================================================================
#  ORACLE GAP EVALUATION
# ==========================================================================

def compute_oracle_gap(cond_dir, disc_physics, disc_closure, oracle_row):
    """Evaluate discovered model's acceleration on FREE test indices."""
    truth_path = cond_dir / "truth_states_raw.csv"
    meta_path  = cond_dir / "meta.json"
    if not truth_path.exists() or not meta_path.exists():
        return {'gap_ratio': np.nan, 'MSE_disc': np.nan, 'gap_status': 'no_data'}

    with open(meta_path) as f:
        meta = json.load(f)
    dt_sim = meta['config']['integration']['dt_sim']
    df = pd.read_csv(truth_path)

    v_p    = df['v_p'].values
    x      = df['x'].values
    at_pin = df['at_pin'].values
    u_b    = df['u_b'].values
    du_b   = df['du_b'].values

    if 'a_force' in df.columns:
        a_true = df['a_force'].values
    else:
        a_true = compute_a_true_numdiff(v_p, dt_sim)

    is_free = (at_pin == 0)
    valid_a = ~np.isnan(a_true)
    free_idx = np.where(is_free & valid_a)[0]
    n_free = len(free_idx)

    if n_free < N_MIN_FREE:
        return {'gap_ratio': np.nan, 'MSE_disc': np.nan,
                'gap_status': 'degenerate', 'n_free': n_free}

    n_train = int(n_free * ORACLE_TRAIN_FRAC)
    n_val   = int(n_free * ORACLE_VAL_FRAC)
    idx_test = free_idx[n_train + n_val:]

    alpha_d = disc_physics['alpha']
    kappa_d = disc_physics['kappa']
    c_d     = disc_physics['c']
    vc_d    = disc_physics['vc']
    a1_d = disc_closure.get('a1', 0.0)
    b1_d = disc_closure.get('b1', 0.0)
    b2_d = disc_closure.get('b2', 0.0)
    d1_d = disc_closure.get('d1', 0.0)
    d2_d = disc_closure.get('d2', 0.0)
    d3_d = disc_closure.get('d3', 0.0)

    a_disc = np.zeros(len(idx_test))
    a_t    = np.zeros(len(idx_test))
    for i, k in enumerate(idx_test):
        vp_k = v_p[k]; x_k = x[k]; ub_k = u_b[k]; dub_k = du_b[k]
        forcing = max(ub_k**2 - vc_d**2, 0.0)
        a_phys = -alpha_d * vp_k - kappa_d * x_k + c_d * forcing
        cl = (-a1_d * vp_k + b1_d * ub_k + b2_d * dub_k
              - d1_d * vp_k**2 - d2_d * vp_k * abs(ub_k)
              - d3_d * vp_k * abs(vp_k))
        a_disc[i] = a_phys + cl
        a_t[i] = a_true[k]

    MSE_disc = float(np.mean((a_t - a_disc)**2))
    MSE_phys   = oracle_row['MSE_phys']
    MSE_oracle = oracle_row['MSE_oracle']

    denom = MSE_phys - MSE_oracle
    if denom <= GAP_EPS:
        return {'gap_ratio': np.nan, 'MSE_disc': MSE_disc,
                'MSE_phys_oracle': float(MSE_phys),
                'MSE_oracle_oracle': float(MSE_oracle),
                'n_test_free': len(idx_test), 'gap_status': 'no_residual'}
    gap_ratio = float((MSE_disc - MSE_oracle) / denom)
    return {'gap_ratio': gap_ratio, 'MSE_disc': MSE_disc,
            'MSE_phys_oracle': float(MSE_phys),
            'MSE_oracle_oracle': float(MSE_oracle),
            'n_test_free': len(idx_test), 'gap_status': 'ok'}


# ==========================================================================
#  SINGLE CONDITION PIPELINE
# ==========================================================================

def run_condition(cond_id, cond_dir, oracle_row, device, out_dir):
    """S1 (PyTorch) + S2 (scipy) + term selection + oracle gap."""
    result_path = out_dir / f"{cond_id}.json"
    if result_path.exists():
        print(f"  [{cond_id}] Already done, loading cached result")
        with open(result_path) as f:
            return json.load(f)

    t0 = time.time()
    x10_path = cond_dir / "x_10hz.csv"
    if not x10_path.exists():
        result = {'condition_id': cond_id, 'status': 'no_data'}
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        return result

    # ---- Load and split 10 Hz data ----
    df = pd.read_csv(x10_path)
    N = len(df)
    n_train = int(0.60 * N)
    n_val   = int(0.20 * N)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    df_test  = df.iloc[n_train+n_val:].reset_index(drop=True)

    # Numpy arrays for S2 / eval
    t_train = df_train['timestamp'].values
    x_train = df_train['displacement'].values
    v_train = df_train['velocity'].values
    t_val = df_val['timestamp'].values
    x_val = df_val['displacement'].values
    v_val = df_val['velocity'].values
    t_test = df_test['timestamp'].values
    x_test = df_test['displacement'].values
    v_test = df_test['velocity'].values

    # Write temp CSVs for S1 StateSpaceDataset
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"disc_{cond_id}_"))
    try:
        for name, d in [('train', df_train), ('val', df_val)]:
            d.to_csv(tmp_dir / f"{name}.csv", index=False)

        torch.manual_seed(SEED); np.random.seed(SEED)

        # ---- S1: Physics only (PyTorch) ----
        print(f"  [{cond_id}] S1 Physics: N={N}, train={n_train}, val={n_val}")
        train_ds = StateSpaceDataset(
            [str(tmp_dir / "train.csv")], L=S1_L, m=S1_L, H=S1_H,
            predict_deltas=False, normalize=False)
        val_ds = StateSpaceDataset(
            [str(tmp_dir / "val.csv")], L=S1_L, m=S1_L, H=S1_H,
            predict_deltas=False, normalize=False)

        if len(train_ds) < 10 or len(val_ds) < 5:
            result = {'condition_id': cond_id, 'status': 'too_few_samples',
                      'n_train_samples': len(train_ds), 'n_val_samples': len(val_ds)}
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            return result

        train_ld = DataLoader(train_ds, batch_size=S1_BATCH, shuffle=True, num_workers=0)
        val_ld   = DataLoader(val_ds, batch_size=S1_BATCH, shuffle=False, num_workers=0)

        model_s1 = KalmanForecaster(use_kappa=True).to(device)
        s1_best_val, s1_best_ep = train_s1(
            model_s1, train_ld, val_ld, device,
            S1_EPOCHS, S1_PATIENCE, S1_LR, S1_SCHED, tag=f"S1-{cond_id}")

        s1_params = model_s1.param_summary()
        print(f"  [{cond_id}] S1: alpha={s1_params['alpha']:.4f} "
              f"kappa={s1_params['kappa']:.4f} c={s1_params['c']:.4f} "
              f"vc={s1_params['vc']:.4f}")

        # Build physics-param dict for numpy filter
        s1_pp = {
            'alpha': s1_params['alpha'], 'c': s1_params['c'],
            'vc': s1_params['vc'], 'kappa': s1_params['kappa'],
            'qx': s1_params['qx'], 'qu': s1_params['qu'],
            'R': s1_params['R'],
            'P0_xx': s1_params['P0_xx'], 'P0_uu': s1_params['P0_uu'],
        }

        # ---- S2: Full 6-term closure (scipy + numpy KF) ----
        print(f"  [{cond_id}] S2 Full 6-term closure (scipy L-BFGS-B)")
        cl_best, s2_train_nll, s2_val_nll, s2_nfeval = train_s2_scipy(
            s1_pp, t_train, x_train, v_train,
            t_val, x_val, v_val, tag=f"S2-{cond_id}")

        print(f"  [{cond_id}] S2: a1={cl_best['a1']:.4f} b1={cl_best['b1']:.4f} "
              f"b2={cl_best['b2']:.4f} d1={cl_best['d1']:.4f} "
              f"d2={cl_best['d2']:.4f} d3={cl_best['d3']:.4f} "
              f"q_scale={cl_best['q_scale']:.4f}")

        # ---- Term selection ----
        selected_terms, term_vars, rel_vars = select_terms(
            cl_best, s1_pp, t_val, x_val, v_val)

        print(f"  [{cond_id}] Selected terms: {selected_terms}")
        print(f"  [{cond_id}] Relative variances: " +
              ", ".join(f"{tn}={rel_vars.get(tn,0):.3f}" for tn in TERM_NAMES))

        # ---- KF evaluation on 10Hz test data ----
        phys_cl = {k: 0.0 for k in TERM_NAMES}
        phys_cl['q_scale'] = 1.0

        filt_phys = kf_filter_2state(s1_pp, phys_cl, t_test, x_test, v_test)
        filt_clos = kf_filter_2state(s1_pp, cl_best, t_test, x_test, v_test)

        phys_acf1 = acf1(filt_phys['innovations'])
        clos_acf1 = acf1(filt_clos['innovations'])
        phys_test_nll = kf_nll_numpy(filt_phys['innovations'], filt_phys['S_values'])
        clos_test_nll = kf_nll_numpy(filt_clos['innovations'], filt_clos['S_values'])

        # ---- Oracle gap ratio ----
        disc_physics = {
            'alpha': s1_params['alpha'], 'kappa': s1_params['kappa'],
            'c': s1_params['c'], 'vc': s1_params['vc'],
        }
        disc_closure = {tn: cl_best[tn] for tn in TERM_NAMES}
        gap_result = compute_oracle_gap(cond_dir, disc_physics, disc_closure, oracle_row)

        elapsed = time.time() - t0
        gr = gap_result.get('gap_ratio', np.nan)
        gr_str = f"{gr:.3f}" if np.isfinite(gr) else "N/A"
        print(f"  [{cond_id}] ACF1 phys={phys_acf1:.4f} clos={clos_acf1:.4f} "
              f"gap_ratio={gr_str} [{elapsed:.0f}s]")

        # ---- Assemble result ----
        result = {
            'condition_id': cond_id,
            'status': 'ok',
            'n_10hz': N,
            # S1
            's1_alpha': s1_params['alpha'],
            's1_kappa': s1_params['kappa'],
            's1_c': s1_params['c'],
            's1_vc': s1_params['vc'],
            's1_best_val': s1_best_val,
            's1_best_ep': s1_best_ep,
            # S2 closure coefficients
            's2_a1': cl_best['a1'],
            's2_b1': cl_best['b1'],
            's2_b2': cl_best['b2'],
            's2_d1': cl_best['d1'],
            's2_d2': cl_best['d2'],
            's2_d3': cl_best['d3'],
            's2_q_scale': cl_best['q_scale'],
            's2_train_nll': s2_train_nll,
            's2_val_nll': s2_val_nll,
            's2_nfeval': s2_nfeval,
            # Term selection
            'selected_terms': selected_terms,
            'n_selected': len(selected_terms),
            'rel_var_a1': rel_vars.get('a1', 0.0),
            'rel_var_b1': rel_vars.get('b1', 0.0),
            'rel_var_b2': rel_vars.get('b2', 0.0),
            'rel_var_d1': rel_vars.get('d1', 0.0),
            'rel_var_d2': rel_vars.get('d2', 0.0),
            'rel_var_d3': rel_vars.get('d3', 0.0),
            # KF metrics on 10Hz test
            'phys_acf1': phys_acf1,
            'clos_acf1': clos_acf1,
            'phys_test_nll': phys_test_nll,
            'clos_test_nll': clos_test_nll,
            # Oracle gap
            'gap_ratio': gap_result.get('gap_ratio', np.nan),
            'MSE_disc': gap_result.get('MSE_disc', np.nan),
            'gap_status': gap_result.get('gap_status', 'unknown'),
            'MSE_phys_oracle': gap_result.get('MSE_phys_oracle', np.nan),
            'MSE_oracle_oracle': gap_result.get('MSE_oracle_oracle', np.nan),
            'oracle_regime': str(oracle_row.get('regime', '')),
            'event_rate': float(oracle_row.get('event_rate', np.nan)),
            'oracle_gain': float(oracle_row.get('gain_oracle', np.nan)),
            'oracle_status': str(oracle_row.get('status', '')),
            # Timing
            'elapsed_s': round(elapsed, 1),
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Handle NaN for JSON
    def sanitize(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    result_clean = {k: sanitize(v) if not isinstance(v, list) else v
                    for k, v in result.items()}

    with open(result_path, 'w') as f:
        json.dump(result_clean, f, indent=2)

    return result


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    t0_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu') if FORCE_CPU else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load oracle summary
    if not ORACLE_CSV.exists():
        print(f"ERROR: {ORACLE_CSV} not found. Run oracle eval first.")
        return
    df_oracle = pd.read_csv(ORACLE_CSV)
    oracle_by_id = {row['condition_id']: row.to_dict()
                    for _, row in df_oracle.iterrows()}
    print(f"Loaded oracle summary: {len(df_oracle)} conditions")

    # Discover conditions
    cond_dirs = sorted(DATASETS.glob("condition_*"))
    print(f"Found {len(cond_dirs)} condition directories")

    all_results = []
    for cond_dir in cond_dirs:
        cond_id = cond_dir.name
        oracle_row = oracle_by_id.get(cond_id)

        if oracle_row is None:
            print(f"\n  [{cond_id}] No oracle row, skipping")
            continue

        print_section(f"CONDITION: {cond_id}")
        result = run_condition(cond_id, cond_dir, oracle_row, device, OUT)
        all_results.append(result)

    # ---- Write discovery_summary.csv ----
    csv_cols = [
        'condition_id', 'status', 'event_rate', 'oracle_regime',
        's1_alpha', 's1_kappa', 's1_c', 's1_vc',
        's2_a1', 's2_b1', 's2_b2', 's2_d1', 's2_d2', 's2_d3', 's2_q_scale',
        'selected_terms', 'n_selected',
        'rel_var_a1', 'rel_var_b1', 'rel_var_b2',
        'rel_var_d1', 'rel_var_d2', 'rel_var_d3',
        'phys_acf1', 'clos_acf1',
        'gap_ratio', 'MSE_disc', 'gap_status',
        'MSE_phys_oracle', 'MSE_oracle_oracle', 'oracle_gain', 'oracle_status',
        'elapsed_s',
    ]
    rows = []
    for r in all_results:
        row = {}
        for c in csv_cols:
            val = r.get(c)
            if isinstance(val, list):
                val = '+'.join(val)
            row[c] = val
        rows.append(row)

    df_out = pd.DataFrame(rows, columns=csv_cols)
    csv_path = OUT / "discovery_summary.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # ---- Summary table ----
    ok_results = [r for r in all_results if r.get('status') == 'ok']
    print_section("DISCOVERY SUMMARY")
    print(f"Total conditions: {len(all_results)}")
    print(f"OK: {len(ok_results)}, Failed: {len(all_results) - len(ok_results)}")

    if ok_results:
        from collections import Counter
        term_counts = Counter()
        for r in ok_results:
            for t in r.get('selected_terms', []):
                term_counts[t] += 1
        n_ok = len(ok_results)
        print(f"\nTerm selection frequency (n={n_ok}):")
        for tn in TERM_NAMES:
            cnt = term_counts.get(tn, 0)
            print(f"  {tn}: {cnt}/{n_ok} ({100*cnt/n_ok:.0f}%)")

        inter = [r for r in ok_results if r.get('oracle_regime') == 'intermittent']
        if inter:
            gaps = [r['gap_ratio'] for r in inter
                    if r.get('gap_ratio') is not None
                    and not math.isnan(r.get('gap_ratio', float('nan')))]
            if gaps:
                print(f"\nIntermittent gap_ratio (n={len(gaps)}): "
                      f"median={np.median(gaps):.3f}, "
                      f"mean={np.mean(gaps):.3f}, "
                      f"IQR=[{np.percentile(gaps,25):.3f}, {np.percentile(gaps,75):.3f}]")

            term_counts_inter = Counter()
            for r in inter:
                for t in r.get('selected_terms', []):
                    term_counts_inter[t] += 1
            n_i = len(inter)
            print(f"\nIntermittent term frequency (n={n_i}):")
            for tn in TERM_NAMES:
                cnt = term_counts_inter.get(tn, 0)
                print(f"  {tn}: {cnt}/{n_i} ({100*cnt/n_i:.0f}%)")

        cont = [r for r in ok_results if r.get('oracle_regime') == 'continuous']
        if cont:
            gaps_c = [r['gap_ratio'] for r in cont
                      if r.get('gap_ratio') is not None
                      and not math.isnan(r.get('gap_ratio', float('nan')))]
            if gaps_c:
                print(f"\nContinuous gap_ratio (n={len(gaps_c)}): "
                      f"median={np.median(gaps_c):.3f}")

    elapsed_total = time.time() - t0_all
    print(f"\nTotal elapsed: {elapsed_total/60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
