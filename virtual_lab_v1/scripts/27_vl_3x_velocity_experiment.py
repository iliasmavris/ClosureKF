"""
27_vl_3x_velocity_experiment.py
===============================
Run the full VL pipeline (Phase 3 sweep + Phase 4 discovery) with ALL
forcing variants scaled by 3x.  Tests whether stronger flow with
d_p=0.03 produces detectable closure terms.

Physics note: 3x velocity -> ~9x drag.  mu_s needs to increase to keep
ER in [0.05, 0.40].  Auto-tune handles this.

Outputs:
  virtual_lab_v1/outputs/sweep_v2_3x/       (sweep + oracle results)
  virtual_lab_v1/datasets_v2_3x/            (condition datasets)
  virtual_lab_v1/outputs/phase4_discovery_3x/ (discovery results)

Usage:
  python -u virtual_lab_v1/scripts/27_vl_3x_velocity_experiment.py
"""

import os, sys, math, json, time, copy, warnings, tempfile, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_num_threads(os.cpu_count() or 4)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
VL   = ROOT / "virtual_lab_v1"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from truth_ball_sim_lib import (simulate_sphere, compute_event_rate,
                                compute_derived_params, compute_pin_statistics)
from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_forecaster import KalmanForecaster

# Oracle eval import
import importlib.util as _ilu
_oracle_spec = _ilu.spec_from_file_location(
    "oracle_eval", SCRIPT_DIR / "18_oracle_eval.py")
_oracle_mod = _ilu.module_from_spec(_oracle_spec)
_oracle_spec.loader.exec_module(_oracle_mod)
oracle_eval_condition = _oracle_mod.evaluate_condition
ORACLE_TERM_NAMES = _oracle_mod.TERM_NAMES

# ==========================================================================
#  CONFIG
# ==========================================================================
VELOCITY_SCALE = 3.0

# Physics params (d_p=0.03, same as ball_params_default.yaml)
import yaml
with open(VL / "configs" / "ball_params_default.yaml") as f:
    BASE_CFG = yaml.safe_load(f)

# Sweep config
FORCING_VARIANTS = [
    "baseline", "scale_093", "scale_097", "scale_103", "scale_107",
    "scale_112", "am_a010_tau3", "am_a015_tau4", "am_a020_tau2",
    "am_a020_tau6", "scale097_am010_tau3", "scale103_am010_tau4",
]
SEEDS_PER_VARIANT = 2
SEED_BASE = 300   # offset from Phase 3 (200) to avoid overlap

# Auto-tune: wider mu_s range for 3x velocity
AUTOTUNE = {
    'target_er_low': 0.05,
    'target_er_high': 0.40,
    'mu_s_step': 0.05,       # larger steps for bigger range
    'mu_s_clamp': [0.01, 1.50],
    'max_iter': 20,
}
BASE_MU_S = 0.40   # starting guess (higher than 0.05 since drag is ~9x)

# Jitter (same as sweep_grid_v2.yaml)
JITTER = {
    'enabled': True,
    'k_spring': {'sigma_log': 0.12},
    'mu_k': {'sigma_log': 0.05},
}

# Discovery config
FORCE_CPU = True
DT = 0.1
VAR_FLOOR = 1e-6
DISC_SEED = 42
S1_L = 32; S1_H = 10; S1_BATCH = 128
S1_EPOCHS = 60; S1_LR = 1e-2; S1_PATIENCE = 12; S1_SCHED = 6
S2_MAXITER = 300
TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']
SELECTION_THRESHOLD = 0.05
NLL_DELTA_MIN = 0.001

# Paths
FORCING_SRC = VL / "outputs" / "sweep_v2" / "forcing"
OUT_SWEEP   = VL / "outputs" / "sweep_v2_3x"
DS_ROOT     = VL / "datasets_v2_3x"
OUT_DISC    = VL / "outputs" / "phase4_discovery_3x"


# ==========================================================================
#  PHASE 3 HELPERS (from 21_build_sweep_v2.py)
# ==========================================================================

def downsample_to_10hz(t_raw, signal_raw, cutoff_hz=4.0):
    fs_raw = 1.0 / np.median(np.diff(t_raw))
    sos = butter(4, cutoff_hz, btype='low', fs=fs_raw, output='sos')
    sig_filt = sosfiltfilt(sos, signal_raw)
    t_10hz = np.arange(t_raw[0], t_raw[-1], 0.1)
    sig_10hz = np.interp(t_10hz, t_raw, sig_filt)
    return t_10hz, sig_10hz


def apply_jitter(cfg, jitter_cfg, rng):
    cfg = copy.deepcopy(cfg)
    if not jitter_cfg.get('enabled', False):
        return cfg
    if 'k_spring' in jitter_cfg:
        sig = jitter_cfg['k_spring']['sigma_log']
        cfg['restoring']['k_spring'] *= float(rng.lognormal(0, sig))
    if 'mu_k' in jitter_cfg:
        sig = jitter_cfg['mu_k']['sigma_log']
        cfg['friction']['mu_k'] *= float(rng.lognormal(0, sig))
        gap = max(0.005, 0.3 * cfg['friction']['mu_s'])
        cfg['friction']['mu_k'] = min(cfg['friction']['mu_k'],
                                       cfg['friction']['mu_s'] - gap)
        cfg['friction']['mu_k'] = max(0.002, cfg['friction']['mu_k'])
    return cfg


def fullrun_autotune(cfg, t_flow, u_flow, seed, target_range, max_iter, step,
                     mu_s_clamp):
    cfg = copy.deepcopy(cfg)
    history = []
    spinup = cfg.get('integration', {}).get('spinup_discard', 30.0)
    lo_mu_s, hi_mu_s = None, None

    for iteration in range(max_iter):
        result = simulate_sphere(cfg, t_flow, u_flow, seed=seed)
        dt_sim = cfg['integration']['dt_sim']
        n_spinup = int(spinup / dt_sim)
        contact_post = result['contact'][n_spinup:]
        er = compute_event_rate(contact_post) if len(contact_post) > 0 else 0.0
        mu_s_current = cfg['friction']['mu_s']
        history.append({'iteration': iteration,
                        'mu_s': round(mu_s_current, 4),
                        'event_rate': round(float(er), 4)})

        if target_range[0] <= er <= target_range[1]:
            break

        if er < target_range[0]:
            hi_mu_s = mu_s_current
            new_mu_s = (lo_mu_s + hi_mu_s) / 2 if lo_mu_s is not None else mu_s_current - step
        else:
            lo_mu_s = mu_s_current
            new_mu_s = (lo_mu_s + hi_mu_s) / 2 if hi_mu_s is not None else mu_s_current + step

        new_mu_s = max(mu_s_clamp[0], min(mu_s_clamp[1], new_mu_s))
        if abs(new_mu_s - mu_s_current) < 1e-6:
            break
        cfg['friction']['mu_s'] = new_mu_s
        cfg['friction']['mu_k'] = min(cfg['friction']['mu_k'], 0.6 * new_mu_s)

    return cfg, history


def make_dataset(cfg, condition_id, seed, t_flow, u_flow, out_base):
    cond_dir = out_base / condition_id
    cond_dir.mkdir(parents=True, exist_ok=True)

    result = simulate_sphere(cfg, t_flow, u_flow, seed=seed)
    er = compute_event_rate(result['contact'])

    spinup = cfg['integration']['spinup_discard']
    mask = result['t'] >= spinup
    t_sim = result['t'][mask]
    x_sim = result['x'][mask]

    # Write truth_states_raw.csv
    truth_dict = {
        'time': result['t'][mask], 'x': result['x'][mask],
        'v_p': result['v_p'][mask], 'at_pin': result['at_pin'][mask],
        'eta': result['eta'][mask], 'u_b': result['u_b'][mask],
        'du_b': result['du_b'][mask],
    }
    if 'a_force' in result:
        truth_dict['a_force'] = result['a_force'][mask]
    pd.DataFrame(truth_dict).to_csv(cond_dir / "truth_states_raw.csv", index=False)

    # Downsample to 10Hz
    t_10hz, x_10hz = downsample_to_10hz(t_sim, x_sim)
    _, ub_10hz = downsample_to_10hz(t_sim, result['u_b'][mask])

    df_x10 = pd.DataFrame({
        'timestamp': t_10hz, 'time_delta': np.full(len(t_10hz), 0.1),
        'velocity': ub_10hz, 'displacement': x_10hz,
    })
    df_x10.to_csv(cond_dir / "x_10hz.csv", index=False)

    dx = np.diff(x_10hz)
    dt_sim = cfg['integration']['dt_sim']
    at_pin_post = result['at_pin'][mask]
    pin_stats = compute_pin_statistics(at_pin_post, dt_sim)
    derived = compute_derived_params(cfg)

    meta = {
        'condition_id': condition_id, 'seed': seed, 'config': cfg,
        'derived_params': {k: float(v) for k, v in derived.items()},
        'spinup_discard_s': spinup, 'event_rate': float(er),
        'stats_10hz': {
            'n_points': len(t_10hz),
            't_range': [float(t_10hz[0]), float(t_10hz[-1])],
            'u_mean': float(np.mean(ub_10hz)), 'u_std': float(np.std(ub_10hz)),
            'x_mean': float(np.mean(x_10hz)), 'x_std': float(np.std(x_10hz)),
            'dx_mean': float(np.mean(dx)), 'dx_std': float(np.std(dx)),
        },
    }
    with open(cond_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


# ==========================================================================
#  PHASE 4 HELPERS (from 23_run_discovery_v2.py)
# ==========================================================================

def gaussian_nll_torch(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def kf_filter_2state(params, cl_params, t, x_obs, v):
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_x = np.zeros(N)
    states_u = np.zeros(N)

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx = params['qx']; qu = params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    a1_v = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1_v = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0)
    d3_v = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]

    for k in range(1, N):
        dt_k = t[k] - t[k-1]
        if dt_k <= 0: dt_k = 0.1
        rho = math.exp(-alpha * dt_k)
        g = max(v[k-1]**2 - vc**2, 0.0)
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
    valid = ~np.isnan(innov) & (S_vals > 0)
    e = innov[valid]; S = np.maximum(S_vals[valid], 1e-12)
    return float(0.5 * np.mean(np.log(2 * math.pi * S) + e**2 / S))


def acf1(innov):
    e = innov[~np.isnan(innov)]
    if len(e) < 10: return np.nan
    ec = e - np.mean(e); var = np.var(e)
    if var < 1e-15: return 0.0
    return float(np.sum(ec[:-1]*ec[1:]) / (len(e)*var))


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
        model.train(); tot, nb = 0.0, 0
        for batch in train_loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            v_h = v_h.to(device); dt_h = dt_h.to(device)
            x_h = x_h.to(device); v_f = v_f.to(device)
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
                x_h = x_h.to(device); v_f = v_f.to(device)
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

    model.load_state_dict(best_state); model.eval()
    return best_loss, best_ep


def train_s2_scipy(s1_pp, t_train, x_train, v_train, t_val, x_val, v_val,
                   tag="S2"):
    n_eval = [0]
    def objective(cl_vec):
        n_eval[0] += 1
        a1, b1, b2, d1, d2, d3, log_qs = cl_vec
        cl = {'a1': a1, 'b1': b1, 'b2': b2,
              'd1': d1, 'd2': d2, 'd3': d3,
              'q_scale': math.exp(np.clip(log_qs, -10, 10))}
        filt = kf_filter_2state(s1_pp, cl, t_train, x_train, v_train)
        nll = kf_nll_numpy(filt['innovations'], filt['S_values'])
        return nll if np.isfinite(nll) else 1e10

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(0, None), (None, None), (None, None),
              (0, None), (0, None), (0, None), (-5, 5)]

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


def select_terms(cl_params, s1_pp, t, x_obs, v):
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
        if nll_without - base_nll >= NLL_DELTA_MIN:
            selected.append(tn)

    return selected, variances, rel_var


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    t0_all = time.time()
    print("=" * 70)
    print(f"VIRTUAL LAB: {VELOCITY_SCALE}x VELOCITY EXPERIMENT")
    print(f"d_p = {BASE_CFG['sphere']['d_p']} m, velocity scale = {VELOCITY_SCALE}x")
    print("=" * 70)

    device = torch.device('cpu') if FORCE_CPU else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Step 1: Create 3x forcing variants ----
    print("\n" + "=" * 70)
    print("STEP 1: Create 3x-scaled forcing variants")
    print("=" * 70)

    forcing_out = OUT_SWEEP / "forcing"
    forcing_out.mkdir(parents=True, exist_ok=True)

    forcing_cache = {}
    for fv_id in FORCING_VARIANTS:
        src_path = FORCING_SRC / f"variant_{fv_id}_u_raw.csv"
        if not src_path.exists():
            print(f"ERROR: Missing forcing variant: {src_path}")
            sys.exit(1)
        df_fv = pd.read_csv(src_path)
        t_flow = df_fv['time'].values
        u_flow = df_fv['u_b_variant'].values * VELOCITY_SCALE

        # Save scaled variant
        dst_path = forcing_out / f"variant_{fv_id}_u_raw.csv"
        pd.DataFrame({'time': t_flow, 'u_b_variant': u_flow}).to_csv(
            dst_path, index=False)

        forcing_cache[fv_id] = (t_flow, u_flow)
        print(f"  {fv_id}: mean={u_flow.mean():.4f}, std={u_flow.std():.4f}, "
              f"range=[{u_flow.min():.4f}, {u_flow.max():.4f}]")

    # ---- Step 2: Phase 3 sweep ----
    print("\n" + "=" * 70)
    print("STEP 2: Phase 3 sweep (auto-tune + data generation + oracle eval)")
    print("=" * 70)

    DS_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_SWEEP.mkdir(parents=True, exist_ok=True)

    conditions = []
    cond_idx = 0
    for fv_id in FORCING_VARIANTS:
        for s_offset in range(SEEDS_PER_VARIANT):
            seed = SEED_BASE + cond_idx
            conditions.append({
                'id': f"condition_{cond_idx:03d}",
                'forcing_variant': fv_id,
                'seed': seed,
            })
            cond_idx += 1

    n_cond = len(conditions)
    print(f"Conditions: {n_cond} ({len(FORCING_VARIANTS)} variants x {SEEDS_PER_VARIANT} seeds)")
    print(f"base_mu_s = {BASE_MU_S}, mu_s_clamp = {AUTOTUNE['mu_s_clamp']}")

    summary_rows = []
    oracle_results = []

    for i, cond in enumerate(conditions):
        cond_id = cond['id']
        fv_id = cond['forcing_variant']
        seed = cond['seed']

        print(f"\n{'='*60}")
        print(f"CONDITION {i+1}/{n_cond}: {cond_id} (forcing={fv_id}, seed={seed})")

        t_flow, u_flow = forcing_cache[fv_id]

        cfg = copy.deepcopy(BASE_CFG)
        cfg['friction']['mu_s'] = BASE_MU_S
        cfg['friction']['mu_k'] = 0.6 * BASE_MU_S  # scale mu_k with mu_s
        jitter_rng = np.random.default_rng(seed + 10000)
        cfg = apply_jitter(cfg, JITTER, jitter_rng)

        d_p_j = cfg['sphere']['d_p']
        k_spring_j = cfg['restoring']['k_spring']
        mu_k_j = cfg['friction']['mu_k']
        mu_s_nom = cfg['friction']['mu_s']
        print(f"  Jittered: d_p={d_p_j:.4f}, k_spring={k_spring_j:.4f}, "
              f"mu_k={mu_k_j:.4f}, mu_s={mu_s_nom:.4f}")

        # Auto-tune
        print(f"  Auto-tune (target ER [{AUTOTUNE['target_er_low']}, "
              f"{AUTOTUNE['target_er_high']}])...")
        t0 = time.time()
        cfg_tuned, at_hist = fullrun_autotune(
            cfg, t_flow, u_flow, seed=seed,
            target_range=(AUTOTUNE['target_er_low'], AUTOTUNE['target_er_high']),
            max_iter=AUTOTUNE['max_iter'],
            step=AUTOTUNE['mu_s_step'],
            mu_s_clamp=AUTOTUNE['mu_s_clamp'],
        )
        mu_s_final = cfg_tuned['friction']['mu_s']
        at_er = at_hist[-1]['event_rate'] if at_hist else 0.0
        print(f"  mu_s: {mu_s_nom:.4f} -> {mu_s_final:.4f} "
              f"(final ER={at_er:.3f}, {len(at_hist)} iters)")

        # Generate dataset
        meta = make_dataset(cfg_tuned, cond_id, seed, t_flow, u_flow, DS_ROOT)
        elapsed = time.time() - t0
        er = meta['event_rate']
        x_std = meta['stats_10hz']['x_std']
        print(f"  Dataset: ER={er:.3f}, x_std={x_std:.6f} m, "
              f"u_mean={meta['stats_10hz']['u_mean']:.4f}, "
              f"u_std={meta['stats_10hz']['u_std']:.4f}, total={elapsed:.1f}s")

        er_in_target = AUTOTUNE['target_er_low'] <= er <= AUTOTUNE['target_er_high']

        # Oracle eval
        print(f"  Oracle eval...")
        oracle = oracle_eval_condition(DS_ROOT / cond_id)
        oracle_results.append(oracle)
        if oracle['status'] == 'ok':
            print(f"    gain={oracle['gain_oracle']:.3f}, "
                  f"R2_phys={oracle['R2_phys']:.3f}, "
                  f"R2_oracle={oracle['R2_oracle']:.3f}")
        else:
            print(f"    {oracle['status'].upper()}: {oracle.get('status_detail','')}")

        summary_rows.append({
            'condition_id': cond_id, 'forcing_variant': fv_id, 'seed': seed,
            'mu_s_final': mu_s_final, 'd_p': d_p_j, 'k_spring': k_spring_j,
            'mu_k': mu_k_j, 'event_rate': er, 'x_std': x_std,
            'er_in_target': er_in_target, 'autotune_iters': len(at_hist),
            'oracle_status': oracle['status'],
            'oracle_gain': oracle.get('gain_oracle'),
            'R2_phys': oracle.get('R2_phys'),
            'R2_oracle': oracle.get('R2_oracle'),
        })

    # Write sweep outputs
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_SWEEP / "sweep_summary_3x.csv", index=False)

    # Oracle summary for discovery
    orows = []
    for r in oracle_results:
        orow = {
            'condition_id': r['condition_id'], 'status': r['status'],
            'regime': r.get('regime'), 'event_rate': r.get('event_rate'),
            'best_alpha': r.get('best_alpha'),
            'MSE_phys': r.get('MSE_phys'), 'MSE_oracle': r.get('MSE_oracle'),
            'gain_oracle': r.get('gain_oracle'),
            'R2_phys': r.get('R2_phys'), 'R2_oracle': r.get('R2_oracle'),
        }
        orows.append(orow)
    df_oracle = pd.DataFrame(orows)
    df_oracle.to_csv(OUT_SWEEP / "oracle_summary_3x.csv", index=False)

    # Phase 3 report
    ers = [r['event_rate'] for r in summary_rows]
    n_in_target = sum(1 for r in summary_rows if r['er_in_target'])
    print(f"\n{'='*70}")
    print(f"PHASE 3 COMPLETE")
    print(f"{'='*70}")
    print(f"ER in target: {n_in_target}/{n_cond}")
    print(f"ER: median={np.median(ers):.3f}, "
          f"range=[{min(ers):.3f}, {max(ers):.3f}]")
    x_stds = [r['x_std'] for r in summary_rows]
    print(f"x_std: median={np.median(x_stds)*1000:.3f} mm, "
          f"range=[{min(x_stds)*1000:.3f}, {max(x_stds)*1000:.3f}] mm")
    mu_s_finals = [r['mu_s_final'] for r in summary_rows]
    print(f"mu_s_final: median={np.median(mu_s_finals):.4f}, "
          f"range=[{min(mu_s_finals):.4f}, {max(mu_s_finals):.4f}]")

    # ---- Step 3: Phase 4 discovery ----
    print("\n" + "=" * 70)
    print("STEP 3: Phase 4 discovery (S1 physics + S2 closure + term selection)")
    print("=" * 70)

    OUT_DISC.mkdir(parents=True, exist_ok=True)

    oracle_by_id = {row['condition_id']: row for _, row in df_oracle.iterrows()}
    cond_dirs = sorted(DS_ROOT.glob("condition_*"))
    print(f"Found {len(cond_dirs)} condition directories")

    all_disc_results = []
    for cond_dir in cond_dirs:
        cond_id = cond_dir.name
        oracle_row = oracle_by_id.get(cond_id)
        if oracle_row is None:
            print(f"\n  [{cond_id}] No oracle row, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"DISCOVERY: {cond_id}")

        result_path = OUT_DISC / f"{cond_id}.json"
        if result_path.exists():
            print(f"  [{cond_id}] Already done, loading cached result")
            with open(result_path) as f:
                all_disc_results.append(json.load(f))
            continue

        t0 = time.time()
        x10_path = cond_dir / "x_10hz.csv"
        if not x10_path.exists():
            continue

        df = pd.read_csv(x10_path)
        N = len(df)
        n_train = int(0.60 * N)
        n_val = int(0.20 * N)
        df_train = df.iloc[:n_train].reset_index(drop=True)
        df_val = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
        df_test = df.iloc[n_train+n_val:].reset_index(drop=True)

        t_train = df_train['timestamp'].values
        x_train = df_train['displacement'].values
        v_train = df_train['velocity'].values
        t_val = df_val['timestamp'].values
        x_val = df_val['displacement'].values
        v_val = df_val['velocity'].values
        t_test = df_test['timestamp'].values
        x_test = df_test['displacement'].values
        v_test = df_test['velocity'].values

        # S1: Physics (PyTorch)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"disc3x_{cond_id}_"))
        try:
            for name, d in [('train', df_train), ('val', df_val)]:
                d.to_csv(tmp_dir / f"{name}.csv", index=False)

            torch.manual_seed(DISC_SEED); np.random.seed(DISC_SEED)

            train_ds = StateSpaceDataset(
                [str(tmp_dir / "train.csv")], L=S1_L, m=S1_L, H=S1_H,
                predict_deltas=False, normalize=False)
            val_ds = StateSpaceDataset(
                [str(tmp_dir / "val.csv")], L=S1_L, m=S1_L, H=S1_H,
                predict_deltas=False, normalize=False)

            if len(train_ds) < 10 or len(val_ds) < 5:
                result = {'condition_id': cond_id, 'status': 'too_few_samples'}
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                all_disc_results.append(result)
                continue

            train_ld = DataLoader(train_ds, batch_size=S1_BATCH, shuffle=True, num_workers=0)
            val_ld = DataLoader(val_ds, batch_size=S1_BATCH, shuffle=False, num_workers=0)

            model_s1 = KalmanForecaster(use_kappa=True).to(device)
            s1_best_val, s1_best_ep = train_s1(
                model_s1, train_ld, val_ld, device,
                S1_EPOCHS, S1_PATIENCE, S1_LR, S1_SCHED, tag=f"S1-{cond_id}")

            s1_params = model_s1.param_summary()
            print(f"  [{cond_id}] S1: alpha={s1_params['alpha']:.4f} "
                  f"kappa={s1_params['kappa']:.4f} c={s1_params['c']:.4f} "
                  f"vc={s1_params['vc']:.4f}")

            s1_pp = {
                'alpha': s1_params['alpha'], 'c': s1_params['c'],
                'vc': s1_params['vc'], 'kappa': s1_params['kappa'],
                'qx': s1_params['qx'], 'qu': s1_params['qu'],
                'R': s1_params['R'],
                'P0_xx': s1_params['P0_xx'], 'P0_uu': s1_params['P0_uu'],
            }

            # S2: Full 6-term closure (scipy)
            print(f"  [{cond_id}] S2 Full 6-term closure (scipy L-BFGS-B)")
            cl_best, s2_train_nll, s2_val_nll, s2_nfeval = train_s2_scipy(
                s1_pp, t_train, x_train, v_train, t_val, x_val, v_val,
                tag=f"S2-{cond_id}")

            print(f"  [{cond_id}] S2: a1={cl_best['a1']:.4f} b1={cl_best['b1']:.4f} "
                  f"b2={cl_best['b2']:.4f} d1={cl_best['d1']:.4f} "
                  f"d2={cl_best['d2']:.4f} d3={cl_best['d3']:.4f} "
                  f"q_scale={cl_best['q_scale']:.4f}")

            # Term selection
            selected_terms, term_vars, rel_vars = select_terms(
                cl_best, s1_pp, t_val, x_val, v_val)
            print(f"  [{cond_id}] Selected: {selected_terms}")
            print(f"  [{cond_id}] RelVar: " +
                  ", ".join(f"{tn}={rel_vars.get(tn,0):.3f}" for tn in TERM_NAMES))

            # Test-set KF metrics
            phys_cl = {k: 0.0 for k in TERM_NAMES}
            phys_cl['q_scale'] = 1.0
            filt_phys = kf_filter_2state(s1_pp, phys_cl, t_test, x_test, v_test)
            filt_clos = kf_filter_2state(s1_pp, cl_best, t_test, x_test, v_test)
            phys_acf1 = acf1(filt_phys['innovations'])
            clos_acf1 = acf1(filt_clos['innovations'])

            elapsed = time.time() - t0
            print(f"  [{cond_id}] ACF1 phys={phys_acf1:.4f} clos={clos_acf1:.4f} [{elapsed:.0f}s]")

            result = {
                'condition_id': cond_id, 'status': 'ok', 'n_10hz': N,
                's1_alpha': s1_params['alpha'], 's1_kappa': s1_params['kappa'],
                's1_c': s1_params['c'], 's1_vc': s1_params['vc'],
                's2_a1': cl_best['a1'], 's2_b1': cl_best['b1'],
                's2_b2': cl_best['b2'], 's2_d1': cl_best['d1'],
                's2_d2': cl_best['d2'], 's2_d3': cl_best['d3'],
                's2_q_scale': cl_best['q_scale'],
                'selected_terms': selected_terms,
                'n_selected': len(selected_terms),
                'rel_var_a1': rel_vars.get('a1', 0.0),
                'rel_var_b1': rel_vars.get('b1', 0.0),
                'rel_var_b2': rel_vars.get('b2', 0.0),
                'rel_var_d1': rel_vars.get('d1', 0.0),
                'rel_var_d2': rel_vars.get('d2', 0.0),
                'rel_var_d3': rel_vars.get('d3', 0.0),
                'phys_acf1': phys_acf1, 'clos_acf1': clos_acf1,
                'event_rate': float(oracle_row.get('event_rate', np.nan)),
                'oracle_gain': float(oracle_row.get('gain_oracle', np.nan)),
                'elapsed_s': round(elapsed, 1),
            }

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Sanitize NaN for JSON
        def sanitize(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        result_clean = {k: sanitize(v) if not isinstance(v, list) else v
                        for k, v in result.items()}
        with open(result_path, 'w') as f:
            json.dump(result_clean, f, indent=2)
        all_disc_results.append(result)

    # ---- Write discovery summary ----
    disc_csv_cols = [
        'condition_id', 'status', 'event_rate',
        's1_alpha', 's1_kappa', 's1_c', 's1_vc',
        's2_a1', 's2_b1', 's2_b2', 's2_d1', 's2_d2', 's2_d3', 's2_q_scale',
        'selected_terms', 'n_selected',
        'rel_var_a1', 'rel_var_b1', 'rel_var_b2',
        'rel_var_d1', 'rel_var_d2', 'rel_var_d3',
        'phys_acf1', 'clos_acf1', 'oracle_gain', 'elapsed_s',
    ]
    rows = []
    for r in all_disc_results:
        row = {}
        for c in disc_csv_cols:
            val = r.get(c)
            if isinstance(val, list):
                val = '+'.join(str(x) for x in val)
            row[c] = val
        rows.append(row)
    df_disc = pd.DataFrame(rows, columns=disc_csv_cols)
    df_disc.to_csv(OUT_DISC / "discovery_summary_3x.csv", index=False)

    # ---- Final summary ----
    ok_results = [r for r in all_disc_results if r.get('status') == 'ok']
    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY: {VELOCITY_SCALE}x VELOCITY EXPERIMENT")
    print("=" * 70)
    print(f"Total conditions: {len(all_disc_results)}")
    print(f"OK: {len(ok_results)}")

    if ok_results:
        term_counts = Counter()
        for r in ok_results:
            for t in r.get('selected_terms', []):
                term_counts[t] += 1
        n_ok = len(ok_results)
        print(f"\nTerm selection frequency (n={n_ok}):")
        for tn in TERM_NAMES:
            cnt = term_counts.get(tn, 0)
            print(f"  {tn}: {cnt}/{n_ok} ({100*cnt/n_ok:.0f}%)")

        # Per-term coefficient statistics (for selected conditions)
        for tn in TERM_NAMES:
            vals = [r[f's2_{tn}'] for r in ok_results if r.get(f's2_{tn}') is not None]
            if vals:
                print(f"  {tn} coeff: mean={np.mean(vals):.4f}, "
                      f"std={np.std(vals):.4f}, "
                      f"range=[{min(vals):.4f}, {max(vals):.4f}]")

        # ACF1 summary
        phys_acfs = [r['phys_acf1'] for r in ok_results
                     if r.get('phys_acf1') is not None
                     and not math.isnan(r.get('phys_acf1', float('nan')))]
        clos_acfs = [r['clos_acf1'] for r in ok_results
                     if r.get('clos_acf1') is not None
                     and not math.isnan(r.get('clos_acf1', float('nan')))]
        if phys_acfs:
            print(f"\nACF1 phys: mean={np.mean(phys_acfs):.4f}")
        if clos_acfs:
            print(f"ACF1 clos: mean={np.mean(clos_acfs):.4f}")

    elapsed_total = time.time() - t0_all
    print(f"\nTotal elapsed: {elapsed_total/60:.1f} min")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
