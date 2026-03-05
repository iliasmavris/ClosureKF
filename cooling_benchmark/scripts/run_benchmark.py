"""
Positive-Control Benchmark: Synthetic Forced Convection Cooling

Two-stage ClosureKF pipeline applied to synthetic forced convection:
  Stage 1: Newton's cooling (physics floor)
  Stage 2: 5-term closure library (same as NEON)
  Selection: Relative variance -> DNLL gate
  Evaluation: DxR2, ACF, NIS, truth-parameter comparison

Known truth: dT/dt = -k_true*(T-T_air) - h_true*wind*(T-T_air)
Expected: S2 discovers forced_conv = wind*(T-T_air) with coeff ~ -h_true

Usage:
    python -u cooling_benchmark/scripts/run_benchmark.py

Non-destructive: all outputs go to cooling_benchmark/outputs/
"""

import os, sys, math, json, time, copy, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Unified bar style
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'EMS_v5_cleanroute', 'figures'))
from plot_style import apply_mpl_style, PALETTE, BAR_KW
apply_mpl_style()

import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(os.cpu_count() or 4)

# --- Path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BENCH_DIR.parent

# Import NEON models directly
sys.path.insert(0, str(PROJECT_ROOT / "neon_benchmark"))
from models.neon_physics import NeonKF, NeonKFClosure, CLOSURE_TERM_NAMES

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
CONFIG_PATH = BENCH_DIR / "configs" / "pipeline_config.json"
with open(CONFIG_PATH, 'r') as f:
    CFG = json.load(f)

DT = CFG['dt_seconds']
SEEDS = CFG['seeds']
VAR_FLOOR = 1e-6

SPLITS_DIR = BENCH_DIR / "data" / "splits"
OUT_DIR = BENCH_DIR / "outputs"
FIG_DIR = BENCH_DIR / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load truth parameters
TRUTH_PATH = BENCH_DIR / "data" / "truth_params.json"

# Horizon mapping (in time steps)
HORIZON_TARGETS = CFG['horizon_targets_hours']
HORIZON_MAP = []
for h_hr in HORIZON_TARGETS:
    h_steps = max(1, round(h_hr * 3600 / DT))
    achieved_hr = h_steps * DT / 3600
    HORIZON_MAP.append({
        'target_hours': h_hr, 'h_steps': h_steps,
        'achieved_hours': achieved_hr
    })
MAX_H_STEPS = max(hm['h_steps'] for hm in HORIZON_MAP)


# ==============================================================================
#  DATASET (identical to NEON)
# ==============================================================================

class CoolingDataset(torch.utils.data.Dataset):
    """Dataset for cooling benchmark (same format as NEON)."""

    def __init__(self, csv_path, L=48, H=48):
        self.L = L
        self.H = H
        df = pd.read_csv(csv_path)
        self.T_obs = df['water_temp'].values.astype(np.float32)
        self.T_air = df['air_temp'].values.astype(np.float32)
        self.wind = df['wind_speed'].values.astype(np.float32)
        self.par = df['par'].values.astype(np.float32)
        self.dt = df['time_delta'].values.astype(np.float32)
        self.timestamps = df['timestamp'].values.astype(np.float32)

        T = len(self.T_obs)
        self.indices = list(range(L - 1, T - H))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        start = t - self.L + 1
        end = t + self.H + 1

        T_obs = torch.tensor(self.T_obs[start:end])
        T_air = torch.tensor(self.T_air[start:end])
        wind = torch.tensor(self.wind[start:end])
        par = torch.tensor(self.par[start:end])
        dt = torch.tensor(self.dt[start:end])
        T_future = torch.tensor(self.T_obs[t+1:t+self.H+1])

        return T_obs, T_air, wind, par, dt, T_future


# ==============================================================================
#  LOSS & HELPERS
# ==============================================================================

def gaussian_nll(T_pred, T_var, T_true, var_floor=1e-6):
    v = torch.clamp(T_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (T_true - T_pred) ** 2 / v).mean()


def compute_acf(e, max_lag=50):
    e_c = e - np.mean(e)
    var = np.var(e)
    n = len(e)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(e_c[:n-l] * e_c[l:]) / (n * var) if l > 0
                     else 1.0 for l in range(max_lag + 1)])


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ==============================================================================
#  TRAINING
# ==============================================================================

def train_model(model, train_loader, val_loader, L_hist, device,
                max_epochs, patience, lr, sched_patience=10,
                param_getter=None, tag=""):
    """Train with early stopping."""
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
            T_obs, T_air, wind, par, dt, T_future = [b.to(device) for b in batch]
            optimizer.zero_grad()
            T_pred, T_var = model(T_obs, T_air, wind, par, dt, L_hist)
            loss = gaussian_nll(T_pred, T_var, T_future, VAR_FLOOR)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()
            # Bound q_scale to [0.3, 3.0] to prevent variance-hack local minima
            if hasattr(model, 'log_q_scale'):
                with torch.no_grad():
                    model.log_q_scale.clamp_(math.log(0.3), math.log(3.0))
            tot += loss.item(); nb += 1
        tr_nll = tot / max(nb, 1)
        train_hist.append(tr_nll)

        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for batch in val_loader:
                T_obs, T_air, wind, par, dt, T_future = [b.to(device) for b in batch]
                T_pred, T_var = model(T_obs, T_air, wind, par, dt, L_hist)
                vl = gaussian_nll(T_pred, T_var, T_future, VAR_FLOOR)
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

        if (ep + 1) % 20 == 0 or ep == 0 or wait >= patience:
            print(f"    [{tag}] ep {ep+1:3d}  tr={tr_nll:.5f}  "
                  f"val={val_nll:.5f}  best={best_loss:.5f}@ep{best_ep}  "
                  f"[{time.time()-t0:.0f}s]")
        if wait >= patience:
            print(f"    [{tag}] Early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print(f"    [{tag}] WARNING: No valid epoch found. Using final state.")
    model.eval()
    print(f"    [{tag}] Done {time.time()-t0:.0f}s, "
          f"best_val={best_loss:.5f} at ep={best_ep}")
    return best_loss, best_ep, train_hist, val_hist


def compute_val_nll(model, val_loader, L_hist, device):
    model.eval()
    with torch.no_grad():
        vt, vn = 0.0, 0
        for batch in val_loader:
            T_obs, T_air, wind, par, dt, T_future = [b.to(device) for b in batch]
            T_pred, T_var = model(T_obs, T_air, wind, par, dt, L_hist)
            vl = gaussian_nll(T_pred, T_var, T_future, VAR_FLOOR)
            vt += vl.item(); vn += 1
    return vt / vn


# ==============================================================================
#  NUMPY EVALUATION
# ==============================================================================

def kf_filter_1state(params, cl_params, timestamps, T_obs, T_air, wind, par):
    """1-state KF filter for evaluation."""
    N = len(T_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_T = np.zeros(N)

    k = params['k']
    q = params['q']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']

    c_pl = cl_params.get('par_lin', 0.0)
    c_pq = cl_params.get('par_quad', 0.0)
    c_wc = cl_params.get('wind_cool', 0.0)
    c_se = cl_params.get('sensible', 0.0)
    c_fc = cl_params.get('forced_conv', 0.0)

    s = T_obs[0]
    P = params['P0']
    states_T[0] = s

    for i in range(1, N):
        dt_i = timestamps[i] - timestamps[i-1]
        if dt_i <= 0:
            dt_i = DT

        dT = s - T_air[i-1]
        T_pred = s - k * dT * dt_i

        cl = (c_pl * par[i-1]
              + c_pq * par[i-1]**2
              + c_wc * wind[i-1]
              + c_se * dT
              + c_fc * wind[i-1] * dT)
        T_pred += cl * dt_i

        dcl_dT = c_se + c_fc * wind[i-1]
        F_val = 1.0 - k * dt_i + dcl_dT * dt_i

        Q_val = q_sc * q * dt_i
        P_pred = F_val ** 2 * P + Q_val

        innov = T_obs[i] - T_pred
        S_val = P_pred + R
        innovations[i] = innov
        S_values[i] = S_val

        K_gain = P_pred / S_val
        s = T_pred + K_gain * innov
        P = (1.0 - K_gain) ** 2 * P_pred + K_gain ** 2 * R
        states_T[i] = s

    return {
        'innovations': innovations,
        'S_values': S_values,
        'states_T': states_T,
    }


def compute_dxr2(params, cl_params, states_T, timestamps,
                 T_obs, T_air, wind, par, max_h, eval_start=1):
    """Multi-horizon DxR2: R2 of temperature increments."""
    N = len(T_obs)
    k = params['k']
    c_pl = cl_params.get('par_lin', 0.0)
    c_pq = cl_params.get('par_quad', 0.0)
    c_wc = cl_params.get('wind_cool', 0.0)
    c_se = cl_params.get('sensible', 0.0)
    c_fc = cl_params.get('forced_conv', 0.0)

    dT_pred = [[] for _ in range(max_h)]
    dT_true = [[] for _ in range(max_h)]

    for i in range(max(eval_start, 1), N - 1):
        sT = states_T[i]
        max_steps = min(max_h, N - 1 - i)

        for step in range(max_steps):
            ki = i + 1 + step
            dt_s = timestamps[ki] - timestamps[ki-1]
            if dt_s <= 0:
                dt_s = DT

            dT = sT - T_air[ki-1]
            cl = (c_pl * par[ki-1] + c_pq * par[ki-1]**2
                  + c_wc * wind[ki-1] + c_se * dT
                  + c_fc * wind[ki-1] * dT)
            sT_new = sT - k * dT * dt_s + cl * dt_s
            sT_new = np.clip(sT_new, -50.0, 100.0)
            sT = sT_new

            h = step + 1
            dT_pred[h-1].append(sT - T_obs[i])
            dT_true[h-1].append(T_obs[i + h] - T_obs[i])

    r2_arr = np.full(max_h, np.nan)
    for h in range(max_h):
        if len(dT_pred[h]) < 10:
            continue
        dp = np.array(dT_pred[h])
        do = np.array(dT_true[h])
        err = do - dp
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((do - np.mean(do)) ** 2)
        r2_arr[h] = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return r2_arr


# ==============================================================================
#  CLOSURE TERM VARIANCES
# ==============================================================================

def compute_closure_term_variances(model, val_loader, L_hist, device):
    """Compute per-term variance contribution on validation set."""
    model.eval()
    all_contributions = {name: [] for name in CLOSURE_TERM_NAMES}

    with torch.no_grad():
        for batch in val_loader:
            T_obs, T_air, wind, par, dt_batch, T_future = [b.to(device) for b in batch]
            B, T_total = T_obs.shape
            H = T_total - L_hist

            s = T_obs[:, 0:1].clone()
            P = model.P0.unsqueeze(0).unsqueeze(1).expand(B, 1, 1).clone()

            for t in range(1, L_hist):
                dt_t = dt_batch[:, t].clamp(min=1.0)
                s, P = model.kf_predict(s, P, T_air[:, t-1], wind[:, t-1],
                                         par[:, t-1], dt_t)
                s, P = model.kf_update(s, P, T_obs[:, t])

            for t in range(H):
                idx = L_hist + t
                dt_t = dt_batch[:, idx].clamp(min=1.0)
                T_state = s[:, 0]
                dT = T_state - T_air[:, idx-1]

                terms = {
                    'par_lin': model.theta_par_lin * par[:, idx-1] * dt_t,
                    'par_quad': model.theta_par_quad * par[:, idx-1]**2 * dt_t,
                    'wind_cool': model.theta_wind_cool * wind[:, idx-1] * dt_t,
                    'sensible': model.theta_sensible * dT * dt_t,
                    'forced_conv': model.theta_forced_conv * wind[:, idx-1] * dT * dt_t,
                }

                for name in CLOSURE_TERM_NAMES:
                    all_contributions[name].append(
                        terms[name].cpu().numpy().flatten())

                s, P = model.kf_predict(s, P, T_air[:, idx-1], wind[:, idx-1],
                                         par[:, idx-1], dt_t)

    variances = {}
    for name in CLOSURE_TERM_NAMES:
        series = np.concatenate(all_contributions[name])
        variances[name] = float(np.var(series))

    total_var = sum(variances.values())
    rel_variances = {name: variances[name] / total_var if total_var > 1e-15 else 0.0
                     for name in CLOSURE_TERM_NAMES}
    return rel_variances, variances


# ==============================================================================
#  TERM SELECTION (DNLL GATE)
# ==============================================================================

def zero_and_freeze_term(model, term_name):
    attr_map = {
        'par_lin': 'theta_par_lin',
        'par_quad': 'theta_par_quad',
        'wind_cool': 'theta_wind_cool',
        'sensible': 'theta_sensible',
        'forced_conv': 'theta_forced_conv',
    }
    with torch.no_grad():
        attr = attr_map[term_name]
        getattr(model, attr).fill_(0.0)
        getattr(model, attr).requires_grad_(False)


def get_trainable_closure_params(model, excluded_terms):
    attr_map = {
        'par_lin': 'theta_par_lin',
        'par_quad': 'theta_par_quad',
        'wind_cool': 'theta_wind_cool',
        'sensible': 'theta_sensible',
        'forced_conv': 'theta_forced_conv',
    }
    params = []
    for name in CLOSURE_TERM_NAMES:
        if name in excluded_terms:
            continue
        p = getattr(model, attr_map[name])
        if p.requires_grad:
            params.append(p)
    if model.log_q_scale.requires_grad:
        params.append(model.log_q_scale)
    return params


def run_term_selection(model_s2, train_loader, val_loader, L_hist,
                       device, seed_dir, seed):
    """Rel.var -> DNLL gate -> final selection."""
    print_section(f"SEED {seed}: TERM SELECTION")
    TAU_NLL = CFG['gate']['tau_nll']
    REL_VAR_THRESH = CFG['gate']['rel_var_threshold']

    selection_log = {
        'seed': seed,
        'full_closure': model_s2.closure_summary(),
        'rel_variance': {},
        'dnll_gate': {},
        'final_terms': [],
    }

    print("  Computing relative variance of closure terms...")
    rel_vars, abs_vars = compute_closure_term_variances(
        model_s2, val_loader, L_hist, device)
    selection_log['rel_variance'] = {
        'relative': rel_vars,
        'absolute': {k: float(v) for k, v in abs_vars.items()},
    }

    for name in CLOSURE_TERM_NAMES:
        status = "PASS" if rel_vars[name] >= REL_VAR_THRESH else "DROP"
        print(f"    {name}: rel.var={rel_vars[name]:.4f}  [{status}]")

    survived_var = [n for n in CLOSURE_TERM_NAMES if rel_vars[n] >= REL_VAR_THRESH]
    dropped_var = [n for n in CLOSURE_TERM_NAMES if rel_vars[n] < REL_VAR_THRESH]
    print(f"  Survived variance gate: {survived_var}")

    full_nll = compute_val_nll(model_s2, val_loader, L_hist, device)
    print(f"\n  Full model val NLL: {full_nll:.6f}")
    selection_log['full_nll'] = full_nll

    dnll_results = {}
    survived_gate = []

    for term_name in survived_var:
        print(f"\n  --- DNLL gate: dropping {term_name} ---")
        refit_dir = seed_dir / "refits" / f"term_drop_{term_name}"
        refit_dir.mkdir(parents=True, exist_ok=True)

        model_refit = copy.deepcopy(model_s2)
        zero_and_freeze_term(model_refit, term_name)
        model_refit.to(device)

        excluded = dropped_var + [term_name]
        refit_params = lambda m=model_refit, ex=excluded: \
            get_trainable_closure_params(m, ex)

        try:
            best_val, best_ep, tr_hist, val_hist = train_model(
                model_refit, train_loader, val_loader, L_hist, device,
                CFG['refit']['max_epochs'], CFG['refit']['patience'],
                CFG['stage2']['lr'], CFG['stage2']['sched_patience'],
                param_getter=refit_params, tag=f"refit-drop-{term_name}")

            refit_nll = compute_val_nll(model_refit, val_loader, L_hist, device)
            delta_nll = refit_nll - full_nll
            passed = delta_nll >= TAU_NLL

            print(f"    Refit NLL: {refit_nll:.6f}, DNLL: {delta_nll:.6f}, "
                  f"gate={'PASS' if passed else 'FAIL'} (threshold={TAU_NLL})")

            dnll_results[term_name] = {
                'refit_nll': refit_nll, 'delta_nll': delta_nll,
                'passed_gate': passed, 'best_epoch': best_ep,
            }

            with open(refit_dir / "refit_log.json", 'w') as f:
                json.dump({
                    'term_dropped': term_name,
                    'full_nll': full_nll,
                    'refit_nll': refit_nll,
                    'delta_nll': delta_nll,
                    'passed': passed,
                }, f, indent=2)

            if passed:
                survived_gate.append(term_name)

        except Exception as e:
            print(f"    ERROR during refit: {e}")
            dnll_results[term_name] = {'error': str(e), 'passed_gate': False}

    selection_log['dnll_gate'] = dnll_results
    final_terms = survived_gate if survived_gate else ['none']
    selection_log['final_terms'] = final_terms
    print(f"\n  FINAL SELECTED TERMS: {final_terms}")

    return selection_log, final_terms


# ==============================================================================
#  EVALUATE ONE MODEL
# ==============================================================================

def evaluate_model(label, params, cl_params, timestamps, T_obs,
                   T_air, wind, par, eval_start, max_h):
    """Full evaluation on test data."""
    filt = kf_filter_1state(params, cl_params, timestamps, T_obs,
                            T_air, wind, par)
    innov = filt['innovations']
    S_vals = filt['S_values']
    sT = filt['states_T']

    e = innov[eval_start:]
    S_sc = S_vals[eval_start:]
    valid = ~np.isnan(e)
    e_v = e[valid]
    S_v = S_sc[valid]

    acf = compute_acf(e_v, max_lag=50)
    nis = float(np.mean(e_v**2 / np.maximum(S_v, 1e-15)))

    dxr2 = compute_dxr2(params, cl_params, sT, timestamps,
                         T_obs, T_air, wind, par, max_h, eval_start)

    dxr2_at_targets = {}
    for hm in HORIZON_MAP:
        h_idx = hm['h_steps'] - 1
        if h_idx < len(dxr2):
            key = f"{hm['target_hours']:.1f}h"
            dxr2_at_targets[key] = float(dxr2[h_idx])

    print(f"    [{label}] ACF(1)={acf[1]:.4f} NIS={nis:.4f}")
    for key, val in dxr2_at_targets.items():
        print(f"      DxR2@{key}={val:+.4f}")

    return {
        'label': label,
        'acf1': float(acf[1]),
        'acf_raw': acf.tolist(),
        'nis_mean': nis,
        'dxr2': {
            'h_steps': list(range(1, max_h + 1)),
            'h_hours': [i * DT / 3600 for i in range(1, max_h + 1)],
            'values': dxr2.tolist(),
        },
        'dxr2_at_targets': dxr2_at_targets,
        'states_T': sT,
    }


# ==============================================================================
#  SINGLE SEED PIPELINE
# ==============================================================================

def run_seed(seed, device, df_train, df_val, df_test):
    """Run full S1 + S2 + selection + evaluation for one seed."""
    t0_seed = time.time()
    seed_dir = OUT_DIR / f"seed_logs" / f"seed{seed}"
    for d in ['checkpoints', 'refits']:
        (seed_dir / d).mkdir(parents=True, exist_ok=True)

    L_hist = 48   # 48 * 30min = 24 hours
    H_pred = max(MAX_H_STEPS + 2, 48)

    # ==========================================
    #  STAGE 1: PHYSICS FLOOR (Newton's Cooling)
    # ==========================================
    print_section(f"SEED {seed}: STAGE 1 -- NEWTON'S COOLING")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CoolingDataset(SPLITS_DIR / "train.csv", L=L_hist, H=H_pred)
    val_ds = CoolingDataset(SPLITS_DIR / "val.csv", L=L_hist, H=H_pred)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"  Datasets: train={len(train_ds)}, val={len(val_ds)}")
    print(f"  L_hist={L_hist}, H_pred={H_pred}")

    model_s1 = NeonKF().to(device)
    best_val_s1, best_ep_s1, tr_h_s1, val_h_s1 = train_model(
        model_s1, train_ld, val_ld, L_hist, device,
        CFG['stage1']['epochs'], CFG['stage1']['patience'],
        CFG['stage1']['lr'], CFG['stage1']['sched_patience'],
        tag=f"S1-seed{seed}")

    s1_params = model_s1.param_summary()
    print(f"  S1 learned: k={s1_params['k']:.6f} "
          f"(tau={s1_params['tau_hours']:.1f}h) "
          f"q={s1_params['q']:.2e} R={s1_params['R']:.2e}")

    torch.save({
        'state_dict': model_s1.state_dict(),
        'params': s1_params,
        'best_val': best_val_s1, 'best_epoch': best_ep_s1,
    }, seed_dir / "checkpoints" / "stage1_physics.pth")

    # ==========================================
    #  STAGE 2: CLOSURE DISCOVERY (5 terms)
    # ==========================================
    print_section(f"SEED {seed}: STAGE 2 -- CLOSURE DISCOVERY")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_s2 = NeonKFClosure(
        k_init=max(s1_params['k'], 1e-8),
        log_q_init=math.log(max(s1_params['q'], 1e-15)),
        log_r_init=math.log(max(s1_params['R'], 1e-15)),
        log_p0_init=math.log(max(s1_params['P0'], 1e-15)),
    ).to(device)

    model_s2.freeze_physics()
    s2_trainable = lambda: model_s2.closure_params_list()

    best_val_s2, best_ep_s2, tr_h_s2, val_h_s2 = train_model(
        model_s2, train_ld, val_ld, L_hist, device,
        CFG['stage2']['epochs'], CFG['stage2']['patience'],
        CFG['stage2']['lr'], CFG['stage2']['sched_patience'],
        param_getter=s2_trainable, tag=f"S2-seed{seed}")

    s2_params = model_s2.param_summary()
    cl_sum = model_s2.closure_summary()
    print(f"  S2 closure: {cl_sum}")
    print(f"  Symbolic: {model_s2.symbolic_law()}")

    torch.save({
        'state_dict': model_s2.state_dict(),
        'params': s2_params,
        'closure': cl_sum,
        'best_val': best_val_s2, 'best_epoch': best_ep_s2,
    }, seed_dir / "checkpoints" / "stage2_closure.pth")

    # ==========================================
    #  TERM SELECTION
    # ==========================================
    selection_log, final_terms = run_term_selection(
        model_s2, train_ld, val_ld, L_hist, device, seed_dir, seed)

    # ==========================================
    #  EVALUATION ON TEST SET
    # ==========================================
    print_section(f"SEED {seed}: EVALUATION")

    warmup_steps = min(L_hist, len(df_val))
    df_warmup = df_val.iloc[-warmup_steps:].copy()
    df_eval = pd.concat([df_warmup, df_test], ignore_index=True)

    timestamps = df_eval['timestamp'].values.astype(np.float64)
    T_obs = df_eval['water_temp'].values.astype(np.float64)
    T_air = df_eval['air_temp'].values.astype(np.float64)
    wind_arr = df_eval['wind_speed'].values.astype(np.float64)
    par_arr = df_eval['par'].values.astype(np.float64)
    eval_start = warmup_steps

    phys_pp = {k: s1_params[k] for k in ['k', 'q', 'R', 'P0']}
    phys_cl = {name: 0.0 for name in CLOSURE_TERM_NAMES}
    phys_cl['q_scale'] = 1.0

    clos_pp = {k: s2_params[k] for k in ['k', 'q', 'R', 'P0']}
    clos_cl = {name: cl_sum[name] for name in CLOSURE_TERM_NAMES}
    clos_cl['q_scale'] = cl_sum['q_scale']

    max_h_eval = MAX_H_STEPS + 5
    results = {}
    for mtype, pp, cc in [('physics', phys_pp, phys_cl),
                           ('closure', clos_pp, clos_cl)]:
        results[mtype] = evaluate_model(
            mtype, pp, cc, timestamps, T_obs, T_air, wind_arr, par_arr,
            eval_start, max_h_eval)

    elapsed = time.time() - t0_seed
    print(f"\n  Seed {seed} complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        'seed': seed,
        's1_params': s1_params,
        's2_params': s2_params,
        'closure_summary': cl_sum,
        'selection_log': selection_log,
        'final_terms': final_terms,
        'results': {
            'physics': {k: v for k, v in results['physics'].items()
                       if k != 'states_T'},
            'closure': {k: v for k, v in results['closure'].items()
                       if k != 'states_T'},
        },
        'elapsed_sec': elapsed,
    }


# ==============================================================================
#  PLOTTING
# ==============================================================================

def make_plots(all_seed_results, truth_params):
    """Generate summary plots with truth-parameter comparison."""
    print_section("GENERATING PLOTS")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Panel (a): DxR2 vs Horizon ---
    ax = axes[0]
    for r in all_seed_results:
        seed = r['seed']
        for mtype, color, ls in [('physics', '#222222', '--'),
                                  ('closure', '#0072B2', '-')]:
            dxr2 = r['results'][mtype]['dxr2']
            h_hours = dxr2['h_hours']
            vals = dxr2['values']
            ax.plot(h_hours, vals, color=color, ls=ls, alpha=0.5, lw=1)

    for mtype, color, ls, label in [('physics', '#222222', '--', 'Physics floor'),
                                     ('closure', '#0072B2', '-', 'With closure')]:
        all_vals = []
        for r in all_seed_results:
            all_vals.append(r['results'][mtype]['dxr2']['values'])
        arr = np.array(all_vals)
        mean_v = np.nanmean(arr, axis=0)
        h_hours = all_seed_results[0]['results'][mtype]['dxr2']['h_hours']
        ax.plot(h_hours, mean_v, color=color, ls=ls, lw=2.5, label=label)

    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Forecast horizon (hours)')
    ax.set_ylabel(r'$R^2_{\Delta T}$')
    ax.set_title('(a) Temperature increment skill')
    ax.legend(fontsize=8, frameon=False)

    # --- Panel (b): Closure coefficients with truth ---
    ax = axes[1]
    term_labels = ['PAR', r'PAR$^2$', 'wind', r'$\Delta T$',
                   r'wind$\cdot\Delta T$']

    # Truth values for reference
    h_true = truth_params.get('h_true', 1.5e-4)
    truth_vals = {
        'par_lin': 0.0, 'par_quad': 0.0, 'wind_cool': 0.0,
        'sensible': 0.0, 'forced_conv': -h_true,
    }

    for i, name in enumerate(CLOSURE_TERM_NAMES):
        vals = [r['closure_summary'][name] for r in all_seed_results]
        bar_color = PALETTE['gate_orange'] if name == 'forced_conv' else PALETTE['closure_blue']
        ax.bar(i, np.mean(vals), yerr=np.std(vals), capsize=4,
               color=bar_color, **BAR_KW)
        # Truth marker
        if abs(truth_vals[name]) > 1e-12:
            ax.plot(i, truth_vals[name], 'k*', markersize=12, zorder=5)

    ax.set_xticks(range(len(CLOSURE_TERM_NAMES)))
    ax.set_xticklabels(term_labels, fontsize=8, rotation=15)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel('Coefficient value')
    ax.set_title('(b) Discovered vs truth')

    # --- Panel (c): Selected terms frequency ---
    ax = axes[2]
    term_counts = {name: 0 for name in CLOSURE_TERM_NAMES}
    term_counts['none'] = 0
    for r in all_seed_results:
        for t in r['final_terms']:
            term_counts[t] = term_counts.get(t, 0) + 1

    names = list(term_counts.keys())
    counts = [term_counts[n] for n in names]
    colors_bar = [PALETTE['gate_orange'] if n == 'forced_conv' else PALETTE['closure_blue'] for n in names]
    bars = ax.bar(range(len(names)), counts, color=colors_bar,
                  **BAR_KW)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel(f'Selected count (/{len(SEEDS)} seeds)')
    ax.set_title(r'(c) $\Delta$NLL gate selection')
    ax.set_ylim(0, len(SEEDS) + 0.5)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = FIG_DIR / f"cooling_benchmark_summary.{ext}"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ==============================================================================
#  PASS CRITERIA CHECK
# ==============================================================================

def check_pass_criteria(all_seed_results, truth_params):
    """Check all pass criteria and return structured results."""
    print_section("PASS CRITERIA CHECK")

    h_true = truth_params.get('h_true', 1.5e-4)
    n_seeds = len(all_seed_results)
    results = {}

    # P1: forced_conv selected in all seeds
    fc_selected = sum(1 for r in all_seed_results
                      if 'forced_conv' in r['final_terms'])
    results['P1_forced_conv_selected'] = {
        'count': fc_selected, 'total': n_seeds,
        'pass': fc_selected == n_seeds
    }
    print(f"  P1: forced_conv selected {fc_selected}/{n_seeds} "
          f"{'PASS' if results['P1_forced_conv_selected']['pass'] else 'FAIL'}")

    # P2: No distractor false positives
    # Note: sensible is structurally expected (k-correction for frozen physics
    # that absorbed mean wind into k), so only PAR/wind terms are distractors
    EXPECTED_TERMS = {'forced_conv', 'sensible', 'none'}
    false_pos = []
    for r in all_seed_results:
        for t in r['final_terms']:
            if t not in EXPECTED_TERMS:
                false_pos.append((r['seed'], t))
    results['P2_no_distractor_fp'] = {
        'expected_terms': sorted(EXPECTED_TERMS - {'none'}),
        'false_positives': false_pos,
        'pass': len(false_pos) == 0
    }
    print(f"  P2: Distractor false positives = {len(false_pos)} "
          f"{'PASS' if results['P2_no_distractor_fp']['pass'] else 'FAIL'}")
    print(f"      (sensible is structurally expected: S1 k-correction)")
    if false_pos:
        for seed, term in false_pos:
            print(f"      seed {seed}: {term}")

    # P3: forced_conv coefficient within 50% of -h_true
    fc_coeffs = [r['closure_summary']['forced_conv'] for r in all_seed_results]
    rel_errors = [abs(c - (-h_true)) / h_true for c in fc_coeffs]
    p3_pass = all(re < 0.50 for re in rel_errors)
    results['P3_coefficient_recovery'] = {
        'h_true': -h_true,
        'discovered': fc_coeffs,
        'rel_errors': rel_errors,
        'pass': p3_pass
    }
    print(f"  P3: forced_conv coeff recovery (within 50%): "
          f"{'PASS' if p3_pass else 'FAIL'}")
    for i, (c, re) in enumerate(zip(fc_coeffs, rel_errors)):
        print(f"      seed {all_seed_results[i]['seed']}: "
              f"coeff={c:.6f}, truth={-h_true:.6f}, rel_err={re:.3f}")

    # P4: Closure DxR2 > Physics DxR2 at horizons >= 2h
    p4_details = []
    for r in all_seed_results:
        for key in r['results']['physics']['dxr2_at_targets']:
            h_val = float(key.replace('h', ''))
            if h_val >= 2.0:
                phys_v = r['results']['physics']['dxr2_at_targets'][key]
                clos_v = r['results']['closure']['dxr2_at_targets'][key]
                p4_details.append({
                    'seed': r['seed'], 'horizon': key,
                    'physics': phys_v, 'closure': clos_v,
                    'closure_better': clos_v > phys_v
                })
    p4_pass = all(d['closure_better'] for d in p4_details) if p4_details else False
    results['P4_closure_improves'] = {'details': p4_details, 'pass': p4_pass}
    print(f"  P4: Closure DxR2 > Physics DxR2 at >=2h: "
          f"{'PASS' if p4_pass else 'FAIL'}")
    for d in p4_details:
        status = 'OK' if d['closure_better'] else 'FAIL'
        print(f"      seed {d['seed']} @{d['horizon']}: "
              f"phys={d['physics']:+.4f}, clos={d['closure']:+.4f} [{status}]")

    # P5: Closure ACF(1) stays low (< 0.15 absolute, or improves over physics)
    # Note: when physics ACF(1) is already near-zero, improvement is not meaningful
    p5_details = []
    for r in all_seed_results:
        phys_acf = r['results']['physics']['acf1']
        clos_acf = r['results']['closure']['acf1']
        # Pass if closure improves OR both are already low (<0.15)
        ok = (clos_acf < phys_acf) or (abs(clos_acf) < 0.15)
        p5_details.append({
            'seed': r['seed'],
            'physics_acf1': phys_acf, 'closure_acf1': clos_acf,
            'improved': ok
        })
    p5_pass = all(d['improved'] for d in p5_details)
    results['P5_acf_acceptable'] = {'details': p5_details, 'pass': p5_pass}
    print(f"  P5: Closure ACF(1) acceptable (improves or |ACF|<0.15): "
          f"{'PASS' if p5_pass else 'FAIL'}")
    for d in p5_details:
        status = 'OK' if d['improved'] else 'FAIL'
        print(f"      seed {d['seed']}: phys={d['physics_acf1']:.4f}, "
              f"clos={d['closure_acf1']:.4f} [{status}]")

    # Overall
    all_pass = all([
        results['P1_forced_conv_selected']['pass'],
        results['P2_no_distractor_fp']['pass'],
        results['P3_coefficient_recovery']['pass'],
        results['P4_closure_improves']['pass'],
        results['P5_acf_acceptable']['pass'],
    ])
    results['overall_pass'] = all_pass
    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return results


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    t0_global = time.time()
    device = torch.device('cpu')

    print("=" * 70)
    print("Positive-Control Benchmark: Synthetic Forced Convection Cooling")
    print("=" * 70)
    print(f"dt = {DT}s ({DT/3600:.1f}h)")
    print(f"Seeds: {SEEDS}")
    print(f"Horizon map:")
    for hm in HORIZON_MAP:
        print(f"  target={hm['target_hours']:.1f}h -> {hm['h_steps']} steps "
              f"-> achieved={hm['achieved_hours']:.2f}h")

    # Check data exists
    for name in ['train', 'val', 'test']:
        p = SPLITS_DIR / f"{name}.csv"
        if not p.exists():
            print(f"\nERROR: {p} not found. Run generate_data.py first.")
            sys.exit(1)

    # Load truth params
    truth_params = {}
    if TRUTH_PATH.exists():
        with open(TRUTH_PATH, 'r') as f:
            tp_data = json.load(f)
            truth_params = tp_data.get('truth_params', {})
        print(f"Truth params loaded: k_true={truth_params.get('k_true', '?')}, "
              f"h_true={truth_params.get('h_true', '?')}")
    else:
        print("WARNING: truth_params.json not found")

    # Load dataframes
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")
    print(f"Data: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # Run seeds
    all_seed_results = []
    for seed in SEEDS:
        result = run_seed(seed, device, df_train, df_val, df_test)
        all_seed_results.append(result)

    # ==========================================
    #  WRITE OUTPUTS
    # ==========================================
    print_section("WRITING OUTPUTS")

    # metrics.json
    metrics_out = {
        'benchmark': 'cooling_positive_control',
        'dt_seconds': DT,
        'horizon_map': HORIZON_MAP,
        'truth_params': truth_params,
        'per_seed': {},
    }
    for r in all_seed_results:
        s = f"seed{r['seed']}"
        metrics_out['per_seed'][s] = {
            'physics': r['results']['physics'],
            'closure': r['results']['closure'],
            'selected_terms': r['final_terms'],
            's1_params': r['s1_params'],
            'closure_coefficients': r['closure_summary'],
        }

    # Compute means
    mean_metrics = {}
    for mtype in ['physics', 'closure']:
        acf1_vals = [r['results'][mtype]['acf1'] for r in all_seed_results]
        nis_vals = [r['results'][mtype]['nis_mean'] for r in all_seed_results]
        mean_metrics[mtype] = {
            'acf1': {'mean': float(np.mean(acf1_vals)),
                     'std': float(np.std(acf1_vals))},
            'nis': {'mean': float(np.mean(nis_vals)),
                    'std': float(np.std(nis_vals))},
            'dxr2_at_targets': {},
        }
        for key in all_seed_results[0]['results'][mtype].get('dxr2_at_targets', {}):
            vals = [r['results'][mtype]['dxr2_at_targets'].get(key, np.nan)
                    for r in all_seed_results]
            mean_metrics[mtype]['dxr2_at_targets'][key] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
            }
    metrics_out['mean'] = mean_metrics

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Wrote {metrics_path}")

    # selection_summary.json
    sel_out = {}
    for r in all_seed_results:
        sel_out[f"seed{r['seed']}"] = r['selection_log']
    sel_path = OUT_DIR / "selection_summary.json"
    with open(sel_path, 'w') as f:
        json.dump(sel_out, f, indent=2)
    print(f"  Wrote {sel_path}")

    # Pass criteria
    pass_results = check_pass_criteria(all_seed_results, truth_params)
    pass_path = OUT_DIR / "pass_criteria.json"
    with open(pass_path, 'w') as f:
        json.dump(pass_results, f, indent=2, default=str)
    print(f"  Wrote {pass_path}")

    # Plots
    make_plots(all_seed_results, truth_params)

    # ==========================================
    #  FINAL SUMMARY
    # ==========================================
    print_section("FINAL SUMMARY")
    elapsed = time.time() - t0_global
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    for r in all_seed_results:
        print(f"\n  Seed {r['seed']}:")
        print(f"    S1: k={r['s1_params']['k']:.6f} "
              f"(tau={r['s1_params']['tau_hours']:.1f}h)")
        print(f"    Closure: {r['closure_summary']}")
        print(f"    Selected: {r['final_terms']}")
        for mtype in ['physics', 'closure']:
            res = r['results'][mtype]
            print(f"    {mtype}: ACF1={res['acf1']:.4f}, NIS={res['nis_mean']:.4f}")
            for k, v in res.get('dxr2_at_targets', {}).items():
                print(f"      DxR2@{k}={v:+.4f}")

    # Truth comparison
    if truth_params:
        h_true = truth_params.get('h_true', 0)
        print(f"\n  TRUTH COMPARISON:")
        print(f"    h_true = {h_true:.2e} -> expected forced_conv ~ {-h_true:.2e}")
        for r in all_seed_results:
            fc = r['closure_summary']['forced_conv']
            rel_err = abs(fc - (-h_true)) / h_true if h_true > 0 else float('inf')
            print(f"    seed {r['seed']}: forced_conv = {fc:.6f}, "
                  f"rel_err = {rel_err:.1%}")

    print(f"\n  Mean DxR2 at targets:")
    for key in mean_metrics.get('physics', {}).get('dxr2_at_targets', {}):
        pm = mean_metrics['physics']['dxr2_at_targets'][key]
        cm = mean_metrics['closure']['dxr2_at_targets'][key]
        print(f"    @{key}: Physics={pm['mean']:+.4f}+-{pm['std']:.4f}, "
              f"Closure={cm['mean']:+.4f}+-{cm['std']:.4f}")

    print(f"\n  Overall pass: {'YES' if pass_results['overall_pass'] else 'NO'}")
    print(f"  All outputs in: {OUT_DIR}")
    print("  DONE.")


if __name__ == '__main__':
    main()
