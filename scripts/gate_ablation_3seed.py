"""
Delta-NLL Gate Ablation on Sediment Transport Data (3-seed).

Tests whether the delta-NLL gate is merely cosmetic threshold dressing or
acts as effective false-positive control.  Two selection modes are applied
to the SAME trained 6-term closure model:

  Gate ON:   variance pre-filter (rel_var >= 0.05) + delta-NLL >= 0.001
  Gate OFF:  variance pre-filter only (delta-NLL still computed and logged)

If Gate OFF selects more terms without commensurate skill improvement,
the gate is confirmed as a meaningful false-positive control.

Pipeline per seed:
  1. Load v11.1 S1 physics checkpoint (no retraining)
  2. Train full 6-term S2 closure via scipy L-BFGS-B
  3. Apply BOTH selection modes to same trained closure
  4. Evaluate: physics-only, gate-ON closure, gate-OFF closure
  5. Record selected terms, DxR2(h), ACF(1), NIS

Usage:  python -u scripts/gate_ablation_3seed.py
Output: final_lockbox_gate_ablation/
"""

import os, sys, math, json, time, warnings, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

import torch
torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import paper style
sys.path.insert(0, str(ROOT / "ems_v1" / "figures"))
from paper_style import apply_gmd_style, COLORS, LINESTYLES, DOUBLE_COL

# ==============================================================================
#  CONFIG
# ==============================================================================
SEEDS = [1, 2, 3]
DT = 0.1
FORCE_CPU = True

# Stage 2 (scipy on numpy KF)
S2_MAXITER = 300
TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']
SELECTION_THRESHOLD = 0.05   # variance pre-filter
NLL_DELTA_MIN = 0.001        # delta-NLL gate threshold

# Evaluation
MAX_H = 100  # 10s at 10 Hz
WARMUP_SEC = 50.0

# Paths
CLEAN_DIR = ROOT / "processed_data_10hz_clean_v1"
OUT = ROOT / "final_lockbox_gate_ablation"

# v11.1 checkpoint pattern
V111_DIR = ROOT / "final_lockbox_v11_1_alpha_fix"


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


def compute_acf(e, max_lag=50):
    e_c = e - np.mean(e)
    var = np.var(e)
    n = len(e)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(e_c[:n-l] * e_c[l:]) / (n * var) if l > 0
                     else 1.0 for l in range(max_lag + 1)])


# ==============================================================================
#  NUMPY 2-STATE KF FILTER (no RV drag -- delta=0 always)
# ==============================================================================

def kf_filter_2state(params, cl_params, t, x_obs, v):
    """2-state KF with full tracking. No RV drag (standard physics)."""
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
        u_st = s[1]
        physics_drift = rho * u_st - kap * s[0] * dt_k + c_val * g * dt_k

        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        v_w = v[k-1]
        cl = (-a1*u_st + b1_v*v_w + b2_v*dv_w
              - d1*u_st**2 - d2_v*u_st*abs(v_w) - d3*u_st*abs(u_st))
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
    e = innov[valid]
    S = np.maximum(S_vals[valid], 1e-12)
    return float(0.5 * np.mean(np.log(2 * math.pi * S) + e**2 / S))


# ==============================================================================
#  DxR2 MULTI-HORIZON
# ==============================================================================

def compute_dxr2(params, cl_params, states_x, states_u,
                 t, x_obs, v, max_h=100, eval_start=1):
    """DxR2(h) and MAE(h) for h=1..max_h, oracle future v."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
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
#  S2: SCIPY CLOSURE ON NUMPY KF
# ==============================================================================

def train_s2_scipy(s1_pp, t_train, x_train, v_train,
                   t_val, x_val, v_val, tag="S2"):
    """Optimize 7 closure params (6 terms + log_q_scale) via L-BFGS-B."""
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
#  TERM SELECTION: BOTH MODES FROM SAME TRAINED MODEL
# ==============================================================================

def select_terms_both_modes(cl_params, s1_pp, t, x_obs, v):
    """
    Apply both Gate ON and Gate OFF selection to the same trained closure.

    Returns:
      selected_on:  list of terms passing variance + delta-NLL gate
      selected_off: list of terms passing variance only
      detail:       dict with per-term rel_var, delta_nll, passed_var, passed_dnll
    """
    N = len(x_obs)
    filt = kf_filter_2state(s1_pp, cl_params, t, x_obs, v)
    su = filt['states_u']

    # Compute per-term contributions on second half of data
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
        detail = {tn: {'rel_var': 0.0, 'delta_nll': np.nan,
                        'passed_var': False, 'passed_dnll': False}
                  for tn in TERM_NAMES}
        return [], [], detail

    rel_var = {tn: variances[tn] / total_var for tn in TERM_NAMES}
    candidates = [tn for tn in TERM_NAMES if rel_var[tn] >= SELECTION_THRESHOLD]

    # Compute delta-NLL for ALL variance-passing candidates
    base_nll = kf_nll_numpy(filt['innovations'], filt['S_values'])
    delta_nll = {}
    for tn in candidates:
        cl_without = dict(cl_params)
        cl_without[tn] = 0.0
        filt_without = kf_filter_2state(s1_pp, cl_without, t, x_obs, v)
        nll_without = kf_nll_numpy(filt_without['innovations'],
                                   filt_without['S_values'])
        delta_nll[tn] = nll_without - base_nll

    # Gate ON: variance + delta-NLL
    selected_on = [tn for tn in candidates if delta_nll[tn] >= NLL_DELTA_MIN]

    # Gate OFF: variance only (but delta-NLL still logged)
    selected_off = list(candidates)

    # Build detail dict
    detail = {}
    for tn in TERM_NAMES:
        detail[tn] = {
            'rel_var': rel_var[tn],
            'delta_nll': delta_nll.get(tn, np.nan),
            'passed_var': tn in candidates,
            'passed_dnll': tn in selected_on,
        }

    return selected_on, selected_off, detail


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
    }


# ==============================================================================
#  BUILD ZEROED CLOSURE PARAMS
# ==============================================================================

def build_closure(cl_full, selected_terms):
    """Zero out non-selected terms, keep q_scale."""
    cl = {k: 0.0 for k in TERM_NAMES}
    cl['q_scale'] = cl_full['q_scale']
    for tn in selected_terms:
        cl[tn] = cl_full[tn]
    return cl


# ==============================================================================
#  FIGURE
# ==============================================================================

def make_figure(all_results, out_dir):
    """2-panel figure: (a) term selection frequency, (b) DxR2 vs horizon."""
    apply_gmd_style()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    n_seeds = len(all_results)

    # ------------------------------------------------------------------
    # Panel A: Term selection frequency (grouped bar)
    # ------------------------------------------------------------------
    count_on = {tn: 0 for tn in TERM_NAMES}
    count_off = {tn: 0 for tn in TERM_NAMES}
    for res in all_results:
        for tn in res['selected_on']:
            count_on[tn] += 1
        for tn in res['selected_off']:
            count_off[tn] += 1

    x_pos = np.arange(len(TERM_NAMES))
    w = 0.35
    bars_on = [count_on[tn] for tn in TERM_NAMES]
    bars_off = [count_off[tn] for tn in TERM_NAMES]

    ax_a.bar(x_pos - w/2, bars_on, w, label='Gate ON',
             color=COLORS['closure_2t'], edgecolor='black', linewidth=0.6, alpha=0.85)
    ax_a.bar(x_pos + w/2, bars_off, w, label='Gate OFF',
             color=COLORS['closure_1t'], edgecolor='black', linewidth=0.6, alpha=0.85)
    ax_a.set_xticks(x_pos)
    term_labels = ['$a_1$', '$d_1$', '$d_2$', '$d_3$', '$b_1$', '$b_2$']
    ax_a.set_xticklabels(term_labels)
    ax_a.set_ylabel('Selection count (out of 3 seeds)')
    ax_a.set_ylim(0, n_seeds + 0.5)
    ax_a.set_yticks(range(n_seeds + 1))
    ax_a.legend()
    ax_a.set_title('(a) Term selection frequency')

    # ------------------------------------------------------------------
    # Panel B: DxR2 vs horizon (mean +/- SD across seeds)
    # ------------------------------------------------------------------
    h_range = np.arange(1, MAX_H + 1) * DT

    # Collect per-seed DxR2 arrays
    phys_arr = np.array([r['eval_physics']['dxr2'] for r in all_results])
    on_arr = np.array([r['eval_gate_on']['dxr2'] for r in all_results])
    off_arr = np.array([r['eval_gate_off']['dxr2'] for r in all_results])

    def plot_mean_band(ax, h, arr, color, ls, label):
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax.plot(h, mu, color=color, ls=ls, label=label)
        ax.fill_between(h, mu - sd, mu + sd, color=color, alpha=0.12)

    plot_mean_band(ax_b, h_range, phys_arr,
                   COLORS['physics_kf'], '-', 'Physics only')
    plot_mean_band(ax_b, h_range, on_arr,
                   COLORS['closure_2t'], LINESTYLES['gate_on'], 'Gate ON')
    plot_mean_band(ax_b, h_range, off_arr,
                   COLORS['closure_1t'], LINESTYLES['gate_off'], 'Gate OFF')

    ax_b.axhline(0, color='gray', ls=':', alpha=0.5, lw=0.8)
    ax_b.set_xlabel('Forecast horizon (s)')
    ax_b.set_ylabel('$R^2_{\\Delta x}$')
    ax_b.set_xlim(0, 10)
    ax_b.legend(loc='upper left', fontsize=7.5)
    ax_b.set_title('(b) Displacement-increment skill')

    # Inset: delta DxR2 (OFF - ON)
    ax_ins = ax_b.inset_axes([0.55, 0.08, 0.42, 0.35])
    delta_arr = off_arr - on_arr
    mu_d = np.nanmean(delta_arr, axis=0)
    sd_d = np.nanstd(delta_arr, axis=0)
    ax_ins.plot(h_range, mu_d, color='#555555', lw=1.2)
    ax_ins.fill_between(h_range, mu_d - sd_d, mu_d + sd_d,
                        color='#555555', alpha=0.15)
    ax_ins.axhline(0, color='gray', ls=':', alpha=0.5, lw=0.6)
    ax_ins.set_xlabel('Horizon (s)', fontsize=7)
    ax_ins.set_ylabel('$\\Delta R^2_{\\Delta x}$', fontsize=7)
    ax_ins.set_title('OFF $-$ ON', fontsize=7)
    ax_ins.tick_params(labelsize=6)
    ax_ins.set_xlim(0, 10)

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "gate_ablation_figure.pdf")
    fig.savefig(out_dir / "figures" / "gate_ablation_figure.png")
    plt.close(fig)
    print(f"  Wrote gate_ablation_figure.pdf/.png")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    t0_all = time.time()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DELTA-NLL GATE ABLATION (3-seed)")
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

    # Numpy arrays for training
    t_train = df_train['timestamp'].values.astype(np.float64)
    x_train = df_train['displacement'].values.astype(np.float64)
    v_train = df_train['velocity'].values.astype(np.float64)
    t_val = df_val['timestamp'].values.astype(np.float64)
    x_val = df_val['displacement'].values.astype(np.float64)
    v_val = df_val['velocity'].values.astype(np.float64)

    # Warm evaluation array
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

    null_cl = {k: 0.0 for k in TERM_NAMES}
    null_cl['q_scale'] = 1.0

    # ==================================================================
    #  SEED LOOP
    # ==================================================================
    all_results = []
    table_rows = []
    detail_rows = []

    for seed in SEEDS:
        print_section(f"SEED {seed}")

        seed_dir = OUT / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------
        #  Load S1 physics checkpoint
        # --------------------------------------------------------------
        ckpt_path = V111_DIR / f"seed{seed}" / "checkpoints" / f"stage1_physics_seed{seed}.pth"
        assert ckpt_path.exists(), f"Missing S1 checkpoint: {ckpt_path}"
        print(f"  Loading S1 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        s1_params = ckpt['params']
        print(f"  S1: alpha={s1_params['alpha']:.4f} "
              f"kappa={s1_params['kappa']:.4f} c={s1_params['c']:.4f} "
              f"vc={s1_params['vc']:.4f}")

        pp = {
            'alpha': s1_params['alpha'], 'c': s1_params['c'],
            'vc': s1_params['vc'], 'kappa': s1_params['kappa'],
            'qx': s1_params['qx'], 'qu': s1_params['qu'],
            'R': s1_params['R'],
            'P0_xx': s1_params['P0_xx'], 'P0_uu': s1_params['P0_uu'],
        }

        # --------------------------------------------------------------
        #  Train 6-term S2 closure
        # --------------------------------------------------------------
        print(f"\n  Training 6-term S2 closure (seed {seed})...")
        np.random.seed(seed)
        cl_full, s2_tr_nll, s2_val_nll, s2_nfev = train_s2_scipy(
            pp, t_train, x_train, v_train,
            t_val, x_val, v_val, tag=f"S2-seed{seed}")

        print(f"  S2 coefficients:")
        for tn in TERM_NAMES:
            print(f"    {tn} = {cl_full[tn]:.6f}")
        print(f"    q_scale = {cl_full['q_scale']:.4f}")

        # Save trained coefficients
        coeff_path = seed_dir / f"closure_6t_scipy_seed{seed}.json"
        with open(coeff_path, 'w') as f:
            json.dump(cl_full, f, indent=2)
        print(f"  Saved: {coeff_path}")

        # --------------------------------------------------------------
        #  Term selection: both modes
        # --------------------------------------------------------------
        print(f"\n  Term selection (both modes)...")
        selected_on, selected_off, detail = select_terms_both_modes(
            cl_full, pp, t_val, x_val, v_val)

        print(f"  Gate ON  selected: {selected_on} ({len(selected_on)} terms)")
        print(f"  Gate OFF selected: {selected_off} ({len(selected_off)} terms)")
        print(f"  Per-term detail:")
        for tn in TERM_NAMES:
            d = detail[tn]
            dnll_str = f"{d['delta_nll']:.6f}" if np.isfinite(d['delta_nll']) else "N/A"
            flags = []
            if d['passed_var']: flags.append("VAR")
            if d['passed_dnll']: flags.append("DNLL")
            print(f"    {tn}: rel_var={d['rel_var']:.4f} "
                  f"delta_nll={dnll_str} [{', '.join(flags) or '-'}]")

        # Collect detail rows for CSV
        for tn in TERM_NAMES:
            d = detail[tn]
            detail_rows.append({
                'seed': seed, 'term': tn,
                'coeff': cl_full[tn],
                'rel_var': d['rel_var'],
                'delta_nll': d['delta_nll'] if np.isfinite(d['delta_nll']) else None,
                'passed_var': d['passed_var'],
                'passed_dnll': d['passed_dnll'],
                'selected_on': tn in selected_on,
                'selected_off': tn in selected_off,
            })

        # --------------------------------------------------------------
        #  Evaluate: physics-only, gate ON, gate OFF
        # --------------------------------------------------------------
        print(f"\n  Evaluating (warm start)...")
        cl_on = build_closure(cl_full, selected_on)
        cl_off = build_closure(cl_full, selected_off)

        r_phys = evaluate_model(f"physics_s{seed}", pp, null_cl,
                                t_warm, x_warm, v_warm, eval_start_warm, MAX_H)
        r_on = evaluate_model(f"gate_on_s{seed}", pp, cl_on,
                              t_warm, x_warm, v_warm, eval_start_warm, MAX_H)
        r_off = evaluate_model(f"gate_off_s{seed}", pp, cl_off,
                               t_warm, x_warm, v_warm, eval_start_warm, MAX_H)

        # Collect table rows
        for mode, r, sel in [('physics', r_phys, []),
                              ('gate_on', r_on, selected_on),
                              ('gate_off', r_off, selected_off)]:
            table_rows.append({
                'seed': seed, 'mode': mode,
                'n_terms': len(sel),
                'terms': '+'.join(sel) if sel else 'none',
                'acf1': r['acf1'],
                'nis': r['nis_mean'],
                'cov90': r['cov90'],
                'dxr2_01s': r['dxr2'][0],
                'dxr2_1s': r['dxr2'][9],
                'dxr2_2s': r['dxr2'][19],
            })

        # Collect for figure
        all_results.append({
            'seed': seed,
            'selected_on': selected_on,
            'selected_off': selected_off,
            'detail': detail,
            'cl_full': cl_full,
            'eval_physics': r_phys,
            'eval_gate_on': r_on,
            'eval_gate_off': r_off,
        })

    # ==================================================================
    #  TABLES
    # ==================================================================
    print_section("TABLES")

    df_table = pd.DataFrame(table_rows)
    table_path = OUT / "tables" / "gate_ablation_table.csv"
    df_table.to_csv(table_path, index=False)
    print(f"  Wrote {table_path}")

    df_detail = pd.DataFrame(detail_rows)
    detail_path = OUT / "tables" / "term_selection_detail.csv"
    df_detail.to_csv(detail_path, index=False)
    print(f"  Wrote {detail_path}")

    # Print summary table
    print(f"\n  {'Seed':>4} {'Mode':>10} {'#Terms':>6} {'Terms':>20} "
          f"{'ACF1':>6} {'NIS':>6} {'DxR2@1s':>8} {'DxR2@2s':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*6} {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for _, row in df_table.iterrows():
        print(f"  {row['seed']:4d} {row['mode']:>10} {row['n_terms']:6d} "
              f"{row['terms']:>20} {row['acf1']:6.3f} {row['nis']:6.3f} "
              f"{row['dxr2_1s']:+8.3f} {row['dxr2_2s']:+8.3f}")

    # ==================================================================
    #  FIGURE
    # ==================================================================
    print_section("FIGURE")
    make_figure(all_results, OUT)

    # ==================================================================
    #  SUMMARY JSON
    # ==================================================================
    print_section("SUMMARY")

    # Compute pass criteria
    inflation_count = 0
    for res in all_results:
        if len(res['selected_off']) > len(res['selected_on']):
            inflation_count += 1
    pass_inflation = inflation_count >= 2

    delta_dxr2_1s = []
    for res in all_results:
        d = res['eval_gate_off']['dxr2'][9] - res['eval_gate_on']['dxr2'][9]
        delta_dxr2_1s.append(d)
    mean_delta = np.mean(delta_dxr2_1s)
    # Original criterion: |delta| < 0.02 (no commensurate skill gain)
    # If delta < 0, gate OFF is WORSE -- even stronger evidence
    pass_no_gain = mean_delta < 0.02  # gate OFF does not improve over gate ON
    gate_off_hurts = mean_delta < -0.02  # gate OFF actively degrades skill

    print(f"  PASS CRITERIA:")
    print(f"    Inflation (>=1 extra term in >=2/3 seeds): "
          f"{inflation_count}/3 -> {'PASS' if pass_inflation else 'FAIL'}")
    print(f"    No commensurate skill gain (mean DxR2@1s(OFF-ON) < 0.02): "
          f"{mean_delta:+.4f} -> {'PASS' if pass_no_gain else 'FAIL'}")
    if gate_off_hurts:
        print(f"    ** STRONGER: Gate OFF actively DEGRADES skill by {abs(mean_delta):.3f} **")
    for i, (seed, d) in enumerate(zip(SEEDS, delta_dxr2_1s)):
        print(f"      Seed {seed}: DxR2@1s delta = {d:+.4f}")

    # Build serializable summary
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': SEEDS,
        'config': {
            'selection_threshold': SELECTION_THRESHOLD,
            'nll_delta_min': NLL_DELTA_MIN,
            's2_maxiter': S2_MAXITER,
            'max_h': MAX_H,
            'warmup_sec': WARMUP_SEC,
        },
        'per_seed': [],
        'pass_criteria': {
            'inflation_seeds': inflation_count,
            'inflation_pass': bool(pass_inflation),
            'mean_delta_dxr2_1s': float(mean_delta),
            'per_seed_delta_dxr2_1s': [float(d) for d in delta_dxr2_1s],
            'no_gain_pass': bool(pass_no_gain),
            'gate_off_hurts': bool(gate_off_hurts),
            'overall_pass': bool(pass_inflation and pass_no_gain),
        },
        'runtime_s': time.time() - t0_all,
    }

    for res in all_results:
        seed_summary = {
            'seed': res['seed'],
            'selected_on': res['selected_on'],
            'selected_off': res['selected_off'],
            'n_terms_on': len(res['selected_on']),
            'n_terms_off': len(res['selected_off']),
            'term_inflation': len(res['selected_off']) - len(res['selected_on']),
            'coefficients': sanitize(res['cl_full']),
            'detail': sanitize(res['detail']),
            'physics_dxr2_1s': res['eval_physics']['dxr2'][9],
            'physics_dxr2_2s': res['eval_physics']['dxr2'][19],
            'gate_on_dxr2_1s': res['eval_gate_on']['dxr2'][9],
            'gate_on_dxr2_2s': res['eval_gate_on']['dxr2'][19],
            'gate_off_dxr2_1s': res['eval_gate_off']['dxr2'][9],
            'gate_off_dxr2_2s': res['eval_gate_off']['dxr2'][19],
            'gate_on_acf1': res['eval_gate_on']['acf1'],
            'gate_off_acf1': res['eval_gate_off']['acf1'],
            'gate_on_nis': res['eval_gate_on']['nis_mean'],
            'gate_off_nis': res['eval_gate_off']['nis_mean'],
        }
        summary['per_seed'].append(sanitize(seed_summary))

    summary_path = OUT / "gate_ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")

    elapsed = time.time() - t0_all
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  All outputs in {OUT}")
    print("  DONE.")


if __name__ == '__main__':
    main()
