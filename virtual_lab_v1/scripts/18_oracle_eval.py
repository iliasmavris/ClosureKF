"""
18_oracle_eval.py - Oracle evaluation: physics-only vs oracle-library gap
=========================================================================
For each condition dataset:
1. Load truth_states_raw.csv (dt_sim resolution)
2. Identify FREE indices (at_pin == 0, valid a_force)
3. Split FREE indices temporally into TRAIN (60%) / VAL (20%) / TEST (20%)
4. Load a_true from 'a_force' column (force-balance RHS/m_eff, logged by
   the simulator). Falls back to smoothed dv_p/dt if column absent.
5. Fit physics-only on TRAIN: a_phys = -alpha*v_p - kappa*x + c*relu(u_b^2 - uc^2)
6. Select ridge alpha on VAL from grid [0.01, 0.1, 1, 10, 100]
7. Fit oracle library on TRAIN with best alpha, evaluate on TEST

Hard gate: require N_free_total >= N_MIN_FREE.  "Degenerate" means too few
free samples overall — not an artifact of bad temporal splitting.

Regime classification: ER < 0.50 = 'intermittent', ER >= 0.50 = 'continuous'.

Library terms (matching manuscript discovery library, truth-model variables):
  Term 1 (a1): -v_p            [linear particle damping]
  Term 2 (d1): -v_p^2          [quadratic self-drag, unsigned]
  Term 3 (d2): -v_p * |u_b|    [cross-drag, flow-modulated]
  Term 4 (d3): -v_p * |v_p|    [quadratic absolute self-drag]
  Term 5 (b1): u_b             [direct flow coupling]
  Term 6 (b2): du_b/dt         [flow acceleration coupling]

Outputs:
  outputs/paper_figs/oracle_report.json   - full per-condition results
  outputs/paper_figs/oracle_summary.csv   - compact table for plotting
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.optimize import nnls
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent

TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']

# Split ratios (of FREE indices)
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# TEST_FRAC = 0.20 (implicit)

# Alpha selection grid (standardized space)
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

UC_STEP = 0.005
UC_MAX = 0.35

# Hard gate: minimum free time (seconds); n_min_free = ceil(MIN_FREE_SECONDS / dt)
MIN_FREE_SECONDS = 25.0

# Regime boundary
ER_INTERMITTENT_MAX = 0.50

# Status codes
STATUS_OK = 'ok'
STATUS_DEGENERATE = 'degenerate'   # too few free samples overall
STATUS_NO_DATA = 'no_data'         # missing files

# Backward compat exports
RIDGE_ALPHA = 1.0  # default; overridden by val selection


# ---- Core functions ----

def compute_a_true_numdiff(v_p, dt, smooth_hz=10.0):
    """Fallback: acceleration from v_p via smoothed central finite differences."""
    fs = 1.0 / dt
    if smooth_hz is not None and smooth_hz < fs / 2:
        sos = butter(4, smooth_hz, btype='low', fs=fs, output='sos')
        v_smooth = sosfiltfilt(sos, v_p)
    else:
        v_smooth = v_p.copy()

    a = np.zeros_like(v_smooth)
    a[1:-1] = (v_smooth[2:] - v_smooth[:-2]) / (2.0 * dt)
    a[0] = (v_smooth[1] - v_smooth[0]) / dt
    a[-1] = (v_smooth[-1] - v_smooth[-2]) / dt
    return a


def fit_physics_profile(a_true, v_p, x, u_b):
    """
    Profile grid search over uc, NNLS for [alpha, kappa, c].

    Physics-only model (manuscript family):
      a_phys = -alpha*v_p - kappa*x + c*relu(u_b^2 - uc^2)

    Returns (alpha, kappa, c, uc), best_mse_train.
    """
    uc_grid = np.arange(0, UC_MAX, UC_STEP)
    best_mse = np.inf
    best_params = None

    for uc in uc_grid:
        exceedance = np.maximum(0, u_b**2 - uc**2)
        # Features with signs baked in so NNLS gives non-negative coefficients
        X = np.column_stack([-v_p, -x, exceedance])
        theta, rnorm = nnls(X, a_true)
        mse = rnorm**2 / len(a_true)
        if mse < best_mse:
            best_mse = mse
            best_params = (float(theta[0]), float(theta[1]),
                           float(theta[2]), float(uc))

    return best_params, best_mse


def physics_predict(params, v_p, x, u_b):
    """Predict acceleration from physics-only model."""
    alpha, kappa, c, uc = params
    return -alpha * v_p - kappa * x + c * np.maximum(0, u_b**2 - uc**2)


def build_library(v_p, u_b, du_b):
    """Build 6-term closure library matrix [N x 6]."""
    return np.column_stack([
        -v_p,                # a1: linear particle damping
        -v_p**2,             # d1: quadratic self-drag (unsigned)
        -v_p * np.abs(u_b),  # d2: cross-drag
        -v_p * np.abs(v_p),  # d3: absolute quadratic self-drag
        u_b,                 # b1: direct flow coupling
        du_b,                # b2: flow acceleration coupling
    ])


def _standardize(Phi_train):
    """Compute feature means and stds from train set."""
    mu = Phi_train.mean(axis=0)
    sigma = Phi_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0  # near-constant features -> unscaled
    return mu, sigma


def _ridge_solve(Phi_s, y_s, alpha, n_feat):
    """Solve ridge in standardized space."""
    return np.linalg.solve(
        Phi_s.T @ Phi_s + alpha * np.eye(n_feat),
        Phi_s.T @ y_s
    )


def standardized_ridge_fit(Phi_train, y_train, Phi_test, alpha):
    """
    Ridge regression with feature standardization.

    Returns: theta_orig, intercept, y_hat_train, y_hat_test
    """
    n_feat = Phi_train.shape[1]
    mu, sigma = _standardize(Phi_train)

    Phi_s = (Phi_train - mu) / sigma
    Phi_test_s = (Phi_test - mu) / sigma

    y_mu = y_train.mean()
    y_s = y_train - y_mu

    theta_s = _ridge_solve(Phi_s, y_s, alpha, n_feat)

    y_hat_train = Phi_s @ theta_s + y_mu
    y_hat_test = Phi_test_s @ theta_s + y_mu

    theta_orig = theta_s / sigma
    intercept = y_mu - np.dot(theta_s, mu / sigma)

    return theta_orig, intercept, y_hat_train, y_hat_test


def select_alpha_on_val(Phi_train, r_train, Phi_val, r_val):
    """
    Select best ridge alpha from ALPHA_GRID using validation MSE.

    Returns (best_alpha, val_mses_dict).
    """
    n_feat = Phi_train.shape[1]
    mu, sigma = _standardize(Phi_train)
    Phi_s = (Phi_train - mu) / sigma
    Phi_val_s = (Phi_val - mu) / sigma
    y_mu = r_train.mean()
    y_s = r_train - y_mu

    best_alpha = ALPHA_GRID[len(ALPHA_GRID) // 2]  # default to middle
    best_val_mse = np.inf
    val_mses = {}

    for alpha in ALPHA_GRID:
        theta_s = _ridge_solve(Phi_s, y_s, alpha, n_feat)
        r_hat_val = Phi_val_s @ theta_s + y_mu
        val_mse = float(np.mean((r_val - r_hat_val)**2))
        val_mses[alpha] = val_mse
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = alpha

    return best_alpha, val_mses


# ---- Degenerate result template ----

def _degenerate_result(condition_id, event_rate, n_free, a_source, detail, regime):
    """Standardized degenerate result dict (never None)."""
    return {
        'condition_id': condition_id,
        'status': STATUS_DEGENERATE,
        'status_detail': detail,
        'event_rate': event_rate,
        'regime': regime,
        'n_free': n_free,
        'n_train_free': 0,
        'n_val_free': 0,
        'n_test_free': 0,
        'a_source': a_source,
        'params_source': None,
        'best_alpha': None,
        'phys_params': {'alpha': None, 'kappa': None, 'c': None, 'u_c': None},
        'theta_lib': {tn: None for tn in TERM_NAMES},
        'intercept': None,
        'MSE_phys': None,
        'MSE_oracle': None,
        'gain_oracle': None,
        'oracle_stable': None,
        'var_a_true': None,
        'R2_phys': None,
        'R2_oracle': None,
        'corr_r_d2': None,
        'corr_a_d2': None,
        # No-a1 library
        'best_alpha_no_a1': None,
        'val_mses_no_a1': None,
        'theta_lib_no_a1': {tn: None for tn in TERM_NAMES[1:]},
        'intercept_no_a1': None,
        'MSE_oracle_no_a1': None,
        'gain_oracle_no_a1': None,
        'R2_oracle_no_a1': None,
        'oracle_stable_no_a1': None,
    }


# ---- Per-condition evaluation ----

def evaluate_condition(cond_dir, truth_phys_params=None):
    """
    Full oracle evaluation for one condition.

    Args:
        cond_dir: Path to condition directory.
        truth_phys_params: Optional dict {condition_id: {alpha, kappa, c, uc}}.
            If provided and matching, skips NNLS profile fit.

    Always returns a dict (never None). The 'status' field indicates:
      'ok'         - valid evaluation with sufficient data
      'degenerate' - too few free samples overall; metrics are NaN
      'no_data'    - missing files
    """
    meta_path = cond_dir / "meta.json"
    truth_path = cond_dir / "truth_states_raw.csv"

    if not meta_path.exists() or not truth_path.exists():
        return {
            'condition_id': cond_dir.name,
            'status': STATUS_NO_DATA,
            'status_detail': 'missing meta.json or truth_states_raw.csv',
        }

    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.read_csv(truth_path)

    cid = meta.get('condition_id', cond_dir.name)

    dt = meta['config']['integration']['dt_sim']
    er = meta['event_rate']
    regime = 'intermittent' if er < ER_INTERMITTENT_MAX else 'continuous'

    t = df['time'].values
    x = df['x'].values
    v_p = df['v_p'].values
    at_pin = df['at_pin'].values
    u_b = df['u_b'].values

    # du_b fallback: compute from u_b if column absent
    if 'du_b' in df.columns:
        du_b = df['du_b'].values
    else:
        du_b = np.gradient(u_b, dt)

    # ---- Get a_true: prefer force-balance, fall back to num diff ----
    if 'a_force' in df.columns:
        a_force_raw = df['a_force'].values
        a_source = 'force_balance'
    else:
        a_force_raw = compute_a_true_numdiff(v_p, dt)
        a_source = 'numdiff_fallback'

    # ---- Identify FREE indices (at_pin==0, valid a_force) ----
    is_free = (at_pin == 0)
    valid_a = ~np.isnan(a_force_raw)
    free_idx = np.where(is_free & valid_a)[0]
    n_free = len(free_idx)

    # ---- Hard gate: enough free samples overall? ----
    n_min_free = int(np.ceil(MIN_FREE_SECONDS / dt))
    if n_free < n_min_free:
        detail = f"N_free={n_free} (min={n_min_free}, {MIN_FREE_SECONDS}s @ dt={dt})"
        print(f"  DEGENERATE: {detail}")
        return _degenerate_result(
            cid, er, n_free, a_source, detail, regime)

    # ---- Split FREE indices into train / val / test (temporal order) ----
    n_train = int(n_free * TRAIN_FRAC)
    n_val = int(n_free * VAL_FRAC)
    # remainder goes to test
    idx_train = free_idx[:n_train]
    idx_val = free_idx[n_train:n_train + n_val]
    idx_test = free_idx[n_train + n_val:]

    n_train_free = len(idx_train)
    n_val_free = len(idx_val)
    n_test_free = len(idx_test)

    a_true = a_force_raw  # use the selected source

    # ---- 1. Physics-only fit on TRAIN ----
    # Check for provided truth physics params
    tp = None
    if truth_phys_params is not None:
        tp = truth_phys_params.get(cid) or truth_phys_params.get(cond_dir.name)

    if tp is not None:
        phys_params = (tp['alpha'], tp['kappa'], tp['c'], tp['uc'])
        train_mse = None
        params_source = 'truth_provided'
    else:
        phys_params, train_mse = fit_physics_profile(
            a_true[idx_train], v_p[idx_train],
            x[idx_train], u_b[idx_train]
        )
        params_source = 'nnls_fit'

    # Physics on TEST
    a_phys_test = physics_predict(
        phys_params, v_p[idx_test], x[idx_test], u_b[idx_test]
    )
    MSE_phys = float(np.mean((a_true[idx_test] - a_phys_test)**2))

    # ---- 2. Library: alpha selection on VAL, then fit on TRAIN ----
    a_phys_train = physics_predict(
        phys_params, v_p[idx_train], x[idx_train], u_b[idx_train]
    )
    r_train = a_true[idx_train] - a_phys_train

    a_phys_val = physics_predict(
        phys_params, v_p[idx_val], x[idx_val], u_b[idx_val]
    )
    r_val = a_true[idx_val] - a_phys_val

    Phi_train = build_library(v_p[idx_train], u_b[idx_train], du_b[idx_train])
    Phi_val = build_library(v_p[idx_val], u_b[idx_val], du_b[idx_val])
    Phi_test = build_library(v_p[idx_test], u_b[idx_test], du_b[idx_test])

    # Select alpha on val
    best_alpha, val_mses = select_alpha_on_val(Phi_train, r_train, Phi_val, r_val)

    # Fit with best alpha, evaluate on test
    theta_lib, intercept, _, r_hat_test = standardized_ridge_fit(
        Phi_train, r_train, Phi_test, alpha=best_alpha
    )

    # Oracle prediction on test
    a_oracle_test = a_phys_test + r_hat_test
    MSE_oracle = float(np.mean((a_true[idx_test] - a_oracle_test)**2))

    # ---- 2b. No-a1 library: drop column 0 (a1 = -v_p) ----
    TERM_NAMES_NO_A1 = TERM_NAMES[1:]  # d1,d2,d3,b1,b2
    Phi_train_no_a1 = Phi_train[:, 1:]
    Phi_val_no_a1 = Phi_val[:, 1:]
    Phi_test_no_a1 = Phi_test[:, 1:]

    best_alpha_no_a1, val_mses_no_a1 = select_alpha_on_val(
        Phi_train_no_a1, r_train, Phi_val_no_a1, r_val)

    theta_lib_no_a1, intercept_no_a1, _, r_hat_test_no_a1 = standardized_ridge_fit(
        Phi_train_no_a1, r_train, Phi_test_no_a1, alpha=best_alpha_no_a1)

    a_oracle_test_no_a1 = a_phys_test + r_hat_test_no_a1
    MSE_oracle_no_a1 = float(np.mean((a_true[idx_test] - a_oracle_test_no_a1)**2))

    # ---- 3. Metrics ----
    var_a = float(np.var(a_true[idx_test]))
    R2_phys = 1.0 - MSE_phys / var_a if var_a > 1e-20 else 0.0
    R2_oracle = 1.0 - MSE_oracle / var_a if var_a > 1e-20 else 0.0

    if MSE_phys > 1e-12:
        gain_oracle = float((MSE_phys - MSE_oracle) / MSE_phys)
    else:
        gain_oracle = 0.0  # physics near-perfect; gain undefined

    # Flag oracle instability (library worse than physics by >5x)
    oracle_stable = bool(MSE_oracle <= 5.0 * max(MSE_phys, 1e-12))

    # No-a1 metrics
    R2_oracle_no_a1 = 1.0 - MSE_oracle_no_a1 / var_a if var_a > 1e-20 else 0.0
    if MSE_phys > 1e-12:
        gain_oracle_no_a1 = float((MSE_phys - MSE_oracle_no_a1) / MSE_phys)
    else:
        gain_oracle_no_a1 = 0.0
    oracle_stable_no_a1 = bool(MSE_oracle_no_a1 <= 5.0 * max(MSE_phys, 1e-12))

    # Non-circularity correlations
    d2_feature = -v_p[idx_test] * np.abs(u_b[idx_test])
    r_test = a_true[idx_test] - a_phys_test

    # Primary: residual-based (what the closure could explain)
    if np.std(d2_feature) > 1e-20 and np.std(r_test) > 1e-20:
        corr_r_d2 = float(np.corrcoef(r_test, d2_feature)[0, 1])
    else:
        corr_r_d2 = 0.0

    # Compat: total-acceleration (matches audit B3 concept)
    if np.std(d2_feature) > 1e-20 and np.std(a_true[idx_test]) > 1e-20:
        corr_a_d2 = float(np.corrcoef(a_true[idx_test], d2_feature)[0, 1])
    else:
        corr_a_d2 = 0.0

    status_detail = f'a_source={a_source}, params_source={params_source}, alpha={best_alpha}'
    if not oracle_stable:
        status_detail += '; oracle_unstable'

    result = {
        'condition_id': cid,
        'status': STATUS_OK,
        'status_detail': status_detail,
        'event_rate': er,
        'regime': regime,
        'n_free': n_free,
        'n_train_free': n_train_free,
        'n_val_free': n_val_free,
        'n_test_free': n_test_free,
        'a_source': a_source,
        'params_source': params_source,
        'best_alpha': best_alpha,
        'val_mses': val_mses,
        'phys_params': {
            'alpha': float(phys_params[0]),
            'kappa': float(phys_params[1]),
            'c': float(phys_params[2]),
            'u_c': float(phys_params[3]),
        },
        'theta_lib': dict(zip(TERM_NAMES, [float(t) for t in theta_lib])),
        'intercept': float(intercept),
        'MSE_phys': MSE_phys,
        'MSE_oracle': MSE_oracle,
        'gain_oracle': gain_oracle,
        'oracle_stable': oracle_stable,
        'var_a_true': var_a,
        'R2_phys': float(R2_phys),
        'R2_oracle': float(R2_oracle),
        'corr_r_d2': corr_r_d2,
        'corr_a_d2': corr_a_d2,
        # No-a1 library
        'best_alpha_no_a1': best_alpha_no_a1,
        'val_mses_no_a1': val_mses_no_a1,
        'theta_lib_no_a1': dict(zip(TERM_NAMES_NO_A1, [float(t) for t in theta_lib_no_a1])),
        'intercept_no_a1': float(intercept_no_a1),
        'MSE_oracle_no_a1': MSE_oracle_no_a1,
        'gain_oracle_no_a1': gain_oracle_no_a1,
        'R2_oracle_no_a1': float(R2_oracle_no_a1),
        'oracle_stable_no_a1': oracle_stable_no_a1,
    }
    return result


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description='Oracle evaluation')
    parser.add_argument('--datasets_root', default=None,
                        help='Root dir containing condition_* dirs '
                             '(default: datasets/)')
    parser.add_argument('--truth_phys_params', default=None,
                        help='Path to JSON: {condition_id: {alpha, kappa, c, uc}}. '
                             'If provided, skips NNLS profile fit for matching conditions.')
    args = parser.parse_args()

    # Load truth physics params if provided
    truth_phys = None
    if args.truth_phys_params:
        with open(args.truth_phys_params) as f:
            truth_phys = json.load(f)
        print(f"  Truth physics params loaded for {len(truth_phys)} conditions")

    print("=" * 60)
    print("ORACLE EVALUATION - Physics-only vs Library")
    print(f"  Split: {TRAIN_FRAC:.0%} train / {VAL_FRAC:.0%} val / "
          f"{1-TRAIN_FRAC-VAL_FRAC:.0%} test (of FREE indices)")
    print(f"  Alpha grid: {ALPHA_GRID}")
    print(f"  Min free time: {MIN_FREE_SECONDS}s")
    print(f"  Regime boundary: ER < {ER_INTERMITTENT_MAX}")
    print("=" * 60)

    if args.datasets_root:
        datasets_dir = Path(args.datasets_root)
    else:
        datasets_dir = ROOT / "datasets"

    cond_dirs = sorted(datasets_dir.glob("condition_*"))

    if not cond_dirs:
        print(f"ERROR: No conditions found in {datasets_dir}")
        sys.exit(1)

    print(f"Found {len(cond_dirs)} conditions in {datasets_dir}")

    results = []
    n_ok = 0
    n_degen = 0
    n_nodata = 0

    for cd in cond_dirs:
        cid = cd.name
        print(f"\n--- {cid} ---")
        r = evaluate_condition(cd, truth_phys_params=truth_phys)
        results.append(r)

        if r['status'] == STATUS_OK:
            n_ok += 1
            stab = '' if r['oracle_stable'] else ' [UNSTABLE]'
            print(f"  [{r['a_source']}] [{r['regime']}] "
                  f"alpha*={r['best_alpha']}{stab}")
            print(f"  N_free={r['n_free']} "
                  f"(train={r['n_train_free']}, "
                  f"val={r['n_val_free']}, "
                  f"test={r['n_test_free']})")
            print(f"  MSE_phys={r['MSE_phys']:.4e}, "
                  f"MSE_oracle={r['MSE_oracle']:.4e}")
            print(f"  Gain={r['gain_oracle']:.3f}  "
                  f"R2_phys={r['R2_phys']:.3f}  "
                  f"R2_oracle={r['R2_oracle']:.3f}")
            print(f"  Phys: alpha={r['phys_params']['alpha']:.4f}, "
                  f"kappa={r['phys_params']['kappa']:.4f}, "
                  f"c={r['phys_params']['c']:.4f}, "
                  f"uc={r['phys_params']['u_c']:.4f}")
            print(f"  Lib: b2={r['theta_lib']['b2']:.4f}, "
                  f"d2={r['theta_lib']['d2']:.4f}")
            print(f"  No-a1: Gain={r['gain_oracle_no_a1']:.3f}  "
                  f"R2={r['R2_oracle_no_a1']:.3f}  "
                  f"alpha*={r['best_alpha_no_a1']}")
        elif r['status'] == STATUS_DEGENERATE:
            n_degen += 1
        else:
            n_nodata += 1

    if not results:
        print("ERROR: No conditions processed")
        sys.exit(1)

    # ---- Output ----
    out_dir = ROOT / "outputs" / "paper_figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # oracle_report.json (full detail)
    # Strip val_mses for JSON (keys must be strings)
    for r in results:
        if 'val_mses' in r and r['val_mses'] is not None:
            r['val_mses'] = {str(k): v for k, v in r['val_mses'].items()}
        if 'val_mses_no_a1' in r and r['val_mses_no_a1'] is not None:
            r['val_mses_no_a1'] = {str(k): v for k, v in r['val_mses_no_a1'].items()}

    report = {
        'n_conditions': len(results),
        'n_ok': n_ok,
        'n_degenerate': n_degen,
        'n_no_data': n_nodata,
        'settings': {
            'alpha_grid': ALPHA_GRID,
            'train_frac': TRAIN_FRAC,
            'val_frac': VAL_FRAC,
            'uc_grid_step': UC_STEP,
            'uc_grid_max': UC_MAX,
            'min_free_seconds': MIN_FREE_SECONDS,
            'er_intermittent_max': ER_INTERMITTENT_MAX,
            'standardize_features': True,
        },
        'conditions': results,
    }
    report_path = out_dir / "oracle_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    # oracle_summary.csv (all conditions, including degenerate)
    rows = []
    for r in results:
        row = {
            'condition_id': r['condition_id'],
            'status': r['status'],
            'regime': r.get('regime'),
            'event_rate': r.get('event_rate'),
            'n_free': r.get('n_free'),
            'n_train_free': r.get('n_train_free'),
            'n_val_free': r.get('n_val_free'),
            'n_test_free': r.get('n_test_free'),
            'a_source': r.get('a_source'),
            'best_alpha': r.get('best_alpha'),
            'MSE_phys': r.get('MSE_phys'),
            'MSE_oracle': r.get('MSE_oracle'),
            'gain_oracle': r.get('gain_oracle'),
            'oracle_stable': r.get('oracle_stable'),
            'R2_phys': r.get('R2_phys'),
            'R2_oracle': r.get('R2_oracle'),
            'corr_r_d2': r.get('corr_r_d2'),
            'corr_a_d2': r.get('corr_a_d2'),
            # No-a1 library
            'best_alpha_no_a1': r.get('best_alpha_no_a1'),
            'MSE_oracle_no_a1': r.get('MSE_oracle_no_a1'),
            'gain_oracle_no_a1': r.get('gain_oracle_no_a1'),
            'R2_oracle_no_a1': r.get('R2_oracle_no_a1'),
            'oracle_stable_no_a1': r.get('oracle_stable_no_a1'),
        }
        if r['status'] == STATUS_OK:
            row['alpha'] = r['phys_params']['alpha']
            row['kappa'] = r['phys_params']['kappa']
            row['c'] = r['phys_params']['c']
            row['u_c'] = r['phys_params']['u_c']
            for tn in TERM_NAMES:
                row[tn] = r['theta_lib'][tn]
        rows.append(row)

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "oracle_summary.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")

    # Print summary table
    ok_results = [r for r in results if r['status'] == STATUS_OK]
    print(f"\n{'='*78}")
    print(f"Status: {n_ok} ok, {n_degen} degenerate, {n_nodata} no_data")
    print(f"{'='*78}")
    hdr = (f"{'Condition':<15} {'Rgm':>5} {'ER':>5} {'a*':>5} {'MSE_phys':>10} "
           f"{'MSE_orac':>10} {'Gain':>6} {'R2_p':>6} {'R2_o':>6}")
    print(hdr)
    print("-" * 78)
    for r in results:
        st = r['status'][:4].upper()
        if r['status'] == STATUS_OK:
            rgm = r['regime'][:5]
            stab = '!' if not r['oracle_stable'] else ' '
            print(f"{r['condition_id']:<15} {rgm:>5} {r['event_rate']:>5.3f} "
                  f"{r['best_alpha']:>5.2f} "
                  f"{r['MSE_phys']:>10.2e} {r['MSE_oracle']:>10.2e} "
                  f"{r['gain_oracle']:>6.3f} {r['R2_phys']:>6.3f} "
                  f"{r['R2_oracle']:>6.3f}{stab}")
        else:
            er = r.get('event_rate', 0)
            er_str = f"{er:.3f}" if er is not None else "  N/A"
            rgm = r.get('regime', '?')[:5]
            nf = r.get('n_free', 0)
            print(f"{r['condition_id']:<15} {rgm:>5} {er_str:>5} "
                  f"{'---':>5} {'---':>10} {'---':>10} "
                  f"{'---':>6} {'---':>6} {'---':>6}  "
                  f"[N_free={nf}]")

    if ok_results:
        # Split by regime
        intermittent = [r for r in ok_results if r['regime'] == 'intermittent']
        continuous = [r for r in ok_results if r['regime'] == 'continuous']
        stable = [r for r in ok_results if r['oracle_stable']]
        n_unstable = len(ok_results) - len(stable)

        print("-" * 78)
        for label, subset in [('ALL ok', ok_results),
                               ('Intermittent', intermittent),
                               ('Continuous', continuous),
                               ('Stable', stable)]:
            if not subset:
                continue
            mg = np.mean([r['gain_oracle'] for r in subset])
            mr2p = np.mean([r['R2_phys'] for r in subset])
            mr2o = np.mean([r['R2_oracle'] for r in subset])
            n = len(subset)
            print(f"{'MEAN '+label:<15} {'':>5} {'':>5} "
                  f"{'':>5} {'':>10} {'':>10} "
                  f"{mg:>6.3f} {mr2p:>6.3f} {mr2o:>6.3f}  (n={n})")

        if n_unstable > 0:
            print(f"  ({n_unstable} unstable conditions flagged)")
    print("=" * 78)


if __name__ == '__main__':
    main()
