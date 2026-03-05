"""
Profile-likelihood confidence intervals for d2 and q_scale.

Approach: For each parameter, grid the raw value around the MLE,
optimize the nuisance parameter at each grid point via Brent's method,
and record the NLL.  The 95% CI is {theta : NLL(theta) - NLL_min <= 1.92}
(chi-sq(1)/2 = 3.84/2 = 1.92).

Only 2 trainable params (d2_raw, log_q_scale), so this is fast.

Usage:
  python -u ems_v1/eval/robustness_step4/profile_ci_d2_qscale.py

Output: ems_v1/eval/robustness_step4/profile_ci_results.json
        ems_v1/eval/robustness_step4/profile_ci_d2.csv
        ems_v1/eval/robustness_step4/profile_ci_qscale.csv
        ems_v1/eval/robustness_step4/fig_profile_ci.png
"""

import os, sys, math, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize_scalar

import torch
torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_closure import KalmanForecasterClosure
from torch.utils.data import DataLoader

# ==============================================================================
#  CONFIG
# ==============================================================================
FORCE_CPU = True
DEVICE = torch.device('cpu')
DT = 0.1
VAR_FLOOR = 1e-6

# Dataset settings (must match Step 4 S2)
S2_L = 64; S2_H = 20
N_WINDOWS = 256  # Fixed deterministic subset (matches Hessian)

# Profile grid
N_GRID = 101
D2_RAW_HALF_RANGE = 2.0   # +/- around MLE in raw space
QS_RAW_HALF_RANGE = 2.0   # +/- around MLE in log space

# Chi-sq threshold
CHI2_HALF = 1.92  # chi2(1)/2 = 3.84/2, for 95% CI

# Paths
DATA_DIR = ROOT / "processed_data_10hz_clean_v1"
CKPT = (ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_10hz_3seed"
        / "seed1" / "checkpoints" / "closure_d2only_seed1.pth")
OUT_DIR = ROOT / "ems_v1" / "eval" / "robustness_step4"


def gaussian_nll(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint ---
    print("Loading checkpoint:", CKPT)
    ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt['model_state']

    # Reconstruct model from S1 params stored in the checkpoint
    # We need to figure out the init args; just create a default and load
    model = KalmanForecasterClosure(alpha_param="softplus").to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Identify the 2 trainable params
    d2_raw_mle = model.d2_raw.item()
    log_qs_mle = model.log_q_scale.item()
    d2_mle = math.log(1.0 + math.exp(d2_raw_mle))
    qs_mle = math.exp(log_qs_mle)
    print(f"MLE: d2_raw={d2_raw_mle:.6f} (d2={d2_mle:.4f}), "
          f"log_q_scale={log_qs_mle:.6f} (q_scale={qs_mle:.4f})")

    # --- Load val data ---
    val_csv = str(DATA_DIR / "val_10hz_ready.csv")
    val_ds = StateSpaceDataset(
        [val_csv], L=S2_L, m=S2_L, H=S2_H,
        predict_deltas=False, normalize=False)

    n_use = min(N_WINDOWS, len(val_ds))
    loader = DataLoader(val_ds, batch_size=n_use, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
    v_h = v_h.to(DEVICE); dt_h = dt_h.to(DEVICE)
    x_h = x_h.to(DEVICE); v_f = v_f.to(DEVICE)
    dt_f = dt_f.to(DEVICE); x_true = x_true.to(DEVICE)
    print(f"Val windows: {n_use}")

    # --- NLL evaluator ---
    def eval_nll(d2_raw_val, log_qs_val):
        with torch.no_grad():
            model.d2_raw.fill_(d2_raw_val)
            model.log_q_scale.fill_(log_qs_val)
            xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
            nll = gaussian_nll(xp, xv, x_true, VAR_FLOOR)
        return nll.item()

    nll_mle = eval_nll(d2_raw_mle, log_qs_mle)
    print(f"NLL at MLE: {nll_mle:.6f}")

    # =====================================================================
    #  Profile d2: grid d2_raw, optimize log_q_scale at each point
    # =====================================================================
    print("\n--- Profiling d2_raw ---")
    d2_grid = np.linspace(d2_raw_mle - D2_RAW_HALF_RANGE,
                          d2_raw_mle + D2_RAW_HALF_RANGE, N_GRID)
    d2_nll = np.full(N_GRID, np.nan)
    d2_qs_opt = np.full(N_GRID, np.nan)

    for i, d2r in enumerate(d2_grid):
        def obj_qs(lqs):
            return eval_nll(d2r, lqs)

        res = minimize_scalar(obj_qs,
                              bounds=(log_qs_mle - 3.0, log_qs_mle + 3.0),
                              method='bounded',
                              options={'xatol': 1e-6, 'maxiter': 200})
        d2_nll[i] = res.fun
        d2_qs_opt[i] = res.x
        if i % 20 == 0:
            print(f"  d2_grid[{i}]: d2_raw={d2r:.4f} -> NLL={res.fun:.6f}")

    # Convert d2_raw to physical d2 via softplus
    d2_phys_grid = np.log(1.0 + np.exp(d2_grid))
    d2_delta_nll = d2_nll - nll_mle

    # Find CI bounds
    in_ci = d2_delta_nll <= CHI2_HALF
    if in_ci.any():
        d2_ci_lo = d2_phys_grid[in_ci].min()
        d2_ci_hi = d2_phys_grid[in_ci].max()
    else:
        d2_ci_lo = d2_ci_hi = d2_mle
        print("  WARNING: No grid points within CI threshold!")

    print(f"  d2 profile CI (95%): [{d2_ci_lo:.4f}, {d2_ci_hi:.4f}]")

    # =====================================================================
    #  Profile q_scale: grid log_q_scale, optimize d2_raw at each point
    # =====================================================================
    print("\n--- Profiling log_q_scale ---")
    qs_grid = np.linspace(log_qs_mle - QS_RAW_HALF_RANGE,
                          log_qs_mle + QS_RAW_HALF_RANGE, N_GRID)
    qs_nll = np.full(N_GRID, np.nan)
    qs_d2_opt = np.full(N_GRID, np.nan)

    for i, lqs in enumerate(qs_grid):
        def obj_d2(d2r):
            return eval_nll(d2r, lqs)

        res = minimize_scalar(obj_d2,
                              bounds=(d2_raw_mle - 3.0, d2_raw_mle + 3.0),
                              method='bounded',
                              options={'xatol': 1e-6, 'maxiter': 200})
        qs_nll[i] = res.fun
        qs_d2_opt[i] = res.x
        if i % 20 == 0:
            print(f"  qs_grid[{i}]: log_qs={lqs:.4f} -> NLL={res.fun:.6f}")

    qs_phys_grid = np.exp(qs_grid)
    qs_delta_nll = qs_nll - nll_mle

    in_ci_qs = qs_delta_nll <= CHI2_HALF
    if in_ci_qs.any():
        qs_ci_lo = qs_phys_grid[in_ci_qs].min()
        qs_ci_hi = qs_phys_grid[in_ci_qs].max()
    else:
        qs_ci_lo = qs_ci_hi = qs_mle
        print("  WARNING: No grid points within CI threshold!")

    print(f"  q_scale profile CI (95%): [{qs_ci_lo:.4f}, {qs_ci_hi:.4f}]")

    # --- Restore MLE ---
    with torch.no_grad():
        model.d2_raw.fill_(d2_raw_mle)
        model.log_q_scale.fill_(log_qs_mle)

    # =====================================================================
    #  Save results
    # =====================================================================

    # CSV: profile curves
    import pandas as pd
    pd.DataFrame({
        'd2_raw': d2_grid, 'd2_phys': d2_phys_grid,
        'delta_nll': d2_delta_nll, 'qs_opt': d2_qs_opt,
    }).to_csv(OUT_DIR / "profile_ci_d2.csv", index=False)

    pd.DataFrame({
        'log_q_scale': qs_grid, 'q_scale_phys': qs_phys_grid,
        'delta_nll': qs_delta_nll, 'd2_opt': qs_d2_opt,
    }).to_csv(OUT_DIR / "profile_ci_qscale.csv", index=False)

    # JSON summary
    results = {
        'nll_mle': nll_mle,
        'chi2_threshold': CHI2_HALF,
        'd2': {
            'mle': d2_mle,
            'mle_raw': d2_raw_mle,
            'profile_ci95_lo': float(d2_ci_lo),
            'profile_ci95_hi': float(d2_ci_hi),
        },
        'q_scale': {
            'mle': qs_mle,
            'mle_raw': log_qs_mle,
            'profile_ci95_lo': float(qs_ci_lo),
            'profile_ci95_hi': float(qs_ci_hi),
        },
        'grid_settings': {
            'n_grid': N_GRID,
            'd2_raw_half_range': D2_RAW_HALF_RANGE,
            'qs_raw_half_range': QS_RAW_HALF_RANGE,
            'n_windows': n_use,
        },
        'checkpoint': str(CKPT),
        'seed': 1,
    }
    with open(OUT_DIR / "profile_ci_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: profile_ci_results.json")

    # =====================================================================
    #  Figure
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: d2 profile
    ax = axes[0]
    ax.plot(d2_phys_grid, d2_delta_nll, 'b-', lw=1.5)
    ax.axhline(CHI2_HALF, color='r', ls='--', lw=0.8, label=f'95% threshold ({CHI2_HALF})')
    ax.axvline(d2_mle, color='gray', ls=':', lw=0.8)
    ax.axvspan(d2_ci_lo, d2_ci_hi, alpha=0.15, color='blue')
    ax.set_xlabel(r'$d_2$ (m$^{-1}$)')
    ax.set_ylabel(r'$\Delta$ NLL')
    ax.set_title(f'Profile CI: $d_2$ = [{d2_ci_lo:.3f}, {d2_ci_hi:.3f}]')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-0.1)

    # Panel B: q_scale profile
    ax = axes[1]
    ax.plot(qs_phys_grid, qs_delta_nll, 'b-', lw=1.5)
    ax.axhline(CHI2_HALF, color='r', ls='--', lw=0.8, label=f'95% threshold ({CHI2_HALF})')
    ax.axvline(qs_mle, color='gray', ls=':', lw=0.8)
    ax.axvspan(qs_ci_lo, qs_ci_hi, alpha=0.15, color='blue')
    ax.set_xlabel(r'$q_{\mathrm{scale}}$')
    ax.set_ylabel(r'$\Delta$ NLL')
    ax.set_title(f'Profile CI: $q_{{scale}}$ = [{qs_ci_lo:.3f}, {qs_ci_hi:.3f}]')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-0.1)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_profile_ci.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_profile_ci.png")

    # Summary
    print(f"\n{'='*50}")
    print(f"Profile-likelihood 95% CIs (seed 1):")
    print(f"  d2:      [{d2_ci_lo:.4f}, {d2_ci_hi:.4f}]  (MLE={d2_mle:.4f})")
    print(f"  q_scale: [{qs_ci_lo:.4f}, {qs_ci_hi:.4f}]  (MLE={qs_mle:.4f})")
    print(f"  Seed-based CI (3 seeds): d2 = [2.049, 2.409]")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
