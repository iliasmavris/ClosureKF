"""
Step 6: Failure modes / regime map (non-destructive, eval-only).

Bins test-set forecast origins by |u| (water velocity; manuscript notation)
into 5 quantile bins, computes per-bin Skill(dx), R2(dx), MAE(dx) at
physical horizons {0.1, 0.2, 0.5, 1.0, 2.0}s for physics-only and
d2-only closure KF, across 3 seeds.

Uses frozen Step 4 checkpoints -- no retraining.
Copies kf_filter_2state and compute_dxr2_multihorizon verbatim from
run_transfer_step5a.py for codepath consistency.

Usage:
  python -u ems_v1/eval/failure_map_step6/run_failure_map_step6.py

Output: ems_v1/eval/failure_map_step6/
        ems_v1/figures/fig_failure_map.pdf
        ems_v1/tables/table_failure_bins.tex
"""

import os, sys, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from ems_v1.eval.metrics_pack import compute_deltax_metrics

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
TAU_PHYS = [0.1, 0.2, 0.5, 1.0, 2.0]   # physical-time horizons
H_STEPS  = [1, 2, 5, 10, 20]            # step counts at dt=0.1
MAX_H    = 20                            # covers 2.0s at 10Hz
WARMUP_SEC = 50.0
SEEDS = [1, 2, 3]
N_BINS = 5

CKPT_DIR = ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_10hz_3seed"
DATA_DIR = ROOT / "processed_data_10hz_clean_v1"
OUT_DIR  = ROOT / "ems_v1" / "eval" / "failure_map_step6"
FIG_DIR  = ROOT / "ems_v1" / "figures"
TAB_DIR  = ROOT / "ems_v1" / "tables"

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})


# ==============================================================================
#  NUMPY 2-STATE FILTER (verbatim from run_transfer_step5a.py lines 78-142)
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

    return {
        'innovations': innovations, 'S_values': S_values,
        'states_x': states_x, 'states_u': states_u,
        'cl_dt': cl_dt_arr, 'physics': phys_arr,
    }


# ==============================================================================
#  DxR2 MULTI-HORIZON (verbatim from run_transfer_step5a.py lines 149-204)
# ==============================================================================

def compute_dxr2_multihorizon(params, cl_params, states_x, states_u,
                               t, x_obs, v, max_h, eval_start, indices=None):
    """Returns per-horizon R2_dx, skill_dx, MAE_dx, RMSE_dx arrays."""
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
    skill_arr = np.full(max_h, np.nan)
    mae_arr = np.full(max_h, np.nan)
    rmse_arr = np.full(max_h, np.nan)
    n_arr = np.zeros(max_h, dtype=int)

    for h in range(max_h):
        if len(dx_pred[h]) < 10:
            continue
        m = compute_deltax_metrics(dx_true[h], dx_pred[h])
        r2_arr[h] = m['r2_dx']
        skill_arr[h] = m['skill_dx']
        mae_arr[h] = m['mae_dx']
        rmse_arr[h] = m['rmse_dx']
        n_arr[h] = m['n']

    return r2_arr, skill_arr, mae_arr, rmse_arr, n_arr


# ==============================================================================
#  HEADLINE EXPORT
# ==============================================================================

def export_headlines(df_summary, out_dir, tab_dir):
    """Compute headline stats from summary CSV and write HEADLINE.txt
    + table_failure_headlines.tex."""
    print(f"\n--- Headline stats ---")

    # max / min DSkill cells
    idx_max = df_summary['delta_skill_mean'].idxmax()
    mx = df_summary.loc[idx_max]
    idx_min = df_summary['delta_skill_mean'].idxmin()
    mn = df_summary.loc[idx_min]

    # max (worst) and min (best) delta_mae_mean
    max_dmae = df_summary['delta_mae_mean'].max()
    min_dmae = df_summary['delta_mae_mean'].min()
    idx_dmae_best = df_summary['delta_mae_mean'].idxmin()
    best_dmae = df_summary.loc[idx_dmae_best]

    mae_uniform = bool(max_dmae <= 0)

    headlines = {
        'max_dskill_bin': int(mx['bin_id']),
        'max_dskill_bin_lo': mx['bin_lo'],
        'max_dskill_bin_hi': mx['bin_hi'],
        'max_dskill_tau': mx['tau_s'],
        'max_dskill_mean': mx['delta_skill_mean'],
        'max_dskill_std': mx['delta_skill_std'],
        'min_dskill_bin': int(mn['bin_id']),
        'min_dskill_bin_lo': mn['bin_lo'],
        'min_dskill_bin_hi': mn['bin_hi'],
        'min_dskill_tau': mn['tau_s'],
        'min_dskill_mean': mn['delta_skill_mean'],
        'min_dskill_std': mn['delta_skill_std'],
        'max_dmae_mean': max_dmae,
        'min_dmae_mean': min_dmae,
        'best_dmae_bin_lo': best_dmae['bin_lo'],
        'best_dmae_bin_hi': best_dmae['bin_hi'],
        'best_dmae_tau': best_dmae['tau_s'],
        'best_dmae_mean': best_dmae['delta_mae_mean'],
        'best_dmae_std': best_dmae['delta_mae_std'],
        'mae_uniform_improvement': mae_uniform,
    }

    # --- HEADLINE.txt ---
    txt = (
        f"Step 6 Headline Stats\n"
        f"=====================\n\n"
        f"Max DSkill cell:\n"
        f"  bin {headlines['max_dskill_bin']} "
        f"[{headlines['max_dskill_bin_lo']:.3f}, "
        f"{headlines['max_dskill_bin_hi']:.3f}) m/s\n"
        f"  tau = {headlines['max_dskill_tau']:.1f} s\n"
        f"  mean = {headlines['max_dskill_mean']:+.4f}\n"
        f"  std  = {headlines['max_dskill_std']:.4f}\n\n"
        f"Min DSkill cell:\n"
        f"  bin {headlines['min_dskill_bin']} "
        f"[{headlines['min_dskill_bin_lo']:.3f}, "
        f"{headlines['min_dskill_bin_hi']:.3f}) m/s\n"
        f"  tau = {headlines['min_dskill_tau']:.1f} s\n"
        f"  mean = {headlines['min_dskill_mean']:+.4f}\n"
        f"  std  = {headlines['min_dskill_std']:.4f}\n\n"
        f"Max delta_mae_mean (worst): {headlines['max_dmae_mean']:+.6f}\n"
        f"Min delta_mae_mean (best):  {headlines['min_dmae_mean']:+.6f}\n"
        f"  best in bin [{headlines['best_dmae_bin_lo']:.3f}, "
        f"{headlines['best_dmae_bin_hi']:.3f}) at tau={headlines['best_dmae_tau']:.1f}s\n"
        f"  mean = {headlines['best_dmae_mean']:+.6f} +/- "
        f"{headlines['best_dmae_std']:.6f}\n\n"
        f"mae_uniform_improvement: {headlines['mae_uniform_improvement']}\n"
    )
    with open(out_dir / "HEADLINE.txt", 'w') as f:
        f.write(txt)
    print(f"  Saved HEADLINE.txt")

    # --- table_failure_headlines.tex (LaTeX macro definitions) ---
    def fmt_bin(v):
        return f"{v:.2f}"

    tex = (
        "% Auto-generated by run_failure_map_step6.py -- do not hand-edit\n"
        "% Step 6: Regime-conditioned failure map headlines\n"
        "\n"
        f"\\newcommand{{\\maxDskill}}{{{headlines['max_dskill_mean']:+.3f}}}\n"
        f"\\newcommand{{\\maxDskillStd}}{{{headlines['max_dskill_std']:.3f}}}\n"
        f"\\newcommand{{\\maxDskillBinLo}}{{{fmt_bin(headlines['max_dskill_bin_lo'])}}}\n"
        f"\\newcommand{{\\maxDskillBinHi}}{{{fmt_bin(headlines['max_dskill_bin_hi'])}}}\n"
        f"\\newcommand{{\\maxDskillTau}}{{{headlines['max_dskill_tau']:.1f}}}\n"
        f"\\newcommand{{\\minDskill}}{{{headlines['min_dskill_mean']:+.3f}}}\n"
        f"\\newcommand{{\\minDskillStd}}{{{headlines['min_dskill_std']:.3f}}}\n"
        f"\\newcommand{{\\minDskillBinLo}}{{{fmt_bin(headlines['min_dskill_bin_lo'])}}}\n"
        f"\\newcommand{{\\minDskillBinHi}}{{{fmt_bin(headlines['min_dskill_bin_hi'])}}}\n"
        f"\\newcommand{{\\minDskillTau}}{{{headlines['min_dskill_tau']:.1f}}}\n"
        f"\\newcommand{{\\bestDmaeBinLo}}{{{fmt_bin(headlines['best_dmae_bin_lo'])}}}\n"
        f"\\newcommand{{\\bestDmaeBinHi}}{{{fmt_bin(headlines['best_dmae_bin_hi'])}}}\n"
        f"\\newcommand{{\\bestDmaeTau}}{{{headlines['best_dmae_tau']:.1f}}}\n"
        f"\\newcommand{{\\bestDmae}}{{{headlines['best_dmae_mean']:+.4f}}}\n"
        f"\\newcommand{{\\bestDmaeStd}}{{{headlines['best_dmae_std']:.4f}}}\n"
        f"\\newcommand{{\\maxDmae}}{{{headlines['max_dmae_mean']:+.6f}}}\n"
    )
    with open(tab_dir / "table_failure_headlines.tex", 'w') as f:
        f.write(tex)
    print(f"  Saved table_failure_headlines.tex")

    # Console summary
    print(f"  Max DSkill: bin {headlines['max_dskill_bin']}, "
          f"tau={headlines['max_dskill_tau']}s, "
          f"{headlines['max_dskill_mean']:+.4f} +/- "
          f"{headlines['max_dskill_std']:.4f}")
    print(f"  Min DSkill: bin {headlines['min_dskill_bin']}, "
          f"tau={headlines['min_dskill_tau']}s, "
          f"{headlines['min_dskill_mean']:+.4f} +/- "
          f"{headlines['min_dskill_std']:.4f}")
    print(f"  mae_uniform_improvement: {headlines['mae_uniform_improvement']}")

    return headlines


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    t0_global = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 6: FAILURE MODES / REGIME MAP")
    print("=" * 70)

    # ------------------------------------------------------------------
    #  Load data (warm-start, matching Step 5A exactly)
    # ------------------------------------------------------------------
    print("\n--- Load data ---")
    df_val  = pd.read_csv(DATA_DIR / "val_10hz_ready.csv")
    df_test = pd.read_csv(DATA_DIR / "test_10hz_ready.csv")

    test_start_time = df_test['timestamp'].iloc[0]
    warmup_start = df_val['timestamp'].iloc[-1] - WARMUP_SEC
    df_warmup = df_val[df_val['timestamp'] >= warmup_start].copy()
    df_warm = pd.concat([df_warmup, df_test], ignore_index=True)

    t_arr = df_warm['timestamp'].values.astype(np.float64)
    x_arr = df_warm['displacement'].values.astype(np.float64)
    v_arr = df_warm['velocity'].values.astype(np.float64)

    score_mask = t_arr >= test_start_time
    eval_start = int(np.argmax(score_mask))
    N = len(t_arr)
    n_scored = N - eval_start

    print(f"  Total pts: {N}, warmup: {eval_start}, scored: {n_scored}")

    # ------------------------------------------------------------------
    #  Bin construction (on scored portion only)
    # ------------------------------------------------------------------
    print("\n--- Bin construction ---")
    v_scored = np.abs(v_arr[eval_start:])
    bin_edges = np.percentile(v_scored, [0, 20, 40, 60, 80, 100])
    bin_edges[0] = 0.0                      # clean lower bound
    bin_edges[-1] = v_scored.max() + 1e-10  # ensure all included

    # Assign bins: digitize into 0..4
    bin_labels_arr = np.digitize(v_scored, bin_edges[1:-1])  # 0..4

    # Convert to absolute indices
    bin_indices = {}
    for b in range(N_BINS):
        relative = np.where(bin_labels_arr == b)[0]
        bin_indices[b] = relative + eval_start  # absolute indices
        print(f"  Bin {b}: |u| in [{bin_edges[b]:.4f}, {bin_edges[b+1]:.4f}) "
              f"n={len(relative)}")

    # Save bin edges
    bin_edges_df = pd.DataFrame({
        'bin_id': range(N_BINS),
        'lo': bin_edges[:-1],
        'hi': bin_edges[1:],
        'n_scored': [len(bin_indices[b]) for b in range(N_BINS)],
    })
    bin_edges_df.to_csv(OUT_DIR / "bin_edges.csv", index=False)

    # ------------------------------------------------------------------
    #  Per-seed loop
    # ------------------------------------------------------------------
    cell_rows = []

    for seed in SEEDS:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")

        # Load checkpoints
        clos_path = (CKPT_DIR / f"seed{seed}" / "checkpoints"
                     / f"closure_d2only_seed{seed}.pth")
        phys_path = (CKPT_DIR / f"seed{seed}" / "checkpoints"
                     / f"stage1_physics_seed{seed}.pth")
        assert clos_path.exists(), f"Missing: {clos_path}"
        assert phys_path.exists(), f"Missing: {phys_path}"

        clos_ckpt = torch.load(str(clos_path), map_location='cpu',
                               weights_only=False)
        phys_ckpt = torch.load(str(phys_path), map_location='cpu',
                               weights_only=False)

        s2_params = clos_ckpt['params']
        cl_sum = clos_ckpt['closure']
        s1_params = phys_ckpt['params']

        # Build param dicts
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

        phys_pp = {
            'alpha': s1_params['alpha'], 'c': s1_params['c'],
            'vc': s1_params['vc'], 'kappa': s1_params['kappa'],
            'qx': s1_params['qx'], 'qu': s1_params['qu'],
            'R': s1_params['R'],
            'P0_xx': s1_params['P0_xx'], 'P0_uu': s1_params['P0_uu'],
        }
        zero_cl = {k: 0.0 for k in ['a1', 'b1', 'b2', 'd1', 'd2', 'd3']}
        zero_cl['q_scale'] = 1.0

        print(f"  Physics: alpha={phys_pp['alpha']:.4f}, "
              f"kappa={phys_pp['kappa']:.4f}")
        print(f"  Closure: d2={clos_cl['d2']:.4f}, "
              f"q_scale={clos_cl['q_scale']:.4f}")

        # Run filter (2 calls)
        print("  Running KF filter (physics)...", end='', flush=True)
        filt_phys = kf_filter_2state(phys_pp, zero_cl, t_arr, x_arr, v_arr)
        print(" done")

        print("  Running KF filter (closure)...", end='', flush=True)
        filt_clos = kf_filter_2state(clos_pp, clos_cl, t_arr, x_arr, v_arr)
        print(" done")

        # Per-bin metrics (5 bins x 2 models)
        print("  Per-bin metrics:")
        for b in range(N_BINS):
            indices = bin_indices[b].tolist()

            # Physics
            r2_p, sk_p, mae_p, rmse_p, n_p = compute_dxr2_multihorizon(
                phys_pp, zero_cl,
                filt_phys['states_x'], filt_phys['states_u'],
                t_arr, x_arr, v_arr, MAX_H, eval_start,
                indices=indices)

            # Closure
            r2_c, sk_c, mae_c, rmse_c, n_c = compute_dxr2_multihorizon(
                clos_pp, clos_cl,
                filt_clos['states_x'], filt_clos['states_u'],
                t_arr, x_arr, v_arr, MAX_H, eval_start,
                indices=indices)

            # Extract at headline horizons
            for hi, hs in zip(H_STEPS, TAU_PHYS):
                idx = hi - 1  # 0-indexed
                delta_skill = sk_c[idx] - sk_p[idx]
                delta_mae = mae_c[idx] - mae_p[idx]

                # n_eff: number of indices where i+h < N
                n_eff = int(np.sum(np.array(indices) + hi < N))

                cell_rows.append({
                    'seed': seed,
                    'bin_id': b,
                    'bin_lo': float(bin_edges[b]),
                    'bin_hi': float(bin_edges[b+1]),
                    'tau_s': hs,
                    'h_steps': hi,
                    'r2_phys': float(r2_p[idx]),
                    'r2_clos': float(r2_c[idx]),
                    'skill_phys': float(sk_p[idx]),
                    'skill_clos': float(sk_c[idx]),
                    'delta_skill': float(delta_skill),
                    'mae_phys': float(mae_p[idx]),
                    'mae_clos': float(mae_c[idx]),
                    'delta_mae': float(delta_mae),
                    'n_eff': n_eff,
                    'n_phys': int(n_p[idx]),
                    'n_clos': int(n_c[idx]),
                })

            # Print 1s headline for this bin
            row_1s = [r for r in cell_rows
                      if r['seed'] == seed and r['bin_id'] == b
                      and r['tau_s'] == 1.0]
            if row_1s:
                r = row_1s[-1]
                print(f"    Bin {b} [{bin_edges[b]:.3f}, {bin_edges[b+1]:.3f}): "
                      f"dSkill@1s={r['delta_skill']:+.4f} "
                      f"dMAE@1s={r['delta_mae']:+.6f} "
                      f"n_eff={r['n_eff']}")

        print(f"  Seed {seed} done ({time.time()-t0_global:.0f}s elapsed)")

    # ------------------------------------------------------------------
    #  Save cells CSV (75 rows = 3 seeds x 5 bins x 5 horizons)
    # ------------------------------------------------------------------
    df_cells = pd.DataFrame(cell_rows)
    df_cells.to_csv(OUT_DIR / "failure_map_cells.csv", index=False)
    print(f"\nSaved failure_map_cells.csv ({len(df_cells)} rows)")

    # ------------------------------------------------------------------
    #  Aggregation across seeds -> summary (25 rows = 5 bins x 5 horizons)
    # ------------------------------------------------------------------
    summary_rows = []
    for b in range(N_BINS):
        for hs in TAU_PHYS:
            sub = df_cells[(df_cells['bin_id'] == b) &
                           (df_cells['tau_s'] == hs)]
            if len(sub) == 0:
                continue
            summary_rows.append({
                'bin_id': b,
                'tau_s': hs,
                'bin_lo': sub['bin_lo'].iloc[0],
                'bin_hi': sub['bin_hi'].iloc[0],
                'delta_skill_mean': sub['delta_skill'].mean(),
                'delta_skill_std': (sub['delta_skill'].std(ddof=1)
                                    if len(sub) > 1 else 0.0),
                'delta_mae_mean': sub['delta_mae'].mean(),
                'delta_mae_std': (sub['delta_mae'].std(ddof=1)
                                  if len(sub) > 1 else 0.0),
                'mean_r2_phys': sub['r2_phys'].mean(),
                'mean_r2_clos': sub['r2_clos'].mean(),
                'mean_skill_phys': sub['skill_phys'].mean(),
                'mean_skill_clos': sub['skill_clos'].mean(),
                'mean_mae_phys': sub['mae_phys'].mean(),
                'mean_mae_clos': sub['mae_clos'].mean(),
                'mean_n_eff': sub['n_eff'].mean(),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "failure_map_summary.csv", index=False)
    print(f"Saved failure_map_summary.csv ({len(df_summary)} rows)")

    # ------------------------------------------------------------------
    #  Headline stats (from summary CSV)
    # ------------------------------------------------------------------
    headlines = export_headlines(df_summary, OUT_DIR, TAB_DIR)

    # ------------------------------------------------------------------
    #  n_eff hard check: min >= 200 across all (bin, tau) cells
    # ------------------------------------------------------------------
    min_neff = df_cells['n_eff'].min()
    neff_pass = min_neff >= 200
    print(f"\nmin(n_eff) = {min_neff} >= 200: "
          f"{'PASS' if neff_pass else 'WARNING'}")
    neff_warning = "" if neff_pass else (
        f"\n## WARNING: min(n_eff) = {min_neff} < 200\n"
        f"Some (bin, tau) cells have fewer than 200 effective samples.\n"
    )

    # ------------------------------------------------------------------
    #  Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Aggregate (3-seed mean):")
    print(f"{'Bin':>5} {'tau':>5} {'dSkill':>9} {'dMAE':>11} {'n_eff':>6}")
    print(f"{'---':>5} {'---':>5} {'--------':>9} {'----------':>11} {'-----':>6}")
    for _, row in df_summary.iterrows():
        print(f"  {int(row['bin_id']):>3}  {row['tau_s']:>4.1f}  "
              f"{row['delta_skill_mean']:>+8.4f}  "
              f"{row['delta_mae_mean']:>+10.6f}  "
              f"{row['mean_n_eff']:>5.0f}")

    # ------------------------------------------------------------------
    #  NaN check
    # ------------------------------------------------------------------
    nan_count = df_summary['delta_skill_mean'].isna().sum()
    print(f"\nNaN in delta_skill_mean: {nan_count} "
          f"({'PASS' if nan_count == 0 else 'FAIL'})")

    # ------------------------------------------------------------------
    #  Figure: 2-panel heatmap
    # ------------------------------------------------------------------
    print(f"\n--- Figure ---")

    piv_dskill = df_summary.pivot(index='bin_id', columns='tau_s',
                                   values='delta_skill_mean')
    piv_dmae = df_summary.pivot(index='bin_id', columns='tau_s',
                                 values='delta_mae_mean')

    # Y-axis labels: bin edges with |u| notation
    ylabels = []
    for b in range(N_BINS):
        ylabels.append(f"[{bin_edges[b]:.2f}, {bin_edges[b+1]:.2f})")
    xlabels = [f"{t:.1f}" for t in TAU_PHYS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel A: Delta Skill (blue = closure better) ---
    ax = axes[0]
    vals_a = piv_dskill.values
    vmax_a = max(np.nanmax(np.abs(vals_a)), 0.01)
    norm_a = TwoSlopeNorm(vmin=-vmax_a, vcenter=0, vmax=vmax_a)
    im1 = ax.imshow(vals_a, aspect='auto', cmap='RdBu', norm=norm_a)
    ax.set_xticks(range(len(TAU_PHYS)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(N_BINS))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel('Forecast horizon $\\tau$ (s)')
    ax.set_ylabel('Flow speed quintile $|u|$ (m/s)')
    ax.set_title('(A) $\\Delta$Skill (closure $-$ physics)')
    for i in range(vals_a.shape[0]):
        for j in range(vals_a.shape[1]):
            val = vals_a[i, j]
            if np.isnan(val):
                continue
            color = 'white' if abs(val) > 0.5 * vmax_a else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                    fontsize=7, color=color)
    cb1 = plt.colorbar(im1, ax=ax, shrink=0.85)
    cb1.set_label('$\\Delta$Skill', fontsize=9)

    # --- Panel B: -delta_MAE (blue = closure better = lower MAE) ---
    ax = axes[1]
    vals_b = -piv_dmae.values  # negate so blue = closure better
    vmax_b = max(np.nanmax(np.abs(vals_b)), 0.001)
    norm_b = TwoSlopeNorm(vmin=-vmax_b, vcenter=0, vmax=vmax_b)
    im2 = ax.imshow(vals_b, aspect='auto', cmap='RdBu', norm=norm_b)
    ax.set_xticks(range(len(TAU_PHYS)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(N_BINS))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel('Forecast horizon $\\tau$ (s)')
    ax.set_ylabel('Flow speed quintile $|u|$ (m/s)')
    ax.set_title('(B) $-\\Delta$MAE (closure $-$ physics)')
    # Annotate with RAW signed delta_mae values
    raw_dmae = piv_dmae.values
    for i in range(vals_b.shape[0]):
        for j in range(vals_b.shape[1]):
            val = raw_dmae[i, j]
            if np.isnan(val):
                continue
            color = 'white' if abs(vals_b[i, j]) > 0.5 * vmax_b else 'black'
            ax.text(j, i, f'{val:+.4f}', ha='center', va='center',
                    fontsize=7, color=color)
    cb2 = plt.colorbar(im2, ax=ax, shrink=0.85)
    cb2.set_label('$-\\Delta$MAE (m)', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_failure_map.pdf", bbox_inches='tight')
    fig.savefig(OUT_DIR / "fig_failure_map.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig_failure_map.pdf -> {FIG_DIR}")
    print(f"  Saved fig_failure_map.png -> {OUT_DIR}")

    # ------------------------------------------------------------------
    #  LaTeX table: table_failure_bins.tex
    # ------------------------------------------------------------------
    print(f"\n--- LaTeX table ---")

    # n_eff at tau=2.0s per bin (use seed 1 as representative)
    neff_2s = {}
    for b in range(N_BINS):
        sub = df_cells[(df_cells['bin_id'] == b) &
                       (df_cells['tau_s'] == 2.0) &
                       (df_cells['seed'] == 1)]
        neff_2s[b] = int(sub['n_eff'].iloc[0]) if len(sub) > 0 else 0

    tex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Quantile bin definitions for the regime-conditioned",
        r"           analysis (Figure~\ref{fig:failure_map}).  $n_{\mathrm{scored}}$",
        r"           is the number of test-set origins in each bin;",
        r"           $n_{\mathrm{eff}}$ is the effective sample size at",
        r"           $\tau = 2.0$\,\si{s} (accounting for end-of-series",
        r"           truncation).}",
        r"  \label{tab:failure_bins}",
        r"  \begin{tabular}{crrc}",
        r"    \toprule",
        r"    Bin & $|u|$ range (m/s) & $n_{\mathrm{scored}}$ "
        r"& $n_{\mathrm{eff}}$ ($\tau{=}2$\,s) \\",
        r"    \midrule",
    ]
    for b in range(N_BINS):
        n_sc = len(bin_indices[b])
        lo, hi = bin_edges[b], bin_edges[b+1]
        ne = neff_2s[b]
        tex_lines.append(
            f"    {b+1} & $[{lo:.3f},\\; {hi:.3f})$ & {n_sc} & {ne} \\\\"
        )
    tex_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex_path = TAB_DIR / "table_failure_bins.tex"
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  Saved {tex_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    #  README
    # ------------------------------------------------------------------
    elapsed = time.time() - t0_global
    readme = f"""# Failure Map -- Step 6

## Protocol
- 5 quantile bins on |u| (water velocity magnitude; manuscript notation)
- Computed on scored (test) segment only ({n_scored} points)
- Warm start: last {WARMUP_SEC:.0f}s of validation prepended for filter spinup
- Uses frozen Step 4 checkpoints (d2-only, 3 seeds)
- No retraining; eval-only

## Bin edges (|u|, m/s)
{', '.join(f'{e:.4f}' for e in bin_edges)}

## Horizons
- Physical seconds: {TAU_PHYS}
- Steps at 10 Hz: {H_STEPS}
- Max horizon: {MAX_H} steps = {MAX_H * 0.1:.1f}s

## Definitions
- delta_skill = skill_dx(closure) - skill_dx(physics)
  - skill_dx = 1 - RMSE/RMSE_base (constant-mean-increment baseline)
  - Positive = closure better
- delta_mae = MAE(closure) - MAE(physics)
  - Negative = closure better
- n_eff = number of bin indices where i + h < N
  - min(n_eff) = {min_neff} {'(>= 200, PASS)' if neff_pass else '(< 200, WARNING)'}
{neff_warning}
## Caveat
In low-variance regimes (small |u|), Var(dx_true) is small, making R2/Skill
ill-conditioned. MAE is shown as a variance-robust complement.

## Files
- failure_map_cells.csv: {len(df_cells)} rows (seed x bin x tau)
- failure_map_summary.csv: {len(df_summary)} rows (bin x tau, 3-seed mean/std)
- bin_edges.csv: bin definitions and per-bin sample counts
- fig_failure_map.png: local copy of 2-panel heatmap
- README.md: this file

## External outputs
- ems_v1/figures/fig_failure_map.pdf: manuscript figure
- ems_v1/tables/table_failure_bins.tex: LaTeX table

## Runtime
{elapsed:.0f}s ({elapsed/60:.1f} min)
"""
    with open(OUT_DIR / "README.md", 'w') as f:
        f.write(readme)
    print(f"  Saved README.md")

    # ------------------------------------------------------------------
    #  Verification checklist
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("VERIFICATION CHECKLIST")
    print(f"{'='*70}")

    checks = [
        (f"failure_map_cells.csv has 75 rows",
         len(df_cells) == 75),
        (f"failure_map_summary.csv has 25 rows",
         len(df_summary) == 25),
        (f"Tau list matches Freeze #1",
         TAU_PHYS == [0.1, 0.2, 0.5, 1.0, 2.0]),
        (f"Bin edges monotonically increasing",
         all(bin_edges[i] < bin_edges[i+1] for i in range(len(bin_edges)-1))),
        (f"All bins have >100 samples",
         all(len(bin_indices[b]) > 100 for b in range(N_BINS))),
        (f"No NaN in delta_skill_mean",
         nan_count == 0),
        (f"fig_failure_map.pdf exists",
         (FIG_DIR / "fig_failure_map.pdf").exists()),
        (f"table_failure_bins.tex exists",
         tex_path.exists()),
        (f"min(n_eff) >= 200",
         neff_pass),
    ]

    all_pass = True
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {desc}")

    print(f"\n{'='*70}")
    print(f"Step 6 complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"All checks: {'PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
