"""
Diagnostic 9: Initialization Sensitivity (Stability/Conditioning)
================================================================
NO TRAINING. Sample N=20 perturbations from posterior covariance at
start of test and near biggest event. Roll out h=200 under oracle
and persistence. Plot ensemble spread.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag09")

N_PERT = 20
MAX_H = 200
SEED = 42

print("=" * 60)
print("DIAG 09: Initialization Sensitivity")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

D = U.load_final_model_and_data()
params, cl = D['params'], D['cl_params']
t_arr, x_arr, v_arr = D['t_arr'], D['x_arr'], D['v_arr']
test_idx = D['test_idx_full']
x_test, t_test = D['x_test'], D['t_test']
filt = D['filter']
N_test = D['N_test']
N_full = len(t_arr)

# Event detection to find biggest event
ev = U.detect_events_from_x(x_test, t_test)
event_indices = ev['event_indices']

# Find biggest event
jump_sizes = []
for ei in event_indices:
    if ei > 0:
        jump_sizes.append(abs(x_test[ei] - x_test[ei - 1]))
    else:
        jump_sizes.append(0.0)

if jump_sizes:
    biggest_ev_local = event_indices[np.argmax(jump_sizes)]
else:
    biggest_ev_local = N_test // 2

# Two launch points
launch_points = {
    'start': 0,           # start of test
    'event': biggest_ev_local,  # near biggest event
}

rng = np.random.RandomState(SEED)
results = {}

for label, j_test in launch_points.items():
    i_full = test_idx[j_test]
    sx0 = filt['states_x'][i_full]
    su0 = filt['states_u'][i_full]
    P0 = np.array(filt['P_post'][i_full])

    print(f"\n  Launch: {label} (t={t_arr[i_full]:.1f}s, x={sx0:.4f}, u={su0:.4f})")
    print(f"    P_post = [[{P0[0,0]:.2e}, {P0[0,1]:.2e}], [{P0[1,0]:.2e}, {P0[1,1]:.2e}]]")

    # Sample perturbations from N(0, P0)
    try:
        L = np.linalg.cholesky(P0)
    except np.linalg.LinAlgError:
        # Fallback: use eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(P0)
        eigvals = np.maximum(eigvals, 1e-12)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    perturbations = rng.randn(N_PERT, 2) @ L.T

    # Roll out nominal + perturbed for oracle and persistence
    h_actual = min(MAX_H, N_full - i_full - 1)
    launch_results = {}

    for mode in ['oracle', 'persistence']:
        # Nominal trajectory
        r_nom = U.rollout_open_loop(sx0, su0, i_full, h_actual,
                                     t_arr, v_arr, params, cl, mode=mode)
        nom_path = r_nom['path_x']

        # Perturbed trajectories
        pert_paths = np.full((N_PERT, h_actual), np.nan)
        for p in range(N_PERT):
            sx_p = sx0 + perturbations[p, 0]
            su_p = su0 + perturbations[p, 1]
            r_p = U.rollout_open_loop(sx_p, su_p, i_full, h_actual,
                                       t_arr, v_arr, params, cl, mode=mode)
            pert_paths[p] = r_p['path_x']

        # Compute spread statistics
        spread = np.nanstd(pert_paths, axis=0)
        max_spread = float(np.nanmax(spread))
        mean_spread_last50 = float(np.nanmean(spread[-50:])) if h_actual >= 50 else np.nan

        # Divergence rate: log(spread[h]) vs h -- slope
        valid_h = ~np.isnan(spread) & (spread > 1e-15)
        if valid_h.sum() > 10:
            log_spread = np.log(spread[valid_h])
            h_valid = np.arange(h_actual)[valid_h]
            # Linear fit
            slope = np.polyfit(h_valid, log_spread, 1)[0]
            lyapunov_approx = float(slope / U.DT)  # per second
        else:
            lyapunov_approx = np.nan

        launch_results[mode] = {
            'nom_path': nom_path,
            'pert_paths': pert_paths,
            'spread': spread,
            'max_spread': max_spread,
            'mean_spread_last50': mean_spread_last50,
            'lyapunov_approx': lyapunov_approx,
        }
        print(f"    {mode}: max_spread={max_spread:.6f}, "
              f"Lyapunov~={lyapunov_approx:.4f}/s")

    results[label] = launch_results

    # Figure for this launch point
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, mode in zip(axes, ['oracle', 'persistence']):
        lr = launch_results[mode]
        h_range = np.arange(h_actual) * U.DT
        t_abs = t_arr[i_full] + h_range

        # True trajectory
        end_idx = min(i_full + h_actual, N_full)
        t_true = t_arr[i_full:end_idx]
        x_true = x_arr[i_full:end_idx]
        ax.plot(t_true, x_true, 'k-', lw=1.2, label='True')

        # Nominal
        valid_n = ~np.isnan(lr['nom_path'])
        ax.plot(t_abs[valid_n], lr['nom_path'][valid_n], '--',
                color=U.MODE_COLORS[mode], lw=1.0, label='Nominal')

        # Perturbed ensemble
        for p in range(N_PERT):
            valid_p = ~np.isnan(lr['pert_paths'][p])
            ax.plot(t_abs[valid_p], lr['pert_paths'][p][valid_p],
                    '-', color=U.MODE_COLORS[mode], alpha=0.15, lw=0.5)

        # Ensemble spread band
        nom_valid = lr['nom_path'][:h_actual]
        sp = lr['spread']
        band_valid = ~np.isnan(nom_valid) & ~np.isnan(sp)
        ax.fill_between(t_abs[band_valid],
                        nom_valid[band_valid] - 2*sp[band_valid],
                        nom_valid[band_valid] + 2*sp[band_valid],
                        color=U.MODE_COLORS[mode], alpha=0.1)

        ax.set_ylabel('Displacement (m)')
        ax.set_title(f'{U.MODE_LABELS[mode]} -- {label} launch '
                     f'(Lyapunov ~ {lr["lyapunov_approx"]:.3f}/s)')
        ax.legend(fontsize=8, loc='upper left')

    axes[1].set_xlabel('Time (s)')
    fig.suptitle(f'Initialization Sensitivity: {N_PERT} perturbations, '
                 f'launch={label} (t={t_arr[i_full]:.1f}s)', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / f"sensitivity_ensemble_{label}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: sensitivity_ensemble_{label}.png")

# Summary
summary = {}
for label in launch_points:
    summary[label] = {}
    for mode in ['oracle', 'persistence']:
        lr = results[label][mode]
        summary[label][mode] = {
            'max_spread': lr['max_spread'],
            'mean_spread_last50': lr['mean_spread_last50'],
            'lyapunov_approx': lr['lyapunov_approx'],
            'divergent': lr['max_spread'] > 0.1,
        }

summary['n_perturbations'] = N_PERT
summary['max_horizon'] = MAX_H
summary['runtime_s'] = time.time() - t0

with open(OUT / "summary_diag09.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag09 complete ({time.time()-t0:.0f}s). Output: {OUT}")
