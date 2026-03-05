"""
Diagnostic 6: PIT / Rank Histogram (Probabilistic Calibration)
==============================================================
NO TRAINING. For h in {10, 50}, compute PIT = Phi((x_true - mu)/sigma)
and plot histograms. Split event vs non-event.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy import stats as sp_stats
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag06")
EVAL_HORIZONS = [10, 50]

print("=" * 60)
print("DIAG 06: PIT Histogram (Probabilistic Calibration)")
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

# Covariance trajectory
P_steady = filt['P_post'][test_idx[0]]
max_h = max(EVAL_HORIZONS)
Pxx_traj = U.propagate_cov_trajectory(P_steady, max_h, params, cl)
sigma_traj = np.sqrt(Pxx_traj)

# Event detection
ev = U.detect_events_from_x(x_test, t_test)
event_times = ev['event_times'] or []

summary = {'horizons': {}}

for H in EVAL_HORIZONS:
    print(f"\n  Computing PIT for h={H} ({H*U.DT:.1f}s)...")
    sig = sigma_traj[H - 1]

    pit_all = []
    pit_event = []
    pit_nonevent = []

    for j in range(N_test):
        i_full = test_idx[j]
        if i_full + H >= N_full:
            continue

        # Oracle rollout for mean prediction
        r = U.rollout_open_loop(
            filt['states_x'][i_full], filt['states_u'][i_full],
            i_full, H, t_arr, v_arr, params, cl, mode='oracle')
        mu_h = r['path_x'][H - 1]
        if np.isnan(mu_h):
            continue

        x_true_h = x_arr[i_full + H]
        pit = float(sp_stats.norm.cdf((x_true_h - mu_h) / sig))
        pit_all.append(pit)

        # Event classification
        t_j = t_test[j]
        is_event = any(t_j < et <= t_j + H * U.DT for et in event_times)
        if is_event:
            pit_event.append(pit)
        else:
            pit_nonevent.append(pit)

    pit_all = np.array(pit_all)
    pit_event = np.array(pit_event)
    pit_nonevent = np.array(pit_nonevent)

    # KS test vs uniform
    ks_stat, ks_p = sp_stats.kstest(pit_all, 'uniform')
    print(f"    N={len(pit_all)}, KS stat={ks_stat:.4f}, p={ks_p:.4f}")

    ks_ev = sp_stats.kstest(pit_event, 'uniform') if len(pit_event) > 10 else (np.nan, np.nan)
    ks_ne = sp_stats.kstest(pit_nonevent, 'uniform') if len(pit_nonevent) > 10 else (np.nan, np.nan)

    # Figure: 3-panel PIT histogram
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, data, title, ks in [
        (axes[0], pit_all, f'All (n={len(pit_all)})', (ks_stat, ks_p)),
        (axes[1], pit_event, f'Event (n={len(pit_event)})', ks_ev),
        (axes[2], pit_nonevent, f'Non-event (n={len(pit_nonevent)})', ks_ne),
    ]:
        if len(data) > 5:
            ax.hist(data, bins=20, density=True, color='steelblue',
                    edgecolor='black', lw=0.5, alpha=0.7)
            ax.axhline(1.0, color='red', ls='--', lw=1.0, label='Uniform')
            # CI bands for uniform
            n = len(data)
            se = np.sqrt(1.0 / (20 * n)) if n > 0 else 0
            ax.axhline(1.0 + 1.96 * se * 20, color='red', ls=':', lw=0.5, alpha=0.5)
            ax.axhline(1.0 - 1.96 * se * 20, color='red', ls=':', lw=0.5, alpha=0.5)
            ks_s = ks[0] if not isinstance(ks[0], float) or not np.isnan(ks[0]) else ks.statistic
            ks_p_val = ks[1] if not isinstance(ks[1], float) or not np.isnan(ks[1]) else ks.pvalue
            ax.set_title(f'{title}\nKS={float(ks_s):.3f}, p={float(ks_p_val):.3f}',
                         fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Too few\nsamples', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
        ax.set_xlabel('PIT')
        ax.set_ylabel('Density')

    fig.suptitle(f'PIT Histogram: h={H} ({H*U.DT:.0f}s), sigma={sig:.5f}m',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / f"pit_hist_h{H}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: pit_hist_h{H}.png")

    summary['horizons'][str(H)] = {
        'n_all': len(pit_all),
        'n_event': len(pit_event),
        'n_nonevent': len(pit_nonevent),
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
        'sigma': float(sig),
        'pit_mean': float(np.mean(pit_all)),
        'pit_std': float(np.std(pit_all)),
    }

summary['runtime_s'] = time.time() - t0
summary['note'] = ('PIT uses Gaussian predictive distribution with mean from '
                    'oracle open-loop rollout and sigma from linearized covariance '
                    'propagation (steady-state P).')

with open(OUT / "summary_diag06.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag06 complete ({time.time()-t0:.0f}s). Output: {OUT}")
