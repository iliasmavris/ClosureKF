"""
Diagnostic 5: Coverage vs Horizon (+/-2 sigma)
===============================================
NO TRAINING. For h=1..200, compute empirical coverage of +/-2sigma
prediction intervals. Split event vs non-event.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag05")
MAX_H = 200

print("=" * 60)
print("DIAG 05: Coverage vs Horizon (+/-2 sigma)")
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

# Covariance trajectory (steady-state P -> open-loop propagation)
P_steady = filt['P_post'][test_idx[0]]
Pxx_traj = U.propagate_cov_trajectory(P_steady, MAX_H, params, cl)
# Pxx_traj[h-1] = P_pred_xx + R at horizon h
sigma_traj = np.sqrt(Pxx_traj)

# Compute oracle rollout paths for coverage check
print(f"\nComputing {MAX_H}-step oracle rollout paths...")
paths_oracle = np.full((N_test, MAX_H), np.nan)
for j in range(N_test):
    i_full = test_idx[j]
    if i_full + 1 >= N_full:
        continue
    r = U.rollout_open_loop(
        filt['states_x'][i_full], filt['states_u'][i_full],
        i_full, MAX_H, t_arr, v_arr, params, cl, mode='oracle')
    paths_oracle[j] = r['path_x']
    if (j + 1) % 300 == 0:
        print(f"    ... {j+1}/{N_test}")

# Event detection for conditional coverage
ev = U.detect_events_from_x(x_test, t_test)
event_times = ev['event_times'] or []

# Compute coverage at each horizon
print("\nComputing coverage...")
horizons = list(range(1, MAX_H + 1))
rows = []
for h in horizons:
    sig = sigma_traj[h - 1]
    n_total = 0
    n_covered = 0
    n_event = 0
    n_event_covered = 0
    n_nonevent = 0
    n_nonevent_covered = 0

    for j in range(N_test):
        i_full = test_idx[j]
        if i_full + h >= N_full:
            continue
        if np.isnan(paths_oracle[j, h - 1]):
            continue

        mu_h = paths_oracle[j, h - 1]
        x_true_h = x_arr[i_full + h]
        covered = abs(x_true_h - mu_h) < 2 * sig

        n_total += 1
        if covered:
            n_covered += 1

        # Is this an event window?
        t_j = t_test[j]
        is_event = any(t_j < et <= t_j + h * U.DT for et in event_times)
        if is_event:
            n_event += 1
            if covered:
                n_event_covered += 1
        else:
            n_nonevent += 1
            if covered:
                n_nonevent_covered += 1

    cov_all = n_covered / n_total if n_total > 0 else np.nan
    cov_event = n_event_covered / n_event if n_event > 0 else np.nan
    cov_nonevent = n_nonevent_covered / n_nonevent if n_nonevent > 0 else np.nan

    rows.append({
        'horizon': h, 'time_s': h * U.DT,
        'sigma_2': 2 * sig, 'n_total': n_total,
        'coverage_all': cov_all,
        'coverage_event': cov_event,
        'coverage_nonevent': cov_nonevent,
        'n_event': n_event, 'n_nonevent': n_nonevent,
    })

    if h in [1, 5, 10, 20, 50, 100, 200]:
        print(f"  h={h:4d}: cov_all={cov_all:.3f} cov_ev={cov_event:.3f} "
              f"cov_ne={cov_nonevent:.3f} sigma={2*sig:.5f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "coverage_table.csv", index=False)

# Figure
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})

h_s = np.array(horizons) * U.DT

ax = axes[0]
ax.plot(h_s, df['coverage_all'], 'k-', lw=1.5, label='All')
ax.plot(h_s, df['coverage_event'], '-', color='red', lw=1.0, alpha=0.8,
        label='Event windows')
ax.plot(h_s, df['coverage_nonevent'], '-', color='blue', lw=1.0, alpha=0.8,
        label='Non-event windows')
ax.axhline(0.954, color='gray', ls='--', lw=0.8, label='Nominal 95.4% (2-sigma)')
ax.set_ylabel('Empirical Coverage')
ax.set_title('Prediction Interval Coverage (+/-2 sigma) vs Horizon')
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.05)

ax = axes[1]
ax.plot(h_s, df['sigma_2'], 'k-', lw=1.0)
ax.set_xlabel('Horizon (s)')
ax.set_ylabel('+/-2 sigma width (m)')
ax.set_title('Prediction Interval Width')

plt.tight_layout()
fig.savefig(OUT / "coverage_curve.png", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {OUT / 'coverage_curve.png'}")

# Summary: identify over/underconfidence regions
cov_all = df['coverage_all'].values
nominal = 0.954
overconfident = np.where(cov_all < nominal - 0.05)[0]
underconfident = np.where(cov_all > nominal + 0.05)[0]

summary = {
    'max_horizon': MAX_H,
    'nominal_coverage': nominal,
    'mean_coverage_h1_10': float(np.mean(cov_all[:10])),
    'mean_coverage_h50_100': float(np.mean(cov_all[49:100])),
    'mean_coverage_h100_200': float(np.mean(cov_all[99:200])),
    'n_overconfident_horizons': int(len(overconfident)),
    'n_underconfident_horizons': int(len(underconfident)),
    'first_overconfident_h': int(overconfident[0] + 1) if len(overconfident) > 0 else None,
    'coverage_at_h50': float(cov_all[49]),
    'coverage_at_h200': float(cov_all[199]),
    'runtime_s': time.time() - t0,
}
with open(OUT / "summary_diag05.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag05 complete ({time.time()-t0:.0f}s). Output: {OUT}")
