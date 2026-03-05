"""
Diagnostic 2: Single Continuous Open-Loop Rollout
==================================================
NO TRAINING. Initialize at start of test, roll out the entire test period
without measurement updates. Oracle/persistence/no_forcing.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag02")

print("=" * 60)
print("DIAG 02: Single Continuous Open-Loop Rollout")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

D = U.load_final_model_and_data()
params, cl = D['params'], D['cl_params']
t_arr, x_arr, v_arr = D['t_arr'], D['x_arr'], D['v_arr']
test_idx = D['test_idx_full']
t_test, x_test = D['t_test'], D['x_test']
N_test = D['N_test']

filt = D['filter']
i_start = test_idx[0]  # start of test in full array

# Detect events for annotation
ev = U.detect_events_from_x(x_test, t_test)

# Rollout entire test from filtered state at start
H = N_test  # roll out full test length
trajectories = {}
for mode in U.MODES:
    print(f"  Rolling out {mode} for {H} steps...")
    r = U.rollout_open_loop(
        filt['states_x'][i_start], filt['states_u'][i_start],
        i_start, H, t_arr, v_arr, params, cl, mode=mode)
    trajectories[mode] = r['path_x']

# Compute metrics
summary_data = {}
for mode in U.MODES:
    pred = trajectories[mode]
    valid = ~np.isnan(pred) & (np.arange(len(pred)) < N_test)
    n_valid = int(valid.sum())
    err = x_test[:n_valid] - pred[:n_valid]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))
    drift = float(pred[n_valid-1] - x_test[n_valid-1]) if n_valid > 0 else np.nan

    # Count event misses (times model is on wrong side of midpoint)
    event_misses = 0
    for ei in ev['event_indices']:
        if ei < n_valid:
            true_state = ev['states'][ei]
            pred_state = 1 if pred[ei] > ev['midpoint'] else 0
            if pred_state != true_state:
                event_misses += 1

    summary_data[mode] = {
        'n_valid': n_valid, 'rmse': rmse, 'mae': mae,
        'max_error': max_err, 'final_drift': drift,
        'event_misses': event_misses,
        'n_events': len(ev['event_indices']),
    }
    print(f"    {mode}: RMSE={rmse:.6f}, MAE={mae:.6f}, drift={drift:.6f}, "
          f"event_misses={event_misses}/{len(ev['event_indices'])}")

# Figure 1: Full overlay
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1, 1.5]})
ax = axes[0]
ax.plot(t_test, x_test, 'k-', lw=0.8, label='True x')
for mode in U.MODES:
    n = min(N_test, int(np.sum(~np.isnan(trajectories[mode]))))
    ax.plot(t_test[:n], trajectories[mode][:n], '-', lw=0.8,
            color=U.MODE_COLORS[mode], alpha=0.8,
            label=f'{U.MODE_LABELS[mode]} (RMSE={summary_data[mode]["rmse"]:.4f})')
for et in (ev['event_times'] or []):
    ax.axvline(et, color='gray', ls=':', lw=0.4, alpha=0.6)
ax.set_ylabel('Displacement (m)')
ax.set_title('Single Continuous Open-Loop Rollout (entire test, no reinitialization)')
ax.legend(fontsize=8, loc='upper left')

ax = axes[1]
ax.plot(t_test, D['v_test'], color='steelblue', lw=0.4)
ax.set_ylabel('Velocity (m/s)')

ax = axes[2]
for mode in U.MODES:
    n = min(N_test, int(np.sum(~np.isnan(trajectories[mode]))))
    err = x_test[:n] - trajectories[mode][:n]
    ax.plot(t_test[:n], err, '-', lw=0.5, color=U.MODE_COLORS[mode],
            alpha=0.7, label=U.MODE_LABELS[mode])
ax.axhline(0, color='gray', ls='--', lw=0.5)
ax.set_ylabel('Error (m)')
ax.set_xlabel('Time (s)')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "full_open_loop_overlay.png", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {OUT / 'full_open_loop_overlay.png'}")

# Figure 2: Error time series (zoomed)
fig, ax = plt.subplots(figsize=(12, 4))
for mode in U.MODES:
    n = min(N_test, int(np.sum(~np.isnan(trajectories[mode]))))
    err = x_test[:n] - trajectories[mode][:n]
    ax.plot(t_test[:n], err, '-', lw=0.6, color=U.MODE_COLORS[mode],
            label=U.MODE_LABELS[mode])
ax.axhline(0, color='gray', ls='--', lw=0.5)
for et in (ev['event_times'] or []):
    ax.axvline(et, color='gray', ls=':', lw=0.4, alpha=0.6)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Error: x_true - x_pred (m)')
ax.set_title('Open-Loop Prediction Error (no reinitialization)')
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "error_time_series.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT / 'error_time_series.png'}")

# Summary JSON
with open(OUT / "summary_diag02.json", 'w') as f:
    json.dump({
        'modes': summary_data,
        'n_events_test': len(ev['event_indices']),
        'event_times': ev['event_times'],
        'test_duration_s': float(t_test[-1] - t_test[0]),
        'runtime_s': time.time() - t0,
    }, f, indent=2)

print(f"\nDiag02 complete ({time.time()-t0:.0f}s). Output: {OUT}")
