"""
Diagnostic 10: Leave-One-Event-Out Influence
=============================================
NO TRAINING. For each major event in test, remove [t_i - 10s, t_i + 10s]
from evaluation. Recompute DxR2@50 and MAE@50 for oracle and persistence.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag10")

H_EVAL = 50
ABLATE_RADIUS = 10.0  # seconds

print("=" * 60)
print("DIAG 10: Leave-One-Event-Out Influence")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

D = U.load_final_model_and_data()
params, cl = D['params'], D['cl_params']
t_arr, x_arr, v_arr = D['t_arr'], D['x_arr'], D['v_arr']
test_idx = D['test_idx_full']
x_test, t_test = D['x_test'], D['t_test']
N_test = D['N_test']
N_full = len(t_arr)

# Compute rollout paths (only need h=50)
print(f"\nComputing {H_EVAL}-step rollout paths...")
paths = U.compute_all_rollout_paths(
    test_idx, H_EVAL, D['filter'], t_arr, v_arr, params, cl)

# Event detection
ev = U.detect_events_from_x(x_test, t_test)
event_times = ev['event_times'] or []
print(f"  {len(event_times)} transitions: {event_times}")

# Full-test baseline metrics
full_metrics = {}
for mode in ['oracle', 'persistence']:
    r2, n = U.dxr2_at_horizon(
        paths[mode], x_test, x_arr, test_idx, H_EVAL, N_full)
    mae, _ = U.mae_at_horizon(paths[mode], x_arr, test_idx, H_EVAL, N_full)
    full_metrics[mode] = {'dxr2': r2, 'mae': mae, 'n': n}
    print(f"  Full test {mode}: DxR2@{H_EVAL}={r2:+.4f}, MAE={mae:.6f}")

# Leave-one-event-out
rows = []
for ev_i, et in enumerate(event_times):
    # Mask: keep all points NOT within +/-ABLATE_RADIUS of this event
    keep = np.ones(N_test, dtype=bool)
    for j in range(N_test):
        if abs(t_test[j] - et) <= ABLATE_RADIUS:
            keep[j] = False

    n_removed = int((~keep).sum())

    for mode in ['oracle', 'persistence']:
        r2, n = U.dxr2_at_horizon(
            paths[mode], x_test, x_arr, test_idx, H_EVAL, N_full, keep)
        mae, _ = U.mae_at_horizon(
            paths[mode], x_arr, test_idx, H_EVAL, N_full, keep)
        delta_r2 = r2 - full_metrics[mode]['dxr2']
        delta_mae = mae - full_metrics[mode]['mae']

        rows.append({
            'event_idx': ev_i,
            'event_time': et,
            'mode': mode,
            'n_removed': n_removed,
            'n_remaining': n,
            'dxr2_loo': r2,
            'dxr2_full': full_metrics[mode]['dxr2'],
            'delta_dxr2': delta_r2,
            'mae_loo': mae,
            'mae_full': full_metrics[mode]['mae'],
            'delta_mae': delta_mae,
        })

    print(f"  Event {ev_i} (t={et:.1f}s): removed {n_removed} pts, "
          f"oracle delta_DxR2={rows[-2]['delta_dxr2']:+.4f}, "
          f"persist delta_DxR2={rows[-1]['delta_dxr2']:+.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "loo_event_influence.csv", index=False)

# Figure: bar plot of delta DxR2 by removed event
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, mode in zip(axes, ['oracle', 'persistence']):
    sub = df[df['mode'] == mode]
    x_pos = np.arange(len(sub))
    deltas = sub['delta_dxr2'].values
    colors = ['red' if d < 0 else 'green' for d in deltas]
    ax.bar(x_pos, deltas, color=colors, edgecolor='k', lw=0.5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f't={et:.0f}s' for et in sub['event_time']],
                       fontsize=8, rotation=30)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel(f'Delta DxR2@{H_EVAL}')
    ax.set_title(f'{U.MODE_LABELS[mode]}: Leave-One-Event-Out')
    for i, d in enumerate(deltas):
        ax.text(i, d + 0.002 * np.sign(d), f'{d:+.3f}', ha='center',
                fontsize=7, va='bottom' if d >= 0 else 'top')

fig.suptitle(f'Event Influence on DxR2@{H_EVAL} '
             f'(+/-{ABLATE_RADIUS:.0f}s removed)', fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "loo_event_influence.png", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {OUT / 'loo_event_influence.png'}")

# Summary: does a few events dominate?
oracle_sub = df[df['mode'] == 'oracle']
persist_sub = df[df['mode'] == 'persistence']

summary = {
    'h_eval': H_EVAL,
    'ablate_radius_s': ABLATE_RADIUS,
    'n_events': len(event_times),
    'event_times': event_times,
    'full_metrics': full_metrics,
    'oracle_delta_range': [float(oracle_sub['delta_dxr2'].min()),
                           float(oracle_sub['delta_dxr2'].max())],
    'persist_delta_range': [float(persist_sub['delta_dxr2'].min()),
                            float(persist_sub['delta_dxr2'].max())],
    'oracle_max_influence': float(oracle_sub['delta_dxr2'].abs().max()),
    'persist_max_influence': float(persist_sub['delta_dxr2'].abs().max()),
    'dominated_by_few': bool(persist_sub['delta_dxr2'].abs().max() > 0.10),
    'runtime_s': time.time() - t0,
}
with open(OUT / "summary_diag10.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag10 complete ({time.time()-t0:.0f}s). Output: {OUT}")
