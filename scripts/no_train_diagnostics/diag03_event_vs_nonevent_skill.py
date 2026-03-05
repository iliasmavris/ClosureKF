"""
Diagnostic 3: Event vs Non-Event Skill Decomposition
=====================================================
NO TRAINING. For h in {10, 50, 100}, label each test point as event/non-event
(transition in (t, t+h]), compute DxR2 and MAE separately.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag03")
MAX_H = 100
EVAL_HORIZONS = [10, 50, 100]

print("=" * 60)
print("DIAG 03: Event vs Non-Event Skill Decomposition")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

D = U.load_final_model_and_data()
params, cl = D['params'], D['cl_params']
t_arr, x_arr, v_arr = D['t_arr'], D['x_arr'], D['v_arr']
test_idx = D['test_idx_full']
x_test, t_test = D['x_test'], D['t_test']
N_full = len(t_arr)
N_test = D['N_test']

# Compute paths
print(f"\nComputing {MAX_H}-step rollout paths...")
paths = U.compute_all_rollout_paths(
    test_idx, MAX_H, D['filter'], t_arr, v_arr, params, cl)

# Detect events
ev = U.detect_events_from_x(x_test, t_test)
event_times = ev['event_times'] or []
print(f"  {len(event_times)} transitions detected")

# For each horizon h, build event/non-event masks
rows = []
for h in EVAL_HORIZONS:
    # Label: event if any transition in (t_j, t_j + h*DT]
    window_s = h * U.DT
    event_mask = np.zeros(N_test, dtype=bool)
    for j in range(N_test):
        t_j = t_test[j]
        for et in event_times:
            if t_j < et <= t_j + window_s:
                event_mask[j] = True
                break
    nonevent_mask = ~event_mask
    n_event = int(event_mask.sum())
    n_nonevent = int(nonevent_mask.sum())

    for mode in U.MODES:
        # All
        r2_all, n_all = U.dxr2_at_horizon(
            paths[mode], x_test, x_arr, test_idx, h, N_full)
        mae_all, _ = U.mae_at_horizon(paths[mode], x_arr, test_idx, h, N_full)
        # Event
        r2_ev, n_ev = U.dxr2_at_horizon(
            paths[mode], x_test, x_arr, test_idx, h, N_full, event_mask)
        mae_ev, _ = U.mae_at_horizon(
            paths[mode], x_arr, test_idx, h, N_full, event_mask)
        # Non-event
        r2_ne, n_ne = U.dxr2_at_horizon(
            paths[mode], x_test, x_arr, test_idx, h, N_full, nonevent_mask)
        mae_ne, _ = U.mae_at_horizon(
            paths[mode], x_arr, test_idx, h, N_full, nonevent_mask)

        rows.append({
            'horizon': h, 'time_s': h * U.DT, 'mode': mode,
            'n_event': n_event, 'n_nonevent': n_nonevent,
            'dxr2_all': r2_all, 'dxr2_event': r2_ev, 'dxr2_nonevent': r2_ne,
            'mae_all': mae_all, 'mae_event': mae_ev, 'mae_nonevent': mae_ne,
            'n_eval_event': n_ev, 'n_eval_nonevent': n_ne,
        })
        print(f"  h={h:3d} {mode:12s}: all={r2_all:+.4f}  "
              f"event={r2_ev:+.4f}(n={n_ev})  nonevent={r2_ne:+.4f}(n={n_ne})")

df = pd.DataFrame(rows)
df.to_csv(OUT / "skill_split_table.csv", index=False)

# Figure: grouped bar chart
fig, axes = plt.subplots(1, len(EVAL_HORIZONS), figsize=(5*len(EVAL_HORIZONS), 5),
                         sharey=True)
if len(EVAL_HORIZONS) == 1:
    axes = [axes]

for ax, h in zip(axes, EVAL_HORIZONS):
    sub = df[df['horizon'] == h]
    x_pos = np.arange(len(U.MODES))
    width = 0.25

    ev_vals = [sub[sub['mode']==m]['dxr2_event'].values[0] for m in U.MODES]
    ne_vals = [sub[sub['mode']==m]['dxr2_nonevent'].values[0] for m in U.MODES]
    all_vals = [sub[sub['mode']==m]['dxr2_all'].values[0] for m in U.MODES]

    ax.bar(x_pos - width, ev_vals, width, label='Event', color='salmon', edgecolor='k', lw=0.5)
    ax.bar(x_pos, ne_vals, width, label='Non-event', color='lightblue', edgecolor='k', lw=0.5)
    ax.bar(x_pos + width, all_vals, width, label='All', color='lightgray', edgecolor='k', lw=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([U.MODE_LABELS[m] for m in U.MODES], fontsize=8)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_title(f'h={h} ({h*U.DT:.0f}s)')
    if ax == axes[0]:
        ax.set_ylabel('DxR2')
        ax.legend(fontsize=8)

fig.suptitle('Event vs Non-Event Skill Decomposition', fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "skill_split_bars.png", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {OUT / 'skill_split_bars.png'}")

# Summary
event_share = {h: float(df[(df['horizon']==h) & (df['mode']=='oracle')]['n_event'].values[0])
               / N_test for h in EVAL_HORIZONS}

# Is skill dominated by non-events?
dominated = {}
for h in EVAL_HORIZONS:
    ne = df[(df['horizon']==h) & (df['mode']=='oracle')]['dxr2_nonevent'].values[0]
    dominated[str(h)] = bool(ne > 0.05)

summary = {
    'horizons': EVAL_HORIZONS,
    'event_share': {str(h): event_share[h] for h in EVAL_HORIZONS},
    'nonevent_skill_positive': dominated,
    'n_events': len(event_times),
    'runtime_s': time.time() - t0,
}
with open(OUT / "summary_diag03.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag03 complete ({time.time()-t0:.0f}s). Output: {OUT}")
