"""
Diagnostic 7: Innovation ACF Conditioned on Forcing / Energy / Events
=====================================================================
NO TRAINING. Bin innovations by |u|, energy proxy, near-event status.
Compute ACF and Ljung-Box for each bin.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag07")
MAX_LAG = 50
EWMA_ALPHA = 0.1  # energy proxy decay (10% new -> ~1s effective window)

print("=" * 60)
print("DIAG 07: Conditional Innovation ACF")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

D = U.load_final_model_and_data()
t_test, x_test, v_test = D['t_test'], D['x_test'], D['v_test']
filt = D['filter']
base = D['baseline_filter']
test_mask = D['test_mask']
N_test = D['N_test']

# Extract test innovations (closure model)
innov = filt['innovations'][test_mask]
valid = ~np.isnan(innov)
innov_v = innov[valid]
v_v = v_test[valid]
x_v = x_test[valid]

# Also baseline innovations
innov_base = base['innovations'][test_mask]
valid_b = ~np.isnan(innov_base)

# Internal velocity from filter states
u_filt = filt['states_u'][test_mask][valid]

# Energy proxy: EWMA of v^2
E = np.zeros(len(v_v))
E[0] = v_v[0]**2
for i in range(1, len(v_v)):
    E[i] = EWMA_ALPHA * v_v[i]**2 + (1 - EWMA_ALPHA) * E[i-1]

# Event detection
ev = U.detect_events_from_x(x_test, D['t_test'])
event_times = ev['event_times'] or []

# Near-event mask (within 1s before any transition)
t_v = D['t_test'][valid]
near_event = np.zeros(len(t_v), dtype=bool)
for et in event_times:
    near_event |= (t_v >= et - 1.0) & (t_v <= et)
far_event = ~near_event

# Bin by |u| quantiles (terciles)
abs_u = np.abs(u_filt)
u_q33, u_q67 = np.percentile(abs_u, [33, 67])
u_low = abs_u <= u_q33
u_med = (abs_u > u_q33) & (abs_u <= u_q67)
u_high = abs_u > u_q67

# Bin by energy quantiles (terciles)
e_q33, e_q67 = np.percentile(E, [33, 67])
e_low = E <= e_q33
e_med = (E > e_q33) & (E <= e_q67)
e_high = E > e_q67

# Define bins
bins = {
    '|u| low': u_low, '|u| med': u_med, '|u| high': u_high,
    'E low': e_low, 'E med': e_med, 'E high': e_high,
    'Near event': near_event, 'Far from event': far_event,
    'All': np.ones(len(innov_v), dtype=bool),
}

# Compute ACF and Ljung-Box for each bin
results = {}
lb_rows = []
sig_band = 1.96 / np.sqrt(len(innov_v))

for name, mask in bins.items():
    e_bin = innov_v[mask]
    if len(e_bin) < 20:
        print(f"  {name}: n={len(e_bin)} -- too few, skipping")
        continue
    acf = U.compute_acf(e_bin, MAX_LAG)
    lb = U.ljung_box(acf, len(e_bin))
    results[name] = {
        'acf': acf.tolist(),
        'n': len(e_bin),
        'acf1': float(acf[1]),
    }
    for r in lb:
        lb_rows.append({'bin': name, **r})
    print(f"  {name:20s}: n={len(e_bin):5d}  ACF(1)={acf[1]:+.4f}  "
          f"LB(10)_p={next((r['p'] for r in lb if r['lag']==10), np.nan):.4f}")

# Save Ljung-Box table
pd.DataFrame(lb_rows).to_csv(OUT / "ljung_box_by_bin.csv", index=False)

# Figure: multi-panel ACF by bin category
categories = [
    ('|u| quantile', ['|u| low', '|u| med', '|u| high']),
    ('Energy quantile', ['E low', 'E med', 'E high']),
    ('Event proximity', ['Near event', 'Far from event']),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors_bins = ['#1f77b4', '#ff7f0e', '#2ca02c']

for ax, (cat_name, bin_names) in zip(axes, categories):
    for i, bn in enumerate(bin_names):
        if bn in results:
            acf = np.array(results[bn]['acf'])
            lags = np.arange(len(acf))
            ax.bar(lags + i*0.2, acf, width=0.2, color=colors_bins[i],
                   alpha=0.7, label=f'{bn} (n={results[bn]["n"]})')
    ax.axhline(sig_band, color='red', ls='--', lw=0.5, alpha=0.7)
    ax.axhline(-sig_band, color='red', ls='--', lw=0.5, alpha=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title(cat_name)
    ax.legend(fontsize=7)
    ax.set_xlim(-0.5, min(20, MAX_LAG) + 1)

fig.suptitle('Innovation ACF by Condition (closure model)', fontsize=12)
plt.tight_layout()
fig.savefig(OUT / "acf_by_bin.png", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {OUT / 'acf_by_bin.png'}")

# Summary: where does autocorrelation concentrate?
acf1_by_bin = {name: results[name]['acf1'] for name in results}
max_acf1_bin = max(acf1_by_bin, key=lambda k: abs(acf1_by_bin[k]))

summary = {
    'bins': {name: {'n': results[name]['n'], 'acf1': results[name]['acf1']}
             for name in results},
    'max_acf1_bin': max_acf1_bin,
    'max_acf1_value': acf1_by_bin[max_acf1_bin],
    'sig_band_95': float(sig_band),
    'runtime_s': time.time() - t0,
}
with open(OUT / "summary_diag07.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag07 complete ({time.time()-t0:.0f}s). Output: {OUT}")
