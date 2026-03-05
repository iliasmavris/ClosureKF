"""
Diagnostic 1: Full Horizon Skill Curves (h=1..200)
===================================================
NO TRAINING. Computes DxR2(h) and MAE(h) for oracle/persistence/no_forcing.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag01")
MAX_H = 200

print("=" * 60)
print("DIAG 01: Horizon Skill Curves h=1..200")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

# Load
D = U.load_final_model_and_data()
params, cl = D['params'], D['cl_params']
t_arr, x_arr, v_arr = D['t_arr'], D['x_arr'], D['v_arr']
test_idx = D['test_idx_full']
x_test = D['x_test']
N_full = len(t_arr)

# Compute all rollout paths
print(f"\nComputing {MAX_H}-step rollout paths for {D['N_test']} test points...")
paths = U.compute_all_rollout_paths(
    test_idx, MAX_H, D['filter'], t_arr, v_arr, params, cl)
print("  Done.")

# Compute DxR2 and MAE at each horizon
horizons = list(range(1, MAX_H + 1))
results = {}
for mode in U.MODES:
    print(f"  Computing metrics for {mode}...")
    dxr2_list = U.dxr2_by_horizon(
        paths[mode], x_test, x_arr, test_idx, horizons, N_full)
    mae_list = U.mae_by_horizon(
        paths[mode], x_arr, test_idx, horizons, N_full)
    results[mode] = {
        'dxr2': [r[0] for r in dxr2_list],
        'mae': [r[0] for r in mae_list],
        'n_valid': [r[1] for r in dxr2_list],
    }

# Find crossover horizons (first h where DxR2 > 0)
crossovers = {}
for mode in U.MODES:
    cross = next((h for h, r2 in zip(horizons, results[mode]['dxr2'])
                  if r2 is not None and not np.isnan(r2) and r2 > 0), None)
    crossovers[mode] = cross

# Print summary table
print(f"\n  {'h':>5s}  {'Oracle':>10s}  {'Persist':>10s}  {'NoForce':>10s}  {'t(s)':>6s}")
print(f"  {'-----':>5s}  {'------':>10s}  {'-------':>10s}  {'-------':>10s}  {'----':>6s}")
for h in [1, 5, 10, 20, 50, 100, 150, 200]:
    if h <= MAX_H:
        vals = [results[m]['dxr2'][h-1] for m in U.MODES]
        line = f"  {h:5d}"
        for v in vals:
            line += f"  {v:+10.4f}" if not np.isnan(v) else f"  {'nan':>10s}"
        line += f"  {h*U.DT:6.1f}"
        print(line)

for mode in U.MODES:
    c = crossovers[mode]
    if c:
        print(f"  {mode} crossover: h={c} ({c*U.DT:.1f}s)")

# Save CSV
rows = []
for i, h in enumerate(horizons):
    row = {'h': h, 'time_s': h * U.DT}
    for mode in U.MODES:
        row[f'dxr2_{mode}'] = results[mode]['dxr2'][i]
        row[f'mae_{mode}'] = results[mode]['mae'][i]
    row['n_valid'] = results['oracle']['n_valid'][i]
    rows.append(row)
df_dxr2 = __import__('pandas').DataFrame(rows)
df_dxr2.to_csv(OUT / "dxr2_by_horizon.csv", index=False)
df_dxr2.to_csv(OUT / "mae_by_horizon.csv", index=False)

# Figure 1: DxR2(h) curve
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
h_s = np.array(horizons) * U.DT
for mode in U.MODES:
    axes[0].plot(h_s, results[mode]['dxr2'], '-', lw=1.5,
                 color=U.MODE_COLORS[mode], label=U.MODE_LABELS[mode])
axes[0].axhline(0, color='gray', ls='--', lw=0.8)
for mode in U.MODES:
    c = crossovers[mode]
    if c:
        axes[0].axvline(c * U.DT, color=U.MODE_COLORS[mode], ls=':', lw=0.6, alpha=0.5)
axes[0].set_ylabel('DxR2(h)')
axes[0].set_title(f'Displacement-Increment Skill: h=1..{MAX_H} (0.1..{MAX_H*U.DT:.0f}s)')
axes[0].legend(loc='lower right')

# Oracle advantage
gap_op = np.array(results['oracle']['dxr2']) - np.array(results['persistence']['dxr2'])
gap_on = np.array(results['oracle']['dxr2']) - np.array(results['no_forcing']['dxr2'])
axes[1].plot(h_s, gap_op, '-', color='purple', lw=1.0, label='Oracle - Persistence')
axes[1].plot(h_s, gap_on, '-', color='orange', lw=1.0, label='Oracle - No forcing')
axes[1].axhline(0, color='gray', ls='--', lw=0.5)
axes[1].set_xlabel('Horizon (s)')
axes[1].set_ylabel('Advantage')
axes[1].legend(loc='best', fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "dxr2_curve.png", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {OUT / 'dxr2_curve.png'}")

# Figure 2: MAE(h) curve
fig, ax = plt.subplots(figsize=(10, 5))
for mode in U.MODES:
    ax.plot(h_s, results[mode]['mae'], '-', lw=1.5,
            color=U.MODE_COLORS[mode], label=U.MODE_LABELS[mode])
ax.set_xlabel('Horizon (s)')
ax.set_ylabel('MAE (m)')
ax.set_title(f'Mean Absolute Error: h=1..{MAX_H}')
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "mae_curve.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT / 'mae_curve.png'}")

# Summary JSON
oracle_gap_area = float(np.nansum(gap_op) * U.DT)
summary = {
    'max_horizon': MAX_H,
    'crossover_h': {m: crossovers[m] for m in U.MODES},
    'crossover_s': {m: crossovers[m] * U.DT if crossovers[m] else None
                    for m in U.MODES},
    'plateau_dxr2_h200': {m: results[m]['dxr2'][-1] for m in U.MODES},
    'dxr2_h50': {m: results[m]['dxr2'][49] for m in U.MODES},
    'oracle_persistence_gap_area': oracle_gap_area,
    'runtime_s': time.time() - t0,
}
with open(OUT / "summary_diag01.json", 'w') as f:
    json.dump(summary, f, indent=2)

elapsed = time.time() - t0
print(f"\nDiag01 complete ({elapsed:.0f}s). Output: {OUT}")
