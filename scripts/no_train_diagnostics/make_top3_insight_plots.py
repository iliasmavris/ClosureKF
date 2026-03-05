"""
Top 3 Insight Plots for Publication
====================================
NO TRAINING.  Uses the final Closure-2t model (v2, seed-averaged closure).

Output:  final_lockbox_vX_no_train_diagnostics/top3_plots/

Plot 1 -- Skill vs Horizon with Event / Non-Event Decomposition
Plot 2 -- Phase Portrait with Flow Coloring
Plot 3 -- Event-Centered Forecast Fan Chart
"""

import sys, time, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# --- matplotlib setup (BEFORE importing utils, which may force Agg) ---
import matplotlib
# Try to keep an interactive backend for plt.show(); fall back to Agg
_BACKEND_OK = False
for _backend in ['TkAgg', 'Qt5Agg', 'WXAgg']:
    try:
        matplotlib.use(_backend)
        _BACKEND_OK = True
        break
    except Exception:
        continue
if not _BACKEND_OK:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as mcm

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3,
})

import utils_no_train as U

# ===================================================================
#  Config
# ===================================================================
OUT = U.OUT_ROOT / "top3_plots"
OUT.mkdir(parents=True, exist_ok=True)

MAX_H       = 100          # 10 s at 10 Hz
DT          = 0.1
H_ANNOTATE  = [10, 50, 100]   # steps to annotate (1 s, 5 s, 10 s)
FAN_EVERY   = 5               # Plot 3: oracle forecast every N steps
FAN_PERSIST = 10               # Plot 3: persistence forecast every N steps
N_BOOT      = 200              # bootstrap iterations for CI
BLOCK_SIZE  = 50               # bootstrap block size (5 s)
WINDOW_SEC  = 10.0             # Plot 3: +/- seconds around event

print("=" * 70)
print("  TOP 3 INSIGHT PLOTS -- NO TRAINING")
print("  Model: Closure-2t (v2), seed-averaged")
print(f"  Output: {OUT}")
print("=" * 70)
t0_global = time.time()

# ===================================================================
#  1. Load model, data, filters, rollouts
# ===================================================================
print("\n[1/6] Loading model and data ...")
D = U.load_final_model_and_data()
params, cl = D['params'], D['cl_params']
t_arr, x_arr, v_arr = D['t_arr'], D['x_arr'], D['v_arr']
test_idx  = D['test_idx_full']
x_test    = D['x_test']
t_test    = D['t_test']
v_test    = D['v_test']
N_full    = len(t_arr)
N_test    = D['N_test']
filt      = D['filter']

print(f"\n[2/6] Computing {MAX_H}-step rollout paths ({N_test} test points) ...")
paths = U.compute_all_rollout_paths(
    test_idx, MAX_H, filt, t_arr, v_arr, params, cl, verbose=True)
print("  Done.")

# ===================================================================
#  2. Event detection
# ===================================================================
print("\n[3/6] Detecting events ...")
ev = U.detect_events_from_x(x_test, t_test)
event_indices = ev['event_indices']
event_times   = ev['event_times'] or []
centroids     = ev['centroids']
states        = ev['states']
n_events      = len(event_indices)
print(f"  {n_events} transitions detected")
print(f"  Centroids: A = {centroids[0]:.4f} m,  B = {centroids[1]:.4f} m")
print(f"  Thresholds: up = {ev['thresh_up']:.4f},  down = {ev['thresh_down']:.4f}")

# Forward distance to next event (efficient O(N) sweep)
event_idx_set = set(event_indices)
fwd_dist = np.full(N_test, MAX_H + 1, dtype=int)
for j in range(N_test - 2, -1, -1):
    if (j + 1) in event_idx_set:
        fwd_dist[j] = 1
    elif fwd_dist[j + 1] < MAX_H + 1:
        fwd_dist[j] = fwd_dist[j + 1] + 1

# ===================================================================
#  3. Pre-compute displacement-increment arrays (vectorised)
# ===================================================================
dx_pred = {}
for mode in ['oracle', 'persistence']:
    dx_pred[mode] = paths[mode] - x_test[:, None]        # (N_test, MAX_H)

dx_obs = np.full((N_test, MAX_H), np.nan)
for h in range(1, MAX_H + 1):
    valid = test_idx + h < N_full
    dx_obs[valid, h - 1] = x_arr[test_idx[valid] + h] - x_test[valid]


# ===================================================================
#  PLOT 1 -- Skill vs Horizon with Event / Non-Event Decomposition
# ===================================================================
print("\n[4/6] PLOT 1: Skill vs Horizon ...")
t1 = time.time()

horizons = np.arange(1, MAX_H + 1)
h_sec    = horizons * DT

curves = {}
for mode in ['oracle', 'persistence']:
    c_all = np.full(MAX_H, np.nan)
    c_ev  = np.full(MAX_H, np.nan)
    c_ne  = np.full(MAX_H, np.nan)
    for h in horizons:
        hi = h - 1
        r2, _ = U.dxr2_at_horizon(
            paths[mode], x_test, x_arr, test_idx, h, N_full)
        c_all[hi] = r2

        event_mask    = fwd_dist <= h
        nonevent_mask = ~event_mask
        r2_ev, _ = U.dxr2_at_horizon(
            paths[mode], x_test, x_arr, test_idx, h, N_full, event_mask)
        r2_ne, _ = U.dxr2_at_horizon(
            paths[mode], x_test, x_arr, test_idx, h, N_full, nonevent_mask)
        c_ev[hi]  = r2_ev
        c_ne[hi]  = r2_ne
    curves[mode] = {'all': c_all, 'event': c_ev, 'nonevent': c_ne}

# Crossovers (first h where DxR2 > 0)
crossover_h = {}
for mode in ['oracle', 'persistence']:
    cross = next((int(h) for h, r2 in zip(horizons, curves[mode]['all'])
                  if not np.isnan(r2) and r2 > 0), None)
    crossover_h[mode] = cross

# Block bootstrap CI (overall curves only, every 5th horizon, then interpolate)
print("  Bootstrap CI bands ...")

def _fast_bootstrap_r2(dp, do, n_boot, block_size, rng):
    valid = ~np.isnan(dp) & ~np.isnan(do)
    dp_v, do_v = dp[valid], do[valid]
    n = len(dp_v)
    if n < 20:
        return np.nan, np.nan
    nb = max(1, n // block_size)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.randint(0, max(1, n - block_size + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block_size, n))
                              for s in starts])[:n]
        dp_b, do_b = dp_v[idx], do_v[idx]
        ss_r = np.sum((do_b - dp_b) ** 2)
        ss_t = np.sum((do_b - do_b.mean()) ** 2)
        boots[b] = 1 - ss_r / ss_t if ss_t > 1e-15 else 0.0
    return float(np.percentile(boots, 5)), float(np.percentile(boots, 95))

ci_bands = {}
rng_boot = np.random.RandomState(42)
for mode in ['oracle', 'persistence']:
    lo_raw = np.full(MAX_H, np.nan)
    hi_raw = np.full(MAX_H, np.nan)
    for h in range(5, MAX_H + 1, 5):
        lo_v, hi_v = _fast_bootstrap_r2(
            dx_pred[mode][:, h - 1], dx_obs[:, h - 1],
            N_BOOT, BLOCK_SIZE, rng_boot)
        lo_raw[h - 1] = lo_v
        hi_raw[h - 1] = hi_v
    mask_ok = ~np.isnan(lo_raw)
    if np.sum(mask_ok) > 2:
        from scipy.interpolate import interp1d
        h_ok = h_sec[mask_ok]
        lo_interp = interp1d(h_ok, lo_raw[mask_ok],
                             fill_value='extrapolate')(h_sec)
        hi_interp = interp1d(h_ok, hi_raw[mask_ok],
                             fill_value='extrapolate')(h_sec)
    else:
        lo_interp, hi_interp = lo_raw, hi_raw
    ci_bands[mode] = {'lo': lo_interp, 'hi': hi_interp}

# --- Draw Plot 1 ---
COL_O = {'all': '#D62828', 'event': '#F77F00', 'nonevent': '#9D0208'}
COL_P = {'all': '#1B4965', 'event': '#5FA8D3', 'nonevent': '#0B2545'}
LS    = {'all': '-', 'event': '--', 'nonevent': '-.'}
LW    = {'all': 2.5, 'event': 1.8, 'nonevent': 1.3}

fig1, ax1 = plt.subplots(figsize=(13, 7))

for mode, COL in [('oracle', COL_O), ('persistence', COL_P)]:
    prefix = 'Oracle-v' if mode == 'oracle' else 'Persist-v'
    for seg in ['all', 'event', 'nonevent']:
        seg_lbl = {'all': 'overall', 'event': 'event', 'nonevent': 'non-event'}[seg]
        ax1.plot(h_sec, curves[mode][seg], ls=LS[seg], lw=LW[seg],
                 color=COL[seg], label=f'{prefix} {seg_lbl}')
    # CI band (overall only)
    ax1.fill_between(h_sec, ci_bands[mode]['lo'], ci_bands[mode]['hi'],
                     color=COL['all'], alpha=0.07)

# Zero line
ax1.axhline(0, color='gray', ls='--', lw=0.8, zorder=1)

# Crossover vertical lines
for mode, col in [('oracle', COL_O['all']), ('persistence', COL_P['all'])]:
    ch = crossover_h.get(mode)
    if ch:
        ax1.axvline(ch * DT, color=col, ls=':', lw=1.2, alpha=0.5)
        ax1.text(ch * DT + 0.08, -0.02, f'{ch * DT:.1f}s',
                 fontsize=9, color=col, va='top', ha='left')

# Annotate specific horizons
for h in H_ANNOTATE:
    t_s = h * DT
    ax1.axvline(t_s, color='#AAAAAA', ls=':', lw=0.5, alpha=0.4)
    r2_o = curves['oracle']['all'][h - 1]
    r2_p = curves['persistence']['all'][h - 1]

    # Oracle label
    ax1.annotate(f'{r2_o:+.3f}', (t_s, r2_o),
                 textcoords='offset points', xytext=(6, 10),
                 fontsize=8, color=COL_O['all'], weight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))
    # Persistence label
    ax1.annotate(f'{r2_p:+.3f}', (t_s, r2_p),
                 textcoords='offset points', xytext=(6, -14),
                 fontsize=8, color=COL_P['all'], weight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))

ax1.set_xlabel('Forecast Horizon (seconds)')
ax1.set_ylabel(r'$\Delta x\;\mathrm{R}^2(h)$')
ax1.set_title('Displacement-Increment Skill: Oracle vs Persistence, Event vs Non-Event')
ax1.legend(loc='lower right', ncol=2, fontsize=9, columnspacing=1.0)
ax1.set_xlim(0, MAX_H * DT + 0.3)

fig1.tight_layout()
fig1.savefig(OUT / 'plot1_skill_horizon_split.png', bbox_inches='tight')
print(f"  Saved: plot1_skill_horizon_split.png  ({time.time()-t1:.0f}s)")


# ===================================================================
#  PLOT 2 -- Phase Portrait with Flow Coloring
# ===================================================================
print("\n[5/6] PLOT 2: Phase portrait ...")
t2 = time.time()

states_x_test = filt['states_x'][D['test_mask']]
states_u_test = filt['states_u'][D['test_mask']]
v_abs = np.abs(v_test)

fig2, ax2 = plt.subplots(figsize=(10, 8))

# Faint trajectory line (temporal order)
ax2.plot(states_x_test, states_u_test, 'k-', lw=0.15, alpha=0.10, zorder=0)

# Scatter coloured by |v|
vmax_clip = float(np.percentile(v_abs, 98))
sc = ax2.scatter(states_x_test, states_u_test, c=v_abs,
                 cmap='plasma', s=8, alpha=0.30, edgecolors='none',
                 vmin=0, vmax=vmax_clip, zorder=2)
cbar = fig2.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label('$|v|$ (m/s)', fontsize=12)

# Event transition markers
legend_ab = set()
for ei in event_indices:
    if 0 < ei < N_test:
        going_up = (states[ei] == 1)
        mk  = '^' if going_up else 'v'
        col = '#E63946' if going_up else '#457B9D'
        lbl = None
        tag = 'A->B' if going_up else 'B->A'
        if tag not in legend_ab:
            lbl = tag
            legend_ab.add(tag)
        ax2.plot(states_x_test[ei], states_u_test[ei], mk,
                 color=col, ms=13, mew=1.2, mec='k', zorder=10, label=lbl)

# Equilibrium centroid lines
for ci_val, lbl in zip(centroids, ['A', 'B']):
    ax2.axvline(ci_val, color='gray', ls=':', lw=1.0, alpha=0.5)
    ylims = ax2.get_ylim()
    ax2.text(ci_val, ylims[1] - 0.01 * (ylims[1] - ylims[0]), lbl,
             fontsize=15, ha='center', va='top', color='gray', weight='bold')

ax2.set_xlabel('Displacement $x$ (m)')
ax2.set_ylabel('Latent velocity $u$ (m/s)')
ax2.set_title('Phase Portrait: Filtered State Trajectory on Test Split')
ax2.legend(loc='upper right', fontsize=10)

fig2.tight_layout()
fig2.savefig(OUT / 'plot2_phase_portrait_colored.png', bbox_inches='tight')
print(f"  Saved: plot2_phase_portrait_colored.png  ({time.time()-t2:.0f}s)")


# ===================================================================
#  PLOT 3 -- Event-Centred Forecast Fan Chart
# ===================================================================
print("\n[6/6] PLOT 3: Event forecast fan chart ...")
t3 = time.time()

# Largest single-step |dx| in test set
dx_abs = np.abs(np.diff(x_test))
largest_idx = int(np.argmax(dx_abs))
jump_time   = float(t_test[largest_idx])
jump_dx     = float(x_test[largest_idx + 1] - x_test[largest_idx])
print(f"  Largest jump: test idx {largest_idx},  t = {jump_time:.1f} s,  "
      f"dx = {jump_dx:+.5f} m")

# Window around event
WINDOW_STEPS = int(WINDOW_SEC / DT)
win_lo = max(0, largest_idx - WINDOW_STEPS)
win_hi = min(N_test - 1, largest_idx + WINDOW_STEPS)

# True trajectory in window (extend a bit for fan tails)
plot_lo = max(0, largest_idx - WINDOW_STEPS)
plot_hi = min(N_test - 1, largest_idx + WINDOW_STEPS + MAX_H)
plot_t  = t_test[plot_lo:plot_hi + 1]
plot_x  = x_test[plot_lo:plot_hi + 1]

fig3, ax3 = plt.subplots(figsize=(14, 6))

# True trajectory
ax3.plot(plot_t, plot_x, 'k-', lw=2.8, zorder=20, label='True $x(t)$')

# Event marker
ax3.axvline(jump_time, color='red', ls='--', lw=1.5, alpha=0.6, zorder=15)

# --- Oracle forecast fans ---
oracle_endpoints = {h: [] for h in H_ANNOTATE}
for j_test in range(win_lo, win_hi + 1, FAN_EVERY):
    i_full = test_idx[j_test]
    sx = filt['states_x'][i_full]
    su = filt['states_u'][i_full]
    res = U.rollout_open_loop(
        sx, su, i_full, MAX_H, t_arr, v_arr, params, cl, mode='oracle')
    fan_x = res['path_x']

    # Build time axis for this forecast
    fan_t = np.empty(MAX_H)
    for step in range(MAX_H):
        k = i_full + step + 1
        fan_t[step] = t_arr[k] if k < N_full else np.nan

    ok = ~np.isnan(fan_x) & ~np.isnan(fan_t)
    if np.any(ok):
        ax3.plot(fan_t[ok], fan_x[ok], '-', color='#D62828',
                 alpha=0.12, lw=0.7, zorder=5)
        # Endpoint markers
        for h, mk, ms in [(10, 'o', 4), (50, 'D', 4), (100, 's', 4)]:
            if h - 1 < MAX_H and not np.isnan(fan_x[h - 1]):
                ax3.plot(fan_t[h - 1], fan_x[h - 1], mk, color='#D62828',
                         ms=ms, alpha=0.25, mew=0, zorder=6)
                oracle_endpoints[h].append((fan_t[h - 1], fan_x[h - 1]))

# --- Persistence forecast fans ---
for j_test in range(win_lo, win_hi + 1, FAN_PERSIST):
    i_full = test_idx[j_test]
    sx = filt['states_x'][i_full]
    su = filt['states_u'][i_full]
    res = U.rollout_open_loop(
        sx, su, i_full, MAX_H, t_arr, v_arr, params, cl, mode='persistence')
    fan_x = res['path_x']

    fan_t = np.empty(MAX_H)
    for step in range(MAX_H):
        k = i_full + step + 1
        fan_t[step] = t_arr[k] if k < N_full else np.nan

    ok = ~np.isnan(fan_x) & ~np.isnan(fan_t)
    if np.any(ok):
        ax3.plot(fan_t[ok], fan_x[ok], '--', color='#1B4965',
                 alpha=0.12, lw=0.7, zorder=4)

# Legend (custom handles)
leg_handles = [
    Line2D([0], [0], color='k', lw=2.8, label='True $x(t)$'),
    Line2D([0], [0], color='red', ls='--', lw=1.5, alpha=0.6,
           label='Jump event'),
    Line2D([0], [0], color='#D62828', lw=1.2, alpha=0.5,
           label='Oracle-v forecasts'),
    Line2D([0], [0], color='#1B4965', ls='--', lw=1.2, alpha=0.5,
           label='Persist-v forecasts'),
    Line2D([0], [0], marker='o', color='#D62828', ms=6, lw=0,
           alpha=0.6, label='h = 10 (1 s)'),
    Line2D([0], [0], marker='D', color='#D62828', ms=6, lw=0,
           alpha=0.6, label='h = 50 (5 s)'),
    Line2D([0], [0], marker='s', color='#D62828', ms=6, lw=0,
           alpha=0.6, label='h = 100 (10 s)'),
]
ax3.legend(handles=leg_handles, loc='best', fontsize=9, ncol=2)

# Axes
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Displacement $x$ (m)')
ax3.set_title(
    f'Forecast Fan Chart: $\\pm${WINDOW_SEC:.0f} s Around Largest Event '
    f'($t$ = {jump_time:.1f} s)')
ax3.set_xlim(t_test[plot_lo] - 0.5, t_test[min(plot_hi, N_test - 1)] + 0.5)

fig3.tight_layout()
fig3.savefig(OUT / 'plot3_event_forecast_fan.png', bbox_inches='tight')
print(f"  Saved: plot3_event_forecast_fan.png  ({time.time()-t3:.0f}s)")


# ===================================================================
#  README
# ===================================================================
readme_text = f"""\
# Top 3 Insight Plots

Generated {time.strftime('%Y-%m-%d %H:%M')} -- NO TRAINING.
Model: Closure-2t (v2), seed-averaged closure.

## Plot 1: Skill vs Horizon with Event / Non-Event Decomposition
(`plot1_skill_horizon_split.png`)

- Shows DxR2 (displacement-increment R^2) vs forecast horizon (0.1--10 s).
- Decomposes skill into event windows (transition within forecast window)
  and non-event windows (quiescent dynamics only).
- Oracle-v uses true future water velocity; Persistence-v repeats last known velocity.
- Light shaded bands are 90% block-bootstrap confidence intervals on the overall curves.
- Key insight: skill is negative at short horizons (noise-dominated increments),
  crosses zero around {crossover_h.get('oracle', '?')} steps
  ({crossover_h.get('oracle', 0) * DT:.1f} s for oracle), and grows at longer horizons
  driven primarily by event dynamics.

## Plot 2: Phase Portrait with Flow Coloring
(`plot2_phase_portrait_colored.png`)

- State-space trajectory (displacement x, latent velocity u) from the Kalman-filtered
  posterior mean on the test split.
- Points coloured by instantaneous water velocity magnitude |v| (plasma colourmap).
- Upward/downward triangles mark A->B and B->A transition events.
- Vertical dashed lines at displacement centroids A ({centroids[0]:.4f} m)
  and B ({centroids[1]:.4f} m).
- Shows two metastable wells with orbiting trajectories and
  velocity-forced transitions between them.

## Plot 3: Event-Centred Forecast Fan Chart
(`plot3_event_forecast_fan.png`)

- Centred on the largest displacement jump in the test split
  (index {largest_idx}, t = {jump_time:.1f} s, dx = {jump_dx:+.5f} m).
- Thin red lines: oracle-v open-loop forecasts starting every {FAN_EVERY} steps.
- Thin blue dashed lines: persistence-v forecasts starting every {FAN_PERSIST} steps.
- Circle / diamond / square markers at h = 10 / 50 / 100 (1 / 5 / 10 s) endpoints.
- Demonstrates how the oracle "sees" the event via future velocity while
  persistence lags; forecast uncertainty fans out with horizon.

## Key Numbers

| Horizon | t (s) | Oracle DxR2 | Persist DxR2 |
|---------|-------|-------------|--------------|
"""

for h in H_ANNOTATE:
    r2_o = curves['oracle']['all'][h - 1]
    r2_p = curves['persistence']['all'][h - 1]
    readme_text += f"| h={h:<5d}| {h*DT:5.1f} | {r2_o:+11.4f} | {r2_p:+12.4f} |\n"

readme_text += f"""
Oracle crossover: h = {crossover_h.get('oracle', 'N/A')} \
({crossover_h.get('oracle', 0) * DT:.1f} s)
Persistence crossover: h = {crossover_h.get('persistence', 'N/A')} \
({crossover_h.get('persistence', 0) * DT:.1f} s)
Events detected: {n_events}
Centroids: A = {centroids[0]:.4f},  B = {centroids[1]:.4f}
"""

(OUT / 'README.md').write_text(readme_text, encoding='utf-8')
print(f"\n  Saved: README.md")


# ===================================================================
#  Console summary
# ===================================================================
elapsed = time.time() - t0_global
print(f"\n{'='*70}")
print("  CONSOLE SUMMARY")
print(f"{'='*70}")
print(f"\n  Plot 3 event: test idx = {largest_idx},  t = {jump_time:.1f} s,  "
      f"dx = {jump_dx:+.5f} m")
print(f"\n  DxR2 at key horizons:")
print(f"  {'h':>5s}  {'t(s)':>6s}  {'Oracle':>10s}  {'Persist':>10s}")
print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}")
for h in H_ANNOTATE:
    r2_o = curves['oracle']['all'][h - 1]
    r2_p = curves['persistence']['all'][h - 1]
    print(f"  {h:5d}  {h*DT:6.1f}  {r2_o:+10.4f}  {r2_p:+10.4f}")

print(f"\n  Event detection: {n_events} transitions")
print(f"  Centroids: A = {centroids[0]:.4f},  B = {centroids[1]:.4f}")
print(f"  Hysteresis thresholds: up = {ev['thresh_up']:.4f},  "
      f"down = {ev['thresh_down']:.4f}")
for mode in ['oracle', 'persistence']:
    ch = crossover_h.get(mode)
    if ch:
        print(f"  {mode} DxR2 crossover: h = {ch} ({ch * DT:.1f} s)")

# Save summary JSON
summary = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'closure_2t_v2_seed_avg',
    'max_horizon': MAX_H,
    'crossover_h': crossover_h,
    'crossover_s': {m: ch * DT if ch else None
                    for m, ch in crossover_h.items()},
    'dxr2': {mode: {str(h): float(curves[mode]['all'][h - 1])
                     for h in H_ANNOTATE}
             for mode in ['oracle', 'persistence']},
    'event_detection': {
        'n_events': n_events,
        'centroids': [float(c) for c in centroids],
        'thresh_up': float(ev['thresh_up']),
        'thresh_down': float(ev['thresh_down']),
    },
    'plot3_event': {
        'test_idx': largest_idx,
        'time_s': jump_time,
        'dx_m': jump_dx,
    },
    'runtime_s': elapsed,
}
with open(OUT / 'summary_top3.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  Saved: summary_top3.json")
print(f"\n  Total runtime: {elapsed:.0f} s")
print(f"  Output directory: {OUT}")

# ===================================================================
#  Interactive display (if backend supports it)
# ===================================================================
try:
    plt.show(block=True)
except Exception:
    pass
