"""
Generate fig_failure_map.pdf: Regime x horizon failure map.

Reads frozen event_skill_table.csv from v11.1 (seed 1) and visualises
where the physics-only and closure models fail (skill < 0) or succeed
across horizons h=1..50 and regimes (event / nonevent / full).

Output: final_lockbox_v11_1_alpha_fix/seed1/figures/fig_failure_map.pdf

NO retraining. Read-only from existing CSV.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TABLE_CSV = (ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "tables"
             / "event_skill_table.csv")
FIG_DIR = ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
print("Loading event_skill_table.csv...")
df = pd.read_csv(TABLE_CSV)
print(f"  Rows: {len(df)}, models: {df['model'].unique()}, "
      f"subsets: {df['subset'].unique()}")

# ---- Compute DxR2 from rmse columns ----
# DxR2 = 1 - (rmse_dx / rmse_baseline)^2
# But we have rmse_skill which appears to be 1 - rmse_dx/rmse_baseline (linear)
# Let me compute it properly: DxR2 = 1 - var(pred_error)/var(baseline_error)
# = 1 - rmse_dx^2 / rmse_baseline^2
df['dxr2'] = 1.0 - (df['rmse_dx'] ** 2) / (df['rmse_baseline'] ** 2)

# ---- Create figure: 2-panel line plot + heatmap ----
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

# Color palette
colors = {
    'full': '#333333',
    'event': '#D62728',
    'nonevent': '#1F77B4',
}
ls = {'physics': '--', 'closure': '-'}
lw = {'physics': 1.2, 'closure': 1.8}

# ========================================
# Panel (a): DxR2 vs horizon by regime
# ========================================
ax_a = fig.add_subplot(gs[0, :])  # full width top

for subset in ['full', 'event', 'nonevent']:
    for model in ['physics', 'closure']:
        mask = (df['model'] == model) & (df['subset'] == subset)
        sub = df[mask].sort_values('h')
        label = f"{subset} ({model})" if model == 'closure' else f"_{subset} ({model})"
        ax_a.plot(sub['h'] * 0.1, sub['dxr2'],
                  color=colors[subset], linestyle=ls[model], linewidth=lw[model],
                  label=label if model == 'closure' else None)
        if model == 'physics':
            ax_a.plot(sub['h'] * 0.1, sub['dxr2'],
                      color=colors[subset], linestyle=ls[model], linewidth=lw[model],
                      alpha=0.5)

# Add physics legend entries manually
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=colors['full'], ls='-', lw=1.8, label='Full (closure)'),
    Line2D([0], [0], color=colors['event'], ls='-', lw=1.8, label='Event (closure)'),
    Line2D([0], [0], color=colors['nonevent'], ls='-', lw=1.8, label='Non-event (closure)'),
    Line2D([0], [0], color='0.5', ls='--', lw=1.2, label='Physics-only (all regimes)'),
]
ax_a.legend(handles=handles, fontsize=8, loc='lower right', ncol=2)

ax_a.axhline(0, color='0.3', linewidth=0.8, linestyle=':')
ax_a.set_xlabel('Forecast horizon (s)')
ax_a.set_ylabel(r'$\Delta x\, R^2$')
ax_a.set_title(r'(a) Displacement-increment skill by regime and horizon')
ax_a.set_xlim([0.1, 5.0])
ax_a.grid(True, alpha=0.3)

# Shade failure zone
ax_a.fill_between([0, 5.0], -1.5, 0, alpha=0.04, color='red')
ax_a.text(4.9, -0.05, 'Skill < 0\n(worse than baseline)',
          ha='right', va='top', fontsize=7, color='red', alpha=0.7)

# ========================================
# Panel (b): Heatmap - physics improvement (closure - physics)
# ========================================
ax_b = fig.add_subplot(gs[1, 0])

# Pivot: DxR2 gain (closure - physics) per subset x horizon
horizons = sorted(df['h'].unique())
subsets_ordered = ['event', 'full', 'nonevent']

# Build gain matrix
gain_matrix = np.zeros((len(subsets_ordered), len(horizons)))
for i, subset in enumerate(subsets_ordered):
    for j, h in enumerate(horizons):
        phys_val = df[(df['model'] == 'physics') & (df['subset'] == subset)
                      & (df['h'] == h)]['dxr2']
        clos_val = df[(df['model'] == 'closure') & (df['subset'] == subset)
                      & (df['h'] == h)]['dxr2']
        if len(phys_val) > 0 and len(clos_val) > 0:
            gain_matrix[i, j] = clos_val.values[0] - phys_val.values[0]

# Use diverging colormap centered on 0
vmax = max(abs(gain_matrix.min()), abs(gain_matrix.max()))
vmax = min(vmax, 0.3)  # cap for readability
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

h_seconds = [h * 0.1 for h in horizons]
im = ax_b.pcolormesh(
    np.arange(len(horizons) + 1) - 0.5,
    np.arange(len(subsets_ordered) + 1) - 0.5,
    gain_matrix,
    cmap='RdBu_r', norm=norm, shading='flat'
)

# Tick labels: show every 5th horizon
tick_pos = list(range(0, len(horizons), 5))
ax_b.set_xticks(tick_pos)
ax_b.set_xticklabels([f'{horizons[i]*0.1:.1f}' for i in tick_pos], fontsize=7)
ax_b.set_yticks(range(len(subsets_ordered)))
ax_b.set_yticklabels(['Event', 'Full', 'Non-event'], fontsize=8)
ax_b.set_xlabel('Forecast horizon (s)')
ax_b.set_title(r'(b) $\Delta x\, R^2$ gain (closure $-$ physics)')

cb = plt.colorbar(im, ax=ax_b, shrink=0.8, pad=0.02)
cb.set_label(r'$\Delta x\, R^2$ gain', fontsize=8)

# ========================================
# Panel (c): Closure DxR2 heatmap (absolute)
# ========================================
ax_c = fig.add_subplot(gs[1, 1])

# Build closure DxR2 matrix
clos_matrix = np.zeros((len(subsets_ordered), len(horizons)))
for i, subset in enumerate(subsets_ordered):
    for j, h in enumerate(horizons):
        val = df[(df['model'] == 'closure') & (df['subset'] == subset)
                 & (df['h'] == h)]['dxr2']
        if len(val) > 0:
            clos_matrix[i, j] = val.values[0]

# Diverging colormap for DxR2 (negative = failure)
vabs = max(abs(clos_matrix.min()), abs(clos_matrix.max()))
vabs = min(vabs, 0.5)
norm_c = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

im_c = ax_c.pcolormesh(
    np.arange(len(horizons) + 1) - 0.5,
    np.arange(len(subsets_ordered) + 1) - 0.5,
    clos_matrix,
    cmap='RdYlGn', norm=norm_c, shading='flat'
)

ax_c.set_xticks(tick_pos)
ax_c.set_xticklabels([f'{horizons[i]*0.1:.1f}' for i in tick_pos], fontsize=7)
ax_c.set_yticks(range(len(subsets_ordered)))
ax_c.set_yticklabels(['Event', 'Full', 'Non-event'], fontsize=8)
ax_c.set_xlabel('Forecast horizon (s)')
ax_c.set_title(r'(c) Closure $\Delta x\, R^2$ (absolute)')

cb_c = plt.colorbar(im_c, ax=ax_c, shrink=0.8, pad=0.02)
cb_c.set_label(r'$\Delta x\, R^2$', fontsize=8)

# ---- Save ----
out_pdf = FIG_DIR / "fig_failure_map.pdf"
out_png = FIG_DIR / "fig_failure_map.png"
fig.savefig(str(out_pdf), dpi=300, bbox_inches='tight')
fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")

# ---- Summary table ----
print("\n--- Failure Map Summary ---")
print(f"{'Subset':<12} {'h@fail->skill':<20} {'h@best_gain':<20} "
      f"{'DxR2@1s_phys':<15} {'DxR2@1s_clos':<15}")
print("-" * 82)

for subset in subsets_ordered:
    phys_h10 = df[(df['model'] == 'physics') & (df['subset'] == subset)
                  & (df['h'] == 10)]
    clos_h10 = df[(df['model'] == 'closure') & (df['subset'] == subset)
                  & (df['h'] == 10)]
    phys_dxr2_10 = phys_h10['dxr2'].values[0] if len(phys_h10) > 0 else float('nan')
    clos_dxr2_10 = clos_h10['dxr2'].values[0] if len(clos_h10) > 0 else float('nan')

    # Find horizon where physics skill first goes positive
    phys_sub = df[(df['model'] == 'physics') & (df['subset'] == subset)].sort_values('h')
    phys_dxr2 = phys_sub['dxr2'].values
    phys_h = phys_sub['h'].values
    first_pos = np.argmax(phys_dxr2 > 0) if np.any(phys_dxr2 > 0) else -1

    # Find horizon of maximum closure gain
    clos_sub = df[(df['model'] == 'closure') & (df['subset'] == subset)].sort_values('h')
    gains = clos_sub['dxr2'].values - phys_dxr2
    best_gain_idx = np.argmax(gains)

    first_str = (f"h={phys_h[first_pos]} ({phys_h[first_pos]*0.1:.1f}s)"
                 if first_pos >= 0 else "never")
    best_str = (f"h={phys_h[best_gain_idx]} ({phys_h[best_gain_idx]*0.1:.1f}s), "
                f"gain={gains[best_gain_idx]:+.3f}")

    print(f"{subset:<12} {first_str:<20} {best_str:<20} "
          f"{phys_dxr2_10:<15.4f} {clos_dxr2_10:<15.4f}")

# Key finding: non-event DxR2 is always negative
nonevent_clos = df[(df['model'] == 'closure') & (df['subset'] == 'nonevent')]
nonevent_all_neg = (nonevent_clos['dxr2'] < 0).all()
print(f"\nNon-event closure DxR2 < 0 at ALL horizons: {nonevent_all_neg}")
if nonevent_all_neg:
    worst = nonevent_clos.loc[nonevent_clos['dxr2'].idxmin()]
    print(f"  Worst: h={int(worst['h'])} ({worst['h']*0.1:.1f}s), "
          f"DxR2={worst['dxr2']:.4f}")

print("\nDone.")
