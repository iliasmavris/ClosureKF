"""
Compare VL trajectories across three particle sizes.
Side-by-side figure for appendix: d_p = 3mm, 1cm, 3cm.
"""
import sys, pathlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(r"C:\Users\Workstation 2\Desktop\2026 research\full code old  state space claude code")
VL   = ROOT / "virtual_lab_v1"

# ── paper style (simplified inline) ──────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'figure.dpi': 150,
})

# Okabe-Ito palette
C_PHYS = '#222222'
C_CLOS = '#0072B2'
C_VEL  = '#CC79A7'

# ── load datasets ────────────────────────────────────────────────
configs = [
    {
        'label': r'$d_p = 3\,\mathrm{mm}$  (ER = 0.27)',
        'path': VL / 'datasets' / 'condition_004' / 'x_10hz.csv',
        'dp_mm': 3,
        'x_unit': 'mm',
        'x_scale': 1000,  # m -> mm
    },
    {
        'label': r'$d_p = 1\,\mathrm{cm}$  (ER = 0.25)',
        'path': VL / 'datasets_v2_dp1cm' / 'condition_001' / 'x_10hz.csv',
        'dp_mm': 10,
        'x_unit': 'mm',
        'x_scale': 1000,
    },
    {
        'label': r'$d_p = 3\,\mathrm{cm}$  (ER = 0.10)',
        'path': VL / 'datasets_v2' / 'condition_004' / 'x_10hz.csv',
        'dp_mm': 30,
        'x_unit': 'mm',
        'x_scale': 1000,
    },
]

fig, axes = plt.subplots(3, 2, figsize=(10, 7),
                         gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.28, 'hspace': 0.35})

for i, cfg in enumerate(configs):
    df = pd.read_csv(cfg['path'])
    t = df['timestamp'].values
    x = df['displacement'].values * cfg['x_scale']
    u = df['velocity'].values

    # ── left panel: time series ──────────────────────────────────
    ax_ts = axes[i, 0]
    ax_ts.plot(t, x, color=C_CLOS, lw=0.4, alpha=0.9)
    ax_ts.set_ylabel(f'x [{cfg["x_unit"]}]')
    ax_ts.set_title(cfg['label'], fontsize=9, fontweight='bold', loc='left')

    # show x_std annotation
    x_std = np.std(x)
    ax_ts.text(0.98, 0.92, f'$\\sigma_x$ = {x_std:.2f} {cfg["x_unit"]}',
               transform=ax_ts.transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.9))

    if i == 2:
        ax_ts.set_xlabel('Time [s]')

    # ── right panel: histogram of displacements ──────────────────
    ax_hist = axes[i, 1]
    ax_hist.hist(x, bins=60, color=C_CLOS, alpha=0.7, edgecolor='white', lw=0.3,
                 orientation='horizontal', density=True)
    ax_hist.set_xlabel('Density')
    ax_hist.set_ylim(ax_ts.get_ylim())
    ax_hist.tick_params(labelleft=False)

# panel labels
for idx, ax in enumerate(axes[:, 0]):
    label = chr(ord('a') + idx)
    ax.text(-0.08, 1.05, f'({label})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom')

OUT = VL / 'outputs' / 'particle_size_comparison'
OUT.mkdir(parents=True, exist_ok=True)
outpath = OUT / 'fig_particle_size_comparison.pdf'
fig.savefig(outpath, bbox_inches='tight', dpi=300)
print(f"Saved: {outpath}")

# also save PNG for quick viewing
outpath_png = OUT / 'fig_particle_size_comparison.png'
fig.savefig(outpath_png, bbox_inches='tight', dpi=150)
print(f"Saved: {outpath_png}")
plt.close()
