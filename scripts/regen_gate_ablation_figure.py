#!/usr/bin/env python
"""Regenerate gate_ablation_figure from saved CSV (no retraining)."""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

# Unified bar style
_PLOT_STYLE_DIR = os.path.join(ROOT, 'EMS_v5_cleanroute', 'figures')
sys.path.insert(0, _PLOT_STYLE_DIR)
from plot_style import apply_mpl_style, PALETTE, BAR_KW, bar
apply_mpl_style()

# Legacy paper_style for line styles / figure sizes
sys.path.insert(0, os.path.join(ROOT, 'ems_v1', 'figures'))
from paper_style import LINESTYLES, DOUBLE_COL

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Load saved table
csv_path = os.path.join(ROOT, 'final_lockbox_gate_ablation',
                        'tables', 'gate_ablation_table.csv')
detail_csv = os.path.join(ROOT, 'final_lockbox_gate_ablation',
                          'tables', 'term_selection_detail.csv')
df = pd.read_csv(csv_path)
df_detail = pd.read_csv(detail_csv)

TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']
n_seeds = df['seed'].nunique()

# Panel (a): term selection counts from detail CSV
count_on  = {tn: 0 for tn in TERM_NAMES}
count_off = {tn: 0 for tn in TERM_NAMES}
for _, row in df_detail.iterrows():
    tn = row['term']
    if row['selected_on']:
        count_on[tn] += 1
    if row['selected_off']:
        count_off[tn] += 1

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL[0], 3.0),
                                  gridspec_kw={'width_ratios': [1.0, 1.0],
                                               'wspace': 0.35})

x_pos = np.arange(len(TERM_NAMES))
w = 0.35
bars_on  = [count_on[tn] for tn in TERM_NAMES]
bars_off = [count_off[tn] for tn in TERM_NAMES]

bar(ax_a, x_pos - w/2, bars_on, color_key='closure_blue',
    label='Gate ON', width=w)
bar(ax_a, x_pos + w/2, bars_off, color_key='gate_orange',
    label='Gate OFF', width=w)
ax_a.set_xticks(x_pos)
term_labels = ['$a_1$', '$d_1$', '$d_2$', '$d_3$', '$b_1$', '$b_2$']
ax_a.set_xticklabels(term_labels)
ax_a.set_ylabel('Selection count (out of 3 seeds)')
ax_a.set_ylim(0, n_seeds + 0.5)
ax_a.set_yticks(range(n_seeds + 1))
ax_a.legend(fontsize=7.5)
ax_a.set_title('(a) Term selection frequency', loc='left',
               fontweight='bold', fontsize=9)

# Panel (b): DxR2 vs horizon (3 points from CSV)
horizons = [0.1, 1.0, 2.0]
dxr2_cols = ['dxr2_01s', 'dxr2_1s', 'dxr2_2s']

for mode, color, ls, label in [
    ('physics', PALETTE['physics_grey'], '-', 'Physics only'),
    ('gate_on', PALETTE['closure_blue'], '--', 'Gate ON'),
    ('gate_off', PALETTE['gate_orange'], ':', 'Gate OFF'),
]:
    sub = df[df['mode'] == mode]
    vals = sub[dxr2_cols].values  # (n_seeds, 3)
    means = vals.mean(axis=0)
    stds  = vals.std(axis=0)
    ax_b.errorbar(horizons, means, yerr=stds, color=color, ls=ls,
                  marker='o', markersize=4, capsize=3, label=label, lw=1.5)

ax_b.set_xlabel('Horizon (s)')
ax_b.set_ylabel(r'$R^2_{\Delta x}$')
ax_b.set_xlim(-0.1, 2.3)
ax_b.legend(fontsize=7.5, loc='upper left')
ax_b.set_title('(b) Displacement-increment skill', loc='left',
               fontweight='bold', fontsize=9)

# Save
out_dir = os.path.join(ROOT, 'final_lockbox_gate_ablation', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, 'gate_ablation_figure.pdf')
out_png = os.path.join(out_dir, 'gate_ablation_figure.png')
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
plt.close(fig)
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')
