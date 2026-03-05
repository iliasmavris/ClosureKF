"""
25_qc_condition_plots.py - Per-condition QC plots (velocity + displacement)
=============================================================================
For each of the 24 conditions, produces a 2-panel figure:
  - Top panel: u(t) water velocity vs time
  - Bottom panel: x(t) particle displacement vs time
  - Color-codes free (blue) vs pinned (red) states

Outputs:
  outputs/sweep_v2/qc_plots/condition_XXX_qc.png   (individual)
  outputs/sweep_v2/qc_plots/qc_grid_all.png         (6x4 grid overview)

Usage:
  python 25_qc_condition_plots.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets_v2"
OUT_DIR = ROOT / "outputs" / "sweep_v2" / "qc_plots"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def plot_condition(cond_dir, out_path):
    """Plot 2-panel QC for one condition."""
    cid = cond_dir.name
    truth_path = cond_dir / "truth_states_raw.csv"
    meta_path = cond_dir / "meta.json"

    if not truth_path.exists():
        print(f"  SKIP {cid}: no truth_states_raw.csv")
        return False

    df = pd.read_csv(truth_path)
    with open(meta_path) as f:
        meta = json.load(f)

    spinup = meta.get('config', {}).get('integration', {}).get('spinup_discard', 30.0)
    mask = df['time'] >= spinup
    df = df[mask].reset_index(drop=True)

    t = df['time'].values
    x = df['x'].values
    u_b = df['u_b'].values
    at_pin = df['at_pin'].values

    er = meta.get('event_rate', 0)
    d_p = meta.get('config', {}).get('sphere', {}).get('d_p', 0)
    k_spring = meta.get('config', {}).get('restoring', {}).get('k_spring', 0)
    mu_s = meta.get('config', {}).get('friction', {}).get('mu_s', 0)

    # Downsample for plotting if very long
    step = max(1, len(t) // 20000)
    t_p = t[::step]
    x_p = x[::step]
    u_p = u_b[::step]
    pin_p = at_pin[::step]

    free_mask = pin_p == 0
    pinned_mask = pin_p == 1

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    # Top: velocity
    ax = axes[0]
    ax.plot(t_p, u_p, color='steelblue', lw=0.3, alpha=0.8)
    ax.set_ylabel('u(t) [m/s]')
    ax.set_title(f'{cid}  |  ER={er:.3f}  d_p={d_p*100:.1f}cm  '
                 f'k={k_spring:.1f}  mu_s={mu_s:.3f}')
    ax.grid(True, alpha=0.2)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Bottom: displacement with free/pinned coloring
    ax = axes[1]
    # Plot pinned segments in red, free in blue
    if pinned_mask.any():
        x_pinned = np.where(pinned_mask, x_p, np.nan)
        ax.plot(t_p, x_pinned, color='#CC3333', lw=0.4, alpha=0.6, label='pinned')
    if free_mask.any():
        x_free = np.where(free_mask, x_p, np.nan)
        ax.plot(t_p, x_free, color='#2266AA', lw=0.4, alpha=0.8, label='free')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('x(t) [m]')
    ax.legend(loc='upper right', ncol=2, frameon=False)
    ax.grid(True, alpha=0.2)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_grid(cond_dirs, out_path):
    """Plot 6x4 grid overview of all conditions."""
    n = len(cond_dirs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharex=False)
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, cond_dir in enumerate(cond_dirs):
        ax = axes[i]
        cid = cond_dir.name
        truth_path = cond_dir / "truth_states_raw.csv"
        meta_path = cond_dir / "meta.json"

        if not truth_path.exists():
            ax.text(0.5, 0.5, f'{cid}\nNO DATA', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        df = pd.read_csv(truth_path)
        with open(meta_path) as f:
            meta = json.load(f)

        spinup = meta.get('config', {}).get('integration', {}).get('spinup_discard', 30.0)
        mask = df['time'] >= spinup
        df = df[mask].reset_index(drop=True)

        t = df['time'].values
        x = df['x'].values
        at_pin = df['at_pin'].values
        er = meta.get('event_rate', 0)

        step = max(1, len(t) // 5000)
        t_p = t[::step]
        x_p = x[::step]
        pin_p = at_pin[::step]

        free_mask = pin_p == 0
        pinned_mask = pin_p == 1

        if pinned_mask.any():
            x_pinned = np.where(pinned_mask, x_p, np.nan)
            ax.plot(t_p, x_pinned, color='#CC3333', lw=0.3, alpha=0.5)
        if free_mask.any():
            x_free = np.where(free_mask, x_p, np.nan)
            ax.plot(t_p, x_free, color='#2266AA', lw=0.3, alpha=0.7)

        ax.set_title(f'{cid[-3:]} ER={er:.2f}', fontsize=7, pad=2)
        ax.tick_params(labelsize=5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.15)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Per-Condition Displacement x(t): blue=free, red=pinned', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    print("=" * 60)
    print("PER-CONDITION QC PLOTS")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cond_dirs = sorted(DATASETS.glob("condition_*"))
    print(f"Found {len(cond_dirs)} conditions")

    n_ok = 0
    for cond_dir in cond_dirs:
        out_path = OUT_DIR / f"{cond_dir.name}_qc.png"
        ok = plot_condition(cond_dir, out_path)
        if ok:
            n_ok += 1
            print(f"  {cond_dir.name}: OK")

    print(f"\nIndividual plots: {n_ok}/{len(cond_dirs)}")

    # Grid overview
    if cond_dirs:
        grid_path = OUT_DIR / "qc_grid_all.png"
        plot_grid(cond_dirs, grid_path)
        print(f"Grid plot: {grid_path}")

    print("=" * 60)
    print("QC PLOTS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
