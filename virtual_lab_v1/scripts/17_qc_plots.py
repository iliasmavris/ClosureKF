"""
17_qc_plots.py - QC figures + report for virtual lab Phase 2
=============================================================
Generates 5 specific PDFs:
  1. fig_event_rate_vs_condition.pdf - bar plot of sliding fraction
  2. fig_x_overlay_examples.pdf     - x_10hz(t) for 3 representative conditions
  3. fig_dx_hist.pdf                - histogram of dx at 10Hz across conditions
  4. fig_x_psd_examples.pdf         - Welch PSD of x_10hz for 3 conditions
  5. fig_waiting_excursion_hist.pdf  - waiting times at pin & excursion durations

Plus report_phase2.md and manifest.json.
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
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent


def load_condition_data(cond_dir):
    """Load meta.json and x_10hz.csv for one condition."""
    with open(cond_dir / "meta.json", 'r') as f:
        meta = json.load(f)
    df = pd.read_csv(cond_dir / "x_10hz.csv")
    return meta, df


def load_all_conditions():
    """Load all conditions, sorted by condition_id."""
    datasets_dir = ROOT / "datasets"
    cond_dirs = sorted(datasets_dir.glob("condition_*"))

    all_meta = []
    all_df = []
    for cd in cond_dirs:
        if (cd / "meta.json").exists() and (cd / "x_10hz.csv").exists():
            meta, df = load_condition_data(cd)
            all_meta.append(meta)
            all_df.append(df)

    return all_meta, all_df


def select_representative(all_meta, n=3):
    """Select low/mid/high event rate conditions."""
    ers = [(i, m['event_rate']) for i, m in enumerate(all_meta)]
    ers.sort(key=lambda x: x[1])

    if len(ers) < n:
        return [i for i, _ in ers]

    # Low, mid, high
    indices = [
        ers[0][0],                    # lowest event rate
        ers[len(ers) // 2][0],        # median
        ers[-1][0],                   # highest event rate
    ]
    return indices


def fig_event_rate(all_meta, out_dir):
    """Bar plot of sliding fraction per condition."""
    fig, ax = plt.subplots(figsize=(10, 5))

    cond_ids = [m['condition_id'] for m in all_meta]
    event_rates = [m['event_rate'] for m in all_meta]
    labels = [cid.replace('condition_', 'C') for cid in cond_ids]

    colors = []
    for er in event_rates:
        if er < 0.01 or er > 0.80:
            colors.append('salmon')
        else:
            colors.append('steelblue')

    bars = ax.bar(labels, event_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Sliding fraction (event rate)')
    ax.set_xlabel('Condition')
    ax.set_title('Event Rate vs Condition')
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='min=0.01')
    ax.axhline(0.80, color='red', linestyle='--', alpha=0.5, label='max=0.80')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)

    # Annotate mu_s on each bar
    for bar, meta in zip(bars, all_meta):
        mu_s = meta['config']['friction']['mu_s']
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'mu_s={mu_s:.2f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    path = out_dir / "fig_event_rate_vs_condition.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_x_overlay(all_meta, all_df, out_dir):
    """x_10hz(t) for 3 representative conditions."""
    indices = select_representative(all_meta)
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 3*len(indices)), sharex=True)
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        df = all_df[idx]
        meta = all_meta[idx]
        cid = meta['condition_id']
        er = meta['event_rate']
        t = df['timestamp'].values
        x = df['displacement'].values

        ax.plot(t, x * 1000, linewidth=0.5, color='navy')
        ax.set_ylabel('x [mm]')
        ax.set_title(f'{cid} (event_rate={er:.3f})')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Displacement x(t) - Representative Conditions', fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig_x_overlay_examples.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_dx_hist(all_meta, all_df, out_dir):
    """Histogram of dx at 10Hz across conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for meta, df in zip(all_meta, all_df):
        cid = meta['condition_id'].replace('condition_', 'C')
        x = df['displacement'].values
        dx = np.diff(x) * 1000  # mm
        ax.hist(dx, bins=100, alpha=0.4, label=cid, density=True)

    ax.set_xlabel('dx [mm / 0.1s]')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of 10Hz Displacement Increments')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fig_dx_hist.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_x_psd(all_meta, all_df, out_dir):
    """Welch PSD of x_10hz for 3 representative conditions."""
    indices = select_representative(all_meta)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['navy', 'firebrick', 'forestgreen']
    for ci, idx in enumerate(indices):
        df = all_df[idx]
        meta = all_meta[idx]
        cid = meta['condition_id']
        er = meta['event_rate']
        x = df['displacement'].values

        f, psd = welch(x, fs=10.0, nperseg=min(256, len(x)))
        ax.semilogy(f, psd, color=colors[ci % len(colors)],
                     label=f'{cid} (er={er:.2f})')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [m^2/Hz]')
    ax.set_title('Power Spectral Density of x(t) at 10Hz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)

    fig.tight_layout()
    path = out_dir / "fig_x_psd_examples.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_waiting_excursion(all_meta, out_dir):
    """Histogram of waiting times at pin and excursion durations across conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_meta), 1)))

    has_data_wt = False
    has_data_et = False

    for ci, meta in enumerate(all_meta):
        cid = meta['condition_id'].replace('condition_', 'C')
        ps = meta.get('pin_stats', {})
        color = colors[ci % len(colors)]

        # Load truth_states_raw to compute distributions directly
        cond_dir = ROOT / "datasets" / meta['condition_id']
        truth_path = cond_dir / "truth_states_raw.csv"
        if truth_path.exists():
            df_truth = pd.read_csv(truth_path)
            at_pin = df_truth['at_pin'].values
            dt_sim = meta['config']['integration']['dt_sim']

            # Compute run lengths
            wt_list = []
            et_list = []
            if len(at_pin) > 0:
                current = at_pin[0]
                run_start = 0
                for j in range(1, len(at_pin)):
                    if at_pin[j] != current:
                        dur = (j - run_start) * dt_sim
                        if current == 1:
                            wt_list.append(dur)
                        else:
                            et_list.append(dur)
                        run_start = j
                        current = at_pin[j]

            if len(wt_list) > 1:
                ax1.hist(wt_list, bins=30, alpha=0.4, label=cid,
                         color=color, density=True)
                has_data_wt = True
            if len(et_list) > 1:
                ax2.hist(et_list, bins=30, alpha=0.4, label=cid,
                         color=color, density=True)
                has_data_et = True

    ax1.set_xlabel('Waiting time at pin [s]')
    ax1.set_ylabel('Density')
    ax1.set_title('Waiting Time Distribution (at pin)')
    if has_data_wt:
        ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Excursion duration [s]')
    ax2.set_ylabel('Density')
    ax2.set_title('Excursion Duration Distribution')
    if has_data_et:
        ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Pin Statistics Across Conditions', fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig_waiting_excursion_hist.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(all_meta, out_dir):
    """Write report_phase2.md."""
    lines = []
    lines.append("# Phase 2 QC Report: Virtual Lab Sphere Truth Generator")
    lines.append("")
    lines.append("## Condition Summary")
    lines.append("")
    lines.append("| Condition | mu_s | mu_k | C_Du | Seed | Event Rate | x_std [mm] | Pass |")
    lines.append("|-----------|------|------|------|------|------------|------------|------|")

    for m in all_meta:
        cid = m['condition_id']
        mu_s = m['config']['friction']['mu_s']
        mu_k = m['config']['friction']['mu_k']
        C_Du = m['config']['added_mass']['C_Du']
        seed = m['seed']
        er = m['event_rate']
        x_std = m['stats_10hz']['x_std'] * 1000
        passed = 0.01 <= er <= 0.80 and m['stats_10hz']['x_std'] > 0
        mark = "PASS" if passed else "FAIL"
        lines.append(f"| {cid} | {mu_s:.2f} | {mu_k:.2f} | {C_Du:.1f} | "
                      f"{seed} | {er:.3f} | {x_std:.3f} | {mark} |")

    lines.append("")
    lines.append("## Event Statistics")
    lines.append("")
    ers = [m['event_rate'] for m in all_meta]
    lines.append(f"- Mean event rate: {np.mean(ers):.3f}")
    lines.append(f"- Range: [{min(ers):.3f}, {max(ers):.3f}]")
    lines.append(f"- Conditions in target [0.01, 0.80]: "
                 f"{sum(1 for e in ers if 0.01 <= e <= 0.80)}/{len(ers)}")

    lines.append("")
    lines.append("## Non-Circularity Statement")
    lines.append("")
    lines.append("The sphere truth model is **structurally different** from the reduced")
    lines.append("EKF/closure model used in the manuscript:")
    lines.append("")
    lines.append("- **Drag:** Blended Stokes + form drag (not relu threshold exceedance)")
    lines.append("- **Friction:** Discontinuous Coulomb stick-slip (not exponential decay)")
    lines.append("- **No** rho*u relaxation term")
    lines.append("- **No** closure terms (b2*du, d2*v|u|)")
    lines.append("- **No** Kalman filter states or innovation structure")
    lines.append("")
    lines.append("The truth model uses: F_D = 3*pi*mu*d*w + 0.5*rho*C_D*A*|w|*w (blended),")
    lines.append("F_fric = mu_k*W_sub*sign(v_p) (sliding) or F_fric = F_drive (stuck),")
    lines.append("integrated via RK4 at dt=0.005s with OU noise forcing.")

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `fig_event_rate_vs_condition.pdf` - Sliding fraction per condition")
    lines.append("- `fig_x_overlay_examples.pdf` - x(t) for representative conditions")
    lines.append("- `fig_dx_hist.pdf` - dx histogram across all conditions")
    lines.append("- `fig_x_psd_examples.pdf` - PSD of x(t) for representative conditions")
    lines.append("- `fig_waiting_excursion_hist.pdf` - Waiting time at pin & excursion durations")

    path = out_dir / "report_phase2.md"
    path.write_text('\n'.join(lines))
    return path


def write_manifest(all_meta, figures, report_path, out_dir):
    """Write manifest.json."""
    manifest = {
        'n_conditions': len(all_meta),
        'n_pass': sum(1 for m in all_meta
                      if 0.01 <= m['event_rate'] <= 0.80
                      and m['stats_10hz']['x_std'] > 0),
        'figures': [str(f.name) for f in figures],
        'report': str(report_path.name),
        'conditions': [m['condition_id'] for m in all_meta],
    }
    path = out_dir / "manifest.json"
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    return path


def main():
    print("=" * 60)
    print("QC PLOTS + REPORT - Phase 2 Virtual Lab")
    print("=" * 60)

    all_meta, all_df = load_all_conditions()
    print(f"\nLoaded {len(all_meta)} conditions")

    if len(all_meta) == 0:
        print("ERROR: No condition data found in datasets/")
        sys.exit(1)

    out_dir = ROOT / "outputs" / "qc_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    figures = []

    print("\n1/5 Event rate bar plot...")
    figures.append(fig_event_rate(all_meta, out_dir))

    print("2/5 x(t) overlay examples...")
    figures.append(fig_x_overlay(all_meta, all_df, out_dir))

    print("3/5 dx histogram...")
    figures.append(fig_dx_hist(all_meta, all_df, out_dir))

    print("4/5 PSD examples...")
    figures.append(fig_x_psd(all_meta, all_df, out_dir))

    print("5/5 Waiting / excursion histograms...")
    figures.append(fig_waiting_excursion(all_meta, out_dir))

    # Write report
    print("\nWriting report...")
    report_path = write_report(all_meta, out_dir)

    # Write manifest
    manifest_path = write_manifest(all_meta, figures, report_path, out_dir)

    print(f"\nOutputs in: {out_dir}")
    for f in figures:
        print(f"  {f.name}")
    print(f"  {report_path.name}")
    print(f"  {manifest_path.name}")

    print("\n" + "=" * 60)
    print("QC COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
