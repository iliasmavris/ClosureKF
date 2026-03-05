"""
Phase 4 plots: term frequency, oracle gap ratio, coefficient vs event rate,
and report.

Usage:  python -u virtual_lab_v1/scripts/24_phase4_plots.py
Input:  virtual_lab_v1/outputs/phase4_discovery/discovery_summary.csv
Output: virtual_lab_v1/outputs/phase4_discovery/  (3 PDFs + report)
"""

import sys, json, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
VL   = ROOT / "virtual_lab_v1"
OUT  = VL / "outputs" / "phase4_discovery"

TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']
ER_BINS = [(0, 0.15, 'low ER'), (0.15, 0.35, 'mid ER'), (0.35, 1.0, 'high ER')]

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ==========================================================================
#  LOAD DATA
# ==========================================================================

GAP_EPS = 1e-12

def load_data():
    csv_path = OUT / "discovery_summary.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 23_run_discovery_v2.py first.")
        return None
    df = pd.read_csv(csv_path)

    # Recompute gap_ratio with correct sign check:
    # gap_ratio is only meaningful when MSE_phys > MSE_oracle (oracle improves)
    for i, row in df.iterrows():
        mse_p = row.get('MSE_phys_oracle')
        mse_o = row.get('MSE_oracle_oracle')
        mse_d = row.get('MSE_disc')
        if pd.notna(mse_p) and pd.notna(mse_o) and pd.notna(mse_d):
            denom = mse_p - mse_o
            if denom <= GAP_EPS:
                df.at[i, 'gap_ratio'] = np.nan
                df.at[i, 'gap_status'] = 'no_residual'
            else:
                df.at[i, 'gap_ratio'] = (mse_d - mse_o) / denom
                df.at[i, 'gap_status'] = 'ok'

    df_ok = df[df['status'] == 'ok'].copy()
    print(f"Loaded {len(df)} conditions, {len(df_ok)} OK")

    n_valid_gap = df_ok['gap_status'].eq('ok').sum()
    n_no_res = df_ok['gap_status'].eq('no_residual').sum()
    print(f"  gap_ratio valid: {n_valid_gap}, no_residual: {n_no_res}")
    return df, df_ok


def parse_terms(term_str):
    """Parse 'b2+d2' -> ['b2','d2']."""
    if pd.isna(term_str) or term_str == '':
        return []
    return [t.strip() for t in str(term_str).split('+') if t.strip()]


def assign_er_bin(er):
    for lo, hi, label in ER_BINS:
        if lo <= er < hi:
            return label
    return ER_BINS[-1][2]


# ==========================================================================
#  FIG 1: TERM FREQUENCY
# ==========================================================================

def fig_term_frequency(df_ok):
    """Bar plot of selection frequency per term, split by ER bins."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall
    ax = axes[0]
    counts = Counter()
    n_total = len(df_ok)
    for _, row in df_ok.iterrows():
        for t in parse_terms(row['selected_terms']):
            counts[t] += 1
    freqs = [counts.get(tn, 0) / max(n_total, 1) for tn in TERM_NAMES]
    colors = ['steelblue' if tn in ('b2', 'd2') else 'silver' for tn in TERM_NAMES]
    ax.bar(TERM_NAMES, freqs, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Selection frequency')
    ax.set_title(f'Overall (n={n_total})')
    ax.set_ylim(0, 1.05)
    for i, f in enumerate(freqs):
        ax.text(i, f + 0.02, f'{f:.0%}', ha='center', fontsize=9)

    # By ER bin
    ax = axes[1]
    df_ok = df_ok.copy()
    df_ok['er_bin'] = df_ok['event_rate'].apply(assign_er_bin)
    bin_labels = [b[2] for b in ER_BINS]
    x = np.arange(len(TERM_NAMES))
    width = 0.25
    bin_colors = ['#4c72b0', '#55a868', '#c44e52']

    for i, bl in enumerate(bin_labels):
        sub = df_ok[df_ok['er_bin'] == bl]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        cnts = Counter()
        for _, row in sub.iterrows():
            for t in parse_terms(row['selected_terms']):
                cnts[t] += 1
        freqs_bin = [cnts.get(tn, 0) / n_sub for tn in TERM_NAMES]
        ax.bar(x + i * width, freqs_bin, width, label=f'{bl} (n={n_sub})',
               color=bin_colors[i], edgecolor='k', linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(TERM_NAMES)
    ax.set_ylabel('Selection frequency')
    ax.set_title('By event-rate bin')
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    path = OUT / "fig_term_frequency.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"  Wrote {path}")


# ==========================================================================
#  FIG 2: ORACLE GAP RATIO
# ==========================================================================

def fig_oracle_gap_ratio(df_ok):
    """Gap ratio distribution, split by ER bins and regime.
    Uses symlog y-scale and annotates counts for robustness."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df_ok = df_ok.copy()
    df_ok['er_bin'] = df_ok['event_rate'].apply(assign_er_bin)

    # Count categories
    n_ok_gap = df_ok['gap_status'].eq('ok').sum()
    n_no_res = df_ok['gap_status'].ne('ok').sum()
    inter = df_ok[df_ok['oracle_regime'] == 'intermittent']
    cont  = df_ok[df_ok['oracle_regime'] == 'continuous']

    def valid_gap(sub):
        return sub[sub['gap_status'] == 'ok'].copy()

    inter_v = valid_gap(inter)
    cont_v  = valid_gap(cont)
    no_res  = df_ok[df_ok['gap_status'] != 'ok']

    # Left: scatter gap_ratio vs event_rate with symlog scale
    ax = axes[0]
    if len(inter_v) > 0:
        ax.scatter(inter_v['event_rate'], inter_v['gap_ratio'],
                   c='steelblue', marker='o', s=60, edgecolors='k',
                   linewidth=0.5, label=f'Intermittent (n={len(inter_v)})', zorder=3)
    if len(cont_v) > 0:
        ax.scatter(cont_v['event_rate'], cont_v['gap_ratio'],
                   c='coral', marker='^', s=60, edgecolors='k',
                   linewidth=0.5, label=f'Continuous (n={len(cont_v)})', zorder=3)
    if len(no_res) > 0:
        ax.scatter(no_res['event_rate'], [0]*len(no_res),
                   c='gray', marker='x', s=40, label=f'No residual (n={len(no_res)})',
                   zorder=2)

    ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='Physics-only level')
    ax.axhline(0.0, color='green', ls='--', alpha=0.5, label='Oracle ceiling')
    ax.axvline(0.50, color='gray', ls=':', alpha=0.4)
    ax.set_yscale('symlog', linthresh=2.0)
    ax.set_xlabel('Event rate')
    ax.set_ylabel('Oracle gap ratio (symlog)')
    ax.set_title('Discovery vs Oracle ceiling')
    ax.legend(fontsize=8, loc='upper left')

    # Annotate counts
    ax.text(0.98, 0.02, f'valid={n_ok_gap}  no_resid={n_no_res}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    # Right: box/strip by ER bin (intermittent only) with symlog
    ax = axes[1]
    bin_labels = [b[2] for b in ER_BINS]
    bin_data = []
    for bl in bin_labels:
        sub = inter_v[inter_v['er_bin'] == bl]
        bin_data.append(sub['gap_ratio'].dropna().values)

    for i, (bl, data) in enumerate(zip(bin_labels, bin_data)):
        if len(data) == 0:
            continue
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(data))
        ax.scatter([i + jitter[j] for j in range(len(data))], data,
                   c='steelblue', s=40, alpha=0.7, edgecolors='k', linewidth=0.3)
        ax.plot([i - 0.2, i + 0.2], [np.median(data)]*2,
                color='red', linewidth=2, zorder=5)
        ax.text(i, ax.get_ylim()[0], f'n={len(data)}', ha='center',
                va='bottom', fontsize=8)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.axhline(1.0, color='red', ls='--', alpha=0.3)
    ax.axhline(0.0, color='green', ls='--', alpha=0.3)
    ax.set_yscale('symlog', linthresh=2.0)
    ax.set_ylabel('Oracle gap ratio (symlog)')
    ax.set_title('Intermittent conditions by ER bin')

    fig.tight_layout()
    path = OUT / "fig_oracle_gap_ratio.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"  Wrote {path}")


# ==========================================================================
#  FIG 3: COEFFICIENTS VS EVENT RATE
# ==========================================================================

def fig_coeff_vs_eventrate(df_ok):
    """Scatter of closure coefficients vs ER, only where term selected."""
    coeff_cols = {
        'a1': 's2_a1', 'b1': 's2_b1', 'b2': 's2_b2',
        'd1': 's2_d1', 'd2': 's2_d2', 'd3': 's2_d3',
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, tn in enumerate(TERM_NAMES):
        ax = axes[i]
        col = coeff_cols[tn]

        # Conditions where this term was selected
        mask = df_ok['selected_terms'].apply(
            lambda s: tn in parse_terms(s) if pd.notna(s) else False)
        sel = df_ok[mask]
        not_sel = df_ok[~mask]

        if len(not_sel) > 0:
            ax.scatter(not_sel['event_rate'], not_sel[col],
                       c='silver', marker='o', s=30, alpha=0.5,
                       edgecolors='gray', linewidth=0.3, label='Not selected')
        if len(sel) > 0:
            inter_sel = sel[sel['oracle_regime'] == 'intermittent']
            cont_sel  = sel[sel['oracle_regime'] == 'continuous']
            if len(inter_sel) > 0:
                ax.scatter(inter_sel['event_rate'], inter_sel[col],
                           c='steelblue', marker='o', s=50, edgecolors='k',
                           linewidth=0.5, label=f'Intermittent (n={len(inter_sel)})')
            if len(cont_sel) > 0:
                ax.scatter(cont_sel['event_rate'], cont_sel[col],
                           c='coral', marker='^', s=50, edgecolors='k',
                           linewidth=0.5, label=f'Continuous (n={len(cont_sel)})')

        ax.axhline(0, color='gray', ls='-', alpha=0.3)
        ax.set_title(tn)
        ax.set_xlabel('Event rate')
        ax.set_ylabel(f'Coefficient ({tn})')
        ax.legend(fontsize=7)

    fig.suptitle('Discovered closure coefficients vs event rate', fontsize=13)
    fig.tight_layout()
    path = OUT / "fig_coeff_vs_eventrate.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"  Wrote {path}")


# ==========================================================================
#  REPORT
# ==========================================================================

def write_report(df_all, df_ok):
    lines = ["# Phase 4: Discovery Pipeline Report", ""]
    n_all = len(df_all)
    n_ok  = len(df_ok)
    lines.append(f"**Conditions:** {n_all} total, {n_ok} OK, "
                 f"{n_all - n_ok} failed/skipped")
    lines.append("")

    if n_ok == 0:
        lines.append("No OK conditions. Nothing to report.")
        path = OUT / "report_phase4.md"
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        return

    # Term frequency
    lines.append("## Term selection frequency")
    lines.append("")
    counts_all = Counter()
    for _, row in df_ok.iterrows():
        for t in parse_terms(row['selected_terms']):
            counts_all[t] += 1

    lines.append(f"| Term | Overall ({n_ok}) | Intermittent | Continuous |")
    lines.append("|------|---------|-------------|-----------|")

    inter = df_ok[df_ok['oracle_regime'] == 'intermittent']
    cont  = df_ok[df_ok['oracle_regime'] == 'continuous']
    counts_inter = Counter()
    counts_cont  = Counter()
    for _, row in inter.iterrows():
        for t in parse_terms(row['selected_terms']):
            counts_inter[t] += 1
    for _, row in cont.iterrows():
        for t in parse_terms(row['selected_terms']):
            counts_cont[t] += 1

    for tn in TERM_NAMES:
        c_a = counts_all.get(tn, 0)
        c_i = counts_inter.get(tn, 0)
        c_c = counts_cont.get(tn, 0)
        n_i = len(inter); n_c = len(cont)
        lines.append(
            f"| {tn} | {c_a}/{n_ok} ({100*c_a/n_ok:.0f}%) | "
            f"{c_i}/{n_i} ({100*c_i/n_i:.0f}%) | "
            f"{c_c}/{n_c} ({100*c_c/n_c:.0f}%) |")
    lines.append("")

    # Most common term set in intermittent
    if len(inter) > 0:
        set_counts = Counter()
        for _, row in inter.iterrows():
            ts = tuple(sorted(parse_terms(row['selected_terms'])))
            set_counts[ts] += 1
        most_common = set_counts.most_common(3)
        lines.append("**Most common term sets (intermittent):**")
        for ts, cnt in most_common:
            lines.append(f"- {'+'.join(ts) if ts else '(none)'}: "
                         f"{cnt}/{len(inter)} ({100*cnt/len(inter):.0f}%)")
        lines.append("")

    # Oracle gap ratio
    lines.append("## Oracle gap ratio")
    lines.append("")

    def gap_stats(sub, label):
        valid = sub[sub['gap_status'] == 'ok']['gap_ratio'].dropna()
        if len(valid) == 0:
            return f"**{label}:** no valid gap ratios"
        return (f"**{label}** (n={len(valid)}): "
                f"median={valid.median():.3f}, "
                f"mean={valid.mean():.3f}, "
                f"IQR=[{valid.quantile(0.25):.3f}, {valid.quantile(0.75):.3f}]")

    lines.append(gap_stats(df_ok, "All OK"))
    lines.append(gap_stats(inter, "Intermittent"))
    lines.append(gap_stats(cont, "Continuous"))
    lines.append("")

    lines.append("**Interpretation:** gap_ratio=0 means discovery matches oracle "
                 "ceiling; gap_ratio=1 means no improvement over physics-only; "
                 "gap_ratio>1 means discovery is worse than physics-only.")
    lines.append("")

    # Continuous regime explanation
    lines.append("## Continuous regime")
    lines.append("")
    lines.append("Continuous-regime conditions (ER >= 0.50) have near-zero "
                 "residual energy: the particle is mostly sliding, so the oracle "
                 "ceiling is close to the physics-only fit. Ridge alpha is typically "
                 "large (heavy regularization), and gains are near zero. The discovery "
                 "pipeline similarly finds little to improve in these conditions.")
    lines.append("")

    # Per-condition table
    lines.append("## Per-condition results")
    lines.append("")
    lines.append("| Condition | ER | Regime | Selected | Gap ratio | ACF1_phys | ACF1_clos | Oracle gain |")
    lines.append("|-----------|-----|--------|----------|-----------|-----------|-----------|-------------|")

    for _, row in df_ok.sort_values('event_rate').iterrows():
        cid = row['condition_id']
        er = row['event_rate']
        regime = row.get('oracle_regime', '')
        sel = row.get('selected_terms', '')
        gr = row.get('gap_ratio')
        gr_str = f"{gr:.3f}" if pd.notna(gr) else "N/A"
        pa = row.get('phys_acf1')
        pa_str = f"{pa:.3f}" if pd.notna(pa) else "N/A"
        ca = row.get('clos_acf1')
        ca_str = f"{ca:.3f}" if pd.notna(ca) else "N/A"
        og = row.get('oracle_gain')
        og_str = f"{og:.3f}" if pd.notna(og) else "N/A"
        lines.append(f"| {cid} | {er:.3f} | {regime} | {sel} | "
                     f"{gr_str} | {pa_str} | {ca_str} | {og_str} |")

    lines.append("")
    path = OUT / "report_phase4.md"
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Wrote {path}")


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    result = load_data()
    if result is None:
        return
    df_all, df_ok = result

    print("\nGenerating plots...")
    fig_term_frequency(df_ok)
    fig_oracle_gap_ratio(df_ok)
    fig_coeff_vs_eventrate(df_ok)
    write_report(df_all, df_ok)
    print("\nDone.")


if __name__ == "__main__":
    main()
