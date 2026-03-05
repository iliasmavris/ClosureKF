"""
22_qc_summary_v2.py - Phase 3 QC summary plots + report
=========================================================
Reads sweep_summary_v2.csv + oracle_summary_v2.csv from outputs/sweep_v2/
Generates:
  fig_eventrate_hist.pdf           - histogram of ER with regime boundary
  fig_oracle_gain_vs_eventrate.pdf - scatter: intermittent vs continuous regimes
  fig_condition_axes.pdf           - distributions of mu_s_final, k_spring, d_p, forcing
  report_phase3.md                 - quantitative summary

Usage:
  python 22_qc_summary_v2.py
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
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "sweep_v2"

ER_INTERMITTENT_MAX = 0.50  # must match 18_oracle_eval.py

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_data():
    """Load sweep summary and oracle summary."""
    df_sweep = pd.read_csv(OUT_DIR / "sweep_summary_v2.csv")
    oracle_path = OUT_DIR / "oracle_summary_v2.csv"
    if oracle_path.exists():
        df_oracle = pd.read_csv(oracle_path)
    else:
        df_oracle = pd.DataFrame()
    return df_sweep, df_oracle


def fig_eventrate_hist(df):
    """Histogram of event rates with target band and regime boundary."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ers = df['event_rate'].values

    # Color bars by regime
    bins = np.linspace(0, max(ers) * 1.1 + 0.02, 20)
    n_int = np.sum(ers < ER_INTERMITTENT_MAX)
    n_cont = np.sum(ers >= ER_INTERMITTENT_MAX)

    ax.hist(ers[ers < ER_INTERMITTENT_MAX], bins=bins,
            color='steelblue', edgecolor='white', alpha=0.85,
            label=f'Intermittent (n={n_int})')
    ax.hist(ers[ers >= ER_INTERMITTENT_MAX], bins=bins,
            color='coral', edgecolor='white', alpha=0.85,
            label=f'Continuous (n={n_cont})')

    ax.axvspan(0.05, 0.40, color='green', alpha=0.10, label='Target [0.05, 0.40]')
    ax.axvline(ER_INTERMITTENT_MAX, color='k', ls='--', lw=1.0,
               label=f'Regime boundary ({ER_INTERMITTENT_MAX})')
    ax.axvline(np.median(ers), color='red', ls=':', lw=1.2,
               label=f'Median={np.median(ers):.3f}')

    ax.set_xlabel('Event rate (sliding fraction)')
    ax.set_ylabel('Count')
    ax.set_title('Event Rate Distribution (Phase 3)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, max(ers) * 1.1 + 0.02)
    path = OUT_DIR / "fig_eventrate_hist.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def fig_oracle_gain_vs_er(df_sweep, df_oracle):
    """
    Scatter of oracle gain vs event rate, clearly separating:
      - intermittent regime (filled circles)
      - continuous regime (open triangles)
      - degenerate (gray x)
      - unstable (red border)
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Merge sweep + oracle on condition_id
    df = df_sweep.merge(
        df_oracle[[c for c in ['condition_id', 'status', 'gain_oracle',
                    'oracle_stable', 'corr_r_d2', 'corr_a_d2', 'regime']
                   if c in df_oracle.columns]],
        on='condition_id', how='left', suffixes=('_sweep', '_oracle'))

    # Resolve column names after merge
    gain_col = 'gain_oracle_oracle' if 'gain_oracle_oracle' in df.columns else 'gain_oracle'
    stable_col = 'oracle_stable_oracle' if 'oracle_stable_oracle' in df.columns else 'oracle_stable'
    regime_col = 'regime_oracle' if 'regime_oracle' in df.columns else 'regime'
    status_col = 'status_oracle' if 'status_oracle' in df.columns else 'status'

    ers_all = df['event_rate'].values

    # Degenerate conditions (no gain)
    degen_mask = df[status_col] != 'ok'
    if degen_mask.any():
        ax.scatter(ers_all[degen_mask],
                   np.zeros(degen_mask.sum()),
                   marker='x', c='gray', s=40, lw=1.5, zorder=2,
                   label=f'Degenerate (n={degen_mask.sum()})')

    # OK conditions split by regime
    ok_mask = df[status_col] == 'ok'
    if not ok_mask.any():
        ax.set_xlabel('Event rate')
        ax.set_ylabel('Oracle gain')
        path = OUT_DIR / "fig_oracle_gain_vs_eventrate.pdf"
        fig.savefig(path)
        plt.close(fig)
        return path

    ers_ok = df.loc[ok_mask, 'event_rate'].values
    gains_ok = df.loc[ok_mask, gain_col].values
    regimes_ok = df.loc[ok_mask, regime_col].values
    stable_ok = df.loc[ok_mask, stable_col].values

    # Intermittent (ER < 0.50): filled circles
    inter_mask = regimes_ok == 'intermittent'
    if inter_mask.any():
        ers_i = ers_ok[inter_mask]
        gains_i = gains_ok[inter_mask]
        stab_i = stable_ok[inter_mask]

        # Stable intermittent
        s_mask = stab_i.astype(bool)
        if s_mask.any():
            ax.scatter(ers_i[s_mask], gains_i[s_mask],
                       marker='o', c='steelblue', s=50, alpha=0.85,
                       edgecolors='k', linewidths=0.5, zorder=4,
                       label=f'Intermittent (n={s_mask.sum()})')
        # Unstable intermittent
        u_mask = ~s_mask
        if u_mask.any():
            ax.scatter(ers_i[u_mask], gains_i[u_mask],
                       marker='o', c='steelblue', s=50, alpha=0.85,
                       edgecolors='red', linewidths=1.5, zorder=4,
                       label=f'Intermittent unstable (n={u_mask.sum()})')

    # Continuous (ER >= 0.50): open triangles
    cont_mask = regimes_ok == 'continuous'
    if cont_mask.any():
        ers_c = ers_ok[cont_mask]
        gains_c = gains_ok[cont_mask]
        stab_c = stable_ok[cont_mask]

        s_mask = stab_c.astype(bool)
        if s_mask.any():
            ax.scatter(ers_c[s_mask], gains_c[s_mask],
                       marker='^', c='coral', s=50, alpha=0.85,
                       edgecolors='k', linewidths=0.5, zorder=3,
                       label=f'Continuous (n={s_mask.sum()})')
        u_mask = ~s_mask
        if u_mask.any():
            ax.scatter(ers_c[u_mask], gains_c[u_mask],
                       marker='^', c='coral', s=50, alpha=0.85,
                       edgecolors='red', linewidths=1.5, zorder=3,
                       label=f'Continuous unstable (n={u_mask.sum()})')

    # Regime boundary
    ax.axvline(ER_INTERMITTENT_MAX, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.text(ER_INTERMITTENT_MAX + 0.01, ax.get_ylim()[1] * 0.9,
            'continuous', fontsize=7, va='top', color='coral')
    ax.text(ER_INTERMITTENT_MAX - 0.01, ax.get_ylim()[1] * 0.9,
            'intermittent', fontsize=7, va='top', ha='right', color='steelblue')

    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.axvspan(0.05, 0.40, color='green', alpha=0.06)
    ax.set_xlabel('Event rate')
    ax.set_ylabel('Oracle gain (fraction of MSE removed)')
    ax.set_title('Oracle Gain vs Event Rate')
    ax.legend(fontsize=6, loc='upper right')

    path = OUT_DIR / "fig_oracle_gain_vs_eventrate.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def fig_condition_axes(df):
    """Distributions of key condition axes."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    fig.suptitle('Condition Axes Distributions (Phase 3)', fontsize=11)

    # mu_s_final
    ax = axes[0, 0]
    ax.hist(df['mu_s_final'], bins=12, color='coral', edgecolor='white', alpha=0.85)
    ax.set_xlabel('mu_s (final)')
    ax.set_ylabel('Count')
    ax.set_title('Static friction')

    # k_spring
    ax = axes[0, 1]
    ax.hist(df['k_spring'], bins=12, color='teal', edgecolor='white', alpha=0.85)
    ax.set_xlabel('k_spring [N/m]')
    ax.set_ylabel('Count')
    ax.set_title('Spring stiffness')

    # d_p
    ax = axes[1, 0]
    ax.hist(df['d_p'] * 1000, bins=12, color='mediumpurple', edgecolor='white', alpha=0.85)
    ax.set_xlabel('d_p [mm]')
    ax.set_ylabel('Count')
    ax.set_title('Particle diameter')

    # Forcing type (bar)
    ax = axes[1, 1]
    fv_counts = df['forcing_variant'].value_counts().sort_index()
    ax.barh(range(len(fv_counts)), fv_counts.values, color='goldenrod', edgecolor='white')
    ax.set_yticks(range(len(fv_counts)))
    labels = [s.replace('scale_', 's').replace('am_', 'AM')[:18] for s in fv_counts.index]
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('Count')
    ax.set_title('Forcing variant')

    fig.tight_layout()
    path = OUT_DIR / "fig_condition_axes.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def write_report(df_sweep, df_oracle):
    """Write report_phase3.md."""
    n = len(df_sweep)
    ers = df_sweep['event_rate'].values
    target_lo, target_hi = 0.05, 0.40
    n_in_target = int(np.sum((ers >= target_lo) & (ers <= target_hi)))
    pct_in_target = 100.0 * n_in_target / n

    # Non-degenerate (ER in [0.01, 0.95] and x_std > 0)
    n_nondegen = int(np.sum(df_sweep['nondegen']))

    # Oracle stats by regime
    has_oracle = 'status' in df_oracle.columns
    if has_oracle:
        ok = df_oracle[df_oracle['status'] == 'ok']
        inter = ok[ok['regime'] == 'intermittent'] if 'regime' in ok.columns else ok
        cont = ok[ok['regime'] == 'continuous'] if 'regime' in ok.columns else pd.DataFrame()
        degen = df_oracle[df_oracle['status'] == 'degenerate'] if has_oracle else pd.DataFrame()
    else:
        ok = df_oracle.dropna(subset=['gain_oracle']) if 'gain_oracle' in df_oracle.columns else pd.DataFrame()
        inter = ok
        cont = pd.DataFrame()
        degen = pd.DataFrame()

    gains = ok['gain_oracle'].values if len(ok) > 0 else np.array([])
    corr_col = 'corr_r_d2' if 'corr_r_d2' in ok.columns else 'corr_a_d2'
    corrs = ok[corr_col].dropna().values if len(ok) > 0 and corr_col in ok.columns else np.array([])

    lines = [
        "# Phase 3 Report: ER-Balanced Virtual Condition Expansion",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        f"- **Conditions generated:** {n}",
        f"- **Non-degenerate (ER in [0.01, 0.95], x_std > 0):** {n_nondegen}/{n}",
        f"- **ER in target [{target_lo}, {target_hi}]:** {n_in_target}/{n} ({pct_in_target:.0f}%)",
        f"- **Oracle eval:** {len(ok)} ok, {len(degen)} degenerate",
        "",
        "## Event Rate Distribution",
        "",
        f"- Median: {np.median(ers):.3f}",
        f"- IQR: [{np.percentile(ers, 25):.3f}, {np.percentile(ers, 75):.3f}]",
        f"- Min: {np.min(ers):.3f}, Max: {np.max(ers):.3f}",
        f"- Intermittent (ER < {ER_INTERMITTENT_MAX}): {int(np.sum(ers < ER_INTERMITTENT_MAX))}",
        f"- Continuous (ER >= {ER_INTERMITTENT_MAX}): {int(np.sum(ers >= ER_INTERMITTENT_MAX))}",
        "",
    ]

    if len(gains) > 0:
        lines += [
            "## Oracle Gain Distribution",
            "",
            f"- All ok: median={np.median(gains):.3f}, "
            f"IQR=[{np.percentile(gains, 25):.3f}, {np.percentile(gains, 75):.3f}]",
        ]
        if len(inter) > 0:
            ig = inter['gain_oracle'].values
            lines.append(f"- **Intermittent** (n={len(inter)}): "
                         f"median={np.median(ig):.3f}, "
                         f"IQR=[{np.percentile(ig, 25):.3f}, {np.percentile(ig, 75):.3f}]")
        if len(cont) > 0:
            cg = cont['gain_oracle'].values
            lines.append(f"- **Continuous** (n={len(cont)}): "
                         f"median={np.median(cg):.3f}, "
                         f"IQR=[{np.percentile(cg, 25):.3f}, {np.percentile(cg, 75):.3f}]")

        if 'oracle_stable' in ok.columns:
            n_stable = int(ok['oracle_stable'].sum())
            n_unstable = len(ok) - n_stable
            lines.append(f"- Stable: {n_stable}, Unstable: {n_unstable}")
        lines.append("")

    if len(corrs) > 0:
        lines += [
            "## Non-Circularity",
            "",
            f"- corr(residual, -v|u|) range: [{np.min(corrs):.4f}, {np.max(corrs):.4f}]",
            f"- Median: {np.median(corrs):.4f}",
            f"- All within [-0.10, +0.10]: {'YES' if np.all(np.abs(corrs) < 0.10) else 'NO'}",
            "",
        ]

    lines += [
        "## k_spring Role",
        "",
        "The restoring spring (k_spring) was introduced in Phase 2 to ensure the particle",
        "returns to the pin position in unidirectional flow. Without it, particles escape",
        "permanently after first breakaway. The default k_spring=0.02 N/m overcomes kinetic",
        "friction at ~5mm displacement. In Phase 3, k_spring is jittered (+-12% lognormal)",
        "across conditions to add diversity while preserving intermittent dynamics.",
        "d_p is held fixed at 3mm (jitter disabled to reduce ER variability).",
        "",
        f"- k_spring range: [{df_sweep['k_spring'].min():.4f}, {df_sweep['k_spring'].max():.4f}] N/m",
        f"- Median: {df_sweep['k_spring'].median():.4f} N/m",
        "",
        "## Forcing Variants",
        "",
    ]
    fv_counts = df_sweep['forcing_variant'].value_counts().sort_index()
    for fv, cnt in fv_counts.items():
        sub = df_sweep[df_sweep['forcing_variant'] == fv]
        lines.append(f"- **{fv}**: {cnt} conditions, "
                     f"median ER={sub['event_rate'].median():.3f}")
    lines.append("")

    # Per-condition table
    lines += [
        "## Per-Condition Summary",
        "",
        "| Condition | Forcing | Regime | mu_s | ER | Gain | Stable | corr_d2 |",
        "|-----------|---------|--------|------|----|------|--------|---------|",
    ]
    for _, row in df_sweep.iterrows():
        cid = row['condition_id']
        # Look up oracle result
        orow = df_oracle[df_oracle['condition_id'] == cid] if len(df_oracle) > 0 else pd.DataFrame()
        if len(orow) > 0:
            orow = orow.iloc[0]
            st = orow.get('status', 'ok')
            rgm = orow.get('regime', '?')
            g = f"{orow['gain_oracle']:.3f}" if pd.notna(orow.get('gain_oracle')) else "N/A"
            stab = 'Y' if orow.get('oracle_stable', True) else 'N'
            corr_val = orow.get('corr_r_d2') if pd.notna(orow.get('corr_r_d2')) else orow.get('corr_a_d2')
            c = f"{corr_val:.4f}" if pd.notna(corr_val) else "N/A"
            if st == 'degenerate':
                g = 'DEGEN'
                stab = '-'
                c = '-'
        else:
            rgm = '?'
            g = 'N/A'
            stab = '-'
            c = 'N/A'

        lines.append(f"| {cid} | {row['forcing_variant'][:18]} | "
                     f"{rgm[:5]} | {row['mu_s_final']:.3f} | "
                     f"{row['event_rate']:.3f} | {g} | {stab} | {c} |")

    report_path = OUT_DIR / "report_phase3.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {report_path.name}")
    return report_path


def main():
    print("=" * 60)
    print("PHASE 3 QC SUMMARY")
    print("=" * 60)

    df_sweep, df_oracle = load_data()
    print(f"Loaded: {len(df_sweep)} conditions, {len(df_oracle)} oracle results")

    # Generate plots
    fig_eventrate_hist(df_sweep)

    if len(df_oracle) > 0:
        fig_oracle_gain_vs_er(df_sweep, df_oracle)

    fig_condition_axes(df_sweep)

    # Write report
    write_report(df_sweep, df_oracle)

    print(f"\n{'='*60}")
    print("QC SUMMARY COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
