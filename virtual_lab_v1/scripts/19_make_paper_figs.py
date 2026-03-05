"""
19_make_paper_figs.py - Main-text quality Phase-2 figure set
============================================================
Produces 5 PDFs + captions in outputs/paper_figs/:

  FIG 1  fig_virtual_schematic.pdf         Hybrid pinned-free dynamics schematic
  FIG 2  fig_virtual_example_timeseries.pdf Example condition: x(t), u(t), at_pin ribbon
  FIG 3  fig_virtual_eventrate_vs_mus.pdf   Event rate vs mu_s across conditions
  FIG 4  fig_virtual_noncircularity.pdf     corr(a_true, -v|u|) per condition
  FIG 5  fig_virtual_oracle_gap.pdf         Oracle gap: R2 physics vs R2 oracle

Also writes captions_virtual_lab.md.

Reads from:
  datasets/*/x_10hz.csv, truth_states_raw.csv, meta.json
  outputs/paper_figs/oracle_summary.csv  (from 18_oracle_eval.py)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_figs"

# ---- Publication style ----
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'figure.dpi': 150,
})

COLORS = {
    'phys': '#2166ac',       # blue
    'oracle': '#b2182b',     # red
    'pinned': '#fee08b',     # yellow fill
    'free': '#d9ef8b',       # green fill
    'bar': '#4393c3',        # bar blue
    'bar2': '#d6604d',       # bar red
    'neutral': '#636363',    # grey
}


# ---- Data loading helpers ----

def load_all_conditions():
    """Load meta.json for all conditions, sorted by id."""
    datasets_dir = ROOT / "datasets"
    cond_dirs = sorted(datasets_dir.glob("condition_*"))
    all_meta = []
    for cd in cond_dirs:
        mp = cd / "meta.json"
        if mp.exists():
            with open(mp) as f:
                meta = json.load(f)
            meta['_dir'] = str(cd)
            all_meta.append(meta)
    return all_meta


def compute_a_true(v_p, dt, smooth_hz=10.0):
    """Acceleration via smoothed central diff (same as 18_oracle_eval.py)."""
    fs = 1.0 / dt
    if smooth_hz is not None and smooth_hz < fs / 2:
        sos = butter(4, smooth_hz, btype='low', fs=fs, output='sos')
        v_smooth = sosfiltfilt(sos, v_p)
    else:
        v_smooth = v_p.copy()
    a = np.zeros_like(v_smooth)
    a[1:-1] = (v_smooth[2:] - v_smooth[:-2]) / (2.0 * dt)
    a[0] = (v_smooth[1] - v_smooth[0]) / dt
    a[-1] = (v_smooth[-1] - v_smooth[-2]) / dt
    return a


# ==================================================================
# FIG 1 - Schematic: Hybrid Pinned-Free Dynamics
# ==================================================================

def fig_schematic(out_dir):
    """Clean schematic of the hybrid pinned-free sphere dynamics."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    # Title
    ax.text(5.0, 6.2, 'Hybrid Pinned-Free Sphere Dynamics',
            ha='center', va='top', fontsize=12, fontweight='bold')

    # ---- PINNED box ----
    pin_box = FancyBboxPatch(
        (0.5, 3.2), 3.5, 2.5,
        boxstyle="round,pad=0.15", linewidth=1.2,
        edgecolor='#333333', facecolor=COLORS['pinned'], alpha=0.4)
    ax.add_patch(pin_box)
    ax.text(2.25, 5.45, 'PINNED', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#333333')
    pin_lines = [
        r'$x = x_{\mathrm{pin}} + \xi$',
        r'$v_p = 0$',
        r'$\xi$: OU jitter',
        r'$(\sigma_x,\, \tau_x)$',
    ]
    for i, line in enumerate(pin_lines):
        ax.text(2.25, 4.95 - i * 0.42, line,
                ha='center', va='center', fontsize=8)

    # ---- FREE box ----
    free_box = FancyBboxPatch(
        (6.0, 3.2), 3.5, 2.5,
        boxstyle="round,pad=0.15", linewidth=1.2,
        edgecolor='#333333', facecolor=COLORS['free'], alpha=0.4)
    ax.add_patch(free_box)
    ax.text(7.75, 5.45, 'FREE (sliding)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#333333')
    free_lines = [
        r'$m_{\mathrm{eff}}\,\dot{v}_p = \sum F_i$',
        r'RK4, $\Delta t = 0.005$ s',
        r'Coulomb kinetic friction',
        r'Blended drag + added mass',
    ]
    for i, line in enumerate(free_lines):
        ax.text(7.75, 4.95 - i * 0.42, line,
                ha='center', va='center', fontsize=8)

    # ---- Arrows: breakaway (top) and capture (bottom) ----
    ax.annotate('',
                xy=(5.9, 5.2), xytext=(4.1, 5.2),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                                color='#b2182b', connectionstyle='arc3,rad=-0.15'))
    ax.text(5.0, 5.55, 'Breakaway', ha='center', fontsize=8,
            fontstyle='italic', color='#b2182b')

    ax.annotate('',
                xy=(4.1, 3.6), xytext=(5.9, 3.6),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                                color='#2166ac', connectionstyle='arc3,rad=-0.15'))
    ax.text(5.0, 3.2, 'Capture', ha='center', fontsize=8,
            fontstyle='italic', color='#2166ac')

    # ---- Transition conditions ----
    ax.text(5.0, 2.5,
            r'Breakaway: $|F_{\mathrm{drive}}| > \mu_s W_{\mathrm{sub}} + F_{\mathrm{pin}}$',
            ha='center', fontsize=8, color='#b2182b')
    ax.text(5.0, 2.0,
            r'Capture: $|x - x_{\mathrm{pin}}| < \varepsilon_x$'
            r'  and  $|F_{\mathrm{drive}}| < \mu_s W_{\mathrm{sub}} + F_{\mathrm{pin}}$',
            ha='center', fontsize=8, color='#2166ac')
    ax.text(5.0, 1.55,
            r'and ($|v_p| < \varepsilon_v$ or velocity sign change)',
            ha='center', fontsize=7.5, color='#2166ac')

    # ---- Force terms (bottom) ----
    ax.plot([0.5, 9.5], [1.1, 1.1], '-', color='#999999', lw=0.5)
    ax.text(5.0, 0.8, 'Free-mode forces:', ha='center', fontsize=8,
            fontweight='bold', color='#333333')
    force_str = (
        r'$F_D = 3\pi\mu d_p w + \frac{1}{2}\rho C_D A_p |w| w$'
        r'$\quad$'
        r'$F_A = C_{Du} m_f \dot{u}_b$'
        r'$\quad$'
        r'$F_\eta = m_{\mathrm{eff}}\,\eta(t)$'
    )
    ax.text(5.0, 0.35, force_str, ha='center', fontsize=7.5, color='#555555')

    fig.tight_layout()
    path = out_dir / "fig_virtual_schematic.pdf"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  FIG 1: {path.name}")
    return path


# ==================================================================
# FIG 2 - Example Time Series
# ==================================================================

def fig_example_timeseries(all_meta, out_dir):
    """3-panel time series for one representative condition."""
    if not all_meta:
        print("  FIG 2: SKIP (no conditions)")
        return None

    # Pick condition nearest to median event rate
    ers = [m['event_rate'] for m in all_meta]
    median_er = np.median(ers)
    idx = int(np.argmin([abs(e - median_er) for e in ers]))
    meta = all_meta[idx]
    cond_dir = Path(meta['_dir'])
    cid = meta['condition_id']

    # Load 10Hz data
    df10 = pd.read_csv(cond_dir / "x_10hz.csv")
    t_10 = df10['timestamp'].values
    x_10 = df10['displacement'].values * 1000  # mm
    u_10 = df10['velocity'].values

    # Load truth for at_pin
    df_truth = pd.read_csv(cond_dir / "truth_states_raw.csv")
    t_raw = df_truth['time'].values
    at_pin_raw = df_truth['at_pin'].values

    # Trim to a nice window for visibility (100s segment from middle)
    t_mid = 0.5 * (t_10[0] + t_10[-1])
    t_win = 60.0  # seconds of window
    t_lo = t_mid - t_win / 2
    t_hi = t_mid + t_win / 2
    mask10 = (t_10 >= t_lo) & (t_10 <= t_hi)
    mask_raw = (t_raw >= t_lo) & (t_raw <= t_hi)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.5), sharex=True,
                              gridspec_kw={'height_ratios': [3, 2, 1]})

    # Panel (a): displacement
    ax = axes[0]
    ax.plot(t_10[mask10], x_10[mask10], color='navy', linewidth=0.6)
    ax.set_ylabel('x [mm]')
    ax.set_title(f'{cid}  (event rate = {meta["event_rate"]:.3f})',
                 fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.text(0.01, 0.92, '(a)', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')

    # Panel (b): flow velocity
    ax = axes[1]
    ax.plot(t_10[mask10], u_10[mask10], color='#d6604d', linewidth=0.6)
    ax.set_ylabel('u [m/s]')
    ax.grid(True, alpha=0.2)
    ax.text(0.01, 0.92, '(b)', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')

    # Panel (c): at_pin ribbon
    ax = axes[2]
    t_r = t_raw[mask_raw]
    ap_r = at_pin_raw[mask_raw]
    ax.fill_between(t_r, 0, ap_r, step='post',
                    color=COLORS['pinned'], alpha=0.7, label='Pinned')
    ax.fill_between(t_r, 0, 1 - ap_r, step='post',
                    color=COLORS['free'], alpha=0.5, label='Free')
    ax.set_ylabel('State')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Free', 'Pinned'])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time [s]')
    ax.text(0.01, 0.92, '(c)', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')

    fig.tight_layout()
    path = out_dir / "fig_virtual_example_timeseries.pdf"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  FIG 2: {path.name}  ({cid})")
    return path


# ==================================================================
# FIG 3 - Event Rate vs mu_s
# ==================================================================

def fig_eventrate_vs_mus(all_meta, out_dir):
    """Event rate vs mu_s (final) across all conditions."""
    if not all_meta:
        print("  FIG 3: SKIP (no conditions)")
        return None

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    mu_s_vals = []
    er_vals = []
    labels = []
    for m in all_meta:
        mu_s = m['config']['friction']['mu_s']
        mu_s_vals.append(mu_s)
        er_vals.append(m['event_rate'])
        labels.append(m['condition_id'].replace('condition_', 'C'))

    mu_s_vals = np.array(mu_s_vals)
    er_vals = np.array(er_vals)

    # Color by mu_k
    mu_k_vals = [m['config']['friction']['mu_k'] for m in all_meta]
    unique_mu_k = sorted(set(mu_k_vals))
    colors_map = {mk: c for mk, c in zip(unique_mu_k,
                  ['#4393c3', '#d6604d', '#5aae61', '#762a83'])}

    for i, (ms, er, mk) in enumerate(zip(mu_s_vals, er_vals, mu_k_vals)):
        ax.scatter(ms, er, s=60, c=colors_map[mk], edgecolors='black',
                   linewidths=0.5, zorder=3,
                   label=f'mu_k={mk:.2f}' if i == mu_k_vals.index(mk) else '')
        ax.annotate(labels[i], (ms, er), fontsize=6,
                    textcoords='offset points', xytext=(4, 4))

    ax.set_xlabel(r'$\mu_s$ (static friction)')
    ax.set_ylabel('Event rate (sliding fraction)')
    ax.set_title('Event Rate vs Static Friction')
    ax.grid(True, alpha=0.2)
    ax.axhline(0.01, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(0.80, color='grey', ls='--', lw=0.5, alpha=0.5)

    # Remove duplicate labels
    handles, labs = ax.get_legend_handles_labels()
    by_label = dict(zip(labs, handles))
    if len(by_label) > 1:
        ax.legend(by_label.values(), by_label.keys(), fontsize=7)

    fig.tight_layout()
    path = out_dir / "fig_virtual_eventrate_vs_mus.pdf"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  FIG 3: {path.name}")
    return path


# ==================================================================
# FIG 4 - Non-Circularity
# ==================================================================

def fig_noncircularity(all_meta, out_dir):
    """Bar plot of corr(a_true, -v_p*|u_b|) for free segments per condition."""
    if not all_meta:
        print("  FIG 4: SKIP (no conditions)")
        return None

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    cond_labels = []
    corr_vals = []

    for meta in all_meta:
        cond_dir = Path(meta['_dir'])
        truth_path = cond_dir / "truth_states_raw.csv"
        if not truth_path.exists():
            continue

        df = pd.read_csv(truth_path)
        dt = meta['config']['integration']['dt_sim']
        v_p = df['v_p'].values
        at_pin = df['at_pin'].values
        u_b = df['u_b'].values

        is_free = (at_pin == 0)
        if np.sum(is_free) < 100:
            continue

        a_true = compute_a_true(v_p, dt)
        d2_feat = -v_p[is_free] * np.abs(u_b[is_free])

        if np.std(d2_feat) > 1e-20 and np.std(a_true[is_free]) > 1e-20:
            corr = np.corrcoef(a_true[is_free], d2_feat)[0, 1]
        else:
            corr = 0.0

        cond_labels.append(meta['condition_id'].replace('condition_', 'C'))
        corr_vals.append(corr)

    if not corr_vals:
        print("  FIG 4: SKIP (no valid data)")
        return None

    x_pos = np.arange(len(corr_vals))
    bar_colors = [COLORS['bar'] if abs(c) < 0.3 else '#d73027'
                  for c in corr_vals]
    ax.bar(x_pos, corr_vals, color=bar_colors, edgecolor='black',
           linewidth=0.5, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, rotation=45, ha='right')
    ax.set_ylabel(r'corr($a_{\mathrm{true}},\; -v_p |u_b|$)')
    ax.set_title('Non-Circularity: Truth vs Closure Feature')
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(0.3, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(-0.3, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.2, axis='y')

    # Annotate values
    for i, c in enumerate(corr_vals):
        ax.text(i, c + 0.05 * np.sign(c), f'{c:.3f}',
                ha='center', va='bottom' if c >= 0 else 'top',
                fontsize=6.5)

    fig.tight_layout()
    path = out_dir / "fig_virtual_noncircularity.pdf"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  FIG 4: {path.name}")
    return path


# ==================================================================
# FIG 5 - Oracle Gap
# ==================================================================

def fig_oracle_gap(out_dir):
    """Grouped bar: R2 physics vs R2 oracle per condition."""
    csv_path = out_dir / "oracle_summary.csv"
    if not csv_path.exists():
        print("  FIG 5: SKIP (oracle_summary.csv not found)")
        return None

    df = pd.read_csv(csv_path)
    n = len(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5),
                                    gridspec_kw={'width_ratios': [3, 2]})

    # Left panel: grouped bars for R2
    x_pos = np.arange(n)
    w = 0.35
    labels = [cid.replace('condition_', 'C') for cid in df['condition_id']]

    ax1.bar(x_pos - w/2, df['R2_phys'], w, color=COLORS['phys'],
            edgecolor='black', linewidth=0.5, label='Physics-only')
    ax1.bar(x_pos + w/2, df['R2_oracle'], w, color=COLORS['bar2'],
            edgecolor='black', linewidth=0.5, label='Oracle (phys+library)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel(r'$R^2$ (acceleration)')
    ax1.set_title('Physics vs Oracle: Acceleration Fit')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.axhline(0, color='black', lw=0.5)
    ax1.text(0.01, 0.97, '(a)', transform=ax1.transAxes, fontsize=9,
             fontweight='bold', va='top')

    # Right panel: oracle gain bars
    ax2.bar(x_pos, df['gain_oracle'], 0.6, color='#66c2a5',
            edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Oracle gain')
    ax2.set_title('Oracle Gain\n(MSE improvement fraction)')
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.axhline(0, color='black', lw=0.5)
    ax2.text(0.01, 0.97, '(b)', transform=ax2.transAxes, fontsize=9,
             fontweight='bold', va='top')

    # Annotate gain values
    for i, g in enumerate(df['gain_oracle']):
        ax2.text(i, g + 0.01, f'{g:.2f}', ha='center', va='bottom',
                 fontsize=7)

    fig.tight_layout()
    path = out_dir / "fig_virtual_oracle_gap.pdf"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  FIG 5: {path.name}")
    return path


# ==================================================================
# Captions
# ==================================================================

def write_captions(out_dir, all_meta):
    """Write captions_virtual_lab.md."""
    n_cond = len(all_meta)
    lines = [
        "# Virtual Lab Phase 2 - Figure Captions",
        "",
        "## Figure 1: Hybrid Pinned-Free Dynamics Schematic",
        "",
        "Schematic of the virtual-lab sphere truth model. The particle "
        "alternates between a PINNED state (resting at the pin with OU "
        "jitter) and a FREE state (sliding under blended Stokes + form "
        "drag, Coulomb friction, added mass, and OU noise, integrated "
        "via RK4). Transition conditions govern breakaway and re-capture.",
        "",
        "## Figure 2: Example Time Series",
        "",
        "Representative condition showing (a) particle displacement x(t) "
        "at 10 Hz, (b) near-bed flow velocity u(t), and (c) binary "
        "pinned/free state. Pinned intervals (yellow) alternate with "
        "free excursions driven by turbulent flow bursts.",
        "",
        "## Figure 3: Event Rate vs Static Friction",
        "",
        f"Sliding fraction (event rate) across {n_cond} conditions as a "
        "function of static friction coefficient. Higher mu_s produces "
        "longer pinned intervals and lower event rates, confirming "
        "monotonic physical response. Colours indicate kinetic friction "
        "mu_k.",
        "",
        "## Figure 4: Non-Circularity Verification",
        "",
        "Pearson correlation between the truth-model acceleration and "
        "the EKF closure feature -v_p|u_b| (the d2 term), computed on "
        "free segments only. Values near zero confirm that the truth "
        "model's dynamics are structurally different from the reduced "
        "closure library, preventing circular validation.",
        "",
        "## Figure 5: Oracle Gap Analysis",
        "",
        "(a) Coefficient of determination R^2 for physics-only (blue) "
        "and oracle library (red) fits to the true acceleration during "
        "free motion. The oracle uses the manuscript's 6-term closure "
        "library with access to true states, establishing an upper "
        "bound for data-driven discovery. "
        "(b) Oracle gain: fraction of physics-only MSE removed by the "
        "library closure.",
    ]
    path = out_dir / "captions_virtual_lab.md"
    path.write_text('\n'.join(lines))
    print(f"  Captions: {path.name}")
    return path


# ==================================================================
# Main
# ==================================================================

def main():
    print("=" * 60)
    print("PAPER FIGURES - Phase 2 Virtual Lab")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_meta = load_all_conditions()
    print(f"Loaded {len(all_meta)} conditions")

    figures = []

    print("\nFIG 1: Schematic...")
    p = fig_schematic(OUT_DIR)
    if p:
        figures.append(p)

    print("\nFIG 2: Example timeseries...")
    p = fig_example_timeseries(all_meta, OUT_DIR)
    if p:
        figures.append(p)

    print("\nFIG 3: Event rate vs mu_s...")
    p = fig_eventrate_vs_mus(all_meta, OUT_DIR)
    if p:
        figures.append(p)

    print("\nFIG 4: Non-circularity...")
    p = fig_noncircularity(all_meta, OUT_DIR)
    if p:
        figures.append(p)

    print("\nFIG 5: Oracle gap...")
    p = fig_oracle_gap(OUT_DIR)
    if p:
        figures.append(p)

    print("\nCaptions...")
    write_captions(OUT_DIR, all_meta)

    print(f"\n{'='*60}")
    print(f"Generated {len(figures)} figures in {OUT_DIR}")
    for f in figures:
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == '__main__':
    main()
