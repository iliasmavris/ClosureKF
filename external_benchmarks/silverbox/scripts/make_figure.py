"""
Step 5: Generate summary figure for Silverbox external benchmark.

Panel (a): Delta DxR2 (closure - physics) vs horizon
Panel (b): Term selection frequency bar chart

Uses paper_style.py via importlib (no modifications to existing files).

Usage: python -u external_benchmarks/silverbox/scripts/make_figure.py
"""

import os, sys, json, hashlib, time, importlib.util
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# --- Path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
SILVERBOX_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SILVERBOX_DIR.parent.parent
OUT_DIR = SILVERBOX_DIR / "outputs"
FIG_DIR = SILVERBOX_DIR / "figures"

# --- Import paper_style ---
STYLE_PATH = os.path.join(PROJECT_ROOT, 'ems_v1', 'figures', 'paper_style.py')
spec = importlib.util.spec_from_file_location("paper_style", STYLE_PATH)
paper_style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_style)
paper_style.apply_gmd_style()
COLORS = paper_style.COLORS
DOUBLE_COL = paper_style.DOUBLE_COL

# Closure term names
TERM_NAMES = ['a1', 'b1', 'b2', 'd1', 'd2', 'd3']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 5: MAKE FIGURE")
    print("=" * 60)

    # Load metrics
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    sel_path = OUT_DIR / "selection_summary.json"
    with open(sel_path, 'r') as f:
        selection = json.load(f)

    dt_eff = metrics['dt_eff']
    seeds = list(metrics['per_seed'].keys())
    n_seeds = len(seeds)

    # =========================================
    #  Panel (a): Delta DxR2 vs horizon
    # =========================================

    # Collect DxR2 curves per seed
    phys_curves = []
    clos_curves = []
    for seed_key in seeds:
        p_dxr2 = np.array(metrics['per_seed'][seed_key]['physics']['dxr2']['values'])
        c_dxr2 = np.array(metrics['per_seed'][seed_key]['closure']['dxr2']['values'])
        phys_curves.append(p_dxr2)
        clos_curves.append(c_dxr2)

    # Align lengths
    min_len = min(min(len(c) for c in phys_curves), min(len(c) for c in clos_curves))
    phys_arr = np.array([c[:min_len] for c in phys_curves])
    clos_arr = np.array([c[:min_len] for c in clos_curves])

    delta_dxr2 = clos_arr - phys_arr  # [n_seeds, max_h]
    mean_delta = np.nanmean(delta_dxr2, axis=0)
    std_delta = np.nanstd(delta_dxr2, axis=0)

    h_steps = np.arange(1, min_len + 1)
    h_sec = h_steps * dt_eff

    # =========================================
    #  Panel (b): Term selection frequency
    # =========================================
    term_counts = {name: 0 for name in TERM_NAMES}
    term_counts['none'] = 0

    for seed_key in seeds:
        final_terms = selection[seed_key].get('final_terms', ['none'])
        if final_terms == ['none'] or len(final_terms) == 0:
            term_counts['none'] += 1
        else:
            for t in final_terms:
                if t in term_counts:
                    term_counts[t] += 1

    # Also get mean ACF1 and NIS for inset text
    phys_acf1 = [metrics['per_seed'][s]['physics']['acf1'] for s in seeds]
    clos_acf1 = [metrics['per_seed'][s]['closure']['acf1'] for s in seeds]
    phys_nis = [metrics['per_seed'][s]['physics']['nis_mean'] for s in seeds]
    clos_nis = [metrics['per_seed'][s]['closure']['nis_mean'] for s in seeds]

    # =========================================
    #  Create figure
    # =========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE_COL,
                                    gridspec_kw={'width_ratios': [2.5, 1],
                                                 'wspace': 0.35})

    # Panel (a)
    ax1.plot(h_sec, mean_delta, color=COLORS['closure_1t'], lw=2,
             label='Closure - Physics')
    ax1.fill_between(h_sec,
                     mean_delta - std_delta,
                     mean_delta + std_delta,
                     color=COLORS['closure_1t'], alpha=0.2)
    ax1.axhline(0, color='#999999', ls='--', lw=0.8)

    # Vertical dotted lines at target horizons
    horizon_targets = metrics.get('horizon_map', [])
    for hm in horizon_targets:
        ax1.axvline(hm['achieved_sec'], color='#BDBDBD', ls=':', lw=0.7, alpha=0.7)

    ax1.set_xlabel('Horizon (s)')
    ax1.set_ylabel(r'$\Delta R^2_{\Delta x}$ (closure $-$ physics)')
    ax1.set_title('(a) Closure improvement')

    # Panel (b)
    bar_names = TERM_NAMES + ['none']
    bar_vals = [term_counts[n] for n in bar_names]
    bar_colors = []
    for n in bar_names:
        if n == 'none':
            bar_colors.append(COLORS['persistence'])
        elif n in ['d2', 'b2']:
            bar_colors.append(COLORS['closure_2t'])
        else:
            bar_colors.append(COLORS['ridge'])

    x_pos = np.arange(len(bar_names))
    ax2.bar(x_pos, bar_vals, color=bar_colors, edgecolor='#444444', linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_names, fontsize=8)
    ax2.set_xlabel('Closure term')
    ax2.set_ylabel(f'Count (of {n_seeds} seeds)')
    ax2.set_title('(b) Term selection')
    ax2.set_ylim(0, n_seeds + 0.5)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Inset text: ACF1 and NIS
    inset_text = (
        f"ACF(1):\n"
        f"  Phys: {np.mean(phys_acf1):.3f}{chr(177)}{np.std(phys_acf1):.3f}\n"
        f"  Clos: {np.mean(clos_acf1):.3f}{chr(177)}{np.std(clos_acf1):.3f}\n"
        f"NIS:\n"
        f"  Phys: {np.mean(phys_nis):.3f}{chr(177)}{np.std(phys_nis):.3f}\n"
        f"  Clos: {np.mean(clos_nis):.3f}{chr(177)}{np.std(clos_nis):.3f}"
    )
    ax2.text(0.98, 0.98, inset_text, transform=ax2.transAxes,
             fontsize=7, verticalalignment='top', horizontalalignment='right',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#CCCCCC', alpha=0.8))

    fig.tight_layout()

    # Save
    pdf_path = FIG_DIR / "fig_ext_silverbox_summary.pdf"
    png_path = FIG_DIR / "fig_ext_silverbox_summary.png"
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=300)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    # Meta JSON
    meta = {
        'script': os.path.basename(__file__),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_files': {
            'metrics': str(metrics_path),
            'metrics_sha256': sha256_file(metrics_path),
            'selection': str(sel_path),
            'selection_sha256': sha256_file(sel_path),
        },
        'environment': {},
    }
    try:
        import platform
        meta['environment']['python'] = platform.python_version()
        meta['environment']['numpy'] = np.__version__
        import pandas; meta['environment']['pandas'] = pandas.__version__
        meta['environment']['matplotlib'] = matplotlib.__version__
    except Exception:
        pass

    meta_path = FIG_DIR / "fig_ext_silverbox_summary_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
