"""
Step 6 headline extractor.

Reads failure_map_summary.csv and bin_edges.csv, computes verified
headline numbers, writes HEADLINE.txt and appends to README.md.

Usage:
  python -u ems_v1/eval/failure_map_step6/verify_headlines.py
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = ROOT / "ems_v1" / "eval" / "failure_map_step6"


def _col(df, candidates):
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")


def main():
    # --- Load data ---
    summary_path = OUT_DIR / "failure_map_summary.csv"
    bins_path = OUT_DIR / "bin_edges.csv"
    cells_path = OUT_DIR / "failure_map_cells.csv"

    assert summary_path.exists(), f"Missing: {summary_path}"
    assert bins_path.exists(), f"Missing: {bins_path}"
    assert cells_path.exists(), f"Missing: {cells_path}"

    df = pd.read_csv(summary_path)
    df_bins = pd.read_csv(bins_path)
    df_cells = pd.read_csv(cells_path)

    print(f"Summary columns: {df.columns.tolist()}")

    # Auto-detect column names (robust to naming variants)
    col_dskill = _col(df, ['mean_dskill', 'delta_skill_mean'])
    col_dskill_std = _col(df, ['std_dskill', 'delta_skill_std'])
    col_dmae = _col(df, ['mean_dmae', 'delta_mae_mean'])
    col_dmae_std = _col(df, ['std_dmae', 'delta_mae_std'])
    col_npts = _col(df, ['mean_n_points', 'mean_n_eff'])

    n_cells = len(df)
    n_bins = df['bin_id'].nunique()
    n_horizons = df['tau_s'].nunique()
    n_seeds = df_cells['seed'].nunique()

    # --- Max DeltaSkill cell ---
    idx_max = df[col_dskill].idxmax()
    max_row = df.loc[idx_max]
    max_bin = int(max_row['bin_id'])
    max_tau = float(max_row['tau_s'])
    max_dskill = float(max_row[col_dskill])
    max_dskill_std = float(max_row[col_dskill_std])
    max_bin_lo = float(max_row['bin_lo'])
    max_bin_hi = float(max_row['bin_hi'])

    # --- Min DeltaSkill cell ---
    idx_min = df[col_dskill].idxmin()
    min_row = df.loc[idx_min]
    min_bin = int(min_row['bin_id'])
    min_tau = float(min_row['tau_s'])
    min_dskill = float(min_row[col_dskill])
    min_dskill_std = float(min_row[col_dskill_std])
    min_bin_lo = float(min_row['bin_lo'])
    min_bin_hi = float(min_row['bin_hi'])

    # --- MAE sign check ---
    max_dmae = float(df[col_dmae].max())
    min_dmae = float(df[col_dmae].min())
    mae_all_nonpositive = max_dmae <= 0.0

    # --- Best DeltaMAE (most negative = best) cell ---
    idx_best_mae = df[col_dmae].idxmin()
    best_mae_row = df.loc[idx_best_mae]
    best_mae_bin = int(best_mae_row['bin_id'])
    best_mae_tau = float(best_mae_row['tau_s'])
    best_mae_val = float(best_mae_row[col_dmae])
    best_mae_std = float(best_mae_row[col_dmae_std])
    best_mae_bin_lo = float(best_mae_row['bin_lo'])
    best_mae_bin_hi = float(best_mae_row['bin_hi'])

    # --- Bin edges and counts ---
    bin_lines = []
    col_nbin = _col(df_bins, ['n_test_pts', 'n_scored', 'n'])
    for _, row in df_bins.iterrows():
        bin_lines.append(
            f"  Bin {int(row['bin_id'])}: [{row['lo']:.4f}, {row['hi']:.4f}) "
            f"m/s  n_scored={int(row[col_nbin])}")

    # --- Write HEADLINE.txt ---
    lines = [
        "STEP 6 HEADLINE NUMBERS",
        f"Source: {summary_path.name} ({n_cells} cells, {n_bins} bins x {n_horizons} horizons, {n_seeds} seeds)",
        "",
        "MAX_DSKILL:",
        f"  bin={max_bin} [{max_bin_lo:.4f}, {max_bin_hi:.4f}) tau={max_tau}s",
        f"  mean_dskill={max_dskill:+.4f} std={max_dskill_std:.4f}",
        "",
        "MIN_DSKILL:",
        f"  bin={min_bin} [{min_bin_lo:.4f}, {min_bin_hi:.4f}) tau={min_tau}s",
        f"  mean_dskill={min_dskill:+.4f} std={min_dskill_std:.4f}",
        "",
        "MAE_SIGN_CHECK:",
        f"  max(delta_mae_mean)={max_dmae:+.6f}",
        f"  min(delta_mae_mean)={min_dmae:+.6f}",
        f"  all_nonpositive={mae_all_nonpositive}",
        "",
        "BEST_DMAE (most negative):",
        f"  bin={best_mae_bin} [{best_mae_bin_lo:.4f}, {best_mae_bin_hi:.4f}) tau={best_mae_tau}s",
        f"  mean_dmae={best_mae_val:+.6f} std={best_mae_std:.6f}",
        "",
        "BIN_EDGES:",
    ] + bin_lines

    headline_path = OUT_DIR / "HEADLINE.txt"
    with open(headline_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Wrote: {headline_path}")

    # Print to stdout
    for l in lines:
        print(l)

    # --- Append to README.md ---
    readme_path = OUT_DIR / "README.md"
    with open(readme_path, 'a') as f:
        f.write("\n\n## Verified Headline Numbers\n\n")
        f.write("```\n")
        f.write('\n'.join(lines) + '\n')
        f.write("```\n")
    print(f"\nAppended headlines to: {readme_path}")

    # --- Print LaTeX macros ---
    print("\n--- LaTeX macros for metrics.tex ---")
    macros = [
        f"\\newcommand{{\\maxDskill}}{{{max_dskill:+.3f}}}",
        f"\\newcommand{{\\maxDskillStd}}{{{max_dskill_std:.3f}}}",
        f"\\newcommand{{\\maxDskillBinLo}}{{{max_bin_lo:.2f}}}",
        f"\\newcommand{{\\maxDskillBinHi}}{{{max_bin_hi:.2f}}}",
        f"\\newcommand{{\\maxDskillTau}}{{{max_tau:.1f}}}",
        f"\\newcommand{{\\minDskill}}{{{min_dskill:+.3f}}}",
        f"\\newcommand{{\\minDskillStd}}{{{min_dskill_std:.3f}}}",
        f"\\newcommand{{\\minDskillBinLo}}{{{min_bin_lo:.2f}}}",
        f"\\newcommand{{\\minDskillBinHi}}{{{min_bin_hi:.2f}}}",
        f"\\newcommand{{\\minDskillTau}}{{{min_tau:.1f}}}",
        f"\\newcommand{{\\bestDmaeBinLo}}{{{best_mae_bin_lo:.2f}}}",
        f"\\newcommand{{\\bestDmaeBinHi}}{{{best_mae_bin_hi:.2f}}}",
        f"\\newcommand{{\\bestDmaeTau}}{{{best_mae_tau:.1f}}}",
        f"\\newcommand{{\\bestDmae}}{{{best_mae_val:+.4f}}}",
        f"\\newcommand{{\\bestDmaeStd}}{{{best_mae_std:.4f}}}",
        f"\\newcommand{{\\maxDmae}}{{{max_dmae:+.6f}}}",
    ]
    for m in macros:
        print(m)


if __name__ == '__main__':
    main()
