#!/usr/bin/env python
"""
audit_flow_input_match.py - Flow input tuning audit (non-destructive)
=======================================================================================
Diagnostic script to verify whether the Virtual Lab forcing was calibrated to match
probe statistics (empirical tuning) or only regime-anchored (statistical bounds).

HARD CONSTRAINTS (non-destructive):
- Read-only: loads existing CSVs, no overwrite to outputs/ used in manuscript.
- Writes only to: virtual_lab_v1/outputs/audit_flow_match/
- No retraining, no VL regeneration.
- Deterministic: fixed random seed.

Usage:
  python -u virtual_lab_v1/tools/audit_flow_input_match.py --dry_run
  python -u virtual_lab_v1/tools/audit_flow_input_match.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats, signal

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
#  SETUP
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
SWEEP_DIR = ROOT / "outputs" / "sweep_v2"
FORCING_DIR = SWEEP_DIR / "forcing"
AUDIT_DIR = ROOT / "outputs" / "audit_flow_match"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
#  FIND & LOAD SERIES
# =============================================================================

def find_probe_series(dry_run=False):
    """
    Load empirical probe velocity at y=0.040 m from condition 0.
    Returns (path_str, series_u, dt_sec).
    """
    # Use condition_000 as representative baseline
    probe_path = DATASETS_DIR / "condition_000" / "u_probes_10hz.csv"

    if not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")

    if dry_run:
        print(f"  [DRY RUN] Would load probe: {probe_path}")
        return (str(probe_path), None, None)

    print(f"  Loading probe: {probe_path}")
    df = pd.read_csv(probe_path)
    u_y040 = df['u_y040'].values
    dt = 0.1  # 10 Hz

    print(f"    Loaded {len(u_y040)} samples (dt={dt}s, duration={len(u_y040)*dt:.1f}s)")
    return (str(probe_path), u_y040, dt)


def find_vl_forcing_series(dry_run=False):
    """
    Load VL forcing (baseline variant, used in sweep).
    Returns (path_str, series_u, dt_sec).
    """
    forcing_path = FORCING_DIR / "variant_baseline_u_raw.csv"

    # If baseline not found, try alternate name
    if not forcing_path.exists():
        variants = list(FORCING_DIR.glob("variant_baseline*.csv"))
        if variants:
            forcing_path = variants[0]

    if not forcing_path.exists():
        raise FileNotFoundError(f"VL forcing file not found in: {FORCING_DIR}")

    if dry_run:
        print(f"  [DRY RUN] Would load VL forcing: {forcing_path}")
        return (str(forcing_path), None, None)

    print(f"  Loading VL forcing: {forcing_path}")
    df = pd.read_csv(forcing_path)

    # Determine column name (might be 'u' or other)
    u_col = [c for c in df.columns if 'u' in c.lower()][0]
    u_vl = df[u_col].values

    # VL forcing is typically at higher sampling rate; downsample to match probe (10 Hz)
    # For now, assume raw forcing is at same 10 Hz
    dt = 0.1

    print(f"    Loaded {len(u_vl)} samples (dt~{dt}s, duration~{len(u_vl)*dt:.1f}s)")
    return (str(forcing_path), u_vl, dt)


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def extract_features(u, dt, name=""):
    """
    Extract comparable statistical and spectral features from velocity series.

    Returns dict with scalar metrics.
    """
    u = np.asarray(u).flatten()
    du = np.diff(u)

    feat = {}
    feat['N'] = len(u)
    feat['mean_u'] = float(np.mean(u))
    feat['std_u'] = float(np.std(u))
    feat['skew_u'] = float(stats.skew(u))
    feat['kurt_u'] = float(stats.kurtosis(u))

    feat['mean_du'] = float(np.mean(du))
    feat['std_du'] = float(np.std(du))
    feat['skew_du'] = float(stats.skew(du))
    feat['kurt_du'] = float(stats.kurtosis(du))

    feat['p95_u'] = float(np.percentile(u, 95))
    feat['p99_u'] = float(np.percentile(u, 99))
    feat['p95_du'] = float(np.percentile(np.abs(du), 95))
    feat['p99_du'] = float(np.percentile(np.abs(du), 99))

    # Zero-crossing rate
    u_dem = u - np.mean(u)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(u_dem))))
    feat['zcr'] = float(zero_crossings / len(u))

    # ACF at specific lags (in seconds)
    acf_lags_s = [1, 5, 10, 50]
    for lag_s in acf_lags_s:
        lag_idx = max(1, int(lag_s / dt))
        if lag_idx < len(du):
            acf_val = np.correlate(du - np.mean(du), du - np.mean(du),
                                    mode='full')[len(du) - 1 + lag_idx] / (feat['std_du']**2 * len(du))
            feat[f'acf_du_lag{lag_s}s'] = float(acf_val)

    # Event rate: |du| > p95(|du|)
    q95_du = np.percentile(np.abs(du), 95)
    events = np.sum(np.abs(du) > q95_du)
    feat['events_per_min'] = float(events / (len(u) * dt / 60.0))

    # Intermittency: fraction of time |du| > p90
    q90_du = np.percentile(np.abs(du), 90)
    feat['intermittency_q90'] = float(np.sum(np.abs(du) > q90_du) / len(du))

    print(f"    {name}: mean_u={feat['mean_u']:.4f}, std_u={feat['std_u']:.4f}, "
          f"events/min={feat['events_per_min']:.2f}, interm={feat['intermittency_q90']:.3f}")

    return feat


# =============================================================================
#  STATISTICAL COMPARISON
# =============================================================================

def compare_features(feat_probe, feat_vl):
    """
    Compute normalized differences and match score.
    Returns dict with comparison metrics.
    """
    comp = {}

    # Normalized differences for key stats
    for key in ['mean_u', 'std_u', 'std_du', 'p95_du', 'p99_du', 'events_per_min']:
        if feat_probe[key] != 0:
            diff = (feat_vl[key] - feat_probe[key]) / abs(feat_probe[key])
        else:
            diff = 0.0
        comp[f'norm_diff_{key}'] = float(diff)

    # Ratio for more intuitive reading
    comp['std_ratio'] = float(feat_vl['std_u'] / feat_probe['std_u']) if feat_probe['std_u'] > 0 else np.nan
    comp['std_du_ratio'] = float(feat_vl['std_du'] / feat_probe['std_du']) if feat_probe['std_du'] > 0 else np.nan
    comp['event_rate_ratio'] = float(feat_vl['events_per_min'] / feat_probe['events_per_min']) \
        if feat_probe['events_per_min'] > 0 else np.nan

    return comp


def compute_match_score(feat_probe, feat_vl, comp):
    """
    Simple heuristic match score based on normalized differences.
    Returns float score and bool pass/fail.
    """
    # Collect key differences
    key_diffs = [
        abs(comp['norm_diff_std_u']),
        abs(comp['norm_diff_std_du']),
        abs(comp['norm_diff_p95_du']),
    ]

    # Weighted average difference
    mean_diff = float(np.mean(key_diffs))

    # PASS/FAIL thresholds (loose, diagnostic only)
    pass_std = abs(comp['std_ratio'] - 1.0) < 0.15
    pass_event = 0.7 <= comp['event_rate_ratio'] <= 1.3
    pass_mean_diff = mean_diff < 0.20

    score = {
        'mean_normalized_diff': float(mean_diff),
        'pass_std': bool(pass_std),
        'pass_event_rate': bool(pass_event),
        'pass_overall': bool(pass_std and pass_event and pass_mean_diff),
    }

    return score


# =============================================================================
#  PLOTTING (OPTIONAL)
# =============================================================================

def make_summary_figure(u_probe, u_vl, dt_probe, dt_vl, feat_p, feat_v, comp, output_pdf):
    """
    Create 2x2 diagnostic figure (time, histogram, PSD, ACF).
    """
    if not HAS_MATPLOTLIB:
        print("    [Matplotlib not available; skipping figure]")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Time snippet (first 300 seconds)
    ax = axes[0, 0]
    n_show = min(3000, len(u_probe))  # 300 s at 10 Hz
    t_probe = np.arange(n_show) * dt_probe
    t_vl = np.arange(n_show) * dt_vl

    u_p_norm = (u_probe[:n_show] - feat_p['mean_u']) / feat_p['std_u']
    u_v_norm = (u_vl[:n_show] - feat_v['mean_u']) / feat_v['std_u']

    ax.plot(t_probe, u_p_norm, label='Probe (empirical)', alpha=0.7, linewidth=0.8)
    ax.plot(t_vl, u_v_norm, label='VL forcing', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized velocity')
    ax.set_title('(a) Time snippet (normalized)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Histogram of du
    ax = axes[0, 1]
    du_p = np.diff(u_probe)
    du_v = np.diff(u_vl)
    ax.hist(du_p, bins=50, alpha=0.5, label='Probe', color='blue', density=True)
    ax.hist(du_v, bins=50, alpha=0.5, label='VL', color='orange', density=True)
    ax.set_xlabel('Δu (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('(b) Histogram of velocity increment')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Power Spectral Density
    ax = axes[1, 0]
    f_p, pxx_p = signal.welch(u_probe, fs=1/dt_probe, nperseg=4096)
    f_v, pxx_v = signal.welch(u_vl, fs=1/dt_vl, nperseg=4096)
    ax.loglog(f_p[1:], pxx_p[1:], label='Probe', alpha=0.7)
    ax.loglog(f_v[1:], pxx_v[1:], label='VL', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (m²/s²/Hz)')
    ax.set_title('(c) Power Spectral Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # (d) ACF
    ax = axes[1, 1]
    max_lag = 100
    acf_p = np.correlate(du_p - np.mean(du_p), du_p - np.mean(du_p), mode='full')[
        len(du_p) - 1:len(du_p) - 1 + max_lag]
    acf_p = acf_p / acf_p[0]
    acf_v = np.correlate(du_v - np.mean(du_v), du_v - np.mean(du_v), mode='full')[
        len(du_v) - 1:len(du_v) - 1 + max_lag]
    acf_v = acf_v / acf_v[0]

    lags_s = np.arange(max_lag) * dt_probe
    ax.plot(lags_s, acf_p, label='Probe', alpha=0.7, marker='o', markersize=2)
    ax.plot(lags_s, acf_v, label='VL', alpha=0.7, marker='s', markersize=2)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('ACF')
    ax.set_title('(d) Autocorrelation of Δu')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_pdf, dpi=150, bbox_inches='tight')
    print(f"    Saved figure: {output_pdf}")
    plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Flow input tuning audit (non-destructive)")
    parser.add_argument('--dry_run', action='store_true',
                        help='Only discover file paths, do not compute')
    args = parser.parse_args()

    print("="*80)
    print("FLOW INPUT TUNING AUDIT (virtual_lab_v1)")
    print("="*80)
    print()

    # Create output directory
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # ===========================================
    # STEP 1: Load series
    # ===========================================
    print("STEP 1: Locate & load timeseries")
    print("-" * 80)

    try:
        probe_path, u_probe, dt_probe = find_probe_series(dry_run=args.dry_run)
        print(f"  Probe path: {probe_path}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1

    try:
        vl_path, u_vl, dt_vl = find_vl_forcing_series(dry_run=args.dry_run)
        print(f"  VL path: {vl_path}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] Completed successfully. Exiting before computation.")
        return 0

    print()

    # ===========================================
    # STEP 2: Extract features
    # ===========================================
    print("STEP 2: Extract features")
    print("-" * 80)

    feat_probe = extract_features(u_probe, dt_probe, name="Probe")
    feat_vl = extract_features(u_vl, dt_vl, name="VL")

    print()

    # ===========================================
    # STEP 3: Compare features
    # ===========================================
    print("STEP 3: Compare features")
    print("-" * 80)

    comp = compare_features(feat_probe, feat_vl)

    print(f"  Std ratio (VL/probe): {comp['std_ratio']:.3f}")
    print(f"  Event rate ratio (VL/probe): {comp['event_rate_ratio']:.3f}")
    print(f"  Mean diff in p95_du: {comp['norm_diff_p95_du']:.3f}")

    score = compute_match_score(feat_probe, feat_vl, comp)
    print()
    print(f"  Match score: {score['mean_normalized_diff']:.4f}")
    print(f"    Pass std check: {score['pass_std']}")
    print(f"    Pass event rate check: {score['pass_event_rate']}")
    print(f"    Pass overall: {score['pass_overall']}")

    print()

    # ===========================================
    # STEP 4: Make figure
    # ===========================================
    print("STEP 4: Generate diagnostics figure")
    print("-" * 80)

    output_pdf = AUDIT_DIR / "flow_match_summary.pdf"
    make_summary_figure(u_probe, u_vl, dt_probe, dt_vl, feat_probe, feat_vl, comp, output_pdf)

    print()

    # ===========================================
    # STEP 5: Write report
    # ===========================================
    print("STEP 5: Write JSON report")
    print("-" * 80)

    report = {
        'audit_type': 'flow_input_tuning',
        'probe_file': probe_path,
        'vl_forcing_file': vl_path,
        'probe_features': feat_probe,
        'vl_features': feat_vl,
        'comparison': comp,
        'match_score': score,
        'interpretation': (
            'Likely probe-calibrated (empirical tuning)' if score['pass_overall']
            else 'Likely regime-anchored only (statistical bounds tuning)'
        ),
    }

    report_json = AUDIT_DIR / "report.json"
    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Saved: {report_json}")
    print()

    # ===========================================
    # SUMMARY
    # ===========================================
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print(f"Compared:")
    print(f"  Probe (empirical):  {probe_path}")
    print(f"  VL forcing:         {vl_path}")
    print()

    if score['pass_overall']:
        print("VERDICT: Likely PROBE-CALIBRATED (empirical tuning)")
        print()
        print("Evidence:")
        print(f"  - Standard deviation ratio:  {comp['std_ratio']:.3f} (target ~1.0, pass: {score['pass_std']})")
        print(f"  - Event rate ratio:          {comp['event_rate_ratio']:.3f} (target 0.7-1.3, pass: {score['pass_event_rate']})")
        print(f"  - Mean normalized difference: {score['mean_normalized_diff']:.4f} (target <0.20)")
        print()
        print("Implication: The VL forcing variants appear to be tuned to match the empirical")
        print("probe statistics, not just bounded by regime-anchoring rules.")
    else:
        print("VERDICT: Likely REGIME-ANCHORED ONLY (statistical bounds tuning)")
        print()
        print("Evidence:")
        print(f"  - Standard deviation ratio:  {comp['std_ratio']:.3f} (target ~1.0, pass: {score['pass_std']})")
        print(f"  - Event rate ratio:          {comp['event_rate_ratio']:.3f} (target 0.7-1.3, pass: {score['pass_event_rate']})")
        print(f"  - Mean normalized difference: {score['mean_normalized_diff']:.4f} (target <0.20)")
        print()
        print("Implication: The VL forcing variants appear to be constrained by regime anchors")
        print("(e.g., event rates, bounds) rather than tuned to empirical probe statistics.")

    print()
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
