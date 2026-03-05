"""
04_validate_u.py - Validate CFD probe velocity against lab data
================================================================
Computes metrics and generates 4 PDF figures:
  1. Time series overlay (full + zoomed)
  2. PSD overlay (Welch)
  3. ACF overlay (lags 0-50)
  4. Velocity histogram overlay

Stop conditions (Phase 1 complete when ALL pass):
  - Mean(u): within +/-10% of lab
  - Std(u): within +/-10-15% of lab (use 15%)
  - Correlation r >= 0.85
  - PSD mismatch score (reported, lower is better)
  - ACF(1..10): same sign and comparable magnitude

All metrics computed on 10 Hz signals AFTER 4 Hz lowpass + resample,
with first 30s discarded as spin-up transient.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

CFD_ROOT = Path(__file__).resolve().parent.parent

# --- Configuration ---
SPINUP_DISCARD_S = 30.0   # Discard first N seconds as spin-up
ACF_MAX_LAG = 50           # Max lag for ACF plot
PSD_NPERSEG = 256          # Welch nperseg
DT = 0.1                   # 10 Hz


def compute_acf(x, max_lag):
    """Compute normalized autocorrelation function."""
    x = x - np.mean(x)
    n = len(x)
    acf = np.correlate(x, x, mode='full')
    acf = acf[n-1:]  # positive lags only
    acf = acf / acf[0]  # normalize
    return acf[:max_lag + 1]


def compute_psd_mismatch(f_lab, psd_lab, f_cfd, psd_cfd, f_max=4.0):
    """
    Compute integrated |log10(PSD_cfd / PSD_lab)| over 0 to f_max Hz.
    Interpolate CFD PSD onto lab frequency grid.
    """
    mask = (f_lab > 0) & (f_lab <= f_max)
    f_common = f_lab[mask]
    psd_lab_sel = psd_lab[mask]
    psd_cfd_interp = np.interp(f_common, f_cfd, psd_cfd)

    # Avoid log of zero
    eps = 1e-30
    log_ratio = np.abs(np.log10((psd_cfd_interp + eps) / (psd_lab_sel + eps)))
    df = np.mean(np.diff(f_common)) if len(f_common) > 1 else 1.0
    score = np.trapz(log_ratio, f_common)

    return score


def main():
    print("=" * 60)
    print("VALIDATE CFD PROBE VELOCITY vs LAB DATA")
    print(f"  Spin-up discard: {SPINUP_DISCARD_S}s")
    print("=" * 60)

    # Load lab data (unmodified reference)
    lab_path = CFD_ROOT / "lab_data" / "u_lab_10hz.csv"
    df_lab = pd.read_csv(lab_path)
    t_lab = df_lab['time'].values
    u_lab = df_lab['velocity'].values
    print(f"\nLab data: N={len(t_lab)}, t=[{t_lab[0]:.2f}, {t_lab[-1]:.2f}]s")

    # Load processed CFD data
    cfd_path = CFD_ROOT / "outputs" / "probe_u_10hz.csv"
    df_cfd = pd.read_csv(cfd_path)
    t_cfd = df_cfd['time'].values
    u_cfd = df_cfd['velocity'].values
    print(f"CFD data: N={len(t_cfd)}, t=[{t_cfd[0]:.2f}, {t_cfd[-1]:.2f}]s")

    # Find common time window (after spin-up discard)
    t_start = max(t_lab[0], t_cfd[0]) + SPINUP_DISCARD_S
    t_end = min(t_lab[-1], t_cfd[-1])
    print(f"\nCommon window: [{t_start:.2f}, {t_end:.2f}]s "
          f"(duration: {t_end - t_start:.1f}s)")

    # Interpolate both to common 10 Hz grid
    t_common = np.arange(t_start, t_end, DT)
    u_lab_c = np.interp(t_common, t_lab, u_lab)
    u_cfd_c = np.interp(t_common, t_cfd, u_cfd)
    n_common = len(t_common)
    print(f"Common grid: N={n_common}")

    # --- Metrics ---
    mean_lab = np.mean(u_lab_c)
    mean_cfd = np.mean(u_cfd_c)
    std_lab = np.std(u_lab_c)
    std_cfd = np.std(u_cfd_c)
    corr = np.corrcoef(u_lab_c, u_cfd_c)[0, 1]
    rmse = np.sqrt(np.mean((u_lab_c - u_cfd_c)**2))

    mean_err_pct = abs(mean_cfd - mean_lab) / abs(mean_lab) * 100
    std_err_pct = abs(std_cfd - std_lab) / abs(std_lab) * 100

    # ACF
    acf_lab = compute_acf(u_lab_c, ACF_MAX_LAG)
    acf_cfd = compute_acf(u_cfd_c, ACF_MAX_LAG)

    # ACF sign check (lags 1-10)
    acf_sign_match = all(
        np.sign(acf_lab[i]) == np.sign(acf_cfd[i]) for i in range(1, 11))

    # PSD
    f_lab, psd_lab = welch(u_lab_c, fs=1.0/DT, nperseg=PSD_NPERSEG)
    f_cfd, psd_cfd = welch(u_cfd_c, fs=1.0/DT, nperseg=PSD_NPERSEG)
    psd_mismatch = compute_psd_mismatch(f_lab, psd_lab, f_cfd, psd_cfd)

    # Cross-correlation for optimal lag (mean-subtracted + normalized)
    max_search_lag = int(30.0 / DT)  # search up to 30s
    u_lab_z = u_lab_c - np.mean(u_lab_c)
    u_cfd_z = u_cfd_c - np.mean(u_cfd_c)
    xcorr = np.correlate(u_lab_z, u_cfd_z, mode='full')
    xcorr = xcorr / (np.std(u_lab_c) * np.std(u_cfd_c) * n_common)
    mid = n_common - 1
    # Search both directions: negative lag (CFD leads) and positive (CFD lags)
    search_lo = max(0, mid - max_search_lag)
    search_hi = min(len(xcorr), mid + max_search_lag + 1)
    search_range = xcorr[search_lo:search_hi]
    best_idx = np.argmax(search_range)
    optimal_lag_idx = best_idx - (mid - search_lo)
    optimal_lag_s = optimal_lag_idx * DT
    peak_xcorr = float(search_range[best_idx])

    # --- Stop conditions ---
    pass_mean = mean_err_pct <= 10.0
    pass_std = std_err_pct <= 15.0
    pass_corr = corr >= 0.85
    pass_acf_sign = acf_sign_match
    all_pass = pass_mean and pass_std and pass_corr and pass_acf_sign

    print(f"\n--- METRICS ---")
    print(f"  Mean:  lab={mean_lab:.4f}, cfd={mean_cfd:.4f}, "
          f"err={mean_err_pct:.1f}% {'PASS' if pass_mean else 'FAIL'}")
    print(f"  Std:   lab={std_lab:.4f}, cfd={std_cfd:.4f}, "
          f"err={std_err_pct:.1f}% {'PASS' if pass_std else 'FAIL'}")
    print(f"  Corr:  r={corr:.4f} {'PASS' if pass_corr else 'FAIL'}")
    print(f"  RMSE:  {rmse:.4f} m/s")
    print(f"  ACF sign (1-10): {'PASS' if pass_acf_sign else 'FAIL'}")
    print(f"  PSD mismatch: {psd_mismatch:.3f}")
    print(f"  Optimal lag: {optimal_lag_s:.1f}s (peak xcorr={peak_xcorr:.4f})")
    print(f"\n  ALL PASS: {'YES' if all_pass else 'NO'}")

    # --- Save metrics JSON ---
    metrics = {
        'spinup_discard_s': SPINUP_DISCARD_S,
        'common_window': [float(t_start), float(t_end)],
        'n_common': n_common,
        'mean_lab': float(mean_lab),
        'mean_cfd': float(mean_cfd),
        'mean_err_pct': float(mean_err_pct),
        'std_lab': float(std_lab),
        'std_cfd': float(std_cfd),
        'std_err_pct': float(std_err_pct),
        'correlation': float(corr),
        'rmse': float(rmse),
        'psd_mismatch_0_4hz': float(psd_mismatch),
        'acf_sign_match_1_10': bool(acf_sign_match),
        'optimal_lag_s': float(optimal_lag_s),
        'peak_xcorr': float(peak_xcorr),
        'acf_lab_1_10': [float(acf_lab[i]) for i in range(1, 11)],
        'acf_cfd_1_10': [float(acf_cfd[i]) for i in range(1, 11)],
        'stop_conditions': {
            'mean_within_10pct': bool(pass_mean),
            'std_within_15pct': bool(pass_std),
            'correlation_ge_085': bool(pass_corr),
            'acf_sign_match': bool(pass_acf_sign),
            'all_pass': bool(all_pass),
        },
    }

    out_dir = CFD_ROOT / "outputs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote: {json_path}")

    # ========================================================
    # FIGURES
    # ========================================================

    # --- Fig 1: Time series overlay ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Full
    axes[0].plot(t_common, u_lab_c, 'b-', alpha=0.7, linewidth=0.5, label='Lab')
    axes[0].plot(t_common, u_cfd_c, 'r-', alpha=0.7, linewidth=0.5, label='CFD')
    axes[0].set_ylabel('u (m/s)')
    axes[0].set_title(f'Velocity: Full Window (r={corr:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoomed (50s window near middle)
    t_mid = (t_start + t_end) / 2
    zoom_mask = (t_common >= t_mid - 25) & (t_common <= t_mid + 25)
    axes[1].plot(t_common[zoom_mask], u_lab_c[zoom_mask], 'b-', linewidth=1.0, label='Lab')
    axes[1].plot(t_common[zoom_mask], u_cfd_c[zoom_mask], 'r-', linewidth=1.0, label='CFD')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('u (m/s)')
    axes[1].set_title('Zoomed: 50s Window')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "fig_u_overlay.pdf"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {fig_path}")

    # --- Fig 2: PSD overlay ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(f_lab, psd_lab, 'b-', linewidth=1.0, label='Lab')
    ax.semilogy(f_cfd, psd_cfd, 'r-', linewidth=1.0, label='CFD')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (m^2/s^2/Hz)')
    ax.set_title(f'Welch PSD (mismatch score: {psd_mismatch:.3f})')
    ax.set_xlim([0, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "fig_psd_overlay.pdf"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {fig_path}")

    # --- Fig 3: ACF overlay ---
    lags = np.arange(ACF_MAX_LAG + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags * DT, acf_lab, 'b-o', markersize=3, linewidth=1.0, label='Lab')
    ax.plot(lags * DT, acf_cfd, 'r-s', markersize=3, linewidth=1.0, label='CFD')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('ACF')
    ax.set_title('Autocorrelation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "fig_acf_overlay.pdf"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {fig_path}")

    # --- Fig 4: Histogram overlay ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(u_lab_c.min(), u_cfd_c.min()),
        max(u_lab_c.max(), u_cfd_c.max()),
        50)
    ax.hist(u_lab_c, bins=bins, alpha=0.5, density=True, label='Lab', color='blue')
    ax.hist(u_cfd_c, bins=bins, alpha=0.5, density=True, label='CFD', color='red')
    ax.set_xlabel('u (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('Velocity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "fig_hist_overlay.pdf"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {fig_path}")

    # --- Report markdown ---
    report_lines = [
        "# Phase 1 Validation Report: CFD Flow Twin",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- Spin-up discard: {SPINUP_DISCARD_S}s",
        f"- Common window: [{t_start:.2f}, {t_end:.2f}]s "
        f"({t_end - t_start:.1f}s duration)",
        f"- Common grid points: {n_common}",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Lab | CFD | Error | Pass? |",
        "|--------|-----|-----|-------|-------|",
        f"| Mean (m/s) | {mean_lab:.4f} | {mean_cfd:.4f} | "
        f"{mean_err_pct:.1f}% | {'PASS' if pass_mean else 'FAIL'} |",
        f"| Std (m/s) | {std_lab:.4f} | {std_cfd:.4f} | "
        f"{std_err_pct:.1f}% | {'PASS' if pass_std else 'FAIL'} |",
        f"| Correlation | - | {corr:.4f} | - | "
        f"{'PASS' if pass_corr else 'FAIL'} |",
        f"| RMSE (m/s) | - | {rmse:.4f} | - | - |",
        f"| ACF sign (1-10) | - | - | - | "
        f"{'PASS' if pass_acf_sign else 'FAIL'} |",
        f"| PSD mismatch | - | {psd_mismatch:.3f} | - | (lower=better) |",
        "",
        f"**Optimal lag (xcorr):** {optimal_lag_s:.1f}s",
        "",
        f"## Overall: {'ALL PASS' if all_pass else 'NOT ALL PASS'}",
        "",
        "## ACF Comparison (lags 1-10)",
        "",
        "| Lag | Lab | CFD |",
        "|-----|-----|-----|",
    ]
    for i in range(1, 11):
        report_lines.append(
            f"| {i} ({i*DT:.1f}s) | {acf_lab[i]:.4f} | {acf_cfd[i]:.4f} |")

    report_lines.extend([
        "",
        "## Figures",
        "",
        "- `fig_u_overlay.pdf`: Time series overlay (full + zoomed)",
        "- `fig_psd_overlay.pdf`: Welch PSD comparison",
        "- `fig_acf_overlay.pdf`: ACF comparison (lags 0-50)",
        "- `fig_hist_overlay.pdf`: Velocity distribution comparison",
    ])

    report_path = out_dir / "report_phase1.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Wrote: {report_path}")

    print("\n" + "=" * 60)
    print(f"VALIDATION COMPLETE - {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
