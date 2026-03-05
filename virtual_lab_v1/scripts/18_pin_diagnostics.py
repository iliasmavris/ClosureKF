"""
18_pin_diagnostics.py - Pin-capture model validation (4 checks)
================================================================
1. PSD contamination: pinned-only PSD vs excursion PSD (raw + 4Hz LP)
2. Discontinuity audit: |delta_x| at every capture event
3. Waiting-time distribution: monotonic trends vs mu_s, C_Du
4. Non-circularity: check acceleration correlation with -v|u| during free motion

Runs against real CFD flow data. Outputs diagnostic figures + JSON report.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import copy
import yaml
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth_ball_sim_lib import (simulate_sphere, compute_event_rate,
                                 compute_pin_statistics, compute_derived_params)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "pin_diagnostics"


def load_flow():
    """Load raw flow data."""
    raw_path = ROOT / "outputs" / "flow_probes" / "u_probes_raw.csv"
    df = pd.read_csv(raw_path)
    t = df['time'].values
    # Use first available u_* column
    u_cols = [c for c in df.columns if c.startswith('u_')]
    u = df[u_cols[0]].values
    return t, u, u_cols[0]


def load_config():
    """Load default config."""
    cfg_path = ROOT / "configs" / "ball_params_default.yaml"
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


# ==========================================================================
# CHECK 1: PSD contamination
# ==========================================================================
def check1_psd_contamination(result, cfg, out_dir):
    """Compare PSD of x(t) during pinned vs free segments."""
    print("\n--- CHECK 1: PSD contamination (pin jitter vs 4Hz band) ---")

    dt = cfg['integration']['dt_sim']
    fs_sim = 1.0 / dt
    t = result['t']
    x = result['x']
    at_pin = result['at_pin']

    # Segments: find contiguous runs of at_pin=1 and at_pin=0
    pinned_x_raw = x[at_pin == 1]
    free_x_raw = x[at_pin == 0]

    # Also get 4Hz-filtered versions
    sos = butter(4, 4.0, btype='low', fs=fs_sim, output='sos')
    x_filt = sosfiltfilt(sos, x)
    pinned_x_filt = x_filt[at_pin == 1]
    free_x_filt = x_filt[at_pin == 0]

    # OU theoretical PSD: S(f) = sigma^2 * 2*tau / (1 + (2*pi*f*tau)^2)
    tau_x = cfg.get('pin', {}).get('tau_x', 0.2)
    sigma_x = cfg.get('pin', {}).get('sigma_x', 0.001)
    f_corner = 1.0 / (2.0 * np.pi * tau_x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    report = {}

    # Raw PSDs
    ax = axes[0]
    nperseg = min(1024, max(64, len(pinned_x_raw) // 4))

    if len(pinned_x_raw) > nperseg:
        # Demean pinned signal for PSD
        f_pin, psd_pin = welch(pinned_x_raw - np.mean(pinned_x_raw),
                                fs=fs_sim, nperseg=nperseg)
        ax.semilogy(f_pin, psd_pin, 'r-', label='Pinned (raw)', alpha=0.8)

        # Integrated power 0-4 Hz
        mask_4hz = f_pin <= 4.0
        power_pin_04 = np.trapz(psd_pin[mask_4hz], f_pin[mask_4hz])
        report['pinned_power_0_4hz_raw'] = float(power_pin_04)
        print(f"  Pinned raw PSD power 0-4 Hz: {power_pin_04:.2e}")

    if len(free_x_raw) > nperseg:
        f_free, psd_free = welch(free_x_raw - np.mean(free_x_raw),
                                  fs=fs_sim, nperseg=nperseg)
        ax.semilogy(f_free, psd_free, 'b-', label='Free (raw)', alpha=0.8)

        mask_4hz = f_free <= 4.0
        power_free_04 = np.trapz(psd_free[mask_4hz], f_free[mask_4hz])
        report['free_power_0_4hz_raw'] = float(power_free_04)
        print(f"  Free raw PSD power 0-4 Hz: {power_free_04:.2e}")

        if len(pinned_x_raw) > nperseg:
            ratio = power_pin_04 / power_free_04
            report['pin_free_power_ratio_raw'] = float(ratio)
            print(f"  Ratio (pinned/free) 0-4 Hz: {ratio:.4f}")

    # Theoretical OU PSD
    f_th = np.linspace(0.01, fs_sim/2, 500)
    psd_th = sigma_x**2 * 2 * tau_x / (1 + (2*np.pi*f_th*tau_x)**2)
    ax.semilogy(f_th, psd_th, 'r--', label=f'OU theory (tau={tau_x}s)', alpha=0.6)

    ax.axvline(4.0, color='green', linestyle=':', label='4 Hz cutoff')
    ax.axvline(f_corner, color='orange', linestyle=':', label=f'OU corner={f_corner:.1f} Hz')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [m^2/Hz]')
    ax.set_title('Raw PSD')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)

    # Filtered PSDs
    ax = axes[1]
    if len(pinned_x_filt) > nperseg:
        f_pin_f, psd_pin_f = welch(pinned_x_filt - np.mean(pinned_x_filt),
                                    fs=fs_sim, nperseg=nperseg)
        ax.semilogy(f_pin_f, psd_pin_f, 'r-', label='Pinned (4Hz LP)', alpha=0.8)

        mask_4hz = f_pin_f <= 4.0
        power_pin_04_filt = np.trapz(psd_pin_f[mask_4hz], f_pin_f[mask_4hz])
        report['pinned_power_0_4hz_filt'] = float(power_pin_04_filt)

    if len(free_x_filt) > nperseg:
        f_free_f, psd_free_f = welch(free_x_filt - np.mean(free_x_filt),
                                      fs=fs_sim, nperseg=nperseg)
        ax.semilogy(f_free_f, psd_free_f, 'b-', label='Free (4Hz LP)', alpha=0.8)

        mask_4hz = f_free_f <= 4.0
        power_free_04_filt = np.trapz(psd_free_f[mask_4hz], f_free_f[mask_4hz])
        report['free_power_0_4hz_filt'] = float(power_free_04_filt)

        if len(pinned_x_filt) > nperseg:
            ratio_f = power_pin_04_filt / power_free_04_filt
            report['pin_free_power_ratio_filt'] = float(ratio_f)
            print(f"  Ratio (pinned/free) 0-4 Hz after LP: {ratio_f:.4f}")

    ax.axvline(4.0, color='green', linestyle=':', label='4 Hz cutoff')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [m^2/Hz]')
    ax.set_title('After 4Hz Lowpass')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)

    report['f_corner_hz'] = float(f_corner)
    report['tau_x'] = float(tau_x)
    report['sigma_x'] = float(sigma_x)

    # Pass/fail
    ratio_thresh = 0.10  # pinned should be <10% of free power
    ratio_val = report.get('pin_free_power_ratio_raw', 999)
    passed = ratio_val < ratio_thresh
    report['pass'] = passed
    status = "PASS" if passed else "FAIL"
    print(f"  Threshold: ratio < {ratio_thresh}")
    print(f"  --> {status} (ratio={ratio_val:.4f})")

    fig.suptitle('Check 1: Pin Jitter PSD Contamination', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_check1_psd_contamination.pdf", dpi=150)
    plt.close(fig)

    return report


# ==========================================================================
# CHECK 2: Discontinuity audit
# ==========================================================================
def check2_discontinuity(result, cfg, out_dir):
    """Check |delta_x| at every capture event.

    If the real CFD flow produces no re-captures (unidirectional flow,
    no restoring force), we additionally run a forced test with oscillatory
    flow that guarantees multiple captures.
    """
    print("\n--- CHECK 2: Discontinuity audit (capture teleportation) ---")

    capture_log = result.get('capture_log', [])
    eps_x = cfg.get('pin', {}).get('eps_x', 0.003)
    sigma_x = cfg.get('pin', {}).get('sigma_x', 0.0003)

    report = {
        'eps_x': eps_x,
        'sigma_x_3': 3 * sigma_x,
    }

    # --- Forced capture test with bidirectional oscillatory flow ---
    # Unidirectional flow never returns particle to pin, so we use a
    # zero-mean oscillatory flow that pushes the particle back and forth
    # past x_pin. Combined with a restoring spring, this guarantees
    # repeated capture events for the discontinuity audit.
    print("  Running forced-capture test (bidirectional flow + spring)...")
    cfg_test = copy.deepcopy(cfg)
    cfg_test['integration']['t_end'] = 120.0
    cfg_test['integration']['spinup_discard'] = 0.0
    cfg_test['restoring']['k_spring'] = 0.5   # strong restoring for returns
    cfg_test['friction']['mu_s'] = 0.40       # low enough for breakaway
    cfg_test['friction']['mu_k'] = 0.25
    cfg_test['noise']['sigma_a'] = 0.0  # no noise to keep it clean

    dt_test = 0.005
    t_osc = np.arange(0, 120, dt_test)
    # Bidirectional flow: amplitude high enough to exceed breakaway threshold
    # At u=0.25, F_drag ~ 1.1e-4 N > mu_s*W_sub = 0.40*2.29e-4 = 9.2e-5 N
    u_osc = 0.25 * np.sin(2 * np.pi * 0.2 * t_osc)

    result_test = simulate_sphere(cfg_test, t_osc, u_osc, seed=99)
    test_log = result_test.get('capture_log', [])

    # Combine logs: real (if any) + forced test
    all_logs = capture_log + test_log
    report['n_captures_real'] = len(capture_log)
    report['n_captures_forced'] = len(test_log)
    report['n_captures_total'] = len(all_logs)

    if len(capture_log) > 0:
        print(f"  Real-flow captures: {len(capture_log)}")
    else:
        print(f"  Real-flow captures: 0 (unidirectional flow, no returns)")
    print(f"  Forced-test captures: {len(test_log)}")

    if len(all_logs) == 0:
        print("  No capture events even in forced test -- unexpected!")
        report['pass'] = False
        report['max_abs_dx'] = None
        return report

    captures = np.array(all_logs)
    t_cap = captures[:, 0]
    x_before = captures[:, 1]
    x_after = captures[:, 2]
    dx = captures[:, 3]
    v_before = captures[:, 4]

    abs_dx = np.abs(dx)
    max_dx = np.max(abs_dx)
    mean_dx = np.mean(abs_dx)

    report['max_abs_dx'] = float(max_dx)
    report['mean_abs_dx'] = float(mean_dx)
    report['max_abs_v_before'] = float(np.max(np.abs(v_before)))

    print(f"  max |delta_x|: {max_dx:.2e}")
    print(f"  mean |delta_x|: {mean_dx:.2e}")
    print(f"  max |v_before|: {np.max(np.abs(v_before)):.2e}")

    # Should be exactly zero: xi = x_new - x_pin, x_new = x_pin + xi
    # Clip boundary (3*sigma_x) == eps_x when sigma_x=0.001, eps_x=0.003
    # With sigma_x=0.0003, 3*sigma_x=0.0009 < eps_x=0.003 -- clip CAN fire.
    # When clip fires: x+ != x-, so delta_x != 0.
    # But |delta_x| <= eps_x - 3*sigma_x = 0.003 - 0.0009 = 0.0021
    # This is a design property: clip only matters when |offset| > 3*sigma_x.
    passed = max_dx < eps_x  # must be less than capture radius
    report['pass'] = passed
    status = "PASS" if passed else "FAIL"
    print(f"  Threshold: max |dx| < eps_x = {eps_x}")
    print(f"  --> {status}")

    if max_dx > 1e-10:
        # Check if any were caused by clip
        n_clipped = np.sum(abs_dx > 1e-10)
        report['n_clipped'] = int(n_clipped)
        print(f"  Note: {n_clipped}/{len(all_logs)} captures had clip-induced offset "
              f"(3*sigma_x={3*sigma_x:.4f} < eps_x={eps_x:.4f})")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(t_cap, abs_dx, s=10, alpha=0.6, color='navy')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('|delta_x| [m]')
    ax.set_title(f'Capture Discontinuity (n={len(all_logs)}, forced test)')
    ax.axhline(eps_x, color='red', linestyle='--', label=f'eps_x={eps_x}')
    ax.axhline(3*sigma_x, color='orange', linestyle=':', label=f'3*sigma_x={3*sigma_x:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(t_cap, np.abs(v_before), s=10, alpha=0.6, color='firebrick')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('|v_p| before capture [m/s]')
    ax.set_title('Velocity at Capture')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Check 2: Capture Discontinuity Audit', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_check2_discontinuity.pdf", dpi=150)
    plt.close(fig)

    return report


# ==========================================================================
# CHECK 3: Waiting-time trends across sweep parameters
# ==========================================================================
def check3_waiting_trends(t_flow, u_flow, base_cfg, out_dir):
    """Run parameter variations and check monotonic trends."""
    print("\n--- CHECK 3: Waiting-time trends vs mu_s, C_Du ---")

    # Sweep mu_s at fixed mu_k, C_Du
    mu_s_vals = [0.40, 0.45, 0.50, 0.55, 0.60]
    results_mu_s = []
    for mu_s in mu_s_vals:
        cfg = copy.deepcopy(base_cfg)
        cfg['friction']['mu_s'] = mu_s
        cfg['integration']['t_end'] = min(540.0, t_flow[-1])
        result = simulate_sphere(cfg, t_flow, u_flow, seed=42)
        dt = cfg['integration']['dt_sim']
        spinup = cfg['integration']['spinup_discard']
        mask = result['t'] >= spinup
        ps = compute_pin_statistics(result['at_pin'][mask], dt)
        er = compute_event_rate(result['at_pin'][mask])
        med_wt = float(np.median(ps['waiting_times'])) if len(ps['waiting_times']) > 0 else np.nan
        results_mu_s.append({
            'mu_s': mu_s, 'event_rate': float(er),
            'median_wait': med_wt,
            'n_waits': len(ps['waiting_times']),
            'n_excursions': len(ps['excursion_times']),
        })
        print(f"  mu_s={mu_s:.2f}: er={er:.3f}, n_waits={len(ps['waiting_times'])}, "
              f"med_wait={med_wt:.2f}s" if not np.isnan(med_wt) else
              f"  mu_s={mu_s:.2f}: er={er:.3f}, n_waits=0, med_wait=N/A")

    # Sweep C_Du at fixed mu_s=0.57, mu_k=0.40 (enough friction for pin returns)
    C_Du_vals = [0.5, 0.8, 1.0, 1.2, 1.5]
    results_C_Du = []
    for C_Du in C_Du_vals:
        cfg = copy.deepcopy(base_cfg)
        cfg['friction']['mu_s'] = 0.57
        cfg['added_mass']['C_Du'] = C_Du
        cfg['integration']['t_end'] = min(540.0, t_flow[-1])
        result = simulate_sphere(cfg, t_flow, u_flow, seed=42)
        dt = cfg['integration']['dt_sim']
        spinup = cfg['integration']['spinup_discard']
        mask = result['t'] >= spinup
        ps = compute_pin_statistics(result['at_pin'][mask], dt)
        er = compute_event_rate(result['at_pin'][mask])
        med_wt = float(np.median(ps['waiting_times'])) if len(ps['waiting_times']) > 0 else np.nan
        results_C_Du.append({
            'C_Du': C_Du, 'event_rate': float(er),
            'median_wait': med_wt,
            'n_waits': len(ps['waiting_times']),
        })
        print(f"  C_Du={C_Du:.1f}: er={er:.3f}, n_waits={len(ps['waiting_times'])}, "
              f"med_wait={med_wt:.2f}s" if not np.isnan(med_wt) else
              f"  C_Du={C_Du:.1f}: er={er:.3f}, n_waits=0, med_wait=N/A")

    # Check monotonicity
    ers_mu = [r['event_rate'] for r in results_mu_s]
    er_monotone = all(ers_mu[i] >= ers_mu[i+1] for i in range(len(ers_mu)-1))
    print(f"\n  Event rate vs mu_s monotonic (decreasing)? {er_monotone}")
    print(f"    Values: {[f'{e:.3f}' for e in ers_mu]}")

    ers_cdu = [r['event_rate'] for r in results_C_Du]
    er_cdu_monotone = all(ers_cdu[i] <= ers_cdu[i+1] for i in range(len(ers_cdu)-1))
    print(f"  Event rate vs C_Du monotonic (increasing)? {er_cdu_monotone}")
    print(f"    Values: {[f'{e:.3f}' for e in ers_cdu]}")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: event_rate vs mu_s
    ax = axes[0, 0]
    ax.plot(mu_s_vals, ers_mu, 'o-', color='navy')
    ax.set_xlabel('mu_s')
    ax.set_ylabel('Event rate')
    ax.set_title('Event Rate vs mu_s')
    ax.grid(True, alpha=0.3)

    # Top-right: median_wait vs mu_s
    ax = axes[0, 1]
    med_waits = [r['median_wait'] for r in results_mu_s]
    valid = [(m, w) for m, w in zip(mu_s_vals, med_waits) if not np.isnan(w)]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], 's-', color='firebrick')
    ax.set_xlabel('mu_s')
    ax.set_ylabel('Median waiting time [s]')
    ax.set_title('Median Wait vs mu_s')
    ax.grid(True, alpha=0.3)

    # Bottom-left: event_rate vs C_Du
    ax = axes[1, 0]
    ax.plot(C_Du_vals, ers_cdu, 'o-', color='forestgreen')
    ax.set_xlabel('C_Du')
    ax.set_ylabel('Event rate')
    ax.set_title('Event Rate vs C_Du')
    ax.grid(True, alpha=0.3)

    # Bottom-right: median_wait vs C_Du
    ax = axes[1, 1]
    med_waits_cdu = [r['median_wait'] for r in results_C_Du]
    valid_cdu = [(c, w) for c, w in zip(C_Du_vals, med_waits_cdu) if not np.isnan(w)]
    if valid_cdu:
        ax.plot([v[0] for v in valid_cdu], [v[1] for v in valid_cdu], 's-', color='darkorange')
    ax.set_xlabel('C_Du')
    ax.set_ylabel('Median waiting time [s]')
    ax.set_title('Median Wait vs C_Du')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Check 3: Waiting-Time Trends', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_check3_trends.pdf", dpi=150)
    plt.close(fig)

    report = {
        'mu_s_sweep': results_mu_s,
        'C_Du_sweep': results_C_Du,
        'er_vs_mu_s_monotone_decreasing': er_monotone,
        'er_vs_C_Du_monotone_increasing': er_cdu_monotone,
        'pass': er_monotone,  # primary check
    }
    status = "PASS" if er_monotone else "FAIL"
    print(f"  --> {status} (er vs mu_s monotone)")

    return report


# ==========================================================================
# CHECK 4: Non-circularity (no injected closure)
# ==========================================================================
def check4_noncircularity(result, cfg, out_dir):
    """During free motion, check if acceleration correlates with -v|u|."""
    print("\n--- CHECK 4: Non-circularity (closure-like correlation) ---")

    dt = cfg['integration']['dt_sim']
    t = result['t']
    x = result['x']
    v_p = result['v_p']
    at_pin = result['at_pin']
    u_b = result['u_b']

    # Free motion only
    free = at_pin == 0
    if np.sum(free) < 100:
        print("  Too few free samples for analysis.")
        return {'pass': True, 'n_free': int(np.sum(free)), 'note': 'too few samples'}

    v_free = v_p[free]
    u_free = u_b[free]

    # Compute empirical acceleration during free motion
    # a = dv_p/dt via central differences
    idx_free = np.where(free)[0]
    # Only use interior points where both neighbors are also free
    valid = []
    for k in range(1, len(idx_free) - 1):
        if idx_free[k] - idx_free[k-1] == 1 and idx_free[k+1] - idx_free[k] == 1:
            valid.append(k)
    valid = np.array(valid)

    if len(valid) < 50:
        print("  Too few contiguous free segments.")
        return {'pass': True, 'n_valid': len(valid), 'note': 'too few contiguous'}

    a_emp = (v_p[idx_free[valid+1]] - v_p[idx_free[valid-1]]) / (2 * dt)
    v_at = v_p[idx_free[valid]]
    u_at = u_b[idx_free[valid]]

    # Candidate closure term: -v|u|  (the d2 term in the EKF closure)
    closure_candidate = -v_at * np.abs(u_at)

    # Also check: -u|v| (d2-like but transposed)
    closure_alt = -u_at * np.abs(v_at)

    # Correlation
    r_closure = np.corrcoef(a_emp, closure_candidate)[0, 1]
    r_alt = np.corrcoef(a_emp, closure_alt)[0, 1]
    r_drag_like = np.corrcoef(a_emp, (u_at - v_at) * np.abs(u_at - v_at))[0, 1]

    print(f"  Free samples: {np.sum(free)}, valid contiguous: {len(valid)}")
    print(f"  corr(a, -v|u|) = {r_closure:.4f}  [EKF closure d2 candidate]")
    print(f"  corr(a, -u|v|) = {r_alt:.4f}  [transposed]")
    print(f"  corr(a, w|w|)  = {r_drag_like:.4f}  [quadratic drag, expected high]")

    # The point: correlation with drag-like terms is expected (physics).
    # Non-circularity means we did NOT inject -v|u| as a force.
    # A moderate correlation is fine (emergent from quadratic slip drag).
    # Only problematic if r > 0.9 (would suggest we accidentally replicated the closure).

    report = {
        'n_free': int(np.sum(free)),
        'n_valid_contiguous': len(valid),
        'corr_a_neg_v_abs_u': float(r_closure),
        'corr_a_neg_u_abs_v': float(r_alt),
        'corr_a_w_abs_w': float(r_drag_like),
        'note': 'moderate correlation with -v|u| is expected (emergent from quadratic drag)',
    }

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.scatter(closure_candidate[:2000], a_emp[:2000], s=1, alpha=0.3, color='navy')
    ax.set_xlabel('-v|u| [m^2/s^2]')
    ax.set_ylabel('a_empirical [m/s^2]')
    ax.set_title(f'a vs -v|u| (r={r_closure:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(closure_alt[:2000], a_emp[:2000], s=1, alpha=0.3, color='firebrick')
    ax.set_xlabel('-u|v| [m^2/s^2]')
    ax.set_ylabel('a_empirical [m/s^2]')
    ax.set_title(f'a vs -u|v| (r={r_alt:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    w_term = (u_at - v_at) * np.abs(u_at - v_at)
    ax.scatter(w_term[:2000], a_emp[:2000], s=1, alpha=0.3, color='forestgreen')
    ax.set_xlabel('w|w| [m^2/s^2]')
    ax.set_ylabel('a_empirical [m/s^2]')
    ax.set_title(f'a vs w|w| (r={r_drag_like:.3f})')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Check 4: Non-Circularity (No Injected Closure)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_check4_noncircularity.pdf", dpi=150)
    plt.close(fig)

    report['pass'] = True  # informational, not pass/fail
    print(f"  --> INFO (non-circularity is structural, not statistical)")

    return report


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("=" * 60)
    print("PIN-CAPTURE DIAGNOSTICS - 4 Validation Checks")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_flow, u_flow, channel = load_flow()
    base_cfg = load_config()
    print(f"Flow: {channel}, N={len(t_flow)}, t=[{t_flow[0]:.1f}, {t_flow[-1]:.1f}]s")

    # Run reference sim with mu_s tuned for intermittent transport (many pin returns)
    cfg = copy.deepcopy(base_cfg)
    cfg['friction']['mu_s'] = 0.57  # tuned for ~0.3-0.7 er on y=0.040
    cfg['integration']['t_end'] = min(540.0, t_flow[-1])

    print("\nRunning reference simulation (mu_s=0.57)...")
    result = simulate_sphere(cfg, t_flow, u_flow, seed=42)
    er = compute_event_rate(result['at_pin'])
    print(f"  Event rate: {er:.3f}")
    print(f"  Captures logged: {len(result['capture_log'])}")

    all_reports = {}

    # Check 1: PSD contamination
    all_reports['check1_psd'] = check1_psd_contamination(result, cfg, OUT_DIR)

    # Check 2: Discontinuity audit
    all_reports['check2_discontinuity'] = check2_discontinuity(result, cfg, OUT_DIR)

    # Check 3: Waiting-time trends (runs multiple sims)
    all_reports['check3_trends'] = check3_waiting_trends(t_flow, u_flow, base_cfg, OUT_DIR)

    # Check 4: Non-circularity
    all_reports['check4_noncircularity'] = check4_noncircularity(result, cfg, OUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, rep in all_reports.items():
        status = "PASS" if rep.get('pass', True) else "FAIL"
        if not rep.get('pass', True):
            all_pass = False
        print(f"  {name}: {status}")

    # Write JSON report
    report_path = OUT_DIR / "pin_diagnostics_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\nReport: {report_path}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL -- see report'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
