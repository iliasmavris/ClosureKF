"""
Synthetic Forced Convection Cooling: Data Generator

Generates 90 days of synthetic temperature data for a small object
cooling in wind, with known closure term for pipeline validation.

Truth ODE:
    dT/dt = -k_true*(T - T_air) - h_true*wind*(T - T_air) + noise

Physics floor (S1): dT/dt = -k*(T - T_air)
Known closure (S2): wind*(T - T_air) with coefficient ~ -h_true

Forcing signals:
    T_air: 25C + 3C diurnal + 1C synoptic (5-day) + OU noise
    wind:  |OU process| * diurnal envelope, mean ~1 m/s
    PAR:   half-sine diurnal (distractor only)

Usage:
    python -u cooling_benchmark/scripts/generate_data.py
"""

import os, sys, json, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path

# ======================================================================
#  TRUTH PARAMETERS
# ======================================================================

TRUTH_PARAMS = {
    'k_true': 1.0e-4,       # s^-1, natural convection (tau ~ 2.8h)
    'h_true': 3.0e-4,       # s^-1/(m/s), wind-enhanced cooling (~3x natural)
    'sigma_q': 0.005,        # degC/sqrt(s), process noise (SNR ~3-4)
    'R_true': 0.005,         # degC^2, observation noise variance
    'T0_init': 30.0,         # degC, initial temperature
    'dt': 1800,              # s, 30 min sampling
    'N_days': 90,            # 90 days of data
}

# Forcing parameters
FORCING_PARAMS = {
    'T_air_mean': 25.0,       # degC
    'T_air_diurnal_amp': 3.0, # degC
    'T_air_synoptic_amp': 1.0,# degC (5-day cycle)
    'T_air_ou_tau': 3*3600,   # s (3 hours)
    'T_air_ou_sigma': 0.3,    # degC
    'wind_ou_tau': 6*3600,    # s (6 hours)
    'wind_ou_sigma': 0.8,     # m/s (higher variability for detectability)
    'wind_mean_target': 1.5,  # m/s (higher mean for stronger signal)
    'wind_clip_min': 0.1,     # m/s
    'wind_clip_max': 3.0,     # m/s (tighter clip for Euler stability)
    'par_peak': 1.0,          # normalized [0,1] (distractor; eliminates scale issue)
}

SEED_DATA = 42  # Fixed seed for data generation (reproducible)


def generate_ou_process(N, dt, tau, sigma, rng):
    """Generate Ornstein-Uhlenbeck process."""
    x = np.zeros(N)
    alpha = 1.0 / tau
    noise_std = sigma * np.sqrt(2 * alpha * dt)
    for i in range(1, N):
        x[i] = x[i-1] - alpha * x[i-1] * dt + noise_std * rng.standard_normal()
    return x


def generate_forcing(N, dt, rng):
    """Generate all forcing signals."""
    fp = FORCING_PARAMS
    times = np.arange(N) * dt  # seconds
    hours = times / 3600.0
    days = times / 86400.0

    # --- T_air ---
    # Base: mean + diurnal + synoptic
    T_air_base = (fp['T_air_mean']
                  + fp['T_air_diurnal_amp'] * np.sin(2 * np.pi * hours / 24.0 - np.pi/2)
                  + fp['T_air_synoptic_amp'] * np.sin(2 * np.pi * days / 5.0))
    # Add OU noise
    T_air_ou = generate_ou_process(N, dt, fp['T_air_ou_tau'], fp['T_air_ou_sigma'], rng)
    T_air = T_air_base + T_air_ou

    # --- Wind ---
    # OU process for base variability
    wind_ou = generate_ou_process(N, dt, fp['wind_ou_tau'], fp['wind_ou_sigma'], rng)
    # Diurnal envelope: stronger during day (afternoon peak)
    diurnal_env = 0.7 + 0.3 * np.maximum(0, np.sin(2 * np.pi * hours / 24.0 - np.pi/3))
    # Combine: shift to positive, scale by diurnal
    wind_raw = (fp['wind_mean_target'] + wind_ou) * diurnal_env
    wind = np.clip(np.abs(wind_raw), fp['wind_clip_min'], fp['wind_clip_max'])

    # --- PAR (distractor) ---
    # Half-sine during daytime, zero at night
    hour_of_day = hours % 24.0
    par = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        fp['par_peak'] * np.sin(np.pi * (hour_of_day - 6) / 12.0),
        0.0
    )

    return times, T_air, wind, par


def integrate_truth_ode(N, dt, T_air, wind, rng):
    """Integrate truth ODE with Euler-Maruyama."""
    tp = TRUTH_PARAMS
    k = tp['k_true']
    h = tp['h_true']
    sigma_q = tp['sigma_q']

    T_true = np.zeros(N)
    T_true[0] = tp['T0_init']

    for i in range(1, N):
        dT = T_true[i-1] - T_air[i-1]
        # Truth: Newton's cooling + forced convection + noise
        drift = -k * dT - h * wind[i-1] * dT
        noise = sigma_q * np.sqrt(dt) * rng.standard_normal()
        T_true[i] = T_true[i-1] + drift * dt + noise

    return T_true


def add_measurement_noise(T_true, R_true, rng):
    """Add Gaussian measurement noise."""
    return T_true + np.sqrt(R_true) * rng.standard_normal(len(T_true))


def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("Synthetic Forced Convection Cooling: Data Generation")
    print("=" * 70)

    tp = TRUTH_PARAMS
    dt = tp['dt']
    N = tp['N_days'] * 86400 // dt  # Total samples
    print(f"  dt = {dt}s, N = {N} samples ({tp['N_days']} days)")
    print(f"  k_true = {tp['k_true']:.2e} s^-1 (tau = {1/tp['k_true']/3600:.1f}h)")
    print(f"  h_true = {tp['h_true']:.2e} s^-1/(m/s)")
    print(f"  sigma_q = {tp['sigma_q']:.3f} degC/sqrt(s)")
    print(f"  R_true = {tp['R_true']:.4f} degC^2")

    rng = np.random.default_rng(SEED_DATA)

    # Generate forcing
    print("\n  Generating forcing signals...")
    times, T_air, wind, par = generate_forcing(N, dt, rng)
    print(f"    T_air: mean={np.mean(T_air):.1f}, std={np.std(T_air):.2f}, "
          f"range=[{np.min(T_air):.1f}, {np.max(T_air):.1f}]")
    print(f"    wind:  mean={np.mean(wind):.2f}, std={np.std(wind):.2f}, "
          f"range=[{np.min(wind):.2f}, {np.max(wind):.2f}]")
    print(f"    PAR:   mean={np.mean(par):.0f}, max={np.max(par):.0f}")

    # Integrate truth ODE
    print("\n  Integrating truth ODE...")
    T_true = integrate_truth_ode(N, dt, T_air, wind, rng)
    print(f"    T_true: mean={np.mean(T_true):.2f}, std={np.std(T_true):.2f}, "
          f"range=[{np.min(T_true):.2f}, {np.max(T_true):.2f}]")

    # Check Euler stability
    k_eff_max = (tp['k_true'] + tp['h_true'] * np.max(wind))
    stability = k_eff_max * dt
    print(f"    Euler stability: max(k_eff*dt) = {stability:.3f} (< 2.0: {'OK' if stability < 2.0 else 'UNSTABLE!'})")

    # Add measurement noise
    print("\n  Adding measurement noise...")
    T_obs = add_measurement_noise(T_true, tp['R_true'], rng)

    # Build DataFrame
    timestamps = times  # seconds from start
    time_delta = np.full(N, float(dt))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'time_delta': time_delta,
        'water_temp': T_obs,
        'air_temp': T_air,
        'wind_speed': wind,
        'par': par,
    })

    # Chronological split: 70/15/15
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val

    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df.iloc[n_train:n_train+n_val].copy().reset_index(drop=True)
    df_test = df.iloc[n_train+n_val:].copy().reset_index(drop=True)

    print(f"\n  Splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"    Train: t=[{df_train['timestamp'].iloc[0]:.0f}, {df_train['timestamp'].iloc[-1]:.0f}]s")
    print(f"    Val:   t=[{df_val['timestamp'].iloc[0]:.0f}, {df_val['timestamp'].iloc[-1]:.0f}]s")
    print(f"    Test:  t=[{df_test['timestamp'].iloc[0]:.0f}, {df_test['timestamp'].iloc[-1]:.0f}]s")

    # Save CSVs
    SCRIPT_DIR = Path(__file__).resolve().parent
    BENCH_DIR = SCRIPT_DIR.parent
    splits_dir = BENCH_DIR / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, df_split in [('train', df_train), ('val', df_val), ('test', df_test)]:
        path = splits_dir / f"{name}.csv"
        df_split.to_csv(path, index=False)
        print(f"    Saved: {path}")

    # Save truth parameters
    truth_path = BENCH_DIR / "data" / "truth_params.json"
    truth_out = {
        'truth_params': {k: float(v) for k, v in TRUTH_PARAMS.items()},
        'forcing_params': {k: float(v) for k, v in FORCING_PARAMS.items()},
        'data_seed': SEED_DATA,
        'total_samples': N,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'forcing_stats': {
            'T_air_mean': float(np.mean(T_air)),
            'T_air_std': float(np.std(T_air)),
            'wind_mean': float(np.mean(wind)),
            'wind_std': float(np.std(wind)),
            'par_mean': float(np.mean(par)),
        },
        'T_true_stats': {
            'mean': float(np.mean(T_true)),
            'std': float(np.std(T_true)),
            'min': float(np.min(T_true)),
            'max': float(np.max(T_true)),
        },
        'euler_stability': float(stability),
        'expected_k_s1': float(tp['k_true'] + tp['h_true'] * np.mean(wind)),
    }
    with open(truth_path, 'w') as f:
        json.dump(truth_out, f, indent=2)
    print(f"    Saved: {truth_path}")

    # SHA-256 manifest
    manifest = {}
    for name in ['train', 'val', 'test']:
        path = splits_dir / f"{name}.csv"
        manifest[name] = compute_sha256(path)
    manifest_path = BENCH_DIR / "data" / "sha256_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"    Saved: {manifest_path}")

    print("\n  Data generation complete.")
    print(f"  Expected S1 k ~ {truth_out['expected_k_s1']:.2e} "
          f"(k_true + h_true*mean(wind))")
    print(f"  Expected S2 forced_conv coeff ~ {-tp['h_true']:.2e}")


if __name__ == '__main__':
    main()
