"""
00_prepare_inflow.py - Prepare OpenFOAM inlet BC from lab velocity data
========================================================================
Loads val+test 10Hz clean data, applies gain/lag/ramp, and writes the
complete 0/U file with uniformFixedValue + inline table.

Also writes lab_data/u_lab_10hz.csv for validation reference.

Usage:
    python 00_prepare_inflow.py [--gain 1.0] [--lag 8.8] [--ramp 5.0]
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
CFD_ROOT = Path(__file__).resolve().parent.parent      # cfd_twin_v1/

# --- Calibration defaults ---
DEFAULT_GAIN = 1.0
DEFAULT_LAG  = 8.8    # seconds (advective delay ~ x_probe / U_mean)
DEFAULT_RAMP = 5.0    # seconds (linear ramp at start)


def load_lab_data():
    """Load val + test clean 10Hz data, concatenate, rebase time to 0."""
    data_dir = ROOT / "processed_data_10hz_clean_v1"
    df_val  = pd.read_csv(data_dir / "val_10hz_ready.csv")
    df_test = pd.read_csv(data_dir / "test_10hz_ready.csv")

    df = pd.concat([df_val, df_test], ignore_index=True)
    t_raw = df['timestamp'].values
    u_raw = df['velocity'].values

    # Rebase time to start at 0
    t0 = t_raw[0]
    t = t_raw - t0

    print(f"Loaded lab data: N={len(t)}, t=[{t[0]:.2f}, {t[-1]:.2f}]s, "
          f"u_mean={np.mean(u_raw):.4f}, u_std={np.std(u_raw):.4f}")

    return t, u_raw


def apply_gain_and_lag(t_lab, u_lab, gain, lag_s):
    """
    Apply gain and lag to lab signal for inlet BC.

    The inlet signal is: u_inlet(t) = gain * u_lab(t - lag_s)
    For t < lag_s, pad with u_lab[0] (constant pre-roll).
    """
    dt = 0.1  # 10 Hz
    n_lag = int(round(lag_s / dt))

    if n_lag > 0:
        # Pad front with first value
        u_padded = np.concatenate([np.full(n_lag, u_lab[0]), u_lab])
        # Trim end to keep same length as original
        u_shifted = u_padded[:len(u_lab)]
    else:
        u_shifted = u_lab.copy()

    u_inlet = gain * u_shifted

    print(f"Applied gain={gain:.3f}, lag={lag_s:.1f}s (n_lag={n_lag})")
    print(f"  Inlet: u_mean={np.mean(u_inlet):.4f}, u_std={np.std(u_inlet):.4f}")

    return u_inlet


def apply_ramp(t, u_inlet, ramp_s):
    """
    Apply linear ramp at start: blend from mean velocity to actual signal
    over ramp_s seconds to avoid step-change transient.
    """
    if ramp_s <= 0:
        return u_inlet

    u_mean = np.mean(u_inlet)
    u_ramped = u_inlet.copy()
    mask = t < ramp_s
    alpha = t[mask] / ramp_s  # 0 at t=0, 1 at t=ramp_s
    u_ramped[mask] = u_mean * (1 - alpha) + u_inlet[mask] * alpha

    print(f"Applied {ramp_s:.1f}s ramp from u_mean={u_mean:.4f} to signal")

    return u_ramped


def write_U_file(t, u_inlet, case_dir):
    """
    Write the complete 0/U file with uniformFixedValue and inline table.
    Each entry: (time  (Ux 0 0))
    """
    u_file = case_dir / "0" / "U"

    # Build the table entries
    table_lines = []
    for i in range(len(t)):
        table_lines.append(f"        ({t[i]:.4f}   ({u_inlet[i]:.6f} 0 0))")

    u_mean = np.mean(u_inlet)

    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({u_mean:.6f} 0 0);

boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        uniformValue    table
        (
{chr(10).join(table_lines)}
        );
        value           uniform ({u_mean:.6f} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    bottom
    {{
        type            noSlip;
    }}
    top
    {{
        type            slip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""

    with open(u_file, 'w') as f:
        f.write(content)

    print(f"Wrote {u_file} ({len(table_lines)} table entries)")


def write_lab_reference(t, u_lab):
    """Write unmodified lab data for validation reference."""
    out_path = CFD_ROOT / "lab_data" / "u_lab_10hz.csv"
    df = pd.DataFrame({'time': t, 'velocity': u_lab})
    df.to_csv(out_path, index=False)
    print(f"Wrote lab reference: {out_path} (N={len(t)})")


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenFOAM inlet BC")
    parser.add_argument('--gain', type=float, default=DEFAULT_GAIN,
                        help=f'Gain factor (default: {DEFAULT_GAIN})')
    parser.add_argument('--lag', type=float, default=DEFAULT_LAG,
                        help=f'Lag in seconds (default: {DEFAULT_LAG})')
    parser.add_argument('--ramp', type=float, default=DEFAULT_RAMP,
                        help=f'Ramp duration in seconds (default: {DEFAULT_RAMP})')
    args = parser.parse_args()

    print("=" * 60)
    print("PREPARE OPENFOAM INLET BC FROM LAB DATA")
    print(f"  gain={args.gain}, lag={args.lag}s, ramp={args.ramp}s")
    print("=" * 60)

    # Load lab data
    t_lab, u_lab = load_lab_data()

    # Write unmodified lab reference
    write_lab_reference(t_lab, u_lab)

    # Apply gain + lag
    u_inlet = apply_gain_and_lag(t_lab, u_lab, args.gain, args.lag)

    # Apply ramp
    u_inlet = apply_ramp(t_lab, u_inlet, args.ramp)

    # Write U file
    case_dir = CFD_ROOT / "case"
    write_U_file(t_lab, u_inlet, case_dir)

    # Summary
    print("\n" + "=" * 60)
    print("INLET PREPARATION COMPLETE")
    print(f"  Simulation duration: {t_lab[-1]:.1f}s")
    print(f"  Table entries: {len(t_lab)}")
    print(f"  Inlet mean: {np.mean(u_inlet):.4f} m/s")
    print(f"  Inlet std:  {np.std(u_inlet):.4f} m/s")
    print("=" * 60)


if __name__ == '__main__':
    main()
