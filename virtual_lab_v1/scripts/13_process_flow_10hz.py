"""
13_process_flow_10hz.py - Butterworth 4Hz lowpass + downsample to 10Hz
======================================================================
Applies the EXACT same pipeline as cfd_twin_v1/scripts/03_process_downsample.py
to each probe channel independently.

Pipeline per channel:
  1. Estimate fs_raw from median(diff(t))
  2. Design 4th-order Butterworth at 4.0 Hz cutoff
  3. sosfiltfilt (zero-phase)
  4. np.interp onto uniform 10 Hz grid

Reads:  outputs/flow_probes/u_probes_raw.csv
Writes: outputs/flow_probes/u_probes_10hz.csv
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def process_channel(t_raw, u_raw, fs_raw, cutoff_hz=4.0, fs_target=10.0):
    """Apply Butter4Hz + filtfilt + 10Hz interp to one channel."""
    sos = butter(4, cutoff_hz, btype='low', fs=fs_raw, output='sos')
    u_filt = sosfiltfilt(sos, u_raw)
    dt_target = 1.0 / fs_target
    t_10hz = np.arange(t_raw[0], t_raw[-1], dt_target)
    u_10hz = np.interp(t_10hz, t_raw, u_filt)
    return t_10hz, u_10hz


def main():
    print("=" * 60)
    print("PROCESS + DOWNSAMPLE MULTI-PROBE DATA")
    print("  Pipeline: Butterworth 4Hz LP + resample to 10Hz")
    print("=" * 60)

    # Load raw probe data
    raw_path = ROOT / "outputs" / "flow_probes" / "u_probes_raw.csv"
    df_raw = pd.read_csv(raw_path)
    t_raw = df_raw['time'].values
    print(f"\nRaw data: N={len(t_raw)}, t=[{t_raw[0]:.4f}, {t_raw[-1]:.4f}]s")

    # Estimate raw sampling rate
    fs_raw = 1.0 / np.median(np.diff(t_raw))
    print(f"  Estimated fs_raw = {fs_raw:.1f} Hz")

    # Get all u_* columns (everything except 'time')
    u_cols = [c for c in df_raw.columns if c != 'time']
    print(f"  Channels: {u_cols}")

    # Process each channel
    result = {}
    t_10hz = None
    for col in u_cols:
        u_raw = df_raw[col].values
        t_out, u_out = process_channel(t_raw, u_raw, fs_raw)
        if t_10hz is None:
            t_10hz = t_out
            result['time'] = t_10hz
        result[col] = u_out
        print(f"  {col}: mean={np.mean(u_out):.4f}, std={np.std(u_out):.4f}")

    print(f"\n10 Hz output: N={len(t_10hz)}, t=[{t_10hz[0]:.4f}, {t_10hz[-1]:.4f}]s")
    print(f"  dt_check: {np.mean(np.diff(t_10hz)):.6f}s (expect 0.1)")

    # Write output
    df_out = pd.DataFrame(result)
    out_path = ROOT / "outputs" / "flow_probes" / "u_probes_10hz.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path} (N={len(df_out)}, cols={list(df_out.columns)})")

    # Phase 1 verification: compare y=0.040 channel if present
    if 'u_y040' in u_cols:
        ref_path = ROOT.parent / "cfd_twin_v1" / "outputs" / "probe_u_10hz.csv"
        if ref_path.exists():
            df_ref = pd.read_csv(ref_path)
            min_len = min(len(result['u_y040']), len(df_ref))
            diff = np.abs(result['u_y040'][:min_len] - df_ref['velocity'].values[:min_len])
            max_diff = np.max(diff)
            print(f"\n  Phase 1 verification (u_y040 vs probe_u_10hz):")
            print(f"    max|diff| = {max_diff:.2e}")
            if max_diff < 1e-6:
                print("    PASS: matches Phase 1 output (atol < 1e-6)")
            else:
                print(f"    WARNING: diff = {max_diff:.2e} (expected < 1e-6)")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
