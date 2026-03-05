"""
03_process_downsample.py - Butterworth 4Hz lowpass + downsample to 10Hz
=======================================================================
Replicates the EXACT lab pipeline from rebuild_downsampled_splits_from_300hz.py:
  1. Estimate fs_raw from median(diff(t))
  2. Design 4th-order Butterworth at 4.0 Hz cutoff
  3. sosfiltfilt (zero-phase)
  4. np.interp onto uniform 10 Hz grid

Reads: outputs/probe_u_raw.csv
Writes: outputs/probe_u_10hz.csv
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

CFD_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 60)
    print("PROCESS + DOWNSAMPLE CFD PROBE DATA")
    print("  Pipeline: Butterworth 4Hz LP + resample to 10Hz")
    print("=" * 60)

    # Load raw probe data
    raw_path = CFD_ROOT / "outputs" / "probe_u_raw.csv"
    df_raw = pd.read_csv(raw_path)
    t_raw = df_raw['time'].values
    u_raw = df_raw['Ux'].values

    print(f"\nRaw data: N={len(t_raw)}, t=[{t_raw[0]:.4f}, {t_raw[-1]:.4f}]s")

    # Step 1: Estimate raw sampling rate
    fs_raw = 1.0 / np.median(np.diff(t_raw))
    print(f"  Estimated fs_raw = {fs_raw:.1f} Hz")

    # Step 2: Design Butterworth lowpass at 4.0 Hz
    cutoff_hz = 4.0
    sos = butter(4, cutoff_hz, btype='low', fs=fs_raw, output='sos')
    print(f"  Butterworth: order=4, cutoff={cutoff_hz} Hz, fs={fs_raw:.1f} Hz")

    # Step 3: Apply zero-phase filter
    u_filt = sosfiltfilt(sos, u_raw)
    print(f"  Filtered: mean={np.mean(u_filt):.4f}, std={np.std(u_filt):.4f}")

    # Step 4: Interpolate to uniform 10 Hz grid
    fs_target = 10.0
    dt_target = 1.0 / fs_target
    t_10hz = np.arange(t_raw[0], t_raw[-1], dt_target)
    u_10hz = np.interp(t_10hz, t_raw, u_filt)

    print(f"\n10 Hz output: N={len(t_10hz)}, t=[{t_10hz[0]:.4f}, {t_10hz[-1]:.4f}]s")
    print(f"  u_mean={np.mean(u_10hz):.4f}, u_std={np.std(u_10hz):.4f}")
    print(f"  dt_check: {np.mean(np.diff(t_10hz)):.6f}s (expect 0.1)")

    # Write output
    out_path = CFD_ROOT / "outputs" / "probe_u_10hz.csv"
    df_out = pd.DataFrame({'time': t_10hz, 'velocity': u_10hz})
    df_out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path} (N={len(df_out)})")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
