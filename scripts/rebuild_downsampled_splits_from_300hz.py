"""
Rebuild Clean 10 Hz and 50 Hz Datasets from Correct 300 Hz Splits
=================================================================
NO TRAINING.  Data rebuild + audit only.

Motivation:
  The split audit found val_10hz_ready.csv contains test_10hz_ready.csv
  (VAL_CONTAINS_TEST).  This script rebuilds clean datasets from the
  three independent 300 Hz source files, ensuring no val-test overlap.

Source files (irregular ~413 Hz):
  - "300 hz train.csv"  (0.0 - 1260.81s)
  - "300 hz val.csv"    (1260.81 - 1526.81s)
  - "300hz test.csv"    (1526.82 - 1799.49s)

Method:
  Replicates the original preprocess_10hz.py pipeline per split:
  1. Estimate raw sampling rate from median(diff(t))
  2. Apply 4th-order Butterworth lowpass (cutoff = 0.8 * Nyquist_target)
     using sosfiltfilt at the raw rate
  3. Interpolate to uniform target grid via np.interp
  4. Write CSV with schema: timestamp, time_delta, velocity, displacement

Outputs:
  processed_data_10hz_clean_v1/  {train,val,test}_10hz_ready.csv
  processed_data_50hz_clean_v1/  {train,val,test}_50hz_ready.csv
  final_lockbox_v10_rebuild_data/  inputs_manifest.json, README
"""

import sys, time, json, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
t0_global = time.time()

# ============================================================
#  Helpers
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def discover_300hz_sources():
    """Find the three 300 Hz source CSVs."""
    candidates = {
        'train': [ROOT / "300 hz train.csv",
                  ROOT / "processed_data" / "transformer_ready_train.csv"],
        'val':   [ROOT / "300 hz val.csv",
                  ROOT / "processed_data" / "transformer_ready_val.csv"],
        'test':  [ROOT / "300hz test.csv",
                  ROOT / "processed_data" / "transformer_ready_test.csv"],
    }
    found = {}
    for split, paths in candidates.items():
        for p in paths:
            if p.exists():
                found[split] = p
                break
        if split not in found:
            raise FileNotFoundError(
                f"Cannot find 300 Hz {split} file. Tried: {paths}")
    return found


def fingerprint_source(path):
    """Return dict with file identity info."""
    df = pd.read_csv(path)
    return {
        'path': str(path),
        'rows': len(df),
        'columns': list(df.columns),
        't_min': float(df['timestamp'].iloc[0]),
        't_max': float(df['timestamp'].iloc[-1]),
        'duration_s': float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]),
        'median_dt': float(np.median(np.diff(df['timestamp'].values))),
        'sha256': sha256_file(path),
        'size_bytes': path.stat().st_size,
    }


def resample_split(t_raw, v_raw, x_raw, fs_target, cutoff_hz):
    """
    Resample one split from irregular ~413 Hz to uniform target rate.

    Replicates the original preprocess_10hz.py pipeline:
      1. Estimate fs_raw from median(diff(t))
      2. Design Butterworth lowpass at cutoff_hz
      3. sosfiltfilt on v, x
      4. np.interp onto uniform target grid
    """
    fs_raw = 1.0 / np.median(np.diff(t_raw))
    dt_target = 1.0 / fs_target

    # Butterworth 4th-order lowpass
    sos = butter(4, cutoff_hz, btype='low', fs=fs_raw, output='sos')
    v_filt = sosfiltfilt(sos, v_raw)
    x_filt = sosfiltfilt(sos, x_raw)

    # Uniform target grid
    t_target = np.arange(t_raw[0], t_raw[-1], dt_target)
    v_target = np.interp(t_target, t_raw, v_filt)
    x_target = np.interp(t_target, t_raw, x_filt)

    return t_target, v_target, x_target, fs_raw


def make_ready_df(t, v, x, dt):
    """Create DataFrame in the *_ready.csv schema."""
    time_delta = np.full_like(t, dt)
    time_delta[0] = 0.0
    return pd.DataFrame({
        'timestamp': t,
        'time_delta': time_delta,
        'velocity': v,
        'displacement': x,
    })


def write_manifest(out_dir, split_info, method_desc, rate_hz):
    """Write manifest.json for one output directory."""
    manifest = {
        'rate_hz': rate_hz,
        'dt': 1.0 / rate_hz,
        'method': method_desc,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'splits': split_info,
    }
    mpath = out_dir / 'manifest.json'
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {mpath}")
    return manifest


def write_readme(out_dir, rate_hz, cutoff_hz, split_info, method_desc):
    """Write README.md for one output directory."""
    lines = [
        f"# Clean {rate_hz} Hz Dataset (v1)",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Sampling rate:** {rate_hz} Hz (dt = {1.0/rate_hz:.4f}s)",
        "",
        "## Source",
        "",
        "Rebuilt from three independent 300 Hz source files:",
        "- `300 hz train.csv` (0.0 - 1260.81s)",
        "- `300 hz val.csv` (1260.81 - 1526.81s)",
        "- `300hz test.csv` (1526.82 - 1799.49s)",
        "",
        "Each split processed independently -- no concatenation, no shared state.",
        "",
        "## Method",
        "",
        method_desc,
        "",
        "## Splits",
        "",
        "| Split | Rows | t_min (s) | t_max (s) | Duration (s) | SHA-256 (first 16) |",
        "|-------|------|-----------|-----------|-------------|---------------------|",
    ]
    for s in ['train', 'val', 'test']:
        info = split_info[s]
        lines.append(
            f"| {s} | {info['rows']} | {info['t_min']:.2f} | "
            f"{info['t_max']:.2f} | {info['duration_s']:.2f} | "
            f"{info['sha256'][:16]}... |")
    lines.extend([
        "",
        "## Column Schema",
        "",
        "Matches `processed_data_10hz/*_10hz_ready.csv`:",
        "- `timestamp`: float seconds",
        f"- `time_delta`: constant {1.0/rate_hz:.4f}s (0.0 for first row)",
        "- `velocity`: float m/s (water velocity)",
        "- `displacement`: float m (sediment position)",
    ])
    rpath = out_dir / 'README.md'
    with open(rpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {rpath}")


# ============================================================
#  Main
# ============================================================

def main():
    print("=" * 70)
    print("REBUILD CLEAN DATASETS FROM 300 Hz SPLITS")
    print("NO TRAINING -- data rebuild + audit only")
    print("=" * 70)

    # ----------------------------------------------------------
    # Step 0: Discover and fingerprint 300 Hz sources
    # ----------------------------------------------------------
    print("\n--- Step 0: Discover 300 Hz source files ---")
    sources = discover_300hz_sources()
    source_manifest = {}
    for split in ['train', 'val', 'test']:
        fp = fingerprint_source(sources[split])
        source_manifest[split] = fp
        print(f"  {split}: {fp['rows']} rows, "
              f"t=[{fp['t_min']:.2f}, {fp['t_max']:.2f}], "
              f"dur={fp['duration_s']:.1f}s, "
              f"median_dt={fp['median_dt']*1000:.2f}ms")

    # Check disjoint time ranges
    t_ranges = [(source_manifest[s]['t_min'], source_manifest[s]['t_max'])
                for s in ['train', 'val', 'test']]
    assert t_ranges[0][1] < t_ranges[1][0], "Train/val overlap!"
    assert t_ranges[1][1] < t_ranges[2][0], "Val/test overlap!"
    print("  Time ranges: DISJOINT (OK)")

    # Save inputs manifest
    v10_dir = ROOT / "final_lockbox_v10_rebuild_data"
    v10_dir.mkdir(parents=True, exist_ok=True)
    with open(v10_dir / "inputs_manifest.json", 'w') as f:
        json.dump(source_manifest, f, indent=2)
    print(f"  Saved: {v10_dir / 'inputs_manifest.json'}")

    # ----------------------------------------------------------
    # Step 1: Rebuild at 10 Hz and 50 Hz
    # ----------------------------------------------------------
    targets = [
        (10.0, 4.0,  ROOT / "processed_data_10hz_clean_v1",
         "10hz_ready", "10hz"),
        (50.0, 20.0, ROOT / "processed_data_50hz_clean_v1",
         "50hz_ready", "50hz"),
    ]

    for fs_target, cutoff, out_dir, suffix, label in targets:
        print(f"\n{'='*70}")
        print(f"  REBUILDING {label.upper()} ({fs_target} Hz, cutoff={cutoff} Hz)")
        print(f"{'='*70}")
        out_dir.mkdir(parents=True, exist_ok=True)

        method_desc = (
            f"1. Estimate raw sampling rate from median(diff(timestamp)).\n"
            f"2. Apply 4th-order Butterworth lowpass at {cutoff} Hz "
            f"(0.8 x Nyquist = 0.8 x {fs_target/2} Hz) using sosfiltfilt.\n"
            f"3. Interpolate to uniform {fs_target} Hz grid "
            f"(dt = {1.0/fs_target:.4f}s) via np.interp.\n"
            f"4. Each split processed independently from its own source file."
        )

        split_info = {}
        for split in ['train', 'val', 'test']:
            print(f"\n  Processing {split}...")
            df_raw = pd.read_csv(sources[split])
            t_raw = df_raw['timestamp'].values
            v_raw = df_raw['velocity'].values
            x_raw = df_raw['displacement'].values

            t_out, v_out, x_out, fs_raw = resample_split(
                t_raw, v_raw, x_raw, fs_target, cutoff)
            df_out = make_ready_df(t_out, v_out, x_out, 1.0 / fs_target)

            fname = f"{split}_{suffix}.csv"
            fpath = out_dir / fname
            df_out.to_csv(fpath, index=False)

            sha = sha256_file(fpath)
            info = {
                'file': fname,
                'rows': len(df_out),
                't_min': float(t_out[0]),
                't_max': float(t_out[-1]),
                'duration_s': float(t_out[-1] - t_out[0]),
                'fs_raw_estimated': float(fs_raw),
                'sha256': sha,
                'size_bytes': fpath.stat().st_size,
            }
            split_info[split] = info
            print(f"    {fname}: {info['rows']} rows, "
                  f"t=[{info['t_min']:.2f}, {info['t_max']:.2f}], "
                  f"dur={info['duration_s']:.1f}s")

        write_manifest(out_dir, split_info, method_desc, fs_target)
        write_readme(out_dir, fs_target, cutoff, split_info, method_desc)

    # ----------------------------------------------------------
    # Step 2: Quick sanity checks
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SANITY CHECKS")
    print(f"{'='*70}")

    for label, out_dir, suffix in [
        ("10 Hz", ROOT / "processed_data_10hz_clean_v1", "10hz_ready"),
        ("50 Hz", ROOT / "processed_data_50hz_clean_v1", "50hz_ready"),
    ]:
        print(f"\n  {label}:")
        for split in ['train', 'val', 'test']:
            df = pd.read_csv(out_dir / f"{split}_{suffix}.csv")
            dt_actual = np.diff(df['timestamp'].values)
            print(f"    {split}: N={len(df)}, "
                  f"dt_mean={np.mean(dt_actual):.6f}, "
                  f"dt_std={np.std(dt_actual):.2e}, "
                  f"v_range=[{df.velocity.min():.3f}, {df.velocity.max():.3f}], "
                  f"x_range=[{df.displacement.min():.4f}, {df.displacement.max():.4f}]")

        # Check no overlap between val and test
        df_val = pd.read_csv(out_dir / f"val_{suffix}.csv")
        df_test = pd.read_csv(out_dir / f"test_{suffix}.csv")
        val_max = df_val['timestamp'].iloc[-1]
        test_min = df_test['timestamp'].iloc[0]
        gap = test_min - val_max
        print(f"    Val-test gap: {gap:.3f}s (>0 = no overlap: "
              f"{'OK' if gap > 0 else 'FAIL!'})")

    elapsed = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f"REBUILD COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  {ROOT / 'processed_data_10hz_clean_v1'}")
    print(f"  {ROOT / 'processed_data_50hz_clean_v1'}")
    print(f"  {ROOT / 'final_lockbox_v10_rebuild_data'}")


if __name__ == '__main__':
    main()
