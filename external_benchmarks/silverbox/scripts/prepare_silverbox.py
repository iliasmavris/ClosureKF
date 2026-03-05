"""
Step 1: Download and preprocess the Silverbox benchmark dataset.

Source: nonlinearbenchmark.org/benchmarks/silverbox
System: Electronic Duffing oscillator (2nd-order LTI + cubic nonlinearity)
Native sampling rate: 610.35 Hz (= 10^7 / 2^14)

Downsamples by factor 61 -> dt_eff = sampling_time * 61 (exact float).
DC removal only on output (no z-scoring).

Usage: python -u external_benchmarks/silverbox/scripts/prepare_silverbox.py
"""

import os, sys, json, hashlib, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SILVERBOX_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(SILVERBOX_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
CONFIG_PATH = os.path.join(SILVERBOX_DIR, 'configs', 'pipeline_config.json')
MANIFEST_PATH = os.path.join(SILVERBOX_DIR, 'manifest.json')

DOWNSAMPLE_FACTOR = 61


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: PREPARE SILVERBOX DATA")
    print("=" * 60)

    # Load via nonlinear_benchmarks package
    import nonlinear_benchmarks
    print("Loading Silverbox dataset via nonlinear_benchmarks...")
    train_val, test = nonlinear_benchmarks.Silverbox()

    # Extract signals
    u_raw = np.array(train_val.u, dtype=np.float64)
    y_raw = np.array(train_val.y, dtype=np.float64)
    native_dt = float(train_val.sampling_time)

    print(f"  Native sampling time: {native_dt:.10f} s")
    print(f"  Native sampling rate: {1.0/native_dt:.2f} Hz")
    print(f"  Train+val raw length: {len(u_raw)} samples")

    # Downsample by factor 61
    dt_eff = native_dt * DOWNSAMPLE_FACTOR
    print(f"\n  Downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"  dt_eff = {dt_eff:.10f} s")
    print(f"  Effective rate: {1.0/dt_eff:.2f} Hz")

    u_ds = u_raw[::DOWNSAMPLE_FACTOR]
    y_ds = y_raw[::DOWNSAMPLE_FACTOR]
    N = len(u_ds)
    print(f"  After downsampling: {N} samples")

    # DC removal only (no z-scoring)
    dc_offset = float(np.mean(y_ds))
    displacement = y_ds - dc_offset
    print(f"\n  DC offset (mean of y_ds): {dc_offset:.8f}")

    # Build DataFrame with exact StateSpaceDataset columns
    timestamps = np.arange(N, dtype=np.float64) * dt_eff
    df = pd.DataFrame({
        'timestamp': timestamps,
        'time_delta': np.full(N, dt_eff),
        'velocity': u_ds,
        'displacement': displacement,
    })

    # Save
    out_path = os.path.join(PROCESSED_DIR, 'silverbox_processed.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df.shape}")

    # Signal statistics
    print("\n  Signal statistics:")
    for col in ['timestamp', 'time_delta', 'velocity', 'displacement']:
        vals = df[col].values
        print(f"    {col:>14s}: mean={np.mean(vals):+.6f}  std={np.std(vals):.6f}  "
              f"min={np.min(vals):+.6f}  max={np.max(vals):+.6f}")

    # SHA-256
    csv_hash = sha256_file(out_path)
    print(f"\n  SHA-256: {csv_hash}")

    # Update pipeline_config.json with dt_eff
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    config['dt_eff'] = dt_eff
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Updated pipeline_config.json with dt_eff={dt_eff}")

    # Write manifest
    manifest = {
        'step1_prepare': {
            'output_csv': out_path,
            'sha256': csv_hash,
            'dt_eff': dt_eff,
            'downsample_factor': DOWNSAMPLE_FACTOR,
            'native_sampling_time': native_dt,
            'dc_offset': dc_offset,
            'n_raw': len(u_raw),
            'n_downsampled': N,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote manifest.json")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
