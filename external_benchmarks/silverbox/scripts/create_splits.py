"""
Step 2: Create chronological train/val/test splits for Silverbox.

Splits: 70% train / 15% val / 15% test (chronological, no shuffle).
Same column schema as StateSpaceDataset expects.

Usage: python -u external_benchmarks/silverbox/scripts/create_splits.py
"""

import os, sys, json, hashlib, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SILVERBOX_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(SILVERBOX_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
MANIFEST_PATH = os.path.join(SILVERBOX_DIR, 'manifest.json')

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 2: CREATE SPLITS")
    print("=" * 60)

    # Load processed CSV
    csv_path = os.path.join(PROCESSED_DIR, 'silverbox_processed.csv')
    df = pd.read_csv(csv_path)
    N = len(df)
    print(f"  Loaded {csv_path}: {N} rows")

    # Chronological splits
    n_train = int(N * TRAIN_FRAC)
    n_val = int(N * VAL_FRAC)
    n_test = N - n_train - n_val

    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    # Save
    paths = {}
    for name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        out_path = os.path.join(SPLITS_DIR, f'{name}.csv')
        split_df.to_csv(out_path, index=False)
        paths[name] = out_path
        print(f"  {name}: {len(split_df)} samples -> {out_path}")
        print(f"    timestamps: {split_df['timestamp'].iloc[0]:.4f} - "
              f"{split_df['timestamp'].iloc[-1]:.4f} s")

    # Verify column schema
    expected_cols = ['timestamp', 'time_delta', 'velocity', 'displacement']
    for name in ['train', 'val', 'test']:
        check_df = pd.read_csv(paths[name])
        assert list(check_df.columns) == expected_cols, \
            f"{name} columns mismatch: {list(check_df.columns)}"
    print("\n  Column schema verified: PASS")

    # SHA-256 hashes
    hashes = {}
    for name, p in paths.items():
        hashes[name] = sha256_file(p)
        print(f"  {name} SHA-256: {hashes[name]}")

    # Update manifest
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest['step2_splits'] = {
        'train_csv': paths['train'],
        'val_csv': paths['val'],
        'test_csv': paths['test'],
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'train_frac': TRAIN_FRAC,
        'val_frac': VAL_FRAC,
        'test_frac': round(1.0 - TRAIN_FRAC - VAL_FRAC, 4),
        'sha256': hashes,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Updated manifest.json")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
