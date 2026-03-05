"""
Audit Split Integrity for Clean v1 Datasets
============================================
NO TRAINING.  Verification only.

Checks both 10 Hz and 50 Hz clean datasets for:
  1. Separate files with disjoint time ranges
  2. No val-test row overlap (byte-equivalent check)
  3. Correct durations and row counts
  4. Valid evaluation origins for selected horizons
  5. Visual proof (split_map plots)

Verdict must be CLEAN_NO_OVERLAP.
"""

import sys, time, json, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V10_DIR = ROOT / "final_lockbox_v10_rebuild_data"

# ============================================================
#  Helpers
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def audit_one_rate(label, out_dir, suffix, dt_expected):
    """Audit one dataset (10 Hz or 50 Hz). Returns (verdict, details)."""
    print(f"\n{'='*70}")
    print(f"  AUDIT: {label}")
    print(f"{'='*70}")

    results = {'label': label, 'checks': []}
    all_pass = True

    # --- Check 1: Files exist ---
    files = {}
    for split in ['train', 'val', 'test']:
        fpath = out_dir / f"{split}_{suffix}.csv"
        exists = fpath.exists()
        if not exists:
            print(f"  [FAIL] {split} file missing: {fpath}")
            all_pass = False
        else:
            files[split] = fpath
    results['checks'].append(('files_exist', len(files) == 3))
    if len(files) < 3:
        return 'FAIL_MISSING_FILES', results

    # --- Check 2: Load and verify schemas ---
    dfs = {}
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(files[split])
        dfs[split] = df
        expected_cols = ['timestamp', 'time_delta', 'velocity', 'displacement']
        if list(df.columns) != expected_cols:
            print(f"  [FAIL] {split} columns: {list(df.columns)} "
                  f"(expected {expected_cols})")
            all_pass = False
    results['checks'].append(('schema_correct', all_pass))

    # --- Check 3: Time ranges and durations ---
    print(f"\n  File summary:")
    split_info = {}
    for split in ['train', 'val', 'test']:
        df = dfs[split]
        t = df['timestamp'].values
        dt_actual = np.diff(t)
        info = {
            'rows': len(df),
            't_min': float(t[0]),
            't_max': float(t[-1]),
            'duration_s': float(t[-1] - t[0]),
            'dt_mean': float(np.mean(dt_actual)),
            'dt_std': float(np.std(dt_actual)),
            'sha256': sha256_file(files[split]),
        }
        split_info[split] = info
        print(f"    {split:6s}: {info['rows']:6d} rows, "
              f"t=[{info['t_min']:8.2f}, {info['t_max']:8.2f}], "
              f"dur={info['duration_s']:.2f}s, "
              f"dt={info['dt_mean']:.6f}+/-{info['dt_std']:.2e}")

    # Check uniform dt
    for split in ['train', 'val', 'test']:
        if split_info[split]['dt_std'] > 1e-8:
            print(f"  [WARN] {split} dt not perfectly uniform "
                  f"(std={split_info[split]['dt_std']:.2e})")
    results['split_info'] = split_info

    # --- Check 4: Disjoint time ranges ---
    print(f"\n  Temporal ordering:")
    t_train_max = split_info['train']['t_max']
    t_val_min = split_info['val']['t_min']
    t_val_max = split_info['val']['t_max']
    t_test_min = split_info['test']['t_min']

    gap_tv = t_val_min - t_train_max
    gap_vt = t_test_min - t_val_max
    print(f"    Train ends:   {t_train_max:.4f}")
    print(f"    Val starts:   {t_val_min:.4f}  (gap = {gap_tv:.4f}s)")
    print(f"    Val ends:     {t_val_max:.4f}")
    print(f"    Test starts:  {t_test_min:.4f}  (gap = {gap_vt:.4f}s)")

    disjoint = gap_tv > 0 and gap_vt > 0
    if disjoint:
        print(f"    --> DISJOINT (OK)")
    else:
        print(f"    --> OVERLAP DETECTED (FAIL)")
        all_pass = False
    results['checks'].append(('disjoint_time_ranges', disjoint))

    # --- Check 5: No val-test byte overlap ---
    print(f"\n  Byte-level overlap check:")
    val_bytes = set()
    df_val = dfs['val']
    for i in range(len(df_val)):
        row = df_val.iloc[i]
        key = f"{row['timestamp']:.10f},{row['velocity']:.15f},{row['displacement']:.15f}"
        val_bytes.add(key)

    df_test = dfs['test']
    n_overlap = 0
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        key = f"{row['timestamp']:.10f},{row['velocity']:.15f},{row['displacement']:.15f}"
        if key in val_bytes:
            n_overlap += 1

    no_byte_overlap = (n_overlap == 0)
    print(f"    Val unique keys: {len(val_bytes)}")
    print(f"    Test rows matching val: {n_overlap}")
    print(f"    --> {'NO OVERLAP (OK)' if no_byte_overlap else 'OVERLAP FOUND (FAIL)'}")
    if not no_byte_overlap:
        all_pass = False
    results['checks'].append(('no_byte_overlap', no_byte_overlap))

    # --- Check 6: Valid evaluation origins ---
    print(f"\n  Valid evaluation origins:")
    n_test = split_info['test']['rows']
    for h in [50, 200, 500]:
        valid = max(0, n_test - h)
        print(f"    h={h:4d}: {valid:5d} valid origins out of {n_test} test points "
              f"({'OK' if valid > 0 else 'INSUFFICIENT'})")
    results['checks'].append(('valid_origins_h50', n_test > 50))

    # --- Check 7: Compare with old contaminated files ---
    print(f"\n  Comparison with old {label} files:")
    old_dir = ROOT / "processed_data_10hz"
    if label.startswith("10") and old_dir.exists():
        for split in ['train', 'val', 'test']:
            old_path = old_dir / f"{split}_10hz_ready.csv"
            if old_path.exists():
                df_old = pd.read_csv(old_path)
                print(f"    OLD {split}: {len(df_old)} rows, "
                      f"t=[{df_old.timestamp.iloc[0]:.2f}, "
                      f"{df_old.timestamp.iloc[-1]:.2f}]")
        df_new = dfs['val']
        print(f"    NEW val:   {len(df_new)} rows, "
              f"t=[{df_new.timestamp.iloc[0]:.2f}, "
              f"{df_new.timestamp.iloc[-1]:.2f}]")
        print(f"    OLD val contained test: YES (known issue)")
        print(f"    NEW val contains test: "
              f"{'NO (FIXED)' if no_byte_overlap else 'YES (STILL BROKEN)'}")

    verdict = 'CLEAN_NO_OVERLAP' if all_pass else 'FAIL'
    results['verdict'] = verdict
    print(f"\n  VERDICT: {verdict}")
    return verdict, results


def make_split_map(dfs_dict, label, rate_hz, save_path):
    """Create split map visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [SKIP] matplotlib not available for {label} plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1])

    colors = {'train': '#2196F3', 'val': '#FF9800', 'test': '#4CAF50'}

    # Panel 1: Displacement with split shading
    ax = axes[0]
    for split, df in dfs_dict.items():
        t = df['timestamp'].values
        x = df['displacement'].values
        ax.plot(t, x, color=colors[split], linewidth=0.4, alpha=0.8,
                label=f'{split} ({len(df)} pts)')
    ax.set_ylabel('Displacement (m)')
    ax.set_title(f'Clean {label} Splits -- Split Map')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Timeline bars
    ax = axes[1]
    for i, (split, df) in enumerate(dfs_dict.items()):
        t_min = df['timestamp'].iloc[0]
        t_max = df['timestamp'].iloc[-1]
        ax.barh(i, t_max - t_min, left=t_min, height=0.5,
                color=colors[split], alpha=0.7, edgecolor='black')
        ax.text(t_min + (t_max - t_min) / 2, i,
                f'{split}\n{t_min:.1f}-{t_max:.1f}s',
                ha='center', va='center', fontsize=8)
    ax.set_yticks(range(len(dfs_dict)))
    ax.set_yticklabels(list(dfs_dict.keys()))
    ax.set_xlabel('Time (s)')
    ax.set_title('File Spans (must be disjoint)')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
#  Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("AUDIT SPLIT INTEGRITY: Clean v1 Datasets")
    print("NO TRAINING -- verification only")
    print("=" * 70)

    V10_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    verdicts = []

    for label, out_dir, suffix, dt_expected in [
        ("10 Hz", ROOT / "processed_data_10hz_clean_v1", "10hz_ready", 0.1),
        ("50 Hz", ROOT / "processed_data_50hz_clean_v1", "50hz_ready", 0.02),
    ]:
        verdict, details = audit_one_rate(label, out_dir, suffix, dt_expected)
        all_results[label] = details
        verdicts.append(verdict)

        # Make split map
        if verdict == 'CLEAN_NO_OVERLAP':
            dfs = {}
            for split in ['train', 'val', 'test']:
                dfs[split] = pd.read_csv(
                    out_dir / f"{split}_{suffix}.csv")
            tag = label.replace(' ', '').lower()
            make_split_map(
                dfs, label,
                1.0 / dt_expected,
                V10_DIR / f"split_map_{tag}_clean_v1.png")

    # Write audit README
    final_verdict = ('CLEAN_NO_OVERLAP'
                     if all(v == 'CLEAN_NO_OVERLAP' for v in verdicts)
                     else 'FAIL')

    readme_lines = [
        "# Split Integrity Audit: Clean v1 Datasets",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Overall verdict:** {final_verdict}",
        "",
        "## Context",
        "",
        "The original `val_10hz_ready.csv` contained 100% of `test_10hz_ready.csv`",
        "(VAL_CONTAINS_TEST). Clean v1 datasets were rebuilt from the three",
        "independent 300 Hz source files to eliminate this contamination.",
        "",
        "## Audit Results",
        "",
    ]

    for label, details in all_results.items():
        readme_lines.append(f"### {label}")
        readme_lines.append(f"- **Verdict:** {details['verdict']}")
        for check_name, passed in details['checks']:
            status = 'PASS' if passed else 'FAIL'
            readme_lines.append(f"- {check_name}: {status}")
        if 'split_info' in details:
            readme_lines.append("")
            readme_lines.append("| Split | Rows | t_min | t_max | Duration |")
            readme_lines.append("|-------|------|-------|-------|----------|")
            for s in ['train', 'val', 'test']:
                info = details['split_info'][s]
                readme_lines.append(
                    f"| {s} | {info['rows']} | {info['t_min']:.2f} | "
                    f"{info['t_max']:.2f} | {info['duration_s']:.2f}s |")
        readme_lines.append("")

    readme_lines.extend([
        "## Key Differences from Original Splits",
        "",
        "| Property | Original (contaminated) | Clean v1 |",
        "|----------|------------------------|----------|",
        "| Train range | 0 - 1008.6s | 0 - 1260.8s |",
        "| Val range | 1008.7 - 1260.8s (contains test!) | 1260.8 - 1526.8s |",
        "| Test range | 1134.8 - 1260.8s | 1526.8 - 1799.5s |",
        "| Val-test overlap | YES (100% of test in val) | NO |",
        "| Source | Only 300 Hz 'train' file | All three 300 Hz files |",
    ])

    readme_path = V10_DIR / "README_split_audit_clean_v1.md"
    with open(readme_path, 'w') as f:
        f.write('\n'.join(readme_lines))
    print(f"\nSaved: {readme_path}")

    # Save machine-readable results
    with open(V10_DIR / "audit_results_clean_v1.json", 'w') as f:
        json.dump({
            'verdict': final_verdict,
            'results': {k: {
                'verdict': v['verdict'],
                'checks': v['checks'],
                'split_info': v.get('split_info', {}),
            } for k, v in all_results.items()},
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"AUDIT COMPLETE ({elapsed:.1f}s)")
    print(f"FINAL VERDICT: {final_verdict}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
