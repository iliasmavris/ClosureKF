"""
98b_rerun_d2_50hz_noresume.py -- Rerun d2-only and 50Hz with --no-resume.

The first orchestrator (98_rerun_alignfix.py) successfully retrained v11.1
but d2/50Hz loaded stale checkpoints because:
  - Output dirs were backed up via copytree (not moved)
  - Both scripts default to --resume, found old checkpoints, skipped training

This script:
  1) Moves old output subdirs to timestamped backup (shutil.move)
  2) Runs each pipeline with --no-resume (belt + suspenders)
  3) Verifies logs show actual training (no "LOADING FROM CHECKPOINT" for S1)
  4) Regression comparison against pre-alignfix metrics

Usage:  python -u scripts/98b_rerun_d2_50hz_noresume.py
Output: outputs/reruns_alignfix_v1/ (logs, regression update)
Runtime: ~3 hours (d2 ~2h, 50Hz ~1h)
"""

import os, sys, json, hashlib, time, shutil, subprocess
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ==============================================================================
#  PATHS
# ==============================================================================
RERUN_DIR  = ROOT / "outputs" / "reruns_alignfix_v1"
LOG_DIR    = RERUN_DIR / "logs"
OLD_DIR    = RERUN_DIR / "old_metrics"
PACKET_DIR = RERUN_DIR / "phase5_packet"

D2_OUT  = ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_10hz_3seed"
HZ50_OUT = ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_50hz_seed1"

PIPELINES = {
    'd2': {
        'script': D2_OUT / "run_lockbox_step4_d2only_3seed.py",
        'out_dir': D2_OUT,
        'agg_csv': D2_OUT / "aggregate" / "summary_seeds_step4.csv",
        'seeds': [1, 2, 3],
        'label': 'd2-only 10Hz closure (3 seeds)',
        'est_min': 120,
        'subdirs_to_move': ['seed1', 'seed2', 'seed3', 'aggregate', 'audit', 'tables'],
        'old_metrics_csv': OLD_DIR / "d2_old_summary_seeds_step4.csv",
    },
    '50hz': {
        'script': HZ50_OUT / "run_lockbox_step5_d2only_50hz.py",
        'out_dir': HZ50_OUT,
        'agg_csv': HZ50_OUT / "aggregate" / "summary_step5_50hz.csv",
        'seeds': [1],
        'label': '50Hz d2-only transfer (seed 1)',
        'est_min': 60,
        'subdirs_to_move': ['seed1', 'aggregate', 'audit'],
        'old_metrics_csv': OLD_DIR / "50hz_old_summary_step5_50hz.csv",
    },
}


# ==============================================================================
#  HELPERS
# ==============================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ==============================================================================
#  PHASE 0: MOVE OLD OUTPUTS TO BACKUP
# ==============================================================================
def phase0_move_to_backup():
    print_section("PHASE 0: MOVE OLD OUTPUTS TO TIMESTAMPED BACKUP")

    for key, pipe in PIPELINES.items():
        out_dir = pipe['out_dir']
        backup_root = out_dir / f"_backup_alignfix_{TIMESTAMP}"
        backup_root.mkdir(parents=True, exist_ok=True)

        moved = 0
        for subdir_name in pipe['subdirs_to_move']:
            src = out_dir / subdir_name
            if src.exists():
                dst = backup_root / subdir_name
                shutil.move(str(src), str(dst))
                print(f"  [{key}] Moved: {subdir_name}/ -> _backup_alignfix_{TIMESTAMP}/{subdir_name}/")
                moved += 1

        # Also move _pre_alignfix backup dirs (cleanup from first orchestrator)
        for item in out_dir.iterdir():
            if item.is_dir() and item.name.endswith('_pre_alignfix'):
                dst = backup_root / item.name
                if not dst.exists():
                    shutil.move(str(item), str(dst))
                    print(f"  [{key}] Moved stale backup: {item.name}/")
                    moved += 1

        if moved == 0:
            print(f"  [{key}] WARNING: No subdirs found to move!")
        else:
            print(f"  [{key}] Moved {moved} subdirs. Clean slate ready.")

    # Verify: no checkpoint files remain
    for key, pipe in PIPELINES.items():
        for seed in pipe['seeds']:
            seed_dir = pipe['out_dir'] / f"seed{seed}"
            if seed_dir.exists():
                print(f"  FATAL: seed{seed}/ still exists in {key} output dir!")
                sys.exit(1)
        agg_dir = pipe['out_dir'] / "aggregate"
        if agg_dir.exists():
            print(f"  FATAL: aggregate/ still exists in {key} output dir!")
            sys.exit(1)

    print("  Clean-slate verification: PASS")


# ==============================================================================
#  PHASE 1: PREFLIGHT -- CONFIRM ALIGNMENT FIX IS IN PLACE
# ==============================================================================
def phase1_preflight():
    print_section("PHASE 1: PREFLIGHT CHECKS")

    ds_file = ROOT / "datasets" / "state_space_dataset.py"
    if not ds_file.exists():
        print("  FATAL: state_space_dataset.py not found!")
        sys.exit(1)

    ds_text = ds_file.read_text(encoding='utf-8')
    fix_pattern = "v_fut = run['v'][t:t + self.H]"
    if fix_pattern not in ds_text:
        print(f"  FATAL: Alignment fix not found in state_space_dataset.py!")
        print(f"  Expected: {fix_pattern}")
        sys.exit(1)

    print(f"  Alignment fix confirmed in state_space_dataset.py")
    print(f"  SHA-256: {sha256_file(ds_file)}")

    # Check old metrics exist for regression
    for key, pipe in PIPELINES.items():
        old_csv = pipe['old_metrics_csv']
        if old_csv.exists():
            print(f"  [{key}] Old metrics: {old_csv.name} (OK)")
        else:
            print(f"  [{key}] WARNING: Old metrics not found: {old_csv}")


# ==============================================================================
#  PHASE 2: RUN PIPELINES WITH --no-resume
# ==============================================================================
def phase2_run_pipeline(key, pipe):
    label = pipe['label']
    script = pipe['script']
    est = pipe['est_min']

    print_section(f"PIPELINE: {label}")
    print(f"  Script: {script}")
    print(f"  Args: --no-resume")
    print(f"  Estimated runtime: ~{est} min")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log_out = LOG_DIR / f"{key}_v2_stdout.log"
    log_err = LOG_DIR / f"{key}_v2_stderr.log"

    t0 = time.time()
    try:
        with open(log_out, 'w', encoding='utf-8') as f_out, \
             open(log_err, 'w', encoding='utf-8') as f_err:
            result = subprocess.run(
                [sys.executable, '-u', str(script), '--no-resume'],
                stdout=f_out,
                stderr=f_err,
                cwd=str(ROOT),
                timeout=est * 180,  # 3x safety
            )
        elapsed = time.time() - t0
        exit_code = result.returncode
        status = 'PASS' if exit_code == 0 else f'FAIL (rc={exit_code})'
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        exit_code = -1
        status = 'TIMEOUT'
    except Exception as e:
        elapsed = time.time() - t0
        exit_code = -2
        status = f'ERROR: {e}'

    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Runtime: {elapsed/60:.1f} min")
    print(f"  Status: {status}")
    print(f"  Logs: {log_out.name}, {log_err.name}")

    return {
        'status': status,
        'runtime_min': round(elapsed / 60, 1),
        'exit_code': exit_code,
        'log_stdout': str(log_out),
        'log_stderr': str(log_err),
    }


# ==============================================================================
#  PHASE 3: VERIFY ACTUAL RETRAINING
# ==============================================================================
def phase3_verify(key, pipe, run_result):
    print_section(f"VERIFY: {key} actually retrained")

    log_path = Path(run_result['log_stdout'])
    if not log_path.exists():
        print(f"  FATAL: Log not found: {log_path}")
        return False

    log_text = log_path.read_text(encoding='utf-8', errors='replace')

    # Check 1: Should NOT see "LOADING FROM CHECKPOINT" for S1
    checks_pass = True

    if 'Resume: False' in log_text or 'Resume: false' in log_text.lower():
        print(f"  [CHECK 1] Resume: False confirmed in log")
    elif 'Resume: True' in log_text:
        print(f"  [CHECK 1] FAIL: Log shows Resume: True!")
        checks_pass = False
    else:
        print(f"  [CHECK 1] Resume flag not found in log (checking training lines instead)")

    # Check 2: S1 should show training epochs, not "LOADING FROM CHECKPOINT"
    s1_loaded = log_text.count('STAGE 1 -- LOADING FROM CHECKPOINT')
    s1_trained = log_text.count('STAGE 1 -- PHYSICS ONLY')
    if s1_loaded > 0:
        print(f"  [CHECK 2] FAIL: Found {s1_loaded} 'LOADING FROM CHECKPOINT' for S1!")
        checks_pass = False
    elif s1_trained > 0:
        print(f"  [CHECK 2] PASS: S1 trained from scratch ({s1_trained} seeds)")
    else:
        print(f"  [CHECK 2] WARNING: No S1 training header found")

    # Check 3: S2 should show training epochs, not "LOADING FROM CHECKPOINT"
    s2_loaded = log_text.count('STAGE 2 -- LOADING FROM CHECKPOINT')
    s2_trained_a = log_text.count('STAGE 2 -- CLOSURE')
    s2_trained_b = log_text.count('STAGE 2 -- d2-ONLY')
    s2_trained = s2_trained_a + s2_trained_b
    if s2_loaded > 0:
        print(f"  [CHECK 3] FAIL: Found {s2_loaded} 'LOADING FROM CHECKPOINT' for S2!")
        checks_pass = False
    elif s2_trained > 0:
        print(f"  [CHECK 3] PASS: S2 trained from scratch ({s2_trained} seeds)")
    else:
        print(f"  [CHECK 3] WARNING: No S2 training header found")

    # Check 4: Extract learned alpha values and compare to stale
    # Stale d2 had alpha=1.7157 (seed1). New v11 rerun got alpha=1.8402.
    # If d2 retrains, its alpha should NOT be exactly 1.7157.
    import re
    alpha_matches = re.findall(r'alpha=([0-9.]+)', log_text)
    if alpha_matches:
        alphas = [float(a) for a in alpha_matches]
        print(f"  [CHECK 4] Learned alphas: {alphas}")
        # The stale values were: d2 seed1=1.7157, seed2=1.7585, seed3=1.6634
        stale_d2_alphas = {1.7157, 1.7585, 1.6634}
        # For 50Hz: stale alpha=2.6762
        stale_50hz_alphas = {2.6762}
        stale_set = stale_d2_alphas if key == 'd2' else stale_50hz_alphas
        exact_stale = sum(1 for a in alphas if any(abs(a - s) < 0.0001 for s in stale_set))
        if exact_stale > 0 and len(alphas) > 0:
            print(f"  [CHECK 4] WARNING: {exact_stale}/{len(alphas)} alphas match stale values exactly!")
        else:
            print(f"  [CHECK 4] PASS: Alphas differ from stale checkpoint values")
    else:
        print(f"  [CHECK 4] No alpha values found in log")

    # Check 5: Verify training wall-clock > minimum threshold
    min_runtime = 5.0  # at least 5 minutes for real training
    actual_runtime = run_result['runtime_min']
    if actual_runtime < min_runtime:
        print(f"  [CHECK 5] FAIL: Runtime {actual_runtime:.1f} min < {min_runtime} min threshold!")
        checks_pass = False
    else:
        print(f"  [CHECK 5] PASS: Runtime {actual_runtime:.1f} min (>= {min_runtime} min)")

    # Check 6: Verify aggregate CSV was newly written
    agg = pipe['agg_csv']
    if agg.exists():
        import datetime as dt_mod
        mtime = dt_mod.datetime.fromtimestamp(agg.stat().st_mtime)
        script_start = dt_mod.datetime.now() - dt_mod.timedelta(minutes=actual_runtime + 5)
        if mtime > script_start:
            print(f"  [CHECK 6] PASS: Aggregate CSV freshly written ({mtime})")
        else:
            print(f"  [CHECK 6] WARNING: Aggregate CSV mtime {mtime} seems old")
    else:
        print(f"  [CHECK 6] FAIL: Aggregate CSV not found: {agg}")
        checks_pass = False

    if checks_pass:
        print(f"  VERIFY OVERALL: PASS")
    else:
        print(f"  VERIFY OVERALL: FAIL -- review log manually")

    return checks_pass


# ==============================================================================
#  PHASE 4: REGRESSION COMPARISON
# ==============================================================================
def phase4_regression(run_results):
    print_section("PHASE 4: REGRESSION COMPARISON (d2 + 50Hz)")

    lines = []
    rows = []
    lines.append("# Regression Summary: d2-only + 50Hz Rerun (v2, --no-resume)\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Fix:** v_fut = v[t:t+H] (start-of-interval alignment)")
    lines.append(f"**Note:** This is the corrected rerun. The first orchestrator")
    lines.append(f"loaded stale checkpoints for d2/50Hz (resume=True).\n")

    # --- d2-only ---
    lines.append("## d2-only 10Hz Pipeline (3 seeds)\n")
    old_csv = PIPELINES['d2']['old_metrics_csv']
    new_csv = PIPELINES['d2']['agg_csv']

    if old_csv.exists() and new_csv.exists():
        df_old = pd.read_csv(old_csv)
        df_new = pd.read_csv(new_csv)

        old_mean = df_old[df_old['seed'].astype(str) == 'mean']
        new_mean = df_new[df_new['seed'].astype(str) == 'mean']

        if len(old_mean) > 0 and len(new_mean) > 0:
            old_mean = old_mean.iloc[0]
            new_mean = new_mean.iloc[0]

            d2_keys = [
                ('s1_alpha', 'alpha'),
                ('s1_kappa', 'kappa'),
                ('s2_d2', 'd2'),
                ('s2_q_scale', 'q_scale'),
                ('phys_acf1', 'ACF1_phys'),
                ('clos_acf1', 'ACF1_clos'),
                ('phys_dxr2_0.5s', 'DxR2@0.5s_phys'),
                ('clos_dxr2_0.5s', 'DxR2@0.5s_clos'),
                ('phys_dxr2_1.0s', 'DxR2@1s_phys'),
                ('clos_dxr2_1.0s', 'DxR2@1s_clos'),
                ('phys_dxr2_2.0s', 'DxR2@2s_phys'),
                ('clos_dxr2_2.0s', 'DxR2@2s_clos'),
            ]

            lines.append("| Metric | OLD (mean) | NEW (mean) | Delta |")
            lines.append("|--------|-----------|-----------|-------|")
            for csv_col, label in d2_keys:
                if csv_col in old_mean.index and csv_col in new_mean.index:
                    o = float(old_mean[csv_col])
                    n = float(new_mean[csv_col])
                    d = n - o
                    lines.append(f"| {label} | {o:.6f} | {n:.6f} | {d:+.6f} |")
                    rows.append({'pipeline': 'd2', 'metric': label, 'old': o, 'new': n, 'delta': d})

            # Per-seed narrative
            for seed_val in [1, 2, 3]:
                seed_str = str(seed_val)
                old_row = df_old[df_old['seed'].astype(str) == seed_str]
                new_row = df_new[df_new['seed'].astype(str) == seed_str]
                if len(old_row) > 0 and len(new_row) > 0:
                    # Check DxR2@1s closure > physics
                    old_p = float(old_row.iloc[0].get('phys_dxr2_1.0s', 0))
                    old_c = float(old_row.iloc[0].get('clos_dxr2_1.0s', 0))
                    new_p = float(new_row.iloc[0].get('phys_dxr2_1.0s', 0))
                    new_c = float(new_row.iloc[0].get('clos_dxr2_1.0s', 0))
                    old_pass = old_c > old_p
                    new_pass = new_c > new_p
                    lines.append(f"  Seed {seed_val}: OLD closure>phys={'PASS' if old_pass else 'FAIL'}"
                                 f" ({old_c:.4f}>{old_p:.4f})"
                                 f" -> NEW={'PASS' if new_pass else 'FAIL'}"
                                 f" ({new_c:.4f}>{new_p:.4f})")
    else:
        lines.append("  Old or new metrics not available.")

    lines.append("")

    # --- 50Hz ---
    lines.append("## 50Hz Transfer Pipeline (seed 1)\n")
    old_csv = PIPELINES['50hz']['old_metrics_csv']
    new_csv = PIPELINES['50hz']['agg_csv']

    if old_csv.exists() and new_csv.exists():
        df_old = pd.read_csv(old_csv)
        df_new = pd.read_csv(new_csv)

        if len(df_old) > 0 and len(df_new) > 0:
            old_r = df_old.iloc[0]
            new_r = df_new.iloc[0]

            hz50_keys = [
                ('s1_alpha', 'alpha'),
                ('s1_kappa', 'kappa'),
                ('s2_d2', 'd2'),
                ('s2_q_scale', 'q_scale'),
                ('phys_acf1', 'ACF1_phys'),
                ('clos_acf1', 'ACF1_clos'),
                ('phys_dxr2_1.0s', 'DxR2@1s_phys'),
                ('clos_dxr2_1.0s', 'DxR2@1s_clos'),
                ('phys_dxr2_2.0s', 'DxR2@2s_phys'),
                ('clos_dxr2_2.0s', 'DxR2@2s_clos'),
            ]

            lines.append("| Metric | OLD | NEW | Delta |")
            lines.append("|--------|-----|-----|-------|")
            for csv_col, label in hz50_keys:
                if csv_col in old_r.index and csv_col in new_r.index:
                    o = float(old_r[csv_col])
                    n = float(new_r[csv_col])
                    d = n - o
                    lines.append(f"| {label} | {o:.6f} | {n:.6f} | {d:+.6f} |")
                    rows.append({'pipeline': '50hz', 'metric': label, 'old': o, 'new': n, 'delta': d})

            # Narrative check
            old_p = float(old_r.get('phys_dxr2_1.0s', 0))
            old_c = float(old_r.get('clos_dxr2_1.0s', 0))
            new_p = float(new_r.get('phys_dxr2_1.0s', 0))
            new_c = float(new_r.get('clos_dxr2_1.0s', 0))
            lines.append(f"  Narrative: OLD closure>phys={'PASS' if old_c > old_p else 'FAIL'}"
                         f" ({old_c:.4f}>{old_p:.4f})"
                         f" -> NEW={'PASS' if new_c > new_p else 'FAIL'}"
                         f" ({new_c:.4f}>{new_p:.4f})")
    else:
        lines.append("  Old or new metrics not available.")

    lines.append("")

    # --- Combined verdict with v11 ---
    lines.append("## Combined Regression Verdict (all 3 pipelines)\n")

    # Load v11 regression from first orchestrator
    v11_reg = RERUN_DIR / "regression_summary.csv"
    if v11_reg.exists():
        v11_rows = pd.read_csv(v11_reg).to_dict('records')
        v11_dxr2 = [r for r in v11_rows
                    if r['pipeline'] == 'v11' and 'DxR2' in str(r['metric']) and 'clos' in str(r['metric']).lower()]
        if v11_dxr2:
            v11_max_delta = max(abs(r['delta']) for r in v11_dxr2)
            lines.append(f"v11 max |delta(DxR2_clos)|: {v11_max_delta:.6f}")

    if rows:
        dxr2_rows = [r for r in rows if 'DxR2' in r['metric'] and 'clos' in r['metric'].lower()]
        if dxr2_rows:
            d2_50hz_max_delta = max(abs(r['delta']) for r in dxr2_rows)
            lines.append(f"d2+50Hz max |delta(DxR2_clos)|: {d2_50hz_max_delta:.6f}")

            all_max = d2_50hz_max_delta
            if v11_reg.exists() and v11_dxr2:
                all_max = max(all_max, v11_max_delta)

            lines.append(f"\nOverall max |delta(DxR2_clos)| across all 3 pipelines: **{all_max:.6f}**")
            if all_max < 0.03:
                lines.append(f"Verdict: **PASS** (all deltas < 0.03)")
            else:
                lines.append(f"Verdict: **CHECK** (some deltas >= 0.03)")

    report = "\n".join(lines)

    # Write v2 regression
    reg_path = RERUN_DIR / "regression_summary_v2.md"
    reg_path.write_text(report, encoding='utf-8')
    print(report)

    if rows:
        df_reg = pd.DataFrame(rows)
        df_reg.to_csv(RERUN_DIR / "regression_summary_v2.csv", index=False)
        print(f"\n  Saved: regression_summary_v2.csv ({len(rows)} rows)")

    print(f"  Saved: regression_summary_v2.md")
    return rows


# ==============================================================================
#  PHASE 5: UPDATE PHASE5 PACKET
# ==============================================================================
def phase5_update_packet():
    print_section("PHASE 5: UPDATE PHASE5 PACKET")

    # Copy new aggregate CSVs
    for key, pipe in PIPELINES.items():
        agg = pipe['agg_csv']
        if agg.exists():
            dest = PACKET_DIR / f"{key}_{agg.name}"
            shutil.copy2(agg, dest)
            print(f"  Updated: {agg.name} -> {dest.name}")

    # Copy v2 regression
    v2_reg = RERUN_DIR / "regression_summary_v2.md"
    if v2_reg.exists():
        shutil.copy2(v2_reg, PACKET_DIR / "regression_summary_v2.md")
        print(f"  Copied: regression_summary_v2.md")

    # Copy d2 figures
    figs_dir = PACKET_DIR / "figures"
    figs_dir.mkdir(exist_ok=True)

    d2_figs = ROOT / "ems_v1" / "figures"
    if d2_figs.exists():
        fig_count = 0
        for pattern in ['fig_seed_variation*', 'fig_d2_distribution*']:
            for fig in d2_figs.glob(pattern):
                shutil.copy2(fig, figs_dir / fig.name)
                fig_count += 1
        print(f"  Updated {fig_count} d2 figures")


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("  98b_rerun_d2_50hz_noresume.py")
    print(f"  Clean-slate rerun of d2-only + 50Hz with --no-resume")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Backup timestamp: {TIMESTAMP}")
    print("=" * 70)

    # Phase 0: Move old outputs
    phase0_move_to_backup()

    # Phase 1: Preflight
    phase1_preflight()

    # Phase 2+3: Run and verify each pipeline
    run_results = {}
    all_verified = True

    for key in ['d2', '50hz']:
        pipe = PIPELINES[key]
        result = phase2_run_pipeline(key, pipe)
        run_results[key] = result

        if not result['status'].startswith('PASS'):
            print(f"\n  WARNING: {key} pipeline failed: {result['status']}")
            all_verified = False
            continue

        verified = phase3_verify(key, pipe, result)
        if not verified:
            all_verified = False

    # Phase 4: Regression
    regression_rows = phase4_regression(run_results)

    # Phase 5: Update packet
    phase5_update_packet()

    # Final summary
    elapsed = time.time() - t_start
    print_section("FINAL SUMMARY")
    print(f"  Total runtime: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    for key, result in run_results.items():
        print(f"  {key}: {result['status']} ({result['runtime_min']} min)")
    print(f"  All verified: {'YES' if all_verified else 'NO -- CHECK LOGS'}")
    print(f"\n  Outputs: {RERUN_DIR}")
    print(f"  Regression: {RERUN_DIR / 'regression_summary_v2.md'}")
    print(f"  Phase5 packet: {PACKET_DIR}")

    all_pass = all(r['status'].startswith('PASS') for r in run_results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
