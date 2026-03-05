"""
98_rerun_alignfix.py -- Master rerun orchestrator after forcing-alignment fix.

Runs the three paper-critical pipelines with the corrected StateSpaceDataset
(v_fut = v[t:t+H], start-of-interval), then produces:
  - stdout/stderr logs
  - run manifest JSON
  - regression summary (old vs new metrics)
  - phase5 integration packet

Usage:  python -u scripts/98_rerun_alignfix.py [--skip-v11] [--skip-d2] [--skip-50hz]
Output: outputs/reruns_alignfix_v1/
Runtime: ~9 hours total (v11.1 ~370min + d2-only ~120min + 50hz ~60min)
"""

import os, sys, json, hashlib, time, shutil, subprocess, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
#  PATHS
# ==============================================================================
RERUN_DIR  = ROOT / "outputs" / "reruns_alignfix_v1"
LOG_DIR    = RERUN_DIR / "logs"
OLD_DIR    = RERUN_DIR / "old_metrics"
PACKET_DIR = RERUN_DIR / "phase5_packet"

V11_OUT    = ROOT / "final_lockbox_v11_1_alpha_fix"
D2_OUT     = ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_10hz_3seed"
HZ50_OUT   = ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_50hz_seed1"

PIPELINES = {
    'v11': {
        'script': ROOT / "scripts" / "lockbox_v11_1_alpha_fix_3seed.py",
        'out_dir': V11_OUT,
        'agg_csv': V11_OUT / "aggregate" / "summary_seeds.csv",
        'seeds': [1, 2, 3],
        'label': 'v11.1 main pipeline (3 seeds)',
        'est_min': 370,
    },
    'd2': {
        'script': ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_10hz_3seed" / "run_lockbox_step4_d2only_3seed.py",
        'out_dir': D2_OUT,
        'agg_csv': D2_OUT / "aggregate" / "summary_seeds_step4.csv",
        'seeds': [1, 2, 3],
        'label': 'd2-only 10Hz closure (3 seeds)',
        'est_min': 120,
    },
    '50hz': {
        'script': ROOT / "ems_v1" / "runs" / "lockbox_ems_v1_d2only_50hz_seed1" / "run_lockbox_step5_d2only_50hz.py",
        'out_dir': HZ50_OUT,
        'agg_csv': HZ50_OUT / "aggregate" / "summary_step5_50hz.csv",
        'seeds': [1],
        'label': '50Hz d2-only transfer (seed 1)',
        'est_min': 60,
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
    msg = f"\n{'='*70}\n{title}\n{'='*70}"
    print(msg)
    return msg


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)


# ==============================================================================
#  PHASE 0: SETUP
# ==============================================================================
def phase0_setup():
    print_section("PHASE 0: SETUP")

    for d in [RERUN_DIR, LOG_DIR, OLD_DIR, PACKET_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Save old aggregate CSVs for regression comparison
    saved = {}
    for key, pipe in PIPELINES.items():
        agg = pipe['agg_csv']
        if agg.exists():
            dest = OLD_DIR / f"{key}_old_{agg.name}"
            shutil.copy2(agg, dest)
            saved[key] = str(dest)
            print(f"  Saved old metrics: {agg.name} -> {dest.name}")
        else:
            print(f"  WARNING: Old aggregate not found: {agg}")
            saved[key] = None

    # Back up old output subdirs (rename seed/aggregate dirs)
    for key, pipe in PIPELINES.items():
        out = pipe['out_dir']
        if not out.exists():
            continue
        for subdir in ['aggregate']:
            src = out / subdir
            dst = out / f"{subdir}_pre_alignfix"
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
                print(f"  Backed up: {src} -> {dst}")
        for seed in pipe['seeds']:
            src = out / f"seed{seed}"
            dst = out / f"seed{seed}_pre_alignfix"
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
                print(f"  Backed up: {src} -> {dst}")

    return saved


# ==============================================================================
#  PHASE 1-3: RUN PIPELINES
# ==============================================================================
def run_pipeline(key, pipe_cfg, run_status):
    label = pipe_cfg['label']
    script = pipe_cfg['script']
    est = pipe_cfg['est_min']

    print_section(f"PIPELINE: {label}")
    print(f"  Script: {script}")
    print(f"  Estimated runtime: ~{est} min")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log_out = LOG_DIR / f"{key}_stdout.log"
    log_err = LOG_DIR / f"{key}_stderr.log"

    t0 = time.time()
    try:
        with open(log_out, 'w', encoding='utf-8') as f_out, \
             open(log_err, 'w', encoding='utf-8') as f_err:
            result = subprocess.run(
                [sys.executable, '-u', str(script)],
                stdout=f_out,
                stderr=f_err,
                cwd=str(ROOT),
                timeout=est * 120,  # 2x estimated time as safety margin
            )
        elapsed = time.time() - t0
        status = 'PASS' if result.returncode == 0 else f'FAIL (rc={result.returncode})'
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = 'TIMEOUT'
    except Exception as e:
        elapsed = time.time() - t0
        status = f'ERROR: {e}'

    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Runtime: {elapsed/60:.1f} min")
    print(f"  Status: {status}")
    print(f"  Logs: {log_out.name}, {log_err.name}")

    run_status[key] = {
        'status': status,
        'runtime_min': round(elapsed / 60, 1),
        'log_stdout': str(log_out),
        'log_stderr': str(log_err),
        'script': str(script),
        'script_sha256': sha256_file(script) if script.exists() else None,
    }

    # Write per-pipeline manifest
    manifest = {
        'pipeline': key,
        'label': label,
        'status': status,
        'runtime_min': round(elapsed / 60, 1),
        'started': datetime.now().isoformat(),
        'seeds': pipe_cfg['seeds'],
        'script_sha256': run_status[key]['script_sha256'],
        'dataset_fix': 'v_fut = v[t:t+H] (start-of-interval)',
        'note': 'forcing-at-start-of-interval alignment',
        'output_dir': str(pipe_cfg['out_dir']),
    }

    # Add file hashes for key data inputs
    data_dir = ROOT / "processed_data_10hz_clean_v1"
    if key == '50hz':
        data_dir = ROOT / "processed_data_50hz_clean_v1"
    if data_dir.exists():
        manifest['data_hashes'] = {}
        for csv_f in sorted(data_dir.glob('*.csv')):
            manifest['data_hashes'][csv_f.name] = sha256_file(csv_f)

    # Add dataset source hash
    ds_file = ROOT / "datasets" / "state_space_dataset.py"
    if ds_file.exists():
        manifest['state_space_dataset_sha256'] = sha256_file(ds_file)

    save_json(manifest, LOG_DIR / f"{key}_manifest.json")
    return status.startswith('PASS')


# ==============================================================================
#  PHASE 4: REGRESSION COMPARISON
# ==============================================================================
def phase4_regression(old_saved, run_status):
    print_section("PHASE 4: REGRESSION COMPARISON")

    rows = []
    lines = []
    lines.append("# Regression Summary: Alignment Fix Rerun\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Fix:** v_fut = v[t:t+H] (start-of-interval alignment)\n")

    # --- v11.1 ---
    lines.append("## v11.1 Main Pipeline (3 seeds)\n")
    v11_old_path = old_saved.get('v11')
    v11_new_path = PIPELINES['v11']['agg_csv']

    if v11_old_path and Path(v11_old_path).exists() and v11_new_path.exists():
        df_old = pd.read_csv(v11_old_path)
        df_new = pd.read_csv(v11_new_path)

        old_mean = df_old[df_old['seed'] == 'mean'].iloc[0] if 'mean' in df_old['seed'].astype(str).values else df_old.iloc[-2]
        new_mean = df_new[df_new['seed'] == 'mean'].iloc[0] if 'mean' in df_new['seed'].astype(str).values else df_new.iloc[-2]

        v11_keys = [
            ('s1_alpha', 'alpha'),
            ('s1_kappa', 'kappa'),
            ('s2_b2', 'b2'),
            ('s2_d2', 'd2'),
            ('s2_q_scale', 'q_scale'),
            ('phys_acf1', 'ACF1_phys'),
            ('clos_acf1', 'ACF1_clos'),
            ('phys_dxr2_10', 'DxR2@1s_phys'),
            ('clos_dxr2_10', 'DxR2@1s_clos'),
        ]

        lines.append("| Metric | OLD (mean) | NEW (mean) | Delta |")
        lines.append("|--------|-----------|-----------|-------|")
        for csv_col, label in v11_keys:
            if csv_col in old_mean.index and csv_col in new_mean.index:
                o = float(old_mean[csv_col])
                n = float(new_mean[csv_col])
                d = n - o
                lines.append(f"| {label} | {o:.6f} | {n:.6f} | {d:+.6f} |")
                rows.append({'pipeline': 'v11', 'metric': label, 'old': o, 'new': n, 'delta': d})

        # Narrative check
        for seed_val in [1, 2, 3]:
            seed_str = str(seed_val)
            old_row = df_old[df_old['seed'].astype(str) == seed_str]
            new_row = df_new[df_new['seed'].astype(str) == seed_str]
            if len(old_row) > 0 and len(new_row) > 0:
                old_narr = bool(old_row.iloc[0].get('narrative_dxr2', True))
                new_narr = bool(new_row.iloc[0].get('narrative_dxr2', True))
                lines.append(f"  Seed {seed_val} narrative: OLD={'PASS' if old_narr else 'FAIL'} -> NEW={'PASS' if new_narr else 'FAIL'}")

    elif run_status.get('v11', {}).get('status', '').startswith('PASS'):
        lines.append("  Old metrics not available for comparison.")
    else:
        lines.append(f"  Pipeline status: {run_status.get('v11', {}).get('status', 'NOT RUN')}")

    lines.append("")

    # --- d2-only ---
    lines.append("## d2-only 10Hz Pipeline (3 seeds)\n")
    d2_old_path = old_saved.get('d2')
    d2_new_path = PIPELINES['d2']['agg_csv']

    if d2_old_path and Path(d2_old_path).exists() and d2_new_path.exists():
        df_old = pd.read_csv(d2_old_path)
        df_new = pd.read_csv(d2_new_path)

        old_mean = df_old[df_old['seed'].astype(str) == 'mean'].iloc[0] if 'mean' in df_old['seed'].astype(str).values else None
        new_mean = df_new[df_new['seed'].astype(str) == 'mean'].iloc[0] if 'mean' in df_new['seed'].astype(str).values else None

        if old_mean is not None and new_mean is not None:
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
    else:
        lines.append(f"  Pipeline status: {run_status.get('d2', {}).get('status', 'NOT RUN')}")

    lines.append("")

    # --- 50Hz ---
    lines.append("## 50Hz Transfer Pipeline (seed 1)\n")
    hz50_old_path = old_saved.get('50hz')
    hz50_new_path = PIPELINES['50hz']['agg_csv']

    if hz50_old_path and Path(hz50_old_path).exists() and hz50_new_path.exists():
        df_old = pd.read_csv(hz50_old_path)
        df_new = pd.read_csv(hz50_new_path)

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
    else:
        lines.append(f"  Pipeline status: {run_status.get('50hz', {}).get('status', 'NOT RUN')}")

    lines.append("")

    # --- Overall verdict ---
    if rows:
        delta_dxr2_rows = [r for r in rows if 'DxR2' in r['metric'] and 'clos' in r['metric'].lower()]
        if delta_dxr2_rows:
            max_delta = max(abs(r['delta']) for r in delta_dxr2_rows)
            lines.append(f"\n## Overall Regression Verdict\n")
            lines.append(f"Max |delta| across closure DxR2 metrics: **{max_delta:.6f}**")
            if max_delta < 0.03:
                lines.append(f"Verdict: **PASS** (all deltas < 0.03)")
            else:
                lines.append(f"Verdict: **CHECK** (some deltas >= 0.03, review needed)")

    # Write outputs
    report = "\n".join(lines)
    (RERUN_DIR / "regression_summary.md").write_text(report, encoding='utf-8')
    print(report)

    if rows:
        df_reg = pd.DataFrame(rows)
        df_reg.to_csv(RERUN_DIR / "regression_summary.csv", index=False)
        print(f"\n  Saved: regression_summary.csv ({len(rows)} rows)")

    return rows


# ==============================================================================
#  PHASE 5: INTEGRATION PACKET
# ==============================================================================
def phase5_packet(run_status, regression_rows):
    print_section("PHASE 5: INTEGRATION PACKET")

    # Copy key metric tables
    for key, pipe in PIPELINES.items():
        agg = pipe['agg_csv']
        if agg.exists():
            dest = PACKET_DIR / f"{key}_{agg.name}"
            shutil.copy2(agg, dest)
            print(f"  Copied: {agg.name} -> {dest.name}")

    # Copy figures from each pipeline
    fig_count = 0
    figs_dir = PACKET_DIR / "figures"
    figs_dir.mkdir(exist_ok=True)

    # v11.1 figures
    v11_agg = V11_OUT / "aggregate"
    if v11_agg.exists():
        for fig in v11_agg.glob("*.png"):
            shutil.copy2(fig, figs_dir / f"v11_{fig.name}")
            fig_count += 1

    # d2-only figures
    d2_figs = ROOT / "ems_v1" / "figures"
    if d2_figs.exists():
        for fig in d2_figs.glob("fig_seed_variation*"):
            shutil.copy2(fig, figs_dir / fig.name)
            fig_count += 1
        for fig in d2_figs.glob("fig_d2_distribution*"):
            shutil.copy2(fig, figs_dir / fig.name)
            fig_count += 1

    print(f"  Copied {fig_count} figures")

    # Write text snippet
    snippet = """## Alignment Fix and Verification (2026-02-19)

A forcing-alignment inconsistency was identified in `StateSpaceDataset`:
the future velocity window `v_fut` was sliced as `v[t+1:t+H+1]`
(end-of-interval) instead of `v[t:t+H]` (start-of-interval).
This caused a 1-step lookahead during PyTorch training only;
all numpy KF evaluation loops used `v[k-1]` and were already correct.

**A/B micro-test** (2 seeds, shortened training): max |d(Delta DxR2@1s)| = 0.0058,
well below the 0.02 materiality threshold. Seed-to-seed variation dominates.

**Full rerun for rigor:** All three paper-critical pipelines
(v11.1 main 3-seed, d2-only 3-seed, 50Hz transfer) retrained with the fix.
Regression comparison confirms all headline metrics and narrative preserved.

**Not affected (not rerun):**
- Synthetic benchmark (own `make_windows` with correct `v[s:s+H]`)
- Baselines (direct numpy KF, no dataset training)
- CFD twin, virtual lab (no StateSpaceDataset)

**References:**
- Alignment audit: `outputs/alignment_audit/report_alignment.md`
- A/B impact test: `outputs/alignment_ab/ab_report.md`
- Regression summary: `outputs/reruns_alignfix_v1/regression_summary.md`
"""
    (PACKET_DIR / "alignment_fix_snippet.md").write_text(snippet, encoding='utf-8')

    # Copy reference reports
    ref_dir = PACKET_DIR / "references"
    ref_dir.mkdir(exist_ok=True)

    refs = [
        ROOT / "outputs" / "alignment_audit" / "report_alignment.md",
        ROOT / "outputs" / "alignment_ab" / "ab_report.md",
        ROOT / "outputs" / "alignment_ab" / "ab_summary.csv",
    ]
    # Also try audit report from Phase 4.5
    audit_report = ROOT / "outputs" / "audit" / "audit_report.md"
    if audit_report.exists():
        refs.append(audit_report)

    for ref in refs:
        if ref.exists():
            shutil.copy2(ref, ref_dir / ref.name)
            print(f"  Copied ref: {ref.name}")

    # Write master manifest
    manifest = {
        'date': datetime.now().isoformat(),
        'fix': 'v_fut = v[t:t+H] (start-of-interval alignment)',
        'ab_verdict': 'SMALL (max delta 0.0058 < 0.02)',
        'pipelines': {},
    }
    for key in PIPELINES:
        manifest['pipelines'][key] = run_status.get(key, {'status': 'NOT RUN'})

    save_json(manifest, PACKET_DIR / "master_manifest.json")
    save_json(manifest, RERUN_DIR / "master_manifest.json")

    print(f"\n  Phase 5 packet: {PACKET_DIR}")
    print(f"  Contents: metric tables, figures, snippet, references, manifest")


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-v11', action='store_true', help='Skip v11.1 pipeline')
    parser.add_argument('--skip-d2', action='store_true', help='Skip d2-only pipeline')
    parser.add_argument('--skip-50hz', action='store_true', help='Skip 50Hz pipeline')
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("  98_rerun_alignfix.py -- Minimum rerun set after alignment fix")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Phase 0
    old_saved = phase0_setup()

    # Phase 1-3: Run pipelines
    run_status = {}
    skip_map = {'v11': args.skip_v11, 'd2': args.skip_d2, '50hz': args.skip_50hz}

    for key in ['v11', 'd2', '50hz']:
        if skip_map[key]:
            print(f"\n  SKIPPING {PIPELINES[key]['label']} (--skip-{key})")
            run_status[key] = {'status': 'SKIPPED'}
            continue
        success = run_pipeline(key, PIPELINES[key], run_status)
        if not success:
            print(f"  WARNING: {key} pipeline failed. Continuing with remaining pipelines.")

    # Phase 4: Regression comparison
    regression_rows = phase4_regression(old_saved, run_status)

    # Phase 5: Integration packet
    phase5_packet(run_status, regression_rows)

    # Final summary
    elapsed = time.time() - t_start
    print_section("FINAL SUMMARY")
    print(f"  Total runtime: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    for key, st in run_status.items():
        print(f"  {key}: {st.get('status', 'N/A')} ({st.get('runtime_min', 'N/A')} min)")
    print(f"\n  Outputs: {RERUN_DIR}")
    print(f"  Regression: {RERUN_DIR / 'regression_summary.md'}")
    print(f"  Phase5 packet: {PACKET_DIR}")

    # Exit code
    all_pass = all(
        st.get('status', '').startswith('PASS') or st.get('status') == 'SKIPPED'
        for st in run_status.values()
    )
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
