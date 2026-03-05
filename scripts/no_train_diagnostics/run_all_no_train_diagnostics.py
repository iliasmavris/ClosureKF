"""
Master Runner: Run All 10 No-Training Diagnostics
==================================================
NO TRAINING. Runs each diagnostic sequentially, collects results,
writes master README.
"""
import sys, time, json, subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import utils_no_train as U

OUT = U.OUT_ROOT
OUT.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("diag01", "diag01_horizon_skill_curve.py",        "Horizon Skill Curves h=1..200"),
    ("diag02", "diag02_single_open_loop_rollout.py",    "Single Continuous Open-Loop"),
    ("diag03", "diag03_event_vs_nonevent_skill.py",     "Event vs Non-Event Skill"),
    ("diag04", "diag04_event_timing_score.py",          "Event Timing Score"),
    ("diag05", "diag05_coverage_vs_horizon.py",         "Coverage vs Horizon"),
    ("diag06", "diag06_pit_histogram.py",               "PIT Histogram"),
    ("diag07", "diag07_conditional_innovation_acf.py",  "Conditional Innovation ACF"),
    ("diag08", "diag08_innovation_input_ccf.py",        "Innovation-Input CCF"),
    ("diag09", "diag09_initialization_sensitivity.py",  "Initialization Sensitivity"),
    ("diag10", "diag10_leave_one_event_out.py",         "Leave-One-Event-Out"),
]


def run_diagnostic(script_name, description):
    """Run a diagnostic script as subprocess. Returns (success, elapsed)."""
    script_path = HERE / script_name
    print(f"\n{'='*70}")
    print(f"  Running: {description}")
    print(f"  Script:  {script_name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=600,
            encoding='utf-8', errors='replace',
        )
        elapsed = time.time() - t0
        print(result.stdout)
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[:500]}")
            print(f"  [FAILED] {script_name} (exit code {result.returncode}, {elapsed:.0f}s)")
            return False, elapsed
        print(f"  [OK] {script_name} ({elapsed:.0f}s)")
        return True, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] {script_name} ({elapsed:.0f}s)")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {script_name}: {e}")
        return False, elapsed


def main():
    t0_global = time.time()
    print("=" * 70)
    print("NO-TRAINING DIAGNOSTICS: Master Runner")
    print("Running 10 diagnostics sequentially")
    print(f"Output: {OUT}")
    print("=" * 70)

    results = []
    for diag_id, script, desc in SCRIPTS:
        ok, elapsed = run_diagnostic(script, desc)
        results.append({
            'diag': diag_id, 'script': script, 'description': desc,
            'success': ok, 'elapsed_s': elapsed,
        })

    # Summary
    n_pass = sum(1 for r in results if r['success'])
    n_fail = sum(1 for r in results if not r['success'])
    total_time = time.time() - t0_global

    print(f"\n{'='*70}")
    print(f"RESULTS: {n_pass} PASS, {n_fail} FAIL ({total_time:.0f}s total)")
    print(f"{'='*70}")
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        print(f"  [{status}] {r['diag']:7s} {r['description']:40s} ({r['elapsed_s']:.0f}s)")

    # Collect summaries from each diagnostic
    summaries = {}
    for diag_id, _, _ in SCRIPTS:
        summary_path = OUT / diag_id / f"summary_{diag_id}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries[diag_id] = json.load(f)

    # Write master README
    readme_lines = [
        "# No-Training Diagnostics: Master Summary",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** Closure (2t) -- final paper model",
        f"**Total runtime:** {total_time:.0f}s",
        f"**Results:** {n_pass}/10 PASS",
        "",
        "## Diagnostics Run",
        "",
        "| # | Diagnostic | Status | Time |",
        "|---|-----------|--------|------|",
    ]
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        readme_lines.append(
            f"| {r['diag']} | {r['description']} | {status} | {r['elapsed_s']:.0f}s |")

    readme_lines.extend(["", "## Key Findings", ""])

    # Diag01: Skill curves
    if 'diag01' in summaries:
        s = summaries['diag01']
        readme_lines.append("### 1. Horizon Skill Curves")
        for mode in ['oracle', 'persistence', 'no_forcing']:
            c = s.get('crossover_s', {}).get(mode)
            p = s.get('plateau_dxr2_h200', {}).get(mode)
            readme_lines.append(
                f"- **{mode}:** crossover at {c:.1f}s, DxR2@200={p:+.3f}" if c and p
                else f"- **{mode}:** no crossover")
        readme_lines.append("")

    # Diag02: Single rollout
    if 'diag02' in summaries:
        s = summaries['diag02']
        readme_lines.append("### 2. Single Continuous Rollout")
        for mode in ['oracle', 'persistence', 'no_forcing']:
            m = s.get('modes', {}).get(mode, {})
            readme_lines.append(
                f"- **{mode}:** RMSE={m.get('rmse', 'N/A'):.4f}, "
                f"drift={m.get('final_drift', 'N/A'):.4f}, "
                f"event_misses={m.get('event_misses', 'N/A')}/{m.get('n_events', 'N/A')}")
        readme_lines.append("")

    # Diag04: Event timing
    if 'diag04' in summaries:
        s = summaries['diag04']
        readme_lines.append("### 4. Event Timing Score")
        for m in s.get('metrics', []):
            readme_lines.append(
                f"- **h={m['horizon']}:** ROC AUC={m['roc_auc']:.3f}, "
                f"PR AUC={m['pr_auc']:.3f}, F1={m['best_f1']:.3f}")
        readme_lines.append("")

    # Diag05: Coverage
    if 'diag05' in summaries:
        s = summaries['diag05']
        readme_lines.append("### 5. Coverage")
        readme_lines.append(
            f"- h=1-10: mean coverage = {s.get('mean_coverage_h1_10', 'N/A'):.3f}")
        readme_lines.append(
            f"- h=50: coverage = {s.get('coverage_at_h50', 'N/A'):.3f}")
        readme_lines.append(
            f"- Over-confident horizons: {s.get('n_overconfident_horizons', 'N/A')}")
        readme_lines.append("")

    # Final questions
    readme_lines.extend([
        "## Answers to Key Questions",
        "",
        "### Where does skill come from: event vs non-event?",
        "*See Diag03 event/non-event decomposition.*",
        "",
        "### Does oracle help materially beyond persistence?",
        "*See Diag01 oracle-persistence gap and Diag03 conditional metrics.*",
        "",
        "### Are predictive intervals calibrated?",
        "*See Diag05 (coverage) and Diag06 (PIT). Both provide horizon-resolved calibration.*",
        "",
        "### Is innovation structure concentrated near events/high energy?",
        "*See Diag07 conditional ACF and Diag08 cross-correlation.*",
        "",
        "### Are metrics dominated by a few events?",
        "*See Diag10 leave-one-event-out influence.*",
        "",
        "### Is the model stable under perturbations and continuous rollout?",
        "*See Diag09 (initialization sensitivity) and Diag02 (continuous rollout).*",
    ])

    with open(OUT / "README.md", 'w') as f:
        f.write('\n'.join(readme_lines))
    print(f"\nSaved: {OUT / 'README.md'}")

    # Save run log
    with open(OUT / "run_log.json", 'w') as f:
        json.dump({
            'results': results,
            'total_time_s': total_time,
            'n_pass': n_pass,
            'n_fail': n_fail,
        }, f, indent=2)


if __name__ == '__main__':
    main()
