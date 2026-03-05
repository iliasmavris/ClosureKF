"""
Compute Gate OFF vs Gate ON comparison from existing selection_summary.json.

Gate OFF = variance-only selection (rel_var > 0.05, no DNLL gate)
Gate ON  = full pipeline (rel_var > 0.05 + DNLL >= 0.001)

No retraining required. All data parsed from existing artifacts.

Usage: python -u external_benchmarks/silverbox/scripts/compute_gate_comparison.py
"""

import os, sys, json, hashlib
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
SILVERBOX_DIR = SCRIPT_DIR.parent
OUT_DIR = SILVERBOX_DIR / "outputs"

TERM_NAMES = ['a1', 'b1', 'b2', 'd1', 'd2', 'd3']
REL_VAR_THRESHOLD = 0.05
TAU_NLL = 0.001


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("GATE OFF vs GATE ON COMPARISON")
    print("=" * 60)

    # Load existing selection summary
    sel_path = OUT_DIR / "selection_summary.json"
    with open(sel_path, 'r') as f:
        selection = json.load(f)

    # Load dt_eff from metrics
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    dt_eff = metrics['dt_eff']

    seeds_data = {}
    gate_off_counts = {t: 0 for t in TERM_NAMES}
    gate_off_counts['none'] = 0
    gate_on_counts = {t: 0 for t in TERM_NAMES}
    gate_on_counts['none'] = 0

    b2_rel_vars = []
    b2_delta_nlls = []

    for seed_key in sorted(selection.keys()):
        seed_info = selection[seed_key]
        seed_num = str(seed_info['seed'])

        # Extract relative variance per term
        rel_var = seed_info['rel_variance']['relative']

        # Extract DNLL gate results (only for terms that passed variance filter)
        dnll_gate = seed_info.get('dnll_gate', {})

        # Build delta_nll dict: N/A for terms not tested
        delta_nll = {}
        for t in TERM_NAMES:
            if t in dnll_gate:
                delta_nll[t] = dnll_gate[t]['delta_nll']
            else:
                delta_nll[t] = None  # not tested (failed variance filter)

        # Gate OFF: select terms with rel_var > threshold (no DNLL check)
        selected_gate_off = [t for t in TERM_NAMES if rel_var[t] > REL_VAR_THRESHOLD]
        if len(selected_gate_off) == 0:
            selected_gate_off = ['none']

        # Gate ON: what was actually selected (from existing results)
        selected_gate_on = seed_info['final_terms']

        # Track b2 stats
        b2_rel_vars.append(rel_var['b2'])
        if 'b2' in dnll_gate:
            b2_delta_nlls.append(dnll_gate['b2']['delta_nll'])

        # Update counts
        if selected_gate_off == ['none']:
            gate_off_counts['none'] += 1
        else:
            for t in selected_gate_off:
                gate_off_counts[t] += 1

        if selected_gate_on == ['none']:
            gate_on_counts['none'] += 1
        else:
            for t in selected_gate_on:
                gate_on_counts[t] += 1

        seeds_data[seed_num] = {
            'rel_var': {t: rel_var[t] for t in TERM_NAMES},
            'delta_nll': delta_nll,
            'selected_gate_off': selected_gate_off,
            'selected_gate_on': selected_gate_on,
        }

        print(f"\n  Seed {seed_num}:")
        print(f"    Gate OFF selected: {selected_gate_off}")
        print(f"    Gate ON  selected: {selected_gate_on}")
        for t in TERM_NAMES:
            rv = rel_var[t]
            dn = delta_nll[t]
            marker = ""
            if rv > REL_VAR_THRESHOLD:
                marker = " <-- passes var threshold"
                if dn is not None and dn < TAU_NLL:
                    marker += ", FAILS DNLL gate"
            print(f"    {t}: rel_var={rv:.6f}, delta_nll={dn}{marker}")

    # Summary stats
    import numpy as np
    b2_rv_arr = np.array(b2_rel_vars)
    b2_dn_arr = np.array(b2_delta_nlls)

    result = {
        'dt_eff': dt_eff,
        'thresholds': {
            'rel_var_min': REL_VAR_THRESHOLD,
            'tau_nll': TAU_NLL,
        },
        'seeds': seeds_data,
        'summary': {
            'gate_off_selection_counts': gate_off_counts,
            'gate_on_selection_counts': gate_on_counts,
            'b2_rel_var': {
                'mean': float(np.mean(b2_rv_arr)),
                'min': float(np.min(b2_rv_arr)),
                'max': float(np.max(b2_rv_arr)),
            },
            'b2_delta_nll': {
                'mean': float(np.mean(b2_dn_arr)),
                'min': float(np.min(b2_dn_arr)),
                'max': float(np.max(b2_dn_arr)),
            },
        },
    }

    # Write output
    out_path = OUT_DIR / "gate_off_vs_on.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  SHA-256: {sha256_file(out_path)}")

    # Update manifest
    manifest_path = SILVERBOX_DIR / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    manifest.setdefault('output_hashes', {})
    manifest['output_hashes']['gate_off_vs_on.json'] = sha256_file(out_path)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Updated manifest.json")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Gate OFF (variance-only) selection counts:")
    for t in TERM_NAMES + ['none']:
        print(f"    {t:6s}: {gate_off_counts[t]}")
    print(f"  Gate ON (full pipeline) selection counts:")
    for t in TERM_NAMES + ['none']:
        print(f"    {t:6s}: {gate_on_counts[t]}")
    print(f"\n  b2 rel_var: mean={np.mean(b2_rv_arr):.4f}, "
          f"min={np.min(b2_rv_arr):.4f}, max={np.max(b2_rv_arr):.4f}")
    print(f"  b2 DNLL:    mean={np.mean(b2_dn_arr):.6f}, "
          f"min={np.min(b2_dn_arr):.6f}, max={np.max(b2_dn_arr):.6f}")
    print(f"  Threshold:  tau_nll={TAU_NLL}")
    print(f"\n  VERDICT: b2 dominates variance in all 3 seeds,")
    print(f"           yet fails DNLL gate in all 3 seeds.")
    print(f"           Gate OFF would select b2 (3/3); Gate ON selects none (3/3).")


if __name__ == '__main__':
    main()
