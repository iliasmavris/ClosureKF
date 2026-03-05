#!/usr/bin/env python
"""
99_rerun_null_test.py
=====================
Rerun VL null test with correct condition selection from
phase4_discovery_v2_gate (d_p=0.003m, 18 inter + 6 cont).

Fixes: the original audit used phase4_discovery/ (d_p=0.03m, all intermittent)
which yielded only 3 conditions (N=6). The manuscript reports N=8.

Selects: 3 intermittent + 1 continuous = 4 conditions x 2 null types = 8 scenarios.
"""

import os, sys, time, shutil, tempfile, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
VL   = ROOT / 'virtual_lab_v1'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VL / 'scripts'))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# -- config ---------------------------------------------------------------
DATASETS_ROOT = VL / 'datasets_v2'
V2_GATE_CSV   = VL / 'outputs' / 'phase4_discovery_v2_gate' / 'discovery_summary.csv'
OUT_DIR       = VL / 'outputs' / 'audit'
OUT_CSV       = OUT_DIR / 'null_test_v2_summary.csv'

NULL_S1_EPOCHS   = 20
NULL_S1_PATIENCE = 5
NULL_S1_LR       = 1e-2
NULL_S1_SCHED    = 4
SEED = 99


def select_conditions(csv_path, n_inter=3, n_cont=1):
    """Select conditions from v2_gate CSV (correct source)."""
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'ok']
    inter = df[df['oracle_regime'] == 'intermittent'].sort_values('event_rate')
    cont  = df[df['oracle_regime'] == 'continuous'].sort_values('event_rate')

    selected = []
    if len(inter) >= n_inter:
        idxs = np.linspace(0, len(inter) - 1, n_inter, dtype=int)
        for i in idxs:
            selected.append(inter.iloc[i]['condition_id'])
    else:
        selected.extend(inter['condition_id'].tolist()[:n_inter])

    if len(cont) >= n_cont:
        selected.append(cont.iloc[len(cont) // 2]['condition_id'])
    elif len(cont) > 0:
        selected.append(cont.iloc[0]['condition_id'])

    return selected


def run_one_null(disc, cond_dir, cond_id, null_type, device, seed=SEED):
    """Run one null discovery: shuffle or reverse velocity."""
    import torch
    from torch.utils.data import DataLoader
    from datasets.state_space_dataset import StateSpaceDataset
    from models.kalman_forecaster import KalmanForecaster

    df = pd.read_csv(cond_dir / 'x_10hz.csv')
    v = df['velocity'].values.copy()

    rng = np.random.RandomState(seed)
    if null_type == 'shuffle':
        rng.shuffle(v)
    elif null_type == 'reverse':
        v = v[::-1].copy()
    df['velocity'] = v

    N = len(df)
    n_train = int(0.60 * N)
    n_val   = int(0.20 * N)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"null_{cond_id}_{null_type}_"))
    try:
        df_train.to_csv(tmp_dir / 'train.csv', index=False)
        df_val.to_csv(tmp_dir / 'val.csv', index=False)

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_ds = StateSpaceDataset(
            [str(tmp_dir / 'train.csv')], L=disc.S1_L, m=disc.S1_L, H=disc.S1_H,
            predict_deltas=False, normalize=False)
        val_ds = StateSpaceDataset(
            [str(tmp_dir / 'val.csv')], L=disc.S1_L, m=disc.S1_L, H=disc.S1_H,
            predict_deltas=False, normalize=False)

        if len(train_ds) < 10 or len(val_ds) < 5:
            return {'condition_id': cond_id, 'null_type': null_type,
                    'status': 'too_few_samples', 'selected_terms': []}

        train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
        val_ld   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

        model_s1 = KalmanForecaster(use_kappa=True).to(device)
        s1_best_val, _ = disc.train_s1(
            model_s1, train_ld, val_ld, device,
            NULL_S1_EPOCHS, NULL_S1_PATIENCE, NULL_S1_LR, NULL_S1_SCHED,
            tag=f"NULL-{null_type[:3]}-{cond_id}")

        s1_params = model_s1.param_summary()
        s1_pp = {
            'alpha': s1_params['alpha'], 'c': s1_params['c'],
            'vc': s1_params['vc'], 'kappa': s1_params['kappa'],
            'qx': s1_params['qx'], 'qu': s1_params['qu'],
            'R': s1_params['R'],
            'P0_xx': s1_params['P0_xx'], 'P0_uu': s1_params['P0_uu'],
        }

        t_tr = df_train['timestamp'].values
        x_tr = df_train['displacement'].values
        v_tr = df_train['velocity'].values
        t_vl = df_val['timestamp'].values
        x_vl = df_val['displacement'].values
        v_vl = df_val['velocity'].values

        cl_best, _, _, _ = disc.train_s2_scipy(
            s1_pp, t_tr, x_tr, v_tr, t_vl, x_vl, v_vl,
            tag=f"NULL-{null_type[:3]}-{cond_id}")

        selected_terms, _, rel_vars = disc.select_terms(
            cl_best, s1_pp, t_vl, x_vl, v_vl)

        return {
            'condition_id': cond_id, 'null_type': null_type,
            'status': 'ok', 'selected_terms': selected_terms,
            'rel_vars': {k: float(vv) for k, vv in rel_vars.items()} if rel_vars else {},
            's2_coeffs': {k: float(cl_best[k]) for k in disc.TERM_NAMES},
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    import torch
    import importlib.util

    device = torch.device('cpu')
    torch.set_num_threads(os.cpu_count() or 4)

    # Load discovery module
    script_path = VL / "scripts" / "23_run_discovery_v2.py"
    spec = importlib.util.spec_from_file_location("disc_v2", str(script_path))
    disc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(disc)

    # Select conditions from CORRECT source
    cond_ids = select_conditions(V2_GATE_CSV, n_inter=3, n_cont=1)
    print(f"Source CSV: {V2_GATE_CSV}")
    print(f"Selected conditions: {cond_ids}")
    print(f"Scenarios: {len(cond_ids)} conditions x 2 null types = {2*len(cond_ids)}")
    print()

    null_types = ['shuffle', 'reverse']
    results = []
    t0 = time.time()

    for cid in cond_ids:
        cond_dir = DATASETS_ROOT / cid
        if not cond_dir.exists():
            print(f"  SKIP {cid}: directory not found")
            continue
        for nt in null_types:
            print(f"  --- NULL [{nt}] {cid} ---")
            t1 = time.time()
            try:
                r = run_one_null(disc, cond_dir, cid, nt, device, seed=SEED)
            except Exception as ex:
                print(f"  ERROR: {ex}")
                r = {'condition_id': cid, 'null_type': nt,
                     'status': 'error', 'selected_terms': [], 'error': str(ex)}
            elapsed = time.time() - t1
            r['elapsed_s'] = round(elapsed, 1)
            results.append(r)
            sel = r.get('selected_terms', [])
            print(f"  Selected: {sel} [{elapsed:.1f}s]")
            print()

    total_elapsed = time.time() - t0
    print(f"Total: {len(results)} runs in {total_elapsed:.0f}s")

    # Analyze
    ok_results = [r for r in results if r['status'] == 'ok']
    N = len(ok_results)

    # Per-term counts
    TERM_NAMES = ['a1', 'b1', 'b2', 'd1', 'd2', 'd3']
    term_counts = Counter()
    for r in ok_results:
        for t in r.get('selected_terms', []):
            term_counts[t] += 1

    print(f"\n{'='*50}")
    print(f"NULL TEST RESULTS (N={N})")
    print(f"{'='*50}")
    for t in TERM_NAMES:
        c = term_counts.get(t, 0)
        print(f"  {t:3s}: {c}/{N}")

    # Categorize by term SET
    set_counts = Counter()
    for r in ok_results:
        sel = sorted(r.get('selected_terms', []))
        key = '+'.join(sel) if sel else 'None'
        set_counts[key] += 1

    print(f"\nBy term set:")
    for key in ['None'] + sorted(k for k in set_counts if k != 'None'):
        c = set_counts.get(key, 0)
        print(f"  {key:10s}: {c}/{N}")

    b2_count = term_counts.get('b2', 0)
    print(f"\nb2 selected: {b2_count}/{N} ({100*b2_count/max(N,1):.0f}%)")
    print(f"PASS: {'YES' if b2_count/max(N,1) <= 0.75 else 'NO'}")

    # Write CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        sel = r.get('selected_terms', [])
        rows.append({
            'condition_id': r['condition_id'],
            'null_type': r['null_type'],
            'status': r['status'],
            'selected_terms': '+'.join(sel) if sel else '(none)',
            'b2_selected': 'b2' in sel,
            'elapsed_s': r.get('elapsed_s', 0),
        })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote: {OUT_CSV}")

    # Also overwrite the old CSV for consistency
    old_csv = OUT_DIR / 'null_test_summary.csv'
    pd.DataFrame(rows).to_csv(old_csv, index=False)
    print(f"Wrote: {old_csv}")


if __name__ == '__main__':
    main()
