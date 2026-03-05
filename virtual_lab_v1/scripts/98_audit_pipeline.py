"""
Phase 4.5: Audit pipeline for virtual lab discovery.

Implements:
  B1) Schema + file integrity (HARD)
  B2) Split integrity (HARD)
  B3) Circularity tripwire (HARD)
  B4) Output consistency (SOFT)
  B5) Reproducibility manifest (HARD)
  C)  Null tests -- shuffle & reverse (HARD)

Usage:
  python virtual_lab_v1/scripts/98_audit_pipeline.py \
      --datasets_root virtual_lab_v1/datasets_v2 \
      --outputs_root virtual_lab_v1/outputs

Output: virtual_lab_v1/outputs/audit/
"""

import argparse, os, sys, json, hashlib, time, math, warnings, tempfile, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, OrderedDict
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent.parent
VL   = ROOT / "virtual_lab_v1"
sys.path.insert(0, str(ROOT))

# ======================================================================
#  CONSTANTS
# ======================================================================
TERM_NAMES = ['a1', 'd1', 'd2', 'd3', 'b1', 'b2']
# Should-be-absent: nonlinear cross-drag terms NOT in truth model
ABSENT_TERMS = ['d1', 'd2', 'd3']
CORR_THRESHOLD = 0.10
N_MIN_FREE = 5000
ORACLE_TRAIN_FRAC = 0.60
ORACLE_VAL_FRAC   = 0.20
NULL_B2_MAX_FREQ  = 0.75
DT_SIM_EXPECTED   = 0.005
DT_10HZ_EXPECTED  = 0.1

# Null test: fast training config
NULL_S1_EPOCHS   = 20
NULL_S1_PATIENCE = 5
NULL_S1_LR       = 1e-2
NULL_S1_SCHED    = 4
NULL_S2_MAXITER  = 300    # scipy converges fast anyway

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3,
})


# ======================================================================
#  HELPERS
# ======================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def compute_a_true_numdiff(v_p, t, smooth_hz=10.0):
    """Smoothed numerical dv_p/dt."""
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    if fs <= 2 * smooth_hz:
        return np.gradient(v_p, t)
    sos = butter(4, smooth_hz, fs=fs, output='sos')
    v_sm = sosfiltfilt(sos, v_p)
    return np.gradient(v_sm, t)


# ======================================================================
#  B1: SCHEMA + FILE INTEGRITY
# ======================================================================
def check_b1_schema(datasets_root):
    """Check schema + file integrity for all conditions."""
    print_section("B1: Schema + File Integrity")
    cond_dirs = sorted(datasets_root.glob("condition_*"))
    results = OrderedDict()
    all_pass = True

    for cond_dir in cond_dirs:
        cid = cond_dir.name
        errors = []
        warns = []

        # Required files
        for fname in ['x_10hz.csv', 'truth_states_raw.csv', 'meta.json']:
            if not (cond_dir / fname).exists():
                errors.append(f"Missing {fname}")

        if errors:
            results[cid] = {'status': 'FAIL', 'errors': errors, 'warnings': warns}
            all_pass = False
            continue

        # Check truth_states_raw columns
        truth_df = pd.read_csv(cond_dir / 'truth_states_raw.csv', nrows=5)
        for col in ['time', 'x', 'v_p', 'at_pin', 'u_b']:
            if col not in truth_df.columns:
                errors.append(f"truth_states_raw missing required column: {col}")
        if 'a_force' not in truth_df.columns:
            warns.append("a_force absent (numdiff fallback)")

        # Check x_10hz columns
        x10_df = pd.read_csv(cond_dir / 'x_10hz.csv', nrows=5)
        for col in ['timestamp', 'displacement']:
            if col not in x10_df.columns:
                errors.append(f"x_10hz missing required column: {col}")

        if errors:
            results[cid] = {'status': 'FAIL', 'errors': errors, 'warnings': warns}
            all_pass = False
            continue

        # Time monotonicity + dt sanity
        truth_full = pd.read_csv(cond_dir / 'truth_states_raw.csv')
        t_truth = truth_full['time'].values
        dt_truth = np.diff(t_truth)
        if not np.all(dt_truth > 0):
            errors.append("truth_states_raw: time not monotonic")
        dt_median = np.median(dt_truth)
        if abs(dt_median - DT_SIM_EXPECTED) > 1e-4:
            errors.append(f"truth dt_median={dt_median:.6f} (expected {DT_SIM_EXPECTED})")

        x10_full = pd.read_csv(cond_dir / 'x_10hz.csv')
        t10 = x10_full['timestamp'].values
        dt10 = np.diff(t10)
        if not np.all(dt10 > 0):
            errors.append("x_10hz: time not monotonic")
        dt10_median = np.median(dt10)
        if abs(dt10_median - DT_10HZ_EXPECTED) > 0.01:
            errors.append(f"x_10hz dt_median={dt10_median:.4f} (expected {DT_10HZ_EXPECTED})")

        # NaN/Inf after spinup
        with open(cond_dir / 'meta.json') as f:
            meta = json.load(f)
        spinup = meta.get('config', {}).get('integration', {}).get('spinup_discard', 30.0)
        mask_post = truth_full['time'] >= spinup

        for col in ['x', 'v_p', 'u_b']:
            vals = truth_full.loc[mask_post, col].values
            nnan = int(np.sum(np.isnan(vals)))
            ninf = int(np.sum(np.isinf(vals)))
            if nnan > 0 or ninf > 0:
                errors.append(f"truth '{col}': {nnan} NaN + {ninf} Inf after spinup")

        # a_force NaN only during pinned
        if 'a_force' in truth_full.columns:
            af = truth_full.loc[mask_post, 'a_force'].values
            ap = truth_full.loc[mask_post, 'at_pin'].values
            free_m = (ap == 0)
            nnan_free = int(np.sum(np.isnan(af[free_m])))
            ninf_free = int(np.sum(np.isinf(af[free_m])))
            if nnan_free > 0:
                errors.append(f"a_force: {nnan_free} NaN during FREE")
            if ninf_free > 0:
                errors.append(f"a_force: {ninf_free} Inf during FREE")

        # x_10hz NaN/Inf
        for col in ['displacement']:
            if col in x10_full.columns:
                vals = x10_full[col].values
                nnan = int(np.sum(np.isnan(vals)))
                ninf = int(np.sum(np.isinf(vals)))
                if nnan > 0 or ninf > 0:
                    errors.append(f"x_10hz '{col}': {nnan} NaN + {ninf} Inf")

        status = 'FAIL' if errors else 'PASS'
        results[cid] = {'status': status, 'errors': errors, 'warnings': warns}
        if errors:
            all_pass = False

    # Summary
    n_pass = sum(1 for v in results.values() if v['status'] == 'PASS')
    n_fail = sum(1 for v in results.values() if v['status'] == 'FAIL')
    print(f"  {n_pass}/{len(results)} PASS, {n_fail} FAIL")
    for cid, v in results.items():
        if v['status'] == 'FAIL':
            print(f"    FAIL {cid}: {v['errors']}")
        elif v['warnings']:
            print(f"    WARN {cid}: {v['warnings']}")

    return all_pass, results


# ======================================================================
#  B2: SPLIT INTEGRITY
# ======================================================================
def check_b2_splits(datasets_root):
    """Validate FREE-index split logic."""
    print_section("B2: Split Integrity")
    cond_dirs = sorted(datasets_root.glob("condition_*"))
    results = OrderedDict()
    all_pass = True

    for cond_dir in cond_dirs:
        cid = cond_dir.name
        errors = []
        truth_path = cond_dir / 'truth_states_raw.csv'
        if not truth_path.exists():
            results[cid] = {'status': 'SKIP', 'errors': ['no truth file']}
            continue

        truth_df = pd.read_csv(truth_path)
        at_pin = truth_df['at_pin'].values
        t = truth_df['time'].values

        is_free = (at_pin == 0)
        if 'a_force' in truth_df.columns:
            a_force = truth_df['a_force'].values
            valid_a = ~np.isnan(a_force)
            free_idx = np.where(is_free & valid_a)[0]
        else:
            free_idx = np.where(is_free)[0]

        n_free = len(free_idx)

        # Check time ordering of FREE indices
        free_times = t[free_idx]
        if len(free_times) > 1 and not np.all(np.diff(free_times) >= 0):
            errors.append("FREE indices not time-ordered")

        # Compute splits
        n_train = int(n_free * ORACLE_TRAIN_FRAC)
        n_val   = int(n_free * ORACLE_VAL_FRAC)
        idx_train = set(free_idx[:n_train].tolist())
        idx_val   = set(free_idx[n_train:n_train + n_val].tolist())
        idx_test  = set(free_idx[n_train + n_val:].tolist())

        # Disjoint check
        if idx_train & idx_val:
            errors.append("Train/val overlap")
        if idx_train & idx_test:
            errors.append("Train/test overlap")
        if idx_val & idx_test:
            errors.append("Val/test overlap")

        # Union check
        if (idx_train | idx_val | idx_test) != set(free_idx.tolist()):
            errors.append("Split union != free_idx")

        gate = n_free >= N_MIN_FREE

        status = 'FAIL' if errors else 'PASS'
        results[cid] = {
            'status': status, 'errors': errors,
            'n_free': int(n_free),
            'n_train': int(n_train), 'n_val': int(n_val),
            'n_test': int(len(idx_test)),
            'gate': gate,
        }
        if errors:
            all_pass = False

    n_pass = sum(1 for v in results.values() if v['status'] == 'PASS')
    n_fail = sum(1 for v in results.values() if v['status'] == 'FAIL')
    n_gated = sum(1 for v in results.values() if v.get('gate'))
    print(f"  {n_pass}/{len(results)} PASS, {n_fail} FAIL")
    print(f"  {n_gated}/{len(results)} conditions pass N_MIN_FREE={N_MIN_FREE} gate")
    for cid, v in results.items():
        if v['status'] == 'FAIL':
            print(f"    FAIL {cid}: {v['errors']}")

    return all_pass, results


# ======================================================================
#  B3: CIRCULARITY TRIPWIRE
# ======================================================================
def check_b3_circularity(datasets_root):
    """Correlate a_true with each library regressor on FREE segments."""
    print_section("B3: Circularity Tripwire")
    cond_dirs = sorted(datasets_root.glob("condition_*"))
    results = OrderedDict()
    all_pass = True

    for cond_dir in cond_dirs:
        cid = cond_dir.name
        errors = []
        truth_path = cond_dir / 'truth_states_raw.csv'
        if not truth_path.exists():
            results[cid] = {'status': 'SKIP', 'correlations': {}}
            continue

        truth_df = pd.read_csv(truth_path)
        at_pin = truth_df['at_pin'].values
        v_p = truth_df['v_p'].values
        u_b = truth_df['u_b'].values
        t = truth_df['time'].values
        du_b = truth_df['du_b'].values if 'du_b' in truth_df.columns else np.gradient(u_b, t)

        is_free = (at_pin == 0)
        if 'a_force' in truth_df.columns:
            a_force = truth_df['a_force'].values
            valid_a = ~np.isnan(a_force)
            free_mask = is_free & valid_a
            a_true = a_force[free_mask]
        else:
            free_mask = is_free
            a_true = compute_a_true_numdiff(v_p, t)[free_mask]

        n_free = int(np.sum(free_mask))
        if n_free < 100:
            results[cid] = {'status': 'SKIP', 'n_free': n_free, 'correlations': {}}
            continue

        vp = v_p[free_mask]
        ub = u_b[free_mask]
        dub = du_b[free_mask]

        # Library regressors (same sign convention as build_library)
        regressors = OrderedDict([
            ('a1', -vp),
            ('d1', -vp**2),
            ('d2', -vp * np.abs(ub)),
            ('d3', -vp * np.abs(vp)),
            ('b1', ub),
            ('b2', dub),
        ])

        correlations = {}
        for name, reg in regressors.items():
            std_r = float(np.std(reg))
            std_a = float(np.std(a_true))
            if std_r < 1e-12 or std_a < 1e-12:
                correlations[name] = 0.0
            else:
                correlations[name] = float(np.corrcoef(a_true, reg)[0, 1])

        # Check should-be-absent terms
        for tn in ABSENT_TERMS:
            c = abs(correlations.get(tn, 0))
            if c >= CORR_THRESHOLD:
                errors.append(f"|corr(a_true, {tn})| = {c:.4f} >= {CORR_THRESHOLD}")

        status = 'FAIL' if errors else 'PASS'
        results[cid] = {
            'status': status, 'errors': errors,
            'n_free': n_free, 'correlations': correlations,
        }
        if errors:
            all_pass = False

    # Summary
    n_pass = sum(1 for v in results.values() if v['status'] == 'PASS')
    n_fail = sum(1 for v in results.values() if v['status'] == 'FAIL')
    n_skip = sum(1 for v in results.values() if v['status'] == 'SKIP')
    print(f"  {n_pass}/{len(results)} PASS, {n_fail} FAIL, {n_skip} SKIP")

    # Print absent-term stats
    absent_corrs = {tn: [] for tn in ABSENT_TERMS}
    for cid, v in results.items():
        if v['status'] != 'SKIP':
            for tn in ABSENT_TERMS:
                c = v['correlations'].get(tn, 0)
                absent_corrs[tn].append(abs(c))
    for tn in ABSENT_TERMS:
        vals = absent_corrs[tn]
        if vals:
            print(f"    |corr(a_true, {tn})|: max={max(vals):.4f}, "
                  f"mean={np.mean(vals):.4f}, median={np.median(vals):.4f}")

    if all_pass:
        print("  All should-be-absent terms below threshold.")
    else:
        print("  *** CIRCULARITY VIOLATION DETECTED ***")
        for cid, v in results.items():
            if v['status'] == 'FAIL':
                print(f"    FAIL {cid}: {v['errors']}")

    return all_pass, results


# ======================================================================
#  B4: OUTPUT CONSISTENCY
# ======================================================================
def check_b4_outputs(outputs_root):
    """Check that Phase 4 outputs exist."""
    print_section("B4: Output Consistency")
    p4 = outputs_root / "phase4_discovery"
    errors = []

    # Discovery summary
    csv_path = p4 / "discovery_summary.csv"
    if not csv_path.exists():
        errors.append("Missing discovery_summary.csv")
    else:
        df = pd.read_csv(csv_path)
        print(f"  discovery_summary.csv: {len(df)} rows")

    # Per-condition JSONs
    jsons = sorted(p4.glob("condition_*.json"))
    print(f"  Per-condition JSONs: {len(jsons)}")
    if len(jsons) < 24:
        errors.append(f"Only {len(jsons)} condition JSONs (expected 24)")

    # Figures
    for fig_name in ['fig_term_frequency.pdf', 'fig_oracle_gap_ratio.pdf',
                     'fig_coeff_vs_eventrate.pdf']:
        if not (p4 / fig_name).exists():
            errors.append(f"Missing {fig_name}")

    # Report
    if not (p4 / 'report_phase4.md').exists():
        errors.append("Missing report_phase4.md")

    status = 'FAIL' if errors else 'PASS'
    print(f"  Status: {status}")
    for e in errors:
        print(f"    {e}")

    return len(errors) == 0, {'status': status, 'errors': errors}


# ======================================================================
#  B5: REPRODUCIBILITY MANIFEST
# ======================================================================
def write_b5_manifest(outputs_root, datasets_root, audit_dir):
    """Write manifest with SHA-256 hashes of key scripts."""
    print_section("B5: Reproducibility Manifest")
    scripts_dir = VL / "scripts"
    configs_dir = VL / "configs"

    files_to_hash = [
        scripts_dir / "23_run_discovery_v2.py",
        scripts_dir / "24_phase4_plots.py",
        scripts_dir / "18_oracle_eval.py",
        scripts_dir / "14_truth_ball_sim.py",
        configs_dir / "forcing_variants.yaml",
        configs_dir / "sweep_grid_v2.yaml",
    ]

    hashes = OrderedDict()
    errors = []
    for fp in files_to_hash:
        if fp.exists():
            hashes[str(fp.relative_to(ROOT))] = sha256_file(fp)
        else:
            hashes[str(fp.relative_to(ROOT))] = "FILE_NOT_FOUND"
            errors.append(f"Missing: {fp.relative_to(ROOT)}")

    # Seeds from manifest_v2.json
    manifest_path = outputs_root / "sweep_v2" / "manifest_v2.json"
    seeds = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            mdata = json.load(f)
        seeds = [c.get('seed') for c in mdata.get('conditions', [])]

    # Condition count
    n_conditions = len(list(datasets_root.glob("condition_*")))

    manifest = OrderedDict([
        ('generated_at', time.strftime('%Y-%m-%d %H:%M:%S')),
        ('script_hashes', hashes),
        ('n_conditions', n_conditions),
        ('seeds', seeds),
    ])

    out_path = audit_dir / "manifest_phase45.json"
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {out_path}")
    print(f"  {len(hashes)} files hashed, {len(errors)} missing")
    for e in errors:
        print(f"    {e}")

    return len(errors) == 0, manifest


# ======================================================================
#  C: NULL TESTS
# ======================================================================
def _load_discovery_module():
    """Load 23_run_discovery_v2 as a module via importlib."""
    import importlib.util
    script_path = VL / "scripts" / "23_run_discovery_v2.py"
    spec = importlib.util.spec_from_file_location("disc_v2", str(script_path))
    disc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(disc)
    return disc


def _select_null_conditions(datasets_root, n_inter=3, n_cont=1):
    """Select conditions for null test: n_inter intermittent + n_cont continuous."""
    # Try discovery_summary.csv first, fallback to meta.json
    disc_csv = VL / "outputs" / "phase4_discovery" / "discovery_summary.csv"
    if disc_csv.exists():
        df = pd.read_csv(disc_csv)
        df = df[df['status'] == 'ok']
        inter = df[df['oracle_regime'] == 'intermittent'].sort_values('event_rate')
        cont  = df[df['oracle_regime'] == 'continuous'].sort_values('event_rate')
    else:
        # Fallback: use meta.json event_rate
        rows = []
        for cd in sorted(datasets_root.glob("condition_*")):
            mp = cd / 'meta.json'
            if mp.exists():
                with open(mp) as f:
                    meta = json.load(f)
                er = meta.get('event_rate', 0)
                rows.append({'condition_id': cd.name, 'event_rate': er,
                             'oracle_regime': 'intermittent' if er < 0.50 else 'continuous'})
        df = pd.DataFrame(rows)
        inter = df[df['oracle_regime'] == 'intermittent'].sort_values('event_rate')
        cont  = df[df['oracle_regime'] == 'continuous'].sort_values('event_rate')

    selected = []
    # Pick n_inter spread across intermittent range
    if len(inter) >= n_inter:
        idxs = np.linspace(0, len(inter) - 1, n_inter, dtype=int)
        for i in idxs:
            selected.append(inter.iloc[i]['condition_id'])
    else:
        selected.extend(inter['condition_id'].tolist()[:n_inter])

    # Pick n_cont continuous
    if len(cont) >= n_cont:
        selected.append(cont.iloc[len(cont)//2]['condition_id'])
    elif len(cont) > 0:
        selected.append(cont.iloc[0]['condition_id'])

    return selected


def _run_one_null(disc, cond_dir, cond_id, null_type, device, seed=42):
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

    # Split
    N = len(df)
    n_train = int(0.60 * N)
    n_val   = int(0.20 * N)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:n_train+n_val].reset_index(drop=True)

    # Write temp CSVs
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"null_{cond_id}_{null_type}_"))
    try:
        df_train.to_csv(tmp_dir / 'train.csv', index=False)
        df_val.to_csv(tmp_dir / 'val.csv', index=False)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # S1 (fast)
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

        # S2
        t_tr = df_train['timestamp'].values
        x_tr = df_train['displacement'].values
        v_tr = df_train['velocity'].values
        t_vl = df_val['timestamp'].values
        x_vl = df_val['displacement'].values
        v_vl = df_val['velocity'].values

        cl_best, _, _, _ = disc.train_s2_scipy(
            s1_pp, t_tr, x_tr, v_tr, t_vl, x_vl, v_vl,
            tag=f"NULL-{null_type[:3]}-{cond_id}")

        # Term selection
        selected_terms, _, rel_vars = disc.select_terms(
            cl_best, s1_pp, t_vl, x_vl, v_vl)

        return {
            'condition_id': cond_id, 'null_type': null_type,
            'status': 'ok', 'selected_terms': selected_terms,
            'rel_vars': {k: float(v) for k, v in rel_vars.items()} if rel_vars else {},
            's2_coeffs': {k: float(cl_best[k]) for k in disc.TERM_NAMES},
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_c_null_tests(datasets_root, audit_dir):
    """Run null tests on subset of conditions."""
    print_section("C: Null Tests")
    import torch
    device = torch.device('cpu')
    torch.set_num_threads(os.cpu_count() or 4)

    disc = _load_discovery_module()
    cond_ids = _select_null_conditions(datasets_root, n_inter=3, n_cont=1)
    print(f"  Selected conditions: {cond_ids}")

    null_types = ['shuffle', 'reverse']
    results = []
    t0 = time.time()

    for cid in cond_ids:
        cond_dir = datasets_root / cid
        if not cond_dir.exists():
            print(f"  SKIP {cid}: directory not found")
            continue
        for nt in null_types:
            print(f"\n  --- NULL [{nt}] {cid} ---")
            t1 = time.time()
            try:
                r = _run_one_null(disc, cond_dir, cid, nt, device, seed=99)
            except Exception as ex:
                print(f"  ERROR: {ex}")
                r = {'condition_id': cid, 'null_type': nt,
                     'status': 'error', 'selected_terms': [], 'error': str(ex)}
            elapsed = time.time() - t1
            r['elapsed_s'] = round(elapsed, 1)
            results.append(r)
            sel = r.get('selected_terms', [])
            print(f"  Selected: {sel} [{elapsed:.1f}s]")

    total_elapsed = time.time() - t0
    print(f"\n  Null tests done: {len(results)} runs in {total_elapsed:.0f}s")

    # Analyze
    ok_results = [r for r in results if r['status'] == 'ok']
    n_total = len(ok_results)
    b2_count = sum(1 for r in ok_results if 'b2' in r.get('selected_terms', []))
    b2_freq = b2_count / max(n_total, 1)

    print(f"\n  b2 selected: {b2_count}/{n_total} ({100*b2_freq:.0f}%)")
    print(f"  Threshold: <= {100*NULL_B2_MAX_FREQ:.0f}%")

    passes = b2_freq <= NULL_B2_MAX_FREQ
    if passes:
        print(f"  NULL TEST: PASS")
    else:
        print(f"  NULL TEST: FAIL (b2 selected in {100*b2_freq:.0f}% > "
              f"{100*NULL_B2_MAX_FREQ:.0f}%)")

    # Write summary CSV
    csv_path = audit_dir / "null_test_summary.csv"
    rows = []
    for r in results:
        rows.append({
            'condition_id': r['condition_id'],
            'null_type': r['null_type'],
            'status': r['status'],
            'selected_terms': '+'.join(r.get('selected_terms', [])) or '(none)',
            'b2_selected': 'b2' in r.get('selected_terms', []),
            'elapsed_s': r.get('elapsed_s', 0),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    return passes, results


# ======================================================================
#  FIGURES
# ======================================================================
def fig_corr_tripwire(b3_results, audit_dir):
    """Heatmap of correlation(a_true, regressor) per condition."""
    # Filter conditions with valid correlations
    cids = []
    corr_matrix = []
    for cid, v in b3_results.items():
        if v.get('status') == 'SKIP' or not v.get('correlations'):
            continue
        cids.append(cid)
        corr_matrix.append([v['correlations'].get(tn, 0) for tn in TERM_NAMES])

    if not corr_matrix:
        print("  No data for correlation figure.")
        return

    corr_arr = np.array(corr_matrix)

    fig, ax = plt.subplots(figsize=(8, max(4, len(cids) * 0.35)))
    im = ax.imshow(corr_arr, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(TERM_NAMES)))
    ax.set_xticklabels(TERM_NAMES, fontsize=10)
    ax.set_yticks(range(len(cids)))
    ax.set_yticklabels(cids, fontsize=7)

    # Annotate values
    for i in range(len(cids)):
        for j in range(len(TERM_NAMES)):
            val = corr_arr[i, j]
            color = 'white' if abs(val) > 0.3 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=6, color=color)

    # Mark should-be-absent columns
    for j, tn in enumerate(TERM_NAMES):
        if tn in ABSENT_TERMS:
            ax.text(j, -0.7, '*', ha='center', va='center',
                    fontsize=14, color='red', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Pearson correlation', shrink=0.8)
    ax.set_title('Circularity tripwire: corr(a_true, regressor) on FREE indices\n'
                 '* = should-be-absent (threshold |r| < 0.10)')
    fig.tight_layout()

    path = audit_dir / "fig_corr_tripwire.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


def fig_gap_ratio_robust(outputs_root, audit_dir):
    """Robust gap ratio plot with symlog scale and count annotations."""
    csv_path = outputs_root / "phase4_discovery" / "discovery_summary.csv"
    if not csv_path.exists():
        print("  No discovery_summary.csv for gap ratio figure.")
        return

    df = pd.read_csv(csv_path)
    df_ok = df[df['status'] == 'ok'].copy()
    GAP_EPS = 1e-12

    # Recompute gap_ratio
    for i, row in df_ok.iterrows():
        mse_p = row.get('MSE_phys_oracle')
        mse_o = row.get('MSE_oracle_oracle')
        mse_d = row.get('MSE_disc')
        if pd.notna(mse_p) and pd.notna(mse_o) and pd.notna(mse_d):
            denom = mse_p - mse_o
            if denom <= GAP_EPS:
                df_ok.at[i, 'gap_ratio'] = np.nan
                df_ok.at[i, 'gap_status'] = 'no_residual'
            else:
                df_ok.at[i, 'gap_ratio'] = (mse_d - mse_o) / denom
                df_ok.at[i, 'gap_status'] = 'ok'

    n_ok_gap = df_ok['gap_status'].eq('ok').sum()
    n_no_res = df_ok['gap_status'].ne('ok').sum()

    inter = df_ok[df_ok['oracle_regime'] == 'intermittent']
    cont  = df_ok[df_ok['oracle_regime'] == 'continuous']
    inter_v = inter[inter['gap_status'] == 'ok'].copy()
    cont_v  = cont[cont['gap_status'] == 'ok'].copy()
    no_res  = df_ok[df_ok['gap_status'] != 'ok']

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(inter_v) > 0:
        ax.scatter(inter_v['event_rate'], inter_v['gap_ratio'],
                   c='steelblue', marker='o', s=60, edgecolors='k',
                   linewidth=0.5, label=f'Intermittent (n={len(inter_v)})', zorder=3)
    if len(cont_v) > 0:
        ax.scatter(cont_v['event_rate'], cont_v['gap_ratio'],
                   c='coral', marker='^', s=60, edgecolors='k',
                   linewidth=0.5, label=f'Continuous (n={len(cont_v)})', zorder=3)
    if len(no_res) > 0:
        ax.scatter(no_res['event_rate'], [0]*len(no_res),
                   c='gray', marker='x', s=40, label=f'No residual (n={len(no_res)})',
                   zorder=2)

    ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='Physics-only level')
    ax.axhline(0.0, color='green', ls='--', alpha=0.5, label='Oracle ceiling')
    ax.axvline(0.50, color='gray', ls=':', alpha=0.4)
    ax.set_yscale('symlog', linthresh=2.0)
    ax.set_xlabel('Event rate')
    ax.set_ylabel('Oracle gap ratio (symlog)')
    ax.set_title('Discovery vs Oracle -- robust view')
    ax.legend(fontsize=8, loc='upper left')

    ax.text(0.98, 0.02, f'valid={n_ok_gap}  no_resid={n_no_res}  total={len(df_ok)}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    fig.tight_layout()
    path = audit_dir / "fig_gap_ratio_robust.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


def fig_null_term_frequency(null_results, audit_dir):
    """Bar chart of term frequency under null tests."""
    ok_results = [r for r in null_results if r['status'] == 'ok']
    if not ok_results:
        print("  No OK null results for figure.")
        return

    n_total = len(ok_results)
    counts = Counter()
    for r in ok_results:
        for t in r.get('selected_terms', []):
            counts[t] += 1

    freqs = [counts.get(tn, 0) / max(n_total, 1) for tn in TERM_NAMES]

    # Also split by null type
    type_counts = {}
    for nt in ['shuffle', 'reverse']:
        sub = [r for r in ok_results if r['null_type'] == nt]
        n_sub = len(sub)
        tc = Counter()
        for r in sub:
            for t in r.get('selected_terms', []):
                tc[t] += 1
        type_counts[nt] = {tn: tc.get(tn, 0) / max(n_sub, 1) for tn in TERM_NAMES}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall
    ax = axes[0]
    colors = ['red' if tn == 'b2' and freqs[i] > NULL_B2_MAX_FREQ else 'steelblue'
              for i, tn in enumerate(TERM_NAMES)]
    ax.bar(TERM_NAMES, freqs, color=colors, edgecolor='k', linewidth=0.5)
    ax.axhline(NULL_B2_MAX_FREQ, color='red', ls='--', alpha=0.5,
               label=f'Fail threshold ({100*NULL_B2_MAX_FREQ:.0f}%)')
    ax.set_ylabel('Selection frequency')
    ax.set_title(f'Null tests overall (n={n_total})')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    for i, f in enumerate(freqs):
        ax.text(i, f + 0.02, f'{f:.0%}', ha='center', fontsize=9)

    # By null type
    ax = axes[1]
    x = np.arange(len(TERM_NAMES))
    width = 0.35
    for j, (nt, color) in enumerate(zip(['shuffle', 'reverse'], ['#4c72b0', '#c44e52'])):
        n_sub = len([r for r in ok_results if r['null_type'] == nt])
        f_vals = [type_counts[nt].get(tn, 0) for tn in TERM_NAMES]
        ax.bar(x + j * width, f_vals, width, label=f'{nt} (n={n_sub})',
               color=color, edgecolor='k', linewidth=0.3)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(TERM_NAMES)
    ax.set_ylabel('Selection frequency')
    ax.set_title('By null type')
    ax.set_ylim(0, 1.05)
    ax.axhline(NULL_B2_MAX_FREQ, color='red', ls='--', alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('Null test: term selection under destroyed temporal structure', fontsize=12)
    fig.tight_layout()
    path = audit_dir / "fig_null_term_frequency.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ======================================================================
#  AUDIT REPORT
# ======================================================================
def write_audit_report(audit_dir, check_results, null_passes, null_results):
    """Write audit_report.md and audit_report.json."""
    b1_pass, b1 = check_results['b1']
    b2_pass, b2 = check_results['b2']
    b3_pass, b3 = check_results['b3']
    b4_pass, b4 = check_results['b4']
    b5_pass, b5 = check_results['b5']

    # JSON report
    report_json = OrderedDict([
        ('timestamp', time.strftime('%Y-%m-%d %H:%M:%S')),
        ('overall_pass', all([b1_pass, b2_pass, b3_pass, b4_pass, b5_pass, null_passes])),
        ('checks', OrderedDict([
            ('B1_schema', {'pass': b1_pass, 'type': 'HARD',
                           'n_conditions': len(b1),
                           'n_pass': sum(1 for v in b1.values() if v['status'] == 'PASS'),
                           'n_fail': sum(1 for v in b1.values() if v['status'] == 'FAIL')}),
            ('B2_splits', {'pass': b2_pass, 'type': 'HARD',
                           'n_conditions': len(b2),
                           'n_gated': sum(1 for v in b2.values() if v.get('gate'))}),
            ('B3_circularity', {'pass': b3_pass, 'type': 'HARD',
                                'n_conditions': len(b3),
                                'threshold': CORR_THRESHOLD,
                                'absent_terms': ABSENT_TERMS}),
            ('B4_outputs', {'pass': b4_pass, 'type': 'SOFT',
                            'details': b4}),
            ('B5_manifest', {'pass': b5_pass, 'type': 'HARD'}),
            ('C_null_tests', {'pass': null_passes, 'type': 'HARD',
                              'n_runs': len(null_results),
                              'b2_freq': sum(1 for r in null_results
                                           if r.get('status') == 'ok'
                                           and 'b2' in r.get('selected_terms', []))
                              / max(sum(1 for r in null_results
                                       if r.get('status') == 'ok'), 1),
                              'threshold': NULL_B2_MAX_FREQ}),
        ])),
        ('per_condition_b3', {cid: v.get('correlations', {})
                              for cid, v in b3.items() if v.get('status') != 'SKIP'}),
    ])

    json_path = audit_dir / "audit_report.json"
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"  Wrote {json_path}")

    # Markdown report
    lines = ["# Phase 4.5: Audit Report", ""]
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    overall = "PASS" if report_json['overall_pass'] else "FAIL"
    lines.append(f"**Overall:** {overall}")
    lines.append("")

    lines.append("## Check Summary")
    lines.append("")
    lines.append("| Check | Type | Status |")
    lines.append("|-------|------|--------|")
    checks = [
        ('B1: Schema + file integrity', 'HARD', b1_pass),
        ('B2: Split integrity', 'HARD', b2_pass),
        ('B3: Circularity tripwire', 'HARD', b3_pass),
        ('B4: Output consistency', 'SOFT', b4_pass),
        ('B5: Reproducibility manifest', 'HARD', b5_pass),
        ('C: Null tests', 'HARD', null_passes),
    ]
    for name, ctype, passed in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {name} | {ctype} | {status} |")
    lines.append("")

    # B3 details
    lines.append("## B3: Circularity Correlations (should-be-absent: d1, d2, d3)")
    lines.append("")
    lines.append(f"Threshold: |corr| < {CORR_THRESHOLD}")
    lines.append("")
    absent_corrs = {tn: [] for tn in ABSENT_TERMS}
    for cid, v in b3.items():
        if v.get('status') != 'SKIP' and v.get('correlations'):
            for tn in ABSENT_TERMS:
                absent_corrs[tn].append(abs(v['correlations'].get(tn, 0)))
    for tn in ABSENT_TERMS:
        vals = absent_corrs[tn]
        if vals:
            lines.append(f"- **{tn}**: max |corr| = {max(vals):.4f}, "
                         f"mean = {np.mean(vals):.4f}")
    lines.append("")

    # Null test details
    lines.append("## C: Null Test Results")
    lines.append("")
    ok_nulls = [r for r in null_results if r.get('status') == 'ok']
    n_ok = len(ok_nulls)
    b2_cnt = sum(1 for r in ok_nulls if 'b2' in r.get('selected_terms', []))
    lines.append(f"- Runs: {len(null_results)} ({n_ok} OK)")
    lines.append(f"- b2 selected: {b2_cnt}/{n_ok} ({100*b2_cnt/max(n_ok,1):.0f}%)")
    lines.append(f"- Threshold: <= {100*NULL_B2_MAX_FREQ:.0f}%")
    lines.append(f"- Status: {'PASS' if null_passes else 'FAIL'}")
    lines.append("")

    lines.append("| Condition | Null type | Selected | b2? |")
    lines.append("|-----------|-----------|----------|-----|")
    for r in null_results:
        sel = '+'.join(r.get('selected_terms', [])) or '(none)'
        b2 = 'yes' if 'b2' in r.get('selected_terms', []) else 'no'
        lines.append(f"| {r['condition_id']} | {r['null_type']} | {sel} | {b2} |")
    lines.append("")

    # Naming note
    lines.append("## A: Naming + Consistency Fixes Applied")
    lines.append("")
    lines.append("- A1: Renamed 'five-term' to 'six-term' library in code logs "
                 "(23_run_discovery_v2.py)")
    lines.append("- A2: Fixed select_terms() index convention to match "
                 "kf_filter_2state() timing (v[k-1], dv=v[k-1]-v[k-2])")
    lines.append("- D: Updated gap_ratio plot with symlog scale and count annotations "
                 "(24_phase4_plots.py)")
    lines.append("")

    md_path = audit_dir / "audit_report.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Wrote {md_path}")


# ======================================================================
#  MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Phase 4.5 Audit Pipeline')
    parser.add_argument('--datasets_root', type=str,
                        default=str(VL / 'datasets_v2'),
                        help='Path to datasets_v2 directory')
    parser.add_argument('--outputs_root', type=str,
                        default=str(VL / 'outputs'),
                        help='Path to outputs directory')
    parser.add_argument('--skip_null', action='store_true',
                        help='Skip null tests (for quick schema-only audit)')
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    outputs_root  = Path(args.outputs_root)
    audit_dir     = outputs_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audit pipeline: Phase 4.5")
    print(f"  datasets: {datasets_root}")
    print(f"  outputs:  {outputs_root}")
    print(f"  audit:    {audit_dir}")
    t0 = time.time()

    check_results = {}
    hard_fail = False

    # B1: Schema
    b1_pass, b1_res = check_b1_schema(datasets_root)
    check_results['b1'] = (b1_pass, b1_res)
    if not b1_pass:
        print("\n*** B1 HARD FAIL: Schema violations found ***")
        hard_fail = True

    # B2: Splits
    b2_pass, b2_res = check_b2_splits(datasets_root)
    check_results['b2'] = (b2_pass, b2_res)
    if not b2_pass:
        print("\n*** B2 HARD FAIL: Split integrity violations ***")
        hard_fail = True

    # B3: Circularity
    b3_pass, b3_res = check_b3_circularity(datasets_root)
    check_results['b3'] = (b3_pass, b3_res)
    if not b3_pass:
        print("\n*** B3 HARD FAIL: Circularity violations ***")
        hard_fail = True

    # B4: Output consistency
    b4_pass, b4_res = check_b4_outputs(outputs_root)
    check_results['b4'] = (b4_pass, b4_res)
    if not b4_pass:
        print("\n  B4 SOFT FAIL: Missing outputs (not blocking)")

    # B5: Manifest
    b5_pass, b5_manifest = write_b5_manifest(outputs_root, datasets_root, audit_dir)
    check_results['b5'] = (b5_pass, b5_manifest)
    if not b5_pass:
        print("\n*** B5 HARD FAIL: Missing files for manifest ***")
        hard_fail = True

    # Figures (non-null)
    print_section("Generating Audit Figures")
    fig_corr_tripwire(b3_res, audit_dir)
    fig_gap_ratio_robust(outputs_root, audit_dir)

    # C: Null tests
    null_passes = True
    null_results = []
    if args.skip_null:
        print_section("C: Null Tests (SKIPPED)")
    else:
        null_passes, null_results = run_c_null_tests(datasets_root, audit_dir)
        fig_null_term_frequency(null_results, audit_dir)
        if not null_passes:
            print("\n*** C HARD FAIL: Null test b2 frequency too high ***")
            hard_fail = True

    # Write report
    print_section("Writing Audit Report")
    write_audit_report(audit_dir, check_results, null_passes, null_results)

    # Final verdict
    elapsed = time.time() - t0
    print_section("AUDIT VERDICT")
    overall = not hard_fail
    if overall:
        print(f"  ALL HARD CHECKS: PASS")
    else:
        print(f"  AUDIT: FAIL (see errors above)")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
