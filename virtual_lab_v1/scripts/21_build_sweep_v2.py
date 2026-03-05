"""
21_build_sweep_v2.py - Build & run Phase 3 sweep (24+ conditions)
=================================================================
1. Read sweep_grid_v2.yaml + forcing_variants.yaml
2. Expand into concrete condition list with jitter draws
3. For each condition: load forcing variant, run truth model, downsample, oracle eval
4. Auto-tune mu_s to hit ER target [0.05, 0.40]

Stores results under datasets_v2/condition_XXX/ (does NOT overwrite Phase 2 datasets/)

Outputs:
  datasets_v2/condition_000..N/  (truth_states_raw.csv, x_10hz.csv, meta.json)
  outputs/sweep_v2/manifest_v2.json
  outputs/sweep_v2/sweep_summary_v2.csv
  outputs/sweep_v2/oracle_report_v2.json
  outputs/sweep_v2/oracle_summary_v2.csv

Usage:
  python 21_build_sweep_v2.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import copy
import json
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth_ball_sim_lib import simulate_sphere, compute_event_rate, compute_derived_params, compute_pin_statistics

ROOT = Path(__file__).resolve().parent.parent

# Oracle eval: import from canonical 18_oracle_eval.py
import importlib.util as _ilu
_oracle_spec = _ilu.spec_from_file_location(
    "oracle_eval", Path(__file__).resolve().parent / "18_oracle_eval.py")
_oracle_mod = _ilu.module_from_spec(_oracle_spec)
_oracle_spec.loader.exec_module(_oracle_mod)
oracle_eval_condition = _oracle_mod.evaluate_condition
TERM_NAMES = _oracle_mod.TERM_NAMES
STATUS_OK = _oracle_mod.STATUS_OK
ALPHA_GRID = _oracle_mod.ALPHA_GRID
TRAIN_FRAC = _oracle_mod.TRAIN_FRAC
VAL_FRAC = _oracle_mod.VAL_FRAC
UC_STEP = _oracle_mod.UC_STEP
UC_MAX = _oracle_mod.UC_MAX
MIN_FREE_SECONDS = _oracle_mod.MIN_FREE_SECONDS
ER_INTERMITTENT_MAX = _oracle_mod.ER_INTERMITTENT_MAX


# ---- Dataset creation (adapted from 15_make_dataset.py) ----

def downsample_to_10hz(t_raw, signal_raw, cutoff_hz=4.0):
    """Butter4Hz + filtfilt + 10Hz interp."""
    fs_raw = 1.0 / np.median(np.diff(t_raw))
    sos = butter(4, cutoff_hz, btype='low', fs=fs_raw, output='sos')
    sig_filt = sosfiltfilt(sos, signal_raw)
    t_10hz = np.arange(t_raw[0], t_raw[-1], 0.1)
    sig_10hz = np.interp(t_10hz, t_raw, sig_filt)
    return t_10hz, sig_10hz


def make_dataset_v2(cfg, condition_id, seed, t_flow, u_flow, out_base):
    """Create one condition dataset using pre-loaded forcing variant."""
    cond_dir = out_base / condition_id
    cond_dir.mkdir(parents=True, exist_ok=True)

    # Run sphere simulation
    result = simulate_sphere(cfg, t_flow, u_flow, seed=seed)
    er = compute_event_rate(result['contact'])

    # Discard spinup
    spinup = cfg['integration']['spinup_discard']
    mask = result['t'] >= spinup
    t_sim = result['t'][mask]
    x_sim = result['x'][mask]
    vp_sim = result['v_p'][mask]

    # Write truth_states_raw.csv
    truth_dict = {
        'time': result['t'][mask],
        'x': result['x'][mask],
        'v_p': result['v_p'][mask],
        'at_pin': result['at_pin'][mask],
        'eta': result['eta'][mask],
        'u_b': result['u_b'][mask],
        'du_b': result['du_b'][mask],
    }
    if 'a_force' in result:
        truth_dict['a_force'] = result['a_force'][mask]
    df_truth = pd.DataFrame(truth_dict)
    df_truth.to_csv(cond_dir / "truth_states_raw.csv", index=False)

    # Downsample to 10Hz
    t_10hz, x_10hz = downsample_to_10hz(t_sim, x_sim)
    _, ub_10hz = downsample_to_10hz(t_sim, result['u_b'][mask])

    # Write x_10hz.csv
    df_x10 = pd.DataFrame({
        'timestamp': t_10hz,
        'time_delta': np.full(len(t_10hz), 0.1),
        'velocity': ub_10hz,
        'displacement': x_10hz,
    })
    df_x10.to_csv(cond_dir / "x_10hz.csv", index=False)

    # Compute stats
    dx = np.diff(x_10hz)
    dt_sim = cfg['integration']['dt_sim']
    at_pin_post = result['at_pin'][mask]
    pin_stats = compute_pin_statistics(at_pin_post, dt_sim)
    wt = pin_stats['waiting_times']
    et = pin_stats['excursion_times']
    derived = compute_derived_params(cfg)

    meta = {
        'condition_id': condition_id,
        'seed': seed,
        'config': cfg,
        'derived_params': {k: float(v) for k, v in derived.items()},
        'spinup_discard_s': spinup,
        'event_rate': float(er),
        'n_events_sliding': int(np.sum(at_pin_post == 0)),
        'n_steps_total': int(np.sum(mask)),
        'pin_stats': {
            'n_waiting': len(wt),
            'n_excursions': len(et),
            'waiting_mean_s': float(np.mean(wt)) if len(wt) > 0 else None,
            'excursion_mean_s': float(np.mean(et)) if len(et) > 0 else None,
        },
        'stats_10hz': {
            'n_points': len(t_10hz),
            't_range': [float(t_10hz[0]), float(t_10hz[-1])],
            'u_mean': float(np.mean(ub_10hz)),
            'u_std': float(np.std(ub_10hz)),
            'x_mean': float(np.mean(x_10hz)),
            'x_std': float(np.std(x_10hz)),
            'dx_mean': float(np.mean(dx)),
            'dx_std': float(np.std(dx)),
        },
    }
    with open(cond_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


# ---- Auto-tune (adapted from 16_sweep_conditions.py) ----

def fullrun_autotune(cfg, t_flow, u_flow, seed, target_range, max_iter, step,
                     mu_s_clamp):
    """Full-run auto-tune: run complete simulation each iteration.

    More expensive but reliable — no preflight representativeness issues.
    Uses bisection-like narrowing once bounds are established.
    """
    cfg = copy.deepcopy(cfg)
    history = []
    spinup = cfg.get('integration', {}).get('spinup_discard', 30.0)

    # Track bounds for bisection
    lo_mu_s, hi_mu_s = None, None  # mu_s values that give ER below/above target

    for iteration in range(max_iter):
        result = simulate_sphere(cfg, t_flow, u_flow, seed=seed)
        dt_sim = cfg['integration']['dt_sim']
        n_spinup = int(spinup / dt_sim)
        contact_post = result['contact'][n_spinup:]
        er = compute_event_rate(contact_post) if len(contact_post) > 0 else 0.0
        mu_s_current = cfg['friction']['mu_s']

        history.append({
            'iteration': iteration,
            'mu_s': round(mu_s_current, 4),
            'event_rate': round(float(er), 4),
        })

        if target_range[0] <= er <= target_range[1]:
            break

        if er < target_range[0]:
            # Too pinned: need lower mu_s
            hi_mu_s = mu_s_current  # this mu_s gives too-low ER
            if lo_mu_s is not None:
                new_mu_s = (lo_mu_s + hi_mu_s) / 2
            else:
                new_mu_s = mu_s_current - step
        else:
            # Too mobile: need higher mu_s
            lo_mu_s = mu_s_current  # this mu_s gives too-high ER
            if hi_mu_s is not None:
                new_mu_s = (lo_mu_s + hi_mu_s) / 2
            else:
                new_mu_s = mu_s_current + step

        new_mu_s = max(mu_s_clamp[0], min(mu_s_clamp[1], new_mu_s))

        # Stop if mu_s didn't change (hit clamp)
        if abs(new_mu_s - mu_s_current) < 1e-6:
            break

        cfg['friction']['mu_s'] = new_mu_s
        # Keep mu_k < mu_s (maintain ~60% ratio)
        cfg['friction']['mu_k'] = min(cfg['friction']['mu_k'],
                                       0.6 * new_mu_s)

    return cfg, history


# Oracle eval is imported from 18_oracle_eval.py (see top of file)


# ---- Jitter ----

def apply_jitter(cfg, jitter_cfg, rng):
    """Apply lognormal jitters to physics params."""
    cfg = copy.deepcopy(cfg)
    if not jitter_cfg.get('enabled', False):
        return cfg

    # d_p jitter
    if 'd_p' in jitter_cfg:
        sig = jitter_cfg['d_p']['sigma_log']
        factor = float(rng.lognormal(0, sig))
        cfg['sphere']['d_p'] *= factor

    # k_spring jitter
    if 'k_spring' in jitter_cfg:
        sig = jitter_cfg['k_spring']['sigma_log']
        factor = float(rng.lognormal(0, sig))
        cfg['restoring']['k_spring'] *= factor

    # mu_k jitter
    if 'mu_k' in jitter_cfg:
        sig = jitter_cfg['mu_k']['sigma_log']
        factor = float(rng.lognormal(0, sig))
        cfg['friction']['mu_k'] *= factor
        # Clamp mu_k < mu_s always (gap scales with mu_s)
        gap = max(0.005, 0.3 * cfg['friction']['mu_s'])
        cfg['friction']['mu_k'] = min(cfg['friction']['mu_k'],
                                       cfg['friction']['mu_s'] - gap)
        cfg['friction']['mu_k'] = max(0.002, cfg['friction']['mu_k'])

    return cfg


# ---- Main ----

def main():
    print("=" * 70)
    print("PHASE 3 SWEEP V2 - Build + Run + Oracle Eval")
    print("=" * 70)
    t_start_all = time.time()

    # Load configs
    with open(ROOT / "configs" / "ball_params_default.yaml") as f:
        base_cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "sweep_grid_v2.yaml") as f:
        sweep_cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "forcing_variants.yaml") as f:
        fv_cfg = yaml.safe_load(f)

    # Lookup forcing variant definitions by ID
    fv_lookup = {v['id']: v for v in fv_cfg['variants']}

    forcing_ids = sweep_cfg['forcing_variants']
    seeds_per = sweep_cfg['seeds_per_variant']
    seed_base = sweep_cfg['seed_base']
    base_mu_s = sweep_cfg['base_mu_s']
    jitter_cfg = sweep_cfg['jitter']
    at_cfg = sweep_cfg['autotune']

    # Build condition list
    conditions = []
    cond_idx = 0
    for fv_id in forcing_ids:
        for s_offset in range(seeds_per):
            seed = seed_base + cond_idx
            cond_id = f"condition_{cond_idx:03d}"
            conditions.append({
                'id': cond_id,
                'forcing_variant': fv_id,
                'seed': seed,
                'seed_offset': s_offset,
            })
            cond_idx += 1

    n_cond = len(conditions)
    print(f"Conditions: {n_cond} ({len(forcing_ids)} variants x {seeds_per} seeds)")

    # Load baseline flow (for forcing variant lookup)
    raw_path = ROOT / "outputs" / "flow_probes" / "u_probes_raw.csv"
    df_raw = pd.read_csv(raw_path)
    t_flow_base = df_raw['time'].values

    # Prepare output dirs
    ds_root = ROOT / "datasets_v2"
    ds_root.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "outputs" / "sweep_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    forcing_dir = out_dir / "forcing"

    # Check forcing variants exist
    for fv_id in forcing_ids:
        fpath = forcing_dir / f"variant_{fv_id}_u_raw.csv"
        if not fpath.exists():
            print(f"ERROR: Forcing variant file missing: {fpath}")
            print("  Run 20_make_forcing_variants.py first.")
            sys.exit(1)

    # Pre-load all forcing variants
    forcing_cache = {}
    for fv_id in set(forcing_ids):
        fpath = forcing_dir / f"variant_{fv_id}_u_raw.csv"
        df_fv = pd.read_csv(fpath)
        forcing_cache[fv_id] = (df_fv['time'].values, df_fv['u_b_variant'].values)
    print(f"Loaded {len(forcing_cache)} unique forcing variants")

    # ---- Run conditions ----
    summary_rows = []
    oracle_results = []
    manifest_conditions = []

    for i, cond in enumerate(conditions):
        cond_id = cond['id']
        fv_id = cond['forcing_variant']
        seed = cond['seed']

        print(f"\n{'='*60}")
        print(f"CONDITION {i+1}/{n_cond}: {cond_id} (forcing={fv_id}, seed={seed})")

        # Get forcing
        t_flow, u_flow = forcing_cache[fv_id]

        # Build config with jitter
        cfg = copy.deepcopy(base_cfg)
        cfg['friction']['mu_s'] = base_mu_s
        jitter_rng = np.random.default_rng(seed + 10000)
        cfg = apply_jitter(cfg, jitter_cfg, jitter_rng)

        # Record jittered params before auto-tune
        d_p_jittered = cfg['sphere']['d_p']
        k_spring_jittered = cfg['restoring']['k_spring']
        mu_k_jittered = cfg['friction']['mu_k']
        mu_s_nominal = cfg['friction']['mu_s']

        print(f"  Jittered: d_p={d_p_jittered:.4f}, "
              f"k_spring={k_spring_jittered:.4f}, mu_k={mu_k_jittered:.3f}")

        # Auto-tune
        print(f"  Auto-tune (target ER [{at_cfg['target_er_low']}, {at_cfg['target_er_high']}], "
              f"mode=full_run, bisection)...")
        t0 = time.time()
        cfg_tuned, at_hist = fullrun_autotune(
            cfg, t_flow, u_flow, seed=seed,
            target_range=(at_cfg['target_er_low'], at_cfg['target_er_high']),
            max_iter=at_cfg['max_iter'],
            step=at_cfg['mu_s_step'],
            mu_s_clamp=at_cfg['mu_s_clamp'],
        )
        mu_s_final = cfg_tuned['friction']['mu_s']
        at_er = at_hist[-1]['event_rate'] if at_hist else 0.0
        print(f"  mu_s: {mu_s_nominal:.3f} -> {mu_s_final:.3f} "
              f"(final ER={at_er:.3f}, {len(at_hist)} iters)")

        # Write dataset from final tuned config (one more full run)
        meta = make_dataset_v2(cfg_tuned, cond_id, seed, t_flow, u_flow, ds_root)
        elapsed = time.time() - t0
        er = meta['event_rate']
        x_std = meta['stats_10hz']['x_std']
        print(f"  Dataset: ER={er:.3f}, x_std={x_std:.6f}, total={elapsed:.1f}s")

        # Check stop conditions
        er_in_target = at_cfg['target_er_low'] <= er <= at_cfg['target_er_high']
        er_nondegen = 0.01 <= er <= 0.95
        pass_std = x_std > 0

        # Oracle eval
        print(f"  Oracle eval...")
        oracle = oracle_eval_condition(ds_root / cond_id)
        oracle_results.append(oracle)
        if oracle['status'] == STATUS_OK:
            stab = '' if oracle.get('oracle_stable', True) else ' [UNSTABLE]'
            print(f"    [{oracle.get('a_source','?')}] gain={oracle['gain_oracle']:.3f}, "
                  f"R2_phys={oracle['R2_phys']:.3f}, "
                  f"R2_oracle={oracle['R2_oracle']:.3f}, "
                  f"corr_r_d2={oracle.get('corr_r_d2', 0):.4f}{stab}")
        else:
            print(f"    {oracle['status'].upper()}: {oracle.get('status_detail','')}")

        # Summary row
        row = {
            'condition_id': cond_id,
            'forcing_variant': fv_id,
            'seed': seed,
            'mu_s_nominal': mu_s_nominal,
            'mu_s_final': mu_s_final,
            'd_p': d_p_jittered,
            'k_spring': k_spring_jittered,
            'mu_k': mu_k_jittered,
            'event_rate': er,
            'x_std': x_std,
            'er_in_target': er_in_target,
            'nondegen': er_nondegen and pass_std,
            'autotune_iters': len(at_hist),
            'n_points_10hz': meta['stats_10hz']['n_points'],
        }
        row['oracle_status'] = oracle['status']
        row['regime'] = oracle.get('regime')
        row['best_alpha'] = oracle.get('best_alpha')
        row['gain_oracle'] = oracle.get('gain_oracle')
        row['R2_phys'] = oracle.get('R2_phys')
        row['R2_oracle'] = oracle.get('R2_oracle')
        row['corr_r_d2'] = oracle.get('corr_r_d2')
        row['corr_a_d2'] = oracle.get('corr_a_d2')
        row['oracle_stable'] = oracle.get('oracle_stable')
        row['gain_oracle_no_a1'] = oracle.get('gain_oracle_no_a1')
        row['R2_oracle_no_a1'] = oracle.get('R2_oracle_no_a1')
        row['oracle_stable_no_a1'] = oracle.get('oracle_stable_no_a1')
        row['best_alpha_no_a1'] = oracle.get('best_alpha_no_a1')
        summary_rows.append(row)

        # Manifest entry
        manifest_conditions.append({
            'condition_id': cond_id,
            'forcing_variant': fv_id,
            'seed': seed,
            'mu_s_nominal': mu_s_nominal,
            'mu_s_final': mu_s_final,
            'd_p': d_p_jittered,
            'k_spring': k_spring_jittered,
            'mu_k': mu_k_jittered,
            'event_rate': er,
            'x_std': x_std,
            'er_in_target': er_in_target,
            'autotune_history': at_hist,
        })

    total_time = time.time() - t_start_all

    # ---- Write outputs ----

    # 1. manifest_v2.json
    manifest = {
        'phase': 3,
        'n_conditions': n_cond,
        'n_oracle_total': len(oracle_results),
        'n_oracle_ok': sum(1 for r in oracle_results if r['status'] == STATUS_OK),
        'total_elapsed_s': total_time,
        'configs': {
            'base_mu_s': base_mu_s,
            'jitter': jitter_cfg,
            'autotune': at_cfg,
            'forcing_variants': forcing_ids,
            'seeds_per_variant': seeds_per,
            'seed_base': seed_base,
        },
        'conditions': manifest_conditions,
    }
    mani_path = out_dir / "manifest_v2.json"
    with open(mani_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {mani_path}")

    # 2. sweep_summary_v2.csv
    df_summary = pd.DataFrame(summary_rows)
    csv_path = out_dir / "sweep_summary_v2.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")

    # 3. oracle_report_v2.json
    # Stringify val_mses keys for JSON
    for r in oracle_results:
        if 'val_mses' in r and r['val_mses'] is not None:
            r['val_mses'] = {str(k): v for k, v in r['val_mses'].items()}
        if 'val_mses_no_a1' in r and r['val_mses_no_a1'] is not None:
            r['val_mses_no_a1'] = {str(k): v for k, v in r['val_mses_no_a1'].items()}
    oracle_report = {
        'n_conditions': len(oracle_results),
        'settings': {
            'alpha_grid': ALPHA_GRID,
            'train_frac': TRAIN_FRAC,
            'val_frac': VAL_FRAC,
            'uc_grid_step': UC_STEP,
            'uc_grid_max': UC_MAX,
            'min_free_seconds': MIN_FREE_SECONDS,
            'er_intermittent_max': ER_INTERMITTENT_MAX,
        },
        'conditions': oracle_results,
    }
    orep_path = out_dir / "oracle_report_v2.json"
    with open(orep_path, 'w') as f:
        json.dump(oracle_report, f, indent=2)
    print(f"Oracle report: {orep_path}")

    # 4. oracle_summary_v2.csv
    orows = []
    for r in oracle_results:
        orow = {
            'condition_id': r['condition_id'],
            'status': r['status'],
            'regime': r.get('regime'),
            'event_rate': r.get('event_rate'),
            'n_free': r.get('n_free'),
            'n_train_free': r.get('n_train_free'),
            'n_val_free': r.get('n_val_free'),
            'n_test_free': r.get('n_test_free'),
            'a_source': r.get('a_source'),
            'best_alpha': r.get('best_alpha'),
            'MSE_phys': r.get('MSE_phys'),
            'MSE_oracle': r.get('MSE_oracle'),
            'gain_oracle': r.get('gain_oracle'),
            'oracle_stable': r.get('oracle_stable'),
            'R2_phys': r.get('R2_phys'),
            'R2_oracle': r.get('R2_oracle'),
            'corr_r_d2': r.get('corr_r_d2'),
            'corr_a_d2': r.get('corr_a_d2'),
            # No-a1 library
            'best_alpha_no_a1': r.get('best_alpha_no_a1'),
            'MSE_oracle_no_a1': r.get('MSE_oracle_no_a1'),
            'gain_oracle_no_a1': r.get('gain_oracle_no_a1'),
            'R2_oracle_no_a1': r.get('R2_oracle_no_a1'),
            'oracle_stable_no_a1': r.get('oracle_stable_no_a1'),
        }
        if r['status'] == STATUS_OK and r.get('theta_lib'):
            for tn in TERM_NAMES:
                orow[tn] = r['theta_lib'][tn]
        orows.append(orow)
    df_oracle = pd.DataFrame(orows)
    ocsv_path = out_dir / "oracle_summary_v2.csv"
    df_oracle.to_csv(ocsv_path, index=False)
    print(f"Oracle CSV: {ocsv_path}")

    # ---- Final report ----
    n_nondegen = sum(1 for r in summary_rows if r['nondegen'])
    n_in_target = sum(1 for r in summary_rows if r['er_in_target'])
    ers = [r['event_rate'] for r in summary_rows]
    gains = [r['gain_oracle'] for r in summary_rows if r['gain_oracle'] is not None]
    corrs = [r['corr_r_d2'] for r in summary_rows if r.get('corr_r_d2') is not None]

    print(f"\n{'='*70}")
    print(f"PHASE 3 SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Conditions: {n_cond}")
    print(f"Non-degenerate: {n_nondegen}/{n_cond}")
    print(f"ER in target [{at_cfg['target_er_low']}, {at_cfg['target_er_high']}]: "
          f"{n_in_target}/{n_cond} ({100*n_in_target/n_cond:.0f}%)")
    print(f"ER: median={np.median(ers):.3f}, "
          f"IQR=[{np.percentile(ers, 25):.3f}, {np.percentile(ers, 75):.3f}]")
    if gains:
        print(f"Oracle gain: median={np.median(gains):.3f}, "
              f"IQR=[{np.percentile(gains, 25):.3f}, {np.percentile(gains, 75):.3f}]")
    if corrs:
        print(f"Non-circ corr: min={min(corrs):.4f}, median={np.median(corrs):.4f}, "
              f"max={max(corrs):.4f}")
    print(f"Oracle eval: {len(oracle_results)}/{n_cond} valid")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
