"""
16_sweep_conditions.py - Run 8 conditions from sweep_grid.yaml
===============================================================
For each condition:
  1. Preflight auto-tune (120s): adjust mu_s to get event_rate in [0.01, 0.80]
  2. Run full make_dataset() with tuned mu_s
  3. Check stop conditions

Writes datasets/sweep_summary.json with per-condition results.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import copy
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth_ball_sim_lib import simulate_sphere, compute_event_rate

ROOT = Path(__file__).resolve().parent.parent


def load_base_config():
    """Load default ball params."""
    cfg_path = ROOT / "configs" / "ball_params_default.yaml"
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(cfg_path)


def load_configs():
    """Load base config and sweep grid."""
    base_path = ROOT / "configs" / "ball_params_default.yaml"
    grid_path = ROOT / "configs" / "sweep_grid.yaml"

    with open(base_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    with open(grid_path, 'r') as f:
        grid = yaml.safe_load(f)

    return base_cfg, grid


def apply_overrides(cfg, overrides):
    """Apply dot-notation overrides to config dict."""
    cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        parts = key.split('.')
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = val
    return cfg


def preflight_autotune(cfg, t_flow, u_flow, seed, target_range=(0.01, 0.80),
                        max_iter=5, step=0.03, preflight_duration=120.0):
    """
    Run 120s preflight sim, adjust mu_s to get event_rate in target range.

    Returns: (tuned_cfg, autotune_history)
    """
    cfg = copy.deepcopy(cfg)
    mu_s_nominal = cfg['friction']['mu_s']
    history = []

    for iteration in range(max_iter):
        # Run short sim
        cfg_short = copy.deepcopy(cfg)
        cfg_short['integration']['t_end'] = min(preflight_duration, t_flow[-1])
        spinup = cfg.get('integration', {}).get('spinup_discard', 30.0)
        cfg_short['integration']['spinup_discard'] = 0.0  # sim runs full

        result = simulate_sphere(cfg_short, t_flow, u_flow, seed=seed)
        # Discard spinup before computing event_rate (matches full-run logic)
        dt_sim = cfg_short['integration']['dt_sim']
        n_spinup = int(spinup / dt_sim)
        contact_post = result['contact'][n_spinup:]
        er = compute_event_rate(contact_post) if len(contact_post) > 0 else 0.0
        mu_s_current = cfg['friction']['mu_s']

        entry = {
            'iteration': iteration,
            'mu_s': mu_s_current,
            'event_rate': float(er),
        }
        history.append(entry)

        print(f"    Preflight iter {iteration}: mu_s={mu_s_current:.3f}, "
              f"event_rate={er:.3f}")

        if target_range[0] <= er <= target_range[1]:
            print(f"    In target range [{target_range[0]}, {target_range[1]}]")
            break

        # Adjust mu_s
        if er < target_range[0]:
            # Too stuck: decrease mu_s to make sliding easier
            cfg['friction']['mu_s'] = mu_s_current - step
        else:
            # Always sliding: increase mu_s to make sliding harder
            cfg['friction']['mu_s'] = mu_s_current + step

        # Clamp to physical range
        cfg['friction']['mu_s'] = max(0.1, min(1.5, cfg['friction']['mu_s']))

    return cfg, history


def make_dataset_from_sweep(cfg, condition_id, seed):
    """Import and call make_dataset from 15_make_dataset.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_dataset_mod",
        Path(__file__).resolve().parent / "15_make_dataset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_dataset(cfg, condition_id, seed)


def main():
    print("=" * 60)
    print("SWEEP CONDITIONS - 8-condition virtual lab")
    print("=" * 60)

    base_cfg, grid = load_configs()
    conditions = grid['conditions']
    print(f"\nLoaded {len(conditions)} conditions from sweep_grid.yaml")

    # Load flow data once
    raw_path = ROOT / "outputs" / "flow_probes" / "u_probes_raw.csv"
    if not raw_path.exists():
        print(f"ERROR: Flow data not found at {raw_path}")
        print("  Run 12_extract_flow_probes.py first.")
        sys.exit(1)

    df_raw = pd.read_csv(raw_path)
    t_flow = df_raw['time'].values

    # Select channel for preflight
    channel = base_cfg['flow']['probe_channel']
    if channel not in df_raw.columns:
        avail = [c for c in df_raw.columns if c != 'time']
        channel = avail[0]
    u_flow = df_raw[channel].values
    print(f"Flow channel: {channel}, N={len(t_flow)}")

    # Run each condition
    summary = []
    t_start_all = time.time()

    for i, cond in enumerate(conditions):
        cond_id = cond['id']
        seed = cond['seed']
        overrides = cond['overrides']

        print(f"\n{'='*60}")
        print(f"CONDITION {i+1}/{len(conditions)}: {cond_id} (seed={seed})")
        print(f"  Overrides: {overrides}")

        cfg = apply_overrides(base_cfg, overrides)
        mu_s_nominal = cfg['friction']['mu_s']

        # Preflight auto-tune
        print(f"  Preflight auto-tune (120s sim)...")
        cfg_tuned, autotune_hist = preflight_autotune(
            cfg, t_flow, u_flow, seed=seed
        )
        mu_s_final = cfg_tuned['friction']['mu_s']

        # Run full dataset
        print(f"  Running full dataset (mu_s: {mu_s_nominal:.3f} -> {mu_s_final:.3f})...")
        t0 = time.time()
        meta = make_dataset_from_sweep(cfg_tuned, cond_id, seed)
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")

        # Check stop conditions
        er = meta['event_rate']
        x_std = meta['stats_10hz']['x_std']
        pass_er = 0.01 <= er <= 0.80
        pass_std = x_std > 0
        all_pass = pass_er and pass_std

        cond_summary = {
            'condition_id': cond_id,
            'seed': seed,
            'overrides': overrides,
            'mu_s_nominal': mu_s_nominal,
            'mu_s_final': mu_s_final,
            'event_rate': er,
            'x_std_10hz': x_std,
            'pass_event_rate': pass_er,
            'pass_x_std': pass_std,
            'all_pass': all_pass,
            'autotune_history': autotune_hist,
            'elapsed_s': elapsed,
        }
        summary.append(cond_summary)

        status = "PASS" if all_pass else "FAIL"
        print(f"  {status}: event_rate={er:.3f}, x_std={x_std:.6f}")

    # Write sweep summary
    total_time = time.time() - t_start_all
    sweep_result = {
        'n_conditions': len(conditions),
        'n_pass': sum(1 for s in summary if s['all_pass']),
        'total_elapsed_s': total_time,
        'conditions': summary,
    }

    out_path = ROOT / "datasets" / "sweep_summary.json"
    with open(out_path, 'w') as f:
        json.dump(sweep_result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: {sweep_result['n_pass']}/{len(conditions)} PASS")
    print(f"Total time: {total_time:.1f}s")
    print(f"Summary: {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
