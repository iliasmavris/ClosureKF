"""
15_make_dataset.py - Orchestrate one condition end-to-end
=========================================================
1. Load flow data from outputs/flow_probes/u_probes_raw.csv
2. Select channel from config (default u_y010)
3. Run sphere simulation (14_truth_ball_sim.py)
4. Downsample x(t) and v_p(t) to 10Hz via Butter4Hz + filtfilt + interp
5. Copy flow probe data into condition directory
6. Write truth_states_raw.csv, x_10hz.csv, meta.json

Usage:
  python 15_make_dataset.py                          # default config
  python 15_make_dataset.py --condition condition_000 --seed 100
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import argparse
import numpy as np
import pandas as pd
import shutil
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# Allow import of sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth_ball_sim_lib import (simulate_sphere, compute_event_rate,
                                compute_derived_params, compute_pin_statistics)

ROOT = Path(__file__).resolve().parent.parent


def load_config(cfg_path=None):
    """Load YAML config."""
    import yaml
    if cfg_path is None:
        cfg_path = ROOT / "configs" / "ball_params_default.yaml"
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def downsample_to_10hz(t_raw, signal_raw, cutoff_hz=4.0):
    """Apply Butter4Hz + filtfilt + 10Hz interp."""
    fs_raw = 1.0 / np.median(np.diff(t_raw))
    sos = butter(4, cutoff_hz, btype='low', fs=fs_raw, output='sos')
    sig_filt = sosfiltfilt(sos, signal_raw)
    t_10hz = np.arange(t_raw[0], t_raw[-1], 0.1)
    sig_10hz = np.interp(t_10hz, t_raw, sig_filt)
    return t_10hz, sig_10hz


def make_dataset(cfg, condition_id, seed, out_base=None):
    """
    Create one condition dataset.

    Returns dict with summary info (event_rate, stats, etc.)
    """
    if out_base is None:
        out_base = ROOT / "datasets"
    cond_dir = out_base / condition_id
    cond_dir.mkdir(parents=True, exist_ok=True)

    # Load flow data
    raw_path = ROOT / "outputs" / "flow_probes" / "u_probes_raw.csv"
    hz10_path = ROOT / "outputs" / "flow_probes" / "u_probes_10hz.csv"

    df_raw = pd.read_csv(raw_path)
    t_flow = df_raw['time'].values

    # Select channel
    channel = cfg['flow']['probe_channel']
    if channel not in df_raw.columns:
        avail = [c for c in df_raw.columns if c != 'time']
        print(f"  WARNING: channel '{channel}' not found, available: {avail}")
        channel = avail[0]
        print(f"  Using fallback channel: {channel}")

    u_flow = df_raw[channel].values
    print(f"  Flow channel: {channel}, N={len(t_flow)}, "
          f"t=[{t_flow[0]:.1f}, {t_flow[-1]:.1f}]s")

    # Run sphere simulation
    print(f"  Running sphere sim (seed={seed})...")
    result = simulate_sphere(cfg, t_flow, u_flow, seed=seed)
    er = compute_event_rate(result['contact'])
    print(f"  Event rate: {er:.3f}")

    # Discard spinup
    spinup = cfg['integration']['spinup_discard']
    mask = result['t'] >= spinup
    t_sim = result['t'][mask]
    x_sim = result['x'][mask]
    vp_sim = result['v_p'][mask]

    # Write truth_states_raw.csv (full sim resolution, after spinup)
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

    # Downsample x(t) and u_b(t) to 10Hz
    t_10hz, x_10hz = downsample_to_10hz(t_sim, x_sim)
    _, ub_10hz = downsample_to_10hz(t_sim, result['u_b'][mask])

    # Write x_10hz.csv matching processed_data_10hz schema
    # timestamp, time_delta, velocity, displacement
    df_x10 = pd.DataFrame({
        'timestamp': t_10hz,
        'time_delta': np.full(len(t_10hz), 0.1),
        'velocity': ub_10hz,
        'displacement': x_10hz,
    })
    df_x10.to_csv(cond_dir / "x_10hz.csv", index=False)

    # Copy flow probe data into condition directory
    shutil.copy2(raw_path, cond_dir / "u_probes_raw.csv")
    if hz10_path.exists():
        shutil.copy2(hz10_path, cond_dir / "u_probes_10hz.csv")

    # Compute dx stats
    dx = np.diff(x_10hz)

    # Compute pin statistics (waiting / excursion distributions)
    dt_sim = cfg['integration']['dt_sim']
    at_pin_post = result['at_pin'][mask]
    pin_stats = compute_pin_statistics(at_pin_post, dt_sim)
    wt = pin_stats['waiting_times']
    et = pin_stats['excursion_times']

    # Compute derived params for meta
    derived = compute_derived_params(cfg)

    # Write meta.json
    meta = {
        'condition_id': condition_id,
        'seed': seed,
        'config': cfg,
        'derived_params': {k: float(v) for k, v in derived.items()},
        'flow_channel': channel,
        'spinup_discard_s': spinup,
        'event_rate': float(er),
        'n_events_sliding': int(np.sum(at_pin_post == 0)),
        'n_steps_total': int(np.sum(mask)),
        'pin_stats': {
            'n_waiting': len(wt),
            'n_excursions': len(et),
            'waiting_mean_s': float(np.mean(wt)) if len(wt) > 0 else None,
            'waiting_median_s': float(np.median(wt)) if len(wt) > 0 else None,
            'waiting_std_s': float(np.std(wt)) if len(wt) > 0 else None,
            'excursion_mean_s': float(np.mean(et)) if len(et) > 0 else None,
            'excursion_median_s': float(np.median(et)) if len(et) > 0 else None,
            'excursion_std_s': float(np.std(et)) if len(et) > 0 else None,
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

    print(f"  Output: {cond_dir}")
    print(f"  x_10hz: N={len(t_10hz)}, std={np.std(x_10hz):.6f}")
    print(f"  dx: mean={np.mean(dx):.2e}, std={np.std(dx):.4f}")

    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make one condition dataset')
    parser.add_argument('--condition', default='condition_000')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(f"MAKE DATASET: {args.condition}")
    print("=" * 60)

    cfg = load_config(args.config)
    meta = make_dataset(cfg, args.condition, args.seed)

    print("\n" + "=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)
