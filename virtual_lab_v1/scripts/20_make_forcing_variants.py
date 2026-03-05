"""
20_make_forcing_variants.py - Generate forcing variant u_b(t) from baseline
==========================================================================
Reads baseline flow probe (dt_sim resolution) and applies transforms
defined in configs/forcing_variants.yaml.

Transforms:
  identity:  u'(t) = u(t)
  scale:     u'(t) = s * u(t)
  am_ou:     u'(t) = e(t) * u(t), e(t) = clip(1 + a*z(t), lo, hi), renorm to mean=1
  scale_am:  scale first, then AM-OU

Output per variant:
  outputs/sweep_v2/forcing/variant_<id>_u_raw.csv  (cols: time, u_b_variant)

Usage:
  python 20_make_forcing_variants.py [--channel u_y040] [--seed 42]
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def generate_ou(n, dt, tau, seed):
    """Generate zero-mean, unit-variance OU process."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-dt / tau)
    sigma_disc = np.sqrt(1.0 - phi**2)
    z = np.zeros(n)
    z[0] = rng.normal(0, 1)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + sigma_disc * rng.normal()
    # Renormalize to exact zero-mean, unit-std
    z = (z - z.mean()) / (z.std() + 1e-30)
    return z


def apply_variant(t, u_base, variant_cfg, global_cfg, seed):
    """Apply a forcing transform to baseline u_b(t).

    Returns: u_variant (same length as u_base)
    """
    vtype = variant_cfg['type']
    dt = np.median(np.diff(t))
    n = len(u_base)

    if vtype == 'identity':
        return u_base.copy()

    elif vtype == 'scale':
        s = variant_cfg['scale']
        return s * u_base

    elif vtype == 'am_ou':
        a = variant_cfg['am_amplitude']
        tau = variant_cfg['tau_env']
        lo = global_cfg.get('am_clip_min', 0.6)
        hi = global_cfg.get('am_clip_max', 1.6)
        z = generate_ou(n, dt, tau, seed)
        e = np.clip(1.0 + a * z, lo, hi)
        # Renormalize to mean 1 (preserve long-term mean flow)
        e = e / e.mean()
        return e * u_base

    elif vtype == 'scale_am':
        s = variant_cfg['scale']
        u_scaled = s * u_base
        a = variant_cfg['am_amplitude']
        tau = variant_cfg['tau_env']
        lo = global_cfg.get('am_clip_min', 0.6)
        hi = global_cfg.get('am_clip_max', 1.6)
        z = generate_ou(n, dt, tau, seed)
        e = np.clip(1.0 + a * z, lo, hi)
        e = e / e.mean()
        return e * u_scaled

    else:
        raise ValueError(f"Unknown variant type: {vtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default=None,
                        help='Flow probe channel (default: from ball_params_default.yaml)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base RNG seed for OU envelope generation')
    args = parser.parse_args()

    print("=" * 60)
    print("MAKE FORCING VARIANTS (Phase 3)")
    print("=" * 60)

    # Load configs
    with open(ROOT / "configs" / "forcing_variants.yaml") as f:
        fv_cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "ball_params_default.yaml") as f:
        base_cfg = yaml.safe_load(f)

    variants = fv_cfg['variants']

    # Load baseline flow
    raw_path = ROOT / "outputs" / "flow_probes" / "u_probes_raw.csv"
    if not raw_path.exists():
        print(f"ERROR: Flow data not found at {raw_path}")
        sys.exit(1)
    df_raw = pd.read_csv(raw_path)
    t = df_raw['time'].values

    channel = args.channel
    if channel is None:
        channel = base_cfg['flow']['probe_channel']
    if channel not in df_raw.columns:
        avail = [c for c in df_raw.columns if c != 'time']
        print(f"WARNING: channel '{channel}' not found, available: {avail}")
        channel = avail[0]
        print(f"Using fallback: {channel}")
    u_base = df_raw[channel].values
    print(f"Baseline: channel={channel}, N={len(t)}, "
          f"mean={u_base.mean():.4f}, std={u_base.std():.4f}")

    # Output directory
    out_dir = ROOT / "outputs" / "sweep_v2" / "forcing"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'baseline_channel': channel,
        'baseline_n': len(t),
        'baseline_mean': float(u_base.mean()),
        'baseline_std': float(u_base.std()),
        'base_seed': args.seed,
        'variants': [],
    }

    for i, vcfg in enumerate(variants):
        vid = vcfg['id']
        # Each variant gets a unique seed derived from base
        vseed = args.seed + i * 1000
        print(f"\n  [{i+1}/{len(variants)}] {vid} (type={vcfg['type']}, seed={vseed})")

        u_var = apply_variant(t, u_base, vcfg, fv_cfg, vseed)

        # Write CSV
        csv_path = out_dir / f"variant_{vid}_u_raw.csv"
        pd.DataFrame({'time': t, 'u_b_variant': u_var}).to_csv(csv_path, index=False)

        stats = {
            'id': vid,
            'type': vcfg['type'],
            'seed': vseed,
            'mean': float(u_var.mean()),
            'std': float(u_var.std()),
            'min': float(u_var.min()),
            'max': float(u_var.max()),
        }
        # Copy transform params
        for k in ['scale', 'am_amplitude', 'tau_env']:
            if k in vcfg:
                stats[k] = vcfg[k]

        manifest['variants'].append(stats)
        print(f"    mean={u_var.mean():.4f}, std={u_var.std():.4f}, "
              f"range=[{u_var.min():.4f}, {u_var.max():.4f}]")

    # Write manifest
    manifest_path = out_dir / "forcing_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {len(variants)} forcing variants written to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
