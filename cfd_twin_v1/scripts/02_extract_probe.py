"""
02_extract_probe.py - Parse OpenFOAM probe output to CSV
=========================================================
Reads postProcessing/probes/0/U and extracts Ux at the probe location.
Writes outputs/probe_u_raw.csv with columns: time, Ux, Uy, Uz

OpenFOAM probe format:
    # Probe 0 (1.5 0.04 0.005)
    # Time
    0.005   (0.170123 1.2345e-05 0)
    0.010   (0.170456 2.3456e-05 0)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import re
import numpy as np
import pandas as pd
from pathlib import Path

CFD_ROOT = Path(__file__).resolve().parent.parent


def parse_probe_file(probe_path):
    """Parse OpenFOAM probe U file."""
    times = []
    ux_vals = []
    uy_vals = []
    uz_vals = []

    # Pattern: time  (Ux Uy Uz)
    pattern = re.compile(
        r'^\s*([\d.eE+-]+)\s+\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)')

    with open(probe_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = pattern.match(line)
            if m:
                times.append(float(m.group(1)))
                ux_vals.append(float(m.group(2)))
                uy_vals.append(float(m.group(3)))
                uz_vals.append(float(m.group(4)))

    return np.array(times), np.array(ux_vals), np.array(uy_vals), np.array(uz_vals)


def main():
    print("=" * 60)
    print("EXTRACT PROBE DATA FROM OPENFOAM OUTPUT")
    print("=" * 60)

    # Find probe file - check multiple possible locations
    case_dir = CFD_ROOT / "case"
    probe_candidates = [
        case_dir / "postProcessing" / "probes" / "0" / "U",
    ]

    # Also check for time directories (OF sometimes creates new dirs)
    pp_dir = case_dir / "postProcessing" / "probes"
    if pp_dir.exists():
        for d in sorted(pp_dir.iterdir()):
            if d.is_dir():
                u_file = d / "U"
                if u_file.exists() and u_file not in probe_candidates:
                    probe_candidates.append(u_file)

    probe_path = None
    for p in probe_candidates:
        if p.exists():
            probe_path = p
            break

    if probe_path is None:
        print("ERROR: No probe file found!")
        print(f"  Searched: {probe_candidates}")
        sys.exit(1)

    print(f"Reading: {probe_path}")
    t, ux, uy, uz = parse_probe_file(probe_path)
    print(f"  Parsed {len(t)} data points")
    print(f"  t = [{t[0]:.4f}, {t[-1]:.4f}]s")
    print(f"  dt_median = {np.median(np.diff(t)):.6f}s")
    print(f"  Ux: mean={np.mean(ux):.4f}, std={np.std(ux):.4f}")

    # Write CSV
    out_path = CFD_ROOT / "outputs" / "probe_u_raw.csv"
    df = pd.DataFrame({'time': t, 'Ux': ux, 'Uy': uy, 'Uz': uz})
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path} (N={len(df)})")

    print("\n" + "=" * 60)
    print("PROBE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
