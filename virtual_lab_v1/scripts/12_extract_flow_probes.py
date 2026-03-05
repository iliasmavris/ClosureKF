"""
12_extract_flow_probes.py - Parse multi-probe OpenFOAM output to CSV
=====================================================================
Extends cfd_twin_v1/scripts/02_extract_probe.py for multiple probe locations.

OpenFOAM multi-probe format:
    # Probe 0 (1.5 0.005 0.005)
    # Probe 1 (1.5 0.010 0.005)
    # Probe 2 (1.5 0.020 0.005)
    # Probe 3 (1.5 0.040 0.005)
    #
    # Time
    0.005   (0.170 1e-5 0) (0.180 2e-5 0) (0.190 3e-5 0) (0.200 4e-5 0)

Output: outputs/flow_probes/u_probes_raw.csv
  columns: time, u_y005, u_y010, u_y020, u_y040

Also works with Phase 1 single-probe format for verification.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Map probe y-coordinate (mm) to column name
PROBE_HEIGHT_NAMES = {
    5: "u_y005",
    10: "u_y010",
    20: "u_y020",
    40: "u_y040",
}


def find_probe_file(base_dir):
    """Find probe U file, checking multiple possible locations."""
    pp_dir = base_dir / "postProcessing" / "probes"
    candidates = []
    if pp_dir.exists():
        for d in sorted(pp_dir.iterdir()):
            if d.is_dir():
                u_file = d / "U"
                if u_file.exists():
                    candidates.append(u_file)

    if not candidates:
        # Fallback: check directly
        u_direct = pp_dir / "0" / "U"
        if u_direct.exists():
            candidates.append(u_direct)

    return candidates[0] if candidates else None


def parse_probe_header(probe_path):
    """Parse probe locations from header comments."""
    probes = []
    pattern = re.compile(r'#\s+Probe\s+(\d+)\s+\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)')
    with open(probe_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                if line and not line.startswith('#'):
                    break
                continue
            m = pattern.match(line)
            if m:
                idx = int(m.group(1))
                x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
                probes.append({'index': idx, 'x': x, 'y': y, 'z': z})
    return probes


def parse_multi_probe_file(probe_path, n_probes):
    """Parse multi-probe U file. Each data line has time + n_probes vector tuples."""
    times = []
    ux_data = [[] for _ in range(n_probes)]

    # Pattern for extracting time at start of line
    time_pat = re.compile(r'^\s*([\d.eE+-]+)\s+\(')
    # Pattern for extracting all (Ux Uy Uz) tuples
    vec_pat = re.compile(r'\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)')

    with open(probe_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            tm = time_pat.match(line)
            if not tm:
                continue

            t = float(tm.group(1))
            vectors = vec_pat.findall(line)

            if len(vectors) != n_probes:
                continue

            times.append(t)
            for i, (ux, uy, uz) in enumerate(vectors):
                ux_data[i].append(float(ux))

    return np.array(times), [np.array(d) for d in ux_data]


def make_column_names(probes):
    """Map probe locations to column names based on y-coordinate."""
    names = []
    for p in probes:
        y_mm = round(p['y'] * 1000)
        name = PROBE_HEIGHT_NAMES.get(y_mm, f"u_y{y_mm:03d}")
        names.append(name)
    return names


def main():
    print("=" * 60)
    print("EXTRACT MULTI-PROBE DATA FROM OPENFOAM OUTPUT")
    print("=" * 60)

    # Try virtual_lab_v1 postProcessing first, then cfd_twin_v1 case
    vl_flow = ROOT / "flow_inputs"
    cfd_case = ROOT.parent / "cfd_twin_v1" / "case"

    probe_path = find_probe_file(vl_flow)
    source = "virtual_lab_v1/flow_inputs"
    if probe_path is None:
        probe_path = find_probe_file(cfd_case)
        source = "cfd_twin_v1/case"

    if probe_path is None:
        print("ERROR: No probe file found!")
        print(f"  Searched: {vl_flow}/postProcessing/probes/*/U")
        print(f"  Searched: {cfd_case}/postProcessing/probes/*/U")
        sys.exit(1)

    print(f"Source: {source}")
    print(f"Reading: {probe_path}")

    # Parse header to find probe locations
    probes = parse_probe_header(probe_path)
    n_probes = max(len(probes), 1)
    print(f"  Found {n_probes} probe(s)")
    for p in probes:
        print(f"    Probe {p['index']}: ({p['x']}, {p['y']}, {p['z']})")

    # Parse data
    times, ux_data = parse_multi_probe_file(probe_path, n_probes)
    print(f"  Parsed {len(times)} timesteps")
    print(f"  t = [{times[0]:.4f}, {times[-1]:.4f}]s")
    print(f"  dt_median = {np.median(np.diff(times)):.6f}s")

    # Build column names
    if probes:
        col_names = make_column_names(probes)
    else:
        col_names = ["u_y040"]  # Phase 1 default (single probe at y=0.04)

    # Build dataframe
    data = {'time': times}
    for name, ux in zip(col_names, ux_data):
        data[name] = ux
        print(f"  {name}: mean={np.mean(ux):.4f}, std={np.std(ux):.4f}")

    df = pd.DataFrame(data)

    # Write output
    out_dir = ROOT / "outputs" / "flow_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "u_probes_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path} (N={len(df)}, cols={list(df.columns)})")

    # Phase 1 verification: if single probe at y=0.04, compare to cfd_twin_v1 output
    if n_probes == 1 and probes and abs(probes[0]['y'] - 0.04) < 1e-6:
        ref_path = ROOT.parent / "cfd_twin_v1" / "outputs" / "probe_u_raw.csv"
        if ref_path.exists():
            df_ref = pd.read_csv(ref_path)
            # Compare Ux values
            min_len = min(len(df), len(df_ref))
            diff = np.abs(ux_data[0][:min_len] - df_ref['Ux'].values[:min_len])
            max_diff = np.max(diff)
            print(f"\n  Phase 1 verification: max|Ux - ref| = {max_diff:.2e}")
            if max_diff < 1e-10:
                print("  PASS: matches Phase 1 probe output")
            else:
                print(f"  WARNING: diff = {max_diff:.2e} (expected < 1e-10)")

    print("\n" + "=" * 60)
    print("PROBE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
