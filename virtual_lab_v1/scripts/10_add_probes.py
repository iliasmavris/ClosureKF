"""
10_add_probes.py - Write multi-height OpenFOAM probes overlay
==============================================================
Creates a modified probes file with 4 probe locations at x=1.5, z=0.005,
y=[0.005, 0.010, 0.020, 0.040] m.

Output: virtual_lab_v1/flow_inputs/case_overlay/system/probes
Does NOT modify cfd_twin_v1/case/ (preserves Phase 1).

Probe height verification (20 cells over H/2=0.10m, grading 2.0):
  Cell 1 center: y ~ 1.73 mm
  Cell 2 center: y ~ 5.25 mm  <-- y=0.005 is near this
  Cell 3 center: y ~ 8.9 mm   <-- y=0.010 is near this
All 4 heights are well-resolved.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write_probes_file(out_dir):
    """Write multi-height OF probes function object."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probes_path = out_dir / "probes"

    content = """\
probes
{
    type            probes;
    libs            (fieldFunctionObjects);
    writeControl    timeStep;
    writeInterval   1;

    fields          (U p);

    probeLocations
    (
        (1.5 0.005 0.005)
        (1.5 0.010 0.005)
        (1.5 0.020 0.005)
        (1.5 0.040 0.005)
    );
}
"""
    probes_path.write_text(content)
    return probes_path


def main():
    print("=" * 60)
    print("ADD MULTI-HEIGHT PROBES FOR VIRTUAL LAB")
    print("=" * 60)

    overlay_dir = ROOT / "flow_inputs" / "case_overlay" / "system"
    probes_path = write_probes_file(overlay_dir)

    print(f"\nWrote: {probes_path}")
    print("\nProbe locations:")
    print("  Probe 0: (1.5, 0.005, 0.005)  -- y = 5 mm")
    print("  Probe 1: (1.5, 0.010, 0.005)  -- y = 10 mm")
    print("  Probe 2: (1.5, 0.020, 0.005)  -- y = 20 mm")
    print("  Probe 3: (1.5, 0.040, 0.005)  -- y = 40 mm")
    print("\nOverlay this onto the OF case with 11_run_flow_case.sh")
    print("=" * 60)


if __name__ == '__main__':
    main()
