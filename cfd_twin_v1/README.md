# CFD Twin v1 - Phase 1: Reproduce u(t) at Probe Location

OpenFOAM URANS simulation to reproduce the laboratory water velocity signal
at the measurement location. 2D channel with time-varying inlet from lab data.

## Quick Start

### Step 1: Prepare inlet BC from lab data
```bash
python cfd_twin_v1/scripts/00_prepare_inflow.py --gain 1.0 --lag 8.8 --ramp 5.0
```

### Step 2: Run OpenFOAM in WSL2
```bash
wsl bash -c "cd '/mnt/c/Users/Workstation 2/Desktop/2026 research/full code old  state space claude code/cfd_twin_v1/scripts' && bash 01_run_case.sh"
```

For 10s test run:
```bash
wsl bash -c "cd '/mnt/c/Users/Workstation 2/Desktop/2026 research/full code old  state space claude code/cfd_twin_v1/scripts' && bash 01_run_case.sh 10"
```

### Step 3: Extract and process probe data
```bash
python cfd_twin_v1/scripts/02_extract_probe.py
python cfd_twin_v1/scripts/03_process_downsample.py
```

### Step 4: Validate
```bash
python cfd_twin_v1/scripts/04_validate_u.py
```

## Domain

- 2D straight channel: L=2.0m x H=0.20m (single cell in z=0.01m)
- 200 x 40 x 1 = 8,000 cells (graded near walls)
- Re ~ 34,000 (U=0.17 m/s, H=0.20m, nu=1e-6)
- Probe at (1.5, 0.04, 0.005)

## Solver

- pimpleFoam (transient incompressible URANS)
- k-omega SST with wall functions
- dt=0.005s, ~108K steps for 540s simulation
- Schemes: backward (time), linearUpwind (convection)

## Calibration Parameters

| Parameter | Default | Controls |
|-----------|---------|----------|
| gain | 1.0 | Mean + amplitude |
| lag_s | 8.8 | Temporal alignment (advective delay) |
| Ks | 0.001 m | Wall roughness / friction |
| y_probe | 0.04 m | Probe height in boundary layer |

## Stop Conditions

- Mean(u): within 10% of lab
- Std(u): within 15% of lab
- Correlation r >= 0.85
- ACF(1..10): same sign as lab
- PSD mismatch: reported (lower is better)

## File Structure

```
cfd_twin_v1/
  lab_data/u_lab_10hz.csv         # Lab reference (rebased t=0)
  case/                           # OpenFOAM case
    0/ U p k omega nut            # Boundary conditions
    constant/                     # Transport + turbulence
    system/                       # Mesh, solver, probes
  scripts/
    00_prepare_inflow.py          # Lab -> OF inlet table
    01_run_case.sh                # WSL launcher
    02_extract_probe.py           # Parse probe output
    03_process_downsample.py      # 4Hz LP + 10Hz resample
    04_validate_u.py              # Metrics + figures
  outputs/
    probe_u_raw.csv               # Raw probe data
    probe_u_10hz.csv              # Processed 10Hz
    validation/                   # Figures + reports
```
