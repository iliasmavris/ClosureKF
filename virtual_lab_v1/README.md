# Virtual Lab v1 -- Phase 2: Synthetic Particle Displacement Generator

Multi-condition "virtual lab" particle displacement x(t) using a physically
consistent near-bed sphere dynamics model (Newtonian drag + added mass +
stick-slip Coulomb friction). The flow field comes from Phase 1 CFD twin
(OpenFOAM pimpleFoam, validated r=0.997).

## Non-circularity

The sphere truth model is **structurally different** from the reduced EKF/closure
model used in the manuscript:
- Drag: blended Stokes + form drag (not relu threshold)
- Friction: discontinuous Coulomb stick-slip (not exponential decay)
- No rho*u relaxation, no closure terms, no Kalman states

## Pipeline

```
10_add_probes.py      -> write multi-height OF probes overlay
11_run_flow_case.sh   -> run OF in WSL2 (~16 min)
12_extract_flow_probes.py -> parse multi-probe OF output to CSV
13_process_flow_10hz.py   -> Butter 4Hz + filtfilt + 10Hz resample
14_truth_ball_sim.py      -> sphere truth model (RK4 + stick-slip)
15_make_dataset.py        -> orchestrate one condition end-to-end
16_sweep_conditions.py    -> run 8 conditions from sweep_grid.yaml
17_qc_plots.py            -> QC figures + report
```

## Quick start

```bash
# Step 0: Set up probes
python virtual_lab_v1/scripts/10_add_probes.py

# Step 1: Run OF (in WSL)
wsl bash virtual_lab_v1/scripts/11_run_flow_case.sh

# Step 2: Extract + process flow
python virtual_lab_v1/scripts/12_extract_flow_probes.py
python virtual_lab_v1/scripts/13_process_flow_10hz.py

# Step 3-4: Run all 8 conditions
python virtual_lab_v1/scripts/16_sweep_conditions.py

# Step 5: QC
python virtual_lab_v1/scripts/17_qc_plots.py
```
