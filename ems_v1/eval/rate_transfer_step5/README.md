# Step 5A: Transfer Evaluation (10 Hz -> 50 Hz)

Generated: 2026-02-16 21:53:27

## Protocol
- 10 Hz-trained d2-only checkpoints (3 seeds) evaluated on both
  10 Hz and 50 Hz test sets using identical codepath
- Warm start: last 50s of validation for filter spinup
- Oracle open-loop rollout for multihorizon DxR2
- dt inferred from CSV at runtime, headline horizons in physical time

## Key Results
- Transfer ratio at tau=1.0s: 1.2865 (PASS, threshold >= 0.80)
- NIS shift: -0.1928
- ACF1 shift: +0.1001

## Files
- transfer_10hz_to_50hz.csv: Per-seed, per-tau transfer ratios
- transfer_diag_10hz_to_50hz.csv: Innovation diagnostics
- transfer_horizon_dense_seed{1,2,3}.csv: Dense DxR2 curves
- fig_transfer_dxr2_vs_tau.png: DxR2 horizon curves
- fig_transfer_calib.png: NIS + coverage bars

## Checkpoints Used
- seed 1: ems_v1\runs\lockbox_ems_v1_d2only_10hz_3seed\seed1\checkpoints\closure_d2only_seed1.pth
- seed 2: ems_v1\runs\lockbox_ems_v1_d2only_10hz_3seed\seed2\checkpoints\closure_d2only_seed2.pth
- seed 3: ems_v1\runs\lockbox_ems_v1_d2only_10hz_3seed\seed3\checkpoints\closure_d2only_seed3.pth