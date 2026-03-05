# Robustness Step 4: d2-Only 3-Seed Pack + Hessian SE

Generated: 2026-02-20 23:01:01
Data: processed_data_10hz_clean_v1/
Seeds: [1, 2, 3]
Model: physics + d2-only closure (b2=0)

## Files
- coeff_uncertainty.csv: Hessian SE + across-seed variability
- summary_seeds_step4.csv: Per-seed + mean/std metrics
- summary_seeds_step4.md: Narrative summary

## Hessian Details
- Method: central finite differences on val NLL
- Epsilon: 0.001
- N windows: 256 (deterministic, indices 0:255)
- Dataset: StateSpaceDataset(L=64, m=64, H=20)
- Delta method for physical-unit SE