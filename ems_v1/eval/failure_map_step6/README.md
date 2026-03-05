# Failure Map -- Step 6

## Protocol
- 5 quantile bins on |u| (water velocity magnitude; manuscript notation)
- Computed on scored (test) segment only (2727 points)
- Warm start: last 50s of validation prepended for filter spinup
- Uses frozen Step 4 checkpoints (d2-only, 3 seeds)
- No retraining; eval-only

## Bin edges (|u|, m/s)
0.0000, 0.1387, 0.1582, 0.1769, 0.1982, 0.2850

## Horizons
- Physical seconds: [0.1, 0.2, 0.5, 1.0, 2.0]
- Steps at 10 Hz: [1, 2, 5, 10, 20]
- Max horizon: 20 steps = 2.0s

## Definitions
- delta_skill = skill_dx(closure) - skill_dx(physics)
  - skill_dx = 1 - RMSE/RMSE_base (constant-mean-increment baseline)
  - Positive = closure better
- delta_mae = MAE(closure) - MAE(physics)
  - Negative = closure better
- n_eff = number of bin indices where i + h < N
  - min(n_eff) = 541 (>= 200, PASS)

## Caveat
In low-variance regimes (small |u|), Var(dx_true) is small, making R2/Skill
ill-conditioned. MAE is shown as a variance-robust complement.

## Files
- failure_map_cells.csv: 75 rows (seed x bin x tau)
- failure_map_summary.csv: 25 rows (bin x tau, 3-seed mean/std)
- bin_edges.csv: bin definitions and per-bin sample counts
- fig_failure_map.png: local copy of 2-panel heatmap
- README.md: this file

## External outputs
- ems_v1/figures/fig_failure_map.pdf: manuscript figure
- ems_v1/tables/table_failure_bins.tex: LaTeX table

## Runtime
2s (0.0 min)


## Verified Headline Numbers

```
STEP 6 HEADLINE NUMBERS
Source: failure_map_summary.csv (25 cells, 5 bins x 5 horizons, 3 seeds)

MAX_DSKILL:
  bin=3 [0.1769, 0.1982) tau=0.2s
  mean_dskill=+0.1155 std=0.0181

MIN_DSKILL:
  bin=0 [0.0000, 0.1387) tau=0.2s
  mean_dskill=-0.0281 std=0.0046

MAE_SIGN_CHECK:
  max(delta_mae_mean)=-0.000127
  min(delta_mae_mean)=-0.011505
  all_nonpositive=True

BEST_DMAE (most negative):
  bin=4 [0.1982, 0.2850) tau=1.0s
  mean_dmae=-0.011505 std=0.001421

BIN_EDGES:
  Bin 0: [0.0000, 0.1387) m/s  n_scored=546
  Bin 1: [0.1387, 0.1582) m/s  n_scored=545
  Bin 2: [0.1582, 0.1769) m/s  n_scored=545
  Bin 3: [0.1769, 0.1982) m/s  n_scored=545
  Bin 4: [0.1982, 0.2850) m/s  n_scored=546
```
