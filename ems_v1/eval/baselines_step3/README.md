# Baselines Step 3 -- README

Generated: 2026-02-16 10:46:30

## Overview

This directory contains the EM&S baseline comparison suite.  All baselines
are evaluated on the 10 Hz clean test set (`processed_data_10hz_clean_v1/
test_10hz_ready.csv`, 2727 samples) using the frozen metric protocol
from `ems_v1/meta/METRICS_FREEZE_1.md`.

## Data Integrity

- **No future peeking**: all features at origin t use only data up to time t.
- **Train/val/test respected**: models fitted on train only, hyperparameters
  tuned on val only, metrics reported on test only.
- **Scalers fit on train only**: StandardScaler for Ridge/MLP/GradBoost.

## Baselines

| ID | Model | Description | Tuning |
|----|-------|-------------|--------|
| B0 | Persistence | dx_pred = 0 | None |
| B1 | Climatology | dx_pred = mean(dx_train_h) | None |
| B2 | AR(10) | AR on 1-step increments, iterated | p from [1, 2, 3, 5, 10], val RMSE@1.0s |
| B3 | Ridge | RidgeCV with 27 compressed features | alpha via RidgeCV internal CV |
| B4 | MLP | sklearn MLPRegressor, 2-layer | hidden from [(32,), (64,), (32, 16)], val RMSE |
| B5 | GradBoost | HistGradientBoostingRegressor | depth/iter grid, val RMSE |

## Feature Set (B3-B5)

27 features per origin, all causal:

- **Velocity lags**: u[t], u[t-1], ..., u[t-9]  (10 features)
- **Increment lags**: dx1[t], dx1[t-1], ..., dx1[t-9]  (10 features)
- **Current state**: x[t], du[t]  (2 features)
- **Summary stats** (last 50 steps = 5.0s): mean(u), std(u),
  mean(|u|), mean(u^2), mean(du)  (5 features)

For val/test origins near the start, lag context is provided by prepending
the tail of the preceding split (causal: these are past observations).

## Grey-box Reference

Grey-box metrics are loaded read-only from:
  `ems_v1/runs/lockbox_v11_1_alpha_fix_FREEZE/seed1/tables/`

The grey-box model is the v11.1 2-term closure (b2+d2) evaluated in oracle
mode.  Per Freeze #2, b2 is operationally inactive (99.1% skill retained
by d2 alone), so these metrics represent the d2-only model.

## Outputs

- `baselines_summary_10hz.csv` -- full comparison table
- `../../tables/table_model_class_comparison.tex` -- LaTeX table for manuscript
- `../../figures/fig_baseline_comparison.png` -- bar chart comparison
- This README
