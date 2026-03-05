# Metric Specification — Freeze #1

Created: 2026-02-16
Status: **FROZEN** after this date. Do not modify without creating METRICS_FREEZE_2.md.

Machine-readable companion: `ems_v1/meta/METRICS_CONFIG_EMS.json`
Reusable helper: `ems_v1/eval/metrics_pack.py`

---

## A. Data Splits and Provenance

### Datasets

| Label | Path | dt (s) |
|-------|------|--------|
| 10 Hz clean v1 | `processed_data_10hz_clean_v1/{train,val,test}_10hz_ready.csv` | 0.1 |
| 50 Hz clean v1 | `processed_data_50hz_clean_v1/{train,val,test}_50hz_ready.csv` | 0.02 |

MD5 fingerprints are locked in `ems_v1/meta/input_md5.csv`.

### Split Roles

| Split | Role |
|-------|------|
| Train | Model fitting (Stages 1 and 2). Never used for evaluation. |
| Val | Hyperparameter selection, early stopping, and warmup prefix source. Never scored. |
| Test | All reported metrics. The ONLY split that contributes to published numbers. |

### Scored Region

Not every test-set index is scorable for every horizon. The scored region depends on:

- **eval_start**: index of first scorable observation (after warmup).
- **H_max**: largest forecast horizon evaluated (in steps).

For innovation diagnostics (NIS, ACF, cov90):

    N_scored = N_test  (all test observations after warmup boundary)

For h-step displacement-increment metrics (DxR2, MAE, RMSE, skill):

    Scorable origins for horizon h: indices i in [max(eval_start, 1), N - h)
    where N = total length of the evaluation array (warmup + test).

Each horizon h has its own count of valid origins; this count decreases as h grows.
A minimum of 10 valid origins per horizon is required; otherwise the metric is NaN.

### Warmup Protocol

The Kalman filter requires a spin-up period for state/covariance convergence.

| Parameter | 10 Hz | 50 Hz |
|-----------|-------|-------|
| Warmup duration | 50.0 s | 50.0 s |
| Warmup steps | 500 | 2500 |
| Source | Last 50 s of validation split | Last 50 s of validation split |

**Rule:** Warmup observations are used for filter conditioning only. They are NEVER scored. The scoring boundary is the first test-set timestamp. An assertion guards this:

    assert n_scored == len(df_test)

Two evaluation modes:
- **Warm start** (primary): 50 s val prefix feeds the filter before test scoring.
- **Cold start** (secondary): filter initialised from learned prior P0; no warmup.

---

## B. Primary Forecasting Targets

### Delta forecast (PRIMARY)

    dx_t(h) = x_{t+h} - x_t

The displacement increment over h steps. This is the primary target because:
1. It directly measures transport prediction skill.
2. It avoids inflated R^2 from level variance (level series is non-stationary).
3. It aligns with the physical quantity of interest (how far does the particle move?).

### Level forecast (SECONDARY, reported for interpretability)

    x_{t+h}

Reported as R^2_level for legacy comparison only. Not used for model selection or headline claims.

---

## C. Metrics (Exact Formulas)

All sums below run over scorable origins for the given horizon h.

### C.1 R^2_dx (displacement-increment R-squared)

    R^2_dx(h) = 1 - SS_res / SS_tot

    SS_res = sum_t [ dx_t^pred(h) - dx_t^true(h) ]^2
    SS_tot = sum_t [ dx_t^true(h) - mean(dx_true(h)) ]^2

where:
- dx_t^pred(h) = x_hat_{t+h|t} - x_t (open-loop Kalman prediction minus initial observation)
- dx_t^true(h) = x_{t+h} - x_t (observed increment)
- mean(dx_true(h)) = sample mean of observed increments at horizon h

**Baseline interpretation:** The denominator is the variance of observed increments (the constant-mean-increment baseline). R^2_dx > 0 means the model beats predicting the average increment. This is NOT zero-change persistence.

Guard: if SS_tot < 1e-15, R^2_dx = 0.0.

### C.2 R^2_level (level R-squared)

    R^2_level(h) = 1 - Var(x_{t+h}^pred - x_{t+h}^true) / Var(x_{t+h}^true)

Reported as secondary metric only.

### C.3 Skill vs Persistence (dx)

The persistence baseline predicts zero displacement change (hold x constant):

    dx_t^persist(h) = 0    (equivalently: x_hat_{t+h} = x_t)

Note: This differs from the R^2_dx denominator (which uses the mean increment, not zero). Both are useful:
- R^2_dx compares to climatological mean increment.
- Skill_dx compares to zero-change persistence.

RMSE-based skill:

    RMSE(h)      = sqrt( mean( [dx_true(h) - dx_pred(h)]^2 ) )
    RMSE_base(h) = sqrt( mean( [dx_true(h) - mean(dx_true(h))]^2 ) )
                 = sqrt( Var(dx_true(h)) )

    Skill_dx(h) = 1 - RMSE(h) / RMSE_base(h)

Guard: if RMSE_base < 1e-15, Skill_dx = 0.0.

Range:
- Skill > 0: model beats constant-mean baseline.
- Skill = 0: model equals baseline.
- Skill < 0: model worse than baseline.

### C.4 MAE and RMSE (dx)

    MAE_dx(h)  = mean( |dx_true(h) - dx_pred(h)| )
    RMSE_dx(h) = sqrt( mean( [dx_true(h) - dx_pred(h)]^2 ) )

Reported alongside R^2_dx for all horizons.

### C.5 Open-Loop Prediction Mechanics

Predictions are generated in **oracle mode**: the model uses actual observed water velocity u(t) and velocity change du(t) = u(t) - u(t-1) as external forcing during the open-loop rollout. The Kalman update step is disabled (predict-only, no measurement corrections). Starting state is the post-update (filtered) state at time t.

Steps for origin t, horizon h:
1. Set (sx, su) = Kalman filtered state at t.
2. For step k = 1, ..., h:
   - dt_k = t[t+k] - t[t+k-1]  (use actual dt; fallback 0.1 if <= 0)
   - v_w = u[t+k-1], dv_w = u[t+k-1] - u[t+k-2]  (oracle forcing)
   - rho = exp(-alpha * dt_k)
   - physics: sx_new = sx + su * dt_k
   - velocity: su_new = rho*su + [-kappa*sx + c*max(v_w^2 - vc^2, 0) + C(su, v_w, dv_w)] * dt_k
3. dx_pred(h) = sx_final - x_obs[t]
4. dx_true(h) = x_obs[t+h] - x_obs[t]

---

## D. Innovation Diagnostics (Filter Fit)

### D.1 Innovations

    e_t = y_t - x_hat_{t|t-1}

where y_t is the observation and x_hat_{t|t-1} is the prior (predicted) state estimate for the displacement component.

### D.2 Innovation Variance

    S_t = P_{t|t-1}^{(0,0)} + R

where P_{t|t-1}^{(0,0)} is the (0,0) element of the prior covariance matrix and R is the observation noise variance.

### D.3 NIS (Normalised Innovation Squared)

    NIS = mean( e_t^2 / S_t )     (computed over scored region, excluding NaN)

Expected value: 1.0 for a correctly specified filter.
- NIS > 1: filter is overconfident (underestimates uncertainty).
- NIS < 1: filter is conservative (overestimates uncertainty).

Guard: S_t clamped to max(S_t, 1e-15).

### D.4 ACF (Autocorrelation Function of Innovations)

    rho(lag) = sum_{t=1}^{N-lag} e_c[t] * e_c[t+lag]  /  (N * Var(e))

where e_c = e - mean(e) (centred innovations), N = number of scored observations.

Note: Denominator uses N (not N - lag). This is the "biased" estimator, consistent with numpy and standard Kalman filter literature.

Reported lags: {1, 5, 10} (minimum). Full ACF up to lag 50 stored.

### D.5 Ljung-Box Portmanteau Test

    Q(m) = N * (N + 2) * sum_{k=1}^{m} [ rho_hat(k)^2 / (N - k) ]

Compared to chi^2(m) distribution. p-value = 1 - CDF(Q, df=m).

Reported at lags: {5, 10, 20, 50}.

Guard: skip lag m if m >= N or m >= len(acf_values).

### D.6 cov90 (90% Predictive Coverage)

    cov90 = mean( |e_t| <= z_90 * sqrt(S_t) )

where z_90 = 1.6449 (critical value such that P(|Z| <= 1.6449) = 0.90 for standard normal).

Expected value: 0.90.
- cov90 < 0.90: filter underestimates uncertainty.
- cov90 > 0.90: filter overestimates uncertainty (conservative).

Guard: S_t clamped to max(S_t, 1e-15).

---

## E. Horizons (Physical-Time Aligned)

### Headline Horizons

| Physical time (s) | 10 Hz steps | 50 Hz steps |
|--------------------|-------------|-------------|
| 0.1 | 1 | 5 |
| 0.2 | 2 | 10 |
| 0.5 | 5 | 25 |
| 1.0 | 10 | 50 |

These four physical times are the headline horizons for cross-rate comparison.

### Extended Horizons

Full horizon curves may be plotted up to:
- 10 Hz: 100 steps (10.0 s)
- 50 Hz: 500 steps (10.0 s)

But the four headline horizons above are the only ones used for summary tables and model-selection claims.

### Horizon Mapping Rule

When comparing models across sampling rates, always align on **physical time**, not step count. For example, "1.0 s horizon" means step 10 at 10 Hz and step 50 at 50 Hz.

---

## F. Event vs Non-Event Subsets

### Event Detection Algorithm

Events are detected by a 2-cluster k-means on displacement x, with hysteresis:

1. **K-means(k=2)** on displacement values:
   - Initial centroids: 25th percentile (c1) and 75th percentile (c2) of x.
   - Iterate: assign labels by nearest centroid, update centroids.
   - Converge when |delta_c| < 1e-8 or max 100 iterations.

2. **Hysteresis filter** (min_persist = 3 steps):
   - A state transition is only accepted if the new label persists for >= 3 consecutive steps.
   - This removes noise-induced label flickering.

3. **Event indices**: transitions where the cleaned label changes (diff != 0).

### Event Windows

    event_radius = 10.0 s
    event_radius_steps = int(event_radius / dt)
        10 Hz: 100 steps
        50 Hz: 500 steps

For each detected event index:
    event_mask[max(eval_start, idx - radius) : min(N, idx + radius + 1)] = True

### Subsets

| Subset | Definition |
|--------|-----------|
| full | All test-set scored origins |
| event | Origins within +/- event_radius of any detected transition |
| nonevent | Origins outside all event windows |

### Metrics Reported per Subset

All primary metrics (R^2_dx, Skill_dx, MAE_dx, RMSE_dx) are computed per subset for each horizon.

**Caveat:** In nonevent periods, Var(dx_true) may be very small, making R^2_dx ill-conditioned (small denominator). Therefore MAE_dx and RMSE_dx are MANDATORY for nonevent subsets; R^2_dx is reported but flagged as potentially unstable when Var(dx_true) < 1e-6.

---

## G. Grey-Box Diagnostics (Closure Contribution)

These quantify how much of the total velocity update comes from the closure term vs physics.

### Grey-box Fraction

    grey_frac = Var(cl_dt) / Var(cl_dt + physics_drift)

where cl_dt is the closure contribution (C * dt) and physics_drift is the non-closure velocity update terms. Computed over scored region.

### Grey-box Median Ratio

    grey_med = median( |cl_dt| / max(|cl_dt + physics_drift|, 1e-15) )

More robust to outliers than the variance-based fraction.

---

## H. Summary: Primary vs Secondary Metrics

### Primary (headline, used for model selection and claims)

| Metric | Type | Reported at |
|--------|------|-------------|
| R^2_dx(h) | Forecast skill | Headline horizons: 0.1, 0.2, 0.5, 1.0 s |
| Skill_dx(h) | Skill vs persistence | Headline horizons |
| MAE_dx(h) | Absolute error | Headline horizons |
| RMSE_dx(h) | Root-mean-square error | Headline horizons |

### Secondary (reported for completeness, not used for claims)

| Metric | Type | Reported at |
|--------|------|-------------|
| R^2_level(h) | Level forecast skill | Headline horizons |
| ACF(1) | Innovation whiteness | Lag 1 (primary), lags 5, 10 (extended) |
| NIS | Filter calibration | Scalar (test set) |
| cov90 | Predictive coverage | Scalar (test set) |
| LB_p(20) | Ljung-Box p-value | Lag 20 |
| grey_frac | Closure contribution | Scalar (closure model only) |
| grey_med | Closure magnitude ratio | Scalar (closure model only) |

---

## I. Versioning

This is Freeze #1 of the metric specification. If any formula, threshold, or convention changes, a new file `METRICS_FREEZE_2.md` must be created with a diff log referencing this document. The JSON companion `METRICS_CONFIG_EMS.json` must be updated in lockstep.
