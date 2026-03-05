# Step 7: Synthetic Validation + Identifiability (Freeze #6)

## Purpose

Generates synthetic data from the known state-space model, fits the
d2-only closure, and quantifies parameter recovery accuracy. Three
cases test identifiability (in-library), false-positive resistance
(null), and diagnostic sensitivity (out-of-library).

## Script

```
python -u ems_v1/eval/synthetic_step7/run_synthetic_step7.py
```

Runtime: ~10 min on CPU (Windows, 8-core).

## Design

### Synthetic data generation

- State equations match the real model exactly (EKF Jacobian included)
- Exogenous forcing: AR(1) + random bursts (phi=0.95, sigma=0.3)
- Ground-truth physics: alpha=1.7, kappa=1.5, c=25.0, vc=0.19
- Process noise: qx=2.5e-3, qu=1.8e-2; observation noise R=1e-6
- Duration: 400s train / 80s val / 150s test at 10 Hz (dt=0.1)
- Reproducible via fixed seeds (SEED_BASE=42)

### Phase A: Recovery scatter (10 draws)

- d2_true varies over [0.5, 4.0] (10 values)
- Physics fixed to ground truth; only d2 + q_scale trained (S2)
- Tests: "Given correct physics, is d2 identifiable?"

### Phase B: Full pipeline (3 cases, 1 draw each)

- S1 (physics-only) trained from warm initialization near truth
- S2 (d2-only closure) trained from S1 physics
- Evaluation: warm-start (30s) + full test set scoring

**Cases:**

| Case | Closure truth | Expected d2_hat |
|------|---------------|-----------------|
| 1. In-library | C = -d2*u*|v| (d2=2.3) | Near 2.3 |
| 2. Out-of-library | C = -d2*u*|v| - gamma*u*v^2 | Biased |
| 3. Null | C = 0 | ~0 |

### Training config

- S1: L=64, H=20, batch=128, 25 epochs, patience=8
- S2: L=32, H=10, batch=256, 50 epochs, patience=10
- Optimizer: Adam, lr=1e-2, ReduceLROnPlateau

## Outputs

| File | Description |
|------|-------------|
| `synthetic_truth.csv` | 10 rows: d2_true, d2_hat, rel_err for recovery draws |
| `synthetic_fit_results.csv` | 3 rows: per-case fit results + metrics |
| `synthetic_metrics.csv` | 6 rows: per-case per-model (phys/clos) metrics |
| `synthetic_config.json` | Full configuration used |
| `synthetic_headlines.txt` | Human-readable headline stats |
| `fig_synth_recovery.png` | Recovery scatter (d2_true vs d2_hat) |
| `fig_synth_forecast.png` | DxR2 vs horizon (Case 1 + Case 3) |
| `fig_synth_diagnostics.png` | ACF1 + NIS bars across 3 cases |

### Manuscript outputs

| File | Description |
|------|-------------|
| `ems_v1/figures/fig_synth_recovery.pdf` | Recovery scatter |
| `ems_v1/figures/fig_synth_forecast.pdf` | Forecast skill curves |
| `ems_v1/figures/fig_synth_diagnostics.pdf` | Diagnostic bars |
| `ems_v1/tables/table_synth_headlines.tex` | Auto-generated macros (SoT) |

## SoT chain

CSV -> export_headlines() -> synthetic_headlines.txt + table_synth_headlines.tex
-> metrics.tex \input -> results.tex macros

## Key findings

- **Recovery:** Median relative error 11.0% across d2_true in [0.5, 4.0]
- **Null:** |d2_hat| = 0.051, no spurious skill gain
- **Out-of-library:** d2 biased (76% error), diagnostics degrade
- **Physics-closure entanglement:** When S1 alpha drifts from truth,
  d2 recovery degrades -- motivates accurate physics estimation

## Dependencies

- Frozen: `models/kalman_forecaster.py`, `models/kalman_closure.py`,
  `ems_v1/eval/metrics_pack.py`
- No real data used; all synthetic
- No frozen directories modified
