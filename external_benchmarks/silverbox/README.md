# Silverbox External Benchmark for ClosureKF

## Purpose

This is a **protocol stress test**, not a model validation. The ClosureKF pipeline
(physics KF + closure discovery + DNLL gate) was developed for sediment transport.
Here we apply it unchanged to the Silverbox benchmark -- an electronic Duffing
oscillator -- to verify the discovery pipeline behaves sensibly on out-of-domain data
where the physics model is structurally wrong.

## Dataset

- **Source:** [nonlinearbenchmark.org/benchmarks/silverbox](https://www.nonlinearbenchmark.org/benchmarks/silverbox)
- **Citation:** Wigren, T. & Schoukens, J. (2013). Three free data sets for development and benchmarking in nonlinear system identification. *European Control Conference (ECC)*.
- **System:** Electronic Duffing oscillator (2nd-order LTI + cubic nonlinearity in feedback path)
- **Native sampling rate:** 610.35 Hz (= 10^7 / 2^14)
- **Signals:**
  - V1 = input (driving force) -> mapped to `velocity` column (dimensionless forcing)
  - V2 = output (displacement) -> mapped to `displacement` column (DC-removed, dimensionless)

**Signals are treated as dimensionless. No sediment-transport interpretation is claimed.**

## Preprocessing

- Downsampled by factor 61: `dt_eff = native_sampling_time * 61` (exact float, NOT rounded)
- DC removal only on output: `displacement = V2 - mean(V2)`. No z-scoring.
- Chronological splits: 70% train / 15% val / 15% test

## Pipeline

The discovery protocol is **identical** to the empirical pipeline (Algorithm 1 in main text):

1. **Stage 1:** Train physics-only `KalmanForecaster(use_kappa=True)` on train, validate on val
2. **Stage 2:** Initialize `KalmanForecasterClosure(alpha_param="softplus")` from S1 physics, freeze physics, train all 6 closure terms + q_scale
3. **Term Selection:**
   - Relative variance filter (threshold: 5%)
   - DNLL gate (threshold: 0.001 NLL units)
   - Ablation check
4. **Evaluation:** DxR2(h), ACF(1), NIS, 90% coverage on test set with val-tail warmup (50s)

## Expected Behavior

The sediment-transport physics model (threshold forcing + restoring spring) is structurally
wrong for a Duffing oscillator. Expected outcomes:

- **S1 physics:** Moderate fit -- linear+threshold can approximate some Duffing dynamics
- **Closure terms:** The 6-term library may identify compensatory terms
- **DNLL gate:** Should function correctly (accept genuine improvements, reject noise)
- **Overall:** Pipeline runs end-to-end without producing nonsensical results

## Horizon Mapping

All target horizons are mapped via `dt_eff`:
- `h_steps = round(target_sec / dt_eff)` for each target {0.1, 0.2, 0.5, 1.0, 2.0} s
- Achieved horizon reported as `h_steps * dt_eff`

## Reproducibility

- 3 seeds (1, 2, 3)
- All hyperparameters in `configs/pipeline_config.json`
- SHA-256 hashes of all data files in `manifest.json`
- CPU-only (deterministic on same hardware)

## Compute Cost

Approximate runtime on CPU: ~30-60 minutes total (3 seeds x S1 + S2 + selection + evaluation).

## Directory Structure

```
silverbox/
  configs/pipeline_config.json    # All hyperparameters
  data/
    processed/silverbox_processed.csv
    splits/{train,val,test}.csv
  scripts/
    prepare_silverbox.py          # Step 1: download + preprocess
    create_splits.py              # Step 2: temporal splits
    run_closurekf_silverbox.py    # Steps 3-4: train + evaluate
    make_figure.py                # Step 5: figure generation
  outputs/
    metrics.json                  # Full evaluation metrics
    selection_summary.json        # Term selection details
    innovation_diagnostics.json   # ACF, NIS, coverage per seed
    seed_logs/seed{1,2,3}/        # Checkpoints + refit logs
  figures/
    fig_ext_silverbox_summary.pdf # Summary figure
    fig_ext_silverbox_summary.png
  manifest.json                   # Data provenance + hashes
  README.md                       # This file
```

## Gate OFF vs Gate ON (Squeezed ROI)

We also report what would happen without the DNLL gate (`gate_off_vs_on.json`).
Variance-only selection (Gate OFF) would select b2 in 3/3 seeds (rel.var 0.10--0.97).
The DNLL gate rejects b2 in all 3 seeds (DNLL = 7.4e-5, threshold = 0.001).
This reproduces the documented false-positive suppression mechanism out-of-domain.

## No Source Modifications

Zero existing source files were modified. All code lives under `external_benchmarks/silverbox/`.
The pipeline imports from the project root via `sys.path.insert`.
