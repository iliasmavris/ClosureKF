# Step 4: d2-Only 3-Seed Robustness Pack

## Model Form
Closure: C = -d2 * v * |u|  (b2 frozen to 0)
Trainable params in S2: d2_raw, log_q_scale (2 total)

## Parameter Stability
- d2: mean=2.4802, std=0.1905, CV=7.68%
- q_scale: mean=1.0391, std=0.0429, CV=4.13%

## Hessian-Based Uncertainty (per seed)
- Seed 1: d2 SE=7.7786, q_scale SE=1.4189
- Seed 2: d2 SE=5.6085, q_scale SE=1.7023
- Seed 3: d2 SE=914.8099, q_scale SE=1.5085

## Narrative Survival
- ACF(1) improvement: 3/3 seeds
- DxR2@1s improvement: 3/3 seeds

## Per-Seed Results (warm start, headline horizons)
- Seed 1: d2=2.2560, DxR2@1s phys=+0.0350 clos=+0.1509, ACF1 phys=0.5905 clos=0.5796
- Seed 2: d2=2.7217, DxR2@1s phys=+0.0783 clos=+0.1877, ACF1 phys=0.5438 clos=0.5410
- Seed 3: d2=2.4629, DxR2@1s phys=+0.0490 clos=+0.1640, ACF1 phys=0.5666 clos=0.5596

## Verdict: Narrative SURVIVES