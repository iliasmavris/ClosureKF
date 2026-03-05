# Baseline Comparison Summary

Generated: 2026-02-16 20:01
Test set: 2727 samples, horizons (physical time): ['0.1s', '0.2s', '0.5s', '1.0s', '2.0s']

## R2_dx Results

| Model | 0.1s| 0.2s| 0.5s| 1.0s| 2.0s |
|-------|------|------|------|------|------|
| Persistence | -0.0000| -0.0000| -0.0000| -0.0000| -0.0000 |
| Mean-increment | -0.0000| -0.0000| -0.0000| -0.0000| -0.0000 |
| AR(10) | +0.4229| +0.2168| +0.0600| +0.0265| +0.0213 |
| Ridge | +0.4383| +0.2455| +0.1151| +0.0990| +0.1695 |
| MLP | +0.3945| +0.3816| -0.0562| -0.0112| +0.0251 |
| RandomForest | +0.6250| +0.5759| +0.4618| +0.3203| +0.2705 |
| GRU | +0.4016| +0.2229| +0.1030| +0.0859| +0.0928 |
| Physics-only KF | -0.0026| -0.0029| +0.0001| +0.0411| +0.2351 |
| Closure KF | +0.0377| +0.0518| +0.0967| +0.1714| +0.3341 |

## GRU 3-Seed Robustness

- 0.1s: mean=+0.4016, std=0.0197, seeds=[+0.3805, +0.4279, +0.3965]
- 0.2s: mean=+0.2229, std=0.0109, seeds=[+0.2121, +0.2378, +0.2189]
- 0.5s: mean=+0.1030, std=0.0029, seeds=[+0.1006, +0.1071, +0.1013]
- 1.0s: mean=+0.0859, std=0.0013, seeds=[+0.0872, +0.0865, +0.0842]
- 2.0s: mean=+0.0928, std=0.0036, seeds=[+0.0961, +0.0879, +0.0945]

## GRU Training Details

- Seed 42: 300 epochs (best @ 300), early_stop=False, train_loss=0.0005449, val_loss=0.0007240, time=54.0s
- Seed 43: 300 epochs (best @ 300), early_stop=False, train_loss=0.0004971, val_loss=0.0006644, time=54.5s
- Seed 44: 300 epochs (best @ 300), early_stop=False, train_loss=0.0005313, val_loss=0.0007065, time=53.6s

## Key Findings

- Best data-only at 1.0s: **RandomForest** (R2_dx=+0.3203)
- Physics-only KF at 1.0s: R2_dx=+0.0411
- Closure KF at 1.0s: R2_dx=+0.1714
- Closure gain over best data-only: -0.1489

## Notes

- All data-only models use oracle future water velocity (same information as KF open-loop prediction).
- AR(10) is fit on 1-step training increments and iterated for multi-step (no future u).
- Ridge/MLP/RF trained per-horizon with StandardScaler.
- GRU: 3 seeds (42, 43, 44), mean reported. Architecture: GRU(3, 64) + Linear(64, 1), 13313 params.
- KF models use v11.1 seed 1 checkpoint, sequential filter with 50s val warmup.
- Horizons in steps: [1, 2, 5, 10, 20], at 10 Hz = ['0.1s', '0.2s', '0.5s', '1.0s', '2.0s'] physical time.