# Step 8: Computational Performance (Freeze #7)

## Purpose

Documents training cost, inference throughput, and model footprint
for EM&S reviewer evidence. No retraining; parses existing logs and
runs short deterministic benchmarks.

## Script

```
python -u ems_v1/eval/performance_step8/run_performance_step8.py
```

Runtime: < 2 min on CPU.

## What is measured

### A) Training wall time (from logs)

Parsed from `progress.log` files in the training run directories.
Timestamps are extracted and differenced to compute S1/S2/total
durations per seed.

- **10 Hz (3 seeds):** Mean 2.3 h/seed
  - S1 (physics, L=512): 2.1 h
  - S2 (closure, L=64):  9 min
- **50 Hz (1 seed):**    32 h total

### B) Inference throughput (CPU benchmark)

Filter and multi-horizon forecast timed with `time.perf_counter()`,
5 repeats, median reported.

- Filter: 0.0410 s for 3227 steps
  (0.0127 ms/step)
- Forecast (20 horizons): 0.0780 s for
  2727 scored origins

### C) Model size

- S1 state_dict: 9 scalar parameters
- S2 trainable: 2 (d2_raw, log_q_scale)
- S1 checkpoint: 3.5 KB
- S2 checkpoint: 5.6 KB

### D) Hardware

- Processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- OS: Windows 10 (10.0.26200)
- Training: CPU only (GPU unused for sequential KF)
- Inference: pure NumPy (no deep learning framework needed)

## Outputs

| File | Description |
|------|-------------|
| `performance_raw.json` | All raw timings and repeats |
| `performance_summary.csv` | One row per benchmark item |
| `HEADLINE.txt` | Human-readable headline stats |
| `README.md` | This file |

### Manuscript outputs

| File | Description |
|------|-------------|
| `ems_v1/figures/fig_compute_perf.pdf` | 2-panel figure |
| `ems_v1/tables/table_compute_headlines.tex` | Auto-generated macros |

## SoT chain

```
progress.log -> parse_training_times() -> perf dict
benchmark -> perf dict
perf dict -> HEADLINE.txt + table_compute_headlines.tex
          -> metrics.tex \input hook
          -> methods.tex macro references
```

## Dependencies

- Frozen checkpoints (read-only): `ems_v1/runs/lockbox_ems_v1_d2only_10hz_3seed/`
- Clean data (read-only): `processed_data_10hz_clean_v1/`
- No frozen directories modified
