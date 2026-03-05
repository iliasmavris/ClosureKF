# Clean 10.0 Hz Dataset (v1)

**Generated:** 2026-02-15 10:25:34
**Sampling rate:** 10.0 Hz (dt = 0.1000s)

## Source

Rebuilt from three independent 300 Hz source files:
- `300 hz train.csv` (0.0 - 1260.81s)
- `300 hz val.csv` (1260.81 - 1526.81s)
- `300hz test.csv` (1526.82 - 1799.49s)

Each split processed independently -- no concatenation, no shared state.

## Method

1. Estimate raw sampling rate from median(diff(timestamp)).
2. Apply 4th-order Butterworth lowpass at 4.0 Hz (0.8 x Nyquist = 0.8 x 5.0 Hz) using sosfiltfilt.
3. Interpolate to uniform 10.0 Hz grid (dt = 0.1000s) via np.interp.
4. Each split processed independently from its own source file.

## Splits

| Split | Rows | t_min (s) | t_max (s) | Duration (s) | SHA-256 (first 16) |
|-------|------|-----------|-----------|-------------|---------------------|
| train | 12609 | 0.00 | 1260.80 | 1260.80 | 75e44f18b644d36a... |
| val | 2661 | 1260.81 | 1526.81 | 266.00 | d30098bbb404726c... |
| test | 2727 | 1526.82 | 1799.42 | 272.60 | aa9ac683bad97699... |

## Column Schema

Matches `processed_data_10hz/*_10hz_ready.csv`:
- `timestamp`: float seconds
- `time_delta`: constant 0.1000s (0.0 for first row)
- `velocity`: float m/s (water velocity)
- `displacement`: float m (sediment position)