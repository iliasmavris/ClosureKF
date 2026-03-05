# Data Provenance (EM&S Freeze #0)

Generated: 2026-02-16 10:02:53

## Dataset Paths

| Rate | Dir (relative to project root) |
|------|-------------------------------|
| 10 Hz | `processed_data_10hz_clean_v1/` |
| 50 Hz | `processed_data_50hz_clean_v1/` |

## File Inventory

| Rate | Split | File | MD5 | Rows | Size (bytes) | Median dt | dt Error | First ts | Last ts |
|------|-------|------|-----|------|-------------|-----------|----------|----------|--------|
| 10Hz | train | `train_10hz_ready.csv` | `f92ed66982613197bbc5513b7e0e6f1c` | 12609 | 708,007 | 0.100000 | OK | 0.000 | 1260.800 |
| 10Hz | val | `val_10hz_ready.csv` | `4c349a45e4832fe35a1365e6fbbd6a33` | 2661 | 171,724 | 0.100000 | OK | 1260.808 | 1526.808 |
| 10Hz | test | `test_10hz_ready.csv` | `ab0f6eaf6aec834836bee5d431b42a90` | 2727 | 175,834 | 0.100000 | OK | 1526.821 | 1799.421 |
| 50Hz | train | `train_50hz_ready.csv` | `022e71e77e083b023ec7a2fdcf452b36` | 63041 | 3,487,993 | 0.020000 | OK | 0.000 | 1260.800 |
| 50Hz | val | `val_50hz_ready.csv` | `39bde97b55b4ed797b125f848239da0d` | 13301 | 871,482 | 0.020000 | OK | 1260.808 | 1526.808 |
| 50Hz | test | `test_50hz_ready.csv` | `18920a42883be42d8aa74dde2276ee50` | 13634 | 892,767 | 0.020000 | OK | 1526.821 | 1799.481 |

## Integrity Checks

- **NaN check**: PASS -- no NaNs in velocity, displacement, or time_delta
- **Column check**: PASS -- all files have [timestamp, time_delta, velocity, displacement]
- **dt check**: PASS -- all median dt values within 1% of expected
- **Path guard**: Only `_clean_v1` datasets used; no references to old `processed_data_10hz/` (without suffix)

## MD5 Cross-Reference

The MD5 hashes above match the lockbox v11.1 provenance for 10 Hz splits:
- train: `f92ed66982613197bbc5513b7e0e6f1c`
- val: `4c349a45e4832fe35a1365e6fbbd6a33`
- test: `ab0f6eaf6aec834836bee5d431b42a90`

See also: `ems_v1/meta/input_md5.csv`
