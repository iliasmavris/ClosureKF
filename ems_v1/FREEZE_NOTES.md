# EM&S Freeze Notes (v1)

Created: 2026-02-16

## What Is Frozen

The following directories are **immutable snapshots**. Their contents must NEVER be modified, renamed, or deleted after this freeze date.

### Lockbox v11.1 (3-seed retrain with alpha fix)
- **Path**: `ems_v1/runs/lockbox_v11_1_alpha_fix_FREEZE/`
- **Source**: Copied from `final_lockbox_v11_1_alpha_fix/` (43 files)
- **Contents**: 3-seed S1+S2 training outputs, checkpoints, metrics tables, horizon curves, training curves, event skill tables, aggregate summaries
- **Audits**:
  - `audit/freeze_integrity.txt` -- all frozen params delta < 1e-12 (PASS)
  - `audit/alpha_parameterization.txt` -- S1 vs S2 alpha equivalence < 1e-6 (PASS)

### Scout Evidence
- **Path**: `ems_v1/scouts/laptop_scout_outputs/` (75 files)
  - MZ-1 gated memory, du-vs-dv proxy, training curves, event skill, failure map
- **Path**: `ems_v1/scouts/scout_pack_v12/` (11 files)
  - D3 AR(3) whitening, D4 sigmoid-gated MZ-1, D5 MZ-2 kernel, D6 failure map
- **Path**: `ems_v1/scouts/scout_whitening_calibration/` (6 files)
  - NIS calibration table, dt-transfer pilot
- **Path**: `ems_v1/scouts/scout_pack_v13_closure_followups/` (17 files)
  - Closure follow-up analyses

### Manuscript Snapshot
- **Path**: `ems_v1/paper/manuscript_ems_v1/` (26 files)
- **Source**: Copied from `manuscript_v12_3_refresh/` (the v12.3 refresh with edits A-D)
- **Note**: This copy is the starting point for EM&S edits. Future TeX edits happen HERE, not in the source folder.

### Data Provenance
- **Path**: `ems_v1/meta/input_md5.csv` -- MD5 hashes for all 6 dataset files (10Hz + 50Hz)
- **Path**: `ems_v1/meta/DATA_PROVENANCE.md` -- full provenance with row counts, dt checks, NaN checks
- 10 Hz MD5s match lockbox v11.1 provenance exactly.

## What Is Allowed to Change

Only the following may be created or modified going forward:

1. **New scripts** under `ems_v1/` (e.g., evaluation scripts, figure generators)
2. **New run outputs** under `ems_v1/runs/` (but NOT inside any `*_FREEZE/` folder)
3. **New evaluation outputs** under `ems_v1/eval/`
4. **New figures** under `ems_v1/figures/`
5. **New tables** under `ems_v1/tables/`
6. **Manuscript edits** in `ems_v1/paper/manuscript_ems_v1/` (TeX content updates)
7. **New meta notes** in `ems_v1/meta/` (but do NOT overwrite existing provenance files)

## Strict Rules

1. **NEVER edit anything inside `ems_v1/runs/*_FREEZE/`** -- these are sealed evidence.
2. **NEVER edit anything inside `ems_v1/scouts/`** -- these are sealed scout outputs.
3. **NEVER modify `ems_v1/meta/input_md5.csv`** or `ems_v1/meta/DATA_PROVENANCE.md` -- data provenance is locked.
4. **New runs** must go to `ems_v1/runs/<descriptive_name>/` (never reuse or overwrite existing run folders).
5. **All evaluation scripts** should reference data via `processed_data_10hz_clean_v1/` or `processed_data_50hz_clean_v1/` (never the old `processed_data_10hz/` path without `_clean_v1`).
6. **Checkpoints** loaded from `ems_v1/runs/lockbox_v11_1_alpha_fix_FREEZE/` are read-only references.

## Provenance Chain

```
300 Hz raw CSVs
  -> rebuild_downsampled_splits_from_300hz.py
    -> processed_data_10hz_clean_v1/ (MD5 fingerprinted)
    -> processed_data_50hz_clean_v1/ (MD5 fingerprinted)
      -> lockbox_v11_1_alpha_fix_3seed.py (3 seeds, softplus alpha)
        -> final_lockbox_v11_1_alpha_fix/ (43 files, audited)
          -> FROZEN into ems_v1/runs/lockbox_v11_1_alpha_fix_FREEZE/
```
