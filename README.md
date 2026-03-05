# ClosureKF v1.0

**A safeguarded protocol for discovering parsimonious closures in state-space environmental models, with application to sediment dynamics**

Ilias Mavris and Manousos Valyrakis
School of Civil Engineering, Aristotle University of Thessaloniki

Paper: *Environmental Modelling & Software* (under review)

---

## Overview

ClosureKF is a two-stage protocol for discovering minimal, unit-consistent
closures within state-space models of sediment particle dynamics.  Stage 1
learns physics parameters via Kalman filter negative log-likelihood; Stage 2
augments the frozen physics with a data-driven closure selected from a
candidate library using innovation diagnostics, a DNLL gate, and
leave-one-out ablation.

The protocol is validated through a virtual-lab benchmark suite with known
truth (no injected closure), null tests, and cross-system stress tests
(Silverbox, NEON, cooling).

## Repository structure

| Directory | Contents |
|-----------|----------|
| `models/` | State-space model implementations (KalmanForecaster, KalmanForecasterClosure) |
| `scripts/` | Training, evaluation, and analysis scripts |
| `utils/` | Data loading and preprocessing utilities |
| `processed_data_10hz_clean_v1/` | Training data (10 Hz, 3 CSV splits) |
| `cooling_benchmark/` | Positive-control benchmark (known closure) |
| `external_benchmarks/` | Silverbox benchmark (no closure expected) |
| `virtual_lab_v1/` | Virtual-lab benchmark scripts and configs |
| `neon_benchmark/` | NEON ecosystem benchmark scripts |
| `cfd_twin_v1/` | CFD flow-twin scripts and lab data |
| `ems_v1/eval/` | Baseline evaluation pipeline |

## Quick start

```bash
pip install -r requirements_frozen.txt
```

## Data and checkpoints

Trained checkpoints and generated virtual-lab datasets are archived on
Zenodo (DOI to be inserted before publication).

## Citation

```bibtex
@article{mavris2026closurekf,
  title   = {A safeguarded protocol for discovering parsimonious
             closures in state-space environmental models, with
             application to sediment dynamics},
  author  = {Mavris, Ilias and Valyrakis, Manousos},
  journal = {Environmental Modelling \& Software},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
