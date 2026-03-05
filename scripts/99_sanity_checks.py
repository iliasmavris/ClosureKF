#!/usr/bin/env python
"""
99_sanity_checks.py -- Fast, deterministic durability audit.

Checks:
  A) Raw CSV integrity (columns, timestamps, NaN, dt consistency)
  B) Dataset window/slicing (shapes, NaN, alignment, normalization)
  C) KalmanForecaster numerical stability (NaN, P symmetry/PSD, S > 0)
  D) Torch vs Numpy micro-equivalence (50-step open-loop comparison)
  E) Outputs: report_99.md, report_99.json, manifest_99.json

Exit 0 if all HARD checks pass, 1 otherwise.
Runtime: < 2 min on CPU.
"""

import sys
import os
import argparse
import json
import time
import hashlib
import math
import traceback
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch

# ── project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(ROOT / "models"))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_forecaster import KalmanForecaster

# ======================================================================
# Helpers
# ======================================================================

HARD = "HARD"
SOFT = "SOFT"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckResult:
    def __init__(self, name, level, passed, detail=""):
        self.name = name
        self.level = level  # HARD or SOFT
        self.passed = passed
        self.detail = detail

    def tag(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{self.level}] [{status}] {self.name}"

    def to_dict(self):
        return {
            "name": self.name,
            "level": self.level,
            "passed": self.passed,
            "detail": self.detail,
        }


# ======================================================================
# A) RAW CSV INTEGRITY
# ======================================================================

def check_csv_integrity(csv_paths, dt_expected=0.1, dt_tol=1e-3, seed=123):
    """Check raw CSV files for column presence, monotonicity, NaN, dt consistency."""
    results = []
    per_file_stats = []
    rng = np.random.RandomState(seed)

    for csv_path in csv_paths:
        p = Path(csv_path)
        tag = p.name

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            results.append(CheckResult(f"A.read:{tag}", HARD, False, str(e)))
            continue

        # A1: required columns
        required = ["timestamp", "time_delta", "velocity", "displacement"]
        missing = [c for c in required if c not in df.columns]
        results.append(CheckResult(
            f"A1.columns:{tag}", HARD, len(missing) == 0,
            f"missing: {missing}" if missing else f"{len(df)} rows, {len(df.columns)} cols"
        ))
        if missing:
            continue

        ts = df["timestamp"].values
        dt_col = df["time_delta"].values
        vel = df["velocity"].values
        disp = df["displacement"].values

        # A2: timestamp monotonic (strictly increasing)
        ts_diff = np.diff(ts)
        strictly_inc = np.all(ts_diff > 0)
        non_dec = np.all(ts_diff >= 0)
        if strictly_inc:
            results.append(CheckResult(f"A2.monotonic:{tag}", HARD, True, "strictly increasing"))
        elif non_dec:
            n_equal = int(np.sum(ts_diff == 0))
            results.append(CheckResult(f"A2.monotonic:{tag}", SOFT, True,
                                       f"non-decreasing ({n_equal} ties)"))
        else:
            n_neg = int(np.sum(ts_diff < 0))
            results.append(CheckResult(f"A2.monotonic:{tag}", HARD, False,
                                       f"{n_neg} reversals"))

        # A3: time_delta sanity
        all_pos = bool(np.all(dt_col > 0) if len(dt_col) > 1 else True)
        # first row dt may be 0 (initial)
        dt_valid = dt_col[1:] if len(dt_col) > 1 else dt_col
        all_pos_skip1 = bool(np.all(dt_valid > 0))
        med_dt = float(np.median(dt_valid)) if len(dt_valid) > 0 else float("nan")
        dt_close = abs(med_dt - dt_expected) < dt_tol
        results.append(CheckResult(
            f"A3.dt_sanity:{tag}", HARD, all_pos_skip1 and dt_close,
            f"median_dt={med_dt:.6f} (expected {dt_expected}), all_pos(skip1)={all_pos_skip1}"
        ))

        # A4: diff(timestamp) matches time_delta
        if len(ts) > 1:
            ts_deltas = np.diff(ts)
            dt_from_col = dt_col[1:]  # time_delta for rows 1..end
            abs_err = np.abs(ts_deltas - dt_from_col)
            max_abs = float(np.max(abs_err))
            med_abs = float(np.median(abs_err))
            results.append(CheckResult(
                f"A4.dt_consistency:{tag}", HARD, max_abs < 1e-4 and med_abs < 1e-6,
                f"max_err={max_abs:.2e}, median_err={med_abs:.2e}"
            ))
        else:
            results.append(CheckResult(f"A4.dt_consistency:{tag}", HARD, False, "single row"))

        # A5: NaN/Inf
        has_nan = bool(np.any(np.isnan(vel)) or np.any(np.isnan(disp))
                       or np.any(np.isnan(ts)) or np.any(np.isnan(dt_col)))
        has_inf = bool(np.any(np.isinf(vel)) or np.any(np.isinf(disp))
                       or np.any(np.isinf(ts)) or np.any(np.isinf(dt_col)))
        results.append(CheckResult(
            f"A5.no_nan_inf:{tag}", HARD, not has_nan and not has_inf,
            f"nan={has_nan}, inf={has_inf}"
        ))

        # A6: basic stats (informational)
        per_file_stats.append({
            "file": tag,
            "n_rows": len(df),
            "v_mean": float(np.mean(vel)),
            "v_std": float(np.std(vel)),
            "x_min": float(np.min(disp)),
            "x_max": float(np.max(disp)),
            "x_range": float(np.max(disp) - np.min(disp)),
            "dt_median": med_dt,
        })

    return results, per_file_stats


# ======================================================================
# B) DATASET WINDOW / SLICING CHECKS
# ======================================================================

def check_dataset(csv_paths, L, m, H, predict_deltas, norm_stats_path, seed, B=8):
    """Instantiate dataset, check shapes, NaN, alignment."""
    results = []

    # B1: Instantiate (no normalization for alignment checks)
    try:
        ds = StateSpaceDataset(
            csv_paths=csv_paths,
            L=L, m=m, H=H,
            predict_deltas=predict_deltas,
            normalize=False,
            norm_stats_path=None,
        )
        results.append(CheckResult("B1.dataset_init", HARD, True,
                                   f"len={len(ds)}"))
    except Exception as e:
        results.append(CheckResult("B1.dataset_init", HARD, False, str(e)))
        return results

    # B2: __len__ > 0
    results.append(CheckResult("B2.len_positive", HARD, len(ds) > 0,
                               f"len={len(ds)}"))
    if len(ds) == 0:
        return results

    # B3: sample shapes
    np.random.seed(seed)
    torch.manual_seed(seed)
    idxs = np.random.choice(len(ds), size=min(B, len(ds)), replace=False)

    shape_ok = True
    shape_detail = []
    for idx in idxs:
        sample = ds[int(idx)]
        v_hist, dt_hist, x_hist, v_fut, dt_fut, x_fut_true, x_current, meta = sample

        expected = {
            "v_hist": (L,), "dt_hist": (L,), "x_hist": (m,),
            "v_fut": (H,), "dt_fut": (H,), "x_fut_true": (H,),
        }
        for name, tensor in [("v_hist", v_hist), ("dt_hist", dt_hist),
                              ("x_hist", x_hist), ("v_fut", v_fut),
                              ("dt_fut", dt_fut), ("x_fut_true", x_fut_true)]:
            if tensor.shape != expected[name]:
                shape_ok = False
                shape_detail.append(f"{name}: got {tensor.shape}, expected {expected[name]}")

        if x_current.shape != ():
            shape_ok = False
            shape_detail.append(f"x_current: got {x_current.shape}, expected scalar")

    results.append(CheckResult(
        "B3.shapes", HARD, shape_ok,
        "all OK" if shape_ok else "; ".join(shape_detail[:5])
    ))

    # B4: No NaN/Inf in tensors
    nan_found = False
    for idx in idxs:
        sample = ds[int(idx)]
        for i, name in enumerate(["v_hist", "dt_hist", "x_hist", "v_fut",
                                   "dt_fut", "x_fut_true", "x_current"]):
            t = sample[i]
            if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
                nan_found = True
                break
        if nan_found:
            break
    results.append(CheckResult("B4.no_nan_inf", HARD, not nan_found))

    # B5: Alignment check (forcing-at-start-of-interval: v_fut[0]=v[t], x_fut[0]=x[t+1])
    # Dataset is instantiated with normalize=False, so x_fut_true is always absolute
    # (predict_deltas only activates inside the normalize block).
    align_ok = True
    align_detail = []
    for idx in idxs[:4]:  # check a few
        sample = ds[int(idx)]
        v_hist, dt_hist, x_hist, v_fut, dt_fut, x_fut_true, x_current, meta = sample
        run_idx = meta["run_idx"]
        t = meta["t_index"]
        run = ds.runs[run_idx]

        # v_fut[0] should be v_raw[t] (forcing-at-start-of-interval)
        v_raw_t = run["v"][t]
        v_got = v_fut[0].item()
        if abs(v_got - v_raw_t) > 1e-5:
            align_ok = False
            align_detail.append(f"idx={idx}: v_fut[0]={v_got:.6f} != v_raw[t]={v_raw_t:.6f}")

        # v_fut[0] should equal v_hist[-1] (continuity at boundary)
        v_hist_last = v_hist[-1].item()
        if abs(v_got - v_hist_last) > 1e-5:
            align_ok = False
            align_detail.append(f"idx={idx}: v_fut[0]={v_got:.6f} != v_hist[-1]={v_hist_last:.6f}")

        # x_current should be x_raw[t]
        x_raw_t = run["x"][t]
        x_got = x_current.item()
        if abs(x_got - x_raw_t) > 1e-5:
            align_ok = False
            align_detail.append(f"idx={idx}: x_current={x_got:.6f} != x_raw[t]={x_raw_t:.6f}")

        # x_fut_true[0] should be x_raw[t+1] (absolute, since normalize=False)
        x_raw_t1 = run["x"][t + 1]
        expected_val = x_raw_t1
        x_fut_got = x_fut_true[0].item()
        if abs(x_fut_got - expected_val) > 1e-5:
            align_ok = False
            align_detail.append(f"idx={idx}: x_fut[0]={x_fut_got:.6f} != x_raw[t+1]={expected_val:.6f}")

    results.append(CheckResult(
        "B5.alignment", HARD, align_ok,
        "v_fut[0]=v[t], x_fut[0]=x[t+1], v_hist[-1]=v[t]" if align_ok
        else "; ".join(align_detail[:5])
    ))

    # B6: Normalization leakage guard
    norm_detail = ""
    if norm_stats_path and os.path.exists(norm_stats_path):
        h = sha256_file(norm_stats_path)
        norm_detail = f"norm_stats sha256={h[:16]}..."
        # Instantiate with normalization to verify it loads from disk
        try:
            ds_norm = StateSpaceDataset(
                csv_paths=csv_paths, L=L, m=m, H=H,
                predict_deltas=predict_deltas, normalize=True,
                norm_stats_path=norm_stats_path,
            )
            # Check stats are non-trivial
            if hasattr(ds_norm, "v_std") and ds_norm.v_std > 1e-8:
                results.append(CheckResult("B6.norm_from_disk", HARD, True, norm_detail))
            else:
                results.append(CheckResult("B6.norm_from_disk", HARD, False,
                                           "v_std too small or missing"))
        except Exception as e:
            results.append(CheckResult("B6.norm_from_disk", HARD, False, str(e)))
    else:
        results.append(CheckResult(
            "B6.norm_from_disk", SOFT, True,
            "no norm_stats_path provided or file missing; skipped"
        ))

    # B7: dt normalization warning
    if len(ds.runs) > 0:
        all_dt = np.concatenate([r["dt"] for r in ds.runs])
        dt_std = float(np.std(all_dt))
        if dt_std < 1e-4:
            results.append(CheckResult(
                "B7.dt_std_warning", SOFT, True,
                f"dt_std={dt_std:.2e} < 1e-4; normalizing dt may amplify jitter. "
                "Consider skipping dt normalization."
            ))
        else:
            results.append(CheckResult("B7.dt_std_warning", SOFT, True,
                                       f"dt_std={dt_std:.2e}, OK"))

    return results


# ======================================================================
# C) KALMAN FORECASTER NUMERICAL STABILITY
# ======================================================================

def check_kf_stability(csv_paths, L, m, H, predict_deltas, device, seed, B=8):
    """Run a small forward pass and check P symmetry, PSD, S > 0, no NaN."""
    results = []

    # Instantiate dataset (unnormalized -- KF expects physical units)
    try:
        ds = StateSpaceDataset(
            csv_paths=csv_paths, L=L, m=m, H=H,
            predict_deltas=predict_deltas, normalize=False,
        )
    except Exception as e:
        results.append(CheckResult("C0.dataset", HARD, False, str(e)))
        return results

    if len(ds) == 0:
        results.append(CheckResult("C0.dataset", HARD, False, "empty dataset"))
        return results

    # Build a small batch
    np.random.seed(seed)
    torch.manual_seed(seed)
    idxs = np.random.choice(len(ds), size=min(B, len(ds)), replace=False)
    batch = [ds[int(i)] for i in idxs]

    v_hist = torch.stack([b[0] for b in batch]).to(device)
    dt_hist = torch.stack([b[1] for b in batch]).to(device)
    x_hist = torch.stack([b[2] for b in batch]).to(device)
    v_fut = torch.stack([b[3] for b in batch]).to(device)
    dt_fut = torch.stack([b[4] for b in batch]).to(device)

    # For KF: x_obs_hist needs to be [B, L]. x_hist is [B, m].
    # We need to reconstruct [B, L] x from the raw data.
    # Easier: build x_obs_hist directly from runs.
    x_obs_list = []
    for b in batch:
        meta = b[7]
        run = ds.runs[meta["run_idx"]]
        t = meta["t_index"]
        x_obs = run["x"][t - L + 1: t + 1].copy()
        x_obs_list.append(torch.tensor(x_obs, dtype=torch.float32))
    x_obs_hist = torch.stack(x_obs_list).to(device)

    # Instantiate KF with reasonable defaults (use_kappa=True, like v11.1)
    model = KalmanForecaster(
        alpha_init=1.7,
        c_init=25.0,
        vc_init=0.19,
        kappa_init=1.5,
        use_kappa=True,
        log_qx_init=-5.0,
        log_qu_init=-3.5,
        log_r_init=-7.0,
        log_p0_xx_init=-8.0,
        log_p0_uu_init=-4.5,
    ).to(device)
    model.eval()

    # C1: Forward pass -- no NaN/Inf
    try:
        with torch.no_grad():
            x_pred, x_var, u_est = model(v_hist, dt_hist, x_obs_hist, v_fut, dt_fut)
        no_nan = not (torch.any(torch.isnan(x_pred)) or torch.any(torch.isnan(x_var))
                      or torch.any(torch.isnan(u_est)))
        no_inf = not (torch.any(torch.isinf(x_pred)) or torch.any(torch.isinf(x_var))
                      or torch.any(torch.isinf(u_est)))
        results.append(CheckResult(
            "C1.forward_no_nan_inf", HARD, no_nan and no_inf,
            f"x_pred range=[{x_pred.min():.4f}, {x_pred.max():.4f}], "
            f"x_var range=[{x_var.min():.6f}, {x_var.max():.6f}]"
        ))
    except Exception as e:
        results.append(CheckResult("C1.forward_no_nan_inf", HARD, False, str(e)))
        return results

    # C2: Covariance P invariants (run manually step-by-step)
    try:
        with torch.no_grad():
            Bs = v_hist.shape[0]
            s = torch.zeros(Bs, 2, device=device)
            s[:, 0] = x_obs_hist[:, 0]
            P = model.P0.unsqueeze(0).expand(Bs, -1, -1).clone()

            sym_max = 0.0
            min_eig_all = float("inf")
            min_S_all = float("inf")

            # Filter phase
            for k in range(1, L):
                v_k_prev = v_hist[:, k - 1]
                dt_k = dt_hist[:, k].clamp(min=1e-6)
                y_k = x_obs_hist[:, k]
                s, P = model.kf_predict(s, P, v_k_prev, dt_k)

                # Check P symmetry
                asym = torch.abs(P - P.transpose(1, 2)).max().item()
                sym_max = max(sym_max, asym)

                # Check P PSD (min eigenvalue)
                eigs = torch.linalg.eigvalsh(P)
                me = eigs.min().item()
                min_eig_all = min(min_eig_all, me)

                # S = P[0,0] + R
                S_vals = P[:, 0, 0] + model.R
                min_S_all = min(min_S_all, S_vals.min().item())

                s, P = model.kf_update(s, P, y_k)

            # Predict phase
            for k in range(H):
                v_k = v_fut[:, k]
                dt_k = dt_fut[:, k].clamp(min=1e-6)
                s, P = model.kf_predict(s, P, v_k, dt_k)

                asym = torch.abs(P - P.transpose(1, 2)).max().item()
                sym_max = max(sym_max, asym)

                eigs = torch.linalg.eigvalsh(P)
                me = eigs.min().item()
                min_eig_all = min(min_eig_all, me)

        results.append(CheckResult(
            "C2.P_symmetric", HARD, sym_max < 1e-6,
            f"max |P - P^T| = {sym_max:.2e}"
        ))
        results.append(CheckResult(
            "C3.P_psd", HARD, min_eig_all >= -1e-8,
            f"min eigenvalue = {min_eig_all:.2e}"
        ))
        results.append(CheckResult(
            "C4.S_positive", HARD, min_S_all > 0,
            f"min S = {min_S_all:.2e}"
        ))
    except Exception as e:
        results.append(CheckResult("C2-C4.P_invariants", HARD, False, traceback.format_exc()))

    # C5: Log-likelihood finite
    try:
        with torch.no_grad():
            x_pred2, x_var2, _ = model(v_hist, dt_hist, x_obs_hist, v_fut, dt_fut)
            # Reconstruct x_fut_true (unnormalized, absolute)
            x_fut_list = []
            for b in batch:
                meta = b[7]
                run = ds.runs[meta["run_idx"]]
                t = meta["t_index"]
                xf = run["x"][t + 1: t + H + 1].copy()
                x_fut_list.append(torch.tensor(xf, dtype=torch.float32))
            x_fut_true = torch.stack(x_fut_list).to(device)

            var_clamped = x_var2.clamp(min=1e-12)
            nll = 0.5 * torch.log(2 * math.pi * var_clamped) + 0.5 * (x_fut_true - x_pred2) ** 2 / var_clamped
            nll_mean = nll.mean().item()
            nll_finite = math.isfinite(nll_mean)
        results.append(CheckResult(
            "C5.nll_finite", HARD, nll_finite,
            f"mean NLL = {nll_mean:.4f}"
        ))
    except Exception as e:
        results.append(CheckResult("C5.nll_finite", HARD, False, str(e)))

    return results


# ======================================================================
# D) TORCH vs NUMPY MICRO-EQUIVALENCE
# ======================================================================

def check_torch_numpy_equiv(device, steps=50):
    """Run 50 open-loop predict steps in Torch vs Numpy, compare."""
    results = []

    alpha = 1.7
    kappa = 1.5
    c_val = 25.0
    vc = 0.19
    qx = math.exp(-5.0)
    qu = math.exp(-3.5)
    dt = 0.1

    # Generate synthetic forcing
    np.random.seed(42)
    v_series = np.random.randn(steps).astype(np.float64) * 0.3 + 0.2

    # Numpy KF predict (no update, open loop)
    def numpy_kf_predict(x, u, v_k, dt_k, alpha, kappa, c_val, vc):
        rho = math.exp(-alpha * dt_k)
        forcing = max(v_k ** 2 - vc ** 2, 0.0)
        x_new = x + u * dt_k
        u_new = rho * u - kappa * x * dt_k + c_val * forcing * dt_k
        return x_new, u_new

    x_np, u_np = 0.0, 0.0
    traj_np = []
    for k in range(steps):
        x_np, u_np = numpy_kf_predict(x_np, u_np, v_series[k], dt, alpha, kappa, c_val, vc)
        traj_np.append(x_np)
    traj_np = np.array(traj_np)

    # Torch KF predict (using model)
    model = KalmanForecaster(
        alpha_init=alpha, c_init=c_val, vc_init=vc,
        kappa_init=kappa, use_kappa=True,
        log_qx_init=-5.0, log_qu_init=-3.5, log_r_init=-7.0,
    ).to(device)
    model.eval()

    with torch.no_grad():
        s = torch.zeros(1, 2, device=device)
        P = model.P0.unsqueeze(0).clone()
        traj_torch = []
        for k in range(steps):
            v_t = torch.tensor([v_series[k]], device=device, dtype=torch.float32)
            dt_t = torch.tensor([dt], device=device, dtype=torch.float32)
            s, P = model.kf_predict(s, P, v_t, dt_t)
            traj_torch.append(s[0, 0].item())
    traj_torch = np.array(traj_torch)

    max_diff = float(np.max(np.abs(traj_np - traj_torch)))
    mean_diff = float(np.mean(np.abs(traj_np - traj_torch)))

    results.append(CheckResult(
        "D1.torch_numpy_equiv", SOFT, max_diff < 1e-4,
        f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} over {steps} steps"
    ))

    return results


# ======================================================================
# REPORT GENERATION
# ======================================================================

def make_table(results):
    """Format results as a text table."""
    lines = []
    lines.append(f"{'Check':<40} {'Level':<6} {'Status':<6} {'Detail'}")
    lines.append("-" * 100)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = r.detail[:60] if len(r.detail) > 60 else r.detail
        lines.append(f"{r.name:<40} {r.level:<6} {status:<6} {detail}")
    return "\n".join(lines)


def make_md_report(results, per_file_stats, args, runtime_s):
    """Generate markdown report."""
    lines = []
    lines.append("# Sanity Check Report (99)")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Runtime:** {runtime_s:.1f}s")
    lines.append(f"**Device:** {args.device}")
    lines.append(f"**Seed:** {args.seed}")
    lines.append("")

    # Summary
    hard_checks = [r for r in results if r.level == HARD]
    soft_checks = [r for r in results if r.level == SOFT]
    hard_pass = sum(1 for r in hard_checks if r.passed)
    soft_pass = sum(1 for r in soft_checks if r.passed)
    all_hard_pass = hard_pass == len(hard_checks)

    lines.append("## Summary")
    lines.append("")
    status = "PASS" if all_hard_pass else "FAIL"
    lines.append(f"**Overall: {status}**")
    lines.append(f"- HARD: {hard_pass}/{len(hard_checks)} passed")
    lines.append(f"- SOFT: {soft_pass}/{len(soft_checks)} passed")
    lines.append("")

    # Full table
    lines.append("## Check Details")
    lines.append("")
    lines.append("| Check | Level | Status | Detail |")
    lines.append("|-------|-------|--------|--------|")
    for r in results:
        status = "PASS" if r.passed else "**FAIL**"
        detail = r.detail.replace("|", "/")
        lines.append(f"| {r.name} | {r.level} | {status} | {detail} |")
    lines.append("")

    # Per-file stats
    if per_file_stats:
        lines.append("## Per-File Statistics")
        lines.append("")
        lines.append("| File | Rows | v_mean | v_std | x_range | dt_median |")
        lines.append("|------|------|--------|-------|---------|-----------|")
        for s in per_file_stats:
            lines.append(
                f"| {s['file']} | {s['n_rows']} | {s['v_mean']:.4f} | {s['v_std']:.4f} "
                f"| {s['x_range']:.4f} | {s['dt_median']:.6f} |"
            )
        lines.append("")

    # Failed checks (if any)
    failed = [r for r in results if not r.passed]
    if failed:
        lines.append("## Failed Checks")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r.name}** [{r.level}]: {r.detail}")
        lines.append("")

    return "\n".join(lines)


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Sanity checks for state-space pipeline")
    parser.add_argument("--csv_dir", default=str(ROOT / "processed_data_10hz_clean_v1"),
                        help="Directory containing CSV files")
    parser.add_argument("--csv_glob", default=None,
                        help="Glob pattern for CSV files (overrides --csv_dir)")
    parser.add_argument("--L", type=int, default=200, help="Velocity history length")
    parser.add_argument("--m", type=int, default=50, help="Displacement history length")
    parser.add_argument("--H", type=int, default=10, help="Forecast horizon")
    parser.add_argument("--predict_deltas", type=int, default=1, help="1=predict deltas, 0=absolute")
    parser.add_argument("--norm_stats_path", default=None, help="Path to norm stats .npz")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--dt_expected", type=float, default=0.1, help="Expected dt (10Hz=0.1)")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "sanity_checks"),
                        help="Output directory")
    args = parser.parse_args()

    t_start = time.time()

    # Resolve CSV files
    if args.csv_glob:
        import glob
        csv_paths = sorted(glob.glob(args.csv_glob))
    else:
        csv_dir = Path(args.csv_dir)
        csv_paths = sorted(csv_dir.glob("*_10hz_ready.csv"))
        csv_paths = [str(p) for p in csv_paths]

    if not csv_paths:
        print("ERROR: No CSV files found")
        sys.exit(1)

    print(f"Found {len(csv_paths)} CSV files:")
    for p in csv_paths:
        print(f"  {p}")
    print()

    all_results = []
    per_file_stats = []

    # ── A: CSV integrity ──
    print("=" * 60)
    print("A) RAW CSV INTEGRITY")
    print("=" * 60)
    a_results, pf_stats = check_csv_integrity(
        csv_paths, dt_expected=args.dt_expected, seed=args.seed
    )
    all_results.extend(a_results)
    per_file_stats = pf_stats
    for r in a_results:
        print(f"  {r.tag()}  {r.detail}")
    print()

    # ── B: Dataset checks ──
    print("=" * 60)
    print("B) DATASET WINDOW / SLICING")
    print("=" * 60)
    b_results = check_dataset(
        csv_paths, args.L, args.m, args.H,
        bool(args.predict_deltas), args.norm_stats_path,
        args.seed,
    )
    all_results.extend(b_results)
    for r in b_results:
        print(f"  {r.tag()}  {r.detail}")
    print()

    # ── C: KF stability ──
    print("=" * 60)
    print("C) KALMAN FORECASTER STABILITY")
    print("=" * 60)
    c_results = check_kf_stability(
        csv_paths, args.L, args.m, args.H,
        bool(args.predict_deltas), args.device, args.seed,
    )
    all_results.extend(c_results)
    for r in c_results:
        print(f"  {r.tag()}  {r.detail}")
    print()

    # ── D: Torch/Numpy equivalence ──
    print("=" * 60)
    print("D) TORCH vs NUMPY MICRO-EQUIVALENCE")
    print("=" * 60)
    d_results = check_torch_numpy_equiv(args.device)
    all_results.extend(d_results)
    for r in d_results:
        print(f"  {r.tag()}  {r.detail}")
    print()

    # ── Summary ──
    runtime_s = time.time() - t_start
    hard_checks = [r for r in all_results if r.level == HARD]
    soft_checks = [r for r in all_results if r.level == SOFT]
    hard_pass = sum(1 for r in hard_checks if r.passed)
    hard_fail = len(hard_checks) - hard_pass
    soft_pass = sum(1 for r in soft_checks if r.passed)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(make_table(all_results))
    print()
    print(f"HARD: {hard_pass}/{len(hard_checks)} passed, {hard_fail} failed")
    print(f"SOFT: {soft_pass}/{len(soft_checks)} passed")
    overall = "PASS" if hard_fail == 0 else "FAIL"
    print(f"OVERALL: {overall}")
    print(f"Runtime: {runtime_s:.1f}s")

    # ── E: Write outputs ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # report_99.md
    md = make_md_report(all_results, per_file_stats, args, runtime_s)
    (out_dir / "report_99.md").write_text(md, encoding="utf-8")

    # report_99.json
    report_json = {
        "overall": overall,
        "hard_pass": hard_pass,
        "hard_total": len(hard_checks),
        "soft_pass": soft_pass,
        "soft_total": len(soft_checks),
        "runtime_s": runtime_s,
        "checks": [r.to_dict() for r in all_results],
        "per_file_stats": per_file_stats,
    }
    with open(out_dir / "report_99.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    # manifest_99.json
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "file_hashes": {},
    }
    # Hash key source files
    for rel in ["datasets/state_space_dataset.py", "models/kalman_forecaster.py",
                "scripts/99_sanity_checks.py"]:
        full = ROOT / rel
        if full.exists():
            manifest["file_hashes"][rel] = sha256_file(str(full))
    # Hash norm stats if provided
    if args.norm_stats_path and os.path.exists(args.norm_stats_path):
        manifest["file_hashes"]["norm_stats"] = sha256_file(args.norm_stats_path)
    # Hash CSV files
    for p in csv_paths:
        manifest["file_hashes"][Path(p).name] = sha256_file(p)

    with open(out_dir / "manifest_99.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nOutputs written to: {out_dir}")

    sys.exit(0 if hard_fail == 0 else 1)


if __name__ == "__main__":
    main()
