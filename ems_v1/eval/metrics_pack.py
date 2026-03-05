"""
metrics_pack.py  --  EM&S reusable evaluation helpers (Freeze #1)

Pure numpy/scipy implementation.  No torch, no model code.
Mirrors the exact formulas in ems_v1/meta/METRICS_FREEZE_1.md.

Usage:
    from ems_v1.eval.metrics_pack import (
        compute_level_metrics, compute_deltax_metrics,
        compute_acf, ljung_box, compute_nis, compute_cov90,
        apply_scoring_mask, detect_events, build_event_mask,
    )
"""
import numpy as np
from scipy import stats as sp_stats


# ── Forecast metrics ──────────────────────────────────────────────

def compute_level_metrics(y_true, y_pred):
    """R^2, MAE, RMSE for level (x) forecasts.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (N,)

    Returns
    -------
    dict with keys: r2, mae, rmse
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"r2": float(r2), "mae": mae, "rmse": rmse}


def compute_deltax_metrics(dx_true, dx_pred):
    """R^2_dx, skill_dx, MAE_dx, RMSE_dx for displacement increments.

    Parameters
    ----------
    dx_true, dx_pred : array-like, shape (N,)
        Observed and predicted displacement increments at a single horizon.

    Returns
    -------
    dict with keys: r2_dx, skill_dx, mae_dx, rmse_dx, var_dx_true,
                    rmse_baseline, n
    """
    dx_true = np.asarray(dx_true, dtype=np.float64)
    dx_pred = np.asarray(dx_pred, dtype=np.float64)
    n = len(dx_true)

    err = dx_true - dx_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((dx_true - np.mean(dx_true)) ** 2)
    r2_dx = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    mae_dx = float(np.mean(np.abs(err)))
    rmse_dx = float(np.sqrt(np.mean(err ** 2)))
    var_true = float(np.var(dx_true))
    rmse_base = float(np.sqrt(np.mean((dx_true - np.mean(dx_true)) ** 2)))
    skill_dx = 1.0 - rmse_dx / rmse_base if rmse_base > 1e-15 else 0.0

    return {
        "r2_dx": float(r2_dx),
        "skill_dx": float(skill_dx),
        "mae_dx": mae_dx,
        "rmse_dx": rmse_dx,
        "var_dx_true": var_true,
        "rmse_baseline": rmse_base,
        "n": n,
    }


# ── Innovation diagnostics ────────────────────────────────────────

def compute_acf(e, max_lag=50):
    """Biased autocorrelation function of innovations.

    Denominator = N * Var(e), matching the project convention.

    Parameters
    ----------
    e : array-like, shape (N,)
    max_lag : int

    Returns
    -------
    np.ndarray of shape (max_lag + 1,), with acf[0] = 1.0.
    """
    e = np.asarray(e, dtype=np.float64)
    e_c = e - np.mean(e)
    var = np.var(e)
    n = len(e)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array(
        [np.sum(e_c[:n - l] * e_c[l:]) / (n * var) if l > 0 else 1.0
         for l in range(max_lag + 1)]
    )


def ljung_box(acf_vals, n, lags=(5, 10, 20, 50)):
    """Ljung-Box portmanteau test.

    Parameters
    ----------
    acf_vals : array-like
        ACF values (acf_vals[0] = 1.0 for lag 0).
    n : int
        Number of observations.
    lags : sequence of int
        Lags at which to compute Q statistic.

    Returns
    -------
    list of dicts with keys: lag, Q, p
    """
    acf_vals = np.asarray(acf_vals, dtype=np.float64)
    results = []
    for m in lags:
        if m >= n or m >= len(acf_vals):
            continue
        Q = n * (n + 2) * np.sum(
            acf_vals[1:m + 1] ** 2 / (n - np.arange(1, m + 1))
        )
        p = 1.0 - sp_stats.chi2.cdf(Q, df=m)
        results.append({"lag": int(m), "Q": float(Q), "p": float(p)})
    return results


def compute_nis(e, S):
    """Normalised Innovation Squared: mean(e^2 / S).

    Parameters
    ----------
    e : array-like, shape (N,)
        Innovations.
    S : array-like, shape (N,)
        Innovation variances.

    Returns
    -------
    float
    """
    e = np.asarray(e, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    S = np.maximum(S, 1e-15)
    return float(np.mean(e ** 2 / S))


def compute_cov90(e, S):
    """Empirical 90% predictive coverage.

    Parameters
    ----------
    e : array-like, shape (N,)
        Innovations.
    S : array-like, shape (N,)
        Innovation variances.

    Returns
    -------
    float  (fraction in [0, 1])
    """
    e = np.asarray(e, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    S = np.maximum(S, 1e-15)
    z90 = 1.6449
    return float(np.mean(np.abs(e) <= z90 * np.sqrt(S)))


# ── Scoring mask ──────────────────────────────────────────────────

def apply_scoring_mask(N, eval_start, h):
    """Return boolean mask of scorable origins for horizon h.

    Parameters
    ----------
    N : int
        Total length of evaluation array (warmup + test).
    eval_start : int
        Index of first scored observation.
    h : int
        Forecast horizon in steps.

    Returns
    -------
    np.ndarray of bool, shape (N,)
        True at indices that are valid origins for horizon h.
    """
    mask = np.zeros(N, dtype=bool)
    lo = max(eval_start, 1)
    hi = N - h  # origin i needs i+h to exist
    if hi > lo:
        mask[lo:hi] = True
    return mask


# ── Event detection ───────────────────────────────────────────────

def detect_events(x, min_persist=3):
    """K-means(k=2) on displacement with hysteresis.

    Parameters
    ----------
    x : array-like, shape (N,)
        Displacement time series.
    min_persist : int
        Minimum consecutive steps for a state transition to be accepted.

    Returns
    -------
    event_idx : np.ndarray
        Indices where state transitions occur.
    labels : np.ndarray
        Cleaned binary labels (0 or 1) for each timestep.
    """
    x = np.asarray(x, dtype=np.float64)
    c1, c2 = np.percentile(x, 25), np.percentile(x, 75)

    for _ in range(100):
        labels = (np.abs(x - c2) < np.abs(x - c1)).astype(int)
        if labels.sum() == 0 or labels.sum() == len(x):
            break
        c1_new = np.mean(x[labels == 0])
        c2_new = np.mean(x[labels == 1])
        if abs(c1_new - c1) + abs(c2_new - c2) < 1e-8:
            break
        c1, c2 = c1_new, c2_new

    # Hysteresis
    state = labels[0]
    clean = [state]
    pending_state = None
    pending_count = 0
    for i in range(1, len(labels)):
        if labels[i] != state:
            if pending_state == labels[i]:
                pending_count += 1
            else:
                pending_state = labels[i]
                pending_count = 1
            if pending_count >= min_persist:
                state = pending_state
                pending_state = None
                pending_count = 0
        else:
            pending_state = None
            pending_count = 0
        clean.append(state)

    clean = np.array(clean)
    event_idx = np.where(np.diff(clean) != 0)[0] + 1
    return event_idx, clean


def build_event_mask(N, event_indices, eval_start, radius_steps):
    """Build boolean event mask over scored region.

    Parameters
    ----------
    N : int
        Total length of evaluation array.
    event_indices : array-like
        Indices of detected events (in the full array coordinates).
    eval_start : int
        Start of scored region.
    radius_steps : int
        Number of steps on each side of an event to include.

    Returns
    -------
    event_mask : np.ndarray of bool, shape (N,)
        True within +/- radius of any event.
    """
    event_mask = np.zeros(N, dtype=bool)
    for eidx in event_indices:
        lo = max(eval_start, eidx - radius_steps)
        hi = min(N, eidx + radius_steps + 1)
        event_mask[lo:hi] = True
    return event_mask
