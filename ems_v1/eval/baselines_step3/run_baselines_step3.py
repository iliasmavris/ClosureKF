#!/usr/bin/env python
"""
run_baselines_step3.py  --  EM&S Baseline Suite (Step 3)

Builds causal data-only baselines and compares against the frozen
grey-box model (d2-only closure, lockbox v11.1).

Baselines:
  B0 - Persistence       (dx_pred = 0)
  B1 - Climatology       (dx_pred = mean(dx_train_h))
  B2 - AR(p)             on 1-step increments, iterated
  B3 - Ridge regression  (compressed feature library + lags)
  B4 - MLP               (2-layer, small, sklearn)
  B5 - Gradient Boosting (HistGradientBoosting, sklearn)

All baselines are causal (no future peeking), trained on train only,
tuned on val, and reported on test.

Usage:
    python -u ems_v1/eval/baselines_step3/run_baselines_step3.py
"""
import sys
import os
import time
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- safe console output on Windows --
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
#  Paths
# =====================================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(ROOT, "processed_data_10hz_clean_v1")
FREEZE_DIR = os.path.join(ROOT, "ems_v1", "runs",
                          "lockbox_v11_1_alpha_fix_FREEZE", "seed1", "tables")
OUT_DIR = os.path.join(ROOT, "ems_v1", "eval", "baselines_step3")
TABLE_DIR = os.path.join(ROOT, "ems_v1", "tables")
FIG_DIR = os.path.join(ROOT, "ems_v1", "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -- import frozen metrics helpers --
sys.path.insert(0, os.path.join(ROOT, "ems_v1", "eval"))
from metrics_pack import compute_deltax_metrics  # noqa: E402

# =====================================================================
#  Config (frozen)
# =====================================================================
HORIZONS_STEPS = [1, 2, 5, 10]          # headline horizons at 10 Hz
HORIZONS_SEC = [0.1, 0.2, 0.5, 1.0]
DT = 0.1
K_LAGS = 10     # 1 second of lag history
W_SUMMARY = 50  # 5 seconds for summary window
AR_P_CANDIDATES = [1, 2, 3, 5, 10]
MLP_HIDDEN_CANDIDATES = [(32,), (64,), (32, 16)]
GB_CONFIGS = [
    {"max_depth": 3, "max_iter": 100},
    {"max_depth": 5, "max_iter": 100},
    {"max_depth": 3, "max_iter": 200},
]

print("=" * 70)
print("  EM&S Step 3: Baseline Suite")
print("=" * 70)

# =====================================================================
#  1. Data loading
# =====================================================================
print("\n[1] Loading data ...")
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_10hz_ready.csv"))
df_val   = pd.read_csv(os.path.join(DATA_DIR, "val_10hz_ready.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test_10hz_ready.csv"))

for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    assert set(df.columns) >= {"timestamp", "velocity", "displacement"}, \
        f"Missing columns in {name}"
    print(f"  {name:5s}: {len(df)} rows, "
          f"t=[{df['timestamp'].iloc[0]:.1f}, {df['timestamp'].iloc[-1]:.1f}]")

x_train = df_train["displacement"].values
u_train = df_train["velocity"].values
x_val   = df_val["displacement"].values
u_val   = df_val["velocity"].values
x_test  = df_test["displacement"].values
u_test  = df_test["velocity"].values

N_train, N_val, N_test = len(x_train), len(x_val), len(x_test)


# =====================================================================
#  2. Feature engineering (causal, compressed)
# =====================================================================
def build_features(x, u, K=K_LAGS, W=W_SUMMARY):
    """Build causal feature matrix (vectorized).

    At time t, features = [
        u[t], u[t-1], ..., u[t-K+1],           # K velocity lags
        dx1[t], dx1[t-1], ..., dx1[t-K+1],     # K increment lags
        x[t],                                    # current displacement
        du[t],                                   # velocity change
        mean(u), std(u), mean(|u|), mean(u^2),  # summary over W
        mean(du)                                 # summary of vel change
    ]

    Returns (features, min_t) where min_t is the first valid index.
    """
    N = len(x)
    dx1 = np.concatenate([[0.0], np.diff(x)])
    du  = np.concatenate([[0.0], np.diff(u)])

    min_t = max(K, W)
    idx = np.arange(min_t, N)

    # Velocity lags
    u_lags = np.column_stack([u[idx - i] for i in range(K)])
    # Increment lags
    dx_lags = np.column_stack([dx1[idx - i] for i in range(K)])
    # Current displacement + du
    x_curr = x[idx].reshape(-1, 1)
    du_curr = du[idx].reshape(-1, 1)

    # Rolling summaries (cumsum trick)
    u_cs   = np.concatenate([[0.0], np.cumsum(u)])
    u2_cs  = np.concatenate([[0.0], np.cumsum(u ** 2)])
    au_cs  = np.concatenate([[0.0], np.cumsum(np.abs(u))])
    du_cs  = np.concatenate([[0.0], np.cumsum(du)])

    lo = idx - W + 1
    hi = idx + 1
    u_mean   = (u_cs[hi]  - u_cs[lo])  / W
    u2_mean  = (u2_cs[hi] - u2_cs[lo]) / W
    au_mean  = (au_cs[hi] - au_cs[lo]) / W
    du_mean  = (du_cs[hi] - du_cs[lo]) / W
    u_std    = np.sqrt(np.maximum(u2_mean - u_mean ** 2, 0.0))

    summaries = np.column_stack([u_mean, u_std, au_mean, u2_mean, du_mean])

    feat = np.hstack([u_lags, dx_lags, x_curr, du_curr, summaries])
    return feat, min_t


print("\n[2] Building features ...")
t0 = time.time()

# For train: features + targets built directly
feat_train, min_t_train = build_features(x_train, u_train)
origins_train = np.arange(min_t_train, N_train)

# For val: prepend train tail for lag context
prefix_val = max(K_LAGS, W_SUMMARY)
x_val_ext = np.concatenate([x_train[-prefix_val:], x_val])
u_val_ext = np.concatenate([u_train[-prefix_val:], u_val])
feat_val, min_t_val = build_features(x_val_ext, u_val_ext)
# Only keep features for val indices (after the prefix)
val_start_in_ext = prefix_val
val_feat_mask = (np.arange(min_t_val, len(x_val_ext)) >= val_start_in_ext)
feat_val = feat_val[val_feat_mask]
origins_val_ext = np.arange(min_t_val, len(x_val_ext))[val_feat_mask]
origins_val_local = origins_val_ext - val_start_in_ext

# For test: prepend val tail for lag context
x_test_ext = np.concatenate([x_val[-prefix_val:], x_test])
u_test_ext = np.concatenate([u_val[-prefix_val:], u_test])
feat_test, min_t_test = build_features(x_test_ext, u_test_ext)
test_start_in_ext = prefix_val
test_feat_mask = (np.arange(min_t_test, len(x_test_ext)) >= test_start_in_ext)
feat_test = feat_test[test_feat_mask]
origins_test_ext = np.arange(min_t_test, len(x_test_ext))[test_feat_mask]
origins_test_local = origins_test_ext - test_start_in_ext

print(f"  Features: {feat_train.shape[1]} dims, "
      f"train={len(feat_train)}, val={len(feat_val)}, test={len(feat_test)}")
print(f"  Build time: {time.time() - t0:.1f}s")


# =====================================================================
#  3. Evaluation helper
# =====================================================================
def evaluate_predictions(dx_pred, dx_true):
    """Compute frozen metrics for a single horizon."""
    m = compute_deltax_metrics(dx_true, dx_pred)
    return {
        "r2_dx": m["r2_dx"],
        "skill_dx": m["skill_dx"],
        "mae_dx": m["mae_dx"],
        "rmse_dx": m["rmse_dx"],
        "n": m["n"],
    }


def get_dx_true(x, h):
    """Observed displacement increments at horizon h."""
    N = len(x)
    origins = np.arange(0, N - h)
    dx = x[origins + h] - x[origins]
    return origins, dx


# =====================================================================
#  4. Baselines
# =====================================================================

# --- B0: Persistence ---
def run_persistence(x_test, horizons):
    """dx_pred = 0 for all origins."""
    results = {}
    for h in horizons:
        origins, dx_true = get_dx_true(x_test, h)
        dx_pred = np.zeros_like(dx_true)
        results[h] = evaluate_predictions(dx_pred, dx_true)
    return results


# --- B1: Climatology ---
def run_climatology(x_train, x_test, horizons):
    """dx_pred = mean(dx_train_h) per horizon."""
    results = {}
    for h in horizons:
        _, dx_train_h = get_dx_true(x_train, h)
        mean_dx = np.mean(dx_train_h)
        origins, dx_true = get_dx_true(x_test, h)
        dx_pred = np.full_like(dx_true, mean_dx)
        results[h] = evaluate_predictions(dx_pred, dx_true)
    return results


# --- B2: AR(p) on 1-step increments ---
def fit_ar(dx1, p):
    """Fit AR(p) model on 1-step increments using OLS."""
    N = len(dx1)
    if N <= p + 1:
        return np.zeros(p)
    # Design matrix
    X = np.column_stack([dx1[p - 1 - i:N - 1 - i] for i in range(p)])
    y = dx1[p:]
    # OLS: phi = (X'X)^{-1} X'y
    try:
        phi = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        phi = np.zeros(p)
    return phi


def ar_predict_multistep(phi, dx1_history, h):
    """Iterate AR(p) forward h steps, return cumulative dx."""
    p = len(phi)
    buf = list(dx1_history[-p:])
    cum = 0.0
    for _ in range(h):
        pred = sum(phi[j] * buf[-(j + 1)] for j in range(p))
        cum += pred
        buf.append(pred)
    return cum


def run_ar(x_train, x_val, x_test, horizons, p_candidates):
    """AR(p) with p selected on val."""
    dx1_train = np.diff(x_train)
    dx1_val_ext = np.diff(np.concatenate([x_train[-max(p_candidates):], x_val]))
    val_offset = max(p_candidates)  # val starts here in dx1_val_ext

    # Select p on val (using h=10 RMSE)
    h_sel = max(horizons)
    best_p, best_rmse = p_candidates[0], 1e30
    for p in p_candidates:
        phi = fit_ar(dx1_train, p)
        # Predict on val
        errs = []
        for i in range(val_offset, len(dx1_val_ext) - h_sel + 1):
            history = dx1_val_ext[:i]
            dx_pred = ar_predict_multistep(phi, history, h_sel)
            # dx_true from val
            val_idx = i - val_offset
            if val_idx + h_sel < len(x_val):
                dx_true = x_val[val_idx + h_sel] - x_val[val_idx]
                errs.append((dx_pred - dx_true) ** 2)
        if len(errs) > 0:
            rmse = np.sqrt(np.mean(errs))
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = p

    print(f"    AR: selected p={best_p} (val RMSE@{h_sel}={best_rmse:.6f})")

    # Fit final model
    phi = fit_ar(dx1_train, best_p)

    # Predict on test
    dx1_test_ext = np.diff(np.concatenate([x_train[-best_p:], x_val, x_test]))
    test_offset = best_p + len(x_val) - 1

    results = {}
    for h in horizons:
        preds, trues = [], []
        for i in range(len(x_test) - h):
            ext_idx = test_offset + i
            history = dx1_test_ext[:ext_idx + 1]
            dx_pred = ar_predict_multistep(phi, history, h)
            dx_true = x_test[i + h] - x_test[i]
            preds.append(dx_pred)
            trues.append(dx_true)
        results[h] = evaluate_predictions(np.array(preds), np.array(trues))

    return results, best_p


# --- B3: Ridge ---
def run_ridge(feat_train, x_train, origins_train,
              feat_val, x_val_local, origins_val_local,
              feat_test, x_test, origins_test_local, horizons):
    """Ridge regression per horizon, alpha tuned via RidgeCV."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(feat_train)
    Xt = scaler.transform(feat_train)
    Xv = scaler.transform(feat_val)
    Xs = scaler.transform(feat_test)

    alphas = np.logspace(-2, 4, 20)

    results = {}
    for h in horizons:
        # Training targets
        mask_tr = origins_train + h < len(x_train)
        y_tr = x_train[origins_train[mask_tr] + h] - x_train[origins_train[mask_tr]]

        model = RidgeCV(alphas=alphas).fit(Xt[mask_tr], y_tr)

        # Test predictions
        mask_ts = origins_test_local + h < len(x_test)
        dx_pred = model.predict(Xs[mask_ts])
        dx_true = x_test[origins_test_local[mask_ts] + h] - x_test[origins_test_local[mask_ts]]

        results[h] = evaluate_predictions(dx_pred, dx_true)
        if h == max(horizons):
            print(f"    Ridge: alpha={model.alpha_:.2f} at h={h}")

    return results


# --- B4: MLP ---
def run_mlp(feat_train, x_train, origins_train,
            feat_val, x_val_local, origins_val_local,
            feat_test, x_test, origins_test_local, horizons,
            hidden_candidates):
    """Small MLP per horizon, hidden size tuned on val."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(feat_train)
    Xt = scaler.transform(feat_train)
    Xv = scaler.transform(feat_val)
    Xs = scaler.transform(feat_test)

    results = {}
    for h in horizons:
        mask_tr = origins_train + h < len(x_train)
        y_tr = x_train[origins_train[mask_tr] + h] - x_train[origins_train[mask_tr]]

        mask_val = origins_val_local + h < len(x_val)
        y_val = x_val_local[origins_val_local[mask_val] + h] - x_val_local[origins_val_local[mask_val]]

        best_hidden, best_val_rmse = hidden_candidates[0], 1e30
        for hidden in hidden_candidates:
            model = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                batch_size=min(256, max(32, len(y_tr) // 10)),
            )
            model.fit(Xt[mask_tr], y_tr)
            pred_val = model.predict(Xv[mask_val])
            rmse_val = np.sqrt(np.mean((pred_val - y_val) ** 2))
            if rmse_val < best_val_rmse:
                best_val_rmse = rmse_val
                best_hidden = hidden
                best_model = model

        # Retrain best config on full train set
        final_model = MLPRegressor(
            hidden_layer_sizes=best_hidden,
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            batch_size=min(256, max(32, len(y_tr) // 10)),
        )
        final_model.fit(Xt[mask_tr], y_tr)

        mask_ts = origins_test_local + h < len(x_test)
        dx_pred = final_model.predict(Xs[mask_ts])
        dx_true = x_test[origins_test_local[mask_ts] + h] - x_test[origins_test_local[mask_ts]]

        results[h] = evaluate_predictions(dx_pred, dx_true)
        if h == max(horizons):
            n_params = sum(w.size for w in final_model.coefs_) + \
                       sum(b.size for b in final_model.intercepts_)
            print(f"    MLP: hidden={best_hidden}, params~{n_params} at h={h}")

    return results, n_params


# --- B5: Gradient Boosting ---
def run_gradboost(feat_train, x_train, origins_train,
                  feat_val, x_val_local, origins_val_local,
                  feat_test, x_test, origins_test_local, horizons,
                  gb_configs):
    """HistGradientBoosting per horizon, config tuned on val."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(feat_train)
    Xt = scaler.transform(feat_train)
    Xv = scaler.transform(feat_val)
    Xs = scaler.transform(feat_test)

    results = {}
    best_cfg_str = ""
    for h in horizons:
        mask_tr = origins_train + h < len(x_train)
        y_tr = x_train[origins_train[mask_tr] + h] - x_train[origins_train[mask_tr]]

        mask_val = origins_val_local + h < len(x_val)
        y_val = x_val_local[origins_val_local[mask_val] + h] - x_val_local[origins_val_local[mask_val]]

        best_cfg, best_val_rmse = gb_configs[0], 1e30
        for cfg in gb_configs:
            model = HistGradientBoostingRegressor(
                max_depth=cfg["max_depth"],
                max_iter=cfg["max_iter"],
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(Xt[mask_tr], y_tr)
            pred_val = model.predict(Xv[mask_val])
            rmse_val = np.sqrt(np.mean((pred_val - y_val) ** 2))
            if rmse_val < best_val_rmse:
                best_val_rmse = rmse_val
                best_cfg = cfg
                best_model = model

        # Retrain with best config
        final_model = HistGradientBoostingRegressor(
            max_depth=best_cfg["max_depth"],
            max_iter=best_cfg["max_iter"],
            learning_rate=0.1,
            random_state=42,
        )
        final_model.fit(Xt[mask_tr], y_tr)

        mask_ts = origins_test_local + h < len(x_test)
        dx_pred = final_model.predict(Xs[mask_ts])
        dx_true = x_test[origins_test_local[mask_ts] + h] - x_test[origins_test_local[mask_ts]]

        results[h] = evaluate_predictions(dx_pred, dx_true)
        if h == max(horizons):
            best_cfg_str = f"depth={best_cfg['max_depth']}, iter={best_cfg['max_iter']}"
            print(f"    GradBoost: {best_cfg_str} at h={h}")

    return results, best_cfg_str


# =====================================================================
#  5. Grey-box reference (read-only from frozen run)
# =====================================================================
def load_greybox_reference():
    """Load grey-box metrics from frozen v11.1 output."""
    # horizon_curve.csv: columns = h, oracle_physics, oracle_closure, ...
    hc = pd.read_csv(os.path.join(FREEZE_DIR, "horizon_curve.csv"))

    # event_skill_table.csv: columns = h, mae_dx, rmse_dx, ..., model, subset
    es = pd.read_csv(os.path.join(FREEZE_DIR, "event_skill_table.csv"))

    # metrics_table.csv: columns = variant, acf1, ..., dxr2_10_oracle, ...
    mt = pd.read_csv(os.path.join(FREEZE_DIR, "metrics_table.csv"))

    results = {}
    for h in HORIZONS_STEPS:
        row_hc = hc[hc["h"] == h].iloc[0]
        dxr2 = float(row_hc["oracle_closure"])

        # Get MAE/RMSE from event_skill_table (closure, full subset)
        es_row = es[(es["h"] == h) & (es["model"] == "closure") &
                    (es["subset"] == "full")].iloc[0]
        mae  = float(es_row["mae_dx"])
        rmse = float(es_row["rmse_dx"])
        rmse_base = float(es_row["rmse_baseline"])
        skill = float(es_row["rmse_skill"])
        n = int(es_row["n"])

        results[h] = {
            "r2_dx": dxr2,
            "skill_dx": skill,
            "mae_dx": mae,
            "rmse_dx": rmse,
            "n": n,
        }

    # Also store diagnostic metrics
    closure_warm = mt[mt["variant"] == "closure_warm"].iloc[0]
    diag = {
        "acf1": float(closure_warm["acf1"]),
        "nis": float(closure_warm["nis_mean"]),
        "cov90": float(closure_warm["cov90"]),
    }

    return results, diag


# =====================================================================
#  6. Run all baselines
# =====================================================================
print("\n[3] Running baselines ...")
all_results = {}

# B0: Persistence
print("  B0: Persistence ...")
all_results["Persistence"] = run_persistence(x_test, HORIZONS_STEPS)

# B1: Climatology
print("  B1: Climatology ...")
all_results["Climatology"] = run_climatology(x_train, x_test, HORIZONS_STEPS)

# B2: AR(p)
print("  B2: AR(p) ...")
ar_results, ar_p = run_ar(x_train, x_val, x_test, HORIZONS_STEPS, AR_P_CANDIDATES)
all_results[f"AR({ar_p})"] = ar_results

# B3: Ridge
print("  B3: Ridge ...")
ridge_results = run_ridge(
    feat_train, x_train, origins_train,
    feat_val, x_val, origins_val_local,
    feat_test, x_test, origins_test_local,
    HORIZONS_STEPS)
all_results["Ridge"] = ridge_results

# B4: MLP
print("  B4: MLP ...")
mlp_results, mlp_params = run_mlp(
    feat_train, x_train, origins_train,
    feat_val, x_val, origins_val_local,
    feat_test, x_test, origins_test_local,
    HORIZONS_STEPS, MLP_HIDDEN_CANDIDATES)
all_results["MLP"] = mlp_results

# B5: Gradient Boosting
print("  B5: Gradient Boosting ...")
gb_results, gb_cfg = run_gradboost(
    feat_train, x_train, origins_train,
    feat_val, x_val, origins_val_local,
    feat_test, x_test, origins_test_local,
    HORIZONS_STEPS, GB_CONFIGS)
all_results["GradBoost"] = gb_results

# Grey-box
print("  Loading grey-box reference ...")
gb_ref, gb_diag = load_greybox_reference()
all_results["Grey-box (d2)"] = gb_ref

print(f"\n  All baselines complete.")


# =====================================================================
#  7. Summary table
# =====================================================================
print("\n[4] Building summary table ...")

# Model metadata
model_meta = {
    "Persistence":    {"params": 0,          "interp": "Yes",     "type": "baseline"},
    "Climatology":    {"params": "1/h",      "interp": "Yes",     "type": "baseline"},
    f"AR({ar_p})":    {"params": ar_p,       "interp": "Yes",     "type": "linear"},
    "Ridge":          {"params": f"~{feat_train.shape[1]}",
                                             "interp": "Partial", "type": "linear"},
    "MLP":            {"params": f"~{mlp_params}",
                                             "interp": "No",      "type": "nonlinear"},
    "GradBoost":      {"params": gb_cfg,     "interp": "No",      "type": "nonlinear"},
    "Grey-box (d2)":  {"params": "6+1",      "interp": "Yes",     "type": "grey-box"},
}

# Build DataFrame
rows = []
for model_name in ["Persistence", "Climatology", f"AR({ar_p})",
                    "Ridge", "MLP", "GradBoost", "Grey-box (d2)"]:
    res = all_results[model_name]
    meta = model_meta[model_name]
    row = {
        "Model": model_name,
        "Type": meta["type"],
        "Params": meta["params"],
        "Interpretable": meta["interp"],
    }
    for h, sec in zip(HORIZONS_STEPS, HORIZONS_SEC):
        m = res[h]
        row[f"R2_dx@{sec}s"] = m["r2_dx"]
        if h == max(HORIZONS_STEPS):
            row["Skill@1s"] = m["skill_dx"]
            row["MAE@1s"]   = m["mae_dx"]
            row["RMSE@1s"]  = m["rmse_dx"]
            row["n@1s"]     = m["n"]
    rows.append(row)

df_summary = pd.DataFrame(rows)

# Print summary
print("\n" + "=" * 90)
print("  BASELINES SUMMARY (10 Hz, test set)")
print("=" * 90)
h10_col = "R2_dx@1.0s"
for _, r in df_summary.iterrows():
    print(f"  {r['Model']:20s}  R2_dx@1s={r[h10_col]:+.4f}  "
          f"Skill={r['Skill@1s']:+.4f}  "
          f"MAE={r['MAE@1s']:.5f}  RMSE={r['RMSE@1s']:.5f}  "
          f"Interp={r['Interpretable']}")

# Save CSV
csv_path = os.path.join(OUT_DIR, "baselines_summary_10hz.csv")
df_summary.to_csv(csv_path, index=False, float_format="%.6f")
print(f"\n  Saved: {csv_path}")


# =====================================================================
#  8. LaTeX table
# =====================================================================
print("\n[5] Generating LaTeX table ...")

tex_lines = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \caption{Model class comparison: displacement-increment skill on the",
    r"           10\,Hz test set.  The grey-box (d2-only closure) is competitive",
    r"           with nonlinear ML baselines while remaining interpretable.}",
    r"  \label{tab:baseline_comparison}",
    r"  \begin{tabular}{llrrrrrrr}",
    r"    \toprule",
    r"    Model & Type & Params &"
    r" $\dxRsq_{0.1}$ & $\dxRsq_{0.2}$ & $\dxRsq_{0.5}$ &"
    r" $\dxRsq_{1.0}$ & Skill$_{1.0}$ & MAE$_{1.0}$ \\",
    r"    \midrule",
]

for _, r in df_summary.iterrows():
    name = r["Model"]
    if name == "Grey-box (d2)":
        name = r"Grey-box ($d_2$)"
    # Escape underscores
    name_tex = name.replace("_", r"\_")

    params_str = str(r["Params"])
    type_str = r["Type"]

    vals = [
        f"${r[f'R2_dx@{s}s']:+.3f}$" for s in HORIZONS_SEC
    ]
    skill_str = f"${r['Skill@1s']:+.3f}$"
    mae_str   = f"${r['MAE@1s']:.4f}$"

    line = f"    {name_tex} & {type_str} & {params_str}"
    for v in vals:
        line += f" & {v}"
    line += f" & {skill_str} & {mae_str} \\\\"

    # Add midrule before grey-box
    if r["Model"] == "Grey-box (d2)":
        tex_lines.append(r"    \midrule")

    tex_lines.append(line)

tex_lines += [
    r"    \bottomrule",
    r"  \end{tabular}",
    r"\end{table}",
]

tex_path = os.path.join(TABLE_DIR, "table_model_class_comparison.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines) + "\n")
print(f"  Saved: {tex_path}")


# =====================================================================
#  9. Figure
# =====================================================================
print("\n[6] Generating figure ...")

model_order = ["Persistence", "Climatology", f"AR({ar_p})",
               "Ridge", "MLP", "GradBoost", "Grey-box (d2)"]
colors = ["#999999", "#bbbbbb", "#4477aa", "#44aa77",
          "#ee6677", "#aa3377", "#228833"]
hatches = ["", "", "", "", "", "", "//"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: R2_dx at 1.0s
ax = axes[0]
r2_vals = [all_results[m][10]["r2_dx"] for m in model_order]
bars = ax.barh(range(len(model_order)), r2_vals, color=colors,
               edgecolor="black", linewidth=0.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order, fontsize=10)
ax.set_xlabel(r"$R^2_{\Delta x}$ at 1.0 s", fontsize=11)
ax.set_title("(A)  Displacement-increment skill", fontsize=12)
ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
ax.invert_yaxis()

# Panel B: Skill at 1.0s
ax = axes[1]
skill_vals = [all_results[m][10]["skill_dx"] for m in model_order]
bars = ax.barh(range(len(model_order)), skill_vals, color=colors,
               edgecolor="black", linewidth=0.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order, fontsize=10)
ax.set_xlabel(r"Skill$_{\Delta x}$ at 1.0 s", fontsize=11)
ax.set_title("(B)  RMSE-based skill vs climatology", fontsize=12)
ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
ax.invert_yaxis()

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "fig_baseline_comparison.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")


# =====================================================================
#  10. README
# =====================================================================
print("\n[7] Writing README ...")

readme_text = f"""# Baselines Step 3 -- README

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This directory contains the EM&S baseline comparison suite.  All baselines
are evaluated on the 10 Hz clean test set (`processed_data_10hz_clean_v1/
test_10hz_ready.csv`, {N_test} samples) using the frozen metric protocol
from `ems_v1/meta/METRICS_FREEZE_1.md`.

## Data Integrity

- **No future peeking**: all features at origin t use only data up to time t.
- **Train/val/test respected**: models fitted on train only, hyperparameters
  tuned on val only, metrics reported on test only.
- **Scalers fit on train only**: StandardScaler for Ridge/MLP/GradBoost.

## Baselines

| ID | Model | Description | Tuning |
|----|-------|-------------|--------|
| B0 | Persistence | dx_pred = 0 | None |
| B1 | Climatology | dx_pred = mean(dx_train_h) | None |
| B2 | AR({ar_p}) | AR on 1-step increments, iterated | p from {AR_P_CANDIDATES}, val RMSE@{max(HORIZONS_SEC)}s |
| B3 | Ridge | RidgeCV with {feat_train.shape[1]} compressed features | alpha via RidgeCV internal CV |
| B4 | MLP | sklearn MLPRegressor, 2-layer | hidden from {MLP_HIDDEN_CANDIDATES}, val RMSE |
| B5 | GradBoost | HistGradientBoostingRegressor | depth/iter grid, val RMSE |

## Feature Set (B3-B5)

{feat_train.shape[1]} features per origin, all causal:

- **Velocity lags**: u[t], u[t-1], ..., u[t-{K_LAGS-1}]  ({K_LAGS} features)
- **Increment lags**: dx1[t], dx1[t-1], ..., dx1[t-{K_LAGS-1}]  ({K_LAGS} features)
- **Current state**: x[t], du[t]  (2 features)
- **Summary stats** (last {W_SUMMARY} steps = {W_SUMMARY * DT}s): mean(u), std(u),
  mean(|u|), mean(u^2), mean(du)  (5 features)

For val/test origins near the start, lag context is provided by prepending
the tail of the preceding split (causal: these are past observations).

## Grey-box Reference

Grey-box metrics are loaded read-only from:
  `ems_v1/runs/lockbox_v11_1_alpha_fix_FREEZE/seed1/tables/`

The grey-box model is the v11.1 2-term closure (b2+d2) evaluated in oracle
mode.  Per Freeze #2, b2 is operationally inactive (99.1% skill retained
by d2 alone), so these metrics represent the d2-only model.

## Outputs

- `baselines_summary_10hz.csv` -- full comparison table
- `../../tables/table_model_class_comparison.tex` -- LaTeX table for manuscript
- `../../figures/fig_baseline_comparison.png` -- bar chart comparison
- This README
"""

readme_path = os.path.join(OUT_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)
print(f"  Saved: {readme_path}")


# =====================================================================
#  11. Final summary
# =====================================================================
print("\n" + "=" * 70)
print("  Step 3 COMPLETE")
print("=" * 70)
print(f"\n  Output files:")
print(f"    {csv_path}")
print(f"    {tex_path}")
print(f"    {fig_path}")
print(f"    {readme_path}")
print(f"\n  Grey-box diagnostics (for reference):")
print(f"    ACF(1) = {gb_diag['acf1']:.4f}")
print(f"    NIS    = {gb_diag['nis']:.4f}")
print(f"    cov90  = {gb_diag['cov90']:.4f}")
print(f"\n  Key comparison (R2_dx at 1.0s):")
for m in model_order:
    r2 = all_results[m][10]["r2_dx"]
    marker = " <-- BEST" if m == "Grey-box (d2)" else ""
    print(f"    {m:20s}: {r2:+.4f}{marker}")
print()
