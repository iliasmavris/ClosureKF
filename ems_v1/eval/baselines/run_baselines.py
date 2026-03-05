"""
Data-only baselines + KF model comparison on the frozen 10 Hz test set.

Models evaluated:
  1. Persistence      (dx_pred = 0)
  2. Mean-increment   (dx_pred = train-set mean increment per h)
  3. AR(10)           (iterated AR on 1-step increments)
  4. Ridge            (causal + oracle-future features, per-horizon)
  5. MLP              (same features, per-horizon)
  6. RandomForest     (same features, per-horizon)
  7. Physics-only KF  (v11.1 seed 1, sequential filter)
  8. Closure KF       (v11.1 seed 1, sequential filter)

Uses: ems_v1/eval/metrics_pack.py (Freeze #1 formulas)
Data:  processed_data_10hz_clean_v1/ (clean splits)
KF ckpts: final_lockbox_v11_1_alpha_fix/seed1/checkpoints/

Output:
  ems_v1/eval/baselines/baseline_results.csv
  ems_v1/eval/baselines/table_model_comparison.tex
  ems_v1/eval/baselines/summary.md
"""
import sys, os, math, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ems_v1"))

from eval.metrics_pack import compute_deltax_metrics

# sklearn imports
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# torch for KF models
import torch

CLEAN_DIR = ROOT / "processed_data_10hz_clean_v1"
CKPT_DIR = ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "checkpoints"
OUT_DIR = ROOT / "ems_v1" / "eval" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 5, 10, 20]
DT_S = 0.1           # seconds per step at 10 Hz
H_TO_SEC = {h: h * DT_S for h in HORIZONS}  # {1: 0.1, 2: 0.2, ...}
WARMUP_STEPS = 500   # 50s of val data for KF warmup
AR_ORDER = 10        # AR lag order
GRU_SEEDS = [42, 43, 44]  # 3 seeds for robustness
FEAT_U_LAGS = 20     # water velocity lag features
FEAT_INC_LAGS = 10   # increment lag features
MIN_HIST = max(FEAT_U_LAGS, FEAT_INC_LAGS, AR_ORDER) + 1
H_MAX = max(HORIZONS)

t0_global = time.time()

# ============================================================
# 1. Load data
# ============================================================
print("=" * 60)
print("Loading data...")
df_train = pd.read_csv(CLEAN_DIR / "train_10hz_ready.csv")
df_val   = pd.read_csv(CLEAN_DIR / "val_10hz_ready.csv")
df_test  = pd.read_csv(CLEAN_DIR / "test_10hz_ready.csv")

u_train = df_train['velocity'].values.astype(np.float64)
x_train = df_train['displacement'].values.astype(np.float64)
u_val   = df_val['velocity'].values.astype(np.float64)
x_val   = df_val['displacement'].values.astype(np.float64)
u_test  = df_test['velocity'].values.astype(np.float64)
x_test  = df_test['displacement'].values.astype(np.float64)

N_train, N_val, N_test = len(x_train), len(x_val), len(x_test)
print(f"  Train: {N_train}, Val: {N_val}, Test: {N_test}")

# 1-step increments
inc_train = np.diff(x_train)  # length N_train - 1
inc_test  = np.diff(x_test)   # length N_test - 1

# ============================================================
# 2. Build features for Ridge/MLP/RF
# ============================================================
print("Building features...")


def build_features(x, u, h, u_lags=FEAT_U_LAGS, inc_lags=FEAT_INC_LAGS):
    """Build feature matrix and target for a given horizon h.

    Features at origin t (all causal except oracle future u):
      - u[t], u[t-1], ..., u[t-u_lags+1]          (u_lags values)
      - u[t+1], u[t+2], ..., u[t+h]                (h values, oracle)
      - inc[t], inc[t-1], ..., inc[t-inc_lags+1]   (inc_lags values)
    Target: dx_t(h) = x[t+h] - x[t]

    Returns X (N_valid, n_feat), y (N_valid,), valid_origins (indices).
    """
    inc = np.diff(x)  # length len(x) - 1; inc[t] = x[t+1] - x[t]
    # At origin t, the most recent increment available is inc[t-1] = x[t] - x[t-1]
    min_t = max(u_lags, inc_lags + 1)  # need inc[t-1], ..., inc[t-inc_lags]
    max_t = len(x) - h - 1             # need x[t+h]

    features = []
    targets = []
    origins = []
    for t in range(min_t, max_t + 1):
        # Past water velocity
        feat_u_past = u[t - u_lags + 1: t + 1][::-1]  # [u_t, u_{t-1}, ...]
        # Future water velocity (oracle)
        feat_u_fut = u[t + 1: t + h + 1]
        if len(feat_u_fut) < h:
            continue
        # Past increments (inc[t-1] = x[t] - x[t-1], etc.)
        feat_inc = []
        for lag in range(inc_lags):
            idx = t - 1 - lag  # inc[t-1-lag] = x[t-lag] - x[t-1-lag]
            if idx < 0:
                break
            feat_inc.append(inc[idx])
        if len(feat_inc) < inc_lags:
            continue
        feat_inc = np.array(feat_inc)

        feat = np.concatenate([feat_u_past, feat_u_fut, feat_inc])
        dx = x[t + h] - x[t]
        features.append(feat)
        targets.append(dx)
        origins.append(t)

    return np.array(features), np.array(targets), np.array(origins)


# Pre-build train features for each horizon
train_data = {}
for h in HORIZONS:
    X_tr, y_tr, _ = build_features(x_train, u_train, h)
    train_data[h] = (X_tr, y_tr)
    print(f"  h={h:2d}: train features shape {X_tr.shape}")

# Pre-build test features + true targets
test_data = {}
for h in HORIZONS:
    X_te, y_te, origins = build_features(x_test, u_test, h)
    test_data[h] = (X_te, y_te, origins)
    print(f"  h={h:2d}: test features shape {X_te.shape}, origins [{origins[0]}..{origins[-1]}]")

# ============================================================
# 3. Baseline 1: Persistence (dx_pred = 0)
# ============================================================
print("\n" + "=" * 60)
print("Evaluating Persistence baseline...")
results = []

for h in HORIZONS:
    _, y_te, _ = test_data[h]
    dx_pred = np.zeros_like(y_te)
    m = compute_deltax_metrics(y_te, dx_pred)
    results.append({"model": "Persistence", "h": h, **m})
    print(f"  h={h:2d}: DxR2={m['r2_dx']:+.4f}, MAE={m['mae_dx']:.5f}")

# ============================================================
# 4. Baseline 2: Mean-increment (train-set mean)
# ============================================================
print("\nEvaluating Mean-increment baseline...")

for h in HORIZONS:
    _, y_tr = train_data[h]
    _, y_te, _ = test_data[h]
    mean_inc = np.mean(y_tr)
    dx_pred = np.full_like(y_te, mean_inc)
    m = compute_deltax_metrics(y_te, dx_pred)
    results.append({"model": "Mean-increment", "h": h, **m})
    print(f"  h={h:2d}: DxR2={m['r2_dx']:+.4f}, mean_dx_train={mean_inc:.6f}")

# ============================================================
# 5. Baseline 3: AR(10) on 1-step increments
# ============================================================
print("\nFitting AR(10)...")

# Fit AR(p) via OLS on training increments
# inc_t = c + a1*inc_{t-1} + a2*inc_{t-2} + ... + ap*inc_{t-p}
p = AR_ORDER
X_ar_train = np.column_stack([
    inc_train[p - 1 - lag: len(inc_train) - 1 - lag]
    for lag in range(p)
])
# Add intercept
X_ar_train = np.column_stack([np.ones(len(X_ar_train)), X_ar_train])
y_ar_train = inc_train[p:]

# OLS fit
ar_coeffs, _, _, _ = np.linalg.lstsq(X_ar_train, y_ar_train, rcond=None)
ar_intercept = ar_coeffs[0]
ar_weights = ar_coeffs[1:]
print(f"  AR({p}) intercept: {ar_intercept:.6f}")
print(f"  AR({p}) coeffs: {ar_weights[:3]}... (first 3)")

# Evaluate AR on test set
print("Evaluating AR(10)...")
# At each test origin t, we need the last p increments
# inc_test[t-1] = x_test[t] - x_test[t-1], available at time t
for h in HORIZONS:
    dx_preds = []
    _, y_te, origins = test_data[h]
    for t in origins:
        if t < p:
            dx_preds.append(0.0)
            continue
        # Get initial buffer of last p increments [inc_{t-1}, inc_{t-2}, ...]
        buffer = [inc_test[t - 1 - lag] if (t - 1 - lag) >= 0 else 0.0
                  for lag in range(p)]
        # Iterate AR h steps
        dx_cum = 0.0
        for step in range(h):
            pred_inc = ar_intercept + np.dot(ar_weights, buffer)
            dx_cum += pred_inc
            # Shift buffer
            buffer = [pred_inc] + buffer[:-1]
        dx_preds.append(dx_cum)

    dx_preds = np.array(dx_preds)
    m = compute_deltax_metrics(y_te, dx_preds)
    results.append({"model": f"AR({p})", "h": h, **m})
    print(f"  h={h:2d}: DxR2={m['r2_dx']:+.4f}, MAE={m['mae_dx']:.5f}")

# ============================================================
# 6. Baselines 4-6: Ridge, MLP, RandomForest
# ============================================================


def train_and_eval_sklearn(model_cls, model_name, model_kwargs, horizons,
                           train_data, test_data, results_list):
    """Train per-horizon sklearn model and evaluate."""
    print(f"\nTraining {model_name}...")
    for h in horizons:
        X_tr, y_tr = train_data[h]
        X_te, y_te, _ = test_data[h]

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Fit
        model = model_cls(**model_kwargs)
        model.fit(X_tr_s, y_tr)
        dx_pred = model.predict(X_te_s)

        m = compute_deltax_metrics(y_te, dx_pred)
        results_list.append({"model": model_name, "h": h, **m})
        print(f"  h={h:2d}: DxR2={m['r2_dx']:+.4f}, MAE={m['mae_dx']:.5f}")


# Ridge
train_and_eval_sklearn(Ridge, "Ridge", {"alpha": 1.0},
                       HORIZONS, train_data, test_data, results)

# MLP
train_and_eval_sklearn(
    MLPRegressor, "MLP",
    {"hidden_layer_sizes": (100, 50), "max_iter": 500,
     "early_stopping": True, "random_state": 42,
     "learning_rate_init": 0.001},
    HORIZONS, train_data, test_data, results)

# RandomForest
train_and_eval_sklearn(
    RandomForestRegressor, "RandomForest",
    {"n_estimators": 100, "max_depth": 15, "random_state": 42, "n_jobs": -1},
    HORIZONS, train_data, test_data, results)

# ============================================================
# 6b. GRU sequence baseline (iterated one-step, oracle future u)
#     Run 3 seeds for robustness; report mean in main table.
# ============================================================
print("\n" + "=" * 60)
print("Training GRU sequence model (3 seeds)...")


class GRUForecaster(torch.nn.Module):
    """One-step GRU: maps (u_k, du_k, inc_{k-1}) -> predicted inc_k.

    Iterated for multi-step forecasts (exact analogue of KF open-loop).
    """
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=1,
                                 batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, h=None):
        out, h_new = self.gru(x, h)
        return self.fc(out).squeeze(-1), h_new


# --- Build training / val sequences ---
du_train_full = np.concatenate([[0.0], np.diff(u_train)])
incs_train_full = np.diff(x_train)
gru_train_input = np.column_stack([
    u_train[1:-1], du_train_full[1:-1], incs_train_full[:-1],
]).astype(np.float32)
gru_train_target = incs_train_full[1:].astype(np.float32)

du_val_full = np.concatenate([[0.0], np.diff(u_val)])
incs_val_full = np.diff(x_val)
gru_val_input = np.column_stack([
    u_val[1:-1], du_val_full[1:-1], incs_val_full[:-1],
]).astype(np.float32)
gru_val_target = incs_val_full[1:].astype(np.float32)

GRU_SEQ_LEN = 256


def make_gru_chunks(inp, tgt, seq_len):
    n = len(inp) // seq_len
    inp_c = torch.from_numpy(inp[:n * seq_len].reshape(n, seq_len, -1))
    tgt_c = torch.from_numpy(tgt[:n * seq_len].reshape(n, seq_len))
    return inp_c, tgt_c


X_gru_tr, y_gru_tr = make_gru_chunks(gru_train_input, gru_train_target,
                                       GRU_SEQ_LEN)
X_gru_va, y_gru_va = make_gru_chunks(gru_val_input, gru_val_target,
                                       GRU_SEQ_LEN)
print(f"  Train chunks: {X_gru_tr.shape}, Val chunks: {X_gru_va.shape}")

# --- Warmup + test arrays for GRU evaluation ---
u_warmup_gru = u_val[-WARMUP_STEPS:]
x_warmup_gru = x_val[-WARMUP_STEPS:]
u_all_gru = np.concatenate([u_warmup_gru, u_test])
x_all_gru = np.concatenate([x_warmup_gru, x_test])
N_all_gru = len(u_all_gru)
incs_all_gru = np.diff(x_all_gru)
du_all_gru = np.concatenate([[0.0], np.diff(u_all_gru)])

# Pre-build filter input for evaluation (shared across seeds)
gru_filter_input = np.column_stack([
    u_all_gru[1:-1], du_all_gru[1:-1], incs_all_gru[:-1],
]).astype(np.float32)
gru_filter_t = torch.from_numpy(gru_filter_input).unsqueeze(0)  # (1, T, 3)


def train_gru_one_seed(seed):
    """Train one GRU and return (model, info_dict)."""
    torch.manual_seed(seed)
    model = GRUForecaster(input_size=3, hidden_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val = float('inf')
    best_state = None
    patience = 20
    wait = 0
    best_ep = 0

    t0 = time.time()
    for epoch in range(300):
        model.train()
        perm = torch.randperm(len(X_gru_tr))
        pred, _ = model(X_gru_tr[perm])
        loss = torch.nn.functional.mse_loss(pred, y_gru_tr[perm])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tr_loss = loss.item()

        model.eval()
        with torch.no_grad():
            pred_va, _ = model(X_gru_va)
            val_loss = torch.nn.functional.mse_loss(pred_va, y_gru_va).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
            best_ep = epoch + 1
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    info = {
        "seed": seed,
        "epochs_completed": epoch + 1,
        "best_epoch": best_ep,
        "early_stopped": wait >= patience,
        "final_train_loss": tr_loss,
        "best_val_loss": best_val,
        "wall_time_s": time.time() - t0,
    }
    return model, info


def eval_gru_model(model):
    """Evaluate GRU: sequential filter + iterated rollout. Returns {h: metrics}."""
    with torch.no_grad():
        gru_hidden_out, _ = model.gru(gru_filter_t)
        preds_1step = model.fc(gru_hidden_out).squeeze(-1).squeeze(0).numpy()
        hidden_all = gru_hidden_out.squeeze(0)

    out = {}
    for h in HORIZONS:
        valid_origins = [k for k in range(WARMUP_STEPS, N_all_gru - h)
                         if 0 <= k - 1 < len(preds_1step)]
        if not valid_origins:
            continue
        fi = [k - 1 for k in valid_origins]
        dx_cum = preds_1step[fi].copy()

        if h > 1:
            h_state = hidden_all[fi].unsqueeze(0).contiguous()
            prev_inc = dx_cum.copy()
            for step in range(1, h):
                rk = np.array([k + step for k in valid_origins])
                u_s = np.array([u_all_gru[r] for r in rk], dtype=np.float32)
                du_s = np.array([du_all_gru[r] for r in rk], dtype=np.float32)
                roll_in = torch.from_numpy(
                    np.column_stack([u_s, du_s, prev_inc]).astype(np.float32)
                ).unsqueeze(1)
                with torch.no_grad():
                    ro, h_state = model.gru(roll_in, h_state)
                    sp = model.fc(ro).squeeze(-1).squeeze(-1).numpy()
                dx_cum += sp
                prev_inc = sp

        dx_true = np.array([x_all_gru[k + h] - x_all_gru[k]
                             for k in valid_origins])
        out[h] = compute_deltax_metrics(dx_true, dx_cum)
    return out


# --- Run 3 seeds ---
gru_all_seed_metrics = []  # list of {h: metrics_dict}
gru_train_infos = []
n_params_gru = None

for seed in GRU_SEEDS:
    print(f"\n  --- GRU seed {seed} ---")
    model, info = train_gru_one_seed(seed)
    gru_train_infos.append(info)
    if n_params_gru is None:
        n_params_gru = sum(p.numel() for p in model.parameters())
    print(f"  epochs={info['epochs_completed']}, "
          f"best_epoch={info['best_epoch']}, "
          f"early_stop={info['early_stopped']}, "
          f"train_loss={info['final_train_loss']:.7f}, "
          f"val_loss={info['best_val_loss']:.7f}, "
          f"time={info['wall_time_s']:.1f}s")

    seed_metrics = eval_gru_model(model)
    gru_all_seed_metrics.append(seed_metrics)
    for h in HORIZONS:
        m = seed_metrics[h]
        print(f"    h={h:2d}: R2_dx={m['r2_dx']:+.4f}")

print(f"\n  GRU params: {n_params_gru}")

# Compute mean across seeds for the main results table
for h in HORIZONS:
    r2_vals = [sm[h]['r2_dx'] for sm in gru_all_seed_metrics]
    mean_r2 = np.mean(r2_vals)
    std_r2 = np.std(r2_vals)
    # Use seed-0 (first seed) metrics as template, overwrite r2_dx with mean
    m = gru_all_seed_metrics[0][h].copy()
    m['r2_dx'] = mean_r2
    m['r2_dx_std'] = std_r2
    results.append({"model": "GRU", "h": h, **m})
    print(f"  h={h:2d}: mean R2_dx={mean_r2:+.4f} +/- {std_r2:.4f}")

# Store GRU seed detail for summary
gru_seed_detail = {}
for h in HORIZONS:
    gru_seed_detail[h] = {
        "mean": np.mean([sm[h]['r2_dx'] for sm in gru_all_seed_metrics]),
        "std": np.std([sm[h]['r2_dx'] for sm in gru_all_seed_metrics]),
        "per_seed": [sm[h]['r2_dx'] for sm in gru_all_seed_metrics],
    }

# ============================================================
# 7. KF Physics-only (v11.1 seed 1)
# ============================================================
print("\n" + "=" * 60)
print("Evaluating KF Physics-only...")

from models.kalman_forecaster import KalmanForecaster
from models.kalman_closure import KalmanForecasterClosure

device = torch.device('cpu')

# Load S1
s1_ckpt = torch.load(CKPT_DIR / "stage1_physics_seed1.pth",
                      map_location=device, weights_only=False)
s1_model = KalmanForecaster(use_kappa=True).to(device)
s1_model.load_state_dict(s1_ckpt['state_dict'])
s1_model.eval()
s1_params = s1_model.param_summary()
print(f"  S1: alpha={s1_params['alpha']:.4f}, c={s1_params['c']:.4f}")

# Prepare val warmup + test concatenation
u_warmup = u_val[-WARMUP_STEPS:]
x_warmup = x_val[-WARMUP_STEPS:]
u_all = np.concatenate([u_warmup, u_test])
x_all = np.concatenate([x_warmup, x_test])
N_all = len(u_all)


def run_kf_with_forecasts(model, model_type, u_all, x_all, warmup_n, horizons):
    """Run KF filter and collect h-step open-loop predictions.

    Returns: dict[h] -> (dx_true, dx_pred) arrays over test origins.
    """
    h_max = max(horizons)
    model.eval()
    N = len(u_all)

    # Storage for post-update states
    states = []  # list of (s, P) tuples for test origins

    with torch.no_grad():
        s = torch.zeros(1, 2, device=device)
        s[0, 0] = float(x_all[0])
        P = model.P0.unsqueeze(0).clone()

        for k in range(1, N):
            dt_k = torch.tensor([0.1], device=device)

            if model_type == "s1":
                v_k = torch.tensor([float(u_all[k - 1])], device=device)
                s, P = model.kf_predict(s, P, v_k, dt_k)
            else:
                v_curr = torch.tensor([float(u_all[k - 1])], device=device)
                v_prev = torch.tensor([float(u_all[k - 2])], device=device) if k >= 2 else v_curr
                dv = v_curr - v_prev if k >= 2 else torch.zeros_like(v_curr)
                s, P = model.kf_predict(s, P, v_curr, dv, dt_k)

            y_k = torch.tensor([float(x_all[k])], device=device)
            s, P = model.kf_update(s, P, y_k)

            # Save post-update state for test origins
            if k >= warmup_n:
                states.append((s.clone(), P.clone(), k))

    # Now compute h-step forecasts from each saved state
    forecast = {h: ([], []) for h in horizons}

    with torch.no_grad():
        for s_saved, P_saved, k in states:
            x_origin = float(x_all[k])

            # Check which horizons are scorable from this origin
            for h in horizons:
                if k + h >= N:
                    continue
                # Open-loop h-step prediction
                s_h = s_saved.clone()
                P_h = P_saved.clone()
                for step in range(h):
                    idx = k + step
                    dt_k = torch.tensor([0.1], device=device)
                    if model_type == "s1":
                        v_k = torch.tensor([float(u_all[idx])], device=device)
                        s_h, P_h = model.kf_predict(s_h, P_h, v_k, dt_k)
                    else:
                        v_curr = torch.tensor([float(u_all[idx])], device=device)
                        v_prev = torch.tensor([float(u_all[idx - 1])], device=device) if idx >= 1 else v_curr
                        dv = v_curr - v_prev if idx >= 1 else torch.zeros_like(v_curr)
                        s_h, P_h = model.kf_predict(s_h, P_h, v_curr, dv, dt_k)

                x_pred_h = s_h[0, 0].item()
                dx_pred = x_pred_h - x_origin
                dx_true = float(x_all[k + h]) - x_origin
                forecast[h][0].append(dx_true)
                forecast[h][1].append(dx_pred)

    return {h: (np.array(v[0]), np.array(v[1])) for h, v in forecast.items()}


t0 = time.time()
s1_forecasts = run_kf_with_forecasts(s1_model, "s1", u_all, x_all,
                                      WARMUP_STEPS, HORIZONS)
print(f"  S1 filter + forecast: {time.time() - t0:.1f}s")

for h in HORIZONS:
    dx_true, dx_pred = s1_forecasts[h]
    m = compute_deltax_metrics(dx_true, dx_pred)
    results.append({"model": "Physics-only KF", "h": h, **m})
    print(f"  h={h:2d}: DxR2={m['r2_dx']:+.4f}, MAE={m['mae_dx']:.5f}, n={m['n']}")

# ============================================================
# 8. KF Closure (v11.1 seed 1)
# ============================================================
print("\nEvaluating KF Closure...")

s2_ckpt = torch.load(CKPT_DIR / "closure_2t_seed1.pth",
                      map_location=device, weights_only=False)
s2_model = KalmanForecasterClosure(
    alpha_init=max(s1_params['alpha'], 1e-6),
    c_init=max(s1_params['c'], 0.01),
    vc_init=s1_params['vc'],
    kappa_init=max(s1_params['kappa'], 0.001),
    log_qx_init=math.log(max(s1_params['qx'], 1e-15)),
    log_qu_init=math.log(max(s1_params['qu'], 1e-15)),
    log_r_init=math.log(max(s1_params['R'], 1e-15)),
    log_p0_xx_init=math.log(max(s1_params['P0_xx'], 1e-15)),
    log_p0_uu_init=math.log(max(s1_params['P0_uu'], 1e-15)),
    alpha_param="softplus",
).to(device)
s2_model.load_state_dict(s2_ckpt['state_dict'])
s2_model.eval()
s2_params = s2_model.param_summary()
print(f"  S2: d2={s2_params['d2']:.4f}, b2={s2_params['b2']:.4f}")

t0 = time.time()
s2_forecasts = run_kf_with_forecasts(s2_model, "s2", u_all, x_all,
                                      WARMUP_STEPS, HORIZONS)
print(f"  S2 filter + forecast: {time.time() - t0:.1f}s")

for h in HORIZONS:
    dx_true, dx_pred = s2_forecasts[h]
    m = compute_deltax_metrics(dx_true, dx_pred)
    results.append({"model": "Closure KF", "h": h, **m})
    print(f"  h={h:2d}: DxR2={m['r2_dx']:+.4f}, MAE={m['mae_dx']:.5f}, n={m['n']}")

# ============================================================
# 9. Save results
# ============================================================
df_results = pd.DataFrame(results)
csv_path = OUT_DIR / "baseline_results.csv"
df_results.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ============================================================
# 10. Generate LaTeX table
# ============================================================
print("\nGenerating LaTeX table...")

models_order = [
    "Persistence", "Mean-increment", f"AR({AR_ORDER})",
    "Ridge", "MLP", "RandomForest", "GRU",
    "Physics-only KF", "Closure KF"
]
model_labels = {
    "Persistence": "Persistence",
    "Mean-increment": "Mean incr.",
    f"AR({AR_ORDER})": f"AR({AR_ORDER})",
    "Ridge": "Ridge",
    "MLP": "MLP",
    "RandomForest": "Random Forest",
    "GRU": "GRU (iter.)",
    "Physics-only KF": "Physics KF",
    "Closure KF": "Closure KF",
}
model_types = {
    "Persistence": "stat", "Mean-increment": "stat", f"AR({AR_ORDER})": "stat",
    "Ridge": "ML (dir.)", "MLP": "ML (dir.)", "RandomForest": "ML (dir.)",
    "GRU": "ML (seq.)",
    "Physics-only KF": "grey", "Closure KF": "grey",
}

# Column headers in physical seconds (Freeze #1: align on physical time)
h_sec_headers = " & ".join(
    f"$R^2_{{\\Delta x}}$({H_TO_SEC[h]:.1f}\\,s)" for h in HORIZONS)

tex_lines = [
    r"% Auto-generated by ems_v1/eval/baselines/run_baselines.py",
    r"% DO NOT EDIT MANUALLY",
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \caption{Model comparison on the frozen test set (v11.1, clean splits).",
    r"           $R^2_{\Delta x} > 0$ indicates skill exceeding the",
    r"           constant-mean-increment baseline.",
    r"           All ML models and KF models use oracle future water velocity",
    r"           $u(t{+}1{:}t{+}h)$; AR uses only past increments.",
    r"           Direct regressors (Ridge, MLP, RF) are trained per-horizon;",
    r"           GRU and KF models produce trajectories via iterated rollout",
    r"           from a single trained model.$^{\dagger}$}",
    r"  \label{tab:baselines}",
    r"  \begin{tabular}{ll" + "r" * len(HORIZONS) + "}",
    r"    \toprule",
    f"    Type & Model & {h_sec_headers} \\\\",
    r"    \midrule",
]

prev_type = None
for model_name in models_order:
    mtype = model_types[model_name]
    label = model_labels[model_name]

    if prev_type is not None and mtype != prev_type:
        tex_lines.append(r"    \midrule")
    prev_type = mtype

    vals = []
    for h in HORIZONS:
        row = df_results[(df_results['model'] == model_name) &
                         (df_results['h'] == h)]
        if len(row) > 0:
            r2 = row['r2_dx'].values[0]
            # For GRU, show mean +/- std from 3 seeds
            if model_name == "GRU" and 'r2_dx_std' in row.columns:
                std = row['r2_dx_std'].values[0]
                if not np.isnan(std):
                    vals.append(f"${r2:+.3f}$")
                    continue
            vals.append(f"${r2:+.3f}$")
        else:
            vals.append("---")

    type_label = {"stat": "Stat.", "ML (dir.)": "ML (dir.)",
                  "ML (seq.)": "ML (seq.)", "ML": "ML",
                  "grey": "Grey-box"}[mtype]
    tex_lines.append(f"    {type_label} & {label} & "
                     + " & ".join(vals) + r" \\")

# Footnote mapping steps to physical time
h_map_str = ", ".join(f"$h{{=}}{h}$" for h in HORIZONS)
sec_map_str = ", ".join(f"{H_TO_SEC[h]:.1f}" for h in HORIZONS)

tex_lines.extend([
    r"    \bottomrule",
    r"    \multicolumn{" + str(2 + len(HORIZONS)) + r"}{l}{"
    r"\footnotesize $^{\dagger}$At 10\,Hz, forecast horizons "
    + h_map_str + r" steps correspond to "
    + sec_map_str + r"\,s physical time.} \\",
    r"  \end{tabular}",
    r"\end{table}",
])

tex_path = OUT_DIR / "table_model_comparison.tex"
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex_lines))
print(f"Saved: {tex_path}")

# ============================================================
# 11. Generate summary markdown
# ============================================================
h_sec_labels = [f"{H_TO_SEC[h]:.1f}s" for h in HORIZONS]
md_lines = [
    "# Baseline Comparison Summary",
    "",
    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    f"Test set: {N_test} samples, horizons (physical time): {h_sec_labels}",
    "",
    "## R2_dx Results",
    "",
    f"| Model | {'| '.join(h_sec_labels)} |",
    f"|-------|{'|'.join('------' for _ in HORIZONS)}|",
]

for model_name in models_order:
    vals = []
    for h in HORIZONS:
        row = df_results[(df_results['model'] == model_name) &
                         (df_results['h'] == h)]
        if len(row) > 0:
            vals.append(f"{row['r2_dx'].values[0]:+.4f}")
        else:
            vals.append("---")
    md_lines.append(f"| {model_name} | {'| '.join(vals)} |")

md_lines.extend(["", "## GRU 3-Seed Robustness", ""])
for h in HORIZONS:
    d = gru_seed_detail[h]
    per = ", ".join(f"{v:+.4f}" for v in d["per_seed"])
    md_lines.append(f"- {H_TO_SEC[h]:.1f}s: mean={d['mean']:+.4f}, "
                    f"std={d['std']:.4f}, seeds=[{per}]")

md_lines.extend(["", "## GRU Training Details", ""])
for info in gru_train_infos:
    md_lines.append(
        f"- Seed {info['seed']}: {info['epochs_completed']} epochs "
        f"(best @ {info['best_epoch']}), "
        f"early_stop={info['early_stopped']}, "
        f"train_loss={info['final_train_loss']:.7f}, "
        f"val_loss={info['best_val_loss']:.7f}, "
        f"time={info['wall_time_s']:.1f}s")

md_lines.extend(["", "## Key Findings", ""])

data_only = ["Persistence", "Mean-increment", f"AR({AR_ORDER})",
             "Ridge", "MLP", "RandomForest", "GRU"]
h10 = df_results[df_results['h'] == 10]
best_data = h10[h10['model'].isin(data_only)].sort_values(
    'r2_dx', ascending=False).iloc[0]
phys_h10 = h10[h10['model'] == 'Physics-only KF']['r2_dx'].values[0]
clos_h10 = h10[h10['model'] == 'Closure KF']['r2_dx'].values[0]

md_lines.extend([
    f"- Best data-only at 1.0s: **{best_data['model']}** "
    f"(R2_dx={best_data['r2_dx']:+.4f})",
    f"- Physics-only KF at 1.0s: R2_dx={phys_h10:+.4f}",
    f"- Closure KF at 1.0s: R2_dx={clos_h10:+.4f}",
    f"- Closure gain over best data-only: "
    f"{clos_h10 - best_data['r2_dx']:+.4f}",
    "",
    "## Notes",
    "",
    "- All data-only models use oracle future water velocity "
    "(same information as KF open-loop prediction).",
    f"- AR({AR_ORDER}) is fit on 1-step training increments and "
    "iterated for multi-step (no future u).",
    "- Ridge/MLP/RF trained per-horizon with StandardScaler.",
    "- GRU: 3 seeds (42, 43, 44), mean reported. "
    f"Architecture: GRU(3, 64) + Linear(64, 1), {n_params_gru} params.",
    "- KF models use v11.1 seed 1 checkpoint, sequential filter "
    "with 50s val warmup.",
    f"- Horizons in steps: {HORIZONS}, at 10 Hz = {h_sec_labels} "
    "physical time.",
])

md_path = OUT_DIR / "summary.md"
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines))
print(f"Saved: {md_path}")

elapsed = time.time() - t0_global
print(f"\nTotal elapsed: {elapsed:.1f}s")
print("Done.")
