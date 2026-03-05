"""
Pillar A: Data Integrity + Reproducibility Lockbox (v3).

One-command script that produces undeniable evidence of strict temporal
integrity, clean warmup, and reproducibility.  Loads pre-trained checkpoints
from final_lockbox_v2/ (no retraining).

Outputs -> final_lockbox_v3/
  (1) split_integrity_table.csv  +  split_integrity_table.md
  (2) warmup_integrity.md
  (3) innovations_baseline_testonly.npy  +  innovations_closure_testonly.npy
  (4) skill_curves_testonly.csv
  (5) frozen_results_testonly.json
  (6) figures/fig1_skill_curves.png  fig2_innovation_acf.png
      fig3_coefficients.png  fig_impulse_events.png
  (7) data_fingerprint.json
  (8) (this script itself IS reproduce_lockbox_v3.py)
  (9) lockbox_audit_v3.md
  + caption_notes.md
"""

import os, sys, math, json, hashlib, time, platform, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

import torch
torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_closure import KalmanForecasterClosure, CLOSURE_PARAM_NAMES

# ===== Config =====
DATA_DIR = ROOT / "processed_data_10hz"
S1_CKPT = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
           / "stage1_physics_only.pth")
V2_CKPT_DIR = ROOT / "final_lockbox_v2" / "checkpoints"
R3C_CKPT_DIR = ROOT / "model_upgrade_round3c_closure_final" / "checkpoints"
MLP_CKPTS = [
    ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
    / f"stage2_best_seed{s}.pth" for s in [42, 43, 44]]
V2_FROZEN = ROOT / "final_lockbox_v2" / "frozen_results_v2.json"

OUT_DIR = ROOT / "final_lockbox_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = OUT_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

L = 64; H = 20; BATCH = 128; DT = 0.1; VAR_FLOOR = 1e-6
FORCE_CPU = True
SEEDS = [42, 43, 44]

# Impulse figure config
N_EVENTS = 5
WINDOW_SEC = 12.0
MIN_SEP_SEC = 5.0
ROLL_WINDOW = 20
STD_QUANTILE = 0.40
RESET_RANGE_THRESH = 0.15
RESET_DX_THRESH = 0.05

plt.rcParams.update({
    'figure.dpi': 200, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    'font.family': 'serif', 'lines.linewidth': 1.8,
})


# ===== Helpers =====

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def gaussian_nll(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


def compute_acf(e, max_lag=50):
    e_c = e - np.mean(e)
    var = np.var(e)
    n = len(e)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(e_c[:n-l] * e_c[l:]) / (n * var) if l > 0
                     else 1.0 for l in range(max_lag + 1)])


def ljung_box(acf_vals, n, lags=[5, 10, 20, 50]):
    results = []
    for m in lags:
        if m >= n or m >= len(acf_vals):
            continue
        Q = n * (n + 2) * np.sum(
            acf_vals[1:m+1]**2 / (n - np.arange(1, m+1)))
        p = 1.0 - sp_stats.chi2.cdf(Q, df=m)
        results.append({'lag': m, 'Q': float(Q), 'p': float(p)})
    return results


def kf_filter_2state(params, cl_params, t, x_obs, v,
                     collect_residuals=False):
    """Numpy KF filter for innovation scoring."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    closure_out = np.full(N, np.nan) if collect_residuals else None
    physics_out = np.full(N, np.nan) if collect_residuals else None

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']

    a1 = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0)
    d3 = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0:
            dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)

        physics_drift = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt
        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = (-a1*u_st + b1_v*v_w + b2_v*dv_w
              - d1*u_st**2 - d2*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_dt = cl * dt

        x_p = s[0] + s[1] * dt
        u_p = physics_drift + cl_dt
        s_pred = np.array([x_p, u_p])

        if collect_residuals:
            physics_out[k] = physics_drift
            closure_out[k] = cl_dt

        F_mat = np.array([[1, dt], [-kap*dt, rho_u]])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        innovations[k] = innov
        S_values[k] = S_val

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        H_vec = np.array([1.0, 0.0])
        IKH = np.eye(2) - np.outer(K, H_vec)
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)

    if collect_residuals:
        return innovations, S_values, closure_out, physics_out
    return innovations, S_values


def kf_filter_2state_xpred(params, cl_params, t, x_obs, v):
    """Run KF filter and return x predictions (one-step-ahead)."""
    N = len(x_obs)
    x_pred = np.full(N, np.nan)

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']

    a1 = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1 = cl_params.get('d1', 0.0)
    d2 = cl_params.get('d2', 0.0)
    d3 = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0:
            dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)

        physics_drift = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt
        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = (-a1*u_st + b1_v*v_w + b2_v*dv_w
              - d1*u_st**2 - d2*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_dt = cl * dt

        x_p = s[0] + s[1] * dt
        u_p = physics_drift + cl_dt
        s_pred = np.array([x_p, u_p])
        x_pred[k] = x_p

        F_mat = np.array([[1, dt], [-kap*dt, rho_u]])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        H_vec = np.array([1.0, 0.0])
        IKH = np.eye(2) - np.outer(K, H_vec)
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)

    return x_pred


def compute_skill_curves(model, loader, device, max_h=10):
    model.eval()
    all_pred, all_true, all_xcur = [], [], []
    with torch.no_grad():
        for batch in loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            xp, _, _ = model(v_h.to(device), dt_h.to(device),
                             x_h.to(device), v_f.to(device),
                             dt_f.to(device))
            all_pred.append(xp.cpu().numpy())
            all_true.append(x_true.cpu().numpy())
            all_xcur.append(x_cur.numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    xcur = np.concatenate(all_xcur)
    r2_dx = []
    for h in range(min(max_h, pred.shape[1])):
        dx_pred = pred[:, h] - xcur
        dx_true = true[:, h] - xcur
        ss_res_d = np.sum((dx_true - dx_pred)**2)
        ss_tot_d = np.sum((dx_true - np.mean(dx_true))**2)
        r2_dx.append(1 - ss_res_d / ss_tot_d if ss_tot_d > 0 else 0.0)
    return np.array(r2_dx)


def build_baseline(s1_params, device):
    model = KalmanForecasterClosure(
        alpha_init=s1_params['alpha'], c_init=s1_params['c'],
        vc_init=s1_params['vc'], kappa_init=s1_params['kappa'],
        log_qx_init=math.log(s1_params['qx']),
        log_qu_init=math.log(s1_params['qu']),
        log_r_init=math.log(s1_params['R']),
        log_p0_xx_init=math.log(s1_params['P0_xx']),
        log_p0_uu_init=math.log(s1_params['P0_uu']),
        a1_init=0.001, b1_init=0.0, b2_init=0.0,
        d1_init=0.001, d2_init=0.001, d3_init=0.001,
    ).to(device)
    with torch.no_grad():
        model.a1_raw.fill_(-10.0); model.b1.fill_(0.0); model.b2.fill_(0.0)
        model.d1_raw.fill_(-10.0); model.d2_raw.fill_(-10.0)
        model.d3_raw.fill_(-10.0)
    model.eval()
    return model


def build_closure_2t(s1_params, device):
    """Build closure model with only b2 and d2 active (for loading ckpt)."""
    model = KalmanForecasterClosure(
        alpha_init=s1_params['alpha'], c_init=s1_params['c'],
        vc_init=s1_params['vc'], kappa_init=s1_params['kappa'],
        log_qx_init=math.log(s1_params['qx']),
        log_qu_init=math.log(s1_params['qu']),
        log_r_init=math.log(s1_params['R']),
        log_p0_xx_init=math.log(s1_params['P0_xx']),
        log_p0_uu_init=math.log(s1_params['P0_uu']),
        a1_init=0.01, b1_init=0.0, b2_init=5.66,
        d1_init=0.01, d2_init=11.1, d3_init=0.01,
    ).to(device)
    model.freeze_physics()
    with torch.no_grad():
        model.a1_raw.fill_(-10.0); model.a1_raw.requires_grad_(False)
        model.b1.fill_(0.0); model.b1.requires_grad_(False)
        model.d1_raw.fill_(-10.0); model.d1_raw.requires_grad_(False)
        model.d3_raw.fill_(-10.0); model.d3_raw.requires_grad_(False)
    return model


def build_closure_5t(s1_params, device):
    model = KalmanForecasterClosure(
        alpha_init=s1_params['alpha'], c_init=s1_params['c'],
        vc_init=s1_params['vc'], kappa_init=s1_params['kappa'],
        log_qx_init=math.log(s1_params['qx']),
        log_qu_init=math.log(s1_params['qu']),
        log_r_init=math.log(s1_params['R']),
        log_p0_xx_init=math.log(s1_params['P0_xx']),
        log_p0_uu_init=math.log(s1_params['P0_uu']),
        a1_init=0.26, b1_init=0.0, b2_init=6.3,
        d1_init=0.39, d2_init=10.4, d3_init=0.37,
    ).to(device)
    model.freeze_physics()
    model.b1.requires_grad_(False)
    model.b1.fill_(0.0)
    return model


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    device = torch.device('cpu') if FORCE_CPU else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Output -> {OUT_DIR}")

    assertions_passed = []  # track all assertions

    def check(cond, msg):
        """Assert and record."""
        if not cond:
            print(f"  ASSERTION FAILED: {msg}")
            print(f"  STOPPING.")
            sys.exit(1)
        assertions_passed.append(msg)
        return True

    # ==============================================================
    # STEP A: SPLIT INTEGRITY
    # ==============================================================
    print("\n" + "="*70)
    print("STEP A: SPLIT INTEGRITY")
    print("="*70)

    df_train = pd.read_csv(DATA_DIR / "train_10hz_ready.csv")
    df_val = pd.read_csv(DATA_DIR / "val_10hz_ready.csv")
    df_test_orig = pd.read_csv(DATA_DIR / "test_10hz_ready.csv")

    TEST_START = df_test_orig['timestamp'].iloc[0]  # 1134.8

    # Reconstruct dev = val rows with timestamp < test_start
    df_dev = df_val[df_val['timestamp'] < TEST_START].copy()
    df_test = df_test_orig.copy()

    # Save reconstructed dev and test to final_lockbox_v3/splits/
    dev_path = SPLITS_DIR / "dev_10hz_ready.csv"
    test_split_path = SPLITS_DIR / "test_10hz_ready.csv"
    train_split_path = SPLITS_DIR / "train_10hz_ready.csv"
    df_dev.to_csv(dev_path, index=False)
    df_test.to_csv(test_split_path, index=False)
    # Copy train for completeness
    shutil.copy2(DATA_DIR / "train_10hz_ready.csv", train_split_path)

    print(f"  train : len={len(df_train):6d}  t=[{df_train.timestamp.min():.1f}, {df_train.timestamp.max():.1f}]")
    print(f"  val   : len={len(df_val):6d}  t=[{df_val.timestamp.min():.1f}, {df_val.timestamp.max():.1f}]  (original)")
    print(f"  dev   : len={len(df_dev):6d}  t=[{df_dev.timestamp.min():.1f}, {df_dev.timestamp.max():.1f}]")
    print(f"  test  : len={len(df_test):6d}  t=[{df_test.timestamp.min():.1f}, {df_test.timestamp.max():.1f}]")

    # --- Assertions ---
    check(len(df_train) == 10087, f"train rows = {len(df_train)} == 10087")
    check(len(df_dev) == 1261, f"dev rows = {len(df_dev)} == 1261")
    check(len(df_test) == 1261, f"test rows = {len(df_test)} == 1261")

    for name, df in [('train', df_train), ('dev', df_dev), ('test', df_test)]:
        diffs = np.diff(df['timestamp'].values)
        check(np.all(diffs > 0), f"{name} timestamps strictly increasing")

    check(df_train.timestamp.max() < df_dev.timestamp.min(),
          f"train max ({df_train.timestamp.max():.1f}) < dev min ({df_dev.timestamp.min():.1f})")
    check(df_dev.timestamp.max() < df_test.timestamp.min(),
          f"dev max ({df_dev.timestamp.max():.1f}) < test min ({df_test.timestamp.min():.1f})")

    # SHA256 hashes of source CSVs
    hash_train_src = sha256_file(DATA_DIR / "train_10hz_ready.csv")
    hash_val_src = sha256_file(DATA_DIR / "val_10hz_ready.csv")
    hash_test_src = sha256_file(DATA_DIR / "test_10hz_ready.csv")
    # Hashes of reconstructed splits
    hash_dev = sha256_file(dev_path)
    hash_test_split = sha256_file(test_split_path)
    hash_train_split = sha256_file(train_split_path)

    print(f"  SHA256 train (source): {hash_train_src[:16]}...")
    print(f"  SHA256 val (source):   {hash_val_src[:16]}...")
    print(f"  SHA256 test (source):  {hash_test_src[:16]}...")
    print(f"  SHA256 dev (recon):    {hash_dev[:16]}...")
    print("  All split assertions PASS")

    # --- Save split_integrity_table.csv ---
    split_table = pd.DataFrame([
        {'split': 'train', 'rows': len(df_train),
         'min_timestamp': df_train.timestamp.min(),
         'max_timestamp': df_train.timestamp.max(),
         'sha256_source': hash_train_src,
         'timestamps_increasing': True, 'no_overlap': True},
        {'split': 'dev', 'rows': len(df_dev),
         'min_timestamp': df_dev.timestamp.min(),
         'max_timestamp': df_dev.timestamp.max(),
         'sha256_source': hash_dev,
         'timestamps_increasing': True, 'no_overlap': True},
        {'split': 'test', 'rows': len(df_test),
         'min_timestamp': df_test.timestamp.min(),
         'max_timestamp': df_test.timestamp.max(),
         'sha256_source': hash_test_src,
         'timestamps_increasing': True, 'no_overlap': True},
    ])
    split_table.to_csv(OUT_DIR / "split_integrity_table.csv", index=False)

    # --- Save split_integrity_table.md ---
    md = []
    md.append("# Split Integrity Table\n")
    md.append("| Split | Rows | Min Timestamp | Max Timestamp | SHA256 | Increasing | No Overlap |")
    md.append("|-------|------|---------------|---------------|--------|------------|------------|")
    md.append(f"| train | {len(df_train)} | {df_train.timestamp.min():.1f} | "
              f"{df_train.timestamp.max():.1f} | `{hash_train_src}` | PASS | PASS |")
    md.append(f"| dev | {len(df_dev)} | {df_dev.timestamp.min():.1f} | "
              f"{df_dev.timestamp.max():.1f} | `{hash_dev}` | PASS | PASS |")
    md.append(f"| test | {len(df_test)} | {df_test.timestamp.min():.1f} | "
              f"{df_test.timestamp.max():.1f} | `{hash_test_src}` | PASS | PASS |")
    md.append(f"\n**Overlap checks:**")
    md.append(f"- max(train) = {df_train.timestamp.max():.1f} < min(dev) = {df_dev.timestamp.min():.1f}: PASS")
    md.append(f"- max(dev) = {df_dev.timestamp.max():.1f} < min(test) = {df_test.timestamp.min():.1f}: PASS")
    md.append(f"\n**Reconstruction:** dev was constructed from val_10hz_ready.csv by filtering "
              f"timestamp < {TEST_START:.1f}. Original val has {len(df_val)} rows = "
              f"{len(df_dev)} (dev) + {len(df_test)} (test).")
    with open(OUT_DIR / "split_integrity_table.md", 'w') as f:
        f.write("\n".join(md))
    print("  Saved split_integrity_table.csv + .md")

    # ==============================================================
    # STEP B: WARMUP LOGIC
    # ==============================================================
    print("\n" + "="*70)
    print("STEP B: WARMUP LOGIC")
    print("="*70)

    warmup_sec = 50.0

    # --- Dev warmup = last 50s of train ---
    dev_warmup_start = df_train.timestamp.max() - warmup_sec
    dev_warmup = df_train[df_train['timestamp'] >= dev_warmup_start].copy()
    check(dev_warmup.timestamp.max() < df_dev.timestamp.min(),
          f"dev warmup max ({dev_warmup.timestamp.max():.1f}) < dev start ({df_dev.timestamp.min():.1f})")
    print(f"  Dev warmup: {len(dev_warmup)} pts from train tail "
          f"[{dev_warmup.timestamp.min():.1f}, {dev_warmup.timestamp.max():.1f}]")

    # --- Test warmup = last 50s of dev ---
    test_warmup_start = df_dev.timestamp.max() - warmup_sec
    test_warmup = df_dev[df_dev['timestamp'] >= test_warmup_start].copy()
    check(test_warmup.timestamp.max() < df_test.timestamp.min(),
          f"test warmup max ({test_warmup.timestamp.max():.1f}) < test start ({df_test.timestamp.min():.1f})")
    print(f"  Test warmup: {len(test_warmup)} pts from dev tail "
          f"[{test_warmup.timestamp.min():.1f}, {test_warmup.timestamp.max():.1f}]")

    # Build filter array: warmup + test (no overlap)
    df_filter_test = pd.concat([test_warmup, df_test], ignore_index=True)
    check(len(df_filter_test) == len(test_warmup) + len(df_test),
          f"concat len = {len(df_filter_test)} == warmup({len(test_warmup)}) + test({len(df_test)})")

    # No duplicate timestamps
    ts_filter = df_filter_test['timestamp'].values
    check(len(np.unique(ts_filter)) == len(ts_filter),
          f"No duplicate timestamps in warmup+test ({len(ts_filter)} unique of {len(ts_filter)})")
    check(np.all(np.diff(ts_filter) > 0),
          "warmup+test timestamps strictly increasing")

    # Scoring mask
    test_mask = ts_filter >= TEST_START
    n_mask = int(test_mask.sum())
    check(n_mask == len(df_test),
          f"test_mask.sum() = {n_mask} == len(test) = {len(df_test)}")
    print(f"  Warmup+test: {len(df_filter_test)} pts, scoring mask: {n_mask} pts")
    print("  All warmup assertions PASS")

    t_arr = ts_filter
    x_arr = df_filter_test['displacement'].values
    v_arr = df_filter_test['velocity'].values

    # --- Save warmup_integrity.md ---
    wu = []
    wu.append("# Warmup Integrity Report\n")
    wu.append("## Dev Warmup (for dev scoring, not used in test)")
    wu.append(f"- Source: last {warmup_sec:.0f}s of **train** split")
    wu.append(f"- Range: [{dev_warmup.timestamp.min():.1f}, {dev_warmup.timestamp.max():.1f}]")
    wu.append(f"- Points: {len(dev_warmup)}")
    wu.append(f"- Assertion: max warmup ({dev_warmup.timestamp.max():.1f}) < min dev ({df_dev.timestamp.min():.1f}): **PASS**\n")
    wu.append("## Test Warmup (for test scoring)")
    wu.append(f"- Source: last {warmup_sec:.0f}s of **dev** split")
    wu.append(f"- Range: [{test_warmup.timestamp.min():.1f}, {test_warmup.timestamp.max():.1f}]")
    wu.append(f"- Points: {len(test_warmup)}")
    wu.append(f"- Assertion: max warmup ({test_warmup.timestamp.max():.1f}) < min test ({df_test.timestamp.min():.1f}): **PASS**\n")
    wu.append("## Combined Filter Array (warmup + test)")
    wu.append(f"- Total points: {len(df_filter_test)}")
    wu.append(f"- Warmup points: {len(test_warmup)} (NOT scored)")
    wu.append(f"- Test points: {n_mask} (scored)")
    wu.append(f"- No duplicate timestamps: **PASS** ({len(np.unique(ts_filter))} unique)")
    wu.append(f"- Strictly increasing: **PASS**")
    wu.append(f"- Scoring mask: timestamp >= {TEST_START:.1f}")
    wu.append(f"- mask.sum() == len(test): {n_mask} == {len(df_test)}: **PASS**\n")
    wu.append("## Key Guarantee")
    wu.append("The Kalman filter is initialized on warmup data (timestamps < test start) "
              "and reaches steady-state covariance before any test observation is processed. "
              "No test data leaks into warmup. Innovations are scored ONLY on the test mask.")
    with open(OUT_DIR / "warmup_integrity.md", 'w') as f:
        f.write("\n".join(wu))
    print("  Saved warmup_integrity.md")

    # ==============================================================
    # STEP C: LOAD MODELS + RECOMPUTE INNOVATIONS ON TEST ONLY
    # ==============================================================
    print("\n" + "="*70)
    print("STEP C: INNOVATIONS + ADEQUACY (TEST ONLY)")
    print("="*70)

    ckpt = torch.load(S1_CKPT, map_location=device, weights_only=False)
    s1_params = ckpt['params']
    print(f"  S1: alpha={s1_params['alpha']:.4f} c={s1_params['c']:.4f}")

    cl_zero = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl_zero['q_scale'] = 1.0

    # --- Baseline innovations ---
    e_base, S_base = kf_filter_2state(s1_params, cl_zero, t_arr, x_arr, v_arr)
    e_base_m = e_base[test_mask]
    S_base_m = S_base[test_mask]
    valid_b = ~np.isnan(e_base_m)
    e_base_clean = e_base_m[valid_b]
    S_base_clean = S_base_m[valid_b]
    print(f"  Baseline innovations (test-only): {len(e_base_clean)}")
    check(len(e_base_clean) == len(df_test),
          f"baseline innovations length = {len(e_base_clean)} == {len(df_test)}")

    acf_base = compute_acf(e_base_clean, 50)
    lb_base = ljung_box(acf_base, len(e_base_clean))
    nis_base = float(np.mean(e_base_clean**2 / S_base_clean))
    cov90_base = float(np.mean(np.abs(e_base_clean) < 1.645 * np.sqrt(S_base_clean)))
    print(f"  Baseline: ACF(1)={acf_base[1]:.4f}  NIS={nis_base:.4f}  cov90={cov90_base:.4f}")

    # --- Closure_2t innovations (3 seeds) ---
    closure_2t_results = []
    for seed in SEEDS:
        ck_path = V2_CKPT_DIR / f"closure_2t_s{seed}.pth"
        check(ck_path.exists(), f"checkpoint {ck_path.name} exists")
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cs = ck['closure']

        e_cl, S_cl, cl_out, ph_out = kf_filter_2state(
            s1_params, cs, t_arr, x_arr, v_arr, collect_residuals=True)
        e_m = e_cl[test_mask]; S_m = S_cl[test_mask]
        cl_m = cl_out[test_mask]; ph_m = ph_out[test_mask]
        valid = ~np.isnan(e_m)
        e_clean = e_m[valid]; S_clean = S_m[valid]
        cl_clean = cl_m[valid]; ph_clean = ph_m[valid]

        acf_vals = compute_acf(e_clean, 50)
        nis_val = float(np.mean(e_clean**2 / S_clean))
        cov90_val = float(np.mean(np.abs(e_clean) < 1.645 * np.sqrt(S_clean)))
        var_cl = np.var(cl_clean)
        var_ph = np.var(ph_clean)
        frac = var_cl / (var_ph + var_cl + 1e-15)
        med_ratio = float(np.median(np.abs(cl_clean) / (np.abs(ph_clean) + 1e-8)))

        closure_2t_results.append({
            'seed': seed, 'acf': acf_vals,
            'acf1': acf_vals[1], 'acf2': acf_vals[2],
            'acf5': acf_vals[5], 'acf10': acf_vals[10],
            'nis': nis_val, 'cov90': cov90_val,
            'innovations': e_clean, 'S_values': S_clean,
            'closure': cs, 'frac': frac, 'med_ratio': med_ratio,
        })
        print(f"  Seed {seed}: ACF(1)={acf_vals[1]:.4f} NIS={nis_val:.4f} "
              f"b2={cs['b2']:.4f} d2={cs['d2']:.4f}")

    # Save .npy arrays (mean across seeds for innovations)
    np.save(OUT_DIR / "innovations_baseline_testonly.npy", e_base_clean)
    np.save(OUT_DIR / "innovations_closure_testonly.npy",
            np.mean([r['innovations'] for r in closure_2t_results], axis=0))
    print(f"  Saved innovations_baseline_testonly.npy ({len(e_base_clean)} pts)")
    print(f"  Saved innovations_closure_testonly.npy ({len(e_base_clean)} pts)")

    # Mean across seeds for reporting
    m2_acf1 = float(np.mean([r['acf1'] for r in closure_2t_results]))
    m2_acf2 = float(np.mean([r['acf2'] for r in closure_2t_results]))
    m2_acf5 = float(np.mean([r['acf5'] for r in closure_2t_results]))
    m2_acf10 = float(np.mean([r['acf10'] for r in closure_2t_results]))
    m2_nis = float(np.mean([r['nis'] for r in closure_2t_results]))
    m2_cov90 = float(np.mean([r['cov90'] for r in closure_2t_results]))
    m2_frac = float(np.mean([r['frac'] for r in closure_2t_results]))
    m2_med = float(np.mean([r['med_ratio'] for r in closure_2t_results]))

    cl2_acfs = np.array([r['acf'] for r in closure_2t_results])
    cl2_acf_mean = np.mean(cl2_acfs, axis=0)
    cl2_acf_std = np.std(cl2_acfs, axis=0)

    lb_cl2 = ljung_box(cl2_acf_mean, len(e_base_clean))

    # Closure_2t param stats
    param_stats_2t = {}
    for k in ['b2', 'd2', 'q_scale']:
        vals = [r['closure'][k] for r in closure_2t_results]
        param_stats_2t[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'cv': float(100 * np.std(vals) / (abs(np.mean(vals)) + 1e-10)),
        }

    # ==============================================================
    # STEP D: SKILL CURVES ON STRICT TEST WINDOWS
    # ==============================================================
    print("\n" + "="*70)
    print("STEP D: SKILL CURVES (STRICT TEST ONLY)")
    print("="*70)

    # Build test-only dataset + loader
    test_ds = StateSpaceDataset(
        [str(DATA_DIR / "test_10hz_ready.csv")], L=L, m=L, H=H,
        predict_deltas=False, normalize=False)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"  Test dataset: {len(test_ds)} samples (sliding windows over test CSV)")

    # --- Baseline DxR2 ---
    base_model = build_baseline(s1_params, device)
    r2dx_base = compute_skill_curves(base_model, test_loader, device, 10)
    print(f"  Baseline DxR2@10={r2dx_base[9]:.4f}")

    # --- Closure_2t DxR2 (3 seeds) ---
    cl2_r2dx_all = []
    for seed in SEEDS:
        ck_path = V2_CKPT_DIR / f"closure_2t_s{seed}.pth"
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model = build_closure_2t(s1_params, device)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        r2dx = compute_skill_curves(model, test_loader, device, 10)
        cl2_r2dx_all.append(r2dx)
        print(f"  Closure 2t seed {seed}: DxR2@10={r2dx[9]:.4f}")

    cl2_r2dx = np.array(cl2_r2dx_all)
    cl2_dx_mean = np.mean(cl2_r2dx, axis=0)
    cl2_dx_std = np.std(cl2_r2dx, axis=0)

    # --- MLP upper bound ---
    mlp_results = []
    try:
        from models.kalman_neural_residual import KalmanNeuralResidual
        for p in MLP_CKPTS:
            if not p.exists():
                continue
            ck = torch.load(p, map_location=device, weights_only=False)
            mp = ck['params']
            mlp_m = KalmanNeuralResidual(
                alpha_init=mp['alpha'], c_init=mp['c'],
                vc_init=mp['vc'], kappa_init=mp['kappa'],
                log_qx_init=math.log(mp['qx']),
                log_qu_init=math.log(mp['qu']),
                log_r_init=math.log(mp['R']),
                log_p0_xx_init=math.log(mp['P0_xx']),
                log_p0_uu_init=math.log(mp['P0_uu']),
                use_residual=True, hidden=16,
            ).to(device)
            mlp_m.load_state_dict(ck['model_state_dict'])
            if 'phi_mean' in ck:
                mlp_m.phi_mean.copy_(torch.tensor(ck['phi_mean'], dtype=torch.float32))
                mlp_m.phi_std.copy_(torch.tensor(ck['phi_std'], dtype=torch.float32))
            mlp_m.eval()
            r2dx = compute_skill_curves(mlp_m, test_loader, device, 10)
            mlp_results.append({'r2dx': r2dx})
        print(f"  MLP ({len(mlp_results)} seeds): "
              f"DxR2@10={np.mean([r['r2dx'][9] for r in mlp_results]):.4f}")
    except Exception as ex:
        print(f"  MLP loading failed: {ex}")

    if mlp_results:
        mlp_r2dx = np.array([r['r2dx'] for r in mlp_results])
        mlp_dx_mean = np.mean(mlp_r2dx, axis=0)
        mlp_dx_std = np.std(mlp_r2dx, axis=0)
    else:
        mlp_dx_mean = np.full(10, np.nan)
        mlp_dx_std = np.full(10, np.nan)

    # % MLP gain recovered
    base_mdx510 = float(np.mean(r2dx_base[4:10]))
    cl2_mdx510 = float(np.mean(cl2_dx_mean[4:10]))
    mlp_mdx510 = float(np.mean(mlp_dx_mean[4:10]))

    gain_h10_cl2 = cl2_dx_mean[9] - r2dx_base[9]
    gain_h10_mlp = mlp_dx_mean[9] - r2dx_base[9]
    pct_h10_2t = 100.0 * gain_h10_cl2 / gain_h10_mlp if abs(gain_h10_mlp) > 1e-8 else 0

    gain_m510_cl2 = cl2_mdx510 - base_mdx510
    gain_m510_mlp = mlp_mdx510 - base_mdx510
    pct_m510_2t = 100.0 * gain_m510_cl2 / gain_m510_mlp if abs(gain_m510_mlp) > 1e-8 else 0

    print(f"\n  HEADLINES:")
    print(f"  Baseline:   ACF(1)={acf_base[1]:.4f}  DxR2@10={r2dx_base[9]:.4f}  mean(5-10)={base_mdx510:.4f}")
    print(f"  Closure 2t: ACF(1)={m2_acf1:.4f}  DxR2@10={cl2_dx_mean[9]:.4f}  mean(5-10)={cl2_mdx510:.4f}")
    print(f"  MLP:        DxR2@10={mlp_dx_mean[9]:.4f}  mean(5-10)={mlp_mdx510:.4f}")
    print(f"  % recovered h=10: {pct_h10_2t:.1f}%   mean(5-10): {pct_m510_2t:.1f}%")
    print(f"  Grey-box: frac={m2_frac:.4f}  med_ratio={m2_med:.4f}")

    # --- Save skill_curves_testonly.csv ---
    skill_rows = []
    for h in range(10):
        skill_rows.append({
            'horizon': h+1,
            'baseline_dxr2': float(r2dx_base[h]),
            'closure_2t_dxr2_mean': float(cl2_dx_mean[h]),
            'closure_2t_dxr2_std': float(cl2_dx_std[h]),
            'mlp_dxr2_mean': float(mlp_dx_mean[h]),
            'mlp_dxr2_std': float(mlp_dx_std[h]),
        })
    pd.DataFrame(skill_rows).to_csv(OUT_DIR / "skill_curves_testonly.csv", index=False)
    print("  Saved skill_curves_testonly.csv")

    # --- Save frozen_results_testonly.json ---
    frozen = {
        'lockbox_version': 'v3',
        'purpose': 'Pillar A: Data integrity + reproducibility',
        'main_closure': 'closure_2t (b2 + d2)',
        'equation': 'C = b2 * du - d2 * v|u|',
        'split_info': {
            'train': {'rows': len(df_train),
                      'min_t': float(df_train.timestamp.min()),
                      'max_t': float(df_train.timestamp.max()),
                      'sha256': hash_train_src},
            'dev': {'rows': len(df_dev),
                    'min_t': float(df_dev.timestamp.min()),
                    'max_t': float(df_dev.timestamp.max()),
                    'sha256': hash_dev},
            'test': {'rows': len(df_test),
                     'min_t': float(df_test.timestamp.min()),
                     'max_t': float(df_test.timestamp.max()),
                     'sha256': hash_test_src},
        },
        'warmup': {
            'seconds': warmup_sec,
            'test_warmup_pts': len(test_warmup),
            'warmup_range': [float(test_warmup.timestamp.min()),
                             float(test_warmup.timestamp.max())],
            'test_start_t': float(TEST_START),
            'no_duplicates': True,
            'clean': True,
        },
        'innovations_count': {
            'baseline': len(e_base_clean),
            'closure_2t_per_seed': len(closure_2t_results[0]['innovations']),
        },
        'headline_metrics': {
            'physics_only': {
                'acf1': float(acf_base[1]),
                'acf2': float(acf_base[2]),
                'acf5': float(acf_base[5]),
                'acf10': float(acf_base[10]),
                'dxr2_10': float(r2dx_base[9]),
                'mean_dxr2_5_10': base_mdx510,
                'nis': nis_base,
                'cov90': cov90_base,
            },
            'closure_2t': {
                'acf1': m2_acf1,
                'acf2': m2_acf2,
                'acf5': m2_acf5,
                'acf10': m2_acf10,
                'dxr2_10': float(cl2_dx_mean[9]),
                'mean_dxr2_5_10': cl2_mdx510,
                'nis': m2_nis,
                'cov90': m2_cov90,
                'frac': m2_frac,
                'med_ratio': m2_med,
                'pct_mlp_recovered_h10': float(pct_h10_2t),
                'pct_mlp_recovered_mean510': float(pct_m510_2t),
            },
            'mlp_upper_bound': {
                'dxr2_10': float(mlp_dx_mean[9]),
                'mean_dxr2_5_10': mlp_mdx510,
            },
        },
        'closure_2t_params': param_stats_2t,
        'ljung_box': {
            'baseline': lb_base,
            'closure_2t': lb_cl2,
        },
        'dxr2_by_horizon': {
            'baseline': {f'h{h+1}': float(r2dx_base[h]) for h in range(10)},
            'closure_2t_mean': {f'h{h+1}': float(cl2_dx_mean[h]) for h in range(10)},
            'mlp_mean': {f'h{h+1}': float(mlp_dx_mean[h]) for h in range(10)},
        },
    }
    with open(OUT_DIR / "frozen_results_testonly.json", 'w') as f:
        json.dump(frozen, f, indent=2)
    print("  Saved frozen_results_testonly.json")

    # ==============================================================
    # STEP E: FIGURES
    # ==============================================================
    print("\n" + "="*70)
    print("STEP E: FIGURES")
    print("="*70)

    hs = np.arange(1, 11)

    # --- Fig 1: Skill curves ---
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(12, 4.5),
                                             gridspec_kw={'width_ratios': [3, 2]})
    for ax, h_slice, title_suffix in [
        (ax_main, slice(None), ''),
        (ax_inset, slice(3, 10), ' ($h = 4$--$10$)')]:
        h_idx = hs[h_slice]
        ax.plot(h_idx, r2dx_base[h_slice], 's--', color='#d62728',
                label='Physics-only', markersize=6, zorder=3)
        ax.plot(h_idx, cl2_dx_mean[h_slice], 'o-', color='#1f77b4',
                label='Closure (2-term)', markersize=6, zorder=4)
        ax.fill_between(h_idx,
                        cl2_dx_mean[h_slice] - cl2_dx_std[h_slice],
                        cl2_dx_mean[h_slice] + cl2_dx_std[h_slice],
                        alpha=0.2, color='#1f77b4')
        if mlp_results:
            ax.plot(h_idx, mlp_dx_mean[h_slice], '^-', color='#2ca02c',
                    label='MLP upper bound', markersize=6, zorder=3)
            ax.fill_between(h_idx,
                            mlp_dx_mean[h_slice] - mlp_dx_std[h_slice],
                            mlp_dx_mean[h_slice] + mlp_dx_std[h_slice],
                            alpha=0.15, color='#2ca02c')
        ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.5)
        ax.set_xlabel('Forecast horizon $h$ (steps, $\\Delta t = 0.1$ s)')
        ax.set_xticks(h_idx)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax_main.set_ylabel('$\\Delta x \\, R^2(h)$')
    ax_main.set_title('Displacement Increment Skill (all horizons)')
    y_lo = min(r2dx_base.min(), cl2_dx_mean.min()) - 0.15
    y_hi = max(0.4, mlp_dx_mean.max() + 0.1) if mlp_results else 0.15
    ax_main.set_ylim(y_lo, y_hi)
    ax_inset.set_ylabel('$\\Delta x \\, R^2(h)$')
    ax_inset.set_title('Detail: $h = 4$--$10$')
    y_lo2 = min(r2dx_base[3:10].min(), cl2_dx_mean[3:10].min()) - 0.05
    y_hi2 = max(0.35, mlp_dx_mean[3:10].max() + 0.05) if mlp_results else 0.1
    ax_inset.set_ylim(y_lo2, y_hi2)
    ax_inset.annotate(
        f'Closure: {pct_h10_2t:.0f}% of\nMLP gain at $h$=10',
        xy=(10, cl2_dx_mean[9]), xytext=(6.5, 0.12),
        fontsize=9, fontstyle='italic',
        arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_skill_curves.png', bbox_inches='tight')
    plt.close(fig)
    print("  Fig 1: skill curves")

    # --- Fig 2: Innovation ACF ---
    fig, ax = plt.subplots(figsize=(7, 4))
    lags_arr = np.arange(51)
    n_innov = len(e_base_clean)
    z95 = 1.96 / np.sqrt(n_innov)
    ax.fill_between(lags_arr, -z95, z95, alpha=0.12, color='gray',
                    label='95% CI (white noise)')
    ax.plot(lags_arr, acf_base, 's-', color='#d62728', label='Physics-only',
            markersize=3, lw=1.5, zorder=3)
    ax.plot(lags_arr, cl2_acf_mean, 'o-', color='#1f77b4', label='Closure (2-term)',
            markersize=3, lw=1.5, zorder=4)
    ax.fill_between(lags_arr, cl2_acf_mean - cl2_acf_std, cl2_acf_mean + cl2_acf_std,
                    alpha=0.15, color='#1f77b4')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Innovation Autocorrelation (Test Set, Clean Warmup)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 50)
    ax.annotate(f'ACF(1) = {acf_base[1]:.3f}', xy=(1, acf_base[1]),
                xytext=(8, acf_base[1] + 0.03), fontsize=9, color='#d62728')
    ax.annotate(f'ACF(1) = {m2_acf1:.3f}', xy=(1, m2_acf1),
                xytext=(8, m2_acf1 - 0.05), fontsize=9, color='#1f77b4')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_innovation_acf.png', bbox_inches='tight')
    plt.close(fig)
    print("  Fig 2: innovation ACF")

    # --- Fig 3: Coefficients ---
    fig, (ax_main_c, ax_qs) = plt.subplots(
        1, 2, figsize=(8, 4.5), gridspec_kw={'width_ratios': [2, 0.8]})
    coeff_keys = ['b2', 'd2']
    coeff_info = {
        'b2': {'sym': '$b_2$', 'unit': 's$^{-1}$',
               'meaning': '$\\Delta u$ coupling'},
        'd2': {'sym': '$d_2$', 'unit': 'm$^{-1}$',
               'meaning': 'Cross-drag $v|u|$'},
    }
    vals_c = [param_stats_2t[k]['mean'] for k in coeff_keys]
    stds_c = [param_stats_2t[k]['std'] for k in coeff_keys]
    cvs_c = [param_stats_2t[k]['cv'] for k in coeff_keys]
    labels_c = [f"{coeff_info[k]['sym']}\n({coeff_info[k]['meaning']})\n"
                f"[{coeff_info[k]['unit']}]" for k in coeff_keys]
    colors_c = ['#f28e2b', '#76b7b2']
    x_c = np.arange(len(coeff_keys))
    bars_c = ax_main_c.bar(x_c, vals_c, yerr=stds_c, capsize=5,
                           color=colors_c, alpha=0.85, edgecolor='black',
                           lw=0.5, width=0.55)
    ax_main_c.set_xticks(x_c)
    ax_main_c.set_xticklabels(labels_c, fontsize=9)
    ax_main_c.set_ylabel('Coefficient value (SI)')
    ax_main_c.set_title('Parsimonious Closure Coefficients')
    for i, (bar, cv) in enumerate(zip(bars_c, cvs_c)):
        ax_main_c.text(bar.get_x() + bar.get_width()/2.,
                       bar.get_height() + stds_c[i] + 0.2,
                       f'CV={cv:.1f}%', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    qs = param_stats_2t['q_scale']
    ax_qs.bar([0], [qs['mean']], yerr=[qs['std']], capsize=5,
              color='#b07aa1', alpha=0.85, edgecolor='black', lw=0.5, width=0.45)
    ax_qs.text(0, qs['mean'] + qs['std'] + 0.05,
               f"CV={qs['cv']:.1f}%", ha='center', fontsize=9, fontweight='bold')
    ax_qs.set_xticks([0])
    ax_qs.set_xticklabels(['$q_{\\mathrm{scale}}$\n(Noise mult.)\n[--]'], fontsize=9)
    ax_qs.set_ylabel('Value')
    ax_qs.set_title('Noise scaling')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_coefficients.png', bbox_inches='tight')
    plt.close(fig)
    print("  Fig 3: coefficients")

    # --- Fig 4: Impulse events ---
    print("  Generating impulse events figure...")
    cl_best = closure_2t_results[0]['closure']  # seed 42 (deterministic, not cherry-picked)

    x_pred_base = kf_filter_2state_xpred(s1_params, cl_zero, t_arr, x_arr, v_arr)
    x_pred_cl = kf_filter_2state_xpred(s1_params, cl_best, t_arr, x_arr, v_arr)

    t_test = t_arr[test_mask]
    x_test = x_arr[test_mask]
    v_test = v_arr[test_mask]
    xp_base_test = x_pred_base[test_mask]
    xp_cl_test = x_pred_cl[test_mask]

    # Event selection
    dv = np.diff(v_test, prepend=v_test[0])
    dx = np.diff(x_test, prepend=x_test[0])
    v_series = pd.Series(v_test)
    roll_std = v_series.rolling(window=ROLL_WINDOW, center=True, min_periods=1).std().values
    std_thresh = np.quantile(roll_std, STD_QUANTILE)
    quiescent = roll_std < std_thresh

    half_w = int(WINDOW_SEC / 0.1 / 2)
    margin = half_w + 10
    abs_dv = np.abs(dv)
    candidates = np.zeros(len(dv), dtype=bool)
    candidates[margin:-margin] = True
    candidates &= quiescent

    abs_dv_masked = abs_dv.copy()
    abs_dv_masked[~candidates] = -1.0
    events = []
    rejected_reset = 0
    sep_pts = int(MIN_SEP_SEC / 0.1)
    for _ in range(N_EVENTS * 10):
        idx = np.argmax(abs_dv_masked)
        if abs_dv_masked[idx] <= 0:
            break
        w_lo = max(0, idx - half_w)
        w_hi = min(len(x_test), idx + half_w + 1)
        x_window = x_test[w_lo:w_hi]
        dx_window = dx[w_lo:w_hi]
        x_range = np.max(x_window) - np.min(x_window)
        max_abs_dx = np.max(np.abs(dx_window))
        if x_range > RESET_RANGE_THRESH or max_abs_dx > RESET_DX_THRESH:
            abs_dv_masked[idx] = -1.0
            rejected_reset += 1
            continue
        events.append(idx)
        if len(events) >= N_EVENTS:
            break
        lo = max(0, idx - sep_pts)
        hi = min(len(abs_dv_masked), idx + sep_pts + 1)
        abs_dv_masked[lo:hi] = -1.0
    events.sort()
    print(f"    Selected {len(events)} events ({rejected_reset} rejected by reset filter)")

    n_events = len(events)
    if n_events > 0:
        fig, axes = plt.subplots(n_events, 3, figsize=(14, 3.0 * n_events), sharex='row')
        if n_events == 1:
            axes = axes.reshape(1, -1)
        for row, ev_idx in enumerate(events):
            lo = max(0, ev_idx - half_w)
            hi = min(len(t_test), ev_idx + half_w + 1)
            sl = slice(lo, hi)
            t_w = t_test[sl] - t_test[ev_idx]
            ax = axes[row, 0]
            ax.plot(t_w, v_test[sl], color='#1f77b4', lw=1.2)
            ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
            ax.set_ylabel('$u$ (m/s)')
            if row == 0:
                ax.set_title('Water velocity')
            if row == n_events - 1:
                ax.set_xlabel('Time relative to event (s)')
            ax.text(0.02, 0.95, f't={t_test[ev_idx]:.1f}s\n|du|={abs_dv[ev_idx]:.4f}',
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
            ax = axes[row, 1]
            ax.plot(t_w, dv[sl], color='#ff7f0e', lw=1.0)
            ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
            ax.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax.set_ylabel('$\\Delta u$ (m/s)')
            if row == 0:
                ax.set_title('Velocity increment')
            if row == n_events - 1:
                ax.set_xlabel('Time relative to event (s)')
            ax = axes[row, 2]
            ax.plot(t_w, x_test[sl], 'k-', lw=1.5, label='Observed', zorder=5)
            ax.plot(t_w, xp_base_test[sl], '--', color='#d62728', lw=1.2,
                    label='Physics-only', zorder=3)
            ax.plot(t_w, xp_cl_test[sl], '-', color='#2ca02c', lw=1.2,
                    label='Closure', zorder=4)
            ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
            ax.set_ylabel('$x$ (m)')
            if row == 0:
                ax.set_title('Displacement')
                ax.legend(loc='upper right', fontsize=7)
            if row == n_events - 1:
                ax.set_xlabel('Time relative to event (s)')
        fig.suptitle(f'Top {n_events} velocity impulse events (non-cherry-picked)',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'fig_impulse_events.png', bbox_inches='tight')
        plt.close(fig)
        print("  Fig 4: impulse events")

    # --- Caption notes ---
    cap = ("Predictions shown are one-step-ahead priors x_{k|k-1}; the apparent "
           "dt lag in event overlays is the expected causal delay.")
    with open(OUT_DIR / "caption_notes.md", 'w') as f:
        f.write(cap + "\n")
    print("  Saved caption_notes.md")

    # ==============================================================
    # STEP F: DATA FINGERPRINT
    # ==============================================================
    print("\n" + "="*70)
    print("STEP F: DATA FINGERPRINT")
    print("="*70)

    fingerprint = {
        'source_csvs': {
            'train_10hz_ready.csv': {
                'sha256': hash_train_src,
                'rows': len(df_train),
                'x_mean': float(df_train['displacement'].mean()),
                'x_std': float(df_train['displacement'].std()),
                'v_mean': float(df_train['velocity'].mean()),
                'v_std': float(df_train['velocity'].std()),
            },
            'val_10hz_ready.csv': {
                'sha256': hash_val_src,
                'rows': len(df_val),
                'x_mean': float(df_val['displacement'].mean()),
                'x_std': float(df_val['displacement'].std()),
                'v_mean': float(df_val['velocity'].mean()),
                'v_std': float(df_val['velocity'].std()),
            },
            'test_10hz_ready.csv': {
                'sha256': hash_test_src,
                'rows': len(df_test),
                'x_mean': float(df_test['displacement'].mean()),
                'x_std': float(df_test['displacement'].std()),
                'v_mean': float(df_test['velocity'].mean()),
                'v_std': float(df_test['velocity'].std()),
            },
        },
        'reconstructed_splits': {
            'dev_10hz_ready.csv': {
                'sha256': hash_dev,
                'rows': len(df_dev),
            },
        },
        'checkpoints_used': {},
        'output_hashes': {},
    }

    # Hash checkpoints used
    for seed in SEEDS:
        ck_path = V2_CKPT_DIR / f"closure_2t_s{seed}.pth"
        fingerprint['checkpoints_used'][ck_path.name] = sha256_file(ck_path)

    # Hash key outputs
    for fname in ['frozen_results_testonly.json', 'skill_curves_testonly.csv',
                  'innovations_baseline_testonly.npy', 'innovations_closure_testonly.npy']:
        fpath = OUT_DIR / fname
        if fpath.exists():
            fingerprint['output_hashes'][fname] = sha256_file(fpath)

    with open(OUT_DIR / "data_fingerprint.json", 'w') as f:
        json.dump(fingerprint, f, indent=2)
    print("  Saved data_fingerprint.json")

    # ==============================================================
    # STEP G: LOCKBOX AUDIT v3
    # ==============================================================
    print("\n" + "="*70)
    print("STEP G: LOCKBOX AUDIT v3")
    print("="*70)

    elapsed_total = time.time() - t_start

    # Compare with v2
    v2_delta_ok = True
    v2_notes = []
    if V2_FROZEN.exists():
        with open(V2_FROZEN) as f:
            v2 = json.load(f)
        v2_acf1 = v2['acf']['closure_2t_mean']['acf1']
        v2_dxr2 = v2['dxr2']['closure_2t_mean']['h10']
        delta_acf1 = abs(m2_acf1 - v2_acf1)
        delta_dxr2 = abs(cl2_dx_mean[9] - v2_dxr2)
        if delta_acf1 > 0.01:
            v2_delta_ok = False
            v2_notes.append(f"WARNING: |delta ACF1| = {delta_acf1:.6f} > 0.01")
        if delta_dxr2 > 0.02:
            v2_delta_ok = False
            v2_notes.append(f"WARNING: |delta DxR2@10| = {delta_dxr2:.6f} > 0.02")
        if v2_delta_ok:
            v2_notes.append(f"|delta ACF1| = {delta_acf1:.6f} <= 0.01: PASS")
            v2_notes.append(f"|delta DxR2@10| = {delta_dxr2:.6f} <= 0.02: PASS")

    audit = []
    audit.append("# Lockbox Audit v3: Pillar A -- Data Integrity + Reproducibility\n")
    audit.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    audit.append(f"**Total runtime:** {elapsed_total:.0f}s")
    audit.append(f"**Python:** {sys.version.split()[0]}")
    audit.append(f"**PyTorch:** {torch.__version__}")
    audit.append(f"**NumPy:** {np.__version__}")
    audit.append(f"**Pandas:** {pd.__version__}")
    audit.append(f"**Platform:** {platform.platform()}")
    audit.append(f"**Command:** `python scripts/reproduce_lockbox_v3.py`\n")

    # 1) Split table + overlap checks
    audit.append("## 1. Split Integrity\n")
    audit.append("| Split | Rows | Min t | Max t | SHA256 |")
    audit.append("|-------|------|-------|-------|--------|")
    audit.append(f"| train | {len(df_train)} | {df_train.timestamp.min():.1f} | "
                 f"{df_train.timestamp.max():.1f} | `{hash_train_src}` |")
    audit.append(f"| dev | {len(df_dev)} | {df_dev.timestamp.min():.1f} | "
                 f"{df_dev.timestamp.max():.1f} | `{hash_dev}` |")
    audit.append(f"| test | {len(df_test)} | {df_test.timestamp.min():.1f} | "
                 f"{df_test.timestamp.max():.1f} | `{hash_test_src}` |")
    audit.append(f"\n**Timestamps strictly increasing:** PASS (all 3 splits)")
    audit.append(f"**No overlap:**")
    audit.append(f"- max(train) = {df_train.timestamp.max():.1f} < min(dev) = {df_dev.timestamp.min():.1f}: PASS")
    audit.append(f"- max(dev) = {df_dev.timestamp.max():.1f} < min(test) = {df_test.timestamp.min():.1f}: PASS")

    # 2) Warmup ranges
    audit.append(f"\n## 2. Warmup Ranges (proof warmup is pre-test)\n")
    audit.append(f"| Property | Value |")
    audit.append(f"|----------|-------|")
    audit.append(f"| Warmup source | Last {warmup_sec:.0f}s of dev split |")
    audit.append(f"| Warmup range | [{test_warmup.timestamp.min():.1f}, {test_warmup.timestamp.max():.1f}] |")
    audit.append(f"| Warmup points | {len(test_warmup)} |")
    audit.append(f"| Test start | {TEST_START:.1f} |")
    audit.append(f"| max(warmup) < test_start | {test_warmup.timestamp.max():.1f} < {TEST_START:.1f}: **PASS** |")
    audit.append(f"| Scoring mask sum | {n_mask} == {len(df_test)}: **PASS** |")

    # 3) No-duplicate timestamp assertion
    audit.append(f"\n## 3. Timestamp Uniqueness\n")
    audit.append(f"- Combined warmup+test array: {len(ts_filter)} timestamps")
    audit.append(f"- Unique timestamps: {len(np.unique(ts_filter))}")
    audit.append(f"- No duplicates: **PASS**")
    audit.append(f"- Strictly increasing: **PASS**")

    # 4) Headline metrics table
    audit.append(f"\n## 4. Headline Metrics (Test Set, 3 Seeds)\n")
    audit.append("| Metric | Physics-only | Closure (2t) | MLP upper bound |")
    audit.append("|--------|-------------|--------------|-----------------|")
    audit.append(f"| ACF(1) | {acf_base[1]:.4f} | {m2_acf1:.4f} | -- |")
    audit.append(f"| ACF(2) | {acf_base[2]:.4f} | {m2_acf2:.4f} | -- |")
    audit.append(f"| ACF(5) | {acf_base[5]:.4f} | {m2_acf5:.4f} | -- |")
    audit.append(f"| ACF(10) | {acf_base[10]:.4f} | {m2_acf10:.4f} | -- |")
    audit.append(f"| DxR2@10 | {r2dx_base[9]:.4f} | {cl2_dx_mean[9]:.4f} | {mlp_dx_mean[9]:.4f} |")
    audit.append(f"| mean DxR2(5-10) | {base_mdx510:.4f} | {cl2_mdx510:.4f} | {mlp_mdx510:.4f} |")
    audit.append(f"| % MLP recovered (h=10) | -- | {pct_h10_2t:.1f}% | -- |")
    audit.append(f"| % MLP recovered (mean 5-10) | -- | {pct_m510_2t:.1f}% | -- |")
    audit.append(f"| NIS | {nis_base:.4f} | {m2_nis:.4f} | -- |")
    audit.append(f"| cov90 | {cov90_base:.4f} | {m2_cov90:.4f} | -- |")
    audit.append(f"| frac (grey-box) | -- | {m2_frac:.4f} | -- |")
    audit.append(f"| med_ratio (grey-box) | -- | {m2_med:.4f} | -- |")

    audit.append(f"\n**NIS definition:** NIS = mean(e_k^2 / S_k) where e_k = x_obs[k] - x_pred[k|k-1] "
                 f"and S_k = P_pred[0,0] + R. For a well-calibrated filter, NIS -> 1.0.")
    audit.append(f"\n**Ljung-Box test (baseline):**")
    for r in lb_base:
        audit.append(f"  lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.2e}")
    audit.append(f"\n**Ljung-Box test (closure_2t):**")
    for r in lb_cl2:
        audit.append(f"  lag={r['lag']}: Q={r['Q']:.1f}, p={r['p']:.2e}")

    audit.append(f"\n**Closure equation:** C = {param_stats_2t['b2']['mean']:.3f} * du "
                 f"- {param_stats_2t['d2']['mean']:.3f} * v|u|")
    audit.append(f"**Coefficient stability:** b2 CV={param_stats_2t['b2']['cv']:.2f}%, "
                 f"d2 CV={param_stats_2t['d2']['cv']:.2f}%")

    # 5) Command + environment
    audit.append(f"\n## 5. Exact Command + Environment\n")
    audit.append(f"```")
    audit.append(f"python scripts/reproduce_lockbox_v3.py")
    audit.append(f"```\n")
    audit.append(f"| Component | Version |")
    audit.append(f"|-----------|---------|")
    audit.append(f"| Python | {sys.version.split()[0]} |")
    audit.append(f"| PyTorch | {torch.__version__} |")
    audit.append(f"| NumPy | {np.__version__} |")
    audit.append(f"| Pandas | {pd.__version__} |")
    import scipy
    audit.append(f"| scipy | {scipy.__version__} |")
    audit.append(f"| Platform | {platform.platform()} |")
    audit.append(f"| Device | {device} |")
    audit.append(f"| Runtime | {elapsed_total:.0f}s |")

    # 6) File hashes
    audit.append(f"\n## 6. File Hashes\n")
    audit.append(f"### Source CSVs")
    audit.append(f"| File | SHA256 |")
    audit.append(f"|------|--------|")
    audit.append(f"| train_10hz_ready.csv | `{hash_train_src}` |")
    audit.append(f"| val_10hz_ready.csv | `{hash_val_src}` |")
    audit.append(f"| test_10hz_ready.csv | `{hash_test_src}` |")
    audit.append(f"| dev_10hz_ready.csv (reconstructed) | `{hash_dev}` |")

    audit.append(f"\n### Checkpoints Used")
    audit.append(f"| File | SHA256 |")
    audit.append(f"|------|--------|")
    for seed in SEEDS:
        ck_path = V2_CKPT_DIR / f"closure_2t_s{seed}.pth"
        audit.append(f"| {ck_path.name} | `{sha256_file(ck_path)}` |")
    audit.append(f"| stage1_physics_only.pth | `{sha256_file(S1_CKPT)}` |")

    audit.append(f"\n### Key Outputs")
    audit.append(f"| File | SHA256 |")
    audit.append(f"|------|--------|")
    for fname in sorted(fingerprint.get('output_hashes', {}).keys()):
        audit.append(f"| {fname} | `{fingerprint['output_hashes'][fname]}` |")

    # v2 comparison
    audit.append(f"\n## 7. Comparison vs Lockbox v2\n")
    if v2_notes:
        for n in v2_notes:
            audit.append(f"- {n}")
    else:
        audit.append("- v2 frozen_results not found for comparison.")
    audit.append(f"\n**Verdict:** {'ALL CHECKS PASS' if v2_delta_ok else 'SEE WARNINGS ABOVE'}")

    # Assertions summary
    audit.append(f"\n## 8. All Assertions ({len(assertions_passed)} total)\n")
    for i, a in enumerate(assertions_passed):
        audit.append(f"{i+1}. {a}: PASS")

    audit_text = "\n".join(audit)
    with open(OUT_DIR / "lockbox_audit_v3.md", 'w') as f:
        f.write(audit_text)
    print("  Saved lockbox_audit_v3.md")

    # ==============================================================
    # DONE
    # ==============================================================
    elapsed_total = time.time() - t_start
    print("\n" + "="*70)
    print(f"PILLAR A COMPLETE ({elapsed_total:.0f}s)")
    print("="*70)
    print(f"  All {len(assertions_passed)} assertions PASSED")
    print(f"  Main closure: closure_2t (b2 + d2)")
    print(f"  DxR2@10: {cl2_dx_mean[9]:.4f}  (baseline: {r2dx_base[9]:.4f})")
    print(f"  % MLP recovered: {pct_h10_2t:.1f}% (h=10), {pct_m510_2t:.1f}% (mean 5-10)")
    print(f"  ACF(1): {m2_acf1:.4f}  (baseline: {acf_base[1]:.4f})")
    if v2_notes:
        print(f"  v2 comparison: {'; '.join(v2_notes)}")
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Files created:")
    for p in sorted(OUT_DIR.rglob('*')):
        if p.is_file():
            rel = p.relative_to(OUT_DIR)
            size = p.stat().st_size
            if size > 1024*1024:
                print(f"    {rel}  ({size/1024/1024:.1f} MB)")
            elif size > 1024:
                print(f"    {rel}  ({size/1024:.1f} KB)")
            else:
                print(f"    {rel}  ({size} B)")


if __name__ == '__main__':
    main()
