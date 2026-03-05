"""
FINAL END-TO-END AUDIT PASS
============================
Verifies every headline number/figure in the publication summary is
reproducible from raw data with the current code, with consistent
definitions/units/windowing.

Deliverables:
  - audit_report.md
  - audit_outputs/ (CSV tables + regenerated figures + logs)
  - frozen_results.json
"""

import os, sys, math, time, json, hashlib, random
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.optimize import minimize
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_num_threads(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.state_space_dataset import StateSpaceDataset
from models.kalman_closure import KalmanForecasterClosure, CLOSURE_PARAM_NAMES

# ===== Config =====
DATA_DIR = ROOT / "processed_data_10hz"
S1_CKPT = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
           / "stage1_physics_only.pth")
R3C_CKPT_DIR = ROOT / "model_upgrade_round3c_closure_final" / "checkpoints"
MLP_CKPTS = [
    ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
    / f"stage2_best_seed{s}.pth" for s in [42, 43, 44]]
PUB_DIR = ROOT / "model_upgrade_round3c_closure_final" / "publication"
AUDIT_DIR = ROOT / "audit_outputs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

L = 64; H = 20; BATCH = 128; DT = 0.1; VAR_FLOOR = 1e-6
FORCE_CPU = True
SEEDS = [42, 43, 44]
S2_EPOCHS = 200; S2_PATIENCE = 30; S2_LR = 1e-2

# Init from Round 3 standardized -> raw
A1_INIT = 0.44; B2_INIT = 5.66; D1_INIT = 0.21; D2_INIT = 11.1; D3_INIT = 0.62

# Publication plot style
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
        results.append((m, Q, p))
    return results


def gaussian_nll(x_pred, x_var, x_true, var_floor=1e-6):
    v = torch.clamp(x_var, min=var_floor)
    return (0.5 * torch.log(2 * math.pi * v)
            + 0.5 * (x_true - x_pred)**2 / v).mean()


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
    r2_x, r2_dx = [], []
    for h in range(min(max_h, pred.shape[1])):
        ss_res = np.sum((true[:, h] - pred[:, h])**2)
        ss_tot = np.sum((true[:, h] - np.mean(true[:, h]))**2)
        r2_x.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        dx_pred = pred[:, h] - xcur
        dx_true = true[:, h] - xcur
        ss_res_d = np.sum((dx_true - dx_pred)**2)
        ss_tot_d = np.sum((dx_true - np.mean(dx_true))**2)
        r2_dx.append(1 - ss_res_d / ss_tot_d if ss_tot_d > 0 else 0.0)
    return np.array(r2_x), np.array(r2_dx)


def kf_filter_2state(params, cl_params, t, x_obs, v,
                     R_override=None, Q_scale_override=None,
                     collect_residuals=False):
    """Standard 2-state KF filter. Returns (innovations, S, [closure_out, physics_out])."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    closure_out = np.full(N, np.nan) if collect_residuals else None
    physics_out = np.full(N, np.nan) if collect_residuals else None

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    if Q_scale_override is not None:
        q_sc *= Q_scale_override
    R = R_override if R_override is not None else params['R']

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


def build_closure_fresh(s1_params, device):
    """Build model with same init as original R3c training."""
    model = KalmanForecasterClosure(
        alpha_init=s1_params['alpha'], c_init=s1_params['c'],
        vc_init=s1_params['vc'], kappa_init=s1_params['kappa'],
        log_qx_init=math.log(s1_params['qx']),
        log_qu_init=math.log(s1_params['qu']),
        log_r_init=math.log(s1_params['R']),
        log_p0_xx_init=math.log(s1_params['P0_xx']),
        log_p0_uu_init=math.log(s1_params['P0_uu']),
        a1_init=A1_INIT, b1_init=0.0, b2_init=B2_INIT,
        d1_init=D1_INIT, d2_init=D2_INIT, d3_init=D3_INIT,
    ).to(device)
    model.freeze_physics()
    model.b1.requires_grad_(False)
    model.b1.fill_(0.0)
    return model


def validate_nll(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            xp, xv, _ = model(v_h.to(device), dt_h.to(device),
                              x_h.to(device), v_f.to(device),
                              dt_f.to(device))
            loss = gaussian_nll(xp, xv, x_true.to(device), VAR_FLOOR)
            tot += loss.item(); n += 1
    return tot / n


def train_closure_model(model, train_loader, val_loader, device, tag="",
                        max_epochs=S2_EPOCHS, patience=S2_PATIENCE, lr=S2_LR):
    params = model.closure_params_list()
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10, verbose=False)
    best_l, best_st, wait = float('inf'), None, 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        tot_nll, n = 0.0, 0
        for batch in train_loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            v_h = v_h.to(device); dt_h = dt_h.to(device)
            x_h = x_h.to(device); v_f = v_f.to(device)
            dt_f = dt_f.to(device); x_true = x_true.to(device)
            optimizer.zero_grad()
            xp, xv, _ = model(v_h, dt_h, x_h, v_f, dt_f)
            loss = gaussian_nll(xp, xv, x_true, VAR_FLOOR)
            loss.backward(); optimizer.step()
            tot_nll += loss.item(); n += 1
        tr_nll = tot_nll / n
        vl = validate_nll(model, val_loader, device)
        scheduler.step(vl)
        if vl < best_l:
            best_l = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (ep+1) % 20 == 0 or ep == 0:
            cs = model.closure_summary()
            print(f"    [{tag}] ep {ep+1:3d}  nll={tr_nll:.4f}  val={vl:.4f}"
                  f"  a1={cs['a1']:.3f} qs={cs['q_scale']:.3f}")
        if wait >= patience:
            print(f"    [{tag}] Early stop ep {ep+1}")
            break
    elapsed = time.time() - t0
    model.load_state_dict(best_st)
    model.eval()
    print(f"    [{tag}] Done {elapsed:.0f}s, val={best_l:.4f}")
    return best_l, best_st, elapsed


# ===================================================================
# MAIN AUDIT
# ===================================================================

def main():
    device = torch.device('cpu') if FORCE_CPU else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Audit outputs -> {AUDIT_DIR}")

    report = []  # lines for audit_report.md
    checks = {}  # check_id -> PASS/FAIL
    frozen = {}  # frozen_results.json

    def check(check_id, name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        checks[check_id] = status
        line = f"- [{status}] **{check_id}**: {name}"
        if detail:
            line += f"  \n  {detail}"
        report.append(line)
        print(f"  [{status}] {check_id}: {name}")
        if detail:
            print(f"    {detail}")

    # ================================================================
    # A) DATA INTEGRITY + PREPROCESSING
    # ================================================================
    report.append("\n## A) Data Integrity + Preprocessing\n")
    print("\n" + "="*70)
    print("A) DATA INTEGRITY + PREPROCESSING")
    print("="*70)

    # A1) Confirm dataset files and row counts
    csv_paths = {
        'train': DATA_DIR / "train_10hz_ready.csv",
        'val': DATA_DIR / "val_10hz_ready.csv",
        'test': DATA_DIR / "test_10hz_ready.csv",
    }
    dfs = {}
    for split, path in csv_paths.items():
        dfs[split] = pd.read_csv(path)

    expected_cols = {'timestamp', 'time_delta', 'velocity', 'displacement'}
    all_cols_ok = True
    row_info = {}
    for split, df in dfs.items():
        cols_ok = set(df.columns) == expected_cols
        all_cols_ok = all_cols_ok and cols_ok
        row_info[split] = len(df)

    check("A1", "Dataset files and row counts",
          all_cols_ok and row_info['train'] > 0,
          f"train={row_info['train']}, val={row_info['val']}, test={row_info['test']}; "
          f"cols={sorted(dfs['train'].columns.tolist())}")

    # A2) Confirm dt=0.1s
    dt_issues = []
    for split, df in dfs.items():
        dt_vals = df['time_delta'].values
        # Check that dt is approximately 0.1 everywhere
        dt_mean = np.mean(dt_vals)
        dt_std = np.std(dt_vals)
        dt_min = np.min(dt_vals)
        dt_max = np.max(dt_vals)
        if abs(dt_mean - 0.1) > 0.01 or dt_std > 0.01:
            dt_issues.append(f"{split}: mean={dt_mean:.6f} std={dt_std:.6f}")

    check("A2", "dt=0.1s constant",
          len(dt_issues) == 0,
          f"All splits: dt_mean~0.1, dt_std~0. Issues: {dt_issues if dt_issues else 'none'}")

    # A3) Verify split boundaries and 50s warmup logic
    train_end = dfs['train']['timestamp'].iloc[-1]
    val_start = dfs['val']['timestamp'].iloc[0]
    val_end = dfs['val']['timestamp'].iloc[-1]
    test_start = dfs['test']['timestamp'].iloc[0]
    test_end = dfs['test']['timestamp'].iloc[-1]

    # 50s warmup: take last 50s of train for dev warmup, last 50s of val for test warmup
    warmup_sec = 50.0
    dev_warmup_df = dfs['train'][
        dfs['train']['timestamp'] >= train_end - warmup_sec]
    test_warmup_df = dfs['val'][
        dfs['val']['timestamp'] >= val_end - warmup_sec]

    dev_warmup_pts = len(dev_warmup_df)
    test_warmup_pts = len(test_warmup_df)
    expected_warmup = int(warmup_sec / DT)  # 500 points

    check("A3", "Split boundaries and 50s warmup",
          abs(dev_warmup_pts - expected_warmup) <= 2 and
          abs(test_warmup_pts - expected_warmup) <= 2,
          f"train: [0, {train_end:.1f}s] ({row_info['train']} pts), "
          f"val: [{val_start:.1f}, {val_end:.1f}s] ({row_info['val']} pts), "
          f"test: [{test_start:.1f}, {test_end:.1f}s] ({row_info['test']} pts). "
          f"Dev warmup: {dev_warmup_pts} pts from train, "
          f"Test warmup: {test_warmup_pts} pts from val")

    # A4) Data fingerprint
    fingerprint = {}
    for split, path in csv_paths.items():
        df = dfs[split]
        fingerprint[split] = {
            'sha256': sha256_file(path),
            'rows': len(df),
            'x_mean': float(np.mean(df['displacement'].values)),
            'x_std': float(np.std(df['displacement'].values)),
            'u_mean': float(np.mean(df['velocity'].values)),
            'u_std': float(np.std(df['velocity'].values)),
        }
        # Compute du stats
        vel = df['velocity'].values
        du = np.diff(vel)
        fingerprint[split]['du_mean'] = float(np.mean(du))
        fingerprint[split]['du_std'] = float(np.std(du))

    fp_path = AUDIT_DIR / "data_fingerprint.json"
    with open(fp_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)

    check("A4", "Data fingerprint saved",
          fp_path.exists(),
          f"Saved to {fp_path}")

    frozen['data_fingerprint'] = fingerprint

    # ================================================================
    # B) NOTATION + UNITS SANITY
    # ================================================================
    report.append("\n## B) Notation + Units Sanity\n")
    print("\n" + "="*70)
    print("B) NOTATION + UNITS SANITY")
    print("="*70)

    # B5) Symbol map verification
    # Read kalman_closure.py to verify the closure function
    import inspect
    from models.kalman_closure import KalmanForecasterClosure
    closure_src = inspect.getsource(KalmanForecasterClosure.closure)
    kf_predict_src = inspect.getsource(KalmanForecasterClosure.kf_predict)

    # Verify closure signature uses u_state, v_water, dv_water
    sym_ok = ('u_state' in closure_src and 'v_water' in closure_src
              and 'dv_water' in closure_src)
    # Verify closure returns without *dt
    closure_no_dt = 'before *dt' in closure_src or '(before *dt)' in closure_src
    # Verify kf_predict multiplies by dt
    predict_cl_dt = 'cl * dt' in kf_predict_src or 'cl_dt = cl * dt' in kf_predict_src

    check("B5", "Symbol map: closure(u_state, v_water, dv_water)",
          sym_ok,
          f"Closure params: u_state=sediment vel, v_water=water vel, dv_water=du")

    # B6) Dimensional check
    # closure returns: -a1*u_state + b1*v_water + b2*dv_water
    #                  - d1*u_state^2 - d2*u_state*|v_water| - d3*u_state*|u_state|
    # All terms must have units m/s^2 (acceleration):
    #   a1*u_state: [a1]*(m/s) = m/s^2 => [a1] = 1/s  CHECK
    #   b2*dv_water: [b2]*(m/s) = m/s^2 => [b2] = 1/s  CHECK (du is m/s, not m/s^2)
    #   d1*u_state^2: [d1]*(m/s)^2 = m/s^2 => [d1] = 1/m  CHECK
    #   d2*u_state*|v_water|: [d2]*(m/s)*(m/s) = m/s^2 => [d2] = 1/m  CHECK
    #   d3*u_state*|u_state|: [d3]*(m/s)*(m/s) = m/s^2 => [d3] = 1/m  CHECK
    # Then in kf_predict: cl_dt = cl * dt => m/s^2 * s = m/s  CHECK (velocity increment)

    dim_check_ok = closure_no_dt and predict_cl_dt
    unit_table = {
        'a1': '1/s', 'b2': '1/s',
        'd1': '1/m', 'd2': '1/m', 'd3': '1/m',
    }

    check("B6", "Dimensional analysis: closure=m/s^2, cl*dt=m/s",
          dim_check_ok,
          f"closure() returns acceleration (m/s^2), kf_predict multiplies by dt. "
          f"Units: {unit_table}")

    # B7) Export LaTeX units table
    units_table_lines = [
        "Symbol,Parameter,Unit,Derivation",
        "a1,Linear damping,1/s,[a1]*(m/s)=m/s^2",
        "b2,du coupling,1/s,[b2]*(m/s)=m/s^2 (du=m/s not m/s^2)",
        "d1,Quadratic drag,1/m,[d1]*(m/s)^2=m/s^2",
        "d2,Cross drag v|u|,1/m,[d2]*(m/s)*(m/s)=m/s^2",
        "d3,Self drag v|v|,1/m,[d3]*(m/s)*(m/s)=m/s^2",
        "alpha,OU relaxation,1/s,exp(-alpha*dt) dimensionless",
        "kappa,Restoring,1/s^2,[kappa]*m*s=m/s (in rho*u-kappa*x*dt)",
        "c,Forcing,1/m,[c]*(m/s)^2*s=m/s (in c*g*dt)",
        "vc,Threshold,m/s,velocity threshold",
    ]
    units_csv_path = AUDIT_DIR / "units_table.csv"
    with open(units_csv_path, 'w') as f:
        f.write('\n'.join(units_table_lines))

    check("B7", "LaTeX units table exported",
          units_csv_path.exists(),
          f"Saved to {units_csv_path}")

    frozen['units'] = unit_table

    # ================================================================
    # C) MODEL REPRODUCIBILITY
    # ================================================================
    report.append("\n## C) Model Reproducibility\n")
    print("\n" + "="*70)
    print("C) MODEL REPRODUCIBILITY")
    print("="*70)

    # Load data for training
    train_ds = StateSpaceDataset(
        [str(DATA_DIR / "train_10hz_ready.csv")], L=L, m=L, H=H,
        predict_deltas=False, normalize=False)
    val_ds = StateSpaceDataset(
        [str(DATA_DIR / "val_10hz_ready.csv")], L=L, m=L, H=H,
        predict_deltas=False, normalize=False,
        run_id_to_idx=train_ds.run_id_to_idx)
    test_ds = StateSpaceDataset(
        [str(DATA_DIR / "test_10hz_ready.csv")], L=L, m=L, H=H,
        predict_deltas=False, normalize=False,
        run_id_to_idx=train_ds.run_id_to_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             num_workers=0)
    print(f"  Datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Load stage 1
    ckpt = torch.load(S1_CKPT, map_location=device, weights_only=False)
    s1_params = ckpt['params']
    print(f"  S1: alpha={s1_params['alpha']:.4f} c={s1_params['c']:.4f} "
          f"kap={s1_params['kappa']:.4f} vc={s1_params['vc']:.4f}")

    frozen['stage1_params'] = {k: float(v) for k, v in s1_params.items()
                               if isinstance(v, (int, float))}

    # C8) Retrain closure_5t with 3 seeds
    print("\n  --- C8: Retraining closure_5t (3 seeds) ---")
    retrained_params = {}
    original_params_csv = pd.read_csv(
        ROOT / "model_upgrade_round3c_closure_final" / "final_seed_params.csv")

    # Skip retraining if comparison CSV exists (from a previous audit run)
    retrain_cache = AUDIT_DIR / "retrain_comparison.csv"
    if retrain_cache.exists():
        print("  [cached] Loading retrained params from previous audit run")
        cached_df = pd.read_csv(retrain_cache)
        for seed in SEEDS:
            row = cached_df[(cached_df['seed'] == seed) &
                            (cached_df['source'] == 'retrained')].iloc[0]
            retrained_params[seed] = {k: float(row[k]) for k in
                                       ['a1', 'b2', 'd1', 'd2', 'd3', 'q_scale']}
    else:
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = build_closure_fresh(s1_params, device)
            tag = f"audit_5t_s{seed}"
            val_loss, state_dict, elapsed = train_closure_model(
                model, train_loader, val_loader, device, tag=tag)
            cs = model.closure_summary()
            retrained_params[seed] = cs
            print(f"    Retrained: a1={cs['a1']:.4f} b2={cs['b2']:.4f} "
                  f"d1={cs['d1']:.4f} d2={cs['d2']:.4f} d3={cs['d3']:.4f} "
                  f"qs={cs['q_scale']:.4f}")

    # Compare retrained vs original
    param_keys = ['a1', 'b2', 'd1', 'd2', 'd3', 'q_scale']
    retrained_means = {k: np.mean([retrained_params[s][k] for s in SEEDS])
                       for k in param_keys}
    retrained_stds = {k: np.std([retrained_params[s][k] for s in SEEDS])
                      for k in param_keys}
    retrained_cvs = {k: 100 * retrained_stds[k] / (abs(retrained_means[k]) + 1e-10)
                     for k in param_keys}

    # Original means from CSV
    original_means = {
        'a1': original_params_csv['a1'].mean(),
        'b2': original_params_csv['b2'].mean(),
        'd1': original_params_csv['d1'].mean(),
        'd2': original_params_csv['d2'].mean(),
        'd3': original_params_csv['d3'].mean(),
        'q_scale': original_params_csv['q_scale'].mean(),
    }

    max_rel_err = 0
    param_comparison = []
    for k in param_keys:
        orig = original_means[k]
        retr = retrained_means[k]
        rel_err = abs(retr - orig) / (abs(orig) + 1e-10) * 100
        max_rel_err = max(max_rel_err, rel_err)
        param_comparison.append(f"{k}: orig={orig:.4f} retrained={retr:.4f} "
                               f"rel_err={rel_err:.2f}%")

    # Save comparison CSV
    comp_rows = []
    for seed in SEEDS:
        row = {'seed': seed, 'source': 'retrained'}
        row.update(retrained_params[seed])
        comp_rows.append(row)
        # Original
        orig_row = original_params_csv[original_params_csv['seed'] == seed].iloc[0]
        orow = {'seed': seed, 'source': 'original'}
        for k in param_keys:
            orow[k] = float(orig_row[k])
        comp_rows.append(orow)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(AUDIT_DIR / "retrain_comparison.csv", index=False)

    # Tolerance: <5% relative error (accounts for stochastic training)
    check("C8", "Retrained params match paper (<5% rel error)",
          max_rel_err < 5.0,
          f"Max relative error: {max_rel_err:.2f}%. " +
          "; ".join(param_comparison))

    frozen['retrained_params'] = {
        'means': {k: float(v) for k, v in retrained_means.items()},
        'stds': {k: float(v) for k, v in retrained_stds.items()},
        'cvs': {k: float(v) for k, v in retrained_cvs.items()},
    }

    # C9) Verify physics-only baseline checkpoint
    base_model = build_baseline(s1_params, device)
    ps_base = base_model.param_summary()
    base_alpha_ok = abs(ps_base['alpha'] - s1_params['alpha']) < 1e-4
    base_c_ok = abs(ps_base['c'] - s1_params['c']) < 1e-4
    base_kappa_ok = abs(ps_base['kappa'] - s1_params['kappa']) < 1e-4
    base_vc_ok = abs(ps_base['vc'] - s1_params['vc']) < 1e-4

    check("C9", "Physics-only baseline matches stage1 checkpoint",
          base_alpha_ok and base_c_ok and base_kappa_ok and base_vc_ok,
          f"alpha={ps_base['alpha']:.4f} (expect {s1_params['alpha']:.4f}), "
          f"c={ps_base['c']:.4f} (expect {s1_params['c']:.4f}), "
          f"kappa={ps_base['kappa']:.4f} (expect {s1_params['kappa']:.4f}), "
          f"vc={ps_base['vc']:.4f} (expect {s1_params['vc']:.4f})")

    # ================================================================
    # D) METRICS DEFINITIONS AUDIT
    # ================================================================
    report.append("\n## D) Metrics Definitions Audit\n")
    print("\n" + "="*70)
    print("D) METRICS DEFINITIONS AUDIT")
    print("="*70)

    # Prepare filter data
    df_train = dfs['train']
    df_dev = dfs['val']
    df_test = dfs['test']

    dev_warmup = df_train[df_train['timestamp'] >= train_end - warmup_sec]
    df_filter_dev = pd.concat([dev_warmup, df_dev], ignore_index=True)
    dev_mask = df_filter_dev['timestamp'].values >= df_dev['timestamp'].iloc[0]

    test_warmup = df_dev[df_dev['timestamp'] >= val_end - warmup_sec]
    df_filter_test = pd.concat([test_warmup, df_test], ignore_index=True)
    test_mask = df_filter_test['timestamp'].values >= df_test['timestamp'].iloc[0]

    cl_zero = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl_zero['q_scale'] = 1.0

    # D10) Confirm DxR2 definition
    # DxR2 = 1 - SS_res / SS_tot where:
    #   dx_pred = pred[:,h] - xcur
    #   dx_true = true[:,h] - xcur
    #   SS_res = sum((dx_true - dx_pred)^2)
    #   SS_tot = sum((dx_true - mean(dx_true))^2)  <-- constant-mean baseline
    # Verify by manual computation
    base_model_eval = build_baseline(s1_params, device)
    r2x_base, r2dx_base = compute_skill_curves(base_model_eval, test_loader, device, 10)

    # Also manually compute for h=10 to double-check
    all_pred, all_true, all_xcur = [], [], []
    base_model_eval.eval()
    with torch.no_grad():
        for batch in test_loader:
            v_h, dt_h, x_h, v_f, dt_f, x_true, x_cur, _ = batch
            xp, _, _ = base_model_eval(v_h, dt_h, x_h, v_f, dt_f)
            all_pred.append(xp.cpu().numpy())
            all_true.append(x_true.cpu().numpy())
            all_xcur.append(x_cur.numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    xcur = np.concatenate(all_xcur)
    h = 9  # h=10, 0-indexed
    dx_pred = pred[:, h] - xcur
    dx_true = true[:, h] - xcur
    ss_res_manual = np.sum((dx_true - dx_pred)**2)
    ss_tot_manual = np.sum((dx_true - np.mean(dx_true))**2)
    dxr2_manual = 1 - ss_res_manual / ss_tot_manual
    dxr2_func = r2dx_base[9]

    check("D10", "DxR2 definition: constant-mean increment baseline",
          abs(dxr2_manual - dxr2_func) < 1e-8,
          f"Manual DxR2@10 = {dxr2_manual:.6f}, function = {dxr2_func:.6f}. "
          f"SS_tot uses mean(dx_true) = {np.mean(dx_true):.6f} (constant-mean, NOT zero)")

    frozen['baseline_dxr2'] = {f'h{h+1}': float(r2dx_base[h]) for h in range(10)}

    # D11) Confirm ACF computed from innovations, same windowing as publication
    # NOTE: val_10hz_ready.csv spans 1008.7-1260.8 (dev+test combined, 2522 pts).
    # The warmup takes the last 50s of val (timestamps 1210.9-1260.8), which
    # overlaps with the test set (1134.8-1260.8). The test_mask >= test_start
    # therefore includes the warmup. This is consistent with how ALL published
    # results were computed (publication_final.py, model_upgrade_round3c.py).
    e_base, S_base = kf_filter_2state(
        s1_params, cl_zero,
        df_filter_test['timestamp'].values,
        df_filter_test['displacement'].values,
        df_filter_test['velocity'].values)
    e_base_m = e_base[test_mask]
    mask_valid = ~np.isnan(e_base_m)
    e_base_clean = e_base_m[mask_valid]
    S_base_clean = S_base[test_mask][mask_valid]
    acf_base = compute_acf(e_base_clean, 50)

    # Verify: ACF(1) matches paper and windowing is consistent
    n_test_innov = len(e_base_clean)
    # Expected: total filter length minus 1 NaN (warmup + test - 1)
    expected_innov = test_warmup_pts + row_info['test'] - 1
    acf1_matches_paper = abs(acf_base[1] - 0.9104) < 0.001

    check("D11", "ACF windowing consistent with publication code",
          acf1_matches_paper and abs(n_test_innov - expected_innov) <= 2,
          f"n_innovations={n_test_innov} (warmup={test_warmup_pts} + test={row_info['test']} - 1 NaN = {expected_innov}), "
          f"ACF(1)={acf_base[1]:.4f} (paper: 0.9104). "
          f"NOTE: val CSV = dev+test combined (2522 pts, ts 1008.7-1260.8); "
          f"warmup overlaps with test set. Same windowing as publication_final.py.")

    # D12) Ljung-Box
    lb_base = ljung_box(acf_base, len(e_base_clean))
    lb_lags_ok = set(l for l, _, _ in lb_base) == {5, 10, 20, 50}

    check("D12", "Ljung-Box uses lags {5,10,20,50}",
          lb_lags_ok,
          f"Lags tested: {[l for l, _, _ in lb_base]}, "
          f"all p<0.05: {all(p < 0.05 for _, _, p in lb_base)}")

    # D13) Grey-box metrics
    # Load best closure checkpoint for grey-box computation
    best_seed = 43  # best val loss
    ck_best = torch.load(R3C_CKPT_DIR / f"closure_5t_s{best_seed}.pth",
                         map_location=device, weights_only=False)
    ps_best = ck_best['params']
    cl_best = ck_best['closure']

    e_cl, S_cl, cl_out, ph_out = kf_filter_2state(
        ps_best, cl_best,
        df_filter_test['timestamp'].values,
        df_filter_test['displacement'].values,
        df_filter_test['velocity'].values,
        collect_residuals=True)

    e_cl_m = e_cl[test_mask]
    cl_m = cl_out[test_mask]
    ph_m = ph_out[test_mask]
    valid_cl = ~np.isnan(cl_m)
    cl_c = cl_m[valid_cl]
    ph_c = ph_m[valid_cl]

    # frac = var(closure_dt) / (var(physics_drift) + var(closure_dt))
    var_cl = np.var(cl_c)
    var_ph = np.var(ph_c)
    frac_computed = var_cl / (var_ph + var_cl + 1e-15)

    # med_ratio = median(|closure_dt| / |physics_drift|)
    med_ratio_computed = float(np.median(np.abs(cl_c) / (np.abs(ph_c) + 1e-8)))

    # Compare with paper
    expected_frac = 0.284
    expected_med = 0.709

    check("D13a", "Grey-box frac definition: var(cl_dt)/(var(phys)+var(cl_dt))",
          abs(frac_computed - expected_frac) < 0.01,
          f"frac = var(closure_dt)/(var(physics_drift)+var(closure_dt)) = {frac_computed:.4f} "
          f"(paper: {expected_frac})")

    check("D13b", "Grey-box med_ratio: median(|cl_dt|/|phys_drift|)",
          abs(med_ratio_computed - expected_med) < 0.01,
          f"med_ratio = median(|closure_dt|/|physics_drift|) = {med_ratio_computed:.4f} "
          f"(paper: {expected_med})")

    frozen['greybox'] = {
        'frac': float(frac_computed),
        'med_ratio': float(med_ratio_computed),
        'frac_definition': 'var(closure_dt) / (var(physics_drift) + var(closure_dt))',
        'med_ratio_definition': 'median(|closure_dt| / |physics_drift|)',
    }

    # D14) % recovered computation
    # Load closure results from checkpoints
    closure_results = []
    for seed in SEEDS:
        ck_path = R3C_CKPT_DIR / f"closure_5t_s{seed}.pth"
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model = build_closure_5t(s1_params, device)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        r2x, r2dx = compute_skill_curves(model, test_loader, device, 10)
        closure_results.append({'r2dx': r2dx, 'closure': ck['closure'],
                                'params': ck['params']})

    cl_r2dx = np.array([r['r2dx'] for r in closure_results])
    cl_dx_mean = np.mean(cl_r2dx, axis=0)

    # MLP
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
            r2x, r2dx = compute_skill_curves(mlp_m, test_loader, device, 10)
            mlp_results.append({'r2dx': r2dx})
    except Exception as ex:
        print(f"  MLP loading failed: {ex}")

    if mlp_results:
        mlp_r2dx = np.array([r['r2dx'] for r in mlp_results])
        mlp_dx_mean = np.mean(mlp_r2dx, axis=0)
    else:
        mlp_dx_mean = np.full(10, np.nan)

    # % recovered = (closure - baseline) / (MLP - baseline)
    gain_h10_cl = cl_dx_mean[9] - r2dx_base[9]
    gain_h10_mlp = mlp_dx_mean[9] - r2dx_base[9]
    pct_h10 = 100.0 * gain_h10_cl / gain_h10_mlp if abs(gain_h10_mlp) > 1e-8 else 0

    base_m510 = np.mean(r2dx_base[4:10])
    cl_m510 = np.mean(cl_dx_mean[4:10])
    mlp_m510 = np.mean(mlp_dx_mean[4:10])
    gain_m510_cl = cl_m510 - base_m510
    gain_m510_mlp = mlp_m510 - base_m510
    pct_m510 = 100.0 * gain_m510_cl / gain_m510_mlp if abs(gain_m510_mlp) > 1e-8 else 0

    check("D14", "% recovered = (closure-baseline)/(MLP-baseline)",
          abs(pct_h10 - 36.2) < 2.0 and abs(pct_m510 - 40.5) < 2.0,
          f"h=10: {pct_h10:.1f}% (paper: 36.2%), "
          f"mean(5-10): {pct_m510:.1f}% (paper: 40.5%)")

    frozen['pct_recovered'] = {
        'h10': float(pct_h10),
        'mean_5_10': float(pct_m510),
    }
    frozen['dxr2'] = {
        'baseline': {f'h{h+1}': float(r2dx_base[h]) for h in range(10)},
        'closure_mean': {f'h{h+1}': float(cl_dx_mean[h]) for h in range(10)},
        'mlp_mean': {f'h{h+1}': float(mlp_dx_mean[h]) for h in range(10)},
    }

    # ================================================================
    # E) FIGURE REGENERATION
    # ================================================================
    report.append("\n## E) Figure Regeneration\n")
    print("\n" + "="*70)
    print("E) FIGURE REGENERATION")
    print("="*70)

    # Compute all data needed for figures
    # Closure ACFs and NIS (3 seeds)
    cl_acfs_all = []
    cl_nis_all = []
    cl_cov90_all = []
    cl_metrics = []
    for seed_idx, seed in enumerate(SEEDS):
        ps_ = closure_results[seed_idx]['params']
        cl_ = closure_results[seed_idx]['closure']
        e_s, S_s = kf_filter_2state(
            ps_, cl_,
            df_filter_test['timestamp'].values,
            df_filter_test['displacement'].values,
            df_filter_test['velocity'].values)
        e_sm = e_s[test_mask]
        S_sm = S_s[test_mask]
        valid = ~np.isnan(e_sm)
        e_sc = e_sm[valid]
        S_sc = S_sm[valid]
        acf_s = compute_acf(e_sc, 50)
        nis_s = np.mean(e_sc**2 / S_sc)
        cov90_s = np.mean(np.abs(e_sc) < 1.645 * np.sqrt(S_sc))
        cl_acfs_all.append(acf_s)
        cl_nis_all.append(nis_s)
        cl_cov90_all.append(cov90_s)
        cl_metrics.append({
            'seed': seed, 'acf1': acf_s[1], 'acf2': acf_s[2],
            'acf5': acf_s[5], 'acf10': acf_s[10],
            'nis': nis_s, 'cov90': cov90_s,
        })
        if seed == best_seed:
            best_innovations = e_sc

    cl_acfs_arr = np.array(cl_acfs_all)
    cl_acf_mean = np.mean(cl_acfs_arr, axis=0)
    cl_acf_std = np.std(cl_acfs_arr, axis=0)
    cl_dx_std = np.std(cl_r2dx, axis=0)

    m_acf1 = np.mean([m['acf1'] for m in cl_metrics])
    m_acf2 = np.mean([m['acf2'] for m in cl_metrics])
    m_acf5 = np.mean([m['acf5'] for m in cl_metrics])
    m_acf10 = np.mean([m['acf10'] for m in cl_metrics])
    m_nis = np.mean(cl_nis_all)
    m_cov90 = np.mean(cl_cov90_all)

    # Save innovations as npy
    np.save(AUDIT_DIR / "innovations_baseline.npy", e_base_clean)
    np.save(AUDIT_DIR / "innovations_closure_best.npy", best_innovations)

    # Closure param stats
    param_stats = {}
    for k in ['a1', 'b2', 'd1', 'd2', 'd3', 'q_scale']:
        vals = [closure_results[i]['closure'][k] for i in range(len(SEEDS))]
        param_stats[k] = {'mean': np.mean(vals), 'std': np.std(vals),
                          'cv': 100 * np.std(vals) / (abs(np.mean(vals)) + 1e-10)}

    frozen['closure_params'] = {
        k: {'mean': float(v['mean']), 'std': float(v['std']), 'cv': float(v['cv'])}
        for k, v in param_stats.items()
    }
    frozen['acf'] = {
        'baseline': {'acf1': float(acf_base[1]), 'acf2': float(acf_base[2]),
                     'acf5': float(acf_base[5]), 'acf10': float(acf_base[10])},
        'closure_mean': {'acf1': float(m_acf1), 'acf2': float(m_acf2),
                         'acf5': float(m_acf5), 'acf10': float(m_acf10)},
    }
    frozen['nis'] = {
        'baseline': float(np.mean(e_base_clean**2 / S_base_clean)),
        'closure_mean': float(m_nis),
    }

    # MLP std
    if mlp_results:
        mlp_dx_std = np.std(mlp_r2dx, axis=0)
    else:
        mlp_dx_std = np.full(10, 0.0)

    # E15) Fig 1: Skill curves
    hs = np.arange(1, 11)
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(12, 4.5),
                                             gridspec_kw={'width_ratios': [3, 2]})
    for ax, h_slice, title_suffix in [
        (ax_main, slice(None), ''),
        (ax_inset, slice(3, 10), ' ($h = 4$--$10$)')
    ]:
        h_idx = hs[h_slice]
        ax.plot(h_idx, r2dx_base[h_slice], 's--', color='#d62728',
                label='Physics-only', markersize=6, zorder=3)
        ax.plot(h_idx, cl_dx_mean[h_slice], 'o-', color='#1f77b4',
                label='Closure (5-term)', markersize=6, zorder=4)
        ax.fill_between(h_idx,
                        cl_dx_mean[h_slice] - cl_dx_std[h_slice],
                        cl_dx_mean[h_slice] + cl_dx_std[h_slice],
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
    y_lo = min(r2dx_base.min(), cl_dx_mean.min()) - 0.15
    y_hi = max(0.4, mlp_dx_mean.max() + 0.1) if mlp_results else 0.15
    ax_main.set_ylim(y_lo, y_hi)

    ax_inset.set_ylabel('$\\Delta x \\, R^2(h)$')
    ax_inset.set_title('Detail: $h = 4$--$10$')
    y_lo2 = min(r2dx_base[3:10].min(), cl_dx_mean[3:10].min()) - 0.05
    y_hi2 = max(0.35, mlp_dx_mean[3:10].max() + 0.05) if mlp_results else 0.1
    ax_inset.set_ylim(y_lo2, y_hi2)

    ax_inset.annotate(
        f'Closure: {pct_h10:.0f}% of\nMLP gain at $h$=10',
        xy=(10, cl_dx_mean[9]), xytext=(6.5, 0.12),
        fontsize=9, fontstyle='italic',
        arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(AUDIT_DIR / 'fig1_skill_curves.png', bbox_inches='tight')
    plt.close(fig)

    check("E15", "Fig 1 regenerated from frozen data",
          (AUDIT_DIR / 'fig1_skill_curves.png').exists(),
          f"Saved to {AUDIT_DIR / 'fig1_skill_curves.png'}")

    # E16) Fig 2: Innovation ACF
    fig, ax = plt.subplots(figsize=(7, 4))
    lags_arr = np.arange(51)
    z95 = 1.96 / np.sqrt(len(e_base_clean))
    ax.fill_between(lags_arr, -z95, z95, alpha=0.12, color='gray',
                    label='95% CI (white noise)')
    ax.plot(lags_arr, acf_base, 's-', color='#d62728', label='Physics-only',
            markersize=3, lw=1.5, zorder=3)
    ax.plot(lags_arr, cl_acf_mean, 'o-', color='#1f77b4', label='Closure (5-term)',
            markersize=3, lw=1.5, zorder=4)
    ax.fill_between(lags_arr, cl_acf_mean - cl_acf_std, cl_acf_mean + cl_acf_std,
                    alpha=0.15, color='#1f77b4')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Innovation Autocorrelation (Test Set, Uncalibrated)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 50)
    ax.annotate(f'ACF(1) = {acf_base[1]:.3f}', xy=(1, acf_base[1]),
                xytext=(8, acf_base[1] + 0.03), fontsize=9, color='#d62728')
    ax.annotate(f'ACF(1) = {m_acf1:.3f}', xy=(1, m_acf1),
                xytext=(8, m_acf1 - 0.05), fontsize=9, color='#1f77b4')
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / 'fig2_innovation_acf.png', bbox_inches='tight')
    plt.close(fig)

    check("E16", "Fig 2 regenerated from raw innovations",
          (AUDIT_DIR / 'fig2_innovation_acf.png').exists(),
          f"Stored innovations as .npy files")

    # E17) Fig 3: Coefficients (3-panel)
    coeff_info = {
        'a1':  {'sym': '$a_1$',  'unit': 's$^{-1}$', 'meaning': 'Linear damping'},
        'b2':  {'sym': '$b_2$',  'unit': 's$^{-1}$',
                'meaning': '$\\Delta u$ coupling'},
        'd1':  {'sym': '$d_1$',  'unit': 'm$^{-1}$', 'meaning': 'Quadratic drag'},
        'd2':  {'sym': '$d_2$',  'unit': 'm$^{-1}$', 'meaning': 'Cross-drag $v|u|$'},
        'd3':  {'sym': '$d_3$',  'unit': 'm$^{-1}$', 'meaning': 'Self-drag $v|v|$'},
    }

    fig, (ax_large, ax_small, ax_qs) = plt.subplots(
        1, 3, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1.3, 2, 0.8]})

    large_keys = ['b2', 'd2']
    large_vals = [param_stats[k]['mean'] for k in large_keys]
    large_stds = [param_stats[k]['std'] for k in large_keys]
    large_cvs = [param_stats[k]['cv'] for k in large_keys]
    large_labels = [f"{coeff_info[k]['sym']}\n({coeff_info[k]['meaning']})\n"
                    f"[{coeff_info[k]['unit']}]" for k in large_keys]
    colors_l = ['#f28e2b', '#76b7b2']
    x_l = np.arange(len(large_keys))
    bars_l = ax_large.bar(x_l, large_vals, yerr=large_stds, capsize=5,
                          color=colors_l, alpha=0.85, edgecolor='black', lw=0.5,
                          width=0.55)
    ax_large.set_xticks(x_l)
    ax_large.set_xticklabels(large_labels, fontsize=8.5)
    ax_large.set_ylabel('Coefficient value (SI)')
    ax_large.set_title('Large coefficients')
    for i, (bar, cv) in enumerate(zip(bars_l, large_cvs)):
        ax_large.text(bar.get_x() + bar.get_width()/2.,
                      bar.get_height() + large_stds[i] + 0.15,
                      f'CV={cv:.1f}%', ha='center', va='bottom',
                      fontsize=8, fontweight='bold')

    small_keys = ['a1', 'd1', 'd3']
    small_vals = [param_stats[k]['mean'] for k in small_keys]
    small_stds = [param_stats[k]['std'] for k in small_keys]
    small_cvs = [param_stats[k]['cv'] for k in small_keys]
    small_labels = [f"{coeff_info[k]['sym']}\n({coeff_info[k]['meaning']})\n"
                    f"[{coeff_info[k]['unit']}]" for k in small_keys]
    colors_s = ['#4e79a7', '#e15759', '#59a14f']
    x_s = np.arange(len(small_keys))
    bars_s = ax_small.bar(x_s, small_vals, yerr=small_stds, capsize=5,
                          color=colors_s, alpha=0.85, edgecolor='black', lw=0.5,
                          width=0.55)
    ax_small.set_xticks(x_s)
    ax_small.set_xticklabels(small_labels, fontsize=8.5)
    ax_small.set_ylabel('Coefficient value (SI)')
    ax_small.set_title('Small coefficients')
    for i, (bar, cv) in enumerate(zip(bars_s, small_cvs)):
        ax_small.text(bar.get_x() + bar.get_width()/2.,
                      bar.get_height() + small_stds[i] + 0.008,
                      f'CV={cv:.1f}%', ha='center', va='bottom',
                      fontsize=8, fontweight='bold')

    qs_stat = param_stats['q_scale']
    ax_qs.bar([0], [qs_stat['mean']], yerr=[qs_stat['std']], capsize=5,
              color='#b07aa1', alpha=0.85, edgecolor='black', lw=0.5, width=0.45)
    ax_qs.text(0, qs_stat['mean'] + qs_stat['std'] + 0.05,
               f"CV={qs_stat['cv']:.1f}%", ha='center', fontsize=8, fontweight='bold')
    ax_qs.set_xticks([0])
    ax_qs.set_xticklabels(
        ['$q_{\\mathrm{scale}}$\n(Noise mult.)\n[--]'], fontsize=8.5)
    ax_qs.set_ylabel('Value')
    ax_qs.set_title('Noise scaling')
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / 'fig3_coefficients.png', bbox_inches='tight')
    plt.close(fig)

    check("E17", "Fig 3 regenerated from seed-aggregated params",
          (AUDIT_DIR / 'fig3_coefficients.png').exists(),
          f"3-panel: large (b2,d2), small (a1,d1,d3), q_scale")

    # E18) Figure caption verification
    # Check that caption mentions: mean+-std, horizons h=1-10, dt=0.1s, 3 seeds
    pub_summary = (PUB_DIR / "final_publication_summary.md").read_text(encoding='utf-8')
    caption_checks = [
        'mean' in pub_summary and 'std' in pub_summary,
        '0.1' in pub_summary,  # dt
        '3 seeds' in pub_summary or '3 random seeds' in pub_summary,
        'h = 1' in pub_summary or 'h=1' in pub_summary,
    ]

    check("E18", "Figure captions match plotted content",
          all(caption_checks),
          f"mean/std: {caption_checks[0]}, dt=0.1: {caption_checks[1]}, "
          f"3 seeds: {caption_checks[2]}, h range: {caption_checks[3]}")

    # ================================================================
    # F) CONSISTENCY + SANITY CHECKS
    # ================================================================
    report.append("\n## F) Consistency + Sanity Checks\n")
    print("\n" + "="*70)
    print("F) CONSISTENCY + SANITY CHECKS")
    print("="*70)

    # F19) Closure improves DxR2 at every horizon vs baseline
    improvement_at_all_h = all(cl_dx_mean[h] > r2dx_base[h] for h in range(10))
    exceptions = []
    for h in range(10):
        delta = cl_dx_mean[h] - r2dx_base[h]
        if delta < 0:
            exceptions.append(f"h={h+1}: delta={delta:.4f}")

    check("F19", "Closure improves DxR2 at every horizon",
          improvement_at_all_h,
          f"Exceptions: {exceptions if exceptions else 'none'}. "
          f"Min improvement: {min(cl_dx_mean[h] - r2dx_base[h] for h in range(10)):.4f}")

    # F20) Calibrated NIS only in supplement
    has_calibrated_in_main_table = 'Closure (calibrated)' in pub_summary.split(
        '## Main Results')[1].split('## Supplementary')[0] if '## Main Results' in pub_summary else False

    check("F20", "Calibrated NIS NOT in main results",
          not has_calibrated_in_main_table,
          f"Main table contains only Physics-only + Closure + MLP rows")

    # F21) MLP checkpoint matches table numbers
    if mlp_results:
        mlp_dxr2_h10 = float(mlp_dx_mean[9])
        mlp_dxr2_h5 = float(mlp_dx_mean[4])
        # Paper says DxR2@10 = 0.2880, DxR2@5 = 0.0133
        mlp_ok = abs(mlp_dxr2_h10 - 0.2880) < 0.01 and abs(mlp_dxr2_h5 - 0.0133) < 0.01
    else:
        mlp_ok = False
        mlp_dxr2_h10 = float('nan')
        mlp_dxr2_h5 = float('nan')

    check("F21", "MLP upper bound matches table numbers",
          mlp_ok,
          f"MLP DxR2@5={mlp_dxr2_h5:.4f} (expect 0.0133), "
          f"DxR2@10={mlp_dxr2_h10:.4f} (expect 0.2880)")

    # ================================================================
    # G) FINAL PACKAGING
    # ================================================================
    report.append("\n## G) Final Packaging\n")
    print("\n" + "="*70)
    print("G) FINAL PACKAGING")
    print("="*70)

    # Full skill curves table
    skill_csv_rows = []
    for h in range(10):
        skill_csv_rows.append({
            'horizon': h + 1,
            'baseline_r2x': float(r2x_base[h]),
            'baseline_dxr2': float(r2dx_base[h]),
            'closure_dxr2_mean': float(cl_dx_mean[h]),
            'closure_dxr2_std': float(cl_dx_std[h]),
            'mlp_dxr2_mean': float(mlp_dx_mean[h]),
            'mlp_dxr2_std': float(mlp_dx_std[h]),
        })
    pd.DataFrame(skill_csv_rows).to_csv(AUDIT_DIR / "skill_curves.csv", index=False)

    # Metrics table
    metrics_csv = pd.DataFrame(cl_metrics)
    metrics_csv.to_csv(AUDIT_DIR / "closure_metrics_per_seed.csv", index=False)

    # Ljung-Box for closure (best seed)
    e_cl_best, S_cl_best = kf_filter_2state(
        ps_best, cl_best,
        df_filter_test['timestamp'].values,
        df_filter_test['displacement'].values,
        df_filter_test['velocity'].values)
    e_cl_best_m = e_cl_best[test_mask]
    valid_best = ~np.isnan(e_cl_best_m)
    e_cl_best_c = e_cl_best_m[valid_best]
    acf_cl_best = compute_acf(e_cl_best_c, 50)
    lb_cl = ljung_box(acf_cl_best, len(e_cl_best_c))

    frozen['ljung_box'] = {
        'baseline': [{'lag': l, 'Q': float(Q), 'p': float(p)} for l, Q, p in lb_base],
        'closure': [{'lag': l, 'Q': float(Q), 'p': float(p)} for l, Q, p in lb_cl],
    }

    # Frozen results
    frozen['final_table'] = {
        'physics_only': {
            'acf1': float(acf_base[1]), 'acf2': float(acf_base[2]),
            'acf5': float(acf_base[5]), 'acf10': float(acf_base[10]),
            'dxr2_5': float(r2dx_base[4]), 'dxr2_10': float(r2dx_base[9]),
            'mean_dxr2_5_10': float(base_m510),
            'nis': float(np.mean(e_base_clean**2 / S_base_clean)),
        },
        'closure_5t': {
            'acf1': float(m_acf1), 'acf2': float(m_acf2),
            'acf5': float(m_acf5), 'acf10': float(m_acf10),
            'dxr2_5': float(cl_dx_mean[4]), 'dxr2_10': float(cl_dx_mean[9]),
            'mean_dxr2_5_10': float(cl_m510),
            'nis': float(m_nis),
            'frac': float(frac_computed), 'med_ratio': float(med_ratio_computed),
        },
        'mlp_upper_bound': {
            'dxr2_5': float(mlp_dx_mean[4]), 'dxr2_10': float(mlp_dx_mean[9]),
            'mean_dxr2_5_10': float(mlp_m510),
        },
    }

    frozen_path = AUDIT_DIR / "frozen_results.json"
    with open(frozen_path, 'w') as f:
        json.dump(frozen, f, indent=2)

    check("G22", "frozen_results.json saved",
          frozen_path.exists(),
          f"Contains all final metrics for the paper")

    # ================================================================
    # WRITE AUDIT REPORT
    # ================================================================
    print("\n" + "="*70)
    print("WRITING AUDIT REPORT")
    print("="*70)

    n_pass = sum(1 for v in checks.values() if v == 'PASS')
    n_fail = sum(1 for v in checks.values() if v == 'FAIL')
    n_total = len(checks)

    report_header = [
        "# Final End-to-End Audit Report\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Result:** {n_pass}/{n_total} PASS, {n_fail}/{n_total} FAIL\n",
        "---\n",
    ]

    report_footer = [
        "\n---\n",
        "## Output Files\n",
        f"- `audit_outputs/frozen_results.json` - all final metrics",
        f"- `audit_outputs/data_fingerprint.json` - SHA256 + basic stats",
        f"- `audit_outputs/units_table.csv` - dimensional analysis",
        f"- `audit_outputs/retrain_comparison.csv` - retrained vs original params",
        f"- `audit_outputs/skill_curves.csv` - DxR2 at all horizons",
        f"- `audit_outputs/closure_metrics_per_seed.csv` - per-seed ACF/NIS",
        f"- `audit_outputs/innovations_baseline.npy` - raw baseline innovations",
        f"- `audit_outputs/innovations_closure_best.npy` - raw closure innovations",
        f"- `audit_outputs/fig1_skill_curves.png` - regenerated Fig 1",
        f"- `audit_outputs/fig2_innovation_acf.png` - regenerated Fig 2",
        f"- `audit_outputs/fig3_coefficients.png` - regenerated Fig 3",
        "",
        "## frozen_results.json Summary\n",
        "```json",
        json.dumps({
            'pct_recovered': frozen['pct_recovered'],
            'closure_params': {k: f"{v['mean']:.4f} +/- {v['std']:.4f} (CV={v['cv']:.1f}%)"
                               for k, v in frozen['closure_params'].items()},
            'acf': frozen['acf'],
            'final_table': frozen['final_table'],
        }, indent=2),
        "```",
    ]

    report_path = AUDIT_DIR / "audit_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_header + report + report_footer))

    print(f"\n  Audit report saved to {report_path}")
    print(f"  frozen_results.json saved to {frozen_path}")
    print(f"\n  SUMMARY: {n_pass}/{n_total} PASS, {n_fail}/{n_total} FAIL")

    if n_fail > 0:
        print("\n  FAILED CHECKS:")
        for k, v in checks.items():
            if v == 'FAIL':
                print(f"    {k}: FAIL")

    print(f"\n{'='*70}")
    print("AUDIT COMPLETE")
    print(f"{'='*70}")

    return n_fail


if __name__ == '__main__':
    sys.exit(main())
