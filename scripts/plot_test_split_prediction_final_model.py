"""
Plot Full Test Split: 1-Step-Ahead + Open-Loop h=50 Prediction Overlay
======================================================================
NO TRAINING. Loads existing checkpoints and runs forward filter only.

Shows the final paper model (Closure 2t) on the full test split:
  - Axis 1: true x vs predicted x_hat (1-step-ahead) + uncertainty
           + open-loop h=50 forecasts (oracle v and persistence v)
  - Axis 2: input velocity
  - Axis 3: innovation e_t
  - Figure 2: zoom around biggest transition event

Usage: python scripts/plot_test_split_prediction_final_model.py
"""

import os, sys, math, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_closure import CLOSURE_PARAM_NAMES

# ===== Paths =====
DATA_DIR = ROOT / "processed_data_10hz"
V2_CKPT  = ROOT / "final_lockbox_v2" / "checkpoints"
S1_CKPT  = (ROOT / "model_upgrade_round2_neural_residual" / "checkpoints"
             / "stage1_physics_only.pth")

OUT = ROOT / "final_lockbox_vX_visual_checks"
OUT.mkdir(parents=True, exist_ok=True)

DT = 0.1
SEEDS_REF = [42, 43, 44]


# ============================================================
#  HELPERS
# ============================================================

def zero_closure():
    cl = {k: 0.0 for k in CLOSURE_PARAM_NAMES}
    cl['q_scale'] = 1.0
    return cl


def load_s1_params():
    ck = torch.load(S1_CKPT, map_location='cpu', weights_only=False)
    return ck['params']


def load_closure_params(seed):
    ck = torch.load(V2_CKPT / f"closure_2t_s{seed}.pth",
                    map_location='cpu', weights_only=False)
    return ck['closure']


def kf_filter_full(params, cl_params, t, x_obs, v):
    """KF forward pass returning predictions, innovations, uncertainty, states."""
    N = len(x_obs)
    x_pred   = np.full(N, np.nan)   # 1-step-ahead predicted x (prior)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)    # prediction variance (includes R)
    states_x = np.zeros(N)           # post-update x
    states_u = np.zeros(N)           # post-update u

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]

    for k in range(1, N):
        dt = t[k] - t[k-1]
        if dt <= 0: dt = 0.1
        rho_u = math.exp(-alpha * dt)
        g = max(v[k-1]**2 - vc**2, 0.0)

        u_st, v_w = s[1], v[k-1]
        dv_w = v[k-1] - v[k-2] if k >= 2 else 0.0
        cl = b2_v * dv_w - d2 * u_st * abs(v_w)
        cl_dt = cl * dt

        xp = s[0] + s[1] * dt
        up = rho_u * s[1] - kap * s[0] * dt + c_val * g * dt + cl_dt
        s_pred = np.array([xp, up])

        x_pred[k] = xp

        F_mat = np.array([[1, dt], [-kap*dt, rho_u]])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - xp
        S_val = P_pred[0, 0] + R
        innovations[k] = innov
        S_values[k] = S_val

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(2) - np.outer(K, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)

        states_x[k] = s[0]; states_u[k] = s[1]

    return x_pred, innovations, S_values, states_x, states_u


def predict_step(sx, su, v_w, dv_w, dt_k, params, cl_params):
    """Single open-loop predict step (no measurement update)."""
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    b2_v = cl_params.get('b2', 0.0)
    d2 = cl_params.get('d2', 0.0)

    rho = math.exp(-alpha * dt_k)
    g = max(v_w**2 - vc**2, 0.0)
    cl = b2_v * dv_w - d2 * su * abs(v_w)

    x_new = sx + su * dt_k
    u_new = rho * su - kap * sx * dt_k + c_val * g * dt_k + cl * dt_k
    return x_new, u_new


def open_loop_rollout(i_start, h, states_x_full, states_u_full,
                      t_full, v_full, params, cl_params, mode='oracle'):
    """h-step open-loop rollout from filtered state at i_start.

    mode='oracle': use true future v at each step
    mode='persistence': use v at launch time for all steps, dv=0
    """
    sx, su = states_x_full[i_start], states_u_full[i_start]
    v_persist = v_full[i_start] if i_start < len(v_full) else 0.0
    path_x = [sx]

    for step in range(h):
        k = i_start + step + 1
        if k >= len(t_full):
            break
        dt_k = t_full[k] - t_full[k - 1]
        if dt_k <= 0:
            dt_k = 0.1

        if mode == 'oracle':
            v_w = v_full[k - 1]
            dv_w = v_full[k - 1] - v_full[k - 2] if k >= 2 else 0.0
        else:  # persistence
            v_w = v_persist
            dv_w = 0.0

        sx, su = predict_step(sx, su, v_w, dv_w, dt_k, params, cl_params)
        path_x.append(sx)

    return path_x


# ============================================================
#  MAIN
# ============================================================

H_OL = 50  # open-loop horizon (50 steps = 5.0s at 10 Hz)

print("=" * 60)
print("NO TRAINING -- open-loop rollout from saved checkpoint only")
print("=" * 60)
print()
print("Loading parameters...")

s1_params = load_s1_params()
print(f"  S1 physics: alpha={s1_params['alpha']:.4f}, c={s1_params['c']:.4f}, "
      f"kappa={s1_params['kappa']:.4f}")

# Average closure params across 3 seeds
cl_avg = zero_closure()
for key in ['b2', 'd2', 'q_scale']:
    vals = [load_closure_params(s)[key] for s in SEEDS_REF]
    cl_avg[key] = float(np.mean(vals))
print(f"  Closure (2t): b2={cl_avg['b2']:.4f}, d2={cl_avg['d2']:.4f}, "
      f"q_scale={cl_avg['q_scale']:.4f}")

# Load data
df_train = pd.read_csv(DATA_DIR / "train_10hz_ready.csv")
df_val   = pd.read_csv(DATA_DIR / "val_10hz_ready.csv")
df_test  = pd.read_csv(DATA_DIR / "test_10hz_ready.csv")

TEST_START = df_test['timestamp'].iloc[0]
df_dev = df_val[df_val['timestamp'] < TEST_START].copy()

# 50s warmup from end of dev
warmup_start = df_dev.timestamp.max() - 50.0
test_warmup = df_dev[df_dev['timestamp'] >= warmup_start].copy()
df_filter = pd.concat([test_warmup, df_test], ignore_index=True)
test_mask = df_filter['timestamp'].values >= TEST_START

t_arr = df_filter['timestamp'].values
x_arr = df_filter['displacement'].values
v_arr = df_filter['velocity'].values

print(f"  Filter array: {len(t_arr)} pts (warmup+test)")
print(f"  Test mask: {test_mask.sum()} pts, t=[{t_arr[test_mask][0]:.1f}, {t_arr[test_mask][-1]:.1f}]")

# Run forward filter (NO TRAINING)
print("Running forward filter...")
x_pred, innovations, S_values, states_x, states_u = kf_filter_full(
    s1_params, cl_avg, t_arr, x_arr, v_arr)

# Extract test portion
t_test = t_arr[test_mask]
x_true = x_arr[test_mask]
x_hat  = x_pred[test_mask]
innov  = innovations[test_mask]
S_test = S_values[test_mask]
v_test = v_arr[test_mask]

# Uncertainty: +/- 2*sqrt(S) where S = P_pred[0,0] + R
sigma = np.sqrt(S_test)
x_upper = x_hat + 2 * sigma
x_lower = x_hat - 2 * sigma

valid = ~np.isnan(x_hat)
print(f"  Valid predictions: {valid.sum()} / {len(x_hat)}")

# ============================================================
#  OPEN-LOOP h=50 FORECASTS
# ============================================================
print(f"\nComputing open-loop h={H_OL} forecasts (oracle + persistence)...")

# Map test indices back to full array indices
test_idx_full = np.where(test_mask)[0]  # indices into t_arr/x_arr/v_arr
N_test = len(test_idx_full)

# For h=50 from test index i, we need i+50 to still be within the full array
x_ol_oracle = np.full(N_test, np.nan)
x_ol_persist = np.full(N_test, np.nan)

n_valid_ol = 0
for j in range(N_test):
    i_full = test_idx_full[j]
    # Need H_OL more steps in the full array
    if i_full + H_OL >= len(t_arr):
        continue

    # Oracle rollout
    path_oracle = open_loop_rollout(
        i_full, H_OL, states_x, states_u, t_arr, v_arr,
        s1_params, cl_avg, mode='oracle')
    if len(path_oracle) == H_OL + 1:
        x_ol_oracle[j] = path_oracle[-1]

    # Persistence rollout
    path_persist = open_loop_rollout(
        i_full, H_OL, states_x, states_u, t_arr, v_arr,
        s1_params, cl_avg, mode='persistence')
    if len(path_persist) == H_OL + 1:
        x_ol_persist[j] = path_persist[-1]
        n_valid_ol += 1

print(f"  Valid h={H_OL} forecasts: {n_valid_ol} / {N_test}")

# The target time for each h=50 prediction is t[j] + 50*dt ~ t[j+50]
# x_ol_oracle[j] predicts x at time t_test[j] + 5.0s
# We plot them at their TARGET time (t + H_OL*0.1)
t_ol_target = t_test + H_OL * DT  # approximate target times

# Compute DxR2@50 and MAE@50
# DxR2: compare dx_pred = x_ol - x_obs[t] vs dx_obs = x_obs[t+H] - x_obs[t]
valid_ol = ~np.isnan(x_ol_oracle)
dx_pred_oracle = x_ol_oracle[valid_ol] - x_true[valid_ol]
dx_pred_persist = x_ol_persist[valid_ol] - x_true[valid_ol]

# True increments: x[t+H] - x[t]
# Need actual x at time t + H_OL steps
x_true_target = np.full(N_test, np.nan)
for j in range(N_test):
    i_full = test_idx_full[j]
    if i_full + H_OL < len(x_arr):
        x_true_target[j] = x_arr[i_full + H_OL]

dx_obs = x_true_target[valid_ol] - x_true[valid_ol]

ss_tot = np.sum((dx_obs - np.mean(dx_obs))**2)

# Oracle metrics
ss_res_oracle = np.sum((dx_obs - dx_pred_oracle)**2)
dxr2_oracle = 1.0 - ss_res_oracle / ss_tot if ss_tot > 1e-15 else 0.0
mae_oracle = np.mean(np.abs(x_true_target[valid_ol] - x_ol_oracle[valid_ol]))

# Persistence-v metrics
ss_res_persist = np.sum((dx_obs - dx_pred_persist)**2)
dxr2_persist = 1.0 - ss_res_persist / ss_tot if ss_tot > 1e-15 else 0.0
mae_persist = np.mean(np.abs(x_true_target[valid_ol] - x_ol_persist[valid_ol]))

print(f"\n  === h={H_OL} Open-Loop Metrics ===")
print(f"  Oracle v:      DxR2@{H_OL} = {dxr2_oracle:+.4f},  MAE = {mae_oracle:.6f} m")
print(f"  Persistence v: DxR2@{H_OL} = {dxr2_persist:+.4f},  MAE = {mae_persist:.6f} m")

# ============================================================
#  DETECT TRANSITION EVENTS (for vertical lines)
# ============================================================
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, random_state=42, n_init=10)
km.fit(x_true.reshape(-1, 1))
centroids = sorted(km.cluster_centers_.flatten())
midpoint = (centroids[0] + centroids[1]) / 2.0
hyst_band = 0.1 * (centroids[1] - centroids[0])
thresh_up = midpoint + hyst_band
thresh_down = midpoint - hyst_band

N_t = len(x_true)
states_ev = np.zeros(N_t, dtype=int)
states_ev[0] = 0 if x_true[0] < midpoint else 1
for i in range(1, N_t):
    if states_ev[i-1] == 0:
        states_ev[i] = 1 if x_true[i] > thresh_up else 0
    else:
        states_ev[i] = 0 if x_true[i] < thresh_down else 1

event_times = [t_test[i] for i in range(1, N_t) if states_ev[i] != states_ev[i-1]]
print(f"  Detected {len(event_times)} transitions in test period")

# ============================================================
#  PLOT
# ============================================================
print("Plotting...")

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1, 1.5]})
fig.suptitle('Final Paper Model: Closure (2t) -- 1-Step + Open-Loop h=50 on Test Split\n'
             f'b2={cl_avg["b2"]:.3f}, d2={cl_avg["d2"]:.3f}, q_scale={cl_avg["q_scale"]:.3f} '
             f'| NO TRAINING -- checkpoint evaluation only',
             fontsize=11, y=0.98)

# ── Axis 1: Displacement overlay ──
ax = axes[0]
ax.fill_between(t_test[valid], x_lower[valid], x_upper[valid],
                alpha=0.15, color='dodgerblue', label=r'$\hat{x} \pm 2\sigma$')
ax.plot(t_test, x_true, 'k-', lw=0.6, alpha=0.9, label=r'True $x_t$')
ax.plot(t_test[valid], x_hat[valid], '-', color='dodgerblue', lw=0.6,
        alpha=0.8, label=r'Predicted $\hat{x}_t$ (1-step)')

# h=50 open-loop markers (plotted at target time t+5s)
ol_valid = valid_ol & (t_ol_target <= t_test[-1])  # only plot within test range
ax.scatter(t_ol_target[ol_valid], x_ol_oracle[ol_valid],
           s=3, color='red', alpha=0.25, zorder=5,
           label=f'h={H_OL} oracle v (DxR2={dxr2_oracle:+.3f})')
ax.scatter(t_ol_target[ol_valid], x_ol_persist[ol_valid],
           s=3, color='green', alpha=0.25, zorder=4,
           label=f'h={H_OL} persist v (DxR2={dxr2_persist:+.3f})')

ax.axhline(midpoint, color='gray', ls=':', lw=0.7, alpha=0.5)
for et in event_times:
    ax.axvline(et, color='lightgray', ls='-', lw=0.4, alpha=0.6)
ax.set_ylabel('Displacement (m)')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
ax.set_title(f'Displacement: True vs 1-Step + Open-Loop h={H_OL}', fontsize=10)

# ── Axis 2: Velocity ──
ax = axes[1]
ax.plot(t_test, v_test, color='steelblue', lw=0.4, alpha=0.8)
for et in event_times:
    ax.axvline(et, color='lightgray', ls='-', lw=0.4, alpha=0.6)
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Input: Water Velocity', fontsize=10)

# ── Axis 3: Innovations ──
ax = axes[2]
ax.plot(t_test[valid], innov[valid], color='darkorange', lw=0.4, alpha=0.8)
ax.axhline(0, color='gray', ls='--', lw=0.5)
# +/- 2*sigma band for innovations
ax.fill_between(t_test[valid], -2*sigma[valid], 2*sigma[valid],
                alpha=0.1, color='darkorange')
for et in event_times:
    ax.axvline(et, color='lightgray', ls='-', lw=0.4, alpha=0.6)
ax.set_ylabel('Innovation $e_t$ (m)')
ax.set_xlabel('Time (s)')
ax.set_title('Innovation (observation - prediction)', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save Figure 1
png_path = OUT / "plot_test_full_overlay.png"
plt.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"Saved: {png_path}")

# ============================================================
#  FIGURE 2: Event-Window Zoom (+/- 10s around biggest transition)
# ============================================================
if len(event_times) > 0:
    print("\nCreating event-window zoom figure...")

    # Find the biggest transition (largest absolute displacement jump)
    jump_sizes = []
    for et in event_times:
        idx = np.argmin(np.abs(t_test - et))
        if idx > 0:
            jump_sizes.append(abs(x_true[idx] - x_true[idx - 1]))
        else:
            jump_sizes.append(0.0)
    biggest_idx = np.argmax(jump_sizes)
    t_event = event_times[biggest_idx]
    t_zoom_lo = t_event - 10.0
    t_zoom_hi = t_event + 10.0

    zoom_mask = (t_test >= t_zoom_lo) & (t_test <= t_zoom_hi)
    zoom_ol = ol_valid & (t_ol_target >= t_zoom_lo) & (t_ol_target <= t_zoom_hi)

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 5))
    ax2.plot(t_test[zoom_mask], x_true[zoom_mask], 'k-', lw=1.2, alpha=0.9,
             label=r'True $x_t$')

    valid_zoom = valid & zoom_mask
    ax2.plot(t_test[valid_zoom], x_hat[valid_zoom], '-', color='dodgerblue',
             lw=1.0, alpha=0.8, label=r'$\hat{x}_t$ (1-step)')
    ax2.fill_between(t_test[valid_zoom], x_lower[valid_zoom], x_upper[valid_zoom],
                     alpha=0.15, color='dodgerblue')

    ax2.scatter(t_ol_target[zoom_ol], x_ol_oracle[zoom_ol],
                s=12, color='red', alpha=0.5, zorder=5,
                label=f'h={H_OL} oracle v')
    ax2.scatter(t_ol_target[zoom_ol], x_ol_persist[zoom_ol],
                s=12, color='green', alpha=0.5, zorder=4,
                label=f'h={H_OL} persist v')

    ax2.axvline(t_event, color='gray', ls='--', lw=1.0, alpha=0.7,
                label='Transition')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement (m)')
    ax2.set_title(f'Event-Window Zoom: t = {t_event:.1f}s +/- 10s  |  '
                  f'h={H_OL} open-loop forecasts', fontsize=11)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.set_xlim(t_zoom_lo, t_zoom_hi)
    plt.tight_layout()

    png_zoom = OUT / "plot_event_zoom_h50.png"
    plt.savefig(png_zoom, dpi=200, bbox_inches='tight')
    print(f"Saved: {png_zoom}")

# README
readme = f"""# Visual Check: Full Test Split Prediction Overlay + Open-Loop h={H_OL}

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** Closure (2t) -- final paper model
**Prediction modes:**
- 1-step-ahead filter prediction (prior x_pred before Kalman update)
- Open-loop h={H_OL} forecast from filtered posterior state (no measurement update)

## Parameters loaded (NO TRAINING)

- Stage-1 physics checkpoint: `{S1_CKPT.relative_to(ROOT)}`
- Closure checkpoints: `{V2_CKPT.relative_to(ROOT)}/closure_2t_s{{42,43,44}}.pth` (averaged)
- b2 = {cl_avg['b2']:.4f}, d2 = {cl_avg['d2']:.4f}, q_scale = {cl_avg['q_scale']:.4f}

## h={H_OL} Open-Loop Metrics

| Mode | DxR2@{H_OL} | MAE (m) |
|------|------------|---------|
| Oracle v | {dxr2_oracle:+.4f} | {mae_oracle:.6f} |
| Persistence v | {dxr2_persist:+.4f} | {mae_persist:.6f} |

## Confirmation

**No training performed.** This script loads existing checkpoints and runs
only the forward Kalman filter on the test split (with 50s dev warmup).
Open-loop rollouts use the filtered posterior mean [x, u] at each launch time.

## Velocity convention

- **Oracle v:** True future velocity used at each rollout step (matches paper's DxR2 metric)
- **Persistence v:** Velocity frozen at launch-time value; dv=0 throughout rollout

## Plot contents

### Figure 1: Full test split
- **Axis 1:** True displacement (black) vs 1-step predicted (blue) + h={H_OL} forecasts
  (red=oracle, green=persistence). Gray vertical lines mark transitions.
- **Axis 2:** Input water velocity.
- **Axis 3:** Innovation sequence e_t = x_obs - x_pred, with +/-2sigma band.

### Figure 2: Event-window zoom
- +/-10s around the biggest displacement transition
- Same overlays at larger marker size for readability
"""

with open(OUT / "README.md", 'w') as f:
    f.write(readme)
print(f"Saved: {OUT / 'README.md'}")

# Interactive display
print("\nOpening interactive plots (close windows to exit)...")
plt.show(block=True)
