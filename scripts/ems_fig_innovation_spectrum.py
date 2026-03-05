"""
Generate fig_innovation_spectrum.pdf: Innovation PSD + coherence with water velocity.

Loads frozen v11.1 checkpoints (seed 1), runs KF filter sequentially over the
test set (with val warmup), extracts innovations, then computes:
  Panel (a): Welch PSD of innovations for physics-only vs closure
  Panel (b): Coherence between innovations and water velocity

Output: final_lockbox_v11_1_alpha_fix/seed1/figures/fig_innovation_spectrum.pdf

NO retraining. Read-only from checkpoints + clean data.
"""
import sys, os, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.signal import welch, coherence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kalman_forecaster import KalmanForecaster
from models.kalman_closure import KalmanForecasterClosure

# ---- Paths ----
CLEAN_DIR = ROOT / "processed_data_10hz_clean_v1"
CKPT_DIR = ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "checkpoints"
FIG_DIR = ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

S1_CKPT = CKPT_DIR / "stage1_physics_seed1.pth"
S2_CKPT = CKPT_DIR / "closure_2t_seed1.pth"

WARMUP_STEPS = 500  # use last 500 val steps for warmup
FS = 10.0           # sampling frequency (Hz)
NPERSEG = 256       # Welch segment length

device = torch.device('cpu')

# ---- Load data ----
print("Loading data...")
df_val = pd.read_csv(CLEAN_DIR / "val_10hz_ready.csv")
df_test = pd.read_csv(CLEAN_DIR / "test_10hz_ready.csv")

v_warmup = df_val['velocity'].values[-WARMUP_STEPS:].astype(np.float32)
x_warmup = df_val['displacement'].values[-WARMUP_STEPS:].astype(np.float32)
dt_warmup = df_val['time_delta'].values[-WARMUP_STEPS:].astype(np.float32)

v_test = df_test['velocity'].values.astype(np.float32)
x_test = df_test['displacement'].values.astype(np.float32)
dt_test = df_test['time_delta'].values.astype(np.float32)

# Concatenate warmup + test
v_all = np.concatenate([v_warmup, v_test])
x_all = np.concatenate([x_warmup, x_test])
dt_all = np.concatenate([dt_warmup, dt_test])
N_total = len(v_all)
print(f"  warmup={WARMUP_STEPS}, test={len(v_test)}, total={N_total}")


def run_kf_s1(model, v_all, x_all, dt_all, warmup_n):
    """Run S1 (physics-only) KF filter sequentially, return test innovations."""
    model.eval()
    innovations = []
    innov_vars = []

    with torch.no_grad():
        s = torch.zeros(1, 2, device=device)
        s[0, 0] = float(x_all[0])
        P = model.P0.unsqueeze(0).clone()

        for k in range(1, len(v_all)):
            v_k = torch.tensor([float(v_all[k - 1])], device=device)
            dt_k = torch.tensor([float(dt_all[k])], device=device).clamp(min=1e-6)
            y_k = float(x_all[k])

            s, P = model.kf_predict(s, P, v_k, dt_k)

            innov = y_k - s[0, 0].item()
            S = (P[0, 0, 0] + model.R).item()
            innovations.append(innov)
            innov_vars.append(S)

            y_t = torch.tensor([y_k], device=device)
            s, P = model.kf_update(s, P, y_t)

    # Discard warmup
    return np.array(innovations[warmup_n:]), np.array(innov_vars[warmup_n:])


def run_kf_s2(model, v_all, x_all, dt_all, warmup_n):
    """Run S2 (closure) KF filter sequentially, return test innovations."""
    model.eval()
    innovations = []
    innov_vars = []

    with torch.no_grad():
        s = torch.zeros(1, 2, device=device)
        s[0, 0] = float(x_all[0])
        P = model.P0.unsqueeze(0).clone()

        for k in range(1, len(v_all)):
            v_curr = torch.tensor([float(v_all[k - 1])], device=device)
            v_prev = torch.tensor([float(v_all[k - 2])], device=device) if k >= 2 else v_curr
            dv = v_curr - v_prev if k >= 2 else torch.zeros_like(v_curr)
            dt_k = torch.tensor([float(dt_all[k])], device=device).clamp(min=1e-6)
            y_k = float(x_all[k])

            s, P = model.kf_predict(s, P, v_curr, dv, dt_k)

            innov = y_k - s[0, 0].item()
            S = (P[0, 0, 0] + model.R).item()
            innovations.append(innov)
            innov_vars.append(S)

            y_t = torch.tensor([y_k], device=device)
            s, P = model.kf_update(s, P, y_t)

    return np.array(innovations[warmup_n:]), np.array(innov_vars[warmup_n:])


# ---- Load S1 model ----
print("Loading S1 (physics-only) model...")
s1_ckpt = torch.load(S1_CKPT, map_location=device, weights_only=False)
s1_model = KalmanForecaster(use_kappa=True).to(device)
s1_model.load_state_dict(s1_ckpt['state_dict'])
s1_model.eval()
s1_params = s1_model.param_summary()
print(f"  S1: alpha={s1_params['alpha']:.4f}, kappa={s1_params['kappa']:.4f}, "
      f"c={s1_params['c']:.4f}")

# ---- Load S2 model ----
print("Loading S2 (closure) model...")
s2_ckpt = torch.load(S2_CKPT, map_location=device, weights_only=False)
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
print(f"  S2: d2={s2_params['d2']:.4f}, b2={s2_params['b2']:.4f}, "
      f"q_scale={s2_params['q_scale']:.4f}")

# ---- Run filters ----
print("Running S1 filter...")
innov_s1, var_s1 = run_kf_s1(s1_model, v_all, x_all, dt_all, WARMUP_STEPS)
print(f"  S1 innovations: n={len(innov_s1)}, std={np.std(innov_s1):.6f}")

print("Running S2 filter...")
innov_s2, var_s2 = run_kf_s2(s2_model, v_all, x_all, dt_all, WARMUP_STEPS)
print(f"  S2 innovations: n={len(innov_s2)}, std={np.std(innov_s2):.6f}")

# ---- Compute Welch PSD ----
print("Computing Welch PSD...")
freq_s1, psd_s1 = welch(innov_s1, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
freq_s2, psd_s2 = welch(innov_s2, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
freq_v, psd_v = welch(v_test[:len(innov_s1)], fs=FS, nperseg=NPERSEG,
                       noverlap=NPERSEG // 2)

# ---- Compute coherence ----
print("Computing coherence with water velocity...")
v_for_coh = v_test[:len(innov_s1)]
freq_c1, coh_s1 = coherence(innov_s1, v_for_coh, fs=FS, nperseg=NPERSEG,
                              noverlap=NPERSEG // 2)
freq_c2, coh_s2 = coherence(innov_s2, v_for_coh, fs=FS, nperseg=NPERSEG,
                              noverlap=NPERSEG // 2)

# ---- PSD ratio (closure reduction) ----
psd_ratio = psd_s2 / np.maximum(psd_s1, 1e-30)

# ---- Figure ----
print("Generating figure...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# Panel (a): Innovation PSD
ax = axes[0]
ax.semilogy(freq_s1, psd_s1, color='#2166AC', linewidth=1.2, label='Physics-only')
ax.semilogy(freq_s2, psd_s2, color='#B2182B', linewidth=1.2, label='Closure')
ax.semilogy(freq_v, psd_v * 1e-4, color='0.6', linewidth=0.8, linestyle='--',
            label=r'$u(t)$ PSD ($\times 10^{-4}$)')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'PSD (m$^2$/Hz)')
ax.set_title('(a) Innovation power spectral density')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([0, FS / 2])
ax.grid(True, alpha=0.3)

# Panel (b): PSD ratio
ax = axes[1]
ax.plot(freq_s1, psd_ratio, color='#4DAF4A', linewidth=1.2)
ax.axhline(1.0, color='0.5', linewidth=0.8, linestyle='--')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD ratio (closure / physics)')
ax.set_title('(b) Spectral reduction')
ax.set_xlim([0, FS / 2])
ax.set_ylim([0, max(1.5, np.percentile(psd_ratio, 99) * 1.1)])
ax.grid(True, alpha=0.3)
# Shade reduction zone
mask = psd_ratio < 1.0
ax.fill_between(freq_s1, 0, 1, where=mask, alpha=0.08, color='#4DAF4A')
ax.text(0.95, 0.95, f'Mean ratio = {np.mean(psd_ratio):.3f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Panel (c): Coherence
ax = axes[2]
ax.plot(freq_c1, coh_s1, color='#2166AC', linewidth=1.2, label='Physics-only')
ax.plot(freq_c2, coh_s2, color='#B2182B', linewidth=1.2, label='Closure')
# 95% significance line for coherence (approximate: 1 - 0.05^(2/(K-2)) where K=segments)
K = 2 * len(innov_s1) // NPERSEG - 1
if K > 2:
    sig_95 = 1.0 - 0.05 ** (2.0 / (K - 2))
    ax.axhline(sig_95, color='0.5', linewidth=0.8, linestyle=':',
               label=f'95% sig. ({sig_95:.3f})')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Coherence $\gamma^2$')
ax.set_title(r'(c) Coherence with $u(t)$')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([0, FS / 2])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_pdf = FIG_DIR / "fig_innovation_spectrum.pdf"
out_png = FIG_DIR / "fig_innovation_spectrum.png"
fig.savefig(str(out_pdf), dpi=300, bbox_inches='tight')
fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")

# ---- Summary stats ----
# Band-averaged PSD reduction
low_band = (freq_s1 > 0) & (freq_s1 <= 1.0)
mid_band = (freq_s1 > 1.0) & (freq_s1 <= 3.0)
high_band = (freq_s1 > 3.0)

print("\n--- PSD Summary ---")
for name, mask in [("0-1 Hz", low_band), ("1-3 Hz", mid_band), (">3 Hz", high_band)]:
    if mask.any():
        ratio_band = np.mean(psd_ratio[mask])
        reduction_pct = (1 - ratio_band) * 100
        print(f"  {name}: mean ratio={ratio_band:.3f} ({reduction_pct:+.1f}% reduction)")

print(f"\n--- Coherence Summary ---")
print(f"  Physics-only: mean coh={np.mean(coh_s1):.4f}, max coh={np.max(coh_s1):.4f}")
print(f"  Closure:      mean coh={np.mean(coh_s2):.4f}, max coh={np.max(coh_s2):.4f}")
print(f"  Delta mean coh: {np.mean(coh_s2) - np.mean(coh_s1):+.4f}")

print("\nDone.")
