"""
Diagnostic 8: Cross-Correlation of Innovation with Inputs
=========================================================
NO TRAINING. Compute CCF(e_t, u_{t-lag}) for lag in [-50, +50].
Also CCF with du and energy proxy E.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag08")
MAX_LAG = 50
EWMA_ALPHA = 0.1

print("=" * 60)
print("DIAG 08: Innovation-Input Cross-Correlation")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

D = U.load_final_model_and_data()
v_test = D['v_test']
filt = D['filter']
test_mask = D['test_mask']

# Extract test innovations
innov = filt['innovations'][test_mask]
valid = ~np.isnan(innov)
e = innov[valid]
v = v_test[valid]

# Velocity difference
dv = np.diff(v, prepend=v[0])

# Energy proxy: EWMA of v^2
E = np.zeros(len(v))
E[0] = v[0]**2
for i in range(1, len(v)):
    E[i] = EWMA_ALPHA * v[i]**2 + (1 - EWMA_ALPHA) * E[i-1]

# Compute CCFs
print("  Computing CCF(e, v)...")
ccf_ev, lags = U.compute_ccf(e, v, MAX_LAG)
print("  Computing CCF(e, dv)...")
ccf_edv, _ = U.compute_ccf(e, dv, MAX_LAG)
print("  Computing CCF(e, E)...")
ccf_eE, _ = U.compute_ccf(e, E, MAX_LAG)

sig_band = 1.96 / np.sqrt(len(e))

# Find peaks
for name, ccf in [('e,v', ccf_ev), ('e,dv', ccf_edv), ('e,E', ccf_eE)]:
    peak_idx = np.argmax(np.abs(ccf))
    peak_lag = lags[peak_idx]
    peak_val = ccf[peak_idx]
    print(f"  CCF({name}): peak lag={peak_lag}, value={peak_val:+.4f}, "
          f"{'significant' if abs(peak_val) > sig_band else 'not significant'}")

# Figures
lag_s = lags * U.DT  # convert to seconds

for ccf, ylabel, title, fname in [
    (ccf_ev, 'CCF(e, v)', 'Innovation vs Water Velocity', 'ccf_e_u.png'),
    (ccf_edv, 'CCF(e, dv)', 'Innovation vs Velocity Change', 'ccf_e_du.png'),
    (ccf_eE, 'CCF(e, E)', 'Innovation vs Energy Proxy', 'ccf_e_E.png'),
]:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lag_s, ccf, width=U.DT*0.8, color='steelblue', alpha=0.7, edgecolor='k', lw=0.3)
    ax.axhline(sig_band, color='red', ls='--', lw=0.8, label=f'+/-1.96/sqrt(N)')
    ax.axhline(-sig_band, color='red', ls='--', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} (N={len(e)})')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")

# Summary
def peak_info(ccf, lags):
    idx = np.argmax(np.abs(ccf))
    return {'peak_lag': int(lags[idx]),
            'peak_lag_s': float(lags[idx] * U.DT),
            'peak_value': float(ccf[idx]),
            'significant': bool(abs(ccf[idx]) > sig_band)}

summary = {
    'n_innovations': len(e),
    'sig_band_95': float(sig_band),
    'ccf_e_v': peak_info(ccf_ev, lags),
    'ccf_e_dv': peak_info(ccf_edv, lags),
    'ccf_e_E': peak_info(ccf_eE, lags),
    'systematic_forcing_error': any(
        abs(ccf_ev[i]) > 2 * sig_band
        for i in range(len(ccf_ev))
        if abs(lags[i]) <= 5
    ),
    'runtime_s': time.time() - t0,
}
with open(OUT / "summary_diag08.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag08 complete ({time.time()-t0:.0f}s). Output: {OUT}")
