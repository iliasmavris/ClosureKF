"""
Diagnostic 4: Event Timing Score (Classification)
==================================================
NO TRAINING. For h in {10, 50}, predict whether displacement will cross
threshold within next h steps. Use Gaussian x_hat +/- sigma for event
probability. Compute ROC AUC, PR AUC, F1.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import utils_no_train as U

plt = U.setup_plotting()
OUT = U.ensure_output_dir("diag04")
EVAL_HORIZONS = [10, 50]

print("=" * 60)
print("DIAG 04: Event Timing Score (Classification)")
print("NO TRAINING -- checkpoint evaluation only")
print("=" * 60)
t0 = time.time()

D = U.load_final_model_and_data()
params, cl = D['params'], D['cl_params']
t_arr, x_arr, v_arr = D['t_arr'], D['x_arr'], D['v_arr']
test_idx = D['test_idx_full']
x_test, t_test = D['x_test'], D['t_test']
filt = D['filter']
N_test = D['N_test']
N_full = len(t_arr)

# Event detection
ev = U.detect_events_from_x(x_test, t_test)
event_times = ev['event_times'] or []
states = ev['states']
midpoint = ev['midpoint']
thresh_up = ev['thresh_up']
thresh_down = ev['thresh_down']
print(f"  {len(event_times)} transitions, midpoint={midpoint:.4f}")

# Get steady-state P_post (converged; take from first test point)
i0 = test_idx[0]
P_steady = filt['P_post'][i0]
R = params['R']

# Precompute covariance trajectory for each horizon
max_h = max(EVAL_HORIZONS)
Pxx_traj = U.propagate_cov_trajectory(P_steady, max_h, params, cl)
# Pxx_traj[h-1] = P_pred_xx + R at horizon h

all_rows = []
for H in EVAL_HORIZONS:
    print(f"\n  --- h={H} ({H*U.DT:.1f}s) ---")
    sigma_h = np.sqrt(Pxx_traj[H - 1])

    # For each test point, compute predicted probability of crossing
    y_true = np.zeros(N_test, dtype=int)
    y_prob = np.zeros(N_test)
    valid = np.ones(N_test, dtype=bool)

    for j in range(N_test):
        i_full = test_idx[j]
        if i_full + H >= N_full:
            valid[j] = False
            continue

        # True label: did a transition happen in (t_j, t_j + H*DT]?
        t_j = t_test[j]
        window_end = t_j + H * U.DT
        for et in event_times:
            if t_j < et <= window_end:
                y_true[j] = 1
                break

        # Predicted x at horizon H (oracle rollout)
        r = U.rollout_open_loop(
            filt['states_x'][i_full], filt['states_u'][i_full],
            i_full, H, t_arr, v_arr, params, cl, mode='oracle')
        mu_h = r['path_x'][H - 1]
        if np.isnan(mu_h):
            valid[j] = False
            continue

        # Current state determines threshold direction
        cur_state = states[j]
        if cur_state == 0:
            # Need to cross up: P(x > thresh_up)
            y_prob[j] = 1.0 - sp_stats.norm.cdf(thresh_up, mu_h, sigma_h)
        else:
            # Need to cross down: P(x < thresh_down)
            y_prob[j] = sp_stats.norm.cdf(thresh_down, mu_h, sigma_h)

    v_mask = valid
    yt = y_true[v_mask]
    yp = y_prob[v_mask]
    n_pos = int(yt.sum())
    n_neg = int((1 - yt).sum())
    print(f"    Valid: {v_mask.sum()}, Events: {n_pos}, Non-events: {n_neg}")

    if n_pos < 3 or n_neg < 3:
        print(f"    Too few events for reliable ROC/PR -- skipping h={H}")
        continue

    # ROC
    fpr, tpr, thresholds_roc = roc_curve(yt, yp)
    roc_auc = auc(fpr, tpr)

    # PR
    prec, rec, thresholds_pr = precision_recall_curve(yt, yp)
    pr_auc = auc(rec, prec)

    # F1 at best threshold
    f1_scores = []
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred_bin = (yp >= thr).astype(int)
        f1_scores.append(f1_score(yt, y_pred_bin, zero_division=0))
    best_f1 = max(f1_scores)
    best_thr = np.linspace(0.01, 0.99, 99)[np.argmax(f1_scores)]

    print(f"    ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, "
          f"Best F1: {best_f1:.4f} (thr={best_thr:.2f})")

    # Lead-time: for each true event, find first t where P > 0.5
    lead_times = []
    for et in event_times:
        # Find test points before event
        pre_event = np.where((t_test < et) & (t_test >= et - H * U.DT) & v_mask)[0]
        if len(pre_event) == 0:
            continue
        early = None
        for j in pre_event:
            if y_prob[j] > 0.5:
                early = et - t_test[j]
                break
        if early is not None:
            lead_times.append(float(early))

    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    print(f"    Mean lead time (P>0.5): {mean_lead:.2f}s "
          f"({len(lead_times)}/{len(event_times)} events)")

    all_rows.append({
        'horizon': H, 'time_s': H * U.DT,
        'roc_auc': roc_auc, 'pr_auc': pr_auc,
        'best_f1': best_f1, 'best_threshold': best_thr,
        'n_events': n_pos, 'n_nonevents': n_neg,
        'mean_lead_time_s': mean_lead,
        'n_events_with_lead': len(lead_times),
    })

    # ROC figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, 'b-', lw=1.5, label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: Event Detection h={H} ({H*U.DT:.0f}s)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / f"roc_event_h{H}.png", bbox_inches='tight')
    plt.close(fig)

    # PR figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, 'r-', lw=1.5, label=f'PR (AUC={pr_auc:.3f})')
    ax.axhline(n_pos / (n_pos + n_neg), color='gray', ls='--', lw=0.8,
               label=f'Baseline ({n_pos/(n_pos+n_neg):.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve: Event Detection h={H} ({H*U.DT:.0f}s)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / f"pr_curve_event_h{H}.png", bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved ROC + PR figures for h={H}")

# Save CSV
df = pd.DataFrame(all_rows)
df.to_csv(OUT / "event_metrics.csv", index=False)

summary = {
    'horizons_evaluated': EVAL_HORIZONS,
    'metrics': all_rows,
    'n_events_test': len(event_times),
    'sigma_h_used': {str(h): float(np.sqrt(Pxx_traj[h-1])) for h in EVAL_HORIZONS},
    'runtime_s': time.time() - t0,
}
with open(OUT / "summary_diag04.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDiag04 complete ({time.time()-t0:.0f}s). Output: {OUT}")
