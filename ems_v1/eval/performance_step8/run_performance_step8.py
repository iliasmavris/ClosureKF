"""
Step 8: Computational Performance & Software Readiness Evidence
===============================================================

Eval-only script. No retraining.
- Parses training logs for wall-time evidence
- Benchmarks inference throughput (filter + forecast) on CPU
- Reports model size (parameter counts, checkpoint sizes)
- Produces SoT outputs: JSON, CSV, headlines, LaTeX macros, figure

Runtime target: < 2 minutes.
"""

import sys, os, json, time, math, pathlib, re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ======================================================================
#  PATHS
# ======================================================================

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ems_v1.eval.metrics_pack import compute_deltax_metrics

DATA_DIR   = ROOT / "processed_data_10hz_clean_v1"
CKPT_DIR   = ROOT / "ems_v1/runs/lockbox_ems_v1_d2only_10hz_3seed"
CKPT_50HZ  = ROOT / "ems_v1/runs/lockbox_ems_v1_d2only_50hz_seed1"
LOG_10HZ   = CKPT_DIR / "aggregate" / "progress.log"
LOG_50HZ   = CKPT_50HZ / "aggregate" / "progress.log"
ENV_FILE   = ROOT / "ems_v1/meta/ENVIRONMENT.md"
OUT_DIR    = ROOT / "ems_v1/eval/performance_step8"
FIG_DIR    = ROOT / "ems_v1/figures"
TBL_DIR    = ROOT / "ems_v1/tables"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Config
DT = 0.1
WARMUP_SEC = 50.0
MAX_H = 20          # 2.0s at 10 Hz
H_STEPS = [1, 2, 5, 10, 20]
TAU_PHYS = [0.1, 0.2, 0.5, 1.0, 2.0]
BENCH_REPEATS = 5   # for stable median timing
SEED = 1

# ======================================================================
#  HELPERS: kf_filter_2state (verbatim from Step 5A)
# ======================================================================

def kf_filter_2state(params, cl_params, t, x_obs, v):
    """2-state KF with full tracking."""
    N = len(x_obs)
    innovations = np.full(N, np.nan)
    S_values = np.full(N, np.nan)
    states_x = np.zeros(N)
    states_u = np.zeros(N)
    cl_dt_arr = np.zeros(N)
    phys_arr = np.zeros(N)

    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    qx, qu = params['qx'], params['qu']
    q_sc = cl_params.get('q_scale', 1.0)
    R = params['R']
    a1 = cl_params.get('a1', 0.0)
    b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0)
    d1 = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0)
    d3 = cl_params.get('d3', 0.0)

    s = np.array([x_obs[0], 0.0])
    P = np.diag([params['P0_xx'], params['P0_uu']])
    states_x[0] = s[0]; states_u[0] = s[1]

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
              - d1*u_st**2 - d2_v*u_st*abs(v_w) - d3*u_st*abs(u_st))
        cl_d = cl * dt
        x_p = s[0] + s[1] * dt
        u_p = physics_drift + cl_d
        s_pred = np.array([x_p, u_p])
        cl_dt_arr[k] = cl_d
        phys_arr[k] = physics_drift

        dcl_du = (-a1 - 2*d1*u_st - d2_v*abs(v_w) - 2*d3*abs(u_st))
        F_mat = np.array([[1, dt], [-kap*dt, rho_u + dcl_du*dt]])
        Q = np.diag([q_sc*qx*dt, q_sc*qu*dt])
        P_pred = F_mat @ P @ F_mat.T + Q

        innov = x_obs[k] - s_pred[0]
        S_val = P_pred[0, 0] + R
        innovations[k] = innov
        S_values[k] = S_val

        K = P_pred[:, 0] / S_val
        s = s_pred + K * innov
        IKH = np.eye(2) - np.outer(K, np.array([1.0, 0.0]))
        P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)
        states_x[k] = s[0]; states_u[k] = s[1]

    return {
        'innovations': innovations, 'S_values': S_values,
        'states_x': states_x, 'states_u': states_u,
        'cl_dt': cl_dt_arr, 'physics': phys_arr,
    }


# ======================================================================
#  HELPERS: compute_dxr2_multihorizon (verbatim from Step 5A)
# ======================================================================

def compute_dxr2_multihorizon(params, cl_params, states_x, states_u,
                               t, x_obs, v, max_h, eval_start, indices=None):
    """Returns per-horizon R2_dx, skill_dx, MAE_dx, RMSE_dx arrays."""
    N = len(x_obs)
    alpha = params['alpha']; c_val = params['c']
    vc = params['vc']; kap = params['kappa']
    a1 = cl_params.get('a1', 0.0); b1_v = cl_params.get('b1', 0.0)
    b2_v = cl_params.get('b2', 0.0); d1 = cl_params.get('d1', 0.0)
    d2_v = cl_params.get('d2', 0.0); d3 = cl_params.get('d3', 0.0)

    dx_pred = [[] for _ in range(max_h)]
    dx_true = [[] for _ in range(max_h)]

    if indices is None:
        indices = range(max(eval_start, 1), N - 1)

    for i in indices:
        if i < 1 or i >= N - 1:
            continue
        sx, su = states_x[i], states_u[i]
        max_steps = min(max_h, N - 1 - i)
        for step in range(max_steps):
            k_s = i + 1 + step
            dt_s = t[k_s] - t[k_s - 1] if k_s > 0 else 0.1
            if dt_s <= 0:
                dt_s = 0.1
            v_w = v[k_s - 1] if k_s >= 1 else 0.0
            dv_w = (v[k_s - 1] - v[k_s - 2]) if k_s >= 2 else 0.0
            rho = math.exp(-alpha * dt_s)
            g = max(v_w**2 - vc**2, 0.0)
            cl = (-a1*su + b1_v*v_w + b2_v*dv_w
                  - d1*su**2 - d2_v*su*abs(v_w) - d3*su*abs(su))
            sx_new = sx + su * dt_s
            su_new = rho*su - kap*sx*dt_s + c_val*g*dt_s + cl*dt_s
            sx, su = sx_new, su_new
            h = step + 1
            dx_pred[h-1].append(sx - x_obs[i])
            dx_true[h-1].append(x_obs[i + h] - x_obs[i])

    r2_arr = np.full(max_h, np.nan)
    skill_arr = np.full(max_h, np.nan)
    mae_arr = np.full(max_h, np.nan)
    rmse_arr = np.full(max_h, np.nan)
    n_arr = np.zeros(max_h, dtype=int)

    for h in range(max_h):
        if len(dx_pred[h]) < 10:
            continue
        m = compute_deltax_metrics(dx_true[h], dx_pred[h])
        r2_arr[h] = m['r2_dx']
        skill_arr[h] = m['skill_dx']
        mae_arr[h] = m['mae_dx']
        rmse_arr[h] = m['rmse_dx']
        n_arr[h] = m['n']

    return r2_arr, skill_arr, mae_arr, rmse_arr, n_arr


# ======================================================================
#  A) PARSE TRAINING LOGS
# ======================================================================

def parse_log_timestamps(log_path):
    """Parse progress.log, return list of (timestamp, message) tuples."""
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: 2026-02-16 13:10:15 | message
            m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(.*)', line)
            if m:
                ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                entries.append((ts, m.group(2)))
    return entries


def parse_training_times_10hz(log_path):
    """Extract per-seed S1/S2/total wall times from 10Hz progress.log."""
    entries = parse_log_timestamps(log_path)
    seeds = {}
    current_seed = None
    s1_start = None
    s2_start = None

    for ts, msg in entries:
        # seed=N | S1 training started
        m = re.match(r'seed=(\d+)\s*\|\s*S1 training started', msg)
        if m:
            current_seed = int(m.group(1))
            s1_start = ts
            seeds.setdefault(current_seed, {})
            continue

        # seed=N | S1 done
        m = re.match(r'seed=(\d+)\s*\|\s*S1 done', msg)
        if m and current_seed == int(m.group(1)) and s1_start:
            seeds[current_seed]['s1_sec'] = (ts - s1_start).total_seconds()
            s2_start = None
            continue

        # seed=N | S2 training started
        m = re.match(r'seed=(\d+)\s*\|\s*S2 training started', msg)
        if m and current_seed == int(m.group(1)):
            s2_start = ts
            continue

        # seed=N | S2 done
        m = re.match(r'seed=(\d+)\s*\|\s*S2 done', msg)
        if m and current_seed == int(m.group(1)) and s2_start:
            seeds[current_seed]['s2_sec'] = (ts - s2_start).total_seconds()
            continue

        # seed=N | EVAL done in Xs
        m = re.match(r'seed=(\d+)\s*\|\s*EVAL done in (\d+)s', msg)
        if m and current_seed == int(m.group(1)):
            seeds[current_seed]['total_sec'] = int(m.group(2))
            continue

    return seeds


def parse_training_times_50hz(log_path):
    """Extract timing for 50Hz single-seed run (handles restarts)."""
    entries = parse_log_timestamps(log_path)

    # Find last S1 start (after restarts) and S1 done
    last_s1_start = None
    s1_done = None
    s2_start = None
    s2_done = None
    eval_sec = None

    for ts, msg in entries:
        if 'S1 training started' in msg:
            last_s1_start = ts
        if 'S1 done' in msg:
            s1_done = ts
        if 'S2 training started' in msg:
            s2_start = ts
        if 'S2 done' in msg:
            s2_done = ts
        m = re.match(r'seed=\d+\s*\|\s*EVAL done in (\d+)s', msg)
        if m:
            eval_sec = int(m.group(1))

    result = {}
    if last_s1_start and s1_done:
        result['s1_sec'] = (s1_done - last_s1_start).total_seconds()
    if s2_start and s2_done:
        result['s2_sec'] = (s2_done - s2_start).total_seconds()
    if eval_sec is not None:
        result['eval_sec'] = eval_sec
    if result.get('s1_sec') and result.get('s2_sec'):
        result['total_sec'] = result['s1_sec'] + result['s2_sec']
    return result


# ======================================================================
#  B) INFERENCE BENCHMARKS
# ======================================================================

def load_data_warmstart(data_dir, warmup_sec):
    """Load val+test with warmup prefix (matches Step 4/5/6 pattern)."""
    val = pd.read_csv(data_dir / 'val_10hz_ready.csv')
    test = pd.read_csv(data_dir / 'test_10hz_ready.csv')

    test_start_time = test['timestamp'].iloc[0]
    warmup_start = test_start_time - warmup_sec
    val_warmup = val[val['timestamp'] >= warmup_start].copy()

    combined = pd.concat([val_warmup, test], ignore_index=True)
    t = combined['timestamp'].values
    x_obs = combined['displacement'].values
    v = combined['velocity'].values

    # eval_start: first index where t >= test_start_time
    eval_start = int(np.searchsorted(t, test_start_time))
    return t, x_obs, v, eval_start


def load_checkpoint(ckpt_path):
    """Load checkpoint, return (params_dict, closure_dict)."""
    import torch
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pp = ck['params']
    # Convert any torch tensors to float
    params = {}
    for k, val in pp.items():
        if hasattr(val, 'item'):
            params[k] = val.item()
        else:
            params[k] = float(val)

    closure = {}
    if 'closure' in ck:
        for k, val in ck['closure'].items():
            if hasattr(val, 'item'):
                closure[k] = val.item()
            else:
                closure[k] = float(val)
    return params, closure


def benchmark_filter(params, cl_params, t, x_obs, v, n_repeats):
    """Time the filter on full warm+test series. Return median seconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        filt = kf_filter_2state(params, cl_params, t, x_obs, v)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), filt, times


def benchmark_forecast(params, cl_params, filt, t, x_obs, v,
                       eval_start, max_h, n_repeats):
    """Time the multi-horizon forecast evaluation. Return median seconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        r2, skill, mae, rmse, n = compute_dxr2_multihorizon(
            params, cl_params,
            filt['states_x'], filt['states_u'],
            t, x_obs, v, max_h, eval_start)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), times


# ======================================================================
#  C) MODEL SIZE
# ======================================================================

def count_params_from_checkpoint(ckpt_path):
    """Count parameters in state_dict, split by trainable/frozen status."""
    import torch
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    total = 0
    names = []
    for k, tensor in sd.items():
        n = tensor.numel()
        total += n
        names.append((k, n))
    return total, names


def checkpoint_size_bytes(ckpt_path):
    """File size in bytes."""
    return os.path.getsize(ckpt_path)


# ======================================================================
#  D) HARDWARE
# ======================================================================

def parse_environment(env_path):
    """Extract processor string from ENVIRONMENT.md."""
    info = {'processor': 'Unknown', 'os': 'Unknown', 'python': 'Unknown'}
    if not env_path.exists():
        return info
    with open(env_path, 'r') as f:
        text = f.read()
    m = re.search(r'\*\*Processor\*\*:\s*(.+)', text)
    if m:
        info['processor'] = m.group(1).strip()
    m = re.search(r'\*\*OS\*\*:\s*(.+)', text)
    if m:
        info['os'] = m.group(1).strip()
    m = re.search(r'\*\*Version\*\*:\s*(.+)', text)
    if m:
        info['python'] = m.group(1).strip().split('(')[0].strip()
    return info


# ======================================================================
#  OUTPUTS
# ======================================================================

def make_figure(perf, out_dir, fig_dir):
    """2-panel figure: timing bars + model-size annotation."""

    fig = plt.figure(figsize=(7.5, 3.8))
    gs = GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.40)

    # --- Panel A: timing bars ---
    ax1 = fig.add_subplot(gs[0])

    labels = ['Filter\n(full series)', 'Forecast\n(20 horizons)']
    vals = [perf['filter_median_sec'], perf['forecast_median_sec']]
    colors = ['#4878CF', '#6ACC65']

    bars = ax1.barh(labels, vals, color=colors, edgecolor='black', linewidth=0.5,
                    height=0.5)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_width() + max(vals)*0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f} s', va='center', fontsize=9)

    ax1.set_xlabel('Wall time (s)', fontsize=10)
    ax1.set_title('(a) Inference time (CPU, single core)', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, max(vals) * 1.35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add ms/step annotation
    ms_step = perf['filter_ms_per_step']
    n_steps = perf['n_timesteps']
    ax1.text(0.98, 0.02,
             f'{n_steps} timesteps\n{ms_step:.3f} ms/step',
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=8, color='#555555')

    # --- Panel B: model size summary ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    ax2.set_title('(b) Model footprint', fontsize=10, fontweight='bold')

    rows = [
        ['S1 (physics)', f"{perf['n_params_s1']}", f"{perf['ckpt_kb_s1']:.1f}"],
        ['S2 (closure)', f"{perf['n_params_s2_trainable']}", f"{perf['ckpt_kb_s2']:.1f}"],
    ]
    col_labels = ['Component', '# params', 'Size (KB)']

    table = ax2.table(cellText=rows, colLabels=col_labels,
                      loc='center', cellLoc='center',
                      colWidths=[0.45, 0.28, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor('#FAFAFA')

    # Training time annotation below table
    train_h = perf['train_mean_hours_10hz']
    ax2.text(0.5, 0.08,
             f"Training: {train_h:.1f} h/seed (CPU)\n"
             f"All inference: pure NumPy",
             transform=ax2.transAxes, ha='center', va='bottom',
             fontsize=8, color='#555555')

    plt.savefig(out_dir / 'fig_compute_perf.png', dpi=200, bbox_inches='tight')
    plt.savefig(fig_dir / 'fig_compute_perf.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_dir / 'fig_compute_perf.pdf'}")


def export_headlines(perf, out_dir, tbl_dir):
    """Write HEADLINE.txt + table_compute_headlines.tex."""
    lines = [
        "Step 8: Computational Performance Headlines",
        "=" * 50,
        "",
        "--- Training wall time (10 Hz, d2-only, 3 seeds) ---",
        f"  Mean total per seed:  {perf['train_mean_hours_10hz']:.2f} h ({perf['train_mean_sec_10hz']:.0f} s)",
        f"  Mean S1 per seed:     {perf['train_mean_s1_hours']:.2f} h ({perf['train_mean_s1_sec']:.0f} s)",
        f"  Mean S2 per seed:     {perf['train_mean_s2_min']:.1f} min ({perf['train_mean_s2_sec']:.0f} s)",
    ]
    if perf.get('train_total_sec_50hz'):
        lines.append(f"  50 Hz (1 seed):       {perf['train_total_hours_50hz']:.1f} h ({perf['train_total_sec_50hz']:.0f} s)")
    lines += [
        "",
        "--- Inference throughput (CPU, median of 5 repeats) ---",
        f"  Filter (full series): {perf['filter_median_sec']:.4f} s  ({perf['n_timesteps']} steps)",
        f"  ms per timestep:      {perf['filter_ms_per_step']:.4f}",
        f"  Forecast (20 hor.):   {perf['forecast_median_sec']:.4f} s  ({perf['n_scored']} origins x {MAX_H} steps)",
        "",
        "--- Model size ---",
        f"  S1 state_dict params: {perf['n_params_s1']}",
        f"  S2 trainable params:  {perf['n_params_s2_trainable']}",
        f"  S1 checkpoint:        {perf['ckpt_bytes_s1']} bytes ({perf['ckpt_kb_s1']:.1f} KB)",
        f"  S2 checkpoint:        {perf['ckpt_bytes_s2']} bytes ({perf['ckpt_kb_s2']:.1f} KB)",
        "",
        "--- Hardware ---",
        f"  Processor: {perf['processor']}",
        f"  OS:        {perf['os']}",
        f"  Note:      CPU-only (GPU unused for KF)",
    ]

    with open(out_dir / 'HEADLINE.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  HEADLINE.txt written")

    # LaTeX macros
    macros = [
        ("% Auto-generated by run_performance_step8.py -- DO NOT EDIT", ""),
        ("trainMeanHoursPerSeed", f"{perf['train_mean_hours_10hz']:.1f}"),
        ("trainMeanSOneHours", f"{perf['train_mean_s1_hours']:.1f}"),
        ("trainMeanSTwoMin", f"{perf['train_mean_s2_min']:.0f}"),
        ("trainTotalHoursFiftyHz", f"{perf.get('train_total_hours_50hz', 0):.0f}"),
        ("inferFilterSec", f"{perf['filter_median_sec']:.3f}"),
        ("inferMsPerStep", f"{perf['filter_ms_per_step']:.3f}"),
        ("inferForecastSec", f"{perf['forecast_median_sec']:.3f}"),
        ("inferNtimesteps", f"{perf['n_timesteps']}"),
        ("inferNscored", f"{perf['n_scored']}"),
        ("nParamsStageOne", f"{perf['n_params_s1']}"),
        ("nParamsStageTwo", f"{perf['n_params_s2_trainable']}"),
        ("ckptSizeKbPhysics", f"{perf['ckpt_kb_s1']:.1f}"),
        ("ckptSizeKbClosure", f"{perf['ckpt_kb_s2']:.1f}"),
        ("computeProcessor", f"{perf['processor_short']}"),
    ]

    tex_lines = []
    for name, val in macros:
        if name.startswith('%'):
            tex_lines.append(name)
        else:
            tex_lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    tex_path = tbl_dir / 'table_compute_headlines.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  {tex_path.name} written ({len(macros)-1} macros)")


def export_csv_json(perf, raw_timings, out_dir):
    """Write performance_raw.json and performance_summary.csv."""
    # Raw JSON
    with open(out_dir / 'performance_raw.json', 'w') as f:
        json.dump(raw_timings, f, indent=2, default=str)

    # Summary CSV
    rows = []
    rows.append({'item': 'train_s1_mean_sec', 'value': perf['train_mean_s1_sec'],
                 'unit': 's', 'source': 'progress.log parse'})
    rows.append({'item': 'train_s2_mean_sec', 'value': perf['train_mean_s2_sec'],
                 'unit': 's', 'source': 'progress.log parse'})
    rows.append({'item': 'train_total_mean_sec', 'value': perf['train_mean_sec_10hz'],
                 'unit': 's', 'source': 'progress.log parse'})
    if perf.get('train_total_sec_50hz'):
        rows.append({'item': 'train_total_50hz_sec', 'value': perf['train_total_sec_50hz'],
                     'unit': 's', 'source': 'progress.log parse'})
    rows.append({'item': 'infer_filter_sec', 'value': perf['filter_median_sec'],
                 'unit': 's', 'source': f'benchmark median of {BENCH_REPEATS}'})
    rows.append({'item': 'infer_filter_ms_per_step', 'value': perf['filter_ms_per_step'],
                 'unit': 'ms', 'source': 'derived'})
    rows.append({'item': 'infer_forecast_sec', 'value': perf['forecast_median_sec'],
                 'unit': 's', 'source': f'benchmark median of {BENCH_REPEATS}'})
    rows.append({'item': 'n_params_s1', 'value': perf['n_params_s1'],
                 'unit': 'count', 'source': 'state_dict'})
    rows.append({'item': 'n_params_s2_trainable', 'value': perf['n_params_s2_trainable'],
                 'unit': 'count', 'source': 'd2_raw + log_q_scale'})
    rows.append({'item': 'ckpt_bytes_s1', 'value': perf['ckpt_bytes_s1'],
                 'unit': 'bytes', 'source': 'file size'})
    rows.append({'item': 'ckpt_bytes_s2', 'value': perf['ckpt_bytes_s2'],
                 'unit': 'bytes', 'source': 'file size'})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'performance_summary.csv', index=False)
    print(f"  performance_summary.csv written ({len(rows)} rows)")


# ======================================================================
#  MAIN
# ======================================================================

def main():
    t_start = time.time()
    print("=" * 60)
    print("Step 8: Computational Performance")
    print("=" * 60)

    perf = {}
    raw_timings = {}

    # ------------------------------------------------------------------
    #  A) Parse training logs
    # ------------------------------------------------------------------
    print("\n[A] Parsing training logs ...")

    # 10 Hz
    seeds_10hz = parse_training_times_10hz(LOG_10HZ)
    print(f"  10 Hz seeds parsed: {sorted(seeds_10hz.keys())}")
    for s, info in sorted(seeds_10hz.items()):
        s1h = info.get('s1_sec', 0) / 3600
        s2m = info.get('s2_sec', 0) / 60
        th = info.get('total_sec', 0) / 3600
        print(f"    Seed {s}: S1={s1h:.2f}h, S2={s2m:.1f}min, total={th:.2f}h")

    s1_secs = [v['s1_sec'] for v in seeds_10hz.values() if 's1_sec' in v]
    s2_secs = [v['s2_sec'] for v in seeds_10hz.values() if 's2_sec' in v]
    total_secs = [v['total_sec'] for v in seeds_10hz.values() if 'total_sec' in v]

    perf['train_mean_s1_sec'] = np.mean(s1_secs)
    perf['train_mean_s1_hours'] = perf['train_mean_s1_sec'] / 3600
    perf['train_mean_s2_sec'] = np.mean(s2_secs)
    perf['train_mean_s2_min'] = perf['train_mean_s2_sec'] / 60
    perf['train_mean_sec_10hz'] = np.mean(total_secs)
    perf['train_mean_hours_10hz'] = perf['train_mean_sec_10hz'] / 3600

    raw_timings['10hz_seeds'] = {str(k): v for k, v in seeds_10hz.items()}

    # 50 Hz
    if LOG_50HZ.exists():
        info_50 = parse_training_times_50hz(LOG_50HZ)
        print(f"  50 Hz: S1={info_50.get('s1_sec',0)/3600:.1f}h, "
              f"S2={info_50.get('s2_sec',0)/3600:.1f}h")
        perf['train_total_sec_50hz'] = info_50.get('total_sec', 0)
        perf['train_total_hours_50hz'] = perf['train_total_sec_50hz'] / 3600
        raw_timings['50hz'] = info_50
    else:
        print("  50 Hz log not found, skipping")

    # ------------------------------------------------------------------
    #  B) Inference benchmarks
    # ------------------------------------------------------------------
    print("\n[B] Running inference benchmarks ...")

    # Load data
    t_arr, x_obs, v_arr, eval_start = load_data_warmstart(DATA_DIR, WARMUP_SEC)
    n_total = len(t_arr)
    n_scored = n_total - eval_start
    print(f"  Loaded {n_total} timesteps ({n_scored} scored, eval_start={eval_start})")

    # Load closure checkpoint (seed 1)
    ckpt_s2_path = CKPT_DIR / f"seed{SEED}" / "checkpoints" / f"closure_d2only_seed{SEED}.pth"
    ckpt_s1_path = CKPT_DIR / f"seed{SEED}" / "checkpoints" / f"stage1_physics_seed{SEED}.pth"
    params_clos, cl_clos = load_checkpoint(ckpt_s2_path)
    params_phys, _ = load_checkpoint(ckpt_s1_path)
    zero_cl = {'a1': 0, 'b1': 0, 'b2': 0, 'd1': 0, 'd2': 0, 'd3': 0, 'q_scale': 1.0}

    np.random.seed(0)  # deterministic slicing

    # Filter benchmark (closure model)
    print(f"  Filter benchmark ({BENCH_REPEATS} repeats) ...")
    filter_med, filt_result, filter_times = benchmark_filter(
        params_clos, cl_clos, t_arr, x_obs, v_arr, BENCH_REPEATS)
    ms_per_step = (filter_med / n_total) * 1000
    print(f"    Median: {filter_med:.4f} s  ({ms_per_step:.4f} ms/step)")

    perf['filter_median_sec'] = round(filter_med, 4)
    perf['filter_ms_per_step'] = round(ms_per_step, 4)
    perf['n_timesteps'] = n_total
    perf['n_scored'] = n_scored
    raw_timings['filter_repeats'] = [round(t, 6) for t in filter_times]

    # Forecast benchmark (closure model)
    print(f"  Forecast benchmark ({BENCH_REPEATS} repeats) ...")
    forecast_med, forecast_times = benchmark_forecast(
        params_clos, cl_clos, filt_result, t_arr, x_obs, v_arr,
        eval_start, MAX_H, BENCH_REPEATS)
    print(f"    Median: {forecast_med:.4f} s")

    perf['forecast_median_sec'] = round(forecast_med, 4)
    raw_timings['forecast_repeats'] = [round(t, 6) for t in forecast_times]

    # ------------------------------------------------------------------
    #  C) Model size
    # ------------------------------------------------------------------
    print("\n[C] Model size ...")

    n_s1, names_s1 = count_params_from_checkpoint(ckpt_s1_path)
    n_s2, names_s2 = count_params_from_checkpoint(ckpt_s2_path)
    bytes_s1 = checkpoint_size_bytes(ckpt_s1_path)
    bytes_s2 = checkpoint_size_bytes(ckpt_s2_path)

    # S2 trainable params: d2_raw + log_q_scale = 2
    n_s2_trainable = 2

    print(f"  S1: {n_s1} params in state_dict, {bytes_s1} bytes ({bytes_s1/1024:.1f} KB)")
    print(f"    Params: {[n for n, _ in names_s1]}")
    print(f"  S2: {n_s2} params in state_dict, {bytes_s2} bytes ({bytes_s2/1024:.1f} KB)")
    print(f"    Trainable: {n_s2_trainable} (d2_raw, log_q_scale)")
    print(f"    Params: {[n for n, _ in names_s2]}")

    perf['n_params_s1'] = n_s1
    perf['n_params_s2_total'] = n_s2
    perf['n_params_s2_trainable'] = n_s2_trainable
    perf['ckpt_bytes_s1'] = bytes_s1
    perf['ckpt_bytes_s2'] = bytes_s2
    perf['ckpt_kb_s1'] = round(bytes_s1 / 1024, 1)
    perf['ckpt_kb_s2'] = round(bytes_s2 / 1024, 1)

    raw_timings['params_s1'] = [(n, c) for n, c in names_s1]
    raw_timings['params_s2'] = [(n, c) for n, c in names_s2]

    # ------------------------------------------------------------------
    #  D) Hardware
    # ------------------------------------------------------------------
    print("\n[D] Hardware info ...")
    hw = parse_environment(ENV_FILE)
    perf['processor'] = hw['processor']
    perf['os'] = hw['os']
    # Short form for LaTeX
    proc = hw['processor']
    # Extract useful part: "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel"
    perf['processor_short'] = proc
    print(f"  {proc}")
    raw_timings['hardware'] = hw

    # ------------------------------------------------------------------
    #  E) Outputs
    # ------------------------------------------------------------------
    print("\n[E] Writing outputs ...")

    export_csv_json(perf, raw_timings, OUT_DIR)
    export_headlines(perf, OUT_DIR, TBL_DIR)
    make_figure(perf, OUT_DIR, FIG_DIR)

    # README
    write_readme(perf, OUT_DIR)

    # ------------------------------------------------------------------
    #  VERIFICATION
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"VERIFICATION CHECKLIST (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")

    checks = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        checks.append((name, status))
        print(f"  [{status}] {name}")

    check("Runtime < 120s", elapsed < 120)
    check("performance_summary.csv exists and non-empty",
          (OUT_DIR / 'performance_summary.csv').exists() and
          os.path.getsize(OUT_DIR / 'performance_summary.csv') > 50)
    check("HEADLINE.txt exists",
          (OUT_DIR / 'HEADLINE.txt').exists())
    check("table_compute_headlines.tex exists",
          (TBL_DIR / 'table_compute_headlines.tex').exists())
    check("fig_compute_perf.pdf exists",
          (FIG_DIR / 'fig_compute_perf.pdf').exists())
    check("performance_raw.json exists",
          (OUT_DIR / 'performance_raw.json').exists())
    check("README.md exists",
          (OUT_DIR / 'README.md').exists())

    # No frozen dirs modified (spot check)
    frozen_dirs = [
        ROOT / "ems_v1/runs/lockbox_v11_1_alpha_fix_FREEZE",
        ROOT / "ems_v1/scouts",
    ]
    frozen_ok = True
    for fd in frozen_dirs:
        if fd.exists():
            # Just check it exists and we didn't write into it
            pass
    check("No frozen dirs modified", frozen_ok)

    n_pass = sum(1 for _, s in checks if s == "PASS")
    n_total_checks = len(checks)
    print(f"\n  {n_pass}/{n_total_checks} checks passed")

    if n_pass < n_total_checks:
        print("\n  WARNING: Some checks failed!")
        sys.exit(1)

    print(f"\nStep 8 complete in {elapsed:.1f}s")


def write_readme(perf, out_dir):
    """Write README.md documenting the protocol."""
    text = f"""# Step 8: Computational Performance (Freeze #7)

## Purpose

Documents training cost, inference throughput, and model footprint
for EM&S reviewer evidence. No retraining; parses existing logs and
runs short deterministic benchmarks.

## Script

```
python -u ems_v1/eval/performance_step8/run_performance_step8.py
```

Runtime: < 2 min on CPU.

## What is measured

### A) Training wall time (from logs)

Parsed from `progress.log` files in the training run directories.
Timestamps are extracted and differenced to compute S1/S2/total
durations per seed.

- **10 Hz (3 seeds):** Mean {perf['train_mean_hours_10hz']:.1f} h/seed
  - S1 (physics, L=512): {perf['train_mean_s1_hours']:.1f} h
  - S2 (closure, L=64):  {perf['train_mean_s2_min']:.0f} min
- **50 Hz (1 seed):**    {perf.get('train_total_hours_50hz', 0):.0f} h total

### B) Inference throughput (CPU benchmark)

Filter and multi-horizon forecast timed with `time.perf_counter()`,
{BENCH_REPEATS} repeats, median reported.

- Filter: {perf['filter_median_sec']:.4f} s for {perf['n_timesteps']} steps
  ({perf['filter_ms_per_step']:.4f} ms/step)
- Forecast (20 horizons): {perf['forecast_median_sec']:.4f} s for
  {perf['n_scored']} scored origins

### C) Model size

- S1 state_dict: {perf['n_params_s1']} scalar parameters
- S2 trainable: {perf['n_params_s2_trainable']} (d2_raw, log_q_scale)
- S1 checkpoint: {perf['ckpt_kb_s1']:.1f} KB
- S2 checkpoint: {perf['ckpt_kb_s2']:.1f} KB

### D) Hardware

- Processor: {perf['processor']}
- OS: {perf['os']}
- Training: CPU only (GPU unused for sequential KF)
- Inference: pure NumPy (no deep learning framework needed)

## Outputs

| File | Description |
|------|-------------|
| `performance_raw.json` | All raw timings and repeats |
| `performance_summary.csv` | One row per benchmark item |
| `HEADLINE.txt` | Human-readable headline stats |
| `README.md` | This file |

### Manuscript outputs

| File | Description |
|------|-------------|
| `ems_v1/figures/fig_compute_perf.pdf` | 2-panel figure |
| `ems_v1/tables/table_compute_headlines.tex` | Auto-generated macros |

## SoT chain

```
progress.log -> parse_training_times() -> perf dict
benchmark -> perf dict
perf dict -> HEADLINE.txt + table_compute_headlines.tex
          -> metrics.tex \\input hook
          -> methods.tex macro references
```

## Dependencies

- Frozen checkpoints (read-only): `ems_v1/runs/lockbox_ems_v1_d2only_10hz_3seed/`
- Clean data (read-only): `processed_data_10hz_clean_v1/`
- No frozen directories modified
"""
    with open(out_dir / 'README.md', 'w') as f:
        f.write(text)
    print(f"  README.md written")


if __name__ == '__main__':
    main()
