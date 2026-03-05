"""
14_truth_ball_sim.py - Sphere truth model (RK4 + pin-capture + Coulomb friction)
=================================================================================
Core physics engine for virtual lab particle displacement generation.

State: [x, v_p, eta, xi] where
  x   = displacement [m]
  v_p = particle velocity [m/s]
  eta = OU noise state [m/s^2] (acceleration units)
  xi  = pin jitter OU state [m] (displacement about x_pin)

Forces (all in Newtons):
  1. Drag (blended):  F_D = 3*pi*mu*d_p*w + 0.5*rho_f*C_Dinf*A_p*|w|*w
     where w = u_b - v_p  (relative velocity)
  2. Added mass:      F_A = C_Du * m_f * a_f(t)
  3. Submerged weight: W_sub = (rho_p - rho_f)*g*V_p
  4. Coulomb friction: kinetic during sliding, static threshold for breakaway
  5. Restoring:        F_spring = -k_spring * (x - x_eq)
  6. OU noise:         F_noise = m_eff * eta
  7. Linear damping:   F_damp = -c_r * v_p

Pin-capture model (replaces generic "stuck"):
  - at_pin = True when particle is near x_pin (|x - x_pin| < eps_x)
    AND net driving force < mu_s * W_sub + F_pin
  - While at_pin: x = x_pin + xi (OU jitter), v_p = 0
  - Breakaway when |F_drive| > mu_s * W_sub + F_pin

Equation of motion (when not at_pin):
  m_eff * dv_p/dt = F_D + F_A - F_fric + F_spring + F_noise - F_damp

Non-circularity: No relu(v^2-vc^2), no rho*u decay, no closure terms.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d


def compute_derived_params(cfg):
    """Compute derived physical quantities from config."""
    d_p = cfg['sphere']['d_p']
    rho_p = cfg['sphere']['rho_p']
    rho_f = cfg['sphere']['rho_f']
    g = cfg['constants']['g']
    nu = cfg['constants']['nu']

    V_p = (np.pi / 6.0) * d_p**3                     # particle volume [m^3]
    A_p = (np.pi / 4.0) * d_p**2                      # projected area [m^2]
    m_p = rho_p * V_p                                  # particle mass [kg]
    m_f = rho_f * V_p                                  # displaced fluid mass [kg]
    C_m = cfg['added_mass']['C_m']
    m_eff = m_p + C_m * m_f                            # effective mass [kg]
    W_sub = (rho_p - rho_f) * g * V_p                  # submerged weight [N]
    mu = rho_f * nu                                    # dynamic viscosity [Pa s]

    return {
        'V_p': V_p, 'A_p': A_p, 'm_p': m_p, 'm_f': m_f,
        'm_eff': m_eff, 'W_sub': W_sub, 'mu': mu, 'd_p': d_p,
        'rho_f': rho_f, 'nu': nu,
    }


def compute_drag_force(w, cfg, derived):
    """Compute drag force [N] given relative velocity w = u_b - v_p."""
    model = cfg['drag']['model']
    d_p = derived['d_p']
    rho_f = derived['rho_f']
    mu = derived['mu']
    A_p = derived['A_p']

    if model == 'blended':
        C_Dinf = cfg['drag']['C_Dinf']
        F_stokes = 3.0 * np.pi * mu * d_p * w
        F_form = 0.5 * rho_f * C_Dinf * A_p * np.abs(w) * w
        return F_stokes + F_form

    elif model == 'schiller_naumann':
        Re_p = np.abs(w) * d_p / derived['nu']
        if Re_p < 1e-10:
            return 3.0 * np.pi * mu * d_p * w
        C_D = (24.0 / Re_p) * (1.0 + 0.15 * Re_p**0.687)
        return 0.5 * rho_f * C_D * A_p * w * np.abs(w)

    else:
        raise ValueError(f"Unknown drag model: {model}")


def compute_du_dt(t_raw, u_raw, smooth_hz, smooth_order, fs_raw):
    """Compute du/dt by smoothing u then central finite differences."""
    sos = butter(smooth_order, smooth_hz, btype='low', fs=fs_raw, output='sos')
    u_smooth = sosfiltfilt(sos, u_raw)

    du = np.zeros_like(u_smooth)
    dt = t_raw[1] - t_raw[0]
    du[1:-1] = (u_smooth[2:] - u_smooth[:-2]) / (2.0 * dt)
    du[0] = (u_smooth[1] - u_smooth[0]) / dt
    du[-1] = (u_smooth[-1] - u_smooth[-2]) / dt

    return du


def simulate_sphere(cfg, t_flow, u_flow, seed=42):
    """
    Run sphere dynamics simulation with pin-capture model.

    Parameters
    ----------
    cfg : dict
        Configuration (from ball_params_default.yaml)
    t_flow : array
        Time array from CFD probe [s] (raw, ~200 Hz)
    u_flow : array
        Streamwise velocity at probe height [m/s] (raw, ~200 Hz)
    seed : int
        Random seed for OU noise

    Returns
    -------
    result : dict with keys:
        t       : time array at dt_sim [s]
        x       : displacement [m]
        v_p     : particle velocity [m/s]
        at_pin  : 1 if at pin, 0 if sliding/free
        contact : 1 if at_pin (kept for backward compat), 0 if sliding
        eta     : OU noise state [m/s^2]
        u_b     : interpolated u_b at sim times [m/s]
        du_b    : du_b/dt at sim times [m/s^2]
        a_force : acceleration from force balance [m/s^2]
                  (RHS/m_eff during FREE, NaN during PINNED)
    """
    rng = np.random.default_rng(seed)
    derived = compute_derived_params(cfg)

    # Integration parameters
    dt = cfg['integration']['dt_sim']
    t_start = cfg['integration']['t_start']
    t_end = cfg['integration']['t_end']
    t_end = min(t_end, t_flow[-1])

    # Build interpolators for u_b(t) and du_b/dt(t)
    fs_raw = 1.0 / np.median(np.diff(t_flow))
    u_interp = interp1d(t_flow, u_flow, kind='linear', fill_value='extrapolate')

    du_smooth_hz = cfg['flow']['du_smooth_hz']
    du_smooth_order = cfg['flow']['du_smooth_order']
    du_raw = compute_du_dt(t_flow, u_flow, du_smooth_hz, du_smooth_order, fs_raw)
    du_interp = interp1d(t_flow, du_raw, kind='linear', fill_value='extrapolate')

    # Physical parameters
    m_eff = derived['m_eff']
    W_sub = derived['W_sub']
    mu_s = cfg['friction']['mu_s']
    mu_k = cfg['friction']['mu_k']
    eps_v = cfg['friction']['eps_v']
    C_Du = cfg['added_mass']['C_Du']
    m_f = derived['m_f']
    k_spring = cfg['restoring']['k_spring']
    x_eq = cfg['restoring']['x_eq']
    c_r = cfg['damping']['c_r']
    sigma_a = cfg['noise']['sigma_a']
    tau_eta = cfg['noise']['tau_eta']

    # Pin parameters
    pin_cfg = cfg.get('pin', {})
    x_pin = pin_cfg.get('x_pin', 0.0)
    eps_x = pin_cfg.get('eps_x', 0.003)
    sigma_x = pin_cfg.get('sigma_x', 0.001)
    tau_x = pin_cfg.get('tau_x', 0.2)

    pin_hold_cfg = cfg.get('pin_hold', {})
    F_pin = pin_hold_cfg.get('F_pin', 0.0)

    # Breakaway threshold
    F_breakaway = mu_s * W_sub + F_pin

    # OU noise exact discretization (eta -- acceleration noise)
    phi_eta = np.exp(-dt / tau_eta)
    sigma_disc_eta = sigma_a * np.sqrt(1.0 - phi_eta**2)

    # Pin jitter OU exact discretization (xi -- position jitter)
    phi_x = np.exp(-dt / tau_x)
    sigma_disc_x = sigma_x * np.sqrt(1.0 - phi_x**2)

    # Time array
    n_steps = int(round((t_end - t_start) / dt))
    t_sim = t_start + np.arange(n_steps + 1) * dt

    # Allocate output arrays
    x_out = np.zeros(n_steps + 1)
    vp_out = np.zeros(n_steps + 1)
    eta_out = np.zeros(n_steps + 1)
    atpin_out = np.zeros(n_steps + 1, dtype=np.int32)
    ub_out = np.zeros(n_steps + 1)
    dub_out = np.zeros(n_steps + 1)
    a_force_out = np.full(n_steps + 1, np.nan)  # NaN for pinned steps

    # Capture event log: (t, x_before, x_after, delta_x, v_before)
    capture_log = []

    # Initial conditions
    x = x_pin      # start at pin
    v_p = 0.0
    eta = 0.0
    xi = 0.0        # pin jitter state
    is_at_pin = True  # start at pin

    def driving_force_at(x_val, v_p_val, eta_val, t_val):
        """Compute total non-friction force on particle."""
        u_b = float(u_interp(t_val))
        du_b = float(du_interp(t_val))
        w = u_b - v_p_val

        F_drag = compute_drag_force(w, cfg, derived)
        F_added = C_Du * m_f * du_b
        F_spring = -k_spring * (x_val - x_eq)
        F_noise = m_eff * eta_val
        F_damp = -c_r * v_p_val

        return F_drag + F_added + F_spring + F_noise + F_damp

    def accel(x_val, v_p_val, eta_val, t_val):
        """Compute acceleration including kinetic friction (sliding state)."""
        F_drive = driving_force_at(x_val, v_p_val, eta_val, t_val)

        # Kinetic friction during sliding
        if np.abs(v_p_val) > eps_v:
            F_fric = mu_k * W_sub * np.sign(v_p_val)
        else:
            # Near-zero velocity: check if force exceeds static threshold
            if np.abs(F_drive) > mu_s * W_sub:
                # Breakaway: apply kinetic friction in direction of drive
                F_fric = mu_k * W_sub * np.sign(F_drive)
            else:
                # Friction balances drive (temporary stop, not at pin)
                F_fric = F_drive

        return (F_drive - F_fric) / m_eff

    # ---- Main integration loop (RK4 with pin-capture) ----
    for i in range(n_steps):
        t_i = t_sim[i]
        u_b_i = float(u_interp(t_i))
        du_b_i = float(du_interp(t_i))

        # Store current state
        x_out[i] = x
        vp_out[i] = v_p
        eta_out[i] = eta
        atpin_out[i] = 1 if is_at_pin else 0
        ub_out[i] = u_b_i
        dub_out[i] = du_b_i

        if is_at_pin:
            # --- At-pin branch ---
            # Check breakaway: compute driving force at pin with v_p=0
            F_drive_pin = driving_force_at(x_pin, 0.0, eta, t_i)
            if np.abs(F_drive_pin) > F_breakaway:
                # Breakaway: leave pin, begin sliding
                is_at_pin = False
                x = x_pin + xi   # depart from jittered position
                v_p = 0.0        # start from rest (RK4 will accelerate)
                # Fall through to RK4 below
            else:
                # Stay at pin: advance jitter OU, keep v_p=0
                xi = phi_x * xi + sigma_disc_x * rng.standard_normal()
                x = x_pin + xi
                v_p = 0.0
                # Advance eta even while pinned (it's a background process)
                eta = phi_eta * eta + sigma_disc_eta * rng.standard_normal()
                continue

        # --- Sliding/free branch: RK4 for (x, v_p) ---
        k1_x = v_p
        k1_v = accel(x, v_p, eta, t_i)
        a_force_out[i] = k1_v  # log force-balance accel at stored state

        k2_x = v_p + 0.5 * dt * k1_v
        k2_v = accel(x + 0.5 * dt * k1_x, v_p + 0.5 * dt * k1_v, eta, t_i + 0.5 * dt)

        k3_x = v_p + 0.5 * dt * k2_v
        k3_v = accel(x + 0.5 * dt * k2_x, v_p + 0.5 * dt * k2_v, eta, t_i + 0.5 * dt)

        k4_x = v_p + dt * k3_v
        k4_v = accel(x + dt * k3_x, v_p + dt * k3_v, eta, t_i + dt)

        x_new = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_p_new = v_p + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        # --- Post-step pin capture check ---
        near_pin = np.abs(x_new - x_pin) < eps_x
        vel_small = np.abs(v_p_new) < eps_v
        sign_change = (v_p * v_p_new < 0)

        if near_pin and (vel_small or sign_change):
            F_drive_new = driving_force_at(x_new, 0.0, eta, t_i + dt)
            if np.abs(F_drive_new) < F_breakaway:
                # Capture at pin
                x_before = x_new
                is_at_pin = True
                xi = x_new - x_pin  # initial jitter = offset from pin
                # Clamp xi to reasonable range
                xi = np.clip(xi, -3*sigma_x, 3*sigma_x)
                x_new = x_pin + xi
                v_p_new = 0.0
                capture_log.append((
                    t_i + dt, x_before, x_new,
                    x_new - x_before, v_p,
                ))

        # Advance OU noise
        eta = phi_eta * eta + sigma_disc_eta * rng.standard_normal()

        x = x_new
        v_p = v_p_new

    # Store final step
    x_out[n_steps] = x
    vp_out[n_steps] = v_p
    eta_out[n_steps] = eta
    atpin_out[n_steps] = 1 if is_at_pin else 0
    ub_out[n_steps] = float(u_interp(t_sim[n_steps]))
    dub_out[n_steps] = float(du_interp(t_sim[n_steps]))
    if not is_at_pin:
        a_force_out[n_steps] = accel(x, v_p, eta, t_sim[n_steps])

    return {
        't': t_sim,
        'x': x_out,
        'v_p': vp_out,
        'at_pin': atpin_out,
        'contact': atpin_out,  # backward compat alias
        'eta': eta_out,
        'u_b': ub_out,
        'du_b': dub_out,
        'a_force': a_force_out,  # RHS/m_eff during free, NaN during pinned
        'capture_log': capture_log,  # list of (t, x_before, x_after, dx, v_before)
    }


def compute_event_rate(contact):
    """Fraction of time the particle is sliding (not at pin)."""
    return 1.0 - np.mean(contact)


def compute_pin_statistics(at_pin, dt):
    """
    Compute waiting time and excursion distributions from at_pin signal.

    Returns dict with:
        waiting_times  : array of durations at pin [s]
        excursion_times: array of durations away from pin [s]
    """
    waiting_times = []
    excursion_times = []

    if len(at_pin) == 0:
        return {'waiting_times': np.array([]), 'excursion_times': np.array([])}

    # Find transitions
    current_state = at_pin[0]
    run_start = 0

    for i in range(1, len(at_pin)):
        if at_pin[i] != current_state:
            duration = (i - run_start) * dt
            if current_state == 1:
                waiting_times.append(duration)
            else:
                excursion_times.append(duration)
            run_start = i
            current_state = at_pin[i]

    # Final run (don't include -- it's censored)

    return {
        'waiting_times': np.array(waiting_times),
        'excursion_times': np.array(excursion_times),
    }


def quick_sim(cfg, t_flow, u_flow, seed=42, t_end_override=None):
    """Run a short simulation for preflight checks."""
    cfg_copy = _deep_copy_cfg(cfg)
    if t_end_override is not None:
        cfg_copy['integration']['t_end'] = t_end_override
    return simulate_sphere(cfg_copy, t_flow, u_flow, seed=seed)


def _deep_copy_cfg(cfg):
    """Deep copy config dict."""
    import copy
    return copy.deepcopy(cfg)


if __name__ == '__main__':
    # Self-test: run with synthetic sinusoidal flow
    print("=" * 60)
    print("TRUTH BALL SIM - SELF TEST")
    print("=" * 60)

    import yaml
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    cfg_path = ROOT / "configs" / "ball_params_default.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Synthetic flow: sinusoidal u(t) at 200 Hz
    dt_flow = 0.005
    t_flow = np.arange(0, 60, dt_flow)
    u_mean = 0.17
    u_flow = u_mean + 0.04 * np.sin(2 * np.pi * 0.5 * t_flow)

    cfg['integration']['t_end'] = 60.0
    result = simulate_sphere(cfg, t_flow, u_flow, seed=42)

    dt_sim = cfg['integration']['dt_sim']
    pin_stats = compute_pin_statistics(result['at_pin'], dt_sim)

    print(f"  Steps: {len(result['t'])}")
    print(f"  x: mean={np.mean(result['x']):.6f}, std={np.std(result['x']):.6f}")
    print(f"  v_p: mean={np.mean(result['v_p']):.6f}, std={np.std(result['v_p']):.6f}")
    er = compute_event_rate(result['at_pin'])
    print(f"  Event rate: {er:.3f}")
    print(f"  at_pin fraction: {np.mean(result['at_pin']):.3f}")
    print(f"  Waiting times: n={len(pin_stats['waiting_times'])}")
    print(f"  Excursion times: n={len(pin_stats['excursion_times'])}")
    print(f"  NaN check: x={np.any(np.isnan(result['x']))}, "
          f"v={np.any(np.isnan(result['v_p']))}")

    if np.any(np.isnan(result['x'])) or np.any(np.isnan(result['v_p'])):
        print("  FAIL: NaN detected!")
    else:
        print("  PASS: simulation stable, no NaN")

    print("=" * 60)
