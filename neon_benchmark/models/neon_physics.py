"""
NEON Lake Temperature: Physics-Informed Kalman Filter Models

Two models following the ClosureKF two-stage pattern:

Stage 1 - Physics Floor (NeonKF):
    T_{t+1} = T_t - k * (T_t - T_air) * dt + process_noise
    Observation: y_t = T_t + obs_noise
    Learnable: k (>0), qT (process noise), R (obs noise), P0

Stage 2 - Closure Discovery (NeonKFClosure):
    T_{t+1} = T_t - k * (T_t - T_air) * dt
              + [theta_1*PAR + theta_2*PAR^2 + theta_3*wind
                 + theta_4*(T-T_air) + theta_5*wind*(T-T_air)] * dt
              + process_noise
    Physics frozen from S1; closure terms + q_scale trainable.

Both use 1-state KF: state = [T], observation = [T + noise].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CLOSURE_TERM_NAMES = ['par_lin', 'par_quad', 'wind_cool', 'sensible', 'forced_conv']


def _softplus_inv(x, eps=1e-6):
    x = max(float(x), eps)
    return x + math.log(-math.expm1(-x))


# ==============================================================================
#  STAGE 1: Physics Floor -- Newton's Law of Cooling
# ==============================================================================

class NeonKF(nn.Module):
    """
    1-state Kalman filter for lake temperature.

    State: s = [T]
    Dynamics: T_{t+1} = T_t - k*(T_t - T_air)*dt
    Observation: y = T + noise

    Args:
        k_init: Initial cooling coefficient (1/s), constrained > 0 via softplus
        log_q_init: log of process noise variance rate
        log_r_init: log of observation noise variance
        log_p0_init: log of initial state covariance
    """

    def __init__(self, k_init=1e-4, log_q_init=-8.0, log_r_init=-4.0,
                 log_p0_init=-2.0):
        super().__init__()
        self.k_raw = nn.Parameter(torch.tensor(_softplus_inv(k_init)))
        self.log_q = nn.Parameter(torch.tensor(log_q_init))
        self.log_r = nn.Parameter(torch.tensor(log_r_init))
        self.log_p0 = nn.Parameter(torch.tensor(log_p0_init))

    @property
    def k(self):
        return F.softplus(self.k_raw)

    @property
    def q(self):
        return torch.exp(self.log_q)

    @property
    def R(self):
        return torch.exp(self.log_r)

    @property
    def P0(self):
        return torch.exp(self.log_p0)

    def kf_predict(self, s, P, T_air, dt):
        """
        Predict step.
        Args:
            s: [B, 1] state (temperature)
            P: [B, 1, 1] covariance
            T_air: [B] air temperature
            dt: [B] time step (seconds)
        Returns:
            s_pred: [B, 1], P_pred: [B, 1, 1]
        """
        k = self.k
        T = s[:, 0]  # [B]

        # Newton's cooling: T_{t+1} = T - k*(T - T_air)*dt
        T_pred = T - k * (T - T_air) * dt  # [B]
        s_pred = T_pred.unsqueeze(1)  # [B, 1]

        # Jacobian: dT_pred/dT = 1 - k*dt
        F_scalar = 1.0 - k * dt  # [B]

        # Process noise: Q = q * dt, broadcast to [B]
        Q = self.q * dt  # [B]

        # P_pred = F^2 * P + Q  (all scalar per batch element)
        P_pred = (F_scalar.unsqueeze(1).unsqueeze(2) ** 2 * P
                  + Q.unsqueeze(1).unsqueeze(2))  # [B, 1, 1]
        P_pred = torch.clamp(P_pred, min=1e-10, max=1e6)

        return s_pred, P_pred

    def kf_update(self, s_pred, P_pred, y_obs):
        """
        Update step.
        Args:
            s_pred: [B, 1]
            P_pred: [B, 1, 1]
            y_obs: [B] observed temperature
        Returns:
            s_upd: [B, 1], P_upd: [B, 1, 1]
        """
        R = self.R
        T_pred = s_pred[:, 0]  # [B]
        innovation = y_obs - T_pred  # [B]

        # H = [1], S = P + R
        S = P_pred[:, 0, 0] + R  # [B]
        K = P_pred[:, 0, 0] / S  # [B] scalar Kalman gain

        # State update
        T_upd = T_pred + K * innovation  # [B]
        s_upd = T_upd.unsqueeze(1)  # [B, 1]

        # Joseph form (1D): P_upd = (1-K)*P*(1-K) + K*R*K
        IKH = 1.0 - K  # [B]
        P_upd = (IKH ** 2 * P_pred[:, 0, 0] + K ** 2 * R).unsqueeze(1).unsqueeze(2)

        return s_upd, P_upd

    def forward(self, T_obs, T_air, wind, par, dt, L_hist):
        """
        Full forward pass: filter over history, then predict into future.

        Args:
            T_obs:  [B, L+H] observed water temperature (L history + H future)
            T_air:  [B, L+H] air temperature
            wind:   [B, L+H] wind speed
            par:    [B, L+H] PAR
            dt:     [B, L+H] time deltas (seconds)
            L_hist: int, number of history steps (filter phase)

        Returns:
            T_pred: [B, H] predicted temperature
            T_var:  [B, H] predicted variance
        """
        B, T_total = T_obs.shape
        H = T_total - L_hist
        device = T_obs.device

        # Initialize state from first observation
        s = T_obs[:, 0:1].clone()  # [B, 1]
        P = self.P0.unsqueeze(0).unsqueeze(1).expand(B, 1, 1).clone()  # [B, 1, 1]

        # Filter phase: t=1..L_hist-1
        for t in range(1, L_hist):
            dt_t = dt[:, t].clamp(min=1.0)
            T_air_t = T_air[:, t - 1]
            s, P = self.kf_predict(s, P, T_air_t, dt_t)
            s, P = self.kf_update(s, P, T_obs[:, t])

        # Predict phase: t=L_hist..L_hist+H-1
        T_preds = []
        T_vars = []
        for t in range(H):
            idx = L_hist + t
            dt_t = dt[:, idx].clamp(min=1.0)
            T_air_t = T_air[:, idx - 1]
            s, P = self.kf_predict(s, P, T_air_t, dt_t)
            T_preds.append(s[:, 0])
            T_vars.append(P[:, 0, 0])

        return torch.stack(T_preds, dim=1), torch.stack(T_vars, dim=1)

    def param_summary(self):
        with torch.no_grad():
            k_val = self.k.item()
            return {
                'k': k_val,
                'tau_hours': (1.0 / k_val / 3600) if k_val > 1e-12 else float('inf'),
                'q': self.q.item(),
                'R': self.R.item(),
                'P0': self.P0.item(),
            }


# ==============================================================================
#  STAGE 2: Closure Discovery -- 5-term library
# ==============================================================================

class NeonKFClosure(nn.Module):
    """
    1-state KF with closure library for lake temperature.

    Physics floor (frozen from S1):
        T_{t+1} = T_t - k*(T_t - T_air)*dt

    Closure library (5 terms):
        C = theta_1 * PAR            (direct solar heating)
          + theta_2 * PAR^2          (non-linear solar heating)
          + theta_3 * wind           (evaporative wind cooling)
          + theta_4 * (T - T_air)    (sensible heat exchange)
          + theta_5 * wind*(T-T_air) (forced convection)

    T_{t+1} = T_t - k*(T_t - T_air)*dt + C*dt + process_noise
    """

    def __init__(self, k_init=1e-4, log_q_init=-8.0, log_r_init=-4.0,
                 log_p0_init=-2.0,
                 par_lin_init=0.0, par_quad_init=0.0,
                 wind_cool_init=0.0, sensible_init=0.0,
                 forced_conv_init=0.0):
        super().__init__()

        # Physics parameters (will be frozen after init from S1)
        self.k_raw = nn.Parameter(torch.tensor(_softplus_inv(max(k_init, 1e-8))))
        self.log_q = nn.Parameter(torch.tensor(log_q_init))
        self.log_r = nn.Parameter(torch.tensor(log_r_init))
        self.log_p0 = nn.Parameter(torch.tensor(log_p0_init))
        self.log_q_scale = nn.Parameter(torch.tensor(0.0))

        # Closure parameters (all unconstrained - can be + or -)
        self.theta_par_lin = nn.Parameter(torch.tensor(float(par_lin_init)))
        self.theta_par_quad = nn.Parameter(torch.tensor(float(par_quad_init)))
        self.theta_wind_cool = nn.Parameter(torch.tensor(float(wind_cool_init)))
        self.theta_sensible = nn.Parameter(torch.tensor(float(sensible_init)))
        self.theta_forced_conv = nn.Parameter(torch.tensor(float(forced_conv_init)))

    @property
    def k(self):
        return F.softplus(self.k_raw)

    @property
    def q(self):
        return torch.exp(self.log_q)

    @property
    def q_scale(self):
        return torch.exp(self.log_q_scale)

    @property
    def R(self):
        return torch.exp(self.log_r)

    @property
    def P0(self):
        return torch.exp(self.log_p0)

    def closure(self, T_state, T_air, wind, par):
        """
        Compute closure contribution (before *dt).
        Returns: [B] closure acceleration (degC/s)
        """
        dT = T_state - T_air  # [B]
        return (self.theta_par_lin * par
                + self.theta_par_quad * par ** 2
                + self.theta_wind_cool * wind
                + self.theta_sensible * dT
                + self.theta_forced_conv * wind * dT)

    def kf_predict(self, s, P, T_air, wind, par, dt):
        """
        Predict step with closure.
        Args:
            s: [B, 1], P: [B, 1, 1]
            T_air, wind, par: [B] forcing
            dt: [B] time step (seconds)
        """
        k = self.k
        T = s[:, 0]  # [B]
        dT = T - T_air

        # Physics + closure
        cl = self.closure(T, T_air, wind, par)
        T_pred = T - k * dT * dt + cl * dt  # [B]
        T_pred = torch.clamp(T_pred, min=-50.0, max=100.0)  # physical bounds
        s_pred = T_pred.unsqueeze(1)  # [B, 1]

        # EKF Jacobian: dT_pred/dT = 1 - k*dt + d(closure)/dT * dt
        # d(closure)/dT = theta_sensible + theta_forced_conv * wind
        dcl_dT = self.theta_sensible + self.theta_forced_conv * wind
        F_scalar = 1.0 - k * dt + dcl_dT * dt  # [B]
        F_scalar = torch.clamp(F_scalar, min=-2.0, max=2.0)

        # Process noise with q_scale
        qs = self.q_scale
        Q = qs * self.q * dt  # scalar, broadcast

        P_pred = (F_scalar.unsqueeze(1).unsqueeze(2) ** 2 * P
                  + Q)  # [B, 1, 1]
        P_pred = torch.clamp(P_pred, min=1e-10, max=1e6)

        return s_pred, P_pred

    def kf_update(self, s_pred, P_pred, y_obs):
        """Update step (same as NeonKF)."""
        R = self.R
        T_pred = s_pred[:, 0]
        innovation = y_obs - T_pred
        S = P_pred[:, 0, 0] + R
        K = P_pred[:, 0, 0] / S
        T_upd = T_pred + K * innovation
        s_upd = T_upd.unsqueeze(1)
        IKH = 1.0 - K
        P_upd = (IKH ** 2 * P_pred[:, 0, 0] + K ** 2 * R).unsqueeze(1).unsqueeze(2)
        return s_upd, P_upd

    def forward(self, T_obs, T_air, wind, par, dt, L_hist):
        """
        Full forward pass: filter + predict.

        Args:
            T_obs:  [B, L+H]
            T_air:  [B, L+H]
            wind:   [B, L+H]
            par:    [B, L+H]
            dt:     [B, L+H]
            L_hist: int

        Returns:
            T_pred: [B, H], T_var: [B, H]
        """
        B, T_total = T_obs.shape
        H = T_total - L_hist

        s = T_obs[:, 0:1].clone()  # [B, 1]
        P = self.P0.unsqueeze(0).unsqueeze(1).expand(B, 1, 1).clone()

        # Filter phase
        for t in range(1, L_hist):
            dt_t = dt[:, t].clamp(min=1.0)
            s, P = self.kf_predict(s, P, T_air[:, t-1], wind[:, t-1],
                                    par[:, t-1], dt_t)
            s, P = self.kf_update(s, P, T_obs[:, t])

        # Predict phase
        T_preds, T_vars = [], []
        for t in range(H):
            idx = L_hist + t
            dt_t = dt[:, idx].clamp(min=1.0)
            s, P = self.kf_predict(s, P, T_air[:, idx-1], wind[:, idx-1],
                                    par[:, idx-1], dt_t)
            T_preds.append(s[:, 0])
            T_vars.append(P[:, 0, 0])

        return torch.stack(T_preds, dim=1), torch.stack(T_vars, dim=1)

    def freeze_physics(self):
        """Freeze physics + noise + P0 for Stage 2 closure training."""
        for name in ['k_raw', 'log_q', 'log_r', 'log_p0']:
            getattr(self, name).requires_grad_(False)

    def closure_params_list(self):
        """Trainable closure params for Stage 2 optimizer."""
        return [self.theta_par_lin, self.theta_par_quad,
                self.theta_wind_cool, self.theta_sensible,
                self.theta_forced_conv, self.log_q_scale]

    def closure_summary(self):
        with torch.no_grad():
            return {
                'par_lin': self.theta_par_lin.item(),
                'par_quad': self.theta_par_quad.item(),
                'wind_cool': self.theta_wind_cool.item(),
                'sensible': self.theta_sensible.item(),
                'forced_conv': self.theta_forced_conv.item(),
                'q_scale': self.q_scale.item(),
            }

    def param_summary(self):
        with torch.no_grad():
            k_val = self.k.item()
            d = {
                'k': k_val,
                'tau_hours': (1.0 / k_val / 3600) if k_val > 1e-12 else float('inf'),
                'q': self.q.item(),
                'q_scale': self.q_scale.item(),
                'R': self.R.item(),
                'P0': self.P0.item(),
            }
            d.update(self.closure_summary())
            return d

    def symbolic_law(self):
        cs = self.closure_summary()
        parts = []
        labels = [
            ('par_lin', 'PAR'), ('par_quad', 'PAR^2'),
            ('wind_cool', 'wind'), ('sensible', '(T-T_air)'),
            ('forced_conv', 'wind*(T-T_air)'),
        ]
        for key, label in labels:
            val = cs[key]
            if abs(val) > 1e-8:
                sign = '+' if val >= 0 else '-'
                parts.append(f" {sign} {abs(val):.6f}*{label}")
        return f"closure(T, T_air, wind, PAR) ={''.join(parts) if parts else ' 0'}"
