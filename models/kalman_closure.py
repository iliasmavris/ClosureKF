"""
Physics + Learned Closure Kalman filter.

State: z = [x, u]  (displacement, internal velocity)
Transition:
    x_{t+1} = x_t + u_t * dt
    u_{t+1} = rho * u_t - kappa * x_t * dt + c * relu(v^2 - vc^2) * dt
              + closure(u_t, v_t, dv_t) * dt + eta_t

where closure = -a1*u - d1*u^2 - d2*u*|v| - d3*u*|u| + b1*v + b2*dv
    u = internal velocity state, v = water velocity (exogenous), dv = v change

Constraints: a1,d1,d2,d3 >= 0 (softplus); b1,b2 free sign.
Observation: y_t = x_t + eps_t
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CLOSURE_PARAM_NAMES = ['a1', 'b1', 'b2', 'd1', 'd2', 'd3']


def _softplus_inv(x, eps=1e-6):
    """Stable inverse of softplus: log(exp(x) - 1) without overflow."""
    x = max(float(x), eps)
    return x + math.log(-math.expm1(-x))


class KalmanForecasterClosure(nn.Module):

    def __init__(
        self,
        alpha_init=0.5, c_init=1.0, vc_init=0.15, kappa_init=0.1,
        log_qx_init=-6.0, log_qu_init=-6.0, log_r_init=-5.0,
        log_p0_xx_init=-6.0, log_p0_uu_init=-4.0,
        vc_min=0.0,
        a1_init=0.1, b1_init=0.0, b2_init=0.0,
        d1_init=0.05, d2_init=0.5, d3_init=0.5,
        alpha_param="sigmoid",
    ):
        super().__init__()
        self.vc_min = vc_min
        self.alpha_param = alpha_param

        # --- Physics parameters (constrained) ---
        if alpha_param == "softplus":
            self.alpha_raw = nn.Parameter(
                torch.tensor(_softplus_inv(alpha_init)))
        else:  # "sigmoid" -- original behavior
            a = max(min(alpha_init, 0.999), 0.001)
            self.alpha_raw = nn.Parameter(
                torch.tensor(math.log(a / (1.0 - a))))

        c_safe = max(c_init, 0.01)
        self.c_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(c_safe) - 1.0 + 1e-6)))

        k_safe = max(kappa_init, 0.001)
        self.kappa_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(k_safe) - 1.0 + 1e-6)))

        vc_eff = max(vc_init - vc_min, 0.01)
        self.vc_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(vc_eff) - 1.0 + 1e-6)))

        # --- Noise parameters ---
        self.log_qx = nn.Parameter(torch.tensor(log_qx_init))
        self.log_qu = nn.Parameter(torch.tensor(log_qu_init))
        self.log_r = nn.Parameter(torch.tensor(log_r_init))
        self.log_q_scale = nn.Parameter(torch.tensor(0.0))

        # --- Initial covariance ---
        self.log_p0_xx = nn.Parameter(torch.tensor(log_p0_xx_init))
        self.log_p0_uu = nn.Parameter(torch.tensor(log_p0_uu_init))

        # --- Closure parameters ---
        def _sp_inv(x):
            return math.log(math.exp(max(x, 1e-4)) - 1.0 + 1e-6)

        self.a1_raw = nn.Parameter(torch.tensor(_sp_inv(a1_init)))
        self.d1_raw = nn.Parameter(torch.tensor(_sp_inv(d1_init)))
        self.d2_raw = nn.Parameter(torch.tensor(_sp_inv(d2_init)))
        self.d3_raw = nn.Parameter(torch.tensor(_sp_inv(d3_init)))
        self.b1 = nn.Parameter(torch.tensor(float(b1_init)))
        self.b2 = nn.Parameter(torch.tensor(float(b2_init)))

    # --- Constrained accessors ---

    @property
    def alpha(self):
        if self.alpha_param == "softplus":
            return F.softplus(self.alpha_raw)
        return torch.sigmoid(self.alpha_raw)

    @property
    def c(self):
        return F.softplus(self.c_raw)

    @property
    def kappa(self):
        return F.softplus(self.kappa_raw)

    @property
    def vc(self):
        return F.softplus(self.vc_raw) + self.vc_min

    @property
    def qx(self):
        return torch.exp(self.log_qx)

    @property
    def qu(self):
        return torch.exp(self.log_qu)

    @property
    def q_scale(self):
        return torch.exp(self.log_q_scale)

    @property
    def R(self):
        return torch.exp(self.log_r)

    @property
    def P0(self):
        P = torch.zeros(2, 2, device=self.log_p0_xx.device,
                         dtype=self.log_p0_xx.dtype)
        P[0, 0] = torch.exp(self.log_p0_xx)
        P[1, 1] = torch.exp(self.log_p0_uu)
        return P

    @property
    def a1(self):
        return F.softplus(self.a1_raw)

    @property
    def d1(self):
        return F.softplus(self.d1_raw)

    @property
    def d2(self):
        return F.softplus(self.d2_raw)

    @property
    def d3(self):
        return F.softplus(self.d3_raw)

    def forcing(self, v):
        vc = self.vc
        return F.relu(v * v - vc * vc)

    def closure(self, u_state, v_water, dv_water):
        """Closure contribution (before *dt)."""
        return (-self.a1 * u_state
                + self.b1 * v_water + self.b2 * dv_water
                - self.d1 * u_state ** 2
                - self.d2 * u_state * torch.abs(v_water)
                - self.d3 * u_state * torch.abs(u_state))

    def kf_predict(self, s, P, v_water, dv_water, dt,
                   collect_residuals=False):
        B = s.shape[0]
        rho = torch.exp(-self.alpha * dt)
        x_old, u_old = s[:, 0], s[:, 1]

        x_pred = x_old + u_old * dt
        physics_drift = (rho * u_old - self.kappa * x_old * dt
                         + self.c * self.forcing(v_water) * dt)
        cl = self.closure(u_old, v_water, dv_water)
        cl_dt = cl * dt
        u_pred = physics_drift + cl_dt
        s_pred = torch.stack([x_pred, u_pred], dim=1)

        # EKF Jacobian: d(u_pred)/du includes closure derivative
        # dcl/du = -a1 - 2*d1*u - d2*|v| - 2*d3*|u|
        dcl_du = (-self.a1
                  - 2.0 * self.d1 * u_old
                  - self.d2 * torch.abs(v_water)
                  - 2.0 * self.d3 * torch.abs(u_old))

        F_mat = torch.zeros(B, 2, 2, device=s.device, dtype=s.dtype)
        F_mat[:, 0, 0] = 1.0
        F_mat[:, 0, 1] = dt
        F_mat[:, 1, 0] = -self.kappa * dt
        F_mat[:, 1, 1] = rho + dcl_du * dt

        qs = self.q_scale
        Q = torch.zeros(B, 2, 2, device=s.device, dtype=s.dtype)
        Q[:, 0, 0] = qs * self.qx * dt
        Q[:, 1, 1] = qs * self.qu * dt

        P_pred = torch.bmm(torch.bmm(F_mat, P), F_mat.transpose(1, 2)) + Q

        if collect_residuals:
            return s_pred, P_pred, cl_dt, physics_drift
        return s_pred, P_pred

    def kf_update(self, s_pred, P_pred, y_obs):
        R = self.R
        innov = y_obs - s_pred[:, 0]
        S = P_pred[:, 0, 0] + R
        K = P_pred[:, :, 0] / S.unsqueeze(1)
        s_upd = s_pred + K * innov.unsqueeze(1)

        eye = torch.eye(2, device=s_pred.device, dtype=s_pred.dtype).unsqueeze(0)
        H_vec = torch.tensor([1.0, 0.0], device=s_pred.device, dtype=s_pred.dtype)
        KH = K.unsqueeze(2) * H_vec.unsqueeze(0).unsqueeze(0)
        IKH = eye - KH
        P_upd = (torch.bmm(torch.bmm(IKH, P_pred), IKH.transpose(1, 2))
                 + R * K.unsqueeze(2) * K.unsqueeze(1))
        return s_upd, P_upd

    def forward(self, v_hist, dt_hist, x_obs_hist, v_fut, dt_fut,
                collect_residuals=False):
        B, L = v_hist.shape
        H = v_fut.shape[1]
        dev = v_hist.device

        s = torch.zeros(B, 2, device=dev, dtype=v_hist.dtype)
        s[:, 0] = x_obs_hist[:, 0]
        P = self.P0.unsqueeze(0).expand(B, -1, -1).clone()

        all_cl = [] if collect_residuals else None
        all_ph = [] if collect_residuals else None

        for k in range(1, L):
            dt_k = dt_hist[:, k].clamp(min=1e-6)
            v_curr = v_hist[:, k - 1]
            v_prev = v_hist[:, k - 2] if k >= 2 else v_hist[:, 0]
            dv = v_curr - v_prev if k >= 2 else torch.zeros_like(v_curr)

            if collect_residuals:
                s, P, cl, phys = self.kf_predict(
                    s, P, v_curr, dv, dt_k, collect_residuals=True)
                all_cl.append(cl)
                all_ph.append(phys)
            else:
                s, P = self.kf_predict(s, P, v_curr, dv, dt_k)

            s, P = self.kf_update(s, P, x_obs_hist[:, k])

        x_preds, x_vars, u_ests = [], [], []
        for k in range(H):
            dt_k = dt_fut[:, k].clamp(min=1e-6)
            v_prev = v_hist[:, -1] if k == 0 else v_fut[:, k - 1]
            v_curr = v_fut[:, k]
            dv = v_curr - v_prev

            if collect_residuals:
                s, P, cl, phys = self.kf_predict(
                    s, P, v_curr, dv, dt_k, collect_residuals=True)
                all_cl.append(cl)
                all_ph.append(phys)
            else:
                s, P = self.kf_predict(s, P, v_curr, dv, dt_k)

            x_preds.append(s[:, 0])
            x_vars.append(P[:, 0, 0])
            u_ests.append(s[:, 1])

        result = (torch.stack(x_preds, dim=1),
                  torch.stack(x_vars, dim=1),
                  torch.stack(u_ests, dim=1))

        if collect_residuals:
            return result + (torch.stack(all_cl, dim=1),
                             torch.stack(all_ph, dim=1))
        return result

    def freeze_physics(self):
        """Freeze physics + R + base Q + P0 for Stage 2."""
        for name in ['alpha_raw', 'c_raw', 'kappa_raw', 'vc_raw',
                      'log_r', 'log_qx', 'log_qu',
                      'log_p0_xx', 'log_p0_uu']:
            getattr(self, name).requires_grad_(False)

    def closure_params_list(self):
        """Trainable params for closure stage: 6 closure + 1 Q scale."""
        return [self.a1_raw, self.b1, self.b2,
                self.d1_raw, self.d2_raw, self.d3_raw,
                self.log_q_scale]

    def closure_summary(self):
        with torch.no_grad():
            return {
                'a1': self.a1.item(), 'b1': self.b1.item(),
                'b2': self.b2.item(), 'd1': self.d1.item(),
                'd2': self.d2.item(), 'd3': self.d3.item(),
                'q_scale': self.q_scale.item(),
            }

    def param_summary(self):
        with torch.no_grad():
            a = self.alpha.item()
            d = {
                'alpha': a,
                'tau': 1.0 / a if a > 1e-8 else float('inf'),
                'c': self.c.item(), 'vc': self.vc.item(),
                'kappa': self.kappa.item(),
                'qx': self.qx.item(), 'qu': self.qu.item(),
                'q_scale': self.q_scale.item(),
                'R': self.R.item(),
                'P0_xx': torch.exp(self.log_p0_xx).item(),
                'P0_uu': torch.exp(self.log_p0_uu).item(),
            }
            d.update(self.closure_summary())
            return d

    def symbolic_law(self):
        """Print closure law using code variables: u=slip state, v=water."""
        cs = self.closure_summary()
        parts = [f"closure(u, v, dv) = -{cs['a1']:.4f}*u"]
        for name, key in [('v', 'b1'), ('dv', 'b2')]:
            val = cs[key]
            sign = '+' if val >= 0 else '-'
            parts.append(f" {sign} {abs(val):.4f}*{name}")
        for name, key in [('u^2', 'd1'), ('u*|v|', 'd2'), ('u*|u|', 'd3')]:
            parts.append(f" - {cs[key]:.4f}*{name}")
        return ''.join(parts)
