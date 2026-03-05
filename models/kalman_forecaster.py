"""
Kalman-filter-based forecaster with physics-informed dynamics.

Replaces the neural SSM (64-dim opaque latent, ~50K params) with a
physics-based Kalman filter (~10 interpretable params).

State: s = [x, u]  where x = ball displacement, u = ball velocity (hidden)
Dynamics (with optional restoring force):
    x_{k+1} = x_k + u_k * dt
    u_{k+1} = rho * u_k - kappa * x_k * dt + c * g(v_k) * dt
where rho = exp(-alpha * dt), kappa >= 0 is the restoring force strength.

Transition matrix:
    F_k = [[1, dt], [-kappa*dt, rho]]

Observation: y = x + noise

Forcing function: g(v) = relu(v^2 - v_c^2)

All parameters are learnable and constrained to valid ranges via
softplus/exp reparameterization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KalmanForecaster(nn.Module):
    """
    Kalman-filter-based forecaster with physics-informed dynamics.

    State: s = [x, u] where x = ball displacement, u = ball velocity (hidden)
    Dynamics:
        x' = x + u*dt
        u' = rho*u - kappa*x*dt + c*g(v)*dt
    where rho = exp(-alpha*dt), kappa is a stabilizing restoring force.
    Observation: y = x + noise

    Forward pass:
      1. Filter phase: run KF predict+update over L history steps
      2. Predict phase: run KF predict-only over H future steps
      3. Output: predicted x, predicted variance, estimated u
    """

    def __init__(
        self,
        alpha_init=0.5,
        c_init=0.0,
        vc_init=0.1,
        log_qx_init=-8.0,
        log_qu_init=-8.0,
        log_r_init=-7.0,
        log_p0_xx_init=-8.0,
        log_p0_uu_init=-4.5,
        kappa_init=0.0,
        use_kappa=False,
        vc_min=0.0,
        use_gamma_gate=False,
        gate_filter=False,
        use_rv_drag=False,
        delta_init=0.1,
    ):
        """
        Args:
            alpha_init: Initial velocity decay rate (tau = 1/alpha ~ 2s)
            c_init: Initial forcing gain
            vc_init: Initial critical velocity threshold
            log_qx_init: log of position process noise rate (per second)
            log_qu_init: log of velocity process noise rate (per second)
            log_r_init: log of observation noise variance
            log_p0_xx_init: log of initial position covariance
            log_p0_uu_init: log of initial velocity covariance (large = uncertain)
            kappa_init: Initial restoring force strength (>= 0)
            use_kappa: If True, kappa is a learnable parameter; if False, kappa=0
            vc_min: Hard floor on critical velocity (prevents threshold collapse)
            use_gamma_gate: If True, adds learnable gamma gate on forcing term
            gate_filter: If True AND use_gamma_gate=True, gamma applies in
                filter phase too (not just predict). Default False.
            use_rv_drag: If True, adds quadratic relative-velocity drag
                delta*(v-u)*|v-u| to velocity dynamics.
            delta_init: Initial value for RV drag coefficient (>= 0).
        """
        super().__init__()

        self.use_kappa = use_kappa
        self.vc_min = vc_min
        self.use_gamma_gate = use_gamma_gate
        self.gate_filter = gate_filter
        self.use_rv_drag = use_rv_drag

        # --- Learnable physics parameters ---

        # Velocity decay: rho = exp(-alpha * dt), alpha > 0 via softplus
        # Inverse softplus: raw = log(exp(val) - 1)
        alpha_raw_init = math.log(math.exp(alpha_init) - 1.0 + 1e-6)
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_raw_init))

        # Forcing gain (unconstrained)
        self.c = nn.Parameter(torch.tensor(c_init))

        # Critical velocity threshold: v_c >= vc_min via softplus + offset
        vc_effective = max(vc_init - vc_min, 0.01)
        vc_raw_init = math.log(math.exp(vc_effective) - 1.0 + 1e-6)
        self.vc_raw = nn.Parameter(torch.tensor(vc_raw_init))

        # Restoring force: kappa >= 0 via softplus (only if use_kappa=True)
        if self.use_kappa:
            kappa_raw_init = math.log(math.exp(kappa_init + 1e-6) - 1.0 + 1e-6)
            self.kappa_raw = nn.Parameter(torch.tensor(kappa_raw_init))

        # Forcing gate: gamma >= 0 via softplus (only if use_gamma_gate=True)
        # gamma multiplies the forcing term during the predict (forecast) phase.
        # Initialized to gamma = 1.0 so baseline behavior is unchanged at start.
        if self.use_gamma_gate:
            gamma_raw_init = math.log(math.exp(1.0) - 1.0)  # softplus(0.5413) = 1.0
            self.gamma_raw = nn.Parameter(torch.tensor(gamma_raw_init))

        # Quadratic relative-velocity drag: delta >= 0 via softplus
        # (only if use_rv_drag=True). Adds delta*(v-u)*|v-u|*dt to u dynamics.
        if self.use_rv_drag:
            delta_raw_init = math.log(math.exp(max(delta_init, 1e-6)) - 1.0 + 1e-6)
            self.delta_raw = nn.Parameter(torch.tensor(delta_raw_init))

        # --- Learnable noise parameters (log-parameterized for positivity) ---

        self.log_qx = nn.Parameter(torch.tensor(log_qx_init))  # position process noise rate
        self.log_qu = nn.Parameter(torch.tensor(log_qu_init))  # velocity process noise rate
        self.log_r = nn.Parameter(torch.tensor(log_r_init))    # observation noise variance

        # --- Initial covariance (diagonal, log-parameterized) ---

        self.log_p0_xx = nn.Parameter(torch.tensor(log_p0_xx_init))  # initial position variance
        self.log_p0_uu = nn.Parameter(torch.tensor(log_p0_uu_init))  # initial velocity variance

    # ---- Constrained parameter accessors ----

    @property
    def alpha(self):
        """Decay rate, always > 0."""
        return F.softplus(self.alpha_raw)

    @property
    def vc(self):
        """Critical velocity threshold, always >= vc_min."""
        return F.softplus(self.vc_raw) + self.vc_min

    @property
    def kappa(self):
        """Restoring force strength, always >= 0. Returns 0 if use_kappa=False."""
        if self.use_kappa:
            return F.softplus(self.kappa_raw)
        return torch.tensor(0.0, device=self.alpha_raw.device, dtype=self.alpha_raw.dtype)

    @property
    def gamma(self):
        """Forcing gate, always >= 0. Returns 1.0 (no gating) if use_gamma_gate=False."""
        if self.use_gamma_gate:
            return F.softplus(self.gamma_raw)
        return 1.0

    @property
    def delta(self):
        """RV drag coefficient, always >= 0. Returns 0 if use_rv_drag=False."""
        if self.use_rv_drag:
            return F.softplus(self.delta_raw)
        return torch.tensor(0.0, device=self.alpha_raw.device, dtype=self.alpha_raw.dtype)

    @property
    def qx(self):
        """Position process noise rate, always > 0."""
        return torch.exp(self.log_qx)

    @property
    def qu(self):
        """Velocity process noise rate, always > 0."""
        return torch.exp(self.log_qu)

    @property
    def R(self):
        """Observation noise variance, always > 0."""
        return torch.exp(self.log_r)

    @property
    def P0(self):
        """Initial covariance matrix [2, 2], diagonal PSD."""
        p_xx = torch.exp(self.log_p0_xx)
        p_uu = torch.exp(self.log_p0_uu)
        P = torch.zeros(2, 2, device=self.log_p0_xx.device, dtype=self.log_p0_xx.dtype)
        P[0, 0] = p_xx
        P[1, 1] = p_uu
        return P

    # ---- Physics functions ----

    def forcing(self, v):
        """
        Forcing function: g(v) = relu(v^2 - v_c^2).

        Args:
            v: [B] or [B, 1] water velocity (physical units)
        Returns:
            g: same shape, forcing magnitude
        """
        vc = self.vc
        return F.relu(v * v - vc * vc)

    # ---- Kalman filter steps ----

    def kf_predict(self, s, P, v, dt, gamma=1.0):
        """
        Kalman filter predict step: (k-1|k-1) -> (k|k-1).

        Args:
            s: [B, 2] state [x, u]
            P: [B, 2, 2] covariance
            v: [B] water velocity at k-1 (physical units)
            dt: [B] time delta to step k (physical units, > 0)
            gamma: scalar gate on the forcing term (default 1.0 = no gating)
        Returns:
            s_pred: [B, 2] predicted state
            P_pred: [B, 2, 2] predicted covariance
        """
        B = s.shape[0]
        alpha = self.alpha  # scalar > 0
        c = self.c          # scalar
        kap = self.kappa    # scalar >= 0
        delt = self.delta   # scalar >= 0 (0 if use_rv_drag=False)

        # Damping factor per sample: rho_k = exp(-alpha * dt_k)
        rho = torch.exp(-alpha * dt)  # [B]

        # State prediction
        x_old = s[:, 0]  # [B]
        u_old = s[:, 1]  # [B]

        # Relative velocity: v_water - u_particle
        rel_v = v - u_old  # [B]

        # dt-consistent dynamics:
        #   x_k = x_{k-1} + u_{k-1} * dt
        #   u_k = rho * u_{k-1} - kappa * x_{k-1} * dt + c * g(v_{k-1}) * dt
        #         + delta * (v - u) * |v - u| * dt    [RV drag, if enabled]
        x_pred = x_old + u_old * dt                                                 # [B]
        u_pred = (rho * u_old - kap * x_old * dt + gamma * c * self.forcing(v) * dt
                  + delt * rel_v * torch.abs(rel_v) * dt)                            # [B]

        s_pred = torch.stack([x_pred, u_pred], dim=1)  # [B, 2]

        # EKF Jacobian: d(u_pred)/du includes RV drag derivative
        # d/du [delta*(v-u)*|v-u|] = -2*delta*|v-u|
        F_mat = torch.zeros(B, 2, 2, device=s.device, dtype=s.dtype)
        F_mat[:, 0, 0] = 1.0
        F_mat[:, 0, 1] = dt
        F_mat[:, 1, 0] = -kap * dt
        F_mat[:, 1, 1] = rho - 2.0 * delt * torch.abs(rel_v) * dt

        # Process noise Q_k = diag(qx * dt, qu * dt) — scales with dt
        qx = self.qx
        qu = self.qu
        Q = torch.zeros(B, 2, 2, device=s.device, dtype=s.dtype)
        Q[:, 0, 0] = qx * dt
        Q[:, 1, 1] = qu * dt

        # Covariance prediction: P_pred = F @ P @ F^T + Q
        P_pred = torch.bmm(torch.bmm(F_mat, P), F_mat.transpose(1, 2)) + Q

        return s_pred, P_pred

    def kf_update(self, s_pred, P_pred, y_obs):
        """
        Kalman filter update step: (k|k-1) -> (k|k) using Joseph form.

        Args:
            s_pred: [B, 2] predicted state
            P_pred: [B, 2, 2] predicted covariance
            y_obs: [B] observation (displacement only)
        Returns:
            s_upd: [B, 2] updated state
            P_upd: [B, 2, 2] updated covariance (Joseph form, guaranteed PSD)
        """
        B = s_pred.shape[0]
        R = self.R  # scalar

        # H = [1, 0]: observe x only
        # innovation = y - H @ s_pred = y - x_pred
        x_pred = s_pred[:, 0]  # [B]
        innovation = y_obs - x_pred  # [B]

        # Innovation covariance: S = H @ P @ H^T + R = P[0,0] + R (scalar per sample)
        S = P_pred[:, 0, 0] + R  # [B]

        # Kalman gain: K = P @ H^T / S = P[:, 0] / S  -> [B, 2]
        K = P_pred[:, :, 0] / S.unsqueeze(1)  # [B, 2]

        # State update: s = s_pred + K * innovation
        s_upd = s_pred + K * innovation.unsqueeze(1)  # [B, 2]

        # Joseph form covariance update: P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
        # I - K @ H:  H = [1, 0], so K @ H = K[:, :, None] @ H[None, :] = outer product
        # IKH[i,j] = I[i,j] - K[i] * H[j]
        eye = torch.eye(2, device=s_pred.device, dtype=s_pred.dtype).unsqueeze(0)  # [1, 2, 2]
        # K @ H = K.unsqueeze(2) @ H.unsqueeze(0) where H = [1, 0]
        # Simpler: IKH[:, i, j] = eye[i,j] - K[:, i] * H[j]
        H_vec = torch.tensor([1.0, 0.0], device=s_pred.device, dtype=s_pred.dtype)  # [2]
        KH = K.unsqueeze(2) * H_vec.unsqueeze(0).unsqueeze(0)  # [B, 2, 1] * [1, 1, 2] = [B, 2, 2]
        IKH = eye - KH  # [B, 2, 2]

        # P_upd = IKH @ P_pred @ IKH^T + K * R * K^T
        P_joseph = torch.bmm(torch.bmm(IKH, P_pred), IKH.transpose(1, 2))
        KKT = K.unsqueeze(2) * K.unsqueeze(1)  # [B, 2, 2]
        P_upd = P_joseph + R * KKT

        return s_upd, P_upd

    # ---- Forward pass ----

    def forward(self, v_hist, dt_hist, x_obs_hist, v_fut, dt_fut):
        """
        Full forward pass: filter over history, then predict into future.

        Args:
            v_hist: [B, L] water velocity history (physical units)
            dt_hist: [B, L] time delta history (physical units, seconds)
            x_obs_hist: [B, L] displacement history (physical units, observations)
            v_fut: [B, H] future water velocity (physical units)
            dt_fut: [B, H] future time delta (physical units, seconds)

        Returns:
            x_pred: [B, H] predicted displacement (mean)
            x_var: [B, H] predicted displacement variance (for uncertainty)
            u_est: [B, H] estimated ball velocity at each future step
        """
        B, L = v_hist.shape
        H = v_fut.shape[1]
        device = v_hist.device
        dtype = v_hist.dtype

        # --- Initialize state from first observation ---
        # s0 = [x_obs[0], 0]  (observe position, assume zero initial velocity)
        s = torch.zeros(B, 2, device=device, dtype=dtype)
        s[:, 0] = x_obs_hist[:, 0]

        # P0: learned initial covariance, broadcast to batch
        P = self.P0.unsqueeze(0).expand(B, -1, -1).clone()  # [B, 2, 2]

        # Gamma gate value (1.0 when gate disabled; softplus(gamma_raw) when enabled)
        gamma_val = self.gamma

        # --- Filter phase: iterate over history steps 1..L-1 ---
        # At each step k: predict using (v[k-1], dt[k]) then update with x_obs[k]
        for k in range(1, L):
            v_k_minus_1 = v_hist[:, k - 1]      # [B] forcing from previous step
            dt_k = dt_hist[:, k]                  # [B] time delta to current step
            y_k = x_obs_hist[:, k]                # [B] observation at current step

            # Clamp dt to avoid numerical issues (should be positive in physical units)
            dt_k = dt_k.clamp(min=1e-6)

            if self.gate_filter:
                s, P = self.kf_predict(s, P, v_k_minus_1, dt_k, gamma=gamma_val)
            else:
                s, P = self.kf_predict(s, P, v_k_minus_1, dt_k)
            s, P = self.kf_update(s, P, y_k)

        # --- Predict phase: iterate over future steps 0..H-1 ---
        # After filtering, s and P represent the state at time L-1
        # Each predict step advances using v_fut[k] and dt_fut[k]
        x_preds = []
        x_vars = []
        u_ests = []

        for k in range(H):
            v_k = v_fut[:, k]    # [B] forcing for this step
            dt_k = dt_fut[:, k]  # [B] time delta
            dt_k = dt_k.clamp(min=1e-6)

            s, P = self.kf_predict(s, P, v_k, dt_k, gamma=gamma_val)

            x_preds.append(s[:, 0])       # predicted x
            x_vars.append(P[:, 0, 0])     # variance of x
            u_ests.append(s[:, 1])         # estimated u

        x_pred = torch.stack(x_preds, dim=1)  # [B, H]
        x_var = torch.stack(x_vars, dim=1)    # [B, H]
        u_est = torch.stack(u_ests, dim=1)    # [B, H]

        return x_pred, x_var, u_est

    # ---- Utility: readable parameter summary ----

    def param_summary(self):
        """Return dict of interpretable parameter values (for logging)."""
        with torch.no_grad():
            alpha_val = self.alpha.item()
            kappa_val = self.kappa.item()
            d = {
                'alpha': alpha_val,
                'tau': 1.0 / alpha_val if alpha_val > 1e-8 else float('inf'),
                'kappa': kappa_val,
                'c': self.c.item(),
                'vc': self.vc.item(),
                'qx': self.qx.item(),
                'qu': self.qu.item(),
                'R': self.R.item(),
                'P0_xx': torch.exp(self.log_p0_xx).item(),
                'P0_uu': torch.exp(self.log_p0_uu).item(),
            }
            if self.use_gamma_gate:
                gamma_val = self.gamma
                d['gamma'] = gamma_val.item() if torch.is_tensor(gamma_val) else gamma_val
            if self.use_rv_drag:
                d['delta'] = self.delta.item()
            return d
