"""Nominal and true dynamics for 2D navigation (pre-stabilized double integrator).

The plant is a pre-stabilized double integrator with an optional *quadratic
velocity drag* term, making the model mildly nonlinear:

    pos⁺ = pos + dt · vel
    acc  = −k_p · pos − k_d · vel + u − c_drag · ‖vel‖₂ · vel
    vel⁺ = vel + dt · acc

The drag is purely dissipative (it does non-positive work on the velocity), so
it only adds damping to the already-stable pre-stabilized loop. ``c_drag = 0``
recovers the original linear model exactly. Nominal and true dynamics share the
same integrator, so when both use the same ``drag_coeff`` the PB disturbance
reconstruction ``w = x⁺_true − f_nom(x, u)`` stays clean (process noise only).
"""

from __future__ import annotations

from typing import Optional

import torch

from pb_core import as_bt


def integrate_double_integrator(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    dt: float,
    pre_kp: float,
    pre_kd: float,
    drag_coeff: float = 0.0,
) -> torch.Tensor:
    """One explicit-Euler step of the (optionally drag-augmented) double integrator."""
    x = as_bt(x)
    u = as_bt(u)
    pos = x[..., :2]
    vel = x[..., 2:]

    pos_next = pos + dt * vel
    acc = -pre_kp * pos - pre_kd * vel + u
    if drag_coeff:
        # eps-smoothed speed so d/dvel is well-defined at vel = 0 (the initial
        # velocity is exactly zero, and the plant is inside the autograd graph).
        speed = torch.sqrt((vel * vel).sum(dim=-1, keepdim=True) + 1e-12)
        acc = acc - drag_coeff * speed * vel
    vel_next = vel + dt * acc
    return torch.cat([pos_next, vel_next], dim=-1)


class DoubleIntegratorNominal:
    """
    Pre-stabilized model around the origin.

    Closed-loop acceleration:
      a = -k_p * pos - k_d * vel + u - c_drag * ||vel|| * vel
    where u is the PB boost input and c_drag is the (optional) quadratic drag.
    """

    def __init__(self, dt: float = 0.05, pre_kp: float = 1.0, pre_kd: float = 1.5,
                 drag_coeff: float = 0.0):
        self.dt = dt
        self.pre_kp = pre_kp
        self.pre_kd = pre_kd
        self.drag_coeff = drag_coeff

    def _pre_stab_acc(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        return -self.pre_kp * pos - self.pre_kd * vel

    def nominal_dynamics(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        return integrate_double_integrator(
            x, u, dt=self.dt, pre_kp=self.pre_kp, pre_kd=self.pre_kd, drag_coeff=self.drag_coeff,
        )


class DoubleIntegratorTrue:
    """True dynamics: same integrator as the nominal model.

    With matched ``drag_coeff`` the true and nominal dynamics are identical (no
    model mismatch); disturbances enter only through the external process noise
    injected by the rollout.
    """

    def __init__(self, dt: float = 0.05, pre_kp: float = 1.0, pre_kd: float = 1.5,
                 drag_coeff: float = 0.0):
        self.dt = dt
        self.pre_kp = pre_kp
        self.pre_kd = pre_kd
        self.drag_coeff = drag_coeff

    def _pre_stab_acc(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        return -self.pre_kp * pos - self.pre_kd * vel

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        return integrate_double_integrator(
            x, u, dt=self.dt, pre_kp=self.pre_kp, pre_kd=self.pre_kd, drag_coeff=self.drag_coeff,
        )


class PayloadSwitchingDoubleIntegratorTrue:
    """Batch-bound double integrator whose control authority changes with payload.

    The nominal model remains the unit-mass ``DoubleIntegratorNominal``.  Before
    each rollout, call :meth:`set_payload_schedule` with one causal schedule per
    batch element.  At step ``t`` the true plant uses

    ``acc = -kp * pos - kd * vel + actuator_gain[t] / mass[t] * u
           - drag[t] * ||vel|| * vel + lateral_bias[t] * e_y``.

    Keeping the pre-stabiliser fixed mirrors a controller whose baseline gains
    are calibrated for a reference payload; mass, actuator effectiveness and
    centre-of-mass bias are therefore genuine, time-varying model mismatch.
    """

    def __init__(
        self,
        dt: float = 0.05,
        pre_kp: float = 1.0,
        pre_kd: float = 1.5,
    ) -> None:
        self.dt = float(dt)
        self.pre_kp = float(pre_kp)
        self.pre_kd = float(pre_kd)
        self._mass: Optional[torch.Tensor] = None
        self._actuator_gain: Optional[torch.Tensor] = None
        self._drag: Optional[torch.Tensor] = None
        self._lateral_bias: Optional[torch.Tensor] = None

    def set_payload_schedule(
        self,
        *,
        mass: torch.Tensor,
        actuator_gain: torch.Tensor,
        drag: torch.Tensor,
        lateral_bias: torch.Tensor,
    ) -> None:
        """Bind ``(B, horizon)`` regime tensors for the next rollout."""
        schedules = (mass, actuator_gain, drag, lateral_bias)
        if any(value.ndim != 2 for value in schedules):
            raise ValueError("Payload schedules must all have shape (batch, horizon).")
        shape = mass.shape
        if any(value.shape != shape for value in schedules[1:]):
            raise ValueError("Payload schedules must share one (batch, horizon) shape.")
        if torch.any(mass <= 0.0):
            raise ValueError("Payload mass must stay strictly positive.")
        if torch.any(actuator_gain <= 0.0):
            raise ValueError("Payload actuator gain must stay strictly positive.")
        self._mass = mass
        self._actuator_gain = actuator_gain
        self._drag = drag
        self._lateral_bias = lateral_bias

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        if self._mass is None or self._actuator_gain is None or self._drag is None or self._lateral_bias is None:
            raise RuntimeError("Call set_payload_schedule before rolling out the payload plant.")
        if t is None or t < 0 or t >= self._mass.shape[1]:
            raise ValueError(f"Payload plant needs an in-range step index, got t={t}.")

        x = as_bt(x)
        u = as_bt(u)
        if x.shape[0] != self._mass.shape[0]:
            raise ValueError(
                "Payload schedule batch size does not match rollout batch: "
                f"{self._mass.shape[0]} vs {x.shape[0]}."
            )

        pos = x[..., :2]
        vel = x[..., 2:]
        mass_t = self._mass[:, t].view(-1, 1, 1)
        gain_t = self._actuator_gain[:, t].view(-1, 1, 1)
        drag_t = self._drag[:, t].view(-1, 1, 1)
        bias_t = self._lateral_bias[:, t].view(-1, 1, 1)

        pos_next = pos + self.dt * vel
        acc = -self.pre_kp * pos - self.pre_kd * vel + (gain_t / mass_t) * u
        speed = torch.sqrt((vel * vel).sum(dim=-1, keepdim=True) + 1e-12)
        acc = acc - drag_t * speed * vel
        acc_y = acc[..., 1:2] + bias_t
        acc = torch.cat([acc[..., :1], acc_y], dim=-1)
        vel_next = vel + self.dt * acc
        return torch.cat([pos_next, vel_next], dim=-1)
