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
