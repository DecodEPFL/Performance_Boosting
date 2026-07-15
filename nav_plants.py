"""Nominal and true dynamics for pre-stabilized 2D navigation.

The legacy plant is a pre-stabilized double integrator with an optional
*quadratic velocity drag* term, making the model mildly nonlinear:

    pos⁺ = pos + dt · vel
    acc  = −k_p · pos − k_d · vel + u − c_drag · ‖vel‖₂ · vel
    vel⁺ = vel + dt · acc

The drag is purely dissipative (it does non-positive work on the velocity), so
it only adds damping to the already-stable pre-stabilized loop. ``c_drag = 0``
recovers the original linear model exactly. Nominal and true dynamics share the
same integrator, so when both use the same ``drag_coeff`` the PB disturbance
reconstruction ``w = x⁺_true − f_nom(x, u)`` stays clean (process noise only).

The module also provides a strongly nonlinear robot model with a radial
Duffing restoring force, viscous/quadratic/Coulomb friction, gyroscopic
cross-axis coupling, a smooth actuator dead-zone and saturation, speed-related
authority loss, and lateral traction loss.  Its restoring and dissipative
terms keep the unforced origin pre-stabilized by construction.  Nominal and
true wrappers share one transition function so PB disturbance reconstruction
continues to contain only externally injected process noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class NonlinearRobotConfig:
    """Physical parameters for the strongly nonlinear navigation robot.

    The state remains ``(p_x, p_y, v_x, v_y)`` and the command remains a 2D
    force/acceleration request, so existing controllers and checkpoints keep
    their dimensions.  The defaults deliberately make every nonlinear effect
    visible without making the standard gate arena numerically stiff.
    """

    dt: float = 0.05
    pre_kp: float = 0.32
    pre_kd: float = 0.80
    mass: float = 1.0
    cubic_stiffness: float = 0.10
    quadratic_drag: float = 0.35
    coulomb_friction: float = 0.06
    friction_velocity: float = 0.12
    gyro_gain: float = 0.18
    gyro_position_scale: float = 1.0
    actuator_limit: float = 2.5
    actuator_deadzone: float = 0.06
    speed_loss: float = 0.20
    lateral_slip: float = 0.30
    physics_substeps: int = 2

    def __post_init__(self) -> None:
        scalar_parameters = {
            name: value
            for name, value in vars(self).items()
            if name != "physics_substeps"
        }
        for name, value in scalar_parameters.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value}.")
        substeps = float(self.physics_substeps)
        if not math.isfinite(substeps) or not substeps.is_integer():
            raise ValueError(
                "physics_substeps must be a finite integer, "
                f"got {self.physics_substeps}."
            )

        strictly_positive = {
            "dt": self.dt,
            "mass": self.mass,
            "friction_velocity": self.friction_velocity,
            "actuator_limit": self.actuator_limit,
        }
        for name, value in strictly_positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be strictly positive, got {value}.")

        nonnegative = {
            "pre_kp": self.pre_kp,
            "pre_kd": self.pre_kd,
            "cubic_stiffness": self.cubic_stiffness,
            "quadratic_drag": self.quadratic_drag,
            "coulomb_friction": self.coulomb_friction,
            "gyro_gain": self.gyro_gain,
            "gyro_position_scale": self.gyro_position_scale,
            "actuator_deadzone": self.actuator_deadzone,
            "speed_loss": self.speed_loss,
            "lateral_slip": self.lateral_slip,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if float(self.pre_kp) == 0.0 and float(self.cubic_stiffness) == 0.0:
            raise ValueError("The nonlinear robot needs a positive restoring term.")
        if (
            float(self.pre_kd) == 0.0
            and float(self.quadratic_drag) == 0.0
            and float(self.coulomb_friction) == 0.0
        ):
            raise ValueError("The nonlinear robot needs a positive dissipative term.")
        if int(substeps) < 1:
            raise ValueError(
                f"physics_substeps must be at least one, got {self.physics_substeps}."
            )

        # The continuous energy argument is not enough if a user makes the
        # semi-implicit step arbitrarily stiff.  Linearizing at the origin gives
        # damping kd + coulomb/friction_velocity and stiffness kp.  The Jury
        # condition below keeps that discrete 2x2 position/velocity map Schur
        # stable (or inside its nonlinear-damping boundary cases).
        substep_dt = float(self.dt) / int(substeps)
        local_damping = float(self.pre_kd) + (
            float(self.coulomb_friction) / float(self.friction_velocity)
        )
        stability_lhs = (
            2.0 * substep_dt * local_damping
            + substep_dt * substep_dt * float(self.pre_kp)
        )
        stability_rhs = 4.0 * float(self.mass)
        if stability_lhs >= stability_rhs:
            raise ValueError(
                "Nonlinear robot parameters make the discrete pre-stabilizer "
                "locally unstable: require "
                "2*(dt/substeps)*(pre_kd + coulomb_friction/friction_velocity) "
                "+ (dt/substeps)^2*pre_kp < 4*mass. Increase mass or physics "
                "substeps, or reduce dt/pre_kp/pre_kd/friction. "
                f"Got {stability_lhs:g} >= {stability_rhs:g}."
            )


def nonlinear_robot_acceleration(
    config: NonlinearRobotConfig,
    position: torch.Tensor,
    velocity: torch.Tensor,
    control: torch.Tensor,
) -> torch.Tensor:
    """Continuous-time acceleration of the nonlinear pre-stabilized robot.

    The unforced conservative part is a radial Duffing oscillator.  Every
    friction term does non-positive work, while the state-dependent gyroscopic
    term is perpendicular to velocity and therefore energy-neutral.  Control
    passes through a smooth dead-zone, a circular acceleration envelope,
    speed-related authority loss, and cross-speed-dependent traction loss.
    """
    position_sq = position.square().sum(dim=-1, keepdim=True)
    velocity_sq = velocity.square().sum(dim=-1, keepdim=True)
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)

    restoring_force = (
        -float(config.pre_kp) * position
        -float(config.cubic_stiffness) * position_sq * position
    )
    friction_force = (
        -float(config.pre_kd) * velocity
        -float(config.quadratic_drag) * speed * velocity
        -float(config.coulomb_friction)
        * torch.tanh(velocity / float(config.friction_velocity))
    )

    deadzone = float(config.actuator_deadzone)
    if deadzone > 0.0:
        # Smooth counterpart of sign(u) * max(|u| - deadzone, 0).  It is odd,
        # differentiable, and exactly zero at the origin.
        command = control - deadzone * torch.tanh(control / deadzone)
    else:
        command = control

    # A radial tanh envelope limits the norm of the total command instead of
    # clipping axes independently.  The series branch preserves a unit small-
    # signal gain when the optional dead-zone is disabled.
    command_norm = torch.linalg.vector_norm(command, dim=-1, keepdim=True)
    saturation_ratio = command_norm / float(config.actuator_limit)
    regular_scale = torch.tanh(saturation_ratio) / saturation_ratio.clamp_min(1e-6)
    saturation_scale = torch.where(
        saturation_ratio < 1e-4,
        1.0 - saturation_ratio.square() / 3.0,
        regular_scale,
    )

    # At high cross-axis speed a wheeled/ground robot cannot realize as much
    # force without slipping.  Both factors remain positive and bounded by one.
    slip = float(config.lateral_slip)
    traction = torch.cat(
        (
            1.0 / (1.0 + slip * velocity[..., 1:2].square()),
            1.0 / (1.0 + slip * velocity[..., 0:1].square()),
        ),
        dim=-1,
    )
    actuator_force = (
        command * saturation_scale * traction
        / (1.0 + float(config.speed_loss) * velocity_sq)
    )

    rotated_velocity = torch.stack(
        (-velocity[..., 1], velocity[..., 0]), dim=-1
    )
    gyro_coefficient = float(config.gyro_gain) * torch.tanh(
        float(config.gyro_position_scale)
        * position[..., 0:1]
        * position[..., 1:2]
    )
    return (
        (restoring_force + friction_force + actuator_force) / float(config.mass)
        + gyro_coefficient * rotated_velocity
    )


def integrate_nonlinear_robot(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    config: NonlinearRobotConfig,
) -> torch.Tensor:
    """One semi-implicit nonlinear robot step with optional physics substeps."""
    state = as_bt(x)
    control = as_bt(u)
    position = state[..., :2]
    velocity = state[..., 2:]
    step = float(config.dt) / int(config.physics_substeps)
    for _ in range(int(config.physics_substeps)):
        acceleration = nonlinear_robot_acceleration(
            config, position, velocity, control
        )
        velocity = velocity + step * acceleration
        position = position + step * velocity
    return torch.cat((position, velocity), dim=-1)


def nonlinear_robot_energy(
    config: NonlinearRobotConfig,
    x: torch.Tensor,
) -> torch.Tensor:
    """Lyapunov energy of the unforced nonlinear robot subsystem."""
    state = as_bt(x)
    position = state[..., :2]
    velocity = state[..., 2:]
    position_sq = position.square().sum(dim=-1)
    return (
        0.5 * float(config.mass) * velocity.square().sum(dim=-1)
        +0.5 * float(config.pre_kp) * position_sq
        +0.25 * float(config.cubic_stiffness) * position_sq.square()
    )


class NonlinearRobotNominal:
    """Strongly nonlinear nominal robot, pre-stabilized at the origin."""

    def __init__(self, config: NonlinearRobotConfig | None = None) -> None:
        self.config = config or NonlinearRobotConfig()
        self.dt = self.config.dt
        self.pre_kp = self.config.pre_kp
        self.pre_kd = self.config.pre_kd

    def nominal_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None,
    ) -> torch.Tensor:
        del t
        return integrate_nonlinear_robot(x, u, config=self.config)


class NonlinearRobotTrue:
    """True wrapper exactly matched to :class:`NonlinearRobotNominal`."""

    def __init__(self, config: NonlinearRobotConfig | None = None) -> None:
        self.config = config or NonlinearRobotConfig()
        self.dt = self.config.dt
        self.pre_kp = self.config.pre_kp
        self.pre_kd = self.config.pre_kd

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None,
    ) -> torch.Tensor:
        del t
        return integrate_nonlinear_robot(x, u, config=self.config)


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
