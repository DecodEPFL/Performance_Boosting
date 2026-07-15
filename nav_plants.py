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

The module also provides a six-state planar rigid-body robot.  Its state is
``(p_x, p_y, yaw, v_x, v_y, yaw_rate)`` and its input is a body-frame wrench
``(force_longitudinal, force_lateral, torque)``.  It includes translational and
rotational inertia, an oriented anisotropic resistance model, combined-slip
traction loss, coupled force/torque saturation, actuator dead-zones, and a
force application offset.  Conservative restoring terms and strictly
dissipative passive terms pre-stabilize the unforced origin.  Nominal and true
wrappers share the exact same transition function so PB disturbance
reconstruction continues to contain only externally injected process noise.
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
    """Physical parameters for the nonlinear planar rigid body.

    The state is ``(p_x, p_y, yaw, v_x, v_y, yaw_rate)``.  Translational
    velocity is expressed in the inertial/world frame; the three control
    channels are longitudinal force, lateral force, and yaw torque expressed
    in the body frame.  Actuators are memoryless: no hidden lag or previous
    command is required by the transition.

    The physical footprint is the oriented rectangle defined by
    ``body_length`` and ``body_width``.  The defaults expose the nonlinear
    effects while remaining well behaved in the gate experiment's normal
    operating envelope.
    """

    dt: float = 0.05
    pre_kp: float = 0.32
    pre_kd: float = 0.35
    yaw_pre_kp: float = 0.22
    yaw_pre_kd: float = 0.16
    mass: float = 1.0
    inertia: float = 0.012
    body_length: float = 0.16
    body_width: float = 0.10
    cubic_stiffness: float = 0.10
    yaw_cubic_stiffness: float = 0.06
    longitudinal_drag: float = 0.15
    lateral_drag: float = 0.85
    quadratic_drag: float = 0.28
    lateral_quadratic_drag: float = 0.55
    coulomb_friction: float = 0.05
    friction_velocity: float = 0.12
    angular_drag: float = 0.05
    angular_quadratic_drag: float = 0.025
    angular_coulomb_friction: float = 0.012
    angular_friction_velocity: float = 0.18
    actuator_limit: float = 2.5
    lateral_force_limit: float = 1.7
    torque_limit: float = 0.18
    actuator_deadzone: float = 0.06
    torque_deadzone: float = 0.008
    speed_loss: float = 0.12
    lateral_slip: float = 0.35
    traction_velocity: float = 0.35
    load_transfer: float = 0.12
    tire_saturation: float = 0.25
    actuator_offset_x: float = 0.025
    physics_substeps: int = 4

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
            "inertia": self.inertia,
            "body_length": self.body_length,
            "body_width": self.body_width,
            "friction_velocity": self.friction_velocity,
            "angular_friction_velocity": self.angular_friction_velocity,
            "actuator_limit": self.actuator_limit,
            "lateral_force_limit": self.lateral_force_limit,
            "torque_limit": self.torque_limit,
            "traction_velocity": self.traction_velocity,
        }
        for name, value in strictly_positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be strictly positive, got {value}.")

        nonnegative = {
            "pre_kp": self.pre_kp,
            "pre_kd": self.pre_kd,
            "yaw_pre_kp": self.yaw_pre_kp,
            "yaw_pre_kd": self.yaw_pre_kd,
            "cubic_stiffness": self.cubic_stiffness,
            "yaw_cubic_stiffness": self.yaw_cubic_stiffness,
            "longitudinal_drag": self.longitudinal_drag,
            "lateral_drag": self.lateral_drag,
            "quadratic_drag": self.quadratic_drag,
            "lateral_quadratic_drag": self.lateral_quadratic_drag,
            "coulomb_friction": self.coulomb_friction,
            "angular_drag": self.angular_drag,
            "angular_quadratic_drag": self.angular_quadratic_drag,
            "angular_coulomb_friction": self.angular_coulomb_friction,
            "actuator_deadzone": self.actuator_deadzone,
            "torque_deadzone": self.torque_deadzone,
            "speed_loss": self.speed_loss,
            "lateral_slip": self.lateral_slip,
            "load_transfer": self.load_transfer,
            "tire_saturation": self.tire_saturation,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if float(self.pre_kp) == 0.0 and float(self.cubic_stiffness) == 0.0:
            raise ValueError("The nonlinear robot needs a positive restoring term.")
        if (
            float(self.yaw_pre_kp) == 0.0
            and float(self.yaw_cubic_stiffness) == 0.0
        ):
            raise ValueError("The nonlinear robot needs a positive yaw restoring term.")
        if (
            float(self.pre_kd) + float(self.longitudinal_drag) == 0.0
            and float(self.quadratic_drag) == 0.0
            and float(self.coulomb_friction) == 0.0
        ):
            raise ValueError(
                "The nonlinear robot needs longitudinal dissipation."
            )
        if (
            float(self.pre_kd) + float(self.lateral_drag) == 0.0
            and float(self.lateral_quadratic_drag) == 0.0
            and float(self.coulomb_friction) == 0.0
        ):
            raise ValueError("The nonlinear robot needs lateral dissipation.")
        if (
            float(self.yaw_pre_kd) + float(self.angular_drag) == 0.0
            and float(self.angular_quadratic_drag) == 0.0
            and float(self.angular_coulomb_friction) == 0.0
        ):
            raise ValueError("The nonlinear robot needs angular dissipation.")
        if float(self.actuator_deadzone) >= min(
            float(self.actuator_limit), float(self.lateral_force_limit)
        ):
            raise ValueError(
                "actuator_deadzone must be smaller than both force limits."
            )
        if float(self.torque_deadzone) >= float(self.torque_limit):
            raise ValueError("torque_deadzone must be smaller than torque_limit.")
        if int(substeps) < 1:
            raise ValueError(
                f"physics_substeps must be at least one, got {self.physics_substeps}."
            )

        # The continuous energy argument is not sufficient if a user makes a
        # semi-implicit substep arbitrarily stiff.  The Jury conditions below
        # keep the linearized translation and yaw maps Schur stable.  The
        # conservative cubic terms have zero derivative at the origin.
        substep_dt = float(self.dt) / int(substeps)
        coulomb_slope = (
            float(self.coulomb_friction) / float(self.friction_velocity)
        )
        local_damping = float(self.pre_kd) + max(
            float(self.longitudinal_drag), float(self.lateral_drag)
        ) + coulomb_slope
        translation_lhs = (
            2.0 * substep_dt * local_damping
            + substep_dt * substep_dt * float(self.pre_kp)
        )
        translation_rhs = 4.0 * float(self.mass)
        if translation_lhs >= translation_rhs:
            raise ValueError(
                "Rigid-body parameters make the discrete translation "
                "pre-stabilizer locally unstable. Increase mass or physics "
                "substeps, or reduce dt/stiffness/damping. "
                f"Got {translation_lhs:g} >= {translation_rhs:g}."
            )
        angular_coulomb_slope = float(self.angular_coulomb_friction) / float(
            self.angular_friction_velocity
        )
        angular_damping = (
            float(self.yaw_pre_kd)
            + float(self.angular_drag)
            + angular_coulomb_slope
        )
        rotation_lhs = (
            2.0 * substep_dt * angular_damping
            + substep_dt * substep_dt * float(self.yaw_pre_kp)
        )
        rotation_rhs = 4.0 * float(self.inertia)
        if rotation_lhs >= rotation_rhs:
            raise ValueError(
                "Rigid-body parameters make the discrete yaw pre-stabilizer "
                "locally unstable. Increase inertia or physics substeps, or "
                "reduce dt/yaw stiffness/angular damping. "
                f"Got {rotation_lhs:g} >= {rotation_rhs:g}."
            )


def nonlinear_robot_acceleration(
    config: NonlinearRobotConfig,
    state: torch.Tensor,
    control: torch.Tensor,
) -> torch.Tensor:
    """Return ``(a_x, a_y, yaw_acceleration)`` for one rigid-body state.

    The state and control accept ``(B, D)`` or ``(B, T, D)`` tensors and must
    have final dimensions six and three respectively.  Passive body-frame
    resistance always does non-positive work.  The actuator model applies a
    smooth component dead-zone, a coupled ellipsoidal wrench envelope, and
    orientation/speed/slip-dependent grip.  It is instantaneous and therefore
    adds no unobserved actuator state.
    """
    state = as_bt(state)
    control = as_bt(control)
    if state.shape[-1] != 6:
        raise ValueError(
            "Nonlinear rigid-body state must have six channels "
            "(x, y, yaw, vx, vy, yaw_rate), got "
            f"shape {tuple(state.shape)}."
        )
    if control.shape[-1] != 3:
        raise ValueError(
            "Nonlinear rigid-body control must have three channels "
            "(longitudinal_force, lateral_force, torque), got "
            f"shape {tuple(control.shape)}."
        )
    if state.shape[:-1] != control.shape[:-1]:
        raise ValueError(
            "Rigid-body state and control batch/time dimensions must match, "
            f"got {tuple(state.shape)} and {tuple(control.shape)}."
        )

    position = state[..., 0:2]
    heading = state[..., 2:3]
    velocity_world = state[..., 3:5]
    yaw_rate = state[..., 5:6]

    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    velocity_longitudinal = (
        cos_heading * velocity_world[..., 0:1]
        + sin_heading * velocity_world[..., 1:2]
    )
    velocity_lateral = (
        -sin_heading * velocity_world[..., 0:1]
        + cos_heading * velocity_world[..., 1:2]
    )
    velocity_body = torch.cat(
        (velocity_longitudinal, velocity_lateral), dim=-1
    )

    # World-frame radial Duffing potential pre-stabilizes translation.  Body-
    # frame resistance produces orientation-dependent rolling/lateral losses.
    position_sq = position.square().sum(dim=-1, keepdim=True)
    restoring_world = (
        -float(config.pre_kp) * position
        - float(config.cubic_stiffness) * position_sq * position
    )
    linear_drag = torch.cat(
        (
            torch.full_like(
                velocity_longitudinal,
                float(config.pre_kd) + float(config.longitudinal_drag),
            ),
            torch.full_like(
                velocity_lateral,
                float(config.pre_kd) + float(config.lateral_drag),
            ),
        ),
        dim=-1,
    )
    quadratic_drag = torch.cat(
        (
            torch.full_like(
                velocity_longitudinal, float(config.quadratic_drag)
            ),
            torch.full_like(
                velocity_lateral, float(config.lateral_quadratic_drag)
            ),
        ),
        dim=-1,
    )
    passive_force_body = (
        -linear_drag * velocity_body
        - quadratic_drag * velocity_body.abs() * velocity_body
        - float(config.coulomb_friction)
        * torch.tanh(velocity_body / float(config.friction_velocity))
    )

    force_command = control[..., 0:2]
    torque_command = control[..., 2:3]
    force_deadzone = float(config.actuator_deadzone)
    if force_deadzone > 0.0:
        force_command = force_command - force_deadzone * torch.tanh(
            force_command / force_deadzone
        )
    torque_deadzone = float(config.torque_deadzone)
    if torque_deadzone > 0.0:
        torque_command = torque_command - torque_deadzone * torch.tanh(
            torque_command / torque_deadzone
        )

    # Saturate the complete wrench, not each actuator independently.  This is
    # a smooth friction-ellipse analogue: requesting large force leaves less
    # authority for simultaneous torque and vice versa.
    normalized_wrench = torch.cat(
        (
            force_command[..., 0:1] / float(config.actuator_limit),
            force_command[..., 1:2] / float(config.lateral_force_limit),
            torque_command / float(config.torque_limit),
        ),
        dim=-1,
    )
    saturation_ratio = torch.linalg.vector_norm(
        normalized_wrench, dim=-1, keepdim=True
    )
    regular_scale = torch.tanh(saturation_ratio) / saturation_ratio.clamp_min(
        1e-7
    )
    saturation_scale = torch.where(
        saturation_ratio < 1e-4,
        1.0 - saturation_ratio.square() / 3.0,
        regular_scale,
    )
    force_command = force_command * saturation_scale
    torque_command = torque_command * saturation_scale

    translational_speed_sq = velocity_world.square().sum(dim=-1, keepdim=True)
    edge_speed_sq = (
        0.5 * float(config.body_length) * yaw_rate
    ).square()
    speed_authority = 1.0 / (
        1.0
        + float(config.speed_loss) * (translational_speed_sq + edge_speed_sq)
    )
    smooth_abs_yaw_rate = torch.sqrt(yaw_rate.square() + 1e-12)
    longitudinal_grip = 1.0 / (
        1.0
        + float(config.lateral_slip) * velocity_lateral.square()
        + float(config.load_transfer)
        * smooth_abs_yaw_rate
        * velocity_lateral.abs()
    )
    lateral_grip = 1.0 / (
        1.0
        + float(config.lateral_slip) * velocity_longitudinal.square()
        + float(config.tire_saturation)
        * (velocity_lateral / float(config.traction_velocity)).square()
        + float(config.load_transfer)
        * smooth_abs_yaw_rate
        * velocity_longitudinal.abs()
    )
    grip = torch.cat((longitudinal_grip, lateral_grip), dim=-1)
    actuator_force_body = force_command * speed_authority * grip
    actuator_torque = torque_command / (
        1.0 + float(config.speed_loss) * yaw_rate.square()
    )

    total_force_body = passive_force_body + actuator_force_body
    total_force_world = torch.cat(
        (
            cos_heading * total_force_body[..., 0:1]
            - sin_heading * total_force_body[..., 1:2],
            sin_heading * total_force_body[..., 0:1]
            + cos_heading * total_force_body[..., 1:2],
        ),
        dim=-1,
    ) + restoring_world

    yaw_shape = torch.sin(heading)
    yaw_restoring_torque = (
        -float(config.yaw_pre_kp) * yaw_shape
        - float(config.yaw_cubic_stiffness)
        * (1.0 - torch.cos(heading))
        * yaw_shape
    )
    passive_yaw_torque = (
        -float(config.yaw_pre_kd) * yaw_rate
        - float(config.angular_drag) * yaw_rate
        - float(config.angular_quadratic_drag) * yaw_rate.abs() * yaw_rate
        - float(config.angular_coulomb_friction)
        * torch.tanh(yaw_rate / float(config.angular_friction_velocity))
    )
    # A lateral force applied away from the centre of mass creates a real yaw
    # moment, coupling translation and attitude without artificial gyroscopic
    # forces that would violate inertial-frame rigid-body mechanics.
    force_offset_torque = (
        float(config.actuator_offset_x) * actuator_force_body[..., 1:2]
    )
    total_torque = (
        yaw_restoring_torque
        + passive_yaw_torque
        + actuator_torque
        + force_offset_torque
    )
    return torch.cat(
        (
            total_force_world / float(config.mass),
            total_torque / float(config.inertia),
        ),
        dim=-1,
    )


def integrate_nonlinear_robot(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    config: NonlinearRobotConfig,
) -> torch.Tensor:
    """One stable semi-implicit rigid-body step with physics substeps."""
    state = as_bt(x)
    control = as_bt(u)
    if state.shape[-1] != 6:
        raise ValueError(
            "Nonlinear rigid-body state must have six channels, got "
            f"shape {tuple(state.shape)}."
        )
    if control.shape[-1] != 3:
        raise ValueError(
            "Nonlinear rigid-body control must have three channels, got "
            f"shape {tuple(control.shape)}."
        )
    if state.shape[:-1] != control.shape[:-1]:
        raise ValueError(
            "Rigid-body state and control batch/time dimensions must match, "
            f"got {tuple(state.shape)} and {tuple(control.shape)}."
        )
    position = state[..., 0:2]
    heading = state[..., 2:3]
    velocity = state[..., 3:5]
    yaw_rate = state[..., 5:6]
    step = float(config.dt) / int(config.physics_substeps)
    for _ in range(int(config.physics_substeps)):
        substep_state = torch.cat(
            (position, heading, velocity, yaw_rate), dim=-1
        )
        acceleration = nonlinear_robot_acceleration(
            config, substep_state, control
        )
        velocity = velocity + step * acceleration[..., 0:2]
        yaw_rate = yaw_rate + step * acceleration[..., 2:3]
        position = position + step * velocity
        # Keep yaw unwrapped.  This avoids a discontinuous modulo operation in
        # the autograd graph; rendering may wrap it for display if desired.
        heading = heading + step * yaw_rate
    return torch.cat((position, heading, velocity, yaw_rate), dim=-1)


def nonlinear_robot_energy(
    config: NonlinearRobotConfig,
    x: torch.Tensor,
) -> torch.Tensor:
    """Mechanical Lyapunov energy of the unforced rigid-body subsystem."""
    state = as_bt(x)
    if state.shape[-1] != 6:
        raise ValueError(
            "Nonlinear rigid-body state must have six channels, got "
            f"shape {tuple(state.shape)}."
        )
    position = state[..., 0:2]
    heading = state[..., 2]
    velocity = state[..., 3:5]
    yaw_rate = state[..., 5]
    position_sq = position.square().sum(dim=-1)
    yaw_potential_shape = 1.0 - torch.cos(heading)
    return (
        0.5 * float(config.mass) * velocity.square().sum(dim=-1)
        + 0.5 * float(config.inertia) * yaw_rate.square()
        + 0.5 * float(config.pre_kp) * position_sq
        + 0.25 * float(config.cubic_stiffness) * position_sq.square()
        + float(config.yaw_pre_kp) * yaw_potential_shape
        + 0.5
        * float(config.yaw_cubic_stiffness)
        * yaw_potential_shape.square()
    )


class NonlinearRobotNominal:
    """Six-state rigid-body nominal robot, pre-stabilized at the origin."""

    state_dim = 6
    control_dim = 3

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
    """True rigid body exactly matched to :class:`NonlinearRobotNominal`."""

    state_dim = 6
    control_dim = 3

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
