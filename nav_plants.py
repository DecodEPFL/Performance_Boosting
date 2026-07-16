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

The module also provides a seven-state planar dynamic-bicycle robot.  Its state
is ``(p_x, p_y, yaw, v_x, v_y, yaw_rate, steering_angle)`` and its input is
``(drive_force, steering_command)``.  Unlike an omnidirectional rigid body, it
has no independently commanded lateral force or yaw torque: lateral motion and
yaw arise from front/rear tire forces.  The model includes tire slip angles,
friction-circle saturation, longitudinal load transfer, steering lag/rate
limits, rolling/aerodynamic resistance, and a hybrid parking pre-stabilizer.
Nominal and true wrappers share the exact same transition function so PB
disturbance reconstruction continues to contain only externally injected
process noise.
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
    """Physical parameters for the nonlinear dynamic-bicycle robot.

    World-frame translational velocity is retained in the public state layout
    so the experiment can share plotting, context, and loss helpers with the
    legacy model.  ``steering_angle`` is an explicit seventh state; this keeps
    steering lag/rate limits deterministic and visible to both nominal and
    real plants instead of hiding actuator memory inside a Python object.
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
    wheelbase: float = 0.12
    cg_to_front: float = 0.062
    cg_height: float = 0.028
    gravity: float = 9.81
    cubic_stiffness: float = 0.10
    yaw_cubic_stiffness: float = 0.06
    parking_lateral_gain: float = 1.35
    parking_lateral_cubic: float = 0.18
    longitudinal_drag: float = 0.15
    lateral_drag: float = 0.18
    quadratic_drag: float = 0.28
    lateral_quadratic_drag: float = 0.24
    coulomb_friction: float = 0.05
    friction_velocity: float = 0.12
    angular_drag: float = 0.05
    angular_quadratic_drag: float = 0.025
    angular_coulomb_friction: float = 0.012
    angular_friction_velocity: float = 0.18
    actuator_limit: float = 2.5
    actuator_deadzone: float = 0.06
    steering_limit: float = 0.60
    steering_deadzone: float = 0.012
    steering_time_constant: float = 0.12
    steering_rate_limit: float = 2.5
    cornering_stiffness_front: float = 4.0
    cornering_stiffness_rear: float = 4.5
    tire_friction: float = 0.90
    slip_speed_floor: float = 0.12
    low_speed_steering_grip: float = 0.18
    speed_loss: float = 0.12
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
            "wheelbase": self.wheelbase,
            "cg_to_front": self.cg_to_front,
            "gravity": self.gravity,
            "friction_velocity": self.friction_velocity,
            "angular_friction_velocity": self.angular_friction_velocity,
            "actuator_limit": self.actuator_limit,
            "steering_limit": self.steering_limit,
            "steering_time_constant": self.steering_time_constant,
            "steering_rate_limit": self.steering_rate_limit,
            "cornering_stiffness_front": self.cornering_stiffness_front,
            "cornering_stiffness_rear": self.cornering_stiffness_rear,
            "tire_friction": self.tire_friction,
            "slip_speed_floor": self.slip_speed_floor,
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
            "parking_lateral_gain": self.parking_lateral_gain,
            "parking_lateral_cubic": self.parking_lateral_cubic,
            "cg_height": self.cg_height,
            "longitudinal_drag": self.longitudinal_drag,
            "lateral_drag": self.lateral_drag,
            "quadratic_drag": self.quadratic_drag,
            "lateral_quadratic_drag": self.lateral_quadratic_drag,
            "coulomb_friction": self.coulomb_friction,
            "angular_drag": self.angular_drag,
            "angular_quadratic_drag": self.angular_quadratic_drag,
            "angular_coulomb_friction": self.angular_coulomb_friction,
            "actuator_deadzone": self.actuator_deadzone,
            "steering_deadzone": self.steering_deadzone,
            "speed_loss": self.speed_loss,
            "low_speed_steering_grip": self.low_speed_steering_grip,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if float(self.pre_kp) == 0.0 and float(self.cubic_stiffness) == 0.0:
            raise ValueError("The nonlinear robot needs a positive restoring term.")
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
        if float(self.actuator_deadzone) >= float(self.actuator_limit):
            raise ValueError("actuator_deadzone must be smaller than actuator_limit.")
        if float(self.steering_deadzone) >= float(self.steering_limit):
            raise ValueError("steering_deadzone must be smaller than steering_limit.")
        if float(self.cg_to_front) >= float(self.wheelbase):
            raise ValueError("cg_to_front must lie strictly inside the wheelbase.")
        if float(self.wheelbase) > float(self.body_length):
            raise ValueError("wheelbase cannot exceed body_length.")
        if float(self.low_speed_steering_grip) > 1.0:
            raise ValueError("low_speed_steering_grip cannot exceed one.")
        if int(substeps) < 1:
            raise ValueError(
                f"physics_substeps must be at least one, got {self.physics_substeps}."
            )

        # Conservative local step-size checks for the stiff longitudinal,
        # lateral-tire, yaw, and steering modes.  They reject obviously unsafe
        # parameter combinations before an autograd rollout can explode.
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
        lateral_damping = (
            float(self.cornering_stiffness_front)
            + float(self.cornering_stiffness_rear)
        ) / float(self.slip_speed_floor) + float(self.lateral_drag)
        if 2.0 * substep_dt * lateral_damping >= 4.0 * float(self.mass):
            raise ValueError(
                "Rigid-body parameters make the discrete lateral tire mode "
                "locally unstable. Increase mass/physics substeps or reduce "
                "cornering stiffness."
            )
        rear_distance = float(self.wheelbase) - float(self.cg_to_front)
        yaw_damping = (
            float(self.cornering_stiffness_front)
            * float(self.cg_to_front) ** 2
            + float(self.cornering_stiffness_rear) * rear_distance**2
        ) / float(self.slip_speed_floor)
        yaw_damping += (
            float(self.angular_drag)
            + float(self.angular_coulomb_friction)
            / float(self.angular_friction_velocity)
        )
        if 2.0 * substep_dt * yaw_damping >= 4.0 * float(self.inertia):
            raise ValueError(
                "Rigid-body parameters make the discrete tire/yaw mode locally "
                "unstable. Increase inertia/physics substeps or reduce cornering "
                "stiffness."
            )
        if substep_dt > 2.0 * float(self.steering_time_constant):
            raise ValueError(
                "The steering actuator is too fast for the physics substep. "
                "Increase steering_time_constant or physics_substeps."
            )


def nonlinear_robot_acceleration(
    config: NonlinearRobotConfig,
    state: torch.Tensor,
    control: torch.Tensor,
) -> torch.Tensor:
    """Return ``(a_x, a_y, yaw_acceleration, steering_rate)``.

    The dynamic bicycle is fully differentiable but not omnidirectionally
    actuated.  Drive acts at the rear contact patch; front/rear lateral tire
    forces arise from slip angles and share the rear friction budget with the
    drive force.  A small low-speed tire-scrub term regularizes parking at the
    origin, where the ideal no-slip bicycle equations are singular.
    """
    state = as_bt(state)
    control = as_bt(control)
    if state.shape[-1] != 7:
        raise ValueError(
            "Nonlinear bicycle state must have seven channels "
            "(x, y, yaw, vx, vy, yaw_rate, steering_angle), got "
            f"shape {tuple(state.shape)}."
        )
    if control.shape[-1] != 2:
        raise ValueError(
            "Nonlinear bicycle control must have two channels "
            "(drive_force, steering_command), got "
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
    steering_angle = state[..., 6:7]

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
    # The parking pre-stabilizer can only request drive and steering; it never
    # injects the forbidden independently controlled lateral force.  Position
    # error is expressed in the body frame so positive longitudinal error asks
    # the car to reverse toward the origin.
    position_sq = position.square().sum(dim=-1, keepdim=True)
    position_norm = (
        torch.sqrt(position_sq + 1e-12) - 1e-6
    ).clamp_min(0.0)
    target_heading = torch.atan2(-position[..., 1:2], -position[..., 0:1])
    target_heading_error = torch.atan2(
        torch.sin(target_heading - heading),
        torch.cos(target_heading - heading),
    )
    # Hybrid forward/reverse selection is intentional.  A continuous,
    # time-invariant feedback cannot asymptotically park a nonholonomic car at
    # a pose (Brockett obstruction); choosing the nearer travel orientation is
    # the standard practical resolution and avoids needless three-point turns.
    travel_direction = torch.where(
        torch.cos(target_heading_error) >= 0.0,
        torch.ones_like(target_heading_error),
        -torch.ones_like(target_heading_error),
    )
    motion_heading = heading + torch.where(
        travel_direction < 0.0,
        torch.full_like(heading, math.pi),
        torch.zeros_like(heading),
    )
    line_heading_error = torch.atan2(
        torch.sin(target_heading - motion_heading),
        torch.cos(target_heading - motion_heading),
    )
    pre_stabilizing_drive = (
        travel_direction
        * (
            float(config.pre_kp) * position_norm
            + float(config.cubic_stiffness) * position_norm.pow(3)
        )
        - float(config.pre_kd) * velocity_longitudinal
    )
    heading_shape = (
        torch.sin(heading)
        + float(config.yaw_cubic_stiffness)
        * (1.0 - torch.cos(heading))
        * torch.sin(heading)
    )
    line_steering = travel_direction * (
        float(config.parking_lateral_gain) * line_heading_error
        + float(config.parking_lateral_cubic) * line_heading_error.pow(3)
    ) - float(config.yaw_pre_kd) * yaw_rate
    pose_steering = (
        -float(config.yaw_pre_kp) * heading_shape
        - float(config.yaw_pre_kd) * yaw_rate
    )
    pose_blend_radius = max(2.0 * float(config.body_length), 1e-6)
    pose_blend = torch.exp(
        -0.5 * position_sq / (pose_blend_radius**2)
    )
    pre_stabilizing_steering = (
        (1.0 - pose_blend) * line_steering + pose_blend * pose_steering
    )

    drive_correction = control[..., 0:1]
    steering_correction = control[..., 1:2]
    force_deadzone = float(config.actuator_deadzone)
    if force_deadzone > 0.0:
        drive_correction = drive_correction - force_deadzone * torch.tanh(
            drive_correction / force_deadzone
        )
    steering_deadzone = float(config.steering_deadzone)
    if steering_deadzone > 0.0:
        steering_correction = steering_correction - steering_deadzone * torch.tanh(
            steering_correction / steering_deadzone
        )

    translational_speed_sq = velocity_world.square().sum(dim=-1, keepdim=True)
    edge_speed_sq = (
        0.5 * float(config.body_length) * yaw_rate
    ).square()
    speed_authority = 1.0 / (
        1.0
        + float(config.speed_loss) * (translational_speed_sq + edge_speed_sq)
    )
    drive_request = (
        pre_stabilizing_drive + drive_correction
    ) * speed_authority
    steering_request = pre_stabilizing_steering + steering_correction
    steering_target = float(config.steering_limit) * torch.tanh(
        steering_request / float(config.steering_limit)
    )
    steering_rate = float(config.steering_rate_limit) * torch.tanh(
        (steering_target - steering_angle)
        / (
            float(config.steering_time_constant)
            * float(config.steering_rate_limit)
        )
    )

    # Quasi-static longitudinal load transfer changes the axle friction
    # budgets.  The proxy is limited before it is used, preventing an arbitrary
    # learned command from producing negative normal loads.
    limited_drive_proxy = float(config.actuator_limit) * torch.tanh(
        drive_request / float(config.actuator_limit)
    )
    longitudinal_accel_proxy = limited_drive_proxy / float(config.mass)
    rear_distance = float(config.wheelbase) - float(config.cg_to_front)
    static_front_load = (
        float(config.mass) * float(config.gravity)
        * rear_distance / float(config.wheelbase)
    )
    static_rear_load = (
        float(config.mass) * float(config.gravity)
        * float(config.cg_to_front) / float(config.wheelbase)
    )
    load_delta = (
        float(config.mass) * float(config.cg_height)
        / float(config.wheelbase) * longitudinal_accel_proxy
    )
    minimum_axle_load = 0.05 * float(config.mass) * float(config.gravity)
    front_normal_load = (static_front_load - load_delta).clamp_min(
        minimum_axle_load
    )
    rear_normal_load = (static_rear_load + load_delta).clamp_min(
        minimum_axle_load
    )
    front_friction_limit = float(config.tire_friction) * front_normal_load
    rear_friction_limit = float(config.tire_friction) * rear_normal_load
    drive_limit = torch.minimum(
        torch.full_like(rear_friction_limit, float(config.actuator_limit)),
        rear_friction_limit,
    )
    drive_force = drive_limit * torch.tanh(
        drive_request / drive_limit.clamp_min(1e-7)
    )

    # Dynamic-bicycle slip angles.  Direction-dependent steering recovers the
    # correct sign while reversing.  The small zero-speed scrub fraction is a
    # finite-compliance parking regularization, not an independent lateral
    # actuator.
    speed_floor = float(config.slip_speed_floor)
    slip_denominator = torch.sqrt(
        velocity_longitudinal.square() + speed_floor**2
    )
    tire_travel_direction = torch.tanh(velocity_longitudinal / speed_floor)
    steering_direction = (
        tire_travel_direction
        + float(config.low_speed_steering_grip)
        * (1.0 - tire_travel_direction.square())
    )
    effective_steering = steering_direction * steering_angle
    front_slip_angle = torch.atan(
        (velocity_lateral + float(config.cg_to_front) * yaw_rate)
        / slip_denominator
    ) - effective_steering
    rear_slip_angle = torch.atan(
        (velocity_lateral - rear_distance * yaw_rate)
        / slip_denominator
    )
    front_lateral_force = -front_friction_limit * torch.tanh(
        float(config.cornering_stiffness_front) * front_slip_angle
        / front_friction_limit.clamp_min(1e-7)
    )
    rear_lateral_budget = torch.sqrt(
        (rear_friction_limit.square() - drive_force.square()).clamp_min(1e-8)
    )
    rear_lateral_force = -rear_lateral_budget * torch.tanh(
        float(config.cornering_stiffness_rear) * rear_slip_angle
        / rear_lateral_budget
    )

    longitudinal_resistance = (
        -float(config.longitudinal_drag) * velocity_longitudinal
        -float(config.quadratic_drag)
        * velocity_longitudinal.abs() * velocity_longitudinal
        -float(config.coulomb_friction)
        * torch.tanh(velocity_longitudinal / float(config.friction_velocity))
    )
    lateral_resistance = (
        -float(config.lateral_drag) * velocity_lateral
        -float(config.lateral_quadratic_drag)
        * velocity_lateral.abs() * velocity_lateral
    )
    force_longitudinal = (
        drive_force
        - front_lateral_force * torch.sin(steering_angle)
        + longitudinal_resistance
    )
    force_lateral = (
        front_lateral_force * torch.cos(steering_angle)
        + rear_lateral_force
        + lateral_resistance
    )
    total_force_body = torch.cat(
        (force_longitudinal, force_lateral), dim=-1
    )
    total_force_world = torch.cat(
        (
            cos_heading * total_force_body[..., 0:1]
            - sin_heading * total_force_body[..., 1:2],
            sin_heading * total_force_body[..., 0:1]
            + cos_heading * total_force_body[..., 1:2],
        ), dim=-1,
    )

    passive_yaw_torque = (
        -float(config.angular_drag) * yaw_rate
        - float(config.angular_quadratic_drag) * yaw_rate.abs() * yaw_rate
        - float(config.angular_coulomb_friction)
        * torch.tanh(yaw_rate / float(config.angular_friction_velocity))
    )
    total_torque = (
        float(config.cg_to_front)
        * front_lateral_force * torch.cos(steering_angle)
        - rear_distance * rear_lateral_force
        + passive_yaw_torque
    )
    return torch.cat(
        (
            total_force_world / float(config.mass),
            total_torque / float(config.inertia),
            steering_rate,
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
    if state.shape[-1] != 7:
        raise ValueError(
            "Nonlinear bicycle state must have seven channels, got "
            f"shape {tuple(state.shape)}."
        )
    if control.shape[-1] != 2:
        raise ValueError(
            "Nonlinear bicycle control must have two channels, got "
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
    steering_angle = state[..., 6:7]
    step = float(config.dt) / int(config.physics_substeps)
    for _ in range(int(config.physics_substeps)):
        substep_state = torch.cat(
            (position, heading, velocity, yaw_rate, steering_angle), dim=-1
        )
        acceleration = nonlinear_robot_acceleration(
            config, substep_state, control
        )
        velocity = velocity + step * acceleration[..., 0:2]
        yaw_rate = yaw_rate + step * acceleration[..., 2:3]
        steering_angle = steering_angle + step * acceleration[..., 3:4]
        steering_angle = steering_angle.clamp(
            -float(config.steering_limit), float(config.steering_limit)
        )
        position = position + step * velocity
        # Keep yaw unwrapped.  This avoids a discontinuous modulo operation in
        # the autograd graph; rendering may wrap it for display if desired.
        heading = heading + step * yaw_rate
    return torch.cat(
        (position, heading, velocity, yaw_rate, steering_angle), dim=-1
    )


def nonlinear_robot_energy(
    config: NonlinearRobotConfig,
    x: torch.Tensor,
) -> torch.Tensor:
    """Positive diagnostic storage for the pre-stabilized bicycle state."""
    state = as_bt(x)
    if state.shape[-1] != 7:
        raise ValueError(
            "Nonlinear bicycle state must have seven channels, got "
            f"shape {tuple(state.shape)}."
        )
    position = state[..., 0:2]
    heading = state[..., 2]
    velocity = state[..., 3:5]
    yaw_rate = state[..., 5]
    steering_angle = state[..., 6]
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
        + 0.5 * steering_angle.square()
    )


class NonlinearRobotNominal:
    """Seven-state dynamic-bicycle nominal robot."""

    state_dim = 7
    control_dim = 2

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
    """True bicycle exactly matched to :class:`NonlinearRobotNominal`."""

    state_dim = 7
    control_dim = 2

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
