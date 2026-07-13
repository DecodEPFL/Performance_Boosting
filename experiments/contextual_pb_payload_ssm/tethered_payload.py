"""Hidden-state tethered-cargo simulator and causal slalom context."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pb_core import PBController, as_bt
from payload_core import (PAYLOAD_CONTEXT_FEATURES, _process_noise, build_controller,
                          resolve_context_features)


@dataclass
class SlalomBatch:
    start: torch.Tensor
    goal: torch.Tensor
    payload_start: torch.Tensor
    payload_velocity_start: torch.Tensor
    gate_centers: torch.Tensor
    payload_mass: torch.Tensor
    payload_mass_obs: torch.Tensor
    tether_length: torch.Tensor
    tether_stiffness: torch.Tensor
    tether_damping: torch.Tensor
    telemetry_noise: torch.Tensor
    process_noise: torch.Tensor
    pair_id: torch.Tensor


@dataclass
class SlalomRollout:
    x_seq: torch.Tensor
    u_seq: torch.Tensor
    w_seq: torch.Tensor
    payload_pos_seq: torch.Tensor
    payload_vel_seq: torch.Tensor
    tension_seq: torch.Tensor


def parse_float_list(raw: str, name: str) -> list[float]:
    try:
        values = [float(token.strip()) for token in str(raw).split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated float list.") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    return values


def gate_geometry(args) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(parse_float_list(args.slalom_gate_xs, "slalom_gate_xs"), np.float32)
    centers = np.asarray(parse_float_list(args.slalom_gate_centers, "slalom_gate_centers"), np.float32)
    if len(xs) != len(centers):
        raise ValueError("slalom_gate_xs and slalom_gate_centers need equal lengths.")
    if len(xs) < 2 or not np.all(np.diff(xs) < 0):
        raise ValueError("Use at least two gate x positions in strictly descending order.")
    if xs[0] >= float(args.start_x_min) or xs[-1] <= 0.0:
        raise ValueError("Every slalom gate must lie strictly between the start and origin.")
    effective = float(args.slalom_gate_half_width) - max(
        float(args.slalom_carrier_radius), float(args.slalom_payload_radius)
    ) - float(args.slalom_collision_margin)
    if effective <= 0.0:
        raise ValueError("Gate half-width must exceed body radius plus collision margin.")
    if np.max(np.abs(centers)) + float(args.slalom_gate_half_width) >= float(args.corridor_limit):
        raise ValueError("Gate openings must fit inside the configured corridor.")
    return xs, centers


def _mass_range(args, test: bool) -> tuple[float, float]:
    if test:
        return float(args.slalom_test_mass_min), float(args.slalom_test_mass_max)
    return float(args.slalom_payload_mass_min), float(args.slalom_payload_mass_max)


def _length_range(args, test: bool) -> tuple[float, float]:
    if test:
        return float(args.slalom_test_tether_length_min), float(args.slalom_test_tether_length_max)
    return float(args.slalom_tether_length_min), float(args.slalom_tether_length_max)


def sample_slalom_batch(
    args, *, batch_size: int, seed: int, paired: bool, shuffle: bool,
    test: bool = False,
) -> SlalomBatch:
    if paired and batch_size % 2:
        raise ValueError("Alias-paired slalom batches require an even batch size.")
    gate_xs, base_centers = gate_geometry(args)
    rng = np.random.default_rng(seed)
    count = batch_size // 2 if paired else batch_size
    records: list[tuple] = []
    mass_lo, mass_hi = _mass_range(args, test)
    length_lo, length_hi = _length_range(args, test)
    for pair_id in range(count):
        start = np.array([
            rng.uniform(float(args.start_x_min), float(args.start_x_max)),
            rng.uniform(-float(args.start_y_max), float(args.start_y_max)),
        ], np.float32)
        route_sign = float(rng.choice([-1.0, 1.0]))
        centers = route_sign * base_centers
        centers += rng.normal(0.0, float(args.slalom_gate_center_jitter), len(gate_xs))
        mass = float(rng.uniform(mass_lo, mass_hi))
        length = float(rng.uniform(length_lo, length_hi))
        stiffness = float(rng.uniform(float(args.slalom_tether_stiffness_min),
                                      float(args.slalom_tether_stiffness_max)))
        damping = float(rng.uniform(float(args.slalom_tether_damping_min),
                                    float(args.slalom_tether_damping_max)))
        sway_speed = float(rng.uniform(float(args.slalom_sway_speed_min),
                                       float(args.slalom_sway_speed_max)))
        sway_sign = float(rng.choice([-1.0, 1.0]))
        members = [sway_sign] if not paired else [sway_sign, -sway_sign]
        for sign in members:
            payload_start = start + np.array([length, 0.0], np.float32)
            payload_velocity = np.array([0.0, sign * sway_speed], np.float32)
            records.append((start.copy(), payload_start, payload_velocity,
                            centers.copy(), mass, length, stiffness, damping,
                            pair_id))
    fields = list(zip(*records))
    starts = np.stack(fields[0]).astype(np.float32)
    payload_starts = np.stack(fields[1]).astype(np.float32)
    payload_velocities = np.stack(fields[2]).astype(np.float32)
    centers = np.stack(fields[3]).astype(np.float32)
    masses = np.asarray(fields[4], np.float32)
    lengths = np.asarray(fields[5], np.float32)
    stiffness = np.asarray(fields[6], np.float32)
    damping = np.asarray(fields[7], np.float32)
    pair_ids = np.asarray(fields[8], np.int64)
    mass_obs = masses[:, None] + rng.normal(
        0.0, float(args.payload_obs_noise_sigma), (batch_size, int(args.horizon)))
    mass_obs = np.clip(mass_obs, 0.1, None).astype(np.float32)
    telemetry_noise = rng.normal(
        0.0, float(args.payload_obs_noise_sigma),
        (batch_size, int(args.horizon), 7),
    ).astype(np.float32)
    noise = _process_noise(args, rng, batch_size, paired)
    order = rng.permutation(batch_size) if shuffle else np.arange(batch_size)
    def tensor(value):
        return torch.from_numpy(np.asarray(value)[order])
    return SlalomBatch(
        start=tensor(starts), goal=torch.zeros(batch_size, 2),
        payload_start=tensor(payload_starts),
        payload_velocity_start=tensor(payload_velocities),
        gate_centers=tensor(centers), payload_mass=tensor(masses),
        payload_mass_obs=tensor(mass_obs), tether_length=tensor(lengths),
        tether_stiffness=tensor(stiffness), tether_damping=tensor(damping),
        telemetry_noise=tensor(telemetry_noise), process_noise=tensor(noise),
        pair_id=tensor(pair_ids),
    )


class TetheredPayloadPlant:
    """4D observed carrier with a private differentiable 4D cargo state."""

    def __init__(self, args) -> None:
        self.args = args
        self.payload_pos: torch.Tensor | None = None
        self.payload_vel: torch.Tensor | None = None
        self._history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def bind(self, batch: SlalomBatch, device: torch.device) -> None:
        self.mass = batch.payload_mass.to(device).view(-1, 1, 1)
        self.length = batch.tether_length.to(device).view(-1, 1, 1)
        self.stiffness = batch.tether_stiffness.to(device).view(-1, 1, 1)
        self.damping = batch.tether_damping.to(device).view(-1, 1, 1)
        self.payload_pos = batch.payload_start.to(device).unsqueeze(1)
        self.payload_vel = batch.payload_velocity_start.to(device).unsqueeze(1)
        zero_tension = torch.zeros_like(self.mass)
        self._history = [(self.payload_pos, self.payload_vel, zero_tension)]

    def observe(self, delay: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._history:
            raise RuntimeError("Bind the slalom batch before requesting payload telemetry.")
        return self._history[max(0, len(self._history) - 1 - max(0, delay))]

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: int | None = None) -> torch.Tensor:
        if self.payload_pos is None or self.payload_vel is None:
            raise RuntimeError("Bind the slalom batch before rollout.")
        carrier_pos, carrier_vel = as_bt(x)[..., :2], as_bt(x)[..., 2:]
        u = as_bt(u)
        step = float(self.args.dt) / max(1, int(self.args.slalom_physics_substeps))
        tension = torch.zeros_like(self.mass)
        for _ in range(max(1, int(self.args.slalom_physics_substeps))):
            relative = self.payload_pos - carrier_pos
            distance = torch.sqrt(relative.square().sum(-1, keepdim=True) + 1e-9)
            direction = relative / distance
            relative_vel = self.payload_vel - carrier_vel
            radial_speed = (relative_vel * direction).sum(-1, keepdim=True)
            raw_tension = self.stiffness * (distance - self.length) + self.damping * radial_speed
            beta = float(self.args.slalom_tension_softness)
            tension = F.softplus(beta * raw_tension) / beta
            force_on_payload = -tension * direction - float(self.args.slalom_payload_air_drag) * self.payload_vel
            payload_acc = force_on_payload / self.mass
            carrier_acc = (-float(self.args.pre_kp) * carrier_pos
                           - float(self.args.pre_kd) * carrier_vel + u
                           + float(self.args.slalom_tether_reaction) * tension * direction)
            carrier_vel = carrier_vel + step * carrier_acc
            self.payload_vel = self.payload_vel + step * payload_acc
            carrier_pos = carrier_pos + step * carrier_vel
            self.payload_pos = self.payload_pos + step * self.payload_vel
        self._history.append((self.payload_pos, self.payload_vel, tension))
        return torch.cat([carrier_pos, carrier_vel], dim=-1)


def build_slalom_context(
    args, batch: SlalomBatch, plant: TetheredPayloadPlant,
    x_t: torch.Tensor, t: int, *, mode: str, training: bool,
    intervention: str = "truth",
) -> torch.Tensor:
    state = as_bt(x_t)
    carrier_pos, carrier_vel = state[:, 0, :2], state[:, 0, 2:]
    delay = int(args.payload_context_delay)
    if intervention == "delayed":
        delay += int(args.intervention_delay_steps)
    payload_pos_bt, payload_vel_bt, tension_bt = plant.observe(delay)
    payload_pos, payload_vel, tension = payload_pos_bt[:, 0], payload_vel_bt[:, 0], tension_bt[:, 0]
    time_index = max(0, min(int(args.horizon) - 1, t - delay))
    mass = batch.payload_mass_obs[:, time_index:time_index + 1].to(x_t.device)
    relative = payload_pos - carrier_pos
    relative_vel = payload_vel - carrier_vel
    gate_xs = torch.tensor(parse_float_list(args.slalom_gate_xs, "slalom_gate_xs"),
                           device=x_t.device, dtype=x_t.dtype)
    passed = (carrier_pos[:, 0:1] <= gate_xs.view(1, -1)).sum(-1).clamp(max=len(gate_xs) - 1)
    next_x = gate_xs[passed].unsqueeze(-1)
    centers = batch.gate_centers.to(x_t.device)
    next_center = centers.gather(1, passed.unsqueeze(-1))
    gate_dx = carrier_pos[:, 0:1] - next_x
    carrier_error = carrier_pos[:, 1:2] - next_center
    payload_error = payload_pos[:, 1:2] - next_center
    noise = batch.telemetry_noise[:, time_index].to(x_t.device)
    mass_scale = max(float(args.slalom_payload_mass_max), 1e-3)
    length_scale = max(float(args.slalom_tether_length_max), 1e-3)
    relative_noisy = relative + noise[:, :2]
    relative_vel_noisy = relative_vel + noise[:, 2:4]
    extension = torch.linalg.vector_norm(relative_noisy, dim=-1, keepdim=True) - batch.tether_length.to(x_t.device).view(-1, 1)
    if intervention == "wrong":
        mass = float(args.payload_mass_ref) ** 2 / mass.clamp_min(0.1)
        flip_y = torch.tensor([1.0, -1.0], device=x_t.device, dtype=x_t.dtype)
        relative_noisy = relative_noisy * flip_y
        relative_vel_noisy = relative_vel_noisy * flip_y
        payload_error = -payload_error
    x_scale = max(float(args.start_x_max), 1.0)
    y_scale = max(float(args.corridor_limit), 1.0)
    values = {
        "next_gate_dx": gate_dx / x_scale,
        "next_gate_center": next_center / y_scale,
        "carrier_gate_error": carrier_error / y_scale,
        "gate_approach": torch.exp(-0.5 * (gate_dx / float(args.slalom_gate_focus_sigma)) ** 2),
        "goal_dx": -carrier_pos[:, 0:1] / x_scale,
        "goal_dy": -carrier_pos[:, 1:2] / y_scale,
        "vel_x": carrier_vel[:, 0:1], "vel_y": carrier_vel[:, 1:2],
        "payload_mass": mass / mass_scale,
        "payload_gate_error": payload_error / y_scale,
        "payload_rel_x": relative_noisy[:, 0:1] / length_scale,
        "payload_rel_y": relative_noisy[:, 1:2] / length_scale,
        "payload_rel_vx": relative_vel_noisy[:, 0:1],
        "payload_rel_vy": relative_vel_noisy[:, 1:2],
        "tether_extension": extension / length_scale,
        "tether_tension": tension / max(float(args.slalom_tension_scale), 1e-3),
    }
    if intervention == "dropout":
        for key in PAYLOAD_CONTEXT_FEATURES:
            values[key] = torch.zeros_like(values[key])
    if training and float(args.payload_context_dropout_p) > 0.0:
        keep = (torch.rand(state.shape[0], 1, device=x_t.device)
                >= float(args.payload_context_dropout_p)).float()
        for key in PAYLOAD_CONTEXT_FEATURES:
            values[key] = values[key] * keep
    if mode == "route_context":
        for key in PAYLOAD_CONTEXT_FEATURES:
            values[key] = torch.zeros_like(values[key])
    selected = resolve_context_features(args)
    z = torch.cat([values[key] for key in selected], dim=-1).unsqueeze(1) * float(args.z_scale)
    return torch.zeros_like(z) if mode == "disturbance_only" else z


def build_slalom_controller(device: torch.device, args, mode: str):
    controller, _ = build_controller(
        device, args, mad=(mode == "mad_context"),
        contextual=(mode == "contextual_ssm"),
    )
    return controller, TetheredPayloadPlant(args)


def rollout_slalom(
    args, batch: SlalomBatch, device: torch.device, *, mode: str,
    controller: PBController | None, plant: TetheredPayloadPlant | None,
    training: bool = False, intervention: str = "truth",
) -> SlalomRollout:
    active = plant or TetheredPayloadPlant(args)
    active.bind(batch, device)
    x = torch.cat([batch.start.to(device), torch.zeros(batch.start.shape[0], 2, device=device)], -1).unsqueeze(1)
    x_log, u_log, w_log, payload_log, payload_vel_log, tension_log = [], [], [], [], [], []
    if controller is not None:
        w0 = torch.clamp(x, -float(args.w0_clip), float(args.w0_clip)) if args.use_w0_clip else x
        controller.reset(x, w0=w0)
    for t in range(int(args.horizon)):
        if controller is None:
            u = torch.zeros(x.shape[0], 1, 2, device=device)
            w = torch.zeros_like(x)
        else:
            z = build_slalom_context(args, batch, active, x, t, mode=mode,
                                      training=training, intervention=intervention)
            u, w = controller.forward_step(x, z, t=t)
            u = torch.clamp(u, -float(args.control_limit), float(args.control_limit))
            controller.set_last_applied_control(u)
        x = active.forward(x, u, t=t)
        x = x + batch.process_noise[:, t:t + 1].to(device)
        payload_pos, payload_vel, tension = active.observe(0)
        x_log.append(x); u_log.append(u); w_log.append(w)
        payload_log.append(payload_pos); payload_vel_log.append(payload_vel); tension_log.append(tension)
    return SlalomRollout(
        x_seq=torch.cat(x_log, 1), u_seq=torch.cat(u_log, 1),
        w_seq=torch.cat(w_log, 1), payload_pos_seq=torch.cat(payload_log, 1),
        payload_vel_seq=torch.cat(payload_vel_log, 1),
        tension_seq=torch.cat(tension_log, 1),
    )
