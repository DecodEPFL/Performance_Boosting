"""Scenario generation, causal telemetry, and PB rollouts for payload regimes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from bounded_mlp_operator import BoundedMLPOperator
from context_lifting import LpContextLifter
from nav_plants import DoubleIntegratorNominal, PayloadSwitchingDoubleIntegratorTrue
from pb_core import PBController, WIntegralAugmenter, as_bt, rollout_pb
from pb_core.factories import build_factorized_controller
from pb_core.rollout import RolloutResult
from ssm_operators import ContextRescale, MpContextualSSM, MpDeepSSM


CONTEXT_FEATURE_ORDER = [
    "payload_mass", "payload_mass_delta", "actuator_gain", "lateral_bias",
    "switch_age", "goal_dx", "goal_dy", "vel_x", "vel_y",
]
CONTEXT_FEATURE_META = {
    "payload_mass": ("fair", "Measured payload mass", ("payload",)),
    "payload_mass_delta": ("fair", "Causal payload-mass change", ("payload",)),
    "actuator_gain": ("fair", "Measured actuator effectiveness", ("payload",)),
    "lateral_bias": ("fair", "Measured lateral load bias", ("payload",)),
    "switch_age": ("fair", "Age of the causal load-change event", ("payload",)),
    "goal_dx": ("fair", "Goal offset x", ("payload",)),
    "goal_dy": ("fair", "Goal offset y", ("payload",)),
    "vel_x": ("fair", "Own velocity vx", ("payload",)),
    "vel_y": ("fair", "Own velocity vy", ("payload",)),
}
FAIR_CONTEXT_DEFAULT = list(CONTEXT_FEATURE_ORDER)


@dataclass
class PayloadBatch:
    start: torch.Tensor
    goal: torch.Tensor
    mass_true: torch.Tensor
    actuator_gain_true: torch.Tensor
    drag_true: torch.Tensor
    lateral_bias_true: torch.Tensor
    mass_obs: torch.Tensor
    actuator_gain_obs: torch.Tensor
    lateral_bias_obs: torch.Tensor
    switch_age: torch.Tensor
    switch_step: torch.Tensor
    mass_before: torch.Tensor
    mass_after: torch.Tensor
    process_noise: torch.Tensor
    pair_id: torch.Tensor

    def to(self, device: torch.device) -> "PayloadBatch":
        return PayloadBatch(**{key: getattr(self, key).to(device) for key in self.__dataclass_fields__})


def resolve_context_features(args) -> list[str]:
    raw = str(getattr(args, "context_features", "") or "").strip()
    selected = set(CONTEXT_FEATURE_ORDER if not raw else (x.strip() for x in raw.split(",") if x.strip()))
    unknown = selected.difference(CONTEXT_FEATURE_ORDER)
    if unknown:
        raise ValueError(f"Unknown payload context features: {sorted(unknown)}")
    features = [key for key in CONTEXT_FEATURE_ORDER if key in selected]
    if not features:
        raise ValueError("Select at least one payload context feature.")
    return features


def context_dim(args) -> int:
    return len(resolve_context_features(args))


def resolve_contextual_modes(args) -> tuple[str, ...]:
    """Context ports for the ContextualDeepSSM variant (--ctx_modes [+ --ctx_select])."""
    modes = tuple(m.strip().lower() for m in str(getattr(args, "ctx_modes", "mixer,input,gate")).split(",") if m.strip())
    if getattr(args, "ctx_select", False) and "select" not in modes:
        modes = modes + ("select",)
    if not modes:
        raise ValueError("--ctx_modes must name at least one of mixer,input,gate,select.")
    if "select" in modes and str(args.ssm_param) not in ("tv", "tvc"):
        raise ValueError(
            "The contextual 'select' port needs a selective core: pass --ssm_param tv or tvc "
            f"(got {args.ssm_param!r}), or drop 'select'."
        )
    return modes


def contextual_modes_string(args) -> str:
    return ",".join(resolve_contextual_modes(args))


def variant_specs(args) -> list[tuple[str, str]]:
    labels = {
        "nominal": "Nominal pre-stabiliser",
        "disturbance_only": "PB+SSM: no payload telemetry",
        "context": "PB+SSM: payload-aware factorization",
        "mad_context": "PB+SSM: payload-aware MAD (s=1)",
        "contextual_ssm": f"PB+SSM: ContextualDeepSSM ({contextual_modes_string(args)})",
    }
    requested = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    unknown = set(requested).difference(labels)
    if unknown:
        raise ValueError(f"Unknown variant(s): {sorted(unknown)}")
    return [(key, labels[key]) for key in requested]


def _payload_levels(args, *, test: bool) -> tuple[float, float]:
    light = float(args.payload_light_mass)
    loaded = float(args.payload_loaded_mass)
    if test and args.test_payload_light_mass is not None:
        light = float(args.test_payload_light_mass)
    if test and args.test_payload_loaded_mass is not None:
        loaded = float(args.test_payload_loaded_mass)
    if min(light, loaded) <= 0:
        raise ValueError("Payload masses must be positive.")
    return light, loaded


def _switch_range(args, *, test: bool, horizon: int) -> tuple[int, int]:
    lo, hi = int(args.payload_switch_min), int(args.payload_switch_max)
    if test and int(args.test_switch_min) >= 0:
        lo = int(args.test_switch_min)
    if test and int(args.test_switch_max) >= 0:
        hi = int(args.test_switch_max)
    lo, hi = max(2, lo), min(horizon - 3, hi)
    if lo > hi:
        raise ValueError("Payload switch range must lie inside the rollout horizon.")
    return lo, hi


def _regime_values(args, mass: float, sign: float) -> tuple[float, float, float]:
    is_loaded = mass >= 0.5 * (float(args.payload_light_mass) + float(args.payload_loaded_mass))
    if is_loaded:
        return float(args.payload_loaded_actuator_gain), float(args.payload_loaded_drag), sign * float(args.payload_lateral_bias)
    return float(args.payload_light_actuator_gain), float(args.payload_light_drag), 0.0


def _make_schedule(args, rng: np.random.Generator, *, test: bool, mirror: float) -> tuple[np.ndarray, ...]:
    horizon = int(args.horizon)
    light, loaded = _payload_levels(args, test=test)
    before = float(rng.choice([light, loaded]))
    after = before
    switch = -1
    protocol = str(args.regime_protocol)
    if protocol != "fixed" and rng.random() < float(args.payload_switch_probability):
        switch = int(rng.integers(_switch_range(args, test=test, horizon=horizon)[0], _switch_range(args, test=test, horizon=horizon)[1] + 1))
        after = loaded if np.isclose(before, light) else light
    sign = float(rng.choice([-1.0, 1.0])) * mirror
    mass = np.full(horizon, before, np.float32)
    if switch >= 0:
        mass[switch:] = after
    gain, drag, bias = (np.zeros(horizon, np.float32) for _ in range(3))
    for t, value in enumerate(mass):
        gain[t], drag[t], bias[t] = _regime_values(args, float(value), sign)
    # Load settling: the lateral (centre-of-mass) bias fades linearly to zero over
    # --payload_bias_settle_steps after each load onset (episode start / switch),
    # like a shifted load being re-secured.  0 keeps the legacy persistent bias —
    # with settling, the reconstructed w decays once the robot docks (u, v -> 0).
    settle = int(getattr(args, "payload_bias_settle_steps", 0))
    if settle > 0:
        onset_age = np.arange(horizon, dtype=np.float32)
        if switch >= 0:
            onset_age[switch:] = np.arange(horizon - switch, dtype=np.float32)
        bias = bias * np.clip(1.0 - onset_age / float(settle), 0.0, 1.0).astype(np.float32)
    age = np.zeros(horizon, np.float32)
    if switch >= 0:
        age[switch:] = np.arange(horizon - switch, dtype=np.float32)
    return mass, gain, drag, bias, age, np.array([switch, before, after], np.float32)


def noise_decay_window(args) -> np.ndarray:
    """Multiplicative window g(t) in [0,1] on the process noise (w_t noise -> 0 by t=T).

    Modes: none (stationary, legacy), taper (flat then cosine roll-off over the
    trailing --noise_decay_ramp steps; 0 -> horizon//2), linear, exponential
    (--noise_decay_rate ** t).  Same semantics as the gate experiment.
    """
    T = int(args.horizon)
    mode = str(getattr(args, "noise_decay", "none")).lower()
    t = np.arange(T, dtype=np.float32)
    if mode == "none":
        return np.ones(T, dtype=np.float32)
    if mode == "linear":
        return np.clip(1.0 - t / max(T - 1, 1), 0.0, 1.0).astype(np.float32)
    if mode == "exponential":
        return np.power(float(getattr(args, "noise_decay_rate", 0.98)), t).astype(np.float32)
    if mode == "taper":
        ramp = int(getattr(args, "noise_decay_ramp", 0)) or max(1, T // 2)
        ramp = min(max(ramp, 1), T)
        flat_until = T - ramp
        window = np.ones(T, dtype=np.float32)
        idx = np.arange(flat_until, T, dtype=np.float32)
        window[flat_until:] = 0.5 * (1.0 + np.cos(np.pi * (idx - flat_until) / ramp))
        return window
    raise ValueError(f"Unknown --noise_decay mode {mode!r}.")


def _process_noise(args, rng: np.random.Generator, batch_size: int, paired: bool) -> np.ndarray:
    horizon, nx = int(args.horizon), 4
    out = np.zeros((batch_size, horizon, nx), dtype=np.float32)
    count = batch_size // 2 if paired else batch_size
    for i in range(count):
        seq = rng.normal(0.0, float(args.noise_vel_sigma), size=(horizon, nx)).astype(np.float32)
        seq[:, :2] *= float(args.noise_pos_multiplier)
        burst = int(rng.integers(int(args.gust_count_min), int(args.gust_count_max) + 1))
        for _ in range(burst):
            start = int(rng.integers(0, max(1, horizon - int(args.gust_duration))))
            amp = float(rng.uniform(-float(args.gust_velocity), float(args.gust_velocity)))
            seq[start:start + int(args.gust_duration), 3] += amp
        if paired:
            out[2 * i:2 * i + 2] = seq
        else:
            out[i] = seq
    window = noise_decay_window(args)
    if not np.allclose(window, 1.0):
        out = out * window[None, :, None]
    return out


def sample_batch(args, *, batch_size: int, seed: int, paired: bool, shuffle: bool, test: bool = False) -> PayloadBatch:
    if paired and batch_size % 2:
        raise ValueError("Paired payload batches require an even batch size.")
    rng, n = np.random.default_rng(seed), (batch_size // 2 if paired else batch_size)
    records: list[tuple] = []
    for pair_id in range(n):
        start_x = float(rng.uniform(float(args.start_x_min), float(args.start_x_max)));
        start_y = float(rng.uniform(-float(args.start_y_max), float(args.start_y_max)))
        raw = _make_schedule(args, rng, test=test, mirror=1.0)
        variants = [(start_y, raw)] if not paired else [
            (start_y, raw), (-start_y, (*raw[:3], -raw[3], raw[4], raw[5]))]
        for y, (mass, gain, drag, bias, age, detail) in variants:
            obs_noise = float(args.payload_obs_noise_sigma)
            mass_obs = np.clip(mass + rng.normal(0.0, obs_noise, size=mass.shape), 0.15, None).astype(np.float32)
            gain_obs = np.clip(gain + rng.normal(0.0, obs_noise, size=gain.shape), 0.15, None).astype(np.float32)
            bias_obs = (bias + rng.normal(0.0, obs_noise, size=bias.shape)).astype(np.float32)
            records.append((np.array([start_x, y], np.float32), mass, gain, drag, bias, mass_obs, gain_obs, bias_obs, age, detail, pair_id))
    fields = list(zip(*records))
    starts = np.stack(fields[0]); schedules = [np.stack(fields[i]).astype(np.float32) for i in range(1, 9)]
    detail, pair_ids = np.stack(fields[9]), np.asarray(fields[10], dtype=np.int64)
    noise = _process_noise(args, rng, batch_size, paired)
    order = rng.permutation(batch_size) if shuffle else np.arange(batch_size)
    starts, schedules, detail, pair_ids, noise = starts[order], [x[order] for x in schedules], detail[order], pair_ids[order], noise[order]
    return PayloadBatch(torch.from_numpy(starts), torch.zeros(batch_size, 2), *map(torch.from_numpy, schedules),
                        torch.from_numpy(detail[:, 0].astype(np.int64)), torch.from_numpy(detail[:, 1]), torch.from_numpy(detail[:, 2]),
                        torch.from_numpy(noise), torch.from_numpy(pair_ids))


def bind_payload(plant: PayloadSwitchingDoubleIntegratorTrue, batch: PayloadBatch, device: torch.device) -> None:
    plant.set_payload_schedule(mass=batch.mass_true.to(device), actuator_gain=batch.actuator_gain_true.to(device),
                               drag=batch.drag_true.to(device), lateral_bias=batch.lateral_bias_true.to(device))


def make_x0(batch: PayloadBatch, device: torch.device) -> torch.Tensor:
    return torch.cat([batch.start.to(device), torch.zeros(batch.start.shape[0], 2, device=device)], dim=-1).unsqueeze(1)


def build_context(args, batch: PayloadBatch, x_t: torch.Tensor, t: int, *, mode: str, training: bool, intervention: str = "truth") -> torch.Tensor:
    state, delay = as_bt(x_t), int(args.payload_context_delay)
    if intervention == "delayed": delay += int(args.intervention_delay_steps)
    ti = max(0, min(int(args.horizon) - 1, t - delay)); prev = max(0, ti - 1)
    mass = batch.mass_obs[:, ti:ti + 1].to(x_t.device); gain = batch.actuator_gain_obs[:, ti:ti + 1].to(x_t.device)
    bias = batch.lateral_bias_obs[:, ti:ti + 1].to(x_t.device); delta = mass - batch.mass_obs[:, prev:prev + 1].to(x_t.device)
    if intervention == "wrong":
        mass, gain, bias, delta = float(args.payload_mass_ref) ** 2 / mass.clamp_min(0.1), 1.0 / gain.clamp_min(0.1), -bias, -delta
    if intervention == "dropout":
        mass, gain, bias, delta = torch.full_like(mass, float(args.payload_mass_ref)), torch.ones_like(gain), torch.zeros_like(bias), torch.zeros_like(delta)
    pos, vel = state[..., :2], state[..., 2:]
    x_scale, y_scale = max(float(args.start_x_max), 1.0), max(float(args.corridor_limit), 1.0)
    values = {"payload_mass": mass / float(args.payload_mass_ref), "payload_mass_delta": delta / float(args.payload_mass_ref),
              "actuator_gain": gain, "lateral_bias": bias / max(float(args.payload_lateral_bias), 1e-3),
              "switch_age": batch.switch_age[:, ti:ti + 1].to(x_t.device) / float(args.horizon),
              "goal_dx": -pos[..., 0] / x_scale, "goal_dy": -pos[..., 1] / y_scale,
              "vel_x": vel[..., 0], "vel_y": vel[..., 1]}
    if training and float(args.payload_context_dropout_p) > 0.0:
        keep = (torch.rand(mass.shape[0], 1, device=x_t.device) >= float(args.payload_context_dropout_p)).float()
        for key in ("payload_mass", "payload_mass_delta", "actuator_gain", "lateral_bias", "switch_age"):
            values[key] = values[key] * keep
    z = torch.cat([values[key] for key in resolve_context_features(args)], dim=-1).unsqueeze(1) * float(args.z_scale)
    return torch.zeros_like(z) if mode == "disturbance_only" else z


def build_contextual_controller(device: torch.device, args) -> tuple[PBController, PayloadSwitchingDoubleIntegratorTrue]:
    """PBController whose operator is a single neural_ssm.ContextualDeepSSM.

    Ports come from --ctx_modes (+ --ctx_select); context is rescaled to
    ~unit range via --ctx_z_scale (the shared builder amplifies by --z_scale).
    """
    nx, nu, z_dim = 4, 2, context_dim(args)
    modes = resolve_contextual_modes(args)
    ramp = int(args.ctx_filter_ramp)
    gamma = float(args.ctx_gamma)
    augmenter = WIntegralAugmenter(float(args.w_augment_decay)).to(device) if args.use_w_augment else None
    rescale = float(args.ctx_z_scale) / max(float(args.z_scale), 1e-12)
    encoder = ContextRescale(rescale) if abs(rescale - 1.0) > 1e-9 else None
    operator = MpContextualSSM(
        nx, z_dim, nu,
        context_modes=modes,
        detach_state=False,
        w_augmenter=augmenter,
        context_encoder=encoder,
        context_filter=args.ctx_filter,
        horizon=int(args.horizon),
        context_filter_ramp=ramp if ramp > 0 else None,
        mixer_bound=float(args.ctx_mixer_bound),
        d_features=int(args.ctx_d_features),
        param=args.ssm_param,
        n_layers=int(args.ssm_layers),
        d_model=int(args.ssm_d_model),
        d_state=int(args.ssm_d_state),
        ff=args.ssm_ff,
        bcd_nonlinearity=args.ssm_bcd_nonlinearity,
        gamma=gamma if gamma > 0.0 else None,
    ).to(device)
    controller = PBController(plant=DoubleIntegratorNominal(dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd)),
                              operator=operator, u_nominal=None, u_dim=nu, detach_state=False).to(device)
    return controller, PayloadSwitchingDoubleIntegratorTrue(float(args.dt), float(args.pre_kp), float(args.pre_kd))


def build_controller(device: torch.device, args, *, mad: bool, contextual: bool = False) -> tuple[PBController, PayloadSwitchingDoubleIntegratorTrue]:
    if contextual:
        return build_contextual_controller(device, args)
    nx, nu, z_dim = 4, 2, context_dim(args); rank = 1 if mad else int(args.feat_dim)
    w_dim = nx * 2 if args.use_w_augment else nx
    augmenter = WIntegralAugmenter(float(args.w_augment_decay)).to(device) if args.use_w_augment else None
    lifter = LpContextLifter(z_dim=z_dim, out_dim=int(args.mp_context_lift_dim), lift_type=args.mp_context_lift_type,
        hidden_dim=int(args.mp_context_hidden_dim), decay_law=args.mp_context_decay_law, decay_rate=float(args.mp_context_decay_rate),
        decay_power=float(args.mp_context_decay_power), decay_horizon=int(args.mp_context_decay_horizon), lp_p=float(args.mp_context_lp_p), scale=float(args.mp_context_scale)).to(device)
    mp = MpDeepSSM(w_dim + int(args.mp_context_lift_dim), rank, mode="loop", param=args.ssm_param, n_layers=int(args.ssm_layers),
                   d_model=int(args.ssm_d_model), d_state=int(args.ssm_d_state), ff=args.ssm_ff).to(device)
    mb = BoundedMLPOperator(w_dim=w_dim, z_dim=z_dim, r=nu, s=rank, hidden_dim=int(args.mb_hidden), num_layers=int(args.mb_layers),
        use_z_residual=True, z_residual_gain=float(args.z_residual_gain), bound_mode="softsign", clamp_value=float(args.mb_bound)).to(device)
    controller = build_factorized_controller(nominal_plant=DoubleIntegratorNominal(dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd)),
        mp=mp, mb=mb, u_dim=nu, detach_state=False, mp_context_lifter=lifter, w_augmenter=augmenter).to(device)
    return controller, PayloadSwitchingDoubleIntegratorTrue(float(args.dt), float(args.pre_kp), float(args.pre_kd))


def rollout_variant(args, batch: PayloadBatch, device: torch.device, *, mode: str, controller: PBController | None, plant: PayloadSwitchingDoubleIntegratorTrue | None, training: bool = False, intervention: str = "truth") -> RolloutResult:
    active_plant = plant or PayloadSwitchingDoubleIntegratorTrue(float(args.dt), float(args.pre_kp), float(args.pre_kd)); bind_payload(active_plant, batch, device)
    x, x_log, u_log = make_x0(batch, device), [], []
    if mode == "nominal":
        for t in range(int(args.horizon)):
            u = torch.zeros(x.shape[0], 1, 2, device=device); x = active_plant.forward(x, u, t=t) + batch.process_noise[:, t:t + 1].to(device)
            x_log.append(x); u_log.append(u)
        return RolloutResult(torch.cat(x_log, 1), torch.cat(u_log, 1), torch.zeros_like(torch.cat(x_log, 1)))
    assert controller is not None
    return rollout_pb(controller=controller, plant_true=active_plant, x0=x, horizon=int(args.horizon), process_noise_seq=batch.process_noise.to(device),
        context_fn=lambda state, t: build_context(args, batch, state, t, mode=mode, training=training, intervention=intervention),
        w0=torch.clamp(x, -float(args.w0_clip), float(args.w0_clip)) if args.use_w0_clip else x,
        u_post_fn=lambda _x, u, _t: torch.clamp(u, -float(args.control_limit), float(args.control_limit)))
