"""2D contextual gate PB+SSM experiment with controlled x/y motion toward the origin. (v2: improved training)"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from bounded_mlp_operator import BoundedMLPOperator
from context_lifting import LpContextLifter
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_controller import PBController, as_bt
from pb_core import rollout_pb, validate_component_compatibility
from pb_core.factories import build_factorized_controller
from ssm_operators import MpDeepSSM


@dataclass
class ScenarioBatch:
    start: torch.Tensor
    goal: torch.Tensor
    gate_y: torch.Tensor
    gate_v: torch.Tensor
    gate_ema: torch.Tensor
    gate_slow_ema: torch.Tensor
    switch_age: torch.Tensor
    process_noise: torch.Tensor
    pair_id: torch.Tensor

    def to(self, device: torch.device) -> "ScenarioBatch":
        return ScenarioBatch(
            start=self.start.to(device),
            goal=self.goal.to(device),
            gate_y=self.gate_y.to(device),
            gate_v=self.gate_v.to(device),
            gate_ema=self.gate_ema.to(device),
            gate_slow_ema=self.gate_slow_ema.to(device),
            switch_age=self.switch_age.to(device),
            process_noise=self.process_noise.to(device),
            pair_id=self.pair_id.to(device),
        )


@dataclass
class RolloutArtifacts:
    x_seq: torch.Tensor
    u_seq: torch.Tensor
    w_seq: torch.Tensor


_PLT = None


def get_plt(show_plots: bool):
    global _PLT
    if _PLT is None:
        import matplotlib

        if not show_plots:
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        _PLT = plt
    return _PLT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("2D contextual PB gate experiment with PBController + SSM")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_batch", type=int, default=512)
    parser.add_argument("--val_batch", type=int, default=512)
    parser.add_argument("--test_batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--disturbance_only_epochs", type=int, default=15)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--no_show_plots", action="store_true")

    # Geometry and dynamics.
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--pre_kp", type=float, default=0.32)
    parser.add_argument("--pre_kd", type=float, default=0.80)
    parser.add_argument("--start_x_min", type=float, default=1.7)
    parser.add_argument("--start_x_max", type=float, default=2.1)
    parser.add_argument("--start_y_max", type=float, default=0.40)
    parser.add_argument("--wall_x", type=float, default=0.55)
    parser.add_argument("--gate_half_width", type=float, default=0.20)
    parser.add_argument("--gate_amplitude", type=float, default=0.95)
    parser.add_argument("--goal_tol", type=float, default=0.18)
    parser.add_argument("--corridor_limit", type=float, default=1.6)
    parser.add_argument("--wall_focus_sigma", type=float, default=0.14)
    parser.add_argument("--gate_settle_steps", type=int, default=2)

    # Gate schedule.
    parser.add_argument("--gate_dwell_min", type=int, default=6)
    parser.add_argument("--gate_dwell_max", type=int, default=16)
    parser.add_argument("--context_ema_alpha", type=float, default=0.35)
    parser.add_argument("--gate_obs_delay", type=int, default=0,
                        help="Steps of delay on gate observations (0=no delay).")
    parser.add_argument("--context_dropout_p", type=float, default=0.0,
                        help="Probability of zeroing gate context features during training.")

    # Disturbance process.
    parser.add_argument("--noise_pos_sigma", type=float, default=3e-4)
    parser.add_argument("--noise_vel_sigma", type=float, default=1.2e-3)
    parser.add_argument("--gust_count_min", type=int, default=2)
    parser.add_argument("--gust_count_max", type=int, default=4)
    parser.add_argument("--gust_duration_min", type=int, default=4)
    parser.add_argument("--gust_duration_max", type=int, default=10)
    parser.add_argument("--gust_vel_y_min", type=float, default=0.010)
    parser.add_argument("--gust_vel_y_max", type=float, default=0.028)
    parser.add_argument("--gust_vel_x_max", type=float, default=0.004)
    parser.add_argument("--gust_clip_y", type=float, default=0.045)

    # PB architecture.
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--mb_hidden", type=int, default=64)
    parser.add_argument("--mb_layers", type=int, default=4)
    parser.add_argument("--z_scale", type=float, default=6.0)
    parser.add_argument("--z_residual_gain", type=float, default=10.0)
    parser.add_argument("--mb_bound", type=float, default=8.0)
    parser.add_argument("--ssm_param", type=str, default="lru", choices=["lru", "tv"])
    parser.add_argument("--ssm_layers", type=int, default=4)
    parser.add_argument("--ssm_d_model", type=int, default=32)
    parser.add_argument("--ssm_d_state", type=int, default=64)
    parser.add_argument("--ssm_ff", type=str, default="GLU")
    parser.add_argument("--mp_only", dest="mp_only", action="store_true",
                        help="Use M_p(w) directly as u_boost, bypassing M_b entirely. "
                             "M_p output dim is set to nu (feat_dim is ignored). "
                             "Compatible with --mp_context_lift.")
    parser.add_argument("--no_mp_only", dest="mp_only", action="store_false")
    parser.set_defaults(mp_only=False)
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=True)
    parser.add_argument("--mp_context_lift_type", type=str, default="linear", choices=["identity", "linear", "mlp"])
    parser.add_argument("--mp_context_lift_dim", type=int, default=6)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=24)
    parser.add_argument("--mp_context_decay_law", type=str, default="finite", choices=["exp", "poly", "finite"])
    parser.add_argument("--mp_context_decay_rate", type=float, default=0.04)
    parser.add_argument("--mp_context_decay_power", type=float, default=0.75)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=160)
    parser.add_argument("--mp_context_lp_p", type=float, default=2.0)
    parser.add_argument("--mp_context_scale", type=float, default=0.25)

    # Loss.
    parser.add_argument("--goal_stage_weight", type=float, default=4.2)
    parser.add_argument("--goal_terminal_weight", type=float, default=60.0)
    parser.add_argument("--wall_track_weight", type=float, default=24.0)
    parser.add_argument("--wall_collision_weight", type=float, default=120.0)
    parser.add_argument("--control_weight", type=float, default=0.05)
    parser.add_argument("--corridor_weight", type=float, default=10.0)
    parser.add_argument("--collision_sharpness", type=float, default=14.0)
    parser.add_argument("--corridor_sharpness", type=float, default=12.0)
    parser.add_argument("--sample_traj_count", type=int, default=4)
    return parser.parse_args(argv)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_python_float(value) -> float:
    return float(value.item() if torch.is_tensor(value) else value)


def variant_specs() -> list[tuple[str, str]]:
    return [
        ("nominal", "Nominal only"),
        ("disturbance_only", "PB+SSM: disturbance only"),
        ("context", "PB+SSM: context-aware"),
    ]


def context_dim() -> int:
    return 11  # +1 slow EMA, +1 time-to-wall vs original 9


def epochs_for_mode(args: argparse.Namespace, mode: str) -> int:
    if mode == "disturbance_only":
        return max(1, int(args.disturbance_only_epochs))
    return max(1, int(args.epochs))


def build_gate_features(gates: np.ndarray, ema_alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gate_velocity = np.zeros_like(gates, dtype=np.float32)
    if gates.shape[1] > 1:
        gate_velocity[:, 1:] = gates[:, 1:] - gates[:, :-1]
        gate_velocity[:, 0] = gate_velocity[:, 1]

    gate_ema = np.zeros_like(gates, dtype=np.float32)
    gate_ema[:, 0] = gates[:, 0]
    for t in range(1, gates.shape[1]):
        gate_ema[:, t] = ema_alpha * gates[:, t] + (1.0 - ema_alpha) * gate_ema[:, t - 1]

    # Slow EMA (long-run gate trend, alpha=0.05 ≈ 20-step window)
    slow_alpha = 0.05
    gate_slow_ema = np.zeros_like(gates, dtype=np.float32)
    gate_slow_ema[:, 0] = gates[:, 0]
    for t in range(1, gates.shape[1]):
        gate_slow_ema[:, t] = slow_alpha * gates[:, t] + (1.0 - slow_alpha) * gate_slow_ema[:, t - 1]

    switch_age = np.zeros_like(gates, dtype=np.float32)
    for b in range(gates.shape[0]):
        age = 0.0
        for t in range(1, gates.shape[1]):
            if abs(float(gates[b, t] - gates[b, t - 1])) > 1e-6:
                age = 0.0
            else:
                age += 1.0
            switch_age[b, t] = age
    return gate_velocity, gate_ema, gate_slow_ema, switch_age


def estimate_expected_cross_index(args: argparse.Namespace) -> int:
    plant = DoubleIntegratorTrue(dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd))
    start_x = 0.5 * (float(args.start_x_min) + float(args.start_x_max))
    x = torch.tensor([[[start_x, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    u = torch.zeros(1, 1, 2, dtype=torch.float32)
    xs = []
    for _ in range(int(args.horizon)):
        x = plant.forward(x, u)
        xs.append(float(x[0, 0, 0].item()))
    xs_np = np.asarray(xs, dtype=np.float32)
    crossed = np.where(xs_np <= float(args.wall_x))[0]
    if crossed.size:
        idx = int(crossed[0])
    else:
        idx = int(np.argmin(np.abs(xs_np - float(args.wall_x))))
    if idx <= 0 or idx >= int(args.horizon):
        raise ValueError("Expected wall crossing must lie inside the rollout horizon.")
    return idx


def sample_switching_gate(
    args: argparse.Namespace,
    rng: np.random.Generator,
    freeze_step: int,
) -> np.ndarray:
    gate = np.zeros(int(args.horizon), dtype=np.float32)
    amp = float(args.gate_amplitude)
    t = 0
    last_level = 0.0
    while t < freeze_step:
        remaining = freeze_step - t
        dwell_hi = min(int(args.gate_dwell_max), remaining)
        dwell_lo = min(int(args.gate_dwell_min), dwell_hi)
        dwell_lo = max(1, dwell_lo)
        dwell = int(rng.integers(dwell_lo, dwell_hi + 1))
        # Continuous random level in [-amp, amp], biased away from last to ensure a meaningful switch
        level = float(rng.uniform(-amp, amp))
        if abs(level - last_level) < 0.25 * amp:
            level = float(rng.uniform(-amp, amp))
        gate[t : t + dwell] = level
        last_level = level
        t += dwell
    # Late adversarial switch: 30% chance of one final change close to freeze
    if rng.random() < 0.30 and freeze_step > 6:
        late_t = int(rng.integers(max(t - 6, 0), freeze_step))
        late_level = float(rng.uniform(-amp, amp))
        if abs(late_level - last_level) < 0.25 * amp:
            late_level = float(rng.uniform(-amp, amp))
        gate[late_t:freeze_step] = late_level
    gate[freeze_step:] = gate[freeze_step - 1]
    return gate


def sample_paired_process_noise(
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    batch_size: int,
    paired: bool,
) -> np.ndarray:
    horizon = int(args.horizon)
    nx = 4
    noise = np.zeros((batch_size, horizon, nx), dtype=np.float32)

    def draw_one() -> np.ndarray:
        seq = np.zeros((horizon, nx), dtype=np.float32)
        seq[:, :2] += rng.normal(scale=float(args.noise_pos_sigma), size=(horizon, 2)).astype(np.float32)
        seq[:, 2:] += rng.normal(scale=float(args.noise_vel_sigma), size=(horizon, 2)).astype(np.float32)
        burst_count = int(rng.integers(int(args.gust_count_min), int(args.gust_count_max) + 1))
        for _ in range(burst_count):
            duration = int(rng.integers(int(args.gust_duration_min), int(args.gust_duration_max) + 1))
            start = int(rng.integers(0, max(1, horizon - duration + 1)))
            amp_y = float(rng.uniform(float(args.gust_vel_y_min), float(args.gust_vel_y_max)))
            amp_y *= float(rng.choice([-1.0, 1.0]))
            amp_x = float(rng.uniform(-float(args.gust_vel_x_max), float(args.gust_vel_x_max)))
            seq[start : start + duration, 3] += amp_y
            seq[start : start + duration, 2] += amp_x
        seq[:, 3] = np.clip(seq[:, 3], -float(args.gust_clip_y), float(args.gust_clip_y))
        return seq

    if paired:
        if batch_size % 2 != 0:
            raise ValueError("Paired noise batches require an even batch size.")
        for i in range(batch_size // 2):
            seq = draw_one()
            noise[2 * i] = seq
            noise[2 * i + 1] = seq
    else:
        for i in range(batch_size):
            noise[i] = draw_one()
    return noise


def sample_batch(
    *,
    args: argparse.Namespace,
    batch_size: int,
    seed: int,
    paired: bool,
    shuffle: bool,
    expected_cross_index: int,
) -> ScenarioBatch:
    if paired and batch_size % 2 != 0:
        raise ValueError("Paired batches require an even batch size.")

    rng = np.random.default_rng(seed)
    base_count = batch_size // 2 if paired else batch_size
    freeze_step = max(1, expected_cross_index - int(args.gate_settle_steps))

    starts = []
    goals = []
    gates = []
    pair_ids = []

    for pair_idx in range(base_count):
        start_x = float(rng.uniform(float(args.start_x_min), float(args.start_x_max)))
        start_y = float(rng.uniform(-float(args.start_y_max), float(args.start_y_max)))
        gate = sample_switching_gate(args, rng, freeze_step)

        if paired:
            starts.extend([[start_x, start_y], [start_x, start_y]])
            goals.extend([[0.0, 0.0], [0.0, 0.0]])
            gates.extend([gate, -gate])
            pair_ids.extend([pair_idx, pair_idx])
        else:
            starts.append([start_x, start_y])
            goals.append([0.0, 0.0])
            gates.append(gate)
            pair_ids.append(pair_idx)

    starts_np = np.asarray(starts, dtype=np.float32)
    goals_np = np.asarray(goals, dtype=np.float32)
    gates_np = np.stack(gates, axis=0).astype(np.float32)
    pair_ids_np = np.asarray(pair_ids, dtype=np.int64)
    noise_np = sample_paired_process_noise(args=args, rng=rng, batch_size=batch_size, paired=paired)

    if shuffle:
        order = rng.permutation(batch_size)
        starts_np = starts_np[order]
        goals_np = goals_np[order]
        gates_np = gates_np[order]
        pair_ids_np = pair_ids_np[order]
        noise_np = noise_np[order]

    gate_v_np, gate_ema_np, gate_slow_ema_np, switch_age_np = build_gate_features(gates_np, float(args.context_ema_alpha))

    return ScenarioBatch(
        start=torch.from_numpy(starts_np),
        goal=torch.from_numpy(goals_np),
        gate_y=torch.from_numpy(gates_np),
        gate_v=torch.from_numpy(gate_v_np),
        gate_ema=torch.from_numpy(gate_ema_np),
        gate_slow_ema=torch.from_numpy(gate_slow_ema_np),
        switch_age=torch.from_numpy(switch_age_np),
        process_noise=torch.from_numpy(noise_np),
        pair_id=torch.from_numpy(pair_ids_np),
    )


def make_x0(batch: ScenarioBatch, device: torch.device) -> torch.Tensor:
    vel0 = torch.zeros(batch.start.shape[0], 2, device=device, dtype=batch.start.dtype)
    return torch.cat([batch.start.to(device), vel0], dim=-1).unsqueeze(1)


def build_context(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    x_t: torch.Tensor,
    t: int,
    expected_cross_index: int,
    training: bool = False,
) -> torch.Tensor:
    state = as_bt(x_t)
    pos = state[..., :2]
    x_pos = pos[..., 0]
    y_pos = pos[..., 1]

    # Apply observation delay: controller sees gate from `delay` steps ago
    delay = int(args.gate_obs_delay)
    t_obs = max(0, t - delay)
    gate_t = batch.gate_y[:, t_obs : t_obs + 1]
    gate_v_t = batch.gate_v[:, t_obs : t_obs + 1]
    gate_ema_t = batch.gate_ema[:, t_obs : t_obs + 1]
    gate_slow_ema_t = batch.gate_slow_ema[:, t_obs : t_obs + 1]
    switch_age_t = batch.switch_age[:, t_obs : t_obs + 1]

    rel_wall_x = x_pos - float(args.wall_x)
    gate_error = y_pos - gate_t
    approach = torch.exp(-0.5 * (rel_wall_x / float(args.wall_focus_sigma)) ** 2)
    goal_dx = -x_pos
    goal_dy = -y_pos
    # Normalised time remaining until expected wall crossing (positive = before wall)
    time_to_wall = torch.full_like(x_pos, (expected_cross_index - t) / max(float(args.horizon), 1.0))

    x_scale = max(float(args.start_x_max), abs(float(args.wall_x)), 1.0)
    y_scale = max(float(args.corridor_limit), abs(float(args.gate_amplitude)), 1.0)
    age_scale = max(float(args.horizon), 1.0)
    z_t = torch.cat(
        [
            gate_t / y_scale,
            gate_v_t / y_scale,
            gate_ema_t / y_scale,
            gate_slow_ema_t / y_scale,
            gate_error / y_scale,
            rel_wall_x / x_scale,
            goal_dx / x_scale,
            goal_dy / y_scale,
            approach,
            switch_age_t / age_scale,
            time_to_wall,
        ],
        dim=-1,
    )
    z_t = float(args.z_scale) * z_t
    # Context dropout: randomly zero gate-related features during training.
    # Forces the SSM to rely on history rather than instantaneous observations.
    dropout_p = float(args.context_dropout_p)
    if training and dropout_p > 0.0:
        # Per-sample mask: drop all gate features (indices 0–4) simultaneously
        mask = (torch.rand(z_t.shape[0], device=z_t.device) > dropout_p).float()
        z_t[:, :5] = z_t[:, :5] * mask.unsqueeze(-1)
    return z_t.unsqueeze(1)


def build_controller(device: torch.device, args: argparse.Namespace) -> tuple[PBController, DoubleIntegratorTrue]:
    nx = 4
    nu = 2
    z_dim = context_dim()
    feat_dim = int(args.feat_dim)
    mp_context_lifter = None
    mp_in_dim = nx

    if bool(args.mp_context_lift):
        mp_context_lifter = LpContextLifter(
            z_dim=z_dim,
            out_dim=int(args.mp_context_lift_dim),
            lift_type=args.mp_context_lift_type,
            hidden_dim=int(args.mp_context_hidden_dim),
            decay_law=args.mp_context_decay_law,
            decay_rate=float(args.mp_context_decay_rate),
            decay_power=float(args.mp_context_decay_power),
            decay_horizon=int(args.mp_context_decay_horizon),
            lp_p=float(args.mp_context_lp_p),
            scale=float(args.mp_context_scale),
        ).to(device)
        mp_in_dim = nx + int(args.mp_context_lift_dim)

    nominal_plant = DoubleIntegratorNominal(
        dt=float(args.dt),
        pre_kp=float(args.pre_kp),
        pre_kd=float(args.pre_kd),
    )
    true_plant = DoubleIntegratorTrue(
        dt=float(args.dt),
        pre_kp=float(args.pre_kp),
        pre_kd=float(args.pre_kd),
    )

    mp_only = bool(args.mp_only)
    # When mp_only, M_p outputs u directly (dim=nu); feat_dim is only used for factorized mode.
    mp_out_dim = nu if mp_only else feat_dim
    mp = MpDeepSSM(
        mp_in_dim,
        mp_out_dim,
        mode="loop",
        param=args.ssm_param,
        n_layers=int(args.ssm_layers),
        d_model=int(args.ssm_d_model),
        d_state=int(args.ssm_d_state),
        ff=args.ssm_ff,
    ).to(device)
    if mp_only:
        mb = None
    else:
        mb = BoundedMLPOperator(
            w_dim=nx,
            z_dim=z_dim,
            r=nu,
            s=feat_dim,
            hidden_dim=int(args.mb_hidden),
            num_layers=int(args.mb_layers),
            use_z_residual=True,
            z_residual_gain=float(args.z_residual_gain),
            bound_mode="softsign",
            clamp_value=float(args.mb_bound),
        ).to(device)
    controller = build_factorized_controller(
        nominal_plant=nominal_plant,
        mp=mp,
        mb=mb,
        u_dim=nu,
        detach_state=False,
        u_nominal=None,
        mp_context_lifter=mp_context_lifter,
        mp_only=mp_only,
    ).to(device)

    x_probe = torch.zeros(4, 1, nx, device=device)
    z_probe = torch.zeros(4, 1, z_dim, device=device)
    ok, msg = validate_component_compatibility(
        controller=controller,
        plant_true=true_plant,
        x0=x_probe,
        z0=z_probe,
        raise_on_error=False,
    )
    if not ok:
        raise RuntimeError(f"PB component compatibility check failed: {msg}")
    return controller, true_plant


def rollout_nominal(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
) -> RolloutArtifacts:
    batch = batch.to(device)
    plant_true = DoubleIntegratorTrue(dt=float(args.dt), pre_kp=float(args.pre_kp), pre_kd=float(args.pre_kd))
    x = make_x0(batch, device)
    x_log = []
    u_log = []
    w_log = []
    u_zero = torch.zeros(x.shape[0], 1, 2, device=device, dtype=x.dtype)
    for t in range(int(args.horizon)):
        x_next = plant_true.forward(x, u_zero, t=t) + batch.process_noise[:, t : t + 1, :]
        w_t = x_next - plant_true.forward(x, u_zero, t=t)
        x_log.append(x_next)
        u_log.append(u_zero)
        w_log.append(w_t)
        x = x_next
    return RolloutArtifacts(
        x_seq=torch.cat(x_log, dim=1),
        u_seq=torch.cat(u_log, dim=1),
        w_seq=torch.cat(w_log, dim=1),
    )


def rollout_pb_variant(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    zero_context: bool,
    expected_cross_index: int = 0,
    training: bool = False,
) -> RolloutArtifacts:
    batch = batch.to(device)
    x0 = make_x0(batch, device)
    z_dim = context_dim()

    if zero_context:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return torch.zeros(x_t.shape[0], 1, z_dim, device=x_t.device, dtype=x_t.dtype)
    else:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return build_context(args=args, batch=batch, x_t=x_t, t=t, expected_cross_index=expected_cross_index, training=training)

    result = rollout_pb(
        controller=controller,
        plant_true=plant_true,
        x0=x0,
        horizon=int(args.horizon),
        context_fn=context_fn,
        w0=x0,
        process_noise_seq=batch.process_noise,
    )
    return RolloutArtifacts(x_seq=result.x_seq, u_seq=result.u_seq, w_seq=result.w_seq)


def rollout_variant(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
    mode: str,
    controller: PBController | None,
    plant_true: DoubleIntegratorTrue | None,
    expected_cross_index: int = 0,
    training: bool = False,
) -> RolloutArtifacts:
    if mode == "nominal":
        return rollout_nominal(args=args, batch=batch, device=device)
    if controller is None or plant_true is None:
        raise ValueError(f"Controller and plant are required for mode {mode}")
    if mode == "disturbance_only":
        return rollout_pb_variant(
            args=args,
            batch=batch,
            device=device,
            controller=controller,
            plant_true=plant_true,
            zero_context=True,
            expected_cross_index=expected_cross_index,
            training=training,
        )
    if mode == "context":
        return rollout_pb_variant(
            args=args,
            batch=batch,
            device=device,
            controller=controller,
            plant_true=plant_true,
            zero_context=False,
            expected_cross_index=expected_cross_index,
            training=training,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def wall_weights(x_pos: torch.Tensor, wall_x: float, sigma: float) -> torch.Tensor:
    raw = torch.exp(-0.5 * ((x_pos - float(wall_x)) / max(float(sigma), 1e-6)) ** 2)
    return raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)


def compute_loss(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    rollout: RolloutArtifacts,
    collision_sharpness_override: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    goal = batch.goal.to(rollout.x_seq.device).unsqueeze(1)
    pos = rollout.x_seq[..., :2]
    x_pos = pos[..., 0]
    y_pos = pos[..., 1]
    gate = batch.gate_y.to(rollout.x_seq.device)

    # Smooth L1 (Huber) for goal distance — more robust than pure L2
    goal_delta = pos - goal
    goal_dist = F.huber_loss(pos, goal.expand_as(pos), reduction="none", delta=0.5).sum(dim=-1)
    goal_stage = float(args.goal_stage_weight) * goal_dist.mean(dim=1)
    goal_dist_l2 = torch.norm(goal_delta, dim=-1)
    goal_term = float(args.goal_terminal_weight) * goal_dist_l2[:, -1]

    w_wall = wall_weights(x_pos, float(args.wall_x), float(args.wall_focus_sigma))
    gate_error = y_pos - gate
    wall_track = float(args.wall_track_weight) * (w_wall * gate_error.square()).sum(dim=1)

    sharpness = collision_sharpness_override if collision_sharpness_override is not None else float(args.collision_sharpness)
    collision_soft = F.softplus(
        sharpness * (gate_error.abs() - float(args.gate_half_width))
    ) / max(sharpness, 1e-6)
    wall_collision = float(args.wall_collision_weight) * (w_wall * collision_soft).sum(dim=1)

    control_mag_sq = torch.sum(rollout.u_seq.square(), dim=-1)
    control_cost = float(args.control_weight) * control_mag_sq.mean(dim=1)

    corridor_soft = F.softplus(
        float(args.corridor_sharpness) * (y_pos.abs() - float(args.corridor_limit))
    ) / float(args.corridor_sharpness)
    corridor_cost = float(args.corridor_weight) * corridor_soft.mean(dim=1)

    total_per = goal_stage + goal_term + wall_track + wall_collision + control_cost + corridor_cost
    parts = {
        "loss_total": to_python_float(total_per.mean()),
        "loss_goal_stage": to_python_float(goal_stage.mean()),
        "loss_goal_term": to_python_float(goal_term.mean()),
        "loss_wall_track": to_python_float(wall_track.mean()),
        "loss_wall_collision": to_python_float(wall_collision.mean()),
        "loss_control": to_python_float(control_cost.mean()),
        "loss_corridor": to_python_float(corridor_cost.mean()),
    }
    return total_per.mean(), parts


def crossing_indices(x_pos: torch.Tensor, wall_x: float) -> torch.Tensor:
    x_cpu = x_pos.detach().cpu()
    idx = torch.zeros(x_cpu.shape[0], dtype=torch.long)
    for b in range(x_cpu.shape[0]):
        crossed = torch.nonzero(x_cpu[b] <= float(wall_x), as_tuple=False).squeeze(-1)
        if crossed.numel() > 0:
            idx[b] = int(crossed[0].item())
        else:
            idx[b] = int(torch.argmin(torch.abs(x_cpu[b] - float(wall_x))).item())
    return idx.to(x_pos.device)


@torch.no_grad()
def evaluate_variant(
    *,
    args: argparse.Namespace,
    batch: ScenarioBatch,
    device: torch.device,
    mode: str,
    controller: PBController | None,
    plant_true: DoubleIntegratorTrue | None,
    expected_cross_index: int = 0,
) -> dict:
    rollout = rollout_variant(
        args=args,
        batch=batch,
        device=device,
        mode=mode,
        controller=controller,
        plant_true=plant_true,
        expected_cross_index=expected_cross_index,
    )
    avg_cost, loss_parts = compute_loss(args=args, batch=batch, rollout=rollout)
    batch_dev = batch.to(rollout.x_seq.device)
    pos = rollout.x_seq[..., :2]
    x_pos = pos[..., 0]
    y_pos = pos[..., 1]
    cross_idx = crossing_indices(x_pos, float(args.wall_x))
    row = torch.arange(x_pos.shape[0], device=x_pos.device)
    y_cross = y_pos[row, cross_idx]
    g_cross = batch_dev.gate_y[row, cross_idx]
    cross_error = y_cross - g_cross
    collided = cross_error.abs() > float(args.gate_half_width)
    terminal_dist = torch.norm(pos[:, -1, :] - batch_dev.goal, dim=-1)
    goal_success = terminal_dist < float(args.goal_tol)
    success = (~collided) & goal_success

    metrics = {
        "avg_cost": to_python_float(avg_cost),
        "success_rate": to_python_float(success.float().mean()),
        "wall_success_rate": to_python_float((~collided).float().mean()),
        "goal_success_rate": to_python_float(goal_success.float().mean()),
        "collision_rate": to_python_float(collided.float().mean()),
        "avg_abs_cross_error": to_python_float(cross_error.abs().mean()),
        "avg_terminal_dist": to_python_float(terminal_dist.mean()),
        "avg_control_energy": to_python_float(torch.sum(rollout.u_seq.square(), dim=-1).mean()),
        "avg_abs_reconstructed_w": to_python_float(rollout.w_seq.abs().mean()),
    }
    metrics.update(loss_parts)
    metrics["rollout"] = {
        "x_seq": rollout.x_seq.detach().cpu(),
        "u_seq": rollout.u_seq.detach().cpu(),
        "w_seq": rollout.w_seq.detach().cpu(),
        "cross_idx": cross_idx.detach().cpu(),
    }
    return metrics


def choose_better_result(candidate: tuple[float, float, float], incumbent: tuple[float, float, float] | None) -> bool:
    if incumbent is None:
        return True
    return candidate < incumbent


def print_train_epoch_status(
    *,
    mode: str,
    epoch: int,
    epochs: int,
    record: dict,
    val_metrics: dict | None = None,
    is_best: bool = False,
) -> None:
    train_msg = (
        f"[{mode}] epoch {epoch:03d}/{epochs:03d} "
        f"train={record['train_loss']:.4f} "
        f"goal_s={record['loss_goal_stage']:.4f} "
        f"goal_T={record['loss_goal_term']:.4f} "
        f"wall={record['loss_wall_track']:.4f} "
        f"coll={record['loss_wall_collision']:.4f} "
        f"ctrl={record['loss_control']:.4f} "
        f"corr={record['loss_corridor']:.4f} "
        f"lr={record['lr']:.5f}"
    )
    print(train_msg)
    if val_metrics is None:
        return
    best_tag = " best" if is_best else ""
    val_msg = (
        f"[{mode}]            "
        f"val={val_metrics['avg_cost']:.4f} "
        f"succ={val_metrics['success_rate']:.3f} "
        f"wall={val_metrics['wall_success_rate']:.3f} "
        f"goal={val_metrics['goal_success_rate']:.3f} "
        f"term={val_metrics['avg_terminal_dist']:.3f}"
        f"{best_tag}"
    )
    print(val_msg)


def train_controller(
    *,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    val_batch: ScenarioBatch,
    expected_cross_index: int,
    warm_start_state: dict | None = None,
) -> tuple[PBController | None, DoubleIntegratorTrue | None, list[dict], dict]:
    if mode == "nominal":
        metrics = evaluate_variant(
            args=args,
            batch=val_batch,
            device=device,
            mode=mode,
            controller=None,
            plant_true=None,
            expected_cross_index=expected_cross_index,
        )
        return None, None, [], metrics

    mode_epochs = epochs_for_mode(args, mode)
    controller, plant_true = build_controller(device, args)

    # Warm-start from a previously trained state (e.g. disturbance_only → context)
    if warm_start_state is not None:
        controller.load_state_dict(warm_start_state)
        print(f"[{mode}] warm-started from provided checkpoint.")

    # AdamW with separate LRs for SSM (operator.mp) and MLP operator (operator.mb + rest)
    mp_param_ids = {id(p) for p in controller.operator.mp.parameters()}
    param_groups = [
        {"params": list(controller.operator.mp.parameters()), "lr": float(args.lr)},
        {"params": [p for p in controller.parameters() if id(p) not in mp_param_ids], "lr": float(args.lr) * 0.5},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Linear warmup (8 epochs) then cosine decay
    warmup_epochs = min(8, mode_epochs // 4)
    cosine_epochs = max(1, mode_epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=float(args.lr_min)),
        ],
        milestones=[warmup_epochs],
    )

    history: list[dict] = []
    best_state = None
    best_score = None
    best_val_metrics = None

    for epoch in range(1, mode_epochs + 1):
        controller.train()
        train_batch = sample_batch(
            args=args,
            batch_size=int(args.train_batch),
            seed=int(args.seed) + 1000 + epoch,
            paired=True,
            shuffle=True,
            expected_cross_index=expected_cross_index,
        )

        # Curriculum: ramp collision sharpness from 20% to 100% over first 40% of training
        sharpness_frac = min(1.0, epoch / max(1, 0.4 * mode_epochs))
        curr_sharpness = float(args.collision_sharpness) * (0.2 + 0.8 * sharpness_frac)

        rollout = rollout_variant(
            args=args,
            batch=train_batch,
            device=device,
            mode=mode,
            controller=controller,
            plant_true=plant_true,
            expected_cross_index=expected_cross_index,
            training=True,
        )
        loss, parts = compute_loss(
            args=args,
            batch=train_batch,
            rollout=rollout,
            collision_sharpness_override=curr_sharpness,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(controller.parameters(), float(args.grad_clip))
        optimizer.step()

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": to_python_float(loss),
            "lr": to_python_float(scheduler.get_last_lr()[0]),
            "curr_sharpness": curr_sharpness,
        }
        record.update(parts)

        if epoch % int(args.eval_every) == 0 or epoch == mode_epochs:
            controller.eval()
            val_metrics = evaluate_variant(
                args=args,
                batch=val_batch,
                device=device,
                mode=mode,
                controller=controller,
                plant_true=plant_true,
                expected_cross_index=expected_cross_index,
            )
            record["val_cost"] = float(val_metrics["avg_cost"])
            record["val_success_rate"] = float(val_metrics["success_rate"])
            candidate_score = (
                1.0 - float(val_metrics["success_rate"]),
                1.0 - float(val_metrics["wall_success_rate"]),
                float(val_metrics["avg_cost"]),
            )
            is_best = False
            if choose_better_result(candidate_score, best_score):
                best_score = candidate_score
                best_val_metrics = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
                is_best = True
            print_train_epoch_status(
                mode=mode,
                epoch=epoch,
                epochs=mode_epochs,
                record=record,
                val_metrics=val_metrics,
                is_best=is_best,
            )
        else:
            print_train_epoch_status(
                mode=mode,
                epoch=epoch,
                epochs=mode_epochs,
                record=record,
            )
        history.append(record)

    if best_state is None:
        raise RuntimeError(f"Training for mode {mode} did not produce any validation checkpoint.")
    controller.load_state_dict(best_state)
    return controller, plant_true, history, best_val_metrics


def setup_plot_style(plt) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def variant_colors() -> dict[str, str]:
    return {
        "nominal": "#4b5563",
        "disturbance_only": "#d97706",
        "context": "#0f766e",
    }


def plot_wall_style_summary(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    labels = {mode: label for mode, label in variant_order}

    fig = plt.figure(figsize=(16.0, 12.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.2])

    sample_idx = select_trajectory_indices(test_batch, args)[0]
    start = test_batch.start[sample_idx].numpy()
    start_x = float(start[0])
    start_y = float(start[1])
    gate_traj = test_batch.gate_y[sample_idx].numpy()
    gate_x_ref = np.linspace(start_x, 0.0, int(args.horizon))
    context_cross_idx = int(test_metrics["context"]["rollout"]["cross_idx"][sample_idx].item())
    gate_center = float(gate_traj[min(context_cross_idx, len(gate_traj) - 1)])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.step(
        gate_x_ref,
        gate_traj,
        where="post",
        color="grey",
        linestyle="--",
        alpha=0.6,
        linewidth=2.0,
        label="Gate schedule $g_t$",
    )
    ax1.fill_between(
        gate_x_ref,
        gate_traj - float(args.gate_half_width),
        gate_traj + float(args.gate_half_width),
        step="post",
        color="grey",
        alpha=0.12,
    )
    draw_wall(ax1, float(args.wall_x), gate_center, float(args.gate_half_width), float(args.corridor_limit))

    for mode, label in variant_order:
        traj = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, :2].numpy()
        traj_full = np.vstack([start, traj])
        lw = 2.6 if mode == "context" else 2.0
        ax1.plot(traj_full[:, 0], traj_full[:, 1], color=colors[mode], lw=lw, label=label)

    ax1.scatter([start_x], [start_y], color="#6b7280", s=36, zorder=4, label="Start")
    ax1.scatter([0.0], [0.0], color="#111827", marker="*", s=95, zorder=5, label="Goal (0,0)")
    ax1.set_title("Top-Down View: Robot Navigating the Corridor", fontsize=14, fontweight="bold")
    ax1.set_xlabel("x position", fontsize=12)
    ax1.set_ylabel("y position", fontsize=12)
    ax1.set_xlim(-0.15, max(float(args.start_x_max), start_x) + 0.15)
    ax1.set_ylim(-float(args.corridor_limit) - 0.1, float(args.corridor_limit) + 0.1)
    ax1.legend(loc="upper right", ncol=2)

    ax2 = fig.add_subplot(gs[1, 0])
    bar_labels = [labels[mode] for mode, _ in variant_order]
    success_rates = [100.0 * float(test_metrics[mode]["wall_success_rate"]) for mode, _ in variant_order]
    bar_colors = [colors[mode] for mode, _ in variant_order]
    ax2.bar(bar_labels, success_rates, color=bar_colors, alpha=0.82)
    ax2.set_title("Gate Crossing Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.0, 105.0)
    for idx, value in enumerate(success_rates):
        ax2.text(idx, value + 2.0, f"{value:.1f}%", ha="center", fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    avg_miss = [float(test_metrics[mode]["avg_abs_cross_error"]) for mode, _ in variant_order]
    ax3.bar(bar_labels, avg_miss, color=bar_colors, alpha=0.82)
    ax3.set_title(
        f"Avg |y - g_t| at Wall (Safe threshold < {float(args.gate_half_width):.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.axhline(float(args.gate_half_width), color="red", linestyle="--", label="Safe bound")
    ax3.legend(loc="upper right")
    y_offset = max(0.03, 0.05 * max(avg_miss + [float(args.gate_half_width)]))
    for idx, value in enumerate(avg_miss):
        ax3.text(idx, value + y_offset, f"{value:.2f}", ha="center", fontweight="bold")

    ax4 = fig.add_subplot(gs[2, :])
    t = np.arange(1, int(args.horizon) + 1)
    ax4.step(
        t,
        gate_traj,
        where="post",
        color="grey",
        linestyle="--",
        alpha=0.7,
        linewidth=2.0,
        label="Gate center $g_t$",
    )
    ax4.fill_between(
        t,
        gate_traj - float(args.gate_half_width),
        gate_traj + float(args.gate_half_width),
        step="post",
        color="grey",
        alpha=0.12,
    )
    for mode, label in variant_order:
        y_seq = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, 1].numpy()
        cross_idx = int(test_metrics[mode]["rollout"]["cross_idx"][sample_idx].item()) + 1
        ax4.plot(t, y_seq, color=colors[mode], lw=2.2, label=f"{label} $y_t$")
        ax4.axvline(cross_idx, color=colors[mode], linestyle=":", alpha=0.45, linewidth=1.4)
    ax4.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax4.set_title("Gate Switching Over Time For One Representative Episode", fontsize=14, fontweight="bold")
    ax4.set_xlabel("time step", fontsize=12)
    ax4.set_ylabel("lateral position / gate center", fontsize=12)
    ax4.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    fig.savefig(run_dir / "wall_style_summary.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_loss_curves(
    *,
    run_dir: Path,
    histories: dict[str, list[dict]],
    variant_order: list[tuple[str, str]],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for mode, label in variant_order:
        history = histories.get(mode, [])
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_epochs = [h["epoch"] for h in history if "val_cost" in h]
        val_cost = [h["val_cost"] for h in history if "val_cost" in h]
        ax.plot(epochs, train_loss, color=colors[mode], alpha=0.28, lw=1.5)
        ax.plot(val_epochs, val_cost, color=colors[mode], lw=2.3, label=label)
    ax.set_title("Training / validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curves.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_control_magnitude(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    t = np.arange(int(args.horizon))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for mode, label in variant_order:
        u_seq = test_metrics[mode]["rollout"]["u_seq"].numpy()
        u_mag = np.linalg.norm(u_seq, axis=-1)
        mean = np.mean(u_mag, axis=0)
        q10 = np.quantile(u_mag, 0.10, axis=0)
        q90 = np.quantile(u_mag, 0.90, axis=0)
        ax.plot(t, mean, color=colors[mode], lw=2.2, label=label)
        ax.fill_between(t, q10, q90, color=colors[mode], alpha=0.12)
    ax.set_title("Control magnitude over time")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\|u_t\|_2$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "control_magnitude_over_time.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def select_trajectory_indices(batch: ScenarioBatch, args: argparse.Namespace) -> list[int]:
    gate = batch.gate_y.numpy()
    noise = batch.process_noise.numpy()
    strength = np.mean(np.abs(noise[..., 3]), axis=1)
    cross_gate = np.mean(gate[:, -12:], axis=1)
    idx_pos = int(np.argmax(np.where(cross_gate > 0, strength, -np.inf))) if np.any(cross_gate > 0) else 0
    idx_neg = int(np.argmax(np.where(cross_gate < 0, strength, -np.inf))) if np.any(cross_gate < 0) else idx_pos
    remaining = [i for i in range(gate.shape[0]) if i not in {idx_pos, idx_neg}]
    picks = [idx_pos, idx_neg]
    for idx in remaining[: max(0, int(args.sample_traj_count) - 2)]:
        picks.append(int(idx))
    return picks[: int(args.sample_traj_count)]


def draw_wall(ax, wall_x: float, gate_center: float, half_width: float, y_limit: float) -> None:
    ax.plot([wall_x, wall_x], [-y_limit, gate_center - half_width], color="black", lw=3.0)
    ax.plot([wall_x, wall_x], [gate_center + half_width, y_limit], color="black", lw=3.0)
    ax.plot([wall_x, wall_x], [gate_center - half_width, gate_center + half_width], color="white", lw=5.0)


def plot_trajectory_samples(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)
    colors = variant_colors()
    idxs = select_trajectory_indices(test_batch, args)
    n = len(idxs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, idx in zip(axes_flat, idxs):
        ctx_roll = test_metrics["context"]["rollout"]
        cross_idx = int(ctx_roll["cross_idx"][idx].item())
        gate_center = float(test_batch.gate_y[idx, cross_idx].item())
        draw_wall(ax, float(args.wall_x), gate_center, float(args.gate_half_width), float(args.corridor_limit))
        for mode, label in variant_order:
            traj = test_metrics[mode]["rollout"]["x_seq"][idx, :, :2].numpy()
            start = test_batch.start[idx].numpy()
            traj_full = np.vstack([start, traj])
            ax.plot(traj_full[:, 0], traj_full[:, 1], color=colors[mode], lw=2.2, label=label)
        ax.scatter([test_batch.start[idx, 0].item()], [test_batch.start[idx, 1].item()], color="#6b7280", s=32, zorder=4)
        ax.scatter([0.0], [0.0], color="#111827", marker="*", s=90, zorder=5, label="Goal (0,0)")
        ax.set_title(f"Sample #{idx} | gate @ wall = {gate_center:+.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.25, float(args.start_x_max) + 0.15)
        ax.set_ylim(-float(args.corridor_limit) - 0.15, float(args.corridor_limit) + 0.15)
        ax.legend(loc="upper right")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Representative trajectory samples", y=0.99, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "trajectory_samples.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_waiting_behavior(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    """Visualise the 'waiting' strategy: controller slows forward/lateral motion after a recent gate switch."""
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    roll      = test_metrics["context"]["rollout"]
    x_seq_t   = roll["x_seq"]                          # (B, T, 4)
    cross_idx = roll["cross_idx"]                       # (B,)
    B, T, _   = x_seq_t.shape

    vx        = x_seq_t[:, :, 2].numpy()               # forward velocity  (B, T)
    vy        = x_seq_t[:, :, 3].numpy()               # lateral velocity  (B, T)
    y_pos     = x_seq_t[:, :, 1].numpy()               # y position        (B, T)
    gate_np   = test_batch.gate_y.numpy()              # (B, T)
    sw_age    = test_batch.switch_age.numpy()          # (B, T)

    # switch_age at the wall-crossing step for each sample
    ci_np     = cross_idx.numpy().astype(int)
    ci_np     = np.clip(ci_np, 0, T - 1)
    age_at_cross = sw_age[np.arange(B), ci_np]         # (B,)

    dwell_min     = int(args.gate_dwell_min)
    recent_mask   = age_at_cross <= dwell_min           # gate switched recently
    committed_mask = ~recent_mask

    # ── align every sample to its crossing step ───────────────────────────────
    W_before, W_after = 28, 8
    W = W_before + W_after
    t_rel = np.arange(-W_before, W_after)

    def _aligned(arr2d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return (n_valid, W) array of arr2d windows centred on crossing."""
        rows = []
        for i in np.where(mask)[0]:
            lo, hi = int(ci_np[i]) - W_before, int(ci_np[i]) + W_after
            if lo >= 0 and hi <= T:
                rows.append(arr2d[i, lo:hi])
        return np.array(rows) if rows else np.zeros((0, W))

    rec_vx     = _aligned(np.abs(vx),                  recent_mask)
    com_vx     = _aligned(np.abs(vx),                  committed_mask)
    rec_vy     = _aligned(np.abs(vy),                  recent_mask)
    com_vy     = _aligned(np.abs(vy),                  committed_mask)
    rec_err    = _aligned(np.abs(y_pos - gate_np),     recent_mask)
    com_err    = _aligned(np.abs(y_pos - gate_np),     committed_mask)

    # ── pick 3 representative individual traces per group ─────────────────────
    def _pick_examples(mask: np.ndarray, n: int = 3) -> list[int]:
        idxs = list(np.where(mask)[0])
        if not idxs:
            return []
        # sort by age_at_cross and pick spread: min / median / max
        idxs.sort(key=lambda i: age_at_cross[i])
        picks = []
        for pos in [0, len(idxs) // 2, len(idxs) - 1]:
            picks.append(idxs[pos])
        return list(dict.fromkeys(picks))[:n]

    rec_ex  = _pick_examples(recent_mask)
    com_ex  = _pick_examples(committed_mask)

    # ── colours ───────────────────────────────────────────────────────────────
    C_REC  = "#f97316"   # orange  — recent switch (uncertain)
    C_COM  = "#22d3ee"   # cyan    — committed (stable gate)
    C_ZERO = "#94a3b8"   # grey

    # ── figure: 4 rows (vx | vy | gate-error | examples) ─────────────────────
    fig = plt.figure(figsize=(14, 14))
    gs  = fig.add_gridspec(4, 2, hspace=0.52, wspace=0.32,
                           height_ratios=[1.0, 1.0, 1.0, 1.2])

    ax_vx    = fig.add_subplot(gs[0, :])   # full-width: forward speed |vx|
    ax_vy    = fig.add_subplot(gs[1, :])   # full-width: lateral speed |vy|
    ax_err   = fig.add_subplot(gs[2, :])   # full-width: |gate error|
    ax_rec   = fig.add_subplot(gs[3, 0])   # example trajectories — recent
    ax_com   = fig.add_subplot(gs[3, 1])   # example trajectories — committed

    def _style(ax_in):
        ax_in.axvline(0, color="#f1f5f9", lw=1.3, linestyle="--", alpha=0.7, zorder=3)
        ax_in.set_xlabel("steps relative to wall crossing")
        ax_in.tick_params(labelsize=9)

    def _plot_speed_panel(ax_in, rec_data, com_data, ylabel, title):
        for data, color, label in [
            (rec_data, C_REC, f"recent switch  (age \u2264 {dwell_min},  n={len(rec_data)})"),
            (com_data, C_COM, f"committed       (age > {dwell_min},  n={len(com_data)})"),
        ]:
            if data.shape[0] == 0:
                continue
            mu  = data.mean(axis=0)
            p_lo = np.percentile(data, 20, axis=0)
            p_hi = np.percentile(data, 80, axis=0)
            ax_in.fill_between(t_rel, p_lo, p_hi, color=color, alpha=0.18)
            ax_in.plot(t_rel, mu, color=color, lw=2.4, label=label)
            for row in data[::max(1, len(data) // 6)]:
                ax_in.plot(t_rel, row, color=color, lw=0.7, alpha=0.25)
        ax_in.axvspan(-dwell_min, 0, color=C_REC, alpha=0.06,
                      label=f"gate-dwell window ({dwell_min} steps)")
        ax_in.set_ylabel(ylabel)
        ax_in.set_title(title, fontweight="bold")
        ax_in.legend(fontsize=9, loc="upper left")
        _style(ax_in)

    # ── Panel 1: forward speed |vx| — primary "waiting" signal ───────────────
    _plot_speed_panel(
        ax_vx, rec_vx, com_vx,
        ylabel=r"$|v_x|$  forward speed",
        title=r"Forward speed $|v_x|$ around wall crossing — slowing = buying time",
    )

    # ── Panel 2: lateral speed |vy| ───────────────────────────────────────────
    _plot_speed_panel(
        ax_vy, rec_vy, com_vy,
        ylabel=r"$|v_y|$  lateral speed",
        title=r"Lateral speed $|v_y|$ around wall crossing — repositioning",
    )

    # ── Panel 3: |gate error| ─────────────────────────────────────────────────
    for data, color in [(rec_err, C_REC), (com_err, C_COM)]:
        if data.shape[0] == 0:
            continue
        mu   = data.mean(axis=0)
        p_lo = np.percentile(data, 20, axis=0)
        p_hi = np.percentile(data, 80, axis=0)
        ax_err.fill_between(t_rel, p_lo, p_hi, color=color, alpha=0.18)
        ax_err.plot(t_rel, mu, color=color, lw=2.4)

    ax_err.axhline(float(args.gate_half_width), color="#ef4444", lw=1.3,
                   linestyle=":", label=f"gate half-width {args.gate_half_width:.2f}")
    ax_err.set_ylabel(r"$|y - g_t|$  gate error")
    ax_err.set_title("|y \u2212 gate| around wall crossing", fontweight="bold")
    ax_err.legend(fontsize=9, loc="upper left")
    _style(ax_err)

    # ── Panels 4 & 5: individual example trajectories ────────────────────────
    for ax_ex, examples, color, title in [
        (ax_rec, rec_ex,  C_REC, f"Recent-switch examples  (age \u2264 {dwell_min})"),
        (ax_com, com_ex,  C_COM, f"Committed examples       (age > {dwell_min})"),
    ]:
        for idx in examples:
            ci   = int(ci_np[idx])
            t_ax = np.arange(T)
            g    = gate_np[idx]
            y    = y_pos[idx]

            ax_ex.fill_between(t_ax, g - float(args.gate_half_width),
                               g + float(args.gate_half_width),
                               step="post", color=C_ZERO, alpha=0.10)
            ax_ex.step(t_ax, g, where="post", color=C_ZERO, lw=1.2,
                       alpha=0.55, linestyle="--")
            ax_ex.plot(t_ax, y, color=color, lw=1.8, alpha=0.85)
            ax_ex.axvline(ci, color=color, lw=1.0, linestyle=":", alpha=0.6)

            # mark switch events as dots on the gate line
            switches = np.where(np.abs(np.diff(g)) > 1e-4)[0]
            ax_ex.scatter(switches, g[switches], color="#fbbf24", s=22, zorder=4)

            # annotate switch_age at crossing
            ax_ex.text(ci + 0.5, float(y[ci]) + 0.05,
                       f"age={int(age_at_cross[idx])}",
                       fontsize=7, color=color, alpha=0.85)

        ax_ex.axhline(0, color=C_ZERO, lw=0.7, alpha=0.4)
        ax_ex.set_xlim(0, T - 1)
        ax_ex.set_ylim(-float(args.corridor_limit) - 0.1, float(args.corridor_limit) + 0.1)
        ax_ex.set_xlabel("time step")
        ax_ex.set_ylabel("y position")
        ax_ex.set_title(title, fontweight="bold", fontsize=10)
        ax_ex.scatter([], [], color="#fbbf24", s=22, label="gate switch event")
        ax_ex.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Waiting-behaviour analysis: does the controller hold back after a recent gate switch?",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "waiting_behavior.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def _animate_one_sample(
    *,
    plt,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    sample_idx: int,
    file_tag: str,
    show_plots: bool,
) -> None:
    """Render and save one animated GIF for the given sample index."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe

    colors  = variant_colors()
    horizon = int(args.horizon)
    wall_x  = float(args.wall_x)
    half_w  = float(args.gate_half_width)
    corr    = float(args.corridor_limit)
    x_max   = float(args.start_x_max) + 0.15

    gate_traj = test_batch.gate_y[sample_idx].numpy()
    start_np  = test_batch.start[sample_idx].numpy()

    # per-variant trajectories: prepend start → (T+1, 2)
    trajs: dict[str, np.ndarray] = {}
    for mode, _ in variant_order:
        xy = test_metrics[mode]["rollout"]["x_seq"][sample_idx, :, :2].numpy()
        trajs[mode] = np.vstack([start_np, xy])

    # outcome labels shown in title
    def _outcome(mode: str) -> str:
        roll   = test_metrics[mode]["rollout"]
        ci     = int(roll["cross_idx"][sample_idx].item())
        y_cross = float(trajs[mode][ci + 1, 1])
        g_cross = float(gate_traj[ci])
        hit     = abs(y_cross - g_cross) <= half_w
        x_term  = float(trajs[mode][-1, 0])
        y_term  = float(trajs[mode][-1, 1])
        goal_ok = (x_term ** 2 + y_term ** 2) ** 0.5 < float(args.goal_tol)
        gate_sym = "[G:pass]" if hit else "[G:fail]"
        goal_sym = "[O:pass]" if goal_ok else "[O:fail]"
        return f"{gate_sym} {goal_sym}"

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8))
    gs  = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.38)
    ax_arena = fig.add_subplot(gs[0])
    ax_time  = fig.add_subplot(gs[1])

    fig.patch.set_facecolor("#0f172a")
    for ax in (ax_arena, ax_time):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")
        ax.grid(color="#334155", linewidth=0.6, alpha=0.5)

    # ── static arena elements ─────────────────────────────────────────────────
    ax_arena.set_xlim(-0.15, x_max)
    ax_arena.set_ylim(-corr - 0.12, corr + 0.12)
    ax_arena.set_xlabel("x position")
    ax_arena.set_ylabel("y position")
    ax_arena.scatter([0.0], [0.0], color="#f8fafc", marker="*", s=160, zorder=6,
                     label="Goal",
                     path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")])
    ax_arena.scatter([start_np[0]], [start_np[1]], color="#94a3b8", s=55,
                     zorder=5, marker="o", label="Start")

    for mode, _ in variant_order:
        ax_arena.plot(trajs[mode][:, 0], trajs[mode][:, 1],
                      color=colors[mode], lw=1.2, alpha=0.15, zorder=2)

    trail_lines: dict[str, object] = {}
    dots: dict[str, object] = {}
    for mode, lbl in variant_order:
        lw = 2.8 if mode == "context" else 1.8
        outcome = _outcome(mode)
        ln, = ax_arena.plot([], [], color=colors[mode], lw=lw, zorder=4,
                            label=f"{lbl}  {outcome}",
                            path_effects=[pe.withStroke(linewidth=lw + 1.5,
                                                        foreground="#0f172a")])
        dot, = ax_arena.plot([], [], "o", color=colors[mode], ms=9, zorder=7,
                             path_effects=[pe.withStroke(linewidth=2.5,
                                                         foreground="#0f172a")])
        trail_lines[mode] = ln
        dots[mode]        = dot

    # wall: two solid rects + transparent gate opening rect (updated each frame)
    rect_upper = Rectangle((wall_x - 0.015, 0.0),    0.030, corr + 0.12,
                            color="#f1f5f9", zorder=3)
    rect_lower = Rectangle((wall_x - 0.015, -corr - 0.12), 0.030, corr + 0.12,
                            color="#f1f5f9", zorder=3)
    gate_patch = Rectangle((wall_x - 0.015, 0.0),    0.030, 2 * half_w,
                            color="#1e293b", zorder=4)
    for p in (rect_upper, rect_lower, gate_patch):
        ax_arena.add_patch(p)

    step_text = ax_arena.text(
        0.02, 0.97, "", transform=ax_arena.transAxes,
        color="#e2e8f0", fontsize=11, fontweight="bold", va="top", ha="left",
        path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")],
    )

    ax_arena.legend(loc="upper right", fontsize=8.5, facecolor="#1e293b",
                    edgecolor="#475569", labelcolor="#e2e8f0", framealpha=0.85)
    ax_arena.set_title(
        f"Gate-Crossing Navigation — Sample #{sample_idx}",
        fontsize=13, fontweight="bold", pad=8,
    )

    # ── time-series panel ─────────────────────────────────────────────────────
    t_ax = np.arange(horizon)
    ax_time.step(t_ax, gate_traj, where="post",
                 color="#94a3b8", lw=1.8, alpha=0.7, label="Gate $g_t$")
    ax_time.fill_between(t_ax, gate_traj - half_w, gate_traj + half_w,
                         step="post", color="#94a3b8", alpha=0.12)
    ax_time.axhline(0, color="#475569", lw=0.8, alpha=0.5)

    y_lines: dict[str, tuple] = {}
    for mode, lbl in variant_order:
        y_seq = trajs[mode][1:, 1]
        lw = 2.4 if mode == "context" else 1.6
        ln, = ax_time.plot([], [], color=colors[mode], lw=lw, label=lbl)
        y_lines[mode] = (ln, y_seq)

    vline = ax_time.axvline(0, color="#f8fafc", lw=1.2, alpha=0.7, linestyle="--")
    ax_time.set_xlim(0, horizon - 1)
    ax_time.set_ylim(-corr - 0.1, corr + 0.1)
    ax_time.set_xlabel("time step")
    ax_time.set_ylabel("y / gate center")
    ax_time.set_title("Lateral position over time", fontsize=11, fontweight="bold")
    ax_time.legend(loc="upper right", fontsize=8, facecolor="#1e293b",
                   edgecolor="#475569", labelcolor="#e2e8f0", framealpha=0.85)

    # ── update function (closed over per-sample data) ─────────────────────────
    def _update(frame: int):
        ti = frame + 1
        g  = float(gate_traj[frame])

        rect_upper.set_y(g + half_w)
        rect_upper.set_height(max(0.0, corr + 0.12 - (g + half_w)))
        rect_lower.set_y(-corr - 0.12)
        rect_lower.set_height(max(0.0, (g - half_w) + corr + 0.12))
        gate_patch.set_y(g - half_w)

        for mode, _ in variant_order:
            xy = trajs[mode][:ti + 1]
            trail_lines[mode].set_data(xy[:, 0], xy[:, 1])
            dots[mode].set_data([xy[-1, 0]], [xy[-1, 1]])

        step_text.set_text(f"step {frame:03d}/{horizon - 1:03d}")
        vline.set_xdata([frame, frame])
        for mode, (ln, y_seq) in y_lines.items():
            ln.set_data(t_ax[:frame + 1], y_seq[:frame + 1])

        return (rect_upper, rect_lower, gate_patch, step_text, vline,
                *trail_lines.values(), *dots.values(),
                *[ln for ln, _ in y_lines.values()])

    frames      = list(range(0, horizon, 2))
    interval_ms = 60
    anim = FuncAnimation(fig, _update, frames=frames,
                         interval=interval_ms, blit=True)

    gif_path = run_dir / f"rollout_animation_{file_tag}.gif"
    anim.save(str(gif_path), writer=PillowWriter(fps=1000 // interval_ms))
    print(f"  Animation saved → {gif_path}")

    try:
        from matplotlib.animation import FFMpegWriter
        mp4_path = run_dir / f"rollout_animation_{file_tag}.mp4"
        anim.save(str(mp4_path),
                  writer=FFMpegWriter(fps=1000 // interval_ms, bitrate=1200))
        print(f"  Animation saved → {mp4_path}")
    except Exception:
        pass

    if show_plots:
        plt.show()
    plt.close(fig)


def animate_rollout(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: ScenarioBatch,
    test_metrics: dict[str, dict],
    show_plots: bool,
    n_samples: int = 5,
) -> None:
    """Render per-sample animated GIFs for n_samples representative episodes."""
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    # collect indices: use select_trajectory_indices, pad with random if needed
    idxs = select_trajectory_indices(test_batch, args)
    batch_size = int(test_batch.gate_y.shape[0])
    if len(idxs) < n_samples:
        used = set(idxs)
        extras = [i for i in range(batch_size) if i not in used]
        idxs = list(idxs) + extras[: n_samples - len(idxs)]
    idxs = idxs[:n_samples]

    print(f"\nRendering {len(idxs)} rollout animations…")
    for rank, sample_idx in enumerate(idxs):
        print(f"  [{rank + 1}/{len(idxs)}] sample #{sample_idx}")
        _animate_one_sample(
            plt=plt,
            args=args,
            run_dir=run_dir,
            variant_order=variant_order,
            test_batch=test_batch,
            test_metrics=test_metrics,
            sample_idx=sample_idx,
            file_tag=f"{rank + 1:02d}_idx{sample_idx}",
            show_plots=show_plots,
        )


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def strip_rollout(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "rollout"}


def build_interpretation(test_metrics: dict[str, dict]) -> str:
    context = test_metrics["context"]
    dist = test_metrics["disturbance_only"]
    nominal = test_metrics["nominal"]
    return (
        f"Context-aware PB+SSM reaches success rate {context['success_rate']:.3f}, "
        f"versus {dist['success_rate']:.3f} for disturbance-only PB+SSM and "
        f"{nominal['success_rate']:.3f} for nominal pre-stabilization. "
        f"Success here requires both clearing the moving wall opening and ending near the origin."
    )


def main() -> None:
    args = parse_args()
    set_seeds(int(args.seed))
    torch.set_num_threads(max(1, torch.get_num_threads()))

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    expected_cross_index = estimate_expected_cross_index(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_id or f"controlled_xy_{timestamp}"
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = dict(vars(args))
    config_payload["context_dim"] = int(context_dim())
    config_payload["expected_cross_index"] = int(expected_cross_index)
    save_json(run_dir / "config.json", config_payload)

    val_batch = sample_batch(
        args=args,
        batch_size=int(args.val_batch),
        seed=int(args.seed) + 50_000,
        paired=True,
        shuffle=False,
        expected_cross_index=expected_cross_index,
    )
    test_batch = sample_batch(
        args=args,
        batch_size=int(args.test_batch),
        seed=int(args.seed) + 60_000,
        paired=True,
        shuffle=False,
        expected_cross_index=expected_cross_index,
    )

    specs = variant_specs()
    controllers: dict[str, PBController | None] = {}
    plants: dict[str, DoubleIntegratorTrue | None] = {}
    histories: dict[str, list[dict]] = {}
    val_metrics: dict[str, dict] = {}
    test_metrics: dict[str, dict] = {}

    print(f"Run directory: {run_dir}")
    print(f"Expected nominal wall crossing occurs near step {expected_cross_index}.")

    for mode, label in specs:
        print(f"\nTraining/evaluating {label}...")
        # Warm-start the context mode from the disturbance_only checkpoint
        warm_start = None
        if mode == "context" and controllers.get("disturbance_only") is not None:
            warm_start = {k: v.detach().cpu().clone() for k, v in controllers["disturbance_only"].state_dict().items()}
            print("[context] warm-starting from disturbance_only checkpoint.")
        controller, plant_true, history, best_val_metrics = train_controller(
            args=args,
            device=device,
            mode=mode,
            val_batch=val_batch,
            expected_cross_index=expected_cross_index,
            warm_start_state=warm_start,
        )
        controllers[mode] = controller
        plants[mode] = plant_true
        histories[mode] = history
        val_metrics[mode] = best_val_metrics
        test_metrics[mode] = evaluate_variant(
            args=args,
            batch=test_batch,
            device=device,
            mode=mode,
            controller=controller,
            plant_true=plant_true,
            expected_cross_index=expected_cross_index,
        )
        if controller is not None:
            torch.save(controller.state_dict(), run_dir / f"{mode}_controller.pt")

    show_plots = not args.no_show_plots
    plot_loss_curves(
        run_dir=run_dir,
        histories=histories,
        variant_order=specs,
        show_plots=show_plots,
    )
    plot_wall_style_summary(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    plot_control_magnitude(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    plot_trajectory_samples(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )

    animate_rollout(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )
    plot_waiting_behavior(
        args=args,
        run_dir=run_dir,
        test_batch=test_batch,
        test_metrics=test_metrics,
        show_plots=show_plots,
    )

    save_json(run_dir / "metrics.json", {mode: strip_rollout(test_metrics[mode]) for mode, _ in specs})
    save_json(run_dir / "val_metrics.json", {mode: strip_rollout(val_metrics[mode]) for mode, _ in specs})
    save_json(run_dir / "train_history.json", histories)

    interpretation = build_interpretation(test_metrics)
    (run_dir / "interpretation.txt").write_text(interpretation + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("RESULTS OVERVIEW")
    print("=" * 76)
    for mode, label in specs:
        metrics = test_metrics[mode]
        print(
            f"{label:30s} "
            f"success={metrics['success_rate']:.3f} "
            f"wall={metrics['wall_success_rate']:.3f} "
            f"goal={metrics['goal_success_rate']:.3f} "
            f"term={metrics['avg_terminal_dist']:.3f} "
            f"cross_err={metrics['avg_abs_cross_error']:.3f}"
        )
    print("=" * 76)
    print(interpretation)


if __name__ == "__main__":
    main()
