"""2D contextual gate PB+SSM experiment with controlled x/y motion toward the origin."""

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
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_batch", type=int, default=256)
    parser.add_argument("--val_batch", type=int, default=512)
    parser.add_argument("--test_batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--disturbance_only_epochs", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--no_show_plots", action="store_true")

    # Geometry and dynamics.
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--pre_kp", type=float, default=0.32)
    parser.add_argument("--pre_kd", type=float, default=0.55)
    parser.add_argument("--start_x_min", type=float, default=1.7)
    parser.add_argument("--start_x_max", type=float, default=2.1)
    parser.add_argument("--start_y_max", type=float, default=0.10)
    parser.add_argument("--wall_x", type=float, default=0.55)
    parser.add_argument("--gate_half_width", type=float, default=0.20)
    parser.add_argument("--gate_amplitude", type=float, default=0.95)
    parser.add_argument("--goal_tol", type=float, default=0.18)
    parser.add_argument("--corridor_limit", type=float, default=1.6)
    parser.add_argument("--wall_focus_sigma", type=float, default=0.14)
    parser.add_argument("--gate_settle_steps", type=int, default=4)
    parser.add_argument("--gate_settle_jitter", type=int, default=4)

    # Gate schedule.
    parser.add_argument("--gate_process", type=str, default="hazard", choices=["alternating", "hazard"])
    parser.add_argument("--gate_dwell_min", type=int, default=8)
    parser.add_argument("--gate_dwell_max", type=int, default=16)
    parser.add_argument("--gate_switch_prob", type=float, default=0.20)
    parser.add_argument("--context_ema_alpha", type=float, default=0.35)

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
    parser.add_argument("--ssm_layers", type=int, default=7)
    parser.add_argument("--ssm_d_model", type=int, default=32)
    parser.add_argument("--ssm_d_state", type=int, default=64)
    parser.add_argument("--ssm_ff", type=str, default="GLU")
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=True)
    parser.add_argument("--mp_context_lift_type", type=str, default="linear", choices=["identity", "linear", "mlp"])
    parser.add_argument("--mp_context_lift_dim", type=int, default=6)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=24)
    parser.add_argument("--mp_context_decay_law", type=str, default="finite", choices=["exp", "poly", "finite"])
    parser.add_argument("--mp_context_decay_rate", type=float, default=0.04)
    parser.add_argument("--mp_context_decay_power", type=float, default=0.75)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=90)
    parser.add_argument("--mp_context_lp_p", type=float, default=2.0)
    parser.add_argument("--mp_context_scale", type=float, default=0.25)

    # Loss.
    parser.add_argument("--goal_stage_weight", type=float, default=4.2)
    parser.add_argument("--goal_terminal_weight", type=float, default=30.0)
    parser.add_argument("--wall_track_weight", type=float, default=24.0)
    parser.add_argument("--wall_collision_weight", type=float, default=120.0)
    parser.add_argument("--post_wall_goal_weight", type=float, default=14.0)
    parser.add_argument("--post_wall_lateral_weight", type=float, default=8.0)
    parser.add_argument("--origin_overshoot_weight", type=float, default=8.0)
    parser.add_argument("--control_weight", type=float, default=0.05)
    parser.add_argument("--corridor_weight", type=float, default=10.0)
    parser.add_argument("--post_wall_sigma", type=float, default=0.08)
    parser.add_argument("--collision_sharpness", type=float, default=14.0)
    parser.add_argument("--corridor_sharpness", type=float, default=12.0)
    parser.add_argument("--overshoot_sharpness", type=float, default=12.0)
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
    return 9


def epochs_for_mode(args: argparse.Namespace, mode: str) -> int:
    if mode == "disturbance_only":
        return max(1, int(args.disturbance_only_epochs))
    return max(1, int(args.epochs))


def build_gate_features(gates: np.ndarray, ema_alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_velocity = np.zeros_like(gates, dtype=np.float32)
    if gates.shape[1] > 1:
        gate_velocity[:, 1:] = gates[:, 1:] - gates[:, :-1]
        gate_velocity[:, 0] = gate_velocity[:, 1]

    gate_ema = np.zeros_like(gates, dtype=np.float32)
    gate_ema[:, 0] = gates[:, 0]
    for t in range(1, gates.shape[1]):
        gate_ema[:, t] = ema_alpha * gates[:, t] + (1.0 - ema_alpha) * gate_ema[:, t - 1]

    switch_age = np.zeros_like(gates, dtype=np.float32)
    for b in range(gates.shape[0]):
        age = 0.0
        for t in range(1, gates.shape[1]):
            if abs(float(gates[b, t] - gates[b, t - 1])) > 1e-6:
                age = 0.0
            else:
                age += 1.0
            switch_age[b, t] = age
    return gate_velocity, gate_ema, switch_age


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
    sign = float(rng.choice([-1.0, 1.0]))
    process = str(args.gate_process)
    t = 0

    if process == "alternating":
        while t < freeze_step:
            remaining = freeze_step - t
            dwell_hi = min(int(args.gate_dwell_max), remaining)
            dwell_lo = min(int(args.gate_dwell_min), dwell_hi)
            dwell_lo = max(1, dwell_lo)
            dwell = int(rng.integers(dwell_lo, dwell_hi + 1))
            gate[t : t + dwell] = sign * float(args.gate_amplitude)
            sign *= -1.0
            t += dwell
    else:
        dwell = 0
        while t < freeze_step:
            gate[t] = sign * float(args.gate_amplitude)
            t += 1
            dwell += 1
            if t >= freeze_step:
                break

            reached_min = dwell >= int(args.gate_dwell_min)
            reached_max = dwell >= int(args.gate_dwell_max)
            should_switch = reached_max or (
                reached_min and rng.random() < float(args.gate_switch_prob)
            )
            if should_switch:
                sign *= -1.0
                dwell = 0

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
    base_freeze_step = max(1, expected_cross_index - int(args.gate_settle_steps))

    starts = []
    goals = []
    gates = []
    pair_ids = []

    for pair_idx in range(base_count):
        start_x = float(rng.uniform(float(args.start_x_min), float(args.start_x_max)))
        start_y = float(rng.uniform(-float(args.start_y_max), float(args.start_y_max)))
        settle_jitter = int(args.gate_settle_jitter)
        if settle_jitter > 0:
            jitter = int(rng.integers(-settle_jitter, settle_jitter + 1))
        else:
            jitter = 0
        freeze_step = int(np.clip(base_freeze_step + jitter, 1, int(args.horizon)))
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

    gate_v_np, gate_ema_np, switch_age_np = build_gate_features(gates_np, float(args.context_ema_alpha))

    return ScenarioBatch(
        start=torch.from_numpy(starts_np),
        goal=torch.from_numpy(goals_np),
        gate_y=torch.from_numpy(gates_np),
        gate_v=torch.from_numpy(gate_v_np),
        gate_ema=torch.from_numpy(gate_ema_np),
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
) -> torch.Tensor:
    state = as_bt(x_t)
    pos = state[..., :2]
    x_pos = pos[..., 0]
    y_pos = pos[..., 1]

    gate_t = batch.gate_y[:, t : t + 1]
    gate_v_t = batch.gate_v[:, t : t + 1]
    gate_ema_t = batch.gate_ema[:, t : t + 1]
    switch_age_t = batch.switch_age[:, t : t + 1]

    rel_wall_x = x_pos - float(args.wall_x)
    gate_error = y_pos - gate_t
    approach = torch.exp(-0.5 * (rel_wall_x / float(args.wall_focus_sigma)) ** 2)
    goal_dx = -x_pos
    goal_dy = -y_pos

    x_scale = max(float(args.start_x_max), abs(float(args.wall_x)), 1.0)
    y_scale = max(float(args.corridor_limit), abs(float(args.gate_amplitude)), 1.0)
    age_scale = max(float(args.horizon), 1.0)
    z_t = torch.cat(
        [
            gate_t / y_scale,
            gate_v_t / y_scale,
            gate_ema_t / y_scale,
            gate_error / y_scale,
            rel_wall_x / x_scale,
            goal_dx / x_scale,
            goal_dy / y_scale,
            approach,
            switch_age_t / age_scale,
        ],
        dim=-1,
    )
    return float(args.z_scale) * z_t.unsqueeze(1)


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

    mp = MpDeepSSM(
        mp_in_dim,
        feat_dim,
        mode="loop",
        param=args.ssm_param,
        n_layers=int(args.ssm_layers),
        d_model=int(args.ssm_d_model),
        d_state=int(args.ssm_d_state),
        ff=args.ssm_ff,
    ).to(device)
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
) -> RolloutArtifacts:
    batch = batch.to(device)
    x0 = make_x0(batch, device)
    z_dim = context_dim()

    if zero_context:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return torch.zeros(x_t.shape[0], 1, z_dim, device=x_t.device, dtype=x_t.dtype)
    else:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return build_context(args=args, batch=batch, x_t=x_t, t=t)

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
        )
    if mode == "context":
        return rollout_pb_variant(
            args=args,
            batch=batch,
            device=device,
            controller=controller,
            plant_true=plant_true,
            zero_context=False,
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
) -> tuple[torch.Tensor, dict[str, float]]:
    goal = batch.goal.to(rollout.x_seq.device).unsqueeze(1)
    pos = rollout.x_seq[..., :2]
    x_pos = pos[..., 0]
    y_pos = pos[..., 1]
    gate = batch.gate_y.to(rollout.x_seq.device)

    goal_dist = torch.norm(pos - goal, dim=-1)
    goal_stage = float(args.goal_stage_weight) * goal_dist.mean(dim=1)
    goal_term = float(args.goal_terminal_weight) * goal_dist[:, -1]
    post_wall_mask = torch.sigmoid(
        (float(args.wall_x) - x_pos) / max(float(args.post_wall_sigma), 1e-6)
    )
    post_wall_goal = float(args.post_wall_goal_weight) * (post_wall_mask * goal_dist).mean(dim=1)
    post_wall_lateral = float(args.post_wall_lateral_weight) * (post_wall_mask * y_pos.square()).mean(dim=1)

    w_wall = wall_weights(x_pos, float(args.wall_x), float(args.wall_focus_sigma))
    gate_error = y_pos - gate
    wall_track = float(args.wall_track_weight) * (w_wall * gate_error.square()).sum(dim=1)
    collision_soft = F.softplus(
        float(args.collision_sharpness) * (gate_error.abs() - float(args.gate_half_width))
    ) / float(args.collision_sharpness)
    wall_collision = float(args.wall_collision_weight) * (w_wall * collision_soft).sum(dim=1)

    control_mag_sq = torch.sum(rollout.u_seq.square(), dim=-1)
    control_cost = float(args.control_weight) * control_mag_sq.mean(dim=1)

    corridor_soft = F.softplus(
        float(args.corridor_sharpness) * (y_pos.abs() - float(args.corridor_limit))
    ) / float(args.corridor_sharpness)
    corridor_cost = float(args.corridor_weight) * corridor_soft.mean(dim=1)
    overshoot_soft = F.softplus(
        float(args.overshoot_sharpness) * (-x_pos)
    ) / float(args.overshoot_sharpness)
    origin_overshoot = float(args.origin_overshoot_weight) * (post_wall_mask * overshoot_soft).mean(dim=1)

    total_per = (
        goal_stage
        + goal_term
        + post_wall_goal
        + post_wall_lateral
        + wall_track
        + wall_collision
        + control_cost
        + corridor_cost
        + origin_overshoot
    )
    parts = {
        "loss_total": to_python_float(total_per.mean()),
        "loss_goal_stage": to_python_float(goal_stage.mean()),
        "loss_goal_term": to_python_float(goal_term.mean()),
        "loss_post_wall_goal": to_python_float(post_wall_goal.mean()),
        "loss_post_wall_lat": to_python_float(post_wall_lateral.mean()),
        "loss_wall_track": to_python_float(wall_track.mean()),
        "loss_wall_collision": to_python_float(wall_collision.mean()),
        "loss_control": to_python_float(control_cost.mean()),
        "loss_corridor": to_python_float(corridor_cost.mean()),
        "loss_origin_overshoot": to_python_float(origin_overshoot.mean()),
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
) -> dict:
    rollout = rollout_variant(
        args=args,
        batch=batch,
        device=device,
        mode=mode,
        controller=controller,
        plant_true=plant_true,
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
        f"post={record['loss_post_wall_goal']:.4f} "
        f"lat={record['loss_post_wall_lat']:.4f} "
        f"wall={record['loss_wall_track']:.4f} "
        f"coll={record['loss_wall_collision']:.4f} "
        f"ctrl={record['loss_control']:.4f} "
        f"corr={record['loss_corridor']:.4f} "
        f"over={record['loss_origin_overshoot']:.4f} "
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
) -> tuple[PBController | None, DoubleIntegratorTrue | None, list[dict], dict]:
    if mode == "nominal":
        metrics = evaluate_variant(
            args=args,
            batch=val_batch,
            device=device,
            mode=mode,
            controller=None,
            plant_true=None,
        )
        return None, None, [], metrics

    mode_epochs = epochs_for_mode(args, mode)
    controller, plant_true = build_controller(device, args)
    optimizer = torch.optim.Adam(controller.parameters(), lr=float(args.lr))
    scheduler = CosineAnnealingLR(optimizer, T_max=mode_epochs, eta_min=float(args.lr_min))

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
        rollout = rollout_variant(
            args=args,
            batch=train_batch,
            device=device,
            mode=mode,
            controller=controller,
            plant_true=plant_true,
        )
        loss, parts = compute_loss(args=args, batch=train_batch, rollout=rollout)

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
        controller, plant_true, history, best_val_metrics = train_controller(
            args=args,
            device=device,
            mode=mode,
            val_batch=val_batch,
            expected_cross_index=expected_cross_index,
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
