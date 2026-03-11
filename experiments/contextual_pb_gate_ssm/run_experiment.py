"""PB+SSM contextual gate experiment with causal enriched context."""

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
from pb_controller import PBController, as_bt
from pb_core import rollout_pb, validate_component_compatibility
from pb_core.factories import build_factorized_controller
from ssm_operators import MpDeepSSM


@dataclass
class EpisodeBatch:
    gate: torch.Tensor
    disturbance: torch.Tensor
    start_y: torch.Tensor
    gate_velocity: torch.Tensor
    gate_ema: torch.Tensor
    switch_age: torch.Tensor
    pair_id: torch.Tensor

    def to(self, device: torch.device) -> "EpisodeBatch":
        return EpisodeBatch(
            gate=self.gate.to(device),
            disturbance=self.disturbance.to(device),
            start_y=self.start_y.to(device),
            gate_velocity=self.gate_velocity.to(device),
            gate_ema=self.gate_ema.to(device),
            switch_age=self.switch_age.to(device),
            pair_id=self.pair_id.to(device),
        )


@dataclass
class RolloutArtifacts:
    y: torch.Tensor
    u_nom: torch.Tensor
    v_pb: torch.Tensor
    u_total: torch.Tensor
    w_rec: torch.Tensor


class LateralIntegratorNominal:
    """Scalar nominal plant used by PB for disturbance reconstruction."""

    def nominal_dynamics(self, x: torch.Tensor, u: torch.Tensor, t: int | None = None) -> torch.Tensor:
        x = as_bt(x)
        u = as_bt(u)
        return x + u


class LateralIntegratorTrue:
    """True plant without internal mismatch; gusts are added as process noise."""

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: int | None = None) -> torch.Tensor:
        x = as_bt(x)
        u = as_bt(u)
        return x + u


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Contextual PB gate experiment with PBController + SSM")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_batch", type=int, default=256)
    parser.add_argument("--val_batch", type=int, default=512)
    parser.add_argument("--test_batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--no_show_plots", action="store_true")

    # Dynamics.
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--x0", type=float, default=-2.5)
    parser.add_argument("--vx", type=float, default=0.05)
    parser.add_argument("--x_gate", type=float, default=0.0)
    parser.add_argument("--gate_half_width", type=float, default=0.2)
    parser.add_argument("--gate_amplitude", type=float, default=1.0)
    parser.add_argument("--nominal_gain", type=float, default=0.12)
    parser.add_argument("--start_y_max", type=float, default=0.08)
    parser.add_argument("--corridor_limit", type=float, default=1.7)

    # Gate schedule.
    parser.add_argument("--gate_dwell_min", type=int, default=10)
    parser.add_argument("--gate_dwell_max", type=int, default=22)
    parser.add_argument("--gate_settle_steps", type=int, default=10)
    parser.add_argument("--context_ema_alpha", type=float, default=0.35)

    # Sparse gusts.
    parser.add_argument("--gust_count_min", type=int, default=2)
    parser.add_argument("--gust_count_max", type=int, default=5)
    parser.add_argument("--gust_duration_min", type=int, default=3)
    parser.add_argument("--gust_duration_max", type=int, default=8)
    parser.add_argument("--gust_amp_min", type=float, default=0.04)
    parser.add_argument("--gust_amp_max", type=float, default=0.12)
    parser.add_argument("--gust_clip", type=float, default=0.18)

    # PB architecture.
    parser.add_argument("--feat_dim", type=int, default=12)
    parser.add_argument("--mb_hidden", type=int, default=64)
    parser.add_argument("--mb_layers", type=int, default=4)
    parser.add_argument("--z_scale", type=float, default=6.0)
    parser.add_argument("--z_residual_gain", type=float, default=8.0)
    parser.add_argument("--mb_bound", type=float, default=8.0)
    parser.add_argument("--ssm_param", type=str, default="lru", choices=["lru", "tv"])
    parser.add_argument("--ssm_layers", type=int, default=4)
    parser.add_argument("--ssm_d_model", type=int, default=16)
    parser.add_argument("--ssm_d_state", type=int, default=32)
    parser.add_argument("--ssm_ff", type=str, default="GLU")
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=True)
    parser.add_argument("--mp_context_lift_type", type=str, default="linear", choices=["identity", "linear", "mlp"])
    parser.add_argument("--mp_context_lift_dim", type=int, default=4)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=16)
    parser.add_argument("--mp_context_decay_law", type=str, default="poly", choices=["exp", "poly", "finite"])
    parser.add_argument("--mp_context_decay_rate", type=float, default=0.05)
    parser.add_argument("--mp_context_decay_power", type=float, default=0.55)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=120)
    parser.add_argument("--mp_context_lp_p", type=float, default=2.0)
    parser.add_argument("--mp_context_scale", type=float, default=0.35)

    # Loss shaping.
    parser.add_argument("--track_weight", type=float, default=4.0)
    parser.add_argument("--miss_weight", type=float, default=18.0)
    parser.add_argument("--collision_weight", type=float, default=120.0)
    parser.add_argument("--control_weight", type=float, default=0.08)
    parser.add_argument("--corridor_weight", type=float, default=1.5)
    parser.add_argument("--collision_sharpness", type=float, default=12.0)
    parser.add_argument("--corridor_sharpness", type=float, default=10.0)
    parser.add_argument("--approach_window", type=float, default=1.0)
    parser.add_argument("--post_window", type=float, default=0.18)
    parser.add_argument("--window_sharpness", type=float, default=0.08)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_python_float(value) -> float:
    return float(value.item() if torch.is_tensor(value) else value)


def variant_specs() -> list[tuple[str, str]]:
    return [
        ("nominal", "Nominal only"),
        ("disturbance_only", "PB+SSM: disturbance only"),
        ("context", "PB+SSM: causal enriched context"),
    ]


def context_dim() -> int:
    return 8


def build_x_positions(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return args.x0 + args.vx * torch.arange(args.horizon + 1, device=device, dtype=torch.float32)


def compute_cross_index(args: argparse.Namespace) -> int:
    cross_step = round((args.x_gate - args.x0) / args.vx)
    if not np.isclose(args.x0 + args.vx * cross_step, args.x_gate, atol=1e-7):
        raise ValueError("Choose x0, x_gate, and vx so the wall is reached exactly on a grid step.")
    if not (0 < cross_step < args.horizon):
        raise ValueError("The wall crossing step must lie inside the rollout horizon.")
    return int(cross_step)


def compute_freeze_step(args: argparse.Namespace, cross_index: int) -> int:
    freeze_step = cross_index - int(args.gate_settle_steps)
    if freeze_step < 1:
        raise ValueError(
            "gate_settle_steps is too large for the chosen wall crossing time. "
            f"Got gate_settle_steps={args.gate_settle_steps}, crossing step={cross_index}."
        )
    return int(freeze_step)


def make_track_window(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    x = build_x_positions(args, device)[:-1]
    pre = torch.sigmoid((x - (args.x_gate - args.approach_window)) / args.window_sharpness)
    post = torch.sigmoid(((args.x_gate + args.post_window) - x) / args.window_sharpness)
    window = pre * post
    return window / window.max().clamp_min(1e-6)


def sample_switching_gate(
    args: argparse.Namespace,
    rng: np.random.Generator,
    *,
    cross_index: int,
) -> np.ndarray:
    gate = np.zeros(args.horizon, dtype=np.float32)
    freeze_step = compute_freeze_step(args, cross_index)
    sign = rng.choice([-1.0, 1.0])
    t = 0
    while t < freeze_step:
        remaining = freeze_step - t
        dwell_hi = min(int(args.gate_dwell_max), remaining)
        dwell_lo = min(int(args.gate_dwell_min), dwell_hi)
        dwell_lo = max(1, dwell_lo)
        dwell = int(rng.integers(dwell_lo, dwell_hi + 1))
        gate[t : t + dwell] = sign * args.gate_amplitude
        sign *= -1.0
        t += dwell
    gate[freeze_step:] = gate[freeze_step - 1]
    return gate


def sample_sparse_gusts(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    disturbance = np.zeros(args.horizon, dtype=np.float32)
    burst_count = int(rng.integers(args.gust_count_min, args.gust_count_max + 1))
    for _ in range(burst_count):
        duration = int(rng.integers(args.gust_duration_min, args.gust_duration_max + 1))
        start = int(rng.integers(0, max(1, args.horizon - duration + 1)))
        amplitude = float(rng.uniform(args.gust_amp_min, args.gust_amp_max))
        amplitude *= float(rng.choice([-1.0, 1.0]))
        disturbance[start : start + duration] += amplitude
    return np.clip(disturbance, -args.gust_clip, args.gust_clip)


def build_gate_features(
    gates: np.ndarray,
    *,
    ema_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < ema_alpha <= 1.0):
        raise ValueError(f"context_ema_alpha must lie in (0, 1], got {ema_alpha}")

    gate_velocity = np.zeros_like(gates, dtype=np.float32)
    if gates.shape[1] > 1:
        gate_velocity[:, 1:] = gates[:, 1:] - gates[:, :-1]

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


def sample_episode_batch(
    *,
    args: argparse.Namespace,
    batch_size: int,
    seed: int,
    paired: bool,
    shuffle: bool,
    cross_index: int,
) -> EpisodeBatch:
    if paired and batch_size % 2 != 0:
        raise ValueError("Paired batches require an even batch size.")

    rng = np.random.default_rng(seed)
    base_count = batch_size // 2 if paired else batch_size
    gates = []
    disturbances = []
    start_ys = []
    pair_ids = []

    for pair_idx in range(base_count):
        gate = sample_switching_gate(args, rng, cross_index=cross_index)
        disturbance = sample_sparse_gusts(args, rng)
        start_y = float(rng.uniform(-args.start_y_max, args.start_y_max))

        if paired:
            gates.extend([gate, -gate])
            disturbances.extend([disturbance, disturbance.copy()])
            start_ys.extend([start_y, start_y])
            pair_ids.extend([pair_idx, pair_idx])
        else:
            gates.append(gate)
            disturbances.append(disturbance)
            start_ys.append(start_y)
            pair_ids.append(pair_idx)

    gates_np = np.stack(gates, axis=0).astype(np.float32)
    disturbances_np = np.stack(disturbances, axis=0).astype(np.float32)
    start_ys_np = np.asarray(start_ys, dtype=np.float32)
    pair_ids_np = np.asarray(pair_ids, dtype=np.int64)

    if shuffle:
        order = rng.permutation(batch_size)
        gates_np = gates_np[order]
        disturbances_np = disturbances_np[order]
        start_ys_np = start_ys_np[order]
        pair_ids_np = pair_ids_np[order]

    gate_velocity_np, gate_ema_np, switch_age_np = build_gate_features(
        gates_np,
        ema_alpha=float(args.context_ema_alpha),
    )

    return EpisodeBatch(
        gate=torch.from_numpy(gates_np),
        disturbance=torch.from_numpy(disturbances_np),
        start_y=torch.from_numpy(start_ys_np),
        gate_velocity=torch.from_numpy(gate_velocity_np),
        gate_ema=torch.from_numpy(gate_ema_np),
        switch_age=torch.from_numpy(switch_age_np),
        pair_id=torch.from_numpy(pair_ids_np),
    )


def build_context_signal(
    *,
    args: argparse.Namespace,
    batch: EpisodeBatch,
    x_t: torch.Tensor,
    t: int,
    cross_index: int,
    track_window: torch.Tensor,
) -> torch.Tensor:
    y_t = as_bt(x_t)[..., 0]
    gate_t = batch.gate[:, t : t + 1]
    gate_vel_t = batch.gate_velocity[:, t : t + 1]
    gate_ema_t = batch.gate_ema[:, t : t + 1]
    switch_age_t = batch.switch_age[:, t : t + 1]

    gate_error_t = gate_t - y_t
    signed_margin_t = float(args.gate_half_width) - gate_error_t.abs()
    time_to_gate_t = torch.full_like(gate_t, max(cross_index - t, 0) / max(cross_index, 1))
    approach_t = torch.full_like(gate_t, float(track_window[t].item()))

    amp_scale = max(abs(float(args.gate_amplitude)), 1e-6)
    err_scale = max(float(args.corridor_limit), abs(float(args.gate_amplitude)), 1.0)
    switch_scale = max(float(cross_index), 1.0)
    margin_scale = max(float(args.gate_half_width), 1e-6)

    z_t = torch.cat(
        [
            gate_t / amp_scale,
            gate_vel_t / amp_scale,
            gate_ema_t / amp_scale,
            gate_error_t / err_scale,
            signed_margin_t / margin_scale,
            switch_age_t / switch_scale,
            time_to_gate_t,
            approach_t,
        ],
        dim=-1,
    )
    return float(args.z_scale) * z_t.unsqueeze(1)


def make_nominal_policy(gain: float):
    def u_nominal(x: torch.Tensor, t: int | None = None) -> torch.Tensor:
        x = as_bt(x)
        return -float(gain) * x

    return u_nominal


def build_controller(device: torch.device, args: argparse.Namespace) -> PBController:
    z_dim = context_dim()
    feat_dim = int(args.feat_dim)
    mp_context_lifter = None
    mp_in_dim = 1

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
        mp_in_dim = 1 + int(args.mp_context_lift_dim)

    # Reconstructed disturbance remains the primary signal, but M_p also receives
    # a bounded l_p-filtered lift of the causal context.
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
        w_dim=1,
        z_dim=z_dim,
        r=1,
        s=feat_dim,
        hidden_dim=int(args.mb_hidden),
        num_layers=int(args.mb_layers),
        use_z_residual=True,
        z_residual_gain=float(args.z_residual_gain),
        bound_mode="softsign",
        clamp_value=float(args.mb_bound),
    ).to(device)
    controller = build_factorized_controller(
        nominal_plant=LateralIntegratorNominal(),
        mp=mp,
        mb=mb,
        u_dim=1,
        detach_state=False,
        u_nominal=make_nominal_policy(float(args.nominal_gain)),
        mp_context_lifter=mp_context_lifter,
    ).to(device)

    probe_x = torch.zeros(4, 1, 1, device=device)
    probe_z = torch.zeros(4, 1, z_dim, device=device)
    ok, msg = validate_component_compatibility(
        controller=controller,
        plant_true=LateralIntegratorTrue(),
        x0=probe_x,
        z0=probe_z,
        raise_on_error=False,
    )
    if not ok:
        raise RuntimeError(f"PB component compatibility check failed: {msg}")

    return controller


def rollout_nominal(args: argparse.Namespace, batch: EpisodeBatch, device: torch.device) -> RolloutArtifacts:
    batch = batch.to(device)
    disturbance = batch.disturbance
    y_t = batch.start_y.unsqueeze(1)

    y_steps = [y_t]
    u_nom_steps = []
    v_steps = []
    u_total_steps = []

    for t in range(args.horizon):
        u_nom_t = -float(args.nominal_gain) * y_t
        v_t = torch.zeros_like(y_t)
        u_total_t = u_nom_t + v_t
        y_t = y_t + u_total_t + disturbance[:, t : t + 1]

        y_steps.append(y_t)
        u_nom_steps.append(u_nom_t)
        v_steps.append(v_t)
        u_total_steps.append(u_total_t)

    if args.horizon > 1:
        w_rec = torch.cat([batch.start_y.unsqueeze(1), disturbance[:, :-1]], dim=1)
    else:
        w_rec = batch.start_y.unsqueeze(1)

    return RolloutArtifacts(
        y=torch.cat(y_steps, dim=1),
        u_nom=torch.cat(u_nom_steps, dim=1),
        v_pb=torch.cat(v_steps, dim=1),
        u_total=torch.cat(u_total_steps, dim=1),
        w_rec=w_rec,
    )


def rollout_pb_variant(
    *,
    args: argparse.Namespace,
    batch: EpisodeBatch,
    device: torch.device,
    controller: PBController,
    zero_context: bool,
    cross_index: int,
    track_window: torch.Tensor,
) -> RolloutArtifacts:
    batch = batch.to(device)
    x0 = batch.start_y.view(-1, 1, 1)
    process_noise_seq = batch.disturbance.unsqueeze(-1)
    track_window = track_window.to(device)
    z_dim = context_dim()

    if zero_context:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return torch.zeros(x_t.shape[0], 1, z_dim, device=x_t.device, dtype=x_t.dtype)
    else:
        def context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return build_context_signal(
                args=args,
                batch=batch,
                x_t=x_t,
                t=t,
                cross_index=cross_index,
                track_window=track_window,
            )

    result = rollout_pb(
        controller=controller,
        plant_true=LateralIntegratorTrue(),
        x0=x0,
        horizon=int(args.horizon),
        context_fn=context_fn,
        w0=x0,
        process_noise_seq=process_noise_seq,
    )

    y_full = torch.cat([x0[..., 0], result.x_seq[..., 0]], dim=1)
    u_total = result.u_seq[..., 0]
    u_nom = -float(args.nominal_gain) * y_full[:, :-1]
    v_pb = u_total - u_nom

    return RolloutArtifacts(
        y=y_full,
        u_nom=u_nom,
        v_pb=v_pb,
        u_total=u_total,
        w_rec=result.w_seq[..., 0],
    )


def rollout_variant(
    *,
    args: argparse.Namespace,
    batch: EpisodeBatch,
    device: torch.device,
    mode: str,
    controller: PBController | None,
    cross_index: int,
    track_window: torch.Tensor,
) -> RolloutArtifacts:
    if mode == "nominal":
        return rollout_nominal(args, batch, device)
    if controller is None:
        raise ValueError(f"Controller is required for mode {mode}")
    if mode == "disturbance_only":
        return rollout_pb_variant(
            args=args,
            batch=batch,
            device=device,
            controller=controller,
            zero_context=True,
            cross_index=cross_index,
            track_window=track_window,
        )
    if mode == "context":
        return rollout_pb_variant(
            args=args,
            batch=batch,
            device=device,
            controller=controller,
            zero_context=False,
            cross_index=cross_index,
            track_window=track_window,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def compute_cost(
    *,
    args: argparse.Namespace,
    batch: EpisodeBatch,
    rollout: RolloutArtifacts,
    track_window: torch.Tensor,
    cross_index: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    gate = batch.gate.to(rollout.y.device)
    y_pre = rollout.y[:, :-1]
    cross_error = rollout.y[:, cross_index] - gate[:, cross_index]
    track_error = y_pre - gate

    track_cost = float(args.track_weight) * (track_window.view(1, -1) * track_error.square()).mean(dim=1)
    miss_cost = float(args.miss_weight) * cross_error.square()
    collision_soft = F.softplus(
        float(args.collision_sharpness) * (cross_error.abs() - float(args.gate_half_width))
    ) / float(args.collision_sharpness)
    collision_cost = float(args.collision_weight) * collision_soft
    control_cost = float(args.control_weight) * rollout.u_total.square().mean(dim=1)
    corridor_soft = F.softplus(
        float(args.corridor_sharpness) * (rollout.y.abs() - float(args.corridor_limit))
    ) / float(args.corridor_sharpness)
    corridor_cost = float(args.corridor_weight) * corridor_soft.mean(dim=1)

    total_per_sample = track_cost + miss_cost + collision_cost + control_cost + corridor_cost
    parts = {
        "loss_total": to_python_float(total_per_sample.mean()),
        "loss_track": to_python_float(track_cost.mean()),
        "loss_miss": to_python_float(miss_cost.mean()),
        "loss_collision": to_python_float(collision_cost.mean()),
        "loss_control": to_python_float(control_cost.mean()),
        "loss_corridor": to_python_float(corridor_cost.mean()),
    }
    return total_per_sample.mean(), parts


@torch.no_grad()
def evaluate_variant(
    *,
    args: argparse.Namespace,
    batch: EpisodeBatch,
    device: torch.device,
    mode: str,
    controller: PBController | None,
    track_window: torch.Tensor,
    cross_index: int,
) -> dict:
    rollout = rollout_variant(
        args=args,
        batch=batch,
        device=device,
        mode=mode,
        controller=controller,
        cross_index=cross_index,
        track_window=track_window,
    )
    avg_cost, loss_parts = compute_cost(
        args=args,
        batch=batch,
        rollout=rollout,
        track_window=track_window.to(rollout.y.device),
        cross_index=cross_index,
    )

    gate = batch.gate.to(rollout.y.device)
    cross_error = rollout.y[:, cross_index] - gate[:, cross_index]
    collisions = cross_error.abs() > float(args.gate_half_width)
    success = ~collisions
    track_error = rollout.y[:, :-1] - gate

    metrics = {
        "avg_cost": to_python_float(avg_cost),
        "success_rate": to_python_float(success.float().mean()),
        "collision_rate": to_python_float(collisions.float().mean()),
        "avg_abs_cross_error": to_python_float(cross_error.abs().mean()),
        "avg_control_energy": to_python_float(rollout.u_total.square().mean()),
        "avg_pb_energy": to_python_float(rollout.v_pb.square().mean()),
        "avg_window_track_mse": to_python_float((track_window.view(1, -1) * track_error.square()).mean()),
        "avg_abs_reconstructed_w": to_python_float(rollout.w_rec.abs().mean()),
        "gate_at_cross_mean": to_python_float(gate[:, cross_index].mean()),
        "disturbance_at_cross_mean": to_python_float(batch.disturbance[:, cross_index].mean()),
    }
    metrics.update(loss_parts)
    metrics["rollout"] = {
        "y": rollout.y.detach().cpu(),
        "u_nom": rollout.u_nom.detach().cpu(),
        "v_pb": rollout.v_pb.detach().cpu(),
        "u_total": rollout.u_total.detach().cpu(),
        "w_rec": rollout.w_rec.detach().cpu(),
    }
    return metrics


def choose_better_result(candidate: tuple[float, float], incumbent: tuple[float, float] | None) -> bool:
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
        f"track={record['loss_track']:.4f} "
        f"miss={record['loss_miss']:.4f} "
        f"coll={record['loss_collision']:.4f} "
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
        f"success={val_metrics['success_rate']:.3f} "
        f"collision={val_metrics['collision_rate']:.3f} "
        f"cross_err={val_metrics['avg_abs_cross_error']:.3f}"
        f"{best_tag}"
    )
    print(val_msg)


def train_controller(
    *,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    val_batch: EpisodeBatch,
    track_window: torch.Tensor,
    cross_index: int,
) -> tuple[PBController | None, list[dict], dict]:
    if mode == "nominal":
        metrics = evaluate_variant(
            args=args,
            batch=val_batch,
            device=device,
            mode=mode,
            controller=None,
            track_window=track_window,
            cross_index=cross_index,
        )
        return None, [], metrics

    controller = build_controller(device, args)
    optimizer = torch.optim.Adam(controller.parameters(), lr=float(args.lr))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=float(args.lr_min))

    history: list[dict] = []
    best_state = None
    best_score = None
    best_val_metrics = None

    for epoch in range(1, args.epochs + 1):
        controller.train()
        train_batch = sample_episode_batch(
            args=args,
            batch_size=int(args.train_batch),
            seed=int(args.seed) + 1000 + epoch,
            paired=True,
            shuffle=True,
            cross_index=cross_index,
        )
        rollout = rollout_variant(
            args=args,
            batch=train_batch,
            device=device,
            mode=mode,
            controller=controller,
            cross_index=cross_index,
            track_window=track_window,
        )
        loss, parts = compute_cost(
            args=args,
            batch=train_batch,
            rollout=rollout,
            track_window=track_window.to(device),
            cross_index=cross_index,
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
        }
        record.update(parts)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            controller.eval()
            val_metrics = evaluate_variant(
                args=args,
                batch=val_batch,
                device=device,
                mode=mode,
                controller=controller,
                track_window=track_window,
                cross_index=cross_index,
            )
            record["val_cost"] = float(val_metrics["avg_cost"])
            record["val_success_rate"] = float(val_metrics["success_rate"])
            record["val_collision_rate"] = float(val_metrics["collision_rate"])

            candidate_score = (
                1.0 - float(val_metrics["success_rate"]),
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
                epochs=int(args.epochs),
                record=record,
                val_metrics=val_metrics,
                is_best=is_best,
            )
        else:
            print_train_epoch_status(
                mode=mode,
                epoch=epoch,
                epochs=int(args.epochs),
                record=record,
            )
        history.append(record)

    if best_state is None:
        raise RuntimeError(f"Training for mode {mode} did not produce any validation checkpoint.")

    controller.load_state_dict(best_state)
    return controller, history, best_val_metrics


def setup_plot_style(plt) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
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


def gate_colors() -> tuple[str, str]:
    return "#b91c1c", "#1d4ed8"


def select_representative_indices(
    batch: EpisodeBatch,
    args: argparse.Namespace,
    cross_index: int,
) -> list[tuple[int, str]]:
    gate = batch.gate.numpy()
    disturbance = batch.disturbance.numpy()
    cross_gate = gate[:, cross_index]
    lo = max(0, cross_index - 12)
    hi = min(args.horizon, cross_index + 4)
    disturbance_strength = np.mean(np.abs(disturbance[:, lo:hi]), axis=1)
    switch_strength = np.sum(np.abs(np.diff(gate[:, lo:hi], axis=1)) > 0, axis=1)

    upper_candidates = np.where(cross_gate > 0)[0]
    lower_candidates = np.where(cross_gate < 0)[0]
    upper_idx = int(upper_candidates[np.argmax(disturbance_strength[upper_candidates])]) if upper_candidates.size else 0
    lower_idx = int(lower_candidates[np.argmax(disturbance_strength[lower_candidates])]) if lower_candidates.size else upper_idx

    excluded = {upper_idx, lower_idx}
    remaining = [idx for idx in range(gate.shape[0]) if idx not in excluded]
    if remaining:
        remaining_scores = disturbance_strength[remaining] + 0.7 * switch_strength[remaining]
        switch_idx = int(remaining[int(np.argmax(remaining_scores))])
    else:
        switch_idx = upper_idx

    return [
        (upper_idx, "Upper gate at crossing"),
        (lower_idx, "Lower gate at crossing"),
        (switch_idx, "High-disturbance case"),
    ]


def select_comparison_index(batch: EpisodeBatch, args: argparse.Namespace, cross_index: int) -> int:
    gate = batch.gate.numpy()
    disturbance = batch.disturbance.numpy()
    lo = max(0, cross_index - 12)
    hi = min(args.horizon, cross_index + 6)
    switch_strength = np.sum(np.abs(np.diff(gate[:, lo:hi], axis=1)) > 0, axis=1)
    disturbance_strength = np.mean(np.abs(disturbance[:, lo:hi]), axis=1)
    score = disturbance_strength + 0.6 * switch_strength
    return int(np.argmax(score))


def select_counterfactual_pair(
    batch: EpisodeBatch,
    args: argparse.Namespace,
    cross_index: int,
) -> tuple[int, int]:
    disturbance = batch.disturbance.numpy()
    pair_ids = batch.pair_id.numpy()
    gate = batch.gate.numpy()
    lo = max(0, cross_index - 12)
    hi = min(args.horizon, cross_index + 4)

    best_pair = None
    best_score = None
    for pair_id in np.unique(pair_ids):
        idx = np.where(pair_ids == pair_id)[0]
        if idx.size < 2:
            continue
        score = float(np.mean(np.abs(disturbance[idx[0], lo:hi])))
        if best_score is None or score > best_score:
            best_score = score
            best_pair = idx

    if best_pair is None:
        return 0, 1 if disturbance.shape[0] > 1 else 0

    idx = np.asarray(best_pair, dtype=np.int64)
    if gate[idx[0], cross_index] > gate[idx[1], cross_index]:
        return int(idx[0]), int(idx[1])
    return int(idx[1]), int(idx[0])


def draw_wall(
    ax,
    x_gate: float,
    gate_center: float,
    half_width: float,
    corridor_limit: float,
    color: str,
    alpha: float,
) -> None:
    ax.plot([x_gate, x_gate], [-corridor_limit, gate_center - half_width], color=color, lw=3.0, alpha=alpha)
    ax.plot([x_gate, x_gate], [gate_center + half_width, corridor_limit], color=color, lw=3.0, alpha=alpha)


def make_wall_demo_batch(args: argparse.Namespace) -> EpisodeBatch:
    gates = np.zeros((2, args.horizon), dtype=np.float32)
    disturbances = np.zeros((2, args.horizon), dtype=np.float32)
    start_y = np.zeros(2, dtype=np.float32)
    pair_id = np.zeros(2, dtype=np.int64)

    pulse_start = min(5, max(args.horizon - 1, 0))
    pulse_end = min(max(pulse_start + 1, 12), args.horizon)
    demo_amp = min(max(0.14, float(args.gust_amp_max)), float(args.gust_clip))
    disturbances[:, pulse_start:pulse_end] = demo_amp

    gates[0, :] = float(args.gate_amplitude)
    gates[1, :] = -float(args.gate_amplitude)

    gate_velocity, gate_ema, switch_age = build_gate_features(
        gates,
        ema_alpha=float(args.context_ema_alpha),
    )
    return EpisodeBatch(
        gate=torch.from_numpy(gates),
        disturbance=torch.from_numpy(disturbances),
        start_y=torch.from_numpy(start_y),
        gate_velocity=torch.from_numpy(gate_velocity),
        gate_ema=torch.from_numpy(gate_ema),
        switch_age=torch.from_numpy(switch_age),
        pair_id=torch.from_numpy(pair_id),
    )


def wall_plot_labels() -> dict[str, str]:
    return {
        "nominal": "Nominal Only",
        "disturbance_only": "PB (Disturbance Only)",
        "context": "PB (Context-Aware)",
    }


def wall_plot_colors() -> dict[str, str]:
    return {
        "nominal": "#1f77b4",
        "disturbance_only": "#ff7f0e",
        "context": "#2ca02c",
    }


def plot_wall_style_summary(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: EpisodeBatch,
    test_metrics: dict[str, dict],
    controllers: dict[str, PBController | None],
    device: torch.device,
    cross_index: int,
    track_window: torch.Tensor,
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    labels_map = wall_plot_labels()
    colors = wall_plot_colors()
    modes = [mode for mode, _ in variant_order]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.2])

    ax1 = fig.add_subplot(gs[0, :])
    test_idx = select_comparison_index(test_batch, args, cross_index)
    x_full = build_x_positions(args, torch.device("cpu")).numpy()
    x_gate = x_full[:-1]
    g_traj = test_batch.gate[test_idx].numpy()

    ax1.step(
        x_gate,
        g_traj,
        where="post",
        color="grey",
        linestyle="--",
        alpha=0.6,
        linewidth=2,
        label="Safe Path $g_t$",
    )
    ax1.fill_between(
        x_gate,
        g_traj - float(args.gate_half_width),
        g_traj + float(args.gate_half_width),
        step="post",
        color="grey",
        alpha=0.12,
    )

    gate_center = float(g_traj[cross_index])
    ax1.axvline(float(args.x_gate), color="black", linewidth=4, label="Physical Wall")
    ax1.plot(
        [args.x_gate, args.x_gate],
        [gate_center - float(args.gate_half_width), gate_center + float(args.gate_half_width)],
        color="white",
        linewidth=6,
    )

    for mode in modes:
        y = test_metrics[mode]["rollout"]["y"][test_idx].numpy()
        lw = 2.5 if mode == "context" else 2.0
        ax1.plot(x_full, y, label=labels_map[mode], linewidth=lw, color=colors[mode])

    ax1.set_title("Top-Down View: Robot Navigating the Corridor", fontsize=14, fontweight="bold")
    ax1.set_xlabel("x position", fontsize=12)
    ax1.set_ylabel("y position", fontsize=12)
    ax1.set_xlim([float(args.x0), float(args.x_gate) + 1.0])
    ax1.set_ylim([-float(args.corridor_limit), float(args.corridor_limit)])
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    labels = [labels_map[mode] for mode in modes]
    success_rates = [100.0 * float(test_metrics[mode]["success_rate"]) for mode in modes]
    bar_colors = [colors[mode] for mode in modes]

    ax2.bar(labels, success_rates, color=bar_colors, alpha=0.8)
    ax2.set_title("Gate Crossing Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim([0.0, 105.0])
    for idx, value in enumerate(success_rates):
        ax2.text(idx, value + 2.0, f"{value:.1f}%", ha="center", fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    avg_misses = [float(test_metrics[mode]["avg_abs_cross_error"]) for mode in modes]
    ax3.bar(labels, avg_misses, color=bar_colors, alpha=0.8)
    ax3.set_title(
        f"Avg |y - g_t| at Wall (Safe threshold < {float(args.gate_half_width):.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.axhline(float(args.gate_half_width), color="red", linestyle="--", label="Safe Bound")
    ax3.legend(loc="upper right")
    y_offset = max(0.03, 0.05 * max(avg_misses + [float(args.gate_half_width)]))
    for idx, value in enumerate(avg_misses):
        ax3.text(idx, value + y_offset, f"{value:.2f}", ha="center", fontweight="bold")

    ax4 = fig.add_subplot(gs[2, :])
    demo_batch = make_wall_demo_batch(args)
    demo_rollout = rollout_variant(
        args=args,
        batch=demo_batch,
        device=device,
        mode="context",
        controller=controllers["context"],
        cross_index=cross_index,
        track_window=track_window,
    )
    t_len = min(25, args.horizon)
    time_axis = np.arange(t_len)
    w_demo = demo_batch.disturbance[0, :t_len].numpy()
    v_demo = demo_rollout.v_pb[:, :t_len].detach().cpu().numpy()

    ax4.plot(
        time_axis,
        w_demo,
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Disturbance (Upward Gust)",
    )
    ax4.plot(
        time_axis,
        v_demo[0],
        color="blue",
        marker="o",
        label=f"PB Correction (Gate is UP at +{float(args.gate_amplitude):.1f})",
    )
    ax4.plot(
        time_axis,
        v_demo[1],
        color="red",
        marker="s",
        label=f"PB Correction (Gate is DOWN at -{float(args.gate_amplitude):.1f})",
    )
    ax4.set_title("Contextual Adaptation: How PB Reacts to the Same Upward Gust", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Time step", fontsize=12)
    ax4.set_ylabel("Value / PB Action ($v_t$)", fontsize=12)
    ax4.axhline(0.0, color="black", linewidth=1)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(run_dir / "wall_style_summary.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_representative_trajectories(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: EpisodeBatch,
    test_metrics: dict[str, dict],
    cross_index: int,
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    episode_specs = select_representative_indices(test_batch, args, cross_index)
    x = build_x_positions(args, torch.device("cpu")).numpy()
    episode_palette = ["#a16207", "#7c2d12", "#155e75"]

    fig, axes = plt.subplots(1, len(variant_order), figsize=(5.9 * len(variant_order), 5.0), sharex=True, sharey=True)
    if len(variant_order) == 1:
        axes = [axes]

    for ax, (mode, label) in zip(axes, variant_order):
        rollout = test_metrics[mode]["rollout"]
        for color, (episode_idx, episode_label) in zip(episode_palette, episode_specs):
            y = rollout["y"][episode_idx].numpy()
            gate_center = float(test_batch.gate[episode_idx, cross_index].item())
            ax.plot(x, y, color=color, lw=2.2, label=episode_label)
            draw_wall(
                ax=ax,
                x_gate=float(args.x_gate),
                gate_center=gate_center,
                half_width=float(args.gate_half_width),
                corridor_limit=float(args.corridor_limit),
                color=color,
                alpha=0.45,
            )
            ax.scatter([args.x_gate], [gate_center], color=color, s=28, zorder=4)

        ax.axhline(args.corridor_limit, color="#9ca3af", lw=1.0, ls="--", alpha=0.6)
        ax.axhline(-args.corridor_limit, color="#9ca3af", lw=1.0, ls="--", alpha=0.6)
        ax.axvline(args.x_gate, color="#111827", lw=1.0, alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_xlim(args.x0 - 0.05, args.x0 + args.vx * args.horizon + 0.05)
        ax.set_ylim(-args.corridor_limit - 0.15, args.corridor_limit + 0.15)

    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right")
    fig.suptitle("Representative trajectories with crossing gates", y=0.99, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "representative_trajectories.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_variant_comparison(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_batch: EpisodeBatch,
    test_metrics: dict[str, dict],
    cross_index: int,
    show_plots: bool,
) -> int:
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    colors = variant_colors()
    episode_idx = select_comparison_index(test_batch, args, cross_index)
    x = build_x_positions(args, torch.device("cpu")).numpy()
    gate_center = float(test_batch.gate[episode_idx, cross_index].item())

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    draw_wall(
        ax=ax,
        x_gate=float(args.x_gate),
        gate_center=gate_center,
        half_width=float(args.gate_half_width),
        corridor_limit=float(args.corridor_limit),
        color="#111827",
        alpha=0.85,
    )
    for mode, label in variant_order:
        y = test_metrics[mode]["rollout"]["y"][episode_idx].numpy()
        ax.plot(x, y, lw=2.6, color=colors[mode], label=label)

    ax.scatter([args.x_gate], [gate_center], color="#111827", s=42, zorder=4, label="Gate center at crossing")
    ax.axhline(args.corridor_limit, color="#9ca3af", lw=1.0, ls="--", alpha=0.6)
    ax.axhline(-args.corridor_limit, color="#9ca3af", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Same episode, different controller variants")
    ax.set_xlim(args.x0 - 0.05, args.x0 + args.vx * args.horizon + 0.05)
    ax.set_ylim(-args.corridor_limit - 0.15, args.corridor_limit + 0.15)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "variant_comparison.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return episode_idx


def plot_metrics_bars(
    *,
    run_dir: Path,
    variant_order: list[tuple[str, str]],
    test_metrics: dict[str, dict],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    colors = variant_colors()
    labels = [label for _, label in variant_order]
    modes = [mode for mode, _ in variant_order]
    x = np.arange(len(modes))

    panels = [
        ("success_rate", "Success rate", (0.0, 1.02)),
        ("collision_rate", "Collision rate", (0.0, 1.02)),
        ("avg_abs_cross_error", "Mean |y - g| at crossing", None),
        ("avg_control_energy", "Control energy", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.ravel()
    for ax, (metric_key, title, ylim) in zip(axes, panels):
        values = [test_metrics[mode][metric_key] for mode in modes]
        ax.bar(x, values, color=[colors[mode] for mode in modes], width=0.65)
        ax.set_xticks(x, labels, rotation=16, ha="right")
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        offset = 0.02 * max(1.0, max(values))
        for idx, value in enumerate(values):
            ax.text(idx, value + offset, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Test metrics across controller variants", y=0.98, fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.09, right=0.97, wspace=0.28, hspace=0.32)
    fig.savefig(run_dir / "metrics_bars.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_counterfactual_context_case(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    test_batch: EpisodeBatch,
    test_metrics: dict[str, dict],
    cross_index: int,
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    upper_idx, lower_idx = select_counterfactual_pair(test_batch, args, cross_index)
    t = np.arange(args.horizon)
    t_slice = slice(max(0, cross_index - 18), min(args.horizon, cross_index + 12))

    w_seq = test_batch.disturbance[upper_idx].numpy()
    g_upper = test_batch.gate[upper_idx].numpy()
    g_lower = test_batch.gate[lower_idx].numpy()
    v_dist_upper = test_metrics["disturbance_only"]["rollout"]["v_pb"][upper_idx].numpy()
    v_dist_lower = test_metrics["disturbance_only"]["rollout"]["v_pb"][lower_idx].numpy()
    v_ctx_upper = test_metrics["context"]["rollout"]["v_pb"][upper_idx].numpy()
    v_ctx_lower = test_metrics["context"]["rollout"]["v_pb"][lower_idx].numpy()
    y_ctx_upper = test_metrics["context"]["rollout"]["y"][upper_idx].numpy()
    y_ctx_lower = test_metrics["context"]["rollout"]["y"][lower_idx].numpy()

    lower_color, upper_color = gate_colors()

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    ax = axes[0, 0]
    ax.plot(t[t_slice], w_seq[t_slice], color="#111827", lw=2.2, label="Shared disturbance")
    ax2 = ax.twinx()
    ax2.plot(t[t_slice], g_upper[t_slice], color=upper_color, lw=2.0, ls="--", label="Upper gate")
    ax2.plot(t[t_slice], g_lower[t_slice], color=lower_color, lw=2.0, ls="--", label="Lower gate")
    ax.set_title("Same disturbance, opposite gate context")
    ax.set_ylabel("w_t")
    ax2.set_ylabel("g_t")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left")

    ax = axes[0, 1]
    ax.plot(t[t_slice], v_dist_upper[t_slice], color=upper_color, lw=2.4, label="Upper gate case")
    ax.plot(t[t_slice], v_dist_lower[t_slice], color=lower_color, lw=2.0, ls="--", label="Lower gate case")
    ax.set_title("Disturbance-only PB+SSM correction")
    ax.set_ylabel("v_t")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    ax.plot(t[t_slice], v_ctx_upper[t_slice], color=upper_color, lw=2.4, label="Upper gate case")
    ax.plot(t[t_slice], v_ctx_lower[t_slice], color=lower_color, lw=2.0, ls="--", label="Lower gate case")
    ax.set_title("Causal enriched-context correction")
    ax.set_xlabel("time step")
    ax.set_ylabel("v_t")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    time_y = np.arange(args.horizon + 1)
    y_slice = slice(max(0, cross_index - 18), min(args.horizon + 1, cross_index + 13))
    ax.plot(time_y[y_slice], y_ctx_upper[y_slice], color=upper_color, lw=2.6, label="Robot y_t (upper gate)")
    ax.plot(time_y[y_slice], y_ctx_lower[y_slice], color=lower_color, lw=2.2, ls="--", label="Robot y_t (lower gate)")
    ax.plot(t[t_slice], g_upper[t_slice], color=upper_color, lw=1.6, alpha=0.5)
    ax.plot(t[t_slice], g_lower[t_slice], color=lower_color, lw=1.6, alpha=0.5)
    ax.axvline(cross_index, color="#111827", lw=1.2, alpha=0.4)
    ax.set_title("Context-aware state response near the wall")
    ax.set_xlabel("time step")
    ax.set_ylabel("y_t")
    ax.legend(loc="upper left")

    fig.suptitle("Counterfactual pair: the correct PB response depends on context", y=0.98, fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.94, wspace=0.26, hspace=0.28)
    fig.savefig(run_dir / "counterfactual_context_case.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_training_curves(
    *,
    run_dir: Path,
    histories: dict[str, list[dict]],
    variant_order: list[tuple[str, str]],
    show_plots: bool,
) -> None:
    plt = get_plt(show_plots)
    setup_plot_style(plt)

    colors = variant_colors()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    for mode, label in variant_order:
        history = histories.get(mode, [])
        if not history:
            continue
        epochs = [entry["epoch"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]
        val_epochs = [entry["epoch"] for entry in history if "val_cost" in entry]
        val_cost = [entry["val_cost"] for entry in history if "val_cost" in entry]
        val_success = [entry["val_success_rate"] for entry in history if "val_success_rate" in entry]

        axes[0].plot(epochs, train_loss, color=colors[mode], lw=1.8, alpha=0.35)
        axes[0].plot(val_epochs, val_cost, color=colors[mode], lw=2.4, label=label)
        axes[1].plot(val_epochs, val_success, color=colors[mode], lw=2.4, label=label)

    axes[0].set_title("Cost during training")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cost")
    axes[1].set_title("Validation success rate")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("success rate")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def strip_rollout_from_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "rollout"}


def build_interpretation(
    variant_order: list[tuple[str, str]],
    test_metrics: dict[str, dict],
    args: argparse.Namespace,
) -> str:
    context = test_metrics["context"]
    disturbance_only = test_metrics["disturbance_only"]
    nominal = test_metrics["nominal"]

    best_label = max(
        variant_order,
        key=lambda item: (test_metrics[item[0]]["success_rate"], -test_metrics[item[0]]["avg_cost"]),
    )[1]

    return (
        f"Best performer: {best_label}. "
        f"Causal enriched-context PB+SSM reaches a success rate of {context['success_rate']:.3f}, versus "
        f"{disturbance_only['success_rate']:.3f} for disturbance-only PB+SSM and "
        f"{nominal['success_rate']:.3f} for nominal control. "
        f"The disturbance-only controller still sees the reconstructed disturbance sequence, but it cannot infer "
        f"whether the opening is above or below, so its crossing error stays high "
        f"({disturbance_only['avg_abs_cross_error']:.3f} vs {context['avg_abs_cross_error']:.3f}). "
        f"The gate schedule is frozen {int(args.gate_settle_steps)} steps before the wall, so the final switch is "
        f"not unrealistically late, and the contextual signal remains causal while still actionable."
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

    cross_index = compute_cross_index(args)
    freeze_step = compute_freeze_step(args, cross_index)
    track_window = make_track_window(args, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_id or f"contextual_pb_gate_ssm_{timestamp}"
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = dict(vars(args))
    config_payload["cross_index"] = int(cross_index)
    config_payload["freeze_step"] = int(freeze_step)
    config_payload["context_dim"] = int(context_dim())
    save_json(run_dir / "config.json", config_payload)

    val_batch = sample_episode_batch(
        args=args,
        batch_size=int(args.val_batch),
        seed=int(args.seed) + 50_000,
        paired=True,
        shuffle=False,
        cross_index=cross_index,
    )
    test_batch = sample_episode_batch(
        args=args,
        batch_size=int(args.test_batch),
        seed=int(args.seed) + 60_000,
        paired=True,
        shuffle=False,
        cross_index=cross_index,
    )

    specs = variant_specs()
    controllers: dict[str, PBController | None] = {}
    histories: dict[str, list[dict]] = {}
    val_metrics: dict[str, dict] = {}
    test_metrics: dict[str, dict] = {}

    print(f"Run directory: {run_dir}")
    print(f"Wall crossing occurs at step {cross_index}. Gate schedule freezes at step {freeze_step}.")

    for mode, label in specs:
        print(f"\nTraining/evaluating {label}...")
        controller, history, best_val_metrics = train_controller(
            args=args,
            device=device,
            mode=mode,
            val_batch=val_batch,
            track_window=track_window,
            cross_index=cross_index,
        )
        controllers[mode] = controller
        histories[mode] = history
        val_metrics[mode] = best_val_metrics
        test_metrics[mode] = evaluate_variant(
            args=args,
            batch=test_batch,
            device=device,
            mode=mode,
            controller=controller,
            track_window=track_window,
            cross_index=cross_index,
        )
        if controller is not None:
            torch.save(controller.state_dict(), run_dir / f"{mode}_controller.pt")

    show_plots = not args.no_show_plots
    plot_wall_style_summary(
        args=args,
        run_dir=run_dir,
        variant_order=specs,
        test_batch=test_batch,
        test_metrics=test_metrics,
        controllers=controllers,
        device=device,
        cross_index=cross_index,
        track_window=track_window,
        show_plots=show_plots,
    )
    plot_training_curves(
        run_dir=run_dir,
        histories=histories,
        variant_order=specs,
        show_plots=show_plots,
    )

    save_json(run_dir / "metrics.json", {mode: strip_rollout_from_metrics(test_metrics[mode]) for mode, _ in specs})
    save_json(run_dir / "val_metrics.json", {mode: strip_rollout_from_metrics(val_metrics[mode]) for mode, _ in specs})
    save_json(run_dir / "train_history.json", histories)

    interpretation = build_interpretation(specs, test_metrics, args)
    (run_dir / "interpretation.txt").write_text(interpretation + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("RESULTS OVERVIEW")
    print("=" * 72)
    for mode, label in specs:
        metrics = test_metrics[mode]
        print(
            f"{label:32s} "
            f"success={metrics['success_rate']:.3f} "
            f"collision={metrics['collision_rate']:.3f} "
            f"cross_err={metrics['avg_abs_cross_error']:.3f} "
            f"energy={metrics['avg_control_energy']:.3f}"
        )
    print("=" * 72)
    print(interpretation)


if __name__ == "__main__":
    main()
