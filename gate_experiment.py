"""
Contextual two-gate PB experiment.

Goal:
  Demonstrate why contextual input z matters in M(w, z):
  for the same start state, different obstacle radii context should trigger
  different safe corridor choices (top vs bottom).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from bounded_mlp_operator import BoundedMLPOperator
from context_lifting import LpContextLifter
from gate_env import (
    GateScenario,
    exp_barrier_penalty,
    fixed_gate_centers,
    gate_choice_from_trajectory,
    gate_gaps,
    mirror_vertical_scenario,
    min_dist_to_edge,
    sample_paired_gate_scenarios,
    sample_gate_scenarios,
    scenario_to_device,
    wall_barrier_penalty,
)
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_core.factories import build_factorized_controller
from pb_core.noise import DecayingGaussianNoise
from pb_core.validation import validate_component_compatibility
from pb_controller import PBController
from ssm_operators import MpDeepSSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Contextual two-gate PB experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--val_batch", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--val_horizon", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--no_show_plots", action="store_true")
    parser.add_argument("--nom_pre_kp", type=float, default=1.0)
    parser.add_argument("--nom_pre_kd", type=float, default=1.5)
    parser.add_argument("--true_pre_kp", type=float, default=0.85)
    parser.add_argument("--true_pre_kd", type=float, default=1.2)
    parser.add_argument("--train_mode", type=str, default="factorized", choices=["factorized", "mp_only"])
    parser.add_argument("--mp_u_bound", type=float, default=10.0)
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=False)
    parser.add_argument(
        "--mp_context_lift_type",
        type=str,
        default="identity",
        choices=["identity", "linear", "mlp"],
    )
    parser.add_argument("--mp_context_lift_dim", type=int, default=5)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=32)
    parser.add_argument(
        "--mp_context_decay_law",
        type=str,
        default="poly",
        choices=["exp", "poly", "finite"],
    )
    parser.add_argument("--mp_context_decay_rate", type=float, default=0.08)
    parser.add_argument("--mp_context_decay_power", type=float, default=.7)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=80)
    parser.add_argument("--mp_context_lp_p", type=float, default=2.0)
    parser.add_argument("--mp_context_scale", type=float, default=1.0)

    # Scenario geometry/sampling.
    parser.add_argument("--center_x", type=float, default=0.9)
    parser.add_argument("--y_sep", type=float, default=0.9)
    parser.add_argument("--r_center", type=float, default=0.43)
    parser.add_argument("--open_r_min", type=float, default=0.16)
    parser.add_argument("--open_r_max", type=float, default=0.22)
    parser.add_argument("--closed_r_min", type=float, default=0.46)
    parser.add_argument("--closed_r_max", type=float, default=0.49)
    parser.add_argument("--ambiguous_frac", type=float, default=0.0)
    parser.add_argument("--paired_train_context", dest="paired_train_context", action="store_true")
    parser.add_argument("--no_paired_train_context", dest="paired_train_context", action="store_false")
    parser.set_defaults(paired_train_context=True)
    parser.add_argument("--start_x_min", type=float, default=1.7)
    parser.add_argument("--start_x_max", type=float, default=2.4)
    parser.add_argument("--start_y_min", type=float, default=-.06)
    parser.add_argument("--start_y_max", type=float, default=.06)
    parser.add_argument("--mirror_augment_train", dest="mirror_augment_train", action="store_true")
    parser.add_argument("--no_mirror_augment_train", dest="mirror_augment_train", action="store_false")
    parser.set_defaults(mirror_augment_train=True)

    # Loss: distance + barrier only.
    parser.add_argument("--w_term", type=float, default=32.0)
    parser.add_argument("--w_stage", type=float, default=59.0)
    parser.add_argument("--stage_w_start", type=float, default=1.55)
    parser.add_argument("--stage_w_end", type=float, default=16.2)
    parser.add_argument("--w_bar", type=float, default=120.0)
    parser.add_argument("--w_wall", type=float, default=160.0)
    parser.add_argument("--vel_dist_weight", type=float, default=0.35)
    parser.add_argument("--bar_margin", type=float, default=0.13)
    parser.add_argument("--bar_alpha", type=float, default=18.0)
    parser.add_argument("--bar_cap", type=float, default=200.0)
    parser.add_argument("--wall_y_limit", type=float, default=0.80)
    parser.add_argument("--wall_margin", type=float, default=0.08)
    parser.add_argument("--wall_alpha", type=float, default=18.0)
    parser.add_argument("--wall_cap", type=float, default=200.0)
    parser.add_argument("--z_gain", type=float, default=8.0)
    parser.add_argument("--z_residual_gain", type=float, default=20.0)

    # Disturbance used during training rollouts.
    parser.add_argument("--noise_sigma0", type=float, default=0.006)
    parser.add_argument("--noise_tau", type=float, default=20.0)
    parser.add_argument("--noisy_test_mc", type=int, default=0)
    parser.add_argument("--noisy_test_sigma0", type=float, default=0.0)
    parser.add_argument("--noisy_test_tau", type=float, default=20.0)

    # Evaluation/checkpoint selection.
    parser.add_argument("--success_tol", type=float, default=0.20)
    parser.add_argument("--choice_gap_margin", type=float, default=0.015)
    parser.add_argument("--gate_x_eval", type=float, default=0.9)
    parser.add_argument(
        "--best_ckpt_metric",
        type=str,
        default="success_then_loss",
        choices=["loss", "collision_then_loss", "success_then_loss"],
    )
    parser.add_argument("--best_ckpt_collision_tol", type=float, default=1e-4)

    # Plot controls.
    parser.add_argument("--traj_plot_each", type=int, default=4)
    parser.add_argument("--overlay_num_starts", type=int, default=4)
    parser.add_argument("--phase_res", type=int, default=80)
    parser.add_argument("--phase_batch", type=int, default=2048)
    parser.add_argument("--phase_start_x", type=float, default=2.0)
    parser.add_argument("--phase_start_y", type=float, default=0.0)
    parser.add_argument("--ambig_r_base", type=float, default=0.33)
    parser.add_argument("--ambig_r_delta", type=float, default=0.015)
    parser.add_argument("--skip_ablations", action="store_true")
    return parser.parse_args()


@dataclass
class LossConfig:
    w_term: float
    w_stage: float
    stage_w_start: float
    stage_w_end: float
    w_bar: float
    w_wall: float
    vel_dist_weight: float
    bar_margin: float
    bar_alpha: float
    bar_cap: float
    wall_y_limit: float
    wall_margin: float
    wall_alpha: float
    wall_cap: float


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


def sample_dataset(
    *,
    batch_size: int,
    seed: int,
    args: argparse.Namespace,
    centers: torch.Tensor,
    paired_context: bool,
    mirror_augment: bool,
):
    if paired_context:
        scenario = sample_paired_gate_scenarios(
            batch_size=batch_size,
            seed=seed,
            start_x=(args.start_x_min, args.start_x_max),
            start_y=(args.start_y_min, args.start_y_max),
            centers=centers,
            r_center=args.r_center,
            open_range=(args.open_r_min, args.open_r_max),
            closed_range=(args.closed_r_min, args.closed_r_max),
        )
    else:
        scenario = sample_gate_scenarios(
            batch_size=batch_size,
            seed=seed,
            start_x=(args.start_x_min, args.start_x_max),
            start_y=(args.start_y_min, args.start_y_max),
            centers=centers,
            r_center=args.r_center,
            open_range=(args.open_r_min, args.open_r_max),
            closed_range=(args.closed_r_min, args.closed_r_max),
            ambiguous_frac=args.ambiguous_frac,
        )

    if mirror_augment:
        rng = torch.Generator()
        rng.manual_seed(int(seed) + 7777)
        mask = torch.rand(scenario.start.shape[0], generator=rng) < 0.5
        scenario = mirror_vertical_scenario(scenario, mask=mask)
    return scenario


def make_x0(scenario: GateScenario, device: torch.device) -> torch.Tensor:
    bsz = scenario.start.shape[0]
    return torch.cat(
        [scenario.start.to(device), torch.zeros(bsz, 2, device=device)],
        dim=-1,
    ).unsqueeze(1)


def shuffled_context_scenario(scenario: GateScenario) -> GateScenario:
    bsz = scenario.start.shape[0]
    perm = torch.randperm(bsz, device=scenario.start.device)
    return GateScenario(
        start=scenario.start,
        goal=scenario.goal,
        centers=scenario.centers,
        radii=scenario.radii[perm],
        mode=scenario.mode[perm],
    )


def build_minimal_context(
    x: torch.Tensor,
    scenario: GateScenario,
    *,
    z_gain: float,
) -> torch.Tensor:
    """
    Minimal context for clear dependency:
      z = [r_top, r_center, r_bottom, gap_top, gap_bottom]
    repeated over time. No state-relative obstacle distances included.
    """
    bsz, t_steps = x.shape[0], x.shape[1]
    radii = scenario.radii.to(x.device).unsqueeze(1).expand(-1, t_steps, -1)  # (B,T,3)
    gaps = gate_gaps(scenario).to(x.device).unsqueeze(1).expand(-1, t_steps, -1)  # (B,T,2)
    z = torch.cat([radii, gaps], dim=-1)
    return float(z_gain) * z


def _to_canonical_state(x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    # x: (B,1,4), sign: (B,1,1) where -1 means mirror y-axis.
    xc = x.clone()
    xc[..., 1] = sign[..., 0] * xc[..., 1]  # y
    xc[..., 3] = sign[..., 0] * xc[..., 3]  # vy
    return xc


def _to_world_control(u_canon: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    uw = u_canon.clone()
    uw[..., 1] = sign[..., 0] * uw[..., 1]  # u_y
    return uw


def _canonical_context_vector(scenario: GateScenario, *, z_gain: float) -> torch.Tensor:
    """
    Return static context vector z0 of shape (B,5):
      [r_top, r_center, r_bottom, gap_top, gap_bottom] in canonical frame.
    Canonical frame enforces "open corridor is bottom" by mirroring top-open samples.
    """
    radii = scenario.radii.to(scenario.start.device).clone()  # (B,3)
    gaps = gate_gaps(scenario).to(scenario.start.device).clone()  # (B,2)
    mode = scenario.mode.to(scenario.start.device)

    top_open = mode == 0
    if bool(top_open.any().item()):
        idx = torch.nonzero(top_open, as_tuple=False).squeeze(-1)
        r0 = radii[idx, 0].clone()
        r2 = radii[idx, 2].clone()
        radii[idx, 0] = r2
        radii[idx, 2] = r0

        g0 = gaps[idx, 0].clone()
        g1 = gaps[idx, 1].clone()
        gaps[idx, 0] = g1
        gaps[idx, 1] = g0

    z0 = torch.cat([radii, gaps], dim=-1)
    return float(z_gain) * z0


def rollout_gate(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: GateScenario,
    *,
    horizon: int,
    device: torch.device,
    noise: torch.Tensor | None,
    context_scenario: GateScenario | None = None,
    zero_context: bool = False,
    mp_only: bool = False,
    z_gain: float = 8.0,
):
    x = make_x0(scenario, device)  # world frame
    if context_scenario is None:
        context_scenario = scenario

    # Canonical mirror transform is only for context-conditioned (factorized) control.
    # For true Mp-only control, keep world frame to avoid leaking mode/context.
    use_canonical = (not mp_only) and hasattr(controller.operator, "mb")
    sign = torch.ones(x.shape[0], 1, 1, device=device, dtype=x.dtype)
    if use_canonical:
        mode = context_scenario.mode.to(device)
        sign[mode == 0] = -1.0  # top-open -> mirrored canonical frame

    z0 = _canonical_context_vector(context_scenario, z_gain=z_gain).to(device)  # (B,5)
    z1 = z0.unsqueeze(1)  # (B,1,5)
    if zero_context:
        z1 = torch.zeros_like(z1)

    if noise is not None:
        if noise.shape[0] != x.shape[0] or noise.shape[1] != horizon or noise.shape[2] != x.shape[2]:
            raise ValueError(
                "noise must have shape (B,horizon,Nx). "
                f"Got {tuple(noise.shape)} expected ({x.shape[0]},{horizon},{x.shape[2]})"
            )

    x_c0 = _to_canonical_state(x, sign)
    controller.reset(x_c0, u_init=None, w0=x_c0)

    x_log = []
    u_log = []
    w_log = []
    for t in range(horizon):
        x_c = _to_canonical_state(x, sign)
        if mp_only:
            # True M_p-only ablation: bypass M_b entirely and map M_p features to control
            # via first u_dim channels.
            if not hasattr(controller.operator, "mp"):
                raise TypeError("mp_only ablation requires a factorized operator with an 'mp' module.")
            if controller.u_dim is None:
                raise ValueError("mp_only ablation requires controller.u_dim to be set.")
            w_c = controller._compute_w_t(x_c, t=t)
            w_mp = w_c
            mp_context_lifter = getattr(controller.operator, "mp_context_lifter", None)
            if mp_context_lifter is not None:
                z_lift = mp_context_lifter(z1)
                w_mp = torch.cat([w_c, z_lift], dim=-1)
            v = controller.operator.mp(w_mp)
            if v.shape[-1] < int(controller.u_dim):
                raise ValueError(
                    f"mp output dim {v.shape[-1]} is smaller than u_dim={controller.u_dim}"
                )
            u_c = v[..., : int(controller.u_dim)]
            if controller.u_nominal is not None:
                u_nom = controller.u_nominal(x_c, t)
                u_c = u_c + (u_nom if u_nom.dim() == 3 else u_nom.unsqueeze(1))

            if controller.detach_state:
                controller.state.x_tm1 = x_c.detach()
                controller.state.u_tm1 = u_c.detach()
            else:
                controller.state.x_tm1 = x_c
                controller.state.u_tm1 = u_c
            controller.state.has_prev = True
        else:
            u_c, w_c = controller.forward_step(x_c, z1, t=t)
        u_w = _to_world_control(u_c, sign)
        x = plant_true.forward(x, u_w, t=t)
        if noise is not None:
            x = x + noise[:, t:t + 1, :]
        x_log.append(x)
        u_log.append(u_w)
        w_log.append(w_c)

    return torch.cat(x_log, dim=1), torch.cat(u_log, dim=1), torch.cat(w_log, dim=1)


def compute_loss_per_sample(
    x_seq: torch.Tensor,
    scenario: GateScenario,
    cfg: LossConfig,
) -> torch.Tensor:
    pos_sq = torch.sum(x_seq[..., :2] ** 2, dim=-1)
    vel_sq = torch.sum(x_seq[..., 2:] ** 2, dim=-1)
    dist = torch.sqrt(pos_sq + float(cfg.vel_dist_weight) * vel_sq + 1e-9)
    term = dist[:, -1] * cfg.w_term
    t_steps = dist.shape[1]
    stage_w = torch.linspace(
        float(cfg.stage_w_start),
        float(cfg.stage_w_end),
        t_steps,
        device=dist.device,
        dtype=dist.dtype,
    )
    # Normalize to keep stage term scale stable while emphasizing late timesteps.
    stage_w = stage_w / stage_w.mean().clamp_min(1e-6)
    stage = (dist * stage_w.unsqueeze(0)).mean(dim=1) * cfg.w_stage
    barrier = exp_barrier_penalty(
        x_seq,
        scenario,
        margin=cfg.bar_margin,
        alpha=cfg.bar_alpha,
        cap=cfg.bar_cap,
    ).mean(dim=(1, 2)) * cfg.w_bar
    wall = wall_barrier_penalty(
        x_seq,
        y_limit=cfg.wall_y_limit,
        margin=cfg.wall_margin,
        alpha=cfg.wall_alpha,
        cap=cfg.wall_cap,
    ).mean(dim=(1, 2)) * cfg.w_wall
    return term + stage + barrier + wall


def compute_loss(
    x_seq: torch.Tensor,
    scenario: GateScenario,
    cfg: LossConfig,
) -> tuple[torch.Tensor, dict]:
    total_per = compute_loss_per_sample(x_seq, scenario, cfg)
    total = total_per.mean()
    pos_sq = torch.sum(x_seq[..., :2] ** 2, dim=-1)
    vel_sq = torch.sum(x_seq[..., 2:] ** 2, dim=-1)
    dist = torch.sqrt(pos_sq + float(cfg.vel_dist_weight) * vel_sq + 1e-9)
    term = dist[:, -1].mean() * cfg.w_term
    t_steps = dist.shape[1]
    stage_w = torch.linspace(
        float(cfg.stage_w_start),
        float(cfg.stage_w_end),
        t_steps,
        device=dist.device,
        dtype=dist.dtype,
    )
    stage_w = stage_w / stage_w.mean().clamp_min(1e-6)
    stage = (dist * stage_w.unsqueeze(0)).mean() * cfg.w_stage
    barrier = exp_barrier_penalty(
        x_seq,
        scenario,
        margin=cfg.bar_margin,
        alpha=cfg.bar_alpha,
        cap=cfg.bar_cap,
    ).mean() * cfg.w_bar
    wall = wall_barrier_penalty(
        x_seq,
        y_limit=cfg.wall_y_limit,
        margin=cfg.wall_margin,
        alpha=cfg.wall_alpha,
        cap=cfg.wall_cap,
    ).mean() * cfg.w_wall
    parts = {
        "loss_total": float(total.item()),
        "loss_term": float(term.item()),
        "loss_stage": float(stage.item()),
        "loss_barrier": float(barrier.item()),
        "loss_wall": float(wall.item()),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: GateScenario,
    *,
    horizon: int,
    device: torch.device,
    loss_cfg: LossConfig,
    success_tol: float,
    choice_gap_margin: float,
    gate_x_eval: float,
    context_mode: str = "true",
    noise: torch.Tensor | None = None,
    z_gain: float = 8.0,
):
    if context_mode not in {"true", "shuffled", "zero", "mp_only"}:
        raise ValueError(f"Unknown context_mode {context_mode}")

    if context_mode == "true":
        ctx_scenario = scenario
        zero_ctx = False
        mp_only = False
    elif context_mode == "shuffled":
        ctx_scenario = shuffled_context_scenario(scenario)
        zero_ctx = False
        mp_only = False
    elif context_mode == "mp_only":
        ctx_scenario = scenario
        zero_ctx = True
        op = controller.operator
        mp_only = hasattr(op, "mp") and hasattr(op, "mb")
    else:
        ctx_scenario = scenario
        zero_ctx = True
        mp_only = False

    controller.eval()
    x_seq, u_seq, w_seq = rollout_gate(
        controller=controller,
        plant_true=plant_true,
        scenario=scenario,
        horizon=horizon,
        device=device,
        noise=noise,
        context_scenario=ctx_scenario,
        zero_context=zero_ctx,
        mp_only=mp_only,
        z_gain=z_gain,
    )
    loss, parts = compute_loss(x_seq, scenario, loss_cfg)
    tail_k = min(10, x_seq.shape[1])
    w_last_mag = torch.norm(w_seq[:, -1, :], dim=-1)
    w_tail_mag = torch.norm(w_seq[:, -tail_k:, :], dim=-1).mean(dim=1)
    min_edge = min_dist_to_edge(x_seq, scenario)
    collided = min_edge.min(dim=1).values < 0.0
    collision_rate = float(collided.float().mean().item())
    terminal_dist = torch.norm(x_seq[:, -1, :2], dim=-1)
    success = (~collided) & (terminal_dist < float(success_tol))
    success_rate = float(success.float().mean().item())

    mode_top = scenario.mode == 0
    mode_bottom = scenario.mode == 1
    if bool(mode_top.any().item()):
        top_collision = float(collided[mode_top].float().mean().item())
        top_success = float(success[mode_top].float().mean().item())
    else:
        top_collision = float("nan")
        top_success = float("nan")
    if bool(mode_bottom.any().item()):
        bottom_collision = float(collided[mode_bottom].float().mean().item())
        bottom_success = float(success[mode_bottom].float().mean().item())
    else:
        bottom_collision = float("nan")
        bottom_success = float("nan")

    gaps = gate_gaps(scenario)
    gap_diff = gaps[:, 0] - gaps[:, 1]
    valid_choice = (torch.abs(gap_diff) > float(choice_gap_margin)) & (~collided)
    chosen_top = gate_choice_from_trajectory(x_seq, gate_x=gate_x_eval)
    optimal_top = gap_diff > 0.0
    if bool(valid_choice.any().item()):
        choice_acc = float((chosen_top[valid_choice] == optimal_top[valid_choice]).float().mean().item())
        choice_count = int(valid_choice.sum().item())
    else:
        choice_acc = float("nan")
        choice_count = 0

    parts.update(
        {
            "context_mode": context_mode,
            "loss": float(loss.item()),
            "collision_rate": collision_rate,
            "success_rate": success_rate,
            "top_collision_rate": top_collision,
            "bottom_collision_rate": bottom_collision,
            "top_success_rate": top_success,
            "bottom_success_rate": bottom_success,
            "terminal_dist": float(terminal_dist.mean().item()),
            "w_last_mag": float(w_last_mag.mean().item()),
            "w_tail_mag": float(w_tail_mag.mean().item()),
            "choice_accuracy": choice_acc,
            "choice_count": choice_count,
        }
    )
    return parts, x_seq, u_seq


def _nanmean_std(vals: torch.Tensor) -> tuple[float, float]:
    mask = ~torch.isnan(vals)
    if not bool(mask.any().item()):
        return float("nan"), float("nan")
    v = vals[mask]
    mean = float(v.mean().item())
    std = float(v.std(unbiased=False).item()) if v.numel() > 1 else 0.0
    return mean, std


@torch.no_grad()
def evaluate_noisy_mc(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: GateScenario,
    *,
    horizon: int,
    device: torch.device,
    loss_cfg: LossConfig,
    success_tol: float,
    choice_gap_margin: float,
    gate_x_eval: float,
    context_mode: str,
    z_gain: float,
    noise_model: DecayingGaussianNoise,
    mc_trials: int,
    seed_base: int,
):
    if mc_trials <= 0:
        raise ValueError(f"mc_trials must be > 0, got {mc_trials}")
    metrics_list: list[dict] = []
    x_ref = None
    u_ref = None
    bsz = int(scenario.start.shape[0])
    nx = 4
    for k in range(int(mc_trials)):
        noise = noise_model.sample(
            bsz=bsz,
            horizon=horizon,
            nx=nx,
            device=device,
            seed=int(seed_base) + int(k),
        )
        metrics_k, x_k, u_k = evaluate(
            controller=controller,
            plant_true=plant_true,
            scenario=scenario,
            horizon=horizon,
            device=device,
            loss_cfg=loss_cfg,
            success_tol=success_tol,
            choice_gap_margin=choice_gap_margin,
            gate_x_eval=gate_x_eval,
            context_mode=context_mode,
            noise=noise,
            z_gain=z_gain,
        )
        metrics_list.append(metrics_k)
        if x_ref is None:
            x_ref = x_k
            u_ref = u_k

    mean_dict: dict[str, float | str] = {}
    std_dict: dict[str, float] = {}
    keys = metrics_list[0].keys()
    for key in keys:
        v0 = metrics_list[0][key]
        if isinstance(v0, str):
            mean_dict[key] = v0
            continue
        vals = torch.tensor([float(m[key]) for m in metrics_list], dtype=torch.float32, device=device)
        mean_v, std_v = _nanmean_std(vals)
        mean_dict[key] = mean_v
        std_dict[key] = std_v

    out = {
        "mc_trials": int(mc_trials),
        "noise_sigma0": float(noise_model.sigma0),
        "noise_tau": float(noise_model.tau),
        "mean": mean_dict,
        "std": std_dict,
    }
    return out, x_ref, u_ref


def plot_loss_curves(train_hist, eval_hist, run_dir, show_plots: bool):
    if not train_hist:
        return
    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([d["epoch"] for d in train_hist], [d["loss"] for d in train_hist], label="train", alpha=0.85)
    if eval_hist:
        ax.plot([d["epoch"] for d in eval_hist], [d["loss"] for d in eval_hist], label="val (true ctx)", linewidth=2.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"))
    if not show_plots:
        plt.close(fig)


def plot_control_magnitude(
    u_seq: torch.Tensor,
    run_dir: str,
    show_plots: bool,
    *,
    out_name: str = "control_magnitude_true.png",
    fig_title: str = "Control Magnitude Over Time",
):
    if u_seq is None or u_seq.numel() == 0:
        return
    plt = get_plt(show_plots)
    u_mag = torch.norm(u_seq, dim=-1)  # (B,T)
    t = torch.arange(u_mag.shape[1], device=u_mag.device).detach().cpu().numpy()
    mean = u_mag.mean(dim=0).detach().cpu().numpy()
    q10 = torch.quantile(u_mag, q=0.10, dim=0).detach().cpu().numpy()
    q90 = torch.quantile(u_mag, q=0.90, dim=0).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, mean, color="C0", linewidth=2.0, label="mean ||u_t||")
    ax.fill_between(t, q10, q90, color="C0", alpha=0.2, label="10-90 percentile")
    ax.set_xlabel("t")
    ax.set_ylabel("||u_t||_2")
    ax.set_title(fig_title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, out_name))
    if not show_plots:
        plt.close(fig)


def _draw_obstacles(ax, centers, radii, color="gray", alpha=0.28, linestyle="-", fill=True, linewidth=1.0):
    plt = get_plt(True)
    for k in range(centers.shape[0]):
        circ = plt.Circle(
            (centers[k, 0], centers[k, 1]),
            radii[k],
            color=color,
            alpha=alpha,
            fill=fill,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.add_patch(circ)


def plot_trajectories_by_mode(
    x_seq: torch.Tensor,
    scenario: GateScenario,
    run_dir: str,
    show_plots: bool,
    max_each: int = 4,
    out_name: str = "trajectories_by_context.png",
    fig_title: str | None = None,
):
    plt = get_plt(show_plots)
    top_idx = torch.nonzero(scenario.mode == 0, as_tuple=False).squeeze(-1)
    bot_idx = torch.nonzero(scenario.mode == 1, as_tuple=False).squeeze(-1)
    if top_idx.numel() == 0 or bot_idx.numel() == 0:
        return

    ncols = int(max_each)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8.5), squeeze=False)
    rows = [top_idx[:ncols], bot_idx[:ncols]]
    titles = ["Top-open context", "Bottom-open context"]
    colors = ["C0", "C3"]

    for r in range(2):
        idxs = rows[r]
        for c in range(ncols):
            ax = axes[r, c]
            if c >= idxs.numel():
                ax.axis("off")
                continue
            i = int(idxs[c].item())
            traj = x_seq[i, :, :2].detach().cpu().numpy()
            ax.plot(traj[:, 0], traj[:, 1], color=colors[r], linewidth=2.0)
            ax.scatter([scenario.start[i, 0].item()], [scenario.start[i, 1].item()], color="green", s=30, label="start")
            ax.scatter([0.0], [0.0], color="red", marker="*", s=90, label="goal")
            centers = scenario.centers[i].detach().cpu().numpy()
            radii = scenario.radii[i].detach().cpu().numpy()
            _draw_obstacles(ax, centers, radii, color="gray", alpha=0.25)
            ax.set_aspect("equal", "box")
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(-1.9, 1.9)
            if c == 0:
                ax.set_ylabel("y")
                ax.set_title(titles[r])
            ax.grid(True)
    for c in range(ncols):
        axes[1, c].set_xlabel("x")
    if fig_title is not None and len(fig_title) > 0:
        fig.suptitle(fig_title, y=1.01)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(run_dir, out_name))
    if not show_plots:
        plt.close(fig)


@torch.no_grad()
def plot_context_switch_overlay(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    ref_scenario: GateScenario,
    *,
    horizon: int,
    device: torch.device,
    run_dir: str,
    show_plots: bool,
    num_starts: int,
    open_radius_ref: float,
    closed_radius_ref: float,
    r_center: float,
    z_gain: float,
    out_name: str = "context_switch_overlay.png",
    fig_title: str = "Context Reactivity: Same Start, Different Radii Context",
):
    plt = get_plt(show_plots)
    starts = ref_scenario.start[:num_starts].to(device)
    bsz = starts.shape[0]
    if bsz == 0:
        return
    goal = torch.zeros(bsz, 2, device=device)
    centers = ref_scenario.centers[0:1].to(device).expand(bsz, -1, -1)

    r_top_open = torch.tensor([open_radius_ref, r_center, closed_radius_ref], dtype=starts.dtype, device=device)
    r_bottom_open = torch.tensor([closed_radius_ref, r_center, open_radius_ref], dtype=starts.dtype, device=device)
    rad_top = r_top_open.unsqueeze(0).expand(bsz, -1)
    rad_bot = r_bottom_open.unsqueeze(0).expand(bsz, -1)

    scen_top = GateScenario(start=starts, goal=goal, centers=centers, radii=rad_top, mode=torch.zeros(bsz, dtype=torch.long, device=device))
    scen_bot = GateScenario(start=starts, goal=goal, centers=centers, radii=rad_bot, mode=torch.ones(bsz, dtype=torch.long, device=device))

    x_top, _, _ = rollout_gate(
        controller=controller,
        plant_true=plant_true,
        scenario=scen_top,
        horizon=horizon,
        device=device,
        noise=None,
        z_gain=z_gain,
    )
    x_bot, _, _ = rollout_gate(
        controller=controller,
        plant_true=plant_true,
        scenario=scen_bot,
        horizon=horizon,
        device=device,
        noise=None,
        z_gain=z_gain,
    )

    ncols = min(2, bsz)
    nrows = (bsz + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)

    top_r_np = r_top_open.detach().cpu().numpy()
    bot_r_np = r_bottom_open.detach().cpu().numpy()
    centers_np = centers[0].detach().cpu().numpy()

    for i in range(bsz):
        ax = axes[i // ncols, i % ncols]
        t1 = x_top[i, :, :2].detach().cpu().numpy()
        t2 = x_bot[i, :, :2].detach().cpu().numpy()
        ax.plot(t1[:, 0], t1[:, 1], color="C0", linewidth=2.0, label="context: top-open")
        ax.plot(t2[:, 0], t2[:, 1], color="C3", linewidth=2.0, label="context: bottom-open")
        ax.scatter([starts[i, 0].item()], [starts[i, 1].item()], color="green", s=28)
        ax.scatter([0.0], [0.0], color="red", marker="*", s=85)

        _draw_obstacles(ax, centers_np, top_r_np, color="C0", alpha=0.10, fill=True)
        _draw_obstacles(ax, centers_np, bot_r_np, color="C3", alpha=0.10, fill=True)
        _draw_obstacles(ax, centers_np, top_r_np, color="C0", alpha=0.7, fill=False, linewidth=1.2)
        _draw_obstacles(ax, centers_np, bot_r_np, color="C3", alpha=0.7, fill=False, linewidth=1.2, linestyle="--")

        ax.set_title(f"Same start ({starts[i,0].item():.2f}, {starts[i,1].item():.2f})")
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-1.9, 1.9)
        ax.set_aspect("equal", "box")
        ax.grid(True)
    total = nrows * ncols
    for j in range(bsz, total):
        axes[j // ncols, j % ncols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(fig_title, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, out_name), bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


@torch.no_grad()
def plot_context_phase_maps(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    *,
    horizon: int,
    device: torch.device,
    run_dir: str,
    show_plots: bool,
    center_x: float,
    y_sep: float,
    r_center: float,
    start_x: float,
    start_y: float,
    r_min: float,
    r_max: float,
    res: int,
    batch: int,
    gate_x_eval: float,
    z_gain: float,
):
    plt = get_plt(show_plots)
    r_vals = torch.linspace(r_min, r_max, res, device=device)
    r_top, r_bottom = torch.meshgrid(r_vals, r_vals, indexing="ij")
    n = r_top.numel()

    starts = torch.tensor([start_x, start_y], dtype=torch.float32, device=device).view(1, 2).expand(n, -1)
    goal = torch.zeros(n, 2, device=device)
    centers = fixed_gate_centers(center_x=center_x, y_sep=y_sep).to(device).view(1, 3, 2).expand(n, -1, -1)
    radii = torch.stack([r_top.reshape(-1), torch.full((n,), r_center, device=device), r_bottom.reshape(-1)], dim=-1)
    d_top = torch.norm(centers[:, 0, :] - centers[:, 1, :], dim=-1)
    d_bottom = torch.norm(centers[:, 2, :] - centers[:, 1, :], dim=-1)
    gap_top = d_top - (radii[:, 0] + radii[:, 1])
    gap_bottom = d_bottom - (radii[:, 2] + radii[:, 1])
    mode = torch.full((n,), 2, dtype=torch.long, device=device)
    mode[gap_top > gap_bottom] = 0
    mode[gap_bottom > gap_top] = 1
    scenario = GateScenario(
        start=starts,
        goal=goal,
        centers=centers,
        radii=radii,
        mode=mode,
    )

    choice_top = torch.zeros(n, device=device, dtype=torch.float32)
    collided = torch.zeros(n, device=device, dtype=torch.float32)
    term_dist = torch.zeros(n, device=device, dtype=torch.float32)

    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        sc = GateScenario(
            start=scenario.start[i0:i1],
            goal=scenario.goal[i0:i1],
            centers=scenario.centers[i0:i1],
            radii=scenario.radii[i0:i1],
            mode=scenario.mode[i0:i1],
        )
        x_seq, _, _ = rollout_gate(
            controller=controller,
            plant_true=plant_true,
            scenario=sc,
            horizon=horizon,
            device=device,
            noise=None,
            z_gain=z_gain,
        )
        ch = gate_choice_from_trajectory(x_seq, gate_x=gate_x_eval)
        md = min_dist_to_edge(x_seq, sc).min(dim=1).values
        td = torch.norm(x_seq[:, -1, :2], dim=-1)
        choice_top[i0:i1] = ch.float()
        collided[i0:i1] = (md < 0.0).float()
        term_dist[i0:i1] = td

    choice_map = choice_top.view(res, res).detach().cpu().numpy()
    coll_map = collided.view(res, res).detach().cpu().numpy()
    term_map = term_dist.view(res, res).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    extent = [r_min, r_max, r_min, r_max]  # x: r_bottom, y: r_top

    im0 = axes[0].imshow(choice_map, origin="lower", extent=extent, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=1.0)
    axes[0].set_title("Chosen Corridor (1=Top, 0=Bottom)")
    axes[0].set_xlabel("r_bottom")
    axes[0].set_ylabel("r_top")
    axes[0].plot([r_min, r_max], [r_min, r_max], "k--", linewidth=1.0, alpha=0.7)
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].imshow(coll_map, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Collision Map")
    axes[1].set_xlabel("r_bottom")
    axes[1].set_ylabel("r_top")
    axes[1].plot([r_min, r_max], [r_min, r_max], "w--", linewidth=1.0, alpha=0.7)
    fig.colorbar(im1, ax=axes[1], shrink=0.9)

    im2 = axes[2].imshow(term_map, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[2].set_title("Terminal Distance to Goal")
    axes[2].set_xlabel("r_bottom")
    axes[2].set_ylabel("r_top")
    axes[2].plot([r_min, r_max], [r_min, r_max], "w--", linewidth=1.0, alpha=0.7)
    fig.colorbar(im2, ax=axes[2], shrink=0.9)

    fig.suptitle("Context Phase Maps (single start, varying top/bottom radii)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "context_phase_maps.png"), bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


def plot_ablation_bars(metrics_true, metrics_shuffled, metrics_zero, metrics_mp_only, run_dir, show_plots: bool):
    plt = get_plt(show_plots)
    labels = ["true context", "shuffled context", "zero context", "mp only"]
    success = [
        float(metrics_true["success_rate"]),
        float(metrics_shuffled["success_rate"]),
        float(metrics_zero["success_rate"]),
        float(metrics_mp_only["success_rate"]),
    ]
    collision = [
        float(metrics_true["collision_rate"]),
        float(metrics_shuffled["collision_rate"]),
        float(metrics_zero["collision_rate"]),
        float(metrics_mp_only["collision_rate"]),
    ]
    choice = [
        float(metrics_true["choice_accuracy"]) if metrics_true["choice_count"] > 0 else float("nan"),
        float(metrics_shuffled["choice_accuracy"]) if metrics_shuffled["choice_count"] > 0 else float("nan"),
        float(metrics_zero["choice_accuracy"]) if metrics_zero["choice_count"] > 0 else float("nan"),
        float(metrics_mp_only["choice_accuracy"]) if metrics_mp_only["choice_count"] > 0 else float("nan"),
    ]

    x = torch.arange(len(labels)).cpu().numpy()
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, success, width=width, label="success rate", color="C2")
    ax.bar(x, collision, width=width, label="collision rate", color="C3")
    ax.bar(x + width, choice, width=width, label="choice accuracy", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Context Ablation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "ablation_context_bars.png"))
    if not show_plots:
        plt.close(fig)


class MpOnlyOperator(nn.Module):
    """Standalone M_p(w) controller with a learned bounded readout to u."""

    def __init__(self, mp: nn.Module, feat_dim: int, u_dim: int, u_bound: float = 10.0):
        super().__init__()
        self.mp = mp
        self.readout = nn.Linear(feat_dim, u_dim, bias=False)
        self.u_bound = float(u_bound)
        if self.u_bound <= 0:
            raise ValueError(f"u_bound must be > 0, got {self.u_bound}")

    def reset(self) -> None:
        if hasattr(self.mp, "reset"):
            self.mp.reset()

    def forward(self, w: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        v = self.mp(w)
        u = self.readout(v)
        return self.u_bound * torch.tanh(u / self.u_bound)


def build_controller(
    device: torch.device,
    *,
    train_mode: str,
    z_residual_gain: float,
    nom_pre_kp: float,
    nom_pre_kd: float,
    mp_u_bound: float,
    mp_context_lift: bool,
    mp_context_lift_type: str,
    mp_context_lift_dim: int,
    mp_context_hidden_dim: int,
    mp_context_decay_law: str,
    mp_context_decay_rate: float,
    mp_context_decay_power: float,
    mp_context_decay_horizon: int,
    mp_context_lp_p: float,
    mp_context_scale: float,
) -> PBController:
    w_dim = 4
    u_dim = 2
    feat_dim = 16
    z_dim = 5  # [r_top, r_center, r_bottom, gap_top, gap_bottom]

    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=nom_pre_kp, pre_kd=nom_pre_kd)
    mp_context_lifter = None
    mp_in_dim = w_dim
    if train_mode == "factorized" and bool(mp_context_lift):
        mp_context_lifter = LpContextLifter(
            z_dim=z_dim,
            out_dim=int(mp_context_lift_dim),
            lift_type=mp_context_lift_type,
            hidden_dim=int(mp_context_hidden_dim),
            decay_law=mp_context_decay_law,
            decay_rate=float(mp_context_decay_rate),
            decay_power=float(mp_context_decay_power),
            decay_horizon=int(mp_context_decay_horizon),
            lp_p=float(mp_context_lp_p),
            scale=float(mp_context_scale),
        ).to(device)
        mp_in_dim = w_dim + int(mp_context_lift_dim)

    mp = MpDeepSSM(
        mp_in_dim,
        feat_dim,
        mode="loop",
        param="lru",
        n_layers=4,
        d_model=16,
        d_state=32,
        ff="GLU",
    ).to(device)
    if train_mode == "factorized":
        mb = BoundedMLPOperator(
            w_dim=w_dim,
            z_dim=z_dim,
            r=u_dim,
            s=feat_dim,
            hidden_dim=64,
            use_z_residual=True,
            z_residual_gain=float(z_residual_gain),
            bound_mode="softsign",
            clamp_value=10.0,
        ).to(device)
        return build_factorized_controller(
            nominal_plant=plant_nom,
            mp=mp,
            mb=mb,
            u_dim=u_dim,
            detach_state=False,
            u_nominal=None,
            mp_context_lifter=mp_context_lifter,
        ).to(device)
    elif train_mode == "mp_only":
        operator = MpOnlyOperator(mp=mp, feat_dim=feat_dim, u_dim=u_dim, u_bound=float(mp_u_bound)).to(device)
    else:
        raise ValueError(f"Unknown train_mode={train_mode!r}")

    return PBController(
        plant=plant_nom,
        operator=operator,
        u_nominal=None,
        u_dim=u_dim,
        detach_state=False,
    ).to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    show_plots = not args.no_show_plots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_horizon = int(args.val_horizon) if int(args.val_horizon) > 0 else int(args.horizon)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        "gate_experiment",
        f"gate_exp_{run_id}",
    )
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    loss_cfg = LossConfig(
        w_term=args.w_term,
        w_stage=args.w_stage,
        stage_w_start=args.stage_w_start,
        stage_w_end=args.stage_w_end,
        w_bar=args.w_bar,
        w_wall=args.w_wall,
        vel_dist_weight=args.vel_dist_weight,
        bar_margin=args.bar_margin,
        bar_alpha=args.bar_alpha,
        bar_cap=args.bar_cap,
        wall_y_limit=args.wall_y_limit,
        wall_margin=args.wall_margin,
        wall_alpha=args.wall_alpha,
        wall_cap=args.wall_cap,
    )
    train_noise_model = DecayingGaussianNoise(
        sigma0=float(args.noise_sigma0),
        tau=float(args.noise_tau),
    )

    centers = fixed_gate_centers(center_x=args.center_x, y_sep=args.y_sep)
    controller = build_controller(
        device=device,
        train_mode=args.train_mode,
        z_residual_gain=args.z_residual_gain,
        nom_pre_kp=args.nom_pre_kp,
        nom_pre_kd=args.nom_pre_kd,
        mp_u_bound=args.mp_u_bound,
        mp_context_lift=bool(args.mp_context_lift),
        mp_context_lift_type=args.mp_context_lift_type,
        mp_context_lift_dim=args.mp_context_lift_dim,
        mp_context_hidden_dim=args.mp_context_hidden_dim,
        mp_context_decay_law=args.mp_context_decay_law,
        mp_context_decay_rate=args.mp_context_decay_rate,
        mp_context_decay_power=args.mp_context_decay_power,
        mp_context_decay_horizon=args.mp_context_decay_horizon,
        mp_context_lp_p=args.mp_context_lp_p,
        mp_context_scale=args.mp_context_scale,
    )
    plant_true = DoubleIntegratorTrue(dt=0.05, pre_kp=args.true_pre_kp, pre_kd=args.true_pre_kd)

    probe_scenario = sample_dataset(
        batch_size=min(8, int(args.val_batch)),
        seed=int(args.seed) + 12345,
        args=args,
        centers=centers,
        paired_context=False,
        mirror_augment=False,
    )
    probe_scenario = scenario_to_device(probe_scenario, device)
    x_probe = make_x0(probe_scenario, device)
    z_probe = _canonical_context_vector(probe_scenario, z_gain=args.z_gain).to(device).unsqueeze(1)
    ok, msg = validate_component_compatibility(
        controller=controller,
        plant_true=plant_true,
        x0=x_probe,
        z0=z_probe,
        raise_on_error=False,
    )
    if not ok:
        raise RuntimeError(f"PB component compatibility check failed: {msg}")

    optimizer = optim.Adam(controller.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    val_scenario_cpu = sample_dataset(
        batch_size=args.val_batch,
        seed=args.seed + 999,
        args=args,
        centers=centers,
        paired_context=False,
        mirror_augment=False,
    )
    val_scenario = scenario_to_device(val_scenario_cpu, device)

    print(f"Starting gate experiment on {device}")
    print(f"Train horizon={args.horizon}, Val horizon={val_horizon}")
    train_hist: List[dict] = []
    eval_hist: List[dict] = []

    best_ckpt_path = os.path.join(run_dir, "best_model.pt")
    init_metrics, _, _ = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=val_horizon,
        device=device,
        loss_cfg=loss_cfg,
        success_tol=args.success_tol,
        choice_gap_margin=args.choice_gap_margin,
        gate_x_eval=args.gate_x_eval,
        context_mode="true",
        z_gain=args.z_gain,
    )
    init_metrics["epoch"] = 0
    eval_hist.append(init_metrics)
    best_epoch = 0
    best_loss = float(init_metrics["loss"])
    best_collision = float(init_metrics["collision_rate"])
    best_success = float(init_metrics["success_rate"])
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": controller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_collision": best_collision,
            "best_success": best_success,
            "args": vars(args),
        },
        best_ckpt_path,
    )
    print(
        f"Epoch 000/{args.epochs} | Val {init_metrics['loss']:.2f} | "
        f"Crash {init_metrics['collision_rate'] * 100:.1f}% | "
        f"Success {init_metrics['success_rate'] * 100:.1f}% | "
        f"TopSuc {init_metrics['top_success_rate'] * 100.0:.1f}% | "
        f"BotSuc {init_metrics['bottom_success_rate'] * 100.0:.1f}% | "
        f"ChoiceAcc "
        f"{(init_metrics['choice_accuracy'] * 100.0 if init_metrics['choice_count'] > 0 else float('nan')):.1f}%"
    )

    for epoch in range(1, args.epochs + 1):
        controller.train()
        optimizer.zero_grad()

        train_scenario_cpu = sample_dataset(
            batch_size=args.batch,
            seed=args.seed + epoch,
            args=args,
            centers=centers,
            paired_context=bool(args.paired_train_context),
            mirror_augment=bool(args.mirror_augment_train),
        )
        train_scenario = scenario_to_device(train_scenario_cpu, device)

        noise = train_noise_model.sample(
            bsz=args.batch,
            horizon=args.horizon,
            nx=4,
            device=device,
            seed=int(args.seed) + 40000 + epoch,
        )
        x_seq, u_train_seq, w_train_seq = rollout_gate(
            controller=controller,
            plant_true=plant_true,
            scenario=train_scenario,
            horizon=args.horizon,
            device=device,
            noise=noise,
            z_gain=args.z_gain,
        )
        loss, train_parts = compute_loss(x_seq, train_scenario, loss_cfg)
        tail_k_train = min(10, u_train_seq.shape[1])
        train_w_last = float(torch.norm(w_train_seq[:, -1, :], dim=-1).mean().item())
        train_w_tail = float(torch.norm(w_train_seq[:, -tail_k_train:, :], dim=-1).mean().item())
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        scheduler.step()
        train_hist.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "w_last_mag": train_w_last,
                "w_tail_mag": train_w_tail,
            }
        )
        if epoch % args.eval_every == 0:
            metrics, _, _ = evaluate(
                controller=controller,
                plant_true=plant_true,
                scenario=val_scenario,
                horizon=val_horizon,
                device=device,
                loss_cfg=loss_cfg,
                success_tol=args.success_tol,
                choice_gap_margin=args.choice_gap_margin,
                gate_x_eval=args.gate_x_eval,
                context_mode="true",
                z_gain=args.z_gain,
            )
            metrics["epoch"] = epoch
            eval_hist.append(metrics)

            if args.best_ckpt_metric == "loss":
                better = metrics["loss"] < best_loss
            elif args.best_ckpt_metric == "success_then_loss":
                better = (
                    metrics["success_rate"] > best_success + 1e-9
                    or (abs(metrics["success_rate"] - best_success) <= 1e-9 and metrics["loss"] < best_loss)
                )
            else:
                tol = float(args.best_ckpt_collision_tol)
                better = (
                    metrics["collision_rate"] < (best_collision - tol)
                    or (abs(metrics["collision_rate"] - best_collision) <= tol and metrics["loss"] < best_loss)
                )

            if better:
                best_epoch = epoch
                best_loss = float(metrics["loss"])
                best_collision = float(metrics["collision_rate"])
                best_success = float(metrics["success_rate"])
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": controller.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "best_collision": best_collision,
                        "best_success": best_success,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train {train_parts['loss_total']:.2f} | "
                f"Val {metrics['loss']:.2f} | "
                f"Crash {metrics['collision_rate'] * 100:.1f}% | "
                f"Success {metrics['success_rate'] * 100:.1f}% | "
                f"TopSuc {metrics['top_success_rate'] * 100.0:.1f}% | "
                f"BotSuc {metrics['bottom_success_rate'] * 100.0:.1f}% | "
                f"ChoiceAcc "
                f"{(metrics['choice_accuracy'] * 100.0 if metrics['choice_count'] > 0 else float('nan')):.1f}% | "
                f"LR {scheduler.get_last_lr()[0]:.1e}"
            )

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        controller.load_state_dict(ckpt["model_state_dict"])

    # Final evaluation.
    metrics_true, x_val, u_val = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=val_horizon,
        device=device,
        loss_cfg=loss_cfg,
        success_tol=args.success_tol,
        choice_gap_margin=args.choice_gap_margin,
        gate_x_eval=args.gate_x_eval,
        context_mode="true",
        z_gain=args.z_gain,
    )

    metrics_final = {
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_collision": float(best_collision),
        "best_success": float(best_success),
        "best_ckpt_metric": args.best_ckpt_metric,
        "final_true_context": metrics_true,
    }

    metrics_shuffled = None
    metrics_zero = None
    metrics_mp_only = None
    x_mp_only = None
    u_mp_only = None
    if not args.skip_ablations:
        metrics_shuffled, _, _ = evaluate(
            controller=controller,
            plant_true=plant_true,
            scenario=val_scenario,
            horizon=val_horizon,
            device=device,
            loss_cfg=loss_cfg,
            success_tol=args.success_tol,
            choice_gap_margin=args.choice_gap_margin,
            gate_x_eval=args.gate_x_eval,
            context_mode="shuffled",
            z_gain=args.z_gain,
        )
        metrics_zero, _, _ = evaluate(
            controller=controller,
            plant_true=plant_true,
            scenario=val_scenario,
            horizon=val_horizon,
            device=device,
            loss_cfg=loss_cfg,
            success_tol=args.success_tol,
            choice_gap_margin=args.choice_gap_margin,
            gate_x_eval=args.gate_x_eval,
            context_mode="zero",
            z_gain=args.z_gain,
        )
        metrics_mp_only, x_mp_only, u_mp_only = evaluate(
            controller=controller,
            plant_true=plant_true,
            scenario=val_scenario,
            horizon=val_horizon,
            device=device,
            loss_cfg=loss_cfg,
            success_tol=args.success_tol,
            choice_gap_margin=args.choice_gap_margin,
            gate_x_eval=args.gate_x_eval,
            context_mode="mp_only",
            z_gain=args.z_gain,
        )
        metrics_final["final_shuffled_context"] = metrics_shuffled
        metrics_final["final_zero_context"] = metrics_zero
        metrics_final["final_mp_only"] = metrics_mp_only

    if args.noisy_test_mc > 0 and args.noisy_test_sigma0 > 0.0:
        noisy_cfg = DecayingGaussianNoise(
            sigma0=float(args.noisy_test_sigma0),
            tau=float(args.noisy_test_tau),
        )
        noisy_true, _, _ = evaluate_noisy_mc(
            controller=controller,
            plant_true=plant_true,
            scenario=val_scenario,
            horizon=val_horizon,
            device=device,
            loss_cfg=loss_cfg,
            success_tol=args.success_tol,
            choice_gap_margin=args.choice_gap_margin,
            gate_x_eval=args.gate_x_eval,
            context_mode="true",
            z_gain=args.z_gain,
            noise_model=noisy_cfg,
            mc_trials=int(args.noisy_test_mc),
            seed_base=int(args.seed) + 50000,
        )
        metrics_final["final_true_context_noisy_mc"] = noisy_true
        m = noisy_true["mean"]
        s = noisy_true["std"]
        print(
            f"NoisyTest (true) | MC {int(args.noisy_test_mc)} | "
            f"sigma0 {float(args.noisy_test_sigma0):.3g} tau {float(args.noisy_test_tau):.1f} | "
            f"Crash {100.0 * float(m['collision_rate']):.1f}% ± {100.0 * float(s['collision_rate']):.1f}% | "
            f"Success {100.0 * float(m['success_rate']):.1f}% ± {100.0 * float(s['success_rate']):.1f}%"
        )

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_final, f, indent=2)
    with open(os.path.join(run_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_hist, f, indent=2)
    with open(os.path.join(run_dir, "eval_history.json"), "w", encoding="utf-8") as f:
        json.dump(eval_hist, f, indent=2)

    # Plots.
    plot_loss_curves(train_hist, eval_hist, run_dir, show_plots=show_plots)
    plot_control_magnitude(
        u_val,
        run_dir,
        show_plots=show_plots,
        out_name="control_magnitude_true.png",
        fig_title="Control Magnitude Over Time (true context)",
    )
    if u_mp_only is not None:
        plot_control_magnitude(
            u_mp_only,
            run_dir,
            show_plots=show_plots,
            out_name="control_magnitude_mp_only.png",
            fig_title="Control Magnitude Over Time (M_p only)",
        )
    plot_trajectories_by_mode(
        x_val,
        val_scenario,
        run_dir,
        show_plots=show_plots,
        max_each=args.traj_plot_each,
    )
    if x_mp_only is not None:
        plot_trajectories_by_mode(
            x_mp_only,
            val_scenario,
            run_dir,
            show_plots=show_plots,
            max_each=args.traj_plot_each,
            out_name="trajectories_mp_only.png",
            fig_title="Trajectories with M_p(w) only",
        )
    plot_context_switch_overlay(
        controller=controller,
        plant_true=plant_true,
        ref_scenario=val_scenario,
        horizon=val_horizon,
        device=device,
        run_dir=run_dir,
        show_plots=show_plots,
        num_starts=args.overlay_num_starts,
        open_radius_ref=0.5 * (args.open_r_min + args.open_r_max),
        closed_radius_ref=0.5 * (args.closed_r_min + args.closed_r_max),
        r_center=args.r_center,
        z_gain=args.z_gain,
        out_name="context_switch_overlay.png",
        fig_title="Context Reactivity: Same Start, Different Radii Context",
    )
    amb_open = max(1e-3, float(args.ambig_r_base) - float(args.ambig_r_delta))
    amb_closed = max(1e-3, float(args.ambig_r_base) + float(args.ambig_r_delta))
    plot_context_switch_overlay(
        controller=controller,
        plant_true=plant_true,
        ref_scenario=val_scenario,
        horizon=val_horizon,
        device=device,
        run_dir=run_dir,
        show_plots=show_plots,
        num_starts=args.overlay_num_starts,
        open_radius_ref=amb_open,
        closed_radius_ref=amb_closed,
        r_center=args.r_center,
        z_gain=args.z_gain,
        out_name="context_switch_overlay_ambiguous.png",
        fig_title="Near-Ambiguous Context Switch (small gap difference)",
    )
    plot_context_phase_maps(
        controller=controller,
        plant_true=plant_true,
        horizon=val_horizon,
        device=device,
        run_dir=run_dir,
        show_plots=show_plots,
        center_x=args.center_x,
        y_sep=args.y_sep,
        r_center=args.r_center,
        start_x=args.phase_start_x,
        start_y=args.phase_start_y,
        r_min=args.open_r_min,
        r_max=args.closed_r_max,
        res=args.phase_res,
        batch=args.phase_batch,
        gate_x_eval=args.gate_x_eval,
        z_gain=args.z_gain,
    )
    if (
        (not args.skip_ablations)
        and (metrics_shuffled is not None)
        and (metrics_zero is not None)
        and (metrics_mp_only is not None)
    ):
        plot_ablation_bars(
            metrics_true,
            metrics_shuffled,
            metrics_zero,
            metrics_mp_only,
            run_dir,
            show_plots=show_plots,
        )

    torch.save(controller.state_dict(), os.path.join(run_dir, "pb_model_final.pt"))
    print(f"Done. Artifacts saved to {run_dir}")

    if show_plots:
        plt = get_plt(True)
        plt.show()


if __name__ == "__main__":
    main()
