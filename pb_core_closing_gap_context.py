"""
Contextual PB experiment: a central gap that closes over time.

Why this setup is useful:
  - Two obstacles form a middle passage.
  - Their radii grow over time, so the middle gap may close early or late.
  - With the right context, the controller can decide:
      * pass through the middle quickly (late-closing case),
      * or detour around obstacles (early-closing case).

Controller:
  M(w,z) = M_b(w,z) x M_p(w)
  - M_p is MpDeepSSM (SSM disturbance processor).
  - M_b is a bounded MLP mixer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from bounded_mlp_operator import BoundedMLPOperator
from context_lifting import LpContextLifter
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_controller import PBController, as_bt
from pb_core import DecayingGaussianNoise, rollout_pb, validate_component_compatibility
from pb_core.factories import build_factorized_controller
from ssm_operators import MpDeepSSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PB closing-gap contextual experiment")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--val_batch", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--val_horizon", type=int, default=140)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--no_show_plots", action="store_true")

    # Start distribution.
    parser.add_argument("--start_x_min", type=float, default=1.7)
    parser.add_argument("--start_x_max", type=float, default=2.4)
    parser.add_argument("--start_y_min", type=float, default=-0.12)
    parser.add_argument("--start_y_max", type=float, default=0.12)
    parser.add_argument("--start_clearance", type=float, default=0.10)

    # Geometry (two obstacles).
    parser.add_argument("--center_x", type=float, default=0.95)
    parser.add_argument("--center_y_sep", type=float, default=0.42)

    # Radius profile.
    parser.add_argument("--r_min", type=float, default=0.16)
    parser.add_argument("--r_max", type=float, default=0.34)
    parser.add_argument("--sigmoid_alpha", type=float, default=12.0)
    parser.add_argument("--t_mid_early_min", type=float, default=0.18)
    parser.add_argument("--t_mid_early_max", type=float, default=0.34)
    parser.add_argument("--t_mid_late_min", type=float, default=0.62)
    parser.add_argument("--t_mid_late_max", type=float, default=0.80)
    parser.add_argument("--asym_amp", type=float, default=0.008)
    parser.add_argument("--paired_train_context", dest="paired_train_context", action="store_true")
    parser.add_argument("--no_paired_train_context", dest="paired_train_context", action="store_false")
    parser.set_defaults(paired_train_context=True)

    # Context scaling.
    parser.add_argument("--z_scale", type=float, default=8.0)

    # Loss: distance + barrier only.
    parser.add_argument("--w_term", type=float, default=34.0)
    parser.add_argument("--w_stage", type=float, default=8.0)
    parser.add_argument("--w_bar_soft", type=float, default=65.0)
    parser.add_argument("--w_bar_hard", type=float, default=130.0)
    parser.add_argument("--bar_margin", type=float, default=0.10)
    parser.add_argument("--bar_beta", type=float, default=16.0)
    parser.add_argument("--success_tol", type=float, default=0.18)

    # Process noise (set sigma0=0 to disable).
    parser.add_argument("--noise_sigma0", type=float, default=0.004)
    parser.add_argument("--noise_tau", type=float, default=24.0)

    # Controller architecture.
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--mb_hidden", type=int, default=64)
    parser.add_argument("--mb_layers", type=int, default=4)
    parser.add_argument("--z_residual_gain", type=float, default=10.0)
    parser.add_argument("--mb_bound", type=float, default=10.0)
    parser.add_argument("--ssm_param", type=str, default="lru", choices=["lru", "tv"])

    # Optional augmentation: M_p sees [w, z_lift].
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=True)
    parser.add_argument("--mp_context_lift_type", type=str, default="linear", choices=["identity", "linear", "mlp"])
    parser.add_argument("--mp_context_lift_dim", type=int, default=8)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=32)
    parser.add_argument("--mp_context_decay_law", type=str, default="poly", choices=["exp", "poly", "finite"])
    parser.add_argument("--mp_context_decay_rate", type=float, default=0.08)
    parser.add_argument("--mp_context_decay_power", type=float, default=1.2)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=110)
    parser.add_argument("--mp_context_lp_p", type=float, default=2.0)
    parser.add_argument("--mp_context_scale", type=float, default=1.0)

    # Plot controls.
    parser.add_argument("--traj_plot_each", type=int, default=4)
    parser.add_argument("--center_pass_band", type=float, default=0.18)
    return parser.parse_args()


@dataclass
class ClosingGapScenario:
    """
    Batch scenario for the closing-gap task.

    mode:
      0 -> late-closing gap (middle passage should stay usable longer)
      1 -> early-closing gap (controller should avoid center later)
    """

    start: torch.Tensor  # (B,2)
    goal: torch.Tensor  # (B,2)
    centers: torch.Tensor  # (B,2,2) (top,bottom)
    radii_seq: torch.Tensor  # (B,T,2)
    mode: torch.Tensor  # (B,)
    t_mid: torch.Tensor  # (B,)


@dataclass
class LossConfig:
    w_term: float
    w_stage: float
    w_bar_soft: float
    w_bar_hard: float
    bar_margin: float
    bar_beta: float


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


def fixed_centers(center_x: float, y_sep: float) -> torch.Tensor:
    return torch.tensor(
        [[center_x, +y_sep], [center_x, -y_sep]],
        dtype=torch.float32,
    )


def _radius_profile(
    *,
    horizon: int,
    r_min: float,
    r_max: float,
    alpha: float,
    t_mid: float,
    asym_amp: float,
    phase: float,
) -> torch.Tensor:
    """
    Time-varying radii profile for top/bottom obstacles.

    The common radius follows a sigmoid in normalized time; top and bottom get a tiny
    opposite sinusoidal asymmetry to break perfect symmetry.
    """
    tau = torch.linspace(0.0, 1.0, horizon)
    common = float(r_min) + (float(r_max) - float(r_min)) * torch.sigmoid(float(alpha) * (tau - float(t_mid)))
    asym = float(asym_amp) * torch.sin(2.0 * math.pi * tau + float(phase))
    r_top = torch.clamp(common + asym, min=float(r_min), max=float(r_max))
    r_bot = torch.clamp(common - asym, min=float(r_min), max=float(r_max))
    return torch.stack([r_top, r_bot], dim=-1)  # (T,2)


def _sample_t_mid_for_mode(mode: int, rng: np.random.RandomState, args: argparse.Namespace) -> float:
    if mode == 1:
        return float(rng.uniform(float(args.t_mid_early_min), float(args.t_mid_early_max)))
    return float(rng.uniform(float(args.t_mid_late_min), float(args.t_mid_late_max)))


def sample_scenario(
    *,
    batch_size: int,
    horizon: int,
    seed: int,
    args: argparse.Namespace,
    centers_fixed: torch.Tensor,
    paired_context: bool,
) -> ClosingGapScenario:
    """
    Sample training/validation batch.

    Paired mode:
      duplicate starts with both modes (early and late closure), which strongly
      emphasizes contextual dependence.
    """
    rng = np.random.RandomState(int(seed))
    start = torch.zeros(batch_size, 2, dtype=torch.float32)
    goal = torch.zeros(batch_size, 2, dtype=torch.float32)
    centers = centers_fixed.view(1, 2, 2).repeat(batch_size, 1, 1).clone()
    radii_seq = torch.zeros(batch_size, horizon, 2, dtype=torch.float32)
    mode = torch.zeros(batch_size, dtype=torch.long)
    t_mid = torch.zeros(batch_size, dtype=torch.float32)

    if paired_context:
        half = batch_size // 2
        rem = batch_size - 2 * half
        starts_base = torch.zeros(half, 2, dtype=torch.float32)
        starts_base[:, 0] = torch.from_numpy(rng.uniform(float(args.start_x_min), float(args.start_x_max), size=half).astype(np.float32))
        starts_base[:, 1] = torch.from_numpy(rng.uniform(float(args.start_y_min), float(args.start_y_max), size=half).astype(np.float32))

        # first half late (0), second half early (1)
        if half > 0:
            start[:half] = starts_base
            mode[:half] = 0
            start[half : 2 * half] = starts_base
            mode[half : 2 * half] = 1
        if rem > 0:
            start[2 * half :, 0] = torch.from_numpy(
                rng.uniform(float(args.start_x_min), float(args.start_x_max), size=rem).astype(np.float32)
            )
            start[2 * half :, 1] = torch.from_numpy(
                rng.uniform(float(args.start_y_min), float(args.start_y_max), size=rem).astype(np.float32)
            )
            mode[2 * half :] = torch.from_numpy(rng.randint(0, 2, size=rem).astype(np.int64))
    else:
        start[:, 0] = torch.from_numpy(rng.uniform(float(args.start_x_min), float(args.start_x_max), size=batch_size).astype(np.float32))
        start[:, 1] = torch.from_numpy(rng.uniform(float(args.start_y_min), float(args.start_y_max), size=batch_size).astype(np.float32))
        mode = torch.from_numpy(rng.randint(0, 2, size=batch_size).astype(np.int64))

    # Build time-varying radii and enforce feasible starts at t=0.
    for b in range(batch_size):
        m = int(mode[b].item())
        tmid_b = _sample_t_mid_for_mode(m, rng, args)
        phase_b = float(rng.uniform(0.0, 2.0 * math.pi))
        rs = _radius_profile(
            horizon=horizon,
            r_min=float(args.r_min),
            r_max=float(args.r_max),
            alpha=float(args.sigmoid_alpha),
            t_mid=tmid_b,
            asym_amp=float(args.asym_amp),
            phase=phase_b,
        )

        # Rejection sample to keep start outside obstacle bodies at t=0.
        ok = False
        for _ in range(250):
            sx = float(rng.uniform(float(args.start_x_min), float(args.start_x_max)))
            sy = float(rng.uniform(float(args.start_y_min), float(args.start_y_max)))
            s = torch.tensor([sx, sy], dtype=torch.float32)
            d0 = torch.norm(s.view(1, 2) - centers_fixed, dim=-1) - rs[0]
            if bool((d0 > float(args.start_clearance)).all().item()):
                start[b] = s
                ok = True
                break
        if not ok:
            raise RuntimeError("Failed to sample a feasible start; relax start_clearance.")

        radii_seq[b] = rs
        t_mid[b] = float(tmid_b)

    return ClosingGapScenario(
        start=start,
        goal=goal,
        centers=centers,
        radii_seq=radii_seq,
        mode=mode,
        t_mid=t_mid,
    )


def scenario_to_device(s: ClosingGapScenario, device: torch.device) -> ClosingGapScenario:
    return ClosingGapScenario(
        start=s.start.to(device),
        goal=s.goal.to(device),
        centers=s.centers.to(device),
        radii_seq=s.radii_seq.to(device),
        mode=s.mode.to(device),
        t_mid=s.t_mid.to(device),
    )


def shuffled_context_scenario(s: ClosingGapScenario) -> ClosingGapScenario:
    bsz = s.start.shape[0]
    perm = torch.randperm(bsz, device=s.start.device)
    return ClosingGapScenario(
        start=s.start,
        goal=s.goal,
        centers=s.centers,
        radii_seq=s.radii_seq[perm],
        mode=s.mode[perm],
        t_mid=s.t_mid[perm],
    )


def make_x0(s: ClosingGapScenario, device: torch.device) -> torch.Tensor:
    vel0 = torch.zeros(s.start.shape[0], 2, device=device)
    return torch.cat([s.start.to(device), vel0], dim=-1).unsqueeze(1)


def infer_context_dim() -> int:
    # goal_dir(2) + rel(2*2) + dist_edge(2) + radii_t(2) + gap_t(1) + t_mid(1) + tau(1) = 13
    return 13


def build_context(
    x_t: torch.Tensor,
    s_ctx: ClosingGapScenario,
    *,
    t: int,
    horizon: int,
    z_scale: float,
) -> torch.Tensor:
    """
    Context z_t for this task.

    Includes a compact predictive cue:
      - t_mid (closure timing parameter), constant per sample.
    """
    x_t = as_bt(x_t)
    pos = x_t[..., :2]  # (B,1,2)
    goal = s_ctx.goal.to(x_t.device).unsqueeze(1)  # (B,1,2)
    centers = s_ctx.centers.to(x_t.device)  # (B,2,2)
    radii_t = s_ctx.radii_seq.to(x_t.device)[:, t : t + 1, :]  # (B,1,2)

    rel = centers.unsqueeze(1) - pos.unsqueeze(2)  # (B,1,2,2)
    dist_edge = torch.norm(rel, dim=-1) - radii_t  # (B,1,2)
    goal_dir = goal - pos
    gap_t = (2.0 * torch.abs(centers[:, 0, 1]).view(-1, 1, 1)) - radii_t.sum(dim=-1, keepdim=True)  # (B,1,1)
    tmid = s_ctx.t_mid.to(x_t.device).view(-1, 1, 1)  # (B,1,1)
    tau = torch.full_like(tmid, float(t) / max(horizon - 1, 1))

    z_t = torch.cat(
        [
            goal_dir,
            rel.reshape(rel.shape[0], rel.shape[1], -1),
            dist_edge,
            radii_t,
            gap_t,
            tmid,
            tau,
        ],
        dim=-1,
    )
    return float(z_scale) * z_t


def obstacle_edge_distances(x_seq: torch.Tensor, s: ClosingGapScenario) -> torch.Tensor:
    x_seq = as_bt(x_seq)
    pos = x_seq[..., :2]  # (B,T,2)
    centers = s.centers.to(x_seq.device).unsqueeze(1)  # (B,1,2,2)
    radii = s.radii_seq.to(x_seq.device)  # (B,T,2)
    dist = torch.norm(pos.unsqueeze(2) - centers, dim=-1)  # (B,T,2)
    return dist - radii


def compute_loss_per_sample(x_seq: torch.Tensor, s: ClosingGapScenario, cfg: LossConfig) -> torch.Tensor:
    goal = s.goal.to(x_seq.device).unsqueeze(1)
    dist_goal = torch.norm(x_seq[..., :2] - goal, dim=-1)
    term = dist_goal[:, -1] * cfg.w_term
    stage = dist_goal.mean(dim=1) * cfg.w_stage

    d_edge = obstacle_edge_distances(x_seq, s)
    if cfg.bar_beta <= 0:
        raise ValueError(f"bar_beta must be > 0, got {cfg.bar_beta}")
    soft = (F.softplus(cfg.bar_beta * (cfg.bar_margin - d_edge)) / cfg.bar_beta).sum(dim=-1)
    hard = (torch.relu(-d_edge) ** 2).sum(dim=-1)
    bar_soft = soft.mean(dim=1) * cfg.w_bar_soft
    bar_hard = hard.mean(dim=1) * cfg.w_bar_hard
    return term + stage + bar_soft + bar_hard


def compute_loss(x_seq: torch.Tensor, s: ClosingGapScenario, cfg: LossConfig) -> tuple[torch.Tensor, dict]:
    per = compute_loss_per_sample(x_seq, s, cfg)
    return per.mean(), {"loss_total": float(per.mean().item())}


def rollout_scenario(
    *,
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario_phys: ClosingGapScenario,
    scenario_ctx: ClosingGapScenario,
    horizon: int,
    device: torch.device,
    z_scale: float,
    zero_context: bool,
    noise: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_phys = scenario_to_device(scenario_phys, device)
    s_ctx = scenario_to_device(scenario_ctx, device)
    x0 = make_x0(s_phys, device)
    z_dim = infer_context_dim()

    if noise is not None:
        if noise.shape[0] != x0.shape[0] or noise.shape[1] != horizon or noise.shape[2] != x0.shape[2]:
            raise ValueError(
                "noise must have shape (B,horizon,Nx). "
                f"Got {tuple(noise.shape)}, expected ({x0.shape[0]},{horizon},{x0.shape[2]})"
            )

    def _ctx_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
        if zero_context:
            return torch.zeros(x_t.shape[0], 1, z_dim, device=x_t.device, dtype=x_t.dtype)
        return build_context(x_t, s_ctx, t=t, horizon=horizon, z_scale=float(z_scale))

    out = rollout_pb(
        controller=controller,
        plant_true=plant_true,
        x0=x0,
        horizon=horizon,
        context_fn=_ctx_fn,
        w0=x0,
        process_noise_seq=noise,
    )
    return out.x_seq, out.u_seq, out.w_seq


def through_center_choice(x_seq: torch.Tensor, center_x: float, band: float) -> torch.Tensor:
    """
    Decide whether trajectory passed through central corridor.
    We inspect y when x is closest to obstacle x-location.
    """
    x = x_seq[..., 0]
    y = x_seq[..., 1]
    idx = torch.argmin(torch.abs(x - float(center_x)), dim=1)  # (B,)
    y_sel = y[torch.arange(x.shape[0], device=x.device), idx]
    return torch.abs(y_sel) <= float(band)


@torch.no_grad()
def evaluate(
    *,
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: ClosingGapScenario,
    horizon: int,
    device: torch.device,
    loss_cfg: LossConfig,
    success_tol: float,
    z_scale: float,
    center_x: float,
    center_pass_band: float,
    context_mode: str = "true",
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    if context_mode not in {"true", "shuffled", "zero"}:
        raise ValueError(f"Unknown context_mode {context_mode!r}")

    if context_mode == "true":
        ctx_s = scenario
        zero_ctx = False
    elif context_mode == "shuffled":
        ctx_s = shuffled_context_scenario(scenario_to_device(scenario, device))
        zero_ctx = False
    else:
        ctx_s = scenario
        zero_ctx = True

    x_seq, u_seq, w_seq = rollout_scenario(
        controller=controller,
        plant_true=plant_true,
        scenario_phys=scenario,
        scenario_ctx=ctx_s,
        horizon=horizon,
        device=device,
        z_scale=z_scale,
        zero_context=zero_ctx,
        noise=None,
    )
    s_dev = scenario_to_device(scenario, device)
    loss, _ = compute_loss(x_seq, s_dev, loss_cfg)
    d_edge = obstacle_edge_distances(x_seq, s_dev)
    collided = d_edge.min(dim=2).values.min(dim=1).values < 0.0
    terminal = torch.norm(x_seq[:, -1, :2] - s_dev.goal, dim=-1)
    success = (~collided) & (terminal < float(success_tol))

    passed_center = through_center_choice(x_seq, center_x=float(center_x), band=float(center_pass_band))
    # Desired behavior by mode:
    # late (0) -> pass center, early (1) -> avoid center.
    desired_center = (s_dev.mode == 0)
    behavior_acc = float((passed_center == desired_center).float().mean().item())

    mode_late = s_dev.mode == 0
    mode_early = s_dev.mode == 1
    late_success = float(success[mode_late].float().mean().item()) if bool(mode_late.any().item()) else float("nan")
    early_success = float(success[mode_early].float().mean().item()) if bool(mode_early.any().item()) else float("nan")

    metrics = {
        "context_mode": context_mode,
        "loss": float(loss.item()),
        "collision_rate": float(collided.float().mean().item()),
        "success_rate": float(success.float().mean().item()),
        "terminal_dist": float(terminal.mean().item()),
        "behavior_acc": behavior_acc,
        "center_pass_rate": float(passed_center.float().mean().item()),
        "late_success_rate": late_success,
        "early_success_rate": early_success,
        "w_last_mag": float(torch.norm(w_seq[:, -1, :], dim=-1).mean().item()),
    }
    return metrics, x_seq, u_seq


def build_controller(device: torch.device, args: argparse.Namespace) -> PBController:
    nx = 4
    nu = 2
    z_dim = infer_context_dim()
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

    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=1.0, pre_kd=1.5)
    mp = MpDeepSSM(
        mp_in_dim,
        feat_dim,
        mode="loop",
        param=args.ssm_param,
        n_layers=4,
        d_model=16,
        d_state=32,
        ff="GLU",
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
    return build_factorized_controller(
        nominal_plant=plant_nom,
        mp=mp,
        mb=mb,
        u_dim=nu,
        detach_state=False,
        u_nominal=None,
        mp_context_lifter=mp_context_lifter,
    ).to(device)


def plot_loss_curves(train_hist: list[dict], eval_hist: list[dict], run_dir: str, show_plots: bool) -> None:
    if len(train_hist) == 0:
        return
    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([d["epoch"] for d in train_hist], [d["loss"] for d in train_hist], label="train")
    if len(eval_hist) > 0:
        ax.plot([d["epoch"] for d in eval_hist], [d["loss"] for d in eval_hist], label="val (true)", linewidth=2.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Closing-gap experiment loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"))
    if not show_plots:
        plt.close(fig)


def plot_trajectories_by_mode(
    x_seq: torch.Tensor,
    s: ClosingGapScenario,
    run_dir: str,
    show_plots: bool,
    max_each: int,
) -> None:
    plt = get_plt(show_plots)
    idx_late = torch.nonzero(s.mode == 0, as_tuple=False).squeeze(-1)[:max_each]
    idx_early = torch.nonzero(s.mode == 1, as_tuple=False).squeeze(-1)[:max_each]
    ncols = max(int(idx_late.numel()), int(idx_early.numel()), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(4.8 * ncols, 8.0), squeeze=False)
    rows = [idx_late, idx_early]
    row_names = ["late-closing", "early-closing"]

    snap_ids = [0, x_seq.shape[1] // 2, x_seq.shape[1] - 1]
    for r in range(2):
        idxs = rows[r]
        for c in range(ncols):
            ax = axes[r, c]
            if c >= int(idxs.numel()):
                ax.axis("off")
                continue
            i = int(idxs[c].item())
            tr = x_seq[i, :, :2].detach().cpu().numpy()
            ax.plot(tr[:, 0], tr[:, 1], color="C0", linewidth=2.0)
            ax.scatter([s.start[i, 0].item()], [s.start[i, 1].item()], color="green", s=25)
            ax.scatter([0.0], [0.0], color="red", marker="*", s=85)

            centers = s.centers[i].detach().cpu().numpy()
            for q, t in enumerate(snap_ids):
                rt = s.radii_seq[i, t].detach().cpu().numpy()
                alpha = 0.12 + 0.18 * q
                for k in range(2):
                    circ = plt.Circle((centers[k, 0], centers[k, 1]), rt[k], color=("C1" if k == 0 else "C2"), alpha=alpha)
                    ax.add_patch(circ)
            if c == 0:
                ax.set_ylabel(row_names[r])
            ax.set_aspect("equal", "box")
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-1.2, 1.2)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Trajectories with obstacle snapshots")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "trajectories_by_mode.png"), bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


def plot_radii_and_control(u_seq: torch.Tensor, s: ClosingGapScenario, run_dir: str, show_plots: bool) -> None:
    plt = get_plt(show_plots)
    t = np.arange(s.radii_seq.shape[1])
    r_mean_late = s.radii_seq[s.mode == 0].mean(dim=0).detach().cpu().numpy() if bool((s.mode == 0).any().item()) else None
    r_mean_early = s.radii_seq[s.mode == 1].mean(dim=0).detach().cpu().numpy() if bool((s.mode == 1).any().item()) else None
    u_mag = torch.norm(u_seq, dim=-1)
    u_mean = u_mag.mean(dim=0).detach().cpu().numpy()
    u_q10 = torch.quantile(u_mag, q=0.10, dim=0).detach().cpu().numpy()
    u_q90 = torch.quantile(u_mag, q=0.90, dim=0).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    if r_mean_late is not None:
        axes[0].plot(t, r_mean_late[:, 0], color="C1", label="top radius (late)")
        axes[0].plot(t, r_mean_late[:, 1], color="C2", label="bottom radius (late)")
    if r_mean_early is not None:
        axes[0].plot(t, r_mean_early[:, 0], "--", color="C1", label="top radius (early)")
        axes[0].plot(t, r_mean_early[:, 1], "--", color="C2", label="bottom radius (early)")
    axes[0].set_ylabel("radius")
    axes[0].set_title("Time-varying radii by mode")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, u_mean, color="C0", linewidth=2.0, label="mean ||u_t||")
    axes[1].fill_between(t, u_q10, u_q90, color="C0", alpha=0.2, label="10-90 percentile")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("||u_t||")
    axes[1].set_title("Control magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "radii_and_control.png"))
    if not show_plots:
        plt.close(fig)


def plot_ablation_bars(m_true: dict, m_shuf: dict, m_zero: dict, run_dir: str, show_plots: bool) -> None:
    plt = get_plt(show_plots)
    labels = ["true", "shuffled", "zero"]
    success = [float(m_true["success_rate"]), float(m_shuf["success_rate"]), float(m_zero["success_rate"])]
    crash = [float(m_true["collision_rate"]), float(m_shuf["collision_rate"]), float(m_zero["collision_rate"])]
    beh = [float(m_true["behavior_acc"]), float(m_shuf["behavior_acc"]), float(m_zero["behavior_acc"])]

    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(x - w, success, width=w, color="C2", label="success")
    ax.bar(x, crash, width=w, color="C3", label="collision")
    ax.bar(x + w, beh, width=w, color="C0", label="behavior acc")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Context ablation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "ablation_context_bars.png"))
    if not show_plots:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show_plots = not args.no_show_plots
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        "pb_core_closing_gap_context",
        f"closing_gap_{run_id}",
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    centers_fixed = fixed_centers(float(args.center_x), float(args.center_y_sep))
    controller = build_controller(device, args)
    plant_true = DoubleIntegratorTrue(dt=0.05, pre_kp=1.0, pre_kd=1.5)
    noise_model = DecayingGaussianNoise(float(args.noise_sigma0), float(args.noise_tau))
    loss_cfg = LossConfig(
        w_term=float(args.w_term),
        w_stage=float(args.w_stage),
        w_bar_soft=float(args.w_bar_soft),
        w_bar_hard=float(args.w_bar_hard),
        bar_margin=float(args.bar_margin),
        bar_beta=float(args.bar_beta),
    )

    # Fixed validation set for consistent comparisons.
    val_scenario = scenario_to_device(
        sample_scenario(
            batch_size=int(args.val_batch),
            horizon=int(args.val_horizon),
            seed=int(args.seed) + 999,
            args=args,
            centers_fixed=centers_fixed,
            paired_context=False,
        ),
        device,
    )

    # Compatibility check at t=0.
    x_probe = make_x0(val_scenario, device)
    z_probe = build_context(x_probe, val_scenario, t=0, horizon=int(args.val_horizon), z_scale=float(args.z_scale))
    ok, msg = validate_component_compatibility(
        controller=controller,
        plant_true=plant_true,
        x0=x_probe,
        z0=z_probe,
        raise_on_error=False,
    )
    if not ok:
        raise RuntimeError(f"Compatibility check failed: {msg}")

    optimizer = optim.Adam(controller.parameters(), lr=float(args.lr))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=float(args.lr_min))

    train_hist: List[dict] = []
    eval_hist: List[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    best_ckpt = os.path.join(run_dir, "best_model.pt")

    print(f"Starting closing-gap experiment on {device}")
    for epoch in range(1, int(args.epochs) + 1):
        controller.train()
        optimizer.zero_grad()

        train_s = scenario_to_device(
            sample_scenario(
                batch_size=int(args.batch),
                horizon=int(args.horizon),
                seed=int(args.seed) + epoch,
                args=args,
                centers_fixed=centers_fixed,
                paired_context=bool(args.paired_train_context),
            ),
            device,
        )
        noise = noise_model.sample(
            bsz=int(args.batch),
            horizon=int(args.horizon),
            nx=4,
            device=device,
            seed=int(args.seed) + 10000 + epoch,
        )
        x_seq, u_seq, _ = rollout_scenario(
            controller=controller,
            plant_true=plant_true,
            scenario_phys=train_s,
            scenario_ctx=train_s,
            horizon=int(args.horizon),
            device=device,
            z_scale=float(args.z_scale),
            zero_context=False,
            noise=noise,
        )
        loss, _ = compute_loss(x_seq, train_s, loss_cfg)
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        scheduler.step()

        train_hist.append({"epoch": epoch, "loss": float(loss.item())})

        if epoch % int(args.eval_every) == 0:
            controller.eval()
            m_true, _, _ = evaluate(
                controller=controller,
                plant_true=plant_true,
                scenario=val_scenario,
                horizon=int(args.val_horizon),
                device=device,
                loss_cfg=loss_cfg,
                success_tol=float(args.success_tol),
                z_scale=float(args.z_scale),
                center_x=float(args.center_x),
                center_pass_band=float(args.center_pass_band),
                context_mode="true",
            )
            m_true["epoch"] = epoch
            eval_hist.append(m_true)

            if float(m_true["loss"]) < best_loss:
                best_loss = float(m_true["loss"])
                best_epoch = int(epoch)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": controller.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "args": vars(args),
                    },
                    best_ckpt,
                )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | Train {float(loss.item()):.2f} | "
                f"Val {float(m_true['loss']):.2f} | Crash {100.0 * float(m_true['collision_rate']):.1f}% | "
                f"Success {100.0 * float(m_true['success_rate']):.1f}% | "
                f"BehAcc {100.0 * float(m_true['behavior_acc']):.1f}% | "
                f"LR {scheduler.get_last_lr()[0]:.1e}"
            )

    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        controller.load_state_dict(ckpt["model_state_dict"])

    # Final evaluations.
    m_true, x_val, u_val = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=int(args.val_horizon),
        device=device,
        loss_cfg=loss_cfg,
        success_tol=float(args.success_tol),
        z_scale=float(args.z_scale),
        center_x=float(args.center_x),
        center_pass_band=float(args.center_pass_band),
        context_mode="true",
    )
    m_shuf, _, _ = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=int(args.val_horizon),
        device=device,
        loss_cfg=loss_cfg,
        success_tol=float(args.success_tol),
        z_scale=float(args.z_scale),
        center_x=float(args.center_x),
        center_pass_band=float(args.center_pass_band),
        context_mode="shuffled",
    )
    m_zero, _, _ = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=int(args.val_horizon),
        device=device,
        loss_cfg=loss_cfg,
        success_tol=float(args.success_tol),
        z_scale=float(args.z_scale),
        center_x=float(args.center_x),
        center_pass_band=float(args.center_pass_band),
        context_mode="zero",
    )

    with open(os.path.join(run_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_hist, f, indent=2)
    with open(os.path.join(run_dir, "eval_history.json"), "w", encoding="utf-8") as f:
        json.dump(eval_hist, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": int(best_epoch),
                "best_loss": float(best_loss),
                "final_true_context": m_true,
                "final_shuffled_context": m_shuf,
                "final_zero_context": m_zero,
            },
            f,
            indent=2,
        )
    torch.save(controller.state_dict(), os.path.join(run_dir, "pb_model_final.pt"))

    plot_loss_curves(train_hist, eval_hist, run_dir, show_plots)
    plot_trajectories_by_mode(
        x_seq=x_val,
        s=val_scenario,
        run_dir=run_dir,
        show_plots=show_plots,
        max_each=int(args.traj_plot_each),
    )
    plot_radii_and_control(u_val, val_scenario, run_dir, show_plots)
    plot_ablation_bars(m_true, m_shuf, m_zero, run_dir, show_plots)

    print(f"Done. Artifacts saved to {run_dir}")
    if show_plots:
        plt = get_plt(True)
        plt.show()


if __name__ == "__main__":
    main()

