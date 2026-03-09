"""
PB experiment: two obstacles with time-varying radii.

Design goal:
  Keep the setup simple, but make context genuinely useful:
    - obstacle radii change over time,
    - controller receives context z_t with current obstacle geometry,
    - M_p is an SSM (MpDeepSSM),
    - training loss = distance-to-goal + obstacle barriers.

This script is intentionally explicit and commented.
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
    parser = argparse.ArgumentParser("Time-varying obstacle PB experiment")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch", type=int, default=768)
    parser.add_argument("--val_batch", type=int, default=768)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--val_horizon", type=int, default=140)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--no_show_plots", action="store_true")

    # Start distribution.
    parser.add_argument("--start_x_min", type=float, default=1.5)
    parser.add_argument("--start_x_max", type=float, default=2.3)
    parser.add_argument("--start_y_min", type=float, default=-0.35)
    parser.add_argument("--start_y_max", type=float, default=0.35)
    parser.add_argument("--start_clearance", type=float, default=0.10)

    # Two fixed obstacle centers.
    parser.add_argument("--center_x", type=float, default=0.95)
    parser.add_argument("--center_y_sep", type=float, default=0.55)

    # Radius range and time variation parameters.
    parser.add_argument("--r_min", type=float, default=0.18)
    parser.add_argument("--r_max", type=float, default=0.42)
    parser.add_argument("--amp_min", type=float, default=0.05)
    parser.add_argument("--amp_max", type=float, default=0.13)
    parser.add_argument("--cycles_min", type=float, default=0.6)
    parser.add_argument("--cycles_max", type=float, default=1.4)
    parser.add_argument("--phase_offset", type=float, default=0.35)  # random offset around anti-phase

    # Context scaling.
    parser.add_argument("--z_scale", type=float, default=8.0)

    # Loss: distance + barriers only.
    parser.add_argument("--w_term", type=float, default=34.0)
    parser.add_argument("--w_stage", type=float, default=7.0)
    parser.add_argument("--w_bar_soft", type=float, default=55.0)
    parser.add_argument("--w_bar_hard", type=float, default=110.0)
    parser.add_argument("--bar_margin", type=float, default=0.12)
    parser.add_argument("--bar_beta", type=float, default=14.0)
    parser.add_argument("--success_tol", type=float, default=0.18)

    # Optional process noise on true rollout (set sigma0=0 to disable).
    parser.add_argument("--noise_sigma0", type=float, default=0.004)
    parser.add_argument("--noise_tau", type=float, default=24.0)

    # Controller architecture.
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--mb_hidden", type=int, default=64)
    parser.add_argument("--mb_layers", type=int, default=4)
    parser.add_argument("--z_residual_gain", type=float, default=8.0)
    parser.add_argument("--mb_bound", type=float, default=10.0)
    parser.add_argument("--ssm_param", type=str, default="lru", choices=["lru", "tv"])

    # Optional augmentation of M_p input: w -> [w, z_lift].
    parser.add_argument("--mp_context_lift", dest="mp_context_lift", action="store_true")
    parser.add_argument("--no_mp_context_lift", dest="mp_context_lift", action="store_false")
    parser.set_defaults(mp_context_lift=True)
    parser.add_argument("--mp_context_lift_type", type=str, default="linear", choices=["identity", "linear", "mlp"])
    parser.add_argument("--mp_context_lift_dim", type=int, default=8)
    parser.add_argument("--mp_context_hidden_dim", type=int, default=32)
    parser.add_argument("--mp_context_decay_law", type=str, default="poly", choices=["exp", "poly", "finite"])
    parser.add_argument("--mp_context_decay_rate", type=float, default=0.08)
    parser.add_argument("--mp_context_decay_power", type=float, default=.7)
    parser.add_argument("--mp_context_decay_horizon", type=int, default=100)
    parser.add_argument("--mp_context_lp_p", type=float, default=2.0)
    parser.add_argument("--mp_context_scale", type=float, default=1.0)

    # Plot controls.
    parser.add_argument("--traj_plot_each", type=int, default=4)
    return parser.parse_args()


@dataclass
class TVScenario:
    """Batch scenario for time-varying two-obstacle task."""

    start: torch.Tensor  # (B,2)
    goal: torch.Tensor  # (B,2)
    centers: torch.Tensor  # (B,2,2)
    radii_seq: torch.Tensor  # (B,T,2)


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
    """Two obstacle centers: top and bottom."""
    return torch.tensor(
        [[center_x, +y_sep], [center_x, -y_sep]],
        dtype=torch.float32,
    )


def _sample_one_radii_seq(
    *,
    horizon: int,
    r_min: float,
    r_max: float,
    amp_min: float,
    amp_max: float,
    cycles_min: float,
    cycles_max: float,
    phase_offset: float,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """
    Build one pair of radius trajectories (top/bottom), mostly anti-phase.

    Anti-phase oscillation creates a simple "which side is more open now?" context signal.
    """
    t = np.arange(horizon, dtype=np.float32)
    # Keep base radius away from bounds so oscillation can move both ways.
    base_lo = r_min + 0.35 * (r_max - r_min)
    base_hi = r_min + 0.65 * (r_max - r_min)
    base_top = rng.uniform(base_lo, base_hi)
    base_bottom = rng.uniform(base_lo, base_hi)
    amp = rng.uniform(amp_min, amp_max)
    cycles = rng.uniform(cycles_min, cycles_max)
    omega = 2.0 * np.pi * cycles / max(horizon - 1, 1)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    # Small random deviation around exact anti-phase.
    delta = rng.uniform(-phase_offset, phase_offset)

    r_top = base_top + amp * np.sin(omega * t + phase)
    r_bottom = base_bottom + amp * np.sin(omega * t + phase + np.pi + delta)
    r_top = np.clip(r_top, r_min, r_max)
    r_bottom = np.clip(r_bottom, r_min, r_max)
    return torch.from_numpy(np.stack([r_top, r_bottom], axis=-1).astype(np.float32))  # (T,2)


def sample_tv_scenario(
    *,
    batch_size: int,
    horizon: int,
    seed: int,
    args: argparse.Namespace,
    centers: torch.Tensor,
) -> TVScenario:
    """Sample a training/evaluation batch."""
    rng = np.random.RandomState(int(seed))
    start = torch.zeros(batch_size, 2, dtype=torch.float32)
    goal = torch.zeros(batch_size, 2, dtype=torch.float32)
    centers_b = centers.view(1, 2, 2).repeat(batch_size, 1, 1).clone()
    radii_seq = torch.zeros(batch_size, horizon, 2, dtype=torch.float32)

    for b in range(batch_size):
        rseq = _sample_one_radii_seq(
            horizon=horizon,
            r_min=float(args.r_min),
            r_max=float(args.r_max),
            amp_min=float(args.amp_min),
            amp_max=float(args.amp_max),
            cycles_min=float(args.cycles_min),
            cycles_max=float(args.cycles_max),
            phase_offset=float(args.phase_offset),
            rng=rng,
        )

        # Rejection sample starts to ensure they are not inside obstacles at t=0.
        ok = False
        for _ in range(200):
            sx = rng.uniform(float(args.start_x_min), float(args.start_x_max))
            sy = rng.uniform(float(args.start_y_min), float(args.start_y_max))
            s = torch.tensor([sx, sy], dtype=torch.float32)
            d0 = torch.norm(s.view(1, 2) - centers, dim=-1) - rseq[0]
            if bool((d0 > float(args.start_clearance)).all().item()):
                start[b] = s
                ok = True
                break
        if not ok:
            raise RuntimeError("Failed to sample a feasible start. Try relaxing start_clearance.")

        radii_seq[b] = rseq

    return TVScenario(start=start, goal=goal, centers=centers_b, radii_seq=radii_seq)


def scenario_to_device(s: TVScenario, device: torch.device) -> TVScenario:
    return TVScenario(
        start=s.start.to(device),
        goal=s.goal.to(device),
        centers=s.centers.to(device),
        radii_seq=s.radii_seq.to(device),
    )


def shuffled_context_scenario(s: TVScenario) -> TVScenario:
    """Shuffle radii trajectories across the batch (context ablation)."""
    bsz = s.start.shape[0]
    perm = torch.randperm(bsz, device=s.start.device)
    return TVScenario(
        start=s.start,
        goal=s.goal,
        centers=s.centers,
        radii_seq=s.radii_seq[perm],
    )


def infer_context_dim(k_obstacles: int = 2) -> int:
    """
    Context features used below:
      goal_dir(2) + rel_to_obs(2K) + dist_to_edge(K) + radii_t(K) + dr_t(K)
    => 2 + 5K
    """
    return 2 + 5 * int(k_obstacles)


def build_tv_context(
    x_t: torch.Tensor,
    scenario_ctx: TVScenario,
    *,
    t: int,
    z_scale: float,
) -> torch.Tensor:
    """
    Build z_t from current state and time-varying obstacle geometry.

    Shapes:
      x_t: (B,1,4)
      z_t: (B,1,Nz)
    """
    x_t = as_bt(x_t)
    pos = x_t[..., :2]  # (B,1,2)
    goal = scenario_ctx.goal.to(x_t.device).unsqueeze(1)  # (B,1,2)
    centers = scenario_ctx.centers.to(x_t.device)  # (B,2,2)
    radii_t = scenario_ctx.radii_seq.to(x_t.device)[:, t : t + 1, :]  # (B,1,2)
    if t == 0:
        dr_t = torch.zeros_like(radii_t)
    else:
        dr_t = scenario_ctx.radii_seq.to(x_t.device)[:, t : t + 1, :] - scenario_ctx.radii_seq.to(x_t.device)[:, t - 1 : t, :]

    rel = centers.unsqueeze(1) - pos.unsqueeze(2)  # (B,1,2,2)
    dist_edge = torch.norm(rel, dim=-1) - radii_t  # (B,1,2)
    goal_dir = goal - pos  # (B,1,2)

    z_t = torch.cat(
        [
            goal_dir,
            rel.reshape(rel.shape[0], rel.shape[1], -1),
            dist_edge,
            radii_t,
            dr_t,
        ],
        dim=-1,
    )
    return float(z_scale) * z_t


def make_x0(s: TVScenario, device: torch.device) -> torch.Tensor:
    vel0 = torch.zeros(s.start.shape[0], 2, device=device)
    return torch.cat([s.start.to(device), vel0], dim=-1).unsqueeze(1)


def obstacle_edge_distances_tv(x_seq: torch.Tensor, s: TVScenario) -> torch.Tensor:
    """
    Return edge distances to each obstacle at each time.
    Shape: (B,T,2)
    """
    x_seq = as_bt(x_seq)
    pos = x_seq[..., :2]  # (B,T,2)
    centers = s.centers.to(x_seq.device).unsqueeze(1)  # (B,1,2,2)
    radii = s.radii_seq.to(x_seq.device)  # (B,T,2)
    dist = torch.norm(pos.unsqueeze(2) - centers, dim=-1)
    return dist - radii


def compute_loss_per_sample(x_seq: torch.Tensor, s: TVScenario, cfg: LossConfig) -> torch.Tensor:
    goal = s.goal.to(x_seq.device).unsqueeze(1)
    dist_goal = torch.norm(x_seq[..., :2] - goal, dim=-1)  # (B,T)
    term = dist_goal[:, -1] * cfg.w_term
    stage = dist_goal.mean(dim=1) * cfg.w_stage

    d_edge = obstacle_edge_distances_tv(x_seq, s)
    if cfg.bar_beta <= 0:
        raise ValueError(f"bar_beta must be > 0, got {cfg.bar_beta}")
    soft = (torch.nn.functional.softplus(cfg.bar_beta * (cfg.bar_margin - d_edge)) / cfg.bar_beta).sum(dim=-1)
    hard = (torch.relu(-d_edge) ** 2).sum(dim=-1)
    bar_soft = soft.mean(dim=1) * cfg.w_bar_soft
    bar_hard = hard.mean(dim=1) * cfg.w_bar_hard
    return term + stage + bar_soft + bar_hard


def compute_loss(x_seq: torch.Tensor, s: TVScenario, cfg: LossConfig) -> tuple[torch.Tensor, dict]:
    per = compute_loss_per_sample(x_seq, s, cfg)
    loss = per.mean()
    return loss, {"loss_total": float(loss.item())}


def rollout_tv(
    *,
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: TVScenario,
    horizon: int,
    device: torch.device,
    z_scale: float,
    context_scenario: TVScenario | None = None,
    zero_context: bool = False,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rollout helper for this task.

    - `scenario` defines physical rollout (start, true obstacle radii for loss/metrics).
    - `context_scenario` can differ for ablations (e.g., shuffled context).
    """
    s_phys = scenario_to_device(scenario, device)
    s_ctx = scenario_to_device(context_scenario if context_scenario is not None else scenario, device)
    x0 = make_x0(s_phys, device)

    if noise is not None:
        if noise.shape[0] != x0.shape[0] or noise.shape[1] != horizon or noise.shape[2] != x0.shape[2]:
            raise ValueError(
                "noise must have shape (B,horizon,Nx). "
                f"Got {tuple(noise.shape)}, expected ({x0.shape[0]},{horizon},{x0.shape[2]})"
            )

    z_dim = infer_context_dim(2)

    def _context_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
        if zero_context:
            return torch.zeros(x_t.shape[0], 1, z_dim, device=x_t.device, dtype=x_t.dtype)
        return build_tv_context(x_t, s_ctx, t=t, z_scale=z_scale)

    out = rollout_pb(
        controller=controller,
        plant_true=plant_true,
        x0=x0,
        horizon=horizon,
        context_fn=_context_fn,
        w0=x0,
        process_noise_seq=noise,
    )
    return out.x_seq, out.u_seq, out.w_seq


@torch.no_grad()
def evaluate(
    *,
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: TVScenario,
    horizon: int,
    device: torch.device,
    loss_cfg: LossConfig,
    success_tol: float,
    z_scale: float,
    context_mode: str = "true",
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    if context_mode not in {"true", "shuffled", "zero"}:
        raise ValueError(f"Unknown context_mode {context_mode!r}")

    if context_mode == "true":
        ctx_scenario = scenario
        zero_ctx = False
    elif context_mode == "shuffled":
        ctx_scenario = shuffled_context_scenario(scenario_to_device(scenario, device))
        zero_ctx = False
    else:
        ctx_scenario = scenario
        zero_ctx = True

    x_seq, u_seq, w_seq = rollout_tv(
        controller=controller,
        plant_true=plant_true,
        scenario=scenario,
        context_scenario=ctx_scenario,
        horizon=horizon,
        device=device,
        z_scale=z_scale,
        zero_context=zero_ctx,
        noise=None,
    )
    loss, parts = compute_loss(x_seq, scenario_to_device(scenario, device), loss_cfg)

    d_edge = obstacle_edge_distances_tv(x_seq, scenario_to_device(scenario, device))
    collided = d_edge.min(dim=2).values.min(dim=1).values < 0.0
    terminal_dist = torch.norm(x_seq[:, -1, :2], dim=-1)
    success = (~collided) & (terminal_dist < float(success_tol))
    parts.update(
        {
            "context_mode": context_mode,
            "loss": float(loss.item()),
            "collision_rate": float(collided.float().mean().item()),
            "success_rate": float(success.float().mean().item()),
            "terminal_dist": float(terminal_dist.mean().item()),
            "w_last_mag": float(torch.norm(w_seq[:, -1, :], dim=-1).mean().item()),
        }
    )
    return parts, x_seq, u_seq


def build_controller(device: torch.device, args: argparse.Namespace) -> PBController:
    nx = 4
    nu = 2
    z_dim = infer_context_dim(2)
    feat_dim = int(args.feat_dim)

    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=1.0, pre_kd=1.5)
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
        u_nominal=None,
        detach_state=False,
        mp_context_lifter=mp_context_lifter,
    ).to(device)


def plot_loss_curves(train_hist: list[dict], eval_hist: list[dict], run_dir: str, show_plots: bool):
    if len(train_hist) == 0:
        return
    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([d["epoch"] for d in train_hist], [d["loss"] for d in train_hist], label="train")
    if len(eval_hist) > 0:
        ax.plot([d["epoch"] for d in eval_hist], [d["loss"] for d in eval_hist], label="val (true ctx)", linewidth=2.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("TV-obstacle training/validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"))
    if not show_plots:
        plt.close(fig)


def plot_trajectories_with_snapshots(
    *,
    x_seq: torch.Tensor,
    scenario: TVScenario,
    run_dir: str,
    show_plots: bool,
    max_each: int = 4,
):
    """
    Plot trajectories plus obstacle snapshots at multiple times.
    """
    plt = get_plt(show_plots)
    n = min(int(max_each), int(x_seq.shape[0]))
    if n <= 0:
        return
    snap_ids = [0, max(0, x_seq.shape[1] // 3), max(0, 2 * x_seq.shape[1] // 3), x_seq.shape[1] - 1]

    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.8), squeeze=False)
    x_cpu = x_seq.detach().cpu()
    s = scenario
    for j in range(n):
        ax = axes[0, j]
        traj = x_cpu[j, :, :2].numpy()
        ax.plot(traj[:, 0], traj[:, 1], color="C0", linewidth=2.0, label="traj")
        ax.scatter([s.start[j, 0].item()], [s.start[j, 1].item()], color="green", s=30, label="start")
        ax.scatter([0.0], [0.0], color="red", marker="*", s=90, label="goal")

        centers = s.centers[j].detach().cpu().numpy()  # (2,2)
        for q, t in enumerate(snap_ids):
            rt = s.radii_seq[j, t].detach().cpu().numpy()  # (2,)
            alpha = 0.12 + 0.16 * q
            for k in range(2):
                circ = plt.Circle((centers[k, 0], centers[k, 1]), rt[k], color=("C1" if k == 0 else "C2"), alpha=alpha)
                ax.add_patch(circ)
        ax.set_title(f"sample {j}")
        ax.set_aspect("equal", "box")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1.4, 1.4)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Trajectories with obstacle-radius snapshots")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "trajectories_tv_obstacles.png"), bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


def plot_radii_and_control(
    *,
    u_seq: torch.Tensor,
    scenario: TVScenario,
    run_dir: str,
    show_plots: bool,
):
    plt = get_plt(show_plots)
    u_mag = torch.norm(u_seq, dim=-1)  # (B,T)
    u_mean = u_mag.mean(dim=0).detach().cpu().numpy()
    u_q10 = torch.quantile(u_mag, q=0.10, dim=0).detach().cpu().numpy()
    u_q90 = torch.quantile(u_mag, q=0.90, dim=0).detach().cpu().numpy()

    r_mean = scenario.radii_seq.mean(dim=0).detach().cpu().numpy()  # (T,2)
    t = np.arange(r_mean.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    axes[0].plot(t, r_mean[:, 0], color="C1", label="top radius")
    axes[0].plot(t, r_mean[:, 1], color="C2", label="bottom radius")
    axes[0].set_ylabel("radius")
    axes[0].set_title("Mean time-varying obstacle radii")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, u_mean, color="C0", linewidth=2.0, label="mean ||u_t||")
    axes[1].fill_between(t, u_q10, u_q90, color="C0", alpha=0.2, label="10-90 percentile")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("||u_t||")
    axes[1].set_title("Control magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "radii_and_control_over_time.png"))
    if not show_plots:
        plt.close(fig)


def plot_ablation_bars(metrics_true: dict, metrics_shuf: dict, metrics_zero: dict, run_dir: str, show_plots: bool):
    plt = get_plt(show_plots)
    labels = ["true", "shuffled", "zero"]
    success = [float(metrics_true["success_rate"]), float(metrics_shuf["success_rate"]), float(metrics_zero["success_rate"])]
    collision = [float(metrics_true["collision_rate"]), float(metrics_shuf["collision_rate"]), float(metrics_zero["collision_rate"])]

    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x - 0.5 * w, success, width=w, color="C2", label="success")
    ax.bar(x + 0.5 * w, collision, width=w, color="C3", label="collision")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Context ablation (time-varying radii)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
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
        "pb_core_timevarying_obstacles",
        f"tv_obs_{run_id}",
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    centers = fixed_centers(float(args.center_x), float(args.center_y_sep))
    controller = build_controller(device, args)
    # Keep true == nominal in this minimal demo.
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

    val_scenario = scenario_to_device(
        sample_tv_scenario(
            batch_size=int(args.val_batch),
            horizon=int(args.val_horizon),
            seed=int(args.seed) + 999,
            args=args,
            centers=centers,
        ),
        device,
    )

    # Compatibility check with t=0 context.
    x_probe = make_x0(val_scenario, device)
    z_probe = build_tv_context(x_probe, val_scenario, t=0, z_scale=float(args.z_scale))
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
    best_ckpt = os.path.join(run_dir, "best_model.pt")
    best_loss = float("inf")
    best_epoch = 0

    print(f"Starting TV-obstacle experiment on {device}")
    for epoch in range(1, int(args.epochs) + 1):
        controller.train()
        optimizer.zero_grad()

        train_scenario = scenario_to_device(
            sample_tv_scenario(
                batch_size=int(args.batch),
                horizon=int(args.horizon),
                seed=int(args.seed) + epoch,
                args=args,
                centers=centers,
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
        x_seq, u_seq, w_seq = rollout_tv(
            controller=controller,
            plant_true=plant_true,
            scenario=train_scenario,
            horizon=int(args.horizon),
            device=device,
            z_scale=float(args.z_scale),
            noise=noise,
        )
        loss, _ = compute_loss(x_seq, train_scenario, loss_cfg)
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
                context_mode="true",
            )
            m_true["epoch"] = epoch
            eval_hist.append(m_true)

            if float(m_true["loss"]) < best_loss:
                best_loss = float(m_true["loss"])
                best_epoch = epoch
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
                f"Success {100.0 * float(m_true['success_rate']):.1f}% | LR {scheduler.get_last_lr()[0]:.1e}"
            )

    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        controller.load_state_dict(ckpt["model_state_dict"])

    # Final evaluation + ablations.
    metrics_true, x_val, u_val = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=int(args.val_horizon),
        device=device,
        loss_cfg=loss_cfg,
        success_tol=float(args.success_tol),
        z_scale=float(args.z_scale),
        context_mode="true",
    )
    metrics_shuf, _, _ = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=int(args.val_horizon),
        device=device,
        loss_cfg=loss_cfg,
        success_tol=float(args.success_tol),
        z_scale=float(args.z_scale),
        context_mode="shuffled",
    )
    metrics_zero, _, _ = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=int(args.val_horizon),
        device=device,
        loss_cfg=loss_cfg,
        success_tol=float(args.success_tol),
        z_scale=float(args.z_scale),
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
                "final_true_context": metrics_true,
                "final_shuffled_context": metrics_shuf,
                "final_zero_context": metrics_zero,
            },
            f,
            indent=2,
        )
    torch.save(controller.state_dict(), os.path.join(run_dir, "pb_model_final.pt"))

    # Plots.
    plot_loss_curves(train_hist, eval_hist, run_dir, show_plots)
    plot_trajectories_with_snapshots(
        x_seq=x_val,
        scenario=val_scenario,
        run_dir=run_dir,
        show_plots=show_plots,
        max_each=int(args.traj_plot_each),
    )
    plot_radii_and_control(
        u_seq=u_val,
        scenario=val_scenario,
        run_dir=run_dir,
        show_plots=show_plots,
    )
    plot_ablation_bars(metrics_true, metrics_shuf, metrics_zero, run_dir, show_plots)

    print(f"Done. Artifacts saved to {run_dir}")
    if show_plots:
        plt = get_plt(True)
        plt.show()


if __name__ == "__main__":
    main()

