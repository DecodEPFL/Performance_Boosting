"""
Minimal experiment: robot navigates to origin with 3 circular obstacles.
Default rollout is perfect-model with PB convention w0=x0; optional decaying process noise can be enabled.

Outputs plots and metrics under runs/nav_experiment/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import torch

from pb_controller import PBController, FactorizedOperator
from bounded_mlp_operator import BoundedMLPOperator
from nav_env import (
    sample_scenarios,
    build_context,
    collision_penalty,
    collision_violation_penalty,
    direct_path_intersection_mask,
    ensure_challenging_starts,
    min_dist_to_edge,
    obstacle_edge_distances,
)
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from rollout_bptt import rollout_bptt
from ssm_operators import MpDeepSSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PB navigation experiment (perfect model)")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--test_batch", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--best_ckpt_metric", type=str, default="collision_then_loss", choices=["loss", "collision_then_loss"])
    parser.add_argument("--best_ckpt_collision_tol", type=float, default=1e-4)

    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--pre_kp", type=float, default=1.0)
    parser.add_argument("--pre_kd", type=float, default=1.5)
    parser.add_argument("--r_min", type=float, default=0.2)
    parser.add_argument("--r_max", type=float, default=.8)
    parser.add_argument("--run_id", type=str, default="")

    parser.add_argument(
        "--u_max",
        type=float,
        default=None,
        help="Optional external actuator clamp. If omitted, no post-controller clamping is used.",
    )

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mb_bound_mode", type=str, default="softsign", choices=["tanh", "softsign", "clamp"])
    parser.add_argument("--mb_bound_value", type=float, default=78.0)
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--mb_no_z_residual", action="store_true")
    parser.add_argument("--mb_z_residual_gain", type=float, default=10.0)
    parser.add_argument("--mp_param", type=str, default="lru")
    parser.add_argument("--mp_layers", type=int, default=4)
    parser.add_argument("--mp_mode", type=str, default="loop", choices=["loop", "scan"])
    parser.add_argument("--mp_d_model", type=int, default=16)
    parser.add_argument("--mp_d_state", type=int, default=32)
    parser.add_argument("--mp_ff", type=str, default="GLU")
    parser.add_argument("--mp_dropout", type=float, default=0.0)
    parser.add_argument("--mp_rho", type=float, default=0.999)
    parser.add_argument("--mp_rmin", type=float, default=0.98)
    parser.add_argument("--mp_rmax", type=float, default=0.999)
    parser.add_argument("--mp_gamma", type=float, default=10.0)
    parser.add_argument("--mp_detach_state", action="store_true")

    parser.add_argument("--loss_mode", type=str, default="shaped", choices=["natural", "shaped", "direct"])
    parser.add_argument("--collision_weight", type=float, default=24.0)
    parser.add_argument("--control_weight", type=float, default=0.02)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--terminal_weight", type=float, default=35.0)
    parser.add_argument("--stage_weight", type=float, default=16.0)
    parser.add_argument("--path_weight", type=float, default=8.0)
    parser.add_argument("--alpha_barrier", type=float, default=2.0)
    parser.add_argument("--alpha_corridor", type=float, default=0)
    parser.add_argument("--d_safe", type=float, default=0.2)
    parser.add_argument("--alpha_q_scaling", type=float, default=0.0)
    parser.add_argument("--robot_radius", type=float, default=0.02)
    parser.add_argument("--barrier_eps", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.06)
    parser.add_argument("--collision_beta", type=float, default=20.0)
    parser.add_argument("--collision_violation_weight", type=float, default=900.0)
    parser.add_argument("--progress_weight", type=float, default=10.0)
    parser.add_argument("--clear_align_weight", type=float, default=4.0)
    parser.add_argument("--clear_path_margin", type=float, default=0.01)
    parser.add_argument("--safety_clearance", type=float, default=0.08)
    parser.add_argument("--safety_weight", type=float, default=220.0)
    parser.add_argument("--context_sensitivity_target_gain", type=float, default=0.8)
    parser.add_argument("--context_sensitivity_band", type=float, default=0.7)
    parser.add_argument("--context_sensitivity_temp", type=float, default=0.2)
    parser.add_argument("--context_sensitivity_scale_small", type=float, default=0.75)
    parser.add_argument("--context_sensitivity_scale_large", type=float, default=1.5)
    parser.add_argument("--challenge_train_frac", type=float, default=0.75)
    parser.add_argument("--challenge_test_frac", type=float, default=0.55)
    parser.add_argument("--challenge_margin", type=float, default=0.05)
    parser.add_argument("--challenge_radius_quantile", type=float, default=0.9)
    parser.add_argument("--challenge_start_clearance", type=float, default=0.20)
    parser.add_argument("--challenge_cone_fraction", type=float, default=0.75)
    parser.add_argument("--challenge_cone_spread", type=float, default=1.2)
    parser.add_argument("--challenge_radial_extension", type=float, default=1.8)
    parser.add_argument("--challenge_diverse_angles", action="store_true")
    parser.add_argument("--challenge_angle_bins", type=int, default=20)
    parser.add_argument("--no_show_plots", action="store_true")
    parser.add_argument("--heatmap_res", type=int, default=300)
    parser.add_argument("--start_box", type=float, default=2)
    parser.add_argument("--radius_cmp_num_starts", type=int, default=4)
    parser.add_argument("--radius_cmp_anchor_x", type=float, default=2.0)
    parser.add_argument("--radius_cmp_anchor_y", type=float, default=0.0)

    parser.add_argument("--num_obstacles", type=int, default=1, choices=[1, 3])
    parser.add_argument("--single_obstacle_x", type=float, default=1.0)
    parser.add_argument("--single_obstacle_y", type=float, default=0.0)
    parser.add_argument("--single_obstacle_curriculum", action="store_true")
    parser.add_argument("--curriculum_stage1_frac", type=float, default=0.35)
    parser.add_argument("--curriculum_stage2_frac", type=float, default=0.70)
    parser.add_argument("--curriculum_easy_r_min", type=float, default=0.18)
    parser.add_argument("--curriculum_easy_r_max", type=float, default=0.30)
    parser.add_argument("--curriculum_easy_challenge_frac", type=float, default=0.60)
    parser.add_argument("--curriculum_mid_r_min", type=float, default=0.24)
    parser.add_argument("--curriculum_mid_r_max", type=float, default=0.44)
    parser.add_argument("--curriculum_mid_challenge_frac", type=float, default=0.78)
    parser.add_argument("--curriculum_hard_r_min", type=float, default=0.32)
    parser.add_argument("--curriculum_hard_r_max", type=float, default=0.62)
    parser.add_argument("--curriculum_hard_challenge_frac", type=float, default=0.88)

    parser.add_argument("--use_decaying_noise", action="store_true")
    parser.add_argument("--noise_sigma0", type=float, default=0.03)
    parser.add_argument("--noise_tau", type=float, default=18.0)
    parser.add_argument("--noise_floor", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=12345)
    parser.add_argument("--noise_resample_each_epoch", action="store_true")
    parser.add_argument("--resample_train_each_epoch", action="store_true")

    return parser.parse_args()


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


def make_rollout_inputs(scenario):
    B = scenario.start.shape[0]
    device = scenario.start.device
    x0 = torch.cat([scenario.start.to(device), torch.zeros(B, 2, device=device)], dim=-1).unsqueeze(1)
    return x0


def make_context_fn(scenario):
    def ctx_fn(x, t):
        return build_context(x, scenario)
    return ctx_fn


def make_u_post_fn(args, scenario=None):
    del scenario
    use_clamp = args.u_max is not None and args.u_max > 0
    if not use_clamp:
        return None

    def post_fn(x, u, t=None):
        del x, t
        return torch.clamp(u, -args.u_max, args.u_max)

    return post_fn

def make_decaying_process_noise(
    batch_size: int,
    horizon: int,
    state_dim: int,
    sigma0: float,
    tau: float,
    floor: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sigma0 < 0 or floor < 0:
        raise ValueError(f"noise_sigma0 and noise_floor must be non-negative, got sigma0={sigma0}, floor={floor}")
    tau = max(float(tau), 1e-6)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    eps = torch.randn(batch_size, horizon, state_dim, generator=gen, device=device, dtype=dtype)
    t = torch.arange(horizon, device=device, dtype=dtype).view(1, horizon, 1)
    sigma_t = floor + sigma0 * torch.exp(-t / tau)
    return eps * sigma_t


def maybe_make_process_noise(
    args,
    batch_size: int,
    horizon: int,
    state_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if not args.use_decaying_noise:
        return None
    return make_decaying_process_noise(
        batch_size=batch_size,
        horizon=horizon,
        state_dim=state_dim,
        sigma0=args.noise_sigma0,
        tau=args.noise_tau,
        floor=args.noise_floor,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def loss_fn(x_seq, u_seq, scenario, args):
    per = loss_per_sample(x_seq, u_seq, scenario, args)
    return per.mean()


def direct_path_blocked_mask_sequence(pos: torch.Tensor, scenario, margin: float = 0.0) -> torch.Tensor:
    """Return mask (B,T): True when straight segment pos_t->goal intersects any obstacle."""
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError(f"pos must have shape (B,T,2), got {tuple(pos.shape)}")

    goal = scenario.goal.to(pos.device).unsqueeze(1)         # (B,1,2)
    centers = scenario.centers.to(pos.device).unsqueeze(1)   # (B,1,K,2)
    radii = scenario.radii.to(pos.device).unsqueeze(1)       # (B,1,K)

    seg = goal - pos                                          # (B,T,2)
    seg2 = (seg * seg).sum(dim=-1, keepdim=True).clamp_min(1e-9)  # (B,T,1)
    rel = centers - pos.unsqueeze(2)                         # (B,T,K,2)
    proj = (rel * seg.unsqueeze(2)).sum(dim=-1) / seg2       # (B,T,K)
    proj = proj.clamp(0.0, 1.0)
    closest = pos.unsqueeze(2) + proj.unsqueeze(-1) * seg.unsqueeze(2)
    dist = torch.norm(centers - closest, dim=-1)             # (B,T,K)

    blocked = (dist <= (radii + float(margin))).any(dim=-1)
    return blocked


def loss_per_sample(x_seq, u_seq, scenario, args):
    pos = x_seq[:, :, :2]
    goal = scenario.goal.to(pos.device).unsqueeze(1)
    goal_vec = goal - pos
    goal_dist = torch.norm(goal_vec, dim=-1)

    terminal = goal_dist[:, -1]
    stage = goal_dist.mean(dim=1)

    if pos.shape[1] > 1:
        step_len = torch.norm(pos[:, 1:, :] - pos[:, :-1, :], dim=-1)
        path_len_mean = step_len.mean(dim=1)
        path_len_total = step_len.sum(dim=1)
        du_cost = ((u_seq[:, 1:, :] - u_seq[:, :-1, :]) ** 2).mean(dim=(1, 2))
        progress_violation = torch.relu(goal_dist[:, 1:] - goal_dist[:, :-1]).mean(dim=1)
    else:
        path_len_mean = torch.zeros_like(terminal)
        path_len_total = torch.zeros_like(terminal)
        du_cost = torch.zeros_like(terminal)
        progress_violation = torch.zeros_like(terminal)

    straight_len = torch.norm(
        scenario.start.to(pos.device) - scenario.goal.to(pos.device),
        dim=-1,
    )
    path_excess = torch.relu(path_len_total - straight_len)

    u_cost = (u_seq ** 2).mean(dim=(1, 2))
    coll_soft = collision_penalty(x_seq, scenario, margin=args.margin, beta=args.collision_beta).mean(dim=(1, 2))
    coll_violation = collision_violation_penalty(x_seq, scenario).mean(dim=(1, 2))

    if args.loss_mode == "natural":
        state_cost = (
            args.terminal_weight * terminal
            + args.stage_weight * stage
            + args.path_weight * path_len_mean
        )
        return (
            state_cost
            + args.collision_weight * coll_soft
            + args.collision_violation_weight * coll_violation
            + args.control_weight * u_cost
            + args.smooth_weight * du_cost
        )

    if args.loss_mode == "shaped":
        radius_stat = scenario.radii.mean(dim=1)
        q_scaling = torch.exp(-args.alpha_q_scaling * radius_stat)
        state_cost = q_scaling * (
            args.terminal_weight * terminal
            + args.stage_weight * stage
            + args.path_weight * path_len_mean
        )

        dist_edge = obstacle_edge_distances(x_seq, scenario, robot_radius=args.robot_radius)
        barrier = args.alpha_barrier / torch.clamp(dist_edge, min=args.barrier_eps)
        corridor = args.alpha_corridor * (dist_edge - args.d_safe) ** 2
        obstacle_cost = (barrier + corridor).mean(dim=(1, 2))

        return (
            state_cost
            + obstacle_cost
            + args.collision_weight * coll_soft
            + args.collision_violation_weight * coll_violation
            + args.control_weight * u_cost
            + args.smooth_weight * du_cost
        )

    if args.loss_mode == "direct":
        if pos.shape[1] > 1:
            vel = x_seq[:, :, 2:]
            speed = torch.norm(vel, dim=-1)
            goal_dir = goal_vec / goal_dist.unsqueeze(-1).clamp_min(1e-6)
            cos = (vel * goal_dir).sum(dim=-1) / speed.clamp_min(1e-6)
            misalign = (1.0 - cos) * torch.tanh(speed)

            blocked = direct_path_blocked_mask_sequence(pos, scenario, margin=args.clear_path_margin)
            clear = (~blocked).float()
            clear_align_cost = (misalign * clear).sum(dim=1) / (clear.sum(dim=1) + 1e-6)
        else:
            clear_align_cost = torch.zeros_like(terminal)

        dist_edge = obstacle_edge_distances(x_seq, scenario, robot_radius=args.robot_radius)
        safety_violation = torch.relu(args.safety_clearance - dist_edge).mean(dim=(1, 2))

        state_cost = (
            args.terminal_weight * terminal
            + args.stage_weight * stage
            + args.path_weight * path_excess
            + args.progress_weight * progress_violation
            + args.clear_align_weight * clear_align_cost
            + args.safety_weight * safety_violation
        )
        return (
            state_cost
            + args.collision_weight * coll_soft
            + args.collision_violation_weight * coll_violation
            + args.control_weight * u_cost
            + args.smooth_weight * du_cost
        )

    raise ValueError(f"Unknown loss_mode={args.loss_mode!r}")


def make_scaled_radii_scenario(scenario, scale: float, r_min: float = 0.05, r_max: float = 1.5):
    radii_scaled = torch.clamp(scenario.radii * float(scale), min=r_min, max=r_max)
    return type(scenario)(
        start=scenario.start,
        goal=scenario.goal,
        centers=scenario.centers,
        radii=radii_scaled,
    )


def context_sensitivity_counterfactual(controller, w_seq, x_seq, scenario_a, scenario_b, args):
    """
    Counterfactual context sensitivity:
    same disturbance sequence w and same state sequence x, different radii context.
    """
    with torch.no_grad():
        w_in = w_seq.detach()
        x_in = x_seq.detach()
        z_a = build_context(x_in, scenario_a)
        z_b = build_context(x_in, scenario_b)
        delta_r = torch.mean(torch.abs(scenario_a.radii - scenario_b.radii), dim=1)  # (B,)
        # Focus sensitivity on states near (larger) obstacles.
        dist_edge = min_dist_to_edge(x_in, scenario_b)  # (B,T)
        near_weight = torch.sigmoid((args.context_sensitivity_band - dist_edge) / args.context_sensitivity_temp)
        target = args.context_sensitivity_target_gain * delta_r.unsqueeze(1) * (0.2 + 0.8 * near_weight)

    controller.operator.reset()
    u_a = controller.operator(w_in, z_a)
    controller.operator.reset()
    u_b = controller.operator(w_in, z_b)

    u_diff = torch.norm(u_a - u_b, dim=-1)
    shortfall = torch.relu(target - u_diff)
    loss = (shortfall ** 2).mean()
    ratio = float((u_diff.mean() / (target.mean() + 1e-6)).item())
    mean_u_diff = float(u_diff.mean().item())
    return loss, ratio, mean_u_diff


def evaluate(controller, plant_true, scenario, horizon, args, process_noise_seq=None):
    controller.eval()
    with torch.no_grad():
        x0 = make_rollout_inputs(scenario)
        ctx_fn = make_context_fn(scenario)
        u_post = make_u_post_fn(args, scenario)
        x_seq, u_seq, w_seq = rollout_bptt(
            controller=controller,
            plant_true=plant_true,
            x0=x0,
            horizon=horizon,
            context_fn=ctx_fn,
            w0=x0,
            u_post_fn=u_post,
            process_noise_seq=process_noise_seq,
        )
        loss = loss_fn(x_seq, u_seq, scenario, args)
        terminal = torch.norm(x_seq[:, -1, :2], dim=-1)
        min_dist = min_dist_to_edge(x_seq, scenario)
        collision_rate = (min_dist.min(dim=1).values < 0.0).float().mean()
        violation = collision_violation_penalty(x_seq, scenario).mean()
        pos = x_seq[:, :, :2]
        if pos.shape[1] > 1:
            path_len = torch.norm(pos[:, 1:, :] - pos[:, :-1, :], dim=-1).sum(dim=1)
        else:
            path_len = torch.zeros(pos.shape[0], device=pos.device)
        straight_len = torch.norm(
            scenario.start.to(pos.device) - scenario.goal.to(pos.device),
            dim=-1,
        )
        path_ratio = (path_len / (straight_len + 1e-6)).mean()
        if w_seq.shape[1] > 1:
            w_mean_tgt1 = w_seq[:, 1:, :].abs().mean()
        else:
            w_mean_tgt1 = torch.tensor(0.0, device=x_seq.device)

        # Context sensitivity diagnostic on paired scaled-radii scenarios.
        scenario_small = make_scaled_radii_scenario(
            scenario,
            scale=args.context_sensitivity_scale_small,
            r_min=max(0.01, args.r_min * 0.25),
            r_max=max(args.r_max * 1.5, args.r_max + 0.1),
        )
        scenario_large = make_scaled_radii_scenario(
            scenario,
            scale=args.context_sensitivity_scale_large,
            r_min=max(0.01, args.r_min * 0.25),
            r_max=max(args.r_max * 1.5, args.r_max + 0.1),
        )
        _, sens_ratio, sens_u_diff = context_sensitivity_counterfactual(
            controller=controller,
            w_seq=w_seq,
            x_seq=x_seq,
            scenario_a=scenario_small,
            scenario_b=scenario_large,
            args=args,
        )

    return {
        "loss": float(loss.item()),
        "terminal_dist": float(terminal.mean().item()),
        "collision_rate": float(collision_rate.item()),
        "collision_violation": float(violation.item()),
        "path_ratio": float(path_ratio.item()),
        "w_mean_tgt1": float(w_mean_tgt1.item()),
        "ctx_sens_ratio": sens_ratio,
        "ctx_sens_u_diff": sens_u_diff,
    }, x_seq, u_seq


def plot_loss_curve(loss_hist, run_dir, show_plots: bool):
    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_hist, marker="o")
    ax.set_title("Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"))
    if not show_plots:
        plt.close(fig)


def plot_trajectories(x_seq, scenario, run_dir, args, max_plots=3, show_plots: bool = False):
    plt = get_plt(show_plots)
    num = min(max_plots, x_seq.shape[0])
    fig, axes = plt.subplots(1, num, figsize=(5 * num, 5), squeeze=False)
    for i in range(num):
        ax = axes[0, i]
        traj = x_seq[i, :, :2].cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], label="traj")
        ax.scatter([scenario.start[i, 0]], [scenario.start[i, 1]], color="green", label="start")
        ax.scatter([0.0], [0.0], color="red", marker="*", label="goal")

        centers = scenario.centers[i].cpu().numpy()
        radii = scenario.radii[i].cpu().numpy()
        for k in range(centers.shape[0]):
            circle = plt.Circle((centers[k, 0], centers[k, 1]), radii[k], color="gray", alpha=0.3)
            ax.add_patch(circle)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-args.start_box, args.start_box)
        ax.set_ylim(-args.start_box, args.start_box)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "trajectories.png"))
    if not show_plots:
        plt.close(fig)


def plot_min_dist_hist(min_dist, run_dir, show_plots: bool):
    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(min_dist.flatten(), bins=30, density=True, alpha=0.7)
    ax.set_title("Min distance to obstacle edge")
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "min_dist_hist.png"))
    if not show_plots:
        plt.close(fig)


def plot_loss_heatmap(controller, plant_true, scenario, args, run_dir, show_plots: bool):
    """Plot loss heatmaps on the same start grid for 3 radius levels (small/median/large)."""
    plt = get_plt(show_plots)
    res = args.heatmap_res
    r = args.start_box

    xs = torch.linspace(-r, r, res)
    ys = torch.linspace(-r, r, res)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    starts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    B = starts.shape[0]

    case_indices = _select_radius_case_indices(scenario.radii)
    if len(case_indices) == 0:
        return
    case_indices = case_indices[:3]
    case_labels = ["small", "median", "large"][:len(case_indices)]

    centers_base = scenario.centers[0]
    goal = torch.zeros(B, 2, dtype=starts.dtype, device=starts.device)

    grid_tensors = []
    radii_levels = []

    controller.eval()
    with torch.no_grad():
        for idx in case_indices:
            radii_level = scenario.radii[idx]
            centers = centers_base.unsqueeze(0).repeat(B, 1, 1)
            radii = radii_level.unsqueeze(0).repeat(B, 1)

            grid_scenario = type(scenario)(
                start=starts,
                goal=goal,
                centers=centers,
                radii=radii,
            )

            x0 = make_rollout_inputs(grid_scenario)
            ctx_fn = make_context_fn(grid_scenario)
            u_post = make_u_post_fn(args, grid_scenario)
            x_seq, u_seq, _ = rollout_bptt(
                controller=controller,
                plant_true=plant_true,
                x0=x0,
                horizon=args.horizon,
                context_fn=ctx_fn,
                w0=x0,
                u_post_fn=u_post,
            )
            loss = loss_per_sample(x_seq, u_seq, grid_scenario, args)

            # Mask invalid starts (inside any obstacle for this radius level).
            rel = starts.unsqueeze(1) - centers
            dist = torch.norm(rel, dim=-1)
            invalid = (dist <= radii).any(dim=1)
            loss = loss.clone()
            loss[invalid] = float("nan")

            grid_tensors.append(loss.reshape(res, res).cpu())
            radii_levels.append(radii_level.cpu())

    finite_vals = [g[torch.isfinite(g)] for g in grid_tensors if torch.isfinite(g).any()]
    if len(finite_vals) > 0:
        finite_all = torch.cat(finite_vals)
        vmin = float(torch.quantile(finite_all, 0.05).item())
        vmax = float(torch.quantile(finite_all, 0.95).item())
        if vmax <= vmin + 1e-9:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = None, None

    n = len(grid_tensors)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    im = None
    centers_np = centers_base.cpu().numpy()

    for j in range(n):
        ax = axes[0, j]
        loss_grid = grid_tensors[j].numpy()
        im = ax.imshow(
            loss_grid.T,
            origin="lower",
            extent=[-r, r, -r, r],
            aspect="equal",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        radii_np = radii_levels[j].numpy()
        for k in range(centers_np.shape[0]):
            circle = plt.Circle((centers_np[k, 0], centers_np[k, 1]), radii_np[k], color="white", alpha=0.55, fill=True)
            ax.add_patch(circle)
            circle_edge = plt.Circle((centers_np[k, 0], centers_np[k, 1]), radii_np[k], color="black", fill=False, linewidth=1.0)
            ax.add_patch(circle_edge)

        lbl = case_labels[j] if j < len(case_labels) else f"case {j+1}"
        ax.set_title(f"{lbl} radii | mean r={float(radii_np.mean()):.2f}")
        ax.set_xlabel("x")
        if j == 0:
            ax.set_ylabel("y")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="cumulative loss")
    fig.suptitle("Loss Heatmap vs Start Position at 3 Radius Levels", y=0.98)
    fig.subplots_adjust(wspace=0.22, top=0.86, right=0.92)
    fig.savefig(os.path.join(run_dir, "loss_heatmap_radius_levels.png"))
    if not show_plots:
        plt.close(fig)


def plot_sample_trajectories(controller, plant_true, scenario, args, run_dir, show_plots: bool, num_samples: int = 6):
    plt = get_plt(show_plots)
    num = min(num_samples, scenario.start.shape[0])
    sample = type(scenario)(
        start=scenario.start[:num],
        goal=scenario.goal[:num],
        centers=scenario.centers[:num],
        radii=scenario.radii[:num],
    )

    x0 = make_rollout_inputs(sample)
    ctx_fn = make_context_fn(sample)
    u_post = make_u_post_fn(args, sample)

    controller.eval()
    with torch.no_grad():
        x_seq, _, _ = rollout_bptt(
            controller=controller,
            plant_true=plant_true,
            x0=x0,
            horizon=args.horizon,
            context_fn=ctx_fn,
            w0=x0,
            u_post_fn=u_post,
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(num):
        traj = x_seq[i, :, :2].cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.8)
        ax.scatter([sample.start[i, 0]], [sample.start[i, 1]], color="green", s=20)

    centers_np = sample.centers[0].cpu().numpy()
    radii_np = sample.radii[0].cpu().numpy()
    for k in range(centers_np.shape[0]):
        circle = plt.Circle((centers_np[k, 0], centers_np[k, 1]), radii_np[k], color="gray", alpha=0.3)
        ax.add_patch(circle)

    ax.scatter([0.0], [0.0], color="red", marker="*", s=80)
    ax.set_title("Sample trajectories")
    ax.set_xlim(-args.start_box, args.start_box)
    ax.set_ylim(-args.start_box, args.start_box)
    ax.set_aspect("equal", "box")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "sample_trajectories.png"))
    if not show_plots:
        plt.close(fig)


def _select_radius_case_indices(radii: torch.Tensor) -> List[int]:
    """Pick small/median/large radius cases from a radii batch."""
    if radii.ndim != 2:
        raise ValueError(f"Expected radii with shape (B,K), got {tuple(radii.shape)}")

    mean_r = radii.mean(dim=1)
    idx_small = int(torch.argmin(mean_r).item())
    idx_large = int(torch.argmax(mean_r).item())

    sorted_idx = torch.argsort(mean_r)
    idx_med = int(sorted_idx[len(sorted_idx) // 2].item())

    # Keep order small->median->large and uniqueness.
    idxs = [idx_small, idx_med, idx_large]
    unique = []
    for i in idxs:
        if i not in unique:
            unique.append(i)
    return unique


def plot_radius_comparison_trajectories(
    controller,
    plant_true,
    test_scenario,
    args,
    run_dir,
    show_plots: bool,
    num_starts: int = 4,
):
    """
    Single radius-comparison plot:
    4 challenging starts (anchored near a user-chosen point) and trajectories for
    increasing radius levels (small/median/large) overlaid on each start panel.
    """
    plt = get_plt(show_plots)

    case_indices = _select_radius_case_indices(test_scenario.radii)
    if len(case_indices) == 0:
        return
    case_indices = case_indices[:3]
    radius_cases = test_scenario.radii[case_indices]  # (R, K)
    centers = test_scenario.centers[0]                # (K, 2)
    r_levels = radius_cases.shape[0]

    anchor = torch.tensor(
        [args.radius_cmp_anchor_x, args.radius_cmp_anchor_y],
        dtype=centers.dtype,
        device=centers.device,
    )

    # Candidate starts around the anchor (includes the exact anchor, e.g. (2,0)).
    base_offsets = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.6],
            [0.0, -0.6],
            [-0.5, 0.35],
            [-0.5, -0.35],
            [0.4, 0.3],
            [0.4, -0.3],
            [-0.8, 0.0],
        ],
        dtype=centers.dtype,
        device=centers.device,
    )
    preset = (anchor.unsqueeze(0) + base_offsets).clamp(min=-args.start_box, max=args.start_box)

    large_r = radius_cases[-1]

    def _valid_and_challenging(starts: torch.Tensor):
        if starts.numel() == 0:
            return torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool)
        dist = torch.norm(starts.unsqueeze(1) - centers.unsqueeze(0), dim=-1)
        valid = (dist > large_r.unsqueeze(0)).all(dim=1)
        goals = torch.zeros(starts.shape[0], 2, dtype=starts.dtype, device=starts.device)
        challenging = direct_path_intersection_mask(
            starts=starts,
            goals=goals,
            centers=centers.unsqueeze(0).expand(starts.shape[0], -1, -1),
            radii=large_r.unsqueeze(0).expand(starts.shape[0], -1),
            margin=args.challenge_margin,
        )
        return valid, challenging

    valid_preset, challenging_preset = _valid_and_challenging(preset)
    selected = []
    for i in range(preset.shape[0]):
        if bool(valid_preset[i].item()) and bool(challenging_preset[i].item()):
            selected.append(preset[i])
        if len(selected) >= num_starts:
            break

    # If needed, supplement from challenging starts in the test set near the anchor.
    if len(selected) < num_starts:
        starts_all = test_scenario.start
        goals_all = test_scenario.goal
        dist_all = torch.norm(starts_all.unsqueeze(1) - centers.unsqueeze(0), dim=-1)
        valid_all = (dist_all > large_r.unsqueeze(0)).all(dim=1)
        challenging_all = direct_path_intersection_mask(
            starts=starts_all,
            goals=goals_all,
            centers=test_scenario.centers,
            radii=large_r.unsqueeze(0).expand(starts_all.shape[0], -1),
            margin=args.challenge_margin,
        )

        candidate_mask = valid_all & challenging_all
        if candidate_mask.sum().item() == 0:
            candidate_mask = valid_all

        candidate_starts = starts_all[candidate_mask]
        if candidate_starts.shape[0] > 0:
            d = torch.norm(candidate_starts - anchor.unsqueeze(0), dim=1)
            order = torch.argsort(d)
            for idx in order.tolist():
                if len(selected) >= num_starts:
                    break
                cand = candidate_starts[idx]
                duplicate = any(torch.norm(cand - s).item() < 1e-5 for s in selected)
                if not duplicate:
                    selected.append(cand)

    if len(selected) == 0:
        print("[warn] Radius comparison: no valid starts found.")
        return

    starts = torch.stack(selected[:num_starts], dim=0)
    s = starts.shape[0]

    starts_batch = starts.unsqueeze(1).expand(s, r_levels, 2).reshape(s * r_levels, 2)
    radii_batch = radius_cases.unsqueeze(0).expand(s, r_levels, -1).reshape(s * r_levels, -1)
    centers_batch = centers.unsqueeze(0).expand(s * r_levels, -1, -1)
    goals_batch = torch.zeros(s * r_levels, 2, dtype=starts.dtype, device=starts.device)

    scenario_cmp = type(test_scenario)(
        start=starts_batch,
        goal=goals_batch,
        centers=centers_batch,
        radii=radii_batch,
    )

    x0 = make_rollout_inputs(scenario_cmp)
    ctx_fn = make_context_fn(scenario_cmp)
    u_post = make_u_post_fn(args, scenario_cmp)

    controller.eval()
    with torch.no_grad():
        x_seq, _, _ = rollout_bptt(
            controller=controller,
            plant_true=plant_true,
            x0=x0,
            horizon=args.horizon,
            context_fn=ctx_fn,
            w0=x0,
            u_post_fn=u_post,
        )

    x_seq = x_seq.view(s, r_levels, x_seq.shape[1], x_seq.shape[2])

    # Diagnostics: how much the trajectory changes between small and large radii.
    if r_levels >= 2:
        d = torch.norm(x_seq[:, -1, :, :2] - x_seq[:, 0, :, :2], dim=-1)
        max_d = d.max(dim=1).values
        print(
            "[plot] Radius comparison max trajectory diffs:",
            ", ".join([f"{float(v.item()):.4f}" for v in max_d]),
        )

    labels = ["small", "median", "large"][:r_levels]
    colors = ["C0", "C1", "C3"][:r_levels]
    styles = ["-", "--", "-"][:r_levels]

    ncols = 2
    nrows = (s + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)

    centers_np = centers.cpu().numpy()
    radius_cases_np = radius_cases.cpu().numpy()

    for i in range(s):
        ax = axes[i // ncols, i % ncols]
        for j in range(r_levels):
            traj = x_seq[i, j, :, :2].cpu().numpy()
            lbl = labels[j] if j < len(labels) else f"level {j+1}"
            ax.plot(traj[:, 0], traj[:, 1], color=colors[j], linestyle=styles[j], linewidth=2.0, label=f"{lbl} (mean r={float(radius_cases_np[j].mean()):.2f})")

            for k in range(centers_np.shape[0]):
                c = plt.Circle(
                    (centers_np[k, 0], centers_np[k, 1]),
                    radius_cases_np[j, k],
                    color=colors[j],
                    fill=False,
                    alpha=0.35,
                    linestyle=styles[j],
                    linewidth=1.0,
                )
                ax.add_patch(c)

        ax.scatter([starts[i, 0]], [starts[i, 1]], color="green", s=30, label="start")
        ax.scatter([0.0], [0.0], color="red", marker="*", s=80, label="goal")
        ax.set_title(f"Start {i+1}: ({float(starts[i,0]):.2f}, {float(starts[i,1]):.2f})")
        ax.set_xlim(-args.start_box, args.start_box)
        ax.set_ylim(-args.start_box, args.start_box)
        ax.set_aspect("equal", "box")
        ax.grid(True)

    total_axes = nrows * ncols
    for i in range(s, total_axes):
        axes[i // ncols, i % ncols].axis("off")

    handles, labels_all = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_all, loc="upper right")
    fig.suptitle("Challenging Radius Comparison (4 starts, increasing radii)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "radius_comparison_challenging.png"), bbox_inches="tight")
    if not show_plots:
        plt.close(fig)

    meta = {
        "case_indices": [int(i) for i in case_indices],
        "case_mean_radii": [float(v) for v in radius_cases.mean(dim=1).cpu().tolist()],
        "anchor_start": [float(anchor[0].item()), float(anchor[1].item())],
        "selected_starts": [[float(x), float(y)] for x, y in starts.cpu().tolist()],
    }
    with open(os.path.join(run_dir, "radius_comparison_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def plot_m_outputs_over_time(controller, plant_true, scenario, args, run_dir, show_plots: bool, num_samples: int = 6):
    """
    Plot trajectories of M outputs over time.
    Uses unclamped rollout, so plotted signal is exactly output of M (u_nominal=None).
    """
    plt = get_plt(show_plots)
    x0 = make_rollout_inputs(scenario)
    ctx_fn = make_context_fn(scenario)

    controller.eval()
    with torch.no_grad():
        _, u_seq, _ = rollout_bptt(
            controller=controller,
            plant_true=plant_true,
            x0=x0,
            horizon=args.horizon,
            context_fn=ctx_fn,
            w0=x0,
            u_post_fn=None,
        )

    num = min(num_samples, u_seq.shape[0])
    T = u_seq.shape[1]
    ts = torch.arange(T).cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    for i in range(num):
        ui = u_seq[i].detach().cpu().numpy()
        axes[0].plot(ts, ui[:, 0], alpha=0.8)
        axes[1].plot(ts, ui[:, 1], alpha=0.8)

    u_norm_mean = torch.norm(u_seq.detach(), dim=-1).mean(dim=0).cpu().numpy()
    axes[2].plot(ts, u_norm_mean, color="black", linewidth=2)

    axes[0].set_ylabel("u_x")
    axes[1].set_ylabel("u_y")
    axes[2].set_ylabel("mean ||u_t||")
    axes[2].set_xlabel("time step")
    axes[0].set_title("M output trajectories over time (raw, unclamped)")
    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "m_outputs_over_time.png"))
    if not show_plots:
        plt.close(fig)



def print_operator_diagnostics(controller, w_dim: int, horizon: int = 64) -> None:
    """Quick sanity checks for M_p state reset and zero-input tail behavior."""
    mp = controller.operator.mp
    with torch.no_grad():
        if hasattr(mp, "reset"):
            mp.reset()
        w = torch.zeros(1, horizon, w_dim)
        w[:, 0, :2] = torch.tensor([1.0, -1.0])
        v1 = mp(w)
        if hasattr(mp, "reset"):
            mp.reset()
        v2 = mp(w)

    reset_err = float((v1 - v2).abs().max().item())
    tail = float(v1[:, -1, :].norm(dim=-1).mean().item())
    print(f"[diag] M_p reset reproducibility max diff: {reset_err:.3e}")
    print(f"[diag] M_p impulse-response tail ||v_T||: {tail:.3e}")

def build_fixed_obstacle_dataset(
    args: argparse.Namespace,
    *,
    batch_size: int,
    seed: int,
    min_fraction: float,
    fixed_centers: torch.Tensor,
    r_min: float | None = None,
    r_max: float | None = None,
    k: int | None = None,
):
    r_min_use = args.r_min if r_min is None else float(r_min)
    r_max_use = args.r_max if r_max is None else float(r_max)
    k_use = int(fixed_centers.shape[0]) if k is None else int(k)

    scenario = sample_scenarios(
        batch_size=batch_size,
        seed=seed,
        r_min=r_min_use,
        r_max=r_max_use,
        k=k_use,
        fixed_centers=fixed_centers,
        start_box=(-args.start_box, args.start_box),
    )

    use_diverse_angles = bool(args.challenge_diverse_angles or args.num_obstacles == 1)
    cone_fraction = 0.0 if use_diverse_angles else args.challenge_cone_fraction

    # ensure_challenging_starts uses torch.rand internally; isolate and seed it for determinism.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 17171)
        scenario, ch_info = ensure_challenging_starts(
            scenario,
            start_box=(-args.start_box, args.start_box),
            min_fraction=min_fraction,
            margin=args.challenge_margin,
            radius_quantile=args.challenge_radius_quantile,
            start_clearance=args.challenge_start_clearance,
            cone_fraction=cone_fraction,
            cone_angle_spread=args.challenge_cone_spread,
            cone_radial_extension=args.challenge_radial_extension,
            diverse_angles=use_diverse_angles,
            num_angle_bins=args.challenge_angle_bins,
        )

    return scenario, ch_info


def make_fixed_centers(args: argparse.Namespace) -> torch.Tensor:
    if args.num_obstacles == 1:
        return torch.tensor([[args.single_obstacle_x, args.single_obstacle_y]], dtype=torch.float32)
    if args.num_obstacles == 3:
        return torch.tensor(
            [
                [1.0, 0.0],
                [0.3, 0.8],
                [0.3, -0.8],
            ],
            dtype=torch.float32,
        )
    raise ValueError(f"Unsupported num_obstacles={args.num_obstacles}. Use 1 or 3.")


def single_obstacle_curriculum_params(args: argparse.Namespace, epoch: int):
    if not args.single_obstacle_curriculum:
        return args.r_min, args.r_max, args.challenge_train_frac, "fixed"

    if args.num_obstacles != 1:
        raise ValueError("--single_obstacle_curriculum requires --num_obstacles=1")

    e = max(1, int(epoch))
    total = max(1, int(args.epochs))
    frac = float(e) / float(total)

    if frac <= args.curriculum_stage1_frac:
        return (
            args.curriculum_easy_r_min,
            args.curriculum_easy_r_max,
            args.curriculum_easy_challenge_frac,
            "easy",
        )
    if frac <= args.curriculum_stage2_frac:
        return (
            args.curriculum_mid_r_min,
            args.curriculum_mid_r_max,
            args.curriculum_mid_challenge_frac,
            "mid",
        )
    return (
        args.curriculum_hard_r_min,
        args.curriculum_hard_r_max,
        args.curriculum_hard_challenge_frac,
        "hard",
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    show_plots = not args.no_show_plots

    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, "runs", "nav_experiment", run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    nominal = DoubleIntegratorNominal(dt=args.dt, pre_kp=args.pre_kp, pre_kd=args.pre_kd)
    plant_true = DoubleIntegratorTrue(dt=args.dt, pre_kp=args.pre_kp, pre_kd=args.pre_kd)
    if args.use_decaying_noise:
        print(
            "[info] Using perfect model + exogenous decaying process noise: "
            f"sigma0={args.noise_sigma0}, tau={args.noise_tau}, floor={args.noise_floor}."
        )
    else:
        print("[info] Using perfect-model rollout: w_t = 0 for t>=1 and w_0 = x_0.")

    if args.barrier_eps <= 0:
        raise ValueError(f"barrier_eps must be > 0, got {args.barrier_eps}")
    if args.single_obstacle_curriculum and not (
        0.0 < args.curriculum_stage1_frac < args.curriculum_stage2_frac < 1.0
    ):
        raise ValueError(
            "Require 0 < curriculum_stage1_frac < curriculum_stage2_frac < 1 when curriculum is enabled."
        )
    if args.single_obstacle_curriculum and not args.resample_train_each_epoch:
        print("[data] enabling --resample_train_each_epoch because curriculum needs epoch-wise resampling.")
        args.resample_train_each_epoch = True

    print(
        f"[info] Loss mode: {args.loss_mode} | "
        f"terminal={args.terminal_weight}, stage={args.stage_weight}, path={args.path_weight}, "
        f"coll={args.collision_weight}, viol={args.collision_violation_weight}, "
        f"u={args.control_weight}, du={args.smooth_weight}"
    )

    w_dim = 4
    u_dim = 2

    fixed_centers = make_fixed_centers(args)
    k = fixed_centers.shape[0]
    z_dim = 2 + 2 * k + k + k  # goal_dir + rel_pos + dist_to_edge + radii

    print(f"[data] num_obstacles={k} | fixed_centers={fixed_centers.tolist()}")
    if args.challenge_diverse_angles or args.num_obstacles == 1:
        print(f"[data] challenging-start angular balancing enabled ({args.challenge_angle_bins} bins).")
    print(f"[data] challenging-start min clearance from obstacle edge: {args.challenge_start_clearance:.2f}")

    mp = MpDeepSSM(
        w_dim,
        args.feat_dim,
        mode=args.mp_mode,
        reset_state_each_call=False,
        detach_state=args.mp_detach_state,
        param=args.mp_param,
        n_layers=args.mp_layers,
        d_model=args.mp_d_model,
        d_state=args.mp_d_state,
        ff=args.mp_ff,
        dropout=args.mp_dropout,
        max_phase_b=None,
        rho=args.mp_rho,
        rmin=args.mp_rmin,
        rmax=args.mp_rmax,
        gamma=args.mp_gamma,
    )
    mb = BoundedMLPOperator(
        w_dim=w_dim,
        z_dim=z_dim,
        r=u_dim,
        s=args.feat_dim,
        hidden_dim=args.hidden,
        gamma=args.gamma,
        use_z_residual=not args.mb_no_z_residual,
        z_residual_gain=args.mb_z_residual_gain,
        bound_mode=args.mb_bound_mode,
        clamp_value=args.mb_bound_value,
    )
    op = FactorizedOperator(mp, mb)

    controller = PBController(
        plant=nominal,
        operator=op,
        u_nominal=None,
        u_dim=u_dim,
        detach_state=False,
    )

    print_operator_diagnostics(controller, w_dim=w_dim, horizon=max(args.horizon, 32))
    if args.mp_ff.upper() in {"GLU", "LGLU"}:
        print("[warn] mp_ff uses biased gates; M_p(0) may be non-zero, so u_t may not decay to 0.")
    if args.mb_no_z_residual:
        print("[info] M_b z-residual branch disabled.")
    else:
        print(f"[info] M_b z-residual branch enabled (gain={args.mb_z_residual_gain:.2f}).")
    print(f"[info] M_b bound mode: {args.mb_bound_mode} (bound_value={args.mb_bound_value:.2f}).")
    print(
        f"[info] M_p spectrum params: rho={args.mp_rho:.4f}, "
        f"rmin={args.mp_rmin:.4f}, rmax={args.mp_rmax:.4f}, gamma={args.mp_gamma:.2f}"
    )

    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)

    train_r_min_0, train_r_max_0, train_ch_frac_0, train_stage_0 = single_obstacle_curriculum_params(args, epoch=1)
    train_scenario, train_ch_info = build_fixed_obstacle_dataset(
        args,
        batch_size=args.batch,
        seed=args.seed + 1,
        min_fraction=train_ch_frac_0,
        fixed_centers=fixed_centers,
        r_min=train_r_min_0,
        r_max=train_r_max_0,
        k=args.num_obstacles,
    )

    if args.single_obstacle_curriculum and args.num_obstacles == 1:
        test_r_min = args.curriculum_hard_r_min
        test_r_max = args.curriculum_hard_r_max
    else:
        test_r_min = args.r_min
        test_r_max = args.r_max

    test_scenario, test_ch_info = build_fixed_obstacle_dataset(
        args,
        batch_size=args.test_batch,
        seed=args.seed + 999,
        min_fraction=args.challenge_test_frac,
        fixed_centers=fixed_centers,
        r_min=test_r_min,
        r_max=test_r_max,
        k=args.num_obstacles,
    )

    if args.resample_train_each_epoch:
        print("[data] train scenarios: resampled each epoch (deterministic seed schedule).")
    else:
        print("[data] train scenarios: fixed across epochs.")
    if args.single_obstacle_curriculum:
        print(
            "[data] single-obstacle curriculum: "
            f"easy[{args.curriculum_easy_r_min:.2f},{args.curriculum_easy_r_max:.2f}] -> "
            f"mid[{args.curriculum_mid_r_min:.2f},{args.curriculum_mid_r_max:.2f}] -> "
            f"hard[{args.curriculum_hard_r_min:.2f},{args.curriculum_hard_r_max:.2f}]"
        )
    print(
        f"[data] train challenging starts: {train_ch_info['after_fraction']:.2f} "
        f"(target {train_ch_info['target_fraction']:.2f}, replaced {train_ch_info['replaced']}, "
        f"cone {train_ch_info.get('cone_replaced', 0)}, diverse {train_ch_info.get('diverse_angles', False)})"
    )
    print(
        f"[data] test challenging starts: {test_ch_info['after_fraction']:.2f} "
        f"(target {test_ch_info['target_fraction']:.2f}, replaced {test_ch_info['replaced']}, "
        f"cone {test_ch_info.get('cone_replaced', 0)}, diverse {test_ch_info.get('diverse_angles', False)})"
    )

    train_noise_fixed = maybe_make_process_noise(
        args,
        batch_size=args.batch,
        horizon=args.horizon,
        state_dim=w_dim,
        seed=args.noise_seed + 1,
        device=train_scenario.start.device,
        dtype=train_scenario.start.dtype,
    )
    test_noise = maybe_make_process_noise(
        args,
        batch_size=args.test_batch,
        horizon=args.horizon,
        state_dim=w_dim,
        seed=args.noise_seed + 2,
        device=test_scenario.start.device,
        dtype=test_scenario.start.dtype,
    )

    if args.use_decaying_noise:
        print(
            f"[noise] train_fixed={'off' if args.noise_resample_each_epoch else 'on'} | "
            f"sigma0={args.noise_sigma0:.4f} | tau={args.noise_tau:.2f} | floor={args.noise_floor:.4f}"
        )

    loss_hist = []
    eval_hist = []
    train_ch_fraction_hist = []
    best_eval_loss = float("inf")
    best_eval_collision = float("inf")
    best_epoch = -1
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        controller.train()

        train_r_min_epoch, train_r_max_epoch, train_ch_frac_epoch, train_stage_epoch = single_obstacle_curriculum_params(
            args,
            epoch=epoch,
        )

        if args.resample_train_each_epoch:
            train_scenario, train_ch_info_epoch = build_fixed_obstacle_dataset(
                args,
                batch_size=args.batch,
                seed=args.seed + epoch,
                min_fraction=train_ch_frac_epoch,
                fixed_centers=fixed_centers,
                r_min=train_r_min_epoch,
                r_max=train_r_max_epoch,
                k=args.num_obstacles,
            )
        else:
            train_ch_info_epoch = train_ch_info
            train_stage_epoch = train_stage_0
            train_r_min_epoch = train_r_min_0
            train_r_max_epoch = train_r_max_0
        train_ch_fraction_hist.append(float(train_ch_info_epoch["after_fraction"]))

        x0 = make_rollout_inputs(train_scenario)
        ctx_fn = make_context_fn(train_scenario)
        u_post = make_u_post_fn(args, train_scenario)

        if args.use_decaying_noise and args.noise_resample_each_epoch:
            train_noise = maybe_make_process_noise(
                args,
                batch_size=args.batch,
                horizon=args.horizon,
                state_dim=w_dim,
                seed=args.noise_seed + 1000 + epoch,
                device=x0.device,
                dtype=x0.dtype,
            )
        else:
            train_noise = train_noise_fixed

        x_seq, u_seq, w_seq = rollout_bptt(
            controller=controller,
            plant_true=plant_true,
            x0=x0,
            horizon=args.horizon,
            context_fn=ctx_fn,
            w0=x0,
            u_post_fn=u_post,
            process_noise_seq=train_noise,
        )
        main_loss = loss_fn(x_seq, u_seq, train_scenario, args)
        loss = main_loss
        if w_seq.shape[1] > 1:
            w_mean_train = float(w_seq[:, 1:, :].abs().mean().item())
        else:
            w_mean_train = 0.0

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip)
        optimizer.step()

        loss_hist.append(float(loss.item()))

        if epoch % args.eval_every == 0 or epoch == 1:
            metrics, _, _ = evaluate(
                controller,
                plant_true,
                test_scenario,
                args.horizon,
                args,
                process_noise_seq=test_noise,
            )
            eval_hist.append({"epoch": epoch, "loss": metrics["loss"], "ctx_sens_ratio": metrics["ctx_sens_ratio"]})
            if args.best_ckpt_metric == "loss":
                better_ckpt = metrics["loss"] < best_eval_loss
            else:
                coll = float(metrics["collision_rate"])
                tol = float(args.best_ckpt_collision_tol)
                better_ckpt = (
                    coll < (best_eval_collision - tol)
                    or (abs(coll - best_eval_collision) <= tol and metrics["loss"] < best_eval_loss)
                )

            if better_ckpt:
                best_eval_loss = metrics["loss"]
                best_eval_collision = float(metrics["collision_rate"])
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": controller.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval_loss": best_eval_loss,
                        "best_eval_collision": best_eval_collision,
                        "best_ckpt_metric": args.best_ckpt_metric,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )
            print(
                f"epoch {epoch:03d} | train {loss.item():.6f} | "
                f"main {main_loss.item():.6f} | "
                f"test {metrics['loss']:.6f} | "
                f"terminal {metrics['terminal_dist']:.4f} | "
                f"collision {metrics['collision_rate']:.3f} | "
                f"viol {metrics['collision_violation']:.4f} | "
                f"path_ratio {metrics['path_ratio']:.3f} | "
                f"stage {train_stage_epoch} | "
                f"r[{train_r_min_epoch:.2f},{train_r_max_epoch:.2f}] | "
                f"chall {train_ch_info_epoch['after_fraction']:.2f} | "
                f"ctx_ratio {metrics['ctx_sens_ratio']:.3f} | "
                f"w_train {w_mean_train:.4f} | "
                f"w_test {metrics['w_mean_tgt1']:.4f}"
            )

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        controller.load_state_dict(ckpt["model_state_dict"])

    metrics, x_test, _ = evaluate(
        controller,
        plant_true,
        test_scenario,
        args.horizon,
        args,
        process_noise_seq=test_noise,
    )
    train_ch_mean = float(sum(train_ch_fraction_hist) / max(len(train_ch_fraction_hist), 1))
    train_ch_last = (
        float(train_ch_fraction_hist[-1])
        if train_ch_fraction_hist
        else float(train_ch_info["after_fraction"])
    )

    metrics["best_eval_loss"] = float(best_eval_loss)
    metrics["best_eval_collision"] = float(best_eval_collision)
    metrics["best_ckpt_metric"] = str(args.best_ckpt_metric)
    metrics["best_epoch"] = int(best_epoch)
    metrics["train_challenging_fraction"] = train_ch_mean
    metrics["train_challenging_fraction_mean"] = train_ch_mean
    metrics["train_challenging_fraction_last"] = train_ch_last
    metrics["test_challenging_fraction"] = float(test_ch_info["after_fraction"])
    metrics["resample_train_each_epoch"] = bool(args.resample_train_each_epoch)
    metrics["num_obstacles"] = int(args.num_obstacles)
    metrics["single_obstacle_curriculum"] = bool(args.single_obstacle_curriculum)
    metrics["challenge_diverse_angles"] = bool(train_ch_info.get("diverse_angles", False))
    metrics["challenge_angle_bins"] = int(train_ch_info.get("num_angle_bins", 0))
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "eval_history.json"), "w", encoding="utf-8") as f:
        json.dump(eval_hist, f, indent=2)

    plot_loss_curve(loss_hist, run_dir, show_plots)
    plot_trajectories(x_test, test_scenario, run_dir, args, show_plots=show_plots)
    min_dist = min_dist_to_edge(x_test, test_scenario).cpu().numpy()
    plot_min_dist_hist(min_dist, run_dir, show_plots)
    plot_loss_heatmap(controller, plant_true, test_scenario, args, run_dir, show_plots)
    plot_sample_trajectories(controller, plant_true, test_scenario, args, run_dir, show_plots)
    plot_radius_comparison_trajectories(
        controller,
        plant_true,
        test_scenario,
        args,
        run_dir,
        show_plots,
        num_starts=args.radius_cmp_num_starts,
    )
    plot_m_outputs_over_time(controller, plant_true, test_scenario, args, run_dir, show_plots)
    print(f"Saved plots and metrics to {run_dir}")
    if show_plots:
        plt = get_plt(True)
        plt.show()



if __name__ == "__main__":
    main()
