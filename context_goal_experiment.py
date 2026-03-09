"""
Contextual goal PB experiment.

Reliable demonstration of contextual utility:
  - same robot dynamics and similar starts
  - context changes target location and obstacle radius
  - objective uses only distance-to-target + barrier
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from bounded_mlp_operator import BoundedMLPOperator
from context_goal_env import (
    GoalScenario,
    build_goal_context,
    exp_barrier_penalty,
    min_dist_to_edge,
    sample_goal_scenarios,
    scenario_to_device,
    shuffled_context_scenario,
)
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_controller import PBController, FactorizedOperator
from rollout_bptt import rollout_bptt
from ssm_operators import MpDeepSSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Contextual goal PB experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--val_batch", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=250)
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

    # Scenario.
    parser.add_argument("--start_x_min", type=float, default=2.0)
    parser.add_argument("--start_x_max", type=float, default=2.3)
    parser.add_argument("--start_y_min", type=float, default=-0.12)
    parser.add_argument("--start_y_max", type=float, default=0.12)
    parser.add_argument("--theta_min", type=float, default=-1.0)
    parser.add_argument("--theta_max", type=float, default=1.0)
    parser.add_argument("--goal_x", type=float, default=-0.1)
    parser.add_argument("--goal_y_scale", type=float, default=0.85)
    parser.add_argument("--obs_center_x", type=float, default=0.9)
    parser.add_argument("--obs_center_y", type=float, default=0.0)
    parser.add_argument("--rad_min", type=float, default=0.35)
    parser.add_argument("--rad_max", type=float, default=0.65)

    # Loss: distance + barrier only.
    parser.add_argument("--w_term", type=float, default=35.0)
    parser.add_argument("--w_stage", type=float, default=1.0)
    parser.add_argument("--w_bar", type=float, default=100.0)
    parser.add_argument("--bar_margin", type=float, default=0.16)
    parser.add_argument("--bar_alpha", type=float, default=18.0)
    parser.add_argument("--bar_cap", type=float, default=500.0)

    parser.add_argument("--noise_sigma0", type=float, default=0.01)
    parser.add_argument("--noise_tau", type=float, default=20.0)

    parser.add_argument("--goal_tol", type=float, default=0.18)
    parser.add_argument(
        "--best_ckpt_metric",
        type=str,
        default="success_then_loss",
        choices=["loss", "collision_then_loss", "success_then_loss"],
    )
    parser.add_argument("--best_ckpt_collision_tol", type=float, default=1e-4)
    parser.add_argument("--z_gain", type=float, default=6.0)
    parser.add_argument("--z_residual_gain", type=float, default=12.0)

    # Plot controls.
    parser.add_argument("--fan_num", type=int, default=14)
    parser.add_argument("--phase_theta_res", type=int, default=70)
    parser.add_argument("--phase_rad_res", type=int, default=70)
    parser.add_argument("--phase_batch", type=int, default=2048)
    parser.add_argument("--phase_start_x", type=float, default=2.1)
    parser.add_argument("--phase_start_y", type=float, default=0.0)
    parser.add_argument("--skip_ablations", action="store_true")
    return parser.parse_args()


@dataclass
class LossCfg:
    w_term: float
    w_stage: float
    w_bar: float
    bar_margin: float
    bar_alpha: float
    bar_cap: float


@dataclass
class NoiseCfg:
    sigma0: float
    tau: float


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


def make_x0(scenario: GoalScenario, device: torch.device) -> torch.Tensor:
    b = scenario.start.shape[0]
    return torch.cat([scenario.start.to(device), torch.zeros(b, 2, device=device)], dim=-1).unsqueeze(1)


def make_decaying_noise(b: int, horizon: int, nx: int, cfg: NoiseCfg, device: torch.device) -> torch.Tensor:
    t = torch.arange(horizon, device=device).view(1, horizon, 1)
    sigma_t = cfg.sigma0 * torch.exp(-t / max(cfg.tau, 1e-6))
    return torch.randn(b, horizon, nx, device=device) * sigma_t


def rollout_goal(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: GoalScenario,
    *,
    horizon: int,
    device: torch.device,
    noise: torch.Tensor | None,
    context_scenario: GoalScenario | None = None,
    zero_context: bool = False,
    z_gain: float = 6.0,
):
    x0 = make_x0(scenario, device)
    if context_scenario is None:
        context_scenario = scenario

    def ctx_fn(x, t):
        z = build_goal_context(x, context_scenario, z_gain=z_gain)
        if zero_context:
            return torch.zeros_like(z)
        return z

    return rollout_bptt(
        controller=controller,
        plant_true=plant_true,
        x0=x0,
        horizon=horizon,
        context_fn=ctx_fn,
        w0=x0,
        process_noise_seq=noise,
    )


def compute_loss_per_sample(x_seq: torch.Tensor, scenario: GoalScenario, cfg: LossCfg) -> torch.Tensor:
    goal = scenario.goal.to(x_seq.device).unsqueeze(1)  # (B,1,2)
    dist = torch.norm(x_seq[..., :2] - goal, dim=-1)
    term = dist[:, -1] * cfg.w_term
    stage = dist.mean(dim=1) * cfg.w_stage
    bar = exp_barrier_penalty(
        x_seq,
        scenario,
        margin=cfg.bar_margin,
        alpha=cfg.bar_alpha,
        cap=cfg.bar_cap,
    ).mean(dim=(1, 2)) * cfg.w_bar
    return term + stage + bar


def compute_loss(x_seq: torch.Tensor, scenario: GoalScenario, cfg: LossCfg) -> tuple[torch.Tensor, dict]:
    total_per = compute_loss_per_sample(x_seq, scenario, cfg)
    total = total_per.mean()
    goal = scenario.goal.to(x_seq.device).unsqueeze(1)
    dist = torch.norm(x_seq[..., :2] - goal, dim=-1)
    term = dist[:, -1].mean() * cfg.w_term
    stage = dist.mean() * cfg.w_stage
    bar = exp_barrier_penalty(
        x_seq,
        scenario,
        margin=cfg.bar_margin,
        alpha=cfg.bar_alpha,
        cap=cfg.bar_cap,
    ).mean() * cfg.w_bar
    parts = {
        "loss_total": float(total.item()),
        "loss_term": float(term.item()),
        "loss_stage": float(stage.item()),
        "loss_barrier": float(bar.item()),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    scenario: GoalScenario,
    *,
    horizon: int,
    device: torch.device,
    loss_cfg: LossCfg,
    goal_tol: float,
    context_mode: str,
    z_gain: float,
):
    if context_mode not in {"true", "shuffled", "zero"}:
        raise ValueError(f"Unknown context_mode {context_mode}")
    if context_mode == "true":
        ctx_sc = scenario
        zero_ctx = False
    elif context_mode == "shuffled":
        ctx_sc = shuffled_context_scenario(scenario)
        zero_ctx = False
    else:
        ctx_sc = scenario
        zero_ctx = True

    controller.eval()
    x_seq, u_seq, _ = rollout_goal(
        controller=controller,
        plant_true=plant_true,
        scenario=scenario,
        horizon=horizon,
        device=device,
        noise=None,
        context_scenario=ctx_sc,
        zero_context=zero_ctx,
        z_gain=z_gain,
    )
    loss, parts = compute_loss(x_seq, scenario, loss_cfg)

    min_edge = min_dist_to_edge(x_seq, scenario)
    collided = min_edge.min(dim=1).values < 0.0
    collision_rate = float(collided.float().mean().item())

    terminal = torch.norm(x_seq[:, -1, :2] - scenario.goal.to(x_seq.device), dim=-1)
    goal_success = (~collided) & (terminal < float(goal_tol))
    success_rate = float(goal_success.float().mean().item())
    terminal_dist = float(terminal.mean().item())

    goal_y = scenario.goal[:, 1].to(x_seq.device)
    yT = x_seq[:, -1, 1]
    valid_sign = torch.abs(goal_y) > 1e-4
    if bool(valid_sign.any().item()):
        sign_acc = float((torch.sign(yT[valid_sign]) == torch.sign(goal_y[valid_sign])).float().mean().item())
    else:
        sign_acc = float("nan")

    parts.update(
        {
            "context_mode": context_mode,
            "loss": float(loss.item()),
            "collision_rate": collision_rate,
            "success_rate": success_rate,
            "terminal_dist": terminal_dist,
            "goal_sign_acc": sign_acc,
        }
    )
    return parts, x_seq, u_seq


def plot_loss_curves(train_hist, eval_hist, run_dir, show_plots: bool):
    if not train_hist:
        return
    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([d["epoch"] for d in train_hist], [d["loss"] for d in train_hist], label="train", alpha=0.85)
    if eval_hist:
        ax.plot([d["epoch"] for d in eval_hist], [d["loss"] for d in eval_hist], label="val", linewidth=2.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training/Validation Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"))
    if not show_plots:
        plt.close(fig)


def plot_trajectory_fan(x_seq, scenario, run_dir, show_plots: bool, fan_num: int):
    plt = get_plt(show_plots)
    b = x_seq.shape[0]
    n = min(int(fan_num), b)
    if n <= 0:
        return
    idx = torch.argsort(scenario.theta)[:n]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("coolwarm")
    for j, i_t in enumerate(idx):
        i = int(i_t.item())
        traj = x_seq[i, :, :2].detach().cpu().numpy()
        color = cmap(j / max(n - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0)
        ax.scatter([scenario.goal[i, 0].item()], [scenario.goal[i, 1].item()], color=color, s=18)
    ax.scatter([scenario.start[0, 0].item()], [scenario.start[0, 1].item()], color="green", s=45, label="start region")
    c = scenario.centers[0, 0].detach().cpu().numpy()
    r = float(scenario.radii[0, 0].item())
    circ = plt.Circle((c[0], c[1]), r, color="gray", alpha=0.3)
    ax.add_patch(circ)
    ax.set_title("Trajectory Fan Across Context (theta)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "trajectory_fan.png"))
    if not show_plots:
        plt.close(fig)


@torch.no_grad()
def plot_same_start_context_sweep(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    *,
    device: torch.device,
    horizon: int,
    run_dir: str,
    show_plots: bool,
    start_x: float,
    start_y: float,
    theta_min: float,
    theta_max: float,
    num_theta: int,
    goal_x: float,
    goal_y_scale: float,
    obs_center_x: float,
    obs_center_y: float,
    radius: float,
    z_gain: float,
):
    plt = get_plt(show_plots)
    th = torch.linspace(theta_min, theta_max, num_theta, device=device)
    b = th.shape[0]
    start = torch.tensor([start_x, start_y], device=device, dtype=torch.float32).view(1, 2).expand(b, -1)
    goal = torch.stack([torch.full_like(th, goal_x), goal_y_scale * torch.sin(th)], dim=-1)
    centers = torch.tensor([obs_center_x, obs_center_y], device=device, dtype=torch.float32).view(1, 1, 2).expand(b, -1, -1)
    radii = torch.full((b, 1), float(radius), device=device)
    sc = GoalScenario(start=start, goal=goal, centers=centers, radii=radii, theta=th)
    x_seq, _, _ = rollout_goal(
        controller=controller,
        plant_true=plant_true,
        scenario=sc,
        horizon=horizon,
        device=device,
        noise=None,
        z_gain=z_gain,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("coolwarm")
    for i in range(b):
        traj = x_seq[i, :, :2].detach().cpu().numpy()
        color = cmap(i / max(b - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0)
        ax.scatter([goal[i, 0].item()], [goal[i, 1].item()], color=color, s=22)
    ax.scatter([start_x], [start_y], color="green", s=60, marker="o")
    circ = plt.Circle((obs_center_x, obs_center_y), radius, color="gray", alpha=0.3)
    ax.add_patch(circ)
    ax.set_title("Same Start, Different Context Targets")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "same_start_context_sweep.png"))
    if not show_plots:
        plt.close(fig)


@torch.no_grad()
def plot_phase_heatmaps(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    *,
    device: torch.device,
    horizon: int,
    run_dir: str,
    show_plots: bool,
    theta_min: float,
    theta_max: float,
    rad_min: float,
    rad_max: float,
    theta_res: int,
    rad_res: int,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y_scale: float,
    obs_center_x: float,
    obs_center_y: float,
    batch: int,
    z_gain: float,
):
    plt = get_plt(show_plots)
    th_vals = torch.linspace(theta_min, theta_max, theta_res, device=device)
    r_vals = torch.linspace(rad_min, rad_max, rad_res, device=device)
    th_grid, r_grid = torch.meshgrid(th_vals, r_vals, indexing="ij")
    n = th_grid.numel()

    start = torch.tensor([start_x, start_y], device=device, dtype=torch.float32).view(1, 2).expand(n, -1)
    th = th_grid.reshape(-1)
    rad = r_grid.reshape(-1)
    goal = torch.stack([torch.full_like(th, goal_x), goal_y_scale * torch.sin(th)], dim=-1)
    centers = torch.tensor([obs_center_x, obs_center_y], device=device, dtype=torch.float32).view(1, 1, 2).expand(n, -1, -1)
    radii = rad.view(-1, 1)
    sc_all = GoalScenario(start=start, goal=goal, centers=centers, radii=radii, theta=th)

    term = torch.zeros(n, device=device)
    coll = torch.zeros(n, device=device)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        sc = GoalScenario(
            start=sc_all.start[i0:i1],
            goal=sc_all.goal[i0:i1],
            centers=sc_all.centers[i0:i1],
            radii=sc_all.radii[i0:i1],
            theta=sc_all.theta[i0:i1],
        )
        x_seq, _, _ = rollout_goal(
            controller=controller,
            plant_true=plant_true,
            scenario=sc,
            horizon=horizon,
            device=device,
            noise=None,
            z_gain=z_gain,
        )
        term[i0:i1] = torch.norm(x_seq[:, -1, :2] - sc.goal, dim=-1)
        coll[i0:i1] = (min_dist_to_edge(x_seq, sc).min(dim=1).values < 0.0).float()

    term_map = term.view(theta_res, rad_res).detach().cpu().numpy()
    coll_map = coll.view(theta_res, rad_res).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    extent = [rad_min, rad_max, theta_min, theta_max]
    im0 = axes[0].imshow(term_map, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0].set_title("Terminal Distance")
    axes[0].set_xlabel("obstacle radius")
    axes[0].set_ylabel("theta")
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].imshow(coll_map, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Collision Rate Map")
    axes[1].set_xlabel("obstacle radius")
    axes[1].set_ylabel("theta")
    fig.colorbar(im1, ax=axes[1], shrink=0.9)

    fig.suptitle("Context Phase Heatmaps", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "context_phase_heatmaps.png"), bbox_inches="tight")
    if not show_plots:
        plt.close(fig)


def plot_ablation(metrics_true, metrics_shuf, metrics_zero, run_dir, show_plots: bool):
    plt = get_plt(show_plots)
    labels = ["true", "shuffled", "zero"]
    success = [metrics_true["success_rate"], metrics_shuf["success_rate"], metrics_zero["success_rate"]]
    coll = [metrics_true["collision_rate"], metrics_shuf["collision_rate"], metrics_zero["collision_rate"]]
    sign = [
        metrics_true["goal_sign_acc"] if not torch.isnan(torch.tensor(metrics_true["goal_sign_acc"])) else 0.0,
        metrics_shuf["goal_sign_acc"] if not torch.isnan(torch.tensor(metrics_shuf["goal_sign_acc"])) else 0.0,
        metrics_zero["goal_sign_acc"] if not torch.isnan(torch.tensor(metrics_zero["goal_sign_acc"])) else 0.0,
    ]
    x = torch.arange(3).cpu().numpy()
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, success, width=w, label="success", color="C2")
    ax.bar(x, coll, width=w, label="collision", color="C3")
    ax.bar(x + w, sign, width=w, label="goal-sign acc", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Context Ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "ablation_context_bars.png"))
    if not show_plots:
        plt.close(fig)


def build_controller(
    device: torch.device,
    *,
    z_residual_gain: float,
    nom_pre_kp: float,
    nom_pre_kd: float,
) -> PBController:
    w_dim = 4
    u_dim = 2
    feat_dim = 16
    z_dim = 3

    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=nom_pre_kp, pre_kd=nom_pre_kd)
    mp = MpDeepSSM(
        w_dim,
        feat_dim,
        mode="loop",
        param="lru",
        n_layers=4,
        d_model=16,
        d_state=32,
        ff="GLU",
    ).to(device)
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
    return PBController(
        plant=plant_nom,
        operator=FactorizedOperator(mp, mb),
        u_nominal=None,
        u_dim=u_dim,
        detach_state=False,
    ).to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    show_plots = not args.no_show_plots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        "context_goal_experiment",
        f"context_goal_{run_id}",
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    loss_cfg = LossCfg(
        w_term=args.w_term,
        w_stage=args.w_stage,
        w_bar=args.w_bar,
        bar_margin=args.bar_margin,
        bar_alpha=args.bar_alpha,
        bar_cap=args.bar_cap,
    )
    noise_cfg = NoiseCfg(sigma0=args.noise_sigma0, tau=args.noise_tau)

    controller = build_controller(
        device=device,
        z_residual_gain=args.z_residual_gain,
        nom_pre_kp=args.nom_pre_kp,
        nom_pre_kd=args.nom_pre_kd,
    )
    plant_true = DoubleIntegratorTrue(dt=0.05, pre_kp=args.true_pre_kp, pre_kd=args.true_pre_kd)
    optimizer = optim.Adam(controller.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    def _sample(bsz: int, seed: int):
        return sample_goal_scenarios(
            batch_size=bsz,
            seed=seed,
            start_x=(args.start_x_min, args.start_x_max),
            start_y=(args.start_y_min, args.start_y_max),
            theta_range=(args.theta_min, args.theta_max),
            goal_x=args.goal_x,
            goal_y_scale=args.goal_y_scale,
            center_x=args.obs_center_x,
            center_y=args.obs_center_y,
            radius_range=(args.rad_min, args.rad_max),
        )

    val_sc_cpu = _sample(args.val_batch, args.seed + 999)
    val_sc = scenario_to_device(val_sc_cpu, device)

    print(f"Starting contextual-goal experiment on {device}")
    train_hist: List[dict] = []
    eval_hist: List[dict] = []

    best_path = os.path.join(run_dir, "best_model.pt")
    init_m, _, _ = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_sc,
        horizon=args.horizon,
        device=device,
        loss_cfg=loss_cfg,
        goal_tol=args.goal_tol,
        context_mode="true",
        z_gain=args.z_gain,
    )
    init_m["epoch"] = 0
    eval_hist.append(init_m)
    best_epoch = 0
    best_loss = float(init_m["loss"])
    best_coll = float(init_m["collision_rate"])
    best_success = float(init_m["success_rate"])
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": controller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_collision": best_coll,
            "best_success": best_success,
            "args": vars(args),
        },
        best_path,
    )
    print(
        f"Epoch 000/{args.epochs} | Val {init_m['loss']:.2f} | "
        f"Crash {init_m['collision_rate'] * 100:.1f}% | "
        f"Success {init_m['success_rate'] * 100:.1f}% | "
        f"SignAcc {(init_m['goal_sign_acc'] * 100.0 if not torch.isnan(torch.tensor(init_m['goal_sign_acc'])) else float('nan')):.1f}%"
    )

    for epoch in range(1, args.epochs + 1):
        controller.train()
        optimizer.zero_grad()
        tr_sc = scenario_to_device(_sample(args.batch, args.seed + epoch), device)
        noise = make_decaying_noise(args.batch, args.horizon, 4, noise_cfg, device)
        x_seq, _, _ = rollout_goal(
            controller=controller,
            plant_true=plant_true,
            scenario=tr_sc,
            horizon=args.horizon,
            device=device,
            noise=noise,
            z_gain=args.z_gain,
        )
        loss, train_parts = compute_loss(x_seq, tr_sc, loss_cfg)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        scheduler.step()
        train_hist.append({"epoch": epoch, "loss": float(loss.item())})

        if epoch % args.eval_every == 0 or epoch == 1:
            m, _, _ = evaluate(
                controller=controller,
                plant_true=plant_true,
                scenario=val_sc,
                horizon=args.horizon,
                device=device,
                loss_cfg=loss_cfg,
                goal_tol=args.goal_tol,
                context_mode="true",
                z_gain=args.z_gain,
            )
            m["epoch"] = epoch
            eval_hist.append(m)

            if args.best_ckpt_metric == "loss":
                better = m["loss"] < best_loss
            elif args.best_ckpt_metric == "success_then_loss":
                better = (m["success_rate"] > best_success + 1e-9) or (
                    abs(m["success_rate"] - best_success) <= 1e-9 and m["loss"] < best_loss
                )
            else:
                tol = float(args.best_ckpt_collision_tol)
                better = (m["collision_rate"] < best_coll - tol) or (
                    abs(m["collision_rate"] - best_coll) <= tol and m["loss"] < best_loss
                )
            if better:
                best_epoch = epoch
                best_loss = float(m["loss"])
                best_coll = float(m["collision_rate"])
                best_success = float(m["success_rate"])
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": controller.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "best_collision": best_coll,
                        "best_success": best_success,
                        "args": vars(args),
                    },
                    best_path,
                )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | Train {train_parts['loss_total']:.2f} | "
                f"Val {m['loss']:.2f} | Crash {m['collision_rate'] * 100:.1f}% | "
                f"Success {m['success_rate'] * 100:.1f}% | "
                f"SignAcc {(m['goal_sign_acc'] * 100.0 if not torch.isnan(torch.tensor(m['goal_sign_acc'])) else float('nan')):.1f}% | "
                f"LR {scheduler.get_last_lr()[0]:.1e}"
            )

    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device)
        controller.load_state_dict(ck["model_state_dict"])

    m_true, x_val, _ = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_sc,
        horizon=args.horizon,
        device=device,
        loss_cfg=loss_cfg,
        goal_tol=args.goal_tol,
        context_mode="true",
        z_gain=args.z_gain,
    )

    metrics = {
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_collision": float(best_coll),
        "best_success": float(best_success),
        "best_ckpt_metric": args.best_ckpt_metric,
        "final_true_context": m_true,
    }

    m_shuf = None
    m_zero = None
    if not args.skip_ablations:
        m_shuf, _, _ = evaluate(
            controller=controller,
            plant_true=plant_true,
            scenario=val_sc,
            horizon=args.horizon,
            device=device,
            loss_cfg=loss_cfg,
            goal_tol=args.goal_tol,
            context_mode="shuffled",
            z_gain=args.z_gain,
        )
        m_zero, _, _ = evaluate(
            controller=controller,
            plant_true=plant_true,
            scenario=val_sc,
            horizon=args.horizon,
            device=device,
            loss_cfg=loss_cfg,
            goal_tol=args.goal_tol,
            context_mode="zero",
            z_gain=args.z_gain,
        )
        metrics["final_shuffled_context"] = m_shuf
        metrics["final_zero_context"] = m_zero

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_hist, f, indent=2)
    with open(os.path.join(run_dir, "eval_history.json"), "w", encoding="utf-8") as f:
        json.dump(eval_hist, f, indent=2)

    plot_loss_curves(train_hist, eval_hist, run_dir, show_plots)
    plot_trajectory_fan(x_val, val_sc, run_dir, show_plots, fan_num=args.fan_num)
    plot_same_start_context_sweep(
        controller=controller,
        plant_true=plant_true,
        device=device,
        horizon=args.horizon,
        run_dir=run_dir,
        show_plots=show_plots,
        start_x=args.phase_start_x,
        start_y=args.phase_start_y,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        num_theta=max(8, args.fan_num),
        goal_x=args.goal_x,
        goal_y_scale=args.goal_y_scale,
        obs_center_x=args.obs_center_x,
        obs_center_y=args.obs_center_y,
        radius=0.5 * (args.rad_min + args.rad_max),
        z_gain=args.z_gain,
    )
    plot_phase_heatmaps(
        controller=controller,
        plant_true=plant_true,
        device=device,
        horizon=args.horizon,
        run_dir=run_dir,
        show_plots=show_plots,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        rad_min=args.rad_min,
        rad_max=args.rad_max,
        theta_res=args.phase_theta_res,
        rad_res=args.phase_rad_res,
        start_x=args.phase_start_x,
        start_y=args.phase_start_y,
        goal_x=args.goal_x,
        goal_y_scale=args.goal_y_scale,
        obs_center_x=args.obs_center_x,
        obs_center_y=args.obs_center_y,
        batch=args.phase_batch,
        z_gain=args.z_gain,
    )
    if (not args.skip_ablations) and (m_shuf is not None) and (m_zero is not None):
        plot_ablation(m_true, m_shuf, m_zero, run_dir, show_plots)

    torch.save(controller.state_dict(), os.path.join(run_dir, "pb_model_final.pt"))
    print(f"Done. Artifacts saved to {run_dir}")
    if show_plots:
        plt = get_plt(True)
        plt.show()


if __name__ == "__main__":
    main()
