"""
Load a trained robot_nav checkpoint and regenerate evaluation plots.

This script does not train. It:
  - Rebuilds the controller architecture used in robot_nav.py.
  - Loads best_model.pt or pb_model_final.pt from a run directory.
  - Recreates evaluation plots (including heatmap and radius comparison).
  - Optionally plots a custom trajectory from user-provided start/radii.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import torch

from bounded_mlp_operator import BoundedMLPOperator
from nav_env import NavScenario, min_dist_to_edge
from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_controller import PBController, FactorizedOperator
from robot_nav import (
    LossWeights,
    evaluate,
    get_plt,
    plot_loss_curves,
    plot_loss_heatmap_radius_levels,
    plot_m_outputs_over_time,
    plot_min_dist_hist,
    plot_radius_comparison_challenging,
    plot_trajectories,
    rollout_on_scenario,
    sample_dataset,
    scenario_to_device,
)
from ssm_operators import MpDeepSSM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize trained robot_nav checkpoint")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Path to runs/nav_experiment/robot_nav_*/ directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="auto",
        choices=["auto", "best", "final"],
        help="Which checkpoint file to load from run_dir",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for new plots (default: run_dir/replot_<timestamp>)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--val_batch", type=int, default=0, help="0 uses saved config value")
    parser.add_argument("--horizon", type=int, default=0, help="0 uses saved config value")
    parser.add_argument(
        "--plot_r_min",
        type=float,
        default=None,
        help="Override min obstacle radius for evaluation/heatmap scenario sampling",
    )
    parser.add_argument(
        "--plot_r_max",
        type=float,
        default=None,
        help="Override max obstacle radius for evaluation/heatmap scenario sampling",
    )
    parser.add_argument("--no_show_plots", action="store_true")
    parser.add_argument("--heatmap_res", type=int, default=140)
    parser.add_argument("--heatmap_batch", type=int, default=4096)
    parser.add_argument("--radius_cmp_num_starts", type=int, default=4)
    parser.add_argument("--radius_cmp_anchor_x", type=float, default=2.0)
    parser.add_argument("--radius_cmp_anchor_y", type=float, default=0.0)
    parser.add_argument("--radius_cmp_margin", type=float, default=0.05)

    # Custom trajectory plot options.
    parser.add_argument(
        "--custom_start",
        type=str,
        default="",
        help='Custom start as "x,y" (example: "2.2,-1.0")',
    )
    parser.add_argument(
        "--custom_radii",
        type=str,
        default="",
        help='Obstacle radii as "r1,r2,r3" or single "r" to broadcast',
    )
    parser.add_argument(
        "--custom_centers",
        type=str,
        default="",
        help='Obstacle centers as "x1,y1;x2,y2;x3,y3" (default: training centers)',
    )
    parser.add_argument("--custom_name", type=str, default="custom_trajectory")
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_robot_nav_run(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Runs base directory not found: {base_dir}")
    candidates = []
    for name in os.listdir(base_dir):
        if not name.startswith("robot_nav_"):
            continue
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        candidates.append(full)
    if not candidates:
        raise FileNotFoundError(f"No robot_nav_* run directories found in {base_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def resolve_run_dir(run_dir_arg: str) -> str:
    if run_dir_arg.strip():
        run_dir = os.path.abspath(run_dir_arg)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "runs", "nav_experiment")
    return _find_latest_robot_nav_run(base_dir)


def select_checkpoint_path(run_dir: str, checkpoint: str) -> str:
    best_path = os.path.join(run_dir, "best_model.pt")
    final_path = os.path.join(run_dir, "pb_model_final.pt")

    if checkpoint == "best":
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Missing checkpoint: {best_path}")
        return best_path
    if checkpoint == "final":
        if not os.path.exists(final_path):
            raise FileNotFoundError(f"Missing checkpoint: {final_path}")
        return final_path

    # auto
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(final_path):
        return final_path
    raise FileNotFoundError(f"No checkpoint found in {run_dir}. Expected best_model.pt or pb_model_final.pt.")


def load_histories(run_dir: str) -> tuple[List[dict], List[dict]]:
    train_hist = []
    eval_hist = []
    train_path = os.path.join(run_dir, "train_history.json")
    eval_path = os.path.join(run_dir, "eval_history.json")
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_hist = json.load(f)
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_hist = json.load(f)
    return train_hist, eval_hist


def build_controller(device: torch.device, k_obstacles: int) -> PBController:
    if k_obstacles != 3:
        raise ValueError("This visualizer currently expects k_obstacles=3 (same as robot_nav.py).")

    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=1.0, pre_kd=1.5)
    w_dim = 4
    u_dim = 2
    feat_dim = 16
    z_dim = 2 + 2 * k_obstacles + k_obstacles + k_obstacles

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


def load_controller_state(controller: PBController, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    controller.load_state_dict(state_dict)


def parse_vec2(text: str) -> torch.Tensor:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != 2:
        raise ValueError(f'Expected 2 values "x,y", got: {text!r}')
    return torch.tensor(vals, dtype=torch.float32)


def parse_centers(text: str, k: int) -> torch.Tensor:
    pairs = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
    if len(pairs) != k:
        raise ValueError(f"Expected {k} centers, got {len(pairs)} from {text!r}")
    ctrs = [parse_vec2(p) for p in pairs]
    return torch.stack(ctrs, dim=0)


def parse_radii(text: str, k: int) -> torch.Tensor:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) == 1:
        vals = vals * k
    if len(vals) != k:
        raise ValueError(f"Expected {k} radii (or one value to broadcast), got {len(vals)} from {text!r}")
    return torch.tensor(vals, dtype=torch.float32)


@torch.no_grad()
def plot_custom_trajectory(
    controller: PBController,
    plant_true: DoubleIntegratorTrue,
    *,
    start_xy: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    horizon: int,
    device: torch.device,
    out_dir: str,
    show_plots: bool,
    start_box: float,
    name: str,
):
    scenario = NavScenario(
        start=start_xy.view(1, 2).to(device),
        goal=torch.zeros(1, 2, device=device),
        centers=centers.view(1, centers.shape[0], 2).to(device),
        radii=radii.view(1, radii.shape[0]).to(device),
    )
    x_seq, u_seq, _ = rollout_on_scenario(
        controller=controller,
        plant_true=plant_true,
        scenario=scenario,
        horizon=horizon,
        device=device,
        noise=None,
    )
    min_edge = float(min_dist_to_edge(x_seq, scenario).min().item())
    terminal = float(torch.norm(x_seq[0, -1, :2]).item())

    plt = get_plt(show_plots)
    fig, ax = plt.subplots(figsize=(6, 6))
    traj = x_seq[0, :, :2].detach().cpu().numpy()
    ax.plot(traj[:, 0], traj[:, 1], color="C0", linewidth=2.0, label="trajectory")
    ax.scatter([float(start_xy[0])], [float(start_xy[1])], color="green", s=50, label="start")
    ax.scatter([0.0], [0.0], color="red", marker="*", s=120, label="goal")

    centers_np = centers.detach().cpu().numpy()
    radii_np = radii.detach().cpu().numpy()
    for k in range(centers_np.shape[0]):
        obs = plt.Circle((centers_np[k, 0], centers_np[k, 1]), radii_np[k], color="gray", alpha=0.35)
        ax.add_patch(obs)

    ax.set_xlim(-start_box, start_box)
    ax.set_ylim(-start_box, start_box)
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title(f"Custom trajectory | terminal={terminal:.3f}, min_edge={min_edge:.3f}")
    fig.tight_layout()

    fig_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(fig_path)
    if not show_plots:
        plt.close(fig)

    meta = {
        "start": [float(start_xy[0]), float(start_xy[1])],
        "centers": [[float(c[0]), float(c[1])] for c in centers.tolist()],
        "radii": [float(r) for r in radii.tolist()],
        "terminal_dist": terminal,
        "min_dist_to_edge": min_edge,
        "plot_file": os.path.basename(fig_path),
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)

    device = resolve_device(args.device)
    show_plots = not args.no_show_plots
    cfg = load_config(run_dir)
    ckpt_path = select_checkpoint_path(run_dir, args.checkpoint)

    out_dir = args.out_dir.strip()
    if not out_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(run_dir, f"replot_{ts}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    k = int(cfg.get("k_obstacles", 3))
    controller = build_controller(device=device, k_obstacles=k)
    load_controller_state(controller, ckpt_path=ckpt_path, device=device)
    controller.eval()

    plant_true = DoubleIntegratorTrue(dt=0.05, pre_kp=1.0, pre_kd=1.5)

    weights = LossWeights(
        term=float(cfg["w_term"]),
        stage=float(cfg["w_stage"]),
        coll_soft=float(cfg["w_coll_soft"]),
        coll_hard=float(cfg["w_coll_hard"]),
        corridor=float(cfg.get("w_corridor", 0.0)),
        control=float(cfg["w_control"]),
        coll_margin=float(cfg["coll_margin"]),
        coll_beta=float(cfg["coll_beta"]),
        corridor_gap_crit=float(cfg.get("corridor_gap_crit", 0.08)),
        corridor_margin=float(cfg.get("corridor_margin", 0.03)),
        corridor_beta=float(cfg.get("corridor_beta", 12.0)),
        corridor_gap_beta=float(cfg.get("corridor_gap_beta", 20.0)),
    )

    fixed_centers = torch.tensor([[1.0, 0.0], [0.3, 0.8], [0.3, -0.8]], dtype=torch.float32)
    if k != 3:
        raise ValueError("Expected k_obstacles=3 for fixed centers used by robot_nav.py.")

    val_batch = int(args.val_batch if args.val_batch > 0 else cfg["val_batch"])
    horizon = int(args.horizon if args.horizon > 0 else cfg["horizon"])
    start_box = float(cfg["start_box"])
    seed = int(cfg["seed"])
    r_min_plot = float(args.plot_r_min) if args.plot_r_min is not None else float(cfg["r_min_end"])
    r_max_plot = float(args.plot_r_max) if args.plot_r_max is not None else float(cfg["r_max_end"])
    if r_min_plot > r_max_plot:
        raise ValueError(f"plot_r_min must be <= plot_r_max, got {r_min_plot} > {r_max_plot}")

    val_scenario_cpu, _ = sample_dataset(
        batch_size=val_batch,
        seed=seed + 999,
        k=k,
        r_min=r_min_plot,
        r_max=r_max_plot,
        fixed_centers=fixed_centers,
        start_box=start_box,
        challenge_frac=float(cfg["challenge_frac"]),
        right_challenge_frac=float(cfg.get("right_challenge_frac_val", 0.35)),
        right_x_min=float(cfg.get("right_x_min", 0.5)),
        right_challenge_margin=float(cfg.get("right_challenge_margin", 0.05)),
        right_challenge_radius_quantile=float(cfg.get("right_challenge_radius_quantile", 0.9)),
        right_start_clearance=float(cfg.get("right_start_clearance", 0.05)),
    )
    val_scenario = scenario_to_device(val_scenario_cpu, device)

    metrics, x_val, u_val = evaluate(
        controller=controller,
        plant_true=plant_true,
        scenario=val_scenario,
        horizon=horizon,
        device=device,
        weights=weights,
        right_x_min=float(cfg.get("right_x_min", 0.5)),
        right_challenge_margin=float(cfg.get("right_challenge_margin", 0.05)),
        right_challenge_radius_quantile=float(cfg.get("right_challenge_radius_quantile", 0.9)),
    )

    train_hist, eval_hist = load_histories(run_dir)
    if train_hist or eval_hist:
        plot_loss_curves(train_hist, eval_hist, out_dir, show_plots=show_plots)
    plot_trajectories(x_val, val_scenario, out_dir, start_box=start_box, show_plots=show_plots, max_plots=4)
    plot_min_dist_hist(x_val, val_scenario, out_dir, show_plots=show_plots)
    plot_m_outputs_over_time(u_val, out_dir, show_plots=show_plots)
    plot_loss_heatmap_radius_levels(
        controller=controller,
        plant_true=plant_true,
        ref_scenario=val_scenario,
        horizon=horizon,
        device=device,
        weights=weights,
        run_dir=out_dir,
        show_plots=show_plots,
        start_box=start_box,
        heatmap_res=args.heatmap_res,
        heatmap_batch=args.heatmap_batch,
    )
    plot_radius_comparison_challenging(
        controller=controller,
        plant_true=plant_true,
        ref_scenario=val_scenario,
        horizon=horizon,
        device=device,
        run_dir=out_dir,
        show_plots=show_plots,
        start_box=start_box,
        num_starts=args.radius_cmp_num_starts,
        anchor_x=args.radius_cmp_anchor_x,
        anchor_y=args.radius_cmp_anchor_y,
        margin=args.radius_cmp_margin,
    )

    custom_plotted = False
    if args.custom_start.strip():
        custom_start = parse_vec2(args.custom_start)

        if args.custom_centers.strip():
            custom_centers = parse_centers(args.custom_centers, k=k)
        else:
            custom_centers = fixed_centers.clone()

        if args.custom_radii.strip():
            custom_radii = parse_radii(args.custom_radii, k=k)
        else:
            # Default to one radii level seen in validation set.
            custom_radii = val_scenario_cpu.radii[0].clone().cpu()

        plot_custom_trajectory(
            controller=controller,
            plant_true=plant_true,
            start_xy=custom_start,
            centers=custom_centers,
            radii=custom_radii,
            horizon=horizon,
            device=device,
            out_dir=out_dir,
            show_plots=show_plots,
            start_box=start_box,
            name=args.custom_name,
        )
        custom_plotted = True

    with open(os.path.join(out_dir, "loaded_eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "run_dir": run_dir,
                "device": str(device),
                "val_batch": val_batch,
                "horizon": horizon,
                "plot_r_min": r_min_plot,
                "plot_r_max": r_max_plot,
                "metrics": metrics,
                "custom_plot_generated": custom_plotted,
            },
            f,
            indent=2,
        )

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Saved plots to: {out_dir}")
    if custom_plotted:
        print(f"Saved custom trajectory plot: {os.path.join(out_dir, args.custom_name + '.png')}")

    if show_plots:
        plt = get_plt(True)
        plt.show()


if __name__ == "__main__":
    main()
