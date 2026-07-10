"""Contextual PB experiment: causal payload telemetry under abrupt regime changes."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from payload_artifacts import render_all
from payload_core import (CONTEXT_FEATURE_META, CONTEXT_FEATURE_ORDER, FAIR_CONTEXT_DEFAULT, PayloadBatch,
                          build_controller, context_dim, rollout_variant, sample_batch, variant_specs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Contextual PB payload-regime switching experiment")
    p.add_argument("--seed", type=int, default=41); p.add_argument("--device", default="cpu")
    p.add_argument("--require_cuda", action="store_true",
                   help="Abort if CUDA is unavailable instead of silently falling back to CPU "
                        "(used by RCP submissions so a broken image/driver fails fast).")
    p.add_argument("--run_id", default=""); p.add_argument("--plot_only", default="")
    p.add_argument("--no_show_plots", action="store_true"); p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--train_batch", type=int, default=512); p.add_argument("--val_batch", type=int, default=512); p.add_argument("--test_batch", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--disturbance_only_epochs", type=int, default=300); p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--lr_min", type=float, default=2e-4); p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--variants", default="nominal,disturbance_only,context,mad_context,contextual_ssm")
    p.add_argument("--horizon", type=int, default=150); p.add_argument("--dt", type=float, default=.05)
    p.add_argument("--start_x_min", type=float, default=1.85); p.add_argument("--start_x_max", type=float, default=2.15); p.add_argument("--start_y_max", type=float, default=.16)
    p.add_argument("--pre_kp", type=float, default=.32); p.add_argument("--pre_kd", type=float, default=.80); p.add_argument("--corridor_limit", type=float, default=.78)
    p.add_argument("--goal_tol", type=float, default=.13); p.add_argument("--terminal_speed_tol", type=float, default=.14); p.add_argument("--control_limit", type=float, default=1.45)
    p.add_argument("--regime_protocol", choices=["fixed", "single_switch"], default="single_switch")
    p.add_argument("--payload_light_mass", type=float, default=.72); p.add_argument("--payload_loaded_mass", type=float, default=1.55); p.add_argument("--payload_mass_ref", type=float, default=1.0)
    p.add_argument("--payload_light_actuator_gain", type=float, default=1.12); p.add_argument("--payload_loaded_actuator_gain", type=float, default=.76)
    p.add_argument("--payload_light_drag", type=float, default=.01); p.add_argument("--payload_loaded_drag", type=float, default=.13); p.add_argument("--payload_lateral_bias", type=float, default=.18)
    p.add_argument("--payload_switch_min", type=int, default=42); p.add_argument("--payload_switch_max", type=int, default=76); p.add_argument("--payload_switch_probability", type=float, default=1.0)
    p.add_argument("--payload_bias_settle_steps", type=int, default=45,
                   help="Lateral-bias settle time: the centre-of-mass bias fades linearly to 0 "
                        "over this many steps after each load onset (episode start / switch), so "
                        "the reconstructed w can decay once the robot docks. 0 = persistent bias (legacy).")
    p.add_argument("--noise_decay", choices=["none", "taper", "linear", "exponential"], default="taper",
                   help="Decay window on the process noise over the horizon (noise part of w -> 0 "
                        "by t=T). 'none' = stationary (legacy).")
    p.add_argument("--noise_decay_ramp", type=int, default=0, help="taper only: trailing roll-off steps (0 = horizon//2).")
    p.add_argument("--noise_decay_rate", type=float, default=.98, help="exponential only: per-step decay factor.")
    p.add_argument("--test_payload_light_mass", type=float, default=None); p.add_argument("--test_payload_loaded_mass", type=float, default=1.78)
    p.add_argument("--test_switch_min", type=int, default=22); p.add_argument("--test_switch_max", type=int, default=40)
    p.add_argument("--payload_context_delay", type=int, default=0); p.add_argument("--payload_obs_noise_sigma", type=float, default=.012); p.add_argument("--payload_context_dropout_p", type=float, default=.04)
    p.add_argument("--context_features", default=",".join(FAIR_CONTEXT_DEFAULT))
    p.add_argument("--intervention_delay_steps", type=int, default=6); p.add_argument("--no_intervention_eval", dest="intervention_eval", action="store_false"); p.set_defaults(intervention_eval=True)
    p.add_argument("--noise_vel_sigma", type=float, default=1.2e-3); p.add_argument("--noise_pos_multiplier", type=float, default=.25)
    p.add_argument("--gust_count_min", type=int, default=1); p.add_argument("--gust_count_max", type=int, default=3); p.add_argument("--gust_duration", type=int, default=7); p.add_argument("--gust_velocity", type=float, default=.014)
    p.add_argument("--feat_dim", type=int, default=16); p.add_argument("--mb_hidden", type=int, default=56); p.add_argument("--mb_layers", type=int, default=3); p.add_argument("--mb_bound", type=float, default=1.0); p.add_argument("--z_scale", type=float, default=4.0); p.add_argument("--z_residual_gain", type=float, default=.35)
    p.add_argument("--ssm_param", choices=["lru", "tv", "tvc"], default="tv"); p.add_argument("--ssm_layers", type=int, default=3); p.add_argument("--ssm_d_model", type=int, default=32); p.add_argument("--ssm_d_state", type=int, default=64); p.add_argument("--ssm_ff", default="GLU")
    p.add_argument("--ssm_bcd_nonlinearity", choices=["tanh", "identity"], default="tanh",
                   help="tvc only: nonlinearity bounding b,c,d ('tanh' trains more stably).")
    # ContextualDeepSSM variant (contextual_ssm): a single context-native operator.
    p.add_argument("--ctx_modes", default="mixer,input,gate", help="Context ports: any of mixer,input,gate,select.")
    p.add_argument("--ctx_select", action="store_true", help="Add the 'select' port (context inside the selective SSM matrices; needs --ssm_param tv/tvc).")
    p.add_argument("--ctx_filter", choices=["auto", "finite_horizon", "taper", "exponential", "polynomial", "difference", "none"], default="taper",
                   help="L2 projection for the 'input' port (time-windowed filters are correct in the T=1 closed loop).")
    p.add_argument("--ctx_filter_ramp", type=int, default=20, help="taper: cosine roll-off steps (0 = full horizon).")
    p.add_argument("--ctx_mixer_bound", type=float, default=4.0, help="Spectral-norm bound on the mixer matrix A_t.")
    p.add_argument("--ctx_d_features", type=int, default=16, help="Core feature width fed to the mixer.")
    p.add_argument("--ctx_gamma", type=float, default=0.0, help="Prescribed L2 gain cap for the contextual core (0 = no cap).")
    p.add_argument("--ctx_z_scale", type=float, default=1.0,
                   help="Target per-feature context scale for ContextualDeepSSM: context is rescaled by "
                        "ctx_z_scale/z_scale before all ports (1.0 = unit range; set to --z_scale for the raw feed).")
    p.add_argument("--mp_context_lift_dim", type=int, default=10); p.add_argument("--mp_context_lift_type", choices=["identity", "linear", "mlp"], default="linear"); p.add_argument("--mp_context_hidden_dim", type=int, default=28)
    p.add_argument("--mp_context_decay_law", choices=["exp", "poly", "finite"], default="finite"); p.add_argument("--mp_context_decay_rate", type=float, default=.04); p.add_argument("--mp_context_decay_power", type=float, default=.73); p.add_argument("--mp_context_decay_horizon", type=int, default=135); p.add_argument("--mp_context_lp_p", type=float, default=2.0); p.add_argument("--mp_context_scale", type=float, default=.25)
    p.add_argument("--use_w_augment", action="store_true"); p.add_argument("--w_augment_decay", type=float, default=.97); p.add_argument("--use_w0_clip", action="store_true"); p.add_argument("--w0_clip", type=float, default=.15)
    p.add_argument("--goal_stage_weight", type=float, default=6.0); p.add_argument("--goal_terminal_weight", type=float, default=72.0); p.add_argument("--terminal_vel_weight", type=float, default=7.0); p.add_argument("--control_weight", type=float, default=2.0); p.add_argument("--control_slew_weight", type=float, default=.5); p.add_argument("--corridor_weight", type=float, default=14.0)
    p.add_argument("--post_switch_track_weight", type=float, default=5.0); p.add_argument("--post_switch_window", type=int, default=28); p.add_argument("--recovery_radius", type=float, default=.28); p.add_argument("--sample_traj_count", type=int, default=4)
    return p


def parse_args(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if not (0 <= args.payload_switch_probability <= 1): raise ValueError("payload_switch_probability must be in [0, 1].")
    if min(args.train_batch, args.val_batch, args.test_batch) < 2 or any(x % 2 for x in (args.train_batch, args.val_batch, args.test_batch)): raise ValueError("All paired batch sizes must be even and >= 2.")
    if min(args.epochs, args.disturbance_only_epochs, args.eval_every) < 1: raise ValueError("epochs, disturbance_only_epochs, and eval_every must be positive.")
    if args.horizon < 12 or args.post_switch_window < 1 or args.control_limit <= 0: raise ValueError("horizon, post_switch_window, and control_limit must be positive (horizon >= 12).")
    return args


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _f(value) -> float: return float(value.detach().item() if torch.is_tensor(value) else value)


def _loss(args, batch: PayloadBatch, rollout) -> tuple[torch.Tensor, dict]:
    device, xy, vel = rollout.x_seq.device, rollout.x_seq[..., :2], rollout.x_seq[..., 2:]
    distance = torch.linalg.vector_norm(xy - batch.goal.to(device).unsqueeze(1), dim=-1); speed = torch.linalg.vector_norm(vel, dim=-1)
    stage = distance[:, -min(34, distance.shape[1]):].square().mean(); terminal = distance[:, -1].square().mean(); terminal_vel = speed[:, -1].square().mean()
    control = rollout.u_seq.square().sum(-1).mean(); slew = (rollout.u_seq[:, 1:] - rollout.u_seq[:, :-1]).square().sum(-1).mean()
    corridor = F.softplus(xy[..., 1].abs() - float(args.corridor_limit)).square().mean()
    time = torch.arange(distance.shape[1], device=device).unsqueeze(0); switch = batch.switch_step.to(device).unsqueeze(1)
    mask = (switch >= 0) & (time >= switch) & (time < switch + int(args.post_switch_window)); mask_f = mask.float(); post = (distance * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    total = float(args.goal_stage_weight) * stage + float(args.goal_terminal_weight) * terminal + float(args.terminal_vel_weight) * terminal_vel + float(args.control_weight) * control + float(args.control_slew_weight) * slew + float(args.corridor_weight) * corridor + float(args.post_switch_track_weight) * post
    return total, {"loss_goal_stage": _f(stage), "loss_goal_terminal": _f(terminal), "loss_terminal_vel": _f(terminal_vel), "loss_control": _f(control), "loss_control_slew": _f(slew), "loss_corridor": _f(corridor), "loss_post_switch": _f(post)}


def evaluate(args, batch: PayloadBatch, device: torch.device, mode: str, controller=None, plant=None, intervention: str = "truth") -> dict:
    with torch.no_grad():
        rollout = rollout_variant(args, batch, device, mode=mode, controller=controller, plant=plant, intervention=intervention)
        cost, parts = _loss(args, batch, rollout); xy, vel = rollout.x_seq[..., :2], rollout.x_seq[..., 2:]; distance = torch.linalg.vector_norm(xy - batch.goal.to(rollout.x_seq.device).unsqueeze(1), dim=-1); speed = torch.linalg.vector_norm(vel, dim=-1)
        success = (distance[:, -1] < float(args.goal_tol)) & (speed[:, -1] < float(args.terminal_speed_tol)) & (xy[..., 1].abs().amax(1) < float(args.corridor_limit))
        time = torch.arange(distance.shape[1], device=device).unsqueeze(0); sw = batch.switch_step.to(device).unsqueeze(1); mask = (sw >= 0) & (time >= sw) & (time < sw + int(args.post_switch_window)); post = (distance * mask).sum(1) / mask.sum(1).clamp_min(1)
        recovery = torch.full_like(post, float(args.post_switch_window));
        for i, step in enumerate(batch.switch_step.tolist()):
            if step >= 0:
                hit = torch.nonzero(distance[i, step:min(step + int(args.post_switch_window), distance.shape[1])] < float(args.recovery_radius), as_tuple=False)
                if len(hit): recovery[i] = float(hit[0, 0])
        saturation = (rollout.u_seq.abs() >= float(args.control_limit) - 1e-5).any(-1).float().mean()
        out = {"avg_cost": _f(cost), "success_rate": _f(success.float().mean()), "goal_success_rate": _f((distance[:, -1] < float(args.goal_tol)).float().mean()), "settled_success_rate": _f((speed[:, -1] < float(args.terminal_speed_tol)).float().mean()), "avg_terminal_dist": _f(distance[:, -1].mean()), "avg_terminal_speed": _f(speed[:, -1].mean()), "avg_post_switch_error": _f(post.mean()), "avg_recovery_steps": _f(recovery.mean()), "avg_control_energy": _f(rollout.u_seq.square().sum(-1).mean()), "avg_control_slew": _f((rollout.u_seq[:, 1:] - rollout.u_seq[:, :-1]).square().sum(-1).mean()), "saturation_rate": _f(saturation), "avg_abs_reconstructed_w": _f(rollout.w_seq.abs().mean())}
        out.update(parts); out["rollout"] = {"x_seq": rollout.x_seq.cpu(), "u_seq": rollout.u_seq.cpu(), "w_seq": rollout.w_seq.cpu(), "distance": distance.cpu(), "success": success.cpu(), "post_switch_error": post.cpu()}; return out


def _train(args, device: torch.device, mode: str, val: PayloadBatch):
    if mode == "nominal": return None, None, [], evaluate(args, val, device, mode)
    controller, plant = build_controller(device, args, mad=(mode == "mad_context"), contextual=(mode == "contextual_ssm")); optimizer = torch.optim.AdamW(controller.parameters(), lr=float(args.lr), weight_decay=1e-4); epochs = int(args.disturbance_only_epochs if mode == "disturbance_only" else args.epochs); scheduler = CosineAnnealingLR(optimizer, max(1, epochs), eta_min=float(args.lr_min))
    history, best, best_state, best_score = [], None, None, None
    for epoch in range(1, epochs + 1):
        controller.train(); train = sample_batch(args, batch_size=int(args.train_batch), seed=int(args.seed) + 1000 + epoch, paired=True, shuffle=True); rollout = rollout_variant(args, train, device, mode=mode, controller=controller, plant=plant, training=True); loss, parts = _loss(args, train, rollout)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(controller.parameters(), float(args.grad_clip)); optimizer.step(); scheduler.step()
        rec = {"epoch": epoch, "train_loss": _f(loss), "lr": float(scheduler.get_last_lr()[0]), **parts}
        if epoch % int(args.eval_every) == 0 or epoch == epochs:
            controller.eval(); metrics = evaluate(args, val, device, mode, controller, plant); rec.update(val_cost=metrics["avg_cost"], val_success_rate=metrics["success_rate"]); score = (-metrics["success_rate"], metrics["avg_cost"])
            if best_score is None or score < best_score: best, best_score, best_state = metrics, score, {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
            print(f"[{mode}] {epoch:03d}/{epochs}: loss={rec['train_loss']:.3f} val_success={metrics['success_rate']:.3f} post={metrics['avg_post_switch_error']:.3f}")
        history.append(rec)
    controller.load_state_dict(best_state); return controller, plant, history, best


def _strip(metrics: dict) -> dict: return {k: v for k, v in metrics.items() if k != "rollout"}


def _save(path: Path, value) -> None: path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _plot_only(args, device: torch.device, run_dir: Path) -> None:
    saved = json.loads((run_dir / "config.json").read_text()); saved.pop("plot_only", None)
    # Saved config fills in run parameters, but flags given on the CLI win
    # (so e.g. --sample_traj_count can be tweaked at re-plot time).
    cli_flags = {token.split("=")[0] for token in sys.argv[1:] if token.startswith("--")}
    merged = vars(args).copy()
    merged.update({k: v for k, v in saved.items() if f"--{k}" not in cli_flags})
    args = argparse.Namespace(**merged); specs = variant_specs(args); batch = sample_batch(args, batch_size=int(args.test_batch), seed=int(args.seed) + 60000, paired=True, shuffle=False)
    controllers, plants, metrics = {}, {}, {}
    for mode, _ in specs:
        if mode == "nominal": controllers[mode] = plants[mode] = None
        else:
            ctrl, plant = build_controller(device, args, mad=(mode == "mad_context"), contextual=(mode == "contextual_ssm"))
            weights = run_dir / f"{mode}_controller.pt"
            if weights.exists(): ctrl.load_state_dict(torch.load(weights, map_location=device))
            else: print(f"[plot_only] Warning: {weights.name} not found — using random weights for {mode}.")
            ctrl.eval(); controllers[mode], plants[mode] = ctrl, plant
        metrics[mode] = evaluate(args, batch, device, mode, controllers[mode], plants[mode])
    interventions = {}
    if args.intervention_eval:
        ood = sample_batch(args, batch_size=int(args.test_batch), seed=int(args.seed) + 70000, paired=True, shuffle=False, test=True)
        for mode in ("context", "mad_context", "contextual_ssm"):
            if mode in controllers:
                interventions[mode] = {"truth": metrics[mode], "delayed telemetry": evaluate(args, batch, device, mode, controllers[mode], plants[mode], "delayed"), "wrong telemetry": evaluate(args, batch, device, mode, controllers[mode], plants[mode], "wrong"), "missing telemetry": evaluate(args, batch, device, mode, controllers[mode], plants[mode], "dropout"), "OOD payload + timing": evaluate(args, ood, device, mode, controllers[mode], plants[mode])}
    histories = json.loads((run_dir / "train_history.json").read_text()) if (run_dir / "train_history.json").exists() else {}; render_all(args, run_dir, batch, specs, metrics, histories, interventions, not args.no_show_plots); _save(run_dir / "metrics.json", {k: _strip(v) for k, v in metrics.items()}); _save(run_dir / "interventions.json", {m: {k: _strip(v) for k, v in values.items()} for m, values in interventions.items()})


def main() -> None:
    args = parse_args(); _seed(int(args.seed))
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require_cuda was set but CUDA is not available on this node/image.")
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    if args.plot_only:
        _plot_only(args, device, Path(__file__).resolve().parent / "runs" / args.plot_only); return
    run_id = args.run_id or f"payload_regimes_{datetime.now():%Y%m%d_%H%M%S}"; run_dir = Path(__file__).resolve().parent / "runs" / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args)); config["context_dim"] = context_dim(args); config["experiment_type"] = "payload_regimes"; _save(run_dir / "config.json", config)
    val = sample_batch(args, batch_size=int(args.val_batch), seed=int(args.seed) + 50000, paired=True, shuffle=False); test = sample_batch(args, batch_size=int(args.test_batch), seed=int(args.seed) + 60000, paired=True, shuffle=False); ood = sample_batch(args, batch_size=int(args.test_batch), seed=int(args.seed) + 70000, paired=True, shuffle=False, test=True)
    specs, controllers, plants, histories, val_metrics, metrics = variant_specs(args), {}, {}, {}, {}, {}
    print(f"Payload-regime run: {run_dir}  |  context dim={context_dim(args)}  |  device={device}")
    for mode, label in specs:
        print(f"Training {label}..."); controller, plant, history, val_result = _train(args, device, mode, val); controllers[mode], plants[mode], histories[mode], val_metrics[mode] = controller, plant, history, val_result; metrics[mode] = evaluate(args, test, device, mode, controller, plant)
        if controller is not None: torch.save(controller.state_dict(), run_dir / f"{mode}_controller.pt")
        _save(run_dir / "metrics.json", {k: _strip(v) for k, v in metrics.items()}); _save(run_dir / "train_history.json", histories)
    interventions = {}
    if args.intervention_eval:
        for mode in ("context", "mad_context", "contextual_ssm"):
            if mode in controllers:
                interventions[mode] = {"truth": metrics[mode], "delayed telemetry": evaluate(args, test, device, mode, controllers[mode], plants[mode], "delayed"), "wrong telemetry": evaluate(args, test, device, mode, controllers[mode], plants[mode], "wrong"), "missing telemetry": evaluate(args, test, device, mode, controllers[mode], plants[mode], "dropout"), "OOD payload + timing": evaluate(args, ood, device, mode, controllers[mode], plants[mode])}
    _save(run_dir / "val_metrics.json", {k: _strip(v) for k, v in val_metrics.items()}); _save(run_dir / "interventions.json", {m: {k: _strip(v) for k, v in values.items()} for m, values in interventions.items()})
    if not args.skip_plots: render_all(args, run_dir, test, specs, metrics, histories, interventions, not args.no_show_plots)
    words = [f"{label}: {metrics[mode]['success_rate']:.3f}" for mode, label in specs]; (run_dir / "interpretation.txt").write_text(" | ".join(words) + "\nPayload telemetry is causal; delayed/wrong/missing telemetry are sensor-integrity interventions, not fairness baselines.\n", encoding="utf-8")
    print("RESULTS  " + " | ".join(words))


if __name__ == "__main__": main()
