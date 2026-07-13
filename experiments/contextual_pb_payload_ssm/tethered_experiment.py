"""Training, evaluation, replay, and orchestration for Tethered Cargo Slalom."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from payload_core import context_dim, variant_specs
from tethered_payload import (SlalomBatch, SlalomRollout, build_slalom_controller,
                              gate_geometry, rollout_slalom, sample_slalom_batch)


def _float(value) -> float:
    return float(value.detach().item() if torch.is_tensor(value) else value)


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _strip(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "rollout"}


def save_batch(path: Path, batch: SlalomBatch) -> None:
    torch.save({field.name: getattr(batch, field.name) for field in fields(batch)}, path)


def load_batch(path: Path) -> SlalomBatch:
    payload = torch.load(path, map_location="cpu")
    return SlalomBatch(**payload)


def segment_points(args, carrier: torch.Tensor, payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(1, int(args.slalom_tether_samples)) + 2
    alpha = torch.linspace(0.0, 1.0, count, device=carrier.device, dtype=carrier.dtype)
    points = carrier.unsqueeze(-2) * (1.0 - alpha.view(1, 1, -1, 1))
    points = points + payload.unsqueeze(-2) * alpha.view(1, 1, -1, 1)
    radii = torch.full((count,), float(args.slalom_tether_radius), device=carrier.device)
    radii[0], radii[-1] = float(args.slalom_carrier_radius), float(args.slalom_payload_radius)
    return points, radii


def soft_gate_loss(args, batch: SlalomBatch, rollout: SlalomRollout, half_width: float) -> torch.Tensor:
    points, radii = segment_points(args, rollout.x_seq[..., :2], rollout.payload_pos_seq)
    gate_xs = torch.tensor(gate_geometry(args)[0], device=points.device, dtype=points.dtype)
    centers = batch.gate_centers.to(points.device)
    dx = points[..., 0].unsqueeze(-1) - gate_xs.view(1, 1, 1, -1)
    dy = points[..., 1].unsqueeze(-1) - centers[:, None, None, :]
    allowed = half_width - float(args.slalom_collision_margin) - radii.view(1, 1, -1, 1)
    sharp = float(args.slalom_collision_sharpness)
    violation = F.softplus(sharp * (dy.abs() - allowed)) / sharp
    focus = torch.exp(-0.5 * (dx / float(args.slalom_gate_focus_sigma)) ** 2)
    per_body_gate = (focus * violation.square()).sum(1) / focus.sum(1).clamp_min(1e-5)
    # Joint success is an all-bodies/all-gates event. A pure mean lets one bad
    # cargo crossing disappear among 21 safe carrier/tether crossings, so train
    # against both average exposure and each episode's worst crossing.
    worst_crossing = per_body_gate.flatten(1).amax(1).mean()
    return per_body_gate.mean() + 3.0 * worst_crossing


def hard_gate_metrics(args, batch: SlalomBatch, rollout: SlalomRollout) -> dict[str, torch.Tensor]:
    carrier = torch.cat([batch.start.to(rollout.x_seq.device).unsqueeze(1), rollout.x_seq[..., :2]], 1)
    payload = torch.cat([batch.payload_start.to(rollout.x_seq.device).unsqueeze(1), rollout.payload_pos_seq], 1)
    points, radii = segment_points(args, carrier, payload)
    gate_xs = gate_geometry(args)[0]
    centers = batch.gate_centers.to(points.device)
    batch_size, _, sample_count, _ = points.shape
    clearance = torch.empty(batch_size, len(gate_xs), sample_count, device=points.device)
    for gate_index, gate_x in enumerate(gate_xs):
        for sample_index in range(sample_count):
            x0 = points[:, :-1, sample_index, 0]
            x1 = points[:, 1:, sample_index, 0]
            crossed = (x0 > float(gate_x)) & (x1 <= float(gate_x))
            has_crossed = crossed.any(1)
            first = crossed.float().argmax(1)
            gather = first.unsqueeze(-1)
            x_before = x0.gather(1, gather).squeeze(1)
            x_after = x1.gather(1, gather).squeeze(1)
            y_before = points[:, :-1, sample_index, 1].gather(1, gather).squeeze(1)
            y_after = points[:, 1:, sample_index, 1].gather(1, gather).squeeze(1)
            fraction = ((float(gate_x) - x_before) / (x_after - x_before).clamp_max(-1e-7)).clamp(0.0, 1.0)
            y_cross = y_before + fraction * (y_after - y_before)
            margin = (float(args.slalom_gate_half_width)
                      - float(args.slalom_collision_margin) - radii[sample_index]
                      - (y_cross - centers[:, gate_index]).abs())
            clearance[:, gate_index, sample_index] = torch.where(
                has_crossed, margin, torch.full_like(margin, -float(args.corridor_limit)))
    safe = clearance >= 0.0
    carrier_safe = safe[..., 0].all(1)
    payload_safe = safe[..., -1].all(1)
    tether_safe = safe[..., 1:-1].all((1, 2)) if sample_count > 2 else torch.ones_like(carrier_safe)
    gate_safe = safe.all((1, 2))
    return {
        "clearance": clearance, "safe": safe, "carrier_safe": carrier_safe,
        "payload_safe": payload_safe, "tether_safe": tether_safe,
        "gate_safe": gate_safe,
    }


def loss(args, batch: SlalomBatch, rollout: SlalomRollout, curriculum: float = 1.0) -> tuple[torch.Tensor, dict]:
    carrier_pos, carrier_vel = rollout.x_seq[..., :2], rollout.x_seq[..., 2:]
    payload_pos, payload_vel = rollout.payload_pos_seq, rollout.payload_vel_seq
    carrier_dist = torch.linalg.vector_norm(carrier_pos - batch.goal.to(carrier_pos.device).unsqueeze(1), dim=-1)
    relative = payload_pos - carrier_pos
    relative_vel = payload_vel - carrier_vel
    target_relative = torch.stack([batch.tether_length.to(carrier_pos.device),
                                   torch.zeros_like(batch.tether_length, device=carrier_pos.device)], -1)
    settle_relative = (relative[:, -1] - target_relative).square().mean()
    settle_speed = relative_vel[:, -1].square().mean() + carrier_vel[:, -1].square().mean()
    terminal = carrier_dist[:, -1].square().mean()
    stage = carrier_dist[:, -min(35, carrier_dist.shape[1]):].square().mean()
    widened = float(args.slalom_gate_half_width) + float(args.slalom_curriculum_extra_width) * (1.0 - curriculum)
    gate = soft_gate_loss(args, batch, rollout, widened)
    corridor = (F.softplus(carrier_pos[..., 1].abs() - float(args.corridor_limit)).square().mean()
                + F.softplus(payload_pos[..., 1].abs() - float(args.corridor_limit)).square().mean())
    control = rollout.u_seq.square().sum(-1).mean()
    slew = (rollout.u_seq[:, 1:] - rollout.u_seq[:, :-1]).square().sum(-1).mean()
    tension = F.softplus(rollout.tension_seq - float(args.slalom_max_tension)).square().mean()
    extension = torch.linalg.vector_norm(relative, dim=-1) - batch.tether_length.to(relative.device).unsqueeze(1)
    stretch = F.softplus(extension - float(args.slalom_max_extension)).square().mean()
    total = (float(args.goal_stage_weight) * stage
             + float(args.goal_terminal_weight) * terminal
             + float(args.terminal_vel_weight) * settle_speed
             + float(args.slalom_payload_settle_weight) * settle_relative
             + float(args.slalom_gate_weight) * gate
             + float(args.corridor_weight) * corridor
             + float(args.control_weight) * control
             + float(args.control_slew_weight) * slew
             + float(args.slalom_tension_weight) * tension
             + float(args.slalom_stretch_weight) * stretch)
    return total, {
        "loss_goal_stage": _float(stage), "loss_goal_terminal": _float(terminal),
        "loss_settle_speed": _float(settle_speed), "loss_payload_settle": _float(settle_relative),
        "loss_gate": _float(gate), "loss_corridor": _float(corridor),
        "loss_control": _float(control), "loss_control_slew": _float(slew),
        "loss_tension": _float(tension), "loss_stretch": _float(stretch),
    }


def evaluate(args, batch: SlalomBatch, device: torch.device, mode: str,
             controller=None, plant=None, intervention: str = "truth") -> dict:
    with torch.no_grad():
        rollout = rollout_slalom(args, batch, device, mode=mode, controller=controller,
                                 plant=plant, intervention=intervention)
        cost, parts = loss(args, batch, rollout)
        hard = hard_gate_metrics(args, batch, rollout)
        carrier_pos, carrier_vel = rollout.x_seq[..., :2], rollout.x_seq[..., 2:]
        relative = rollout.payload_pos_seq - carrier_pos
        relative_vel = rollout.payload_vel_seq - carrier_vel
        goal_dist = torch.linalg.vector_norm(carrier_pos[:, -1], dim=-1)
        carrier_speed = torch.linalg.vector_norm(carrier_vel[:, -1], dim=-1)
        payload_speed = torch.linalg.vector_norm(rollout.payload_vel_seq[:, -1], dim=-1)
        lateral_swing = relative[:, -1, 1].abs()
        extension = (torch.linalg.vector_norm(relative[:, -1], dim=-1)
                     - batch.tether_length.to(device)).abs()
        goal_ok = goal_dist < float(args.goal_tol)
        settled = ((carrier_speed < float(args.terminal_speed_tol))
                   & (payload_speed < float(args.slalom_payload_speed_tol))
                   & (lateral_swing < float(args.slalom_settle_swing_tol))
                   & (extension < float(args.slalom_settle_extension_tol)))
        success = hard["gate_safe"] & goal_ok & settled
        clearance = hard["clearance"]
        out = {
            "avg_cost": _float(cost), "success_rate": _float(success.float().mean()),
            "gate_success_rate": _float(hard["gate_safe"].float().mean()),
            "carrier_gate_success_rate": _float(hard["carrier_safe"].float().mean()),
            "payload_gate_success_rate": _float(hard["payload_safe"].float().mean()),
            "tether_gate_success_rate": _float(hard["tether_safe"].float().mean()),
            "goal_success_rate": _float(goal_ok.float().mean()),
            "settled_success_rate": _float(settled.float().mean()),
            "payload_collision_rate": _float((~hard["payload_safe"]).float().mean()),
            "avg_min_clearance": _float(clearance.amin((1, 2)).mean()),
            "avg_min_payload_clearance": _float(clearance[..., -1].amin(1).mean()),
            "avg_terminal_dist": _float(goal_dist.mean()),
            "avg_terminal_swing": _float(lateral_swing.mean()),
            "avg_control_energy": _float(rollout.u_seq.square().sum(-1).mean()),
            "avg_max_tension": _float(rollout.tension_seq.amax(1).mean()),
            "avg_abs_reconstructed_w": _float(rollout.w_seq.abs().mean()),
        }
        if controller is not None:
            expected_w = torch.zeros_like(rollout.w_seq)
            expected_w[:, 1:] = batch.process_noise[:, :-1].to(device)
            out["max_w_noise_identity_error"] = _float(
                (rollout.w_seq - expected_w).abs().amax())
        out.update(parts)
        out["rollout"] = {
            "x_seq": rollout.x_seq.cpu(), "u_seq": rollout.u_seq.cpu(),
            "w_seq": rollout.w_seq.cpu(), "payload_pos_seq": rollout.payload_pos_seq.cpu(),
            "payload_vel_seq": rollout.payload_vel_seq.cpu(),
            "tension_seq": rollout.tension_seq.cpu(), "gate_clearance": clearance.cpu(),
            "gate_safe": hard["safe"].cpu(), "success": success.cpu(),
        }
        return out


def train(args, device: torch.device, mode: str, validation: SlalomBatch):
    if mode == "nominal":
        result = evaluate(args, validation, device, mode)
        return None, None, [], result
    controller, plant = build_slalom_controller(device, args, mode)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=float(args.lr), weight_decay=1e-4)
    epochs = int(args.disturbance_only_epochs if mode == "disturbance_only" else args.epochs)
    scheduler = CosineAnnealingLR(optimizer, max(1, epochs), eta_min=float(args.lr_min))
    history, best_metrics, best_state, best_score = [], None, None, None
    for epoch in range(1, epochs + 1):
        controller.train()
        batch = sample_slalom_batch(args, batch_size=int(args.train_batch),
                                     seed=int(args.seed) + 1000 + epoch,
                                     paired=True, shuffle=True)
        rollout = rollout_slalom(args, batch, device, mode=mode, controller=controller,
                                 plant=plant, training=True)
        curriculum = min(1.0, epoch / max(1.0, float(args.slalom_curriculum_fraction) * epochs))
        objective, parts = loss(args, batch, rollout, curriculum)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), float(args.grad_clip))
        optimizer.step(); scheduler.step()
        record = {"epoch": epoch, "train_loss": _float(objective),
                  "lr": float(scheduler.get_last_lr()[0]), "curriculum": curriculum, **parts}
        if epoch % int(args.eval_every) == 0 or epoch == epochs:
            controller.eval()
            metrics = evaluate(args, validation, device, mode, controller, plant)
            record.update(val_cost=metrics["avg_cost"],
                          val_success_rate=metrics["success_rate"],
                          val_gate_success_rate=metrics["gate_success_rate"])
            score = (-metrics["success_rate"], -metrics["gate_success_rate"], metrics["avg_cost"])
            if best_score is None or score < best_score:
                best_score, best_metrics = score, metrics
                best_state = {key: value.detach().cpu().clone()
                              for key, value in controller.state_dict().items()}
            print(f"[{mode}] {epoch:03d}/{epochs}: loss={record['train_loss']:.3f} "
                  f"joint={metrics['success_rate']:.3f} gates={metrics['gate_success_rate']:.3f} "
                  f"payload_collision={metrics['payload_collision_rate']:.3f}")
        history.append(record)
    if best_state is None:
        raise RuntimeError(f"Training {mode} did not produce a validation checkpoint.")
    controller.load_state_dict(best_state)
    return controller, plant, history, best_metrics


def _evaluate_interventions(args, batch, ood, device, controllers, plants, metrics):
    interventions = {}
    for mode in ("context", "mad_context", "contextual_ssm"):
        if mode not in controllers:
            continue
        interventions[mode] = {
            "truth": metrics[mode],
            "delayed telemetry": evaluate(args, batch, device, mode, controllers[mode], plants[mode], "delayed"),
            "wrong telemetry": evaluate(args, batch, device, mode, controllers[mode], plants[mode], "wrong"),
            "missing telemetry": evaluate(args, batch, device, mode, controllers[mode], plants[mode], "dropout"),
            "OOD mass + tether": evaluate(args, ood, device, mode, controllers[mode], plants[mode]),
        }
    return interventions


def _load_args_for_replay(args, run_dir: Path):
    saved = json.loads((run_dir / "config.json").read_text())
    if int(saved.get("experiment_schema_version", 0)) != 3:
        raise RuntimeError("This renderer only accepts nonlinear Tethered Cargo Slalom schema version 3 runs.")
    cli_flags = {token.split("=")[0] for token in sys.argv[1:] if token.startswith("--")}
    merged = vars(args).copy()
    merged.update({key: value for key, value in saved.items() if f"--{key}" not in cli_flags})
    return argparse.Namespace(**merged)


def replay(args, device: torch.device, run_dir: Path) -> None:
    args = _load_args_for_replay(args, run_dir)
    specs = variant_specs(args)
    batch_path = run_dir / "test_batch.pt"
    if not batch_path.exists():
        raise FileNotFoundError(
            "Cannot replay this schema-v3 run: test_batch.pt is missing, so an exact "
            "artifact reproduction would be impossible.")
    batch = load_batch(batch_path)
    ood = sample_slalom_batch(args, batch_size=int(args.test_batch),
                              seed=int(args.seed) + 70000, paired=True,
                              shuffle=False, test=True)
    controllers, plants, metrics = {}, {}, {}
    for mode, _ in specs:
        if mode == "nominal":
            controllers[mode] = plants[mode] = None
        else:
            checkpoint = run_dir / f"{mode}_controller.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Cannot replay {mode}: missing {checkpoint.name}.")
            controller, plant = build_slalom_controller(device, args, mode)
            controller.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
            controller.eval(); controllers[mode], plants[mode] = controller, plant
        metrics[mode] = evaluate(args, batch, device, mode, controllers[mode], plants[mode])
    interventions = _evaluate_interventions(args, batch, ood, device, controllers, plants, metrics)
    histories = json.loads((run_dir / "train_history.json").read_text())
    from tethered_artifacts import render_all
    render_all(args, run_dir, batch, specs, metrics, histories, interventions,
               not args.no_show_plots)
    _save_json(run_dir / "metrics.json", {mode: _strip(value) for mode, value in metrics.items()})
    _save_json(run_dir / "interventions.json", {
        mode: {name: _strip(value) for name, value in cases.items()}
        for mode, cases in interventions.items()})


def run(args) -> None:
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    if args.plot_only:
        replay(args, device, base_dir / "runs" / args.plot_only)
        return
    gate_geometry(args)
    run_id = args.run_id or f"tethered_slalom_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args)); config.update(
        context_dim=context_dim(args), experiment_type="tethered_payload_slalom",
        experiment_schema_version=3,
        disturbance_identity="w[0]=0; w[t]=process_noise[t-1] for t>=1")
    _save_json(run_dir / "config.json", config)
    validation = sample_slalom_batch(args, batch_size=int(args.val_batch),
                                      seed=int(args.seed) + 50000,
                                      paired=True, shuffle=False)
    test = sample_slalom_batch(args, batch_size=int(args.test_batch),
                               seed=int(args.seed) + 60000,
                               paired=True, shuffle=False)
    ood = sample_slalom_batch(args, batch_size=int(args.test_batch),
                              seed=int(args.seed) + 70000,
                              paired=True, shuffle=False, test=True)
    save_batch(run_dir / "test_batch.pt", test)
    specs = variant_specs(args)
    controllers, plants, histories, val_metrics, metrics = {}, {}, {}, {}, {}
    print(f"Tethered Cargo Slalom: {run_dir} | context dim={context_dim(args)} | device={device}")
    for mode, label in specs:
        print(f"Training {label}...")
        controller, plant, history, val_result = train(args, device, mode, validation)
        controllers[mode], plants[mode] = controller, plant
        histories[mode], val_metrics[mode] = history, val_result
        metrics[mode] = evaluate(args, test, device, mode, controller, plant)
        if controller is not None:
            torch.save(controller.state_dict(), run_dir / f"{mode}_controller.pt")
        _save_json(run_dir / "metrics.json", {key: _strip(value) for key, value in metrics.items()})
        _save_json(run_dir / "train_history.json", histories)
    interventions = _evaluate_interventions(args, test, ood, device, controllers, plants, metrics)
    _save_json(run_dir / "val_metrics.json", {key: _strip(value) for key, value in val_metrics.items()})
    _save_json(run_dir / "interventions.json", {
        mode: {name: _strip(value) for name, value in cases.items()}
        for mode, cases in interventions.items()})
    if not args.skip_plots:
        from tethered_artifacts import render_all
        render_all(args, run_dir, test, specs, metrics, histories, interventions,
                   not args.no_show_plots)
    summary = " | ".join(f"{label}: {metrics[mode]['success_rate']:.3f}"
                         for mode, label in specs)
    (run_dir / "interpretation.txt").write_text(
        summary + "\nJoint success requires carrier, cargo, and tether to clear every gate, "
        "then dock and settle. Route-only has identical context capacity but all payload "
        "telemetry slots are zero. Nominal and true observed dynamics share the same "
        "nonlinear pre-stabilised carrier, so reconstructed w is exactly the tapered "
        "process noise; the hidden nonlinear cargo affects the task constraints, not w.\n",
        encoding="utf-8")
    print("RESULTS  " + summary)
