"""Publication-style figures and GIFs for the payload-regime experiment."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from payload_core import PayloadBatch

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))


def _plt(show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False,
                         "axes.spines.right": False, "grid.alpha": .20, "font.size": 10})
    return plt


def _colors() -> dict[str, str]:
    return {"nominal": "#64748b", "disturbance_only": "#d97706", "context": "#0f766e", "mad_context": "#2563eb", "contextual_ssm": "#16a34a"}


def _close(plt, fig, show: bool) -> None:
    if show:
        plt.show(block=False)
    plt.close(fig)


def plot_loss_curves(run_dir: Path, histories: dict, specs: list[tuple[str, str]], show: bool) -> None:
    plt, colors = _plt(show), _colors(); fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for mode, label in specs:
        hist = histories.get(mode, [])
        if not hist:
            continue
        epochs = [x["epoch"] for x in hist]; axes[0].plot(epochs, [x["train_loss"] for x in hist], lw=2, color=colors[mode], label=label)
        val = [x for x in hist if "val_success_rate" in x]
        if val:
            axes[1].plot([x["epoch"] for x in val], [x["val_success_rate"] for x in val], lw=2, color=colors[mode], label=label)
    axes[0].set(title="Training objective", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation docking success", xlabel="epoch", ylabel="success rate", ylim=(-.02, 1.02))
    if axes[0].lines: axes[0].legend(fontsize=8)
    if axes[1].lines: axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(run_dir / "loss_curves.png", bbox_inches="tight"); _close(plt, fig, show)


def plot_summary(run_dir: Path, specs: list[tuple[str, str]], metrics: dict, show: bool) -> None:
    plt, colors = _plt(show), _colors(); labels = [label for _, label in specs]; x = np.arange(len(specs))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2)); c = [colors[k] for k, _ in specs]
    axes[0].bar(x, [metrics[k]["success_rate"] for k, _ in specs], color=c); axes[0].set(title="Docking success", ylim=(0, 1.02), ylabel="rate")
    axes[1].bar(x, [metrics[k]["avg_terminal_dist"] for k, _ in specs], color=c); axes[1].set(title="Terminal position error", ylabel="distance")
    axes[2].bar(x, [metrics[k]["avg_post_switch_error"] for k, _ in specs], color=c); axes[2].set(title="Post-switch error", ylabel="mean distance")
    for ax in axes: ax.set_xticks(x, labels, rotation=13, ha="right")
    fig.suptitle("Payload-regime performance on matched held-out scenarios", y=1.03, fontweight="bold")
    fig.tight_layout(); fig.savefig(run_dir / "payload_regime_summary.png", bbox_inches="tight"); _close(plt, fig, show)


def _aligned(batch: PayloadBatch, values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    switch = batch.switch_step.numpy(); valid = np.flatnonzero(switch >= 0); grid = np.arange(-window, window + 1)
    rows = []
    for idx in valid:
        times = switch[idx] + grid; keep = (times >= 0) & (times < values.shape[1])
        row = np.full(grid.shape, np.nan, np.float32); row[keep] = values[idx, times[keep]]; rows.append(row)
    return grid, np.asarray(rows, np.float32)


def _band(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 15–85% bands without warnings at ragged episode edges."""
    mean, low, high = (np.full(rows.shape[1], np.nan, np.float32) for _ in range(3))
    for col in range(rows.shape[1]):
        values = rows[:, col][np.isfinite(rows[:, col])]
        if len(values):
            mean[col], low[col], high[col] = values.mean(), np.percentile(values, 15), np.percentile(values, 85)
    return mean, low, high


def plot_switch_response(args, run_dir: Path, batch: PayloadBatch, specs: list[tuple[str, str]], metrics: dict, show: bool) -> None:
    plt, colors = _plt(show), _colors(); fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.1)); window = min(32, int(args.horizon) // 3)
    mass = batch.mass_true.numpy(); grid, mass_a = _aligned(batch, mass, window)
    if mass_a.size:
        mean, low, high = _band(mass_a); valid = np.isfinite(mean)
        axes[0].plot(grid[valid], mean[valid], color="#0f172a", lw=2.2, label="true payload mass")
        axes[0].fill_between(grid[valid], low[valid], high[valid], color="#94a3b8", alpha=.35)
    for mode, label in specs:
        roll = metrics[mode]["rollout"]; distance = roll["distance"].numpy(); control = np.linalg.norm(roll["u_seq"].numpy(), axis=-1)
        for ax, values in zip(axes[1:], (distance, control)):
            _, aligned = _aligned(batch, values, window)
            if aligned.size:
                mean, low, high = _band(aligned); valid = np.isfinite(mean)
                ax.plot(grid[valid], mean[valid], color=colors[mode], lw=2, label=label)
                ax.fill_between(grid[valid], low[valid], high[valid], color=colors[mode], alpha=.12)
    for ax in axes:
        ax.axvline(0, color="#be123c", lw=1.3, ls="--"); ax.set_xlabel("steps relative to load switch")
    axes[0].set(title="Payload event", ylabel="mass ratio"); axes[1].set(title="Goal error response", ylabel="distance"); axes[2].set(title="Control response", ylabel="|u|")
    axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(run_dir / "switch_response_summary.png", bbox_inches="tight"); _close(plt, fig, show)


def plot_transition_heatmap(run_dir: Path, batch: PayloadBatch, mode: str, metrics: dict, show: bool) -> None:
    plt = _plt(show); before, after = batch.mass_before.numpy(), batch.mass_after.numpy(); success = metrics[mode]["rollout"]["success"].numpy().astype(float)
    levels = np.unique(np.r_[before, after]); matrix = np.full((len(levels), len(levels)), np.nan)
    for i, b in enumerate(levels):
        for j, a in enumerate(levels):
            mask = np.isclose(before, b) & np.isclose(after, a)
            if mask.any(): matrix[i, j] = success[mask].mean()
    fig, ax = plt.subplots(figsize=(5.4, 4.5)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
    for i in range(len(levels)):
        for j in range(len(levels)):
            if np.isfinite(matrix[i, j]): ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontweight="bold")
    ax.set(xticks=range(len(levels)), yticks=range(len(levels)), xticklabels=[f"{x:.2f}" for x in levels], yticklabels=[f"{x:.2f}" for x in levels], xlabel="payload mass after switch", ylabel="payload mass before switch", title=f"{mode}: success by payload transition")
    fig.colorbar(image, ax=ax, label="success rate"); fig.tight_layout(); fig.savefig(run_dir / "regime_transition_heatmap.png", bbox_inches="tight"); _close(plt, fig, show)


def plot_interventions(run_dir: Path, intervention_metrics: dict, show: bool) -> None:
    if not intervention_metrics:
        return
    plt, colors = _plt(show), _colors(); modes = list(intervention_metrics); labels = list(next(iter(intervention_metrics.values())))
    display = {"truth": "ID truth", "delayed telemetry": "delay", "wrong telemetry": "wrong", "missing telemetry": "missing", "OOD payload + timing": "OOD"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2)); x = np.arange(len(labels)); width = .78 / max(len(modes), 1)
    for i, mode in enumerate(modes):
        values = intervention_metrics[mode]; offset = (i - (len(modes) - 1) / 2) * width; color = colors.get(mode, "#0f766e")
        axes[0].bar(x + offset, [values[k]["success_rate"] for k in labels], width, label=mode, color=color, alpha=.88)
        axes[1].bar(x + offset, [values[k]["avg_post_switch_error"] for k in labels], width, label=mode, color=color, alpha=.88)
    labels = [display.get(label, label) for label in labels]
    axes[0].set(title="Sensor-integrity intervention: success", xticks=x, xticklabels=labels, ylim=(0, 1.02), ylabel="rate")
    axes[1].set(title="Sensor-integrity intervention: post-switch error", xticks=x, xticklabels=labels, ylabel="distance")
    for ax in axes: ax.tick_params(axis="x", rotation=12)
    axes[0].legend(title="controller", fontsize=8); fig.tight_layout(); fig.savefig(run_dir / "payload_context_intervention.png", bbox_inches="tight"); _close(plt, fig, show)


def _sample_indices(batch: PayloadBatch, count: int) -> list[int]:
    changed = np.flatnonzero(batch.switch_step.numpy() >= 0)
    candidates = list(range(len(batch.start))) if not len(changed) else changed.tolist()
    # Mirrored pair members have the same event/noise. Show independent
    # scenarios in the animation gallery instead of near-duplicate rollouts.
    first_of_pair, seen = [], set()
    for idx in candidates:
        pair = int(batch.pair_id[idx])
        if pair not in seen:
            first_of_pair.append(idx); seen.add(pair)
    if not first_of_pair: return []
    return [first_of_pair[i] for i in np.linspace(0, len(first_of_pair) - 1, min(count, len(first_of_pair)), dtype=int)]


def plot_trajectories(args, run_dir: Path, batch: PayloadBatch, specs: list[tuple[str, str]], metrics: dict, show: bool) -> None:
    plt, colors = _plt(show), _colors(); indices = _sample_indices(batch, int(args.sample_traj_count)); fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4.4), sharey=True)
    if len(specs) == 1: axes = [axes]
    for ax, (mode, label) in zip(axes, specs):
        xy = metrics[mode]["rollout"]["x_seq"].numpy()[..., :2]
        for idx in indices:
            s = int(batch.switch_step[idx]); ax.plot(xy[idx, :max(s, 0) + 1, 0], xy[idx, :max(s, 0) + 1, 1], color=colors[mode], lw=1.6, alpha=.34)
            ax.plot(xy[idx, max(s, 0):, 0], xy[idx, max(s, 0):, 1], color=colors[mode], lw=2.0, alpha=.82)
            if s >= 0: ax.scatter(xy[idx, s, 0], xy[idx, s, 1], s=24, color="#be123c", marker="D", zorder=4)
        ax.scatter(batch.start[indices, 0], batch.start[indices, 1], s=16, color="#475569", label="start")
        ax.scatter([0], [0], s=90, marker="*", color="#0f172a", label="dock"); ax.axhline(float(args.corridor_limit), color="#cbd5e1", lw=1); ax.axhline(-float(args.corridor_limit), color="#cbd5e1", lw=1)
        ax.set(title=label, xlabel="x", xlim=(-.1, float(args.start_x_max) + .15), ylim=(-float(args.corridor_limit) - .1, float(args.corridor_limit) + .1), aspect="equal")
    axes[0].set_ylabel("y"); axes[-1].legend(fontsize=8); fig.suptitle("Solid trail = after causal payload switch; diamond = switch", y=1.02)
    fig.tight_layout(); fig.savefig(run_dir / "trajectory_samples.png", bbox_inches="tight"); _close(plt, fig, show)


def _animate_one(args, run_dir: Path, batch: PayloadBatch, specs: list[tuple[str, str]], metrics: dict, idx: int, tag: str, show: bool) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt, colors = _plt(show), _colors(); horizon, time = int(args.horizon), np.arange(int(args.horizon)) * float(args.dt)
    fig = plt.figure(figsize=(8.8, 5.3)); fig.patch.set_facecolor("#0f172a"); grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), height_ratios=(1.35, 1), hspace=.36, wspace=.28)
    arena, mass_ax, err_ax = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    for ax in (arena, mass_ax, err_ax):
        ax.set_facecolor("#172033"); ax.tick_params(colors="#cbd5e1"); ax.xaxis.label.set_color("#e2e8f0"); ax.yaxis.label.set_color("#e2e8f0"); ax.title.set_color("#f8fafc"); ax.grid(color="#334155", alpha=.55)
    arena.scatter([0], [0], marker="*", s=165, color="#f8fafc", label="dock"); arena.scatter([batch.start[idx, 0]], [batch.start[idx, 1]], s=42, color="#94a3b8", label="start")
    arena.axhline(float(args.corridor_limit), color="#475569"); arena.axhline(-float(args.corridor_limit), color="#475569"); arena.set(xlim=(-.1, float(args.start_x_max) + .15), ylim=(-float(args.corridor_limit) - .1, float(args.corridor_limit) + .1), xlabel="x", ylabel="y", title="Payload-aware docking", aspect="equal")
    trajs, lines, dots = {}, {}, {}
    for mode, label in specs:
        xy = metrics[mode]["rollout"]["x_seq"][idx, :, :2].numpy(); trajs[mode] = xy
        arena.plot(xy[:, 0], xy[:, 1], color=colors[mode], alpha=.13, lw=1.2)
        lines[mode], = arena.plot([], [], color=colors[mode], lw=2.3, label=label); dots[mode], = arena.plot([], [], "o", color=colors[mode], ms=7)
    arena.legend(fontsize=7, loc="best")
    true_mass, observed = batch.mass_true[idx].numpy(), batch.mass_obs[idx].numpy(); mass_ax.plot(time, true_mass, color="#f8fafc", lw=2, label="true mass"); mass_ax.plot(time, observed, color="#38bdf8", lw=1.2, ls="--", alpha=.9, label="telemetry")
    switch = int(batch.switch_step[idx]);
    if switch >= 0: mass_ax.axvline(time[switch], color="#fb7185", ls="--", lw=1.5, label="switch")
    marker = mass_ax.axvline(0, color="#facc15", lw=1.6); mass_ax.set(title="Causal payload telemetry", xlabel="time [s]", ylabel="mass", xlim=(0, time[-1])); mass_ax.legend(fontsize=7)
    for mode, label in specs:
        distance = metrics[mode]["rollout"]["distance"][idx].numpy(); err_ax.plot(time, distance, color=colors[mode], lw=1.8, label=label)
    err_marker = err_ax.axvline(0, color="#facc15", lw=1.6); err_ax.set(title="Goal-error response", xlabel="time [s]", ylabel="distance", xlim=(0, time[-1])); err_ax.legend(fontsize=7)
    title = fig.suptitle("", color="#f8fafc", fontsize=13, fontweight="bold")
    def update(frame):
        for mode in trajs: lines[mode].set_data(trajs[mode][:frame + 1, 0], trajs[mode][:frame + 1, 1]); dots[mode].set_data([trajs[mode][frame, 0]], [trajs[mode][frame, 1]])
        marker.set_xdata([time[frame], time[frame]]); err_marker.set_xdata([time[frame], time[frame]]); title.set_text(f"Payload regime rollout  •  t = {time[frame]:.2f}s")
        return [*lines.values(), *dots.values(), marker, err_marker, title]
    animation = FuncAnimation(fig, update, frames=range(0, horizon, max(1, horizon // 44)), interval=65, blit=False)
    animation.save(run_dir / f"rollout_animation_{tag}_idx{idx}.gif", writer=PillowWriter(fps=12)); _close(plt, fig, show)


def make_animations(args, run_dir: Path, batch: PayloadBatch, specs: list[tuple[str, str]], metrics: dict, show: bool) -> None:
    for old in run_dir.glob("rollout_animation_*.gif"):
        old.unlink()
    for number, idx in enumerate(_sample_indices(batch, 3), start=1): _animate_one(args, run_dir, batch, specs, metrics, idx, f"{number:02d}", show)


def make_storyboard(args, run_dir: Path, batch: PayloadBatch, specs: list[tuple[str, str]], metrics: dict, show: bool) -> None:
    plt = _plt(show); indices = _sample_indices(batch, 1)
    if not indices: return
    idx, time = indices[0], np.arange(int(args.horizon)) * float(args.dt); fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True); colors = _colors()
    axes[0].plot(time, batch.mass_true[idx], color="#0f172a", lw=2.3, label="true mass"); axes[0].plot(time, batch.mass_obs[idx], color="#38bdf8", ls="--", label="telemetry")
    for mode, label in specs:
        roll = metrics[mode]["rollout"]; axes[1].plot(time, roll["distance"][idx], color=colors[mode], lw=2, label=label); axes[2].plot(time, np.linalg.norm(roll["u_seq"][idx].numpy(), axis=-1), color=colors[mode], lw=2, label=label)
    switch = int(batch.switch_step[idx]);
    for ax, title, ylabel in zip(axes, ("Payload schedule", "Docking error", "Applied PB correction"), ("mass", "distance", "|u|")):
        if switch >= 0: ax.axvline(time[switch], color="#be123c", ls="--", lw=1.2)
        ax.set(title=title, ylabel=ylabel); ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("time [s]"); fig.suptitle("Payload-switching contextual PB storyboard", y=.995, fontweight="bold"); fig.tight_layout(); fig.savefig(run_dir / "payload_storyboard.pdf", bbox_inches="tight"); _close(plt, fig, show)


def render_all(args, run_dir: Path, batch: PayloadBatch, specs: list[tuple[str, str]], metrics: dict, histories: dict, interventions: dict, show: bool) -> None:
    plot_loss_curves(run_dir, histories, specs, show); plot_summary(run_dir, specs, metrics, show); plot_switch_response(args, run_dir, batch, specs, metrics, show)
    if "context" in metrics: plot_transition_heatmap(run_dir, batch, "context", metrics, show)
    plot_interventions(run_dir, interventions, show); plot_trajectories(args, run_dir, batch, specs, metrics, show); make_animations(args, run_dir, batch, specs, metrics, show); make_storyboard(args, run_dir, batch, specs, metrics, show)
