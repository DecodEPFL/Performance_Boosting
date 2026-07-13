"""Publication-style plots, animation, and storyboard for Tethered Cargo Slalom."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from tethered_payload import gate_geometry


COLORS = {
    "nominal": "#94A3B8", "disturbance_only": "#F59E0B",
    "route_context": "#64748B", "context": "#06B6D4",
    "mad_context": "#8B5CF6", "contextual_ssm": "#2563EB",
}
INK, MUTED, WALL, SAFE, HIT = "#172033", "#667085", "#243047", "#16A34A", "#E11D48"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#F7F8FC", "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D8DEEA", "axes.labelcolor": INK,
        "axes.titlecolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#E8ECF3", "grid.linewidth": .8,
    })


def _label(specs, mode: str) -> str:
    found = dict(specs).get(mode, mode.replace("_", " ").title())
    return found.replace("PB+SSM: ", "").replace(" (mixer,input,gate)", "")


def _draw_course(ax, args, centers: np.ndarray, *, faint: bool = False) -> None:
    xs, _ = gate_geometry(args); limit = float(args.corridor_limit)
    half = float(args.slalom_gate_half_width); alpha = .40 if faint else .92
    ax.axhspan(-limit, limit, color="#F8FAFD", zorder=0)
    ax.axhline(limit, color="#CBD3E1", lw=1.2); ax.axhline(-limit, color="#CBD3E1", lw=1.2)
    for number, (x, center) in enumerate(zip(xs, centers), 1):
        ax.plot([x, x], [-limit, center - half], color=WALL, lw=7, alpha=alpha,
                solid_capstyle="round", zorder=3)
        ax.plot([x, x], [center + half, limit], color=WALL, lw=7, alpha=alpha,
                solid_capstyle="round", zorder=3)
        ax.plot([x, x], [center - half, center + half], color="#A7F3D0", lw=2,
                alpha=.75, zorder=2)
        ax.text(x, limit + .065, f"G{number}", ha="center", va="bottom",
                color=MUTED, fontsize=8, weight="bold")
    ax.scatter([0], [0], marker="*", s=135, color=SAFE, edgecolor="white", linewidth=.8, zorder=8)
    ax.set(xlim=(-.13, float(args.start_x_max) + float(args.slalom_test_tether_length_max) + .12),
           ylim=(-limit - .12, limit + .16), xlabel="down-course x", ylabel="lateral y")
    ax.set_aspect("equal", adjustable="box"); ax.grid(False)


def _body(ax, carrier: np.ndarray, payload: np.ndarray, args, color: str, alpha: float = 1.0) -> None:
    ax.plot([carrier[0], payload[0]], [carrier[1], payload[1]], color=color, lw=1.6, alpha=.75 * alpha, zorder=6)
    ax.add_patch(Circle(carrier, float(args.slalom_carrier_radius), facecolor="white",
                        edgecolor=color, lw=2, alpha=alpha, zorder=8))
    ax.add_patch(Circle(payload, float(args.slalom_payload_radius), facecolor=color,
                        edgecolor="white", lw=1, alpha=alpha, zorder=8))


def _trace(ax, args, centers, rollout, index: int, color: str, end: int | None = None,
           *, snapshots: bool = True) -> None:
    carrier = rollout["x_seq"][index, :, :2].numpy()
    payload = rollout["payload_pos_seq"][index].numpy()
    end = len(carrier) if end is None else max(1, min(int(end), len(carrier)))
    ax.plot(carrier[:end, 0], carrier[:end, 1], color=color, lw=2.4, zorder=5)
    ax.plot(payload[:end, 0], payload[:end, 1], color=color, lw=1.6,
            ls=(0, (2, 2)), alpha=.72, zorder=4)
    if snapshots:
        for step in np.linspace(0, end - 1, 7, dtype=int):
            ax.plot([carrier[step, 0], payload[step, 0]],
                    [carrier[step, 1], payload[step, 1]], color=color, lw=.7, alpha=.22, zorder=2)
    _body(ax, carrier[end - 1], payload[end - 1], args, color)


def _comparison_modes(metrics: dict) -> list[str]:
    preferred = ["disturbance_only", "route_context", "contextual_ssm", "context", "mad_context", "nominal"]
    return [mode for mode in preferred if mode in metrics]


def _best_context_mode(metrics: dict) -> str:
    candidates = [mode for mode in ("contextual_ssm", "context", "mad_context") if mode in metrics]
    return max(candidates, key=lambda mode: (metrics[mode]["success_rate"],
                                             metrics[mode]["gate_success_rate"],
                                             metrics[mode]["avg_min_clearance"]))


def _scenario(metrics: dict, reference: str, contextual: str) -> int:
    ref = metrics[reference]["rollout"]; ctx = metrics[contextual]["rollout"]
    ref_success = ref["success"].numpy().astype(bool); ctx_success = ctx["success"].numpy().astype(bool)
    contrast = np.flatnonzero(ctx_success & ~ref_success)
    if len(contrast):
        return int(contrast[0])
    ref_clear = ref["gate_clearance"].amin((1, 2)).numpy()
    ctx_clear = ctx["gate_clearance"].amin((1, 2)).numpy()
    return int(np.argmax(ctx_clear - ref_clear))


def render_hero(args, run_dir: Path, batch, specs, metrics) -> tuple[int, str, str]:
    contextual = _best_context_mode(metrics)
    reference = "route_context" if "route_context" in metrics else "disturbance_only"
    index = _scenario(metrics, reference, contextual)
    modes = [mode for mode in ("disturbance_only", reference, contextual) if mode in metrics]
    modes = list(dict.fromkeys(modes))
    fig, axes = plt.subplots(1, len(modes), figsize=(5.0 * len(modes), 4.15), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    centers = batch.gate_centers[index].numpy()
    for ax, mode in zip(axes, modes):
        _draw_course(ax, args, centers)
        _trace(ax, args, centers, metrics[mode]["rollout"], index, COLORS[mode])
        clear = float(metrics[mode]["rollout"]["gate_clearance"][index].amin())
        ok = bool(metrics[mode]["rollout"]["success"][index])
        ax.set_title(_label(specs, mode), loc="left", pad=12)
        ax.text(.02, .02, f"{'JOINT PASS' if ok else 'FAIL'}   min clearance {clear:+.3f} m",
                transform=ax.transAxes, color=SAFE if ok else HIT, fontsize=9, weight="bold")
    fig.suptitle("Tethered Cargo Slalom", x=.055, y=.99, ha="left", fontsize=18, weight="bold")
    fig.text(.055, .895, "Solid = carrier  ·  dashed = hidden cargo  ·  every tether segment must clear every gate",
             color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .82)); fig.savefig(run_dir / "tethered_slalom_hero.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return index, reference, contextual


def render_clearance(args, run_dir: Path, specs, metrics) -> None:
    modes = _comparison_modes(metrics)[:5]; gate_count = len(gate_geometry(args)[0])
    x = np.arange(gate_count); width = .78 / max(1, len(modes))
    fig, (ax, joint) = plt.subplots(1, 2, figsize=(11.8, 4.5),
                                    gridspec_kw={"width_ratios": [3.0, 1.15]})
    for order, mode in enumerate(modes):
        clearance = metrics[mode]["rollout"]["gate_clearance"].amin(2).numpy().mean(0)
        offset = (order - (len(modes) - 1) / 2) * width
        bars = ax.bar(x + offset, clearance, width * .90, color=COLORS[mode], label=_label(specs, mode), alpha=.92)
        for bar, value in zip(bars, clearance):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (.012 if value >= 0 else -.018),
                    f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7, color=INK)
    all_values = [metrics[mode]["rollout"]["gate_clearance"].amin(2).numpy().mean(0) for mode in modes]
    low, high = min(float(values.min()) for values in all_values), max(float(values.max()) for values in all_values)
    ax.axhline(0, color=HIT, lw=1.2); ax.set_xticks(x, [f"Gate {i + 1}" for i in x])
    ax.set(ylabel="mean worst-body signed clearance [m]",
           title="Signed clearance at each precision gate",
           ylim=(min(-.05, low - .025), max(.14, high + .04)))
    ax.grid(axis="y")
    y = np.arange(len(modes)); rates = [metrics[mode]["gate_success_rate"] for mode in modes]
    joint.hlines(y, 0, rates, color=[COLORS[mode] for mode in modes], lw=4, alpha=.55)
    joint.scatter(rates, y, color=[COLORS[mode] for mode in modes], s=75, zorder=4)
    for row, value in zip(y, rates):
        joint.text(min(value + .035, .95), row, f"{value:.0%}", va="center", weight="bold", color=INK)
    joint.set_yticks(y, [_label(specs, mode) for mode in modes]); joint.invert_yaxis()
    joint.set(xlim=(0, 1.02), xlabel="joint gate success", title="All 3 gates safe")
    joint.grid(axis="x"); joint.spines["left"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=min(3, len(modes)),
               loc="lower center", bbox_to_anchor=(.5, -.015))
    fig.suptitle("One weak cargo crossing fails the episode", x=.06, y=.99,
                 ha="left", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, .10, 1, .92))
    fig.savefig(run_dir / "gate_clearance_profile.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def render_ambiguity(args, run_dir: Path, batch, specs, metrics, reference: str, contextual: str) -> None:
    pair_ids = batch.pair_id.numpy(); pair = pair_ids[0]; members = np.flatnonzero(pair_ids == pair)[:2]
    if len(members) < 2: return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    centers = batch.gate_centers[members[0]].numpy(); start = batch.start[members[0]].numpy()
    _draw_course(axes[0], args, centers, faint=True)
    for member, color in zip(members, ("#DB2777", "#0EA5E9")):
        payload = batch.payload_start[member].numpy(); velocity = batch.payload_velocity_start[member].numpy()
        _body(axes[0], start, payload, args, color, .84)
        axes[0].arrow(payload[0], payload[1], 0, velocity[1] * .36, color=color,
                      width=.006, head_width=.06, length_includes_head=True, zorder=9)
    axes[0].set_title("Same carrier + route, opposite hidden swing", loc="left")
    axes[0].text(.02, .02, "The observed PB state is identical at launch.", transform=axes[0].transAxes, color=MUTED)
    axes[1].axhline(0, color="#D8DEEA", lw=1); axes[1].axvline(0, color="#D8DEEA", lw=1)
    y = np.array([-.18, .18]); labels = ["alias A", "alias B"]
    for row, member, marker in zip(y, members, ("o", "s")):
        for mode, shift in ((reference, -.045), (contextual, .045)):
            control = metrics[mode]["rollout"]["u_seq"][member, 0].numpy()
            axes[1].arrow(0, row + shift, control[0] * .12, control[1] * .12,
                          color=COLORS[mode], width=.008, head_width=.055,
                          length_includes_head=True)
        axes[1].scatter([0], [row], marker=marker, color=INK, s=32, zorder=8)
    axes[1].set_yticks(y, labels); axes[1].set(xlim=(-.28, .28), ylim=(-.36, .36),
                                              xlabel="first control response (scaled)", title="Context breaks the alias")
    axes[1].plot([], [], color=COLORS[reference], lw=3, label=_label(specs, reference))
    axes[1].plot([], [], color=COLORS[contextual], lw=3, label=_label(specs, contextual))
    axes[1].grid(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.suptitle("Why context is necessary", x=.055, y=.99, ha="left", fontsize=17, weight="bold")
    fig.legend(handles, labels, frameon=False, loc="upper center",
               bbox_to_anchor=(.75, .86), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .79)); fig.savefig(run_dir / "context_alias_pair.png", dpi=185, bbox_inches="tight"); plt.close(fig)


def render_interventions(run_dir: Path, specs, interventions: dict) -> None:
    if not interventions: return
    modes = [mode for mode in ("context", "mad_context", "contextual_ssm") if mode in interventions]
    cases = list(interventions[modes[0]])
    x = np.arange(len(cases)); width = .76 / len(modes)
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    for i, mode in enumerate(modes):
        values = [interventions[mode][case]["success_rate"] for case in cases]
        ax.bar(x + (i - (len(modes) - 1) / 2) * width, values, width * .9,
               color=COLORS[mode], label=_label(specs, mode))
    ax.set_xticks(x, [case.replace(" telemetry", "\ntelemetry") for case in cases])
    ax.set(ylim=(0, 1.04), ylabel="joint success rate", title="Causal sensor interventions expose what the policy uses")
    ax.grid(axis="y"); ax.legend(frameon=False, ncol=len(modes)); fig.tight_layout()
    fig.savefig(run_dir / "payload_context_interventions.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def render_training(run_dir: Path, specs, histories: dict) -> None:
    modes = [mode for mode in _comparison_modes(histories) if histories.get(mode)]
    if not modes: return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for mode in modes:
        history = histories[mode]; epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], color=COLORS[mode], label=_label(specs, mode))
        points = [(row["epoch"], row["val_gate_success_rate"]) for row in history if "val_gate_success_rate" in row]
        if points: axes[1].plot(*zip(*points), color=COLORS[mode], marker="o", ms=3)
    axes[0].set(title="Optimisation", xlabel="epoch", ylabel="training objective"); axes[0].set_yscale("log")
    axes[1].set(title="Validation gate clearance", xlabel="epoch", ylabel="joint gate success", ylim=(-.02, 1.02))
    for ax in axes: ax.grid(True, alpha=.8)
    axes[0].legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(run_dir / "tethered_training.png", dpi=175, bbox_inches="tight"); plt.close(fig)


def _event_steps(args, rollout: dict, index: int) -> list[int]:
    x = rollout["x_seq"][index, :, 0].numpy(); xs = gate_geometry(args)[0]
    steps = [0]
    for gate_x in xs:
        hit = np.flatnonzero(x <= gate_x)
        steps.append(int(hit[0]) if len(hit) else len(x) - 1)
    steps.append(len(x) - 1)
    return steps


def render_storyboard(args, run_dir: Path, batch, specs, metrics, index: int,
                      reference: str, contextual: str) -> None:
    modes = [reference, contextual]; events = _event_steps(args, metrics[contextual]["rollout"], index)
    names = ["launch"] + [f"gate {i + 1}" for i in range(len(events) - 2)] + ["dock + settle"]
    fig, axes = plt.subplots(2, len(events), figsize=(3.25 * len(events), 5.35), sharex=True, sharey=True)
    centers = batch.gate_centers[index].numpy()
    for row, mode in enumerate(modes):
        for column, (step, name) in enumerate(zip(events, names)):
            ax = axes[row, column]; _draw_course(ax, args, centers, faint=True)
            _trace(ax, args, centers, metrics[mode]["rollout"], index, COLORS[mode], step + 1, snapshots=False)
            if row == 0: ax.set_title(name.title(), fontsize=10)
            if column == 0: ax.set_ylabel(_label(specs, mode) + "\nlateral y")
            else: ax.set_ylabel("")
            ax.set_xlabel("" if row == 0 else "x")
    fig.suptitle("Event storyboard · one aliased swing scenario", x=.05, y=.985, ha="left", fontsize=17, weight="bold")
    fig.subplots_adjust(left=.055, right=.995, bottom=.08, top=.87, wspace=.06, hspace=.16)
    fig.savefig(run_dir / "tethered_storyboard.png", dpi=170, bbox_inches="tight")
    fig.savefig(run_dir / "tethered_storyboard.pdf", bbox_inches="tight"); plt.close(fig)


def render_gif(args, run_dir: Path, batch, specs, metrics, index: int,
               reference: str, contextual: str) -> None:
    modes = [reference, contextual]; horizon = int(args.horizon)
    frames = np.unique(np.linspace(0, horizon - 1, min(96, horizon), dtype=int))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True, sharey=True)
    centers = batch.gate_centers[index].numpy()
    def update(frame_index):
        step = int(frames[frame_index]); artists = []
        for ax, mode in zip(axes, modes):
            ax.clear(); _draw_course(ax, args, centers, faint=True)
            _trace(ax, args, centers, metrics[mode]["rollout"], index, COLORS[mode], step + 1, snapshots=False)
            clear = float(metrics[mode]["rollout"]["gate_clearance"][index].amin())
            ax.set_title(_label(specs, mode), loc="left")
            ax.text(.98, .02, f"t = {step * float(args.dt):.1f} s  ·  run clearance {clear:+.2f} m",
                    transform=ax.transAxes, ha="right", color=MUTED, fontsize=8)
            artists.extend(ax.lines + ax.patches + ax.texts)
        fig.suptitle("Same route. Same carrier state. Payload telemetry changes the move.",
                     x=.06, ha="left", fontsize=14, weight="bold")
        return artists
    animation = FuncAnimation(fig, update, frames=len(frames), interval=65, blit=False)
    animation.save(run_dir / "tethered_slalom_comparison.gif", writer=PillowWriter(fps=15), dpi=105)
    plt.close(fig)


def render_all(args, run_dir: Path, batch, specs, metrics, histories, interventions, show: bool = False) -> None:
    del show
    _style(); index, reference, contextual = render_hero(args, run_dir, batch, specs, metrics)
    render_clearance(args, run_dir, specs, metrics)
    render_ambiguity(args, run_dir, batch, specs, metrics, reference, contextual)
    render_interventions(run_dir, specs, interventions)
    render_training(run_dir, specs, histories)
    render_storyboard(args, run_dir, batch, specs, metrics, index, reference, contextual)
    render_gif(args, run_dir, batch, specs, metrics, index, reference, contextual)
    manifest = {
        "hero_scenario": index, "reference_variant": reference,
        "contextual_variant": contextual,
        "artifacts": [path.name for path in sorted(run_dir.glob("tethered_*"))]
                     + ["gate_clearance_profile.png", "context_alias_pair.png",
                        "payload_context_interventions.png"],
    }
    (run_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
