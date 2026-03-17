# Moving Gate Experiment

A planar navigation task that demonstrates the advantage of **contextual
Performance Boosting** over a disturbance-only baseline.

![Sample rollout](experiments/contextual_pb_gate_ssm/runs/controlled_xy_20260317_150748/rollout_animation_04_idx1.gif)

---

## Task description

A pre-stabilised double integrator must reach the origin `(0, 0)` while
passing through a **moving gate** embedded in a wall at `x = x_w`.
The gate opening has half-width `h` and its centre `g_t` switches
randomly between two positions at discrete times, then freezes before
the crossing window.

The robot must decide **in real time** when to commit to a crossing
direction — too early and it may be caught by a late switch; too late
and it overshoots the goal.  Neither the nominal pre-stabiliser nor a
disturbance-only PB operator has access to `g_t`, so they cannot adapt.
The context-enriched PB operator receives a compact, causal summary of
the gate and learns to time its corrective action accordingly.

---

## Setup

| Quantity | Description |
|---|---|
| State | 2D position + velocity `(x, y, vx, vy)` |
| Control | 2D force input `(ux, uy)` |
| Gate | Centre `g_t` switches stochastically, freezes `gate_settle_steps` before the wall |
| Context `z_t` | Gate error `(y_t − g_t)`, proximity to wall `α_t`, gate switch age `σ_t` |
| Horizon | 160 steps, `dt = 0.05 s` |

The three context features are **directly observable** without knowledge
of the freeze schedule: a rising switch age `σ_t` near the wall
(`α_t ≈ 1`) indicates that the gate has been stable long enough to
commit.

---

## Variants compared

| Variant | Description |
|---|---|
| **Nominal** | Pre-stabiliser only, no PB correction |
| **PB: no context** | PB+SSM operator seeing only disturbance `w_t` |
| **PB: factorized M_b × M_p** | PB+SSM with context-aware factorized operator |

---

## Running the experiment

From the repository root:

```bash
cd experiments/contextual_pb_gate_ssm
python Moving_gate_exp.py --no_show_plots
```

To reproduce plots from a completed run without retraining:

```bash
cd experiments/contextual_pb_gate_ssm
python Moving_gate_exp.py --plot_only controlled_xy_<timestamp>
```

Results are written to:

```
experiments/contextual_pb_gate_ssm/runs/<run_id>/
```

Key outputs per run:

| File | Description |
|---|---|
| `wall_style_summary.png` | Trajectory overview + success rates |
| `trajectory_samples.png` | Per-sample top-down trajectories |
| `adversarial_switching.png` | Performance under late adversarial gate switch |
| `sample_trajectory.pdf` | Publication-quality trajectory figure |
| `rollout_animation_*.gif` | Animated rollouts |
| `*_controller.pt` | Saved controller weights |
| `metrics.json` | Numerical evaluation metrics |
