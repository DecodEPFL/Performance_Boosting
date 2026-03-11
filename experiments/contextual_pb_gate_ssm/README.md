# Contextual PB Gate SSM

This folder contains an isolated version of the contextual gate experiment.

What is different from the older self-contained skeleton:

- It uses the shared `PBController` stack instead of a tiny stand-alone MLP.
- `M_p` is `MpDeepSSM`, so disturbance processing is handled by the project SSM.
- `M_p(w)` is augmented with a context signal passed through `LpContextLifter`, so the SSM sees a filtered `l_p`-compatible context sequence.
- The wall is placed at the origin and trajectories start to the left, so the final goal geometry is centered at `x=0`.
- The gate schedule is frozen before the wall, so the final switch is not too late to matter.
- The contextual signal is richer than a single gate scalar, but it remains causal.

Run the scalar lateral experiment from the repository root with:

```bash
python experiments/contextual_pb_gate_ssm/run_experiment.py --no_show_plots
```

Run the controlled-`x/y` variant, where the learned control also affects the forward motion and the terminal goal is the origin `(0, 0)`, with:

```bash
python experiments/contextual_pb_gate_ssm/run_experiment_controlled_xy.py --no_show_plots
```

By default, the disturbance-only PB baseline is trained for fewer epochs than the context-aware model. Override it with `--disturbance_only_epochs`.

The controlled-`x/y` script now defaults to a less predictable gate process: stochastic switch timing (`--gate_process hazard`, `--gate_switch_prob`) and per-trajectory settle-time jitter (`--gate_settle_jitter`). Use `--gate_process alternating` to recover the older deterministic alternating schedule.

It also includes explicit post-wall recovery penalties so trajectories are pushed back toward the origin after clearing the wall instead of taking large detours (`--post_wall_goal_weight`, `--post_wall_lateral_weight`, `--origin_overshoot_weight`).

To plot-check that the overall factorized operator `M(w,z)` decays to zero for decaying source signals, run:

```bash
python experiments/contextual_pb_gate_ssm/plot_mp_input_decay_test.py --no_show_plots
```

Outputs are written under:

```text
experiments/contextual_pb_gate_ssm/runs/<run_id>/
```

The controlled-`x/y` script writes additional figures:

- `wall_style_summary.png`
- `loss_curves.png`
- `control_magnitude_over_time.png`
- `trajectory_samples.png`

The decay diagnostic writes:

- `operator_decay_test.png`
- `metrics.json`
