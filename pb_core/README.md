# `pb_core` quickstart

`pb_core` provides reusable building blocks to run PB experiments with consistent signatures and compatibility checks.

## Core pieces

- `interfaces.py`: Protocols for dataset/context/noise/loss/metrics.
- `rollout.py`: Generic causal rollout (`rollout_pb`).
- `runner.py`: Generic trainer/evaluator (`PBExperimentRunner`).
- `validation.py`: One-step compatibility checks.
- `noise.py`: Reusable noise models.
- `factories.py`: Convenience assembly for factorized controllers.

## What an experiment must provide

- nominal plant (`nominal_dynamics`) for disturbance reconstruction.
- true plant (`forward`) for rollout.
- controller (`PBController`) with compatible `u`/`w` shapes.
- dataset provider for train/val scenario sampling and initial state generation.
- context builder from `(x_t, batch)` to `z_t`.
- loss function (optimization objective).
- metrics function (evaluation KPIs).
- optional noise models for train/eval.

## What else is usually needed

- checkpoint policy (best-metric selection).
- validation suites/ablations (true vs shuffled vs zero context).
- plotting/reporting hooks.

`pb_core` keeps these concerns modular, so task logic stays outside the controller core.

## Expected tensor shapes

- state: `(B,T,Nx)`
- control: `(B,T,Nu)`
- disturbance (reconstructed): `(B,T,Nx)`
- context: `(B,T,Nz)` or `(B,1,Nz)` at step-time

## Minimal usage skeleton

```python
from pb_core import (
    PBExperimentRunner, RunnerConfig, DecayingGaussianNoise,
    validate_component_compatibility
)

# Build controller/plant/dataset/context/loss/metrics (task-specific).

ok, msg = validate_component_compatibility(
    controller=controller,
    plant_true=plant_true,
    x0=x_probe,   # (B,1,Nx)
    z0=z_probe,   # (B,1,Nz)
    raise_on_error=False,
)
if not ok:
    raise RuntimeError(msg)

runner = PBExperimentRunner(
    controller=controller,
    plant_true=plant_true,
    dataset=dataset,
    context_builder=context_builder,
    loss_fn=loss_fn,
    metrics_fn=metrics_fn,
    train_noise=DecayingGaussianNoise(sigma0=0.01, tau=20.0),
)

result = runner.train(
    RunnerConfig(epochs=200, eval_every=5, horizon=80, val_horizon=120)
)
```

## Backward compatibility

Existing scripts can continue to use `PBController` directly.
`pb_core` is additive: it does not require changing old experiments.

## Minimal runnable template

See:
- `/Users/leo/Library/CloudStorage/GoogleDrive-l.massai@epfl.ch/My Drive/Code/Projects/Performance_Boosting/pb_core_minimal_example.py`

Quick smoke run:

```bash
source .venv/bin/activate
python pb_core_minimal_example.py --epochs 2 --eval_every 1 --batch 64 --val_batch 64 --horizon 30 --val_horizon 40 --no_show_plots
```
