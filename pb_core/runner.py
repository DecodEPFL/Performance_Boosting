"""Generic user-friendly trainer for PB experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from pb_controller import PBController, as_bt

from .interfaces import BatchData, ContextBuilder, DatasetProvider, LossFn, MetricsFn, NoiseModel
from .noise import ZeroNoise
from .rollout import RolloutResult, rollout_pb


@dataclass
class RunnerConfig:
    seed: int = 0
    epochs: int = 100
    eval_every: int = 5
    batch_size: int = 256
    val_batch_size: int = 256
    horizon: int = 80
    val_horizon: int = 0
    lr: float = 2e-3
    lr_min: float = 1e-4
    grad_clip: float = 2.0
    best_metric: str = "loss"
    best_mode: str = "min"  # "min" or "max"
    verbose: bool = True


class PBExperimentRunner:
    """
    Generic training/evaluation loop for PB controllers.

    This runner keeps all task-specific logic modular:
      - dataset provider
      - context builder
      - noise model
      - loss
      - metrics
    """

    def __init__(
        self,
        *,
        controller: PBController,
        plant_true,
        dataset: DatasetProvider,
        context_builder: ContextBuilder,
        loss_fn: LossFn,
        metrics_fn: MetricsFn,
        train_noise: NoiseModel | None = None,
        eval_noise: NoiseModel | None = None,
        device: torch.device | None = None,
    ):
        self.controller = controller
        self.plant_true = plant_true
        self.dataset = dataset
        self.context_builder = context_builder
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.train_noise = train_noise if train_noise is not None else ZeroNoise()
        self.eval_noise = eval_noise if eval_noise is not None else ZeroNoise()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _context_fn(self, batch: BatchData):
        def _fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
            z = self.context_builder.build(x_t, batch)
            return as_bt(z)

        return _fn

    def _rollout(
        self,
        *,
        batch: BatchData,
        horizon: int,
        noise_model: NoiseModel,
        seed: int | None,
    ) -> RolloutResult:
        x0 = self.dataset.make_x0(batch, device=self.device)
        nx = x0.shape[-1]
        noise = noise_model.sample(
            bsz=x0.shape[0],
            horizon=horizon,
            nx=nx,
            device=self.device,
            seed=seed,
        )
        return rollout_pb(
            controller=self.controller,
            plant_true=self.plant_true,
            x0=x0,
            horizon=horizon,
            context_fn=self._context_fn(batch),
            process_noise_seq=noise,
            w0=x0,
        )

    @torch.no_grad()
    def evaluate(self, *, batch: BatchData, horizon: int, seed: int | None = None) -> tuple[dict, RolloutResult]:
        self.controller.eval()
        out = self._rollout(batch=batch, horizon=horizon, noise_model=self.eval_noise, seed=seed)
        loss, loss_parts = self.loss_fn(x_seq=out.x_seq, u_seq=out.u_seq, w_seq=out.w_seq, batch=batch)
        metrics = self.metrics_fn(x_seq=out.x_seq, u_seq=out.u_seq, w_seq=out.w_seq, batch=batch)
        metrics = {**metrics, **loss_parts, "loss": float(loss.item())}
        return metrics, out

    def train(self, cfg: RunnerConfig) -> dict[str, Any]:
        torch.manual_seed(int(cfg.seed))
        optimizer = optim.Adam(self.controller.parameters(), lr=float(cfg.lr))
        scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg.epochs), eta_min=float(cfg.lr_min))

        val_h = int(cfg.val_horizon) if int(cfg.val_horizon) > 0 else int(cfg.horizon)
        val_batch = self.dataset.sample_val(batch_size=int(cfg.val_batch_size), seed=int(cfg.seed) + 999)

        train_history: list[dict[str, float]] = []
        eval_history: list[dict[str, float]] = []

        best_score = None
        best_state = None
        best_epoch = 0

        for epoch in range(1, int(cfg.epochs) + 1):
            self.controller.train()
            optimizer.zero_grad()

            batch = self.dataset.sample_train(
                epoch=epoch,
                batch_size=int(cfg.batch_size),
                seed=int(cfg.seed) + epoch,
            )
            out = self._rollout(
                batch=batch,
                horizon=int(cfg.horizon),
                noise_model=self.train_noise,
                seed=int(cfg.seed) + 10000 + epoch,
            )
            loss, loss_parts = self.loss_fn(x_seq=out.x_seq, u_seq=out.u_seq, w_seq=out.w_seq, batch=batch)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=float(cfg.grad_clip))
            optimizer.step()
            scheduler.step()
            train_history.append({"epoch": float(epoch), "loss": float(loss.item()), **{k: float(v) for k, v in loss_parts.items()}})

            if epoch % int(cfg.eval_every) == 0:
                metrics, _ = self.evaluate(batch=val_batch, horizon=val_h, seed=int(cfg.seed) + 20000 + epoch)
                metrics["epoch"] = float(epoch)
                eval_history.append(metrics)
                if cfg.verbose:
                    print(
                        f"Epoch {epoch:03d}/{int(cfg.epochs)} | "
                        f"Train {float(loss.item()):.4f} | "
                        f"Val {float(metrics['loss']):.4f} | "
                        f"{cfg.best_metric} {float(metrics[cfg.best_metric]):.4f} | "
                        f"LR {float(scheduler.get_last_lr()[0]):.2e}"
                    )

                score = float(metrics[cfg.best_metric])
                if best_score is None:
                    better = True
                elif cfg.best_mode == "max":
                    better = score > best_score
                else:
                    better = score < best_score
                if better:
                    best_score = score
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu() for k, v in self.controller.state_dict().items()}

        if best_state is not None:
            self.controller.load_state_dict(best_state)

        final_metrics, final_rollout = self.evaluate(
            batch=val_batch,
            horizon=val_h,
            seed=int(cfg.seed) + 30000,
        )
        if cfg.verbose:
            print(
                f"Training done | best_epoch {int(best_epoch)} | "
                f"best_{cfg.best_metric} {float(best_score) if best_score is not None else float('nan'):.4f} | "
                f"final_loss {float(final_metrics['loss']):.4f}"
            )
        return {
            "train_history": train_history,
            "eval_history": eval_history,
            "final_metrics": final_metrics,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score) if best_score is not None else float("nan"),
            "final_rollout": final_rollout,
        }
