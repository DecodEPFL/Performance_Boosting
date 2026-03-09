"""Core interfaces for reusable PB experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import torch


@dataclass
class BatchData:
    """Generic batch container passed across dataset/context/loss/metrics."""

    payload: Any


class NominalPlant(Protocol):
    """Plant model used inside PB for disturbance reconstruction."""

    def nominal_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, t: int | None = None
    ) -> torch.Tensor:
        ...


class TruePlant(Protocol):
    """True plant used for rollout."""

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: int | None = None) -> torch.Tensor:
        ...


class ContextBuilder(Protocol):
    """Build context sequence z from rollout state and batch metadata."""

    def build(self, x: torch.Tensor, batch: BatchData) -> torch.Tensor:
        """
        Args:
            x: state sequence (B,T,Nx) or current state (B,1,Nx)
            batch: batch metadata
        Returns:
            z: context (B,T,Nz) or (B,1,Nz)
        """
        ...


class NoiseModel(Protocol):
    """Produce process-noise sequences for rollouts."""

    def sample(
        self,
        *,
        bsz: int,
        horizon: int,
        nx: int,
        device: torch.device,
        seed: int | None = None,
    ) -> torch.Tensor | None:
        ...


class DatasetProvider(Protocol):
    """Task-level batch provider and state initializer."""

    def sample_train(self, *, epoch: int, batch_size: int, seed: int) -> BatchData:
        ...

    def sample_val(self, *, batch_size: int, seed: int) -> BatchData:
        ...

    def make_x0(self, batch: BatchData, *, device: torch.device) -> torch.Tensor:
        ...


class LossFn(Protocol):
    """Task loss over rollout traces."""

    def __call__(
        self,
        *,
        x_seq: torch.Tensor,
        u_seq: torch.Tensor,
        w_seq: torch.Tensor,
        batch: BatchData,
    ) -> Tuple[torch.Tensor, dict]:
        ...


class MetricsFn(Protocol):
    """Task metrics over rollout traces."""

    def __call__(
        self,
        *,
        x_seq: torch.Tensor,
        u_seq: torch.Tensor,
        w_seq: torch.Tensor,
        batch: BatchData,
    ) -> dict:
        ...
