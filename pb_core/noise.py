"""Reusable process-noise models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ZeroNoise:
    """No process noise."""

    def sample(
        self,
        *,
        bsz: int,
        horizon: int,
        nx: int,
        device: torch.device,
        seed: int | None = None,
    ) -> torch.Tensor | None:
        return None


@dataclass
class DecayingGaussianNoise:
    """
    Decaying Gaussian process noise:
      eta_t ~ N(0, sigma_t^2 I), sigma_t = sigma0 * exp(-t/tau)
    """

    sigma0: float
    tau: float

    def sample(
        self,
        *,
        bsz: int,
        horizon: int,
        nx: int,
        device: torch.device,
        seed: int | None = None,
    ) -> torch.Tensor | None:
        if self.sigma0 <= 0:
            return None
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        t = torch.arange(horizon, device=device).view(1, horizon, 1)
        sigma_t = float(self.sigma0) * torch.exp(-t / float(self.tau))
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        return torch.randn(bsz, horizon, nx, device=device, generator=gen) * sigma_t
