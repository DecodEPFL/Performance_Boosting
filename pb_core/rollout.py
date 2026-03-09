"""Generic PB rollout engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from pb_controller import PBController, as_bt


@dataclass
class RolloutResult:
    x_seq: torch.Tensor
    u_seq: torch.Tensor
    w_seq: torch.Tensor


def rollout_pb(
    *,
    controller: PBController,
    plant_true,
    x0: torch.Tensor,
    horizon: int,
    z_seq: torch.Tensor | None = None,
    context_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    w0: torch.Tensor | None = None,
    process_noise_seq: torch.Tensor | None = None,
    u_post_fn: Callable[..., torch.Tensor] | None = None,
) -> RolloutResult:
    """
    Generic causal rollout.

    Provide either:
      - z_seq: (B,T,Nz), or
      - context_fn(x_t, t): (B,1,Nz)
    """
    if (z_seq is None) == (context_fn is None):
        raise ValueError("Provide exactly one of z_seq or context_fn.")

    x = as_bt(x0)
    if z_seq is not None:
        z_seq = as_bt(z_seq)
        if z_seq.shape[1] != horizon:
            raise ValueError(f"z_seq horizon mismatch: got {z_seq.shape[1]}, expected {horizon}")

    if process_noise_seq is not None:
        process_noise_seq = as_bt(process_noise_seq)
        if process_noise_seq.shape[0] != x.shape[0] or process_noise_seq.shape[1] != horizon:
            raise ValueError(
                "process_noise_seq must have shape (B,horizon,Nx). "
                f"Got {tuple(process_noise_seq.shape)} expected ({x.shape[0]},{horizon},{x.shape[-1]})"
            )
        if process_noise_seq.shape[-1] != x.shape[-1]:
            raise ValueError(
                "process_noise_seq state dim mismatch. "
                f"Got {process_noise_seq.shape[-1]} expected {x.shape[-1]}"
            )

    controller.reset(x, u_init=None, w0=w0)
    x_log: list[torch.Tensor] = []
    u_log: list[torch.Tensor] = []
    w_log: list[torch.Tensor] = []

    for t in range(horizon):
        z_t = z_seq[:, t : t + 1, :] if z_seq is not None else context_fn(x, t)
        u_t, w_t = controller.forward_step(x, z_t, t=t)
        if u_post_fn is not None:
            try:
                u_applied = u_post_fn(x, u_t, t)
            except TypeError:
                u_applied = u_post_fn(u_t)
            controller.set_last_applied_control(u_applied)
            u_t = u_applied
        x = plant_true.forward(x, u_t, t=t)
        if process_noise_seq is not None:
            x = x + process_noise_seq[:, t : t + 1, :]
        x_log.append(x)
        u_log.append(u_t)
        w_log.append(w_t)

    return RolloutResult(
        x_seq=torch.cat(x_log, dim=1),
        u_seq=torch.cat(u_log, dim=1),
        w_seq=torch.cat(w_log, dim=1),
    )
