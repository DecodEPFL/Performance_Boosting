"""User-friendly factories for PB controller assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from pb_controller import FactorizedOperator, PBController


@dataclass
class FactorizedBuildSpec:
    w_dim: int
    z_dim: int
    u_dim: int
    feat_dim: int
    detach_state: bool = False


def build_factorized_controller(
    *,
    nominal_plant,
    mp,
    mb,
    u_dim: int,
    detach_state: bool = False,
    u_nominal: Callable | None = None,
    mp_context_lifter=None,
) -> PBController:
    """
    Assemble PBController from nominal plant + factorized operators.

    This is a thin convenience wrapper with explicit defaults.
    """
    op = FactorizedOperator(mp=mp, mb=mb, mp_context_lifter=mp_context_lifter)
    return PBController(
        plant=nominal_plant,
        operator=op,
        u_nominal=u_nominal,
        u_dim=int(u_dim),
        detach_state=bool(detach_state),
    )


def infer_dims_from_probe(
    *,
    controller: PBController,
    x_probe: torch.Tensor,
    z_probe: torch.Tensor,
) -> dict[str, int]:
    """
    Infer basic dims by running a one-step probe.
    """
    x_probe = x_probe if x_probe.dim() == 3 else x_probe.unsqueeze(1)
    z_probe = z_probe if z_probe.dim() == 3 else z_probe.unsqueeze(1)
    controller.reset(x_probe, u_init=None, w0=x_probe)
    u0, w0 = controller.forward_step(x_probe, z_probe, t=0)
    return {
        "nx": int(x_probe.shape[-1]),
        "nz": int(z_probe.shape[-1]),
        "nu": int(u0.shape[-1]),
        "nw": int(w0.shape[-1]),
    }
