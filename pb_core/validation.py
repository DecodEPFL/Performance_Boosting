"""Compatibility checks for PB components."""

from __future__ import annotations

import torch

from pb_controller import PBController, as_bt


def validate_component_compatibility(
    *,
    controller: PBController,
    plant_true,
    x0: torch.Tensor,
    z0: torch.Tensor,
    raise_on_error: bool = True,
) -> tuple[bool, str]:
    """
    Validate basic signature and shape compatibility across controller + true plant.
    """
    try:
        x0 = as_bt(x0)
        z0 = as_bt(z0)
        if x0.shape[0] != z0.shape[0]:
            raise ValueError(f"Batch mismatch x0{tuple(x0.shape)} vs z0{tuple(z0.shape)}")
        if x0.shape[1] != 1 or z0.shape[1] != 1:
            raise ValueError(
                "Expected single-step probes with shape (B,1,N). "
                f"Got x0{tuple(x0.shape)} z0{tuple(z0.shape)}"
            )

        controller.reset(x0, u_init=None, w0=x0)
        u0, w0 = controller.forward_step(x0, z0, t=0)
        x1 = plant_true.forward(x0, u0, t=0)

        if u0.shape[:2] != x0.shape[:2]:
            raise ValueError(f"u shape mismatch u{tuple(u0.shape)} vs x0{tuple(x0.shape)}")
        if w0.shape != x0.shape:
            raise ValueError(f"w shape mismatch w{tuple(w0.shape)} vs x0{tuple(x0.shape)}")
        if x1.shape != x0.shape:
            raise ValueError(f"true plant shape mismatch x1{tuple(x1.shape)} vs x0{tuple(x0.shape)}")

        return True, "ok"
    except Exception as exc:  # pragma: no cover
        if raise_on_error:
            raise
        return False, str(exc)
