"""Adapters for using neural_ssm modules as PB operators."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pb_core import as_bt, OperatorBase


class MpDeepSSM(nn.Module):
    """
    Adapter for M_p(w) implemented with neural_ssm.DeepSSM.

    Input:
      w: (B, T, w_dim) or (B, w_dim)
    Output:
      v: (B, T, s_dim)
    """

    def __init__(
        self,
        w_dim: int,
        s_dim: int,
        *,
        mode: str = "loop",
        reset_state_each_call: bool = False,
        detach_state: bool = False,
        **ssm_kwargs: Any,
    ):
        super().__init__()
        try:
            from neural_ssm import DeepSSM
        except ImportError as exc:
            raise ImportError(
                "Could not import neural_ssm.DeepSSM. Install neural_ssm in this environment."
            ) from exc

        self.core = DeepSSM(w_dim, s_dim, **ssm_kwargs)
        self.mode = mode
        self.reset_state_each_call = reset_state_each_call
        self.detach_state = detach_state

    def reset(self) -> None:
        if hasattr(self.core, "reset"):
            self.core.reset()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w = as_bt(w)
        out = self.core(
            w,
            mode=self.mode,
            reset_state=self.reset_state_each_call,
            detach_state=self.detach_state,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = as_bt(out)
        return out


class ContextRescale(nn.Module):
    """Stateless scalar rescale of the context (streaming-safe, no parameters).

    Pass as ContextualDeepSSM's ``context_encoder`` so the rescale applies
    before every context port (input/gate/mixer/select) uniformly.
    """

    def __init__(self, factor: float):
        super().__init__()
        self.factor = float(factor)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.factor


class MpContextualSSM(OperatorBase):
    """PB operator built on ``neural_ssm.ContextualDeepSSM``.

    Maps a reconstructed disturbance ``w`` and a context signal ``z`` directly to
    a control correction ``u`` in a single operator, injecting context through any
    combination of the ports exposed by ``ContextualDeepSSM``:

      * ``"mixer"`` — a bounded time-wise matrix on the core features (a router:
        output tracks the disturbance feature and vanishes as ``w -> 0``);
      * ``"input"`` — context, ``L2``-projected by ``context_filter``, concatenated
        to ``w`` before the core (a driver: acts on context even when ``w ~ 0``);
      * ``"gate"`` — per-block sigmoid gates that attenuate the SSM/FF branches;
      * ``"select"`` — context conditions a selective core's per-step dynamics.

    Designed for the causal, step-by-step (T=1) PB rollout: it keeps the SSM
    recurrent state between steps (cleared by :meth:`reset`) and threads an
    absolute ``time_offset`` so time-windowed context filters (``taper`` /
    ``finite_horizon`` / ``exponential`` / ``polynomial``) stay correct in the
    closed loop.  (The ``difference`` and ``none`` filters are *not* well defined
    across single-step chunks; prefer a windowed filter for closed-loop use.)

    Input:
      w: (B, T, w_dim) or (B, w_dim)         — reconstructed disturbance
      z: (B, T, z_dim) or (B, z_dim) or None — context
    Output:
      u: (B, T, u_dim)
    """

    def __init__(
        self,
        w_dim: int,
        z_dim: int,
        u_dim: int,
        *,
        context_modes: Any = ("mixer", "input", "gate"),
        mode: str = "loop",
        detach_state: bool = False,
        w_augmenter: nn.Module | None = None,
        **contextual_kwargs: Any,
    ):
        super().__init__()
        try:
            from neural_ssm import ContextualDeepSSM
        except ImportError as exc:
            raise ImportError(
                "Could not import neural_ssm.ContextualDeepSSM. Install/upgrade "
                "neural_ssm (>= 0.4, the Clean_SSM source) in this environment."
            ) from exc

        self.w_augmenter = w_augmenter
        core_in_dim = int(w_dim) * 2 if w_augmenter is not None else int(w_dim)
        self.core = ContextualDeepSSM(
            core_in_dim,
            int(z_dim),
            int(u_dim),
            context_modes=context_modes,
            **contextual_kwargs,
        )
        self.mode = mode
        self.detach_state = detach_state
        self._t = 0

    def reset(self) -> None:
        if self.w_augmenter is not None and hasattr(self.w_augmenter, "reset"):
            self.w_augmenter.reset()
        self.core.reset()
        self._t = 0

    def forward(self, w: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        w = as_bt(w)
        if self.w_augmenter is not None:
            w = self.w_augmenter(w)
        z = as_bt(z) if z is not None else None
        out, _ = self.core(
            w,
            z,
            mode=self.mode,
            reset_state=False,
            detach_state=self.detach_state,
            time_offset=self._t,
        )
        out = as_bt(out)
        self._t += int(w.shape[1])
        return out

    @torch.no_grad()
    def gain_diagnostics(self) -> dict:
        return self.core.gain_diagnostics()
