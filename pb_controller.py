"""
Performance Boosting (PB) controller core.

- Causal, step-by-step operation using (B, T, N) tensors (T=1 typical).
- Reconstructs disturbance w from nominal model and last state/control.
- Modular plant and operator interfaces.
- Supports multi-input operators and factorization: M(w,z) = M_p(w) X M_b(w,z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import torch
import torch.nn as nn


def as_bt(x: torch.Tensor) -> torch.Tensor:
    """Ensure input is shaped (B, T, N). Accepts (B, N) or (B, T, N)."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected tensor with 2 or 3 dims, got shape {tuple(x.shape)}")


def strip_t(x: torch.Tensor) -> torch.Tensor:
    """If tensor is (B, 1, N), return (B, N). Otherwise return as-is."""
    if x.dim() == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    return x


class NominalPlant(Protocol):
    """Protocol for nominal plant dynamics used to reconstruct disturbance w."""

    def nominal_dynamics(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        """Return x_next_nominal given x and u. Shapes (B, T, Nx) -> (B, T, Nx)."""
        ...


class OperatorBase(nn.Module):
    """Base class for PB operators. Accepts w and optional z."""

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def reset(self) -> None:
        """Optional: reset internal dynamical state."""
        return None


class GenericOperator(OperatorBase):
    """Wraps an arbitrary module. If z is provided, concatenates [w,z] by default."""

    def __init__(self, module: nn.Module, concat_z: bool = True):
        super().__init__()
        self.module = module
        self.concat_z = concat_z

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is None or not self.concat_z:
            return self.module(w)
        return self.module(torch.cat([w, z], dim=-1))

    def reset(self) -> None:
        if hasattr(self.module, "reset"):
            self.module.reset()


def boxtimes_timewise(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Timewise matrix–vector product.

    Preferred shapes:
      A: (B, T, r, s)
      v: (B, T, s)
      y: (B, T, r)

    Also accepts:
      A: (T, B, r, s)
      v: (T, B, s)
      y: (T, B, r)
    """
    if A.ndim != 4:
        raise ValueError(f"A must be 4D (B,T,r,s) or (T,B,r,s). Got shape {tuple(A.shape)}")
    if v.ndim != 3:
        raise ValueError(f"v must be 3D (B,T,s) or (T,B,s). Got shape {tuple(v.shape)}")
    if A.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"Leading dims of A and v must match. Got A{tuple(A.shape)} vs v{tuple(v.shape)}"
        )
    if A.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"Incompatible inner dims: A.shape[-1]={A.shape[-1]} vs v.shape[-1]={v.shape[-1]}"
        )

    # Performs y_t = A_t v_t for each leading (batch,time) index.
    return torch.einsum("btrs,bts->btr", A, v)


class TimewiseMatVec(nn.Module):
    """nn.Module wrapper for the timewise matrix–vector product."""

    def forward(self, A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return boxtimes_timewise(A, v)


class FactorizedOperator(OperatorBase):
    """M(w,z) = M_p(w) X M_b(w,z) with configurable product X."""

    def __init__(
        self,
        mp: nn.Module,
        mb: nn.Module,
        mp_context_lifter: Optional[nn.Module] = None,
        product_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = boxtimes_timewise,
    ):
        super().__init__()
        self.mp = mp
        self.mb = mb
        self.mp_context_lifter = mp_context_lifter
        self.product_fn = product_fn

    def reset(self) -> None:
        if hasattr(self.mp, "reset"):
            self.mp.reset()
        if hasattr(self.mb, "reset"):
            self.mb.reset()
        if self.mp_context_lifter is not None and hasattr(self.mp_context_lifter, "reset"):
            self.mp_context_lifter.reset()

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mp_context_lifter is None:
            w_mp = w
        else:
            if z is None:
                raise ValueError("z must be provided when mp_context_lifter is enabled.")
            w_bt = as_bt(w)
            z_lift = self.mp_context_lifter(z)
            z_lift_bt = as_bt(z_lift)
            if w_bt.shape[:2] != z_lift_bt.shape[:2]:
                raise ValueError(
                    "w and lifted z must share leading dims. "
                    f"Got w{tuple(w_bt.shape)} vs z_lift{tuple(z_lift_bt.shape)}"
                )
            w_mp = torch.cat([w_bt, z_lift_bt], dim=-1)
        v = self.mp(w_mp)
        A = self.mb(w, z) if z is not None else self.mb(w)
        return self.product_fn(A, v)


@dataclass
class PBState:
    x_tm1: torch.Tensor
    u_tm1: torch.Tensor
    w0: torch.Tensor
    has_prev: bool


class PBController(nn.Module):
    """
    Performance Boosting controller.

    At time t, the controller receives x_t and optional z_t.
    It reconstructs w_t = x_t - f_nom(x_{t-1}, u_{t-1}).
    Then u_boost_t = M(w_t, z_t) (causal).
    Optionally adds a nominal policy u_nom(x_t).

    Shapes are (B, T, N), with T=1 in step-by-step usage.
    """

    def __init__(
        self,
        plant: NominalPlant,
        operator: OperatorBase,
        u_nominal: Optional[Callable[[torch.Tensor, Optional[int]], torch.Tensor]] = None,
        u_dim: Optional[int] = None,
        detach_state: bool = True,
    ):
        super().__init__()
        self.plant = plant
        self.operator = operator
        self.u_nominal = u_nominal
        self.u_dim = u_dim
        self.detach_state = detach_state
        self.state: Optional[PBState] = None

    def reset(
        self,
        x_init: torch.Tensor,
        u_init: Optional[torch.Tensor] = None,
        w0: Optional[torch.Tensor] = None,
    ) -> None:
        x_init = as_bt(x_init)
        if u_init is None:
            if self.u_dim is None:
                raise ValueError("u_init is None and u_dim is not set.")
            u_init = torch.zeros(x_init.shape[0], x_init.shape[1], self.u_dim, device=x_init.device)
        u_init = as_bt(u_init)
        if w0 is None:
            w0 = torch.zeros_like(x_init)
        else:
            w0 = as_bt(w0)
            if w0.shape != x_init.shape:
                raise ValueError(
                    f"w0 must match x_init shape. Got w0{tuple(w0.shape)} vs x_init{tuple(x_init.shape)}"
                )
        self.state = PBState(x_tm1=x_init, u_tm1=u_init, w0=w0, has_prev=False)
        self.operator.reset()

    def _compute_w_t(self, x_t: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        assert self.state is not None
        if not self.state.has_prev:
            return self.state.w0
        x_tm1 = self.state.x_tm1
        u_tm1 = self.state.u_tm1
        # Reconstruct disturbance for transition (t-1 -> t):
        #   x_t = f_{t-1}(x_{t-1}, u_{t-1}) + w_t
        t_prev = None if t is None else (t - 1)
        x_nom = self.plant.nominal_dynamics(x_tm1, u_tm1, t_prev)
        return x_t - x_nom

    def forward_step(
        self,
        x_t: torch.Tensor,
        z_t: Optional[torch.Tensor] = None,
        t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute u_t and return (u_t, w_t)."""
        if self.state is None:
            raise RuntimeError("Call reset(x_init) before forward_step().")

        x_t = as_bt(x_t)
        z_t = as_bt(z_t) if z_t is not None else None

        w_t = self._compute_w_t(x_t, t)
        u_boost = self.operator(w_t, z_t)

        if self.u_nominal is None:
            u_nom = torch.zeros_like(u_boost)
        else:
            u_nom = self.u_nominal(x_t, t)
            u_nom = as_bt(u_nom)

        u_t = u_nom + u_boost

        if self.detach_state:
            self.state.x_tm1 = x_t.detach()
            self.state.u_tm1 = u_t.detach()
        else:
            self.state.x_tm1 = x_t
            self.state.u_tm1 = u_t
        self.state.has_prev = True

        return u_t, w_t

    def set_last_applied_control(self, u_applied: torch.Tensor) -> None:
        """Update internal memory with the control actually applied to the true plant."""
        if self.state is None:
            raise RuntimeError("Call reset(x_init) before set_last_applied_control().")

        u_applied = as_bt(u_applied)
        if u_applied.shape != self.state.u_tm1.shape:
            raise ValueError(
                f"u_applied shape mismatch: got {tuple(u_applied.shape)} vs expected {tuple(self.state.u_tm1.shape)}"
            )

        self.state.u_tm1 = u_applied.detach() if self.detach_state else u_applied

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        z_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence causally (B, T, Nx). Returns u_seq and w_seq."""
        if self.state is None:
            raise RuntimeError("Call reset(x_init) before forward_sequence().")

        x_seq = as_bt(x_seq)
        z_seq = as_bt(z_seq) if z_seq is not None else None

        _, T, _ = x_seq.shape
        u_out = []
        w_out = []
        for t in range(T):
            z_t = z_seq[:, t:t + 1, :] if z_seq is not None else None
            u_t, w_t = self.forward_step(x_seq[:, t:t + 1, :], z_t, t)
            u_out.append(u_t)
            w_out.append(w_t)
        return torch.cat(u_out, dim=1), torch.cat(w_out, dim=1)


# -------------------------------
# Minimal example (optional)
# -------------------------------

class LinearPlant:
    """Simple linear nominal plant: x_{t+1} = A x_t + B u_t"""

    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A
        self.B = B

    def nominal_dynamics(self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        x = as_bt(x)
        u = as_bt(u)
        return torch.einsum("ij,btj->bti", self.A, x) + torch.einsum("ij,btj->bti", self.B, u)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 32, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset(self) -> None:
        return None


class MbMLP(nn.Module):
    """Outputs A_t with shape (B,T,r,s)."""

    def __init__(self, in_dim: int, r: int, s: int, hidden_dim: int = 32):
        super().__init__()
        self.r = r
        self.s = s
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * s),
        )

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            x = torch.cat([w, z], dim=-1)
        else:
            x = w
        out = self.net(x)
        return out.view(out.shape[0], out.shape[1], self.r, self.s)

    def reset(self) -> None:
        return None


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 1
    Nx, Nu, Nz = 4, 2, 3

    A = torch.eye(Nx)
    Bmat = torch.randn(Nx, Nu) * 0.1
    plant = LinearPlant(A, Bmat)

    mp = MLP(Nx, 8)
    mb = MbMLP(Nx + Nz, r=Nu, s=8)
    op = FactorizedOperator(mp, mb)

    def u_nominal(x, t=None):
        return torch.zeros(x.shape[0], x.shape[1], Nu, device=x.device)

    controller = PBController(plant, op, u_nominal=u_nominal, u_dim=Nu)

    x0 = torch.zeros(B, T, Nx)
    controller.reset(x0, u_init=torch.zeros(B, T, Nu), w0=x0)

    x1 = torch.randn(B, T, Nx) * 0.1
    z1 = torch.randn(B, T, Nz)
    u1, w0 = controller.forward_step(x1, z1, t=1)

    print("u1 shape:", tuple(u1.shape))
    print("w0 shape:", tuple(w0.shape))
