"""
Ultra-minimal PB example: step by step.

What this script does:
1) Define the two factors M_p and M_b for M(w,z) = M_b(w,z) x M_p(w)
2) Build a PB controller using a nominal plant
3) Define the true plant equal to the nominal one
4) Run a rollout example and print key outputs
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from nav_plants import DoubleIntegratorNominal, DoubleIntegratorTrue
from pb_controller import as_bt
from pb_core.factories import build_factorized_controller
from pb_core.rollout import rollout_pb


# -----------------------------
# 1) Define M_p and M_b factors
# -----------------------------


class MinimalMp(nn.Module):
    """
    Disturbance processor M_p.

    Role in PB:
      Receives reconstructed disturbance w_t and maps it to latent features v_t.
      In the factorization, this is the "dynamic" branch that processes disturbance.

    Signature:
      input  w: (B,T,Nw)
      output v: (B,T,s)
    """

    def __init__(self, w_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w_dim, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, feat_dim, bias=False),
        )

    def reset(self) -> None:
        # Stateless module, reset is a no-op.
        return None

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # as_bt allows both (B,Nw) and (B,T,Nw), returning (B,T,Nw).
        w = as_bt(w)
        return self.net(w)


class MinimalMb(nn.Module):
    """
    Bounded context mixer M_b.

    Role in PB:
      Builds a time-varying mixing matrix A_t from disturbance+context.
      A_t maps feature space (size s) to control space (size Nu).

    Signature:
      input  w: (B,T,Nw), z: (B,T,Nz)
      output A: (B,T,Nu,s)
    """

    def __init__(self, w_dim: int, z_dim: int, u_dim: int, feat_dim: int, bound: float = 0.2):
        super().__init__()
        self.w_dim = int(w_dim)
        self.z_dim = int(z_dim)
        self.u_dim = int(u_dim)
        self.feat_dim = int(feat_dim)
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.Linear(self.w_dim + self.z_dim, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, self.u_dim * self.feat_dim, bias=False),
        )

    def reset(self) -> None:
        # Stateless module, reset is a no-op.
        return None

    def forward(self, w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Normalize shapes to (B,T,*).
        w = as_bt(w)
        z = as_bt(z)
        # Basic interface checks to catch wrong wiring early.
        if w.shape[:2] != z.shape[:2]:
            raise ValueError(f"Leading dims mismatch: w{tuple(w.shape)} vs z{tuple(z.shape)}")
        if w.shape[-1] != self.w_dim:
            raise ValueError(f"Expected w dim {self.w_dim}, got {w.shape[-1]}")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"Expected z dim {self.z_dim}, got {z.shape[-1]}")

        bsz, t_steps, _ = w.shape
        # Build one matrix A_t per sample and per timestep.
        inp = torch.cat([w, z], dim=-1).reshape(bsz * t_steps, -1)
        out = self.net(inp)
        # Keep M_b bounded (important for stable factorized design).
        out = self.bound * torch.tanh(out / self.bound)
        return out.view(bsz, t_steps, self.u_dim, self.feat_dim)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dimensions.
    nx = 4  # state [px, py, vx, vy]
    nu = 2  # control [ux, uy]
    nz = 2  # context (example: goal position)
    feat_dim = 12

    # Build the two factors M_p and M_b.
    mp = MinimalMp(w_dim=nx, feat_dim=feat_dim).to(device)
    mb = MinimalMb(w_dim=nx, z_dim=nz, u_dim=nu, feat_dim=feat_dim, bound=0.15).to(device)

    # --------------------------------------------
    # 2) Nominal plant and PB controller (nominal)
    # --------------------------------------------
    plant_nom = DoubleIntegratorNominal(dt=0.05, pre_kp=1.0, pre_kd=1.5)
    # Use the shared PB factory:
    #   M(w,z) = M_b(w,z) x M_p(w)
    controller = build_factorized_controller(
        nominal_plant=plant_nom,
        mp=mp,
        mb=mb,
        u_dim=nu,
        u_nominal=None,
        detach_state=False,
    ).to(device)

    # ------------------------------------------
    # 3) True plant = nominal plant (same model)
    # ------------------------------------------
    plant_true = DoubleIntegratorTrue(dt=0.05, pre_kp=1.0, pre_kd=1.5)

    # -------------------------------------------------
    # 4) Minimal toy training loop
    # -------------------------------------------------
    train_steps = 200
    train_batch = 128
    train_horizon = 50
    optimizer = optim.Adam(controller.parameters(), lr=2e-3)

    for step in range(1, train_steps + 1):
        # Sample random starts.
        x0_pos = torch.empty(train_batch, 2, device=device)
        x0_pos[:, 0].uniform_(1.2, 2.0)
        x0_pos[:, 1].uniform_(-0.8, 0.8)
        x0_vel = torch.zeros(train_batch, 2, device=device)
        x0_train = torch.cat([x0_pos, x0_vel], dim=-1).unsqueeze(1)  # (B,1,4)

        # Sample goal-context for each sample (constant across rollout).
        goals_train = torch.empty(train_batch, 2, device=device)
        goals_train[:, 0].uniform_(-0.2, 0.2)
        goals_train[:, 1].uniform_(-0.2, 0.2)
        z_train = goals_train.unsqueeze(1).expand(-1, train_horizon, -1).contiguous()  # (B,T,2)

        out = rollout_pb(
            controller=controller,
            plant_true=plant_true,
            x0=x0_train,
            horizon=train_horizon,
            z_seq=z_train,
            w0=x0_train,
            process_noise_seq=None,
        )
        x_seq_train, u_seq_train = out.x_seq, out.u_seq

        # Goal-reaching toy loss: terminal + stage distance, small control effort.
        goal_seq = goals_train.unsqueeze(1)  # (B,1,2)
        dist = torch.norm(x_seq_train[..., :2] - goal_seq, dim=-1)
        loss_term = dist[:, -1].mean()
        loss_stage = dist.mean()
        loss_u = (u_seq_train ** 2).mean()
        loss = 20.0 * loss_term + 2.0 * loss_stage + 0.01 * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(
                f"train step {step:03d}/{train_steps} | "
                f"loss {loss.item():.3f} | "
                f"term {loss_term.item():.3f} | "
                f"stage {loss_stage.item():.3f}"
            )

    # -----------------------------
    # 5) Example rollout after toy training
    # -----------------------------
    bsz = 3
    horizon = 60

    # Initial state x0: nonzero positions, zero velocity.
    pos0 = torch.tensor([[1.8, 0.4], [1.6, -0.5], [1.4, 0.0]], dtype=torch.float32, device=device)
    vel0 = torch.zeros(bsz, 2, dtype=torch.float32, device=device)
    x0 = torch.cat([pos0, vel0], dim=-1).unsqueeze(1)  # (B,1,Nx)

    # Context z: here we use a constant goal-like signal per sample.
    goals = torch.tensor([[0.0, 0.0], [0.2, -0.1], [-0.1, 0.1]], dtype=torch.float32, device=device)
    z_seq = goals.unsqueeze(1).expand(-1, horizon, -1).contiguous()  # (B,T,Nz)

    # PB convention:
    # - w0 is the initial disturbance proxy (often x0).
    # - then w_t is reconstructed online by PB from model mismatch.
    rollout = rollout_pb(
        controller=controller,
        plant_true=plant_true,
        x0=x0,
        horizon=horizon,
        z_seq=z_seq,
        w0=x0,
        process_noise_seq=None,
    )

    x_seq, u_seq, w_seq = rollout.x_seq, rollout.u_seq, rollout.w_seq
    print(f"x_seq shape: {tuple(x_seq.shape)}")
    print(f"u_seq shape: {tuple(u_seq.shape)}")
    print(f"w_seq shape: {tuple(w_seq.shape)}")
    print("final positions:")
    print(x_seq[:, -1, :2])


if __name__ == "__main__":
    main()
