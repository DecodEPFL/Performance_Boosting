"""
Contextual two-gate navigation environment.

The robot starts on the right and must reach the origin while avoiding three circles:
  - top obstacle
  - center obstacle
  - bottom obstacle

Top and bottom obstacle radii are context-dependent and determine which corridor
(top or bottom) is easier/safe. This is designed to highlight contextual control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class GateScenario:
    start: torch.Tensor    # (B, 2)
    goal: torch.Tensor     # (B, 2)
    centers: torch.Tensor  # (B, 3, 2) [top, center, bottom]
    radii: torch.Tensor    # (B, 3)    [r_top, r_center, r_bottom]
    mode: torch.Tensor     # (B,)      0=top-open, 1=bottom-open, 2=ambiguous


def fixed_gate_centers(
    center_x: float = 0.9,
    y_sep: float = 0.9,
) -> torch.Tensor:
    return torch.tensor(
        [
            [center_x, +y_sep],  # top
            [center_x, 0.0],     # center
            [center_x, -y_sep],  # bottom
        ],
        dtype=torch.float32,
    )


def sample_gate_scenarios(
    *,
    batch_size: int,
    seed: int,
    start_x: Tuple[float, float] = (1.7, 2.4),
    start_y: Tuple[float, float] = (-0.45, 0.45),
    centers: torch.Tensor | None = None,
    r_center: float = 0.43,
    open_range: Tuple[float, float] = (0.22, 0.30),
    closed_range: Tuple[float, float] = (0.40, 0.45),
    ambiguous_frac: float = 0.05,
) -> GateScenario:
    if centers is None:
        centers = fixed_gate_centers()
    centers = centers.float()
    if centers.shape != (3, 2):
        raise ValueError(f"Expected centers shape (3,2), got {tuple(centers.shape)}")

    rng = np.random.RandomState(int(seed))

    starts = np.zeros((batch_size, 2), dtype=np.float32)
    starts[:, 0] = rng.uniform(start_x[0], start_x[1], size=batch_size)
    starts[:, 1] = rng.uniform(start_y[0], start_y[1], size=batch_size)
    goals = np.zeros((batch_size, 2), dtype=np.float32)

    modes = np.zeros((batch_size,), dtype=np.int64)
    radii = np.zeros((batch_size, 3), dtype=np.float32)

    p_amb = float(min(max(ambiguous_frac, 0.0), 1.0))
    p_main = 1.0 - p_amb
    p_top_open = 0.5 * p_main
    p_bottom_open = 0.5 * p_main

    u = rng.rand(batch_size)
    top_mask = u < p_top_open
    bottom_mask = (u >= p_top_open) & (u < p_top_open + p_bottom_open)
    amb_mask = ~(top_mask | bottom_mask)

    modes[top_mask] = 0
    modes[bottom_mask] = 1
    modes[amb_mask] = 2

    n_top = int(top_mask.sum())
    n_bottom = int(bottom_mask.sum())
    n_amb = int(amb_mask.sum())

    # top-open: top small, bottom large
    if n_top > 0:
        r_top = rng.uniform(open_range[0], open_range[1], size=n_top).astype(np.float32)
        r_bottom = rng.uniform(closed_range[0], closed_range[1], size=n_top).astype(np.float32)
        radii[top_mask, 0] = r_top
        radii[top_mask, 1] = float(r_center)
        radii[top_mask, 2] = r_bottom

    # bottom-open: bottom small, top large
    if n_bottom > 0:
        r_top = rng.uniform(closed_range[0], closed_range[1], size=n_bottom).astype(np.float32)
        r_bottom = rng.uniform(open_range[0], open_range[1], size=n_bottom).astype(np.float32)
        radii[bottom_mask, 0] = r_top
        radii[bottom_mask, 1] = float(r_center)
        radii[bottom_mask, 2] = r_bottom

    # ambiguous: both middle range
    if n_amb > 0:
        mid_min = min(open_range[1], closed_range[0])
        mid_max = max(open_range[1], closed_range[0])
        r_top = rng.uniform(mid_min, mid_max, size=n_amb).astype(np.float32)
        r_bottom = rng.uniform(mid_min, mid_max, size=n_amb).astype(np.float32)
        radii[amb_mask, 0] = r_top
        radii[amb_mask, 1] = float(r_center)
        radii[amb_mask, 2] = r_bottom

    centers_b = np.broadcast_to(centers.cpu().numpy()[None, :, :], (batch_size, 3, 2)).copy()
    return GateScenario(
        start=torch.from_numpy(starts),
        goal=torch.from_numpy(goals),
        centers=torch.from_numpy(centers_b).float(),
        radii=torch.from_numpy(radii).float(),
        mode=torch.from_numpy(modes).long(),
    )


def sample_paired_gate_scenarios(
    *,
    batch_size: int,
    seed: int,
    start_x: Tuple[float, float] = (1.7, 2.4),
    start_y: Tuple[float, float] = (-0.45, 0.45),
    centers: torch.Tensor | None = None,
    r_center: float = 0.43,
    open_range: Tuple[float, float] = (0.22, 0.30),
    closed_range: Tuple[float, float] = (0.40, 0.45),
) -> GateScenario:
    """
    Build paired contexts:
      - same start appears twice
      - one sample top-open, one sample bottom-open
    This strongly exposes contextual dependency during training.
    """
    if centers is None:
        centers = fixed_gate_centers()
    centers = centers.float()
    if centers.shape != (3, 2):
        raise ValueError(f"Expected centers shape (3,2), got {tuple(centers.shape)}")

    rng = np.random.RandomState(int(seed))
    half = int(batch_size // 2)
    rem = int(batch_size - 2 * half)

    starts_base = np.zeros((half, 2), dtype=np.float32)
    starts_base[:, 0] = rng.uniform(start_x[0], start_x[1], size=half)
    starts_base[:, 1] = rng.uniform(start_y[0], start_y[1], size=half)

    # Paired top-open and bottom-open samples.
    starts = np.concatenate([starts_base, starts_base], axis=0)
    goals = np.zeros((2 * half, 2), dtype=np.float32)
    modes = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(half, dtype=np.int64)], axis=0)
    radii = np.zeros((2 * half, 3), dtype=np.float32)

    r_open = rng.uniform(open_range[0], open_range[1], size=half).astype(np.float32)
    r_closed = rng.uniform(closed_range[0], closed_range[1], size=half).astype(np.float32)

    # first half: top-open
    radii[:half, 0] = r_open
    radii[:half, 1] = float(r_center)
    radii[:half, 2] = r_closed
    # second half: bottom-open
    radii[half:, 0] = r_closed
    radii[half:, 1] = float(r_center)
    radii[half:, 2] = r_open

    # Optional remainder sampled with standard generator (rare when batch even).
    if rem > 0:
        extra = sample_gate_scenarios(
            batch_size=rem,
            seed=seed + 12345,
            start_x=start_x,
            start_y=start_y,
            centers=centers,
            r_center=r_center,
            open_range=open_range,
            closed_range=closed_range,
            ambiguous_frac=0.0,
        )
        starts = np.concatenate([starts, extra.start.numpy()], axis=0)
        goals = np.concatenate([goals, extra.goal.numpy()], axis=0)
        radii = np.concatenate([radii, extra.radii.numpy()], axis=0)
        modes = np.concatenate([modes, extra.mode.numpy()], axis=0)

    n = starts.shape[0]
    perm = rng.permutation(n)
    starts = starts[perm]
    goals = goals[perm]
    radii = radii[perm]
    modes = modes[perm]

    centers_b = np.broadcast_to(centers.cpu().numpy()[None, :, :], (n, 3, 2)).copy()
    return GateScenario(
        start=torch.from_numpy(starts),
        goal=torch.from_numpy(goals),
        centers=torch.from_numpy(centers_b).float(),
        radii=torch.from_numpy(radii).float(),
        mode=torch.from_numpy(modes).long(),
    )


def mirror_vertical_scenario(scenario: GateScenario, mask: torch.Tensor) -> GateScenario:
    """
    Apply y -> -y mirror on selected samples and swap top/bottom obstacle slots.
    Also swaps mode labels 0<->1.
    """
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if mask.shape[0] != scenario.start.shape[0]:
        raise ValueError(f"mask length mismatch: {mask.shape[0]} vs batch {scenario.start.shape[0]}")

    start = scenario.start.clone()
    goal = scenario.goal.clone()
    centers = scenario.centers.clone()
    radii = scenario.radii.clone()
    mode = scenario.mode.clone()

    if bool(mask.any().item()):
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        start[idx, 1] = -start[idx, 1]
        goal[idx, 1] = -goal[idx, 1]
        centers[idx, :, 1] = -centers[idx, :, 1]

        # swap top/bottom obstacle channels
        top = centers[idx, 0, :].clone()
        bottom = centers[idx, 2, :].clone()
        centers[idx, 0, :] = bottom
        centers[idx, 2, :] = top

        r_top = radii[idx, 0].clone()
        r_bottom = radii[idx, 2].clone()
        radii[idx, 0] = r_bottom
        radii[idx, 2] = r_top

        m = mode[idx].clone()
        top_open = m == 0
        bottom_open = m == 1
        m[top_open] = 1
        m[bottom_open] = 0
        mode[idx] = m

    return GateScenario(start=start, goal=goal, centers=centers, radii=radii, mode=mode)


def scenario_to_device(scenario: GateScenario, device: torch.device) -> GateScenario:
    return GateScenario(
        start=scenario.start.to(device),
        goal=scenario.goal.to(device),
        centers=scenario.centers.to(device),
        radii=scenario.radii.to(device),
        mode=scenario.mode.to(device),
    )


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected (B,N) or (B,T,N), got {tuple(x.shape)}")


def gate_gaps(scenario: GateScenario) -> torch.Tensor:
    centers = scenario.centers
    radii = scenario.radii
    top = centers[:, 0, :]
    mid = centers[:, 1, :]
    bottom = centers[:, 2, :]
    d_top = torch.norm(top - mid, dim=-1)
    d_bottom = torch.norm(bottom - mid, dim=-1)
    gap_top = d_top - (radii[:, 0] + radii[:, 1])
    gap_bottom = d_bottom - (radii[:, 2] + radii[:, 1])
    return torch.stack([gap_top, gap_bottom], dim=-1)


def build_gate_context(x: torch.Tensor, scenario: GateScenario) -> torch.Tensor:
    """
    Returns z_t with shape (B, T, Nz):
      [goal_dir(2), rel_to_obs(3*2), dist_to_edge(3), radii(3), gate_gaps(2)]
    """
    x = _as_bt(x)
    pos = x[..., :2]
    bsz, t_steps, _ = pos.shape

    centers = scenario.centers.to(x.device).unsqueeze(1)  # (B,1,3,2)
    radii = scenario.radii.to(x.device).unsqueeze(1)      # (B,1,3)
    goal = scenario.goal.to(x.device).unsqueeze(1)        # (B,1,2)

    rel = centers - pos.unsqueeze(2)                      # (B,T,3,2)
    dist_edge = torch.norm(rel, dim=-1) - radii           # (B,T,3)
    goal_dir = goal - pos                                 # (B,T,2)
    radii_t = radii.expand(-1, t_steps, -1)               # (B,T,3)
    gaps = gate_gaps(scenario).to(x.device).unsqueeze(1).expand(-1, t_steps, -1)  # (B,T,2)

    return torch.cat(
        [
            goal_dir,
            rel.reshape(bsz, t_steps, -1),
            dist_edge,
            radii_t,
            gaps,
        ],
        dim=-1,
    )


def obstacle_edge_distances(x: torch.Tensor, scenario: GateScenario) -> torch.Tensor:
    x = _as_bt(x)
    pos = x[..., :2]
    centers = scenario.centers.to(x.device).unsqueeze(1)  # (B,1,3,2)
    radii = scenario.radii.to(x.device).unsqueeze(1)      # (B,1,3)
    dist = torch.norm(pos.unsqueeze(2) - centers, dim=-1) # (B,T,3)
    return dist - radii


def min_dist_to_edge(x: torch.Tensor, scenario: GateScenario) -> torch.Tensor:
    return obstacle_edge_distances(x, scenario).min(dim=-1).values


def exp_barrier_penalty(
    x: torch.Tensor,
    scenario: GateScenario,
    *,
    margin: float = 0.12,
    alpha: float = 18.0,
    cap: float = 200.0,
) -> torch.Tensor:
    """
    Bell-like steep barrier:
      exp(alpha * (margin - d_edge))
    with large-value cap for numerical robustness.
    Returns (B, T, 1).
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")

    d = obstacle_edge_distances(x, scenario)
    arg = alpha * (float(margin) - d)
    arg = torch.clamp(arg, max=50.0)  # avoid exp overflow
    pen = torch.exp(arg)
    pen = torch.clamp(pen, max=float(cap))
    return pen.sum(dim=-1, keepdim=True)


def wall_barrier_penalty(
    x: torch.Tensor,
    *,
    y_limit: float = 1.25,
    margin: float = 0.10,
    alpha: float = 16.0,
    cap: float = 200.0,
) -> torch.Tensor:
    """
    Soft wall barriers at y = +/- y_limit to discourage huge vertical detours.
    Returns (B, T, 1).
    """
    x = _as_bt(x)
    if y_limit <= 0:
        raise ValueError(f"y_limit must be > 0, got {y_limit}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")

    y = x[..., 1]
    d_wall = float(y_limit) - torch.abs(y)
    arg = float(alpha) * (float(margin) - d_wall)
    arg = torch.clamp(arg, max=50.0)
    pen = torch.exp(arg)
    pen = torch.clamp(pen, max=float(cap))
    return pen.unsqueeze(-1)


def gate_choice_from_trajectory(
    x_seq: torch.Tensor,
    gate_x: float = 0.9,
    x_low: float = 0.45,
    x_high: float = 1.35,
) -> torch.Tensor:
    """
    Infer chosen corridor (top/bottom) from trajectory in the obstacle region.

    Strategy:
      1) In x-window [x_low, x_high], choose the point with max |y|.
      2) Use sign(y) there as corridor label.
      3) Fallback to nearest point to gate_x if the trajectory never enters window.
    """
    x_seq = _as_bt(x_seq)
    pos = x_seq[..., :2]
    xs = pos[..., 0]  # (B,T)
    ys = pos[..., 1]  # (B,T)
    bsz = xs.shape[0]

    chosen = torch.zeros(bsz, dtype=torch.bool, device=xs.device)
    for b in range(bsz):
        mask = (xs[b] >= float(x_low)) & (xs[b] <= float(x_high))
        if bool(mask.any().item()):
            y_win = ys[b][mask]
            k = torch.argmax(torch.abs(y_win))
            y_pick = y_win[k]
        else:
            j = torch.argmin(torch.abs(xs[b] - float(gate_x)))
            y_pick = ys[b, j]
        chosen[b] = y_pick >= 0.0
    return chosen
