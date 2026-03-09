"""
Contextual goal navigation environment.

Same robot state can map to different desired targets based on context.
Context includes:
  - target position (goal_x, goal_y)
  - obstacle radius
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class GoalScenario:
    start: torch.Tensor        # (B,2)
    goal: torch.Tensor         # (B,2)
    centers: torch.Tensor      # (B,1,2)
    radii: torch.Tensor        # (B,1)
    theta: torch.Tensor        # (B,)


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected (B,N) or (B,T,N), got {tuple(x.shape)}")


def sample_goal_scenarios(
    *,
    batch_size: int,
    seed: int,
    start_x: Tuple[float, float] = (2.0, 2.3),
    start_y: Tuple[float, float] = (-0.12, 0.12),
    theta_range: Tuple[float, float] = (-1.0, 1.0),  # radians
    goal_x: float = -0.1,
    goal_y_scale: float = 0.85,
    center_x: float = 0.9,
    center_y: float = 0.0,
    radius_range: Tuple[float, float] = (0.35, 0.65),
) -> GoalScenario:
    rng = np.random.RandomState(int(seed))

    starts = np.zeros((batch_size, 2), dtype=np.float32)
    starts[:, 0] = rng.uniform(start_x[0], start_x[1], size=batch_size)
    starts[:, 1] = rng.uniform(start_y[0], start_y[1], size=batch_size)

    theta = rng.uniform(theta_range[0], theta_range[1], size=batch_size).astype(np.float32)
    goals = np.zeros((batch_size, 2), dtype=np.float32)
    goals[:, 0] = float(goal_x)
    goals[:, 1] = float(goal_y_scale) * np.sin(theta)

    radii = rng.uniform(radius_range[0], radius_range[1], size=(batch_size, 1)).astype(np.float32)
    centers = np.zeros((batch_size, 1, 2), dtype=np.float32)
    centers[:, 0, 0] = float(center_x)
    centers[:, 0, 1] = float(center_y)

    return GoalScenario(
        start=torch.from_numpy(starts),
        goal=torch.from_numpy(goals),
        centers=torch.from_numpy(centers),
        radii=torch.from_numpy(radii),
        theta=torch.from_numpy(theta),
    )


def scenario_to_device(scenario: GoalScenario, device: torch.device) -> GoalScenario:
    return GoalScenario(
        start=scenario.start.to(device),
        goal=scenario.goal.to(device),
        centers=scenario.centers.to(device),
        radii=scenario.radii.to(device),
        theta=scenario.theta.to(device),
    )


def shuffled_context_scenario(scenario: GoalScenario) -> GoalScenario:
    b = scenario.start.shape[0]
    perm = torch.randperm(b, device=scenario.start.device)
    return GoalScenario(
        start=scenario.start,
        goal=scenario.goal[perm],
        centers=scenario.centers,
        radii=scenario.radii[perm],
        theta=scenario.theta[perm],
    )


def build_goal_context(x: torch.Tensor, scenario: GoalScenario, *, z_gain: float = 6.0) -> torch.Tensor:
    """
    Minimal static context:
      z = [goal_x, goal_y, obstacle_radius]
    repeated over time.
    """
    x = _as_bt(x)
    bsz, t_steps = x.shape[0], x.shape[1]
    goal_t = scenario.goal.to(x.device).unsqueeze(1).expand(-1, t_steps, -1)    # (B,T,2)
    rad_t = scenario.radii.to(x.device).unsqueeze(1).expand(-1, t_steps, -1)    # (B,T,1)
    z = torch.cat([goal_t, rad_t], dim=-1)
    return float(z_gain) * z


def obstacle_edge_distances(x: torch.Tensor, scenario: GoalScenario) -> torch.Tensor:
    x = _as_bt(x)
    pos = x[..., :2]
    centers = scenario.centers.to(x.device).unsqueeze(1)  # (B,1,1,2)
    radii = scenario.radii.to(x.device).unsqueeze(1)      # (B,1,1)
    dist = torch.norm(pos.unsqueeze(2) - centers, dim=-1)
    return dist - radii


def min_dist_to_edge(x: torch.Tensor, scenario: GoalScenario) -> torch.Tensor:
    return obstacle_edge_distances(x, scenario).min(dim=-1).values


def exp_barrier_penalty(
    x: torch.Tensor,
    scenario: GoalScenario,
    *,
    margin: float = 0.14,
    alpha: float = 18.0,
    cap: float = 500.0,
) -> torch.Tensor:
    d = obstacle_edge_distances(x, scenario)
    arg = float(alpha) * (float(margin) - d)
    arg = torch.clamp(arg, max=50.0)
    pen = torch.exp(arg)
    pen = torch.clamp(pen, max=float(cap))
    return pen.sum(dim=-1, keepdim=True)

