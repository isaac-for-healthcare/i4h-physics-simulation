# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch views and action bounds of a Stepper for the Isaac Lab env; no Isaac Lab import, so the worktree
suite exercises them and the chole path has a stub loop.

Nothing here names a system class: the systems declare what a host reads (``System.is_control``,
``action_bounds``, ``recorded_actions``, ``state_products``), so a system defined outside these packages takes part
the same way. Actions: the control systems (``instrument_control``: ``(E, n, 10)`` = position, quaternion xyzw, grip,
energy, fire per instrument; ``catheter_control``: ``(E, 3)`` normalised insertion, rotation and tip-bend rates) in
system-list order, each flattened to ``(E, width)``; the policy sees their concatenation as one normalised ``[-1, 1]``
vector that ``map_actions`` maps onto the physical bounds each control declares (``action_bounds``). The zero action
is the reset pose for the built-in controls. Observations: one ``(E, 3n)`` zero-copy view per integrated body (fluid
pools are counts, not positions), or one part's particles for ``observe = "body.part"``, followed by the systems'
``state_products`` cast to float. Every view shares memory with the Warp array it comes from, so a policy
writing the action views or reading the position views touches the scene directly.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp


def _flat(arr: wp.array, E: int) -> torch.Tensor:
    """``(E, -1)`` torch view of a Warp array; ``view`` raises instead of copying, so a strided part view stays a view."""
    t = wp.to_torch(arr)
    flat = t.view(E, -1)
    if flat.data_ptr() != arr.ptr:
        raise RuntimeError(f"expected a zero-copy view of a {arr.shape} {arr.dtype.__name__} array")
    return flat


def controls(sim) -> list:
    """The control systems with a non-empty actions array, in system-list order."""
    return [s for s in sim.systems if getattr(s, "is_control", False) and s.actions.size]


def action_views(sim) -> list[torch.Tensor]:
    """Zero-copy ``(E, width)`` views of every control system's actions."""
    return [_flat(s.actions, sim.scene.envs) for s in controls(sim)]


def action_bounds(sim, workspace: float) -> tuple[np.ndarray, np.ndarray]:
    """Physical ``(A,)`` float32 lower and upper bounds in the concatenated column order of ``action_views``."""
    low, high = [], []
    for s in controls(sim):
        lo, hi = s.action_bounds(workspace)
        width = int(np.prod(s.actions.shape[1:]))
        if len(lo) != width or len(hi) != width:
            raise ValueError(f"{s.id or type(s).__name__}.action_bounds: {len(lo)}/{len(hi)} bounds for {width} action columns")
        low += list(lo)
        high += list(hi)
    return np.asarray(low, np.float32), np.asarray(high, np.float32)


def map_actions(actions: torch.Tensor, mid: torch.Tensor, half: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Clamp a normalised ``(E, A)`` action to [-1, 1] and map it onto ``mid ± half`` (``(A,)`` device tensors from
    the bounds: ``mid = (low + high) / 2``, ``half = (high - low) / 2``) into ``out`` without allocating."""
    torch.clamp(actions, -1.0, 1.0, out=out)
    return out.mul_(half).add_(mid)


def write_actions(views: list[torch.Tensor], actions: torch.Tensor) -> None:
    """Copy consecutive column slices of a physical ``(E, A)`` action into the control systems' views."""
    offset = 0
    for view in views:
        width = int(view.shape[1])
        view.copy_(actions[:, offset : offset + width])
        offset += width


def observation_views(sim, observe: str = "") -> list[torch.Tensor]:
    """Zero-copy ``(E, 3n)`` positions: all integrated bodies, ``observe = "body"``,
    or one part for ``observe = "body.part"``.
    """
    scene = sim.scene
    if observe:
        body_name, separator, part = observe.partition(".")
        return [_flat(scene.field("q", scene.body(body_name), part if separator else None), scene.envs)]
    return [_flat(scene.view(scene.q, b), scene.envs) for b in scene.bodies if b.integrate]


def state_views(sim) -> list[torch.Tensor]:
    """Zero-copy ``(E, k)`` views of the systems' ``state_products``, system-list order."""
    E = sim.scene.envs
    return [_flat(getattr(s, name), E) for s in sim.systems for name in getattr(s, "state_products", ())]


def state_names(sim) -> list[str]:
    """``"<system id>.<product>"`` per ``state_views`` entry, same order (dataset keys, logs)."""
    return [f"{s.id}.{name}" for s in sim.systems for name in getattr(s, "state_products", ())]


def replay_actions(sim, base, low: np.ndarray, high: np.ndarray, decimation: int = 1) -> tuple[np.ndarray, float]:
    """A session recorded by the app (``--record <base>`` writes ``<base>_<device>.npy`` per device, one row
    ``[px, py, pz, qx, qy, qz, qw, grip, energy, fire]`` per physics step) as normalised ``(T, A)`` actions in the
    column order of ``action_views``: every ``decimation``-th row (Isaac Lab applies one action per ``decimation``
    physics steps), instruments without a file hold their reset pose, catheters get zero rates, the last row is
    repeated to the longest recording. Also returns how far the recording left the bounds (0 = inside)."""
    from pathlib import Path

    base = Path(base)
    mid = (low + high) / 2.0
    columns, length = [], 1
    recordings = {}
    for s in controls(sim):
        for width, rows in s.recorded_actions(base):
            recordings[len(columns)] = rows
            columns.append(width)
            if rows is not None:
                length = max(length, len(rows))
    physical = np.tile(mid[None], (length, 1))  # the reset pose (grip, energy, fire at 0 since their mid is 0)
    offset = 0
    for index, width in enumerate(columns):
        rows = recordings[index]
        if rows is not None:
            padded = np.concatenate([rows, np.repeat(rows[-1:], length - len(rows), axis=0)]) if len(rows) < length else rows[:length]
            physical[:, offset : offset + width] = padded
        offset += width
    normalised, excess = physical_to_normalized(physical, low, high)
    return normalised[:: max(1, int(decimation))], excess


def physical_to_normalized(physical: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, float]:
    """Physical ``(..., A)`` rows as normalised actions clipped to [-1, 1] (``map_actions`` inverted), and how far the
    rows left the bounds (0 = inside, else the largest excess in half-ranges)."""
    mid, half = (low + high) / 2.0, (high - low) / 2.0
    normalised = (np.asarray(physical, np.float32) - mid) / np.where(half > 0.0, half, 1.0)
    excess = float(max(0.0, np.abs(normalised).max(initial=0.0) - 1.0))
    return np.clip(normalised, -1.0, 1.0).astype(np.float32), excess


def observation(obs: list[torch.Tensor], state: list[torch.Tensor]) -> torch.Tensor:
    """The policy observation ``(E, W)`` float32: positions then state (the one copy per step)."""
    return torch.cat(obs + state, dim=1)  # cat promotes the int32 state views to float32
