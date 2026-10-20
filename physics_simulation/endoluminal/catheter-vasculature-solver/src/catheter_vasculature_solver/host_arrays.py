# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""One place to bring caller-supplied buffers across to the host.

A GPU-resident caller is the normal case for this package, not the exception:
Isaac Lab hands ``reset`` the CUDA index tensor it builds, and a batched
controller keeps its per-env velocities on device precisely to avoid a host
sync per substep. ``np.asarray`` cannot address device memory, so the entry
points that genuinely need host values -- validating indices, reading a scalar
for the single-rod path -- have to copy across explicitly.

Doing that in one place keeps the accepted container types from drifting
between siblings. They already had: the batched control path marshalled a torch
tensor correctly while the single-rod fallback beside it raised on the same
input.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp


def to_host(values, dtype) -> np.ndarray:
    """Copy *values* into a host numpy array of *dtype*, preserving shape.

    Accepts ``torch.Tensor`` (host or device), ``wp.array``, and anything
    ``np.asarray`` already handles. Callers reshape for themselves, since a
    ``(N, 3)`` position array and a flat index list both come through here.
    """
    if isinstance(values, torch.Tensor):
        values = values.detach().to("cpu").contiguous().numpy()
    elif isinstance(values, wp.array):
        values = values.numpy()
    return np.asarray(values, dtype=dtype)


def to_host_ids(env_ids) -> np.ndarray:
    """Flatten *env_ids* into a host int64 index array.

    The rod solver and the vessel runtime validate env ids separately and have
    to agree on what a caller may pass, so both read them through this.
    """
    return to_host(env_ids, np.int64).reshape(-1)


def env_ids_from_world_mask(world_mask, num_envs: int) -> np.ndarray | None:
    """Translate a Newton world mask; ``None`` selects all environments."""
    if world_mask is None:
        return None
    values = to_host(world_mask, np.bool_).reshape(-1)
    if len(values) != num_envs:
        raise ValueError(
            f"world_mask has {len(values)} entries but the solver spans {num_envs} env(s). "
            "Pass one entry per env, or None to reset all of them."
        )
    return np.flatnonzero(values).astype(np.int32)


__all__ = ["env_ids_from_world_mask", "to_host", "to_host_ids"]
