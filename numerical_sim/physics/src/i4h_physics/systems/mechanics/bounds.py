# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Axis-aligned position bounds (floor and walls).

Projected ``iterations`` times per substep (default once, on the last iteration; omnisurg
projected on every iteration, which cost one launch per body per iteration for a box the tissue
rarely reaches). Systems listed after it in the same iteration (contacts, grasp, clips) may still
move a particle past the box; the next substep's projection brings it back.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, due


@dataclass(eq=False)
class Bounds(System):
    lower: tuple = (-1.0, -1.0, -1.0)  # structural
    upper: tuple = (1.0, 1.0, 1.0)  # structural
    iterations: int = 1  # structural: projections per substep

    structural = System.structural + ("lower", "upper", "iterations")

    def __post_init__(self):
        self.lower = tuple(float(v) for v in self.lower)
        self.upper = tuple(float(v) for v in self.upper)
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("bounds.lower and bounds.upper must have three components")


    def iteration(self, scene, dt, k, iterations):
        if due(k, self.iterations, iterations)[0] < 0:
            return
        lo, hi = wp.vec3(*self.lower), wp.vec3(*self.upper)  # built here so a rebind + prepare() recaptures the new box
        views = [(scene.block_field, scene.main.width)] if scene.main.width else []  # the main block in one launch
        views += [((lambda name, b=b: scene.field(name, b)), b.count) for b in scene.pools]  # then each fluid pool
        for f, n in views:
            wp.launch(project_bounds, dim=(scene.envs, n), inputs=[f("inv_mass"), f("flags"), lo, hi], outputs=[f("q")],
                      device=scene.device)


# Ported from omnisurg/tet/kernels/tet.py:1869-1895 (project_bounds_range).
# Edits: (env, local) prologue; inactive particles skipped (the source checked mass only);
# the velocity write is dropped because finalize recomputes velocity from positions every
# substep, which made the source's velocity output dead.
@wp.kernel
def project_bounds(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    positions: wp.array2d(dtype=wp.vec3),
):
    env, i = wp.tid()
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        return

    position = positions[env, i]
    for axis in range(3):
        if position[axis] < bounds_min[axis]:
            position[axis] = bounds_min[axis]
        elif position[axis] > bounds_max[axis]:
            position[axis] = bounds_max[axis]

    positions[env, i] = position
