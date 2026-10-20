# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin attachments: selected particles are pulled toward a fixed rest position each iteration."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System


@dataclass(eq=False)
class Attachments(System):
    body: str  # asset id in YAML; the Body instance after construction
    stiffness: float = 1.0  # live, per env; per-iteration lerp fraction toward rest

    def __post_init__(self):
        scene = self.scene
        self.body = b = scene.body(self.body)
        if b.pin is None:
            raise ValueError(f"attachments: body {b.name!r} has no pin")
        dev = scene.device
        scene.check_memory({"pins": b.pin.index.nbytes + b.pin.rest.nbytes, "stiffness": scene.envs * 4})
        self.variant = b.variant  # (E,) int32, shared by the body's systems
        self.index = wp.array(b.pin.index, dtype=wp.int32, device=dev)  # (V, A), -1 = padding
        self.rest = wp.array(b.pin.rest, dtype=wp.vec3, device=dev)  # (V, A)
        self._live("stiffness")


    def iteration(self, scene, dt, k, iterations):
        b = self.body
        if self.index.shape[1] == 0:
            return
        wp.launch(
            solve_attachments,
            dim=(scene.envs, self.index.shape[1]),
            inputs=[scene.field("inv_mass", b), scene.field("flags", b), self.variant, self.index, self.rest, self.stiffness],
            outputs=[scene.field("q", b)],
            device=scene.device,
        )


# Ported from omnisurg/tet/kernels/tet.py:1568-1590 (solve_attachment_constraints).
# Edits: (env, attachment) prologue with -1 padding and per-variant pins; stiffness per env;
# active_mask dropped. Each thread owns one distinct particle, so the in-place write is safe.
@wp.kernel
def solve_attachments(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    index: wp.array2d(dtype=wp.int32),
    rest: wp.array2d(dtype=wp.vec3),
    stiffness: wp.array(dtype=wp.float32),
    q: wp.array2d(dtype=wp.vec3),
):
    env, a = wp.tid()
    v = variant[env]
    i = index[v, a]
    if i < 0:
        return
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        return

    strength = wp.clamp(stiffness[env], 0.0, 1.0)
    q[env, i] = q[env, i] + strength * (rest[v, a] - q[env, i])
