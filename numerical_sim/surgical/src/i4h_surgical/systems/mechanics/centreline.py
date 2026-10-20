# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centreline tracker: each vessel node follows the mean of its clamp particles.

Ported from omnisurg/tet/kernels/tet.py:37-52 (update_centreline_positions). omnisurg ran it every
substep_end (tet/solver/centreline.py:61-80); here it runs once in step_end, since nothing moves the
tissue after the last substep (consumers read it in step_start, or in a step_end listed after it). Edits: (env, node) prologue
over per-variant clamp tables; inactive (removed) particles drop out of the mean and a node
whose particles are all gone keeps its last position (omnisurg averaged whatever was there).
Consumers: clipping (attach by nearest node, pinch its ring) and bleeding (cut sites). ``positions``
``(E, K)`` and ``active_count`` ``(E, K)`` are torch-viewable per env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.scene import ACTIVE, restore_variant_rows
from i4h_physics.systems.base import System


@dataclass(eq=False)
class Centreline(System):
    body: str  # asset id with ``centrelines:``; the Body after construction

    def __post_init__(self):
        scene = self.scene
        self.body = b = scene.body(self.body)
        c = getattr(b, "centrelines", None)
        if c is None:
            raise ValueError(f"centreline: body {b.name!r} has no centrelines")
        E, dev = scene.envs, scene.device
        V, K = c.clamp_start.shape
        scene.check_memory({"clamp": c.clamp.nbytes + c.clamp_start.nbytes * 2 + c.rest.nbytes, "positions": E * K * 16})
        self.names = list(c.names)
        self.variant = b.variant
        self.line = np.asarray(c.line, np.int32)  # (K,) line of each node; only read on the host (clipping, bleeding)
        self.clamp_start = wp.array(c.clamp_start, dtype=wp.int32, device=dev)  # (V, K)
        self.clamp_count = wp.array(c.clamp_count, dtype=wp.int32, device=dev)  # (V, K)
        self.clamp = wp.array(c.clamp, dtype=wp.int32, device=dev)  # (V, C) body-local
        self.rest = wp.array(c.rest, dtype=wp.vec3, device=dev)  # (V, K)
        self.positions = wp.empty((E, K), dtype=wp.vec3, device=dev)
        self.active_count = wp.zeros((E, K), dtype=wp.int32, device=dev)  # live clamp particles per node
        self.reset(scene, wp.ones(E, dtype=wp.bool, device=dev))

    def step_end(self, scene, dt):
        b = self.body
        K = self.positions.shape[1]
        if K == 0:
            return
        wp.launch(update_centreline, dim=(scene.envs, K),
                  inputs=[scene.field("q", b), scene.field("flags", b), self.variant, self.clamp_start, self.clamp_count, self.clamp],
                  outputs=[self.positions, self.active_count], device=scene.device)

    def reset(self, scene, mask):
        """Rest positions and full clamp counts (every particle is alive again), so a clip can attach before
        the next ``step_end`` refreshes the tracker."""
        if self.positions.shape[1]:
            for src, dst in ((self.rest, self.positions), (self.clamp_count, self.active_count)):
                wp.launch(restore_variant_rows, dim=dst.shape, inputs=[mask, self.variant, src], outputs=[dst], device=scene.device)


@wp.kernel
def update_centreline(
    q: wp.array2d(dtype=wp.vec3),
    flags: wp.array2d(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    clamp_start: wp.array2d(dtype=wp.int32),
    clamp_count: wp.array2d(dtype=wp.int32),
    clamp: wp.array2d(dtype=wp.int32),
    positions: wp.array2d(dtype=wp.vec3),
    active_count: wp.array2d(dtype=wp.int32),
):
    env, k = wp.tid()
    v = variant[env]
    s = clamp_start[v, k]
    total = wp.vec3(0.0, 0.0, 0.0)
    m = wp.int32(0)
    for j in range(clamp_count[v, k]):
        p = clamp[v, s + j]
        if p < 0:
            continue
        if (flags[env, p] & ACTIVE) == 0:
            continue
        total = total + q[env, p]
        m = m + 1
    active_count[env, k] = m
    if m > 0:
        positions[env, k] = total / wp.float32(m)
