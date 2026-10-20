# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tri-point connectors: a particle of one part held at a rest distance from a triangle of another."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, live


@dataclass(eq=False)
class Connectors(System):
    body: str  # asset id in YAML; the Body instance after construction

    def __post_init__(self):
        scene = self.scene
        self.body = b = scene.body(self.body)
        if b.connectors is None:
            raise ValueError(f"connectors: body {b.name!r} has no connectors")
        c = b.connectors
        E, dev = scene.envs, scene.device
        n = E * b.count  # scratch is body-local: index env * count + local
        scene.check_memory({
            "connectors": c.particle.nbytes + c.tri.nbytes + c.bar.nbytes + c.rest.nbytes + c.group.nbytes,
            "parameters": 3 * E * len(c.stiffness) * 4,
            "delta": n * 12,
        })
        self.variant = b.variant  # (E,) int32, shared by the body's systems
        self.particle = wp.array(c.particle, dtype=wp.int32, device=dev)  # (V, C), -1 = padding
        self.tri = wp.array(c.tri, dtype=wp.vec3i, device=dev)  # (V, C)
        self.bar = wp.array(c.bar, dtype=wp.vec3, device=dev)  # (V, C)
        self.rest = wp.array(c.rest, dtype=wp.float32, device=dev)  # (V, C)
        self.group = wp.array(c.group, dtype=wp.int32, device=dev)  # (C,)
        self.stiffness = live(scene, c.stiffness)  # (E, G) live
        self.register_live("stiffness")
        self.damping = live(scene, c.damping)
        self.register_live("damping")
        self.weight_ratio = live(scene, c.weight_ratio)
        self.register_live("weight_ratio")
        # Zero between steps: apply_direct clears what solve_connectors wrote.
        self.delta = wp.zeros(n, dtype=wp.vec3, device=dev)


    def iteration(self, scene, dt, k, iterations):
        b, E, dev = self.body, scene.envs, scene.device
        if self.particle.shape[1] == 0:
            return
        wp.launch(
            solve_connectors,
            dim=(E, self.particle.shape[1]),
            inputs=[scene.field("q", b), scene.field("inv_mass", b), scene.field("flags", b), self.variant, self.particle,
                    self.tri, self.bar, self.rest, self.group, self.stiffness, self.damping, self.weight_ratio, b.count, dt],
            outputs=[self.delta],
            device=dev,
        )
        wp.launch(
            apply_direct,
            dim=(E, b.count),
            inputs=[scene.field("flags", b), b.count],
            outputs=[scene.field("q", b), self.delta],
            device=dev,
        )


# Ported from omnisurg/tet/kernels/tet.py:1660-1726 (solve_tri_point_connectors_range).
# Edits: (env, connector) prologue with -1 padding and per-variant records; the
# TriPointConnector struct fields are separate arrays; (E, groups) parameters; active_mask
# dropped (flags only); (E, count) particle views; body-local accumulator (index env * count + local).
@wp.kernel
def solve_connectors(
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    particle: wp.array2d(dtype=wp.int32),
    tri: wp.array2d(dtype=wp.vec3i),
    bar: wp.array2d(dtype=wp.vec3),
    rest: wp.array2d(dtype=wp.float32),
    group: wp.array(dtype=wp.int32),
    stiffness: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
    weight_ratio: wp.array2d(dtype=wp.float32),
    count: wp.int32,
    dt: wp.float32,
    delta: wp.array(dtype=wp.vec3),
):
    env, c = wp.tid()
    v = variant[env]
    p = particle[v, c]
    if p < 0:
        return
    g = group[c]
    ke = stiffness[env, g]
    if ke <= 0.0:
        return

    base = env * count
    li = base + p
    t = tri[v, c]
    l0 = base + t[0]
    l1 = base + t[1]
    l2 = base + t[2]
    i = p
    t0 = t[0]
    t1 = t[1]
    t2 = t[2]
    if ((flags[env, i] & ACTIVE) == 0 or (flags[env, t0] & ACTIVE) == 0 or (flags[env, t1] & ACTIVE) == 0
            or (flags[env, t2] & ACTIVE) == 0):
        return
    b = bar[v, c]
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]

    tri_pos = q[env, t0] * b0 + q[env, t1] * b1 + q[env, t2] * b2
    direction = q[env, i] - tri_pos
    length = wp.length(direction)
    if length <= 1.0e-7:
        return

    wi = 0.0
    if inv_mass[env, i] > 0.0:
        wi = wp.min(wp.max(weight_ratio[env, g], 0.0), 1.0)
    wt = 1.0 - wi
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    if inv_mass[env, t0] > 0.0:
        w0 = wt
    if inv_mass[env, t1] > 0.0:
        w1 = wt
    if inv_mass[env, t2] > 0.0:
        w2 = wt
    denom = wi + w0 * b0 * b0 + w1 * b1 * b1 + w2 * b2 * b2
    if denom <= 1.0e-8:
        return

    kd = wp.max(0.0, damping[env, g])
    k = ke / (1.0 + kd * dt)
    correction = ((length - rest[v, c]) / denom) * (direction / length) * k
    wp.atomic_add(delta, li, -correction * wi)
    wp.atomic_add(delta, l0, correction * b0 * w0)
    wp.atomic_add(delta, l1, correction * b1 * w1)
    wp.atomic_add(delta, l2, correction * b2 * w2)


# Ported from omnisurg/tet/kernels/tet.py:1744-1752 (apply_direct_deltas_range).
# Edits: (env, local) over (E, count) views; body-local accumulator; inactive particles clear it without moving.
@wp.kernel
def apply_direct(
    flags: wp.array2d(dtype=wp.int32),
    count: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
):
    env, local = wp.tid()
    flat_index = env * count + local
    if (flags[env, local] & ACTIVE) != 0:
        q[env, local] = q[env, local] + delta[flat_index]
    delta[flat_index] = wp.vec3(0.0, 0.0, 0.0)
