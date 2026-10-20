# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tet deformation: distance constraints on the edge list and volume constraints on tets.

Jacobi solve as in omnisurg: both kernels accumulate weighted corrections and counts per
particle with atomics, then one launch applies the average and clears the accumulators.
Topology and rest geometry are per mesh variant, looked up through ``variant[env]``;
the active masks are per env (cutting flips them); materials are ``(E, parts)`` live arrays.

``model: xpbd`` (opt-in; the default ``pbd`` is omnisurg's per-iteration stiffness factors) reads the parts'
``spring`` as an edge stiffness (N/m) and ``volume`` as a volume stiffness (N/m^5, for C = V - V0), each constraint
keeping its multiplier over the substep (compliance ``1 / (k dt^2)``), as the ``shell`` system's XPBD Jacobi (the
corrections averaged by the constraints that fired). Its stiffness then no longer depends on dt, substeps or
iterations once converged; the damping factors are not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.scene import ACTIVE, restore_variant_rows
from i4h_physics.systems.base import System, live


@dataclass(eq=False)
class Tet(System):
    body: str  # asset id in YAML; the Body instance after construction
    iterations: int = 1  # repeated existing solve/apply sweeps per runner iteration
    model: str = "pbd"  # structural: pbd (stiffness factors per iteration, omnisurg) or xpbd (compliance, see above)

    structural = System.structural + ("iterations", "model")

    def prepare(self, scene):
        super().prepare(scene)  # model sizes the multipliers: fixed at construction
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("tet.iterations must be a positive integer")
        if self.model not in ("pbd", "xpbd"):
            raise ValueError(f"tet.model must be 'pbd' or 'xpbd', got {self.model!r}")

    def __post_init__(self):
        scene = self.scene
        self.prepare(scene)
        self.body = b = scene.body(self.body)
        if b.tet is None:
            raise ValueError(f"tet: body {b.name!r} has no tet mesh")
        m = b.tet
        E, dev = scene.envs, scene.device
        n = E * b.count  # scratch is body-local: index env * count + local
        scene.check_memory({
            "variants": m.tets.nbytes + m.active0.nbytes + m.rest_volume.nbytes + m.edges.nbytes
                        + m.edge_active0.nbytes + m.edge_rest.nbytes + m.part.nbytes + m.edge_part.nbytes,
            "active": E * (m.active0.shape[1] + m.edge_active0.shape[1]) * 4,
            "materials": 4 * E * len(m.spring) * 4,
            "delta": n * 12 + n * 4,
        })
        self.variant = b.variant  # (E,) int32, shared by the body's systems
        self.tets = wp.array(m.tets, dtype=wp.vec4i, device=dev)  # (V, T)
        self.active0 = wp.array(m.active0, dtype=wp.int32, device=dev)  # (V, T)
        self.active = wp.empty((E, m.active0.shape[1]), dtype=wp.int32, device=dev)  # (E, T); cutting flips this
        self.rest_volume = wp.array(m.rest_volume, dtype=wp.float32, device=dev)  # (V, T)
        self.tet_part = wp.array(m.part, dtype=wp.int32, device=dev)
        self.edges = wp.array(m.edges, dtype=wp.vec2i, device=dev)  # (V, S)
        self.edge_active0 = wp.array(m.edge_active0, dtype=wp.int32, device=dev)  # (V, S)
        self.edge_active = wp.empty((E, m.edge_active0.shape[1]), dtype=wp.int32, device=dev)  # (E, S)
        self.edge_rest = wp.array(m.edge_rest, dtype=wp.float32, device=dev)  # (V, S)
        self.edge_part = wp.array(m.edge_part, dtype=wp.int32, device=dev)
        self.reset(scene, wp.ones(E, dtype=wp.bool, device=dev))  # live masks start as their variant's masks
        # Live (E, parts) materials, named after their YAML keys.
        self.spring = live(scene, m.spring)
        self.register_live("spring")
        self.spring_damping = live(scene, m.spring_damping)
        self.register_live("spring_damping")
        self.volume = live(scene, m.volume)
        self.register_live("volume")
        self.volume_damping = live(scene, m.volume_damping)
        self.register_live("volume_damping")
        # Accumulators are zero between steps: every apply launch clears what the solves wrote.
        self.delta = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.delta_count = wp.zeros(n, dtype=wp.int32, device=dev)
        S, T = m.edges.shape[1], m.tets.shape[1]
        xpbd = self.model == "xpbd"  # the multipliers, zeroed every substep (xpbd only)
        self.edge_lambda = wp.zeros((E, S) if xpbd else (1, 1), dtype=wp.float32, device=dev)
        self.tet_lambda = wp.zeros((E, T) if xpbd else (1, 1), dtype=wp.float32, device=dev)
        self.freeze("model")

    def substep_start(self, scene, dt, s, substeps):
        if self.model == "xpbd":
            self.edge_lambda.zero_()
            self.tet_lambda.zero_()


    def iteration(self, scene, dt, k, iterations):
        b, E, dev = self.body, scene.envs, scene.device
        S, T = self.edges.shape[1], self.tets.shape[1]
        for _ in range(self.iterations):
            if S + T:
                wp.launch(
                    solve_tet,
                    dim=(E, S + T),
                    inputs=[scene.field("q", b), scene.field("inv_mass", b), self.variant, self.edges, self.edge_active,
                            self.edge_rest, self.edge_part, self.spring, self.spring_damping, self.tets, self.active,
                            self.rest_volume, self.tet_part, self.volume, self.volume_damping, S, b.count, dt,
                            int(self.model == "xpbd")],
                    outputs=[self.delta, self.delta_count, self.edge_lambda, self.tet_lambda],
                    device=dev,
                )
            wp.launch(
                apply_average,
                dim=(E, b.count),
                inputs=[scene.field("flags", b), b.count],
                outputs=[scene.field("q", b), self.delta, self.delta_count],
                device=dev,
            )

    def reset(self, scene, mask):
        """Restore the selected envs' active masks from their variant's initial masks."""
        for src, dst in ((self.active0, self.active), (self.edge_active0, self.edge_active)):
            if dst.shape[1]:
                wp.launch(restore_variant_rows, dim=dst.shape, inputs=[mask, self.variant, src], outputs=[dst],
                          device=scene.device)


# Ported from omnisurg/tet/kernels/tet.py:1506-1565 (solve_distance_constraints_range).
# Edits: (env, edge) prologue with body-local vec2i edges looked up per variant; (E, parts)
# stiffness and damping via edge_part; the incident-tet liveness scan (spring_tet_offsets/ids)
# is replaced by the per-env edge_active; the body's particles are (E, count) views, the accumulators
# body-local (index env * count + local).
@wp.func
def distance_constraint(
    env: wp.int32,
    s: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    edges: wp.array2d(dtype=wp.vec2i),
    edge_active: wp.array2d(dtype=wp.int32),
    edge_rest: wp.array2d(dtype=wp.float32),
    edge_part: wp.array(dtype=wp.int32),
    stiffness: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
    count: wp.int32,
    dt: wp.float32,
    xpbd: wp.int32,
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
    lambdas: wp.array2d(dtype=wp.float32),
):
    if edge_active[env, s] == 0:
        return
    v = variant[env]
    part = edge_part[s]
    ke = stiffness[env, part]
    e = edges[v, s]
    li = env * count + e[0]
    lj = env * count + e[1]
    wi = inv_mass[env, e[0]]
    wj = inv_mass[env, e[1]]
    w = wi + wj
    if w <= 0.0 or ke <= 0.0:
        return

    xi = q[env, e[0]]
    xj = q[env, e[1]]
    diff = xi - xj
    length = wp.length(diff)
    if length <= 1.0e-8:
        return

    direction = diff / length
    if xpbd != 0:  # compliance 1 / (k dt^2), the multiplier kept over the substep (shell.py's edge solve)
        alpha = 1.0 / (ke * dt * dt)
        dlambda = (edge_rest[v, s] - length - alpha * lambdas[env, s]) / (w + alpha)
        lambdas[env, s] = lambdas[env, s] + dlambda
        correction = dlambda * direction
    else:
        kd = wp.max(0.0, damping[env, part])
        k = ke / (1.0 + kd * dt)
        correction = -((length - edge_rest[v, s]) / w) * k * direction

    wp.atomic_add(delta, li, wi * correction)
    wp.atomic_add(delta, lj, -wj * correction)
    wp.atomic_add(delta_count, li, 1)
    wp.atomic_add(delta_count, lj, 1)


# Ported from omnisurg/tet/kernels/tet.py:1593-1657 (solve_volume_constraints_range).
# Edits: (env, tet) prologue with body-local vec4i tets looked up per variant; (E, parts)
# stiffness and damping via tet_part; (E, count) particle views, body-local accumulators.
@wp.func
def volume_constraint(
    env: wp.int32,
    t: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    tets: wp.array2d(dtype=wp.vec4i),
    active: wp.array2d(dtype=wp.int32),
    rest_volume: wp.array2d(dtype=wp.float32),
    tet_part: wp.array(dtype=wp.int32),
    stiffness: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
    count: wp.int32,
    dt: wp.float32,
    xpbd: wp.int32,
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
    lambdas: wp.array2d(dtype=wp.float32),
):
    if active[env, t] == 0:
        return
    v = variant[env]
    part = tet_part[t]
    ke = stiffness[env, part]
    if ke <= 0.0:
        return

    ids = tets[v, t]
    l0 = env * count + ids[0]
    l1 = env * count + ids[1]
    l2 = env * count + ids[2]
    l3 = env * count + ids[3]

    p0 = q[env, ids[0]]
    p1 = q[env, ids[1]]
    p2 = q[env, ids[2]]
    p3 = q[env, ids[3]]
    w0 = inv_mass[env, ids[0]]
    w1 = inv_mass[env, ids[1]]
    w2 = inv_mass[env, ids[2]]
    w3 = inv_mass[env, ids[3]]

    volume = wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    constraint = volume - rest_volume[v, t]

    grad0 = wp.cross(p1 - p2, p3 - p2) / 6.0
    grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0
    grad2 = wp.cross(p0 - p1, p3 - p1) / 6.0
    grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0

    denom = (w0 * wp.length_sq(grad0) + w1 * wp.length_sq(grad1)
             + w2 * wp.length_sq(grad2) + w3 * wp.length_sq(grad3))
    if denom <= 1.0e-8:
        return

    scale = float(0.0)
    if xpbd != 0:  # compliance 1 / (k dt^2); the corrections below are -grad * w * scale
        alpha = 1.0 / (ke * dt * dt)
        dlambda = (-constraint - alpha * lambdas[env, t]) / (denom + alpha)
        lambdas[env, t] = lambdas[env, t] + dlambda
        scale = -dlambda
    else:
        kd = wp.max(0.0, damping[env, part])
        k = ke / (1.0 + kd * dt)
        scale = k * constraint / denom
    wp.atomic_add(delta, l0, -grad0 * scale * w0)
    wp.atomic_add(delta, l1, -grad1 * scale * w1)
    wp.atomic_add(delta, l2, -grad2 * scale * w2)
    wp.atomic_add(delta, l3, -grad3 * scale * w3)

    wp.atomic_add(delta_count, l0, 1)
    wp.atomic_add(delta_count, l1, 1)
    wp.atomic_add(delta_count, l2, 1)
    wp.atomic_add(delta_count, l3, 1)


# One launch for both constraint kinds (they accumulate into the same buffers and never read each other's output):
# threads [0, S) are edges, [S, S + T) tets.
@wp.kernel
def solve_tet(
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    edges: wp.array2d(dtype=wp.vec2i),
    edge_active: wp.array2d(dtype=wp.int32),
    edge_rest: wp.array2d(dtype=wp.float32),
    edge_part: wp.array(dtype=wp.int32),
    spring: wp.array2d(dtype=wp.float32),
    spring_damping: wp.array2d(dtype=wp.float32),
    tets: wp.array2d(dtype=wp.vec4i),
    active: wp.array2d(dtype=wp.int32),
    rest_volume: wp.array2d(dtype=wp.float32),
    tet_part: wp.array(dtype=wp.int32),
    volume: wp.array2d(dtype=wp.float32),
    volume_damping: wp.array2d(dtype=wp.float32),
    S: wp.int32,
    count: wp.int32,
    dt: wp.float32,
    xpbd: wp.int32,
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
    edge_lambda: wp.array2d(dtype=wp.float32),
    tet_lambda: wp.array2d(dtype=wp.float32),
):
    env, i = wp.tid()
    if i < S:
        distance_constraint(env, i, q, inv_mass, variant, edges, edge_active, edge_rest, edge_part, spring, spring_damping,
                            count, dt, xpbd, delta, delta_count, edge_lambda)
    else:
        volume_constraint(env, i - S, q, inv_mass, variant, tets, active, rest_volume, tet_part, volume, volume_damping,
                          count, dt, xpbd, delta, delta_count, tet_lambda)


# Ported from omnisurg/tet/kernels/tet.py:1729-1741 (apply_average_deltas_range).
# Edits: (env, local) over the body's (E, count) views; body-local accumulators (index env * count + local);
# inactive particles clear their accumulator without moving (omnisurg zeroed inv_mass on deactivation instead).
@wp.kernel
def apply_average(
    flags: wp.array2d(dtype=wp.int32),
    count: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    flat_index = env * count + local
    if (flags[env, local] & ACTIVE) != 0 and delta_count[flat_index] > 0:
        q[env, local] = q[env, local] + delta[flat_index] / wp.float32(delta_count[flat_index])

    delta[flat_index] = wp.vec3(0.0, 0.0, 0.0)
    delta_count[flat_index] = 0
