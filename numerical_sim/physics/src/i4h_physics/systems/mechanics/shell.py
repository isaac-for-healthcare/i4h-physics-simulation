# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deformable shell (cloth) vessel wall: XPBD distance and dihedral bending constraints, Jacobi.

Ported from omniendo/xpbd_rod/kernels_deformable.py:83-109 (distance, Jacobi), :112-185 (dihedral angle and
its gradients), :226-261 (dihedral, Jacobi); the coloured Gauss-Seidel variants are not ported (ragged
per-colour launches do not batch). The schedule is solver_xpbd_rod.py:1200-1340 (``_step_deformable``):
the multipliers are zeroed once per substep, then ``deformation_iterations`` sweeps of distance, dihedral
and apply. Edits: the ``(env, constraint)`` prologue with per-variant topology; the compliance is computed
inline from the live stiffness (``1/k`` for an edge, ``1/(k * rest edge length)`` for a dihedral,
solver_xpbd_rod.py:519-551) so nothing is rebound; the corrections are averaged by the number of constraints
that fired this sweep (``tet.apply_average``) where omniendo divided by a static incidence count (identical
whenever every constraint fires). Predict and finalize are the ``particles`` system's (the body integrates like
tissue); pinned vertices have zero inverse mass and never move. The body comes from a ``vessel_mesh`` asset
with a ``deformable`` block.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.systems.base import System
from i4h_physics.systems.mechanics.tet import apply_average


@dataclass(eq=False)
class Shell(System):
    body: str
    stretch: float = 2.0e5  # live, per env: edge stiffness (omniendo stretch_stiffness; 0 disables the edges)
    bend: float = 1.0e4  # live, per env: dihedral stiffness (bending_stiffness; 0 disables)
    relaxation: float = 1.0  # live, per env: fraction of each correction applied (deformation_relaxation)
    iterations: int = 8  # structural: Jacobi sweeps per runner iteration (deformation_iterations)

    structural = System.structural + ("iterations",)

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        self.body = b = scene.body(self.body)
        if b.shell is None:
            raise ValueError(f"shell: body {b.name!r} has no shell record (a target_mesh asset with a deformable block)")
        m = b.shell
        if m.tris.shape[0] != 1:
            raise ValueError("shell: one mesh variant only")
        self.iterations = check_int("shell.iterations", self.iterations, 1)
        S, D, n = m.edges.shape[1], m.dihedrals.shape[1], b.count
        self.S, self.D = S, D
        scene.check_memory({"lambdas": E * (S + D) * 4, "delta": E * n * 16, "topology": (S * 3 + D * 6) * 4})
        self.variant = b.variant
        self.edges = wp.array(m.edges, dtype=wp.vec2i, device=dev)  # (V, S)
        self.edge_rest = wp.array(m.edge_rest, dtype=wp.float32, device=dev)
        self.dihedrals = wp.array(m.dihedrals, dtype=wp.vec4i, device=dev)  # (V, D)
        self.dihedral_rest = wp.array(m.dihedral_rest, dtype=wp.float32, device=dev)
        self.dihedral_edge_len = wp.array(m.dihedral_edge_len, dtype=wp.float32, device=dev)
        self.lambda_s = wp.zeros((E, S), dtype=wp.float32, device=dev)
        self.lambda_d = wp.zeros((E, D), dtype=wp.float32, device=dev)
        self.delta = wp.zeros(E * n, dtype=wp.vec3, device=dev)  # body-local, self-clearing
        self.delta_count = wp.zeros(E * n, dtype=wp.int32, device=dev)
        self._live("stretch", "bend", "relaxation")


    def substep_start(self, scene, dt, s, substeps):
        self.lambda_s.zero_()  # solver_xpbd_rod.py:1234-1236: once per substep, not per sweep
        self.lambda_d.zero_()

    def iteration(self, scene, dt, k, iterations):
        b, E, dev = self.body, scene.envs, scene.device
        for _ in range(self.iterations):
            if self.S + self.D:
                wp.launch(solve_shell_xpbd, dim=(E, self.S + self.D),
                          inputs=[scene.field("q", b), scene.field("inv_mass", b), self.variant, self.edges, self.edge_rest,
                                  self.stretch, self.dihedrals, self.dihedral_rest, self.dihedral_edge_len, self.bend,
                                  self.relaxation, self.S, b.count, dt],
                          outputs=[self.lambda_s, self.lambda_d, self.delta, self.delta_count], device=dev)
            wp.launch(apply_average, dim=(E, b.count), inputs=[scene.field("flags", b), b.count],
                      outputs=[scene.field("q", b), self.delta, self.delta_count], device=dev)


# Ported from kernels_deformable.py:83-109 (_warp_project_distance_jacobi). Edits: (env, edge) prologue with
# per-variant topology and per-env multipliers; alpha = 1/(k dt^2) from the live stiffness (compliance 1/k,
# solver_xpbd_rod.py:452-475), k <= 0 disables the constraint; the contribution count is accumulated for
# tet.apply_average. Positions are read only; the apply writes them.
@wp.func
def distance_xpbd(
    env: wp.int32,
    s: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    edges: wp.array2d(dtype=wp.vec2i),
    edge_rest: wp.array2d(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    relaxation: wp.array(dtype=wp.float32),
    count: wp.int32,
    dt: wp.float32,
    lambdas: wp.array2d(dtype=wp.float32),
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
):
    k = stiffness[env]
    if k <= 0.0:
        return
    v = variant[env]
    e = edges[v, s]
    li = env * count + e[0]
    lj = env * count + e[1]
    i = e[0]
    j = e[1]
    d = q[env, j] - q[env, i]
    length = wp.length(d)
    if length > 1.0e-12:
        n = d / length
        alpha = 1.0 / (k * dt * dt)
        wi = inv_mass[env, i]
        wj = inv_mass[env, j]
        denom = wi + wj + alpha
        if denom > 0.0:
            dlambda = (edge_rest[v, s] - length - alpha * lambdas[env, s]) / denom
            lambdas[env, s] = lambdas[env, s] + dlambda
            r = relaxation[env]
            wp.atomic_add(delta, li, -n * (r * wi * dlambda))
            wp.atomic_add(delta, lj, n * (r * wj * dlambda))
            wp.atomic_add(delta_count, li, 1)
            wp.atomic_add(delta_count, lj, 1)


# kernels_deformable.py:112-117 (_normalized_vector_derivative). Edits: none.
@wp.func
def normalized_vector_derivative(vector_length: float, unit_vector: wp.vec3, derivative: wp.mat33) -> wp.mat33:
    projection = wp.identity(n=3, dtype=float) - wp.outer(unit_vector, unit_vector)
    return projection * derivative / vector_length


# kernels_deformable.py:120-132 (_angle_derivative). Edits: none.
@wp.func
def angle_derivative(n0: wp.vec3, n1: wp.vec3, edge: wp.vec3, dn0: wp.mat33, dn1: wp.mat33, sin_angle: float,
                     cos_angle: float) -> wp.vec3:
    dsin = wp.transpose(wp.skew(n0) * dn1 - wp.skew(n1) * dn0) * edge
    dcos = wp.transpose(dn0) * n1 + wp.transpose(dn1) * n0
    return dsin * cos_angle - dcos * sin_angle


# kernels_deformable.py:135-175 (_dihedral_angle_gradients): signed angle and gradients for
# (opposite0, opposite1, edge0, edge1); ``valid`` False on a degenerate configuration. Edits: none.
@wp.func
def dihedral_angle_gradients(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    edge = x3 - x2

    raw_n0 = wp.cross(x02, x03)
    raw_n1 = wp.cross(x13, x12)
    n0_length = wp.length(raw_n0)
    n1_length = wp.length(raw_n1)
    edge_length = wp.length(edge)
    if n0_length <= 1.0e-8 or n1_length <= 1.0e-8 or edge_length <= 1.0e-8:
        zero = wp.vec3(0.0)
        return 0.0, zero, zero, zero, zero, False

    n0 = raw_n0 / n0_length
    n1 = raw_n1 / n1_length
    edge_unit = edge / edge_length
    sin_angle = wp.dot(wp.cross(n0, n1), edge_unit)
    cos_angle = wp.clamp(wp.dot(n0, n1), -1.0, 1.0)
    angle = wp.atan2(sin_angle, cos_angle)

    dn0_x0 = normalized_vector_derivative(n0_length, n0, wp.skew(edge))
    dn1_x0 = wp.mat33(0.0)
    dn0_x1 = wp.mat33(0.0)
    dn1_x1 = normalized_vector_derivative(n1_length, n1, -wp.skew(edge))
    dn0_x2 = normalized_vector_derivative(n0_length, n0, -wp.skew(x03))
    dn1_x2 = normalized_vector_derivative(n1_length, n1, wp.skew(x13))
    dn0_x3 = normalized_vector_derivative(n0_length, n0, wp.skew(x02))
    dn1_x3 = normalized_vector_derivative(n1_length, n1, -wp.skew(x12))

    g0 = angle_derivative(n0, n1, edge_unit, dn0_x0, dn1_x0, sin_angle, cos_angle)
    g1 = angle_derivative(n0, n1, edge_unit, dn0_x1, dn1_x1, sin_angle, cos_angle)
    g2 = angle_derivative(n0, n1, edge_unit, dn0_x2, dn1_x2, sin_angle, cos_angle)
    g3 = angle_derivative(n0, n1, edge_unit, dn0_x3, dn1_x3, sin_angle, cos_angle)
    return angle, g0, g1, g2, g3, True


# kernels_deformable.py:178-185 (_wrapped_angle_difference). Edits: none.
@wp.func
def wrapped_angle_difference(angle: float, rest_angle: float) -> float:
    difference = angle - rest_angle
    if difference > wp.pi:
        difference = difference - 2.0 * wp.pi
    elif difference < -wp.pi:
        difference = difference + 2.0 * wp.pi
    return difference


# Ported from kernels_deformable.py:226-261 (_warp_project_dihedral_jacobi). Edits: as distance_xpbd;
# alpha = 1/(k * rest edge length * dt^2) (compliance 1/(ke * |x_l - x_k|), solver_xpbd_rod.py:519-551).
@wp.func
def dihedral_xpbd(
    env: wp.int32,
    c: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    dihedrals: wp.array2d(dtype=wp.vec4i),
    rest_angle: wp.array2d(dtype=wp.float32),
    edge_len: wp.array2d(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    relaxation: wp.array(dtype=wp.float32),
    count: wp.int32,
    dt: wp.float32,
    lambdas: wp.array2d(dtype=wp.float32),
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
):
    k = stiffness[env]
    if k <= 0.0:
        return
    v = variant[env]
    bend = dihedrals[v, c]
    base = env * count
    li = base + bend[0]
    lj = base + bend[1]
    lk = base + bend[2]
    ll = base + bend[3]
    i = bend[0]
    j = bend[1]
    kk = bend[2]
    fourth_vertex = bend[3]
    angle, g0, g1, g2, g3, valid = dihedral_angle_gradients(q[env, i], q[env, j], q[env, kk], q[env, fourth_vertex])
    if valid:
        alpha = 1.0 / (k * wp.max(edge_len[v, c], 1.0e-12) * dt * dt)  # a zero-length hinge would make alpha inf and the multiplier NaN
        wi = inv_mass[env, i]
        wj = inv_mass[env, j]
        wk = inv_mass[env, kk]
        wl = inv_mass[env, fourth_vertex]
        denom = wi * wp.dot(g0, g0) + wj * wp.dot(g1, g1) + wk * wp.dot(g2, g2) + wl * wp.dot(g3, g3) + alpha
        if denom > 0.0:
            constraint = wrapped_angle_difference(angle, rest_angle[v, c])
            dlambda = (-constraint - alpha * lambdas[env, c]) / denom
            lambdas[env, c] = lambdas[env, c] + dlambda
            scale = relaxation[env] * dlambda
            wp.atomic_add(delta, li, g0 * (wi * scale))
            wp.atomic_add(delta, lj, g1 * (wj * scale))
            wp.atomic_add(delta, lk, g2 * (wk * scale))
            wp.atomic_add(delta, ll, g3 * (wl * scale))
            wp.atomic_add(delta_count, li, 1)
            wp.atomic_add(delta_count, lj, 1)
            wp.atomic_add(delta_count, lk, 1)
            wp.atomic_add(delta_count, ll, 1)


# One launch per sweep for both constraint kinds (they accumulate into the same buffers, each into its own multiplier
# array, and read only the positions the apply writes afterwards): threads [0, S) are edges, [S, S + D) hinges.
@wp.kernel
def solve_shell_xpbd(
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    edges: wp.array2d(dtype=wp.vec2i),
    edge_rest: wp.array2d(dtype=wp.float32),
    stretch: wp.array(dtype=wp.float32),
    dihedrals: wp.array2d(dtype=wp.vec4i),
    rest_angle: wp.array2d(dtype=wp.float32),
    edge_len: wp.array2d(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    relaxation: wp.array(dtype=wp.float32),
    S: wp.int32,
    count: wp.int32,
    dt: wp.float32,
    lambda_s: wp.array2d(dtype=wp.float32),
    lambda_d: wp.array2d(dtype=wp.float32),
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
):
    env, i = wp.tid()
    if i < S:
        distance_xpbd(env, i, q, inv_mass, variant, edges, edge_rest, stretch, relaxation, count, dt, lambda_s, delta, delta_count)
    else:
        dihedral_xpbd(env, i - S, q, inv_mass, variant, dihedrals, rest_angle, edge_len, bend, relaxation, count, dt, lambda_d,
                      delta, delta_count)
