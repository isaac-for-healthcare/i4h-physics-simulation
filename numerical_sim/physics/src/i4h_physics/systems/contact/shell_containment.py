# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rod nodes kept inside a deforming shell vessel, with the wall pushed back (two-way).

Ported from omniendo xcath/xcath_solver.py:1044-1190 (``_project_coupled_vessel_containment_kernel``: the
inverse-mass-weighted split of a point/triangle penetration between the rod node and the three wall vertices,
with the response fading near pinned vertices), :900-926 (``_classify_collision_sides_kernel``, the two-sided
side history) and :1216-1224 (the wall apply). omniendo queried one ``wp.Mesh`` (``mesh_query_point_sign_normal``)
refit after every apply; ``wp.Mesh`` has no grouped point query and its refit allocates, so this system follows
``fluid_contact``: one grouped ``wp.Bvh`` over every env's triangle boxes (group = env, tight boxes from the live
wall), refit before each projection, a candidate loop per node with ``closest_triangle_barycentric``, and the
face normal for the side (omniendo's query sign; ``smooth_normals`` is not ported). Edits: ``(env, node)``
prologue; positions are ``scene.q`` slices of the two bodies; the target clearance is the node's radius; the
wall corrections accumulate in a body-local array applied by a second launch (a sum, as the source: the
mass-normalised split already resolves the penetration); products ``contact_depth`` / ``contact_count`` per env
(batched_xcath.py:339-341); resets from batched_xcath.py:439-447, 557-565. The one-way kernel's outward-velocity
clamp (:1028-1041) is not ported: ``two_way: false`` runs the coupled kernel with a zero response, which the
source tests assert equals it.

List this system after ``catheter_control`` (the track projection comes first, omniendo's post stage) and after
``shell`` (the wall's own constraints ran this iteration before the boxes are refit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_bool, check_int
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System
from i4h_physics.systems.contact.geometry import closest_triangle_barycentric
from i4h_physics.systems.mechanics.connectors import apply_direct


@dataclass(eq=False)
class ShellContainment(System):
    state_products = ("contact_depth", "contact_count")  # per-env products a host observes (System.state_products)
    body: str  # rod body (any body: the single-node tests use a plain body)
    shell: str  # shell body (vessel_mesh with a deformable block)
    sign: float = -1.0  # live, per env: -1 keeps the rod on the side the face normals point away from (inward-facing wall)
    target_response: float = 0.5  # live, per env: fraction of the penetration the wall takes (0 = rigid one-way)
    max_distance: float = 0.1  # live, per env: activation shell (omniendo containment.max_distance) and the query half-extent
    iterations: int = 1  # structural: projections per runner iteration, the wall refit between them
    two_sided: bool = False  # structural: preserve each node's side from the substep start (omniendo sidedness two_sided)
    two_way: bool = True  # structural: push the wall (omniendo containment.two_way)

    structural = System.structural + ("iterations", "two_sided", "two_way")

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        self.body = b = scene.body(self.body)
        self.shell = sh = scene.body(self.shell)
        if sh is b:
            raise ValueError("shell_containment: body and shell must be different bodies")
        if sh.shell is None:
            raise ValueError(f"shell_containment: body {sh.name!r} has no shell record")
        if sh.shell.tris.shape[0] != 1:
            raise ValueError("shell_containment: one mesh variant only")
        self.iterations = check_int("shell_containment.iterations", self.iterations, 1)
        self.two_sided = check_bool("shell_containment.two_sided", self.two_sided)
        self.two_way = check_bool("shell_containment.two_way", self.two_way)
        P, n = b.count, sh.count
        F = self.F = sh.shell.tris.shape[1]
        scene.check_memory({"boxes": E * F * 24, "groups": E * F * 4, "wall_delta": E * n * 12, "sides": E * P * 4})
        self.q_rod = scene.field("q", b)  # (E, P) and (E, n) views: one launch reads the wall and writes the rod
        self.q_shell = scene.field("q", sh)
        self.q_prev_rod = scene.field("q_prev", b)
        self.tris = wp.array(sh.shell.tris, dtype=wp.vec3i, device=dev)  # (V, F)
        self.variant = sh.variant
        self.lowers = wp.zeros(E * F, dtype=wp.vec3, device=dev)
        self.uppers = wp.zeros(E * F, dtype=wp.vec3, device=dev)
        self.tri_group = wp.array(np.repeat(np.arange(E, dtype=np.int32), F), dtype=wp.int32, device=dev)
        self.sides = wp.full(E * P, -1.0, dtype=wp.float32, device=dev)  # two-sided history: -1 inside, +1 outside
        self.wall_delta = wp.zeros(E * n, dtype=wp.vec3, device=dev)  # body-local, self-clearing
        self.contact_depth = wp.zeros(E, dtype=wp.float32, device=dev)  # deepest penetration of the last substep
        self.contact_count = wp.zeros(E, dtype=wp.int32, device=dev)  # nodes pushed in the last substep
        self._live("sign", 'target_response', "max_distance")
        self._bounds(scene)
        self.bvh = wp.Bvh(self.lowers, self.uppers, groups=self.tri_group)

    def _bounds(self, scene):
        wp.launch(shell_bounds, dim=(scene.envs, self.F),
                  inputs=[self.q_shell, self.tris, self.variant],
                  outputs=[self.lowers, self.uppers], device=scene.device)

    def _refit(self, scene):
        self._bounds(scene)
        self.bvh.refit()

    def readouts(self):
        return {"contact_depth": self.contact_depth, "contact_count": self.contact_count}


    def substep_start(self, scene, dt, s, substeps):
        self.contact_depth.zero_()
        self.contact_count.zero_()
        if self.two_sided:  # xcath_solver.py:2513-2526: classify once per substep from the current positions
            self._refit(scene)
            b = self.body
            wp.launch(classify_sides, dim=(scene.envs, b.count),
                      inputs=[self.q_prev_rod, scene.field("inv_mass", b), scene.field("flags", b), self.bvh.id, self.q_shell,
                              self.tris, self.variant, self.sign, self.max_distance, b.count, self.F],
                      outputs=[self.sides], device=scene.device)

    def iteration(self, scene, dt, k, iterations):
        b, sh, E, dev = self.body, self.shell, scene.envs, scene.device
        for sweep in range(self.iterations):
            if sweep == 0 or self.two_way:  # the shell system moved the wall earlier in this iteration; within the sweeps
                self._refit(scene)  # only the wall apply moves it, and only when two-way
            wp.launch(project_coupled, dim=(E, b.count),
                      inputs=[self.q_shell, self.q_prev_rod, scene.field("inv_mass", b), scene.field("flags", b),
                              scene.field("radius", b), scene.field("inv_mass", sh), self.bvh.id, self.tris, self.variant,
                              self.sides, self.sign, self.target_response, self.max_distance, b.count, sh.count, self.F,
                              int(self.two_sided), int(self.two_way)],
                      outputs=[self.q_rod, self.wall_delta, self.contact_depth, self.contact_count], device=dev)
            if self.two_way:
                # xcath_solver.py:1216-1224 (_apply_deformable_collision_corrections_kernel): connectors' self-clearing
                # apply; a pinned vertex (zero inverse mass) gets no share of the wall correction in project_coupled
                wp.launch(apply_direct, dim=(E, sh.count), inputs=[scene.field("flags", sh), sh.count],
                          outputs=[self.q_shell, self.wall_delta], device=dev)

    def reset(self, scene, mask):
        self.clear(mask, (self.sides, wp.float32(-1.0)))
        self.clear(mask, (self.contact_depth, wp.float32(0.0)))
        self.clear(mask, (self.contact_count, wp.int32(0)))


@wp.kernel
def shell_bounds(
    q: wp.array2d(dtype=wp.vec3),
    tris: wp.array2d(dtype=wp.vec3i),
    variant: wp.array(dtype=wp.int32),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    """Tight box of every wall triangle (fluid_contact.triangle_bounds without the shell growth and the dead-slot
    park: a shell has no dead triangles). The query box is the node's own search extent."""
    env, f = wp.tid()
    tri = tris[variant[env], f]
    i = env * tris.shape[1] + f
    p0 = q[env, tri[0]]
    p1 = q[env, tri[1]]
    p2 = q[env, tri[2]]
    lowers[i] = wp.min(wp.min(p0, p1), p2)
    uppers[i] = wp.max(wp.max(p0, p1), p2)


@wp.func
def nearest_triangle(
    bvh: wp.uint64, env: wp.int32, p: wp.vec3, half: wp.float32, q: wp.array2d(dtype=wp.vec3),
    tris: wp.array2d(dtype=wp.vec3i), v: wp.int32, F: wp.int32,
):
    """The wall triangle of ``env`` whose closest point to ``p`` is nearest, within ``half``: ``(found, face,
    barycentric, closest point, unit face normal)``. Degenerate triangles are skipped (omniendo _mesh_face_normal
    fell back to a fixed normal). Replaces omniendo's ``mesh_query_point_sign_normal``."""
    best = float(1.0e30)
    found = int(0)
    face = int(-1)
    bary = wp.vec3(0.0)
    closest = wp.vec3(0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)
    h = wp.vec3(half)
    query = wp.bvh_query_aabb(bvh, p - h, p + h, wp.bvh_get_group_root(bvh, env))
    item = int(0)
    while wp.bvh_query_next(query, item):
        f = item - env * F
        tri = tris[v, f]
        p0 = q[env, tri[0]]
        p1 = q[env, tri[1]]
        p2 = q[env, tri[2]]
        c = wp.cross(p1 - p0, p2 - p0)
        c_len = wp.length(c)
        if c_len <= 1.0e-12:
            continue
        b = closest_triangle_barycentric(p, p0, p1, p2)
        cp = p0 * b[0] + p1 * b[1] + p2 * b[2]
        d2 = wp.length_sq(p - cp)
        if d2 < best and d2 <= half * half:
            best = d2
            found = 1
            face = f
            bary = b
            closest = cp
            normal = c / c_len
    return found, face, bary, closest, normal


# Ported from xcath_solver.py:900-926 (_classify_collision_sides_kernel). Edits: (env, node) prologue; the
# candidate loop replaces the mesh query; the face-normal side replaces the query sign.
@wp.kernel
def classify_sides(
    q_prev_rod: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    bvh: wp.uint64,
    q_shell: wp.array2d(dtype=wp.vec3),
    tris: wp.array2d(dtype=wp.vec3i),
    variant: wp.array(dtype=wp.int32),
    sign: wp.array(dtype=wp.float32),
    max_distance: wp.array(dtype=wp.float32),
    P: wp.int32,
    F: wp.int32,
    sides: wp.array(dtype=wp.float32),
):
    env, local = wp.tid()
    flat_index = env * P + local
    sides[flat_index] = -1.0
    if (flags[env, local] & ACTIVE) == 0 or inv_mass[env, local] <= 0.0:
        return
    p = q_prev_rod[env, local]
    found, face, bary, closest, normal = nearest_triangle(bvh, env, p, max_distance[env], q_shell, tris, variant[env], F)
    if found == 0:
        return
    diff = p - closest
    sign_face = 1.0
    if wp.dot(diff, normal) < 0.0:
        sign_face = -1.0
    # A point exactly on the surface deterministically belongs to the interior.
    if wp.length(diff) > 1.0e-6 and sign_face * sign[env] > 0.0:
        sides[flat_index] = 1.0


# Ported from xcath_solver.py:1044-1190 (_project_coupled_vessel_containment_kernel). Edits: (env, node) prologue;
# the candidate loop replaces the mesh query, the face normal the query sign (smooth normals not ported); the
# two-sided fallback queries from the substep-start position (q_prev) and recomputes the closest point of the
# predicted one on that face; target_phi = -radius[node]; the wall corrections go to a body-local accumulator;
# ``two_way`` off is a zero response; contact products per env (batched_xcath.py:339-341).
@wp.kernel
def project_coupled(
    q_shell: wp.array2d(dtype=wp.vec3),
    q_prev_rod: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    radius: wp.array2d(dtype=wp.float32),
    inv_mass_shell: wp.array2d(dtype=wp.float32),
    bvh: wp.uint64,
    tris: wp.array2d(dtype=wp.vec3i),
    variant: wp.array(dtype=wp.int32),
    sides: wp.array(dtype=wp.float32),
    sign: wp.array(dtype=wp.float32),
    target_response: wp.array(dtype=wp.float32),
    max_distance: wp.array(dtype=wp.float32),
    P: wp.int32,
    n: wp.int32,
    F: wp.int32,
    two_sided: wp.int32,
    two_way: wp.int32,
    q_rod: wp.array2d(dtype=wp.vec3),
    wall_delta: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    flat_index = env * P + local
    i = local
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        return
    pos = q_rod[env, i]
    v = variant[env]
    half = max_distance[env]
    found, face, bary, closest, normal = nearest_triangle(bvh, env, pos, half, q_shell, tris, v, F)
    from_current = int(0)
    if found == 0:
        if two_sided == 0:
            return
        found, face, bary, closest, normal = nearest_triangle(bvh, env, q_prev_rod[env, i], half, q_shell, tris, v, F)
        if found == 0:
            return
        tri_c = tris[v, face]
        p0 = q_shell[env, tri_c[0]]
        p1 = q_shell[env, tri_c[1]]
        p2 = q_shell[env, tri_c[2]]
        bary = closest_triangle_barycentric(pos, p0, p1, p2)
        closest = p0 * bary[0] + p1 * bary[1] + p2 * bary[2]
        from_current = 1

    resolve_wall_contact(env, i, flat_index, pos, closest, normal, bary, tris[v, face], from_current,
                         sign, radius, sides, inv_mass_shell, target_response, n, two_sided, two_way,
                         q_rod, wall_delta, contact_depth, contact_count)


# Shared unchanged shell response equations, also used for live tetrahedral walls.
@wp.func
def resolve_wall_contact(
    env: int, i: int, flat_index: int, pos: wp.vec3, closest: wp.vec3, normal: wp.vec3, bary: wp.vec3,
    tri: wp.vec3i, from_current: int,
    sign: wp.array(dtype=wp.float32), radius: wp.array2d(dtype=wp.float32),
    sides: wp.array(dtype=wp.float32), inv_mass_shell: wp.array2d(dtype=wp.float32),
    target_response: wp.array(dtype=wp.float32), n: int, two_sided: int, two_way: int,
    q_rod: wp.array2d(dtype=wp.vec3), wall_delta: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=wp.float32), contact_count: wp.array(dtype=wp.int32),
):
    diff = pos - closest
    dist = wp.length(diff)
    sign_face = 1.0
    if wp.dot(diff, normal) < 0.0:
        sign_face = -1.0
    side = sign_face * sign[env]

    grad = wp.vec3(0.0)
    if from_current == 1:
        grad = normal * sign[env]
    elif dist > 1.0e-8:
        grad = (diff / dist) * side
    else:
        grad = normal * side

    r = radius[env, i]
    preserved_side = float(-1.0)
    penetration = float(0.0)
    if two_sided == 1:
        preserved_side = sides[flat_index]
        penetration = r - preserved_side * wp.dot(diff, grad)
    else:
        phi = side * dist
        penetration = phi + r  # target_phi = -radius
    if penetration <= 0.0:
        return

    m0 = inv_mass_shell[env, tri[0]]
    m1 = inv_mass_shell[env, tri[1]]
    m2 = inv_mass_shell[env, tri[2]]
    i0 = env * n + tri[0]  # the body-local wall accumulator
    i1 = env * n + tri[1]
    i2 = env * n + tri[2]

    # Pinned barycentric support fades the requested response continuously (xcath_solver.py:1158-1168).
    support = float(0.0)
    if m0 > 0.0:
        support = support + bary[0]
    if m1 > 0.0:
        support = support + bary[1]
    if m2 > 0.0:
        support = support + bary[2]
    effective_response = float(0.0)
    if two_way == 1:
        effective_response = target_response[env] * wp.clamp(support, 0.0, 1.0)

    rod_direction = -grad
    if two_sided == 1:
        rod_direction = grad * preserved_side
    q_rod[env, i] = pos + rod_direction * penetration * (1.0 - effective_response)
    wp.atomic_max(contact_depth, env, penetration)
    wp.atomic_add(contact_count, env, 1)
    if effective_response <= 0.0:
        return

    denominator = bary[0] * bary[0] * m0 + bary[1] * bary[1] * m1 + bary[2] * bary[2] * m2
    if denominator <= 1.0e-12:
        q_rod[env, i] = pos + rod_direction * penetration  # no supported wall motion: the rod absorbs everything
        return
    wd = -rod_direction * penetration * effective_response
    wp.atomic_add(wall_delta, i0, wd * (bary[0] * m0 / denominator))
    wp.atomic_add(wall_delta, i1, wd * (bary[1] * m1 / denominator))
    wp.atomic_add(wall_delta, i2, wd * (bary[2] * m2 / denominator))
