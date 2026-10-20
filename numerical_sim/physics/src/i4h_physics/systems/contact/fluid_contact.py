# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fluid particles against a tissue body's live skin: one-way, particle-owned.

omnisurg (fluids/surface_contact.py + kernels/surface_contact.py) built a ``wp.Bvh`` per surface, cached
up to 64 candidate triangles per target particle, accumulated closest-point pushes with atomics and
averaged them in an apply pass. Here one grouped ``wp.Bvh`` holds every env's live triangles (one box
per slot of ``surface.tris``, group = env; a dead slot is a point box that the contact kernel skips),
refit once per substep with the boxes grown by the fluid's shell and ``search``, and each fluid particle
thread queries its env's subtree, tests the candidates inline with omnisurg's numerics
(``_accumulate_one_way_contact``, kernels/surface_contact.py:146-209), averages the pushes in registers
and writes its own position: no candidate buffer, no atomics, no apply. Always the exterior side of the
skin (omnisurg's ``side`` key is dropped: the only interior surface, the cavity, is ``static_contact``).
The tissue is not pushed back (omnisurg was one-way too); friction is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, due
from i4h_physics.systems.contact.geometry import closest_triangle_barycentric
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_physics.systems.fluids.fluid_kernels import pool_slot
from i4h_physics.systems.mechanics.surface import Surface


@dataclass(eq=False)
class FluidContact(System):
    fluid: Fluid
    surface: Surface
    margin: float = 0.002  # live, per env: shell beyond the particle radius (omnisurg particle_margin)
    stiffness: float = 1.0  # live, per env: fraction of the penetration corrected per event
    max_correction: float = 0.0  # live, per env; 0 = unlimited
    iterations: int = 2  # structural: contact events per substep (omnisurg 4; each is a BVH traversal per particle)
    search: float = 0.01  # structural: slack on the triangle boxes for tissue motion within the substep

    refs = ("fluid", "surface")
    structural = System.structural + ("iterations", "search")

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        if not isinstance(self.fluid, Fluid):
            raise TypeError("fluid_contact.fluid must name a fluid system")
        if not isinstance(self.surface, Surface):
            raise TypeError("fluid_contact.surface must name a surface system")
        t = self.target = self.surface.body
        F = self.F = self.surface.tris.shape[1]
        scene.check_memory({"boxes": E * F * 24, "groups": E * F * 4})
        self.q_t = scene.field("q", t)  # the tissue body (E, count); the pools are the fluid's span
        self.lowers = wp.zeros(E * F, dtype=wp.vec3, device=dev)
        self.uppers = wp.zeros(E * F, dtype=wp.vec3, device=dev)
        self.tri_group = wp.array(np.repeat(np.arange(E, dtype=np.int32), F), dtype=wp.int32, device=dev)
        self.grow = max(float(b.rest.radius.max()) for b in self.fluid.bodies) + float(self.search)  # host rest data, no readback
        self._live("margin", "stiffness", "max_correction")
        self._bounds(scene)
        self.bvh = wp.Bvh(self.lowers, self.uppers, groups=self.tri_group)
        self.freeze("search")


    def _bounds(self, scene):
        wp.launch(triangle_bounds, dim=(scene.envs, self.F),
                  inputs=[self.q_t, self.surface.tris, self.grow, self.margin],
                  outputs=[self.lowers, self.uppers], device=scene.device)

    def substep_start(self, scene, dt, s, substeps):
        self._bounds(scene)  # the tissue was predicted by ``particles`` earlier in the list
        self.bvh.refit()

    def iteration(self, scene, dt, k, iterations):
        if due(k, self.iterations, iterations)[0] < 0:
            return
        fl, c = self.fluid, self.fluid.const
        wp.launch(push_out, dim=(scene.envs, fl.total),
                  inputs=[self.bvh.id, self.q_t, self.surface.tris, self.F, fl.radius_span, fl.flags_span,
                          self.margin, self.stiffness, self.max_correction, c["prefix"], c["counts"], c["base"], fl.nf],
                  outputs=[fl.q_span], device=scene.device)


@wp.kernel
def triangle_bounds(
    q: wp.array2d(dtype=wp.vec3),
    tris: wp.array2d(dtype=wp.vec3i),
    grow: wp.float32,
    margin: wp.array(dtype=wp.float32),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    """Box of each live triangle slot grown by the contact shell and the search slack. A dead slot (padding, a
    removed triangle, a free cut slot) is a point box parked far away at its own offset: hundreds of identical
    points would degenerate the tree (refit measured 12x slower with them stacked)."""
    env, f = wp.tid()
    tri = tris[env, f]
    i = env * tris.shape[1] + f
    if tri[0] == tri[1]:
        park = wp.vec3(1.0e4 + wp.float32(i), 0.0, 0.0)
        lowers[i] = park
        uppers[i] = park
        return
    p0 = q[env, tri[0]]
    p1 = q[env, tri[1]]
    p2 = q[env, tri[2]]
    r = wp.vec3(grow + margin[env])
    lowers[i] = wp.min(wp.min(p0, p1), p2) - r
    uppers[i] = wp.max(wp.max(p0, p1), p2) + r


# omnisurg/fluids/kernels/surface_contact.py:146-209 (_accumulate_one_way_contact) with omnisurg's ``side != 0``
# (exterior) branch baked in: the constraint keeps the particle on the negative side of ``constraint_normal``, so
# the skin's outward normal is negated to keep it outside. Returns the push instead of atomically accumulating it.
# The ``shell`` acceptance is omnisurg's: a particle deeper than the shell is lost (accepting a "crossed" particle
# from farther away was tried and flings drops resting near an edge, which lie on the inner side of a side face's
# plane; a substep of travel must stay within the shell, so cases bound the fluid's ``max_velocity``).
@wp.func
def one_way_contact(
    point: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, shell: wp.float32, stiffness: wp.float32, maximum: wp.float32
) -> wp.vec3:
    zero = wp.vec3(0.0, 0.0, 0.0)
    outward = wp.cross(p1 - p0, p2 - p0)
    normal_length = wp.length(outward)
    if normal_length != normal_length or normal_length <= 1.0e-12:
        return zero
    constraint_normal = -outward / normal_length
    barycentric = closest_triangle_barycentric(point, p0, p1, p2)
    triangle_point = p0 * barycentric[0] + p1 * barycentric[1] + p2 * barycentric[2]
    offset = triangle_point - point
    distance = wp.length(offset)
    if distance != distance or distance >= shell:
        return zero
    side_distance = wp.dot(point - triangle_point, constraint_normal)
    crossed = side_distance > 1.0e-7
    penetration = shell - distance
    response_normal = constraint_normal
    if crossed:
        penetration = shell + distance
    elif distance > 1.0e-12:
        radial = offset / distance
        if wp.dot(radial, constraint_normal) >= 0.0:
            response_normal = radial
    correction_depth = penetration * stiffness
    if maximum > 0.0:
        correction_depth = wp.min(correction_depth, maximum)
    if correction_depth != correction_depth or correction_depth <= 0.0:
        return zero
    return -response_normal * correction_depth


@wp.kernel
def push_out(
    bvh: wp.uint64,
    q_t: wp.array2d(dtype=wp.vec3),
    tris: wp.array2d(dtype=wp.vec3i),
    F: wp.int32,
    radius: wp.array(dtype=wp.float32),
    flags: wp.array(dtype=wp.int32),
    margin: wp.array(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    max_correction: wp.array(dtype=wp.float32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nf: wp.int32,
    q: wp.array(dtype=wp.vec3),
):
    """omnisurg apply_fluid_surface_contact_deltas: the mean of the candidate pushes, here in registers. One launch
    over every pool (``pool_slot``); ``q``, ``radius`` and ``flags`` are the fluid span."""
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    li = base[ps[0]] + env * counts[ps[0]] + ps[1]
    if (flags[li] & ACTIVE) == 0:
        return
    p = q[li]
    shell = radius[li] + margin[env]
    half = wp.vec3(shell)
    total = wp.vec3(0.0, 0.0, 0.0)
    hits = int(0)
    query = wp.bvh_query_aabb(bvh, p - half, p + half, wp.bvh_get_group_root(bvh, env))
    item = int(0)
    while wp.bvh_query_next(query, item):
        tri = tris[env, item - env * F]
        if tri[0] == tri[1]:
            continue
        push = one_way_contact(p, q_t[env, tri[0]], q_t[env, tri[1]], q_t[env, tri[2]], shell, stiffness[env],
                               max_correction[env])
        if wp.length_sq(push) > 0.0:
            total += push
            hits += 1
    if hits > 0:
        q[li] = p + total / wp.float32(hits)
