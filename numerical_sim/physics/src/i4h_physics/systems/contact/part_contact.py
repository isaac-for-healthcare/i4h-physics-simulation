# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-way normal contact between disjoint parts of one tet body.

One grouped BVH over the selected Surface slots, refitted at every contact event.
Overflow particles bypass their truncated buffers and traverse the complete query.
All responses read the same positions; a separate launch averages and clears deltas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.loaders.tet_bundle import TET_FACES
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, due
from i4h_physics.systems.contact.geometry import closest_triangle_barycentric
from i4h_physics.systems.mechanics.surface import Surface


@dataclass(eq=False)
class PartContact(System):
    surface: Surface
    surface_part: str
    particle_parts: tuple[str, ...]
    margin: float = 0.002
    stiffness: float = 1.0
    max_correction: float = 0.0
    iterations: int = 4
    max_candidates: int = 64

    refs = ("surface",)
    structural = System.structural + ("iterations", "max_candidates", "surface_part", "particle_parts")

    def __post_init__(self):
        if not isinstance(self.surface, Surface) or self.surface.scene is not self.scene:
            raise TypeError("part_contact.surface must name a surface system in the same scene")
        self.body = b = self.surface.body
        if isinstance(self.particle_parts, str) or not self.particle_parts:
            raise ValueError("part_contact.particle_parts must be a nonempty list of parts")
        self.particle_parts = tuple(self.particle_parts)
        if len(set(self.particle_parts)) != len(self.particle_parts):
            raise ValueError("part_contact: repeated target parts")
        if self.surface_part in self.particle_parts:
            raise ValueError("part_contact: source/target overlap")
        for name in (self.surface_part, *self.particle_parts):
            if name not in b.parts:
                raise ValueError(f"part_contact: missing part {name!r}")
        self._validate_parameters()
        source = b.parts[self.surface_part]
        validate_boundary(b, source)
        targets = np.concatenate([np.arange(b.parts[n].particle_start,
                                            b.parts[n].particle_start + b.parts[n].particle_count, dtype=np.int32)
                                  for n in self.particle_parts])
        if not len(targets):
            raise ValueError("part_contact: target parts must contain particles")
        E, dev = self.scene.envs, self.scene.device
        self.F, self.N = source.tri_count, len(targets)
        self.face_start = source.tri_start
        self.scene.check_memory({"contact boxes and BVH": E * self.F * 100,
                                 "contact candidates": E * self.N * (self.max_candidates + 1) * 4,
                                 "contact deltas": E * b.count * 16})
        self.targets = wp.array(targets, dtype=wp.int32, device=dev)
        self.lowers = wp.zeros(E * self.F, dtype=wp.vec3, device=dev)
        self.uppers = wp.zeros_like(self.lowers)
        self.groups = wp.array(np.repeat(np.arange(E, dtype=np.int32), self.F), dtype=wp.int32, device=dev)
        self.candidates = wp.zeros((E, self.N * self.max_candidates), dtype=wp.int32, device=dev)
        # Counts greater than capacity select the fallback; no truncated response is applied.
        self.candidate_count = wp.zeros((E, self.N), dtype=wp.int32, device=dev)
        self.delta = wp.zeros((E, b.count), dtype=wp.vec3, device=dev)
        self.delta_count = wp.zeros((E, b.count), dtype=wp.int32, device=dev)
        self.contact_count = wp.zeros(E, dtype=wp.int32, device=dev)
        self.overflow_count = wp.zeros(E, dtype=wp.int32, device=dev)
        self.q = self.scene.field("q", b)
        self.flags = self.scene.field("flags", b)
        self.radius = self.scene.field("radius", b)
        self.inv_mass = self.scene.field("inv_mass", b)
        self._live("margin", "stiffness", "max_correction")
        self._bounds()
        self.bvh = wp.Bvh(self.lowers, self.uppers, groups=self.groups)
        self.freeze("surface_part", "particle_parts", "max_candidates")

    def _validate_parameters(self):
        for name in ("iterations", "max_candidates"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0:
                raise ValueError(f"part_contact.{name} must be a positive integer")
        for name in ("margin", "stiffness", "max_correction"):
            x = getattr(self, name)
            expected = (self.scene.envs,) if isinstance(x, wp.array) else ()
            x = x.numpy() if isinstance(x, wp.array) else np.asarray(x)
            if not np.isfinite(x).all() or (x < 0).any() or (name == "stiffness" and (x > 1).any()):
                raise ValueError(f"part_contact.{name} must be finite and nonnegative (stiffness <= 1)")
            if x.shape != expected:
                raise ValueError(f"part_contact.{name} must be a scalar at construction")

    def prepare(self, scene):
        super().prepare(scene)
        self._validate_parameters()
        self.zero(None, "delta", "delta_count", "candidate_count", "contact_count", "overflow_count")


    def _bounds(self):
        s = self.surface
        wp.launch(contact_bounds, dim=(self.scene.envs, self.F),
                  inputs=[self.q, self.flags, s.support, s.tris, s.tri_tet, s.active, self.face_start, self.F],
                  outputs=[self.lowers, self.uppers], device=self.scene.device)

    def step_start(self, scene, dt):
        self.zero(None, "contact_count", "overflow_count")

    def iteration(self, scene, dt, k, iterations):
        if due(k, self.iterations, iterations)[0] < 0:
            return
        s = self.surface
        self._bounds()
        self.bvh.refit()
        wp.launch(query_candidates, dim=(scene.envs, self.N),
                  inputs=[self.bvh.id, self.q, self.radius, self.flags, s.support, s.tris, s.tri_tet, s.active,
                          self.targets, self.margin, self.face_start, self.F, self.max_candidates],
                  outputs=[self.candidates, self.candidate_count, self.overflow_count], device=scene.device)
        common = [self.q, self.radius, self.inv_mass, self.flags, s.support, s.tris, s.tri_tet, s.active,
                  self.targets, self.margin, self.stiffness, self.max_correction,
                  self.face_start, self.F, self.max_candidates, self.candidate_count]
        outputs = [self.delta, self.delta_count, self.contact_count]
        wp.launch(accumulate_candidates, dim=(scene.envs, self.N, self.max_candidates),
                  inputs=[*common, self.candidates], outputs=outputs, device=scene.device)
        wp.launch(accumulate_overflow, dim=(scene.envs, self.N),
                  inputs=[*common, self.bvh.id], outputs=outputs, device=scene.device)
        wp.launch(apply_deltas, dim=self.delta.shape, inputs=[],
                  outputs=[self.q, self.delta, self.delta_count], device=scene.device)

    def reset(self, scene, mask):
        self.zero(mask, "delta", "delta_count", "candidates", "candidate_count", "contact_count", "overflow_count")


# Ported from omnisurg/tet/runtime/surface_contact.py:440-500; validate every padded variant.
def validate_boundary(body, part):
    m = body.tet
    for v in range(m.tets.shape[0]):
        selected = (m.part == part.index) & (m.active0[v] != 0)
        volumes = m.rest_volume[v, selected]
        if not len(volumes) or not np.isfinite(volumes).all() or (volumes <= 0).any():
            raise ValueError("part_contact: source requires live tets with positive rest volumes")
        faces = {}
        for tet in m.tets[v, selected]:
            for corners in TET_FACES:
                face = tuple(int(i) for i in tet[corners])
                faces.setdefault(tuple(sorted(face)), []).append(face)
        if any(len(owners) > 2 for owners in faces.values()):
            raise ValueError("part_contact: non-manifold source tet faces")
        boundary = {key: owners[0] for key, owners in faces.items() if len(owners) == 1}
        authored = {}
        for tri in m.tris[v, part.tri_start:part.tri_start + part.tri_count]:
            if tri[0] == tri[1]:  # reserved and padded slots
                continue
            face = tuple(int(i) for i in tri)
            key = tuple(sorted(face))
            if key in authored:
                raise ValueError("part_contact: duplicate source triangles")
            authored[key] = face
        if authored.keys() != boundary.keys():
            raise ValueError("part_contact: authored triangles do not match source tet boundary")
        for key, face in authored.items():
            a, b, c = boundary[key]
            if face not in ((a, b, c), (b, c, a), (c, a, b)):
                raise ValueError("part_contact: source requires outward authored triangle winding")


@wp.func
def particle_live(env: int, i: int, flags: wp.array2d(dtype=wp.int32), support: wp.array(dtype=wp.int32)) -> bool:
    return (flags[env, i] & ACTIVE) != 0 and support[env * flags.shape[1] + i] > 0


@wp.func
def face_live(env: int, f: int, flags: wp.array2d(dtype=wp.int32), support: wp.array(dtype=wp.int32),
              tris: wp.array2d(dtype=wp.vec3i), owners: wp.array2d(dtype=wp.int32),
              active: wp.array2d(dtype=wp.int32)) -> bool:
    owner = owners[env, f]
    if owner < 0 or active[env, owner] == 0:
        return False
    tri = tris[env, f]
    return (tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2]
            and particle_live(env, tri[0], flags, support)
            and particle_live(env, tri[1], flags, support)
            and particle_live(env, tri[2], flags, support))


@wp.kernel
def contact_bounds(q: wp.array2d(dtype=wp.vec3), flags: wp.array2d(dtype=wp.int32), support: wp.array(dtype=wp.int32),
                   tris: wp.array2d(dtype=wp.vec3i), owners: wp.array2d(dtype=wp.int32), active: wp.array2d(dtype=wp.int32),
                   start: int, F: int, lowers: wp.array(dtype=wp.vec3), uppers: wp.array(dtype=wp.vec3)):
    env, local = wp.tid()
    i, f = env * F + local, start + local
    park = wp.vec3(1.0e4 + wp.float32(i), 0.0, 0.0)
    lo, hi = park, park
    if face_live(env, f, flags, support, tris, owners, active):
        tri = tris[env, f]
        a, b, c = q[env, tri[0]], q[env, tri[1]], q[env, tri[2]]
        lo, hi = wp.min(wp.min(a, b), c), wp.max(wp.max(a, b), c)
    lowers[i] = lo
    uppers[i] = hi


# omnisurg/tet/kernels/surface_contact.py:110-173; grouped, live-owner-aware query, count capacity+1 on overflow.
@wp.kernel
def query_candidates(bvh: wp.uint64, q: wp.array2d(dtype=wp.vec3), radius: wp.array2d(dtype=float),
                     flags: wp.array2d(dtype=wp.int32), support: wp.array(dtype=wp.int32),
                     tris: wp.array2d(dtype=wp.vec3i), owners: wp.array2d(dtype=wp.int32), active: wp.array2d(dtype=wp.int32),
                     targets: wp.array(dtype=wp.int32), margin: wp.array(dtype=float), start: int, F: int, capacity: int,
                     candidates: wp.array2d(dtype=wp.int32), counts: wp.array2d(dtype=wp.int32),
                     overflows: wp.array(dtype=wp.int32)):
    env, slot = wp.tid()
    target = targets[slot]
    counts[env, slot] = 0
    if not particle_live(env, target, flags, support):
        return
    shell = radius[env, target] + margin[env]
    if shell <= 0.0:
        return
    p, h = q[env, target], wp.vec3(shell)
    query = wp.bvh_query_aabb(bvh, p - h, p + h, wp.bvh_get_group_root(bvh, env))
    item, count = int(0), int(0)
    while wp.bvh_query_next(query, item):
        f = start + item - env * F
        if not face_live(env, f, flags, support, tris, owners, active):
            continue
        if count == capacity:
            count += 1
            wp.atomic_add(overflows, env, 1)
            break
        candidates[env, slot * capacity + count] = f
        count += 1
    counts[env, slot] = count


# Ported from omnisurg/tet/kernels/surface_contact.py:230-340; exterior side, shared body-local accumulators.
@wp.func
def respond(
    env: int, target: int, face: int,
    q: wp.array2d(dtype=wp.vec3), radius: wp.array2d(dtype=float), inv_mass: wp.array2d(dtype=float),
    flags: wp.array2d(dtype=wp.int32), support: wp.array(dtype=wp.int32), tris: wp.array2d(dtype=wp.vec3i),
    owners: wp.array2d(dtype=wp.int32), active: wp.array2d(dtype=wp.int32),
    margin: wp.array(dtype=float), stiffness: wp.array(dtype=float), maximum: wp.array(dtype=float),
    delta: wp.array2d(dtype=wp.vec3), delta_count: wp.array2d(dtype=wp.int32), contacts: wp.array(dtype=wp.int32),
):
    if not particle_live(env, target, flags, support) or not face_live(env, face, flags, support, tris, owners, active):
        return
    shell = radius[env, target] + margin[env]
    if shell <= 0.0:
        return
    vertices = tris[env, face]
    p0 = q[env, vertices[0]]
    p1 = q[env, vertices[1]]
    p2 = q[env, vertices[2]]
    outward = wp.cross(p1 - p0, p2 - p0)
    normal_length = wp.length(outward)
    if normal_length <= 1.0e-12:
        return
    outward = outward / normal_length
    constraint_normal = -outward

    point = q[env, target]
    barycentric = closest_triangle_barycentric(point, p0, p1, p2)
    triangle_point = (
        p0 * barycentric[0]
        + p1 * barycentric[1]
        + p2 * barycentric[2]
    )
    offset = triangle_point - point
    distance = wp.length(offset)
    if distance >= shell:
        return

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

    correction_depth = penetration * stiffness[env]
    limit = maximum[env]
    if limit > 0.0:
        correction_depth = wp.min(correction_depth, limit)
    if correction_depth <= 0.0:
        return

    target_weight = wp.max(inv_mass[env, target], 0.0)
    w0 = wp.max(inv_mass[env, vertices[0]], 0.0)
    w1 = wp.max(inv_mass[env, vertices[1]], 0.0)
    w2 = wp.max(inv_mass[env, vertices[2]], 0.0)
    denominator = (
        target_weight
        + w0 * barycentric[0] * barycentric[0]
        + w1 * barycentric[1] * barycentric[1]
        + w2 * barycentric[2] * barycentric[2]
    )
    if denominator <= 1.0e-12:
        return
    multiplier = correction_depth / denominator
    if target_weight > 0.0:
        wp.atomic_add(
            delta,
            env, target,
            -response_normal * multiplier * target_weight,
        )
        wp.atomic_add(delta_count, env, target, wp.int32(1))

    weights = wp.vec3(w0, w1, w2)
    for c in range(3):
        if weights[c] > 0.0:
            wp.atomic_add(delta, env, vertices[c], response_normal * multiplier * weights[c] * barycentric[c])
            wp.atomic_add(delta_count, env, vertices[c], 1)
    wp.atomic_add(contacts, env, 1)


@wp.kernel
def accumulate_candidates(
    q: wp.array2d(dtype=wp.vec3), radius: wp.array2d(dtype=float), inv_mass: wp.array2d(dtype=float),
    flags: wp.array2d(dtype=wp.int32), support: wp.array(dtype=wp.int32), tris: wp.array2d(dtype=wp.vec3i),
    owners: wp.array2d(dtype=wp.int32), active: wp.array2d(dtype=wp.int32),
    targets: wp.array(dtype=wp.int32),
    margin: wp.array(dtype=float), stiffness: wp.array(dtype=float), maximum: wp.array(dtype=float),
    start: int, F: int, capacity: int, counts: wp.array2d(dtype=wp.int32),
    candidates: wp.array2d(dtype=wp.int32),
    delta: wp.array2d(dtype=wp.vec3), delta_count: wp.array2d(dtype=wp.int32), contacts: wp.array(dtype=wp.int32),
):
    env, slot, candidate = wp.tid()
    count = counts[env, slot]
    if count > capacity or candidate >= count:
        return
    target = targets[slot]
    face = candidates[env, slot * capacity + candidate]
    respond(env, target, face, q, radius, inv_mass, flags, support, tris, owners, active,
                margin, stiffness, maximum, delta, delta_count, contacts)


@wp.kernel
def accumulate_overflow(
    q: wp.array2d(dtype=wp.vec3), radius: wp.array2d(dtype=float), inv_mass: wp.array2d(dtype=float),
    flags: wp.array2d(dtype=wp.int32), support: wp.array(dtype=wp.int32), tris: wp.array2d(dtype=wp.vec3i),
    owners: wp.array2d(dtype=wp.int32), active: wp.array2d(dtype=wp.int32),
    targets: wp.array(dtype=wp.int32),
    margin: wp.array(dtype=float), stiffness: wp.array(dtype=float), maximum: wp.array(dtype=float),
    start: int, F: int, capacity: int, counts: wp.array2d(dtype=wp.int32),
    bvh: wp.uint64,
    delta: wp.array2d(dtype=wp.vec3), delta_count: wp.array2d(dtype=wp.int32), contacts: wp.array(dtype=wp.int32),
):
    env, slot = wp.tid()
    if counts[env, slot] <= capacity:
        return
    target = targets[slot]
    p = q[env, target]
    h = wp.vec3(radius[env, target] + margin[env])
    query = wp.bvh_query_aabb(bvh, p - h, p + h, wp.bvh_get_group_root(bvh, env))
    item = int(0)
    while wp.bvh_query_next(query, item):
        face = start + item - env * F
        respond(env, target, face, q, radius, inv_mass, flags, support, tris, owners, active,
                margin, stiffness, maximum, delta, delta_count, contacts)


# omnisurg/tet/kernels/surface_contact.py:344-385; merged disjoint source and target apply/clear passes.
@wp.kernel
def apply_deltas(q: wp.array2d(dtype=wp.vec3), delta: wp.array2d(dtype=wp.vec3), counts: wp.array2d(dtype=wp.int32)):
    env, i = wp.tid()
    n = counts[env, i]
    if n > 0:
        q[env, i] = q[env, i] + delta[env, i] / wp.float32(n)
    delta[env, i] = wp.vec3(0.0)
    counts[env, i] = 0
