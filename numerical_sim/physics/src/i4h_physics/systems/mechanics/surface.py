# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live surface topology and deletion bookkeeping of one tet body.

Ported from omnisurg/tet/kernels/tet.py: tet deactivation 485-502 and 655-672 (atomic
exchange on the active flag, particle support counters, deletion counter), authored
re-emission 1393-1424 and exposed-face emission 1427-1470; omnisurg/tet/surface.py:20-128
ran the three launches after every deletion. Edits: a per-env ``(E, F)`` triangle buffer
over the loader's part-major capacity layout, where authored slots keep their index and go
degenerate when their tet dies and exposed faces fill the deletable part's reserved
``cut_faces`` slots by atomics with an overflow latch per env (omnisurg dropped silently and
shuffled every slot on each rebuild); particle and edge liveness follow the support counters
inside ``deactivate_tet`` (omnisurg swept the particles afterwards and gated edges at solve
time through an incidence list); no surface revision counter: consumers store the owning tet
and test ``active[env, tet]``. The surgical appearance sidecar carries the damage, coagulation and blood channels
(originally omnisurg's TissueVisualStateSidecar); this class owns only physical topology. Re-emission runs once per step in ``step_end``, gated per env on
``dirty``; the deleters (thermal, cutting, stretch_breaking) run in ``substep_end`` of the
same step, so the surface the contact systems and the renderer see next is current.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.scene import INACTIVE, restore_variant_rows
from i4h_physics.systems.base import System
from i4h_physics.systems.mechanics.tet import Tet


@dataclass(eq=False)
class Surface(System):
    state_products = ("deleted", "overflow")  # per-env products a host observes (System.state_products)
    tet: Tet  # YAML: id of the body's tet system; the live masks are shared with it

    refs = ("tet",)

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.tet, Tet):
            raise TypeError("surface.tet must name a tet system")
        self.body = b = self.tet.body
        m = b.tet
        E, dev = scene.envs, scene.device
        S, F, P = m.edges.shape[1], m.tris.shape[1], len(b.parts)
        n = E * b.count
        if F < P:
            raise ValueError(f"surface: body {b.name!r} has fewer triangle slots ({F}) than parts ({P})")
        scene.check_memory({
            "variants": m.tet_edges.nbytes + m.face_neighbor.nbytes + m.face_tri.nbytes + m.tris.nbytes + m.tri_tet.nbytes
                        + m.support0.nbytes + m.edge_support0.nbytes,
            "tris": E * F * 16,
            "support": n * 4 + E * S * 4,
        })
        t = self.tet
        # The same arrays as the tet system: the masks the deleters flip and the topology they read.
        self.variant, self.active, self.edge_active = t.variant, t.active, t.edge_active
        self.tets, self.tet_part, self.edges, self.edge_rest, self.edge_part = t.tets, t.tet_part, t.edges, t.edge_rest, t.edge_part
        def i32(a):
            return wp.array(a, dtype=wp.int32, device=dev)
        self.tet_edges = i32(m.tet_edges)  # (V, T, 6)
        self.face_neighbor = i32(m.face_neighbor)  # (V, T, 4)
        self.face_tri = i32(m.face_tri)  # (V, T, 4)
        self.tris0 = wp.array(m.tris, dtype=wp.vec3i, device=dev)  # (V, F) authored skin plus degenerate padding
        self.tri_tet0 = i32(m.tri_tet)  # (V, F)
        self.tri_part = i32(m.tri_part)  # (F,)
        self.support0 = i32(m.support0)  # (V, count)
        self.edge_support0 = i32(m.edge_support0)  # (V, S)
        self.heat_enabled = i32(m.heat)  # (P,) parts that conduct and burn
        self.deletable = i32(m.deletable)  # (P,)
        self.cut_start = i32(m.cut_start)  # (P,)
        self.cut_capacity = i32(m.cut_capacity)  # (P,)
        # Per-env live state.
        self.tris = wp.empty((E, F), dtype=wp.vec3i, device=dev)  # body-local particle indices
        self.tri_tet = wp.empty((E, F), dtype=wp.int32, device=dev)  # owning tet, -1 for degenerate slots
        self.support = wp.empty(n, dtype=wp.int32, device=dev)  # body-local (E*count,), live tets per particle
        self.edge_support = wp.empty((E, S), dtype=wp.int32, device=dev)
        self.cut_count = wp.zeros((E, P), dtype=wp.int32, device=dev)  # exposed faces per part after the last rebuild (may exceed the cut capacity: overflow latches)
        self.overflow = wp.zeros(E, dtype=wp.int32, device=dev)  # latched once a part ran out of cut slots
        self.deleted = wp.zeros(E, dtype=wp.int32, device=dev)  # tets removed since the last reset
        self.dirty = wp.zeros(E, dtype=wp.int32, device=dev)  # a deletion happened this step
        self.reset(scene, wp.ones(E, dtype=wp.bool, device=dev))

    def step_end(self, scene, dt):
        E, dev = scene.envs, scene.device
        F, T = self.tris.shape[1], self.tets.shape[1]
        wp.launch(rebuild_authored, dim=(E, F),
                  inputs=[self.dirty, self.variant, self.active, self.tris0, self.tri_tet0],
                  outputs=[self.tris, self.tri_tet, self.cut_count], device=dev)
        if T:
            wp.launch(emit_cut_faces, dim=(E, T, 4),
                      inputs=[self.dirty, self.variant, self.active, self.tets, self.tet_part, self.face_neighbor, self.face_tri,
                              self.tri_tet0, self.cut_start, self.cut_capacity],
                      outputs=[self.tris, self.tri_tet, self.cut_count, self.overflow], device=dev)
        self.dirty.zero_()  # capture-safe memset: the next step starts clean

    def reset(self, scene, mask):
        """Selected envs get their variant's authored surface, support counts and a clean visual state."""
        E, dev, count = scene.envs, scene.device, self.body.count
        for src, dst in ((self.tris0, self.tris), (self.tri_tet0, self.tri_tet), (self.support0, self.support.reshape((E, count))),
                         (self.edge_support0, self.edge_support)):
            if dst.shape[1]:
                wp.launch(restore_variant_rows, dim=dst.shape, inputs=[mask, self.variant, src], outputs=[dst], device=dev)
        zero = wp.int32(0)
        for value, dst in ((zero, self.cut_count), (zero, self.overflow.reshape((E, 1))), (zero, self.deleted.reshape((E, 1))),
                           (zero, self.dirty.reshape((E, 1)))):
            self.clear(mask, (dst, value))


@wp.func
def deactivate_tet(
    env: wp.int32,
    t: wp.int32,
    v: wp.int32,
    count: wp.int32,
    tets: wp.array2d(dtype=wp.vec4i),
    tet_edges: wp.array3d(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    edge_active: wp.array2d(dtype=wp.int32),
    support: wp.array(dtype=wp.int32),
    edge_support: wp.array2d(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    deleted: wp.array(dtype=wp.int32),
    dirty: wp.array(dtype=wp.int32),
):
    """tet.py:655-672: remove tet ``t`` of ``env`` once (the atomic exchange makes many threads safe).
    The thread that takes a particle's or an edge's last supporting tet clears its flag, so every
    transition has exactly one writer."""
    if wp.atomic_exch(active, env, t, wp.int32(0)) == 0:
        return
    ids = tets[v, t]
    for c in range(4):
        flat_index = env * count + ids[c]
        if wp.atomic_add(support, flat_index, wp.int32(-1)) == 1:
            flags[env, ids[c]] = flags[env, ids[c]] & INACTIVE
    for e in range(6):
        s = tet_edges[v, t, e]
        if s >= 0:
            if wp.atomic_add(edge_support, env, s, wp.int32(-1)) == 1:
                edge_active[env, s] = wp.int32(0)
    wp.atomic_add(deleted, env, wp.int32(1))
    dirty[env] = wp.int32(1)


@wp.func
def tet_face(ids: wp.vec4i, face: wp.int32) -> wp.vec3i:
    """tet.py:55-86: corners of the four outward faces of a positively oriented tet."""
    if face == 0:
        return wp.vec3i(ids[1], ids[2], ids[3])
    if face == 1:
        return wp.vec3i(ids[0], ids[3], ids[2])
    if face == 2:
        return wp.vec3i(ids[0], ids[1], ids[3])
    return wp.vec3i(ids[0], ids[2], ids[1])


# tet.py:1034-1058 and 1393-1424 folded: authored slots are rewritten in place (live -> the authored
# triangle, dead owner -> degenerate), which also clears the cut region and the exposed-face counts.
@wp.kernel
def rebuild_authored(
    dirty: wp.array(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    tris0: wp.array2d(dtype=wp.vec3i),
    tri_tet0: wp.array2d(dtype=wp.int32),
    tris: wp.array2d(dtype=wp.vec3i),
    tri_tet: wp.array2d(dtype=wp.int32),
    cut_count: wp.array2d(dtype=wp.int32),
):
    env, f = wp.tid()
    if dirty[env] == 0:
        return
    if f < cut_count.shape[1]:
        cut_count[env, f] = 0
    v = variant[env]
    tri = tris0[v, f]
    owner = tri_tet0[v, f]
    if owner >= 0 and active[env, owner] == 0:
        tri = wp.vec3i(tri[0], tri[0], tri[0])
        owner = -1
    tris[env, f] = tri
    tri_tet[env, f] = owner


# tet.py:1427-1470 (emit_tet_cut_surface_triangles). Edits: (env, tet, face) prologue; the slot comes
# from the part's reserved cut region; an exhausted region latches overflow instead of dropping quietly.
@wp.kernel
def emit_cut_faces(
    dirty: wp.array(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    tets: wp.array2d(dtype=wp.vec4i),
    tet_part: wp.array(dtype=wp.int32),
    face_neighbor: wp.array3d(dtype=wp.int32),
    face_tri: wp.array3d(dtype=wp.int32),
    tri_tet0: wp.array2d(dtype=wp.int32),
    cut_start: wp.array(dtype=wp.int32),
    cut_capacity: wp.array(dtype=wp.int32),
    tris: wp.array2d(dtype=wp.vec3i),
    tri_tet: wp.array2d(dtype=wp.int32),
    cut_count: wp.array2d(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    env, t, face = wp.tid()
    if dirty[env] == 0 or active[env, t] == 0:
        return
    v = variant[env]
    authored = face_tri[v, t, face]
    if authored >= 0:
        owner = tri_tet0[v, authored]
        if owner < 0 or active[env, owner] != 0:
            return  # the authored slot shows this face
    n = face_neighbor[v, t, face]
    if n < 0 or active[env, n] != 0:
        return  # boundary, or still interior
    part = tet_part[t]
    slot = wp.atomic_add(cut_count, env, part, wp.int32(1))
    if slot >= cut_capacity[part]:
        overflow[env] = wp.int32(1)
        return
    f = cut_start[part] + slot
    tris[env, f] = tet_face(tets[v, t], face)
    tri_tet[env, f] = t
