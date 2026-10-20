# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hook engagement: the perpendicular limb pierces the engageable part and retains it.

Ported from omnisurg/instruments/hook_kernels.py: anchor store 202-258, paired exit anchor
260-467 (a tet walk from the entry point along the contact normal until a boundary face),
acquisition from the jaw-right capsule's contact records 470-731, retention shell with axial
stick-slip and release 733-1064, the two-slot driver 1066-1202 and the apply 1205-1220.
Edits: anchors are ``(E, 2)`` arrays (slot 0 entry, slot 1 exit) with body-local vertex
indices; the eligible part is tested through ``tri_part`` and ``tet_part``; anchors store
their owning tet and drop when cutting or burning removes it (omnisurg rescanned the surface
on a revision bump); the exit face's vertices and barycentrics come from the tet's corner
table and the ray hit (omnisurg searched the live triangle list for that face); the entry
triangle is written into the contact system's suppression slot so the capsule stops pushing
it (kernels.py:3156-3170); one thread per env owns both slots, so the corrections are applied
in place instead of through an accumulator and an apply launch (omnisurg averaged them with
the contact deltas). Runs after ``instrument_contact`` in the iteration (omnisurg ran it
before); acquisition at every substep end. Tool force (M11): the in-place corrections are exact, so
their mass-weighted sum goes straight into ``impulse (E, n)`` at the hook's column for ``tool_force``.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System
from i4h_physics.systems.mechanics.surface import tet_face
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.instrument_control import PRIM_KINDS, InstrumentControl

MAX_WALK = 128  # tets the exit search may cross (hook_kernels.py:355 walked up to the part's tet count)


@dataclass(eq=False)
class Hook(System):
    instruments: InstrumentControl
    contacts: InstrumentContact
    instrument: str  # instrument name (kind hook)
    part: str  # engageable part of the contact body (chole: fat)
    engage_overlap: float = 0.001  # live, per env
    engage_inward_speed: float = 0.02  # live, per env
    retention_slack: float = 0.003  # live, per env
    axial_friction: float = 0.3  # live, per env
    puncture_distance: float = 0.020  # live, per env: how far past the entry the exit anchor may lie (paired_puncture_max_distance)

    refs = ("instruments", "contacts")
    structural = System.structural + ("instrument", "part")  # both are baked into the launches as indices

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.instruments, InstrumentControl) or not isinstance(self.contacts, InstrumentContact):
            raise TypeError("hook.instruments must name an instrument_control system and hook.contacts an instrument_contact system")
        ic, cc = self.instruments, self.contacts
        E, dev = scene.envs, scene.device
        if self.instrument not in ic.names:
            raise ValueError(f"hook: unknown instrument {self.instrument!r}; known: {ic.names}")
        self.tool = k = ic.names.index(self.instrument)
        kinds, starts, counts = ic.host_prim_kind, ic.host_prim_start, ic.host_prim_counts
        limbs = [p for p in range(starts[k], starts[k] + counts[k]) if kinds[p] == PRIM_KINDS["jaw_right"]]  # the hook's limb
        if len(limbs) != 1:
            raise ValueError(f"hook: instrument {self.instrument!r} needs exactly one jaw_right chain")
        self.prim = int(limbs[0])
        self.body = b = cc.body
        self.surface = cc.surface
        if self.part not in b.parts:
            raise ValueError(f"hook.part {self.part!r}: unknown part of body {b.name!r}")
        self.part_index = b.parts[self.part].index
        self.freeze("instrument", "part")
        self._live("engage_overlap", "engage_inward_speed", "retention_slack", "axial_friction", "puncture_distance")
        # Anchors (E, 2): slot 0 the entry (owns the engagement), slot 1 the exit on the far side of the pierced
        # part. Triangle (body-local vertices, owning tet), barycentric, axial and friction coordinates, radial.
        self.verts = wp.full((E, 2), wp.vec3i(-1, -1, -1), dtype=wp.vec3i, device=dev)
        self.tet = wp.full((E, 2), -1, dtype=wp.int32, device=dev)
        self.bary = wp.zeros((E, 2), dtype=wp.vec3, device=dev)
        self.friction_coord = wp.zeros((E, 2), dtype=wp.float32, device=dev)
        self.radial = wp.zeros((E, 2), dtype=wp.vec3, device=dev)
        self.active = wp.zeros((E, 2), dtype=wp.int32, device=dev)
        self.entry_active = self.active[:, 0]  # (E,) views for thermal and observations
        self.entry_tet = self.tet[:, 0]
        self.impulse = wp.zeros((E, ic.count), dtype=wp.vec3, device=dev)  # product: sum of m * dx the anchors imposed this step
        self.moment = wp.zeros((E, ic.count), dtype=wp.vec3, device=dev)  # product: its moment about the tool root (coupling)
        if cc.suppress_prim != -1:
            raise ValueError("hook: the contact system already has a hook suppression slot (one hook per instrument_contact)")
        cc.suppress_prim = self.prim  # the contact kernel reads the entry anchor through these views: no sync launch
        cc.suppress_active = self.active[:, 0]
        cc.suppress_verts = self.verts[:, 0]
        self._substeps = 1


    def step_start(self, scene, dt):
        self.impulse.zero_()
        self.moment.zero_()

    def substep_start(self, scene, dt, s, substeps):
        self._substeps = substeps

    def iteration(self, scene, dt, k, iterations):
        ic, b = self.instruments, self.body
        wp.launch(
            retain_hook,
            dim=scene.envs,
            inputs=[scene.field("q_prev", b), scene.field("inv_mass", b), scene.field("flags", b), self.surface.active, ic.prim_a,
                    ic.prim_b, ic.sphere_q_prev, ic.sphere_q, ic.sphere_radius, ic.sphere_active, self.prim, self.retention_slack,
                    self.axial_friction, self.engage_inward_speed, dt * float(self._substeps), dt, self.tool, self.tet,
                    self.verts, self.bary, ic.root],
            outputs=[self.friction_coord, self.radial, self.active, scene.field("q", b), self.impulse, self.moment],
            device=scene.device,
        )

    def substep_end(self, scene, dt):
        ic, cc, sf, b = self.instruments, self.contacts, self.surface, self.body
        wp.launch(
            acquire_hook,
            dim=scene.envs,
            inputs=[scene.field("q", b), scene.field("q_prev", b), scene.field("flags", b), sf.active, sf.variant, sf.tets, sf.tet_part, sf.face_neighbor, sf.tri_part,
                    ic.prim_a, ic.prim_b, ic.sphere_q_prev, ic.sphere_q, ic.sphere_active, cc.record_count, cc.rec_tri, cc.rec_tet,
                    cc.rec_verts, cc.rec_bary, cc.rec_pen, cc.rec_t, cc.rec_normal, self.prim, self.part_index,
                    self.engage_overlap, self.engage_inward_speed, self.puncture_distance, dt * float(self._substeps), dt],
            outputs=[self.verts, self.tet, self.bary, self.friction_coord, self.radial, self.active],
            device=scene.device,
        )

    def prepare(self, scene):
        super().prepare(scene)
        if not self.enabled:  # no longer zeroed at step start: its last contribution would stay in the tool force
            self.zero(None, "impulse", "moment")
            self.active.zero_()  # contact suppression aliases this transient anchor flag

    def reset(self, scene, mask):
        self.zero(mask, "impulse", "moment")
        for value, dst in ((wp.int32(0), self.active), (wp.vec3i(-1, -1, -1), self.verts), (wp.int32(-1), self.tet)):
            self.clear(mask, (dst, value))


@wp.func
def perpendicular(axis: wp.vec3):
    direction = wp.cross(axis, wp.vec3(1.0, 0.0, 0.0))
    if wp.length(direction) <= 1.0e-8:
        direction = wp.cross(axis, wp.vec3(0.0, 1.0, 0.0))
    return wp.normalize(direction)


@wp.func
def same_triangle(first: wp.vec3i, second: wp.vec3i) -> wp.int32:
    """hook_kernels.py:156-166: the same three vertices in any order."""
    for a in range(3):
        found = wp.int32(0)
        for b in range(3):
            if first[a] == second[b]:
                found = wp.int32(1)
        if found == 0:
            return wp.int32(0)
    return wp.int32(1)


@wp.func
def ray_triangle(origin: wp.vec3, direction: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, minimum: wp.float32) -> wp.vec4:
    """hook_kernels.py:169-199: barycentric weights of the hit and its distance, -1 distance on a miss."""
    edge1 = p1 - p0
    edge2 = p2 - p0
    pvec = wp.cross(direction, edge2)
    determinant = wp.dot(edge1, pvec)
    if wp.abs(determinant) <= 1.0e-10:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    inverse = 1.0 / determinant
    tvec = origin - p0
    second = wp.dot(tvec, pvec) * inverse
    qvec = wp.cross(tvec, edge1)
    third = wp.dot(direction, qvec) * inverse
    first = 1.0 - second - third
    distance = wp.dot(edge2, qvec) * inverse
    if distance <= minimum or first < -1.0e-5 or second < -1.0e-5 or third < -1.0e-5:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    return wp.vec4(first, second, third, distance)


@wp.func
def store_anchor(
    env: wp.int32,
    slot: wp.int32,
    v: wp.vec3i,
    w: wp.vec3,
    t: wp.int32,
    tissue: wp.vec3,
    hook_a: wp.vec3,
    axis: wp.vec3,
    axis_length: wp.float32,
    hint: wp.vec3,
    verts: wp.array2d(dtype=wp.vec3i),
    tet: wp.array2d(dtype=wp.int32),
    bary: wp.array2d(dtype=wp.vec3),
    friction_coord: wp.array2d(dtype=wp.float32),
    radial: wp.array2d(dtype=wp.vec3),
    active: wp.array2d(dtype=wp.int32),
):
    """hook_kernels.py:202-258: the anchor's limb coordinate and radial direction (the hint when the tissue
    point sits on the axis)."""
    coordinate = wp.dot(tissue - hook_a, axis)
    center = hook_a + axis * wp.clamp(coordinate, 0.0, axis_length)
    direction = tissue - center
    if wp.length(direction) > 1.0e-8:
        direction = wp.normalize(direction)
    else:
        direction = hint - axis * wp.dot(hint, axis)
        if wp.length(direction) > 1.0e-8:
            direction = wp.normalize(direction)
        else:
            direction = perpendicular(axis)
    verts[env, slot] = v
    tet[env, slot] = t
    bary[env, slot] = w
    friction_coord[env, slot] = coordinate
    radial[env, slot] = direction
    active[env, slot] = 1


@wp.func
def acquire_exit(
    env: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    flags: wp.array2d(dtype=wp.int32),
    tet_active: wp.array2d(dtype=wp.int32),
    v_index: wp.int32,
    tets: wp.array2d(dtype=wp.vec4i),
    tet_part: wp.array(dtype=wp.int32),
    face_neighbor: wp.array3d(dtype=wp.int32),
    part: wp.int32,
    entry_normal: wp.vec3,
    maximum_distance: wp.float32,
    hook_a: wp.vec3,
    axis: wp.vec3,
    axis_length: wp.float32,
    verts: wp.array2d(dtype=wp.vec3i),
    tet: wp.array2d(dtype=wp.int32),
    bary: wp.array2d(dtype=wp.vec3),
    friction_coord: wp.array2d(dtype=wp.float32),
    radial: wp.array2d(dtype=wp.vec3),
    active: wp.array2d(dtype=wp.int32),
):
    """hook_kernels.py:260-467: from the entry point, march along the contact normal (flipped into the entry
    tet) through face neighbours of the engageable part; the first face with no live neighbour of that part
    within maximum_distance becomes the exit anchor."""
    if maximum_distance <= 0.0:
        return
    entry_verts = verts[env, 0]
    entry_tet = tet[env, 0]
    if entry_tet < 0 or tet_active[env, entry_tet] == 0 or tet_part[entry_tet] != part:
        return
    ids = tets[v_index, entry_tet]
    entry_face = wp.int32(-1)
    for face in range(4):
        if same_triangle(tet_face(ids, face), entry_verts) != 0:
            entry_face = face
    if entry_face < 0:
        return
    for c in range(4):
        if (flags[env, ids[c]] & ACTIVE) == 0:
            return
    q0 = q[env, ids[0]]
    q1 = q[env, ids[1]]
    q2 = q[env, ids[2]]
    q3 = q[env, ids[3]]
    if wp.abs(wp.dot(q1 - q0, wp.cross(q2 - q0, q3 - q0))) <= 1.0e-14:
        return
    w = bary[env, 0]
    origin = q[env, entry_verts[0]] * w[0] + q[env, entry_verts[1]] * w[1] + q[env, entry_verts[2]] * w[2]
    if wp.length(entry_normal) <= 1.0e-8:
        return
    direction = wp.normalize(entry_normal)
    inward = wp.dot(direction, (q0 + q1 + q2 + q3) * 0.25 - origin)
    if wp.abs(inward) <= 1.0e-8:
        return
    if inward < 0.0:
        direction = -direction

    current = entry_tet
    current_distance = wp.float32(0.0)
    for _step in range(MAX_WALK):
        if current < 0 or tet_active[env, current] == 0 or tet_part[current] != part:
            return
        ids = tets[v_index, current]
        selected_face = wp.int32(-1)
        selected_distance = maximum_distance + 1.0
        selected_hit = wp.vec4(0.0, 0.0, 0.0, -1.0)
        for face in range(4):
            fv = tet_face(ids, face)
            hit = ray_triangle(origin, direction, q[env, fv[0]], q[env, fv[1]], q[env, fv[2]], current_distance + 1.0e-6)
            if hit[3] >= 0.0 and hit[3] < selected_distance:
                selected_face = face
                selected_distance = hit[3]
                selected_hit = hit
        if selected_face < 0 or selected_distance > maximum_distance:
            return
        neighbor = face_neighbor[v_index, current, selected_face]
        if neighbor >= 0 and tet_active[env, neighbor] != 0 and tet_part[neighbor] == part:
            current = neighbor
            current_distance = selected_distance
            continue
        fv = tet_face(ids, selected_face)
        for c in range(3):
            if (flags[env, fv[c]] & ACTIVE) == 0:
                return
        weights = wp.vec3(selected_hit[0], selected_hit[1], selected_hit[2])
        tissue = q[env, fv[0]] * weights[0] + q[env, fv[1]] * weights[1] + q[env, fv[2]] * weights[2]
        store_anchor(env, 1, fv, weights, current, tissue, hook_a, axis, axis_length, direction,
                     verts, tet, bary, friction_coord, radial, active)
        return


@wp.kernel
def acquire_hook(
    q: wp.array2d(dtype=wp.vec3),
    q_prev: wp.array2d(dtype=wp.vec3),
    flags: wp.array2d(dtype=wp.int32),
    tet_active: wp.array2d(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    tets: wp.array2d(dtype=wp.vec4i),
    tet_part: wp.array(dtype=wp.int32),
    face_neighbor: wp.array3d(dtype=wp.int32),
    tri_part: wp.array(dtype=wp.int32),
    prim_a: wp.array(dtype=wp.int32),
    prim_b: wp.array(dtype=wp.int32),
    sphere_q_prev: wp.array2d(dtype=wp.vec3),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_active: wp.array2d(dtype=wp.int32),
    record_count: wp.array2d(dtype=wp.int32),
    rec_tri: wp.array3d(dtype=wp.int32),
    rec_tet: wp.array3d(dtype=wp.int32),
    rec_verts: wp.array3d(dtype=wp.vec3i),
    rec_bary: wp.array3d(dtype=wp.vec3),
    rec_pen: wp.array3d(dtype=wp.float32),
    rec_t: wp.array3d(dtype=wp.float32),
    rec_normal: wp.array3d(dtype=wp.vec3),
    prim: wp.int32,
    part: wp.int32,
    engage_overlap: wp.array(dtype=wp.float32),
    engage_inward_speed: wp.array(dtype=wp.float32),
    puncture_distance: wp.array(dtype=wp.float32),
    frame_dt: wp.float32,
    dt: wp.float32,
    verts: wp.array2d(dtype=wp.vec3i),
    tet: wp.array2d(dtype=wp.int32),
    bary: wp.array2d(dtype=wp.vec3),
    friction_coord: wp.array2d(dtype=wp.float32),
    radial: wp.array2d(dtype=wp.vec3),
    active: wp.array2d(dtype=wp.int32),
):
    """hook_kernels.py:470-731: the deepest record of the limb's distal half (t in [0.5, 1]) on the engageable
    part, moving into the tissue faster than engage_inward_speed, becomes the entry anchor; then the exit
    anchor is searched behind it."""
    env = wp.tid()
    if active[env, 0] != 0:
        return
    active[env, 1] = 0
    a = prim_a[prim]
    b = prim_b[prim]
    if sphere_active[env, a] == 0 or sphere_active[env, b] == 0:
        return
    hook_a = sphere_q[env, a]
    hook_b = sphere_q[env, b]
    hook_a_prev = sphere_q_prev[env, a]
    hook_b_prev = sphere_q_prev[env, b]
    axis_vector = hook_b - hook_a
    axis_length = wp.length(axis_vector)
    if axis_length <= 1.0e-8:
        return
    deepest = engage_overlap[env]
    selected = wp.int32(-1)
    for r in range(wp.min(record_count[env, prim], rec_tri.shape[2])):
        t = rec_t[env, prim, r]
        if t < 0.5 or t > 1.0:
            continue
        penetration = rec_pen[env, prim, r]
        if penetration < deepest:
            continue
        if tri_part[rec_tri[env, prim, r]] != part:
            continue
        owner = rec_tet[env, prim, r]
        if owner < 0 or tet_active[env, owner] == 0:
            continue
        v = rec_verts[env, prim, r]
        i0 = v[0]  # body-local
        i1 = v[1]
        i2 = v[2]
        if (flags[env, i0] & ACTIVE) == 0 or (flags[env, i1] & ACTIVE) == 0 or (flags[env, i2] & ACTIVE) == 0:
            continue
        p0 = q[env, i0]
        p1 = q[env, i1]
        p2 = q[env, i2]
        if wp.length_sq(wp.cross(p1 - p0, p2 - p0)) <= 1.0e-16:
            continue
        w = rec_bary[env, prim, r]
        if w[0] < -1.0e-4 or w[1] < -1.0e-4 or w[2] < -1.0e-4 or wp.abs(w[0] + w[1] + w[2] - 1.0) > 1.0e-3:
            continue
        tissue = p0 * w[0] + p1 * w[1] + p2 * w[2]
        tissue_prev = q_prev[env, i0] * w[0] + q_prev[env, i1] * w[1] + q_prev[env, i2] * w[2]
        hook_center = hook_a + axis_vector * t
        hook_center_prev = hook_a_prev + (hook_b_prev - hook_a_prev) * t
        normal = rec_normal[env, prim, r]
        if wp.length(normal) <= 1.0e-8:
            continue
        normal = wp.normalize(normal)
        relative_speed = wp.dot((hook_center - hook_center_prev) / frame_dt - (tissue - tissue_prev) / dt, normal)
        if relative_speed < engage_inward_speed[env]:
            continue
        if selected >= 0 and penetration <= deepest:
            continue
        deepest = penetration
        selected = r
    if selected < 0:
        return
    v = rec_verts[env, prim, selected]
    w = rec_bary[env, prim, selected]
    axis = axis_vector / axis_length
    tissue = q[env, v[0]] * w[0] + q[env, v[1]] * w[1] + q[env, v[2]] * w[2]
    store_anchor(env, 0, v, w, rec_tet[env, prim, selected], tissue, hook_a, axis, axis_length, rec_normal[env, prim, selected],
                 verts, tet, bary, friction_coord, radial, active)
    acquire_exit(env, q, flags, tet_active, variant[env], tets, tet_part, face_neighbor, part, rec_normal[env, prim, selected],
                 puncture_distance[env], hook_a, axis, axis_length, verts, tet, bary, friction_coord, radial, active)


@wp.func
def retain_anchor(
    env: wp.int32,
    slot: wp.int32,
    entry_owned: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    q_prev: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    tet_active: wp.array2d(dtype=wp.int32),
    hook_a: wp.vec3,
    hook_b: wp.vec3,
    hook_a_prev: wp.vec3,
    hook_b_prev: wp.vec3,
    radius: wp.float32,
    retention_slack: wp.float32,
    axial_friction: wp.float32,
    engage_inward_speed: wp.float32,
    frame_dt: wp.float32,
    dt: wp.float32,
    tool: wp.int32,
    tet: wp.array2d(dtype=wp.int32),
    verts: wp.array2d(dtype=wp.vec3i),
    bary: wp.array2d(dtype=wp.vec3),
    friction_coord: wp.array2d(dtype=wp.float32),
    radial: wp.array2d(dtype=wp.vec3),
    active: wp.array2d(dtype=wp.int32),
    impulse: wp.array2d(dtype=wp.vec3),
    root: wp.array2d(dtype=wp.vec3),
    moment: wp.array2d(dtype=wp.vec3),
) -> wp.int32:
    """hook_kernels.py:733-1064 for one slot: radial shell [r, r + slack] around the limb, axial stick-slip
    friction; only the entry (entry_owned) releases when the tissue slides past the limb end while
    withdrawing. The tool moved over the frame (sphere_q_prev is per frame), the tissue over the substep.
    The correction is written into ``q`` directly (this thread owns the env's anchors). Returns 1 while
    the anchor holds."""
    owner = tet[env, slot]
    if owner < 0 or tet_active[env, owner] == 0:
        active[env, slot] = 0  # the anchored triangle's tet was cut or burnt away
        return wp.int32(0)
    v = verts[env, slot]
    w = bary[env, slot]
    i0 = v[0]  # body-local
    i1 = v[1]
    i2 = v[2]
    if (flags[env, i0] & ACTIVE) == 0 or (flags[env, i1] & ACTIVE) == 0 or (flags[env, i2] & ACTIVE) == 0:
        active[env, slot] = 0
        return wp.int32(0)
    p0 = q[env, i0]
    p1 = q[env, i1]
    p2 = q[env, i2]
    if wp.length_sq(wp.cross(p1 - p0, p2 - p0)) <= 1.0e-16:
        active[env, slot] = 0
        return wp.int32(0)
    axis_vector = hook_b - hook_a
    axis_length = wp.length(axis_vector)
    axis = axis_vector / axis_length
    tissue = p0 * w[0] + p1 * w[1] + p2 * w[2]
    tissue_prev = q_prev[env, i0] * w[0] + q_prev[env, i1] * w[1] + q_prev[env, i2] * w[2]
    coordinate = wp.dot(tissue - hook_a, axis)
    if entry_owned != 0 and coordinate > axis_length + retention_slack and dt > 0.0:
        previous_axis = hook_b_prev - hook_a_prev
        if wp.length(previous_axis) > 1.0e-8:
            previous_axis = wp.normalize(previous_axis)
        else:
            previous_axis = axis
        tip_motion = hook_b - hook_b_prev
        withdrawal = wp.dot((tissue - tissue_prev) / dt - tip_motion / frame_dt, previous_axis)
        if withdrawal > engage_inward_speed:
            active[env, slot] = 0
            return wp.int32(0)

    center = hook_a + axis * wp.clamp(coordinate, 0.0, axis_length)
    offset = tissue - center
    distance = wp.length(offset)
    direction = radial[env, slot]
    if distance > 1.0e-8:
        direction = offset / distance
    else:
        direction = direction - axis * wp.dot(direction, axis)
        if wp.length(direction) > 1.0e-8:
            direction = wp.normalize(direction)
        else:
            direction = perpendicular(axis)
    radial[env, slot] = direction

    target = distance
    if distance < radius:
        target = radius
    elif distance > radius + retention_slack:
        target = radius + retention_slack
    radial_scalar = target - distance
    correction = direction * radial_scalar
    friction_error = friction_coord[env, slot] - coordinate
    friction_limit = axial_friction * wp.abs(radial_scalar)
    axial_correction = wp.clamp(friction_error, -friction_limit, friction_limit)
    correction = correction + axis * axial_correction
    if wp.abs(friction_error) > friction_limit:
        friction_coord[env, slot] = coordinate + axial_correction

    w0 = inv_mass[env, i0]
    w1 = inv_mass[env, i1]
    w2 = inv_mass[env, i2]
    denom = w0 * w[0] * w[0] + w1 * w[1] * w[1] + w2 * w[2] * w[2]
    if denom <= 1.0e-12:
        active[env, slot] = 0
        return wp.int32(0)
    if wp.length_sq(correction) > 0.0:
        scale = 1.0 / denom
        moved = wp.float32(0.0)  # sum of m_i * dx_i = correction * scale * sum of the moved vertices' barycentrics
        arm = wp.vec3(0.0, 0.0, 0.0)  # sum of the moved vertices' barycentric-weighted offsets from the root (the moment)
        r = root[env, tool]
        if w0 > 0.0 and w[0] > 0.0:
            q[env, i0] = p0 + correction * (scale * w0 * w[0])
            moved += w[0]
            arm += (p0 - r) * w[0]
        if w1 > 0.0 and w[1] > 0.0:
            q[env, i1] = p1 + correction * (scale * w1 * w[1])
            moved += w[1]
            arm += (p1 - r) * w[1]
        if w2 > 0.0 and w[2] > 0.0:
            q[env, i2] = p2 + correction * (scale * w2 * w[2])
            moved += w[2]
            arm += (p2 - r) * w[2]
        impulse[env, tool] = impulse[env, tool] + correction * (scale * moved)
        moment[env, tool] = moment[env, tool] + wp.cross(arm, correction * scale)
    return wp.int32(1)


@wp.kernel
def retain_hook(
    q_prev: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    tet_active: wp.array2d(dtype=wp.int32),
    prim_a: wp.array(dtype=wp.int32),
    prim_b: wp.array(dtype=wp.int32),
    sphere_q_prev: wp.array2d(dtype=wp.vec3),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_active: wp.array2d(dtype=wp.int32),
    prim: wp.int32,
    retention_slack: wp.array(dtype=wp.float32),
    axial_friction: wp.array(dtype=wp.float32),
    engage_inward_speed: wp.array(dtype=wp.float32),
    frame_dt: wp.float32,
    dt: wp.float32,
    tool: wp.int32,
    tet: wp.array2d(dtype=wp.int32),
    verts: wp.array2d(dtype=wp.vec3i),
    bary: wp.array2d(dtype=wp.vec3),
    root: wp.array2d(dtype=wp.vec3),
    friction_coord: wp.array2d(dtype=wp.float32),
    radial: wp.array2d(dtype=wp.vec3),
    active: wp.array2d(dtype=wp.int32),
    q: wp.array2d(dtype=wp.vec3),
    impulse: wp.array2d(dtype=wp.vec3),
    moment: wp.array2d(dtype=wp.vec3),
):
    """hook_kernels.py:1066-1202: the entry first; losing it drops the exit too; the exit only drops itself."""
    env = wp.tid()
    if active[env, 0] == 0:
        active[env, 1] = 0
        return
    a = prim_a[prim]
    b = prim_b[prim]
    if sphere_active[env, a] == 0 or sphere_active[env, b] == 0:
        active[env, 0] = 0
        active[env, 1] = 0
        return
    hook_a = sphere_q[env, a]
    hook_b = sphere_q[env, b]
    if wp.length(hook_b - hook_a) <= 1.0e-8:
        active[env, 0] = 0
        active[env, 1] = 0
        return
    radius = wp.max(sphere_radius[a], sphere_radius[b])
    kept = retain_anchor(env, 0, 1, q, q_prev, inv_mass, flags, tet_active, hook_a, hook_b, sphere_q_prev[env, a], sphere_q_prev[env, b],
                         radius, retention_slack[env], axial_friction[env], engage_inward_speed[env], frame_dt, dt,
                         tool, tet, verts, bary, friction_coord, radial, active, impulse, root, moment)
    if kept == 0:
        active[env, 1] = 0
        return
    if active[env, 1] != 0:
        retain_anchor(env, 1, 0, q, q_prev, inv_mass, flags, tet_active, hook_a, hook_b, sphere_q_prev[env, a], sphere_q_prev[env, b],
                      radius, retention_slack[env], axial_friction[env], engage_inward_speed[env], frame_dt, dt,
                      tool, tet, verts, bary, friction_coord, radial, active, impulse, root, moment)
