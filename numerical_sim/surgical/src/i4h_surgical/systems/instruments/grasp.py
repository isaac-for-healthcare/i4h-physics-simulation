# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface grasp: a closed jaw holds the tissue points its capsules were touching.

Ported from omnisurg/instruments/kernels.py: capture 854-1156 (one thread per instrument
scanning its jaw capsules' contact records, deepest first), parallel transport 1158-1196,
revalidation 1198-1256, barycentric constraint 1258-1453, apply 1471-1493, and the
release/arm logic of system.py:1053-1074 (jaw crossing ``grasp_threshold``).
Edits: (env, instrument) and (env, slot) prologues; records are per capsule ``(E, P, R)``
with body-local vertex indices; slots per instrument are jaw capsules x records; a capture
is armed when ``instrument_control`` raises ``grasp_pending`` and attempted at the end of
every step's last substep until something is held (system.py:1053-1074 kept the arm set);
``released`` clears the slots; a slot whose triangle lost its tet (cutting) is dropped.
Tool force (M11): the hold writes each tool's share of ``delta`` into ``delta_k`` (and signed by the capsule's jaw
side into ``delta_jaw``) and ``tool_delta.apply_average_tools`` sums the applied share times mass into ``impulse (E, n)`` and
``jaw_impulse (E, n)`` for ``tool_force``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, live
from i4h_physics.systems.contact.geometry import closest_point_segment_parameter
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.instrument_control import PRIM_KINDS, InstrumentControl
from i4h_surgical.systems.mechanics.tool_delta import apply_average_tools


@dataclass(eq=False)
class Grasp(System):
    instruments: InstrumentControl
    contacts: InstrumentContact

    refs = ("instruments", "contacts")

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.instruments, InstrumentControl) or not isinstance(self.contacts, InstrumentContact):
            raise TypeError("grasp.instruments must name an instrument_control system and grasp.contacts an instrument_contact system")
        ic, cc = self.instruments, self.contacts
        E, dev = scene.envs, scene.device
        n, R = ic.count, cc.rec_tri.shape[2]
        kinds, starts, counts = ic.host_prim_kind, ic.host_prim_start, ic.host_prim_counts
        jaw = (kinds == PRIM_KINDS["jaw_left"]) | (kinds == PRIM_KINDS["jaw_right"])
        jaw_prims = [int(jaw[starts[k] : starts[k] + counts[k]].sum()) for k in range(n)]
        self.slots = G = max([j * R for j in jaw_prims] + [1])
        self.body = cc.body
        self.surface = cc.surface
        self.stiffness = live(scene, [r.grasp_stiffness for r in scene.instruments.values()])  # (E, n)
        self.register_live("stiffness")
        self.grasps = wp.array(np.int32([r.kind == "grasper" for r in scene.instruments.values()]).reshape(n), dtype=wp.int32, device=dev)
        self.delta = wp.zeros(E * self.body.count, dtype=wp.vec3, device=dev)
        self.delta_count = wp.zeros(E * self.body.count, dtype=wp.int32, device=dev)
        self.delta_k = wp.zeros((E * self.body.count, max(n, 1)), dtype=wp.vec3, device=dev)  # per-tool share of delta
        self.delta_jaw = wp.zeros((E * self.body.count, max(n, 1)), dtype=wp.vec3, device=dev)  # signed by the jaw side
        self.impulse = wp.zeros((E, max(n, 1)), dtype=wp.vec3, device=dev)  # product: sum of m * dx the hold imposed this step
        self.jaw_impulse = wp.zeros((E, max(n, 1)), dtype=wp.vec3, device=dev)  # product: left minus right jaw share
        self.moment = wp.zeros((E, max(n, 1)), dtype=wp.vec3, device=dev)  # product: the impulse's moment about the tool root (coupling)
        # Slots (E, n, G): capsule, coordinate along it, radial direction, previous axis, triangle (vertices and
        # owning tet), barycentric, gap.
        self.prim = wp.full((E, n, G), -1, dtype=wp.int32, device=dev)
        self.tet = wp.full((E, n, G), -1, dtype=wp.int32, device=dev)
        self.t = wp.zeros((E, n, G), dtype=wp.float32, device=dev)
        self.radial = wp.zeros((E, n, G), dtype=wp.vec3, device=dev)
        self.prev_axis = wp.zeros((E, n, G), dtype=wp.vec3, device=dev)
        self.verts = wp.zeros((E, n, G), dtype=wp.vec3i, device=dev)
        self.bary = wp.zeros((E, n, G), dtype=wp.vec3, device=dev)
        self.gap = wp.zeros((E, n, G), dtype=wp.float32, device=dev)
        self.active = wp.zeros((E, n, G), dtype=wp.int32, device=dev)
        self.count = wp.zeros((E, n), dtype=wp.int32, device=dev)
        self.pending = wp.zeros((E, n), dtype=wp.int32, device=dev)  # armed capture per instrument
        self._substep, self._substeps = 0, 1


    def step_start(self, scene, dt):
        # ``released`` is the step's jaw-opening edge (set once per step by instrument_control), so one clear per step.
        self.impulse.zero_()
        self.jaw_impulse.zero_()
        self.moment.zero_()
        if self.instruments.count == 0:
            return
        wp.launch(release_grasps, dim=(scene.envs, self.instruments.count), inputs=[self.instruments.released, self.slots],
                  outputs=[self.active, self.prim, self.count, self.pending], device=scene.device)

    def substep_start(self, scene, dt, s, substeps):
        self._substep, self._substeps = s, substeps

    def iteration(self, scene, dt, k, iterations):
        ic, b = self.instruments, self.body
        if ic.count == 0:
            return
        alpha = float(self._substep * iterations + k + 1) / float(self._substeps * iterations)  # kernels.py:700-709
        wp.launch(
            hold_grasps,
            dim=(scene.envs, ic.count * self.slots),
            inputs=[scene.field("q", b), scene.field("inv_mass", b), scene.field("flags", b), self.surface.active, ic.prim_a,
                    ic.prim_b, ic.sphere_q_prev, ic.sphere_q, ic.sphere_radius, ic.sphere_jaw_sign, self.stiffness, alpha, iterations,
                    self.slots, b.count, self.tet,
                    self.prim, self.t, self.verts, self.bary, self.gap],
            outputs=[self.radial, self.prev_axis, self.active, self.delta, self.delta_count, self.delta_k, self.delta_jaw],
            device=scene.device,
        )
        wp.launch(apply_average_tools, dim=(scene.envs, b.count),
                  inputs=[scene.field("flags", b), scene.field("inv_mass", b), b.count, ic.count, ic.root],
                  outputs=[scene.field("q", b), self.delta, self.delta_count, self.delta_k, self.delta_jaw, self.impulse,
                           self.jaw_impulse, self.moment], device=scene.device)

    def substep_end(self, scene, dt):
        ic, cc, b = self.instruments, self.contacts, self.body
        if ic.count == 0:
            return
        wp.launch(
            revalidate_grasps,
            dim=(scene.envs, ic.count * self.slots),
            inputs=[cc.record_count, cc.rec_verts, cc.rec_normal, cc.rec_pen, self.surface.active, self.prim, self.verts,
                    self.tet, self.radial, self.slots],
            outputs=[self.active],
            device=scene.device,
        )
        if self._substep != self._substeps - 1:
            return  # capture (and the count) on the last substep only; the earlier launches did no other work
        wp.launch(
            capture_grasps,
            dim=(scene.envs, ic.count),
            inputs=[scene.field("q", b), scene.field("inv_mass", b), scene.field("flags", b), self.surface.active, self.grasps,
                    ic.grasp_pending, ic.prim_start,
                    ic.prim_counts, ic.prim_kind, ic.prim_a, ic.prim_b, ic.sphere_q, ic.sphere_radius, ic.sphere_active,
                    cc.margin, cc.record_count, cc.rec_tri, cc.rec_tet, cc.rec_verts, cc.rec_bary, cc.rec_pen, cc.rec_normal,
                    self.slots],
            outputs=[self.prim, self.t, self.radial, self.prev_axis, self.verts, self.tet, self.bary, self.gap, self.active,
                     self.count, self.pending],
            device=scene.device,
        )

    IMPULSES = ("impulse", "jaw_impulse", "moment")  # the tool-force products (tool_force reads them)

    def prepare(self, scene):
        super().prepare(scene)
        if not self.enabled:  # no longer zeroed at step start: its last contribution would stay in the tool force
            self.zero(None, *self.IMPULSES)

    def reset(self, scene, mask):
        E, n, G = scene.envs, self.instruments.count, int(self.slots)
        self.zero(mask, *self.IMPULSES)
        if n == 0:
            return
        for value, dst in ((wp.int32(0), self.active.reshape((E, n * G))), (wp.int32(-1), self.prim.reshape((E, n * G))),
                           (wp.int32(0), self.count), (wp.int32(0), self.pending)):
            self.clear(mask, (dst, value))


@wp.kernel
def release_grasps(
    released: wp.array2d(dtype=wp.int32),
    slots: wp.int32,
    active: wp.array3d(dtype=wp.int32),
    prim: wp.array3d(dtype=wp.int32),
    count: wp.array2d(dtype=wp.int32),
    pending: wp.array2d(dtype=wp.int32),
):
    """Opening the jaw frees the tissue before the step's first hold iteration (system.py:1053-1074)."""
    env, k = wp.tid()
    if released[env, k] == 0:
        return
    for g in range(slots):
        active[env, k, g] = 0
        prim[env, k, g] = -1
    count[env, k] = 0
    pending[env, k] = 0


@wp.kernel
def capture_grasps(
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    tet_active: wp.array2d(dtype=wp.int32),
    grasps: wp.array(dtype=wp.int32),
    grasp_pending: wp.array2d(dtype=wp.int32),
    prim_start: wp.array(dtype=wp.int32),
    prim_counts: wp.array(dtype=wp.int32),
    prim_kind: wp.array(dtype=wp.int32),
    prim_a: wp.array(dtype=wp.int32),
    prim_b: wp.array(dtype=wp.int32),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_active: wp.array2d(dtype=wp.int32),
    margin: wp.array(dtype=wp.float32),
    record_count: wp.array2d(dtype=wp.int32),
    rec_tri: wp.array3d(dtype=wp.int32),
    rec_tet: wp.array3d(dtype=wp.int32),
    rec_verts: wp.array3d(dtype=wp.vec3i),
    rec_bary: wp.array3d(dtype=wp.vec3),
    rec_pen: wp.array3d(dtype=wp.float32),
    rec_normal: wp.array3d(dtype=wp.vec3),
    slots: wp.int32,
    prim: wp.array3d(dtype=wp.int32),
    t: wp.array3d(dtype=wp.float32),
    radial: wp.array3d(dtype=wp.vec3),
    prev_axis: wp.array3d(dtype=wp.vec3),
    verts: wp.array3d(dtype=wp.vec3i),
    tet: wp.array3d(dtype=wp.int32),
    bary: wp.array3d(dtype=wp.vec3),
    gap: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    slot_count: wp.array2d(dtype=wp.int32),
    pending: wp.array2d(dtype=wp.int32),
):
    """kernels.py:854-1156 per (env, instrument): an armed capture fills the slots with the deepest jaw-capsule
    records whose tissue point sits on the capsule surface and whose contact normal points away from the capsule
    (ties: lower capsule, lower triangle), and stays armed until something is held."""
    env, k = wp.tid()
    if grasps[k] == 0:  # only graspers hold tissue (the hook's chains are its limbs)
        return
    held = wp.int32(0)  # slots the hold and revalidation kernels dropped this substep leave the count
    for g in range(slots):
        held = held + active[env, k, g]
    slot_count[env, k] = held
    if grasp_pending[env, k] != 0:
        pending[env, k] = 1
    if pending[env, k] == 0:
        return
    p_lo = prim_start[k]
    p_hi = p_lo + prim_counts[k]
    capacity = rec_tri.shape[2]
    selected = wp.int32(0)
    for out in range(slots):
        best_p = wp.int32(-1)
        best_r = wp.int32(-1)
        best_pen = wp.float32(-1.0)
        best_tri = wp.int32(2147483647)
        for p in range(p_lo, p_hi):
            kind = prim_kind[p]
            if kind != 1 and kind != 2:
                continue
            a = prim_a[p]
            b = prim_b[p]
            if sphere_active[env, a] == 0 or sphere_active[env, b] == 0:
                continue
            live = wp.min(record_count[env, p], capacity)
            for r in range(live):
                penetration = rec_pen[env, p, r]
                if penetration < 0.0:
                    continue
                owner = rec_tet[env, p, r]
                if owner < 0 or tet_active[env, owner] == 0:
                    continue  # the triangle's tet was removed since the record was written
                v = rec_verts[env, p, r]
                i0 = v[0]  # body-local
                i1 = v[1]
                i2 = v[2]
                if (flags[env, i0] & ACTIVE) == 0 or (flags[env, i1] & ACTIVE) == 0 or (flags[env, i2] & ACTIVE) == 0:
                    continue
                if inv_mass[env, i0] <= 0.0 and inv_mass[env, i1] <= 0.0 and inv_mass[env, i2] <= 0.0:
                    continue
                p0 = q[env, i0]
                p1 = q[env, i1]
                p2 = q[env, i2]
                if wp.length_sq(wp.cross(p1 - p0, p2 - p0)) <= 1.0e-20:
                    continue
                bc = rec_bary[env, p, r]
                tissue = p0 * bc[0] + p1 * bc[1] + p2 * bc[2]
                pa = sphere_q[env, a]
                pb = sphere_q[env, b]
                coord = closest_point_segment_parameter(tissue, pa, pb)
                offset = tissue - (pa + (pb - pa) * coord)
                distance = wp.length(offset)
                radius = wp.max(sphere_radius[a], sphere_radius[b])
                if distance <= 1.0e-8 or wp.abs(distance - radius) > margin[env] + 1.0e-4:
                    continue
                if wp.dot(rec_normal[env, p, r], offset / distance) <= 0.0:
                    continue  # a sticky anchor the contact stage would push away from every iteration
                duplicate = wp.int32(0)
                for prior in range(selected):
                    pv = verts[env, k, prior]
                    if prim[env, k, prior] == p and pv[0] == v[0] and pv[1] == v[1] and pv[2] == v[2]:
                        duplicate = wp.int32(1)
                if duplicate != 0:
                    continue
                tri = rec_tri[env, p, r]
                better = penetration > best_pen
                if penetration == best_pen:
                    better = p < best_p
                    if p == best_p:
                        better = tri < best_tri
                if better:
                    best_p = p
                    best_r = r
                    best_pen = penetration
                    best_tri = tri
        if best_p < 0:
            break
        v = rec_verts[env, best_p, best_r]
        bc = rec_bary[env, best_p, best_r]
        tissue = q[env, v[0]] * bc[0] + q[env, v[1]] * bc[1] + q[env, v[2]] * bc[2]
        pa = sphere_q[env, prim_a[best_p]]
        pb = sphere_q[env, prim_b[best_p]]
        coord = closest_point_segment_parameter(tissue, pa, pb)
        axis = pb - pa
        offset = tissue - (pa + axis * coord)
        distance = wp.length(offset)
        radius = wp.max(sphere_radius[prim_a[best_p]], sphere_radius[prim_b[best_p]])
        g = selected
        prim[env, k, g] = best_p
        t[env, k, g] = coord
        radial[env, k, g] = offset / distance
        if wp.length(axis) > 1.0e-8:
            prev_axis[env, k, g] = wp.normalize(axis)
        else:
            prev_axis[env, k, g] = wp.vec3(0.0, 0.0, 0.0)
        verts[env, k, g] = v
        tet[env, k, g] = rec_tet[env, best_p, best_r]
        bary[env, k, g] = bc
        gap[env, k, g] = wp.max(distance - radius, 0.0)
        active[env, k, g] = 1
        selected = selected + 1
    for g in range(selected, slots):
        active[env, k, g] = 0
        prim[env, k, g] = -1
    slot_count[env, k] = selected
    if selected > 0:
        pending[env, k] = 0


@wp.func
def parallel_transport_radial(radial: wp.vec3, previous_axis: wp.vec3, current_axis: wp.vec3):
    """kernels.py:1158-1196: rotate the radial with the capsule axis."""
    radial_length = wp.length(radial)
    if radial_length <= 1.0e-8:
        radial = wp.vec3(1.0, 0.0, 0.0)
    else:
        radial = radial / radial_length
    previous_length = wp.length(previous_axis)
    current_length = wp.length(current_axis)
    if previous_length <= 1.0e-8 or current_length <= 1.0e-8:
        return radial
    previous_axis = previous_axis / previous_length
    current_axis = current_axis / current_length
    cross_axis = wp.cross(previous_axis, current_axis)
    sine = wp.length(cross_axis)
    cosine = wp.min(wp.max(wp.dot(previous_axis, current_axis), -1.0), 1.0)
    transported = radial
    if sine > 1.0e-8:
        rotation_axis = cross_axis / sine
        transported = radial * cosine + wp.cross(rotation_axis, radial) * sine + rotation_axis * wp.dot(rotation_axis, radial) * (1.0 - cosine)
    elif cosine < 0.0:
        rotation_axis = wp.cross(previous_axis, wp.vec3(1.0, 0.0, 0.0))
        if wp.length(rotation_axis) <= 1.0e-8:
            rotation_axis = wp.cross(previous_axis, wp.vec3(0.0, 1.0, 0.0))
        rotation_axis = wp.normalize(rotation_axis)
        transported = -radial + 2.0 * rotation_axis * wp.dot(rotation_axis, radial)
    transported_length = wp.length(transported)
    if transported_length <= 1.0e-8:
        return radial
    return transported / transported_length


@wp.kernel
def hold_grasps(
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    tet_active: wp.array2d(dtype=wp.int32),
    prim_a: wp.array(dtype=wp.int32),
    prim_b: wp.array(dtype=wp.int32),
    sphere_q_prev: wp.array2d(dtype=wp.vec3),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_jaw_sign: wp.array(dtype=wp.float32),
    stiffness: wp.array2d(dtype=wp.float32),
    alpha: wp.float32,
    iterations: wp.int32,
    slots: wp.int32,
    count: wp.int32,
    tet: wp.array3d(dtype=wp.int32),
    prim: wp.array3d(dtype=wp.int32),
    t: wp.array3d(dtype=wp.float32),
    verts: wp.array3d(dtype=wp.vec3i),
    bary: wp.array3d(dtype=wp.vec3),
    gap: wp.array3d(dtype=wp.float32),
    radial: wp.array3d(dtype=wp.vec3),
    prev_axis: wp.array3d(dtype=wp.vec3),
    active: wp.array3d(dtype=wp.int32),
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
    delta_k: wp.array2d(dtype=wp.vec3),
    delta_jaw: wp.array2d(dtype=wp.vec3),
):
    """kernels.py:1258-1453: pull the held tissue point toward its spot on the (interpolated) capsule surface.
    A slot whose triangle lost its tet (cutting) is dropped: that replaces omnisurg's surface-revision rescan."""
    env, tid = wp.tid()
    k = tid // slots
    g = tid - k * slots
    if active[env, k, g] == 0:
        return
    owner = tet[env, k, g]
    if owner < 0 or tet_active[env, owner] == 0:
        active[env, k, g] = 0
        return
    p = prim[env, k, g]
    a = prim_a[p]
    b = prim_b[p]
    pa = sphere_q_prev[env, a] + (sphere_q[env, a] - sphere_q_prev[env, a]) * alpha
    pb = sphere_q_prev[env, b] + (sphere_q[env, b] - sphere_q_prev[env, b]) * alpha
    axis = pb - pa
    transported = parallel_transport_radial(radial[env, k, g], prev_axis[env, k, g], axis)
    if wp.length(axis) > 1.0e-8:
        prev_axis[env, k, g] = wp.normalize(axis)
    radial[env, k, g] = transported
    coord = wp.clamp(t[env, k, g], 0.0, 1.0)
    radius = wp.max(sphere_radius[a], sphere_radius[b])
    target = pa + axis * coord + transported * (radius + wp.max(gap[env, k, g], 0.0))

    v = verts[env, k, g]
    i0 = v[0]  # body-local: the views take (env, i), the accumulators env * count + i
    i1 = v[1]
    i2 = v[2]
    l0 = env * count + i0
    l1 = env * count + i1
    l2 = env * count + i2
    if (flags[env, i0] & ACTIVE) == 0 or (flags[env, i1] & ACTIVE) == 0 or (flags[env, i2] & ACTIVE) == 0:
        active[env, k, g] = 0
        return
    p0 = q[env, i0]
    p1 = q[env, i1]
    p2 = q[env, i2]
    if wp.length_sq(wp.cross(p1 - p0, p2 - p0)) <= 1.0e-20:
        active[env, k, g] = 0
        return
    bc = bary[env, k, g]
    w0 = wp.max(inv_mass[env, i0], 0.0)
    w1 = wp.max(inv_mass[env, i1], 0.0)
    w2 = wp.max(inv_mass[env, i2], 0.0)
    denom = w0 * bc[0] * bc[0] + w1 * bc[1] * bc[1] + w2 * bc[2] * bc[2]
    if denom <= 1.0e-8:
        active[env, k, g] = 0
        return
    tissue = p0 * bc[0] + p1 * bc[1] + p2 * bc[2]
    configured = wp.clamp(stiffness[env, k], 0.0, 1.0)
    solve_alpha = 1.0 - wp.pow(1.0 - configured, 1.0 / wp.float32(wp.max(iterations, 1)))
    residual = target - tissue
    side = sphere_jaw_sign[a]  # the holding capsule's jaw: +1 left, -1 right
    if w0 > 0.0 and bc[0] > 0.0:
        d0 = residual * (solve_alpha * w0 * bc[0] / denom)
        wp.atomic_add(delta, l0, d0)
        wp.atomic_add(delta_count, l0, 1)
        wp.atomic_add(delta_k, l0, k, d0)
        if side != 0.0:
            wp.atomic_add(delta_jaw, l0, k, d0 * side)
    if w1 > 0.0 and bc[1] > 0.0:
        d1 = residual * (solve_alpha * w1 * bc[1] / denom)
        wp.atomic_add(delta, l1, d1)
        wp.atomic_add(delta_count, l1, 1)
        wp.atomic_add(delta_k, l1, k, d1)
        if side != 0.0:
            wp.atomic_add(delta_jaw, l1, k, d1 * side)
    if w2 > 0.0 and bc[2] > 0.0:
        d2 = residual * (solve_alpha * w2 * bc[2] / denom)
        wp.atomic_add(delta, l2, d2)
        wp.atomic_add(delta_count, l2, 1)
        wp.atomic_add(delta_k, l2, k, d2)
        if side != 0.0:
            wp.atomic_add(delta_jaw, l2, k, d2 * side)


@wp.kernel
def revalidate_grasps(
    record_count: wp.array2d(dtype=wp.int32),
    rec_verts: wp.array3d(dtype=wp.vec3i),
    rec_normal: wp.array3d(dtype=wp.vec3),
    rec_pen: wp.array3d(dtype=wp.float32),
    tet_active: wp.array2d(dtype=wp.int32),
    prim: wp.array3d(dtype=wp.int32),
    verts: wp.array3d(dtype=wp.vec3i),
    tet: wp.array3d(dtype=wp.int32),
    radial: wp.array3d(dtype=wp.vec3),
    slots: wp.int32,
    active: wp.array3d(dtype=wp.int32),
):
    """kernels.py:1198-1256: drop a slot whose deepest live contact now pushes against its radial (tissue slipped
    to the other side of the capsule), or whose triangle lost its tet."""
    env, tid = wp.tid()
    k = tid // slots
    g = tid - k * slots
    if active[env, k, g] == 0:
        return
    owner = tet[env, k, g]
    if owner < 0 or tet_active[env, owner] == 0:
        active[env, k, g] = 0
        return
    p = prim[env, k, g]
    v = verts[env, k, g]
    capacity = rec_verts.shape[2]
    deepest = wp.int32(-1)
    deepest_pen = wp.float32(-1.0)
    for r in range(wp.min(record_count[env, p], capacity)):
        rv = rec_verts[env, p, r]
        pen = rec_pen[env, p, r]
        if rv[0] == v[0] and rv[1] == v[1] and rv[2] == v[2] and pen >= 0.0 and pen > deepest_pen:
            deepest = r
            deepest_pen = pen
    if deepest >= 0 and wp.dot(rec_normal[env, p, deepest], radial[env, k, g]) <= 0.0:
        active[env, k, g] = 0
