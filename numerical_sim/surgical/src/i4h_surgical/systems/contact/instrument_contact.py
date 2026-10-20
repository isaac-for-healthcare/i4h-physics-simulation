# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument vs tissue contact: capsules swept over the substep against surface triangles, with friction.

Ported from omnisurg/instruments/kernels.py: closest-point helpers 576-660 and 1702-1827,
legacy contact normal 1840-1883, record + friction + scatter 2287-2469 and the primitive
kernel 3004-3406 (brute force: one thread per (capsule, triangle)); the apply is ``tool_delta.apply_average_tools``
(omnisurg's 3419-3445 did the same). The broad phase bounds the triangle by a sphere and the swept
capsule by its start segment grown by the sweep length, a conservative cull, so every surviving pair
computes exactly what the unculled kernel would. Contact records are kept for the tip capsules only
(sphere, jaw chains): no consumer reads a wrist or shaft record.
Dropped: swept CCD, swept-tet recovery, firm response, smoothed normals, side persistence,
projection microsteps, per-sphere parameter arrays (parameters are per system, live per env).
Front-side-only parts use the bundle's outward triangle normal instead of the tet-owner lookup.

Contact events run on ``iterations`` of the loop turns (``due``, capped at the solver's
iteration count so the whole sweep is covered); on the last event of a substep the contacts
are recorded per capsule, ``records`` deep, for grasp and hook: each thread owns one
(capsule, triangle) pair, keeps its deepest sweep sample and publishes it once with one
atomic, so no thread reads another's slot. Records hold body-local particle indices and the
owning tet, so consumers can test the triangle's liveness after cutting with one flag read.
Triangles are the ``surface`` system's live per-env buffer (degenerate slots are skipped).
``suppress_*`` is an input slot a hook fills so its anchored triangle is not pushed away from
the anchoring capsule (kernels.py:3156-3170). ``records`` and ``front_side_only`` size buffers
at construction; changing them is rejected at the next capture.

Tool force (M11): each pair also writes its share of the correction into the per-instrument channel
``delta_k`` (and, signed by the capsule's jaw side, into ``delta_jaw``); ``tool_delta.apply_average_tools`` turns the share
that ``apply_average`` applies into the tool's ``impulse (E, n)`` and ``jaw_impulse (E, n)`` (mass x displacement over
the frame; ``impulse_count`` pairs), read by ``tool_force``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, due
from i4h_physics.systems.contact.geometry import clamp01, closest_segment_triangle, contact_normal
from i4h_physics.systems.mechanics.surface import Surface
from i4h_surgical.systems.instruments.instrument_control import PRIM_KINDS, InstrumentControl
from i4h_surgical.systems.mechanics.tool_delta import apply_average_tools


@dataclass(eq=False)
class InstrumentContact(System):
    instruments: InstrumentControl  # YAML: id of the instrument_control system
    surface: Surface  # YAML: id of the surface system whose live triangles the tools touch
    margin: float = 0.002  # live, per env
    stiffness: float = 1.0  # live, per env
    static_friction: float = 0.3  # live, per env
    kinetic_friction: float = 0.2  # live, per env
    max_correction: float = 0.0  # live, per env; caps one contact's push-out depth (0 = unlimited)
    iterations: int = 3  # structural: contact events per substep
    sweep_samples: int = 8  # structural: capsule poses sampled over the event interval
    records: int = 8  # structural: contact records kept per capsule
    front_side_only: tuple = ()  # structural: parts the tool may only touch from outside (chole: liver)

    refs = ("instruments", "surface")
    structural = System.structural + ("iterations", "sweep_samples", "records", "front_side_only", "suppress_prim")

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.instruments, InstrumentControl) or not isinstance(self.surface, Surface):
            raise TypeError("instrument_contact.instruments must name an instrument_control system and "
                            "instrument_contact.surface a surface system")
        self.body = b = self.surface.body
        E, dev = scene.envs, scene.device
        self.iterations = check_int("instrument_contact.iterations", self.iterations)
        self.sweep_samples = check_int("instrument_contact.sweep_samples", self.sweep_samples)
        self.records = check_int("instrument_contact.records", self.records)
        P, R = self.instruments.prim_count, self.records
        n = E * b.count
        self.front_side_only = tuple(self.front_side_only)
        unknown = set(self.front_side_only) - set(b.parts)
        if unknown:
            raise ValueError(f"instrument_contact.front_side_only: unknown parts {sorted(unknown)}")
        n_tools = max(self.instruments.count, 1)
        scene.check_memory({"delta": n * 16, "delta_k": n * n_tools * 24, "records": E * P * R * 52})
        front = np.int32([name in self.front_side_only for name in b.parts])
        self.tri_front_only = wp.array(front[b.tet.tri_part], dtype=wp.int32, device=dev)  # (F,) per triangle slot
        kinds = self.instruments.host_prim_kind
        tips = (kinds == PRIM_KINDS["sphere"]) | (kinds == PRIM_KINDS["jaw_left"]) | (kinds == PRIM_KINDS["jaw_right"])
        self.prim_records = wp.array(np.int32(tips), dtype=wp.int32, device=dev)  # (P,) capsules whose records are read
        self._live("margin")
        self._live("stiffness")
        self._live("static_friction")
        self._live("kinetic_friction")
        self._live("max_correction")
        self.delta = wp.zeros(n, dtype=wp.vec3, device=dev)  # body-local, zero between launches
        self.delta_count = wp.zeros(n, dtype=wp.int32, device=dev)
        # Tool force channel: each instrument's share of ``delta``, reduced into ``impulse`` before the apply.
        self.delta_k = wp.zeros((n, n_tools), dtype=wp.vec3, device=dev)
        self.delta_jaw = wp.zeros((n, n_tools), dtype=wp.vec3, device=dev)  # the same share signed by the jaw side (+1 left, -1 right)
        self.impulse = wp.zeros((E, n_tools), dtype=wp.vec3, device=dev)  # product: sum of m * dx the tool imposed this step
        self.jaw_impulse = wp.zeros((E, n_tools), dtype=wp.vec3, device=dev)  # product: left minus right jaw share (the squeeze)
        self.moment = wp.zeros((E, n_tools), dtype=wp.vec3, device=dev)  # product: the impulse's moment about the tool root (coupling)
        self.impulse_count = wp.zeros((E, n_tools), dtype=wp.int32, device=dev)  # product: (capsule, triangle) pairs this step
        # Contact records (E, P, R): triangle slot, owning tet, body-local vertices, barycentric,
        # penetration, capsule coordinate t, normal (tool -> tissue). Consumers rebuild the points.
        # contacts found per capsule; keeps counting past the R record slots, so a reader clamps it: min(count, R)
        self.record_count = wp.zeros((E, P), dtype=wp.int32, device=dev)
        self.rec_tri = wp.zeros((E, P, R), dtype=wp.int32, device=dev)
        self.rec_tet = wp.zeros((E, P, R), dtype=wp.int32, device=dev)
        self.rec_verts = wp.zeros((E, P, R), dtype=wp.vec3i, device=dev)
        self.rec_bary = wp.zeros((E, P, R), dtype=wp.vec3, device=dev)
        self.rec_pen = wp.zeros((E, P, R), dtype=wp.float32, device=dev)
        self.rec_t = wp.zeros((E, P, R), dtype=wp.float32, device=dev)
        self.rec_normal = wp.zeros((E, P, R), dtype=wp.vec3, device=dev)
        # product: contacts found beyond the R record slots this step (summed over capsules and substeps). Which R
        # records a capsule keeps follows thread scheduling, not depth; a nonzero count says the records missed some.
        self.record_overflow = wp.zeros(E, dtype=wp.int32, device=dev)
        self.freeze("records", "front_side_only")
        # Hook suppression slot: a hook system sets the capsule index (structural) and points these at its entry anchor.
        self.suppress_prim = -1
        self.suppress_active = wp.zeros(E, dtype=wp.int32, device=dev)
        self.suppress_verts = wp.zeros(E, dtype=wp.vec3i, device=dev)
        self._substep, self._substeps = 0, 1


    def readouts(self) -> dict:
        return {"record_overflow": self.record_overflow}

    def step_start(self, scene, dt):
        self.impulse.zero_()
        self.jaw_impulse.zero_()
        self.moment.zero_()
        self.impulse_count.zero_()
        self.record_overflow.zero_()

    def substep_end(self, scene, dt):
        if self.instruments.prim_count:  # reads the substep's counts only
            wp.launch(count_record_overflow, dim=self.record_count.shape, inputs=[self.record_count, self.records],
                      outputs=[self.record_overflow], device=scene.device)

    def substep_start(self, scene, dt, s, substeps):
        self._substep, self._substeps = s, substeps
        self.record_count.zero_()

    def iteration(self, scene, dt, k, iterations):
        ic, b = self.instruments, self.body
        event, events = due(k, self.iterations, iterations)
        if ic.prim_count == 0 or event < 0:
            return
        span = float(self._substeps * events)
        alpha_substep = self._substep / float(self._substeps)
        alpha_start = (self._substep * events + event) / span
        alpha_end = (self._substep * events + event + 1) / span
        sf = self.surface
        F = sf.tris.shape[1]
        wp.launch(
            project_contacts,
            dim=(scene.envs, ic.prim_count * F),
            inputs=[scene.field("q", b), scene.field("q_prev", b), scene.field("inv_mass", b), scene.field("flags", b), sf.tris,
                    sf.tri_tet, sf.active,
                    self.tri_front_only, self.prim_records, ic.prim_a, ic.prim_b, ic.sphere_q_prev, ic.sphere_q, ic.sphere_radius,
                    ic.sphere_active, self.margin, self.stiffness, self.static_friction, self.kinetic_friction,
                    self.max_correction, alpha_substep, alpha_start, alpha_end, int(self.sweep_samples), int(event == events - 1),
                    int(self.suppress_prim), self.suppress_active, self.suppress_verts, ic.prim_instrument, ic.sphere_jaw_sign, b.count, F],
            outputs=[self.delta, self.delta_count, self.delta_k, self.delta_jaw, self.impulse_count, self.record_count, self.rec_tri, self.rec_tet,
                     self.rec_verts, self.rec_bary, self.rec_pen, self.rec_t, self.rec_normal],
            device=scene.device,
        )
        wp.launch(apply_average_tools, dim=(scene.envs, b.count),
                  inputs=[scene.field("flags", b), scene.field("inv_mass", b), b.count, ic.count, ic.root],
                  outputs=[scene.field("q", b), self.delta, self.delta_count, self.delta_k, self.delta_jaw, self.impulse,
                           self.jaw_impulse, self.moment], device=scene.device)

    IMPULSES = ("impulse", "jaw_impulse", "moment", "impulse_count")  # the tool-force products (tool_force reads them)
    PRODUCTS = IMPULSES + ("record_overflow",)

    def prepare(self, scene):
        super().prepare(scene)
        if not self.enabled:  # no longer zeroed at step start: its last contribution would stay in the tool force
            self.zero(None, *self.PRODUCTS)
            self.record_count.zero_()

    def reset(self, scene, mask):
        """Selected envs start the next episode without contact records (the hook clears its own anchors)."""
        self.zero(mask, *self.PRODUCTS)
        if self.instruments.prim_count:
            self.clear(mask, (self.record_count, wp.int32(0)))


@wp.kernel
def count_record_overflow(record_count: wp.array2d(dtype=wp.int32), records: int, overflow: wp.array(dtype=wp.int32)):
    env, p = wp.tid()
    extra = record_count[env, p] - records
    if extra > 0:
        wp.atomic_add(overflow, env, extra)


# --- the contact kernel (kernels.py:3004-3406 with 2287-2469 inlined) ------------------------------------------


@wp.kernel
def project_contacts(
    q: wp.array2d(dtype=wp.vec3),
    q_prev: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    tris: wp.array2d(dtype=wp.vec3i),
    tri_tet: wp.array2d(dtype=wp.int32),
    tet_active: wp.array2d(dtype=wp.int32),
    tri_front_only: wp.array(dtype=wp.int32),
    prim_records: wp.array(dtype=wp.int32),
    prim_a: wp.array(dtype=wp.int32),
    prim_b: wp.array(dtype=wp.int32),
    sphere_q_prev: wp.array2d(dtype=wp.vec3),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_active: wp.array2d(dtype=wp.int32),
    margin: wp.array(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    static_friction: wp.array(dtype=wp.float32),
    kinetic_friction: wp.array(dtype=wp.float32),
    max_correction: wp.array(dtype=wp.float32),
    alpha_substep: wp.float32,
    alpha_start: wp.float32,
    alpha_end: wp.float32,
    samples: wp.int32,
    record: wp.int32,
    suppress_prim: wp.int32,
    suppress_active: wp.array(dtype=wp.int32),
    suppress_verts: wp.array(dtype=wp.vec3i),
    prim_instrument: wp.array(dtype=wp.int32),
    sphere_jaw_sign: wp.array(dtype=wp.float32),
    count: wp.int32,
    tri_count: wp.int32,
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
    delta_k: wp.array2d(dtype=wp.vec3),
    delta_jaw: wp.array2d(dtype=wp.vec3),
    impulse_count: wp.array2d(dtype=wp.int32),
    record_count: wp.array2d(dtype=wp.int32),
    rec_tri: wp.array3d(dtype=wp.int32),
    rec_tet: wp.array3d(dtype=wp.int32),
    rec_verts: wp.array3d(dtype=wp.vec3i),
    rec_bary: wp.array3d(dtype=wp.vec3),
    rec_pen: wp.array3d(dtype=wp.float32),
    rec_t: wp.array3d(dtype=wp.float32),
    rec_normal: wp.array3d(dtype=wp.vec3),
):
    env, tid = wp.tid()
    p = tid // tri_count
    f = tid - p * tri_count
    a = prim_a[p]
    b = prim_b[p]
    if sphere_active[env, a] == 0 or sphere_active[env, b] == 0:
        return
    tri = tris[env, f]
    if tri[0] == tri[1]:  # degenerate slot: padding of a smaller variant, a removed triangle or a free cut slot
        return
    radius = wp.max(sphere_radius[a], sphere_radius[b]) + margin[env]
    if radius <= 0.0:
        return
    i0 = tri[0]  # body-local corners: the particle views take (env, i), the accumulators env * count + i
    i1 = tri[1]
    i2 = tri[2]
    p0 = q[env, i0]
    p1 = q[env, i1]
    p2 = q[env, i2]

    # Broad phase before any other load: the triangle's bounding sphere against the capsule's start
    # segment grown by how far either end travels over the event. Any swept axis point lies within
    # that travel of the start segment, so a pair with penetration can never be rejected here.
    a_prev = sphere_q_prev[env, a]
    b_prev = sphere_q_prev[env, b]
    a_cur = sphere_q[env, a]
    b_cur = sphere_q[env, b]
    a_s = a_prev + (a_cur - a_prev) * alpha_start
    b_s = b_prev + (b_cur - b_prev) * alpha_start
    a_e = a_prev + (a_cur - a_prev) * alpha_end
    b_e = b_prev + (b_cur - b_prev) * alpha_end
    centroid = (p0 + p1 + p2) / 3.0
    tri_radius = wp.max(wp.length(p0 - centroid), wp.max(wp.length(p1 - centroid), wp.length(p2 - centroid)))
    travel = wp.max(wp.length(a_e - a_s), wp.length(b_e - b_s))
    axis = b_s - a_s
    t_axis = wp.clamp(wp.dot(centroid - a_s, axis) / wp.max(wp.dot(axis, axis), 1.0e-12), 0.0, 1.0)
    if wp.length(centroid - (a_s + axis * t_axis)) > tri_radius + radius + travel:
        return

    owner = tri_tet[env, f]
    if owner >= 0 and tet_active[env, owner] == 0:
        return  # removed this step; the surface system rewrites the slot at step end
    e0 = (flags[env, i0] & ACTIVE) != 0
    e1 = (flags[env, i1] & ACTIVE) != 0
    e2 = (flags[env, i2] & ACTIVE) != 0
    if not e0 and not e1 and not e2:
        return
    w0 = wp.float32(0.0)
    w1 = wp.float32(0.0)
    w2 = wp.float32(0.0)
    if e0:
        w0 = wp.max(inv_mass[env, i0], 0.0)
    if e1:
        w1 = wp.max(inv_mass[env, i1], 0.0)
    if e2:
        w2 = wp.max(inv_mass[env, i2], 0.0)
    if w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0:
        return

    k_contact = clamp01(stiffness[env])
    fs = wp.max(static_friction[env], 0.0)
    fk = wp.max(kinetic_friction[env], 0.0)
    if p == suppress_prim and suppress_active[env] != 0:
        sv = suppress_verts[env]
        if tri[0] == sv[0] and tri[1] == sv[1] and tri[2] == sv[2]:
            k_contact = 0.0  # the hook's anchored triangle is held by the hook, not pushed by the capsule
    a_sub = a_prev + (a_cur - a_prev) * alpha_substep
    b_sub = b_prev + (b_cur - b_prev) * alpha_substep

    triangle_normal = wp.cross(p1 - p0, p2 - p0)
    triangle_normal_length = wp.length(triangle_normal)
    front_only = tri_front_only[f]
    n_samples = wp.max(samples, 1)
    if wp.length_sq(a_e - a_s) <= 1.0e-12 and wp.length_sq(b_e - b_s) <= 1.0e-12:
        n_samples = 1
    best_pen = wp.float32(0.0)  # deepest sample of this (capsule, triangle) pair, published once below
    best_bary = wp.vec3(0.0, 0.0, 0.0)
    best_t = wp.float32(0.0)
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    k = prim_instrument[p]
    side = sphere_jaw_sign[a]  # +1 left jaw, -1 right jaw, 0 for the probe, wrist, shaft and the hook's static chains
    pushed = wp.int32(0)
    for si in range(n_samples):
        alpha = alpha_end
        if n_samples > 1:
            alpha = alpha_start + (alpha_end - alpha_start) * (wp.float32(si) / wp.float32(n_samples - 1))
        ca = a_prev + (a_cur - a_prev) * alpha
        cb = b_prev + (b_cur - b_prev) * alpha
        res = closest_segment_triangle(ca, cb, p0, p1, p2)
        bary = wp.vec3(res[0], res[1], res[2])
        t = res[3]
        tool = ca + (cb - ca) * t
        tool_sub = a_sub + (b_sub - a_sub) * t
        closest = p0 * bary[0] + p1 * bary[1] + p2 * bary[2]
        offset = closest - tool
        distance = wp.length(offset)
        penetration = radius - distance
        if penetration <= 0.0:
            continue
        if front_only != 0 and triangle_normal_length > 1.0e-8:
            normal = -triangle_normal / triangle_normal_length  # the bundle's triangles face outward
        else:
            normal = contact_normal(offset, distance, triangle_normal, triangle_normal_length)

        if penetration > best_pen:
            best_pen = penetration
            best_bary = bary
            best_t = t
            best_normal = normal

        # kernels.py:2398-2468 push-out with stick/slip friction against the substep-start tissue point
        depth = penetration * k_contact
        if max_correction[env] > 0.0:
            depth = wp.min(depth, max_correction[env])
        if depth <= 0.0:
            continue
        denom = w0 * bary[0] * bary[0] + w1 * bary[1] * bary[1] + w2 * bary[2] * bary[2]
        if denom <= 1.0e-8:
            continue
        correction = normal
        scale = depth / denom
        if fs > 0.0 or fk > 0.0:
            correction = normal * depth
            tissue_start = q_prev[env, i0] * bary[0] + q_prev[env, i1] * bary[1] + q_prev[env, i2] * bary[2]
            relative = (closest - tissue_start) - (tool - tool_sub)
            slip = relative - normal * wp.dot(relative, normal)
            slip_length = wp.length(slip)
            if slip_length <= fs * depth:
                correction = correction - slip
            elif slip_length > 1.0e-8 and fk > 0.0:
                correction = correction - slip * wp.min(fk * depth / slip_length, 1.0)
            scale = 1.0 / denom
        l0 = env * count + i0
        l1 = env * count + i1
        l2 = env * count + i2
        pushed = wp.int32(1)
        if w0 > 0.0 and bary[0] > 0.0:
            d0 = correction * (scale * w0 * bary[0])
            wp.atomic_add(delta, l0, d0)
            wp.atomic_add(delta_count, l0, 1)
            wp.atomic_add(delta_k, l0, k, d0)
            if side != 0.0:
                wp.atomic_add(delta_jaw, l0, k, d0 * side)
        if w1 > 0.0 and bary[1] > 0.0:
            d1 = correction * (scale * w1 * bary[1])
            wp.atomic_add(delta, l1, d1)
            wp.atomic_add(delta_count, l1, 1)
            wp.atomic_add(delta_k, l1, k, d1)
            if side != 0.0:
                wp.atomic_add(delta_jaw, l1, k, d1 * side)
        if w2 > 0.0 and bary[2] > 0.0:
            d2 = correction * (scale * w2 * bary[2])
            wp.atomic_add(delta, l2, d2)
            wp.atomic_add(delta_count, l2, 1)
            wp.atomic_add(delta_k, l2, k, d2)
            if side != 0.0:
                wp.atomic_add(delta_jaw, l2, k, d2 * side)
    if pushed != 0:
        wp.atomic_add(impulse_count, env, k, wp.int32(1))

    # kernels.py:2340-2396 record on the last event of the substep: one slot per thread, taken with one atomic;
    # only the tip capsules (sphere, jaw chains) are read back by grasp, hook and thermal
    if record != 0 and best_pen > 0.0 and prim_records[p] != 0:
        slot = wp.atomic_add(record_count, env, p, wp.int32(1))
        if slot < rec_tri.shape[2]:
            rec_tri[env, p, slot] = f
            rec_tet[env, p, slot] = tri_tet[env, f]
            rec_verts[env, p, slot] = tri
            rec_bary[env, p, slot] = best_bary
            rec_pen[env, p, slot] = best_pen
            rec_t[env, p, slot] = best_t
            rec_normal[env, p, slot] = best_normal
