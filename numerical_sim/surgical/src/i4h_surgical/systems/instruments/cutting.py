# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mechanical cut: tets touched by an instrument's cutting spheres are removed while it cuts.

Ported from omnisurg/tet/kernels/tet.py: point/triangle and point/tet distance 130-233, the
sphere-slice test 1122-1169 and mark_cuttable_tets_intersecting_spheres 1172-1214 (the scissors
path; the grasper crush cut 1061 is the same test without the deletable mask), the neighbour blood
1248-1274 (a surviving tet's face whose neighbour was just cut paints its three corners) and
1276-1294 (blood into the visual state, monotone).
Edits: (env, tet) prologue; the spheres are the instrument's cutting chains when it has them (the
scissors' ``cutting`` block, no capsules) else its fixed chains (jaw or hook piece; the probe's one
sphere), not a host-resolved range; the trigger is ``fire`` (the instrument's ``fire_on`` level: the
probe, the hook) or ``closing`` (omnisurg system.py:1016-1032: the scissors cut in every frame the
jaw closes by more than ``closure_delta``; here the relaxed jaw angle's decrease per step, in radians,
so a held-closed jaw never cuts and reopening rearms by itself); ``cut_on`` (E, n) publishes the
step's cut signal for ``bleeding``; removal goes through ``surface.deactivate_tet`` and the surface
system re-emits at step end; the blood amount is a live value (0 paints nothing); no cut preview.
Runs once per step, in the last substep's ``substep_end``, against the frame's final tissue positions
(the spheres are placed once per step, so they are the same in every substep).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System
from i4h_physics.systems.mechanics.surface import Surface, deactivate_tet, tet_face
from i4h_surgical.loaders.instrument_usd import CHAIN_CUT, CHAIN_FIXED
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

TRIGGERS = ("fire", "closing")


@dataclass(eq=False)
class Cutting(System):
    state_products = ("cut_on",)  # per-env products a host observes (System.state_products)
    surface: Surface
    instruments: InstrumentControl
    instrument: str  # the cutting instrument (chole: the scissors; two_worlds: the probe)
    trigger: str = "fire"  # structural: fire = the tool's fire level; closing = the jaw closes by more than closure_delta
    radius_scale: float = 1.0  # live, per env: sphere radius multiplier for the cut
    margin: float = 0.004  # live, per env: added to the scaled radius; contact holds tissue 0.002 off the tool
    closure_delta: float = 1.0e-6  # live, per env: jaw angle decrease per step (radians) that counts as closing (schema.py:1990)
    blood: float = 0.0  # live, per env: blood painted on the faces a cut exposes (omnisurg blood_amount 0.35; 0 = none)

    refs = ("surface", "instruments")
    structural = System.structural + ("instrument", "trigger")  # the sphere range and the gate are baked into the launches

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        if not isinstance(self.surface, Surface) or not isinstance(self.instruments, InstrumentControl):
            raise TypeError("cutting.surface must name a surface system and cutting.instruments an instrument_control system")
        if self.trigger not in TRIGGERS:
            raise ValueError(f"cutting.trigger must be one of {TRIGGERS}, got {self.trigger!r}")
        ic, sf = self.instruments, self.surface
        if self.instrument not in ic.names:
            raise ValueError(f"cutting: unknown instrument {self.instrument!r}; known: {ic.names}")
        self.k = k = ic.names.index(self.instrument)
        base = int(ic.host_sphere_start[k])
        own = ic.host_sphere_chain[base : base + int(ic.host_sphere_counts[k])]
        cut = np.flatnonzero(own == CHAIN_CUT)
        if cut.size:  # the scissors' cutting chains, contiguous after the collision chains (instrument_usd.py)
            self.sphere_start, self.sphere_count = base + int(cut[0]), int(cut.size)
        else:
            self.sphere_start, self.sphere_count = base, int((own == CHAIN_FIXED).sum())  # fixed chains come first (instrument_control.py:74)
        self.freeze("instrument", "trigger")
        T, count = sf.tets.shape[1], sf.body.count
        scene.check_memory({"cut_tets": E * T * 4, "blood": E * count * 4})
        self.cut_on = wp.zeros((E, ic.count), dtype=wp.int32, device=dev)  # product: 1 where the tool cuts this step
        self.cut_tets = wp.zeros((E, T), dtype=wp.int32, device=dev)  # tets this step's cut removed (the blood's neighbours)
        self.blood_acc = wp.zeros(E * count, dtype=wp.float32, device=dev)  # body-local, self-clearing
        self._live("radius_scale", "margin", "closure_delta", "blood")
        self._last = False


    def prepare(self, scene):
        super().prepare(scene)
        if not self.enabled:  # bleeding keeps reading cut_on: a disabled cutter must not leave the last step's signal latched
            self.cut_on.zero_()

    def step_start(self, scene, dt):
        ic = self.instruments
        wp.launch(arm_cuts, dim=(scene.envs, ic.count),
                  inputs=[ic.fire_on, ic.jaw_delta, self.closure_delta, self.k, TRIGGERS.index(self.trigger)],
                  outputs=[self.cut_on], device=scene.device)

    def substep_start(self, scene, dt, s, substeps):
        self._last = s == substeps - 1

    def substep_end(self, scene, dt):
        sf, ic, b, E, dev = self.surface, self.instruments, self.surface.body, scene.envs, scene.device
        T = sf.tets.shape[1]
        if not self._last or T == 0 or self.sphere_count == 0:
            return
        wp.launch(
            cut_tets,
            dim=(E, T),
            inputs=[scene.field("q", b), self.cut_on, self.k, sf.variant, sf.tets, sf.tet_part, sf.deletable, sf.tet_edges, ic.sphere_q,
                    ic.sphere_radius, ic.sphere_active, self.sphere_start, self.sphere_count, self.radius_scale, self.margin,
                    b.count],
            outputs=[sf.active, sf.edge_active, sf.support, sf.edge_support, scene.field("flags", b), sf.deleted, sf.dirty, self.cut_tets],
            device=dev,
        )
        wp.launch(neighbour_blood, dim=(E, T, 4),
                  inputs=[sf.variant, sf.tets, sf.face_neighbor, sf.active, self.cut_tets, self.blood, b.count],
                  outputs=[self.blood_acc], device=dev)
        wp.launch(apply_blood, dim=(E, b.count), inputs=[scene.field("flags", b), b.count],
                  outputs=[self.blood_acc, sf.appearance.visual], device=dev)

    def reset(self, scene, mask):
        for dst in (self.cut_on, self.cut_tets):
            self.clear(mask, (dst, wp.int32(0)))
        self.clear(mask, (self.blood_acc, wp.float32(0.0)))


# omnisurg system.py:1016-1032 (the scissors' closure test) and the fire level, as one per-step signal per instrument.
@wp.kernel
def arm_cuts(
    fire_on: wp.array2d(dtype=wp.int32),
    jaw_delta: wp.array2d(dtype=wp.float32),
    closure_delta: wp.array(dtype=wp.float32),
    k: wp.int32,
    trigger: wp.int32,
    cut_on: wp.array2d(dtype=wp.int32),
):
    env, j = wp.tid()
    on = wp.int32(0)
    if j == k:
        if trigger == 0:
            on = fire_on[env, k]
        elif jaw_delta[env, k] > closure_delta[env]:
            on = wp.int32(1)
    cut_on[env, j] = on


# --- omnisurg/tet/kernels/tet.py:130-233 point/triangle and point/tet distances, verbatim -------------------


@wp.func
def point_triangle_distance_sq(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.dot(ap, ap)
    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        delta = point - (a + v * ab)
        return wp.dot(delta, delta)
    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        delta = point - (a + w * ac)
        return wp.dot(delta, delta)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        delta = point - (b + w * (c - b))
        return wp.dot(delta, delta)
    normal = wp.cross(ab, ac)
    normal_length_sq = wp.dot(normal, normal)
    if normal_length_sq <= 1.0e-16:
        return wp.min(wp.min(wp.dot(ap, ap), wp.dot(bp, bp)), wp.dot(cp, cp))
    distance = wp.dot(point - a, normal)
    return distance * distance / normal_length_sq


@wp.func
def point_inside_tet(point: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3) -> wp.int32:
    a = p1 - p0
    b = p2 - p0
    c = p3 - p0
    v = point - p0
    det = wp.dot(a, wp.cross(b, c))
    if wp.abs(det) <= 1.0e-12:
        return wp.int32(0)
    b1 = wp.dot(v, wp.cross(b, c)) / det
    b2 = wp.dot(a, wp.cross(v, c)) / det
    b3 = wp.dot(a, wp.cross(b, v)) / det
    b0 = 1.0 - b1 - b2 - b3
    if (b0 >= -1.0e-6 and b1 >= -1.0e-6 and b2 >= -1.0e-6 and b3 >= -1.0e-6
            and b0 <= 1.0 + 1.0e-6 and b1 <= 1.0 + 1.0e-6 and b2 <= 1.0 + 1.0e-6 and b3 <= 1.0 + 1.0e-6):
        return wp.int32(1)
    return wp.int32(0)


@wp.func
def point_tet_distance_sq(point: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3) -> wp.float32:
    if point_inside_tet(point, p0, p1, p2, p3) != 0:
        return 0.0
    d = wp.float32(3.4028234663852886e38)
    d = wp.min(d, point_triangle_distance_sq(point, p1, p2, p3))
    d = wp.min(d, point_triangle_distance_sq(point, p0, p3, p2))
    d = wp.min(d, point_triangle_distance_sq(point, p0, p1, p3))
    d = wp.min(d, point_triangle_distance_sq(point, p0, p2, p1))
    return d


# tet.py:1122-1214 (_tet_intersects_sphere_slice + mark_cuttable_tets_intersecting_spheres). Edits: the gate is the
# per-step ``cut_on`` signal; a removed tet is marked in ``cut_tets`` for the blood launch (a tet another deleter took
# in the same substep is marked too: harmless extra blood).
@wp.kernel
def cut_tets(
    q: wp.array2d(dtype=wp.vec3),
    cut_on: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    variant: wp.array(dtype=wp.int32),
    tets: wp.array2d(dtype=wp.vec4i),
    tet_part: wp.array(dtype=wp.int32),
    deletable: wp.array(dtype=wp.int32),
    tet_edges: wp.array3d(dtype=wp.int32),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_active: wp.array2d(dtype=wp.int32),
    sphere_start: wp.int32,
    sphere_count: wp.int32,
    radius_scale: wp.array(dtype=wp.float32),
    margin: wp.array(dtype=wp.float32),
    count: wp.int32,
    active: wp.array2d(dtype=wp.int32),
    edge_active: wp.array2d(dtype=wp.int32),
    support: wp.array(dtype=wp.int32),
    edge_support: wp.array2d(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    deleted: wp.array(dtype=wp.int32),
    dirty: wp.array(dtype=wp.int32),
    cut_marks: wp.array2d(dtype=wp.int32),
):
    env, t = wp.tid()
    cut_marks[env, t] = wp.int32(0)  # this substep's marks (neighbour_blood reads them next)
    if cut_on[env, k] == 0 or active[env, t] == 0 or deletable[tet_part[t]] == 0:
        return
    v = variant[env]
    ids = tets[v, t]
    p0 = q[env, ids[0]]
    p1 = q[env, ids[1]]
    p2 = q[env, ids[2]]
    p3 = q[env, ids[3]]
    bounds_min = wp.min(wp.min(p0, p1), wp.min(p2, p3))
    bounds_max = wp.max(wp.max(p0, p1), wp.max(p2, p3))
    for local in range(sphere_count):
        s = sphere_start + local
        if sphere_active[env, s] == 0:
            continue
        radius = sphere_radius[s] * radius_scale[env] + margin[env]
        if radius < 0.0:
            continue
        center = sphere_q[env, s]
        if (center[0] < bounds_min[0] - radius or center[1] < bounds_min[1] - radius or center[2] < bounds_min[2] - radius
                or center[0] > bounds_max[0] + radius or center[1] > bounds_max[1] + radius or center[2] > bounds_max[2] + radius):
            continue
        threshold = radius + 1.0e-6
        if point_tet_distance_sq(center, p0, p1, p2, p3) <= threshold * threshold:
            cut_marks[env, t] = wp.int32(1)
            deactivate_tet(env, t, v, count, tets, tet_edges, active, edge_active, support, edge_support, flags, deleted, dirty)
            return


# tet.py:1248-1274 (accumulate_scissors_cut_neighbour_blood). Edits: (env, tet, face) prologue; ``tet_face`` for
# ``_tet_face_corner_offset``; the amount is the live per-env value.
@wp.kernel
def neighbour_blood(
    variant: wp.array(dtype=wp.int32),
    tets: wp.array2d(dtype=wp.vec4i),
    face_neighbor: wp.array3d(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    cut_marks: wp.array2d(dtype=wp.int32),
    blood: wp.array(dtype=wp.float32),
    count: wp.int32,
    blood_acc: wp.array(dtype=wp.float32),
):
    env, t, face = wp.tid()
    if active[env, t] == 0:
        return
    v = variant[env]
    neighbor = face_neighbor[v, t, face]
    if neighbor < 0 or cut_marks[env, neighbor] == 0:
        return
    amount = wp.min(wp.max(blood[env], 0.0), 1.0)
    if amount <= 0.0:
        return
    corners = tet_face(tets[v, t], face)
    for c in range(3):
        wp.atomic_max(blood_acc, env * count + corners[c], amount)


# tet.py:1276-1294 (apply_scissors_cut_blood). Edits: (env, particle) prologue; the accumulator clears itself.
@wp.kernel
def apply_blood(
    flags: wp.array2d(dtype=wp.int32),
    count: wp.int32,
    blood_acc: wp.array(dtype=wp.float32),
    visual: wp.array(dtype=wp.vec4f),
):
    env, local = wp.tid()
    flat_index = env * count + local
    amount = blood_acc[flat_index]
    blood_acc[flat_index] = 0.0
    if amount <= 0.0 or (flags[env, local] & ACTIVE) == 0:
        return
    state = visual[flat_index]
    if amount > state[2] + 1.0e-6:
        visual[flat_index] = wp.vec4f(state[0], state[1], amount, state[3])
