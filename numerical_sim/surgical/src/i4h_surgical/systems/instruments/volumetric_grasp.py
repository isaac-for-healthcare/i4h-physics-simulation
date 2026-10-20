# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Volumetric grasp: a closed jaw holds the tissue particles inside the union of its jaw spheres.

Ported from omnisurg/instruments/runtime_state.py:311-380 (``select_jaw_chain_grasp_particles``: active particles
whose nearest jaw sphere is within ``particle_radius + jaw_sphere_radius``, up to ``max_grasp_particles``, each
remembering its jaw sphere and its offset from it), system.py:2501-2553 and 1055-1058 (capture when the jaw closes
or while it is held with nothing in hand; release when it opens) and instruments/kernels.py:289-317, 480-556
(``update_grasp_targets_from_jaw_indices``, ``accumulate_grasp_distance_constraints``,
``apply_accumulated_grasp_deltas_range``: hold each particle at its captured radial distance from its jaw sphere,
the direction re-derived every solve, the stiffness spread over the iterations). Chole ran it beside the surface
grasp on the grasper (``max_grasp_particles 64``); so does ours: list it after ``grasp``.

Edits: the selection is a device kernel over the body's particles with an atomic slot claim (the source kept the
``max_count`` nearest particles on the host; ours keeps the first ``capacity`` claimants in launch order, a
nondeterministic subset only when more particles lie inside the jaw union than there are slots); the jaw target
is the sphere's position interpolated over the substep like the surface grasp's (the source clipped the target's
motion by ``max_delta``; the root is velocity-limited already); the deltas are applied with ``tool_delta.apply_average_tools``
(no velocity write, as the source's range apply). Only kind ``grasper`` grasps.
Tool force (M11): the hold writes each tool's share of ``delta`` into ``delta_k`` (and signed by the jaw sphere's side
into ``delta_jaw``) and ``tool_delta.apply_average_tools`` sums the applied share times mass into ``impulse (E, n)`` and
``jaw_impulse (E, n)`` for ``tool_force``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, live
from i4h_surgical.loaders.instrument_usd import CHAIN_FIXED
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.tool_delta import apply_average_tools


@dataclass(eq=False)
class VolumetricGrasp(System):
    state_products = ("count",)  # per-env products a host observes (System.state_products)
    instruments: InstrumentControl
    body: str  # the tissue body whose particles a jaw can hold
    capacity: int = 64  # structural: particles per env and instrument (omnisurg max_grasp_particles)

    refs = ("instruments",)
    structural = System.structural + ("body", "capacity")

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.instruments, InstrumentControl):
            raise TypeError("volumetric_grasp.instruments must name an instrument_control system")
        ic = self.instruments
        self.body = b = scene.body(self.body)
        if b.tet is None:
            raise ValueError(f"volumetric_grasp: body {b.name!r} has no tet mesh")
        self.capacity = G = check_int("volumetric_grasp.capacity", self.capacity, 1)
        E, dev, n = scene.envs, scene.device, ic.count
        records = list(scene.instruments.values())
        starts, counts = [], []
        for k, r in enumerate(records):
            base = int(ic.host_sphere_start[k])
            own = ic.host_sphere_chain[base : base + int(ic.host_sphere_counts[k])]
            starts.append(base)
            counts.append(int((own == CHAIN_FIXED).sum()))  # the jaw chains come first (instrument_control.py:74)
        self.grasps = wp.array(np.int32([r.kind == "grasper" for r in records]).reshape(n), dtype=wp.int32, device=dev)
        self.jaw_start = wp.array(np.int32(starts).reshape(n), dtype=wp.int32, device=dev)
        self.jaw_count = wp.array(np.int32(counts).reshape(n), dtype=wp.int32, device=dev)
        self.stiffness = live(scene, [r.grasp_stiffness for r in records])  # (E, n)
        self.register_live("stiffness")
        self.freeze("body", "capacity")
        scene.check_memory({"slots": E * n * G * 24, "delta": E * b.count * 16, "delta_k": E * b.count * max(n, 1) * 24})
        self.particle = wp.full((E, n, G), -1, dtype=wp.int32, device=dev)  # body-local particle per slot
        self.sphere = wp.zeros((E, n, G), dtype=wp.int32, device=dev)  # its jaw sphere (global index)
        self.rest = wp.zeros((E, n, G), dtype=wp.vec3, device=dev)  # particle - sphere at capture
        self.active = wp.zeros((E, n, G), dtype=wp.int32, device=dev)
        self.count = wp.zeros((E, n), dtype=wp.int32, device=dev)  # product: held particles
        self.pending = wp.zeros((E, n), dtype=wp.int32, device=dev)  # armed capture
        self.claim = wp.zeros((E, n), dtype=wp.int32, device=dev)  # slot cursor of the capture launch
        self.delta = wp.zeros(E * b.count, dtype=wp.vec3, device=dev)
        self.delta_count = wp.zeros(E * b.count, dtype=wp.int32, device=dev)
        self.delta_k = wp.zeros((E * b.count, max(n, 1)), dtype=wp.vec3, device=dev)  # per-tool share of delta
        self.delta_jaw = wp.zeros((E * b.count, max(n, 1)), dtype=wp.vec3, device=dev)  # signed by the jaw side
        self.impulse = wp.zeros((E, max(n, 1)), dtype=wp.vec3, device=dev)  # product: sum of m * dx the hold imposed this step
        self.jaw_impulse = wp.zeros((E, max(n, 1)), dtype=wp.vec3, device=dev)  # product: left minus right jaw share
        self.moment = wp.zeros((E, max(n, 1)), dtype=wp.vec3, device=dev)  # product: the impulse's moment about the tool root (coupling)
        self._substep, self._substeps = 0, 1


    def step_start(self, scene, dt):
        ic = self.instruments
        self.impulse.zero_()
        self.jaw_impulse.zero_()
        self.moment.zero_()
        if ic.count == 0:
            return
        wp.launch(release_slots, dim=(scene.envs, ic.count), inputs=[ic.released, self.capacity],
                  outputs=[self.active, self.particle, self.count, self.pending, self.claim], device=scene.device)

    def substep_start(self, scene, dt, s, substeps):
        self._substep, self._substeps = s, substeps

    def iteration(self, scene, dt, k, iterations):
        ic, b = self.instruments, self.body
        if ic.count == 0:
            return
        alpha = float(self._substep * iterations + k + 1) / float(self._substeps * iterations)  # grasp.py: the substep's share of the tool motion
        wp.launch(hold_particles, dim=(scene.envs, ic.count * self.capacity),
                  inputs=[scene.field("q", b), scene.field("inv_mass", b), scene.field("flags", b), ic.sphere_q_prev, ic.sphere_q,
                          ic.sphere_active, ic.sphere_jaw_sign, self.stiffness, self.particle, self.sphere, self.rest, self.active,
                          alpha, iterations, self.capacity, b.count],
                  outputs=[self.delta, self.delta_count, self.delta_k, self.delta_jaw], device=scene.device)
        wp.launch(apply_average_tools, dim=(scene.envs, b.count),
                  inputs=[scene.field("flags", b), scene.field("inv_mass", b), b.count, ic.count, ic.root],
                  outputs=[scene.field("q", b), self.delta, self.delta_count, self.delta_k, self.delta_jaw, self.impulse,
                           self.jaw_impulse, self.moment], device=scene.device)

    def substep_end(self, scene, dt):
        ic, b, E, dev = self.instruments, self.body, scene.envs, scene.device
        if ic.count == 0 or self._substep != self._substeps - 1:
            return  # capture on the last substep only (as the surface grasp)
        wp.launch(arm_capture, dim=(E, ic.count),
                  inputs=[self.grasps, ic.grasp_pending, ic.held, scene.field("flags", b), scene.field("inv_mass", b), self.capacity],
                  outputs=[self.count, self.pending, self.claim, self.active, self.particle], device=dev)
        wp.launch(capture_particles, dim=(E, b.count),
                  inputs=[scene.field("q", b), scene.field("inv_mass", b), scene.field("flags", b), scene.field("radius", b), ic.sphere_q,
                          ic.sphere_radius, ic.sphere_active, self.pending, self.jaw_start, self.jaw_count, ic.count, self.capacity],
                  outputs=[self.claim, self.particle, self.sphere, self.rest, self.active], device=dev)
        wp.launch(finish_capture, dim=(E, ic.count), inputs=[self.claim, self.capacity], outputs=[self.count, self.pending], device=dev)

    IMPULSES = ("impulse", "jaw_impulse", "moment")  # the tool-force products (tool_force reads them)

    def prepare(self, scene):
        super().prepare(scene)
        if not self.enabled:  # no longer zeroed at step start: its last contribution would stay in the tool force
            self.zero(None, *self.IMPULSES)

    def reset(self, scene, mask):
        E, n, G = scene.envs, self.instruments.count, self.capacity
        self.zero(mask, *self.IMPULSES)
        if n == 0:
            return
        for value, dst in ((wp.int32(-1), self.particle.reshape((E, n * G))), (wp.int32(0), self.active.reshape((E, n * G))),
                           (wp.int32(0), self.sphere.reshape((E, n * G))), (wp.int32(0), self.count), (wp.int32(0), self.pending),
                           (wp.int32(0), self.claim)):
            self.clear(mask, (dst, value))


# system.py:1055-1058, 3185-3193 (_clear_grasp_state): the jaw's opening edge frees everything.
@wp.kernel
def release_slots(
    released: wp.array2d(dtype=wp.int32),
    capacity: wp.int32,
    active: wp.array3d(dtype=wp.int32),
    particle: wp.array3d(dtype=wp.int32),
    count: wp.array2d(dtype=wp.int32),
    pending: wp.array2d(dtype=wp.int32),
    claim: wp.array2d(dtype=wp.int32),
):
    env, k = wp.tid()
    if released[env, k] == 0:
        return
    for g in range(capacity):
        active[env, k, g] = 0
        particle[env, k, g] = -1
    count[env, k] = 0
    pending[env, k] = 0
    claim[env, k] = 0


# system.py:2501-2553 (_capture_state when ``not previous_held or count == 0``): a closing jaw, or a held jaw with
# nothing in hand, arms a capture; the slots are cleared before it (system.py:2525). Edits: a slot whose particle died
# (its tets cut, burnt or torn away) is dropped first and the count recomputed, so a jaw left holding nothing retries.
@wp.kernel
def arm_capture(
    grasps: wp.array(dtype=wp.int32),
    grasp_pending: wp.array2d(dtype=wp.int32),
    held: wp.array2d(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    inv_mass: wp.array2d(dtype=wp.float32),
    capacity: wp.int32,
    count: wp.array2d(dtype=wp.int32),
    pending: wp.array2d(dtype=wp.int32),
    claim: wp.array2d(dtype=wp.int32),
    active: wp.array3d(dtype=wp.int32),
    particle: wp.array3d(dtype=wp.int32),
):
    env, k = wp.tid()
    if grasps[k] == 0:
        return
    live = wp.int32(0)
    for g in range(capacity):
        if active[env, k, g] != 0:
            i = particle[env, k, g]
            if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
                active[env, k, g] = 0
                particle[env, k, g] = -1
            else:
                live += 1
    count[env, k] = live
    if grasp_pending[env, k] != 0 or (held[env, k] != 0 and live == 0):
        pending[env, k] = 1
    if pending[env, k] == 0:
        return
    claim[env, k] = 0
    for g in range(capacity):
        active[env, k, g] = 0
        particle[env, k, g] = -1


# runtime_state.py:318-380 (select_jaw_chain_grasp_particles) as one thread per particle: the nearest active jaw sphere
# of every armed instrument, a hit when the particle lies within particle radius + sphere radius, a slot claimed by
# atomic increment (the source sorted the hits by distance on the host and kept the nearest max_count).
@wp.kernel
def capture_particles(
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    radius: wp.array2d(dtype=wp.float32),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_active: wp.array2d(dtype=wp.int32),
    pending: wp.array2d(dtype=wp.int32),
    jaw_start: wp.array(dtype=wp.int32),
    jaw_count: wp.array(dtype=wp.int32),
    n: wp.int32,
    capacity: wp.int32,
    claim: wp.array2d(dtype=wp.int32),
    particle: wp.array3d(dtype=wp.int32),
    sphere: wp.array3d(dtype=wp.int32),
    rest: wp.array3d(dtype=wp.vec3),
    active: wp.array3d(dtype=wp.int32),
):
    env, local = wp.tid()
    i = local
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        return
    p = q[env, i]
    for k in range(n):
        if pending[env, k] == 0:
            continue
        best = wp.int32(-1)
        best_d = wp.float32(3.4028234663852886e38)
        for s in range(jaw_start[k], jaw_start[k] + jaw_count[k]):
            if sphere_active[env, s] == 0:
                continue
            d = wp.length(p - sphere_q[env, s])
            if d < best_d:
                best_d = d
                best = s
        if best < 0 or best_d > wp.max(radius[env, i], 0.0) + sphere_radius[best]:
            continue
        g = wp.atomic_add(claim, env, k, wp.int32(1))
        if g < capacity:
            particle[env, k, g] = local
            sphere[env, k, g] = best
            rest[env, k, g] = p - sphere_q[env, best]
            active[env, k, g] = 1


@wp.kernel
def finish_capture(
    claim: wp.array2d(dtype=wp.int32),
    capacity: wp.int32,
    count: wp.array2d(dtype=wp.int32),
    pending: wp.array2d(dtype=wp.int32),
):
    env, k = wp.tid()
    if pending[env, k] == 0:
        return
    held = wp.min(claim[env, k], capacity)
    count[env, k] = held
    if held > 0:
        pending[env, k] = 0  # armed until something is held (system.py:1070-1073)


# kernels.py:289-317 (update_grasp_targets_from_jaw_indices) and 480-539 (accumulate_grasp_distance_constraints), fused:
# the target is the jaw sphere's position at this iteration's share of the step, the particle is pulled to its captured
# distance from it along the current direction, the stiffness spread over the iterations.
@wp.kernel
def hold_particles(
    q: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    sphere_q_prev: wp.array2d(dtype=wp.vec3),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_active: wp.array2d(dtype=wp.int32),
    sphere_jaw_sign: wp.array(dtype=wp.float32),
    stiffness: wp.array2d(dtype=wp.float32),
    particle: wp.array3d(dtype=wp.int32),
    sphere: wp.array3d(dtype=wp.int32),
    rest: wp.array3d(dtype=wp.vec3),
    active: wp.array3d(dtype=wp.int32),
    alpha: wp.float32,
    iterations: wp.int32,
    capacity: wp.int32,
    count: wp.int32,
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
    delta_k: wp.array2d(dtype=wp.vec3),
    delta_jaw: wp.array2d(dtype=wp.vec3),
):
    env, idx = wp.tid()
    k = idx // capacity
    g = idx - k * capacity
    if active[env, k, g] == 0:
        return
    local = particle[env, k, g]
    if local < 0:
        return
    flat_index = env * count + local  # the accumulators; the views take (env, local)
    i = local
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        return
    s = sphere[env, k, g]
    if sphere_active[env, s] == 0:
        return
    target = sphere_q_prev[env, s] + (sphere_q[env, s] - sphere_q_prev[env, s]) * alpha
    configured = wp.clamp(stiffness[env, k], 0.0, 1.0)
    a = 1.0 - wp.pow(1.0 - configured, 1.0 / wp.float32(wp.max(iterations, 1)))
    r = rest[env, k, g]
    rest_length = wp.length(r)
    d = q[env, i] - target
    dist = wp.length(d)
    direction = wp.vec3(1.0, 0.0, 0.0)
    if dist > 1.0e-8:
        direction = d / dist
    elif rest_length > 1.0e-8:
        direction = r / rest_length
    goal = target + direction * rest_length
    d = (goal - q[env, i]) * a
    wp.atomic_add(delta, flat_index, d)
    wp.atomic_add(delta_count, flat_index, wp.int32(1))
    wp.atomic_add(delta_k, flat_index, k, d)
    side = sphere_jaw_sign[s]  # the holding jaw sphere's side: +1 left, -1 right
    if side != 0.0:
        wp.atomic_add(delta_jaw, flat_index, k, d * side)
