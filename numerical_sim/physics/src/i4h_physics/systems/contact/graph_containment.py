# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catheter kept inside the lumen of a centreline tree, with the tree pushed back (two-way).

Ported from omniendo/xpbd_rod/kernels_xcath_centerline.py (the packed, fixed-topology kernels; world = env): sample
positions :10-30 (every catheter node plus three interior samples per catheter edge), the motion clamps :33-75,
the candidate cache :78-126 and :195-296 (each sample keeps the tree edges within reach; the cache is rebuilt for
an env when the catheter's or the tree's motion since the last build consumes the ``skin``; a sample with more
edges than ``capacity`` falls back to an exact scan and latches ``overflow``), the cardinal-spline evaluation
:129-192, the linear projection :299-479 (nearest tapered-capsule point over the candidate edges, the activation
shell ``gap <= max_distance``, the open-root entry half-space, the inverse-mass-weighted split between the two
catheter nodes of a sample and the two tree nodes of the edge with ``vessel_response``) and the spline projection
:482-791 (the linear seed's edge and its two path neighbours, four Newton seeds each, the four-control Hermite
coupling with ghost controls folded at path ends), the apply :846-858 (average of the accumulated corrections,
self-clearing). Stage order from batched_xcath.py:1051-1072 and :1199-1283 (the post-constraint stage: clamp,
then per iteration candidates, project, apply, clamp for both bodies); resets from :519-565.

Edits: launches ``dim=(E, n)`` on the two bodies' ``(E, count)`` views of the scene arrays, indexed ``[env, local]``
(the project kernels read both views and write only the system's accumulators; the applies write the positions); the
system's own accumulators and caches are indexed ``env * n + local``; tree topology is body-local (one variant) and
read through the tree's view, candidate slots hold body-local edge ids; the
catheter radius is the sampled node radius from ``scene.radius`` (omniendo's per-world ``collision_radius``; a
uniform rod gives the same value), tree radii are the tree nodes' ``scene.radius``; ``collision_enabled`` is the
system's ``enabled``; a sample on an inactive catheter node returns early; the spline Newton loop runs
``newton_iterations + 1`` evaluations instead of a fixed nine with a predicate (same arithmetic); ``overflow`` is
zeroed once per step (omniendo: per substep) so the observation sees any overflow of the step. The substep-start
reference of both clamps is ``scene.q_prev`` (written by ``particles.predict``), omniendo's ``*_step_start``.
The cache also records the ``max_distance`` and ``skin`` it was built with and rebuilds when a live edit changes
them (omniendo rebuilt on motion only), and ``prepare()`` forces a rebuild and recomputes the open-root indices
so a structural change takes effect on recapture.

List this system after ``catheter_control`` (the track projection first) and after the tree's ``rod_pbd`` (the
tree solve never sees this substep's push, as in the source).
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.checks import check_bool, check_float, check_int
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System

EDGE_SAMPLES = 3  # kernels_xcath_centerline.py:7 CENTERLINE_EDGE_SAMPLES


@dataclass(eq=False)
class GraphContainment(System):
    state_products = ("contact_depth", "contact_count", "overflow")  # per-env products a host observes (System.state_products)
    body: str  # catheter body (a rod asset)
    centreline: str  # tree body (vessel_mesh with a centreline block)
    max_distance: float = 0.09  # live, per env: activation shell (omniendo containment.max_distance)
    skin: float = 0.04  # live, per env: candidate skin (omniendo centerline.candidate_skin)
    target_response: float = 0.5  # live, per env: fraction of the gap the tree takes when two-way
    source_max_delta: float = 0.01  # live, per env: catheter motion clamp per substep
    target_max_delta: float = 0.01  # live, per env: tree motion clamp per substep
    iterations: int = 4  # structural: projections per runner iteration (omniendo collision_iterations)
    capacity: int = 64  # structural: candidate slots per sample (omniendo candidate_capacity)
    candidates: bool = True  # structural: candidate cache (off = exact scan every projection)
    spline: bool = True  # structural: cardinal-spline lumen instead of straight capsules
    tangent_scale: float = 0.35  # structural: cardinal spline tension
    newton_iterations: int = 4  # structural: Newton steps per spline seed
    two_way: bool = True  # structural: push the tree
    open_root: bool = True  # structural: no cap at the root (samples upstream of the root are free)

    structural = System.structural + ("iterations", "capacity", "candidates", "spline", "tangent_scale", "newton_iterations",
                                      "two_way", "open_root")

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        self.body = b = scene.body(self.body)
        self.centreline = t = scene.body(self.centreline)
        if t is b:
            raise ValueError("graph_containment: body and centreline must be different bodies")
        if t.tree is None:
            raise ValueError(f"graph_containment: body {t.name!r} has no centreline tree record")
        if b.count < 1:
            raise ValueError("graph_containment: the source body has no nodes")
        self.iterations = check_int("graph_containment.iterations", self.iterations, 1)
        self.capacity = check_int("graph_containment.capacity", self.capacity, 1)
        self.newton_iterations = check_int("graph_containment.newton_iterations", self.newton_iterations, 0)
        self.tangent_scale = check_float("graph_containment.tangent_scale", self.tangent_scale)
        for name in ("candidates", "spline", "two_way", "open_root"):
            setattr(self, name, check_bool(f"graph_containment.{name}", getattr(self, name)))
        tree = t.tree
        self._roots()
        Pc, Pt, S = b.count, t.count, tree.edges.shape[1]
        self.Pc, self.Pt, self.S = Pc, Pt, S
        self.samples = n = Pc + EDGE_SAMPLES * (Pc - 1)
        scene.check_memory({"candidates": E * n * self.capacity * 4, "samples": E * n * 24, "tree": E * Pt * 40, "source": E * Pc * 16})
        self.freeze("capacity")  # iterations is a launch count: recapture-safe

        def part(name, body):  # the body's (E, count) view
            return scene.field(name, body)

        self.q_cath, self.q_tree = part("q", b), part("q", t)
        self.q_prev_cath, self.q_prev_tree = part("q_prev", b), part("q_prev", t)
        self.inv_mass_cath, self.inv_mass_tree = part("inv_mass", b), part("inv_mass", t)
        self.radius_cath, self.radius_tree = part("radius", b), part("radius", t)
        self.flags_cath, self.flags_tree = part("flags", b), part("flags", t)
        self.edges = wp.array(tree.edges[0], dtype=wp.vec2i, device=dev)  # body-local, one variant
        self.controls = wp.array(tree.spline_controls[0], dtype=wp.vec4i, device=dev)
        self.neighbors = wp.array(tree.spline_neighbors[0], dtype=wp.vec2i, device=dev)
        self.cand_idx = wp.full(E * n * self.capacity, -1, dtype=wp.int32, device=dev)
        self.cand_count = wp.zeros(E * n, dtype=wp.int32, device=dev)
        self.cand_overflow = wp.zeros(E * n, dtype=wp.int32, device=dev)
        self.ref_samples = wp.zeros(E * n, dtype=wp.vec3, device=dev)
        self.ref_tree = wp.zeros(E * Pt, dtype=wp.vec3, device=dev)
        self.max_disp = wp.zeros(2 * E, dtype=wp.float32, device=dev)  # (catheter, tree) motion since the last build
        self.rebuild = wp.ones(E, dtype=wp.int32, device=dev)
        self.built = wp.zeros(E, dtype=wp.vec2, device=dev)  # (max_distance, skin) the cache was built with
        self.cath_corr = wp.zeros(E * Pc, dtype=wp.vec3, device=dev)  # self-clearing accumulators
        self.cath_count = wp.zeros(E * Pc, dtype=wp.float32, device=dev)
        self.tree_corr = wp.zeros(E * Pt, dtype=wp.vec3, device=dev)
        self.tree_count = wp.zeros(E * Pt, dtype=wp.float32, device=dev)
        self.contact_depth = wp.zeros(E, dtype=wp.float32, device=dev)  # deepest gap of the last substep
        self.contact_count = wp.zeros(E, dtype=wp.int32, device=dev)  # samples pushed in the last substep
        self.overflow = wp.zeros(E, dtype=wp.int32, device=dev)  # 1 when a sample fell back to the exact scan this step
        self._live("max_distance", "skin", 'target_response', 'source_max_delta', 'target_max_delta')

    def _roots(self):
        """The open root's node and neighbour (``-1`` when closed); validated against the tree record."""
        self.open_root = check_bool("graph_containment.open_root", self.open_root)
        tree = self.centreline.tree
        self.root, self.root_neighbor = -1, -1
        if self.open_root:
            if tree.root_neighbor < 0:
                raise ValueError("graph_containment.open_root needs a tree built with open_root (a degree-one root)")
            self.root, self.root_neighbor = int(tree.root), int(tree.root_neighbor)


    def readouts(self):
        return {"contact_depth": self.contact_depth, "contact_count": self.contact_count, "candidate_overflow": self.overflow}

    def prepare(self, scene):
        super().prepare(scene)
        self._roots()  # a structural change of open_root takes effect on recapture
        self.rebuild.fill_(1)  # the cache bound depends on the spline setting and the tangent scale

    def step_start(self, scene, dt):
        self.overflow.zero_()

    def substep_start(self, scene, dt, s, substeps):
        self.contact_depth.zero_()
        self.contact_count.zero_()

    def _refresh_candidates(self, scene):
        E, n, Pt, dev = scene.envs, self.samples, self.Pt, scene.device
        # max_disp is zero here: request_rebuild clears it after reading (and allocation and reset start it at zero)
        wp.launch(measure_displacement, dim=(E, max(n, Pt)), inputs=[self.q_cath, self.Pc, n, self.ref_samples, self.q_tree, Pt,
                                                                      self.ref_tree], outputs=[self.max_disp], device=dev)
        wp.launch(request_rebuild, dim=E,
                  inputs=[self.skin, 1.0 + 0.5 * self.tangent_scale if self.spline else 1.0, self.max_distance, self.built],
                  outputs=[self.max_disp, self.rebuild], device=dev)
        wp.launch(build_candidates, dim=(E, n),
                  inputs=[self.q_cath, self.Pc, n, self.q_tree, self.radius_tree, self.edges, self.controls, self.neighbors,
                          self.S, int(self.spline), self.tangent_scale, self.max_distance, self.skin, self.capacity, self.rebuild],
                  outputs=[self.cand_idx, self.cand_count, self.cand_overflow, self.ref_samples], device=dev)
        wp.launch(update_tree_reference, dim=(E, Pt), inputs=[self.q_tree, Pt, self.rebuild], outputs=[self.ref_tree], device=dev)
        wp.launch(finish_rebuild, dim=E, inputs=[self.max_distance, self.skin], outputs=[self.built, self.rebuild], device=dev)

    def iteration(self, scene, dt, k, iterations):
        E, n, Pc, Pt, dev = scene.envs, self.samples, self.Pc, self.Pt, scene.device
        wp.launch(clamp_deltas, dim=(E, Pc), inputs=[self.q_prev_cath, self.inv_mass_cath, self.source_max_delta],
                  outputs=[self.q_cath], device=dev)
        for _ in range(self.iterations):
            if self.candidates:
                self._refresh_candidates(scene)
            common = [self.q_cath, self.inv_mass_cath, self.radius_cath, self.flags_cath, Pc, n,
                      self.q_tree, self.inv_mass_tree, self.radius_tree, Pt, self.edges]
            if self.spline:
                common += [self.controls, self.neighbors]
            common += [self.S, self.root, self.root_neighbor, self.max_distance]
            if self.spline:
                common += [self.tangent_scale, self.newton_iterations]
            common += [int(self.two_way), self.target_response, int(self.candidates), self.capacity, self.cand_idx, self.cand_count,
                       self.cand_overflow]
            wp.launch(project_spline if self.spline else project_linear, dim=(E, n), inputs=common,
                      outputs=[self.cath_corr, self.cath_count, self.tree_corr, self.tree_count, self.contact_depth, self.contact_count,
                               self.overflow],
                      device=dev)
            wp.launch(apply_corrections, dim=(E, Pc),
                      inputs=[self.inv_mass_cath, self.flags_cath, Pc, self.q_prev_cath, self.source_max_delta],
                      outputs=[self.q_cath, self.cath_corr, self.cath_count], device=dev)
            if self.two_way:
                wp.launch(apply_corrections, dim=(E, Pt), inputs=[self.inv_mass_tree, self.flags_tree, Pt, self.q_prev_tree, self.target_max_delta],
                          outputs=[self.q_tree, self.tree_corr, self.tree_count], device=dev)

    def reset(self, scene, mask):
        for arr, value in ((self.contact_depth, wp.float32(0.0)), (self.contact_count, wp.int32(0)), (self.overflow, wp.int32(0)),
                           (self.rebuild, wp.int32(1))):
            self.clear(mask, (arr, value))
        self.clear(mask, (self.max_disp, wp.float32(0.0)))
        self.clear(mask, (self.built, wp.vec2(0.0, 0.0)))
        self.clear(mask, (self.cand_idx, wp.int32(-1)))
        for arr in (self.cand_count, self.cand_overflow):
            self.clear(mask, (arr, wp.int32(0)))
        self.clear(mask, (self.ref_samples, wp.vec3(0.0)))
        self.clear(mask, (self.ref_tree, wp.vec3(0.0)))


# Ported from omniendo kernels_xcath_centerline.py:10-30 (_packed_centerline_sample_position). Edits: returns the two
# catheter nodes (body-local) and the weight instead of the position, so the callers read the radius too.
@wp.func
def sample_anchor(Pc: wp.int32, local: wp.int32):
    if local < Pc:
        return local, local, float(0.0)
    k = local - Pc
    a = k // EDGE_SAMPLES
    return a, a + 1, float(k % EDGE_SAMPLES + 1) / float(EDGE_SAMPLES + 1)


# Ported from omniendo kernels_xcath_centerline.py:33-40 (_clamp_delta). Edits: none.
@wp.func
def clamp_delta(position: wp.vec3, start: wp.vec3, limit: float) -> wp.vec3:
    delta = position - start
    length = wp.length(delta)
    if length > limit and length > 1.0e-12:
        return start + delta * (limit / length)
    return position


# Ported from omniendo kernels_xcath_centerline.py:64-75 (clamp_packed_deltas). Edits: (env, local) prologue.
@wp.kernel
def clamp_deltas(
    starts: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    limits: wp.array(dtype=wp.float32),
    positions: wp.array2d(dtype=wp.vec3),
):
    env, local = wp.tid()
    if inv_mass[env, local] > 0.0:
        positions[env, local] = clamp_delta(positions[env, local], starts[env, local], limits[env])


@wp.func
def sample_point(q: wp.array2d(dtype=wp.vec3), env: wp.int32, ca: wp.int32, cb: wp.int32, t: float) -> wp.vec3:
    return q[env, ca] * (1.0 - t) + q[env, cb] * t


# Ported from omniendo kernels_xcath_centerline.py:78-111 (measure_packed_candidate_displacement,
# measure_packed_centerline_displacement), fused: threads [0, samples) measure the catheter samples, [0, Pt) the tree nodes
# (one launch over max(samples, Pt)). Edits: prologue.
@wp.kernel
def measure_displacement(
    q_cath: wp.array2d(dtype=wp.vec3),
    Pc: wp.int32,
    samples: wp.int32,
    ref_samples: wp.array(dtype=wp.vec3),
    q_tree: wp.array2d(dtype=wp.vec3),
    Pt: wp.int32,
    ref_tree: wp.array(dtype=wp.vec3),
    max_disp: wp.array(dtype=wp.float32),
):
    env, local = wp.tid()
    if local < samples:
        ca, cb, t = sample_anchor(Pc, local)
        point = sample_point(q_cath, env, ca, cb, t)
        wp.atomic_max(max_disp, env * 2, wp.length(point - ref_samples[env * samples + local]))
    if local < Pt:
        wp.atomic_max(max_disp, env * 2 + 1, wp.length(q_tree[env, local] - ref_tree[env * Pt + local]))


# Ported from omniendo kernels_xcath_centerline.py:114-126 (request_packed_candidate_rebuild). Edits: a live edit of
# max_distance or skin since the last build also trips the rebuild (the candidate bound depends on both).
@wp.kernel
def request_rebuild(
    skin: wp.array(dtype=wp.float32),
    target_motion_scale: float,
    max_distance: wp.array(dtype=wp.float32),
    built: wp.array(dtype=wp.vec2),
    max_disp: wp.array(dtype=wp.float32),
    rebuild: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    if max_disp[env * 2] + target_motion_scale * max_disp[env * 2 + 1] >= skin[env]:
        rebuild[env] = 1
    max_disp[env * 2] = 0.0  # consumed: the next measure starts from zero (one thread per env owns both slots)
    max_disp[env * 2 + 1] = 0.0
    if max_distance[env] != built[env][0] or skin[env] != built[env][1]:
        rebuild[env] = 1


# Ported from omniendo kernels_xcath_centerline.py:129-192 (_cardinal_curve and derivatives). Edits: none.
@wp.func
def cardinal_curve(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, u: float, scale: float) -> wp.vec3:
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return p1 * h00 + (p2 - p0) * (scale * h10) + p2 * h01 + (p3 - p1) * (scale * h11)


@wp.func
def cardinal_curve_derivative(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, u: float, scale: float) -> wp.vec3:
    u2 = u * u
    h00 = 6.0 * u2 - 6.0 * u
    h10 = 3.0 * u2 - 4.0 * u + 1.0
    h01 = -6.0 * u2 + 6.0 * u
    h11 = 3.0 * u2 - 2.0 * u
    return p1 * h00 + (p2 - p0) * (scale * h10) + p2 * h01 + (p3 - p1) * (scale * h11)


@wp.func
def cardinal_curve_second_derivative(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, u: float, scale: float) -> wp.vec3:
    h00 = 12.0 * u - 6.0
    h10 = 6.0 * u - 4.0
    h01 = -12.0 * u + 6.0
    h11 = 6.0 * u - 2.0
    return p1 * h00 + (p2 - p0) * (scale * h10) + p2 * h01 + (p3 - p1) * (scale * h11)


# Ported from omniendo kernels_xcath_centerline.py:195-269 (build_packed_centerline_candidates). Edits: (env, sample)
# prologue; tree topology body-local with the env offset at each lookup; body-local edge ids in the slots.
@wp.kernel
def build_candidates(
    q_cath: wp.array2d(dtype=wp.vec3),
    Pc: wp.int32,
    samples: wp.int32,
    q_tree: wp.array2d(dtype=wp.vec3),
    radius_tree: wp.array2d(dtype=wp.float32),
    edges: wp.array(dtype=wp.vec2i),
    controls: wp.array(dtype=wp.vec4i),
    neighbors: wp.array(dtype=wp.vec2i),
    S: wp.int32,
    spline_enabled: wp.int32,
    tangent_scale: float,
    max_distance: wp.array(dtype=wp.float32),
    skin: wp.array(dtype=wp.float32),
    capacity: wp.int32,
    rebuild: wp.array(dtype=wp.int32),
    cand_idx: wp.array(dtype=wp.int32),
    cand_count: wp.array(dtype=wp.int32),
    cand_overflow: wp.array(dtype=wp.int32),
    ref_samples: wp.array(dtype=wp.vec3),
):
    env, local = wp.tid()
    if rebuild[env] == 0:
        return
    sample = env * samples + local
    ca, cb, t = sample_anchor(Pc, local)
    point = sample_point(q_cath, env, ca, cb, t)
    count = int(0)
    for e in range(S):
        edge = edges[e]
        a = q_tree[env, edge[0]]
        b = q_tree[env, edge[1]]
        axis = b - a
        length_sq = wp.dot(axis, axis)
        tt = float(0.0)
        if length_sq > 1.0e-16:
            tt = wp.clamp(wp.dot(point - a, axis) / length_sq, 0.0, 1.0)
        distance = wp.length(point - (a + axis * tt))
        curve_margin = float(0.0)
        if spline_enabled != 0:
            control = controls[e]
            adjacent = neighbors[e]
            p1 = q_tree[env, control[1]]
            p2 = q_tree[env, control[2]]
            p0 = q_tree[env, control[0]]
            p3 = q_tree[env, control[3]]
            if adjacent[0] < 0:
                p0 = p1 * 2.0 - p2
            if adjacent[1] < 0:
                p3 = p2 * 2.0 - p1
            curve_margin = 0.25 * wp.length(p2 - p1) + 0.25 * tangent_scale * (wp.length(p2 - p0) + wp.length(p3 - p1))
        bound = wp.max(radius_tree[env, edge[0]], radius_tree[env, edge[1]]) + max_distance[env] + skin[env] + curve_margin
        if distance <= bound:
            if count < capacity:
                cand_idx[sample * capacity + count] = e
            count = count + 1
    cand_count[sample] = wp.min(count, capacity)
    cand_overflow[sample] = int(count > capacity)  # the project kernels fall back to the exact scan and flag ``overflow``
    ref_samples[sample] = point


# Ported from omniendo kernels_xcath_centerline.py:272-296 (update_packed_candidate_centerline_reference,
# finish_packed_candidate_rebuild). Edits: prologues; aggregate_packed_candidate_overflow is folded into the project kernels.
@wp.kernel
def update_tree_reference(
    q_tree: wp.array2d(dtype=wp.vec3),
    Pt: wp.int32,
    rebuild: wp.array(dtype=wp.int32),
    ref_tree: wp.array(dtype=wp.vec3),
):
    env, local = wp.tid()
    if rebuild[env] != 0:
        node = env * Pt + local
        ref_tree[node] = q_tree[env, local]


@wp.kernel
def finish_rebuild(
    max_distance: wp.array(dtype=wp.float32),
    skin: wp.array(dtype=wp.float32),
    built: wp.array(dtype=wp.vec2),
    rebuild: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    if rebuild[env] != 0:
        built[env] = wp.vec2(max_distance[env], skin[env])
    rebuild[env] = 0


@wp.func
def upstream_of_open_root(q_tree: wp.array2d(dtype=wp.vec3), env: wp.int32, root: wp.int32, root_neighbor: wp.int32, point: wp.vec3) -> bool:
    """kernels_xcath_centerline.py:349-355: a sample behind the root's entry plane is outside the lumen's remit."""
    if root >= 0 and root_neighbor >= 0:
        r = q_tree[env, root]
        inward = q_tree[env, root_neighbor] - r
        if wp.dot(point - r, inward) < 0.0:
            return True
    return False


# Ported from omniendo kernels_xcath_centerline.py:299-480 (project_packed_linear_centerline_containment). Edits:
# (env, sample) prologue on body slices; catheter radius sampled from the nodes; inactive catheter nodes skipped;
# ``collision_enabled`` dropped; ``two_way`` a Python int.
@wp.kernel
def project_linear(
    q_cath: wp.array2d(dtype=wp.vec3),
    inv_mass_cath: wp.array2d(dtype=wp.float32),
    radius_cath: wp.array2d(dtype=wp.float32),
    flags_cath: wp.array2d(dtype=wp.int32),
    Pc: wp.int32,
    samples: wp.int32,
    q_tree: wp.array2d(dtype=wp.vec3),
    inv_mass_tree: wp.array2d(dtype=wp.float32),
    radius_tree: wp.array2d(dtype=wp.float32),
    Pt: wp.int32,
    edges: wp.array(dtype=wp.vec2i),
    S: wp.int32,
    root: wp.int32,
    root_neighbor: wp.int32,
    max_distance: wp.array(dtype=wp.float32),
    two_way: wp.int32,
    target_response: wp.array(dtype=wp.float32),
    use_candidates: wp.int32,
    capacity: wp.int32,
    cand_idx: wp.array(dtype=wp.int32),
    cand_count: wp.array(dtype=wp.int32),
    cand_overflow: wp.array(dtype=wp.int32),
    cath_corr: wp.array(dtype=wp.vec3),
    cath_count: wp.array(dtype=wp.float32),
    tree_corr: wp.array(dtype=wp.vec3),
    tree_count: wp.array(dtype=wp.float32),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    sample = env * samples + local
    cath_base = env * Pc
    la, lb, source_t = sample_anchor(Pc, local)
    ca = cath_base + la
    cb = cath_base + lb
    if (flags_cath[env, la] & ACTIVE) == 0 or (flags_cath[env, lb] & ACTIVE) == 0:
        return
    point = q_cath[env, la] * (1.0 - source_t) + q_cath[env, lb] * source_t
    source_radius = radius_cath[env, la] * (1.0 - source_t) + radius_cath[env, lb] * source_t
    base = env * Pt
    if upstream_of_open_root(q_tree, env, root, root_neighbor, point):
        return

    best_gap = float(1.0e30)
    best_edge = int(-1)
    best_t = float(0.0)
    best_gradient = wp.vec3(0.0)
    search_count = S
    cached = use_candidates != 0 and cand_overflow[sample] == 0
    if cached:
        search_count = cand_count[sample]
    elif use_candidates != 0:
        wp.atomic_max(overflow, env, 1)  # this sample fell back to the exact scan
    for search_index in range(search_count):
        e = search_index
        if cached:
            e = cand_idx[sample * capacity + search_index]
        edge = edges[e]
        a = q_tree[env, edge[0]]
        b = q_tree[env, edge[1]]
        axis = b - a
        length = wp.length(axis)
        if length > 1.0e-8:
            tangent = axis / length
            projected = wp.clamp(wp.dot(point - a, tangent) / length, 0.0, 1.0)
            radial_at_projection = point - (a + axis * projected)
            radial_length = wp.length(radial_at_projection)
            r0 = radius_tree[env, edge[0]]
            r1 = radius_tree[env, edge[1]]
            dr = r1 - r0
            t = projected
            if wp.abs(dr) < length - 1.0e-8:
                denominator = length * wp.sqrt(wp.max(length * length - dr * dr, 1.0e-12))
                t = wp.clamp(projected + dr * radial_length / denominator, 0.0, 1.0)
            closest = a + axis * t
            delta = point - closest
            distance = wp.length(delta)
            radius = r0 * (1.0 - t) + r1 * t
            gap = distance - (radius - source_radius)
            if gap < best_gap:
                radial = wp.vec3(1.0, 0.0, 0.0)
                if distance > 1.0e-8:
                    radial = delta / distance
                gradient = radial - tangent * (dr / length)
                gradient_length = wp.length(gradient)
                if gradient_length > 1.0e-8:
                    gradient = gradient / gradient_length
                best_gap = gap
                best_edge = e
                best_t = t
                best_gradient = gradient

    if best_edge < 0 or best_gap <= 0.0 or best_gap > max_distance[env]:
        return
    target_edge = edges[best_edge]
    v0 = base + target_edge[0]  # tree accumulator slots; the views take vessel_edge[.]
    v1 = base + target_edge[1]
    source_a_weight = 1.0 - source_t
    source_b_weight = source_t
    if ca == cb:
        source_a_weight = 1.0
        source_b_weight = 0.0
    target_a_weight = 1.0 - best_t
    target_b_weight = best_t
    wc0 = inv_mass_cath[env, la]
    wc1 = inv_mass_cath[env, lb]
    wv0 = inv_mass_tree[env, target_edge[0]]
    wv1 = inv_mass_tree[env, target_edge[1]]
    reciprocal = float(0.0)
    if two_way != 0:
        reciprocal = target_response[env]
    denominator = (wc0 * source_a_weight * source_a_weight + wc1 * source_b_weight * source_b_weight
                   + reciprocal * (wv0 * target_a_weight * target_a_weight + wv1 * target_b_weight * target_b_weight))
    if denominator <= 1.0e-12:
        return
    magnitude = best_gap / denominator
    if wc0 > 0.0 and source_a_weight > 0.0:
        wp.atomic_add(cath_corr, ca, -best_gradient * magnitude * wc0 * source_a_weight)
        wp.atomic_add(cath_count, ca, source_a_weight)
    if wc1 > 0.0 and source_b_weight > 0.0:
        wp.atomic_add(cath_corr, cb, -best_gradient * magnitude * wc1 * source_b_weight)
        wp.atomic_add(cath_count, cb, source_b_weight)
    if reciprocal > 0.0 and wv0 > 0.0 and target_a_weight > 0.0:
        wp.atomic_add(tree_corr, v0, best_gradient * magnitude * reciprocal * wv0 * target_a_weight)
        wp.atomic_add(tree_count, v0, target_a_weight)
    if reciprocal > 0.0 and wv1 > 0.0 and target_b_weight > 0.0:
        wp.atomic_add(tree_corr, v1, best_gradient * magnitude * reciprocal * wv1 * target_b_weight)
        wp.atomic_add(tree_count, v1, target_b_weight)
    wp.atomic_max(contact_depth, env, best_gap)
    wp.atomic_add(contact_count, env, 1)


# Ported from omniendo kernels_xcath_centerline.py:483-791 (project_packed_spline_centerline_containment). Edits: as
# project_linear; the Newton loop runs ``newton_iterations + 1`` evaluations (the source's ``range(9)`` with the
# ``iteration <= newton_iterations`` predicate), the last one evaluates without stepping. Unlike project_linear, a
# correction counts 1 (not its barycentric weight) toward the averaging, as in the source (:754-789 vs :456-477).
@wp.kernel
def project_spline(
    q_cath: wp.array2d(dtype=wp.vec3),
    inv_mass_cath: wp.array2d(dtype=wp.float32),
    radius_cath: wp.array2d(dtype=wp.float32),
    flags_cath: wp.array2d(dtype=wp.int32),
    Pc: wp.int32,
    samples: wp.int32,
    q_tree: wp.array2d(dtype=wp.vec3),
    inv_mass_tree: wp.array2d(dtype=wp.float32),
    radius_tree: wp.array2d(dtype=wp.float32),
    Pt: wp.int32,
    edges: wp.array(dtype=wp.vec2i),
    controls: wp.array(dtype=wp.vec4i),
    neighbors: wp.array(dtype=wp.vec2i),
    S: wp.int32,
    root: wp.int32,
    root_neighbor: wp.int32,
    max_distance: wp.array(dtype=wp.float32),
    tangent_scale: float,
    newton_iterations: wp.int32,
    two_way: wp.int32,
    target_response: wp.array(dtype=wp.float32),
    use_candidates: wp.int32,
    capacity: wp.int32,
    cand_idx: wp.array(dtype=wp.int32),
    cand_count: wp.array(dtype=wp.int32),
    cand_overflow: wp.array(dtype=wp.int32),
    cath_corr: wp.array(dtype=wp.vec3),
    cath_count: wp.array(dtype=wp.float32),
    tree_corr: wp.array(dtype=wp.vec3),
    tree_count: wp.array(dtype=wp.float32),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    sample = env * samples + local
    cath_base = env * Pc
    la, lb, source_t = sample_anchor(Pc, local)
    ca = cath_base + la
    cb = cath_base + lb
    if (flags_cath[env, la] & ACTIVE) == 0 or (flags_cath[env, lb] & ACTIVE) == 0:
        return
    point = q_cath[env, la] * (1.0 - source_t) + q_cath[env, lb] * source_t
    source_radius = radius_cath[env, la] * (1.0 - source_t) + radius_cath[env, lb] * source_t
    base = env * Pt
    if upstream_of_open_root(q_tree, env, root, root_neighbor, point):
        return

    linear_gap = float(1.0e30)
    linear_edge = int(-1)
    linear_t = float(0.0)
    search_count = S
    cached = use_candidates != 0 and cand_overflow[sample] == 0
    if cached:
        search_count = cand_count[sample]
    elif use_candidates != 0:
        wp.atomic_max(overflow, env, 1)  # this sample fell back to the exact scan
    for search_index in range(search_count):
        e = search_index
        if cached:
            e = cand_idx[sample * capacity + search_index]
        edge = edges[e]
        a = q_tree[env, edge[0]]
        b = q_tree[env, edge[1]]
        axis = b - a
        length = wp.length(axis)
        if length > 1.0e-8:
            tangent = axis / length
            projected = wp.clamp(wp.dot(point - a, tangent) / length, 0.0, 1.0)
            radial_length = wp.length(point - (a + axis * projected))
            r0 = radius_tree[env, edge[0]]
            r1 = radius_tree[env, edge[1]]
            dr = r1 - r0
            t = projected
            if wp.abs(dr) < length - 1.0e-8:
                denominator = length * wp.sqrt(wp.max(length * length - dr * dr, 1.0e-12))
                t = wp.clamp(projected + dr * radial_length / denominator, 0.0, 1.0)
            distance = wp.length(point - (a + axis * t))
            radius = r0 * (1.0 - t) + r1 * t
            gap = distance - (radius - source_radius)
            if gap < linear_gap:
                linear_gap = gap
                linear_edge = e
                linear_t = t
    if linear_edge < 0:
        return

    best_gap = float(1.0e30)
    best_edge = int(-1)
    best_u = float(0.0)
    best_gradient = wp.vec3(0.0)
    seed_neighbors = neighbors[linear_edge]
    for candidate_slot in range(3):
        candidate_edge = linear_edge
        if candidate_slot == 1:
            candidate_edge = seed_neighbors[0]
        elif candidate_slot == 2:
            candidate_edge = seed_neighbors[1]
        if candidate_edge >= 0:
            control = controls[candidate_edge]
            adjacent = neighbors[candidate_edge]
            p1 = q_tree[env, control[1]]
            p2 = q_tree[env, control[2]]
            p0 = q_tree[env, control[0]]
            p3 = q_tree[env, control[3]]
            if adjacent[0] < 0:
                p0 = p1 * 2.0 - p2
            if adjacent[1] < 0:
                p3 = p2 * 2.0 - p1
            r1 = radius_tree[env, control[1]]
            r2 = radius_tree[env, control[2]]
            dr = r2 - r1
            candidate_seed = linear_t
            if candidate_slot == 0:
                linear_nodes = edges[linear_edge]
                if linear_nodes[0] != control[1]:
                    candidate_seed = 1.0 - linear_t
            elif candidate_slot == 1:
                candidate_seed = 1.0
            elif candidate_slot == 2:
                candidate_seed = 0.0

            for start_slot in range(4):
                u = candidate_seed
                if start_slot == 1:
                    u = 0.5
                elif start_slot == 2:
                    u = 0.0
                elif start_slot == 3:
                    u = 1.0
                for iteration in range(newton_iterations + 1):
                    curve = cardinal_curve(p0, p1, p2, p3, u, tangent_scale)
                    delta = point - curve
                    distance = wp.length(delta)
                    gap = distance - ((r1 * (1.0 - u) + r2 * u) - source_radius)
                    if wp.isfinite(gap) and gap < best_gap:
                        gradient = wp.vec3(1.0, 0.0, 0.0)
                        if distance > 1.0e-8:
                            gradient = delta / distance
                        best_gap = gap
                        best_edge = candidate_edge
                        best_u = u
                        best_gradient = gradient
                    if iteration < newton_iterations and distance > 1.0e-8:
                        derivative = cardinal_curve_derivative(p0, p1, p2, p3, u, tangent_scale)
                        second = cardinal_curve_second_derivative(p0, p1, p2, p3, u, tangent_scale)
                        projection = wp.dot(delta, derivative)
                        objective_derivative = -projection / distance - dr
                        objective_second = ((wp.dot(derivative, derivative) - wp.dot(delta, second)) / distance
                                            - projection * projection / (distance * distance * distance))
                        if objective_second > 1.0e-8 and wp.isfinite(objective_second):
                            step = wp.clamp(objective_derivative / objective_second, -0.25, 0.25)
                            next_u = wp.clamp(u - step, 0.0, 1.0)
                            if wp.isfinite(next_u):
                                u = next_u

    if best_edge < 0 or best_gap <= 0.0 or best_gap > max_distance[env]:
        return

    control = controls[best_edge]
    adjacent = neighbors[best_edge]
    u = best_u
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    w0 = -tangent_scale * h10
    w1 = h00 - tangent_scale * h11
    w2 = h01 + tangent_scale * h10
    w3 = tangent_scale * h11
    if adjacent[0] < 0:
        w1 = w1 + 2.0 * w0
        w2 = w2 - w0
        w0 = 0.0
    if adjacent[1] < 0:
        w2 = w2 + 2.0 * w3
        w1 = w1 - w3
        w3 = 0.0

    source_a_weight = 1.0 - source_t
    source_b_weight = source_t
    if ca == cb:
        source_a_weight = 1.0
        source_b_weight = 0.0
    n0 = base + control[0]
    n1 = base + control[1]
    n2 = base + control[2]
    n3 = base + control[3]
    wc0 = inv_mass_cath[env, la]
    wc1 = inv_mass_cath[env, lb]
    wv0 = inv_mass_tree[env, control[0]]
    wv1 = inv_mass_tree[env, control[1]]
    wv2 = inv_mass_tree[env, control[2]]
    wv3 = inv_mass_tree[env, control[3]]
    reciprocal = float(0.0)
    if two_way != 0:
        reciprocal = target_response[env]
    denominator = (wc0 * source_a_weight * source_a_weight + wc1 * source_b_weight * source_b_weight
                   + reciprocal * (wv0 * w0 * w0 + wv1 * w1 * w1 + wv2 * w2 * w2 + wv3 * w3 * w3))
    if denominator <= 1.0e-12 or not wp.isfinite(denominator):
        return
    magnitude = best_gap / denominator
    if wc0 > 0.0 and source_a_weight > 0.0:
        wp.atomic_add(cath_corr, ca, -best_gradient * magnitude * wc0 * source_a_weight)
        wp.atomic_add(cath_count, ca, 1.0)
    if wc1 > 0.0 and source_b_weight > 0.0:
        wp.atomic_add(cath_corr, cb, -best_gradient * magnitude * wc1 * source_b_weight)
        wp.atomic_add(cath_count, cb, 1.0)
    if reciprocal > 0.0 and wv0 > 0.0 and wp.abs(w0) > 1.0e-8:
        wp.atomic_add(tree_corr, n0, best_gradient * magnitude * reciprocal * wv0 * w0)
        wp.atomic_add(tree_count, n0, 1.0)
    if reciprocal > 0.0 and wv1 > 0.0 and wp.abs(w1) > 1.0e-8:
        wp.atomic_add(tree_corr, n1, best_gradient * magnitude * reciprocal * wv1 * w1)
        wp.atomic_add(tree_count, n1, 1.0)
    if reciprocal > 0.0 and wv2 > 0.0 and wp.abs(w2) > 1.0e-8:
        wp.atomic_add(tree_corr, n2, best_gradient * magnitude * reciprocal * wv2 * w2)
        wp.atomic_add(tree_count, n2, 1.0)
    if reciprocal > 0.0 and wv3 > 0.0 and wp.abs(w3) > 1.0e-8:
        wp.atomic_add(tree_corr, n3, best_gradient * magnitude * reciprocal * wv3 * w3)
        wp.atomic_add(tree_count, n3, 1.0)
    wp.atomic_max(contact_depth, env, best_gap)
    wp.atomic_add(contact_count, env, 1)


# Ported from omniendo kernels_xcath_centerline.py:846-859 (apply_centerline_corrections). Edits: (env, local)
# prologue; an inactive node takes no correction; the motion clamp that followed (clamp_packed_deltas) is folded in.
@wp.kernel
def apply_corrections(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    count: wp.int32,
    starts: wp.array2d(dtype=wp.vec3),
    limits: wp.array(dtype=wp.float32),
    positions: wp.array2d(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    correction_counts: wp.array(dtype=wp.float32),
):
    env, local = wp.tid()
    node = env * count + local
    c = correction_counts[node]
    if inv_mass[env, local] > 0.0:
        p = positions[env, local]
        if c > 0.0 and (flags[env, local] & ACTIVE) != 0:
            p = p + corrections[node] / c
        positions[env, local] = clamp_delta(p, starts[env, local], limits[env])
    corrections[node] = wp.vec3(0.0)
    correction_counts[node] = 0.0
