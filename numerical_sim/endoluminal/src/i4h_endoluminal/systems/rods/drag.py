# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mouse dragging of a centreline-tree node: the picked node follows a target, its geodesic neighbourhood follows
with a cosine falloff, everything else is left to the tree solver.

Ported from omniendo xcath/vessel_drag.py:219-257 (``apply_centerline_drag``: desired position = baseline +
(target - baseline[selected]) * weight, clamped to ``max_delta`` from the substep-start position, written to
both the committed and the predicted positions so the dragged nodes carry no velocity) and :398-427
(``_compute_centerline_weights``: Dijkstra over the rest edge lengths within four median edge lengths, weight
0.5 (1 + cos(pi d / r)), zero on pinned nodes, one at the pick). Edits: one selection per env (``active``,
``node``, ``target`` are ``(E,)``; the baseline and weights are body-local ``(E * P)``), the ``(env, node)``
prologue, the substep-start reference is the system's own copy ``ref`` of the body's ``q_prev`` taken before the
drag writes (omniendo's ``step_start`` array; the clamp reference must not move within the substep, and
``q_prev`` is then rewritten so the dragged nodes carry no velocity, omniendo's ``current`` write),
``max_delta`` a live per-env value; omniendo's release-pending flag (:282-286, one more application after the
mouse-up inside the step) is dropped: ``begin``/``move``/``release`` run between steps through the worker's edit
queue, so a release simply stops the pin. The ``kind`` switch is dropped too (the deformable-vertex drag is not
carried; a shell body is not a tree).

``substep_start`` pins the drag after the prediction (list this system after ``particles``); ``iteration`` re-pins
after the tree sweeps and the containment push (list it after ``rod_pbd`` and ``lumen_containment``).
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.systems.base import System


def geodesic_weights(positions, edges, inv_mass, node: int, radius=None) -> np.ndarray:
    """vessel_drag.py:398-427: ``(P,)`` weights of ``node``'s neighbourhood over the graph; ``radius`` defaults to four
    median edge lengths; pinned nodes get 0, the pick 1 (0 when pinned)."""
    P = np.asarray(positions, dtype=np.float64)
    lengths = np.linalg.norm(P[edges[:, 1]] - P[edges[:, 0]], axis=1)
    adjacency = defaultdict(list)
    for (a, b), L in zip(edges.tolist(), lengths.tolist()):
        if np.isfinite(L) and L > 0.0:
            adjacency[a].append((b, L))
            adjacency[b].append((a, L))
    positive = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    radius = 4.0 * float(np.median(positive)) if radius is None and len(positive) else float(radius or 0.0)
    weights = np.zeros(len(P), np.float32)
    if radius <= 0.0 or not adjacency:
        weights[node] = 1.0 if inv_mass[node] > 0.0 else 0.0
        return weights
    dist = np.full(len(P), np.inf)
    dist[node] = 0.0
    pending = [(0.0, int(node))]
    while pending:
        d, n = heapq.heappop(pending)
        if d != dist[n] or d > radius:
            continue
        for other, L in adjacency[n]:
            c = d + L
            if c <= radius and c < dist[other]:
                dist[other] = c
                heapq.heappush(pending, (c, other))
    within = dist <= radius
    weights[within] = 0.5 * (1.0 + np.cos(np.pi * dist[within] / radius))
    weights[np.asarray(inv_mass) <= 0.0] = 0.0
    if inv_mass[node] > 0.0:
        weights[node] = 1.0
    return weights


@dataclass(eq=False)
class Drag(System):
    body: str  # the centreline-tree body
    max_delta: float = 0.01  # live, per env: motion clamp per substep (omniendo vessel_max_delta)

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        self.body = b = scene.body(self.body)
        if b.tree is None:
            raise ValueError(f"drag: body {b.name!r} is not a centreline tree")
        self.P = P = b.count
        self.active = wp.zeros(E, dtype=wp.int32, device=dev)
        self.node = wp.zeros(E, dtype=wp.int32, device=dev)
        self.target = wp.zeros(E, dtype=wp.vec3, device=dev)
        self.weights = wp.zeros(E * P, dtype=wp.float32, device=dev)
        self.baseline = wp.zeros(E * P, dtype=wp.vec3, device=dev)
        self.ref = wp.zeros(E * P, dtype=wp.vec3, device=dev)  # the body's substep-start positions, frozen before the drag writes
        self._live("max_delta")
        self._rest_inv_mass = b.rest.inv_mass[0]

    # host API: run between steps on the physics thread (the app's worker.edit) ------------------------------------

    def begin(self, env: int, node: int, target) -> None:
        """Pick ``node`` of ``env``: freeze the body's current positions as the baseline, weight the neighbourhood."""
        scene, E, P = self.scene, self.scene.envs, self.P
        if not 0 <= int(env) < E or not 0 <= int(node) < P:
            raise ValueError(f"drag.begin: env {env} or node {node} out of range")
        t = self.body.tree
        w = geodesic_weights(t.positions[0], t.edges[0], self._rest_inv_mass, int(node))
        self.weights.reshape((E, P))[int(env)].assign(w)
        wp.copy(self.baseline.reshape((E, P))[int(env)], scene.field("q", self.body)[int(env)])  # one env's row: contiguous
        self.node.reshape((E, 1))[int(env)].assign(np.asarray([int(node)], np.int32))
        self.move(env, target)
        self.active.reshape((E, 1))[int(env)].assign(np.asarray([1], np.int32))

    def move(self, env: int, target) -> None:
        self.target.reshape((self.scene.envs, 1))[int(env)].assign(np.asarray([target], np.float32).reshape(1, 3))

    def release(self, env: int) -> None:
        self.active.reshape((self.scene.envs, 1))[int(env)].assign(np.asarray([0], np.int32))

    # stage hooks --------------------------------------------------------------------------------------------------


    def substep_start(self, scene, dt, s, substeps):
        b, E = self.body, scene.envs
        # after particles.predict wrote q_prev, before the drag: one gather from the body's view (no per-env copies)
        wp.launch(gather_rows, dim=(E, b.count), inputs=[scene.field("q_prev", b), b.count], outputs=[self.ref], device=scene.device)
        wp.launch(apply_drag, dim=(E, b.count),
                  inputs=[scene.field("inv_mass", b), self.active, self.node, self.target, self.baseline, self.weights, self.max_delta,
                          self.ref, b.count, 1],
                  outputs=[scene.field("q", b), scene.field("q_prev", b)], device=scene.device)

    def iteration(self, scene, dt, k, iterations):
        b = self.body
        wp.launch(apply_drag, dim=(scene.envs, b.count),
                  inputs=[scene.field("inv_mass", b), self.active, self.node, self.target, self.baseline, self.weights, self.max_delta,
                          self.ref, b.count, 0],
                  outputs=[scene.field("q", b), scene.field("q_prev", b)], device=scene.device)

    def reset(self, scene, mask):
        self.clear(mask, (self.active, wp.int32(0)))


# Ported from omniendo xcath/vessel_drag.py:219-257 (apply_centerline_drag). Edits: (env, node) prologue on the body's
# slice; per-env controls; ``ref`` (body-local) is the frozen substep-start reference, omniendo's ``step_start``;
# ``write_prev`` 1 in substep_start (omniendo's ``current`` write: the dragged nodes end the substep with no
# velocity), 0 in the iterations (the same pinned position is rewritten after the other systems moved the node).
@wp.kernel
def apply_drag(
    inv_mass: wp.array2d(dtype=wp.float32),
    active: wp.array(dtype=wp.int32),
    node: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.vec3),
    baseline: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=wp.float32),
    max_delta: wp.array(dtype=wp.float32),
    ref: wp.array(dtype=wp.vec3),
    P: wp.int32,
    write_prev: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    q_prev: wp.array2d(dtype=wp.vec3),
):
    env, local = wp.tid()
    if active[env] == 0:
        return
    selected = node[env]
    if selected < 0 or selected >= P:
        return
    i = local
    li = env * P + local
    if inv_mass[env, i] <= 0.0 or weights[li] <= 0.0:
        return
    limit = max_delta[env]
    lsel = env * P + selected
    selected_target = target[env]
    selected_delta = selected_target - ref[lsel]
    selected_delta_length = wp.length(selected_delta)
    if selected_delta_length > limit and selected_delta_length > 1.0e-12:
        selected_target = ref[lsel] + selected_delta * (limit / selected_delta_length)
    displacement = selected_target - baseline[lsel]
    desired = baseline[li] + displacement * weights[li]
    delta = desired - ref[li]
    length = wp.length(delta)
    if length > limit and length > 1.0e-12:
        desired = ref[li] + delta * (limit / length)
    q[env, i] = desired
    if write_prev != 0:
        q_prev[env, i] = desired


@wp.kernel
def gather_rows(src: wp.array2d(dtype=wp.vec3), count: wp.int32, dst: wp.array(dtype=wp.vec3)):
    """``dst[env * count + local] = src[env, local]``: a body's (E, count) view into a body-local array."""
    env, local = wp.tid()
    dst[env * count + local] = src[env, local]
