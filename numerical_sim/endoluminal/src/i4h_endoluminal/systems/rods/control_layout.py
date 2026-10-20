# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored control spheres shaping a centreline tree: each sphere pulls the nodes around its pinned node ``pid``
toward a line through its origin along its local z (with a hop falloff), and scales the tree rod's stretch and
bend stiffness (and rest bend) around it. Several layouts serve the envs round-robin: layout ``env % L``.

Ported from omniendo xcath/centerline_controls.py: the projection kernel :21-45 (``target = origin + direction *
arc_offset``, ``alpha = stiffness * weight``, corrections averaged over the spheres that act on a node), the static
tables :103-141 (``root_arc`` geodesic arc lengths, ``hops`` BFS from each ``pid``), ``sync_values`` :231-268
(``weight = falloff ** hops``, ``falloff 0`` keeps only the pinned node, direction = rotation applied to local z)
and ``apply_materials`` :270-339 (segment weight = max of its node weights, pair weight = max of its segment
weights, ``stretch = max(0, 1 + sum w (m - 1))``, likewise bend; rest bend rotated into the rest Darboux), the
per-world layout choice ``layouts[world % len(layouts)]`` (batched_xcath.py:1542). Edits: one thread per node sums
its spheres (no accumulator: each node has one writer), the material tables are device kernels run every step
(``sync_values``/``apply_materials`` were host code rerun on edits; live edits of the ``(L, C)`` arrays are then
capture-safe), per-layout immutable tables ``(L, C, P)`` read through ``layout_of_env``. Recorded deviation: the
rest bend is added to the rod's live ``bend_offset`` as half the rotation vector ``0.5 * sum w (bx, by, 0)``
(rod_pbd adds it to the imaginary part of the rest Darboux quaternion), the small-angle form of omniendo's
post-multiplied quaternion; exact for a straight rest.

List after the tree's ``rod_pbd`` (omniendo projects after the Cosserat sweeps) and before ``lumen_containment``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.systems.base import System
from i4h_physics.systems.rods.rod_pbd import RodPbd


def hop_distances(edges: np.ndarray, P: int, pid: int) -> np.ndarray:
    """centerline_controls.py:126-141: BFS hop count from ``pid`` over the graph."""
    adjacency = [[] for _ in range(P)]
    for a, b in edges.tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)
    hops = np.full(P, -1, np.int32)
    hops[pid] = 0
    queue = deque([pid])
    while queue:
        node = queue.popleft()
        for other in adjacency[node]:
            if hops[other] < 0:
                hops[other] = hops[node] + 1
                queue.append(other)
    if (hops < 0).any():
        raise ValueError("control layout: the centreline tree is not connected")
    return hops


def quat_rotate(q, v):  # centerline_controls.py:80-84
    q, v = np.asarray(q, np.float64), np.asarray(v, np.float64)
    u = q[:3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[3] * v)


@dataclass(eq=False)
class ControlLayout(System):
    centreline: str  # the tree body (its record carries the layouts)
    rod: object = None  # optional legacy ref, validated only at construction; writes body-owned controls

    refs = ("rod",)

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        self.centreline = t = scene.body(self.centreline)
        if t.tree is None:
            raise ValueError(f"control_layout: body {t.name!r} is not a centreline tree")
        if self.rod is not None and (not isinstance(self.rod, RodPbd) or self.rod.body is not t):
            raise TypeError("control_layout.rod must name the rod_pbd system of the same tree body")
        if t.rod_controls is None:
            raise ValueError("control_layout requires a rod_pbd on the same body before it")
        self.controls = t.rod_controls
        self.S, self.K = t.tree.edges.shape[1], t.tree.bend_pairs.shape[1]
        tree = t.tree
        layouts = t.anatomy.layouts
        if not layouts:
            raise ValueError(f"control_layout: body {t.name!r} has no control layouts (centreline.control_layouts)")
        P, edges = t.count, tree.edges[0]
        L, C = len(layouts), max(len(spheres) for spheres in layouts)
        if C == 0:
            raise ValueError("control_layout: every layout is empty")
        self.L, self.C = L, C
        hops = np.zeros((L, C, P), np.int32)
        arc = np.zeros((L, C, P), np.float32)
        origin, direction = np.zeros((L, C, 3), np.float32), np.tile(np.float32([0.0, 0.0, 1.0]), (L, C, 1))
        stiffness, falloff, bend_mult, stretch_mult = (np.zeros((L, C), np.float32) for _ in range(4))
        stretch_mult[:] = 1.0
        bend_mult[:] = 1.0
        rest_bend, active = np.zeros((L, C, 2), np.float32), np.zeros((L, C), np.int32)
        root_arc = tree.root_arc[0]
        for layout_index, spheres in enumerate(layouts):
            for c, s in enumerate(spheres):
                pid = int(s["pid"])
                hops[layout_index, c] = hop_distances(edges, P, pid)
                arc[layout_index, c] = root_arc - root_arc[pid]
                origin[layout_index, c] = s["position"]
                q = np.asarray(s["rotation"], np.float64)
                norm = float(np.linalg.norm(q))
                d = quat_rotate(q / norm, [0.0, 0.0, 1.0]) if norm > 1e-8 and np.isfinite(q).all() else np.float64([0.0, 0.0, 1.0])
                if np.linalg.norm(d) > 1e-8 and np.isfinite(d).all():
                    direction[layout_index, c] = d / np.linalg.norm(d)
                stiffness[layout_index, c] = np.clip(s["stiffness"], 0.0, 1.0)
                falloff[layout_index, c] = np.clip(s["falloff"], 0.0, 1.0)
                rest_bend[layout_index, c] = s.get("rest_bend", (0.0, 0.0))
                bend_mult[layout_index, c] = s.get("bend_stiffness_multiplier", 1.0)
                stretch_mult[layout_index, c] = s.get("stretch_stiffness_multiplier", 1.0)
                active[layout_index, c] = int(bool(s["active"]))
        def arr(a, dtype):
            return wp.array(a, dtype=dtype, device=dev)
        self.hops, self.arc_offset = arr(hops, wp.int32), arr(arc, wp.float32)  # (L, C, P) immutable
        self.origin, self.direction = arr(origin, wp.vec3), arr(direction, wp.vec3)  # (L, C) live
        self.stiffness, self.falloff = arr(stiffness, wp.float32), arr(falloff, wp.float32)
        self.rest_bend, self.bend_mult, self.stretch_mult = arr(rest_bend, wp.vec2), arr(bend_mult, wp.float32), arr(stretch_mult, wp.float32)
        self.active = arr(active, wp.int32)
        self.layout_of_env = arr(np.arange(E, dtype=np.int32) % L, wp.int32)
        self.edges = arr(edges, wp.vec2i)  # body-local (S,)
        self.pairs = arr(tree.bend_pairs[0], wp.vec2i)  # (K,)

    def prepare(self, scene):
        super().prepare(scene)
        self.controls.stretch_scale.fill_(1.0)
        self.controls.bend_scale.fill_(1.0)
        self.controls.bend_offset.zero_()

    def step_start(self, scene, dt):
        rod, E, dev = self.controls, scene.envs, scene.device
        wp.launch(materials, dim=(E, self.S), inputs=[self.layout_of_env, self.active, self.falloff, self.stretch_mult, self.hops, self.edges, self.C],
                  outputs=[rod.stretch_scale], device=dev)
        if self.K:
            wp.launch(materials_pairs, dim=(E, self.K),
                      inputs=[self.layout_of_env, self.active, self.falloff, self.bend_mult, self.rest_bend, self.hops, self.edges, self.pairs, self.C],
                      outputs=[rod.bend_scale, rod.bend_offset], device=dev)

    def iteration(self, scene, dt, k, iterations):
        t = self.centreline
        wp.launch(project, dim=(scene.envs, t.count),
                  inputs=[scene.field("inv_mass", t), self.layout_of_env, self.active, self.origin, self.direction, self.stiffness,
                          self.falloff, self.hops, self.arc_offset, self.C],
                  outputs=[scene.field("q", t)], device=scene.device)


@wp.func
def node_weight(falloff: wp.float32, hops: wp.int32) -> wp.float32:
    """centerline_controls.py:250-255: ``falloff ** hops``; a zero falloff keeps only the pinned node."""
    if falloff <= 0.0:
        if hops == 0:
            return 1.0
        return 0.0
    return wp.pow(falloff, wp.float32(hops))


# Ported from omniendo centerline_controls.py:21-45 (_project_centerline_control_layout_kernel) with the average of
# :48-57 folded in. Edits: (env, node) prologue, one thread per node looping over its layout's spheres.
@wp.kernel
def project(
    inv_mass: wp.array2d(dtype=wp.float32),
    layout_of_env: wp.array(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    origin: wp.array2d(dtype=wp.vec3),
    direction: wp.array2d(dtype=wp.vec3),
    stiffness: wp.array2d(dtype=wp.float32),
    falloff: wp.array2d(dtype=wp.float32),
    hops: wp.array3d(dtype=wp.int32),
    arc_offset: wp.array3d(dtype=wp.float32),
    C: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
):
    env, node = wp.tid()
    i = node
    if inv_mass[env, i] <= 0.0:
        return
    layout_index = layout_of_env[env]
    p = q[env, i]
    total = wp.vec3(0.0)
    count = float(0.0)
    for c in range(C):
        if active[layout_index, c] == 0:
            continue
        alpha = stiffness[layout_index, c] * node_weight(falloff[layout_index, c], hops[layout_index, c, node])
        if alpha <= 0.0:
            continue
        target = origin[layout_index, c] + direction[layout_index, c] * arc_offset[layout_index, c, node]
        total += (target - p) * alpha
        count += 1.0
    if count > 0.0:
        q[env, i] = p + total / count


@wp.func
def segment_weight(falloff: wp.float32, hops: wp.array3d(dtype=wp.int32), layout_index: wp.int32, c: wp.int32, e: wp.vec2i) -> wp.float32:
    """centerline_controls.py:293-296: a segment takes the larger of its two node weights."""
    return wp.max(node_weight(falloff, hops[layout_index, c, e[0]]), node_weight(falloff, hops[layout_index, c, e[1]]))


# Ported from omniendo centerline_controls.py:302-315 (apply_materials, the stretch multiplier). Edits: device kernel
# over (env, segment); the clamp at zero as the source.
@wp.kernel
def materials(
    layout_of_env: wp.array(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    falloff: wp.array2d(dtype=wp.float32),
    stretch_mult: wp.array2d(dtype=wp.float32),
    hops: wp.array3d(dtype=wp.int32),
    edges: wp.array(dtype=wp.vec2i),
    C: wp.int32,
    stretch_scale: wp.array2d(dtype=wp.float32),
):
    env, s = wp.tid()
    layout_index = layout_of_env[env]
    delta = float(0.0)
    for c in range(C):
        if active[layout_index, c] != 0:
            delta += segment_weight(falloff[layout_index, c], hops, layout_index, c, edges[s]) * (stretch_mult[layout_index, c] - 1.0)
    stretch_scale[env, s] = wp.max(0.0, 1.0 + delta)


# Ported from omniendo centerline_controls.py:308-335 (apply_materials, the bend multiplier and rest bend). Edits:
# device kernel over (env, pair); the rest bend becomes half a rotation vector in the rod's live bend_offset.
@wp.kernel
def materials_pairs(
    layout_of_env: wp.array(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    falloff: wp.array2d(dtype=wp.float32),
    bend_mult: wp.array2d(dtype=wp.float32),
    rest_bend: wp.array2d(dtype=wp.vec2),
    hops: wp.array3d(dtype=wp.int32),
    edges: wp.array(dtype=wp.vec2i),
    pairs: wp.array(dtype=wp.vec2i),
    C: wp.int32,
    bend_scale: wp.array2d(dtype=wp.float32),
    bend_offset: wp.array2d(dtype=wp.vec3),
):
    env, k = wp.tid()
    layout_index = layout_of_env[env]
    pr = pairs[k]
    delta = float(0.0)
    bend = wp.vec2(0.0, 0.0)
    for c in range(C):
        if active[layout_index, c] != 0:
            w = wp.max(segment_weight(falloff[layout_index, c], hops, layout_index, c, edges[pr[0]]), segment_weight(falloff[layout_index, c], hops, layout_index, c, edges[pr[1]]))
            delta += w * (bend_mult[layout_index, c] - 1.0)
            bend += rest_bend[layout_index, c] * w
    bend_scale[env, k] = wp.max(0.0, 1.0 + delta)
    bend_offset[env, k] = wp.vec3(0.5 * bend[0], 0.5 * bend[1], 0.0)
