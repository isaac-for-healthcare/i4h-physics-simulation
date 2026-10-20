# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stretch breaking: over-stretched tets and edges of deletable parts break; damage and blood are painted.

Ported from omnisurg/tet/kernels/tet.py: measure_tet_max_edge_stretch 822-853,
map_tet_stretch_to_tissue_visual 856-903, disable_stretched_target_springs 906-931 and
break_stretched_target_tets 934-961; omnisurg/tet/solver/breaking.py:64-259 ran them in
substep_end (disabled in the shipped chole case, which kept threshold 6.0). Edits: (env, tet)
and (env, edge) prologues; rest lengths come from the edge list through ``tet_edges``
(omnisurg kept a (T, 6) copy); the per-particle stretch maximum is an ``atomic_max`` into a
self-clearing accumulator that collects the step's substeps and is painted once at step end
(omnisurg walked a particle -> tet list every substep); the target region is ``parts`` (omnisurg
``deletion.stretch`` per part, tet/bundle/topology.py:296-320; chole: fat) within the ``deletable``
parts, every deletable part when empty; removal goes through ``surface.deactivate_tet``; edges break
by clearing ``edge_active`` (omnisurg's spring_active). Damage (r) and blood (b) go into
``surface.visual``. ``max_stretch`` (E,) is the step's largest target stretch (the tuning observable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.systems.base import System
from i4h_physics.systems.mechanics.surface import Surface, deactivate_tet


@dataclass(eq=False)
class StretchBreaking(System):
    state_products = ("max_stretch",)  # per-env products a host observes (System.state_products)
    surface: Surface
    threshold: float = 6.0  # live, per env: length / rest above which a tet or edge breaks
    break_edges: bool = True  # structural: also clear over-stretched edges of the target parts
    parts: tuple = ()  # structural: the deletable parts that break (omnisurg deletion.stretch; empty = every deletable part)

    refs = ("surface",)
    structural = System.structural + ("break_edges", "parts")

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.surface, Surface):
            raise TypeError("stretch_breaking.surface must name a surface system")
        self.body = b = self.surface.body
        E, dev = scene.envs, scene.device
        m = b.tet
        parts = (self.parts,) if isinstance(self.parts, str) else tuple(self.parts)
        target = np.asarray(m.deletable, np.int32).copy()
        if parts:
            unknown = [name for name in parts if name not in b.parts]
            if unknown:
                raise ValueError(f"stretch_breaking.parts: unknown part(s) {unknown}; known: {list(b.parts)}")
            frozen = [name for name in parts if not m.deletable[b.parts[name].index]]
            if frozen:
                raise ValueError(f"stretch_breaking.parts: {frozen} are not deletable (no cut-face slots), so they cannot break")
            chosen = np.zeros_like(target)
            for name in parts:
                chosen[b.parts[name].index] = 1
            target &= chosen
        self.parts = parts
        self.freeze("parts")
        self.target = wp.array(target, dtype=wp.int32, device=dev)  # (P,) parts that break
        self._live("threshold")
        self.stretch = wp.zeros(E * b.count, dtype=wp.float32, device=dev)  # body-local per particle; zero between steps
        self.max_stretch = wp.zeros(E, dtype=wp.float32, device=dev)  # product: the step's largest target stretch


    def step_start(self, scene, dt):
        self.max_stretch.zero_()

    def substep_end(self, scene, dt):
        sf, b = self.surface, self.body
        E, dev = scene.envs, scene.device
        T, S = sf.tets.shape[1], sf.edges.shape[1]
        if T == 0:
            return
        wp.launch(measure_and_break, dim=(E, T),
                  inputs=[scene.field("q", b), sf.variant, sf.tets, sf.tet_part, self.target, sf.tet_edges, sf.edges, sf.edge_rest, self.threshold,
                          b.count],
                  outputs=[self.stretch, sf.active, sf.edge_active, sf.support, sf.edge_support, scene.field("flags", b), sf.deleted,
                           sf.dirty],
                  device=dev)
        if self.break_edges and S:
            wp.launch(break_edges, dim=(E, S),
                      inputs=[scene.field("q", b), sf.variant, sf.edges, sf.edge_part, self.target, sf.edge_rest, self.threshold],
                      outputs=[sf.edge_active], device=dev)

    def step_end(self, scene, dt):
        wp.launch(finish_stretch, dim=(scene.envs, self.body.count), inputs=[self.body.count],
                  outputs=[self.stretch, self.max_stretch], device=scene.device)

    def reset(self, scene, mask):
        self.clear(mask, (self.stretch, wp.float32(0.0)))
        self.clear(mask, (self.max_stretch, wp.float32(0.0)))


# tet.py:822-853 and 934-961 (measure_tet_max_edge_stretch, break_stretched_target_tets) fused: one
# pass per tet measures, pushes the maximum onto the corners and breaks.
@wp.kernel
def measure_and_break(
    q: wp.array2d(dtype=wp.vec3),
    variant: wp.array(dtype=wp.int32),
    tets: wp.array2d(dtype=wp.vec4i),
    tet_part: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
    tet_edges: wp.array3d(dtype=wp.int32),
    edges: wp.array2d(dtype=wp.vec2i),
    edge_rest: wp.array2d(dtype=wp.float32),
    threshold: wp.array(dtype=wp.float32),
    count: wp.int32,
    stretch_p: wp.array(dtype=wp.float32),
    active: wp.array2d(dtype=wp.int32),
    edge_active: wp.array2d(dtype=wp.int32),
    support: wp.array(dtype=wp.int32),
    edge_support: wp.array2d(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    deleted: wp.array(dtype=wp.int32),
    dirty: wp.array(dtype=wp.int32),
):
    env, t = wp.tid()
    if active[env, t] == 0 or target[tet_part[t]] == 0:
        return
    v = variant[env]
    ids = tets[v, t]
    stretch = wp.float32(0.0)
    for e in range(6):
        s = tet_edges[v, t, e]
        if s < 0:
            continue
        rest = edge_rest[v, s]
        if rest <= 1.0e-8:
            continue
        ab = edges[v, s]  # the tet's corner pair: tet_edges matched sorted corner pairs to the edge list (tet_bundle.py)
        stretch = wp.max(stretch, wp.length(q[env, ab[0]] - q[env, ab[1]]) / rest)
    for c in range(4):
        wp.atomic_max(stretch_p, env * count + ids[c], stretch)
    if stretch > threshold[env]:
        deactivate_tet(env, t, v, count, tets, tet_edges, active, edge_active, support, edge_support, flags, deleted, dirty)


# tet.py:906-931 (disable_stretched_target_springs): one-way, never re-enabled until reset.
@wp.kernel
def break_edges(
    q: wp.array2d(dtype=wp.vec3),
    variant: wp.array(dtype=wp.int32),
    edges: wp.array2d(dtype=wp.vec2i),
    edge_part: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
    edge_rest: wp.array2d(dtype=wp.float32),
    threshold: wp.array(dtype=wp.float32),
    edge_active: wp.array2d(dtype=wp.int32),
):
    env, s = wp.tid()
    if edge_active[env, s] == 0 or target[edge_part[s]] == 0:
        return
    v = variant[env]
    rest = edge_rest[v, s]
    if rest <= 1.0e-8:
        return
    e = edges[v, s]
    if wp.length(q[env, e[0]] - q[env, e[1]]) / rest > threshold[env]:
        edge_active[env, s] = wp.int32(0)


@wp.kernel
def finish_stretch(count: int, stretch: wp.array(dtype=float), maximum: wp.array(dtype=float)):
    e, p = wp.tid()
    i = e * count + p
    value = stretch[i]
    stretch[i] = 0.0
    if value > 0.0:
        wp.atomic_max(maximum, e, value)
