# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clips drawn at clipped centreline nodes: one torus per clip slot, its axis along the vessel, refilled per frame.

Ours: omnisurg had no clip kind and drew no clips (its clip bookkeeping was never called). The torus sits at the
node's live position (the snapshot's centreline positions), oriented by the neighbouring nodes of the same line,
with the major radius the median rest ring radius of the clippable lines and a tube of 0.3 of that; an inactive
slot collapses to its node (zero area), like an inactive collision sphere in the overlay. Both renderers call
``ClipDraw.place``; the buffers they write into are theirs.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from i4h_standalone.render.instruments import sphere_mesh_indices
from i4h_surgical.systems.instruments.instrument_control import EPS, orthogonal_axis, unit_or_zero

CLIP_COLOR = (0.86, 0.87, 0.92)  # polished titanium, brighter than the tool greys


def torus(segments: int = 12, sides: int = 8):
    """Unit torus template: ``centre (V, 3)`` on the unit ring in the x-y plane, ``tube (V, 3)`` the unit tube direction
    (also the normal), ``faces (F, 3)`` with the same winding rule as ``uv_sphere``."""
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    centre, tube = [], []
    for t in theta:
        for p in phi:
            centre.append((np.cos(t), np.sin(t), 0.0))
            tube.append((np.cos(t) * np.cos(p), np.sin(t) * np.cos(p), np.sin(p)))
    faces = []
    for i in range(segments):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = ((i + 1) % segments) * sides + j
            d = ((i + 1) % segments) * sides + (j + 1) % sides
            faces += [(a, c, b), (b, c, d)]
    return np.asarray(centre, np.float32), np.asarray(tube, np.float32), np.asarray(faces, np.int32)


class ClipDraw:
    """Clip geometry of one clipping system: ``G`` tori refilled per frame from the snapshot's slots and node positions."""

    def __init__(self, clipping, device):
        cl = clipping.centreline
        body, c = cl.body, cl.body.centrelines
        self.clipping, self.centreline = clipping, cl
        self.name = f"clips-{clipping.instrument}"
        centre, tube, faces = torus()
        self.centre = wp.array(centre, dtype=wp.vec3, device=device)
        self.tube = wp.array(tube, dtype=wp.vec3, device=device)
        self.G = int(clipping.node.shape[1])
        self.count = self.G * len(centre)
        self.indices = sphere_mesh_indices(self.G, faces, len(centre))
        self.line = wp.array(np.asarray(c.line, np.int32), dtype=wp.int32, device=device)  # the renderer's own copy of the host table
        eligible = np.isin(c.line, [c.names.index(name) for name in clipping.line])
        q0 = body.rest.q0[int(np.flatnonzero(body.variant_of_env == 0)[0]) if (body.variant_of_env == 0).any() else 0]
        radii = []
        for k in np.flatnonzero(eligible):
            ids = c.clamp[0, c.clamp_start[0, k] : c.clamp_start[0, k] + c.clamp_count[0, k]]
            if len(ids) >= 2:
                radii.append(float(np.linalg.norm(q0[ids] - q0[ids].mean(axis=0), axis=1).mean()))
        self.major = float(np.median(radii)) if radii else 0.02
        self.minor = 0.3 * self.major

    def place(self, snap, env: int, out_pos, out_normals, device, stream=None) -> None:
        node, active = snap.clip[self.clipping]
        wp.launch(place_clip_mesh, dim=self.count,
                  inputs=[self.centre, self.tube, node, active, snap.centreline[self.centreline], self.line, int(env), self.major, self.minor],
                  outputs=[out_pos, out_normals], device=device, stream=stream)


@wp.kernel
def place_clip_mesh(
    centre: wp.array(dtype=wp.vec3),
    tube: wp.array(dtype=wp.vec3),
    node: wp.array2d(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    positions: wp.array2d(dtype=wp.vec3),
    line: wp.array(dtype=wp.int32),
    env: wp.int32,
    major: wp.float32,
    minor: wp.float32,
    out: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Vertex ``g * V + v``: slot ``g``'s torus around node ``node[env, g]`` with its axis along the vessel (the neighbours on
    the same line); an inactive slot puts every vertex at the node."""
    tid = wp.tid()
    V = centre.shape[0]
    g = tid // V
    v = tid - g * V
    n = node[env, g]
    scale = float(1.0)
    if n < 0 or active[env, g] == 0:
        n = wp.max(n, 0)
        scale = 0.0
    K = positions.shape[1]
    a = n
    b = n
    if n > 0 and line[n - 1] == line[n]:
        a = n - 1
    if n + 1 < K and line[n + 1] == line[n]:
        b = n + 1
    t = unit_or_zero(positions[env, b] - positions[env, a])
    if wp.length(t) <= EPS:
        t = wp.vec3(0.0, 0.0, 1.0)
    x = orthogonal_axis(t)
    y = wp.cross(t, x)
    c = centre[v]
    d = tube[v]
    normal = x * d[0] + y * d[1] + t * d[2]
    out[tid] = positions[env, n] + ((x * c[0] + y * c[1]) * major + normal * minor) * scale
    out_normals[tid] = normal
