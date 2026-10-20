# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rod tube mesh for the renderers: one ring of ``resolution`` vertices per node, placed with the rod's frames.

Topology from omniendo/xpbd_rod/rod_mesher.py:233-258 (``_build_topology``; one ring per node, no
Hermite sub-rings). Placement deviates from omniendo's ``_mesh_one_rod`` (rod_mesher.py:76-173, a
single-thread serial sweep with its own parallel-transported frame): one thread per vertex uses the
node's physics material frame (``d1 = R(q) x``, ``d2 = R(q) y``; local z is the tangent), so the tube
is parallel over ``(P, resolution)`` and shows the twist the solver computes. The PBD backend has
``P - 1`` segment frames, so the last node reuses the last segment's. Normals are the radial
directions. Both renderers call ``place``; the buffers they write into are theirs.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from i4h_isaaclab.appearance import appearance

ROD_COLOR = (0.85, 0.3, 0.2)


# Ported from omniendo/xpbd_rod/rod_mesher.py:233-258 (_build_topology). Edits: the UV array is dropped
# (the tubes are drawn flat-coloured).
def build_topology(num_rings: int, resolution: int) -> np.ndarray:
    """Build the static index array for one tube."""
    num_tris = (num_rings - 1) * resolution * 2 if num_rings > 1 else 0

    indices = np.zeros(num_tris * 3, dtype=np.int32)

    tri = 0
    for r in range(1, num_rings):
        base = r * resolution
        for v in range(resolution):
            cur = base + v
            nxt = base + (v + 1) % resolution
            indices[tri * 3] = cur
            indices[tri * 3 + 1] = cur - resolution
            indices[tri * 3 + 2] = nxt - resolution
            tri += 1
            indices[tri * 3] = nxt - resolution
            indices[tri * 3 + 1] = nxt
            indices[tri * 3 + 2] = cur
            tri += 1

    return indices


class RodDraw:
    """Tube geometry of one rod body: static indices, ``count`` vertices refilled per frame from the drawn env."""

    def __init__(self, body, rod, resolution: int = 12):
        self.body, self.rod = body, rod
        self.name = body.name
        self.resolution = int(resolution)
        self.count = body.count * self.resolution
        self.indices = build_topology(body.count, self.resolution)
        part = next(iter(body.parts.values()))
        self.color = tuple(float(c) for c in (appearance(part) or {}).get("color", ROD_COLOR))
        self.radius = float(body.rest.radius[0, 0])  # every node shares the rod's radius

    def place(self, snap, env: int, out_pos, out_normals, device, stream=None) -> None:
        """Tube vertices and normals of ``env`` from ``snap`` (a ``Snapshot``: positions and this rod's frames)."""
        b = self.body
        wp.launch(place_tube, dim=(b.count, self.resolution),
                  inputs=[self.rod.scene.view(snap.q, b)[int(env)], snap.frame[self.rod], self.rod.frame_count, int(env), self.radius,
                          self.resolution],
                  outputs=[out_pos, out_normals], device=device, stream=stream)


@wp.kernel
def place_tube(
    q: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.quat),
    F: wp.int32,
    env: wp.int32,
    radius: wp.float32,
    resolution: wp.int32,
    out_pos: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Vertex ``p * resolution + k``: node ``p`` plus ``radius`` along the ring direction in its frame's x-y plane."""
    p, k = wp.tid()
    f = frame[env * F + wp.min(p, F - 1)]
    theta = 2.0 * wp.pi * wp.float32(k) / wp.float32(resolution)
    d1 = wp.quat_rotate(f, wp.vec3(1.0, 0.0, 0.0))
    d2 = wp.quat_rotate(f, wp.vec3(0.0, 1.0, 0.0))
    n = d1 * wp.cos(theta) + d2 * wp.sin(theta)
    v = p * resolution + k
    out_pos[v] = q[p] + n * radius  # q: the env's row of the body's view
    out_normals[v] = n
