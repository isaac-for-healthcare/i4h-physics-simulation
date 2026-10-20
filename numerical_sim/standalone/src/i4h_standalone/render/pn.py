# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PN triangle subdivision of a body's drawn surface: a compute pre-pass expanding every triangle into a lattice of
``(s+1)(s+2)/2`` curved vertices and ``s^2`` sub-triangles (``render.pn: {segments: 2, curvature: 1.0}``).

Port of omnisurg/rendering/pn.py: the lattice ``build_subdivision_records`` 66-101, ``_safe_normalize`` 104-109,
``_pn_edge_position`` 113-149, ``_pn_edge_normal`` 152-181, ``_pn_position`` 184-229, ``_pn_normal`` 232-268, the
evaluation ``_evaluate_pn_geometry`` 349-388 and the interpolations ``_interpolate_pn_uvs`` 391-405 and
``_interpolate_pn_tissue_layers`` 408-424 fused into one launch over the lattice vertices (``tri = tid // R``, the
barycentrics indexed ``tid % R`` instead of a tiled copy). Base normals come from the renderer's
``update_normals_tangents`` (the same area-weighted average as pn.py 271-317). Edit: a tangent per lattice vertex (the
barycentric blend of the corner tangents made perpendicular to the PN normal; omnisurg's PN meshes drew flat colours
and had none). An inactive surface triangle is ``(i, i, i)`` and expands to zero-area sub-triangles by itself.
"""

from __future__ import annotations

import numpy as np
import warp as wp

DEFAULTS = dict(segments=2, curvature=1.0)  # pn.py:14-18


def lattice(segments: int):
    """Barycentrics ``(R, 3)`` of one triangle's lattice and its ``(S*3,)`` local sub-triangle indices (pn.py:66-91)."""
    bary, lookup = [], {}
    for i in range(segments + 1):
        for j in range(segments + 1 - i):
            u, v = i / segments, j / segments
            lookup[(i, j)] = len(bary)
            bary.append((1.0 - u - v, u, v))
    local = []
    for i in range(segments):
        for j in range(segments - i):
            v0, v1, v2 = lookup[(i, j)], lookup[(i + 1, j)], lookup[(i, j + 1)]
            local += [v0, v1, v2]
            if j < segments - i - 1:
                local += [v1, lookup[(i + 1, j + 1)], v2]
    return np.asarray(bary, np.float32), np.asarray(local, np.int32)


class PnDraw:
    """Expanded streams of one BodyDraw: ``F*R`` lattice vertices, a static index buffer of ``F*S`` sub-triangles."""

    def __init__(self, renderer, name: str, F: int, segments: int, curvature: float):
        r, spy, dev = renderer, renderer.spy, renderer.scene.device
        self.segments, self.curvature = int(segments), float(curvature)
        if not 2 <= self.segments <= 4:
            raise ValueError("render.pn.segments must be in [2, 4]")
        if not 0.0 <= self.curvature <= 1.0:
            raise ValueError("render.pn.curvature must be in [0, 1]")
        bary, local = lattice(self.segments)
        self.R, self.S = len(bary), len(local) // 3
        self.F = int(F)
        n = self.F * self.R
        self.pos = r.shared(n, wp.vec3f, 12, f"{name}-pn-positions")
        self.normals = r.shared(n, wp.vec3f, 12, f"{name}-pn-normals")
        self.tangents = r.shared(n, wp.vec4f, 16, f"{name}-pn-tangents")
        self.uvs = r.shared(n, wp.vec2f, 8, f"{name}-pn-uvs")
        self.colors = r.shared(n, wp.vec4f, 16, f"{name}-pn-colors")
        indices = (np.arange(self.F, dtype=np.int64)[:, None] * self.R + local[None, :]).reshape(-1).astype(np.uint32)
        self.index_buffer = r.device.create_buffer(usage=spy.BufferUsage.index_buffer, data=indices, label=f"{name}-pn-indices")
        self.expanded = False  # the uv and colour lattices of a body without a surface are written once
        self.bary = wp.array(bary, dtype=wp.vec3f, device=dev)
        self.streams = [self.pos.buffer, self.normals.buffer, self.tangents.buffer, self.uvs.buffer, self.colors.buffer]

    def expand(self, positions, normals, tangents, uvs, colors, indices, device, stream=None, static: bool = False):
        """The curved lattice from the base streams; ``indices`` is the flat ``(3F,)`` live triangle list. With ``static``
        (no surface system: fixed triangles, uvs and colours) the uv and colour lattices are written on the first call only."""
        wp.launch(pn_expand, dim=self.F * self.R,
                  inputs=[indices, positions, normals, tangents, uvs, colors, self.bary, self.R, self.curvature,
                          int(not static or not self.expanded)],
                  outputs=[self.pos.array, self.normals.array, self.tangents.array, self.uvs.array, self.colors.array],
                  device=device, stream=stream)
        self.expanded = True

    def close(self):
        for shared in (self.pos, self.normals, self.tangents, self.uvs, self.colors):
            shared.close()
        self.index_buffer = None


# pn.py:104-109
@wp.func
def _safe_normalize(v: wp.vec3f) -> wp.vec3f:
    length = wp.length(v)
    if length > 1.0e-12:
        return v / length
    return wp.vec3f(0.0, 1.0, 0.0)


# pn.py:113-149
@wp.func
def _pn_edge_position(pa: wp.vec3f, pb: wp.vec3f, na: wp.vec3f, nb: wp.vec3f, wa: float, wb: float, ida: int, idb: int,
                      curvature: float) -> wp.vec3f:
    p0 = pa
    p1 = pb
    n0 = na
    n1 = nb
    t = wb
    if idb < ida:
        p0 = pb
        p1 = pa
        n0 = nb
        n1 = na
        t = wa
    one_minus_t = 1.0 - t
    c0 = p0
    c3 = p1
    c1 = (p0 * 2.0 + p1 - n0 * wp.dot(p1 - p0, n0)) / 3.0
    c2 = (p1 * 2.0 + p0 - n1 * wp.dot(p0 - p1, n1)) / 3.0
    pn = (c0 * one_minus_t * one_minus_t * one_minus_t + c1 * 3.0 * one_minus_t * one_minus_t * t
          + c2 * 3.0 * one_minus_t * t * t + c3 * t * t * t)
    linear = p0 * one_minus_t + p1 * t
    return linear * (1.0 - curvature) + pn * curvature


# pn.py:152-181
@wp.func
def _pn_edge_normal(pa: wp.vec3f, pb: wp.vec3f, na: wp.vec3f, nb: wp.vec3f, wa: float, wb: float, ida: int, idb: int) -> wp.vec3f:
    p0 = pa
    p1 = pb
    n0 = na
    n1 = nb
    t = wb
    if idb < ida:
        p0 = pb
        p1 = pa
        n0 = nb
        n1 = na
        t = wa
    edge = p1 - p0
    edge_length_sq = wp.dot(edge, edge)
    mid = n0 + n1
    if edge_length_sq > 1.0e-12:
        mid = mid - edge * (2.0 * wp.dot(edge, n0 + n1) / edge_length_sq)
    one_minus_t = 1.0 - t
    normal = n0 * one_minus_t * one_minus_t + mid * 2.0 * one_minus_t * t + n1 * t * t
    return _safe_normalize(normal)


# pn.py:184-229
@wp.func
def _pn_position(p0: wp.vec3f, p1: wp.vec3f, p2: wp.vec3f, n0: wp.vec3f, n1: wp.vec3f, n2: wp.vec3f, bary: wp.vec3f,
                 curvature: float) -> wp.vec3f:
    a = bary[0]
    b = bary[1]
    c = bary[2]
    b300 = p0
    b030 = p1
    b003 = p2
    b210 = (p0 * 2.0 + p1 - n0 * wp.dot(p1 - p0, n0)) / 3.0
    b120 = (p1 * 2.0 + p0 - n1 * wp.dot(p0 - p1, n1)) / 3.0
    b021 = (p1 * 2.0 + p2 - n1 * wp.dot(p2 - p1, n1)) / 3.0
    b012 = (p2 * 2.0 + p1 - n2 * wp.dot(p1 - p2, n2)) / 3.0
    b102 = (p2 * 2.0 + p0 - n2 * wp.dot(p0 - p2, n2)) / 3.0
    b201 = (p0 * 2.0 + p2 - n0 * wp.dot(p2 - p0, n0)) / 3.0
    edge_average = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0
    vertex_average = (p0 + p1 + p2) / 3.0
    b111 = edge_average + (edge_average - vertex_average) * 0.5
    pn = (b300 * a * a * a + b030 * b * b * b + b003 * c * c * c + b210 * 3.0 * a * a * b + b120 * 3.0 * a * b * b
          + b021 * 3.0 * b * b * c + b012 * 3.0 * b * c * c + b102 * 3.0 * a * c * c + b201 * 3.0 * a * a * c + b111 * 6.0 * a * b * c)
    linear = p0 * a + p1 * b + p2 * c
    return linear * (1.0 - curvature) + pn * curvature


# pn.py:232-268
@wp.func
def _pn_normal(p0: wp.vec3f, p1: wp.vec3f, p2: wp.vec3f, n0: wp.vec3f, n1: wp.vec3f, n2: wp.vec3f, bary: wp.vec3f) -> wp.vec3f:
    a = bary[0]
    b = bary[1]
    c = bary[2]
    e01 = p1 - p0
    e12 = p2 - p1
    e20 = p0 - p2
    n110 = n0 + n1
    n011 = n1 + n2
    n101 = n2 + n0
    e01_len = wp.dot(e01, e01)
    e12_len = wp.dot(e12, e12)
    e20_len = wp.dot(e20, e20)
    if e01_len > 1.0e-12:
        n110 = n110 - e01 * (2.0 * wp.dot(e01, n0 + n1) / e01_len)
    if e12_len > 1.0e-12:
        n011 = n011 - e12 * (2.0 * wp.dot(e12, n1 + n2) / e12_len)
    if e20_len > 1.0e-12:
        n101 = n101 - e20 * (2.0 * wp.dot(e20, n2 + n0) / e20_len)
    normal = n0 * a * a + n1 * b * b + n2 * c * c + n110 * 2.0 * a * b + n011 * 2.0 * b * c + n101 * 2.0 * a * c
    return _safe_normalize(normal)


# pn.py:349-388 (_evaluate_pn_geometry) with 391-405 (uvs) and 408-424 (tissue layers) folded in; the barycentrics
# are indexed ``tid % R`` (omnisurg tiled them per triangle). Edit: the tangent (see the module docstring).
@wp.kernel
def pn_expand(
    tris: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    base_normals: wp.array(dtype=wp.vec3f),
    base_tangents: wp.array(dtype=wp.vec4f),
    uvs: wp.array(dtype=wp.vec2f),
    colors: wp.array(dtype=wp.vec4f),
    barycentrics: wp.array(dtype=wp.vec3f),
    R: int,
    curvature: float,
    write_attributes: int,
    out_pos: wp.array(dtype=wp.vec3f),
    out_normals: wp.array(dtype=wp.vec3f),
    out_tangents: wp.array(dtype=wp.vec4f),
    out_uvs: wp.array(dtype=wp.vec2f),
    out_colors: wp.array(dtype=wp.vec4f),
):
    tid = wp.tid()
    tri = tid / R
    base = tri * 3
    i0 = tris[base + 0]
    i1 = tris[base + 1]
    i2 = tris[base + 2]
    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    n0 = base_normals[i0]
    n1 = base_normals[i1]
    n2 = base_normals[i2]
    bary = barycentrics[tid - tri * R]
    if bary[0] == 0.0:
        out_pos[tid] = _pn_edge_position(p1, p2, n1, n2, bary[1], bary[2], i1, i2, curvature)
        normal = _pn_edge_normal(p1, p2, n1, n2, bary[1], bary[2], i1, i2)
    elif bary[1] == 0.0:
        out_pos[tid] = _pn_edge_position(p0, p2, n0, n2, bary[0], bary[2], i0, i2, curvature)
        normal = _pn_edge_normal(p0, p2, n0, n2, bary[0], bary[2], i0, i2)
    elif bary[2] == 0.0:
        out_pos[tid] = _pn_edge_position(p0, p1, n0, n1, bary[0], bary[1], i0, i1, curvature)
        normal = _pn_edge_normal(p0, p1, n0, n1, bary[0], bary[1], i0, i1)
    else:
        out_pos[tid] = _pn_position(p0, p1, p2, n0, n1, n2, bary, curvature)
        normal = _pn_normal(p0, p1, p2, n0, n1, n2, bary)
    out_normals[tid] = normal
    if write_attributes != 0:
        out_uvs[tid] = uvs[i0] * bary[0] + uvs[i1] * bary[1] + uvs[i2] * bary[2]
        out_colors[tid] = colors[i0] * bary[0] + colors[i1] * bary[1] + colors[i2] * bary[2]
    t0 = base_tangents[i0]
    t1 = base_tangents[i1]
    t2 = base_tangents[i2]
    t = wp.vec3f(t0[0], t0[1], t0[2]) * bary[0] + wp.vec3f(t1[0], t1[1], t1[2]) * bary[1] + wp.vec3f(t2[0], t2[1], t2[2]) * bary[2]
    t = t - normal * wp.dot(t, normal)
    if wp.length(t) <= 1.0e-12:
        t = wp.cross(normal, wp.vec3f(0.0, 1.0, 0.0))
        if wp.length(t) <= 1.0e-12:
            t = wp.cross(normal, wp.vec3f(1.0, 0.0, 0.0))
    t = _safe_normalize(t)
    w = t0[3]
    if bary[1] > bary[0] and bary[1] >= bary[2]:
        w = t1[3]
    elif bary[2] > bary[0] and bary[2] > bary[1]:
        w = t2[3]
    out_tangents[tid] = wp.vec4f(t[0], t[1], t[2], w)
