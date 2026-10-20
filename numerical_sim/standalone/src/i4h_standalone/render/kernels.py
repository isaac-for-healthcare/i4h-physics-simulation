# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-frame vertex normals and tangents for a deformable render mesh.

Ported from omnisurg/rendering/slang.py:860-1011 (kernels) and 4444-4544 (drivers), fused
into two launches: one pass over triangles accumulates face normals, tangents and
bitangents with atomics; one pass over vertices normalises, orthogonalises the tangent,
sets the handedness and clears the accumulators, which are therefore zero between frames
by construction (the same convention as the systems' delta buffers). No hit counters: a
zero accumulator selects the fallback. Accumulators and results are ``outputs``.
"""

from __future__ import annotations

import warp as wp


class Scratch:
    """Accumulators for one mesh of ``count`` vertices; zero between frames."""

    def __init__(self, count: int, device):
        self.normal = wp.zeros(count, dtype=wp.vec3f, device=device)
        self.tangent = wp.zeros(count, dtype=wp.vec3f, device=device)
        self.bitangent = wp.zeros(count, dtype=wp.vec3f, device=device)


def update_normals_tangents(positions, indices, uvs, scratch: Scratch, normals, tangents, device, stream=None) -> None:
    """``indices`` is the flat ``(3F,)`` int32 triangle list; ``tangents`` is ``(n,)`` vec4 (xyz + handedness).
    ``stream``: the renderer's stream (None: the current one)."""
    wp.launch(accumulate_frames, dim=indices.shape[0] // 3, inputs=[indices, positions, uvs],
              outputs=[scratch.normal, scratch.tangent, scratch.bitangent], device=device, stream=stream)
    wp.launch(finalize_frames, dim=positions.shape[0],
              outputs=[scratch.normal, scratch.tangent, scratch.bitangent, normals, tangents], device=device, stream=stream)


@wp.kernel
def accumulate_frames(
    indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    uvs: wp.array(dtype=wp.vec2f),
    normal_accum: wp.array(dtype=wp.vec3f),
    tangent_accum: wp.array(dtype=wp.vec3f),
    bitangent_accum: wp.array(dtype=wp.vec3f),
):
    tri = wp.tid()
    base = tri * 3
    i = indices[base + 0]
    j = indices[base + 1]
    k = indices[base + 2]
    p0 = positions[i]
    p1 = positions[j]
    p2 = positions[k]
    edge1 = p1 - p0
    edge2 = p2 - p0

    n = wp.cross(edge1, edge2)  # area-weighted
    if wp.length(n) > 1.0e-12:
        wp.atomic_add(normal_accum, i, n)
        wp.atomic_add(normal_accum, j, n)
        wp.atomic_add(normal_accum, k, n)

    uv0 = uvs[i]
    duv1 = uvs[j] - uv0
    duv2 = uvs[k] - uv0
    det = duv1[0] * duv2[1] - duv1[1] * duv2[0]
    if det <= 1.0e-12 and det >= -1.0e-12:
        return
    inv_det = 1.0 / det
    tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * inv_det
    bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) * inv_det
    if wp.length(tangent) <= 1.0e-12 or wp.length(bitangent) <= 1.0e-12:
        return
    wp.atomic_add(tangent_accum, i, tangent)
    wp.atomic_add(bitangent_accum, i, bitangent)
    wp.atomic_add(tangent_accum, j, tangent)
    wp.atomic_add(bitangent_accum, j, bitangent)
    wp.atomic_add(tangent_accum, k, tangent)
    wp.atomic_add(bitangent_accum, k, bitangent)


@wp.func
def fallback_tangent(normal: wp.vec3f):
    axis = wp.vec3f(0.0, 1.0, 0.0)
    if normal[1] > 0.95 or normal[1] < -0.95:
        axis = wp.vec3f(1.0, 0.0, 0.0)
    tangent = wp.cross(axis, normal)
    if wp.length(tangent) <= 1.0e-12:
        return wp.vec3f(1.0, 0.0, 0.0)
    return wp.normalize(tangent)


@wp.kernel
def finalize_frames(
    normal_accum: wp.array(dtype=wp.vec3f),
    tangent_accum: wp.array(dtype=wp.vec3f),
    bitangent_accum: wp.array(dtype=wp.vec3f),
    normals: wp.array(dtype=wp.vec3f),
    tangents: wp.array(dtype=wp.vec4f),
):
    tid = wp.tid()
    n = normal_accum[tid]
    if wp.length(n) > 1.0e-12:
        n = wp.normalize(n)
    else:
        n = wp.vec3f(0.0, 1.0, 0.0)

    t = tangent_accum[tid]
    if wp.length(t) > 1.0e-12:
        t = t - n * wp.dot(n, t)
        if wp.length(t) > 1.0e-12:
            t = wp.normalize(t)
        else:
            t = fallback_tangent(n)
    else:
        t = fallback_tangent(n)

    b = bitangent_accum[tid]
    handedness = 1.0
    if wp.length(b) > 1.0e-12 and wp.dot(wp.cross(n, t), b) < 0.0:
        handedness = -1.0

    normals[tid] = n
    tangents[tid] = wp.vec4f(t[0], t[1], t[2], handedness)
    normal_accum[tid] = wp.vec3f(0.0, 0.0, 0.0)
    tangent_accum[tid] = wp.vec3f(0.0, 0.0, 0.0)
    bitangent_accum[tid] = wp.vec3f(0.0, 0.0, 0.0)
