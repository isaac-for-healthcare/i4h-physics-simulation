# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-quaternion skinning of a vessel surface bound to its centreline tree, for the renderers.

Ported from omniendo xcath/vessel_skinning.py:201-304 (``compute_vertex_dual_quaternions_kernel``,
``smooth_dual_quaternions_kernel``, ``transform_rest_vertices_kernel``) and the ``VesselSkinner.update`` pass
order (:425-470: one dual quaternion per vertex from its bound segment, four one-ring smoothing passes a↔b, the
transform). Edits: the tree's positions are read from the env's row of the snapshot's body view at ``node`` and the
segment frames from the tree's rod system at ``env * S + segment`` (omniendo skinned one world); the normals are
the renderers' own (``kernels.update_normals_tangents``), so the three normal kernels are not ported. The bind
(:143-198) runs once at load in ``loaders/centreline.py``.
"""

from __future__ import annotations

import warp as wp
from i4h_physics.systems.rods.rod_pbd import qconj, qmul

SMOOTHING_PASSES = 4  # vessel_skinning.py:12


class Skinner:
    """Device copies of a body's ``SkinMesh`` and the dual-quaternion buffers; ``place`` writes the skinned vertices."""

    def __init__(self, body, rod, device):
        s, t = body.skin, body.tree
        if s is None or t is None:
            raise ValueError(f"body {body.name!r} has no skin or no tree record")
        self.body, self.rod = body, rod
        self.count = n = len(s.points)
        self.indices = s.indices
        self.edges = wp.array(t.edges[0], dtype=wp.vec2i, device=device)
        self.rest_points = wp.array(s.points, dtype=wp.vec3, device=device)
        self.bound_edge = wp.array(s.bound_edge, dtype=wp.int32, device=device)
        self.bound_t = wp.array(s.bound_t, dtype=wp.float32, device=device)
        self.rest_center = wp.array(s.rest_center, dtype=wp.vec3, device=device)
        self.rest_orientation = wp.array(s.rest_orientation, dtype=wp.quat, device=device)
        self.adj_offsets = wp.array(s.adj_offsets, dtype=wp.int32, device=device)
        self.adj_vertices = wp.array(s.adj_vertices, dtype=wp.int32, device=device)
        self.real = [wp.zeros(n, dtype=wp.quat, device=device) for _ in range(2)]
        self.dual = [wp.zeros(n, dtype=wp.quat, device=device) for _ in range(2)]

    def place(self, snap, env: int, out_pos, device, stream=None) -> None:
        """Skinned vertex positions of ``env`` from ``snap`` (positions and this tree's frames) into ``out_pos``."""
        b, rod, n = self.body, self.rod, self.count
        wp.launch(vertex_dual_quaternions, dim=n,
                  inputs=[rod.scene.view(snap.q, b)[int(env)], snap.frame[rod], int(env) * rod.frame_count, self.edges,
                          self.bound_edge, self.bound_t, self.rest_center, self.rest_orientation],
                  outputs=[self.real[0], self.dual[0]], device=device, stream=stream)
        for k in range(SMOOTHING_PASSES):
            i, o = k % 2, (k + 1) % 2
            wp.launch(smooth_dual_quaternions, dim=n,
                      inputs=[self.real[i], self.dual[i], self.adj_offsets, self.adj_vertices],
                      outputs=[self.real[o], self.dual[o]], device=device, stream=stream)
        wp.launch(transform_rest_vertices, dim=n, inputs=[self.rest_points, self.real[0], self.dual[0]],
                  outputs=[out_pos], device=device, stream=stream)


# Ported from omniendo xcath/vessel_skinning.py:201-231 (compute_vertex_dual_quaternions_kernel). Edits: ``positions`` is
# the env's row of the tree body's view; ``fbase`` offsets the env's segment frames.
@wp.kernel
def vertex_dual_quaternions(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    fbase: wp.int32,
    edges: wp.array(dtype=wp.vec2i),
    bound_edges: wp.array(dtype=wp.int32),
    bound_coordinates: wp.array(dtype=wp.float32),
    rest_centers: wp.array(dtype=wp.vec3),
    rest_orientations: wp.array(dtype=wp.quat),
    real: wp.array(dtype=wp.quat),
    dual: wp.array(dtype=wp.quat),
):
    vertex_index = wp.tid()
    edge_index = bound_edges[vertex_index]
    edge = edges[edge_index]
    t = bound_coordinates[vertex_index]
    center = positions[edge[0]] * (1.0 - t) + positions[edge[1]] * t
    rotation = qmul(orientations[fbase + edge_index], qconj(rest_orientations[vertex_index]))
    rotation_length = wp.sqrt(wp.dot(rotation, rotation))
    if rotation_length <= 1.0e-12:
        rotation = wp.quat_identity()
    else:
        rotation = rotation * (1.0 / rotation_length)
    translation = center - wp.quat_rotate(rotation, rest_centers[vertex_index])
    translation_quaternion = wp.quat(translation[0], translation[1], translation[2], 0.0)
    real[vertex_index] = rotation
    dual[vertex_index] = qmul(translation_quaternion, rotation) * 0.5


# Ported from omniendo xcath/vessel_skinning.py:234-282 (smooth_dual_quaternions_kernel). Edits: none.
@wp.kernel
def smooth_dual_quaternions(
    input_real: wp.array(dtype=wp.quat),
    input_dual: wp.array(dtype=wp.quat),
    adjacency_offsets: wp.array(dtype=wp.int32),
    adjacency_vertices: wp.array(dtype=wp.int32),
    output_real: wp.array(dtype=wp.quat),
    output_dual: wp.array(dtype=wp.quat),
):
    vertex_index = wp.tid()
    reference = input_real[vertex_index]
    neighbor_real = wp.quat(0.0, 0.0, 0.0, 0.0)
    neighbor_dual = wp.quat(0.0, 0.0, 0.0, 0.0)
    begin = adjacency_offsets[vertex_index]
    end = adjacency_offsets[vertex_index + 1]
    for adjacency_index in range(begin, end):
        neighbor_index = adjacency_vertices[adjacency_index]
        candidate_real = input_real[neighbor_index]
        candidate_dual = input_dual[neighbor_index]
        sign = 1.0
        if wp.dot(reference, candidate_real) < 0.0:
            sign = -1.0
        neighbor_real = neighbor_real + candidate_real * sign
        neighbor_dual = neighbor_dual + candidate_dual * sign

    blended_real = reference
    blended_dual = input_dual[vertex_index]
    count = end - begin
    if count > 0:
        inverse_count = 1.0 / float(count)
        blended_real = reference * 0.5 + neighbor_real * (0.5 * inverse_count)
        blended_dual = input_dual[vertex_index] * 0.5 + neighbor_dual * (0.5 * inverse_count)

    length = wp.sqrt(wp.dot(blended_real, blended_real))
    if length <= 1.0e-12:
        output_real[vertex_index] = reference
        output_dual[vertex_index] = input_dual[vertex_index]
        return
    normalized_real = blended_real * (1.0 / length)
    normalized_dual = blended_dual * (1.0 / length)
    # A unit dual quaternion additionally requires real dot dual == 0.
    normalized_dual = normalized_dual - normalized_real * wp.dot(normalized_real, normalized_dual)
    output_real[vertex_index] = normalized_real
    output_dual[vertex_index] = normalized_dual


# Ported from omniendo xcath/vessel_skinning.py:285-304 (transform_rest_vertices_kernel). Edits: none.
@wp.kernel
def transform_rest_vertices(
    rest_vertices: wp.array(dtype=wp.vec3),
    real: wp.array(dtype=wp.quat),
    dual: wp.array(dtype=wp.quat),
    output_positions: wp.array(dtype=wp.vec3),
):
    vertex_index = wp.tid()
    rotation = real[vertex_index]
    translation_quaternion = qmul(dual[vertex_index], qconj(rotation))
    translation = wp.vec3(2.0 * translation_quaternion[0], 2.0 * translation_quaternion[1], 2.0 * translation_quaternion[2])
    output_positions[vertex_index] = wp.quat_rotate(rotation, rest_vertices[vertex_index]) + translation
