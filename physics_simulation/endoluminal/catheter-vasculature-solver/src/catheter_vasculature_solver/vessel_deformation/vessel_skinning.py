# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers / OmniEndo contributors
# SPDX-License-Identifier: Apache-2.0
#
# Vendored/adapted from the OmniEndo upstream vessel skinning module. The debug-colormap
# path (CenterlineDebugColorer) is app UI and intentionally left upstream.

"""Topology-smoothed GPU skinning driven by a PBD Cosserat centerline.

Turns the deforming centerline into a render/collision surface: each surface
vertex binds to the nearest centerline *edge*, and the edge's rigid motion is
applied as a dual quaternion. Dual quaternions are then smoothed across mesh
adjacency so junctions and bends do not crease.
"""

from __future__ import annotations

import numpy as np
import warp as wp

SKINNING_SMOOTHING_PASSES = 4


@wp.func
def _quat_conj(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])


@wp.func
def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    av = wp.vec3(a[0], a[1], a[2])
    bv = wp.vec3(b[0], b[1], b[2])
    v = av * b[3] + bv * a[3] + wp.cross(av, bv)
    return wp.quat(v[0], v[1], v[2], a[3] * b[3] - wp.dot(av, bv))


@wp.kernel
def bind_vertices_to_edges_kernel(
    vertices: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=wp.vec2i),
    edge_count: int,
    bound_edges: wp.array(dtype=wp.int32),
    bound_coordinates: wp.array(dtype=wp.float32),
):
    """Bind to the nearest source-space edge; strict comparison breaks ties by index."""
    vertex_index = wp.tid()
    point = vertices[vertex_index]
    best_edge = int(0)
    best_t = float(0.0)
    best_distance_sq = float(3.402823466e38)
    for edge_index in range(edge_count):
        edge = edges[edge_index]
        a = positions[edge[0]]
        delta = positions[edge[1]] - a
        denominator = wp.dot(delta, delta)
        t = float(0.0)
        if denominator > 1.0e-20:
            t = wp.clamp(wp.dot(point - a, delta) / denominator, 0.0, 1.0)
        offset = point - (a + delta * t)
        distance_sq = wp.dot(offset, offset)
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_edge = edge_index
            best_t = t
    bound_edges[vertex_index] = best_edge
    bound_coordinates[vertex_index] = best_t


@wp.kernel
def capture_rest_transforms_kernel(
    vertices: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    edges: wp.array(dtype=wp.vec2i),
    bound_edges: wp.array(dtype=wp.int32),
    bound_coordinates: wp.array(dtype=wp.float32),
    rest_vertices: wp.array(dtype=wp.vec3),
    rest_centers: wp.array(dtype=wp.vec3),
    rest_orientations: wp.array(dtype=wp.quat),
):
    vertex_index = wp.tid()
    edge_index = bound_edges[vertex_index]
    edge = edges[edge_index]
    t = bound_coordinates[vertex_index]
    rest_vertices[vertex_index] = vertices[vertex_index]
    rest_centers[vertex_index] = positions[edge[0]] * (1.0 - t) + positions[edge[1]] * t
    # Cosserat orientations are segment quantities. Sampling the bound segment
    # directly is essential for cyclic graphs: every authored edge can then
    # drive the surface vertices bound to it.
    rest_orientations[vertex_index] = orientations[edge_index]


@wp.kernel
def compute_vertex_dual_quaternions_kernel(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
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
    rotation = _quat_mul(orientations[edge_index], _quat_conj(rest_orientations[vertex_index]))
    rotation_length = wp.sqrt(wp.dot(rotation, rotation))
    if rotation_length <= 1.0e-12:
        rotation = wp.quat_identity()
    else:
        rotation = rotation * (1.0 / rotation_length)
    translation = center - wp.quat_rotate(rotation, rest_centers[vertex_index])
    translation_quaternion = wp.quat(translation[0], translation[1], translation[2], 0.0)
    real[vertex_index] = rotation
    dual[vertex_index] = _quat_mul(translation_quaternion, rotation) * 0.5


@wp.kernel
def smooth_dual_quaternions_kernel(
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
        # Quaternion double cover: align neighbours before averaging.
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


@wp.kernel
def transform_rest_vertices_kernel(
    rest_vertices: wp.array(dtype=wp.vec3),
    real: wp.array(dtype=wp.quat),
    dual: wp.array(dtype=wp.quat),
    output_positions: wp.array(dtype=wp.vec3),
):
    vertex_index = wp.tid()
    rotation = real[vertex_index]
    translation_quaternion = _quat_mul(dual[vertex_index], _quat_conj(rotation))
    translation = wp.vec3(
        2.0 * translation_quaternion[0],
        2.0 * translation_quaternion[1],
        2.0 * translation_quaternion[2],
    )
    output_positions[vertex_index] = (
        wp.quat_rotate(rotation, rest_vertices[vertex_index]) + translation
    )


@wp.kernel
def reset_normals_kernel(normals: wp.array(dtype=wp.vec3)):
    normals[wp.tid()] = wp.vec3(0.0)


@wp.kernel
def accumulate_triangle_normals_kernel(
    positions: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.vec3i),
    normals: wp.array(dtype=wp.vec3),
):
    triangle = triangles[wp.tid()]
    normal = wp.cross(
        positions[triangle[1]] - positions[triangle[0]],
        positions[triangle[2]] - positions[triangle[0]],
    )
    wp.atomic_add(normals, triangle[0], normal)
    wp.atomic_add(normals, triangle[1], normal)
    wp.atomic_add(normals, triangle[2], normal)


@wp.kernel
def normalize_normals_kernel(normals: wp.array(dtype=wp.vec3)):
    vertex_index = wp.tid()
    normal = normals[vertex_index]
    length = wp.length(normal)
    if length > 1.0e-12:
        normals[vertex_index] = normal * (1.0 / length)
    else:
        normals[vertex_index] = wp.vec3(0.0, 0.0, 1.0)


@wp.kernel
def gather_path_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    path_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    output[i] = positions[path_indices[i]]


class VesselSkinner:
    """Persistent device buffers and launches for centerline surface skinning.

    Usage is bind-once / update-per-step::

        skinner = VesselSkinner(vertices, triangles, edges, device=device)
        skinner.bind(rest_positions)
        skinner.compute_rest_coordinates(None, None, rest_positions_wp, rest_orientations)
        ...
        skinner.update(rod.x, rod.q)   # each step
        render(skinner.positions, skinner.normals)
    """

    def __init__(self, source_vertices, triangle_indices, edges, *, device=None):
        self.device = device or wp.get_device()
        vertices = np.asarray(source_vertices, dtype=np.float32).reshape((-1, 3))
        triangles = np.asarray(triangle_indices, dtype=np.int32).reshape((-1, 3))
        edges_np = np.asarray(edges, dtype=np.int32).reshape((-1, 2))
        self.vertex_count = len(vertices)
        self.triangle_count = len(triangles)
        self.edge_count = len(edges_np)
        if self.vertex_count == 0 or self.triangle_count == 0 or self.edge_count == 0:
            raise ValueError("skinning requires vertices, triangles, and centerline edges")
        if np.any(triangles < 0) or np.any(triangles >= self.vertex_count):
            raise ValueError("skinning triangle index is out of range")

        adjacency = [set() for _ in range(self.vertex_count)]
        for a, b, c in triangles.tolist():
            adjacency[a].update((b, c))
            adjacency[b].update((a, c))
            adjacency[c].update((a, b))
        adjacency_offsets = np.zeros(self.vertex_count + 1, dtype=np.int32)
        adjacency_offsets[1:] = np.cumsum(
            [len(neighbors) for neighbors in adjacency], dtype=np.int32
        )
        adjacency_vertices = np.asarray(
            [neighbor for neighbors in adjacency for neighbor in sorted(neighbors)],
            dtype=np.int32,
        )

        d = self.device
        self.source_vertices = wp.array(vertices, dtype=wp.vec3, device=d)
        self.triangles = wp.array(triangles, dtype=wp.vec3i, device=d)
        self.edges = wp.array(edges_np, dtype=wp.vec2i, device=d)
        self.adjacency_offsets = wp.array(adjacency_offsets, dtype=wp.int32, device=d)
        self.adjacency_vertices = wp.array(adjacency_vertices, dtype=wp.int32, device=d)
        self.bound_edges = wp.zeros(self.vertex_count, dtype=wp.int32, device=d)
        self.bound_coordinates = wp.zeros(self.vertex_count, dtype=wp.float32, device=d)
        self.rest_vertices = wp.zeros(self.vertex_count, dtype=wp.vec3, device=d)
        self.rest_centers = wp.zeros(self.vertex_count, dtype=wp.vec3, device=d)
        self.rest_orientations = wp.zeros(self.vertex_count, dtype=wp.quat, device=d)
        self.dq_real_a = wp.zeros(self.vertex_count, dtype=wp.quat, device=d)
        self.dq_dual_a = wp.zeros(self.vertex_count, dtype=wp.quat, device=d)
        self.dq_real_b = wp.zeros(self.vertex_count, dtype=wp.quat, device=d)
        self.dq_dual_b = wp.zeros(self.vertex_count, dtype=wp.quat, device=d)
        self.output_positions = wp.zeros(self.vertex_count, dtype=wp.vec3, device=d)
        self.output_normals = wp.zeros(self.vertex_count, dtype=wp.vec3, device=d)

    def bind(self, source_positions) -> None:
        """Assign each surface vertex to its nearest rest-pose centerline edge."""
        wp.launch(
            bind_vertices_to_edges_kernel,
            dim=self.vertex_count,
            inputs=[
                self.source_vertices,
                source_positions,
                self.edges,
                self.edge_count,
                self.bound_edges,
                self.bound_coordinates,
            ],
            device=self.device,
        )

    def compute_rest_coordinates(self, vertices, normals, positions, orientations) -> None:
        """Capture the rest vertex / center / frame each vertex is skinned from.

        Argument order matches upstream so its call sites work unchanged.
        ``normals`` is unused (upstream discards it too); pass ``None`` for
        ``vertices`` to reuse the vertices this skinner was constructed with.
        """
        del normals
        wp.launch(
            capture_rest_transforms_kernel,
            dim=self.vertex_count,
            inputs=[
                self.source_vertices if vertices is None else vertices,
                positions,
                orientations,
                self.edges,
                self.bound_edges,
                self.bound_coordinates,
                self.rest_vertices,
                self.rest_centers,
                self.rest_orientations,
            ],
            device=self.device,
        )

    def update(self, positions, orientations) -> None:
        """Skin the surface to the current centerline node positions / frames."""
        wp.launch(
            compute_vertex_dual_quaternions_kernel,
            dim=self.vertex_count,
            inputs=[
                positions,
                orientations,
                self.edges,
                self.bound_edges,
                self.bound_coordinates,
                self.rest_centers,
                self.rest_orientations,
                self.dq_real_a,
                self.dq_dual_a,
            ],
            device=self.device,
        )
        # Even pass count leaves the result back in the "a" buffers.
        for pass_index in range(SKINNING_SMOOTHING_PASSES):
            if pass_index % 2 == 0:
                input_real, input_dual = self.dq_real_a, self.dq_dual_a
                output_real, output_dual = self.dq_real_b, self.dq_dual_b
            else:
                input_real, input_dual = self.dq_real_b, self.dq_dual_b
                output_real, output_dual = self.dq_real_a, self.dq_dual_a
            wp.launch(
                smooth_dual_quaternions_kernel,
                dim=self.vertex_count,
                inputs=[
                    input_real,
                    input_dual,
                    self.adjacency_offsets,
                    self.adjacency_vertices,
                    output_real,
                    output_dual,
                ],
                device=self.device,
            )
        wp.launch(
            transform_rest_vertices_kernel,
            dim=self.vertex_count,
            inputs=[self.rest_vertices, self.dq_real_a, self.dq_dual_a, self.output_positions],
            device=self.device,
        )
        wp.launch(
            reset_normals_kernel,
            dim=self.vertex_count,
            inputs=[self.output_normals],
            device=self.device,
        )
        wp.launch(
            accumulate_triangle_normals_kernel,
            dim=self.triangle_count,
            inputs=[self.output_positions, self.triangles, self.output_normals],
            device=self.device,
        )
        wp.launch(
            normalize_normals_kernel,
            dim=self.vertex_count,
            inputs=[self.output_normals],
            device=self.device,
        )

    @property
    def positions(self) -> np.ndarray:
        return self.output_positions.numpy()

    @property
    def normals(self) -> np.ndarray:
        return self.output_normals.numpy()

    def graph_arrays(self) -> tuple:
        """Every persistent array retained by captured render launches."""
        return (
            self.source_vertices,
            self.triangles,
            self.edges,
            self.adjacency_offsets,
            self.adjacency_vertices,
            self.bound_edges,
            self.bound_coordinates,
            self.rest_vertices,
            self.rest_centers,
            self.rest_orientations,
            self.dq_real_a,
            self.dq_dual_a,
            self.dq_real_b,
            self.dq_dual_b,
            self.output_positions,
            self.output_normals,
        )


__all__ = [
    "SKINNING_SMOOTHING_PASSES",
    "VesselSkinner",
    "accumulate_triangle_normals_kernel",
    "bind_vertices_to_edges_kernel",
    "capture_rest_transforms_kernel",
    "compute_vertex_dual_quaternions_kernel",
    "gather_path_positions_kernel",
    "normalize_normals_kernel",
    "reset_normals_kernel",
    "smooth_dual_quaternions_kernel",
    "transform_rest_vertices_kernel",
]
