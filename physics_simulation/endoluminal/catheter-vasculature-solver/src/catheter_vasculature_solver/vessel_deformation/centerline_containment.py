# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Linear tapered-tube centerline containment; candidate cache / spline deferred.

"""Live tapered-tube containment against a deforming Cosserat centerline."""

from __future__ import annotations

import warp as wp

CENTERLINE_EDGE_SAMPLES = 3


@wp.func
def _clamp_position_delta(position: wp.vec3, start: wp.vec3, max_delta: float) -> wp.vec3:
    delta = position - start
    length = wp.length(delta)
    if length > max_delta and length > 1.0e-12:
        return start + delta * (max_delta / length)
    return position


@wp.kernel
def clamp_positions_delta_kernel(
    positions: wp.array(dtype=wp.vec3),
    starts: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    max_delta: float,
):
    i = wp.tid()
    if inv_masses[i] > 0.0:
        positions[i] = _clamp_position_delta(positions[i], starts[i], max_delta)


@wp.kernel
def apply_centerline_corrections_kernel(
    positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    corrections: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if inv_masses[i] > 0.0 and counts[i] > 0.0:
        positions[i] = positions[i] + corrections[i] / counts[i]


@wp.kernel
def project_centerline_containment_kernel(
    catheter: wp.array(dtype=wp.vec3),
    catheter_inv_mass: wp.array(dtype=wp.float32),
    catheter_count: int,
    centerline: wp.array(dtype=wp.vec3),
    centerline_edges: wp.array(dtype=wp.vec2i),
    centerline_radii: wp.array(dtype=wp.float32),
    centerline_inv_mass: wp.array(dtype=wp.float32),
    open_root: int,
    open_root_neighbor: int,
    catheter_radius: float,
    max_distance: float,
    two_way: int,
    vessel_response: float,
    catheter_corrections: wp.array(dtype=wp.vec3),
    catheter_counts: wp.array(dtype=wp.float32),
    centerline_corrections: wp.array(dtype=wp.vec3),
    centerline_counts: wp.array(dtype=wp.float32),
):
    """Project nodes and interior edge samples against a tapered-tube union.

    Always searches all centerline edges (no candidate cache). ``two_way`` is
    ``1``/``0`` for Warp friendliness.
    """
    sample = wp.tid()
    ca = int(sample)
    cb = int(sample)
    catheter_t = float(0.0)
    point = wp.vec3(0.0)
    if sample < catheter_count:
        point = catheter[sample]
    else:
        local = sample - catheter_count
        ca = local // CENTERLINE_EDGE_SAMPLES
        cb = ca + 1
        catheter_t = float(local % CENTERLINE_EDGE_SAMPLES + 1) / float(CENTERLINE_EDGE_SAMPLES + 1)
        point = catheter[ca] * (1.0 - catheter_t) + catheter[cb] * catheter_t

    if open_root >= 0 and open_root_neighbor >= 0:
        root = centerline[open_root]
        inward = centerline[open_root_neighbor] - root
        if wp.dot(point - root, inward) < 0.0:
            return

    best_gap = float(1.0e30)
    best_edge = int(-1)
    best_t = float(0.0)
    best_gradient = wp.vec3(0.0)
    search_count = centerline_edges.shape[0]
    for edge_index in range(search_count):
        edge = centerline_edges[edge_index]
        a = centerline[edge[0]]
        b = centerline[edge[1]]
        axis = b - a
        length = wp.length(axis)
        if length > 1.0e-8:
            tangent = axis / length
            projected = wp.clamp(wp.dot(point - a, tangent) / length, 0.0, 1.0)
            radial_at_projection = point - (a + axis * projected)
            radial_length = wp.length(radial_at_projection)
            dr = centerline_radii[edge[1]] - centerline_radii[edge[0]]
            t = projected
            if wp.abs(dr) < length - 1.0e-8:
                denom = length * wp.sqrt(wp.max(length * length - dr * dr, 1.0e-12))
                t = wp.clamp(projected + dr * radial_length / denom, 0.0, 1.0)
            closest = a + axis * t
            delta = point - closest
            distance = wp.length(delta)
            radius = centerline_radii[edge[0]] * (1.0 - t) + centerline_radii[edge[1]] * t
            gap = distance - (radius - catheter_radius)
            if gap < best_gap:
                radial = wp.vec3(1.0, 0.0, 0.0)
                if distance > 1.0e-8:
                    radial = delta / distance
                gradient = radial - tangent * (dr / length)
                gradient_length = wp.length(gradient)
                if gradient_length > 1.0e-8:
                    gradient = gradient / gradient_length
                best_gap = gap
                best_edge = edge_index
                best_t = t
                best_gradient = gradient

    if best_edge < 0 or best_gap <= 0.0 or best_gap > max_distance:
        return
    vessel_edge = centerline_edges[best_edge]
    catheter_a_weight = 1.0 - catheter_t
    catheter_b_weight = catheter_t
    if ca == cb:
        catheter_a_weight = 1.0
        catheter_b_weight = 0.0
    vessel_a_weight = 1.0 - best_t
    vessel_b_weight = best_t
    wc0 = catheter_inv_mass[ca]
    wc1 = catheter_inv_mass[cb]
    wv0 = centerline_inv_mass[vessel_edge[0]]
    wv1 = centerline_inv_mass[vessel_edge[1]]
    reciprocal = vessel_response if two_way != 0 else 0.0
    denominator = (
        wc0 * catheter_a_weight * catheter_a_weight
        + wc1 * catheter_b_weight * catheter_b_weight
        + reciprocal * (
            wv0 * vessel_a_weight * vessel_a_weight
            + wv1 * vessel_b_weight * vessel_b_weight
        )
    )
    if denominator <= 1.0e-12:
        return
    magnitude = best_gap / denominator
    if wc0 > 0.0 and catheter_a_weight > 0.0:
        wp.atomic_add(catheter_corrections, ca, -best_gradient * magnitude * wc0 * catheter_a_weight)
        wp.atomic_add(catheter_counts, ca, catheter_a_weight)
    if wc1 > 0.0 and catheter_b_weight > 0.0:
        wp.atomic_add(catheter_corrections, cb, -best_gradient * magnitude * wc1 * catheter_b_weight)
        wp.atomic_add(catheter_counts, cb, catheter_b_weight)
    if reciprocal > 0.0 and wv0 > 0.0 and vessel_a_weight > 0.0:
        wp.atomic_add(
            centerline_corrections,
            vessel_edge[0],
            best_gradient * magnitude * reciprocal * wv0 * vessel_a_weight,
        )
        wp.atomic_add(centerline_counts, vessel_edge[0], vessel_a_weight)
    if reciprocal > 0.0 and wv1 > 0.0 and vessel_b_weight > 0.0:
        wp.atomic_add(
            centerline_corrections,
            vessel_edge[1],
            best_gradient * magnitude * reciprocal * wv1 * vessel_b_weight,
        )
        wp.atomic_add(centerline_counts, vessel_edge[1], vessel_b_weight)


__all__ = [
    "CENTERLINE_EDGE_SAMPLES",
    "apply_centerline_corrections_kernel",
    "clamp_positions_delta_kernel",
    "project_centerline_containment_kernel",
]
