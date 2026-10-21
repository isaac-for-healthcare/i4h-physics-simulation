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


@wp.func
def _containment_pull(
    gap: float,
    free_radius: float,
    interior_deadband: float,
    interior_stiffness: float,
) -> wp.vec2:
    """How far to pull a sample toward the vessel axis, and whether it is contact.

    ``gap`` is the sample's distance from the axis less the radius it is free to
    occupy, so it is positive once the wire's surface is through the wall and
    negative while the wire is inside.

    Outside is a contact and is projected in full, which is what this kernel has
    always done. Inside was left entirely alone, and that is what lets excess arc
    length bow across the lumen until it finds the wall: nothing in the solve
    prefers the axis, so any spare length is free to go sideways. A restoring
    pull that begins only past ``interior_deadband`` of the free radius takes
    that freedom away without prescribing a route -- the inner lumen stays
    somewhere the wire may be, so its shape there is still the solve's answer
    rather than the centerline's.

    Returns:
        ``(pull, is_contact)``. ``pull`` is how far to move toward the axis, and
        is zero when the sample should be left alone. ``is_contact`` is ``1.0``
        only for a real wall contact, so an interior pull neither deforms the
        vessel nor is reported as penetration.
    """
    if gap > 0.0:
        return wp.vec2(gap, 1.0)
    if interior_stiffness <= 0.0:
        return wp.vec2(0.0, 0.0)
    deadband = interior_deadband * free_radius
    distance = gap + free_radius
    if distance <= deadband:
        return wp.vec2(0.0, 0.0)
    # At unit stiffness this lands the sample exactly on the deadband surface, so
    # no stiffness in (0, 1] can carry it past that surface, let alone the axis.
    return wp.vec2((distance - deadband) * interior_stiffness, 0.0)


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
    interior_deadband: float,
    interior_stiffness: float,
    two_way: int,
    vessel_response: float,
    catheter_corrections: wp.array(dtype=wp.vec3),
    catheter_counts: wp.array(dtype=wp.float32),
    centerline_corrections: wp.array(dtype=wp.vec3),
    centerline_counts: wp.array(dtype=wp.float32),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
):
    """Project nodes and interior edge samples against a tapered-tube union.

    Always searches all centerline edges (no candidate cache). ``two_way`` is
    ``1``/``0`` for Warp friendliness.

    ``contact_depth`` and ``contact_count`` are single-element accumulators
    holding the deepest penetration and the number of contacting samples. They
    are recorded only once a contact has produced a correction, so a contact
    whose weights all vanish is not counted.
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
    best_free_radius = float(0.0)
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
                best_free_radius = radius - catheter_radius

    if best_edge < 0 or best_gap > max_distance:
        return
    pull = _containment_pull(best_gap, best_free_radius, interior_deadband, interior_stiffness)
    if pull[0] <= 0.0:
        return
    is_contact = pull[1]
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
    # An interior pull is a regulariser on the wire, not a force the wall exerts,
    # so it must not push the vessel back: a wall deformed by it would report a
    # contact that never happened.
    reciprocal = float(0.0)
    if two_way != 0 and is_contact > 0.0:
        reciprocal = vessel_response
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
    magnitude = pull[0] / denominator
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
    if is_contact > 0.0:
        wp.atomic_max(contact_depth, 0, best_gap)
        wp.atomic_add(contact_count, 0, 1)


@wp.kernel
def reset_centerline_nodes_kernel(
    env_ids: wp.array(dtype=wp.int32),
    nodes_per_env: int,
    rest_x: wp.array(dtype=wp.vec3),
    x: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
):
    """Restore rest node positions and clear velocity for the listed envs."""
    slot, local = wp.tid()
    i = env_ids[slot] * nodes_per_env + local
    rest = rest_x[i]
    x[i] = rest
    p[i] = rest
    v[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def reset_centerline_contact_kernel(
    env_ids: wp.array(dtype=wp.int32),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
):
    """Clear contact accumulators for the listed envs only.

    Zeroing the whole buffer would erase penetration and contact history for
    envs still mid-episode, so a trainer that resets a subset and then builds
    observations before the next step would read no wall contact for episodes
    that are in fact touching the wall.
    """
    env = env_ids[wp.tid()]
    contact_depth[env] = 0.0
    contact_count[env] = 0


@wp.kernel
def reset_centerline_segments_kernel(
    env_ids: wp.array(dtype=wp.int32),
    segments_per_env: int,
    rest_q: wp.array(dtype=wp.quat),
    q: wp.array(dtype=wp.quat),
    u: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
):
    """Restore rest segment frames and clear angular velocity for the listed envs."""
    slot, local = wp.tid()
    i = env_ids[slot] * segments_per_env + local
    rest = rest_q[i]
    q[i] = rest
    u[i] = rest
    omega[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _contain_env_sample(
    env: int,
    sample: int,
    catheter: wp.array(dtype=wp.vec3),
    catheter_inv_mass: wp.array(dtype=wp.float32),
    catheter_points_per_env: int,
    centerline: wp.array(dtype=wp.vec3),
    centerline_edges: wp.array(dtype=wp.vec2i),
    centerline_radii: wp.array(dtype=wp.float32),
    centerline_inv_mass: wp.array(dtype=wp.float32),
    centerline_nodes_per_env: int,
    centerline_edges_per_env: int,
    edge_count: int,
    open_root: int,
    open_root_neighbor: int,
    catheter_radius: float,
    max_distance: float,
    interior_deadband: float,
    interior_stiffness: float,
    two_way: int,
    vessel_response: float,
    catheter_corrections: wp.array(dtype=wp.vec3),
    catheter_counts: wp.array(dtype=wp.float32),
    centerline_corrections: wp.array(dtype=wp.vec3),
    centerline_counts: wp.array(dtype=wp.float32),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
):
    """One env's sample against that env's first ``edge_count`` vessel edges.

    ``centerline_edges_per_env`` is the window stride and ``edge_count`` how many
    edges at the start of the window are real; the rest is padding.
    """
    cath_base = env * catheter_points_per_env
    node_base = env * centerline_nodes_per_env

    ca = int(sample)
    cb = int(sample)
    catheter_t = float(0.0)
    point = wp.vec3(0.0)
    if sample < catheter_points_per_env:
        ca = cath_base + sample
        cb = ca
        point = catheter[ca]
    else:
        local = sample - catheter_points_per_env
        ca = cath_base + local // CENTERLINE_EDGE_SAMPLES
        cb = ca + 1
        catheter_t = float(local % CENTERLINE_EDGE_SAMPLES + 1) / float(CENTERLINE_EDGE_SAMPLES + 1)
        point = catheter[ca] * (1.0 - catheter_t) + catheter[cb] * catheter_t

    if open_root >= 0 and open_root_neighbor >= 0:
        root = centerline[node_base + open_root]
        inward = centerline[node_base + open_root_neighbor] - root
        if wp.dot(point - root, inward) < 0.0:
            return

    best_gap = float(1.0e30)
    best_edge = int(-1)
    best_t = float(0.0)
    best_gradient = wp.vec3(0.0)
    best_free_radius = float(0.0)
    edge_begin = env * centerline_edges_per_env
    for edge_index in range(edge_begin, edge_begin + edge_count):
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
                best_free_radius = radius - catheter_radius

    if best_edge < 0 or best_gap > max_distance:
        return
    pull = _containment_pull(best_gap, best_free_radius, interior_deadband, interior_stiffness)
    if pull[0] <= 0.0:
        return
    is_contact = pull[1]
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
    # An interior pull is a regulariser on the wire, not a force the wall exerts,
    # so it must not push the vessel back: a wall deformed by it would report a
    # contact that never happened.
    reciprocal = float(0.0)
    if two_way != 0 and is_contact > 0.0:
        reciprocal = vessel_response
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
    magnitude = pull[0] / denominator
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
    if is_contact > 0.0:
        wp.atomic_max(contact_depth, env, best_gap)
        wp.atomic_add(contact_count, env, 1)


@wp.kernel
def project_centerline_containment_batched_kernel(
    catheter: wp.array(dtype=wp.vec3),
    catheter_inv_mass: wp.array(dtype=wp.float32),
    catheter_points_per_env: int,
    centerline: wp.array(dtype=wp.vec3),
    centerline_edges: wp.array(dtype=wp.vec2i),
    centerline_radii: wp.array(dtype=wp.float32),
    centerline_inv_mass: wp.array(dtype=wp.float32),
    centerline_nodes_per_env: int,
    centerline_edges_per_env: int,
    open_root: int,
    open_root_neighbor: int,
    catheter_radius: float,
    max_distance: float,
    interior_deadband: float,
    interior_stiffness: float,
    two_way: int,
    vessel_response: float,
    catheter_corrections: wp.array(dtype=wp.vec3),
    catheter_counts: wp.array(dtype=wp.float32),
    centerline_corrections: wp.array(dtype=wp.vec3),
    centerline_counts: wp.array(dtype=wp.float32),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
):
    """Per-env tapered-tube containment in one launch over ``(num_envs, samples)``.

    Same projection as :func:`project_centerline_containment_kernel`, with every
    catheter and centerline index shifted into the calling env's own window, so
    each env contains its own rod against its own vessel and no correction can
    cross between envs. ``centerline_edges`` therefore holds *global* node
    indices (as tiled by :class:`CosseratRod`), while ``open_root`` and
    ``open_root_neighbor`` are env-local and shifted here. Every env shares one
    vessel topology; see :func:`project_centerline_containment_per_env_kernel`
    for envs whose vessels differ.

    ``contact_depth`` and ``contact_count`` hold one slot per env.
    """
    env, sample = wp.tid()
    _contain_env_sample(
        env, sample, catheter, catheter_inv_mass, catheter_points_per_env,
        centerline, centerline_edges, centerline_radii, centerline_inv_mass,
        centerline_nodes_per_env, centerline_edges_per_env, centerline_edges_per_env,
        open_root, open_root_neighbor, catheter_radius, max_distance,
        interior_deadband, interior_stiffness, two_way, vessel_response,
        catheter_corrections, catheter_counts, centerline_corrections, centerline_counts,
        contact_depth, contact_count,
    )


@wp.kernel
def project_centerline_containment_per_env_kernel(
    catheter: wp.array(dtype=wp.vec3),
    catheter_inv_mass: wp.array(dtype=wp.float32),
    catheter_points_per_env: int,
    centerline: wp.array(dtype=wp.vec3),
    centerline_edges: wp.array(dtype=wp.vec2i),
    centerline_radii: wp.array(dtype=wp.float32),
    centerline_inv_mass: wp.array(dtype=wp.float32),
    centerline_nodes_per_env: int,
    centerline_edges_per_env: int,
    edge_counts: wp.array(dtype=wp.int32),
    open_roots: wp.array(dtype=wp.int32),
    open_root_neighbors: wp.array(dtype=wp.int32),
    catheter_radius: float,
    max_distance: float,
    interior_deadband: float,
    interior_stiffness: float,
    two_way: int,
    vessel_response: float,
    catheter_corrections: wp.array(dtype=wp.vec3),
    catheter_counts: wp.array(dtype=wp.float32),
    centerline_corrections: wp.array(dtype=wp.vec3),
    centerline_counts: wp.array(dtype=wp.float32),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
):
    """:func:`project_centerline_containment_batched_kernel` for envs with different vessels.

    Each env's vessel occupies a padded window of ``centerline_nodes_per_env``
    nodes and ``centerline_edges_per_env`` edges, of which ``edge_counts[env]``
    are real. ``open_roots`` and ``open_root_neighbors`` give each env's own
    entry, env-local, ``-1`` for a closed root.
    """
    env, sample = wp.tid()
    _contain_env_sample(
        env, sample, catheter, catheter_inv_mass, catheter_points_per_env,
        centerline, centerline_edges, centerline_radii, centerline_inv_mass,
        centerline_nodes_per_env, centerline_edges_per_env, edge_counts[env],
        open_roots[env], open_root_neighbors[env], catheter_radius, max_distance,
        interior_deadband, interior_stiffness, two_way, vessel_response,
        catheter_corrections, catheter_counts, centerline_corrections, centerline_counts,
        contact_depth, contact_count,
    )


__all__ = [
    "CENTERLINE_EDGE_SAMPLES",
    "apply_centerline_corrections_kernel",
    "clamp_positions_delta_kernel",
    "project_centerline_containment_batched_kernel",
    "project_centerline_containment_kernel",
    "project_centerline_containment_per_env_kernel",
    "reset_centerline_contact_kernel",
    "reset_centerline_nodes_kernel",
    "reset_centerline_segments_kernel",
]
