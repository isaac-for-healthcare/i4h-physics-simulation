# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels used by the cloth and tetrahedral XPBD projection path."""

from __future__ import annotations

import warp as wp


@wp.kernel
def _warp_gather_deformable(
    particle_indices: wp.array(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = particle_indices[i]
    positions[i] = q[p]
    velocities[i] = qd[p]


@wp.kernel
def _warp_predict_deformable(
    particle_indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    particle_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    damping: float,
    predicted: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = particle_indices[i]
    w = inv_masses[i]
    if w > 0.0:
        world = particle_world[i]
        if world < 0:
            world = 0
        v = velocities[i] + (particle_f[p] * w + gravity[world]) * dt
        v = v * (1.0 - damping)
        velocities[i] = v
        predicted[i] = positions[i] + v * dt
    else:
        velocities[i] = wp.vec3(0.0)
        predicted[i] = positions[i]


@wp.kernel
def _warp_project_distance_gs(
    indices: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    predicted: wp.array(dtype=wp.vec3),
    constraint_offset: int,
    dt: float,
    relaxation: float,
):
    ci = wp.tid() + constraint_offset
    edge = indices[ci]
    i = edge[0]
    j = edge[1]
    d = predicted[j] - predicted[i]
    length = wp.length(d)
    if length > 1.0e-12:
        n = d / length
        alpha = compliance[ci] / (dt * dt)
        denom = inv_masses[i] + inv_masses[j] + alpha
        if denom > 0.0:
            dlambda = (rest_lengths[ci] - length - alpha * lambdas[ci]) / denom
            lambdas[ci] = lambdas[ci] + dlambda
            predicted[i] = predicted[i] - n * (relaxation * inv_masses[i] * dlambda)
            predicted[j] = predicted[j] + n * (relaxation * inv_masses[j] * dlambda)


@wp.kernel
def _warp_project_distance_jacobi(
    indices: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    predicted: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    dt: float,
    relaxation: float,
):
    ci = wp.tid()
    edge = indices[ci]
    i = edge[0]
    j = edge[1]
    d = predicted[j] - predicted[i]
    length = wp.length(d)
    if length > 1.0e-12:
        n = d / length
        alpha = compliance[ci] / (dt * dt)
        denom = inv_masses[i] + inv_masses[j] + alpha
        if denom > 0.0:
            dlambda = (rest_lengths[ci] - length - alpha * lambdas[ci]) / denom
            lambdas[ci] = lambdas[ci] + dlambda
            wp.atomic_add(corrections, i, -n * (relaxation * inv_masses[i] * dlambda))
            wp.atomic_add(corrections, j, n * (relaxation * inv_masses[j] * dlambda))


@wp.func
def _normalized_vector_derivative(
    vector_length: float, unit_vector: wp.vec3, derivative: wp.mat33
) -> wp.mat33:
    projection = wp.identity(n=3, dtype=float) - wp.outer(unit_vector, unit_vector)
    return projection * derivative / vector_length


@wp.func
def _angle_derivative(
    n0: wp.vec3,
    n1: wp.vec3,
    edge: wp.vec3,
    dn0: wp.mat33,
    dn1: wp.mat33,
    sin_angle: float,
    cos_angle: float,
) -> wp.vec3:
    dsin = wp.transpose(wp.skew(n0) * dn1 - wp.skew(n1) * dn0) * edge
    dcos = wp.transpose(dn0) * n1 + wp.transpose(dn1) * n0
    return dsin * cos_angle - dcos * sin_angle


@wp.func
def _dihedral_angle_gradients(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3
):
    """Return signed angle and gradients for (opposite0, opposite1, edge0, edge1)."""
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    edge = x3 - x2

    raw_n0 = wp.cross(x02, x03)
    raw_n1 = wp.cross(x13, x12)
    n0_length = wp.length(raw_n0)
    n1_length = wp.length(raw_n1)
    edge_length = wp.length(edge)
    if n0_length <= 1.0e-8 or n1_length <= 1.0e-8 or edge_length <= 1.0e-8:
        zero = wp.vec3(0.0)
        return 0.0, zero, zero, zero, zero, False

    n0 = raw_n0 / n0_length
    n1 = raw_n1 / n1_length
    edge_unit = edge / edge_length
    sin_angle = wp.dot(wp.cross(n0, n1), edge_unit)
    cos_angle = wp.clamp(wp.dot(n0, n1), -1.0, 1.0)
    angle = wp.atan2(sin_angle, cos_angle)

    dn0_x0 = _normalized_vector_derivative(n0_length, n0, wp.skew(edge))
    dn1_x0 = wp.mat33(0.0)
    dn0_x1 = wp.mat33(0.0)
    dn1_x1 = _normalized_vector_derivative(n1_length, n1, -wp.skew(edge))
    dn0_x2 = _normalized_vector_derivative(n0_length, n0, -wp.skew(x03))
    dn1_x2 = _normalized_vector_derivative(n1_length, n1, wp.skew(x13))
    dn0_x3 = _normalized_vector_derivative(n0_length, n0, wp.skew(x02))
    dn1_x3 = _normalized_vector_derivative(n1_length, n1, -wp.skew(x12))

    g0 = _angle_derivative(n0, n1, edge_unit, dn0_x0, dn1_x0, sin_angle, cos_angle)
    g1 = _angle_derivative(n0, n1, edge_unit, dn0_x1, dn1_x1, sin_angle, cos_angle)
    g2 = _angle_derivative(n0, n1, edge_unit, dn0_x2, dn1_x2, sin_angle, cos_angle)
    g3 = _angle_derivative(n0, n1, edge_unit, dn0_x3, dn1_x3, sin_angle, cos_angle)
    return angle, g0, g1, g2, g3, True


@wp.func
def _wrapped_angle_difference(angle: float, rest_angle: float) -> float:
    difference = angle - rest_angle
    if difference > wp.pi:
        difference = difference - 2.0 * wp.pi
    elif difference < -wp.pi:
        difference = difference + 2.0 * wp.pi
    return difference


@wp.kernel
def _warp_project_dihedral_gs(
    indices: wp.array(dtype=wp.vec4i),
    rest_angles: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    predicted: wp.array(dtype=wp.vec3),
    constraint_offset: int,
    dt: float,
    relaxation: float,
):
    ci = wp.tid() + constraint_offset
    bend = indices[ci]
    i, j, k, p3 = bend[0], bend[1], bend[2], bend[3]
    angle, g0, g1, g2, g3, valid = _dihedral_angle_gradients(
        predicted[i], predicted[j], predicted[k], predicted[p3]
    )
    if valid:
        alpha = compliance[ci] / (dt * dt)
        denom = (
            inv_masses[i] * wp.dot(g0, g0)
            + inv_masses[j] * wp.dot(g1, g1)
            + inv_masses[k] * wp.dot(g2, g2)
            + inv_masses[p3] * wp.dot(g3, g3)
            + alpha
        )
        if denom > 0.0:
            constraint = _wrapped_angle_difference(angle, rest_angles[ci])
            dlambda = (-constraint - alpha * lambdas[ci]) / denom
            lambdas[ci] = lambdas[ci] + dlambda
            scale = relaxation * dlambda
            predicted[i] = predicted[i] + g0 * (inv_masses[i] * scale)
            predicted[j] = predicted[j] + g1 * (inv_masses[j] * scale)
            predicted[k] = predicted[k] + g2 * (inv_masses[k] * scale)
            predicted[p3] = predicted[p3] + g3 * (inv_masses[p3] * scale)


@wp.kernel
def _warp_project_dihedral_jacobi(
    indices: wp.array(dtype=wp.vec4i),
    rest_angles: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    predicted: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    dt: float,
    relaxation: float,
):
    ci = wp.tid()
    bend = indices[ci]
    i, j, k, p3 = bend[0], bend[1], bend[2], bend[3]
    angle, g0, g1, g2, g3, valid = _dihedral_angle_gradients(
        predicted[i], predicted[j], predicted[k], predicted[p3]
    )
    if valid:
        alpha = compliance[ci] / (dt * dt)
        denom = (
            inv_masses[i] * wp.dot(g0, g0)
            + inv_masses[j] * wp.dot(g1, g1)
            + inv_masses[k] * wp.dot(g2, g2)
            + inv_masses[p3] * wp.dot(g3, g3)
            + alpha
        )
        if denom > 0.0:
            constraint = _wrapped_angle_difference(angle, rest_angles[ci])
            dlambda = (-constraint - alpha * lambdas[ci]) / denom
            lambdas[ci] = lambdas[ci] + dlambda
            scale = relaxation * dlambda
            wp.atomic_add(corrections, i, g0 * (inv_masses[i] * scale))
            wp.atomic_add(corrections, j, g1 * (inv_masses[j] * scale))
            wp.atomic_add(corrections, k, g2 * (inv_masses[k] * scale))
            wp.atomic_add(corrections, p3, g3 * (inv_masses[p3] * scale))


@wp.func
def _tet_gradients(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    g1 = wp.cross(x2 - x0, x3 - x0) / 6.0
    g2 = wp.cross(x3 - x0, x1 - x0) / 6.0
    g3 = wp.cross(x1 - x0, x2 - x0) / 6.0
    g0 = -(g1 + g2 + g3)
    return g0, g1, g2, g3


@wp.kernel
def _warp_project_volume_gs(
    indices: wp.array(dtype=wp.vec4i),
    rest_volumes: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    predicted: wp.array(dtype=wp.vec3),
    constraint_offset: int,
    dt: float,
    relaxation: float,
):
    ci = wp.tid() + constraint_offset
    tet = indices[ci]
    i, j, k, p3 = tet[0], tet[1], tet[2], tet[3]
    x0, x1, x2, x3 = predicted[i], predicted[j], predicted[k], predicted[p3]
    g0, g1, g2, g3 = _tet_gradients(x0, x1, x2, x3)
    volume = wp.dot(x1 - x0, wp.cross(x2 - x0, x3 - x0)) / 6.0
    alpha = compliance[ci] / (dt * dt)
    denom = (
        inv_masses[i] * wp.dot(g0, g0)
        + inv_masses[j] * wp.dot(g1, g1)
        + inv_masses[k] * wp.dot(g2, g2)
        + inv_masses[p3] * wp.dot(g3, g3)
        + alpha
    )
    if denom > 0.0:
        dlambda = (-volume + rest_volumes[ci] - alpha * lambdas[ci]) / denom
        lambdas[ci] = lambdas[ci] + dlambda
        predicted[i] = predicted[i] + g0 * (relaxation * inv_masses[i] * dlambda)
        predicted[j] = predicted[j] + g1 * (relaxation * inv_masses[j] * dlambda)
        predicted[k] = predicted[k] + g2 * (relaxation * inv_masses[k] * dlambda)
        predicted[p3] = predicted[p3] + g3 * (relaxation * inv_masses[p3] * dlambda)


@wp.kernel
def _warp_project_volume_jacobi(
    indices: wp.array(dtype=wp.vec4i),
    rest_volumes: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    predicted: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    dt: float,
    relaxation: float,
):
    ci = wp.tid()
    tet = indices[ci]
    i, j, k, p3 = tet[0], tet[1], tet[2], tet[3]
    x0, x1, x2, x3 = predicted[i], predicted[j], predicted[k], predicted[p3]
    g0, g1, g2, g3 = _tet_gradients(x0, x1, x2, x3)
    volume = wp.dot(x1 - x0, wp.cross(x2 - x0, x3 - x0)) / 6.0
    alpha = compliance[ci] / (dt * dt)
    denom = (
        inv_masses[i] * wp.dot(g0, g0)
        + inv_masses[j] * wp.dot(g1, g1)
        + inv_masses[k] * wp.dot(g2, g2)
        + inv_masses[p3] * wp.dot(g3, g3)
        + alpha
    )
    if denom > 0.0:
        dlambda = (-volume + rest_volumes[ci] - alpha * lambdas[ci]) / denom
        lambdas[ci] = lambdas[ci] + dlambda
        wp.atomic_add(corrections, i, g0 * (relaxation * inv_masses[i] * dlambda))
        wp.atomic_add(corrections, j, g1 * (relaxation * inv_masses[j] * dlambda))
        wp.atomic_add(corrections, k, g2 * (relaxation * inv_masses[k] * dlambda))
        wp.atomic_add(corrections, p3, g3 * (relaxation * inv_masses[p3] * dlambda))


@wp.kernel
def _warp_apply_deformable_corrections(
    predicted: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    incidence_counts: wp.array(dtype=wp.int32),
    inv_masses: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    count = incidence_counts[i]
    if inv_masses[i] > 0.0 and count > 0:
        predicted[i] = predicted[i] + corrections[i] / float(count)


@wp.kernel
def _warp_scatter_deformable(
    particle_indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    dt: float,
    q_out: wp.array(dtype=wp.vec3),
    qd_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = particle_indices[i]
    if inv_masses[i] > 0.0:
        q_out[p] = predicted[i]
        qd_out[p] = (predicted[i] - positions[i]) / dt
    else:
        q_out[p] = positions[i]
        qd_out[p] = wp.vec3(0.0)
