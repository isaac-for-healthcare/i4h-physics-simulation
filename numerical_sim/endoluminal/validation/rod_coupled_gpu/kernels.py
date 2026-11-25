# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-side linear algebra for the PR #22 validation fixture."""

import warp as wp

wp.set_module_options({"enable_backward": False})

Matrix6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
Matrix5 = wp.types.matrix(shape=(5, 5), dtype=wp.float64)
Vector5 = wp.types.vector(length=5, dtype=wp.float64)


@wp.kernel
def build_fixture_rows(
    q: wp.array(dtype=wp.vec3),
    guides: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=float),
    rows: wp.array2d(dtype=wp.vec3d),
    residual: wp.array(dtype=wp.float64),
    normals: wp.array(dtype=wp.vec3d),
):
    # Two transverse rows per sliding guide, plus one row for the nose.
    for row in range(5):
        for node in range(q.shape[0]):
            rows[row, node] = wp.vec3d(0.0)
    for guide in range(2):
        point = wp.vec3d(guides[guide])
        best = wp.float64(1.0e100)
        segment = int(0)
        fraction = wp.float64(0.0)
        gap = wp.vec3d(0.0)
        for edge in range(q.shape[0] - 1):
            start = wp.vec3d(q[edge])
            direction = wp.vec3d(q[edge + 1]) - start
            offset = point - start
            length_sq = wp.max(wp.dot(direction, direction), wp.float64(1.0e-30))
            u = wp.clamp(wp.dot(offset, direction) / length_sq, wp.float64(0.0), wp.float64(1.0))
            delta = offset - u * direction
            distance_sq = wp.dot(delta, delta)
            if distance_sq < best:
                best = distance_sq
                segment = edge
                fraction = u
                gap = delta
        tangent = wp.normalize(wp.vec3d(q[segment + 1]) - wp.vec3d(q[segment]))
        axis_index = int(0)
        for i in range(1, 3):
            if wp.abs(tangent[i]) < wp.abs(tangent[axis_index]):
                axis_index = i
        axis = wp.vec3d(0.0)
        axis[axis_index] = wp.float64(1.0)
        normal = wp.normalize(wp.cross(tangent, axis))
        other = wp.cross(tangent, normal)
        row = 2 * guide
        normals[row] = normal
        normals[row + 1] = other
        rows[row, segment] = (wp.float64(1.0) - fraction) * normal
        rows[row + 1, segment] = (wp.float64(1.0) - fraction) * other
        rows[row, segment + 1] = fraction * normal
        rows[row + 1, segment + 1] = fraction * other
        residual[row] = wp.dot(normal, gap)
        residual[row + 1] = wp.dot(other, gap)
    mid = q.shape[0] // 2
    rows[4, mid] = wp.vec3d(0.0, 1.0, 0.0)
    residual[4] = wp.float64(target[0]) - wp.float64(q[mid][1])


@wp.func
def rotational_row(jacobian: wp.array(dtype=float), edge: int, row: int, side: int) -> wp.vec3d:
    index = 36 * edge + 6 * row + 3 * side
    return wp.vec3d(wp.float64(jacobian[index]), wp.float64(jacobian[index + 1]), wp.float64(jacobian[index + 2]))


@wp.kernel
def assemble_blocks(
    jacobian: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.mat33),
    mass_inv: wp.array(dtype=float),
    compliance: wp.array(dtype=float),
    rod_rhs: wp.array(dtype=float),
    rows: wp.array2d(dtype=wp.vec3d),
    diagonal: wp.array(dtype=Matrix6),
    upper: wp.array(dtype=Matrix6),
    rhs: wp.array(dtype=Matrix6),
):
    edge = wp.tid()
    left = wp.mat33d(inertia[edge])
    right = wp.mat33d(inertia[edge + 1])
    w0 = wp.float64(mass_inv[edge])
    w1 = wp.float64(mass_inv[edge + 1])
    block = Matrix6(0.0)
    upper_block = Matrix6(0.0)
    vectors = Matrix6(0.0)
    for i in range(6):
        a = rotational_row(jacobian, edge, i, 0)
        b = rotational_row(jacobian, edge, i, 1)
        for j in range(6):
            block[i, j] = wp.dot(a, left * rotational_row(jacobian, edge, j, 0)) + wp.dot(
                b, right * rotational_row(jacobian, edge, j, 1)
            )
            if i == j:
                block[i, j] += wp.float64(compliance[6 * edge + i]) + wp.float64(1.0e-6)
                if i < 3:
                    block[i, j] += w0 + w1
            if edge + 1 < diagonal.shape[0]:
                upper_block[i, j] = wp.dot(b, right * rotational_row(jacobian, edge + 1, j, 0))
                if i == j and i < 3:
                    upper_block[i, j] -= w1
        vectors[i, 0] = wp.float64(rod_rhs[6 * edge + i])
        if i < 3:
            for row in range(5):
                vectors[i, row + 1] = w0 * rows[row, edge][i] - w1 * rows[row, edge + 1][i]
    diagonal[edge] = block
    upper[edge] = upper_block
    rhs[edge] = vectors


@wp.func
def solve_block(factor: Matrix6, rhs: Matrix6) -> Matrix6:
    answer = rhs
    for col in range(6):
        for i in range(6):
            value = answer[i, col]
            for j in range(i):
                value -= factor[i, j] * answer[j, col]
            answer[i, col] = value / factor[i, i]
        for i in range(5, -1, -1):
            value = answer[i, col]
            for j in range(i + 1, 6):
                value -= factor[j, i] * answer[j, col]
            answer[i, col] = value / factor[i, i]
    return answer


# Validation copy of the forward elimination and back substitution in
# numerical_sim/physics/src/i4h_physics/systems/rods/rod_direct_kernels.py::thomas
# (PR #22). Each factor is shared by the internal RHS and five fixture RHSs.
@wp.kernel
def thomas_fixture(
    diagonal: wp.array(dtype=Matrix6),
    upper: wp.array(dtype=Matrix6),
    rhs: wp.array(dtype=Matrix6),
    transfer: wp.array(dtype=Matrix6),
    answer: wp.array(dtype=Matrix6),
    failure: wp.array(dtype=int),
):
    if failure[0] != 0:
        return
    for edge in range(diagonal.shape[0]):
        block = diagonal[edge]
        vectors = rhs[edge]
        if edge > 0:
            lower = wp.transpose(upper[edge - 1])
            block -= lower * transfer[edge - 1]
            vectors -= lower * answer[edge - 1]

        factor = Matrix6(0.0)
        for i in range(6):
            for j in range(i + 1):
                value = block[i, j]
                for k in range(j):
                    value -= factor[i, k] * factor[j, k]
                if i == j:
                    if not wp.isfinite(value) or value <= wp.float64(0.0):
                        failure[0] = 1
                        return
                    factor[i, i] = wp.sqrt(value)
                else:
                    factor[i, j] = value / factor[j, j]

        if edge < diagonal.shape[0] - 1:
            transfer[edge] = solve_block(factor, upper[edge])
        answer[edge] = solve_block(factor, vectors)

    for edge in range(diagonal.shape[0] - 2, -1, -1):
        answer[edge] -= transfer[edge] * answer[edge + 1]


@wp.kernel
def solve_fixture(
    mass_inv: wp.array(dtype=float),
    rows: wp.array2d(dtype=wp.vec3d),
    residual: wp.array(dtype=wp.float64),
    rhs: wp.array(dtype=Matrix6),
    answer: wp.array(dtype=Matrix6),
    fixture_impulse: wp.array(dtype=Vector5),
    failure: wp.array(dtype=int),
):
    if failure[0] != 0:
        return
    # H = B W B^T - G^T A^-1 G, with G = J W B^T.
    matrix = Matrix5(0.0)
    impulse = Vector5(0.0)
    for i in range(5):
        impulse[i] = residual[i]
        for j in range(i + 1):
            v = wp.float64(0.0)
            for node in range(mass_inv.shape[0]):
                v += wp.float64(mass_inv[node]) * wp.dot(rows[i, node], rows[j, node])
            matrix[i, j] = v
    for edge in range(rhs.shape[0]):
        g = rhs[edge]
        solved = answer[edge]
        for i in range(5):
            for k in range(3):
                impulse[i] -= g[k, i + 1] * solved[k, 0]
            for j in range(i + 1):
                for k in range(3):
                    matrix[i, j] -= g[k, i + 1] * solved[k, j + 1]
    factor = Matrix5(0.0)
    for i in range(5):
        for j in range(i + 1):
            value = matrix[i, j]
            for k in range(j):
                value -= factor[i, k] * factor[j, k]
            if i == j:
                if not wp.isfinite(value) or value <= wp.float64(0.0):
                    failure[0] = 2
                    return
                factor[i, i] = wp.sqrt(value)
            else:
                factor[i, j] = value / factor[j, j]
        value = impulse[i]
        for j in range(i):
            value -= factor[i, j] * impulse[j]
        impulse[i] = value / factor[i, i]
    for i in range(4, -1, -1):
        value = impulse[i]
        for j in range(i + 1, 5):
            value -= factor[j, i] * impulse[j]
        impulse[i] = value / factor[i, i]
    fixture_impulse[0] = impulse


@wp.kernel
def recover_rod_impulses(
    answer: wp.array(dtype=Matrix6),
    fixture_impulse: wp.array(dtype=Vector5),
    failure: wp.array(dtype=int),
    first: int,
    delta: wp.array(dtype=wp.float64),
    accumulated: wp.array(dtype=float),
):
    if failure[0] != 0:
        return
    edge = wp.tid()
    solved = answer[edge]
    mu = fixture_impulse[0]
    for row in range(6):
        value = solved[row, 0]
        for i in range(5):
            value -= solved[row, i + 1] * mu[i]
        index = edge * 6 + row
        delta[index] = value
        if first != 0:
            accumulated[index] = wp.float32(value)
        else:
            accumulated[index] = wp.float32(wp.float64(accumulated[index]) + value)


@wp.kernel
def compute_fixture_corrections(
    jacobian: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.mat33),
    mass_inv: wp.array(dtype=float),
    rows: wp.array2d(dtype=wp.vec3d),
    delta: wp.array(dtype=wp.float64),
    fixture_impulse: wp.array(dtype=Vector5),
    failure: wp.array(dtype=int),
    position: wp.array(dtype=wp.vec3),
    rotation: wp.array(dtype=wp.vec3),
):
    if failure[0] != 0:
        return
    node = wp.tid()
    dp = wp.vec3d(0.0)
    dr = wp.vec3d(0.0)
    mu = fixture_impulse[0]
    for i in range(5):
        dp += rows[i, node] * mu[i]
    if node > 0:
        start = 6 * (node - 1)
        dp -= wp.vec3d(delta[start], delta[start + 1], delta[start + 2])
        for i in range(6):
            dr += rotational_row(jacobian, node - 1, i, 1) * delta[start + i]
    if node < mass_inv.shape[0] - 1:
        start = 6 * node
        dp += wp.vec3d(delta[start], delta[start + 1], delta[start + 2])
        for i in range(6):
            dr += rotational_row(jacobian, node, i, 0) * delta[start + i]
    position[node] = wp.vec3(wp.float64(mass_inv[node]) * dp)
    rotation[node] = wp.vec3(wp.mat33d(inertia[node]) * dr)


@wp.kernel
def record_reactions(
    fixture_impulse: wp.array(dtype=Vector5),
    normals: wp.array(dtype=wp.vec3d),
    failure: wp.array(dtype=int),
    dt: float,
    first: int,
    total_force: wp.array(dtype=wp.float64),
    total_reaction: wp.array(dtype=wp.vec3d),
    force: wp.array(dtype=float),
    reaction: wp.array(dtype=wp.vec3),
):
    if failure[0] != 0:
        force[0] = wp.float32(wp.nan)
        return
    inverse_dt_sq = wp.float64(1.0) / (wp.float64(dt) * wp.float64(dt))
    mu = fixture_impulse[0]
    nose = -mu[4] * inverse_dt_sq
    if first == 0:
        nose += total_force[0]
    total_force[0] = nose
    force[0] = wp.float32(nose)
    for guide in range(2):
        row = 2 * guide
        support = (normals[row] * mu[row] + normals[row + 1] * mu[row + 1]) * inverse_dt_sq
        if first == 0:
            support += total_reaction[guide]
        total_reaction[guide] = support
        reaction[guide] = wp.vec3(support)
