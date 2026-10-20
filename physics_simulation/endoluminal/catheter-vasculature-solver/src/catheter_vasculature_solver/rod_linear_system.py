# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Double precision block factorization for stiff, slender Cosserat rods.

State stays float32. Assembly and elimination need more precision: physical
rotational inverse inertia can exceed translational inverse mass by millions,
and eliminating adjacent constraints subtracts nearly equal blocks.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})

Mat6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
Vec6 = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def mass_product(
    jp: wp.array(dtype=float),
    jr: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=float),
    edge_a: int,
    row: int,
    side_a: int,
    edge_b: int,
    col: int,
    side_b: int,
    particle: int,
) -> wp.float64:
    value = wp.float64(0.0)
    for k in range(3):
        a = edge_a * 36 + row * 6 + side_a * 3 + k
        b = edge_b * 36 + col * 6 + side_b * 3 + k
        value += wp.float64(jp[a]) * wp.float64(mass[particle]) * wp.float64(jp[b])
        for axis in range(3):
            b_rot = edge_b * 36 + col * 6 + side_b * 3 + axis
            value += wp.float64(jr[a]) * wp.float64(inertia[particle * 9 + k * 3 + axis]) * wp.float64(jr[b_rot])
    return value


@wp.kernel
def assemble(
    jp: wp.array(dtype=float),
    jr: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=float),
    compliance: wp.array(dtype=float),
    constraint: wp.array(dtype=float),
    multipliers: wp.array(dtype=float),
    edges_per_rod: int,
    diagonal: wp.array(dtype=Mat6),
    upper: wp.array(dtype=Mat6),
    rhs: wp.array(dtype=Vec6),
):
    edge = wp.tid()
    local = edge % edges_per_rod
    particle = edge + edge // edges_per_rod
    d = Mat6()
    u = Mat6()
    b = Vec6()
    for row in range(6):
        alpha = wp.float64(compliance[edge * 6 + row])
        b[row] = -wp.float64(constraint[edge * 6 + row]) - alpha * wp.float64(multipliers[edge * 6 + row])
        for col in range(6):
            d[row, col] = mass_product(jp, jr, mass, inertia, edge, row, 0, edge, col, 0, particle)
            d[row, col] += mass_product(jp, jr, mass, inertia, edge, row, 1, edge, col, 1, particle + 1)
            if local + 1 < edges_per_rod:
                u[row, col] = mass_product(jp, jr, mass, inertia, edge, row, 1, edge + 1, col, 0, particle + 1)
        d[row, row] += alpha + wp.float64(1.0e-6)
    diagonal[edge] = d
    upper[edge] = u
    rhs[edge] = b


@wp.func
def cholesky(a: Mat6) -> Mat6:
    lower = Mat6()
    for i in range(6):
        for j in range(i + 1):
            v = a[i, j]
            for k in range(j):
                v -= lower[i, k] * lower[j, k]
            if i == j:
                lower[i, j] = wp.sqrt(v)
            else:
                lower[i, j] = v / lower[j, j]
    return lower


@wp.func
def triangular_solve(lower: Mat6, b: Vec6) -> Vec6:
    y = Vec6()
    x = Vec6()
    for i in range(6):
        v = b[i]
        for j in range(i):
            v -= lower[i, j] * y[j]
        y[i] = v / lower[i, i]
    for i in range(5, -1, -1):
        v = y[i]
        for j in range(i + 1, 6):
            v -= lower[j, i] * x[j]
        x[i] = v / lower[i, i]
    return x


@wp.kernel
def solve(
    diagonal: wp.array(dtype=Mat6),
    upper: wp.array(dtype=Mat6),
    rhs: wp.array(dtype=Vec6),
    edges_per_rod: int,
    factors: wp.array(dtype=Mat6),
    reduced: wp.array(dtype=Vec6),
    solution: wp.array(dtype=Vec6),
    delta: wp.array(dtype=float),
):
    start = wp.tid() * edges_per_rod
    for local in range(edges_per_rod):
        edge = start + local
        d = diagonal[edge]
        b = rhs[edge]
        if local > 0:
            lower = wp.transpose(upper[edge - 1])
            d -= lower * factors[edge - 1]
            b -= lower * reduced[edge - 1]
        # Roundoff may break symmetry during elimination; factor its symmetric part.
        d = (d + wp.transpose(d)) * wp.float64(0.5)
        chol = cholesky(d)
        c = Mat6()
        if local + 1 < edges_per_rod:
            for col in range(6):
                v = Vec6()
                for row in range(6):
                    v[row] = upper[edge][row, col]
                x = triangular_solve(chol, v)
                for row in range(6):
                    c[row, col] = x[row]
        factors[edge] = c
        reduced[edge] = triangular_solve(chol, b)
    for local in range(edges_per_rod - 1, -1, -1):
        edge = start + local
        x = reduced[edge]
        if local + 1 < edges_per_rod:
            x -= factors[edge] * solution[edge + 1]
        solution[edge] = x
        for row in range(6):
            delta[edge * 6 + row] = float(x[row])


class RodLinearSystem:
    """Reusable scratch; one independent block-tridiagonal solve per rod."""

    def __init__(self, edges_per_rod, num_envs, device):
        self.edges_per_rod = edges_per_rod
        self.num_envs = num_envs
        self.total_edges = edges_per_rod * num_envs
        self.device = device
        self.diagonal = wp.zeros(self.total_edges, dtype=Mat6, device=device)
        self.upper = wp.zeros(self.total_edges, dtype=Mat6, device=device)
        self.rhs = wp.zeros(self.total_edges, dtype=Vec6, device=device)
        self.factors = wp.zeros(self.total_edges, dtype=Mat6, device=device)
        self.reduced = wp.zeros(self.total_edges, dtype=Vec6, device=device)
        self.solution = wp.zeros(self.total_edges, dtype=Vec6, device=device)

    def project(self, workspace):
        wp.launch(
            assemble,
            dim=self.total_edges,
            inputs=[
                workspace.jacobian_pos,
                workspace.jacobian_rot,
                workspace.inv_masses,
                workspace.inv_inertia,
                workspace.compliance,
                workspace.constraint_values,
                workspace.lambda_sum,
                self.edges_per_rod,
                self.diagonal,
                self.upper,
                self.rhs,
            ],
            device=self.device,
        )
        wp.launch(
            solve,
            dim=self.num_envs,
            inputs=[
                self.diagonal,
                self.upper,
                self.rhs,
                self.edges_per_rod,
                self.factors,
                self.reduced,
                self.solution,
                workspace.delta_lambda,
            ],
            device=self.device,
        )
        return workspace.delta_lambda
