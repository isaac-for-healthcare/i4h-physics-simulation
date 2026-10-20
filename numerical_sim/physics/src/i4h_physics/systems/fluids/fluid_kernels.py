# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Position-based fluid kernels, ported from omnisurg/fluids/kernels/pbf.py.

Every kernel launches once for all pools with ``dim=(E, sum of pool counts)``: the second index is mapped
to a pool ``f`` and a body-local slot ``li`` through the prefix table (``pool_slot``), and ``base[f] + li``
addresses the span of the pool block ``scene.pool`` that holds the pools (``q_span`` and friends are slices of
that block's arrays over the span). Per-pool constants are ``(nf,)`` arrays; the neighbour list keeps omnisurg's
slot-major layout per pool (``nbr[nbr_base[f] + slot * (E * count[f]) + li]``) with body-local ids, filled
once per substep from a grouped ``wp.HashGrid`` query with group ``env * nf + f`` (dead particles carry
group -1 and are never visited), so the per-particle overflow flags and query diagnostics become one
overflow counter per env and pool. ``newton._src`` ``ParticleFlags`` -> scene ``ACTIVE``; ``inv_mass`` is
constant and positive on a pool, so its tests are dropped; the surface-tension branch
(``cohesion_model == 1``) and the neighbour-weight averaging are not ported (unused by every case).
"""

from __future__ import annotations

import warp as wp
from i4h_physics.scene import ACTIVE

_PI = 3.141592653589793


# omnisurg/fluids/kernels/pbf.py:11-66 and 199-209, verbatim.
@wp.func
def _poly6_weight(distance: wp.float32, radius: wp.float32) -> wp.float32:
    if distance >= radius:
        return 0.0
    radius_sq = radius * radius
    value = radius_sq - distance * distance
    coefficient = 315.0 / (_PI * 64.0 * radius_sq * radius_sq * radius_sq * radius_sq * radius)
    return coefficient * value * value * value


@wp.func
def _spiky_derivative(distance: wp.float32, radius: wp.float32) -> wp.float32:
    coefficient = -45.0 / (_PI * radius * radius * radius * radius * radius * radius)
    return coefficient * (radius - distance) * (radius - distance)


@wp.func
def _gradient(offset: wp.vec3, distance: wp.float32, radius: wp.float32) -> wp.vec3:
    # Desbrun spiky gradient; the FleX spiky density kernel and gradient (use_spiky_kernels) are not carried
    if distance <= 1.0e-8 or distance >= radius:
        return wp.vec3(0.0, 0.0, 0.0)
    return _spiky_derivative(distance, radius) * (offset / distance)


@wp.func
def _cohesion_weight(distance: wp.float32, radius: wp.float32, rest_distance: wp.float32) -> wp.float32:
    rest = wp.max(rest_distance / radius, 1.0e-6)
    x = distance / radius
    c1 = -(1.0 + rest) / (rest * rest)
    c2 = (rest * rest + rest + 1.0) / (rest * rest)
    return c1 * x * x * x + c2 * x * x - 1.0


@wp.func
def pool_slot(local: wp.int32, prefix: wp.array(dtype=wp.int32), nf: wp.int32) -> wp.vec2i:
    """(pool, body-local slot) of the launch's second index: pools are laid out one after another."""
    f = wp.int32(0)
    while f + 1 < nf and local >= prefix[f + 1]:
        f += 1
    return wp.vec2i(f, local - prefix[f])


# particles.py:57-84 (predict) for the pools, with the grid group of the particle written alongside:
# ``env * nf + f`` while alive, -1 (never queried) while dead.
@wp.kernel
def predict_fluid(
    qd: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    flags: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    damping: wp.array(dtype=wp.float32),
    max_velocity: wp.array(dtype=wp.float32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nf: wp.int32,
    dt: wp.float32,
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    group: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    i = base[f] + env * counts[f] + ps[1]
    q0 = q[i]
    q_prev[i] = q0
    if (flags[i] & ACTIVE) == 0 or inv_mass[i] <= 0.0:
        group[i] = -1
        return
    group[i] = env * nf + f
    v1 = (qd[i] + gravity[env] * dt) * damping[env]
    v_max = max_velocity[env]
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 = v1 * (v_max / v1_mag)
    q[i] = q0 + v1 * dt


# omnisurg/fluids/kernels/pbf.py:626-676 (build_pbf_neighbor_list_range): the grouped query replaces the alive and
# range tests on a candidate; body-local ids are stored; overflow is counted per env and pool.
@wp.kernel
def neighbors(
    grid: wp.uint64,
    q: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nbr_base: wp.array(dtype=wp.int32),
    max_neighbors: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=wp.float32),
    envs: wp.int32,
    nf: wp.int32,
    nbr: wp.array(dtype=wp.int32),
    nbr_count: wp.array(dtype=wp.int32),
    overflow: wp.array2d(dtype=wp.int32),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    li = env * counts[f] + ps[1]
    i = base[f] + li
    if (flags[i] & ACTIVE) == 0:
        nbr_count[i] = 0
        return
    n_total = envs * counts[f]
    h = radius[f]
    cap = max_neighbors[f]
    position = q[i]
    n = int(0)
    overflowed = int(0)
    query = wp.hash_grid_query(grid, position, h, env * nf + f)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if wp.length(position - q[j]) < h:
            if n >= cap:
                overflowed = 1
                break
            nbr[nbr_base[f] + n * n_total + li] = j - base[f]
            n += 1
    nbr_count[i] = n
    if overflowed != 0:
        wp.atomic_add(overflow, env, f, 1)


# omnisurg/fluids/kernels/pbf.py:761-822 (compute_pbf_lambdas_range) without the surface-tension normal; ``due_mask``
# bit f is set when pool f solves on this iteration (Python's due() at capture time).
@wp.kernel
def lambdas(
    q: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
    nbr: wp.array(dtype=wp.int32),
    nbr_count: wp.array(dtype=wp.int32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nbr_base: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=wp.float32),
    rest_density: wp.array(dtype=wp.float32),
    density_epsilon: wp.array(dtype=wp.float32),
    envs: wp.int32,
    nf: wp.int32,
    due_mask: wp.int32,
    density: wp.array(dtype=wp.float32),
    lam: wp.array(dtype=wp.float32),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    if ((due_mask >> f) & 1) == 0:
        return
    li = env * counts[f] + ps[1]
    i = base[f] + li
    if (flags[i] & ACTIVE) == 0:
        density[i] = 0.0
        lam[i] = 0.0
        return
    n_total = envs * counts[f]
    h = radius[f]
    rho0 = rest_density[f]
    position = q[i]
    rho = wp.float32(0.0)
    grad_i = wp.vec3(0.0, 0.0, 0.0)
    sum_grad_sq = wp.float32(0.0)
    n = nbr_count[i]
    slot = int(0)
    while slot < n:
        j = base[f] + nbr[nbr_base[f] + slot * n_total + li]
        offset = position - q[j]
        distance = wp.length(offset)
        rho += _poly6_weight(distance, h)
        if j != i and distance > 1.0e-8:
            grad_j = _gradient(offset, distance, h) / rho0
            grad_i += grad_j
            sum_grad_sq += wp.dot(grad_j, grad_j)
        slot += 1
    sum_grad_sq += wp.dot(grad_i, grad_i)
    normalized_density = rho / rho0
    constraint = wp.max(normalized_density - 1.0, 0.0)
    density[i] = normalized_density
    lam[i] = -constraint / (sum_grad_sq + density_epsilon[f])


# omnisurg/fluids/kernels/pbf.py:825-879 (solve_pbf_position_deltas_range) without the surface-tension branch and
# the averaging weights; cohesion per env and pool.
@wp.kernel
def deltas(
    q: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    nbr: wp.array(dtype=wp.int32),
    nbr_count: wp.array(dtype=wp.int32),
    cohesion: wp.array2d(dtype=wp.float32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nbr_base: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=wp.float32),
    rest_density: wp.array(dtype=wp.float32),
    rest_distance: wp.array(dtype=wp.float32),
    envs: wp.int32,
    nf: wp.int32,
    due_mask: wp.int32,
    delta: wp.array(dtype=wp.vec3),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    if ((due_mask >> f) & 1) == 0:
        return
    li = env * counts[f] + ps[1]
    i = base[f] + li
    if (flags[i] & ACTIVE) == 0:
        delta[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    n_total = envs * counts[f]
    h = radius[f]
    rho0 = rest_density[f]
    position = q[i]
    lambda_i = lam[i]
    coh = cohesion[env, f]
    d = wp.vec3(0.0, 0.0, 0.0)
    n = nbr_count[i]
    slot = int(0)
    while slot < n:
        j = base[f] + nbr[nbr_base[f] + slot * n_total + li]
        if j != i:
            offset = position - q[j]
            distance = wp.length(offset)
            if distance > 1.0e-8:
                gradient = _gradient(offset, distance, h) / rho0
                d += (lambda_i + lam[j]) * gradient
                if coh > 0.0:
                    direction = offset / distance
                    weight = _cohesion_weight(distance, h, rest_distance[f])
                    d -= coh * h * weight * direction
        slot += 1
    delta[i] = d


# omnisurg/fluids/kernels/pbf.py:882-902 (apply_pbf_deltas_range), relaxation per env and pool, no averaging.
@wp.kernel
def apply(
    flags: wp.array(dtype=wp.int32),
    delta: wp.array(dtype=wp.vec3),
    relaxation: wp.array2d(dtype=wp.float32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nf: wp.int32,
    due_mask: wp.int32,
    q: wp.array(dtype=wp.vec3),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    if ((due_mask >> f) & 1) == 0:
        return
    i = base[f] + env * counts[f] + ps[1]
    if (flags[i] & ACTIVE) == 0:
        return
    q[i] = q[i] + delta[i] * relaxation[env, f]  # ``deltas`` rewrites every slot before the next read


# particles.py:91-115 (finalize) for the pools, writing the velocity into the XSPH scratch instead of ``scene.qd``
# (``xsph`` writes ``scene.qd``), which saves a copy per substep.
@wp.kernel
def finalize_fluid(
    q_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    flags: wp.array(dtype=wp.int32),
    max_velocity: wp.array(dtype=wp.float32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nf: wp.int32,
    dt: wp.float32,
    q: wp.array(dtype=wp.vec3),
    qd_scratch: wp.array(dtype=wp.vec3),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    i = base[f] + env * counts[f] + ps[1]
    if (flags[i] & ACTIVE) == 0 or inv_mass[i] <= 0.0:
        qd_scratch[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    displacement = q[i] - q_prev[i]
    distance = wp.length(displacement)
    max_delta = max_velocity[env] * dt
    if max_delta > 0.0 and distance > max_delta:
        displacement = displacement * (max_delta / distance)
        q[i] = q_prev[i] + displacement
    qd_scratch[i] = displacement / dt


# omnisurg/fluids/kernels/pbf.py:1043-1087 (apply_xsph_viscosity_range); reads the finalized velocities from the
# scratch and writes ``scene.qd`` for every pool slot.
@wp.kernel
def xsph(
    q: wp.array(dtype=wp.vec3),
    qd_in: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
    nbr: wp.array(dtype=wp.int32),
    nbr_count: wp.array(dtype=wp.int32),
    viscosity: wp.array2d(dtype=wp.float32),
    max_velocity: wp.array(dtype=wp.float32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nbr_base: wp.array(dtype=wp.int32),
    radius: wp.array(dtype=wp.float32),
    rest_density: wp.array(dtype=wp.float32),
    envs: wp.int32,
    nf: wp.int32,
    qd: wp.array(dtype=wp.vec3),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    li = env * counts[f] + ps[1]
    i = base[f] + li
    if (flags[i] & ACTIVE) == 0:
        qd[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    n_total = envs * counts[f]
    h = radius[f]
    position = q[i]
    velocity = qd_in[i]
    d = wp.vec3(0.0, 0.0, 0.0)
    n = nbr_count[i]
    slot = int(0)
    while slot < n:
        j = base[f] + nbr[nbr_base[f] + slot * n_total + li]
        if j != i:
            distance = wp.length(position - q[j])
            d += (qd_in[j] - velocity) * _poly6_weight(distance, h)
        slot += 1
    velocity = velocity + viscosity[env, f] * d / rest_density[f]
    v_max = max_velocity[env]
    velocity_mag = wp.length(velocity)
    if velocity_mag > v_max:
        velocity = velocity * (v_max / velocity_mag)
    qd[i] = velocity


@wp.kernel
def count_alive(
    flags: wp.array(dtype=wp.int32),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nf: wp.int32,
    alive: wp.array2d(dtype=wp.int32),
):
    env, local = wp.tid()
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    if (flags[base[f] + env * counts[f] + ps[1]] & ACTIVE) != 0:
        wp.atomic_add(alive, env, f, 1)
