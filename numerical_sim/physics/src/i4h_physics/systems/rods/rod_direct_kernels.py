# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the direct (block-Thomas) XPBD Cosserat rod, ported from omniendo/xpbd_rod.

Sources: kernels_math.py (quaternion, 3x3 and 6x6-block helpers), kernels_integration.py
(rotation predict and integrate), kernels_constraints.py (compliance, constraint values,
Jacobians, right-hand side), kernels_assembly.py (world inverse inertia, block assembly),
kernels_solvers.py (batched block Thomas) and kernels_collision.py (correction accumulation
and application). Every kernel launches with ``dim=(E, n)``; the prologue is
``env, e = wp.tid()`` and omniendo's ``rod_offsets`` / ``edge_offsets`` / ``particle_rod_id`` /
``edge_rod_id`` lookups become arithmetic on the fixed node count ``P`` and edge count ``Ed``:
node ``l = env * P + e`` (body-local; ``(env, e)`` in the scene arrays' body views), edge ``g = env * Ed + e``,
dof ``g * 6 + k``, block ``g * 36 + row * 6 + col``. Per-variant rest data is read as ``[variant[env], e]``.
Inactive nodes (``scene.flags`` without ``ACTIVE``) keep their frame and position: an edge with an inactive
node has zero constraint values and Jacobians (so it drops out of the block system like a padding tet
drops out of the tet solve), and the apply skips the node.
Dropped: ``torques``, ``prev_orientations``, the ``_delta_lambda_max`` / ``_correction_max``
diagnostics, the single-rod kernel variants and the split/Jacobi/banded/tiled solver paths.

Two of omniendo's arrays are constants and are not kept: the position Jacobian is ``[[I, -I], [0, 0]]``
for every edge (``kernels_constraints.py:425-429`` writes nothing else), so its products with the
inverse masses are written out as ``inv_m0 + inv_m1`` / ``-inv_m0``, bitwise what the loops summed; and
the world inverse inertia ``R diag(1, 1, 1) R^T`` of a unit local inertia (the only value omniendo ever
set, ``solver_xpbd_rod.py:196``) is the identity for a free node and zero for a locked one, so it is the
scalar ``1`` or ``0`` (a 6e-7 change from the rounded matrix product). ``edge_w`` carries an edge's two
inverse masses and its live flag from the constraint kernel to the assembly and correction kernels.
The multipliers' substep boundary is the ``first`` iteration flag (fixed at capture): the right-hand
side skips the ``compliance * lambda_sum`` term and the corrections overwrite ``lambda_sum`` instead
of accumulating, which is ``-c - 0`` and ``0 + dl`` exactly, so no memset is needed.
"""

from __future__ import annotations

import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.rods.rod_pbd import qconj, qmul

# --- kernels_math.py:31-68 quaternion helpers. Edits: the product and the conjugate are rod_pbd's qmul/qconj. ------


@wp.func
def _warp_quat_normalize(q: wp.quat) -> wp.quat:
    """Normalize a quaternion to unit length."""
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1.0e-8:
        return q
    inv = 1.0 / norm
    return wp.quat(q.x * inv, q.y * inv, q.z * inv, q.w * inv)


@wp.func
def _warp_quat_rotate_vector(q: wp.quat, v: wp.vec3) -> wp.vec3:
    """Rotate a vector by a quaternion."""
    tx = 2.0 * (q.y * v.z - q.z * v.y)
    ty = 2.0 * (q.z * v.x - q.x * v.z)
    tz = 2.0 * (q.x * v.y - q.y * v.x)
    return wp.vec3(
        v.x + q.w * tx + q.y * tz - q.z * ty,
        v.y + q.w * ty + q.z * tx - q.x * tz,
        v.z + q.w * tz + q.x * ty - q.y * tx,
    )


# --- kernels_math.py:76-79 Jacobian indexing. Edits: none. ---------------------------------------


@wp.func
def _warp_jacobian_index(edge: int, row: int, col: int) -> int:
    """Compute flat index for Jacobian storage (6x6 per edge)."""
    return edge * 36 + row * 6 + col


# --- kernels_math.py:87-196 3x3 matrix operations. Edits: none. -----------------------------------


@wp.func
def _mat33_add(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    """Add two 3x3 matrices."""
    return wp.mat33(
        a[0, 0] + b[0, 0], a[0, 1] + b[0, 1], a[0, 2] + b[0, 2],
        a[1, 0] + b[1, 0], a[1, 1] + b[1, 1], a[1, 2] + b[1, 2],
        a[2, 0] + b[2, 0], a[2, 1] + b[2, 1], a[2, 2] + b[2, 2],
    )


@wp.func
def _mat33_sub(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    """Subtract two 3x3 matrices."""
    return wp.mat33(
        a[0, 0] - b[0, 0], a[0, 1] - b[0, 1], a[0, 2] - b[0, 2],
        a[1, 0] - b[1, 0], a[1, 1] - b[1, 1], a[1, 2] - b[1, 2],
        a[2, 0] - b[2, 0], a[2, 1] - b[2, 1], a[2, 2] - b[2, 2],
    )


@wp.func
def _mat33_mul(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    """Multiply two 3x3 matrices."""
    return wp.mat33(
        a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0],
        a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1],
        a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2],
        a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0],
        a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1],
        a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2],
        a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0],
        a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1],
        a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2],
    )


@wp.func
def _mat33_mul_vec3(a: wp.mat33, v: wp.vec3) -> wp.vec3:
    """Multiply a 3x3 matrix by a 3D vector."""
    return wp.vec3(
        a[0, 0] * v[0] + a[0, 1] * v[1] + a[0, 2] * v[2],
        a[1, 0] * v[0] + a[1, 1] * v[1] + a[1, 2] * v[2],
        a[2, 0] * v[0] + a[2, 1] * v[1] + a[2, 2] * v[2],
    )


@wp.func
def _mat33_transpose(a: wp.mat33) -> wp.mat33:
    """Transpose a 3x3 matrix."""
    return wp.mat33(a[0, 0], a[1, 0], a[2, 0], a[0, 1], a[1, 1], a[2, 1], a[0, 2], a[1, 2], a[2, 2])


@wp.func
def _mat33_cholesky(a: wp.mat33) -> wp.mat33:
    """Compute the Cholesky decomposition of a 3x3 SPD matrix."""
    eps = 1.0e-9
    l00 = wp.sqrt(wp.max(a[0, 0], eps))
    l10 = a[1, 0] / l00
    l20 = a[2, 0] / l00
    l11 = wp.sqrt(wp.max(a[1, 1] - l10 * l10, eps))
    l21 = (a[2, 1] - l20 * l10) / l11
    l22 = wp.sqrt(wp.max(a[2, 2] - l20 * l20 - l21 * l21, eps))
    return wp.mat33(l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22)


@wp.func
def _mat33_solve_lower(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    """Solve L * x = b where L is lower triangular."""
    y0 = b[0] / L[0, 0]
    y1 = (b[1] - L[1, 0] * y0) / L[1, 1]
    y2 = (b[2] - L[2, 0] * y0 - L[2, 1] * y1) / L[2, 2]
    return wp.vec3(y0, y1, y2)


@wp.func
def _mat33_solve_upper(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    """Solve L^T * x = b where L is lower triangular."""
    x2 = b[2] / L[2, 2]
    x1 = (b[1] - L[2, 1] * x2) / L[1, 1]
    x0 = (b[0] - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]
    return wp.vec3(x0, x1, x2)


# --- kernels_math.py:290-510 6x6 blocks as four 3x3 matrices. Edits: the *_offset wrappers (each the
# only caller of its plain variant) are folded into the plain functions, which take the absolute block;
# _block_index (identical to _warp_jacobian_index) is dropped; _block_solve no longer takes the unused B.


@wp.func
def _load_block(blocks: wp.array(dtype=wp.float32), block: int) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Load a 6x6 block as four 3x3 matrices."""
    base = block * 36
    A = wp.mat33(blocks[base + 0], blocks[base + 1], blocks[base + 2], blocks[base + 6], blocks[base + 7], blocks[base + 8],
                 blocks[base + 12], blocks[base + 13], blocks[base + 14])
    B = wp.mat33(blocks[base + 3], blocks[base + 4], blocks[base + 5], blocks[base + 9], blocks[base + 10], blocks[base + 11],
                 blocks[base + 15], blocks[base + 16], blocks[base + 17])
    C = wp.mat33(blocks[base + 18], blocks[base + 19], blocks[base + 20], blocks[base + 24], blocks[base + 25], blocks[base + 26],
                 blocks[base + 30], blocks[base + 31], blocks[base + 32])
    D = wp.mat33(blocks[base + 21], blocks[base + 22], blocks[base + 23], blocks[base + 27], blocks[base + 28], blocks[base + 29],
                 blocks[base + 33], blocks[base + 34], blocks[base + 35])
    return A, B, C, D


@wp.func
def _load_vec(values: wp.array(dtype=wp.float32), block: int) -> tuple[wp.vec3, wp.vec3]:
    """Load a 6-vector as two 3-vectors."""
    base = block * 6
    v0 = wp.vec3(values[base + 0], values[base + 1], values[base + 2])
    v1 = wp.vec3(values[base + 3], values[base + 4], values[base + 5])
    return v0, v1


@wp.func
def _store_vec(values: wp.array(dtype=wp.float32), block: int, v0: wp.vec3, v1: wp.vec3):
    """Store two 3-vectors as a 6-vector."""
    base = block * 6
    values[base + 0] = v0[0]
    values[base + 1] = v0[1]
    values[base + 2] = v0[2]
    values[base + 3] = v1[0]
    values[base + 4] = v1[1]
    values[base + 5] = v1[2]


@wp.func
def _block_column(blocks: wp.array(dtype=wp.float32), block: int, col: int) -> tuple[wp.vec3, wp.vec3]:
    """Load a column from a 6x6 block as two 3-vectors."""
    base = block * 36
    v0 = wp.vec3(blocks[base + 0 * 6 + col], blocks[base + 1 * 6 + col], blocks[base + 2 * 6 + col])
    v1 = wp.vec3(blocks[base + 3 * 6 + col], blocks[base + 4 * 6 + col], blocks[base + 5 * 6 + col])
    return v0, v1


@wp.func
def _block_set_column(blocks: wp.array(dtype=wp.float32), block: int, col: int, v0: wp.vec3, v1: wp.vec3):
    """Store two 3-vectors as a column in a 6x6 block."""
    base = block * 36
    blocks[base + 0 * 6 + col] = v0[0]
    blocks[base + 1 * 6 + col] = v0[1]
    blocks[base + 2 * 6 + col] = v0[2]
    blocks[base + 3 * 6 + col] = v1[0]
    blocks[base + 4 * 6 + col] = v1[1]
    blocks[base + 5 * 6 + col] = v1[2]


@wp.func
def _block_mul(
    A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33, E: wp.mat33, F: wp.mat33, G: wp.mat33, H: wp.mat33
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Multiply two 2x2 block matrices (each block is 3x3)."""
    return (
        _mat33_add(_mat33_mul(A, E), _mat33_mul(B, G)),
        _mat33_add(_mat33_mul(A, F), _mat33_mul(B, H)),
        _mat33_add(_mat33_mul(C, E), _mat33_mul(D, G)),
        _mat33_add(_mat33_mul(C, F), _mat33_mul(D, H)),
    )


@wp.func
def _block_sub(
    A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33, E: wp.mat33, F: wp.mat33, G: wp.mat33, H: wp.mat33
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """Subtract two 2x2 block matrices (each block is 3x3)."""
    return (_mat33_sub(A, E), _mat33_sub(B, F), _mat33_sub(C, G), _mat33_sub(D, H))


@wp.func
def _block_mul_vec(A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33, v0: wp.vec3, v1: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """Multiply a 2x2 block matrix by a 2-block vector."""
    top = _mat33_mul_vec3(A, v0) + _mat33_mul_vec3(B, v1)
    bot = _mat33_mul_vec3(C, v0) + _mat33_mul_vec3(D, v1)
    return top, bot


@wp.func
def _block_factor(A: wp.mat33, C: wp.mat33, D: wp.mat33) -> tuple[wp.mat33, wp.mat33, wp.mat33]:
    """Block Cholesky factors ``(L11, L21, L22)`` of the 6x6 SPD block ``[[A, C^T], [C, D]]`` (the first half of
    kernels_solver.py's block solve), shared by the seven solves against one Thomas pivot."""
    L11 = _mat33_cholesky(A)
    c0 = wp.vec3(C[0, 0], C[0, 1], C[0, 2])
    c1 = wp.vec3(C[1, 0], C[1, 1], C[1, 2])
    c2 = wp.vec3(C[2, 0], C[2, 1], C[2, 2])
    y0 = _mat33_solve_lower(L11, c0)
    y1 = _mat33_solve_lower(L11, c1)
    y2 = _mat33_solve_lower(L11, c2)
    L21 = wp.mat33(y0[0], y0[1], y0[2], y1[0], y1[1], y1[2], y2[0], y2[1], y2[2])
    S = _mat33_sub(D, _mat33_mul(L21, _mat33_transpose(L21)))
    L22 = _mat33_cholesky(S)
    return L11, L21, L22


@wp.func
def _block_solve_factored(L11: wp.mat33, L21: wp.mat33, L22: wp.mat33, b0: wp.vec3, b1: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """Solve with the factors of ``_block_factor`` (the second half of the block solve; same operations, same order)."""
    yb0 = _mat33_solve_lower(L11, b0)
    tmp = b1 - _mat33_mul_vec3(L21, yb0)
    yb1 = _mat33_solve_lower(L22, tmp)
    x1 = _mat33_solve_upper(L22, yb1)
    x0 = _mat33_solve_upper(L11, yb0 - _mat33_mul_vec3(_mat33_transpose(L21), x1))
    return x0, x1


# --- kernels_collision.py:159-197 correction helpers. Edits: none. --------------------------------


@wp.func
def _warp_quat_correction_g(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    """Apply a small angular correction to a quaternion."""
    norm_sq = dtheta.x * dtheta.x + dtheta.y * dtheta.y + dtheta.z * dtheta.z
    if norm_sq < 1.0e-20:
        return q
    corr_x = 0.5 * (q.w * dtheta.x + q.z * dtheta.y - q.y * dtheta.z)
    corr_y = 0.5 * (-q.z * dtheta.x + q.w * dtheta.y + q.x * dtheta.z)
    corr_z = 0.5 * (q.y * dtheta.x - q.x * dtheta.y + q.w * dtheta.z)
    corr_w = 0.5 * (-q.x * dtheta.x - q.y * dtheta.y - q.z * dtheta.z)
    q_new = wp.quat(q.x + corr_x, q.y + corr_y, q.z + corr_z, q.w + corr_w)
    return _warp_quat_normalize(q_new)


@wp.func
def _warp_jacobian_dot(
    jacobian: wp.array(dtype=wp.float32), edge: int, col: int,
    dl0: float, dl1: float, dl2: float, dl3: float, dl4: float, dl5: float,
) -> float:
    """Compute dot product of Jacobian column with delta lambda."""
    return (
        jacobian[_warp_jacobian_index(edge, 0, col)] * dl0
        + jacobian[_warp_jacobian_index(edge, 1, col)] * dl1
        + jacobian[_warp_jacobian_index(edge, 2, col)] * dl2
        + jacobian[_warp_jacobian_index(edge, 3, col)] * dl3
        + jacobian[_warp_jacobian_index(edge, 4, col)] * dl4
        + jacobian[_warp_jacobian_index(edge, 5, col)] * dl5
    )


# --- kernels_integration.py:238-282 (_warp_predict_rotations_batched). -----------------------------
# Edits: (env, node) prologue; torques dropped; damping[env] instead of damping[particle_rod_id];
# inverse rotational mass read per variant; inactive nodes are treated as locked; omega is read and
# written by its own thread (output).
@wp.kernel
def predict_rot(
    frame: wp.array(dtype=wp.quat),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    damping: wp.array(dtype=wp.float32),
    P: wp.int32,
    dt: wp.float32,
    omega: wp.array(dtype=wp.vec3),
    frame_pred: wp.array(dtype=wp.quat),
):
    env, e = wp.tid()
    i = env * P + e
    inv_mass = inv_mass_q[variant[env], e]
    if inv_mass > 0.0 and (flags[env, e] & ACTIVE) != 0:
        half_dt = 0.5 * dt
        w = omega[i]
        w = w * (1.0 - damping[env])
        omega[i] = w
        q = frame[i]
        omega_q = wp.quat(w.x, w.y, w.z, 0.0)
        qdot = qmul(omega_q, q)
        q_pred = wp.quat(q.x + qdot.x * half_dt, q.y + qdot.y * half_dt, q.z + qdot.z * half_dt, q.w + qdot.w * half_dt)
        frame_pred[i] = _warp_quat_normalize(q_pred)
    else:
        omega[i] = wp.vec3(0.0, 0.0, 0.0)
        frame_pred[i] = frame[i]


# --- kernels_integration.py:283-309 (_warp_integrate_rotations_batched). ---------------------------
# Edits: (env, node) prologue; prev_orientations dropped (never read); variant lookup; inactive nodes skipped.
@wp.kernel
def integrate_rot(
    frame_pred: wp.array(dtype=wp.quat),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    P: wp.int32,
    dt: wp.float32,
    frame: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
):
    env, e = wp.tid()
    i = env * P + e
    if inv_mass_q[variant[env], e] > 0.0 and (flags[env, e] & ACTIVE) != 0:
        q = frame[i]
        rel = qmul(frame_pred[i], qconj(q))
        omega[i] = wp.vec3(rel.x, rel.y, rel.z) * (2.0 / dt)
        frame[i] = frame_pred[i]


# --- kernels_constraints.py:250-292 (_warp_prepare_compliance_batched). ---------------------------
# Edits: (env, edge) prologue; the per-edge bend_stiffness vec3 becomes the live (E,) bend and twist
# ((bend, bend, twist) per omniendo builder.py:244-247); young/torsion per env; rest length per variant.
@wp.kernel
def prepare_compliance(
    rest_len: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    young: wp.array(dtype=wp.float32),
    torsion: wp.array(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    twist: wp.array(dtype=wp.float32),
    Ed: wp.int32,
    dt: wp.float32,
    compliance: wp.array(dtype=wp.float32),
):
    env, e = wp.tid()
    i = env * Ed + e
    L = rest_len[variant[env], e]
    dt2 = dt * dt
    eps = 1.0e-10

    E = young[env]
    G = torsion[env]

    k_bend1_ref = E * bend[env]
    k_bend2_ref = E * bend[env]
    k_twist_ref = G * twist[env]

    k_bend1_eff = k_bend1_ref * L
    k_bend2_eff = k_bend2_ref * L
    k_twist_eff = k_twist_ref * L

    stretch_compliance = 1.0e-10

    base = i * 6
    compliance[base + 0] = stretch_compliance
    compliance[base + 1] = stretch_compliance
    compliance[base + 2] = stretch_compliance
    compliance[base + 3] = 1.0 / (k_bend1_eff * dt2 + eps)
    compliance[base + 4] = 1.0 / (k_bend2_eff * dt2 + eps)
    compliance[base + 5] = 1.0 / (k_twist_eff * dt2 + eps)


# --- kernels_constraints.py:308-360 (_warp_update_constraints_batched_v2), :369-487
# (_warp_compute_jacobians_batched) and :230-252 (_warp_build_rhs) fused: one edge thread computes the
# constraint values, the rotation Jacobian and the right-hand side from one load of the two frames.
# Edits: (env, edge) prologue; positions from the body's (E, P) view at (env, e) (the predicted
# positions of the substep); rest data per variant; the rest Darboux gets the live bend_offset added
# (omniendo's _apply_tip_controls wrote the sum into rest_darboux instead); an edge with an inactive
# node has zero rhs, zero Jacobian and zero edge_w, so it decouples from the block system (its diagonal
# block is the compliance plus the regularisation) while the Thomas sweep still visits it; the constant
# position Jacobian is not stored (see the module docstring); rhs skips the multiplier term on the
# substep's first iteration, where omniendo had just zeroed lambda_sum.
@wp.kernel
def edge_constraints(
    q: wp.array2d(dtype=wp.vec3),
    frame_pred: wp.array(dtype=wp.quat),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    rest_len: wp.array2d(dtype=wp.float32),
    rest_darboux: wp.array2d(dtype=wp.vec3),
    preferred_rest: wp.array2d(dtype=wp.vec3),
    regional: int,
    bend_offset: wp.array2d(dtype=wp.vec3),
    variant: wp.array(dtype=wp.int32),
    compliance: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    first: wp.int32,
    P: wp.int32,
    Ed: wp.int32,
    edge_w: wp.array(dtype=wp.vec3),
    jacobian_rot: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
):
    env, e = wp.tid()
    v = variant[env]
    i = env * Ed + e
    p0_idx = env * P + e
    p1_idx = p0_idx + 1
    base = i * 6
    if (flags[env, e] & ACTIVE) == 0 or (flags[env, e + 1] & ACTIVE) == 0:
        edge_w[i] = wp.vec3(0.0, 0.0, 0.0)
        for k in range(36):
            jacobian_rot[i * 36 + k] = 0.0
        for k in range(6):
            rhs[base + k] = 0.0
        return
    edge_w[i] = wp.vec3(inv_mass[env, e], inv_mass[env, e + 1], 1.0)

    p0 = q[env, e]
    p1 = q[env, e + 1]
    q0 = frame_pred[p0_idx]
    q1 = frame_pred[p1_idx]

    half_L = 0.5 * rest_len[v, e]
    r0 = _warp_quat_rotate_vector(q0, wp.vec3(0.0, 0.0, half_L))
    r1 = _warp_quat_rotate_vector(q1, wp.vec3(0.0, 0.0, -half_L))

    # constraint values (_warp_update_constraints_batched_v2)
    c0 = p0 + r0
    c1 = p1 + r1
    stretch_error = c0 - c1

    q_rel = qmul(qconj(q0), q1)
    omega = wp.vec3(q_rel.x, q_rel.y, q_rel.z)
    target = rest_darboux[v, e]
    if regional != 0:
        target = preferred_rest[env, e]
    darboux_error = omega - (target + bend_offset[env, e])

    # right-hand side (_warp_build_rhs): rhs = -constraint - compliance * lambda_sum
    if first != 0:
        rhs[base + 0] = -stretch_error.x
        rhs[base + 1] = -stretch_error.y
        rhs[base + 2] = -stretch_error.z
        rhs[base + 3] = -darboux_error.x
        rhs[base + 4] = -darboux_error.y
        rhs[base + 5] = -darboux_error.z
    else:
        rhs[base + 0] = -stretch_error.x - compliance[base + 0] * lambda_sum[base + 0]
        rhs[base + 1] = -stretch_error.y - compliance[base + 1] * lambda_sum[base + 1]
        rhs[base + 2] = -stretch_error.z - compliance[base + 2] * lambda_sum[base + 2]
        rhs[base + 3] = -darboux_error.x - compliance[base + 3] * lambda_sum[base + 3]
        rhs[base + 4] = -darboux_error.y - compliance[base + 4] * lambda_sum[base + 4]
        rhs[base + 5] = -darboux_error.z - compliance[base + 5] * lambda_sum[base + 5]

    # rotation Jacobian (_warp_compute_jacobians_batched): -skew(r0) and skew(r1) for the stretch rows
    jacobian_rot[_warp_jacobian_index(i, 0, 0)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 1)] = r0.z
    jacobian_rot[_warp_jacobian_index(i, 0, 2)] = -r0.y

    jacobian_rot[_warp_jacobian_index(i, 1, 0)] = -r0.z
    jacobian_rot[_warp_jacobian_index(i, 1, 1)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 2)] = r0.x

    jacobian_rot[_warp_jacobian_index(i, 2, 0)] = r0.y
    jacobian_rot[_warp_jacobian_index(i, 2, 1)] = -r0.x
    jacobian_rot[_warp_jacobian_index(i, 2, 2)] = 0.0

    jacobian_rot[_warp_jacobian_index(i, 0, 3)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 4)] = -r1.z
    jacobian_rot[_warp_jacobian_index(i, 0, 5)] = r1.y

    jacobian_rot[_warp_jacobian_index(i, 1, 3)] = r1.z
    jacobian_rot[_warp_jacobian_index(i, 1, 4)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 5)] = -r1.x

    jacobian_rot[_warp_jacobian_index(i, 2, 3)] = -r1.y
    jacobian_rot[_warp_jacobian_index(i, 2, 4)] = r1.x
    jacobian_rot[_warp_jacobian_index(i, 2, 5)] = 0.0

    x0 = q0.x
    y0 = q0.y
    z0 = q0.z
    w0 = q0.w
    x1 = q1.x
    y1 = q1.y
    z1 = q1.z
    w1 = q1.w

    j0_r0 = wp.vec4(-w1, -z1, y1, x1)
    j0_r1 = wp.vec4(z1, -w1, -x1, y1)
    j0_r2 = wp.vec4(-y1, x1, -w1, z1)

    g0_c0 = wp.vec4(0.5 * w0, -0.5 * z0, 0.5 * y0, -0.5 * x0)
    g0_c1 = wp.vec4(0.5 * z0, 0.5 * w0, -0.5 * x0, -0.5 * y0)
    g0_c2 = wp.vec4(-0.5 * y0, 0.5 * x0, 0.5 * w0, -0.5 * z0)

    j1_r0 = wp.vec4(w0, z0, -y0, -x0)
    j1_r1 = wp.vec4(-z0, w0, x0, -y0)
    j1_r2 = wp.vec4(y0, -x0, w0, -z0)

    g1_c0 = wp.vec4(0.5 * w1, -0.5 * z1, 0.5 * y1, -0.5 * x1)
    g1_c1 = wp.vec4(0.5 * z1, 0.5 * w1, -0.5 * x1, -0.5 * y1)
    g1_c2 = wp.vec4(-0.5 * y1, 0.5 * x1, 0.5 * w1, -0.5 * z1)

    jacobian_rot[_warp_jacobian_index(i, 3, 0)] = wp.dot(j0_r0, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 1)] = wp.dot(j0_r0, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 2)] = wp.dot(j0_r0, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 0)] = wp.dot(j0_r1, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 1)] = wp.dot(j0_r1, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 2)] = wp.dot(j0_r1, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 0)] = wp.dot(j0_r2, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 1)] = wp.dot(j0_r2, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 2)] = wp.dot(j0_r2, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 3, 3)] = wp.dot(j1_r0, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 4)] = wp.dot(j1_r0, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 5)] = wp.dot(j1_r0, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 3)] = wp.dot(j1_r1, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 4)] = wp.dot(j1_r1, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 5)] = wp.dot(j1_r1, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 3)] = wp.dot(j1_r2, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 4)] = wp.dot(j1_r2, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 5)] = wp.dot(j1_r2, g1_c2)


# --- kernels_assembly.py:419-551 (_warp_assemble_jmjt_blocks_batched). -----------------------------
# Edits: (env, edge) prologue; the first edge of a rod is local edge 0 and the previous edge is i - 1;
# the position Jacobian products are written out (J_pos = [I, -I]: inv_m0 + inv_m1 on the stretch
# diagonal, -inv_m0 on the stretch off-diagonal, both taken from edge_w so a dead edge contributes 0);
# the world inverse inertia is the scalar 1 (free node) or 0 (locked node) instead of R diag(1) R^T, unless
# ``physical``: then each node's R diag(inv, inv, polar) R^T from ``world_inv_inertia``.
@wp.func
def inv_inertia_times(physical: wp.int32, inv_inertia: wp.array(dtype=wp.mat33), node: wp.int32, unit: wp.float32,
                      x: wp.vec3):
    if physical != 0:
        return inv_inertia[node] * x
    return x * unit


@wp.kernel
def assemble_jmjt(
    edge_w: wp.array(dtype=wp.vec3),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    Ed: wp.int32,
    physical: wp.int32,
    inv_inertia: wp.array(dtype=wp.mat33),
    P: wp.int32,
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
):
    env, local_edge = wp.tid()
    v = variant[env]
    i = env * Ed + local_edge
    n0 = env * P + local_edge

    regularization = 1.0e-6
    w = edge_w[i]
    inv_m0 = w[0]
    inv_m1 = w[1]
    wq0 = wp.float32(0.0)  # world inverse inertia of node 0: identity (1) for a free node, zero for a locked one
    if inv_mass_q[v, local_edge] > 0.0:
        wq0 = 1.0
    wq1 = wp.float32(0.0)
    if inv_mass_q[v, local_edge + 1] > 0.0:
        wq1 = 1.0

    # Diagonal block
    for row in range(6):
        for col in range(6):
            val = 0.0

            # Position contributions of both nodes: J_pos = [I, -I], so only the stretch diagonal
            if row == col and row < 3:
                val += inv_m0
                val += inv_m1

            # Rotation contribution from segment 0
            j_t0_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 0)],
                jacobian_rot[_warp_jacobian_index(i, row, 1)],
                jacobian_rot[_warp_jacobian_index(i, row, 2)],
            )
            j_t0_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            val += wp.dot(j_t0_r_vec, inv_inertia_times(physical, inv_inertia, n0, wq0, j_t0_c_vec))

            # Rotation contribution from segment 1
            j_t1_r_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, row, 3)],
                jacobian_rot[_warp_jacobian_index(i, row, 4)],
                jacobian_rot[_warp_jacobian_index(i, row, 5)],
            )
            j_t1_c_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 3)],
                jacobian_rot[_warp_jacobian_index(i, col, 4)],
                jacobian_rot[_warp_jacobian_index(i, col, 5)],
            )
            val += wp.dot(j_t1_r_vec, inv_inertia_times(physical, inv_inertia, n0 + 1, wq1, j_t1_c_vec))

            if row == col:
                val += compliance[i * 6 + row] + regularization
            diag_blocks[_warp_jacobian_index(i, row, col)] = val

    # Off-diagonal block (coupling with previous edge within same rod)
    if local_edge == 0:
        # First edge in rod has no off-diagonal coupling
        for row in range(6):
            for col in range(6):
                offdiag_blocks[_warp_jacobian_index(i, row, col)] = 0.0
        return

    prev = i - 1
    both_live = w[2] != 0.0 and edge_w[prev][2] != 0.0

    for row in range(6):
        for col in range(6):
            val = 0.0

            # Position contribution through the shared node: J_pos of the previous edge's second node
            # (-I) times the current edge's first node (I), zero when either edge is dead
            if row == col and row < 3 and both_live:
                val += -inv_m0

            # Rotation contribution
            j_t1_prev_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(prev, row, 3)],
                jacobian_rot[_warp_jacobian_index(prev, row, 4)],
                jacobian_rot[_warp_jacobian_index(prev, row, 5)],
            )
            j_t0_cur_vec = wp.vec3(
                jacobian_rot[_warp_jacobian_index(i, col, 0)],
                jacobian_rot[_warp_jacobian_index(i, col, 1)],
                jacobian_rot[_warp_jacobian_index(i, col, 2)],
            )
            val += wp.dot(j_t1_prev_vec, inv_inertia_times(physical, inv_inertia, n0, wq0, j_t0_cur_vec))

            offdiag_blocks[_warp_jacobian_index(i, row, col)] = val


# --- kernels_solvers.py:173-274 (_warp_block_thomas_solve_batched). ---------------------------------
# Edits: one thread per env (dim=E); edge_offsets lookups become edge_start = env * Ed, n_edges = Ed;
# the n_rods and empty-rod guards are dropped; d' is written straight into x (the forward pass reads
# d'_{i-1} before the back substitution overwrites x_i, in the same thread), so d_prime is gone;
# dead edges (zero Jacobian, compliance-only diagonal) stay in the sweep so the recurrence holds.
@wp.kernel
def thomas(
    diag_blocks: wp.array(dtype=wp.float32),
    offdiag_blocks: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    Ed: wp.int32,
    c_blocks: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """Solve block-tridiagonal systems for multiple rods in parallel.

    Solves (D - L - L^T) x = b where:
    - D is block diagonal (6x6 blocks)
    - L is block lower triangular
    """
    env = wp.tid()
    edge_start = env * Ed
    n_edges = Ed

    # Forward elimination phase
    # Load first block
    A0, B0, C0, D0 = _load_block(diag_blocks, edge_start)
    b0, b1 = _load_vec(rhs, edge_start)

    # Factor D_0 once (edit: kernels_solver.py factored it in every solve)
    L11, L21, L22 = _block_factor(A0, C0, D0)

    # Compute c_0 = D_0^{-1} * L_1^T (only if we have more than one block; the last block's c is never read)
    if n_edges > 1:
        for col in range(6):
            u0, u1 = _block_column(offdiag_blocks, edge_start + 1, col)
            x0, x1 = _block_solve_factored(L11, L21, L22, u0, u1)
            _block_set_column(c_blocks, edge_start, col, x0, x1)

    # Compute d'_0 = D_0^{-1} * b_0
    d0, d1 = _block_solve_factored(L11, L21, L22, b0, b1)
    _store_vec(x, edge_start, d0, d1)

    # Forward pass: eliminate lower triangular part
    for i in range(1, n_edges):
        # Load D_i (diagonal block)
        DiA, DiB, DiC, DiD = _load_block(diag_blocks, edge_start + i)

        # Load L_i^T (super-diagonal stored in offdiag_blocks)
        LtA, LtB, LtC, LtD = _load_block(offdiag_blocks, edge_start + i)

        # Load C_{i-1}
        CpA, CpB, CpC, CpD = _load_block(c_blocks, edge_start + i - 1)

        # Transpose to get L_i: [A B; C D]^T = [A^T C^T; B^T D^T]
        LiA = _mat33_transpose(LtA)
        LiB = _mat33_transpose(LtC)  # Swapped: B position gets C^T
        LiC = _mat33_transpose(LtB)  # Swapped: C position gets B^T
        LiD = _mat33_transpose(LtD)

        # Schur complement: T_i = D_i - L_i * C_{i-1}
        LCA, LCB, LCC, LCD = _block_mul(LiA, LiB, LiC, LiD, CpA, CpB, CpC, CpD)
        TiA, TiB, TiC, TiD = _block_sub(DiA, DiB, DiC, DiD, LCA, LCB, LCC, LCD)

        # RHS update: b'_i = b_i - L_i * d'_{i-1}
        bi0, bi1 = _load_vec(rhs, edge_start + i)
        dp0, dp1 = _load_vec(x, edge_start + i - 1)
        ld0, ld1 = _block_mul_vec(LiA, LiB, LiC, LiD, dp0, dp1)
        bi0 = bi0 - ld0
        bi1 = bi1 - ld1

        # Factor T_i once; c_i = T_i^{-1} * L_{i+1}^T (if not last block)
        L11, L21, L22 = _block_factor(TiA, TiC, TiD)
        if i < n_edges - 1:
            for col in range(6):
                u0, u1 = _block_column(offdiag_blocks, edge_start + i + 1, col)
                x0, x1 = _block_solve_factored(L11, L21, L22, u0, u1)
                _block_set_column(c_blocks, edge_start + i, col, x0, x1)

        # Compute d'_i = T_i^{-1} * b'_i
        di0, di1 = _block_solve_factored(L11, L21, L22, bi0, bi1)
        _store_vec(x, edge_start + i, di0, di1)

    # Back substitution: x_{n-1} = d'_{n-1} (already in place); x_i = d'_i - c_i * x_{i+1}
    for i in range(n_edges - 2, -1, -1):
        CiA, CiB, CiC, CiD = _load_block(c_blocks, edge_start + i)
        xn0, xn1 = _load_vec(x, edge_start + i + 1)
        cx0, cx1 = _block_mul_vec(CiA, CiB, CiC, CiD, xn0, xn1)
        di0, di1 = _load_vec(x, edge_start + i)
        xi0 = di0 - cx0
        xi1 = di1 - cx1
        _store_vec(x, edge_start + i, xi0, xi1)


# --- kernels_collision.py:284-367 (_warp_compute_corrections_parallel_batched). --------------------
# Edits: (env, edge) prologue; the unused predicted_positions argument and the max_delta_out /
# max_corr_out diagnostics are dropped; inverse masses come from edge_w (zero for a dead edge, which
# then leaves early); the position Jacobian products are written out (J_pos = [I, -I]); the world
# inverse inertia is the identity for a free node; lambda_sum is overwritten on the substep's first
# iteration (omniendo zeroed it) and accumulated afterwards.
@wp.kernel
def compute_corrections(
    edge_w: wp.array(dtype=wp.vec3),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    first: wp.int32,
    P: wp.int32,
    Ed: wp.int32,
    physical: wp.int32,
    inv_inertia: wp.array(dtype=wp.mat33),
    lambda_sum: wp.array(dtype=wp.float32),
    pos_corrections: wp.array(dtype=wp.vec3),
    rot_corrections: wp.array(dtype=wp.vec3),
):
    env, local_edge = wp.tid()
    v = variant[env]
    global_edge = env * Ed + local_edge
    p0_idx = env * P + local_edge
    p1_idx = p0_idx + 1

    base_idx = global_edge * 6
    dl0 = delta_lambda[base_idx + 0]
    dl1 = delta_lambda[base_idx + 1]
    dl2 = delta_lambda[base_idx + 2]
    dl3 = delta_lambda[base_idx + 3]
    dl4 = delta_lambda[base_idx + 4]
    dl5 = delta_lambda[base_idx + 5]

    if first != 0:
        lambda_sum[base_idx + 0] = dl0
        lambda_sum[base_idx + 1] = dl1
        lambda_sum[base_idx + 2] = dl2
        lambda_sum[base_idx + 3] = dl3
        lambda_sum[base_idx + 4] = dl4
        lambda_sum[base_idx + 5] = dl5
    else:
        lambda_sum[base_idx + 0] = lambda_sum[base_idx + 0] + dl0
        lambda_sum[base_idx + 1] = lambda_sum[base_idx + 1] + dl1
        lambda_sum[base_idx + 2] = lambda_sum[base_idx + 2] + dl2
        lambda_sum[base_idx + 3] = lambda_sum[base_idx + 3] + dl3
        lambda_sum[base_idx + 4] = lambda_sum[base_idx + 4] + dl4
        lambda_sum[base_idx + 5] = lambda_sum[base_idx + 5] + dl5

    w = edge_w[global_edge]
    if w[2] == 0.0:
        return  # dead edge: zero Jacobian, zero multipliers
    inv_m0 = w[0]
    inv_m1 = w[1]

    if inv_m0 > 0.0:
        wp.atomic_add(pos_corrections, p0_idx, wp.vec3(dl0 * inv_m0, dl1 * inv_m0, dl2 * inv_m0))

    if inv_m1 > 0.0:
        wp.atomic_add(pos_corrections, p1_idx, wp.vec3(-dl0 * inv_m1, -dl1 * inv_m1, -dl2 * inv_m1))

    if inv_mass_q[v, local_edge] > 0.0:
        dtheta0 = wp.vec3(
            _warp_jacobian_dot(jacobian_rot, global_edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
        )
        wp.atomic_add(rot_corrections, p0_idx, inv_inertia_times(physical, inv_inertia, p0_idx, 1.0, dtheta0))

    if inv_mass_q[v, local_edge + 1] > 0.0:
        dtheta1 = wp.vec3(
            _warp_jacobian_dot(jacobian_rot, global_edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
            _warp_jacobian_dot(jacobian_rot, global_edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
        )
        wp.atomic_add(rot_corrections, p1_idx, inv_inertia_times(physical, inv_inertia, p1_idx, 1.0, dtheta1))


# --- kernels_collision.py:357-372 (_warp_apply_accumulated_corrections). ---------------------------
# Edits: (env, node) prologue; positions are the body's (E, P) view; inactive nodes are skipped;
# the accumulators are cleared after use so they are zero between launches by construction (tet.py
# apply_average).
@wp.kernel
def apply_corrections(
    flags: wp.array2d(dtype=wp.int32),
    P: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    frame_pred: wp.array(dtype=wp.quat),
    pos_corrections: wp.array(dtype=wp.vec3),
    rot_corrections: wp.array(dtype=wp.vec3),
):
    env, e = wp.tid()
    particle = env * P + e
    if (flags[env, e] & ACTIVE) != 0:
        q[env, e] = q[env, e] + pos_corrections[particle]
        frame_pred[particle] = _warp_quat_correction_g(frame_pred[particle], rot_corrections[particle])
    pos_corrections[particle] = wp.vec3(0.0, 0.0, 0.0)
    rot_corrections[particle] = wp.vec3(0.0, 0.0, 0.0)


# Regional compliance retains the arithmetic order of the former direct_regions kernel.
@wp.kernel
def prepare_regional_compliance(
    rest_len: wp.array2d(dtype=float),
    young: wp.array(dtype=float), torsion: wp.array(dtype=float), bend: wp.array(dtype=float), twist: wp.array(dtype=float),
    ry: wp.array2d(dtype=float), rt: wp.array2d(dtype=float), rb: wp.array2d(dtype=float), rw: wp.array2d(dtype=float),
    Ed: int, dt: float, compliance: wp.array(dtype=float),
):
    env, e = wp.tid()
    L = rest_len[0, e]
    dt2 = dt * dt
    base = (env * Ed + e) * 6
    compliance[base] = 1.0e-10
    compliance[base + 1] = 1.0e-10
    compliance[base + 2] = 1.0e-10
    kb = young[env] * ry[env, e] * bend[env] * rb[env, e]
    kt = torsion[env] * rt[env, e] * twist[env] * rw[env, e]
    compliance[base + 3] = 1.0 / (kb * L * dt2 + 1.0e-10)
    compliance[base + 4] = compliance[base + 3]
    compliance[base + 5] = 1.0 / (kt * L * dt2 + 1.0e-10)


# Physical rotational inertia (catheter_vasculature_solver xpbd_rod_solver.py:820-865, _xr_compute_inv_inertia):
# each node's world inverse inertia R diag(local) R^T from its predicted frame, zero for a locked node.
@wp.kernel
def world_inv_inertia(
    frame_pred: wp.array(dtype=wp.quat),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    local: wp.array(dtype=wp.vec3),
    P: wp.int32,
    inv_inertia: wp.array(dtype=wp.mat33),
):
    env, e = wp.tid()
    i = env * P + e
    if inv_mass_q[variant[env], e] <= 0.0:
        inv_inertia[i] = wp.mat33(0.0)
        return
    r = wp.quat_to_matrix(frame_pred[i])
    d = local[i]  # this env's node
    inv_inertia[i] = r * wp.diag(d) * wp.transpose(r)
