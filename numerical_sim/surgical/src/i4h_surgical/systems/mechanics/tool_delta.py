# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Surgical contact/grasp displacement reduction; kernel and launch order unchanged."""
import warp as wp
from i4h_physics.scene import ACTIVE


# ``apply_average`` for the tool systems (M11 haptics), in one launch: per-instrument channels ``(E*count, n)`` hold each
# tool's share of ``delta`` (``delta_k``: the whole share; ``delta_jaw``: the share signed by the jaw side, +1 left, -1
# right, 0 elsewhere, so a squeeze adds up where the net cancels). Before the average is applied, the share that will
# actually move the particle, ``channel / delta_count``, times the particle's mass is summed into the tool's ``impulse``
# and ``jaw_impulse`` (mass x displacement, sim units), under the same conditions as the apply (active particle, a
# contribution this launch), so the sums equal what is applied; then the average is applied and every accumulator
# cleared, as ``apply_average`` does. ``moment`` sums the same shares' moments about the tool's ``root``, taken at the
# particle's position before the apply (the coupling's torque, i4h_isaaclab.coupling).
@wp.kernel
def apply_average_tools(
    flags: wp.array2d(dtype=wp.int32),
    inv_mass: wp.array2d(dtype=wp.float32),
    count: wp.int32,
    tools: wp.int32,
    root: wp.array2d(dtype=wp.vec3),
    q: wp.array2d(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    delta_count: wp.array(dtype=wp.int32),
    delta_k: wp.array2d(dtype=wp.vec3),
    delta_jaw: wp.array2d(dtype=wp.vec3),
    impulse: wp.array2d(dtype=wp.vec3),
    jaw_impulse: wp.array2d(dtype=wp.vec3),
    moment: wp.array2d(dtype=wp.vec3),
):
    env, local = wp.tid()
    flat_index = env * count + local
    n = delta_count[flat_index]
    active = (flags[env, local] & ACTIVE) != 0
    w = inv_mass[env, local]
    if active and n > 0 and w > 0.0:
        scale = 1.0 / (w * wp.float32(n))
        for k in range(tools):
            d = delta_k[flat_index, k]
            if wp.length_sq(d) > 0.0:
                wp.atomic_add(impulse, env, k, d * scale)
                wp.atomic_add(moment, env, k, wp.cross(q[env, local] - root[env, k], d * scale))
                delta_k[flat_index, k] = wp.vec3(0.0, 0.0, 0.0)
            j = delta_jaw[flat_index, k]
            if wp.length_sq(j) > 0.0:
                wp.atomic_add(jaw_impulse, env, k, j * scale)
                delta_jaw[flat_index, k] = wp.vec3(0.0, 0.0, 0.0)
    if active and n > 0:
        q[env, local] = q[env, local] + delta[flat_index] / wp.float32(n)
    delta[flat_index] = wp.vec3(0.0, 0.0, 0.0)
    delta_count[flat_index] = 0
