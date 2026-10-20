# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host quaternion helpers, ``[x, y, z, w]`` (omnisurg/instruments/grasper.py conventions; the device
versions are ``@wp.func`` in the systems). Used by the loaders, the input package and the keyboard hub."""

from __future__ import annotations

import math

import numpy as np
import warp as wp


def normalize_quat(q) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(4)
    n = np.linalg.norm(q)
    return (q / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])).astype(np.float32)


def quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = normalize_quat(a)
    bx, by, bz, bw = normalize_quat(b)
    return normalize_quat([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                           aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def axis_angle_quat(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    s = math.sin(0.5 * angle)
    return normalize_quat([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)])


def euler_rz_ry_rx_degrees_quat(euler) -> np.ndarray:  # omnisurg/config/schema.py:452-456: qz * qy * qx
    qx = axis_angle_quat((1, 0, 0), math.radians(euler[0]))
    qy = axis_angle_quat((0, 1, 0), math.radians(euler[1]))
    qz = axis_angle_quat((0, 0, 1), math.radians(euler[2]))
    return quat_mul(qz, quat_mul(qy, qx))


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float32)


def matrix_to_quat(m) -> np.ndarray:
    """``[x, y, z, w]`` of a proper 3x3 rotation matrix (Warp's ``quat_from_matrix``, callable on the host)."""
    q = wp.quat_from_matrix(wp.mat33(*np.asarray(m, np.float32).ravel().tolist()))
    return np.array([q[0], q[1], q[2], q[3]], np.float32)


def quat_conjugate(q) -> np.ndarray:  # omnisurg/instruments/controller.py:430-437
    q = normalize_quat(q)
    return np.float32([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q, v) -> np.ndarray:  # controller.py:440-452: rotate a vector by a unit quaternion
    return (quat_to_matrix(normalize_quat(q)) @ np.asarray(v, np.float32)).astype(np.float32)

# Existing input YAML uses Rz @ Ry @ Rx.
euler_xyz_degrees_quat = euler_rz_ry_rx_degrees_quat
