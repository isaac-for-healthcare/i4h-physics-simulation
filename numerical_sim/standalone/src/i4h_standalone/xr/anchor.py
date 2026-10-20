# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XR <-> world pose math, pure numpy (omnisurg ``rendering/openxr.py`` 1814-1954).

OpenXR poses are metres in the runtime's LOCAL space (y up, -z forward); the world is the case's scene units. The
first valid head pose is anchored onto the case's authored camera (``PoseAnchor``), so the headset starts where the
desktop camera looks from; ``world_units_per_meter`` scales every XR position. Quaternions are xyzw.
"""

from __future__ import annotations

import math

import numpy as np


def quaternion_multiply(left, right) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = (float(v) for v in left)
    rx, ry, rz, rw = (float(v) for v in right)
    result = np.asarray((lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx,
                         lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz), np.float64)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(v) for v in result / norm)


def quaternion_from_matrix(matrix) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, np.float64)
    trace = float(np.trace(m[:3, :3]))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    index = int(np.argmax(np.diag(m[:3, :3])))
    if index == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if index == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


def pose_matrix(position, rotation) -> np.ndarray:
    """``(4, 4)`` rigid transform of a position and an xyzw rotation."""
    x, y, z, w = quaternion_multiply((0.0, 0.0, 0.0, 1.0), rotation)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = ((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                      (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                      (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)))
    matrix[:3, 3] = np.asarray(position, np.float64)
    return matrix


def transform_xr_pose(position, rotation, world_from_xr, world_units_per_meter: float):
    """An XR pose (metres, LOCAL space) in the world through the anchor: ``(position (3,), rotation xyzw (4,))``."""
    matrix = np.asarray(world_from_xr, np.float64)
    point = np.ones(4, dtype=np.float64)
    point[:3] = np.asarray(position, np.float64) * float(world_units_per_meter)
    world_position = matrix @ point
    world_rotation = quaternion_multiply(quaternion_from_matrix(matrix[:3, :3]), rotation)
    return tuple(float(v) for v in world_position[:3]), world_rotation


def projection_from_fov(angle_left: float, angle_right: float, angle_up: float, angle_down: float, near: float,
                        far: float) -> np.ndarray:
    """The off-axis projection of an OpenXR eye fov (radians): right-handed view (-z forward), clip y up, depth 0 at
    ``near`` and 1 at ``far``, w the distance along the view (``Camera.view_projection``'s conventions)."""
    left, right = math.tan(angle_left), math.tan(angle_right)
    down, up = math.tan(angle_down), math.tan(angle_up)
    width, height = right - left, up - down
    if width <= 0.0 or height <= 0.0 or near <= 0.0 or far <= near:
        raise ValueError("invalid OpenXR frustum or clip planes")
    return np.asarray([[2.0 / width, 0.0, (right + left) / width, 0.0],
                       [0.0, 2.0 / height, (up + down) / height, 0.0],
                       [0.0, 0.0, far / (near - far), far * near / (near - far)],
                       [0.0, 0.0, -1.0, 0.0]], np.float32)


class PoseAnchor:
    """The first valid XR head pose placed at the authored camera pose (position, xyzw rotation whose columns are
    right, up, -forward); reset on a reconnect (a new session anchors again)."""

    def __init__(self, authored_position, authored_rotation, world_units_per_meter: float):
        if not float(world_units_per_meter) > 0.0:
            raise ValueError("world_units_per_meter must be positive")
        self.authored_position = tuple(float(v) for v in authored_position)
        self.authored_rotation = tuple(float(v) for v in authored_rotation)
        self.world_units_per_meter = float(world_units_per_meter)
        self.world_from_xr: np.ndarray | None = None

    @classmethod
    def from_camera(cls, camera, world_units_per_meter: float) -> "PoseAnchor":
        """Anchored at a ``render.camera.Camera`` (its basis: right, up, forward)."""
        pos, right, up, forward = camera.basis()
        rotation = quaternion_from_matrix(np.column_stack((right, up, -np.asarray(forward))))
        return cls(pos, rotation, world_units_per_meter)

    @property
    def anchored(self) -> bool:
        return self.world_from_xr is not None

    def observe_head(self, position, rotation, *, valid: bool) -> bool:
        if self.anchored or not valid:
            return False
        world_from_head = pose_matrix(self.authored_position, self.authored_rotation)
        xr_from_head = pose_matrix(np.asarray(position, np.float64) * self.world_units_per_meter, rotation)
        self.world_from_xr = world_from_head @ np.linalg.inv(xr_from_head)
        return True

    def transform(self, position, rotation):
        if self.world_from_xr is None:
            raise RuntimeError("no valid OpenXR head pose has anchored the world yet")
        return transform_xr_pose(position, rotation, self.world_from_xr, self.world_units_per_meter)

    def eye_camera(self, position, rotation, fov, near: float, far: float):
        """``(basis, view_projection)`` of an XR eye (pose in LOCAL space, fov angles in radians) for
        ``SlangRenderer._frame_uniforms``: basis ``(position, right, up, forward)``, float32."""
        world_position, world_rotation = self.transform(position, rotation)
        world_from_eye = pose_matrix(world_position, world_rotation)
        view = np.linalg.inv(world_from_eye)
        projection = projection_from_fov(fov[0], fov[1], fov[2], fov[3], near, far)
        r = world_from_eye[:3, :3]
        basis = (np.asarray(world_position, np.float32), r[:, 0].astype(np.float32), r[:, 1].astype(np.float32),
                 (-r[:, 2]).astype(np.float32))
        return basis, (projection.astype(np.float64) @ view).astype(np.float32)

    def reset_for_reconnect(self) -> None:
        self.world_from_xr = None
