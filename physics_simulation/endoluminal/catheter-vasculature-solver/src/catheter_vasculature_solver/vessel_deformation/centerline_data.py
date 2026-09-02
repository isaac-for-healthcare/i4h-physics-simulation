# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Slim centerline IO types, kept separate from any scene config so the library
# does not depend on a full YAML scene schema.

"""Centerline data types for deformable vessel construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Vec3 = tuple[float, float, float]


class SceneConfigError(ValueError):
    """Validation error for centerline / vessel inputs."""


@dataclass
class VesselTransform:
    scale: Vec3 = (1.0, 1.0, 1.0)
    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation_euler_degrees: Vec3 = (0.0, 0.0, 0.0)


@dataclass
class VesselConfig:
    """Minimal vessel pose wrapper used by ``transform_tree``."""

    id: str = "vessel"
    transform: VesselTransform = field(default_factory=VesselTransform)


@dataclass
class CenterlineData:
    """Mosaic-style branch samples with optional radius envelopes."""

    starts: np.ndarray
    ends: np.ndarray
    branch_ids: np.ndarray
    start_radius_min: np.ndarray | None = None
    end_radius_min: np.ndarray | None = None
    start_radius_max: np.ndarray | None = None
    end_radius_max: np.ndarray | None = None


def rotation_matrix_degrees(rotation: Vec3) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(rotation, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.asarray(
        [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
        ],
        dtype=np.float32,
    )


__all__ = [
    "CenterlineData",
    "SceneConfigError",
    "VesselConfig",
    "VesselTransform",
    "Vec3",
    "rotation_matrix_degrees",
]
