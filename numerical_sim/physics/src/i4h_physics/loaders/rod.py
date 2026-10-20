# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline straight elastic rod asset (a catheter): ``count`` nodes from ``start`` along ``direction``.

Material frames and rest Darboux vectors are ported from omniendo/xpbd_rod/builder.py:85-179
(``_quat_from_axis_angle``, ``_quat_multiply``, ``_material_frames``; the matrix-to-quaternion
conversion is Warp's builtin through ``i4h_physics.quat.matrix_to_quat``);
the rest of ``add_elastic_rod`` (Newton ``ModelBuilder`` particles and custom attributes) is
replaced by a ``ParticleRest`` row set and a ``RodMesh`` record on the body. Geometry and mass
live here; stiffness lives on the rod system (the two backends have different stiffness models).
"""

from __future__ import annotations

import math

import numpy as np
from i4h_physics.quat import axis_angle_quat, matrix_to_quat, quat_mul
from i4h_physics.scene import ACTIVE_INT, Body, Part, ParticleRest, RodMesh, Scene


# omniendo/xpbd_rod/builder.py:85-105 had its own axis-angle and Hamilton product; ``i4h_physics.quat`` has the same
# formulas (the product is normalised there, a float32 rounding on the rest frames).
def _quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    if np.linalg.norm(axis) < 1.0e-8:
        return np.float32([0.0, 0.0, 0.0, 1.0])
    return axis_angle_quat(axis, angle).astype(np.float32)


_quat_multiply = quat_mul


# Ported from omniendo/xpbd_rod/builder.py:128-179 (_material_frames). Edits: none.
def material_frames(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build twist-minimizing frames and their rest Darboux vectors.

    The local z axis follows the centerline tangent.  The local x/y axes are
    parallel transported, which avoids arbitrary frame flips on curved input.
    """
    segments = np.diff(positions, axis=0).astype(np.float64)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1.0e-8):
        raise ValueError("Rod centerline contains coincident consecutive points")
    edge_tangents = segments / lengths[:, None]
    tangents = np.vstack((edge_tangents, edge_tangents[-1]))

    z0 = tangents[0]
    # Prefer the legacy straight-rod frame: for a +X centerline this gives
    # local X=-Z, local Y=+Y, local Z=+X (a +90 degree rotation about Y).
    reference = np.array([0.0, 0.0, -1.0])
    if abs(float(np.dot(reference, z0))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x = reference - np.dot(reference, z0) * z0
    x /= np.linalg.norm(x)
    frames: list[np.ndarray] = []
    previous_z = z0
    for z in tangents:
        cross = np.cross(previous_z, z)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(previous_z, z), -1.0, 1.0))
        if cross_norm > 1.0e-10:
            rotation = _quat_from_axis_angle(cross / cross_norm, math.atan2(cross_norm, dot))
            vector_quat = np.array([x[0], x[1], x[2], 0.0], dtype=np.float32)
            conjugate = np.array([-rotation[0], -rotation[1], -rotation[2], rotation[3]], dtype=np.float32)
            x = _quat_multiply(_quat_multiply(rotation, vector_quat), conjugate)[:3].astype(np.float64)
        elif dot < 0.0:
            # A 180-degree cusp has no unique transport axis; retain a stable
            # perpendicular axis and rotate around it.
            rotation = _quat_from_axis_angle(x, math.pi)
            vector_quat = np.array([x[0], x[1], x[2], 0.0], dtype=np.float32)
            conjugate = np.array([-rotation[0], -rotation[1], -rotation[2], rotation[3]], dtype=np.float32)
            x = _quat_multiply(_quat_multiply(rotation, vector_quat), conjugate)[:3].astype(np.float64)
        x -= np.dot(x, z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        frames.append(matrix_to_quat(np.column_stack((x, y, z))))
        previous_z = z

    orientations = np.asarray(frames, dtype=np.float32)
    rest_darboux = np.empty((len(positions) - 1, 3), dtype=np.float32)
    for i in range(len(rest_darboux)):
        conjugate = np.array([-orientations[i, 0], -orientations[i, 1], -orientations[i, 2], orientations[i, 3]], dtype=np.float32)
        relative = _quat_multiply(conjugate, orientations[i + 1])
        # q and -q encode the same frame; choose the short relative rotation.
        if relative[3] < 0.0:
            relative = -relative
        rest_darboux[i] = relative[:3]
    return orientations, rest_darboux


def add_rod(scene: Scene, name: str, start, direction, count: int, segment_length: float, radius: float,
            mass: float = 1.0, lock_root: bool = True, visual=None, membership=None) -> Body:
    """A straight rod of ``count`` nodes spaced ``segment_length`` apart, one part named after the body.
    ``mass`` is per node; a locked root has zero inverse mass and inverse rotational mass (the
    catheter's insertion point, driven by ``catheter_control``)."""
    P = int(count)
    if P < 3:
        raise ValueError(f"rod {name!r}: count must be at least 3, got {P}")
    if float(segment_length) <= 0.0 or float(radius) <= 0.0 or float(mass) <= 0.0:
        raise ValueError(f"rod {name!r}: segment_length, radius and mass must be positive")
    start = np.asarray(start, np.float32).reshape(3)
    direction = np.asarray(direction, np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-8:
        raise ValueError(f"rod {name!r}: direction must be non-zero")
    direction = (direction / norm).astype(np.float32)
    positions = start[None] + np.arange(P, dtype=np.float32)[:, None] * float(segment_length) * direction[None]
    frames, rest_darboux = material_frames(positions)
    rest_len = np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(np.float32)
    inv_mass = np.full(P, 1.0 / float(mass), np.float32)
    inv_mass_q = np.ones(P, np.float32)
    if lock_root:
        inv_mass[0] = 0.0
        inv_mass_q[0] = 0.0
    E = scene.envs
    rest = ParticleRest(
        q0=np.tile(positions[None], (E, 1, 1)),
        inv_mass=np.tile(inv_mass[None], (E, 1)),
        radius=np.full((E, P), float(radius), np.float32),
        flags=np.full((E, P), ACTIVE_INT, np.int32),
    )
    parts = {name: Part(0, 0, P)}
    body = scene.add_body(name, rest, parts)
    body.rod = RodMesh(
        frames=frames[None].astype(np.float32),
        rest_len=rest_len[None],
        rest_darboux=rest_darboux[None],
        inv_mass_q=inv_mass_q[None],
        start=start[None],
        direction=direction[None],
    )
    if membership is not None:
        body.regions = membership
    return body
