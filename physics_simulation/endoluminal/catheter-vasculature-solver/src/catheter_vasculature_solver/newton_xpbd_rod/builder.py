# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builder helper for adding Cosserat elastic rods to a Newton model."""

from __future__ import annotations

import math

import numpy as np

from newton import ModelBuilder


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create a quaternion [x, y, z, w] from an axis-angle."""
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / norm
    half = angle * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float32)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions stored as ``[x, y, z, w]``."""
    return np.asarray(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ],
        dtype=np.float32,
    )


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to ``[x, y, z, w]``."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q /= np.linalg.norm(q)
    return q.astype(np.float32)


def _material_frames(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        frames.append(_quat_from_matrix(np.column_stack((x, y, z))))
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


def add_elastic_rod(
    builder: ModelBuilder,
    positions: np.ndarray,
    radius: float = 0.01,
    particle_mass: float = 0.1,
    bend_stiffness: float = 1.0,
    twist_stiffness: float = 1.0,
    young_modulus: float = 1.0e6,
    torsion_modulus: float = 1.0e6,
    lock_root: bool = True,
    lock_root_rotation: bool = True,
) -> list[int]:
    """Add a Cosserat elastic rod to the model.

    Adds particles and stores rod metadata in the ``xpbd_rod`` custom
    namespace. The solver reads this data at construction time to build
    internal GPU arrays.

    Args:
        builder: Model builder to add the rod to.
        positions: Initial positions as ``(N, 3)`` array [m].
        radius: Rod cross-section radius [m].
        particle_mass: Mass of each particle [kg].
        bend_stiffness: Bending stiffness coefficient.
        twist_stiffness: Twist stiffness coefficient.
        young_modulus: Young's modulus [Pa].
        torsion_modulus: Torsion modulus [Pa].
        lock_root: Whether the first particle is position-locked.
        lock_root_rotation: Whether the first particle is rotation-locked.

    Returns:
        List of particle indices.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")

    num_points = positions.shape[0]
    if num_points < 2:
        raise ValueError("Rod requires at least 2 points")

    num_edges = num_points - 1

    # Compute rest lengths from positions
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(np.float32)

    # Continuous, twist-minimizing frames also handle curved polylines. Their
    # relative rotations form the nonzero baseline curvature of the rod.
    orientations, rest_darboux = _material_frames(positions)

    # Compute edge bend/twist stiffness vectors
    bend_stiffness_vec = np.zeros((num_edges, 3), dtype=np.float32)
    bend_stiffness_vec[:, 0] = bend_stiffness
    bend_stiffness_vec[:, 1] = bend_stiffness
    bend_stiffness_vec[:, 2] = twist_stiffness

    # Inverse masses
    inv_mass = 0.0 if particle_mass == 0.0 else 1.0 / particle_mass
    inv_masses = np.full(num_points, inv_mass, dtype=np.float32)
    if lock_root:
        inv_masses[0] = 0.0

    # Quaternion inverse masses (rotation lock)
    quat_inv_masses = np.ones(num_points, dtype=np.float32)
    if lock_root_rotation:
        quat_inv_masses[0] = 0.0

    # Add particles
    particle_indices = []
    for i in range(num_points):
        mass = 0.0 if inv_masses[i] == 0.0 else particle_mass
        idx = builder.add_particle(
            pos=tuple(positions[i]),
            vel=(0.0, 0.0, 0.0),
            mass=mass,
            radius=radius,
        )
        particle_indices.append(idx)

    # Store rod data in xpbd_rod namespace lists
    ns = builder._xpbd_rod_data

    ns["rod_num_points"].append(num_points)
    ns["rod_particle_start"].append(particle_indices[0])
    ns["rod_young_modulus"].append(young_modulus)
    ns["rod_torsion_modulus"].append(torsion_modulus)

    ns["orientations"].extend(orientations.tolist())
    ns["quat_inv_masses"].extend(quat_inv_masses.tolist())
    ns["rest_lengths"].extend(rest_lengths.tolist())
    ns["rest_darboux"].extend(rest_darboux.tolist())
    ns["bend_stiffness"].extend(bend_stiffness_vec.tolist())

    return particle_indices
