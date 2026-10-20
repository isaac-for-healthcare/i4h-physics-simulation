# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument render geometry: skinned mesh pieces and the collision-sphere overlay for one env.

The pieces are skinned per frame with the same ``skin_point`` the control system uses for its
spheres (omnisurg drew pre-skinned world-space vertices per piece, omnisurg/rendering/renderers.py:1278-1357).
A piece is rigid, so its vertex normals are computed once in local space and rotated with it
(``skin_normal``: the same rotations without the translations); the sphere overlay's normals are
the unit-sphere template. Both renderers call ``skin_pieces`` / ``place_sphere_mesh``; the buffers
they write into are theirs (shared GPU buffers for Slang, plain Warp arrays for GL).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp
from i4h_physics.loaders.static_mesh import smooth_normals
from i4h_surgical.systems.instruments.instrument_control import skin_point


class PieceSkin:
    """One rigid piece: constant local points and bind matrix, world points recomputed per frame."""

    def __init__(self, instrument_index: int, piece, device):
        self.name = piece.name
        self.instrument = int(instrument_index)
        self.role = int(piece.role)
        self.jaw_sign = float(piece.jaw_sign)
        self.color = tuple(piece.color)
        self.count = len(piece.points)
        self.local = wp.array(piece.points, dtype=wp.vec3, device=device)
        bind = np.asarray(piece.bind, np.float64).reshape(4, 4)
        det = float(np.linalg.det(bind[:3, :3]))  # the normals are rotated by the bind's 3x3, so it must be a rotation
        scaled = bind[:3, :3] / max(abs(det), 1e-30) ** (1.0 / 3.0)  # times a uniform scale (normalised away in the kernel)
        if abs(det) < 1e-12 or not np.allclose(scaled @ scaled.T, np.eye(3), atol=1e-4):
            raise ValueError(f"piece {piece.name!r}: the bind matrix is not a rotation times a uniform scale (det {det:.3g})")
        self.bind = wp.mat44(*bind.astype(np.float32).reshape(16))
        self.indices = np.ascontiguousarray(piece.indices.reshape(-1), dtype=np.int32)
        normals = smooth_normals(piece.points, self.indices.reshape(-1, 3), np.arange(len(piece.points)))
        if det < 0.0:  # a reflection flips the winding: the geometric normal of the transformed triangle is -R n
            normals = -normals
        self.local_normals = wp.array(normals, dtype=wp.vec3, device=device)


def piece_skins(control, device) -> list[PieceSkin]:
    return [PieceSkin(k, piece, device) for k, record in enumerate(control.scene.instruments.values()) for piece in record.pieces]


def skin_pieces(control, env: int, skins: list[PieceSkin], positions: list, normals: list, device, stream=None) -> None:
    """World points and normals of every piece for ``env`` into ``positions`` / ``normals`` (one ``(count,)`` vec3
    array per piece each). ``control`` is the instrument control system or a snapshot of it (``snapshots.ControlState``)."""
    for skin, out, out_n in zip(skins, positions, normals):
        wp.launch(
            skin_piece,
            dim=skin.count,
            inputs=[skin.local, skin.local_normals, skin.bind, skin.role, skin.jaw_sign, skin.instrument, int(env),
                    control.root, control.shaft_q, control.wrist_q, control.jaw_q, control.jaw_angle, control.wrist_pivot,
                    control.jaw_anchor],
            outputs=[out, out_n],
            device=device,
            stream=stream,
        )


@wp.func
def skin_normal(local: wp.vec3, jaw_sign: wp.float32, jaw_angle: wp.float32, bind: wp.mat44, role: wp.int32,
                shaft_q: wp.quat, wrist_q: wp.quat, jaw_q: wp.quat):
    """``skin_point`` without the translations: the piece's rotations applied to a direction."""
    n = local
    if jaw_sign != 0.0:
        n = wp.quat_rotate(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), jaw_sign * jaw_angle), n)
    n = wp.transform_vector(bind, n)
    if role == 0:
        n = wp.quat_rotate(shaft_q, n)
    elif role == 1:
        n = wp.quat_rotate(wrist_q, n)
    else:
        n = wp.quat_rotate(jaw_q, n)
    return wp.normalize(n)


@wp.kernel
def skin_piece(
    local: wp.array(dtype=wp.vec3),
    local_normals: wp.array(dtype=wp.vec3),
    bind: wp.mat44,
    role: wp.int32,
    jaw_sign: wp.float32,
    k: wp.int32,
    env: wp.int32,
    root: wp.array2d(dtype=wp.vec3),
    shaft_q: wp.array2d(dtype=wp.quat),
    wrist_q: wp.array2d(dtype=wp.quat),
    jaw_q: wp.array2d(dtype=wp.quat),
    jaw_angle: wp.array2d(dtype=wp.float32),
    wrist_pivot: wp.array(dtype=wp.vec3),
    jaw_anchor: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out[i] = skin_point(local[i], jaw_sign, jaw_angle[env, k], bind, role, root[env, k], shaft_q[env, k],
                        wrist_q[env, k], jaw_q[env, k], wrist_pivot[k], jaw_anchor[k])
    out_normals[i] = skin_normal(local_normals[i], jaw_sign, jaw_angle[env, k], bind, role, shaft_q[env, k],
                                 wrist_q[env, k], jaw_q[env, k])


def uv_sphere(rings: int = 6, segments: int = 8):
    """Unit sphere template: ``(V, 3)`` points and ``(F, 3)`` indices."""
    points = [(0.0, 1.0, 0.0)]
    for r in range(1, rings):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            points.append((math.sin(phi) * math.cos(theta), math.cos(phi), math.sin(phi) * math.sin(theta)))
    points.append((0.0, -1.0, 0.0))
    top, bottom = 0, len(points) - 1
    def ring(r, s):
        return 1 + (r - 1) * segments + s % segments
    faces = []
    for s in range(segments):
        faces.append((top, ring(1, s + 1), ring(1, s)))
        faces.append((bottom, ring(rings - 1, s), ring(rings - 1, s + 1)))
    for r in range(1, rings - 1):
        for s in range(segments):
            a, b, c, d = ring(r, s), ring(r, s + 1), ring(r + 1, s), ring(r + 1, s + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))
    return np.asarray(points, np.float32), np.asarray(faces, np.int32)


def sphere_mesh_indices(sphere_count: int, template_faces: np.ndarray, template_points: int) -> np.ndarray:
    """Index buffer for ``sphere_count`` copies of the template laid out sphere by sphere."""
    offsets = (np.arange(sphere_count, dtype=np.int32) * template_points)[:, None, None]
    return np.ascontiguousarray((template_faces[None] + offsets).reshape(-1))


@wp.kernel
def place_sphere_mesh(
    template_points: wp.array(dtype=wp.vec3),  # not ``template``: Warp emits the name into C++ (the CPU build)
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_active: wp.array2d(dtype=wp.int32),
    show: wp.array(dtype=wp.int32),
    env: wp.int32,
    out: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Vertex ``s * V + v`` of the overlay mesh and its normal (the unit template point); inactive spheres and spheres
    this overlay does not ``show`` (the cut chains have their own colour) collapse to their centre."""
    tid = wp.tid()
    v_count = template_points.shape[0]
    s = tid // v_count
    v = tid - s * v_count
    radius = sphere_radius[s]
    if sphere_active[env, s] == 0 or show[s] == 0:
        radius = 0.0
    out[tid] = sphere_q[env, s] + template_points[v] * radius
    out_normals[tid] = template_points[v]
