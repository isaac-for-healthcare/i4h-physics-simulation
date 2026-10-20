# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static textured mesh (the abdominal cavity): ASCII ``<name>.vertices/.tris/.uvs`` files.

Welding and smooth normals follow omnisurg/scene/static_mesh.py:155-330, vectorised with
numpy. Two deliberate deviations: the orientation fixer (flood fill plus a centroid-based
flip) is not ported, the winding is kept as authored and checked to be a closed,
consistently wound manifold (the tissue shader flips normals on backfaces, so an
inward-facing cavity viewed from inside renders correctly); and tangents are accumulated
per original vertex, not per welded position, so vertices that share a position across a
UV seam (mirrored or rotated charts) keep their own tangent frame instead of an average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from i4h_physics.loaders.tet_bundle import read_rows
from i4h_physics.scene import Scene, StaticMesh


def weld_map(points: np.ndarray) -> np.ndarray:
    """``(n,)`` index of the welded position of each vertex (vertices closer than 1e-6 merge)."""
    keys = np.rint(points.astype(np.float64) / 1.0e-6).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def check_closed(indices: np.ndarray, weld: np.ndarray, name: str) -> None:
    """Every undirected edge of the welded mesh appears exactly twice, once per direction."""
    w = weld[indices]
    directed = np.concatenate([w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    bad = np.count_nonzero(counts != 2)
    if bad:
        raise ValueError(f"static mesh {name!r}: {bad} edges are not shared by exactly two triangles (not closed)")
    balance = np.zeros(len(counts), np.int64)
    np.add.at(balance, inverse.reshape(-1), np.where(directed[:, 0] < directed[:, 1], 1, -1))
    flipped = np.count_nonzero(balance)
    if flipped:
        raise ValueError(f"static mesh {name!r}: {flipped} edges are traversed twice in the same direction (inconsistent winding)")


def smooth_normals(points: np.ndarray, indices: np.ndarray, weld: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals accumulated on welded positions, scattered back per vertex."""
    p0, p1, p2 = (points[indices[:, k]].astype(np.float64) for k in range(3))
    face = np.cross(p1 - p0, p2 - p0)
    accum = np.zeros((int(weld.max()) + 1, 3))
    for k in range(3):
        np.add.at(accum, weld[indices[:, k]], face)
    length = np.linalg.norm(accum, axis=1, keepdims=True)
    normals = np.where(length > 1e-12, accum / np.maximum(length, 1e-30), np.array([[0.0, 1.0, 0.0]]))
    return normals[weld].astype(np.float32)


def smooth_tangents(points: np.ndarray, indices: np.ndarray, uvs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Per-vertex tangents ``(n, 4)`` (xyz plus handedness) from the UV parametrisation, orthogonal to ``normals``."""
    p0, p1, p2 = (points[indices[:, k]].astype(np.float64) for k in range(3))
    t0, t1, t2 = (uvs[indices[:, k]].astype(np.float64) for k in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    valid = np.abs(det) > 1e-12
    inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)[:, None]
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inv
    bitangent = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inv
    t_acc, b_acc = np.zeros((len(points), 3)), np.zeros((len(points), 3))
    for k in range(3):
        np.add.at(t_acc, indices[valid, k], tangent[valid])
        np.add.at(b_acc, indices[valid, k], bitangent[valid])
    n = normals.astype(np.float64)
    t = t_acc - n * np.sum(n * t_acc, axis=1, keepdims=True)  # Gram-Schmidt against the normal
    length = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.where(np.abs(n[:, 1:2]) > 0.95, np.array([[1.0, 0.0, 0.0]]), np.cross(n, np.array([0.0, 1.0, 0.0])))
    fallback /= np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-30)
    t = np.where(length > 1e-12, t / np.maximum(length, 1e-30), fallback)
    handedness = np.where(np.sum(np.cross(n, t) * b_acc, axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([t, handedness[:, None]], axis=1).astype(np.float32)


def add_static_mesh(scene: Scene, name: str, path, translation=(0.0, 0.0, 0.0), scale=1.0, visual=None) -> StaticMesh:
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"static mesh {name!r}: {root} is not a directory")
    base = root.name if (root / f"{root.name}.vertices").is_file() else "model"
    points = read_rows(root / f"{base}.vertices", 3, np.float32)
    indices = read_rows(root / f"{base}.tris", 3, np.int32)
    n = len(points)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= n:
        raise ValueError(f"static mesh {name!r}: triangle index outside [0, {n})")
    uvs = None
    if (root / f"{base}.uvs").is_file():
        uvs = read_rows(root / f"{base}.uvs", 2, np.float32)
        if len(uvs) != n:
            raise ValueError(f"static mesh {name!r}: {len(uvs)} uv rows for {n} vertices")
    points = (points * np.asarray(scale, np.float32) + np.asarray(translation, np.float32)).astype(np.float32)
    weld = weld_map(points)
    check_closed(indices, weld, name)
    normals = smooth_normals(points, indices, weld)
    tangents = smooth_tangents(points, indices, uvs, normals) if uvs is not None else None
    mesh = StaticMesh(name, np.ascontiguousarray(points), np.ascontiguousarray(indices), uvs, normals, tangents)
    scene.static_meshes.append(mesh)
    return mesh
