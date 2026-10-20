# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import numpy as np


def triangulate(face_indices: np.ndarray, face_counts: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygon faces (grasper.py:808-829), vectorised for the common all-triangle case."""
    if face_counts.size and (face_counts == 3).all():
        return face_indices.reshape(-1, 3).astype(np.int32)
    tris, start = [], 0
    for count in face_counts.tolist():
        if count >= 3:
            first = int(face_indices[start])
            for k in range(1, count - 1):
                tris.append((first, int(face_indices[start + k]), int(face_indices[start + k + 1])))
        start += count
    return np.asarray(tris, np.int32).reshape(-1, 3)



# Ported from omniendo xcath/config.py:1410-1413 (rotation_matrix_degrees). Edits: none.
def euler_rx_ry_rz_degrees_matrix(rotation) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(rotation, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.asarray([[cy * cz, -cy * sz, sy], [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy], [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy]], dtype=np.float32)


rotation_matrix_degrees = euler_rx_ry_rz_degrees_matrix


def select_pins(points: np.ndarray, tris: np.ndarray, pins, name: str) -> np.ndarray:
    """Union of the pin selectors as vertex indices (omniendo config.py:1850-1880). ``boundary``: vertices of edges
    owned by one triangle; ``sphere``: ``{center, radius}``; ``box``: ``{minimum, maximum}``. A selector matching
    nothing is an error, as is an unknown type."""
    selected: set[int] = set()
    for k, sel in enumerate(pins or ()):
        sel = dict(sel)
        kind = sel.pop("type", None)
        if kind == "boundary":
            keys = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
            edges, counts = np.unique(keys, axis=0, return_counts=True)
            matches = np.unique(edges[counts == 1])
        elif kind == "sphere":
            delta = points - np.asarray(sel.pop("center"), np.float32)
            matches = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= float(sel.pop("radius")) ** 2)
        elif kind == "box":
            lo, hi = np.asarray(sel.pop("minimum"), np.float32), np.asarray(sel.pop("maximum"), np.float32)
            matches = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
        else:
            raise ValueError(f"mesh {name!r}: pins[{k}].type must be boundary, sphere or box, got {kind!r}")
        if sel:
            raise ValueError(f"mesh {name!r}: pins[{k}]: unknown keys {sorted(sel)}")
        if len(matches) == 0:
            raise ValueError(f"mesh {name!r}: pins[{k}] ({kind}) matched no vertices")
        selected.update(int(v) for v in matches)
    return np.asarray(sorted(selected), np.int32)


def shell_topology(points: np.ndarray, tris: np.ndarray, name: str):
    """Unique edges with rest lengths and interior-edge dihedrals ``(o0, o1, v0, v1)`` with rest angles, as Newton's
    ``add_cloth_mesh`` derives them (utils/mesh.py MeshAdjacency; builder.py:8344-8414). Zero-length edges are dropped
    with a warning (omniendo solver_xpbd_rod.py:708-720); an edge in three triangles is a non-manifold error."""
    pts = points.astype(np.float64)
    sides = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])  # directed, slot-major
    opposite = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])
    keys = np.sort(sides, axis=1)
    edges, first, inverse, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if counts.max() > 2:
        raise ValueError(f"mesh {name!r}: {int((counts > 2).sum())} edges belong to more than two triangles (not a manifold)")
    length = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)
    keep = length > 1.0e-12
    if not keep.all():
        warnings.warn(f"mesh {name!r}: {int((~keep).sum())} zero-length edges dropped", RuntimeWarning, stacklevel=3)
    # Dihedrals: an interior edge's first side (in the first triangle's winding) gives (v0, v1) and opposite o0; the
    # second triangle traverses it reversed and gives o1. Newton: triangles (o0, v0, v1) and (o1, v1, v0) are both CCW.
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    starts = np.searchsorted(sorted_inverse, np.arange(len(edges)))
    interior = np.flatnonzero((counts == 2) & keep)
    a, b = order[starts[interior]], order[starts[interior] + 1]
    dihedrals = np.stack([opposite[a], opposite[b], sides[a, 0], sides[a, 1]], axis=1).astype(np.int32)
    xi, xj, xk, xl = (pts[dihedrals[:, c]] for c in range(4))
    n0 = np.cross(xk - xi, xl - xi)
    n1 = np.cross(xl - xj, xk - xj)
    edge = xl - xk
    def norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)
    n0, n1, e = norm(n0), norm(n1), norm(edge)
    rest_angle = np.arctan2(np.sum(np.cross(n0, n1) * e, axis=1), np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0))
    return (edges[keep].astype(np.int32), length[keep].astype(np.float32), dihedrals, rest_angle.astype(np.float32),
            np.linalg.norm(edge, axis=1).astype(np.float32), counts)
