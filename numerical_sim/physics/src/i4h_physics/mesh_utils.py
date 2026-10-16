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


def boundary_loops(points: np.ndarray, tris: np.ndarray) -> list[np.ndarray]:
    """Closed loops of the boundary half-edges (edges owned by one triangle) of a triangle mesh whose vertices are
    welded within 1e-6 (as ``loaders.static_mesh.weld_map``), each as original vertex indices in the order of the
    triangles' winding. A loop with fewer than three distinct vertices (a sliver) is dropped."""
    keys = np.rint(points.astype(np.float64) / 1.0e-6).astype(np.int64)
    _, weld = np.unique(keys, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    _, first = np.unique(weld, return_index=True)  # the lowest original index of each welded vertex
    w = weld[tris]
    directed = np.concatenate([w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]]])
    directed = directed[directed[:, 0] != directed[:, 1]]
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    half = directed[counts[inverse.reshape(-1)] == 1].tolist()
    outgoing: dict[int, list[int]] = {}
    for k, (a, _) in enumerate(half):
        outgoing.setdefault(a, []).append(k)
    used = [False] * len(half)
    loops = []
    for start_edge in range(len(half)):
        if used[start_edge]:
            continue
        start, k, loop, closed = half[start_edge][0], start_edge, [], False
        while k is not None:  # a vertex where the boundary pinches has two outgoing half-edges: either closes a loop
            used[k] = True
            loop.append(half[k][0])
            closed = half[k][1] == start
            if closed:
                break
            k = next((j for j in outgoing.get(half[k][1], ()) if not used[j]), None)
        if closed and len(set(loop)) >= 3:
            loops.append(first[np.asarray(loop)])
    return loops


def open_end_caps(points: np.ndarray, tris: np.ndarray, through=(), seam_tolerance: float = 0.5):
    """Triangles closing the open ends of a triangle mesh, for a containment query that must not let anything out
    through them. Returns ``(cap points (m, 3) float32, cap triangles (k, 3) int32, capped loops)``; the triangles index
    the points after ``points`` (append both). Each boundary loop gets a fan around its centroid wound against the loop,
    so the closed mesh keeps the authored orientation (an inward-facing mesh gets inward-facing caps), except

    - a crack: a loop enclosing less than a quarter of the area a circle of its perimeter would (``area < perimeter**2 /
      (16 pi)``; an open end is close to round) or passing a vertex twice; a fan over a slit folds onto itself;
    - a seam: a loop lying on another boundary loop (two pieces meeting without shared vertices), every vertex within
      ``seam_tolerance`` mean edge lengths of the other loop's polyline; capping it would wall off the passage;
    - an entry: a loop whose cap a segment of one of the ``through`` polylines ``(n, 3)`` crosses (a rod passing through
      the opening at rest)."""
    ids = [i for i in boundary_loops(points, tris) if len(np.unique(i)) == len(i)]  # a welded vertex is one index
    loops = [points[i].astype(np.float64) for i in ids]
    centroids = [loop.mean(axis=0) for loop in loops]
    edge = [float(np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1).mean()) for loop in loops]
    reach = [float(np.linalg.norm(loop - c, axis=1).max()) for loop, c in zip(loops, centroids)]

    def distance_to_polyline(x, poly):  # (n, 3) points to a closed polyline: (n,)
        a, b = poly, np.roll(poly, -1, axis=0)
        d = b - a
        t = np.clip(np.einsum("nsk,sk->ns", x[:, None] - a[None], d) / np.maximum(np.einsum("sk,sk->s", d, d), 1e-30), 0.0, 1.0)
        return np.linalg.norm(x[:, None] - (a[None] + t[..., None] * d[None]), axis=2).min(axis=1)

    segments = [(np.asarray(p, np.float64)[:-1], np.asarray(p, np.float64)[1:]) for p in through]
    cap_points, cap_tris, capped = [], [], []
    for i, loop in enumerate(loops):
        area = 0.5 * np.linalg.norm(np.cross(loop - centroids[i], np.roll(loop, -1, axis=0) - centroids[i]).sum(axis=0))
        if area < (edge[i] * len(loop)) ** 2 / (16.0 * np.pi):
            continue
        seam = any(j != i and np.linalg.norm(centroids[i] - centroids[j]) <= reach[i] + reach[j] + seam_tolerance * edge[i]
                   and distance_to_polyline(loop, loops[j]).max() <= seam_tolerance * edge[i] for j in range(len(loops)))
        if seam:
            continue
        a, b = loop, np.roll(loop, -1, axis=0)  # half-edge a -> b; the cap triangle runs b -> a -> centroid
        c = np.broadcast_to(centroids[i], a.shape)
        if any(_segments_cross_triangles(p, q, b, a, c) for p, q in segments):
            continue
        centre = len(points) + len(cap_points)
        cap_points.append(centroids[i])
        cap_tris.append(np.stack([np.roll(ids[i], -1), ids[i], np.full(len(ids[i]), centre)], axis=1))
        capped.append(ids[i])
    if not cap_tris:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), capped
    return np.asarray(cap_points, np.float32), np.concatenate(cap_tris).astype(np.int32), capped


def _segments_cross_triangles(p, q, a, b, c) -> bool:
    """Whether any segment ``p[s] -> q[s]`` crosses any triangle ``(a[t], b[t], c[t])`` (Moller-Trumbore, all pairs)."""
    e1, e2, d = b - a, c - a, q - p
    h = np.cross(d[None], e2[:, None])  # (T, S, 3)
    det = np.einsum("tk,tsk->ts", e1, h)
    ok = np.abs(det) > 1e-12
    inv = 1.0 / np.where(ok, det, 1.0)
    s = p[None] - a[:, None]
    u = np.einsum("tsk,tsk->ts", s, h) * inv
    r = np.cross(s, e1[:, None])
    v = np.einsum("sk,tsk->ts", d, r) * inv
    t = np.einsum("tk,tsk->ts", e2, r) * inv
    return bool((ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)).any())
