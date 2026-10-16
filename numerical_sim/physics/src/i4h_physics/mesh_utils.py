# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from dataclasses import dataclass

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


CAP_MIN_COSINE = 0.5  # a fan cap triangle turned more than 60 degrees from its loop's normal: ear clipping is tried


@dataclass(eq=False)
class BoundaryLoop:
    """A closed loop of boundary half-edges (edges owned by one triangle) of a triangle mesh: original vertex indices in
    the order of the triangles' winding, and what the open-end classification made of it. ``normal`` is the unit vector
    area of that winding; it points away from the side the face normals point to (out of the lumen through an
    inward-facing vessel mesh's open end)."""

    ids: np.ndarray  # (n,) distinct original vertex indices
    points: np.ndarray  # (n, 3) float64
    pinched: bool = False  # it shares a rim vertex with another loop (a non-manifold vertex: a figure eight's lobes)
    kind: str = ""  # classify_boundary: open, seam or crack; open_end_caps: capped, entry, seam or crack
    method: str = ""  # an open end's cap (cap_loop): fan or ear clipping
    cosine: float = float("nan")  # its least-aligned cap triangle: the cosine of its normal with the cap's (1 flat)

    def __post_init__(self):
        p = self.points
        self.centroid = p.mean(axis=0)
        area = 0.5 * np.cross(p - self.centroid, np.roll(p, -1, axis=0) - self.centroid).sum(axis=0)
        self.area = float(np.linalg.norm(area))
        self.normal = area / max(self.area, 1.0e-30)
        edges = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        self.perimeter, self.edge = float(edges.sum()), float(edges.mean())
        self.radius = float(np.linalg.norm(p - self.centroid, axis=1).max())

    @property
    def width(self) -> float:
        """A slit's width: its area over half its perimeter."""
        return 2.0 * self.area / max(self.perimeter, 1.0e-30)


def boundary_loops(points: np.ndarray, tris: np.ndarray) -> list[BoundaryLoop]:
    """Closed loops of the boundary half-edges of a triangle mesh whose vertices are welded within 1e-6 (as
    ``loaders.static_mesh.weld_map``), each as original vertex indices in the order of the triangles' winding. Where a
    rim pinches (a vertex with two outgoing boundary half-edges, non-manifold), its walk is split into simple loops,
    each marked ``pinched`` (a figure eight: its two lobes). A loop with fewer than three vertices is dropped."""
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
    pinches = {a for a, ks in outgoing.items() if len(ks) > 1}  # rim vertices shared by two loops or lobes
    used = [False] * len(half)
    loops = []
    for start_edge in range(len(half)):
        if used[start_edge]:
            continue
        start, k, walk, closed = half[start_edge][0], start_edge, [], False
        while k is not None:  # at a pinch the boundary has two outgoing half-edges: either continues the walk
            used[k] = True
            walk.append(half[k][0])
            closed = half[k][1] == start
            if closed:
                break
            k = next((j for j in outgoing.get(half[k][1], ()) if not used[j]), None)
        if not closed:
            continue
        for piece in _simple_loops(walk):
            if len(piece) >= 3:
                ids = first[np.asarray(piece)]
                loops.append(BoundaryLoop(ids, points[ids].astype(np.float64), pinched=not pinches.isdisjoint(piece)))
    return loops


def _simple_loops(walk: list[int]) -> list[list[int]]:
    """A closed walk split at the vertices it passes more than once into simple loops, each in the walk's order."""
    pieces, stack, at = [], [], {}
    for v in walk:
        if v in at:  # the walk came back to v: the vertices since then are a loop through v
            i = at[v]
            pieces.append(stack[i:])
            for u in stack[i + 1:]:
                del at[u]
            del stack[i + 1:]
        else:
            at[v] = len(stack)
            stack.append(v)
    return pieces + [stack]


def classify_boundary(points: np.ndarray, tris: np.ndarray, seam_tolerance: float = 0.5) -> list[BoundaryLoop]:
    """``boundary_loops`` with their ``kind``:

    - crack: a slit, enclosing less than a quarter of the area a circle of its perimeter would (``area < perimeter**2 /
      (16 pi)``; an open end is close to round). Nothing passes it that the wall on both sides does not already stop,
      unless it is wider than a node (``BoundaryLoop.width``; ``rod_containment`` warns);
    - seam: a loop lying on another boundary loop (two pieces meeting without shared vertices), every vertex within
      ``seam_tolerance`` mean edge lengths of the other loop's polyline; capping it would wall off the passage;
    - open: an open end."""
    loops = boundary_loops(points, tris)
    for i, loop in enumerate(loops):
        if loop.area < loop.perimeter ** 2 / (16.0 * np.pi):
            loop.kind = "crack"
        elif any(j != i and np.linalg.norm(loop.centroid - other.centroid) <= loop.radius + other.radius + seam_tolerance * loop.edge
                 and _distance_to_polyline(loop.points, other.points).max() <= seam_tolerance * loop.edge
                 for j, other in enumerate(loops)):
            loop.kind = "seam"
        else:
            loop.kind = "open"
    return loops


def cap_loop(loop: BoundaryLoop, first: int, min_cosine: float = CAP_MIN_COSINE):
    """Triangles closing one loop, wound against it like the wall, so the closed mesh keeps the authored orientation (an
    inward-facing mesh gets inward-facing caps). Returns ``(new points (m, 3), triangles (k, 3), method, cosine)``: the
    triangles index the mesh's points, the new ones from ``first`` on; ``cosine`` is the least-aligned triangle's
    normal against the cap's. A fan around the loop's centroid, unless one of its triangles turns more than
    ``acos(min_cosine)`` from the cap's normal; then ear clipping in the plane normal to it, when that is better (no ear
    can flip: each has a positive area in that plane). Raises ValueError when neither closes the loop without folding
    (a cosine <= 0): a containment's inside/outside sign near that cap would be wrong."""
    a, b = loop.ids, np.roll(loop.ids, -1)  # the half-edge a -> b; a cap triangle runs b -> a -> ...
    pa, pb = loop.points, np.roll(loop.points, -1, axis=0)
    best = (loop.centroid[None], np.stack([b, a, np.full(len(a), first)], axis=1), "fan",
            _worst_cosine(pb, pa, np.broadcast_to(loop.centroid, pa.shape), -loop.normal))
    if best[3] < min_cosine:
        ears = _ear_clip(loop.points, loop.normal)
        if ears is not None:
            ears = ears[:, ::-1]  # against the loop
            p = loop.points[ears]
            cosine = _worst_cosine(p[:, 0], p[:, 1], p[:, 2], -loop.normal)
            if cosine > best[3]:
                best = (np.zeros((0, 3)), loop.ids[ears], "ear clipping", cosine)
    if not best[3] > 0.0:
        raise ValueError(f"the open end at {np.round(loop.centroid, 3).tolist()} ({len(loop.ids)} vertices) cannot be capped "
                         f"without folding (best cap triangle cosine {best[3]:.3f})")
    return best


def open_end_caps(points: np.ndarray, tris: np.ndarray, lines=(), keep=None, lumen_front: bool = True,
                  clearance: float = 0.0, seam_tolerance: float = 0.5, min_cosine: float = CAP_MIN_COSINE):
    """Triangles closing the open ends of a vessel mesh, for a containment query that must not let anything out through
    them. Returns ``(cap points (m, 3) float32, cap triangles (k, 3) int32, loops)``: the triangles index the points
    after ``points`` (append both); ``loops`` is every boundary loop (``classify_boundary``) with ``kind`` capped, entry,
    seam or crack. Every open end is capped (``cap_loop``) except the entries:

    - ``keep`` None: for each of the ``lines`` (segments ``(start, end)``: an insertion path), the open end it first
      crosses into the lumen, if it crosses no wall before. A line that starts in the lumen (its first crossing leaves
      it) or reaches the vessel through its wall keeps none (a warning for the wall);
    - ``keep`` a list of points: the open ends whose centroids are nearest them, each within that end's radius.

    ``lumen_front``: the lumen is on the side the face normals point to (an inward-facing mesh). Raises ValueError when
    one of the ``lines`` passes within ``clearance`` of a capped end before it enters the vessel: a containment with
    that search radius would push a node coming along it through the cap, a one-way valve."""
    loops = classify_boundary(points, tris, seam_tolerance)
    ends = [loop for loop in loops if loop.kind == "open"]
    base, extra, news, caps, offsets = len(points), [], [], [], []
    for loop in ends:
        offsets.append(base + len(extra))
        new, cap, loop.method, loop.cosine = cap_loop(loop, offsets[-1], min_cosine)
        extra.extend(new)
        news.append(new)
        caps.append(cap)
    pts = np.concatenate([points.astype(np.float64), np.reshape(extra, (-1, 3))])
    closed = np.concatenate([tris] + caps) if caps else tris
    owner = np.concatenate([np.full(len(tris), -1)] + [np.full(len(c), k) for k, c in enumerate(caps)])
    corners = pts[closed]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    approaches, crossed = [], set()  # each line's stretch outside the vessel; the open ends lines enter by
    for start, end in lines:
        p, q = np.asarray(start, np.float64), np.asarray(end, np.float64)
        face, t = _first_crossing(p, q, corners)
        if face < 0:  # never reaches the vessel
            approaches.append((p, q))
            continue
        if (float(np.dot(q - p, normals[face])) > 0.0) != lumen_front:
            continue  # leaves the lumen: it starts inside
        hit = p + t * (q - p)
        approaches.append((p, hit))
        if owner[face] >= 0:
            crossed.add(int(owner[face]))
        else:
            warnings.warn(f"the insertion line from {np.round(p, 3).tolist()} reaches the vessel through its wall at "
                          f"{np.round(hit, 3).tolist()}, not through an open end: no entry is left open", RuntimeWarning, stacklevel=3)
    entries = crossed
    if keep is not None:
        entries = set()
        for x in keep:
            d = [float(np.linalg.norm(loop.centroid - np.asarray(x, np.float64))) for loop in ends]
            k = int(np.argmin(d)) if d else -1
            if k < 0 or d[k] > ends[k].radius:
                raise ValueError(f"keep_open {list(x)}: no open end has its centroid within its radius of it; the open ends: "
                                 + ", ".join(f"{np.round(e.centroid, 3).tolist()} (radius {e.radius:.3f})" for e in ends))
            entries.add(k)
    for k, loop in enumerate(ends):
        loop.kind = "entry" if k in entries else "capped"
    capped = [k for k in range(len(ends)) if k not in entries]
    if clearance > 0.0 and capped:
        cap_corners = pts[np.concatenate([caps[k] for k in capped])]
        cap_owner = np.concatenate([np.full(len(caps[k]), k) for k in capped])
        for p, q in approaches:
            n = int(np.ceil(np.linalg.norm(q - p) / (0.25 * clearance))) + 1  # samples a quarter clearance apart
            for chunk in np.array_split(np.linspace(0.0, 1.0, max(n, 2)), max(1, n // 256)):
                x = p + chunk[:, None] * (q - p)
                d = _distance_to_triangles(x, cap_corners)
                s, f = np.unravel_index(int(np.argmin(d)), d.shape)
                if d[s, f] < clearance:
                    end = ends[cap_owner[f]]
                    raise ValueError(f"the insertion line from {np.round(p, 3).tolist()} passes {d[s, f]:.4f} from the capped "
                                     f"open end at {np.round(end.centroid, 3).tolist()} at {np.round(x[s], 3).tolist()}, before "
                                     f"it enters the vessel (within {clearance}): a node there would be pushed in through the cap")
    out_points, out_tris = [], []
    for k in capped:  # the entries' caps dropped, the new points renumbered
        out_tris.append(np.where(caps[k] >= base, caps[k] - offsets[k] + base + len(out_points), caps[k]))
        out_points.extend(news[k])
    if not out_tris:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), loops
    return np.asarray(out_points, np.float32).reshape(-1, 3), np.concatenate(out_tris).astype(np.int32), loops


def _distance_to_polyline(x: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point ``(n, 3)`` to a closed polyline: ``(n,)``."""
    a, b = poly, np.roll(poly, -1, axis=0)
    d = b - a
    t = np.clip(np.einsum("nsk,sk->ns", x[:, None] - a[None], d) / np.maximum(np.einsum("sk,sk->s", d, d), 1e-30), 0.0, 1.0)
    return np.linalg.norm(x[:, None] - (a[None] + t[..., None] * d[None]), axis=2).min(axis=1)


def _worst_cosine(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, normal: np.ndarray) -> float:
    """The least cosine between a triangle's normal (``p0 -> p1 -> p2``, ``(k, 3)`` each) and ``normal``; 0 if one has
    no area."""
    n = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(n, axis=1)
    return float(np.where(length > 0.0, n @ normal / np.maximum(length, 1e-300), 0.0).min())


def _ear_clip(points: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    """Ear clipping of a loop ``(n, 3)`` projected onto the plane normal to its vector area ``normal``, where it runs
    counter-clockwise: ``(n - 2, 3)`` local indices wound like the loop, each ear the one whose normal is closest to
    ``normal``; None when the projection crosses itself (no ear left)."""
    u = np.cross(normal, (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)  # (u, v, normal) right-handed: the loop's positive area turns it counter-clockwise
    rel = points - points.mean(axis=0)
    xy = np.stack([rel @ u, rel @ v], axis=1)
    tiny = 1.0e-12 * float(np.ptp(xy, axis=0).max()) ** 2

    def turn(a, b, c):  # twice the signed area of (a, b, c), b and c broadcast
        return (b[..., 0] - a[0]) * (c[..., 1] - a[1]) - (b[..., 1] - a[1]) * (c[..., 0] - a[0])

    left, ears = list(range(len(points))), []
    while len(left) > 3:
        best = None
        for k in range(len(left)):
            i, j, m = left[k - 1], left[k], left[(k + 1) % len(left)]
            a, b, c = xy[i], xy[j], xy[m]
            if turn(a, b, c) <= tiny:
                continue  # reflex or flat
            others = xy[[o for o in left if o not in (i, j, m)]]
            if ((turn(a, b, others) >= -tiny) & (turn(b, c, others) >= -tiny) & (turn(c, a, others) >= -tiny)).any():
                continue  # another vertex inside or on it
            cosine = _worst_cosine(points[[i]], points[[j]], points[[m]], normal)
            if best is None or cosine > best[0]:
                best = (cosine, k, (i, j, m))
        if best is None:
            return None
        ears.append(best[2])
        del left[best[1]]
    if turn(xy[left[0]], xy[left[1]], xy[left[2]]) <= tiny:
        return None
    return np.asarray(ears + [tuple(left)], np.int64)


def _first_crossing(p: np.ndarray, q: np.ndarray, corners: np.ndarray) -> tuple[int, float]:
    """The triangle (``corners (T, 3, 3)``) the segment ``p -> q`` crosses closest to ``p``, and the crossing's fraction
    along it; ``(-1, inf)`` if none (Moller-Trumbore, edges and corners included)."""
    a, e1, e2, d = corners[:, 0], corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], q - p
    h = np.cross(d, e2)
    det = np.einsum("tk,tk->t", e1, h)
    ok = np.abs(det) > 1.0e-12 * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d)
    inv = 1.0 / np.where(ok, det, 1.0)
    s = p - a
    u = np.einsum("tk,tk->t", s, h) * inv
    r = np.cross(s, e1)
    v = (r @ d) * inv
    t = np.einsum("tk,tk->t", e2, r) * inv
    eps = 1.0e-9
    hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t >= 0.0) & (t <= 1.0)
    if not hit.any():
        return -1, float("inf")
    t = np.where(hit, t, np.inf)
    face = int(np.argmin(t))
    return face, float(t[face])


def _distance_to_triangles(x: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Distance from each point ``(n, 3)`` to each triangle ``(T, 3, 3)``: ``(n, T)``."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    height = np.einsum("ntk,tk->nt", x[:, None] - a[None], n)
    foot = x[:, None] - height[..., None] * n[None]  # the projection onto each triangle's plane

    def side(p0, p1):
        return np.einsum("ntk,tk->nt", np.cross(p1 - p0, foot - p0), n)

    inside = (side(a, b) >= 0.0) & (side(b, c) >= 0.0) & (side(c, a) >= 0.0)

    def to_edge(p0, p1):
        d = p1 - p0
        t = np.clip(np.einsum("ntk,tk->nt", x[:, None] - p0[None], d) / np.maximum(np.einsum("tk,tk->t", d, d), 1e-300), 0.0, 1.0)
        return np.linalg.norm(x[:, None] - (p0[None] + t[..., None] * d[None]), axis=2)

    edges = np.minimum(np.minimum(to_edge(a, b), to_edge(b, c)), to_edge(c, a))
    return np.where(inside, np.abs(height), edges)
