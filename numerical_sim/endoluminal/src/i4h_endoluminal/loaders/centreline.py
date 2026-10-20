# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anatomical centreline trees (omniendo ``vessels[].centerline`` with dynamics): a body of graph nodes with segment
frames, bend pairs, lumen radii and spline topology, simulated by ``rod_pbd`` and used by ``lumen_containment``.

Ported from omniendo: the Mosaic JSON reader ``xcath/config.py:1624-1692, 1764-1847`` (branches of samples with
``position``, ``radius_min``, ``radius_max``; non-positive radii are sentinels repaired by ``np.interp``; the
topology sidecar, the branch exclusions and the ``width`` fallback are not carried, both radii are required);
``xcath/centerline_tree.py:202-326`` (``build_centerline_tree``: weld by a quantised key in source coordinates,
stitch components nearer than ``stitch_threshold``, node radius = mean of its samples, render paths between
non-degree-2 nodes, root = nearest node to a position), ``:43-95`` (``build_spline_edge_topology``),
``:139-160`` (``_components``); ``omniendo/xpbd_rod/pbd_cosserat.py:485-534`` (quaternion helpers), ``:563-616``
(``_quat_between``, ``_tree_frames``: parallel transport over a BFS spanning tree), ``:629-661``
(``derive_bend_pairs``: every segment pairs with its parent in the spanning tree, so a junction of degree d gives
d-1 pairs and a cycle's extra segment keeps stretch and shear only); ``xcath/centerline_runtime.py:129-137``
(the world-to-source point) and ``:206-230`` (open root, pinned nodes and segments);
``xcath/centerline_controls.py:103-123`` (root arc lengths). Deviation: the BFS that seeds the frames and the
bend pairs starts at the tree's root (omniendo rooted it at its lowest-indexed fixed node).
"""

from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import warp as wp
from i4h_endoluminal.scene import GraphMetadata, SkinMesh
from i4h_physics.scene import ACTIVE_INT, Body, Part, ParticleRest, RodGraph, Scene
from i4h_physics.systems.rods.rod_pbd import _qmul_np, rest_darboux_quats

# --- Mosaic JSON (xcath/config.py:1624-1692, 1764-1847) ------------------------------------------------------


def load_centreline_json(path, with_sources=False):
    """Per-segment ``(starts, ends, start_radius_min, end_radius_min, start_radius_max, end_radius_max)`` in the
    file's coordinates (float64). Branches with fewer than two samples are skipped."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"centreline not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    branches = payload.get("branches") if isinstance(payload, dict) else None
    if not isinstance(branches, list) or not branches:
        raise ValueError(f"centreline {path}: expected a non-empty 'branches' list")

    def sample_radii(samples, key, branch_index):
        values = np.full(len(samples), np.nan)
        present = False
        for k, sample in enumerate(samples):
            if key not in sample:
                continue
            present = True
            value = float(sample[key])
            if not math.isfinite(value):
                raise ValueError(f"centreline {path}: branch {branch_index} sample {k} has an invalid {key}")
            if value > 0.0:
                values[k] = value
        if not present or not np.any(np.isfinite(values)):
            raise ValueError(f"centreline {path}: branch {branch_index} has no positive {key} (both radii are required)")
        valid = np.flatnonzero(np.isfinite(values))
        # Mosaic endpoint sentinels are non-positive: nearest-value repair at the ends, linear fill inside.
        return np.interp(np.arange(len(values)), valid, values[valid])

    starts, ends, r = [], [], [[] for _ in range(4)]
    seen = set()
    sources = []
    for index, branch in enumerate(branches):
        samples = branch.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"centreline {path}: branch {index} has no samples list")
        branch_id = branch.get("branch_id", index)
        if branch_id in seen:
            raise ValueError(f"centreline {path}: duplicate branch_id {branch_id}")
        seen.add(branch_id)
        if len(samples) < 2:
            continue
        points = np.asarray([s["position"] for s in samples], dtype=np.float64)
        if points.shape != (len(samples), 3) or not np.all(np.isfinite(points)):
            raise ValueError(f"centreline {path}: branch {index} contains invalid positions")
        rmin, rmax = sample_radii(samples, "radius_min", index), sample_radii(samples, "radius_max", index)
        sources.extend([branch_id] * (len(points) - 1))
        starts.extend(points[:-1])
        ends.extend(points[1:])
        for lst, values in zip(r, (rmin[:-1], rmin[1:], rmax[:-1], rmax[1:])):
            lst.extend(values)
    if not starts:
        raise ValueError(f"centreline {path}: no branch has two samples")
    result = (np.asarray(starts), np.asarray(ends), *(np.asarray(v) for v in r))
    return (*result, sources, seen) if with_sources else result


# --- graph (xcath/centerline_tree.py) ---------------------------------------------------------------------------


def _components(edges, count: int) -> list[list[int]]:  # centerline_tree.py:139-160
    adjacency = [[] for _ in range(count)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen, result = set(), []
    for start in range(count):
        if start in seen:
            continue
        seen.add(start)
        queue, component = deque([start]), []
        while queue:
            node = queue.popleft()
            component.append(node)
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        result.append(component)
    return result


def build_tree(starts, ends, start_radii, end_radii, *, weld_epsilon=1.0e-5, stitch_threshold=0.03, root_position=None, sources=None, branch_regions=None):
    """centerline_tree.py:202-326 (``build_centerline_tree``) with the radius mode already applied to the endpoint
    radii: ``(positions (P, 3) f32, edges (S, 2) i32, radii (P,) f32, render_paths, root)``; raises on a graph that
    stays disconnected after stitching."""
    starts, ends = np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)
    node_of: dict[tuple, int] = {}
    positions: list[np.ndarray] = []
    radius_samples: dict[int, list[float]] = defaultdict(list)

    def node_id(point):
        key = tuple(np.rint(point / weld_epsilon).astype(np.int64).tolist())
        if key not in node_of:
            node_of[key] = len(positions)
            positions.append(point.copy())
        return node_of[key]

    edge_set: set[tuple[int, int]] = set()
    provenance = defaultdict(set)
    assigned = {}
    for i, (start, end) in enumerate(zip(starts, ends)):
        a, b = node_id(start), node_id(end)
        for node, radius in ((a, start_radii[i]), (b, end_radii[i])):
            if np.isfinite(radius) and radius > 0.0:
                radius_samples[node].append(float(radius))
        if a != b:
            key = (min(a, b), max(a, b))
            edge_set.add(key)
            if sources is not None:
                provenance[key].add(sources[i])
                if branch_regions is not None:
                    region = branch_regions[sources[i]]
                    if key in assigned and assigned[key] != region:
                        raise ValueError("conflicting regions on a deduplicated edge")
                    assigned[key] = region

    pos = np.asarray(positions, dtype=np.float64)
    edges = sorted(edge_set)
    components = sorted(_components(edges, len(pos)), key=len, reverse=True)
    connected = set(components[0]) if components else set()
    pending = [set(c) for c in components[1:]]
    while pending:  # nearest pair between the growing connected set and any pending component
        best = None
        connected_indices = np.asarray(sorted(connected), dtype=np.int32)
        for ci, component in enumerate(pending):
            for node in component:
                distances = np.linalg.norm(pos[connected_indices] - pos[node], axis=1)
                j = int(np.argmin(distances))
                candidate = (float(distances[j]), node, int(connected_indices[j]), ci)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None or best[0] > stitch_threshold:
            break
        _, a, b, ci = best
        edges.append((min(a, b), max(a, b)))
        connected.update(pending.pop(ci))

    used = sorted({node for edge in edges for node in edge})
    remap = {old: new for new, old in enumerate(used)}
    positions_out = pos[used].astype(np.float32)
    edges_out = [(remap[a], remap[b]) for a, b in edges]
    if len(_components(edges_out, len(used))) != 1:
        raise ValueError(f"centreline: graph is disconnected ({len(_components(edges_out, len(used)))} components) after stitching")

    means = {node: float(np.mean(v)) for node, v in radius_samples.items() if v}
    if not means:
        raise ValueError("centreline: no positive radius data")
    fallback = float(np.median(list(means.values())))
    radii = np.asarray([means.get(old, fallback) for old in used], dtype=np.float32)

    paths = render_paths_from_edges(edges_out, len(used))

    target = positions_out[0] if root_position is None else np.asarray(root_position, dtype=np.float32)
    root = int(np.argmin(np.linalg.norm(positions_out - target, axis=1)))
    result = (positions_out, np.asarray(edges_out, dtype=np.int32), radii, paths, root)
    if sources is not None:
        return (*result, tuple(tuple(sorted(provenance[e], key=str)) for e in edges),
                np.asarray([assigned.get(e, 0) for e in edges], np.int32))
    return result


def render_paths_from_edges(edges, node_count: int) -> list[list[int]]:
    """centerline_tree.py:280-304: one node path per branch between junctions (every path starts at a non-degree-2
    node and walks the degree-2 chain); a pure cycle without a junction is left out."""
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for edge_index, (a, b) in enumerate(edges):
        adjacency[int(a)].append((int(b), edge_index))
        adjacency[int(b)].append((int(a), edge_index))
    paths, visited = [], set()
    for node, neighbors in enumerate(adjacency):
        if len(neighbors) == 2:
            continue
        for neighbor, edge_index in neighbors:
            if edge_index in visited:
                continue
            path = [node, neighbor]
            visited.add(edge_index)
            previous_edge, current = edge_index, neighbor
            while len(adjacency[current]) == 2:
                nxt = next(item for item in adjacency[current] if item[1] != previous_edge)
                if nxt[1] in visited:
                    break
                current, previous_edge = nxt
                visited.add(previous_edge)
                path.append(current)
            paths.append(path)
    return paths


def spline_topology(edges: np.ndarray, render_paths) -> tuple[np.ndarray, np.ndarray]:
    """centerline_tree.py:43-95 (``build_spline_edge_topology``): per segment the cardinal-spline controls
    ``(previous, start, end, next)`` along its render path (a missing end repeats the endpoint) and the previous
    and next segment on the path, ``-1`` at a path end. Paths stop at junctions, so no tangent is blended across."""
    S = len(edges)
    controls = np.full((S, 4), -1, dtype=np.int32)
    neighbors = np.full((S, 2), -1, dtype=np.int32)
    lookup = {(min(int(a), int(b)), max(int(a), int(b))): i for i, (a, b) in enumerate(edges)}
    assigned = np.zeros(S, dtype=bool)
    for path in render_paths:
        for k in range(max(0, len(path) - 1)):
            a, b = int(path[k]), int(path[k + 1])
            e = lookup[(min(a, b), max(a, b))]
            if assigned[e]:
                raise ValueError(f"centreline segment {(a, b)} occurs in two render paths")
            assigned[e] = True
            controls[e] = (int(path[k - 1]) if k > 0 else a, a, b, int(path[k + 2]) if k + 2 < len(path) else b)
            if k > 0:
                neighbors[e, 0] = lookup[tuple(sorted((int(path[k - 1]), a)))]
            if k + 2 < len(path):
                neighbors[e, 1] = lookup[tuple(sorted((b, int(path[k + 2]))))]
    if not assigned.all():
        raise ValueError(f"centreline render paths do not cover segments {np.flatnonzero(~assigned).tolist()}")
    return controls, neighbors


# --- frames and pairs (pbd_cosserat.py:485-661) ----------------------------------------------------------------


def _quat_axis_angle(axis, angle):
    h = 0.5 * angle
    s = math.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)], dtype=np.float64)


def _quat_from_matrix(m):  # pbd_cosserat.py:503-521
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _frame_from_tangent(z):  # pbd_cosserat.py:524-534 without the reference override
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, -1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return _quat_from_matrix(np.column_stack((x, y, z)))


def _quat_between(t_from, t_to):  # pbd_cosserat.py:563-576
    c = np.cross(t_from, t_to)
    s = float(np.linalg.norm(c))
    d = float(np.dot(t_from, t_to))
    if s < 1.0e-8:
        if d > 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0])
        perp = np.cross(t_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1.0e-6:
            perp = np.cross(t_from, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        return _quat_axis_angle(perp, math.pi)
    return _quat_axis_angle(c / s, math.atan2(s, d))


def tree_frames(positions, edges, root: int) -> np.ndarray:
    """pbd_cosserat.py:579-616 (``_tree_frames``): per-segment frames parallel transported over a BFS spanning tree
    from the root's lowest incident segment; a symmetric fork gets mirror-symmetric branch frames."""
    P = np.asarray(positions, dtype=np.float64)
    tang = [(P[b] - P[a]) / np.linalg.norm(P[b] - P[a]) for a, b in edges]
    node_edges = defaultdict(list)
    for e, (a, b) in enumerate(edges):
        node_edges[a].append(e)
        node_edges[b].append(e)
    frames = [None] * len(edges)
    visited = set()

    def bfs(seed):
        frames[seed] = _frame_from_tangent(tang[seed])
        visited.add(seed)
        dq = deque([seed])
        while dq:
            e = dq.popleft()
            for node in edges[e]:
                for f in node_edges[node]:
                    if f in visited:
                        continue
                    qf = _qmul_np(_quat_between(tang[e], tang[f]), frames[e])
                    frames[f] = qf / np.linalg.norm(qf)
                    visited.add(f)
                    dq.append(f)

    incident = node_edges[int(root)]
    if incident:
        bfs(min(incident))
    for e in range(len(edges)):  # a disconnected component gets its own seed (the loader rejects one anyway)
        if e not in visited:
            bfs(e)
    return np.asarray(frames, dtype=np.float32)


def derive_bend_pairs(edges, root: int) -> np.ndarray:
    """pbd_cosserat.py:629-661: each segment is bend-coupled to its parent in a BFS spanning tree from ``root``."""
    adj = defaultdict(list)
    for e, (a, b) in enumerate(edges):
        adj[int(a)].append((e, int(b)))
        adj[int(b)].append((e, int(a)))
    parent_edge, visited, pairs = {int(root): -1}, {int(root)}, set()
    dq = deque([int(root)])
    while dq:
        node = dq.popleft()
        pe = parent_edge[node]
        incident = [e for e, _ in adj[node]]
        local_parent = pe if pe != -1 else (min(incident) if incident else -1)
        for e, nb in adj[node]:
            if e == pe:
                continue
            if local_parent != -1 and e != local_parent:
                pairs.add((min(local_parent, e), max(local_parent, e)))
            if nb not in visited:
                visited.add(nb)
                parent_edge[nb] = e
                dq.append(nb)
    return np.asarray(sorted(pairs), dtype=np.int32).reshape(-1, 2)


def root_arc_lengths(positions, edges, root: int) -> np.ndarray:
    """centerline_controls.py:103-123: geodesic arc length of every node from the root over the graph."""
    P = np.asarray(positions, dtype=np.float64)
    adj = defaultdict(list)
    for a, b in edges:
        L = float(np.linalg.norm(P[b] - P[a]))
        adj[int(a)].append((int(b), L))
        adj[int(b)].append((int(a), L))
    dist = np.full(len(P), np.inf)
    dist[root] = 0.0
    heap = [(0.0, int(root))]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for other, L in adj[node]:
            nd = d + L
            if nd < dist[other]:
                dist[other] = nd
                heapq.heappush(heap, (nd, other))
    return dist.astype(np.float32)


# --- bodies ---------------------------------------------------------------------------------------------------


def add_centreline_tree(scene: Scene, name: str, positions, edges, radii, root: int, *, pinned_nodes=(), pinned_segments=(),
                        mass: float = 0.01, inertia: float = 0.01, render_paths=None, open_root: bool = False, visual=None) -> Body:
    """A body of ``P`` graph nodes with a ``CentrelineTree`` record. ``mass`` and ``inertia`` are per node and per
    segment (omniendo ``node_mass``, ``segment_inertia``); pinned nodes and segments get zero inverse mass. With
    ``open_root`` the root must be a degree-one endpoint (centerline_runtime.py:206-220)."""
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    edges = np.ascontiguousarray(edges, dtype=np.int32).reshape(-1, 2)
    P, S = len(positions), len(edges)
    if P < 2 or S < 1:
        raise ValueError(f"centreline {name!r}: at least two nodes and one segment")
    if float(mass) <= 0.0 or float(inertia) <= 0.0:
        raise ValueError(f"centreline {name!r}: mass and inertia must be positive")
    root = int(root)
    degree = np.bincount(edges.reshape(-1), minlength=P)
    root_neighbor = -1
    if open_root:
        if degree[root] != 1:
            raise ValueError(f"centreline {name!r}: open_root needs a degree-one root, node {root} has degree {int(degree[root])}")
        e = edges[int(np.flatnonzero(np.any(edges == root, axis=1))[0])]
        root_neighbor = int(e[1] if int(e[0]) == root else e[0])
    if render_paths is None:  # a chain or a tree given as edges only
        render_paths = render_paths_from_edges(edges, P)
    frames = tree_frames(positions, edges, root)
    pairs = derive_bend_pairs(edges, root)
    controls, neighbors = spline_topology(edges, render_paths)
    rest_len = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1).astype(np.float32)
    inv_mass = np.full(P, 1.0 / float(mass), np.float32)
    inv_mass[np.asarray(list(pinned_nodes), dtype=np.int64)] = 0.0
    inv_mass_q = np.full(S, 1.0 / float(inertia), np.float32)
    inv_mass_q[np.asarray(list(pinned_segments), dtype=np.int64)] = 0.0
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float32), (P,)).copy()
    E = scene.envs
    rest = ParticleRest(q0=np.tile(positions[None], (E, 1, 1)), inv_mass=np.tile(inv_mass[None], (E, 1)),
                        radius=np.tile(radii[None], (E, 1)), flags=np.full((E, P), ACTIVE_INT, np.int32))
    body = scene.add_body(name, rest, {name: Part(0, 0, P)})
    from types import SimpleNamespace
    body.parts[name].appearance = SimpleNamespace(visual=dict(visual) if visual else None)
    body.anatomy = GraphMetadata()
    body.tree = RodGraph(
        positions=positions[None], edges=edges[None], bend_pairs=pairs[None], frames=frames[None],
        rest_darboux=rest_darboux_quats(frames, pairs)[None], rest_len=rest_len[None], inv_mass_q=inv_mass_q[None],
        radii=radii[None], spline_controls=controls[None], spline_neighbors=neighbors[None],
        root_arc=root_arc_lengths(positions, edges, root)[None], root=root, root_neighbor=root_neighbor,
        render_paths=[list(map(int, p)) for p in render_paths],
    )
    return body


SPHERE_KEYS = {"id", "active", "pid", "position", "rotation", "radius", "stiffness", "falloff", "rest_bend", "bend_stiffness_multiplier",
               "stretch_stiffness_multiplier"}
SPHERE_REQUIRED = ("id", "active", "pid", "position", "rotation", "radius", "stiffness", "falloff")


def load_control_layout(path, node_count: int, vessel: str) -> list[dict]:
    """omniendo xcath/config.py:967-1073 (``load_control_layout``): a schema-1 sidecar ``{schema_version: 1, vessel,
    line_axis: local_z, control_spheres: [...]}``; every sphere needs id, active, pid (< node count), position,
    rotation (xyzw), radius, stiffness and falloff in [0, 1]; optional rest_bend (2), bend/stretch stiffness
    multipliers (>= 0, default 1); duplicate ids and unknown keys are errors. Returns the spheres as plain dicts."""
    import yaml

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"control layout not found: {path}")
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError(f"control layout {path}: expected a mapping")
    unknown = set(root) - {"schema_version", "vessel", "line_axis", "control_spheres"}
    if unknown:
        raise ValueError(f"control layout {path}: unknown keys {sorted(unknown)}")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError(f"control layout {path}: schema_version must be 1")
    if root.get("vessel") != vessel:
        raise ValueError(f"control layout {path}: vessel {root.get('vessel')!r} does not name the body {vessel!r}")
    if root.get("line_axis", "local_z") != "local_z":
        raise ValueError(f"control layout {path}: line_axis must be local_z")
    spheres, ids = [], set()
    for i, s in enumerate(root.get("control_spheres") or []):
        where = f"control layout {path}: control_spheres[{i}]"
        if not isinstance(s, dict):
            raise ValueError(f"{where}: expected a mapping")
        unknown = set(s) - SPHERE_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        missing = [k for k in SPHERE_REQUIRED if k not in s]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        if s["id"] in ids:
            raise ValueError(f"{where}: duplicate id {s['id']!r}")
        ids.add(s["id"])
        pid = int(s["pid"])
        if not 0 <= pid < node_count:
            raise ValueError(f"{where}: pid {pid} outside [0, {node_count})")
        position, rotation = np.asarray(s["position"], np.float64), np.asarray(s["rotation"], np.float64)
        if position.shape != (3,) or rotation.shape != (4,) or not (np.isfinite(position).all() and np.isfinite(rotation).all()):
            raise ValueError(f"{where}: position needs 3 finite values, rotation 4 (xyzw)")
        stiffness, falloff, radius = float(s["stiffness"]), float(s["falloff"]), float(s["radius"])
        if not (0.0 <= stiffness <= 1.0 and 0.0 <= falloff <= 1.0 and radius >= 0.0):
            raise ValueError(f"{where}: stiffness and falloff must be in [0, 1], radius >= 0")
        rest_bend = np.asarray(s.get("rest_bend", (0.0, 0.0)), np.float64)
        if rest_bend.shape != (2,):
            raise ValueError(f"{where}: rest_bend needs 2 values")
        bend_mult, stretch_mult = float(s.get("bend_stiffness_multiplier", 1.0)), float(s.get("stretch_stiffness_multiplier", 1.0))
        if bend_mult < 0.0 or stretch_mult < 0.0:
            raise ValueError(f"{where}: stiffness multipliers must be >= 0")
        spheres.append({"id": str(s["id"]), "active": bool(s["active"]), "pid": pid, "position": position.tolist(),
                        "rotation": rotation.tolist(), "radius": radius, "stiffness": stiffness, "falloff": falloff,
                        "rest_bend": rest_bend.tolist(), "bend_stiffness_multiplier": bend_mult, "stretch_stiffness_multiplier": stretch_mult})
    return spheres


def add_centreline_from_json(scene: Scene, name: str, path, *, scale, rotation, translation, root_position, weld_epsilon=1.0e-5,
                             stitch_threshold=0.03, radius_mode="mean", radius_scale=0.9, root_locked=True, endpoints_locked=True,
                             open_root=True, mass=0.01, inertia=0.01, control_layouts=(), visual=None, regions=None) -> Body:
    """The tree of a vessel asset: built in the file's coordinates (topology tolerances are scale-free, the root is
    the node nearest ``root_position`` mapped back through the vessel transform), then posed like the mesh:
    ``(p * scale) @ R.T + t``, radii scaled by ``min(scale) * radius_scale``. Pins: the root (and its segment) when
    ``root_locked``, every degree-one node when ``endpoints_locked``. ``control_layouts`` are layout sidecar paths (the
    ``control_layout`` system shapes the tree with them, one layout per env round-robin)."""
    from i4h_endoluminal.regions import RegionMembership, vessel_regions
    starts, ends, smin, emin, smax, emax, sources, known = load_centreline_json(path, with_sources=True)
    names, mapping = vessel_regions(regions, known) if regions is not None else (("unassigned",), None)
    if radius_mode == "min":
        sr, er = smin, emin
    elif radius_mode == "max":
        sr, er = smax, emax
    elif radius_mode == "mean":
        sr, er = 0.5 * (smin + smax), 0.5 * (emin + emax)
    else:
        raise ValueError(f"centreline {name!r}: radius_mode must be min, max or mean, got {radius_mode!r}")
    scale = np.asarray(scale, dtype=np.float64) * np.ones(3)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    root_src = ((np.asarray(root_position, dtype=np.float64) - translation) @ rotation) / scale  # centerline_runtime.py:129-137
    pos_src, edges, radii_src, paths, root, edge_sources, edge_regions = build_tree(starts, ends, sr, er, weld_epsilon=weld_epsilon,
                                                        stitch_threshold=stitch_threshold, root_position=root_src, sources=sources, branch_regions=mapping)
    positions = ((pos_src * scale.astype(np.float32)) @ rotation.astype(np.float32).T + translation.astype(np.float32)).astype(np.float32)
    radii = (radii_src * float(np.min(scale)) * float(radius_scale)).astype(np.float32)
    degree = np.bincount(edges.reshape(-1), minlength=len(positions))
    pinned = set()
    pinned_segments = []
    if root_locked:
        pinned.add(root)
        pinned_segments.append(int(np.flatnonzero(np.any(edges == root, axis=1))[0]))  # the root's lowest incident segment
    if endpoints_locked:
        pinned.update(np.flatnonzero(degree == 1).tolist())
    body = add_centreline_tree(scene, name, positions, edges, radii, root, pinned_nodes=sorted(pinned), pinned_segments=pinned_segments,
                               mass=mass, inertia=inertia, render_paths=paths, open_root=open_root, visual=visual)
    body.anatomy.edge_sources = edge_sources
    body.tree.material_radius = (radii_src * float(np.min(scale)))[None]
    if regions is not None:
        edge_regions.flags.writeable = False
        body.regions = RegionMembership(names, edge_regions, edge_sources)
    body.anatomy.layouts = [load_control_layout(p, len(positions), name) for p in (control_layouts or ())]  # layout l serves envs e % L == l
    return body


# --- skinning bind (xcath/vessel_skinning.py:143-198, 363-375) -------------------------------------------------


# Ported from omniendo xcath/vessel_skinning.py:143-172 (bind_vertices_to_edges_kernel). Edits: none.
@wp.kernel
def bind_vertices(
    vertices: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=wp.vec2i),
    edge_count: int,
    bound_edges: wp.array(dtype=wp.int32),
    bound_coordinates: wp.array(dtype=wp.float32),
):
    """Bind to the nearest source-space edge; strict comparison breaks ties by index."""
    vertex_index = wp.tid()
    point = vertices[vertex_index]
    best_edge = int(0)
    best_t = float(0.0)
    best_distance_sq = float(3.402823466e38)
    for edge_index in range(edge_count):
        edge = edges[edge_index]
        a = positions[edge[0]]
        delta = positions[edge[1]] - a
        denominator = wp.dot(delta, delta)
        t = float(0.0)
        if denominator > 1.0e-20:
            t = wp.clamp(wp.dot(point - a, delta) / denominator, 0.0, 1.0)
        distance_sq = wp.dot(point - (a + delta * t), point - (a + delta * t))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_edge = edge_index
            best_t = t
    bound_edges[vertex_index] = best_edge
    bound_coordinates[vertex_index] = best_t


# Ported from omniendo xcath/vessel_skinning.py:175-198 (capture_rest_transforms_kernel). Edits: the rest-vertex
# copy is dropped (the record keeps the points themselves).
@wp.kernel
def capture_rest_transforms(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    edges: wp.array(dtype=wp.vec2i),
    bound_edges: wp.array(dtype=wp.int32),
    bound_coordinates: wp.array(dtype=wp.float32),
    rest_centers: wp.array(dtype=wp.vec3),
    rest_orientations: wp.array(dtype=wp.quat),
):
    vertex_index = wp.tid()
    edge_index = bound_edges[vertex_index]
    edge = edges[edge_index]
    t = bound_coordinates[vertex_index]
    rest_centers[vertex_index] = positions[edge[0]] * (1.0 - t) + positions[edge[1]] * t
    # Cosserat orientations are segment quantities.  Sampling the bound
    # segment directly is essential for cyclic graphs: every authored edge can
    # then drive the surface vertices bound to it.
    rest_orientations[vertex_index] = orientations[edge_index]


def bind_skin(name: str, points: np.ndarray, indices: np.ndarray, tree: RodGraph, device, visual=None) -> SkinMesh:
    """Bind a vessel surface to its tree at rest (on ``device``, at load) and build the one-ring adjacency
    (vessel_skinning.py:363-375). Binding happens in scene coordinates; omniendo bound in source coordinates,
    which is the same nearest-segment relation under a uniform scale."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.int32).reshape(-1, 3)
    n = len(points)
    if n == 0 or len(indices) == 0:
        raise ValueError(f"vessel mesh {name!r}: skinning needs vertices and triangles")
    edges = tree.edges[0]
    verts = wp.array(points, dtype=wp.vec3, device=device)
    positions = wp.array(tree.positions[0], dtype=wp.vec3, device=device)
    frames = wp.array(tree.frames[0], dtype=wp.quat, device=device)
    edges_d = wp.array(edges, dtype=wp.vec2i, device=device)
    bound_edge = wp.zeros(n, dtype=wp.int32, device=device)
    bound_t = wp.zeros(n, dtype=wp.float32, device=device)
    rest_center = wp.zeros(n, dtype=wp.vec3, device=device)
    rest_orientation = wp.zeros(n, dtype=wp.quat, device=device)
    wp.launch(bind_vertices, dim=n, inputs=[verts, positions, edges_d, len(edges)], outputs=[bound_edge, bound_t], device=device)
    wp.launch(capture_rest_transforms, dim=n, inputs=[positions, frames, edges_d, bound_edge, bound_t],
              outputs=[rest_center, rest_orientation], device=device)
    pairs = np.concatenate([indices[:, [0, 1]], indices[:, [1, 2]], indices[:, [2, 0]]])
    pairs = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)  # both directions, sorted by (vertex, neighbour)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    adj_offsets = np.zeros(n + 1, dtype=np.int32)
    adj_offsets[1:] = np.cumsum(np.bincount(pairs[:, 0], minlength=n)).astype(np.int32)
    return SkinMesh(points=points, indices=indices, visual=dict(visual or {}), bound_edge=bound_edge.numpy(), bound_t=bound_t.numpy(),
                    rest_center=rest_center.numpy(), rest_orientation=rest_orientation.numpy(), adj_offsets=adj_offsets,
                    adj_vertices=np.ascontiguousarray(pairs[:, 1], dtype=np.int32))
