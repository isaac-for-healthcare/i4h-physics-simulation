# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Uses vessel_deformation.centerline_data (no YAML scene schema).

"""Reusable vascular centerline graph construction.

Connectivity is deliberately built in source coordinates.  The authored vessel
transform is applied only after welding and component stitching, so topology
tolerances do not change when a scene is scaled.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .centerline_data import CenterlineData, SceneConfigError, VesselConfig, rotation_matrix_degrees


@dataclass
class CenterlineTree:
    positions: np.ndarray
    edges: np.ndarray
    radii: np.ndarray
    render_paths: list[list[int]]
    root: int
    root_segment: int | None
    edge_branch_ids: np.ndarray | None = None
    render_path_branch_ids: list[int] = field(default_factory=list)


def build_spline_edge_topology(
    tree: CenterlineTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Build path-local cardinal-spline controls for every tree edge.

    Controls are ordered ``(previous, start, end, next)`` in render-path
    direction.  A missing endpoint control repeats the endpoint; the GPU
    evaluator recognizes it through the corresponding ``-1`` neighboring-edge
    index and extrapolates the ghost position.  Keeping render paths separate
    deliberately prevents tangents from being blended across branch junctions.
    """
    edge_count = len(tree.edges)
    controls = np.full((edge_count, 4), -1, dtype=np.int32)
    neighbors = np.full((edge_count, 2), -1, dtype=np.int32)
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_index, (a, b) in enumerate(np.asarray(tree.edges, dtype=np.int32)):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key in edge_lookup:
            raise ValueError(f"duplicate centerline edge {key}")
        edge_lookup[key] = edge_index

    assigned = np.zeros(edge_count, dtype=np.bool_)
    for path in tree.render_paths:
        for path_edge in range(max(0, len(path) - 1)):
            a = int(path[path_edge])
            b = int(path[path_edge + 1])
            key = (min(a, b), max(a, b))
            edge_index = edge_lookup.get(key)
            if edge_index is None:
                raise ValueError(f"render path references missing edge {key}")
            if assigned[edge_index]:
                raise ValueError(f"centerline edge {key} occurs in multiple render paths")
            assigned[edge_index] = True
            controls[edge_index] = (
                int(path[path_edge - 1]) if path_edge > 0 else a,
                a,
                b,
                int(path[path_edge + 2]) if path_edge + 2 < len(path) else b,
            )
            if path_edge > 0:
                previous_key = tuple(sorted((int(path[path_edge - 1]), a)))
                neighbors[edge_index, 0] = edge_lookup[previous_key]
            if path_edge + 2 < len(path):
                next_key = tuple(sorted((b, int(path[path_edge + 2]))))
                neighbors[edge_index, 1] = edge_lookup[next_key]

    missing = np.flatnonzero(~assigned)
    if len(missing):
        raise ValueError(
            "render paths do not cover centerline edges "
            + ", ".join(str(int(index)) for index in missing)
        )
    return controls, neighbors


def rooted_node_segments(tree: CenterlineTree) -> np.ndarray:
    """Map each node to one shared material frame in a connected graph.

    The root uses its configured segment, or its first incident segment when
    none is configured.  Every other node uses the segment through which a
    deterministic breadth-first traversal first reached it.  In particular,
    a junction has one frame regardless of which outgoing edge is being
    skinned; edges outside the traversal's spanning tree remain part of the
    centerline graph.
    """
    count = len(tree.positions)
    if count == 0:
        raise ValueError("centerline graph must be non-empty")
    if len(tree.edges) == 0:
        raise ValueError("centerline graph must contain at least one segment")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    for edge_index, (a, b) in enumerate(np.asarray(tree.edges, dtype=np.int32)):
        adjacency[int(a)].append((int(b), edge_index))
        adjacency[int(b)].append((int(a), edge_index))
    result = np.full(count, -1, dtype=np.int32)
    queue = deque([int(tree.root)])
    visited = {int(tree.root)}
    while queue:
        node = queue.popleft()
        for other, edge_index in sorted(adjacency[node]):
            if other in visited:
                continue
            visited.add(other)
            result[other] = edge_index
            queue.append(other)
    if len(visited) != count:
        raise ValueError("centerline graph must be connected")
    root_segment = tree.root_segment
    if root_segment is None and adjacency[tree.root]:
        root_segment = adjacency[tree.root][0][1]
    if root_segment is None:
        raise ValueError("centerline graph must contain at least one segment")
    result[tree.root] = int(root_segment)
    return result


def _components(edges: list[tuple[int, int]], count: int) -> list[list[int]]:
    adjacency = [[] for _ in range(count)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen: set[int] = set()
    result: list[list[int]] = []
    for start in range(count):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        result.append(component)
    return result


def _endpoint_radii(data: CenterlineData, mode: str) -> tuple[np.ndarray, np.ndarray]:
    minimum = (data.start_radius_min, data.end_radius_min)
    maximum = (data.start_radius_max, data.end_radius_max)
    if minimum[0] is None or minimum[1] is None or maximum[0] is None or maximum[1] is None:
        raise SceneConfigError(
            "vessel centerline.dynamics: radius_min and radius_max data are required"
        )
    if mode == "min":
        return np.asarray(minimum[0]), np.asarray(minimum[1])
    if mode == "max":
        return np.asarray(maximum[0]), np.asarray(maximum[1])
    return (
        0.5 * (np.asarray(minimum[0]) + np.asarray(maximum[0])),
        0.5 * (np.asarray(minimum[1]) + np.asarray(maximum[1])),
    )


def transform_tree(tree: CenterlineTree, vessel: VesselConfig) -> CenterlineTree:
    """Return ``tree`` in the vessel's authored pose without changing topology."""
    scale = np.asarray(vessel.transform.scale, dtype=np.float32)
    rotation = rotation_matrix_degrees(vessel.transform.rotation_euler_degrees)
    translation = np.asarray(vessel.transform.translation, dtype=np.float32)
    positions = ((tree.positions * scale) @ rotation.T + translation).astype(np.float32)
    # Circular tubes cannot express anisotropic cross-sections; use the same
    # conservative convention as the upstream static centerline renderer.
    radii = (tree.radii * float(np.min(scale))).astype(np.float32)
    return CenterlineTree(
        positions, tree.edges.copy(), radii, [p.copy() for p in tree.render_paths],
        tree.root, tree.root_segment,
    )


def build_centerline_tree(
    data: CenterlineData,
    *,
    weld_epsilon: float = 1.0e-5,
    stitch_threshold: float = 0.03,
    radius_mode: str = "mean",
    root_position: np.ndarray | None = None,
    require_connected: bool = True,
) -> CenterlineTree:
    """Weld branch samples, stitch geometric gaps, and derive render paths."""
    starts = np.asarray(data.starts, dtype=np.float64)
    ends = np.asarray(data.ends, dtype=np.float64)
    start_radii, end_radii = _endpoint_radii(data, radius_mode)
    node_of: dict[tuple[int, int, int], int] = {}
    positions: list[np.ndarray] = []
    radius_samples: dict[int, list[float]] = defaultdict(list)
    edge_branch_by_key: dict[tuple[int, int], int] = {}

    def node_id(point: np.ndarray) -> int:
        key = tuple(np.rint(point / weld_epsilon).astype(np.int64).tolist())
        if key not in node_of:
            node_of[key] = len(positions)
            positions.append(point.copy())
        return node_of[key]

    edge_set: set[tuple[int, int]] = set()
    for i, (start, end) in enumerate(zip(starts, ends)):
        a, b = node_id(start), node_id(end)
        for node, radius in ((a, start_radii[i]), (b, end_radii[i])):
            if np.isfinite(radius) and radius > 0.0:
                radius_samples[node].append(float(radius))
        if a != b:
            key = (min(a, b), max(a, b))
            edge_set.add(key)
            edge_branch_by_key.setdefault(key, int(data.branch_ids[i]))

    pos = np.asarray(positions, dtype=np.float64)
    edges = sorted(edge_set)
    components = sorted(_components(edges, len(pos)), key=len, reverse=True)
    connected = set(components[0]) if components else set()
    # Recompute the nearest pair against the growing connected component.  A
    # component that cannot be stitched is not silently declared connected.
    pending = [set(component) for component in components[1:]]
    while pending:
        best: tuple[float, int, int, int] | None = None
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
        _, a, b, component_index = best
        edges.append((min(a, b), max(a, b)))
        edge_branch_by_key[(min(a, b), max(a, b))] = -1
        connected.update(pending.pop(component_index))

    used = sorted({node for edge in edges for node in edge})
    remap = {old: new for new, old in enumerate(used)}
    positions_out = pos[used].astype(np.float32)
    edges_out = [(remap[a], remap[b]) for a, b in edges]
    edge_branch_ids = np.asarray(
        [edge_branch_by_key.get((min(a, b), max(a, b)), -1) for a, b in edges],
        dtype=np.int32,
    )
    components_out = _components(edges_out, len(used))
    if require_connected and len(components_out) != 1:
        raise SceneConfigError(
            f"vessel centerline.dynamics: graph is disconnected ({len(components_out)} components)"
        )

    means = {node: float(np.mean(values)) for node, values in radius_samples.items() if values}
    if not means:
        raise SceneConfigError("vessel centerline.dynamics: no positive radius data")
    fallback = float(np.median(list(means.values())))
    radii = np.asarray([means.get(old, fallback) for old in used], dtype=np.float32)

    adjacency: list[list[tuple[int, int]]] = [[] for _ in used]
    for edge_index, (a, b) in enumerate(edges_out):
        adjacency[a].append((b, edge_index))
        adjacency[b].append((a, edge_index))
    paths: list[list[int]] = []
    visited: set[int] = set()
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

    target = positions_out[0] if root_position is None else np.asarray(root_position)
    root = int(np.argmin(np.linalg.norm(positions_out - target, axis=1)))
    root_segment = adjacency[root][0][1] if adjacency[root] else None
    edge_lookup = {
        (min(int(a), int(b)), max(int(a), int(b))): index
        for index, (a, b) in enumerate(edges_out)
    }
    render_path_branch_ids = []
    for path in paths:
        edge_index = edge_lookup[(min(path[0], path[1]), max(path[0], path[1]))]
        render_path_branch_ids.append(int(edge_branch_ids[edge_index]))
    return CenterlineTree(
        positions_out,
        np.asarray(edges_out, dtype=np.int32),
        radii,
        paths,
        root,
        root_segment,
        edge_branch_ids=edge_branch_ids,
        render_path_branch_ids=render_path_branch_ids,
    )


__all__ = [
    "CenterlineTree", "build_centerline_tree", "build_spline_edge_topology",
    "rooted_node_segments", "transform_tree"
]
