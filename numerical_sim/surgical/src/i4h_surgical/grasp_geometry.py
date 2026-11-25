# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host mesh preparation, performed before capture. All IDs are body-local."""

import heapq

import numpy as np


def patches(rest, triangles, attached):
    edges = np.unique(
        np.sort(
            np.concatenate(
                [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
            ),
            axis=1,
        ),
        axis=0,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    median = float(np.median(lengths))
    adjacency = [[] for _ in rest]
    for (a, b), length in zip(edges, lengths):
        adjacency[a].append((int(b), float(length)))
        adjacency[b].append((int(a), float(length)))
    attached = set(map(int, attached))
    rows = []
    for centre in range(len(rest)):
        distances, queue = {centre: 0.0}, [(0.0, centre)]
        while queue:
            distance, vertex = heapq.heappop(queue)
            if distance != distances[vertex]:
                continue
            for neighbour, weight in adjacency[vertex]:
                d = distance + weight
                if d <= 1.5 * median and d < distances.get(neighbour, np.inf):
                    distances[neighbour] = d
                    heapq.heappush(queue, (d, neighbour))
        rows.append(
            sorted((set(distances) | {v for v, _ in adjacency[centre]}) - attached)
        )
    counts = np.array([len(row) for row in rows], np.int32)
    table = np.full((len(rest), max(1, int(counts.max()))), -1, np.int32)
    for i, row in enumerate(rows):
        table[i, : len(row)] = row
    eligible = np.zeros(len(rest), bool)
    surface_ids = np.unique(triangles)
    eligible[surface_ids] = True
    eligible[list(attached)] = False
    eligible &= counts >= 5
    if attached:
        distance = np.linalg.norm(
            rest[:, None] - rest[sorted(attached)][None], axis=2
        ).min(axis=1)
        eligible &= distance >= 0.01
    return median, table, counts, eligible


def normals(q, triangles):
    result = np.zeros_like(q)
    cross = np.cross(
        q[triangles[:, 1]] - q[triangles[:, 0]], q[triangles[:, 2]] - q[triangles[:, 0]]
    )
    for corner in range(3):
        np.add.at(result, triangles[:, corner], cross)
    length = np.linalg.norm(result, axis=1)
    return result / np.maximum(length[:, None], 1.0e-20), length > 1.0e-12


def select_targets(rest, live, triangles, eligible, trocar, grasp_centre):
    normal, valid = normals(live, triangles)
    toward = trocar - live
    toward /= np.maximum(np.linalg.norm(toward, axis=1)[:, None], 1.0e-20)
    facing = np.sum(normal * toward, axis=1)
    candidates = np.flatnonzero(eligible & valid & (facing >= 0.25))
    if not len(candidates):
        return [], facing
    distance = np.linalg.norm(live[candidates] - grasp_centre, axis=1)
    first = int(candidates[np.lexsort((candidates, distance))[0]])
    candidates = candidates[
        np.linalg.norm(rest[candidates] - rest[first], axis=1) <= 0.03
    ]
    chosen = [first]
    while len(chosen) < min(8, len(candidates)):
        remaining = np.array([v for v in candidates if v not in chosen], np.int32)
        distances = np.linalg.norm(
            rest[remaining, None] - rest[chosen][None], axis=2
        ).min(axis=1)
        chosen.append(int(remaining[np.lexsort((remaining, -distances))[0]]))
    return chosen, facing


def approach_increment(target, normal, sphere, envelope, gap):
    """Translation only, live nearest-sphere feedback; caller handles overshoot."""
    goal = target + normal * (envelope - 0.00025)
    delta = goal - sphere
    limit = 0.001
    if gap < 0.002:
        limit = min(0.0001, max(0.0, gap + 0.00025))
    length = np.linalg.norm(delta)
    return delta * min(1.0, limit / max(length, 1.0e-20))
