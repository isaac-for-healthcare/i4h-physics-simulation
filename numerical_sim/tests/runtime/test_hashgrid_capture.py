# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Capture isolation at the pinned Warp boundary; analytic neighbor sets, no solver noise."""

from typing import Any

import numpy as np
import pytest
import warp as wp


@wp.kernel
def query_neighbors(grid: wp.uint64, points: wp.array(dtype=Any), radius: Any,
                    groups: wp.array(dtype=wp.int32), masks: wp.array(dtype=wp.int32)):
    i = wp.tid()
    query = wp.hash_grid_query(grid, points[i], radius, groups[i])
    j = int(0)
    mask = int(0)
    while wp.hash_grid_query_next(query, j):
        if wp.length(points[i] - points[j]) < radius:
            mask = mask | (1 << j)
    masks[i] = mask


def expected_masks(points, radius, groups=None):
    return np.asarray([sum(1 << j for j, q in enumerate(points)
                           if np.linalg.norm(p - q) < radius and (groups is None or groups[i] == groups[j]))
                       for i, p in enumerate(points)], dtype=np.int32)


def launch_query(grid, points, radius, groups, masks):
    wp.launch(query_neighbors, points.size, [wp.uint64(grid.id), points, grid.dtype(radius), groups, masks], device=grid.device)


@pytest.mark.parametrize("dtype,vector", [(wp.float16, wp.vec3h), (wp.float32, wp.vec3), (wp.float64, wp.vec3d)])
@pytest.mark.parametrize("grouped", [False, True])
def test_distinct_grids_keep_captured_neighbor_sets(dtype, vector, grouped):
    """Two graphs survive later captures and live position/group edits without sharing buckets."""
    records = []
    for shift in (0.0, 4.0):
        positions = np.asarray([[shift, 0, 0], [shift + 0.125, 0, 0]] * 2)
        group_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)
        points = wp.array(positions, dtype=vector, device="cuda:0")
        groups = wp.array(group_ids, dtype=wp.int32, device="cuda:0")
        masks = wp.zeros(4, dtype=wp.int32, device="cuda:0")
        grid = wp.HashGrid(32, 32, 32, dtype=dtype, device="cuda:0")
        grid.reserve(4, with_groups=True)
        grid.build(points, 0.25, groups if grouped else None)
        launch_query(grid, points, 0.25, groups, masks)  # compile before capture
        np.testing.assert_array_equal(masks.numpy(), expected_masks(positions, 0.25, group_ids if grouped else None))
        with wp.ScopedCapture(device="cuda:0") as capture:
            grid.build(points, 0.25, groups if grouped else None)
            launch_query(grid, points, 0.25, groups, masks)
        records.append((grid, points, groups, masks, capture.graph, positions, group_ids))

    for replay in range(3):
        for grid, points, groups, masks, graph, positions, group_ids in records:
            if replay == 1:
                positions[1, 0] += 1.0
                group_ids[:] = [0, 1, 0, 1]
                points.assign(positions)
                groups.assign(group_ids)
            wp.capture_launch(graph)
            np.testing.assert_array_equal(masks.numpy(), expected_masks(positions, 0.25, group_ids if grouped else None))


@pytest.mark.parametrize("one_graph", [False, True])
def test_one_grid_keeps_each_builds_captured_metadata(one_graph):
    """A per-grid host staging buffer is insufficient: each recorded build owns its cell width."""
    positions = np.asarray([[0, 0, 0], [0.125, 0, 0], [0.5, 0, 0], [0.625, 0, 0]], dtype=np.float32)
    points = wp.array(positions, dtype=wp.vec3, device="cuda:0")
    groups = wp.zeros(4, dtype=wp.int32, device="cuda:0")
    grid = wp.HashGrid(32, 32, 32, device="cuda:0")
    grid.reserve(4)
    outputs = [wp.zeros(4, dtype=wp.int32, device="cuda:0") for _ in range(2)]

    def record(i):
        radius = (0.25, 1.0)[i]
        grid.build(points, radius)
        launch_query(grid, points, radius, groups, outputs[i])

    record(0)
    record(1)
    graphs = []
    if one_graph:
        with wp.ScopedCapture(device="cuda:0") as capture:
            record(0)
            record(1)
        graphs.append(capture.graph)
    else:
        for i in range(2):
            with wp.ScopedCapture(device="cuda:0") as capture:
                record(i)
            graphs.append(capture.graph)
    for _ in range(3):
        for graph in graphs:
            wp.capture_launch(graph)
        for radius, output in zip((0.25, 1.0), outputs):
            np.testing.assert_array_equal(output.numpy(), expected_masks(positions, radius))


def test_captured_empty_grid_restores_empty_descriptor():
    """Empty rebuilds stay queryable and empty after a populated graph replays."""
    grid = wp.HashGrid(32, 32, 32, device="cuda:0")
    grid.reserve(4, with_groups=True)
    empty = wp.empty(0, dtype=wp.vec3, device="cuda:0")
    points = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
    groups = wp.zeros(4, dtype=wp.int32, device="cuda:0")
    result = wp.zeros(4, dtype=wp.int32, device="cuda:0")
    grid.build(points, 0.5)
    launch_query(grid, points, 0.25, groups, result)
    with wp.ScopedCapture(device="cuda:0") as empty_capture:
        grid.build(empty, 0.25)
        launch_query(grid, points, 0.25, groups, result)
    with wp.ScopedCapture(device="cuda:0") as full_capture:
        grid.build(points, 0.5)
        launch_query(grid, points, 0.5, groups, result)
    for _ in range(3):
        wp.capture_launch(full_capture.graph)
        np.testing.assert_array_equal(result.numpy(), [15] * 4)
        wp.capture_launch(empty_capture.graph)
        np.testing.assert_array_equal(result.numpy(), [0] * 4)
