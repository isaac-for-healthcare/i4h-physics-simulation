# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centerline-driven surface skinning (the render surface for viewer / fluoro)."""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
wp.init()

from catheter_vasculature_solver.vessel_deformation import (  # noqa: E402
    CenterlineData,
    CenterlineDynamicsParams,
    CenterlineVesselRuntime,
    VesselSkinner,
    build_centerline_tree,
)

N_SEG = 8
SPACING = 0.05
RADIUS = 0.02
RING_SIDES = 12


def _device_params():
    params = [pytest.param("cpu", id="cpu")]
    try:
        has_cuda = wp.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - no CUDA driver at all
        has_cuda = False
    if has_cuda:
        params.append(pytest.param("cuda:0", id="cuda", marks=pytest.mark.gpu))
    return params


DEVICES = _device_params()


def _chain_tree(n_seg: int = N_SEG, spacing: float = SPACING, radius: float = RADIUS):
    x = np.arange(n_seg + 1, dtype=np.float32) * spacing
    starts = np.stack([x[:-1], np.zeros(n_seg), np.zeros(n_seg)], axis=1).astype(np.float32)
    ends = np.stack([x[1:], np.zeros(n_seg), np.zeros(n_seg)], axis=1).astype(np.float32)
    r = np.full(n_seg, radius, dtype=np.float32)
    return build_centerline_tree(
        CenterlineData(starts, ends, np.zeros(n_seg, dtype=np.int32), r, r, r, r)
    )


def _tube_mesh(tree, radius: float = RADIUS, sides: int = RING_SIDES):
    """Closed triangulated tube: one ring of vertices per centerline node."""
    axis = tree.positions
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([np.zeros(sides), np.cos(angles), np.sin(angles)], axis=1) * radius

    vertices = np.concatenate([node + ring for node in axis]).astype(np.float32)
    triangles = []
    for i in range(len(axis) - 1):
        base, nxt = i * sides, (i + 1) * sides
        for j in range(sides):
            k = (j + 1) % sides
            triangles.append([base + j, nxt + j, nxt + k])
            triangles.append([base + j, nxt + k, base + k])
    return vertices, np.asarray(triangles, dtype=np.int32)


def _runtime(tree, device, **kwargs):
    kwargs.setdefault("catheter_radius", 0.002)
    kwargs.setdefault("max_distance", 1.0)
    return CenterlineVesselRuntime.from_tree(
        tree,
        device=device,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        **kwargs,
    )


def _radii_about_axis(vertices: np.ndarray) -> np.ndarray:
    """Distance of each surface vertex from the (undeformed) X axis."""
    return np.linalg.norm(vertices[:, 1:], axis=1)


# ─────────────────────────────────────────────────────────────────── binding


@pytest.mark.parametrize("device", DEVICES)
def test_skinning_reproduces_rest_surface(device):
    """At rest the skinned surface must equal the input mesh."""
    tree = _chain_tree()
    vertices, triangles = _tube_mesh(tree)
    runtime = _runtime(tree, device)

    runtime.attach_surface(vertices, triangles)

    assert runtime.surface_positions.shape == vertices.shape
    assert np.allclose(runtime.surface_positions, vertices, atol=1.0e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_skinned_normals_point_outward(device):
    tree = _chain_tree()
    vertices, triangles = _tube_mesh(tree)
    runtime = _runtime(tree, device)
    runtime.attach_surface(vertices, triangles)

    normals = runtime.surface_normals
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-4)

    # Radial component: consistently signed, i.e. all in or all out (winding).
    radial = vertices[:, 1:] / np.maximum(np.linalg.norm(vertices[:, 1:], axis=1, keepdims=True), 1e-9)
    alignment = np.sum(normals[:, 1:] * radial, axis=1)
    interior = np.abs(vertices[:, 0] - vertices[:, 0].mean()) < 0.1  # skip open end caps
    assert np.all(alignment[interior] > 0.5) or np.all(alignment[interior] < -0.5)


@pytest.mark.parametrize("device", DEVICES)
def test_vertices_bind_to_nearest_edge(device):
    tree = _chain_tree()
    vertices, triangles = _tube_mesh(tree)
    skinner = VesselSkinner(vertices, triangles, tree.edges, device=device)
    positions = wp.array(np.asarray(tree.positions, dtype=np.float32), dtype=wp.vec3, device=device)
    skinner.bind(positions)

    bound = skinner.bound_edges.numpy()
    assert bound.min() >= 0 and bound.max() < len(tree.edges)
    # Each ring lies on a node, so consecutive rings must not all share one edge.
    assert len(np.unique(bound)) > 1


def test_skinner_rejects_degenerate_input():
    tree = _chain_tree()
    vertices, triangles = _tube_mesh(tree)

    with pytest.raises(ValueError, match="requires vertices"):
        VesselSkinner(np.zeros((0, 3), dtype=np.float32), triangles, tree.edges, device="cpu")
    with pytest.raises(ValueError, match="out of range"):
        bad = triangles.copy()
        bad[0, 0] = len(vertices)
        VesselSkinner(vertices, bad, tree.edges, device="cpu")


# ────────────────────────────────────────────────────────────────── deforming


@pytest.mark.parametrize("device", DEVICES)
def test_surface_follows_deforming_centerline(device):
    """Contact indents the wall, and the render surface must track it."""
    tree = _chain_tree()
    vertices, triangles = _tube_mesh(tree)
    runtime = _runtime(tree, device, two_way=True, vessel_response=1.0, collision_iterations=2)
    runtime.attach_surface(vertices, triangles)
    rest_surface = runtime.surface_positions.copy()

    # Press a tool sample into the wall at mid-span, then commit the step.
    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    tool = wp.array(np.array([[mid_x, 0.05, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    inv = wp.array(np.array([1.0], dtype=np.float32), dtype=wp.float32, device=device)

    runtime.begin_step(tool)
    runtime.predict(1.0e-3)
    runtime.project_constraints()
    runtime.project_containment(tool, inv, num_points=1, num_edges=0)
    runtime.finalize(1.0e-3)

    surface = runtime.surface_positions
    assert np.isfinite(surface).all()
    displacement = np.linalg.norm(surface - rest_surface, axis=1)
    assert displacement.max() > 1.0e-4  # the wall visibly moved

    # Motion is local: vertices near the contact move more than those at the inlet.
    near = np.abs(vertices[:, 0] - mid_x) < SPACING
    far = vertices[:, 0] < SPACING * 0.5
    assert displacement[near].mean() > displacement[far].mean()


@pytest.mark.parametrize("device", DEVICES)
def test_skinning_preserves_wall_thickness(device):
    """Skinning is rigid per bound segment: the tube must not collapse or balloon."""
    tree = _chain_tree()
    vertices, triangles = _tube_mesh(tree)
    runtime = _runtime(tree, device, two_way=True, vessel_response=1.0, collision_iterations=2)
    runtime.attach_surface(vertices, triangles)

    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    tool = wp.array(np.array([[mid_x, 0.05, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    inv = wp.array(np.array([1.0], dtype=np.float32), dtype=wp.float32, device=device)
    for _ in range(10):
        runtime.begin_step(tool)
        runtime.predict(1.0e-3)
        runtime.project_constraints()
        runtime.project_containment(tool, inv, num_points=1, num_edges=0)
        runtime.finalize(1.0e-3)

    surface = runtime.surface_positions.reshape(len(tree.positions), RING_SIDES, 3)
    # Each ring should stay close to a circle of the original radius.
    for ring in surface:
        spread = np.linalg.norm(ring - ring.mean(axis=0), axis=1)
        assert np.allclose(spread, RADIUS, rtol=0.25)


@pytest.mark.parametrize("device", DEVICES)
def test_surface_refreshes_every_finalize(device):
    """finalize() keeps the render buffers in sync without an explicit call."""
    tree = _chain_tree()
    vertices, triangles = _tube_mesh(tree)
    runtime = _runtime(tree, device)
    runtime.attach_surface(vertices, triangles)

    calls = []
    original = runtime.skinner.update

    def counting(positions, orientations):
        calls.append(1)
        return original(positions, orientations)

    runtime.skinner.update = counting

    runtime.begin_step()
    runtime.predict(1.0e-3)
    runtime.project_constraints()
    runtime.finalize(1.0e-3)
    assert len(calls) == 1


def test_surface_is_absent_until_attached():
    tree = _chain_tree()
    runtime = _runtime(tree, "cpu")
    assert runtime.skinner is None
    assert runtime.surface_positions is None
    assert runtime.surface_normals is None
    runtime.update_surface()  # no-op, must not raise
