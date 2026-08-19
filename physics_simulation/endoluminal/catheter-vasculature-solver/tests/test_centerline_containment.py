# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OmniEndo-sourced centerline Cosserat vessel + tube containment.

These run on CPU so the vendored port is covered in CI, and additionally on
CUDA when a device is present (those cases carry the ``gpu`` marker).
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
wp.init()

from catheter_vasculature_solver.vessel_deformation import (  # noqa: E402
    CenterlineData,
    CenterlineDynamicsParams,
    CenterlineVesselRuntime,
    build_centerline_tree,
)


def _device_params():
    """CPU always; CUDA additionally when the driver exposes a device."""
    params = [pytest.param("cpu", id="cpu")]
    try:
        has_cuda = wp.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - no CUDA driver at all
        has_cuda = False
    if has_cuda:
        params.append(pytest.param("cuda:0", id="cuda", marks=pytest.mark.gpu))
    return params


DEVICES = _device_params()


def _chain_tree(n_seg: int = 8, spacing: float = 0.05, radii: np.ndarray | float = 0.02):
    """Straight chain of ``n_seg`` segments along +X with per-node radii."""
    node_x = np.arange(n_seg + 1, dtype=np.float32) * spacing
    if np.isscalar(radii):
        node_r = np.full(n_seg + 1, float(radii), dtype=np.float32)
    else:
        node_r = np.asarray(radii, dtype=np.float32)
        assert node_r.shape == (n_seg + 1,)

    starts = np.stack([node_x[:-1], np.zeros(n_seg), np.zeros(n_seg)], axis=1).astype(np.float32)
    ends = np.stack([node_x[1:], np.zeros(n_seg), np.zeros(n_seg)], axis=1).astype(np.float32)
    r0, r1 = node_r[:-1], node_r[1:]
    data = CenterlineData(starts, ends, np.zeros(n_seg, dtype=np.int32), r0, r1, r0, r1)
    return build_centerline_tree(data)


def _branching_tree(spacing: float = 0.05, radius: float = 0.02):
    """Y-shaped tree: trunk along +X with a branch leaving the midpoint in +Y."""
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    branch_ids: list[int] = []

    for i in range(4):  # trunk
        starts.append([i * spacing, 0.0, 0.0])
        ends.append([(i + 1) * spacing, 0.0, 0.0])
        branch_ids.append(0)

    fork_x = 2 * spacing  # branch off an interior trunk node
    for i in range(3):
        starts.append([fork_x, i * spacing, 0.0])
        ends.append([fork_x, (i + 1) * spacing, 0.0])
        branch_ids.append(1)

    n = len(starts)
    r = np.full(n, radius, dtype=np.float32)
    data = CenterlineData(
        np.asarray(starts, dtype=np.float32),
        np.asarray(ends, dtype=np.float32),
        np.asarray(branch_ids, dtype=np.int32),
        r,
        r,
        r,
        r,
    )
    return build_centerline_tree(data)


def _runtime(tree, device, **kwargs):
    params = kwargs.pop("params", CenterlineDynamicsParams(iterations=2, root_locked=True))
    kwargs.setdefault("catheter_radius", 0.002)
    kwargs.setdefault("max_distance", 1.0)
    return CenterlineVesselRuntime.from_tree(tree, device=device, params=params, **kwargs)


def _static_vessel(runtime):
    """Freeze the vessel so containment sees predicted == committed positions."""
    wp.copy(runtime.rod.p, runtime.rod.x)


def _contain(runtime, points: np.ndarray, device: str, inv_mass: float = 1.0):
    """Run containment on ``points`` and return their post-projection positions."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    n = points.shape[0]
    catheter = wp.array(points.copy(), dtype=wp.vec3, device=device)
    inv = wp.array(np.full(n, inv_mass, dtype=np.float32), dtype=wp.float32, device=device)
    runtime.begin_step(catheter)
    _static_vessel(runtime)
    runtime.project_containment(catheter, inv, num_points=n, num_edges=max(n - 1, 0))
    return catheter.numpy()


def _wall_displacement(runtime) -> np.ndarray:
    """Per-node vessel motion this step: predicted minus committed positions.

    ``rod.p`` only becomes meaningful once primed from ``rod.x``, so the
    committed state is the reference a contact correction is measured against.
    """
    return runtime.predicted_positions - runtime.positions


# ─────────────────────────────────────────────────────────── vessel dynamics


@pytest.mark.parametrize("device", DEVICES)
def test_runtime_step_is_finite(device):
    tree = _chain_tree()
    runtime = _runtime(tree, device, two_way=True, vessel_response=0.5)

    assert runtime.rod.n_nodes == len(tree.positions)
    runtime.begin_step()
    runtime.predict(1.0e-3)
    runtime.project_constraints()
    runtime.finalize(1.0e-3)

    positions = runtime.positions
    assert positions.shape == (runtime.rod.n_nodes, 3)
    assert np.isfinite(positions).all()


@pytest.mark.parametrize("device", DEVICES)
def test_unloaded_vessel_holds_rest_shape(device):
    """With no gravity or contact the rest configuration is a fixed point."""
    tree = _chain_tree()
    runtime = _runtime(tree, device, params=CenterlineDynamicsParams(iterations=4, root_locked=True))
    rest = runtime.positions.copy()

    for _ in range(20):
        runtime.begin_step()
        runtime.predict(1.0e-3)
        runtime.project_constraints()
        runtime.finalize(1.0e-3)

    assert np.allclose(runtime.positions, rest, atol=1.0e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_branching_tree_remains_stable(device):
    """Y-shaped trees must stay finite and bounded over many steps."""
    tree = _branching_tree()
    runtime = _runtime(tree, device, params=CenterlineDynamicsParams(iterations=4, root_locked=True))
    assert len(tree.edges) == 7  # 4 trunk + 3 branch, welded at the fork

    rest = runtime.positions.copy()
    for _ in range(50):
        runtime.begin_step()
        runtime.predict(1.0e-3)
        runtime.project_constraints()
        runtime.finalize(1.0e-3)

    positions = runtime.positions
    assert np.isfinite(positions).all()
    assert np.abs(positions - rest).max() < 1.0e-3


# ──────────────────────────────────────────────────────────────── containment


@pytest.mark.parametrize("device", DEVICES)
def test_containment_pulls_outside_point_inward(device):
    tree = _chain_tree(radii=0.02)
    runtime = _runtime(tree, device, two_way=False, collision_iterations=4)

    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    outside = np.array([[mid_x, 0.08, 0.0]], dtype=np.float32)

    after = _contain(runtime, outside, device)[0]
    assert np.isfinite(after).all()
    assert abs(after[1]) < abs(outside[0, 1])


@pytest.mark.parametrize("device", DEVICES)
def test_containment_leaves_interior_point_untouched(device):
    """A sample well inside the lumen must not be corrected at all."""
    tree = _chain_tree(radii=0.02)
    runtime = _runtime(tree, device, collision_iterations=4)

    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    inside = np.array([[mid_x, 0.001, 0.0]], dtype=np.float32)

    after = _contain(runtime, inside, device)[0]
    assert np.allclose(after, inside[0], atol=1.0e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_containment_respects_radius_taper(device):
    """Same radial offset: inside the wide end, contained at the narrow end."""
    n_seg = 8
    radii = np.linspace(0.03, 0.008, n_seg + 1).astype(np.float32)
    tree = _chain_tree(n_seg=n_seg, radii=radii)
    offset = 0.02  # between the narrow and wide radius

    wide = np.array([[float(tree.positions[1, 0]), offset, 0.0]], dtype=np.float32)
    narrow = np.array([[float(tree.positions[n_seg - 1, 0]), offset, 0.0]], dtype=np.float32)

    runtime = _runtime(tree, device, collision_iterations=4)
    wide_after = _contain(runtime, wide, device)[0]
    narrow_after = _contain(runtime, narrow, device)[0]

    assert np.allclose(wide_after, wide[0], atol=1.0e-7)
    assert abs(narrow_after[1]) < offset


@pytest.mark.parametrize("device", DEVICES)
def test_open_root_skips_upstream_samples(device):
    """The root cap is open: samples behind the inlet are not projected."""
    tree = _chain_tree(radii=0.02)
    upstream = np.array([[-0.05, 0.08, 0.0]], dtype=np.float32)

    capped = _runtime(tree, device, collision_iterations=4)
    assert capped.open_root >= 0 and capped.open_root_neighbor >= 0
    assert np.allclose(_contain(capped, upstream, device)[0], upstream[0], atol=1.0e-7)

    # open_root=-1 disables the inlet test, so the same sample is now contained.
    closed = _runtime(tree, device, collision_iterations=4, open_root=-1, open_root_neighbor=-1)
    assert not np.allclose(_contain(closed, upstream, device)[0], upstream[0], atol=1.0e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_catheter_max_delta_clamps_correction(device):
    tree = _chain_tree(radii=0.02)
    max_delta = 1.0e-3
    runtime = _runtime(tree, device, collision_iterations=1, catheter_max_delta=max_delta)

    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    outside = np.array([[mid_x, 0.08, 0.0]], dtype=np.float32)

    after = _contain(runtime, outside, device)[0]
    assert np.linalg.norm(after - outside[0]) <= max_delta + 1.0e-6


# ─────────────────────────────────────────────────────────── two-way coupling


@pytest.mark.parametrize("device", DEVICES)
def test_one_way_contact_leaves_vessel_fixed(device):
    tree = _chain_tree(radii=0.02)
    runtime = _runtime(tree, device, two_way=False, collision_iterations=2)

    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    _contain(runtime, np.array([[mid_x, 0.08, 0.0]], dtype=np.float32), device)

    assert np.allclose(_wall_displacement(runtime), 0.0, atol=1.0e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_two_way_contact_displaces_vessel(device):
    tree = _chain_tree(radii=0.02)
    runtime = _runtime(tree, device, two_way=True, vessel_response=0.5, collision_iterations=2)

    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    _contain(runtime, np.array([[mid_x, 0.08, 0.0]], dtype=np.float32), device)

    delta = _wall_displacement(runtime)
    assert np.isfinite(delta).all()
    # The wall yields outward, i.e. away from the lumen axis toward the tool.
    assert delta[:, 1].max() > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_vessel_response_splits_correction(device):
    """Higher vessel_response ⇒ vessel absorbs more, catheter moves less."""
    tree = _chain_tree(radii=0.02)
    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    outside = np.array([[mid_x, 0.08, 0.0]], dtype=np.float32)

    catheter_shift = {}
    vessel_shift = {}
    for response in (0.1, 0.9):
        runtime = _runtime(
            tree, device, two_way=True, vessel_response=response, collision_iterations=1
        )
        after = _contain(runtime, outside, device)[0]
        catheter_shift[response] = np.linalg.norm(after - outside[0])
        vessel_shift[response] = np.abs(_wall_displacement(runtime)).max()

    assert catheter_shift[0.9] < catheter_shift[0.1]
    assert vessel_shift[0.9] > vessel_shift[0.1]


@pytest.mark.parametrize("device", DEVICES)
def test_locked_vessel_nodes_absorb_no_correction(device):
    """Kinematic anchors (inv_mass == 0) must never be displaced by contact."""
    tree = _chain_tree(radii=0.02)
    runtime = _runtime(
        tree,
        device,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True, endpoints_locked=True),
        two_way=True,
        vessel_response=1.0,
        collision_iterations=2,
    )
    inv_mass = runtime.rod.inv_mass.numpy()
    locked = np.flatnonzero(inv_mass == 0.0)
    assert locked.size >= 2  # root + distal endpoint

    # Press on the locked distal end as well as the compliant middle.
    samples = np.array(
        [[float(tree.positions[-1, 0]), 0.08, 0.0], [float(tree.positions[4, 0]), 0.08, 0.0]],
        dtype=np.float32,
    )
    _contain(runtime, samples, device)

    assert np.allclose(_wall_displacement(runtime)[locked], 0.0, atol=1.0e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_vessel_max_delta_clamps_wall_motion(device):
    tree = _chain_tree(radii=0.02)
    max_delta = 5.0e-4
    runtime = _runtime(
        tree,
        device,
        two_way=True,
        vessel_response=1.0,
        collision_iterations=2,
        vessel_max_delta=max_delta,
    )

    mid_x = float(tree.positions[len(tree.positions) // 2, 0])
    _contain(runtime, np.array([[mid_x, 0.08, 0.0]], dtype=np.float32), device)

    shift = np.linalg.norm(_wall_displacement(runtime), axis=1)
    assert shift.max() <= max_delta + 1.0e-6
