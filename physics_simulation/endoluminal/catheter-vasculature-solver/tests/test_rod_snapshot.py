# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-solving a substep: the rod save and restore, and the proxy root it iterates with.

Coupling iterations are only as good as the restore between them. Anything the
snapshot misses advances once per iteration instead of once per substep, so
these tests demand a bit-identical re-step, including an attached vessel that
deforms under the catheter.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="needs warp installed")

from catheter_vasculature_solver import CathRodSolver, RodConfig, XPBDRodSolver  # noqa: E402
from catheter_vasculature_solver.isaaclab_integration.rod_snapshot import RodStateSnapshot  # noqa: E402
from catheter_vasculature_solver.vessel_deformation import (  # noqa: E402
    CenterlineData,
    CenterlineDynamicsParams,
    CenterlineVesselRuntime,
    build_centerline_tree,
)

DEVICE = "cpu"
DT = 1.0 / 240.0
ROD_HEIGHT = 0.5
SEGMENT_LENGTH = 0.02
NUM_SEGMENTS = 12


def bare_rod(num_envs: int, gravity=(0.0, 0.0, 0.0)) -> XPBDRodSolver:
    cfg = RodConfig()
    cfg.device = DEVICE
    cfg.geometry.num_segments = 6
    cfg.solver.gravity = gravity
    return XPBDRodSolver(cfg, num_envs=num_envs, floor_z=None)


def rod_in_a_vessel() -> tuple[CathRodSolver, CenterlineVesselRuntime]:
    """A catheter fed into a lumen that bends, so the wall deforms as it is loaded."""
    length = SEGMENT_LENGTH * NUM_SEGMENTS * 1.25
    x = np.linspace(0.0, length, 9).astype(np.float32)
    y = (0.06 * (x / length) ** 2).astype(np.float32)
    z = np.full(9, ROD_HEIGHT, dtype=np.float32)
    radius = np.full(8, 0.02, dtype=np.float32)
    tree = build_centerline_tree(CenterlineData(
        np.stack([x[:-1], y[:-1], z[:-1]], axis=1), np.stack([x[1:], y[1:], z[1:]], axis=1),
        np.zeros(8, dtype=np.int32), radius, radius, radius, radius,
    ))
    runtime = CenterlineVesselRuntime.from_tree(
        tree, device=DEVICE, params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        catheter_radius=0.002, max_distance=0.5,
    )
    cfg = RodConfig()
    cfg.device = DEVICE
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH
    cfg.solver.gravity = (0.0, 0.0, 0.0)
    rod = CathRodSolver(
        cfg, collision_mesh=None,
        track_start=np.array([0.0, 0.0, ROD_HEIGHT], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32), track_length=0.5,
        tip_num_edges=4, particle_radius=0.002, segment_length=SEGMENT_LENGTH,
        collision_enabled=False, track_enabled=True, centerline_runtime=runtime,
    )
    return rod, runtime


def push(rod, num_envs: int = 1) -> None:
    rod.apply_proximal_control_gpu([0.4] * num_envs, [0.8] * num_envs, DT)
    rod.step(DT)


@pytest.mark.parametrize("num_envs", [1, 2])
def test_restore_reproduces_a_bare_rod_step_exactly(num_envs) -> None:
    rod = bare_rod(num_envs)
    for _ in range(3):
        push(rod, num_envs)
    snapshot = RodStateSnapshot(rod)
    snapshot.save()
    push(rod, num_envs)
    expected = rod.position_array.numpy().copy(), rod.orientation_array.numpy().copy()

    snapshot.restore()
    push(rod, num_envs)

    np.testing.assert_array_equal(rod.position_array.numpy(), expected[0])
    np.testing.assert_array_equal(rod.orientation_array.numpy(), expected[1])


def test_restore_reproduces_a_step_against_a_deforming_vessel_exactly() -> None:
    rod, runtime = rod_in_a_vessel()
    rest = runtime.positions.copy()
    for _ in range(30):
        push(rod)
    assert np.abs(runtime.positions - rest).max() > 0.0, "the wall has to be deforming for this to test anything"

    snapshot = RodStateSnapshot(rod)
    snapshot.save()
    for _ in range(3):
        push(rod)
    expected = rod.position_array.numpy().copy(), runtime.positions.copy(), rod.contact_depth

    snapshot.restore()
    for _ in range(3):
        push(rod)

    np.testing.assert_array_equal(rod.position_array.numpy(), expected[0])
    np.testing.assert_array_equal(runtime.positions, expected[1])
    assert rod.contact_depth == expected[2]


def test_restore_rebinds_buffers_a_step_swapped() -> None:
    first = wp.array([1.0], dtype=wp.float32, device=DEVICE)
    second = wp.array([2.0], dtype=wp.float32, device=DEVICE)
    holder = SimpleNamespace(current=first, spare=second, count=3)
    snapshot = RodStateSnapshot(holder)
    snapshot.save()

    holder.current, holder.spare, holder.count = holder.spare, holder.current, 4
    holder.current.fill_(7.0)
    snapshot.restore()

    assert holder.current is first and holder.spare is second and holder.count == 3
    assert first.numpy()[0] == 1.0 and second.numpy()[0] == 2.0


def test_restore_before_save_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="before save"):
        RodStateSnapshot(bare_rod(1)).restore()


def test_a_proxy_root_carries_the_holder_mass_but_stays_held_for_the_hooks() -> None:
    rod = bare_rod(2, gravity=(0.0, 0.0, -9.81))
    stride = rod.num_points
    roots = rod.position_array.numpy()[::stride].copy()
    orientations = rod.orientation_array.numpy()[::stride].copy()

    workspace = rod._bws

    rod.set_root_proxy_gpu(roots, [[0.1, 0.0, 0.0]] * 2, [0.5, 0.25], orientations)
    rod.step(DT)

    np.testing.assert_allclose(workspace.inv_masses.numpy()[::stride], [0.5, 0.25])
    np.testing.assert_array_equal(rod.hook_inv_masses.numpy()[::stride], [0.0, 0.0])
    np.testing.assert_allclose(workspace.forces.numpy()[::stride], [[0.0, 0.0, 9.81 / 0.5], [0.0, 0.0, 9.81 / 0.25]],
                               rtol=1e-6)
    moved = rod.position_array.numpy()[::stride] - roots
    assert np.all(moved[:, 0] > 0.0), "the proxy root should travel with the holder's velocity"

    rod.set_root_pose_gpu(roots, orientations)
    rod.step(DT)

    np.testing.assert_array_equal(workspace.inv_masses.numpy()[::stride], [0.0, 0.0])
    assert rod.hook_inv_masses is workspace.inv_masses
    assert not workspace.forces.numpy().any()


def test_a_proxy_root_does_not_fall_under_gravity() -> None:
    """Its weight is the holder's, already in the rigid solve; the rod must not add it again."""
    rod = bare_rod(1, gravity=(0.0, 0.0, -9.81))
    root = rod.position_array.numpy()[:1].copy()
    orientation = rod.orientation_array.numpy()[:1].copy()
    pinned = bare_rod(1, gravity=(0.0, 0.0, -9.81))

    rod.set_root_proxy_gpu(root, [[0.0, 0.0, 0.0]], [1.0e-6], orientation)
    rod.step(DT)
    pinned.set_root_pose_gpu(root, orientation)
    pinned.step(DT)

    free_fall = 9.81 * DT * DT
    np.testing.assert_allclose(rod.position_array.numpy()[0], pinned.position_array.numpy()[0], atol=free_fall / 20)
