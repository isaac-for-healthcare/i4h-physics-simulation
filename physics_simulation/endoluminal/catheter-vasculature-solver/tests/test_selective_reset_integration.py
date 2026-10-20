# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset must preserve surviving environments across the integration boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from catheter_vasculature_solver import RodConfig, RodParticleBridge, RodParticleRange, XPBDRodSolver
from catheter_vasculature_solver.isaaclab_integration.proxy_coupling import DriveReactionCoupler


@pytest.fixture(params=["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def device(request):
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("needs a CUDA device")
    return request.param


def make_rod(device):
    cfg = RodConfig()
    cfg.device = device
    cfg.geometry.num_segments = 4
    cfg.geometry.segment_length = 0.02
    cfg.solver.num_substeps = 1
    cfg.solver.gravity = (0.0, 0.0, 0.0)
    rod = XPBDRodSolver(cfg, num_envs=3, floor_z=None)
    rod.set_cuda_graph_enabled(False)
    return rod


def test_reset_publishes_only_selected_particle_slices(device):
    rod = make_rod(device)
    particle_range = RodParticleRange(offset=2, count=rod.num_points * 3, num_envs=3)
    bridge = RodParticleBridge(rod, particle_range)
    total = particle_range.end + 3
    state = SimpleNamespace(
        particle_q=wp.array(np.full((total, 3), 7.0, dtype=np.float32), dtype=wp.vec3, device=device),
        particle_qd=wp.array(np.full((total, 3), 9.0, dtype=np.float32), dtype=wp.vec3, device=device),
    )
    before_q, before_qd = state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()
    rest = rod.position_array.numpy().copy()
    rod.position_array.assign(rest + 1.0)

    bridge.reset(state, wp.array([False, True, False], dtype=wp.bool, device=device))

    start = particle_range.offset + rod.num_points
    end = start + rod.num_points
    expected_q, expected_qd = before_q.copy(), before_qd.copy()
    expected_q[start:end] = rest[rod.num_points:2 * rod.num_points]
    expected_qd[start:end] = 0.0
    np.testing.assert_array_equal(state.particle_q.numpy(), expected_q)
    np.testing.assert_array_equal(state.particle_qd.numpy(), expected_qd)
    bridge.read_from(state)
    np.testing.assert_array_equal(rod.position_array.numpy(), expected_q[particle_range.offset:particle_range.end])


@pytest.mark.parametrize("kind", ["rate", "pose"])
@pytest.mark.parametrize("borrowed", [False, True], ids=["numpy-input", "warp-input"])
def test_partial_reset_preserves_pending_commands_and_caller_buffers(device, kind, borrowed):
    rod, reference = make_rod(device), make_rod(device)
    rest = rod.position_array.numpy().copy()[::rod.num_points]
    dt = 1.0 / 240.0
    if kind == "rate":
        first = np.array([0.3, 0.6, 0.9], dtype=np.float32)
        second = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        dtypes = (wp.float32, wp.float32)
        method, extra = "apply_proximal_control_gpu", (dt,)
    else:
        first = rest + np.array([0.01, 0.02, 0.03], dtype=np.float32)
        second = np.tile([0.0, 0.0, np.sin(0.1), np.cos(0.1)], (3, 1)).astype(np.float32)
        dtypes = (wp.vec3, wp.quat)
        method, extra = "set_root_pose_gpu", ()
    inputs = [wp.array(value, dtype=dtype, device=device) if borrowed else value.copy()
              for value, dtype in zip((first, second), dtypes)]
    getattr(rod, method)(*inputs, *extra)
    getattr(reference, method)(first, second, *extra)

    rod.reset([0])
    rod.step(dt)
    reference.step(dt)

    roots = rod.position_array.numpy()[::rod.num_points]
    np.testing.assert_array_equal(roots[0], rest[0])
    np.testing.assert_allclose(roots[1:], reference.position_array.numpy()[::rod.num_points][1:], atol=1e-7)
    np.testing.assert_allclose(
        rod._bws.orientations.numpy()[::rod.num_points][1:],
        reference._bws.orientations.numpy()[::rod.num_points][1:], atol=1e-7,
    )
    for value, expected in zip(inputs, (first, second)):
        np.testing.assert_array_equal(value.numpy() if borrowed else value, expected)
    # A new command must reactivate environments masked by the previous reset.
    getattr(rod, method)(*inputs, *extra)
    rod.step(dt)
    assert not np.array_equal(rod.position_array.numpy()[0], rest[0])


def test_repeated_partial_resets_cancel_only_selected_commands(device):
    rod = make_rod(device)
    dt = 1.0 / 240.0
    rod.apply_proximal_control_gpu([0.6, 0.6, 0.6], [0.0, 0.0, 0.0], dt)
    rod.reset([0])
    rod.reset([1])
    rod.reset([])
    rod.step(dt)
    np.testing.assert_allclose(rod.position_array.numpy()[::rod.num_points, 0], [0.0, 0.0, 0.6 * dt], atol=1e-7)


@pytest.mark.parametrize("shared_body", [False, True])
def test_coupler_reset_preserves_other_environments_and_relaxation(device, shared_body):
    bodies = [0, 0, 1] if shared_body else [0, 1, 2]
    coupler = DriveReactionCoupler(
        body_count=3, num_envs=3, root_offset=0, root_stride=5,
        drive_bodies=bodies, relaxation=0.5, device=device,
    )
    state = SimpleNamespace(
        particle_q=wp.zeros(15, dtype=wp.vec3, device=device),
        body_q=wp.array([wp.transform_identity()] * 3, dtype=wp.transform, device=device),
        body_f=wp.zeros(3, dtype=wp.spatial_vector, device=device),
    )
    loads = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], [3, 6, 9, 12, 15, 18]], dtype=np.float32)
    rod = SimpleNamespace(proximal_wrench=lambda out: out.assign(loads))
    coupler.harvest(state, rod)
    coupler.harvest(state, rod)

    coupler.reset(world_mask=wp.array([True, False, False], dtype=wp.bool, device=device))

    expected = np.zeros((3, 6), dtype=np.float32)
    for env in (1, 2):
        expected[bodies[env]] += 0.75 * loads[env]
    np.testing.assert_array_equal(coupler.coupling_forces.numpy(), expected)
    coupler.apply(state)
    np.testing.assert_array_equal(state.body_f.numpy(), expected)
    # Only surviving history should contribute to the next relaxed update.
    loads[:] = 0.0
    coupler.harvest(state, rod)
    np.testing.assert_array_equal(coupler.coupling_forces.numpy(), 0.5 * expected)

    coupler.reset(world_mask=[0, 0, 0])
    np.testing.assert_array_equal(coupler.coupling_forces.numpy(), 0.5 * expected)
    coupler.reset()
    np.testing.assert_array_equal(coupler.coupling_forces.numpy(), np.zeros((3, 6)))
