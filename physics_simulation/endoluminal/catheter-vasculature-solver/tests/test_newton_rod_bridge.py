# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Newton particle-buffer bridge.

Exercises the marshalling layer that lets the rod solver run behind Newton's
``SolverBase``, without requiring ``newton`` to be installed: the bridge is
deliberately newton-free and reads any object exposing ``particle_q`` /
``particle_qd``.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp


class FakeState:
    """Stand-in for ``newton.State`` exposing only the particle buffers."""

    def __init__(self, num_particles: int, device: str = "cpu", with_velocities: bool = True):
        self.particle_q = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        self.particle_qd = (
            wp.zeros(num_particles, dtype=wp.vec3, device=device) if with_velocities else None
        )


@pytest.fixture
def rod(cpu_rod_config):
    from catheter_vasculature_solver import XPBDRodSolver

    return XPBDRodSolver(cpu_rod_config, floor_z=None)


@pytest.fixture
def full_range(rod):
    from catheter_vasculature_solver import RodParticleRange

    return RodParticleRange(offset=0, count=rod.num_points, num_envs=1)


def test_particle_range_rejects_invalid_geometry():
    from catheter_vasculature_solver import RodParticleRange

    with pytest.raises(ValueError, match="offset"):
        RodParticleRange(offset=-1, count=4)
    with pytest.raises(ValueError, match="count"):
        RodParticleRange(offset=0, count=0)
    with pytest.raises(ValueError, match="num_envs"):
        RodParticleRange(offset=0, count=4, num_envs=0)
    with pytest.raises(ValueError, match="divisible"):
        RodParticleRange(offset=0, count=5, num_envs=2)


def test_particle_range_end():
    from catheter_vasculature_solver import RodParticleRange

    assert RodParticleRange(offset=7, count=25).end == 32


def test_bridge_rejects_count_mismatch(rod):
    from catheter_vasculature_solver import RodParticleBridge, RodParticleRange

    wrong = RodParticleRange(offset=0, count=rod.num_points + 1, num_envs=1)
    with pytest.raises(ValueError, match="does not match the rod's particle count"):
        RodParticleBridge(rod, wrong)


def test_bridge_rejects_num_envs_mismatch(rod):
    from catheter_vasculature_solver import RodParticleBridge, RodParticleRange

    # Same total particle count, but claiming several envs contradicts the
    # single-env rod. Pick a divisor so the range itself stays self-consistent.
    num_envs = next(d for d in range(2, rod.num_points + 1) if rod.num_points % d == 0)
    mismatched = RodParticleRange(offset=0, count=rod.num_points, num_envs=num_envs)
    with pytest.raises(ValueError, match="num_envs"):
        RodParticleBridge(rod, mismatched)


def test_write_to_exports_rod_positions(rod, full_range):
    from catheter_vasculature_solver import RodParticleBridge

    bridge = RodParticleBridge(rod, full_range)
    state = FakeState(rod.num_points)
    bridge.write_to(state)

    exported = state.particle_q.numpy()
    expected = rod.positions.detach().cpu().numpy()
    np.testing.assert_allclose(exported, expected, rtol=0, atol=0)


def test_read_from_imports_state_positions(rod, full_range):
    from catheter_vasculature_solver import RodParticleBridge

    bridge = RodParticleBridge(rod, full_range)
    state = FakeState(rod.num_points)

    shifted = rod.positions.detach().cpu().numpy() + np.array([0.0, 0.25, 0.0], dtype=np.float32)
    state.particle_q.assign(wp.array(shifted, dtype=wp.vec3, device="cpu"))
    bridge.read_from(state)

    np.testing.assert_allclose(rod.positions.detach().cpu().numpy(), shifted, rtol=0, atol=1e-6)


def test_read_from_is_skipped_when_sync_disabled(rod, full_range):
    from catheter_vasculature_solver import RodParticleBridge

    bridge = RodParticleBridge(rod, full_range, sync_from_state=False)
    before = rod.positions.detach().cpu().numpy().copy()

    state = FakeState(rod.num_points)  # all zeros
    bridge.read_from(state)

    np.testing.assert_allclose(rod.positions.detach().cpu().numpy(), before, rtol=0, atol=0)


def test_offset_range_touches_only_its_own_window(rod):
    """A rod sharing Newton's particle array must not disturb its neighbours."""
    from catheter_vasculature_solver import RodParticleBridge, RodParticleRange

    offset = 5
    padding = 3
    total = offset + rod.num_points + padding
    bridge = RodParticleBridge(rod, RodParticleRange(offset=offset, count=rod.num_points))

    state = FakeState(total)
    sentinel = np.full((total, 3), 7.0, dtype=np.float32)
    state.particle_q.assign(wp.array(sentinel, dtype=wp.vec3, device="cpu"))

    bridge.write_to(state)
    written = state.particle_q.numpy()

    np.testing.assert_allclose(written[:offset], sentinel[:offset], rtol=0, atol=0)
    np.testing.assert_allclose(written[offset + rod.num_points :], sentinel[offset + rod.num_points :], rtol=0, atol=0)
    np.testing.assert_allclose(
        written[offset : offset + rod.num_points], rod.positions.detach().cpu().numpy(), rtol=0, atol=0
    )


def test_round_trip_after_stepping_preserves_rod_state(rod, full_range):
    from catheter_vasculature_solver import RodParticleBridge

    bridge = RodParticleBridge(rod, full_range)
    state = FakeState(rod.num_points)

    bridge.write_to(state)
    bridge.read_from(state)
    bridge.advance(1.0 / 240.0)
    bridge.write_to(state)

    exported = state.particle_q.numpy()
    assert np.isfinite(exported).all()
    np.testing.assert_allclose(exported, rod.positions.detach().cpu().numpy(), rtol=0, atol=0)


def test_missing_particle_buffer_raises(rod, full_range):
    from catheter_vasculature_solver import RodParticleBridge

    class Empty:
        particle_q = None

    bridge = RodParticleBridge(rod, full_range)
    with pytest.raises(RuntimeError, match="no 'particle_q' buffer"):
        bridge.write_to(Empty())


def test_range_outside_buffer_raises(rod):
    from catheter_vasculature_solver import RodParticleBridge, RodParticleRange

    bridge = RodParticleBridge(rod, RodParticleRange(offset=4, count=rod.num_points))
    state = FakeState(rod.num_points)  # too short once offset by 4
    with pytest.raises(RuntimeError, match="does not fit in"):
        bridge.write_to(state)


def test_velocities_are_optional(rod, full_range):
    from catheter_vasculature_solver import RodParticleBridge

    bridge = RodParticleBridge(rod, full_range)
    state = FakeState(rod.num_points, with_velocities=False)
    bridge.write_to(state)  # must not raise
    assert np.isfinite(state.particle_q.numpy()).all()


def test_cuda_graph_capture_can_be_disabled(rod):
    """The Newton wrapper disables internal capture; outer runtimes capture instead."""
    rod.set_cuda_graph_enabled(False)
    assert rod._can_use_cuda_graph(1.0 / 240.0) is False
    rod.step(1.0 / 240.0)
    assert rod._cuda_graph is None
