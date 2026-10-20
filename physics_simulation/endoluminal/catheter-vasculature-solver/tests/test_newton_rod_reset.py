# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-env reset across the Newton boundary.

Isaac Lab resets environments by handing a solver a per-world mask. These cover
the translation from that mask to rod env ids and the republish that keeps a
reset from being undone, all against the newton-free bridge so no ``newton``
install is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
wp.init()

from catheter_vasculature_solver import RodConfig, RodParticleBridge, RodParticleRange, XPBDRodSolver  # noqa: E402

NUM_ENVS = 4
NUM_SEGMENTS = 16


class FakeState:
    """Stand-in for ``newton.State`` exposing only the particle buffers."""

    def __init__(self, num_particles: int, device: str = "cpu"):
        self.particle_q = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        self.particle_qd = wp.zeros(num_particles, dtype=wp.vec3, device=device)


@pytest.fixture
def rod():
    """Four-env rod on CPU, so a mask can select a strict subset."""
    cfg = RodConfig()
    cfg.device = "cpu"
    cfg.geometry.num_segments = NUM_SEGMENTS
    return XPBDRodSolver(cfg, num_envs=NUM_ENVS)


@pytest.fixture
def bridge(rod):
    rng = RodParticleRange(
        offset=0, count=rod.num_points * rod.num_envs, num_envs=rod.num_envs
    )
    return RodParticleBridge(rod, rng)


@pytest.fixture
def state(rod):
    return FakeState(rod.num_points * rod.num_envs)


def _positions(rod) -> np.ndarray:
    """Per-env positions as ``(num_envs, num_points, 3)``."""
    return rod.positions.detach().cpu().numpy().copy()


def _disturb(rod) -> np.ndarray:
    """Drive every env away from rest and return the disturbed positions."""
    rest = _positions(rod)
    push = np.full(rod.num_envs, 0.4, dtype=np.float32)
    rod.apply_proximal_control_gpu(push, np.zeros(rod.num_envs, dtype=np.float32), 1.0 / 240.0)
    for _ in range(25):
        rod.step(1.0 / 240.0)
    moved = _positions(rod)
    assert np.abs(moved - rest).max() > 1.0e-4, (
        "rod did not move, so the reset assertions would be vacuous"
    )
    return moved


def test_mask_resets_only_the_selected_envs(bridge, rod, state):
    """The point of a mask: envs that did not terminate keep their state."""
    rest = _positions(rod)
    moved = _disturb(rod)

    bridge.reset(state, wp.array([1, 0, 0, 1], dtype=wp.int32, device="cpu"))

    after = _positions(rod)
    for env in (0, 3):
        np.testing.assert_allclose(after[env], rest[env], rtol=0, atol=1.0e-6)
    for env in (1, 2):
        np.testing.assert_allclose(after[env], moved[env], rtol=0, atol=0)


def test_reset_republishes_into_the_state(bridge, rod, state):
    """Without the write-back, the next read_from would copy the reset away."""
    rest = _positions(rod)
    _disturb(rod)

    bridge.reset(state, wp.array([1, 1, 1, 1], dtype=wp.int32, device="cpu"))

    published = state.particle_q.numpy().reshape(rod.num_envs, rod.num_points, 3)
    np.testing.assert_allclose(published, rest, rtol=0, atol=1.0e-6)

    # The real hazard is the next step reading stale positions back in.
    bridge.read_from(state)
    np.testing.assert_allclose(_positions(rod), rest, rtol=0, atol=1.0e-6)


def test_all_zero_mask_resets_nothing(bridge, rod, state):
    """An empty selection must not be mistaken for 'reset everything'."""
    moved = _disturb(rod)

    bridge.reset(state, wp.array([0, 0, 0, 0], dtype=wp.int32, device="cpu"))

    np.testing.assert_allclose(_positions(rod), moved, rtol=0, atol=0)


def test_none_mask_resets_every_env(bridge, rod, state):
    rest = _positions(rod)
    _disturb(rod)

    bridge.reset(state, None)

    np.testing.assert_allclose(_positions(rod), rest, rtol=0, atol=1.0e-6)


def test_mask_length_mismatch_is_rejected(bridge, rod, state):
    """Truncating or padding would reset envs the caller never named."""
    with pytest.raises(ValueError, match="one entry per env"):
        bridge.reset(state, wp.array([1, 0], dtype=wp.int32, device="cpu"))


def test_mask_accepts_plain_sequences(bridge, rod, state):
    """Isaac Lab passes warp arrays; scripts and tests should not have to."""
    rest = _positions(rod)
    moved = _disturb(rod)

    bridge.reset(state, [0, 1, 0, 0])

    after = _positions(rod)
    np.testing.assert_allclose(after[1], rest[1], rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(after[0], moved[0], rtol=0, atol=0)


def test_reset_through_the_bridge_also_restores_the_vessel():
    """A reset env must start against an undeformed wall, not last episode's.

    The bridge only knows about particles, so vessel restoration rides on the
    solver's own reset. This checks the composition actually holds across the
    Newton boundary.
    """
    from catheter_vasculature_solver import CathRodSolver
    from catheter_vasculature_solver.vessel_deformation import (
        CenterlineData,
        CenterlineDynamicsParams,
        CenterlineVesselRuntime,
        build_centerline_tree,
    )

    segment_length, num_segments, height, lumen_radius = 0.02, 16, 0.5, 0.02
    length = segment_length * num_segments * 1.25
    x = np.linspace(0.0, length, 13).astype(np.float32)
    y = (0.06 * (x / length) ** 2).astype(np.float32)
    z = np.full(13, height, dtype=np.float32)
    radii = np.full(12, lumen_radius, dtype=np.float32)
    tree = build_centerline_tree(
        CenterlineData(
            np.stack([x[:-1], y[:-1], z[:-1]], axis=1),
            np.stack([x[1:], y[1:], z[1:]], axis=1),
            np.zeros(12, dtype=np.int32),
            radii,
            radii,
            radii,
            radii,
        )
    )
    runtime = CenterlineVesselRuntime.from_tree(
        tree,
        device="cpu",
        num_envs=NUM_ENVS,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        catheter_radius=0.002,
        max_distance=0.5,
        two_way=True,
        vessel_response=1.0,
        catheter_max_delta=0.01,
        vessel_max_delta=0.01,
    )

    cfg = RodConfig()
    cfg.device = "cpu"
    cfg.geometry.num_segments = num_segments
    cfg.geometry.segment_length = segment_length
    solver = CathRodSolver(
        cfg,
        num_envs=NUM_ENVS,
        collision_mesh=None,
        track_start=np.array([0.0, 0.0, height], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.6,
        tip_num_edges=6,
        particle_radius=0.002,
        segment_length=segment_length,
        collision_enabled=False,
        track_enabled=False,
        centerline_runtime=runtime,
    )
    rng = RodParticleRange(
        offset=0, count=solver.num_points * NUM_ENVS, num_envs=NUM_ENVS
    )
    bridge = RodParticleBridge(solver, rng)
    fake = FakeState(solver.num_points * NUM_ENVS)

    rest_wall = runtime.positions_per_env.copy()
    push = np.full(NUM_ENVS, 0.6, dtype=np.float32)
    solver.apply_proximal_control_gpu(push, np.zeros(NUM_ENVS, dtype=np.float32), 1.0 / 240.0)
    for _ in range(40):
        solver.step(1.0 / 240.0)
    deformed = runtime.positions_per_env.copy()
    assert np.abs(deformed - rest_wall).max() > 1.0e-6, "wall never deformed"

    bridge.reset(fake, wp.array([1, 0, 0, 0], dtype=wp.int32, device="cpu"))

    after = runtime.positions_per_env
    np.testing.assert_allclose(after[0], rest_wall[0], rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(after[1], deformed[1], rtol=0, atol=0)


def test_solver_reset_signature_matches_isaac_lab_call():
    """Isaac Lab calls ``solver.reset(state, world_mask=..., flags=0)``.

    Drift here would surface as a TypeError deep inside the manager's reset
    path, so pin the signature where the failure message is obvious.
    """
    pytest.importorskip("newton.solvers", reason="needs a working newton install")
    import inspect

    from catheter_vasculature_solver.newton_rod_solver import NewtonCathRodSolver

    params = inspect.signature(NewtonCathRodSolver.reset).parameters
    assert "world_mask" in params
    assert "flags" in params
    assert params["world_mask"].default is None
    assert params["flags"].default == 0
