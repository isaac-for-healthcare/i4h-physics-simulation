# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adding catheter rod particles to a Newton ``ModelBuilder``.

Uses a fake builder so the Newton-side authoring contract is covered without a
``newton`` install; only ``particle_count`` and ``add_particle`` are used.
"""

from __future__ import annotations

import numpy as np
import pytest


class FakeBuilder:
    """Records ``add_particle`` calls like Newton's ``ModelBuilder`` would."""

    def __init__(self, existing: int = 0):
        self.positions: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * existing
        self.masses: list[float] = [1.0] * existing
        self.radii: list[float] = [0.0] * existing

    @property
    def particle_count(self) -> int:
        return len(self.positions)

    def add_particle(self, pos, vel, mass, radius=None) -> int:
        del vel
        self.positions.append(tuple(pos))
        self.masses.append(float(mass))
        self.radii.append(radius)
        return len(self.positions) - 1


class FakeSolverCfg:
    """Duck-typed stand-in for ``XPBDRodSolverCfg`` (no Isaac Lab needed)."""

    young_modulus = 2.0e8
    torsion_modulus = None
    density = 1200.0
    linear_damping = 0.05
    bend_stiffness = 0.2
    twist_stiffness = 0.3
    num_segments = 8
    segment_length = 0.01
    radius = 0.0015
    num_substeps = 3
    gravity = (0.0, 0.0, -9.81)


def test_initial_positions_match_solver_rest_state(cpu_rod_config):
    """A mismatch here would make the rod snap on its first Newton step."""
    from catheter_vasculature_solver import XPBDRodSolver
    from catheter_vasculature_solver.isaaclab_integration import initial_rod_positions

    initial_height = 0.37
    authored = initial_rod_positions(cpu_rod_config, initial_height=initial_height)
    solver = XPBDRodSolver(cpu_rod_config, floor_z=None, initial_height=initial_height)

    assert authored.shape == (solver.num_points, 3)
    np.testing.assert_allclose(authored, solver.positions.detach().cpu().numpy(), rtol=0, atol=1e-6)


def test_initial_positions_follow_start_and_direction(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import initial_rod_positions

    start = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positions = initial_rod_positions(cpu_rod_config, start=start, direction=np.array([0.0, 2.0, 0.0]))

    np.testing.assert_allclose(positions[0], start, rtol=0, atol=1e-6)
    step = float(cpu_rod_config.geometry.segment_length)
    np.testing.assert_allclose(positions[1] - positions[0], [0.0, step, 0.0], rtol=0, atol=1e-6)


def test_initial_positions_rejects_zero_direction(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import initial_rod_positions

    with pytest.raises(ValueError, match="non-zero"):
        initial_rod_positions(cpu_rod_config, direction=np.zeros(3))


def test_add_rod_returns_range_after_existing_particles(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder

    builder = FakeBuilder(existing=11)
    particle_range = add_catheter_rod_to_builder(builder, cpu_rod_config)

    num_points = cpu_rod_config.geometry.num_segments + 1
    assert particle_range.offset == 11
    assert particle_range.count == num_points
    assert particle_range.num_envs == 1
    assert builder.particle_count == 11 + num_points


def test_add_rod_replicates_per_env(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder

    builder = FakeBuilder()
    particle_range = add_catheter_rod_to_builder(builder, cpu_rod_config, num_envs=3)

    num_points = cpu_rod_config.geometry.num_segments + 1
    assert particle_range.count == num_points * 3
    assert particle_range.num_envs == 3
    # Each replica starts from the same rest layout.
    first = builder.positions[:num_points]
    second = builder.positions[num_points : 2 * num_points]
    assert first == second


def test_locked_root_is_kinematic(cpu_rod_config):
    """Zero mass marks the particle kinematic in Newton, matching the rod's locked root."""
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder

    builder = FakeBuilder()
    add_catheter_rod_to_builder(builder, cpu_rod_config, lock_root=True)

    assert builder.masses[0] == 0.0
    assert all(m > 0.0 for m in builder.masses[1:])


def test_unlocked_root_has_mass(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder

    builder = FakeBuilder()
    add_catheter_rod_to_builder(builder, cpu_rod_config, lock_root=False)

    assert all(m > 0.0 for m in builder.masses)


def test_add_rod_rejects_wrong_position_shape(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder

    builder = FakeBuilder()
    bad = np.zeros((cpu_rod_config.geometry.num_segments, 3), dtype=np.float32)  # missing last point
    with pytest.raises(ValueError, match="num_segments \\+ 1"):
        add_catheter_rod_to_builder(builder, cpu_rod_config, positions=bad)


def test_add_rod_rejects_bad_num_envs(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder

    with pytest.raises(ValueError, match="num_envs"):
        add_catheter_rod_to_builder(FakeBuilder(), cpu_rod_config, num_envs=0)


def test_rod_config_from_solver_cfg_maps_fields():
    from catheter_vasculature_solver.isaaclab_integration import rod_config_from_solver_cfg

    cfg = rod_config_from_solver_cfg(FakeSolverCfg(), device="cpu")

    assert cfg.device == "cpu"
    assert cfg.geometry.num_segments == 8
    assert cfg.geometry.segment_length == pytest.approx(0.01)
    assert cfg.geometry.radius == pytest.approx(0.0015)
    assert cfg.material.young_modulus == pytest.approx(2.0e8)
    assert cfg.material.density == pytest.approx(1200.0)
    assert cfg.material.bend_stiffness == pytest.approx(0.2)
    assert cfg.solver.num_substeps == 3
    assert cfg.solver.gravity == (0.0, 0.0, -9.81)
    # shear_modulus is derived when torsion_modulus is not supplied
    assert cfg.material.shear_modulus == pytest.approx(2.0e8 / (2.0 * 1.3))


def test_rod_config_honours_explicit_torsion_modulus():
    from catheter_vasculature_solver.isaaclab_integration import rod_config_from_solver_cfg

    class Cfg(FakeSolverCfg):
        torsion_modulus = 5.0e7

    cfg = rod_config_from_solver_cfg(Cfg(), device="cpu")
    assert cfg.material.shear_modulus == pytest.approx(5.0e7)


def test_particle_mass_matches_solver_formula(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import particle_mass_from_rod_config

    radius = float(cpu_rod_config.geometry.radius)
    length = float(cpu_rod_config.geometry.segment_length)
    density = float(cpu_rod_config.material.density)
    expected = np.pi * radius * radius * length * density

    assert particle_mass_from_rod_config(cpu_rod_config) == pytest.approx(expected)
