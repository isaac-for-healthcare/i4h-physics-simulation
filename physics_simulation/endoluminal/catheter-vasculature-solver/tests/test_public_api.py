# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic tests covering the public API examples from the package README."""

from __future__ import annotations

import numpy as np
import pytest


def test_package_version_and_exports():
    import catheter_vasculature_solver as cvs

    assert isinstance(cvs.__version__, str)
    assert cvs.__version__ == "0.1.0"
    for name in (
        "RodConfig",
        "XPBDRodSolver",
        "CathRodSolver",
        "NewtonXPBDRodSolver",
        "compute_smooth_vertex_normals",
    ):
        assert hasattr(cvs, name)


def test_readme_xpbd_rod_solver_example(cpu_rod_config):
    """Exercise the primary README usage example for XPBDRodSolver."""
    from catheter_vasculature_solver import RodConfig, XPBDRodSolver

    cfg = cpu_rod_config
    assert isinstance(cfg, RodConfig)

    solver = XPBDRodSolver(cfg)
    # Keep the step loop short for CI while preserving the README call pattern.
    for _ in range(5):
        solver.step(cfg.solver.dt)

    positions = solver.positions
    assert positions.shape == (cfg.geometry.num_segments + 1, 3)
    assert np.isfinite(positions.detach().cpu().numpy()).all()


def test_readme_cath_rod_solver_import_and_step(cpu_rod_config, track_params):
    """Exercise the vessel-aware README API without requiring a vessel mesh."""
    from catheter_vasculature_solver import CathRodSolver, RodConfig

    cfg = cpu_rod_config
    assert isinstance(cfg, RodConfig)
    track_start, track_dir, track_length = track_params

    solver = CathRodSolver(
        cfg,
        collision_mesh=None,
        track_start=track_start,
        track_dir=track_dir,
        track_length=track_length,
        tip_num_edges=10,
        particle_radius=0.002,
        segment_length=cfg.geometry.segment_length,
        collision_enabled=False,
    )
    for _ in range(3):
        solver.step(cfg.solver.dt)

    positions = solver.positions
    assert positions.shape == (cfg.geometry.num_segments + 1, 3)
    assert np.isfinite(positions.detach().cpu().numpy()).all()


def test_readme_newton_bridge_is_importable():
    """README documents NewtonXPBDRodSolver as an optional-dependency import."""
    from catheter_vasculature_solver import NewtonXPBDRodSolver, RodConfig

    assert NewtonXPBDRodSolver is not None
    assert RodConfig is not None


def test_newton_solver_requires_optional_dependency(cpu_rod_config):
    """Constructing NewtonXPBDRodSolver without newton installed should fail clearly."""
    pytest.importorskip("catheter_vasculature_solver")
    from catheter_vasculature_solver import NewtonXPBDRodSolver

    try:
        import newton  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="Newton"):
            NewtonXPBDRodSolver(cpu_rod_config)
    else:
        pytest.skip("newton is installed; skip missing-dependency assertion")


def test_compute_smooth_vertex_normals_public_helper():
    from catheter_vasculature_solver import compute_smooth_vertex_normals

    # Unit triangle in the XY plane.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.int32)

    normals = compute_smooth_vertex_normals(vertices, indices)
    assert normals.shape == (3, 3)
    # Face normal should point along +Z.
    assert np.allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-5)
