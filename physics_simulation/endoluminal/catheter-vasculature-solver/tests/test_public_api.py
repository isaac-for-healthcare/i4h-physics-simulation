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


def test_newton_extra_is_satisfiable_from_an_index():
    """The ``[newton]`` extra must resolve from a package index, not a git ref.

    It previously resolved PyPI ``newton`` while the code needed a symbol only a
    closed pull request carried, so the install looked supported and then failed
    at construction. Now that the rod solver is vendored, the extra needs only a
    released newton -- so a direct reference here would be a regression.
    """
    import pathlib
    import re

    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    section = re.search(
        r"^\[project\.optional-dependencies\]$(.*?)(?=^\[)",
        pyproject.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert section, "could not locate [project.optional-dependencies] in pyproject.toml"

    body = section.group(1)
    assert re.search(r"^\s*newton\s*=", body, re.MULTILINE), "the `newton` extra is missing"

    # Strip comments first: the prose around this block mentions URLs and PR refs.
    requirements = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    for marker in ("git+", "@ http", "refs/pull"):
        assert marker not in requirements, (
            f"the `newton` extra uses a direct reference ({marker!r}); it must stay "
            "installable from an index, since the rod solver is vendored in-tree"
        )


@pytest.mark.gpu
def test_newton_bridge_constructs_and_steps():
    """PHYS-3.5a: the reported construct+step must succeed on a released Newton.

    Reproduces the bug report's call verbatim. It used to raise ``ImportError:
    cannot import name 'SolverXPBDRod' from 'newton.solvers'`` because that
    symbol ships in no Newton release; the solver is now vendored in-tree, so a
    plain ``[newton]`` install is enough. Marked gpu because ``RodConfig``
    defaults to CUDA and the Warp kernels compile on first run.
    """
    pytest.importorskip("newton", reason="the [newton] extra is not installed")
    from catheter_vasculature_solver import NewtonXPBDRodSolver, RodConfig

    cfg = RodConfig()
    cfg.geometry.num_segments = 24
    solver = NewtonXPBDRodSolver(cfg)
    solver.step(cfg.solver.dt)

    positions = solver.positions
    assert positions.shape[-1] == 3
    assert np.isfinite(positions.detach().cpu().numpy()).all()


def test_newton_bridge_does_not_import_the_unreleased_symbol():
    """The bridge must resolve the rod solver in-tree, not from ``newton.solvers``.

    Importing it from Newton is what made the documented install fail, so keep
    the dependency pointed at the vendored copy.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "catheter_vasculature_solver"
    offenders = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if not path.is_relative_to(src / "newton_xpbd_rod")
        and "from newton.solvers import SolverXPBDRod" in path.read_text()
    ]
    assert not offenders, f"these modules still import SolverXPBDRod from newton: {offenders}"


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


@pytest.mark.parametrize("stale_name", ["xray_simulator", "omniendo"])
def test_docstrings_do_not_name_pre_port_sources(stale_name):
    """Docstrings must name this package, not the sources it was ported from.

    The solvers came out of ``xray_simulator.catheter`` and the vessel backends
    out of OmniEndo, and the docstrings kept advertising both. Nothing imports a
    docstring, so stale ``from xray_simulator.catheter import ...`` lines stayed
    green in CI while being the first thing a reader copies and runs.
    """
    import pkgutil

    import catheter_vasculature_solver as cvs

    stale = []
    for info in pkgutil.walk_packages(cvs.__path__, prefix=f"{cvs.__name__}."):
        try:
            module = __import__(info.name, fromlist=["_"])
        except ImportError:
            continue  # optional extras (e.g. the newton bridge) may not install
        for name, obj in [("<module>", module), *vars(module).items()]:
            doc = getattr(obj, "__doc__", None)
            if isinstance(doc, str) and stale_name in doc.lower():
                stale.append(f"{info.name}:{name}")
    assert not stale, f"docstrings still name {stale_name!r}: " + ", ".join(sorted(set(stale)))


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
