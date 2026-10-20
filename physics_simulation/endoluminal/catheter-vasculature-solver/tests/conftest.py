# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for catheter-vasculature-solver tests."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def cpu_rod_config():
    """RodConfig matching the README public API example, forced onto CPU for CI."""
    from catheter_vasculature_solver import RodConfig

    cfg = RodConfig()
    cfg.device = "cpu"
    cfg.geometry.num_segments = 24
    cfg.solver.newton_iterations = 4
    return cfg


@pytest.fixture
def track_params():
    """Minimal insertion-track parameters for CathRodSolver."""
    track_start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    track_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    track_length = 0.5
    return track_start, track_dir, track_length


@pytest.fixture
def rod_managers(monkeypatch):
    """Load the real managers; substitute Lab's imports only in CPU-only CI.

    With Isaac Lab installed this exercises its actual configs and base class.
    The fallback keeps manager regressions running in the package-only CI jobs,
    without replacing any of this package's production code.
    """
    import catheter_vasculature_solver  # noqa: F401 -- initialize Torch before Warp/Isaac Lab

    stubbed = importlib.util.find_spec("isaaclab_newton") is None
    package = "catheter_vasculature_solver.isaaclab_integration"
    names = ("solver_cfg", "xpbd_rod_manager", "coupled_manager")
    if stubbed:
        def configclass(cls):
            def init(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            cls.__init__ = init
            return cls

        for name in ("isaaclab", "isaaclab.utils", "isaaclab.utils.configclass", "isaaclab_newton", "isaaclab_newton.physics"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        sys.modules["isaaclab.utils.configclass"].configclass = configclass
        physics = sys.modules["isaaclab_newton.physics"]
        physics.NewtonManager = type("NewtonManager", (), {})
        physics.NewtonSolverCfg = type("NewtonSolverCfg", (), {})
        physics.MJWarpSolverCfg = type("MJWarpSolverCfg", (), {})

    modules = [importlib.import_module(f"{package}.{name}") for name in names]
    cfg, rod_module, coupled_module = modules
    rod = rod_module.NewtonXPBDRodManager
    coupled = coupled_module.NewtonCoupledMJWarpXPBDRodManager
    rod._solver_specific_clear()
    coupled._solver_specific_clear()
    try:
        yield SimpleNamespace(
            rod=rod,
            coupled=coupled,
            base=rod_module.NewtonManager,
            cfg=cfg.XPBDRodSolverCfg,
            coupled_cfg=cfg.CoupledMJWarpXPBDRodSolverCfg,
        )
    finally:
        coupled._solver_specific_clear()
        if stubbed:
            for name, module in zip(names, modules):
                sys.modules.pop(f"{package}.{name}", None)
                if getattr(sys.modules[package], name, None) is module:
                    delattr(sys.modules[package], name)
