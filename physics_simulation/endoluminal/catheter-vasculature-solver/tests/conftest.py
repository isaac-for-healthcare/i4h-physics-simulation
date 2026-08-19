# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for catheter-vasculature-solver tests."""

from __future__ import annotations

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
