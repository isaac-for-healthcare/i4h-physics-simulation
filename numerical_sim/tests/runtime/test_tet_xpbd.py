# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tet.model: xpbd (opt-in): compliance-based edge and volume constraints, so a converged tissue's stiffness does not
depend on the time step, substeps or iterations (the default pbd stiffness factors do). The F2 coupled-probe
qualification uses it for its refinement row."""

import copy

import numpy as np
import pytest
from i4h_isaaclab.composition import load_case, read_case


def settled_bottom(model, dt, substeps, iterations, spring=500.0, volume=1.0e9, seconds=3.0):
    """two_worlds' block hanging from its pinned top under gravity: its lowest point after ``seconds``."""
    cfg = copy.deepcopy(read_case("two_worlds"))
    cfg["sim"].update(dt=dt, substeps=substeps, iterations=iterations, cuda_graph=False)
    cfg["assets"] = {"block": dict(cfg["assets"]["block"], parts={"block": {"spring": spring, "volume": volume}})}
    cfg["systems"] = [{"type": "particles", "damping": 0.99}, {"type": "tet", "body": "block", "model": model}]
    sim = load_case(cfg)
    for _ in range(int(round(seconds / dt))):
        sim.step()
    return float(sim.scene.field("q", sim.scene.body("block")).numpy()[0][:, 2].min())


SCHEDULES = ((1 / 60, 4, 8), (1 / 120, 4, 8), (1 / 60, 8, 8), (1 / 60, 4, 16))


def test_xpbd_stiffness_does_not_depend_on_the_schedule():
    sag = [0.05 - settled_bottom("xpbd", *s) for s in SCHEDULES]  # the bottom face starts at z 0.05
    assert min(sag) > 0.05, "the block sags visibly"
    assert (max(sag) - min(sag)) / np.mean(sag) < 0.05, sag
    pbd = [0.05 - settled_bottom("pbd", *s, spring=1.0, volume=1.0) for s in SCHEDULES]
    assert (max(pbd) - min(pbd)) / np.mean(pbd) > 0.2, "the pbd factors do depend on it (the contrast)"


def test_model_is_validated_and_structural():
    with pytest.raises(ValueError, match="'pbd' or 'xpbd'"):
        settled_bottom("fem", 1 / 60, 1, 1, seconds=0.0)
    cfg = copy.deepcopy(read_case("two_worlds"))
    cfg["sim"]["cuda_graph"] = False
    sim = load_case(cfg)
    tet = next(s for s in sim.systems if type(s).__name__ == "Tet")
    assert tet.model == "pbd" and tet.edge_lambda.shape == (1, 1)
    tet.model = "xpbd"
    with pytest.raises(ValueError, match="fixed at construction"):
        sim.prepare()
