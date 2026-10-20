# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F3 (C6): the host reads what the systems declare (``is_control``, ``action_bounds``, ``recorded_actions``,
``state_products``), so a system and a loader defined outside the packages take part; explicit maps replace the
installed-domain discovery. No Isaac Lab needed."""

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp
from i4h_isaaclab import views
from i4h_isaaclab.composition import load_case
from i4h_physics.loaders.grid import add_tet_grid
from i4h_physics.systems.base import System


def add_cube(scene, name):
    return add_tet_grid(scene, name, dim=[1, 1, 1], cell=0.02, parts={name: {"spring": 1.0, "volume": 1.0}})


@dataclass(eq=False)
class Dial(System):
    """A control with five action columns and one state product."""

    is_control = True
    state_products = ("turns",)

    def __post_init__(self):
        E, dev = self.scene.envs, self.scene.device
        self.actions = wp.zeros((E, 5), dtype=wp.float32, device=dev)
        self.turns = wp.zeros(E, dtype=wp.float32, device=dev)

    def action_bounds(self, workspace):
        return [0.0, -1.0, -2.0, -3.0, -4.0], [1.0, 1.0, 2.0, 3.0, 4.0 + workspace]


class Silent(Dial):
    def action_bounds(self, workspace):
        return [0.0], [1.0]  # declares one column for five


def case(*extra):
    return {"sim": {"device": "cuda:0", "envs": 2, "cuda_graph": False}, "assets": {"cube": {"type": "cube"}},
            "systems": [{"type": "particles"}, {"type": "tet", "body": "cube"}, *extra]}


def test_a_custom_control_and_loader_take_part():
    sim = load_case(case({"type": "dial", "id": "knob"}, {"type": "bounds"}),
                    loaders={"cube": add_cube}, systems={"dial": Dial})
    assert [type(s).__name__ for s in views.controls(sim)] == ["Dial"]
    low, high = views.action_bounds(sim, workspace=0.5)
    np.testing.assert_array_equal(low, [0.0, -1.0, -2.0, -3.0, -4.0])
    np.testing.assert_array_equal(high, [1.0, 1.0, 2.0, 3.0, 4.5])
    assert views.state_names(sim) == ["knob.turns"] and tuple(views.state_views(sim)[0].shape) == (2, 1)
    torch = pytest.importorskip("torch")
    views.write_actions(views.action_views(sim), torch.arange(10, dtype=torch.float32, device="cuda:0").view(2, 5))
    np.testing.assert_array_equal(sim.systems[2].actions.numpy(), np.arange(10, dtype=np.float32).reshape(2, 5))
    normalised, excess = views.replay_actions(sim, "/nonexistent/take", low, high)
    assert normalised.shape == (1, 5) and excess == 0.0  # no recording: the control holds its bounds' midpoint


def test_bounds_must_cover_the_columns():
    sim = load_case(case({"type": "dial"}), loaders={"cube": add_cube}, systems={"dial": Silent})
    with pytest.raises(ValueError, match="1/1 bounds for 5 action columns"):
        views.action_bounds(sim, workspace=0.1)


def test_explicit_maps_replace_discovery():
    with pytest.raises(ValueError, match="unknown type 'instrument_control'"):
        load_case(case({"type": "instrument_control"}), loaders={"cube": add_cube}, systems={})
    with pytest.raises(ValueError, match="unknown type 'cube'"):
        load_case(case(), systems={"dial": Dial})  # explicit systems only: no domain loaders either, and no cube
