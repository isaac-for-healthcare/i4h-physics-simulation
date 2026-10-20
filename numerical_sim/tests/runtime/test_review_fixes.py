# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for the bugs found in the full-tree review (2026-09-25)."""

import copy
import os

import numpy as np
import pytest
import warp as wp
from helpers import find, push_actions
from i4h_isaaclab.composition import data_dir, load_case, read_case
from i4h_physics import Scene
from i4h_surgical.systems.fluids.emitter import Emitter
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


@pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (data_dir() / "vessels" / "AortaWithVesselsStatic.usdc").is_file(), reason="aorta assets missing")
def test_tiled_camera_on_a_tree_case():
    """A centreline tree is a rod_pbd body without a RodMesh: the tiled camera skips it instead of crashing."""
    pytest.importorskip("torch")
    from i4h_isaaclab.camera.tiled import TiledCamera

    cam = TiledCamera(load_case("aorta_centreline"), 16, 16)
    cam.prepare()
    cam.render()
    wp.synchronize()
    assert np.isfinite(cam.depth.numpy()).all()


def test_emitter_leaves_no_backlog():
    """A rate far above max_spawn used to pile an unbounded carry into ``acc`` that kept emitting after the rate dropped."""
    sim = load_case("two_worlds")
    em = find(sim, Emitter)
    em.rate.fill_(8192.0)
    for _ in range(30):
        sim.step()
    wp.synchronize()
    assert (em.acc.numpy() < 1.0).all()
    assert (em.spawned.numpy() == em.max_spawn).all(), "the cap bounds each step's births"
    em.rate.fill_(0.0)
    for _ in range(2):
        sim.step()
    wp.synchronize()
    assert (em.spawned.numpy() == 0).all(), "nothing is emitted once the rate is zero (spawned: this step's births)"


def test_sim_debug_key_in_the_case():
    case = copy.deepcopy(read_case("two_worlds"))
    case["sim"]["debug"] = True
    sim = load_case(case)
    assert sim.debug and "debug" in sim.structural
    sim.step()


def test_zero_threshold_does_not_latch_energy():
    """A zero energy or fire command is off even when the threshold is 0 (it used to compare 0 >= 0)."""
    sim = load_case("two_worlds")
    control = find(sim, InstrumentControl)
    control.energy_threshold.fill_(0.0)
    control.fire_threshold.fill_(0.0)
    control.actions.assign(push_actions(control, sim.scene.envs))  # energy and fire 0
    sim.step()
    wp.synchronize()
    assert not control.energized.numpy().any() and not control.fire_on.numpy().any()


def test_scene_needs_an_env():
    with pytest.raises(ValueError, match="at least 1"):
        Scene("cuda:0", 0)


def test_clear_rejects_a_buffer_of_the_wrong_length():
    sim = load_case("two_worlds")
    control = find(sim, InstrumentControl)
    mask = wp.ones(sim.scene.envs, dtype=wp.bool, device=sim.scene.device)
    with pytest.raises(ValueError, match="not"):
        control.clear(mask, (wp.zeros(3, dtype=wp.int32, device=sim.scene.device), wp.int32(0)))
