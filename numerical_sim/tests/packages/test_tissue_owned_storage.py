# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tissue, fluid and containment history (heat, colour, fluid counters, smoke, candidates): lifecycle contracts
that ownership must preserve."""
import numpy as np
import pytest
import warp as wp
from interaction_helpers import MODES, authored, runtime, system
from tissue_helpers import CONTAINMENT, RESET, TISSUE, aorta, history, poison, world_rows


def cases():
    return [pytest.param(authored, TISSUE, id="chole"), pytest.param(aorta, CONTAINMENT, id="aorta_centreline")]


def test_history_set_is_the_inventoried_one():
    assert sum(len(v) for v in TISSUE.values()) + sum(len(v) for v in CONTAINMENT.values()) == 25


@pytest.mark.parametrize("build,names", cases())
@pytest.mark.parametrize("device,captured", MODES)
def test_episode_reset_clears_history_in_selected_worlds_only(device, captured, build, names):
    cfg, a, m = build(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim, names)
    before = history(sim, names)
    sim.reset_physics(mask=wp.array([False, True], dtype=wp.bool, device=device))
    after = history(sim, names)
    for key in before:
        old = np.frombuffer(before[key], np.uint8).reshape(2, -1)
        new = np.frombuffer(after[key], np.uint8).reshape(2, -1)
        np.testing.assert_array_equal(new[0], old[0], err_msg=f"unselected world changed: {key}")
        np.testing.assert_array_equal(world_rows(sim, *key, 1), RESET.get(key, 0), err_msg=str(key))


@pytest.mark.parametrize("device,captured", MODES)
def test_cache_only_preparation_preserves_tissue_history(device, captured):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim, TISSUE)
    before = history(sim, TISSUE)
    sim.prepare()
    sim.prepare()
    assert history(sim, TISSUE) == before


@pytest.mark.parametrize("device,captured", MODES)
def test_containment_preparation_forces_a_rebuild_and_keeps_the_cache_bytes(device, captured):
    """A named exception: preparation invalidates the candidate cache in every world, because the spline settings
    it protects are not part of the cache's validity record."""
    cfg, a, m = aorta(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim, CONTAINMENT)
    before = history(sim, CONTAINMENT)
    sim.prepare()
    after = history(sim, CONTAINMENT)
    key = ("GraphContainment", "rebuild")
    np.testing.assert_array_equal(system(sim, "GraphContainment").rebuild.numpy(), 1)
    assert {k: v for k, v in after.items() if k != key} == {k: v for k, v in before.items() if k != key}


@pytest.mark.parametrize("device,captured", MODES)
def test_smoke_serial_counts_steps_and_its_source_point_is_an_output(device, captured):
    """``serial`` seeds the spawn random numbers and advances once per step in every world. ``centre`` is written
    only while a source exists and no kernel reads it, so it is a product, not history."""
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    smoke = system(sim, "Smoke")
    for _ in range(3):
        sim.step()
    np.testing.assert_array_equal(smoke.serial.numpy(), 3)
    sim.reset_physics(mask=wp.array([True, False], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(smoke.serial.numpy(), [0, 3])
