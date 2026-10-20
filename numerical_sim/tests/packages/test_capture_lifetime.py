# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unrelated cyclic CUDA owners must not be collected while a graph is recorded."""
import gc

import pytest
import warp as wp
from i4h_physics import System, load_case
from i4h_physics.systems.mechanics.particles import Particles


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph lifetime check")
@pytest.mark.parametrize("collecting", [False, True])
@pytest.mark.parametrize("reject", [False, True])
def test_capture_defers_collection_and_restores_state(reject, collecting):
    collections_during_capture = []
    device = wp.get_device("cuda:0")

    def observe(phase, info):
        if phase == "start" and device.is_capturing:
            collections_during_capture.append(info["generation"])

    class Probe(System):
        def step_start(self, scene, dt):
            # Force collection pressure from Python bookkeeping during dispatch.
            # In the full render suite these cycles include owners of prior scenes.
            for _ in range(100):
                cycle = []
                cycle.append(cycle)
            if reject:
                raise RuntimeError("extension rejected capture")

    sim = load_case({
        "sim": {"device": "cuda:0", "envs": 2, "gravity": [0, 0, 0]},
        "assets": {"rod": {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0], "radius": .001,
                            "count": 3, "segment_length": .01}},
        "systems": [{"type": "probe"}, {"type": "particles"}],
    }, systems={"probe": Probe, "particles": Particles})
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    gc.callbacks.append(observe)
    if collecting:
        gc.enable()
    else:
        gc.disable()
    gc.set_threshold(1, 1, 1)
    try:
        if reject:
            with pytest.raises(RuntimeError, match="extension rejected capture"):
                sim.prepare()
            assert sim._prepared is None
        else:
            sim.prepare()
            assert sim.graph is not None and sim.reset_graph is not None
            sim.step()
            sim.reset([0])
        assert not collections_during_capture
        assert gc.isenabled() == collecting
    finally:
        gc.callbacks.remove(observe)
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()
