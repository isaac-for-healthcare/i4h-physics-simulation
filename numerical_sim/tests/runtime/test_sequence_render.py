# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from helpers import slang_tests
from test_procedure_sequence import sequence


@slang_tests
def test_tiled_sequence_has_no_target_markers():
    from i4h_host.composition import build_procedure
    from i4h_isaaclab.camera.tiled import TiledCamera

    sim, _, cfg = sequence(device="cuda:0", graph=True)
    cfg["procedure"]["steps"][1].update(target={"position": [1, 1, 1]}, radius=0.4)
    sim.procedure = task = build_procedure(cfg, sim)
    camera = TiledCamera(sim.simulation, 64, 64)
    camera.prepare()
    camera.render()
    before = camera.rgb_t.clone()
    task.update(0.1)
    camera.render()
    assert (before == camera.rgb_t).all()
