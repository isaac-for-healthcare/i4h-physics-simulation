# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from helpers import needs_external_data

pytest.importorskip("torch")

from i4h_isaaclab.composition import (  # noqa: E402 - optional dependency guard must run first
    build_procedure,
    load_case,
    read_case,
)
from i4h_isaaclab.views import observation_views, state_views  # noqa: E402 - optional dependency guard must run first


@pytest.mark.parametrize("case", ["aorta_shell_reach", "aorta_centreline_reach"])
@needs_external_data
def test_task_products_and_observation_selection(case):
    cfg = read_case(case)
    assert cfg["procedure"]["hold"] == .5 and cfg["procedure"]["timeout"] == 120
    sim = load_case(cfg, envs=2)
    assert not hasattr(sim, "procedure")
    procedure = build_procedure(cfg, sim)
    q = observation_views(sim, "catheter")[0]
    assert q.shape == (2, sim.scene.body("catheter").count*3)
    assert q.data_ptr() == sim.scene.field("q", sim.scene.body("catheter")).ptr
    sim.prepare()
    sim.step()
    procedure.update(sim.dt)
    assert all(np.isfinite(v.cpu().numpy()).all() for v in state_views(sim))
    for cont in (s for s in sim.systems if hasattr(s, "contact_depth")):
        pointers = [v.data_ptr() for v in state_views(sim)]
        assert cont.contact_depth.ptr in pointers and cont.contact_count.ptr in pointers
