# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9d: every system's panel() lists live arrays and recapture-safe fields only; the loader keeps the YAML ids."""

import numpy as np
import pytest
import warp as wp
from helpers import needs_external_data
from i4h_endoluminal.systems import SYSTEMS as ENDOLUMINAL_SYSTEMS
from i4h_physics.systems import SYSTEMS as PHYSICAL_SYSTEMS
from i4h_standalone.app.panel_adapters import panel
from i4h_standalone.config import load_case
from i4h_surgical.systems import SYSTEMS as SURGICAL_SYSTEMS

SYSTEMS = {**PHYSICAL_SYSTEMS, **SURGICAL_SYSTEMS, **ENDOLUMINAL_SYSTEMS}

CASES = ["two_worlds", *[pytest.param(case, marks=needs_external_data)
                         for case in ("chole_tet_med", "aorta_shell", "aorta_centreline")]]


@pytest.mark.parametrize("case", CASES)
def test_panels_are_live_or_recapture_safe(case):
    sim = load_case(case, envs=2)
    scene = sim.scene
    assert all(s.id for s in sim.systems)
    assert len({s.id for s in sim.systems}) == len(sim.systems)
    knobs = 0
    for s in sim.systems:
        for k in panel(s):
            knobs += 1
            assert (k.array is None) != (k.field is None), (s.id, k.label)
            assert k.lo < k.hi and (not k.log or k.lo > 0.0)
            if k.array is not None:
                assert isinstance(k.array, wp.array) and k.array.dtype is wp.float32 and k.array.device == scene.device
                assert k.array.shape == ((scene.envs,) if not k.labels else (scene.envs, len(k.labels))), (s.id, k.label)
                assert getattr(s, k.label) is k.array  # the label names the attribute (overrides bind by it)
                k.array.fill_(float(k.hi))
            else:
                assert k.field in s.structural and k.field == k.label
                value = getattr(s, k.field)
                setattr(s, k.field, not value if isinstance(value, bool) else int(k.hi))
    assert knobs > 0
    sim.prepare()  # nothing exposed raises on change
    sim.step()
    assert np.isfinite(scene.q.numpy()).all()


@needs_external_data
def test_ids_follow_the_yaml():
    sim = load_case("chole_tet_med")
    ids = [s.id for s in sim.systems]
    assert {"blood_cavity", "irrigation_cavity", "static_contact"} <= set(ids)
    kinds = {v: k for k, v in SYSTEMS.items()}
    assert sum(kinds[type(s)] == "static_contact" for s in sim.systems) == 3
