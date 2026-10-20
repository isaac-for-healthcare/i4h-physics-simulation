# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run generated rod, tet, fluid and contact mechanics with only i4h-physics installed."""
import argparse
from importlib.util import find_spec

import numpy as np
import warp as wp
from i4h_physics import load_case
from i4h_physics.scene import ACTIVE_INT

parser = argparse.ArgumentParser()
parser.add_argument("--qualify-isolation", action="store_true")
parser.add_argument("--device", choices=("cpu", "cuda:0"))
args = parser.parse_args()
if args.qualify_isolation:
    assert all(find_spec(name) is None for name in ("i4h_surgical", "i4h_endoluminal", "i4h_standalone", "i4h_isaaclab"))
for device in ((args.device,) if args.device else (("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",))):
    for captured in ((False, True) if device.startswith("cuda") else (False,)):
        cfg = {"sim": {"device": device, "envs": 2, "dt": .01, "gravity": [0, 0, 0],
                       "substeps": 2, "iterations": 4, "cuda_graph": captured},
               "assets": {"solid": {"type": "tet_grid", "dim": [3, 3, 3], "cell": .02, "fix_top": True},
                          "rod": {"type": "rod", "start": [.2, 0, .1], "direction": [1, 0, 0],
                                  "count": 6, "segment_length": .02, "radius": .002},
                          "fluid": {"type": "fluid", "count": 8, "radius": .003, "mass": .001,
                                    "interaction_radius": .012, "rest_distance": .006}},
               "systems": [{"type": "particles"}, {"type": "tet", "body": "solid", "id": "solid"},
                           {"type": "surface", "tet": "solid", "id": "surface"},
                           {"type": "rod_direct", "body": "rod"}, {"type": "fluid", "id": "fluid"},
                           {"type": "fluid_contact", "fluid": "fluid", "surface": "surface", "margin": .004}]}
        sim = load_case(cfg)
        sim.prepare()
        body = sim.scene.body("fluid")
        q = sim.scene.field("q", body).numpy()
        q[:] = [.02, .02, .059]
        q[:, :, 0] += np.arange(8, dtype=np.float32)[None, :] * .001
        sim.scene.field("q", body).assign(q)
        sim.scene.field("flags", body).fill_(ACTIVE_INT)
        for _ in range(12):
            sim.step()
        for body in sim.scene.bodies + sim.scene.pools:
            assert np.isfinite(sim.scene.field("q", body).numpy()).all()
        assert sim.scene.field("q", sim.scene.body("fluid")).numpy()[0, 0, 2] > .06
        sim.reset([0])
        print(f"core-only generated rod/tet/contact/fluid {device} captured={captured}: passed")
