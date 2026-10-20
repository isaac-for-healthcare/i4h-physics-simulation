# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiled observation runs outside the checkout with standalone and surgical absent."""
import sys
from importlib.util import find_spec

import numpy as np
from i4h_endoluminal import load_case
from i4h_endoluminal.config import read_case
from i4h_isaaclab.camera.tiled import TiledCamera

assert find_spec("i4h_standalone") is None and find_spec("i4h_surgical") is None
sim = load_case("aorta_static", data=sys.argv[1], envs=2)
cam = TiledCamera(sim, 32, 32, camera=read_case("aorta_static", data=sys.argv[1])["render"]["camera"])
cam.prepare()
sim.step()
cam.render()
assert cam.rgb.shape == (2, 32, 32, 4)
assert np.isfinite(cam.depth.numpy()).all()
assert (cam.depth.numpy() > 0).any()
sim.reset([1])
cam.render()
print("installed tiled observations without standalone/surgical: passed")
