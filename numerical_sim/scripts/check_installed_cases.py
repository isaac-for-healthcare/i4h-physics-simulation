# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check one independently installed domain's shipped cases with external production data."""
import importlib
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

package = importlib.import_module(sys.argv[1])
other = "i4h_surgical" if package.__name__ == "i4h_endoluminal" else "i4h_endoluminal"
assert find_spec(other) is None and find_spec("i4h_standalone") is None and find_spec("i4h_isaaclab") is None
cases = sorted((Path(package.__file__).parent / "cases").glob("*.yaml"))
for case in cases:
    sim = package.load_case(str(case), data=sys.argv[2], envs=2)
    sim.step()
    sim.reset([1])
    for body in sim.scene.bodies + sim.scene.pools:
        assert np.isfinite(sim.scene.field("q", body).numpy()).all(), case.name
    print(f"{package.__name__}: {case.stem} reset/step passed", flush=True)
print(f"{len(cases)} shipped {package.__name__} cases passed")
