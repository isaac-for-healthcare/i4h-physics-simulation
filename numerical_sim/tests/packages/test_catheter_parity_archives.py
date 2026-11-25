# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Incomplete historical results must not silently qualify the migration."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("catheter_parity", Path(__file__).parents[2] / "scripts/parity/catheter.py")
parity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parity
spec.loader.exec_module(parity)


def archives():
    data = {"rod": json.dumps(parity.asdict(parity.Rod())), "schema": 2, "complete": True,
            "source": json.dumps({"revision": parity.REFERENCE_REVISION, "status": ""})}
    rod = parity.Rod()
    for name, (_, steps, command) in parity.scenarios(rod).items():
        sample_steps = range(parity.STRIDE - 1, steps, parity.STRIDE)
        T, P = len(sample_steps), rod.segments + 1
        wall = parity.tube(name)
        W = 0 if wall is None else len(wall)
        shapes = {"positions": (T, P, 3), "orientations": (T, P, 4), "velocities": (T, P, 3),
                  "angular_velocities": (T, P, 3), "reaction": (T, 6), "commands": (T, 3),
                  "time": (T,), "vessel_displacement": (T, W, 3)}
        for half in ("", "@half"):
            for field, shape in shapes.items():
                data[name + half + "/" + field] = np.zeros(shape)
            data[name + half + "/commands"] = np.asarray([command(k) for k in sample_steps])
            data[name + half + "/time"] = np.asarray([(k + 1) * rod.dt for k in sample_steps])
    return {**data, "engine": "reference"}, {**data, "engine": "ours"}


def test_requires_every_tube_and_refinement_trace():
    ref, ours = archives()
    parity.validate_archives(ref, ours)
    del ours["tube_bend@half/reaction"]
    with pytest.raises(ValueError, match="missing or nonfinite"):
        parity.compare(ref, ours)


@pytest.mark.parametrize("fault", ["complete", "schema", "revision", "dirty", "nan", "shape", "commands", "time", "rod"])
def test_rejects_invalid_qualification_inputs(fault):
    ref, ours = archives()
    if fault == "complete":
        ref["complete"] = False
    elif fault == "schema":
        ref["schema"] = 1
    elif fault in ("revision", "dirty"):
        ref["source"] = json.dumps({"revision": "e4bed31" if fault == "revision" else parity.REFERENCE_REVISION,
                                    "status": " M solver.py" if fault == "dirty" else ""})
    elif fault == "rod":
        ours["rod"] = "{}"
    else:
        field = fault if fault in ("commands", "time") else "positions"
        ours["sag/" + field] = np.full((1, 3) if fault == "shape" else (2, 3), np.nan if fault == "nan" else 1.0)
    with pytest.raises(ValueError):
        parity.compare(ref, ours)
