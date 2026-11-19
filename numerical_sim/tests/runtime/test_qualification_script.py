# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the qualification script's host-owned assessment loop on real physics."""
import os
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("case", ["aorta_static_reach", "aorta_centreline_reach", "chole_grasp"])
@pytest.mark.skipif(not os.environ.get("I4H_DATA"), reason="external cases require I4H_DATA")
def test_qualification_script(case, graph):
    script = Path(__file__).resolve().parents[2] / "scripts" / "qualify_procedures.py"
    qualify = runpy.run_path(str(script))["qualify"]
    result = qualify(case, graph=graph, lifecycle=case == "aorta_static_reach")
    assert result["result"] == "success"
