# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""scripts/isaaclab_run.py --check (PHYS-5.1) compares Warp and Newton with i4h-physics' own pins, not with numbers
of its own: after the Newton 1.6.1 bump it still asserted 1.6.0 and failed on every install."""

import importlib.util
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_check_pins_are_the_physics_package_pins():
    spec = importlib.util.spec_from_file_location("isaaclab_run", ROOT / "scripts" / "isaaclab_run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    declared = tomllib.loads((ROOT / "physics" / "pyproject.toml").read_text())["project"]["dependencies"]
    expected = dict(d.split("==") for d in declared if "==" in d)
    pins = run.simulator_pins()
    assert pins["warp-lang"] == expected["warp-lang"] and pins["newton"] == expected["newton"], (pins, expected)
    source = (ROOT / "scripts" / "isaaclab_run.py").read_text()
    assert not re.search(r'__version__ == "\d', source), "no hard-coded version in the check"
