# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify all six installed member roots outside the checkout, with no forwarding package."""
import importlib
import importlib.metadata
import sys
from importlib.util import find_spec
from pathlib import Path

from i4h_physics import Body, I4hSolver, PhysicsBinding, Scene, Stepper, System
from i4h_standalone.app.cli import main

expected = Path(sys.argv[1]).resolve()
assert find_spec("i4h_sim") is None
for member in ("physics", "assessment", "endoluminal", "surgical", "isaaclab", "standalone"):
    name = f"i4h_{member}"
    package = importlib.import_module(name)
    assert Path(package.__file__).resolve().is_relative_to(expected), (name, package.__file__, expected)
    assert importlib.metadata.version(f"i4h-{member}") == "0.9.0"
    print(f"{name}: installed root and version passed")
assert all(callable(value) for value in (Scene, Body, System, I4hSolver, Stepper, PhysicsBinding, main))
