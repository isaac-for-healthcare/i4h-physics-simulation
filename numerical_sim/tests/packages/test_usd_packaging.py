# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD providers are exclusive on standalone platforms and opt-in for Lab."""
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("machine,provider", [("x86_64", "usd-core"), ("aarch64", "usd-exchange"), ("arm64", "usd-core")])
@pytest.mark.parametrize("member", ["", "surgical", "endoluminal"])
def test_platform_selects_one_usd_provider(member, machine, provider):
    project = tomllib.loads((ROOT / member / "pyproject.toml").read_text())["project"]
    requirements = project["optional-dependencies"]["usd"] if member else project["dependencies"]
    selected = [req.name for raw in requirements if (req := Requirement(raw)).name.startswith("usd-")
                and (req.marker is None or req.marker.evaluate({"platform_machine": machine}))]
    assert selected == [provider]


@pytest.mark.parametrize("member", ["physics", "assessment", "surgical", "endoluminal", "isaaclab", "standalone"])
def test_lab_base_install_does_not_request_usd(member):
    project = tomllib.loads((ROOT / member / "pyproject.toml").read_text())["project"]
    for raw in project.get("dependencies", []):
        req = Requirement(raw)
        assert not req.name.startswith("usd-") and "usd" not in req.extras
