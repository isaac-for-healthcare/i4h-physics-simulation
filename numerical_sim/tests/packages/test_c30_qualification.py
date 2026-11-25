# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The pin checkpoint permits only its declared dependency transition."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def qualification(monkeypatch):
    scripts = Path(__file__).parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("c30_qualification", scripts / "qualify_stage_b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", [None, "newton", "numpy", "gpu", "commit"])
def test_exact_transition_does_not_relax_other_identity_checks(qualification, monkeypatch, change):
    revision = qualification.C30_NEWTON_REVISION
    candidate = {"software": {"newton": "1.9.0.dev0", "numpy": "2.4.6"}, "gpu": "same"}
    reference = {"software": {"newton": "1.8.0rc1", "numpy": "2.4.6"}, "gpu": "same"}
    commit = revision
    if change in ("newton", "numpy"):
        candidate["software"][change] = "other"
    elif change == "gpu":
        candidate["gpu"] = "other"
    elif change == "commit":
        commit = "other"
    monkeypatch.setattr(qualification.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        read_text=lambda name: json.dumps({"vcs_info": {"commit_id": commit}})))
    if change:
        with pytest.raises(ValueError):
            qualification.check_reference_identity(candidate, reference, revision)
    else:
        qualification.check_reference_identity(candidate, reference, revision)
        assert candidate["software"]["newton"] == "1.9.0.dev0"
    with pytest.raises(ValueError):
        qualification.check_reference_identity(candidate, reference)
