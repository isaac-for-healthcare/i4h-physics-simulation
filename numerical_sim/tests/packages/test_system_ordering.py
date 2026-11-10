# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The shipped cases retain their authored order and the solver's five hooks."""

import pytest

from i4h_isaaclab.composition import maps, read_case
from i4h_physics.config import load_case


@pytest.mark.parametrize("case", ["two_worlds", "chole_tet_med", "aorta_static", "aorta_shell", "aorta_centreline", "tissue_contact"])
def test_shipped_case_order(case):
    cfg, _, constructors = maps(read_case(case))
    specs = cfg["systems"]
    kinds = [s["type"] for s in specs]
    assert kinds[0] == "particles"
    expected_last = {"aorta_static": "rod_containment", "aorta_shell": "shell_containment", "aorta_centreline": "drag"}
    assert kinds[-1] == expected_last.get(case, "bounds")
    ids = []
    for spec in specs:
        for ref in constructors[spec["type"]].refs:
            if ref in spec:
                assert spec[ref] in ids
        ids.append(spec.get("id", spec["type"]))
    if "instrument_contact" in kinds:
        assert kinds.index("surface") < kinds.index("instrument_contact") < kinds.index("tool_force")
    if "catheter_control" in kinds:
        rod = "rod_direct" if "rod_direct" in kinds else "rod_pbd"
        assert kinds.index(rod) < kinds.index("catheter_control")


def test_five_hook_loop_preserves_list_order(monkeypatch):
    from i4h_physics.systems.base import System

    class A(System):
        pass

    class B(System):
        pass

    sim = load_case({"sim": {"device": "cpu", "substeps": 2, "iterations": 3, "cuda_graph": False},
                     "systems": [{"type": "a"}, {"type": "b"}]}, systems={"a": A, "b": B})
    calls = []
    for system in sim.systems:
        for hook in ("step_start", "substep_start", "iteration", "substep_end", "step_end"):
            monkeypatch.setattr(system, hook, lambda *args, name=system.id, phase=hook: calls.append((phase, name)))
    sim.step()
    phases = ["step_start"] + (["substep_start"] + ["iteration"] * 3 + ["substep_end"]) * 2 + ["step_end"]
    assert calls == [(phase, name) for phase in phases for name in ("a", "b")]
