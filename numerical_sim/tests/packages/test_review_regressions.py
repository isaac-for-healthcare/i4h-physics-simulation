# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public CLI recording and teardown regressions with generated or small packaged cases."""


import dataclasses
from types import SimpleNamespace

import pytest


def test_missing_lab_api_names_the_qualified_revision():
    from i4h_isaaclab.lab import QUALIFIED, require

    require(SimpleNamespace(present=1), "present")

    @dataclasses.dataclass
    class Cfg:
        compute_final_obs: list = dataclasses.field(default_factory=list)  # a field with no class attribute

    require(Cfg, "compute_final_obs")
    with pytest.raises(RuntimeError, match="_invalidate_graph.*4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1"):
        require(type("NewtonManager", (), {}), "_invalidate_graph")
    assert "e53cd288727a5ca3dae7187d1d415b4d32b67e11" in QUALIFIED
    assert "newton18-lab30-cloner.patch" in QUALIFIED


def test_nested_alias_checks_check_once_and_restore_at_the_outermost(monkeypatch):
    import warp as wp
    from i4h_physics import debug

    original, checks = wp.launch, []
    monkeypatch.setattr(debug, "assert_no_alias", lambda kernel, inputs, outputs: checks.append(kernel))

    @wp.kernel
    def touch(a: wp.array(dtype=float)):
        a[wp.tid()] = 1.0

    a = wp.zeros(2, dtype=float, device="cpu")
    with debug.alias_checks():
        with debug.alias_checks():
            wp.launch(touch, 2, outputs=[a], device="cpu")
        assert len(checks) == 1, "a nested context does not wrap twice"
        with pytest.raises(RuntimeError, match="inner"):
            with debug.alias_checks():
                raise RuntimeError("inner")
        assert wp.launch is not original, "the inner exception left the outer context's wrapper in place"
        wp.launch(touch, 2, outputs=[a], device="cpu")
        assert len(checks) == 2
    assert wp.launch is original
