# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C3.0 capability/order probe only; no instrument ownership migration."""
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from i4h_physics.newton_compat import NEWTON_REVISION, require_model_finalizer


@pytest.mark.parametrize("hook", [None, 1, lambda: None, lambda key: None])
def test_missing_or_incompatible_hook_names_required_contract(hook):
    with pytest.raises(RuntimeError, match=NEWTON_REVISION):
        require_model_finalizer(SimpleNamespace(_add_custom_attribute_model_finalizer=hook))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_finalizer_order_templates_and_retained_builder(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    builder = newton.ModelBuilder()
    assignment = newton.Model.AttributeAssignment
    frequency = newton.Model.AttributeFrequency.ONCE
    calls = []
    for name, owner in (("early", assignment.MODEL), ("geometry", assignment.MODEL),
                        ("history", assignment.STATE), ("command", assignment.CONTROL),
                        ("initialize", assignment.MODEL)):
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name=name, namespace="probe", dtype=wp.float32, frequency=frequency,
            assignment=owner, default=2.0))

    def finish(b, model, attr):
        assert b is builder
        calls.append((model, attr.name))
        if attr.name == "early":
            assert not hasattr(getattr(model, "probe", None), "geometry")
        else:
            # A finalizer is per-attribute, not a global after-finalize hook.
            # Register the initializer attribute after every prerequisite template.
            np.testing.assert_array_equal(model.probe.geometry.numpy(), [2.0])
            np.testing.assert_array_equal(model.probe.history.numpy(), [2.0])
            np.testing.assert_array_equal(model.probe.command.numpy(), [2.0])
            model.probe.history.fill_(7.0)
            model.probe.command.fill_(11.0)
        model.add_attribute(attr.name, wp.array([1.0], dtype=wp.float32, device=model.device),
                            attr.frequency, attr.assignment, attr.namespace, attr.references)

    register = require_model_finalizer(builder)
    register("probe:early", finish)
    register("probe:initialize", finish)
    register("probe:initialize", finish)  # same callback is idempotent
    with pytest.raises(ValueError, match="different callback"):
        register("probe:initialize", lambda *args: None)
    first = builder.finalize(device=device)
    live_state, live_control = first.state(), first.control()
    live_state.probe.history.fill_(17.0)
    live_control.probe.command.fill_(19.0)
    second = builder.finalize(device=device)
    assert [(id(m), name) for m, name in calls] == [
        (id(first), "early"), (id(first), "initialize"),
        (id(second), "early"), (id(second), "initialize")]
    for name in ("geometry", "history", "command"):
        assert getattr(first.probe, name).ptr != getattr(second.probe, name).ptr
    np.testing.assert_array_equal(live_state.probe.history.numpy(), [17.0])
    np.testing.assert_array_equal(live_control.probe.command.numpy(), [19.0])
    for model in (first, second):
        state, control = model.state(), model.control()
        assert state.probe.history.ptr != model.probe.history.ptr
        assert control.probe.command.ptr != model.probe.command.ptr
        np.testing.assert_array_equal(state.probe.history.numpy(), [7.0])
        np.testing.assert_array_equal(control.probe.command.numpy(), [11.0])
        # Ordinary live sharing does not mutate a template.
        shared = control
        shared.probe.command.fill_(23.0)
        np.testing.assert_array_equal(model.probe.command.numpy(), [11.0])
        # Explicit opt-in template alias has intentionally different semantics.
        alias = model.control(clone_variables=False)
        assert alias.probe.command.ptr == model.probe.command.ptr
        alias.probe.command.fill_(29.0)
        np.testing.assert_array_equal(model.control().probe.command.numpy(), [29.0])
    np.testing.assert_array_equal(first.probe.geometry.numpy(), [2.0])
