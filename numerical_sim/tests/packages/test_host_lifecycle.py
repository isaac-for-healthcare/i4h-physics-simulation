# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise hosted reset and callback contracts without booting Kit."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_mixed_reset_preserves_restored_history():
    path = Path(__file__).parents[2] / "isaaclab/src/i4h_isaaclab/manager.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "I4hHostMixin"
    )
    namespace = {"torch": torch, "I4hComponent": object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    trace = []

    def event(label):
        return lambda ids=None, **kw: trace.append(
            (label, (ids if ids is not None else kw.get("env_ids", torch.tensor([]))).tolist())
        )

    class Base:
        def _reset_idx(self, ids):
            trace.append(("lab_events", ids.tolist()))

    class Host(namespace["I4hHostMixin"], Base):
        def i4h_restore_ids(self, ids):
            return ids[ids == 1]

        def i4h_restore_state(self, ids):
            event("restore")(ids)

        def i4h_save_terminal(self, ids):
            event("terminal")(ids)

        def i4h_after_reset(self, ids):
            event("arena_reset")(ids)

    host = Host.__new__(Host)
    host.i4h = SimpleNamespace(
        reset=event("physical"),
        sync_views=lambda: trace.append(("sync", [])),
        initialize_particle_history=event("history"),
        reset_procedure=event("procedure"),
    )
    host.i4h_task = None
    host.extras = {}
    host._reset_idx(torch.tensor([0, 1, 2]))
    assert trace == [
        ("terminal", [0, 1, 2]),
        ("physical", [0, 2]),
        ("sync", []),
        ("lab_events", [0, 1, 2]),
        ("history", [0, 2]),
        ("arena_reset", [0, 2]),
        ("restore", [1]),
        ("sync", []),
    ]


def method(filename, classname, name):
    path = Path(__file__).parents[2] / "isaaclab/src/i4h_isaaclab" / filename
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == classname)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_post_physics_registration_requires_rebuild():
    register = method("component.py", "I4hComponent", "on_post_physics")
    binding = SimpleNamespace(post_physics=[])
    owner = SimpleNamespace(_live=lambda: binding)
    def callback():
        pass

    with pytest.raises(RuntimeError, match="before graph capture"):
        register(owner, callback)
    owner._binding_callbacks_open = True
    register(owner, callback)
    assert binding.post_physics == [callback]
    owner._binding_callbacks_open = False
    with pytest.raises(RuntimeError, match="before graph capture"):
        register(owner, callback)


def test_post_physics_runs_after_procedure_on_every_physics_step():
    step = method("newton_manager.py", "ConstructionContext", "physics_step")
    trace = []
    binding = SimpleNamespace(
        dt=1 / 60,
        procedure=SimpleNamespace(update=lambda dt: trace.append(("procedure", dt))),
        post_physics=[lambda: trace.append("probe"), lambda: trace.append("host")],
    )
    owner = SimpleNamespace(binding=binding)
    step(owner)
    step(owner)
    assert trace == [("procedure", 1 / 60), "probe", "host"] * 2
