# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M14 teleop (T3): teleop demos in Isaac Lab's HDF5 format and their replay. Isaac Lab venv only; fake devices only."""


import pytest


pytest.importorskip("isaaclab.envs")


pytest.importorskip("h5py")


from helpers import haptic_case  # noqa: E402


from i4h_isaaclab.recording import Recorder, load_demo, replay_episode, replay_path  # noqa: E402


def test_recorder_without_surgical_package(tmp_path, monkeypatch):
    """An endoluminal-only install (no i4h_surgical) can construct a Recorder."""
    import sys
    from types import SimpleNamespace

    class NotInstalled:  # what a missing distribution raises: ModuleNotFoundError(name="i4h_surgical")
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname == "i4h_surgical":
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    for name in [m for m in sys.modules if m == "i4h_surgical" or m.startswith("i4h_surgical.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NotInstalled(), *sys.meta_path])
    cfg = SimpleNamespace(case="aorta_static", backend="newton", decimation=2, workspace=0.5, seed=0, observe="", camera=0,
                          camera_mode="", episode_length_s=5.0, stagger_episodes=True, step_weights=())
    u = SimpleNamespace(cfg=cfg, resolved_case={}, num_envs=1, physics_dt=1.0 / 60.0, simulator=SimpleNamespace(systems=[]))
    recorder = Recorder(tmp_path / "endo", SimpleNamespace(u=u))
    try:
        assert recorder.tool_force is None
    finally:
        recorder.close()


def test_action_layout_names_the_instruments_in_order(tmp_path):
    """Two instruments in the other order: the same width, but the recorded rows would drive the other tool."""
    from types import SimpleNamespace

    import yaml
    from i4h_host.composition import load_case
    from i4h_isaaclab.recording import action_layout, check_replay_compatible

    cfg = yaml.safe_load(haptic_case(tmp_path, envs=2).read_text())
    probe = cfg["instruments"]["probe"]
    other = dict(probe, initial={"position": [0.1, 0.0, -0.1]})
    layouts = []
    for order in (("probe", "other_probe"), ("other_probe", "probe")):
        cfg["instruments"] = {name: probe if name == "probe" else other for name in order}
        layouts.append(action_layout(load_case(cfg)))
    assert [entry[1] for entry in layouts[0]] == [entry[1] for entry in layouts[1]], "same widths"
    assert layouts[0] != layouts[1] and layouts[0][0][2] == ["probe", "other_probe"]
    with pytest.raises(ValueError, match="action layout"):
        check_replay_compatible(SimpleNamespace(simulator=load_case(cfg)), {"action_layout": layouts[0]})
