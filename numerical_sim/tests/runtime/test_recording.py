# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M14 teleop (T3): teleop demos in Isaac Lab's HDF5 format and their replay. Isaac Lab venv only; fake devices only."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")
pytest.importorskip("h5py")

from helpers import haptic_case  # noqa: E402
from i4h_isaaclab.composition import read_case  # noqa: E402
from i4h_isaaclab.env import I4hEnv, I4hEnvCfg  # noqa: E402
from i4h_isaaclab.recording import Recorder, load_demo, replay_episode, replay_path  # noqa: E402
from i4h_standalone.teleop import Teleop  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def make_env(case, num_envs=2, backend="newton"):
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    cfg = I4hEnvCfg(case=str(case), action_mode="physical", stagger_episodes=False, backend=backend, episode_length_s=1.0)
    cfg.scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=1.0)
    return I4hEnv(cfg)


def close(env):
    env.close()
    SimulationContext.clear_instance()


def record_two_episodes(path, case_path, inputs, backend="newton", steps=12):
    env = make_env(case_path, backend=backend)
    teleop = Teleop(env, inputs)
    recorder = Recorder(path, teleop, record_camera=False)
    try:
        teleop.reset()
        names = []
        for success in (True, False):
            recorder.start()
            for _ in range(steps):
                obs, _ = teleop.step()
                recorder.record(obs)
            names.append(recorder.end(success))
        recorder.start()
        teleop.step()
        recorder.discard()  # a discarded take writes nothing
        assert recorder.end(True) is None
    finally:
        recorder.close()
        teleop.close()
        close(env)
    return recorder.path, names


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
    u = SimpleNamespace(cfg=cfg, _case={}, num_envs=1, physics_dt=1.0 / 60.0, simulator=SimpleNamespace(systems=[]))
    recorder = Recorder(tmp_path / "endo", SimpleNamespace(u=u))
    try:
        assert recorder.tool_force is None
    finally:
        recorder.close()


def test_record_two_episodes_and_replay(tmp_path):
    """Two fake-device episodes on two_worlds: the file loads with Isaac Lab's handler, carries what the teleop loop
    applied and saw, and replaying demo_0 from a reset reproduces env 0's observations."""
    case_path = haptic_case(tmp_path, envs=2)
    inputs = read_case(str(case_path))["inputs"]
    path, names = record_two_episodes(tmp_path / "demos", case_path, inputs)
    assert names == ["demo_0", "demo_1"] and path.endswith(".hdf5")
    episode, env_args = load_demo(path, "demo_0")
    data = episode.data
    assert episode.success and data["actions"].shape == (12, 10) and data["obs"]["policy"].shape[0] == 12
    assert set(data["devices"]["pad"]) == {"position", "rotation", "grip", "buttons"}
    assert data["haptics"]["force"].shape == (12, 1, 3) and any(k.startswith("tool_force.") for k in data["states"])
    assert (data["actions_normalized"].abs() <= 1.0).all()
    assert env_args["backend"] == "newton" and "instruments:" in env_args["case_yaml"]
    second, _ = load_demo(path, "demo_1")
    assert not second.success
    assert replay_path(f"{path}:demo_1") == (type(replay_path(path)[0])(path), "demo_1") and replay_path(path)[1] is None

    env = make_env(case_path)
    try:
        diffs = replay_episode(env, episode)
    finally:
        close(env)
    assert diffs.shape == (12,) and diffs.max() < 1e-3, diffs  # atomics noise ~1e-5 (relative); a wrong action ~1e-1


def test_every_take_replays_from_its_start(tmp_path):
    """Takes started by K (save + next), X (discard + restart) and R (stop, then resume) after the env has moved on
    begin from a reset, so each replays on its own, not only the first."""
    case_path = haptic_case(tmp_path, envs=2)
    inputs = read_case(str(case_path))["inputs"]
    env = make_env(case_path)
    teleop = Teleop(env, inputs)
    recorder = Recorder(tmp_path / "takes", teleop)

    def take(steps):
        for _ in range(steps):
            obs, _ = teleop.step()
            recorder.record(obs)

    try:
        teleop.reset()
        recorder.start(reset=False)
        take(10)
        recorder.end(True)  # K: demo_0 saved ...
        recorder.start()  # ... and the next take starts, after 10 steps of motion
        take(10)
        recorder.end(True)  # demo_1
        recorder.start()
        take(5)
        recorder.discard()  # X ...
        recorder.start()  # ... and restart
        take(10)
        recorder.end(False)  # R stops: demo_2
        for _ in range(5):
            teleop.step()  # stopped: not recorded
        recorder.start()  # R resumes
        take(10)
        recorder.end(True)  # demo_3
    finally:
        recorder.close()
        teleop.close()
        close(env)
    assert recorder.written == ["demo_0", "demo_1", "demo_2", "demo_3"]
    for demo in recorder.written[1:]:
        episode, _ = load_demo(recorder.path, demo)
        env = make_env(case_path)
        try:
            diffs = replay_episode(env, episode)
        finally:
            close(env)
        assert diffs.shape == (10,) and diffs.max() < 1e-3, (demo, diffs)


def test_replay_on_the_other_backend_matches(tmp_path):
    """A demo recorded on the newton backend replays on the fallback backend: same kernels, same observations."""
    case_path = haptic_case(tmp_path, envs=2)
    inputs = read_case(str(case_path))["inputs"]
    path, _ = record_two_episodes(tmp_path / "demos", case_path, inputs, backend="newton", steps=8)
    episode, _ = load_demo(path)
    env = make_env(case_path, backend="i4h")
    try:
        diffs = replay_episode(env, episode)
    finally:
        close(env)
    assert diffs.max() < 1e-3, diffs


def test_replay_needs_physical_actions(tmp_path):
    case_path = haptic_case(tmp_path, envs=2)
    path, _ = record_two_episodes(tmp_path / "demos", case_path, read_case(str(case_path))["inputs"], steps=2)
    episode, _ = load_demo(path)
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    cfg = I4hEnvCfg(case=str(case_path), stagger_episodes=False)
    cfg.scene = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    env = I4hEnv(cfg)
    try:
        with pytest.raises(ValueError, match="physical"):
            replay_episode(env, episode)
    finally:
        close(env)
    assert np.isfinite(episode.data["actions"].cpu().numpy()).all()


def test_replay_rebuilds_the_recorded_env_after_moving_case_and_data(tmp_path, monkeypatch):
    """A long take on non-default settings replays from the recording alone: the case file is gone, the assets live in
    another data directory, and decimation, workspace, horizon (longer than the 5 s default) and staggering come from
    the recording."""
    import re
    import shutil
    import warnings

    from i4h_isaaclab.recording import env_cfg_from_recording
    from i4h_physics.config import data_dir

    case_path = haptic_case(tmp_path, envs=2)
    inputs = read_case(str(case_path))["inputs"]
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    cfg = I4hEnvCfg(case=str(case_path), action_mode="physical", stagger_episodes=False, decimation=3, workspace=0.3,
                    episode_length_s=30.0)
    cfg.scene = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    env = I4hEnv(cfg)
    teleop = Teleop(env, inputs)
    recorder = Recorder(tmp_path / "long", teleop)
    try:
        teleop.reset()
        recorder.start(reset=False)
        for _ in range(120):  # 120 x 0.05 s = 6 s, past the default 5 s horizon
            obs, done = teleop.step()
            assert not done
            recorder.record(obs)
        recorder.end(True)
    finally:
        recorder.close()
        teleop.close()
        close(env)
    episode, env_args = load_demo(recorder.path)
    root = str(data_dir())
    assert env_args["format_version"] == 1 and root not in env_args["case_yaml"]
    assert (env_args["decimation"], env_args["workspace"], env_args["episode_length_s"]) == (3, 0.3, 30.0)
    # relocate: a new data directory holding only the referenced assets, and no case file
    moved = tmp_path / "moved_data"
    for rel in set(re.findall(r"\$\{DATA\}/([^\s'\"]+)", env_args["case_yaml"])):
        target = moved / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(root) / rel, target)
    assert any(p.is_file() for p in moved.rglob("*")), "the case references assets under ${DATA}"
    case_path.unlink()
    monkeypatch.setenv("I4H_DATA", str(moved))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing differs: no warning
        cfg = env_cfg_from_recording(env_args, workdir=tmp_path)
    assert (cfg.decimation, cfg.workspace, cfg.episode_length_s, cfg.stagger_episodes) == (3, 0.3, 30.0, False)
    assert str(moved) in Path(cfg.case).read_text()
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    env = I4hEnv(cfg)
    try:
        diffs = replay_episode(env, episode, env_args)
        bad = dict(env_args, action_layout=[["other", 3]])
        with pytest.raises(ValueError, match="action layout"):
            replay_episode(env, episode, bad)
    finally:
        close(env)
    assert diffs.shape == (120,) and diffs.max() < 1e-3, diffs.max()
    with pytest.warns(RuntimeWarning, match="decimation=2"):
        env_cfg_from_recording(env_args, {"decimation": 2, "num_envs": 1}, workdir=tmp_path)
    legacy = {k: env_args[k] for k in ("case", "decimation", "backend", "num_envs", "physics_dt", "workspace")}
    with pytest.warns(DeprecationWarning, match="format_version"):
        assert env_cfg_from_recording(legacy).decimation == 3

    import importlib.util

    spec = importlib.util.spec_from_file_location("isaaclab_run", Path(__file__).parents[2] / "scripts" / "isaaclab_run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    args = run.parse(["--policy", f"replay_hdf5:{recorder.path}"])
    cfg = run.make_cfg(args)
    assert (cfg.decimation, cfg.workspace, cfg.episode_length_s, cfg.scene.num_envs) == (3, 0.3, 30.0, 2)
    cfg = run.make_cfg(run.parse(["--policy", f"replay_hdf5:{recorder.path}", "--num_envs", "1"]))
    assert cfg.scene.num_envs == 1 and cfg.decimation == 3


def test_takes_replay_with_episode_staggering(tmp_path):
    """Staggering gives a fresh env's first episode a random counter; a later take starts at 0. Each take stores its
    starting counter and the replay restores it, so time-outs fall on the recorded steps (a 9-step horizon here)."""
    case_path = haptic_case(tmp_path, envs=2)
    inputs = read_case(str(case_path))["inputs"]

    def staggered_env():
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
        cfg = I4hEnvCfg(case=str(case_path), action_mode="physical", stagger_episodes=True, episode_length_s=0.3)
        cfg.scene = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
        return I4hEnv(cfg)

    env = staggered_env()
    teleop = Teleop(env, inputs)
    recorder = Recorder(tmp_path / "staggered", teleop)
    try:
        teleop.reset()
        recorder.start(reset=False)
        for _ in range(6):
            obs, _ = teleop.step()
            recorder.record(obs)
        recorder.end(True)  # demo_0: from the staggered counter
        recorder.start()
        for _ in range(8):
            obs, _ = teleop.step()
            recorder.record(obs)
        recorder.end(True)  # demo_1: from 0
    finally:
        recorder.close()
        teleop.close()
        close(env)
    for demo in ("demo_0", "demo_1"):
        episode, env_args = load_demo(recorder.path, demo)
        env = staggered_env()
        try:
            diffs = replay_episode(env, episode, env_args)
        finally:
            close(env)
        assert diffs.max() < 1e-3, (demo, diffs)


def test_action_layout_names_the_instruments_in_order(tmp_path):
    """Two instruments in the other order: the same width, but the recorded rows would drive the other tool."""
    from types import SimpleNamespace

    import yaml
    from i4h_isaaclab.composition import load_case
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
