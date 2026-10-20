# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M14 teleop (T1): the app's device path drives env 0 of an I4hEnv in physical-action mode, with force feedback, on
both backends. Isaac Lab venv only (``$PY -m pytest tests/runtime/test_teleop.py``); stub devices only, forces within
the case's 0.5 N cap."""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")

from helpers import haptic_case, wait_until  # noqa: E402
from i4h_isaaclab.composition import read_case  # noqa: E402
from i4h_isaaclab.env import I4hEnv, I4hEnvCfg  # noqa: E402
from i4h_isaaclab.views import physical_to_normalized  # noqa: E402
from i4h_standalone.app.haptics import HapticsConfig  # noqa: E402
from i4h_standalone.teleop import Teleop, fake_inputs  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


@pytest.fixture(params=["newton", "i4h"])
def backend(request):
    return request.param


@pytest.fixture
def make_env(backend):
    envs = []

    def _make(case, num_envs=2, backend=backend, **kw):
        for env in envs:
            env.close()
        envs.clear()
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
        cfg = I4hEnvCfg(case=str(case), action_mode=kw.pop("action_mode", "physical"), stagger_episodes=False, **kw)
        cfg.scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=1.0)
        cfg.backend = backend
        env = I4hEnv(cfg)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()


def test_fake_devices_drive_env0_with_capped_force_feedback(make_env, tmp_path):
    """The fake pad presses the probe into the block: the physical rows reach env 0 unclamped, env 1 holds its reset
    pose, the servo sends the reaction force within the 0.5 N cap, and env 0's time-out zeroes it until fresh data."""
    path = haptic_case(tmp_path, envs=2)
    case = read_case(str(path))
    env = make_env(path, episode_length_s=1.0)
    u = env.unwrapped
    teleop = Teleop(env, case["inputs"], HapticsConfig(**case["haptics"]))
    try:
        teleop.reset()
        held = teleop.physical[1].clone()
        pad = teleop.haptics.servo.devices["pad"]["source"]
        for _ in range(20):
            teleop.step()
            time.sleep(1.0 / 240.0)  # servo ticks between steps
        torch.cuda.synchronize()
        written = torch.cat([v for v in u.action_views], dim=1)
        np.testing.assert_allclose(written[0, teleop.columns].cpu().numpy(), teleop.rows.reshape(-1), atol=1e-6)
        assert teleop.rows[0, 2] == pytest.approx(0.05)  # the pad's z as written: a physical row, not mapped from [-1, 1]
        assert torch.equal(written[1], held), "env 1 holds its reset pose"
        wait_until(lambda: any(np.linalg.norm(f) > 1e-4 for _, f in pad.forces[-50:]), timeout=2.0)  # the reaction arrives
        norms = [np.linalg.norm(f) for _, f in pad.forces]
        assert max(norms) <= 0.5 + 1e-6, max(norms)  # the YAML cap
        u.episode_length_buf[0] = u.max_episode_length - 1
        _, done = teleop.step()
        assert done
        time.sleep(0.02)
        assert pad.forces[-1][1] == (0.0, 0.0, 0.0), "the servo zeroes after env 0's reset until a fresh sample"
        assert teleop.haptics.servo.status()["devices"]["pad"].reason == "reset"
    finally:
        teleop.close()
    assert pad.forces[-1][1] == (0.0, 0.0, 0.0)


def test_mirror_and_hold(make_env, tmp_path):
    path = haptic_case(tmp_path, envs=3)
    env = make_env(path, num_envs=3)
    teleop = Teleop(env, read_case(str(path))["inputs"], others="mirror")
    try:
        teleop.reset()
        for _ in range(3):
            teleop.step()
        assert torch.equal(teleop.physical[1, teleop.columns], teleop.physical[0, teleop.columns])
        assert torch.equal(teleop.physical[2, teleop.columns], teleop.physical[0, teleop.columns])
    finally:
        teleop.close()


def test_replay_rows_match_the_app(make_env, tmp_path):
    """Replay parity: for the same recorded session, the rows the teleop loop feeds env 0 equal the rows the app's
    InstrumentDriver uploads, frame for frame (the two share DeviceRows)."""
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.app.device_rows import replay_inputs
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    rng = np.random.default_rng(0)
    session = np.zeros((30, 10), np.float32)
    session[:, :3] = [0.0, 0.0, 0.03] + 0.01 * rng.standard_normal((30, 3))
    session[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    session[:, 7:] = rng.random((30, 3)) > 0.5
    base = tmp_path / "session"
    np.save(tmp_path / "session_probe.npy", session)

    sim = load_case("two_worlds")
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    inputs = replay_inputs(sim.scene.instruments.values(), base)
    assert inputs, "the recording names the probe"
    driver = InstrumentDriver(sim, control, inputs, "devices")
    app_rows = []
    for n in range(30):
        driver.update(n * sim.dt)
        app_rows.append(control.actions.numpy()[0].copy())
    driver.close()

    env = make_env("two_worlds")
    teleop = Teleop(env, replay_inputs(env.unwrapped.simulator.scene.instruments.values(), base))
    try:
        teleop.reset()
        teleop_rows = []
        for _ in range(30):
            teleop.step()
            teleop_rows.append(teleop.rows.copy())
    finally:
        teleop.close()
    np.testing.assert_array_equal(np.asarray(teleop_rows), np.asarray(app_rows))


def test_physical_mode_equals_normalized_mode(make_env):
    """The same actions, once normalised and once as the physical rows they map to: identical observations."""
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(5)
    sequence = [torch.rand((2, 10), generator=gen, device=dev) * 2 - 1 for _ in range(20)]
    results = {}
    for mode in ("normalized", "physical"):
        env = make_env("two_worlds", action_mode=mode)
        u = env.unwrapped
        env.reset()
        for a in sequence:
            obs, *_ = env.step(a if mode == "normalized" else u._mid + u._half * a)
        torch.cuda.synchronize()
        results[mode] = obs["policy"].clone()
    assert torch.allclose(results["normalized"], results["physical"], atol=1e-5)


def test_fake_inputs_and_physical_to_normalized():
    inputs = fake_inputs(read_case("chole_tet_med")["inputs"])
    assert "box" not in inputs and inputs["pickup"]["keys"]
    assert all(spec["backend"] == "fake" for dev, spec in inputs.items() if dev != "pickup")
    assert inputs["primary"]["settings"]["position_offset"] == [0.0, 1.2, -4.0]  # the calibration stays
    low, high = np.float32([-1.0, 0.0]), np.float32([1.0, 4.0])
    n, excess = physical_to_normalized(np.float32([[0.0, 2.0], [2.0, 4.0]]), low, high)
    np.testing.assert_allclose(n, [[0.0, 0.0], [1.0, 1.0]])  # clipped
    assert excess == pytest.approx(1.0)  # x = 2 is one half-range beyond the upper bound
