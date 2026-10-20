# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M14 teleop (T2): Isaac Lab ``DeviceBase`` adapters. Isaac Lab venv only; stub (fake) devices only, forces within the
case's 0.5 N cap."""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")

from helpers import haptic_case, wait_until  # noqa: E402
from i4h_standalone.devices import (  # noqa: E402
    I4hTeleopDevice,
    I4hTeleopDeviceCfg,
    IsaacLabDeviceSource,
    configure_teleop,
)
from isaaclab.devices.device_base import DeviceBase  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def make_device(tmp_path, **kw):
    cfg = I4hTeleopDeviceCfg(case=str(haptic_case(tmp_path, envs=2)), **kw)
    return cfg.class_type(cfg)


class Press:
    def __init__(self, name):
        from types import SimpleNamespace

        self.key = SimpleNamespace(name=name)

    def is_key_press(self):
        return True

    def is_key_release(self):
        return False

    def is_key_repeat(self):
        return False


def test_advance_rows_callbacks_and_reset(tmp_path):
    device = make_device(tmp_path, inputs={"pad": {"backend": "fake", "settings": {"position": [0.0, 0.0, 0.05], "grip": 0.7,
                                                                                    "buttons": {"secondary": True}}}})
    keys = type("Keys", (), {"key_handlers": []})()
    device.key_source = keys
    fired = []
    try:
        assert isinstance(device, DeviceBase) and "pad" in str(device)
        device.add_callback("secondary", lambda: fired.append("pedal"))
        device.add_callback("R", lambda: fired.append("reset"))
        with pytest.raises(ValueError, match="key_source"):
            I4hTeleopDevice.add_callback(type("D", (), {"key_source": None, "_callbacks": {}})(), "Q", lambda: None)
        for _ in range(3):
            out = device.advance()
        assert out.shape == (10,) and out.device.type == "cuda"
        row = out.cpu().numpy()
        assert row[2] == pytest.approx(0.05) and row[7] == pytest.approx(0.7) and row[8] == 1.0  # pose, grip, energy
        assert fired == ["pedal"], "a rising edge fires once, not every advance"
        keys.key_handlers[0](Press("r"))
        assert fired == ["pedal", "reset"]
        device.reset()
        np.testing.assert_array_equal(device.rows, device.initial)
    finally:
        device.close()
    idle = make_device(tmp_path, inputs={"pad": {"backend": "noop"}})
    try:
        assert idle.advance() is None  # no device reports yet
    finally:
        idle.close()


def test_push_force_select_sum_cap_and_exclusive_with_attach(tmp_path):
    device = make_device(tmp_path, haptics=True)  # the case's fake pad, max_force_n 0.5, newtons_per_unit 1e-3
    try:
        device.advance()
        pad = device.devices.manager.devices[0].source
        device.push_force(torch.tensor([[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]]), torch.tensor([0, 0]))  # summed: 0.2 N
        wait_until(lambda: pad.forces and pad.forces[-1][1][2] > 0.0, timeout=2.0)
        assert pad.forces[-1][1][2] == pytest.approx(0.2, rel=1e-3)
        device.push_force(np.array([[0.0, 0.0, 1e6]]))  # far beyond: the YAML cap holds
        wait_until(lambda: pad.forces[-1][1][2] > 0.3, timeout=2.0)
        assert max(np.linalg.norm(f) for _, f in pad.forces) <= 0.5 + 1e-6
        device.push_force(np.array([[np.nan, 0.0, 0.0]]))
        wait_until(lambda: pad.forces[-1][1] == (0.0, 0.0, 0.0), timeout=2.0)  # non-finite: zero
        with pytest.raises(ValueError, match="rows for 1 instruments"):
            device.push_force(np.zeros((2, 3)))
        with pytest.raises(ValueError, match="instrument index"):
            device.push_force(np.zeros((1, 3)), [3])
        device.reset()
        time.sleep(0.02)
        assert pad.forces[-1][1] == (0.0, 0.0, 0.0)  # a reset zeroes until the next push
    finally:
        device.close()
    assert pad.forces[-1][1] == (0.0, 0.0, 0.0)
    off = make_device(tmp_path)  # haptics off: push_force sends nothing, starts no servo
    try:
        off.push_force(np.zeros((1, 3)))
        assert off._servo is None
    finally:
        off.close()


@pytest.fixture
def two_worlds_env(tmp_path):
    from i4h_isaaclab.env import I4hEnv, I4hEnvCfg

    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    cfg = I4hEnvCfg(case=str(haptic_case(tmp_path, envs=2)), stagger_episodes=False)
    configure_teleop(cfg)
    cfg.scene = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    env = I4hEnv(cfg)
    yield env
    env.close()
    SimulationContext.clear_instance()


def test_attach_normalized_output_and_tool_force_feedback(tmp_path, two_worlds_env):
    """Attached to the env: normalised rows within [-1, 1] that the env maps back to the device's pose, and the env's
    tool_force reaching the fake pad through the servo within the cap."""
    env = two_worlds_env
    u = env.unwrapped
    assert u.cfg.teleop_devices.devices["i4h"].case == u.cfg.case  # Isaac Lab tooling finds a device for this case
    device = make_device(tmp_path, haptics=True, output="normalized")
    try:
        device.attach(env)
        with pytest.raises(ValueError, match="not both"):
            device.push_force(np.zeros((1, 3)))
        env.reset()
        pad = device.devices.manager.devices[0].source
        for _ in range(30):
            a = device.advance()
            assert a is None or (a.abs() <= 1.0).all()
            if a is not None:
                env.step(a.expand(2, -1))
            time.sleep(1.0 / 240.0)
        torch.cuda.synchronize()
        np.testing.assert_allclose(u.action_views[0][0, :3].cpu().numpy(), device.rows[0, :3], atol=1e-5)
        wait_until(lambda: any(np.linalg.norm(f) > 1e-4 for _, f in pad.forces[-50:]), timeout=2.0)
        assert max(np.linalg.norm(f) for _, f in pad.forces) <= 0.5 + 1e-6
    finally:
        device.close()


class FakeIsaacDevice(DeviceBase):
    """A kitless stand-in for an Isaac Lab device: returns the queued rows from ``advance()``."""

    def __init__(self, rows):
        super().__init__()
        self.rows = [torch.as_tensor(r, dtype=torch.float32) for r in rows]
        self.resets = 0

    def reset(self):
        self.resets += 1

    def add_callback(self, key, func):
        pass

    def advance(self):
        return self.rows.pop(0) if self.rows else None


def test_isaac_lab_devices_drive_an_instrument(tmp_path, two_worlds_env):
    """Absolute (Haply-style) and delta (Se3-style) outputs of another Isaac Lab device drive the two_worlds probe through
    DeviceRows and the teleop loop."""
    from i4h_standalone.teleop import Teleop

    q = [0.0, 0.0, 0.0, 1.0]
    absolute = IsaacLabDeviceSource(FakeIsaacDevice([[0.01, 0.02, 0.04, *q, 0.6, 1.0, 0.0]]))
    s = absolute.poll()
    assert s["active"] and np.allclose(s["position"], [0.01, 0.02, 0.04]) and s["grip"] == pytest.approx(0.6)
    assert s["buttons"] == {"secondary": True, "fire": False} and absolute.poll() == {"active": False}

    half_turn = [0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 1.0]
    delta = IsaacLabDeviceSource(FakeIsaacDevice([[0.01, 0.0, 0.0, 0, 0, 0], half_turn]), mode="delta", start=((0.0, 0.0, 0.03), q))
    assert np.allclose(delta.poll()["position"], [0.01, 0.0, 0.03])
    s = delta.poll()
    assert np.allclose(s["rotation"], [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-6) and s["grip"] == 1.0

    env = two_worlds_env
    env.unwrapped.cfg.action_mode = "physical"
    teleop = Teleop(env, {"pad": {"backend": "noop"}})
    try:
        source = IsaacLabDeviceSource(FakeIsaacDevice([[0.0, 0.0, 0.002, 0, 0, 0]] * 10), mode="delta", start=((0.0, 0.0, 0.03), q))
        teleop.devices.replace_source("pad", source)
        teleop.reset()
        for _ in range(10):
            teleop.step()
        torch.cuda.synchronize()
        assert teleop.rows[0, 2] == pytest.approx(0.05, abs=1e-6)  # 0.03 + 10 x 0.002
        assert env.unwrapped.action_views[0][0, 2].item() == pytest.approx(0.05, abs=1e-6)
    finally:
        teleop.close()
    assert source.device.resets == 1  # the source's close resets the Isaac Lab device


def test_registered_tasks_have_no_default_device_and_configured_ones_follow_case():
    """i4h-isaaclab does not depend on i4h-standalone: no task names its device unless configure_teleop adds it."""
    import i4h_isaaclab.tasks as tasks
    from i4h_isaaclab.env import sync_teleop_case

    for entry in {**tasks.TASKS, **tasks.PROCEDURE_TASKS}.values():
        cfg = getattr(tasks, entry)()
        assert not cfg.teleop_devices.devices
        configure_teleop(cfg)
        device = cfg.teleop_devices.devices["i4h"]
        assert device.case == cfg.case
        cfg.case = "two_worlds"
        sync_teleop_case(cfg)
        assert device.case == "two_worlds"


def test_lab_factory_constructs_configured_device(monkeypatch, tmp_path):
    from i4h_isaaclab.env import I4hEnvCfg
    from isaaclab_teleop.deprecated.teleop_device_factory import create_teleop_device

    cfg = configure_teleop(I4hEnvCfg(case=str(haptic_case(tmp_path, envs=2))), fake=True)
    device = create_teleop_device("i4h", cfg.teleop_devices.devices)
    try:
        assert isinstance(device, I4hTeleopDevice)
        assert device.cfg.case == cfg.case
        assert device.advance().shape[-1] == 10
    finally:
        device.close()
