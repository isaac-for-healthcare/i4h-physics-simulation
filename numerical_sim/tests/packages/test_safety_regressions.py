# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded safety regressions using generated assets and fake devices."""
from types import SimpleNamespace

import numpy as np
import pytest
from i4h_standalone.app.haptics import Conditioning, HapticsConfig, HapticServo, HostForceMailbox


def servo():
    cfg = HapticsConfig(max_force_n=2, devices={"a": {"max_force_n": 1, "handle": {"max_force_n": 3}},
                                              "b": {"max_force_n": 4, "handle": {"max_force_n": 6}}})
    sources = {k: SimpleNamespace(wants_jaw=True) for k in ("a", "b")}
    return HapticServo(cfg, sources, HostForceMailbox(1), ["tool"])


def test_force_ceiling_reduce_restore_and_atomic_validation():
    s = servo()
    assert s.force_ceiling() == 4 and s.force_ceiling(handle=True) == 6
    s.set_param("max_force_n", .5)
    s.set_param("max_force_n", 4)
    assert [d["conditioner"].params.max_force_n for d in s.devices.values()] == [1, 4]
    s.set_handle_param("max_force_n", 1)
    s.set_handle_param("max_force_n", 6)
    assert [d["handle"].params.max_force_n for d in s.devices.values()] == [3, 6]
    before = [(d["conditioner"].params, d["handle"].params) for d in s.devices.values()]
    for name, value in (("max_force_n", 5), ("low_pass_hz", float("nan")), ("axis_gains", (1, float("inf"), 1))):
        with pytest.raises(ValueError):
            s.set_param(name, value)
    with pytest.raises(ValueError):
        s.set_handle_param("max_force_n", 7)
    assert before == [(d["conditioner"].params, d["handle"].params) for d in s.devices.values()]
    with pytest.raises(TypeError):
        s.pose_ceilings["a"] = 10
    with pytest.raises(ValueError):
        Conditioning(axis_gains=(1, float("nan"), 1))


def test_device_cleanup_attempts_every_device_and_retains_initial_failure():
    from i4h_standalone.input.manager import InputManager

    calls = []
    class Device:
        def __init__(self, name, fail_init=False):
            self.config = SimpleNamespace(id=name, backend="fake")
            self.state = None
            self.fail_init = fail_init
        def init(self):
            if self.fail_init:
                raise ValueError("original initialization failure")
        def close(self):
            calls.append(self.config.id)
            raise RuntimeError("close failure")
    manager = InputManager([Device("a"), Device("b"), Device("c", True)])
    with pytest.raises(ValueError, match="original initialization") as error:
        manager.init()
    assert calls == ["b", "a"] and len(error.value.__notes__) == 2
    calls.clear()
    with pytest.raises(ExceptionGroup):
        manager.close()
    assert calls == ["c", "b", "a"]


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), -1, 0])
def test_nonfinite_timesteps_rejected(dt):
    from i4h_physics.config import check_dt, sim_settings
    with pytest.raises(ValueError):
        check_dt(dt, .01)
    with pytest.raises(ValueError):
        sim_settings({"sim": {"dt": dt}})


def test_pbd_binding_and_appearance_free_surface():
    from i4h_endoluminal.bindings import CatheterBinding
    from i4h_isaaclab.composition import load_case
    from i4h_physics.systems.mechanics.surface import Surface
    from i4h_standalone.render.snapshots import Snapshot

    cfg = {"sim": {"device": "cpu", "envs": 2, "cuda_graph": False},
           "assets": {"wire": {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0], "count": 8, "segment_length": .02, "radius": .002},
                      "solid": {"type": "tet_grid", "dim": [2, 2, 2], "cell": .02}},
           "systems": [{"type": "particles"}, {"type": "rod_pbd", "body": "wire", "id": "rod"},
                       {"type": "catheter_control", "rod": "rod", "track_length": 0},
                       {"type": "tet", "body": "solid", "id": "tet"}]}
    sim = load_case(cfg)
    surface = Surface(sim.scene, tet=sim.systems[-1])
    sim.systems.append(surface)
    sim.prepare()
    binding = CatheterBinding(sim, "wire")
    assert binding.material_backend == "pbd" and set(binding.materials) == {"stretch", "bend", "twist"}
    snapshot = Snapshot(sim)
    assert surface in snapshot.tris and surface not in snapshot.visual


def test_euler_conventions_remain_distinct():
    from i4h_physics.mesh_utils import euler_rx_ry_rz_degrees_matrix
    from i4h_physics.quat import euler_rz_ry_rx_degrees_quat, quat_to_matrix
    angle = [20, 30, 40]
    # Reversing multiplication order preserves each historical YAML interpretation.
    x = euler_rx_ry_rz_degrees_matrix(angle)
    y = quat_to_matrix(euler_rz_ry_rx_degrees_quat(angle))
    assert not np.allclose(x, y)
    np.testing.assert_allclose(x @ x.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(y @ y.T, np.eye(3), atol=1e-6)


def test_missing_device_retains_availability_failure_when_release_fails(monkeypatch):
    import i4h_standalone.input.follou as follou

    manager = SimpleNamespace(get_device_controller=lambda *a, **k: None, devices=[])
    monkeypatch.setattr(follou, "acquire_manager", lambda: (manager, object))
    calls = []
    def release():
        calls.append(True)
        raise RuntimeError("native shutdown failure")
    monkeypatch.setattr(follou, "release_manager", release)
    with pytest.raises(RuntimeError, match="MiniMou device index 1 not available") as error:
        follou.MiniMouController(device_index=1)
    assert calls == [True]
    assert any("native shutdown failure" in note for note in error.value.__notes__)


def test_optional_box_cleanup_failure_is_not_silenced():
    from i4h_standalone.input.manager import InputManager

    original = RuntimeError("box unavailable")
    original.add_note("Follou initialization cleanup failed: native shutdown failure")
    def init():
        raise original
    box = SimpleNamespace(config=SimpleNamespace(id="box", backend="ibcontroller"), state=None, init=init)
    manager = InputManager([box])
    with pytest.raises(RuntimeError, match="box unavailable") as error:
        manager.init()
    assert error.value is original and manager.devices == [box]
