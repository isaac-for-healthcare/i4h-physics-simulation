# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M15 XR (X2): the XR <-> world math, the runtime modes, the OpenXR session against a fake ``xr`` module (adapted from
omnisurg's tests/test_openxr.py), and the XR renderer end to end on the real GPU with fake swapchain images. No
headset, no CloudXR runtime."""

import ctypes
import math
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import needs_external_data
from i4h_isaaclab.camera.camera import Camera
from i4h_standalone.xr.anchor import PoseAnchor, projection_from_fov, quaternion_from_matrix
from xr_fakes import FakeCuda, FakeSpy, FakeXr, SessionState, install_fake_xr  # noqa: E402

# --- math ------------------------------------------------------------------------------------------------------------


def test_centred_eye_equals_the_desktop_camera():
    """A head at the LOCAL origin looking down -z, anchored at the case camera: a symmetric eye fov of the camera's fov
    gives the camera's own view-projection (the same conventions end to end)."""
    cam = Camera(position=(0.2, 1.1, -3.0), pitch=-15.0, yaw=30.0, fov=60.0)
    cam.near, cam.far = 0.05, 40.0
    anchor = PoseAnchor.from_camera(cam, world_units_per_meter=1.0)
    assert anchor.observe_head((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), valid=True) and anchor.anchored
    half = math.radians(30.0)
    basis, vp = anchor.eye_camera((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (-half, half, half, -half), cam.near, cam.far)
    np.testing.assert_allclose(vp, cam.view_projection(100.0, 100.0), atol=1e-5)
    for got, want in zip(basis, cam.basis()):
        np.testing.assert_allclose(got, want, atol=1e-6)


def test_anchor_scales_and_follows_the_head():
    cam = Camera(position=(0.0, 1.0, 0.0), pitch=0.0, yaw=0.0, fov=60.0)
    anchor = PoseAnchor.from_camera(cam, world_units_per_meter=10.0)
    anchor.observe_head((0.5, 1.6, 0.0), (0.0, 0.0, 0.0, 1.0), valid=True)
    pos, _ = anchor.transform((0.5, 1.6, 0.0), (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(pos, cam.position, atol=1e-5)  # the anchored head sits on the camera
    moved, _ = anchor.transform((0.6, 1.6, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert np.linalg.norm(np.asarray(moved) - np.asarray(pos)) == pytest.approx(1.0, abs=1e-5)  # 10 cm -> 1 unit
    assert not anchor.observe_head((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), valid=True)  # anchored once
    anchor.reset_for_reconnect()
    assert not anchor.anchored
    with pytest.raises(ValueError):
        projection_from_fov(0.1, -0.1, 0.1, -0.1, 0.05, 10.0)
    np.testing.assert_allclose(quaternion_from_matrix(np.eye(3)), (0.0, 0.0, 0.0, 1.0))


# --- runtime modes ----------------------------------------------------------------------------------------------------


def test_runtime_modes(tmp_path, monkeypatch):
    from i4h_standalone.xr import runtime as rt

    monkeypatch.setenv("XR_RUNTIME_JSON", "/before.json")
    system = rt.XrRuntime("system")
    system.start()
    assert "XR_RUNTIME_JSON" not in __import__("os").environ
    system.close()
    assert __import__("os").environ["XR_RUNTIME_JSON"] == "/before.json"

    (tmp_path / "openxr_cloudxr.json").write_text("{}")
    external = rt.XrRuntime.from_cfg({"mode": "external", "directory": str(tmp_path)})
    external.start()
    assert __import__("os").environ["XR_RUNTIME_JSON"] == str(tmp_path / "openxr_cloudxr.json")
    external.close()
    with pytest.raises(FileNotFoundError):
        rt.XrRuntime("external", directory=str(tmp_path / "missing")).start()
    with pytest.raises(ValueError, match="unknown keys"):
        rt.XrRuntime.from_cfg({"mode": "managed", "profile": "x"})
    with pytest.raises(ValueError, match="mode"):
        rt.XrRuntime("cloud")


def test_managed_runtime_starts_and_stops_cloudxr(monkeypatch):
    from i4h_standalone.xr import runtime as rt

    monkeypatch.setenv("XR_RUNTIME_JSON", "/before.json")
    launched = []

    class FakeLauncher:
        def __init__(self, **kwargs):
            launched.append(kwargs)
            __import__("os").environ["XR_RUNTIME_JSON"] = "/cloudxr.json"

        def stop(self):
            launched.append("stop")

    launcher_module = pytest.importorskip("isaacteleop.cloudxr.launcher")  # the xr extra

    monkeypatch.setattr(launcher_module, "CloudXRLauncher", FakeLauncher)
    managed = rt.XrRuntime("managed")
    managed.start()
    assert launched[0]["device_profile"] == "auto-webrtc" and __import__("os").environ["XR_RUNTIME_JSON"] == "/cloudxr.json"
    managed.close()
    assert launched[-1] == "stop" and __import__("os").environ["XR_RUNTIME_JSON"] == "/before.json"
    monkeypatch.delenv("TELEOP_PROXY_HOST", raising=False)
    hosted = rt.XrRuntime.from_cfg({"mode": "managed", "host_client": True, "host": "192.0.2.10"})
    hosted.start()  # the web client served from here; the headset's address for this machine handed to isaacteleop
    assert launched[-1]["host_client"] is True and __import__("os").environ["TELEOP_PROXY_HOST"] == "192.0.2.10"
    hosted.close()
    assert "TELEOP_PROXY_HOST" not in __import__("os").environ


def test_lan_address_skips_vpn_and_containers(monkeypatch):
    import subprocess

    from i4h_standalone.xr import runtime as rt

    out = ("2: eno2    inet 192.168.0.10/24 brd 192.168.0.255 scope global eno2\n"
           "3: gpd0    inet 10.0.0.10/32 scope global gpd0\n"
           "4: docker0    inet 172.17.0.1/16 scope global docker0\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out))
    assert rt.lan_ipv4() == "192.168.0.10"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out.splitlines()[1] + "\n"))
    assert rt.lan_ipv4() is None  # only a VPN link: no guess


# --- the session against a fake xr ------------------------------------------------------------------------------------


def make_session(xr, **kwargs):
    from i4h_standalone.xr.session import XrSession

    events = xr.events
    calls = SimpleNamespace(device=[], release=[])
    session = XrSession(xr, FakeSpy(events), FakeCuda(), SimpleNamespace(is_cuda=True, context=55, ordinal=0),
                        on_device=lambda s: calls.device.append(s.device), on_release=lambda s: calls.release.append(s.device),
                        requirements_getter=lambda *a: None, instance_extensions_getter=lambda *a: ("VK_KHR_a",),
                        device_extensions_getter=lambda *a: ("VK_KHR_b",), graphics_device_getter=lambda *a: 301, **kwargs)
    session.start()
    return session, calls


def test_session_waits_connects_renders_and_reconnects():
    xr = FakeXr()
    session, calls = make_session(xr)
    actions = {e[1] for e in xr.events if e[0] == "action"}
    assert {"grip_pose", "trigger", "squeeze", "thumbstick", "thumbstick_click", "haptic", "a", "b", "x", "y"} <= actions
    assert any(e[0] == "bindings" and e[1].endswith("touch_plus_controller") for e in xr.events if isinstance(e, tuple))
    assert session.begin_frame() is None and session.session is None  # no headset yet
    xr.client = True
    assert session.begin_frame() is None and calls.device and session.session == "session"  # connected, not running
    assert ("device", [0, 1], ("VK_KHR_b",)) in xr.events  # the CUDA GPU's UUID, OpenXR's device extensions
    assert [e for e in xr.events if e[0] == "wrap"] == [("wrap", 10001, 64, 48), ("wrap", 10002, 64, 48)]
    xr.pending_events.append(SessionState.READY)
    frame = session.begin_frame()
    assert session.running and frame.should_render and len(frame.views) == 2 and frame.head is not None
    right = frame.samples["right"]
    assert right["active"] and right["trigger"] == 0.75 and right["squeeze"] == 0.5 and right["buttons"]["a"] is True
    assert frame.samples["left"]["buttons"] == {"thumbstick": False, "x": False, "y": False}
    assert right["thumbstick"] == pytest.approx((0.1, -0.2))
    encoded = []
    assert session.end_frame(lambda encoder, textures: encoded.append(len(textures)) or True, cuda_stream="stream")
    assert encoded == [2] and ("submit", True) in xr.events and xr.events[-1] == ("end_frame", 1)
    assert [e for e in xr.events if e[0] == "release"] == [("release", "swapchain-1"), ("release", "swapchain-2")]
    session.begin_frame()
    session.end_frame(lambda encoder, textures: False)  # nothing encoded: the frame still ends, with no layer
    assert xr.events[-1] == ("end_frame", 0)
    session.begin_frame()
    session.end_frame(None)
    assert xr.events[-1] == ("end_frame", 0)

    xr.client = False  # the headset disconnects
    xr.pending_events.append(SessionState.STOPPING)
    assert session.begin_frame() is None
    assert calls.release and session.session is None and session.reconnects == 1
    xr.client = True
    assert xr.events.index("device_close") > max(i for i, e in enumerate(xr.events) if isinstance(e, tuple) and e[0] == "texture_close")
    session.begin_frame()  # the client is still there: a new session and device
    assert len(calls.device) == 2
    session.close()
    assert "destroy_instance" in xr.events and not session.alive


def test_session_haptics_apply_on_the_frame_and_stop():
    xr = FakeXr()
    xr.client = True
    session, _ = make_session(xr)
    session.begin_frame()
    xr.pending_events.append(SessionState.READY)
    session.begin_frame()
    session.set_haptic("right", 0.6)
    session.set_haptic("left", 0.0)  # never started: nothing to stop
    session.end_frame(None)
    assert ("haptic", "right", 0.6) in xr.events and ("haptic_stop", "left") not in xr.events
    session.begin_frame()
    session.set_haptic("right", 0.0)
    session.end_frame(None)
    assert ("haptic_stop", "right") in xr.events
    session.begin_frame()
    session.set_haptic("right", 2.0)  # clamped to 1
    session.end_frame(None)
    assert ("haptic", "right", 1.0) in xr.events
    session.close()  # stops the vibration on the way out
    assert xr.events.count(("haptic_stop", "right")) == 2


# --- the XR renderer on the GPU with fake swapchain images ------------------------------------------------------------


@pytest.mark.skipif(not __import__("helpers").RENDER_TESTS, reason="set I4H_RENDER_TESTS=1")
def test_xr_renderer_renders_both_eyes(monkeypatch, tmp_path):
    """The real XrRenderer on the real GPU: the fake runtime hands out textures made on the session's own Slang device
    as swapchain images; both eyes get the two_worlds scene, seen from 3.2 cm apart (they differ)."""
    spy = pytest.importorskip("slangpy")
    if not hasattr(spy.Device, "create_texture_from_native_handle"):
        pytest.skip("needs the xr extra (the slangpy fork)")
    from i4h_isaaclab.composition import load_case, read_case

    xr, owned, holder = install_fake_xr(monkeypatch, tmp_path, spy)
    from i4h_standalone.render.xr import XrRenderer

    sim = load_case("two_worlds")
    render = {k: v for k, v in (read_case("two_worlds").get("render") or {}).items() if k not in ("backend", "size")}
    renderer = XrRenderer(sim, xr={"world_units_per_meter": 1.0, "runtime": {"mode": "external"}, "mirror": "off"}, **render)
    holder.session = renderer.session
    try:
        renderer.publish()
        renderer.draw()  # connects: device, swapchains, the scene renderer
        xr.pending_events.append(SessionState.READY)
        for _ in range(3):
            renderer.publish()
            assert renderer.draw()
        from i4h_standalone.xr import hands

        hands.set_haptic("right", 0.6)  # the servo's command reaches the controller with the next frame
        renderer.publish()
        renderer.draw()
        assert ("haptic", "right", 0.6) in xr.events
        right = hands.read("right")  # published in world coordinates through the head anchor
        want, _ = renderer.anchor.transform((0.2, -0.3, -0.4), (0.0, 0.0, 0.0, 1.0))
        assert right["active"] and np.allclose(right["position"], want) and right["trigger"] == 0.75
        renderer.session.device.wait()
        left, right = (np.asarray(t.to_numpy(), np.float32) for t in owned[:2])
    finally:
        renderer.close()
    dump = __import__("os").environ.get("I4H_XR_DUMP")
    if dump:  # a look at the eyes: I4H_XR_DUMP=/dir writes left.png, right.png
        from PIL import Image

        for name, eye in (("left", left), ("right", right)):
            Image.fromarray(eye.astype(np.uint8)).save(f"{dump}/{name}.png")
    from i4h_standalone.xr import hands

    assert renderer.frames == 4 and xr.events.count(("end_frame", 1)) == 4
    assert not hands.read("right")["active"], "a closed renderer leaves no controller behind"
    for eye in (left, right):
        assert eye.shape == (120, 160, 4) and np.isfinite(eye).all() and eye[..., :3].std() > 1.0  # an image, not a clear
    assert np.abs(left - right).mean() > 0.1  # two viewpoints



def test_session_state_events_decode_from_the_runtime_buffer():
    """pyopenxr's poll_event returns a fresh generic EventDataBuffer, referenced by nothing else: the state must be read
    while the buffer is alive (a struct view through cast(byref(buffer)) kept no reference, the buffer was freed and
    the state read as 0 = UNKNOWN on the real runtime; the ready-made events of the fake above hid it)."""
    import gc

    real = pytest.importorskip("xr")
    from i4h_standalone.xr.session import XrSession

    pending = [2]

    def poll_event(instance):
        if not pending:
            raise real.EventUnavailable()
        buffer = real.EventDataBuffer()  # as pyopenxr: a new buffer per event, returned and forgotten
        raw = (ctypes.c_ubyte * ctypes.sizeof(buffer)).from_buffer(buffer)
        ctypes.c_int.from_buffer(raw, 0).value = int(real.StructureType.EVENT_DATA_SESSION_STATE_CHANGED)
        ctypes.c_int.from_buffer(raw, real.EventDataSessionStateChanged.state.offset).value = pending.pop()
        del raw
        return buffer

    seen = []

    def record(state):
        gc.collect()  # anything the decoded event does not keep alive is gone by now
        seen.append(state)

    session = SimpleNamespace(instance="instance", _session_state=record, _lost_instance=lambda: None)
    session.xr = SimpleNamespace(poll_event=poll_event, EventUnavailable=real.EventUnavailable, StructureType=real.StructureType,
                                 EventDataSessionStateChanged=real.EventDataSessionStateChanged, SessionState=real.SessionState)
    XrSession._poll_events(session)
    assert seen == [real.SessionState.READY]



# --- X3: the controllers as input devices -----------------------------------------------------------------------------


def test_openxr_input_settings():
    from i4h_standalone.input import DeviceConfig

    right = DeviceConfig("primary", "openxr", settings={"hand": "right"}).settings
    assert (right["grip"], right["energy"], right["fire"], right["pickup"]) == ("trigger", "a", "b", "thumbstick")
    left = DeviceConfig("secondary", "openxr", settings={"hand": "left", "grip": "squeeze"}).settings
    assert (left["grip"], left["energy"], left["fire"]) == ("squeeze", "x", "y")
    for bad in ({}, {"hand": "middle"}, {"hand": "left", "grip": "palm"}, {"hand": "left", "fire": "z"}, {"hand": "left", "haptic": 1}):
        with pytest.raises(ValueError):
            DeviceConfig("d", "openxr", settings=bad)


@needs_external_data
def test_controllers_drive_and_pick_up_tools():
    """A published controller drives its device's row through DeviceRows; its thumbstick click puts the held tool down
    and takes the nearest (the pickup's button), its trigger is the jaw, A the energy."""
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.app.device_rows import DeviceRows
    from i4h_standalone.xr import hands

    sim = load_case("chole_tet_med")
    records = list(sim.scene.instruments.values())
    cfg = {"pickup": {"distance": 0.5, "proxy_radius": 0.05, "keys": {"primary": "1", "secondary": "2"}},
           "primary": {"backend": "openxr", "settings": {"hand": "right"}},
           "secondary": {"backend": "openxr", "settings": {"hand": "left"}}}
    rows = np.zeros((len(records), 10), np.float32)
    for k, r in enumerate(records):
        rows[k, :3], rows[k, 3:7] = r.initial_position, r.initial_rotation
    devices = DeviceRows(records, cfg)
    try:
        held = devices.pickup.holding["primary"]
        assert held is not None
        start = rows[held, :3].copy()

        def right(position, buttons=(), trigger=0.0):
            hands.publish("right", {"active": True, "position": tuple(position), "rotation": (0.0, 0.0, 0.0, 1.0),
                                    "trigger": trigger, "squeeze": 0.0, "buttons": {b: True for b in buttons}})

        right(start)
        devices.update(rows)
        right(start + np.float32([0.05, 0.0, 0.0]), trigger=1.0, buttons=("a",))
        devices.update(rows)
        np.testing.assert_allclose(rows[held, :3], start + np.float32([0.05, 0.0, 0.0]), atol=1e-5)  # the tool follows
        assert rows[held, 8] == 1.0 or records[held].kind == "hook"  # energy from A
        right(rows[held, :3], buttons=("thumbstick",))  # click: put it down
        devices.update(rows)
        assert devices.pickup.holding["primary"] is None
        right(rows[held, :3] + np.float32([0.0, 2.0, 0.0]))
        devices.update(rows)  # away: the re-grab guard on the tool just put down lifts
        right(rows[held, :3])
        devices.update(rows)
        right(rows[held, :3], buttons=("thumbstick",))  # click again: take the nearest (it)
        devices.update(rows)
        assert devices.pickup.holding["primary"] == held
        hands.clear()
        devices.update(rows)
        assert not devices.states["primary"].active
    finally:
        devices.close()
        hands.clear()


@needs_external_data
def test_chole_xr_hands_start_with_hook_right_and_grasper_left():
    """render.xr.inputs.pickup.start: the right controller (primary) starts on the hook, the left (secondary) on the
    grasper, whatever the desk devices' input: bindings say."""
    from i4h_isaaclab.composition import load_case, read_case
    from i4h_standalone.app.device_rows import DeviceRows

    cfg = read_case("chole_tet_med")
    sim = load_case(cfg)
    records = list(sim.scene.instruments.values())
    names = [r.name for r in records]
    devices = DeviceRows(records, cfg["render"]["xr"]["inputs"])
    try:
        assert devices.pickup.holding == {"primary": names.index("hook"), "secondary": names.index("grasper")}
        assert devices.manager.devices[0].config.settings["hand"] in ("left", "right")
        hands = {d.config.id: d.config.settings["hand"] for d in devices.manager.devices}
        assert hands == {"primary": "right", "secondary": "left"}
    finally:
        devices.close()
    bad = dict(cfg["render"]["xr"]["inputs"])
    bad["pickup"] = {**bad["pickup"], "start": {"primary": "hook", "secondary": "hook"}}
    with pytest.raises(ValueError, match="distinct"):
        DeviceRows(records, bad)



# --- X4: controller vibration ---------------------------------------------------------------------------------------


def xr_source(**settings):
    from i4h_standalone.input import DeviceConfig
    from i4h_standalone.input.sources import XrHandSource

    return XrHandSource(DeviceConfig("primary", "openxr", settings={"hand": "right", **settings}).settings)


def test_vibration_amplitude_from_force_and_jaw(monkeypatch):
    from i4h_standalone.xr import hands

    hands.clear()
    src = xr_source()  # full_scale_n 0.5, jaw_full_scale_n 2.0, force+jaw
    src.io((0.0, 0.0, 0.25))
    assert hands.haptic("right") == pytest.approx(0.5)
    src.io((0.0, 0.0, 0.05), 1.5)  # the squeeze dominates
    assert hands.haptic("right") == pytest.approx(0.75)
    src.io((3.0, 0.0, 0.0))
    assert hands.haptic("right") == 1.0  # clamped
    src.io((float("nan"), 0.0, 0.0))
    assert hands.haptic("right") == 0.0
    force_only = xr_source(vibrate_on="force", full_scale_n=1.0)
    assert not force_only.wants_jaw
    force_only.io((0.0, 0.2, 0.0), 2.0)
    assert hands.haptic("right") == pytest.approx(0.2)
    real = hands.time.perf_counter
    monkeypatch.setattr(hands.time, "perf_counter", lambda: real() + 0.2)
    assert hands.haptic("right") == 0.0  # stale: the servo stopped refreshing
    monkeypatch.undo()
    hands.clear()
    assert hands.haptic("right") == 0.0 and not hands.read("right")["active"]
    with pytest.raises(ValueError):
        xr_source(vibrate_on="jaw")


def test_servo_vibrates_the_controller_and_stops():
    """The haptic servo owns an openxr source like a MiniMou: the conditioned (capped) force of the tool that hand
    holds becomes its vibration; stopping the servo leaves the controller still."""
    import time

    from i4h_standalone.app.haptics import HapticsConfig, HapticServo, HostForceMailbox
    from i4h_standalone.xr import hands

    hands.clear()
    hands.publish("right", {"active": True, "position": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0), "trigger": 0.0,
                            "squeeze": 0.0, "buttons": {}})
    src = xr_source()
    src.mailbox = {"active": False}
    cfg = HapticsConfig(enabled=True, newtons_per_unit=1.0, max_force_n=0.5, low_pass=False, slew=False, ramp=False, stale=False)
    mailbox = HostForceMailbox(1)
    servo = HapticServo(cfg, {"primary": src}, mailbox, ["hook"])
    assert servo.devices["primary"]["handle"] is not None  # the jaw term: a handle conditioner without a box handle
    servo.start()
    try:
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline and hands.haptic("right") < 0.99:
            mailbox.push(np.float32([[0.0, 0.0, 3.0]]), {"primary": 0})  # 3 N asked, the 0.5 N cap holds
            time.sleep(0.005)
        assert hands.haptic("right") == pytest.approx(1.0)  # the capped 0.5 N is full scale
    finally:
        servo.stop()
    assert hands.haptic("right") == 0.0  # the servo's final zero round trips
    hands.clear()


# Render-thread lifecycle regressions.


def test_a_stalled_controller_sample_goes_inactive(monkeypatch):
    """No publish for 0.1 s (a stalled render thread): the hand reads inactive, so the jaw, energy and fire release."""
    from i4h_standalone.xr import hands

    hands.clear()
    hands.publish("right", {"active": True, "position": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0), "trigger": 1.0,
                            "squeeze": 0.0, "buttons": {"a": True}})
    src = xr_source()
    sample = src.poll()
    assert sample["active"] and sample["grip"] == 1.0 and sample["buttons"]["secondary"]
    real = hands.time.perf_counter
    monkeypatch.setattr(hands.time, "perf_counter", lambda: real() + 0.15)
    assert not hands.read("right")["active"] and src.poll() == {"active": False}
    monkeypatch.undo()
    hands.clear()


class _StubRenderer:
    def __init__(self, draw):
        self.draw, self.closed = draw, False
        self.stream = self.lock = None

    def is_running(self):
        return True

    def publish(self, snap=None):
        pass

    def close(self):
        self.closed = True


def _view(renderer):
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.teleop import XrView

    return XrView(SimpleNamespace(unwrapped=SimpleNamespace(simulator=load_case("two_worlds"))), renderer)


def test_xr_view_refuses_to_return_with_a_live_render_thread():
    """A render thread blocked in OpenXR: close() raises instead of returning (the caller must not free the env)."""
    import threading

    release = threading.Event()
    entered = threading.Event()
    def draw():
        entered.set()
        release.wait()
    view = _view(_StubRenderer(draw=draw))
    view.join_timeout = 0.2
    view.start()
    assert entered.wait(5.0), "render thread must enter its blocking draw before close"
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            view.close()
    finally:
        release.set()
    view._thread.join(5.0)
    assert not view._thread.is_alive() and view.renderer.closed


def test_xr_view_reports_a_render_thread_failure():
    """A failure on the render thread ends running and is raised by check() (the script checks after its loop)."""
    import time

    def draw():
        raise ValueError("render boom")

    view = _view(_StubRenderer(draw=draw))
    view.start()
    deadline = time.perf_counter() + 5.0
    while view.running and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert not view.running
    with pytest.raises(RuntimeError, match="render thread failed"):
        view.check()
    view.close()
    assert view.renderer.closed  # closed on its own thread even after the failure


@pytest.mark.skipif(not __import__("helpers").RENDER_TESTS, reason="set I4H_RENDER_TESTS=1")
def test_renderer_reconnect_during_worker_recapture(monkeypatch, tmp_path):
    """XR GPU ownership changes share the worker's publication/capture lock."""
    import threading

    import warp as wp
    spy = pytest.importorskip("slangpy")
    if not hasattr(spy.Device, "create_texture_from_native_handle"):
        pytest.skip("requires the pinned XR SlangPy fork")
    import i4h_standalone.render.slang as slang
    from i4h_isaaclab.composition import load_case, read_case
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.snapshots import Snapshots
    from i4h_standalone.render.xr import XrRenderer

    xr, owned, holder = install_fake_xr(monkeypatch, tmp_path, spy)
    sim = load_case("two_worlds")
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    cfg = {k: v for k, v in read_case("two_worlds")["render"].items() if k not in ("backend", "size")}
    renderer = XrRenderer(sim, xr={"world_units_per_meter": 1, "runtime": {"mode": "external"}, "mirror": "off"}, **cfg)
    renderer.stream, renderer.lock = wp.Stream(sim.scene.device), ring.lock
    holder.session = renderer.session
    real_renderer = slang.SlangRenderer
    creations = []
    def create(*args, **kwargs):
        assert ring.lock._is_owned()
        creations.append(True)
        return real_renderer(*args, **kwargs)
    monkeypatch.setattr(slang, "SlangRenderer", create)
    worker = PhysicsWorker(sim, [], ring, realtime=True)
    recaptured = threading.Event()
    def recapture(sim):
        assert ring.lock._is_owned()
        sim.substeps += 1
        sim.prepare()
        recaptured.set()
    worker.start()
    try:
        renderer.draw()
        xr.pending_events.append(SessionState.READY)
        renderer.publish(ring.acquire())
        renderer.draw()
        first = renderer.scene_renderer
        worker.edit(recapture)
        xr.client = False
        xr.pending_events.append(SessionState.STOPPING)
        renderer.draw()
        assert renderer.scene_renderer is None and not renderer._views
        xr.client = True
        renderer.draw()
        xr.pending_events.append(SessionState.READY)
        renderer.publish(ring.acquire())
        renderer.draw()
        assert recaptured.wait(5), "worker recapture did not complete"
        worker.check()
        assert renderer.scene_renderer is not first and len(creations) == 2
        assert worker.steps > 0 and sim.graph is not None
        first = None  # release references before the session's device is closed
    finally:
        worker.stop()
        renderer.close()
    worker.check()


def test_unequal_eye_dimensions_fail_before_gpu_resource_creation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import slangpy as spy
    from i4h_standalone.xr.session import XrSession

    xr, owned, holder = install_fake_xr(monkeypatch, tmp_path, spy)
    original = xr.enumerate_view_configuration_views
    def views(*args):
        left, right = original(*args)
        right.recommended_image_rect_width += 8
        return left, right
    monkeypatch.setattr(xr, "enumerate_view_configuration_views", views)
    session = XrSession(xr, spy, None, SimpleNamespace(is_cuda=True))
    session.start()
    try:
        with pytest.raises(RuntimeError, match="equal left and right"):
            session.begin_frame()
        assert session.device is None
    finally:
        session.close()
