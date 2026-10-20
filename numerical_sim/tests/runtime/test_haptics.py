# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M11 haptics: the ``tool_force`` product on two_worlds and chole."""

import contextlib
import os

import numpy as np
import pytest
import warp as wp
from helpers import haptic_case
from i4h_isaaclab.composition import data_dir, load_case
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.tool_force import ToolForce

CHOLE = data_dir() / "choly-mid"
needs_chole = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not CHOLE.is_dir(), reason=f"chole data not found at {CHOLE}")


def find(sim, cls):
    return next(s for s in sim.systems if isinstance(s, cls))


def press(sim, z, envs=(0,), steps=60):
    """Hold every probe at rest and the listed envs' probes at height ``z`` (pushing the block's bottom face up)."""
    control = find(sim, InstrumentControl)
    actions = np.zeros((sim.scene.envs, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    for env in envs:
        actions[env, :, 2] = z
    control.actions.assign(actions)
    forces = []
    for _ in range(steps):
        sim.step()
        forces.append(find(sim, ToolForce).force.numpy().copy())
    return np.array(forces)  # (steps, E, n, 3)


def test_probe_force_direction_penetration_gain_cap_and_release():
    sim = load_case("two_worlds")
    tf, contact = find(sim, ToolForce), find(sim, InstrumentContact)
    E = sim.scene.envs

    settled = press(sim, 0.0, envs=(), steps=60)  # probes parked below the block: no contact anywhere
    assert np.all(settled == 0.0) and tf.count.numpy().max() == 0

    shallow = press(sim, 0.045, steps=90)
    f = shallow[-30:, 0, 0].mean(axis=0)
    assert np.isfinite(shallow).all() and tf.count.numpy()[0, 0] > 0
    assert f[2] < 0.0 and abs(f[2]) > 5.0 * np.hypot(f[0], f[1]), f  # the block pushes the probe back down
    assert np.all(shallow[:, 1] == 0.0) and tf.count.numpy()[1, 0] == 0  # env 1's probe is parked: isolation
    assert np.all(tf.jaw_load.numpy() == 0.0)  # a probe has no jaws: the signed channel stays empty while the force is not
    sim.reset()
    deeper = press(sim, 0.05, steps=90)[-30:, 0, 0].mean(axis=0)
    assert deeper[2] < f[2], (deeper, f)  # deeper penetration, larger reaction

    tf.contact_gain.fill_(0.0)  # live: no recapture
    assert np.all(press(sim, 0.05, steps=5)[-1] == 0.0) and tf.count.numpy()[0, 0] > 0
    tf.contact_gain.fill_(1.0)
    cap = 0.25 * abs(deeper[2])
    tf.max_force.fill_(cap)
    capped = press(sim, 0.05, steps=10)[-1, 0, 0]
    assert np.isclose(np.linalg.norm(capped), cap, rtol=1e-4), (capped, cap)
    tf.max_force.fill_(0.0)

    released = press(sim, 0.0, envs=(), steps=3)[-1]
    assert np.all(released == 0.0) and tf.count.numpy().max() == 0  # the product is per step, not a latch
    assert contact.impulse.numpy().shape == (E, 1, 3) and contact.delta_k.numpy().shape == (E * contact.body.count, 1, 3)


def test_channel_matches_the_applied_displacement():
    """One tool: the reduced channel must equal the mass-weighted displacement ``apply_average`` applies, so a
    contact event's impulse increment is ``sum m_i * delta_i / count_i`` over the particles it moved."""

    sim = load_case("two_worlds")
    sim.cuda_graph = False
    scene, contact = sim.scene, find(sim, InstrumentContact)
    press(sim, 0.05, steps=20)
    # Re-run one contact event by hand: the contact kernel scatters, the reduce sums, the apply moves.
    control = find(sim, InstrumentControl)
    control.step_start(scene, sim.dt)
    contact.substep_start(scene, sim.dt / sim.substeps, 0, sim.substeps)
    contact.impulse.zero_()
    q_before = scene.q.numpy().copy()
    contact.iteration(scene, sim.dt / sim.substeps, sim.iterations - 1, sim.iterations)  # the last turn always carries an event
    q_after = scene.q.numpy()
    dx = q_after - q_before
    m = np.where(scene.inv_mass.numpy() > 0.0, 1.0 / np.maximum(scene.inv_mass.numpy(), 1e-30), 0.0)
    expected = (dx * m[:, None]).reshape(scene.envs, -1, 3).sum(axis=1)  # only the block moved (one body touched)
    got = contact.impulse.numpy()[:, 0]
    assert np.linalg.norm(expected[0]) > 0.0
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-7 * np.abs(expected).max()), (got, expected)
    assert np.all(contact.delta_k.numpy() == 0.0)  # the reduce clears the channel


def test_masked_reset_captured_and_batching():
    sim = load_case("two_worlds")
    tf = find(sim, ToolForce)
    press(sim, 0.05, envs=(0, 1), steps=30)
    f0 = tf.force.numpy()[0].copy()
    assert (tf.force.numpy() != 0.0).any(axis=(1, 2)).all()
    sim.reset([1])
    assert np.array_equal(tf.force.numpy()[0], f0) and np.all(tf.force.numpy()[1] == 0.0) and tf.count.numpy()[1].max() == 0
    assert np.all(tf.jaw_load.numpy()[1] == 0.0)

    captured, eager = load_case("two_worlds"), load_case("two_worlds")
    eager.cuda_graph = False
    a = press(captured, 0.05, envs=(0, 1), steps=40)
    b = press(eager, 0.05, envs=(0, 1), steps=40)
    scale = np.abs(a).max()
    assert np.allclose(a[-1], b[-1], rtol=1e-3, atol=2e-3 * scale), (a[-1], b[-1])  # float atomics: tolerance, not bitwise

    launches = []
    orig = wp.launch

    def spy(kernel, dim, *args, **kwargs):
        launches.append((kernel.key, dim))
        return orig(kernel, dim, *args, **kwargs)

    wp.launch = spy
    try:
        eager.step()
    finally:
        wp.launch = orig
    ours = [d for k, d in launches if k in ("tool_force", "reduce_tool_delta")]
    assert ours and all((d[0] if isinstance(d, (tuple, list)) else d) == eager.scene.envs for d in ours)


def test_state_views_expose_force_and_count():
    torch = pytest.importorskip("torch")
    from i4h_isaaclab.views import state_views

    sim = load_case("two_worlds")
    tf = find(sim, ToolForce)
    views = state_views(sim)
    force = next(v for v in views if v.data_ptr() == tf.force.ptr)
    count = next(v for v in views if v.data_ptr() == tf.count.ptr)
    jaw = next(v for v in views if v.data_ptr() == tf.jaw_load.ptr)
    assert tuple(force.shape) == (2, 3) and tuple(count.shape) == (2, 1) and tuple(jaw.shape) == (2, 1) and force.dtype == torch.float32


@needs_chole
def test_grasp_and_pull_tugs_the_grasper_back():
    """The scripted grasper holds gallbladder tissue; pulling it away yields a force on the grasper toward the tissue."""
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_surgical.systems.instruments.grasp import Grasp

    sim = load_case("chole_tet_med", envs=2)
    control, grasp, tf = find(sim, InstrumentControl), find(sim, Grasp), find(sim, ToolForce)
    driver = InstrumentDriver(sim, control, None, "scripted")
    sim.prepare()
    for step in range(132):
        driver.update(step * sim.dt)
        sim.step()
    assert grasp.count.numpy()[0, 0] > 0
    held = tf.force.numpy()[0, 0].copy()
    jaw = tf.jaw_load.numpy()
    assert np.isfinite(jaw).all() and jaw[0, 0] > 0.0 and jaw[0, 1] == 0.0, jaw  # the grasper squeezes tissue; the hook has no jaws
    tf.grasp_gain.fill_(0.0)
    tf.contact_gain.fill_(0.0)
    sim.step()
    assert np.all(tf.jaw_load.numpy() == 0.0) and np.all(tf.force.numpy()[:, 0] == 0.0)  # live gains, no recapture
    tf.grasp_gain.fill_(1.0)
    tf.contact_gain.fill_(1.0)
    sim.step()
    assert tf.jaw_load.numpy()[0, 0] > 0.0
    actions = control.actions.numpy()
    pull = np.array([0.0, 0.0, 0.25], np.float32)  # back toward the camera, away from the liver bed
    forces = []
    for step in range(12):
        actions[0, 0, :3] += pull / 12.0
        control.actions.assign(actions)
        sim.step()
        forces.append(tf.force.numpy()[0].copy())
    forces = np.array(forces)
    assert np.isfinite(forces).all()
    grasper = forces[-6:, 0].mean(axis=0)
    assert grasp.count.numpy()[0, 0] > 0
    # Held in place the closed jaw presses into the liver bed (force along +z, away from the tissue); pulled back it is
    # tugged toward the tissue: the component along the pull flips sign and dominates.
    assert np.dot(held, pull) > 0.0 and np.dot(grasper, pull) < -0.5 * np.linalg.norm(grasper) * np.linalg.norm(pull), (grasper, held)
    assert np.isfinite(tf.force.numpy()[1]).all()  # env 1 keeps its own scripted values


# --- the servo (app/haptics.py) ------------------------------------------------------------------------------------


def make_haptics(path, capture_lock=None):
    from i4h_isaaclab.composition import read_case
    from i4h_standalone.app.haptics import HapticsConfig, HapticsDriver
    from i4h_standalone.app.instruments import InstrumentDriver

    cfg = read_case(str(path))
    sim = load_case(cfg)
    control, tf = find(sim, InstrumentControl), find(sim, ToolForce)
    driver = InstrumentDriver(sim, control, cfg["inputs"], "devices")
    haptics = HapticsDriver(sim, tf, driver, HapticsConfig(**cfg["haptics"]), capture_lock=capture_lock)
    return sim, driver, haptics


def open_stub_box(monkeypatch, stub):
    """``InstrumentBox.open`` returns an adapter over ``stub`` (a ``StubBox``): no hardware."""
    from i4h_standalone.input import avatar as av

    monkeypatch.setattr(av.InstrumentBox, "open", classmethod(lambda cls, index, channels=6: cls(stub, channels)))
    return stub


def open_stub_avatar(monkeypatch, stub):
    """``Avatar.open`` returns an adapter over ``stub`` (a ``StubHaptic``): no hardware."""
    from i4h_standalone.input import avatar as av

    monkeypatch.setattr(av.Avatar, "open", classmethod(lambda cls, index, **kw: cls(stub, **kw)))
    return stub


@contextlib.contextmanager
def haptics_panels(path):
    """The app's panels over a haptic case (the fake UI of test_ui): yields ``(haptics, panels, screen)``; closes the
    panels, the servo and the devices."""
    from types import SimpleNamespace

    from i4h_standalone.app.ui import Panels, SimControls
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.snapshots import Snapshots
    from test_ui import FAKE_UI, _Widget

    sim, driver, haptics = make_haptics(path)
    panels = None
    try:
        sim.prepare()
        ring = Snapshots(sim)
        ring.publish()
        renderer = SimpleNamespace(env=0, key_handlers=[], debug_spheres=False, exposure=1.0, tonemap=0, width=1280, height=720)
        worker = PhysicsWorker(sim, [driver, haptics], ring)
        screen = _Widget(None)
        panels = Panels(worker, renderer, SimControls(worker, (0.0, 0.0, -9.81)), None, ui=FAKE_UI, screen=screen, haptics=haptics)
        yield haptics, panels, screen
    finally:
        if panels is not None:
            panels.close()
        haptics.close()
        driver.close()


def run_haptic(sim, drivers, steps, dt_wall=1.0 / 120.0):
    import time

    for n in range(steps):
        for d in drivers:
            d.update(n * sim.dt)
        sim.step()
        time.sleep(dt_wall)  # give the servo ticks between steps


def wait_for(predicate, timeout=2.0):
    import time

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def test_conditioner_follows_omnisurg():
    """input/haptics.py:20-76: source switch outputs zero once, low-pass step response, slew, ramp, axis gains, cap."""
    from i4h_standalone.app.haptics import Conditioner, Conditioning

    c = Conditioner(Conditioning(max_force_n=10.0, low_pass_hz=0.0, max_slew_n_per_s=0.0, switch_ramp_s=0.0))
    f, sat = c.condition((1.0, 2.0, 3.0), 1e-3, "a")
    assert np.all(f == 0.0) and not sat  # the first sample of a new source
    f, sat = c.condition((1.0, 2.0, 3.0), 1e-3, "a")
    assert np.allclose(f, (1.0, 2.0, 3.0)) and not sat  # no filtering: passes through
    f, _ = c.condition((float("nan"), 2.0, 3.0), 1e-3, "a")
    assert np.all(f == 0.0)  # a non-finite input is zero
    f, sat = c.condition((30.0, 0.0, 0.0), 1e-3, "b")
    assert np.all(f == 0.0)  # switching the source resets
    f, sat = c.condition((30.0, 0.0, 0.0), 1e-3, "b")
    assert np.allclose(f, (10.0, 0.0, 0.0)) and sat  # norm cap

    c = Conditioner(Conditioning(max_force_n=10.0, low_pass_hz=20.0, max_slew_n_per_s=0.0, switch_ramp_s=0.0))
    c.condition((0.0, 0.0, 0.0), 1e-3, "a")
    alpha = 1.0 - np.exp(-2.0 * np.pi * 20.0 * 1e-3)
    f, _ = c.condition((1.0, 0.0, 0.0), 1e-3, "a")
    assert np.isclose(f[0], alpha)
    f, _ = c.condition((1.0, 0.0, 0.0), 1e-3, "a")
    assert np.isclose(f[0], alpha + alpha * (1.0 - alpha))

    c = Conditioner(Conditioning(max_force_n=10.0, low_pass_hz=0.0, max_slew_n_per_s=100.0, switch_ramp_s=0.0))
    c.condition((0.0, 0.0, 0.0), 1e-3, "a")
    f, _ = c.condition((1.0, 0.0, 0.0), 1e-3, "a")
    assert np.isclose(f[0], 0.1)  # 100 N/s over 1 ms

    c = Conditioner(Conditioning(max_force_n=10.0, low_pass_hz=0.0, max_slew_n_per_s=0.0, switch_ramp_s=0.01, axis_gains=(1.0, 0.5, 0.0)))
    c.condition((0.0, 0.0, 0.0), 1e-3, "a")
    f, _ = c.condition((1.0, 1.0, 1.0), 1e-3, "a")
    assert np.allclose(f, (0.1, 0.05, 0.0))  # 1 ms into a 10 ms ramp, then the axis gains

    c = Conditioner(Conditioning(max_force_n=10.0, low_pass_hz=20.0, max_slew_n_per_s=1.0, switch_ramp_s=1.0, low_pass=False, slew=False, ramp=False))
    c.condition((0.0, 0.0, 0.0), 1e-3, "a")
    f, _ = c.condition((0.3, 0.0, 0.0), 1e-3, "a")
    assert np.isclose(f[0], 0.3)  # every stage switched off: pass-through, the values kept for switching back on

    c = Conditioner(Conditioning(max_force_n=0.0, low_pass_hz=0.0, max_slew_n_per_s=0.0, switch_ramp_s=0.0))
    c.condition((0.0, 0.0, 0.0), 1e-3, "a")
    f, sat = c.condition((1.0, 0.0, 0.0), 1e-3, "a")
    assert np.all(f == 0.0) and sat  # no explicit cap: no output


def test_haptics_config_validation():
    from i4h_standalone.app.haptics import HapticsConfig

    cfg = HapticsConfig(enabled=True, max_force_n=3.0, devices={"primary": {"max_force_n": 2.0, "axis_gains": [1, 1, 0]}})
    assert cfg.conditioning("primary").max_force_n == 2.0 and cfg.conditioning("primary").axis_gains == (1.0, 1.0, 0.0)
    assert cfg.conditioning("other").max_force_n == 3.0
    with pytest.raises(TypeError):
        HapticsConfig(force_scale=1.0)  # unknown key
    with pytest.raises(ValueError):
        HapticsConfig(devices={"primary": {"gain": 2.0}})
    with pytest.raises(ValueError):
        HapticsConfig(low_pass_hz=-1.0)
    with pytest.raises(ValueError):
        HapticsConfig(rate_hz=0.0)
    with pytest.raises(ValueError):
        HapticsConfig(axis_gains=[1.0, 1.0])
    with pytest.raises(ValueError):
        HapticsConfig(slew="off")
    assert HapticsConfig(stale=False, devices={"primary": {"ramp": False}}).conditioning("primary") == HapticsConfig(stale=False, ramp=False).base
    # the handles block inherits the pose values and states what differs; per-device handle overrides sit under devices.<id>.handle
    cfg = HapticsConfig(max_force_n=0.5, low_pass_hz=30.0, handles={"newtons_per_unit": 0.02, "max_force_n": 0.3},
                        devices={"primary": {"max_force_n": 0.4, "handle": {"max_force_n": 0.1}}})
    assert cfg.handle_base.low_pass_hz == 30.0 and cfg.handle_base.max_force_n == 0.3 and cfg.handle_base.newtons_per_unit == 0.02
    assert cfg.conditioning("primary").max_force_n == 0.4 and cfg.handle_conditioning("primary").max_force_n == 0.1
    assert cfg.handle_conditioning("other").max_force_n == 0.3 and cfg.conditioning("primary").low_pass_hz == 30.0
    with pytest.raises(ValueError):
        HapticsConfig(handles={"gain": 1.0})
    with pytest.raises(ValueError):
        HapticsConfig(devices={"primary": {"handle": {"gain": 1.0}}})


def test_servo_renders_the_probe_force_and_zeroes_on_pause_reset_switch_stale_and_close(tmp_path):
    sim, driver, haptics = make_haptics(haptic_case(tmp_path))
    servo = haptics.servo
    src = driver.manager.devices[0].source
    try:
        assert servo.is_alive() and servo.enabled
        run_haptic(sim, [driver, haptics], 40)
        # the physics thread reads the servo's sample: the probe sits where the fake device says
        state = driver.manager.states["pad"]
        assert state.active and np.allclose(state.position, [0.0, 0.0, 0.05])
        assert find(sim, ToolForce).force.numpy()[0, 0, 2] < 0.0  # the block pushes the probe down (drawn env 0)
        forces = np.array([f for _, f in src.forces[-20:]])
        assert forces.shape[0] == 20 and (forces[:, 2] < 0.0).all() and np.abs(forces[:, 2]).max() <= 0.5 + 1e-6, forces  # sign, cap
        expected = find(sim, ToolForce).force.numpy()[0, 0] * 1e-3
        assert np.allclose(forces[-1], np.clip(expected, -0.5, 0.5), rtol=0.5, atol=0.1), (forces[-1], expected)  # one step old at most
        status = servo.status()
        assert status["ticks"] > 40 and status["devices"]["pad"].reason == "on" and status["devices"]["pad"].tool == "probe"
        assert "servo" in servo.status_text() and "probe" in servo.status_text()

        # pause: zero within a few ticks; resume: force returns
        haptics.pause(True)
        assert wait_for(lambda: src.forces[-1][1] == (0.0, 0.0, 0.0)) and servo.status()["devices"]["pad"].reason == "paused"
        haptics.pause(False)
        run_haptic(sim, [driver, haptics], 10)
        assert src.forces[-1][1][2] < 0.0

        # reset of the drawn env: a new generation zeroes and restarts the conditioners
        gen = haptics.mailbox.generation
        haptics.reset([1])
        assert haptics.mailbox.generation == gen  # env 1 is not drawn
        haptics.reset()
        assert haptics.mailbox.generation == gen + 1
        assert wait_for(lambda: src.forces[-1][1] == (0.0, 0.0, 0.0)) and servo.status()["devices"]["pad"].reason == "reset"  # old samples refused
        run_haptic(sim, [driver, haptics], 10)  # fresh samples of the new generation: the force returns
        assert src.forces[-1][1][2] < 0.0

        # no tool held (the device put its tool down): zero
        binding, driver.binding = driver.binding, {}
        run_haptic(sim, [driver, haptics], 8)
        assert src.forces[-1][1] == (0.0, 0.0, 0.0) and servo.status()["devices"]["pad"].reason == "no tool held"
        driver.binding = binding
        run_haptic(sim, [driver, haptics], 10)
        assert src.forces[-1][1][2] < 0.0

        # stale: physics stops staging, the force goes to zero after stale_timeout_s
        import time

        time.sleep(0.3)
        assert src.forces[-1][1] == (0.0, 0.0, 0.0) and servo.status()["devices"]["pad"].reason == "stale"

        # output switch (the panel's checkbox)
        run_haptic(sim, [driver, haptics], 10)
        assert src.forces[-1][1][2] < 0.0
        servo.enabled = False
        assert wait_for(lambda: src.forces[-1][1] == (0.0, 0.0, 0.0))
        servo.enabled = True
        servo.set_param("max_force_n", 0.001)
        run_haptic(sim, [driver, haptics], 10)
        assert 0.0 < abs(src.forces[-1][1][2]) <= 0.001 + 1e-9
    finally:
        haptics.close()
        driver.close()
    assert not servo.is_alive() and src.forces[-1][1] == (0.0, 0.0, 0.0)  # motors off on close
    servo.check()


def test_servo_keeps_the_force_in_the_sim_frame_under_a_rotation_offset(tmp_path):
    """``InputDevice.poll`` rotates only the orientation by settings.rotation_offset, never the position, so the device's
    translational axes are the sim's and the force must not be rotated (omnisurg inverse-rotated it: wrong)."""
    import yaml

    path = haptic_case(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["inputs"]["pad"]["settings"]["rotation_offset_euler_degrees"] = [0.0, 0.0, 90.0]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    sim, driver, haptics = make_haptics(path)
    servo = haptics.servo
    try:
        from i4h_standalone.app.haptics import Sample

        d = servo.devices["pad"]
        d["status"].connected = True
        sample = Sample(np.array([[400.0, 0.0, 0.0]], np.float32), np.zeros((1, 3), np.float32), np.zeros((1, 4), np.float32), 0, 0, {"pad": 0},
                        __import__("time").perf_counter(), 0)
        servo._command("pad", d, sample, sample.t_wall, 1e-3)  # first sample of the source: zero
        f, handle_n = servo._command("pad", d, sample, sample.t_wall, 1e-3)
        assert np.allclose(f, (0.4, 0.0, 0.0), atol=1e-6) and handle_n == 0.0, f  # 400 sim units * 1e-3: +x stays +x whatever the orientation offset
    finally:
        haptics.close()
        driver.close()


def test_servo_under_the_physics_worker_records_timings_and_closes_in_order(tmp_path):
    from i4h_standalone.app.metrics import Timings
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.snapshots import Snapshots

    sim, driver, haptics = make_haptics(haptic_case(tmp_path))
    haptics.servo.timings = timings = Timings(warmup=0)
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    haptics.mailbox.capture_lock = ring.lock
    worker = PhysicsWorker(sim, [driver, haptics], ring, steps=90, realtime=True, timings=timings)
    worker.start()
    assert wait_for(lambda: worker.done, timeout=10.0)
    worker.check()
    src = driver.manager.devices[0].source
    assert src.forces[-1][1][2] < 0.0 and haptics.mailbox.staged == 90
    summary = timings.summary(("haptics.tick", "haptics.io"))
    assert summary["haptics.tick"]["count"] > 100 and summary["haptics.io"]["p50"] < 5.0, summary
    for d in reversed([driver, haptics]):  # the CLI's order: the servo stops before the devices close
        d.close()
    assert not haptics.servo.is_alive() and src.forces[-1][1] == (0.0, 0.0, 0.0)


def test_cli_runs_headless_with_haptics(tmp_path, capsys):
    from i4h_standalone.app.cli import main

    path = haptic_case(tmp_path, envs=1)
    assert main([str(path), "--input", "devices", "--render", "noop", "--steps", "60", "--haptics", "--metrics"]) == 0
    out = capsys.readouterr().out
    assert "haptics.tick" in out and "haptics.io" in out and "60 steps" in out
    with pytest.raises(SystemExit):  # haptics needs devices
        main([str(path), "--input", "scripted", "--render", "noop", "--steps", "1", "--haptics"])
    assert main([str(path), "--input", "devices", "--render", "noop", "--steps", "5", "--no-haptics"]) == 0


def test_panels_show_a_haptics_window(tmp_path):
    from test_ui import by_label

    with haptics_panels(haptic_case(tmp_path)) as (haptics, panels, screen):
        assert panels.haptics_window is not None and len(panels.windows) == 4
        by_label(screen, "output").fire(False)
        by_label(screen, "max_force_n").fire(0.25)
        by_label(screen, "slew on").fire(False)  # the stage switch above the slider
        panels.update()
        p = haptics.servo.params()
        assert haptics.servo.enabled is False and p.max_force_n == 0.25 and p.slew is False and p.low_pass is True
        labels = [w.label for w in screen.children[2].children]  # the Haptics window: each switch right above its slider
        assert labels.index("low_pass on") == labels.index("low_pass_hz") - 1 and labels.index("stale on") == labels.index("stale_timeout_s") - 1
        by_label(screen, "slew on").fire(True)
        panels.update()
        assert haptics.servo.params().slew is True
        assert "OFF" in panels.haptics_text.text
        panels.close()
        assert not screen.children


# --- the Avatar, the instrument box and the handles (input/avatar.py), on stubs: no hardware here -----------------------


class StubHaptic:
    """The SDK HapticDevice getters the adapter uses: tip (cm), tool direction, roll (rad); records the motor command."""

    def __init__(self, tip_cm=(0.0, 0.0, 20.0), direction=(0.0, 0.0, 1.0), roll=0.0):
        self.tip_cm, self.direction, self.roll = tip_cm, direction, roll
        self.commands = []
        self.updates = 0

    def perform_update(self):
        self.updates += 1

    def get_tip_position(self):
        return self.tip_cm

    def get_instrument_direction(self):
        return self.direction

    def get_value(self, dof):
        assert dof.value == 0  # Dof.ROT
        return self.roll

    def set_force_and_torques(self, z, yaw, pitch, rot):
        self.commands.append((z, yaw, pitch, rot))


class StubBox:
    """The SDK IBController getters the adapter uses; ``forces`` is what is queued, ``sent`` what a round trip transmitted."""

    def __init__(self, openings_pct=(100.0, 0.0, 50.0), pedals=(0, 1), ids=(7, 8, 0), measured=(0.0, 0.1, 0.2)):
        self.openings_pct, self.pedals, self.ids, self.measured = list(openings_pct), pedals, list(ids), list(measured)
        self.forces, self.sent = {}, {}
        self.updates = 0
        self.fail = False

    def perform_update(self):
        if self.fail:
            raise RuntimeError("box unplugged")
        self.updates += 1
        self.sent = dict(self.forces)

    def get_opening_prc_from_channel(self, c):
        return self.openings_pct[c] if c < len(self.openings_pct) else 0.0

    def get_pedal_state(self, p):
        return self.pedals[p]

    def get_handle_id_from_channel(self, c):
        return self.ids[c] if c < len(self.ids) else 0

    def get_force_from_channel(self, c):
        return self.measured[c] if c < len(self.measured) else 0.0

    def set_handle_force_to_channel(self, c, f):
        self.forces[c] = f  # queued


class StubMiniMou:
    """What poll_minimou_controller reads: a pose, a disconnected handle port (state -1), no handle readings."""

    def perform_update(self):
        pass

    def get_handle_connection_state(self):
        return -1

    def get_position(self):
        return (10.0, 20.0, 30.0, 1.0)

    def get_orientation(self):
        return (0.0, 0.0, 0.0, 1.0)

    def get_tool_pos(self):
        return 0.4

    def get_handle_opening_value(self):
        return 0.4

    def get_handle_activity(self):
        return 0.0


def test_avatar_pose_and_motor_mapping():
    from i4h_physics.quat import quat_rotate
    from i4h_standalone.input.avatar import CM, Avatar, quat_from_direction_roll

    stub = StubHaptic(tip_cm=(0.0, 0.0, 20.0), direction=(0.0, 0.0, 1.0), roll=0.3)
    dev = Avatar(stub, scale=(2.0, 2.0, 2.0))
    sample = dev.poll()
    assert sample["active"] and np.allclose(sample["position"], [0.0, 0.0, 0.2 * 2.0])  # cm -> m, then the workspace scale
    assert np.allclose(quat_rotate(sample["rotation"], [0.0, 0.0, -1.0]), [0.0, 0.0, 1.0], atol=1e-6)  # local -z along the tool
    assert sample["grip"] == 0.0 and sample["buttons"] == {} and sample["button"] is False  # no grip of its own: a handle supplies it
    q0, q1 = quat_from_direction_roll((0.0, 0.0, 1.0), 0.0), quat_from_direction_roll((0.0, 0.0, 1.0), 0.3)
    x0, x1 = quat_rotate(q0, [1.0, 0.0, 0.0]), quat_rotate(q1, [1.0, 0.0, 0.0])
    assert np.isclose(np.dot(x0, x1), np.cos(0.3), atol=1e-6) and np.isclose(x1[2], 0.0, atol=1e-6)

    dev.queue_force((0.0, 0.0, 0.4))  # along the tool: pure insertion force, no torque
    assert np.allclose(stub.commands[-1], (0.4, 0.0, 0.0, 0.0))
    dev.queue_force((0.4, 0.0, 0.0))  # lateral at the tip r = 0.2 m along z: moment r x F = (0, 0.08, 0) -> the yaw axis (y)
    assert np.allclose(stub.commands[-1], (0.0, 0.08, 0.0, 0.0), atol=1e-9)
    dev.queue_force((0.0, 0.4, 0.0))  # r x F = (-0.08, 0, 0) -> the pitch axis (x)
    assert np.allclose(stub.commands[-1], (0.0, 0.0, -0.08, 0.0), atol=1e-9)
    dev.queue_force((float("nan"), 0.0, 0.0))
    assert stub.commands[-1] == (0.0, 0.0, 0.0, 0.0)
    assert CM == 0.01


def test_handle_supplies_grip_and_pedals_on_any_pose_device():
    """The box channel replaces the device's own grip readout: on a fake, an Avatar and a MiniMou stub (whose handle port is
    disconnected and would have made the whole device inactive); an inactive sample is left untouched; a box that stops
    answering gives grip 0 and no pedals while the pose stays."""
    from i4h_standalone.input.avatar import INACTIVE, Avatar, Handle, InstrumentBox
    from i4h_standalone.input.follou import _MiniMouSampleState, poll_minimou_controller
    from i4h_standalone.input.sources import DeviceSource, FakeSource

    stub_box = StubBox(openings_pct=(100.0, 10.0, 50.0), pedals=(1, 0))
    box = InstrumentBox(stub_box, channels=3)
    assert box.poll()["grips"].tolist() == [0.0, pytest.approx(0.9), 0.5] and box.pedals == (1, 0)
    assert box.ids.tolist() == [7, 8, 0] and box.measured.tolist() == pytest.approx([0.0, 0.1, 0.2])
    handle = Handle(box, 1)
    assert handle.grip() == pytest.approx(0.9) and handle.buttons() == {"secondary": True, "fire": False}
    assert handle.status() == {"channel": 1, "id": 8, "opening": pytest.approx(0.9), "queued": 0.0, "measured": pytest.approx(0.1), "pwm": 0, "connected": True}
    with pytest.raises(ValueError):
        Handle(box, 3)

    fake = FakeSource({"position": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0), "grip": 0.0, "buttons": {}, "io_ms": 0.0})
    fake.handle = handle
    sample = fake.poll()
    assert sample["grip"] == pytest.approx(0.9) and sample["button"] is True and sample["buttons"] == {"secondary": True, "fire": False}
    fake.io((0.0, 0.0, 0.0), 0.3)
    assert stub_box.forces[1] == pytest.approx(0.3) and fake.handle_forces[-1][1] == 0.3  # queued on the channel, sent by the box's round trip

    avatar = DeviceSource(Avatar(StubHaptic()), "avatar")
    avatar.handle = handle
    assert avatar.poll()["grip"] == pytest.approx(0.9) and avatar.io((0.0, 0.0, 0.0), 0.2)["buttons"]["secondary"] is True

    state = _MiniMouSampleState()
    assert not poll_minimou_controller(StubMiniMou(), state, handle_readout="raw")["active"]  # the port says disconnected: the old rule
    plain = poll_minimou_controller(StubMiniMou(), state, handle_readout="box")
    assert plain["active"] and plain["grip"] == 0.0 and plain["button"] is False  # the port is ignored; the handle fills the grip
    assert handle.apply(plain)["grip"] == pytest.approx(0.9)
    assert handle.apply(INACTIVE) is INACTIVE  # never written

    stub_box.fail = True
    box.poll()
    assert not box.active and handle.grip() == 0.0 and handle.buttons() == {} and handle.status()["connected"] is False
    sample = fake.poll()
    assert sample["active"] and sample["grip"] == 0.0 and sample["buttons"] == {}  # the pose stays, the jaw opens


def test_manager_pairs_handles_and_validates(monkeypatch):
    from i4h_standalone.app.instruments import SQUEEZE_BACKENDS
    from i4h_standalone.input import DeviceConfig, InputManager, normalize_input_backend

    stub_haptic, stub_box = StubHaptic(), StubBox(openings_pct=(0.0, 10.0))
    open_stub_avatar(monkeypatch, stub_haptic)
    open_stub_box(monkeypatch, stub_box)
    assert normalize_input_backend("follou_haptic") == "avatar"  # the old name still loads
    configs = [DeviceConfig("box", "ibcontroller", settings={"channels": 2}),
               DeviceConfig("port", "follou_haptic", scale=1.0, settings={"handle": {"box": "box", "channel": 1}, "position_offset": [1.0, 2.0, 3.0]}),
               DeviceConfig("pad", "fake", settings={"handle": {"box": "box"}})]
    assert configs[1].backend == "avatar" and configs[2].settings["handle"] == {"box": "box", "channel": 0}
    manager = InputManager.from_configs(configs)
    manager.init()
    port, box, pad = manager.devices[1].source, manager.devices[0].source, manager.devices[2].source
    assert port.handle.box is box.adapter and port.handle.channel == 1 and pad.handle.channel == 0
    box.io((0.0, 0.0, 0.0))  # the box's round trip refreshes the openings the handles read
    sample = port.io((0.0, 0.0, 0.4), 0.25)
    assert sample["grip"] == pytest.approx(0.9, abs=1e-6) and stub_haptic.commands[-1][0] == pytest.approx(0.4)
    assert stub_box.forces[1] == pytest.approx(0.25)  # queued on the handle's channel
    box.io((0.0, 0.0, 0.0))
    assert stub_box.sent[1] == pytest.approx(0.25)  # transmitted by the box's next round trip (the servo lists boxes last)
    manager.poll()  # the physics thread: the box binds to no instrument, the port carries the pose plus the offset
    assert not manager.states["box"].active
    assert manager.states["port"].active and np.allclose(manager.states["port"].position, np.array([0.0, 0.0, 0.2]) + [1.0, 2.0, 3.0])
    assert manager.states["port"].grip == pytest.approx(0.9, abs=1e-6) and manager.states["pad"].grip == 1.0  # channel 0: 0 % open
    manager.close()
    assert stub_haptic.commands[-1] == (0.0, 0.0, 0.0, 0.0) and stub_box.sent[1] == 0.0  # motors released

    with pytest.raises(ValueError):  # the box must be an instrument box
        InputManager.from_configs([DeviceConfig("pad", "fake"), DeviceConfig("port", "fake", settings={"handle": {"box": "pad"}})]).init()
    with pytest.raises(ValueError):  # channel beyond the box
        InputManager.from_configs([DeviceConfig("box", "ibcontroller", settings={"channels": 1}),
                                   DeviceConfig("port", "avatar", settings={"handle": {"box": "box", "channel": 1}})]).init()
    with pytest.raises(ValueError):
        DeviceConfig("port", "avatar", settings={"handle": {"channel": 1}})
    with pytest.raises(ValueError):
        DeviceConfig("port", "avatar", settings={"yaw_axis": [0, 0, 0]})
    with pytest.raises(ValueError):
        DeviceConfig("port", "avatar", settings={"handle_gain": 1.0})  # moved to haptics.handles
    mm = DeviceConfig("m", "minimou", settings={"handle": {"box": "box", "channel": 2}, "handle_readout": "raw"})
    assert mm.settings["handle_readout"] == "raw"  # kept for a missing box; mounting the handle bypasses it
    assert "minimou" in SQUEEZE_BACKENDS


def test_a_missing_box_leaves_the_devices_their_own_grip(monkeypatch, capsys):
    from i4h_standalone.input import DeviceConfig, InputManager
    from i4h_standalone.input import avatar as av
    from i4h_standalone.input import follou
    from i4h_standalone.input.sources import create_input_source

    def absent(cls, index, channels=6):
        raise RuntimeError(f"IBController device index {index} not available")

    monkeypatch.setattr(av.InstrumentBox, "open", classmethod(absent))
    pad = {"handle": {"box": "box", "channel": 1}, "grip": 0.7}
    for configs in ([DeviceConfig("box", "ibcontroller"), DeviceConfig("pad", "fake", settings=pad)],  # not present
                    [DeviceConfig("pad", "fake", settings=pad)]):  # not configured
        manager = InputManager.from_configs(configs)
        manager.init()
        assert [d.config.id for d in manager.devices] == ["pad"] and set(manager.states) == {"pad"}
        assert manager.devices[0].source.handle is None
        manager.poll()
        assert manager.states["pad"].grip == pytest.approx(0.7)  # the device's own grip
        manager.close()
        assert "box" in capsys.readouterr().out

    class StubController(follou.MiniMouController):  # a MiniMou keeps its configured readout until a box handle is mounted
        def __init__(self, *, device_index, scale, handle_readout):
            self.handle_readout = handle_readout

    monkeypatch.setattr(follou, "MiniMouController", StubController)
    mm = create_input_source(DeviceConfig("mm", "minimou", settings={"handle_readout": "raw"}))
    assert mm.adapter.handle_readout == "raw"
    mm.mount_handle(object())
    assert mm.adapter.handle_readout == "box"


def test_squeeze_to_fire_is_off_for_a_minimou_with_a_handle(tmp_path, monkeypatch):
    from i4h_standalone.app.instruments import InstrumentDriver

    stub_box = StubBox(openings_pct=(0.0, 0.0))
    open_stub_box(monkeypatch, stub_box)
    sim = load_case("two_worlds")
    control = find(sim, InstrumentControl)
    sim.scene.instruments["probe"].input = "pad"
    inputs = {"box": {"backend": "ibcontroller", "settings": {"channels": 2}},
              "pad": {"backend": "fake", "settings": {"handle": {"box": "box", "channel": 0}, "io_ms": 0.0}}}
    driver = InstrumentDriver(sim, control, inputs, "devices")
    try:
        assert driver.squeeze_fire == set()  # a fake never squeezes; the rule is exercised on the config below
        from i4h_standalone.input import DeviceConfig

        cfgs = [DeviceConfig("a", "minimou"), DeviceConfig("b", "minimou", settings={"handle": {"box": "box"}})]
        from i4h_standalone.app.instruments import SQUEEZE_BACKENDS

        assert {c.id for c in cfgs if c.backend in SQUEEZE_BACKENDS and not c.settings.get("handle")} == {"a"}
        driver.update(0.0)
        assert control.actions.numpy()[0, 0, 7] == 1.0  # the probe's grip comes from the box: channel 0 is 0 % open
    finally:
        driver.close()


# --- the review fixes -----------------------------------------------------------------------------------------------


def test_disabled_sources_and_a_disabled_tool_force_publish_no_stale_force():
    """A source disabled through prepare() stops zeroing its impulse; the force must not keep its last contribution."""
    sim = load_case("two_worlds")
    tf, contact = find(sim, ToolForce), find(sim, InstrumentContact)
    press(sim, 0.05, steps=30)
    assert np.abs(tf.force.numpy()[0, 0]).max() > 0.0
    contact.record_count.fill_(1)
    contact.enabled = False
    sim.prepare()
    assert not contact.record_count.numpy().any()
    press(sim, 0.0, envs=(), steps=3)  # the probe retreats: nothing touches
    assert np.all(tf.force.numpy() == 0.0) and tf.count.numpy().max() == 0 and np.all(contact.impulse.numpy() == 0.0)
    assert np.all(contact.jaw_impulse.numpy() == 0.0) and np.all(tf.jaw_load.numpy() == 0.0)
    contact.enabled = True
    sim.prepare()
    press(sim, 0.05, steps=30)
    assert np.abs(tf.force.numpy()[0, 0]).max() > 0.0
    contact.enabled = False
    sim.prepare()
    press(sim, 0.05, steps=2)
    sim.reset([0])  # a reset with a disabled source: its stale impulse goes too
    press(sim, 0.05, steps=2)
    assert np.all(tf.force.numpy()[0] == 0.0)
    contact.enabled = True
    tf.enabled = False
    sim.prepare()
    press(sim, 0.05, steps=5)
    assert np.all(tf.force.numpy() == 0.0)  # a disabled tool_force publishes nothing


def test_mailbox_publishes_complete_samples_in_order_and_refuses_old_generations():
    from i4h_standalone.app.haptics import ForceMailbox

    sim = load_case("two_worlds")
    control, tf = find(sim, InstrumentControl), find(sim, ToolForce)
    box = ForceMailbox(control, tf.force, sim.scene.device)
    press(sim, 0.05, steps=20)
    seen = []
    for step in range(12):
        f = tf.force.numpy()[0].copy()
        assert box.stage(0, {"pad": 0}, step)
        wp.synchronize_device(sim.scene.device)
        sample = box.latest()
        assert sample is not None and sample.step == step and sample.generation == 0
        assert np.array_equal(sample.force, f)  # the slot holds this step's data, never an unfinished copy
        seen.append(sample.step)
        sim.step()
    assert seen == sorted(seen)
    # an in-flight slot is never handed out as published: publish order follows the step, the newest wins
    box.generation += 1
    sample = box.latest()
    assert sample.generation == 0  # staged before the reset: the servo treats it as "reset" and outputs zero
    assert box.stage(0, {"pad": 0}, 99)
    wp.synchronize_device(sim.scene.device)
    assert box.latest().generation == 1 and box.latest().step == 99
    box.close()


def test_servo_zeroes_after_a_reset_until_fresh_data_and_counts_every_missed_period(tmp_path):
    import time

    from i4h_standalone.app.haptics import Sample

    sim, driver, haptics = make_haptics(haptic_case(tmp_path))
    servo = haptics.servo
    try:
        d = servo.devices["pad"]
        d["status"].connected = True
        old = Sample(np.array([[400.0, 0.0, 0.0]], np.float32), np.zeros((1, 3), np.float32), np.zeros((1, 4), np.float32), 0, 0, {"pad": 0},
                     time.perf_counter(), haptics.mailbox.generation)
        servo._command("pad", d, old, old.t_wall, 1e-3)
        assert servo._command("pad", d, old, old.t_wall, 1e-3)[0][0] == pytest.approx(0.4)
        haptics.reset()  # the drawn env: the generation bumps
        assert servo._command("pad", d, old, old.t_wall, 1e-3) == ((0.0, 0.0, 0.0), 0.0) and d["status"].reason == "reset"
        fresh = Sample(old.force, old.root, old.shaft, 1, 0, {"pad": 0}, time.perf_counter(), haptics.mailbox.generation)
        servo._command("pad", d, fresh, fresh.t_wall, 1e-3)  # the conditioner restarted: the first sample is zero
        assert servo._command("pad", d, fresh, fresh.t_wall, 1e-3)[0][0] == pytest.approx(0.4)
    finally:
        haptics.close()
        driver.close()

    # missed ticks count whole periods: an io of ~10 ms at a 1 ms period misses about nine per tick
    import yaml

    path = haptic_case(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["inputs"]["pad"]["settings"]["io_ms"] = 10.0
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    sim, driver, haptics = make_haptics(path)
    try:
        time.sleep(0.3)
        ticks, missed = haptics.servo.ticks, haptics.servo.missed
        assert ticks >= 15 and missed >= 7 * ticks, (ticks, missed)
    finally:
        haptics.close()
        driver.close()


def test_step_once_pauses_the_servo_and_a_dead_servo_releases_the_inputs(tmp_path):
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.snapshots import Snapshots

    sim, driver, haptics = make_haptics(haptic_case(tmp_path))
    servo, src = haptics.servo, driver.manager.devices[0].source
    try:
        sim.prepare()
        ring = Snapshots(sim)
        ring.publish()
        worker = PhysicsWorker(sim, [driver, haptics], ring)
        worker.start()
        assert wait_for(lambda: worker.steps > 10)
        worker.step_once()
        assert wait_for(lambda: servo.paused) and worker.paused  # step-once notifies the pause hook too
        worker.pause(False)
        assert wait_for(lambda: not servo.paused)
        # the servo dies: its last samples must not keep the tools active, and the driver reports the failure
        assert driver.manager.states["pad"].active
        servo.mailbox.latest = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        assert wait_for(lambda: not servo.is_alive())
        assert src.mailbox == {"active": False} and src.forces[-1][1] == (0.0, 0.0, 0.0)
        assert wait_for(lambda: worker.done)
        with pytest.raises(RuntimeError, match="physics thread failed"):
            worker.check()  # update() re-raised the servo's error
        driver.manager.poll()
        assert not driver.manager.states["pad"].active and driver.manager.states["pad"].grip == 0.0
    finally:
        try:
            haptics.close()
        except RuntimeError:
            pass
        driver.close()


def test_servo_shutdown_flushes_a_zero_handle_force_through_the_box(monkeypatch):
    """A handle force queued on the box after the box's own round trip would stay transmitted; the shutdown sends a
    second zero round to every device, and the mailboxes end inactive."""
    from types import SimpleNamespace

    from i4h_standalone.app.haptics import HapticsConfig, HapticServo
    from i4h_standalone.input import DeviceConfig, InputManager

    stub_haptic, stub_box = StubHaptic(), StubBox(openings_pct=(0.0, 0.0))
    open_stub_avatar(monkeypatch, stub_haptic)
    open_stub_box(monkeypatch, stub_box)
    manager = InputManager.from_configs([DeviceConfig("box", "ibcontroller", settings={"channels": 2}),
                                         DeviceConfig("port", "avatar", settings={"handle": {"box": "box", "channel": 1}})])
    manager.init()
    sources = {d.config.id: d.source for d in manager.devices}
    mailbox = SimpleNamespace(latest=lambda: None, generation=0)
    servo = HapticServo(HapticsConfig(enabled=True, max_force_n=0.5), sources, mailbox, ["tool"])
    sources["port"].io((0.0, 0.0, 0.4), 0.4)  # a closed handle under load: 0.4 N queued into the box after its round trip
    assert stub_box.forces[1] == pytest.approx(0.4)
    servo.start()
    servo.stop()
    assert stub_box.sent[1] == 0.0 and stub_haptic.commands[-1] == (0.0, 0.0, 0.0, 0.0)  # transmitted, not merely queued
    assert sources["box"].mailbox == {"active": False} and sources["port"].mailbox == {"active": False}
    manager.close()


# --- the handle force through the servo --------------------------------------------------------------------------------


def test_servo_sends_the_handle_force_last_in_the_tick_and_zeroes_it_with_the_pose_force(tmp_path, monkeypatch):
    """A fake pose device with a box handle (box listed first in the YAML): the servo serves boxes last, so the handle force
    of a tick leaves with the box's own round trip; the force is the conditioned jaw load; every gate zeroes it too."""
    import time

    from i4h_standalone.app.haptics import Sample

    stub_box = StubBox(openings_pct=(0.0, 100.0), pedals=(0, 0), ids=(5, 0))
    open_stub_box(monkeypatch, stub_box)
    path = haptic_case(tmp_path, handle=True, handles={"newtons_per_unit": 0.5, "max_force_n": 0.4, "low_pass_hz": 0.0, "max_slew_n_per_s": 0.0, "switch_ramp_s": 0.0})
    sim, driver, haptics = make_haptics(path)
    servo = haptics.servo
    try:
        assert list(servo.devices) == ["pad", "box"] and servo.has_handles()  # boxes last whatever the YAML order
        pad = servo.devices["pad"]
        assert pad["handle"] is not None and pad["handle"].params.newtons_per_unit == 0.5 and pad["handle"].params.max_force_n == 0.4
        run_haptic(sim, [driver, haptics], 30)
        # the probe's grip comes from channel 0 (0 % open -> closed), pedals off
        assert driver.manager.states["pad"].grip == 1.0 and driver.manager.states["pad"].buttons["fire"] is False
        # a probe has no jaws: jaw_load 0, so the handle force stays 0 while the pose force flows
        src = driver.manager.devices[1].source
        assert src.forces[-1][1][2] < 0.0 and src.handle_forces[-1][1] == 0.0 and stub_box.sent.get(0, 0.0) == 0.0
        # feed a synthetic jaw load through _command: gain 0.5 * 0.6 = 0.3 N, then the 0.4 N cap on 1.0
        pad["status"].connected = True
        sample = Sample(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32), np.zeros((1, 4), np.float32), 0, 0, {"pad": 0},
                        time.perf_counter(), haptics.mailbox.generation, np.float32([0.6]))
        servo._command("pad", pad, sample, sample.t_wall, 1e-3)  # first sample of the source: zero
        force, handle_n = servo._command("pad", pad, sample, sample.t_wall, 1e-3)
        assert handle_n == pytest.approx(0.3) and force == (0.0, 0.0, 0.0)
        sample.jaw[0] = 1.0
        _, handle_n = servo._command("pad", pad, sample, sample.t_wall, 1e-3)
        assert handle_n == pytest.approx(0.4) and pad["handle_saturated"]
        servo.handles_enabled = False
        assert servo._command("pad", pad, sample, sample.t_wall, 1e-3)[1] == 0.0
        servo.handles_enabled = True
        servo._command("pad", pad, sample, sample.t_wall, 1e-3)
        assert servo._command("pad", pad, sample, sample.t_wall, 1e-3)[1] == pytest.approx(0.4)
        haptics.pause(True)
        assert servo._command("pad", pad, sample, sample.t_wall, 1e-3) == ((0.0, 0.0, 0.0), 0.0) and pad["status"].reason == "paused"
        haptics.pause(False)
        servo.set_handle_param("max_force_n", 0.05)
        servo._command("pad", pad, sample, sample.t_wall, 1e-3)
        assert servo._command("pad", pad, sample, sample.t_wall, 1e-3)[1] == pytest.approx(0.05)
        # the tick order: a handle force queued on the box is transmitted by the same tick's box round trip
        stub_box.forces.clear()
        stub_box.sent.clear()
        src.handle.queue_force(0.2)
        servo.devices["box"]["source"].io((0.0, 0.0, 0.0))
        assert stub_box.sent[0] == pytest.approx(0.2)
        status = servo.status_text()
        assert "handle ch 0 id 5" in status and "box, 2 channels, open % [0 100], handle ids [5 0]" in status
    finally:
        haptics.close()
        driver.close()
    assert stub_box.sent[0] == 0.0 and src.mailbox == {"active": False}  # shutdown: zero transmitted, inputs released


def test_panels_show_a_handles_group(tmp_path, monkeypatch):
    from test_ui import by_label

    open_stub_box(monkeypatch, StubBox(openings_pct=(0.0, 100.0)))
    with haptics_panels(haptic_case(tmp_path, handle=True)) as (haptics, panels, screen):
        assert panels.handles_box is not None
        by_label(screen, "handle output").fire(False)
        by_label(screen, "max_force_n##handle").fire(0.2)
        by_label(screen, "slew on##handle").fire(False)
        panels.update()
        hp = haptics.servo.handle_params()
        assert haptics.servo.handles_enabled is False and hp.max_force_n == 0.2 and hp.slew is False
        assert haptics.servo.params().max_force_n == 0.5 and haptics.servo.params().slew is True  # the pose force's values untouched
        win = panels.haptics_window  # beside the Simulation window, folded: only the expanded switch shows
        assert win.position == (400.0, 10.0) and win.size == (380.0, 60.0)
        assert [w.label for w in win.children if w.visible] == ["expanded"]
        by_label(panels.haptics_window, "expanded").fire(True)
        by_label(screen, "console printout").fire(True)
        panels.update()
        assert all(w.visible for w in win.children) and win.size == (380.0, 700.0) and haptics.console is True
        slider = by_label(screen, "pad handle")  # the handle's current force, display only
        assert set(haptics.servo.handle_forces()) == {"pad"}
        monkeypatch.setattr(haptics.servo, "handle_forces", lambda: {"pad": 0.15})  # the servo thread rewrites the status every tick
        panels.update()
        assert slider.value == pytest.approx(0.15) and slider.max == pytest.approx(0.2)


# --- review fixes (2026-09-28): shutdown, status snapshot, device errors ---------------------------------------------


def test_a_hung_worker_still_stops_the_servo(tmp_path, monkeypatch):
    """The servo stops first and always: a physics worker that hangs on exit (its memory must not be freed) no longer
    leaves the 1 kHz thread sending forces until the interpreter exits."""
    import threading

    from i4h_standalone.app import cli
    from i4h_standalone.app.haptics import HapticsDriver
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render import NoopRenderer

    release = threading.Event()
    made, servos = [], []

    class Recording(PhysicsWorker):
        stop_timeout = 0.3

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    real_init = HapticsDriver.__init__

    def recording_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        servos.append(self.servo)

    monkeypatch.setattr(HapticsDriver, "__init__", recording_init)
    monkeypatch.setattr(InstrumentDriver, "update", lambda self, t: release.wait())  # a device read that never returns
    frames = []
    monkeypatch.setattr(NoopRenderer, "is_running", lambda self: len(frames) < 4)
    monkeypatch.setattr(NoopRenderer, "draw", lambda self: frames.append(1))
    monkeypatch.setattr(cli, "PhysicsWorker", Recording)
    monkeypatch.setattr(cli, "make_renderer", lambda cfg, sim, procedure=None: NoopRenderer())
    path = haptic_case(tmp_path, envs=1)
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            cli.main([str(path), "--input", "devices", "--render", "slang", "--steps", "3", "--fps", "200", "--haptics"])
        assert made[0].is_alive(), "the worker is still hung"
        assert servos and not servos[0].is_alive(), "the servo was stopped anyway"
        pad = servos[0].devices["pad"]["source"] if servos[0].devices else None
        if pad is not None:
            assert pad.forces[-1][1] == (0.0, 0.0, 0.0)  # its final round trips sent zero
    finally:
        release.set()
        for w in made:
            w.join(10.0)


def test_a_failure_while_opening_closes_what_is_open(tmp_path, monkeypatch):
    """Everything that opens something is built inside the try: a haptics driver that fails to start after the instrument
    driver opened the devices still closes that driver (its devices) and the window."""
    from i4h_standalone.app import cli
    from i4h_standalone.app.haptics import HapticsDriver
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_standalone.render import NoopRenderer

    closed = []
    real_close = InstrumentDriver.close
    monkeypatch.setattr(InstrumentDriver, "close", lambda self: (closed.append("instruments"), real_close(self)))
    monkeypatch.setattr(NoopRenderer, "close", lambda self: closed.append("renderer"))
    monkeypatch.setattr(HapticsDriver, "__init__", lambda self, *a, **k: (_ for _ in ()).throw(ValueError("servo boom")))
    with pytest.raises(ValueError, match="servo boom"):
        cli.main([str(haptic_case(tmp_path, envs=1)), "--input", "devices", "--render", "noop", "--steps", "2", "--haptics"])
    assert closed == ["instruments", "renderer"]


def test_status_is_a_snapshot_under_the_servo_lock(tmp_path):
    """status() and handle_forces() copy what the servo thread writes under its lock (a UI or console read while it runs)."""
    import threading

    sim, driver, haptics = make_haptics(haptic_case(tmp_path, envs=1))
    try:
        run_haptic(sim, [driver, haptics], 5)
        servo = haptics.servo
        done = threading.Event()
        with servo._lock:
            reader = threading.Thread(target=lambda: (servo.status(), servo.handle_forces(), done.set()))
            reader.start()
            assert not done.wait(0.1), "status() read without the servo's lock"
        assert done.wait(2.0)
        reader.join()
        for _ in range(200):  # and concurrently with the running servo
            s = servo.status()
            assert s["ticks"] >= 0 and "pad" in s["devices"]
    finally:
        haptics.close()
        driver.close()
    assert haptics.final_status is not None and "pad" in haptics.final_status


class PwmFailingBox(StubBox):
    def get_last_pwm_from_channel(self, c):
        raise RuntimeError("pwm query failed")


def test_device_errors_are_kept_and_a_failed_diagnostic_keeps_the_grips():
    """A failed poll keeps its reason (the Haptics status shows it) instead of a bare 'disconnected'; a failed
    status-only read (pwm) is reported but leaves the grips and pedals live; ids and pwm are read every 100 polls."""
    from i4h_standalone.input.avatar import InstrumentBox
    from i4h_standalone.input.sources import DeviceSource

    stub = StubBox(openings_pct=(100.0, 20.0), pedals=(1, 0), ids=(7, 8))
    box = InstrumentBox(stub, channels=2)
    src = DeviceSource(box, "box")
    assert src.io((0.0, 0.0, 0.0))["pedals"] == (1, 0) and src.error == ""
    stub.fail = True
    assert src.io((0.0, 0.0, 0.0)) == {"active": False} and not box.active
    assert src.error == "RuntimeError: box unplugged"
    stub.fail = False
    src.io((0.0, 0.0, 0.0))
    assert box.active and src.error == ""

    failing = PwmFailingBox(openings_pct=(0.0, 80.0), pedals=(0, 1))
    fbox = InstrumentBox(failing, channels=2)
    sample = fbox.poll()
    assert fbox.active and sample["pedals"] == (0, 1) and sample["grips"][1] == pytest.approx(0.2)
    assert "pwm[0]: RuntimeError: pwm query failed" in fbox.status_error and fbox.last_error == ""

    calls = []
    counting = StubBox(openings_pct=(0.0, 0.0), ids=(1, 2))
    counting.get_handle_id_from_channel = lambda c: calls.append(c) or 1
    cbox = InstrumentBox(counting, channels=2)
    for _ in range(250):
        cbox.poll()
    assert len(calls) == 2 * 3  # polls 0, 100 and 200


def test_servo_status_shows_the_device_error(tmp_path, monkeypatch):
    """The servo reads the adapter's reason after each round trip: the Haptics status names why a box is disconnected."""

    stub_box = StubBox(openings_pct=(0.0, 0.0))
    open_stub_box(monkeypatch, stub_box)
    sim, driver, haptics = make_haptics(haptic_case(tmp_path, envs=1, handle=True))
    try:
        run_haptic(sim, [driver, haptics], 3)
        assert wait_for(lambda: "box: box" in haptics.servo.status_text() and "connected" in haptics.servo.status_text())
        stub_box.fail = True
        assert wait_for(lambda: "disconnected: RuntimeError: box unplugged" in haptics.servo.status_text())
    finally:
        haptics.close()
        with pytest.raises(ExceptionGroup, match="input device cleanup failed") as error:
            driver.close()
        assert "box unplugged" in str(error.value.exceptions[0])


def test_drag_picks_with_the_drawn_envs_masses(monkeypatch):
    """A mesh variant pins other nodes: the pick uses the drawn env's inverse masses, not env 0's."""
    from types import SimpleNamespace

    import i4h_standalone.app.drag as dragmod

    inv_mass = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], np.float32)
    body = SimpleNamespace(rest=SimpleNamespace(inv_mass=inv_mass), count=3)
    renderer = SimpleNamespace(mouse_handlers=[], env=1, camera=None, window=SimpleNamespace(width=10, height=10))
    d = dragmod.DragDriver(SimpleNamespace(scene=None), SimpleNamespace(body=body), renderer, apply=lambda f: None)
    seen = []
    monkeypatch.setattr(d, "positions", lambda env: np.zeros((3, 3), np.float32))
    monkeypatch.setattr(dragmod, "pick_node", lambda points, im, *a, **k: seen.append(np.asarray(im).copy()))
    assert d.press(1.0, 1.0) is False
    np.testing.assert_array_equal(seen[0], inv_mass[1])


def test_input_settings_reject_unknown_keys():
    """A typo in a device's settings fails (AGENTS.md: unknown keys fail) instead of being silently ignored."""
    from i4h_standalone.input import DeviceConfig

    with pytest.raises(ValueError, match="unknown settings \\['device_indx'\\]"):
        DeviceConfig("mm", "minimou", settings={"device_indx": 1})
    with pytest.raises(ValueError, match="unknown settings"):
        DeviceConfig("box", "ibcontroller", settings={"handle": {"box": "x"}})  # a box mounts no handle
    with pytest.raises(ValueError, match="moved to the haptics"):
        DeviceConfig("av", "avatar", settings={"handle_gain": 1.0})
    DeviceConfig("mm", "minimou", settings={"device_index": 1, "handle_readout": "raw", "position_offset": [0, 1, 2]})


# --- review fixes (2026-09-28): handle safety and the box's status telemetry --------------------------------------------


def _handle_sample():
    import time

    from i4h_standalone.app.haptics import Sample

    return Sample(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 4)), 1, 0, {"pad": 0}, time.perf_counter(), 0, np.array([0.4]))


def test_the_handle_has_its_own_stale_gate():
    from types import SimpleNamespace

    from i4h_standalone.app.haptics import HapticsConfig, HapticServo
    from i4h_standalone.input.avatar import Handle, InstrumentBox
    from i4h_standalone.input.sources import FakeSource

    box = InstrumentBox(StubBox(), 3)
    box.active = True
    pad = FakeSource({"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "grip": 1, "buttons": {}, "io_ms": 0})
    pad.mount_handle(Handle(box, 0))
    cfg = HapticsConfig(enabled=True, max_force_n=0.5, low_pass=False, slew=False, ramp=False, stale=False,
                        handles={"stale": True, "stale_timeout_s": 0.02})
    s = _handle_sample()
    servo = HapticServo(cfg, {"pad": pad}, SimpleNamespace(generation=0), ["tool"])
    d = servo.devices["pad"]
    d["status"].connected = True
    servo._command("pad", d, s, s.t_wall, 0.001)  # the first tick on a tool primes the conditioner
    assert servo._command("pad", d, s, s.t_wall, 0.001)[1] > 0.0  # fresh: the jaw load reaches the handle
    assert servo._command("pad", d, s, s.t_wall + 1.0, 0.001)[1] == 0.0  # 1 s old: the handle's 20 ms gate, pose gate off


def test_a_failed_pose_write_zeroes_the_handle():
    import time
    from types import SimpleNamespace

    from i4h_standalone.app.haptics import HapticsConfig, HapticServo
    from i4h_standalone.input.avatar import Avatar, Handle, InstrumentBox
    from i4h_standalone.input.sources import DeviceSource

    class FailingHaptic(StubHaptic):
        fail = False

        def set_force_and_torques(self, *args):
            if self.fail:
                raise RuntimeError("pose write failed")
            super().set_force_and_torques(*args)

    ctl, bctl = FailingHaptic(), StubBox()
    box = InstrumentBox(bctl, 3)
    box.poll()
    pad = DeviceSource(Avatar(ctl), "pad")
    pad.mount_handle(Handle(box, 0))
    s = _handle_sample()
    servo = HapticServo(HapticsConfig(enabled=True, max_force_n=0.5, low_pass=False, slew=False, ramp=False, stale_timeout_s=1.0),
                        {"pad": pad, "box": DeviceSource(box, "box")}, SimpleNamespace(generation=0, latest=lambda: s), ["tool"])
    servo.start()
    try:
        deadline = time.perf_counter() + 2.0
        while bctl.sent.get(0, 0.0) == 0.0 and time.perf_counter() < deadline:
            time.sleep(0.002)
        assert bctl.sent.get(0, 0.0) > 0.0
        ctl.fail = True
        deadline = time.perf_counter() + 2.0
        while bctl.sent.get(0) != 0.0 and time.perf_counter() < deadline:
            time.sleep(0.002)
        assert bctl.sent[0] == 0.0 and servo.devices["pad"]["status"].error  # the box still transmits: zero, not the last command
    finally:
        servo.stop()
    assert bctl.sent[0] == 0.0
    box.close()


def test_bad_status_telemetry_keeps_the_grip_and_pedals():
    from i4h_standalone.input.avatar import Handle, InstrumentBox

    box = InstrumentBox(StubBox(openings_pct=(20.0,), pedals=(1, 1), ids=(4294967295,)), 1)  # an id beyond int32
    box.poll()
    handle = Handle(box, 0)
    assert box.active and handle.grip() == pytest.approx(0.8) and handle.buttons() == {"secondary": True, "fire": True}
    assert box.ids.tolist() == [0]


def test_opened_adapters_release_the_sdk_manager(monkeypatch):
    import sys
    from types import ModuleType

    from i4h_standalone.input.avatar import Avatar, InstrumentBox

    releases = []
    manager = ModuleType("follou_sdk.manager")
    sdk = ModuleType("follou_sdk")
    sdk.__path__ = []
    class StubManager:
        def get_device_controller(self, cls, count=0):
            return StubBox() if cls.__name__ == "IBController" else StubHaptic()
    manager.DeviceManager = StubManager
    manager.acquire_manager = StubManager
    manager.release_manager = lambda: releases.append(True)
    for name in ("minimou", "haptic", "ibcontroller"):
        module = ModuleType("follou_sdk.devices." + name)
        setattr(module, {"minimou": "MiniMou", "haptic": "HapticDevice", "ibcontroller": "IBController"}[name],
                type({"minimou": "MiniMou", "haptic": "HapticDevice", "ibcontroller": "IBController"}[name], (), {}))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "follou_sdk", sdk)
    monkeypatch.setitem(sys.modules, "follou_sdk.manager", manager)
    for open_ in (InstrumentBox.open, Avatar.open):
        first = open_()
        first.close()
        first.close()
        second = open_()
        assert first.controller is not second.controller
        second.close()
    assert len(releases) == 4


@needs_chole
def test_disabled_contact_and_hook_clear_transients_preserve_damage():
    from i4h_surgical.systems.instruments.hook import Hook
    sim = load_case("chole_tet_med", device="cpu", envs=2)
    contact = find(sim, InstrumentContact)
    hook = find(sim, Hook)
    contact.record_count.fill_(1)
    hook.active.fill_(1)
    contact.impulse.fill_((1, 2, 3))
    hook.impulse.fill_((1, 2, 3))
    contact.surface.appearance.visual.fill_((.2, .3, .4, .5))
    before = contact.surface.appearance.visual.numpy().copy()
    contact.enabled = hook.enabled = False
    sim.prepare()
    assert not contact.record_count.numpy().any()
    assert not contact.suppress_active.numpy().any()
    assert not contact.impulse.numpy().any() and not hook.impulse.numpy().any()
    np.testing.assert_array_equal(contact.surface.appearance.visual.numpy(), before)
    sim.step()  # enabled tool-force/thermal consumers see cleared producers
    assert not find(sim, ToolForce).force.numpy().any()
