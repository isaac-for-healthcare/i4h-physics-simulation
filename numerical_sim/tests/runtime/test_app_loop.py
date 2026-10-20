# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M8: the snapshot ring, the physics worker thread, its commands and pacing (two_worlds, no window)."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from helpers import find, wait_until
from i4h_isaaclab.composition import load_case
from i4h_physics.systems.mechanics.particles import Particles
from i4h_standalone.app.instruments import InstrumentDriver
from i4h_standalone.app.worker import Pacer, PhysicsWorker
from i4h_standalone.render import NoopRenderer
from i4h_standalone.render.snapshots import Snapshot, Snapshots
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.surface import Surface


class SlowRenderer(NoopRenderer):
    """A renderer whose frame takes ``delay`` seconds of wall time, holding no GIL while it sleeps."""

    def __init__(self, delay: float):
        self.delay = delay
        self.frames = 0

    def draw(self) -> bool:
        time.sleep(self.delay)
        self.frames += 1
        return True


def make_worker(sim, steps=0, realtime=False):
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    control = find(sim, InstrumentControl)
    driver = InstrumentDriver(sim, control, None, "scripted")
    worker = PhysicsWorker(sim, [driver], ring, steps=steps, realtime=realtime)
    return ring, worker


def test_live_snapshot_aliases_and_slots_copy(case):
    sim = load_case(case)
    scene = sim.scene
    live = Snapshot(sim)
    surface = find(sim, Surface)
    assert live.q.ptr == scene.q.ptr and live.pool_flags.ptr == scene.pool.flags.ptr and live.tris[surface].ptr == surface.tris.ptr
    assert live.control is find(sim, InstrumentControl) and live.event is None
    assert all(a.is_contiguous for a in live.arrays)
    ring = Snapshots(sim, slots=3)
    assert len(ring.slots) == 3 and ring.acquire() is None  # nothing published yet
    for slot in ring.slots:
        assert slot.q.ptr != scene.q.ptr and slot.q.shape == scene.q.shape and slot.step == -1
        assert slot.control is not live.control and slot.control.sphere_radius is live.control.sphere_radius
        assert len(slot.arrays) == len(live.arrays)
    with pytest.raises(ValueError):
        Snapshots(sim, slots=1)


def test_ring_publish_acquire_hold(case):
    sim = load_case(case)
    scene = sim.scene
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    q0 = scene.q.numpy().copy()
    sim.step()
    wp.synchronize_device(scene.device)
    snap = ring.acquire()
    assert snap is not None and snap.step == 0 and snap.held
    assert np.array_equal(snap.q.numpy(), q0)  # the pre-step positions, bitwise
    assert not np.array_equal(scene.q.numpy(), q0)
    assert ring.acquire() is snap  # nothing newer complete: the held slot again
    for _ in range(5):  # the held slot is never a write target
        sim.step()
        ring.publish()
    wp.synchronize_device(scene.device)
    assert snap.step == 0 and np.array_equal(snap.q.numpy(), q0)
    assert sorted(s.step for s in ring.slots) == [0, 4, 5]  # the worker alternates the two free slots
    newest = ring.acquire()
    assert newest is not snap and newest.step == 5 and newest.held and not snap.held
    assert np.array_equal(newest.q.numpy(), scene.q.numpy())
    # The renderer-side protocol: wait on the ready event, record the release; publishing into the released
    # slot then waits for that record on the GPU (a stream ordering, no host sync).
    stream = wp.Stream(scene.device)
    newest.wait(stream)
    newest.done(stream)
    ring.publish()
    ring.publish()
    wp.synchronize_device(scene.device)
    assert snap.step == 6 or snap.step == 7  # the released slot was reused


def test_threaded_matches_serial(case):
    serial = load_case(case)
    serial.prepare()
    driver = InstrumentDriver(serial, find(serial, InstrumentControl), None, "scripted")
    for n in range(60):
        driver.update(n * serial.dt)
        serial.step()

    threaded = load_case(case)
    ring, worker = make_worker(threaded, steps=60)
    worker.start()
    wait_until(lambda: worker.done)
    worker.check()
    assert worker.steps == 60 and worker.elapsed > 0.0 and len(worker.intervals) == 59
    for b in serial.scene.bodies:
        if b.integrate:  # float atomics: a tolerance, as in test_captured_equals_uncaptured
            assert np.allclose(serial.scene.view(serial.scene.q, b).numpy(), threaded.scene.view(threaded.scene.q, b).numpy(),
                               rtol=1e-4, atol=1e-3), b.name
    snap = ring.acquire()
    assert snap is not None and snap.step == 60 and np.array_equal(snap.q.numpy(), threaded.scene.q.numpy())


def test_worker_commands(case):
    sim = load_case(case)
    scene = sim.scene
    ring, worker = make_worker(sim)
    q0 = scene.q0.numpy().copy()
    resets = []
    worker.drivers.append(SimpleNamespace(update=lambda t: None, reset=resets.append))

    def owner_read(read):
        """Read GPU arrays on their issuing thread, then acknowledge its publication."""
        completed = threading.Event()
        result = {}
        def command(s):
            result["value"] = read(s)
            result["count"] = ring.count
            completed.set()
        worker.edit(command)
        assert completed.wait(10), "worker command did not complete"
        wait_until(lambda: ring.count > result["count"])
        worker.check()
        return result["value"]

    worker.start()
    try:
        wait_until(lambda: worker.steps >= 20)
        worker.pause(True)
        n = owner_read(lambda s: worker.steps)
        assert owner_read(lambda s: (worker.steps, worker.paused)) == (n, True)
        worker.step_once()
        wait_until(lambda: worker.steps == n + 1)
        assert owner_read(lambda s: (worker.steps, worker.paused)) == (n + 1, True)
        assert owner_read(lambda s: threading.get_ident()) == worker.ident

        published = ring.count
        worker.reset()
        owner_read(lambda s: None)  # finish the reset and queue its snapshot copy
        def reset_snapshot(s):
            wp.synchronize_device(scene.device)
            snap = ring.acquire()
            active = (scene.flags.numpy() & 1) != 0
            return snap.q.numpy()[active].copy(), active
        positions, active = owner_read(reset_snapshot)
        assert ring.count > published
        assert np.array_equal(positions, q0[active]) and worker.t == 0.0 and resets == [None]

        block = scene.bodies[0]
        worker.edit(lambda s: find(s, Particles).max_velocity.fill_(1e-6))
        worker.pause(False)
        wait_until(lambda: not worker.paused and worker.steps >= n + 30)
        def sample(s):
            return worker.steps, scene.view(scene.q, block).numpy().copy()
        m, before = owner_read(sample)
        wait_until(lambda: worker.steps >= m + 20)
        after_step, after = owner_read(sample)
        assert after_step > m and np.allclose(after, before, atol=1e-5)
    finally:
        worker.stop()
    worker.check()
    assert worker.done and not worker.is_alive()
    assert np.allclose(find(sim, Particles).max_velocity.numpy(), 1e-6)


def test_worker_error_propagates(case):
    sim = load_case(case)
    _, worker = make_worker(sim)
    worker.start()
    wait_until(lambda: worker.steps >= 5)
    worker.edit(lambda s: 1 / 0)
    wait_until(lambda: worker.done)
    assert not worker.is_alive() and isinstance(worker.error, ZeroDivisionError)
    with pytest.raises(RuntimeError, match="physics thread failed"):
        worker.check()
    worker.stop()  # joins a dead thread without raising


def test_physics_rate_independent_of_render(case):
    """A slow frame must not slow the physics: the render loop only polls the ring between sleeps."""
    sim = load_case(case)
    ring, worker = make_worker(sim)
    renderer = SlowRenderer(0.03)
    worker.start()
    try:
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            snap = ring.acquire()
            renderer.publish(snap)
            renderer.draw()
    finally:
        worker.stop()
    worker.check()
    assert renderer.frames >= 10 and worker.steps > 5 * renderer.frames, (worker.steps, renderer.frames)
    assert max(worker.intervals) < 0.1  # no frame-long stalls of the worker


class FakeClock:
    """``perf_counter`` and ``sleep`` on a simulated clock: the pacer's arithmetic without the machine's scheduling."""

    def __init__(self):
        self.now = 100.0
        self.slept = []

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_pacer_resyncs(monkeypatch):
    from i4h_standalone.app import worker as worker_module

    clock = FakeClock()
    monkeypatch.setattr(worker_module, "time", clock)
    pacer = Pacer(0.01)
    t0 = clock.now
    for _ in range(5):
        pacer.wait()
    assert abs(clock.now - t0 - 0.05) < 1e-9  # five periods, each slept to its deadline
    clock.now += 0.1  # a hitch of ten periods
    late = clock.now
    clock.slept.clear()
    pacer.wait()
    assert clock.slept == [] and pacer.next == late + 0.01  # resync: no sleep, no ten-period catch-up burst
    pacer.wait()
    assert abs(clock.slept[0] - 0.01) < 1e-9  # then one period per wait again


def test_cli_headless_direct_loop(tmp_path, capsys):
    """``--render noop`` steps on the main thread (no ring, no worker); the metrics table has the worker's scopes."""
    from i4h_standalone.app import cli

    called = []
    original = cli.threaded_loop
    cli.threaded_loop = lambda *a, **k: called.append(1) or original(*a, **k)
    try:
        assert cli.main(["two_worlds", "--steps", "90", "--render", "noop", "--metrics", str(tmp_path / "m.json")]) == 0  # past the 60-sample warm-up
        assert cli.main(["two_worlds", "--steps", "10", "--render", "noop", "--realtime"]) == 0
    finally:
        cli.threaded_loop = original
    assert not called
    out = capsys.readouterr().out
    assert "graph=on, render=noop" in out
    scopes = json.loads((tmp_path / "m.json").read_text())["scopes"]
    assert {"pace", "input", "submit", "interval"} <= set(scopes) and ("gpu" in scopes) == wp.get_device().is_cuda


def test_dt_edit_repaces(case):
    """A structural edit that changes dt (and recaptures) also changes the real-time pacing period."""
    sim = load_case(case)
    ring, worker = make_worker(sim, realtime=True)
    dt0 = sim.dt
    worker.start()
    try:
        wait_until(lambda: worker.steps >= 12)
        assert abs(float(np.median(tuple(worker.intervals))) - dt0) < 0.4 * dt0
        worker.edit(lambda s: (setattr(s, "dt", 0.05), s.prepare()))
        n = worker.steps
        wait_until(lambda: worker.steps >= n + 12, timeout=15.0)
        late = tuple(worker.intervals)[-8:]
        assert 0.04 < float(np.median(late)) < 0.08, late
    finally:
        worker.stop()
    worker.check()
    assert sim.dt == 0.05 and worker.t > 12 * 0.05


def test_cli_keeps_device_alive_when_the_worker_hangs(monkeypatch):
    """A physics thread that does not stop keeps issuing device work: the app must not free its memory."""
    from i4h_standalone.app import cli

    release = threading.Event()
    made, closed, frames = [], [], []

    class Recording(PhysicsWorker):
        stop_timeout = 0.3

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(InstrumentDriver, "update", lambda self, t: release.wait())  # a device read that never returns
    monkeypatch.setattr(NoopRenderer, "is_running", lambda self: len(frames) < 4 or frames.append(1) is None and False)
    monkeypatch.setattr(NoopRenderer, "draw", lambda self: frames.append(1) is None)
    monkeypatch.setattr(NoopRenderer, "close", lambda self: closed.append(True))
    monkeypatch.setattr(cli, "PhysicsWorker", Recording)
    monkeypatch.setattr(cli, "make_renderer", lambda cfg, sim, procedure=None: NoopRenderer())  # the worker path (slang) without a device window
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            cli.main(["two_worlds", "--steps", "3", "--render", "slang", "--input", "scripted", "--fps", "200"])
        assert made and made[0].is_alive() and not closed
    finally:
        release.set()
        for w in made:
            w.join(10.0)
    assert not made[0].is_alive() and made[0].error is None


def test_cli_cpu_serial(case):
    """A CPU device has no streams or events: the app steps and publishes on one thread; the ring refuses it."""
    from i4h_standalone.app.cli import main

    assert main(["two_worlds", "--device", "cpu", "--no-cuda-graph", "--render", "noop", "--steps", "1"]) == 0
    with pytest.raises(ValueError, match="CUDA device"):
        Snapshots(load_case(case, device="cpu"))


def test_timings_percentiles_and_warmup():
    from i4h_standalone.app.metrics import Timings

    t = Timings(warmup=0)
    for v in range(1, 101):
        t.add("a", v * 1e-3)
    s = t.summary()["a"]
    assert s["count"] == 100 and abs(s["p50"] - 50.5) < 1e-6 and abs(s["max"] - 100.0) < 1e-6 and s["p95"] > 95.0
    w = Timings(warmup=60)
    for v in range(1, 101):
        w.add("a", v * 1e-3)
    assert w.summary()["a"]["count"] == 40 and w.summary()["a"]["p50"] > 80.0
    assert "a" in w.brief(["a", "missing"]) and "missing" not in w.brief(["a", "missing"])
    assert "p95" in w.table({"case": "x"}) and "case: x" in w.table({"case": "x"})


def test_worker_timings_gpu_without_stall(case):
    """The GPU scope reads the previous step's events after publish() waited for them; pacing is not solve time."""
    from i4h_standalone.app.metrics import Timings

    sim = load_case(case)
    timings = Timings(warmup=5)
    ring, worker = make_worker(sim, steps=60, realtime=True)
    worker.timings = timings
    dev = sim.scene.device
    worker._events = [(wp.Event(dev, enable_timing=True), wp.Event(dev, enable_timing=True)) for _ in range(2)]
    worker.start()
    worker.join(30.0)
    assert not worker.is_alive(), "the worker did not finish its 60 steps"
    worker.check()
    s = timings.summary()
    assert s["gpu"]["count"] >= 50 and s["gpu"]["p50"] > 0.0 and s["submit"]["count"] >= 50
    assert s["submit"]["p50"] + s["gpu"]["p50"] < 0.5 * 1e3 * sim.dt  # two_worlds is well below real time
    assert abs(s["interval"]["p50"] - 1e3 * sim.dt) < 0.3 * 1e3 * sim.dt  # paced to dt
    assert s["pace"]["p50"] > 0.0  # the wait is its own scope


def test_live_edit_keeps_the_pacer_deadline(case):
    """A live edit must not reset the real-time pacer (a slider drag would stall physics one period per frame)."""
    sim = load_case(case)
    ring, worker = make_worker(sim, steps=0, realtime=True)
    worker.start()
    wait_until(lambda: worker.steps >= 5)
    nxt = worker._pacer.next
    worker.edit(lambda s: find(s, Particles).damping.fill_(0.9))
    wait_until(lambda: worker.steps >= 8)
    assert worker._pacer.period == sim.dt and worker._pacer.next > nxt  # advanced, not re-anchored to "now + period" repeatedly
    worker.stop()


def test_cli_metrics_json(tmp_path):
    from i4h_standalone.app.cli import main

    out = tmp_path / "m.json"
    assert main(["two_worlds", "--steps", "80", "--render", "noop", "--fps", "200", "--metrics", str(out)]) == 0
    import json

    data = json.loads(out.read_text())
    assert data["run"]["case"] == "two_worlds" and data["run"]["steps"] == 80 and data["unit"] == "ms"
    assert data["scopes"]["gpu"]["p50"] > 0.0 and data["scopes"]["submit"]["count"] > 0


def test_cli_profile(capsys):
    from i4h_standalone.app.cli import main

    assert main(["two_worlds", "--steps", "5", "--render", "noop", "--fps", "200", "--profile", "3"]) == 0
    text = capsys.readouterr().out
    assert "eager kernel profile over 3 steps" in text and "systems.mechanics.particles" in text


def test_cli_a_failing_close_still_closes_the_rest(monkeypatch):
    """Every shutdown step runs on its own: a driver whose close raises leaves the renderer closed, and main re-raises."""
    from i4h_standalone.app import cli

    closed = []
    monkeypatch.setattr(InstrumentDriver, "close", lambda self: (_ for _ in ()).throw(RuntimeError("close boom")))
    monkeypatch.setattr(NoopRenderer, "close", lambda self: closed.append("renderer"))
    with pytest.raises(RuntimeError, match="close boom"):
        cli.main(["two_worlds", "--steps", "2", "--render", "noop", "--input", "scripted"])
    assert closed == ["renderer"]
