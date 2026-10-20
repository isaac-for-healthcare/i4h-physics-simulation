# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9d: the panels bind every system's knobs and the simulation controls without slangpy (fake widgets)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
import yaml
from helpers import find
from i4h_isaaclab.camera.camera import Camera
from i4h_physics.systems.mechanics.particles import Particles
from i4h_standalone.app.instruments import InstrumentDriver
from i4h_standalone.app.panel_adapters import panel
from i4h_standalone.app.ui import Panels, SimControls, check_overrides, load_overrides
from i4h_standalone.app.worker import PhysicsWorker
from i4h_standalone.config import load_case
from i4h_standalone.input.keyboard import KeyboardHub
from i4h_standalone.render.snapshots import Snapshots
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.thermal import Thermal


class _Widget:
    def __init__(self, parent, label="", value=None, callback=None, *args, **kwargs):
        self.parent, self.label, self.value, self.callback = parent, label, value, callback
        self.args, self.kwargs = args, kwargs
        self.visible = self.enabled = True
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def remove_child(self, child):
        self.children.remove(child)

    def fire(self, value=None):  # what a user edit does: the widget's callback with the new value
        if value is None:
            self.callback()
        else:
            self.value = value
            self.callback(value)


class _Text(_Widget):
    def __init__(self, parent, text=""):
        super().__init__(parent)
        self.text = text


class _Button(_Widget):
    def __init__(self, parent, label="", callback=None):
        super().__init__(parent, label, None, callback)


class _Combo(_Widget):
    def __init__(self, parent, label, value, callback, items, height_in_items=-1):
        super().__init__(parent, label, value, callback)
        self.items = list(items)


class _Window(_Widget):
    def __init__(self, parent, title="", position=None, size=None):
        super().__init__(parent, title)
        self.position, self.size = position, size


class _Slider(_Widget):
    def __init__(self, parent, label, value, callback, min=0.0, max=1.0, *args):
        super().__init__(parent, label, value, callback, *args)
        self.min, self.max = min, max


FAKE_UI = SimpleNamespace(Window=_Window, Group=lambda parent, label="": _Widget(parent, label), Text=_Text, Button=_Button,
                          CheckBox=_Widget, ComboBox=_Combo, ListBox=_Combo, SliderFloat=_Slider, SliderInt=_Slider, InputText=_Widget)


def widgets(root):
    out = []
    stack = [root]
    while stack:
        w = stack.pop()
        out.append(w)
        stack.extend(getattr(w, "children", []))
    return out


def by_label(root, label):
    hits = [w for w in widgets(root) if getattr(w, "label", None) == label]
    assert len(hits) == 1, (label, len(hits))
    return hits[0]


def setup(envs=2, keyboard=True):
    sim = load_case("two_worlds", envs=envs)
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    control = find(sim, InstrumentControl)
    calls = []
    renderer = SimpleNamespace(env=0, camera=Camera(), key_handlers=[], debug_spheres=False, exposure=1.0, tonemap=0, width=1280, height=720)
    renderer.set_env = lambda env: (calls.append(env), setattr(renderer, "env", int(env)))
    driver = InstrumentDriver(sim, control, None, "keyboard" if keyboard else "scripted", renderer=renderer)
    worker = PhysicsWorker(sim, [driver], ring, steps=0, realtime=False)
    controls = SimControls(worker, (0.0, 0.0, -9.81))
    screen = _Widget(None)
    panels = Panels(worker, renderer, controls, driver.hub, ui=FAKE_UI, screen=screen)
    return sim, worker, renderer, panels, screen, calls


def drain(worker):
    """Run the queued commands on the physics thread without stepping (paused)."""
    worker.pause(True)
    if not worker.is_alive():
        worker.start()
    done = [False]
    worker.edit(lambda sim: done.__setitem__(0, True))
    t0 = time.perf_counter()
    while not done[0] and time.perf_counter() - t0 < 10.0:
        time.sleep(0.005)
    worker.check()
    assert done[0]
    wp.synchronize_device(worker.sim.scene.device)


def test_systems_window_lists_every_knob_and_edits_reach_the_device():
    sim, worker, renderer, panels, screen, _ = setup()
    labels = {w.label for w in widgets(screen)}
    for s in sim.systems:
        assert f"enabled##{s.id}" in labels
        for k in panel(s):
            cols = k.labels or (None,)
            for col in cols:
                assert (k.label if col is None else f"{k.label} {col}") + f"##{s.id}" in labels
    assert [g.visible for g in panels.groups] == [True] + [False] * (len(sim.systems) - 1)  # folded at start
    by_label(screen, "##systems").fire(2)
    panels.update()
    assert [g.visible for g in panels.groups] == [i == 2 for i in range(len(sim.systems))]
    particles = find(sim, Particles)
    slider = by_label(screen, f"max_velocity##{particles.id}")
    slider.fire(7.0)
    panels.update()
    drain(worker)
    np.testing.assert_allclose(particles.max_velocity.numpy(), [7.0, 7.0])
    by_label(screen, "edit all envs").fire(False)  # drawn env only
    slider.fire(3.0)
    panels.update()
    drain(worker)
    np.testing.assert_allclose(particles.max_velocity.numpy(), [3.0, 7.0])
    renderer.env = 1  # PgUp: the sliders show env 1's row
    panels.update()
    assert slider.value == 7.0 and panels.env_slider.value == 1
    slider.fire(5.0)  # env 1 edited to 5 by the user, then back to env 0 (3) and to env 1 again: each switch shows the row
    panels.update()
    drain(worker)
    np.testing.assert_allclose(particles.max_velocity.numpy(), [3.0, 5.0])
    renderer.env = 0
    panels.update()
    assert slider.value == 3.0
    renderer.env = 1
    panels.update()
    assert slider.value == 5.0
    worker.stop()


def test_enabled_toggle_recaptures_once_per_batch():
    sim, worker, renderer, panels, screen, _ = setup()
    thermal = find(sim, Thermal)
    graph = sim.graph
    by_label(screen, f"enabled##{thermal.id}").fire(False)
    by_label(screen, "iterations").fire(3)  # a second structural edit in the same frame: one recapture
    prepares = []
    original = sim.prepare
    sim.prepare = lambda: (prepares.append(1), original())
    panels.update()
    drain(worker)
    assert thermal.enabled is False and sim.iterations == 3 and sim.graph is not graph and len(prepares) == 1
    panels.update()  # the refresh shows the new state and greys the system's sliders
    assert by_label(screen, f"enabled##{thermal.id}").value is False
    assert all(not w.enabled for row in panels.rows if row.system is thermal for w in row.widgets)
    assert panels.iterations_slider.value == 3
    worker.stop()


def test_simulation_controls_and_keys():
    sim, worker, renderer, panels, screen, calls = setup()
    hub: KeyboardHub = panels.hub
    tool = min(1, len(hub.order) - 1)  # two_worlds has one keyboard-driven tool
    by_label(screen, "env").fire(1)
    by_label(screen, "tool").fire(tool)
    by_label(screen, "spheres (X)").fire(True)
    by_label(screen, "exposure").fire(2.0)
    by_label(screen, "gravity").fire(False)
    by_label(screen, "realtime").fire(True)
    by_label(screen, "Pause").fire()
    panels.update()
    assert calls == [1] and renderer.env == 1 and hub.active == hub.order[tool] and renderer.debug_spheres and renderer.exposure == 2.0
    assert panels.controls.gravity_on is False
    drain(worker)
    assert worker.paused and worker.realtime and not sim.scene.gravity.numpy().any()
    panels.update()  # the next frame's refresh shows the worker's state
    assert panels.pause_button.label == "Resume" and panels.gravity_box.value is False and panels.realtime_box.value is True
    # G toggles back through the shared controls; the checkbox follows
    press = SimpleNamespace(key=SimpleNamespace(name="g"), is_key_press=lambda: True, is_key_release=lambda: False)
    panels.controls.handle_key(press)
    panels.update()
    assert panels.gravity_box.value is True
    # Tab moves the active tool; the combo follows
    hub.handle_key(SimpleNamespace(key=SimpleNamespace(name="tab"), is_key_press=lambda: True, is_key_release=lambda: False))
    panels.update()
    assert panels.tool_combo.value == hub.order.index(hub.active)
    # F1 hides both windows
    panels.handle_key(SimpleNamespace(key=SimpleNamespace(name="f1"), is_key_press=lambda: True, is_key_release=lambda: False))
    assert not panels.systems_window.visible and not panels.sim_window.visible
    # physics rate: a structural edit re-paces
    by_label(screen, "physics Hz").fire(120)
    panels.update()
    drain(worker)
    assert abs(sim.dt - 1.0 / 120.0) < 1e-9 and worker._pacer.period == sim.dt
    worker.stop()


def test_overrides_round_trip(tmp_path):
    sim, worker, renderer, panels, screen, _ = setup()
    particles, thermal = find(sim, Particles), find(sim, Thermal)
    path = tmp_path / "o.yaml"
    panels.overrides_path = str(path)
    by_label(screen, f"max_velocity##{particles.id}").fire(4.0)
    by_label(screen, f"enabled##{thermal.id}").fire(False)
    by_label(screen, "substeps").fire(3)
    panels.update()
    drain(worker)
    panels.save()
    data = yaml.safe_load(path.read_text())
    assert data == {"sim": {"substeps": 3}, "systems": {particles.id: {"max_velocity": 4.0}, thermal.id: {"enabled": False}}}
    worker.stop()
    fresh = load_case("two_worlds", envs=2)
    load_overrides(fresh, data)
    assert fresh.substeps == 3 and find(fresh, Thermal).enabled is False
    np.testing.assert_allclose(find(fresh, Particles).max_velocity.numpy(), 4.0)
    fresh.prepare()
    fresh.step()
    with pytest.raises(ValueError, match="unknown system"):
        check_overrides(fresh, {"systems": {"nope": {}}})
    with pytest.raises(ValueError, match="unknown knob"):
        check_overrides(fresh, {"systems": {particles.id: {"speed": 1.0}}})
    with pytest.raises(ValueError, match="unknown field"):
        check_overrides(fresh, {"sim": {"gravity": 1.0}})


def test_panel_load_applies_in_one_edit(tmp_path):
    sim, worker, renderer, panels, screen, _ = setup()
    particles = find(sim, Particles)
    path = tmp_path / "o.yaml"
    path.write_text(yaml.safe_dump({"sim": {"iterations": 2}, "systems": {particles.id: {"damping": 0.5}}}))
    panels.overrides_path = str(path)
    by_label(screen, "Load").fire()
    panels.update()
    drain(worker)
    assert sim.iterations == 2
    np.testing.assert_allclose(particles.damping.numpy(), 0.5)
    assert by_label(screen, f"damping##{particles.id}").value == 0.5 and "loaded" in panels.status_text
    panels.save()  # a loaded value is saved again even though unchanged since
    assert yaml.safe_load(path.read_text())["systems"][particles.id]["damping"] == 0.5
    worker.stop()


def test_cli_overrides_headless(tmp_path):
    from i4h_standalone.app.cli import main

    path = tmp_path / "two_worlds.overrides.yaml"
    path.write_text(yaml.safe_dump({"sim": {"substeps": 1}}))
    assert main(["two_worlds", "--steps", "10", "--render", "noop", "--fps", "200", "--overrides", str(path)]) == 0


def test_invalid_overrides_are_rejected_whole(tmp_path):
    """Values are validated before anything is written: a bad file neither half-applies nor kills the worker."""
    sim, worker, renderer, panels, screen, _ = setup()
    particles = find(sim, Particles)
    for bad, match in (({"sim": {"substeps": 0}}, "at least 1"), ({"sim": {"dt": 0.0}}, "positive"),
                       ({"sim": {"dt": 0.01, "iterations": "x"}}, "finite number"),
                       ({"systems": {particles.id: {"damping": float("nan")}}}, "finite"),
                       ({"systems": {"instrument_contact": {"iterations": 99}}}, r"int in \[1, 8\]"),
                       ({"systems": {particles.id: {"enabled": 1}}}, "bool")):
        with pytest.raises(ValueError, match=match):
            check_overrides(sim, bad)
    dt = sim.dt
    with pytest.raises(ValueError):
        load_overrides(sim, {"sim": {"dt": 0.01, "iterations": 0}})
    assert sim.dt == dt  # nothing applied
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"sim": {"substeps": 0}}))
    panels.overrides_path = str(path)
    substeps = sim.substeps
    by_label(screen, "Load").fire()
    panels.update()
    drain(worker)
    assert "load failed" in panels.status_text and sim.substeps == substeps and worker.is_alive()
    worker.check()
    worker.stop()


def test_loaded_structural_knobs_refresh_their_widgets(tmp_path):
    sim, worker, renderer, panels, screen, _ = setup()
    contact = next(s for s in sim.systems if s.id == "instrument_contact")
    path = tmp_path / "o.yaml"
    path.write_text(yaml.safe_dump({"systems": {"instrument_contact": {"iterations": 4, "sweep_samples": 12}}}))
    panels.overrides_path = str(path)
    by_label(screen, "Load").fire()
    panels.update()
    drain(worker)
    panels.update()
    assert contact.iterations == 4
    assert by_label(screen, "iterations##instrument_contact").value == 4 and by_label(screen, "sweep_samples##instrument_contact").value == 12
    worker.stop()


def test_timings_summary_is_safe_beside_a_writer():
    import threading

    from i4h_standalone.app.metrics import Timings

    t = Timings(maxlen=64, warmup=0)
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            t.add(f"scope{i % 50}", 1e-3)
            i += 1
            if i % 64 == 0:
                time.sleep(0.0)  # let the reader in
    th = threading.Thread(target=writer)
    th.start()
    try:
        for _ in range(100):
            t.summary()
            t.brief(["scope1"])
    finally:
        stop.set()
        th.join(10.0)
    assert not th.is_alive(), "the writer thread did not stop"
    summary = t.summary()  # the reads above raised nothing; the final summary sees the writer's scopes
    assert len(summary) == 50 and all(v["count"] > 0 for v in summary.values())


def test_post_knobs_apply_on_the_render_thread():
    from i4h_standalone.render.post import Post

    sim, worker, renderer, panels, screen, _ = setup()
    renderer.post = Post()
    panels2 = Panels(worker, renderer, panels.controls, panels.hub, ui=FAKE_UI, screen=_Widget(None))
    assert panels2.procedure_window is None and panels2.task_text is None
    win = panels2.render_window
    assert win.position == (400.0, 10.0) and win.size == (380.0, 60.0)
    assert win in panels2.windows and win.children[0].label == "expanded"
    assert not any(w.visible for w in win.children[1:])
    assert "exposure" not in {w.label for w in widgets(panels2.sim_window)}
    by_label(win, "expanded").fire(True)
    panels2.update()
    assert all(w.visible for w in win.children) and win.size == panels2.render_sizes[True]
    by_label(win, "bloom_enabled").fire(True)
    by_label(panels2.systems_window.parent, "vignette_strength").fire(0.5)
    by_label(panels2.systems_window.parent, "tonemap").fire(2)
    panels2.update()
    assert renderer.post.bloom_enabled is True and renderer.post.vignette_strength == 0.5 and renderer.tonemap == 2
    by_label(win, "expanded").fire(False)
    panels2.update()
    assert not any(w.visible for w in win.children[1:])
    assert renderer.post.bloom_enabled is True  # folding keeps the current settings
    labels = {w.label for w in widgets(win)}
    assert "white_balance" not in labels and "lens_dirt_texture" not in labels and "scope_enabled" in labels
