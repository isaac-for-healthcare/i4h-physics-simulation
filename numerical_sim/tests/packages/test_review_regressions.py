# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public CLI recording and teardown regressions with generated or small packaged cases."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("backend,window,explicit_steps,expected", [
    ("noop", None, None, 200),
    ("slang", False, None, 200),
    ("slang", True, None, 0),
    ("slang", None, None, 0),
    ("gl", False, None, 0),
    ("xr", False, None, 0),
    ("slang", False, 7, 7),
])
def test_cli_step_budget_respects_offscreen_rendering(monkeypatch, tmp_path, backend, window, explicit_steps, expected):
    from i4h_standalone.app import cli
    from i4h_standalone.render import NoopRenderer

    cfg = cli.read_case("two_worlds")
    cfg["render"] = {"backend": backend}
    if window is not None:
        cfg["render"]["window"] = window
    monkeypatch.setattr(cli, "read_case", lambda _: cfg)
    monkeypatch.setattr(cli, "make_renderer", lambda *a, **kw: NoopRenderer())

    class ReachedLoop(Exception):
        pass

    def check_budget(sim, renderer, drivers, steps, *a, **kw):
        assert steps == expected
        raise ReachedLoop

    monkeypatch.setattr(cli, "serial_loop", check_budget)
    args = ["two_worlds", "--device", "cpu", "--input", "noop", "--no-haptics",
            "--overrides", str(tmp_path / "unused.yaml")]
    if explicit_steps is not None:
        args += ["--steps", str(explicit_steps)]
    with pytest.raises(ReachedLoop):
        cli.main(args)


def test_keyboard_cli_records_replayable_rows(monkeypatch, tmp_path):
    from i4h_isaaclab.camera.camera import Camera
    from i4h_standalone.app import cli
    from i4h_standalone.render import NoopRenderer

    renderer = NoopRenderer()
    renderer.camera = Camera()
    renderer.key_handlers = []
    monkeypatch.setattr(cli, "make_renderer", lambda *a, **kw: renderer)
    base = tmp_path / "keyboard"
    assert cli.main(["two_worlds", "--device", "cpu", "--render", "noop", "--input", "keyboard",
                     "--steps", "4", "--no-haptics", "--record", str(base)]) == 0
    files = list(tmp_path.glob("keyboard_*.npy"))
    assert files
    for file in files:
        rows = np.load(file)
        assert rows.shape == (4, 10) and np.isfinite(rows).all()


@pytest.mark.parametrize("mode", ["scripted", "noop"])
def test_unsupported_recording_does_not_start_physics(monkeypatch, tmp_path, mode):
    from i4h_physics.stepper import Stepper
    from i4h_standalone.app import cli

    def started(*a, **kw):
        pytest.fail("unsupported recording started physics")
    monkeypatch.setattr(Stepper, "prepare", started)
    with pytest.raises(SystemExit) as error:
        cli.main(["two_worlds", "--device", "cpu", "--render", "noop", "--input", mode,
                  "--steps", "4", "--record", str(tmp_path / "bad")])
    assert error.value.code == 2 and not list(tmp_path.glob("bad*"))


def test_empty_worker_reset_preserves_previous_partial_reset_state():
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_surgical import load_case

    sim = load_case("two_worlds", device="cpu", envs=2)
    calls = []
    driver = SimpleNamespace(reset=lambda ids: calls.append(("driver", ids)))
    procedure = SimpleNamespace(reset=lambda **kw: calls.append(("procedure", kw)))
    worker = PhysicsWorker(sim, [driver], None, procedure=procedure)
    assert worker._do_reset([0])
    assert len(calls) == 2
    worker.t = 7.0
    deadline = worker._pacer.next
    q = sim.scene.q.numpy().copy()
    assert not worker._do_reset([])
    assert len(calls) == 2 and worker.t == 7.0 and worker._pacer.next == deadline
    np.testing.assert_array_equal(sim.scene.q.numpy(), q)


def test_renderer_releases_task_pipeline_and_ldr_view_before_device(monkeypatch):
    import warp as wp
    from i4h_standalone.render.slang import SlangRenderer

    renderer = SlangRenderer.__new__(SlangRenderer)
    renderer.scene = SimpleNamespace(device="cpu")
    renderer.stream = None
    close = SimpleNamespace(close=lambda: None)
    for name in ("interop", "post_passes"):
        setattr(renderer, name, close)
    for name in ("region_highlights", "bodies", "statics", "skinned", "pieces", "rods", "clips", "fluids", "smokes"):
        setattr(renderer, name, [])
    for name in ("task_markers", "markers", "capture", "overlay", "overlay_cut", "fluid_passes", "gbuffer", "window"):
        setattr(renderer, name, None)
    renderer._textures, renderer._surface_views = {}, {}
    renderer.task_pipeline, renderer.ldr_view = object(), object()
    renderer.owns_device = True
    closed = []
    def device_close():
        assert renderer.task_pipeline is None and renderer.ldr_view is None
        closed.append(True)
    renderer.device = SimpleNamespace(wait=lambda: None, close=device_close)
    monkeypatch.setattr(wp, "synchronize_stream", lambda *a: None)
    renderer.close()
    renderer.close()
    assert closed == [True]


def test_missing_lab_api_names_the_qualified_revision():
    from i4h_isaaclab.lab import QUALIFIED, require

    require(SimpleNamespace(present=1), "present")

    @dataclasses.dataclass
    class Cfg:
        compute_final_obs: list = dataclasses.field(default_factory=list)  # a field with no class attribute

    require(Cfg, "compute_final_obs")
    with pytest.raises(RuntimeError, match="_invalidate_graph.*6864dd126fd8efd04aaebe06bc9d00c1b9b989dc"):
        require(type("NewtonManager", (), {}), "_invalidate_graph")
    assert "release/3.0.0" in QUALIFIED


def test_overflow_readouts_warn_once_from_host_copies():
    import warnings

    from i4h_standalone.render.snapshots import warn_overflows

    warned = set()
    readouts = {("contact", "record_overflow"): np.int32([0, 3]), ("contact", "count"): np.int32([5, 5]),
                ("graph", "candidate_overflow"): np.int32([0, 0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_overflows(readouts, warned)
        warn_overflows(readouts, warned)
    assert [str(w.message).split(" ")[0] for w in caught] == ["contact.record_overflow"]



@pytest.mark.parametrize("debug", [True, False])
def test_serial_loop_warns_on_overflow_readouts_in_debug(debug):
    """The serial loop (headless, CPU, GL) reports a nonzero *overflow readout after a completed step, in debug only."""
    import warnings
    from dataclasses import dataclass

    import warp as wp
    from i4h_physics.config import load_case
    from i4h_physics.systems.base import System
    from i4h_physics.systems.mechanics.particles import Particles
    from i4h_standalone.app.cli import serial_loop
    from i4h_standalone.render import NoopRenderer

    @dataclass(eq=False)
    class Dropping(System):
        def __post_init__(self):
            self.dropped = wp.zeros(self.scene.envs, dtype=wp.int32, device=self.scene.device)

        def step_end(self, scene, dt):
            self.dropped.fill_(3)

        def readouts(self):
            return {"record_overflow": self.dropped}

    rod = {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0], "count": 4, "segment_length": .02, "radius": .002}
    sim = load_case({"sim": {"device": "cpu", "envs": 2, "dt": .01, "cuda_graph": False, "debug": debug},
                     "assets": {"rod": rod}, "systems": [{"type": "particles"}, {"type": "dropping", "id": "contact"}]},
                    systems={"particles": Particles, "dropping": Dropping})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        serial_loop(sim, NoopRenderer(), [], 3, realtime=False)
    messages = [str(w.message) for w in caught if "record_overflow" in str(w.message)]
    assert len(messages) == (1 if debug else 0), messages


def test_nested_alias_checks_check_once_and_restore_at_the_outermost(monkeypatch):
    import warp as wp
    from i4h_physics import debug

    original, checks = wp.launch, []
    monkeypatch.setattr(debug, "assert_no_alias", lambda kernel, inputs, outputs: checks.append(kernel))

    @wp.kernel
    def touch(a: wp.array(dtype=float)):
        a[wp.tid()] = 1.0

    a = wp.zeros(2, dtype=float, device="cpu")
    with debug.alias_checks():
        with debug.alias_checks():
            wp.launch(touch, 2, outputs=[a], device="cpu")
        assert len(checks) == 1, "a nested context does not wrap twice"
        with pytest.raises(RuntimeError, match="inner"):
            with debug.alias_checks():
                raise RuntimeError("inner")
        assert wp.launch is not original, "the inner exception left the outer context's wrapper in place"
        wp.launch(touch, 2, outputs=[a], device="cpu")
        assert len(checks) == 2
    assert wp.launch is original
