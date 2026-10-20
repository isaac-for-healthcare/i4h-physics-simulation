# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import needs_external_data, slang_tests
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import build_procedure, load_case, read_case
from i4h_physics.systems.mechanics.particles import Particles


@slang_tests
@pytest.mark.parametrize("case", ["aorta_shell_reach", "aorta_centreline_reach"])
@needs_external_data
def test_views_panels_and_keyboard(case):
    import warp as wp
    from i4h_standalone.app.catheter import CatheterDriver
    from i4h_standalone.app.ui import Panels, SimControls
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshots

    cfg = read_case(case)
    sim = load_case(cfg, envs=2)
    procedure = build_procedure(cfg, sim)
    render = cfg["render"].copy()
    render.pop("backend")
    renderer = SlangRenderer(sim, procedure=procedure, **dict(render, size=(320, 180), window=False))
    sim.prepare()
    control = next(s for s in sim.systems if isinstance(s, CatheterControl))
    driver = CatheterDriver(sim, control, "keyboard", renderer)
    # Exercise the same keyboard handler registered on the real renderer.
    driver.held.handle_key(SimpleNamespace(key=SimpleNamespace(name="i"), is_key_press=lambda: True,
                                           is_key_release=lambda: False))
    driver.update(0)
    assert control.actions.numpy()[0, 0] == 1 and not control.actions.numpy()[1].any()
    sim.step()
    procedure.update(sim.dt)
    renderer.publish()
    renderer.draw()
    free = renderer.read_frame()
    assert np.std(free[..., :3], axis=(0, 1)).max() > 0.001  # anatomy is actually in view
    renderer.set_camera_mode("tip")
    renderer.publish()
    renderer.draw()
    tip = renderer.read_frame()
    assert np.isfinite(tip).all() and not np.allclose(free, tip, atol=0.01)
    renderer.set_camera_mode("free")
    ring = Snapshots(sim, procedure=procedure)
    ring.publish()
    renderer.stream, renderer.lock = wp.Stream(sim.scene.device), ring.lock
    worker = PhysicsWorker(sim, [driver], ring, steps=0, procedure=procedure)
    panels = Panels(worker, renderer, SimControls(worker, (0, 0, 0)), overrides_path="/dev/null", procedure=procedure)
    assert panels.procedure_window in panels.windows
    assert panels.task_text.parent is panels.procedure_window
    assert panels.procedure_window.title == "Procedure"
    panels.publish_procedure(procedure.rows.numpy()[0])
    assert panels.task_text.text
    particles = next(s for s in sim.systems if isinstance(s, Particles))
    row = next(r for r in panels.rows if r.system is particles and r.knob.label == "damping")
    row.widgets[0].callback(0.98)
    panels.update()
    worker.pause(True)
    worker.start()
    try:
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            panels.update()
            if worker._queue.empty():
                break
            time.sleep(0.01)
    finally:
        worker.stop()
        worker.join(timeout=10)
    assert not worker.is_alive()
    np.testing.assert_allclose(particles.damping.numpy(), 0.98)
    panels.close()
    driver.close()
    renderer.close()
