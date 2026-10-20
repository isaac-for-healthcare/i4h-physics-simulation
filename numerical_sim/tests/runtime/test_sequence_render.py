# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from helpers import render_tests, slang_tests
from test_procedure_sequence import sequence


@slang_tests
def test_sequence_snapshot_markers_history_env_and_reset():
    from i4h_isaaclab.composition import build_procedure
    from i4h_standalone.render.procedure import task_text
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshots

    sim, _, cfg = sequence(device="cuda:0", graph=True)
    cfg["procedure"]["steps"][1].update(target={"position": [0.5, 0.1, 0]}, radius=0.2)
    sim.procedure = task = build_procedure(cfg, sim)
    renderer = SlangRenderer(sim.simulation, procedure=task, size=(320, 180), window=False)
    ring = Snapshots(sim.simulation, procedure=task)
    try:
        task.update(0.1)
        sim.reset([1])
        ring.publish()
        wp.synchronize_device(sim.scene.device)
        snap = ring.acquire()
        # Advance/reset live state after publication; the rendered slot must stay frozen.
        held_progress = snap.task_progress_host.copy()
        held_history = snap.task_history_host.copy()
        sim.reset()
        for env in (0, 1):
            renderer.env = env
            renderer.publish(snap)
            assert renderer.draw()
            q = renderer.task_markers.q_dev.numpy()[0]
            np.testing.assert_allclose(q[0], snap.task_host[env, :3])
            np.testing.assert_allclose(
                np.linalg.norm(q[1:] - q[0], axis=1), held_progress[env, 1], atol=1e-6
            )
            text = task_text(
                task, snap.task_host[env], held_progress[env], held_history[env]
            )
            assert "Tip path length" in text and "pending" in text
            assert ("completed" in text) == (env == 0)
            assert np.isfinite(renderer.read_frame()).all()
        np.testing.assert_array_equal(snap.task_progress_host, held_progress)
        np.testing.assert_array_equal(snap.task_history_host, held_history)
        ring.publish()
        wp.synchronize_device(sim.scene.device)
        reset = ring.acquire()
        assert np.all(reset.task_progress_host[:, 0] == 0)
        assert np.all(reset.task_history_host[:, :, 1] == -1)
        renderer.publish(reset)
        assert renderer.draw()
    finally:
        renderer.close()


@slang_tests
def test_tiled_sequence_has_no_target_markers():
    from i4h_isaaclab.camera.tiled import TiledCamera
    from i4h_isaaclab.composition import build_procedure

    sim, _, cfg = sequence(device="cuda:0", graph=True)
    cfg["procedure"]["steps"][1].update(target={"position": [1, 1, 1]}, radius=0.4)
    sim.procedure = task = build_procedure(cfg, sim)
    camera = TiledCamera(sim.simulation, 64, 64)
    camera.prepare()
    camera.render()
    before = camera.rgb_t.clone()
    task.update(0.1)
    camera.render()
    assert (before == camera.rgb_t).all()


@render_tests
def test_gl_sequence_omits_markers_and_keeps_legacy():
    import pytest

    pytest.importorskip("pyglet")
    from i4h_standalone.render.gl import GlRenderer
    from test_procedures import rod_task

    renderers = []
    try:
        # Newton's close exits pyglet's process-wide event loop; draw both before closing it.
        for sim, legacy in (
            (sequence(device="cuda:0")[0], False),
            (rod_task("cuda:0")[0], True),
        ):
            renderer = GlRenderer(sim.simulation, procedure=sim.procedure, size=(160, 120))
            renderers.append(renderer)
            assert (renderer.task_markers is not None) == legacy
            renderer.publish()
            assert renderer.draw()
    finally:
        for renderer in reversed(renderers):
            renderer.close()
