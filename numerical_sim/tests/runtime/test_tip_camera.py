# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9e: the tip camera (rod tips and instrument shafts) with stable roll, and the camera pose helpers."""

import numpy as np
import pytest
import warp as wp
from helpers import free_rod, render_tests, run_rates
from i4h_isaaclab.camera.camera import Camera, stable_frame
from i4h_isaaclab.camera.rig import TipCamera, resolve_tip
from i4h_isaaclab.composition import load_case
from i4h_physics.quat import quat_rotate
from i4h_standalone.render.snapshots import Snapshot
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


def assert_frame(right, up, forward):
    for a, b in ((right, up), (up, forward), (forward, right)):
        assert abs(float(np.dot(a, b))) < 1e-5
    assert np.allclose([np.linalg.norm(v) for v in (right, up, forward)], 1.0, atol=1e-5)
    assert np.allclose(np.cross(forward, up), right, atol=1e-5)  # one handedness (camera.stable_frame's)


def test_stable_frame_continuity_and_fallbacks():
    world_up = np.float32([0.0, 0.0, 1.0])
    prev = None
    for k in range(200):  # a helix tangent with a slowly turning hint: no flips
        t = k * 0.05
        forward = np.float32([-np.sin(t), np.cos(t), 0.3])
        hint = np.float32([np.cos(0.2 * t) * 0.1, np.sin(0.2 * t) * 0.1, 1.0])
        right, up, f = stable_frame(forward, hint, np.float32([1.0, 0.0, 0.0]), world_up)
        assert_frame(right, up, f)
        if prev is not None:
            assert float(np.dot(up, prev)) > 0.99
        prev = up
    f = np.float32([0.0, 0.0, 1.0])
    right, up, _ = stable_frame(f, f, np.float32([0.0, 1.0, 0.0]), world_up)  # the hint is parallel: the alternate axis
    assert np.allclose(up, [0.0, 1.0, 0.0], atol=1e-6)
    right, up, _ = stable_frame(f, f, f, np.float32([1.0, 0.0, 0.0]))  # both hints parallel: the world up
    assert np.allclose(up, [1.0, 0.0, 0.0], atol=1e-6)
    right, up, _ = stable_frame(f, f, f, f)  # every hint parallel: world y
    assert np.allclose(up, [0.0, 1.0, 0.0], atol=1e-6)
    right, up, _ = stable_frame(np.float32([0.0, 1.0, 0.0]), np.zeros(3, np.float32), np.zeros(3, np.float32), np.float32([0.0, 1.0, 0.0]))
    assert_frame(right, up, np.float32([0.0, 1.0, 0.0]))


def test_camera_set_pose_round_trip():
    cam = Camera(position=(1.0, 2.0, 3.0), up=(0.0, 0.0, 1.0))
    forward = np.float32([1.0, 1.0, 0.5])
    forward /= np.linalg.norm(forward)
    right, up, f = stable_frame(forward, np.float32([0.3, -0.3, 1.0]), np.float32([1.0, 0.0, 0.0]), np.float32([0.0, 0.0, 1.0]))
    assert abs(float(np.dot(up, [0.0, 0.0, 1.0]))) < 0.99  # a rolled frame: not derivable from world up
    saved = cam.pose()
    cam.set_pose((0.5, 0.5, 0.5), f, up, lookahead=0.2)
    p, r, u, fw = cam.basis()
    assert np.allclose(p, [0.5, 0.5, 0.5]) and np.allclose(fw, f, atol=1e-6) and np.allclose(u, up, atol=1e-6) and np.allclose(r, right, atol=1e-6)
    assert np.allclose(cam.target - cam.position, f * 0.2, atol=1e-6)
    cam.restore(saved)
    assert np.allclose(cam.position, [1.0, 2.0, 3.0]) and np.allclose(cam.up, [0.0, 0.0, 1.0]) and cam.fov == saved[3]


def test_tip_camera_follows_the_rod_tip():
    sim, rod, ctl = free_rod(envs=2)
    scene, body = sim.scene, rod.body
    rig = TipCamera(sim, {"body": "rod", "distance": 0.04})
    assert rig.kind == "rod" and rig.names == ["rod:rod"] and rig.index == 0
    run_rates(sim, 60, np.float32([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    snap = Snapshot(sim)
    P, F = body.count, rod.frame_count
    q = scene.view(scene.q, body).numpy()
    frames = rod.frame.numpy()
    tips = []
    for env in range(2):
        cam = Camera(up=(0.0, 0.0, 1.0))
        rig.publish(snap, env, None)
        assert rig.apply(cam, wp.get_stream(scene.device))
        f = q[env, P - 1] - q[env, P - 2]
        fq = frames[env * F + min(P - 1, F - 1)]
        right, up, fwd = stable_frame(f, quat_rotate(fq, (0.0, 1.0, 0.0)), quat_rotate(fq, (1.0, 0.0, 0.0)), -scene.down)
        p, r, u, fw = cam.basis()
        assert np.allclose(p, q[env, P - 1] - fwd * 0.04, atol=1e-5) and np.allclose(fw, fwd, atol=1e-5) and np.allclose(u, up, atol=1e-5)
        tips.append(q[env, P - 1, 0])
    assert abs((tips[0] - 0.0) - 2.0 * (tips[1] - 0.0)) < 0.05 or tips[0] > tips[1]  # env 0 inserted twice as far
    assert not rig.apply(Camera(), None)  # nothing pending: the pose is kept
    with pytest.raises(ValueError, match="not an instrument"):
        TipCamera(sim, {"instrument": "probe"})
    with pytest.raises(ValueError, match="not both"):
        resolve_tip(sim, {"body": "rod", "instrument": "x"})


def test_tip_camera_on_an_instrument_shaft():
    sim = load_case("two_worlds")
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    rig = TipCamera(sim, {"instrument": "probe", "distance": 0.04})
    assert rig.kind == "instrument" and "instrument:probe" in rig.names
    sim.prepare()
    actions = control.actions.numpy()
    actions[:, 0, :3] = [0.1, 0.2, 0.3]
    actions[:, 0, 3:7] = [0.0, np.sin(0.4), 0.0, np.cos(0.4)]  # a rotation about y (xyzw)
    control.actions.assign(actions)
    for _ in range(3):
        sim.step()
    snap = Snapshot(sim)
    cam = Camera(up=(0.0, 0.0, 1.0))
    rig.publish(snap, 0, None)
    assert rig.apply(cam, None)
    root, shaft = control.root.numpy()[0, 0], control.shaft_q.numpy()[0, 0]
    forward = quat_rotate(shaft, (0.0, 0.0, -1.0))
    p, r, u, fw = cam.basis()
    assert np.allclose(fw, forward, atol=1e-5) and np.allclose(p, root - forward * 0.04, atol=1e-5)
    assert abs(float(np.dot(u, quat_rotate(shaft, (0.0, 1.0, 0.0))))) > 0.99  # roll from the shaft frame
    rig.select(rig.names.index("instrument:probe"))
    assert rig.kind == "instrument"


@render_tests
def test_slang_tip_mode_offscreen_tree_case():
    """C toggles the endoscope view: the tip frame differs from the free one, follows insertion, and leaving tip mode
    restores the free camera exactly."""
    from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
    from i4h_isaaclab.composition import read_case
    from i4h_standalone.render.slang import SlangRenderer

    try:
        sim = load_case("aorta_centreline", envs=1)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    cfg = read_case("aorta_centreline")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    assert renderer.rig is not None and renderer.camera_mode == "free"
    sim.prepare()
    sim.step()
    renderer.publish()
    renderer.draw()
    free = renderer.read_frame()
    pose = renderer.camera.pose()
    renderer.set_camera_mode("tip")
    assert renderer.camera.fov == 100.0 and renderer.camera.near == 0.005
    renderer.publish()
    renderer.draw()
    tip0 = renderer.read_frame()
    assert not np.allclose(tip0, free, atol=0.05)
    control = next(s for s in sim.systems if isinstance(s, CatheterControl))
    for _ in range(60):
        control.actions.assign(np.float32([[1.0, 0.0, 0.0]]))
        sim.step()
    renderer.publish()
    renderer.draw()
    tip1 = renderer.read_frame()
    assert not np.allclose(tip1, tip0, atol=0.05)
    p_tip = renderer.camera.basis()[0]
    renderer.set_camera_mode("free")
    for a, b in zip(renderer.camera.pose(), pose):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.allclose(renderer.camera.basis()[0], p_tip)
    renderer.close()
