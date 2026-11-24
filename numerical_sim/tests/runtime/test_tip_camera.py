# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9e: the tip camera (rod tips and instrument shafts) with stable roll, and the camera pose helpers."""


import numpy as np


from i4h_host.camera.camera import Camera, stable_frame


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
