# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Renderer plumbing: camera basis, render config, noop backend, CLI loop; Slang offscreen smoke test on demand."""


import math

import numpy as np
import pytest
from helpers import needs_chole
from i4h_host.camera.camera import Camera
from i4h_host.composition import data_dir, read_case

try:
    import slangpy  # noqa: F401

    HAVE_SLANG = True
except ImportError:
    HAVE_SLANG = False


def assert_orthonormal(right, up, forward):
    for v in (right, up, forward):
        assert math.isclose(float(np.linalg.norm(v)), 1.0, abs_tol=1e-5)
    assert abs(float(np.dot(right, up))) < 1e-5 and abs(float(np.dot(right, forward))) < 1e-5
    assert np.allclose(np.cross(right, forward), up, atol=1e-5)  # right-handed: right x forward = up


def test_camera_basis_y_up():
    cam = Camera(position=(0.0, 0.0, 0.0), pitch=0.0, yaw=0.0)
    _, right, up, forward = cam.basis()
    assert np.allclose(forward, (0.0, 0.0, -1.0), atol=1e-6) and np.allclose(right, (1.0, 0.0, 0.0), atol=1e-6)
    assert_orthonormal(right, up, forward)
    chole = Camera(position=(0.469625, 1.20226, -2.45614), pitch=6.8, yaw=4.3, fov=45.0)
    _, right, up, forward = chole.basis()
    assert math.isclose(float(forward[1]), math.sin(math.radians(6.8)), abs_tol=1e-6)
    assert math.isclose(float(forward[0]), math.cos(math.radians(6.8)) * math.sin(math.radians(4.3)), abs_tol=1e-6)
    assert_orthonormal(right, up, forward)


def test_camera_basis_z_up():
    cam = Camera(position=(0.15, -0.2, 0.17), pitch=-18.0, yaw=-37.0, up=(0.0, 0.0, 1.0))
    _, right, up, forward = cam.basis()
    assert math.isclose(float(forward[2]), math.sin(math.radians(-18.0)), abs_tol=1e-6)
    assert float(np.dot(up, (0.0, 0.0, 1.0))) > 0.9
    assert_orthonormal(right, up, forward)
    cam.orbit(80.0, -30.0)
    _, right, up, forward = cam.basis()
    assert_orthonormal(right, up, forward)


def test_camera_rejects_bad_clip():
    with pytest.raises(ValueError):
        Camera(near=0.0)
    with pytest.raises(ValueError):
        Camera(near=1.0, far=0.5)


@needs_chole
def test_read_case_render():
    cfg = read_case("chole_tet_med")
    render = cfg["render"]
    assert render["backend"] == "slang" and render["camera"]["up"] == [0.0, 1.0, 0.0]
    assert "${DATA}" not in render["environment"]["path"] and render["environment"]["path"].endswith("abdomen.hdr")
    liver = cfg["assets"]['tissue']["parts"]["liver"]["visual"]
    assert liver["set"] == str(data_dir() / "textures" / "liver")  # the authored base, damage, coag and blood maps
    assert cfg["assets"]["cavity"]["type"] == "static_mesh"


def test_view_projection_matches_project():
    """The shaders' world -> clip matrix (X1) is the same perspective as Camera.project: pixels and depth agree."""
    cam = Camera(position=(0.3, 1.2, -4.0), pitch=-10.0, yaw=20.0, fov=50.0)
    cam.near, cam.far = 0.05, 50.0
    pts = np.random.default_rng(0).uniform(-1.0, 1.0, (256, 3)).astype(np.float32) + cam.target
    pts = pts[cam.project(pts, 1.0, 1.0)[:, 2] > cam.near]  # in front of the camera
    w, h = 640.0, 360.0
    clip = np.c_[pts, np.ones(len(pts), np.float32)] @ cam.view_projection(w, h).T
    px = (clip[:, 0] / clip[:, 3] + 1.0) * 0.5 * w
    py = (1.0 - clip[:, 1] / clip[:, 3]) * 0.5 * h
    ref = cam.project(pts, w, h)
    np.testing.assert_allclose(px, ref[:, 0], rtol=1e-5, atol=1e-2)  # float32; a point near the camera plane projects far off-screen
    np.testing.assert_allclose(py, ref[:, 1], rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(clip[:, 3], ref[:, 2], rtol=1e-5)
    z = clip[:, 2] / clip[:, 3]
    assert ((z > 0.0) & (z < 1.0)).all()  # between near and far: depth in [0, 1]
