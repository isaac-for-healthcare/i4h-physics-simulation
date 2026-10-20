# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static mesh loader on the cavity: welding, closed-manifold check, normals, tangents, no orientation flip."""

import os
import shutil

import numpy as np
import pytest
from i4h_isaaclab.composition import data_dir
from i4h_physics import Scene
from i4h_physics.loaders import add_static_mesh
from i4h_physics.loaders.static_mesh import smooth_tangents, weld_map

DATA = data_dir() / "choly" / "cavity"
pytestmark = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not DATA.is_dir(), reason=f"cavity data not found at {DATA}")
TRANSLATION = (0.8, -3.3, -0.4)


def load(path=DATA, translation=TRANSLATION):
    scene = Scene("cuda:0", 1, (0.0, -9.81, 0.0))
    mesh = add_static_mesh(scene, "cavity", str(path), translation=translation)
    assert scene.static_meshes == [mesh]
    return mesh


def test_cavity_loads():
    m = load()
    assert m.points.shape == (12427, 3) and m.indices.shape == (18868, 3) and m.uvs.shape == (12427, 2)
    assert len(np.unique(weld_map(m.points))) == 9436
    assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-5)
    assert m.tangents.shape == (12427, 4) and set(np.unique(m.tangents[:, 3]).tolist()) <= {-1.0, 1.0}
    assert np.allclose(np.linalg.norm(m.tangents[:, :3], axis=1), 1.0, atol=1e-5)
    assert np.abs(np.sum(m.tangents[:, :3] * m.normals, axis=1)).max() < 1e-5  # orthogonal to the normal
    # translation applied: the chole camera sits inside the cavity's bounding box
    camera = np.float32([0.469625, 1.20226, -2.45614])
    assert (m.points.min(axis=0) < camera).all() and (camera < m.points.max(axis=0)).all()
    raw = load(translation=(0.0, 0.0, 0.0))
    assert np.allclose(m.points - raw.points, np.float32(TRANSLATION), atol=1e-6)
    # inward-facing winding kept as authored (negative signed volume)
    p0, p1, p2 = (m.points[m.indices[:, k]].astype(np.float64) for k in range(3))
    volume = np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0
    assert volume < 0.0


def test_open_mesh_rejected(tmp_path):
    root = tmp_path / "cavity"
    shutil.copytree(DATA, root)
    tris = np.loadtxt(root / "cavity.tris", dtype=np.int64).reshape(-1, 3)
    np.savetxt(root / "cavity.tris", tris[1:], fmt="%d")
    with pytest.raises(ValueError, match="not closed"):
        load(root)


def test_tangents_keep_uv_seams():
    """Two quads in the z=0 plane share the x=0 edge with duplicated vertices; the left chart mirrors u.
    Per-vertex accumulation keeps opposite tangents and handedness on each side (welding would average them)."""
    points = np.float32([[-1, -1, 0], [0, -1, 0], [0, 1, 0], [-1, 1, 0], [0, -1, 0], [1, -1, 0], [1, 1, 0], [0, 1, 0]])
    indices = np.int32([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    uvs = np.float32([[1, 0], [0, 0], [0, 1], [1, 1], [0, 0], [1, 0], [1, 1], [0, 1]])  # left: u = -x, right: u = x
    normals = np.tile(np.float32([0, 0, 1]), (8, 1))
    t = smooth_tangents(points, indices, uvs, normals)
    assert np.allclose(t[:4, :3], [-1, 0, 0], atol=1e-6) and np.allclose(t[4:, :3], [1, 0, 0], atol=1e-6)
    assert (t[:4, 3] == -1).all() and (t[4:, 3] == 1).all()
