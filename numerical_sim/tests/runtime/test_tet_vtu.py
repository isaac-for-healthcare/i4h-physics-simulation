# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")

from i4h_physics import Scene  # noqa: E402 - optional dependency guard must run first
from i4h_physics.loaders.tet_vtu import (  # noqa: E402 - optional dependency guard must run first
    add_tet_vtu,
    read_tet_vtu,
)

POINTS = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]])


def write_mesh(path, points=POINTS, tets=((0, 1, 2, 3),), kind="tetra"):
    meshio.write(path, meshio.Mesh(points, [(kind, np.asarray(tets))]))
    return path


def test_topology_mass_transform_and_pins(tmp_path):
    path = write_mesh(tmp_path / "one.vtu")
    m = read_tet_vtu(path, density=2, scale=2, translation=(1, 2, 3))
    assert len(m.tris) == 4 and len(m.edges) == 6
    np.testing.assert_allclose(1 / m.inv_mass, 2 * (8 / 6) / 4)
    for face in m.tris:
        p = m.q0[face]
        assert np.dot(np.cross(p[1]-p[0], p[2]-p[0]), p.mean(axis=0)-m.q0.mean(axis=0)) > 0
    m = read_tet_vtu(path, pins=[{"type": "sphere", "center": [0, 0, -1], "radius": 0.01}])
    assert np.array_equal(m.inv_mass == 0, [False, False, False, True])
    scene = Scene("cpu", 2, gravity=(0, 0, 0))
    body = add_tet_vtu(scene, "wall", path)
    assert np.all(body.tet.tri_tet >= 0) and np.all(body.tet.support0 == 1)
    assert body.rest.q0.shape == (2, 4, 3)


@pytest.mark.parametrize("tets,message", [
    ([[0, 1, 2, 4]], "outside"), ([[0, 1, 2, 2]], "repeated"),
    ([[0, 1, 2, 3], [3, 2, 1, 0]], "duplicate"),
])
def test_invalid_connectivity(tmp_path, tets, message):
    with pytest.raises(ValueError, match=message):
        read_tet_vtu(write_mesh(tmp_path / "bad.vtu", tets=tets))


def test_invalid_coordinates_volume_pins_and_order(tmp_path):
    path = tmp_path / "bad.vtu"
    points = POINTS.copy()
    points[3] = [0.2, 0.2, 0]
    with pytest.raises(ValueError, match="non-positive"):
        read_tet_vtu(write_mesh(path, points=points))
    points[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        read_tet_vtu(write_mesh(path, points=points))
    write_mesh(path)
    with pytest.raises(ValueError, match="matched no vertices"):
        read_tet_vtu(path, pins=[{"type": "box", "minimum": [5, 5, 5], "maximum": [6, 6, 6]}])
    with pytest.raises(ValueError, match="positive"):
        read_tet_vtu(path, density=0)
    write_mesh(path, points=np.zeros((10, 3)), tets=[list(range(10))], kind="tetra10")
    with pytest.raises(ValueError, match="higher-order"):
        read_tet_vtu(path)


def test_nonmanifold_vertex_and_unreferenced_vertex(tmp_path):
    # Two otherwise closed tetrahedral boundaries meet at a single vertex.
    q = np.concatenate([POINTS, -POINTS[1:]])
    path = write_mesh(tmp_path / "pinch.vtu", points=q, tets=[[0, 1, 2, 3], [0, 4, 5, 6]])
    with pytest.raises(ValueError, match="boundary vertex"):
        read_tet_vtu(path)
    path = write_mesh(tmp_path / "unused.vtu", points=np.concatenate([POINTS, [[5, 5, 5]]]))
    with pytest.raises(ValueError, match="no incident volume"):
        read_tet_vtu(path)
