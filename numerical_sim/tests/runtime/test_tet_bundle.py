# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tet bundle loader on the choly-mid data: layout, connectors, pins, ${DATA} resolution."""

import os

import numpy as np
import pytest
from i4h_isaaclab.composition import data_dir, resolve_data_path
from i4h_physics import Scene
from i4h_surgical.loaders.tet_bundle import add_tet_bundle

DATA = data_dir() / "choly-mid"
pytestmark = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not DATA.is_dir(), reason=f"chole data not found at {DATA}")

PARTS = {
    "liver": {"spring": 1.0, "volume": 1.0},
    "fat": {"spring": 0.1, "volume": 0.1},
    "gallbladder": {"spring": 0.8, "volume": 0.1},
}
CONNECTORS = [
    dict(path="fat-liver.connector", particle_part="fat", triangle_part="liver", stiffness=0.05, weight_ratio=0.99),
    dict(path="gallbladder-fat.connector", particle_part="gallbladder", triangle_part="fat", stiffness=0.05, weight_ratio=0.8),
]
PIN = {"center": [0.5, 1.5, -5.0], "radius": 1.0}


def load(envs=1, connectors=CONNECTORS):
    scene = Scene("cuda:0", envs, (0.0, -9.81, 0.0))
    body = add_tet_bundle(scene, 'tissue', str(DATA), PARTS, translation=(0.0, -3.0, 0.0),
                          connectors=connectors, pin=PIN, mass=0.1, radius=0.01)
    return scene, body


def test_layout():
    scene, body = load()
    assert body.count == 5612
    ranges = {n: (p.particle_start, p.particle_start + p.particle_count) for n, p in body.parts.items()}
    assert ranges == {"liver": (0, 1986), "fat": (1986, 4674), "gallbladder": (4674, 5612)}
    tet = body.tet
    assert tet.tets.shape == (1, 18823, 4) and tet.edges.shape == (1, 30038, 2)
    assert tet.active0.all() and tet.edge_active0.all()
    assert (tet.rest_volume > 0.0).all() and (tet.edge_rest > 0.0).all()
    assert np.array_equal(np.bincount(tet.part), [6653, 9102, 3068])
    assert np.array_equal(tet.spring, np.float32([1.0, 0.1, 0.8])) and np.array_equal(tet.volume, np.float32([1.0, 0.1, 0.1]))
    q0 = body.rest.q0[0]
    assert q0[:, 1].min() > 0.5 and q0[:, 1].max() < 1.8  # translated by -3 in y


def test_surface_and_uvs():
    scene, body = load()
    tet = body.tet
    assert tet.tris.shape == (1, 11212, 3) and np.array_equal(np.bincount(tet.tri_part), [3968, 5372, 1872])
    for name, p in body.parts.items():
        tris = tet.tris[0, p.tri_start : p.tri_start + p.tri_count]
        assert p.tri_count == np.count_nonzero(tet.tri_part == p.index)
        assert tris.min() >= p.particle_start and tris.max() < p.particle_start + p.particle_count
        assert p.appearance.visual is None
    assert tet.uvs.shape == (1, 5612, 2) and tet.uvs.min() >= 0.0 and tet.uvs.max() <= 1.0
    visual = {"diffuse": "${DATA}/textures/liver/diffuse-base.png"}
    parts = {**PARTS, "liver": {**PARTS["liver"], "visual": visual}}
    body = add_tet_bundle(Scene("cuda:0", 1, (0.0, -9.81, 0.0)), 'tissue', str(DATA), parts)
    assert body.parts["liver"].appearance.visual == visual and body.parts["fat"].appearance.visual is None


def test_connectors_and_pins():
    scene, body = load()
    c = body.connectors
    assert c.particle.shape == (1, 1479)
    assert np.array_equal(np.bincount(c.group), [709, 770])
    assert (c.particle >= 0).all()
    fat, liver, gb = body.parts["fat"], body.parts["liver"], body.parts["gallbladder"]
    p, t = c.particle[0], c.tri[0]
    g0, g1 = c.group == 0, c.group == 1
    assert np.all((p[g0] >= fat.particle_start) & (p[g0] < fat.particle_start + fat.particle_count))
    assert np.all((t[g0] >= liver.particle_start) & (t[g0] < liver.particle_start + liver.particle_count))
    assert np.all((p[g1] >= gb.particle_start) & (p[g1] < gb.particle_start + gb.particle_count))
    assert np.all((t[g1] >= fat.particle_start) & (t[g1] < fat.particle_start + fat.particle_count))
    assert np.array_equal(c.weight_ratio, np.float32([0.99, 0.8])) and np.array_equal(c.stiffness, np.float32([0.05, 0.05]))
    # The authored fat-liver rest distances equal the rest geometry; gallbladder-fat carry slack.
    q0 = body.rest.q0[0]
    tri_pos = np.einsum("ijk,ij->ik", q0[t[g0]], c.bar[0][g0])
    assert np.abs(np.linalg.norm(q0[p[g0]] - tri_pos, axis=1) - c.rest[0][g0]).max() < 1e-4

    pins = body.pin.index[0]
    assert pins.shape == (391,) and (pins >= 0).all() and (pins < 1986).all()
    assert np.allclose(body.pin.rest[0], q0[pins])


def test_swapped_connector_roles_fail():
    swapped = [dict(CONNECTORS[0]), dict(CONNECTORS[1], particle_part="fat", triangle_part="gallbladder")]
    with pytest.raises(ValueError, match=r"gallbladder-fat\.connector.*check particle_part/triangle_part"):
        load(connectors=swapped)


def test_variants_cycle_over_envs():
    """Two bundle roots (the same data twice) give two variants stored once and cycled over envs."""
    scene = Scene("cuda:0", 3, (0.0, -9.81, 0.0))
    body = add_tet_bundle(scene, 'tissue', [str(DATA), str(DATA)], PARTS, translation=(0.0, -3.0, 0.0),
                          connectors=CONNECTORS, pin=PIN)
    assert body.variants == 2 and body.variant_of_env.tolist() == [0, 1, 0]
    assert body.tet.tets.shape == (2, 18823, 4) and body.connectors.particle.shape == (2, 1479)
    assert body.pin.index.shape == (2, 391)
    assert body.rest.q0.shape == (3, 5612, 3) and np.array_equal(body.rest.q0[0], body.rest.q0[2])

    single = load(envs=3)[1]
    assert single.variants == 1 and single.tet.tets.shape == (1, 18823, 4)

    # variants counts what was loaded, not what the envs happen to use
    one_env = Scene("cuda:0", 1, (0.0, -9.81, 0.0))
    body = add_tet_bundle(one_env, 'tissue', [str(DATA), str(DATA)], PARTS, translation=(0.0, -3.0, 0.0))
    assert body.variants == 2 and body.variant_of_env.tolist() == [0]


def test_fractional_connector_index_fails(tmp_path):
    from i4h_physics.loaders.tet_bundle import read_connectors

    path = tmp_path / "bad.connector"
    path.write_text("1 0.01 2 3 4 0.2 0.3 0.5\n2.5 0.01 2 3 4 0.2 0.3 0.5\n")
    with pytest.raises(ValueError, match="row 1: .*must be integers"):
        read_connectors(path, "a", 10, "b", 10)


def test_data_path(monkeypatch):
    assert resolve_data_path("${DATA}/choly-mid") == str(DATA)
    monkeypatch.setenv("I4H_DATA", "/tmp/elsewhere")
    assert resolve_data_path("${DATA}/x") == "/tmp/elsewhere/x"
    assert resolve_data_path(["${DATA}/a", "${DATA}/b"]) == ["/tmp/elsewhere/a", "/tmp/elsewhere/b"]
