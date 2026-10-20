# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M4 surface bookkeeping: loader face and edge tables, tet deactivation, per-env re-emission, masked reset."""

import numpy as np
import pytest
from helpers import CHOLE, find, kill, needs_chole
from i4h_isaaclab.composition import load_case
from i4h_physics import Scene
from i4h_physics.scene import ACTIVE_INT
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.loaders.tet_bundle import add_tet_bundle
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.mechanics.surface import Surface


def check_tables(m, v):
    """Live tets know their six edges and their neighbours both ways; authored triangles have owners;
    support counts add up to the incidences."""
    live = np.flatnonzero(m.active0[v])
    assert (m.tet_edges[v][live] >= 0).all()
    nb = m.face_neighbor[v]
    for t in live[: min(len(live), 300)]:
        for f in range(4):
            n = nb[t, f]
            if n >= 0:
                assert m.active0[v, n] and t in nb[n], (t, f, n)
    authored = m.tris[v][:, 0] != m.tris[v][:, 1]
    assert (m.tri_tet[v][authored] >= 0).all(), "an authored triangle matched no tet face"
    assert (m.tri_tet[v][~authored] == -1).all()
    owned = m.face_tri[v] >= 0
    assert owned.sum() == authored.sum() and np.array_equal(np.sort(m.face_tri[v][owned]), np.flatnonzero(authored))
    assert m.support0[v].sum() == 4 * len(live)
    assert m.edge_support0[v].sum() == 6 * len(live)
    return authored


def test_grid_tables():
    """4x4x4 cells: 320 tets, 192 authored skin triangles that are exactly its boundary faces."""
    body = load_case("two_worlds").scene.bodies[0]
    m, p = body.tet, body.parts["block"]
    assert m.heat.tolist() == [1] and m.deletable.tolist() == [1]
    assert p.tri_count == 2 * 192 and m.cut_start.tolist() == [192] and m.cut_capacity.tolist() == [192]
    for v, cells_per_side in ((0, 4), (1, 3)):
        authored = check_tables(m, v)
        assert authored.sum() == 6 * 2 * cells_per_side**2
        live = np.flatnonzero(m.active0[v])
        assert (m.face_neighbor[v][live] < 0).sum() == authored.sum()  # boundary faces are the authored skin


@needs_chole
def test_bundle_tables():
    scene = Scene("cuda:0", 1, (0.0, -9.81, 0.0))
    parts = {"liver": {"spring": 1.0, "volume": 1.0}, "fat": {"spring": 0.1, "volume": 0.1, "heat": True, "deletable": True},
             "gallbladder": {"spring": 0.8, "volume": 0.1}}
    body = add_tet_bundle(scene, 'tissue', str(CHOLE), parts, translation=(0.0, -3.0, 0.0))
    m = body.tet
    assert m.heat.tolist() == [0, 1, 0] and m.deletable.tolist() == [0, 1, 0]
    assert m.cut_capacity.tolist() == [0, 5372, 0] and m.cut_start[1] == 3968 + 5372
    assert m.tris.shape == (1, 11212 + 5372, 3) and [p.tri_count for p in body.parts.values()] == [3968, 2 * 5372, 1872]
    check_tables(m, 0)
    with pytest.raises(ValueError, match="cut_faces needs deletable"):
        add_tet_bundle(Scene("cuda:0", 1), "t", str(CHOLE), {**parts, "liver": {"spring": 1.0, "volume": 1.0, "cut_faces": 10}})
    small = add_tet_bundle(Scene("cuda:0", 1), "t", str(CHOLE), {**parts, "fat": {**parts["fat"], "cut_faces": 100}})
    assert small.tet.cut_capacity.tolist() == [0, 100, 0]


def test_deactivate_reemits_per_env():
    sim = load_case("two_worlds")
    scene, body = sim.scene, sim.scene.bodies[0]
    surface, tet, _contact = find(sim, Surface), find(sim, Tet), find(sim, InstrumentContact)
    assert surface.active is tet.active  # the live masks are shared with the tet system
    sim.step()
    graph = sim.graph
    m, v = body.tet, int(body.variant_of_env[0])
    tris_before, tri_tet_before = surface.tris.numpy().copy(), surface.tri_tet.numpy().copy()
    support_before = surface.support.numpy().reshape(scene.envs, -1).copy()
    flags_before = scene.view(scene.flags, body).numpy().copy()
    assert np.array_equal(tris_before, m.tris[body.variant_of_env]) and surface.deleted.numpy().tolist() == [0, 0]

    t = int(np.flatnonzero(m.active0[v])[7])
    kill(sim, surface, 0, t)
    assert tet.active.numpy()[0, t] == 0 and surface.deleted.numpy().tolist() == [1, 0] and surface.dirty.numpy().tolist() == [1, 0]
    kill(sim, surface, 0, t)  # a second removal of the same tet is a no-op
    assert surface.deleted.numpy().tolist() == [1, 0]
    surface.step_end(scene, sim.dt)
    tris, tri_tet = surface.tris.numpy(), surface.tri_tet.numpy()
    assert np.array_equal(tris[1], tris_before[1]) and np.array_equal(tri_tet[1], tri_tet_before[1])
    assert surface.dirty.numpy().tolist() == [0, 0] and surface.overflow.numpy().tolist() == [0, 0]
    # the dead tet's authored faces are degenerate slots; its interior faces are exposed on the live neighbours
    dead = np.flatnonzero(m.tri_tet[v] == t)
    assert (tris[0, dead][:, 0] == tris[0, dead][:, 1]).all() and (tri_tet[0, dead] == -1).all()
    neighbours = [int(n) for n in m.face_neighbor[v, t] if n >= 0]
    cut_start, exposed = int(m.cut_start[0]), int(surface.cut_count.numpy()[0, 0])
    assert exposed == len(neighbours) and surface.cut_count.numpy()[1, 0] == 0
    for f in range(cut_start, cut_start + exposed):
        owner = int(tri_tet[0, f])
        assert owner in neighbours and set(tris[0, f]) <= set(m.tets[v, owner]) and set(tris[0, f]) <= set(m.tets[v, t])
    assert (tris[0, cut_start + exposed :][:, 0] == tris[0, cut_start + exposed :][:, 1]).all()
    # every live triangle has a live owner and live vertices; supports match the live incidences
    live = tri_tet[0] >= 0
    assert tet.active.numpy()[0][tri_tet[0][live]].all()
    flags = scene.view(scene.flags, body).numpy()
    assert (flags[0][tris[0][live].reshape(-1)] & ACTIVE_INT).all()
    support = surface.support.numpy().reshape(scene.envs, -1)
    assert support[0].sum() == 4 * tet.active.numpy()[0].sum() and np.array_equal(support[1], support_before[1])
    assert np.array_equal(flags, flags_before)  # every particle of t still has other tets
    edge_support = surface.edge_support.numpy()
    assert np.array_equal(tet.edge_active.numpy()[0] != 0, edge_support[0] > 0)

    # remove every tet around the particle with the fewest: it loses ACTIVE, its edges go inactive
    s0 = m.support0[v]
    pi = int(np.flatnonzero(s0 > 0)[np.argmin(s0[s0 > 0])])
    around = np.flatnonzero((m.tets[v] == pi).any(axis=1) & (m.active0[v] != 0))
    for t2 in around:
        kill(sim, surface, 0, int(t2))
    surface.step_end(scene, sim.dt)
    flags = scene.view(scene.flags, body).numpy()
    assert flags[0, pi] & ACTIVE_INT == 0 and (flags[1] == flags_before[1]).all()
    others = np.setdiff1d(m.tets[v][around].reshape(-1), [pi])
    assert (flags[0][others] & ACTIVE_INT).all()  # their other tets keep them alive
    tri_tet = surface.tri_tet.numpy()
    assert not (surface.tris.numpy()[0][tri_tet[0] >= 0] == pi).any()  # no live triangle references the dead particle
    edge_support = surface.edge_support.numpy()
    assert np.array_equal(tet.edge_active.numpy()[0] != 0, edge_support[0] > 0) and (edge_support[0] == 0).any()
    assert surface.deleted.numpy().tolist() == [1 + len(around), 0]

    # the run continues on the captured graph and stays finite; contact reads the live buffer
    for _ in range(5):
        sim.step()
    assert sim.graph is graph and np.isfinite(scene.q.numpy()).all()

    # masked reset restores env 0 bitwise and leaves env 1 alone
    sim.reset([0])
    assert np.array_equal(surface.tris.numpy(), tris_before) and np.array_equal(surface.tri_tet.numpy(), tri_tet_before)
    assert np.array_equal(surface.support.numpy().reshape(scene.envs, -1), support_before)
    assert np.array_equal(scene.view(scene.flags, body).numpy(), flags_before)
    assert surface.deleted.numpy().tolist() == [0, 0] and tet.active.numpy()[0].sum() == m.active0[v].sum()
    assert np.array_equal(surface.edge_support.numpy()[0], m.edge_support0[v])
    assert not surface.appearance.visual.numpy().any()


def test_surface_needs_tet():
    scene = Scene("cuda:0", 1)
    from i4h_physics.loaders import add_tet_grid

    add_tet_grid(scene, "g", (2, 2, 2), 0.02)
    scene.allocate()
    with pytest.raises(TypeError, match="surface.tet"):
        Surface(scene, tet="g")
