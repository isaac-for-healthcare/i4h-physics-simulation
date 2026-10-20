# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pytest.importorskip("meshio")
from i4h_physics import Scene, Stepper  # noqa: E402 - optional dependency guard must run first
from i4h_physics.loaders.tet_vtu import add_tet_vtu  # noqa: E402 - optional dependency guard must run first
from i4h_physics.scene import ACTIVE_INT, Part, ParticleRest  # noqa: E402 - optional dependency guard must run first
from i4h_physics.systems.contact.tet_containment import (  # noqa: E402 - optional dependency guard must run first
    TetContainment,
)
from i4h_physics.systems.mechanics.tet import Tet  # noqa: E402 - optional dependency guard must run first
from i4h_surgical.systems.mechanics.surface import Surface  # noqa: E402 - optional dependency guard must run first
from test_tet_vtu import write_mesh  # noqa: E402 - optional dependency guard must run first


def contact(tmp_path, graph=False, pins=(), **kw):
    scene = Scene("cuda:0", 2, gravity=(0, 0, 0))
    rest = ParticleRest(np.tile(np.float32([[[0.25, 0.25, 0.05]]]), (2, 1, 1)),
                        np.ones((2, 1), np.float32), np.full((2, 1), 0.1, np.float32),
                        np.full((2, 1), ACTIVE_INT, np.int32))
    rod = scene.add_body("rod", rest, {"rod": Part(0, 0, 1)})
    wall = add_tet_vtu(scene, "wall", write_mesh(tmp_path / "wall.vtu"), pins=pins)
    scene.allocate()
    tet = Tet(scene, body="wall")
    surf = Surface(scene, tet=tet)
    cont = TetContainment(scene, body="rod", surface=surf, sign=1, max_distance=0.2, **kw)
    sim = Stepper.create(scene, [surf, cont], cuda_graph=graph, iterations=1)
    sim.prepare()
    return sim, rod, wall, surf, cont


@pytest.mark.parametrize("graph", [False, True])
def test_half_split_isolation_reset_disable(tmp_path, graph):
    sim, rod, wall, surf, cont = contact(tmp_path, graph)
    scene = sim.scene
    q0 = scene.field("q", wall).numpy().copy()
    sim.step()
    q = scene.field("q", rod).numpy()
    np.testing.assert_allclose(q[:, 0, 2], -0.025, atol=1e-6)
    w = scene.field("q", wall).numpy()
    np.testing.assert_allclose(0.5*w[:, 0, 2]+0.25*w[:, 1, 2]+0.25*w[:, 2, 2], 0.075, atol=1e-6)
    assert np.all(cont.contact_count.numpy() == 1)
    np.testing.assert_allclose(cont.contact_depth.numpy(), 0.15, atol=1e-6)
    assert not cont.wall_delta.numpy().any()
    keep = w[1].copy()
    sim.reset([0])
    np.testing.assert_array_equal(scene.field("q", wall).numpy()[1], keep)
    np.testing.assert_array_equal(scene.field("q", wall).numpy()[0], q0[0])
    assert cont.contact_count.numpy()[0] == 0
    cont.enabled = False
    sim.prepare()
    sim.step()
    np.testing.assert_array_equal(scene.field("q", wall).numpy()[0], q0[0])
    cont.enabled = True
    sim.prepare()
    sim.step()
    assert cont.contact_count.numpy()[0] > 0


@pytest.mark.parametrize("dead", ["owner", "particle", "support"])
def test_dead_face_rejection(tmp_path, dead):
    sim, rod, wall, surf, cont = contact(tmp_path)
    if dead == "owner":
        a = surf.active.numpy()
        a[0] = 0
        surf.active.assign(a)
    elif dead == "particle":
        a = sim.scene.field("flags", wall).numpy()
        a[0] = 0
        sim.scene.field("flags", wall).assign(a)
    else:
        a = surf.support.numpy()
        a[:wall.count] = 0
        surf.support.assign(a)
    before = sim.scene.field("q", rod).numpy().copy()
    sim.step()
    np.testing.assert_array_equal(sim.scene.field("q", rod).numpy()[0], before[0])
    assert np.array_equal(cont.contact_count.numpy(), [0, 1])


@pytest.mark.parametrize("two_way", [False, True])
def test_pinned_wall_and_side_preservation(tmp_path, two_way):
    pins = [{"type": "box", "minimum": [-2, -2, -2], "maximum": [2, 2, 2]}]
    sim, rod, wall, surf, cont = contact(tmp_path, pins=pins, two_way=two_way, two_sided=True)
    scene = sim.scene
    initial = scene.field("q", wall).numpy().copy()
    cont.sides.fill_(1.0)
    q = scene.field("q", rod).numpy()
    q[:, 0, 2] = -0.05
    scene.field("q", rod).assign(q)
    cont.iteration(scene, sim.dt, 0, 1)
    np.testing.assert_allclose(scene.field("q", rod).numpy()[:, 0, 2], 0.1, atol=1e-6)
    np.testing.assert_array_equal(scene.field("q", wall).numpy(), initial)


def test_default_tet_sweep_and_rest_deformation(tmp_path):
    sim, rod, wall, surf, cont = contact(tmp_path)
    tet = surf.tet
    assert tet.iterations == 1
    initial = sim.scene.field("q", wall).numpy().copy()
    tet.iteration(sim.scene, sim.dt, 0, 1)
    np.testing.assert_allclose(sim.scene.field("q", wall).numpy(), initial, atol=1e-7)
    displaced = initial.copy()
    displaced[:, 3, 2] -= 0.2
    sim.scene.field("q", wall).assign(displaced)
    tet.iterations = 8
    tet.iteration(sim.scene, sim.dt, 0, 1)
    assert np.linalg.norm(sim.scene.field("q", wall).numpy()-initial) < np.linalg.norm(displaced-initial)
    tet.iterations = 0
    with pytest.raises(ValueError, match="positive integer"):
        tet.prepare(sim.scene)


def test_eager_captured_agreement(tmp_path):
    eager = contact(tmp_path, graph=False)
    captured = contact(tmp_path, graph=True)
    for sim, *_ in (eager, captured):
        for _ in range(5):
            sim.step()
    np.testing.assert_allclose(eager[0].scene.q.numpy(), captured[0].scene.q.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(eager[-1].contact_count.numpy(), captured[-1].contact_count.numpy())
