# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M4 cutting: the probe removes block tets while it fires, in its own env only; reset restores them."""

import numpy as np
from helpers import PUSH, find, live_surface_is_consistent, push_actions, run
from i4h_isaaclab.composition import load_case
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.systems.instruments.cutting import Cutting
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.surface import Surface


def test_probe_cut_is_env_local():
    sim = load_case("two_worlds")
    scene, body = sim.scene, sim.scene.bodies[0]
    control, surface, tet, cutting = find(sim, InstrumentControl), find(sim, Surface), find(sim, Tet), find(sim, Cutting)
    assert cutting.sphere_count == 1 and cutting.k == 0
    run(sim, 20, push_actions(control, scene.envs))  # the probe sits inside the bottom face of env 0's block
    graph = sim.graph
    active_before, tris_before = tet.active.numpy().copy(), surface.tris.numpy().copy()
    assert surface.deleted.numpy().tolist() == [0, 0]

    actions = push_actions(control, scene.envs)
    actions[0, 0, 9] = 1.0  # fire in env 0 only
    run(sim, 1, actions)
    deleted = surface.deleted.numpy()
    assert deleted[0] > 0 and deleted[1] == 0, deleted
    assert np.array_equal(tet.active.numpy()[1], active_before[1]) and np.array_equal(surface.tris.numpy()[1], tris_before[1])
    assert tet.active.numpy()[0].sum() == active_before[0].sum() - deleted[0]
    assert surface.overflow.numpy().tolist() == [0, 0] and surface.dirty.numpy().tolist() == [0, 0]
    live_surface_is_consistent(sim, 0)
    # every removed tet was within the probe's reach
    v = body.variant_of_env[0]
    removed = np.flatnonzero((active_before[0] != 0) & (tet.active.numpy()[0] == 0))
    q = scene.view(scene.q, body).numpy()[0]
    centre = control.sphere_q.numpy()[0, 0]
    assert np.linalg.norm(q[body.tet.tets[v][removed]].mean(axis=1) - centre, axis=1).max() < 0.015 + 0.02

    # fire held: the advancing probe keeps cutting; released: the count stays; the graph is untouched
    actions[0, 0, 2] = PUSH[2] + 0.02
    run(sim, 10, actions)
    more = surface.deleted.numpy()
    assert more[0] > deleted[0] and more[1] == 0
    actions[0, 0, 9] = 0.0
    run(sim, 10, actions)
    assert surface.deleted.numpy().tolist() == more.tolist() and sim.graph is graph
    live_surface_is_consistent(sim, 0)
    assert np.isfinite(scene.q.numpy()).all()

    # masked reset brings env 0's topology back; env 1 was never touched
    sim.reset([0])
    assert np.array_equal(tet.active.numpy(), active_before) and np.array_equal(surface.tris.numpy(), tris_before)
    assert surface.deleted.numpy().tolist() == [0, 0]
    assert np.all(scene.view(scene.flags, body).numpy()[0] == body.rest.flags[0])


def test_cut_kernels_keep_inputs_and_outputs_disjoint():
    sim = load_case("two_worlds", debug=True)
    sim.cuda_graph = False
    control = find(sim, InstrumentControl)
    actions = push_actions(control, sim.scene.envs)
    actions[:, 0, 9] = 1.0
    run(sim, 3, actions)
    assert find(sim, Surface).deleted.numpy()[0] > 0 and np.isfinite(sim.scene.q.numpy()).all()


def test_removed_triangles_stop_colliding_before_the_rebuild():
    """A tet removed in a substep must not push tissue through its stale triangle slot until the surface
    system rewrites the slot at step end (review finding: shared vertices kept dead faces collidable)."""
    from helpers import kill

    sim = load_case("two_worlds")
    scene, body = sim.scene, sim.scene.bodies[0]
    control, surface, tet = find(sim, InstrumentControl), find(sim, Surface), find(sim, Tet)
    contact = next(s for s in sim.systems if type(s).__name__ == "InstrumentContact")
    run(sim, 20, push_actions(control, scene.envs))  # the probe presses into env 0's block: live records
    sdt = sim.dt / sim.substeps
    for t in np.flatnonzero(tet.active.numpy()[0]):  # remove every tet of env 0 without rebuilding the surface
        kill(sim, surface, 0, int(t))
    assert (surface.tris.numpy()[0][:, 0] != surface.tris.numpy()[0][:, 1]).any()  # stale live-looking slots
    before = scene.view(scene.q, body).numpy().copy()
    contact.substep_start(scene, sdt, 0, sim.substeps)
    contact.iteration(scene, sdt, 0, sim.iterations)
    assert np.array_equal(scene.view(scene.q, body).numpy(), before), "a dead triangle produced a contact correction"
