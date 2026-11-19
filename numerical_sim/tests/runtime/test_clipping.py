# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M4 centrelines and clipping on the chole data: loader, tracker, attach on the fire edge, pinch that holds, masked reset."""

import os

import numpy as np
import pytest
import warp as wp
from helpers import CHOLE, find
from i4h_isaaclab.composition import load_case
from i4h_physics import Scene
from i4h_surgical.loaders.tet_bundle import add_tet_bundle
from i4h_surgical.systems.instruments.clipping import Clipping
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.centreline import Centreline

pytestmark = pytest.mark.skipif(not os.environ.get("I4H_DATA") and (not CHOLE.is_dir()), reason=f"chole data not found at {CHOLE}")

PARTS = {"liver": {"spring": 1.0, "volume": 1.0}, "fat": {"spring": 0.1, "volume": 0.1}, "gallbladder": {"spring": 0.8, "volume": 0.1}}
LINES = [{"name": "cystic_artery", "path": "centrelines/cystic_artery.cntr", "part": "gallbladder"},
         {"name": "cystic_duct", "path": "centrelines/cystic_duct.cntr", "part": "gallbladder"}]


def ring_radius(scene, body, centreline, env, node):
    """Mean distance of a node's live clamp particles from their centroid (in float64: a float32 mean leaves about one
    ulp of the coordinates, ~5e-7 here, on a ring collapsed onto its centroid)."""
    c = body.centrelines
    v = body.variant_of_env[env]
    ids = c.clamp[v, c.clamp_start[v, node] : c.clamp_start[v, node] + c.clamp_count[v, node]]
    q = scene.view(scene.q, body).numpy()[env][ids].astype(np.float64)
    return float(np.linalg.norm(q - q.mean(axis=0), axis=1).mean())


def test_centreline_loader():
    scene = Scene("cuda:0", 1, (0.0, -9.81, 0.0))
    body = add_tet_bundle(scene, 'tissue', str(CHOLE), PARTS, translation=(0.0, -3.0, 0.0), centrelines=LINES)
    c = body.centrelines
    assert c.names == ["cystic_artery", "cystic_duct"] and c.line.tolist() == [0] * 14 + [1] * 11
    assert c.clamp_start.shape == (1, 25) and (c.clamp_count[0] > 0).all() and c.clamp.shape[1] == c.clamp_count[0].sum()
    gb = body.parts["gallbladder"]
    assert c.clamp.min() >= gb.particle_start and c.clamp.max() < gb.particle_start + gb.particle_count
    q0 = body.rest.q0[0]
    for k in (0, 13, 14, 24):
        ids = c.clamp[0, c.clamp_start[0, k] : c.clamp_start[0, k] + c.clamp_count[0, k]]
        assert np.allclose(c.rest[0, k], q0[ids].mean(axis=0), atol=1e-6)
    with pytest.raises(ValueError, match="unknown part"):
        add_tet_bundle(Scene("cuda:0", 1), "t", str(CHOLE), PARTS, centrelines=[{**LINES[0], "part": "spleen"}])


def test_tracker_and_clip():
    sim = load_case("chole_tet_med", envs=2)
    scene, body = sim.scene, sim.scene.body('tissue')
    control, centreline, clipping = find(sim, InstrumentControl), find(sim, Centreline), find(sim, Clipping)
    ka = clipping.k  # the applier's instrument index
    assert control.names[ka] == "clip_applier" and clipping.line == ("cystic_duct", "cystic_artery") and clipping.slots == 4
    assert clipping.eligible.numpy().sum() == 25  # both lines' nodes may be clipped (M6)
    for _ in range(60):  # the tissue falls and hangs
        sim.step()
    graph = sim.graph
    # every node is the mean of its clamp particles, per env
    c = body.centrelines
    q = scene.view(scene.q, body).numpy()
    positions = centreline.positions.numpy()
    for env in range(2):
        for k in range(25):
            ids = c.clamp[0, c.clamp_start[0, k] : c.clamp_start[0, k] + c.clamp_count[0, k]]
            assert np.allclose(positions[env, k], q[env][ids].mean(axis=0), atol=1e-5)
    assert (centreline.active_count.numpy() == c.clamp_count[0][None]).all()
    assert not np.allclose(positions[0], c.rest[0], atol=1e-3)  # the duct moved with the falling tissue

    # steer env 0's grasper root onto a duct node (the tissue yields as the tool arrives, so follow it) and
    # fire once; env 1's grasper fires far from the duct
    duct = np.flatnonzero(c.line == 1)
    node = int(duct[len(duct) // 2])
    actions = control.actions.numpy()
    for _ in range(90):
        actions[0, ka, :3] = centreline.positions.numpy()[0, node]
        control.actions.assign(actions)
        sim.step()
    # the open rings, before the fire step: hold_clips pinches from the step that attaches the clip on (lumen 0,
    # stiffness 1 in every iteration), so a ring measured after it is already closed, fully on some GPUs (NVBug 6940611)
    open_rings = {int(k): ring_radius(scene, body, centreline, 0, k) for k in duct}
    root = control.root.numpy()[0, ka]
    gap = float(np.linalg.norm(root - centreline.positions.numpy()[0, node]))
    assert gap < 0.25, f"the root stayed {gap:.3f} from the node"
    clipping.reach.fill_(0.25)  # live: the default 0.1 is for a tip placed by hand
    actions[:, ka, 9] = 1.0  # fire in both envs
    control.actions.assign(actions)
    sim.step()
    assert clipping.count.numpy().tolist() == [1, 0]  # env 1's applier is not within reach of any duct node
    node = int(clipping.node.numpy()[0, 0])  # the nearest duct node at that moment
    assert c.line[node] == 1 and clipping.clipped.numpy()[0, node] == 1 and clipping.clipped.numpy()[1].max() == 0
    before = open_rings[node]
    for _ in range(30):  # fire held: no second clip (edge), the ring pinches
        control.actions.assign(actions)
        sim.step()
    assert clipping.count.numpy().tolist() == [1, 0] and sim.graph is graph
    after = ring_radius(scene, body, centreline, 0, node)
    assert after < 0.35 * before, (before, after)
    others = [k for k in duct if abs(k - node) > 2]
    assert ring_radius(scene, body, centreline, 0, others[0]) > 0.6 * open_rings[int(others[0])]  # neighbours keep their lumen
    actions[:, ka, 9] = 0.0
    for _ in range(60):  # the clip holds while the tissue keeps sagging
        control.actions.assign(actions)
        sim.step()
    assert ring_radius(scene, body, centreline, 0, node) < 0.35 * before and np.isfinite(scene.q.numpy()).all()
    # a second press takes the next nearest node
    actions[0, ka, 9] = 1.0
    control.actions.assign(actions)
    sim.step()
    assert clipping.count.numpy()[0] == 2 and clipping.node.numpy()[0, 1] != node and clipping.clipped.numpy()[0].sum() == 2
    # masked reset of env 1 leaves env 0's clips; of env 0 clears them
    sim.reset([1])
    assert clipping.count.numpy().tolist() == [2, 0]
    sim.reset([0])
    assert clipping.count.numpy().tolist() == [0, 0] and clipping.clipped.numpy().max() == 0 and (clipping.node.numpy() == -1).all()
    assert np.allclose(centreline.positions.numpy()[0], c.rest[0])


def test_clipping_structural():
    sim = load_case("chole_tet_med")
    clipping = find(sim, Clipping)
    sim.step()
    clipping.slots = 8
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    with pytest.raises(ValueError, match="fixed at construction"):
        sim.prepare()


def test_clip_attaches_before_the_first_substep():
    """The tracker's clamp counts are full after construction and after a masked reset, so a fire on the very
    first step attaches (review finding: reset zeroed the counts and the first press was lost)."""
    sim = load_case("chole_tet_med", envs=2)
    _scene, body = sim.scene, sim.scene.body('tissue')
    control, centreline, clipping = find(sim, InstrumentControl), find(sim, Centreline), find(sim, Clipping)
    c = body.centrelines
    assert np.array_equal(centreline.active_count.numpy(), c.clamp_count[body.variant_of_env])
    node = int(np.flatnonzero(c.line == 1)[0])
    k = clipping.k
    actions = control.actions.numpy()
    actions[0, k, :3] = c.rest[0, node]  # park the applier on a duct node of the rest pose; reset restores it
    actions[0, k, 9] = 1.0
    clipping.reach.fill_(10.0)
    for _ in range(3):
        control.actions.assign(actions)
        sim.step()
    sim.reset([0])
    assert np.array_equal(centreline.active_count.numpy()[0], c.clamp_count[0]) and clipping.count.numpy()[0] == 0
    actions[0, k, 9] = 0.0
    control.actions.assign(actions)
    sim.step()
    actions[0, k, 9] = 1.0
    control.actions.assign(actions)
    sim.step()  # the press right after the reset counts
    assert clipping.count.numpy().tolist() == [1, 0]


def test_clip_mesh_kernel_places_a_torus_on_the_node():
    """``place_clip_mesh`` on the CPU: an active slot is a ring around its node in the plane normal to the vessel tangent,
    an inactive slot collapses to the node."""
    from i4h_standalone.render.clips import place_clip_mesh, torus

    dev = wp.get_device("cpu")
    centre, tube, faces = torus()
    V = len(centre)
    assert faces.min() == 0 and faces.max() == V - 1 and len(faces) == 2 * V
    positions = wp.array(np.float32([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]]), dtype=wp.vec3, device=dev)  # (1, 3) along +x
    node = wp.array(np.int32([[1, -1]]), dtype=wp.int32, device=dev)
    active = wp.array(np.int32([[1, 0]]), dtype=wp.int32, device=dev)
    line = wp.zeros(3, dtype=wp.int32, device=dev)
    out, normals = (wp.zeros(2 * V, dtype=wp.vec3, device=dev) for _ in range(2))
    wp.launch(place_clip_mesh, dim=2 * V,
              inputs=[wp.array(centre, dtype=wp.vec3, device=dev), wp.array(tube, dtype=wp.vec3, device=dev), node, active, positions, line, 0, 0.1, 0.03],
              outputs=[out, normals], device=dev)
    p, n = out.numpy().reshape(2, V, 3), normals.numpy().reshape(2, V, 3)
    np.testing.assert_allclose(p[0].mean(axis=0), [0.1, 0.0, 0.0], atol=1e-6)
    d = np.linalg.norm(p[0] - [0.1, 0.0, 0.0], axis=1)
    assert d.min() >= 0.07 - 1e-6 and d.max() <= 0.13 + 1e-6
    assert np.abs(p[0][:, 0] - 0.1).max() <= 0.03 + 1e-6  # the ring plane is perpendicular to the tangent (x)
    assert np.allclose(np.linalg.norm(n[0], axis=1), 1.0, atol=1e-6)
    assert np.allclose(p[1], [0.0, 0.0, 0.0])  # slot 1 inactive: collapsed onto node max(-1, 0)


def test_snapshot_carries_clip_slots_and_centreline_positions():  # the module skips without chole
    from i4h_standalone.render.snapshots import Snapshot, Snapshots

    sim = load_case("chole_tet_med", envs=2)
    clipping, centreline = find(sim, Clipping), find(sim, Centreline)
    live = Snapshot(sim)
    node, active = live.clip[clipping]
    assert node.ptr == clipping.node.ptr and active.ptr == clipping.active.ptr and live.centreline[centreline].ptr == centreline.positions.ptr
    assert live.arrays.index(node) < live.arrays.index(active) < live.arrays.index(live.centreline[centreline])
    ring = Snapshots(sim)
    slot = ring.slots[0]
    assert len(slot.arrays) == len(live.arrays) and all(a.shape == b.shape and a.ptr != b.ptr for a, b in zip(slot.arrays, live.arrays))
    assert slot.clip[clipping][0].shape == clipping.node.shape and slot.centreline[centreline].shape == centreline.positions.shape
