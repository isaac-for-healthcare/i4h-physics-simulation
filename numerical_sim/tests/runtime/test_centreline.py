# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9a part 2: centreline trees (JSON loader, welding and stitching, spline topology, tree frames and bend pairs, the
Y junction on rod_pbd from omniendo tests/test_pbd_branching.py), then lumen containment and the aorta centreline case."""

import contextlib
import json

import numpy as np
import pytest
import warp as wp
import yaml
from helpers import find, run_rates
from i4h_endoluminal.loaders.centreline import (
    add_centreline_tree,
    build_tree,
    derive_bend_pairs,
    load_centreline_json,
    spline_topology,
    tree_frames,
)
from i4h_physics import Scene, Stepper
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.rods.rod_pbd import RodPbd


def branch(points, rmin, rmax):
    return {"samples": [{"position": list(map(float, p)), "radius_min": float(a), "radius_max": float(b)}
                        for p, a, b in zip(points, rmin, rmax)]}


def test_json_weld_stitch_and_radii(tmp_path):
    """Two branches meeting within the weld epsilon share a node; a third 2 cm away is stitched; sentinel radii
    (zero, negative) are repaired from their neighbours; the root is the node nearest the given position."""
    trunk = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]])
    side = np.array([[0.4, 1.0e-7, 0.0], [0.4, 0.1, 0.0], [0.4, 0.2, 0.0], [0.4, 0.3, 0.0]])  # welds onto the trunk's end
    loose = np.array([[0.2, -0.02, 0.0], [0.2, -0.12, 0.0], [0.2, -0.22, 0.0]])  # 2 cm gap: stitched
    data = {"branches": [branch(trunk, [0.0, 0.05, 0.05, 0.05, 0.05], [0.07] * 5),
                         branch(side, [0.04] * 4, [0.06, -1.0, 0.06, 0.06]),
                         {"branch_id": 7, "samples": [{"position": [9, 9, 9], "radius_min": 1, "radius_max": 1}]},  # one sample: skipped
                         branch(loose, [0.03] * 3, [0.05] * 3)]}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(data))
    starts, ends, smin, emin, smax, emax = load_centreline_json(path)
    assert len(starts) == 4 + 3 + 2 and (smin > 0).all() and (emax > 0).all()  # sentinels repaired
    assert smin[0] == 0.05  # the trunk's zero start radius took its neighbour's value
    pos, edges, radii, paths, root = build_tree(starts, ends, 0.5 * (smin + smax), 0.5 * (emin + emax), stitch_threshold=0.03,
                                                root_position=[0.02, 0.0, 0.0])
    assert len(pos) == 5 + 3 + 3 and len(edges) == 4 + 3 + 2 + 1 and root == 0
    assert (radii > 0).all() and radii[4] == pytest.approx(0.055, abs=1e-6)  # mean of the trunk end (0.06) and side start (0.05)
    degree = np.bincount(edges.reshape(-1), minlength=len(pos))
    assert degree[4] == 2 and degree[2] == 3  # the side branch joins the trunk end; the stitch joins at the trunk's middle
    controls, neighbors = spline_topology(edges, paths)
    assert (controls[:, 1:3].min(axis=1) == edges.min(axis=1)).all() and (neighbors == -1).sum() == 2 * len(paths)
    with pytest.raises(ValueError, match="disconnected"):
        build_tree(starts, ends, 0.5 * (smin + smax), 0.5 * (emin + emax), stitch_threshold=0.01)
    scene = Scene("cuda:0", 1)
    with pytest.raises(ValueError, match="degree-one root"):
        add_centreline_tree(scene, "t", pos, edges, radii, root=2, open_root=True)
    body = add_centreline_tree(scene, "t", pos, edges, radii, root, pinned_nodes=[root], pinned_segments=[0], open_root=True)
    assert body.tree.root_neighbor == 1 and body.tree.root_arc[0, 4] == pytest.approx(0.4, abs=1e-5)


# omniendo tests/test_pbd_branching.py:24-49: three branches 120 degrees apart with a +z tilt from a shared junction.
def build_y(per_branch=12, seg_len=0.05):
    dirs = []
    for k in range(3):
        a = 2 * np.pi * k / 3
        d = np.array([np.cos(a), np.sin(a), 0.3])
        dirs.append(d / np.linalg.norm(d))
    pos, edges = [np.zeros(3)], []
    for d in dirs:
        prev = 0
        for i in range(1, per_branch + 1):
            pos.append(d * (i * seg_len))
            edges.append((prev, len(pos) - 1))
            prev = len(pos) - 1
    return np.asarray(pos, dtype=np.float32), np.asarray(edges, dtype=np.int32), dirs


def y_sim(envs=1, gravity=(0.0, 0.0, 0.0), damping=1.0, sweeps=20, **rod):
    pos, edges, dirs = build_y()
    scene = Scene("cuda:0", envs, gravity=gravity)
    body = add_centreline_tree(scene, "y", pos, edges, 0.05, root=0, pinned_nodes=[0], pinned_segments=[0, 12, 24], mass=1.0, inertia=1.0)
    scene.allocate()
    tree = RodPbd(scene, body="y", bend=0.6, twist=0.6, sweeps=sweeps, **rod)
    sim = Stepper.create(scene, [Particles(scene, damping=damping), tree], dt=1.0 / 60.0, substeps=1, iterations=1)
    return sim, scene, body, tree, dirs


def branch_angles(q, dirs):
    tips = [q[12], q[24], q[36]]
    return [np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
            for a, b in ((tips[0], tips[1]), (tips[1], tips[2]), (tips[2], tips[0]))]


def test_y_topology():
    sim, scene, body, tree, dirs = y_sim()
    t = body.tree
    assert body.count == 37 and t.edges.shape == (1, 36, 2) and t.bend_pairs.shape == (1, 35, 2)
    cross = [(a, b) for a, b in t.bend_pairs[0] if set(t.edges[0, a]) & set(t.edges[0, b]) == {0}]
    assert len(cross) == 2  # spanning-tree coupling: each branch to the trunk's local parent, never branch to branch
    assert tree.frame_count == 36 and tree.bend_count == 35 and tree.seg.shape == (1, 36) and tree.pair.shape == (1, 35)
    assert np.allclose(np.linalg.norm(t.frames[0], axis=1), 1.0, atol=1e-6) and t.root_arc[0, 12] == pytest.approx(0.6, abs=1e-5)
    assert (body.rest.inv_mass[0, 0] == 0.0) and (t.inv_mass_q[0, [0, 12, 24]] == 0.0).all() and (t.inv_mass_q[0, 1] > 0)


def test_y_bend_pairs_and_frames():
    pos, edges, _ = build_y()
    pairs = derive_bend_pairs(edges, 0)
    assert len(pairs) == 2 * len(edges) - len(pos)  # one pair per (node, non-parent incident edge): sum of (degree - 1)
    frames = tree_frames(pos, edges, 0)
    assert frames.shape == (len(edges), 4) and np.allclose(np.linalg.norm(frames, axis=1), 1.0, atol=1e-5)


def test_y_rest_holds_splay():
    sim, scene, body, tree, dirs = y_sim()
    q0 = scene.view(scene.q, body).numpy()[0].copy()
    angles0 = branch_angles(q0, dirs)
    for _ in range(200):
        sim.step()
    q = scene.view(scene.q, body).numpy()[0]
    assert np.isfinite(q).all()
    assert max(abs(a - b) for a, b in zip(branch_angles(q, dirs), angles0)) < 2.0
    assert np.abs(q - q0).max() < 0.01


def test_y_gravity_droop():
    sim, scene, body, tree, dirs = y_sim(gravity=(0.0, 0.0, -9.81), damping=0.98)
    q0 = scene.view(scene.q, body).numpy()[0].copy()
    for _ in range(600):
        sim.step()
    q = scene.view(scene.q, body).numpy()[0]
    assert np.isfinite(q).all() and np.linalg.norm(q[0]) < 1e-6
    assert all(q[tip, 2] < q0[tip, 2] - 1e-3 for tip in (12, 24, 36))
    e = body.tree.edges[0]
    length = np.linalg.norm(q[e[:, 1]] - q[e[:, 0]], axis=1).sum()
    assert abs(length / (36 * 0.05) - 1.0) < 0.05
    assert np.abs(np.linalg.norm(tree.frame.numpy(), axis=1) - 1.0).max() < 1e-3


def test_y_masked_reset_and_isolation():
    sim, scene, body, tree, dirs = y_sim(envs=2, gravity=(0.0, 0.0, -9.81), damping=0.98)
    for _ in range(100):
        sim.step()
    q = scene.view(scene.q, body).numpy()
    assert np.array_equal(q[0], q[1])  # identical envs, identical work (no float-atomic order dependence in a rod)
    keep = q[0].copy()
    sim.reset([1])
    q = scene.view(scene.q, body).numpy()
    assert np.array_equal(q[0], keep) and np.array_equal(q[1], scene.view(scene.q0, body).numpy()[1])
    assert np.array_equal(tree.frame.numpy().reshape(2, -1, 4)[1], body.tree.frames[0])


# --- lumen containment (omniendo tests/test_xcath_centerline_collision.py:39-525, through a Scene) ------------------

from i4h_physics.scene import ACTIVE_INT, Part, ParticleRest  # noqa: E402
from i4h_physics.systems.contact.graph_containment import (  # noqa: E402
    GraphContainment,
    clamp_deltas,
    measure_displacement,
    request_rebuild,
)


def lumen_scene(catheter, centreline, edges, radii, *, source_radius=0.05, catheter_inv_mass=None, tree_inv_mass=None,
                root=0, open_root=False, envs=1):
    """A catheter body ``c`` and a tree body ``v``; the tree is fully pinned unless ``tree_inv_mass`` is given."""
    catheter = np.asarray(catheter, np.float32).reshape(-1, 3)
    n, P = len(catheter), len(centreline)
    scene = Scene("cuda:0", envs, gravity=(0.0, 0.0, 0.0))
    inv = np.ones(n, np.float32) if catheter_inv_mass is None else np.asarray(catheter_inv_mass, np.float32)
    rest = ParticleRest(q0=np.tile(catheter[None], (envs, 1, 1)), inv_mass=np.tile(inv[None], (envs, 1)),
                        radius=np.full((envs, n), source_radius, np.float32), flags=np.full((envs, n), ACTIVE_INT, np.int32))
    cath = scene.add_body("c", rest, {"c": Part(0, 0, n)})
    pinned = range(P) if tree_inv_mass is None else [i for i, w in enumerate(tree_inv_mass) if w == 0.0]
    tree = add_centreline_tree(scene, "v", np.asarray(centreline, np.float32), np.asarray(edges, np.int32), radii, root,
                               pinned_nodes=list(pinned), mass=1.0, open_root=open_root)
    scene.allocate()
    if tree_inv_mass is not None:
        scene.view(scene.inv_mass, tree).assign(np.tile(np.asarray(tree_inv_mass, np.float32)[None], (envs, 1)))
    return scene, cath, tree


def project(catheter, centreline, edges, radii, *, spline=False, max_distance=0.2, two_way=False, response=0.5, open_root=False,
            candidates=False, capacity=64, skin=0.04, **kw):
    """One projection (a single ``iteration`` with ``iterations 1`` and the motion clamps off): moved catheter, moved tree, system."""
    scene, cath, tree = lumen_scene(catheter, centreline, edges, radii, open_root=open_root, **kw)
    lc = GraphContainment(scene, body="c", centreline="v", iterations=1, spline=spline, two_way=two_way, open_root=open_root,
                          candidates=candidates, capacity=capacity, max_distance=max_distance, target_response=response, skin=skin,
                          source_max_delta=10.0, target_max_delta=10.0)
    wp.copy(scene.q_prev, scene.q)
    lc.iteration(scene, 1.0 / 60.0, 0, 1)
    return scene.view(scene.q, cath).numpy()[0], scene.view(scene.q, tree).numpy()[0], lc


VESSEL2 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_lumen_tapered_radius_and_activation_shell():
    moved, _, _ = project(((0.75, 0.10, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.1))
    assert moved[0, 1] < 0.10 and moved[0, 0] < 0.75  # the taper gradient pulls toward the wider end
    gated, _, lc = project(((0.5, 0.19, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2), max_distance=0.02)
    np.testing.assert_allclose(gated, ((0.5, 0.19, 0.0),), atol=1e-7)
    assert lc.contact_count.numpy()[0] == 0


def test_lumen_union_chooses_greatest_clearance():
    moved, _, _ = project(((0.5, 0.05, 0.0),), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((0, 1), (0, 2)), (0.2, 0.2, 0.2))
    np.testing.assert_allclose(moved, ((0.5, 0.05, 0.0),), atol=1e-7)


def test_lumen_edge_subsamples_catch_a_crossing_segment():
    catheter = ((0.5, 0.0, 0.0), (1.0, 0.5, 0.0))
    moved, _, lc = project(catheter, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)), ((0, 1), (1, 2)), (0.2, 0.2, 0.2))
    assert np.linalg.norm(moved - np.asarray(catheter)) > 1e-4 and lc.samples == 2 + 3


def test_lumen_open_root_removes_cap_but_keeps_inside_wall():
    closed, _, _ = project(((-0.18, 0.0, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2))
    assert closed[0, 0] > -0.18
    outside, vessel, _ = project(((-0.18, 0.18, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2), tree_inv_mass=(1.0, 1.0), two_way=True, open_root=True)
    np.testing.assert_allclose(outside, ((-0.18, 0.18, 0.0),), atol=1e-7)
    np.testing.assert_array_equal(vessel, np.asarray(VESSEL2, np.float32))
    inside, _, _ = project(((0.1, 0.18, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2), open_root=True)
    assert inside[0, 1] < 0.18


def test_lumen_one_way_pins_vessel_and_two_way_respects_pinned_root():
    args = (((0.5, 0.18, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2))
    _, one_way, _ = project(*args, tree_inv_mass=(1.0, 1.0), two_way=False)
    np.testing.assert_array_equal(one_way, np.asarray(VESSEL2, np.float32))
    catheter, coupled, lc = project(*args, tree_inv_mass=(0.0, 2.0), two_way=True, response=0.5)
    np.testing.assert_array_equal(coupled[0], np.asarray(VESSEL2[0], np.float32))
    assert coupled[1, 1] > 0.0 and catheter[0, 1] < 0.18
    assert lc.contact_count.numpy()[0] == 1 and lc.contact_depth.numpy()[0] == pytest.approx(0.03, abs=1e-6)


def test_lumen_straight_spline_matches_linear():
    centreline = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    point = ((1.25, 0.18, 0.0),)
    linear, _, _ = project(point, centreline, ((0, 1), (1, 2)), (0.2, 0.2, 0.2))
    spline, _, _ = project(point, centreline, ((0, 1), (1, 2)), (0.2, 0.2, 0.2), spline=True)
    np.testing.assert_allclose(spline, linear, atol=1e-6)


def test_lumen_spline_direction_is_c1_across_degree_two_knot():
    centreline = ((0.0, 0.0, 0.0), (1.0, 0.25, 0.0), (2.0, 1.0, 0.0))
    tangent = np.asarray(centreline[2]) - np.asarray(centreline[0])
    normal = np.asarray((-tangent[1], tangent[0], 0.0))
    normal /= np.linalg.norm(normal)
    directions = []
    for alpha in (-1e-4, 1e-4):
        point = np.asarray(centreline[1]) + alpha * tangent + normal * 0.18
        moved, _, _ = project((point,), centreline, ((0, 1), (1, 2)), (0.2, 0.2, 0.2), spline=True)
        correction = moved[0] - point
        directions.append(correction / np.linalg.norm(correction))
    assert float(np.dot(directions[0], directions[1])) > 0.999


def test_lumen_spline_taper_and_open_root():
    tapered, _, _ = project(((0.75, 0.10, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.1), spline=True)
    assert tapered[0, 0] < 0.75
    upstream, vessel, _ = project(((-0.1, 0.18, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2), spline=True, tree_inv_mass=(1.0, 1.0),
                                  two_way=True, open_root=True)
    np.testing.assert_allclose(upstream, ((-0.1, 0.18, 0.0),), atol=1e-7)
    np.testing.assert_allclose(vessel, VESSEL2, atol=1e-7)


CHAIN4 = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)), np.float32)
EDGES4 = ((0, 1), (1, 2), (2, 3))


def test_lumen_spline_four_control_coupling_and_pinned_control():
    _, moved, _ = project(((1.5, 0.18, 0.0),), CHAIN4, EDGES4, (0.2,) * 4, spline=True, catheter_inv_mass=(0.0,),
                          tree_inv_mass=(0.0, 1.0, 1.0, 1.0), two_way=True, response=1.0)
    np.testing.assert_array_equal(moved[0], CHAIN4[0])
    assert np.isfinite(moved).all() and moved[1, 1] > 0.0 and moved[2, 1] > 0.0 and moved[3, 1] < 0.0


def test_lumen_equal_mass_full_response_resolves_relative_spline_gap():
    catheter, moved, _ = project(((1.5, 0.18, 0.0),), CHAIN4, EDGES4, (0.2,) * 4, spline=True, tree_inv_mass=(1.0,) * 4,
                                 two_way=True, response=1.0)
    scale = 0.35
    weights = np.asarray((-0.125 * scale, 0.5 + 0.125 * scale, 0.5 + 0.125 * scale, -0.125 * scale))
    gap = np.linalg.norm(catheter[0] - weights @ moved) - 0.15
    assert np.isfinite(catheter).all() and np.isfinite(moved).all() and abs(gap) < 1e-6


def test_lumen_candidates_match_brute_force():
    catheter = ((0.45, 0.18, 0.0), (0.75, 0.22, 0.0))
    centreline = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.15, 0.0), (1.4, 0.45, 0.0))
    radii = (0.24, 0.22, 0.18, 0.15)
    for spline in (False, True):
        brute = project(catheter, centreline, EDGES4, radii, spline=spline)
        cached = project(catheter, centreline, EDGES4, radii, spline=spline, candidates=True)
        for b, c in zip(brute[:2], cached[:2]):
            np.testing.assert_allclose(c, b, atol=1e-7)
        assert cached[2].overflow.numpy()[0] == 0 and (cached[2].cand_count.numpy() > 0).any()


def test_lumen_candidate_overflow_uses_exact_scan():
    centreline = tuple((float(i), 0.0, 0.0) for i in range(5))
    edges = tuple((i, i + 1) for i in range(4))
    brute = project(((2.0, 10.05, 0.0),), centreline, edges, (10.0,) * 5)
    overflowed = project(((2.0, 10.05, 0.0),), centreline, edges, (10.0,) * 5, candidates=True, capacity=1)
    for expected, actual in zip(brute[:2], overflowed[:2]):
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert overflowed[2].overflow.numpy()[0] == 1 and brute[2].overflow.numpy()[0] == 0


def test_lumen_rebuild_trips_on_combined_motion():
    dev = "cuda:0"
    catheter = wp.array((((0.0, 0.0, 0.0),),), dtype=wp.vec3, device=dev)  # (E, Pc) views
    ref_samples = wp.array(((-0.025, 0.0, 0.0),), dtype=wp.vec3, device=dev)
    tree = wp.array((((0.0, 0.0, 0.0),),), dtype=wp.vec3, device=dev)
    ref_tree = wp.array(((0.0, -0.02, 0.0),), dtype=wp.vec3, device=dev)
    max_disp = wp.zeros(2, dtype=wp.float32, device=dev)
    rebuild = wp.zeros(1, dtype=wp.int32, device=dev)
    skin = wp.full(1, 0.04, dtype=wp.float32, device=dev)
    max_distance = wp.full(1, 0.09, dtype=wp.float32, device=dev)
    built = wp.array([(0.09, 0.04)], dtype=wp.vec2, device=dev)  # the cache was built with the current parameters
    wp.launch(measure_displacement, dim=(1, 1), inputs=[catheter, 1, 1, ref_samples, tree, 1, ref_tree], outputs=[max_disp], device=dev)
    np.testing.assert_allclose(max_disp.numpy(), (0.025, 0.02), atol=1e-7)
    wp.launch(request_rebuild, dim=1, inputs=[skin, 1.0, max_distance, built], outputs=[max_disp, rebuild], device=dev)
    assert int(rebuild.numpy()[0]) == 1
    assert not max_disp.numpy().any(), "request_rebuild consumes the measured motion"
    spline_disp = wp.array((0.0, 0.035), dtype=wp.float32, device=dev)
    spline_rebuild = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(request_rebuild, dim=1, inputs=[skin, 1.175, max_distance, built], outputs=[spline_disp, spline_rebuild], device=dev)
    assert int(spline_rebuild.numpy()[0]) == 1
    below = wp.array((0.0, 0.03), dtype=wp.float32, device=dev)
    rebuild.zero_()
    wp.launch(request_rebuild, dim=1, inputs=[skin, 1.0, max_distance, built], outputs=[below, rebuild], device=dev)
    assert int(rebuild.numpy()[0]) == 0
    below.assign(np.float32([0.0, 0.03]))  # consumed by the previous request
    max_distance.fill_(0.2)  # a live edit of the activation shell invalidates the cache without any motion
    wp.launch(request_rebuild, dim=1, inputs=[skin, 1.0, max_distance, built], outputs=[below, rebuild], device=dev)
    assert int(rebuild.numpy()[0]) == 1


def test_lumen_delta_clamp_preserves_fixed_nodes():
    dev = "cuda:0"
    positions = wp.array((((0.03, 0.04, 0.0), (1.0, 0.0, 0.0)),), dtype=wp.vec3, device=dev)  # (E, P) views
    starts = wp.zeros((1, 2), dtype=wp.vec3, device=dev)
    inv_mass = wp.array(((1.0, 0.0),), dtype=wp.float32, device=dev)
    limit = wp.full(1, 0.01, dtype=wp.float32, device=dev)
    wp.launch(clamp_deltas, dim=(1, 2), inputs=[starts, inv_mass, limit], outputs=[positions], device=dev)
    clamped = positions.numpy()[0]
    assert float(np.linalg.norm(clamped[0])) == pytest.approx(0.01, abs=1e-7)
    np.testing.assert_array_equal(clamped[1], (1.0, 0.0, 0.0))


def test_lumen_live_max_distance_edit_rebuilds_candidates():
    """A stationary catheter node outside the activation shell has no candidates; widening max_distance live must
    rebuild the cache (the bound depends on it) so the next projection pulls the node in."""
    scene, cath, tree = lumen_scene(((0.5, 0.45, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2))  # gap 0.3 to the lumen
    lc = GraphContainment(scene, body="c", centreline="v", iterations=1, spline=False, open_root=False, candidates=True, max_distance=0.1,
                          source_max_delta=10.0)
    wp.copy(scene.q_prev, scene.q)
    lc.iteration(scene, 1.0 / 60.0, 0, 1)
    assert (lc.cand_count.numpy() == 0).all() and scene.view(scene.q, cath).numpy()[0, 0, 1] == 0.45
    np.testing.assert_allclose(lc.built.numpy()[0], (0.1, 0.04))
    lc.max_distance.fill_(0.5)
    lc.iteration(scene, 1.0 / 60.0, 0, 1)
    assert (lc.cand_count.numpy() == 1).all() and scene.view(scene.q, cath).numpy()[0, 0, 1] < 0.45 - 0.1
    np.testing.assert_allclose(lc.built.numpy()[0], (0.5, 0.04))


def test_lumen_open_root_follows_prepare():
    """open_root is structural: changing it and recapturing must recompute the root indices."""
    scene, cath, tree = lumen_scene(((-0.18, 0.0, 0.0),), VESSEL2, ((0, 1),), (0.2, 0.2), open_root=True)
    lc = GraphContainment(scene, body="c", centreline="v", iterations=1, spline=False, candidates=False, open_root=True, source_max_delta=10.0)
    wp.copy(scene.q_prev, scene.q)
    lc.iteration(scene, 1.0 / 60.0, 0, 1)
    assert lc.root == 0 and lc.root_neighbor == 1 and scene.view(scene.q, cath).numpy()[0, 0, 0] == -0.18  # upstream: free
    lc.open_root = False
    lc.prepare(scene)
    assert lc.root == -1 and lc.root_neighbor == -1 and (lc.rebuild.numpy() == 1).all()
    lc.iteration(scene, 1.0 / 60.0, 0, 1)
    assert scene.view(scene.q, cath).numpy()[0, 0, 0] > -0.18  # closed: the cap pushes the node in
    lc.open_root = True
    lc.prepare(scene)
    assert lc.root == 0 and lc.root_neighbor == 1


def test_lumen_open_root_requires_open_tree():
    scene, cath, tree = lumen_scene(((0.5, 0.0, 0.0),), CHAIN4, EDGES4, 0.2, root=1)
    with pytest.raises(ValueError, match="open_root"):
        GraphContainment(scene, body="c", centreline="v", open_root=True)
    lc = GraphContainment(scene, body="c", centreline="v", open_root=False)
    assert lc.root == -1 and lc.samples == 1 and lc.S == 3


# --- a real tree case: aorta_centreline ---------------------------------------------------------------------------

from conftest import env_slice  # noqa: E402
from helpers import needs_vessels  # noqa: E402
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl  # noqa: E402
from i4h_isaaclab.composition import load_case  # noqa: E402
from i4h_physics.systems.rods.rod_direct import RodDirect  # noqa: E402

TREE_RATES = np.float32([[1.0, 0.3, 0.2], [0.75, -0.2, 0.4]])  # insertion, rotation, tip bend per env


@needs_vessels
def test_tree_case_masked_reset():
    sim = load_case("aorta_centreline", envs=2)
    scene = sim.scene
    run_rates(sim, 150, TREE_RATES)
    lc, ctl, tree_rod = find(sim, GraphContainment), find(sim, CatheterControl), find(sim, RodPbd)
    cath_rod = find(sim, RodDirect)
    keep = {name: env_slice(scene, scene.q, scene.body(name), 0) for name in ("catheter", "aorta")}
    frames0 = {r: r.frame.numpy().reshape(2, -1, 4)[0].copy() for r in (tree_rod, cath_rod)}
    targets0 = ctl.targets.numpy()[0].copy()
    sim.reset([1])
    for name, before in keep.items():
        body = scene.body(name)
        assert np.array_equal(env_slice(scene, scene.q, body, 0), before), name
        assert np.array_equal(env_slice(scene, scene.q, body, 1), env_slice(scene, scene.q0, body, 1)), name
    for r, before in frames0.items():
        frames = r.frame.numpy().reshape(2, -1, 4)
        assert np.array_equal(frames[0], before)
    assert np.array_equal(tree_rod.frame.numpy().reshape(2, -1, 4)[1], scene.body("aorta").tree.frames[0])
    assert np.array_equal(ctl.targets.numpy()[0], targets0) and not ctl.targets.numpy()[1].any()
    assert lc.contact_count.numpy()[1] == 0 and lc.contact_depth.numpy()[1] == 0.0 and lc.overflow.numpy()[1] == 0
    assert lc.rebuild.numpy()[1] == 1 and (lc.cand_idx.numpy().reshape(2, -1)[1] == -1).all()


@needs_vessels
def test_tree_case_system_order_and_batching():
    sim = load_case("aorta_centreline", envs=3)
    kinds = [type(s) for s in sim.systems]
    assert kinds.index(Particles) < kinds.index(RodDirect) < kinds.index(CatheterControl) < kinds.index(RodPbd) < kinds.index(GraphContainment)
    dims = []
    real = wp.launch

    def record(kernel, dim, *args, **kwargs):
        dims.append(tuple(dim) if isinstance(dim, (tuple, list)) else (dim,))
        return real(kernel, dim, *args, **kwargs)

    wp.launch = record
    try:
        sim.cuda_graph = False
        sim.prepare()
        sim.step()
    finally:
        wp.launch = real
    assert dims and all(d[0] == 3 for d in dims if len(d) > 1)
    from i4h_isaaclab.views import state_views

    assert "contact_depth" in GraphContainment.state_products
    rows = [v for v in state_views(sim) if v.shape == (3, 1)]
    assert len(rows) == 3  # contact_depth, contact_count, overflow of the containment (targets is (E, 3))


# --- skinning ------------------------------------------------------------------------------------------------------

from i4h_endoluminal.loaders.centreline import bind_skin  # noqa: E402
from i4h_physics.systems.rods.rod_pbd import _qmul_np  # noqa: E402
from i4h_standalone.render.skin import Skinner  # noqa: E402
from i4h_standalone.render.snapshots import Snapshot  # noqa: E402


def tube(rings=4, around=8, radius=0.1, length=1.0):
    """A tube along +x: ``rings`` rings of ``around`` vertices, triangulated between rings."""
    x = np.linspace(0.0, length, rings)
    theta = np.linspace(0.0, 2 * np.pi, around, endpoint=False)
    pts = np.asarray([[xi, radius * np.cos(t), radius * np.sin(t)] for xi in x for t in theta], np.float32)
    tris = []
    for r in range(rings - 1):
        for k in range(around):
            a, b = r * around + k, r * around + (k + 1) % around
            tris += [(a, b, a + around), (b, b + around, a + around)]
    return pts, np.asarray(tris, np.int32)


def test_skin_rigid_motion():
    pts, tris = tube()
    pos = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    scene = Scene("cuda:0", 2, gravity=(0.0, 0.0, 0.0))
    body = add_centreline_tree(scene, "v", pos, np.asarray([[0, 1], [1, 2]], np.int32), 0.1, root=0, pinned_nodes=[0], mass=1.0)
    scene.allocate()
    body.skin = skin = bind_skin("v", pts, tris, body.tree, "cuda:0", {"color": [0.5, 0.5, 0.5]})
    assert skin.adj_offsets[-1] == len(skin.adj_vertices) and (np.diff(skin.adj_offsets) >= 4).all()
    assert (skin.bound_edge[:16] == 0).all() and (skin.bound_edge[16:] == 1).all()  # rings at x 0, 1/3 | 2/3, 1
    rod = RodPbd(scene, body="v")
    sim = Stepper.create(scene, [Particles(scene), rod], dt=1.0 / 60.0, substeps=1, iterations=1)
    snap = Snapshot(sim)
    sk = Skinner(body, rod, "cuda:0")
    out = wp.zeros(sk.count, dtype=wp.vec3, device="cuda:0")
    sk.place(snap, 1, out, "cuda:0")
    np.testing.assert_allclose(out.numpy(), pts, atol=1e-5)  # the rest pose skins to itself
    d = np.float32([0.2, -0.1, 0.3])
    q = scene.view(scene.q, body).numpy()
    q[1] += d
    scene.view(scene.q, body).assign(q)
    sk.place(snap, 1, out, "cuda:0")
    np.testing.assert_allclose(out.numpy(), pts + d, atol=1e-5)
    sk.place(snap, 0, out, "cuda:0")
    np.testing.assert_allclose(out.numpy(), pts, atol=1e-5)  # env 0 untouched
    # A rigid rotation of positions and frames rotates the surface.
    axis, angle = np.float32([0.0, 0.0, 1.0]), 0.7
    r = np.float32([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])
    R = np.asarray([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]], np.float32)
    q[1] = pos @ R.T
    scene.view(scene.q, body).assign(q)
    frames = rod.frame.numpy().reshape(2, -1, 4)
    frames[1] = np.asarray([_qmul_np(r, f) for f in body.tree.frames[0]], np.float32)
    rod.frame.assign(frames.reshape(-1, 4))
    sk.place(snap, 1, out, "cuda:0")
    np.testing.assert_allclose(out.numpy(), pts @ R.T, atol=1e-4)


@needs_vessels
def test_tree_case_skin_binds_every_vertex():
    sim = load_case("aorta_centreline", envs=1)
    body = sim.scene.body("aorta")
    skin = body.skin
    assert skin is not None and not sim.scene.static_meshes  # the surface follows the tree, no static copy
    n, S = len(skin.points), body.tree.edges.shape[1]
    assert n > 1000 and (0 <= skin.bound_edge).all() and (skin.bound_edge < S).all() and (0 <= skin.bound_t).all() and (skin.bound_t <= 1).all()
    assert np.allclose(np.linalg.norm(skin.rest_orientation, axis=1), 1.0, atol=1e-5)
    assert len(np.unique(skin.bound_edge)) > S // 2  # most segments drive some surface


# --- dragging -------------------------------------------------------------------------------------------------------

from i4h_endoluminal.systems.rods.drag import Drag, geodesic_weights  # noqa: E402
from i4h_isaaclab.camera.camera import Camera  # noqa: E402
from i4h_standalone.app.drag import DragDriver, pick_node, ray_plane  # noqa: E402


def test_drag_weights_and_reach_target():
    pos, edges, dirs = build_y()
    scene = Scene("cuda:0", 2, gravity=(0.0, 0.0, 0.0))
    body = add_centreline_tree(scene, "y", pos, edges, 0.05, root=0, pinned_nodes=[0], pinned_segments=[0, 12, 24], mass=1.0, inertia=1.0)
    scene.allocate()
    rod = RodPbd(scene, body="y", bend=0.6, twist=0.6, sweeps=8)
    drag = Drag(scene, body="y")
    sim = Stepper.create(scene, [Particles(scene), rod, drag], dt=1.0 / 60.0, substeps=1, iterations=1)
    w = geodesic_weights(pos, edges, body.rest.inv_mass[0], 6)  # mid first branch: radius 4 * 0.05 = 0.2 -> 4 hops
    assert w[6] == 1.0 and w[0] == 0.0 and w[5] == pytest.approx(w[7]) and 0 < w[8] < w[7] < 1 and w[10] < 1e-6 and w[13] == 0.0  # 4 hops = the radius
    q0 = scene.view(scene.q, body).numpy().copy()
    target = q0[1, 6] + np.float32([0.0, 0.0, 0.05])
    sim.prepare()
    drag.begin(1, 6, target)
    assert drag.active.numpy().tolist() == [0, 1] and drag.node.numpy()[1] == 6
    for _ in range(60):
        sim.step()
    q = scene.view(scene.q, body).numpy()
    assert np.allclose(q[0], q0[0], atol=1e-5)  # env 0 untouched (the Jacobi rod drifts ~1e-6 at rest)
    assert np.linalg.norm(q[1, 6] - target) < 1e-3
    moved = np.linalg.norm(q[1] - q0[1], axis=1)
    assert moved[5] > moved[4] > moved[3] > 1e-4 and moved[13] < 1e-3  # neighbours follow with the falloff, the other branch stays
    assert np.linalg.norm(scene.view(scene.qd, body).numpy()[1, 6]) < 1e-6  # a pinned node carries no velocity
    drag.release(1)
    for _ in range(60):
        sim.step()
    q = scene.view(scene.q, body).numpy()
    assert np.isfinite(q).all() and np.linalg.norm(q[1, 6] - target) > 1e-3  # released: the rod springs back
    drag.begin(1, 6, target)
    sim.reset([1])
    assert drag.active.numpy()[1] == 0 and np.array_equal(scene.view(scene.q, body).numpy()[1], q0[1])


@pytest.mark.parametrize("substeps", [1, 4])
def test_drag_motion_limit_and_zero_velocity(substeps):
    """One substep moves the picked node at most max_delta from the substep start (an immutable reference: the
    iterations re-pin the same position) and the dragged nodes end the step with no velocity."""
    pos, edges, dirs = build_y()
    scene = Scene("cuda:0", 1, gravity=(0.0, 0.0, 0.0))
    body = add_centreline_tree(scene, "y", pos, edges, 0.05, root=0, pinned_nodes=[0], pinned_segments=[0, 12, 24], mass=1.0, inertia=1.0)
    scene.allocate()
    rod = RodPbd(scene, body="y", sweeps=4)
    drag = Drag(scene, body="y", max_delta=0.01)
    sim = Stepper.create(scene, [Particles(scene), rod, drag], dt=1.0 / 60.0, substeps=substeps, iterations=2)
    q0 = scene.view(scene.q, body).numpy()[0].copy()
    sim.prepare()
    drag.begin(0, 6, q0[6] + np.float32([0.0, 0.0, 0.5]))
    for step in range(1, 4):
        sim.step()
        q, qd = scene.view(scene.q, body).numpy()[0], scene.view(scene.qd, body).numpy()[0]
        assert np.linalg.norm(q[6] - q0[6]) <= step * substeps * 0.01 + 1e-5, (step, np.linalg.norm(q[6] - q0[6]))
        assert np.linalg.norm(q[6] - q0[6]) >= step * substeps * 0.01 - 1e-4  # and it does use the whole allowance
        assert np.linalg.norm(qd[6]) < 1e-5 and np.linalg.norm(qd[5]) < 1e-5


def test_camera_project_ray_roundtrip_and_pick():
    cam = Camera(position=(3.0, -10.0, 2.0), pitch=-10.8, yaw=-16.7, fov=45.0, up=(0.0, 0.0, 1.0))
    W, H = 640.0, 360.0
    rng = np.random.default_rng(3)
    pts = rng.uniform(-2.0, 2.0, (50, 3)).astype(np.float32)
    scr = cam.project(pts, W, H)
    assert (scr[:, 2] > 0).all()
    for p, (px, py, depth) in zip(pts, scr):
        origin, direction = cam.ray(px, py, W, H)
        closest = origin + direction * float(np.dot(p - origin, direction))
        assert np.linalg.norm(closest - p) < 1e-4 * max(1.0, depth)
    centre = cam.project(np.asarray([cam.target]), W, H)[0]
    assert abs(centre[0] - W / 2) < 1e-3 and abs(centre[1] - H / 2) < 1e-3
    above = cam.project(np.asarray([cam.target + cam.up * 0.5]), W, H)[0]
    assert above[1] < centre[1]  # up on screen is a smaller row index
    inv_mass = np.ones(len(pts), np.float32)
    k = 7
    assert pick_node(pts, inv_mass, cam, scr[k, 0] + 3.0, scr[k, 1] - 2.0, W, H) == k
    inv_mass[k] = 0.0
    picked = pick_node(pts, inv_mass, cam, scr[k, 0], scr[k, 1], W, H)
    assert picked != k
    assert pick_node(pts, inv_mass, cam, -100.0, -100.0, W, H) is None
    hit = ray_plane((0.0, 0.0, 5.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    np.testing.assert_allclose(hit, (0.0, 0.0, 0.0), atol=1e-7)
    assert ray_plane((0.0, 0.0, 5.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)) is None


class FakeEvent:
    def __init__(self, kind, button="right", pos=(0.0, 0.0)):
        self.kind, self.pos = kind, type("P", (), {"x": pos[0], "y": pos[1]})()
        self.button = type("B", (), {"name": button})()

    def is_button_down(self):
        return self.kind == "down"

    def is_button_up(self):
        return self.kind == "up"

    def is_move(self):
        return self.kind == "move"

    def is_scroll(self):
        return False


def test_drag_driver_picks_and_edits():
    """The driver on a fake renderer (camera, live snapshot, no window): press picks the node under the cursor and
    begins the drag through ``apply``, motion moves the target in the drag plane, release ends it."""
    pos, edges, dirs = build_y()
    scene = Scene("cuda:0", 1, gravity=(0.0, 0.0, 0.0))
    body = add_centreline_tree(scene, "y", pos, edges, 0.05, root=0, pinned_nodes=[0], pinned_segments=[0, 12, 24], mass=1.0)
    scene.allocate()
    rod = RodPbd(scene, body="y")
    drag = Drag(scene, body="y")
    sim = Stepper.create(scene, [Particles(scene), rod, drag], dt=1.0 / 60.0, substeps=1, iterations=1)
    cam = Camera(position=(0.0, -3.0, 0.5), pitch=-9.0, yaw=0.0, fov=45.0, up=(0.0, 0.0, 1.0))
    renderer = type("R", (), {})()
    renderer.camera, renderer.env, renderer.snap, renderer.live = cam, 0, None, Snapshot(sim)
    renderer.stream, renderer.lock, renderer.window, renderer.width, renderer.height = None, contextlib.nullcontext(), None, 640, 360
    renderer.mouse_handlers = []
    applied = []
    driver = DragDriver(sim, drag, renderer, lambda fn: (applied.append(fn), fn(sim)))
    assert renderer.mouse_handlers == [driver.handle_mouse]
    scr = cam.project(pos, 640, 360)
    cursor = (scr[8, 0], scr[8, 1])
    k = pick_node(pos, body.rest.inv_mass[0], cam, *cursor, 640, 360)  # the nearest to the camera within 18 px (omniendo's rule)
    assert k is not None and np.hypot(scr[k, 0] - cursor[0], scr[k, 1] - cursor[1]) <= 18.0
    assert driver.handle_mouse(FakeEvent("down", pos=cursor))  # consumed: not a camera pan
    assert driver.node == k and drag.active.numpy()[0] == 1 and drag.node.numpy()[0] == k
    assert not driver.handle_mouse(FakeEvent("down", button="left", pos=(0.0, 0.0)))
    driver.handle_mouse(FakeEvent("move", pos=(scr[k, 0] + 40.0, scr[k, 1])))
    target = drag.target.numpy()[0]
    _, right, _, _ = cam.basis()
    assert np.dot(target - pos[k], right) > 0.05 and abs(np.dot(target - pos[k], cam.basis()[3])) < 1e-4  # moved right, in the plane
    assert driver.handle_mouse(FakeEvent("up"))
    assert driver.env is None and drag.active.numpy()[0] == 0 and len(applied) == 3
    assert not driver.handle_mouse(FakeEvent("down", pos=(5.0, 5.0)))  # nothing under the cursor: the camera pans
    assert not driver.handle_mouse(FakeEvent("up")) and len(applied) == 3  # and gets the release too, so the pan ends


# --- control layouts -----------------------------------------------------------------------------------------------

from i4h_endoluminal.loaders.centreline import load_control_layout  # noqa: E402
from i4h_endoluminal.systems.rods.control_layout import ControlLayout  # noqa: E402


def sphere(pid, position, *, active=True, stiffness=0.2, falloff=0.5, rotation=(0.0, 0.0, 0.0, 1.0), **extra):
    return {"id": f"s{pid}", "active": active, "pid": pid, "position": list(position), "rotation": list(rotation), "radius": 0.1,
            "stiffness": stiffness, "falloff": falloff, **extra}


def layout_sim(layouts, envs=1, sweeps=1):
    pos, edges, dirs = build_y()
    scene = Scene("cuda:0", envs, gravity=(0.0, 0.0, 0.0))
    body = add_centreline_tree(scene, "y", pos, edges, 0.05, root=0, pinned_nodes=[0], pinned_segments=[0, 12, 24], mass=1.0, inertia=1.0)
    scene.allocate()
    body.anatomy.layouts = layouts
    rod = RodPbd(scene, body="y", sweeps=sweeps)
    ctl = ControlLayout(scene, centreline="y", rod=rod)
    sim = Stepper.create(scene, [Particles(scene), rod, ctl], dt=1.0 / 60.0, substeps=1, iterations=1)
    return sim, scene, body, rod, ctl, pos


def test_control_layout_weights_pull_toward_the_line():
    """One sphere at node 6 (six hops out the first branch), line along +z through 0.1 above it: node 6 moves
    stiffness * offset, its neighbours by stiffness * falloff ** hops of their own offsets (omniendo weights)."""
    pos = build_y()[0]
    sim, scene, body, rod, ctl, _ = layout_sim([[sphere(6, pos[6] + np.float32([0.0, 0.0, 0.1]))]])
    arc = body.tree.root_arc[0]
    q0 = scene.view(scene.q, body).numpy()[0].copy()
    ctl.iteration(scene, 1.0 / 60.0, 0, 1)
    q = scene.view(scene.q, body).numpy()[0]
    origin = pos[6] + np.float32([0.0, 0.0, 0.1])
    for node, hops in ((6, 0), (5, 1), (7, 1), (4, 2), (8, 2), (1, 5)):
        target = origin + np.float32([0.0, 0.0, 1.0]) * (arc[node] - arc[6])
        expected = q0[node] + (target - q0[node]) * 0.2 * 0.5**hops
        np.testing.assert_allclose(q[node], expected, atol=1e-6)
    assert np.array_equal(q[0], q0[0])  # pinned root untouched
    assert 0 < np.abs(q[13] - q0[13]).max() < 1e-3  # the other branch: seven hops, 0.2 * 0.5 ** 7 of a ~0.3 offset


def test_control_layout_inactive_sphere_and_materials():
    pos = build_y()[0]
    inactive = [[sphere(6, pos[6] + 0.1, active=False, stretch_stiffness_multiplier=0.0, bend_stiffness_multiplier=3.0)]]
    sim, scene, body, rod, ctl, _ = layout_sim(inactive)
    q0 = scene.view(scene.q, body).numpy().copy()
    ctl.step_start(scene, 1.0 / 60.0)
    ctl.iteration(scene, 1.0 / 60.0, 0, 1)
    assert np.array_equal(scene.view(scene.q, body).numpy(), q0)  # bitwise: an inactive sphere does nothing
    assert (rod.stretch_scale.numpy() == 1.0).all() and (rod.bend_scale.numpy() == 1.0).all() and not rod.bend_offset.numpy().any()
    # falloff 0: only the pinned node counts; its two segments get the multipliers, pairs touching them the bend values.
    active = [[sphere(6, pos[6], falloff=0.0, stretch_stiffness_multiplier=0.0, bend_stiffness_multiplier=3.0, rest_bend=[0.2, 0.0])]]
    sim, scene, body, rod, ctl, _ = layout_sim(active)
    ctl.step_start(scene, 1.0 / 60.0)
    seg, pair = body.tree.edges[0], body.tree.bend_pairs[0]
    touching = np.isin(seg, [6]).any(axis=1)
    assert touching.sum() == 2
    np.testing.assert_array_equal(rod.stretch_scale.numpy()[0], np.where(touching, 0.0, 1.0))
    pair_touch = touching[pair[:, 0]] | touching[pair[:, 1]]
    assert pair_touch.sum() == 3  # (4,5)-(5,6), (5,6)-(6,7), (6,7)-(7,8)
    np.testing.assert_array_equal(rod.bend_scale.numpy()[0], np.where(pair_touch, 3.0, 1.0))
    offsets = rod.bend_offset.numpy()[0]
    np.testing.assert_allclose(offsets[pair_touch], np.tile(np.float32([0.1, 0.0, 0.0]), (3, 1)), atol=1e-7)  # half the rest-bend rotation vector
    assert not offsets[~pair_touch].any()


def test_control_layout_round_robin_envs_and_step():
    pos = build_y()[0]
    layouts = [[sphere(6, pos[6] + np.float32([0.0, 0.0, 0.1]))], [sphere(18, pos[18] + np.float32([0.0, 0.0, 0.1]), stretch_stiffness_multiplier=0.5)]]  # PBD stiffness stays <= 1
    sim, scene, body, rod, ctl, _ = layout_sim(layouts, envs=3)
    assert ctl.layout_of_env.numpy().tolist() == [0, 1, 0] and ctl.L == 2 and ctl.C == 1
    q0 = scene.view(scene.q, body).numpy().copy()
    for _ in range(30):
        sim.step()
    q = scene.view(scene.q, body).numpy()
    assert np.isfinite(q).all()
    assert np.allclose(q[0], q[2], atol=1e-6) and not np.allclose(q[0], q[1], atol=1e-3)  # env 2 shares layout 0
    assert q[0, 6, 2] > q0[0, 6, 2] + 0.02 and q[1, 18, 2] > q0[1, 18, 2] + 0.02  # each env pulled toward its own sphere
    scales = rod.stretch_scale.numpy()
    assert (scales[0] == 1.0).all() and scales[1].min() == pytest.approx(0.5) and (scales[1] == 0.5).sum() == 2 and (scales[2] == 1.0).all()
    sim.reset([1])
    assert np.array_equal(scene.view(scene.q, body).numpy()[1], q0[1])
    with pytest.raises(TypeError):
        ControlLayout(scene, centreline="y", rod=Particles(scene))


def test_control_layout_loader(tmp_path):
    spheres = [{"id": f"control_{i:03d}", "active": i > 0, "pid": pid, "position": [float(i), 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0], "radius": 0.5, "stiffness": 0.1 * i, "falloff": 0.1,
                "rest_bend": [0.0, 0.0], "bend_stiffness_multiplier": 1.0, "stretch_stiffness_multiplier": 1.0}
               for i, pid in enumerate((0, 16, 32, 127))]
    layout = tmp_path / "layout.yaml"
    layout.write_text(yaml.safe_dump({"schema_version": 1, "vessel": "colon", "line_axis": "local_z", "control_spheres": spheres}))
    loaded = load_control_layout(layout, 128, "colon")
    assert [s["id"] for s in loaded] == [f"control_{i:03d}" for i in range(4)]
    assert loaded[0]["active"] is False and all(s["active"] for s in loaded[1:]) and loaded[3]["pid"] == 127
    assert all(0.0 <= s["stiffness"] <= 1.0 and 0.0 <= s["falloff"] <= 1.0 for s in loaded)
    with pytest.raises(ValueError, match="pid"):
        load_control_layout(layout, 100, "colon")
    with pytest.raises(ValueError, match="vessel"):
        load_control_layout(layout, 128, "aorta")
