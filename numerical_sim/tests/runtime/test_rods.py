# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M5 checks: the rod loader, the cantilever in two_worlds, catheter controls on a free rod, and the aorta
case (insertion per env, containment, isolation, masked reset, graph replay, both backends, Isaac Lab views)."""

import numpy as np
import pytest
import warp as wp
from conftest import env_slice
from helpers import find, free_rod, needs_vessels, run_rates, signed_phi
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import load_case, read_case
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_rod
from i4h_physics.systems.contact.rod_containment import RodContainment
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd

ROD = (RodDirect, RodPbd)


def phi_of(sim, body):
    scene, cont = sim.scene, find(sim, RodContainment)
    out = wp.zeros(scene.particle_count, dtype=wp.float32, device=scene.device)
    wp.launch(signed_phi, dim=scene.particle_count, inputs=[cont.wp_mesh.id, scene.q], outputs=[out], device=scene.device)
    return scene.view(out, body).numpy()  # out is laid out like scene.q


def inside_vessel(sim, body, track_length):
    """Max phi over nodes well past the track end (in the vessel), per env; must be <= -radius."""
    q = sim.scene.view(sim.scene.q, body).numpy()
    phi = phi_of(sim, body)
    return np.array([phi[e, q[e, :, 0] > track_length + 0.3].max() for e in range(sim.scene.envs)])


def segment_error(sim, body):
    q = sim.scene.view(sim.scene.q, body).numpy()
    L = float(body.rod.rest_len[0, 0])
    return np.abs(np.linalg.norm(np.diff(q, axis=1), axis=2) - L).max()


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], np.float64)


def roll_about_z(f0, f):
    """Angle about the local z axis taking frame ``f0`` to ``f`` (both xyzw)."""
    rel = quat_mul([-f0[0], -f0[1], -f0[2], f0[3]], f)
    return 2.0 * np.arctan2(rel[2], rel[3])


# loader ---------------------------------------------------------------------------------------------------


def test_rod_loader():
    scene = Scene("cuda:0", 2, (0.0, 0.0, 0.0))
    body = add_rod(scene, "rod", [0.1, 0.2, 0.3], [0.0, 0.0, 2.0], 9, 0.02, 0.004, mass=0.01)
    m = body.rod
    q0 = body.rest.q0
    assert body.count == 9 and q0.shape == (2, 9, 3)
    assert np.allclose(np.diff(q0[0], axis=0), [[0.0, 0.0, 0.02]] * 8, atol=1e-6)
    assert (body.rest.inv_mass[:, 0] == 0.0).all() and np.allclose(body.rest.inv_mass[:, 1:], 100.0)
    assert m.frames.shape == (1, 9, 4) and np.allclose(np.linalg.norm(m.frames[0], axis=1), 1.0, atol=1e-6)
    assert m.rest_len.shape == (1, 8) and np.allclose(m.rest_len, 0.02, atol=1e-6)
    assert np.abs(m.rest_darboux).max() < 1e-6  # straight: consecutive frames coincide
    assert m.inv_mass_q[0, 0] == 0.0 and (m.inv_mass_q[0, 1:] == 1.0).all()
    assert np.allclose(m.direction[0], [0.0, 0.0, 1.0])
    # local z of every frame is the tangent
    x, y, z, w = m.frames[0, 3]
    tangent = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
    assert np.allclose(tangent, [0.0, 0.0, 1.0], atol=1e-6)
    with pytest.raises(ValueError, match="at least 3"):
        add_rod(scene, "short", [0, 0, 0], [1, 0, 0], 2, 0.02, 0.004)
    with pytest.raises(ValueError, match="non-zero"):
        add_rod(scene, "flat", [0, 0, 0], [0, 0, 0], 5, 0.02, 0.004)


# two_worlds cantilever ------------------------------------------------------------------------------------


def gravity_rod(backend, **stiffness):
    """A 9-node cantilever under gravity, root locked, on either backend (two_worlds runs the PBD one)."""
    scene = Scene("cuda:0", 2, (0.0, 0.0, -9.81))
    add_rod(scene, "rod", [0.0, 0.0, 0.2], [1.0, 0.0, 0.0], 9, 0.02, 0.004, mass=0.01)
    scene.allocate()
    rod = backend(scene, body="rod", **stiffness)
    return Stepper.create(scene, [Particles(scene, max_velocity=5.0), rod], dt=1 / 60, substeps=4, iterations=8), rod


# omniendo's unit rotational inertia makes the direct rod nearly rigid under its own 10 g weight (4 um of
# droop at young 1e6), so the gravity check uses a soft modulus there; the PBD multipliers droop as shipped.
@pytest.mark.parametrize("backend, stiffness, softer", [(RodDirect, {"young": 1.0e3}, {"young": 1.0e2}), (RodPbd, {}, {"bend": 0.05})])
def test_cantilever_droops(backend, stiffness, softer):
    sim, rod = gravity_rod(backend, **stiffness)
    scene, body = sim.scene, rod.body
    for _ in range(300):
        sim.step()
    q = scene.view(scene.q, body).numpy()
    q0 = scene.view(scene.q0, body).numpy()
    assert np.isfinite(q).all()
    droop = q0[:, -1, 2] - q[:, -1, 2]
    assert (droop > 5e-4).all() and (droop < 0.15).all(), droop  # sags, but a 0.16 m rod cannot hang lower than that
    assert np.allclose(q[:, 0], q0[:, 0]), "the locked root moved"
    L = float(body.rod.rest_len[0, 0])
    assert np.abs(np.linalg.norm(np.diff(q, axis=1), axis=2) - L).max() < 0.05 * L
    assert np.allclose(np.linalg.norm(rod.frame.numpy(), axis=1), 1.0, atol=1e-4)

    soft, soft_rod = gravity_rod(backend, **softer)  # softer: more droop
    for _ in range(300):
        soft.step()
    qs = soft.scene.view(soft.scene.q, soft_rod.body).numpy()
    assert (q0[:, -1, 2] - qs[:, -1, 2] > droop).all()


def test_cantilever_masked_reset(case):
    sim = load_case(case)
    scene, body = sim.scene, sim.scene.body("rod")
    rod = find(sim, ROD)
    for _ in range(100):
        sim.step()
    q_before, frame_before = env_slice(scene, scene.q, body, 0), rod.frame.numpy().reshape(scene.envs, -1, 4)[0].copy()
    sim.reset([1])
    frames = rod.frame.numpy().reshape(scene.envs, -1, 4)
    assert np.array_equal(env_slice(scene, scene.q, body, 0), q_before) and np.array_equal(frames[0], frame_before)
    assert np.array_equal(env_slice(scene, scene.q, body, 1), env_slice(scene, scene.q0, body, 1))
    assert np.array_equal(frames[1], body.rod.frames[0, :frames.shape[1]])
    assert not rod.omega.numpy().reshape(scene.envs, -1, 3)[1].any()


# controls on a free rod (no vessel) -------------------------------------------------------------------------


def test_insertion_follows_the_target():
    sim, rod, ctl = free_rod()
    scene, body = sim.scene, rod.body
    run_rates(sim, 60, np.float32([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    targets = ctl.targets.numpy()
    assert np.allclose(targets[:, 0], [0.5, 0.25], atol=1e-4)  # 0.5 m/s for 1 s, half of it in env 1
    q = scene.view(scene.q, body).numpy()
    assert np.allclose(q[:, 0, 0], targets[:, 0], atol=1e-5) and np.allclose(q[:, 0, 1:], [0.0, 1.0], atol=1e-6)
    assert np.abs(q[:, 1:-8, 1] - 0.0).max() < 1e-4 and np.abs(q[:, 1:-8, 2] - 1.0).max() < 1e-4  # shaft on the track
    assert segment_error(sim, body) < 2e-3
    # the slew clamp: a target far ahead is approached at max_delta per substep
    run_rates(sim, 1, np.float32([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    ctl.targets.assign(np.float32([[0.9, 0.0, 0.0], [0.25, 0.0, 0.0]]))
    x0 = scene.view(scene.q, body).numpy()[0, 0, 0]
    run_rates(sim, 1, np.zeros((2, 3), np.float32))
    assert abs(scene.view(scene.q, body).numpy()[0, 0, 0] - (x0 + 8 * 0.01)) < 1e-4


def test_rotation_twists_the_rod():
    sim, rod, ctl = free_rod()
    scene, body = sim.scene, rod.body
    run_rates(sim, 30, np.float32([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]))  # 1.5 rad/s for 0.5 s
    run_rates(sim, 60, np.zeros((2, 3), np.float32))  # settle
    targets = ctl.targets.numpy()
    assert np.allclose(targets[:, 1], [0.75, -0.75], atol=1e-4)
    frames = rod.frame.numpy().reshape(scene.envs, -1, 4)
    rest = body.rod.frames[0]
    for e in range(scene.envs):
        assert abs(roll_about_z(rest[0], frames[e, 0]) - targets[e, 1]) < 1e-4  # root frame = base * twist
        assert abs(roll_about_z(rest[-1], frames[e, -1]) - targets[e, 1]) < 1e-3  # torsion (1e9) carries it to the tip
    assert np.allclose(np.linalg.norm(frames, axis=2), 1.0, atol=1e-4)


def test_tip_bend_curls_the_tip():
    sim, rod, ctl = free_rod()
    scene, body = sim.scene, rod.body
    run_rates(sim, 60, np.float32([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))  # 1 rad/s for 1 s in env 0
    run_rates(sim, 120, np.zeros((2, 3), np.float32))
    offset = rod.bend_offset.numpy()
    B = rod.bend_count
    assert np.allclose(offset[0, B - 8:], [1.0 / 8.0, 0.0, 0.0], atol=1e-6) and not offset[0, :B - 8].any()
    assert not offset[1].any()
    q = scene.view(scene.q, body).numpy()
    off_line = np.linalg.norm(q[:, -1, 1:] - [0.0, 1.0], axis=1)
    assert off_line[0] > 0.05 and off_line[1] < 1e-3, off_line  # env 0's tip left the line, env 1's did not
    assert segment_error(sim, body) < 1e-3


@pytest.mark.parametrize("backend", [RodDirect, RodPbd])
def test_inactive_rod_holds_still(backend):
    """The inactive-body contract: nodes without the ACTIVE flag are not moved by the solver or the controls."""
    sim, rod, ctl = free_rod(backend=backend)
    scene, body = sim.scene, rod.body
    scene.view(scene.flags, body)[0].fill_(0)  # env 0's rod is present but empty
    q_before = env_slice(scene, scene.q, body, 0)
    frame_before = rod.frame.numpy().reshape(2, -1, 4)[0].copy()
    run_rates(sim, 10, np.float32([[1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]))
    assert np.array_equal(env_slice(scene, scene.q, body, 0), q_before)
    assert np.array_equal(rod.frame.numpy().reshape(2, -1, 4)[0], frame_before)
    assert np.abs(env_slice(scene, scene.q, body, 1) - env_slice(scene, scene.q0, body, 1)).max() > 0.05, "env 1 did not move"
    assert np.isfinite(scene.q.numpy()).all()
    sim.reset([0])  # flags come back with the rest state
    assert (scene.view(scene.flags, body).numpy()[0] == 1).all()
    run_rates(sim, 5, np.float32([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert env_slice(scene, scene.q, body, 0)[0, 0] > 0.0


@pytest.mark.parametrize("backend", [RodDirect, RodPbd])
def test_inactive_node_splits_the_rod(backend):
    """A dead node inside a live rod: its edges drop out of the solve, so the part past it is not driven."""
    sim, rod, ctl = free_rod(backend=backend)
    scene, body = sim.scene, rod.body
    flags = scene.field("flags", body)
    host = flags.numpy()
    host[0, 10] = 0  # env 0, node 10
    flags.assign(host)
    q_before = env_slice(scene, scene.q, body, 0)
    run_rates(sim, 20, np.float32([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    q = env_slice(scene, scene.q, body, 0)
    assert np.isfinite(scene.q.numpy()).all()
    assert np.array_equal(q[10], q_before[10]), "the dead node moved"
    assert np.allclose(q[11:], q_before[11:], atol=1e-5), "the disconnected tail was driven"
    assert q[0, 0] > q_before[0, 0] + 0.1 and (q[1:10, 0] > q_before[1:10, 0] + 0.05).all(), "the live part did not follow the root"
    q1 = env_slice(scene, scene.q, body, 1)
    assert q1[-1, 0] > env_slice(scene, scene.q0, body, 1)[-1, 0] + 0.1, "env 1 (all live) did not move as a whole"


@pytest.mark.parametrize("backend", [RodDirect, RodPbd])
def test_rod_tube_rings_follow_the_nodes(backend):
    from i4h_standalone.render.rods import RodDraw
    from i4h_standalone.render.snapshots import Snapshot

    sim, rod, ctl = free_rod(backend=backend)
    run_rates(sim, 30, np.float32([[1.0, 0.5, 0.5], [0.0, 0.0, 0.0]]))
    scene, body = sim.scene, rod.body
    draw = RodDraw(body, rod, resolution=8)
    pos = wp.zeros(draw.count, dtype=wp.vec3, device=scene.device)
    normals = wp.zeros(draw.count, dtype=wp.vec3, device=scene.device)
    draw.place(Snapshot(sim), 0, pos, normals, scene.device)
    ring = pos.numpy().reshape(body.count, 8, 3)
    q = env_slice(scene, scene.q, body, 0)
    assert np.allclose(ring.mean(axis=1), q, atol=1e-5)  # each ring is centred on its node (PBD: the last node reuses the last segment frame)
    assert np.allclose(np.linalg.norm(ring - q[:, None], axis=2), draw.radius, atol=1e-5)
    assert np.allclose(np.linalg.norm(normals.numpy(), axis=1), 1.0, atol=1e-5)
    assert draw.indices.size == (body.count - 1) * 8 * 6 and draw.indices.max() == draw.count - 1


def test_structural_and_live_settings():
    sim, rod, ctl = free_rod()
    sim.step()
    graph = sim.graph
    rod.young.fill_(2.0e6)  # live: no recapture
    ctl.track_length.fill_(0.5)
    sim.step()
    assert sim.graph is graph
    ctl.tip_edges = 4  # structural: baked into the launches
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    sim.prepare()  # nothing is sized by it: the recapture picks the new value up
    sim.step()
    with pytest.raises(ValueError, match="tip_edges"):
        free_rod(tip_edges=40)
    with pytest.raises(TypeError):
        CatheterControl(sim.scene, rod=find(sim, Particles), track_length=1.0)


# aorta ------------------------------------------------------------------------------------------------------


@needs_vessels
def test_aorta_envs_insert_differently_and_stay_inside():
    sim = load_case("aorta_static", envs=8)
    scene = sim.scene
    body = scene.body("catheter")
    ctl = find(sim, CatheterControl)
    rates = np.zeros((8, 3), np.float32)
    rates[:, 0] = (np.arange(8) + 1) / 8.0  # 0.0625 .. 0.5 m/s
    run_rates(sim, 300, rates)
    q = scene.view(scene.q, body).numpy()
    assert np.isfinite(q).all()
    depth = ctl.targets.numpy()[:, 0]
    assert np.allclose(depth, 2.5 * rates[:, 0], atol=1e-3) and np.allclose(q[:, 0, 0], depth, atol=1e-4)  # 0.5 m/s for 5 s
    assert (np.diff(q[:, -1, 0]) > 0.1).all(), "tips do not advance with the insertion speed"
    assert (inside_vessel(sim, body, 6.35) <= -0.02 + 1e-3).all(), "a node left the aorta"
    assert segment_error(sim, body) < 5e-4  # the direct backend's stretch is near-rigid (compliance 1e-10)
    cont = find(sim, RodContainment)
    assert (cont.contact_count.numpy() > 0).all() and cont.contact_depth.numpy().max() < 0.01


@needs_vessels
def test_aorta_env_isolation():
    sim = load_case("aorta_static", envs=2)
    scene, body = sim.scene, sim.scene.body("catheter")
    run_rates(sim, 100, np.zeros((2, 3), np.float32))
    baseline = scene.view(scene.q, body).numpy().copy()
    sim.reset()
    run_rates(sim, 100, np.float32([[1.0, 0.5, 0.5], [0.0, 0.0, 0.0]]))
    q = scene.view(scene.q, body).numpy()
    assert np.allclose(q[1], baseline[1], rtol=1e-4, atol=1e-3), "controls in env 0 leaked into env 1"
    assert q[0, 0, 0] > 0.5, "env 0 did not insert"


@needs_vessels
def test_aorta_masked_reset():
    sim = load_case("aorta_static", envs=2)
    scene, body = sim.scene, sim.scene.body("catheter")
    rod, ctl, cont = find(sim, ROD), find(sim, CatheterControl), find(sim, RodContainment)
    run_rates(sim, 100, np.float32([[1.0, 0.5, 0.5], [0.5, -0.5, 0.2]]))
    keep = (env_slice(scene, scene.q, body, 0), rod.frame.numpy().reshape(2, -1, 4)[0].copy(), ctl.targets.numpy()[0].copy(),
            int(cont.contact_count.numpy()[0]))
    sim.reset([1])
    frames = rod.frame.numpy().reshape(2, -1, 4)
    assert np.array_equal(env_slice(scene, scene.q, body, 0), keep[0]) and np.array_equal(frames[0], keep[1])
    assert np.array_equal(ctl.targets.numpy()[0], keep[2]) and int(cont.contact_count.numpy()[0]) == keep[3]
    assert np.array_equal(env_slice(scene, scene.q, body, 1), env_slice(scene, scene.q0, body, 1))
    assert np.array_equal(frames[1], body.rod.frames[0, :frames.shape[1]])
    assert not rod.omega.numpy().reshape(2, -1, 3)[1].any() and not rod.bend_offset.numpy()[1].any()
    assert not ctl.targets.numpy()[1].any() and not ctl.actions.numpy()[1].any()
    assert cont.contact_count.numpy()[1] == 0 and cont.contact_depth.numpy()[1] == 0.0
    run_rates(sim, 5, np.zeros((2, 3), np.float32))  # keeps stepping after the reset
    assert np.isfinite(scene.q.numpy()).all()


@needs_vessels
def test_aorta_captured_matches_uncaptured():
    rates = np.float32([[0.6, 0.3, 0.4], [0.2, -0.3, -0.2]])
    captured, eager = load_case("aorta_static", envs=2), load_case("aorta_static", envs=2)
    eager.cuda_graph = False
    run_rates(captured, 100, rates)
    run_rates(eager, 100, rates)
    assert captured.graph is not None and eager.graph is None
    body = captured.scene.body("catheter")
    qa, qb = captured.scene.view(captured.scene.q, body).numpy(), eager.scene.view(eager.scene.q, body).numpy()
    assert np.allclose(qa, qb, rtol=1e-4, atol=1e-3)  # float atomics in the correction accumulation: not bitwise
    fa, fb = find(captured, ROD).frame.numpy(), find(eager, ROD).frame.numpy()
    assert np.allclose(fa, fb, atol=1e-3)


@needs_vessels
def test_rod_pbd_runs_the_aorta():
    cfg = read_case("aorta_static")
    cfg["sim"]["iterations"] = 4  # Jacobi needs several sweeps per substep to carry the root motion down 192 nodes
    systems = []
    for entry in cfg["systems"]:
        if entry["type"] == "rod_direct":
            entry = {"type": "rod_pbd", "id": entry["id"], "body": entry["body"], "stretch": 1.0, "bend": 0.5, "twist": 0.5}
        systems.append(entry)
    cfg["systems"] = systems
    sim = load_case(cfg, envs=2)
    scene, body = sim.scene, sim.scene.body("catheter")
    assert isinstance(find(sim, ROD), RodPbd)
    run_rates(sim, 200, np.float32([[0.5, 0.3, 0.3], [0.25, 0.0, 0.0]]))
    q = scene.view(scene.q, body).numpy()
    assert np.isfinite(q).all()
    assert np.allclose(q[:, 0, 0], find(sim, CatheterControl).targets.numpy()[:, 0], atol=1e-4)
    assert np.linalg.norm(q[0, -1] - q[1, -1]) > 0.1  # env 0 inserted, rotated and bent; env 1 only inserted
    assert (inside_vessel(sim, body, 6.35) <= -0.02 + 2e-3).all()
    # PBD stretch is a soft multiplier: the shaft holds to millimetres, the segment pressing the bent tip
    # into the wall strains up to half a segment (the direct backend holds 1e-4 everywhere).
    seg = np.linalg.norm(np.diff(q, axis=1), axis=2) - float(body.rod.rest_len[0, 0])
    shaft = q[:, :-1, 0] < 6.35
    assert np.abs(seg[shaft]).max() < 1e-2 and np.abs(seg[~shaft]).max() < 0.05
    assert np.allclose(np.linalg.norm(find(sim, ROD).frame.numpy(), axis=1), 1.0, atol=1e-4)


@needs_vessels
@pytest.mark.parametrize("case", ["aorta_static"])
def test_system_order(case):
    sim = load_case(case)
    kinds = [type(s) for s in sim.systems]
    assert kinds.index(RodDirect) < kinds.index(CatheterControl) < kinds.index(RodContainment)


@needs_vessels
def test_aorta_launches_are_batched_over_envs():
    counts = {}
    for envs in (1, 3):
        sim = load_case("aorta_static", envs=envs)
        sim.cuda_graph = False
        sim.step()
        launches, leading = [], set()
        real = wp.launch

        def counting(kernel, dim, *args, **kwargs):
            launches.append(kernel.key)
            leading.add((dim if isinstance(dim, (tuple, list)) else (dim,))[0])
            return real(kernel, dim, *args, **kwargs)

        wp.launch = counting
        try:
            sim.step()
        finally:
            wp.launch = real
        counts[envs] = len(launches)
        assert leading == {envs}, f"envs={envs}: launch leading dims {leading}"
    assert counts[1] == counts[3] > 0


@needs_vessels
def test_isaaclab_stub_loop_aorta():
    torch = pytest.importorskip("torch")
    from i4h_isaaclab.views import (
        action_bounds,
        action_views,
        observation,
        observation_views,
        state_views,
        write_actions,
    )

    sim = load_case("aorta_static", envs=2)
    scene, body = sim.scene, sim.scene.body("catheter")
    ctl = find(sim, CatheterControl)
    dev = wp.device_to_torch(scene.device)
    acts, obs, state = action_views(sim), observation_views(sim), state_views(sim)
    assert len(acts) == 1 and acts[0].shape == (2, 3) and acts[0].data_ptr() == ctl.actions.ptr
    obs_t = obs[0]
    assert obs_t.shape == (2, 3 * body.count) and [tuple(v.shape) for v in state] == [(2, 3), (2, 1), (2, 1)]
    targets_t, contacts_t = state[0], state[2]
    assert observation(obs, state).shape == (2, 3 * body.count + 5)  # 581 for the 192-node catheter
    low, high = action_bounds(sim, 0.5)
    assert (low == -1.0).all() and (high == 1.0).all() and low.shape == (3,)
    policy = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=dev)
    for _ in range(10):
        for _ in range(2):  # decimation
            write_actions(acts, policy)
            sim.step()
        assert torch.isfinite(observation(obs, state)).all()
    assert targets_t[0, 0].item() > 0.1 and targets_t[1, 0].item() == 0.0
    q0_env0 = obs_t[0].clone()
    sim.reset(mask=torch.tensor([False, True], device=dev))  # Isaac Lab's reset_buf, no nonzero()
    torch.cuda.synchronize()
    assert torch.equal(obs_t[0], q0_env0)
    assert torch.equal(obs_t[1], wp.to_torch(scene.view(scene.q0, body)).reshape(2, -1)[1])
    assert contacts_t[1].item() == 0


def test_rod_pbd_sweeps_equal_iterations():
    """``sweeps`` repeats the Jacobi sweeps inside one runner iteration: with only ``particles`` beside it (no
    iteration hook) four sweeps in one iteration equal one sweep in four iterations bitwise; it is structural."""
    a, rod_a = gravity_rod(RodPbd, sweeps=4)
    b, rod_b = gravity_rod(RodPbd)
    a.iterations, b.iterations = 1, 4
    for _ in range(60):
        a.step()
        b.step()
    # two simulations: float atomics make them differ at the rounding level (AGENTS.md), so a tolerance, not equality
    assert np.allclose(a.scene.q.numpy(), b.scene.q.numpy(), atol=1e-6) and np.allclose(rod_a.frame.numpy(), rod_b.frame.numpy(), atol=1e-6)
    assert rod_a.seg.shape == (1, 8) and rod_a.pair.shape == (1, 7) and rod_a.frame_count == 8 and rod_a.bend_count == 7
    rod_a.sweeps = 2
    with pytest.raises(RuntimeError, match="structural"):
        a.step()
