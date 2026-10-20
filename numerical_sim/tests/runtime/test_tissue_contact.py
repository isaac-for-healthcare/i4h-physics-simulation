# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded same-body contact: original equations, topology, lifecycle and integration."""
import numpy as np
import pytest
from helpers import find, kill, needs_chole, needs_external_data, slang_tests
from i4h_isaaclab.composition import load_case, read_case
from i4h_physics import Scene, Stepper
from i4h_physics.loaders.tet_bundle import TET_EDGES, TET_FACES, PartMesh, pack_tet_body, parse_materials
from i4h_physics.systems.contact.part_contact import PartContact
from i4h_physics.systems.mechanics.bounds import Bounds
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.systems.mechanics.surface import Surface

pytestmark = needs_external_data

def fixture(**kw):
    cfg = read_case('tissue_contact')
    cfg['systems'][3].update(kw)
    sim = load_case(cfg)
    return sim, find(sim, PartContact)


def event(c):
    c.step_start(c.scene, 1 / 60)
    c.iteration(c.scene, 1 / 60, 0, 1)


@pytest.mark.parametrize('weights', [(1, 1, 1, 1), (2, 3, 4, 5), (0, 0, 0, 1), (1, 0, 2, 0), (0, 0, 0, 0)])
@pytest.mark.parametrize('z', [-0.009, 0.0, 0.003])
def test_original_single_contact_equations(weights, z):
    _, c = fixture()
    q = c.q.numpy()
    q[:, 4, 2] = z
    c.q.assign(q)
    inv = c.inv_mass.numpy()
    inv[:, [0, 1, 2, 4]] = weights
    c.inv_mass.assign(inv)
    # Closest point on the z=0 face: barycentric weights (0.5, 0.25, 0.25).
    bary = np.array([.5, .25, .25])
    w = np.array(weights)
    denom = w[3] + (w[:3] * bary**2).sum()
    expected = q.copy()
    if denom:
        depth = .012 + z  # outside: shell-distance; crossed: shell+distance
        expected[:, :3, 2] += depth / denom * w[:3] * bary
        expected[:, 4, 2] -= depth / denom * w[3]
    event(c)
    np.testing.assert_allclose(c.q.numpy(), expected, atol=3e-8)
    assert c.contact_count.numpy().tolist() == [int(denom > 0)] * 2
    assert not c.delta.numpy().any() and not c.delta_count.numpy().any()


@pytest.mark.parametrize('settings,z', [({'stiffness': 0}, -.009), ({}, -.02), ({}, .02)])
def test_no_response(settings, z):
    _, c = fixture(**settings)
    q = c.q.numpy()
    q[:, 4, 2] = z
    c.q.assign(q)
    event(c)
    np.testing.assert_array_equal(c.q.numpy(), q)
    assert not c.contact_count.numpy().any()


def test_limit_and_live_environment_parameters():
    _, c = fixture(max_correction=.0002)
    inv = c.inv_mass.numpy()
    inv[:, :4] = 0
    c.inv_mass.assign(inv)
    c.stiffness.assign(np.array([1, 0], np.float32))
    q = c.q.numpy()
    event(c)
    assert c.q.numpy()[0, 4, 2] == pytest.approx(q[0, 4, 2] - .0002, abs=1e-8)
    np.testing.assert_array_equal(c.q.numpy()[1], q[1])
    np.testing.assert_array_equal(c.q.numpy()[:, :4], q[:, :4])


@pytest.mark.parametrize('budget', [.0005, .01])
def test_full_step_preserves_motion_cap_and_bounds(budget):
    sim, c = fixture()
    sim.substeps = sim.iterations = 1
    find(sim, Tet).enabled = False
    find(sim, Particles).max_velocity.fill_(budget / sim.dt)
    inv = c.inv_mass.numpy()
    inv[:, :4] = 0
    c.inv_mass.assign(inv)
    # Exercise an independent out-of-bounds target vertex within the same motion budget.
    bounds = find(sim, Bounds)
    bounds.upper = (.35 - budget / 2, 2, 2)
    q = c.q.numpy()
    sim.step()
    after = c.q.numpy()
    assert np.linalg.norm(after - q, axis=-1).max() <= budget + 5e-8
    assert after[0, 5, 0] <= bounds.upper[0] + 2e-8
    assert after[0, 4, 2] == pytest.approx(-.009 - min(.003, budget), abs=3e-8)
    np.testing.assert_array_equal(after[:, :4], q[:, :4])


def test_overflow_complete_query():
    results = []
    for capacity in (1, 64):
        _, c = fixture(max_candidates=capacity)
        q = c.q.numpy()
        q[:, 4] = [.003, .003, -.003]
        c.q.assign(q)
        event(c)
        results.append((c.q.numpy(), c.contact_count.numpy()))
        assert (c.overflow_count.numpy() > 0).all() == (capacity == 1)
        assert (c.contact_count.numpy() >= 3).all()
    np.testing.assert_allclose(results[0][0], results[1][0], atol=2e-8)
    np.testing.assert_array_equal(results[0][1], results[1][1])


@pytest.mark.parametrize('remove', ['source', 'target', 'flags', 'support', 'source_flags', 'source_support'])
def test_liveness_before_surface_rebuild(remove):
    sim, c = fixture()
    if remove in ('source', 'target'):
        kill(sim, c.surface, 0, 0 if remove == 'source' else 1)
    elif remove in ('flags', 'source_flags'):
        flags = c.flags.numpy()
        flags[0, 0 if remove == 'source_flags' else 4] = 0
        c.flags.assign(flags)
    else:
        support = c.surface.support.numpy()
        support[0 if remove == 'source_support' else 4] = 0
        c.surface.support.assign(support)
    q = c.q.numpy()
    event(c)
    np.testing.assert_array_equal(c.q.numpy()[0], q[0])
    assert c.contact_count.numpy().tolist() == [0, 1]
    assert not np.array_equal(c.q.numpy()[1], q[1])
    if remove == 'source':
        c.surface.step_end(sim.scene, sim.dt)
        event(c)
        assert c.contact_count.numpy()[0] == 0  # entirely inactive grouped subtree


def mesh(q, t):
    q, t = np.asarray(q, np.float32), np.asarray(t, np.int32)
    faces = {}
    for tet in t:
        for tri in tet[TET_FACES]:
            faces.setdefault(tuple(sorted(tri)), []).append(tri)
    tris = np.array([owners[0] for owners in faces.values() if len(owners) == 1], np.int32)
    edges = np.unique(np.sort(t[:, TET_EDGES].reshape(-1, 2), axis=1), axis=0)
    return PartMesh(q, np.ones(len(q), np.float32), t, edges, tris)


def topology_fixture():
    # Env 0 has an internal z=0 face. Env 1 has that face authored, plus padded slots.
    q = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0,0,-1]]
    target = mesh([[.25,.25,-.009], [.35,.25,-.15], [.25,.35,-.15], [.25,.25,-.15]], [[0,2,1,3]])
    unrelated = mesh(np.array(q[:4]) + 3, [[0,1,2,3]])
    variants = [{'source': mesh(q, [[0,1,2,3], [0,2,1,4]]), 'target': target, 'unrelated': unrelated},
                {'source': mesh(q[:4], [[0,1,2,3]]), 'target': target, 'unrelated': unrelated}]
    scene = Scene('cuda:0', 2, (0,0,0))
    materials, visuals = parse_materials({n: {'spring': 0, 'volume': 0, 'deletable': True} for n in variants[0]})
    body = pack_tet_body(scene, 'tissue', variants, materials, .01, visuals)
    scene.allocate()
    tet = Tet(scene, body='tissue')
    surface = Surface(scene, tet=tet)
    contact = PartContact(scene, surface=surface, surface_part='source', particle_parts=['target'])
    return Stepper.create(scene, [tet, surface, contact]), contact, body


def test_cut_faces_padded_variants_and_unrelated_part():
    sim, c, body = topology_fixture()
    original = c.q.numpy()
    event(c)
    assert c.contact_count.numpy().tolist() == [0, 1]
    c.q.assign(original)
    kill(sim, c.surface, 0, 1)
    event(c)  # no ghost response; the newly exposed face is not published until rebuild
    assert c.contact_count.numpy()[0] == 0
    c.q.assign(original)
    c.surface.step_end(sim.scene, sim.dt)
    assert c.surface.cut_count.numpy()[0, 0] == 1
    event(c)
    np.testing.assert_allclose(c.q.numpy()[0, :4], c.q.numpy()[1, :4], atol=2e-8)
    start = body.parts['target'].particle_start
    np.testing.assert_allclose(c.q.numpy()[0, start:start+4], c.q.numpy()[1, start:start+4], atol=2e-8)
    start = body.parts['unrelated'].particle_start
    np.testing.assert_array_equal(c.q.numpy()[:, start:], original[:, start:])


def test_reset_disable_recapture_and_isolation():
    sim, c = fixture()
    sim.step()
    saved = c.q.numpy()[1].copy()
    c.contact_count.assign(np.array([7, 9], np.int32))
    sim.reset([0])
    np.testing.assert_array_equal(c.q.numpy()[1], saved)
    assert c.contact_count.numpy().tolist() == [0, 9]
    c.enabled = False
    sim.prepare()
    assert not c.contact_count.numpy().any() and not c.overflow_count.numpy().any()
    sim.step()
    c.enabled = True
    c.iterations = 2
    sim.prepare()
    sim.reset()
    sim.step()
    assert c.contact_count.numpy().min() > 0
    # Moving env 0's source far away must not see env 1's overlapping local coordinates.
    sim.reset()
    q = c.q.numpy()
    q[0, :4] += 10
    c.q.assign(q)
    event(c)
    assert c.contact_count.numpy().tolist() == [0, 1]


@pytest.mark.parametrize('graph', [False, True])
def test_demo_300_steps(graph):
    sim, c = fixture()
    sim.cuda_graph = graph
    sim.debug = True
    q0 = c.q.numpy()
    for _ in range(300):
        sim.step()
    assert np.isfinite(c.q.numpy()).all()
    assert c.q.numpy()[0, 4, 2] < q0[0, 4, 2]
    np.testing.assert_allclose(c.q.numpy()[0], c.q.numpy()[1], atol=1e-6)


def test_eager_captured_agreement():
    results = []
    for graph in (False, True):
        sim, c = fixture()
        sim.cuda_graph = graph
        for _ in range(30):
            sim.step()
        results.append(c.q.numpy())
    np.testing.assert_allclose(*results, atol=2e-6)


@pytest.mark.parametrize('kw', [{'surface_part': 'absent'}, {'particle_parts': ['target', 'target']},
                               {'particle_parts': ['source']}, {'particle_parts': []}, {'particle_parts': 'target'},
                               {'margin': -1}, {'stiffness': 1.1}, {'max_correction': float('nan')},
                               {'iterations': 0}, {'iterations': 1.5}, {'max_candidates': True}, {'margin': [1, 2]}])
def test_invalid_configuration(kw):
    with pytest.raises(ValueError):
        fixture(**kw)


def test_refs_and_reconstruction():
    sim, c = fixture()
    with pytest.raises(TypeError, match='surface'):
        PartContact(sim.scene, surface=find(sim, Tet), surface_part='source', particle_parts=['target'])
    for name, value in [('max_candidates', 2), ('surface_part', 'target'), ('particle_parts', ('source',))]:
        old = getattr(c, name)
        setattr(c, name, value)
        with pytest.raises(ValueError, match='fixed at construction'):
            sim.prepare()
        setattr(c, name, old)
    c.iterations = -1
    with pytest.raises(ValueError, match='iterations'):
        sim.prepare()


@pytest.mark.parametrize('damage', ['missing', 'winding', 'duplicate'])
def test_authored_boundary_validation(damage):
    sim, c = fixture()
    tris = c.body.tet.tris
    if damage == 'missing':
        tris[0, 0] = tris[0, 0, 0]
    elif damage == 'winding':
        tris[0, 0] = tris[0, 0, ::-1]
    else:
        tris[0, 1] = tris[0, 0]
    with pytest.raises(ValueError, match='boundary|winding|duplicate'):
        PartContact(sim.scene, surface=c.surface, surface_part='source', particle_parts=['target'])


@needs_chole
def test_chole_disabled_preserves_physics():
    cfg = read_case('chole_tet_med')
    with_contact = load_case(cfg, device="cpu")
    assert not find(with_contact, PartContact).enabled
    cfg['systems'] = [s for s in cfg['systems'] if s['type'] != 'tissue_contact']
    without = load_case(cfg, device="cpu")
    for sim in (with_contact, without):
        for _ in range(5):
            sim.step()
    assert find(with_contact, PartContact) not in with_contact.solver.active
    np.testing.assert_array_equal(with_contact.scene.q.numpy(), without.scene.q.numpy())


@needs_chole
def test_chole_enabled_scripted_grasp_pull():
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_surgical.systems.instruments.grasp import Grasp
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    sim = load_case("chole_tet_med", envs=2)
    c = find(sim, PartContact)
    c.enabled = True
    control = find(sim, InstrumentControl)
    driver = InstrumentDriver(sim, control, None, "scripted")
    total_contacts = 0
    for step in range(132):
        driver.update(step * sim.dt)
        sim.step()
        total_contacts += int(c.contact_count.numpy().sum())
    assert total_contacts > 0
    grasp = find(sim, Grasp)
    assert grasp.count.numpy()[0, 0] > 0
    before = c.q.numpy().copy()
    actions = control.actions.numpy()
    for step in range(30):
        actions[:, 0, 2] += .002  # closed jaw pulls back 6 cm
        control.actions.assign(actions)
        sim.step()
    assert np.isfinite(c.q.numpy()).all()
    assert grasp.count.numpy()[0, 0] > 0
    assert np.linalg.norm(c.q.numpy() - before, axis=-1).max() > .001


@slang_tests
@needs_chole
def test_slang_contact_panel_activation():
    import time

    import warp as wp
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_standalone.app.ui import Panels, SimControls
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshots
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    sim = load_case('chole_tet_med', envs=2)
    c = find(sim, PartContact)
    cfg = read_case('chole_tet_med')['render']
    cfg.pop('backend')
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    renderer.stream, renderer.lock = wp.Stream(sim.scene.device), ring.lock
    driver = InstrumentDriver(sim, find(sim, InstrumentControl), None, 'keyboard', renderer=renderer)
    worker = PhysicsWorker(sim, [driver], ring, steps=0)
    panels = Panels(worker, renderer, SimControls(worker, (0, -9.81, 0)), driver.hub, overrides_path='/dev/null')
    try:
        panels.enabled_boxes[id(c)].callback(True)
        row = next(r for r in panels.rows if r.system is c and r.knob.label == 'stiffness')
        row.widgets[0].callback(.75)
        panels.update()
        worker.pause(True)
        worker.start()
        done = [False]
        worker.edit(lambda s: done.__setitem__(0, True))
        until = time.monotonic() + 10
        while not done[0] and time.monotonic() < until:
            time.sleep(.005)
        worker.check()
        assert done[0]
        for _ in range(3):
            panels.update()
            snap = ring.acquire()
            if snap is not None:
                renderer.publish(snap)
            renderer.draw()
        frame = renderer.read_frame()
        assert np.isfinite(frame).all() and frame.max() > frame.min()
        worker.stop()
        assert c.enabled
        np.testing.assert_allclose(c.stiffness.numpy(), .75)
    finally:
        worker.stop()
        panels.close()
        renderer.close()


@pytest.mark.parametrize('point,bary', [([-.003, .25, -.004], [.75, 0, .25]),
                                       ([-.003, -.003, -.004], [1, 0, 0])])
def test_edge_vertex_closest_point_response(point, bary):
    _, c = fixture()
    # Unit response of one boundary face; other faces excluded from the query.
    owners = c.surface.tri_tet.numpy()
    owners[:, :3] = -1
    c.surface.tri_tet.assign(owners)
    q = c.q.numpy()
    q[:, 4] = point
    c.q.assign(q)
    b = np.array(bary)
    closest = b @ q[0, :3]
    normal = (np.array(point) - closest) / np.linalg.norm(np.array(point) - closest)
    depth = .012 - np.linalg.norm(np.array(point) - closest)
    expected = q.copy()
    # All four participating masses are equal; common inverse mass cancels.
    delta = normal * depth / (1 + (b*b).sum())
    expected[:, 4] += delta
    expected[:, :3] -= b[:, None] * delta
    event(c)
    np.testing.assert_allclose(c.q.numpy(), expected, atol=3e-8)
    assert c.contact_count.numpy().tolist() == [1, 1]


@pytest.mark.parametrize('graph', [False, True])
def test_overflow_full_steps_and_live_shell(graph):
    results = []
    for capacity in (1, 64):
        sim, c = fixture(max_candidates=capacity)
        sim.cuda_graph = graph
        sim.substeps = sim.iterations = 1
        find(sim, Tet).enabled = False
        c.margin.assign(np.array([.002, 0], np.float32))
        c.max_correction.assign(np.array([.001, .0005], np.float32))
        q = c.q.numpy()
        q[:, 4] = [.003, .003, -.003]
        c.q.assign(q)
        sim.prepare()
        overflow = 0
        for _ in range(10):
            sim.step()
            overflow += int(c.overflow_count.numpy().sum())
        assert (overflow > 0) == (capacity == 1)
        results.append(c.q.numpy())
    np.testing.assert_allclose(*results, atol=5e-7)
