# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regional metadata, mechanics, graph edits and transactional persistence."""

import copy

import numpy as np
import pytest
import warp as wp
from helpers import needs_external_data, regional_case
from i4h_endoluminal.loaders.centreline import add_centreline_tree, build_tree
from i4h_endoluminal.loaders.rod import add_rod
from i4h_endoluminal.regions import instrument_regions, vessel_regions
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import read_case
from i4h_physics import Scene, Stepper
from i4h_physics.regions import RegionMembership
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.rods.material import RodDirectMaterial, RodPbdMaterial
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd
from i4h_standalone.app.panel_adapters import panel
from i4h_standalone.app.ui import load_overrides
from i4h_standalone.app.workspace import resolve_workspace
from i4h_standalone.config import load_case

INSTRUMENT = {'schema_version': 1, 'default': 'unassigned', 'regions': [
    {'name': 'shaft', 'interval': [0, .8]}, {'name': 'transition', 'interval': [.8, .9]}, {'name': 'tip', 'interval': [.9, 1]}]}


def rod_sim(regional=True, graph=True):
    scene = Scene('cuda:0', 2, (0, 0, 0))
    body = add_rod(scene, 'wire', [0, 0, 0], [1, 0, 0], 21, .02, .002,
                   regions=INSTRUMENT if regional else None)
    scene.allocate()
    materials = [RodDirectMaterial(scene, body='wire', id='material')] if regional else []
    rod = RodDirect(scene, body='wire', id='rod', angular_damping=.05)
    return Stepper.create(scene, [Particles(scene), *materials, rod], dt=.01, substeps=4, iterations=2, cuda_graph=graph), rod, body


def test_instrument_material_coordinates_and_invalid_intervals():
    for n in (10, 20, 100):
        m = instrument_regions(INSTRUMENT, np.ones(n))
        assert m.names == ('unassigned', 'shaft', 'transition', 'tip')
        assert np.count_nonzero(m.edge == 1) == n * .8
        assert np.count_nonzero(m.edge == 3) == n * .1
        np.testing.assert_array_equal(m.edge, instrument_regions(INSTRUMENT, np.ones(n) * 7).edge)
    for interval in ([.7, .85], [-.1, .1], [1, 0], [0, np.nan]):
        invalid = copy.deepcopy(INSTRUMENT)
        invalid['regions'][1]['interval'] = interval
        with pytest.raises(ValueError):
            instrument_regions(invalid, np.ones(10))


def test_weld_dedup_filter_stitch_provenance():
    starts = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [2.01, 0, 0], [9, 0, 0]]
    ends = [[1, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [9, 0, 0]]
    args = (starts, ends, np.ones(5), np.ones(5))
    # Degenerate isolated node would remain a disconnected component during stitching; keep it near the graph.
    starts[-1] = ends[-1] = [1, 0, 0]
    result = build_tree(*args, sources=[10, 11, 10, 12, 10], branch_regions={10: 1, 11: 1, 12: 2}, stitch_threshold=.03)
    _, edges, _, _, _, sources, regions = result
    assert len(edges) == 4
    assert (10, 11) in sources and () in sources
    assert regions[sources.index(())] == 0
    with pytest.raises(ValueError, match='conflicting'):
        build_tree(*args, sources=[10, 11, 10, 12, 10], branch_regions={10: 1, 11: 2, 12: 2})
    with pytest.raises(ValueError, match='unknown branch'):
        vessel_regions({'schema_version': 1, 'branches': {99: 'a'}}, {10})


def test_direct_compliance_inputs_and_env_isolation():
    sim, rod, body = rod_sim()
    props = body.region_properties
    props.edit({'young_multiplier': 2, 'bend_multiplier': 3, 'torsion_multiplier': 4, 'twist_multiplier': 5}, 'tip', [1])
    sim.step()
    c = rod.compliance.numpy().reshape(2, 20, 6)
    tip = body.regions.edge == 3
    dt = sim.dt / sim.substeps
    expected = 1 / (1e6 * 6 * body.rod.rest_len[0, tip] * dt**2 + 1e-10)
    np.testing.assert_allclose(c[1, tip, 3], expected, rtol=2e-6)
    np.testing.assert_allclose(c[1, tip, 5], 1 / (1e6 * 20 * body.rod.rest_len[0, tip] * dt**2 + 1e-10), rtol=2e-6)
    assert (c[:, :, :3] == np.float32(1e-10)).all()
    assert (props.tables['young_multiplier'].numpy()[0] == 1).all()


@pytest.mark.parametrize('graph', [False, True])
def test_default_and_zero_curvature_preserve_direct_path(graph):
    plain, pr, pb = rod_sim(False, graph)
    regional, rr, rb = rod_sim(True, graph)
    for sim, rod in ((plain, pr), (regional, rr)):
        q = sim.scene.field('q', rod.body).numpy().copy()
        q[:, -1, 1] += .01
        sim.scene.field('q', rod.body).assign(q)
        for _ in range(10):
            sim.step()
    np.testing.assert_allclose(plain.scene.field('q', pb).numpy(), regional.scene.field('q', rb).numpy(), atol=1e-7)
    np.testing.assert_allclose(pr.compliance.numpy(), rr.compliance.numpy(), rtol=1e-6)


def test_curvature_relaxation_steering_and_reset_geometry():
    sim, rod, body = rod_sim()
    original = sim.scene.field('q', body).numpy().copy()
    body.region_properties.edit({'curvature_x': 6}, 'tip', [1])
    np.testing.assert_array_equal(sim.scene.field('q', body).numpy(), original)
    rod.bend_offset.fill_(wp.vec3(0, .01, 0))
    for _ in range(60):
        sim.step()
    preferred = rod.preferred_rest.numpy().copy()
    assert abs(preferred[1, -1, 0]) > .05
    assert np.max(np.abs(sim.scene.field('q', body).numpy()[1] - sim.scene.field('q', body).numpy()[0])) > 1e-4
    rod.bend_offset.zero_()
    sim.step()
    np.testing.assert_array_equal(rod.preferred_rest.numpy(), preferred)
    sim.reset([1])
    np.testing.assert_array_equal(sim.scene.field('q', body).numpy()[1], original[1])
    np.testing.assert_array_equal(rod.frame.numpy().reshape(2, 21, 4)[1], body.rod.frames[0])
    assert not rod.bend_offset.numpy()[1].any()
    assert body.region_properties.tables['curvature_x'].numpy()[1, -1] == 6


def test_insertion_does_not_change_material_membership():
    sim, rod, body = rod_sim()
    control = CatheterControl(sim.scene, rod=rod, track_length=.1)
    sim.systems.append(control)
    sim.prepare()
    membership = body.regions.edge.copy()
    control.actions.fill_(.5)
    for _ in range(10):
        sim.step()
    np.testing.assert_array_equal(body.regions.edge, membership)


def pbd_sim(radius_scale=1, anatomical_scale=1):
    scene = Scene('cuda:0', 2, (0, 0, 0))
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
    body = add_centreline_tree(scene, 'vessel', positions, [[0, 1], [1, 2]], np.array([1, 2, 3])*radius_scale, 0)
    body.tree.material_radius = np.array([[1, 2, 3]], np.float32)*anatomical_scale
    body.regions = RegionMembership(('a', 'b'), np.array([0, 1], np.int32))
    scene.allocate()
    material = RodPbdMaterial(scene, body='vessel', id='material')
    rod = RodPbd(scene, body='vessel', id='pbd')
    sim = Stepper.create(scene, [Particles(scene), material, rod], dt=.01)
    return sim, rod, body


def test_pbd_harmonic_layout_radius_weighting():
    sim, rod, body = pbd_sim()
    props = body.region_properties
    props.edit({'bend': 2, 'twist': 0, 'stretch': 3}, 'a')
    props.edit({'bend': 6, 'twist': 4}, 'b')
    rod.bend_scale.fill_(.5)
    rod.stretch_scale.fill_(.25)
    sim.step()
    np.testing.assert_allclose(rod.effective_bend.numpy(), 1.5)
    assert not rod.effective_twist.numpy().any()
    np.testing.assert_allclose(rod.effective_stretch.numpy(), [[.75, .25]]*2)
    next(s for s in sim.systems if isinstance(s, RodPbdMaterial)).radius_weighting = True
    with pytest.raises(RuntimeError):
        sim.step()
    sim.prepare()
    sim.step()
    reference = rod.effective_bend.numpy()
    for rs, a in ((9, 1), (1, 8)):
        other, r, b = pbd_sim(rs, a)
        b.region_properties.edit({'bend': 2}, 'a')
        b.region_properties.edit({'bend': 6}, 'b')
        r.bend_scale.fill_(.5)
        next(s for s in other.systems if isinstance(s, RodPbdMaterial)).radius_weighting = True
        other.prepare()
        other.step()
        np.testing.assert_allclose(r.effective_bend.numpy(), reference, rtol=1e-6)
    sim.reset([1])
    assert props.tables['bend'].numpy()[1, 0] == 2


def test_atomic_overrides_and_restore_defaults():
    sim, rod, body = rod_sim()
    data = {'regions': {'wire': {'rod': {'tip': {'curvature_x': [0, 5], 'bend_multiplier': [1, 3]}}}}}
    load_overrides(sim, data)
    assert body.region_properties.tables['curvature_x'].numpy()[1, -1] == 5
    for changes in ({'curvature_x': 200}, {'bend_multiplier': 11}, {'curvature_x': [1]}, {'unknown': 2}):
        invalid = {'systems': {'rod': {'young': 1234}}, 'regions': {'wire': {'rod': {'tip': changes}}}}
        with pytest.raises(ValueError):
            load_overrides(sim, invalid)
        assert (rod.young.numpy() == 1e6).all()
        assert body.region_properties.tables['curvature_x'].numpy()[1, -1] == 5
    body.region_properties.restore([1])
    assert not body.region_properties.tables['curvature_x'].numpy().any()


@pytest.mark.parametrize('backend', ['direct', 'pbd'])
def test_eager_and_capture_regional_edits(backend):
    sims = []
    for graph in (False, True):
        sim, rod, body = rod_sim(graph=graph) if backend == 'direct' else pbd_sim()
        sim.cuda_graph = graph
        sim.prepare()
        if backend == 'direct':
            body.region_properties.edit({'curvature_y': 5, 'young_multiplier': .7}, 'tip', [1])
        else:
            body.region_properties.edit({'bend': 2, 'twist': .3, 'stretch': .7}, 'a', [1])
        for _ in range(20):
            sim.step()
        sims.append(sim)
    np.testing.assert_allclose(sims[0].scene.q.numpy(), sims[1].scene.q.numpy(), atol=1e-7)


@needs_external_data
def test_workspace_validation_and_wrapper():
    sim, rod, body = rod_sim()
    for declaration in ({'groups': [{'knobs': [['rod', 'missing']]}]}, {'region_bodies': ['missing']}, {'actions': ['unknown']}):
        with pytest.raises((ValueError, KeyError)):
            resolve_workspace(sim, declaration)
    cfg = read_case({'case': 'aorta_centreline', 'ui': {'title': 'Wrapper', 'groups': []}})
    assert cfg['ui']['title'] == 'Wrapper'
    s = load_case(cfg, envs=2)
    assert s.workspace['title'] == 'Wrapper'


def test_layout_disable_and_masked_reset_preserve_regions():
    from i4h_endoluminal.systems.rods.control_layout import ControlLayout
    sim, rod, body = pbd_sim()
    body.anatomy.layouts = [[{'pid': 1, 'position': [1, 0, 0], 'rotation': [0, 0, 0, 1],
                          'stiffness': 0, 'falloff': 0, 'rest_bend': [0, 0], 'active': True,
                          'bend_stiffness_multiplier': 2, 'stretch_stiffness_multiplier': .5}]]
    control = ControlLayout(sim.scene, centreline='vessel')
    sim.systems.append(control)
    body.region_properties.edit({'bend': 3, 'stretch': 4}, 'a', [1])
    sim.prepare()
    sim.step()
    assert control.controls is body.rod_controls
    np.testing.assert_allclose(rod.effective_bend.numpy()[:, 0], [2, 3])
    control.enabled = False
    sim.prepare()
    sim.step()
    np.testing.assert_allclose(rod.effective_bend.numpy()[:, 0], [1, 1.5])
    sim.reset([1])
    assert body.region_properties.tables['bend'].numpy()[1, 0] == 3
    np.testing.assert_array_equal(rod.bend_scale.numpy(), 1)
    # Legacy references still receive construction-time validation.
    with pytest.raises(TypeError):
        ControlLayout(sim.scene, centreline='vessel', rod=object())


@needs_external_data
def test_pbd_unit_regions_match_existing_path(tmp_path):
    # Compare both backends in a real regional case, with and without sidecars. The catheter starts against the
    # aorta wall, and contact accumulation is not bitwise reproducible, so both runs leave containment out.
    from i4h_isaaclab.composition import read_case
    case = str(regional_case(tmp_path))

    def contact_free(regions):
        cfg = read_case(case)
        cfg.pop('ui')
        removed = ('lumen_containment',) if regions else ('lumen_containment', 'rod_pbd_material', 'rod_direct_material')
        cfg['systems'] = [s for s in cfg['systems'] if s['type'] not in removed]
        if not regions:
            cfg['assets']['catheter'].pop('regions')
            cfg['assets']['aorta']['centreline'].pop('regions')
        return cfg

    unconfigured = load_case(contact_free(False), envs=2)
    configured = load_case(contact_free(True), envs=2)
    for sim in (unconfigured, configured):
        for _ in range(4):
            sim.step()
    np.testing.assert_allclose(unconfigured.scene.q.numpy(), configured.scene.q.numpy(), atol=1e-7)


def test_radius_weighting_without_region_configuration():
    scene = Scene('cuda:0', 2, (0, 0, 0))
    body = add_centreline_tree(scene, 'vessel', [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1], [1, 2]], [1, 2, 3], 0)
    scene.allocate()
    material = RodPbdMaterial(scene, body='vessel', id='material')
    rod = RodPbd(scene, body='vessel')
    sim = Stepper.create(scene, [Particles(scene), material, rod])
    next(s for s in sim.systems if isinstance(s, RodPbdMaterial)).radius_weighting = True
    sim.prepare()
    sim.step()
    assert body.regions.names == ('unassigned',)
    a = (1.5 / 2.5)**3
    expected = 2*a/(1+a)
    np.testing.assert_allclose(rod.effective_bend.numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(rod.effective_twist.numpy(), expected, rtol=1e-6)
    next(s for s in sim.systems if isinstance(s, RodPbdMaterial)).radius_weighting = False
    sim.prepare()
    sim.step()
    np.testing.assert_array_equal(rod.effective_bend.numpy(), 1)
    np.testing.assert_array_equal(rod.effective_twist.numpy(), 1)


@pytest.mark.parametrize('backend', ['direct', 'pbd'])
@pytest.mark.parametrize('graph', [False, True])
def test_material_disable_enable_persistence(backend, graph):
    sim, rod, body = rod_sim(graph=graph) if backend == 'direct' else pbd_sim()
    sim.cuda_graph = graph
    material = next(s for s in sim.systems if isinstance(s, (RodDirectMaterial, RodPbdMaterial)))
    field = 'bend_multiplier' if backend == 'direct' else 'bend'
    region = 'tip' if backend == 'direct' else 'a'
    body.region_properties.edit({field: 3}, region, [1])
    if backend == 'pbd':
        rod.bend_scale.fill_(.5)
    rod.bend_offset.fill_(wp.vec3(0, .01, 0))
    sim.step()
    controls = body.rod_controls
    material.enabled = False
    sim.prepare()
    sim.step()
    assert not rod.regional and body.rod_controls is controls
    np.testing.assert_array_equal(rod.bend_offset.numpy(), np.tile([0, .01, 0], (2, rod.bend_count, 1)).astype(np.float32))
    if backend == 'pbd':
        np.testing.assert_array_equal(rod.effective_bend.numpy(), .5)
    else:
        assert rod.preferred_rest is rod.rest_darboux
        assert (rod.compliance.numpy().reshape(2, rod.Ed, 6)[0] == rod.compliance.numpy().reshape(2, rod.Ed, 6)[1]).all()
    sim.reset([1])
    assert body.region_properties.tables[field].numpy()[1, body.regions.names.index(region)] == 3
    material.enabled = True
    sim.prepare()
    sim.step()
    assert rod.regional
    if backend == 'pbd':
        assert rod.effective_bend.numpy()[1, 0] > rod.effective_bend.numpy()[0, 0]
    else:
        assert rod.compliance.numpy().reshape(2, rod.Ed, 6)[1, -1, 3] < rod.compliance.numpy().reshape(2, rod.Ed, 6)[0, -1, 3]


def test_explicit_material_validation_before_capture():
    scene = Scene('cuda:0', 2, (0, 0, 0))
    body = add_rod(scene, 'wire', [0, 0, 0], [1, 0, 0], 5, .02, .002, regions=INSTRUMENT)
    scene.allocate()
    with pytest.raises(ValueError, match='requires'):
        RodDirect(scene, body='wire')
    provider = RodDirectMaterial(scene, body='wire', id='material')
    with pytest.raises(ValueError, match='duplicate'):
        RodDirectMaterial(scene, body='wire')
    with pytest.raises(ValueError, match='mismatched'):
        RodPbd(scene, body='wire')
    rod = RodDirect(scene, body='wire', id='rod')
    for systems in ([rod, provider], [rod], [provider], [provider, provider, rod]):
        sim = Stepper.create(scene, systems)
        with pytest.raises(ValueError):
            sim.prepare()
        assert sim.graph is None
    sim = Stepper.create(scene, [provider, rod])
    body.rod_material.bend_multiplier = wp.ones((2, 1), dtype=float, device=scene.device)
    with pytest.raises(ValueError, match='dimensions'):
        sim.prepare()


def test_legacy_resolution_and_conflicts_are_transactional():
    sim, rod, body = rod_sim()
    resolved = resolve_workspace(sim, {'groups': [{'knobs': [['rod', 'region_curvature_x'], ['rod', 'young']]}]})
    assert resolved['groups'][0]['knobs'] == [['material', 'region_curvature_x'], ['rod', 'young']]
    load_overrides(sim, {'regions': {'wire': {'rod': {'tip': {'curvature_x': 2}},
                                           'material': {'tip': {'bend_multiplier': 3}}}}})
    assert body.region_properties.tables['curvature_x'].numpy()[0, -1] == 2
    assert body.region_properties.tables['bend_multiplier'].numpy()[0, -1] == 3
    before = {n: a.numpy().copy() for n, a in body.region_properties.tables.items()}
    with pytest.raises(ValueError, match='conflicting'):
        load_overrides(sim, {'systems': {'rod': {'young': 1234}}, 'regions': {'wire': {
            'rod': {'tip': {'curvature_x': 4}}, 'material': {'tip': {'curvature_x': 5}}}}})
    np.testing.assert_array_equal(rod.young.numpy(), 1e6)
    for n, a in body.region_properties.tables.items():
        np.testing.assert_array_equal(a.numpy(), before[n])
    sim, rod, body = pbd_sim()
    load_overrides(sim, {'systems': {'pbd': {'radius_weighting': True, 'bend': 2}}})
    provider = next(s for s in sim.systems if isinstance(s, RodPbdMaterial))
    assert provider.radius_weighting
    np.testing.assert_array_equal(rod.bend.numpy(), 2)
    resolved = resolve_workspace(sim, {'groups': [{'knobs': [['pbd', 'radius_weighting']]}]})
    assert resolved['groups'][0]['knobs'] == [['material', 'radius_weighting']]
    with pytest.raises(ValueError, match='conflicting'):
        load_overrides(sim, {'systems': {'pbd': {'radius_weighting': False, 'bend': 3},
                                        'material': {'radius_weighting': True}}})
    assert provider.radius_weighting
    np.testing.assert_array_equal(rod.bend.numpy(), 2)
    assert not any(k.scope for k in panel(rod))
    assert 'radius_weighting' not in {k.label for k in panel(rod)}
    assert {k.label for k in panel(provider)} == {'region_stretch', 'region_bend', 'region_twist', 'radius_weighting'}
