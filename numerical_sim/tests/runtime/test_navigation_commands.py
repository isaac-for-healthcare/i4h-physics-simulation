# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Migration fixtures: authored material, external root ownership and wire-only pull."""
import numpy as np
import pytest
import warp as wp
from i4h_endoluminal.loaders.centreline import add_centreline_tree
from i4h_isaaclab.composition import load_case
from i4h_physics import Scene, Stepper
from i4h_physics.loaders.rod import add_rod, material_frames
from i4h_physics.systems.contact.graph_containment import GraphContainment
from test_catheter_port import case, system


def external_case(envs=2, captured=False):
    cfg = case(inertia="physical", tip_mapping="polyline", gravity=(0, 0, 0))
    cfg['sim'].update(envs=envs, cuda_graph=captured)
    cfg['systems'][-1].update(root_mode="external_pose")
    return cfg


@pytest.mark.parametrize('envs', [1, 2, 16])
@pytest.mark.parametrize('captured', [False, True])
def test_root_pose_roll_reset_and_buffer_identity(envs, captured):
    sim = load_case(external_case(envs, captured))
    cc = system(sim, 'CatheterControl')
    rod = system(sim, 'RodDirect')
    pointers = (cc.root_positions.ptr, cc.root_rotations.ptr)
    positions = cc.root_positions.numpy()
    positions[:, 1] += np.arange(envs) * 0.0001 + 0.001
    # Roll around the root's local Z must survive, not just its tangent.
    rotations = np.tile(np.float32([0.3, 0.4, 0, np.sqrt(0.75)]), (envs, 1))
    cc.write_root_poses(positions, rotations)
    cc.actions.fill_(1)
    sim.step()
    np.testing.assert_array_equal(sim.scene.field('q', cc.body).numpy()[:, 0], positions)
    np.testing.assert_allclose(rod.frame.numpy().reshape(envs, -1, 4)[:, 0], rotations, atol=1e-7)
    np.testing.assert_array_equal(cc.targets.numpy()[:, :2], 0)
    assert np.all(cc.targets.numpy()[:, 2] > 0)
    before = sim.scene.field('q', cc.body).numpy().copy()
    sim.reset([0])
    np.testing.assert_array_equal(cc.root_positions.numpy()[0], cc._root_start_positions.numpy()[0])
    np.testing.assert_array_equal(cc.root_rotations.numpy()[0], cc._root_start_rotations.numpy()[0])
    if envs > 1:
        np.testing.assert_array_equal(sim.scene.field('q', cc.body).numpy()[1:], before[1:])
        np.testing.assert_array_equal(cc.root_positions.numpy()[1:], positions[1:])
    assert pointers == (cc.root_positions.ptr, cc.root_rotations.ptr)
    with pytest.raises(ValueError, match='cannot both own'):
        cc.mount()


@pytest.mark.parametrize('kind', ['numpy', 'warp', 'warp_scalar', 'torch'])
def test_pose_inputs_and_complete_batch(kind):
    sim = load_case(external_case())
    cc = system(sim, 'CatheterControl')
    pos, rot = cc.root_positions.numpy(), cc.root_rotations.numpy()
    pos[:, 1] = 0.002
    if kind == 'warp':
        pos, rot = wp.array(pos, dtype=wp.vec3, device='cuda:0'), wp.array(rot, dtype=wp.quat, device='cuda:0')
    if kind == 'warp_scalar':
        pos, rot = wp.array(pos, dtype=wp.float32, device='cuda:0'), wp.array(rot, dtype=wp.float32, device='cuda:0')
    if kind == 'torch':
        import torch

        # Producer stream differs from Warp's; event ordering is part of the API.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            pos = torch.as_tensor(pos, device='cuda:0') + 0.001
            rot = torch.as_tensor(rot, device='cuda:0')
            cc.write_root_poses(pos, rot)
    else:
        cc.write_root_poses(pos, rot)
    sim.step()
    assert np.allclose(sim.scene.field('q', cc.body).numpy()[:, 0, 1], 0.003 if kind == 'torch' else 0.002)
    with pytest.raises(ValueError, match='shape'):
        cc.write_root_poses(np.zeros((1, 3)), np.zeros((1, 4)))
    with pytest.raises(ValueError, match='nonzero'):
        cc.write_root_poses(np.zeros((2, 3)), np.zeros((2, 4)))


def test_curved_startup_is_not_material_shape_and_reset_restores_frames():
    cfg = external_case()
    asset = cfg['assets']['catheter']
    t = np.linspace(0, 0.8, asset['count'])
    positions = np.float32(np.stack([0.5 * np.sin(t), 0.5 * (1-np.cos(t)), np.full_like(t, 0.5)], 1))
    frames, curvature = material_frames(positions)
    asset.update(startup_positions=positions.tolist(), startup_frames=frames.tolist())
    sim = load_case(cfg)
    cc = system(sim, 'CatheterControl')
    np.testing.assert_allclose(cc.body.rod.rest_darboux, 0, atol=1e-7)
    assert np.max(abs(curvature)) > 0.001
    sim.step()
    sim.reset([0])
    np.testing.assert_array_equal(sim.scene.field('q', cc.body).numpy()[0], positions)
    np.testing.assert_allclose(system(sim, 'RodDirect').frame.numpy().reshape(2, -1, 4)[0], frames, atol=1e-7)
    asset['rest_curvature'] = curvature.tolist()
    seeded = load_case(cfg)
    np.testing.assert_array_equal(seeded.scene.body('catheter').rod.rest_darboux[0], curvature)


@pytest.mark.parametrize('field,value', [('startup_positions', [[0,0,0]]), ('startup_frames', [[0,0,0,0]]*3),
                                        ('rest_lengths', [0, 1]), ('rest_curvature', [[np.nan,0,0]]*2)])
def test_invalid_startup_rejected_before_body_allocation(field, value):
    scene = Scene('cuda:0', 1)
    with pytest.raises(ValueError):
        add_rod(scene, 'rod', [0,0,0], [1,0,0], 3, 0.01, 0.001, **{field:value})
    assert not scene.bodies


def test_curved_guide_endpoints_and_independent_free_nodes():
    cfg = external_case()
    cfg['systems'][-1].update(track_path=[[0,0,0.5], [0.1,0,0.5], [0.1,0.1,0.5]],
                                distal_free_nodes=3, track_stiffness=1)
    sim = load_case(cfg)
    cc = system(sim, 'CatheterControl')
    q = sim.scene.field('q', cc.body)
    points = q.numpy()
    points[:, 1] = [0.11, 0.06, 0.5]
    points[:, 2] = [0.1, 0.12, 0.5]  # beyond endpoint: clamp projection to endpoint
    points[:, -3:] = [0.12, 0.06, 0.5]
    q.assign(points)
    cc.iteration(sim.scene, 0.01, 0, 1)
    actual = q.numpy()
    np.testing.assert_allclose(actual[:, 1], [[0.1,0.06,0.5]]*2, atol=1e-7)
    np.testing.assert_allclose(actual[:, 2], [[0.1,0.1,0.5]]*2, atol=1e-7)
    np.testing.assert_array_equal(actual[:, -3:], points[:, -3:])
    np.testing.assert_array_equal(actual[:, 0], points[:, 0])
    cc.track_path = ((0,0,0), (1,0,0))
    with pytest.raises(ValueError, match='fixed at construction'):
        sim.prepare()


@pytest.mark.parametrize('envs', [1, 2, 16])
@pytest.mark.parametrize('candidates,capacity', [(False, 64), (True, 64), (True, 1)])
@pytest.mark.parametrize('offset,contact', [(0.006, False), (0.012, True)])
def test_interior_pull_is_wire_only(envs, candidates, capacity, offset, contact):
    scene = Scene('cuda:0', envs, (0,0,0))
    cath = add_rod(scene, 'wire', [0.02,offset,0], [1,0,0], 5, 0.01, 0.001, mass=1)
    tree = add_centreline_tree(scene, 'vessel', np.float32([[0,0,0],[0.04,0,0],[0.08,0,0]]),
                              np.int32([[0,1],[1,2]]), 0.01, root=0, pinned_nodes=[0], open_root=True)
    scene.allocate()
    control = GraphContainment(scene, body='wire', centreline='vessel', spline=False, candidates=candidates,
                               capacity=capacity, interior_deadband=0.5, interior_stiffness=0.25,
                               iterations=1, source_max_delta=1, target_max_delta=1)
    sim = Stepper.create(scene, [control], dt=0.01, substeps=1, iterations=1)
    before, vessel_before = scene.field('q', cath).numpy(), scene.field('q', tree).numpy()
    sim.step()
    after, vessel_after = scene.field('q', cath).numpy(), scene.field('q', tree).numpy()
    assert np.all(after[:, 1:, 1] < before[:, 1:, 1])
    np.testing.assert_array_equal(after[:, 0], before[:, 0])
    np.testing.assert_array_equal(vessel_after[:, 0], vessel_before[:, 0])
    if contact:
        assert np.any(vessel_after != vessel_before)
        assert np.all(control.contact_count.numpy() > 0)
    else:
        np.testing.assert_array_equal(vessel_after, vessel_before)
        np.testing.assert_array_equal(control.contact_count.numpy(), 0)
        np.testing.assert_array_equal(control.contact_depth.numpy(), 0)
    if candidates and capacity == 1:
        assert np.all(control.overflow.numpy() == 1)


@pytest.mark.parametrize('envs', [1, 2, 16])
def test_external_commands_captured_equal_eager(envs):
    sims = [load_case(external_case(envs, captured)) for captured in (False, True)]
    for sim in sims:
        cc = system(sim, 'CatheterControl')
        position, rotations = cc.root_positions.numpy(), cc.root_rotations.numpy()
        for i in range(12):
            position[:, 0] += 0.0001
            cc.write_root_poses(position, rotations)
            cc.actions.assign(np.tile(np.float32([1, 1, 0.3]), (envs, 1)))
            sim.step()
        assert np.isfinite(sim.scene.q.numpy()).all()
    assert sims[0].graph is None and sims[1].graph is not None
    # Existing aorta captured/eager bounds, not adjusted for this migration.
    np.testing.assert_allclose(sims[0].scene.q.numpy(), sims[1].scene.q.numpy(), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(system(sims[0], 'RodDirect').frame.numpy(),
                               system(sims[1], 'RodDirect').frame.numpy(), atol=1e-3)


def test_default_guide_exclusion_follows_tip_edits():
    cfg = external_case()
    cfg['systems'][-1].update(track_length=1, track_stiffness=1)
    sim = load_case(cfg)
    cc = system(sim, 'CatheterControl')
    cc.tip_edges = 2
    sim.prepare()
    q = sim.scene.field('q', cc.body)
    points = q.numpy()
    points[:, -4, 1] = 0.001
    q.assign(points)
    cc.iteration(sim.scene, 0.01, 0, 1)
    np.testing.assert_array_equal(q.numpy()[:, -4, 1], 0)
