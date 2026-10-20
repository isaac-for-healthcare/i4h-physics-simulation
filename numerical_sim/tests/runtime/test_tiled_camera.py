# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M7 camera: Newton's SensorTiledCamera over a scene-derived render model. Content is checked through
the shape-index image (tissue = triangle hits, instruments = shape ids, fluid and rods = particle hits) and depth
ranges against the scene's own positions; env isolation and masked resets through frame comparisons."""

import numpy as np
import pytest
import warp as wp
from helpers import find, needs_chole, needs_vessels, push_actions, quiet, rest_actions, run, run_rates
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.camera.camera import Camera
from i4h_isaaclab.camera.tiled import TiledCamera
from i4h_isaaclab.composition import load_case, read_case
from i4h_physics.quat import quat_to_matrix
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_standalone.render.instruments import piece_skins, skin_pieces
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.surface import Surface

torch = pytest.importorskip("torch")

PROBE_IN_VIEW = np.array([0.08, -0.10, 0.15], np.float32)  # between the two_worlds camera and the block


def camera_for(sim, case, size=64):
    return TiledCamera(sim, size, size, camera=read_case(case).get("render", {}).get("camera"))


def frames(cam):
    cam.render()
    wp.synchronize_device(cam.scene.device)
    return cam.rgb.numpy().copy(), cam.depth.numpy().copy(), cam.shape_index.numpy().copy()


def differing(depth_a, depth_b, tol=1e-3):
    """Fraction of pixels whose depth differs by more than ``tol``: float-atomic noise in the physics moves a
    silhouette across a pixel now and then, so frames of the same state agree everywhere but a few edge pixels."""
    return float((np.abs(depth_a - depth_b) > tol).mean())


def test_two_worlds_frame_layout_and_content(case):
    sim = load_case(case)
    scene, control = sim.scene, find(sim, InstrumentControl)
    block = scene.body("block")
    cam = camera_for(sim, case)
    cam.prepare()
    actions = push_actions(control, scene.envs)
    actions[0, 0, :3] = PROBE_IN_VIEW
    run(sim, 20, actions)
    rgb, depth, shape = frames(cam)
    assert rgb.shape == (2, 64, 64, 4) and rgb.dtype == np.uint8
    assert depth.shape == (2, 64, 64) and depth.dtype == np.float32 and shape.shape == (2, 64, 64) and shape.dtype == np.uint32
    hit = depth > 0.0
    assert (rgb[..., 3][hit] == 255).all() and (rgb[..., 3][~hit] == 0).all(), "alpha marks hits"
    assert (depth >= 0.0).all() and depth.max() <= 1000.0 and ((shape == cam.MISS) == ~hit).all()
    assert cam.rgb_t.data_ptr() == cam.rgb.ptr and cam.depth_t.data_ptr() == cam.depth.ptr
    target = shape == cam.TRIANGLES
    position = Camera(**read_case(case)["render"]["camera"]).position
    q = scene.view(scene.q, block).numpy()
    active = scene.view(scene.flags, block).numpy() != 0
    for e in range(2):
        assert target[e].mean() > 0.05, f"env {e}: the block is visible"
        verts = q[e][active[e]]
        nearest = np.linalg.norm(np.clip(position, verts.min(0), verts.max(0)) - position)  # distance to the block's box
        d = depth[e][target[e]]
        assert d.min() >= nearest - 1e-3 and d.max() <= np.linalg.norm(verts - position, axis=1).max() + 1e-3
    assert (shape[0] < cam.TRIANGLES).sum() > 0, "env 0's probe (a shape) is in view"
    assert (shape == cam.PARTICLES).sum(axis=(1, 2)).min() > 0, "the dripping water shows as particle hits in both envs"
    water = rgb[shape == cam.PARTICLES][:, :3].astype(np.float32)
    assert water[:, 2].mean() > water[:, 0].mean() + 20.0, "water drops carry the pool's blue, not Newton's white"
    assert not np.array_equal(depth[0], depth[1]), "different block variants and probe poses"


def test_two_worlds_isolation_and_reset(case):
    # No dripping water: splashing drops are chaotic, so two runs' env 1 drift apart at the 1e-5 level within 30 steps
    # and every drop's silhouette flips pixels; the hanging block deforms the same way in both runs.
    sims = [quiet(load_case(case)), quiet(load_case(case))]
    cams = [camera_for(sim, case) for sim in sims]
    for cam in cams:
        cam.prepare()
    initial = [frames(cam) for cam in cams]
    driven, held = (push_actions(find(sim, InstrumentControl), 2) for sim in sims)
    driven[0, 0, :3] = PROBE_IN_VIEW
    driven[0, 0, 8] = 1.0  # energy: the probe sucks water and heats the block where it touches
    run(sims[0], 30, driven)
    run(sims[1], 30, held)
    a, b = frames(cams[0]), frames(cams[1])
    assert differing(a[1][1], b[1][1]) <= 0.002, "env 1 unaffected by env 0's actions (edge pixels aside)"
    assert differing(a[1][0], b[1][0]) > 0.01, "env 0 differs between the driven and the held run"
    # masked reset of env 0: its frame is bitwise the initial one again, env 1's frame bitwise the pre-reset frame
    before = frames(cams[0])
    sims[0].reset([0])
    after = frames(cams[0])
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(after, initial[0])), "env 0 back at its first frame"
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(after, before)), "env 1 untouched by the reset"


def test_captured_render_matches_eager(case):
    sim = load_case(case)
    cam = camera_for(sim, case)
    cam.prepare()
    run(sim, 10, push_actions(find(sim, InstrumentControl), 2))
    captured = frames(cam)
    cam._render()
    wp.synchronize_device(sim.scene.device)
    eager = (cam.rgb.numpy(), cam.depth.numpy(), cam.shape_index.numpy())
    assert all(np.array_equal(x, y) for x, y in zip(captured, eager))


@needs_chole
def test_piece_transforms_match_skinning():
    sim = load_case("chole_tet_med", envs=1)
    scene, control = sim.scene, find(sim, InstrumentControl)
    actions = rest_actions(control, 1)
    actions[:, :, 7] = 1.0  # close every jaw so the hinge term is exercised
    run(sim, 30, actions)
    cam = camera_for(sim, "chole_tet_med", 32)
    cam._update_geometry()
    wp.synchronize_device(scene.device)
    body_q = cam.body_q.numpy().reshape(scene.envs, cam.P + cam.S, 7)
    skins = piece_skins(control, scene.device)
    out = [wp.zeros(s.count, dtype=wp.vec3, device=scene.device) for s in skins]
    skin_pieces(control, 0, skins, out, [wp.zeros_like(o) for o in out], scene.device)
    assert len(skins) == cam.P
    for i, (skin, points) in enumerate(zip(skins, cam.piece_points)):
        t, q = body_q[0, i, :3], body_q[0, i, 3:]
        world = points @ quat_to_matrix(q).T + t
        assert np.allclose(world, out[i].numpy(), atol=1e-4), f"piece {skin.name} of instrument {skin.instrument}"


@needs_chole
def test_chole_frames_follow_fluid_cuts_and_reset():
    sim = load_case("chole_tet_med", envs=2)
    scene, control = sim.scene, find(sim, InstrumentControl)
    surface, fluid = find(sim, Surface), find(sim, Fluid)
    cam = camera_for(sim, "chole_tet_med", 128)
    cam.prepare()
    initial = frames(cam)
    rgb, depth, shape = initial
    for e in range(2):
        assert (shape[e] == cam.TRIANGLES).mean() > 0.1, "tissue fills the view from the case camera"
        assert (shape[e] < cam.TRIANGLES).sum() > 0, "the tools are in view"
    assert (shape == cam.PARTICLES).sum() == 0, "no fluid yet"
    target = rgb[shape == cam.TRIANGLES][:, :3].astype(np.float32)
    assert target.mean() < 200.0 and target.std(axis=0).max() > 10.0, "tissue is coloured and shaded, not flat white"
    assert (target[:, 0] > target[:, 2]).mean() > 0.5, "the liver's red dominates the tissue pixels"
    # irrigation from the suction tube's grip in env 1 shows as particle hits there only (the jet is in view early;
    # later the sagging tissue hides where it lands)
    actions = rest_actions(control, 2)
    s = control.names.index("suction")
    actions[1, s, 7] = 1.0
    run(sim, 30, actions)
    assert fluid.alive_count.numpy()[1].sum() > 0
    wet = frames(cam)
    assert (wet[2][1] == cam.PARTICLES).sum() > 0 and (wet[2][0] == cam.PARTICLES).sum() == 0
    drops = wet[0][1][wet[2][1] == cam.PARTICLES][:, :3].astype(np.float32)
    assert drops[:, 2].mean() > drops[:, 0].mean() + 20.0, "irrigation is blue (the blood pool would be red)"
    # the hook burns and cuts fat in env 0
    actions[1, s, 7] = 0.0
    k = control.names.index("hook")
    fat = scene.part_view(scene.q, scene.body('tissue'), "fat").numpy()[0].mean(axis=0)
    actions[0, k, :3] = fat
    actions[0, k, 8] = actions[0, k, 9] = 1.0
    run(sim, 80, actions)
    assert surface.deleted.numpy()[0] > 0
    cut = frames(cam)
    assert differing(cut[1][0], wet[1][0]) > 0.01, "env 0's frame changed where fat was removed and the hook moved"
    # masked reset of env 0 restores its initial frame bitwise; env 1 keeps its frame bitwise
    sim.reset([0])
    after = frames(cam)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(after, initial)), "env 0 back at its first frame"
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(after, cut)), "env 1 untouched by the reset"


@needs_vessels
def test_aorta_vessel_and_catheter():
    """The inward-facing vessel (``cull: back``) is drawn see-through from outside: Newton culls its near wall and
    stops, our wall pass writes the far wall; the catheter inside is a coloured capsule tube that moves with insertion."""
    sim = load_case("aorta_static", envs=2)
    scene = sim.scene
    cam = camera_for(sim, "aorta_static", 256)  # the tube is 0.02 wide seen from 10 units away: under a pixel at 128
    cam.prepare()
    rgb, depth, shape = frames(cam)
    per_env = cam.static_count + cam.B
    static_ids = [np.arange(e * per_env, e * per_env + cam.static_count) for e in range(2)]
    capsule_ids = [np.arange(e * per_env + cam.static_count + cam.P + cam.S, (e + 1) * per_env) for e in range(2)]
    assert cam.static_count == 1 and cam.C == scene.body("catheter").count - 1 and cam.P == cam.S == 0
    for e in range(2):
        vessel = np.isin(shape[e], static_ids[e])
        tube = np.isin(shape[e], capsule_ids[e])
        assert vessel.mean() > 0.02, "the far wall of the vessel is visible from the case camera"
        assert tube.sum() > 0, "the catheter tube (capsules) is visible through the culled near wall"
        assert (depth[e][vessel] > 0).all() and (rgb[e][vessel][:, 3] == 255).all()
        assert (shape[e] == cam.PARTICLES).sum() == 0, "rod nodes are not drawn as spheres"
        grey = rgb[e][vessel][:, :3].astype(np.float32)
        assert abs(grey[:, 0].mean() - grey[:, 2].mean()) < 8.0, "the vessel keeps its grey colour"
    run_rates(sim, 60, np.float32([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))  # env 0 inserts, env 1 holds
    after = frames(cam)
    assert find(sim, CatheterControl).targets.numpy()[0, 0] > 0.1
    moved = [differing(after[1][e], depth[e], tol=1e-3) for e in range(2)]
    assert moved[0] > 0.0005, "env 0's catheter moved in the frame"
    assert moved[1] <= 0.01, "env 1's thin tube only jitters at silhouette pixels while it settles"


@needs_vessels
def test_aorta_shell_wall_and_catheter():
    """A deformable vessel shell (``cull: back``) is drawn like the static one: its far wall from outside, the catheter
    tube visible through the culled near wall; the wall follows the shell's live positions, per env."""
    sim = load_case("aorta_shell", envs=2)
    scene = sim.scene
    aorta = scene.body("aorta")
    cam = camera_for(sim, "aorta_shell", 256)
    assert len(cam.shells) == 1 and cam.shells[0][3] is aorta and cam.shells[0][2] == 1
    cam.prepare()
    rgb, depth, shape = frames(cam)
    per_env = cam.static_count + cam.B
    capsule_ids = [np.arange(e * per_env + cam.static_count + cam.P + cam.S, (e + 1) * per_env) for e in range(2)]
    walls = []
    for e in range(2):
        wall = shape[e] == cam.TRIANGLES  # aorta_shell has no surface system: every triangle pixel is the shell
        tube = np.isin(shape[e], capsule_ids[e])
        assert wall.mean() > 0.02, "the shell's far wall is visible from the case camera"
        assert tube.sum() > 0, "the catheter tube is visible through the culled near wall"
        assert (depth[e][wall] > 0).all() and (rgb[e][wall][:, 3] == 255).all()
        grey = rgb[e][wall][:, :3].astype(np.float32)
        assert abs(grey[:, 0].mean() - grey[:, 2].mean()) < 8.0, "the shell keeps its grey colour"
        walls.append(wall)
    q = scene.field("q", aorta)
    moved = q.numpy()
    moved[0] += np.float32([0.0, 0.0, 0.5])  # env 0's wall only
    q.assign(moved)
    after = frames(cam)
    assert differing(after[1][0][walls[0]], depth[0][walls[0]]) > 0.05, "env 0's wall moved with its nodes"
    assert differing(after[1][1], depth[1]) == 0.0, "env 1's wall did not"


def host_tip_transform(forward, hint_y, hint_x, world_up, tip, distance):
    from i4h_isaaclab.camera.camera import stable_frame

    right, up, f = stable_frame(forward, hint_y, hint_x, world_up)
    q = __import__("i4h_physics.quat", fromlist=["matrix_to_quat"]).matrix_to_quat(np.stack([right, up, -f], axis=1))
    return np.concatenate([tip - f * distance, q]).astype(np.float32)


@needs_vessels
def test_tiled_tip_camera_per_env_rods():
    """aorta, two envs inserting at different rates: each env's camera sits at its own catheter tip with the rod frame's
    roll; graph replay == eager. (Tree catheters, as in centreline-tree cases, are not drawn by the tiled camera: recorded at M9a.)"""
    from i4h_physics.quat import quat_rotate
    from i4h_physics.systems.rods.rod_direct import RodDirect
    from i4h_physics.systems.rods.rod_pbd import RodPbd

    sim = load_case("aorta_static", envs=2)
    scene = sim.scene
    cam = TiledCamera(sim, 32, 32, camera=read_case("aorta_static")["render"]["camera"], camera_mode="tip", tip={"body": "catheter"})
    assert cam.tip is not None and cam.fov == 100.0
    cam.prepare()
    run_rates(sim, 120, np.float32([[1.0, 0.0, 0.0], [0.4, 0.0, 0.0]]))
    cam.render()
    wp.synchronize_device(scene.device)
    xf = cam.camera_transforms.numpy()[0]
    rod = next(s for s in sim.systems if isinstance(s, (RodDirect, RodPbd)))
    body, P, F = rod.body, rod.body.count, rod.frame_count
    q = scene.view(scene.q, body).numpy()
    frames = rod.frame.numpy()
    for env in range(2):
        fq = frames[env * F + min(P - 1, F - 1)]
        expect = host_tip_transform(q[env, P - 1] - q[env, P - 2], quat_rotate(fq, (0.0, 1.0, 0.0)), quat_rotate(fq, (1.0, 0.0, 0.0)),
                                    -scene.down, q[env, P - 1], 0.04)
        got = np.asarray(xf[env], np.float32).reshape(7)
        assert np.allclose(got[:3], expect[:3], atol=1e-4)
        assert np.allclose(got[3:], expect[3:], atol=1e-4) or np.allclose(got[3:], -expect[3:], atol=1e-4)
    assert not np.allclose(xf[0], xf[1], atol=1e-3)  # two insertions, two poses
    replay = cam.rgb.numpy().copy()
    cam._render()
    wp.synchronize_device(scene.device)
    assert np.array_equal(cam.rgb.numpy(), replay)


def test_tiled_tip_camera_per_env_instruments():
    sim = load_case("two_worlds", envs=2)
    scene = sim.scene
    control = find(sim, InstrumentControl)
    cam = TiledCamera(sim, 32, 32, camera=read_case("two_worlds")["render"]["camera"], camera_mode="tip", tip={"instrument": "probe", "distance": 0.1})
    cam.prepare()
    actions = control.actions.numpy()
    actions[0, 0, 3:7] = [0.0, np.sin(0.5), 0.0, np.cos(0.5)]
    actions[1, 0, 3:7] = [np.sin(0.3), 0.0, 0.0, np.cos(0.3)]
    for _ in range(3):
        control.actions.assign(actions)
        sim.step()
    cam.render()
    wp.synchronize_device(scene.device)
    xf = cam.camera_transforms.numpy()[0]
    roots, shafts = control.root.numpy(), control.shaft_q.numpy()
    from i4h_physics.quat import quat_rotate

    for env in range(2):
        sq = shafts[env, 0]
        f = quat_rotate(sq, (0.0, 0.0, -1.0))
        expect = host_tip_transform(f, quat_rotate(sq, (0.0, 1.0, 0.0)), quat_rotate(sq, (1.0, 0.0, 0.0)), -scene.down, roots[env, 0], 0.1)
        got = np.asarray(xf[env], np.float32).reshape(7)
        assert np.allclose(got[:3], expect[:3], atol=1e-5)
        assert np.allclose(got[3:], expect[3:], atol=1e-4) or np.allclose(got[3:], -expect[3:], atol=1e-4)
    assert not np.allclose(xf[0], xf[1], atol=1e-3)
    with pytest.raises(ValueError, match="camera_mode"):
        TiledCamera(sim, 8, 8, camera_mode="orbit")
