# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Renderer plumbing: camera basis, render config, noop backend, CLI loop; Slang offscreen smoke test on demand."""

import gc
import math
import os
import pathlib
import weakref

import numpy as np
import pytest
import warp as wp
from helpers import needs_chole, needs_external_data, slang_tests
from i4h_isaaclab.camera.camera import Camera
from i4h_isaaclab.composition import data_dir, load_case, read_case
from i4h_standalone.app.cli import main
from i4h_standalone.render import NoopRenderer, make_renderer

try:
    import slangpy  # noqa: F401

    HAVE_SLANG = True
except ImportError:
    HAVE_SLANG = False


def assert_orthonormal(right, up, forward):
    for v in (right, up, forward):
        assert math.isclose(float(np.linalg.norm(v)), 1.0, abs_tol=1e-5)
    assert abs(float(np.dot(right, up))) < 1e-5 and abs(float(np.dot(right, forward))) < 1e-5
    assert np.allclose(np.cross(right, forward), up, atol=1e-5)  # right-handed: right x forward = up


def test_camera_basis_y_up():
    cam = Camera(position=(0.0, 0.0, 0.0), pitch=0.0, yaw=0.0)
    _, right, up, forward = cam.basis()
    assert np.allclose(forward, (0.0, 0.0, -1.0), atol=1e-6) and np.allclose(right, (1.0, 0.0, 0.0), atol=1e-6)
    assert_orthonormal(right, up, forward)
    chole = Camera(position=(0.469625, 1.20226, -2.45614), pitch=6.8, yaw=4.3, fov=45.0)
    _, right, up, forward = chole.basis()
    assert math.isclose(float(forward[1]), math.sin(math.radians(6.8)), abs_tol=1e-6)
    assert math.isclose(float(forward[0]), math.cos(math.radians(6.8)) * math.sin(math.radians(4.3)), abs_tol=1e-6)
    assert_orthonormal(right, up, forward)


def test_camera_basis_z_up():
    cam = Camera(position=(0.15, -0.2, 0.17), pitch=-18.0, yaw=-37.0, up=(0.0, 0.0, 1.0))
    _, right, up, forward = cam.basis()
    assert math.isclose(float(forward[2]), math.sin(math.radians(-18.0)), abs_tol=1e-6)
    assert float(np.dot(up, (0.0, 0.0, 1.0))) > 0.9
    assert_orthonormal(right, up, forward)
    cam.orbit(80.0, -30.0)
    _, right, up, forward = cam.basis()
    assert_orthonormal(right, up, forward)


def test_camera_rejects_bad_clip():
    with pytest.raises(ValueError):
        Camera(near=0.0)
    with pytest.raises(ValueError):
        Camera(near=1.0, far=0.5)


@needs_chole
def test_read_case_render():
    cfg = read_case("chole_tet_med")
    render = cfg["render"]
    assert render["backend"] == "slang" and render["camera"]["up"] == [0.0, 1.0, 0.0]
    assert "${DATA}" not in render["environment"]["path"] and render["environment"]["path"].endswith("abdomen.hdr")
    liver = cfg["assets"]['tissue']["parts"]["liver"]["visual"]
    assert liver["set"] == str(data_dir() / "textures" / "liver")  # the authored base, damage, coag and blood maps
    assert cfg["assets"]["cavity"]["type"] == "static_mesh"


def test_noop_renderer_steps():
    sim = load_case("two_worlds")
    renderer = make_renderer({**read_case("two_worlds")["render"], "backend": "noop"}, sim)  # as --render noop
    assert isinstance(renderer, NoopRenderer)
    sim.prepare()
    for _ in range(3):
        sim.step()
        renderer.publish()
        assert renderer.draw() and renderer.is_running()
    assert renderer.read_frame() is None
    renderer.close()
    with pytest.raises(ValueError, match="render.backend"):
        make_renderer({"backend": "rtx"}, sim)


def test_cli_runs():
    assert main(["two_worlds", "--steps", "5", "--render", "noop"]) == 0
    with pytest.raises(SystemExit):  # --steps 0 needs a window
        main(["two_worlds", "--steps", "0", "--render", "noop"])


@pytest.mark.skipif(not HAVE_SLANG, reason="slangpy not installed")
@needs_chole
def test_texture_values_preserved():
    """Loading must not gamma-convert: normal maps are data (mean xy ~ 127.5), only the GPU format decodes diffuse."""
    import slangpy as spy
    from i4h_standalone.render.slang import read_rgba

    path = data_dir() / "textures" / "liver" / "normal-base.png"
    rgba = read_rgba(path)
    source = np.asarray(spy.Bitmap(str(path)).convert(pixel_format=spy.Bitmap.PixelFormat.rgba,
                                                      component_type=spy.Bitmap.ComponentType.uint8))
    assert rgba.dtype == np.uint8 and rgba.shape == source.shape and rgba.shape[2] == 4
    assert np.array_equal(rgba, source[::-1])  # rows flipped, values untouched
    assert abs(float(rgba[..., :2].mean()) - 127.5) < 2.0 and float(rgba[..., 2].mean()) > 250.0


@slang_tests
@needs_chole
def test_slang_offscreen_chole():
    from i4h_standalone.render.slang import SlangRenderer

    sim = load_case("chole_tet_med", envs=2)
    cfg = read_case("chole_tet_med")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    sim.prepare()
    for _ in range(3):
        sim.step()
        renderer.publish()
        assert renderer.draw()
    frame = renderer.read_hdr()
    assert frame.shape == (180, 320, 4) and frame.dtype == np.float32 and np.isfinite(frame).all()
    display = renderer.read_frame()
    assert display.shape == frame.shape and display.dtype == np.float32 and 0.0 <= display.min() and display.max() <= 1.0
    differs = (np.abs(frame[..., :3] - np.float32(renderer.clear[:3])) > 1e-3).any(axis=-1)
    assert differs.mean() > 0.05
    rgb = frame[..., :3].mean(axis=(0, 1))
    assert rgb[0] > rgb[2]  # tissue and cavity are red
    body_draw = renderer.bodies[0]
    renderer.set_env(1)  # same mesh variant: buffers are kept, only the source slice changes
    assert renderer.bodies[0] is body_draw and renderer.env == 1
    assert renderer.draw()
    assert np.allclose(renderer.read_hdr(), frame, atol=0.05)  # identical envs draw the same picture
    assert len(renderer.pieces) == sum(len(r.pieces) for r in sim.scene.instruments.values()) == 16  # every tool's pieces, drawn with the pbr pipeline
    renderer.debug_spheres = True  # the collision-sphere overlay changes the picture
    renderer.publish()
    assert renderer.draw()
    assert not np.allclose(renderer.read_frame(), frame, atol=0.05)
    renderer.close()


@slang_tests
def test_slang_offscreen_follows_cut_and_burn():
    """The index buffer is the surface system's live triangle list and the colour stream its visual state."""
    import warp as wp
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
    from i4h_surgical.systems.mechanics.surface import Surface

    sim = load_case("two_worlds")
    scene = sim.scene
    control, surface = (next(s for s in sim.systems if isinstance(s, cls)) for cls in (InstrumentControl, Surface))
    cfg = read_case("two_worlds")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    body_draw = renderer.bodies[0]
    assert body_draw.surface is surface and body_draw.indices is not None and body_draw.colors is not None
    sim.prepare()
    actions = np.zeros((scene.envs, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    actions[0, 0, :3] = [0.0, 0.0, 0.045]  # into the bottom face of env 0's block
    for _ in range(10):
        control.actions.assign(actions)
        sim.step()
    renderer.publish()
    assert renderer.draw()
    before = renderer.read_frame()
    assert np.array_equal(body_draw.indices.array.numpy(), surface.tris.numpy()[0])
    actions[0, 0, 8] = 1.0  # energy: burn
    actions[0, 0, 9] = 1.0  # fire: cut
    for _ in range(40):
        control.actions.assign(actions)
        sim.step()
    assert surface.deleted.numpy()[0] > 0
    renderer.publish()
    assert renderer.draw()
    after = renderer.read_frame()
    assert np.isfinite(after).all() and not np.allclose(after, before, atol=0.02)
    wp.synchronize_device(scene.device)
    assert np.array_equal(body_draw.indices.array.numpy(), surface.tris.numpy()[0])
    colors = body_draw.colors.array.numpy()
    assert colors[:, 1].max() > 0.0 and np.array_equal(colors, surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)[0])
    renderer.set_env(1)  # the other variant: rebuilt buffers bound to env 1's untouched surface
    assert renderer.bodies[0] is not body_draw and renderer.draw()
    assert np.array_equal(renderer.bodies[0].indices.array.numpy(), surface.tris.numpy()[1])
    renderer.close()


@slang_tests
def test_slang_offscreen_two_worlds_variant():
    from i4h_standalone.render.slang import SlangRenderer

    sim = load_case("two_worlds")
    cfg = read_case("two_worlds")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, env=1, size=(320, 180), window=False))
    sim.prepare()
    sim.step()
    renderer.publish()
    assert renderer.draw()
    frame = renderer.read_hdr()
    differs = (np.abs(frame[..., :3] - np.float32(renderer.clear[:3])) > 1e-3).any(axis=-1)
    assert np.isfinite(frame).all() and 0.01 < differs.mean() < 0.5
    body_draw = renderer.bodies[0]
    renderer.set_env(0)  # different variant: buffers rebuilt; the larger block covers more pixels
    assert renderer.bodies[0] is not body_draw
    assert renderer.draw()
    frame0 = renderer.read_hdr()
    differs0 = (np.abs(frame0[..., :3] - np.float32(renderer.clear[:3])) > 1e-3).any(axis=-1)
    assert differs0.mean() > differs.mean()
    renderer.close()
    assert renderer.device is None and not renderer.is_running()
    ref = weakref.ref(renderer)
    del renderer, body_draw
    gc.collect()
    assert ref() is None  # close() unregistered the atexit hook, nothing pins the renderer


@slang_tests
def test_slang_offscreen_fluid():
    """The four fluid passes run every frame; drops change the pixels where they fall and nowhere else much."""
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_surgical.systems.fluids.emitter import Emitter

    sim = load_case("two_worlds")
    cfg = read_case("two_worlds")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, fluid={"blur": 0.01}))
    assert renderer.fluid_passes is not None and len(renderer.fluids) == 1
    emitter = next(s for s in sim.systems if isinstance(s, Emitter))
    emitter.enabled_env.fill_(0.0)
    sim.prepare()
    sim.step()
    renderer.publish()
    assert renderer.draw()
    dry = renderer.read_frame()
    assert np.isfinite(dry).all() and renderer.final is renderer.fluid_passes.color
    emitter.enabled_env.fill_(1.0)
    emitter.rate.fill_(3000.0)
    for _ in range(45):
        sim.step()
    renderer.publish()
    assert renderer.draw()
    wet = renderer.read_frame()
    assert np.isfinite(wet).all()
    changed = (np.abs(wet[..., :3] - dry[..., :3]) > 1e-3).any(axis=-1)
    assert 0.002 < changed.mean() < 0.5, changed.mean()
    renderer.close()
    assert renderer.device is None


@slang_tests
def test_slang_offscreen_smoke():
    from helpers import find, push_actions, run
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_surgical.systems.fluids.smoke import Smoke
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    sim = load_case("two_worlds")
    cfg = read_case("two_worlds")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    assert len(renderer.smokes) == 1
    sim.prepare()
    sim.step()
    renderer.publish()
    assert renderer.draw()
    clear = renderer.read_frame()
    actions = push_actions(find(sim, InstrumentControl), sim.scene.envs)
    actions[0, :, 8] = 1.0
    run(sim, 30, actions)
    assert find(sim, Smoke).alive_count.numpy()[0] > 0
    renderer.publish()
    assert renderer.draw()
    smoky = renderer.read_frame()
    changed = (np.abs(smoky[..., :3] - clear[..., :3]) > 1e-3).any(axis=-1)
    assert np.isfinite(smoky).all() and changed.mean() > 0.001
    renderer.close()


@slang_tests
def test_slang_offscreen_threaded():
    """The renderer on its own stream, fed ring snapshots by the physics worker: frames follow the physics, and a
    slot publishes the same picture as the live scene it was copied from."""
    import warp as wp
    from helpers import find, wait_until
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshots
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    sim = load_case("two_worlds")
    cfg = read_case("two_worlds")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    renderer.stream, renderer.lock = wp.Stream(sim.scene.device), ring.lock
    driver = InstrumentDriver(sim, find(sim, InstrumentControl), None, "scripted")
    worker = PhysicsWorker(sim, [driver], ring)
    first = ring.acquire()
    assert first is not None and first.step == 0
    renderer.publish(first)
    assert renderer.draw()
    frame0 = renderer.read_frame()
    worker.start()
    try:
        wait_until(lambda: worker.steps >= 40)
        frames = 0
        while worker.steps < 80:
            renderer.publish(ring.acquire())
            assert renderer.draw()
            frames += 1
        assert frames >= 1
        renderer.set_env(1)  # env 1 is the other mesh variant: the draws are rebuilt under the ring lock while physics runs
        assert renderer.env == 1 and renderer.snap is not None
        for _ in range(3):
            renderer.publish(ring.acquire())
            assert renderer.draw()
        assert np.isfinite(renderer.read_frame()).all()
        renderer.set_env(0)
        worker.pause(True)
        wait_until(lambda: worker.paused)
        wp.synchronize_device(sim.scene.device)
        snap = ring.acquire()
        assert snap.step == ring.count - 1  # the last publish: the live state
        renderer.publish(snap)
        assert renderer.draw()
        from_slot = renderer.read_frame()
        renderer.publish()  # the live scene, same state
        assert renderer.draw()
        live = renderer.read_frame()
    finally:
        worker.stop()
    worker.check()
    assert np.isfinite(from_slot).all() and np.allclose(from_slot, live, atol=1e-3)
    assert not np.allclose(from_slot, frame0, atol=0.02)  # the block moved
    renderer.publish(snap)
    renderer.set_env(1)  # rebuild under the ring lock, republish from the held slot
    assert renderer.draw() and renderer.snap is snap
    renderer.close()


@needs_chole
def test_rigid_piece_normals_match_recomputed():
    """Skinning rotates each piece's precomputed normals; they must equal the normals the deformable-mesh kernels compute
    from the skinned points (a rigid transform keeps area-weighted normals)."""
    import warp as wp
    from i4h_standalone.render.instruments import piece_skins, skin_pieces
    from i4h_standalone.render.kernels import Scratch, update_normals_tangents
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    sim = load_case("chole_tet_med")
    dev = sim.scene.device
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    actions = control.actions.numpy()
    actions[:, :, 7] = 0.7  # jaws closing: the jaw pieces rotate about their hinge
    control.actions.assign(actions)
    sim.prepare()
    for _ in range(5):
        sim.step()
    skins = piece_skins(control, dev)
    pos = [wp.zeros(s.count, dtype=wp.vec3, device=dev) for s in skins]
    nrm = [wp.zeros(s.count, dtype=wp.vec3, device=dev) for s in skins]
    skin_pieces(control, 0, skins, pos, nrm, dev)
    assert any(abs(float(control.jaw_angle.numpy()[0, k])) > 1e-3 for k in range(control.count))
    for skin, p, n in zip(skins, pos, nrm):
        ref = wp.zeros(skin.count, dtype=wp.vec3, device=dev)
        update_normals_tangents(p, wp.array(skin.indices, dtype=wp.int32, device=dev), wp.zeros(skin.count, dtype=wp.vec2, device=dev),
                                Scratch(skin.count, dev), ref, wp.zeros(skin.count, dtype=wp.vec4, device=dev), dev)
        rotated, recomputed = n.numpy(), ref.numpy()
        assert np.abs(np.linalg.norm(rotated, axis=1) - 1.0).max() < 1e-4
        agree = (np.abs(rotated - recomputed) < 2e-3).all(axis=1)
        assert agree.mean() > 0.95, (skin.name, 1.0 - agree.mean())  # seams: duplicated vertices take the welded average, the recompute the unwelded one (rts.usdc 2.4%)


@slang_tests
@pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (data_dir() / "vessels" / "AortaWithVesselsStatic.usdc").is_file(), reason="aorta assets missing")
def test_slang_offscreen_tree_skin():
    """The skinned aorta draws, sits on the rest surface at rest and follows the tree when it moves."""
    from i4h_standalone.render.slang import SlangRenderer

    sim = load_case("aorta_centreline", envs=1)
    cfg = read_case("aorta_centreline")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    sim.prepare()  # not stepped: the tree is at rest
    renderer.publish()
    assert renderer.draw()
    frame = renderer.read_frame()
    assert frame.shape == (180, 320, 4) and np.isfinite(frame).all()
    differs = (np.abs(frame[..., :3] - np.float32(renderer.clear[:3])) > 1e-3).any(axis=-1)
    assert differs.mean() > 0.02
    assert len(renderer.skinned) == 1 and not renderer.statics
    d = renderer.skinned[0]
    wp.synchronize()
    np.testing.assert_allclose(d.pos.array.numpy(), d.body.skin.points, atol=1e-4)
    scene, body = sim.scene, d.body
    q = scene.field("q", body).numpy()
    q += np.float32([0.0, 0.0, 0.5])
    scene.field("q", body).assign(q)
    renderer.publish()
    wp.synchronize()
    np.testing.assert_allclose(d.pos.array.numpy(), d.body.skin.points + np.float32([0.0, 0.0, 0.5]), atol=1e-4)
    assert renderer.draw()
    assert not np.allclose(renderer.read_frame(), frame, atol=0.05)
    renderer.close()


@slang_tests
@needs_chole
def test_slang_offscreen_clips():
    """The clip tori collapse while no clip is placed and sit around the clipped node once one is."""
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_surgical.systems.instruments.clipping import Clipping
    from i4h_surgical.systems.mechanics.centreline import Centreline

    sim = load_case("chole_tet_med", envs=1)
    cfg = read_case("chole_tet_med")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    clipping = next(s for s in sim.systems if isinstance(s, Clipping))
    centreline = next(s for s in sim.systems if isinstance(s, Centreline))
    sim.prepare()
    for _ in range(30):
        sim.step()
    renderer.publish()
    assert renderer.draw()
    before = renderer.read_frame()
    assert len(renderer.clips) == 1
    cd, d = renderer.clips[0]
    wp.synchronize()
    pos = d.pos.array.numpy().reshape(cd.G, -1, 3)
    assert np.ptp(pos, axis=1).max() == 0.0  # every torus collapsed onto its node (identical vertices)
    n = int(np.flatnonzero(sim.scene.body('tissue').centrelines.line == 1)[5])
    node, active = clipping.node.numpy(), clipping.active.numpy()
    node[0, 0], active[0, 0] = n, 1
    clipping.node.assign(node)
    clipping.active.assign(active)
    renderer.publish()
    assert renderer.draw()
    wp.synchronize()
    pos = d.pos.array.numpy().reshape(cd.G, -1, 3)
    target = centreline.positions.numpy()[0, n]
    np.testing.assert_allclose(pos[0].mean(axis=0), target, atol=1e-4)
    assert abs(np.linalg.norm(pos[0] - target, axis=1).max() - (cd.major + cd.minor)) < 1e-4
    assert np.ptp(pos[1:], axis=1).max() == 0.0
    assert (np.abs(renderer.read_frame() - before) > 1e-3).any()
    renderer.close()


@slang_tests
def test_slang_offscreen_markers():
    """The app's idle-device proxies: two spheres placed from the physics thread's host arrays, drawn in two colours."""
    from i4h_standalone.render.slang import SlangRenderer

    sim = load_case("two_worlds")
    cfg = read_case("two_worlds")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    markers = renderer.add_markers(2)
    sim.prepare()
    sim.step()
    renderer.publish()
    assert renderer.draw()
    before = renderer.read_frame()
    mid = 0.5 * (renderer.camera.position + renderer.camera.target) + np.float32([0.0, 0.0, 0.12])  # above the floor, in front of the block
    _, right, _, _ = renderer.camera.basis()
    markers.set(0, mid, 0.03, False)
    markers.set(1, mid + right * 0.08, 0.03, True)
    renderer.publish()
    assert renderer.draw()
    after = renderer.read_frame()
    changed = (np.abs(after[..., :3] - before[..., :3]) > 1e-3).any(axis=-1)
    assert changed.sum() > 50
    wp.synchronize()
    idle, candidate = (d.pos.array.numpy() for d in markers.draws)
    assert np.ptp(idle.reshape(2, -1, 3)[1], axis=0).max() == 0.0 and np.ptp(candidate.reshape(2, -1, 3)[0], axis=0).max() == 0.0  # each in its colour only
    markers.set(0, mid, 0.0, False)
    markers.set(1, mid, 0.0, True)
    renderer.publish()
    assert renderer.draw()
    assert np.allclose(renderer.read_frame(), before, atol=0.05)
    renderer.close()
    assert renderer.device is None


@slang_tests
def test_slang_offscreen_ui_draws(case):
    """The ImGui context renders a retained widget tree into the offscreen frame; ui=False has no context."""
    import gc

    import slangpy.ui as ui
    from i4h_standalone.render.slang import SlangRenderer

    cfg = read_case(case)["render"]
    cfg.pop("backend")
    sim = load_case(case)
    plain = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, ui=False))
    assert plain.ui is None
    sim.prepare()
    sim.step()
    plain.publish()
    plain.draw()
    before = plain.read_frame()
    plain.close()
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    assert renderer.ui is not None
    seen = []
    win = ui.Window(renderer.ui.screen, "test", position=renderer.spy.float2(5.0, 5.0), size=renderer.spy.float2(200.0, 120.0))
    ui.SliderFloat(win, "value", 0.5, lambda v: seen.append(v), 0.0, 1.0)
    ui.CheckBox(win, "flag", True, lambda v: seen.append(v))
    for _ in range(3):
        renderer.publish()
        renderer.draw()
    after = renderer.read_frame()
    assert np.isfinite(after).all() and after.shape == before.shape
    assert np.abs(after - before).max() > 0.05  # the window is drawn over the scene
    renderer.ui.screen.remove_all_children()
    renderer.close()
    gc.collect()
    again = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))  # construct/close twice: no leak, no crash
    again.publish()
    again.draw()
    again.close()


@slang_tests
def test_slang_offscreen_panels(case):
    """The real panels on the offscreen ImGui context: a knob edit reaches the device, frames draw, close is clean."""
    import time

    from i4h_physics.systems.mechanics.particles import Particles
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_standalone.app.ui import Panels, SimControls
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshots
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    find = lambda sim, cls: next(s for s in sim.systems if isinstance(s, cls))  # noqa: E731
    cfg = read_case(case)["render"]
    cfg.pop("backend")
    sim = load_case(case, envs=2)
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    sim.prepare()
    ring = Snapshots(sim)
    ring.publish()
    renderer.stream, renderer.lock = wp.Stream(sim.scene.device), ring.lock
    driver = InstrumentDriver(sim, find(sim, InstrumentControl), None, "keyboard", renderer=renderer)
    worker = PhysicsWorker(sim, [driver], ring, steps=0)
    panels = Panels(worker, renderer, SimControls(worker, (0.0, 0.0, -9.81)), driver.hub, overrides_path="/dev/null")
    particles = find(sim, Particles)
    row = next(r for r in panels.rows if r.system is particles and r.knob.label == "damping")
    row.widgets[0].callback(0.5)  # what the slider does on a user edit
    panels.update()
    worker.pause(True)
    worker.start()
    done = [False]
    worker.edit(lambda s: done.__setitem__(0, True))
    t0 = time.perf_counter()
    while not done[0] and time.perf_counter() - t0 < 10.0:
        time.sleep(0.005)
    worker.check()
    for _ in range(3):
        panels.update()
        snap = ring.acquire()
        if snap is not None:
            renderer.publish(snap)
        renderer.draw()
    frame = renderer.read_frame()
    assert np.isfinite(frame).all()
    panels.set_metrics("gpu 1.0 / 2.0 ms")
    renderer.draw()
    worker.stop()
    np.testing.assert_allclose(particles.damping.numpy(), 0.5)
    panels.close()
    renderer.close()


@slang_tests
@needs_chole
def test_slang_texture_sets_blend_by_visual_state():
    """The coag and blood layers are the authored sets (visual.set): painting the visual state changes the blended
    diffuse; with the base maps alone in every layer (the pre-M9e binding) the same paint changes nothing."""
    import warp as wp
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_surgical.systems.mechanics.surface import Surface

    sim = load_case("chole_tet_med")
    surface = next(s for s in sim.systems if isinstance(s, Surface))
    cfg = read_case("chole_tet_med")["render"]
    cfg.pop("backend")
    sim.prepare()
    sim.step()

    def frames(renderer):
        renderer.tissue_params["debug_mode"] = 3  # blended diffuse only: no lighting in the comparison
        visual = surface.appearance.visual.numpy()
        visual[:] = 0.0
        surface.appearance.visual.assign(visual)
        renderer.publish()
        renderer.draw()
        plain = renderer.read_hdr()
        visual[:, 1] = 1.0  # full coagulation everywhere
        surface.appearance.visual.assign(visual)
        renderer.publish()
        renderer.draw()
        coag = renderer.read_hdr()
        visual[:, 1], visual[:, 2] = 0.0, 1.0  # blood
        surface.appearance.visual.assign(visual)
        renderer.publish()
        renderer.draw()
        blood = renderer.read_hdr()
        visual[:] = 0.0
        surface.appearance.visual.assign(visual)
        wp.synchronize_device(sim.scene.device)
        return plain, coag, blood

    body = surface.body
    with_sets = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    assert all("set" in p.appearance.visual for p in body.parts.values())
    plain, coag, blood = frames(with_sets)
    with_sets.close()
    target = (np.abs(plain[..., :3] - np.float32(with_sets.clear[:3])) > 1e-3).any(axis=-1)
    assert target.mean() > 0.1
    assert np.abs(coag - plain)[target].mean() > 0.02 and np.abs(blood - plain)[target].mean() > 0.02
    assert not np.allclose(coag, blood, atol=0.02)
    saved = {name: dict(p.appearance.visual) for name, p in body.parts.items()}
    try:
        for name, p in body.parts.items():
            root = saved[name]["set"]
            p.appearance.visual = {"diffuse": f"{root}/diffuse-base.png", "normal": f"{root}/normal-base.png", "specular": f"{root}/spec-base.png"}
        base_only = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
        plain2, coag2, blood2 = frames(base_only)
        base_only.close()
    finally:
        for name, p in body.parts.items():
            p.appearance.visual = saved[name]
    assert np.allclose(plain2, plain, atol=1e-3) and np.allclose(coag2, plain2, atol=1e-3) and np.allclose(blood2, plain2, atol=1e-3)
    with pytest.raises(ValueError, match="unknown keys"):
        with_sets.material({"diffuse": "x.png", "roughness": 0.5})


def _post_frames(case, post, steps=3, frames=1, frame_dt=None, size=(320, 180)):
    """Offscreen two_worlds frames under a Post configuration; returns (frames, hdr, renderer.clear)."""
    from i4h_standalone.render.slang import SlangRenderer

    sim = load_case(case)
    cfg = read_case(case)["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=size, window=False, post=post))
    renderer.frame_dt = frame_dt
    sim.prepare()
    for _ in range(steps):
        sim.step()
    renderer.publish()
    out = []
    for _ in range(frames):
        assert renderer.draw()
        out.append(renderer.read_frame())
    hdr = renderer.read_hdr()
    value = renderer.post_passes.exposure_value()
    renderer.close()
    return out, hdr, value


@slang_tests
def test_post_defaults_are_deterministic_and_display_referred(case):
    (a, b), hdr, _ = _post_frames(case, {}, frames=2)
    assert np.array_equal(a, b)  # no noise: bitwise repeatable draws (every other pixel test relies on it)
    assert a.min() >= 0.0 and a.max() <= 1.0 and np.isfinite(hdr).all()
    background = (np.abs(hdr[..., :3] - hdr[0, 0, :3]) < 1e-6).all(axis=-1)
    assert background.mean() > 0.3 and (np.abs(a[background] - a[0, 0]) < 1.5 / 255).all()  # the display of one colour is one colour


@slang_tests
def test_post_scope_vignette_grade(case):
    (base,), _, _ = _post_frames(case, {})
    (scoped,), _, _ = _post_frames(case, {"scope_enabled": True, "scope_radius": 0.9})
    corners = [scoped[0, 0], scoped[0, -1], scoped[-1, 0], scoped[-1, -1]]
    assert all((c[:3] <= 1.5 / 255).all() for c in corners)
    h, w = base.shape[:2]
    centre = (slice(h // 2 - 10, h // 2 + 10), slice(w // 2 - 10, w // 2 + 10))
    assert np.allclose(scoped[centre], base[centre], atol=2 / 255)
    (vignetted,), _, _ = _post_frames(case, {"vignette_strength": 0.8, "vignette_radius": 0.3})
    border = np.zeros(base.shape[:2], bool)
    border[:10], border[-10:], border[:, :10], border[:, -10:] = True, True, True, True
    assert vignetted[border, :3].mean() < 0.6 * base[border, :3].mean() and np.allclose(vignetted[centre], base[centre], atol=2 / 255)
    (grey,), _, _ = _post_frames(case, {"color_grade_enabled": True, "color_saturation": 0.0})
    assert (grey[..., :3].max(axis=-1) - grey[..., :3].min(axis=-1)).max() < 2.5 / 255
    (warm,), _, _ = _post_frames(case, {"color_grade_enabled": True, "color_warmth": 1.0})
    assert warm[..., 0].mean() > base[..., 0].mean() and warm[..., 2].mean() < base[..., 2].mean()


@slang_tests
def test_post_distortion_and_aberration(case):
    (base,), hdr, _ = _post_frames(case, {})
    (zoomed,), _, _ = _post_frames(case, {"lens_distortion_enabled": True, "lens_distortion_strength": 0.0, "lens_distortion_zoom": 1.2})
    h, w = base.shape[:2]
    assert not np.allclose(zoomed, base, atol=0.02) and np.allclose(zoomed[h // 2, w // 2], base[h // 2, w // 2], atol=3 / 255)
    (ca,), _, _ = _post_frames(case, {"chromatic_aberration_enabled": True, "chromatic_aberration_strength": 4.0})
    diff = np.abs(ca[..., [0, 2]] - base[..., [0, 2]]).sum(axis=-1)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    foreground = ~(np.abs(hdr[..., :3] - hdr[0, 0, :3]) < 1e-6).all(axis=-1)  # the uniform background cannot shift
    split = np.median(r[foreground])
    assert diff[foreground].max() > 2 / 255 and diff[foreground & (r > split)].mean() > diff[foreground & (r <= split)].mean()


@slang_tests
def test_post_bloom_ao_fxaa(case):
    (base,), hdr, _ = _post_frames(case, {})
    def lum(f):
        return f[..., :3].mean(axis=-1)
    (bloomed,), _, _ = _post_frames(case, {"bloom_enabled": True, "bloom_threshold": 0.0, "bloom_intensity": 2.0})
    assert lum(bloomed).mean() > lum(base).mean() and (bloomed[..., :3] - base[..., :3]).min() > -2.5 / 255
    (occluded,), _, _ = _post_frames(case, {"ao_enabled": True, "ao_intensity": 4.0, "ao_radius": 0.05})
    background = (np.abs(hdr[..., :3] - hdr[0, 0, :3]) < 1e-6).all(axis=-1)
    far = background.copy()  # background more than 4 px from geometry: the half-res AO is upsampled across edges (as in omnisurg)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            far &= np.roll(np.roll(background, dy, axis=0), dx, axis=1)
    delta = lum(occluded) - lum(base)
    assert far.mean() > 0.2 and np.abs(delta[far]).max() < 1.5 / 255 and delta.min() < -0.02 and delta.max() < 1.5 / 255
    (smoothed,), _, _ = _post_frames(case, {"fxaa_enabled": True})
    changed = np.abs(lum(smoothed) - lum(base)) > 1.5 / 255
    gy, gx = np.gradient(lum(base))
    edges = np.hypot(gx, gy)
    assert changed.any() and abs(lum(smoothed).mean() - lum(base).mean()) < 0.01 * max(lum(base).mean(), 1e-3)
    assert edges[changed].mean() > 2.0 * edges.mean()  # the changes sit on edges


@slang_tests
def test_post_auto_exposure_and_noise(case):
    post = {"auto_exposure_enabled": True, "auto_exposure_speed": 4.0, "auto_exposure_min": 0.05, "auto_exposure_max": 8.0}
    (base,), _, _ = _post_frames(case, {})
    one, _, (adapted1, target1, luma1) = _post_frames(case, post, frames=1, frame_dt=1.0 / 30.0)
    sixty, _, (adapted60, target60, _) = _post_frames(case, post, frames=60, frame_dt=1.0 / 30.0)
    assert luma1 > 0.0 and abs(adapted1 - target1) > 0.1 * target1  # gradual adaptation
    assert abs(adapted60 - target60) < 0.01 * target60 and not np.allclose(sixty[-1], base, atol=0.02)
    (a, b), _, _ = _post_frames(case, {"sensor_noise_enabled": True, "sensor_noise_strength": 0.02}, frames=2)
    d = (a - b)[..., :3]
    assert not np.array_equal(a, b) and 0.005 < d.std() < 0.2  # the sRGB encode stretches noise in the dark background


def _declared_uniforms(shader: str) -> set:
    """Names a .slang file declares as globals: ``uniform T name;``, ``Texture2D<..> name;``, ``SamplerState name;``."""
    import re

    from i4h_standalone.render import slang

    text = (pathlib.Path(slang.__file__).parent / "shaders" / f"{shader}.slang").read_text()
    names = re.findall(r"^\s*(?:uniform\s+\w+|Texture2D(?:<[^>]*>)?|SamplerState)\s+(\w+)\s*;", text, re.M)
    return set(names)


def test_shader_uniform_names_match_the_key_lists():
    """``ShaderCursor.write`` ignores unknown names silently: every name Python writes must be declared, and every declared
    name written (a stale declaration is dead code)."""
    from i4h_standalone.render.post import SHADER_KEYS
    from i4h_standalone.render.slang import PRESENT_KEYS

    for shader, keys in {**SHADER_KEYS, "present": PRESENT_KEYS}.items():
        assert set(keys) == _declared_uniforms(shader), shader
    assert len(set(SHADER_KEYS["post"])) == len(SHADER_KEYS["post"])


def test_post_config_and_ranges():
    from i4h_standalone.render.post import RANGES, SKIP, Post, bloom_sizes, uniforms

    p = Post()
    u = uniforms(p)
    assert u["ao_enabled"] == 0 and u["scope_radius"] == 1.5 and "white_balance" not in u and "lens_dirt_texture" not in u
    assert set(RANGES) <= {k for k in u if isinstance(u[k], float)} and all(lo < hi for lo, hi in RANGES.values())
    assert all(k in u or k in SKIP for k in RANGES)
    assert bloom_sizes(320, 180) == [(160, 90), (80, 45), (40, 23), (20, 12), (10, 6)] and bloom_sizes(2, 2) == [(1, 1)]
    with pytest.raises(TypeError):
        Post(bloom=1)


def test_pn_lattice():
    from i4h_standalone.render.pn import lattice

    for s, R in ((2, 6), (3, 10), (4, 15)):
        bary, local = lattice(s)
        assert bary.shape == (R, 3) and local.shape == (3 * s * s,) and np.allclose(bary.sum(axis=1), 1.0)
        assert set(local.tolist()) == set(range(R))


@slang_tests
def test_slang_offscreen_pn(case):
    """PN subdivision: the expanded lattice interpolates the base at curvature 0, curves at 1, degenerate slots stay flat."""
    import warp as wp
    from i4h_standalone.render.pn import lattice
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_surgical.systems.mechanics.surface import Surface

    sim = load_case(case)
    surface = next(s for s in sim.systems if isinstance(s, Surface))
    cfg = read_case(case)["render"]
    cfg.pop("backend")
    sim.prepare()
    sim.step()
    plain = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    plain.publish()
    plain.draw()
    base = plain.read_hdr()
    plain.close()
    flat = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, pn={"segments": 2, "curvature": 0.0}))
    d = next(b for b in flat.bodies if b.surface is surface)
    F = surface.tris.shape[1]
    assert d.pn is not None and d.pn.pos.array.shape[0] == F * 6 and d.pn.S == 4
    flat.publish()
    flat.draw()
    wp.synchronize_device(sim.scene.device)
    bary, _ = lattice(2)
    q = d.pos.array.numpy()
    tris = surface.tris.numpy()[0]
    expect = np.einsum("fkc,rk->frc", q[tris], bary).reshape(-1, 3)  # barycentric interpolation of every slot
    np.testing.assert_allclose(d.pn.pos.array.numpy(), expect, atol=1e-5)
    normals = d.pn.normals.array.numpy()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    dead = tris[:, 0] == tris[:, 1]  # (i, i, i) slots
    if dead.any():
        lattice_pos = d.pn.pos.array.numpy().reshape(F, 6, 3)
        assert np.allclose(lattice_pos[dead], lattice_pos[dead][:, :1], atol=1e-6)  # zero area
    linear = flat.read_hdr()
    flat.close()
    curved = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, pn={"segments": 2}))
    curved.publish()
    curved.draw()
    bent = curved.read_hdr()
    curved.close()
    def mask(f):
        return (np.abs(f[..., :3] - np.float32(plain.clear[:3])) > 0.001).any(axis=-1)
    assert np.array_equal(mask(linear), mask(base))  # curvature 0 draws the base geometry (the PN normals still shade it)
    assert abs(linear[mask(base)].mean() - base[mask(base)].mean()) < 0.03 * base[mask(base)].mean()
    assert not np.allclose(bent, base, atol=0.02) and 0.0 < (mask(bent) != mask(base)).mean() < 0.05  # the silhouette bends a little
    assert abs(bent[mask(base)].mean() - base[mask(base)].mean()) < 0.05 * base[mask(base)].mean()
    with pytest.raises(ValueError, match="segments"):
        SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, pn={"segments": 5}))


@slang_tests
def test_slang_offscreen_gbuffer(case):
    """Normals and labels beside the RGB frame: labels name the drawn objects, normals face the camera, the fluid labels
    its pixels once it flows."""
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_surgical.systems.fluids.emitter import Emitter

    sim = load_case(case)
    cfg = read_case(case)["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, gbuffer=True))
    emitter = next(s for s in sim.systems if isinstance(s, Emitter))
    emitter.enabled_env.fill_(0.0)
    sim.prepare()
    sim.step()
    renderer.publish()
    renderer.draw()
    hdr = renderer.read_hdr()
    nd, labels = renderer.read_gbuffer()
    assert nd.shape == (180, 320, 4) and labels.shape == (180, 320) and labels.dtype == np.uint32
    ids = set(np.unique(labels).tolist())
    assert ids <= {0} | set(renderer.labels.values()) and len(ids) >= 2  # the background and at least the block (the probe has no mesh)
    block = next(v for k, v in renderer.labels.items() if k.endswith("/block") or k.startswith("block/"))
    background = (np.abs(hdr[..., :3] - hdr[0, 0, :3]) < 1e-6).all(axis=-1)
    assert block in ids and (~background[labels == block]).mean() > 0.95  # the labelled pixels are drawn pixels
    assert np.array_equal(labels == 0, (nd == 0.0).all(axis=-1))  # background <=> nothing written
    lit = labels > 0
    normals = nd[lit, :3]
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-3)
    h, w = labels.shape  # facing the eye: the normal points against the pixel's view ray (camera space right, up, forward)
    yy, xx = np.mgrid[0:h, 0:w]
    tan_half = np.tan(np.radians(renderer.camera.fov) * 0.5)
    rays = np.stack([(2.0 * (xx + 0.5) / w - 1.0) * tan_half * (w / h), (1.0 - 2.0 * (yy + 0.5) / h) * tan_half, np.ones_like(xx, np.float32)], -1)
    assert ((normals * rays[lit]).sum(axis=1) <= 1e-3).all()
    assert (nd[lit, 3] > renderer.camera.near).all() and (nd[lit, 3] < renderer.camera.far).all()
    emitter.enabled_env.fill_(1.0)
    emitter.rate.fill_(3000.0)
    for _ in range(45):
        sim.step()
    renderer.publish()
    renderer.draw()
    _, labels2 = renderer.read_gbuffer()
    fluid = next(v for k, v in renderer.labels.items() if k.endswith("/fluid"))
    assert (labels2 == fluid).any() and not (labels == fluid).any()
    renderer.close()
    with pytest.raises(RuntimeError, match="gbuffer"):
        plain = SlangRenderer(sim, **dict(cfg, size=(64, 36), window=False))
        try:
            plain.read_gbuffer()
        finally:
            plain.close()


@slang_tests
@needs_chole
def test_gbuffer_labels_each_fluid_pool():
    """Two pools in view: every fluid pixel carries the label of the pool whose particle is nearest, not the first pool's."""

    from i4h_physics.scene import ACTIVE_INT
    from i4h_standalone.render.slang import SlangRenderer

    sim = load_case("chole_tet_med")
    scene = sim.scene
    cfg = read_case("chole_tet_med")["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, gbuffer=True))
    sim.prepare()
    sim.step()
    pos, right, up, forward = renderer.camera.basis()
    blood, irrigation = scene.body("blood"), scene.body("irrigation")
    grid = np.stack(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), indexing="ij"), -1).reshape(-1, 2).astype(np.float32) * 0.012
    for body, distance, shift in ((irrigation, 0.30, 0.0), (blood, 0.31, 0.03)):  # irrigation in front, blood just behind and to the side
        q = scene.field("q", body).numpy()
        flags = scene.field("flags", body).numpy()
        n = len(grid)
        q[0, :n] = pos + forward * distance + np.outer(grid[:, 0] + shift, right) + np.outer(grid[:, 1], up)
        flags[0, :n] |= ACTIVE_INT
        scene.field("q", body).assign(q)
        scene.field("flags", body).assign(flags)
    renderer.publish()
    renderer.draw()
    _, labels = renderer.read_gbuffer()
    li, lb = renderer.labels["irrigation/fluid"], renderer.labels["blood/fluid"]
    assert (labels == li).sum() > 50 and (labels == lb).sum() > 50  # both pools labelled where they alone are seen
    centre = renderer.camera.project(pos + forward * 0.30, 320, 180)[0]
    x, y = int(centre[0]), int(centre[1])
    assert labels[y, x] == li  # the nearer pool wins where both overlap
    renderer.close()


def test_view_projection_matches_project():
    """The shaders' world -> clip matrix (X1) is the same perspective as Camera.project: pixels and depth agree."""
    cam = Camera(position=(0.3, 1.2, -4.0), pitch=-10.0, yaw=20.0, fov=50.0)
    cam.near, cam.far = 0.05, 50.0
    pts = np.random.default_rng(0).uniform(-1.0, 1.0, (256, 3)).astype(np.float32) + cam.target
    pts = pts[cam.project(pts, 1.0, 1.0)[:, 2] > cam.near]  # in front of the camera
    w, h = 640.0, 360.0
    clip = np.c_[pts, np.ones(len(pts), np.float32)] @ cam.view_projection(w, h).T
    px = (clip[:, 0] / clip[:, 3] + 1.0) * 0.5 * w
    py = (1.0 - clip[:, 1] / clip[:, 3]) * 0.5 * h
    ref = cam.project(pts, w, h)
    np.testing.assert_allclose(px, ref[:, 0], rtol=1e-5, atol=1e-2)  # float32; a point near the camera plane projects far off-screen
    np.testing.assert_allclose(py, ref[:, 1], rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(clip[:, 3], ref[:, 2], rtol=1e-5)
    z = clip[:, 2] / clip[:, 3]
    assert ((z > 0.0) & (z < 1.0)).all()  # between near and far: depth in [0, 1]


@slang_tests
def test_off_axis_view_projection_shifts_the_image():
    """An XR eye's off-axis frustum reaches the shaders: shifting clip x by half the clip width moves the scene right by a
    quarter of the image, with nothing else changed."""
    from i4h_standalone.render.slang import SlangRenderer

    sim = load_case("two_worlds")
    cfg = dict(read_case("two_worlds").get("render") or {})
    for key in ("backend", "size"):
        cfg.pop(key, None)
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False))
    try:
        renderer.publish()
        renderer.draw()
        base = renderer.read_hdr()
        view_projection = renderer.camera.view_projection
        shift = np.eye(4, dtype=np.float32)
        shift[0, 3] = 0.5  # clip.x += 0.5 w: NDC +0.5, a quarter of the width
        renderer.camera.view_projection = lambda w, h: shift @ view_projection(w, h)
        renderer.draw()
        shifted = renderer.read_hdr()
    finally:
        renderer.close()
    clear = np.float32(renderer.clear[:3])

    def centroid_x(img):
        mask = (np.abs(img[..., :3] - clear) > 1e-3).any(axis=-1)
        return float(np.nonzero(mask)[1].mean())

    assert centroid_x(shifted) - centroid_x(base) == pytest.approx(320 / 4, abs=6.0)


@slang_tests
@needs_external_data
def test_procedure_marker_completed_snapshot():
    from i4h_isaaclab.composition import build_procedure
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshots
    cfg = read_case('aorta_static_reach')
    sim = load_case(cfg)
    procedure = build_procedure(cfg, sim)
    render = dict(cfg['render'])
    render.pop('backend')
    renderer = SlangRenderer(sim, procedure=procedure, **dict(render, size=(320,180), window=False))
    try:
        sim.step()
        procedure.update(sim.dt)
        ring = Snapshots(sim, procedure=procedure)
        ring.publish(sim.dt)
        wp.synchronize_device(sim.scene.device)
        snap = ring.acquire()
        renderer.publish(snap)
        assert renderer.draw()
        np.testing.assert_array_equal(renderer.task_row,snap.task_host[0])
        assert np.isfinite(renderer.read_frame()).all()
        # The centre is the task target and the ring samples have exactly its acceptance radius.
        from i4h_standalone.render.procedure import marker_points
        points = np.asarray(marker_points(renderer.task_row,procedure.radius))
        np.testing.assert_allclose(np.linalg.norm(points[1:]-points[0],axis=1),procedure.radius,atol=1e-6)
    finally:
        renderer.close()


@slang_tests
def test_slang_texture_upload_between_interop_frames():
    """A late font/texture upload must not stall the CUDA/Vulkan handoff (bound the native hang)."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import time
        import numpy as np
        import warp as wp
        from i4h_standalone.app.catheter import CatheterDriver
        from i4h_standalone.app.worker import PhysicsWorker
        from i4h_isaaclab.composition import load_case, read_case
        from i4h_standalone.render.slang import SlangRenderer
        from i4h_standalone.render.snapshots import Snapshots
        from i4h_endoluminal.systems.rods.catheter_control import CatheterControl

        sim = load_case("aorta_static")
        cfg = read_case("aorta_static")["render"]
        cfg.pop("backend")
        renderer = SlangRenderer(sim, **dict(cfg, size=(160, 90), window=False, ui=False))
        sim.prepare()
        ring = Snapshots(sim)
        ring.publish()
        wp.synchronize_device(sim.scene.device)
        renderer.stream, renderer.lock = wp.Stream(sim.scene.device), ring.lock
        control = next(s for s in sim.systems if isinstance(s, CatheterControl))
        driver = CatheterDriver(sim, control, "noop")
        worker = PhysicsWorker(sim, [driver], ring, realtime=True)
        # Reproduce the dynamic font atlas upload, without depending on the UI's glyph cache.
        worker.start()
        try:
            for i in range(120):
                worker.check()
                renderer.publish(ring.acquire())
                if i % 10 == 0:
                    texture = renderer.device.create_texture(format=renderer.spy.Format.rgba8_unorm,
                        width=32, height=32, usage=renderer.spy.TextureUsage.shader_resource,
                        data=np.full((32, 32, 4), i, np.uint8))
                assert renderer.draw()
                time.sleep(1 / 60)
        finally:
            worker.stop()
        worker.check()
        assert worker.steps > 30
        assert np.isfinite(renderer.read_frame()).all()
        texture = None  # release GPU objects before Device.close()
        renderer.close()
        print("texture upload handoff completed", flush=True)
    """)
    result = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "texture upload handoff completed" in result.stdout


@slang_tests
def test_physical_surface_without_appearance():
    from i4h_physics import load_case as load_physics
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshots
    cfg = {"sim": {"device": "cuda:0", "envs": 2, "gravity": [0, 0, 0]},
           "assets": {"solid": {"type": "tet_grid", "dim": [3, 3, 3], "cell": .02}},
           "systems": [{"type": "tet", "body": "solid", "id": "tet"},
                       {"type": "surface", "tet": "tet"}]}
    sim = load_physics(cfg)
    sim.prepare()
    ring = Snapshots(sim)
    renderer = SlangRenderer(sim, size=(64, 64), window=False, ui=False)
    try:
        ring.publish()
        wp.synchronize_device(sim.scene.device)
        snap = ring.acquire()
        assert not snap.visual
        renderer.publish(snap)
        assert renderer.draw()
        assert np.isfinite(renderer.read_frame()).all()
    finally:
        renderer.close()
