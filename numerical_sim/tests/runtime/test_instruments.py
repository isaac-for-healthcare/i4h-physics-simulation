# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument loader and kinematics on the grasper and hook assets: pieces, sphere layout, RCM, jaw, clamp, reset."""

import os

import numpy as np
import pytest
import warp as wp
from helpers import needs_chole
from i4h_isaaclab.composition import data_dir
from i4h_physics import Scene, Stepper
from i4h_physics.quat import quat_to_matrix
from i4h_surgical.loaders.instrument_usd import add_instrument
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

ASSETS = data_dir() / "instruments"
pytest.importorskip("pxr", reason="usd-core not installed")
pytestmark = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (ASSETS / "pgrasp.usdc").is_file(), reason=f"instrument assets not found at {ASSETS}")

GRASPER = dict(
    kind="grasper", mesh=str(ASSETS / "pgrasp.usdc"), scale=0.0075, trocar=[-1.0, 5.0, 3.0],
    prims={"shaft": "/root/shaft_color/shaft_color_001", "body": "/root/logobody_color/logobody_color_002",
           "jaw_left": "/root/jawLeft_color/jawLeft_color_001", "jaw_right": "/root/jawRight_color/jawRight_color_001"},
    initial={"position": [0.3, 1.2, -3.0]},
    jaw_spheres={"count": 13, "radius": 0.015, "distance": 0.015, "offset": [0.0, -0.015, 0.0]},
    shaft_spheres={"radius": 0.04, "distance": 0.02},
)
HOOK = dict(
    kind="hook", mesh=str(ASSETS / "hook.usdc"), scale=0.0075, trocar=[5.5, 5.0, 5.0],
    prims={"shaft": "/root/shaft_color_001/shaft_color_001", "body": "/root/logobody_color_002/logobody_color_002",
           "hook": "/root/hook/Mesh_001"},
    initial={"position": [0.7, 1.2, -3.0]},
    jaw_spheres={"count": 5, "radius": 0.015, "distance": 0.01,
                 "chains": {"jaw_left": {"offset": [0.0, 0.0, 0.1], "euler": [-20.0, 0.0, 0.0]},
                            "jaw_right": {"offset": [0.0, 0.02, 0.15], "euler": [80.0, 0.0, 0.0]}}},
    wrist_end=[0.0, 0.0, 0.06], shaft_spheres={"radius": 0.04, "distance": 0.02},
)


def make(envs=1):
    scene = Scene("cuda:0", envs, (0.0, -9.81, 0.0))
    grasper = add_instrument(scene, "grasper", **GRASPER)
    hook = add_instrument(scene, "hook", **HOOK)
    scene.allocate()
    control = InstrumentControl(scene)
    sim = Stepper.create(scene, [control], dt=1.0 / 60.0, substeps=1, iterations=1)
    return sim, control, grasper, hook


def test_loader_pieces_and_spheres():
    sim, control, grasper, hook = make()
    counts = {p.name: len(p.points) for p in grasper.pieces}
    assert counts == {"shaft": 39725, "body": 39000, "jaw_left": 8600, "jaw_right": 11327}
    assert {p.name: p.role for p in grasper.pieces} == {"shaft": 0, "body": 1, "jaw_left": 2, "jaw_right": 2}
    assert {p.name: p.jaw_sign for p in grasper.pieces} == {"shaft": 0.0, "body": 0.0, "jaw_left": 1.0, "jaw_right": -1.0}
    assert all(np.abs(p.points).max() < 4.0 for p in grasper.pieces)  # scaled (shaft is ~3.6 long)
    assert len(grasper.spheres) == 2 * 13 + 32 + 32 and len(hook.spheres) == 2 * 5 + 32 + 32
    assert [name for _, _, name in grasper.prims] == ["jaw_left", "jaw_right", "wrist", "shaft"]
    assert [name for _, _, name in hook.prims] == ["jaw_left", "jaw_right", "wrist", "shaft"]
    assert np.allclose(grasper.spheres[0].local, [0.0, -0.015, 0.0]) and np.allclose(grasper.spheres[12].local, [0.0, -0.015, 0.18])
    assert np.allclose(grasper.spheres[13].local, [0.0, 0.015, 0.0])  # jaw_right mirrors the offset
    assert np.allclose(hook.spheres[5].local, [0.0, 0.02, 0.15], atol=1e-6)  # jaw_right chain root
    assert hook.wrist_end is not None and np.isfinite(grasper.jaw_anchor).all() and np.isfinite(grasper.wrist_pivot).all()
    assert control.sphere_count == 90 + 74 and control.prim_count == 8
    assert control.sphere_instrument.numpy()[90] == 1 and control.prim_instrument.numpy().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def rcm_reference(root, q_in, trocar):
    """numpy port of omnisurg/instruments/runtime_state.py:29-57 for the generic (non-degenerate) case."""
    direction = root - trocar
    direction /= np.linalg.norm(direction)
    z = -direction
    x_in = quat_to_matrix(q_in)[:, 0]
    x = x_in - np.dot(x_in, direction) * direction
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def test_rcm_jaw_clamp_and_chains():
    sim, control, grasper, hook = make()
    sim.prepare()
    root = control.root.numpy()[0]
    for k, rec in enumerate((grasper, hook)):
        R = quat_to_matrix(control.shaft_q.numpy()[0, k])
        assert np.allclose(R, rcm_reference(root[k].astype(np.float64), np.float32([0, 0, 0, 1]), rec.trocar), atol=1e-4)
        # shaft chain: from the root toward the trocar, spaced 0.02
        start = control.host_sphere_start[k] + (rec.prims[3][0])
        spheres = control.sphere_q.numpy()[0]
        direction = rec.trocar - root[k]
        direction /= np.linalg.norm(direction)
        for i in (0, 1, 31):
            assert np.allclose(spheres[start + i], root[k] + direction * 0.02 * i, atol=1e-5)
        assert control.sphere_active.numpy()[0, start : start + 32].all()
        # every sphere sits within the tool's reach of the root
        own = control.sphere_q.numpy()[0, control.host_sphere_start[k] : control.host_sphere_start[k] + len(rec.spheres)]
        assert np.isfinite(own).all() and np.linalg.norm(own - root[k], axis=1).max() < 1.0
    # grasper wrist chain ends at the midpoint of the two jaw-chain roots
    gs = control.host_sphere_start[0]
    wrist_start = gs + grasper.prims[2][0]
    spheres = control.sphere_q.numpy()[0]
    midpoint = 0.5 * (spheres[gs + 0] + spheres[gs + 13])
    assert np.allclose(spheres[wrist_start + 31], midpoint, atol=1e-5)
    # jaw closes toward the closed angle when grip is 1, opens back when 0
    actions = control.actions.numpy()
    actions[0, 0, 7] = 1.0
    control.actions.assign(actions)
    for _ in range(60):
        sim.step()
    assert control.jaw_angle.numpy()[0, 0] < 0.01 and control.held.numpy()[0, 0] == 1
    assert abs(control.jaw_angle.numpy()[0, 1] - 0.5) < 1e-6  # the hook's grip stayed 0
    # a far target moves the root by at most max_velocity * dt per frame
    actions[0, 0, 0] += 10.0
    control.actions.assign(actions)
    before = control.root.numpy()[0, 0].copy()
    sim.step()
    moved = np.linalg.norm(control.root.numpy()[0, 0] - before)
    assert abs(moved - 2.5 / 60.0) < 1e-5


def test_energy_and_fire_signals():
    """Columns 8 and 9: energy is a thresholded level per env, fire a level with a one-frame rising edge."""
    sim, control, grasper, hook = make(envs=2)
    sim.prepare()
    actions = control.actions.numpy()
    actions[0, 1, 8] = 0.9  # env 0 hook energised
    actions[1, 0, 9] = 1.0  # env 1 grasper fires
    control.actions.assign(actions)
    sim.step()
    assert control.energized.numpy().tolist() == [[0, 1], [0, 0]]
    assert control.fire_on.numpy().tolist() == [[0, 0], [1, 0]] and control.fired.numpy().tolist() == [[0, 0], [1, 0]]
    sim.step()
    assert control.fire_on.numpy()[1, 0] == 1 and control.fired.numpy()[1, 0] == 0  # held: no second edge
    control.energy_threshold.fill_(0.95)  # live threshold, no recapture
    graph = sim.graph
    sim.step()
    assert sim.graph is graph and control.energized.numpy()[0, 1] == 0
    actions[1, 0, 9] = 0.0
    control.actions.assign(actions)
    sim.step()
    sim.step()
    actions[1, 0, 9] = 1.0
    control.actions.assign(actions)
    sim.step()
    assert control.fired.numpy()[1, 0] == 1  # released and pressed again: a new edge
    sim.reset([1])
    assert control.fire_on.numpy()[1].max() == 0 and (control.actions.numpy()[1, :, 7:] == 0.0).all()
    assert control.energized.numpy()[0, 1] == 0 and control.actions.numpy()[0, 1, 8] == np.float32(0.9)  # env 0 kept


def test_reset_restores_selected_env():
    sim, control, grasper, hook = make(envs=2)
    sim.prepare()
    actions = control.actions.numpy()
    actions[:, :, 0] += 0.02
    actions[:, 0, 7] = 1.0
    control.actions.assign(actions)
    for _ in range(10):
        sim.step()
    root_before = control.root.numpy().copy()
    spheres_before = control.sphere_q.numpy().copy()
    sim.reset([1])
    assert np.array_equal(control.root.numpy()[0], root_before[0]) and np.array_equal(control.sphere_q.numpy()[0], spheres_before[0])
    assert np.allclose(control.root.numpy()[1], control.rest) and control.jaw_angle.numpy()[1, 0] == 0.5
    assert np.array_equal(control.sphere_q.numpy()[1], control.sphere_q_prev.numpy()[1])
    assert (control.actions.numpy()[1, :, 7] == 0.0).all() and control.held.numpy()[1].max() == 0


@needs_chole
def test_scripted_grasp_per_env():
    """The scripted grasper closes on the gallbladder: env 0 holds tissue by t = 2.2 s with body-local slots;
    a masked reset of env 1 clears only env 1's slots; opening the jaw releases."""
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
    from i4h_surgical.systems.instruments.grasp import Grasp

    sim = load_case("chole_tet_med", envs=2)
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    grasp = next(s for s in sim.systems if isinstance(s, Grasp))
    contact = next(s for s in sim.systems if isinstance(s, InstrumentContact))
    driver = InstrumentDriver(sim, control, None, "scripted")
    sim.prepare()
    for step in range(132):
        driver.update(step * sim.dt)
        sim.step()
    count = grasp.count.numpy()
    assert count[0, 0] > 0, count  # env 0 grasper holds gallbladder tissue
    assert count[:, 1].max() == 0  # the hook never grasps
    body = contact.body
    active = grasp.active.numpy()[0, 0].astype(bool)
    verts = grasp.verts.numpy()[0, 0][active]
    assert verts.min() >= 0 and verts.max() < body.count  # body-local indices (the jaw closes on the liver it passes)
    assert (grasp.prim.numpy()[0, 0][active] >= 0).all() and grasp.gap.numpy()[0, 0][active].min() >= 0.0
    assert np.isfinite(sim.scene.q.numpy()).all()

    # records: one slot per (capsule, triangle), never more slots than the capacity, triangles unique per capsule
    rc = contact.record_count.numpy()[0]
    tris = contact.rec_tri.numpy()[0]
    for p in range(contact.instruments.prim_count):
        n = min(int(rc[p]), tris.shape[1])
        assert len(set(tris[p, :n].tolist())) == n and (tris[p, :n] >= 0).all()

    before = grasp.count.numpy()[0].copy()
    sim.reset([1])
    after = grasp.count.numpy()
    assert np.array_equal(after[0], before) and after[1].max() == 0 and grasp.pending.numpy()[1].max() == 0

    # slots the hold or revalidation kernels invalidate are recounted at the substep end
    active = grasp.active.numpy()
    active[0, 0, :] = 0
    grasp.active.assign(active)
    grasp.substep_end(sim.scene, sim.dt / sim.substeps)
    assert grasp.count.numpy()[0, 0] == 0
    sim.reset([0])
    for step in range(132):
        driver.update(step * sim.dt)
        sim.step()
    assert grasp.count.numpy()[0, 0] > 0

    # opening the jaw frees the tissue at step start, before any hold iteration of the step
    actions = control.actions.numpy()
    actions[0, 0, 7] = 0.0
    control.actions.assign(actions)
    scene = sim.scene
    for s in sim.systems:
        s.step_start(scene, sim.dt)
    assert control.released.numpy()[0, 0] == 1
    assert grasp.active.numpy()[0, 0].max() == 0 and grasp.count.numpy()[0, 0] == 0  # cleared before any iteration
    sim.step()
    assert grasp.count.numpy()[0, 0] == 0 and grasp.active.numpy()[0, 0].max() == 0


@needs_chole
def test_hook_acquires_from_record():
    """A distal-limb contact record on fat, moving inward, becomes the entry anchor and holds while stepping;
    records on the liver, on the proximal half or while withdrawing do not."""
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.app.instruments import InstrumentDriver
    from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
    from i4h_surgical.systems.instruments.hook import Hook
    from i4h_surgical.systems.instruments.thermal import Thermal

    sim = load_case("chole_tet_med")
    scene = sim.scene
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    hook = next(s for s in sim.systems if isinstance(s, Hook))
    contact = next(s for s in sim.systems if isinstance(s, InstrumentContact))
    next(s for s in sim.systems if isinstance(s, Thermal)).power.fill_(0.0)  # the scripted hook energises; keep the fat
    driver = InstrumentDriver(sim, control, None, "scripted")
    sim.prepare()
    for step in range(120):  # bring the hook next to the fat
        driver.update(step * sim.dt)
        sim.step()
    body = contact.body
    tris, tri_part = body.tet.tris[0], body.tet.tri_part
    p = hook.prim
    a, b = int(control.prim_a.numpy()[p]), int(control.prim_b.numpy()[p])

    def inject(part_name, t=0.8, penetration=0.004, inward=True):
        q = scene.view(scene.q, body).numpy()[0]
        sq = control.sphere_q.numpy()
        limb = sq[0, a] + (sq[0, b] - sq[0, a]) * t
        candidates = np.flatnonzero(tri_part == body.parts[part_name].index)
        f = int(candidates[np.argmin(np.linalg.norm(q[tris[candidates]].mean(axis=1) - limb, axis=1))])  # nearest triangle
        tri = tris[f]
        target = q[tri].mean(axis=0)
        normal = target - limb
        normal /= np.linalg.norm(normal)
        prev = control.sphere_q_prev.numpy()
        prev[0] = sq[0] - (normal * 0.01 if inward else -normal * 0.01)  # the limb moved 1 cm toward the tissue this frame
        control.sphere_q_prev.assign(prev)
        rc = np.zeros_like(contact.record_count.numpy())
        rc[0, p] = 1
        contact.record_count.assign(rc)
        for arr, value in ((contact.rec_tri, f), (contact.rec_tet, body.tet.tri_tet[0, f]), (contact.rec_t, t),
                           (contact.rec_pen, penetration), (contact.rec_verts, tri), (contact.rec_bary, [1 / 3, 1 / 3, 1 / 3]),
                           (contact.rec_normal, normal)):
            host = arr.numpy()
            host[0, p, 0] = value
            arr.assign(host)
        return tri

    sdt = sim.dt / sim.substeps
    inject("liver")
    hook.substep_end(scene, sdt)
    assert hook.active.numpy()[0, 0] == 0  # liver is not the engageable part
    inject("fat", t=0.3)
    hook.substep_end(scene, sdt)
    assert hook.active.numpy()[0, 0] == 0  # proximal half of the limb never punctures
    inject("fat", inward=False)
    hook.substep_end(scene, sdt)
    assert hook.active.numpy()[0, 0] == 0  # withdrawing
    tri = inject("fat")
    hook.substep_end(scene, sdt)
    assert hook.active.numpy()[0, 0] == 1 and np.array_equal(hook.verts.numpy()[0, 0], tri)
    fat = body.parts["fat"]
    assert body.tet.part[hook.tet.numpy()[0, 0]] == fat.index  # the anchor knows its tet
    assert contact.suppress_active.numpy()[0] == 1 and np.array_equal(contact.suppress_verts.numpy()[0], tri)

    # The paired exit anchor (hook_kernels.py:260-467): the walk from the entry along the contact normal leaves
    # the fat on its far side within the 2 cm puncture distance (the fat is ~8.5 mm thick there).
    from helpers import kill
    from i4h_surgical.systems.mechanics.surface import Surface

    assert hook.active.numpy()[0].tolist() == [1, 1]
    tets, verts, bary = hook.tet.numpy()[0], hook.verts.numpy()[0], hook.bary.numpy()[0]
    m, v = body.tet, body.variant_of_env[0]
    assert m.part[tets[1]] == fat.index and tets[1] != tets[0]
    assert set(verts[1]) <= set(m.tets[v, tets[1]])  # the exit face belongs to its tet ...
    exit_faces = m.face_neighbor[v, tets[1]]
    assert (exit_faces < 0).any() or any(m.part[n] != fat.index for n in exit_faces if n >= 0)  # ... on the fat's boundary
    q = scene.view(scene.q, body).numpy()[0]
    depth = np.linalg.norm(bary[1] @ q[verts[1]] - bary[0] @ q[verts[0]])
    assert 1e-3 < depth <= 0.02 + 1e-4, depth
    for step in range(120, 130):  # both anchors survive ordinary stepping
        driver.update(step * sim.dt)
        sim.step()
    assert hook.active.numpy()[0].tolist() == [1, 1] and np.isfinite(scene.q.numpy()).all()
    # removing the exit tet drops only the exit; removing the entry tet drops both
    surface = next(s for s in sim.systems if isinstance(s, Surface))
    kill(sim, surface, 0, int(tets[1]))
    hook.iteration(scene, sdt, 0, sim.iterations)
    assert hook.active.numpy()[0].tolist() == [1, 0]
    kill(sim, surface, 0, int(tets[0]))
    hook.iteration(scene, sdt, 0, sim.iterations)
    assert hook.active.numpy()[0].tolist() == [0, 0]
    sim.reset([0])
    assert (hook.tet.numpy() == -1).all() and contact.suppress_active.numpy()[0] == 0
    # a hook piercing thick tissue gets no exit anchor
    hook.puncture_distance.fill_(1.0e-4)
    inject("fat")
    hook.substep_end(scene, sdt)
    assert hook.active.numpy()[0].tolist() == [1, 0]


@needs_chole
def test_keyboard_hub_follows_the_viewed_env():
    """Switching the drawn env continues from that env's tool pose instead of writing the old env's pose into it."""
    from types import SimpleNamespace

    from i4h_isaaclab.camera.camera import Camera
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.app.instruments import InstrumentDriver

    sim = load_case("chole_tet_med", envs=2)
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    renderer = SimpleNamespace(env=0, camera=Camera(), key_handlers=[])
    driver = InstrumentDriver(sim, control, None, "keyboard", renderer=renderer)
    driver.update(0.0)
    actions = control.actions.numpy()
    actions[1, 0, 0] += 0.2  # env 1's grasper sits elsewhere
    control.actions.assign(actions)
    driver.actions = actions.copy()
    renderer.env = 1
    driver.update(sim.dt)
    after = control.actions.numpy()
    assert abs(after[1, 0, 0] - actions[1, 0, 0]) < 1e-6 and np.allclose(after[0], actions[0])
    assert np.allclose(driver.hub.position[driver.binding[0]], actions[1, 0, :3])
    driver.close()


def test_keyboard_hub_clears_momentary_controls_on_tab():
    """Shift (energy) and F (fire) belong to the active tool: switching tools with them held must not leave the
    previous tool burning or cutting (review finding); grip stays latched so a grasper keeps holding."""
    from types import SimpleNamespace

    from i4h_isaaclab.camera.camera import Camera
    from i4h_standalone.input.keyboard import KeyboardHub

    class Key(SimpleNamespace):
        def is_key_press(self):
            return self.press

        def is_key_release(self):
            return not self.press

    hub = KeyboardHub({"a": (np.zeros(3), np.float32([0, 0, 0, 1])), "b": (np.zeros(3), np.float32([0, 0, 0, 1]))}, Camera())
    for name in ("left_shift", "f", "space"):
        hub.handle_key(Key(key=name, press=True))
    hub.advance()
    assert hub.energy["a"] == 1.0 and hub.fire["a"] == 1.0 and hub.grip["a"] == 1.0
    hub.handle_key(Key(key="tab", press=True))
    hub.advance()
    assert hub.active == "b" and hub.energy["a"] == 0.0 and hub.fire["a"] == 0.0 and hub.grip["a"] == 1.0
    assert hub.energy["b"] == 1.0 and hub.fire["b"] == 1.0  # still held on the new tool
    for name in ("left_shift", "f", "space"):
        hub.handle_key(Key(key=name, press=False))
    hub.advance()
    assert hub.energy["b"] == 0.0 and hub.fire["b"] == 0.0 and hub.grip["b"] == 0.0 and hub.grip["a"] == 1.0


@needs_chole
def test_inactive_device_releases_its_controls(tmp_path):
    """A device that stops reporting (MiniMou unplugged, replay ended) keeps its pose but must not leave the jaw
    closed, the pedal pressed or the trigger held; its recording keeps one row per frame so a replay keeps the
    session's time base."""
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.app.instruments import InstrumentDriver

    sim = load_case("chole_tet_med")
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    inputs = {"primary": {"backend": "noop"}, "secondary": {"backend": "noop"}}
    driver = InstrumentDriver(sim, control, inputs, "devices", record=str(tmp_path / "session"))
    driver.manager.poll = lambda: None  # the states set below stand in for device samples
    k, dev = next((k, dev) for k, dev in driver.binding.items() if driver.records[k].kind != "hook")
    state = driver.manager.states[dev]
    initial = control.actions.numpy()[0, k, :7].copy()
    driver.update(0.0)  # the device has never reported: the tool holds its reset pose and the recording must say so
    row = driver.recorded[dev][0]
    assert np.allclose(row[:7], initial) and row[7:] == [0.0, 0.0, 0.0], row
    state.active, state.position, state.rotation, state.grip = True, (0.1, 1.2, -3.1), (0.0, 0.0, 0.0, 1.0), 1.0
    state.buttons = {"secondary": True, "fire": True}
    driver.update(sim.dt)
    a = control.actions.numpy()[0, k].copy()
    assert np.allclose(a[:3], (0.1, 1.2, -3.1)) and a[7:].tolist() == [1.0, 1.0, 1.0]
    assert np.allclose(driver.recorded[dev][1], a)
    state.active = False  # the manager also zeroes grip and buttons on a real device; the driver must not keep old values
    driver.update(2 * sim.dt)
    b = control.actions.numpy()[0, k]
    assert np.array_equal(b[:7], a[:7]) and b[7:].tolist() == [0.0, 0.0, 0.0]
    assert len(driver.recorded[dev]) == 3 and np.allclose(driver.recorded[dev][2], b)
    driver.close()
    assert (tmp_path / f"session_{dev}.npy").is_file()


def test_hook_handle_is_its_cautery_pedal():
    """In device and keyboard mode a hook's grip (handle, Space) lands in the energy column with grip 0;
    a grasper's grip stays grip. The MiniMou has no second button, so this is the device's only way to burn."""
    from types import SimpleNamespace

    from i4h_isaaclab.camera.camera import Camera
    from i4h_isaaclab.composition import load_case
    from i4h_standalone.app.instruments import InstrumentDriver

    sim = load_case("chole_tet_med")
    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    renderer = SimpleNamespace(env=0, camera=Camera(), key_handlers=[])
    driver = InstrumentDriver(sim, control, None, "keyboard", renderer=renderer)
    grasper, hook = driver.binding[0], driver.binding[1]
    driver.hub.grip[grasper] = 1.0
    driver.hub.grip[hook] = 1.0
    driver.hub.advance = lambda: None  # keep the hub values as set
    driver.update(0.0)
    actions = control.actions.numpy()[0]
    assert actions[0, 7] == 1.0 and actions[0, 8] == 0.0  # grasper closes
    assert actions[1, 7] == 0.0 and actions[1, 8] == 1.0  # hook energises
    driver.hub.grip[hook] = 0.0
    driver.hub.energy[hook] = 1.0  # Shift still works
    driver.update(sim.dt)
    actions = control.actions.numpy()[0]
    assert actions[1, 7] == 0.0 and actions[1, 8] == 1.0
    driver.close()


class _Key:
    def __init__(self, name, press):
        self.key, self._press = name, press

    def is_key_press(self):
        return self._press

    def is_key_release(self):
        return not self._press


def _keyboard_driver(sim):
    from types import SimpleNamespace

    from i4h_isaaclab.camera.camera import Camera
    from i4h_standalone.app.instruments import InstrumentDriver

    control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
    renderer = SimpleNamespace(env=0, camera=Camera(), key_handlers=[])
    return control, renderer, InstrumentDriver(sim, control, None, "keyboard", renderer=renderer)


def test_env_switch_releases_the_left_env():
    """Leaving an env with Shift or F held must not keep cautery or the trigger on where nobody watches; grip stays
    latched so a grasp holds."""
    from i4h_isaaclab.composition import load_case

    sim = load_case("two_worlds")
    control, renderer, driver = _keyboard_driver(sim)
    for name in ("left_shift", "f", "space"):
        driver.hub.handle_key(_Key(name, True))
    driver.update(0.0)
    a = control.actions.numpy()
    assert a[0, 0, 7] == 1.0 and a[0, 0, 8] == 1.0 and a[0, 0, 9] == 1.0
    renderer.env = 1
    driver.update(sim.dt)
    a = control.actions.numpy()
    assert a[0, 0, 8] == 0.0 and a[0, 0, 9] == 0.0 and a[0, 0, 7] == 1.0  # env 0: pedal and trigger off, grip kept
    assert a[1, 0, 8] == 1.0 and a[1, 0, 9] == 1.0  # still held on the env now driven
    driver.close()


def test_driver_reset_restores_initial_poses():
    """After a simulation reset the driver's host rows and the keyboard poses return to the initial state, else the next
    update would put the tool straight back; a partial reset leaves the other envs' rows."""
    from i4h_isaaclab.composition import load_case

    sim = load_case("two_worlds")
    control, renderer, driver = _keyboard_driver(sim)
    dev = driver.binding[0]
    initial = driver.hub.initial[dev][0].copy()
    driver.hub.position[dev] = initial + np.float32([0.3, 0.0, 0.0])
    driver.hub.handle_key(_Key("space", True))
    driver.update(0.0)
    a = control.actions.numpy()
    assert np.allclose(a[0, 0, :3], initial + [0.3, 0.0, 0.0]) and a[0, 0, 7] == 1.0
    sim.reset()
    driver.reset()
    assert np.array_equal(driver.actions, driver.initial) and np.allclose(driver.hub.position[dev], initial) and driver.hub.grip[dev] == 0.0
    driver.hub.handle_key(_Key("space", False))
    driver.update(sim.dt)
    assert np.allclose(control.actions.numpy()[0, 0, :3], initial)
    driver.actions[1, 0, 0] = 5.0
    driver.reset([0])
    assert driver.actions[1, 0, 0] == 5.0 and np.array_equal(driver.actions[0], driver.initial[0])
    driver.close()


def _fake_control(device, n=1):
    """One instrument at the origin with identity rotations: the arrays ``skin_pieces`` reads."""
    from types import SimpleNamespace

    q = wp.array(np.tile(np.float32([0.0, 0.0, 0.0, 1.0]), (1, n, 1)), dtype=wp.quat, device=device)
    return SimpleNamespace(root=wp.zeros((1, n), dtype=wp.vec3, device=device), shaft_q=q, wrist_q=q, jaw_q=q,
                           jaw_angle=wp.full((1, n), 0.4, dtype=wp.float32, device=device),
                           wrist_pivot=wp.zeros(n, dtype=wp.vec3, device=device), jaw_anchor=wp.zeros(n, dtype=wp.vec3, device=device))


def _piece(bind, role=2, jaw_sign=1.0):
    from types import SimpleNamespace

    from i4h_standalone.render.instruments import uv_sphere

    points, faces = uv_sphere()
    return SimpleNamespace(name="p", role=role, jaw_sign=jaw_sign, points=points + np.float32([0.2, 0.0, 0.0]), indices=faces,
                           bind=np.asarray(bind, np.float32), color=(1.0, 1.0, 1.0))


def _skinned_normals_agree(device, bind):
    """Rotated precomputed normals against the deformable kernels' normals of the skinned points."""
    from i4h_standalone.render.instruments import PieceSkin, skin_pieces
    from i4h_standalone.render.kernels import Scratch, update_normals_tangents

    skin = PieceSkin(0, _piece(bind), device)
    pos, nrm = (wp.zeros(skin.count, dtype=wp.vec3, device=device) for _ in range(2))
    skin_pieces(_fake_control(device), 0, [skin], [pos], [nrm], device)
    ref = wp.zeros(skin.count, dtype=wp.vec3, device=device)
    update_normals_tangents(pos, wp.array(skin.indices, dtype=wp.int32, device=device), wp.zeros(skin.count, dtype=wp.vec2, device=device),
                            Scratch(skin.count, device), ref, wp.zeros(skin.count, dtype=wp.vec4, device=device), device)
    return np.abs(nrm.numpy() - ref.numpy()).max()


def test_piece_bind_matrices():
    """Rotations and uniform scales rotate the normals; a reflection flips them with the winding;
    singular and non-uniformly scaled binds are refused instead of producing wrong normals."""
    from i4h_physics.quat import axis_angle_quat, quat_to_matrix
    from i4h_standalone.render.instruments import PieceSkin

    dev = wp.get_device("cuda:0")
    rot = np.eye(4, dtype=np.float32)
    rot[:3, :3] = quat_to_matrix(axis_angle_quat((0.3, 1.0, 0.2), 1.1))
    rot[:3, 3] = [0.1, -0.2, 0.05]
    assert _skinned_normals_agree(dev, rot) < 2e-3
    assert _skinned_normals_agree(dev, rot @ np.diag([0.002, 0.002, 0.002, 1.0]).astype(np.float32)) < 2e-3  # uniform scale
    assert _skinned_normals_agree(dev, rot @ np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)) < 2e-3  # reflection
    for bad in (np.diag([0.001, 0.002, 0.003, 1.0]), np.diag([0.0, 1.0, 1.0, 1.0]), rot @ np.diag([1.0, 1.0, 0.5, 1.0])):
        with pytest.raises(ValueError, match="rotation times a uniform scale"):
            PieceSkin(0, _piece(bad.astype(np.float32)), dev)


def test_instrument_render_kernels_build_on_cpu():
    """``place_sphere_mesh`` had a parameter named ``template``, a C++ keyword Warp emits verbatim into the CPU build."""
    from i4h_standalone.render.instruments import PieceSkin, place_sphere_mesh, skin_pieces, uv_sphere

    dev = wp.get_device("cpu")
    points, faces = uv_sphere()
    out, normals = (wp.zeros(len(points), dtype=wp.vec3, device=dev) for _ in range(2))
    wp.launch(place_sphere_mesh, dim=len(points),
              inputs=[wp.array(points, dtype=wp.vec3, device=dev), wp.zeros((1, 1), dtype=wp.vec3, device=dev),
                      wp.full(1, 0.5, dtype=wp.float32, device=dev), wp.ones((1, 1), dtype=wp.int32, device=dev),
                      wp.ones(1, dtype=wp.int32, device=dev), 0],
              outputs=[out, normals], device=dev)
    assert np.allclose(out.numpy(), 0.5 * points) and np.allclose(normals.numpy(), points)
    wp.launch(place_sphere_mesh, dim=len(points),
              inputs=[wp.array(points, dtype=wp.vec3, device=dev), wp.zeros((1, 1), dtype=wp.vec3, device=dev),
                      wp.full(1, 0.5, dtype=wp.float32, device=dev), wp.ones((1, 1), dtype=wp.int32, device=dev),
                      wp.zeros(1, dtype=wp.int32, device=dev), 0],
              outputs=[out, normals], device=dev)
    assert not out.numpy().any()  # a sphere the overlay does not show collapses
    skin = PieceSkin(0, _piece(np.eye(4, dtype=np.float32)), dev)
    pos, nrm = (wp.zeros(skin.count, dtype=wp.vec3, device=dev) for _ in range(2))
    skin_pieces(_fake_control(dev), 0, [skin], [pos], [nrm], dev)
    assert np.isfinite(pos.numpy()).all() and np.allclose(np.linalg.norm(nrm.numpy(), axis=1), 1.0, atol=1e-5)


# --- M9b tunnelling acceptance ---------------------------------------------------------------------------------------


def capsule_penetration(points, a, b, radius):
    """Depth of each point inside the capsule ``a``-``b`` of ``radius`` (negative outside)."""
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return radius - np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


@pytest.mark.parametrize("speed", [2.5, 5.0,
                                   10.0])
def test_sweep_does_not_tunnel(speed):
    """The grasper stabbed along its shaft through the gallbladder at ``max_velocity`` never leaves a tissue particle deeper
    than half a capsule radius inside any of its capsules (a tunnelled particle would sit inside the trailing shaft capsules);
    the tissue yields instead. CCD, the proxy pose and manifolds stay excluded; the passing speed is the documented bound."""
    from i4h_isaaclab.composition import load_case
    from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
    from i4h_surgical.systems.mechanics.surface import Surface

    sim = load_case("chole_tet_med", envs=1)
    scene, body = sim.scene, sim.scene.body('tissue')
    control, contact, surface = (next(s for s in sim.systems if isinstance(s, cls)) for cls in (InstrumentControl, InstrumentContact, Surface))
    control.max_velocity.fill_(speed)
    k = control.names.index("grasper")
    gb = body.parts["gallbladder"]
    lo, hi = gb.particle_start, gb.particle_start + gb.particle_count
    centre = body.rest.q0[0, lo:hi].mean(axis=0)
    trocar = control.trocar.numpy()[k]
    d = centre - trocar
    d /= np.linalg.norm(d)
    actions = np.zeros((1, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    sim.prepare()
    for _ in range(60):  # the tissue settles
        control.actions.assign(actions)
        sim.step()
    q0 = scene.view(scene.q, body).numpy()[0, lo:hi].copy()
    prims = np.flatnonzero(control.host_prim_instrument == k)
    pa, pb, rad = control.prim_a.numpy()[prims], control.prim_b.numpy()[prims], control.sphere_radius.numpy()
    actions[0, k, :3] = centre - 0.45 * d
    for _ in range(40):  # to the near side
        control.actions.assign(actions)
        sim.step()
    actions[0, k, :3] = centre + 0.45 * d
    worst, records = 0.0, 0
    for step in range(45):  # the stab: 0.9 at max_velocity * dt per step, then settling
        control.actions.assign(actions)
        sim.step()
        q = scene.view(scene.q, body).numpy()[0, lo:hi]
        flags = scene.view(scene.flags, body).numpy()[0, lo:hi]
        live = q[flags != 0]
        sq, active = control.sphere_q.numpy()[0], control.sphere_active.numpy()[0]
        records += int(contact.record_count.numpy()[0, prims].sum())
        for a, b in zip(pa, pb):
            if active[a] == 0 and active[b] == 0:
                continue
            r = max(rad[a], rad[b])
            worst = max(worst, float(capsule_penetration(live, sq[a], sq[b], r).max()) / r)
    q = scene.view(scene.q, body).numpy()[0, lo:hi]
    moved = np.linalg.norm(q - q0, axis=1)
    assert np.isfinite(scene.q.numpy()).all() and surface.deleted.numpy()[0] == 0 and records > 0
    assert moved.max() > 0.03, "the tissue never yielded to the sweep"
    assert worst < 0.5, f"a particle sat {worst:.2f} capsule radii inside the tool at {speed} (tunnelled)"


# --- M9b proximity pickup and toggle ---------------------------------------------------------------------------------


class _Press:
    def __init__(self, name, repeat=False):
        from types import SimpleNamespace

        self.key, self._repeat = SimpleNamespace(name=name), repeat

    def is_key_press(self):
        return True

    def is_key_release(self):
        return False

    def is_key_repeat(self):
        return self._repeat


PICKUP = {"pickup": {"distance": 0.15, "proxy_radius": 0.05, "keys": {"primary": "1", "secondary": "2"}},
          "primary": {"backend": "noop"}, "secondary": {"backend": "noop"}}


def _pickup_driver(envs=1, renderer=None):
    """A grasper without ``input`` and a hook bound to ``secondary``, two noop devices whose states the tests set by hand."""
    from types import SimpleNamespace

    from i4h_standalone.app.instruments import InstrumentDriver

    scene = Scene("cuda:0", envs, (0.0, -9.81, 0.0))
    add_instrument(scene, "grasper", **GRASPER)
    add_instrument(scene, "hook", input="secondary", **HOOK)
    scene.allocate()
    control = InstrumentControl(scene)
    sim = Stepper.create(scene, [control], dt=1.0 / 60.0, substeps=1, iterations=1)
    renderer = renderer or SimpleNamespace(env=0, key_handlers=[])
    driver = InstrumentDriver(sim, control, PICKUP, "devices", renderer=renderer)
    driver.manager.poll = lambda: None
    for dev in ("primary", "secondary"):
        st = driver.manager.states[dev]
        st.active, st.rotation, st.grip, st.buttons = True, (0.0, 0.0, 0.0, 1.0), 0.0, {}
    driver.manager.states["primary"].position = (5.0, 5.0, 5.0)  # far from every tool
    driver.manager.states["secondary"].position = tuple(HOOK["initial"]["position"])
    return sim, control, driver, renderer


def test_pickup_config_and_events():
    from i4h_standalone.app.pickup import RESERVED_KEYS, Pickup, key_name

    sim, control, driver, renderer = _pickup_driver()
    pickup = driver.pickup
    assert pickup is not None and pickup.keys == {"primary": "key1", "secondary": "key2"} and key_name("a") == "a"
    assert pickup.holding == {"primary": None, "secondary": 1} and driver.pickup.handle_key in renderer.key_handlers
    assert driver.markers is None  # the fake renderer has no add_markers
    pickup.handle_key(_Press("key1"))
    pickup.handle_key(_Press("key1", repeat=True))
    pickup.handle_key(_Press("key9"))
    assert list(pickup.toggles) == ["primary"]
    pickup.toggles.clear()
    records = list(sim.scene.instruments.values())
    with pytest.raises(ValueError, match="share the key"):
        Pickup({"keys": {"a": "1", "b": "1"}}, records, {})
    with pytest.raises(ValueError, match="taken by the window"):
        Pickup({"keys": {"a": "p"}}, records, {})
    with pytest.raises(ValueError, match="unknown keys"):
        Pickup({"keys": {"a": "1"}, "radius": 1.0}, records, {})
    assert "n" in RESERVED_KEYS and "m" in RESERVED_KEYS
    from i4h_standalone.app.instruments import InstrumentDriver

    with pytest.raises(ValueError, match="does not define"):
        InstrumentDriver(sim, control, {"pickup": {"keys": {"ghost": "1"}}, "primary": {"backend": "noop"}}, "devices")


def test_pickup_takes_the_nearest_tool_without_a_jump():
    from i4h_physics.quat import axis_angle_quat, quat_mul

    sim, control, driver, renderer = _pickup_driver()
    pickup, states = driver.pickup, driver.manager.states
    grasper_row = control.actions.numpy()[0, 0].copy()
    states["primary"].position = tuple(np.asarray(GRASPER["initial"]["position"]) + [0.1, 0.0, 0.0])
    states["primary"].rotation = tuple(axis_angle_quat((0.0, 1.0, 0.0), 0.5))
    driver.update(0.0)
    assert pickup.holding["primary"] is None and np.array_equal(control.actions.numpy()[0, 0], grasper_row)  # no press yet
    pickup.handle_key(_Press("key1"))
    driver.update(sim.dt)
    assert pickup.holding["primary"] == 0
    assert np.allclose(control.actions.numpy()[0, 0, :7], grasper_row[:7], atol=1e-6)  # the tool did not jump
    # the tool follows the device's translation and rotation through the offsets
    states["primary"].position = tuple(np.asarray(states["primary"].position) + [0.02, 0.01, 0.0])
    driver.update(2 * sim.dt)
    np.testing.assert_allclose(control.actions.numpy()[0, 0, :3], grasper_row[:3] + [0.02, 0.01, 0.0], atol=1e-6)
    turn = axis_angle_quat((0.0, 0.0, 1.0), 0.3)
    states["primary"].rotation = tuple(quat_mul(turn, states["primary"].rotation))
    driver.update(3 * sim.dt)
    np.testing.assert_allclose(control.actions.numpy()[0, 0, 3:7], quat_mul(turn, grasper_row[3:7]), atol=1e-5)
    # a tool 0.4 away is never taken; the secondary device already holds the hook so a press drops it
    states["primary"].position = (5.0, 5.0, 5.0)
    pickup.release("primary")
    pickup.handle_key(_Press("key1"))
    driver.update(4 * sim.dt)
    assert pickup.holding["primary"] is None


def test_pickup_put_down_freezes_the_tool_and_releases_its_controls():
    sim, control, driver, renderer = _pickup_driver()
    pickup, states = driver.pickup, driver.manager.states
    hook = states["secondary"]
    hook.grip, hook.buttons = 1.0, {"secondary": True}
    driver.update(0.0)
    row = control.actions.numpy()[0, 1]
    assert row[8] == 1.0 and row[7] == 0.0  # the hook's handle is its pedal
    pickup.handle_key(_Press("key2"))
    driver.update(sim.dt)
    assert pickup.holding["secondary"] is None and pickup.guard["secondary"] == 1
    frozen = control.actions.numpy()[0, 1].copy()
    assert frozen[7:].tolist() == [0.0, 0.0, 0.0]
    hook.position = tuple(np.asarray(hook.position) + [0.05, 0.0, 0.0])
    driver.update(2 * sim.dt)
    assert np.array_equal(control.actions.numpy()[0, 1], frozen)  # frozen where it was put down
    # the re-grab guard: a press on the spot takes nothing; moving away and back takes it again
    pickup.handle_key(_Press("key2"))
    driver.update(3 * sim.dt)
    assert pickup.holding["secondary"] is None
    hook.position = tuple(np.asarray(frozen[:3]) + [0.3, 0.0, 0.0])
    driver.update(4 * sim.dt)
    assert "secondary" not in pickup.guard
    hook.position = tuple(np.asarray(frozen[:3]) + [0.05, 0.0, 0.0])
    pickup.handle_key(_Press("key2"))
    driver.update(5 * sim.dt)
    assert pickup.holding["secondary"] == 1
    # a held grasp lets go on put-down: grip drops to 0 and the control system raises released
    grasper = states["primary"]
    grasper.position = tuple(GRASPER["initial"]["position"])
    pickup.handle_key(_Press("key1"))
    driver.update(6 * sim.dt)
    assert pickup.holding["primary"] == 0
    grasper.grip = 1.0
    sim.prepare()
    for _ in range(5):
        driver.update(7 * sim.dt)
        sim.step()
    assert control.held.numpy()[0, 0] == 1
    pickup.handle_key(_Press("key1"))
    driver.update(8 * sim.dt)
    sim.step()
    assert control.released.numpy()[0, 0] == 1 and control.actions.numpy()[0, 0, 7] == 0.0


def test_pickup_jaw_takeover_env_switch_and_reset():
    from types import SimpleNamespace

    renderer = SimpleNamespace(env=0, key_handlers=[])
    sim, control, driver, _ = _pickup_driver(envs=2, renderer=renderer)
    pickup, states = driver.pickup, driver.manager.states
    grasper = states["primary"]
    grasper.position, grasper.grip = tuple(GRASPER["initial"]["position"]), 1.0  # the handle is squeezed while the tool's jaw is open
    pickup.handle_key(_Press("key1"))
    driver.update(0.0)
    assert pickup.holding["primary"] == 0 and control.actions.numpy()[0, 0, 7] == 0.0  # parked open
    for g in (0.9, 0.5, 0.2):
        grasper.grip = g
        driver.update(0.1)
        assert control.actions.numpy()[0, 0, 7] == 0.0
    grasper.grip = 0.04  # within the tolerance: taken over
    driver.update(0.2)
    assert control.actions.numpy()[0, 0, 7] == pytest.approx(0.04)
    grasper.grip = 0.7
    driver.update(0.3)
    assert control.actions.numpy()[0, 0, 7] == pytest.approx(0.7)  # follows from then on
    # crossing takes over too
    pickup.release("primary")
    pickup.jaw.clear()
    control.actions.numpy()  # no-op read
    pickup.take("primary", 0, grasper, np.float32([*grasper.position, 0, 0, 0, 1, 0.5, 0, 0]))
    grasper.grip = 0.9
    assert pickup.grip("primary", 0.9) == 0.5
    assert pickup.grip("primary", 0.3) == pytest.approx(0.3)  # 0.9 -> 0.3 crossed the parked 0.5
    # env switch keeps the holding and recomputes the offsets against the new env's row; reset restores the bindings
    grasper.position = tuple(np.asarray(GRASPER["initial"]["position"]) + [0.05, 0.0, 0.0])
    driver.update(0.4)
    renderer.env = 1
    row1 = control.actions.numpy()[1, 0].copy()
    driver.update(0.5)
    assert pickup.holding["primary"] == 0 and np.allclose(control.actions.numpy()[1, 0, :7], row1[:7], atol=1e-6)
    assert control.actions.numpy()[0, 0, 8:10].tolist() == [0.0, 0.0]
    driver.reset([0])  # another env: the drawn env's holdings stay (review finding: it dropped the held tool)
    assert pickup.holding["primary"] == 0
    driver.reset([1])  # the drawn env: back to the input: bindings
    assert pickup.holding == {"primary": None, "secondary": 1} and not pickup.guard and not pickup.jaw
    np.testing.assert_allclose(pickup.offset_p["secondary"], 0.0)
    pickup.take("primary", 0, grasper, control.actions.numpy()[1, 0])
    driver.reset()
    assert pickup.holding["primary"] is None  # every env: everything resets


def test_pickup_proxies_and_recording(tmp_path):
    from i4h_standalone.app.instruments import InstrumentDriver

    sim, control, driver, renderer = _pickup_driver()
    pickup, states = driver.pickup, driver.manager.states
    rows = control.actions.numpy()[0]
    proxies = pickup.proxies(states, rows)
    assert [(dev, r, c) for dev, _, r, c in proxies] == [("primary", 0.05, False), ("secondary", 0.0, False)]  # idle far away; holding
    states["primary"].position = tuple(np.asarray(GRASPER["initial"]["position"]) + [0.1, 0.0, 0.0])
    assert pickup.proxies(states, rows)[0][3] is True  # a tool in reach: candidate
    states["primary"].active = False
    assert pickup.proxies(states, rows)[0][2] == 0.0  # a silent device shows nothing
    # recordings are per instrument replay id; a replay's inputs carry no pickup block
    driver2 = InstrumentDriver(sim, control, PICKUP, "devices", record=str(tmp_path / "s"))
    driver2.manager.poll = lambda: None
    driver2.update(0.0)
    assert set(driver2.recorded) == {"grasper", "secondary"} and len(driver2.recorded["grasper"]) == 1
    driver2.close()
    assert (tmp_path / "s_grasper.npy").is_file() and (tmp_path / "s_secondary.npy").is_file()


def test_gl_key_presses_reach_the_pickup_keys():
    """The GL view's pyglet key symbols arrive in the Slang event shape Pickup.handle_key reads (the teleop loop's keys)."""
    pytest.importorskip("pyglet")
    from i4h_isaaclab.camera.camera import key_event
    from i4h_standalone.render.gl import _KeyPress
    from pyglet.window import key

    assert key_event(_KeyPress(key._1))[:2] == ("key1", True)
    assert key_event(_KeyPress(key.A))[0] == "a" and key_event(_KeyPress(key.PAGEUP))[0] == "page_up"
