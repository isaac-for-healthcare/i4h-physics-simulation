# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument loader and kinematics on the grasper and hook assets: pieces, sphere layout, RCM, jaw, clamp, reset."""


import os

import numpy as np
import pytest
from i4h_host.composition import data_dir
from i4h_physics import Scene, Stepper
from i4h_physics.quat import quat_to_matrix
from i4h_surgical.instrument_storage import declare_instruments
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
    declare_instruments(scene)
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
    from i4h_host.composition import load_case
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
