# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9b volumetric grasp: particles inside the closed jaws are captured up to the capacity, held at their captured
distance from their jaw sphere while the tool moves, released when the jaw opens; env isolation, reset, graph."""

import os

import numpy as np
import pytest
from helpers import find, needs_chole
from i4h_isaaclab.composition import data_dir, load_case
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_tet_grid
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.loaders.instrument_usd import add_instrument
from i4h_surgical.systems.instruments.grasp import Grasp
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp

ASSETS = data_dir() / "instruments"
pytest.importorskip("pxr", reason="usd-core not installed")
pytestmark = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (ASSETS / "pgrasp.usdc").is_file(), reason=f"instrument assets not found at {ASSETS}")

GRASPER = dict(
    kind="grasper", mesh=str(ASSETS / "pgrasp.usdc"), scale=0.0075, trocar=[0.0, 0.0, -2.0],
    prims={"shaft": "/root/shaft_color/shaft_color_001", "body": "/root/logobody_color/logobody_color_002",
           "jaw_left": "/root/jawLeft_color/jawLeft_color_001", "jaw_right": "/root/jawRight_color/jawRight_color_001"},
    initial={"position": [0.0, 0.0, -0.3]},
    jaw_spheres={"count": 13, "radius": 0.015, "distance": 0.015, "offset": [0.0, -0.015, 0.0]},
)


def block_and_grasper(envs=2, capacity=64):
    """A pinned-top tet block above an open grasper (Z-up, no gravity, no contact: only the grasp moves the block)."""
    scene = Scene("cuda:0", envs, gravity=(0.0, 0.0, 0.0))
    block = add_tet_grid(scene, "block", [6, 6, 4], 0.02, origin=(-0.06, -0.06, 0.0), fix_top=True, radius=0.005)
    add_instrument(scene, "grasper", **GRASPER)
    scene.allocate()
    ic = InstrumentControl(scene)
    tet = Tet(scene, body="block")
    vg = VolumetricGrasp(scene, instruments=ic, body="block", capacity=capacity)
    sim = Stepper.create(scene, [ic, tet, vg], dt=1.0 / 60.0, substeps=2, iterations=4)
    return sim, scene, block, ic, vg


def servo(sim, ic, actions, sphere, target, steps=40):
    for _ in range(steps):
        ic.actions.assign(actions)
        sim.step()
        p = ic.sphere_q.numpy()[0, sphere]
        if np.linalg.norm(p - target) < 1e-4:
            return
        actions[0, 0, :3] = ic.root.numpy()[0, 0] + (target - p)
    raise AssertionError(f"servo did not converge: {np.linalg.norm(p - target)}")


def test_capture_hold_release():
    sim, scene, block, ic, vg = block_and_grasper()
    assert vg.jaw_count.numpy().tolist() == [26] and vg.grasps.numpy().tolist() == [1]
    sim.prepare()
    actions = ic.actions.numpy()
    q0 = scene.view(scene.q, block).numpy().copy()
    bottom = np.flatnonzero(q0[0][:, 2] < q0[0][:, 2].min() + 1e-4)
    centre = q0[0][bottom[np.argmin(np.linalg.norm(q0[0][bottom, :2], axis=1))]]
    servo(sim, ic, actions, int(vg.jaw_start.numpy()[0]) + 2, centre)  # the third jaw sphere of the left chain on the bottom face
    for _ in range(5):
        sim.step()
    assert (vg.count.numpy() == 0).all() and (vg.pending.numpy() == 0).all()  # open jaw: nothing
    actions[0, 0, 7] = 1.0  # env 0 closes
    ic.actions.assign(actions)
    sim.step()
    count = vg.count.numpy()
    assert 0 < count[0, 0] <= 64 and count[1, 0] == 0 and vg.pending.numpy().tolist() == [[0], [0]]
    slots = vg.particle.numpy()[0, 0]
    held = slots[: count[0, 0]]
    assert (held >= 0).all() and len(set(held.tolist())) == count[0, 0] and (slots[count[0, 0] :] == -1).all()
    q = scene.view(scene.q, block).numpy()[0]
    sq = ic.sphere_q.numpy()[0]
    for g in range(count[0, 0]):
        s = vg.sphere.numpy()[0, 0, g]
        assert vg.jaw_start.numpy()[0] <= s < vg.jaw_start.numpy()[0] + 26
        assert np.linalg.norm(q[held[g]] - sq[s]) <= 0.005 + 0.015 + 2e-3  # captured within reach (the hold moved it a little)
    # hold: pulling the tool down by 0.05 takes the held particles along at their captured distance
    before = scene.view(scene.q, block).numpy()[0][held].copy()
    rest = vg.rest.numpy()[0, 0, : count[0, 0]]
    for step in range(30):
        actions[0, 0, 2] = ic.root.numpy()[0, 0][2] - 0.05 / 30.0
        ic.actions.assign(actions)
        sim.step()
    q = scene.view(scene.q, block).numpy()[0]
    sq = ic.sphere_q.numpy()[0]
    dist = np.linalg.norm(q[held] - sq[vg.sphere.numpy()[0, 0, : count[0, 0]]], axis=1)
    assert (np.abs(dist - np.linalg.norm(rest, axis=1)) < 0.2 * np.linalg.norm(rest, axis=1) + 2e-3).all()
    assert (q[held, 2] < before[:, 2] - 0.02).all()  # the held particles followed the pull
    assert np.allclose(scene.view(scene.q, block).numpy()[1], q0[1], atol=1e-6)  # env 1 untouched
    # release: the jaw opens, the slots clear and the block relaxes back
    actions[0, 0, 7] = 0.0
    ic.actions.assign(actions)
    sim.step()
    assert vg.count.numpy().tolist() == [[0], [0]] and not vg.active.numpy().any() and (vg.particle.numpy() == -1).all()
    for _ in range(60):
        sim.step()
    assert scene.view(scene.q, block).numpy()[0][held, 2].mean() > q[held, 2].mean() + 0.005
    assert np.isfinite(scene.q.numpy()).all()


def test_capacity_retry_reset_and_graph():
    sim, scene, block, ic, vg = block_and_grasper(capacity=4)
    sim.prepare()
    actions = ic.actions.numpy()
    actions[0, 0, 7] = 1.0  # closes far from the block: armed, nothing held
    ic.actions.assign(actions)
    for _ in range(3):
        sim.step()
    assert vg.count.numpy()[0, 0] == 0 and vg.pending.numpy()[0, 0] == 1 and ic.held.numpy()[0, 0] == 1
    q0 = scene.view(scene.q, block).numpy()[0]
    bottom = np.flatnonzero(q0[:, 2] < q0[:, 2].min() + 1e-4)
    centre = q0[bottom[np.argmin(np.linalg.norm(q0[bottom, :2], axis=1))]]
    servo(sim, ic, actions, int(vg.jaw_start.numpy()[0]) + 2, centre)  # the closed jaw reaches the block: the retry captures
    assert vg.count.numpy()[0, 0] == 4 and vg.pending.numpy()[0, 0] == 0  # capped at the capacity
    kept = scene.view(scene.q, block).numpy()[1].copy()
    sim.reset([0])
    assert vg.count.numpy().tolist() == [[0], [0]] and (vg.particle.numpy() == -1).all() and not vg.active.numpy().any()
    assert np.array_equal(scene.view(scene.q, block).numpy()[1], kept)
    vg.capacity = 8
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    with pytest.raises(ValueError, match="fixed"):
        sim.prepare()
    # captured == uncaptured: the same script gives the same count
    counts = []
    for graph in (True, False):
        sim, scene, block, ic, vg = block_and_grasper(envs=1, capacity=16)
        sim.cuda_graph = graph
        sim.prepare()
        actions = ic.actions.numpy()
        q0 = scene.view(scene.q, block).numpy()[0]
        bottom = np.flatnonzero(q0[:, 2] < q0[:, 2].min() + 1e-4)
        servo(sim, ic, actions, int(vg.jaw_start.numpy()[0]) + 2, q0[bottom[np.argmin(np.linalg.norm(q0[bottom, :2], axis=1))]])
        actions[0, 0, 7] = 1.0
        ic.actions.assign(actions)
        for _ in range(10):
            sim.step()
        counts.append((int(vg.count.numpy()[0, 0]), sim.graph is not None))
    assert counts[0][1] and not counts[1][1] and counts[0][0] == counts[1][0] > 0


@needs_chole
def test_chole_both_grasps_hold_the_gallbladder():
    from i4h_standalone.app.instruments import InstrumentDriver

    sim = load_case("chole_tet_med", envs=2)
    control, grasp, vg = find(sim, InstrumentControl), find(sim, Grasp), find(sim, VolumetricGrasp)
    kinds = [type(s) for s in sim.systems]
    assert kinds.index(InstrumentControl) < kinds.index(Grasp) < kinds.index(VolumetricGrasp)
    assert vg.body.name == 'tissue' and vg.capacity == 64
    k = control.names.index("grasper")
    driver = InstrumentDriver(sim, control, None, "scripted")
    best = np.zeros(2, int)
    for step in range(132):
        driver.update(step * sim.dt)
        sim.step()
        best = np.maximum(best, vg.count.numpy()[:, k])
    assert best[0] > 0 and grasp.count.numpy()[:, k].max() >= 0 and np.isfinite(sim.scene.q.numpy()).all()
    assert (vg.count.numpy()[:, [i for i in range(control.count) if i != k]] == 0).all()  # only the grasper holds


def test_dead_particles_free_their_slots_and_the_jaw_retries():
    """Held particles whose tets are gone (cut, burnt, torn) drop out of the grasp; a jaw left holding nothing retries and
    captures the live particles in reach (review finding: dead slots kept the count up and blocked the retry)."""

    sim, scene, block, ic, vg = block_and_grasper(envs=1, capacity=4)
    sim.prepare()
    actions = ic.actions.numpy()
    q0 = scene.view(scene.q, block).numpy()[0]
    bottom = np.flatnonzero(q0[:, 2] < q0[:, 2].min() + 1e-4)
    servo(sim, ic, actions, int(vg.jaw_start.numpy()[0]) + 2, q0[bottom[np.argmin(np.linalg.norm(q0[bottom, :2], axis=1))]])
    actions[0, 0, 7] = 1.0
    ic.actions.assign(actions)
    sim.step()
    assert vg.count.numpy()[0, 0] == 4
    held = vg.particle.numpy()[0, 0, :4].copy()
    live = scene.view(scene.flags, block).numpy().copy()
    flags = live.copy()
    flags[0, held] &= ~1  # the four held particles die (their tets were removed)
    scene.view(scene.flags, block).assign(flags)
    sim.step()  # the dead slots go and the closed jaw retries in the same substep: live neighbours are held instead
    count = vg.count.numpy()[0, 0]
    fresh = vg.particle.numpy()[0, 0, :count]
    assert count > 0 and not set(fresh.tolist()) & set(held.tolist()) and (flags[0, fresh] & 1).all()
    assert vg.pending.numpy()[0, 0] == 0
    flags[0] = 0  # every particle dies: nothing to retry with
    scene.view(scene.flags, block).assign(flags)
    sim.step()
    assert vg.count.numpy()[0, 0] == 0 and (vg.particle.numpy()[0, 0] == -1).all() and not vg.active.numpy().any()
    assert vg.pending.numpy()[0, 0] == 1  # armed until something is held
    scene.view(scene.flags, block).assign(live)  # the tissue is back (a reset would do the same)
    sim.step()
    assert vg.count.numpy()[0, 0] > 0 and vg.pending.numpy()[0, 0] == 0
