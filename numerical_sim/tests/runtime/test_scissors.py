# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9b scissors: the kind (collision jaw chains plus cutting chains), the progressive cut while the jaw closes,
the neighbour blood, the centreline node cut in the same step, and the clip applier kind."""

import os

import numpy as np
import pytest
from helpers import find, live_surface_is_consistent
from i4h_isaaclab.composition import data_dir, load_case
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_tet_grid
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.loaders.instrument_usd import (
    CHAIN_CUT,
    CHAIN_FIXED,
    CHAIN_SHAFT,
    CHAIN_WRIST,
    KIND_COLORS,
    PIECE_COLORS,
    add_instrument,
)
from i4h_surgical.systems.fluids.bleeding import Bleeding
from i4h_surgical.systems.instruments.cutting import Cutting
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.centreline import Centreline
from i4h_surgical.systems.mechanics.surface import Surface

ASSETS = data_dir() / "instruments"
CHOLE = data_dir() / "choly-mid"
pytest.importorskip("pxr", reason="usd-core not installed")
needs_rts = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (ASSETS / "rts.usdc").is_file(), reason=f"scissors mesh rts.usdc not under {ASSETS}")
needs_grasper = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (ASSETS / "pgrasp.usdc").is_file(), reason=f"grasper mesh not under {ASSETS}")
needs_chole = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (CHOLE.is_dir() and (ASSETS / "rts.usdc").is_file()), reason="chole data or rts.usdc missing")

SCISSORS = dict(
    kind="scissors", mesh=str(ASSETS / "rts.usdc"), scale=0.0075,
    prims={"shaft": "/root/shaft_color_001/shaft_color_003", "body": "/root/logobody_color_001/logobody_color_004",
           "jaw_left": "/root/jawLeft_color_001/jawLeft_color_003", "jaw_right": "/root/jawRight_color_001/jawRight_color_003"},
    jaw_spheres={"count": 5, "radius": 0.015, "distance": 0.015, "offset": [0.0, -0.015, 0.0]},
)
GRASPER_PRIMS = {"shaft": "/root/shaft_color/shaft_color_001", "body": "/root/logobody_color/logobody_color_002",
                 "jaw_left": "/root/jawLeft_color/jawLeft_color_001", "jaw_right": "/root/jawRight_color/jawRight_color_001"}


def scissors_block(envs=2, trigger="closing", blood=1.0):
    """A pinned-top tet block above the scissors (Z-up, no gravity, no contact: the block stays put and only the cut moves it)."""
    scene = Scene("cuda:0", envs, gravity=(0.0, 0.0, 0.0))
    block = add_tet_grid(scene, "block", [6, 6, 4], 0.02, origin=(-0.06, -0.06, 0.0), fix_top=True,
                         parts={"block": {"spring": 1.0, "volume": 1.0, "deletable": True}})
    add_instrument(scene, "scissors", trocar=[0.0, 0.0, -2.0], initial={"position": [0.0, 0.0, -0.3]}, **SCISSORS)
    scene.allocate()
    ic = InstrumentControl(scene)
    tet = Tet(scene, body="block")
    surface = Surface(scene, tet=tet)
    cutting = Cutting(scene, surface=surface, instruments=ic, instrument="scissors", trigger=trigger, blood=blood)
    sim = Stepper.create(scene, [ic, tet, surface, cutting], dt=1.0 / 60.0, substeps=1, iterations=1)
    return sim, scene, block, ic, tet, surface, cutting


def servo(sim, ic, actions, sphere, target, steps=40):
    """Move the root until sphere ``sphere`` sits on ``target`` (the root is velocity-clamped, so several steps)."""
    for _ in range(steps):
        ic.actions.assign(actions)
        sim.step()
        p = ic.sphere_q.numpy()[0, sphere]
        if np.linalg.norm(p - target) < 1e-4:
            return
        actions[0, 0, :3] = ic.root.numpy()[0, 0] + (target - p)
    raise AssertionError(f"servo did not converge: {np.linalg.norm(p - target)}")


@needs_rts
def test_scissors_loader():
    scene = Scene("cuda:0", 1)
    rec = add_instrument(scene, "scissors", trocar=[0.0, 1.7, -2.0], initial={"position": [0.3, 1.2, -3.0]}, **SCISSORS)
    roles = {p.name: (p.role, p.jaw_sign) for p in rec.pieces}
    assert roles == {"shaft": (0, 0.0), "body": (1, 0.0), "jaw_left": (2, 1.0), "jaw_right": (2, -1.0)}
    chains = np.array([s.chain for s in rec.spheres])
    assert len(rec.spheres) == 10 + 16 + 64 and [n for _, _, n in rec.prims] == ["jaw_left", "jaw_right", "wrist", "shaft"]
    assert chains.tolist() == [CHAIN_FIXED] * 10 + [CHAIN_CUT] * 16 + [CHAIN_WRIST] * 32 + [CHAIN_SHAFT] * 32
    cut = [s for s in rec.spheres if s.chain == CHAIN_CUT]
    np.testing.assert_allclose(cut[0].local, [0.0, -0.005, 0.0], atol=1e-7)
    np.testing.assert_allclose(cut[8].local, [0.0, 0.005, 0.0], atol=1e-7)  # jaw_right: the offset mirrors
    np.testing.assert_allclose(cut[7].local, [0.0, -0.005, 0.07], atol=1e-6)
    assert all(s.radius == 0.01 and s.role == 2 for s in cut) and cut[0].jaw_sign == 1.0 and cut[8].jaw_sign == -1.0
    assert {p.name: p.color for p in rec.pieces}["jaw_left"] == KIND_COLORS["scissors"]["jaw_left"]
    with pytest.raises(ValueError, match="only scissors"):
        add_instrument(scene, "g", kind="grasper", mesh=SCISSORS["mesh"], prims=SCISSORS["prims"], scale=0.0075, cutting={})
    with pytest.raises(ValueError, match="unknown cutting keys"):
        add_instrument(scene, "s2", cutting={"radius": 0.01, "spacing": 0.01}, **SCISSORS)
    with pytest.raises(ValueError, match="jaw_left and jaw_right"):
        add_instrument(scene, "s3", kind="scissors", mesh=SCISSORS["mesh"], scale=0.0075,
                       prims={"shaft": SCISSORS["prims"]["shaft"], "body": SCISSORS["prims"]["body"]})
    scene.allocate()
    ic = InstrumentControl(scene)
    tet_free = Cutting.__dataclass_fields__  # the system takes trigger and blood
    assert "trigger" in tet_free and "closure_delta" in tet_free and "blood" in tet_free
    assert ic.host_sphere_chain[10:26].tolist() == [CHAIN_CUT] * 16


@needs_grasper
def test_clip_applier_loader_colours():
    scene = Scene("cuda:0", 1)
    applier = add_instrument(scene, "applier", kind="clip_applier", mesh=str(ASSETS / "pgrasp.usdc"), prims=GRASPER_PRIMS, scale=0.0075,
                             trocar=[3.0, 5.0, 3.0], initial={"position": [1.1, 1.2, -3.0]}, jaw_spheres={"distance": 0.015, "offset": [0.0, -0.015, 0.0]})
    grasper = add_instrument(scene, "grasper", kind="grasper", mesh=str(ASSETS / "pgrasp.usdc"), prims=GRASPER_PRIMS, scale=0.0075,
                             trocar=[-1.0, 5.0, 3.0], jaw_spheres={"distance": 0.015, "offset": [0.0, -0.015, 0.0]})
    colours = {p.name: p.color for p in applier.pieces}
    assert colours["jaw_left"] == KIND_COLORS["clip_applier"]["jaw_left"] and colours["shaft"] == PIECE_COLORS["shaft"]
    assert {p.name: p.color for p in grasper.pieces} == {p.name: PIECE_COLORS[p.name] for p in grasper.pieces}
    assert len(applier.spheres) == 2 * 13 + 64 and len(applier.prims) == 4 and applier.kind == "clip_applier"
    with pytest.raises(ValueError, match="kind must be one of"):
        add_instrument(scene, "x", kind="stapler")


@needs_rts
def test_scissors_jaw_delta_and_cut_chains_close():
    sim, scene, block, ic, tet, surface, cutting = scissors_block(envs=1)
    assert cutting.sphere_start == ic.host_sphere_start[0] + 10 and cutting.sphere_count == 16
    sim.prepare()
    actions = ic.actions.numpy()
    sim.step()
    assert ic.jaw_delta.numpy()[0, 0] == 0.0
    def tips():
        return ic.sphere_q.numpy()[0, [cutting.sphere_start + 7, cutting.sphere_start + 15]]
    gap_open = np.linalg.norm(np.subtract(*tips()))
    actions[0, 0, 7] = 1.0
    ic.actions.assign(actions)
    deltas = []
    for _ in range(6):
        sim.step()
        deltas.append(float(ic.jaw_delta.numpy()[0, 0]))
    deltas = np.asarray(deltas)
    assert (deltas > 0).all() and np.allclose(deltas[1:] / deltas[:-1], 2.0 / 3.0, atol=1e-3)  # dt * jaw_response = 1/3
    assert np.linalg.norm(np.subtract(*tips())) < gap_open  # the cut tips converge while closing
    assert ic.sphere_active.numpy()[0, cutting.sphere_start : cutting.sphere_start + 16].all()
    actions[0, 0, 7] = 0.0
    ic.actions.assign(actions)
    sim.step()
    assert ic.jaw_delta.numpy()[0, 0] < 0.0  # reopening
    for _ in range(60):
        sim.step()
    assert abs(ic.jaw_delta.numpy()[0, 0]) < 1e-6


@needs_rts
def test_scissors_cut_block_while_closing():
    sim, scene, block, ic, tet, surface, cutting = scissors_block(envs=2)
    sim.prepare()
    actions = ic.actions.numpy()
    q0 = scene.view(scene.q, block).numpy()[0]
    flags0 = scene.view(scene.flags, block).numpy()[0]
    bottom = np.flatnonzero((flags0 != 0) & (q0[:, 2] < q0[:, 2].min() + 1e-4))
    target = q0[bottom[np.argmin(np.linalg.norm(q0[bottom, :2], axis=1))]]  # the bottom particle nearest the axis
    servo(sim, ic, actions, cutting.sphere_start + 3, target)  # cut sphere 3 of the left chain sits on it, jaws open
    active_before = tet.active.numpy().copy()
    for _ in range(20):
        sim.step()
    assert surface.deleted.numpy().tolist() == [0, 0] and (cutting.cut_on.numpy() == 0).all()  # open and still: no cut
    actions[0, 0, 7] = 1.0  # env 0 closes
    ic.actions.assign(actions)
    sim.step()
    deleted = surface.deleted.numpy()
    assert cutting.cut_on.numpy().tolist() == [[1], [0]] and deleted[0] > 0 and deleted[1] == 0, deleted
    assert np.array_equal(tet.active.numpy()[1], active_before[1])
    # every removed tet lies within a cut sphere's reach (0.01 + margin 0.004)
    v = block.variant_of_env[0]
    removed = np.flatnonzero((active_before[0] != 0) & (tet.active.numpy()[0] == 0))
    q = scene.view(scene.q, block).numpy()[0]
    centres = ic.sphere_q.numpy()[0, cutting.sphere_start : cutting.sphere_start + 16]
    for t in removed:
        corners = q[block.tet.tets[v][t]]
        assert np.linalg.norm(corners[:, None, :] - centres[None], axis=2).min() <= 0.014 + 0.02  # a corner near a centre
    # blood: the corners of every surviving face whose neighbour was cut are red; nothing else is painted
    visual = surface.appearance.visual.numpy().reshape(2, -1, 4)
    fn = block.tet.face_neighbor[v]
    cut_mask = np.zeros(active_before.shape[1], bool)
    cut_mask[removed] = True
    expected = set()
    for t in np.flatnonzero(tet.active.numpy()[0]):
        for face in range(4):
            nb = fn[t, face]
            if nb >= 0 and cut_mask[nb]:
                ids = block.tet.tets[v][t]
                expected.update(ids[list(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))[face])].tolist())
    painted = set(np.flatnonzero(visual[0][:, 2] > 0).tolist())
    assert painted == expected and len(expected) > 0 and (visual[0][list(expected), 2] == 1.0).all()
    assert not visual[0][:, [0, 1]].any() and not visual[1].any()
    # progressive: cutting continues while the jaw closes and stops once it is still; held closed never cuts again
    for _ in range(40):
        sim.step()
    assert ic.jaw_delta.numpy()[0, 0] <= 1e-6
    settled = surface.deleted.numpy()[0]
    for _ in range(20):
        sim.step()
    assert surface.deleted.numpy()[0] == settled and cutting.cut_on.numpy()[0, 0] == 0
    # opening never cuts; a deeper bite and a new closure cut again (reopening rearms)
    actions[0, 0, 7] = 0.0
    ic.actions.assign(actions)
    for _ in range(30):
        sim.step()
        assert cutting.cut_on.numpy()[0, 0] == 0
    assert surface.deleted.numpy()[0] == settled
    actions[0, 0, :3] = ic.root.numpy()[0, 0] + np.float32([0.0, 0.0, 0.02])
    ic.actions.assign(actions)
    for _ in range(3):
        sim.step()
    actions[0, 0, 7] = 1.0
    ic.actions.assign(actions)
    for _ in range(10):
        sim.step()
    assert surface.deleted.numpy()[0] > settled
    live_surface_is_consistent(sim, 0)
    assert np.isfinite(scene.q.numpy()).all()
    # masked reset of env 0 restores its tets, blood and cut signal; env 1 untouched
    q1 = scene.view(scene.q, block).numpy()[1].copy()
    sim.reset([0])
    assert np.array_equal(tet.active.numpy(), active_before) and not surface.appearance.visual.numpy().any()
    assert (cutting.cut_on.numpy() == 0).all() and surface.deleted.numpy().tolist() == [0, 0]
    assert np.array_equal(scene.view(scene.q, block).numpy()[1], q1)


@needs_rts
def test_scissors_captured_matches_uncaptured_and_fire_trigger():
    runs = []
    for graph in (True, False):
        sim, scene, block, ic, tet, surface, cutting = scissors_block(envs=1)
        sim.cuda_graph = graph
        sim.prepare()
        actions = ic.actions.numpy()
        q0 = scene.view(scene.q, block).numpy()[0]
        bottom = np.flatnonzero(q0[:, 2] < q0[:, 2].min() + 1e-4)
        servo(sim, ic, actions, cutting.sphere_start + 3, q0[bottom[np.argmin(np.linalg.norm(q0[bottom, :2], axis=1))]])
        actions[0, 0, 7] = 1.0
        ic.actions.assign(actions)
        for _ in range(15):
            sim.step()
        runs.append((int(surface.deleted.numpy()[0]), scene.view(scene.q, block).numpy()[0].copy(), sim.graph is not None))
    assert runs[0][2] and not runs[1][2] and runs[0][0] == runs[1][0] > 0
    assert np.allclose(runs[0][1], runs[1][1], rtol=1e-4, atol=1e-3)
    # trigger fire: the jaw is irrelevant, fire cuts
    sim, scene, block, ic, tet, surface, cutting = scissors_block(envs=1, trigger="fire")
    sim.prepare()
    actions = ic.actions.numpy()
    q0 = scene.view(scene.q, block).numpy()[0]
    bottom = np.flatnonzero(q0[:, 2] < q0[:, 2].min() + 1e-4)
    servo(sim, ic, actions, cutting.sphere_start + 3, q0[bottom[np.argmin(np.linalg.norm(q0[bottom, :2], axis=1))]])
    actions[0, 0, 7] = 1.0
    ic.actions.assign(actions)
    for _ in range(5):
        sim.step()
    assert surface.deleted.numpy()[0] == 0 and cutting.cut_on.numpy()[0, 0] == 0
    actions[0, 0, 9] = 1.0
    ic.actions.assign(actions)
    sim.step()
    assert surface.deleted.numpy()[0] > 0 and cutting.cut_on.numpy()[0, 0] == 1
    with pytest.raises(ValueError, match="trigger"):
        Cutting(scene, surface=surface, instruments=ic, instrument="scissors", trigger="edge")
    cutting.trigger = "fire"
    cutting.instrument = "scissors"
    cutting.trigger = "closing"
    with pytest.raises(ValueError, match="fixed"):
        cutting.prepare(scene)


@needs_chole
def test_scissors_cut_the_artery():
    """The scissors sever an artery node between their closing blades in env 0; the node is marked in the same step
    as the tets, and the cut ends bleed (replaces the hook stand-in test of M4)."""
    sim = load_case("chole_tet_med", envs=2)
    scene, control = sim.scene, find(sim, InstrumentControl)
    fluid, bleeding, centreline, cutting = find(sim, Fluid), find(sim, Bleeding), find(sim, Centreline), find(sim, Cutting)
    kinds = [type(s) for s in sim.systems]
    assert kinds.index(Cutting) < kinds.index(Centreline) < kinds.index(Bleeding)
    assert bleeding.cutting is cutting and bleeding.instrument == "scissors" and cutting.trigger == "closing"
    assert bleeding.sphere_start == cutting.sphere_start and bleeding.sphere_count == 16
    actions = np.zeros((2, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    for _ in range(5):
        control.actions.assign(actions)
        sim.step()
    assert (bleeding.sources.numpy() == 0).all() and (fluid.alive_count.numpy() == 0).all()
    k = control.names.index("scissors")
    node = (bleeding.node_lo + bleeding.node_hi) // 2
    sphere = cutting.sphere_start  # the left cut chain's root sphere, between the jaws
    cut_step = None
    for _ in range(14):
        actions[0, k, 7] = 0.0  # jaws open while approaching
        for _ in range(15):
            target = centreline.positions.numpy()[0, node]
            actions[0, k, :3] = control.root.numpy()[0, k] + (target - control.sphere_q.numpy()[0, sphere])
            control.actions.assign(actions)
            sim.step()
        actions[0, k, 7] = 1.0  # close: the blades cut what lies between them
        for _ in range(12):
            control.actions.assign(actions)
            deleted_before = int(find(sim, Surface).deleted.numpy()[0])
            sim.step()
            if bleeding.cut.numpy()[0, node]:
                cut_step = (int(cutting.cut_on.numpy()[0, k]), int(find(sim, Surface).deleted.numpy()[0]) - deleted_before)
                break
        if cut_step is not None:
            break
    cut = bleeding.cut.numpy()
    assert cut[0, node] == 1 and cut[1].sum() == 0, "only env 0's node is cut"
    assert cut_step[0] == 1, "the node was marked in a step the scissors were cutting"
    actions[0, k, 7] = 0.0
    actions[0, k, :3] = control.rest[k]  # withdraw
    for _ in range(30):
        control.actions.assign(actions)
        sim.step()
    sources, alive = bleeding.sources.numpy(), fluid.alive_count.numpy()
    cuts = np.flatnonzero(cut[0])
    runs = 1 + int((np.diff(cuts) > 1).sum())
    expected = 2 * runs - int(cuts.min() == bleeding.node_lo) - int(cuts.max() == bleeding.node_hi - 1)
    assert sources[0] == expected > 0 and sources[1] == 0, (sources, cuts)
    assert alive[0, fluid.index["blood"]] > 50 and alive[1].sum() == 0
    assert np.isfinite(scene.q.numpy()).all()
    sim.reset([0])
    assert (bleeding.cut.numpy() == 0).all() and (bleeding.sources.numpy() == 0).all() and (fluid.alive_count.numpy() == 0).all()


# --- clip applier ---------------------------------------------------------------------------------------------------

from i4h_surgical.systems.instruments.clipping import Clipping  # noqa: E402
from i4h_surgical.systems.instruments.grasp import Grasp  # noqa: E402


@needs_chole
def test_clip_applier_never_grasps_and_its_fire_clips():
    sim = load_case("chole_tet_med", envs=2)
    scene, control = sim.scene, find(sim, InstrumentControl)
    grasp, clipping, centreline = find(sim, Grasp), find(sim, Clipping), find(sim, Centreline)
    k = control.names.index("clip_applier")
    assert clipping.k == k and grasp.grasps.numpy()[k] == 0 and grasp.grasps.numpy()[control.names.index("grasper")] == 1
    body = scene.body('tissue')
    part = body.parts["gallbladder"]
    target = body.rest.q0[0, part.particle_start : part.particle_start + part.particle_count].mean(axis=0)
    actions = np.zeros((2, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    for _ in range(60):  # the applier reaches into the gallbladder with open jaws, then closes them
        actions[0, k, :3] = target
        control.actions.assign(actions)
        sim.step()
    actions[0, k, 7] = 1.0
    for _ in range(30):
        control.actions.assign(actions)
        sim.step()
    assert control.held.numpy()[0, k] == 1 and (grasp.count.numpy()[:, k] == 0).all()  # closed on tissue, holding nothing
    assert np.isfinite(scene.q.numpy()).all()
    # fire near a duct node clips it (env 0 only)
    c = body.centrelines
    duct = np.flatnonzero(c.line == c.names.index("cystic_duct"))
    node = int(duct[len(duct) // 2])
    actions[0, k, 7] = 0.0
    for _ in range(90):
        actions[0, k, :3] = centreline.positions.numpy()[0, node]
        control.actions.assign(actions)
        sim.step()
    clipping.reach.fill_(0.25)
    actions[:, k, 9] = 1.0
    control.actions.assign(actions)
    sim.step()
    assert clipping.count.numpy().tolist() == [1, 0]
    clipped = int(clipping.node.numpy()[0, 0])
    assert c.line[clipped] == c.names.index("cystic_duct")
    assert np.linalg.norm(centreline.positions.numpy()[0, clipped] - control.root.numpy()[0, k]) <= 0.25 + 1e-3
    with pytest.raises(ValueError, match="clip_applier"):
        Clipping(scene, centreline=centreline, instruments=control, instrument="grasper", line="cystic_duct")


@needs_chole
def test_clip_applier_scripted_places_a_clip():
    from i4h_standalone.app.instruments import InstrumentDriver

    sim = load_case("chole_tet_med", envs=1)
    control, clipping = find(sim, InstrumentControl), find(sim, Clipping)
    driver = InstrumentDriver(sim, control, None, "scripted")
    k, ks = control.names.index("clip_applier"), control.names.index("scissors")
    assert any(entry[0] == k for entry in driver.script) and any(entry[0] == ks for entry in driver.script)
    fired_scissors = 0
    for step in range(240):  # up to the deepest point of the first cycle (PERIOD 6 s)
        driver.update(step * sim.dt)
        sim.step()
        fired_scissors += int(control.fired.numpy()[0, ks])
    assert clipping.count.numpy()[0] >= 1 and fired_scissors == 0  # the applier fired once at the closest approach; the scissors only closed
    assert np.isfinite(sim.scene.q.numpy()).all()


@needs_grasper
def test_clip_applier_device_squeeze_fires():
    """A MiniMou has no fire button: a clip applier driven by a device fires when the handle is squeezed to SQUEEZE."""
    from i4h_standalone.app.instruments import SQUEEZE, InstrumentDriver

    scene = Scene("cuda:0", 1)
    add_instrument(scene, "applier", kind="clip_applier", mesh=str(ASSETS / "pgrasp.usdc"), prims=GRASPER_PRIMS, scale=0.0075,
                   trocar=[3.0, 5.0, 3.0], initial={"position": [1.1, 1.2, -3.0]}, input="primary")
    add_instrument(scene, "grasper", kind="grasper", mesh=str(ASSETS / "pgrasp.usdc"), prims=GRASPER_PRIMS, scale=0.0075,
                   trocar=[-1.0, 5.0, 3.0], initial={"position": [0.3, 1.2, -3.0]}, input="secondary")
    scene.allocate()
    control = InstrumentControl(scene)
    sim = Stepper.create(scene, [control], dt=1.0 / 60.0, substeps=1, iterations=1)
    driver = InstrumentDriver(sim, control, {"primary": {"backend": "noop"}, "secondary": {"backend": "noop"}}, "devices")
    driver.manager.poll = lambda: None
    driver.squeeze_fire = {"primary", "secondary"}  # what the minimou backend registers (a noop device carries no fire of its own)
    for dev in ("primary", "secondary"):
        st = driver.manager.states[dev]
        st.active, st.position, st.rotation, st.buttons = True, (0.1, 1.2, -3.1), (0.0, 0.0, 0.0, 1.0), {}
    driver.manager.states["primary"].grip = driver.manager.states["secondary"].grip = SQUEEZE
    driver.update(0.0)
    a = control.actions.numpy()[0]
    assert a[0, 7] == SQUEEZE and a[0, 9] == 1.0  # the applier fires when squeezed
    assert a[1, 7] == SQUEEZE and a[1, 9] == 0.0  # the grasper only grips
    driver.manager.states["primary"].grip = 0.5
    driver.update(sim.dt)
    assert control.actions.numpy()[0, 0, 9] == 0.0
    driver.manager.states["primary"].buttons = {"fire": True}
    driver.update(2 * sim.dt)
    assert control.actions.numpy()[0, 0, 9] == 1.0  # a real fire button still works


@needs_grasper
def test_squeeze_fires_only_on_the_hardware_backend():
    """Keyboard Space and replayed rows carry their own fire value: only a MiniMou's squeezed handle fires the applier."""
    from types import SimpleNamespace

    from i4h_isaaclab.camera.camera import Camera
    from i4h_standalone.app.instruments import SQUEEZE, InstrumentDriver

    scene = Scene("cuda:0", 1)
    add_instrument(scene, "applier", kind="clip_applier", mesh=str(ASSETS / "pgrasp.usdc"), prims=GRASPER_PRIMS, scale=0.0075,
                   trocar=[3.0, 5.0, 3.0], initial={"position": [1.1, 1.2, -3.0]}, input="primary")
    scene.allocate()
    control = InstrumentControl(scene)
    sim = Stepper.create(scene, [control], dt=1.0 / 60.0, substeps=1, iterations=1)
    driver = InstrumentDriver(sim, control, {"primary": {"backend": "noop"}}, "devices")  # a replay or a noop device: explicit fire
    driver.manager.poll = lambda: None
    st = driver.manager.states["primary"]
    st.active, st.position, st.rotation, st.buttons, st.grip = True, (1.1, 1.2, -3.0), (0.0, 0.0, 0.0, 1.0), {}, SQUEEZE
    driver.update(0.0)
    assert control.actions.numpy()[0, 0, 7] == SQUEEZE and control.actions.numpy()[0, 0, 9] == 0.0 and not driver.squeeze_fire
    st.buttons = {"fire": True}  # the explicit value is what fires
    driver.update(sim.dt)
    assert control.actions.numpy()[0, 0, 9] == 1.0
    assert InstrumentDriver(sim, control, {"primary": {"backend": "noop"}}, "devices").squeeze_fire == set()
    renderer = SimpleNamespace(env=0, camera=Camera(), key_handlers=[])
    keyboard = InstrumentDriver(sim, control, None, "keyboard", renderer=renderer)
    keyboard.hub.held.down.add("space")
    keyboard.update(0.0)
    row = control.actions.numpy()[0, 0]
    assert row[7] == 1.0 and row[9] == 0.0  # Space closes the jaw; only F fires


@needs_rts
def test_disabled_cutting_clears_its_cut_signal():
    sim, scene, block, ic, tet, surface, cutting = scissors_block(envs=1)
    sim.prepare()
    actions = ic.actions.numpy()
    actions[0, 0, 7] = 1.0
    ic.actions.assign(actions)
    sim.step()
    assert cutting.cut_on.numpy()[0, 0] == 1
    cutting.enabled = False
    sim.prepare()
    assert not cutting.cut_on.numpy().any() and not cutting.cut_tets.numpy().any()  # nothing latched for bleeding to read
    for _ in range(40):  # the jaw finishes closing meanwhile; the disabled system never re-arms
        sim.step()
    assert not cutting.cut_on.numpy().any()
    cutting.enabled = True
    sim.prepare()
    sim.step()
    assert cutting.cut_on.numpy()[0, 0] == 0  # the jaw is closed and still: no cut
    actions[0, 0, 7] = 0.0
    ic.actions.assign(actions)
    for _ in range(40):
        sim.step()
    actions[0, 0, 7] = 1.0
    ic.actions.assign(actions)
    sim.step()
    assert cutting.cut_on.numpy()[0, 0] == 1  # reopening and closing again cuts
