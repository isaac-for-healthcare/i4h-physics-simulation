# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Read-only outputs, analytic scaling, final-event records and capture lifetime."""
import json

import numpy as np
import pytest
import warp as wp
from helpers import find, push_actions
from i4h_isaaclab.composition import load_case, read_case
from i4h_physics.observables import KINDS
from i4h_physics.observables import ObservableKind as Kind
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.tool_force import ToolForce


def case(device="cpu", capacity=2, multiple=False):
    cfg = read_case("two_worlds")
    cfg["systems"] = [s for s in cfg["systems"] if s["type"] in
                      ("particles", "tet", "surface", "instrument_control", "instrument_contact", "tool_force")]
    for s in cfg["systems"]:
        if s["type"] == "instrument_contact":
            s["records"] = capacity
            if multiple:
                extra = dict(s, id="other_contact")
    if multiple:
        cfg["systems"].append(extra)
    return load_case(cfg, device=device)


@pytest.mark.parametrize("dt,substeps", [(0.1, 1), (0.1, 4), (0.05, 4)])
def test_raw_wrench_sum_sign_scaling_and_conditioned_separation(dt, substeps):
    sim = case(multiple=True)
    obs = sim.solver.observables((Kind.I4H_WRENCH,))
    sim.dt, sim.substeps = dt, substeps
    sim.prepare()
    sources = [s for s in sim.systems if isinstance(s, InstrumentContact)]
    before = sim.scene.q.numpy().copy()
    for i, source in enumerate(sources):
        source.impulse.fill_(wp.vec3(i + 1, 2, -1))
        source.moment.fill_(wp.vec3(0, 3 * (i + 1), 2))
    haptic = find(sim, ToolForce)
    haptic.contact_gain.fill_(0.1)
    haptic.max_force.fill_(0.01)
    haptic._substeps = substeps
    haptic.step_end(sim.scene, dt)
    sim.solver.publisher.publish(obs, dt, substeps)
    expected = -np.array([3, 4, -2, 0, 9, 4]) * substeps / dt**2
    np.testing.assert_allclose(obs.i4h_wrench.numpy(), np.tile(expected, (2, 1)), rtol=1e-6)
    assert np.linalg.norm(haptic.force.numpy()[0, 0]) == pytest.approx(0.01)
    np.testing.assert_array_equal(sim.scene.q.numpy(), before)
    for i, source in enumerate(sources):
        np.testing.assert_array_equal(source.impulse.numpy()[0, 0], [i + 1, 2, -1])
    assert obs.row_keys[Kind.I4H_WRENCH] == ((0, "instrument_control", 0, "probe"), (1, "instrument_control", 0, "probe"))
    sources[0].enabled = False
    sim.prepare()
    sim.solver.publisher.publish(obs, dt, substeps)
    np.testing.assert_allclose(obs.i4h_wrench.numpy()[0], -np.array([2, 2, -1, 0, 6, 2]) * substeps / dt**2, rtol=1e-6)


@pytest.mark.parametrize("device,captured", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
def test_exports_do_not_change_physics_schedule(device, captured):
    plain, exported = case(device), case(device)
    for sim in (plain, exported):
        sim.cuda_graph = captured
        find(sim, InstrumentControl).actions.assign(push_actions(find(sim, InstrumentControl), 2))
    assert plain.observables is None
    exported.observables = exported.solver.observables(KINDS)
    for _ in range(3):
        plain.step()
        exported.step()
    if device == "cpu":
        assert plain.scene.q.numpy().tobytes() == exported.scene.q.numpy().tobytes()
    else:
        np.testing.assert_allclose(plain.scene.q.numpy(), exported.scene.q.numpy(), rtol=1e-5, atol=1e-7)
    obs = exported.observables
    contact = find(exported, InstrumentContact)
    expected = np.concatenate((contact.impulse.numpy(), contact.moment.numpy()), axis=-1).reshape(-1, 6)
    np.testing.assert_allclose(obs.i4h_wrench.numpy(), -expected * exported.substeps / exported.dt**2, rtol=2e-6)
    selected = obs.select((Kind.I4H_WRENCH,))
    assert selected.i4h_wrench is obs.i4h_wrench and selected.i4h_surgical_contact is None
    saved = obs.i4h_surgical_contact.numpy().copy()
    exported.solver.step(exported.state, exported.state, exported.control, None, exported.dt, selected)
    np.testing.assert_array_equal(obs.i4h_surgical_contact.numpy(), saved)
    for owner, name in ((obs, "i4h_wrench"), (contact, "impulse")):
        original = getattr(owner, name)
        setattr(owner, name, wp.clone(original))
        try:
            with pytest.raises(RuntimeError, match="replaced"):
                exported.step()
        finally:
            setattr(owner, name, original)
    old = exported.observables
    exported.observables = selected
    with pytest.raises(RuntimeError, match="structural setting"):
        exported.step()
    exported.observables = old


@pytest.mark.parametrize("found", [0, 2, 5])
def test_records_exact_capacity_overflow_order_and_completed_geometry(found):
    sim = case(multiple=True)
    obs = sim.solver.observables((Kind.I4H_SURGICAL_CONTACT, Kind.I4H_SURGICAL_COUNT))
    sim.prepare()
    contacts = [s for s in sim.systems if isinstance(s, InstrumentContact)]
    for source in contacts:
        source.record_count.fill_(found)
        source.rec_tri.assign(np.tile([9, 3], (2, 1, 1)).astype(np.int32))
        source.rec_tet.fill_(7)
        source.rec_verts.fill_(wp.vec3i(0, 1, 2))
        source.rec_bary.fill_(wp.vec3(0.25, 0.25, 0.5))
        source.rec_pen.fill_(0.125)
        source.rec_t.fill_(0.75)
        source.rec_normal.fill_(wp.vec3(0, 1, 0))
    q = sim.scene.field("q", contacts[0].body).numpy()
    sim.solver.publisher.publish(obs, sim.dt, sim.substeps)
    counts = obs.i4h_surgical_count.numpy()
    assert (counts["found"] == found).all()
    assert (counts["valid"] == min(found, 2)).all()
    assert (counts["capacity"] == 2).all()
    assert (counts["overflow"] == max(found - 2, 0)).all()
    records = obs.i4h_surgical_contact.numpy().reshape(2, 2, 2)
    assert (records["valid"] == int(found > 0)).all()
    if found:
        np.testing.assert_array_equal(records["triangle"][0, 0], [9, 3])
        np.testing.assert_allclose(records["surface_point_completed"][0, 0, 0], q[0, 0] * .25 + q[0, 1] * .25 + q[0, 2] * .5)
        assert (records["penetration"] == .125).all()
    keys = obs.row_keys[Kind.I4H_SURGICAL_CONTACT]
    assert keys[0][0:3] == (0, "instrument_contact", 0)
    assert keys[2][0:3] == (0, "other_contact", 0)
    for source in contacts:
        source.enabled = False
    sim.prepare()
    sim.solver.publisher.publish(obs, sim.dt, sim.substeps)
    assert not obs.i4h_surgical_contact.numpy()["valid"].any()
    assert not obs.i4h_surgical_count.numpy()["found"].any()
    assert (obs.i4h_surgical_count.numpy()["capacity"] == 2).all()


def test_partial_empty_reset_and_empty_sources():
    sim = case()
    sim.observables = obs = sim.solver.observables(KINDS)
    sim.prepare()
    source = find(sim, InstrumentContact)
    source.impulse.fill_(wp.vec3(1, 2, 3))
    source.record_count.fill_(4)
    sim.solver.publisher.publish(obs, sim.dt, sim.substeps)
    before = obs.i4h_wrench.numpy().copy()
    sim.reset([])
    np.testing.assert_array_equal(obs.i4h_wrench.numpy(), before)
    sim.reset([0])
    assert not obs.i4h_wrench.numpy()[0].any()
    np.testing.assert_array_equal(obs.i4h_wrench.numpy()[1], before[1])
    assert obs.i4h_surgical_count.numpy()[0]["capacity"] == 2
    assert obs.i4h_surgical_count.numpy()[0]["found"] == 0
    assert obs.i4h_surgical_count.numpy()[1]["found"] == 4
    assert obs.i4h_contact_depth.shape == (0,)
    with pytest.raises(ValueError, match="does not support"):
        sim.solver.observables((Kind.CONTACT_F,))
    cfg = {"sim": {"envs": 2}, "assets": {}, "systems": []}
    empty = load_case(cfg, device="cpu")
    empty.observables = empty.solver.observables(KINDS)
    empty.step()
    empty.reset()
    assert all(getattr(empty.observables, kind).size == 0 for kind in KINDS)
    assert json.loads(empty.model.i4h.output_layout[0]) == {k: [] for k in ("tool", "depth", "capsule", "record")}


def test_averaged_applied_correction_and_root_moment():
    from i4h_physics.scene import ACTIVE_INT
    from i4h_surgical.systems.mechanics.tool_delta import apply_average_tools

    # Two equal contact corrections on one mass-2 particle produce dx=(0,1,0),
    # impulse=(0,2,0), and moment=(0,0,4) about root (1,0,0).
    def array(values, dtype):
        return wp.array(np.asarray(values), dtype=dtype, device="cpu")
    q = array([[[3., 0, 0], [4, 0, 0], [5, 0, 0]]], wp.vec3)
    flags = array([[ACTIVE_INT, ACTIVE_INT, 0]], wp.int32)
    inv = array([[.5, 0., 1.]], wp.float32)
    delta = array([[0., 2, 0], [0, 0, 0], [0, 0, 0]], wp.vec3)
    counts = array([2, 0, 0], wp.int32)
    channel = array([[[0., 2, 0]], [[0, 0, 0]], [[0, 0, 0]]], wp.vec3)
    jaw = wp.clone(channel)
    impulse, jaw_impulse, moment = [wp.zeros((1, 1), dtype=wp.vec3, device="cpu") for _ in range(3)]
    wp.launch(apply_average_tools, dim=(1, 3),
              inputs=[flags, inv, 3, 1, array([[[1., 0, 0]]], wp.vec3)],
              outputs=[q, delta, counts, channel, jaw, impulse, jaw_impulse, moment], device="cpu")
    np.testing.assert_array_equal(q.numpy()[0], [[3, 1, 0], [4, 0, 0], [5, 0, 0]])
    np.testing.assert_array_equal(impulse.numpy()[0, 0], [0, 2, 0])
    np.testing.assert_array_equal(moment.numpy()[0, 0], [0, 0, 4])
    for scratch in (delta, counts, channel, jaw):
        assert not scratch.numpy().any()


def test_component_wrench_to_world_com_and_pending_is_separate():
    from i4h_physics.coupling import apply_reaction
    sim = case()
    obs = sim.solver.observables((Kind.I4H_WRENCH,))
    source = find(sim, InstrumentContact)
    source.impulse.fill_(wp.vec3(0, 2, 0))
    source.moment.fill_(wp.vec3(0, 0, 3))
    sim.solver.publisher.publish(obs, 1., 2)
    raw = obs.i4h_wrench.numpy().copy()
    force = wp.array(raw[:, :3].reshape(2, 1, 3), dtype=wp.vec3, device="cpu")
    torque = wp.array(raw[:, 3:].reshape(2, 1, 3), dtype=wp.vec3, device="cpu")
    rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)
    xforms = wp.array([wp.transform((5, 7, 2), rotation)] * 2, dtype=wp.transform, device="cpu")
    body = wp.array([[0], [1]], dtype=int, device="cpu")
    target = wp.array([0], dtype=int, device="cpu")
    com = wp.zeros(2, dtype=wp.vec3, device="cpu")
    root = wp.full((2, 1), wp.vec3(1, 0, 0), dtype=wp.vec3, device="cpu")
    body_f = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
    wp.launch(apply_reaction, dim=(2, 1), inputs=[xforms, com, body, target, xforms, root],
              outputs=[force, torque, body_f], device="cpu")
    np.testing.assert_allclose(body_f.numpy(), [[4, 0, 0, 0, 0, -10]] * 2, atol=2e-6)
    assert not force.numpy().any() and not torque.numpy().any()
    np.testing.assert_array_equal(obs.i4h_wrench.numpy(), raw)
    np.testing.assert_array_equal(source.impulse.numpy()[0, 0], [0, 2, 0])


def test_catheter_root_guide_and_containment_last_substep():
    from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
    from i4h_physics.systems.contact.rod_containment import RodContainment
    sim = load_case("aorta_static", envs=2)
    sim.observables = obs = sim.solver.observables((Kind.I4H_WRENCH, Kind.I4H_CONTACT_DEPTH))
    ctl, contact = find(sim, CatheterControl), find(sim, RodContainment)
    assert ctl.root_reaction
    for _ in range(3):
        sim.step()
    expected = -np.concatenate((ctl.impulse.numpy(), ctl.moment.numpy()), axis=-1).reshape(2, 6) * sim.substeps / sim.dt**2
    np.testing.assert_allclose(obs.i4h_wrench.numpy(), expected, rtol=2e-6)
    np.testing.assert_array_equal(obs.i4h_contact_depth.numpy()["depth"], contact.contact_depth.numpy())
    np.testing.assert_array_equal(obs.i4h_contact_depth.numpy()["count"], contact.contact_count.numpy())
    contact.enabled = False
    sim.prepare()
    sim.step()
    assert not obs.i4h_contact_depth.numpy()["count"].any()
    assert not obs.i4h_contact_depth.numpy()["depth"].any()


def test_cuts_and_degenerate_surface_do_not_relabel_recorded_event():
    sim = case()
    obs = sim.solver.observables((Kind.I4H_SURGICAL_CONTACT, Kind.I4H_SURGICAL_COUNT))
    sim.prepare()
    contact = find(sim, InstrumentContact)
    contact.record_count.fill_(1)
    contact.rec_tri.fill_(3)
    contact.rec_tet.fill_(0)
    contact.rec_verts.fill_(wp.vec3i(0, 1, 2))
    contact.rec_bary.fill_(wp.vec3(1, 0, 0))
    # A cut after recording changes the completed-step surface, not the recorded event.
    contact.surface.active.zero_()
    contact.surface.tris.zero_()
    sim.solver.publisher.publish(obs, sim.dt, sim.substeps)
    records = obs.i4h_surgical_contact.numpy().reshape(2, 2)
    assert (records["valid"][:, 0] == 1).all()
    assert (records["triangle"][:, 0] == 3).all()
    contact.substep_start(sim.scene, sim.dt / sim.substeps, sim.substeps - 1, sim.substeps)
    contact.iteration(sim.scene, sim.dt / sim.substeps, sim.iterations - 1, sim.iterations)
    sim.solver.publisher.publish(obs, sim.dt, sim.substeps)
    assert not obs.i4h_surgical_contact.numpy()["valid"].any()
    assert not obs.i4h_surgical_count.numpy()["found"].any()


def test_allocating_new_outputs_requires_recapture():
    sim = case("cuda:0")
    sim.step()
    sim.solver.observables((Kind.I4H_WRENCH,))
    with pytest.raises(RuntimeError, match="structural setting"):
        sim.reset()
    sim.prepare()
    sim.reset()
