# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M6 smoke: born where the energised tool touches hot tissue, expires, per env, masked reset, capture."""

import numpy as np
from helpers import find, push_actions, run
from i4h_isaaclab.composition import load_case
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.fluids.smoke import Smoke
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.thermal import Thermal


def energised_push(sim):
    """two_worlds: env 0's probe pressed into the block with energy on; env 1's at rest."""
    control = find(sim, InstrumentControl)
    actions = push_actions(control, sim.scene.envs)
    actions[0, :, 8] = 1.0
    return actions


def test_smoke_rises_from_hot_tissue_in_the_energised_env(case):
    sim = load_case(case)
    smoke, thermal = find(sim, Smoke), find(sim, Thermal)
    actions = energised_push(sim)
    run(sim, 30, actions)
    assert thermal.heat.numpy()[0].max() >= float(thermal.coag_heat.numpy()[0]), "the probe heated a tet to coagulation"
    hot, alive = smoke.hot.numpy(), smoke.alive_count.numpy()
    assert hot[0] == 1 and hot[1] == 0
    assert alive[0] > 20 and alive[1] == 0, alive
    E, n = sim.scene.envs, int(smoke.count)
    pa = smoke.pos_age.numpy().reshape(E, n, 4)
    active = pa[..., 3] < smoke.vel_life.numpy().reshape(E, n, 4)[..., 3]
    centre = smoke.centre.numpy()[0]
    assert np.linalg.norm(pa[0][active[0], :3] - centre, axis=1).max() < 0.1, "smoke stays near its source"
    assert (pa[0][active[0], 2] > centre[2] - 0.03).all(), "smoke drifts up (against gravity), not down"
    assert not active[1].any()
    actions[0, :, 8] = 0.0  # energy off: no new smoke; the rest expires within the longest lifetime
    run(sim, 20, actions)
    assert smoke.hot.numpy()[0] == 0 and smoke.alive_count.numpy()[0] == 0
    assert smoke.cursor.numpy()[0] > 20 and smoke.cursor.numpy()[1] == 0


def test_smoke_masked_reset_and_capture(case):
    sims = [load_case(case), load_case(case)]
    sims[1].cuda_graph = False
    for sim in sims:
        run(sim, 25, energised_push(sim))
    a, b = (find(s, Smoke) for s in sims)
    assert a.alive_count.numpy()[0] > 0
    # The trigger is a heat threshold that float-atomic noise can cross a substep apart between runs, so the two
    # runs agree on how much smoke there is (to one step's spawns), not on which slot holds which particle.
    assert abs(int(a.alive_count.numpy()[0]) - int(b.alive_count.numpy()[0])) <= int(a.max_spawn)
    assert abs(int(a.cursor.numpy()[0]) - int(b.cursor.numpy()[0])) <= int(a.max_spawn)
    assert np.allclose(a.centre.numpy()[0], b.centre.numpy()[0], atol=5e-3), "same source point"
    sim, smoke = sims[0], a
    before = (smoke.pos_age.numpy().copy(), smoke.cursor.numpy().copy())
    sim.reset([1])
    E, n = sim.scene.envs, int(smoke.count)
    assert np.array_equal(smoke.pos_age.numpy().reshape(E, n, 4)[0], before[0].reshape(E, n, 4)[0]) and smoke.cursor.numpy()[0] == before[1][0]
    sim.reset([0])
    assert (smoke.vel_life.numpy() == 0).all() and (smoke.alive_count.numpy() == 0).all() and (smoke.cursor.numpy() == 0).all()
    run(sim, 3)
    assert np.isfinite(smoke.pos_age.numpy()).all()


def test_smoke_source_scan_stays_within_record_capacity(case):
    """Contacts beyond the record capacity are counted but not stored (instrument_contact.py): the source scan must
    clamp to the capacity instead of reading the slots that follow the row in memory."""
    sim = load_case(case)
    sim.cuda_graph = False
    smoke, contacts, thermal = find(sim, Smoke), find(sim, InstrumentContact), find(sim, Thermal)
    run(sim, 30, energised_push(sim))
    assert smoke.hot.numpy()[0] == 1
    E, P, R = contacts.rec_tet.shape
    assert E >= 2
    hot_tet = int(np.argmax(thermal.heat.numpy()[0]))
    assert thermal.heat.numpy()[0, hot_tet] >= float(thermal.coag_heat.numpy()[0])
    # Env 0's own records go cold and its last prim claims R + 3 records; the slots that follow that row in memory
    # (env 1, prim 0) hold a hot record. An unclamped scan finds it, a clamped one finds nothing.
    arrays = (contacts.rec_tet, contacts.rec_verts, contacts.rec_bary, contacts.record_count)
    rec_tet, rec_verts, rec_bary, count = (a.numpy() for a in arrays)
    rec_tet[0] = -1
    count[0] = 0
    count[0, P - 1] = R + 3
    rec_tet[1, 0, 0] = hot_tet
    rec_verts[1, 0, 0] = (0, 1, 2)
    rec_bary[1, 0, 0] = (1.0, 0.0, 0.0)
    for arr, host in zip(arrays, (rec_tet, rec_verts, rec_bary, count)):
        arr.assign(host)
    smoke.step_end(sim.scene, sim.dt)
    assert int(smoke.hot.numpy()[0]) == 0, "the source scan read past the record capacity"


def test_smoke_follows_this_steps_contact(case):
    """Energy stays on while the probe leaves the tissue: the step whose contact count is zero spawns nothing."""
    sim = load_case(case)
    smoke, contacts, control = find(sim, Smoke), find(sim, InstrumentContact), find(sim, InstrumentControl)
    actions = energised_push(sim)
    run(sim, 30, actions)
    assert smoke.hot.numpy()[0] == 1
    actions[0, :, :3] = control.rest[None]  # back to the parked pose below the block, energy still on
    for _ in range(20):
        run(sim, 1, actions)
        touching = int(contacts.record_count.numpy()[0].sum()) > 0
        assert int(smoke.hot.numpy()[0]) == int(touching), "hot follows the records of the same step"
    assert int(contacts.record_count.numpy()[0].sum()) == 0
    cursor = int(smoke.cursor.numpy()[0])
    run(sim, 5, actions)
    assert int(smoke.cursor.numpy()[0]) == cursor, "no contact, no smoke"
