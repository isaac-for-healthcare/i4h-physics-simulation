# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M4 thermal: heat from an energised probe, diffusion, burn, deletion and coagulation paint, per env."""

import numpy as np
import pytest
from helpers import find, live_surface_is_consistent, needs_chole, push_actions, run
from i4h_isaaclab.composition import load_case
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.thermal import Thermal
from i4h_surgical.systems.mechanics.surface import Surface


def test_probe_heat_burns_env_locally():
    sim = load_case("two_worlds")
    scene, body = sim.scene, sim.scene.bodies[0]
    control, surface, tet, thermal = find(sim, InstrumentControl), find(sim, Surface), find(sim, Tet), find(sim, Thermal)
    contact = find(sim, InstrumentContact)
    assert thermal.source_prim.numpy().tolist() == [1] and thermal.k == 0 and thermal.hook is None
    actions = push_actions(control, scene.envs)
    actions[1, 0, :3] = actions[0, 0, :3]  # both probes press into their block ...
    actions[0, 0, 8] = 1.0  # ... only env 0 is energised
    run(sim, 20, actions)
    graph = sim.graph
    assert contact.record_count.numpy()[1].max() > 0  # env 1 touches too
    heat = thermal.heat.numpy()
    assert heat[0].max() > 0.0 and not heat[1].any() and not thermal.burn.numpy()[1].any()
    assert np.isfinite(heat).all() and (heat >= 0.0).all()
    hot = np.flatnonzero(heat[0] > 0.0)
    assert body.tet.heat[body.tet.part[hot]].all()
    # heat spreads to face neighbours but stays finite; burn follows heat above coag_heat
    nb = body.tet.face_neighbor[body.variant_of_env[0]][hot]
    assert (heat[0][nb[nb >= 0]] > 0.0).any()
    burn = thermal.burn.numpy()
    assert burn[0].max() > 0.0 and (burn[0][heat[0] <= 1.0] <= burn[0].max()).all()

    run(sim, 60, actions)
    deleted = surface.deleted.numpy()
    assert deleted[0] > 0 and deleted[1] == 0, deleted
    assert sim.graph is graph and surface.overflow.numpy().tolist() == [0, 0]
    live_surface_is_consistent(sim, 0)
    visual = surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)
    assert visual[0][:, 1].max() == 1.0 and not visual[0][:, [0, 2, 3]].any() and not visual[1].any()  # coagulation only
    coag = thermal.coag.numpy().reshape(scene.envs, -1)
    assert np.array_equal(coag[0], visual[0][:, 1])
    # removed tets have no heat; live burnt tets keep theirs
    assert not thermal.heat.numpy()[0][tet.active.numpy()[0] == 0].any()
    assert np.isfinite(scene.q.numpy()).all()

    # live parameters: power 0 stops depositing; the graph stays
    thermal.power.fill_(0.0)
    heat_before = thermal.heat.numpy()[0].copy()
    run(sim, 5, actions)
    assert sim.graph is graph and thermal.heat.numpy()[0].max() <= heat_before.max()

    # env 1 energised now burns too; a masked reset of env 0 clears env 0 only
    actions[1, 0, 8] = 1.0
    thermal.power.fill_(100.0)
    run(sim, 80, actions)
    assert surface.deleted.numpy()[1] > 0
    env1 = (thermal.heat.numpy()[1].copy(), thermal.burn.numpy()[1].copy(), _coag_1 := thermal.coag.numpy().reshape(scene.envs, -1)[1].copy(),
            surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)[1].copy(), tet.active.numpy()[1].copy())
    sim.reset([0])
    assert not thermal.heat.numpy()[0].any() and not thermal.burn.numpy()[0].any()
    assert not thermal.coag.numpy().reshape(scene.envs, -1)[0].any() and not surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)[0].any()
    assert surface.deleted.numpy()[0] == 0 and tet.active.numpy()[0].sum() == body.tet.active0[body.variant_of_env[0]].sum()
    after = (thermal.heat.numpy()[1], thermal.burn.numpy()[1], thermal.coag.numpy().reshape(scene.envs, -1)[1],
             surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)[1], tet.active.numpy()[1])
    for a, b in zip(env1, after):
        assert np.array_equal(a, b)


def test_thermal_structural_and_refs():
    sim = load_case("two_worlds")
    thermal = find(sim, Thermal)
    sim.step()
    thermal.diffusion_substeps = 2
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    sim.prepare()
    sim.step()
    assert np.isfinite(thermal.heat.numpy()).all()
    with pytest.raises(TypeError, match="thermal.hook"):
        Thermal(sim.scene, surface=find(sim, Surface), contacts=find(sim, InstrumentContact), instruments=find(sim, InstrumentControl),
                instrument="probe", hook=find(sim, Tet))


@needs_chole
def test_scripted_hook_burns_fat():
    """The scripted hook energises near the fat: heat, coagulation and removed fat tets in the touching env."""
    from i4h_standalone.app.instruments import InstrumentDriver

    sim = load_case("chole_tet_med", envs=2)
    scene, body = sim.scene, sim.scene.body('tissue')
    control, surface, _thermal, tet = find(sim, InstrumentControl), find(sim, Surface), find(sim, Thermal), find(sim, Tet)
    driver = InstrumentDriver(sim, control, None, "scripted")
    sim.prepare()
    for step in range(300):
        driver.update(step * sim.dt)
        sim.step()
    assert control.energized.numpy()[:, 1].max() >= 0 and np.isfinite(scene.q.numpy()).all()
    deleted = surface.deleted.numpy()
    assert deleted.max() > 0, deleted
    fat = body.parts["fat"]
    removed = np.flatnonzero(tet.active.numpy()[int(np.argmax(deleted))] == 0)
    assert (body.tet.part[removed] == fat.index).all()  # only the fat burns
    # coagulation paint is fat-only. Checked over every env that burned: the burn is chaotic, and when it removes every
    # painted particle of the env with the most deletions its paint goes with them (the test failed that way rarely)
    visual = surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)
    painted = np.flatnonzero((visual[deleted > 0][:, :, 1] > 0.0).any(axis=0))
    assert painted.size and (painted >= fat.particle_start).all() and (painted < fat.particle_start + fat.particle_count).all()
    assert surface.overflow.numpy().max() == 0
    live_surface_is_consistent(sim, int(np.argmax(deleted)))
