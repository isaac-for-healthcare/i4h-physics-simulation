# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Deleters operate on the chosen Newton State after fresh runtime binding."""
import newton
import numpy as np
import pytest

from helpers import find, push_actions, run
from i4h_isaaclab.composition import maps, read_case, sim_settings
from i4h_physics import Stepper, author_case, tet_fields
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.stretch_breaking import StretchBreaking


@pytest.mark.parametrize("consumer", ["cutting", "thermal", "stretch"])
@pytest.mark.parametrize("capture", [False, True])
def test_deletion_uses_selected_state_without_touching_other_state(consumer, capture):
    cfg, loaders, constructors = maps(read_case("two_worlds"))
    a = author_case(cfg, loaders, constructors)
    builder = newton.ModelBuilder()
    for e in range(a.scene.envs):
        builder.begin_world(gravity=a.scene.gravity_vector)
        a.add_world(builder, e)
        builder.end_world()
    a.complete(builder)
    model = builder.finalize(device="cuda:0")
    model.particle_grid = None
    selected, other = model.state(), model.state()
    # Deliberately restored masks and physical topology survive bind and prepare.
    selected.i4h.tet_active.reshape((2, -1))[1].zero_()
    selected.i4h.tet_edge_active.reshape((2, -1))[1].zero_()
    before = [getattr(other.i4h, "tet_" + key).numpy().tobytes() for key in ("active", "edge_active")]
    templates = [getattr(model.i4h, "tet_" + key).numpy().tobytes() for key in ("active", "edge_active")]
    from i4h_physics.surface_storage import STATE_FIELDS
    surface_before = {key: getattr(other.i4h, "surface_"+key).numpy().tobytes() for key in STATE_FIELDS}
    surface_templates = {key: getattr(model.i4h, "surface_"+key).numpy().tobytes() for key in STATE_FIELDS}
    control = model.control()
    scene, systems = a.bind(model, selected, control)
    sim = Stepper.create(scene, systems, control=control, **sim_settings(cfg, False))
    sim.cuda_graph = capture
    sim.prepare()
    tet = find(sim, Tet)
    assert tet.fields is tet_fields(scene, "tet")
    assert not tet.active.numpy()[1].any() and not tet.edge_active.numpy()[1].any()
    initial_tets = tet.active.numpy()[0].sum()
    initial_edges = tet.edge_active.numpy()[0].sum()
    if consumer == "stretch":
        sim.step()
        breaking = find(sim, StretchBreaking)
        breaking.threshold.fill_(1.5)
        q = scene.field("q", tet.body).numpy()
        z = q[0, :, 2]
        q[0, z < z.min() + 1e-4, 2] -= .06
        scene.field("q", tet.body).assign(q)
        breaking.substep_end(scene, sim.dt / sim.substeps)
    else:
        ic = find(sim, InstrumentControl)
        actions = push_actions(ic, scene.envs)
        if consumer == "cutting":
            run(sim, 20, actions)
            actions[0, 0, 9] = 1
            run(sim, 1, actions)
        else:
            actions[0, 0, 8] = 1
            run(sim, 80, actions)
    assert tet.active.numpy()[0].sum() < initial_tets
    assert tet.edge_active.numpy()[0].sum() < initial_edges
    assert not tet.active.numpy()[1].any() and not tet.edge_active.numpy()[1].any()
    assert before == [getattr(other.i4h, "tet_" + key).numpy().tobytes() for key in ("active", "edge_active")]
    assert templates == [getattr(model.i4h, "tet_" + key).numpy().tobytes() for key in ("active", "edge_active")]
    np.testing.assert_array_equal(tet.active.numpy().reshape(-1), selected.i4h.tet_active.numpy())

    assert surface_before == {key: getattr(other.i4h, "surface_"+key).numpy().tobytes() for key in STATE_FIELDS}
    assert surface_templates == {key: getattr(model.i4h, "surface_"+key).numpy().tobytes() for key in STATE_FIELDS}
    assert selected.i4h.surface_deleted.numpy()[0] > 0
