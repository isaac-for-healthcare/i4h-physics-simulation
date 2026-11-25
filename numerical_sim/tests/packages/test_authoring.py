# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared construction contracts using CPU host allocations."""

import copy

import newton
import numpy as np
import pytest
import warp as wp
from i4h_physics import I4hSolver, author_case
from i4h_physics.newton_model import main_rows

CASE = {"sim": {"device": "cpu", "envs": 2, "cuda_graph": False},
        "assets": {"rod": {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0],
                           "count": 5, "segment_length": .02, "radius": .002}},
        "systems": [{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]}


def authored(cfg=CASE, *, uneven=False):
    authoring = author_case(cfg)
    b = newton.ModelBuilder()
    for world in range(authoring.scene.envs):
        b.begin_world()
        for _ in range(2 if uneven and world else 1):
            b.add_particle((5, 6, 7), (1, 2, 3), 1.)
        authoring.add_world(b, world, ((world, 0, 0), (0, 0, 1, 0)))
        b.add_particle((8, 9, 10), (3, 2, 1), 2.)
        b.end_world()
    authoring.complete(b)
    return authoring, b


def test_authoring_allocates_no_runtime_and_declares_idempotently():
    a, b = authored()
    assert a.scene.q is None and a.scene.state is None
    count = b.particle_count
    declarations = len(b.custom_attributes)
    a.declare(b)
    assert b.particle_count == count and len(b.custom_attributes) == declarations
    b.begin_world()
    with pytest.raises(ValueError, match="active world"):
        a.add_world(b, 0)
    b.end_world()
    with pytest.raises(ValueError, match="different i4h"):
        author_case(CASE).declare(b)


def test_duplicate_world_fails_without_adding_particles():
    a = author_case(CASE)
    b = newton.ModelBuilder()
    b.begin_world()
    a.add_world(b, 0)
    count = b.particle_count
    with pytest.raises(ValueError, match="duplicate"):
        a.add_world(b, 0)
    assert b.particle_count == count


def test_binding_preserves_host_state_and_model_and_authored_history():
    a, b = authored()
    model = b.finalize(device="cpu")
    state, control = model.state(), model.control()
    state.particle_q.fill_(wp.vec3(2., 3., 4.))
    state.particle_qd.fill_(wp.vec3(4., 5., 6.))
    model.particle_q.fill_(wp.vec3(8., 8., 8.))  # native Model defaults are the live rest source
    arrays = [state.particle_q, state.particle_qd, model.particle_q, model.particle_mass,
              model.particle_inv_mass, model.particle_radius, model.particle_flags]
    before = [x.numpy().copy() for x in arrays]
    scene, systems = a.bind(model, state, control)
    for x, value in zip(arrays, before):
        np.testing.assert_array_equal(x.numpy(), value)
    assert scene.q.ptr == state.particle_q.ptr
    np.testing.assert_array_equal(scene.block_field("q_prev").numpy(), main_rows(a.scene, "q0"))
    np.testing.assert_array_equal(scene.block_field("q0").numpy(), 8.)
    np.testing.assert_array_equal(scene.block_field("inv_mass").numpy(), main_rows(a.scene, "inv_mass"))
    assert model.tet_count == model.tri_count == model.edge_count == 0
    other_scene, other_systems = a.bind(model, model.state(), model.control())
    assert other_scene is not scene and all(x is not y for x, y in zip(systems, other_systems))
    assert other_scene.q_prev.ptr != scene.q_prev.ptr
    solver = I4hSolver(model)
    with pytest.raises(RuntimeError, match="unbound"):
        solver.step(state, state, control, None, .01)
    solver.bind(scene, systems, state, control)
    with pytest.raises(RuntimeError, match="already bound"):
        solver.bind(other_scene, other_systems, other_scene.state, other_scene.control)
    solver.configure(.01, 1, 1)
    with pytest.raises(ValueError, match="bound control"):
        solver.step(state, state, model.control(), None, .01)
    with pytest.raises(ValueError, match="external collision"):
        solver.step(state, state, control, object(), .01)


def test_foreign_ownership_and_layout_rejected():
    a, b = authored()
    m = b.finalize(device="cpu")
    foreign_a, foreign_b = authored()
    foreign_m = foreign_b.finalize(device="cpu")
    for state, control in [(foreign_m.state(), m.control()), (m.state(), foreign_m.control())]:
        with pytest.raises(ValueError, match="foreign"):
            a.bind(m, state, control)
    a, b = authored(uneven=True)
    m = b.finalize(device="cpu")
    with pytest.raises(RuntimeError, match="equally sized worlds"):
        a.bind(m, m.state(), m.control())


def test_zero_sized_scene():
    cfg = copy.deepcopy(CASE)
    cfg["assets"], cfg["systems"] = {}, []
    a, b = authored(cfg)
    a.complete(b)
    m = b.finalize(device="cpu")
    scene, systems = a.bind(m, m.state(), m.control())
    assert not systems and scene.main.width == 0


def test_retained_builder_allocations_have_distinct_ownership():
    a, b = authored()
    first = b.finalize(device="cpu")
    second = b.finalize(device="cpu")
    with pytest.raises(ValueError, match="foreign"):
        a.bind(second, first.state(), second.control())
    with pytest.raises(ValueError, match="foreign"):
        a.bind(second, second.state(), first.control())


def test_replacing_host_buffer_requires_explicit_new_binding():
    from i4h_physics import load_case

    sim = load_case(CASE)
    sim.prepare()
    sim.state.particle_q = wp.clone(sim.state.particle_q)
    with pytest.raises(RuntimeError, match="buffer was replaced"):
        sim.step()
    with pytest.raises(RuntimeError, match="buffer was replaced"):
        sim.prepare()


def test_empty_instrument_geometry_is_authored_without_runtime_arrays():
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    a = author_case({"sim": {"device": "cpu"}, "systems": [{"type": "instrument_control"}]},
                    systems={"instrument_control": InstrumentControl})
    assert a.scene.instrument_layout.sphere_count == a.scene.instrument_layout.prim_count == 0
    b = newton.ModelBuilder()
    b.begin_world()
    a.add_world(b, 0)
    b.end_world()
    a.complete(b)
    m = b.finalize(device="cpu")
    scene, systems = a.bind(m, m.state(), m.control())
    assert systems[0].actions.shape == (1, 0, 10)
    assert scene.q.size == 0


def test_particle_reset_flags_touch_only_owned_selected_rows():
    a, b = authored()
    m = b.finalize(device="cpu")
    state, control = m.state(), m.control()
    scene, systems = a.bind(m, state, control)
    solver = I4hSolver(m, scene, systems, state, control)
    state.particle_q.fill_(wp.vec3(10.))
    state.particle_qd.fill_(wp.vec3(11.))
    state.particle_f.fill_(wp.vec3(12.))
    scene.q_prev.fill_(wp.vec3(13.))
    flags_before = m.particle_flags.numpy().copy()
    mask = wp.array([True, False, False], dtype=wp.bool, device="cpu")
    solver.reset(state, mask, flags=None)
    own = np.arange(a.worlds[0][0], a.worlds[0][1])
    untouched = np.setdiff1d(np.arange(m.particle_count), own)
    for array, sentinel in [(state.particle_q, 10.), (state.particle_qd, 11.), (state.particle_f, 12.), (scene.q_prev, 13.)]:
        np.testing.assert_array_equal(array.numpy()[untouched], sentinel)
    np.testing.assert_array_equal(state.particle_q.numpy()[own], main_rows(a.scene, "q0")[0])
    np.testing.assert_array_equal(state.particle_qd.numpy()[own], 0.)
    np.testing.assert_array_equal(state.particle_f.numpy()[own], 0.)
    np.testing.assert_array_equal(m.particle_flags.numpy(), flags_before)


def test_conflicting_declaration_and_bad_label_fail_before_authoring():
    a = author_case(CASE)
    b = newton.ModelBuilder()
    b.begin_world()
    with pytest.raises(KeyError):
        a.add_world(b, 0, label_template="{missing}")
    assert not a.worlds and b.particle_count == 0
    b.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
        name="case_id", namespace="i4h", dtype=str, frequency=newton.Model.AttributeFrequency.ONCE,
        default="another-case"))
    with pytest.raises(ValueError, match="conflicting"):
        a.add_world(b, 0)
    assert not a.worlds and b.particle_count == 0


def test_storage_completion_shares_tet_rows_and_preserves_allocations():
    cfg = {"sim": {"device": "cpu", "envs": 2},
           "assets": {"solid": {"type": "tet_grid", "dim": [2, 2, 2], "cell": .02}},
           "systems": [{"type": "particles"}, {"type": "tet", "body": "solid", "id": "tet"},
                       {"type": "surface", "tet": "tet", "id": "skin"}]}
    a = author_case(cfg)
    b = newton.ModelBuilder()
    with pytest.raises(ValueError, match="all worlds"):
        a.complete(b)
    for world in range(2):
        b.begin_world()
        b.add_particle((5, 6, 7), (0, 0, 0), 1.)
        a.add_world(b, world)
        b.end_world()
    with pytest.raises(RuntimeError, match="complete Scene storage before finalization"):
        b.finalize(device="cpu")
    a.complete(b)
    counts = dict(b._custom_frequency_counts)
    a.complete(b)
    assert b._custom_frequency_counts == counts
    assert a.storage_records["skin"]["variant"] is a.storage_records["tet"]["variant"]
    m = b.finalize(device="cpu")
    state, control = m.state(), m.control()
    state.i4h.tet_active.zero_()
    before = state.i4h.tet_active.numpy().tobytes()
    scene, _ = a.bind(m, state, control)
    assert scene.storage.attr(a.storage_records["tet"]["state"], "active").numpy().tobytes() == before
    assert scene.attr(a.storage_records["tet"]["state"], "active") is scene.storage.attr(a.storage_records["tet"]["state"], "active")
    other = m.state()
    assert np.any(other.i4h.tet_active.numpy())
    assert m.tet_count == m.tri_count == 0
    assert m.i4h.tet_tet_variant_owner.numpy().min() >= 1
    a.cfg["systems"].append({"type": "particles"})
    with pytest.raises(RuntimeError, match="reconstruct"):
        a.complete(b)
