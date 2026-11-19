# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C2.3 ownership, restoration, fixed layouts and scratch lifetime."""
import newton
import numpy as np
import pytest
import warp as wp
from i4h_physics import Scene, Stepper, author_case, declare_tet, tet_fields
from i4h_physics.loaders import add_tet_grid
from i4h_physics.newton_model import add_env_particles, complete_scene
from i4h_physics.systems.mechanics.tet import Tet
from i4h_physics.tet_storage import MATERIAL_FIELDS, TET_FIELDS


def authored(device, model="pbd", repeated=True):
    cfg = {"sim": {"device": device, "envs": 3, "cuda_graph": False},
           "assets": {name: {"type": "tet_grid", "dim": [1, 1, 1], "cell": .02,
                             "origin": [x, 0, 0]} for name, x in (("a", 0), ("b", .2))},
           "systems": [{"type": "particles"}, {"type": "tet", "body": "a", "id": "first", "model": model},
                       {"type": "surface", "tet": "first", "id": "skin"}]}
    if repeated:
        cfg["systems"] += [{"type": "tet", "body": "a", "id": "second"}, {"type": "tet", "body": "b", "id": "third"}]
    a = author_case(cfg)
    b = newton.ModelBuilder()
    b.add_particle((8, 9, 10), (1, 2, 3), 1.)
    for e in range(3):
        b.begin_world()
        b.add_particle((8, 9, 10), (1, 2, 3), 1.)
        a.add_world(b, e)
        b.add_particle((8, 9, 10), (1, 2, 3), 1.)
        b.end_world()
    a.complete(b)
    counts = dict(b._custom_frequency_counts)
    a.complete(b)
    assert counts == b._custom_frequency_counts
    m = b.finalize(device=device)
    m.particle_grid = None
    return a, b, m


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("shared", [False, True])
def test_selected_state_material_aliases_resets_and_foreign_rows(device, shared):
    a, builder, m = authored(device)
    s1, s2 = m.state(), m.state()
    c1 = m.control(clone_variables=not shared)
    c2 = c1 if shared else m.control()
    for name in ("active", "edge_active"):
        getattr(s1.i4h, f"first_{name}").fill_(7)
    m.i4h.first_spring.fill_(.125)
    scene, systems = a.bind(m, s1, c1)
    other, others = a.bind(m, s2, c2)
    tet, sf, second, third = systems[1:]
    fields = tet_fields(scene, "first")
    assert fields is tet_fields(scene, "first") is tet.fields is sf.tet_fields
    for key in TET_FIELDS:
        arr = getattr(fields, key)
        assert getattr(tet, key) is arr
        assert scene.attr(fields.domains["fields"][key], key) is arr
        owner = s1 if key in ("active", "edge_active") else m
        assert arr.ptr == getattr(owner.i4h, f"first_{key}").ptr
    for key in ("active", "edge_active", "tets", "edges", "edge_rest", "edge_part", "tet_part", "variant"):
        assert getattr(sf, key) is getattr(fields, key)
    assert second.variant is tet.variant
    assert second.active.ptr != tet.active.ptr and third.tets.ptr != tet.tets.ptr
    assert other.body("a").variant.ptr == tet.variant.ptr
    assert others[1].spring.ptr == tet.spring.ptr
    for key in MATERIAL_FIELDS:
        getattr(tet, key).fill_(.375)
        np.testing.assert_array_equal(getattr(others[1], key).numpy(), .375)
    before = [x.numpy().tobytes() for x in (s1.particle_q, s1.particle_qd, s1.i4h.particle_q_prev)]
    # Preparation is not reset; physical Surface preserves selected State too.
    sim = Stepper.create(scene, systems, control=c1, cuda_graph=False)
    np.testing.assert_array_equal(tet.active.numpy(), 7)
    np.testing.assert_array_equal(tet.edge_active.numpy(), 7)
    for key, default in (("active", "active0"), ("edge_active", "edge_active0")):
        initial = getattr(others[1], key).numpy().copy()
        getattr(tet, default).fill_(3)  # reset reads live Model defaults
        for selection in ([False, True, False], [False]*3, [False, True, False]):
            tet.reset(scene, wp.array(selection, dtype=wp.bool, device=device))
        actual = getattr(tet, key).numpy()
        np.testing.assert_array_equal(actual[1], 3)
        np.testing.assert_array_equal(actual[[0, 2]], 7)
        np.testing.assert_array_equal(getattr(others[1], key).numpy(), initial)
    for key in MATERIAL_FIELDS:
        np.testing.assert_array_equal(getattr(tet, key).numpy(), .375)
    assert before == [x.numpy().tobytes() for x in (s1.particle_q, s1.particle_qd, s1.i4h.particle_q_prev)]
    np.testing.assert_array_equal(tet.native_tet.numpy(), -1)
    starts = np.asarray(a.particle_starts(a.scene.body("a")))
    np.testing.assert_array_equal(m.i4h.first_tet_owner.numpy().reshape(3, -1), starts[:, None] + a.scene.body("a").tet.tets[0, :, 0])
    assert m.tet_count == m.tri_count == m.edge_count == 0
    # A retained builder creates independent templates, never duplicate rows.
    rebuilt = builder.finalize(device=device)
    np.testing.assert_array_equal(rebuilt.i4h.first_spring.numpy(), a.scene.body("a").tet.spring[0])
    assert sim.scene is scene


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("key", TET_FIELDS)
def test_all_migrated_buffers_and_compatibility_aliases_are_guarded(device, key):
    a, _, m = authored(device, repeated=False)
    s, c = m.state(), m.control()
    scene, systems = a.bind(m, s, c)
    tet = systems[1]
    old = getattr(tet, key)
    setattr(tet, key, wp.clone(old))
    with pytest.raises(RuntimeError, match="buffer was replaced"):
        tet_fields(scene, "first")
    setattr(tet, key, old)
    owner = s if key in ("active", "edge_active") else m
    name = f"first_{key}"
    backing = getattr(owner.i4h, name)
    setattr(owner.i4h, name, wp.clone(backing))
    with pytest.raises(RuntimeError, match="buffer was replaced"):
        scene.storage.check()
    setattr(owner.i4h, name, wp.zeros(1, dtype=wp.float64, device=device))
    with pytest.raises(ValueError, match="incompatible persistent buffer"):
        a.bind(m, s, c)


@pytest.mark.parametrize("change", ["edges", "materials", "topology", "mapping"])
def test_changed_topology_dimensions_and_mapping_require_reconstruction(change):
    a, b, m = authored("cpu", repeated=False)
    body = a.scene.body("a")
    if change == "edges":
        body.tet.edges = body.tet.edges[:, :-1]
    elif change == "materials":
        body.tet.spring = np.append(body.tet.spring, np.float32(1))
    elif change == "topology":
        body.tet.tets[0, 0, 0] += 1
    else:
        body.variant_of_env[0] += 1
    for action in (lambda: a.complete(b), lambda: a.bind(m, m.state(), m.control())):
        with pytest.raises(RuntimeError, match="reconstruct"):
            action()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_direct_scene_predeclaration_and_missing_declaration(device):
    scene = Scene(device, 2)
    add_tet_grid(scene, "body", [1, 1, 1], .02)
    declare_tet(scene, "body")
    declare_tet(scene, "body", system_id="other")
    builder = newton.ModelBuilder()
    builder.add_particle((9, 8, 7), (0, 0, 0), 1.)
    for e in range(2):
        builder.begin_world()
        builder.add_particle((9, 8, 7), (0, 0, 0), 1.)
        add_env_particles(builder, scene, e, xform=((0, 0, 0), (0, 0, 0, 1)))
        builder.end_world()
    complete_scene(builder, scene)
    with pytest.raises(RuntimeError, match="reconstruct"):
        declare_tet(scene, "body", system_id="late")
    m = builder.finalize(device=device)
    scene.allocate(m, m.state())
    assert Tet(scene, "body").id == "tet"
    assert Tet(scene, "body", id="other").fields is tet_fields(scene, "other")
    with pytest.raises(RuntimeError, match="declare_tet.*reconstruct"):
        Tet(scene, "body", id="missing")
    missing = Scene(device, 1)
    add_tet_grid(missing, "body", [1, 1, 1], .02)
    missing.allocate()
    with pytest.raises(RuntimeError, match="declare_tet.*reconstruct"):
        Tet(missing, "body")


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("model", ["pbd", "xpbd"])
@pytest.mark.parametrize("capture", [False, True])
def test_scratch_lifetime_and_eager_capture_equivalence(model, capture, device):
    a, _, m = authored(device, model=model, repeated=False)
    scene, systems = a.bind(m, m.state(), m.control())
    other, other_systems = a.bind(m, m.state(), m.control())
    tet = systems[1]
    sim = Stepper.create(scene, systems, cuda_graph=capture, substeps=2, iterations=3)
    ref = Stepper.create(other, other_systems, cuda_graph=False, substeps=2, iterations=3)
    for _ in range(3):
        sim.step()
        ref.step()
    np.testing.assert_allclose(scene.q.numpy(), other.q.numpy(), atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(tet.delta.numpy(), 0)
    np.testing.assert_array_equal(tet.delta_count.numpy(), 0)
    tet.edge_lambda.fill_(5)
    tet.tet_lambda.fill_(6)
    tet.substep_start(scene, .01, 0, 2)
    np.testing.assert_array_equal(tet.edge_lambda.numpy(), 0 if model == "xpbd" else 5)
    np.testing.assert_array_equal(tet.tet_lambda.numpy(), 0 if model == "xpbd" else 6)
    if model == "pbd":
        assert tet.edge_lambda.shape == tet.tet_lambda.shape == (1, 1)
    else:
        assert tet.edge_lambda.shape == tet.edge_active.shape
        tet.spring.fill_(1000)
        tet.volume.fill_(1000)
        # A solve accumulates the current multipliers; iteration does not clear them.
        tet.edge_lambda.fill_(1)
        tet.iteration(scene, .01, 0, 3)
        assert np.any(tet.edge_lambda.numpy() != 0)


def test_measurement_accounts_maximum_alias_extent_and_tet_formulas():
    import runpy
    from pathlib import Path

    from i4h_physics.tet_storage import estimate_tet_bytes
    helpers = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/measure_stage_c_construction.py"))
    full = wp.zeros(7, dtype=wp.vec3, device="cpu")
    short = full[:3]
    assert helpers["byte_count"]([full, short]) == helpers["byte_count"]([short, full]) == 84
    a, _, m = authored("cpu")
    scene, _ = a.bind(m, m.state(), m.control())
    reports = helpers["tet_accounting"](scene)
    total = 0
    for sid, row in reports.items():
        record = scene.tet_domains[sid]
        model_bytes = state_bytes = 0
        for key, domain in record["fields"].items():
            array = scene.attr(domain, key)
            nbytes = array.size * wp.types.type_size_in_bytes(array.dtype)
            if key in ("active", "edge_active"):
                state_bytes += nbytes
            else:
                model_bytes += nbytes
        assert row["model_data_bytes"] == model_bytes
        assert row["each_state_bytes"] == row["initialization_template_bytes"] == state_bytes
        assert row["c22_additive_model_bytes"] + row["new_model_data_bytes"] == model_bytes
        assert row["c22_additive_state_bytes"] + row["new_state_bytes"] == state_bytes
        total += model_bytes + 2*state_bytes + row["identity_column_bytes"]
    assert total == estimate_tet_bytes(scene)
