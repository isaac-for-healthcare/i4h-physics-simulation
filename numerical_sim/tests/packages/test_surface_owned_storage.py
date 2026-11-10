# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C2.4 aliases, authored identity, restored topology and selected-world reset."""
from copy import deepcopy
from pathlib import Path
import runpy

import newton
import numpy as np
import pytest
import warp as wp

from i4h_physics import Scene, Stepper, author_case, declare_surface, declare_tet, surface_fields
from i4h_physics.loaders import add_tet_grid
from i4h_physics.surface_storage import MODEL_FIELDS, STATE_FIELDS, SURFACE_FIELDS, estimate_surface_bytes
from i4h_physics.systems.mechanics.surface import Surface
from i4h_physics.systems.mechanics.tet import Tet


def authored(device, empty=False):
    cfg = {"sim": {"device": device, "envs": 3, "cuda_graph": False},
           "assets": {name: {"type": "tet_grid", "dim": [1, 1, 1], "cell": .02,
                             "origin": [x, 0, 0]} for name, x in (("a", 0), ("b", .2))},
           "systems": [{"type": "particles"}, {"type": "tet", "body": "a", "id": "first"},
                       {"type": "surface", "tet": "first", "id": "skin"},
                       {"type": "surface", "tet": "first", "id": "second"},
                       {"type": "tet", "body": "b", "id": "third"},
                       {"type": "surface", "tet": "third", "id": "other"}]}
    a = author_case(cfg)
    body = a.scene.body("a")
    body.variants = 3
    body.variant_of_env[:] = [1, 0, 1]
    m = body.tet
    for key in ("tets", "active0", "rest_volume", "edges", "edge_active0", "edge_rest", "tris", "tri_tet",
                "support0", "edge_support0", "tet_edges", "face_neighbor", "face_tri", "uvs"):
        setattr(m, key, np.repeat(getattr(m, key), 3, axis=0))
    # An unused variant, inactive padding and reserved face slots remain authored bytes.
    m.tris[1, -1] = 0
    m.tri_tet[1, -1] = -1
    m.support0[1, -1] = 0
    body.rest.flags[[0, 2], -1] = 0
    body.rest.flags[2, 0] = 0
    if empty:
        for key in ("tets", "active0", "rest_volume", "edges", "edge_active0", "edge_rest", "tris", "tri_tet",
                    "edge_support0", "tet_edges", "face_neighbor", "face_tri"):
            setattr(m, key, getattr(m, key)[:, :0])
        for key in ("part", "edge_part", "tri_part", "spring", "spring_damping", "volume", "volume_damping",
                    "heat", "deletable", "cut_start", "cut_capacity"):
            setattr(m, key, getattr(m, key)[:0])
        m.support0[:] = 0
        body.parts.clear()
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
    model = b.finalize(device=device)
    model.particle_grid = None
    return a, b, model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("empty", [False, True])
def test_exact_defaults_identity_and_empty_backing(device, empty):
    a, b, m = authored(device, empty)
    scene, systems = a.bind(m, m.state(), m.control())
    f = surface_fields(scene, "skin")
    mesh, selected = f.body.tet, f.body.variant_of_env
    for key in SURFACE_FIELDS:
        source = {"tris0": "tris", "tri_tet0": "tri_tet", "heat_enabled": "heat",
                  "support": "support0", "edge_support": "edge_support0"}.get(key, key)
        arr = getattr(f, key)
        if key in ("cut_count", "overflow", "deleted", "dirty"):
            expected = np.zeros(arr.shape, np.int32)
        else:
            expected = getattr(mesh, source)
            if key in STATE_FIELDS:
                expected = expected[selected]
            if key == "support":
                expected = expected.reshape(-1)
        assert arr.numpy().tobytes() == expected.tobytes()
        assert getattr(systems[2], key) is arr
        assert arr is scene.attr(*f.domains["fields"][key])
        field = scene.declarations.fields[f.domains["fields"][key]]
        assert field.references is None
        if arr.size:
            assert arr.ptr != getattr(systems[3], key).ptr
    for label in ("face_variant", "support_variant"):
        owner = getattr(m.i4h, "skin_" + label + "_owner").numpy()
        np.testing.assert_array_equal(getattr(m.i4h, "skin_" + label + "_world").numpy(), -1)
        if owner.size > 1:
            assert (owner.reshape(3, -1)[2] == -1).all()
    if not empty:
        face_owners = m.i4h.skin_face_owner.numpy().reshape(3, -1)
        starts = a.particle_starts(a.scene.body("a"))
        for e, v in enumerate(selected):
            tri = mesh.tris[v]
            valid = (tri[:, 0] != tri[:, 1]) & (tri[:, 0] != tri[:, 2]) & (tri[:, 1] != tri[:, 2])
            np.testing.assert_array_equal(face_owners[e], np.where(valid, starts[e] + tri[:, 0], -1))
        assert m.i4h.skin_support_owner.numpy().reshape(3, -1)[0, -1] == -1
        assert m.i4h.skin_support_owner.numpy().reshape(3, -1)[2, 0] == -1
    else:
        assert m.i4h.skin_tris0.shape == (1,)
        assert systems[2].tris.shape == (3, 0)
        assert m.i4h.skin_face_owner.numpy().tolist() == [-1]
    assert m.tet_count == m.tri_count == m.edge_count == 0
    rebuilt = b.finalize(device=device)
    assert rebuilt.i4h.skin_tris0.numpy().tobytes() == m.i4h.skin_tris0.numpy().tobytes()
    helpers = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/measure_stage_c_construction.py"))
    total = 0
    for sid, row in helpers["surface_accounting"](scene).items():
        fields = surface_fields(scene, sid)
        model_bytes = sum(getattr(fields, k).size * wp.types.type_size_in_bytes(getattr(fields, k).dtype) for k in MODEL_FIELDS)
        state_bytes = sum(getattr(fields, k).size * wp.types.type_size_in_bytes(getattr(fields, k).dtype) for k in STATE_FIELDS)
        assert model_bytes == row["model_data_bytes"]
        assert state_bytes == row["each_state_bytes"] == row["initialization_template_bytes"]
        padding = row["empty_domain_field_bytes"]
        total += row["model_data_bytes"] + 2*row["each_state_bytes"] + row["identity_column_bytes"]
        total += padding["model"] + 2*padding["state"]
    assert total == estimate_surface_bytes(scene)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("shared", [False, True])
def test_binding_preserves_all_selected_bytes_and_reset_reads_current_model(device, shared):
    a, _, m = authored(device)
    s, other = m.state(), m.state()
    c = m.control(clone_variables=not shared)
    before_other = {k: getattr(other.i4h, "skin_"+k).numpy().tobytes() for k in STATE_FIELDS}
    templates = {k: getattr(m.i4h, "skin_"+k).numpy().tobytes() for k in STATE_FIELDS}
    for key in STATE_FIELDS:
        arr = getattr(s.i4h, "skin_"+key)
        arr.fill_(wp.vec3i(4, 5, 6) if key == "tris" else 7)
    m.i4h.skin_heat_enabled.fill_(5)
    scene, systems = a.bind(m, s, c)
    sf = systems[2]
    before = {k: getattr(sf, k).numpy().tobytes() for k in STATE_FIELDS}
    Stepper.create(scene, systems, cuda_graph=False)
    again, rebound = a.bind(m, s, c)
    assert before == {k: getattr(rebound[2], k).numpy().tobytes() for k in STATE_FIELDS}
    assert sf.fields is surface_fields(scene, "skin")
    # Valid edits to shared defaults and the current variant table drive reset.
    for key in ("tris0", "tri_tet0", "support0", "edge_support0"):
        arr = getattr(sf, key)
        values = arr.numpy()
        values[2] = 2
        arr.assign(values)
    sf.variant.assign(np.array([1, 2, 1], np.int32))
    native = [x.numpy().tobytes() for x in (s.particle_q, s.particle_qd, s.i4h.particle_flags)]
    sf.active.fill_(9)
    for selection in ([False, True, False], [False]*3, [False, True, False]):
        sf.reset(scene, wp.array(selection, dtype=wp.bool, device=device))
    for key in STATE_FIELDS:
        values = getattr(sf, key).numpy().reshape((3, -1))
        assert (values[1] == (0 if key in ("cut_count", "deleted", "dirty", "overflow") else 2)).all()
        original = np.frombuffer(before[key], dtype=np.int32).reshape(3, -1)
        np.testing.assert_array_equal(values[[0, 2]], original[[0, 2]])
    np.testing.assert_array_equal(sf.active.numpy(), 9)
    np.testing.assert_array_equal(sf.heat_enabled.numpy(), 5)
    assert native == [x.numpy().tobytes() for x in (s.particle_q, s.particle_qd, s.i4h.particle_flags)]
    assert before_other == {k: getattr(other.i4h, "skin_"+k).numpy().tobytes() for k in STATE_FIELDS}
    assert templates == {k: getattr(m.i4h, "skin_"+k).numpy().tobytes() for k in STATE_FIELDS}
    assert again is not scene


@pytest.mark.parametrize("key", SURFACE_FIELDS)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_every_buffer_and_cached_alias_is_checked(key, device):
    a, _, m = authored(device)
    s, c = m.state(), m.control()
    scene, systems = a.bind(m, s, c)
    sf = systems[2]
    arr = getattr(sf, key)
    for obj in (sf, sf.fields):
        setattr(obj, key, wp.clone(arr))
        with pytest.raises(RuntimeError, match="buffer was replaced"):
            surface_fields(scene, "skin")
        setattr(obj, key, arr)
    owner = m if key in MODEL_FIELDS else s
    backing = getattr(owner.i4h, "skin_"+key)
    setattr(owner.i4h, "skin_"+key, wp.clone(backing))
    with pytest.raises(RuntimeError, match="buffer was replaced"):
        scene.storage.check()
    setattr(owner.i4h, "skin_"+key, wp.zeros(1, dtype=wp.float64, device=device))
    with pytest.raises(ValueError, match="incompatible persistent buffer"):
        a.bind(m, s, c)


@pytest.mark.parametrize("key", ["tris", "tri_tet", "tri_part", "cut_start", "cut_capacity", "tet_edges", "face_tri", "support0"])
def test_changed_authoring_requires_reconstruction(key):
    a, b, m = authored("cpu")
    getattr(a.scene.body("a").tet, key).flat[0] += 1
    for action in (lambda: a.complete(b), lambda: a.bind(m, m.state(), m.control())):
        with pytest.raises(RuntimeError, match="reconstruct"):
            action()


def test_direct_declarations_late_missing_wrong_references_and_appearance_side_effects(monkeypatch):
    from i4h_surgical.systems.mechanics.surface import Surface as SurgicalSurface
    import i4h_surgical.systems.mechanics.surface as surgical
    scene = Scene("cpu", 2)
    add_tet_grid(scene, "body", [1, 1, 1], .02)
    with pytest.raises(ValueError, match="matching Tet"):
        declare_surface(scene)
    declare_tet(scene, "body")
    declare_tet(scene, "body", system_id="other")
    declare_surface(scene)
    declare_surface(scene, system_id="second")
    with pytest.raises(ValueError, match="duplicate"):
        declare_surface(scene)
    scene.allocate()
    tet = Tet(scene, "body")
    sf = Surface(scene, tet)
    assert sf.id == "surface" and sf.fields is surface_fields(scene)
    assert Surface(scene, tet, id="second").tris.ptr != sf.tris.ptr
    def forbidden(*args):
        raise AssertionError("Appearance allocated before reference validation")
    monkeypatch.setattr(surgical, "Appearance", forbidden)
    with pytest.raises(RuntimeError, match="referenced Tet.*reconstruct"):
        SurgicalSurface(scene, Tet(scene, "body", id="other"))
    with pytest.raises(RuntimeError, match="declare_surface.*reconstruct"):
        SurgicalSurface(scene, tet, id="late")
    with pytest.raises(RuntimeError, match="reconstruct"):
        declare_surface(scene, system_id="late")
    other = Scene("cpu", 1)
    add_tet_grid(other, "body", [1, 1, 1], .02)
    declare_tet(other, "body")
    other.allocate()
    with pytest.raises(RuntimeError, match="declare_surface.*reconstruct"):
        Surface(other, Tet(other, "body"))


def test_face_capacity_validation_precedes_allocation():
    a, _, _ = authored("cpu")
    cfg = deepcopy(a.cfg)
    fresh = author_case(cfg)
    fresh.scene.body("a").tet.tris = np.zeros((1, 0, 3), np.int32)
    with pytest.raises(ValueError, match="fewer triangle slots"):
        fresh.declare(newton.ModelBuilder())
