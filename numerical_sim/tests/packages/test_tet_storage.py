# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-tet declaration values, sharing, padding and native owner references."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
from i4h_physics.storage import Declarations
from i4h_physics.tet_storage import declare_surface_tet_fields, declare_tet_fields
from i4h_surgical.tet_storage import declare_thermal_tet_fields


def test_complete_per_tet_set_preserves_variants_local_columns_and_exact_defaults():
    mesh = SimpleNamespace(
        tets=np.array([[[0, 1, 2, 3], [1, 2, 3, 4]], [[2, 3, 4, 5], [0, 0, 0, 0]]], np.int32),
        active0=np.array([[1, 1], [1, 0]], np.int32),
        rest_volume=np.array([[.123456789, .24], [.33, 0]], np.float32),
        part=np.array([0, 1], np.int32),
        tet_edges=np.arange(24, dtype=np.int32).reshape(2, 2, 6),
        face_neighbor=np.arange(16, dtype=np.int32).reshape(2, 2, 4),
        face_tri=np.arange(16, dtype=np.int32).reshape(2, 2, 4) - 1)
    add_edges_materials(mesh)
    body = SimpleNamespace(tet=mesh, variant_of_env=np.array([1, 0, 1], np.int32))
    starts = [2, 10, 18]
    d = Declarations()
    variant, state_domain, _ = declare_tet_fields(d, body, "tissue", starts)
    surface = declare_surface_tet_fields(d, body, "surface", starts, tet_body=body, tet_variant=variant)
    assert surface is variant
    with pytest.raises(ValueError, match="same body"):
        declare_surface_tet_fields(d, SimpleNamespace(tet=mesh), "wrong", starts, tet_body=body, tet_variant=variant)
    second_surface = declare_surface_tet_fields(d, body, "surface2", starts, tet_body=body, tet_variant=variant)
    assert second_surface is variant
    thermal = declare_thermal_tet_fields(d, body, "heat", starts)
    b = newton.ModelBuilder()
    d.register(b)
    assert "i4h:surface_tet_variant" not in b.custom_frequencies
    assert "i4h:surface_tet_edges" in b.custom_attributes
    for _ in range(3):
        b.begin_world()
        b.add_particles([(0, 0, 0)] * 8, [(0, 0, 0)] * 8, [1.] * 8)
        b.end_world()
    m = b.finalize(device="cpu")
    s1, s2, c = m.state(), m.state(), m.control()
    v1, v2 = d.bind(m, s1, c), d.bind(m, s2, c)
    assert m.tet_count == m.tri_count == m.edge_count == 0
    assert len(d.fields) == 23
    np.testing.assert_array_equal(m.i4h.surface2_tet_edges.numpy(), mesh.tet_edges.reshape(-1, 6))
    np.testing.assert_array_equal(m.i4h.tissue_tet_owner.numpy(), [4, -1, 10, 11, 20, -1])
    np.testing.assert_array_equal(m.i4h.tissue_tet_world.numpy(), [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(m.i4h.tissue_tet_variant_world.numpy(), -1)
    assert v1.attr(variant, "rest_volume").numpy().tobytes() == mesh.rest_volume.tobytes()
    assert v1.attr(variant, "tets").shape == (2, 2)
    assert v1.attr(variant, "tets").ptr == v2.attr(variant, "tets").ptr
    np.testing.assert_array_equal(v1.attr(surface, "surface_tet_edges").numpy(), mesh.tet_edges)
    np.testing.assert_array_equal(v1.attr(surface, "surface_face_neighbor").numpy(), mesh.face_neighbor)
    np.testing.assert_array_equal(v1.attr(state_domain, "active").numpy(), mesh.active0[body.variant_of_env])
    v1.attr(state_domain, "active").zero_()
    v1.attr(thermal, "heat").fill_(10.)
    np.testing.assert_array_equal(v2.attr(state_domain, "active").numpy(), mesh.active0[body.variant_of_env])
    np.testing.assert_array_equal(v2.attr(thermal, "heat").numpy(), 0.)


def test_zero_world_tet_declarations_keep_shared_data_and_empty_state():
    mesh = SimpleNamespace(tets=np.array([[[0, 1, 2, 3]]], np.int32),
                           active0=np.ones((1, 1), np.int32),
                           rest_volume=np.ones((1, 1), np.float32),
                           part=np.zeros(1, np.int32))
    add_edges_materials(mesh)
    body = SimpleNamespace(tet=mesh, variant_of_env=np.empty(0, np.int32))
    d = Declarations()
    variant, state, parts = declare_tet_fields(d, body, "empty", [])
    b = newton.ModelBuilder()
    d.register(b)
    m = b.finalize(device="cpu")
    views = d.bind(m, m.state(), m.control())
    assert views.attr(state, "active").shape == (0, 1)
    np.testing.assert_array_equal(views.attr(variant, "tets").numpy(), mesh.tets)
    np.testing.assert_array_equal(views.attr(parts, "tet_part").numpy(), mesh.part)
    np.testing.assert_array_equal(m.i4h.empty_tet_part_owner.numpy(), [-1])
    assert m.particle_count == m.tet_count == 0


def add_edges_materials(mesh):
    mesh.edges = mesh.tets[:, :, :2].copy()
    mesh.edge_active0 = mesh.active0.copy()
    mesh.edge_rest = mesh.rest_volume.copy()
    mesh.edge_part = mesh.part.copy()
    for name in ("spring", "spring_damping", "volume", "volume_damping"):
        setattr(mesh, name, np.full(max(1, mesh.part.max(initial=0) + 1), .375, np.float32))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("empty", [False, True])
def test_edge_material_domains_unused_variants_padding_and_empty_backing(device, empty):
    from i4h_physics.tet_storage import MATERIAL_FIELDS
    T = 0 if empty else 2
    mesh = SimpleNamespace(tets=np.tile(np.int32([[[0, 1, 2, 3], [1, 2, 3, 4]]]), (3, 1, 1))[:, :T],
                           active0=np.ones((3, T), np.int32), rest_volume=np.full((3, T), .123456789, np.float32),
                           part=np.arange(T, dtype=np.int32))
    if T:
        mesh.active0[1, 1] = 0
    add_edges_materials(mesh)
    if empty:
        for key in MATERIAL_FIELDS:
            setattr(mesh, key, np.empty(0, np.float32))
    body = SimpleNamespace(tet=mesh, variant_of_env=np.array([1, 0], np.int32))
    starts = [3, 12]
    d = Declarations()
    declare_tet_fields(d, body, "first", lambda: starts)
    declare_tet_fields(d, body, "second", lambda: starts)
    b = newton.ModelBuilder()
    d.register_schema(b)
    b.add_particles([(0, 0, 0)] * 3, [(0, 0, 0)] * 3, [1.] * 3)
    for _ in range(2):
        b.begin_world()
        b.add_particles([(0, 0, 0)] * 9, [(0, 0, 0)] * 9, [1.] * 9)
        b.end_world()
    d.author_rows(b)
    m = b.finalize(device=device)
    state = m.state()
    views = d.bind(m, state, m.control())
    for (domain, key), field in d.fields.items():
        assert field.references is None
        arr = views.attr(domain, key)
        if key in MATERIAL_FIELDS:
            expected = np.tile(getattr(mesh, key)[None], (2, 1))
        elif key == "tet_part":
            expected = mesh.part
        elif key in ("active", "edge_active"):
            expected = getattr(mesh, key + "0")[body.variant_of_env]
        elif key == "native_tet":
            expected = np.full((2, T), -1, np.int32)
        else:
            expected = getattr(mesh, key)
        assert arr.numpy().tobytes() == expected.tobytes()
        assert arr.shape == field.shape
    np.testing.assert_array_equal(m.i4h.first_edge_variant_world.numpy(), -1)
    if not empty:
        np.testing.assert_array_equal(m.i4h.first_edge_variant_owner.numpy(), [12, 13, 3, -1, -1, -1])
        np.testing.assert_array_equal(m.i4h.first_edge_owner.numpy(), [3, -1, 12, 13])
        np.testing.assert_array_equal(m.i4h.first_material_owner.numpy(), [3, -1, 12, 13])
        np.testing.assert_array_equal(m.i4h.first_material_world.numpy(), [0, 0, 1, 1])
    else:
        assert m.i4h.first_edges.shape == (1,)
        assert state.i4h.first_edge_active.shape == (1,)
        np.testing.assert_array_equal(m.i4h.first_edge_owner.numpy(), -1)
    assert m.i4h.first_edges.ptr != m.i4h.second_edges.ptr
    assert m.tet_count == m.tri_count == m.edge_count == 0
    assert views.allocated_bytes()["state"] == 2 * 8 * max(1, 2*T)
