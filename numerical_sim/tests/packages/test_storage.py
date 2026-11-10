# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Newton allocation contracts for the explicit Stage C declaration layer."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from i4h_physics.storage import Declarations, Domain, Field, Owner

FIRST = Domain("system", "first", "i4h:first_rows", (2, 3), "particle")
SECOND = Domain("system", "second", "i4h:second_rows", (2, 3), "particle")
EMPTY = Domain("system", "first", "i4h:first_empty", (2, 0, 3), "particle")


def declared(*domains):
    d = Declarations()
    for domain in domains:
        if domain.custom:
            d.domain(domain, worlds=np.full(domain.shape, -1, np.int32), owners=np.full(domain.shape, -1, np.int32))
        else:
            d.domain(domain)
    return d


def field(domain=FIRST, key="history", owner=Owner.STATE, dtype=wp.float32, shape=(2, 3), **kwargs):
    return Field(domain, key, owner, dtype, shape, "world, local slot", "restore authored rows", **kwargs)


def authored(device="cpu"):
    declarations = declared(FIRST, SECOND, EMPTY)
    declarations.add(field(owner=Owner.MODEL, key="rest", dtype=wp.vec3), np.arange(18).reshape(2, 3, 3))
    declarations.add(field(), np.arange(6).reshape(2, 3))
    declarations.add(field(domain=SECOND), np.full((2, 3), 17))
    declarations.add(field(key="commands", owner=Owner.CONTROL), np.arange(6).reshape(2, 3))
    declarations.add(field(domain=EMPTY, key="empty", shape=(2, 0, 3)), np.empty((2, 0, 3)))
    b = newton.ModelBuilder()
    declarations.register(b)
    m = b.finalize(device=device)
    return declarations, b, m


def test_independent_states_default_controls_shared_model_and_cached_views():
    d, _, m = authored()
    s1, s2, c1, c2 = m.state(), m.state(), m.control(), m.control()
    a, b = d.bind(m, s1, c1), d.bind(m, s2, c2)
    assert a.attr(FIRST, "history") is a.attr(FIRST, "history")
    assert a.attr(FIRST, "rest").ptr == b.attr(FIRST, "rest").ptr
    assert a.attr(FIRST, "history").ptr != b.attr(FIRST, "history").ptr
    assert a.attr(FIRST, "commands").ptr != b.attr(FIRST, "commands").ptr
    a.attr(FIRST, "history").fill_(41.)
    a.attr(FIRST, "commands").fill_(42.)
    np.testing.assert_array_equal(b.attr(FIRST, "history").numpy(), np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(b.attr(FIRST, "commands").numpy(), np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(a.attr(SECOND, "history").numpy(), 17)
    assert a.attr(EMPTY, "empty").shape == (2, 0, 3)
    assert a.allocated_bytes() == dict(model=176, templates=76, state=52, control=24, control_shared_with_template=0)


def test_intentional_control_sharing_and_template_accounting():
    d, _, m = authored()
    a = d.bind(m, m.state(), m.control(clone_variables=False))
    b = d.bind(m, m.state(), m.control(clone_variables=False))
    assert a.attr(FIRST, "commands").ptr == b.attr(FIRST, "commands").ptr
    a.attr(FIRST, "commands").fill_(43.)
    np.testing.assert_array_equal(b.attr(FIRST, "commands").numpy(), 43)
    assert a.allocated_bytes()["control_shared_with_template"] == 24
    # Default controls copy the current template but remain independent.
    c = d.bind(m, m.state(), m.control())
    assert c.attr(FIRST, "commands").ptr != a.attr(FIRST, "commands").ptr
    np.testing.assert_array_equal(c.attr(FIRST, "commands").numpy(), 43)


def test_fresh_binding_does_not_initialize_history_or_commands():
    d, _, m = authored()
    s, c = m.state(), m.control()
    a = d.bind(m, s, c)
    a.attr(FIRST, "history").fill_(-17.)
    a.attr(FIRST, "commands").fill_(29.)
    b = d.bind(m, s, c)
    for key in ("history", "commands"):
        assert a.attr(FIRST, key).ptr == b.attr(FIRST, key).ptr
        assert a.attr(FIRST, key).numpy().tobytes() == b.attr(FIRST, key).numpy().tobytes()


def test_registration_is_idempotent_but_late_declarations_require_reconstruction():
    d, builder, _ = authored()
    count = len(builder.custom_attributes)
    d.register(builder)
    assert count == len(builder.custom_attributes)
    with pytest.raises(RuntimeError, match="reconstruct"):
        d.add(field(key="late"), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="fresh declarations"):
        d.register(newton.ModelBuilder())
    with pytest.raises(ValueError, match="already contains"):
        Declarations().register(builder)


def test_same_authoring_different_finalization_is_foreign():
    d, builder, first = authored()
    second = builder.finalize(device="cpu")
    for state, control in ((first.state(), second.control()), (second.state(), first.control())):
        with pytest.raises(ValueError, match="foreign"):
            d.bind(second, state, control)


@pytest.mark.parametrize("replace", ["namespace", "array", "layout"])
def test_nested_replacement_rejected(replace):
    d, _, m = authored()
    s, c = m.state(), m.control()
    a = d.bind(m, s, c)
    name = d._names[FIRST, "history"]
    if replace == "namespace":
        s.i4h = SimpleNamespace(**vars(s.i4h))
    elif replace == "array":
        setattr(s.i4h, name, wp.clone(getattr(s.i4h, name)))
    else:
        getattr(s.i4h, name).shape = (2, 3)
    with pytest.raises(RuntimeError, match="replaced|layout"):
        a.check()


def test_validation_before_device_allocation():
    d = declared(FIRST)
    with pytest.raises(ValueError, match="shape"):
        d.add(field(), np.zeros(6))
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    d.add(field(), values)
    values.fill(99)
    np.testing.assert_array_equal(d._values[FIRST, "history"], np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError, match="duplicate"):
        d.add(field(), values)
    with pytest.raises(TypeError, match="Newton enum or string"):
        field(references=123)
    with pytest.raises(ValueError, match="sentinel"):
        field(key="native_tet", references=newton.Model.AttributeFrequency.TETRAHEDRON)


def test_empty_schema_and_local_sentinel_do_not_author_native_topology():
    d = declared(FIRST)
    d.add(field(key="native_tet", dtype=wp.int32), np.full((2, 3), -1, np.int32))
    b = newton.ModelBuilder()
    d.register(b)
    m = b.finalize(device="cpu")
    assert m.tet_count == m.tri_count == m.edge_count == 0
    assert b.custom_attributes[f"i4h:{d._names[FIRST, 'native_tet']}"].references is None
    np.testing.assert_array_equal(d.bind(m, m.state(), m.control()).attr(FIRST, "native_tet").numpy(), -1)
    empty = Declarations()
    b = newton.ModelBuilder()
    empty.register(b)
    m = b.finalize(device="cpu")
    views = empty.bind(m, m.state(), m.control())
    views.check()
    assert sum(views.allocated_bytes().values()) == 0


@pytest.mark.parametrize("dtype,components", [(wp.int32, ()), (wp.bool, ()), (wp.quat, (4,)),
                                             (wp.transform, (7,)), (wp.mat33, (3, 3))])
def test_preserves_scalar_vector_and_matrix_layouts(dtype, components):
    d = declared(FIRST)
    shape = (2, 3) + components
    values = np.arange(np.prod(shape)).reshape(shape).astype(wp.dtype_to_numpy(getattr(dtype, "_wp_scalar_type_", dtype)))
    d.add(field(dtype=dtype), values)
    b = newton.ModelBuilder()
    d.register(b)
    m = b.finalize(device="cpu")
    view = d.bind(m, m.state(), m.control()).attr(FIRST, "history")
    assert view.shape == (2, 3)
    np.testing.assert_array_equal(view.numpy(), values)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bidirectional_torch_aliases(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    torch = pytest.importorskip("torch")
    d, _, m = authored(device)
    s, c = m.state(), m.control()
    views = d.bind(m, s, c)
    commands = views.attr(FIRST, "commands")
    tensor = wp.to_torch(commands)
    assert tensor.data_ptr() == commands.ptr
    tensor[:, 1].fill_(91.)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    np.testing.assert_array_equal(commands.numpy()[:, 1], 91.)
    commands.fill_(63.)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(tensor.cpu().numpy(), 63.)
    fresh = d.bind(m, s, c)
    assert fresh.attr(FIRST, "commands").ptr == tensor.data_ptr()


def test_owned_fields_do_not_touch_native_foreign_particles():
    d = declared(FIRST)
    d.add(field(), np.zeros((2, 3)))
    b = newton.ModelBuilder()
    for _ in range(2):
        b.begin_world()
        b.add_particle((9., 8., 7.), (6., 5., 4.), 2.)
        b.end_world()
    d.register(b)
    m = b.finalize(device="cpu")
    s, c = m.state(), m.control()
    before = s.particle_q.numpy().tobytes(), s.particle_qd.numpy().tobytes(), m.particle_flags.numpy().tobytes()
    views = d.bind(m, s, c)
    views.attr(FIRST, "history").fill_(100.)
    assert before == (s.particle_q.numpy().tobytes(), s.particle_qd.numpy().tobytes(), m.particle_flags.numpy().tobytes())


def test_incompatible_buffers_rejected_at_binding():
    d, _, m = authored()
    s, c = m.state(), m.control()
    name = d._names[FIRST, "history"]
    setattr(s.i4h, name, wp.zeros(6, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="incompatible"):
        d.bind(m, s, c)
    setattr(s.i4h, name, wp.zeros(12, dtype=wp.float32, device="cpu")[::2])
    with pytest.raises(ValueError, match="incompatible"):
        d.bind(m, s, c)


def test_one_row_call_for_all_domain_fields_and_readable_names(monkeypatch):
    d = declared(FIRST)
    for name, owner in (("rest", Owner.MODEL), ("history", Owner.STATE), ("actions", Owner.CONTROL)):
        d.add(field(key=name, owner=owner), np.ones((2, 3)))
    b = newton.ModelBuilder()
    calls = []
    original = b.add_custom_values
    def record(**values):
        calls.append(values)
        return original(**values)
    monkeypatch.setattr(b, "add_custom_values", record)
    d.register(b)
    assert [f for f in b.custom_frequencies if f.startswith("i4h:")] == [FIRST.frequency]
    assert len(calls) == FIRST.count
    assert all(set(row) == {"i4h:first_rest", "i4h:first_history", "i4h:first_actions",
                            "i4h:first_rows_world", "i4h:first_rows_owner"} for row in calls)
    assert b.custom_attributes["i4h:first_rows_world"].references == "world"
    assert b.custom_attributes["i4h:first_rows_owner"].references == "particle"


@pytest.mark.parametrize("reference", ["particle", newton.Model.AttributeFrequency.PARTICLE, "i4h:instrument"])
def test_reference_columns_remap_through_add_builder(reference):
    # Two genuine particle rows and two i4h instrument rows in the template.
    instrument = Domain("system", "tool", "i4h:instrument", (2,), "particle")
    samples = Domain("system", "evidence", "i4h:evidence", (2,), "i4h:instrument")
    d = Declarations()
    for domain in (instrument, samples):
        d.domain(domain, worlds=np.zeros(2, np.int32), owners=np.arange(2, dtype=np.int32))
    d.add(field(domain=samples, key="target", dtype=wp.int32, shape=(2,), owner=Owner.MODEL,
                references=reference), np.array([0, 1], np.int32))
    d.add(field(domain=samples, key="native_tet", dtype=wp.int32, shape=(2,), owner=Owner.MODEL), np.full(2, -1))
    source = newton.ModelBuilder()
    d.register(source)
    source.begin_world()
    source.add_particles([(0, 0, 0)] * 2, [(0, 0, 0)] * 2, [1.] * 2)
    source.end_world()
    dest = newton.ModelBuilder()
    for _ in range(2):
        dest.begin_world()
        dest.add_builder(source)
        dest.end_world()
    m = dest.finalize(device="cpu")
    np.testing.assert_array_equal(m.i4h.evidence_target.numpy(), [0, 1, 2, 3])
    np.testing.assert_array_equal(m.i4h.evidence_owner.numpy(), [0, 1, 2, 3])
    np.testing.assert_array_equal(m.i4h.instrument_owner.numpy(), [0, 1, 2, 3])
    np.testing.assert_array_equal(m.i4h.evidence_world.numpy(), [0, 0, 1, 1])
    np.testing.assert_array_equal(m.i4h.evidence_native_tet.numpy(), -1)
    assert m.tet_count == 0


def test_native_particle_values_authored_with_particles_and_foreign_offsets():
    domain = Domain("scene", "particle", newton.Model.AttributeFrequency.PARTICLE, (2, 3))
    d = declared(domain)
    values = np.arange(6, dtype=np.float32).reshape(2, 3) + .1234567
    d.add(field(domain=domain, key="inv_mass", owner=Owner.MODEL), values)
    d.add(field(domain=domain, key="flags", dtype=wp.int32), np.ones((2, 3), np.int32))
    b = newton.ModelBuilder()
    d.register(b)
    assert not any(f.startswith("i4h:") for f in b.custom_frequencies)
    assert not b.custom_attributes["i4h:particle_inv_mass"].values
    for world in range(2):
        b.begin_world()
        b.add_particle((8, 9, 10), (0, 0, 0), 1.)
        b.add_particles([(0, 0, 0)] * 3, [(0, 0, 0)] * 3, [1.] * 3,
                        custom_attributes=d.entity_values(domain, world * 3, 3))
        b.add_particle((8, 9, 10), (0, 0, 0), 1.)
        b.end_world()
    m = b.finalize(device="cpu")
    s1, s2, c = m.state(), m.state(), m.control()
    with pytest.raises(ValueError, match="placement"):
        d.bind(m, s1, c)
    v = d.bind(m, s1, c, placements={domain: (1, 5)})
    assert v.attr(domain, "inv_mass").numpy().tobytes() == values.tobytes()
    v.attr(domain, "flags").fill_(7)
    np.testing.assert_array_equal(s1.i4h.particle_flags.numpy(), [0, 7, 7, 7, 0, 0, 7, 7, 7, 0])
    np.testing.assert_array_equal(s2.i4h.particle_flags.numpy(), [0, 1, 1, 1, 0, 0, 1, 1, 1, 0])


def test_builtin_world_and_empty_native_domain():
    world = Domain("system", "control", newton.Model.AttributeFrequency.WORLD, (2,))
    particle = Domain("scene", "particle", newton.Model.AttributeFrequency.PARTICLE, (2, 0))
    d = declared(world, particle)
    d.add(field(domain=world, key="rate", owner=Owner.MODEL, shape=(2,)), np.array([2., 5.]))
    d.add(field(domain=particle, key="q_prev", dtype=wp.vec3, shape=(2, 0)), np.empty((2, 0, 3)))
    b = newton.ModelBuilder()
    d.register(b)
    for _ in range(2):
        b.begin_world()
        b.end_world()
    m = b.finalize(device="cpu")
    v = d.bind(m, m.state(), m.control())
    np.testing.assert_array_equal(v.attr(world, "rate").numpy(), [2., 5.])
    assert v.attr(particle, "q_prev").shape == (2, 0)
    assert m.particle_count == 0 and m.i4h.particle_q_prev.shape == (1,)
    assert v.allocated_bytes() == dict(model=8, templates=12, state=12, control=0, control_shared_with_template=0)


def test_per_row_scalar_columns_keep_original_layout_and_lifetime():
    import gc
    d = declared(FIRST)
    values = np.arange(36, dtype=np.int32).reshape(2, 3, 6)
    d.add(field(key="tet_edges", dtype=wp.int32, owner=Owner.MODEL, shape=(2, 3, 6), row_width=6), values)
    d.add(field(key="active", dtype=wp.int32), np.ones((2, 3), np.int32))
    b = newton.ModelBuilder()
    d.register(b)
    m = b.finalize(device="cpu")
    view = d.bind(m, m.state(), m.control()).attr(FIRST, "tet_edges")
    assert m.i4h.first_tet_edges.shape == (6,)
    assert view.ptr == m.i4h.first_tet_edges.ptr
    del m, b
    gc.collect()
    np.testing.assert_array_equal(view.numpy(), values)


def test_unknown_reference_rejected_before_builder_mutation():
    d = declared(FIRST)
    d.add(field(references="i4h:missing", dtype=wp.int32), np.zeros((2, 3), np.int32))
    b = newton.ModelBuilder()
    before = dict(b.custom_attributes)
    with pytest.raises(ValueError, match="unknown Newton reference"):
        d.register(b)
    assert b.custom_attributes == before


def test_cached_view_layout_and_domain_metadata_replacement_rejected():
    d, _, m = authored()
    v = d.bind(m, m.state(), m.control())
    v.attr(FIRST, "history").shape = (6,)
    with pytest.raises(RuntimeError, match="layout"):
        v.check()
    v = d.bind(m, m.state(), m.control())
    m.i4h.first_rows_owner = wp.clone(m.i4h.first_rows_owner)
    with pytest.raises(RuntimeError, match="replaced"):
        v.check()


def test_names_row_counts_and_native_tet_values_are_validated():
    d = declared(FIRST)
    with pytest.raises(ValueError, match="row count"):
        field(shape=(3, 3))
    with pytest.raises(ValueError, match="sentinel"):
        d.add(field(key="native_tet", dtype=wp.int32), np.zeros((2, 3), np.int32))
    with pytest.raises(ValueError, match="particle_.*pool_"):
        Domain("scene", "arbitrary", newton.Model.AttributeFrequency.WORLD, (2,))
    with pytest.raises(ValueError, match="own frequency"):
        d.domain(Domain("system", "other", FIRST.frequency, FIRST.shape, "particle"),
                 worlds=np.zeros((2, 3), np.int32), owners=np.zeros((2, 3), np.int32))


def test_deferred_custom_values_resolve_after_host_entities_before_any_rows():
    d = declared(FIRST)
    good = field(key="body", dtype=wp.int32, owner=Owner.MODEL, references=newton.Model.AttributeFrequency.BODY)
    d.add(good, lambda builder: np.full((2, 3), builder.body_count - 1, np.int32))
    invalid = {"shape": (1,)}
    d.add(field(key="late"), lambda builder: np.zeros(invalid["shape"], np.float32))
    b = newton.ModelBuilder()
    d.register_schema(b)
    b.add_body()
    b.add_body()
    before = dict(b._custom_frequency_counts)
    with pytest.raises(ValueError, match="authored values have shape"):
        d.author_rows(b)
    assert dict(b._custom_frequency_counts) == before  # no partially emitted row
    invalid["shape"] = (2, 3)
    d.author_rows(b)
    m = b.finalize(device="cpu")
    v = d.bind(m, m.state(), m.control())
    np.testing.assert_array_equal(v.attr(FIRST, "body").numpy(), 1)
    np.testing.assert_array_equal(v.attr(FIRST, "late").numpy(), 0)
    native = Domain("scene", "particle", newton.Model.AttributeFrequency.PARTICLE, (1,))
    other = Declarations()
    other.domain(native)
    with pytest.raises(ValueError, match="custom field"):
        other.add(field(domain=native, shape=(1,)), lambda builder: np.zeros(1))


def test_package_record_registration_is_atomic_and_frozen():
    d = Declarations()
    scene = SimpleNamespace()
    record = {'domain': FIRST}
    d.register_package(records=((scene, 'tet_domains', {'tet': record}),))
    retained = scene.tet_domains
    d.register_package(records=((scene, 'tet_domains', {'another': record}),))
    assert scene.tet_domains is retained and retained == {'tet': record, 'another': record}
    with pytest.raises(ValueError, match='duplicate package record'):
        d.register_package(records=((scene, 'surface_domains', {'new': record}),
                                    (scene, 'tet_domains', {'tet': record})))
    assert not hasattr(scene, 'surface_domains')
    d.register_schema(newton.ModelBuilder())
    with pytest.raises(RuntimeError, match='package records are frozen'):
        d.register_package(records=((scene, 'surface_domains', {'new': record}),))


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_package_registration_identities_checks_and_explicit_routing(device):
    d, _, m = authored(device)
    v = d.bind(m, m.state(), m.control())
    original = v.attr(FIRST, 'history')
    target = SimpleNamespace(field=original, nested=SimpleNamespace())
    calls = []
    v.register_package(identities=((target, 'field', original), (target, 'nested', target.nested)),
                       checks=(lambda: calls.append('checked'),))
    v.register_package(identities=((target, 'field', original),))
    v.check()
    assert calls == ['checked']
    target.field = v.attr(SECOND, 'history')
    with pytest.raises(RuntimeError, match='replaced'):
        v.register_package(identities=((target, 'field', original),), replace=True)
    target.field = original
    replacement = v.attr(SECOND, 'history')
    v.register_package(identities=((target, 'field', replacement),), replace=True)
    v.check()
    assert target.field is replacement
    before = target.nested
    with pytest.raises(ValueError, match='conflicting package identity'):
        v.register_package(identities=((target, 'nested', object()), (target, 'nested', object())), replace=True)
    assert target.nested is before
    with pytest.raises(TypeError, match='callable'):
        v.register_package(checks=(0,))
    target.nested = object()
    with pytest.raises(RuntimeError, match='nested.*replaced'):
        v.check()


def test_package_registration_checks_in_place_array_layout_changes():
    d, _, m = authored()
    v = d.bind(m, m.state(), m.control())
    extra = SimpleNamespace(values=wp.zeros(3, device='cpu'))
    v.register_package(identities=((extra, 'values', extra.values),))
    extra.values.shape = (1, 3)
    with pytest.raises(RuntimeError, match='array layout'):
        v.check()
