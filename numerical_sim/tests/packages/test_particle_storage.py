# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C2.1 native ownership, exact defaults, and binding without history writes."""
import newton
import numpy as np
import pytest
import warp as wp
from i4h_physics import author_case
from i4h_physics.newton_model import main_rows
from i4h_physics.storage import Declarations, Domain, Field, Owner


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_native_ownership_exact_bytes_and_selected_reset(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    cfg = {"sim": {"device": device, "envs": 2},
           "assets": {"rod": {"type": "rod", "count": 3, "segment_length": .02,
                               "radius": .002, "start": [0, 0, 0], "direction": [1, 0, 0]}},
           "systems": [{"type": "particles"}]}
    a = author_case(cfg)
    rest = a.scene.body("rod").rest
    # This value changes under float32 reciprocal round trips.
    value = np.float32(.1234567)
    assert np.float32(1) / (np.float32(1) / value) != value
    rest.inv_mass[:] = value
    b = newton.ModelBuilder()
    b.add_particle((7, 8, 9), (1, 2, 3), 2.)
    for e in range(2):
        b.begin_world()
        b.add_particle((7, 8, 9), (1, 2, 3), 2.)
        a.add_world(b, e)
        b.add_particle((7, 8, 9), (1, 2, 3), 2.)
        b.end_world()
    a.complete(b)
    m = b.finalize(device=device)
    s1, s2, c = m.state(), m.state(), m.control()
    s1.i4h.particle_q_prev.fill_(wp.vec3(13.))
    s1.i4h.particle_flags.fill_(17)
    protected = [m.particle_q, m.particle_inv_mass, m.particle_flags,
                 s1.particle_q, s1.particle_qd, s1.i4h.particle_q_prev, s1.i4h.particle_flags]
    before = [x.numpy().tobytes() for x in protected]
    scene, _ = a.bind(m, s1, c)
    assert [x.numpy().tobytes() for x in protected] == before
    assert not hasattr(m.i4h, "particle_q0")
    assert scene.q0 is m.particle_q
    assert scene.attr(a.particle_domain, "q0").numpy().tobytes() == main_rows(a.scene, "q0").tobytes()
    assert scene.inv_mass.ptr == m.i4h.particle_inv_mass.ptr
    assert scene.q_prev.ptr == s1.i4h.particle_q_prev.ptr
    assert scene.flags.ptr == s1.i4h.particle_flags.ptr
    assert scene.block_field("inv_mass").numpy().tobytes() == main_rows(a.scene, "inv_mass").tobytes()
    other, _ = a.bind(m, s2, m.control())
    assert other.q0.ptr == scene.q0.ptr and other.flags.ptr != scene.flags.ptr
    np.testing.assert_array_equal(other.block_field("q_prev").numpy(), main_rows(a.scene, "q0"))
    np.testing.assert_array_equal(other.block_field("flags").numpy(), main_rows(a.scene, "flags"))
    mask = wp.array([True, False], dtype=wp.bool, device=device)
    scene.reset(mask)
    first = np.arange(*a.worlds[0][:2])
    untouched = np.setdiff1d(np.arange(m.particle_count), first)
    np.testing.assert_array_equal(scene.flags.numpy()[untouched], 17)
    np.testing.assert_array_equal(scene.q_prev.numpy()[untouched], 13.)
    for flags in ([False, False], [True, False]):
        snapshot = [x.numpy().tobytes() for x in protected]
        scene.reset(wp.array(flags, dtype=wp.bool, device=device))
        assert [x.numpy().tobytes() for x in protected] == snapshot
    np.testing.assert_array_equal(m.particle_flags.numpy().tobytes(), before[2])
    # Explicit host teleports update only selected history, even across foreign
    # prefixes and strides. Binding and cache-only refresh are not teleports.
    s1.particle_q.fill_(wp.vec3(23.))
    history = scene.q_prev.numpy().copy()
    protected_edits = [x.numpy().tobytes() for x in protected if x is not scene.q_prev]
    scene.initialize_particle_history(mask)
    np.testing.assert_array_equal(scene.q_prev.numpy()[first], 23.)
    np.testing.assert_array_equal(scene.q_prev.numpy()[untouched], history[untouched])
    assert [x.numpy().tobytes() for x in protected if x is not scene.q_prev] == protected_edits
    for flags in ([False, False], [True, False]):
        history = scene.q_prev.numpy().tobytes()
        scene.initialize_particle_history(wp.array(flags, dtype=wp.bool, device=device))
        assert scene.q_prev.numpy().tobytes() == history
    # Native rest edits deliberately become the reset source, not a stale copy.
    m.particle_q.fill_(wp.vec3(31.))
    scene.reset(mask)
    np.testing.assert_array_equal(scene.q.numpy()[first], 31.)
    m.particle_q = wp.clone(m.particle_q)
    with pytest.raises(RuntimeError, match="replaced"):
        scene.storage.check()
    m.particle_q = scene.q0
    s1.i4h.particle_q_prev = wp.clone(s1.i4h.particle_q_prev)
    with pytest.raises(RuntimeError, match="replaced"):
        scene.storage.check()


def test_schema_freezes_before_native_authoring_and_rows_are_idempotent():
    d = Declarations()
    native = Domain("scene", "particle", newton.Model.AttributeFrequency.PARTICLE, (1, 1))
    custom = Domain("system", "test", "i4h:test_rows", (1,), "particle")
    d.domain(native)
    d.domain(custom, worlds=[0], owners=lambda: [0])
    d.add(Field(native, "flags", Owner.STATE, wp.int32, (1, 1), "world, particle", "authored"), [[9]])
    d.add(Field(custom, "history", Owner.STATE, wp.int32, (1,), "row", "authored"), [13])
    b = newton.ModelBuilder()
    d.register_schema(b)
    with pytest.raises(RuntimeError, match="frozen"):
        d.add(Field(custom, "late", Owner.MODEL, wp.int32, (1,), "row", "authored"), [0])
    b.begin_world()
    b.add_particles([(0, 0, 0)], [(0, 0, 0)], [1.], custom_attributes=d.entity_values(native))
    b.end_world()
    d.author_rows(b)
    counts = dict(b._custom_frequency_counts)
    d.author_rows(b)
    assert counts == b._custom_frequency_counts
    m = b.finalize(device="cpu")
    v = d.bind(m, m.state(), m.control())
    np.testing.assert_array_equal(v.attr(native, "flags").numpy(), [[9]])
    np.testing.assert_array_equal(v.attr(custom, "history").numpy(), [13])


def test_empty_native_estimate_includes_real_padding_and_missing_schema_fails():
    from i4h_physics import Scene

    scene = Scene("cpu", 1)
    scene.allocate()
    measured = scene.storage.allocated_bytes()
    assert scene.model.particle_count == 0
    assert scene.q.size == scene.q_prev.size == scene.flags.size == 0
    assert sum(scene.estimate_bytes().values()) >= measured["model"] + measured["templates"] + measured["state"]
    assert measured["model"] + measured["state"] > 0

    unregistered = Scene("cpu", 1)
    b = newton.ModelBuilder()
    b.begin_world()
    b.add_particle((7, 8, 9), (1, 2, 3), 2.)
    b.end_world()
    m = b.finalize(device="cpu")
    state = m.state()
    before = state.particle_q.numpy().tobytes()
    with pytest.raises(RuntimeError, match="authored storage declarations"):
        unregistered.allocate(m, state)
    assert state.particle_q.numpy().tobytes() == before
    assert unregistered.pool.q is None


def test_changed_capacity_requires_reconstruction_even_after_completion():
    a = author_case({"sim": {"device": "cpu"},
                     "assets": {"rod": {"type": "rod", "count": 3, "segment_length": .02,
                                         "radius": .002, "start": [0, 0, 0], "direction": [1, 0, 0]}}})
    b = newton.ModelBuilder()
    b.begin_world()
    a.add_world(b, 0)
    b.end_world()
    a.complete(b)
    a.cfg["assets"]["rod"]["count"] = 4
    with pytest.raises(RuntimeError, match="capacities changed; reconstruct"):
        a.complete(b)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_native_rest_variants_padding_and_pool_history(device):
    from i4h_physics import Scene
    from i4h_physics.scene import ParticleRest

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene = Scene(device, 3)

    def rest(count):
        q = (np.arange(3 * count * 3, dtype=np.float32).reshape(3, count, 3) * np.float32(.1234567))
        return ParticleRest(q, np.ones((3, count), np.float32), np.full((3, count), .02, np.float32),
                            np.ones((3, count), np.int32))

    main = rest(4)
    main.flags[1, 2:] = 0  # inactive capacity padding retains exact rest bytes
    scene.add_body("main", main, variant_of_env=np.array([0, 1, 0], np.int32), variants=2)
    for name, count in (("empty", 0), ("small", 2), ("large", 5)):
        scene.add_body(name, rest(count), pool=True)
    scene.allocate()
    assert scene.block_field("q0").numpy().tobytes() == main.q0.tobytes()
    assert scene.attr(scene.particle_domain, "q0").numpy().tobytes() == main.q0.tobytes()
    assert scene.q0 is scene.model.particle_q
    assert not hasattr(scene.model.i4h, "particle_q0")
    mask = wp.array([False, True, False], dtype=wp.bool, device=device)
    scene.q.fill_(wp.vec3(23.))
    scene.pool.q.fill_(wp.vec3(29.))
    scene.q_prev.fill_(wp.vec3(13.))
    scene.pool.q_prev.fill_(wp.vec3(17.))
    scene.initialize_particle_history(mask)
    for body in scene.bodies + scene.pools:
        if not body.count:
            continue
        previous = scene.field("q_prev", body).numpy()
        np.testing.assert_array_equal(previous[1], 29. if body.pool else 23.)
        np.testing.assert_array_equal(previous[[0, 2]], 17. if body.pool else 13.)
