# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C2.2 pool and Scene metadata ownership and binding boundaries."""
import newton
import numpy as np
import pytest
import warp as wp
from i4h_physics import Scene, Stepper, author_case
from i4h_physics.newton_model import add_env_particles, complete_scene
from i4h_physics.scene import Block, ParticleRest, StaticMesh
from i4h_physics.systems.fluids.fluid import Fluid


def rest(envs, count):
    q = np.arange(envs * count * 3, dtype=np.float32).reshape(envs, count, 3) * np.float32(.1234567)
    return ParticleRest(q, np.full((envs, count), .1234567, np.float32),
                        np.full((envs, count), .003, np.float32), np.zeros((envs, count), np.int32))


def authored(device):
    a = author_case({"sim": {"device": device, "envs": 3},
        "assets": {"rod": {"type": "rod", "count": 3, "segment_length": .02, "radius": .002,
                           "start": [0, 0, 0], "direction": [1, 0, 0]},
                   "water": {"type": "fluid", "count": 2}, "blood": {"type": "fluid", "count": 5}},
        "systems": [{"type": "particles"}, {"type": "fluid", "id": "first"}, {"type": "fluid", "id": "second"}]})
    scene = a.scene
    gap = scene.add_body("non-fluid gap", rest(3, 4), pool=True, variants=2,
                         variant_of_env=np.array([1, 0, 1], np.int32))
    empty = scene.add_body("empty", rest(3, 0), pool=True)
    scene.pool.bodies[:] = [scene.body("water"), gap, empty, scene.body("blood")]
    offset = 0
    for body in scene.pools:
        body.offset = offset
        offset += scene.envs * body.count
        body.rest = rest(3, body.count)
    scene.static_meshes.append(StaticMesh("floor", np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                                         np.int32([[0, 1, 2]]), None, np.zeros((3, 3), np.float32), None))
    return a


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("shared_control", [False, True])
def test_pool_bytes_independent_states_views_and_full_selected_reset(device, shared_control):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    a = authored(device)
    b = newton.ModelBuilder()
    b.add_particle((9, 8, 7), (1, 2, 3), 1.)
    for world in range(3):
        b.begin_world()
        b.add_particle((9, 8, 7), (1, 2, 3), 1.)
        a.add_world(b, world)
        b.add_particle((9, 8, 7), (1, 2, 3), 1.)
        b.end_world()
    a.complete(b)
    counts = dict(b._custom_frequency_counts)
    a.complete(b)
    assert counts == b._custom_frequency_counts
    m = b.finalize(device=device)
    m.particle_grid = None
    s1, s2 = m.state(), m.state()
    c = m.control(clone_variables=not shared_control)
    # Distinct restored position/history/velocity/flags must survive every bind.
    for name, value in (("q", 13.), ("qd", 17.), ("q_prev", 19.)):
        getattr(s1.i4h, f"pool_{name}").fill_(wp.vec3(value))
    s1.i4h.pool_flags.fill_(23)
    native = [m.particle_q, s1.particle_q, s1.particle_qd, s1.i4h.particle_q_prev]
    before = [v.numpy().tobytes() for v in native]
    restored = [getattr(s1.i4h, f"pool_{key}").numpy().tobytes() for key in ("q", "qd", "q_prev", "flags")]
    scene, systems = a.bind(m, s1, c)
    other, other_systems = a.bind(m, s2, c if shared_control else m.control())
    assert restored == [getattr(scene.pool, key).numpy().tobytes() for key in ("q", "qd", "q_prev", "flags")]
    assert m.particle_count == 1 + 3 * (2 + 3) and m.tet_count == m.tri_count == 0
    for key in Block.FIELDS:
        expected_owner = m if key in ("q0", "inv_mass", "radius", "flags0") else s1
        assert getattr(scene.pool, key).ptr == getattr(expected_owner.i4h, f"pool_{key}").ptr
        assert scene.attr(scene.pool_domain, key) is getattr(scene.pool, key)
    for body in scene.pools:
        for key, source in (("q0", "q0"), ("inv_mass", "inv_mass"), ("radius", "radius"), ("flags0", "flags")):
            assert scene.field(key, body).numpy().tobytes() == getattr(body.rest, source).tobytes()
        assert body.variant.ptr == getattr(m.i4h, f"{scene.body_domains[body.name].instance}_variant").ptr
        assert scene.field("q", body) is scene.field("q", body)
        assert body.variant.ptr == other.body(body.name).variant.ptr
    worlds = np.concatenate([np.repeat(np.arange(3), body.count) for body in scene.pools])
    np.testing.assert_array_equal(m.i4h.pool_rows_world.numpy(), worlds)
    np.testing.assert_array_equal(m.i4h.pool_rows_owner.numpy(), -1)
    fluids = [s for s in systems if isinstance(s, Fluid)]
    for fluid in fluids:
        assert isinstance(fluid, Fluid)
        assert fluid.q_span.ptr == scene.pool.q.ptr
        for body, view in zip(fluid.bodies, fluid.q_b):
            assert view.ptr == scene.field("q", body).ptr
    assert fluids[0].bodies[0].variant is fluids[1].bodies[0].variant
    assert scene.pool.q0.ptr == other.pool.q0.ptr and scene.pool.q.ptr != other.pool.q.ptr
    other_before = other.pool.q.numpy().tobytes()
    # Pool reset remains full when main q and qd resets are disabled.
    for selected in ([False, True, False], [False, False, False], [False, True, False]):
        scene.reset(wp.array(selected, dtype=wp.bool, device=device), positions=False, velocities=False)
    for body in scene.pools:
        if not body.count:
            continue
        for key, expected in (("q", body.rest.q0[1]), ("q_prev", body.rest.q0[1]),
                              ("qd", np.zeros((body.count, 3))), ("flags", body.rest.flags[1])):
            values = scene.field(key, body).numpy()
            np.testing.assert_array_equal(values[1], expected)
            np.testing.assert_array_equal(values[[0, 2]], {"q": 13., "qd": 17., "q_prev": 19., "flags": 23}[key])
    assert [v.numpy().tobytes() for v in native] == before
    assert other.pool.q.numpy().tobytes() == other_before
    foreign = np.r_[0, [1 + e * 5 + i for e in range(3) for i in (0, 4)]]
    protected_foreign = [s2.particle_q, s2.particle_qd, s2.i4h.particle_q_prev, s2.i4h.particle_flags]
    before_foreign = [v.numpy()[foreign].tobytes() for v in protected_foreign]
    # One fluid solver here; repeated types above exercise shared binding only.
    sim = Stepper.create(other, other_systems[:2], cuda_graph=device.startswith("cuda"))
    sim.step()
    sim.reset([1])
    assert [v.numpy()[foreign].tobytes() for v in protected_foreign] == before_foreign
    scene.pool.q0.fill_(wp.vec3(31.))
    scene.pool.inv_mass.fill_(.25)
    scene.pool.radius.fill_(.007)
    scene.body("non-fluid gap").variant.fill_(0)
    scene.reset(wp.array([False, True, False], dtype=wp.bool, device=device))
    for body in scene.pools:
        if body.count:
            np.testing.assert_array_equal(scene.field("q", body).numpy()[1], 31.)
    np.testing.assert_array_equal(scene.pool.inv_mass.numpy(), .25)
    np.testing.assert_array_equal(scene.pool.radius.numpy(), np.float32(.007))
    np.testing.assert_array_equal(scene.body("non-fluid gap").variant.numpy(), 0)
    # Geometry is Model-owned, local indices and shared acceleration structure.
    mesh = scene.static_wp_mesh("floor")
    assert scene.static_wp_mesh("floor") is mesh
    points, indices = scene.static_domains["floor"]
    assert mesh.points.ptr == scene.attr(points, "points").ptr
    assert mesh.indices.ptr == scene.attr(indices, "indices").ptr
    np.testing.assert_array_equal(mesh.indices.numpy(), [0, 1, 2])
    assert other.static_wp_mesh("floor").points.ptr == mesh.points.ptr
    s1.i4h.pool_q = wp.clone(s1.i4h.pool_q)
    with pytest.raises(RuntimeError, match="replaced"):
        scene.reset(wp.array([False] * 3, dtype=wp.bool, device=device))
    s1.i4h.pool_q = wp.zeros(1, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="incompatible persistent buffer"):
        a.bind(m, s1, c)


def test_direct_completion_retained_builder_gravity_and_reconstruction():
    from i4h_physics.quat import axis_angle_quat
    scene = Scene("cpu", 2, gravity=None)
    scene.add_body("pool", rest(2, 2), pool=True)
    b = newton.ModelBuilder()
    def add(e, q):
        add_env_particles(b, scene, e, xform=(np.zeros(3), np.asarray(q)))
    b.begin_world()
    add(0, [0., 0., 0., 1.])
    b.end_world()
    with pytest.raises(RuntimeError, match="all authored worlds"):
        complete_scene(b, scene)
    b.begin_world()
    add(1, axis_angle_quat((1, 0, 0), np.pi / 2))
    b.end_world()
    with pytest.raises(RuntimeError, match="before finalization"):
        b.finalize(device="cpu")
    complete_scene(b, scene)
    complete_scene(b, scene)
    m = b.finalize(device="cpu")
    rebuilt = b.finalize(device="cpu")
    np.testing.assert_array_equal(m.i4h.scene_gravity.numpy(), rebuilt.i4h.scene_gravity.numpy())
    initial = m.i4h.scene_gravity.numpy().copy()
    m.set_gravity((1, 2, 3))
    # Binding preserves a derived buffer even when host edits require refresh.
    scene.allocate(m, m.state())
    np.testing.assert_array_equal(scene.gravity.numpy(), initial)
    ptr = scene.gravity.ptr
    scene.refresh_gravity()
    assert scene.gravity.ptr == ptr
    np.testing.assert_allclose(scene.gravity.numpy(), [[1, 2, 3], [1, 3, -2]], atol=1e-6)
    scene.refresh_gravity(rebuilt)  # Explicit pre-bind refresh on another finalized Model.
    scene.env_xforms[0][0][0] = 1
    with pytest.raises(RuntimeError, match="transforms changed"):
        complete_scene(b, scene)
    scene.env_xforms[0][0][0] = 0
    scene.body("pool").variant_of_env[0] = 1
    with pytest.raises(RuntimeError, match="reconstruct"):
        complete_scene(b, scene)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_empty_pool_padding_and_model_edits(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene = Scene(device, 2)
    scene.add_body("empty", rest(2, 0), pool=True)
    scene.allocate()
    assert scene.gravity.ptr == scene.model.gravity.ptr
    assert scene.pool.q.size == 0 and scene.model.i4h.pool_q0.size == 1
    assert scene.model.particle_count == 0
    assert scene.model.i4h.pool_rows_world.numpy().tolist() == [-1]
    assert scene.storage.allocated_bytes()["templates"] == 56
    scene.model.i4h.body_0_variant = wp.clone(scene.model.i4h.body_0_variant)
    with pytest.raises(RuntimeError, match="replaced"):
        scene.storage.check()


def test_static_geometry_layout_changes_and_undeclared_storage_rejected():
    scene = Scene("cpu", 1)
    mesh = StaticMesh("empty", np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                      None, np.zeros((0, 3), np.float32), None)
    scene.static_meshes.append(mesh)
    with pytest.raises(RuntimeError, match="completed storage"):
        scene.static_wp_mesh("empty")
    from i4h_physics.newton_model import build_builder
    b = build_builder(scene)
    m = b.finalize(device="cpu")
    scene.allocate(m, m.state())
    for domain, key in zip(scene.static_domains["empty"], ("points", "indices")):
        assert scene.attr(domain, key).size == 0
    mesh.indices = np.zeros((1, 3), np.int32)
    with pytest.raises(RuntimeError, match="geometry layout changed"):
        complete_scene(b, scene)
