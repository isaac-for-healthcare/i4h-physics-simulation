# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F1: i4h particles inside a host Newton model that holds other particles, and env frames (docs/component-contracts.md
sections 1-2). CPU, generated assets."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
from i4h_physics.config import build_systems, load_case, load_scene, sim_settings  # noqa: E402
from i4h_physics.newton_model import add_env_particles  # noqa: E402
from i4h_physics.quat import axis_angle_quat, quat_to_matrix  # noqa: E402
from i4h_physics.sim import I4hSolver  # noqa: E402
from i4h_physics.stepper import Stepper  # noqa: E402

ROD = {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0], "count": 5, "segment_length": .02, "radius": .002}
CASE = {"sim": {"device": "cpu", "envs": 2, "dt": .01, "substeps": 2}, "assets": {"rod": ROD},
        "systems": [{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]}  # no sim.gravity: the host's
HOST_BEFORE, HOST_AFTER = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0)], [(7.0, 5.0, 5.0)]


def host_sim(rotations, gravity=(0.0, 0.0, -9.81), case_gravity=None):
    """A host model: one global particle in front, then per world two host particles, ours, one more host particle;
    world ``e`` placed at ``(e, 0, 0)`` with orientation ``rotations[e]`` (xyzw)."""
    case = CASE if case_gravity is None else CASE | {"sim": CASE["sim"] | {"gravity": list(case_gravity)}}
    scene = load_scene(case)
    b = newton.ModelBuilder(gravity=gravity[2])
    b.add_particle((9.0, 9.0, 9.0), (0.0, 0.0, 0.0), 1.0)  # a global particle
    for e in range(scene.envs):
        b.begin_world()
        b.add_particles(HOST_BEFORE, [(0.0, 0.0, 0.0)] * 2, [1.0] * 2)
        add_env_particles(b, scene, e, xform=(np.float32([e, 0.0, 0.0]), np.float32(rotations[e])))
        b.add_particles(HOST_AFTER, [(0.0, 0.0, 0.0)], [1.0])
        b.end_world()
    model = b.finalize(device="cpu")
    model.set_gravity(gravity)
    state = model.state()
    scene.allocate(model, state)
    sim = Stepper(I4hSolver(model, scene, build_systems(scene, case), state), **sim_settings(case))
    return sim, model, state


def ours(sim):
    return sim.scene.field("q", sim.scene.body("rod")).numpy().copy()


def test_rows_found_among_host_particles_and_host_rows_untouched():
    identity = [(0.0, 0.0, 0.0, 1.0)] * 2
    sim, model, state = host_sim(identity)
    assert sim.scene.main.placement == (1, 8, 2), sim.scene.main.placement  # 1 global; worlds of 2 + 5 + 1
    before = state.particle_q.numpy().copy()
    alone = load_case(CASE | {"sim": CASE["sim"] | {"gravity": [0.0, 0.0, -9.81]}})
    for _ in range(10):
        sim.step()
        alone.step()
    np.testing.assert_allclose(ours(sim), ours(alone), atol=1e-6)  # our rod advances as it does alone
    q = state.particle_q.numpy()
    host = np.r_[0, [1 + e * 8 + i for e in range(2) for i in (0, 1, 7)]]
    np.testing.assert_array_equal(q[host], before[host])  # nobody integrates host particles here
    sim.reset([0])
    np.testing.assert_allclose(ours(sim)[0], ours(load_case(CASE | {"sim": CASE["sim"] | {"gravity": [0, 0, -9.81]}}))[0])
    np.testing.assert_array_equal(state.particle_q.numpy()[host], before[host])


def test_rotated_env_sees_gravity_in_the_component_frame():
    """World 1 is turned 90 degrees about x: in the component frame its gravity is R^T g, so its rod sags along that
    direction, exactly like a standalone rod under R^T g; world 0 matches the standalone rod under g."""
    q1 = axis_angle_quat((1.0, 0.0, 0.0), np.pi / 2)
    sim, model, state = host_sim([(0.0, 0.0, 0.0, 1.0), tuple(q1)])
    g = np.float32([0.0, 0.0, -9.81])
    g1 = quat_to_matrix(q1).T @ g
    np.testing.assert_allclose(sim.scene.gravity.numpy(), [g, g1], atol=1e-5)
    refs = [load_case(CASE | {"sim": CASE["sim"] | {"gravity": list(map(float, v))}}) for v in (g, g1)]
    for _ in range(10):
        sim.step()
        for r in refs:
            r.step()
    for e in range(2):
        np.testing.assert_allclose(ours(sim)[e], ours(refs[e])[e], atol=1e-6)


def test_misplaced_rows_are_rejected():
    scene = load_scene(CASE)
    b = newton.ModelBuilder()
    for e in range(scene.envs):
        b.begin_world()
        if e == 1:
            b.add_particles(HOST_BEFORE, [(0.0, 0.0, 0.0)] * 2, [1.0] * 2)  # only world 1 has host particles
        add_env_particles(b, scene, e, xform=(np.zeros(3), np.float32([0, 0, 0, 1])))
        b.end_world()
    model = b.finalize(device="cpu")
    with pytest.raises(RuntimeError, match="one column of equally sized worlds"):
        scene.allocate(model, model.state())


def test_host_gravity_changes_reach_a_rotated_component():
    """A rotated component holds R^T g in its own array: MODEL_PROPERTIES refreshes it in place (captured graphs keep
    reading the same buffer)."""
    from newton import ModelFlags

    q1 = axis_angle_quat((1.0, 0.0, 0.0), np.pi / 2)
    sim, model, state = host_sim([(0.0, 0.0, 0.0, 1.0), tuple(q1)])
    buffer = sim.scene.gravity.ptr
    for g in ((0.0, 0.0, 0.0), (0.0, -3.0, 0.0)):
        model.set_gravity(g)
        sim.solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)
        g = np.float32(g)
        np.testing.assert_allclose(sim.scene.gravity.numpy(), [g, quat_to_matrix(q1).T @ g], atol=1e-6)
        assert sim.scene.gravity.ptr == buffer



def test_declared_case_gravity_is_the_world_gravity_of_a_rotated_component():
    """A case's sim.gravity declares the world's gravity (I4hComponent.configure copies it to the host): a rotated
    component whose case declares the host's gravity builds, and still sees R^T g in its own frame; a different
    declaration is rejected."""
    q1 = axis_angle_quat((1.0, 0.0, 0.0), np.pi / 2)
    sim, model, state = host_sim([(0.0, 0.0, 0.0, 1.0), tuple(q1)], case_gravity=(0.0, 0.0, -9.81))
    g = np.float32([0.0, 0.0, -9.81])
    np.testing.assert_allclose(sim.scene.gravity.numpy(), [g, quat_to_matrix(q1).T @ g], atol=1e-5)
    with pytest.raises(ValueError, match="differs from the case's sim.gravity"):
        host_sim([(0.0, 0.0, 0.0, 1.0), tuple(q1)], case_gravity=(0.0, -9.81, 0.0))
