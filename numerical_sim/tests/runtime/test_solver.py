# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I4hSolver behind Newton's SolverBase and the Stepper around it: the scene's main block is the Newton model's and
state's arrays, the solver accepts only its bound state and prepared dt, and reset follows the documented flag table."""

import newton
import numpy as np
import pytest
import warp as wp
from helpers import find, push_actions, run
from i4h_isaaclab.composition import load_case
from i4h_physics import I4hSolver
from i4h_physics.scene import ACTIVE_INT
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.surface import Surface


def world_mask(scene, envs):
    m = np.zeros(scene.envs + 1, bool)
    m[list(envs)] = True
    return wp.array(m, dtype=wp.bool, device=scene.device)


def test_main_block_is_the_newton_state():
    sim = load_case("two_worlds")
    scene, model, state = sim.scene, sim.model, sim.state
    assert isinstance(sim.solver, I4hSolver) and isinstance(sim.solver, newton.solvers.SolverBase)
    assert scene.q.ptr == state.particle_q.ptr and scene.qd.ptr == state.particle_qd.ptr
    assert scene.q0.ptr == model.particle_q.ptr and scene.flags.ptr == model.particle_flags.ptr
    assert scene.inv_mass.ptr == model.particle_inv_mass.ptr and scene.radius.ptr == model.particle_radius.ptr
    assert scene.gravity.ptr == model.gravity.ptr and scene.gravity.shape == (scene.envs,)
    assert model.world_count == scene.envs and model.particle_count == scene.particle_count
    starts = model.particle_world_start.numpy()
    assert starts[: scene.envs + 1].tolist() == [e * scene.main.width for e in range(scene.envs + 1)]


def test_newton_model_rows_are_consistent():
    """Per-variant rows, a consistent mass pair, the true maximum radius (no template values left)."""
    sim = load_case("two_worlds")
    scene, model = sim.scene, sim.model
    inv = model.particle_inv_mass.numpy()
    mass = model.particle_mass.numpy()
    np.testing.assert_allclose(mass[inv > 0] * inv[inv > 0], 1.0, rtol=1e-6)
    assert (mass[inv == 0] == 0).all()
    assert model.particle_max_radius == pytest.approx(float(model.particle_radius.numpy().max()))
    block = scene.body("block")
    assert (scene.field("flags", block).numpy()[0] != 0).sum() == 125 and (scene.field("flags", block).numpy()[1] != 0).sum() == 64


def test_cuts_are_visible_in_the_newton_model():
    """Particle flags are model arrays mutated in place: a burn in the scene is a flag change Newton sees."""
    sim = load_case("two_worlds")
    scene, surface = sim.scene, find(sim, Surface)
    control = find(sim, InstrumentControl)
    actions = push_actions(control, scene.envs)
    actions[0, :, 8] = 1.0  # energy on the pushing probe of env 0
    run(sim, 40, actions)
    assert surface.deleted.numpy()[0] > 0
    assert np.array_equal(sim.model.particle_flags.numpy(), scene.flags.numpy())


def test_solver_accepts_only_its_bound_state_and_dt():
    sim = load_case("two_worlds")
    sim.prepare()
    other = sim.model.state()
    with pytest.raises(ValueError, match="bound state"):
        sim.solver.step(other, other, sim.control, None, sim.dt)
    with pytest.raises(ValueError, match="single-state"):
        sim.solver.step(sim.state, other, sim.control, None, sim.dt)
    with pytest.raises(ValueError, match="differs from the prepared dt"):
        sim.solver.step(sim.state, sim.state, sim.control, None, 2.0 * sim.dt)
    with pytest.raises(ValueError, match="bound state"):
        sim.solver.reset(other)
    sim.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)  # a host's rigid shapes: not i4h's concern
    sim.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)  # gravity is read live


def test_structural_edit_needs_prepare():
    sim = load_case("two_worlds")
    sim.step()
    sim.dt = 1.0 / 120.0
    with pytest.raises(RuntimeError, match="prepare"):
        sim.step()
    with pytest.raises(RuntimeError, match="prepare"):
        sim.reset()
    sim.prepare()  # the runtime Hz change: the solver takes the new dt with the recapture
    sim.step()
    assert sim.solver.dt == pytest.approx(1.0 / 120.0)


def disturbed(case="two_worlds"):
    """Both envs pushed and burnt a little, so every reset field has something to restore."""
    sim = load_case(case)
    scene = sim.scene
    control = find(sim, InstrumentControl)
    actions = push_actions(control, scene.envs)
    actions[:, :, :3] = actions[0, :, :3]  # push both envs
    actions[:, :, 8] = 1.0
    run(sim, 40, actions)
    wp.synchronize()
    return sim


@pytest.mark.parametrize("flags", [None, 0, int(newton.StateFlags.PARTICLE_Q), int(newton.StateFlags.PARTICLE_QD),
                                   int(newton.StateFlags.PARTICLE_F), int(newton.StateFlags.PARTICLE)],
                         ids=["none", "zero", "q", "qd", "f", "q_qd"])
def test_reset_flag_table(flags):
    sim = disturbed()
    scene, state, block = sim.scene, sim.state, sim.scene.body("block")
    tet, surface = find(sim, Tet), find(sim, Surface)
    assert surface.deleted.numpy().min() > 0, "the fixture burns both envs"
    state.particle_f.fill_(wp.vec3(1.0, 2.0, 3.0))
    def field(name):
        return scene.field(name, block).numpy().copy()
    q, qd, f = field("q"), field("qd"), state.particle_f.numpy().reshape(scene.envs, -1, 3)[:, block.offset : block.offset + block.count].copy()
    active_before = tet.active.numpy().copy()
    sim.solver.reset(state, world_mask(scene, [1]), flags)
    wp.synchronize()
    fl = int(newton.StateFlags.ALL) if flags is None else flags
    q1, qd1, qp1 = field("q"), field("qd"), field("q_prev")
    f1 = state.particle_f.numpy().reshape(scene.envs, -1, 3)[:, block.offset : block.offset + block.count]
    q0 = field("q0")
    # env 0 unchanged, whatever the flags
    np.testing.assert_array_equal(q1[0], q[0])
    np.testing.assert_array_equal(qd1[0], qd[0])
    np.testing.assert_array_equal(tet.active.numpy()[0], active_before[0])
    # env 1 follows the table
    np.testing.assert_array_equal(q1[1], q0[1] if fl & newton.StateFlags.PARTICLE_Q else q[1])
    np.testing.assert_array_equal(qd1[1], 0.0 if fl & newton.StateFlags.PARTICLE_QD else qd[1])
    np.testing.assert_array_equal(f1[1], 0.0 if fl & newton.StateFlags.PARTICLE_F else f[1])
    np.testing.assert_array_equal(qp1[1], q1[1])  # q_prev follows q in every mode
    # topology and flags come back together in every mode
    assert tet.active.numpy()[1].sum() > active_before[1].sum() and surface.deleted.numpy()[1] == 0
    np.testing.assert_array_equal(field("flags")[1], field("flags0")[1])


def test_repeated_and_full_resets():
    sim = disturbed()
    scene, block = sim.scene, sim.scene.body("block")
    for _ in range(3):
        sim.reset()
    wp.synchronize()
    np.testing.assert_array_equal(scene.field("q", block).numpy(), scene.field("q0", block).numpy())
    assert (scene.field("qd", block).numpy() == 0).all() and find(sim, Surface).deleted.numpy().tolist() == [0, 0]
    before = scene.q.numpy().copy()
    sim.reset([])  # nothing selected
    np.testing.assert_array_equal(scene.q.numpy(), before)


def test_flags_restore_matches_topology_after_cut():
    """A burnt particle marked inactive comes back ACTIVE with its tets even under flags=0."""
    sim = disturbed()
    scene, block = sim.scene, sim.scene.body("block")
    flags = scene.field("flags", block).numpy()
    dead = (scene.field("flags0", block).numpy()[0] & ACTIVE_INT) & ~(flags[0] & ACTIVE_INT)
    sim.solver.reset(sim.state, world_mask(scene, [0]), 0)
    wp.synchronize()
    assert ((scene.field("flags", block).numpy()[0] & ACTIVE_INT)[dead != 0] != 0).all()


def test_cpu_steps():
    sim = load_case("two_worlds", device="cpu")
    for _ in range(5):
        sim.step()
    assert np.isfinite(sim.scene.q.numpy()).all() and sim.graph is None


def test_per_world_builder_matches_replicate():
    """add_env_particles inside each world (Isaac Lab's per-world hook form) lays the main block out exactly like the
    replicate fast path, and an owner's finalized model plus its early state end with every env's own rows."""
    import newton
    from i4h_isaaclab.composition import load_scene, read_case
    from i4h_physics import Scene
    from i4h_physics.newton_model import add_env_particles, build_builder

    scene = load_scene(read_case("two_worlds"), envs=3)  # mesh variants 0, 1, 0
    per_world = newton.ModelBuilder()
    for e in range(scene.envs):
        per_world.begin_world()
        add_env_particles(per_world, scene, e)
        per_world.end_world()
    a = per_world.finalize(device=scene.device)
    b = build_builder(scene).finalize(device=scene.device)
    assert a.world_count == b.world_count == 3 and a.particle_count == b.particle_count
    np.testing.assert_array_equal(a.particle_world.numpy(), b.particle_world.numpy())
    state = b.state()  # made before the rows are written, as NewtonManager does
    scene.allocate(b, state)
    np.testing.assert_array_equal(state.particle_q.numpy(), a.particle_q.numpy())  # the per-world path had exact rows
    np.testing.assert_array_equal(b.particle_flags.numpy(), a.particle_flags.numpy())
    np.testing.assert_allclose(b.particle_inv_mass.numpy(), a.particle_inv_mass.numpy(), rtol=1e-6)
    assert scene.q.ptr == state.particle_q.ptr and isinstance(scene, Scene)
