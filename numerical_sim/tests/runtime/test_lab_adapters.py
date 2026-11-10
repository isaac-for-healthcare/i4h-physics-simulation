# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native allocation, lifetime and contact ownership on both qualified Lab releases."""

from contextlib import nullcontext

import numpy as np
import pytest
import warp as wp

pytest.importorskip("isaaclab.envs")
import torch  # noqa: E402
from helpers import lab_runtime  # noqa: E402
from i4h_isaaclab.env import I4hEnv, I4hEnvCfg  # noqa: E402
from i4h_isaaclab.newton_manager import I4hSolverCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_newton.cloner import newton_builder_world_hook  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg  # noqa: E402
from newton.solvers import SolverBase  # noqa: E402


@pytest.fixture
def make():
    envs = []

    def build(observables=None, captured=True, **kwargs):
        cfg = I4hEnvCfg(stagger_episodes=False, **kwargs)
        cfg.sim.physics.solver_cfg.observables = (SolverBase.ObservableKind.CONTACT_F,) if cfg.rigid_solver_cfg else ()
        if observables is not None:
            cfg.sim.physics.solver_cfg.observables = observables
        cfg.sim.physics.use_cuda_graph = captured
        env = I4hEnv(cfg)
        envs.append(env)
        return env

    yield build
    for env in envs:
        env.close()
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()


def test_adapter_capabilities_are_fixed():
    pure = I4hSolverCfg()
    coupled = I4hSolverCfg(rigid_solver_cfg=MJWarpSolverCfg())
    assert pure.class_type is not coupled.class_type
    assert not pure.class_type.contact_sensors and coupled.class_type.contact_sensors
    assert not I4hSolverCfg().class_type.contact_sensors
    with pytest.raises(ValueError, match="collision pipeline"):
        I4hSolverCfg(rigid_solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False))


@pytest.mark.parametrize("coupled", [False, True])
def test_main_coupling_requires_in_place(coupled):
    pytest.importorskip("isaaclab_newton.physics.newton_solver")
    cfg = I4hSolverCfg(rigid_solver_cfg=MJWarpSolverCfg() if coupled else None)
    assert "in-place" in cfg.class_type.coupling_error(cfg, in_place=False)
    assert cfg.class_type.coupling_error(cfg, in_place=True) is None


def test_main_coupling_preserves_rigid_child_validation():
    pytest.importorskip("isaaclab_newton.physics.newton_solver")
    cfg = I4hSolverCfg(rigid_solver_cfg=MJWarpSolverCfg())
    # Lab's later coupling check must still inspect the child after configuration edits.
    cfg.rigid_solver_cfg.use_mujoco_cpu = True
    reason = cfg.class_type.coupling_error(cfg, in_place=True)
    assert reason is not None and "use_mujoco_cpu" in reason


def test_repeated_hard_rebuild_keeps_entities_and_subscriptions(make):
    env = make()
    owner = env._construction
    authoring = owner.authoring
    particles = env.simulator.model.particle_count
    subscriptions = len(owner.subscriptions)
    manager_callbacks = len(owner.manager._callbacks)
    for _ in range(3):
        old = env.simulator
        env.sim.reset(soft=False)
        assert not old.valid and old is not env.simulator
        assert owner.authoring is authoring and len(authoring.worlds) == 2
        assert len(owner.subscriptions) == subscriptions and len(owner.manager._callbacks) == manager_callbacks
        binding = env.simulator
        native = lab_runtime()
        assert binding.state is native.state_0 and binding.control is native.control
        assert binding.model.particle_count == particles
        assert binding.model.tet_count == binding.model.tri_count == 0
        assert binding.model.particle_grid is None
        assert binding.scene.q0 is binding.model.particle_q
        assert binding.scene.inv_mass.ptr != binding.model.particle_inv_mass.ptr
        assert not hasattr(binding, "step") and not hasattr(binding, "reset_graph")
        env.step(torch.zeros((2, 10), device=env.device))
    owner.close()
    assert not env.simulator.valid and not owner.subscriptions and owner.backend is None


def test_replaced_captured_buffer_is_rejected_before_replay(make):
    env = make()
    env.step(torch.zeros((2, 10), device=env.device))
    state = env.simulator.state
    original = state.particle_q
    before = original.numpy().copy()
    state.particle_q = wp.clone(original)
    try:
        with pytest.raises(RuntimeError, match="buffer was replaced"):
            env.sim.step(render=False)
        np.testing.assert_array_equal(original.numpy(), before)
    finally:
        state.particle_q = original
    env.sim.step(render=False)


def test_pure_rejects_force_contact_request(make):
    def request(builder, world, position, rotation):
        builder.request_contact_attributes("force")
    with newton_builder_world_hook(request), pytest.raises(ValueError, match="force-contact sensors"):
        make()


def rigid_contact_world(builder, world, position, rotation):
    builder.request_contact_attributes("force")
    builder.add_ground_plane()
    body = builder.add_link(xform=wp.transform((float(world), 2.0, 0.04), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    joint = builder.add_joint_free(body)
    builder.add_articulation([joint])


def test_coupled_native_forces_and_observable_ownership(make):
    with newton_builder_world_hook(rigid_contact_world):
        env = make(rigid_solver_cfg=MJWarpSolverCfg())
    binding = env.simulator
    solver = binding.coupled
    output = binding.observables
    contacts = lab_runtime().contacts
    assert output._solver is solver
    assert output._rigid._solver is solver.rigid and output._ours._solver is solver.ours
    assert output.contact_f.ptr != output._rigid.contact_f.ptr
    selected = output.select((SolverBase.ObservableKind.CONTACT_F,))
    assert selected.contact_f is output.contact_f and selected._rigid._solver is solver.rigid
    ptr = output.contact_f.ptr
    for _ in range(10):
        env.step(torch.zeros((2, 10), device=env.device))
    assert output.contact_f.ptr == ptr
    assert np.isfinite(output.contact_f.numpy()).all() and np.abs(output.contact_f.numpy()).max() > 0
    np.testing.assert_array_equal(output.contact_f.numpy(), output._rigid.contact_f.numpy())
    solver.update_contacts(contacts, binding.state)
    assert np.abs(contacts.force.numpy()).max() > 0, "native rigid force export is real"
    with pytest.raises(TypeError, match="instance of Observables"):
        solver.rigid.step(binding.state, binding.state, binding.control, contacts, binding.dt, observables=output)
    with pytest.raises(ValueError, match="solver instance"):
        solver.validate_observables(solver.Observables(), contacts)
    with pytest.raises(ValueError, match="does not support"):
        solver.ours.observables((SolverBase.ObservableKind.CONTACT_F,))
    for owner, name, replacement in ((output, "contact_f", wp.clone(output.contact_f)),
                                      (output, "_rigid", output._rigid.select(output._rigid.kinds)),
                                      (binding, "observables", selected)):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        try:
            with pytest.raises(RuntimeError, match="buffer was replaced"):
                env.sim.step(render=False)
        finally:
            setattr(owner, name, original)
    env._construction.cfg.observables = ()
    try:
        with pytest.raises(RuntimeError, match="requested observables changed"):
            env.sim.step(render=False)
    finally:
        env._construction.cfg.observables = (SolverBase.ObservableKind.CONTACT_F,)
    env.sim.step(render=False)


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("coupled", [False, True])
def test_custom_output_routing_and_rebuild(make, captured, coupled):
    from i4h_physics.observables import KINDS, ObservableKind as Kind
    kinds = tuple(KINDS) + ((Kind.CONTACT_F,) if coupled else ())
    with newton_builder_world_hook(rigid_contact_world) if coupled else nullcontext():
        env = make(observables=kinds, captured=captured, rigid_solver_cfg=MJWarpSolverCfg() if coupled else None)
    binding = env.simulator
    output = binding.observables
    assert binding.model.particle_grid is None
    for _ in range(3):
        env.sim.step(render=False)
    assert output.i4h_wrench.shape == (2,)
    assert output.i4h_surgical_count.shape == (2,)
    if coupled:
        assert output._rigid.kinds == {Kind.CONTACT_F}
        assert output._ours.kinds == KINDS
        assert output.i4h_wrench is output._ours.i4h_wrench
        selected = output.select((Kind.I4H_WRENCH,))
        assert not selected._rigid.kinds and selected._ours.kinds == {Kind.I4H_WRENCH}
        assert selected.contact_f is None
        binding.coupled.step(binding.state, binding.state, binding.control, binding.contacts, binding.dt, selected)
        assert output.select(())._ours.kinds == frozenset()
    before = binding.state.particle_q.numpy().copy()
    original = output.i4h_wrench
    output.i4h_wrench = wp.clone(original)
    try:
        with pytest.raises(RuntimeError, match="replaced"):
            env.sim.step(render=False)
        np.testing.assert_array_equal(binding.state.particle_q.numpy(), before)
    finally:
        output.i4h_wrench = original
    env.sim.reset(soft=False)
    assert not binding.valid
    assert env.simulator.observables is not output
    assert env.simulator.observables.i4h_wrench is not original
    assert env.simulator.model.particle_grid is None
    env.sim.step(render=False)


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("component_contact", [False, True])
def test_contact_sensor_and_component_exports_coexist(make, captured, component_contact):
    from i4h_physics.observables import KINDS
    kinds = tuple(KINDS) + ((SolverBase.ObservableKind.CONTACT_F,) if component_contact else ())
    with newton_builder_world_hook(rigid_contact_world):
        env = make(observables=kinds, captured=captured, rigid_solver_cfg=MJWarpSolverCfg(), decimation=3)
    owner = env._construction
    manager = owner.backend if hasattr(owner.backend, "add_contact_sensor") else owner.manager
    sensor = manager.add_contact_sensor(body_names_expr=".*")
    if not hasattr(sensor, "total_force"):
        sensor = manager._newton_contact_sensors[sensor]
        owner.api.invalidate_graph(owner)
    # Adding an observable is a structural edit under the existing binding contract.
    env.simulator.prepare()
    output = env.simulator.observables
    pointers = {kind: getattr(output, kind).ptr for kind in KINDS}
    # Poison an exported wrench to prove it is refreshed even when the host
    # also requests contact observables on the last decimation substep.
    for _ in range(4):
        output.i4h_wrench.fill_(wp.spatial_vector(123.0))
        env.step(torch.zeros((2, 10), device=env.device))
        assert not np.all(output.i4h_wrench.numpy() == 123.0)
    assert pointers == {kind: getattr(output, kind).ptr for kind in KINDS}
    assert np.isfinite(sensor.total_force.numpy()).all()
    assert np.abs(sensor.total_force.numpy()).max() > 0
    if component_contact:
        assert np.abs(output.contact_f.numpy()).max() > 0
        np.testing.assert_array_equal(output.contact_f.numpy(), output._rigid.contact_f.numpy())
    if hasattr(manager, "contact_observables"):
        assert manager.contact_observables._solver is env.simulator.coupled
        assert manager.add_contact_sensor(body_names_expr=".*") is sensor
        if component_contact:
            np.testing.assert_array_equal(output.contact_f.numpy(), manager.contact_observables.contact_f.numpy())


def test_pure_contact_sensor_registration_is_explicitly_rejected(make):
    env = make()
    owner = env._construction
    manager = owner.backend if hasattr(owner.backend, "add_contact_sensor") else owner.manager
    with pytest.raises(NotImplementedError, match="contact sensors|CONTACT_F"):
        manager.add_contact_sensor(body_names_expr=".*")
