# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F3: i4h physics as a component of a host Isaac Lab env (``i4h_isaaclab.component``), built from explicit parameters
next to a host rigid body, with a loader and a control system defined here, outside the adapter packages. Runs only
with the Isaac Lab venv's python (CUDA, MuJoCo-Warp)."""

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp
from helpers import lab_runtime

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")
pytest.importorskip("mujoco_warp")

from i4h_isaaclab.component import I4hComponent, I4hComponentCfg  # noqa: E402
from i4h_physics.loaders.grid import add_tet_grid  # noqa: E402
from i4h_physics.systems.base import System  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402

# --- defined by the "workflow": a loader and a control the adapter has never heard of ------------------------------

def add_slab(scene, name, size=3):
    """A hanging block of 3x3x3 tet cells (64 nodes) (a custom asset type over the physics package's grid loader)."""
    return add_tet_grid(scene, name, dim=[size] * 3, cell=0.02, origin=(-0.02, -0.02, 0.05), fix_top=True,
                        parts={name: {"spring": 1.0, "volume": 1.0}})


@dataclass(eq=False)
class Push(System):
    """A custom control: ``actions (E, 2)`` push every free particle sideways (a velocity change per step)."""

    gain: float = 1.0
    spin: int = 0  # a busy loop before the counter update: Warp's step outlasts a racing torch read

    is_control = True
    state_products = ("pushes",)

    def __post_init__(self):
        E, dev = self.scene.envs, self.scene.device
        self.actions = wp.zeros((E, 2), dtype=wp.float32, device=dev)
        self.pushes = wp.zeros(E, dtype=wp.int32, device=dev)

    def action_bounds(self, workspace):
        return [-2.0, -2.0], [2.0, 2.0]

    def step_start(self, scene, dt):
        if self.spin:
            wp.launch(busy, dim=scene.envs, inputs=[self.spin], outputs=[self.pushes], device=scene.device)
        f = scene.block_field
        wp.launch(push, dim=(scene.envs, scene.main.width), inputs=[self.actions, f("inv_mass"), self.gain],
                  outputs=[f("qd"), self.pushes], device=scene.device)

    def reset(self, scene, mask):
        self.zero(mask, "pushes")
        self.clear(mask, (self.actions, wp.float32(0.0)))


@wp.kernel
def busy(loops: wp.int32, pushes: wp.array(dtype=wp.int32)):
    env = wp.tid()
    acc = wp.int32(0)
    for i in range(loops):
        acc = (acc * 1103515245 + 12345) % 2147483647
    if acc == -1:  # never: keeps the loop
        pushes[env] = acc


@wp.kernel
def push(actions: wp.array2d(dtype=wp.float32), inv_mass: wp.array2d(dtype=wp.float32), gain: wp.float32,
         qd: wp.array2d(dtype=wp.vec3), pushes: wp.array(dtype=wp.int32)):
    env, i = wp.tid()
    a = wp.vec3(actions[env, 0], actions[env, 1], 0.0) * gain
    if inv_mass[env, i] > 0.0:
        qd[env, i] = qd[env, i] + a * 0.01
    if i == 0 and wp.length(a) > 0.0:
        pushes[env] = pushes[env] + 1


def component_cfg(sim=None, **kw):
    return I4hComponentCfg(
        assets={"tissue": {"type": "slab"}},
        systems=[{"type": "particles"}, {"type": "tet", "body": "tissue"}, {"type": "push", "id": "push"},
                 {"type": "bounds", "lower": [-0.5, -0.5, 0.0], "upper": [0.5, 0.5, 1.0]}],
        sim={"substeps": 4, "iterations": 4, "cuda_graph": True, **(sim or {})},
        loaders={"slab": add_slab}, systems_registry={"push": Push}, **kw)


# --- the host env: its own scene, robots (a hook body here), actions, observations and resets ----------------------

@configclass
class HostEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 100.0
    action_space = 2
    observation_space = 1
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)


class HostEnv(DirectRLEnv):
    def __init__(self, cfg, component):
        self.component = component
        super().__init__(cfg)

    def _setup_scene(self):
        """A host scene replicates its assets into one world per env; this one has none of its own (its "robot" comes
        from a builder hook), so it replicates an empty env root as I4hEnv does."""
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.cloner import ReplicateSession

        root = AssetBaseCfg(prim_path=self.scene.cloner_cfg.clone_template.format("[^/]+"), cloning_contexts=None)
        with ReplicateSession([root], self.num_envs, self.cfg.scene.env_spacing, env_template=self.scene.cloner_cfg.clone_template):
            pass

    def _pre_physics_step(self, actions):
        self.component.write_actions(actions)

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {"policy": self.component.positions("tissue")[:, :1].clone()}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        none = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return none, none

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.component.reset(env_ids)


def pendulum(builder, env, position, quaternion):
    frame = wp.transform(wp.vec3(*map(float, position)), wp.quat(*map(float, quaternion)))
    link = builder.add_link(xform=frame * wp.transform((0.0, 2.0, 1.0), wp.quat_identity()), mass=1.0, label="arm")
    builder.add_shape_capsule(link, radius=0.05, half_height=0.2)
    j = builder.add_joint_revolute(-1, link, parent_xform=frame * wp.transform((0.0, 2.0, 1.5), wp.quat_identity()),
                                   child_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), axis=(1.0, 0.0, 0.0))
    builder.add_articulation([j])


@pytest.fixture
def host():
    from isaaclab_newton.cloner import newton_builder_world_hook

    built = []

    def make(cfg=None):
        component = I4hComponent(cfg or component_cfg())
        env_cfg = HostEnvCfg()
        component.configure(env_cfg.sim)
        with newton_builder_world_hook(pendulum), component.construction(device=env_cfg.sim.device, envs=2):
            env = HostEnv(env_cfg, component)
        built.append(env)
        return env, component

    yield make
    for env in built:
        env.close()
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()


def test_configuration_preserves_host_settings():
    from dataclasses import fields

    component = I4hComponent(component_cfg())
    cfg = HostEnvCfg()
    cfg.sim.physics.use_cuda_graph = False
    cfg.sim.physics.num_substeps = 3
    # Even unsupported settings must reach the adapter's explicit validation.
    cfg.sim.physics.collision_cfg = {"unsupported": True}
    previous = cfg.sim.physics
    component.configure(cfg.sim)
    for field in fields(previous):
        if field.name not in ("class_type", "solver_cfg", "num_substeps"):
            assert getattr(cfg.sim.physics, field.name) == getattr(previous, field.name)
    assert cfg.sim.physics.num_substeps == 1 and component.solver_cfg.rigid_substeps == 3


def test_component_from_parameters_beside_a_host_body(host):
    env, component = host()
    sim = component.sim
    assert sim.coupled is not None and lab_runtime().solver is sim.coupled, "the host's MJWarp steps beside i4h"
    assert [type(s).__name__ for s in component.systems] == ["Particles", "Tet", "Push", "Bounds"]
    low, high = component.action_bounds()
    np.testing.assert_array_equal(low, [-2.0, -2.0]) and np.testing.assert_array_equal(high, [2.0, 2.0])
    base, stride, col, width = component.particle_range
    assert width == 64 and stride >= width and 0 <= col <= stride - width
    env.reset()
    still = component.positions("tissue").clone()
    actions = torch.tensor([[2.0, 0.0], [0.0, 0.0]], device=env.device)
    for _ in range(20):
        env.step(actions)
    component.sync_views()
    q = component.positions("tissue").view(2, -1, 3)
    drift = (q - still.view(2, -1, 3))[..., 0].mean(dim=1)
    assert drift[0] > 5e-4 and drift[0] > 5.0 * abs(float(drift[1])), "only env 0 was pushed (env 1 only settles)"
    state = component.state()
    assert state["push.pushes"].tolist() == [[20], [0]]
    arm = lab_runtime().state_0.body_q.numpy()
    assert np.isfinite(arm).all()
    component.reset(env_ids=[0])
    component.sync_views()
    np.testing.assert_allclose(component.positions("tissue")[0].cpu().numpy(), still[0].cpu().numpy(), atol=1e-6)
    assert component.state()["push.pushes"].tolist() == [[0], [0]]


def test_handle_rebinds_across_a_hard_rebuild(host):
    env, component = host()
    seen = []
    component.on_rebuild(lambda c: seen.append(c.sim))
    old = component.sim
    env.sim.reset(soft=False)
    assert component.sim is not old and seen == [component.sim], "the handle rebound and told the host"
    with pytest.raises(RuntimeError, match="invalidated"):
        old.physics.check_valid()
    assert component.particle_range[3] == 64
    env.step(torch.zeros((2, 2), device=env.device))


def test_position_initialization_after_rebuild_and_selected_host_edits(host):
    env, component = host()
    dev = torch.device(env.device)

    def initialize(c):
        c.positions("tissue").add_(0.01)  # construction orders and completes these writes automatically

    component.on_rebuild(initialize)
    env.sim.reset(soft=False)
    component.sync_views()
    q = component.positions("tissue")
    history = wp.to_torch(component.scene.block_field("q_prev")).view(2, -1)
    assert torch.equal(history, q)
    rest = wp.to_torch(component.scene.block_field("q0")).view(2, -1)
    assert torch.equal(q, rest + 0.01), "rebuild callback position edits survive construction"
    original = history.clone()
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        component.sync_views()
        q[1].add_(0.02)
        component.initialize_particle_history([1])
        component.sync_views()
    torch.cuda.synchronize()
    assert torch.equal(history[1], q[1])
    assert torch.equal(history[0], original[0])
    unchanged = history.clone()
    component.initialize_particle_history([])
    component.initialize_particle_history([1])
    component.sync_views()
    assert torch.equal(history, unchanged)


def test_explicit_maps_replace_domain_discovery(host):
    """With explicit maps only the physics package's types and the given ones exist: a domain system is unknown."""
    cfg = component_cfg()
    cfg.systems = cfg.systems + [{"type": "instrument_control"}]
    with pytest.raises(ValueError, match="unknown type 'instrument_control'"):
        host(cfg)


def test_configure_keeps_the_hosts_rigid_substeps():
    sim_cfg = SimulationCfg(dt=1.0 / 120, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=2))
    component = I4hComponent(component_cfg())
    component.configure(sim_cfg)
    assert sim_cfg.physics.num_substeps == 1, "Lab steps the coupled solver once per physics step"
    assert component.solver_cfg.rigid_substeps == 2 and isinstance(component.solver_cfg.rigid_solver_cfg, MJWarpSolverCfg)
    assert component.solver_cfg.case["sim"]["dt"] == 1.0 / 120
    assert sim_cfg.gravity == (0.0, 0.0, -9.81), "a case without sim.gravity keeps the host's"
    zero = SimulationCfg(dt=1.0 / 120, gravity=(0.0, 0.0, -9.81))
    I4hComponent(component_cfg(sim={"gravity": [0.0, 0.0, 0.0]})).configure(zero, rigid=False)
    assert zero.gravity == (0.0, 0.0, 0.0), "the case's gravity sets the host's"
    keep = SimulationCfg(dt=1.0 / 120, gravity=(0.0, 0.0, -9.81))
    I4hComponent(component_cfg(sim={"gravity": [0.0, 0.0, 0.0]})).configure(keep, rigid=False, gravity_from_case=False)
    assert keep.gravity == (0.0, 0.0, -9.81)
    with pytest.raises(ValueError, match="already steps an i4h component"):
        I4hComponent(component_cfg()).configure(sim_cfg)


def test_component_from_a_case_without_rigid_bodies(host):
    """I4hComponentCfg.from_case keeps the physical keys; configure(rigid=False) runs i4h alone under a host whose
    NewtonCfg names MuJoCo-Warp; a handle over the configured solver cfg builds the same env; detach() releases it."""

    cfg = I4hComponentCfg.from_case("two_worlds")
    assert set(cfg.assets) >= {"block", "rod"} and "probe" in cfg.instruments and cfg.sim["substeps"] == 4
    env_cfg = HostEnvCfg()
    env_cfg.sim.dt = cfg.sim["dt"]
    I4hComponent(cfg).configure(env_cfg.sim, rigid=False)
    solver_cfg = env_cfg.sim.physics.solver_cfg
    assert solver_cfg.rigid_solver_cfg is None and "render" not in solver_cfg.case
    component = I4hComponent.from_solver_cfg(solver_cfg)
    with pytest.raises(RuntimeError, match="needs an I4hComponentCfg"):
        component.configure(env_cfg.sim)
    with component.construction(device=env_cfg.sim.device, envs=2):
        env = HostEnv(env_cfg, component)
    try:
        assert component.sim.coupled is None and lab_runtime().solver is component.sim.solver
        owner = component._construction
        assert owner.on_solver == component._bind and owner.binding is component.sim
        commands = component.command_views("instrument_control")
        assert component.command_views("instrument_control") is commands
        assert commands["position"].shape == (2, 1, 3)
        commands["energy"].fill_(.75)
        component._order("in")
        np.testing.assert_array_equal(component.sim.physics.command_views("instrument_control")["energy"].numpy(), .75)
        component.detach()
        with pytest.raises(RuntimeError, match="invalidated"):
            component.command_views("instrument_control")
        assert owner.closed and not owner.subscriptions and not component.sim.valid
    finally:
        env.close()
    with pytest.raises(ValueError, match="takes an I4hComponentCfg or"):
        I4hComponent()


def test_views_read_after_the_physics_step_on_any_torch_stream(host):
    """The handle orders torch after Warp's step (the manager's after_step), so a host's observation, reward or
    termination function reading on a non-default torch stream sees the step that just ran, also after a rebuild."""
    cfg = component_cfg()
    cfg.systems = [dict(s, spin=2_000_000) if s["type"] == "push" else s for s in cfg.systems]
    env, component = host(cfg)
    env.reset()
    stream = torch.cuda.Stream()
    actions = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=env.device)
    for rebuilt in (False, True):
        if rebuilt:
            env.sim.reset(soft=False)
            env.reset()
        counts = []
        with torch.cuda.stream(stream):
            for _ in range(4):
                env.step(actions)
                counts.append(int(component.state()["push.pushes"][0, 0].clone().item()))
        assert counts == [counts[0] + k for k in range(4)], f"stale reads {counts} (rebuilt={rebuilt})"


def test_save_and_restore_through_the_component(host, monkeypatch):
    """A host saves the i4h rows of its live State and Control and restores them later: positions return exactly,
    the host's own bodies are not written, and no cache reset or preparation runs after the restore."""
    from i4h_physics.sim import I4hSolver

    env, component = host()
    env.reset()
    actions = torch.tensor([[2.0, 0.0], [0.0, 1.0]], device=env.device)
    for _ in range(5):
        env.step(actions)
    model = component.scene.model
    saved = model.state(), model.control()
    component.save(*saved)
    component.sync_views()
    at_save = component.positions("tissue").clone()
    for _ in range(5):
        env.step(actions)
    component.sync_views()
    assert not torch.equal(component.positions("tissue"), at_save)
    arm = lab_runtime().state_0.body_q.numpy().copy()
    resets = []
    reset = I4hSolver.reset
    monkeypatch.setattr(I4hSolver, "reset", lambda self, *args, **kwargs: (resets.append(args), reset(self, *args, **kwargs))[1])
    component.restore(*saved, env_ids=[1])
    component.sync_views()
    assert torch.equal(component.positions("tissue")[1], at_save[1]) and not torch.equal(component.positions("tissue")[0], at_save[0])
    component.restore(*saved)
    component.sync_views()
    assert torch.equal(component.positions("tissue"), at_save)
    np.testing.assert_array_equal(lab_runtime().state_0.body_q.numpy(), arm)
    env.step(actions)
    assert not resets, "a cache reset ran after the restore"
    # A host that also restores its own bodies writes them through Lab, which schedules the solver's cache reset
    # for those worlds. Written first and forwarded, that reset runs before the i4h rows are restored.
    sim = SimulationContext.instance()
    manager = sim.backend_manager if hasattr(sim, "backend_manager") else sim.physics_manager
    worlds = wp.ones(2, dtype=wp.bool, device=model.device)
    manager.invalidate_fk(env_mask=worlds)
    manager.forward()
    assert resets, "the host's forward did not run the cache reset"
    resets.clear()
    component.restore(*saved)
    env.step(actions)
    assert not resets, "a cache reset ran after the restore"
    component.restore(*saved)
    manager.invalidate_fk(env_mask=worlds)  # the wrong order: the cache reset follows the restore
    env.step(actions)
    assert resets
    component.sync_views()
    assert torch.isfinite(component.positions("tissue")).all()
