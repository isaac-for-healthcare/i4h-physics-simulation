# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F3: i4h physics as a component of a host Isaac Lab env (``i4h_isaaclab.component``), built from explicit parameters
next to a host rigid body, with a loader and a control system defined here, outside the adapter packages. Runs only
with the Isaac Lab venv's python (CUDA, MuJoCo-Warp)."""

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

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
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager  # noqa: E402

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


def test_component_from_parameters_beside_a_host_body(host):
    env, component = host()
    sim = component.sim
    assert sim.coupled is not None and NewtonManager._solver is sim.coupled, "the host's MJWarp steps beside i4h"
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
    arm = NewtonManager.get_state_0().body_q.numpy()
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
    from i4h_isaaclab.newton_manager import I4hNewtonManager

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
        assert component.sim.coupled is None and NewtonManager._solver is component.sim.solver
        assert component._bind in I4hNewtonManager.on_solver
        component.detach()
        assert component._bind not in I4hNewtonManager.on_solver
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


def test_configure_preserves_the_hosts_newton_settings():
    """Adding the component swaps only the solver and the substep split: the host's shape contact defaults,
    determinism, visual-shape loading and graph setting stay as the workflow set them."""
    from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg

    shape = NewtonShapeCfg(margin=0.015, gap=0.02, ke=1.0e4, kd=50.0, mu=0.25)
    sim_cfg = SimulationCfg(dt=1.0 / 120, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=2,
                                                            deterministic_mode="run_to_run", default_shape_cfg=shape,
                                                            load_visual_shapes=True, use_cuda_graph=False))
    I4hComponent(component_cfg()).configure(sim_cfg)
    physics = sim_cfg.physics
    assert (physics.default_shape_cfg.mu, physics.default_shape_cfg.margin) == (0.25, 0.015)
    assert physics.deterministic_mode == "run_to_run" and physics.load_visual_shapes is True
    assert physics.use_cuda_graph is False and physics.num_substeps == 1
    assert physics.solver_cfg.rigid_substeps == 2 and isinstance(physics.solver_cfg.rigid_solver_cfg, MJWarpSolverCfg)


def test_configure_rejects_an_unsupported_rigid_backend():
    """rigid=True needs NewtonCfg(MJWarpSolverCfg): a PhysX host would otherwise lose its rigid simulation silently.
    i4h alone stays an explicit rigid=False."""
    from isaaclab_physx.physics import PhysxCfg

    host = SimulationCfg(dt=1.0 / 120, physics=PhysxCfg())
    with pytest.raises(ValueError, match="NewtonCfg\\(solver_cfg=MJWarpSolverCfg"):
        I4hComponent(component_cfg()).configure(host)
    assert isinstance(host.physics, PhysxCfg), "a rejected configure leaves the host untouched"
    I4hComponent(component_cfg()).configure(host, rigid=False)
    assert host.physics.solver_cfg.rigid_solver_cfg is None
    from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg

    collide = NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False), collision_cfg=NewtonCollisionPipelineCfg())
    with pytest.raises(ValueError, match="use_mujoco_contacts=False"):
        I4hComponent(component_cfg()).configure(SimulationCfg(dt=1.0 / 120, physics=collide))
    with pytest.raises(ValueError, match="collision_cfg"):  # even without rigid bodies: no Newton pipeline runs
        I4hComponent(component_cfg()).configure(SimulationCfg(dt=1.0 / 120, physics=collide), rigid=False)
