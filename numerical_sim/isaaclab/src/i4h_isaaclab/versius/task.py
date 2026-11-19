"""Native Versius articulation and manager terms beside an i4h component."""

from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from i4h_isaaclab.component import I4hComponent, I4hComponentCfg
from i4h_isaaclab.manager import I4H_ROOT, I4hHostMixin
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from . import terms
from .profiles import TCP_OFFSET, profile


class CartesianVelocity(ActionTerm):
    @property
    def action_dim(self):
        return 3

    @property
    def raw_actions(self):
        return self._env.actions

    @property
    def processed_actions(self):
        return self._env.target_tcp

    def process_actions(self, actions):
        env = self._env
        if not torch.isfinite(actions).all():
            raise ValueError("nonfinite Cartesian action")
        env.previous_patch[:] = terms.measurements(env)[0]
        env.previous_actions[:] = env.actions
        env.actions[:] = actions.clamp(-1, 1)
        # Norm cap also bounds diagonal XYZ movement.
        velocity = env.actions / env.actions.norm(dim=1, keepdim=True).clamp_min(1.0)
        env.target_tcp += velocity * (0.04 * env.step_dt)
        env.target_tcp[:] = env.target_tcp.clamp(
            env.home_tcp - env.workspace_lower, env.home_tcp + env.workspace_upper
        )

    def apply_actions(self):
        self._env.apply_ik()

    def reset(self, env_ids=None):
        if hasattr(self._env, "actions"):
            self._env.actions[slice(None) if env_ids is None else env_ids] = 0


@configclass
class CartesianVelocityCfg(ActionTermCfg):
    class_type: type = CartesianVelocity
    asset_name: str = "robot"


@configclass
class SceneCfg(InteractiveSceneCfg):
    i4h: AssetBaseCfg = I4H_ROOT
    robot: ArticulationCfg = None


@configclass
class ActionsCfg:
    cartesian: CartesianVelocityCfg = CartesianVelocityCfg()


@configclass
class PolicyCfg(ObservationGroupCfg):
    state: ObservationTermCfg = ObservationTermCfg(func=terms.observations)
    enable_corruption: bool = False
    concatenate_terms: bool = True


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    poke: RewardTermCfg = RewardTermCfg(func=terms.reward, weight=1.0)


@configclass
class TerminationsCfg:
    success: TerminationTermCfg = TerminationTermCfg(func=terms.success)
    failure: TerminationTermCfg = TerminationTermCfg(func=terms.failure)
    timeout: TerminationTermCfg = TerminationTermCfg(func=terms.timeout, time_out=True)


@configclass
class PokeCfg(ManagerBasedRLEnvCfg):
    seed: int = 42
    profile: str = "block"
    teleop: bool = False
    decimation: int = 2
    episode_length_s: float = 5.0
    target_depth: float = 0.003
    depth_tolerance: float = 0.0015
    max_depth: float = 0.012
    max_tool_depth: float = 0.018
    compute_final_obs: bool = True
    ui_window_class_type = None
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=8.0)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60, render_interval=2, device="cuda:0", physics=NewtonCfg()
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


def configuration(name, usd, *, envs=1, graph=True, coupling="two_way", teleop=False):
    import os

    usd = usd or os.environ.get("I4H_VERSIUS_USD")
    if not usd or not Path(usd).expanduser().is_file():
        raise ValueError("Versius requires --usd or I4H_VERSIUS_USD pointing to the prepared robot USD")
    physical = profile(name)
    cfg = PokeCfg(profile=name, teleop=teleop)
    cfg.scene.num_envs = envs
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(usd).expanduser().resolve()),
            variants={"Physics": "mujoco"},
            rigid_props={
                "/.*": [
                    PhysxRigidBodyCfg(disable_gravity=True),
                    MujocoRigidBodyCfg(gravcomp=1.0),
                ]
            },
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                "J1": 0.0,
                "J2": -0.5,
                "J3": 0.0,
                "J4": 1.2,
                "J5": 0.0,
                "J6": -0.2,
                "J7": 0.25,
                "J8": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["J[1-8]"],
                stiffness=1200.0,
                damping=80.0,
                joint_effort_limit=200.0,
                joint_velocity_limit=3.0,
                armature=0.1,
            )
        },
    )
    cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(integrator="implicitfast"),
        load_visual_shapes=True,
        num_substeps=5,
        use_cuda_graph=graph,
    )
    # Gravity is rotated with the entire component, while its systems retain authored coordinates.
    cfg.sim.gravity = (0.0, 0.0, -9.81) if name != "block" else (0.0, 0.0, 9.81)
    component_cfg = I4hComponentCfg.from_case(
        physical.case,
        pose=physical.pose,
        mounts={
            "probe": {
                "body": "J8_rotating_model",
                "offset": (TCP_OFFSET, (0.0, 0.0, 0.0, 1.0)),
            }
        },
        coupling_mode=coupling,
    )
    component = I4hComponent(component_cfg)
    component.configure(cfg.sim, gravity_from_case=False)
    return cfg, physical


class PokeEnv(I4hHostMixin, ManagerBasedRLEnv):
    def __init__(self, cfg=None, physical=None, profile_name="block", **kwargs):
        if cfg is None:
            import os

            usd = os.environ.get("I4H_VERSIUS_USD")
            cfg, physical = configuration(profile_name, usd)
        self.physical = physical or profile(cfg.profile)
        import hashlib
        import json

        from .terms import OBSERVATION_LAYOUT

        spec = self.physical
        solver_cfg = cfg.sim.physics.solver_cfg
        self.task_metadata = {
            "schema": "i4h_solver_versius_v1",
            "profile": cfg.profile,
            "observation_layout": OBSERVATION_LAYOUT,
            "pose": spec.pose,
            "patch_ids": spec.patch_ids,
            "component_sha256": hashlib.sha256(
                json.dumps(solver_cfg.case, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "coupling": solver_cfg.coupling_mode,
            "dt": cfg.sim.dt,
            "decimation": cfg.decimation,
            "rigid_substeps": solver_cfg.rigid_substeps,
            "speed": 0.04,
            "success": {
                "depth": cfg.target_depth,
                "tolerance": cfg.depth_tolerance,
                "lateral": 0.006,
                "distance": 0.012,
                "hold": 0.5,
            },
        }
        # JSON round-trip makes tuple/list representations stable in checkpoint sidecars.
        self.task_metadata = json.loads(json.dumps(self.task_metadata, default=float))
        self.ready = False
        self.graph_seen = False
        super().__init__(cfg, **kwargs)

    def step(self, actions):
        result = super().step(actions)
        self.graph_seen |= bool(self.sim.backend_manager.backend.graphs)
        return result

    def _bind_i4h(self, component):
        self.ready = False
        super()._bind_i4h(component)

    def hard_rebuild(self):
        self.ready = False
        self.sim.reset(soft=False)
        self.initialize()
        self.reset()

    def load_managers(self):
        self.initialize()
        super().load_managers()

    def world_points(self):
        local = self.i4h.positions(self.physical.body).reshape(self.num_envs, -1, 3)
        return (
            local @ self.rotation.T
            + self.translation
            + self.scene.env_origins[:, None, :]
        )

    def world_velocities(self):
        import warp as wp

        scene = self.i4h.scene
        local = wp.to_torch(scene.field("qd", scene.body(self.physical.body)))
        return local @ self.rotation.T

    def tcp_position(self):
        return self.robot.data.body_pos_w.torch[:, self.ee_id] + quat_apply(
            self.robot.data.body_quat_w.torch[:, self.ee_id], self.offset
        )

    def joint_targets(self):
        tcp = self.tcp_position()
        quat = self.robot.data.body_quat_w.torch[:, self.ee_id]
        jac = self.robot.data.body_link_jacobian_w.torch[
            :, self.jacobian_row, :, self.joint_ids
        ].clone()
        lever = quat_apply(quat, self.offset)
        jac[:, :3] += torch.linalg.cross(jac[:, 3:], lever[:, :, None], dim=1)
        q = self.robot.data.joint_pos.torch[:, self.joint_ids]
        self.controller.set_command(
            torch.cat((self.target_tcp, self.orientation), dim=-1), tcp, quat
        )
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.joint_ids]
        return self.controller.compute(tcp, quat, jac, q).clamp(
            limits[..., 0], limits[..., 1]
        )

    def apply_ik(self):
        self.robot.actuators.target_command.set_position_index(
            value=self.joint_targets(), joint_ids=self.joint_ids
        )

    def initialize(self):
        from isaaclab.utils.math import matrix_from_quat

        self.robot = self.scene["robot"]
        self.ee_id = self.robot.find_bodies("J8_rotating_model")[0][0]
        self.joint_ids = self.robot.find_joints(["J[1-8]"])[0]
        self.jacobian_row = self.ee_id - int(self.robot.is_fixed_base)
        self.offset = torch.tensor(TCP_OFFSET, device=self.device).repeat(
            self.num_envs, 1
        )
        pos, xyzw = self.physical.pose
        self.rotation = matrix_from_quat(torch.tensor(xyzw, device=self.device))
        self.translation = torch.tensor(pos, device=self.device, dtype=torch.float32)
        self.patch_ids = list(self.physical.patch_ids)
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            self.num_envs,
            self.device,
        )
        self.orientation = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=self.device
        ).repeat(self.num_envs, 1)
        self.robot.write_joint_position_to_sim_index(
            position=self.robot.data.default_joint_pos.torch
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.robot.data.default_joint_pos.torch)
        )
        self.target_tcp = self.world_points()[:, self.patch_ids].mean(1).clone()
        self.target_tcp[:, 2] += 0.08
        for _ in range(100):
            q = self.robot.data.joint_pos.torch.clone()
            q[:, self.joint_ids] += (self.joint_targets() - q[:, self.joint_ids]).clamp(
                -0.1, 0.1
            )
            self.robot.write_joint_position_to_sim_index(position=q)
        error = (self.tcp_position() - self.target_tcp).norm(dim=1)
        if error.max() > 0.005:
            raise RuntimeError(f"Unreachable clear approach: {error.tolist()}")
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        self.robot.set_joint_position_target_index(target=q)
        for _ in range(240):
            self.apply_ik()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            if self.i4h.state()["tool_force.count"].any():
                raise RuntimeError("Tool contact during contact-free settling")
        if not torch.isfinite(self.world_points()).all():
            raise RuntimeError("Nonfinite tissue during settling")
        self.reference_patch = self.world_points()[:, self.patch_ids].mean(1).clone()
        self.target_tcp[:] = self.reference_patch
        self.target_tcp[:, 2] += 0.025
        for _ in range(120):
            self.apply_ik()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            if self.i4h.state()["tool_force.count"].any():
                raise RuntimeError("Tool contact during contact-free settling")
        error = (self.tcp_position() - self.target_tcp).norm(dim=1)
        if error.max() > 0.005:
            raise RuntimeError(f"Unreachable settled approach: {error.tolist()}")
        self.home_joint_pos = self.robot.data.joint_pos.torch.clone()
        self.home_tcp = self.tcp_position().clone()
        self.target_tcp[:] = self.home_tcp
        self.reference_patch = self.world_points()[:, self.patch_ids].mean(1).clone()
        self.previous_patch = self.reference_patch.clone()
        self.workspace_lower = torch.tensor((0.06, 0.06, 0.08), device=self.device)
        self.workspace_upper = torch.tensor((0.06, 0.06, 0.04), device=self.device)
        self.actions = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.hold_time = torch.zeros(self.num_envs, device=self.device)
        self.peak_depth = torch.zeros_like(self.hold_time)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failure = torch.zeros_like(self.success)
        self.progress_step = torch.full(
            (self.num_envs,),
            self.common_step_counter,
            device=self.device,
            dtype=torch.long,
        )
        model = self.sim.backend_manager.backend.model
        self.saved_state, self.saved_control = model.state(), model.control()
        self.i4h.save(self.saved_state, self.saved_control)
        self.ready = True
        print(
            "VERSIUS_READY",
            self.physical.name,
            "TCP",
            self.home_tcp.tolist(),
            "patch",
            self.reference_patch.tolist(),
            flush=True,
        )

    def _reset_idx(self, env_ids):
        if not self.ready:
            return ManagerBasedRLEnv._reset_idx(self, env_ids)
        ids = (
            torch.arange(self.num_envs, device=self.device)[env_ids]
            if isinstance(env_ids, slice)
            else env_ids
        )
        self.last_episode = {
            "success": self.success[ids].clone(),
            "failure": self.failure[ids].clone(),
            "depth": terms.measurements(self)[2][ids].clone(),
            "ids": ids.clone(),
        }
        ManagerBasedRLEnv._reset_idx(self, ids)
        self.robot.write_joint_position_to_sim_index(
            position=self.home_joint_pos[ids], env_ids=ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.home_joint_pos[ids]), env_ids=ids
        )
        self.robot.set_joint_position_target_index(
            target=self.home_joint_pos[ids], env_ids=ids
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.i4h.restore(self.saved_state, self.saved_control, ids)
        self.i4h.sync_views()
        self.target_tcp[ids] = self.home_tcp[ids]
        self.previous_patch[ids] = self.reference_patch[ids]
        self.controller.reset(ids)
        self.controller.ee_pos_des[ids] = self.home_tcp[ids]
        self.controller.ee_quat_des[ids] = self.orientation[ids]
        for name in (
            "actions",
            "previous_actions",
            "hold_time",
            "peak_depth",
            "success",
            "failure",
        ):
            getattr(self, name)[ids] = 0
        self.progress_step[ids] = self.common_step_counter


def block_cfg():
    return configuration("block", None)[0]


def chole_cfg():
    return configuration("chole", None)[0]


def chole_1x_cfg():
    return configuration("chole-1x", None)[0]
