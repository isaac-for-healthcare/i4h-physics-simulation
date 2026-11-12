# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DirectRLEnv wrapper around an i4h_sim case (one env per process; import only inside an Isaac Lab process).

Isaac Lab owns the loop. The action space is the normalised ``Box(-1, 1)`` over the control systems'
columns; ``_pre_physics_step`` maps it onto the physical bounds of ``views.action_bounds`` (instrument
positions within ``workspace`` of their initial pose, quaternion, grip, energy, fire; catheter rates) and
``_apply_action`` copies the result into the systems' actions arrays through zero-copy torch views. With
``action_mode = "physical"`` (teleop, recorded sessions) the action is those physical rows, written unclamped.

Isaac Lab's Newton backend owns model/state/control allocation, stepping and capture.
A component construction context authors particles into every world and publishes a
PhysicsBinding after solver construction, again after hard rebuilds. Observations
are ``views.observation``: positions of the integrated bodies (or one part, ``observe = "body.part"``) and
the per-env state products of the systems, cast to float; with ``camera > 0`` an ``rgb`` and a ``depth``
image per env are added, rendered by ``i4h_isaaclab.camera.tiled.TiledCamera`` after the step.

Streams: Warp launches, both graph replays and the camera run on Warp's own (blocking) stream; torch orders
against it with events only (``_torch_to_warp`` before Warp consumes torch writes, ``_warp_to_torch``
before torch reads Warp results; ``select_envs`` orders the reset ids and masks itself). The physics manager calls
``_warp_to_torch`` after every physics call, so ``_get_dones``/``_get_rewards``, overridden or not, read synced views;
``sync_views()`` does the same for code outside the step. Resets:
``_reset_envs_from_buffer`` hands Isaac Lab's ``reset_buf`` to ``simulator.reset(mask=...)`` and returns None, so the per-step ``nonzero()`` sync of the base class never
runs; the id path (``_reset_idx``) remains for ``env.reset()``, for cfgs that need ids (events, noise,
final observations) and for subclasses that override ``_reset_idx`` (they keep the base class's per-reset sync).
``_reset_idx`` resets the physics, then runs Isaac Lab's reset events, then resets the procedure.

Rewards and dones: without a procedure the reward is zero and episodes end on the episode-length limit only. With
one (the case's ``procedure:``), the reward is the step-completion pulses weighted by ``step_weights``; an episode
terminates on success or failure and times out on the procedure's deadline or the episode-length limit. The task row
is the ``task`` observation; terminal rows survive the automatic reset in ``extras``.

Gravity: with ``gravity_from_case`` (default) a case's ``sim.gravity`` sets ``sim.gravity``; otherwise
``sim.gravity`` rules and a different case gravity raises. The host builds with the resolved value.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from i4h_isaaclab.composition import check_dt, read_case, sim_settings
from i4h_isaaclab.lab import require
from i4h_isaaclab.newton_manager import I4hSolverCfg, physics_cfg
from i4h_isaaclab.terminal import ProcedureTorch
from i4h_isaaclab.views import (
    action_bounds,
    action_views,
    map_actions,
    observation,
    observation_views,
    state_views,
    write_actions,
)
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

require(DirectRLEnv, "_reset_envs_from_buffer")
require(DirectRLEnvCfg, "compute_final_obs")

@configclass
class I4hEnvCfg(DirectRLEnvCfg):
    case: str = "two_worlds"
    observe: str = ""  # "body" or "body.part" to select particles, else every integrated body
    workspace: float = 0.5  # sim units: instrument positions may be commanded within this of their initial pose
    camera: int = 0  # square resolution of the rgb/depth observation rendered by the tiled camera; 0 = none
    camera_mode: str = ""  # free | tip (per-env tip camera on the case's render.tip); empty: the case's render.camera_mode
    step_weights: tuple = ()  # completion reward weights in authored order; empty defaults to ones
    stagger_episodes: bool = True  # randomise episode_length_buf at the first reset so episodes end at different steps
    seed: int = 0
    decimation: int = 2
    episode_length_s: float = 5.0
    action_space: int = 0  # replaced by the normalised Box in _setup_scene
    observation_space: int = 0  # set from the loaded case in _setup_scene
    state_space: int = 0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2, device="cuda:0", physics=physics_cfg(I4hSolverCfg()))
    ui_window_class_type = None
    backend: str = "newton"  # sole supported host path; legacy recording metadata is translated by the reader
    action_mode: str = "normalized"  # normalized: Box(-1, 1) mapped onto the action bounds; physical: rows as given (teleop, replay)
    rigid_solver_cfg: object = None  # newton backend: MJWarpSolverCfg steps the scene's rigid bodies beside i4h
    coupling_mode: str = "one_way"  # with rigid_solver_cfg: two_way also returns i4h's reactions to the bodies
    mounts: dict | None = None  # with rigid_solver_cfg: {instrument: {"body": label, "offset": (pos, quat_xyzw)}}
    rigid_substeps: int = 1  # with rigid_solver_cfg: rigid solver steps per physics step
    gravity_from_case: bool = True  # a case's sim.gravity sets sim.gravity; False: sim.gravity rules, a different one raises
    teleop_devices: DevicesCfg = DevicesCfg()  # none by default; i4h_standalone.devices.configure_teleop adds "i4h"


    def __post_init__(self):
        sync_teleop_case(self)


def sync_teleop_case(cfg):
    """Device configurations that carry a ``case`` (the standalone i4h device) follow the environment's case."""
    for device in cfg.teleop_devices.devices.values():
        if hasattr(device, "case"):
            device.case = cfg.case


def apply_backend(cfg: I4hEnvCfg) -> None:
    """Configure the sole host path, selecting a distinct adapter at cfg construction."""
    from dataclasses import fields, replace

    from isaaclab_newton.physics import NewtonCfg

    if cfg.backend != "newton":
        raise ValueError("the alternate i4h backend was removed; use the newton host path")
    previous = cfg.sim.physics
    solver = previous.solver_cfg if isinstance(previous, NewtonCfg) and isinstance(previous.solver_cfg, I4hSolverCfg) else I4hSolverCfg()
    solver = replace(solver, rigid_solver_cfg=cfg.rigid_solver_cfg, coupling_mode=cfg.coupling_mode,
                     mounts=cfg.mounts, rigid_substeps=cfg.rigid_substeps)
    kwargs = {f.name: getattr(previous, f.name) for f in fields(previous)
              if f.name not in ("class_type", "solver_cfg", "num_substeps")} if isinstance(previous, NewtonCfg) else {}
    cfg.sim.physics = physics_cfg(solver, **kwargs)


def resolve_gravity(cfg: I4hEnvCfg, case: dict) -> tuple:
    """The env's gravity. With ``gravity_from_case`` (default) a case that declares
    ``sim.gravity`` (chole is Y-up, the aorta cases are weightless) sets ``cfg.sim.gravity``; otherwise
    ``cfg.sim.gravity`` rules and a different declared gravity raises. A case without one uses ``cfg.sim.gravity``."""
    declared = (case.get("sim") or {}).get("gravity")
    if declared is not None:
        declared = tuple(float(g) for g in declared)
        if cfg.gravity_from_case:
            cfg.sim.gravity = declared
        elif np.abs(np.subtract(declared, cfg.sim.gravity)).max() > 1.0e-6 * max(1.0, float(np.linalg.norm(declared))):
            raise ValueError(f"cfg.sim.gravity {tuple(cfg.sim.gravity)} differs from the case's sim.gravity {declared}; set "
                             "gravity_from_case=True to use the case's, or drop sim.gravity from the case")
    return tuple(float(g) for g in cfg.sim.gravity)


def with_gravity(case: dict, gravity: tuple) -> dict:
    """``case`` with ``sim.gravity`` set to the resolved value, so both backends (and recordings) build with it."""
    return {**case, "sim": {**(case.get("sim") or {}), "gravity": list(gravity)}}


class I4hEnv(DirectRLEnv):
    cfg: I4hEnvCfg

    def __init__(self, cfg: I4hEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.action_mode not in ("normalized", "physical"):
            raise ValueError(f"I4hEnvCfg.action_mode must be normalized or physical, got {cfg.action_mode!r}")
        sync_teleop_case(cfg)
        apply_backend(cfg)
        case = read_case(cfg.case)  # once per construction
        self.resolved_case = with_gravity(case, resolve_gravity(cfg, case))  # _setup_scene runs inside super().__init__
        from i4h_isaaclab.newton_manager import construction_context

        cfg.sim.physics.solver_cfg.case = self.resolved_case
        self._construction = None
        try:
            with construction_context(cfg.sim.physics.solver_cfg, device=cfg.sim.device,
                                      envs=cfg.scene.num_envs, on_solver=self._bind_newton) as owner:
                self._construction = owner
                super().__init__(cfg, render_mode, **kwargs)
                if cfg.events and "startup" in self.event_manager.available_modes:
                    self._torch_to_warp()
                    self.simulator.initialize_particle_history()
                    self._warp_to_torch()
        except BaseException:
            from isaaclab.sim import SimulationContext
            if SimulationContext.instance() is not None:
                SimulationContext.clear_instance()
            raise

    def _setup_scene(self):
        case = self.resolved_case
        check_dt(self.physics_dt, sim_settings(case)["dt"])  # the physics dt is Isaac Lab's; the manager checks it again
        self._torch_device = torch.device(self.device)
        self._zero_reward = torch.zeros(self.num_envs, device=self._torch_device)
        self._never_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self._torch_device)
        self._ext = wp.stream_to_torch(wp.get_stream(self.device))  # torch's view of Warp's stream, for event ordering
        self._ev_in, self._ev_out = torch.cuda.Event(), torch.cuda.Event()  # reused: no event creation per step
        self._ids_reset_needed = bool(self.cfg.events or self.cfg.action_noise_model or self.cfg.observation_noise_model
                                      or self.cfg.compute_final_obs or type(self)._reset_idx is not I4hEnv._reset_idx)
        self._staggered = False
        self.camera = None
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.cloner import ReplicateSession

        # The authored environment root routes an empty USD world through Lab, so Lab
        # still builds one native world per environment through the public hook.
        root = AssetBaseCfg(prim_path=self.scene.cloner_cfg.clone_template.format("[^/]+"), cloning_contexts=None)
        with ReplicateSession([root], self.num_envs, self.cfg.scene.env_spacing,
                              env_template=self.scene.cloner_cfg.clone_template):
            pass
        self.simulator = None  # bound when sim.reset() builds the solver

    def _bind_newton(self, sim):
        """Rebind views after initial construction and each hard rebuild."""
        self.simulator = sim
        self._bind(sim)
        self._warp_to_torch()  # startup events may read the freshly constructed particle views
        self._camera_prepared = False

    def close(self):
        if getattr(self, "_construction", None) is not None:
            self._construction.close()
            self._construction = None
        super().close()

    def _configure_gym_env_spaces(self):
        if self.simulator is None:
            raise RuntimeError("I4hEnv: the solver was not built by sim.reset()")
        self._set_spaces()
        super()._configure_gym_env_spaces()
        # DirectRLEnv declares only "policy" and batches it alone: declare every returned key and batch the whole Dict
        spaces = self.single_observation_space
        if self.procedure is not None:
            spaces["task"] = gym.spaces.Box(-np.inf, np.inf, shape=(self.procedure.observation_size,), dtype=np.float32)
        if self.camera is not None:
            h, w = self.camera.height, self.camera.width
            spaces["rgb"] = gym.spaces.Box(0, 255, shape=(h, w, 4), dtype=np.uint8)
            spaces["depth"] = gym.spaces.Box(-np.inf, np.inf, shape=(h, w), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(spaces, self.num_envs)

    def _set_spaces(self):
        if self.cfg.action_mode == "physical":  # informational: physical rows are written unclamped
            self.cfg.action_space = gym.spaces.Box(self.low.cpu().numpy(), self.high.cpu().numpy(), dtype=np.float32)
        else:
            self.cfg.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.low.shape[0],), dtype=np.float32)
        policy_size = int(sum(v.shape[1] for v in self.obs_views + self.state_views))
        self.cfg.observation_space = policy_size  # the "policy" group; the other keys are added in _configure_gym_env_spaces

    def _bind(self, sim):
        """Zero-copy views, bounds and camera over the current PhysicsBinding."""
        case = self.resolved_case
        from i4h_isaaclab.composition import build_procedure

        self.procedure = build_procedure(case, sim, horizon_s=self.max_episode_length * self.step_dt)
        sim.procedure = self.procedure
        sim.after_step = self._warp_to_torch
        self.task = None
        if self.procedure is not None:
            self.task = task = ProcedureTorch(self.procedure, self.cfg.step_weights, self._torch_device)
            self.task_rows, self.task_completions, self.step_weights = task.rows, task.completions, task.step_weights
            self.task_progress, self.task_history, self.task_metrics = task.progress, task.history, task.metrics
            self.terminal_task, self.terminal_history = task.terminal_task, task.terminal_history
            self.terminal_metrics, self.new_terminal = task.terminal_metrics, task.new_terminal
        self.action_views = action_views(sim)
        low, high = action_bounds(sim, self.cfg.workspace)
        self.low, self.high = torch.as_tensor(low, device=self._torch_device), torch.as_tensor(high, device=self._torch_device)
        self.action_mid, self.action_half = (self.low + self.high) / 2.0, (self.high - self.low) / 2.0
        self._actions = torch.zeros((self.num_envs, low.shape[0]), device=self._torch_device)
        self.obs_views = observation_views(sim, self.cfg.observe)
        self.state_views = state_views(sim)
        self.camera = None
        if self.cfg.camera:
            from i4h_isaaclab.camera.tiled import TiledCamera

            render = case.get("render") or {}
            self.camera = TiledCamera(sim, self.cfg.camera, self.cfg.camera, camera=render.get("camera"),
                                      camera_mode=self.cfg.camera_mode or render.get("camera_mode", "free"), tip=render.get("tip"))

    # stream ordering ----------------------------------------------------------------------------------------

    def _torch_to_warp(self):
        """Warp's stream waits for torch's current stream (after torch wrote actions or the reset mask)."""
        self._ev_in.record(torch.cuda.current_stream(self._torch_device))
        self._ext.wait_event(self._ev_in)

    def _warp_to_torch(self):
        """Torch's current stream waits for Warp's (before torch reads positions, state or images)."""
        self._ev_out.record(self._ext)
        torch.cuda.current_stream(self._torch_device).wait_event(self._ev_out)

    def sync_views(self):
        """Order torch's current stream after Warp's work so far: call before reading the zero-copy views (positions,
        state, task rows) outside the step. Within ``step()`` the physics manager already does it after each physics
        call, so overridden ``_get_dones``/``_get_rewards`` read them directly."""
        self._warp_to_torch()

    # DirectRLEnv hooks --------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.task is not None:
            self._torch_to_warp()
            self.task.begin_action(self.extras)
        if self.cfg.action_mode == "physical":  # the teleop rows (poses in sim units, quaternion, grip, energy, fire) as they are
            self._actions.copy_(actions)
        else:
            map_actions(actions, self.action_mid, self.action_half, self._actions)

    def _apply_action(self):
        self.simulator.check_prepared()
        write_actions(self.action_views, self._actions)
        self._torch_to_warp()

    def _get_observations(self):
        if self.camera is not None:
            if not getattr(self, "_camera_prepared", True):  # newton backend: after the manager's step graph capture
                self.camera.prepare()
                self._camera_prepared = True
            self.camera.render()  # Warp's stream, right after the step graph
        self._warp_to_torch()
        obs = {"policy": observation(self.obs_views, self.state_views)}  # torch.cat copies: earlier observations never mutate
        if self.task is not None:
            obs["task"] = self.task.observation()
        if self.camera is not None:  # copies: a kept observation (replay buffer, terminal frame) survives the next render
            obs["rgb"], obs["depth"] = self.camera.rgb_t.clone(), self.camera.depth_t.clone()
        return obs

    def _get_rewards(self) -> torch.Tensor:
        if self.task is not None:
            return self.task.reward()
        return self._zero_reward

    def _get_dones(self):
        if self.task is not None:
            time_out = self.task.timed_out() | (self.episode_length_buf >= self.max_episode_length)  # the host's limit holds too
            return self.task.terminated(), time_out
        time_out = self.episode_length_buf >= self.max_episode_length
        return self._never_terminated, time_out  # DirectRLEnv copies both into its own buffers

    def _reset_envs_from_buffer(self):
        """Mask-based reset of the envs flagged in ``reset_buf`` without the base class's ``nonzero()`` sync."""
        if self._ids_reset_needed or (self.render_enabled and self.sim.is_rendering and self.has_rtx_sensors
                                      and self.cfg.num_rerenders_on_reset > 0):
            return super()._reset_envs_from_buffer()
        self._save_terminal_task(mask=self.reset_buf)
        self.simulator.reset(mask=self.reset_buf)  # select_envs orders Warp after torch's writes of reset_buf
        if self.task is not None:
            self.procedure.reset(mask=self.simulator.env_mask)
            self.task.clear_recorded(mask=self.reset_buf)
        self.episode_length_buf.masked_fill_(self.reset_buf, 0)
        return None

    def _reset_idx(self, env_ids):
        """Physical reset, then Isaac Lab's reset events (which may read and randomise that state), then the
        procedure, which samples the state the events left."""
        self._save_terminal_task(env_ids=env_ids)
        reset = self.simulator.reset_physics(env_ids if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self._torch_device))  # cast and ordered in select_envs
        self._warp_to_torch()  # the events run on torch's stream, after the physical reset
        super()._reset_idx(env_ids)
        self._torch_to_warp()  # Warp (the procedure, the next step, the camera) consumes the events' writes
        if reset:
            self.simulator.scene.initialize_particle_history(self.simulator.env_mask)
        if reset and self.task is not None:
            self.procedure.reset(mask=self.simulator.env_mask)
            self.task.clear_recorded(env_ids=env_ids)

    def _save_terminal_task(self, env_ids=None, mask=None):
        if self.task is None:
            return
        self._warp_to_torch()
        # Retain terminal rows across automatic masked reset; running/manual resets do not erase them.
        self.task.save_terminal(self.extras, env_ids=env_ids, mask=mask)
        self._torch_to_warp()

    def reset(self, seed=None, options=None):
        out = super().reset(seed, options)
        if self.procedure is None and self.cfg.stagger_episodes and not self._staggered:
            self.episode_length_buf.random_(0, int(self.max_episode_length))
            self._staggered = True
        return out
