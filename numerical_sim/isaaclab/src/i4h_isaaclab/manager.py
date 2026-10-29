# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manager-based Isaac Lab env over an i4h case, beside ``I4hEnv`` (import only inside an Isaac Lab process).

``I4hEnv`` (direct) and ``I4hManagerBasedRLEnv`` step the same physics through the same component handle
(``i4h_isaaclab.component``), the same views (``i4h_isaaclab.views``) and the same procedure tensors
(``i4h_isaaclab.terminal``), so a policy trained on one runs on the other: the ``policy`` group is the positions then
the state products, the ``task`` group the procedure row, the action the normalised ``Box(-1, 1)`` over the control
columns (``workspace`` as ``I4hEnvCfg``), the reward the weighted completion pulses, the dones the procedure's result
and Lab's episode-length limit.

The pieces are ordinary manager terms, so a host task (a robot beside the tissue, an Arena embodiment) composes them
with its own:

- ``I4hActionTermCfg``: every control system's columns, normalised or physical;
- ``i4h_policy`` / ``i4h_task``: observation functions;
- ``procedure_reward``: reward function (Lab multiplies by ``step_dt``, the function divides by it);
- ``procedure_terminated`` / ``procedure_timeout``: termination functions;
- ``I4hHostMixin``: builds the env inside the component's construction, owns the i4h side of reset (physical state
  before Lab's reset events, the procedure after them, terminal extras before both) and detaches on close. Mix it in
  front of any ``ManagerBasedRLEnv`` subclass; the scene needs ``I4H_ROOT`` (an authored env root) so Lab replicates
  one world per env for the component's particles.

``configure_case(cfg)`` fills an ``I4hManagerEnvCfg`` from its ``case``; ``I4hManagerBasedRLEnv`` calls it at
construction, so a cfg can be edited (case, envs, decimation) after it was created, as ``I4hEnvCfg`` can.
"""

from __future__ import annotations

import math

import torch
import warp as wp
from i4h_isaaclab import views
from i4h_isaaclab.component import I4hComponent, I4hComponentCfg
from i4h_isaaclab.composition import check_dt, read_case, sim_settings
from i4h_isaaclab.env import resolve_gravity, with_gravity
from i4h_isaaclab.lab import require
from i4h_isaaclab.terminal import ProcedureTorch
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import time_out
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg

require(ManagerBasedRLEnvCfg, "compute_final_obs")

I4H_ROOT = AssetBaseCfg(prim_path="{ENV_REGEX_NS}", cloning_contexts=None)
"""An authored env root: with it in the scene, Lab replicates one Newton world per env through the public world hook,
which adds the component's particles. It spawns nothing and owns no state."""


# --- the env side: construction, reset, close -------------------------------------------------------------------


class I4hHostMixin:
    """The i4h side of a ``ManagerBasedRLEnv`` (``class Env(I4hHostMixin, ManagerBasedRLEnv)``); ``env.i4h`` is the
    component handle, ``env.i4h_task`` the procedure tensors (None without a procedure)."""

    def __init__(self, cfg, *args, **kwargs):
        from i4h_isaaclab.newton_manager import I4hSolverCfg

        solver_cfg = getattr(cfg.sim.physics, "solver_cfg", None)
        if not isinstance(solver_cfg, I4hSolverCfg):
            raise ValueError("the env's sim.physics does not step an i4h component: I4hComponent.configure(cfg.sim) first")
        self.i4h = I4hComponent.from_solver_cfg(solver_cfg)
        step_dt = float(cfg.sim.dt) * int(cfg.decimation)
        self.i4h.horizon_s = math.ceil(cfg.episode_length_s / step_dt) * step_dt  # max_episode_length * step_dt
        self.i4h_task = None
        self._i4h_views = {}
        self.i4h.on_rebuild(self._bind_i4h)
        try:
            with self.i4h.construction(device=cfg.sim.device, envs=cfg.scene.num_envs):
                super().__init__(cfg, *args, **kwargs)
        except BaseException:
            self.i4h.detach()
            if SimulationContext.instance() is not None:
                SimulationContext.clear_instance()
            raise

    def _bind_i4h(self, component: I4hComponent) -> None:
        """After every solver build: views and procedure tensors of the new binding."""
        self._i4h_views = {}
        self.i4h_task = None
        if component.procedure is not None:
            device = wp.device_to_torch(component.scene.device)
            self.i4h_task = ProcedureTorch(component.procedure, getattr(self.cfg, "step_weights", ()), device)

    def i4h_views(self, key, build):
        """A view list cached per binding (``build(sim)`` runs once per solver build)."""
        found = self._i4h_views.get(key)
        if found is None:
            found = self._i4h_views[key] = build(self.i4h.sim)
        return found

    def _reset_idx(self, env_ids):
        """Terminal extras, then the physical reset, then Lab's reset (scene, events that may read and randomise the
        new state, managers), then the procedure, which samples the state the events left."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        task = self.i4h_task
        if task is not None:
            self.i4h.sync_views()
            task.save_terminal(self.extras, env_ids=env_ids)
        self.i4h.reset(env_ids)  # Warp waits for torch's ids and extras first
        self.i4h.sync_views()  # the events run on torch's stream, after the physical reset
        super()._reset_idx(env_ids)
        if task is not None:
            self.i4h.reset_procedure(env_ids)  # select_envs orders Warp after the events' writes
            task.clear_recorded(env_ids=env_ids)
        self.i4h.sync_views()

    def close(self):
        self.i4h.detach()
        if self.i4h.sim is not None:
            self.i4h.sim.physics.invalidate()
        super().close()


class I4hManagerBasedRLEnv(I4hHostMixin, ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` over the case of an ``I4hManagerEnvCfg`` (``configure_case`` runs at construction)."""

    cfg: I4hManagerEnvCfg

    def __init__(self, cfg: I4hManagerEnvCfg, render_mode: str | None = None, **kwargs):
        configure_case(cfg)
        super().__init__(cfg, render_mode, **kwargs)


# --- actions -----------------------------------------------------------------------------------------------------


class I4hActionTerm(ActionTerm):
    """Every control system's action columns (``views.action_bounds`` order). ``normalized``: a ``[-1, 1]`` action
    mapped onto the physical bounds of ``workspace``, as ``I4hEnv``; ``physical``: rows written unclamped (teleop,
    replay). Starts the procedure's action (completion totals, terminal notification) once per policy step."""

    cfg: I4hActionTermCfg

    def __init__(self, cfg: I4hActionTermCfg, env):
        if cfg.mode not in ("normalized", "physical"):
            raise ValueError(f"I4hActionTermCfg.mode must be normalized or physical, got {cfg.mode!r}")
        super().__init__(cfg, env)
        self._bounds_for = None
        self._bind()
        self._raw = torch.zeros((env.num_envs, self._low.shape[0]), device=env.device)
        self._processed = torch.zeros_like(self._raw)

    def _bind(self) -> None:
        sim = self._env.i4h.sim
        if sim is self._bounds_for:
            return
        low, high = self._env.i4h.action_bounds(self.cfg.workspace)
        self._low = torch.as_tensor(low, device=self._env.device)
        self._high = torch.as_tensor(high, device=self._env.device)
        self._mid, self._half = (self._low + self._high) / 2.0, (self._high - self._low) / 2.0
        self._bounds_for = sim

    @property
    def action_dim(self) -> int:
        return int(self._low.shape[0])

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    @property
    def low(self) -> torch.Tensor:
        return self._low

    @property
    def high(self) -> torch.Tensor:
        return self._high

    def process_actions(self, actions: torch.Tensor) -> None:
        self._bind()  # new bounds after a hard rebuild
        self._raw.copy_(actions)
        task = self._env.i4h_task
        if task is not None:
            self._env.i4h._order("in")
            task.begin_action(self._env.extras)
        if self.cfg.mode == "physical":
            self._processed.copy_(actions)
        else:
            views.map_actions(actions, self._mid, self._half, self._processed)

    def apply_actions(self) -> None:
        self._env.i4h.write_actions(self._processed)

    def reset(self, env_ids=None) -> None:
        self._raw[slice(None) if env_ids is None else env_ids] = 0.0


@configclass
class I4hActionTermCfg(ActionTermCfg):
    class_type: type = I4hActionTerm
    asset_name: str = "i4h"  # the scene's I4H_ROOT entry
    mode: str = "normalized"  # normalized: [-1, 1] onto the bounds; physical: rows as given
    workspace: float = 0.5  # sim units, as I4hEnvCfg.workspace


# --- observations, rewards, terminations -------------------------------------------------------------------------


def i4h_policy(env, observe: str = "") -> torch.Tensor:
    """``(E, W)`` float32: positions of the integrated bodies (or ``observe = "body"`` / ``"body.part"``), then the
    systems' state products; the ``policy`` observation of ``I4hEnv``."""
    obs = env.i4h_views(("obs", observe), lambda sim: views.observation_views(sim, observe))
    state = env.i4h_views(("state",), views.state_views)
    env.i4h.sync_views()
    return views.observation(obs, state)


def i4h_task(env) -> torch.Tensor:
    """The procedure row (8 columns, plus the normalised step and elapsed time of a sequence)."""
    env.i4h.sync_views()
    return env.i4h_task.observation()


def procedure_reward(env) -> torch.Tensor:
    """The weighted completion pulses of the policy step, per second: Lab's reward manager multiplies by ``step_dt``."""
    return env.i4h_task.reward() / env.step_dt


def procedure_terminated(env) -> torch.Tensor:
    """Success or an invalid target."""
    return env.i4h_task.terminated()


def procedure_timeout(env) -> torch.Tensor:
    """The procedure's deadline (the host's horizon when that is earlier)."""
    return env.i4h_task.timed_out()


# --- the configuration -------------------------------------------------------------------------------------------


@configclass
class I4hSceneCfg(InteractiveSceneCfg):
    i4h: AssetBaseCfg = I4H_ROOT


@configclass
class I4hManagerEnvCfg(ManagerBasedRLEnvCfg):
    """The ``I4hEnvCfg`` settings for a manager-based env; ``configure_case`` derives the managers from ``case``."""

    case: str = "two_worlds"
    observe: str = ""  # "body" or "body.part" to select particles, else every integrated body
    workspace: float = 0.5  # sim units: instrument positions may be commanded within this of their initial pose
    step_weights: tuple = ()  # completion reward weights in authored order; empty defaults to ones
    action_mode: str = "normalized"  # normalized: Box(-1, 1) mapped onto the action bounds; physical: rows as given
    gravity_from_case: bool = True  # a case's sim.gravity sets sim.gravity; False: sim.gravity rules, a different one raises
    rigid_solver_cfg: object = None  # MJWarpSolverCfg: step the scene's rigid bodies beside i4h
    coupling_mode: str = "one_way"  # with rigid_solver_cfg: two_way also returns i4h's reactions to the bodies
    mounts: dict | None = None  # with rigid_solver_cfg: {instrument: {"body": label, "offset": (pos, quat_xyzw)}}
    seed: int = 0
    decimation: int = 2
    episode_length_s: float = 5.0
    scene: I4hSceneCfg = I4hSceneCfg(num_envs=2, env_spacing=1.0)
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2, device="cuda:0", physics=NewtonCfg())  # kitless
    ui_window_class_type = None
    actions: object = None  # configure_case
    observations: object = None
    rewards: object = None
    terminations: object = None


def configure_case(cfg: I4hManagerEnvCfg) -> I4hManagerEnvCfg:
    """Fill ``cfg`` from its case: gravity and dt checked as ``I4hEnv`` does, the component configured on ``cfg.sim``,
    and the action, observation, reward and termination terms. Terms the cfg already sets are kept."""
    case = read_case(cfg.case)
    case = with_gravity(case, resolve_gravity(cfg, case))
    check_dt(cfg.sim.dt, sim_settings(case)["dt"])
    procedure = bool(case.get("procedure"))
    component = I4hComponent(I4hComponentCfg.from_case(case, procedure=True, mounts=cfg.mounts,
                                                       coupling_mode=cfg.coupling_mode))
    # The host's physics before the component replaces it (again on a reused cfg): the rigid solver, if any
    graph = {"use_cuda_graph": cfg.sim.physics.use_cuda_graph} if isinstance(cfg.sim.physics, NewtonCfg) else {}
    rigid = {"solver_cfg": cfg.rigid_solver_cfg} if cfg.rigid_solver_cfg is not None else {}
    cfg.sim.physics = NewtonCfg(**rigid, **graph)
    component.configure(cfg.sim, rigid=cfg.rigid_solver_cfg is not None, gravity_from_case=cfg.gravity_from_case)

    if cfg.actions is None:
        cfg.actions = _ActionsCfg(i4h=I4hActionTermCfg(mode=cfg.action_mode, workspace=cfg.workspace))
    if cfg.observations is None:
        policy = _GroupCfg(i4h=ObservationTermCfg(func=i4h_policy, params={"observe": cfg.observe}))
        cfg.observations = _ObservationsCfg(policy=policy, task=_GroupCfg(i4h=ObservationTermCfg(func=i4h_task))
                                            if procedure else None)
    if cfg.rewards is None:
        cfg.rewards = _RewardsCfg(procedure=RewardTermCfg(func=procedure_reward, weight=1.0) if procedure else None)
    if cfg.terminations is None:
        cfg.terminations = _TerminationsCfg(
            time_out=TerminationTermCfg(func=time_out, time_out=True),
            procedure_terminated=TerminationTermCfg(func=procedure_terminated) if procedure else None,
            procedure_timeout=TerminationTermCfg(func=procedure_timeout, time_out=True) if procedure else None)
    return cfg


@configclass
class _ActionsCfg:
    i4h: I4hActionTermCfg = None


@configclass
class _GroupCfg(ObservationGroupCfg):
    i4h: ObservationTermCfg = None
    enable_corruption: bool = False
    concatenate_terms: bool = True


@configclass
class _ObservationsCfg:
    policy: _GroupCfg = None
    task: _GroupCfg | None = None


@configclass
class _RewardsCfg:
    procedure: RewardTermCfg | None = None


@configclass
class _TerminationsCfg:
    time_out: TerminationTermCfg = None
    procedure_terminated: TerminationTermCfg | None = None
    procedure_timeout: TerminationTermCfg | None = None
