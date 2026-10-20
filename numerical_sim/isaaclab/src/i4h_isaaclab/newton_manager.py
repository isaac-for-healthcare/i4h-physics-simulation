# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lab owns construction, native backend storage and stepping of I4hSolver.

Use construction_context around environment construction, before scene replication.
The retained Lab builder supports hard rebuilds; callbacks rebind consumers each time.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, ClassVar

import isaaclab_newton.cloner
import numpy as np
import warp as wp
from i4h_isaaclab.composition import build_systems, check_dt, load_scene, sim_settings
from i4h_isaaclab.lab import require
from i4h_physics.checks import check_bool, check_int
from i4h_physics.sim import I4hSolver
from i4h_physics.stepper import select_envs, signature
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonManager, NewtonSolverCfg

require(NewtonManager, "_invalidate_graph")  # recapture after prepare(): checks device, capture setting, solver support
require(isaaclab_newton.cloner, "newton_builder_world_hook")


@configclass
class I4hSolverCfg(NewtonSolverCfg):
    """Select the solver and its physical case; no process-global pending configuration."""

    class_type: type | str = "i4h_isaaclab.newton_manager:I4hNewtonManager"
    solver_type: str = "i4h"
    case: dict | None = None
    debug: bool = False
    rigid_solver_cfg: object = None  # MJWarpSolverCfg: step the host's rigid bodies beside i4h (i4h_isaaclab.coupling)
    coupling_mode: str = "one_way"  # one_way: bodies drive i4h; two_way: i4h's reactions reach the bodies too
    mounts: dict | None = None  # {instrument: {"body": label, "offset": (pos, quat_xyzw)}}: host bodies hold instruments
    rigid_substeps: int = 1  # rigid solver steps per physics step (the host's num_substeps; Lab's must be 1 here)
    loaders: dict | None = None  # explicit asset loader map (with systems: no installed-domain discovery)
    systems: dict | None = None  # explicit system map {type: class}
    pose: tuple | None = None  # ((x, y, z), (qx, qy, qz, qw)): the component frame in each env (identity by default)


class NewtonSim:
    """The simulation under ``I4hNewtonManager``, with what the env, the views and the camera use of a ``Stepper``: the
    scene, the systems, ``dt``, ``prepare()``, ``step()`` and an immediate masked ``reset()``. ``dt`` is Isaac Lab's
    (read-only here); ``substeps``, ``iterations`` and ``debug`` are structural like the systems' own settings."""

    structural = ("substeps", "iterations", "debug")

    def __init__(self, scene, systems, solver, settings: dict, dt: float):
        self.procedure = None
        self.after_step = None  # I4hEnv: orders torch after Warp's step, before Isaac Lab's _get_dones/_get_rewards
        self.scene, self.systems, self.solver = scene, solver.systems, solver
        self._dt = float(dt)  # checked against the case's sim.dt by _build_solver, the only caller
        self.substeps = check_int("sim.substeps", settings["substeps"], 1)
        self.iterations = check_int("sim.iterations", settings["iterations"], 1)
        self.debug = check_bool("sim.debug", settings["debug"])
        self.world_mask = wp.zeros(scene.envs + 1, dtype=wp.bool, device=scene.device)  # [E] (world -1) stays False
        self._mask = self.world_mask[: scene.envs]
        from i4h_physics.bindings import PhysicsBinding

        self.physics = PhysicsBinding(self)
        self.reset_graph = None
        self._prepared = None
        self.coupled = None  # i4h_isaaclab.coupling.CoupledSolver when the host's rigid bodies step too

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state(self):
        return self.solver.state

    def prepare(self) -> None:
        """Validate the systems, reconfigure the solver and recapture the reset graph and the manager's step graph."""
        self._prepare(recapture_step=True)

    def _prepare(self, recapture_step: bool) -> None:
        self.reset_graph = None
        self._prepared = None
        from i4h_physics.systems.rods.material import validate_material_systems

        validate_material_systems(self.scene, self.systems)
        for s in self.systems:
            s.prepare(self.scene)
        self.solver.configure(self._dt, self.substeps, self.iterations, self.debug)
        sig = signature(self, self.systems)
        dev = self.scene.device
        cfg = NewtonManager._cfg
        if dev.is_cuda and cfg is not None and cfg.use_cuda_graph:
            with wp.ScopedDevice(dev):
                with wp.ScopedCapture(device=dev) as capture:
                    self._reset_selected()
            self.reset_graph = capture.graph
        if recapture_step:  # at build time initialize_solver/set_decimation capture it after _build_solver
            I4hNewtonManager._invalidate_graph()
        self._prepared = sig

    def check_prepared(self) -> None:
        if self._prepared is None or signature(self, self.systems) != self._prepared:
            raise RuntimeError("a structural setting changed or a live array was rebound after prepare(); "
                               "call simulator.prepare() to recapture")

    def step(self) -> None:
        """One manager call, including its configured decimation, outside the env loop."""
        I4hNewtonManager.step()

    def reset(self, env_ids=None, *, mask=None) -> None:
        """Reset physical state; the environment resets its assessment."""
        self.reset_physics(env_ids, mask=mask)

    def reset_physics(self, env_ids=None, *, mask=None) -> bool:
        """Full episode reset of the selected envs, now (every env when neither is given): ``env_ids`` or an ``(E,)`` bool
        ``mask`` as ``Stepper.reset`` takes them (device masks and ids need no host sync), replaying the captured graph."""
        self.check_prepared()
        if not select_envs(self._mask, env_ids, mask, debug=self.debug):
            return False
        if self.reset_graph is not None:
            wp.capture_launch(self.reset_graph)
        else:
            self._reset_selected()
        return True

    def _reset_selected(self) -> None:
        """The selected worlds' i4h state, and the coupling's reactions still pending for them (a new episode must not
        push its holder with the last one's contact). The rigid solver's internals are Lab's (its lazy reset)."""
        self.solver.reset(self.state, self.world_mask)
        if self.coupled is not None and self.coupled.mounts is not None:
            self.coupled.mounts.reset(self.world_mask)


class I4hNewtonManager(NewtonManager):
    simulator: ClassVar[NewtonSim | None] = None  # the current solver's simulation (rebuilt by a hard reset)
    on_solver: ClassVar[list[Callable[[NewtonSim], None]]] = []  # run after every solver build (the env binds its views)
    pending_scene: ClassVar[tuple | None] = None  # (case, unallocated scene) from construction_context, used once
    world_record: ClassVar[tuple | None] = None  # (world_starts, env_xforms) the hook recorded; a rebuild reuses them

    @classmethod
    def _build_solver(cls, model, solver_cfg) -> None:
        if I4hNewtonManager.simulator is not None:
            I4hNewtonManager.simulator.physics.invalidate()
            I4hNewtonManager.simulator = None
        if solver_cfg.case is None:
            raise RuntimeError("I4hSolverCfg.case must contain the physical case")
        if int(cls._num_substeps) != 1:
            raise ValueError("I4hSolver runs the case's substeps itself: NewtonCfg.num_substeps must be 1")
        cfg = solver_cfg.case
        settings = sim_settings(cfg, solver_cfg.debug)
        dt = float(cls._solver_dt)  # what _run_solver_substeps passes to I4hSolver.step
        check_dt(dt, settings["dt"])
        scene = cls._take_pending_scene(cfg, model, solver_cfg)
        scene.allocate(model, NewtonManager.get_state_0())
        systems = build_systems(scene, cfg, **registries(solver_cfg))
        solver = I4hSolver(model, scene, systems, NewtonManager.get_state_0())
        sim = NewtonSim(scene, systems, solver, settings, dt)
        rigid_cfg = getattr(solver_cfg, "rigid_solver_cfg", None)
        mounts = getattr(solver_cfg, "mounts", None)
        if mounts and rigid_cfg is None:
            raise ValueError("I4hSolverCfg.mounts needs rigid_solver_cfg: the mount bodies must be stepped")
        if rigid_cfg is not None:  # the host's bodies step too: MuJoCo-Warp in place, then i4h (i4h_isaaclab.coupling)
            from i4h_isaaclab.coupling import CoupledSolver, Mounts, check_rigid_solver_cfg

            check_rigid_solver_cfg(rigid_cfg, solver_cfg.coupling_mode)
            held = Mounts(model, scene, systems, mounts, solver_cfg.coupling_mode == "two_way") if mounts else None
            sim.coupled = CoupledSolver(model, rigid_cfg.class_type._create_solver(model, rigid_cfg), solver, held,
                                        getattr(solver_cfg, "rigid_substeps", 1))
            NewtonManager._solver = sim.coupled
            NewtonManager._supports_rigid_body_force_input = True
        else:
            NewtonManager._solver = solver
            NewtonManager._supports_rigid_body_force_input = False
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False  # i4h's contacts are its own; MuJoCo-Warp uses MuJoCo's
        I4hNewtonManager.simulator = sim
        try:
            for callback in list(cls.on_solver):
                callback(sim)
            sim._prepare(recapture_step=False)
        except BaseException:
            sim.physics.invalidate()
            I4hNewtonManager.simulator = None
            raise

    @classmethod
    def step(cls) -> None:
        sim = cls.simulator
        if sim is not None:
            sim.check_prepared()
        super().step()
        if sim is not None and sim.procedure is not None:
            sim.procedure.update(sim.dt)
        if sim is not None and sim.after_step is not None:
            sim.after_step()

    @classmethod
    def set_decimation(cls, decimation: int) -> None:
        enabled = cls._cfg is not None and bool((cls._cfg.solver_cfg.case or {}).get('procedure'))
        super().set_decimation(1 if enabled else decimation)

    @classmethod
    def handles_decimation(cls) -> bool:
        if cls._cfg is not None and (cls._cfg.solver_cfg.case or {}).get('procedure'):
            return False
        return super().handles_decimation()

    @classmethod
    def _reset_solver_internals(cls, world_mask) -> None:
        """The rigid solver's internals only (coupled): Isaac Lab applies this lazily at the next ``step()``/``forward()``.
        i4h physics is not reset here: direct envs and workflow reset events reset it immediately through the simulation
        owner, after the reset envs' observations were computed."""
        sim = cls.simulator
        if sim is not None and getattr(sim, "coupled", None) is not None and world_mask is not None:
            sim.coupled.reset(cls.backend.state_0, world_mask=world_mask)

    @classmethod
    def reset_worlds(cls, env_ids=None, *, mask=None) -> None:
        """``simulator.reset``: a full episode reset of the selected envs, now."""
        if cls.simulator is None:
            raise RuntimeError("I4hNewtonManager.reset_worlds before the solver was built (sim.reset())")
        cls.simulator.reset(env_ids, mask=mask)

    @classmethod
    def _take_pending_scene(cls, cfg, model, solver_cfg=None):
        """The scene construction_context loaded for this case (one load per construction), else a fresh one (a hard
        reset rebuilds outside the context; an allocated scene is never reused)."""
        pending, I4hNewtonManager.pending_scene = I4hNewtonManager.pending_scene, None
        if pending is None or pending[0] is not cfg:
            scene = load_scene(cfg, device=model.device, envs=model.world_count, **registries(solver_cfg))
            if I4hNewtonManager.world_record is not None:  # the retained builder still holds the hook's layout
                scene.world_starts, scene.env_xforms = (list(r) for r in I4hNewtonManager.world_record)
            return scene
        scene = pending[1]
        if scene.world_starts is not None:
            I4hNewtonManager.world_record = (list(scene.world_starts), list(scene.env_xforms))
        if scene.envs != model.world_count or scene.device != model.device:
            raise RuntimeError(f"the scene built for the replication ({scene.envs} envs on {scene.device}) does not match "
                               f"the finalized model ({model.world_count} worlds on {model.device})")
        return scene

    @classmethod
    def _solver_specific_clear(cls) -> None:
        if I4hNewtonManager.simulator is not None:
            I4hNewtonManager.simulator.physics.invalidate()
        I4hNewtonManager.simulator = None
        I4hNewtonManager.on_solver = []
        I4hNewtonManager.pending_scene = None
        I4hNewtonManager.world_record = None


def registries(solver_cfg) -> dict:
    """The explicit loader and system maps of a solver cfg, as ``composition`` keywords."""
    return {"loaders": getattr(solver_cfg, "loaders", None), "systems": getattr(solver_cfg, "systems", None)}


def place(position, quaternion, pose):
    """The component frame in the world: Lab's env frame (``position``, ``quaternion`` xyzw) composed with ``pose``."""
    if pose is None:
        return position, quaternion
    env = wp.transform(wp.vec3(*map(float, position)), wp.quat(*map(float, quaternion)))
    world = env * wp.transform(wp.vec3(*map(float, pose[0])), wp.normalize(wp.quat(*map(float, pose[1]))))
    return np.float32(wp.transform_get_translation(world)), np.float32(wp.transform_get_rotation(world))


@contextmanager
def construction_context(solver_cfg, *, device, envs, on_solver=None):
    """Add environment-local particles during Lab replication and retain a rebuild callback.

    The hook is removed after construction. A successful callback registration lives
    until manager close; a failed construction removes only this context's callback.
    """
    from i4h_physics.newton_model import add_env_particles
    from isaaclab_newton.cloner import newton_builder_world_hook

    if solver_cfg.case is None:
        raise ValueError("I4hSolverCfg.case must contain the physical case")
    scene = load_scene(solver_cfg.case, device=device, envs=envs, **registries(solver_cfg))
    scene.check_memory(scene.estimate_bytes(construction=True))
    pose = getattr(solver_cfg, "pose", None)

    def add_world(builder, env, position, quaternion):
        add_env_particles(builder, scene, env, xform=place(position, quaternion, pose))

    if on_solver is not None:
        I4hNewtonManager.on_solver.append(on_solver)
    I4hNewtonManager.pending_scene = (solver_cfg.case, scene)  # _build_solver, inside this context, takes it
    try:
        with newton_builder_world_hook(add_world):
            yield
    except BaseException:
        if on_solver in I4hNewtonManager.on_solver:
            I4hNewtonManager.on_solver.remove(on_solver)
        raise
    finally:
        I4hNewtonManager.pending_scene = None
