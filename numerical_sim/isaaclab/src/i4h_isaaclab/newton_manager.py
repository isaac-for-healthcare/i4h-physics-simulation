# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Component-owned authoring and binding; release-specific host APIs live in adapters."""

from contextlib import contextmanager
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import warp as wp
from i4h_isaaclab.composition import check_dt, maps, sim_settings
from i4h_physics import PhysicsBinding, author_case
from i4h_physics.coupling import CoupledSolver, Mounts
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonSolverCfg


def host_api():
    name = "newton_solver" if find_spec("isaaclab_newton.physics.newton_solver") is not None else "lab30"
    return import_module(f"i4h_isaaclab.{name}")


@configclass
class I4hSolverCfg(NewtonSolverCfg):
    class_type: type | str = "i4h_isaaclab.newton_solver:I4hNewtonSolver"
    solver_type: str = "i4h"
    case: dict | None = None
    debug: bool = False
    rigid_solver_cfg: object = None
    coupling_mode: str = "one_way"
    mounts: dict | None = None
    rigid_substeps: int = 1
    loaders: dict | None = None
    systems: dict | None = None
    pose: tuple | None = None
    observables: tuple = ()  # requested Newton observable kinds, allocated once before host capture

    def __post_init__(self):
        if self.rigid_solver_cfg is not None:
            from i4h_isaaclab.coupling import check_rigid_solver_cfg
            check_rigid_solver_cfg(self.rigid_solver_cfg, self.coupling_mode)
        elif self.mounts:
            raise ValueError("mounts require rigid_solver_cfg")
        self.class_type = host_api().solver_adapter(self.rigid_solver_cfg is not None)


def physics_cfg(solver_cfg, **kwargs):
    """Select the release's host manager without changing any adapter's capabilities."""
    return host_api().physics_cfg(solver_cfg, **kwargs)


def context(cfg):
    owner = getattr(cfg, "__i4h_context", None)
    if owner is None or owner.closed:
        raise RuntimeError("construct the host inside the component's construction context")
    return owner


def place(position, quaternion, pose):
    if pose is None:
        return position, quaternion
    env = wp.transform(wp.vec3(*map(float, position)), wp.quat(*map(float, quaternion)))
    world = env * wp.transform(wp.vec3(*map(float, pose[0])), wp.normalize(wp.quat(*map(float, pose[1]))))
    return np.float32(wp.transform_get_translation(world)), np.float32(wp.transform_get_rotation(world))


class ConstructionContext:
    """One component's immutable authoring, lifecycle subscriptions and current binding."""

    def __init__(self, cfg, device, envs, on_solver):
        if cfg.case is None:
            raise ValueError("I4hSolverCfg.case must contain the physical case")
        self.cfg, self.device, self.envs = cfg, device, envs
        self.on_solver = on_solver
        self.authoring = self._author()
        self.binding = self.manager = self.backend = None
        self.subscriptions = []
        self.closed = False
        self.api = host_api()
        # Explicit component-owned reference. Config serialization skips double-underscore fields.
        setattr(cfg, "__i4h_context", self)

    def __deepcopy__(self, memo):
        # Lab copies declarative configs; runtime ownership stays with this component.
        return self

    def _author(self):
        cfg, loaders, systems = maps(self.cfg.case, self.cfg.loaders, self.cfg.systems)
        result = author_case(cfg, loaders, systems, device=self.device, envs=self.envs,
                             mounts=self.cfg.mounts, coupling_mode=self.cfg.coupling_mode)
        result.scene.check_memory(result.scene.estimate_bytes(construction=True))
        return result

    def add_world(self, builder, world, position, quaternion):
        if self.closed:
            raise RuntimeError("construction context is closed")
        if self.manager is None:
            self.api.attach(self)
        if self.authoring.builder is not None and self.authoring.builder is not builder:
            self.authoring = self._author()
        self.authoring.add_world(builder, world, place(position, quaternion, self.cfg.pose))

    def bind(self, solver, model, state, control, dt, *, outer_substeps, collision_cfg):
        self.check_coupling_config()
        self.invalidate()
        if outer_substeps != 1:
            raise ValueError("i4h owns its internal substeps: NewtonCfg.num_substeps must be 1")
        if collision_cfg is not None:
            raise ValueError("i4h adapters do not support Newton collision_cfg; coupled rigid contacts belong to MuJoCo")
        settings = sim_settings(self.cfg.case, self.cfg.debug)
        check_dt(dt, settings["dt"])
        core = solver.ours if isinstance(solver, CoupledSolver) else solver
        try:
            # Both adapters bypass Newton's particle collision pipeline. Release
            # its unused grid before binding builds i4h's own meshes and BVHs,
            # matching standalone allocation. q0/inv_mass remain i4h-owned.
            model.particle_grid = None
            scene, systems = self.authoring.bind(model, state, control)
            core.bind(scene, systems, state, control)
            core.configure(dt, settings["substeps"], settings["iterations"], settings["debug"])
            if isinstance(solver, CoupledSolver) and self.cfg.mounts:
                solver.mounts = Mounts(model, scene, systems, self.cfg.mounts, self.cfg.coupling_mode == "two_way")
            self.binding = PhysicsBinding.hosted(core, solver, self.prepare)
        except BaseException:
            self.invalidate()
            raise

    def publish(self):
        """Native contacts exist; bind consumers and prepare before the host captures."""
        try:
            binding = self.binding
            binding.contacts = self.api.contacts(self)
            self._output_kinds = tuple(self.cfg.observables)
            if self.cfg.observables:
                binding.observables = binding.runtime.observables(self.cfg.observables, contacts=binding.contacts)
                containers = [binding.observables, getattr(binding.observables, "rigid", None),
                              getattr(binding.observables, "ours", None)]
                binding.register_outputs([(owner, name, value) for owner in containers if owner is not None
                                          for name, value in vars(owner).items() if isinstance(value, wp.array)], replace=True)
                binding.register_outputs((binding.observables, name, getattr(binding.observables, name))
                                         for name in ("rigid", "ours") if hasattr(binding.observables, name))
            binding.register_outputs([(binding, "observables", binding.observables)])
            if self.on_solver is not None:
                self.on_solver(self.binding)
            self.prepare()
        except BaseException:
            self.invalidate()
            raise

    def check_coupling_config(self):
        from i4h_physics.mount_storage import configuration
        config = configuration(self.cfg.mounts, self.cfg.coupling_mode == "two_way"), self.cfg.coupling_mode
        if config != self.authoring.mount_config:
            raise RuntimeError("mount configuration changed; reconstruct the model")

    def prepare(self):
        self.check_coupling_config()
        binding = self.binding
        binding.check_valid()
        binding.invalidate_preparation()
        self.api.invalidate_graph(self)
        from i4h_physics.systems.rods.material import validate_material_systems
        binding.solver.check_binding()
        validate_material_systems(binding.scene, binding.systems)
        for system in binding.systems:
            system.prepare(binding.scene)
        binding.solver.configure(binding.dt, binding.substeps, binding.iterations, binding.debug)
        binding.mark_prepared()

    def physics_step(self):
        """Device-resident work, called exactly once per physics step inside the host graph."""
        if self.binding.procedure is not None:
            self.binding.procedure.update(self.binding.dt)

    def check_prepared(self):
        self.check_coupling_config()
        if self.binding is None:
            raise RuntimeError("i4h runtime is unbound")
        self.binding.check_prepared()
        if tuple(self.cfg.observables) != self._output_kinds:
            raise RuntimeError("requested observables changed; bind a fresh runtime before capture")
        if self.api.contacts(self) is not self.binding.contacts:
            raise RuntimeError("host contacts were replaced; bind a fresh runtime before capture")

    def sync_views(self):
        """Host stream boundary, outside graph capture and decimation loops."""
        if self.binding is not None and self.binding.after_step is not None:
            self.binding.after_step()

    def invalidate(self, event=None):
        if self.binding is not None:
            self.binding.invalidate()
            self.binding = None

    def close(self):
        self.invalidate()
        for handle in self.subscriptions:
            handle.deregister()
        self.subscriptions.clear()
        self.api.detach(self)
        self.on_solver = self.manager = self.backend = self.authoring = None
        self.closed = True


@contextmanager
def construction_context(solver_cfg, *, device, envs, on_solver=None):
    from isaaclab_newton.cloner import newton_builder_world_hook
    owner = ConstructionContext(solver_cfg, device, envs, on_solver)
    try:
        with newton_builder_world_hook(owner.add_world):
            yield owner
    except BaseException:
        owner.close()
        raise
