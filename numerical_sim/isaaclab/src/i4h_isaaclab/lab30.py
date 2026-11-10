# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Narrow bridge for patched Lab 3.0's class-based Newton manager hooks.

All authoring, binding, restoration and numerical loops are shared with main.
Only this module accesses release manager class state.
"""

from i4h_isaaclab.newton_manager import context
from i4h_physics.coupling import CoupledSolver
from i4h_physics.sim import I4hSolver
from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics import NewtonCfg, NewtonManager


def solver_adapter(coupled):
    return NewtonI4hCoupledLab30Manager if coupled else NewtonI4hLab30Manager


def physics_cfg(solver_cfg, **kwargs):
    return NewtonCfg(solver_cfg=solver_cfg, num_substeps=1, **kwargs)


def attach(owner):
    owner.manager = SimulationContext.instance().physics_manager
    owner.subscriptions.append(owner.manager.register_callback(
        lambda event: _stop(owner), PhysicsEvent.STOP, order=-20))


def _stop(owner):
    owner.invalidate()
    owner.backend = None


def detach(owner):
    owner.backend = None


def contacts(owner):
    return owner.manager._contacts


def invalidate_graph(owner):
    owner.manager._invalidate_graph()


def _cfg():
    return PhysicsManager._cfg.solver_cfg


def _rigid():
    cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
    return getattr(cfg, "rigid_solver_cfg", None)


class NewtonI4hLab30Manager(NewtonManager):
    contact_sensors = False

    @classmethod
    def _register_builder_attributes(cls, builder):
        rigid = _rigid()
        if rigid is not None:
            rigid.class_type._register_builder_attributes(builder)

    @classmethod
    def _registers_builder_attributes_from_solver(cls, solver_cls):
        rigid = _rigid()
        return rigid is not None and rigid.class_type._registers_builder_attributes_from_solver(solver_cls)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder):
        rigid = _rigid()
        if rigid is not None:
            rigid.class_type._prepare_builder_for_finalize(builder)
        context(_cfg()).authoring.complete(builder)

    @classmethod
    def _build_solver(cls, model, solver_cfg):
        owner = context(solver_cfg)
        owner.backend = cls.backend
        core = I4hSolver(model)
        rigid = solver_cfg.rigid_solver_cfg
        if rigid is None:
            if model.get_requested_contact_attributes():
                raise ValueError("pure i4h does not support native force-contact sensors or CONTACT_F")
            solver = core
        else:
            solver = CoupledSolver(model, rigid.class_type._create_solver(model, rigid), core,
                                   rigid_substeps=solver_cfg.rigid_substeps)
        # Canonical backend slots required by the release's manager API.
        NewtonManager._solver = solver
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._supports_rigid_body_force_input = rigid is not None
        NewtonManager._supports_contact_sensors = cls.contact_sensors
        owner.bind(solver, model, cls.backend.state_0, cls.backend.control, cls._solver_dt,
                   outer_substeps=cls._num_substeps, collision_cfg=cls._collision_cfg)

    @classmethod
    def _initialize_contacts(cls):
        rigid = _rigid()
        if rigid is not None:
            # The rigid adapter sizes native contacts using the wrapper's delegated capacity.
            rigid.class_type._initialize_contacts.__func__(cls)
        context(_cfg()).publish()

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
        binding = context(_cfg()).binding
        contacts = binding.contacts if contacts is None else contacts
        cls._solver.step(state_0, state_1, control, contacts, substep_dt, observables=binding.observables)
        context(_cfg()).physics_step()  # outer substeps are one, even with decimation > 1

    @classmethod
    def step(cls):
        owner = context(_cfg())
        owner.check_prepared()
        super().step()
        owner.sync_views()

    @classmethod
    def _reset_solver_internals(cls, world_mask):
        if world_mask is not None:
            cls._solver.reset(cls.backend.state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _solver_specific_clear(cls):
        cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
        owner = getattr(cfg, "__i4h_context", None)
        if owner is not None:
            owner.invalidate()
            owner.backend = None


class NewtonI4hCoupledLab30Manager(NewtonI4hLab30Manager):
    contact_sensors = True

    @classmethod
    def create_fixed_tendon_control(cls, articulation):
        rigid = _rigid()
        return rigid.class_type.create_fixed_tendon_control(articulation)
