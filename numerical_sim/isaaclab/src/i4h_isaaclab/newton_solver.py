# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab main adapters. Lab owns allocation, contacts, stepping and capture."""

from copy import copy

import warp as wp
from i4h_isaaclab.newton_manager import context
from i4h_physics.coupling import CoupledSolver
from i4h_physics.observables import KINDS, identities
from i4h_physics.sim import I4hSolver
from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics import NewtonCfg, NewtonManager
from isaaclab_newton.physics.newton_solver import NewtonSolver


def solver_adapter(coupled):
    return I4hCoupledNewtonSolver if coupled else I4hNewtonSolver


def physics_cfg(solver_cfg, **kwargs):
    return NewtonCfg(class_type=I4hHostManager, solver_cfg=solver_cfg, num_substeps=1, **kwargs)


def attach(owner):
    manager = owner.manager = SimulationContext.instance().backend_manager
    owner.subscriptions.extend([
        manager.register_callback(lambda event: _ready(owner), PhysicsEvent.PHYSICS_READY, order=20),
        manager.register_callback(lambda event: _stop(owner), PhysicsEvent.STOP, order=-20),
    ])


def _ready(owner):
    owner.backend = owner.manager.backend
    owner.bind_callback = lambda: _bind_main(owner)
    owner.manager.register_solver_init_callback(owner.bind_callback)


def _bind_main(owner):
    backend = owner.backend
    owner.bind(backend.solver, backend.model, backend.state_0, backend.control, backend.solver_dt,
               outer_substeps=backend.num_substeps, collision_cfg=backend.cfg.collision_cfg)
    owner.publish()


def _stop(owner):
    owner.invalidate()
    detach(owner)


def detach(owner):
    if owner.backend is not None:
        callback = owner.bind_callback
        if callback in owner.backend.solver_init_callbacks:
            owner.backend.solver_init_callbacks.remove(callback)
    owner.backend = None


def contacts(owner):
    return owner.backend.contacts


def invalidate_graph(owner):
    if owner.backend is not None:
        owner.backend.invalidate_graph()


class I4hHostManager(NewtonManager):
    """Only host boundary validation/stream ordering; the inherited backend owns the loop.

    Main's adapter pre/post hooks run inside capture. These two host operations
    must bracket the inherited manager call, including a captured decimation batch.
    """

    def step(self):
        owner = context(self._cfg.solver_cfg)
        owner.check_prepared()
        self.backend.solver_ops.prepare_outputs(self.backend.contact_observables)
        super().step()
        owner.sync_views()


class I4hNewtonSolver(NewtonSolver):
    solver_class = I4hSolver
    single_state = True
    contact_sensors = False

    @classmethod
    def prepare_builder(cls, cfg, builder):
        context(cfg).authoring.complete(builder)

    @classmethod
    def coupling_error(cls, cfg, in_place):
        if not in_place:
            return "i4h requires in-place stepping on its bound state"
        return None

    def build(self, model, mode, env_template=None):
        context(self.cfg)
        if model.get_requested_contact_attributes():
            raise ValueError("pure i4h does not support native force-contact sensors or CONTACT_F")
        return I4hSolver(model)

    def applies_body_forces(self):
        return False

    def uses_collision_pipeline(self):
        return False

    def prepare_outputs(self, sensor_output):
        """Combine host sensor and component outputs once, before capture, using aliases."""
        output = context(self.cfg).binding.observables
        key = (output, sensor_output)
        if getattr(self, "_output_key", None) == key:
            return
        self._output_key = key
        self._sensor_output = sensor_output
        if output is None or sensor_output is None:
            self._combined_output = sensor_output if output is None else output
            return
        # Only coupled solvers support host CONTACT_F. Keep the host's rigid
        # output and the component's i4h output in one non-owning container.
        combined = copy(sensor_output)
        combined._kinds = sensor_output.kinds | output.kinds
        combined.ours = output.ours
        combined.row_keys = output.row_keys
        for kind in KINDS:
            setattr(combined, kind, getattr(output, kind))
        combined.identities = identities(combined)
        self._combined_output = combined

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        binding = context(self.cfg).binding
        contacts = binding.contacts if contacts is None else contacts
        output = binding.observables if observables is None else self._combined_output
        self.native.step(state_in, state_out, control, contacts, dt, observables=output)
        if observables is not None and binding.observables is not None and binding.observables.contact_f is not None:
            wp.copy(binding.observables.contact_f, observables.contact_f)
            wp.copy(binding.observables.rigid.contact_f, observables.contact_f)

    def post_step(self, state):
        context(self.cfg).physics_step()


class I4hCoupledNewtonSolver(I4hNewtonSolver):
    """Existing custom coupling, with native rigid contacts supplied by its rigid adapter."""

    contact_sensors = True

    @classmethod
    def coupling_error(cls, cfg, in_place):
        reason = super().coupling_error(cfg, in_place)
        if reason is not None:
            return reason
        rigid = cfg.rigid_solver_cfg
        return rigid.class_type.coupling_error(rigid, in_place=True)

    @classmethod
    def solver_classes(cls, cfg):
        rigid = cfg.rigid_solver_cfg
        return rigid.class_type.solver_classes(rigid)

    @classmethod
    def prepare_builder(cls, cfg, builder):
        rigid = cfg.rigid_solver_cfg
        rigid.class_type.prepare_builder(rigid, builder)
        super().prepare_builder(cfg, builder)

    @classmethod
    def shares_materials(cls, cfg):
        rigid = cfg.rigid_solver_cfg
        return rigid.class_type.shares_materials(rigid)

    def build(self, model, mode, env_template=None):
        context(self.cfg)
        rigid = self.cfg.rigid_solver_cfg
        self.rigid = rigid.class_type(rigid, model, mode, env_template)
        return CoupledSolver(model, self.rigid.native, I4hSolver(model),
                             rigid_substeps=self.cfg.rigid_substeps)

    def applies_body_forces(self):
        return self.rigid.applies_body_forces()

    def supports_capture(self):
        return self.rigid.supports_capture()

    def create_contacts(self):
        return self.rigid.create_contacts()

    def prepare_contacts(self, contacts, pipeline):
        self.rigid.prepare_contacts(contacts, pipeline)

    def pre_step(self, state):
        self.rigid.pre_step(state)

    def post_step(self, state):
        self.rigid.post_step(state)
        context(self.cfg).physics_step()

    def reset(self, state, world_mask):
        self.rigid.reset(state, world_mask)
        mounts = self.native.mounts
        if mounts is not None:
            # A host resets or randomizes its holders (joint state) before this reset: forward kinematics brings body_q
            # up to date and the mounts are driven from it, so the instrument reset starts from the holder's new pose.
            import newton

            newton.eval_fk(self.native.model, state.joint_q, state.joint_qd, state)
            mounts.drive(state)
        self.native.ours.reset(state, world_mask=world_mask, flags=0)

    def eval_fk(self, state, world_mask, fk_mask):
        self.rigid.eval_fk(state, world_mask, fk_mask)

    def check_step(self, captured, debug):
        self.rigid.check_step(captured, debug)

    @classmethod
    def fixed_tendon_control(cls, cfg, articulation, model):
        rigid = cfg.rigid_solver_cfg
        return rigid.class_type.fixed_tendon_control(rigid, articulation, model)
