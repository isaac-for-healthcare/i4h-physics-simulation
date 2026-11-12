# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-place Newton solver over the ordered i4h five-hook loop.

Construction may precede runtime binding. ``reset(flags=0)`` refreshes caches;
``restore_episode`` restores the selected worlds' physical history and commands.
Particle reset flags never imply restoration of surgical topology or history.
"""

from __future__ import annotations

import contextlib
import math

import warp as wp
from i4h_physics.checks import check_bool, check_float, check_int
from i4h_physics.debug import alias_checks
from i4h_physics.observables import KINDS, ObservableKind, Observables, OutputPublisher, check_identities, identities
from i4h_physics.scene import Scene, fill_rows
from newton import ModelFlags, StateFlags
from newton.solvers import SolverBase


class I4hSolver(SolverBase):
    ObservableKind = ObservableKind
    Observables = Observables
    SUPPORTED_OBSERVABLES = KINDS

    def __init__(self, model, scene: Scene = None, systems=None, state=None, control=None):
        super().__init__(model)
        self.scene = self.state = self.control = None
        self.systems = []
        self.dt, self.substeps, self.iterations = None, 1, 1
        self.debug = False
        self._outputs = []
        self._output_identities = []
        self.publisher = None
        if scene is not None:
            self.bind(scene, systems, state, control)

    def bind(self, scene, systems, state, control):
        if self.scene is not None:
            raise RuntimeError("I4hSolver is already bound; construct a fresh solver for replacement buffers")
        if scene.model is not self.model or scene.state is not state:
            raise ValueError("I4hSolver: model and bound state must belong to the scene")
        if self.model.world_count != scene.envs or self.model.device != scene.device:
            raise ValueError("I4hSolver: scene world count/device differs from model")
        if hasattr(scene, "control") and control is not None and control is not scene.control:
            raise ValueError("I4hSolver: foreign control")
        self.scene, self.systems, self.state = scene, list(systems), state
        self.control = control if control is not None else getattr(scene, "control", None)
        self._all = wp.ones(scene.envs + 1, dtype=wp.bool, device=scene.device)
        self._bound_arrays = [(state, name, getattr(state, name)) for name in ("particle_q", "particle_qd", "particle_f")]
        self._bound_arrays += [(self.model, name, getattr(self.model, name))
                               for name in ("particle_q", "particle_mass", "particle_inv_mass", "particle_radius", "particle_flags")]

        if self.control is not None:
            self._bound_arrays += [(self.control, name, array) for name, array in vars(self.control).items()
                                   if isinstance(array, wp.array)]
        self._bound_arrays += [(scene, name, getattr(scene, name))
                               for name in ("q", "qd", "q_prev", "q0", "inv_mass", "radius", "flags", "flags0")]

        from i4h_physics.continuation import Continuation
        self.continuation = Continuation(scene) if self.control is not None else None

        from i4h_physics.commands import CommandViews
        self._command_views = {}
        for system in self.systems:
            if system.command_view_spec is not None:
                if system.id in self._command_views:
                    raise ValueError(f"duplicate command system id {system.id!r}")
                self._command_views[system.id] = CommandViews(scene, system)

    def observables(self, kinds, *, contacts=None, requires_grad=None):
        if self.scene is None:
            raise RuntimeError("bind i4h before allocating outputs")
        if self.scene.device.is_capturing:
            raise RuntimeError("allocate observables before graph capture")
        if self.publisher is None:
            self.publisher = OutputPublisher(self)
        result = super().observables(kinds, contacts=contacts, requires_grad=requires_grad)
        self.publisher.initialize(result)
        result.identities = identities(result)
        self._outputs.append(result)
        self._output_identities.extend(identities(result))
        return result

    def validate_observables(self, observables, contacts=None):
        super().validate_observables(observables, contacts)
        if observables is not None:
            check_identities(observables.identities)

    def check_binding(self):
        """Host boundary check; capture never performs structural or buffer validation."""
        self.check_state(self.state, "check_binding")
        self.scene.storage.check()
        for commands in self._command_views.values():
            commands.check()
        check_identities(self._output_identities)
        if self.publisher is not None:
            check_identities(self.publisher.identities)
        for owner, name, array in self._bound_arrays:
            if getattr(owner, name) is not array:
                raise RuntimeError(f"bound {name} buffer was replaced; construct and bind a fresh runtime")

    def configure(self, dt: float, substeps: int, iterations: int, debug: bool = False) -> None:
        """The prepared timestep and loop counts (Stepper.prepare); a graph captured afterwards bakes them."""
        if self.scene is None:
            raise RuntimeError("I4hSolver is unbound")
        if check_float("dt", dt) <= 0.0:
            raise ValueError("dt must be positive")
        self.dt, self.debug = float(dt), check_bool("debug", debug)
        self.substeps, self.iterations = check_int("substeps", substeps, 1), check_int("iterations", iterations, 1)
        self.active = [s for s in self.systems if check_bool(f"{s.id or type(s).__name__}.enabled", s.enabled)]  # structural

    def check_state(self, state, what: str) -> None:
        if self.scene is None:
            raise RuntimeError("I4hSolver is unbound")
        if state is not self.state:
            raise ValueError(f"I4hSolver.{what}: only the bound state is supported (the scene's arrays are its arrays)")

    # SolverBase -------------------------------------------------------------------------------------------------

    def step(self, state_in, state_out, control, contacts, dt: float, observables=None) -> None:
        self.check_state(state_in, "step")
        if self.control is not None and control is not self.control:
            raise ValueError("I4hSolver.step: only the bound control is supported")
        if contacts is not None:
            raise ValueError("pure i4h does not accept external collision inputs")
        self.validate_observables(observables, contacts)
        if state_out is not state_in:
            raise ValueError("I4hSolver.step: state_out must be state_in (single-state stepping)")
        if self.dt is None:
            raise RuntimeError("I4hSolver.step before configure(); use Stepper.prepare()")
        if not math.isfinite(float(dt)) or abs(float(dt) - self.dt) > 1.0e-9 * max(1.0, self.dt):
            raise ValueError(f"I4hSolver.step: dt {dt} differs from the prepared dt {self.dt}; prepare again to change it")
        scene, dt = self.scene, self.dt
        systems = self.active
        sdt = dt / self.substeps
        with alias_checks() if self.debug else contextlib.nullcontext():
            # Frame commands and reaction accumulators.
            for s in systems:
                s.step_start(scene, dt)
            for sub in range(self.substeps):
                # Predict particle positions and rod frames.
                for s in systems:
                    s.substep_start(scene, sdt, sub, self.substeps)
                # Elasticity, contact and surgical constraints in authored order.
                for k in range(self.iterations):
                    for s in systems:
                        s.iteration(scene, sdt, k, self.iterations)
                # Finalize velocities, then surgical deletion in case order.
                for s in systems:
                    s.substep_end(scene, sdt)
            # Rebuild surfaces and publish frame measurements in case order.
            for s in systems:
                s.step_end(scene, dt)
            if observables is not None:
                self.publisher.publish(observables, dt, self.substeps)

    def reset(self, state, world_mask=None, flags=0) -> None:
        """Refresh caches against current state, optionally restoring supported particle fields."""
        self.check_state(state, "reset")
        supported = int(StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD | StateFlags.PARTICLE_F)
        flags = supported if flags is None else int(flags)
        if flags & ~supported:
            raise ValueError("pure i4h reset supports particle flags only; rigid body/joint flags are unsupported")
        mask = self._all if world_mask is None else self._normalize_reset_world_mask(world_mask)
        scene, E = self.scene, self.scene.envs
        envs = mask[:E]
        if scene.main.width and flags:
            wp.launch(reset_particle_fields, dim=(E, scene.main.width),
                      inputs=[envs, scene.block_field("q0"), flags],
                      outputs=[scene.block_field("q"), scene.block_field("qd"), scene.block_field("q_prev")],
                      device=scene.device)
            if flags & StateFlags.PARTICLE_F and state.particle_f is not None:
                wp.launch(fill_rows, dim=(E, scene.main.width), inputs=[envs, wp.vec3(0.0)],
                          outputs=[scene.main.rows(state.particle_f, E)], device=scene.device)
        for system in self.systems:
            system.refresh(self.scene, envs)

    def restore_episode(self, state, control, world_mask=None) -> None:
        """Restore authored physical state, history and command defaults for selected worlds."""
        self.check_state(state, "restore_episode")
        if self.control is not None and control is not self.control:
            raise ValueError("I4hSolver.restore_episode: only the bound control is supported")
        mask = self._all if world_mask is None else self._normalize_reset_world_mask(world_mask)
        envs = mask[:self.scene.envs]
        with alias_checks() if self.debug else contextlib.nullcontext():
            self.scene.reset(envs)
            if self.scene.main.width and state.particle_f is not None:
                wp.launch(fill_rows, dim=(self.scene.envs, self.scene.main.width), inputs=[envs, wp.vec3(0.0)],
                          outputs=[self.scene.main.rows(state.particle_f, self.scene.envs)], device=self.scene.device)
            for system in self.systems:
                system.restore_episode(self.scene, envs)
            self.reset(state, mask, flags=0)
            for output in self._outputs:
                self.publisher.clear(output, mask)

    def save_i4h(self, dst_state, dst_control, mask=None) -> None:
        """Copy this runtime's i4h-owned State and Control rows into another State and Control of the same Model.

        Copied: every declared State and Control field and the scene's own rows of ``particle_q``, ``particle_qd``
        and ``particle_f``. Never written: foreign host particles, padding rows, body and joint fields, Model fields
        and initialization templates. ``mask`` is an ``(E+1,)`` world mask whose last slot selects rows shared by
        all worlds; None selects everything. Call it at a completed step boundary, outside graph capture.

        Together with ``restore_i4h`` this is continuation: bind and prepare a runtime, restore a saved pair into
        it, and stepping continues from the saved step. It is distinct from episode reset and from the cache-only
        reset, and runs neither. Not saved: a rigid solver's own history (including MuJoCo-Warp's), host time and
        episode counters, assessment, procedure and tip evidence, renderer state and recordings.

        Side effect: both calls rebuild the contact search trees from current positions, which is what makes a
        restored runtime reproduce the saving one. A runtime that saves therefore differs at rounding level from
        one that never saved. The trees span all worlds, so with a partial mask unselected worlds keep their State
        and Control bytes but may also differ at rounding level in later steps.
        """
        self._continue(dst_state, dst_control, mask, True)

    def restore_i4h(self, src_state, src_control, mask=None) -> None:
        """Copy saved i4h-owned rows into this runtime's State and Control; see ``save_i4h``. Completed-step
        outputs of the selected worlds are cleared; system products refresh at the next step."""
        self._continue(src_state, src_control, mask, False)

    def _continue(self, state, control, world_mask, save) -> None:
        self.check_binding()
        if self.continuation is None:
            raise RuntimeError("continuation needs a bound Control")
        if self.scene.device.is_capturing:
            raise RuntimeError("save and restore run outside graph capture")
        mask = None if world_mask is None else self._normalize_reset_world_mask(world_mask)
        if mask is not None and (mask.dtype is not wp.bool or mask.shape != (self.scene.envs + 1,) or mask.device != self.scene.device):
            raise ValueError(f"continuation mask must be a ({self.scene.envs + 1},) bool array on {self.scene.device}; "
                             "the last slot selects rows shared by all worlds")
        self.continuation.copy(state, control, mask, save)
        for system in self.systems:
            system.rebuild_caches(self.scene)
        if not save:
            for output in self._outputs:
                self.publisher.clear(output, self.all_worlds if mask is None else mask)

    @property
    def outputs(self):
        """The observable containers this solver allocated, in allocation order."""
        return tuple(self._outputs)

    def command_view(self, system_id):
        """The command view record of one control system (``views``, ``torch_views()``, ``check()``)."""
        return self._command_views[system_id]

    @property
    def all_worlds(self):
        """The ``(E+1,)`` mask selecting every world and the shared rows."""
        return self._all

    def notify_model_changed(self, flags) -> None:
        """Gravity (``MODEL_PROPERTIES``) is read live from ``model.gravity``, except by a rotated component, whose
        ``R_e^T g`` copy is refreshed here. Body, joint, shape, actuator, tendon and constraint changes belong to a host's
        rigid solver (``i4h_isaaclab.coupling``) and do not concern i4h."""
        if int(flags) & int(ModelFlags.MODEL_PROPERTIES):
            self.scene.refresh_gravity()


@wp.kernel
def reset_particle_fields(mask: wp.array(dtype=wp.bool), rest: wp.array2d(dtype=wp.vec3), flags: int,
                          q: wp.array2d(dtype=wp.vec3), qd: wp.array2d(dtype=wp.vec3),
                          previous: wp.array2d(dtype=wp.vec3)):
    e, i = wp.tid()
    if mask[e]:
        if flags & int(StateFlags.PARTICLE_Q):
            q[e, i] = rest[e, i]
            previous[e, i] = rest[e, i]
        if flags & int(StateFlags.PARTICLE_QD):
            qd[e, i] = wp.vec3(0.0)
