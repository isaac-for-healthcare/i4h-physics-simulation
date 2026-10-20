# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I4hSolver: the i4h systems behind Newton's ``SolverBase`` interface.

The scene's main particle block *is* the Newton model's and state's particle arrays (``Scene.allocate``), so
``step`` runs the ordered system loop directly on the bound state: no copies in or out. The solver is restricted to
that one bound ``State``: ``step(state, state, control, None, dt)`` and ``reset(state, ...)`` with any other state raise.
Mutable model flags and the solver-owned histories (tet topology and active masks, heat, grasp, clip and hook records,
rod frames, the private fluid pools) make this a single simulation timeline: copying a Newton ``State`` does not
clone the simulation. ``control`` and ``contacts`` are accepted and not read (actions live in the control systems;
contacts are the contact systems' own). ``Stepper`` owns ``dt``, ``substeps``, ``iterations`` and the CUDA graphs and
pushes the prepared values in through ``configure``; ``step`` refuses any other ``dt``.

``reset(state, world_mask=None, flags=None)``, ``world_mask`` of shape ``(world_count + 1,)`` (``None``: every env;
the last entry is world -1, which holds no i4h entity). Every mode restores all solver-owned history of the selected
worlds: tet topology and active masks from the variant together with the particle flags from their rest values (so
restored topology never references a particle still marked inactive), heat, grasp, clip, hook and contact records,
rod frames and angular velocities, the fluid pools and their counters. The standard state fields follow ``flags``:

=====================  =====================  ==============  ==============  =================
``flags``              ``particle_q``         ``particle_qd`` ``particle_f``  ``q_prev``
=====================  =====================  ==============  ==============  =================
``None`` / ``ALL``     restored from ``q0``   zeroed          zeroed          ``= q``
``0``                  preserved              preserved       preserved       ``= current q``
``PARTICLE_Q``         restored from ``q0``   --              --              ``= q``
``PARTICLE_QD``        --                     zeroed          --              ``= current q``
``PARTICLE_F``         --                     --              zeroed          ``= current q``
=====================  =====================  ==============  ==============  =================

The bits combine; joint and body bits have no i4h fields. External particle forces are not consumed by the systems.
"""

from __future__ import annotations

import contextlib
import math

import warp as wp
from i4h_physics.checks import check_bool, check_float, check_int
from i4h_physics.debug import alias_checks
from i4h_physics.scene import Scene, fill_rows
from i4h_physics.systems.base import System
from newton import ModelFlags, StateFlags
from newton.solvers import SolverBase


class I4hSolver(SolverBase):
    def __init__(self, model, scene: Scene, systems, state):
        super().__init__(model)
        if scene.model is not model or scene.state is not state:
            raise ValueError("I4hSolver: the model and state must be the ones the scene was allocated on")
        if model.world_count != scene.envs:
            raise ValueError(f"I4hSolver: the model has {model.world_count} worlds, the scene {scene.envs} envs")
        self.scene = scene
        self.systems: list[System] = list(systems)
        self.state = state
        self.dt, self.substeps, self.iterations = None, 1, 1  # set by configure() (Stepper.prepare)
        self.debug = False
        E, dev = scene.envs, scene.device
        self._all = wp.ones(E + 1, dtype=wp.bool, device=dev)  # world_mask=None: every env (preallocated, capture-safe)

    def configure(self, dt: float, substeps: int, iterations: int, debug: bool = False) -> None:
        """The prepared timestep and loop counts (Stepper.prepare); a graph captured afterwards bakes them."""
        if check_float("dt", dt) <= 0.0:
            raise ValueError("dt must be positive")
        self.dt, self.debug = float(dt), check_bool("debug", debug)
        self.substeps, self.iterations = check_int("substeps", substeps, 1), check_int("iterations", iterations, 1)
        self.active = [s for s in self.systems if check_bool(f"{s.id or type(s).__name__}.enabled", s.enabled)]  # structural

    def _check_state(self, state, what: str) -> None:
        if state is not self.state:
            raise ValueError(f"I4hSolver.{what}: only the bound state is supported (the scene's arrays are its arrays)")

    # SolverBase -------------------------------------------------------------------------------------------------

    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        self._check_state(state_in, "step")
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
            for s in systems:
                s.step_start(scene, dt)
            for sub in range(self.substeps):
                for s in systems:
                    s.substep_start(scene, sdt, sub, self.substeps)
                for k in range(self.iterations):
                    for s in systems:
                        s.iteration(scene, sdt, k, self.iterations)
                for s in systems:
                    s.substep_end(scene, sdt)
            for s in systems:
                s.step_end(scene, dt)

    def reset(self, state, world_mask=None, flags=None) -> None:
        self._check_state(state, "reset")
        scene, E = self.scene, self.scene.envs
        mask = self._all if world_mask is None else self._normalize_reset_world_mask(world_mask)
        envs = mask[:E]
        f = int(StateFlags.ALL) if flags is None else int(flags)
        with alias_checks() if self.debug else contextlib.nullcontext():
            scene.reset(envs, positions=bool(f & StateFlags.PARTICLE_Q), velocities=bool(f & StateFlags.PARTICLE_QD))
            if f & StateFlags.PARTICLE_F and state.particle_f is not None and state.particle_f.size:
                W = scene.main.width
                wp.launch(fill_rows, dim=(E, W), inputs=[envs, wp.vec3(0.0, 0.0, 0.0)],
                          outputs=[scene.main.rows(state.particle_f, E)], device=scene.device)
            for s in self.systems:
                s.reset(scene, envs)

    def notify_model_changed(self, flags) -> None:
        """Gravity (``MODEL_PROPERTIES``) is read live from ``model.gravity``, except by a rotated component, whose
        ``R_e^T g`` copy is refreshed here. Body, joint, shape, actuator, tendon and constraint changes belong to a host's
        rigid solver (``i4h_isaaclab.coupling``) and do not concern i4h."""
        if int(flags) & int(ModelFlags.MODEL_PROPERTIES):
            self.scene.refresh_gravity()
