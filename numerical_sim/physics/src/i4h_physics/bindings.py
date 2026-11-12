# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Lifetime, views and episode restoration over a selected physics allocation.

A hosted binding never steps, captures, allocates host state/control, or advances
an independent clock. Standalone bindings delegate restoration to their Stepper.
"""

import warp as wp
from i4h_physics.stepper import select_envs, signature


class PhysicsBinding:
    structural = ("substeps", "iterations", "debug")

    def __init__(self, sim):
        self.valid = True
        self._standalone = sim
        self._core = sim.solver
        self._runtime = sim.solver
        self._prepare = None
        self._prepared = None
        self.procedure = self.after_step = self.observables = self.contacts = None
        self._output_arrays = []

    @classmethod
    def hosted(cls, core, runtime, prepare):
        binding = cls.__new__(cls)
        binding.valid = True
        binding._standalone = None
        binding._core, binding._runtime, binding._prepare = core, runtime, prepare
        binding._prepared = None
        binding.procedure = binding.after_step = binding.observables = binding.contacts = None
        binding._output_arrays = []
        binding.world_mask = wp.zeros(core.scene.envs + 1, dtype=wp.bool, device=core.scene.device)
        binding.env_mask = binding.world_mask[:core.scene.envs]
        return binding

    def check_valid(self):
        if not self.valid:
            raise RuntimeError("Physics binding invalidated by stop, hard rebuild, failure or close; bind again")

    def invalidate(self):
        self.valid = False
        self._standalone = self._core = self._runtime = self._prepare = None
        self.procedure = self.after_step = self.observables = self.contacts = None
        self._output_arrays = []
        self._prepared = None
        if hasattr(self, "world_mask"):
            self.world_mask = self.env_mask = None

    @property
    def physics(self):
        return self

    @property
    def solver(self):
        self.check_valid()
        return self._core

    @property
    def coupled(self):
        self.check_valid()
        return self._runtime if self._runtime is not self._core else None

    @property
    def runtime(self):
        """The solver the host steps: the coupled wrapper when there is one, otherwise ``solver``."""
        self.check_valid()
        return self._runtime

    @property
    def prepared(self) -> bool:
        """Whether a hosted preparation completed and has not been invalidated since."""
        return self._prepared is not None

    def invalidate_preparation(self):
        self._prepared = None

    def mark_prepared(self):
        """Record the structural signature that ``check_prepared`` compares against."""
        self._prepared = signature(self, self.systems)

    def register_outputs(self, entries, replace=False):
        """Add ``(owner, name, value)`` records whose replacement ``check_prepared`` rejects."""
        if replace:
            self._output_arrays = []
        self._output_arrays.extend(entries)

    @property
    def scene(self):
        return self.solver.scene

    @property
    def systems(self):
        return self.solver.systems

    @property
    def model(self):
        return self.solver.model

    @property
    def state(self):
        return self.solver.state

    @property
    def control(self):
        return self.solver.control

    @property
    def dt(self):
        return self.solver.dt

    @property
    def substeps(self):
        return self.solver.substeps

    @substeps.setter
    def substeps(self, value):
        self.solver.substeps = value

    @property
    def iterations(self):
        return self.solver.iterations

    @iterations.setter
    def iterations(self, value):
        self.solver.iterations = value

    @property
    def debug(self):
        return self.solver.debug

    @debug.setter
    def debug(self, value):
        self.solver.debug = value

    def prepare(self):
        self.check_valid()
        if self._standalone is not None:
            return self._standalone.prepare()
        self._prepare()

    def check_prepared(self):
        self.check_valid()
        self.solver.check_binding()
        for owner, name, array in self._output_arrays:
            if getattr(owner, name) is not array:
                raise RuntimeError(f"observable {name} buffer was replaced; bind a fresh runtime")
        if self._prepared is None or signature(self, self.systems) != self._prepared:
            raise RuntimeError("a structural setting changed or a live array was rebound after prepare(); "
                               "call binding.prepare() to invalidate the host graph")

    def command_views(self, system_id):
        """Cached read-only mapping of writable zero-copy Warp command slices."""
        self.solver.check_binding()
        system = next((s for s in self.systems if s.id == system_id), None)
        if system is None:
            raise KeyError(system_id)
        if system.command_view_spec is None:
            raise NotImplementedError(f"system {system_id!r} does not supply command views")
        return self.solver.command_view(system_id).views

    def reset(self, env_ids=None, *, mask=None):
        return self.reset_physics(env_ids, mask=mask)

    def initialize_particle_history(self, env_ids=None, *, mask=None):
        """Finish explicit position edits, without restoring other physical state.

        Order position writes onto Warp's stream first. This is also available
        during construction callbacks, before preparation or capture.
        """
        self.solver.check_binding()
        selection = self._standalone.env_mask if self._standalone is not None else self.env_mask
        if not select_envs(selection, env_ids, mask, debug=self.debug):
            return False
        self.scene.initialize_particle_history(selection)
        return True

    def save(self, dst_state, dst_control, env_ids=None, *, mask=None):
        """Copy the selected worlds' i4h rows into another State and Control of this Model (``I4hSolver.save_i4h``);
        every world and the shared rows when neither ``env_ids`` nor ``mask`` is given."""
        return self._continue("save_i4h", dst_state, dst_control, env_ids, mask)

    def restore(self, src_state, src_control, env_ids=None, *, mask=None):
        """Copy saved i4h rows back into the bound State and Control (``I4hSolver.restore_i4h``). No reset, cache
        refresh or preparation runs; with a rigid solver the host restores body state itself."""
        return self._continue("restore_i4h", src_state, src_control, env_ids, mask)

    def _continue(self, call, state, control, env_ids, mask):
        self.check_valid()
        runtime = self._standalone.solver if self._standalone is not None else self._runtime
        if env_ids is None and mask is None:
            getattr(runtime, call)(state, control, None)
            return True
        selection, worlds = ((self._standalone.env_mask, self._standalone.world_mask) if self._standalone is not None
                             else (self.env_mask, self.world_mask))
        if not select_envs(selection, env_ids, mask, debug=self.debug):
            return False
        getattr(runtime, call)(state, control, worlds)
        return True

    def reset_physics(self, env_ids=None, *, mask=None):
        self.check_valid()
        if self._standalone is not None:
            return self._standalone.reset_physics(env_ids, mask=mask)
        self.check_prepared()
        if not select_envs(self.env_mask, env_ids, mask, debug=self.debug):
            return False
        self._runtime.restore_episode(self.state, self.control, self.world_mask)
        return True
