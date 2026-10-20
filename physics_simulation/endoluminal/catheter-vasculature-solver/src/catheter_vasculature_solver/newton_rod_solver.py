# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Newton ``SolverBase`` implementation backed by the in-tree XPBD rod solver.

This is what lets the catheter participate in a Newton scene — and therefore in
Isaac Lab, whose ``NewtonManager`` drives a ``SolverBase`` and nothing else. It
wraps :class:`~catheter_vasculature_solver.CathRodSolver` rather than Newton's
``SolverXPBDRod``, which ships in no installable ``newton`` release.

Everything catheter-specific — track guidance, vessel containment, centerline
vessel deformation — keeps running inside the wrapped solver, so the Newton
integration adds coupling without giving up behaviour.

Only ``newton.Model``, ``newton.State`` and ``newton.solvers.SolverBase`` are
required, all of which released Newton provides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton.solvers import SolverBase

from .newton_rod_bridge import RodParticleBridge, RodParticleRange

if TYPE_CHECKING:
    from newton import Contacts, Control, Model, State

    from .xpbd_rod_solver import XPBDRodSolver


class NewtonCathRodSolver(SolverBase):
    """Drive the catheter rod solver through Newton's solver interface.

    Newton's contract is a single :meth:`step` that reads one
    :class:`~newton.State` and writes another. Each call mirrors the incoming
    particle positions into the rod solver, advances the rod, and mirrors the
    result back out, leaving the rest of the Newton scene untouched.

    Substeps compose multiplicatively: an outer runtime that calls :meth:`step`
    ``n`` times per frame while the rod is configured for ``m`` substeps runs
    ``n * m`` rod substeps. Set ``RodConfig.solver.num_substeps = 1`` when the
    outer runtime already substeps, or accept the extra accuracy and cost.

    Args:
        model: Finalized Newton model whose particle arrays include the rod.
        rod: Rod solver to drive.
        particle_range: Index window the rod owns in Newton's particle arrays,
            as returned by
            :func:`~catheter_vasculature_solver.isaaclab_integration.rod_builder.add_catheter_rod_to_builder`.
        sync_velocities: Mirror ``particle_qd`` alongside ``particle_q``.
        sync_from_state: Read positions back from the incoming state before
            stepping, so resets and other solvers' corrections are respected.
        allow_internal_cuda_graph: Leave the rod solver's own CUDA graph capture
            enabled. Defaults to ``False`` to avoid nested capture in external
            runtimes. Isaac Lab managers also require outer capture disabled
            because replay cannot consume the rod's Python command latch.
            Only enable internal capture when nothing outside captures.
    """

    def __init__(
        self,
        model: "Model",
        rod: "XPBDRodSolver",
        particle_range: RodParticleRange,
        *,
        sync_velocities: bool = True,
        sync_from_state: bool = True,
        allow_internal_cuda_graph: bool = False,
    ) -> None:
        super().__init__(model)

        model_device = str(wp.get_device(getattr(model, "device", None)))
        rod_device = str(wp.get_device(rod.device))
        if model_device != rod_device:
            raise ValueError(
                f"rod solver device ({rod_device}) must match the Newton model device "
                f"({model_device}); construct RodConfig with device={model_device!r}"
            )

        self._bridge = RodParticleBridge(
            rod,
            particle_range,
            sync_velocities=sync_velocities,
            sync_from_state=sync_from_state,
        )
        if not allow_internal_cuda_graph:
            rod.set_cuda_graph_enabled(False)

    @property
    def rod(self) -> "XPBDRodSolver":
        """The wrapped rod solver, for track/vessel configuration and readback."""
        return self._bridge.rod

    @property
    def particle_range(self) -> RodParticleRange:
        """Index window this solver owns in Newton's particle arrays."""
        return self._bridge.particle_range

    def step(
        self,
        state_in: "State",
        state_out: "State",
        control: "Control",
        contacts: "Contacts | None",
        dt: float,
    ) -> None:
        """Advance the rod by ``dt``, reading ``state_in`` and writing ``state_out``.

        ``control`` and ``contacts`` are unused: rod actuation goes through the
        rod solver's own proximal-control API, and vessel contact is resolved by
        its containment kernels rather than Newton's contact buffers.
        """
        del control, contacts
        self._bridge.read_from(state_in)
        self._bridge.advance(dt)
        self._bridge.write_to(state_out)

    def reset(self, state: "State", world_mask=None, flags: int = 0) -> None:
        """Restore the envs selected by ``world_mask`` to rest, vessels included.

        Signature matches what Isaac Lab's ``NewtonManager`` calls on a
        ``SolverBase`` from its ``_reset_solver_internals`` hook, so per-env
        reset engages by itself on a stack that has that hook. On older Lab
        revisions, scenes call their manager's ``reset_rod(world_mask=...)``
        explicitly so a coupled scene also clears the selected feedback history.
        Direct Newton users can call this solver method with their current state.

        Args:
            state: State to republish restored positions into.
            world_mask: Per-world reset mask, one entry per env, non-zero to
                reset. ``None`` resets every env.
            flags: Accepted for signature compatibility and unused. The flags
                select which solver-owned buffers survive a reset, and the rod
                owns no joint state to preserve: a reset env returns to its
                construction-time rest pose outright.
        """
        del flags
        self._bridge.reset(state, world_mask)


__all__ = ["NewtonCathRodSolver"]
