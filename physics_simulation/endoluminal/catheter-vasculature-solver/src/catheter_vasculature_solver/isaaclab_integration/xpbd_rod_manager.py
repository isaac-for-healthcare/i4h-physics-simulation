# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""``NewtonManager`` that drives the catheter rod solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from isaaclab_newton.physics import NewtonManager

from ..newton_rod_bridge import RodParticleRange
from .rod_builder import rod_config_from_solver_cfg
from .solver_cfg import XPBDRodSolverCfg

if TYPE_CHECKING:
    from newton import Model

    from ..xpbd_rod_solver import XPBDRodSolver


class NewtonXPBDRodManager(NewtonManager):
    """Newton manager for an XPBD Cosserat catheter rod.

    Implements the Isaac Lab solver-manager contract: build the solver in
    :meth:`_build_solver` and populate ``_solver``, ``_use_single_state`` and
    ``_needs_collision_pipeline``. The solver built here is
    :class:`~catheter_vasculature_solver.newton_rod_solver.NewtonCathRodSolver`,
    which wraps the in-tree rod solver behind Newton's ``SolverBase``.

    The rod's particles must already exist on the model, so scene setup runs
    :func:`~catheter_vasculature_solver.isaaclab_integration.rod_builder.add_catheter_rod_to_builder`
    before finalize and passes the resulting range to :meth:`register_rod`::

        particle_range = add_catheter_rod_to_builder(builder, rod_config)
        NewtonXPBDRodManager.register_rod(particle_range)

    Vessel meshes, insertion tracks and centerline vessel runtimes are GPU
    buffers that cannot live in a config, so scenes needing them build the rod
    solver themselves and hand it over::

        NewtonXPBDRodManager.register_rod(particle_range, rod=CathRodSolver(...))
    """

    _rod_particle_range: RodParticleRange | None = None
    _rod: Any = None
    _rod_factory: Callable[..., Any] | None = None

    @classmethod
    def register_rod(
        cls,
        particle_range: RodParticleRange,
        *,
        rod: "XPBDRodSolver | None" = None,
        rod_factory: Callable[["Model", XPBDRodSolverCfg], "XPBDRodSolver"] | None = None,
    ) -> None:
        """Declare which particles the rod owns, and optionally how to build it.

        Args:
            particle_range: Range returned by ``add_catheter_rod_to_builder``.
            rod: Pre-built rod solver. Use this when the rod needs a vessel
                mesh, insertion track or centerline runtime. Its device must
                match the Newton model's.
            rod_factory: Called as ``factory(model, solver_cfg)`` at solver-build
                time when the rod depends on the finalized model. Ignored if
                ``rod`` is given.
        """
        if rod is not None and rod_factory is not None:
            raise ValueError("pass either rod or rod_factory, not both")
        cls._rod_particle_range = particle_range
        cls._rod = rod
        cls._rod_factory = rod_factory

    @classmethod
    def _resolve_rod(cls, model: "Model", solver_cfg: XPBDRodSolverCfg) -> "XPBDRodSolver":
        if cls._rod is not None:
            return cls._rod
        if cls._rod_factory is not None:
            return cls._rod_factory(model, solver_cfg)

        if solver_cfg.track_enabled or solver_cfg.collision_enabled:
            raise ValueError(
                "Track guidance and collision require geometry. Supply a configured rod "
                "through register_rod(..., rod=...) or register_rod(..., rod_factory=...)."
            )

        from ..xpbd_rod_solver import XPBDRodSolver

        particle_range = cls._require_particle_range()
        config = rod_config_from_solver_cfg(solver_cfg, device=str(model.device))
        return XPBDRodSolver(
            config,
            num_envs=particle_range.num_envs,
            floor_z=solver_cfg.floor_z,
            solver_backend=str(solver_cfg.solver_backend),
        )

    @classmethod
    def _require_particle_range(cls) -> RodParticleRange:
        if cls._rod_particle_range is None:
            raise RuntimeError(
                "No rod particle range registered. Call add_catheter_rod_to_builder() during "
                "scene setup and pass its return value to NewtonXPBDRodManager.register_rod()."
            )
        return cls._rod_particle_range

    @classmethod
    def _build_solver(cls, model: "Model", solver_cfg: XPBDRodSolverCfg) -> None:
        cls._reject_cuda_graph_capture(getattr(cls, "_cfg", None))
        NewtonManager._solver = cls.build_nested_solver(model, solver_cfg)
        # Rod dynamics read one state and write another.
        NewtonManager._use_single_state = False
        # Vessel containment runs inside the rod solver's own kernels, which
        # ignore Newton's contact buffers, so the shared pipeline is dead weight
        # here. The coupled manager turns it on for rigid-body contact.
        NewtonManager._needs_collision_pipeline = False

    @classmethod
    def build_nested_solver(cls, model: "Model", solver_cfg: XPBDRodSolverCfg):
        """Construct the Newton-facing rod solver without writing manager slots.

        Used by :class:`~catheter_vasculature_solver.isaaclab_integration.coupled_manager.NewtonCoupledMJWarpXPBDRodManager`
        so rigid and rod solvers can coexist.
        """
        cls._reject_cuda_graph_capture(getattr(cls, "_cfg", None))

        from ..newton_rod_solver import NewtonCathRodSolver

        particle_range = cls._require_particle_range()
        rod = cls._resolve_rod(model, solver_cfg)
        return NewtonCathRodSolver(
            model,
            rod,
            particle_range,
            sync_from_state=bool(solver_cfg.sync_from_state),
            sync_velocities=bool(solver_cfg.sync_velocities),
        )

    @staticmethod
    def _reject_cuda_graph_capture(cfg) -> None:
        """Reject outer capture until pending commands use persistent device state.

        Graph replay bypasses the Python control latch in ``rod.step``. This
        affects bare rods as well as rods carrying deformable vessels, whose
        host-side tunables would also be frozen by capture.
        """
        if not getattr(cfg, "use_cuda_graph", False):
            return
        raise ValueError(
            "Catheter rod managers cannot use CUDA graph capture: replay bypasses the "
            "Python pending-control latch and ignores new commands. Set "
            "NewtonCfg(use_cuda_graph=False) on the simulation config."
        )

    @classmethod
    def reset_rod(cls, world_mask=None) -> None:
        """Reset selected rod environments in the current Newton state.

        ``world_mask`` has one entry per environment; ``None`` resets all.
        Call this from the scene's reset path on Isaac Lab ffff603, which has
        no solver-reset hook. Newer Lab revisions call the hook below instead.
        """
        solver = getattr(NewtonManager, "_solver", None)
        if solver is None:
            raise RuntimeError("The rod solver must be built before reset_rod().")
        state = cls.get_state_0()
        if state is None:
            raise RuntimeError("Newton state must be initialized before reset_rod().")
        solver.reset(state, world_mask=world_mask, flags=0)

    @classmethod
    def _reset_solver_internals(cls, world_mask) -> None:
        """Handle Isaac Lab's reset boundary when that lifecycle hook exists."""
        if world_mask is not None:
            cls.reset_rod(world_mask)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._rod = None
        cls._rod_factory = None
        cls._rod_particle_range = None


__all__ = ["NewtonXPBDRodManager"]
