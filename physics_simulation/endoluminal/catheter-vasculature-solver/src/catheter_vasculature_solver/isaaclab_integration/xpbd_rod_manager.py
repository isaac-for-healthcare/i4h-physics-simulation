# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-solver NewtonManager for ``SolverXPBDRod`` (nested Path B component)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonManager

from .rod_builder import register_xpbd_rod_builder_attributes
from .solver_cfg import XPBDRodSolverCfg

if TYPE_CHECKING:
    from newton import Model


def _load_solver_xpbd_rod():
    from ..newton_xpbd_rod import SolverXPBDRod

    return SolverXPBDRod


class NewtonXPBDRodManager(NewtonManager):
    """``NewtonManager`` specialization that builds Newton's ``SolverXPBDRod``.

    Used alone (Path A / catheter-only) or as the rod half of
    :class:`~catheter_vasculature_solver.isaaclab_integration.coupled_manager.NewtonCoupledMJWarpXPBDRodManager`.
    """

    @classmethod
    def _register_builder_attributes(cls, builder) -> None:
        register_xpbd_rod_builder_attributes(builder)

    @classmethod
    def _build_solver(cls, model: "Model", solver_cfg: XPBDRodSolverCfg) -> None:
        SolverXPBDRod = _load_solver_xpbd_rod()
        NewtonManager._solver = SolverXPBDRod(
            model=model,
            linear_damping=float(solver_cfg.linear_damping),
            angular_damping=float(solver_cfg.angular_damping),
            solver_backend=str(solver_cfg.solver_backend),
            floor_z=solver_cfg.floor_z,
        )
        # SolverXPBDRod advances with swapped input/output states.
        NewtonManager._use_single_state = False
        # Vessel mesh as a Newton collider → use the shared collision pipeline.
        # Set False only if you rely exclusively on the solver's internal BVH.
        NewtonManager._needs_collision_pipeline = bool(solver_cfg.collision_enabled)

    @classmethod
    def build_nested_solver(cls, model: "Model", solver_cfg: XPBDRodSolverCfg):
        """Construct a rod solver instance without writing NewtonManager slots.

        Used by the coupled manager so rigid + rod solvers can coexist.
        """
        SolverXPBDRod = _load_solver_xpbd_rod()
        return SolverXPBDRod(
            model=model,
            linear_damping=float(solver_cfg.linear_damping),
            angular_damping=float(solver_cfg.angular_damping),
            solver_backend=str(solver_cfg.solver_backend),
            floor_z=solver_cfg.floor_z,
        )


__all__ = ["NewtonXPBDRodManager"]
