# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Newton Manager integration (Path B: coupled MJWarp + XPBD rod).

This subpackage is **optional**. Importing configs/managers requires:

* Isaac Lab ``develop`` with ``isaaclab_newton`` (Newton Manager abstraction)
* Newton build that exports ``SolverXPBDRod`` (PR #1981+)

Builder helpers in :mod:`rod_builder` only need Newton + this package.

Typical Path B wiring::

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import NewtonCfg
    from catheter_vasculature_solver.isaaclab_integration import (
        CoupledMJWarpXPBDRodSolverCfg,
        XPBDRodSolverCfg,
        add_catheter_rod_to_builder,
    )

    sim_cfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=CoupledMJWarpXPBDRodSolverCfg(
                coupling_mode="two_way",
                rod_solver_cfg=XPBDRodSolverCfg(num_segments=24),
            ),
            num_substeps=4,
        )
    )
"""

from __future__ import annotations

from .rod_builder import (
    add_catheter_rod_to_builder,
    initial_rod_positions,
    particle_mass_from_rod_config,
    register_xpbd_rod_builder_attributes,
    torsion_modulus_from_rod_config,
)

__all__ = [
    "add_catheter_rod_to_builder",
    "initial_rod_positions",
    "particle_mass_from_rod_config",
    "register_xpbd_rod_builder_attributes",
    "torsion_modulus_from_rod_config",
    "XPBDRodSolverCfg",
    "CoupledMJWarpXPBDRodSolverCfg",
    "NewtonXPBDRodManager",
    "NewtonCoupledMJWarpXPBDRodManager",
]


def __getattr__(name: str):
    """Lazy-load Isaac Lab-dependent symbols so rod_builder stays importable."""
    if name in ("XPBDRodSolverCfg", "CoupledMJWarpXPBDRodSolverCfg"):
        from .solver_cfg import CoupledMJWarpXPBDRodSolverCfg, XPBDRodSolverCfg

        return {
            "XPBDRodSolverCfg": XPBDRodSolverCfg,
            "CoupledMJWarpXPBDRodSolverCfg": CoupledMJWarpXPBDRodSolverCfg,
        }[name]
    if name == "NewtonXPBDRodManager":
        from .xpbd_rod_manager import NewtonXPBDRodManager

        return NewtonXPBDRodManager
    if name == "NewtonCoupledMJWarpXPBDRodManager":
        from .coupled_manager import NewtonCoupledMJWarpXPBDRodManager

        return NewtonCoupledMJWarpXPBDRodManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
