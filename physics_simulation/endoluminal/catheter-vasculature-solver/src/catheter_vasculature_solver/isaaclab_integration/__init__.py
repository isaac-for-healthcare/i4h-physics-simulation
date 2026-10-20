# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Newton Manager integration for the catheter rod.

Importing the configs and managers requires Isaac Lab with ``isaaclab_newton``
(the Newton Manager abstraction). The builder helpers in :mod:`rod_builder` need
only Newton and this package, so they stay importable without Isaac Lab.

The solver behind these managers is the in-tree
:class:`~catheter_vasculature_solver.CathRodSolver` wrapped as a Newton
``SolverBase``, so no unreleased Newton build is required.

Catheter-only scene::

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import NewtonCfg
    from catheter_vasculature_solver.isaaclab_integration import (
        NewtonXPBDRodManager,
        XPBDRodSolverCfg,
        add_catheter_rod_to_builder,
    )

    # During scene setup, before ModelBuilder.finalize():
    particle_range = add_catheter_rod_to_builder(builder, rod_config)
    NewtonXPBDRodManager.register_rod(particle_range)

    sim_cfg = SimulationCfg(
        physics=NewtonCfg(solver_cfg=XPBDRodSolverCfg(num_segments=24), num_substeps=4, use_cuda_graph=False)
    )

Coupled rigid + catheter scene: swap in
:class:`CoupledMJWarpXPBDRodSolverCfg`, which nests the same rod config.

Both rod-only and coupled managers require outer CUDA graph capture to be off.
Replay bypasses the rod's Python pending-control latch, so a graph captured
before commands arrive never consumes them::

    sim_cfg = SimulationCfg(
        physics=NewtonCfg(solver_cfg=XPBDRodSolverCfg(), use_cuda_graph=False)
    )

Leaving capture enabled raises before solver construction, including on the
pinned Isaac Lab revision that has no capture-support hook.

Per-environment reset arrives through ``SolverBase.reset``, which
:class:`~catheter_vasculature_solver.newton_rod_solver.NewtonCathRodSolver`
implements. Isaac Lab versions with the ``_reset_solver_internals`` hook call
the active manager for terminated environments automatically. On the pinned
``ffff603`` revision, the scene must call its manager's
``reset_rod(world_mask=...)`` before the next step. The coupled manager also
clears selected coupling history and calls the rigid solver's reset if available.
"""

from __future__ import annotations

from ..newton_rod_bridge import RodParticleRange
from .rod_builder import (
    add_catheter_rod_to_builder,
    initial_rod_positions,
    particle_mass_from_rod_config,
    rod_config_from_solver_cfg,
    torsion_modulus_from_rod_config,
)

__all__ = [
    "RodParticleRange",
    "add_catheter_rod_to_builder",
    "initial_rod_positions",
    "particle_mass_from_rod_config",
    "rod_config_from_solver_cfg",
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
