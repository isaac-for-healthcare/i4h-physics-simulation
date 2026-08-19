# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Newton solver configs for Path B (coupled rigid + catheter rod).

Requires ``isaaclab`` / ``isaaclab_newton`` at import time. Install Isaac Lab
develop and a Newton build with ``SolverXPBDRod`` before using this module.
"""

from __future__ import annotations

from typing import Literal

from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonManager, NewtonSolverCfg

# MJWarp cfg lives next to other Newton solver configs in isaaclab_newton.
try:
    from isaaclab_newton.physics import MJWarpSolverCfg
except ImportError:  # pragma: no cover - older Lab layouts
    from isaaclab_newton.physics.newton_cfg import MJWarpSolverCfg  # type: ignore


@configclass
class XPBDRodSolverCfg(NewtonSolverCfg):
    """Newton ``SolverXPBDRod`` config for Cosserat catheter / guidewire rods.

    ``class_type`` points at :class:`~catheter_vasculature_solver.isaaclab_integration.xpbd_rod_manager.NewtonXPBDRodManager`.
    Nested under :class:`CoupledMJWarpXPBDRodSolverCfg` for Path B, or used
    alone for a catheter-only (Path A) scene.
    """

    class_type: type[NewtonManager] | str = (
        "catheter_vasculature_solver.isaaclab_integration.xpbd_rod_manager:NewtonXPBDRodManager"
    )
    solver_type: str = "xpbd_rod"

    # SolverXPBDRod knobs (parity with NewtonXPBDRodSolver / Newton PR #1981)
    solver_backend: Literal["block_thomas", "split_thomas", "block_jacobi", "banded_cholesky"] = (
        "block_thomas"
    )
    linear_damping: float = 0.01
    angular_damping: float = 0.01
    floor_z: float | None = None

    # Material / geometry mirrors of RodConfig (used when spawning via helpers)
    young_modulus: float = 1.0e9
    torsion_modulus: float | None = None
    bend_stiffness: float = 0.1
    twist_stiffness: float = 0.4
    density: float = 7800.0
    radius: float = 0.002
    num_segments: int = 24
    segment_length: float = 0.02
    lock_root: bool = True
    lock_root_rotation: bool = True

    # Vessel / track policy (applied in coupled manager hooks; not SolverXPBDRod itself)
    track_enabled: bool = False
    collision_enabled: bool = True
    tip_num_edges: int = 10
    track_stiffness: float = 1.0


@configclass
class CoupledMJWarpXPBDRodSolverCfg(NewtonSolverCfg):
    """Path B: MJWarp (rigid) + SolverXPBDRod (catheter) coupled Newton solver.

    Follows the Isaac Lab coupled-manager pattern (MJWarp + VBD): nested
    solver configs plus a coupling mode; the manager owns substep order.
    """

    class_type: type[NewtonManager] | str = (
        "catheter_vasculature_solver.isaaclab_integration.coupled_manager:"
        "NewtonCoupledMJWarpXPBDRodManager"
    )
    solver_type: str = "coupled_mjwarp_xpbd_rod"

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    rod_solver_cfg: XPBDRodSolverCfg = XPBDRodSolverCfg()

    # one_way: contacts affect the rod only (tools/robots push the catheter)
    # two_way: also inject particle→body reactions into the rigid solver
    coupling_mode: Literal["one_way", "two_way"] = "one_way"

    # Soft contact material for particle–shape pairs (tune like Franka soft-lift)
    soft_contact_ke: float = 5.0e3
    soft_contact_kd: float = 1.0e2
    soft_contact_mu: float = 0.5

    # After SolverXPBDRod.step, optionally run catheter-style track / tip hooks
    # (implemented in the coupled manager when buffers are registered).
    apply_track_guidance: bool = False
    apply_vessel_containment_hooks: bool = False


__all__ = [
    "CoupledMJWarpXPBDRodSolverCfg",
    "XPBDRodSolverCfg",
]
