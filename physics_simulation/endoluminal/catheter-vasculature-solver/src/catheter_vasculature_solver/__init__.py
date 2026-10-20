# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Solvers module for position-based dynamics simulations.

This module contains implementations of various constraint solvers,
including the Direct Position-Based Solver for Stiff Rods based on
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods".

Features:
- XPBD (Extended Position-Based Dynamics) framework
- Cosserat rod model for bending and twisting
- Separate stiffness controls (stretch, shear, bend, twist)
- Tip shaping for catheter/guidewire simulation
- BVH-accelerated mesh collision
- Friction models (Coulomb, viscous, static/dynamic)
"""

# Solver implementations (single comment block keeps isort/ruff-format stable —
# interleaving comments between these imports causes the two hooks to oscillate):
#   - cath_rod_solver: vessel mesh collision extension — catheter-in-vessel containment + track guidance
#   - newton_rod_bridge: Newton particle-buffer marshalling (no newton import; safe to export)
#   - newton_rod_solver: Newton SolverBase wrapper around CathRodSolver — imports newton, so it is
#     not exported here; import it from the module when a Newton Model/State is in play
#   - newton_xpbd_rod_wrapper: Newton interop over the vendored SolverXPBDRod (see newton_xpbd_rod/)
#   - rod_solver: kept for Isaac Lab/Newton integration (not used by standalone viewport path)
#   - xpbd_rod_solver: self-contained XPBD rod solver — no external newton dependency
from .cath_rod_solver import (
    COLLISION_PROJECTION_STAGE_POST,
    COLLISION_PROJECTION_STAGE_PRE,
    CathRodSolver,
    compute_signed_distances,
    compute_smooth_vertex_normals,
)
from .newton_rod_bridge import RodParticleBridge, RodParticleRange
from .newton_xpbd_rod_wrapper import NewtonXPBDRodSolver, orientations_xyzw_along_polyline
from .rod_data import (
    CollisionMeshConfig,
    FrictionConfig,
    RodConfig,
    RodData,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolverConfig,
    RodTipConfig,
)
from .rod_solver import RodSolver
from .vessel_config import VesselConfig, VesselDeformationConfig
from .xpbd_rod_solver import XPBDRodSolver

__all__ = [
    "RodSolver",
    "NewtonXPBDRodSolver",
    "XPBDRodSolver",
    "CathRodSolver",
    "COLLISION_PROJECTION_STAGE_POST",
    "COLLISION_PROJECTION_STAGE_PRE",
    "compute_smooth_vertex_normals",
    "compute_signed_distances",
    "orientations_xyzw_along_polyline",
    "RodParticleBridge",
    "RodParticleRange",
    "RodConfig",
    "RodData",
    "RodMaterialConfig",
    "RodGeometryConfig",
    "RodSolverConfig",
    "RodTipConfig",
    "FrictionConfig",
    "CollisionMeshConfig",
    "VesselConfig",
    "VesselDeformationConfig",
]

__version__ = "0.1.0"
