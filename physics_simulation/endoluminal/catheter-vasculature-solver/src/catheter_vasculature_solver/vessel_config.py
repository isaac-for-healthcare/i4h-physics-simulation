# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Vessel deformation configuration surface (scaffolding).

Modes mirror OmniEndo capabilities. Primary path (per Przemek) is
``centerline_cosserat``; cloth/tet remain opt-in.

Foundation modules are vendored from ``/omniendo`` under
``catheter_vasculature_solver.vessel_deformation`` (``pbd_cosserat``,
``centerline_tree``, ``vessel_skinning``). Live tapered-tube containment, two-way
contact and step orchestration are available via ``CenterlineVesselRuntime``
attached to ``CathRodSolver``. Cloth/tet remain gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VesselDeformationMode = Literal["off", "centerline_cosserat", "cloth", "tet"]


@dataclass
class VesselDeformationConfig:
    """Config for deformable vessel backends.

    Attributes:
        deformation: Backend selector. ``off`` = static SDF / mesh-edge only
            (current public behavior). ``centerline_cosserat`` is the primary
            path for CTA / branching. ``cloth`` / ``tet`` are secondary.
        two_way: When True, contact corrections also displace vessel DOFs
            (centerline nodes or wall particles) via ``vessel_response``.
        vessel_response: Mass-split weight for vessel side of two-way contact
            in ``[0, 1]``.
        max_delta: Optional per-step displacement clamp for vessel DOFs.
        enable_skinning: Update render surface from centerline frames
            (centerline mode only; see
            ``CenterlineVesselRuntime.attach_surface``).
        collide_during_deformable_iters: Cloth/tet only — project contacts
            inside deformable iterations, not only post-rod.
    """

    deformation: VesselDeformationMode = "off"
    two_way: bool = False
    vessel_response: float = 0.5
    max_delta: float | None = None
    enable_skinning: bool = False
    collide_during_deformable_iters: bool = False

    def require_implemented(self) -> None:
        """Raise until the selected backend is usable."""
        if self.deformation in ("off", "centerline_cosserat"):
            return
        raise NotImplementedError(
            f"vessel.deformation={self.deformation!r} is not implemented yet "
            "(cloth/tet are secondary). Use 'off' or 'centerline_cosserat' with "
            "CenterlineVesselRuntime + CathRodSolver.set_centerline_runtime(...)."
        )


@dataclass
class VesselConfig:
    """Top-level vessel settings attached to catheter simulation."""

    deformation_cfg: VesselDeformationConfig = field(default_factory=VesselDeformationConfig)

    @property
    def deformation(self) -> VesselDeformationMode:
        return self.deformation_cfg.deformation


__all__ = [
    "VesselConfig",
    "VesselDeformationConfig",
    "VesselDeformationMode",
]
