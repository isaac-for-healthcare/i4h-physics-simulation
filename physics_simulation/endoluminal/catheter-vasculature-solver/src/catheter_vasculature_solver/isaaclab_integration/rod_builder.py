# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Path B helper: register SolverXPBDRod attributes and add catheter rods
# onto a Newton ModelBuilder owned by Isaac Lab's NewtonManager.

"""Map :class:`~catheter_vasculature_solver.RodConfig` onto Newton builder APIs.

Under Isaac Lab, :class:`~isaaclab_newton.physics.NewtonManager` owns
``ModelBuilder.finalize()``. Rods must be registered on that builder
*before* finalize — typically from an asset spawn / scene setup hook —
while the manager only constructs ``SolverXPBDRod`` from the finalized model.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..rod_data import RodConfig

if TYPE_CHECKING:
    pass


def _load_newton_xpbd_rod():
    """Import the in-tree XPBD rod APIs (vendored; see ``newton_xpbd_rod``)."""
    from .. import newton_xpbd_rod as vendored_xpbd_rod
    from ..newton_xpbd_rod import SolverXPBDRod

    return SolverXPBDRod, vendored_xpbd_rod


def particle_mass_from_rod_config(config: RodConfig) -> float:
    """Uniform per-particle mass [kg] from cylinder segment density."""
    r = (
        float(config.geometry.radius)
        if not isinstance(config.geometry.radius, list)
        else float(config.geometry.radius[0])
    )
    L = float(config.geometry.segment_length)
    rho = float(config.material.density)
    return max(math.pi * r * r * L * rho, 1.0e-9)


def torsion_modulus_from_rod_config(config: RodConfig) -> float:
    """Shear / torsion modulus [Pa] from material config."""
    mat = config.material
    if mat.shear_modulus is not None:
        return float(mat.shear_modulus)
    return float(mat.young_modulus) / (2.0 * (1.0 + float(mat.poisson_ratio)))


def initial_rod_positions(
    config: RodConfig,
    *,
    z_height: float = 1.0,
    start: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Build rest centerline sample points for ``add_elastic_rod``.

    Args:
        config: Rod geometry / material config.
        z_height: World Z of the centerline when ``start`` is omitted.
        start: Optional world-space origin ``(3,)``.
        direction: Optional unit insertion direction ``(3,)``. Defaults to +X.

    Returns:
        ``(num_segments, 3)`` float32 positions (segment centers along the axis).
    """
    n = int(config.geometry.num_segments)
    L = float(config.geometry.segment_length)
    if start is None:
        origin = np.array([0.0, 0.0, float(z_height)], dtype=np.float32)
    else:
        origin = np.asarray(start, dtype=np.float32).reshape(3)
    if direction is None:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.asarray(direction, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            raise ValueError("direction must be non-zero")
        axis = axis / norm

    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        pos[i] = origin + axis * ((i + 0.5) * L)
    return pos


def register_xpbd_rod_builder_attributes(builder: Any) -> None:
    """Register SolverXPBDRod custom particle/shape attributes on *builder*.

    Call from ``NewtonManager._register_builder_attributes`` before particles
    are added / finalize runs.
    """
    SolverXPBDRod, _ = _load_newton_xpbd_rod()
    SolverXPBDRod.register_custom_attributes(builder)


def add_catheter_rod_to_builder(
    builder: Any,
    config: RodConfig,
    *,
    positions: np.ndarray | None = None,
    z_height: float = 1.0,
    start: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    lock_root: bool = True,
    lock_root_rotation: bool = True,
    num_envs: int = 1,
) -> None:
    """Add one (or ``num_envs``) elastic rod(s) to a Newton ``ModelBuilder``.

    This is the Path B spawn hook: Isaac Lab owns finalize; you only append
    rod particles/constraints here during scene construction.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    _, xpbd_rod = _load_newton_xpbd_rod()

    if positions is None:
        positions = initial_rod_positions(config, z_height=z_height, start=start, direction=direction)
    else:
        positions = np.asarray(positions, dtype=np.float32)

    r = config.geometry.radius
    radius = float(r[0]) if isinstance(r, list) else float(r)
    particle_mass = particle_mass_from_rod_config(config)
    torsion_mod = torsion_modulus_from_rod_config(config)
    bend = float(max(config.material.bend_stiffness, 1.0e-4))
    twist = float(max(config.material.twist_stiffness, 1.0e-4))
    young = float(config.material.young_modulus)

    for _ in range(num_envs):
        xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            radius=radius,
            particle_mass=particle_mass,
            bend_stiffness=bend,
            twist_stiffness=twist,
            young_modulus=young,
            torsion_modulus=torsion_mod,
            lock_root=lock_root,
            lock_root_rotation=lock_root_rotation,
        )


__all__ = [
    "add_catheter_rod_to_builder",
    "initial_rod_positions",
    "particle_mass_from_rod_config",
    "register_xpbd_rod_builder_attributes",
    "torsion_modulus_from_rod_config",
]
