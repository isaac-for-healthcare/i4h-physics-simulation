# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Add catheter rod particles to a Newton ``ModelBuilder``.

Isaac Lab's ``NewtonManager`` owns ``ModelBuilder.finalize()``, so rods must be
registered on the builder *before* finalize — typically from scene setup or an
asset spawn hook — while the manager only constructs the solver from the
finalized model.

Only particles are added. Rod orientations, rest Darboux frames, per-edge
stiffness and contact topology live inside
:class:`~catheter_vasculature_solver.CathRodSolver`, so no Newton custom
attributes are needed and the particle positions authored here are the entire
Newton-side footprint of the rod.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..newton_rod_bridge import RodParticleRange
from ..rod_data import RodConfig, RodGeometryConfig, RodMaterialConfig, RodSolverConfig


def _load_newton_xpbd_rod():
    """Import the in-tree XPBD rod APIs (vendored; see ``newton_xpbd_rod``)."""
    from .. import newton_xpbd_rod as vendored_xpbd_rod
    from ..newton_xpbd_rod import SolverXPBDRod

    return SolverXPBDRod, vendored_xpbd_rod


def rod_config_from_solver_cfg(solver_cfg: Any, *, device: str) -> RodConfig:
    """Translate an Isaac Lab rod solver config into a :class:`RodConfig`.

    Duck-typed on purpose so it stays importable and testable without Isaac Lab;
    any object exposing the same field names works.

    Args:
        solver_cfg: ``XPBDRodSolverCfg``-shaped config.
        device: Warp device the rod must live on — normally the Newton model's
            device, since the two have to match.
    """
    return RodConfig(
        material=RodMaterialConfig(
            young_modulus=float(solver_cfg.young_modulus),
            shear_modulus=(
                None if solver_cfg.torsion_modulus is None else float(solver_cfg.torsion_modulus)
            ),
            density=float(solver_cfg.density),
            damping=float(solver_cfg.linear_damping),
            bend_stiffness=float(solver_cfg.bend_stiffness),
            twist_stiffness=float(solver_cfg.twist_stiffness),
        ),
        geometry=RodGeometryConfig(
            num_segments=int(solver_cfg.num_segments),
            segment_length=float(solver_cfg.segment_length),
            radius=float(solver_cfg.radius),
        ),
        solver=RodSolverConfig(
            num_substeps=int(solver_cfg.num_substeps),
            gravity=tuple(float(g) for g in solver_cfg.gravity),
        ),
        device=device,
    )


def particle_mass_from_rod_config(config: RodConfig, segment_length: float | None = None) -> float:
    """Uniform per-particle mass [kg] from cylinder segment density.

    Mirrors the rod solver's own mass computation so Newton and the rod agree.
    Clamped away from zero because a zero mass marks a particle kinematic in
    Newton, which is reserved for the locked root. ``segment_length`` overrides
    the config's, for a rod whose length differs per environment.
    """
    r = (
        float(config.geometry.radius)
        if not isinstance(config.geometry.radius, list)
        else float(config.geometry.radius[0])
    )
    length = float(config.geometry.segment_length if segment_length is None else segment_length)
    density = float(config.material.density)
    return max(math.pi * r * r * length * density, 1.0e-9)


def torsion_modulus_from_rod_config(config: RodConfig) -> float:
    """Shear / torsion modulus [Pa] from material config."""
    mat = config.material
    if mat.shear_modulus is not None:
        return float(mat.shear_modulus)
    return float(mat.young_modulus) / (2.0 * (1.0 + float(mat.poisson_ratio)))


def initial_rod_positions(
    config: RodConfig,
    *,
    initial_height: float = 0.5,
    start: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Build the rod's rest particle positions.

    Matches :class:`~catheter_vasculature_solver.XPBDRodSolver`'s own rest
    layout — ``num_segments + 1`` particles spaced by ``segment_length`` — so
    the Newton model and the rod solver start from identical state. A mismatch
    would show up as the rod snapping on the first step, since the bridge treats
    the Newton state as authoritative.

    Args:
        config: Rod geometry / material config.
        initial_height: World Z of the centerline when ``start`` is omitted.
            Same meaning as the solver's ``initial_height``.
        start: Optional world-space origin ``(3,)``.
        direction: Optional unit insertion direction ``(3,)``. Defaults to +X,
            matching the solver's rest orientation.

    Returns:
        ``(num_segments + 1, 3)`` float32 particle positions.
    """
    num_points = int(config.geometry.num_segments) + 1
    length = float(config.geometry.segment_length)

    if start is None:
        origin = np.array([0.0, 0.0, float(initial_height)], dtype=np.float32)
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

    offsets = (np.arange(num_points, dtype=np.float32) * length)[:, None]
    return (origin[None, :] + axis[None, :] * offsets).astype(np.float32)


def _per_env_vectors(value, num_envs: int, name: str) -> list:
    """``value`` as one ``(3,)`` vector (or ``None``) per env."""
    if value is None:
        return [None] * num_envs
    vectors = np.asarray(value, dtype=np.float32)
    if vectors.ndim == 1:
        return [vectors] * num_envs
    if vectors.shape != (num_envs, 3):
        raise ValueError(f"{name} must have shape (3,) or ({num_envs}, 3), got {vectors.shape}")
    return list(vectors)


def add_catheter_rod_to_builder(
    builder: Any,
    config: RodConfig,
    *,
    positions: np.ndarray | None = None,
    initial_height: float = 0.5,
    start: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    lock_root: bool = True,
    num_envs: int = 1,
    env_segment_lengths: Any = None,
) -> RodParticleRange:
    """Add one rod's particles (or ``num_envs`` replicas) to a Newton builder.

    Args:
        builder: Newton ``ModelBuilder`` being populated before finalize.
        config: Rod geometry / material config, also used by the solver.
        positions: Optional explicit rest positions: ``(num_segments + 1, 3)``
            shared by every replica, or ``(num_envs, num_segments + 1, 3)`` with
            one shape per env (each env's own route into its own anatomy).
            Defaults to :func:`initial_rod_positions`.
        initial_height: Forwarded to :func:`initial_rod_positions`.
        start: Forwarded to :func:`initial_rod_positions`; ``(3,)`` or
            ``(num_envs, 3)``.
        direction: Forwarded to :func:`initial_rod_positions`; ``(3,)`` or
            ``(num_envs, 3)``.
        lock_root: Give the root particle zero mass so Newton treats it as
            kinematic, matching the rod solver's locked root.
        num_envs: Number of rods to replicate into the same model.
        env_segment_lengths: Optional ``(num_envs,)`` segment length per env,
            matching the solver's ``env_segment_lengths`` so each env's particle
            mass agrees with its rod. Defaults to the config's segment length.

    Returns:
        The :class:`~catheter_vasculature_solver.newton_rod_bridge.RodParticleRange`
        describing where these particles landed. Pass it to
        :class:`~catheter_vasculature_solver.newton_rod_solver.NewtonCathRodSolver`.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    num_points = int(config.geometry.num_segments) + 1
    if positions is None:
        starts = _per_env_vectors(start, num_envs, "start")
        directions = _per_env_vectors(direction, num_envs, "direction")
        positions = np.stack([
            initial_rod_positions(config, initial_height=initial_height, start=s, direction=d)
            for s, d in zip(starts, directions, strict=True)
        ])
    else:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 2:
            positions = np.broadcast_to(positions, (num_envs,) + positions.shape)
    if positions.shape != (num_envs, num_points, 3):
        raise ValueError(
            f"positions must have shape ({num_points}, 3) or ({num_envs}, {num_points}, 3) "
            f"to match num_segments + 1, got {positions.shape}"
        )

    radius = config.geometry.radius
    radius = float(radius[0]) if isinstance(radius, list) else float(radius)
    if env_segment_lengths is None:
        particle_masses = [particle_mass_from_rod_config(config)] * num_envs
    else:
        lengths = np.asarray(env_segment_lengths, dtype=np.float64).reshape(-1)
        if lengths.shape != (num_envs,):
            raise ValueError(f"env_segment_lengths must have shape ({num_envs},), got {lengths.shape}")
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise ValueError("env_segment_lengths must be finite and positive")
        particle_masses = [particle_mass_from_rod_config(config, float(length)) for length in lengths]
    zero_velocity = (0.0, 0.0, 0.0)

    offset = int(builder.particle_count)
    for env in range(num_envs):
        for index in range(num_points):
            mass = 0.0 if (lock_root and index == 0) else particle_masses[env]
            builder.add_particle(
                pos=tuple(float(v) for v in positions[env, index]),
                vel=zero_velocity,
                mass=mass,
                radius=radius,
            )

    return RodParticleRange(offset=offset, count=num_points * num_envs, num_envs=num_envs)


__all__ = [
    "add_catheter_rod_to_builder",
    "initial_rod_positions",
    "particle_mass_from_rod_config",
    "rod_config_from_solver_cfg",
    "torsion_modulus_from_rod_config",
]
