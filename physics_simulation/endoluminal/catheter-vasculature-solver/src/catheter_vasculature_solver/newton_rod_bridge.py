# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""State marshalling between Newton particle buffers and the in-tree rod solver.

Deliberately free of any ``newton`` import: the copy logic, index arithmetic and
validation live here so they can be exercised without a Newton installation,
while :mod:`catheter_vasculature_solver.newton_rod_solver` holds the thin
``SolverBase`` subclass that Newton and Isaac Lab actually call.

Only positions and velocities cross the boundary. Rod orientations, Darboux
rest state, per-edge stiffness and contact topology stay inside the rod solver,
which is why this integration needs no Newton custom particle attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import warp as wp

from .host_arrays import env_ids_from_world_mask

if TYPE_CHECKING:
    from .xpbd_rod_solver import XPBDRodSolver


@dataclass(frozen=True)
class RodParticleRange:
    """Window of Newton particle indices owned by one rod solver.

    Newton keeps every particle in the scene in one flat array, so a rod that
    shares a :class:`~newton.Model` with cloth, soft bodies or other rods owns
    the half-open index range ``[offset, offset + count)``.

    Args:
        offset: Index of the rod's first particle in Newton's particle arrays.
        count: Number of particles owned, i.e. ``num_points * num_envs``.
        num_envs: Number of rods replicated in this range.
    """

    offset: int
    count: int
    num_envs: int = 1

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"offset must be >= 0, got {self.offset}")
        if self.count <= 0:
            raise ValueError(f"count must be > 0, got {self.count}")
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.count % self.num_envs != 0:
            raise ValueError(f"count {self.count} is not divisible by num_envs {self.num_envs}")

    @property
    def end(self) -> int:
        """One past the last owned particle index."""
        return self.offset + self.count


class RodParticleBridge:
    """Copy rod particle state to and from an external Newton-style buffer.

    The rod solver remains authoritative for rod dynamics; this only mirrors
    positions (and optionally velocities) so the rest of a Newton scene —
    collision, renderers, scene-data consumers — observes the rod.

    Args:
        rod: Rod solver instance (:class:`~catheter_vasculature_solver.XPBDRodSolver`
            or a subclass such as ``CathRodSolver``).
        particle_range: Index window this rod owns in the Newton particle arrays.
        sync_velocities: Also mirror ``particle_qd``. Disable when another
            solver owns velocities for these particles.
        sync_from_state: Read positions back from the incoming state before
            stepping. Keeps externally authored motion (resets, teleports,
            another solver's corrections) from being silently discarded. Disable
            to let the rod solver own its state outright.
    """

    def __init__(
        self,
        rod: "XPBDRodSolver",
        particle_range: RodParticleRange,
        *,
        sync_velocities: bool = True,
        sync_from_state: bool = True,
    ) -> None:
        expected = int(rod.num_points) * int(rod.num_envs)
        if particle_range.count != expected:
            raise ValueError(
                f"particle_range.count ({particle_range.count}) does not match the rod's particle "
                f"count ({expected} = num_points {rod.num_points} * num_envs {rod.num_envs})"
            )
        if particle_range.num_envs != int(rod.num_envs):
            raise ValueError(
                f"particle_range.num_envs ({particle_range.num_envs}) does not match the rod's "
                f"num_envs ({rod.num_envs})"
            )

        self._rod = rod
        self._range = particle_range
        self._sync_velocities = bool(sync_velocities)
        self._sync_from_state = bool(sync_from_state)

    @property
    def rod(self) -> "XPBDRodSolver":
        """The wrapped rod solver."""
        return self._rod

    @property
    def particle_range(self) -> RodParticleRange:
        """Index window this bridge copies through."""
        return self._range

    def read_from(self, state: Any) -> None:
        """Copy ``state.particle_q`` (and ``particle_qd``) into the rod solver."""
        if not self._sync_from_state:
            return
        positions = self._require_buffer(state, "particle_q")
        wp.copy(self._rod.position_array, positions, 0, self._range.offset, self._range.count)
        velocities = self._optional_buffer(state, "particle_qd")
        if self._sync_velocities and velocities is not None:
            wp.copy(self._rod.velocity_array, velocities, 0, self._range.offset, self._range.count)

    def write_to(self, state: Any) -> None:
        """Copy rod positions (and velocities) back into ``state``."""
        self._write_particle_slice(state, 0, self._range.count)

    def _write_particle_slice(self, state: Any, source_offset: int, count: int) -> None:
        """Publish a contiguous slice without touching other environments."""
        destination_offset = self._range.offset + source_offset
        positions = self._require_buffer(state, "particle_q")
        wp.copy(positions, self._rod.position_array, destination_offset, source_offset, count)
        velocities = self._optional_buffer(state, "particle_qd")
        if self._sync_velocities and velocities is not None:
            wp.copy(velocities, self._rod.velocity_array, destination_offset, source_offset, count)

    def advance(self, dt: float) -> None:
        """Step the rod solver by ``dt`` seconds."""
        self._rod.step(dt)

    def reset(self, state: Any, world_mask: Any = None) -> None:
        """Reset the envs selected by *world_mask* and republish them to *state*.

        The write-back is not optional. With ``sync_from_state`` enabled the next
        :meth:`read_from` would copy ``state.particle_q`` back over the restored
        rest pose, silently undoing the reset, so the freshly reset positions
        have to land in the Newton buffers before anything steps again.

        Args:
            state: Newton-style state to republish into. Anything exposing
                ``particle_q`` (and optionally ``particle_qd``) works.
            world_mask: Per-world reset mask, one entry per env, non-zero to
                reset. ``None`` resets every env. An all-zero mask is a no-op.

        Raises:
            ValueError: If ``world_mask`` length does not match the rod's env
                count. Guessing a correspondence would reset the wrong envs.
        """
        env_ids = self._env_ids_from_world_mask(world_mask)
        if env_ids is not None and len(env_ids) == 0:
            return
        self._rod.reset(env_ids)
        if env_ids is None:
            self.write_to(state)
        else:
            count = self._range.count // self._range.num_envs
            for env in env_ids:
                self._write_particle_slice(state, int(env) * count, count)

    def _env_ids_from_world_mask(self, world_mask: Any) -> Any:
        """Convert a per-world mask into rod env ids, or ``None`` for all envs."""
        return env_ids_from_world_mask(world_mask, self._range.num_envs)

    def _require_buffer(self, state: Any, name: str) -> Any:
        buffer = getattr(state, name, None)
        if buffer is None:
            raise RuntimeError(
                f"Newton state has no '{name}' buffer. The rod's particles must be added to the "
                "ModelBuilder before finalize() — see add_catheter_rod_to_builder()."
            )
        self._check_bounds(buffer, name)
        return buffer

    def _optional_buffer(self, state: Any, name: str) -> Any:
        buffer = getattr(state, name, None)
        if buffer is None:
            return None
        self._check_bounds(buffer, name)
        return buffer

    def _check_bounds(self, buffer: Any, name: str) -> None:
        length = len(buffer)
        if self._range.end > length:
            raise RuntimeError(
                f"rod particle range [{self._range.offset}, {self._range.end}) does not fit in "
                f"'{name}' of length {length}"
            )


__all__ = ["RodParticleBridge", "RodParticleRange"]
