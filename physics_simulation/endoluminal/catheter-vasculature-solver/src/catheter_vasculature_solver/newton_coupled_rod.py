# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catheter rod as a participant in Newton's coupled-solver framework.

Newton 1.8 ships proxy coupling as ``newton.solvers.experimental.coupled``.
:func:`build_coupled_rod_solver` puts the rod and a rigid solver into one
``SolverCoupledProxy``: the rigid entry owns every body, joint and shape, the
rod entry owns the rod's particles, and each drive body is mirrored into the
rod's view as a proxy. Newton then runs the lagged proxy loop itself, with its
own iterations, fixed or Aitken relaxation, and articulated effective mass for
the proxy.

:class:`CoupledCathRodSolver` implements the hooks that loop calls on the rod:

* ``coupling_rewind_proxy_body`` places each rod's root on its drive mount at
  the proxy's begin pose, moving with the holder's end velocity with the lagged
  reaction rewound out, and gives it the proxy's inverse mass;
* ``coupling_harvest_proxy_wrenches`` reports the rod's proximal reaction from
  its constraint multipliers, carried onto the drive body's center of mass. The
  rod's root is a particle, not a body, so the default momentum harvest of a
  proxy body would see nothing;
* ``coupling_notify_input_state_update`` saves the rod's full state at the start
  of each coupled step and restores it before each further iteration.

The rod's root is translational only: its orientation follows the proxy's
rotation kinematically, and the torque part of the lagged reaction is not
rewound. A rewound proxy needs the step re-solved, so the builder requires at
least two iterations.

This backend needs ``newton>=1.8``, which needs a newer Warp than Isaac Lab
pins; Isaac Lab scenes keep using
:class:`~catheter_vasculature_solver.isaaclab_integration.coupled_manager.NewtonCoupledMJWarpXPBDRodManager`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import warp as wp

from .host_arrays import env_ids_from_world_mask
from .isaaclab_integration.proxy_coupling import _drive_reaction_wrench, _unbind_envs_kernel
from .isaaclab_integration.rod_snapshot import RodStateSnapshot
from .newton_rod_bridge import RodParticleRange
from .newton_rod_solver import NewtonCathRodSolver

try:
    from newton import StateFlags
    from newton.solvers.experimental.coupled import CouplingInterface, SolverCoupled, SolverCoupledProxy
except ImportError:
    StateFlags = CouplingInterface = SolverCoupled = SolverCoupledProxy = None

if TYPE_CHECKING:
    from newton import Model, State

    from .xpbd_rod_solver import XPBDRodSolver

#: Entry names in the coupled solver, for ``SolverCoupled.solver(name)`` and ``entry_state(name)``.
RIGID_ENTRY = "rigid"
ROD_ENTRY = "rod"


def coupled_framework_available() -> bool:
    """Whether the installed Newton ships ``newton.solvers.experimental.coupled``."""
    return SolverCoupledProxy is not None


@wp.kernel
def _proxy_root_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=wp.float32),
    coupling_forces: wp.array(dtype=wp.spatial_vector),
    proxy_local: wp.array(dtype=wp.int32),
    proxy_global: wp.array(dtype=wp.int32),
    mount_local: wp.array(dtype=wp.vec3),
    rod_orientations: wp.array(dtype=wp.quat),
    root_stride: int,
    dt: float,
    bound: wp.array(dtype=wp.int32),
    root_rotation_local: wp.array(dtype=wp.quat),
    out_positions: wp.array(dtype=wp.vec3),
    out_velocities: wp.array(dtype=wp.vec3),
    out_inv_masses: wp.array(dtype=wp.float32),
    out_orientations: wp.array(dtype=wp.quat),
):
    """Start each root on its mount at the proxy's begin pose, with the holder's free velocity.

    ``body_qd`` is the source's end-of-step velocity, which already holds the
    lagged reaction; ``dt·F/m`` takes it back out at the proxy's mass, as
    Newton's default body rewind does through ``body_f``.
    """
    env = wp.tid()
    local = proxy_local[env]
    pose = body_q[local]
    rotation = wp.transform_get_rotation(pose)

    if bound[env] == 0:
        root_rotation_local[env] = wp.quat_inverse(rotation) * rod_orientations[env * root_stride]
        bound[env] = 1

    inv_mass = body_inv_mass[local]
    twist = body_qd[local]
    angular = wp.spatial_bottom(twist)
    linear = wp.spatial_top(twist) - wp.spatial_top(coupling_forces[proxy_global[env]]) * (inv_mass * dt)
    arm = wp.quat_rotate(rotation, mount_local[env] - body_com[local])
    spin = wp.quat(angular[0], angular[1], angular[2], 0.0) * rotation
    end_rotation = wp.normalize(rotation + spin * (0.5 * dt))

    out_positions[env] = wp.transform_point(pose, mount_local[env])
    out_velocities[env] = linear + wp.cross(angular, arm)
    out_inv_masses[env] = inv_mass
    out_orientations[env] = wp.normalize(end_rotation * root_rotation_local[env])


@wp.kernel
def _harvest_proxy_wrench_kernel(
    proximal_wrench: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    proxy_local: wp.array(dtype=wp.int32),
    proxy_global: wp.array(dtype=wp.int32),
    mount_local: wp.array(dtype=wp.vec3),
    out_body_f: wp.array(dtype=wp.spatial_vector),
):
    env = wp.tid()
    local = proxy_local[env]
    arm = wp.transform_vector(body_q[local], mount_local[env] - body_com[local])
    out_body_f[proxy_global[env]] = _drive_reaction_wrench(proximal_wrench[env], arm)


_Base = (NewtonCathRodSolver, CouplingInterface) if CouplingInterface is not None else (NewtonCathRodSolver,)


class CoupledCathRodSolver(*_Base):
    """The rod solver as the destination of Newton proxy coupling.

    Built by :func:`build_coupled_rod_solver` on the rod entry's view; the hooks
    are called by ``SolverCoupledProxy`` and are not meant to be called directly.

    Args:
        model: The rod entry's ``ModelView``.
        rod: Rod solver to drive.
        particle_range: The rod's particles in the view's own indexing.
        drive_bodies: Per-env global id of the body holding each rod.
        mount_local: Per-env grip point in the drive body's frame. Defaults to
            the body origin.
        allow_internal_cuda_graph: As on :class:`NewtonCathRodSolver`.
    """

    def __init__(
        self,
        model: "Model",
        rod: "XPBDRodSolver",
        particle_range: RodParticleRange,
        *,
        drive_bodies: Sequence[int],
        mount_local=None,
        allow_internal_cuda_graph: bool = False,
    ) -> None:
        super().__init__(model, rod, particle_range, allow_internal_cuda_graph=allow_internal_cuda_graph)
        num_envs = int(rod.num_envs)
        drive = [int(body) for body in drive_bodies]
        if len(drive) != num_envs:
            raise ValueError(f"drive_bodies has {len(drive)} entries but the rod has {num_envs} envs")
        mounts = [(0.0, 0.0, 0.0)] * num_envs if mount_local is None else [tuple(m) for m in mount_local]
        if len(mounts) != num_envs or any(len(m) != 3 for m in mounts):
            raise ValueError(f"mount_local needs {num_envs} 3-vectors, got {mounts!r}")

        device = rod.device
        self._drive_bodies = drive
        self._root_stride = int(rod.num_points)
        self._mount_local = wp.array(mounts, dtype=wp.vec3, device=device)
        self._proxy_local = None
        self._proxy_global = wp.array(drive, dtype=wp.int32, device=device)
        self._bound = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self._root_rotation_local = wp.zeros(num_envs, dtype=wp.quat, device=device)
        self._root_positions = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self._root_velocities = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self._root_inv_masses = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self._root_orientations = wp.zeros(num_envs, dtype=wp.quat, device=device)
        self._proximal_wrench = wp.zeros(num_envs, dtype=wp.spatial_vector, device=device)
        self._reset_env_ids = wp.empty(num_envs, dtype=wp.int32, device=device)
        self._restarts = False
        self._snapshot = None

    @property
    def root_positions(self):
        """Per-env root positions placed by the most recent rewind."""
        return self._root_positions

    @property
    def root_velocities(self):
        """Per-env root velocities placed by the most recent rewind."""
        return self._root_velocities

    @property
    def root_inv_masses(self):
        """Per-env proxy inverse masses placed by the most recent rewind."""
        return self._root_inv_masses

    def _bind_proxies(self, body_local_to_proxy_global) -> None:
        if self._proxy_local is not None:
            return
        local_of = {int(g): i for i, g in enumerate(body_local_to_proxy_global.numpy()) if g >= 0}
        missing = [body for body in self._drive_bodies if body not in local_of]
        if missing:
            raise RuntimeError(f"drive bodies {missing} are not proxies in the rod's view")
        self._proxy_local = wp.array(
            [local_of[body] for body in self._drive_bodies], dtype=wp.int32, device=self.rod.device,
        )

    # ------------------------------------------------------------ coupling hooks
    def coupling_notify_iteration_restarts(self, enabled: bool) -> None:
        self._restarts = bool(enabled)
        if self._restarts and self._snapshot is None:
            self._snapshot = RodStateSnapshot(self.rod)

    def coupling_notify_input_state_update(self, state, flags, *, iteration_restart=False, dt=0.0) -> None:
        if not self._restarts:
            return
        if iteration_restart:
            if dt > 0.0:
                self._snapshot.restore()
            return
        # Only the step's own distribution redistributes particles; the proxy sync and rewind touch bodies.
        if int(flags) & int(StateFlags.PARTICLE_Q):
            self._snapshot.save()

    def coupling_rewind_proxy_body(
        self, body_local_to_proxy_global, state: "State", coupling_forces, body_gravity_acceleration, dt: float,
    ) -> None:
        del body_gravity_acceleration
        if not dt > 0.0:
            raise ValueError(f"a proxy root needs the step length, got dt={dt}")
        if self.rod.has_pending_control:
            raise RuntimeError(
                "the coupled solver holds the catheter's root on its drive body, but a rate or pose "
                "command is already latched on the rod and would be silently overridden. Command "
                "the drive body instead."
            )
        self._bind_proxies(body_local_to_proxy_global)
        wp.launch(
            _proxy_root_kernel,
            dim=self.rod.num_envs,
            inputs=[
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                coupling_forces,
                self._proxy_local,
                self._proxy_global,
                self._mount_local,
                self.rod.orientation_array,
                self._root_stride,
                float(dt),
                self._bound,
                self._root_rotation_local,
                self._root_positions,
                self._root_velocities,
                self._root_inv_masses,
                self._root_orientations,
            ],
            device=self.rod.device,
        )
        self.rod.set_root_proxy_gpu(
            self._root_positions, self._root_velocities, self._root_inv_masses, self._root_orientations,
        )

    def coupling_harvest_proxy_wrenches(
        self, body_local_to_proxy_global, out_body_f, *, body_qd_before, state, state_out, contacts, dt,
    ) -> None:
        """Write each rod's proximal reaction over its drive body's entry, replacing the guess."""
        del body_qd_before, state_out, contacts, dt
        self._bind_proxies(body_local_to_proxy_global)
        self.rod.proximal_wrench(out=self._proximal_wrench)
        wp.launch(
            _harvest_proxy_wrench_kernel,
            dim=self.rod.num_envs,
            inputs=[
                self._proximal_wrench,
                state.body_q,
                self.model.body_com,
                self._proxy_local,
                self._proxy_global,
                self._mount_local,
                out_body_f,
            ],
            device=self.rod.device,
        )

    def coupling_prepare_proxy_contacts(self, state, contacts, *, contacts_freshly_detected=False):
        del state, contacts_freshly_detected
        return contacts

    def reset(self, state: "State", world_mask=None, flags: int = 0) -> None:
        """Reset the selected envs and re-bind their root rotation at the next rewind."""
        super().reset(state, world_mask, flags)
        env_ids = env_ids_from_world_mask(world_mask, self.rod.num_envs)
        if env_ids is None:
            self._bound.zero_()
            return
        if len(env_ids) == 0:
            return
        wp.copy(self._reset_env_ids, wp.array(env_ids, dtype=wp.int32, device="cpu"), count=len(env_ids))
        wp.launch(
            _unbind_envs_kernel, dim=len(env_ids), inputs=[self._reset_env_ids, self._bound],
            device=self.rod.device,
        )


def _view_particle_range(view, model, particle_range: RodParticleRange) -> RodParticleRange:
    count = int(view.particle_count)
    if count == int(model.particle_count):
        return particle_range
    if count == particle_range.count:
        return RodParticleRange(offset=0, count=particle_range.count, num_envs=particle_range.num_envs)
    raise RuntimeError(
        f"the rod's view holds {count} particles, neither the model's {model.particle_count} "
        f"nor the rod's {particle_range.count}"
    )


def build_coupled_rod_solver(
    model: "Model",
    rod: "XPBDRodSolver",
    particle_range: RodParticleRange,
    *,
    rigid_solver: Callable,
    drive_bodies: Sequence[int],
    mount_local=None,
    iterations: int = 2,
    relaxation: float = 1.0,
    relaxation_mode: str = "fixed",
    relaxation_min: float = 0.1,
    relaxation_max: float = 1.0,
    mass_scale: float = 1.0,
    rigid_bodies: Sequence[int] | None = None,
    rigid_joints: Sequence[int] | None = None,
    rigid_shapes: Sequence[int] | None = None,
    allow_internal_cuda_graph: bool = False,
):
    """Couple the rod to a rigid solver with Newton's ``SolverCoupledProxy``.

    Args:
        model: Finalized model holding the rigid scene and the rod's particles.
        rod: Rod solver for the particles in ``particle_range``.
        particle_range: The rod's particles in ``model``.
        rigid_solver: Factory called with the rigid entry's ``ModelView``, for
            example ``lambda view: SolverMuJoCo(view)``.
        drive_bodies: Per-env global id of the body holding each rod.
        mount_local: Per-env grip point in the drive body's frame.
        iterations: Proxy iterations per step; at least 2.
        relaxation: ``proxy_relaxation``; with Aitken, the first iteration's weight.
        relaxation_mode: ``"fixed"`` or ``"aitken"``.
        relaxation_min: Lower clamp on the Aitken weight.
        relaxation_max: Upper clamp on the Aitken weight.
        mass_scale: Multiplies the articulated effective mass the proxy carries.
        rigid_bodies: Bodies the rigid entry owns; all by default.
        rigid_joints: Joints the rigid entry owns; all by default.
        rigid_shapes: Shapes the rigid entry owns; all by default.
        allow_internal_cuda_graph: As on :class:`NewtonCathRodSolver`.

    Returns:
        A ``SolverCoupledProxy`` with entries :data:`RIGID_ENTRY` and :data:`ROD_ENTRY`.

    Raises:
        ImportError: If the installed Newton has no coupled-solver framework.
        ValueError: If ``iterations < 2``.
    """
    if not coupled_framework_available():
        raise ImportError(
            "build_coupled_rod_solver needs newton>=1.8 (newton.solvers.experimental.coupled); "
            "install the 'newton-coupled' extra. Isaac Lab scenes use "
            "NewtonCoupledMJWarpXPBDRodManager instead."
        )
    if int(iterations) < 2:
        raise ValueError(
            f"iterations={iterations}: the rod's root is a rewound proxy, which is only consistent "
            "when the step is re-solved with the result; a single pass diverges. Use 2 or more."
        )
    drive = [int(body) for body in drive_bodies]

    def rod_factory(view):
        return CoupledCathRodSolver(
            view,
            rod,
            _view_particle_range(view, model, particle_range),
            drive_bodies=drive,
            mount_local=mount_local,
            allow_internal_cuda_graph=allow_internal_cuda_graph,
        )

    entries = [
        SolverCoupled.Entry(
            name=RIGID_ENTRY,
            solver=rigid_solver,
            bodies=range(int(model.body_count)) if rigid_bodies is None else rigid_bodies,
            joints=range(int(model.joint_count)) if rigid_joints is None else rigid_joints,
            shapes=range(int(model.shape_count)) if rigid_shapes is None else rigid_shapes,
        ),
        SolverCoupled.Entry(
            name=ROD_ENTRY,
            solver=rod_factory,
            particles=range(particle_range.offset, particle_range.end),
        ),
    ]
    proxy = SolverCoupledProxy.Proxy(
        source=RIGID_ENTRY,
        destination=ROD_ENTRY,
        bodies=drive,
        mass_scale=float(mass_scale),
        mode="lagged",
        proxy_relaxation=float(relaxation),
        proxy_relaxation_mode=str(relaxation_mode),
        proxy_relaxation_min=float(relaxation_min),
        proxy_relaxation_max=float(relaxation_max),
    )
    return SolverCoupledProxy(
        model, entries, SolverCoupledProxy.Config(proxies=[proxy], iterations=int(iterations)),
    )


__all__ = [
    "RIGID_ENTRY",
    "ROD_ENTRY",
    "CoupledCathRodSolver",
    "build_coupled_rod_solver",
    "coupled_framework_available",
]
