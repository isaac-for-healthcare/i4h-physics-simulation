# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Newton solver configs for the catheter rod.

Requires ``isaaclab`` / ``isaaclab_newton`` at import time. The solver these
configs dispatch to is the in-tree
:class:`~catheter_vasculature_solver.CathRodSolver` wrapped as a Newton
``SolverBase``, so no unreleased Newton build is needed.
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
    """XPBD Cosserat catheter / guidewire rod config.

    ``class_type`` points at
    :class:`~catheter_vasculature_solver.isaaclab_integration.xpbd_rod_manager.NewtonXPBDRodManager`,
    which builds a :class:`~catheter_vasculature_solver.XPBDRodSolver` by default
    behind Newton's ``SolverBase`` interface. Nested under
    :class:`CoupledMJWarpXPBDRodSolverCfg` for coupled rigid + catheter scenes,
    or used alone for a catheter-only scene.

    Vessel geometry, insertion track and centerline vessel runtimes are GPU
    buffers rather than config values, so they are supplied through
    :meth:`~catheter_vasculature_solver.isaaclab_integration.xpbd_rod_manager.NewtonXPBDRodManager.register_rod`
    instead of here.
    """

    class_type: type[NewtonManager] | str = (
        "catheter_vasculature_solver.isaaclab_integration.xpbd_rod_manager:NewtonXPBDRodManager"
    )
    solver_type: str = "xpbd_rod"

    # Direct-solve backend and damping (see XPBDRodSolver)
    solver_backend: Literal["block_thomas", "split_thomas", "block_jacobi", "banded_cholesky"] = (
        "block_thomas"
    )
    linear_damping: float = 0.01
    angular_damping: float = 0.01
    floor_z: float | None = None

    # Material / geometry mirrors of RodConfig
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

    # Rod-internal substeps, applied per Isaac Lab substep. Keep at 1 when the
    # outer NewtonCfg already substeps, since the two multiply.
    num_substeps: int = 1
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)

    # Track guidance / containment require an explicitly registered rod or factory.
    track_enabled: bool = False
    collision_enabled: bool = False
    tip_num_edges: int = 10
    track_stiffness: float = 1.0

    # Mirror rod positions/velocities back from the incoming Newton state each
    # step so resets and other solvers' corrections are respected.
    sync_from_state: bool = True
    sync_velocities: bool = True


@configclass
class CoupledMJWarpXPBDRodSolverCfg(NewtonSolverCfg):
    """MJWarp (rigid) + XPBD catheter rod coupled Newton solver.

    Follows the Isaac Lab coupled-manager pattern (MJWarp + VBD): nested solver
    configs plus a coupling mode, with the manager owning substep order.
    """

    class_type: type[NewtonManager] | str = (
        "catheter_vasculature_solver.isaaclab_integration.coupled_manager:"
        "NewtonCoupledMJWarpXPBDRodManager"
    )
    solver_type: str = "coupled_mjwarp_xpbd_rod"

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    rod_solver_cfg: XPBDRodSolverCfg = XPBDRodSolverCfg()

    # one_way: contacts affect the rod only (tools/robots push the catheter)
    # two_way: also feed the catheter's drive-point reaction back to the rigid
    #          body holding its proximal end, so the driving arm feels the
    #          catheter's stiffness, buckling and vessel contact
    coupling_mode: Literal["one_way", "two_way"] = "one_way"

    # Body holding the catheter's proximal end, indexed within a single env and
    # replicated across envs by the manager. Two-way coupling has nothing to
    # push on until this names a body, so it or drive_body_name is required for
    # that mode.
    drive_body_index: int | None = None

    # The same body given by name instead, resolved against the builder's
    # ``body_label`` when the solver is built. Preferred over an index, which
    # silently means a different link if the scene's body order changes. Either
    # a leaf name such as "TCP" or a full USD path.
    drive_body_name: str | None = None

    # Fixed grip position in the drive body's local frame, not insertion depth.
    drive_mount_local: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Blend weight against the previously harvested reaction, matching
    # ``proxy_relaxation`` in Newton's coupled framework. With one coupling
    # iteration, below 1 this low-pass filters the reaction across substeps: it
    # damps a ringing drive point at the cost of lagging the force further.
    # With several, it relaxes the iteration, and with "aitken" it is the first
    # iteration's weight.
    drive_reaction_relaxation: float = 1.0

    # "fixed" keeps drive_reaction_relaxation; "aitken" adapts it between
    # coupling iterations of a substep within [min, max], as Newton's
    # proxy_relaxation="aitken". Only differs from "fixed" with iterations > 1.
    drive_relaxation_mode: Literal["fixed", "aitken"] = "fixed"
    drive_relaxation_min: float = 0.05
    drive_relaxation_max: float = 1.0

    # Times each substep's rigid and rod solves are re-run from the substep's
    # start with an updated guess of the catheter's reaction, as Newton's
    # Config.iterations. Converged, the holder's force and the rod's vessel
    # contacts are solved together rather than a substep apart, including the
    # holder's articulation and drive gains. 1 is the single lagged pass; more
    # need two_way and cost a full rod save and restore per iteration.
    drive_coupling_iterations: int = 1

    # Hold the catheter's root on drive_mount_local of the drive body inside
    # every substep, between the rigid and rod solves, instead of leaving it on
    # the path a task commands. Requires a drive body in either coupling mode,
    # and the task must then command the body rather than the rod's root.
    drive_root_sync: bool = False

    # With drive_root_sync, let the root stand in for the holder, as Newton's
    # proxies do: it carries the drive link's mass times
    # drive_proxy_mass_scale and starts each substep moving with the holder's
    # free (rewound) velocity. Vessel containment, track guidance and friction
    # still treat it as held. Needs two_way and drive_coupling_iterations >= 2:
    # with a single pass the rewind is inconsistent and the coupling diverges.
    drive_proxy_mass: bool = False
    drive_proxy_mass_scale: float = 1.0

    # Soft contact material for particle–shape pairs (tune like Franka soft-lift)
    soft_contact_ke: float = 5.0e3
    soft_contact_kd: float = 1.0e2
    soft_contact_mu: float = 0.5


__all__ = [
    "CoupledMJWarpXPBDRodSolverCfg",
    "XPBDRodSolverCfg",
]
