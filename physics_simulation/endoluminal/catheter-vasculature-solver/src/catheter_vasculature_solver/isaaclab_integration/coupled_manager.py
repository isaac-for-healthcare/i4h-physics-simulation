# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled MJWarp (rigid) + XPBD catheter rod ``NewtonManager``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_newton.physics import NewtonManager

from .solver_cfg import CoupledMJWarpXPBDRodSolverCfg, XPBDRodSolverCfg
from .xpbd_rod_manager import NewtonXPBDRodManager

if TYPE_CHECKING:
    from newton import Model


class NewtonCoupledMJWarpXPBDRodManager(NewtonManager):
    """Couple a rigid MJWarp solve with an XPBD Cosserat catheter rod.

    Mirrors the Isaac Lab MJWarp + VBD coupled-manager pattern:

    * Nested configs construct two sub-solvers.
    * :meth:`_step_solver` owns contact and substep ordering.
    * The base :class:`~isaaclab_newton.physics.NewtonManager` still owns model
      lifecycle, Fabric sync, resets and the outer substep loop.

    Substep algorithm::

        1. Clear rigid / particle force accumulators
        2. Apply the lagged catheter reaction  (two_way only)
        3. Run Newton collision pipeline (shared contacts)
        4. Step MJWarp rigid solver     (state_0 → state_1)
        5. Step the catheter particles   (state_0 → state_1)
        6. Harvest the catheter reaction      (two_way only)

    Steps 2 and 6 form the relaxed lagged fixed point that Newton's own
    coupled-solver framework uses for proxy coupling; see
    :mod:`~catheter_vasculature_solver.isaaclab_integration.proxy_coupling` for
    where this follows it and where it departs.

    Vessel containment and track guidance need no step here: they run inside the
    rod solver's own kernels. Scenes supply the vessel mesh or centerline
    runtime by handing a configured rod to
    :meth:`~catheter_vasculature_solver.isaaclab_integration.xpbd_rod_manager.NewtonXPBDRodManager.register_rod`.
    """

    _rigid_solver: Any = None
    _rod_solver: Any = None
    _coupler: Any = None
    _coupling_mode: str = "one_way"
    _coupled_cfg: CoupledMJWarpXPBDRodSolverCfg | None = None
    _rod_solver_cfg: XPBDRodSolverCfg | None = None
    #: Label of the resolved drive body, kept for the debug report.
    _drive_body_label: str | None = None
    #: Steps between coupling reports, or ``None`` when the diagnostic is off.
    _coupling_debug_interval: int | None = None
    _coupling_debug_steps: int = 0

    @classmethod
    def _register_builder_attributes(cls, builder) -> None:
        """Forward builder attribute registration to the rigid manager.

        The rod itself registers nothing: it needs no Newton custom attributes,
        because rod orientations and constraint topology stay inside its own
        solver rather than on the Newton model.
        """
        rigid_mgr = cls._resolve_rigid_manager_type()
        if rigid_mgr is not None and hasattr(rigid_mgr, "_register_builder_attributes"):
            rigid_mgr._register_builder_attributes(builder)

    @classmethod
    def _resolve_rigid_manager_type(cls):
        """Return the ``NewtonMJWarpManager`` class if importable."""
        try:
            from isaaclab_newton.physics import NewtonMJWarpManager

            return NewtonMJWarpManager
        except ImportError:
            try:
                from isaaclab_newton.physics.newton_mjwarp_manager import NewtonMJWarpManager  # type: ignore

                return NewtonMJWarpManager
            except ImportError:
                return None

    @classmethod
    def _build_rigid_solver(cls, model: "Model", rigid_cfg) -> Any:
        """Build the MJWarp solver via the in-tree manager when possible."""
        rigid_mgr = cls._resolve_rigid_manager_type()
        if rigid_mgr is not None and hasattr(rigid_mgr, "build_nested_solver"):
            return rigid_mgr.build_nested_solver(model, rigid_cfg)

        try:
            from newton.solvers import SolverMuJoCo
        except ImportError as e:
            raise ImportError(
                f"The coupled manager needs SolverMuJoCo (MJWarp) for the rigid half. Original error: {e}"
            ) from e

        kwargs: dict[str, Any] = {}
        for key in ("iterations", "ls_iterations", "njmax", "nconmax"):
            if hasattr(rigid_cfg, key):
                kwargs[key] = getattr(rigid_cfg, key)
        return SolverMuJoCo(model, **kwargs)

    @classmethod
    def _resolve_drive_body_name(cls, name: str, bodies_per_env: int) -> int:
        """Turn a drive-body name into its per-env index.

        ``Model`` drops body names, so this reads them off the ``ModelBuilder``
        that produced it. IsaacLab finalizes the builder before building the
        solver but keeps holding it, so the labels are still there; a caller
        without one has to fall back to ``drive_body_index``.
        """
        from .proxy_coupling import resolve_drive_body_index

        builder = getattr(NewtonManager, "_builder", None)
        labels = getattr(builder, "body_label", None) if builder is not None else None
        if not labels:
            raise RuntimeError(
                f"cannot resolve drive_body_name={name!r}: body names live on the Newton "
                "ModelBuilder, which is unavailable here, and the finalized Model does not "
                "carry them. Set drive_body_index instead."
            )
        index = resolve_drive_body_index(labels, name, bodies_per_env)
        cls._drive_body_label = str(list(labels)[index])
        return index

    @classmethod
    def _build_coupler(cls, model: "Model", solver_cfg: CoupledMJWarpXPBDRodSolverCfg):
        """Build the drive-reaction coupler for ``coupling_mode='two_way'``.

        The reaction has to land on a body, so a two-way scene must identify the
        one holding the catheter's proximal end, by name or by index. That index
        is given per env and replicated here, which assumes Newton replicated
        the bodies uniformly across envs.
        """
        drive_index = solver_cfg.drive_body_index
        drive_name = getattr(solver_cfg, "drive_body_name", None)
        if drive_index is None and drive_name is None:
            raise ValueError(
                "coupling_mode='two_way' needs drive_body_name or drive_body_index: the "
                "catheter's reaction is transmitted through whichever rigid body holds its "
                "proximal end, and there is nothing to push on until that body is named. Use "
                "'one_way' for a scene where rigid bodies drive the catheter without feeling "
                "it back."
            )
        if drive_index is not None and drive_name is not None:
            raise ValueError(
                f"drive_body_name={drive_name!r} and drive_body_index={drive_index} are both "
                "set, so the intended drive body is ambiguous. Give exactly one."
            )

        particle_range = NewtonXPBDRodManager._require_particle_range()
        num_envs = int(particle_range.num_envs)
        body_count = int(model.body_count)

        if body_count % num_envs:
            raise ValueError(
                f"Cannot map the drive body across envs: the model has {body_count} bodies, "
                f"which is not a multiple of {num_envs} envs, so per-env body replication "
                "cannot be inferred. Build the scene with a uniform body set per env."
            )
        bodies_per_env = body_count // num_envs

        if drive_name is not None:
            drive_index = cls._resolve_drive_body_name(str(drive_name), bodies_per_env)

        if not 0 <= int(drive_index) < bodies_per_env:
            raise IndexError(
                f"drive_body_index={drive_index} is out of range: each env owns "
                f"{bodies_per_env} bodies, so it must be indexed within a single env."
            )

        if int(particle_range.count) % num_envs:
            raise ValueError(
                f"Rod particle range holds {particle_range.count} particles across {num_envs} "
                "envs, which is not a whole number per rod."
            )

        from .proxy_coupling import DriveReactionCoupler, debug_log_interval

        coupler = DriveReactionCoupler(
            body_count=body_count,
            num_envs=num_envs,
            root_offset=int(particle_range.offset),
            root_stride=int(particle_range.count) // num_envs,
            drive_bodies=[int(drive_index) + env * bodies_per_env for env in range(num_envs)],
            mount_local=[tuple(solver_cfg.drive_mount_local)] * num_envs,
            body_com=model.body_com,
            relaxation=float(solver_cfg.drive_reaction_relaxation),
            device=getattr(model, "device", None),
        )

        cls._coupling_debug_interval = debug_log_interval()
        cls._coupling_debug_steps = 0
        if cls._coupling_debug_interval is not None:
            print(
                f"[coupling] two-way drive body {cls._drive_body_label or drive_index} "
                f"(per-env index {int(drive_index)} of {bodies_per_env}), "
                f"relaxation={float(solver_cfg.drive_reaction_relaxation)}, "
                f"{num_envs} env(s)",
                flush=True,
            )
        return coupler

    @classmethod
    def _build_solver(cls, model: "Model", solver_cfg: CoupledMJWarpXPBDRodSolverCfg) -> None:
        NewtonXPBDRodManager._reject_cuda_graph_capture(getattr(cls, "_cfg", None))
        cls._coupled_cfg = solver_cfg
        cls._rod_solver_cfg = solver_cfg.rod_solver_cfg
        cls._coupling_mode = str(solver_cfg.coupling_mode)

        cls._rigid_solver = cls._build_rigid_solver(model, solver_cfg.rigid_solver_cfg)
        cls._rod_solver = NewtonXPBDRodManager.build_nested_solver(model, solver_cfg.rod_solver_cfg)
        cls._coupler = (
            cls._build_coupler(model, solver_cfg) if cls._coupling_mode == "two_way" else None
        )

        # Expose the rod solver as the primary handle for diagnostics / Fabric
        # readers that expect one solver; stepping goes through _step_solver.
        NewtonManager._solver = cls._rod_solver
        NewtonManager._use_single_state = False
        # Rigid bodies need Newton contacts even though the rod does not.
        NewtonManager._needs_collision_pipeline = True

        # Stash class-level copies where the base class looks for them.
        NewtonManager._rigid_solver = cls._rigid_solver  # type: ignore[attr-defined]
        NewtonManager._rod_solver = cls._rod_solver  # type: ignore[attr-defined]
        NewtonManager._coupler = cls._coupler  # type: ignore[attr-defined]
        NewtonManager._coupling_mode = cls._coupling_mode  # type: ignore[attr-defined]

    @classmethod
    def _clear_force_buffers(cls, state) -> None:
        """Zero particle / body force accumulators when present."""
        for name in ("particle_f", "body_f", "joint_f"):
            buffer = getattr(state, name, None)
            if buffer is None:
                continue
            if hasattr(buffer, "zero_"):
                buffer.zero_()
            elif hasattr(buffer, "fill_"):
                buffer.fill_(0)

    @classmethod
    def _apply_soft_to_rigid_reactions(cls, contacts, state) -> None:
        """Inject the catheter's proximal reaction into ``body_f`` (two-way).

        Applies the wrench harvested by the previous substep, so the rigid
        solver feels the catheter's stiffness, buckling and vessel contact. A
        lagged pass rather than a simultaneous solve, matching the proxy path in
        Newton's coupled-solver framework.

        ``contacts`` is unused: the reaction is transmitted through the rod's own
        constraints at the drive point rather than through Newton's contact
        buffers, which the rod solver does not read.
        """
        del contacts
        coupler = cls._coupler or getattr(NewtonManager, "_coupler", None)
        if coupler is None:
            return
        coupler.apply(state)

    @classmethod
    def _run_collision_pipeline(cls, state_0, state_1, control) -> Any:
        """Invoke NewtonManager collision detection when allocated."""
        pipeline = getattr(NewtonManager, "_collision_pipeline", None)
        contacts = getattr(NewtonManager, "_contacts", None)
        if pipeline is None:
            return contacts
        # The API varies slightly across Lab revisions; prefer step() then collide().
        if hasattr(pipeline, "step"):
            return pipeline.step(state_0, state_1, control, contacts)
        if hasattr(pipeline, "collide"):
            # ``CollisionPipeline.collide(state, contacts)``. The pipeline is
            # constructed from the model and holds its own reference, so passing
            # the model again shifts every argument by one and raises TypeError
            # before any coupled step can run. Isaac Lab's own NewtonManager
            # calls it the same way. It fills the buffer in place and returns
            # nothing, so hand back the buffer rather than the call's result.
            pipeline.collide(state_0, contacts)
            return contacts
        return contacts

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt: float) -> None:
        """One coupled substep: contacts, then rigid, then rod."""
        rigid = cls._rigid_solver or getattr(NewtonManager, "_rigid_solver", None)
        rod = cls._rod_solver or getattr(NewtonManager, "_rod_solver", None)
        if rigid is None or rod is None:
            raise RuntimeError(
                "NewtonCoupledMJWarpXPBDRodManager sub-solvers are not built. "
                "Ensure _build_solver ran during NewtonManager.initialize()."
            )

        cls._clear_force_buffers(state_0)
        # Feedback from the previous substep has to land after the clear and
        # before the rigid solve, or it would be zeroed before anything read it.
        cls._apply_soft_to_rigid_reactions(contacts, state_0)
        contacts = cls._run_collision_pipeline(state_0, state_1, control) or contacts

        rigid.step(state_0, state_1, control, contacts, substep_dt)
        # MJWarp writes bodies and joints but leaves output particles untouched.
        # Read the current particles, then publish only the rod's particle slice
        # into the output, preserving the rigid solver's body and joint results.
        rod.step(state_0, state_1, control, contacts, substep_dt)

        coupler = cls._coupler or getattr(NewtonManager, "_coupler", None)
        if coupler is not None:
            coupler.harvest(state_1, rod.rod)

    @classmethod
    def _log_solver_debug(cls) -> None:
        """Report what the catheter is doing to its holder, when asked to.

        Fires once per env step rather than per substep, and only when
        ``I4H_COUPLING_DEBUG`` is set, because reading the wrench back
        synchronises with the device.

        The force is the whole point of two-way coupling: it should sit near the
        catheter's weight while the wire hangs free and rise along the insertion
        axis as the tip loads against the vessel. A number that grows without
        settling means the lagged feedback is ringing, which is what
        ``drive_reaction_relaxation`` below 1 is for.
        """
        interval = cls._coupling_debug_interval
        if interval is None:
            return
        coupler = cls._coupler or getattr(NewtonManager, "_coupler", None)
        if coupler is None:
            return

        cls._coupling_debug_steps += 1
        if (cls._coupling_debug_steps - 1) % interval:
            return

        body, force, torque = coupler.report()
        print(
            f"[coupling] step {cls._coupling_debug_steps} body {body} "
            f"force={force} torque={torque}",
            flush=True,
        )

    @classmethod
    def reset_rod(cls, world_mask=None) -> None:
        """Reset rod state and selected coupling history at an episode boundary.

        ``world_mask`` has one entry per environment; ``None`` resets all.
        This is the explicit scene entry point on Isaac Lab ffff603. Body and
        joint poses remain those authored by the scene's rigid reset path.
        """
        rod = cls._rod_solver or getattr(NewtonManager, "_rod_solver", None)
        if rod is None:
            raise RuntimeError("The coupled solvers must be built before reset_rod().")
        state = cls.get_state_0()
        if state is None:
            raise RuntimeError("Newton state must be initialized before reset_rod().")
        rod.reset(state, world_mask=world_mask, flags=0)
        rigid = cls._rigid_solver or getattr(NewtonManager, "_rigid_solver", None)
        reset_rigid = getattr(rigid, "reset", None)
        if reset_rigid is not None:
            reset_rigid(state, world_mask=world_mask, flags=0)
        coupler = cls._coupler or getattr(NewtonManager, "_coupler", None)
        if coupler is not None:
            coupler.reset(world_mask=world_mask)

    @classmethod
    def _reset_solver_internals(cls, world_mask) -> None:
        """Extend newer Isaac Lab's reset boundary to both solvers and feedback."""
        if world_mask is not None:
            cls.reset_rod(world_mask)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._rigid_solver = None
        cls._rod_solver = None
        cls._coupler = None
        cls._coupled_cfg = None
        cls._rod_solver_cfg = None
        cls._drive_body_label = None
        cls._coupling_debug_interval = None
        cls._coupling_debug_steps = 0
        NewtonXPBDRodManager._solver_specific_clear()
        for attr in ("_rigid_solver", "_rod_solver", "_coupler", "_coupling_mode"):
            if hasattr(NewtonManager, attr):
                setattr(NewtonManager, attr, None)


__all__ = ["NewtonCoupledMJWarpXPBDRodManager"]
