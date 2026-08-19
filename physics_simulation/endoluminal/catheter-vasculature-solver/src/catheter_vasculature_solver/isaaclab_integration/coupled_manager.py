# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Path B: coupled MJWarp (rigid) + SolverXPBDRod (catheter) NewtonManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_newton.physics import NewtonManager

from .rod_builder import register_xpbd_rod_builder_attributes
from .solver_cfg import CoupledMJWarpXPBDRodSolverCfg, XPBDRodSolverCfg
from .xpbd_rod_manager import NewtonXPBDRodManager

if TYPE_CHECKING:
    from newton import Model


class NewtonCoupledMJWarpXPBDRodManager(NewtonManager):
    """Couple a rigid MJWarp solve with an XPBD Cosserat catheter rod.

    Mirrors the Isaac Lab MJWarp + VBD coupled-manager pattern:

    * Nested configs construct two sub-solvers.
    * :meth:`_step_solver` owns contact + substep ordering.
    * Base :class:`~isaaclab_newton.physics.NewtonManager` still owns model
      lifecycle, Fabric sync, resets, and the outer substep loop.

    Substep algorithm (``coupling_mode="two_way"``)::

        1. Clear rigid / particle force accumulators
        2. Run Newton collision pipeline (shared contacts)
        3. Optional: inject particle→body reactions into body_f
        4. Step MJWarp rigid solver
        5. Preserve contacts; clear particle forces written by rigid step
        6. Step SolverXPBDRod
        7. Optional: track / vessel containment hooks on particle_q

    Class-level slots (in addition to NewtonManager)::

        _rigid_solver  – MJWarp / SolverMuJoCo instance
        _rod_solver    – SolverXPBDRod instance
        _coupling_mode – \"one_way\" | \"two_way\"
        _rod_solver_cfg / _coupled_cfg – retained configs for hooks
    """

    _rigid_solver: Any = None
    _rod_solver: Any = None
    _coupling_mode: str = "one_way"
    _coupled_cfg: CoupledMJWarpXPBDRodSolverCfg | None = None
    _rod_solver_cfg: XPBDRodSolverCfg | None = None

    # Optional GPU buffers registered by the env for catheter-style hooks
    _track_start: Any = None
    _track_dir: Any = None
    _track_length: float = 0.0
    _vessel_mesh: Any = None

    @classmethod
    def _register_builder_attributes(cls, builder) -> None:
        register_xpbd_rod_builder_attributes(builder)
        # If MJWarp also registers attrs via its manager, call that path too.
        rigid_mgr = cls._resolve_rigid_manager_type()
        if rigid_mgr is not None and hasattr(rigid_mgr, "_register_builder_attributes"):
            rigid_mgr._register_builder_attributes(builder)

    @classmethod
    def _resolve_rigid_manager_type(cls):
        """Return NewtonMJWarpManager class if importable."""
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
        """Build MJWarp solver via the in-tree manager when possible."""
        rigid_mgr = cls._resolve_rigid_manager_type()
        if rigid_mgr is not None and hasattr(rigid_mgr, "build_nested_solver"):
            return rigid_mgr.build_nested_solver(model, rigid_cfg)

        # Fallback: construct SolverMuJoCo directly from common cfg fields.
        try:
            from newton.solvers import SolverMuJoCo
        except ImportError as e:
            raise ImportError(
                "Coupled Path B needs SolverMuJoCo (MJWarp) for the rigid half. "
                f"Original error: {e}"
            ) from e

        kwargs: dict[str, Any] = {}
        for key in ("iterations", "ls_iterations", "njmax", "nconmax"):
            if hasattr(rigid_cfg, key):
                kwargs[key] = getattr(rigid_cfg, key)
        return SolverMuJoCo(model, **kwargs)

    @classmethod
    def _build_solver(cls, model: "Model", solver_cfg: CoupledMJWarpXPBDRodSolverCfg) -> None:
        NewtonManager._coupled_cfg = solver_cfg  # type: ignore[attr-defined]
        cls._coupled_cfg = solver_cfg
        cls._rod_solver_cfg = solver_cfg.rod_solver_cfg
        cls._coupling_mode = str(solver_cfg.coupling_mode)

        cls._rigid_solver = cls._build_rigid_solver(model, solver_cfg.rigid_solver_cfg)
        cls._rod_solver = NewtonXPBDRodManager.build_nested_solver(model, solver_cfg.rod_solver_cfg)

        # Expose the rod solver as the primary _solver for diagnostics / Fabric
        # readers that expect a single handle; stepping uses _step_solver.
        NewtonManager._solver = cls._rod_solver
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

        # Stash class-level copies for _step_solver (base reads NewtonManager.*)
        NewtonManager._rigid_solver = cls._rigid_solver  # type: ignore[attr-defined]
        NewtonManager._rod_solver = cls._rod_solver  # type: ignore[attr-defined]
        NewtonManager._coupling_mode = cls._coupling_mode  # type: ignore[attr-defined]

    @classmethod
    def _clear_force_buffers(cls, state) -> None:
        """Zero particle / body force accumulators when present."""
        for name in ("particle_f", "body_f", "joint_f"):
            buf = getattr(state, name, None)
            if buf is not None and hasattr(buf, "zero_"):
                buf.zero_()
            elif buf is not None and hasattr(buf, "fill_"):
                buf.fill_(0)

    @classmethod
    def _apply_soft_to_rigid_reactions(cls, contacts, state) -> None:
        """Inject particle–body contact reactions into ``body_f`` (two-way).

        Concrete impulse mapping depends on the Newton contact buffer layout
        in your Lab/Newton revision. Override this in a subclass or fill in
        once your contact schema is fixed; default is a no-op placeholder so
        one_way coupling still runs.
        """
        del contacts, state

    @classmethod
    def _run_collision_pipeline(cls, state_0, state_1, control) -> Any:
        """Invoke NewtonManager collision detection when allocated."""
        pipeline = getattr(NewtonManager, "_collision_pipeline", None)
        contacts = getattr(NewtonManager, "_contacts", None)
        if pipeline is None:
            return contacts
        # API varies slightly across Lab revisions; prefer a step() / collide().
        if hasattr(pipeline, "step"):
            return pipeline.step(state_0, state_1, control, contacts)
        if hasattr(pipeline, "collide"):
            return pipeline.collide(NewtonManager._model, state_0, contacts)
        return contacts

    @classmethod
    def _apply_post_rod_hooks(cls, state) -> None:
        """Optional track / vessel hooks after the rod substep.

        Register buffers via :meth:`register_vessel_track` from the env.
        Full vessel SDF kernels stay in :class:`CathRodSolver`; this hook is
        the integration seam for Path B.
        """
        cfg = cls._coupled_cfg
        if cfg is None:
            return
        if cfg.apply_track_guidance and cls._track_start is not None:
            # Env-specific: project non-tip particles onto the insertion axis.
            # Left as an extension point — call into the solver kernels when wired.
            pass
        if cfg.apply_vessel_containment_hooks and cls._vessel_mesh is not None:
            pass
        del state

    @classmethod
    def register_vessel_track(
        cls,
        *,
        track_start=None,
        track_dir=None,
        track_length: float = 0.0,
        vessel_mesh=None,
    ) -> None:
        """Attach vessel mesh / insertion track for optional post-step hooks."""
        cls._track_start = track_start
        cls._track_dir = track_dir
        cls._track_length = float(track_length)
        cls._vessel_mesh = vessel_mesh

    @classmethod
    def _step_solver(cls, state_0, state_1, control, substep_dt: float) -> None:
        """One coupled substep: contacts → rigid → rod (+ optional hooks)."""
        rigid = cls._rigid_solver or getattr(NewtonManager, "_rigid_solver", None)
        rod = cls._rod_solver or getattr(NewtonManager, "_rod_solver", None)
        if rigid is None or rod is None:
            raise RuntimeError(
                "NewtonCoupledMJWarpXPBDRodManager sub-solvers are not built. "
                "Ensure _build_solver ran during NewtonManager.initialize()."
            )

        mode = cls._coupling_mode
        contacts = getattr(NewtonManager, "_contacts", None)

        # 1) Clear force accumulators on the input state.
        cls._clear_force_buffers(state_0)

        # 2) Shared Newton collision pass.
        contacts = cls._run_collision_pipeline(state_0, state_1, control) or contacts

        # 3) Soft → rigid reactions (two-way only).
        if mode == "two_way":
            cls._apply_soft_to_rigid_reactions(contacts, state_0)

        # 4) Advance rigid solver (MJWarp).
        rigid.step(state_0, state_1, control, contacts, substep_dt)

        # 5) Preserve contacts; clear particle forces written during rigid step.
        if hasattr(state_1, "particle_f") and state_1.particle_f is not None:
            if hasattr(state_1.particle_f, "zero_"):
                state_1.particle_f.zero_()

        # 6) Advance catheter rod (SolverXPBDRod).
        # Swap convention: rigid wrote into state_1; rod reads state_1 → state_0
        # then we leave state_0 as the latest (matches double-buffer ping-pong
        # used by NewtonManager when _use_single_state is False — the base
        # loop typically swaps after _step_solver; keep both buffers updated).
        rod.step(state_1, state_0, control, contacts, substep_dt)

        # Copy rod-updated particle state back so Fabric sees one coherent state.
        # Many Lab revisions treat state_1 as the next write target; re-sync
        # particle_q/qd if both buffers exist.
        if hasattr(state_0, "particle_q") and hasattr(state_1, "particle_q"):
            if state_0.particle_q is not None and state_1.particle_q is not None:
                state_1.particle_q.assign(state_0.particle_q)
            if (
                getattr(state_0, "particle_qd", None) is not None
                and getattr(state_1, "particle_qd", None) is not None
            ):
                state_1.particle_qd.assign(state_0.particle_qd)

        # 7) Optional catheter-style post hooks.
        cls._apply_post_rod_hooks(state_1)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._rigid_solver = None
        cls._rod_solver = None
        cls._coupled_cfg = None
        cls._rod_solver_cfg = None
        cls._track_start = None
        cls._track_dir = None
        cls._vessel_mesh = None
        for attr in ("_rigid_solver", "_rod_solver", "_coupling_mode"):
            if hasattr(NewtonManager, attr):
                setattr(NewtonManager, attr, None)


__all__ = ["NewtonCoupledMJWarpXPBDRodManager"]
