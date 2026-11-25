# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""i4h physics as a component of a host Isaac Lab scene (physics/component-contracts.md, F3).

The host decides what exists and what success means; the component computes how the deformables respond. A host env
(a workflow's, with its own robots, tasks and recording) declares the physics with explicit parameters, no YAML and no
``I4hEnv``::

    component = I4hComponent(I4hComponentCfg(assets={...}, systems=[...], sim={"dt": 1 / 60, "substeps": 4},
                                             mounts={"probe": {"body": "tool_link"}}, coupling_mode="two_way"))
    component.configure(env_cfg.sim)          # in the env cfg: the host's MJWarpSolverCfg becomes the rigid solver
    with component.construction(device=env_cfg.sim.device, envs=env_cfg.scene.num_envs):
        env = HostEnv(env_cfg)                # Lab builds the scene; our particles join every world

    component.reset(env_ids)                  # in the host's reset, after its own state
    component.initialize_particle_history(env_ids)  # after host reset events that edit particle positions
    component.reset_procedure(env_ids)        # with a procedure: after the host's reset events
    component.write_actions(physical)         # (E, A) in the columns of component.action_bounds()
    component.sync_views()                    # torch after Warp, before reading views outside the step
    q = component.positions("tissue")         # (E, 3n) zero-copy torch view, component frame

After every physics step the handle orders torch's current stream after Warp's (the manager's ``after_step``), so the
host's observation, reward and termination functions read the step that just ran. Views come from the current binding.
A hard rebuild (``sim.reset(soft=False)``) invalidates it: the handle rebinds
(``on_rebuild`` callbacks run) and views taken before must be taken again; a method called on a closed or invalidated
binding raises.

A cfg with a ``procedure`` (``from_case(case, procedure=True)``) makes the component own the case's assessment: every
solver build constructs it (``composition.build_procedure`` with the host's ``horizon_s``) as ``component.procedure``,
the manager updates it after every physics step, and the host resets it with ``reset_procedure`` after its reset events.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields

import numpy as np
import torch
import warp as wp
from i4h_isaaclab import views


@dataclass
class I4hComponentCfg:
    """The physical description: assets (by loader type), systems (by system type, in step order), instruments, the
    ``sim`` settings (``dt``, ``substeps``, ``iterations``, ``gravity``; device and env count are the host's), the
    component's ``pose`` in each env, explicit ``loaders``/``systems_registry`` maps (with either, no installed-domain
    discovery: the physics package's own plus exactly these), and the coupling with the host's bodies."""

    assets: dict
    systems: list
    instruments: dict = field(default_factory=dict)
    sim: dict = field(default_factory=dict)
    pose: tuple | None = None  # ((x, y, z), (qx, qy, qz, qw)) in the env frame
    loaders: dict | None = None  # {asset type: loader(scene, name, **params)}
    systems_registry: dict | None = None  # {system type: System subclass}
    mounts: dict | None = (
        None  # {instrument or catheter: {"body": label, "offset": (pos, quat_xyzw)}}
    )
    coupling_mode: str = "one_way"
    procedure: dict | None = (
        None  # the case's assessment (i4h-assessment), owned by the component when given
    )

    @classmethod
    def from_case(cls, case, *, procedure: bool = False, **kwargs) -> I4hComponentCfg:
        """The physical part of a case (a name, path or mapping, ``composition.read_case``): its assets, systems,
        instruments and ``sim``, and with ``procedure`` its ``procedure:``; an app's other keys (render, inputs, ...)
        are left out."""
        from i4h_host.composition import read_case

        cfg = read_case(case)
        if procedure:
            kwargs.setdefault(
                "procedure", dict(cfg["procedure"]) if cfg.get("procedure") else None
            )
        return cls(
            assets=dict(cfg.get("assets") or {}),
            systems=list(cfg.get("systems") or []),
            instruments=dict(cfg.get("instruments") or {}),
            sim=dict(cfg.get("sim") or {}),
            **kwargs,
        )

    def case(self) -> dict:
        case = {
            "sim": dict(self.sim),
            "assets": dict(self.assets),
            "systems": [dict(s) for s in self.systems],
        }
        if self.instruments:
            case["instruments"] = dict(self.instruments)
        if self.procedure:
            case["procedure"] = dict(self.procedure)
        return case


class I4hComponent:
    """The handle a host env holds: configuration, construction, and the current binding's views and reset."""

    def __init__(self, cfg: I4hComponentCfg | None = None, *, solver_cfg=None):
        if (cfg is None) == (solver_cfg is None):
            raise ValueError(
                "I4hComponent takes an I4hComponentCfg or an already configured I4hSolverCfg"
            )
        self.cfg = cfg
        self.sim = None  # current PhysicsBinding, invalidated and replaced on every solver build
        self._construction = None
        self.solver_cfg = solver_cfg
        self.procedure = None  # the case's procedure, built with every solver build when the case has one
        self.horizon_s = None  # the host's episode horizon (max_episode_length * step_dt), set before construction
        self._rebuild = []
        self._events = None

    # host configuration and construction ---------------------------------------------------------------------

    @classmethod
    def from_solver_cfg(cls, solver_cfg) -> I4hComponent:
        """A handle over a host cfg that ``configure`` already wrote (``env_cfg.sim.physics.solver_cfg``): an env built
        from a composed cfg constructs its handle here."""
        return cls(solver_cfg=solver_cfg)

    def configure(
        self, sim_cfg, *, rigid: bool = True, gravity_from_case: bool = True
    ) -> None:
        """Make the host's ``SimulationCfg`` step this component: its Newton solver cfg (MuJoCo-Warp) becomes the rigid
        solver beside ``I4hSolver``, keeping its ``num_substeps`` (run inside the coupled step); every other host Newton
        setting (shape contact defaults, determinism, visual shapes, CUDA graph) is kept. ``rigid=True`` requires
        ``NewtonCfg(solver_cfg=MJWarpSolverCfg(...))``: any other host would lose its rigid simulation, so it raises;
        ``rigid=False`` is i4h physics alone. ``sim.dt`` defaults to the host's physics dt. A case that declares
        ``sim.gravity`` sets the host's (``gravity_from_case``, as ``I4hEnvCfg``); otherwise the two must agree."""
        if self.cfg is None:
            raise RuntimeError(
                "this handle wraps a configured I4hSolverCfg; configure() needs an I4hComponentCfg"
            )
        from i4h_isaaclab.coupling import check_rigid_solver_cfg
        from i4h_isaaclab.newton_manager import I4hSolverCfg, physics_cfg
        from isaaclab_newton.physics import NewtonCfg

        host = sim_cfg.physics
        newton_host = isinstance(host, NewtonCfg)
        host_solver = getattr(host, "solver_cfg", None) if newton_host else None
        if isinstance(host_solver, I4hSolverCfg):
            raise ValueError("the host's physics already steps an i4h component")
        if rigid and (not newton_host or str(getattr(host_solver, "solver_type", "")) != "mujoco_warp"):
            raise ValueError(f"I4hComponent.configure(rigid=True) needs the host's physics to be NewtonCfg(solver_cfg="
                             f"MJWarpSolverCfg(...)), got {type(host).__name__}"
                             f"{'(' + type(host_solver).__name__ + ')' if newton_host else ''}; pass rigid=False to run "
                             "i4h physics alone")
        if rigid:
            check_rigid_solver_cfg(host_solver, self.cfg.coupling_mode)
        if newton_host and host.collision_cfg is not None:
            raise ValueError("NewtonCfg.collision_cfg is not supported with an i4h component: the coupled step runs no "
                             "Newton collision pipeline (MuJoCo-Warp collides with its own contacts)")
        rigid = host_solver if rigid else None
        case = self.cfg.case()
        case["sim"].setdefault("dt", float(sim_cfg.dt))
        if gravity_from_case and case["sim"].get("gravity") is not None:
            sim_cfg.gravity = tuple(float(g) for g in case["sim"]["gravity"])
        self.solver_cfg = I4hSolverCfg(
            case=case,
            rigid_solver_cfg=rigid,
            coupling_mode=self.cfg.coupling_mode,
            mounts=self.cfg.mounts,
            loaders=self.cfg.loaders,
            systems=self.cfg.systems_registry,
            pose=self.cfg.pose,
            rigid_substeps=int(getattr(host, "num_substeps", 1))
            if rigid is not None
            else 1,
        )
        kwargs = (
            {
                f.name: getattr(host, f.name)
                for f in fields(host)
                if f.name not in ("class_type", "solver_cfg", "num_substeps")
            }
            if isinstance(host, NewtonCfg)
            else {}
        )
        sim_cfg.physics = physics_cfg(
            self.solver_cfg, **kwargs
        )  # our substeps run inside
        self.solver_cfg = (
            sim_cfg.physics.solver_cfg
        )  # configclass copied the declaration; bind its actual host copy

    @contextmanager
    def construction(self, *, device, envs):
        """Around the host env's construction: our particles join every replicated world, and the handle binds to each
        solver build (again after a hard rebuild)."""
        from i4h_isaaclab.newton_manager import construction_context

        if self.solver_cfg is None:
            raise RuntimeError("I4hComponent.configure(env_cfg.sim) first")
        if self._construction is not None:
            self._construction.close()
        with construction_context(
            self.solver_cfg, device=device, envs=envs, on_solver=self._bind
        ) as owner:
            self._construction = owner
            yield self

    def detach(self) -> None:
        """Stop rebinding (a host env's ``close``): the manager forgets this handle's build callback."""
        if self._construction is not None:
            self._construction.close()
            self._construction = None

    def on_rebuild(self, callback) -> None:
        """``callback(component)`` on fresh construction: re-take views and initialize positions.

        After these callbacks, history is initialized from the new live positions.
        Restoring an independent State's history belongs after construction.
        """
        self._rebuild.append(callback)

    def on_post_physics(self, callback) -> None:
        """Register allocation-free device work during on_rebuild, before host capture.

        Callbacks run in registration order after each completed physics step,
        including both steps of a decimated action. Re-register on hard rebuild.
        """
        if not getattr(self, "_binding_callbacks_open", False):
            raise RuntimeError("register post-physics callbacks during on_rebuild, before graph capture")
        self._live().post_physics.append(callback)

    def _bind(self, sim) -> None:
        self.sim = sim
        sim.post_physics = []
        self._events = None
        self.procedure = None
        case = self.solver_cfg.case or {}
        if case.get("procedure"):
            from i4h_host.composition import build_procedure

            self.procedure = build_procedure(case, sim, horizon_s=self.horizon_s)
        sim.procedure = (
            self.procedure
        )  # updated by the manager after every physics step
        sim.after_step = (
            self.sync_views
        )  # after every manager step: torch reads (observations, rewards) see it
        if self._rebuild:
            self.sync_views()
        self._binding_callbacks_open = True
        try:
            for callback in list(self._rebuild):
                callback(self)
        finally:
            self._binding_callbacks_open = False
        if self._rebuild:
            self.initialize_particle_history()

    def _live(self):
        if self.sim is None:
            raise RuntimeError(
                "the component is not built yet (construct the host env inside construction())"
            )
        self.sim.physics.check_valid()
        return self.sim

    # the binding ---------------------------------------------------------------------------------------------

    @property
    def physics(self):
        """Current physics binding; observers must bind again after a hard rebuild."""
        return self._live().physics

    @property
    def systems(self) -> list:
        return self._live().systems

    @property
    def scene(self):
        return self._live().scene

    @property
    def particle_range(self) -> tuple[int, int, int, int]:
        """``(base, stride, col, width)``: our rows of world ``e`` are ``[base + e*stride + col, ... + width)``."""
        scene = self.scene
        base, stride, col = scene.main.place(scene.envs)
        return int(base), int(stride), int(col), int(scene.main.width)

    def positions(
        self, body: str | None = None, part: str | None = None
    ) -> torch.Tensor:
        """``(E, 3n)`` zero-copy view of a body's (or one part's) positions, component frame; all integrated bodies
        concatenated is ``views.observation_views``' job, so a body is required when the case has several."""
        sim = self._live()
        observe = "" if body is None else body + ("." + part if part else "")
        found = views.observation_views(sim, observe)
        if len(found) != 1:
            raise ValueError(
                f"the component has {len(found)} integrated bodies: name one"
            )
        return found[0]

    def state(self) -> dict[str, torch.Tensor]:
        """``{"<system id>.<product>": (E, k) view}`` of every system's declared ``state_products``."""
        sim = self._live()
        return dict(zip(views.state_names(sim), views.state_views(sim)))

    def action_views(self) -> list[torch.Tensor]:
        return views.action_views(self._live())

    def command_views(self, system_id):
        """Cached writable Torch slices; use the existing stream-ordering boundary."""
        binding = self._live().physics
        binding.command_views(system_id)
        return binding.solver.command_view(system_id).torch_views()

    def action_bounds(self, workspace: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Physical ``(A,)`` bounds of the concatenated control columns, as each control declares them."""
        return views.action_bounds(self._live(), workspace)

    def write_actions(self, physical: torch.Tensor) -> None:
        """A physical ``(E, A)`` action into the controls' arrays (ordered before Warp's next launch)."""
        sim = self._live()
        views.write_actions(views.action_views(sim), physical)
        self._order("in")

    def reset(self, env_ids=None, *, mask=None) -> None:
        """The selected worlds' physical state, now (``env_ids`` or an ``(E,)`` bool ``mask``, host or device)."""
        self._order("in")
        self._live().physics.reset(env_ids, mask=mask)

    def save(self, dst_state, dst_control, env_ids=None, *, mask=None) -> None:
        """Copy the selected worlds' i4h-owned rows into another State and Control of the host's Model."""
        self._order("in")
        self._live().physics.save(dst_state, dst_control, env_ids, mask=mask)

    def restore(self, src_state, src_control, env_ids=None, *, mask=None) -> None:
        """Copy saved i4h-owned rows back into the live State and Control. Rigid body state, host time, assessment
        and procedure state are the host's to restore; no reset or cache refresh runs here.

        Order matters: Isaac Lab runs the solver's cache reset for every world whose state was written through its
        asset API, at the next forward or step. Write the host's own state first, call the simulation's
        ``forward()``, and restore the i4h rows last; otherwise that cache reset follows the restore and replaces
        graph containment's saved rebuild request with a forced rebuild."""
        self._order("in")
        self._live().physics.restore(src_state, src_control, env_ids, mask=mask)

    def capture_continuation(self, env_ids=None):
        """Portable selected-world State/Control arrays, including particle histories and pending reactions.

        Each first-axis row is an independent episode. Host assets, task history,
        renderer state and source/configuration identity remain the host's contract.
        """
        self.sync_views()
        return self.physics.solver.continuation.export_worlds(env_ids)

    def validate_continuation(self, bundle, env_ids=None):
        """Validate all numerical fields before the host begins restoration."""
        return self.physics.solver.continuation.validate_worlds(bundle, env_ids)

    def restore_continuation(self, bundle, env_ids=None):
        """Restore portable rows after host assets and caches, without initializing history."""
        self.order_inputs()
        self.physics.solver.continuation.import_worlds(bundle, env_ids)
        self.sync_views()

    def initialize_particle_history(self, env_ids=None, *, mask=None) -> None:
        """After host position edits, initialize selected history on Warp's stream.

        Call after teleports through ``positions()`` or native State views and
        before the next step. Other state, including velocity, remains unchanged.
        """
        self._order("in")
        self._live().physics.initialize_particle_history(env_ids, mask=mask)

    def reset_procedure(self, env_ids=None, *, mask=None) -> None:
        """The selected worlds' procedure (no-op without one), after the host's reset events sampled the new state."""
        self._live()
        if self.procedure is not None:
            self.procedure.reset(
                env_ids, mask=mask
            )  # select_envs orders Warp after torch's ids or mask

    def sync_views(self) -> None:
        """Torch's current stream waits for Warp's: before reading views outside the step."""
        self._live()
        self._order("out")

    def order_inputs(self) -> None:
        """Order Warp after Torch's current stream, before a Warp write that follows Torch writes into shared views."""
        self._order("in")

    def _order(self, direction: str) -> None:
        dev = self.scene.device
        if not dev.is_cuda:
            return
        if self._events is None:
            torch_device = torch.device(f"cuda:{dev.ordinal}")
            self._events = (
                torch.cuda.ExternalStream(
                    wp.get_stream(dev).cuda_stream, device=torch_device
                ),
                torch.cuda.Event(),
                torch.cuda.Event(),
                torch_device,
            )
        ext, ev_in, ev_out, torch_device = self._events
        if direction == "in":
            ev_in.record(torch.cuda.current_stream(torch_device))
            ext.wait_event(ev_in)
        else:
            ev_out.record(ext)
            torch.cuda.current_stream(torch_device).wait_event(ev_out)
