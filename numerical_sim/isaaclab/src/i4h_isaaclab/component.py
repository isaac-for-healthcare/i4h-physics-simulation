# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""i4h physics as a component of a host Isaac Lab scene (docs/component-contracts.md, F3).

The host decides what exists and what success means; the component computes how the deformables respond. A host env
(a workflow's, with its own robots, tasks and recording) declares the physics with explicit parameters, no YAML and no
``I4hEnv``::

    component = I4hComponent(I4hComponentCfg(assets={...}, systems=[...], sim={"dt": 1 / 60, "substeps": 4},
                                             mounts={"probe": {"body": "tool_link"}}, coupling_mode="two_way"))
    component.configure(env_cfg.sim)          # in the env cfg: the host's MJWarpSolverCfg becomes the rigid solver
    with component.construction(device=env_cfg.sim.device, envs=env_cfg.scene.num_envs):
        env = HostEnv(env_cfg)                # Lab builds the scene; our particles join every world

    component.reset(env_ids)                  # in the host's reset, after its own state
    component.write_actions(physical)         # (E, A) in the columns of component.action_bounds()
    component.sync_views()                    # torch after Warp, before reading views outside the step
    q = component.positions("tissue")         # (E, 3n) zero-copy torch view, component frame

After every physics step the handle orders torch's current stream after Warp's (the manager's ``after_step``), so the
host's observation, reward and termination functions read the step that just ran. Views come from the current binding.
A hard rebuild (``sim.reset(soft=False)``) invalidates it: the handle rebinds
(``on_rebuild`` callbacks run) and views taken before must be taken again; a method called on a closed or invalidated
binding raises.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, field

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
    mounts: dict | None = None  # {instrument or catheter: {"body": label, "offset": (pos, quat_xyzw)}}
    coupling_mode: str = "one_way"

    @classmethod
    def from_case(cls, case, **kwargs) -> I4hComponentCfg:
        """The physical part of a case (a name, path or mapping, ``composition.read_case``): its assets, systems,
        instruments and ``sim``; an app's keys (procedure, render, inputs, ...) are left out."""
        from i4h_isaaclab.composition import read_case

        cfg = read_case(case)
        return cls(assets=dict(cfg.get("assets") or {}), systems=list(cfg.get("systems") or []),
                   instruments=dict(cfg.get("instruments") or {}), sim=dict(cfg.get("sim") or {}), **kwargs)

    def case(self) -> dict:
        case = {"sim": dict(self.sim), "assets": dict(self.assets), "systems": [dict(s) for s in self.systems]}
        if self.instruments:
            case["instruments"] = dict(self.instruments)
        return case


class I4hComponent:
    """The handle a host env holds: configuration, construction, and the current binding's views and reset."""

    def __init__(self, cfg: I4hComponentCfg | None = None, *, solver_cfg=None):
        if (cfg is None) == (solver_cfg is None):
            raise ValueError("I4hComponent takes an I4hComponentCfg or an already configured I4hSolverCfg")
        self.cfg = cfg
        self.sim = None  # the current NewtonSim (rebound by every solver build)
        self.solver_cfg = solver_cfg
        self._rebuild = []
        self._events = None

    # host configuration and construction ---------------------------------------------------------------------

    @classmethod
    def from_solver_cfg(cls, solver_cfg) -> I4hComponent:
        """A handle over a host cfg that ``configure`` already wrote (``env_cfg.sim.physics.solver_cfg``): an env built
        from a composed cfg constructs its handle here."""
        return cls(solver_cfg=solver_cfg)

    def configure(self, sim_cfg, *, rigid: bool = True, gravity_from_case: bool = True) -> None:
        """Make the host's ``SimulationCfg`` step this component: its Newton solver cfg (MuJoCo-Warp) becomes the rigid
        solver beside ``I4hSolver``, keeping its ``num_substeps`` (run inside the coupled step); every other host Newton
        setting (shape contact defaults, determinism, visual shapes, CUDA graph) is kept. ``rigid=True`` requires
        ``NewtonCfg(solver_cfg=MJWarpSolverCfg(...))``: any other host would lose its rigid simulation, so it raises;
        ``rigid=False`` is i4h physics alone. ``sim.dt`` defaults to the host's physics dt. A case that declares
        ``sim.gravity`` sets the host's (``gravity_from_case``, as ``I4hEnvCfg``); otherwise the two must agree."""
        if self.cfg is None:
            raise RuntimeError("this handle wraps a configured I4hSolverCfg; configure() needs an I4hComponentCfg")
        from i4h_isaaclab.newton_manager import I4hSolverCfg
        from isaaclab_newton.physics import NewtonCfg

        from i4h_isaaclab.coupling import check_rigid_solver_cfg

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
        self.solver_cfg = I4hSolverCfg(case=case, rigid_solver_cfg=rigid, coupling_mode=self.cfg.coupling_mode,
                                       mounts=self.cfg.mounts, loaders=self.cfg.loaders,
                                       systems=self.cfg.systems_registry, pose=self.cfg.pose,
                                       rigid_substeps=int(getattr(host, "num_substeps", 1)) if rigid is not None else 1)
        # Every host setting kept; only the solver and the substep split change (our substeps run inside the step).
        # class_type is derived from the solver cfg by NewtonCfg itself.
        kept = {f.name: getattr(host, f.name) for f in dataclasses.fields(host)
                if f.name not in ("class_type", "solver_cfg", "num_substeps")} if newton_host else {}
        sim_cfg.physics = NewtonCfg(solver_cfg=self.solver_cfg, num_substeps=1, **kept)

    @contextmanager
    def construction(self, *, device, envs):
        """Around the host env's construction: our particles join every replicated world, and the handle binds to each
        solver build (again after a hard rebuild)."""
        from i4h_isaaclab.newton_manager import construction_context

        if self.solver_cfg is None:
            raise RuntimeError("I4hComponent.configure(env_cfg.sim) first")
        with construction_context(self.solver_cfg, device=device, envs=envs, on_solver=self._bind):
            yield self

    def detach(self) -> None:
        """Stop rebinding (a host env's ``close``): the manager forgets this handle's build callback."""
        from i4h_isaaclab.newton_manager import I4hNewtonManager

        if self._bind in I4hNewtonManager.on_solver:
            I4hNewtonManager.on_solver.remove(self._bind)

    def on_rebuild(self, callback) -> None:
        """``callback(component)`` after every solver build: re-take the views there."""
        self._rebuild.append(callback)

    def _bind(self, sim) -> None:
        self.sim = sim
        self._events = None
        sim.after_step = self.sync_views  # after every manager step: torch reads (observations, rewards) see it
        for callback in list(self._rebuild):
            callback(self)

    def _live(self):
        if self.sim is None:
            raise RuntimeError("the component is not built yet (construct the host env inside construction())")
        self.sim.physics.check_valid()
        return self.sim

    # the binding ---------------------------------------------------------------------------------------------

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

    def positions(self, body: str | None = None, part: str | None = None) -> torch.Tensor:
        """``(E, 3n)`` zero-copy view of a body's (or one part's) positions, component frame; all integrated bodies
        concatenated is ``views.observation_views``' job, so a body is required when the case has several."""
        sim = self._live()
        observe = "" if body is None else body + ("." + part if part else "")
        found = views.observation_views(sim, observe)
        if len(found) != 1:
            raise ValueError(f"the component has {len(found)} integrated bodies: name one")
        return found[0]

    def state(self) -> dict[str, torch.Tensor]:
        """``{"<system id>.<product>": (E, k) view}`` of every system's declared ``state_products``."""
        sim = self._live()
        return dict(zip(views.state_names(sim), views.state_views(sim)))

    def action_views(self) -> list[torch.Tensor]:
        return views.action_views(self._live())

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

    def sync_views(self) -> None:
        """Torch's current stream waits for Warp's: before reading views outside the step."""
        self._live()
        self._order("out")

    def _order(self, direction: str) -> None:
        dev = self.scene.device
        if not dev.is_cuda:
            return
        if self._events is None:
            torch_device = torch.device(f"cuda:{dev.ordinal}")
            self._events = (torch.cuda.ExternalStream(wp.get_stream(dev).cuda_stream, device=torch_device),
                            torch.cuda.Event(), torch.cuda.Event(), torch_device)
        ext, ev_in, ev_out, torch_device = self._events
        if direction == "in":
            ev_in.record(torch.cuda.current_stream(torch_device))
            ext.wait_event(ev_in)
        else:
            ev_out.record(ext)
            torch.cuda.current_stream(torch_device).wait_event(ev_out)
