# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML case -> Scene (on a Newton model and state) + ordered systems -> I4hSolver -> Stepper.

Construction order: the loaders fill the scene's host records; ``Scene.allocate`` checks memory, builds the Newton
model (one world per env), creates its state and aliases the main particle block onto them; the systems are built on
the allocated scene (they cache views of its arrays); ``I4hSolver`` binds the model, scene, systems and state; the
``Stepper`` owns the timestep, loop counts and graphs."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml
from i4h_physics.checks import check_bool, check_float, check_int
from i4h_physics.loaders import LOADERS
from i4h_physics.scene import Scene
from i4h_physics.stepper import Stepper
from i4h_physics.systems import SYSTEMS

TOP_LEVEL_KEYS = ("sim", "assets", "systems")


def data_dir(data=None) -> Path:
    """``${DATA}`` in case YAML: explicit path, then ``I4H_DATA``. Without either it names the package's own ``data``
    folder, which ships nothing, so a case that needs external assets fails with the setup hint."""
    override = data or os.environ.get("I4H_DATA")
    return Path(override).expanduser().resolve() if override else Path(__file__).parent / "data"


def require_asset(path):
    """Validate an explicitly consumed asset; never discover a checkout or substitute data."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Expected file {path}. Set I4H_DATA or pass --data.\n"
                                "Repository checkout example: uv run i4h-sim CASE --data ./data")
    return path


def resolve_data_path(value, data=None):
    """Replace ``${DATA}`` and a leading ``~`` in every string of a nested YAML value."""
    if isinstance(value, dict):
        return {k: resolve_data_path(v, data) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve_data_path(v, data) for v in value]
    if isinstance(value, str) and ("${DATA}" in value or value.startswith("~")):
        return str(Path(value.replace("${DATA}", str(data_dir(data)))).expanduser())
    return value


def resolve_case(case, case_dirs=()) -> Path:
    """A path to a YAML file, or a name in explicitly supplied packaged case directories."""
    path = Path(case)
    if path.is_file():
        return path
    packaged = path
    for directory in case_dirs:
        packaged = directory / f"{case}.yaml"
        if packaged.is_file():
            return packaged
    raise FileNotFoundError(f"case {case!r} not found (looked for a file and for {packaged})")


def read_case(case, *, data=None, case_dirs=(), allowed_keys=TOP_LEVEL_KEYS) -> dict:
    """Read a name, path or mapping; a ``case:`` wrapper adds only an optional procedure to a physics case."""
    path = None if isinstance(case, dict) else resolve_case(case, case_dirs)
    cfg = case if path is None else yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or "sim" not in cfg:
        raise ValueError(f"case {case!r}: expected a mapping with a 'sim' section")
    unknown = sorted(set(cfg) - set(allowed_keys))
    if unknown:
        raise ValueError(f"case {case!r}: unknown top-level keys {unknown}; expected {list(allowed_keys)}")
    return resolve_data_path(cfg, data)


SIM_DEFAULTS = {"dt": 1.0 / 60.0, "substeps": 1, "iterations": 4, "cuda_graph": True}


def sim_settings(cfg: dict, debug=False) -> dict:
    """The case's ``sim:`` block without the scene keys, defaults filled in: ``dt``, ``substeps``, ``iterations``,
    ``cuda_graph``, ``debug`` (the one place those defaults are resolved)."""
    sim = dict(SIM_DEFAULTS)
    sim.update({k: v for k, v in dict(cfg.get("sim") or {}).items() if k not in ("device", "envs", "gravity")})
    if check_float("sim.dt", sim["dt"]) <= 0:
        raise ValueError("sim.dt must be finite and positive")
    sim["dt"] = float(sim["dt"])
    sim["substeps"] = check_int("sim.substeps", sim["substeps"], 1)
    sim["iterations"] = check_int("sim.iterations", sim["iterations"], 1)
    sim["cuda_graph"] = check_bool("sim.cuda_graph", sim["cuda_graph"])
    sim["debug"] = check_bool("sim.debug", sim.pop("debug", False)) or check_bool("debug", debug)  # the case's or the caller's
    return sim


def check_dt(dt: float, case_dt: float) -> None:
    """A host's timestep (Isaac Lab's ``sim.dt``) must match the case's ``sim.dt``, whose substep size is tuned per case:
    within 1e-4 relative, so a rounded 1/60 passes; the host's value is then the one the solver is configured with."""
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in (dt, case_dt)):
        raise ValueError("timesteps must be finite and positive")
    if abs(float(dt) - float(case_dt)) > 1.0e-4 * abs(float(case_dt)):
        raise ValueError(f"the physics dt {dt} differs from the case's sim.dt {case_dt}; the substep size is tuned per case")


def load_scene(cfg: dict, device=None, envs=None, *, loaders=None) -> Scene:
    """The scene's host records: the case's assets loaded, nothing allocated on the device yet."""
    loaders = LOADERS if loaders is None else loaders
    sim = cfg.get("sim") or {}
    scene = Scene(sim.get("device", "cuda:0") if device is None else device, sim.get("envs", 1) if envs is None else envs,
                  sim.get("gravity"))
    for name, asset in (cfg.get("assets") or {}).items():
        asset = dict(asset)
        kind = asset.pop("type")
        if kind not in loaders:
            raise ValueError(f"asset {name!r}: unknown type {kind!r}; known: {sorted(loaders)}")
        try:
            loaders[kind](scene, name, **asset)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{error}\nSet I4H_DATA to the production asset directory, or pass --data.\n"
                "Repository checkout example: uv run i4h-sim CASE --data ./data"
            ) from error
    return scene


def load_case(case, device=None, envs=None, debug=False, *, loaders=None, systems=None, data=None, case_dirs=(), allowed_keys=TOP_LEVEL_KEYS) -> Stepper:
    """Build physics only from a case name, path or mapping. Assessment is constructed by the caller."""
    cfg = read_case(case, data=data, case_dirs=case_dirs, allowed_keys=allowed_keys)
    scene = load_scene(cfg, device, envs, loaders=loaders)
    scene.allocate()
    sim = Stepper.create(scene, build_systems(scene, cfg, constructors=systems), **sim_settings(cfg, debug))
    return sim


def build_systems(scene: Scene, cfg: dict, *, constructors=None) -> list:
    """The case's ``systems:`` in list order on an allocated scene; YAML id strings in ``refs`` become the earlier
    instances."""
    constructors = SYSTEMS if constructors is None else constructors
    systems, by_id = [], {}
    for entry in cfg.get("systems") or []:
        entry = dict(entry)
        kind = entry.pop("type")
        sid = entry.pop("id", kind)
        if kind not in constructors:
            raise ValueError(f"system {sid!r}: unknown type {kind!r}; known: {sorted(constructors)}")
        if sid in by_id:
            raise ValueError(f"duplicate system id {sid!r}")
        if "enabled" in entry:
            entry["enabled"] = check_bool(f"{sid}.enabled", entry["enabled"])
        for name in constructors[kind].refs:  # id strings of earlier systems become the instances
            if name not in entry:
                continue  # the dataclass constructor reports a missing required field
            ref = entry[name]
            if not isinstance(ref, str) or ref not in by_id:
                raise ValueError(f"system {sid!r}: {name!r} must name an earlier system; got {ref!r}, known: {sorted(by_id)}")
            entry[name] = by_id[ref]
        system = constructors[kind](scene, id=sid, **entry)
        systems.append(system)
        by_id[sid] = system
    check_particles(scene, systems)
    return systems


def check_particles(scene: Scene, systems: list) -> None:
    """Exactly one ``particles`` system integrates the main block: with none, the other systems' bodies never move;
    with two, gravity and the velocity update run twice per substep. A case with no systems at all is a static
    scene."""
    from i4h_physics.systems.mechanics.particles import Particles

    ids = [s.id for s in systems if isinstance(s, Particles)]
    if len(ids) > 1:
        raise ValueError(f"the case has {len(ids)} particles systems {ids}; exactly one integrates the main block")
    if not ids and systems and scene.main.width:
        raise ValueError("the case has systems but no particles system; add {type: particles} to integrate the bodies")
