# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec
from pathlib import Path

import i4h_physics.config as physical
import yaml


def domains():
    packages = []
    for name in ("i4h_surgical", "i4h_endoluminal"):
        if find_spec(name) is not None:
            packages.append(__import__(name, fromlist=["__file__"]))
    return packages

def case_dirs():
    return [Path(package.__file__).parent / "cases" for package in domains()]

def resolve_case(case):
    return physical.resolve_case(case, case_dirs())

HOST_KEYS = (*physical.TOP_LEVEL_KEYS, "instruments", "inputs", "render", "haptics", "procedure", "ui")

def read_case(case, *, data=None):
    """A case name, path or mapping. A procedure (or a ``case:`` reference adding one) needs i4h-assessment
    (the ``i4h-isaaclab[apps]`` extra); without it, physics cases still read."""
    try:
        from i4h_assessment.config import read_assessment_case
    except ModuleNotFoundError as error:
        if error.name != "i4h_assessment":
            raise
        raw = case if isinstance(case, dict) else yaml.safe_load(physical.resolve_case(case, case_dirs()).read_text())
        if isinstance(raw, dict) and ("procedure" in raw or "case" in raw):
            raise ModuleNotFoundError(f"case {case!r} has a procedure: install i4h-assessment (the i4h-isaaclab[apps] extra)",
                                      name="i4h_assessment") from error
        return physical.read_case(raw, data=data, allowed_keys=HOST_KEYS)
    return read_assessment_case(case, data=data, case_dirs=case_dirs(), allowed_keys=HOST_KEYS)


def maps(cfg, loaders=None, systems=None):
    """The case with its instruments as assets, and the loader and system maps: the physics package's plus, when
    ``loaders``/``systems`` are given, exactly those (a component's explicit maps), else every installed domain's."""
    explicit = loaders is not None or systems is not None
    extra_loaders, extra_systems = dict(loaders or {}), dict(systems or {})
    loaders, systems = dict(physical.LOADERS), dict(physical.SYSTEMS)
    if explicit:
        loaders.update(extra_loaders)
        systems.update(extra_systems)
    else:
        for package in domains():
            loaders.update(__import__(package.__name__ + ".loaders", fromlist=["LOADERS"]).LOADERS)
            systems.update(__import__(package.__name__ + ".systems", fromlist=["SYSTEMS"]).SYSTEMS)
    config = dict(cfg)
    config["assets"] = dict(cfg.get("assets") or {})
    for name, entry in (cfg.get("instruments") or {}).items():
        config["assets"][name] = {"type": "instrument", **entry}
    return config, loaders, systems

def load_scene(cfg, device=None, envs=None, *, loaders=None, systems=None):
    cfg, loaders, _ = maps(cfg, loaders, systems)
    return physical.load_scene(cfg, device, envs, loaders=loaders)

def build_systems(scene, cfg, *, loaders=None, systems=None):
    cfg, _, systems = maps(cfg, loaders, systems)
    return physical.build_systems(scene, cfg, constructors=systems)

def load_case(case, device=None, envs=None, debug=False, *, data=None, loaders=None, systems=None):
    cfg, loaders, systems = maps(read_case(case, data=data), loaders, systems)
    return physical.load_case(cfg, device, envs, debug, loaders=loaders, systems=systems, allowed_keys=HOST_KEYS)

def build_procedure(cfg, sim, *, horizon_s=None):
    if not cfg.get("procedure"):
        return None
    from i4h_assessment.executor import Procedure  # the [apps] extra; only cases with a procedure need it

    measurements = {}
    for package in domains():
        if package.__name__ == "i4h_surgical":
            from i4h_surgical.assessment import GraspTarget
            measurements["grasp_target"] = GraspTarget
        else:
            from i4h_endoluminal.assessment import ReachTarget
            measurements["reach_target"] = ReachTarget
    return Procedure(cfg["procedure"], sim, measurements=measurements, horizon_s=horizon_s)

check_dt = physical.check_dt
sim_settings = physical.sim_settings
resolve_data_path = physical.resolve_data_path
data_dir = physical.data_dir
