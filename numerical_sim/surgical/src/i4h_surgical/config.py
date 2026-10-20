# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml
from i4h_physics import config as physical

from .loaders import LOADERS
from .systems import SYSTEMS

CASES_DIR = Path(__file__).parent / "cases"

HOST_KEYS = (*physical.TOP_LEVEL_KEYS, "instruments", "inputs", "render", "haptics", "procedure", "ui")

def read_case(case, *, data=None):
    """A case name, path or mapping. A procedure (or a ``case:`` reference adding one) needs i4h-assessment
    (the ``i4h-surgical[assessment]`` extra); without it, physics cases still read."""
    try:
        from i4h_assessment.config import read_assessment_case
    except ModuleNotFoundError as error:
        if error.name != "i4h_assessment":
            raise
        raw = case if isinstance(case, dict) else yaml.safe_load(physical.resolve_case(case, (CASES_DIR,)).read_text())
        if isinstance(raw, dict) and ("procedure" in raw or "case" in raw):
            raise ModuleNotFoundError(f"case {case!r} has a procedure: install i4h-assessment (the i4h-surgical[assessment] extra)",
                                      name="i4h_assessment") from error
        return physical.read_case(raw, data=data, allowed_keys=HOST_KEYS)
    return read_assessment_case(case, data=data, case_dirs=(CASES_DIR,), allowed_keys=HOST_KEYS)

def load_case(case, device=None, envs=None, debug=False, *, data=None):
    cfg = read_case(case, data=data)
    loaders = {**physical.LOADERS, **LOADERS}
    systems = {**physical.SYSTEMS, **SYSTEMS}
    cfg = dict(cfg)
    cfg["assets"] = dict(cfg.get("assets") or {})
    for name, entry in (cfg.get("instruments") or {}).items():
        cfg["assets"][name] = {"type": "instrument", **entry}
    return physical.load_case(cfg, device, envs, debug, loaders=loaders, systems=systems, allowed_keys=HOST_KEYS)
