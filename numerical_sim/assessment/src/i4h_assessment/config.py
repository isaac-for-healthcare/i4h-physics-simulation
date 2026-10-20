# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Assessment wrapper policy over explicitly supplied domain case directories."""
from pathlib import Path

import yaml
from i4h_physics import config as physical


def read_assessment_case(case, *, data=None, case_dirs=(), allowed_keys=physical.TOP_LEVEL_KEYS):
    path = None if isinstance(case, dict) else physical.resolve_case(case, case_dirs)
    cfg = case if path is None else yaml.safe_load(path.read_text())
    if isinstance(cfg, dict) and "case" in cfg:
        if cfg.keys() - {"case", "procedure", "ui"}:
            raise ValueError("case reference permits only 'case', 'procedure' and 'ui'; physics overrides are not supported")
        reference = cfg["case"]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("case reference must be a nonempty name or path")
        relative = (path.parent if path is not None else Path.cwd()) / reference
        base = yaml.safe_load(physical.resolve_case(relative if relative.is_file() else reference, case_dirs).read_text())
        if not isinstance(base, dict) or "case" in base or "procedure" in base:
            raise ValueError("case reference must point directly to a physics case without 'case' or 'procedure'")
        if "procedure" in cfg:
            base["procedure"] = cfg["procedure"]
        if "ui" in cfg:
            base["ui"] = cfg["ui"]
        cfg = base
    return physical.read_case(cfg, data=data, allowed_keys=allowed_keys)
