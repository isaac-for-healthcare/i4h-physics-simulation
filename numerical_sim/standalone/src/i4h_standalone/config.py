# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_isaaclab.composition import load_case as physical_case
from i4h_isaaclab.composition import read_case
from i4h_standalone.app.workspace import resolve_workspace


def load_case(case, device=None, envs=None, debug=False, *, data=None):
    cfg = read_case(case, data=data)
    sim = physical_case(cfg, device, envs, debug)
    sim.workspace = resolve_workspace(sim, cfg.get("ui"))
    return sim
