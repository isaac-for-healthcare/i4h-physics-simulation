# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Isaac Lab revision this package targets (runtime suite passing; physics qualification pending), and a
check for the Lab APIs it relies on.

Older Isaac Lab 3.0 checkouts lack some of them (``NewtonManager._invalidate_graph``, ``newton_builder_world_hook``,
``DirectRLEnvCfg.compute_final_obs``, ``DirectRLEnv._reset_envs_from_buffer``); later revisions are not qualified
until the suites are rerun. No Isaac Lab import here."""

import dataclasses

QUALIFIED = "isaac-sim/IsaacLab release/3.0.0 @ 6864dd126fd8efd04aaebe06bc9d00c1b9b989dc"


def require(owner, name: str) -> None:
    """Raise a clear error naming the target revision when ``owner`` (a Lab class, configclass or module) lacks
    ``name`` (an attribute, or a configclass field: those keep no class attribute)."""
    fields = {f.name for f in dataclasses.fields(owner)} if dataclasses.is_dataclass(owner) else set()
    if not hasattr(owner, name) and name not in fields:
        raise RuntimeError(f"i4h_isaaclab needs {getattr(owner, '__name__', owner)}.{name}, which this Isaac Lab "
                           f"lacks; it targets {QUALIFIED}")
