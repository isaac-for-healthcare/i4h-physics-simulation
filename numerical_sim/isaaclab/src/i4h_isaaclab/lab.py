# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Isaac Lab revisions this package targets and a
check for the Lab APIs it relies on.

Older Isaac Lab 3.0 checkouts lack some of them (``NewtonManager._invalidate_graph``, ``newton_builder_world_hook``,
``DirectRLEnvCfg.compute_final_obs``, ``DirectRLEnv._reset_envs_from_buffer``); later revisions are not qualified
until the suites are rerun. No Isaac Lab import here."""

import dataclasses

QUALIFIED = "Isaac Lab main @ 4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1 or patched 3.0 @ e53cd288727a5ca3dae7187d1d415b4d32b67e11 plus docs/newton18-lab30-cloner.patch"


def require(owner, name: str) -> None:
    """Raise a clear error naming the target revision when ``owner`` (a Lab class, configclass or module) lacks
    ``name`` (an attribute, or a configclass field: those keep no class attribute)."""
    fields = {f.name for f in dataclasses.fields(owner)} if dataclasses.is_dataclass(owner) else set()
    if not hasattr(owner, name) and name not in fields:
        raise RuntimeError(f"i4h_isaaclab needs {getattr(owner, '__name__', owner)}.{name}, which this Isaac Lab "
                           f"lacks; it targets {QUALIFIED}")
