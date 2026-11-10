# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate and validate Lab coupling configuration; numerical coupling lives in i4h_physics."""

from i4h_physics.coupling import COUPLING_MODES


def check_rigid_solver_cfg(rigid_cfg, coupling_mode: str) -> None:
    """MuJoCo-Warp with its own contacts only (it leaves particles alone); a known coupling mode."""
    if coupling_mode not in COUPLING_MODES:
        raise ValueError(f"I4hSolverCfg.coupling_mode must be one of {COUPLING_MODES}, got {coupling_mode!r}")
    kind = str(getattr(rigid_cfg, "solver_type", ""))
    if kind != "mujoco_warp":
        raise ValueError(f"I4hSolverCfg.rigid_solver_cfg: the coupled step supports MuJoCo-Warp (MJWarpSolverCfg), got "
                         f"{kind or type(rigid_cfg).__name__!r}. Featherstone integrates every particle and applies "
                         "particle contact forces to bodies, so it would integrate i4h particles a second time.")
    if not getattr(rigid_cfg, "use_mujoco_contacts", True):
        raise ValueError("I4hSolverCfg.rigid_solver_cfg: use_mujoco_contacts=False needs Newton's collision pipeline, which "
                         "the coupled step does not run; keep MuJoCo's contacts")

    if getattr(rigid_cfg, "use_mujoco_cpu", False):
        raise ValueError("i4h coupling requires MuJoCo-Warp on the GPU; the global CPU reset is not world-mask safe")
