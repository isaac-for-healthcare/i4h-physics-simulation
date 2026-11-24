# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only task measurements and failure diagnostics."""

from . import terms


def snapshot(env):
    patch, tcp, depth, lateral, distance, failed, valid = terms.measurements(env)
    flags = terms.failure_conditions(env, env.world_points(), tcp, depth)
    products = env.i4h.state()
    force = products["tool_force.force"].reshape(env.num_envs, 3)
    contacts = products["tool_force.count"].reshape(env.num_envs, -1).sum(1)
    import torch

    # One batched diagnostic readback; the viewer throttles these snapshots to 6 Hz.
    values = torch.cat((
        torch.stack((depth[0], lateral[0], distance[0], contacts[0], force[0].norm(),
                     env.hold_time[0], env.success[0], valid[0], failed.any(),
                     env.reference_patch[0, 2] - tcp[0, 2],
                     (env.target_tcp[0] - tcp[0]).norm(), env.episode_length_buf[0])),
        force[0], tcp[0], patch[0],
        *(mask.float() for mask in flags.values()),
    )).detach().cpu().tolist()
    reasons = []
    names = ("nonfinite tissue/robot state", "tissue indentation limit exceeded",
             "tool tip below allowed depth")
    for index, name in enumerate(names):
        for world in range(env.num_envs):
            if values[21 + index * env.num_envs + world]:
                reasons.append(f"env {world}: {name}")
    return {
        "profile": env.physical.name,
        "geometry_scale": getattr(env.physical, "geometry_scale", 1.0),
        "envs": env.num_envs,
        "depth": values[0],
        "depth_low": env.cfg.target_depth - env.cfg.depth_tolerance,
        "depth_high": env.cfg.target_depth + env.cfg.depth_tolerance,
        "depth_limit": env.cfg.max_depth,
        "tip_depth_limit": env.cfg.max_tool_depth,
        "tip_depth": values[9],
        "lateral": values[1],
        "distance": values[2],
        "contacts": int(values[3]),
        "force": values[12:15],
        "force_norm": values[4],
        "hold": values[5],
        "success": bool(values[6]),
        "valid": bool(values[7]),
        "failed": bool(values[8]),
        "reasons": reasons,
        "tcp": values[15:18],
        "tracking_error": values[10],
        "steps": int(values[11]),
        "graph": bool(env.graph_seen),
        "patch": values[18:21],
    }


def failure_status(env):
    """Synchronize one failure bit before physics; read detailed diagnostics only on failure."""
    if bool(terms.measurements(env)[5].any()):
        return snapshot(env)
    return {"failed": False}
