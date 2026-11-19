# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured material patch task. Observations and rewards do not advance history."""

import torch

OBSERVATION_LAYOUT = {
    "patch_to_tcp": [0, 3],
    "goal_to_patch": [3, 6],
    "goal_to_tcp": [6, 9],
    "tracking_error": [9, 12],
    "tcp_velocity": [12, 15],
    "patch_velocity": [15, 18],
    "previous_action": [18, 21],
    "indentation": [21, 22],
    "hold_fraction": [22, 23],
    "peak_depth": [23, 24],
}


def measurements(env):
    env.i4h.sync_views()
    points = env.world_points()
    patch = points[:, env.patch_ids].mean(1)
    tcp = env.tcp_position()
    depth = env.reference_patch[:, 2] - patch[:, 2]
    lateral = (tcp[:, :2] - patch[:, :2]).norm(dim=-1)
    distance = (tcp - patch).norm(dim=-1)
    flags = failure_conditions(env, points, tcp, depth)
    failure = flags["nonfinite"] | flags["indentation_limit"] | flags["tip_depth_limit"]
    valid = (
        ((depth - env.cfg.target_depth).abs() < env.cfg.depth_tolerance)
        & (lateral < 0.006)
        & (distance < 0.012)
        & ~failure
    )
    return patch, tcp, depth, lateral, distance, failure, valid


def failure_conditions(env, points, tcp, depth):
    """Named failure masks shared by task semantics and diagnostic UI; no history updates."""
    finite = (
        torch.isfinite(points).all(dim=(1, 2))
        & torch.isfinite(env.world_velocities()).all(dim=(1, 2))
        & torch.isfinite(tcp).all(1)
        & torch.isfinite(env.robot.data.joint_pos.torch).all(1)
        & torch.isfinite(env.robot.data.joint_vel.torch).all(1)
    )
    return {
        "nonfinite": ~finite,
        "indentation_limit": depth > env.cfg.max_depth,
        "tip_depth_limit": tcp[:, 2]
        < env.reference_patch[:, 2] - env.cfg.max_tool_depth,
    }


def advance(env):
    """Called by the first termination term, once for each completed control step."""
    _, _, depth, _, _, failure, valid = measurements(env)
    fresh = env.progress_step != env.common_step_counter
    env.peak_depth[:] = torch.where(
        fresh, torch.maximum(env.peak_depth, depth), env.peak_depth
    )
    held = torch.where(valid, env.hold_time + env.step_dt, 0.0)
    env.hold_time[:] = torch.where(fresh, held, env.hold_time)
    env.failure[:] = failure
    env.success[:] = (env.hold_time >= 0.5 - 1e-6) & ~failure
    env.progress_step[fresh] = env.common_step_counter
    return env.success


def success(env):
    result = advance(env)
    return torch.zeros_like(result) if env.cfg.teleop else result.clone()


def failure(env):
    return torch.zeros_like(env.failure) if env.cfg.teleop else measurements(env)[5]


def timeout(env):
    return (
        (env.episode_length_buf >= env.max_episode_length)
        & ~env.success
        & ~env.failure
        & (not env.cfg.teleop)
    )


def observations(env):
    patch, tcp, depth, *_ = measurements(env)
    goal = env.reference_patch.clone()
    goal[:, 2] -= env.cfg.target_depth
    lever = tcp - env.robot.data.body_com_pos_w.torch[:, env.ee_id]
    velocity = env.robot.data.body_lin_vel_w.torch[:, env.ee_id] + torch.linalg.cross(
        env.robot.data.body_ang_vel_w.torch[:, env.ee_id], lever, dim=-1
    )
    patch_velocity = env.world_velocities()[:, env.patch_ids].mean(1)
    return torch.cat(
        (
            (patch - tcp) / 0.05,
            (goal - patch) / 0.05,
            (goal - tcp) / 0.05,
            (env.target_tcp - tcp) / 0.05,
            velocity / 0.1,
            patch_velocity / 0.1,
            env.actions,
            depth[:, None] / 0.01,
            env.hold_time[:, None] / 0.5,
            env.peak_depth[:, None] / 0.01,
        ),
        dim=-1,
    )


def reward(env):
    _, _, depth, lateral, distance, failed, valid = measurements(env)
    value = (
        2 * torch.exp(-distance / 0.03)
        + 6 * torch.exp(-(((depth - 0.003) / 0.0015) ** 2) - (lateral / 0.002) ** 2)
        + 4 * valid
        - 0.02 * env.actions.square().sum(1)
        - 0.03 * (env.actions - env.previous_actions).square().sum(1)
    )
    return (
        torch.nan_to_num(value) * (~failed)
        + (50 * env.success.float() - 5 * failed.float()) / env.step_dt
    )
