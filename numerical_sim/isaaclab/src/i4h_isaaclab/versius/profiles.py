# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable block, original chole and reference life-size anatomy profiles."""

from dataclasses import dataclass

import numpy as np

TCP_OFFSET = (0.0, 0.0, 0.51745)
PROXY_RADIUS = 0.006


@dataclass(frozen=True)
class Profile:
    name: str
    case: dict
    body: str
    pose: tuple
    patch_ids: tuple
    target: tuple
    geometry_scale: float = 1.0


def profile(name):
    from i4h_host.composition import load_scene, read_case

    if name not in ("block", "chole", "chole-1x"):
        raise ValueError(name)
    case = read_case("versius_" + name.replace("-", "_"))
    geometry_scale = 1 / 7 if name == "chole-1x" else 1.0
    body_name = "block" if name == "block" else "tissue"
    scene = load_scene(case, device="cpu", envs=1)
    body = scene.body(body_name)
    q = body.rest.q0[0]
    # Block: turn the hanging fixture upside down. Chole: authored Y-up -> Z-up.
    rotation = (
        np.diag([1.0, -1.0, -1.0])
        if name == "block"
        else np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    )
    quat = (1.0, 0.0, 0.0, 0.0) if name == "block" else (2**-0.5, 0.0, 0.0, 2**-0.5)
    surface = np.unique(body.tet.tris[0].reshape(-1))
    surface = surface[(surface >= 0) & (surface < body.count)]
    if name != "block":
        part = body.parts["liver"]
        surface = surface[
            (surface >= part.particle_start)
            & (surface < part.particle_start + part.particle_count)
        ]
    free = surface[body.rest.inv_mass[0, surface] > 0]
    if body.pin is not None:
        free = free[~np.isin(free, body.pin.index.reshape(-1))]
    world = q @ rotation.T
    # Highest free surface point has an unobstructed vertical particle approach.
    cap = free[
        world[free, 2]
        >= world[free, 2].max() - (0.006 if name == "block" else 0.01 * geometry_scale)
    ]
    center = world[cap].mean(0)
    chosen = cap[np.argmin(np.linalg.norm(world[cap] - center, axis=1))]
    near = free[np.argsort(np.linalg.norm(q[free] - q[chosen], axis=1))[:4]]
    local_target = q[near].mean(0)
    shift = np.array([0.5, 0.0, 0.2]) - rotation @ local_target
    return Profile(
        name,
        case,
        body_name,
        (tuple(shift), quat),
        tuple(map(int, near)),
        tuple(local_target),
        geometry_scale,
    )
