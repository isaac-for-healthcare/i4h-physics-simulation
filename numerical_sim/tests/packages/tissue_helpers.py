# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tissue, fluid and containment history: the arrays C4.2 owns and the two cases that hold them."""
import newton
import numpy as np
import warp as wp
from i4h_physics import author_case
from i4h_physics import config as physical
from interaction_helpers import system

# Arrays kept across steps, by system. ``Surface`` is the surgical surface; its colour lives on ``appearance``.
TISSUE = {
    "Thermal": ("heat", "burn", "coag"),
    "Surface": ("appearance.visual",),
    "Fluid": ("cursor", "overflow"),
    "Bleeding": ("cut", "source", "acc", "serial"),
    "Emitter": ("acc", "serial"),
    "Smoke": ("pos_age", "vel_life", "cursor", "acc", "serial"),
    "Suction": ("removed",),
}
CONTAINMENT = {"GraphContainment": ("cand_idx", "cand_count", "cand_overflow", "ref_samples", "ref_tree", "rebuild", "built")}
# What an episode reset writes into the selected worlds (everything not listed becomes zero).
RESET = {("GraphContainment", "cand_idx"): -1, ("GraphContainment", "rebuild"): 1}


def field(sim, name, key):
    value = system(sim, name)
    for part in key.split("."):
        value = getattr(value, part)
    return value


def history(sim, names):
    wp.synchronize()
    return {(name, key): field(sim, name, key).numpy().tobytes() for name, keys in names.items() for key in keys}


def poison(sim, names, seed=11):
    """Deterministic non-default bytes in every listed array (no physical meaning)."""
    rng = np.random.default_rng(seed)
    for name, keys in names.items():
        for key in keys:
            a = field(sim, name, key)
            host = a.numpy()
            values = rng.integers(2, 6, host.shape).astype(host.dtype) if host.dtype.kind in "iu" else rng.uniform(0.5, 1.5, host.shape).astype(host.dtype)
            wp.copy(a, wp.array(values, dtype=a.dtype, device=a.device))
    wp.synchronize()


def world_rows(sim, name, key, env):
    """Every component of one world's rows (NumPy exposes vector dtypes as a trailing axis)."""
    return field(sim, name, key).numpy().reshape(sim.scene.envs, -1)[env]


def aorta(device, envs=2):
    """The aorta_centreline case (catheter in a centreline tree, graph containment) as an explicit authoring."""
    from i4h_endoluminal.config import LOADERS, SYSTEMS, read_case
    cfg = read_case("aorta_centreline")
    a = author_case(cfg, loaders={**physical.LOADERS, **LOADERS}, systems={**physical.SYSTEMS, **SYSTEMS}, device=device, envs=envs)
    b = newton.ModelBuilder()
    a.declare(b)
    for world in range(envs):
        b.begin_world(gravity=a.scene.gravity_vector)
        a.add_world(b, world)
        b.end_world()
    a.complete(b)
    m = b.finalize(device=device)
    m.particle_grid = None
    return cfg, a, m
