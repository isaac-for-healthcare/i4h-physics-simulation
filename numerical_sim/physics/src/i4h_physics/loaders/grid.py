# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline tetrahedral grid asset, generated with Newton's ModelBuilder as a point and tet source."""

from __future__ import annotations

import newton
import numpy as np
import warp as wp
from i4h_physics.loaders.tet_bundle import TET_EDGES, PartMesh, pack_tet_body, parse_materials
from i4h_physics.scene import Body, Scene


def unique_edges(tets: np.ndarray) -> np.ndarray:
    pairs = np.concatenate([tets[:, list(ab)] for ab in TET_EDGES]) if len(tets) else np.zeros((0, 2), np.int32)
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0).astype(np.int32)


def grid_variant(dim, cell, origin, density, radius, fix_top) -> PartMesh:
    dx, dy, dz = (int(v) for v in dim)
    cx, cy, cz = (float(cell),) * 3 if np.isscalar(cell) else (float(v) for v in cell)
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(*[float(v) for v in origin]),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dx, dim_y=dy, dim_z=dz,
        cell_x=cx, cell_y=cy, cell_z=cz,
        density=float(density),
        k_mu=0.0, k_lambda=0.0, k_damp=0.0,
        fix_top=bool(fix_top),
        particle_radius=float(radius),
    )
    q0 = np.array(builder.particle_q, dtype=np.float32).reshape(-1, 3)
    mass = np.array(builder.particle_mass, dtype=np.float32)
    inv_mass = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-30), 0.0).astype(np.float32)
    tets = np.array(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
    tris = np.array(builder.tri_indices, dtype=np.int32).reshape(-1, 3)  # the grid's open faces
    return PartMesh(q0, inv_mass, tets, unique_edges(tets), tris)


def add_tet_grid(scene: Scene, name: str, dim, cell, origin=(0.0, 0.0, 0.0), density=1000.0, radius=0.005,
                 fix_top=False, parts=None) -> Body:
    """``dim`` is one ``[nx, ny, nz]`` or a list of them (one mesh variant per entry, cycled over envs)."""
    dims = [dim] if np.isscalar(dim[0]) else list(dim)
    parts = dict(parts) if parts else {name: {"spring": 1.0, "volume": 1.0}}
    if len(parts) != 1:
        raise ValueError(f"tet_grid {name!r}: exactly one part expected, got {list(parts)}")
    part_name = next(iter(parts))
    variants = [{part_name: grid_variant(d, cell, origin, density, radius, fix_top)} for d in dims]
    materials, visuals = parse_materials(parts)
    return pack_tet_body(scene, name, variants, materials, float(radius), visuals)
