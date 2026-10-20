# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A body's particles (tissue, or a fluid pool) against a static mesh (the abdominal cavity) through a Warp mesh query.

Numerics after omnisurg/tet/kernels/static_contact.py:169-263 (shell = radius + margin,
correction = min((shell - distance) * stiffness, max_correction) along the surface normal,
in place on the thread's own particle). The BVH candidate buffers, overflow snap-back and
the separate index system are replaced by ``wp.mesh_query_point_sign_normal`` on one
``wp.Mesh`` shared by every env (the mesh is static and identical per env). The cavity is
authored inward-facing, so its face normals point at the tissue: a particle on the front
side at signed distance d gets pushed along the normal by shell - d; one that crossed the
wall (d < 0) is pulled back in. ``search`` bounds the query: particles move at most
``max_velocity * dt`` per substep, so a crosser is still within reach at the next event.
``parts`` limits the check to the parts that can reach the wall (the chole liver is authored
through it, by up to 13 cm, so it is excluded).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, due


@dataclass(eq=False)
class StaticContact(System):
    body: str  # tissue body
    mesh: str  # static mesh name (scene.static_meshes)
    parts: tuple = ()  # structural: parts checked (empty = all)
    margin: float = 0.01  # live, per env
    stiffness: float = 1.0  # live, per env
    max_correction: float = 0.0  # live, per env; 0 = unlimited
    iterations: int = 1  # structural: contact events per substep
    search: float = 0.02  # structural: query radius beyond the shell

    structural = System.structural + ("parts", "iterations", "search")

    def __post_init__(self):
        scene = self.scene
        self.body = scene.body(self.body)
        dev = scene.device
        b = self.body
        self.parts = tuple(self.parts)
        unknown = set(self.parts) - set(b.parts)
        if unknown:
            raise ValueError(f"static_contact.parts: unknown parts {sorted(unknown)}")
        enabled = np.zeros(b.count, np.int32)
        for name, part in b.parts.items():
            if not self.parts or name in self.parts:
                enabled[part.particle_start : part.particle_start + part.particle_count] = 1
        selected = np.flatnonzero(enabled)
        self.lo, self.hi = (int(selected.min()), int(selected.max()) + 1) if selected.size else (0, 0)  # launch this range
        self.particle_enabled = wp.array(enabled, dtype=wp.int32, device=dev)  # body-local; gaps inside [lo, hi)
        self.freeze("parts")
        self.wp_mesh = scene.static_wp_mesh(self.mesh)  # shared with every other system on the same static mesh
        self._live("margin")
        self._live("stiffness")
        self._live("max_correction")


    def iteration(self, scene, dt, k, iterations):
        if due(k, self.iterations, iterations)[0] < 0 or self.hi <= self.lo:
            return
        b = self.body  # a tissue body or a fluid pool: scene.field resolves the block
        wp.launch(
            project_static,
            dim=(scene.envs, self.hi - self.lo),
            inputs=[scene.field("inv_mass", b), scene.field("flags", b), scene.field("radius", b), self.particle_enabled,
                    self.wp_mesh.id, self.margin, self.stiffness, self.max_correction, float(self.search), self.lo],
            outputs=[scene.field("q", b)],
            device=scene.device,
        )


@wp.kernel
def project_static(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    radius: wp.array2d(dtype=wp.float32),
    particle_enabled: wp.array(dtype=wp.int32),
    mesh: wp.uint64,
    margin: wp.array(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    max_correction: wp.array(dtype=wp.float32),
    search: wp.float32,
    lo: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
):
    env, offset = wp.tid()
    local = lo + offset
    if particle_enabled[local] == 0:
        return
    i = local
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        return
    p = q[env, i]
    shell = radius[env, i] + margin[env]
    query = wp.mesh_query_point_sign_normal(mesh, p, shell + search)
    if not query.result:
        return
    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    normal = wp.mesh_eval_face_normal(mesh, query.face)
    signed_distance = wp.length(p - closest) * query.sign  # positive on the normal's side (inside the cavity)
    penetration = shell - signed_distance
    if penetration <= 0.0:
        return
    correction = penetration * stiffness[env]
    if max_correction[env] > 0.0:
        correction = wp.min(correction, max_correction[env])
    q[env, i] = p + normal * correction
