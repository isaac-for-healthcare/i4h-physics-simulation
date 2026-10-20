# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rod nodes kept inside a static vessel mesh through a Warp mesh query.

Ported from omniendo/xpbd_rod/batched_xcath.py:163-219 (``_project_static_anatomy_local_frame``),
one-sided branch: the closest surface point within ``search`` is found, ``phi = sign * query.sign *
distance`` is the node's signed distance (``sign`` -1 with an inward-facing mesh makes the lumen
negative, as omniendo's ``sign_scale``), and a node with ``phi > -radius`` is pushed back along the
surface direction by ``phi + radius``. Per env the maximum depth and the number of contacts of the
last substep are recorded (Isaac Lab observations). No friction, no velocity term (omniendo had
none). Dropped: the two-sided branch and its side classification, the dynamic-mesh path, and the
per-env anatomy pose (omniendo used it for the env grid origins; positions here are in a per-env
local frame and one ``wp.Mesh`` serves every env as is).

A cousin of ``static_contact.project_static`` (same query, same push), separate because the target is
the node's own radius, contacts are reported, and there are no parts or margin. The static mesh comes
from a ``vessel_mesh`` asset. List this system after ``catheter_control`` (the track projection comes first).
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.checks import check_float
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, due


@dataclass(eq=False)
class RodContainment(System):
    state_products = ("contact_depth", "contact_count")  # per-env products a host observes (System.state_products)
    body: str  # rod body
    mesh: str  # static mesh name (scene.static_meshes)
    search: float = 0.1  # structural: query radius (omniendo containment.max_distance, default 0.1)
    iterations: int = 1  # structural: projections per substep
    sign: float = -1.0  # live, per env: -1 keeps the rod on the side the face normals point away from (inward-facing mesh)

    structural = System.structural + ("search", "iterations")

    def __post_init__(self):
        scene = self.scene
        self.body = b = scene.body(self.body)
        if b.rod is None:
            raise ValueError(f"rod_containment: body {b.name!r} has no rod record")
        E, dev = scene.envs, scene.device
        self.search = check_float("rod_containment.search", self.search)
        self.wp_mesh = scene.static_wp_mesh(self.mesh)  # shared with every other system on the same static mesh
        self._live("sign")
        self.contact_depth = wp.zeros(E, dtype=wp.float32, device=dev)  # deepest push of the last substep
        self.contact_count = wp.zeros(E, dtype=wp.int32, device=dev)  # nodes pushed in the last substep

    def readouts(self):
        return {"contact_depth": self.contact_depth, "contact_count": self.contact_count}


    def substep_start(self, scene, dt, s, substeps):
        self.contact_depth.zero_()
        self.contact_count.zero_()

    def iteration(self, scene, dt, k, iterations):
        if due(k, self.iterations, iterations)[0] < 0:
            return
        b = self.body
        wp.launch(project_static, dim=(scene.envs, b.count),
                  inputs=[scene.field("inv_mass", b), scene.field("flags", b), scene.field("radius", b), self.wp_mesh.id,
                          self.sign, self.search],
                  outputs=[scene.field("q", b), self.contact_depth, self.contact_count], device=scene.device)

    def reset(self, scene, mask):
        self.clear(mask, (self.contact_depth, wp.float32(0.0)))
        self.clear(mask, (self.contact_count, wp.int32(0)))


# Ported from batched_xcath.py:163-219 (_project_static_anatomy_local_frame), one-sided branch. Edits:
# (env, node) prologue; predicted positions are the body's (E, P) view of scene.q; the Warp 1.16 struct
# query; target_phi is -radius[i]; collision_enabled dropped (disable the system); inactive nodes skipped;
# the anatomy pose transform, current_positions, two_sided and collision_sides (two-sided path) dropped.
@wp.kernel
def project_static(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    radius: wp.array2d(dtype=wp.float32),
    mesh_id: wp.uint64,
    sign_scale: wp.array(dtype=wp.float32),
    max_distance: wp.float32,
    q: wp.array2d(dtype=wp.vec3),
    contact_depth: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    particle = local
    if (flags[env, particle] & ACTIVE) == 0 or inv_mass[env, particle] <= 0.0:
        return
    local_position = q[env, particle]
    query = wp.mesh_query_point_sign_normal(mesh_id, local_position, max_distance)
    if not query.result:
        return
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    delta = local_position - closest
    distance = wp.length(delta)
    side = query.sign * sign_scale[env]
    phi = side * distance
    target = -radius[env, particle]
    normal = wp.mesh_eval_face_normal(mesh_id, query.face) * sign_scale[env]  # the limit of delta * side / distance on the wall
    if distance > 1.0e-8:
        normal = delta * (side / distance)
    if phi <= target:
        return
    depth = phi - target
    q[env, particle] = local_position - normal * depth
    wp.atomic_max(contact_depth, env, depth)
    wp.atomic_add(contact_count, env, 1)
