# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected material-region markers placed only from completed render snapshots."""
import numpy as np
import warp as wp
from i4h_standalone.render.instruments import sphere_mesh_indices, uv_sphere


class RegionHighlight:
    def __init__(self, renderer, body):
        from i4h_standalone.render.slang import FlatDraw
        self.body = body
        self.selected = -1
        points, faces = uv_sphere()
        self.template = wp.array(points, dtype=wp.vec3, device=renderer.scene.device)
        edges = body.tree.edges[0] if body.tree is not None else np.column_stack((np.arange(body.count - 1), np.arange(1, body.count)))
        self.edges = wp.array(edges, dtype=wp.vec2i, device=renderer.scene.device)
        self.membership = wp.array(body.regions.edge, dtype=int, device=renderer.scene.device)
        self.radius = float(np.min(body.tree.radii)) * 0.15 if body.tree is not None else float(body.rest.radius[0, 0]) * 1.2
        self.draw = FlatDraw(renderer, f'region-{body.name}', len(edges) * len(points),
                             sphere_mesh_indices(len(edges), faces, len(points)), (1.0, 0.75, 0.05))

    def publish(self, renderer, snap, env, stream):
        wp.launch(place_region, dim=self.draw.pos.array.size,
                  inputs=[renderer.scene.view(snap.q, self.body), env, self.edges, self.membership,
                          self.selected, self.template, self.template.size, self.radius],
                  outputs=[self.draw.pos.array, self.draw.normals.array], device=renderer.scene.device, stream=stream)


@wp.kernel
def place_region(q: wp.array2d(dtype=wp.vec3), env: int, edges: wp.array(dtype=wp.vec2i), membership: wp.array(dtype=int),
                 selected: int, template: wp.array(dtype=wp.vec3), vertices: int, radius: float,
                 pos: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    e = i // vertices
    ends = edges[e]
    centre = 0.5 * (q[env, ends[0]] + q[env, ends[1]])
    r = float(0.0)
    if membership[e] == selected:
        r = radius
    n = template[i % vertices]
    pos[i] = centre + r * n
    normals[i] = n
