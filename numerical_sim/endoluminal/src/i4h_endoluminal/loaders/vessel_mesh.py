# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vessel mesh (aorta, airways) from one USD mesh prim: a static mesh for containment and rendering, or with a
``deformable`` block a shell body whose wall the ``shell`` system simulates.

Ported from omniendo xcath/config.py:1410-1421 (``rotation_matrix_degrees``, ``transform_mesh``);
the prim is read with ``pxr`` and fan-triangulated like the instrument pieces instead of through
``newton.usd.get_mesh`` (same result, no vertex splitting). Unlike the cavity loader there is no
closed-manifold check: vessels are open at their rims. Smooth normals reuse the cavity's welded
accumulation. The meshes are authored inward-facing (like the cavity), so the containment's
``sign`` convention and the renderer's ``cull: back`` (omniendo ``hide_external_faces``) apply.

``centreline: {path, root_position, ...}`` adds the vessel's anatomical centreline tree as a body (see
``loaders/centreline.py``); the mesh itself is a static render mesh, or with ``skinning: true`` a ``SkinMesh``
on the tree body that the renderers deform with the tree.

``deformable: {density, particle_radius, pins}`` builds what Newton's ``add_cloth_mesh`` built for omniendo
(xcath/xcath.py:899-932): particle mass = density × incident triangle area / 3 (newton builder.py:8732-8737),
one distance constraint per unique triangle edge, one dihedral constraint per interior edge between its two
triangles with the rest angle of builder.py:8389-8414, and pinned vertices with zero mass (config.py:1850-1880:
``boundary`` = vertices of edges owned by one triangle, ``sphere``, ``box``; a selector matching nothing raises).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from i4h_endoluminal.loaders.centreline import add_centreline_from_json, bind_skin
from i4h_physics.loaders.static_mesh import smooth_normals, weld_map
from i4h_physics.mesh_utils import rotation_matrix_degrees, select_pins, shell_topology, triangulate
from i4h_physics.scene import ACTIVE_INT, Part, ParticleRest, Scene, ShellMesh, StaticMesh
from i4h_physics.usd import usd


def read_usd_mesh(path, prim: str) -> tuple[np.ndarray, np.ndarray]:
    """``(points (n, 3) float32, triangles (f, 3) int32)`` of one mesh prim, in the prim's local space."""
    Usd, UsdGeom = usd()

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"vessel mesh not found: {path}")
    stage = Usd.Stage.Open(str(path))
    p = stage.GetPrimAtPath(str(prim))
    if not p or not p.IsA(UsdGeom.Mesh):
        meshes = [str(x.GetPath()) for x in stage.Traverse() if x.IsA(UsdGeom.Mesh)]
        raise ValueError(f"vessel mesh {path}: prim {prim!r} is not a mesh; meshes: {meshes}")
    mesh = UsdGeom.Mesh(p)
    points = np.asarray(mesh.GetPointsAttr().Get(), np.float32).reshape(-1, 3)
    indices = triangulate(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32),
                          np.asarray(mesh.GetFaceVertexCountsAttr().Get(), np.int32))
    return points, indices






def build_shell(name: str, points: np.ndarray, tris: np.ndarray, envs: int, density=1.0, particle_radius=0.0,
                pins=()) -> tuple[ParticleRest, ShellMesh]:
    """Rest state and constraint topology of a shell body from a transformed triangle mesh."""
    n = len(points)
    edges, edge_rest, dihedrals, dihedral_rest, dihedral_edge_len, _ = shell_topology(points, tris, name)
    pts = points.astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]]), axis=1)
    mass = np.zeros(n)
    for c in range(3):  # newton builder.py:8732-8737: a third of each incident triangle's area, times the density
        np.add.at(mass, tris[:, c], float(density) * area / 3.0)
    inv_mass = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-30), 0.0)
    inv_mass[select_pins(points, tris, pins, name)] = 0.0  # hard pins: xcath/xcath.py:928-932 zeroes their mass
    rest = ParticleRest(
        q0=np.ascontiguousarray(np.tile(points.astype(np.float32)[None], (envs, 1, 1))),
        inv_mass=np.ascontiguousarray(np.tile(inv_mass.astype(np.float32)[None], (envs, 1))),
        radius=np.full((envs, n), float(particle_radius), np.float32),
        flags=np.full((envs, n), ACTIVE_INT, np.int32),
    )
    mesh = ShellMesh(
        tris=np.ascontiguousarray(tris[None], dtype=np.int32),
        edges=edges[None], edge_rest=edge_rest[None],
        dihedrals=dihedrals[None], dihedral_rest=dihedral_rest[None], dihedral_edge_len=dihedral_edge_len[None],
        uvs=np.zeros((1, n, 2), np.float32),
    )
    return rest, mesh


def add_vessel_mesh(scene: Scene, name: str, path, prim: str, scale=1.0, translation=(0.0, 0.0, 0.0),
                    rotation_euler_degrees=(0.0, 0.0, 0.0), visual=None, deformable=None, centreline=None):
    """A ``StaticMesh`` (default) or, with ``deformable``, a shell ``Body`` carrying a ``ShellMesh``. With ``centreline``
    the vessel's centreline tree becomes a body named ``name`` (the static mesh keeps the name in its own list)."""
    points, indices = read_usd_mesh(path, prim)
    n = len(points)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= n:
        raise ValueError(f"vessel mesh {name!r}: triangle index outside [0, {n})")
    # xcath/config.py:1414-1417 transform_mesh: scale, then rotate, then translate.
    rotation = rotation_matrix_degrees(rotation_euler_degrees)
    scale = np.asarray(scale, dtype=np.float32)
    points = ((points * scale) @ rotation.T + np.asarray(translation, dtype=np.float32)).astype(np.float32)
    points, indices = np.ascontiguousarray(points), np.ascontiguousarray(indices)
    if centreline is not None:
        if deformable is not None:
            raise ValueError(f"vessel mesh {name!r}: a vessel is either deformable or has a centreline tree")
        c = dict(centreline)
        skinning = bool(c.pop("skinning", False))
        tree = add_centreline_from_json(scene, name, c.pop("path"), scale=scale, rotation=rotation, translation=translation,
                                        root_position=c.pop("root_position"), visual=visual, **c)  # unknown keys fail here
        if skinning:  # the surface follows the tree (dual-quaternion skinning in the renderers)
            tree.skin = bind_skin(name, points, indices, tree.tree, scene.device, visual)
        else:
            normals = smooth_normals(points, indices, weld_map(points))
            mesh = StaticMesh(name, points, indices, None, normals, None)
            from types import SimpleNamespace
            mesh.appearance = SimpleNamespace(visual=dict(visual or {}))
            scene.static_meshes.append(mesh)
        return tree
    if deformable is None:
        normals = smooth_normals(points, indices, weld_map(points))
        mesh = StaticMesh(name, points, indices, None, normals, None)
        from types import SimpleNamespace
        mesh.appearance = SimpleNamespace(visual=dict(visual or {}))
        scene.static_meshes.append(mesh)
        return mesh
    rest, shell = build_shell(name, points, indices, scene.envs, **dict(deformable))
    part = Part(0, 0, n, tri_start=0, tri_count=len(indices))
    from types import SimpleNamespace
    part.appearance = SimpleNamespace(visual=dict(visual or {}))
    body = scene.add_body(name, rest, {name: part})
    body.shell = shell
    return body
