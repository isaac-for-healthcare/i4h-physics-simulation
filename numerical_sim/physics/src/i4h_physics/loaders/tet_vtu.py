# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-order VTU tetrahedra, packed through the ordinary tet-body path.

No remeshing or cell removal. Coordinates use the vessel loader's scale/rotate/translate
convention; negative winding is normalized before deriving outward boundary faces.
"""

from pathlib import Path

import numpy as np
from i4h_physics.loaders.grid import unique_edges
from i4h_physics.loaders.tet_bundle import TET_FACES, PartMesh, pack_tet_body, parse_materials, rest_volumes
from i4h_physics.mesh_utils import rotation_matrix_degrees, select_pins


def read_tet_vtu(path, scale=1.0, translation=(0, 0, 0), rotation_euler_degrees=(0, 0, 0), density=1.0, pins=()):
    import meshio  # optional vtu extra; ordinary cases do not import meshio

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"tet_vtu: expected file {path}")
    mesh = meshio.read(path)
    if any(c.type.startswith("tetra") and c.type != "tetra" for c in mesh.cells):
        raise ValueError("tet_vtu: higher-order tetrahedra are unsupported")
    if any(c.type not in ("tetra", "triangle", "line", "vertex") for c in mesh.cells):
        raise ValueError("tet_vtu: unsupported non-tetrahedral volume cells")
    blocks = [c.data for c in mesh.cells if c.type == "tetra"]
    if not blocks:
        raise ValueError("tet_vtu: no first-order tetrahedra")
    q = np.asarray(mesh.points, np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("tet_vtu: coordinates must be finite xyz")
    q = np.asarray((q * np.asarray(scale)) @ rotation_matrix_degrees(rotation_euler_degrees).T + translation, np.float32)
    if not np.isfinite(q).all():
        raise ValueError("tet_vtu: transformed coordinates must be finite")
    raw = np.concatenate(blocks)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.array_equal(raw, np.floor(raw)):
        raise ValueError("tet_vtu: tetrahedral connectivity must have four integer corners")
    if raw.min() < 0 or raw.max() >= len(q):
        raise ValueError("tet_vtu: connectivity outside particle range")
    t = raw.astype(np.int32)
    if np.any(np.diff(np.sort(t, axis=1), axis=1) == 0):
        raise ValueError("tet_vtu: repeated tetrahedral corner")
    if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
        raise ValueError("tet_vtu: duplicate tetrahedra")
    p = q[t].astype(np.float64)
    signed = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    negative = signed < 0
    t[negative, :2] = t[negative, :2][:, ::-1]
    volumes = rest_volumes(q, t, str(path))
    faces = t[:, TET_FACES].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("tet_vtu: non-manifold tetrahedral faces")
    tris = faces[first[counts == 1]]
    boundary_edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, edge_counts = np.unique(boundary_edges, axis=0, return_counts=True)
    if not len(tris) or np.any(edge_counts != 2):
        raise ValueError("tet_vtu: boundary must be a closed manifold")
    # Edge manifoldness alone misses two closed sheets touching at a vertex.
    # The link of every boundary vertex must be one connected cycle.
    links = {}
    for a, b, c in tris:
        for vertex, x, y in ((a, b, c), (b, c, a), (c, a, b)):
            links.setdefault(int(vertex), []).append((int(x), int(y)))
    for pairs in links.values():
        neighbors = {}
        for x, y in pairs:
            neighbors.setdefault(x, []).append(y)
            neighbors.setdefault(y, []).append(x)
        seen = set()
        todo = [next(iter(neighbors))]
        while todo:
            vertex = todo.pop()
            if vertex not in seen:
                seen.add(vertex)
                todo.extend(neighbors[vertex])
        if len(seen) != len(neighbors) or any(len(n) != 2 for n in neighbors.values()):
            raise ValueError("tet_vtu: non-manifold boundary vertex")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("tet_vtu: density must be finite and positive")
    mass = np.zeros(len(q), np.float64)
    for c in range(4):
        np.add.at(mass, t[:, c], density * volumes.astype(np.float64) / 4)
    if np.any(mass <= 0):
        raise ValueError("tet_vtu: unsupported vertices with no incident volume")
    inv_mass = (1 / mass).astype(np.float32)
    if not np.isfinite(inv_mass).all():
        raise ValueError("tet_vtu: inverse masses overflow float32")
    if any(p.get("type") not in ("box", "sphere") for p in pins):
        raise ValueError("tet_vtu: pins support box and sphere selectors")
    inv_mass[select_pins(q, tris, pins, str(path))] = 0
    return PartMesh(q, inv_mass, t, unique_edges(t), tris)


def add_tet_vtu(scene, name, path, scale=1.0, translation=(0, 0, 0), rotation_euler_degrees=(0, 0, 0),
                density=1.0, radius=0.0, pins=(), parts=None):
    parts = dict(parts) if parts else {name: {"spring": 1.0, "volume": 1.0}}
    if len(parts) != 1:
        raise ValueError("tet_vtu: exactly one material part required")
    mesh = read_tet_vtu(path, scale, translation, rotation_euler_degrees, density, pins)
    materials, visuals = parse_materials(parts)
    return pack_tet_body(scene, name, [{next(iter(parts)): mesh}], materials, radius, visuals)
